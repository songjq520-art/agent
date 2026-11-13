# tests/test_taskstore_concurrency.py（GPT 第一轮审查 P2：TaskStore 并发安全）
# 作用：验证任务存储在并发下的行为——
#   1. 同一进程里多个 TaskStore 实例共用一把锁，CAS 不会两个人同时抢到；
#   2. 多线程同时 CAS 抢同一条任务，更新不丢；
#   3. 两个进程同时写同一个文件，文件不会被写坏（唯一临时文件 +
#      原子改名），写完不留临时文件；
#   4. 跨进程写现在也串行化了（2026-11-13 修：跨进程锁文件 + 替换重试，
#      Windows 上 os.replace 报 WinError 5 的根因）：两个进程各写 20 条，
#      40 条一条都不能丢——读-改-写都在锁里，写完不留临时文件。
# 全部用临时目录，不碰真正的 tasks/tasks.json。

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from tasks.task_store import TaskStore, make_task

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_two_instances_share_one_lock(tmp_path):
    """同一文件路径的两个实例共用同一把锁；不同路径用不同的锁。"""
    path = str(tmp_path / "tasks.json")
    a = TaskStore(file_path=path)
    b = TaskStore(file_path=path)
    assert a._lock is b._lock
    other = TaskStore(file_path=str(tmp_path / "other.json"))
    assert other._lock is not a._lock


def test_cas_across_instances_second_loser_does_not_overwrite(tmp_path):
    """实例 A 把任务从 RUNNING 改成别的之后，实例 B 再拿旧状态
    RUNNING 来 CAS 抢不到，也不会把 A 的改动盖回去。"""
    path = str(tmp_path / "tasks.json")
    a = TaskStore(file_path=path)
    b = TaskStore(file_path=path)
    task = make_task(goal="查一下", model="deepseek-flash")
    task["status"] = "RUNNING"
    a.save(task)
    ok, _ = a.cas(task["task_id"], "RUNNING", lambda t: {**t, "status": "PAUSED"})
    assert ok is True
    ok, current = b.cas(
        task["task_id"], "RUNNING", lambda t: {**t, "status": "COMPLETED"}
    )
    assert ok is False
    assert current["status"] == "PAUSED"
    assert a.load(task["task_id"])["status"] == "PAUSED"


def test_concurrent_cas_from_threads_no_lost_updates(tmp_path):
    """20 个线程同时对同一条任务做 CAS 递增（读-比-改-写带重试），
    20 次递增一次都不能丢——锁是进程内共享的，CAS 才有原子性。"""
    path = str(tmp_path / "tasks.json")
    stores = [TaskStore(file_path=path) for _ in range(4)]
    task = make_task(goal="计数", model="deepseek-flash")
    task["status"] = "RUNNING"
    task["counter"] = 0
    stores[0].save(task)
    task_id = task["task_id"]
    errors = []

    def _increment(store):
        for _ in range(50):
            ok, _ = store.cas(
                task_id,
                "RUNNING",
                lambda t: {**t, "counter": t.get("counter", 0) + 1},
            )
            if ok:
                return
        errors.append("CAS 50 次都没抢到")

    threads = [
        threading.Thread(target=_increment, args=(stores[i % len(stores)],))
        for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    final = stores[0].load(task_id)
    assert final["counter"] == 20
    assert final["status"] == "RUNNING"


def test_concurrent_process_writes_keep_file_valid(tmp_path):
    """两个进程同时写同一个任务文件（各写 20 条）：文件必须保持
    能解析的 JSON 对象，40 条一条都不能丢，不留临时文件。
    第 3 批的真事故就是两个进程同时写把文件写坏了——唯一临时文件 +
    原子改名保证不再写坏；2026-11-13 再修一层：跨进程锁文件把
    “读-改-写”串行化，Windows 上替换被读进程挡住（WinError 5）时
    还有重试兜底——所以现在更新也不丢了，不只文件不坏。
    注意：锁文件（tasks.json.lock）留在盘上是正常的，不算垃圾。"""
    path = tmp_path / "tasks.json"
    code = (
        "import sys\n"
        "from tasks.task_store import TaskStore, make_task\n"
        "store = TaskStore(file_path=sys.argv[1])\n"
        "prefix = sys.argv[2]\n"
        "for i in range(20):\n"
        "    t = make_task(goal=f'{prefix}-{i}', model='deepseek-flash')\n"
        "    t['task_id'] = f'{prefix}-{i}'\n"
        "    store.save(t)\n"
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path), prefix],
            cwd=str(REPO_ROOT),
        )
        for prefix in ("proc-a", "proc-b")
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert len(data) == 40
    assert {f"proc-a-{i}" for i in range(20)} <= set(data)
    assert {f"proc-b-{i}" for i in range(20)} <= set(data)
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []


def test_replace_retry_survives_transient_permission_error(tmp_path, monkeypatch):
    """替换重试：os.replace 第一次报 PermissionError（模拟 Windows 上
    目标文件正被别的进程打开读），第二次成功——save 照样成功，
    文件是完整 JSON，不留临时文件。"""
    import tasks.task_store as task_store_module

    real_replace = os.replace
    calls = {"n": 0}

    def _flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "Permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(task_store_module.os, "replace", _flaky_replace)
    path = str(tmp_path / "tasks.json")
    store = TaskStore(file_path=path)
    task = make_task(goal="重试", model="deepseek-flash")
    store.save(task)
    assert calls["n"] == 2
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data[task["task_id"]]["goal"] == "重试"
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []
