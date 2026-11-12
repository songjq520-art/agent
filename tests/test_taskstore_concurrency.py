# tests/test_taskstore_concurrency.py（GPT 第一轮审查 P2：TaskStore 并发安全）
# 作用：验证任务存储在并发下的行为——
#   1. 同一进程里多个 TaskStore 实例共用一把锁，CAS 不会两个人同时抢到；
#   2. 多线程同时 CAS 抢同一条任务，更新不丢；
#   3. 两个进程同时写同一个文件，文件不会被写坏（唯一临时文件 +
#      原子改名），写完不留临时文件；
#   4. 跨进程 CAS 不是原子的（只保证文件不损坏，不保证更新不丢），
#      这一点写在 tasks/task_store.py 的文档里，不假装它能跨进程。
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
    能解析的 JSON 对象，不留临时文件。第 3 批的真事故就是两个进程
    同时写把文件写坏了——唯一临时文件 + 原子改名保证不再写坏。
    注意：跨进程不保证更新不丢（后写的可能盖掉先写的），这里只验
    文件不损坏，不数条数。"""
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
    assert len(data) > 0
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []
