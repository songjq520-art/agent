# tests/test_r1r5_20261124.py（R-1/R-5 修复，2026-11-24）
# 全部离线（向量打桩、判官打桩成不可用，不调真 API、不花钱）。
#
# R-1 纠正只顶一条（P1 重查）：带明确纠正词时 _apply_same_slot_update
#    和 add_memory_text 都把同槽冲突 [:1] 只顶最新一条，correct_memory
#    落盘后也不清其余同槽活值——单值槽里更早的错误值永远留着。
#    BOOS 定“像十一”：一次纠正把同槽真冲突的旧记忆全部顶掉。
#    修法：主目标（被纠正的那条）照旧走原裁决；主目标之外走连带
#    清扫（_sweep_extra_conflicts），门槛=防广州式误伤的护栏：
#    只清 BOOS 对话里亲口说过的（metadata 带 turn_id）、非核心/置顶
#    的、且 _resolve_conflict 裁决为 supersede_old 的；取代走账本
#    SUPERSEDE 事件（标记不删除）。点名旧值的纠正（old != new）
#    仍是外科手术式，只改被点名的那条，不清扫。
# R-5 衰减/自检没有生产调用方（P1 重查）：sweep_decay /
#    run_self_check 函数都在，服务跑起来永远不会执行。修法：
#    web/server.py 加进程内轻量调度 MemoryMaintenanceScheduler
#    （衰减每天、自检每月，到期才跑、上次运行时间落盘、启动到期
#    补跑、异常只记日志不带崩服务），main() 启动时挂上。

import json
import time

import pytest

from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso
from web.server import MemoryMaintenanceScheduler


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词
    路），判官打桩成不可用（模型断了降级走程序，是既定生产模式）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _turn(store, text, turn_id):
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的"}],
        turn_id=turn_id,
    )


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _plant_row(store, text, **meta_over):
    """绕过 add 入口直接插一行（模拟迁移/直存来的旧行），meta 可覆盖。"""
    created_at = _now_iso()
    key = _ledger.ledger_event_key(text, created_at)
    meta = {
        "source": "inferred", "confidence": 0.7, "user_confirmed": False,
        "created_at": created_at, "write_seq": 1, "turn_id": None,
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "episodic", "importance": "normal",
        "event_date": None,
    }
    meta.update(meta_over)
    store._insert_memory_row(key, text, meta, None)
    return key


# ---------- R-1：纠正顶掉全部真冲突 ----------


@pytest.mark.requires_judge
def test_r1_correction_supersedes_all_turn_stated_conflicts(store):
    """同槽两条亲口说过的冲突旧值（武汉、长沙，都是对话里说的），
    一次纠正“错了，我住在郑州”：两条全顶掉，账本各记一笔
    supersede（标记，不是删除），郑州是活值。"""
    _turn(store, "我住在武汉", "t1")
    _turn(store, "我住在长沙", "t2")
    assert "我住在武汉" in _live_texts(store)
    assert "我住在长沙" in _live_texts(store)
    receipt = _turn(store, "错了，我住在郑州", "t3")
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated", receipt
    assert _is_superseded(store, "我住在武汉")
    assert _is_superseded(store, "我住在长沙")
    assert "我住在郑州" in _live_texts(store)
    events, _info = _ledger.read_events(store.ledger_path)
    sup_old_texts = {
        e.get("old_text") for e in events if e.get("type") == "supersede"
    }
    assert "我住在武汉" in sup_old_texts
    assert "我住在长沙" in sup_old_texts


@pytest.mark.requires_judge
def test_r1_correction_keeps_non_conflicting_rows(store):
    """同槽不冲突的（同值异说法“我家在郑州”）和别的槽位的
    （“我叫小明”）都不动：只有真冲突的武汉被顶掉。"""
    _turn(store, "我住在武汉", "t1")
    store.add_memory_text("我家在郑州", source="user")
    store.add_memory_text("我叫小明", source="user")
    receipt = _turn(store, "错了，我住在郑州", "t2")
    assert (receipt.get("correction") or {}).get("status") == "updated", receipt
    assert _is_superseded(store, "我住在武汉")
    assert not _is_superseded(store, "我家在郑州")
    assert "我家在郑州" in _live_texts(store)
    assert not _is_superseded(store, "我叫小明")
    assert "我叫小明" in _live_texts(store)


@pytest.mark.requires_judge
def test_r1_correction_skips_core_row(store):
    """冲突候选是核心/置顶记忆（curated）时不自动顶：哪怕它也是
    对话里说过的（turn_id 在案），也只有普通行武汉被顶掉，
    核心行留着走问 BOOS 的路径。"""
    _plant_row(
        store, "我住在南京",
        source="user", confidence=1.0, user_confirmed=True,
        curated=True, turn_id="t-core", write_seq=1,
    )
    _turn(store, "我住在武汉", "t1")
    receipt = _turn(store, "错了，我住在郑州", "t2")
    assert (receipt.get("correction") or {}).get("status") == "updated", receipt
    assert _is_superseded(store, "我住在武汉")
    assert not _is_superseded(store, "我住在南京")
    assert "我住在南京" in _live_texts(store)


@pytest.mark.requires_judge
def test_r1_correction_skips_migrated_inferred_row(store):
    """广州式护栏：迁移来的推断行（“用户住在广州”，turn_id 为空、
    未确认）哪怕同槽冲突也不连带顶——只有对话里刚说错的武汉被顶。"""
    _plant_row(store, "用户住在广州")
    _turn(store, "我住在武汉", "t1")
    receipt = _turn(store, "错了，我住在长沙", "t2")
    assert (receipt.get("correction") or {}).get("status") == "updated", receipt
    assert _is_superseded(store, "我住在武汉")
    assert not _is_superseded(store, "用户住在广州")
    assert "用户住在广州" in _live_texts(store)


@pytest.mark.requires_judge
def test_r1_direct_store_supersedes_turn_stated_extras(store):
    """直存路（allow_slot_supersede=True）：主目标（最新冲突）照旧
    顶，其余冲突里对话亲口说过的连带清掉、直存/迁移来的不连带。"""
    store.add_memory_text("用户住在广州", source="user")
    _turn(store, "我住在武汉", "t1")
    _turn(store, "我住在西安", "t2")
    result = store.add_memory_text(
        "我住在长沙", source="user", allow_slot_supersede=True
    )
    assert result.get("event") == "ADD"
    assert _is_superseded(store, "我住在西安")
    assert _is_superseded(store, "我住在武汉")
    assert not _is_superseded(store, "用户住在广州")
    assert "用户住在广州" in _live_texts(store)
    assert "我住在长沙" in _live_texts(store)


# ---------- R-5：维护调度 ----------


class _FakeStore:
    def __init__(self, fail_decay=False):
        self.decay_calls = 0
        self.check_calls = 0
        self.fail_decay = fail_decay

    def sweep_decay(self):
        self.decay_calls += 1
        if self.fail_decay:
            raise RuntimeError("衰减失败（测试打桩）")
        return {"dormant": 1, "delete_due": 0}

    def run_self_check(self):
        self.check_calls += 1
        return {"summary": "ok"}


def test_r5_scheduler_runs_when_due_and_persists(tmp_path):
    """到期会触发两项、没到期不触发、上次运行标记落盘持久化
    （新调度器读同一份状态不重复跑，模拟重启）。"""
    state_path = tmp_path / "maintenance_state.json"
    fake = _FakeStore()
    sched = MemoryMaintenanceScheduler(
        fake, state_path,
        decay_interval_seconds=100, self_check_interval_seconds=1000,
        tick_seconds=60,
    )
    ran = sched.run_due(now=10_000)
    assert ran == {"decay": True, "self_check": True}
    assert fake.decay_calls == 1 and fake.check_calls == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["decay_last_run"] == 10_000
    assert state["self_check_last_run"] == 10_000
    ran = sched.run_due(now=10_050)
    assert ran == {"decay": False, "self_check": False}
    assert fake.decay_calls == 1 and fake.check_calls == 1
    ran = sched.run_due(now=10_150)
    assert ran == {"decay": True, "self_check": False}
    assert fake.decay_calls == 2 and fake.check_calls == 1
    sched2 = MemoryMaintenanceScheduler(
        fake, state_path,
        decay_interval_seconds=100, self_check_interval_seconds=1000,
        tick_seconds=60,
    )
    ran = sched2.run_due(now=10_160)
    assert ran == {"decay": False, "self_check": False}
    assert fake.decay_calls == 2 and fake.check_calls == 1


def test_r5_scheduler_failure_isolated(tmp_path):
    """调度异常不影响主流程：一个任务抛错，run_due 不抛、另一个
    照跑，失败的记了本次时间、没到期不连环重试。"""
    fake = _FakeStore(fail_decay=True)
    sched = MemoryMaintenanceScheduler(
        fake, tmp_path / "state.json",
        decay_interval_seconds=100, self_check_interval_seconds=100,
        tick_seconds=60,
    )
    ran = sched.run_due(now=5_000)
    assert ran["self_check"] is True
    assert fake.decay_calls == 1 and fake.check_calls == 1
    ran = sched.run_due(now=5_010)
    assert ran == {"decay": False, "self_check": False}
    assert fake.decay_calls == 1


def test_r5_scheduler_missing_methods_skipped(tmp_path):
    """store 没有对应方法时跳过该任务，不报错。"""
    class _Bare:
        pass

    sched = MemoryMaintenanceScheduler(
        _Bare(), tmp_path / "state.json",
        decay_interval_seconds=100, self_check_interval_seconds=100,
        tick_seconds=60,
    )
    ran = sched.run_due(now=1_000)
    assert ran == {"decay": False, "self_check": False}


def test_r5_scheduler_thread_smoke(tmp_path):
    """线程起得来、到期补跑一次、停得下。"""
    fake = _FakeStore()
    sched = MemoryMaintenanceScheduler(
        fake, tmp_path / "state.json",
        decay_interval_seconds=3600, self_check_interval_seconds=3600,
        tick_seconds=0.05,
    )
    sched.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and fake.check_calls < 1:
            time.sleep(0.05)
        assert fake.decay_calls >= 1 and fake.check_calls >= 1
    finally:
        sched.stop()
        sched._thread.join(timeout=10)
    assert not sched._thread.is_alive()
