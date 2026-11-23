# tests/test_fx451011_20261124.py（缺点清单 4/5/11 修复，2026-11-24）
# 全部离线（向量打桩、判官打桩，不调真 API、不花钱）。
#
# 第 4 条 自检太稀（BOOS 定「修」）：维护调度里自检每月一次、第 9 项
#    抽查 10 条——改每周一次、抽查 20 条。
# 第 5 条 槽位分错顶不掉（BOOS 定「让他概率最小」）：纠正的主路和
#    同槽兜底全落空后，加一层跨槽安全网：对全部活记忆高精度语义
#    扫描（阈值 0.80，高于主路 0.70），命中必须再过判官确认真冲突
#    才取代；核心/置顶仍不自动顶。
# 第 11 条「吗」碎片消息（BOOS 定「修」）：用户一句话分两条发、后半
#    条是碎片时，save_turn 组装本轮输入把两条拼成整句再处理；
#    会话历史原样不动。
# （第 10 条备份改每天：驱动是分支外的 Muse 定时任务
#  shier-memory-weekly-backup，本轮按规矩不动它，无代码变更、
#  无测试，结论见本轮报告。）

import inspect

import pytest

from memory import ledger as _ledger
from memory import self_check as self_check_module
from memory.mem0_store import Mem0Store
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso
from web.server import MemoryMaintenanceScheduler, start_memory_maintenance


# ---------- 第 4 条：自检加密 + 加量 ----------


def test_item4_scheduler_default_self_check_weekly():
    """调度器默认：衰减每天、自检每周（原每月）。"""
    sig = inspect.signature(MemoryMaintenanceScheduler.__init__)
    assert sig.parameters["decay_interval_seconds"].default == 24 * 3600
    assert sig.parameters["self_check_interval_seconds"].default == 7 * 24 * 3600


def test_item4_start_memory_maintenance_default_weekly(tmp_path, monkeypatch):
    """start_memory_maintenance 不设环境变量时，自检间隔是 7 天
    （环境变量覆盖仍保留：见 _maintenance_interval_value 既有测试）。"""
    monkeypatch.delenv("SHIER_SELF_CHECK_INTERVAL_DAYS", raising=False)
    monkeypatch.delenv("SHIER_DECAY_SWEEP_INTERVAL_HOURS", raising=False)
    monkeypatch.delenv("SHIER_MAINTENANCE_TICK_SECONDS", raising=False)

    class _FakeStore:
        ledger_path = str(tmp_path / "memory" / "memory_events.jsonl")

        def sweep_decay(self):
            return {}

        def run_self_check(self):
            return {}

    sched = start_memory_maintenance(_FakeStore())
    try:
        assert sched._self_check_interval == 7 * 24 * 3600
        assert sched._decay_interval == 24 * 3600
    finally:
        sched.stop()
        if sched._thread is not None:
            sched._thread.join(timeout=10)


def test_item4_sample_size_defaults_20():
    """三个入口的第 9 项抽查默认都是 20 条（原 10 条）。"""
    assert (
        inspect.signature(SQLiteStore.run_self_check)
        .parameters["sample_size"].default
        == 20
    )
    assert (
        inspect.signature(Mem0Store.run_self_check)
        .parameters["sample_size"].default
        == 20
    )
    assert (
        inspect.signature(self_check_module.run_self_check)
        .parameters["sample_size"].default
        == 20
    )


def test_item4_item9_samples_20_by_default():
    """行为级：30 条记忆、不传 sample_size 跑自检，判官恰好被问
    20 次（第 9 项抽查量）；假记忆都不是核心，第 10 项不调判官。"""
    calls = []

    def judge(system_prompt, user_prompt):
        calls.append(user_prompt)
        return {"accurate": True, "evidence": "测试判官"}

    class _Adapter:
        def get_all_items(self):
            return [
                {"id": f"m{i:02d}", "text": f"记忆{i}", "metadata": {}}
                for i in range(30)
            ]

        def supersede_events_since(self, days):
            return []

        def verify_backup(self):
            return {"ok": True, "detail": ""}

        def index_counts(self):
            return {"sqlite": 30, "chroma": 30, "mismatched_ids": []}

        def replay_promotion_counts(self):
            return {"checked": 30, "corrections": []}

        def archive_items(self):
            return []

        def embed(self, text):
            return None

    report = self_check_module.run_self_check(_Adapter(), model_judge=judge)
    assert len(calls) == 20
    assert report["items"][9]["status"] == "pass"


# ---------- 第 5 条：跨槽安全网 ----------

import math  # noqa: E402


def _net_embed(self, text):
    """确定性假向量：米饭/白饭同向（余弦 1.0），面条与米饭 0.68，
    火锅正交，其余正交——专为跨槽网的门槛场景设计。"""
    text = text or ""
    if "米饭" in text or "白饭" in text:
        return [1.0, 0.0, 0.0, 0.0]
    if "面条" in text:
        return [0.68, math.sqrt(1 - 0.68 ** 2), 0.0, 0.0]
    if "火锅" in text:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def _plant_row(store, text, **meta_over):
    """绕过 add 入口直接插一行（模拟别的路存进来的旧行），meta 可覆盖。"""
    created_at = _now_iso()
    key = _ledger.ledger_event_key(text, created_at)
    meta = {
        "source": "user", "confidence": 1.0, "user_confirmed": True,
        "created_at": created_at, "write_seq": 1, "turn_id": "t-old",
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "stable", "importance": "normal",
        "event_date": None,
    }
    meta.update(meta_over)
    store._insert_memory_row(key, text, meta, None)
    return key


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _judge_confirming(target_marker):
    """判官打桩：候选里带 target_marker 的那条判真冲突、取代它。"""
    def _judge(self, text, candidates, context=None):
        for cand in candidates or []:
            if target_marker in (cand.get("text") or ""):
                return {
                    "should_supersede": True,
                    "target_id": cand.get("id"),
                    "slot": None, "provider": "test", "model": "test",
                    "reason": "测试判官确认冲突", "cost_usd": 0.0,
                }
        return {"should_supersede": False, "target_id": None}
    return _judge


@pytest.fixture
def net_store(tmp_path, monkeypatch):
    """带假向量 + 可控判官的 store（判官默认确认“白饭”那条）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", _net_embed)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge", _judge_confirming("白饭")
    )
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def test_item5_net_catches_cross_slot_conflict_at_exit_a(net_store, monkeypatch):
    """出口 A：召回一条候选都没找到（打桩模拟真目标不在候选里），
    跨槽行“我的主食是白饭”分错了槽、主路和同槽兜底都够不着——
    安全网扫到（与旧说法余弦 1.0 ≥ 0.80）、判官确认后顶掉，
    账本记 supersede（标记不删除），新说法成活行。"""
    _plant_row(net_store, "我的主食是白饭")
    monkeypatch.setattr(
        SQLiteStore, "_find_correction_candidates",
        lambda self, old_fact, new_fact=None, **kw: [],
    )
    result = net_store.correct_memory(
        "我喜欢吃米饭", "我喜欢吃面条", explicit_correction=True
    )
    assert result["status"] == "updated", result
    assert result["resolution"] == "cross_slot_net", result
    assert _is_superseded(net_store, "我的主食是白饭")
    assert "我喜欢吃面条" in _live_texts(net_store)
    events, _info = _ledger.read_events(net_store.ledger_path)
    assert any(
        e.get("type") == "supersede" and e.get("old_text") == "我的主食是白饭"
        for e in events
    )


def test_item5_net_judge_says_no_means_not_found(net_store, monkeypatch):
    """判官说不冲突就绝不顶：命中行在、相似度够，判官判不取代时
    维持 not_found，旧行活着。模型断了（判官返回 None）同理。"""
    _plant_row(net_store, "我的主食是白饭")
    monkeypatch.setattr(
        SQLiteStore, "_find_correction_candidates",
        lambda self, old_fact, new_fact=None, **kw: [],
    )
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: {
            "should_supersede": False, "target_id": None
        },
    )
    result = net_store.correct_memory(
        "我喜欢吃米饭", "我喜欢吃面条", explicit_correction=True
    )
    assert result["status"] == "not_found", result
    assert not _is_superseded(net_store, "我的主食是白饭")
    assert "我的主食是白饭" in _live_texts(net_store)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    result = net_store.correct_memory(
        "我喜欢吃米饭", "我喜欢吃面条", explicit_correction=True
    )
    assert result["status"] == "not_found", result
    assert not _is_superseded(net_store, "我的主食是白饭")


def test_item5_net_never_touches_core_row(net_store, monkeypatch):
    """核心/置顶行哪怕被网扫中、判官也确认，仍不自动顶：
    维持 not_found，核心行活着（动核心永远走问 BOOS 的路径）。"""
    _plant_row(net_store, "我的主食是白饭", curated=True)
    monkeypatch.setattr(
        SQLiteStore, "_find_correction_candidates",
        lambda self, old_fact, new_fact=None, **kw: [],
    )
    result = net_store.correct_memory(
        "我喜欢吃米饭", "我喜欢吃面条", explicit_correction=True
    )
    assert result["status"] == "not_found", result
    assert not _is_superseded(net_store, "我的主食是白饭")
    assert "我的主食是白饭" in _live_texts(net_store)


def test_item5_net_catches_at_exit_b(net_store, monkeypatch):
    """出口 B：召回有两条同槽候选但与新说法相似度都为 0（直选和
    相似度筛全落空），R-9 同槽兜底也捞不到（跨槽行与新说法只有
    0.68）——安全网按与旧说法的高相似（1.0）扫中、判官确认后落盘。"""
    _plant_row(net_store, "我的主食是白饭")
    id_a = _plant_row(net_store, "我喜欢吃火锅")
    id_b = _plant_row(net_store, "我喜欢吃牛肉面")
    monkeypatch.setattr(
        SQLiteStore, "_find_correction_candidates",
        lambda self, old_fact, new_fact=None, **kw: [id_a, id_b],
    )
    result = net_store.correct_memory(
        "我喜欢吃米饭", "我喜欢吃面条", explicit_correction=True
    )
    assert result["status"] == "updated", result
    assert result["resolution"] == "cross_slot_net", result
    assert _is_superseded(net_store, "我的主食是白饭")
    assert not _is_superseded(net_store, "我喜欢吃火锅")
    assert not _is_superseded(net_store, "我喜欢吃牛肉面")


def test_item5_net_below_threshold_not_even_judged(net_store, monkeypatch):
    """门槛宁缺毋滥：与新旧说法相似度都 < 0.80 的行（火锅，与米饭
    正交）连判官都不会被问到——判官打桩记录调用次数为 0，
    结果维持 not_found。"""
    _plant_row(net_store, "我喜欢吃火锅")
    monkeypatch.setattr(
        SQLiteStore, "_find_correction_candidates",
        lambda self, old_fact, new_fact=None, **kw: [],
    )
    calls = []

    def _judge(self, text, candidates, context=None):
        calls.append(candidates)
        return {"should_supersede": True, "target_id": None}

    monkeypatch.setattr(SQLiteStore, "_ask_semantic_judge", _judge)
    result = net_store.correct_memory(
        "我喜欢吃米饭", "我喜欢吃面条", explicit_correction=True
    )
    assert result["status"] == "not_found", result
    assert calls == []
    assert not _is_superseded(net_store, "我喜欢吃火锅")

