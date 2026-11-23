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
