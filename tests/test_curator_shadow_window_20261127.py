# tests/test_curator_shadow_window_20261127.py（G4 观察窗生产接线，2026-11-27）
#
# G4 沙盒交付的是影子机制＋四关证据；本文件测的是把观察窗
# 接进真服务所缺的生产三件：
#   1. 生产协议件（memory/curator_protocol，她侧协议逐字移植）
#      与夹具真件判定一致、落盘互通——她侧 BatchJournal 能
#      直接读宿主生产件落的批次并 compare 出 matched；
#   2. 生产跑批器（memory/curator_shadow_runner）：开窗落
#      台账（起始日按北京时间、钉住 POLICY_VERSION）、同日
#      重跑幂等、次日新批次、口径一变窗口作废停跑（G3-U4）、
#      批次日志与台账不落记忆原文；
#   3. 调度触发（MemoryMaintenanceScheduler 的 shadow 项）：
#      注入才有 shadow 键、到期才跑、异常只记日志不拖垮调度、
#      start_memory_maintenance 的生产接线真能跑出第一批。
#
# 全程离线：她侧协议用逐字夹具真件对照，宿主 store 用真
# SQLiteStore（嵌模/判官照 G4 测试的关法）。

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "curator_protocol"
sys.path.insert(0, str(_FIXTURE_DIR))

import host_review as _her_review  # noqa: E402
import shadow_batch as _her_batch  # noqa: E402

from memory import curator_adapter as _adapter  # noqa: E402
from memory import curator_protocol as _proto  # noqa: E402
from memory import curator_shadow as _shadow  # noqa: E402
from memory import curator_shadow_runner as _runner  # noqa: E402
from memory.sqlite_store import SQLiteStore  # noqa: E402
from web.server import (  # noqa: E402
    MemoryMaintenanceScheduler,
    start_memory_maintenance,
)

_SHANGHAI = _runner.SHANGHAI_TZ


def _epoch(year, month, day, hour=12):
    return datetime(year, month, day, hour, tzinfo=_SHANGHAI).timestamp()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.organizing_router = lambda: None
    return s


def _add_user(store, text):
    saved = store.add_memory_text(text, source="user")
    assert saved.get("event") == "ADD", saved
    return saved["id"]


def _seed_two(store):
    plain = _add_user(store, "用户每天定投五块钱买基金")
    pinned = _add_user(store, "用户亲口定的铁律不许动")
    assert store.set_protection_tier(pinned, "pinned")["ok"] is True
    return plain, pinned


# ---------- 1. 生产协议件与她侧夹具互通 ----------

def test_production_journal_decisions_match_fixture(store, tmp_path):
    _seed_two(store)
    snap_id = store.take_stable_snapshot()["snapshot_id"]
    prod = _shadow.run_shadow_batch(
        store, _proto.BatchJournal(tmp_path / "prod"),
        batch_id="batch-x", snapshot_id=snap_id,
        proposal_cls=_proto.Proposal)
    hers = _shadow.run_shadow_batch(
        store, _her_batch.BatchJournal(tmp_path / "hers"),
        batch_id="batch-x", snapshot_id=snap_id,
        proposal_cls=_her_review.Proposal)
    assert prod == hers
    assert prod["execution_allowed"] is False
    assert prod["scope"] == "isolated_shadow_only"


def test_fixture_journal_reads_production_batch(store, tmp_path):
    plain, pinned = _seed_two(store)
    journal_dir = tmp_path / "journal"
    prod_journal = _proto.BatchJournal(journal_dir)
    payload = _shadow.run_shadow_batch(
        store, prod_journal, batch_id="batch-compat",
        proposal_cls=_proto.Proposal)
    # 她侧真件打开同一目录：读得回、比得上
    her_journal = _her_batch.BatchJournal(journal_dir)
    assert her_journal.read("batch-compat") == payload
    report = her_journal.compare(
        "batch-compat", {plain: "candidate", pinned: "hold"})
    assert report["matched"] is True
    assert report["execution_allowed"] is False


def test_production_journal_idempotent_and_conflict_rejected(store, tmp_path):
    _seed_two(store)
    journal = _proto.BatchJournal(tmp_path / "journal")
    snap_id = store.take_stable_snapshot()["snapshot_id"]
    first = _shadow.run_shadow_batch(
        store, journal, batch_id="batch-i", snapshot_id=snap_id,
        proposal_cls=_proto.Proposal)
    second = _shadow.run_shadow_batch(
        store, journal, batch_id="batch-i", snapshot_id=snap_id,
        proposal_cls=_proto.Proposal)
    assert first == second
    _add_user(store, "用户每周三去游泳")
    with pytest.raises(ValueError):
        _shadow.run_shadow_batch(
            store, journal, batch_id="batch-i",
            proposal_cls=_proto.Proposal)


# ---------- 2. 生产跑批器与窗口台账 ----------

def test_runner_opens_window_and_records_batch(store, tmp_path):
    plain, pinned = _seed_two(store)
    base = tmp_path / "memory-data"
    now = _epoch(2026, 11, 28)
    result = _runner.run_daily_shadow_batch(store, base, now=now)
    assert result["status"] == "ran"
    assert result["batch_id"] == "shadow-20261128"
    assert result["date"] == "2026-11-28"
    assert result["rows"] == 2
    assert result["proposals"] == 1
    assert result["holds"] == 1
    state = _runner.load_window_state(base)
    assert state["window_start"] == "2026-11-28"
    assert state["pinned_policy_version"] == _adapter.POLICY_VERSION
    assert state["voided"] is False
    assert [b["batch_id"] for b in state["batches"]] == ["shadow-20261128"]
    # 批次真落盘、她侧格式读得回
    journal = _proto.BatchJournal(_runner.journal_dir(base))
    payload = journal.read("shadow-20261128")
    assert payload["execution_allowed"] is False
    results = {r["memory_id"]: r["result"] for r in payload["rows"]}
    assert results[plain] == "shadow_proposal_only"
    assert results[pinned].startswith("hold_")
    status = _runner.window_status(base, now=now)
    assert status["opened"] is True
    assert status["batches"] == 1
    assert status["days_elapsed"] == 1
    assert status["window_complete"] is False


def test_runner_same_day_rerun_idempotent(store, tmp_path):
    _seed_two(store)
    base = tmp_path / "memory-data"
    now = _epoch(2026, 11, 28, hour=9)
    first = _runner.run_daily_shadow_batch(store, base, now=now)
    assert first["status"] == "ran"
    second = _runner.run_daily_shadow_batch(store, base, now=now + 3600)
    assert second["status"] == "already_ran"
    assert second["batch_id"] == "shadow-20261128"
    state = _runner.load_window_state(base)
    assert len(state["batches"]) == 1


def test_runner_next_day_new_batch(store, tmp_path):
    _seed_two(store)
    base = tmp_path / "memory-data"
    day1 = _epoch(2026, 11, 28)
    day2 = _epoch(2026, 11, 29)
    _runner.run_daily_shadow_batch(store, base, now=day1)
    result = _runner.run_daily_shadow_batch(store, base, now=day2)
    assert result["status"] == "ran"
    assert result["batch_id"] == "shadow-20261129"
    state = _runner.load_window_state(base)
    assert [b["batch_id"] for b in state["batches"]] == [
        "shadow-20261128", "shadow-20261129"]
    status = _runner.window_status(base, now=day2)
    assert status["days_elapsed"] == 2
    assert status["batches"] == 2


def test_runner_policy_change_voids_window(store, tmp_path, monkeypatch):
    _seed_two(store)
    base = tmp_path / "memory-data"
    day1 = _epoch(2026, 11, 28)
    _runner.run_daily_shadow_batch(store, base, now=day1)
    monkeypatch.setattr(
        _adapter, "POLICY_VERSION", "curator-policy-2099-v9")
    day2 = _epoch(2026, 11, 29)
    result = _runner.run_daily_shadow_batch(store, base, now=day2)
    assert result["status"] == "voided"
    assert result["pinned_policy_version"] == "curator-policy-20261124-v1"
    state = _runner.load_window_state(base)
    assert state["voided"] is True
    assert state["voided_reason"] == "policy_version_changed"
    assert len(state["batches"]) == 1  # 作废后绝不混跑新批次
    with pytest.raises(KeyError):
        _proto.BatchJournal(_runner.journal_dir(base)).read("shadow-20261129")
    # 停跑是持续的：再调还是 voided，不自作主张重开
    again = _runner.run_daily_shadow_batch(store, base, now=day2 + 3600)
    assert again["status"] == "voided"
    status = _runner.window_status(base, now=day2)
    assert status["voided"] is True
    assert status["window_complete"] is False


def test_runner_logs_contain_no_memory_text(store, tmp_path):
    secret_text = "用户每天定投五块钱买基金"
    _add_user(store, secret_text)
    base = tmp_path / "memory-data"
    _runner.run_daily_shadow_batch(store, base, now=_epoch(2026, 11, 28))
    journal_bytes = (
        _runner.journal_dir(base) / "shadow_batches.sqlite3"
    ).read_bytes()
    assert secret_text.encode("utf-8") not in journal_bytes
    state_text = _runner.window_state_path(base).read_text(encoding="utf-8")
    assert secret_text not in state_text


def test_runner_window_not_opened_status(store, tmp_path):
    assert _runner.window_status(tmp_path) == {
        "opened": False, "voided": False, "batches": 0}
    assert _runner.load_window_state(tmp_path) is None


def test_runner_corrupt_window_state_refused(store, tmp_path):
    base = tmp_path / "memory-data"
    base.mkdir(parents=True)
    _runner.window_state_path(base).write_text("{坏了", encoding="utf-8")
    with pytest.raises(ValueError):
        _runner.run_daily_shadow_batch(store, base, now=_epoch(2026, 11, 28))


# ---------- 3. 调度触发 ----------

class _BareStore:
    """没有 sweep_decay / run_self_check 的空 store：调度里那
    两项按既有语义自己跳过，本组用例只看 shadow 项。"""


def _sched(tmp_path, runner, interval=100):
    return MemoryMaintenanceScheduler(
        _BareStore(), tmp_path / "maintenance_state.json",
        decay_interval_seconds=100, self_check_interval_seconds=100,
        tick_seconds=60, shadow_runner=runner,
        shadow_interval_seconds=interval,
    )


def test_scheduler_without_shadow_runner_shape_unchanged(tmp_path):
    sched = MemoryMaintenanceScheduler(
        _BareStore(), tmp_path / "maintenance_state.json",
        decay_interval_seconds=100, self_check_interval_seconds=100,
        tick_seconds=60,
    )
    ran = sched.run_due(now=1_000)
    assert "shadow" not in ran


def test_scheduler_shadow_due_then_interval(tmp_path):
    calls = []

    def runner():
        calls.append(1)
        return {"status": "ran", "batch_id": "shadow-20261128"}

    sched = _sched(tmp_path, runner)
    ran = sched.run_due(now=10_000)
    assert ran["shadow"] is True
    assert calls == [1]
    state = json.loads(
        (tmp_path / "maintenance_state.json").read_text(encoding="utf-8"))
    assert state["shadow_last_run"] == 10_000
    assert state["shadow_last_status"] == "ran"
    assert state["shadow_last_batch"] == "shadow-20261128"
    # 没到间隔：不重跑
    ran = sched.run_due(now=10_050)
    assert ran["shadow"] is False
    assert calls == [1]
    # 到间隔：再跑
    ran = sched.run_due(now=10_100)
    assert ran["shadow"] is True
    assert calls == [1, 1]


def test_scheduler_shadow_failure_logged_not_fatal(tmp_path):
    def runner():
        raise RuntimeError("simulated shadow failure")

    sched = _sched(tmp_path, runner)
    ran = sched.run_due(now=10_000)  # 不许抛
    assert ran["shadow"] is True  # 尝试过、记了时间，下个间隔再试
    state = json.loads(
        (tmp_path / "maintenance_state.json").read_text(encoding="utf-8"))
    assert state["shadow_last_run"] == 10_000
    assert state["shadow_last_status"] == "error"
    ran = sched.run_due(now=10_050)
    assert ran["shadow"] is False  # 不每个 tick 重试刷屏


def test_start_memory_maintenance_wires_shadow_runner(store, tmp_path,
                                                      monkeypatch):
    monkeypatch.delenv("SHIER_HEALTHCHECK_CONFIG", raising=False)
    monkeypatch.setattr(
        MemoryMaintenanceScheduler, "start", lambda self: None)
    scheduler = start_memory_maintenance(store)
    result = scheduler._shadow_runner()
    assert result["status"] == "ran"
    assert result["rows"] == 0
    # 台账与批次落在账本同目录（维护状态隔壁）
    memory_dir = Path(store.ledger_path).parent
    state = _runner.load_window_state(memory_dir)
    assert state is not None
    assert state["pinned_policy_version"] == _adapter.POLICY_VERSION
    assert len(state["batches"]) == 1
