# tests/test_curator_shadow_g4_20261127.py（立项 G4：影子观察＋四关，2026-11-27）
#
# 四关口径（施工契约六＋立项清单 G4）：
#   关一 身份保护：钉死行在批次里永远落 hold、绝不
#     shadow_proposal_only；批次 execution_allowed 恒 False；
#   关二 并发/崩溃恢复：同批重收幂等（同证据同结果）、同
#     batch_id 异证据拒收、collect 中途崩溃不落半批、
#     两批并发落盘互不串扰；
#   关三 账本对账：跑批前后宿主账本事件数、行版本、座位
#     全不变（影子只读）；同状态在新日志里重收得同判定
#     （证据可复现对账）；
#   关四 差异复核：独立人工标签与机器判定一致 → matched；
#     标签错 → differences 点名；标签缺/不确定 →
#     comparison_complete=False；compare 恒不放行。
#
# 她侧 BatchJournal / Proposal 用逐字夹具真件；宿主组装走
# memory/curator_shadow.py（G1 适配层之上）。全程离线。

import sys
import threading
import uuid
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "curator_protocol"
sys.path.insert(0, str(_FIXTURE_DIR))

import host_review as _her_review  # noqa: E402
import shadow_batch as _her_batch  # noqa: E402

from memory import curator_shadow as _shadow  # noqa: E402
from memory.sqlite_store import SQLiteStore  # noqa: E402


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


def _journal(tmp_path, name="shadow"):
    return _her_batch.BatchJournal(tmp_path / name)


def _run(store, journal, batch_id="batch-g4"):
    return _shadow.run_shadow_batch(
        store, journal, batch_id=batch_id,
        snapshot_id="snap-" + batch_id,
        proposal_cls=_her_review.Proposal,
    )


def _results(payload):
    return {row["memory_id"]: row["result"] for row in payload["rows"]}


# ---------- 关一：身份保护 ----------

def test_gate1_protected_rows_never_proposed(store, tmp_path):
    plain = _add_user(store, "用户每天定投五块钱买基金")
    pinned = _add_user(store, "用户亲口定的铁律不许动")
    assert store.set_protection_tier(pinned, "pinned")["ok"] is True
    payload = _run(store, _journal(tmp_path))
    assert payload["execution_allowed"] is False
    assert payload["scope"] == "isolated_shadow_only"
    results = _results(payload)
    assert results[plain] == "shadow_proposal_only"
    assert results[pinned] != "shadow_proposal_only"
    assert results[pinned].startswith("hold_")


# ---------- 关二：并发 / 崩溃恢复 ----------

def test_gate2_recollect_idempotent(store, tmp_path):
    _add_user(store, "用户每天定投五块钱买基金")
    journal = _journal(tmp_path)
    first = _run(store, journal)
    second = _run(store, journal)
    assert first == second


def test_gate2_same_batch_id_different_evidence_rejected(store, tmp_path):
    _add_user(store, "用户每天定投五块钱买基金")
    journal = _journal(tmp_path)
    _run(store, journal)
    _add_user(store, "用户每周三去游泳")
    with pytest.raises(ValueError):
        _run(store, journal)


def test_gate2_guard_error_holds_row_not_batch(store, tmp_path):
    # 她侧复核层把回调异常收成 hold_host_guard_error：单行
    # 守卫炸了只挂那一行，不炸整批（故障注入实测口径）。
    _add_user(store, "用户每天定投五块钱买基金")
    _add_user(store, "用户每周三去游泳")
    journal = _journal(tmp_path)
    items = _shadow.build_batch_items(store, proposal_cls=_her_review.Proposal)
    from memory import curator_adapter as _adapter
    calls = {"n": 0}

    def exploding_classify(meta, live):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated guard crash")
        return _adapter.host_classify(meta, live)

    payload = journal.collect(
        "batch-guard", "snap-guard", _adapter.POLICY_VERSION, items,
        protected=lambda meta: _adapter.host_protected(store, meta),
        classify=exploding_classify,
    )
    results = sorted(row["result"] for row in payload["rows"])
    assert results == ["hold_host_guard_error", "shadow_proposal_only"]


def test_gate2_crash_at_write_leaves_no_partial_batch(store, tmp_path,
                                                      monkeypatch):
    _add_user(store, "用户每天定投五块钱买基金")
    journal = _journal(tmp_path)
    real_connect = _her_batch.sqlite3.connect

    def dead_connect(*args, **kwargs):
        raise RuntimeError("simulated crash at write")

    monkeypatch.setattr(_her_batch.sqlite3, "connect", dead_connect)
    with pytest.raises(RuntimeError):
        _run(store, journal, batch_id="batch-crash")
    monkeypatch.setattr(_her_batch.sqlite3, "connect", real_connect)
    with pytest.raises(KeyError):
        journal.read("batch-crash")
    # 恢复后同 batch_id 可干净重收（崩溃不留毒）
    payload = _run(store, journal, batch_id="batch-crash")
    assert payload["rows"][0]["result"] == "shadow_proposal_only"


def test_gate2_concurrent_batches_do_not_interfere(store, tmp_path):
    _add_user(store, "用户每天定投五块钱买基金")
    journal = _journal(tmp_path)
    errors = []

    def worker(n):
        try:
            _run(store, journal, batch_id=f"batch-conc-{n}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    for n in range(2):
        payload = journal.read(f"batch-conc-{n}")
        assert payload["execution_allowed"] is False
        assert len(payload["rows"]) == 1


# ---------- 关三：账本对账 ----------

def test_gate3_shadow_run_leaves_host_untouched(store, tmp_path):
    _add_user(store, "用户每天定投五块钱买基金")
    pinned = _add_user(store, "用户亲口定的铁律不许动")
    assert store.set_protection_tier(pinned, "pinned")["ok"] is True

    def fingerprint():
        state = store.ledger_state()
        snap = store.take_stable_snapshot()
        return (
            state["seq_max"],
            {mid: row["version"] for mid, row in snap["rows"].items()},
            store.core_seat_snapshot()["seats_used"],
        )

    before = fingerprint()
    _run(store, _journal(tmp_path))
    assert fingerprint() == before


def test_gate3_same_state_reproduces_same_decisions(store, tmp_path):
    _add_user(store, "用户每天定投五块钱买基金")
    first = _run(store, _journal(tmp_path, "j1"))
    second = _run(store, _journal(tmp_path, "j2"))
    assert _results(first) == _results(second)


# ---------- 关四：差异复核 ----------

def test_gate4_matching_labels_match(store, tmp_path):
    plain = _add_user(store, "用户每天定投五块钱买基金")
    pinned = _add_user(store, "用户亲口定的铁律不许动")
    assert store.set_protection_tier(pinned, "pinned")["ok"] is True
    journal = _journal(tmp_path)
    _run(store, journal)
    report = _shadow.compare_batch(
        journal, "batch-g4", {plain: "candidate", pinned: "hold"})
    assert report["matched"] is True
    assert report["comparison_complete"] is True
    assert report["execution_allowed"] is False


def test_gate4_wrong_label_reported_as_difference(store, tmp_path):
    plain = _add_user(store, "用户每天定投五块钱买基金")
    journal = _journal(tmp_path)
    _run(store, journal)
    report = _shadow.compare_batch(journal, "batch-g4", {plain: "hold"})
    assert report["matched"] is False
    assert report["differences"] == [
        {"memory_id": plain, "expected": "hold", "actual": "candidate"}]


def test_gate4_missing_and_uncertain_labels_incomplete(store, tmp_path):
    plain = _add_user(store, "用户每天定投五块钱买基金")
    other = _add_user(store, "用户每周三去游泳")
    journal = _journal(tmp_path)
    _run(store, journal)
    report = _shadow.compare_batch(journal, "batch-g4", {plain: "uncertain"})
    assert report["comparison_complete"] is False
    assert report["uncertain"] == [plain]
    assert report["missing"] == [other]
    assert report["matched"] is False


# ---------- 接线护栏 ----------

def test_unwired_journal_refused(store):
    _add_user(store, "用户每天定投五块钱买基金")
    with pytest.raises(PermissionError):
        _shadow.run_shadow_batch(
            store, None, batch_id="b", snapshot_id="s",
            proposal_cls=_her_review.Proposal)


def test_unknown_memory_id_rejected(store, tmp_path):
    _add_user(store, "用户每天定投五块钱买基金")
    with pytest.raises(ValueError):
        _shadow.run_shadow_batch(
            store, _journal(tmp_path), batch_id="b", snapshot_id="s",
            proposal_cls=_her_review.Proposal,
            memory_ids=["no-such-" + uuid.uuid4().hex])
