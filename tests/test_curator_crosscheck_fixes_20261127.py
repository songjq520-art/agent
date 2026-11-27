# tests/test_curator_crosscheck_fixes_20261127.py（贵妃互查五项修复反例，2026-11-27）
#
# 贵妃互查 G1/G2/G4 不签收，五项发现（F1/F2/F5 高、F4/F3 中）
# 逐项落反例：场景照她的探针 crosscheck_g1g2g4_probes_20261127.py
# ——同一批假宿主形状、同一批分叉构造，断言方向反过来（修前
# 复现缺陷、修后必须朝安全侧倒）。判定一律走她侧夹具真函数
# （curator.assess / host_review.review_proposal /
# correction_receipt / shadow_batch.BatchJournal），不复刻。
#
#   F1 待问读取抛 OSError → 不许空集洗白：pending_memory_ids
#      抛 PendingQuestionsUnavailable；build_current 标
#      pending_unknown 且 pending_classification=True；
#      execute_demote held 在 pending 段、H1 零提交。
#   F2 protection_resolved 只认 H5 核对函数
#      verify_protection_migration 的 ok：无此函数 / 核对不过 /
#      核对抛错一律 False；真库投影被绕账本篡改 → False。
#   F5 纠正证据回查同一份快照：账本折叠说已顶替、快照里旧行
#      还活着 → old_row 如实报未顶替、correction incomplete；
#      两边一致才 complete。
#   F4 影子批次绑真实快照：未登记的 snapshot_id 直接拒绝；
#      落盘 id 恒为组装所用快照的真实 id。
#   F3 影子人口只算活行：默认人口排除死行；显式点名死行报错。
#
# 全程离线：假宿主不接真库；F2 真库用例同 G1 口径打桩。

import copy
import json
import sys
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "curator_protocol"
sys.path.insert(0, str(_FIXTURE_DIR))

import curator as _her_curator  # noqa: E402
import host_review as _her_review  # noqa: E402
import shadow_batch as _her_batch  # noqa: E402

from memory import curator_adapter as _adapter  # noqa: E402
from memory import curator_execute as _exec  # noqa: E402
from memory import curator_shadow as _shadow  # noqa: E402
from memory.sqlite_store import SQLiteStore  # noqa: E402


def _row(mid="m", live=True, state=None, **meta):
    metadata = {"source": "user", "user_confirmed": True, **meta}
    return {
        "memory_id": mid,
        "version": "stamp-" + mid,
        "live": live,
        "state": state if state is not None
        else ("current" if live else "deleted"),
        "meta": metadata,
    }


class FakeStore:
    """最小假宿主：只实现适配/执行/影子三层用到的口。

    故意不带 verify_protection_migration——那是 F2 的「无逐条
    核实凭据」情形。_is_core_protected_row 借宿主真实现（纯
    meta 谓词），不另立口径。
    """

    _is_core_protected_row = SQLiteStore._is_core_protected_row

    def __init__(self, rows=None, *, pending_error=False, ledger=None):
        self.rows = rows if rows is not None else {"m": _row()}
        self.pending_error = pending_error
        self._ledger = ledger or {"by_id": {}}
        self.submissions = []
        self._snapshots = {}

    def list_pending_questions(self):
        if self.pending_error:
            raise OSError("isolated pending read failure")
        return []

    def take_stable_snapshot(self):
        snap = {"snapshot_id": "actual-h4-snapshot",
                "rows": copy.deepcopy(self.rows)}
        self._snapshots[snap["snapshot_id"]] = copy.deepcopy(snap)
        return snap

    def get_stable_snapshot(self, snapshot_id):
        if snapshot_id not in self._snapshots:
            raise KeyError(snapshot_id)
        return copy.deepcopy(self._snapshots[snapshot_id])

    def ledger_state(self):
        return copy.deepcopy(self._ledger)

    def submit_controlled_proposal(self, proposal):
        self.submissions.append(copy.deepcopy(proposal))
        return {"status": "shadow", "ok": False, "duplicate": False}


class VerifiedFakeStore(FakeStore):
    """带 H5 核对函数的假宿主：核对结论由 verify_ok 控制。"""

    def __init__(self, *args, verify_ok=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._verify_ok = verify_ok

    def verify_protection_migration(self, memory_id=None,
                                    evidence_id=None):
        return {
            "ok": self._verify_ok,
            "memory_id": memory_id,
            "evidence_id": evidence_id,
            "checks": {},
            "mismatches": [] if self._verify_ok else ["probe_mismatch"],
            "detail": {},
        }


class RaisingVerifyStore(FakeStore):
    def verify_protection_migration(self, memory_id=None,
                                    evidence_id=None):
        raise RuntimeError("verify backend down")


def _review_fn(store):
    def _review(proposal_fields, current):
        return _her_review.review_proposal(
            _her_review.Proposal(**proposal_fields), current,
            protected=lambda meta: _adapter.host_protected(store, meta),
            classify=_adapter.host_classify,
        )
    return _review


def _assess_fn(fields):
    return _her_curator.assess(_her_curator.Candidate(**fields))


# ---------- F1：待问读取失败不许洗成无待问 ----------

def test_f1_pending_read_failure_raises_not_empty_set():
    store = FakeStore(pending_error=True)
    with pytest.raises(_adapter.PendingQuestionsUnavailable):
        _adapter.pending_memory_ids(store)


def test_f1_pending_unknown_marks_current_and_holds_execute():
    # 核对函数在且给 ok：把保护关放过去，单独验待问未知这道门。
    store = VerifiedFakeStore(pending_error=True)
    row = _row()
    current = _adapter.build_current(store, row)
    assert current["pending_unknown"] is True
    assert current["pending_classification"] is True
    result = _exec.execute_demote(
        store, row, judge_status="valid", idempotency_key="f1-key",
        assess_fn=_assess_fn, review_fn=_review_fn(store),
    )
    assert result["outcome"] == "held"
    assert result["stage"] == "pending"
    assert store.submissions == []


def test_f1_pending_readable_has_no_unknown_flag():
    store = VerifiedFakeStore()
    current = _adapter.build_current(store, _row())
    assert current["pending_unknown"] is False
    assert current["pending_classification"] is False


# ---------- F2：protection_resolved 只认逐条核对凭据 ----------

def test_f2_no_verify_function_means_unresolved():
    store = FakeStore()
    row = _row()
    current = _adapter.build_current(store, row)
    assert current["protection_resolved"] is False
    verdict = _review_fn(store)(
        _adapter.build_proposal_fields(row), current)
    assert verdict == "hold_unresolved_protection"


def test_f2_verify_not_ok_means_unresolved():
    store = VerifiedFakeStore(verify_ok=False)
    current = _adapter.build_current(store, _row())
    assert current["protection_resolved"] is False


def test_f2_verify_raising_means_unresolved():
    store = RaisingVerifyStore()
    current = _adapter.build_current(store, _row())
    assert current["protection_resolved"] is False


def test_f2_verify_ok_means_resolved():
    store = VerifiedFakeStore(verify_ok=True)
    current = _adapter.build_current(store, _row())
    assert current["protection_resolved"] is True


@pytest.fixture
def real_store(tmp_path, monkeypatch):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.organizing_router = lambda: None
    return s


def _snap_row(store, mid):
    snap = store.take_stable_snapshot()
    row = dict(snap["rows"][mid])
    row["memory_id"] = mid
    return row


def test_f2_real_store_projection_tamper_unresolved(real_store):
    saved = real_store.add_memory_text("用户的工牌号不告诉别人",
                                       source="user")
    assert saved.get("event") == "ADD", saved
    mid = saved["id"]
    assert real_store.set_protection_tier(mid, "pinned")["ok"] is True
    # 正面对照：刚迁移完，核对过，resolved 为真
    current = _adapter.build_current(real_store, _snap_row(real_store, mid))
    assert current["protection_resolved"] is True
    # 绕账本改投影：投影档位与账本档位链分叉（她的 F2 场景）
    row = _snap_row(real_store, mid)
    bad_meta = dict(row["meta"])
    bad_meta["protection_tier"] = "temp_pinned"
    real_store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (json.dumps(bad_meta, ensure_ascii=False), mid),
    )
    real_store._conn.commit()
    current = _adapter.build_current(real_store, _snap_row(real_store, mid))
    assert current["protection_resolved"] is False


# ---------- F5：纠正证据回查同一份快照 ----------

def _f5_store(old_state, old_live):
    rows = {
        "old": _row("old", old_live, state=old_state),
        "new": _row("new", True, turn_id="turn"),
    }
    ledger = {"by_id": {
        "old": {"state": "superseded", "superseded_by": "new", "seq": 7},
    }}
    return FakeStore(rows, ledger=ledger)


def test_f5_divergent_snapshot_old_still_live_incomplete():
    # 账本折叠说旧行已顶替，快照里旧行还活着：分叉，不许报完成
    store = _f5_store("current", True)
    ev, old, new = _adapter.correction_evidence(
        store, {"status": "updated", "memory_id": "new"}, turn_id="turn")
    assert old["superseded"] is False
    result = _her_review.correction_receipt(ev, old, new, turn_id="turn")
    assert result["complete"] is False


def test_f5_consistent_snapshot_and_ledger_complete():
    store = _f5_store("superseded", False)
    ev, old, new = _adapter.correction_evidence(
        store, {"status": "updated", "memory_id": "new"}, turn_id="turn")
    assert old["superseded"] is True
    assert old["superseded_by"] == "new"
    result = _her_review.correction_receipt(ev, old, new, turn_id="turn")
    assert result["complete"] is True


# ---------- F4：影子批次绑真实快照 ----------

def test_f4_unregistered_snapshot_id_rejected(tmp_path):
    store = FakeStore()
    journal = _her_batch.BatchJournal(tmp_path / "shadow")
    with pytest.raises(ValueError):
        _shadow.run_shadow_batch(
            store, journal, batch_id="b", snapshot_id="not-the-h4-id",
            proposal_cls=_her_review.Proposal)


def test_f4_registered_snapshot_id_bound_and_persisted(tmp_path):
    store = FakeStore()
    snap_id = store.take_stable_snapshot()["snapshot_id"]
    journal = _her_batch.BatchJournal(tmp_path / "shadow")
    payload = _shadow.run_shadow_batch(
        store, journal, batch_id="b", snapshot_id=snap_id,
        proposal_cls=_her_review.Proposal)
    assert payload["snapshot_id"] == "actual-h4-snapshot"
    assert journal.read("b")["snapshot_id"] == "actual-h4-snapshot"
    assert payload["execution_allowed"] is False


def test_f4_no_id_persists_actual_snapshot_id(tmp_path):
    store = FakeStore()
    journal = _her_batch.BatchJournal(tmp_path / "shadow")
    payload = _shadow.run_shadow_batch(
        store, journal, batch_id="b",
        proposal_cls=_her_review.Proposal)
    assert payload["snapshot_id"] == "actual-h4-snapshot"


# ---------- F3：影子人口只算活行 ----------

def test_f3_dead_rows_excluded_from_population():
    store = FakeStore({"m": _row(), "dead": _row("dead", False)})
    items = _shadow.build_batch_items(
        store, proposal_cls=_her_review.Proposal)
    assert [p.memory_id for p, _ in items] == ["m"]


def test_f3_explicit_dead_selection_rejected():
    store = FakeStore({"m": _row(), "dead": _row("dead", False)})
    with pytest.raises(ValueError):
        _shadow.build_batch_items(
            store, proposal_cls=_her_review.Proposal,
            memory_ids=["dead"])


def test_f3_shadow_batch_population_excludes_dead(tmp_path):
    store = FakeStore({"m": _row(), "dead": _row("dead", False)})
    journal = _her_batch.BatchJournal(tmp_path / "shadow")
    payload = _shadow.run_shadow_batch(
        store, journal, batch_id="b",
        proposal_cls=_her_review.Proposal)
    assert [r["memory_id"] for r in payload["rows"]] == ["m"]
