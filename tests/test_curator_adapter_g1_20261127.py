# tests/test_curator_adapter_g1_20261127.py（立项 G1：字段映射，2026-11-27）
#
# 覆盖口径（立项清单 G1＋映射表 g1_field_mapping_20261127.md）：
#   - 她侧三模块（curator / host_review / shadow_batch）逐字夹具
#     导入，适配层（memory/curator_adapter.py）产物直接喂她的
#     真判定函数跑，不是复刻她的逻辑自说自话；
#   - 版本映射：宿主单戳双绑 content/protection_version，
#     policy_version 两边同常量；行一变，旧提案对新现状必
#     hold_stale_or_missing_version；
#   - 正例：普通核心成员 → review shadow_proposal_only、
#     assess（judge valid）shadow_candidate_only、影子批次
#     collect 落 shadow_proposal_only 且 execution_allowed=False；
#   - 反例逐项落码：钉死行 hold_core_protected_or_unknown /
#     hold_protected、性情投影行 hold_readonly_or_unknown、
#     待问未答 hold_pending_or_unknown、非成员 hold_not_seated_core、
#     推断来源 assess hold_unverified_source、判官非 valid
#     落 hold_judge_*；
#   - H1 侧：适配层组装的 demote 提案影子模式得 shadow 回执
#     （开关关不真执行）；expected_version 写错 → version_stale；
#   - 纠正证据：真落账纠正拼装后因缺口 G1-F1（宿主无共享
#     event_id）得 incomplete——断言 incomplete 是当前诚实
#     预期，缺口关上后须改断言为 complete（见 test 尾注）。
#
# 全程离线：嵌入/语义判官打桩为 None；纠正落账用例单独把
# _judge_degraded 打桩为 False（本文件作用域内，不污染全套）。

import sys
import uuid
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "curator_protocol"
sys.path.insert(0, str(_FIXTURE_DIR))

import curator as _her_curator  # noqa: E402
import host_review as _her_review  # noqa: E402
import shadow_batch as _her_batch  # noqa: E402

from memory import curator_adapter as _adapter  # noqa: E402
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


def _add_user(store, text, **kwargs):
    saved = store.add_memory_text(text, source="user", **kwargs)
    assert saved.get("event") == "ADD", saved
    return saved["id"]


def _snap_row(store, mid):
    snap = store.take_stable_snapshot()
    row = dict(snap["rows"][mid])
    row["memory_id"] = mid
    return row


def _review(store, proposal_fields, current):
    proposal = _her_review.Proposal(**proposal_fields)
    return _her_review.review_proposal(
        proposal, current,
        protected=lambda meta: _adapter.host_protected(store, meta),
        classify=_adapter.host_classify,
    )


def _assess(candidate_fields):
    return _her_curator.assess(_her_curator.Candidate(**candidate_fields))


# ---------- 版本与正例 ----------

def test_versions_bind_single_host_stamp(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    row = _snap_row(store, mid)
    fields = _adapter.build_proposal_fields(row)
    assert fields["memory_id"] == mid
    assert fields["content_version"] == row["version"]
    # 宿主单戳双绑：保护版本与内容版本同值（映射表第一节）
    assert fields["protection_version"] == row["version"]
    assert fields["policy_version"] == _adapter.POLICY_VERSION
    # 与受控入口现算戳同法
    assert fields["content_version"] == store.controlled_entry_version(mid)


def test_ordinary_core_member_passes_review_and_assess(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    row = _snap_row(store, mid)
    proposal = _adapter.build_proposal_fields(row)
    current = _adapter.build_current(store, row)
    assert _review(store, proposal, current) == "shadow_proposal_only"
    candidate = _adapter.build_candidate_fields(row, judge_status="valid")
    assert candidate["core"] is True
    assert candidate["protection"] == "ordinary"
    assert candidate["source_verified"] is True
    assert _assess(candidate) == "shadow_candidate_only"


def test_shadow_batch_collect_with_adapter_inputs(store, tmp_path):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    row = _snap_row(store, mid)
    proposal = _her_review.Proposal(**_adapter.build_proposal_fields(row))
    current = _adapter.build_current(store, row)
    journal = _her_batch.BatchJournal(tmp_path / "shadow")
    payload = journal.collect(
        "batch-g1", "snap-g1", _adapter.POLICY_VERSION,
        [(proposal, current)],
        protected=lambda meta: _adapter.host_protected(store, meta),
        classify=_adapter.host_classify,
    )
    assert payload["execution_allowed"] is False
    assert payload["rows"][0]["result"] == "shadow_proposal_only"


# ---------- 反例逐项落码 ----------

def test_version_drift_holds_stale(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    proposal = _adapter.build_proposal_fields(_snap_row(store, mid))
    assert store.set_core_drawer(mid, "money_decisions")["ok"] is True
    current = _adapter.build_current(store, _snap_row(store, mid))
    assert _review(store, proposal, current) == "hold_stale_or_missing_version"


def test_pinned_row_held_protected(store):
    mid = _add_user(store, "用户的工作是程序员")
    assert store.set_protection_tier(mid, "pinned")["ok"] is True
    row = _snap_row(store, mid)
    proposal = _adapter.build_proposal_fields(row)
    current = _adapter.build_current(store, row)
    verdict = _review(store, proposal, current)
    assert verdict in ("hold_core_protected_or_unknown", "hold_protected")
    candidate = _adapter.build_candidate_fields(row, judge_status="valid")
    assert candidate["protection"] == "pinned"
    assert _assess(candidate) == "hold_protected_or_unknown"


def test_persona_projection_row_held_readonly(store):
    row = {
        "memory_id": "proj-1",
        "version": "stamp-x",
        "live": True,
        "state": "current",
        "meta": {"persona_projection": True, "source": "user",
                 "user_confirmed": True},
    }
    current = _adapter.build_current(store, row, pending_ids=set())
    assert current["readonly_projection"] is True
    proposal = _adapter.build_proposal_fields(row)
    assert _review(store, proposal, current) == "hold_readonly_or_unknown"


def test_pending_question_holds(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    store.pending_questions.add(
        "importance", "这条要进核心吗", ["进核心", "当一次性经历", "不存"],
        "测试待问", memory_id=mid,
    )
    row = _snap_row(store, mid)
    current = _adapter.build_current(store, row)
    assert current["pending_classification"] is True
    proposal = _adapter.build_proposal_fields(row)
    assert _review(store, proposal, current) == "hold_pending_or_unknown"


def test_non_member_and_unverified_source_held(store):
    # 推断来源、未确认：非核心成员且来源未核实
    saved = store.add_memory_text("用户可能喜欢爬山", source="inferred")
    mid = saved["id"]
    row = _snap_row(store, mid)
    candidate = _adapter.build_candidate_fields(row, judge_status="valid")
    assert candidate["core"] is False
    assert candidate["source_verified"] is False
    assert _assess(candidate) == "skip_not_core"
    proposal = _adapter.build_proposal_fields(row)
    current = _adapter.build_current(store, row)
    assert _review(store, proposal, current) == "hold_not_seated_core"


def test_judge_status_mapping(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    row = _snap_row(store, mid)
    for status, expected in (
        ("unreachable", "hold_judge_unreachable"),
        ("budget_exhausted", "hold_judge_budget_exhausted"),
        ("uncertain", "hold_judge_uncertain"),
        (None, "hold_judge_uncertain"),  # 缺失归一 uncertain
    ):
        candidate = _adapter.build_candidate_fields(row, judge_status=status)
        assert _assess(candidate) == expected, status


# ---------- H1 提案组装 ----------

def test_h1_proposal_shadow_roundtrip(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    fields = _adapter.build_proposal_fields(_snap_row(store, mid))
    proposal = _adapter.h1_proposal(
        fields, reason="G1 映射验收", instruction_source="owner",
        idempotency_key="g1-" + uuid.uuid4().hex,
    )
    assert proposal["action"] == "demote"
    assert proposal["expected_version"] == fields["content_version"]
    receipt = store.submit_controlled_proposal(proposal)
    # 执行开关默认关：只出影子回执，不真执行
    assert receipt["status"] == "shadow", receipt
    summary = _adapter.receipt_summary(receipt)
    assert summary["shadow"] is True and summary["executed"] is False
    # 行未被动过：仍是占座核心成员
    again = _adapter.build_candidate_fields(_snap_row(store, mid))
    assert again["core"] is True


def test_h1_proposal_wrong_version_refused(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    fields = _adapter.build_proposal_fields(_snap_row(store, mid))
    fields["content_version"] = "0" * 64
    proposal = _adapter.h1_proposal(
        fields, instruction_source="owner",
        idempotency_key="g1-bad-" + uuid.uuid4().hex,
    )
    receipt = store.submit_controlled_proposal(proposal)
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "version_stale"


# ---------- 纠正证据（缺口 G1-F1） ----------

def test_correction_evidence_incomplete_until_g1f1(store, monkeypatch):
    # 判官降级期纠正只挂待问、不落账；本文件内打桩成不降级，
    # 让纠正真落账（作用域仅本测试，不影响全套其他文件）。
    monkeypatch.setattr(SQLiteStore, "_judge_degraded", lambda self: False)
    mid = _add_user(store, "BOOS 喜欢喝美式咖啡")
    receipt = store.correct_memory(
        "BOOS 喜欢喝美式咖啡", "BOOS 喜欢喝拿铁", turn_id="turn-g1-1",
    )
    assert receipt["status"] == "updated", receipt
    ev_receipt, old_row, new_row = _adapter.correction_evidence(
        store, receipt, turn_id="turn-g1-1",
    )
    assert ev_receipt["new_id"] == receipt["memory_id"]
    assert ev_receipt["old_id"] == mid
    assert old_row.get("superseded") is True
    assert new_row.get("live") is True
    result = _her_review.correction_receipt(
        ev_receipt, old_row, new_row, turn_id="turn-g1-1",
    )
    # 关联证据走账本折叠：顶替事件一条点名新旧两行，其 seq
    # 即三方共用 event_id（映射表第五节，实测后定版）。
    assert ev_receipt["event_id"].startswith("ledger-seq:")
    assert result["complete"] is True
    assert result["reason"] == "readback_verified_only"
