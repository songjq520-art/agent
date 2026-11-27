# tests/test_curator_execute_g2_20261127.py（立项 G2：执行接线，2026-11-27）
#
# 覆盖口径（立项清单 G2）：
#   - 接线段只有 demote：评估两段（她侧 assess / review_proposal
#     真函数，经夹具注入）都放行才组装 H1 提案提交；任一段
#     挂起即停、不提交（以 H1 无该幂等键回执为证）；
#   - 执行只认 H1 回执：开关关 → outcome=shadow 且行未动；
#     开关开＋主人直达会话 → outcome=executed、座位真释放；
#     开关开但无会话 / 非本人来源 → refused 且行未动；
#   - 未就绪段继续拒绝：demote 以外动作抛 SegmentNotWired
#     （PermissionError 子类，与她侧原 execute 口径一致）；
#     评估未注入同样拒绝；缺幂等键提交前 ValueError；
#   - 同键重提走 H1 幂等：第二次回执 duplicate=True，不重复执行。
#
# 全程离线：嵌入/语义判官打桩为 None，同 G1 口径。

import sys
import uuid
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "curator_protocol"
sys.path.insert(0, str(_FIXTURE_DIR))

import curator as _her_curator  # noqa: E402
import host_review as _her_review  # noqa: E402

from memory import curator_adapter as _adapter  # noqa: E402
from memory import curator_execute as _exec  # noqa: E402
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


def _assess_fn(fields):
    return _her_curator.assess(_her_curator.Candidate(**fields))


def _review_fn(store):
    def _review(proposal_fields, current):
        return _her_review.review_proposal(
            _her_review.Proposal(**proposal_fields), current,
            protected=lambda meta: _adapter.host_protected(store, meta),
            classify=_adapter.host_classify,
        )
    return _review


def _run(store, mid, **overrides):
    kwargs = {
        "judge_status": "valid",
        "instruction_source": "owner",
        "idempotency_key": "g2-" + uuid.uuid4().hex,
        "assess_fn": _assess_fn,
        "review_fn": _review_fn(store),
    }
    kwargs.update(overrides)
    return _exec.execute_demote(store, _snap_row(store, mid), **kwargs)


def _session(store):
    sess = store.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=True
    )
    assert sess is not None
    return sess


# ---------- 影子段（开关关） ----------

def test_shadow_when_switch_off(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    result = _run(store, mid)
    assert result["outcome"] == "shadow"
    assert result["receipt"]["status"] == "shadow"
    # 行未动：仍占座
    assert store.core_seat_snapshot()["seats_used"] == 1


def test_duplicate_key_not_reexecuted(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    key = "g2-dup-" + uuid.uuid4().hex
    first = _run(store, mid, idempotency_key=key)
    second = _run(store, mid, idempotency_key=key)
    assert first["summary"]["duplicate"] is False
    assert second["summary"]["duplicate"] is True
    # H1 口径：重复回执 status 独立为 duplicate、版本戳沿原
    # 回执；接线层如实报 outcome=duplicate，不冒充二次执行。
    assert second["outcome"] == "duplicate"
    assert second["receipt"]["status"] == "duplicate"
    assert second["receipt"]["version_before"] == first["receipt"]["version_before"]


# ---------- 评估段挂起即停 ----------

def test_held_at_assess_never_submits(store):
    mid = _add_user(store, "用户的工作是程序员")
    assert store.set_protection_tier(mid, "pinned")["ok"] is True
    key = "g2-held-" + uuid.uuid4().hex
    result = _run(store, mid, idempotency_key=key)
    assert result["outcome"] == "held" and result["stage"] == "assess"
    assert result["assess"] == "hold_protected_or_unknown"
    assert store.controlled_entry_receipt(key) is None


def test_held_at_review_never_submits(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    store.pending_questions.add(
        "importance", "这条要进核心吗", ["进核心", "当一次性经历", "不存"],
        "测试待问", memory_id=mid,
    )
    key = "g2-held2-" + uuid.uuid4().hex
    result = _run(store, mid, idempotency_key=key)
    assert result["outcome"] == "held" and result["stage"] == "review"
    assert result["review"] == "hold_pending_or_unknown"
    assert store.controlled_entry_receipt(key) is None


def test_held_when_judge_not_valid(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    result = _run(store, mid, judge_status="unreachable")
    assert result["outcome"] == "held" and result["stage"] == "assess"
    assert result["assess"] == "hold_judge_unreachable"


# ---------- 真执行段（开关开） ----------

def test_executed_with_owner_session(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    store.controlled_entry_execution_enabled = True
    result = _run(store, mid, session=_session(store))
    assert result["outcome"] == "executed", result["receipt"]
    assert result["summary"]["executed"] is True
    # 真降级：座位释放、原文不动
    assert store.core_seat_snapshot()["seats_used"] == 0
    row = _snap_row(store, mid)
    assert row["text"] == "用户每天定投五块钱买基金"


def test_refused_without_session_when_switch_on(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    store.controlled_entry_execution_enabled = True
    result = _run(store, mid)  # 不给会话凭据
    assert result["outcome"] == "refused"
    assert store.core_seat_snapshot()["seats_used"] == 1


def test_refused_for_non_owner_source(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    store.controlled_entry_execution_enabled = True
    result = _run(
        store, mid, instruction_source="third_party_relay",
        session=_session(store),
    )
    assert result["outcome"] == "refused"
    assert store.core_seat_snapshot()["seats_used"] == 1


# ---------- 未接线段继续拒绝 ----------

def test_unwired_actions_still_refused(store):
    for action in ("set_protection_tier", "set_core_drawer",
                   "mark_dormant", "delete"):
        with pytest.raises(PermissionError):
            _exec.execute(store, action, None)


def test_unwired_without_assessment_injection(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    with pytest.raises(PermissionError):
        _exec.execute_demote(
            store, _snap_row(store, mid), judge_status="valid",
            idempotency_key="g2-noassess",
        )


def test_missing_idempotency_key_rejected_before_submit(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    with pytest.raises(ValueError):
        _exec.execute_demote(
            store, _snap_row(store, mid), judge_status="valid",
            assess_fn=_assess_fn, review_fn=_review_fn(store),
        )
