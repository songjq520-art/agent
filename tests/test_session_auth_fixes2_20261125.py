# H2 第二轮小补修回归（贵妃第二十六份复查 F3 残留，十一裁决成立）：
#
#   残留 A：SessionCredential 用 __slots__，槽位可被 delattr
#     摘除。补修一轮的形状前验直读字段——validate 首步直读
#     session_id、_credential_shape_ok 直读其余六字段，都在
#     摘要 try 之外，缺字段抛 AttributeError（逐字段 7/7＋
#     未初始化实例 8/8）。本轮改安全取值：缺字段一律判畸形，
#     validate「绝不抛异常」的自家契约全路径成立。
#   残留 B：_is_aware_datetime 直调 tzinfo.utcoffset 不接
#     异常，异常时区在 issue(now) / validate(now) /
#     issued_at / expires_at 四路抛 ValueError。本轮接住
#     异常，异常时区一律判不合格，四路一次收口。
#   外加 store 两条入口（submit / revoke）对这两类畸形的
#   回执仍结构化（refused＋明确 reason），状态不被污染。
#
# F1 不在本轮范围（裁决：实现无缺口，只有信件表述过头）。

import uuid
from datetime import datetime, timezone, tzinfo

import pytest

from memory.session_auth import (
    REASON_NO_SESSION,
    REASON_SESSION_TAMPERED,
    SessionAuthority,
    SessionCredential,
)
from memory.sqlite_store import SQLiteStore

ALL_ACTIONS = {"demote", "mark_dormant", "set_core_drawer",
               "set_protection_tier"}

CREDENTIAL_FIELDS = (
    "session_id", "speaker", "scope", "issued_at",
    "expires_at", "owner_direct", "digest",
)


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


def _confirmed(store, mid):
    r = store._conn.execute(
        "SELECT user_confirmed FROM memories WHERE id=?", (mid,)
    ).fetchone()
    assert r is not None, mid
    return bool(r[0])


def _prop(action="demote", memory_id=None, key=None, **extra):
    p = {
        "action": action,
        "memory_id": memory_id,
        "reason": "测试提案",
        "instruction_source": "owner",
        "idempotency_key": key or uuid.uuid4().hex,
    }
    p.update(extra)
    return p


def _direct(store, **kwargs):
    kwargs.setdefault("speaker", "owner")
    kwargs.setdefault("issuance_source", "owner")
    kwargs.setdefault("owner_direct", True)
    sess = store.issue_session(**kwargs)
    assert sess is not None
    return sess


def _plain(store, **kwargs):
    kwargs.setdefault("speaker", "curator")
    kwargs.setdefault("issuance_source", "system")
    sess = store.issue_session(**kwargs)
    assert sess is not None
    return sess


def _authority_with_cred():
    auth = SessionAuthority()
    cred = auth.issue(
        speaker="curator", scope={"demote", "mark_dormant"},
        issuance_source="system", allowed_actions=ALL_ACTIONS,
    )
    assert cred is not None
    return auth, cred


class _ExplodingTZ(tzinfo):
    """utcoffset 一调用就抛 ValueError 的畸形时区。"""

    def utcoffset(self, dt):
        raise ValueError("exploding utcoffset")

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return "BOOM"


def _bad_tz_datetime():
    return datetime(2026, 11, 25, 12, 0, tzinfo=_ExplodingTZ())


# ---------- 残留 A：逐字段缺失，validate 绝不抛 ----------

@pytest.mark.parametrize("field", CREDENTIAL_FIELDS)
def test_validate_missing_field_no_raise(field):
    auth, cred = _authority_with_cred()
    delattr(cred, field)
    verdict = auth.validate(cred, "demote")
    assert verdict["ok"] is False
    if field == "session_id":
        # 槽位没了连登记都查不了：按「不是本权威签的」拒绝
        assert verdict["reason"] == REASON_NO_SESSION
        assert verdict["session_id"] is None
    else:
        # 登记还在、形状前验判畸形：session_tampered
        assert verdict["reason"] == REASON_SESSION_TAMPERED


def test_validate_uninitialized_instance_no_raise():
    auth, _cred = _authority_with_cred()
    bare = SessionCredential.__new__(SessionCredential)
    verdict = auth.validate(bare, "demote")
    assert verdict["ok"] is False
    assert verdict["reason"] == REASON_NO_SESSION
    assert verdict["session_id"] is None


def test_validate_none_and_wrong_type_no_raise():
    auth, _cred = _authority_with_cred()
    for bogus in (None, "not-a-credential", 42, object()):
        verdict = auth.validate(bogus, "demote")
        assert verdict["ok"] is False
        assert verdict["reason"] == REASON_NO_SESSION


# ---------- 残留 B：异常时区四路，绝不抛 ----------

def test_issue_bad_tz_now_refused():
    auth = SessionAuthority()
    cred = auth.issue(
        speaker="curator", scope={"demote"},
        issuance_source="system", allowed_actions=ALL_ACTIONS,
        now=_bad_tz_datetime(),
    )
    assert cred is None


def test_validate_bad_tz_now_no_raise():
    auth, cred = _authority_with_cred()
    verdict = auth.validate(cred, "demote", now=_bad_tz_datetime())
    assert verdict["ok"] is False
    assert verdict["reason"] == REASON_SESSION_TAMPERED


def test_validate_bad_tz_issued_at_no_raise():
    auth, cred = _authority_with_cred()
    cred.issued_at = _bad_tz_datetime()
    verdict = auth.validate(cred, "demote")
    assert verdict["ok"] is False
    assert verdict["reason"] == REASON_SESSION_TAMPERED


def test_validate_bad_tz_expires_at_no_raise():
    auth, cred = _authority_with_cred()
    cred.expires_at = _bad_tz_datetime()
    verdict = auth.validate(cred, "demote")
    assert verdict["ok"] is False
    assert verdict["reason"] == REASON_SESSION_TAMPERED


# ---------- store 两条入口：两类畸形回执仍结构化 ----------

def test_submit_missing_field_session_structured_refusal(store):
    mid = _add_user(store, "用户喜欢喝绿茶")
    store.controlled_entry_execution_enabled = True
    sess = _plain(store)
    delattr(sess, "digest")
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_tampered"
    sess2 = _direct(store)
    delattr(sess2, "session_id")
    receipt2 = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess2)
    )
    assert receipt2["status"] == "refused"
    assert receipt2["reason"] == "no_session"
    assert _confirmed(store, mid) is True


def test_submit_bad_tz_session_structured_refusal(store):
    mid = _add_user(store, "用户周末去爬山")
    store.controlled_entry_execution_enabled = True
    sess = _plain(store)
    sess.expires_at = _bad_tz_datetime()
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_tampered"
    assert _confirmed(store, mid) is True


def test_revoke_change_missing_field_session_structured_refusal(store):
    mid = _add_user(store, "用户周末去钓鱼")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess,
              expected_version=store.controlled_entry_version(mid))
    )
    assert receipt["status"] == "executed"
    op_id = receipt["detail"]["op_id"]
    bad = _plain(store)
    delattr(bad, "digest")
    refused = store.revoke_controlled_change({
        "target_op_id": op_id,
        "reason": "测试撤销",
        "instruction_source": "owner",
        "idempotency_key": uuid.uuid4().hex,
        "session": bad,
    })
    assert refused["status"] == "refused"
    assert refused["reason"] == "session_tampered"
    # 补偿没落：仍是降级后的状态
    assert _confirmed(store, mid) is False


def test_revoke_change_bad_tz_session_structured_refusal(store):
    mid = _add_user(store, "用户在学游泳")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess,
              expected_version=store.controlled_entry_version(mid))
    )
    assert receipt["status"] == "executed"
    op_id = receipt["detail"]["op_id"]
    bad = _plain(store)
    bad.issued_at = _bad_tz_datetime()
    refused = store.revoke_controlled_change({
        "target_op_id": op_id,
        "reason": "测试撤销",
        "instruction_source": "owner",
        "idempotency_key": uuid.uuid4().hex,
        "session": bad,
    })
    assert refused["status"] == "refused"
    assert refused["reason"] == "session_tampered"
    assert _confirmed(store, mid) is False
