# tests/test_session_auth_fixes_20261125.py（立项 H2 补修轮，2026-11-25）
#
# 针对贵妃第二十五份复查的补修回归：
#   - F1 签发 fail-closed：未标来源（None）任何凭据都签不出
#     （含全范围普通凭据）；store 包装 issue_session 的
#     issuance_source 必传、无默认值，不许裸签；来源与说话人
#     组合钉死（主人通道只给 owner 签、owner 只从主人通道出，
#     伪造组合签不出直达凭据）；
#   - F3 畸形结构化拒绝：issue / validate 对 session_id、
#     scope、时间、digest、ttl 先验形状，畸形只返回拒签 /
#     原因码、绝不抛异常；ttl 必须有限正数；store 的 submit /
#     revoke 两处 validate 调用点异常转结构化回执；
#   - F4 到期边界：now >= expires_at 即失效（到期即失效），
#     等号边界与前一微秒对照；
#   - 撤销并发语义钉死：撤销在 validate 中途落下 → 本次验签
#     即拒；撤销在验签后、落笔前落下 → 落笔前复查阻断执行
#     （submit 与 revoke_controlled_change 两条路径各一条
#     交错回归），状态不落、意图按 aborted 收尾。
#
# 全程离线：嵌入/判官打桩为 None，同 test_session_auth 口径。

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from memory.session_auth import SessionAuthority
from memory.sqlite_store import SQLiteStore

ALL_ACTIONS = {"demote", "mark_dormant", "set_core_drawer",
               "set_protection_tier"}


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


def _intent_for_key(store, key):
    for intent in store._controlled_intents.values():
        if intent.get("idempotency_key") == key:
            return intent
    return None


# ---------- F1：未标来源签不出任何凭据 ----------

def test_issue_none_source_refused_plain_and_direct(store):
    # 裁决 F1 的原场景：source=None＋全范围，旧口径签得出且
    # 能授权 demote/set_core_drawer/mark_dormant——现在拒签。
    assert store.issue_session(
        speaker="owner", issuance_source=None,
        scope=set(store.CONTROLLED_ACTIONS),
    ) is None
    assert store.issue_session(
        speaker="owner", issuance_source=None,
    ) is None
    assert store.issue_session(
        speaker="owner", issuance_source=None, owner_direct=True,
    ) is None
    assert store.issue_session(
        speaker="curator", issuance_source=None,
    ) is None
    # 权威层同口径
    assert store._session_authority.issue(
        speaker="owner", scope=set(store.CONTROLLED_ACTIONS),
        issuance_source=None,
        allowed_actions=set(store.CONTROLLED_ACTIONS),
    ) is None
    # 畸形来源（不可哈希/非 str）同样拒签、不抛异常
    assert store.issue_session(
        speaker="owner", issuance_source=[],
    ) is None
    assert store.issue_session(
        speaker="owner", issuance_source=7,
    ) is None


def test_issue_session_wrapper_requires_explicit_source(store):
    # 包装层收口：issuance_source 必传，裸调用直接 TypeError，
    # 不许靠默认参数签出「owner＋全范围＋无来源」凭据。
    with pytest.raises(TypeError):
        store.issue_session()
    with pytest.raises(TypeError):
        store.issue_session(speaker="owner", owner_direct=True)


def test_issue_forged_source_speaker_combination_refused(store):
    # 伪造组合：自称经主人通道、说话人却是别人——签不出
    # 直达，也签不出普通凭据（第二十五份复查 F2 探针组合）。
    assert store.issue_session(
        speaker="unverified_friend", issuance_source="owner",
        owner_direct=True,
    ) is None
    assert store.issue_session(
        speaker="unverified_friend", issuance_source="owner",
    ) is None
    assert store.issue_session(
        speaker="curator", issuance_source="owner",
    ) is None
    # 系统内部路径不许冒名签 owner 说话人的凭据
    assert store.issue_session(
        speaker="owner", issuance_source="system",
    ) is None
    assert store.issue_session(
        speaker="owner", issuance_source="system",
        owner_direct=True,
    ) is None
    # 系统路径给非主人签直达同样不行
    assert store.issue_session(
        speaker="curator", issuance_source="system",
        owner_direct=True,
    ) is None
    # 正对照：组合对得上的照常签
    assert _direct(store) is not None
    assert _plain(store) is not None


# ---------- F3：畸形一律结构化拒绝、绝不抛异常 ----------

def test_validate_malformed_session_id_no_raise():
    auth, cred = _authority_with_cred()
    cred.session_id = []
    verdict = auth.validate(cred, "demote")
    assert verdict["ok"] is False
    assert verdict["reason"] == "no_session"
    cred.session_id = None
    verdict = auth.validate(cred, "demote")
    assert verdict["ok"] is False
    assert verdict["reason"] == "no_session"


def test_validate_malformed_scope_no_raise():
    auth, cred = _authority_with_cred()
    cred.scope = None
    assert auth.validate(cred, "demote")["reason"] == "session_tampered"
    cred.scope = ["demote"]
    assert auth.validate(cred, "demote")["reason"] == "session_tampered"
    cred.scope = frozenset({7})
    assert auth.validate(cred, "demote")["reason"] == "session_tampered"


def test_validate_malformed_times_no_raise():
    auth, cred = _authority_with_cred()
    cred.issued_at = None
    assert auth.validate(cred, "demote")["reason"] == "session_tampered"
    auth2, cred2 = _authority_with_cred()
    cred2.expires_at = None
    assert auth2.validate(cred2, "demote")["reason"] == "session_tampered"
    auth3, cred3 = _authority_with_cred()
    cred3.expires_at = datetime.now()  # 无时区的朴素时间：签发不产出
    assert auth3.validate(cred3, "demote")["reason"] == "session_tampered"


def test_validate_malformed_digest_no_raise():
    auth, cred = _authority_with_cred()
    cred.digest = 7
    assert auth.validate(cred, "demote")["reason"] == "session_tampered"
    auth2, cred2 = _authority_with_cred()
    cred2.digest = None
    assert auth2.validate(cred2, "demote")["reason"] == "session_tampered"


def test_validate_malformed_action_and_now_no_raise():
    auth, cred = _authority_with_cred()
    verdict = auth.validate(cred, ["demote"])
    assert verdict["ok"] is False
    assert verdict["reason"] == "out_of_scope"
    verdict = auth.validate(cred, "demote", now="tomorrow")
    assert verdict["ok"] is False
    verdict = auth.validate(cred, "demote", now=datetime.now())
    assert verdict["ok"] is False


def test_issue_malformed_ttl_refused():
    auth = SessionAuthority()
    for bad_ttl in (math.nan, math.inf, -math.inf, 0, -1,
                    "3600", True, None, 1e300, 10 ** 400):
        assert auth.issue(
            speaker="owner", scope={"demote"}, ttl_seconds=bad_ttl,
            issuance_source="owner", allowed_actions=ALL_ACTIONS,
        ) is None, bad_ttl
    # 正对照：正常 ttl 照常签
    assert auth.issue(
        speaker="owner", scope={"demote"}, ttl_seconds=60,
        issuance_source="owner", allowed_actions=ALL_ACTIONS,
    ) is not None


def test_issue_malformed_shapes_refused():
    auth = SessionAuthority()
    assert auth.issue(
        speaker="owner", scope=7, issuance_source="owner",
        allowed_actions=ALL_ACTIONS,
    ) is None
    assert auth.issue(
        speaker="owner", scope={"demote"}, issuance_source="owner",
        allowed_actions=7,
    ) is None
    assert auth.issue(
        speaker="owner", scope={"demote"}, issuance_source="owner",
        allowed_actions=ALL_ACTIONS, now="just now",
    ) is None
    assert auth.issue(
        speaker="owner", scope={"demote"}, issuance_source="owner",
        allowed_actions=ALL_ACTIONS, now=datetime.now(),
    ) is None
    assert auth.issue(
        speaker=7, scope={"demote"}, issuance_source="owner",
        allowed_actions=ALL_ACTIONS,
    ) is None


def test_revoke_malformed_session_id_no_raise(store):
    auth, cred = _authority_with_cred()
    assert auth.revoke([]) is False
    assert auth.revoke(None) is False
    assert store.revoke_session([]) is False
    # is_revoked 闸口：形状不对按已撤销处理（fail-closed）
    assert auth.is_revoked([]) is True
    assert auth.is_revoked(cred.session_id) is False
    assert auth.revoke(cred.session_id) is True
    assert auth.is_revoked(cred.session_id) is True


def test_submit_malformed_session_structured_refusal(store):
    mid = _add_user(store, "用户喜欢喝绿茶")
    store.controlled_entry_execution_enabled = True
    sess = _plain(store)
    sess.digest = 7  # compare_digest 旧口径会抛 TypeError
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_tampered"
    sess2 = _direct(store)
    sess2.session_id = []  # 字典查找旧口径会抛 TypeError
    receipt2 = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess2)
    )
    assert receipt2["status"] == "refused"
    assert receipt2["reason"] == "no_session"
    receipt3 = store.submit_controlled_proposal(
        _prop(memory_id=mid, session="not-a-credential")
    )
    assert receipt3["status"] == "refused"
    assert receipt3["reason"] == "no_session"
    assert _confirmed(store, mid) is True


def test_revoke_change_malformed_session_structured_refusal(store):
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
    bad.digest = 7
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


def test_submit_validate_exception_becomes_structured_refusal(
        store, monkeypatch):
    mid = _add_user(store, "用户在学游泳")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(store._session_authority, "validate", _boom)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_tampered"
    assert _confirmed(store, mid) is True


def test_revoke_change_validate_exception_becomes_structured(
        store, monkeypatch):
    mid = _add_user(store, "用户养了一盆绿植")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess,
              expected_version=store.controlled_entry_version(mid))
    )
    assert receipt["status"] == "executed"
    op_id = receipt["detail"]["op_id"]

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(store._session_authority, "validate", _boom)
    refused = store.revoke_controlled_change({
        "target_op_id": op_id,
        "reason": "测试撤销",
        "instruction_source": "owner",
        "idempotency_key": uuid.uuid4().hex,
        "session": sess,
    })
    assert refused["status"] == "refused"
    assert refused["reason"] == "session_tampered"
    assert _confirmed(store, mid) is False


# ---------- F4：到期即失效（等号边界） ----------

def test_expiry_boundary_equal_is_expired():
    auth = SessionAuthority()
    t0 = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)
    cred = auth.issue(
        speaker="owner", scope={"demote"}, ttl_seconds=3600,
        owner_direct=True, issuance_source="owner",
        allowed_actions=ALL_ACTIONS, now=t0,
    )
    assert cred is not None
    expires = t0 + timedelta(seconds=3600)
    assert cred.expires_at == expires
    # 等号：now == expires_at 已失效
    verdict = auth.validate(cred, "demote", now=expires)
    assert verdict["ok"] is False
    assert verdict["reason"] == "session_expired"
    # 前一微秒：仍有效
    verdict = auth.validate(
        cred, "demote", now=expires - timedelta(microseconds=1)
    )
    assert verdict["ok"] is True
    # 后一微秒：失效（旧行为不变）
    verdict = auth.validate(
        cred, "demote", now=expires + timedelta(microseconds=1)
    )
    assert verdict["ok"] is False
    assert verdict["reason"] == "session_expired"


# ---------- 撤销并发：在途验签/未落笔执行必须被阻断 ----------

def test_validate_revoke_midway_returns_revoked(monkeypatch):
    auth, cred = _authority_with_cred()
    orig_digest = auth._digest

    def hooked(payload):
        # 在 validate 的摘要计算点触发撤销（裁决交错探针口径）
        auth.revoke(cred.session_id)
        return orig_digest(payload)

    monkeypatch.setattr(auth, "_digest", hooked)
    verdict = auth.validate(cred, "demote")
    assert verdict["ok"] is False
    assert verdict["reason"] == "session_revoked"


def test_submit_revoke_during_validate_blocks_execution(
        store, monkeypatch):
    mid = _add_user(store, "用户喜欢登山")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)
    auth = store._session_authority
    orig_digest = auth._digest

    def hooked(payload):
        store.revoke_session(sess.session_id)
        return orig_digest(payload)

    monkeypatch.setattr(auth, "_digest", hooked)
    key = uuid.uuid4().hex
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=sess,
              expected_version=store.controlled_entry_version(mid))
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_revoked"
    # 执行被阻断：状态没动，也没有已提交的意图
    assert _confirmed(store, mid) is True
    intent = _intent_for_key(store, key)
    assert intent is None or intent.get("status") == "aborted"


def test_submit_revoke_after_validate_before_write_blocks(
        store, monkeypatch):
    mid = _add_user(store, "用户在学书法")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)
    orig_save = store._controlled_save_intents
    state = {"done": False}

    def hooked_save():
        if not state["done"]:
            state["done"] = True
            # 验签已过、意图落盘这一刻撤销落下——落笔前复查
            # 必须拦住后面的执行。
            store.revoke_session(sess.session_id)
        return orig_save()

    monkeypatch.setattr(store, "_controlled_save_intents", hooked_save)
    key = uuid.uuid4().hex
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=sess,
              expected_version=store.controlled_entry_version(mid))
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_revoked"
    assert _confirmed(store, mid) is True
    intent = _intent_for_key(store, key)
    assert intent is not None
    assert intent.get("status") == "aborted"
    assert intent.get("abort_reason") == "session_revoked"


def test_revoke_change_revoke_after_validate_before_write_blocks(
        store, monkeypatch):
    mid = _add_user(store, "用户喜欢集邮")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess,
              expected_version=store.controlled_entry_version(mid))
    )
    assert receipt["status"] == "executed"
    assert _confirmed(store, mid) is False
    op_id = receipt["detail"]["op_id"]
    sess_b = _plain(store, scope={"demote"})
    orig_save = store._controlled_save_intents
    state = {"done": False}

    def hooked_save():
        if not state["done"]:
            state["done"] = True
            store.revoke_session(sess_b.session_id)
        return orig_save()

    monkeypatch.setattr(store, "_controlled_save_intents", hooked_save)
    rev_key = uuid.uuid4().hex
    refused = store.revoke_controlled_change({
        "target_op_id": op_id,
        "reason": "测试撤销",
        "instruction_source": "owner",
        "idempotency_key": rev_key,
        "session": sess_b,
    })
    assert refused["status"] == "refused"
    assert refused["reason"] == "session_revoked"
    # 补偿被阻断：原变更没有被回滚
    assert _confirmed(store, mid) is False
    rev_intent = _intent_for_key(store, rev_key)
    assert rev_intent is not None
    assert rev_intent.get("status") == "aborted"
    assert rev_intent.get("abort_reason") == "session_revoked"
