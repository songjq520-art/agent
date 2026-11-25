# tests/test_session_auth_20261125.py（立项 H2：会话认证/可信直达，2026-11-25）
#
# 覆盖口径（施工清单 H2＋契约三）：
#   - 有效凭据走通：执行开关打开后带主人直达会话的提案正常执行，
#     回执 detail 记 session_id；影子模式（开关关）带不带凭据都
#     只出影子回执、不验凭据、不落状态；
#   - 无凭据被拒（no_session）：开关打开后没带凭据的提案一律拒，
#     状态不动；别家权威签的凭据在本店同样 no_session；
#   - 过期被拒（session_expired）：签发时把时钟拨到过去，凭据
#     到手即过期，执行拒；
#   - 越界被拒（out_of_scope）：范围只含 mark_dormant 的凭据提
#     demote 被拒，提 mark_dormant 正常执行；
#   - 敏感动作只认直达：set_protection_tier（保护变更）用非直达
#     会话提被拒（not_owner_direct），主人直达会话正常执行；
#   - 转述签不出直达：issuance_source 为 relayed/third_party 时
#     直达与普通凭据都签不出来；未标来源（None）也签不出直达；
#   - 篡改被识破（session_tampered）：改凭据的 scope / owner_direct
#     / 过期时间任一字段，验凭据即拒；
#   - 撤销接缝（H3）：revoke_session 后凭据被拒（session_revoked），
#     重复撤销不二次生效；
#   - 签发口径：范围超出受控动作全集、空范围、非正有效期都签不出。
#
# 全程离线：嵌入/判官打桩为 None，同 test_controlled_entry 口径。

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from memory.sqlite_store import SQLiteStore


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


# ---------- 有效凭据走通（执行 / 影子两态） ----------

def test_valid_direct_session_executes(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    store.controlled_entry_execution_enabled = True
    sess = _direct(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt["status"] == "executed"
    assert receipt["detail"]["session_id"] == sess.session_id
    assert _confirmed(store, mid) is False


def test_shadow_mode_does_not_require_session(store):
    mid = _add_user(store, "用户在学吉他")
    assert store.controlled_entry_execution_enabled is False
    # 没凭据：影子回执（开关关时不验凭据，H1 口径不变）
    receipt = store.submit_controlled_proposal(_prop(memory_id=mid))
    assert receipt["status"] == "shadow"
    assert receipt["reason"] == "execution_disabled"
    # 带有效凭据：同样影子，凭据不消费、状态不落
    sess = _direct(store)
    receipt2 = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt2["status"] == "shadow"
    assert _confirmed(store, mid) is True


# ---------- 无凭据 / 外来凭据 ----------

def test_no_session_refused_when_enabled(store):
    mid = _add_user(store, "用户喜欢喝茶")
    store.controlled_entry_execution_enabled = True
    receipt = store.submit_controlled_proposal(_prop(memory_id=mid))
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "no_session"
    assert receipt["detail"]["session_reason"] == "no_session"
    assert _confirmed(store, mid) is True


def test_foreign_authority_credential_refused(store, tmp_path):
    mid = _add_user(store, "用户喜欢下棋")
    other = SQLiteStore(memory_dir=str(tmp_path / "other"))
    foreign = other.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=True
    )
    assert foreign is not None
    store.controlled_entry_execution_enabled = True
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=foreign)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "no_session"
    assert _confirmed(store, mid) is True


# ---------- 过期 ----------

def test_expired_session_refused(store):
    mid = _add_user(store, "用户养了一只猫")
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    sess = store._session_authority.issue(
        speaker="owner", scope=store.CONTROLLED_ACTIONS,
        ttl_seconds=3600, owner_direct=True,
        issuance_source="owner", allowed_actions=store.CONTROLLED_ACTIONS,
        now=past,
    )
    assert sess is not None
    store.controlled_entry_execution_enabled = True
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_expired"
    assert _confirmed(store, mid) is True


# ---------- 范围 ----------

def test_out_of_scope_refused_in_scope_executes(store):
    mid = _add_user(store, "用户上周去了一趟超市")
    sess = _plain(store, scope={"mark_dormant"})
    store.controlled_entry_execution_enabled = True
    refused = store.submit_controlled_proposal(
        _prop(action="demote", memory_id=mid, session=sess)
    )
    assert refused["status"] == "refused"
    assert refused["reason"] == "out_of_scope"
    ok = store.submit_controlled_proposal(
        _prop(action="mark_dormant", memory_id=mid, session=sess)
    )
    assert ok["status"] == "executed"


# ---------- 敏感动作只认主人直达 ----------

def test_sensitive_action_requires_owner_direct(store):
    mid = _add_user(store, "用户的生日是五月一号")
    store.controlled_entry_execution_enabled = True
    plain = _plain(store)  # 范围全集，但不是直达
    refused = store.submit_controlled_proposal(
        _prop(
            action="set_protection_tier", memory_id=mid,
            params={"tier": "pinned"}, session=plain,
        )
    )
    assert refused["status"] == "refused"
    assert refused["reason"] == "not_owner_direct"
    direct = _direct(store)
    ok = store.submit_controlled_proposal(
        _prop(
            action="set_protection_tier", memory_id=mid,
            params={"tier": "pinned"}, session=direct,
        )
    )
    assert ok["status"] == "executed"


# ---------- 转述签不出直达 ----------

def test_relayed_source_cannot_issue_direct(store):
    for source in ("relayed", "third_party", "curator"):
        assert store.issue_session(
            speaker="friend", issuance_source=source,
            owner_direct=True,
        ) is None
        # 非本人显式来源连普通凭据也签不出来
        assert store.issue_session(
            speaker="friend", issuance_source=source,
        ) is None
    # 未标来源（None）也签不出直达——新接口不吃老口径的默认放行
    assert store.issue_session(
        speaker="owner", issuance_source=None, owner_direct=True,
    ) is None
    # 主人通道显式签发直达：正常
    assert _direct(store) is not None


# ---------- 篡改 ----------

def test_tampered_credential_refused(store):
    mid = _add_user(store, "用户的工作是程序员")
    store.controlled_entry_execution_enabled = True

    sess = _plain(store, scope={"mark_dormant"})
    sess.scope = frozenset({"mark_dormant", "demote"})  # 偷扩范围
    receipt = store.submit_controlled_proposal(
        _prop(action="demote", memory_id=mid, session=sess)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_tampered"

    sess2 = _plain(store)
    sess2.owner_direct = True  # 偷升直达
    receipt2 = store.submit_controlled_proposal(
        _prop(
            action="set_protection_tier", memory_id=mid,
            params={"tier": "pinned"}, session=sess2,
        )
    )
    assert receipt2["status"] == "refused"
    assert receipt2["reason"] == "session_tampered"

    sess3 = _direct(store)
    sess3.expires_at = sess3.expires_at + timedelta(days=365)  # 偷延寿
    receipt3 = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess3)
    )
    assert receipt3["status"] == "refused"
    assert receipt3["reason"] == "session_tampered"
    assert _confirmed(store, mid) is True


# ---------- 撤销接缝 ----------

def test_revoked_session_refused(store):
    mid = _add_user(store, "用户喜欢爬山")
    sess = _direct(store)
    assert store.revoke_session(sess.session_id) is True
    # 重复撤销不二次生效；撤销不存在的凭据返回 False
    assert store.revoke_session(sess.session_id) is False
    assert store.revoke_session("no-such-session") is False
    store.controlled_entry_execution_enabled = True
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=sess)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "session_revoked"
    assert _confirmed(store, mid) is True


# ---------- 签发口径 ----------

def test_issue_rejects_bad_scope_and_ttl(store):
    # 范围超出受控动作全集（delete 永远不在动作集里）
    assert store.issue_session(
        speaker="owner", issuance_source="owner",
        scope={"delete"},
    ) is None
    # 空范围
    assert store.issue_session(
        speaker="owner", issuance_source="owner", scope=set(),
    ) is None
    # 非正有效期
    assert store.issue_session(
        speaker="owner", issuance_source="owner", ttl_seconds=0,
    ) is None
    assert store.issue_session(
        speaker="owner", issuance_source="owner", ttl_seconds=-5,
    ) is None
