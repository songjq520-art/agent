# tests/test_revocation_20261125.py（立项 H3：撤销接口，2026-11-25）
#
# 覆盖口径（施工清单 H3＋契约三）：
#   - 正常撤销恢复原状：demote / set_protection_tier /
#     set_core_drawer / mark_dormant 四动作执行→撤销往返后，
#     字段、占座与变更前逐项相等；
#   - 门禁：非 BOOS 本人来源拒绝（not_owner_instruction）＋
#     攒知情待问；执行开关打开时无凭据拒绝、敏感动作（保护
#     变更）的撤销无主人直达凭据拒绝（not_owner_direct）；
#   - 幂等：同撤销键重提回 duplicate、不二次补偿；原 op 被
#     撤后换个键再撤明确拒绝（already_revoked）；
#   - 定位失败如实拒绝：op_id 对不上（op_not_found）、缺定位
#     （missing_target）、缺撤销键（missing_idempotency_key）；
#     没真执行的提案（影子）没有可撤之物（op_not_found）；
#   - 栈纪律：目标字段被后来的变更改过 → state_drifted 拒绝；
#   - 当前行已成核心保护行 → core_protected 拒绝，不许借
#     撤销绕护栏；
#   - 开关关时撤销只出影子回执、不落状态；
#   - 可追溯：账本里原 op 与补偿 op 两笔事件都在、审计日志
#     同带 revokes_op_id；重启新实例后撤销登记与已撤销标记
#     都还在；
#   - 撤销不碰原文与无关字段（逐字比对＋meta 其余键比对）。
#
# 全程离线：嵌入/判官打桩为 None，同 test_controlled_entry 口径。

import json
import uuid

import pytest

from memory import ledger as _ledger
from memory import pending_questions as _pq
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


def _row(store, mid):
    r = store._conn.execute(
        "SELECT text, metadata_json, user_confirmed FROM memories"
        " WHERE id=?",
        (mid,),
    ).fetchone()
    assert r is not None, mid
    return {
        "text": r[0],
        "meta": json.loads(r[1] or "{}"),
        "user_confirmed": bool(r[2]),
    }


def _session(store, direct=True):
    sess = store.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=direct
    )
    assert sess is not None
    return sess


def _execute(store, action, mid, key=None, **params):
    """开开关、带主人直达会话与版本戳执行一笔受控提案。"""
    store.controlled_entry_execution_enabled = True
    prop = {
        "action": action,
        "memory_id": mid,
        "reason": "测试执行",
        "instruction_source": "owner",
        "idempotency_key": key or uuid.uuid4().hex,
        "expected_version": store.controlled_entry_version(mid),
        "session": _session(store),
    }
    if params:
        prop["params"] = params
    receipt = store.submit_controlled_proposal(prop)
    assert receipt["status"] == "executed", receipt
    return receipt


def _revoke_req(op_id=None, target_key=None, source="owner",
                rev_key=None, session=None):
    req = {
        "reason": "测试撤销",
        "instruction_source": source,
        "idempotency_key": rev_key or uuid.uuid4().hex,
    }
    if op_id is not None:
        req["target_op_id"] = op_id
    if target_key is not None:
        req["target_idempotency_key"] = target_key
    if session is not None:
        req["session"] = session
    return req


def _blocked_questions(store):
    return [
        q for q in store.list_pending_questions()
        if q.get("kind") == _pq.QUESTION_KIND_AUTH_CLAIM
        and q.get("auth_action") == "controlled_entry_blocked"
        and q.get("status") != _pq.QUESTION_STATUS_ANSWERED
    ]


def _trace_events(store):
    path = store._controlled_log_path
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ledger_events(store):
    events, _info = _ledger.read_events(store.ledger_path)
    return events


# ---------- 正常撤销：四动作往返恢复原状 ----------

def test_revoke_demote_restores_membership(store):
    mid = _add_user(store, "BOOS 每天定投五块钱买纳指")
    seats0 = store.core_seat_snapshot()["seats_used"]
    assert _row(store, mid)["user_confirmed"] is True
    receipt = _execute(store, "demote", mid)
    assert _row(store, mid)["user_confirmed"] is False
    assert store.core_seat_snapshot()["seats_used"] == seats0 - 1

    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"],
                    session=_session(store))
    )
    assert rev["status"] == "revoked" and rev["ok"] is True, rev
    assert rev["revokes_op_id"] == receipt["detail"]["op_id"]
    assert rev["revocation_op_id"]
    row = _row(store, mid)
    assert row["user_confirmed"] is True
    assert store.core_seat_snapshot()["seats_used"] == seats0
    # 字段口径恢复到位：补偿后的行戳与原执行前的行戳一致。
    assert rev["version_after"] == receipt["version_before"]


def test_revoke_protection_tier_and_drawer(store):
    mid = _add_user(store, "BOOS 住在盐湖城朋友家")
    r1 = _execute(store, "set_protection_tier", mid, tier="pinned")
    assert _row(store, mid)["meta"].get("protection_tier") == "pinned"
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=r1["detail"]["op_id"],
                    session=_session(store))
    )
    assert rev["status"] == "revoked" and rev["ok"] is True, rev
    assert _row(store, mid)["meta"].get("protection_tier") is None

    r2 = _execute(store, "set_core_drawer", mid, drawer="person")
    assert _row(store, mid)["meta"].get("core_drawer") == "person"
    # 这一笔按原提案的幂等键定位撤销。
    rev2 = store.revoke_controlled_change(
        _revoke_req(target_key=r2["idempotency_key"],
                    session=_session(store))
    )
    assert rev2["status"] == "revoked" and rev2["ok"] is True, rev2
    assert _row(store, mid)["meta"].get("core_drawer") is None


def test_revoke_mark_dormant_restores_active(store):
    mid = _add_user(store, "BOOS 上周去了一趟洛杉矶")
    receipt = _execute(store, "mark_dormant", mid)
    assert _row(store, mid)["meta"].get("decay_state") == "dormant"
    state = _ledger.resolve_state(_ledger_events(store))
    assert (state.get("key_states") or {}).get(mid) == "dormant"

    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"],
                    session=_session(store))
    )
    assert rev["status"] == "revoked" and rev["ok"] is True, rev
    meta = _row(store, mid)["meta"]
    assert meta.get("decay_state") != "dormant"
    state = _ledger.resolve_state(_ledger_events(store))
    assert (state.get("key_states") or {}).get(mid) != "dormant"


# ---------- 门禁 ----------

def test_revoke_non_owner_refused(store):
    mid = _add_user(store, "BOOS 的笔记本是华硕的")
    receipt = _execute(store, "demote", mid)
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"],
                    source="curator", session=_session(store))
    )
    assert rev["status"] == "refused", rev
    assert rev["reason"] == "not_owner_instruction"
    assert _row(store, mid)["user_confirmed"] is False  # 没被撤
    assert _blocked_questions(store)  # 知情待问攒上了


def test_revoke_session_gates(store):
    mid = _add_user(store, "BOOS 在美国出差用现金")
    # 敏感动作（保护变更）的撤销：无凭据拒、非直达凭据拒。
    r1 = _execute(store, "set_protection_tier", mid, tier="pinned")
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=r1["detail"]["op_id"])
    )
    assert rev["status"] == "refused" and rev["reason"] == "no_session"
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=r1["detail"]["op_id"],
                    session=_session(store, direct=False))
    )
    assert rev["status"] == "refused", rev
    assert rev["reason"] == "not_owner_direct"
    assert _row(store, mid)["meta"].get("protection_tier") == "pinned"
    # 非敏感动作（降级）的撤销：普通凭据在范围内即可
    # （另用一条记忆——钉死行本身不许再被降级）。
    mid2 = _add_user(store, "BOOS 用微信联系朋友")
    r2 = _execute(store, "demote", mid2)
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=r2["detail"]["op_id"],
                    session=_session(store, direct=False))
    )
    assert rev["status"] == "revoked" and rev["ok"] is True, rev


def test_revoke_core_protected_row_refused(store):
    mid = _add_user(store, "BOOS 的姐姐在西安")
    receipt = _execute(store, "demote", mid)
    # 宿主侧后来把这行标成规则（核心保护行）——测试里直改
    # meta 模拟宿主路的变更，撤销路不许绕护栏碰它。
    row = _row(store, mid)
    meta = dict(row["meta"])
    meta["is_rule"] = True
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (json.dumps(meta, ensure_ascii=False), mid),
    )
    store._conn.commit()
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"],
                    session=_session(store))
    )
    assert rev["status"] == "refused", rev
    assert rev["reason"] == "core_protected"
    assert _row(store, mid)["user_confirmed"] is False


def test_revoke_shadow_when_switch_off(store):
    mid = _add_user(store, "BOOS 每天睡白天")
    receipt = _execute(store, "demote", mid)
    store.controlled_entry_execution_enabled = False
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"])
    )
    assert rev["status"] == "shadow", rev
    assert rev["reason"] == "execution_disabled"
    assert _row(store, mid)["user_confirmed"] is False  # 没落状态


# ---------- 幂等与定位 ----------

def test_revoke_duplicate_and_already_revoked(store):
    mid = _add_user(store, "BOOS 有一台安卓手机")
    receipt = _execute(store, "demote", mid)
    op_id = receipt["detail"]["op_id"]
    req = _revoke_req(op_id=op_id, rev_key="rev-key-1",
                      session=_session(store))
    rev = store.revoke_controlled_change(req)
    assert rev["status"] == "revoked" and rev["ok"] is True, rev
    # 同键重提：duplicate，不二次补偿（账本里补偿事件只有一笔）。
    dup = store.revoke_controlled_change(req)
    assert dup["status"] == "duplicate" and dup["duplicate"] is True
    rev_events = [
        e for e in _ledger_events(store)
        if e.get("op_id") == rev["revocation_op_id"]
    ]
    assert len(rev_events) == 1
    # 换个键再撤同一笔：明确拒绝。
    rev2 = store.revoke_controlled_change(
        _revoke_req(op_id=op_id, session=_session(store))
    )
    assert rev2["status"] == "refused", rev2
    assert rev2["reason"] == "already_revoked"


def test_revoke_locator_failures(store):
    mid = _add_user(store, "BOOS 的朋友在盐湖城")
    rev = store.revoke_controlled_change(
        _revoke_req(op_id="op-does-not-exist",
                    session=_session(store))
    )
    assert rev["status"] == "refused" and rev["reason"] == "op_not_found"
    rev = store.revoke_controlled_change(
        {"instruction_source": "owner",
         "idempotency_key": uuid.uuid4().hex}
    )
    assert rev["reason"] == "missing_target"
    rev = store.revoke_controlled_change(
        {"target_op_id": "op-x", "instruction_source": "owner"}
    )
    assert rev["reason"] == "missing_idempotency_key"
    # 只出过影子回执的提案没真执行，没有可撤之物。
    prop = {
        "action": "demote",
        "memory_id": mid,
        "reason": "影子提案",
        "instruction_source": "owner",
        "idempotency_key": "shadow-key-1",
        "expected_version": store.controlled_entry_version(mid),
    }
    shadow = store.submit_controlled_proposal(prop)
    assert shadow["status"] == "shadow"
    rev = store.revoke_controlled_change(
        _revoke_req(target_key="shadow-key-1",
                    session=_session(store))
    )
    assert rev["status"] == "refused" and rev["reason"] == "op_not_found"


def test_revoke_state_drifted_refused(store):
    mid = _add_user(store, "BOOS 喜欢喝茶")
    receipt = _execute(store, "demote", mid)
    # 后来宿主路又把这条确认了回来（字段被别人动过）——
    # 栈纪律：不许踩掉后来的变更。
    store.confirm_memory(mid)
    assert _row(store, mid)["user_confirmed"] is True
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"],
                    session=_session(store))
    )
    assert rev["status"] == "refused", rev
    assert rev["reason"] == "state_drifted"
    assert _row(store, mid)["user_confirmed"] is True


# ---------- 可追溯与不碰无关 ----------

def test_revocation_traceable_in_ledger_and_audit(store):
    mid = _add_user(store, "BOOS 的邮箱是谷歌的")
    receipt = _execute(store, "set_core_drawer", mid, drawer="person")
    op_id = receipt["detail"]["op_id"]
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=op_id, session=_session(store))
    )
    assert rev["status"] == "revoked"
    events = _ledger_events(store)
    orig = [e for e in events if e.get("op_id") == op_id]
    comp = [e for e in events
            if e.get("op_id") == rev["revocation_op_id"]]
    assert len(orig) == 1 and len(comp) == 1
    traces = [
        t for t in _trace_events(store) if t.get("revocation")
    ]
    assert traces and traces[-1].get("revokes_op_id") == op_id
    assert traces[-1].get("op_id") == rev["revocation_op_id"]
    # 原 op 的回执与账本事件不删不改，还查得到。
    assert store.controlled_entry_receipt(
        receipt["idempotency_key"]
    ) is not None


def test_revoke_does_not_touch_text_or_unrelated_fields(store):
    mid = _add_user(store, "BOOS 的电脑是 i5 处理器")
    before = _row(store, mid)
    receipt = _execute(store, "set_protection_tier", mid,
                       tier="temp_pinned")
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"],
                    session=_session(store))
    )
    assert rev["status"] == "revoked" and rev["ok"] is True, rev
    after = _row(store, mid)
    assert after["text"] == before["text"]
    strip = ("protection_tier",)
    b = {k: v for k, v in before["meta"].items() if k not in strip}
    a = {k: v for k, v in after["meta"].items() if k not in strip}
    assert a == b


def test_revocation_survives_restart(store, tmp_path):
    mid = _add_user(store, "BOOS 夜班作息")
    receipt = _execute(store, "demote", mid)
    req = _revoke_req(op_id=receipt["detail"]["op_id"],
                      rev_key="rev-restart-1",
                      session=_session(store))
    rev = store.revoke_controlled_change(req)
    assert rev["status"] == "revoked"

    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    got = store2.controlled_revocation_receipt("rev-restart-1")
    assert got is not None and got["status"] == "revoked"
    # 新实例里原 op 已带 revoked_by 标记，换键再撤仍被拒。
    store2.controlled_entry_execution_enabled = True
    sess2 = store2.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=True
    )
    rev2 = store2.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"], session=sess2)
    )
    assert rev2["status"] == "refused", rev2
    assert rev2["reason"] == "already_revoked"
