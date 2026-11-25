# tests/test_controlled_entry_20261125.py（立项 H1：统一受控入口，2026-11-25）
#
# 覆盖口径（施工清单 H1＋契约三）：
#   - 正常提案走通：降级经正门执行，只改成员状态（user_confirmed
#     落 False）、不删原文、不清保护标记，占座释放；
#   - 验来源：非 BOOS 本人来源（curator/third_party）一律拒绝
#     （not_owner_instruction）＋攒知情待问，状态不动；答知情待问
#     只是确认知晓，永不放行被拦提案；
#   - 验保护：核心保护行（is_rule）与钉死行降级一律拒绝
#     （core_protected）＋攒知情待问；
#   - 幂等：同一幂等键重复提交、重启新实例重放，都只生效一次，
#     回执标 duplicate；
#   - 未知动作（含 delete——删除永不走这道门）拒绝；缺幂等键拒绝；
#   - 执行开关默认关：只出影子回执（shadow），不落状态；
#   - 验版本：expected_version 对不上拒（version_stale），对上才执行；
#   - 其余动作走既有路径：set_protection_tier / set_core_drawer /
#     mark_dormant（DORMANT 账本事件＋decay_state）；
#   - 回执形状在 executed/refused/shadow/unknown 各路下键集合恒定。
#
# 全程离线：嵌入/判官打桩为 None，同 test_tx_recovery_fixes 口径。
#
# H2 口径更新（2026-11-25）：执行开关打开后入口强制验会话凭据，
# 本文件的执行类用例改带 issue_session 签发的主人直达会话
# （_session 助手）；原断言不变，凭据本身的验收在
# test_session_auth_20261125.py。

import json
import uuid

import pytest

from memory import core_cap as _core_cap
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


def _prop(action="demote", memory_id=None, source="owner", key=None, **extra):
    p = {
        "action": action,
        "memory_id": memory_id,
        "reason": "测试提案",
        "instruction_source": source,
        "idempotency_key": key or uuid.uuid4().hex,
    }
    p.update(extra)
    return p


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


def _enable(store):
    store.controlled_entry_execution_enabled = True
    return store


def _session(store):
    sess = store.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=True
    )
    assert sess is not None
    return sess


# ---------- 正常提案走通 ----------

def test_demote_executes_and_frees_seat(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    assert store.core_seat_snapshot()["seats_used"] == 1
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=_session(store))
    )
    assert receipt["ok"] is True and receipt["status"] == "executed"
    assert receipt["duplicate"] is False
    assert receipt["reason"] == ""
    row = _row(store, mid)
    assert row["user_confirmed"] is False
    assert row["meta"]["user_confirmed"] is False
    assert _core_cap.classify(row["meta"])["member"] is False
    assert store.core_seat_snapshot()["seats_used"] == 0
    # 只改成员状态：原文不动
    assert row["text"] == "用户每天定投五块钱买基金"
    assert receipt["version_before"]
    assert receipt["version_after"] != receipt["version_before"]
    assert receipt["detail"]["invariants_ok"] is True


def test_set_protection_tier_via_entry(store):
    mid = _add_user(store, "用户的生日是五月一号")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(
            action="set_protection_tier", memory_id=mid,
            params={"tier": "pinned"}, session=_session(store),
        )
    )
    assert receipt["status"] == "executed"
    assert _core_cap.normalize_protection_tier(
        _row(store, mid)["meta"]
    ) == "pinned"
    events, _info = _ledger.read_events(store.ledger_path)
    assert any(
        e.get("type") == "promote" and e.get("memory_id") == mid
        for e in events
    )


def test_set_core_drawer_via_entry(store):
    mid = _add_user(store, "用户每月工资一号到账")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(
            action="set_core_drawer", memory_id=mid,
            params={"drawer": "money_decisions"}, session=_session(store),
        )
    )
    assert receipt["status"] == "executed"
    assert _core_cap.classify(
        _row(store, mid)["meta"]
    )["drawer"] == "money_decisions"


def test_mark_dormant_via_entry(store):
    mid = _add_user(store, "用户上周去了一趟超市")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(action="mark_dormant", memory_id=mid,
              session=_session(store))
    )
    assert receipt["status"] == "executed"
    assert _row(store, mid)["meta"]["decay_state"] == "dormant"
    events, _info = _ledger.read_events(store.ledger_path)
    assert any(
        e.get("type") == "dormant" and e.get("memory_id") == mid
        for e in events
    )


# ---------- 验来源 ----------

def test_non_owner_source_refused_with_question(store):
    mid = _add_user(store, "用户喜欢喝茶")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, source="curator")
    )
    assert receipt["ok"] is False and receipt["status"] == "refused"
    assert receipt["reason"] == "not_owner_instruction"
    assert receipt["question_id"]
    qs = _blocked_questions(store)
    assert len(qs) == 1 and qs[0]["target_memory_id"] == mid
    assert qs[0]["attempted_action"] == "demote"
    # 状态不动
    assert _core_cap.classify(_row(store, mid)["meta"])["member"] is True


def test_blocked_question_answer_is_ack_only(store):
    mid = _add_user(store, "用户喜欢下棋")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, source="third_party")
    )
    out = store.answer_question(receipt["question_id"], "知道了")
    assert out["ok"] is True
    assert out["applied"] == "attempt_acknowledged"
    # 答知情待问不等于放行：仍是核心成员、未被降级
    assert _core_cap.classify(_row(store, mid)["meta"])["member"] is True
    assert _row(store, mid)["user_confirmed"] is True


# ---------- 验保护 ----------

def test_core_protected_row_refused_with_question(store):
    mid = _add_user(store, "回答问题要先核实再下结论", is_rule=True)
    assert store._is_core_protected_row(_row(store, mid)["meta"]) is True
    _enable(store)
    receipt = store.submit_controlled_proposal(_prop(memory_id=mid))
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "core_protected"
    assert receipt["question_id"]
    assert len(_blocked_questions(store)) == 1
    assert _row(store, mid)["user_confirmed"] is True


def test_pinned_row_demote_refused(store):
    mid = _add_user(store, "用户的工作是程序员")
    assert store.set_protection_tier(mid, "pinned")["ok"] is True
    _enable(store)
    receipt = store.submit_controlled_proposal(_prop(memory_id=mid))
    assert receipt["reason"] == "core_protected"
    # 降级被拒且保护标记原样不清
    assert _core_cap.normalize_protection_tier(
        _row(store, mid)["meta"]
    ) == "pinned"
    assert _row(store, mid)["user_confirmed"] is True


# ---------- 幂等 ----------

def test_duplicate_key_no_reexecution(store, tmp_path):
    mid = _add_user(store, "用户养了一只猫")
    _enable(store)
    key = uuid.uuid4().hex
    first = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store))
    )
    assert first["status"] == "executed"
    second = store.submit_controlled_proposal(_prop(memory_id=mid, key=key))
    assert second["duplicate"] is True and second["status"] == "duplicate"
    assert second["ok"] is True  # 原回执副本：当时是执行成功的
    executed = [
        e for e in _trace_events(store)
        if e["event"] == "executed" and e["idempotency_key"] == key
    ]
    assert len(executed) == 1
    # 重启重放：同目录新实例、同键，仍只回 duplicate、不再执行
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    third = store2.submit_controlled_proposal(_prop(memory_id=mid, key=key))
    assert third["duplicate"] is True and third["status"] == "duplicate"


# ---------- 形状与开关 ----------

def test_unknown_action_refused(store):
    mid = _add_user(store, "用户喜欢爬山")
    _enable(store)
    for bad in ("delete", "explode"):
        receipt = store.submit_controlled_proposal(
            _prop(action=bad, memory_id=mid)
        )
        assert receipt["status"] == "refused"
        assert receipt["reason"] == "unknown_action"
    assert _blocked_questions(store) == []
    assert _row(store, mid)["user_confirmed"] is True


def test_missing_idempotency_key_refused(store):
    mid = _add_user(store, "用户喜欢游泳")
    p = _prop(memory_id=mid)
    p["idempotency_key"] = None
    receipt = store.submit_controlled_proposal(p)
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "missing_idempotency_key"


def test_not_found_refused(store):
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id="no-such-id")
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "not_found"


def test_shadow_mode_default_no_execution(store):
    mid = _add_user(store, "用户在学吉他")
    assert store.controlled_entry_execution_enabled is False
    receipt = store.submit_controlled_proposal(_prop(memory_id=mid))
    assert receipt["status"] == "shadow"
    assert receipt["reason"] == "execution_disabled"
    assert receipt["ok"] is False
    # 影子只出回执：状态不落
    assert _row(store, mid)["user_confirmed"] is True
    assert _core_cap.classify(_row(store, mid)["meta"])["member"] is True


# ---------- 验版本 ----------

def test_version_stale_refused_and_match_executes(store):
    mid = _add_user(store, "用户计划明年去旅行")
    _enable(store)
    stale = store.submit_controlled_proposal(
        _prop(memory_id=mid, expected_version="0" * 64)
    )
    assert stale["status"] == "refused"
    assert stale["reason"] == "version_stale"
    assert _row(store, mid)["user_confirmed"] is True
    stamp = store.controlled_entry_version(mid)
    assert stamp and stamp != "0" * 64
    ok = store.submit_controlled_proposal(
        _prop(memory_id=mid, expected_version=stamp,
              session=_session(store))
    )
    assert ok["status"] == "executed"
    assert ok["version_before"] == stamp


# ---------- 执行面边界 ----------

def test_demote_non_member_refused(store):
    saved = store.add_memory_text("朋友说他喜欢钓鱼", source="inferred")
    assert saved.get("event") == "ADD", saved
    mid = saved["id"]
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=_session(store))
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "not_core_member"


def test_receipt_shape_stable(store):
    mid = _add_user(store, "用户喜欢喝咖啡")
    receipts = [
        store.submit_controlled_proposal(_prop(memory_id=mid)),  # shadow
    ]
    _enable(store)
    receipts.append(
        store.submit_controlled_proposal(
            _prop(memory_id=mid, source="curator")  # refused 来源
        )
    )
    receipts.append(
        store.submit_controlled_proposal(
            _prop(memory_id=mid, session=_session(store))
        )  # executed
    )
    receipts.append(
        store.submit_controlled_proposal(
            _prop(action="nope", memory_id=mid)  # refused 未知动作
        )
    )
    expected = {
        "ok", "status", "reason", "action", "memory_id",
        "idempotency_key", "duplicate", "question_id", "trace_index",
        "version_before", "version_after", "detail", "ts",
    }
    for r in receipts:
        assert set(r.keys()) == expected, r
        assert isinstance(r["detail"], dict)
        assert isinstance(r["trace_index"], int)
