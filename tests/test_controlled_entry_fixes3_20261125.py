# tests/test_controlled_entry_fixes3_20261125.py（立项 H1 补修三轮，2026-11-25）
#
# 贵妃第二十二份复查（TWENTY_SECOND_REVIEW）四个残留点，经
# 十一逐项裁决成立后的补修验收：
#   A. 跨重启冻结承接：已决意图的 final_reason 属于冻结集
#      合（invariant_failed/commit_unverified/
#      reconcile_undetermined）时，新进程构造按已落盘意图
#      重建同一面冻结旗——进程1 造不变量失败冻结并落盘，
#      进程2 全新启动后冻结仍在、第二笔被拒、健康记录如
#      实标 frozen，不许变 clean。
#   D. 撤销路径读失败：revoke_controlled_change 的异常分
#      支消费三态查询——读失败/存疑一律未定案
#      （committed_unverified），不许记 aborted/
#      execute_failed。
#   E. 按 op 查询消费 _info：_controlled_ledger_op_state
#      三态（found/absent/undetermined），坏行/截断存疑时
#      查无此 op 也只能判 undetermined；submit 兄弟路径同
#      口径，不判 aborted。
#   小项①：恢复回执缺 version_after 时 detail 加
#      version_after_missing 明示，不许静默留空；有值时
#      不变。
#
# 全程离线：嵌入/判官打桩为 None，同 test_controlled_entry 口径。

import json
import uuid

import pytest

from memory import ledger as _ledger
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


def _prop(action="demote", memory_id=None, source="owner", key=None, **extra):
    p = {
        "action": action,
        "memory_id": memory_id,
        "reason": "补修三轮测试提案",
        "instruction_source": source,
        "idempotency_key": key or uuid.uuid4().hex,
    }
    p.update(extra)
    return p


def _session(store):
    sess = store.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=True
    )
    assert sess is not None
    return sess


def _exec_prop(store, action="demote", memory_id=None, **extra):
    extra.setdefault("session", _session(store))
    extra.setdefault(
        "expected_version", store.controlled_entry_version(memory_id)
    )
    return _prop(action=action, memory_id=memory_id, **extra)


def _enable(store):
    store.controlled_entry_execution_enabled = True
    return store


def _append_bad_line(store):
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("\n{this is not json\n")
    _events, info = _ledger.read_events(store.ledger_path)
    assert info.get("bad_lines"), info


# ---------- A. 跨重启冻结承接 ----------

def test_invariant_freeze_carried_across_restart(store, tmp_path,
                                                 monkeypatch):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)

    def fake_invariants(*args, **kwargs):
        return {"ok": False, "checks": {"members_match_ledger": False}}

    monkeypatch.setattr(store, "_controlled_check_invariants",
                        fake_invariants)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed"
    assert receipt["reason"] == "invariant_failed"
    assert store._controlled_frozen_reason == "invariant_failed"
    # 失败意图已落盘（承接的依据）
    intent = store._controlled_intents[receipt["detail"]["op_id"]]
    assert intent["final_reason"] == "invariant_failed"

    # —— 进程 2：全新实例，状态只许来自落盘文件 ——
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason == "invariant_failed"
    carried = [
        i for i in store2._controlled_health_issues
        if i.get("kind") == "invariant_failed"
        and (i.get("detail") or {}).get("carried_from_intent")
    ]
    assert carried, store2._controlled_health_issues
    # 健康记录如实标 frozen，不许变 clean
    store2.startup_recovery()
    rec = store2.recovery_health()
    assert rec is not None
    assert rec["status"] == "frozen"
    assert rec["controlled_entry"]["frozen"] is True
    assert rec["controlled_entry"]["frozen_reason"] == (
        "invariant_failed"
    )
    # 第二笔提案被拒
    _enable(store2)
    mid2 = _add_user(store2, "用户喜欢喝茶")
    second = store2.submit_controlled_proposal(
        _exec_prop(store2, memory_id=mid2)
    )
    assert second["status"] == "refused"
    assert second["reason"] == "entry_frozen"
    assert second["detail"]["frozen_reason"] == "invariant_failed"


def test_carry_skips_clean_and_pending_intents(store, tmp_path):
    mid = _add_user(store, "用户住在西安")
    # 干净的已决意图：不承接
    store._controlled_intents["op-clean"] = {
        "op_id": "op-clean",
        "idempotency_key": "k-clean",
        "action": "demote",
        "memory_id": mid,
        "status": "committed",
        "final_ok": True,
        "final_reason": "",
        "version_after": "v1",
    }
    # pending 意图哪怕带了冻结级原因也不归承接管（对账管）
    store._controlled_intents["op-pend"] = {
        "op_id": "op-pend",
        "idempotency_key": "k-pend",
        "action": "demote",
        "memory_id": mid,
        "status": "pending",
        "final_reason": "invariant_failed",
        "version_after": None,
    }
    store._controlled_save_intents()
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason is None


def test_commit_unverified_freeze_carried_across_restart(store, tmp_path):
    mid = _add_user(store, "用户喜欢游泳")
    store._controlled_intents["op-cu"] = {
        "op_id": "op-cu",
        "idempotency_key": "k-cu",
        "action": "demote",
        "memory_id": mid,
        "status": "committed_unverified",
        "final_ok": False,
        "final_reason": "commit_unverified",
        "version_after": None,
    }
    store._controlled_save_intents()
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason == "commit_unverified"


# ---------- E. 按 op 查询三态 ----------

def test_ledger_op_state_tri_state(store):
    _add_user(store, "用户住在西安")
    # 干净账本、查无此 op → absent
    state, event = store._controlled_ledger_op_state("op-nope")
    assert state == SQLiteStore._LEDGER_OP_ABSENT
    assert event is None
    assert store._controlled_op_read_failed is False
    # 账本里真有一笔 → found（手写一行带 op_id 的事件）
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {"type": "promote", "op_id": "op-real", "seq": 999}
        ) + "\n")
    state, event = store._controlled_ledger_op_state("op-real")
    assert state == SQLiteStore._LEDGER_OP_FOUND
    assert event["op_id"] == "op-real"
    assert store._controlled_op_read_failed is False
    # 坏行存疑：查无此 op 只能判 undetermined
    _append_bad_line(store)
    state, event = store._controlled_ledger_op_state("op-nope")
    assert state == SQLiteStore._LEDGER_OP_UNDETERMINED
    assert event is None
    assert store._controlled_op_read_failed is True
    # 兼容旧签名：返回 None，但旗标明未定案
    assert store._controlled_ledger_op("op-nope") is None
    assert store._controlled_op_read_failed is True
    # 找得到的那笔不因坏行变未定案
    state, event = store._controlled_ledger_op_state("op-real")
    assert state == SQLiteStore._LEDGER_OP_FOUND


def test_ledger_op_state_read_exception_undetermined(store, monkeypatch):
    def boom(path):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(_ledger, "read_events", boom)
    state, event = store._controlled_ledger_op_state("op-nope")
    assert state == SQLiteStore._LEDGER_OP_UNDETERMINED
    assert event is None
    assert store._controlled_op_read_failed is True


def test_exec_error_with_bad_lines_not_aborted(store, monkeypatch):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)
    _append_bad_line(store)

    def exploding_fields(memory_id, changes, _op_id=None, _idem_key=None,
                         _source=None):
        raise RuntimeError("half-way boom")

    monkeypatch.setattr(store, "_update_core_fields", exploding_fields)
    monkeypatch.setattr(
        store, "_controlled_check_invariants",
        lambda *a, **k: {"ok": True, "checks": {}},
    )
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    # 账本存疑：不许判 aborted，按已提交待核实如实落账
    assert receipt["status"] == "executed"
    assert receipt["detail"]["commit_state"] == "committed_unverified"
    assert receipt["detail"]["ledger_undetermined"] is True
    intent = store._controlled_intents[receipt["detail"]["op_id"]]
    assert intent["status"] == "committed_unverified"


# ---------- D. 撤销异常分支三态 ----------

def _executed_demote(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed", receipt
    assert receipt["ok"] is True, receipt
    return mid, receipt["detail"]["op_id"]


def _revoke_req(op_id, store):
    return {
        "reason": "补修三轮测试撤销",
        "instruction_source": "owner",
        "idempotency_key": uuid.uuid4().hex,
        "target_op_id": op_id,
        "session": _session(store),
    }


def test_revoke_exec_error_ledger_read_failure_undetermined(
    store, monkeypatch
):
    mid, op_id = _executed_demote(store)

    def exploding_fields(memory_id, changes, _op_id=None, _idem_key=None,
                         _source=None):
        raise RuntimeError("compensation boom")

    monkeypatch.setattr(store, "_update_core_fields", exploding_fields)
    real_read = _ledger.read_events
    calls = {"n": 0}

    def flaky_read(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("ledger unreadable")
        return real_read(path)

    monkeypatch.setattr(_ledger, "read_events", flaky_read)
    rev = store.revoke_controlled_change(_revoke_req(op_id, store))
    # 不许假报没执行：不是 refused/execute_failed
    assert rev["status"] != "refused", rev
    assert rev["reason"] != "execute_failed", rev
    rev_intent = store._controlled_intents[rev["revocation_op_id"]]
    assert rev_intent["status"] == "committed_unverified"
    assert rev_intent["status"] != "aborted"
    assert rev["detail"]["ledger_undetermined"] is True
    # 冻结由 commit_unverified 负责
    assert store._controlled_frozen_reason == "commit_unverified"


def test_revoke_exec_error_bad_lines_undetermined(store, monkeypatch):
    mid, op_id = _executed_demote(store)
    _append_bad_line(store)

    def exploding_fields(memory_id, changes, _op_id=None, _idem_key=None,
                         _source=None):
        raise RuntimeError("compensation boom")

    monkeypatch.setattr(store, "_update_core_fields", exploding_fields)
    rev = store.revoke_controlled_change(_revoke_req(op_id, store))
    assert rev["status"] != "refused", rev
    assert rev["reason"] != "execute_failed", rev
    rev_intent = store._controlled_intents[rev["revocation_op_id"]]
    assert rev_intent["status"] == "committed_unverified"
    assert rev["detail"]["ledger_undetermined"] is True


# ---------- 小项①：恢复回执缺 version_after 明示 ----------

def test_receipt_from_intent_marks_missing_version_after(store):
    intent = {
        "op_id": "op-miss",
        "idempotency_key": "k-miss",
        "action": "demote",
        "memory_id": "m1",
        "status": "committed",
        "final_ok": True,
        "final_reason": "",
        "ledger_seq": 7,
        "expected_version": "vb",
        "version_after": None,
    }
    receipt = store._controlled_receipt_from_intent(intent)
    assert receipt["version_after"] is None
    assert receipt["detail"]["version_after_missing"] is True


def test_receipt_from_intent_keeps_present_version_after(store):
    intent = {
        "op_id": "op-have",
        "idempotency_key": "k-have",
        "action": "demote",
        "memory_id": "m1",
        "status": "committed",
        "final_ok": True,
        "final_reason": "",
        "ledger_seq": 7,
        "expected_version": "vb",
        "version_after": "va-123",
    }
    receipt = store._controlled_receipt_from_intent(intent)
    assert receipt["version_after"] == "va-123"
    assert "version_after_missing" not in receipt["detail"]
