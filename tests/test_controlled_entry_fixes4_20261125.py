# tests/test_controlled_entry_fixes4_20261125.py（立项 H1 补修四轮，2026-11-25）
#
# 贵妃第二十三份复查（TWENTY_THIRD_REVIEW）唯一新残留，经
# 十一裁决成立（高）：commit_unverified 类冻结跨重启断链。
# 病根：两个独立事实——「提交未定案」与「不变量结论」——
# 挤在 final_reason 一个字段里，而生产者只写不变量结论
# （""/invariant_failed），未定案事实只活在 status 里；
# 三轮的承接函数却拿 final_reason 当闸口，空串被挡，重启
# 后冻结蒸发（同进程冻着、重启解冻）。
#
# 本轮口径（裁决推荐第 1 条：事实分开存、承接看状态）：
#   _controlled_carry_intent_freezes 对 status ==
#   "committed_unverified" 的已决意图无条件以
#   commit_unverified 承接（detail 标 carried_from_status）；
#   final_reason 命中冻结集合的原承接保留（detail 标
#   carried_from_intent），两条来源分开标注。
#
# 契约（必备）：正向＋撤销各一条真实生产者路径——未定案
# ＋不变量/核验通过 → 落盘意图 final_reason == "" →
# 新构造（落盘重建）→ 冻结仍在、第二笔 entry_frozen；
# 负测：真核实 committed 不承接冻结。
#
# 全程离线：嵌入/判官打桩为 None，同 test_controlled_entry 口径。

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
        "reason": "补修四轮测试提案",
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


def _exploding_fields(memory_id, changes, _op_id=None, _idem_key=None,
                      _source=None):
    raise RuntimeError("half-way boom")


def _carried(store, kind, source_key):
    return [
        i for i in store._controlled_health_issues
        if i.get("kind") == kind
        and (i.get("detail") or {}).get(source_key)
    ]


def _assert_second_refused(store2, tmp_path, reason):
    store2.startup_recovery()
    rec = store2.recovery_health()
    assert rec is not None
    assert rec["status"] == "frozen"
    assert rec["controlled_entry"]["frozen"] is True
    assert rec["controlled_entry"]["frozen_reason"] == reason
    _enable(store2)
    mid2 = _add_user(store2, "用户喜欢喝茶")
    second = store2.submit_controlled_proposal(
        _exec_prop(store2, memory_id=mid2)
    )
    assert second["status"] == "refused"
    assert second["reason"] == "entry_frozen"
    assert second["detail"]["frozen_reason"] == reason


# ---------- 正向：未定案＋不变量通过 → 跨重启冻结仍在 ----------

def test_submit_commit_unverified_freeze_carried_across_restart(
    store, tmp_path, monkeypatch
):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)
    _append_bad_line(store)
    monkeypatch.setattr(store, "_update_core_fields", _exploding_fields)
    monkeypatch.setattr(
        store, "_controlled_check_invariants",
        lambda *a, **k: {"ok": True, "checks": {}},
    )
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed"
    assert receipt["detail"]["commit_state"] == "committed_unverified"
    intent = store._controlled_intents[receipt["detail"]["op_id"]]
    assert intent["status"] == "committed_unverified"
    # 断链的要害：生产者只写不变量结论，未定案事实不在
    # final_reason 里——三轮的闸口正是被这个空串挡住的。
    assert intent["final_ok"] is True
    assert intent["final_reason"] == ""
    assert store._controlled_frozen_reason == "commit_unverified"

    # —— 进程 2：全新实例，状态只许来自落盘文件 ——
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason == "commit_unverified"
    assert _carried(store2, "commit_unverified", "carried_from_status")
    _assert_second_refused(store2, tmp_path, "commit_unverified")


# ---------- 撤销：未定案＋核验通过 → 跨重启冻结仍在 ----------

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
        "reason": "补修四轮测试撤销",
        "instruction_source": "owner",
        "idempotency_key": uuid.uuid4().hex,
        "target_op_id": op_id,
        "session": _session(store),
    }


def test_revoke_commit_unverified_freeze_carried_across_restart(
    store, tmp_path, monkeypatch
):
    mid, op_id = _executed_demote(store)
    monkeypatch.setattr(store, "_update_core_fields", _exploding_fields)
    monkeypatch.setattr(
        store, "_controlled_verify_revocation",
        lambda *a, **k: {"ok": True, "checks": {}},
    )
    real_read = _ledger.read_events
    calls = {"n": 0}

    def flaky_read(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("ledger unreadable")
        return real_read(path)

    monkeypatch.setattr(_ledger, "read_events", flaky_read)
    rev = store.revoke_controlled_change(_revoke_req(op_id, store))
    assert rev["status"] != "refused", rev
    assert rev["reason"] != "execute_failed", rev
    rev_intent = store._controlled_intents[rev["revocation_op_id"]]
    assert rev_intent["status"] == "committed_unverified"
    # 撤销生产者同病同断：核验通过 → final_reason 天生空串。
    assert rev_intent["final_ok"] is True
    assert rev_intent["final_reason"] == ""
    assert store._controlled_frozen_reason == "commit_unverified"

    # —— 进程 2：全新实例，状态只许来自落盘文件 ——
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason == "commit_unverified"
    carried = _carried(store2, "commit_unverified", "carried_from_status")
    assert carried, store2._controlled_health_issues
    assert carried[0]["detail"]["carried_from_status"] == (
        rev["revocation_op_id"]
    )
    _assert_second_refused(store2, tmp_path, "commit_unverified")


# ---------- 负测：真核实 committed 不承接冻结 ----------

def test_verified_committed_does_not_carry_freeze(store, tmp_path):
    mid, op_id = _executed_demote(store)
    intent = store._controlled_intents[op_id]
    assert intent["status"] == "committed"
    assert intent["final_ok"] is True
    assert intent["final_reason"] == ""
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason is None
    assert not _carried(store2, "commit_unverified", "carried_from_status")
    assert not _carried(store2, "commit_unverified", "carried_from_intent")


# ---------- 同笔两条都命中：来源分开、顺序与进程内一致 ----------

def test_carry_combined_status_and_reason_sources(store, tmp_path):
    mid = _add_user(store, "用户喜欢游泳")
    # 真实可达的组合：账本未定案（status）＋不变量失败
    # （final_reason）。进程内产生顺序是未定案留痕在前、
    # 不变量结论在后，冻结原因以最早的 commit_unverified
    # 为准；承接必须同口径，且两条来源分别留痕。
    store._controlled_intents["op-both"] = {
        "op_id": "op-both",
        "idempotency_key": "k-both",
        "action": "demote",
        "memory_id": mid,
        "status": "committed_unverified",
        "final_ok": False,
        "final_reason": "invariant_failed",
        "version_after": None,
    }
    store._controlled_save_intents()
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason == "commit_unverified"
    assert _carried(store2, "commit_unverified", "carried_from_status")
    assert _carried(store2, "invariant_failed", "carried_from_intent")
