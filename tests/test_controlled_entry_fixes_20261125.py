# tests/test_controlled_entry_fixes_20261125.py（立项 H1 补修轮，2026-11-25）
#
# 贵妃第十六封互查（框架认可、完整验收暂不通过）逐项补齐的验收：
#   1. 版本必填：执行路径 expected_version 必填非空合法、锁内
#      复检；回执 before/after 戳的状态语义逐路验（早拒双 None、
#      载行后拒 after None、影子 after None、执行双有值）；
#   2. 参数严验：params 非 dict 拒、tier/drawer 缺字段拒（不许
#      吞成 None 意外摘档/清抽屉）、显式 None 是显式清除决定
#      （detail 标 explicit_clear）、键/目标/来源/版本先验类型、
#      非法一律结构化拒绝不抛 TypeError；
#   3. 校验前移：影子路径也先过参数关，非法参数不再出 shadow；
#   4. 同键异提案：规范指纹绑定，同键同指纹 duplicate、同键
#      异指纹拒 idempotency_conflict，绝不回旧提案的成功回执；
#   5. 提交/恢复与事件关联：op_id 贯通账本事件 meta/审计/回执；
#      执行前落盘 pending 意图，硬崩后启动对账补标 committed /
#      aborted；回执登记丢失时按意图重建 duplicate 不重执行；
#      登记/意图损坏与审计写失败冻结危险写（entry_frozen），
#      已落状态如实标已提交；
#   6. 不变量健康闸口：座位/成员与账本/原文/字段范围/保护与
#      账本/列与 meta 逐项验，失败回执不报 ok 成功
#      （invariant_failed）＋健康标记进 pending_notices；
#   7. 绕行负测：整理员身份绕不过正门（无凭据/非本人来源/
#      删除无批准都拒）；宿主直调 set_* 是宿主自用路、现状
#      未收口——如实记录，不许误封（写路清单在 sqlite_store
#      受控入口段头注释）。
#
# 全程离线：嵌入/判官打桩为 None，同 test_controlled_entry 口径。

import json
import os
import uuid
from pathlib import Path

import pytest

from memory import core_cap as _core_cap
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
        "reason": "补修测试提案",
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
    """一条可执行的提案：带主人直达会话＋当时的行戳。"""
    extra.setdefault("session", _session(store))
    extra.setdefault(
        "expected_version", store.controlled_entry_version(memory_id)
    )
    return _prop(action=action, memory_id=memory_id, **extra)


def _enable(store):
    store.controlled_entry_execution_enabled = True
    return store


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


# ---------- 1. 版本必填与戳语义 ----------

def test_execution_requires_expected_version(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=_session(store))
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "missing_version"
    # 没执行：状态不动
    assert _row(store, mid)["user_confirmed"] is True


def test_expected_version_shape_validated(store):
    mid = _add_user(store, "用户喜欢喝茶")
    _enable(store)
    for bad in ("", 123, ["x"]):
        receipt = store.submit_controlled_proposal(
            _prop(memory_id=mid, session=_session(store),
                  expected_version=bad)
        )
        assert receipt["status"] == "refused"
        assert receipt["reason"] == "bad_expected_version"
    assert _row(store, mid)["user_confirmed"] is True


def test_version_stamp_semantics_per_path(store):
    mid = _add_user(store, "用户计划明年去旅行")
    # 早拒（未知动作）：双 None
    early = store.submit_controlled_proposal(
        _prop(action="explode", memory_id=mid)
    )
    assert early["version_before"] is None
    assert early["version_after"] is None
    # not_found：双 None
    nf = store.submit_controlled_proposal(_prop(memory_id="no-such"))
    assert nf["version_before"] is None and nf["version_after"] is None
    # 载行后拒（来源）：before 有值、after None
    blocked = store.submit_controlled_proposal(
        _prop(memory_id=mid, source="curator")
    )
    assert blocked["reason"] == "not_owner_instruction"
    assert blocked["version_before"]
    assert blocked["version_after"] is None
    # 影子：before 有值、after None
    shadow = store.submit_controlled_proposal(_prop(memory_id=mid))
    assert shadow["status"] == "shadow"
    assert shadow["version_before"]
    assert shadow["version_after"] is None
    # 执行：双有值且不同
    _enable(store)
    done = store.submit_controlled_proposal(_exec_prop(store, memory_id=mid))
    assert done["status"] == "executed"
    assert done["version_before"] == shadow["version_before"]
    assert done["version_after"]
    assert done["version_after"] != done["version_before"]


def test_version_recheck_inside_lock(store):
    # 提案后、执行前正文被改（别的写路）：戳对不上必须拒，
    # 不许拿旧戳改新内容。
    mid = _add_user(store, "用户住在广州")
    _enable(store)
    stamp = store.controlled_entry_version(mid)
    store._conn.execute(
        "UPDATE memories SET text=? WHERE id=?", ("用户住在深圳", mid)
    )
    store._conn.commit()
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, session=_session(store),
              expected_version=stamp)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "version_stale"
    assert _row(store, mid)["user_confirmed"] is True


# ---------- 2. 参数严验 ----------

def test_missing_params_field_refused_no_accidental_clear(store):
    mid = _add_user(store, "用户每月工资一号到账")
    _enable(store)
    ok = store.submit_controlled_proposal(
        _exec_prop(store, action="set_core_drawer", memory_id=mid,
                   params={"drawer": "money_decisions"})
    )
    assert ok["status"] == "executed"
    assert _core_cap.classify(_row(store, mid)["meta"])["drawer"] == (
        "money_decisions"
    )
    # 缺 params：不许吞成 {} 再把 drawer 当 None 清掉
    refused = store.submit_controlled_proposal(
        _exec_prop(store, action="set_core_drawer", memory_id=mid)
    )
    assert refused["status"] == "refused"
    assert refused["reason"] == "bad_params"
    assert refused["detail"]["param_error"] == "missing_field:drawer"
    # params 给了但缺字段：同样拒
    refused2 = store.submit_controlled_proposal(
        _exec_prop(store, action="set_core_drawer", memory_id=mid,
                   params={})
    )
    assert refused2["reason"] == "bad_params"
    # 抽屉原样没被清
    assert _core_cap.classify(_row(store, mid)["meta"])["drawer"] == (
        "money_decisions"
    )
    # tier 缺字段同理
    refused3 = store.submit_controlled_proposal(
        _exec_prop(store, action="set_protection_tier", memory_id=mid,
                   params={})
    )
    assert refused3["reason"] == "bad_params"
    assert refused3["detail"]["param_error"] == "missing_field:tier"


def test_params_non_dict_refused(store):
    mid = _add_user(store, "用户喜欢下棋")
    _enable(store)
    for bad in (["tier"], "pinned", 7):
        receipt = store.submit_controlled_proposal(
            _exec_prop(store, action="set_protection_tier",
                       memory_id=mid, params=bad)
        )
        assert receipt["status"] == "refused"
        assert receipt["reason"] == "bad_params"
    assert _core_cap.normalize_protection_tier(
        _row(store, mid)["meta"]
    ) is None


def test_bad_param_values_refused(store):
    mid = _add_user(store, "用户的工作是程序员")
    _enable(store)
    r1 = store.submit_controlled_proposal(
        _exec_prop(store, action="set_protection_tier", memory_id=mid,
                   params={"tier": "gold"})
    )
    assert r1["reason"] == "bad_params"
    r2 = store.submit_controlled_proposal(
        _exec_prop(store, action="set_core_drawer", memory_id=mid,
                   params={"drawer": "nowhere"})
    )
    assert r2["reason"] == "bad_params"
    row = _row(store, mid)
    assert _core_cap.normalize_protection_tier(row["meta"]) is None
    assert _core_cap.normalize_core_drawer(row["meta"]) is None


def test_explicit_none_is_explicit_clear(store):
    mid = _add_user(store, "用户的生日是五月一号")
    _enable(store)
    ok = store.submit_controlled_proposal(
        _exec_prop(store, action="set_protection_tier", memory_id=mid,
                   params={"tier": "pinned"})
    )
    assert ok["status"] == "executed"
    assert _core_cap.normalize_protection_tier(
        _row(store, mid)["meta"]
    ) == "pinned"
    # 显式 None：是明确决定，照办并留痕（detail 标 explicit_clear）
    cleared = store.submit_controlled_proposal(
        _exec_prop(store, action="set_protection_tier", memory_id=mid,
                   params={"tier": None})
    )
    assert cleared["status"] == "executed"
    assert cleared["detail"]["explicit_clear"] is True
    assert _core_cap.normalize_protection_tier(
        _row(store, mid)["meta"]
    ) is None


def test_field_type_validation_structured(store):
    mid = _add_user(store, "用户养了一只猫")
    _enable(store)
    # 非 str 幂等键（老版实测抛 TypeError）：结构化拒绝
    r = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=["not", "a", "str"])
    )
    assert r["status"] == "refused" and r["reason"] == "bad_idempotency_key"
    # 非 str 目标
    r = store.submit_controlled_proposal(_prop(memory_id=123))
    assert r["reason"] == "bad_target"
    # 非 str 来源
    r = store.submit_controlled_proposal(_prop(memory_id=mid, source=42))
    assert r["reason"] == "bad_source"
    assert _row(store, mid)["user_confirmed"] is True


# ---------- 3. 校验前移（影子也先过参数关） ----------

def test_shadow_path_validates_params_first(store):
    mid = _add_user(store, "用户在学吉他")
    assert store.controlled_entry_execution_enabled is False
    # 开关关着：非法参数出 bad_params，不许出 shadow 回执
    r = store.submit_controlled_proposal(
        _prop(action="set_core_drawer", memory_id=mid, params={})
    )
    assert r["status"] == "refused" and r["reason"] == "bad_params"
    r2 = store.submit_controlled_proposal(
        _prop(action="demote", memory_id=mid, params=["x"])
    )
    assert r2["status"] == "refused" and r2["reason"] == "bad_params"
    # 合法提案在影子模式照常出 shadow
    r3 = store.submit_controlled_proposal(_prop(memory_id=mid))
    assert r3["status"] == "shadow"


# ---------- 4. 同键异提案 ----------

def test_same_key_different_proposal_conflict(store):
    mid_a = _add_user(store, "用户喜欢喝茶")
    mid_b = _add_user(store, "用户喜欢喝咖啡")
    _enable(store)
    key = uuid.uuid4().hex
    first = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid_a, key=key)
    )
    assert first["status"] == "executed"
    # 同键换目标：拒冲突，绝不回旧提案的成功回执
    conflict = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid_b, key=key)
    )
    assert conflict["status"] == "refused"
    assert conflict["reason"] == "idempotency_conflict"
    assert conflict["ok"] is False
    assert conflict["detail"]["prior_fingerprint"]
    assert conflict["detail"]["fingerprint"] != (
        conflict["detail"]["prior_fingerprint"]
    )
    # b 没被动；a 的原回执仍可查
    assert _row(store, mid_b)["user_confirmed"] is True
    original = store.controlled_entry_receipt(key)
    assert original["status"] == "executed"
    assert original["memory_id"] == mid_a
    # 同键同目标但换动作：同样冲突
    conflict2 = store.submit_controlled_proposal(
        _exec_prop(store, action="mark_dormant", memory_id=mid_a,
                   key=key)
    )
    assert conflict2["reason"] == "idempotency_conflict"


def test_same_key_same_proposal_duplicate(store):
    mid = _add_user(store, "用户喜欢爬山")
    _enable(store)
    key = uuid.uuid4().hex
    stamp = store.controlled_entry_version(mid)
    sess = _session(store)
    first = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=sess,
              expected_version=stamp)
    )
    assert first["status"] == "executed"
    # 换一张凭据重提同一提案：凭据不进指纹，仍算同一提案
    second = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store),
              expected_version=stamp)
    )
    assert second["status"] == "duplicate"
    assert second["duplicate"] is True
    assert second["ok"] is True


# ---------- 5. 提交/恢复与事件关联 ----------

def test_op_id_links_ledger_audit_receipt(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)
    key = uuid.uuid4().hex
    stamp = store.controlled_entry_version(mid)
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store),
              expected_version=stamp)
    )
    assert receipt["status"] == "executed"
    detail = receipt["detail"]
    op_id = detail["op_id"]
    assert op_id.startswith("op-")
    assert detail["commit_state"] == "committed"
    assert isinstance(detail["ledger_seq"], int)
    # 账本事件带 op_id＋idem_key，seq 与回执对得上
    promote = [
        e for e in _ledger_events(store)
        if e.get("type") == "promote" and e.get("memory_id") == mid
    ]
    assert len(promote) == 1
    assert promote[0]["op_id"] == op_id
    assert promote[0]["idem_key"] == key
    assert promote[0]["seq"] == detail["ledger_seq"]
    # 审计行同带 op_id/指纹/参数/版本
    executed = [
        e for e in _trace_events(store) if e["event"] == "executed"
    ]
    assert len(executed) == 1
    line = executed[0]
    assert line["op_id"] == op_id
    assert line["fingerprint"]
    assert line["expected_version"] == stamp
    assert line["version_before"] == stamp
    assert line["version_after"] == receipt["version_after"]
    # 意图登记：committed、可按 op 查
    intent = store._controlled_intents[op_id]
    assert intent["status"] == "committed"
    assert intent["ledger_seq"] == detail["ledger_seq"]


def test_crash_reconcile_committed_after_hard_crash(store, tmp_path):
    mid = _add_user(store, "用户养了一只狗")
    _enable(store)
    key = uuid.uuid4().hex
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid, key=key)
    )
    assert receipt["status"] == "executed"
    op_id = receipt["detail"]["op_id"]
    # 模拟硬崩：意图文件里这笔被拨回 pending（死在标提交前）
    intents_path = store._controlled_intents_path
    data = json.loads(intents_path.read_text(encoding="utf-8"))
    data["intents"][op_id]["status"] = "pending"
    data["intents"][op_id]["ledger_seq"] = None
    intents_path.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    # 重建 store：启动对账按账本 op_id 补标 committed
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    report = store2._controlled_reconcile_report
    assert report["committed"] == 1 and report["aborted"] == 0
    assert store2._controlled_intents[op_id]["status"] == "committed"
    # 回执可查（登记还在时认登记）
    found = store2.controlled_entry_receipt(key)
    assert found is not None and found["status"] == "executed"


def test_crash_reconcile_aborted_then_retry_allowed(store, tmp_path):
    mid = _add_user(store, "用户住在西安")
    _enable(store)
    stamp = store.controlled_entry_version(mid)
    key = "crafted-key-1"
    fingerprint = SQLiteStore._controlled_fingerprint(
        "demote", mid, {}, "owner", stamp
    )
    # 手工造一笔 pending 意图：账本里没有这笔 op（死在落账前）
    store._controlled_intents["op-deadbeef"] = {
        "op_id": "op-deadbeef",
        "idempotency_key": key,
        "fingerprint": fingerprint,
        "action": "demote",
        "memory_id": mid,
        "expected_version": stamp,
        "status": "pending",
        "ts": "2026-11-25T00:00:00+00:00",
        "ledger_seq": None,
        "version_after": None,
    }
    store._controlled_save_intents()
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    report = store2._controlled_reconcile_report
    assert report["aborted"] == 1
    assert store2._controlled_intents["op-deadbeef"]["status"] == (
        "aborted"
    )
    # 已中止的同键同指纹提案可以重试执行（不是冲突、不是
    # duplicate）：那次根本没落账
    store2.controlled_entry_execution_enabled = True
    receipt = store2.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store2),
              expected_version=stamp)
    )
    assert receipt["status"] == "executed"
    assert _row(store2, mid)["user_confirmed"] is False


def test_receipts_lost_reconstruct_from_intent_no_reexecution(
    store, tmp_path
):
    mid = _add_user(store, "用户喜欢游泳")
    _enable(store)
    key = uuid.uuid4().hex
    stamp = store.controlled_entry_version(mid)
    first = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store),
              expected_version=stamp)
    )
    assert first["status"] == "executed"
    # 回执登记文件丢失（意图还在）
    os.remove(store._controlled_receipts_path)
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_receipts == {}
    again = store2.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store2),
              expected_version=stamp)
    )
    assert again["status"] == "duplicate"
    assert again["ok"] is True
    assert again["detail"]["reconstructed_from_intent"] is True
    # 没有重执行：账本里带这把键的 PROMOTE 只有一笔
    promotes = [
        e for e in _ledger_events(store2)
        if e.get("type") == "promote" and e.get("idem_key") == key
    ]
    assert len(promotes) == 1


def test_corrupt_intents_freezes_execution(store, tmp_path):
    mid = _add_user(store, "用户喜欢读书")
    store._controlled_intents_path.write_text(
        "{not json", encoding="utf-8"
    )
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_intents_load_error
    assert store2._controlled_frozen_reason == "intents_load_error"
    # 原文件改名保全
    backups = list((tmp_path / "memory").glob(
        "controlled_entry_intents.json.corrupt-*"
    ))
    assert len(backups) == 1
    # 执行被冻结；影子仍可出
    shadow = store2.submit_controlled_proposal(_prop(memory_id=mid))
    assert shadow["status"] == "shadow"
    store2.controlled_entry_execution_enabled = True
    frozen = store2.submit_controlled_proposal(
        _prop(memory_id=mid, session=_session(store2),
              expected_version=store2.controlled_entry_version(mid))
    )
    assert frozen["status"] == "refused"
    assert frozen["reason"] == "entry_frozen"
    assert frozen["detail"]["frozen_reason"] == "intents_load_error"
    assert _row(store2, mid)["user_confirmed"] is True
    # 健康痕迹在 pending_notices 里露头
    kinds = [n["kind"] for n in store2.pending_notices()]
    assert "controlled_entry_frozen" in kinds


def test_corrupt_receipts_freezes_execution(store, tmp_path):
    mid = _add_user(store, "用户喜欢跑步")
    store._controlled_receipts_path.write_text(
        "[1,2", encoding="utf-8"
    )
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_receipts_load_error
    assert store2._controlled_frozen_reason == "receipts_load_error"
    store2.controlled_entry_execution_enabled = True
    frozen = store2.submit_controlled_proposal(
        _prop(memory_id=mid, session=_session(store2),
              expected_version=store2.controlled_entry_version(mid))
    )
    assert frozen["reason"] == "entry_frozen"
    assert _row(store2, mid)["user_confirmed"] is True


def test_audit_write_failure_freezes_but_receipt_returned(
    store, tmp_path
):
    mid = _add_user(store, "用户每天喝两杯水")
    _enable(store)
    # 审计日志路径指向一个目录：追加必失败
    store._controlled_log_path = Path(tmp_path)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    # 已落状态如实报已提交：回执不丢、执行不假报 refused
    assert receipt["status"] == "executed"
    assert receipt["ok"] is True
    assert receipt["detail"]["audit_write_failed"] is True
    assert receipt["trace_index"] is None
    assert _row(store, mid)["user_confirmed"] is False
    # 冻结后续危险写
    assert store._controlled_frozen_reason == "audit_write_failed"
    mid2 = _add_user(store, "用户喜欢弹琴")
    frozen = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid2)
    )
    assert frozen["status"] == "refused"
    assert frozen["reason"] == "entry_frozen"


def test_receipts_save_failure_freezes(store, monkeypatch):
    mid = _add_user(store, "用户喜欢画画")
    _enable(store)

    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_controlled_save_receipts", boom)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed"
    assert receipt["detail"]["receipt_persist_failed"] is True
    assert store._controlled_frozen_reason == "receipts_save_failed"
    mid2 = _add_user(store, "用户喜欢唱歌")
    frozen = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid2)
    )
    assert frozen["reason"] == "entry_frozen"


def test_intent_persist_failure_refuses_before_execution(
    store, monkeypatch
):
    mid = _add_user(store, "用户喜欢跳舞")
    _enable(store)

    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_controlled_save_intents", boom)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    # 意图落不了盘就不许执行：没提交点不开工
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "intent_persist_failed"
    assert _row(store, mid)["user_confirmed"] is True
    assert store._controlled_frozen_reason == "intents_save_failed"


# ---------- 6. 不变量健康闸口 ----------

def test_invariant_detail_all_checks_present(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed"
    checks = receipt["detail"]["invariants"]
    for name in (
        "seats_reconciled", "members_match_ledger", "text_unchanged",
        "fields_confined", "protection_matches_ledger",
        "columns_match_meta", "row_present",
    ):
        assert checks[name] is True, name
    assert receipt["detail"]["invariants_ok"] is True


def test_mark_dormant_invariants_include_dormant_check(store):
    mid = _add_user(store, "用户上周去了一趟超市")
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, action="mark_dormant", memory_id=mid)
    )
    assert receipt["status"] == "executed"
    assert receipt["detail"]["invariants"]["dormant_matches_ledger"] is (
        True
    )
    # DORMANT 事件同样带 op_id 贯通
    dormant = [
        e for e in _ledger_events(store)
        if e.get("type") == "dormant" and e.get("memory_id") == mid
    ]
    assert len(dormant) == 1
    assert dormant[0]["op_id"] == receipt["detail"]["op_id"]


def test_invariant_failure_not_reported_as_clean_success(
    store, monkeypatch
):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)

    def silent_fields(memory_id, changes, _op_id=None, _idem_key=None):
        # 模拟半截提交：声称改了核心字段，实际 meta 没动、
        # 账本没落——提交后列被入口的第二笔写改掉，列与
        # meta 立刻分叉，不变量闸口必须逮住。
        return {"ok": True, "memory_id": memory_id}

    monkeypatch.setattr(store, "_update_core_fields", silent_fields)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed"
    assert receipt["ok"] is False
    assert receipt["reason"] == "invariant_failed"
    assert receipt["detail"]["invariants_ok"] is False
    assert receipt["detail"]["invariants"]["columns_match_meta"] is (
        False
    )
    # 如实标已提交/待核实（账本里没有这笔 op 的事件）
    assert receipt["detail"]["commit_state"] == "committed_unverified"
    # 健康标记落下、pending_notices 露头
    issues = [
        n for n in store.pending_notices()
        if n["kind"] == "controlled_entry_health"
    ]
    assert any(n["issue"] == "invariant_failed" for n in issues)
    assert any(n["issue"] == "commit_unverified" for n in issues)


# ---------- 7. 绕行负测 ----------

def test_bypass_attempts_refused_and_host_paths_documented(store):
    mid = _add_user(store, "用户喜欢喝茶")
    _enable(store)
    # 整理员身份从正门进、无凭据：拒（H2 口径）
    r = store.submit_controlled_proposal(
        _prop(memory_id=mid,
              expected_version=store.controlled_entry_version(mid))
    )
    assert r["status"] == "refused" and r["reason"] == "no_session"
    # 删除绕行：没批准不删、非本人来源带批准也不删
    assert store.delete_memory(mid, approve=False)["deleted"] is False
    out = store.delete_memory(
        mid, approve=True, instruction_source="curator"
    )
    assert out["deleted"] is False
    assert out["reason"] == "not_owner_instruction"
    assert _row(store, mid)["user_confirmed"] is True
    # 宿主直调 set_protection_tier / set_core_drawer 是宿主
    # 自用路（写路清单已点名、未收口）：现状是宿主能直改，
    # 本测试如实记录这一点——不许误封宿主自用路；整理员的
    # 同名动作只许走正门（上面的正门拒绝用例即边界）。
    assert store.set_protection_tier(mid, "pinned")["ok"] is True
    assert _core_cap.normalize_protection_tier(
        _row(store, mid)["meta"]
    ) == "pinned"
    assert store.set_core_drawer(mid, "person")["ok"] is True
    assert _core_cap.normalize_core_drawer(
        _row(store, mid)["meta"]
    ) == "person"
