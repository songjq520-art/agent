# tests/test_controlled_entry_fixes2_20261125.py（立项 H1 补修二轮，2026-11-25）
#
# 贵妃第十八份复查（EIGHTEENTH_REVIEW）六项弱点 A–F＋附带项＋
# 两处小项，经十一逐项裁决成立后的补修验收：
#   A. 不变量失败/提交待核实进冻结集合（与 H6 同一把锁），
#      冻结后第二笔被拒 entry_frozen；H6 恢复结论把冻结类痕
#      迹列为冻结级残余（controlled_entry_frozen），不再混进
#      注意级；恢复自身不许替受控入口的冻结背 recovery_unclean。
#   B. 意图存最终 ok/reason：回执登记丢失后按意图重建，失败
#      仍是失败（invariant_failed），不许对 committed 一律
#      ok=True；无最终结论的老意图按未核实交回。
#   C. drawer/tier 非字符串参数（list/dict）先验类型，结构化
#      bad_params 拒绝，不许 TypeError 逃出；宿主直调同口径。
#   D. 账本读失败 fail-closed：对账冻结＋待决意图按未定案，
#      同键重提拒 prior_unresolved，不许按新提案放行；执行半
#      路出错＋账本读失败时不许判 aborted。
#   E. 对账消费 read_events 的 _info：账本有坏行时待决意图留
#      pending 标未定案，不许直接判 aborted。
#   F. repair 无条件核对 user_confirmed 列（meta 已一致也修列
#      分叉，计数并入 meta_fixed、报告键集不变）；H6 复检查询
#      覆盖该列，列/meta 分叉被复检抓住。
#   附带：members_match_ledger 比成员 ID 集合——同计数换成员
#      必须报不一致。
#   小项①：版本戳覆盖 user_confirmed 列，列单独变戳必变。
#   小项②：无指纹老回执在查询/重提时带 legacy/unverified 标
#      记（只标返回副本，不改登记原件）。
#   持久化：save_intents/save_receipts 写后 fsync 文件。
#
# 全程离线：嵌入/判官打桩为 None，同 test_controlled_entry 口径。

import json
import os
import uuid

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
        "reason": "补修二轮测试提案",
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


def _set_meta_field(store, mid, field, value):
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE id=?", (mid,)
    ).fetchone()
    meta = json.loads(row[0] or "{}")
    if value is None:
        meta.pop(field, None)
    else:
        meta[field] = value
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (json.dumps(meta, ensure_ascii=False), mid),
    )
    store._conn.commit()


def _craft_pending_intent(store, mid, key):
    stamp = store.controlled_entry_version(mid)
    fingerprint = SQLiteStore._controlled_fingerprint(
        "demote", mid, {}, "owner", stamp
    )
    store._controlled_intents["op-crafted"] = {
        "op_id": "op-crafted",
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
    return stamp


# ---------- A. 不变量失败/待核实冻结 ----------

def test_invariant_failure_freezes_entry(store, monkeypatch):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)

    def fake_invariants(*args, **kwargs):
        return {"ok": False, "checks": {"members_match_ledger": False}}

    monkeypatch.setattr(store, "_controlled_check_invariants", fake_invariants)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed"
    assert receipt["ok"] is False
    assert receipt["reason"] == "invariant_failed"
    # 真执行已落账（无 commit_unverified 在前），冻结原因就是
    # invariant_failed 本人
    assert store._controlled_frozen_reason == "invariant_failed"
    # 冻结后第二笔被拒
    mid2 = _add_user(store, "用户喜欢喝茶")
    second = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid2)
    )
    assert second["status"] == "refused"
    assert second["reason"] == "entry_frozen"
    assert second["detail"]["frozen_reason"] == "invariant_failed"


def test_commit_unverified_freezes_entry(store, monkeypatch):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)

    def silent_fields(memory_id, changes, _op_id=None, _idem_key=None,
                      _source=None):
        return {"ok": True, "memory_id": memory_id}

    monkeypatch.setattr(store, "_update_core_fields", silent_fields)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    assert receipt["status"] == "executed"
    assert receipt["ok"] is False
    # 账本里没有这笔 op：先落 commit_unverified，它先进冻结集合
    assert store._controlled_frozen_reason == "commit_unverified"
    mid2 = _add_user(store, "用户喜欢游泳")
    second = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid2)
    )
    assert second["status"] == "refused"
    assert second["reason"] == "entry_frozen"


def test_recovery_record_marks_invariant_failure_freezing(store):
    _add_user(store, "用户养了一只猫")
    store._controlled_note_health("invariant_failed", {"op_id": "op-x"})
    assert store._controlled_frozen_reason == "invariant_failed"
    store.startup_recovery()
    rec = store.recovery_health()
    assert rec is not None
    assert rec["status"] == "frozen"
    frozen_res = [
        r for r in rec["residuals"] if r["kind"] == "controlled_entry_frozen"
    ]
    assert len(frozen_res) == 1
    assert frozen_res[0]["freezing"] is True
    assert "invariant_failed" in frozen_res[0]["detail"]["kinds"]
    # 不再混进注意级
    health_res = [
        r for r in rec["residuals"]
        if r["kind"] == "controlled_entry_health"
    ]
    assert health_res == []
    # 恢复不许把受控入口的冻结改标成 recovery_unclean
    assert store._controlled_frozen_reason == "invariant_failed"


def test_persistence_freeze_not_relabelled_by_recovery(store, tmp_path):
    _add_user(store, "用户喜欢读书")
    store._controlled_receipts_path.write_text("{not json", encoding="utf-8")
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason == "receipts_load_error"
    store2.startup_recovery()
    assert store2._controlled_frozen_reason == "receipts_load_error"


# ---------- B. 意图重建如实恢复最终结论 ----------

def test_registry_loss_preserves_invariant_failure(store, tmp_path,
                                                    monkeypatch):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)
    key = uuid.uuid4().hex
    stamp = store.controlled_entry_version(mid)

    def fake_invariants(*args, **kwargs):
        return {"ok": False, "checks": {"members_match_ledger": False}}

    monkeypatch.setattr(store, "_controlled_check_invariants", fake_invariants)
    first = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store),
              expected_version=stamp)
    )
    assert first["status"] == "executed"
    assert first["reason"] == "invariant_failed"
    # 意图里已落最终结论
    intent = store._controlled_intents[first["detail"]["op_id"]]
    assert intent["final_ok"] is False
    assert intent["final_reason"] == "invariant_failed"
    # 回执登记丢失，只剩意图
    os.remove(store._controlled_receipts_path)
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_receipts == {}
    again = store2.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store2),
              expected_version=stamp)
    )
    assert again["status"] == "duplicate"
    # 失败仍是失败，不许洗成 ok
    assert again["ok"] is False
    assert again["reason"] == "invariant_failed"
    assert again["detail"]["reconstructed_from_intent"] is True
    # 没有重执行
    events, _info = _ledger.read_events(store2.ledger_path)
    promotes = [
        e for e in events
        if e.get("type") == "promote" and e.get("idem_key") == key
    ]
    assert len(promotes) == 1


def test_legacy_intent_without_verdict_unverified(store):
    mid = _add_user(store, "用户住在西安")
    _enable(store)
    key = "crafted-legacy-1"
    stamp = _craft_pending_intent(store, mid, key)
    intent = store._controlled_intents["op-crafted"]
    intent["status"] = "committed"
    intent["ledger_seq"] = 1
    # 老意图没有 final_* 字段（补修前执行落的）
    assert "final_ok" not in intent
    store._controlled_save_intents()
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store),
              expected_version=stamp)
    )
    assert receipt["status"] == "duplicate"
    assert receipt["ok"] is False
    assert receipt["reason"] == "committed_unverified"
    assert receipt["detail"]["unverified_reconstruction"] is True


# ---------- C. 非可哈希参数结构化拒绝 ----------

def test_unhashable_params_structured_refusal(store):
    mid = _add_user(store, "用户喜欢喝茶")
    _enable(store)
    for bad_params in ({"drawer": []}, {"drawer": {}}, {"drawer": 3}):
        receipt = store.submit_controlled_proposal(
            _exec_prop(store, action="set_core_drawer", memory_id=mid,
                       params=bad_params)
        )
        assert receipt["status"] == "refused", bad_params
        assert receipt["reason"] == "bad_params", bad_params
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, action="set_protection_tier", memory_id=mid,
                   params={"tier": []})
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "bad_params"
    # 行没被动过
    row = _row(store, mid)
    assert _core_cap.normalize_core_drawer(row["meta"]) is None
    assert _core_cap.normalize_protection_tier(row["meta"]) is None


def test_unhashable_params_refused_on_shadow_path(store):
    mid = _add_user(store, "用户喜欢喝茶")
    # 执行开关关着：校验前移，非法参数也不许出 shadow
    receipt = store.submit_controlled_proposal(
        _prop(action="set_core_drawer", memory_id=mid,
              params={"drawer": []})
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "bad_params"


def test_host_set_core_drawer_unhashable_no_raise(store):
    mid = _add_user(store, "用户喜欢喝茶")
    for bad in ([], {}, 3):
        out = store.set_core_drawer(mid, bad)
        assert out["ok"] is False, bad
        assert out["reason"] == "bad_drawer", bad
    # 正常值照常可用
    assert store.set_core_drawer(mid, "person")["ok"] is True


# ---------- D/E. 对账 fail-closed ----------

def test_reconcile_read_failure_fail_closed(store, monkeypatch):
    mid = _add_user(store, "用户住在西安")
    _enable(store)
    key = "crafted-key-readfail"
    stamp = _craft_pending_intent(store, mid, key)

    def boom(path):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(_ledger, "read_events", boom)
    report = store.controlled_entry_reconcile()
    assert report["aborted"] == 0
    assert report["still_pending"] == 1
    assert store._controlled_intents["op-crafted"]["status"] == "pending"
    assert store._controlled_frozen_reason == "reconcile_undetermined"
    # 同键重提：拒 prior_unresolved，绝不按新提案执行
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store),
              expected_version=stamp)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "prior_unresolved"
    assert _row(store, mid)["user_confirmed"] is True
    # 账本恢复可读后，对账照旧能把没落账的意图判 aborted
    monkeypatch.undo()
    report2 = store.controlled_entry_reconcile()
    assert report2["aborted"] == 1


def test_reconcile_bad_lines_undetermined_not_aborted(store):
    mid = _add_user(store, "用户住在西安")
    _enable(store)
    key = "crafted-key-badline"
    stamp = _craft_pending_intent(store, mid, key)
    # 往账本尾巴塞一行坏行：read_events 的 _info 必须被消费
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("\n{this is not json\n")
    _events, info = _ledger.read_events(store.ledger_path)
    assert info.get("bad_lines"), info
    report = store.controlled_entry_reconcile()
    assert report["aborted"] == 0
    assert report["still_pending"] == 1
    assert store._controlled_intents["op-crafted"]["status"] == "pending"
    assert store._controlled_frozen_reason == "reconcile_undetermined"
    receipt = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key, session=_session(store),
              expected_version=stamp)
    )
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "prior_unresolved"
    assert _row(store, mid)["user_confirmed"] is True


def test_exec_error_with_ledger_read_failure_not_aborted(
    store, monkeypatch
):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    _enable(store)

    def exploding_fields(memory_id, changes, _op_id=None, _idem_key=None,
                         _source=None):
        raise RuntimeError("half-way boom")

    monkeypatch.setattr(store, "_update_core_fields", exploding_fields)

    def boom(path):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(_ledger, "read_events", boom)
    monkeypatch.setattr(
        store, "_controlled_check_invariants",
        lambda *a, **k: {"ok": True, "checks": {}},
    )
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid)
    )
    # 不许假报 refused/aborted：按已提交待核实如实落账
    assert receipt["status"] == "executed"
    assert receipt["detail"]["commit_state"] == "committed_unverified"
    intent = store._controlled_intents[receipt["detail"]["op_id"]]
    assert intent["status"] == "committed_unverified"


# ---------- F. user_confirmed 列无条件核对 ----------

def test_repair_fixes_column_only_divergence(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    assert _row(store, mid)["user_confirmed"] is True
    # 只改独立列，meta 不动：meta 与账本仍一致，列分叉
    store._conn.execute(
        "UPDATE memories SET user_confirmed=0 WHERE id=?", (mid,)
    )
    store._conn.commit()
    assert _row(store, mid)["user_confirmed"] is False
    report = store.repair_projection_from_ledger()
    # 报告键集不变（不许新增键）
    assert set(report.keys()) == {
        "rows_checked", "superseded_fixed", "deleted_fixed",
        "meta_fixed", "extra_rows",
    }
    assert report["meta_fixed"] >= 1
    assert _row(store, mid)["user_confirmed"] is True


def test_recheck_catches_column_divergence(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    store._conn.execute(
        "UPDATE memories SET user_confirmed=0 WHERE id=?", (mid,)
    )
    store._conn.commit()
    residual = store._projection_residual_mismatches()
    assert residual["mismatch_count"] >= 1
    assert mid in residual["sample_ids"]


# ---------- 附带：成员按 ID 集合比对 ----------

def test_members_match_ledger_compares_id_sets(store):
    a = _add_user(store, "用户的好朋友叫阿明")
    b = _add_user(store, "用户计划下个月去旅行")
    c = _add_user(store, "用户每天喝两杯水")
    # 账本侧：b 被钉死（宿主正路，账本＋投影一致）
    assert store.set_protection_tier(b, "pinned")["ok"] is True
    # 投影侧偷偷换人：a 的 meta 标 pinned、b 的 meta 摘掉——
    # 四个计数两边完全相等，只有成员 ID 集合不同
    tier_field = _core_cap.PROTECTION_TIER_FIELD
    _set_meta_field(store, a, tier_field, "pinned")
    _set_meta_field(store, b, tier_field, None)
    _enable(store)
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=c)
    )
    assert receipt["status"] == "executed"
    assert receipt["detail"]["invariants"]["members_match_ledger"] is (
        False
    )
    assert receipt["reason"] == "invariant_failed"


# ---------- 小项①：版本戳覆盖 user_confirmed 列 ----------

def test_version_stamp_covers_user_confirmed_column(store):
    mid = _add_user(store, "用户每天定投五块钱买基金")
    stamp1 = store.controlled_entry_version(mid)
    assert stamp1 == store.controlled_entry_version(mid)
    # 只翻独立列（meta 不动）：戳必须变
    store._conn.execute(
        "UPDATE memories SET user_confirmed=0 WHERE id=?", (mid,)
    )
    store._conn.commit()
    stamp2 = store.controlled_entry_version(mid)
    assert stamp2 != stamp1
    assert stamp2 == store.controlled_entry_version(mid)


# ---------- 小项②：老回执 legacy/unverified 标记 ----------

def test_legacy_receipt_marked_on_query_and_resubmit(store):
    mid = _add_user(store, "用户喜欢游泳")
    key = "legacy-key-1"
    store._controlled_receipts[key] = {
        "ok": True,
        "status": "executed",
        "reason": "",
        "action": "demote",
        "memory_id": mid,
        "idempotency_key": key,
        "duplicate": False,
        "question_id": None,
        "trace_index": 0,
        "version_before": "vb",
        "version_after": "va",
        "detail": {"op_id": "op-legacy"},
        "ts": "2026-11-25T00:00:00+00:00",
    }
    store._controlled_save_receipts()
    assert key not in store._controlled_fingerprints
    found = store.controlled_entry_receipt(key)
    assert found is not None
    assert found["detail"]["legacy_receipt"] is True
    assert found["detail"]["unverified"] is True
    # 标记只落在返回副本上，登记原件不许被改
    assert "legacy_receipt" not in store._controlled_receipts[key][
        "detail"
    ]
    again = store.submit_controlled_proposal(
        _prop(memory_id=mid, key=key)
    )
    assert again["status"] == "duplicate"
    assert again["detail"]["legacy_receipt"] is True
    assert again["detail"]["unverified"] is True


# ---------- 持久化：写后 fsync ----------

def test_controlled_saves_fsync_files(store, monkeypatch):
    calls = []
    real_fsync = os.fsync

    def counting_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    store._controlled_save_intents()
    store._controlled_save_receipts()
    # 两个文件各至少一次文件级 fsync（目录 fsync 平台相关，
    # 只断言下限）
    assert len(calls) >= 2
    intents = json.loads(
        store._controlled_intents_path.read_text(encoding="utf-8")
    )
    receipts = json.loads(
        store._controlled_receipts_path.read_text(encoding="utf-8")
    )
    assert intents["version"] == 1
    assert receipts["version"] == 3
