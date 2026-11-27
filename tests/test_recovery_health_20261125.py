# tests/test_recovery_health_20261125.py（立项 H6：恢复健康度一组，2026-11-25）
#
# 覆盖口径（施工清单 H6＋契约五）：
#   - 落盘：干净恢复产健康记录（recovery_health.json），内容与恢复
#     报告一致；重启后新实例不跑恢复也读得出上一份；历史留最近
#     若干份；健康文件本身损坏不堵构造、恢复后重写干净记录；
#   - 危险写冻结：账本序号缺口 / 账本里没有的行（投影与账本对不
#     上）→ 恢复不干净 → 受控入口执行被拒 entry_frozen、撤销执行
#     被拒，影子照常出回执；
#   - 冻结落盘跨重启承接：新实例没跑恢复先冻着；账本修好后一次
#     干净恢复解冻（记录里留 unfrozen 痕迹）；显式只读复检能解
#     recovery_unclean 来源的冻，解不了持久化故障来源的冻；
#   - 结论页映射：pending_notices 的 recovery_health 条字段对得
#     上（status / residuals / frozen_reason），干净时不出这条；
#   - H5 待补证计数进健康记录（注意级残余、不冻结、status=
#     attention）。
#
# 全程离线：嵌入/判官打桩为 None，同 H1/H5 口径。

import json
import uuid

import pytest

from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso


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


def _plant(store, text, **meta_over):
    """账本 add + SQLite 行成对种一行（账本一致的活行）。"""
    created_at = _now_iso()
    meta = {
        "source": "user", "confidence": 1.0, "user_confirmed": True,
        "created_at": created_at, "write_seq": 1, "turn_id": None,
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "episodic", "importance": "normal",
        "event_date": None,
    }
    meta.update(meta_over)
    store._ledger_append_add(text, meta)
    key = _ledger.ledger_event_key(text, created_at)
    store._insert_memory_row(key, text, meta, None)
    return key


def _prop(action="demote", memory_id=None, source="owner", key=None,
          **extra):
    p = {
        "action": action,
        "memory_id": memory_id,
        "reason": "测试提案",
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


def _enable(store):
    store.controlled_entry_execution_enabled = True
    return store


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


def _restart(store):
    """模拟进程重启：同一目录新建一个 store（连接独立）。"""
    return SQLiteStore(memory_dir=str(store.memory_dir))


def _gap_scenario(store):
    """种三行、经正门降级第一行，再删掉第二行的账本 add 事件
    （中间行）造序号缺口。返回 (mid_a, mid_c, demote回执,
    完好账本字节)。"""
    mid_a = _add_user(store, "用户每天定投五块钱买基金")
    _add_user(store, "用户喜欢吃火锅")
    mid_c = _add_user(store, "用户住在成都")
    _enable(store)
    receipt = store.submit_controlled_proposal(_prop(
        memory_id=mid_a, session=_session(store),
        expected_version=store.controlled_entry_version(mid_a),
    ))
    assert receipt["status"] == "executed", receipt
    good = store.ledger_path.read_bytes()
    events, _info = _ledger.read_events(store.ledger_path)
    b_events = [
        e for e in events
        if e.get("text") == "用户喜欢吃火锅"
        and isinstance(e.get("seq"), int)
    ]
    assert len(b_events) == 1, b_events
    seq_b = b_events[0]["seq"]
    lines = store.ledger_path.read_text(encoding="utf-8").splitlines()
    kept, dropped = [], 0
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            ev = {}
        if ev.get("seq") == seq_b:
            dropped += 1
            continue
        kept.append(line)
    assert dropped == 1
    store.ledger_path.write_text(
        "\n".join(kept) + "\n", encoding="utf-8"
    )
    return mid_a, mid_c, receipt, good


# ---------- 落盘：干净恢复 ----------

def test_clean_recovery_persists_record_and_readable_after_restart(
    store, tmp_path
):
    _plant(store, "用户养了一只猫")
    report = store.startup_recovery()
    assert report["verify"]["ok"] is True
    rec = store.recovery_health()
    assert rec is not None
    assert rec["status"] == "clean" and rec["clean"] is True
    assert rec["source"] == "startup_recovery"
    assert rec["recovery"]["verify"]["ok"] is True
    assert rec["recovery"]["projection"]["rows_checked"] == 1
    assert rec["controlled_entry"]["frozen"] is False
    assert (
        rec["controlled_entry"]["protection_evidence_pending_count"]
        == 0
    )
    assert rec["residuals"] == []
    # 干净时结论页不出 recovery_health 条（什么都不报）
    assert store.pending_notices() == []
    # 落盘形状：latest 与内存记录一致
    path = tmp_path / "memory" / "recovery_health.json"
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["latest"]["recorded_at"] == rec["recorded_at"]
    assert on_disk["latest"]["status"] == "clean"
    # 重启后新实例不跑恢复也读得出上一份、且不承接冻结
    store2 = _restart(store)
    rec2 = store2.recovery_health()
    assert rec2 is not None
    assert rec2["recorded_at"] == rec["recorded_at"]
    assert store2._controlled_frozen_reason is None


def test_history_accumulates_records(store, tmp_path):
    _plant(store, "用户养了一只猫")
    store.startup_recovery()
    store.startup_recovery()
    data = json.loads(
        (tmp_path / "memory" / "recovery_health.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(data["history"]) == 2
    assert data["latest"] == data["history"][0]


def test_corrupt_health_file_does_not_break_store(store, tmp_path):
    _plant(store, "用户养了一只猫")
    (tmp_path / "memory" / "recovery_health.json").write_text(
        "[坏", encoding="utf-8"
    )
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._recovery_health_load_error
    assert store2.recovery_health() is None
    store2.startup_recovery()
    rec = store2.recovery_health()
    assert rec["status"] == "clean"


# ---------- 冻结：账本缺口 ----------

def test_ledger_gap_freezes_execution_and_revoke_but_not_shadow(store):
    mid_a, mid_c, receipt, _good = _gap_scenario(store)
    store2 = _restart(store)
    report = store2.startup_recovery()
    assert report["verify"]["ok"] is False
    rec = store2.recovery_health()
    assert rec["status"] == "frozen"
    assert rec["controlled_entry"]["frozen"] is True
    assert rec["controlled_entry"]["frozen_reason"] == "recovery_unclean"
    kinds = [r["kind"] for r in rec["residuals"]]
    assert "ledger_verify_failed" in kinds
    # 执行被拒
    _enable(store2)
    out = store2.submit_controlled_proposal(_prop(
        memory_id=mid_c, session=_session(store2),
        expected_version=store2.controlled_entry_version(mid_c),
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "entry_frozen"
    assert out["detail"]["frozen_reason"] == "recovery_unclean"
    # 撤销执行同样被拒
    rev = store2.revoke_controlled_change(_revoke_req(
        op_id=receipt["detail"]["op_id"], session=_session(store2)
    ))
    assert rev["status"] == "refused"
    assert rev["reason"] == "entry_frozen"
    # 影子照常出回执
    store2.controlled_entry_execution_enabled = False
    shadow = store2.submit_controlled_proposal(_prop(memory_id=mid_c))
    assert shadow["status"] == "shadow"


def test_freeze_carried_across_restart_and_clean_recovery_unfreezes(
    store
):
    _mid_a, _mid_c, _receipt, good = _gap_scenario(store)
    store2 = _restart(store)
    store2.startup_recovery()
    assert store2._controlled_frozen_reason == "recovery_unclean"
    # 新实例还没跑恢复：冻结靠落盘记录承接
    store3 = _restart(store)
    assert store3._controlled_frozen_reason == "recovery_unclean"
    kinds = [n["kind"] for n in store3.pending_notices()]
    assert "recovery_health" in kinds
    # 账本修回原样 → 一次干净恢复解冻、留痕
    store3.ledger_path.write_bytes(good)
    store3.startup_recovery()
    assert store3._controlled_frozen_reason is None
    rec = store3.recovery_health()
    assert rec["status"] == "clean", rec
    assert rec["unfrozen"]["previous_reason"] == "recovery_unclean"


def test_recheck_unfreezes_recovery_freeze(store):
    _mid_a, _mid_c, _receipt, good = _gap_scenario(store)
    store2 = _restart(store)
    store2.startup_recovery()
    assert store2._controlled_frozen_reason == "recovery_unclean"
    store2.ledger_path.write_bytes(good)
    rec = store2.recovery_health_recheck()
    assert rec["source"] == "recheck"
    assert rec["controlled_entry"]["frozen"] is False
    assert store2._controlled_frozen_reason is None
    assert rec["status"] == "clean", rec


def test_recheck_does_not_unfreeze_persistence_freeze(store, tmp_path):
    _add_user(store, "用户喜欢跑步")
    store._controlled_receipts_path.write_text(
        "[1,2", encoding="utf-8"
    )
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    assert store2._controlled_frozen_reason == "receipts_load_error"
    rec = store2.recovery_health_recheck()
    assert rec["controlled_entry"]["frozen"] is True
    assert (
        rec["controlled_entry"]["frozen_reason"]
        == "receipts_load_error"
    )
    assert store2._controlled_frozen_reason == "receipts_load_error"


# ---------- 冻结：投影与账本对不上 ----------

def test_extra_row_projection_divergence_freezes(store):
    mid = _plant(store, "用户养了一只猫")
    created_at = _now_iso()
    orphan_meta = {
        "source": "user", "confidence": 1.0, "user_confirmed": True,
        "created_at": created_at, "write_seq": 1, "turn_id": None,
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "episodic", "importance": "normal",
        "event_date": None,
    }
    # 只写 SQLite 行、不写账本：投影里有、账本里没有的行
    store._insert_memory_row(
        "orphan-row-1", "来历不明的一行", orphan_meta, None
    )
    report = store.startup_recovery()
    assert report["projection"]["extra_rows"] == 1
    rec = store.recovery_health()
    assert rec["status"] == "frozen"
    kinds = [r["kind"] for r in rec["residuals"]]
    assert "projection_extra_rows" in kinds
    _enable(store)
    out = store.submit_controlled_proposal(_prop(
        action="mark_dormant", memory_id=mid,
        session=_session(store),
        expected_version=store.controlled_entry_version(mid),
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "entry_frozen"
    store.controlled_entry_execution_enabled = False
    shadow = store.submit_controlled_proposal(
        _prop(action="mark_dormant", memory_id=mid)
    )
    assert shadow["status"] == "shadow"


# ---------- 结论页映射 ----------

def test_pending_notices_mapping_fields(store):
    _gap_scenario(store)
    store2 = _restart(store)
    store2.startup_recovery()
    notices = store2.pending_notices()
    health = [n for n in notices if n["kind"] == "recovery_health"]
    assert len(health) == 1
    h = health[0]
    assert h["status"] == "frozen"
    assert "ledger_verify_failed" in h["residuals"]
    assert h["frozen_reason"] == "recovery_unclean"
    assert h["at"] == store2.recovery_health()["recorded_at"]
    frozen = [
        n for n in notices if n["kind"] == "controlled_entry_frozen"
    ]
    assert frozen and frozen[0]["reason"] == "recovery_unclean"


# ---------- H5 待补证计数进健康记录 ----------

def test_protection_evidence_pending_count_in_record(store, tmp_path):
    mid = _add_user(store, "用户的伞是蓝色")
    # 把证据路径指到一个目录：追加必失败（同 H5 测试造法）
    blocker = tmp_path / "evidence_blocker"
    blocker.mkdir()
    store._protection_evidence_path = blocker
    out = store.set_protection_tier(mid, "pinned")
    assert out["protection_evidence"]["pending"] is True
    store.startup_recovery()
    rec = store.recovery_health()
    assert (
        rec["controlled_entry"]["protection_evidence_pending_count"]
        == 1
    )
    kinds = [r["kind"] for r in rec["residuals"]]
    assert "protection_evidence_pending" in kinds
    # 注意级：不冻结
    assert rec["status"] == "attention", rec
    assert rec["controlled_entry"]["frozen"] is False


# ---------- 待问损坏＋重建失败 → 冻结危险写（故障注入补测） ----------

def test_pending_corrupt_and_rebuild_failure_freezes(store, monkeypatch):
    """待问文件损坏且重建没跑成 → pending_questions_unrecovered
    冻结级残余 → 恢复健康度 frozen、受控入口执行被拒
    entry_frozen、影子照常（H6 判据「待问损坏且重建没跑成」
    的直接故障注入；此前只有「损坏但重建成功→attention」
    与其他冻结来源的用例，这条冻结路径没被直接打过）。"""
    mid = _add_user(store, "用户每天定投五块钱买基金")
    # 待问文件损坏：新实例构造时保全原件并记 load_error。
    pq_path = store.memory_dir / "pending_questions.json"
    pq_path.write_text("garbage!!!", encoding="utf-8")
    store2 = _restart(store)
    assert store2.pending_questions.recovery_report() is not None
    # 故障注入：重建这一步抛错（模拟重建没跑成）。
    def _boom(self):
        raise RuntimeError("待问重建失败（测试注入的故障）。")
    monkeypatch.setattr(SQLiteStore, "rebuild_pending_questions", _boom)
    report = store2.startup_recovery()
    assert report["pending"] is not None
    assert report["pending_rebuilt"] is None
    rec = store2.recovery_health()
    assert rec["status"] == "frozen", rec
    kinds = {r["kind"]: r for r in rec["residuals"]}
    assert kinds["pending_questions_unrecovered"]["freezing"] is True
    assert rec["controlled_entry"]["frozen"] is True
    assert rec["controlled_entry"]["frozen_reason"] == "recovery_unclean"
    # 危险写被拒：执行路 entry_frozen。
    _enable(store2)
    out = store2.submit_controlled_proposal(_prop(
        memory_id=mid, session=_session(store2),
        expected_version=store2.controlled_entry_version(mid),
    ))
    assert out["status"] == "refused", out
    assert out["reason"] == "entry_frozen"
    assert out["detail"]["frozen_reason"] == "recovery_unclean"
    # 影子照常出回执。
    store2.controlled_entry_execution_enabled = False
    shadow = store2.submit_controlled_proposal(_prop(memory_id=mid))
    assert shadow["status"] == "shadow"
