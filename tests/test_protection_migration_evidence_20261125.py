# tests/test_protection_migration_evidence_20261125.py
# （立项 H5：保护迁移完成证据，2026-11-25）
#
# 覆盖口径（施工清单 H5＋契约二）：
#   - 经宿主直调 set_protection_tier 迁移产当场证据且字段全
#     （迁移前后档位、来源、账本 seq、op_id、前后版本戳、
#     原文哈希），evidence_id 与账本 seq 绑死（pme-<seq>）；
#   - 经受控入口 set_protection_tier 迁移产证据且带 op_id、
#     来源为提案 instruction_source；
#   - 档位没真变的调用（只改抽屉、同档重设）不产证据；
#   - list_protection_events 做真：按 memory_id 查历史按
#     seq 升序、全库列表与 limit 口径、web 无参调用形状兼容；
#   - H5 之前的迁移（手工补账本 PROMOTE、无当场证据）也
#     列得出，标 evidence_recorded=False，单份证据可按
#     pme-<seq> 从账本重建（reconstructed=True）；
#   - verify_protection_migration：刚迁移完对得上（ok）；
#     人为绕账本改投影档位核对出 projection_tier_mismatch；
#     旧证据被后续迁移顶替时注明 superseded_by、不算不一致；
#     缺当场证据报 evidence_record_missing、不假报完成；
#   - 证据追加不可改：后续迁移不改旧行字节，且本类没有
#     改/删证据的接口；
#   - 证据落盘失败：迁移本体照常成立（账本先行），返回如
#     实标 pending、pending_notices 露头，不假报已记录。
#
# 全程离线：嵌入/判官打桩为 None，同 H1 测试口径。

import hashlib
import json
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


def _prop(action="set_protection_tier", memory_id=None,
          source="owner", key=None, **extra):
    p = {
        "action": action,
        "memory_id": memory_id,
        "reason": "H5 测试提案",
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


def _exec_prop(store, action="set_protection_tier", memory_id=None,
               **extra):
    extra.setdefault("session", _session(store))
    extra.setdefault(
        "expected_version", store.controlled_entry_version(memory_id)
    )
    return _prop(action=action, memory_id=memory_id, **extra)


def _evidence_records(store):
    records, bad = store._read_protection_evidence()
    assert bad == 0
    return records


# ---------- 当场证据：宿主直调 ----------

def test_direct_tier_change_records_full_evidence(store):
    mid = _add_user(store, "用户的生日是五月一号")
    out = store.set_protection_tier(mid, "pinned")
    assert out["ok"] is True
    pe = out["protection_evidence"]
    assert pe["recorded"] is True and pe["pending"] is False

    records = _evidence_records(store)
    assert len(records) == 1
    ev = records[0]
    assert ev["evidence_id"] == pe["evidence_id"]
    assert ev["memory_id"] == mid
    assert ev["tier_before"] is None
    assert ev["tier_after"] == "pinned"
    assert ev["source"] == "host_direct"
    assert ev["op_id"] is None and ev["idem_key"] is None
    assert ev["version_before"] and ev["version_after"]
    assert ev["version_before"] != ev["version_after"]
    assert ev["text_hash"] == hashlib.sha256(
        "用户的生日是五月一号".encode("utf-8")
    ).hexdigest()
    # evidence_id 与账本 seq 绑死：账本里那笔 PROMOTE 对得上
    assert ev["evidence_id"] == "pme-%s" % ev["ledger_seq"]
    events, _info = _ledger.read_events(store.ledger_path)
    mine = [
        e for e in events
        if e.get("seq") == ev["ledger_seq"]
    ]
    assert len(mine) == 1
    assert mine[0].get("type") == "promote"
    assert mine[0].get("memory_id") == mid
    assert (
        mine[0].get("meta", {}).get("protection_tier") == "pinned"
    )
    # 单份取证据：完整版、非重建
    got = store.get_protection_migration_evidence(pe["evidence_id"])
    assert got["reconstructed"] is False
    assert got["tier_after"] == "pinned"
    assert got["version_after"] == ev["version_after"]


def test_no_tier_change_no_evidence(store):
    mid = _add_user(store, "用户每月工资一号到账")
    # 只改抽屉：档位没动，不产证据、返回不带证据键
    out = store.set_core_drawer(mid, "person")
    assert out["ok"] is True
    assert "protection_evidence" not in out
    # 同档重设（None -> None）：没迁移，不产证据
    out2 = store.set_protection_tier(mid, None)
    assert out2["ok"] is True
    assert "protection_evidence" not in out2
    assert _evidence_records(store) == []
    assert store.list_protection_events(memory_id=mid) == []


# ---------- 当场证据：受控入口 ----------

def test_controlled_entry_tier_change_evidence_with_op_id(store):
    mid = _add_user(store, "用户的护照放在抽屉里")
    store.controlled_entry_execution_enabled = True
    receipt = store.submit_controlled_proposal(
        _exec_prop(store, memory_id=mid, params={"tier": "pinned"})
    )
    assert receipt["status"] == "executed", receipt
    records = [
        r for r in _evidence_records(store)
        if r["memory_id"] == mid
    ]
    assert len(records) == 1
    ev = records[0]
    assert ev["tier_after"] == "pinned"
    assert ev["source"] == "owner"
    assert ev["op_id"]
    # op_id 与账本那笔 PROMOTE 三方对照
    events, _info = _ledger.read_events(store.ledger_path)
    mine = [e for e in events if e.get("seq") == ev["ledger_seq"]]
    assert len(mine) == 1
    assert mine[0].get("op_id") == ev["op_id"]
    assert mine[0].get("idem_key") == ev["idem_key"]


# ---------- list_protection_events 做真 ----------

def test_history_order_filter_and_limit(store):
    a = _add_user(store, "用户住在广州")
    b = _add_user(store, "用户养了一只猫")
    store.set_protection_tier(a, "temp_pinned")
    store.set_protection_tier(a, "pinned")
    store.set_protection_tier(b, "pinned")
    store.set_protection_tier(a, None)

    hist_a = store.list_protection_events(memory_id=a)
    assert [
        (h["tier_before"], h["tier_after"]) for h in hist_a
    ] == [
        (None, "temp_pinned"),
        ("temp_pinned", "pinned"),
        ("pinned", None),
    ]
    seqs = [h["ledger_seq"] for h in hist_a]
    assert seqs == sorted(seqs)
    assert all(h["evidence_recorded"] for h in hist_a)

    hist_all = store.list_protection_events()
    assert len(hist_all) == 4
    last2 = store.list_protection_events(limit=2)
    assert [h["ledger_seq"] for h in last2] == [
        h["ledger_seq"] for h in hist_all[-2:]
    ]
    hist_b = store.list_protection_events(memory_id=b)
    assert len(hist_b) == 1
    assert hist_b[0]["tier_after"] == "pinned"


def test_list_protection_events_web_shape(store):
    # web/server.py 无参调用：返回 list、键齐全（原桩返回 []）
    mid = _add_user(store, "用户的车是白色")
    store.set_protection_tier(mid, "temp_pinned")
    events = store.list_protection_events()
    assert isinstance(events, list) and len(events) == 1
    for key in (
        "evidence_id", "memory_id", "tier_before", "tier_after",
        "ledger_seq", "ts", "op_id", "source", "evidence_recorded",
    ):
        assert key in events[0], key


# ---------- H5 之前的迁移：派生与重建 ----------

def _legacy_migration(store, mid, tier):
    """模拟 H5 之前的迁移：账本 PROMOTE＋投影同步，无当场证据。"""
    row = _row(store, mid)
    new_meta = dict(row["meta"])
    new_meta["protection_tier"] = tier
    record = _ledger.append_event(
        store.ledger_path,
        {
            "type": _ledger.LEDGER_EVENT_PROMOTE,
            "memory_id": mid,
            "key": mid,
            "text": row["text"],
            "meta": new_meta,
        },
        lock=store._ledger_lock,
    )
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (json.dumps(new_meta, ensure_ascii=False), mid),
    )
    store._conn.commit()
    return record["seq"]


def test_legacy_migration_listed_and_reconstructed(store):
    mid = _add_user(store, "用户的老家在西安")
    seq = _legacy_migration(store, mid, "pinned")
    hist = store.list_protection_events(memory_id=mid)
    assert len(hist) == 1
    entry = hist[0]
    assert entry["evidence_id"] == "pme-%s" % seq
    assert entry["tier_before"] is None
    assert entry["tier_after"] == "pinned"
    assert entry["evidence_recorded"] is False
    # 单份证据按 seq 从账本重建：如实标 reconstructed
    got = store.get_protection_migration_evidence("pme-%s" % seq)
    assert got["reconstructed"] is True
    assert got["tier_after"] == "pinned"
    assert got["version_after"] is None
    # 对不上的证据号返回 None
    assert store.get_protection_migration_evidence("pme-999999") is None
    assert store.get_protection_migration_evidence(None) is None


# ---------- verify_protection_migration ----------

def test_verify_ok_after_migration(store):
    mid = _add_user(store, "用户每天定投五块钱")
    store.set_protection_tier(mid, "pinned")
    out = store.verify_protection_migration(memory_id=mid)
    assert out["ok"] is True, out
    assert out["mismatches"] == []
    assert out["checks"]["evidence_recorded"] is True
    assert out["checks"]["evidence_matches_ledger"] is True
    assert out["checks"]["projection_matches_ledger"] is True
    assert out["checks"]["version_stamp_matches"] is True
    # 按 evidence_id 核对同一笔，结论一致
    out2 = store.verify_protection_migration(
        evidence_id=out["evidence_id"]
    )
    assert out2["ok"] is True
    assert out2["memory_id"] == mid


def test_verify_detects_projection_tamper(store):
    mid = _add_user(store, "用户的工牌号不告诉别人")
    store.set_protection_tier(mid, "pinned")
    # 人为绕账本改投影：投影档位与账本链分叉
    row = _row(store, mid)
    bad_meta = dict(row["meta"])
    bad_meta["protection_tier"] = "temp_pinned"
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (json.dumps(bad_meta, ensure_ascii=False), mid),
    )
    store._conn.commit()
    out = store.verify_protection_migration(memory_id=mid)
    assert out["ok"] is False
    assert "projection_tier_mismatch" in out["mismatches"]
    assert out["checks"]["projection_tier"] == "temp_pinned"
    assert out["checks"]["ledger_tier"] == "pinned"


def test_verify_old_evidence_superseded_not_mismatch(store):
    mid = _add_user(store, "用户周末去爬山")
    first = store.set_protection_tier(mid, "pinned")
    second = store.set_protection_tier(mid, "temp_pinned")
    out = store.verify_protection_migration(
        evidence_id=first["protection_evidence"]["evidence_id"]
    )
    # 第一笔确实落了账：核对通过，后续迁移只注明不算不一致
    assert out["ok"] is True, out
    assert out["detail"]["superseded_by"] == (
        second["protection_evidence"]["evidence_id"]
    )


def test_verify_legacy_migration_reports_missing_evidence(store):
    mid = _add_user(store, "用户的书桌靠窗")
    seq = _legacy_migration(store, mid, "pinned")
    out = store.verify_protection_migration(
        evidence_id="pme-%s" % seq
    )
    # 迁移本体对得上，但当场证据缺失：不假报完成
    assert out["ok"] is False
    assert "evidence_record_missing" in out["mismatches"]
    assert out["checks"]["projection_matches_ledger"] is True
    assert out["checks"]["evidence_matches_ledger"] is True


def test_verify_no_migration_memory(store):
    mid = _add_user(store, "用户喜欢喝茶")
    out = store.verify_protection_migration(memory_id=mid)
    assert out["ok"] is True, out
    assert out["evidence_id"] is None
    assert out["detail"]["no_migration"] is True
    # 两个参数都不给：如实报缺目标
    out2 = store.verify_protection_migration()
    assert out2["ok"] is False
    assert "missing_target" in out2["mismatches"]


# ---------- 证据追加不可改 ----------

def test_evidence_append_only_and_no_mutation_api(store):
    mid = _add_user(store, "用户的钥匙挂在门后")
    store.set_protection_tier(mid, "temp_pinned")
    path = store._protection_evidence_path
    lines_before = path.read_text(encoding="utf-8").splitlines()
    assert len(lines_before) == 1
    store.set_protection_tier(mid, "pinned")
    store.set_protection_tier(mid, None)
    lines_after = path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == 3
    # 旧行字节不动：只许追加
    assert lines_after[:1] == lines_before
    # 本类没有改/删证据的接口
    for name in (
        "update_protection_evidence",
        "delete_protection_evidence",
        "remove_protection_evidence",
        "clear_protection_evidence",
    ):
        assert not hasattr(store, name), name


# ---------- 证据落盘失败：不假报 ----------

def test_evidence_write_failure_marks_pending(store, tmp_path):
    mid = _add_user(store, "用户的伞是蓝色")
    # 把证据路径指到一个目录：追加必失败
    blocker = tmp_path / "evidence_blocker"
    blocker.mkdir()
    store._protection_evidence_path = blocker
    out = store.set_protection_tier(mid, "pinned")
    # 迁移本体照常成立（账本＋投影一致），但证据标待补证
    assert out["ok"] is True
    pe = out["protection_evidence"]
    assert pe["recorded"] is False and pe["pending"] is True
    assert _core_cap.normalize_protection_tier(
        _row(store, mid)["meta"]
    ) == "pinned"
    notices = store.pending_notices()
    assert any(
        n.get("kind") == "protection_evidence_pending"
        and n.get("evidence_id") == pe["evidence_id"]
        for n in notices
    )
    # 核对函数同样把缺证列出来，不假报完成
    out2 = store.verify_protection_migration(memory_id=mid)
    assert out2["ok"] is False
    assert "evidence_record_missing" in out2["mismatches"]
