# tests/test_tx_recovery_fixes_20261125.py（崩溃恢复互查修复，2026-11-25）
#
# 贵妃互查崩溃恢复（251f5a6）指出、十一核实成立的问题，逐个补回归：
#   修复一：repair_projection_from_ledger 写 slot 是「旧值非空就不
#     覆盖」，与同函数其他字段「按最新一笔事件覆盖」不一致——账本
#     后来改了槽位，投影永远盖不上。改无条件覆盖后钉住三种情形。
#   修复二：pending_questions 的 questions 键缺失 / 值不是 list
#     （{}、""、0、False）被 data.get(...) or [] 静默吞成空队列，
#     绕过 bad_shape 保全+报警，之后 _save 还会把原坏文件覆盖掉。
#   修复三：repair 对齐 deleted/superseded 不对称——账本终态是
#     deleted 时不清 superseded，能留下双标行。改按终态精确对齐。
#   (a) _preserve_corrupt 改名失败时原件无副本，后续 _save 照样
#     覆盖：改名失败先试复制保全；都没保住时 _save 拒绝覆盖。
#   (b) startup_recovery 的待问重建跑在 reconcile 补缺之前，账本
#     里有、SQLite 里缺的 auth_claim 行当时不存在，重建漏掉它的
#     问题：auth_claim 标记先进账本 add（不然捞回来的行没标记），
#     create_sqlite_store 再在补缺后补跑一遍重建。
#
# 全部离线：向量/判官打桩成 None（同 test_tx_recovery 口径）；
# 种行走 _plant（账本 add + SQLite 行成对写）。

import json
import os
import shutil
from pathlib import Path

import pytest

from memory import auth_guard as _auth_guard
from memory import ledger as _ledger
from memory.pending_questions import PendingQuestions
from memory.sqlite_store import SQLiteStore, create_sqlite_store
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


def _flags(store, memory_id):
    row = store._conn.execute(
        "SELECT superseded, deleted, voided, decay_kind, metadata_json"
        " FROM memories WHERE id=?",
        (memory_id,),
    ).fetchone()
    assert row is not None, memory_id
    return {
        "superseded": row[0], "deleted": row[1], "voided": row[2],
        "decay_kind": row[3], "meta": json.loads(row[4] or "{}"),
    }


def _slot_event(store, key, slot):
    _ledger.append_event(
        store.ledger_path,
        {"type": _ledger.LEDGER_EVENT_SLOT, "key": key,
         "memory_id": key, "slot": slot},
        lock=store._ledger_lock,
    )


# ---------- 修复一：slot 按最新 SLOT 事件无条件覆盖 ----------

def test_slot_overlay_overrides_stale_meta_slot(store):
    mid = _plant(store, "我习惯把钥匙挂在玄关的钩子上", slot="slot_alpha")
    _slot_event(store, mid, "slot_beta")
    report = store.repair_projection_from_ledger()
    # 账本最新一笔说 slot_beta：meta 和 slot 列都得跟上、计数 +1。
    assert _flags(store, mid)["meta"].get("slot") == "slot_beta"
    col = store._conn.execute(
        "SELECT slot FROM memories WHERE id=?", (mid,)
    ).fetchone()[0]
    assert col == "slot_beta"
    assert report["meta_fixed"] == 1


def test_slot_overlay_fills_empty_meta_slot(store):
    mid = _plant(store, "我习惯把钥匙挂在玄关的钩子上")
    _slot_event(store, mid, "slot_beta")
    report = store.repair_projection_from_ledger()
    assert _flags(store, mid)["meta"].get("slot") == "slot_beta"
    assert report["meta_fixed"] == 1


def test_slot_overlay_without_event_keeps_meta_slot(store):
    mid = _plant(store, "我习惯把钥匙挂在玄关的钩子上", slot="slot_alpha")
    report = store.repair_projection_from_ledger()
    # 没有 SLOT 事件就不许动：保持原行为。
    assert _flags(store, mid)["meta"].get("slot") == "slot_alpha"
    assert report["meta_fixed"] == 0


# ---------- 修复三：deleted/superseded 按折叠终态精确对齐 ----------

def test_repair_deleted_state_clears_superseded_flag(store):
    mid = _plant(store, "周五下午三点有一个会议")
    # 半截状态：SQLite 还挂着 superseded，账本终态已经是 deleted。
    store._conn.execute(
        "UPDATE memories SET superseded=1 WHERE id=?", (mid,)
    )
    store._conn.commit()
    _ledger.append_event(
        store.ledger_path,
        {"type": _ledger.LEDGER_EVENT_DELETE, "memory_id": mid,
         "key": mid, "text": "周五下午三点有一个会议", "meta": {}},
        lock=store._ledger_lock,
    )
    report = store.repair_projection_from_ledger()
    flags = _flags(store, mid)
    assert flags["deleted"] == 1 and flags["superseded"] == 0
    assert report["deleted_fixed"] == 1
    assert report["superseded_fixed"] == 1


def test_repair_superseded_state_clears_deleted_flag(store):
    old_id = _plant(store, "我下个月要去一趟成都，住春熙路附近")
    new_id = _plant(store, "我下个月要去一趟重庆，住春熙路附近")
    # 反方向半截：SQLite 还挂着 deleted，账本终态是 superseded
    # （这一路原写法本来就对，钉住不许改坏）。
    store._conn.execute(
        "UPDATE memories SET deleted=1 WHERE id=?", (old_id,)
    )
    store._conn.commit()
    _ledger.append_event(
        store.ledger_path,
        {
            "type": _ledger.LEDGER_EVENT_SUPERSEDE,
            "memory_id": new_id, "supersedes_id": old_id,
            "old_text": "我下个月要去一趟成都，住春熙路附近",
            "text": "我下个月要去一趟重庆，住春熙路附近",
            "key": old_id, "current_key": new_id,
        },
        lock=store._ledger_lock,
    )
    report = store.repair_projection_from_ledger()
    flags = _flags(store, old_id)
    assert flags["superseded"] == 1 and flags["deleted"] == 0
    assert report["superseded_fixed"] == 1
    assert report["deleted_fixed"] == 1


# ---------- 修复二：pending 坏形状五种都不许静默吞 ----------

BAD_SHAPES = [
    '{"questions": {}}',
    '{"questions": ""}',
    '{"questions": 0}',
    '{"questions": false}',
    '{"other": 1}',
]


@pytest.mark.parametrize("content", BAD_SHAPES)
def test_pending_bad_shape_variants_preserved(tmp_path, content):
    path = tmp_path / "pending_questions.json"
    path.write_text(content, encoding="utf-8")
    pq = PendingQuestions(path=path)
    assert pq.list_unanswered() == []
    # 五种坏形状都必须走 bad_shape 保全 + load_error，不许静默。
    assert pq.load_error is not None
    assert pq.load_error["reason"].startswith("bad_shape")
    backup = Path(pq.load_error["backup_path"])
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == content
    assert not path.exists()  # 原件已改名保全


def test_pending_bad_shape_add_keeps_corrupt_copy(tmp_path):
    content = '{"questions": {}}'
    path = tmp_path / "pending_questions.json"
    path.write_text(content, encoding="utf-8")
    pq = PendingQuestions(path=path)
    pq.add(
        "importance", "这条要不要紧",
        ["进核心", "当一次性经历", "不存"], "回归测试",
    )
    # add 之后的 _save 写了新文件，但原坏内容必须还在保全副本里，
    # 不许无副本消失（旧静默加载正是这样把原件覆盖没的）。
    backups = list(tmp_path.glob("pending_questions.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == content
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["questions"]) == 1


# ---------- (a) 保全失败的兜底：复制保全 / 拒绝覆盖 ----------

def _flaky_replace(monkeypatch):
    real_replace = os.replace

    def _replace(src, dst, *args, **kwargs):
        if "corrupt" in str(dst):
            raise OSError("模拟保全改名失败")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _replace)


def test_preserve_rename_failure_falls_back_to_copy(tmp_path, monkeypatch):
    content = '{"questions": {}}'
    path = tmp_path / "pending_questions.json"
    path.write_text(content, encoding="utf-8")
    _flaky_replace(monkeypatch)
    pq = PendingQuestions(path=path)
    assert pq.load_error is not None
    backup = Path(pq.load_error["backup_path"])
    # 改名失败但复制成功：副本在手、内容一字不差。
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == content
    assert path.exists()  # 原件动不了，留在原地
    # 有副本了，后续保存可以正常覆盖原文件，副本不动。
    pq.add(
        "importance", "这条要不要紧",
        ["进核心", "当一次性经历", "不存"], "回归测试",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["questions"]) == 1
    assert backup.read_text(encoding="utf-8") == content


def test_preserve_total_failure_refuses_save(tmp_path, monkeypatch):
    content = '{"questions": {}}'
    path = tmp_path / "pending_questions.json"
    path.write_text(content, encoding="utf-8")
    _flaky_replace(monkeypatch)

    def _no_copy(src, dst, *args, **kwargs):
        raise OSError("模拟复制保全失败")

    monkeypatch.setattr(shutil, "copyfile", _no_copy)
    pq = PendingQuestions(path=path)
    assert pq.load_error is not None
    assert pq.load_error["backup_path"] is None
    pq.add(
        "importance", "这条要不要紧",
        ["进核心", "当一次性经历", "不存"], "回归测试",
    )
    # 问题先攒在内存里，但盘上的坏原件一个字节都不许动。
    assert len(pq.list_unanswered()) == 1
    assert path.read_text(encoding="utf-8") == content
    assert list(tmp_path.glob("pending_questions.json.corrupt-*")) == []


# ---------- (b) 组合：行靠 reconcile 捞回后，auth_claim 问题不漏 ----------

def test_create_store_rebuilds_auth_claim_after_reconcile(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    memory_dir = str(tmp_path / "memory")
    s1 = SQLiteStore(memory_dir=memory_dir)
    s1.organizing_router = lambda: None
    mid = _plant(
        s1, "我把备用钥匙放在门口花盆下面",
        auth_claim=_auth_guard.AUTH_CLAIM_RELAYED,
    )
    # SQLite 行丢了（账本还在）+ 待问文件损坏：旧顺序下重建跑在
    # 补缺之前，扫不到这行，问题永远漏掉。
    s1._conn.execute("DELETE FROM memories WHERE id=?", (mid,))
    s1._conn.commit()
    pq_path = Path(memory_dir) / "pending_questions.json"
    pq_path.write_text("garbage!!!", encoding="utf-8")
    s2 = create_sqlite_store(memory_dir=memory_dir)
    assert s2 is not None
    # 行被 reconcile 按账本捞回，auth_claim 标记也得在（标记已随
    # add 进账本，不然捞回来的行没标记、重建照样找不到）。
    assert (
        _flags(s2, mid)["meta"].get("auth_claim")
        == _auth_guard.AUTH_CLAIM_RELAYED
    )
    auth_qs = [
        q for q in s2.pending_questions.list_unanswered()
        if q["kind"] == "auth_claim"
    ]
    assert len(auth_qs) == 1 and auth_qs[0]["memory_id"] == mid
