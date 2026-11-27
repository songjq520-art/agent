# tests/test_tx_recovery_20261125.py（崩溃恢复地基，追加决定十五）
#
# 覆盖备忘《correction_tx_evidence_memo》材料二列的崩溃恢复缺口：
#   1. repair_ledger 只被 compact_ledger 调用、verify_ledger 无调用者
#      → 两个函数本身补单元测试；启动路径接入由 store 级测试覆盖；
#   2. supersede 两步之间硬崩溃（SQLite 已标、账本没写）无修复
#      → 用 BaseException 注入模拟进程当场死亡（回滚只接 Exception，
#        拦不住），重启后按账本撤标记；
#   3. reconcile 只补缺行不修已有行 → 账本先行的各路（delete /
#      promote / dormant / revive）只写完账本那步就「崩」，重启后
#      按账本把投影补齐；
#   4. pending_questions.json 损坏静默清零 → 原件保全 + load_error
#      报警 + 按存活记录（auth_claim 行标记 / intake_log 留痕）重建；
#   5. 账本里没有的行：不删不改、只计数报警。
#
# 全部离线：向量/判官打桩成 None（同 sameturn/g7 口径）；种行走
# _plant（账本 add + SQLite 行成对写，不惊动入口/槽位管道）。

import json
from pathlib import Path

import pytest

from memory import auth_guard as _auth_guard
from memory import intake_gate as _intake
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


def _restart(store):
    """模拟进程重启：同一目录新建一个 store（连接独立）。"""
    return SQLiteStore(memory_dir=str(store.memory_dir))


# ---------- 账本层：verify_ledger / repair_ledger 单元测试 ----------

def _append_add(path, text):
    return _ledger.append_event(
        path,
        {
            "type": _ledger.LEDGER_EVENT_ADD,
            "key": _ledger.ledger_event_key(text, "2026-11-25T00:00:00Z"),
            "text": text,
            "meta": {},
        },
    )


def test_verify_ledger_clean_then_gap(tmp_path):
    path = tmp_path / "memory_events.jsonl"
    for t in ("甲", "乙", "丙"):
        _append_add(path, t)
    report = _ledger.verify_ledger(path)
    assert report["ok"] is True
    assert report["events"] == 3 and report["seq_max"] == 3
    # 人为删掉中间一行：序号缺口必须被查出来（以前没人调这个函数）。
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    report = _ledger.verify_ledger(path)
    assert report["ok"] is False
    assert report["gaps"] == [2]


def test_repair_ledger_drops_truncated_tail_then_append_continues(tmp_path):
    path = tmp_path / "memory_events.jsonl"
    _append_add(path, "甲")
    _append_add(path, "乙")
    # 写一半的残行：没有换行结尾（记账本中间进程死了）。
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "add", "key": "half')
    events, info = _ledger.read_events(path)
    assert info["truncated"] is True and len(events) == 2
    assert _ledger.verify_ledger(path)["ok"] is False
    result = _ledger.repair_ledger(path)
    assert result == {"repaired": True, "dropped": 1}
    # 修完再追加：序号连续、不许把新行粘到残行上写成坏行。
    _append_add(path, "丙")
    events, info = _ledger.read_events(path)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert info["bad_lines"] == 0 and info["truncated"] is False
    assert _ledger.verify_ledger(path)["ok"] is True


def test_repair_ledger_noop_when_clean(tmp_path):
    path = tmp_path / "memory_events.jsonl"
    _append_add(path, "甲")
    assert _ledger.repair_ledger(path) == {"repaired": False, "dropped": 0}


# ---------- 崩溃窗口一：supersede 两步之间硬崩溃 ----------

class _SimulatedCrash(BaseException):
    """模拟进程当场死亡：不是 Exception，回滚代码接不住。"""


def test_half_supersede_sqlite_mark_without_ledger_event(store, monkeypatch):
    old_id = _plant(store, "我下个月要去一趟成都，住春熙路附近")
    new_id = _plant(store, "我下个月要去一趟重庆，住春熙路附近")

    real_append = _ledger.append_event

    def _crash_append(*args, **kwargs):
        raise _SimulatedCrash()

    monkeypatch.setattr(_ledger, "append_event", _crash_append)
    with pytest.raises(_SimulatedCrash):
        store._supersede_row(
            old_id, "我下个月要去一趟成都，住春熙路附近", {}, new_id,
            "我下个月要去一趟重庆，住春熙路附近",
        )
    monkeypatch.setattr(_ledger, "append_event", real_append)
    # 崩溃残局：SQLite 已标 superseded，账本里没有这笔 supersede。
    assert _flags(store, old_id)["superseded"] == 1
    # 重启恢复：按账本（没这笔事件 = 没顶替过）撤标记，记忆不丢。
    store2 = _restart(store)
    report = store2.startup_recovery()
    assert report["projection"]["superseded_fixed"] == 1
    assert _flags(store2, old_id)["superseded"] == 0
    assert _flags(store2, new_id)["superseded"] == 0


def test_half_supersede_ledger_event_without_sqlite_mark(store):
    old_id = _plant(store, "我下个月要去一趟成都，住春熙路附近")
    new_id = _plant(store, "我下个月要去一趟重庆，住春熙路附近")
    # 反方向残局：账本事件已落、SQLite 标记没写（进程死在事件后、
    # 标记前，或库是从旧备份恢复的）。直接补记账本事件模拟。
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
    assert _flags(store, old_id)["superseded"] == 0
    store2 = _restart(store)
    report = store2.startup_recovery()
    assert report["projection"]["superseded_fixed"] == 1
    assert _flags(store2, old_id)["superseded"] == 1


# ---------- 崩溃窗口二：账本先行各路只写完账本 ----------

def test_half_delete_ledger_event_without_sqlite_flag(store):
    mid = _plant(store, "周五下午三点有一个会议")
    _ledger.append_event(
        store.ledger_path,
        {"type": _ledger.LEDGER_EVENT_DELETE, "memory_id": mid,
         "key": mid, "text": "周五下午三点有一个会议", "meta": {}},
        lock=store._ledger_lock,
    )
    assert _flags(store, mid)["deleted"] == 0
    store2 = _restart(store)
    report = store2.startup_recovery()
    assert report["projection"]["deleted_fixed"] == 1
    assert _flags(store2, mid)["deleted"] == 1


def test_half_promote_ledger_event_without_sqlite_meta(store):
    mid = _plant(store, "我每天定投五块钱买万家纳指")
    meta = dict(_flags(store, mid)["meta"])
    meta["decay_kind"] = "stable"
    meta["protection_tier"] = "pinned"
    _ledger.append_event(
        store.ledger_path,
        {"type": _ledger.LEDGER_EVENT_PROMOTE, "memory_id": mid,
         "key": mid, "text": "我每天定投五块钱买万家纳指", "meta": meta},
        lock=store._ledger_lock,
    )
    assert _flags(store, mid)["decay_kind"] != "stable"
    store2 = _restart(store)
    report = store2.startup_recovery()
    assert report["projection"]["meta_fixed"] == 1
    after = _flags(store2, mid)
    assert after["decay_kind"] == "stable"
    assert after["meta"]["protection_tier"] == "pinned"


def test_half_dormant_then_revive_ledger_only(store):
    mid = _plant(store, "我有一辆黑色的自行车")
    _ledger.append_event(
        store.ledger_path,
        {"type": _ledger.LEDGER_EVENT_DORMANT, "memory_id": mid,
         "key": mid, "text": "我有一辆黑色的自行车"},
        lock=store._ledger_lock,
    )
    store2 = _restart(store)
    store2.startup_recovery()
    assert _flags(store2, mid)["meta"]["decay_state"] == "dormant"
    # 再只写 REVIVE 的账本那步就「崩」：投影还挂着 dormant 牌子。
    _ledger.append_event(
        store2.ledger_path,
        {"type": _ledger.LEDGER_EVENT_REVIVE, "memory_id": mid,
         "key": mid, "text": "我有一辆黑色的自行车"},
        lock=store2._ledger_lock,
    )
    store3 = _restart(store)
    store3.startup_recovery()
    assert _flags(store3, mid)["meta"]["decay_state"] == "verbatim"


# ---------- 账本里没有的行：不删不改，只报警 ----------

def test_extra_row_flagged_not_deleted(store):
    _plant(store, "我住在广州")
    store._insert_memory_row(
        "ghost-key-not-in-ledger", "来历不明的一行",
        {"created_at": _now_iso(), "source": "user",
         "decay_kind": "episodic"},
        None,
    )
    report = store.startup_recovery()
    assert report["projection"]["extra_rows"] == 1
    row = store._conn.execute(
        "SELECT superseded, deleted FROM memories WHERE id=?",
        ("ghost-key-not-in-ledger",),
    ).fetchone()
    assert row is not None and row[0] == 0 and row[1] == 0


def test_recovery_noop_on_healthy_store(store):
    _plant(store, "我住在广州")
    report = store.startup_recovery()
    assert report["verify"]["ok"] is True
    assert report["ledger_repair"] == {"repaired": False, "dropped": 0}
    assert report["projection"] == {
        "rows_checked": 1, "superseded_fixed": 0, "deleted_fixed": 0,
        "meta_fixed": 0, "extra_rows": 0,
    }
    assert report["pending"] is None
    assert store.pending_notices() == []


# ---------- 启动路径：残行账本 + 缺行投影一次捞回 ----------

def test_create_sqlite_store_recovers_truncated_ledger_and_missing_row(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    memory_dir = str(tmp_path / "memory")
    s1 = SQLiteStore(memory_dir=memory_dir)
    mid = _plant(s1, "我住在广州")
    # 投影行没了（写索引那步崩了）+ 账本尾巴带残行。
    s1._conn.execute("DELETE FROM memories WHERE id=?", (mid,))
    s1._conn.commit()
    with open(s1.ledger_path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "add", "key": "half')
    s2 = create_sqlite_store(memory_dir=memory_dir)
    assert s2 is not None
    assert _flags(s2, mid)["superseded"] == 0
    assert _ledger.verify_ledger(s2.ledger_path)["ok"] is True


# ---------- pending_questions.json 损坏：保全 + 报警 + 重建 ----------

def test_corrupt_pending_file_preserved_and_reported(tmp_path):
    path = tmp_path / "pending_questions.json"
    path.write_text("this is not json {{{", encoding="utf-8")
    pq = PendingQuestions(path=path)
    assert pq.list_unanswered() == []
    assert pq.load_error is not None
    assert pq.load_error["reason"].startswith("bad_json")
    backup = Path(pq.load_error["backup_path"])
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "this is not json {{{"
    assert not path.exists()  # 原件已改名保全，不再冒充正常文件
    assert pq.recovery_report()["backup_path"] == str(backup)


def test_bad_shape_pending_file_preserved(tmp_path):
    path = tmp_path / "pending_questions.json"
    path.write_text('{"questions": {"a": 1}}', encoding="utf-8")
    pq = PendingQuestions(path=path)
    assert pq.load_error is not None
    assert pq.load_error["reason"].startswith("bad_shape")
    assert Path(pq.load_error["backup_path"]).exists()


def test_missing_pending_file_is_not_corruption(tmp_path):
    pq = PendingQuestions(path=tmp_path / "pending_questions.json")
    assert pq.load_error is None
    assert pq.recovery_report() is None


def test_pending_rebuild_from_surviving_records(store):
    mid = _plant(store, "我把备用钥匙放在门口花盆下面")
    # 给行打上转述未核实标记（auth_claim 问题的存活痕迹在行 meta 里）。
    meta = dict(_flags(store, mid)["meta"])
    meta["auth_claim"] = _auth_guard.AUTH_CLAIM_RELAYED
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (json.dumps(meta, ensure_ascii=False), mid),
    )
    store._conn.commit()
    # 入口挂起的存活痕迹在 intake_log 里。
    held_text = "也许大概可能要不要记这句呢"
    store._log_intake(
        held_text, _intake.VERDICT_HOLD, "unsure", "拿不准，先挂起"
    )
    # 待问文件损坏（进程把文件写花了）。
    pq_path = store.memory_dir / "pending_questions.json"
    pq_path.write_text("garbage!!!", encoding="utf-8")
    # 重启：新 store 的 PendingQuestions 先保全原件，启动恢复再重建。
    store2 = _restart(store)
    report = store2.startup_recovery()
    assert report["pending"] is not None
    assert Path(report["pending"]["backup_path"]).exists()
    assert report["pending_rebuilt"]["auth_claim"] == 1
    assert report["pending_rebuilt"]["intake_hold"] == 1
    assert "importance" in report["pending_rebuilt"]["unrecoverable_kinds"]
    unanswered = store2.pending_questions.list_unanswered()
    auth_qs = [q for q in unanswered if q["kind"] == "auth_claim"]
    assert len(auth_qs) == 1 and auth_qs[0]["memory_id"] == mid
    hold_qs = [q for q in unanswered if q["kind"] == "intake_hold"]
    assert len(hold_qs) == 1 and hold_qs[0]["held_text"] == held_text
    # 结论可见的痕迹：pending_notices 报一条。
    notices = store2.pending_notices()
    assert len(notices) == 1
    assert notices[0]["kind"] == "pending_questions_corrupt"
