# tests/test_locked_snapshot_20261125.py（立项 H4：持锁稳定版本快照，2026-11-25）
#
# 覆盖口径（施工清单 H4＋契约三）：
#   - 快照在 store 锁内一次冻结：行版本戳（与受控入口
#     controlled_entry_version 同一戳法）＋保护/抽屉＋账本序号
#     （ledger_state 的 seq_max）同一临界区取齐；生成不写账本
#     不写库；
#   - 座位账与快照同源同读（count_seats 口径），且与同时刻
#     recount_core_seats_from_ledger() 对账相等；
#   - 冻结性：快照后库内发生变更，快照视图纹丝不动、当前视图
#     已变；并发写入下每份快照内部自洽（逐行戳可复算、座位账
#     可由快照行复算）；
#   - 差异：diff 按 memory_id 列 added / removed / changed，
#     changed 带状态迁移——改抽屉（current→current 戳变）、
#     顶替（→superseded）、进档案（→dormant）、删除（→deleted）
#     都对得出；判定只认版本戳与状态，不靠时间；
#   - 只读隔离：返回的是深拷贝，调用方改快照对象污染不了
#     登记原件，也污染不了店内状态；
#   - 生命周期：release 后 get/diff 如实 KeyError，不给假
#     数据；重复 release 返回 False；快照不妨碍正常读写。
#
# 全程离线：嵌入/判官打桩为 None，同 test_controlled_entry 口径。

import json
import threading
import uuid

import pytest

from memory import core_cap as _core_cap
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


def _session(store, direct=True):
    sess = store.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=direct
    )
    assert sess is not None
    return sess


def _execute(store, action, mid, **params):
    """开开关、带主人直达会话与版本戳执行一笔受控提案。"""
    store.controlled_entry_execution_enabled = True
    prop = {
        "action": action,
        "memory_id": mid,
        "reason": "测试执行",
        "instruction_source": "owner",
        "idempotency_key": uuid.uuid4().hex,
        "expected_version": store.controlled_entry_version(mid),
        "session": _session(store),
    }
    if params:
        prop["params"] = params
    receipt = store.submit_controlled_proposal(prop)
    assert receipt["status"] == "executed", receipt
    return receipt


def test_snapshot_shape_anchor_and_seat_reconcile(store):
    a = _add_user(store, "BOOS 住在盐湖城朋友家")
    b = _add_user(store, "BOOS 每天定投五块钱买纳指")
    snap = store.take_stable_snapshot()
    assert snap["snapshot_id"]
    assert snap["generated_at"]
    assert snap["scoped"] is False
    assert snap["row_count"] == 2
    # 账本锚：与同时刻 ledger_state 的 seq_max 相等。
    assert snap["ledger_seq"] == store.ledger_state()["seq_max"]
    assert snap["ledger_seq"] > 0
    # 行版本戳与受控入口绑版本用的戳逐字相等（H1 漂移检出共用）。
    for mid in (a, b):
        assert snap["rows"][mid]["version"] == store.controlled_entry_version(mid)
        assert snap["rows"][mid]["state"] == "current"
        assert snap["rows"][mid]["live"] is True
    # 座位账内部自洽 + 与账本重算对账相等。
    seats = snap["seats"]
    assert seats["reconciled"] is True
    recount = store.recount_core_seats_from_ledger()
    for key in ("seats_used", "members", "pinned", "temp_pinned",
                "unclassified_seats", "live_rows"):
        assert seats[key] == recount[key], key
    assert seats["drawers"] == recount["drawers"]


def test_snapshot_frozen_after_writes(store):
    a = _add_user(store, "BOOS 的电脑是华硕灵耀")
    snap = store.take_stable_snapshot()
    old_version = snap["rows"][a]["version"]
    old_seq = snap["ledger_seq"]
    # 快照之后：改一条 + 加一条。
    store.set_core_drawer(a, "environment")
    _add_user(store, "BOOS 有一台安卓手机")
    again = store.get_stable_snapshot(snap["snapshot_id"])
    assert again["rows"][a]["version"] == old_version
    assert again["rows"][a]["core_drawer"] is None
    assert again["ledger_seq"] == old_seq
    assert again["row_count"] == 1
    # 当前视图已变（对照组）。
    assert store.controlled_entry_version(a) != old_version


def test_diff_added_and_changed(store):
    a = _add_user(store, "BOOS 喜欢喝茶")
    b = _add_user(store, "BOOS 的姐姐在西安")
    snap = store.take_stable_snapshot()
    store.set_core_drawer(a, "person")
    c = _add_user(store, "BOOS 每天睡白天")
    diff = store.diff_stable_snapshot(snap["snapshot_id"])
    assert diff["added"] == [c]
    assert diff["removed"] == []
    changed = {entry["memory_id"]: entry for entry in diff["changed"]}
    assert set(changed) == {a}
    assert changed[a]["state_before"] == "current"
    assert changed[a]["state_after"] == "current"
    assert changed[a]["version_before"] != changed[a]["version_after"]
    assert diff["unchanged_count"] == 1  # b 没动
    assert diff["ledger_seq_now"] > diff["ledger_seq_then"]


def test_diff_state_transitions(store, monkeypatch):
    # 本条要靠 correct_memory 真落账产生顶替迁移；无判官钥匙时
    # _judge_degraded 为 True、纠正会被挂成 pending_question。
    # 本测试作用域内打桩成不降级（同 G1 口径，不污染全套）。
    monkeypatch.setattr(SQLiteStore, "_judge_degraded", lambda self: False)
    a = _add_user(store, "BOOS 打算去成都玩")
    b = _add_user(store, "BOOS 的书房里有一台打印机")
    c = _add_user(store, "BOOS 养了一只猫")
    snap = store.take_stable_snapshot()
    # 顶替：a 被新说法顶掉。
    result = store.correct_memory("BOOS 打算去成都玩", "BOOS 打算去重庆玩")
    assert result.get("ok") is not False, result
    # 进档案：b 经受控入口标沉底。
    _execute(store, "mark_dormant", b)
    # 删除：c 走批准路删掉。
    gone = store.delete_memory(c, approve=True)
    assert gone.get("deleted") is True, gone
    diff = store.diff_stable_snapshot(snap["snapshot_id"])
    changed = {entry["memory_id"]: entry for entry in diff["changed"]}
    assert changed[a]["state_before"] == "current"
    assert changed[a]["state_after"] == "superseded", changed
    assert changed[b]["state_after"] == "dormant", changed
    assert changed[c]["state_after"] == "deleted", changed
    # 纠正落账会新建一行装新说法：added 里正是它，不是漏报。
    assert len(diff["added"]) == 1
    new_text = store._conn.execute(
        "SELECT text FROM memories WHERE id=?", (diff["added"][0],)
    ).fetchone()[0]
    assert new_text == "BOOS 打算去重庆玩"
    assert diff["removed"] == []


def test_snapshot_is_deep_copy(store):
    a = _add_user(store, "BOOS 的笔记本是 Vivobook")
    snap = store.take_stable_snapshot()
    # 调用方把返回的快照改烂：登记原件不受污染。
    snap["rows"][a]["meta"]["core_drawer"] = "钱与决定"
    snap["rows"][a]["text"] = "被改烂的原文"
    snap["seats"]["seats_used"] = 999
    again = store.get_stable_snapshot(snap["snapshot_id"])
    assert again["rows"][a]["text"] == "BOOS 的笔记本是 Vivobook"
    assert again["rows"][a]["meta"].get("core_drawer") is None
    assert again["seats"]["seats_used"] != 999
    # get 返回的也是拷贝：再改再取仍干净。
    again["rows"][a]["meta"]["protection_tier"] = "pinned"
    third = store.get_stable_snapshot(snap["snapshot_id"])
    assert third["rows"][a]["meta"].get("protection_tier") is None
    # 店内状态更没被碰。
    assert store.controlled_entry_version(a) == third["rows"][a]["version"]


def test_release_lifecycle(store):
    _add_user(store, "BOOS 在美国出差")
    snap = store.take_stable_snapshot()
    sid = snap["snapshot_id"]
    assert store.release_stable_snapshot(sid) is True
    with pytest.raises(KeyError):
        store.get_stable_snapshot(sid)
    with pytest.raises(KeyError):
        store.diff_stable_snapshot(sid)
    assert store.release_stable_snapshot(sid) is False
    with pytest.raises(KeyError):
        store.get_stable_snapshot("never-existed")


def test_snapshot_generation_writes_nothing(store):
    _add_user(store, "BOOS 用 iPhone")
    import os
    ledger_size = os.path.getsize(store.ledger_path)
    seq_before = store.ledger_state()["seq_max"]
    rows_before = store._conn.execute(
        "SELECT COUNT(*) FROM memories"
    ).fetchone()[0]
    snap = store.take_stable_snapshot()
    store.get_stable_snapshot(snap["snapshot_id"])
    store.diff_stable_snapshot(snap["snapshot_id"])
    store.release_stable_snapshot(snap["snapshot_id"])
    assert os.path.getsize(store.ledger_path) == ledger_size
    assert store.ledger_state()["seq_max"] == seq_before
    assert store._conn.execute(
        "SELECT COUNT(*) FROM memories"
    ).fetchone()[0] == rows_before


def test_scoped_snapshot(store):
    a = _add_user(store, "BOOS 每天定投恒生五块")
    b = _add_user(store, "BOOS 的奶奶喜欢听戏")
    snap = store.take_stable_snapshot(memory_ids=[a])
    assert snap["scoped"] is True
    assert set(snap["rows"]) == {a}
    # 座位账仍是全库口径（两条活行都在账上）。
    assert snap["seats"]["live_rows"] == 2
    store.set_core_drawer(a, "money_decisions")
    store.set_core_drawer(b, "people")
    diff = store.diff_stable_snapshot(snap["snapshot_id"])
    assert [entry["memory_id"] for entry in diff["changed"]] == [a]
    assert diff["added"] == []


def test_concurrent_writes_snapshots_self_consistent(store):
    ids = [_add_user(store, "BOOS 的事实第 %d 条" % i) for i in range(4)]
    stop = threading.Event()

    def writer():
        n = 0
        drawers = ["person", "people", "long_term", "environment", "money_decisions"]
        while not stop.is_set():
            mid = ids[n % len(ids)]
            store.set_core_drawer(mid, drawers[n % len(drawers)])
            n += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(30):
            snap = store.take_stable_snapshot()
            # 逐行戳可复算：行视图内部同源，没有半新半旧。
            for row in snap["rows"].values():
                assert row["version"] == SQLiteStore._controlled_version_stamp(
                    row["text"], row["meta"]
                )
            # 座位账可由快照自己的活行复算：与冻结读数同源。
            live_metas = [
                row["meta"] for row in snap["rows"].values() if row["live"]
            ]
            recount = _core_cap.count_seats(
                live_metas, limit=store.core_seat_limit
            )
            assert snap["seats"] == recount
            assert snap["seats"]["reconciled"] is True
    finally:
        stop.set()
        t.join(timeout=30)
    assert not t.is_alive()


def test_snapshot_does_not_block_normal_io(store):
    a = _add_user(store, "BOOS 喜欢下雨天")
    snap = store.take_stable_snapshot()
    b = _add_user(store, "BOOS 喜欢晴天散步")
    assert store.controlled_entry_version(b)
    snap2 = store.take_stable_snapshot()
    assert snap2["row_count"] == snap["row_count"] + 1
    diff = store.diff_stable_snapshot(snap["snapshot_id"])
    assert diff["added"] == [b]
    assert store.release_stable_snapshot(snap2["snapshot_id"]) is True
    assert store.release_stable_snapshot(snap["snapshot_id"]) is True
