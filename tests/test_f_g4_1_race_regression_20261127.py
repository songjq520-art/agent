# tests/test_f_g4_1_race_regression_20261127.py（整包验收 F-G4-1
# 竞态丢更新：回归测试，2026-11-27，BOOS 拍板修进本包）
#
# 缺陷（验收第一批定位、十一亲跑确定性复现）：save_turn 的后台
# 整理线程在 _organize_turn_memories 开头读行 meta 快照，中间隔着
# 模型调用，之后 _record_occurrence(mid, 陈旧meta, ...) 无条件整行
# 回写——主线程在此期间已记好的新出现、已完成的晋升会被整体
# 打回（账本 PROMOTE 事件还在、投影退回去，两边分叉）。连发多轮
# 消息即可踩到，不限测试环境。
#
# 修法（memory/sqlite_store.py _record_occurrence）：真正写回前在
# 锁内复读现行行，以现行 meta 为底、出现集合取并集、状态按合并
# 后的事实重算；现行已晋升的不许被打回。
#
# 本文件三条：
# 1. 复现原场景：晋升完成后迟到复核，断言 stable 与三天三会
#    出现记录不回滚；
# 2. 未晋升时迟到复核，断言主线程新记的出现不丢（并集合并）；
# 3. 合并后的证据凑满晋升线时，迟到复核应按合并事实完成晋升
#    （账本有 PROMOTE、投影 stable，两边一致）。
#
# 路径与生产一致：主 store 走 save_turn 同步路；「整理线程」用
# 同目录另开的 store 实例读快照、迟到调 _record_occurrence——
# 正是 _organize_turn_memories 晋升计数复核那一步的原样调用。

import json
from datetime import date

import memory.sqlite_store as store_mod
from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore
from test_phase2_importance_questions import FakeImportanceJudge

DAY1, DAY2, DAY3 = date(2026, 11, 20), date(2026, 11, 21), date(2026, 11, 22)
TEXT = "昨天去了一趟超市。"


def _turn(store, turn_id):
    return store.save_turn(
        [{"role": "user", "content": TEXT},
         {"role": "assistant", "content": "好的。"}],
        turn_id=turn_id,
    )


def _row(store, mid):
    kind, meta_json = store._conn.execute(
        "SELECT decay_kind, metadata_json FROM memories WHERE id=?",
        (mid,),
    ).fetchone()
    meta = json.loads(meta_json or "{}")
    return (
        kind,
        json.loads(meta.get("seen_dates") or "[]"),
        json.loads(meta.get("seen_sessions") or "[]"),
    )


def _make_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    main = SQLiteStore(
        memory_dir=str(tmp_path / "memory"),
        judge_provider=FakeImportanceJudge(),
    )
    return main


def _snapshot(worker, mid):
    text, meta_json = worker._conn.execute(
        "SELECT text, metadata_json FROM memories WHERE id=?", (mid,)
    ).fetchone()
    return text, json.loads(meta_json or "{}")


def _only_mid(store):
    rows = store._conn.execute("SELECT id FROM memories").fetchall()
    assert len(rows) == 1
    return rows[0][0]


def test_stale_recheck_does_not_roll_back_promotion(tmp_path, monkeypatch):
    main = _make_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: DAY1)
    _turn(main, "t1")
    mid = _only_mid(main)
    assert _row(main, mid) == ("episodic", ["2026-11-20"], ["t1"])

    # 整理线程开头读快照（此后隔着模型调用，迟到才复核）。
    worker = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    wtext, stale_meta = _snapshot(worker, mid)

    # 主线程接着连发两轮：记满 3 天 3 会并晋升 stable。
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: DAY2)
    _turn(main, "t2")
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: DAY3)
    _turn(main, "t3")
    assert _row(main, mid) == (
        "stable",
        ["2026-11-20", "2026-11-21", "2026-11-22"],
        ["t1", "t2", "t3"],
    )

    # 迟到复核拿陈旧快照回写：不许把晋升和出现记录打回。
    worker._record_occurrence(mid, stale_meta, DAY1.isoformat(), "t1", wtext)
    assert _row(main, mid) == (
        "stable",
        ["2026-11-20", "2026-11-21", "2026-11-22"],
        ["t1", "t2", "t3"],
    )


def test_stale_recheck_merges_newer_occurrences(tmp_path, monkeypatch):
    main = _make_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: DAY1)
    _turn(main, "t1")
    mid = _only_mid(main)

    worker = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    wtext, stale_meta = _snapshot(worker, mid)

    # 主线程又记了一天（还没到晋升线）。
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: DAY2)
    _turn(main, "t2")
    assert _row(main, mid) == (
        "episodic", ["2026-11-20", "2026-11-21"], ["t1", "t2"]
    )

    # 迟到复核：第二天的出现不许被快照盖掉。
    worker._record_occurrence(mid, stale_meta, DAY1.isoformat(), "t1", wtext)
    assert _row(main, mid) == (
        "episodic", ["2026-11-20", "2026-11-21"], ["t1", "t2"]
    )


def test_merged_evidence_completes_promotion(tmp_path, monkeypatch):
    main = _make_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: DAY1)
    _turn(main, "t1")
    mid = _only_mid(main)

    worker = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    wtext, stale_meta = _snapshot(worker, mid)

    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: DAY2)
    _turn(main, "t2")

    # 迟到复核带来第三天的一次真出现：单看快照只有 {d1,d3}、
    # 单看现行只有 {d1,d2}，合并后才凑满 3 天 3 会——应按合并
    # 事实晋升，且账本先行（PROMOTE 事件在、投影 stable）。
    promoted = worker._record_occurrence(
        mid, stale_meta, DAY3.isoformat(), "t3", wtext
    )
    assert promoted is True
    assert _row(main, mid) == (
        "stable",
        ["2026-11-20", "2026-11-21", "2026-11-22"],
        ["t1", "t2", "t3"],
    )
    events, _info = _ledger.read_events(main.ledger_path)
    assert any(
        e.get("type") == _ledger.LEDGER_EVENT_PROMOTE
        and e.get("memory_id") == mid
        for e in events
    )
