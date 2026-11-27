# tests/test_correction_sameturn_20261124.py（四件套·实测问题二：纠正当轮落账）
#
# 覆盖契约「三、状态变更入口」纠正当轮落账条款与「六、验收·实测
# 坏用例」第 2 例（成都改重庆）的程序版断言：
#   - 斜改识别：「成都不去了，改到重庆」没有经典纠正词，以前识别
#     不出来、整句当新事实存、旧行留活，两行并存——现在按旧值
#     定位旧行、当轮完成顶替（旧行 superseded、新行唯一活行）；
#   - 收尾复核：回执 correction 带 verified/landed，landed 以回查
#     新旧两行真实状态为准，不听各路自报；
#   - 护栏拦下（核心行）时：status=pending_question、landed=False，
#     旧行不动、待问留痕，不假报成功；
#   - 斜改多候选分不清时：ambiguous + landed=False，不猜不顶，
#     新值碎片不单独落账；
#   - correct_memory 假报 updated（行状态没变）时，复核照样判
#     未落账。
#
# 全部离线：向量/判官打桩成 None（同 g7/fx8prot 口径）。

import json

import pytest

from memory import ledger as _ledger
from memory.correction_text import _detect_correction
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso

TRIP_OLD = "我下个月要去一趟成都，住春熙路附近"
TRIP_NEW = "我下个月要去一趟重庆，住春熙路附近"
REAL_FIX = "刚定的事情，成都不去了，改到重庆，时间不变"
IDENTITY_TEXT = "Agent 的名字是十二，你叫十二"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """干净的离线 SQLiteStore：向量/判官打桩，整理路由关掉。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.organizing_router = lambda: None
    return s


def _turn(store, text, turn_id):
    receipt = store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的"}],
        turn_id=turn_id,
    )
    thread = getattr(store, "_turn_organize_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=30)
    return receipt


def _rows(store):
    out = []
    for r in store._conn.execute(
        "SELECT id, text, superseded, deleted, voided FROM memories"
        " ORDER BY rowid"
    ).fetchall():
        out.append({
            "id": r[0], "text": r[1],
            "live": not (r[2] or r[3] or r[4]),
            "superseded": bool(r[2]),
        })
    return out


def _live_texts(store):
    return [r["text"] for r in _rows(store) if r["live"]]


def _plant_row(store, text, **meta_over):
    """直接插一行（模拟已在库的旧行），meta 可覆盖（同 g7 口径）。"""
    created_at = _now_iso()
    key = _ledger.ledger_event_key(text, created_at + text)
    meta = {
        "source": "user", "confidence": 1.0, "user_confirmed": True,
        "created_at": created_at, "write_seq": 1, "turn_id": None,
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "episodic", "importance": "normal",
        "event_date": None,
    }
    meta.update(meta_over)
    store._insert_memory_row(key, text, meta, None)
    return key


# ---------- 斜改识别（纯程序信号） ----------


def test_detect_plan_change_recognized():
    d = _detect_correction(REAL_FIX)
    assert d is not None and d.get("plan_change") is True, d
    assert d["old_value"] == "成都"
    assert d["fact"] == "重庆"
    # 换成式斜改同样走计划路（不落回弱信号的碎片路）。
    d2 = _detect_correction("成都不去了，换成重庆，时间不变")
    assert d2 is not None and d2.get("plan_change") is True, d2
    assert d2["old_value"] == "成都" and d2["fact"] == "重庆"
    # 不去 X 了式。
    d3 = _detect_correction("不去成都了，改到重庆")
    assert d3 is not None and d3.get("plan_change") is True, d3
    assert d3["old_value"] == "成都" and d3["fact"] == "重庆"


def test_detect_plan_change_rejects_non_values():
    # 被否定的是说话人自己（“我不去了”），不是一个值：不认。
    assert _detect_correction("我不去了，改到明天") is None
    # 只有否定、没有换值动词：不认。
    assert _detect_correction("成都不去了") is None
    # 经典纠正不受影响。
    d = _detect_correction("不对，我住深圳")
    assert d is not None and d.get("phrase") == "不对", d
    assert not d.get("plan_change")


# ---------- 实测原例：成都改重庆，当轮落账 ----------


def test_sameturn_trip_correction_lands(store, monkeypatch):
    # 无判官钥匙时 _judge_degraded 为 True，G7 锁 #2 会把程序层
    # 纠正挂成 pending_question、本条断言的「当轮落账」无从谈起。
    # 本测试作用域内打桩成不降级（同 G1 口径，不污染全套、
    # 不在全套层面塞假钥匙）；降级期挂起行为由专门用例覆盖。
    monkeypatch.setattr(SQLiteStore, "_judge_degraded", lambda self: False)
    store.add_memory_text(TRIP_OLD, source="user")
    receipt = _turn(store, REAL_FIX, "sameturn-trip-1")
    corr = receipt.get("correction") or {}
    assert corr.get("status") == "updated", corr
    assert corr.get("verified") is True, corr
    assert corr.get("landed") is True, corr
    rows = _rows(store)
    old = [r for r in rows if r["text"] == TRIP_OLD]
    assert len(old) == 1 and old[0]["superseded"], rows
    # 新行是唯一活行：旧行已顶、没有碎片/引子行混进来。
    assert _live_texts(store) == [TRIP_NEW], rows


def test_sameturn_classic_correction_verified_landed(store, monkeypatch):
    # 经典纠正（不对+整句新说法）照常落账，且带复核标记。
    # 同上：本测试作用域内打桩 _judge_degraded=False，
    # 免无钥匙环境把纠正挂成 pending_question。
    monkeypatch.setattr(SQLiteStore, "_judge_degraded", lambda self: False)
    store.add_memory_text(TRIP_OLD, source="user")
    receipt = _turn(store, "不对，我下个月去重庆出差，住春熙路附近",
                    "sameturn-classic-1")
    corr = receipt.get("correction") or {}
    assert corr.get("status") == "updated", corr
    assert corr.get("landed") is True, corr
    live = _live_texts(store)
    assert TRIP_OLD not in live, live
    assert any("重庆" in t for t in live), live


# ---------- 失败路：护栏拦下，不假报成功 ----------


def test_sameturn_core_protected_correction_not_landed(store):
    _plant_row(store, IDENTITY_TEXT, slot="agent_name",
               decay_kind="stable", importance="high")
    receipt = _turn(store, "不对，Agent 的名字是十三，你叫十三",
                    "sameturn-core-1")
    corr = receipt.get("correction") or {}
    # 护栏挂起：没执行、回执如实标未落账。
    assert corr.get("status") == "pending_question", corr
    assert corr.get("verified") is True, corr
    assert corr.get("landed") is False, corr
    assert corr.get("verify_note"), corr
    # 旧行（核心）不动；新说法留痕存下；待问已挂。
    live = _live_texts(store)
    assert IDENTITY_TEXT in live, live
    assert "Agent 的名字是十三，你叫十三" in live, live
    qs = [
        q for q in store.list_pending_questions()
        if q.get("kind") == "conflict"
        and q.get("status") != "answered"
    ]
    assert len(qs) >= 1, qs


def test_sameturn_fake_updated_report_caught_by_verify(store, monkeypatch):
    # correct_memory 假报 updated、库里纹丝不动：收尾复核必须
    # 判未落账，不许跟着假报成功。
    seeded = store.add_memory_text(TRIP_OLD, source="user")

    def _fake_correct(old_fact, new_fact, **kwargs):
        return {
            "status": "updated",
            "memory_id": "missing-id",
            "old_text": TRIP_OLD,
            "new_text": "我下个月去重庆出差，住春熙路附近",
            "candidates": [{"id": seeded["id"], "text": TRIP_OLD}],
            "reason": "已更正",
            "layer": "exact", "resolution": "exact",
            "audit": {"source": "user", "method": "fake"},
        }

    monkeypatch.setattr(store, "correct_memory", _fake_correct)
    receipt = _turn(store, "不对，我下个月去重庆出差，住春熙路附近",
                    "sameturn-fake-1")
    corr = receipt.get("correction") or {}
    assert corr.get("status") == "updated", corr
    assert corr.get("landed") is False, corr
    assert "旧行还活着" in (corr.get("verify_note") or ""), corr
    # 库里确实没动：旧行还活着。
    assert TRIP_OLD in _live_texts(store)


# ---------- 斜改多候选：不猜、不顶、留痕 ----------


def test_sameturn_ambiguous_plan_change_traced(store):
    store.add_memory_text("我下个月要去一趟成都出差", source="user")
    store.add_memory_text("我十一月还要去成都参加行业会议", source="user")
    receipt = _turn(store, REAL_FIX, "sameturn-amb-1")
    corr = receipt.get("correction") or {}
    assert corr.get("status") == "ambiguous", corr
    assert corr.get("landed") is False, corr
    assert corr.get("verified") is True, corr
    assert len(corr.get("candidates") or []) == 2, corr
    # 两条旧行都不动；新值碎片不单独落账；没有多余的行。
    live = _live_texts(store)
    assert "我下个月要去一趟成都出差" in live, live
    assert "我十一月还要去成都参加行业会议" in live, live
    assert "重庆" not in live, live
    assert len(live) == 2, live


def test_plan_change_without_target_stores_statement(store):
    # 旧值在库里没有对应行：没有可顶的，不算纠正失败——整句按
    # 普通陈述存下，新计划不丢。
    receipt = _turn(store, REAL_FIX, "sameturn-notarget-1")
    assert receipt.get("correction") is None, receipt.get("correction")
    live = _live_texts(store)
    assert any("重庆" in t for t in live), live
