# tests/test_group1_fixes_20261124.py（第一组修复，2026-11-24）
# 全部离线（向量打桩，不加载 bge 模型、不调真 API、不花钱）。
#
# 1a 纠错新记忆 turn_id 丢了（Gemini Bug 1）：_apply_correction_result 建
#    新记忆时 metadata 的 turn_id 写死 None，save_turn 记的
#    _last_stored_turn_id 对不上，下一句“算了”撤不掉纠正出的新记忆。
#    修法：correct_memory / _apply_correction_result / 判官两路都透传
#    turn_id，新记忆带本轮号，supersede 事件也带。
# 1b “已纠正过”护栏靠 5 分钟墙钟（Gemini Bug 2 / 自审 P1-2）：上一轮
#    刚改完、这一轮改主意时，最近一条 supersede 还在 5 分钟内，护栏
#    误拦，真该顶的那条顶不掉。修法：护栏只认 turn_id 相同的
#    supersede；工具路没 turn_id 时用当前轮登记（_active_turn_id）兜底。
# 1c 槽位直选要求恰好 1 条候选（自审 P1-1）：真记忆同槽永远多条，
#    live 里从未触发。修法：同槽多条时按相似度差距直选（第一名
#    ≥0.70 且第二名 <0.50），差距不够走老路，不猜。

import json

import pytest

from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _meta_of(store, text):
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE text=?", (text,)
    ).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def _turn(store, text, turn_id):
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的"}],
        turn_id=turn_id,
    )


# ---------- 1a：纠正出的新记忆带 turn_id，撤回能作废 ----------


def test_correction_new_memory_carries_turn_id(store):
    store.add_memory_text("我喜欢吃米饭", source="user")
    out = store.correct_memory(
        "我喜欢吃米饭", "我喜欢吃面条",
        explicit_correction=True, turn_id="turn-42",
    )
    assert out["status"] == "updated"
    meta = _meta_of(store, "我喜欢吃面条")
    assert meta.get("turn_id") == "turn-42"
    # supersede 事件也带本轮号（护栏 1b 靠它认本轮）。
    events, _info = _ledger.read_events(store.ledger_path)
    sup = [e for e in events if e.get("type") == "supersede"]
    assert sup and sup[-1].get("turn_id") == "turn-42"


def test_correction_then_retract_voids_new_memory(store):
    _turn(store, "我喜欢吃米饭", "t1")
    assert "我喜欢吃米饭" in _live_texts(store)
    receipt = _turn(store, "不对，我喜欢吃面条", "t2")
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated"
    assert "我喜欢吃面条" in _live_texts(store)
    assert _meta_of(store, "我喜欢吃面条").get("turn_id") == "t2"
    # 下一句“算了”：上一轮（t2）存下的整轮作废，面条要被作废掉。
    receipt = _turn(store, "算了", "t3")
    assert receipt.get("retracted") is True
    assert receipt.get("dropped_count", 0) >= 1
    assert "我喜欢吃面条" not in _live_texts(store)
    row = store._conn.execute(
        "SELECT voided FROM memories WHERE text=?", ("我喜欢吃面条",)
    ).fetchone()
    assert row and row[0] == 1


# ---------- 1b：护栏绑 turn_id ----------


def test_guard_blocks_same_turn_repeat(store):
    """同轮重复纠正（工具先改、save_turn 再调）：护栏拦住。"""
    store.add_memory_text("我住在武汉", source="user")
    first = store.correct_memory(
        "我住在武汉", "我住在长沙", explicit_correction=True, turn_id="t1"
    )
    assert first["status"] == "updated"
    second = store.correct_memory(
        "我住在长沙", "我住在长沙", explicit_correction=True, turn_id="t1"
    )
    assert second["status"] == "updated"
    assert second["resolution"] == "already_corrected"
    assert second["old_text"] == "我住在武汉"


def test_guard_does_not_block_other_turn(store):
    """上一轮的 supersede 不是本轮的：护栏不许拦（旧 5 分钟窗口会拦）。"""
    store.add_memory_text("我住在武汉", source="user")
    first = store.correct_memory(
        "我住在武汉", "我住在长沙", explicit_correction=True, turn_id="t1"
    )
    assert first["status"] == "updated"
    second = store.correct_memory(
        "我住在长沙", "我住在长沙", explicit_correction=True, turn_id="t9"
    )
    assert second["resolution"] != "already_corrected"
    assert second["status"] == "unchanged"


def test_guard_no_false_block_change_back(store):
    """改回旧值的场景：长沙是 t1 顶掉武汉存下的，郑州是后来新存的；
    t3 用户说“不对，我住在长沙”——该顶的是郑州，不是回一句
    “已纠正过（武汉）”就完事（旧 5 分钟窗口会误拦，郑州顶不掉）。"""
    store.add_memory_text("我住在武汉", source="user")
    r1 = store.correct_memory(
        "我住在武汉", "我住在长沙", explicit_correction=True, turn_id="t1"
    )
    assert r1["status"] == "updated"
    store.add_memory_text("我住在郑州", source="user")
    out = store.correct_memory(
        "我住在长沙", "我住在长沙", explicit_correction=True, turn_id="t3"
    )
    assert out["status"] == "updated"
    assert out["layer"] != "already"
    assert out["old_text"] == "我住在郑州"
    assert _is_superseded(store, "我住在郑州")
    assert "我住在长沙" in _live_texts(store)


def test_change_back_supersedes_current_value(store):
    """武汉→长沙→郑州→改回长沙：最后一次纠正要顶掉郑州、长沙活着。"""
    store.add_memory_text("我住在武汉", source="user")
    r1 = store.correct_memory(
        "我住在武汉", "我住在长沙", explicit_correction=True, turn_id="t1"
    )
    assert r1["status"] == "updated"
    r2 = store.correct_memory(
        "我住在长沙", "我住在郑州", explicit_correction=True, turn_id="t2"
    )
    assert r2["status"] == "updated"
    r3 = store.correct_memory(
        "我住在郑州", "我住在长沙", explicit_correction=True, turn_id="t3"
    )
    assert r3["status"] == "updated"
    assert r3["old_text"] == "我住在郑州"
    assert _is_superseded(store, "我住在郑州")
    assert "我住在长沙" in _live_texts(store)


# ---------- 1c：槽位直选按相似度差距 ----------


def _ids_by_text(store):
    return {item["text"]: item["id"] for item in store.get_all()}


def _stub_similarity(monkeypatch, sims):
    monkeypatch.setattr(
        SQLiteStore, "_candidate_similarity",
        lambda self, mid, fact: sims.get(mid),
    )


def test_slot_target_picks_clear_winner(store, monkeypatch):
    """3 条同槽候选，第一名 0.75、第二名 0.40：直选第一名。"""
    for text in ("我住在广州", "我住在北京", "我住在上海"):
        store.add_memory_text(text, source="user")
    ids = _ids_by_text(store)
    _stub_similarity(monkeypatch, {
        ids["我住在广州"]: 0.75,
        ids["我住在北京"]: 0.40,
        ids["我住在上海"]: 0.30,
    })
    target = store._explicit_correction_slot_target(
        list(ids.values()), "我住在深圳"
    )
    assert target == ids["我住在广州"]


def test_slot_target_close_race_returns_none(store, monkeypatch):
    """第一名 0.75、第二名 0.72：差距不够，不直选（走老路问判官）。"""
    for text in ("我住在广州", "我住在北京", "我住在上海"):
        store.add_memory_text(text, source="user")
    ids = _ids_by_text(store)
    _stub_similarity(monkeypatch, {
        ids["我住在广州"]: 0.75,
        ids["我住在北京"]: 0.72,
        ids["我住在上海"]: 0.30,
    })
    target = store._explicit_correction_slot_target(
        list(ids.values()), "我住在深圳"
    )
    assert target is None


def test_slot_target_end_to_end(store, monkeypatch):
    """端到端：明确纠正走槽位直选，layer=slot，顶掉的是相似度最高那条。"""
    for text in ("我住在广州", "我住在北京", "我住在上海"):
        store.add_memory_text(text, source="user")
    ids = _ids_by_text(store)
    _stub_similarity(monkeypatch, {
        ids["我住在广州"]: 0.75,
        ids["我住在北京"]: 0.40,
        ids["我住在上海"]: 0.30,
    })
    out = store.correct_memory(
        "住在广州", "我住在深圳", explicit_correction=True, turn_id="t1"
    )
    assert out["status"] == "updated"
    assert out["layer"] == "slot"
    assert out["old_text"] == "我住在广州"
    assert _is_superseded(store, "我住在广州")
    assert not _is_superseded(store, "我住在北京")
    assert "我住在深圳" in _live_texts(store)
