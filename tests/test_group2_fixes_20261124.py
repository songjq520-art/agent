# tests/test_group2_fixes_20261124.py（第二组修复，2026-11-24）
# 全部离线（向量打桩，不加载 bge 模型、不调真 API、不花钱）。
#
# 2a _dedupe_key 砍逗号（Gemini Bug 3，最危险）：无条件截掉最后一个
#    逗号后的内容，“每天定投5块，买纳指”与“买标普”算出同一个 key，
#    第二条被当 UNCHANGED 静默吞掉。修法：逗号只截“改写补充”尾巴
#    （“……是X”且 X 已在前半句出现，如“，主食偏好是面条”），普通
#    分句不截——该合并的改写版照旧合并，不同的事实一条都不丢。
# 2b “我现在”多子句误拒（Gemini 隐患 1）： “我现在住深圳了，喜欢吃
#    海鲜”整句结尾不是“了”，_correction_fact_after 返回 None，住址
#    变更漏掉。修法：整句不合格时按逗号取第一小句再校验，认第一
#    小句为新说法。

import pytest

from memory.correction_text import _detect_correction
from memory.sqlite_store import SQLiteStore, _dedupe_key


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


def _turn(store, text, turn_id):
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的"}],
        turn_id=turn_id,
    )


# ---------- 2a：逗号不截 ----------


def test_dedupe_key_comma_tail_kept():
    assert _dedupe_key("每天定投5块，买纳指") != _dedupe_key("每天定投5块，买标普")
    assert _dedupe_key("我喜欢吃面条，不喜欢吃米饭") != _dedupe_key(
        "我喜欢吃面条，也喜欢吃饺子"
    )


def test_dedupe_key_comma_supplement_still_merged():
    """逗号版改写补充照旧合并：和括号版算同一条（R2 后续③的实弹）。"""
    assert _dedupe_key("BOOS 改吃面条了，主食偏好是面条") == _dedupe_key(
        "我改吃面条了（主食偏好是面条）"
    )


def test_dedupe_key_parenthetical_still_stripped():
    """括号补充照旧剥：改写版和原句仍算同一条。"""
    assert _dedupe_key("我改吃面条了（主食偏好是面条）") == _dedupe_key("我改吃面条了")


def test_both_investment_facts_stored(store):
    r1 = store.add_memory_text("每天定投5块，买纳指", source="user")
    r2 = store.add_memory_text("每天定投5块，买标普", source="user")
    assert (r1 or {}).get("event") == "ADD"
    assert (r2 or {}).get("event") == "ADD"
    texts = _live_texts(store)
    assert "每天定投5块，买纳指" in texts
    assert "每天定投5块，买标普" in texts


def test_both_noodle_facts_stored(store):
    r1 = store.add_memory_text("我喜欢吃面条，不喜欢吃米饭", source="user")
    r2 = store.add_memory_text("我喜欢吃面条，也喜欢吃饺子", source="user")
    assert (r1 or {}).get("event") == "ADD"
    assert (r2 or {}).get("event") == "ADD"
    texts = _live_texts(store)
    assert "我喜欢吃面条，不喜欢吃米饭" in texts
    assert "我喜欢吃面条，也喜欢吃饺子" in texts


def test_same_text_still_deduped(store):
    """完全相同的句子照旧去重（逗号不截不等于不去重）。"""
    r1 = store.add_memory_text("每天定投5块，买纳指", source="user")
    r2 = store.add_memory_text("每天定投5块，买纳指", source="user")
    assert (r1 or {}).get("event") == "ADD"
    assert (r2 or {}).get("event") != "ADD"
    assert _live_texts(store).count("每天定投5块，买纳指") == 1


# ---------- 2b：“我现在”多子句 ----------


def test_now_multi_clause_detected():
    out = _detect_correction("我现在住深圳了，喜欢吃海鲜")
    assert out is not None
    assert out["fact"] == "我住深圳了"


def test_now_single_clause_still_detected():
    out = _detect_correction("我现在住深圳了")
    assert out is not None
    assert out["fact"] == "我住深圳了"


def test_now_plain_statement_not_detected():
    """只是现状陈述（没“了”、没槽位变更）不触发纠正。"""
    assert _detect_correction("我现在喜欢吃海鲜") is None


def test_now_multi_clause_save_turn_end_to_end(store):
    store.add_memory_text("我住在广州", source="user")
    receipt = _turn(store, "我现在住深圳了，喜欢吃海鲜", "t1")
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated"
    assert _is_superseded(store, "我住在广州")
    assert "我住深圳了" in _live_texts(store)
