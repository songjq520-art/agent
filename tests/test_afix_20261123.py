# tests/test_afix_20261123.py（A 组备注行修复，2026-11-23）
# 全部离线（向量打桩，不加载 bge 模型、不调真 API、不花钱）。
# 背景：A 组 T2“记错了，我喜欢吃面条”时，模型在 save_turn 之前自己调
# add_memory 写了备注行“BOOS 喜欢吃面条（纠正：不是喜欢吃米饭）”——
# 这行已经顶掉了米饭，但 slot 是 null、相似度又被括号里的旧说法拉低
# （0.699x < 0.70），纠正分支认不出它报 not_found；新说法又被
# _dedupe_key 当成备注行的重复而不存成 exact 行。
# 修法：correct_memory 带明确纠正词时，先认备注行——文本含“纠正：”
# 且标记前半句与新说法归一化相同，就用 exact 新说法取代备注行，
# 回执 updated；备注行里被纠正掉的旧说法若还活着，一并取代。

import math

import pytest

from memory.sqlite_store import (
    SQLiteStore,
    _extract_note_old_fact,
)
from web.server import _correction_reply_text


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _sim68_embed(self, text):
    """米饭/面条的余弦相似度恰好 0.68（低于 0.70 阈值），复现 A 组卡点。"""
    if "米饭" in text:
        return [1.0, 0.0]
    if "面条" in text:
        return [0.68, math.sqrt(1 - 0.68 ** 2)]
    return [0.0, 1.0]


@pytest.fixture
def sim68_store(tmp_path, monkeypatch):
    """向量按 0.68 相似度打桩的 store：米饭行和面条说法的相似度 0.68。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", _sim68_embed)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _turn(store, text, turn_id):
    """一轮对话：用户说一句话，助手回一句“好的”。"""
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的。"}],
        turn_id=turn_id,
    )


NOTE = "BOOS 喜欢吃面条（纠正：不是喜欢吃米饭）"


def test_afix_extract_note_old_fact():
    """从备注行提旧说法：只认“纠正：不是X”这种干净格式。"""
    assert _extract_note_old_fact(NOTE) == "BOOS 喜欢吃米饭"
    # 真纠正便条没有“纠正：”冒号标记，不提。
    assert _extract_note_old_fact(
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。"
    ) is None
    # “之前说……是错的”这种元评论不是事实，不提。
    assert _extract_note_old_fact(
        "BOOS 喜欢吃面条（纠正：之前说喜欢吃米饭是错的）"
    ) is None
    # 普通句子没有标记，不提。
    assert _extract_note_old_fact("我喜欢吃面条") is None
    # 半角冒号也认。
    assert _extract_note_old_fact(
        NOTE.replace("：", ":")
    ) == "BOOS 喜欢吃米饭"


@pytest.mark.requires_judge
def test_afix_note_line_recognized_by_correct_memory(sim68_store):
    """备注行认得出：米饭 + 备注行都在，带明确纠正词纠正到面条时，
    correct_memory 必须 updated（而不是 not_found），exact 新行入库，
    备注行被取代，米饭也被一并取代（兜底），回执 old_text 是米饭。"""
    store = sim68_store
    store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text(NOTE, source="user")
    out = store.correct_memory(
        "我喜欢吃面条", "我喜欢吃面条", explicit_correction=True
    )
    assert out.get("status") == "updated"
    assert out.get("layer") == "note"
    assert out.get("old_text") == "BOOS 喜欢吃米饭"
    assert out.get("new_text") == "我喜欢吃面条"
    assert _is_superseded(store, NOTE) is True
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert "我喜欢吃面条" in _live_texts(store)


@pytest.mark.requires_judge
def test_afix_save_turn_end_to_end_with_note(sim68_store):
    """A 组端到端（含备注行）：T1 米饭 → 模型抢跑写备注行 →
    T2“记错了，我喜欢吃面条”→ 纠正回执 updated，米饭被取代，
    面条是 exact 新行、当前答案。"""
    store = sim68_store
    _turn(store, "我喜欢吃米饭", "t1")
    assert "我喜欢吃米饭" in _live_texts(store)
    # 模型抢跑：add_memory 工具直存备注行（source=user，和线上一致）。
    store.add_memory_text(NOTE, source="user")
    assert NOTE in _live_texts(store)
    receipt = _turn(store, "记错了，我喜欢吃面条", "t2")
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated"
    assert correction.get("old_text") == "BOOS 喜欢吃米饭"
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert _is_superseded(store, NOTE) is True
    assert "我喜欢吃面条" in _live_texts(store)
    # 回复措辞：含“改好了”。
    reply = _correction_reply_text(correction)
    assert reply is not None and "改好了" in reply
    assert reply == "已经改好了：把“BOOS 喜欢吃米饭”改成了“我喜欢吃面条”。"


def test_afix_note_head_mismatch_not_matched(sim68_store):
    """备注行主题不对时不认：备注行说的是米线，纠正到面条时不能
    把米线备注行当成目标，必须走老路（not_found），不硬猜。"""
    store = sim68_store
    store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text("BOOS 喜欢吃米线（纠正：不是喜欢吃米饭）", source="user")
    out = store.correct_memory(
        "我喜欢吃面条", "我喜欢吃面条", explicit_correction=True
    )
    assert out.get("status") == "not_found"
    assert _is_superseded(store, "BOOS 喜欢吃米线（纠正：不是喜欢吃米饭）") is False


def test_afix_real_correction_note_not_mistaken(sim68_store):
    """真纠正便条（72cc6132 同款，没有“纠正：”冒号）不被当成备注行：
    纠正到鸡杂面时，鸡杂面便条不能被 note 分支顶掉。"""
    store = sim68_store
    store.add_memory_text(
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。",
        source="user",
    )
    store.add_memory_text("BOOS 喜欢吃鸡杂面。", source="user")
    out = store.correct_memory(
        "我不喜欢吃鸡杂面", "我不喜欢吃鸡杂面", explicit_correction=True
    )
    # 不管走哪条路，都不能把真便条当成模型备注行顶掉。
    assert _is_superseded(
        store,
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。",
    ) is False


def test_afix_note_branch_needs_explicit_correction(sim68_store):
    """备注行只在带明确纠正词时认：不带纠正词的 correct_memory
    不走 note 分支（走老路），不扩大纠正口子。"""
    store = sim68_store
    store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text(NOTE, source="user")
    out = store.correct_memory("我喜欢吃面条", "我喜欢吃面条")
    assert out.get("layer") != "note"
