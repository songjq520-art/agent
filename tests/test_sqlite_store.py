# tests/test_sqlite_store.py（SQLite 第二步第一部分的回归测试，2026-11-21）
# 只测 memory/sqlite_store.py；全部离线（向量打桩，不加载 bge 模型），
# 用 tmp_path，不碰真实账本、真实 Chroma、不联网。
# 依据：~/workspace/十二网页测-2026-11-21.md（病根 + 验收）。
#
# 覆盖：
#   1. save_turn 原样存事实、留回执（饺子/豆浆案的根治）
#   2. 问句/闲聊不存；同一 turn_id 只存一次
#   3. 日期换算：昨天/今天/前天/明天按 Asia/Shanghai 的今天算
#   4. 来源标签：原样存的事实是 inferred（提取所得），不是 unconfirmed——
#      标错了问答提示词会把它当“没来源”处理、问起来只说不知道
#      （第一轮线上验收就是这么卡住的）；confirmed 键对齐 mem0 _provenance
#   5. 对账只重建当前态（被取代/删了的不进索引）
#   6. Chroma 一次性迁移幂等、跳过已删
#   7. 检索原样匹配在前；correct_memory 明确报未实现、不悄悄猜
#   8. 账本是真相源：删库重建后能从账本对账恢复

from datetime import date, timedelta

import numpy as np
import pytest

from memory.sqlite_store import SQLiteStore, _shanghai_today, resolve_event_date


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _ledger_lines(store):
    try:
        return open(store.ledger_path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return []  # 什么都没存过时账本文件不存在，算 0 行


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


# ---------- 1. 原样存事实 ----------


def test_save_turn_stores_verbatim_fact_with_receipt(store):
    """饺子案：用户亲口说的事实要同步进账本 + SQLite，留回执。"""
    receipt = store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线"}],
        turn_id="t-mixian",
    )
    assert receipt["stored"] == ["我今天晚上吃了米线"]
    assert receipt["dropped"] == []
    assert any("米线" in line for line in _ledger_lines(store))
    assert "我今天晚上吃了米线" in _live_texts(store)


def test_save_turn_skips_questions_and_greetings(store):
    """问句、闲聊不存进长期记忆。"""
    receipt = store.save_turn(
        [{"role": "user", "content": "你好，今天天气怎么样？"}],
        turn_id="t-hello",
    )
    assert receipt["stored"] == []
    assert _live_texts(store) == []
    assert _ledger_lines(store) == []


def test_save_turn_skips_questions_without_question_mark(store):
    """没打问号的问句也不存：2026-11-21 回归抽查抓到的，
    “定投计划怎么样了”没打问号，被当事实存进去、检索还排第一。"""
    for text in ["定投计划怎么样了", "我昨天中午吃了什么", "我爱吃什么"]:
        receipt = store.save_turn(
            [{"role": "user", "content": text}],
            turn_id="t-q-" + text[:4],
        )
        assert receipt["stored"] == [], text
    assert _live_texts(store) == []
    assert _ledger_lines(store) == []


def test_save_turn_same_turn_id_deduped(store):
    """web 的 ask() 和 run_question() 会各调一次 save_turn：
    同一 turn_id 第二次直接返回、不重复存。"""
    messages = [{"role": "user", "content": "我今天晚上吃了米线"}]
    first = store.save_turn(messages, turn_id="t-dup")
    second = store.save_turn(messages, turn_id="t-dup")
    assert first["stored"] == ["我今天晚上吃了米线"]
    assert second.get("duplicate_turn") is True
    assert _live_texts(store).count("我今天晚上吃了米线") == 1


# ---------- 2. 日期换算 ----------


def test_resolve_event_date_relative_days():
    """昨天/今天/前天/明天按今天算（Asia/Shanghai 的今天由调用方传入）。
    返回 ISO 日期字符串（metadata 里 event_date 的格式）。"""
    today = date(2026, 11, 21)
    assert resolve_event_date("我昨天中午吃了饺子", today) == "2026-11-20"
    assert resolve_event_date("我今天早上喝了一杯豆浆", today) == "2026-11-21"
    assert resolve_event_date("我前天去游泳了", today) == "2026-11-19"
    assert resolve_event_date("我明天要去游泳", today) == "2026-11-22"
    # 没提时间的，不硬猜日期。
    assert resolve_event_date("我喜欢吃火锅", today) is None


def test_save_turn_writes_event_date(store):
    """饺子日期错案：今天说“昨天”，event_date 必须是今天减一天。
    （按今天现算，不写死日期，不然明天跑就失效。）"""
    today = _shanghai_today()
    store.save_turn(
        [{"role": "user", "content": "我昨天中午吃了饺子"}],
        turn_id="t-jiaozi",
    )
    items = store.search("饺子", top_k=3)
    assert items and items[0]["text"] == "我昨天中午吃了饺子"
    assert items[0]["metadata"]["event_date"] == (today - timedelta(days=1)).isoformat()


# ---------- 3. 来源标签（第一轮验收卡住的根因） ----------


def test_save_turn_source_is_inferred_not_unconfirmed(store):
    """原样存的事实来源是 inferred（提取所得），出处完整（turn_id 在）；
    unconfirmed 是留给“完全没来源信息”的旧记忆的桶，标错了问起来
    模型只能说不知道。"""
    store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线"}],
        turn_id="t-src",
    )
    meta = store.search("米线", top_k=1)[0]["metadata"]
    assert meta["source"] == "inferred"
    assert meta["confirmed"] is False
    assert meta["turn_id"] == "t-src"


def test_add_memory_text_user_source_is_confirmed(store):
    """source=user（BOOS 亲口让记住）时 metadata 里 confirmed 必须为 True，
    和 memory/provenance.py _provenance 的算法一致；问答提示词认这个键。"""
    result = store.add_memory_text("每天定投 5 块买万家纳指", source="user")
    assert result.get("event") == "ADD"
    meta = store.search("万家纳指", top_k=1)[0]["metadata"]
    assert meta["confirmed"] is True
    assert meta["user_confirmed"] is True


# ---------- 4. 对账：只重建当前态 ----------


def test_reconcile_drops_deleted_and_keeps_current(store):
    """账本里删了的，索引里不能再有；没删的照常回来。"""
    a = store.add_memory_text("我喜欢喝可乐", source="user")
    store.add_memory_text("我喜欢吃火锅", source="user")
    assert store.delete_memory(a["id"], approve=True)["deleted"] is True
    fresh = SQLiteStore(memory_dir=str(store.memory_dir))
    fresh.reconcile_ledger_index()
    texts = _live_texts(fresh)
    assert "我喜欢吃火锅" in texts
    assert "我喜欢喝可乐" not in texts


def test_ledger_is_truth_source_rebuild_after_db_deleted(store, tmp_path):
    """账本是真相源：把 SQLite 库文件删了、重建后从账本对账能恢复记忆。"""
    store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线"}],
        turn_id="t-rebuild",
    )
    store._conn.close()
    for suffix in ("", "-wal", "-shm"):
        try:
            (store.db_path.parent / (store.db_path.name + suffix)).unlink()
        except FileNotFoundError:
            pass
    rebuilt = SQLiteStore(memory_dir=str(store.memory_dir))
    rebuilt.reconcile_ledger_index()
    assert "我今天晚上吃了米线" in _live_texts(rebuilt)


# ---------- 5. Chroma 一次性迁移 ----------


class _StubMem0:
    def __init__(self, items):
        self._items = items

    def get_all(self, include_sensitive=True):
        return self._items


def test_import_from_mem0_idempotent(store):
    """迁移幂等：跑两次，第二遍 imported 为 0、不重复。"""
    stub = _StubMem0(
        [
            {"text": "我爱的是冒菜，不是麻辣烫",
             "metadata": {"source": "user", "created_at": "2026-11-17T00:38:33Z"}},
            {"text": "BOOS 喜欢吃牛肉面",
             "metadata": {"source": "inferred", "created_at": "2026-11-19T00:00:00Z"}},
        ]
    )
    first = store.import_from_mem0(stub)
    assert first == {"imported": 2, "skipped_existing": 0, "skipped_deleted": 0}
    second = store.import_from_mem0(stub)
    assert second["imported"] == 0
    assert second["skipped_existing"] == 2
    assert len(_live_texts(store)) == 2


def test_import_from_mem0_skips_deleted(store):
    """账本里删了的，迁移也不复活。"""
    result = store.add_memory_text("我喜欢喝可乐", source="user")
    assert store.delete_memory(result["id"], approve=True)["deleted"] is True
    stub = _StubMem0([{"text": "我喜欢喝可乐", "metadata": {}}])
    out = store.import_from_mem0(stub)
    assert out["skipped_deleted"] == 1
    assert out["imported"] == 0
    assert "我喜欢喝可乐" not in _live_texts(store)


# ---------- 6. 检索与未实现项 ----------


def test_import_normalizes_confirmed_from_user_confirmed(store):
    """读时归一化：Chroma/账本来的旧 metadata 里没有 confirmed 键，
    search() 必须按 user_confirmed 现算出来——不然 source=user 的
    已确认记忆会被问答提示词当成【来源未确认】（迁移退化）。"""
    stub = _StubMem0(
        [
            {"text": "我爱的是冒菜，不是麻辣烫",
             "metadata": {"source": "user", "user_confirmed": True}},
            {"text": "BOOS 喜欢吃牛肉面",
             "metadata": {"source": "inferred", "user_confirmed": False}},
        ]
    )
    store.import_from_mem0(stub)
    maocai = store.search("冒菜", top_k=1)[0]
    assert maocai["metadata"]["confirmed"] is True
    assert maocai["confirmed"] is True
    beef = store.search("牛肉面", top_k=1)[0]
    assert beef["metadata"]["confirmed"] is False
    assert beef["confirmed"] is False


def test_search_verbatim_first(store):
    """原样匹配排最前，分数 1.0。"""
    store.add_memory_text("我今天晚上吃了米线", source="inferred")
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    hits = store.search("米线", top_k=3)
    assert hits[0]["text"] == "我今天晚上吃了米线"
    assert hits[0]["score"] == 1.0


def test_search_semantic_with_fake_vectors(tmp_path, monkeypatch):
    """语义路：向量打桩成定值，含“米线”的句向量 [1, 0]，其余 [0, 1]；
    查“米线”时语义分把米线那条顶上去。"""

    def _fake_embed(self, text):
        if "米线" in (text or ""):
            return np.array([1.0, 0.0], dtype="float32")
        return np.array([0.0, 1.0], dtype="float32")

    monkeypatch.setattr(SQLiteStore, "_embed_text", _fake_embed)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.add_memory_text("我今天晚上吃了米线", source="inferred")
    s.add_memory_text("我喜欢吃火锅", source="inferred")
    hits = s.search("米线好吃吗", top_k=2)
    assert hits[0]["text"] == "我今天晚上吃了米线"


def test_correct_memory_not_yet_implemented(store):
    """纠错还没接：明确报未实现，不悄悄猜着改。"""
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    out = store.correct_memory("我喜欢吃火锅", "我喜欢吃冒菜")
    assert out.get("status") == "failed"
    assert out.get("reason") == "sqlite_correct_not_yet"
    assert "我喜欢吃火锅" in _live_texts(store)
