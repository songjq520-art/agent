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


# ---------- 7. BM25 + RRF 融合检索（第二步第一段，2026-11-21） ----------
# 旧逻辑：memory/bm25.py + mem0_store.py search（commit 7a69699）：
# 宽召回（语义 top_k*4 + BM25 全库 top_k*4）→ 标准 RRF k=60 融合 →
# 原样匹配在前 → 先筛后截。语义路按原始余弦分排位，不是混合分。


def test_bm25_road_recalls_fund_rows_without_vectors(store):
    """BM25 路：向量全无（模型起不来）时，只靠关键词也能把万家、恒生
    两行都召进 Q4 前 5——旧 SQLite 单靠语义时恒生掉在第 7。"""
    store.add_memory_text("2026年11月20日，用户表示之前提到的所有定投计划全部取消，都不再投了。", source="inferred")
    store.add_memory_text("用户计划从 2026-11-12 起每天定投 5 元，一直投到年底（12 月 31 日）。", source="inferred")
    store.add_memory_text("用户表示从今天（2026年11月12日）起每天定投5块，一直投到年底（12月31日）。", source="inferred")
    store.add_memory_text("用户在计算每天5块的定投总额时，要求只按美股交易日计算，不含周末和节假日。", source="inferred")
    store.add_memory_text("是每天定投 5 块钱买万家纳指。基金代码0194442", source="user")
    store.add_memory_text("用户表示现在要每天定投5美元的沪深300。", source="inferred")
    store.add_memory_text("我想每天定投5块恒生", source="user")
    store.add_memory_text("每月预算 5 美元，到 5 美元提醒，不自动停。", source="inferred")
    hits = store.search("定投计划怎么样了", top_k=5)
    texts = [h["text"] for h in hits]
    assert texts == [
        "2026年11月20日，用户表示之前提到的所有定投计划全部取消，都不再投了。",
        "用户计划从 2026-11-12 起每天定投 5 元，一直投到年底（12 月 31 日）。",
        "我想每天定投5块恒生",
        "用户表示现在要每天定投5美元的沪深300。",
        "是每天定投 5 块钱买万家纳指。基金代码0194442",
    ]


def test_rrf_fusion_order_differs_from_each_road(tmp_path, monkeypatch):
    """RRF 融合顺序：语义路 [冒菜，火锅，饺子]、BM25 路 [火锅，饺子，冒菜]，
    融合后必须是 [火锅，冒菜，饺子]——跟两路都不一样，跟 RRF 算出来的一样。
    （BM25 顺序用 memory/bm25.py 的实现当场算过：火锅 0.94、饺子 0.516、
    冒菜 0.431。）"""

    def _fake_embed(self, text):
        vectors = {
            "我爱吃什么": [1.0, 0.0],
            "我爱吃火锅": [0.8, 0.6],
            "我爱的是冒菜": [1.0, 0.0],
            "爱吃饺子": [0.6, 0.8],
        }
        return np.array(vectors[text], dtype="float32")

    monkeypatch.setattr(SQLiteStore, "_embed_text", _fake_embed)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.add_memory_text("我爱吃火锅", source="inferred")
    s.add_memory_text("我爱的是冒菜", source="inferred")
    s.add_memory_text("爱吃饺子", source="inferred")
    hits = s.search("我爱吃什么", top_k=3)
    assert [h["text"] for h in hits] == ["我爱吃火锅", "我爱的是冒菜", "爱吃饺子"]
    assert all(h["decay_weight"] == 1.0 for h in hits)


def test_verbatim_still_first_with_rrf(tmp_path, monkeypatch):
    """原样仍在前：原样命中的那行余弦只有 0.41、另一行 0.99，
    原样行也必须排第一，分数 1.0、source="verbatim"。"""

    def _fake_embed(self, text):
        vectors = {
            "定投5块恒生": [1.0, 0.0],
            "我爱吃火锅": [0.99, float(np.sqrt(1 - 0.99 ** 2))],
            "我想每天定投5块恒生": [0.41, float(np.sqrt(1 - 0.41 ** 2))],
        }
        return np.array(vectors[text], dtype="float32")

    monkeypatch.setattr(SQLiteStore, "_embed_text", _fake_embed)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.add_memory_text("我爱吃火锅", source="inferred")
    s.add_memory_text("我想每天定投5块恒生", source="user")
    hits = s.search("定投5块恒生", top_k=2)
    assert hits[0]["text"] == "我想每天定投5块恒生"
    assert hits[0]["score"] == 1.0
    assert hits[0]["source"] == "verbatim"


# Q4 八行原文 + 2026-11-21 真实 bge 余弦（4 位小数），离线复现线上排名。
# 注意：万家行原文里的基金代码库里实际存的是 0194442（7 位），不是报告里
# 写的 019441——这是旧 mem0 提取时就存下的原文，原样不动。
_Q4_ROWS = [
    ("2026年11月20日，用户表示之前提到的所有定投计划全部取消，都不再投了。", 0.6192, "inferred"),
    ("用户计划从 2026-11-12 起每天定投 5 元，一直投到年底（12 月 31 日）。", 0.6415, "inferred"),
    ("用户表示从今天（2026年11月12日）起每天定投5块，一直投到年底（12月31日）。", 0.6210, "inferred"),
    ("用户在计算每天5块的定投总额时，要求只按美股交易日计算，不含周末和节假日。", 0.4914, "inferred"),
    ("是每天定投 5 块钱买万家纳指。基金代码0194442", 0.5665, "user"),
    ("用户表示现在要每天定投5美元的沪深300。", 0.5551, "inferred"),
    ("我想每天定投5块恒生", 0.5425, "user"),
    ("每月预算 5 美元，到 5 美元提醒，不自动停。", 0.4254, "inferred"),
]


def test_q4_both_fund_rows_in_top5(tmp_path, monkeypatch):
    """Q4 验收：问“定投计划怎么样了”，前 5 必须同时有万家、恒生两行
    （已确认）。语义路按原始余弦排 + BM25 路 + RRF k=60 融合后的顺序，
    和线上真实 bge 跑出来的融合顺序一致。"""
    _cos = {text: cos for text, cos, _ in _Q4_ROWS}

    def _fake_embed(self, text):
        if text == "定投计划怎么样了":
            return np.array([1.0, 0.0], dtype="float32")
        cos = _cos[text]
        return np.array([cos, float(np.sqrt(1 - cos ** 2))], dtype="float32")

    monkeypatch.setattr(SQLiteStore, "_embed_text", _fake_embed)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    for text, _cos_value, source in _Q4_ROWS:
        s.add_memory_text(text, source=source)
    hits = s.search("定投计划怎么样了", top_k=5)
    texts = [h["text"] for h in hits]
    assert texts == [
        "用户计划从 2026-11-12 起每天定投 5 元，一直投到年底（12 月 31 日）。",
        "2026年11月20日，用户表示之前提到的所有定投计划全部取消，都不再投了。",
        "用户表示从今天（2026年11月12日）起每天定投5块，一直投到年底（12月31日）。",
        "我想每天定投5块恒生",
        "是每天定投 5 块钱买万家纳指。基金代码0194442",
    ]
    assert hits[3]["confirmed"] is True
    assert hits[4]["confirmed"] is True


def test_search_result_shape_unchanged(store):
    """返回形状不变：还是 {"id","text","score","source","metadata",
    "confirmed","decay_weight","is_history"} 这 8 个键，decay_weight 是 1.0。"""
    store.add_memory_text("我今天晚上吃了米线", source="inferred")
    hit = store.search("米线", top_k=1)[0]
    assert set(hit.keys()) == {
        "id", "text", "score", "source", "metadata",
        "confirmed", "decay_weight", "is_history",
    }
    assert hit["decay_weight"] == 1.0
    assert hit["is_history"] is False


# ---------- 8. 类型权重：稳定偏好 > 一次性事件（Q3 修法，2026-11-21） ----------
# 双写比对 18 问里唯一退化：问“我爱吃什么”，纠正行“不太喜欢鸡杂面”
# 被新存的饺子/米线两行按余弦挤出前 5（RRF 0.0152 vs 米线 0.0154，
# 一位之差）。修法是类型权重，不是改余弦、不是改阈值、不是精选清单
# （库里 curated 行是 0 条，精选先查接了也修不了这一问）。
# 权重只管排位、不管 0.4 资格线（decay.py 坑 1）；返回形状还是 8 个键。


def test_type_weight_stable_preference_vs_event():
    """_type_weight：说喜欢/不喜欢的偏好句 1.25，一次性事件 1.0。"""
    from memory.sqlite_store import _type_weight

    assert _type_weight("我爱吃火锅") == 1.25
    assert _type_weight("我爱的是冒菜，不是麻辣烫") == 1.25
    assert _type_weight("BOOS 喜欢吃牛肉面。") == 1.25
    assert _type_weight("用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。") == 1.25
    assert _type_weight("BOOS 不喜欢冰美式了，现在喜欢喝美式。") == 1.25
    assert _type_weight("我昨天中午吃了饺子") == 1.0
    assert _type_weight("我今天晚上吃了米线") == 1.0
    assert _type_weight("2026年11月20日，用户表示之前提到的所有定投计划全部取消，都不再投了。") == 1.0
    assert _type_weight("我想每天定投5块恒生") == 1.0
    assert _type_weight("") == 1.0


# Q3 七行原文 + 2026-11-21 真实 bge 余弦（4 位小数），离线复现线上排名。
_Q3_ROWS = [
    ("我爱吃火锅", 0.6558, "user"),
    ("我爱的是冒菜，不是麻辣烫", 0.5676, "user"),
    ("我昨天中午吃了饺子", 0.4789, "inferred"),
    ("BOOS 喜欢吃牛肉面。", 0.4700, "inferred"),
    ("我今天晚上吃了米线", 0.4624, "inferred"),
    ("用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。", 0.4390, "inferred"),
    ("BOOS 喜欢吃鸡杂面。", 0.4298, "inferred"),
]


def test_q3_correction_row_back_in_top5(tmp_path, monkeypatch):
    """Q3 验收：问“我爱吃什么”，前 5 必须有纠正行“不太喜欢鸡杂面”
    （修前它掉到第 6，被饺子/米线挤出；修后前 5 应和旧 mem0 的前 5
    一致：火锅、冒菜、牛肉面喜、纠正行、鸡杂面喜）。
    语义路按原始余弦排 + BM25 路 + RRF k=60 融合，再乘类型权重。"""
    _cos = {text: cos for text, cos, _ in _Q3_ROWS}

    def _fake_embed(self, text):
        if text == "我爱吃什么":
            return np.array([1.0, 0.0], dtype="float32")
        cos = _cos[text]
        return np.array([cos, float(np.sqrt(1 - cos ** 2))], dtype="float32")

    monkeypatch.setattr(SQLiteStore, "_embed_text", _fake_embed)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    for text, _cos_value, source in _Q3_ROWS:
        s.add_memory_text(text, source=source)
    hits = s.search("我爱吃什么", top_k=5)
    texts = [h["text"] for h in hits]
    assert texts == [
        "我爱吃火锅",
        "我爱的是冒菜，不是麻辣烫",
        "BOOS 喜欢吃牛肉面。",
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。",
        "BOOS 喜欢吃鸡杂面。",
    ]
    assert all(h["decay_weight"] == 1.0 for h in hits)


def test_event_question_not_swamped_by_preference_boost(tmp_path, monkeypatch):
    """类型权重不能喧宾夺主：问“我昨天吃了什么”（事件问），饺子那行
    余弦 0.9、BM25 也命中，必须还是第 1——被加权的偏好行（冒菜，
    余弦 0.5）顶不上来。"""
    _cos = {
        "我昨天中午吃了饺子": 0.9,
        "我爱的是冒菜，不是麻辣烫": 0.5,
        "2026年11月19日晚上，用户在盐湖城吃了一碗牛肉面，觉得特别好吃。": 0.45,
    }

    def _fake_embed(self, text):
        if text == "我昨天吃了什么":
            return np.array([1.0, 0.0], dtype="float32")
        cos = _cos[text]
        return np.array([cos, float(np.sqrt(1 - cos ** 2))], dtype="float32")

    monkeypatch.setattr(SQLiteStore, "_embed_text", _fake_embed)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    for text in _cos:
        s.add_memory_text(text, source="inferred")
    hits = s.search("我昨天吃了什么", top_k=3)
    assert hits[0]["text"] == "我昨天中午吃了饺子"
    assert "我爱的是冒菜，不是麻辣烫" in [h["text"] for h in hits]


# ---------- 9. 边界修补：整轮撤回 / 玩笑作废落账本 / 兜底删除 / 纠错句隔离 ----------


def test_void_joke_memory_persists_to_ledger_and_returns_ok(store):
    """作废玩笑记忆必须同步写入账本且返回 ok=True；删库重建后不复活。"""
    saved = store.add_memory_text("我其实是个外星人", source="inferred")
    mem_id = saved["id"]
    res = store.void_joke_memory(mem_id)
    assert res["ok"] is True
    assert res["voided"] is True
    assert _live_texts(store) == []

    # 模拟删库后对账重建
    store._conn.close()
    for suffix in ("", "-wal", "-shm"):
        try:
            (store.db_path.parent / (store.db_path.name + suffix)).unlink()
        except FileNotFoundError:
            pass
    rebuilt = SQLiteStore(memory_dir=str(store.memory_dir))
    rebuilt.reconcile_ledger_index()
    assert _live_texts(rebuilt) == []


def test_bare_retraction_voids_entire_multi_sentence_turn(store):
    """上一轮包含多句话时，光秃秃撤回必须把整轮存下的条目全部作废。"""
    first = store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线。顺便喝了一杯豆浆。"}],
        turn_id="t-multi",
    )
    assert len(first["stored"]) == 2
    assert len(_live_texts(store)) == 2

    retract = store.save_turn(
        [{"role": "user", "content": "撤回"}],
        turn_id="t-retract",
    )
    assert retract["retracted"] is True
    assert retract["dropped_count"] == 2
    assert _live_texts(store) == []


def test_delete_memory_supports_ledger_only_key_and_rejects_missing(store):
    """delete_memory 对不存在/重复删除返回 not_found，对仅在账本中的 key 可正常删除。"""
    from memory import ledger as _ledger

    assert store.delete_memory("non-existent-id", approve=True) == {
        "deleted": False,
        "reason": "not_found",
    }
    key = _ledger.ledger_event_key("仅在账本里的旧记录", "2026-11-20T00:00:00Z")
    _ledger.append_event(
        store.ledger_path,
        {
            "type": _ledger.LEDGER_EVENT_ADD,
            "key": key,
            "text": "仅在账本里的旧记录",
            "meta": {"source": "inferred", "created_at": "2026-11-20T00:00:00Z"},
        },
    )
    assert store.delete_memory(key, approve=True) == {
        "deleted": True,
        "memory_id": key,
    }
    assert store.delete_memory(key, approve=True) == {
        "deleted": False,
        "reason": "not_found",
    }


def test_save_turn_skips_correction_utterance_before_wired(store):
    """纠错未接线前，'你记错了...' 不当成事实污染进库；句中含'记错了'的正常陈述仍保留。"""
    r1 = store.save_turn(
        [{"role": "user", "content": "你记错了，我住上海"}],
        turn_id="t-corr",
    )
    assert r1["stored"] == []
    assert _live_texts(store) == []

    r2 = store.save_turn(
        [{"role": "user", "content": "我昨天记错了开会时间"}],
        turn_id="t-normal",
    )
    assert r2["stored"] == ["我昨天记错了开会时间"]
