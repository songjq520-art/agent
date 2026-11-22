# tests/test_sqlite_store.py（SQLite 第二步第一部分的回归测试，2026-11-21）
# 只测 memory/sqlite_store.py；全部离线（向量打桩，不加载 bge 模型），
# 用 tmp_path，不碰真实账本、真实 Chroma、不联网。
# 依据：~/workspace/十二网页测-2026-11-21.md（病根 + 验收）。
#
# 覆盖：
#   1. save_turn 原样存事实、留回执（饺子/豆浆案的根治）
#   2. 问句/闲聊不存；同一 turn_id 只存一次
#   3. 日期换算：昨天/今天/前天/明天按 Asia/Shanghai 的今天算
#   4. 来源标签：原样存的事实是 user（已确认），不是 unconfirmed——
#      标错了问答提示词会把它当“没来源”处理、问起来只说不知道
#      （第一轮线上验收就是这么卡住的）；confirmed 键对齐 mem0 _provenance。
#      2026-11-22 BOOS 定选 A：save_turn 只存用户亲口说的话，一律按已确认算
#      （之前标 inferred，S10 里新事实赢不了已确认旧行、被 pending_new 拦掉）。
#   5. 对账只重建当前态（被取代/删了的不进索引）
#   6. Chroma 一次性迁移幂等、跳过已删
#   7. 检索原样匹配在前；correct_memory 已接线（2026-11-22，见第 13 节）
#   8. 账本是真相源：删库重建后能从账本对账恢复

from datetime import date, timedelta
import json
import threading

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


def _is_superseded(store, text):
    """某条文本的行是不是已被取代（superseded=1）。"""
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


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
# 2026-11-22 BOOS 定选 A 之后：save_turn 只存用户亲口说的话，来源是
# user（已确认），不再是 inferred。


def test_save_turn_source_is_user_confirmed(store):
    """S10（2026-11-22，BOOS 定选 A）：save_turn 只存用户本轮直接说的
    话，来源是 user（已确认），不是 inferred——之前标 inferred/0.7，
    新事实赢不了已确认的旧行，会被槽位裁决 pending_new 拦掉（郑州案）。
    unconfirmed 仍是留给“完全没来源信息”的旧记忆的桶；模型推断的
    事实不走 save_turn，仍按 inferred 存（见低可信度挂起的测试）。"""
    store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线"}],
        turn_id="t-src",
    )
    meta = store.search("米线", top_k=1)[0]["metadata"]
    assert meta["source"] == "user"
    assert meta["confirmed"] is True
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
    assert first == {"imported": 2, "skipped_existing": 0, "skipped_deleted": 0, "skipped_sensitive": 0}
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


def test_correct_memory_updates_exact_match(store):
    """纠错已接线（2026-11-22）：原样在库里时直接改那条。
    追加式：旧行标被取代、新说法另存一条，原话留在账本里。"""
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    out = store.correct_memory("我喜欢吃火锅", "我喜欢吃冒菜")
    assert out.get("status") == "updated"
    assert out.get("old_text") == "我喜欢吃火锅"
    assert out.get("new_text") == "我喜欢吃冒菜"
    assert out.get("layer") == "exact"
    assert _live_texts(store) == ["我喜欢吃冒菜"]
    assert "我喜欢吃火锅" not in _live_texts(store)
    lines = _ledger_lines(store)
    assert any('"type": "supersede"' in line for line in lines)
    assert any("我喜欢吃火锅" in line for line in lines)


def test_correct_memory_not_found(store):
    """纠错找不到旧记忆时报 not_found，原记忆不动、不瞎改。"""
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    out = store.correct_memory("我住北京", "我住上海")
    assert out.get("status") == "not_found"
    assert out.get("layer") == "none"
    assert _live_texts(store) == ["我喜欢吃火锅"]


def test_correct_memory_unchanged_when_already_correct(store):
    """新说法和旧的一样时报 unchanged，不重复存。"""
    store.add_memory_text("我喜欢吃冒菜", source="inferred")
    out = store.correct_memory("我喜欢吃冒菜", "我喜欢吃冒菜")
    assert out.get("status") == "unchanged"
    assert _live_texts(store) == ["我喜欢吃冒菜"]
    assert len(_ledger_lines(store)) == 1  # 只一笔 add，没多写 supersede


def test_correct_memory_ambiguous_with_two_candidates(store):
    """两条都像、没有原样时报 ambiguous，不猜改哪条。"""
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    store.add_memory_text("我喜欢吃冒菜", source="inferred")
    out = store.correct_memory("我喜欢吃辣的", "我喜欢吃米线")
    assert out.get("status") == "ambiguous"
    assert len(out.get("candidates") or []) == 2
    assert set(_live_texts(store)) == {"我喜欢吃火锅", "我喜欢吃冒菜"}


def test_save_turn_correction_updates_old_memory(store):
    """save_turn 端到端纠错：先存“我住北京”，再说“你记错了，我住上海”，
    旧的被取代、新的存下来，回执里带 correction（2026-11-22 接线）。"""
    r1 = store.save_turn(
        [{"role": "user", "content": "我住北京"}], turn_id="t-home-1",
    )
    assert r1["stored"] == ["我住北京"]
    r2 = store.save_turn(
        [{"role": "user", "content": "你记错了，我住上海"}],
        turn_id="t-home-2",
    )
    assert (r2.get("correction") or {}).get("status") == "updated"
    assert r2["stored"] == []
    assert _live_texts(store) == ["我住上海"]


def test_save_turn_correction_not_found_saves_new_fact(store):
    """纠错找不到旧记忆时（not_found）：新说法当新记忆存下来，
    回执里同样带 correction（mem0 同款）。"""
    r = store.save_turn(
        [{"role": "user", "content": "你记错了，我住上海"}],
        turn_id="t-home-nf",
    )
    assert (r.get("correction") or {}).get("status") == "not_found"
    assert r["stored"] == ["我住上海"]
    assert _live_texts(store) == ["我住上海"]


def test_save_turn_same_turn_id_returns_cached_receipt_with_correction(store):
    """根因 a（2026-11-22）：同一 turn_id 调两次 save_turn，第二次返回
    上次的回执（含 correction），不重复存、纠正结果不丢。"""
    store.save_turn(
        [{"role": "user", "content": "我住北京"}], turn_id="t-dup-corr-1",
    )
    first = store.save_turn(
        [{"role": "user", "content": "你记错了，我住上海"}],
        turn_id="t-dup-corr-2",
    )
    assert (first.get("correction") or {}).get("status") == "updated"
    second = store.save_turn(
        [{"role": "user", "content": "你记错了，我住上海"}],
        turn_id="t-dup-corr-2",
    )
    assert second.get("duplicate_turn") is True
    assert (second.get("correction") or {}).get("status") == "updated"
    assert _live_texts(store) == ["我住上海"]


def test_save_turn_ambiguous_correction_still_stores_new_fact(store):
    """根因 c（2026-11-22）：纠正判 ambiguous 时，新说法仍当新事实存下来，
    回执里 correction 状态照实是 ambiguous（不是 updated，也不是丢了）。"""
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    store.add_memory_text("我喜欢吃冒菜", source="inferred")
    r = store.save_turn(
        [{"role": "user", "content": "不对，我喜欢吃米线"}],
        turn_id="t-amb-new",
    )
    assert (r.get("correction") or {}).get("status") == "ambiguous"
    assert "我喜欢吃米线" in _live_texts(store)
    assert "我喜欢吃火锅" in _live_texts(store)
    assert "我喜欢吃冒菜" in _live_texts(store)


def _tea_fake_embed(self, text):
    """测试用假向量：茶类（红/绿/花）同向，其余正交。"""
    if any(word in (text or "") for word in ("红茶", "绿茶", "花茶")):
        return [1.0, 0.1]
    return [0.0, 1.0]


def test_correct_memory_multi_candidate_filters_by_similarity(store, monkeypatch):
    """根因 b（2026-11-22）：多候选时按相似度筛——只留 ≥0.70 的；
    筛后剩 1 条就取代它，不再一刀切判 ambiguous。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", _tea_fake_embed)
    store.add_memory_text("我喜欢喝红茶", source="inferred")
    store.add_memory_text("我喜欢喝咖啡", source="inferred")
    store.add_memory_text("我喜欢吃包子", source="inferred")
    store.add_memory_text("我喜欢去爬山", source="inferred")
    out = store.correct_memory("我喜欢喝绿茶", "我喜欢喝绿茶")
    assert out.get("status") == "updated"
    assert out.get("old_text") == "我喜欢喝红茶"
    assert len(out.get("candidates") or []) >= 2  # 确实走了多候选那条路
    assert "我喜欢喝绿茶" in _live_texts(store)
    assert "我喜欢喝红茶" not in _live_texts(store)
    assert "我喜欢喝咖啡" in _live_texts(store)


def test_correct_memory_multi_candidate_all_high_similarity_is_ambiguous(store, monkeypatch):
    """根因 b：多条候选都 ≥0.70 时仍判 ambiguous，不猜改哪条。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", _tea_fake_embed)
    store.add_memory_text("我喜欢喝红茶", source="inferred")
    store.add_memory_text("我喜欢喝花茶", source="inferred")
    out = store.correct_memory("我喜欢喝绿茶", "我喜欢喝绿茶")
    assert out.get("status") == "ambiguous"
    assert len(out.get("candidates") or []) == 2
    assert set(_live_texts(store)) == {"我喜欢喝红茶", "我喜欢喝花茶"}


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
    assert all(h["decay_weight"] == pytest.approx(1.0, abs=1e-3) for h in hits)


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
    "confirmed","decay_weight","is_history"} 这 8 个键，decay_weight 是 1.0。
    2026-11-22 decay 接线后权重是真算出来的：刚存的记忆闲置约 0 天，
    权重 ≈1.0（不是字面 1.0），所以用 approx 断言。"""
    store.add_memory_text("我今天晚上吃了米线", source="inferred")
    hit = store.search("米线", top_k=1)[0]
    assert set(hit.keys()) == {
        "id", "text", "score", "source", "metadata",
        "confirmed", "decay_weight", "is_history",
    }
    assert hit["decay_weight"] == pytest.approx(1.0, abs=1e-3)
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
    assert all(h["decay_weight"] == pytest.approx(1.0, abs=1e-3) for h in hits)


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


def test_revive_memory_restores_superseded_row(store):
    """被错取代的记忆能恢复：revive 后行重新变活，账本里有一笔 REVIVE 事件。"""
    added = store.add_memory_text("我住广州", source="user")
    gid = added.get("id")
    assert gid
    out = store.correct_memory("我住广州", "我住杭州")
    assert out.get("status") == "updated"
    assert "我住广州" not in _live_texts(store)
    revived = store.revive_memory(gid)
    assert revived.get("status") == "revived"
    assert revived.get("seq") is not None
    assert "我住广州" in _live_texts(store)
    assert "我住杭州" in _live_texts(store)
    lines = _ledger_lines(store)
    assert any('"type": "revive"' in line for line in lines)
    assert store.revive_memory(gid).get("status") == "unchanged"
    assert store.revive_memory("no-such-id").get("status") == "not_found"


def test_slot_conflict_same_credibility_new_wins_even_with_smaller_seq(store):
    """S1（2026-11-22，BOOS 定：同可信度时新的赢，最新说的算数）：
    旧行写序号比新探测大（服务重启后 _WRITE_SEQ 归零的真实情形），
    新说法仍取代旧的，不再 keep_old 丢掉。"""
    store.add_memory_text("我住在南京", source="inferred")
    row = store._conn.execute(
        "SELECT id, metadata_json FROM memories WHERE text=?", ("我住在南京",)
    ).fetchone()
    meta = json.loads(row[1] or "{}")
    meta["write_seq"] = 10 ** 9
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (json.dumps(meta, ensure_ascii=False), row[0]),
    )
    store._conn.commit()
    r = store.save_turn(
        [{"role": "user", "content": "我住在苏州"}], turn_id="t-slot-s1",
    )
    assert "我住在苏州" in _live_texts(store)
    assert "我住在南京" not in _live_texts(store)
    assert r["stored"] == ["我住在苏州"]
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))


def test_save_turn_correction_utterance_goes_through_correction(store):
    """纠错已接线（2026-11-22）：“你记错了，我住上海”走 correct_memory，
    不当成事实污染进库；句中含“记错了”的正常陈述仍照常存。"""
    r1 = store.save_turn(
        [{"role": "user", "content": "你记错了，我住上海"}],
        turn_id="t-corr",
    )
    assert (r1.get("correction") or {}).get("status") == "not_found"
    assert r1["stored"] == ["我住上海"]
    assert _live_texts(store) == ["我住上海"]

    r2 = store.save_turn(
        [{"role": "user", "content": "我昨天记错了开会时间"}],
        turn_id="t-normal",
    )
    assert "correction" not in r2
    assert r2["stored"] == ["我昨天记错了开会时间"]
# ---------- 10. Gemini 第二轮：敏感信息拦截（#1）+ 单连接多线程锁（#2） ----------


def _plant_memory_row(store, text):
    """绕过 add 入口、直接插一行（模拟迁移带进来的旧敏感记忆）。

    只在单线程测试里用：直接调 _insert_memory_row，不走锁、不记账本。
    """
    from memory import ledger as _ledger
    from memory.timeutil import _now_iso

    created_at = _now_iso()
    key = _ledger.ledger_event_key(text, created_at)
    store._insert_memory_row(
        key,
        text,
        {
            "source": "inferred",
            "confidence": 0.7,
            "user_confirmed": False,
            "created_at": created_at,
            "write_seq": 1,
            "turn_id": None,
            "slot": None,
            "is_rule": False,
            "curated": False,
            "decay_kind": "episodic",
            "importance": "normal",
            "event_date": None,
        },
        None,
    )
    return key


def test_add_memory_text_blocks_sensitive_password_sentence(store):
    """含密码特征的句子不进账本、不进 SQLite，返回写明原因。"""
    out = store.add_memory_text("我的密码是 abc12345", source="user")
    assert out.get("event") == "SENSITIVE"
    assert out.get("reason") == "含敏感信息"
    assert _live_texts(store) == []
    assert _ledger_lines(store) == []


def test_add_memory_text_blocks_sensitive_card_number(store):
    """过 Luhn 的卡号同样拦下：根本没存，get_all(True) 里也没有。"""
    out = store.add_memory_text("我的银行卡号是 4111111111111111", source="user")
    assert out.get("event") == "SENSITIVE"
    assert _live_texts(store) == []
    assert store.get_all(include_sensitive=True) == []
    assert _ledger_lines(store) == []


def test_save_turn_drops_sensitive_sentence_with_reason(store):
    """save_turn 回执：敏感句进 dropped、写明“含敏感信息”，正常句照存。"""
    receipt = store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线。我的密码是 abc12345。"}],
        turn_id="t-sensitive",
    )
    assert receipt["stored"] == ["我今天晚上吃了米线"]
    assert receipt["dropped"] == [{"text": "我的密码是 abc12345", "reason": "含敏感信息"}]
    assert _live_texts(store) == ["我今天晚上吃了米线"]


def test_get_all_filters_sensitive_unless_included(store):
    """已入库的敏感记忆：get_all() 默认不返回，get_all(True) 才全量。"""
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    _plant_memory_row(store, "我的密码是 abc12345")
    assert _live_texts(store) == ["我喜欢吃火锅"]
    assert {it["text"] for it in store.get_all(include_sensitive=True)} == {
        "我喜欢吃火锅",
        "我的密码是 abc12345",
    }


def test_search_does_not_return_sensitive_memory(store):
    """search 不返回已入库的敏感记忆（不注入提示词），正常记忆照常。"""
    store.add_memory_text("我喜欢吃火锅", source="inferred")
    _plant_memory_row(store, "我的密码是 abc12345")
    hits = store.search("密码", top_k=5)
    assert all(h["text"] != "我的密码是 abc12345" for h in hits)
    hits2 = store.search("火锅", top_k=5)
    assert hits2 and hits2[0]["text"] == "我喜欢吃火锅"


def test_search_ledger_fallback_skips_sensitive(store):
    """账本兜底也不补敏感条目：账本里有、索引里没有的敏感原话不返回。

    直接调 _search_inner：search() 会把异常吞掉返回 []，测不出过滤本身。
    """
    from memory import ledger as _ledger

    store.add_memory_text("我喜欢吃火锅", source="inferred")
    key = _ledger.ledger_event_key(
        "我的银行卡号是 4111111111111111，平时用它交水电费",
        "2026-11-20T00:00:00Z",
    )
    _ledger.append_event(
        store.ledger_path,
        {
            "type": _ledger.LEDGER_EVENT_ADD,
            "key": key,
            "text": "我的银行卡号是 4111111111111111，平时用它交水电费",
            "meta": {"source": "inferred", "created_at": "2026-11-20T00:00:00Z"},
        },
    )
    hits = store._search_inner("银行卡号交水电费", top_k=5)
    assert all("4111111111111111" not in h["text"] for h in hits)


def test_add_memory_text_nested_embed_lock_is_reentrant(tmp_path):
    """RLock 亲手验证：add 持锁时调真实 _embed_text（缓存查库也加锁）。

    不是 RLock 的话这里会死锁：放线程里跑、10 秒超时判失败，不挂住全套。
    _embedder_failed=True 只走缓存查库那条锁路，不加载模型。
    """
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s._embedder_failed = True
    done = []

    def _run():
        try:
            done.append(s.add_memory_text("我喜欢吃火锅", source="inferred"))
        except Exception as exc:  # noqa: BLE001 - 死锁之外的错也要记下来
            done.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "add_memory_text 在 _embed_text 里死锁了"
    assert len(done) == 1 and isinstance(done[0], dict)
    assert done[0].get("event") == "ADD"


def test_concurrent_search_and_add_no_errors(store):
    """多线程并发：3 写 × 30 条 add + 3 读 × 100 次检索/列表。

    不抛 OperationalError/InterfaceError，库里 90 条一字不少。
    读路直接调 _search_inner：search() 会把异常吞掉返回 []，测不出并发错。
    """
    errors = []

    def _writer(n):
        try:
            for i in range(30):
                store.add_memory_text(
                    "我今天晚上吃了第%d号米线第%d碗" % (n, i), source="inferred"
                )
        except Exception as exc:  # noqa: BLE001 - 并发错都记下来再断言
            errors.append(exc)

    def _reader():
        try:
            for _ in range(100):
                store._search_inner("米线", top_k=5)
                store.get_all()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(w,)) for w in range(3)]
    threads += [threading.Thread(target=_reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "并发线程挂住了"
    assert errors == []
    assert len(_live_texts(store)) == 90


# ---------- 11. Gemini 第三轮：串扰 Bug + 正则前缀 + 锁收拢 + 迁移拦截 ----------


def test_direct_add_resets_retraction_marker(store):
    """“记住”直存不走 save_turn：直存后“撤回”只作废直存的那 1 条，
    上一轮 save_turn 存的 2 条完好（串扰 Bug，2026-11-21 Gemini 第三轮）。"""
    first = store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线。顺便喝了一杯豆浆。"}],
        turn_id="t-round3-a",
    )
    assert len(first["stored"]) == 2
    saved = store.add_memory_text("我下周三出差", source="user")
    assert saved.get("event") == "ADD"
    assert len(_live_texts(store)) == 3

    retract = store.save_turn(
        [{"role": "user", "content": "撤回"}],
        turn_id="t-round3-retract",
    )
    assert retract["retracted"] is True
    assert retract["dropped_count"] == 1
    assert set(_live_texts(store)) == {"我今天晚上吃了米线", "顺便喝了一杯豆浆"}


def test_save_turn_without_turn_id_retracts_whole_turn(store):
    """没传 turn_id 的多句轮次：内部兜底 ID 让每句带同一个 turn_id，
    “撤回”整轮作废，不退化成只作废最后 1 条。"""
    first = store.save_turn(
        [{"role": "user", "content": "我今天晚上吃了米线。顺便喝了一杯豆浆。"}]
    )
    assert len(first["stored"]) == 2
    assert first["turn_id"].startswith("auto-")
    assert len(_live_texts(store)) == 2

    retract = store.save_turn([{"role": "user", "content": "撤回"}])
    assert retract["retracted"] is True
    assert retract["dropped_count"] == 2
    assert _live_texts(store) == []


def test_correction_with_prefix_goes_through_correction(store):
    """“不对/不是”开头的纠错句也走 correct_memory（2026-11-22 接线）；
    句中含“记错了”的正常陈述仍照常存。"""
    r1 = store.save_turn(
        [{"role": "user", "content": "不对，你记错了，我住上海"}],
        turn_id="t-round3-corr1",
    )
    assert (r1.get("correction") or {}).get("status") == "not_found"
    assert r1["stored"] == ["我住上海"]
    r2 = store.save_turn(
        [{"role": "user", "content": "不是，记错了，我住广州"}],
        turn_id="t-round3-corr2",
    )
    # “记错了”已补进 _detect_correction 的说法里：裸“记错了”也认作纠错，
    # 新说法“我住广州”走 correct_memory，取代“我住上海”。
    assert "correction" in r2
    assert (r2.get("correction") or {}).get("status") == "updated"
    assert r2["stored"] == []
    assert set(_live_texts(store)) == {"我住广州"}

    r3 = store.save_turn(
        [{"role": "user", "content": "我昨天记错了开会时间"}],
        turn_id="t-round3-corr3",
    )
    assert "correction" not in r3
    assert r3["stored"] == ["我昨天记错了开会时间"]


def test_import_from_mem0_skips_sensitive(store):
    """迁移时也拦敏感行：敏感行不在账本、不在 SQLite，正常行正常迁入，
    返回里 skipped_sensitive == 1（2026-11-21 Gemini 第三轮，BOOS 定）。"""
    stub = _StubMem0(
        [
            {"text": "我的密码是 abc12345", "metadata": {"source": "user"}},
            {"text": "我喜欢吃火锅", "metadata": {"source": "user"}},
        ]
    )
    out = store.import_from_mem0(stub)
    assert out["imported"] == 1
    assert out["skipped_sensitive"] == 1
    assert out["skipped_existing"] == 0
    assert out["skipped_deleted"] == 0
    assert _live_texts(store) == ["我喜欢吃火锅"]
    assert all("abc12345" not in line for line in _ledger_lines(store))


# ---------- 12. 网页测第二轮：确认版优先（#2）+ 删除范围真查库（#6）
#       + “忘记…”指令句不存（#7）+ “记住”前缀剥掉再存（#8） ----------


def test_search_prefers_confirmed_copy_over_inferred(store):
    """包子同文本：已确认版赢过推断版，search 只返回确认版（#2）。

    生产现状：seq 117 是 BOOS 确认过的“我今天早上吃了俩包子”，
    另有一行同文本的推断版；以前先到先得，推断版抢了名额，
    问起来还加一句“没逐条确认过”。
    2026-11-22 Bug 2 去重改了行为：同文只留一行（确认版），不再两行
    并存、只在检索层聚合——断言改成 1 行。
    """
    saved = store.add_memory_text("我今天早上吃了俩包子", source="user")
    store.confirm_memory(saved["id"])
    store.save_turn(
        [{"role": "user", "content": "我今天早上吃了俩包子"}],
        turn_id="t-baozi-dup",
    )
    assert len(_live_texts(store)) == 1  # Bug 2 去重：同文只留一行（确认版）
    results = store.search("我今天早上吃了什么", top_k=5)
    assert results, "包子该搜得到"
    assert results[0]["text"] == "我今天早上吃了俩包子"
    assert results[0]["confirmed"] is True
    assert len([r for r in results if r["text"] == "我今天早上吃了俩包子"]) == 1


def test_search_confirmed_beats_identical_unconfirmed(store):
    """确认优先（#2）：关键词重合完全相同、分也完全相同时，确认版排在未确认版前面。

    两行除了基金名以外一字不差、等长，BM25 分数相等，排位只由确认
    优先（平局裁决）决定——万家（确认）必须在恒生（未确认）前面。
    未确认行直接用 inferred 直存来种：2026-11-22 BOOS 定选 A 之后
    save_turn 存的都是已确认版，搭不出“一个确认、一个未确认”的对照
    （见 test_save_turn_source_is_user_confirmed）。"""
    saved = store.add_memory_text("我每天定投 5 块买万家纳指", source="user")
    store.confirm_memory(saved["id"])
    store.add_memory_text("我每天定投 5 块买恒生指数", source="inferred")
    results = store.search("我每天定投 5 块买什么", top_k=5)
    assert [r["text"] for r in results] == [
        "我每天定投 5 块买万家纳指",
        "我每天定投 5 块买恒生指数",
    ]


def test_deletion_scope_finds_live_memory(store):
    """删除范围真查库（#6）：在库的记忆 found=True，带原文和范围。"""
    saved = store.add_memory_text("我喜欢喝茶", source="user")
    assert saved.get("event") == "ADD"
    scope = store.deletion_scope(saved["id"])
    assert scope["found"] is True
    assert scope["memory"]["text"] == "我喜欢喝茶"
    assert scope["memory"]["id"] == saved["id"]
    assert scope["in_scope"] == [scope["memory"]]
    assert scope["out_of_scope"] == []
    assert isinstance(scope["history_entries"], list)
    assert scope["history_count"] == len(scope["history_entries"])


def test_deletion_scope_missing_returns_not_found(store):
    """删除范围真查库（#6）：不存在的 id 和空 id 都返回 found=False。"""
    assert store.deletion_scope("不存在的id")["found"] is False
    assert store.deletion_scope("")["found"] is False


def test_deletion_scope_after_delete_returns_not_found(store):
    """删除范围真查库（#6）：删了的记忆不再出现在删除范围里。"""
    saved = store.add_memory_text("我喜欢喝茶", source="user")
    deleted = store.delete_memory(saved["id"], approve=True)
    assert deleted.get("deleted") is True
    assert store.deletion_scope(saved["id"])["found"] is False


def test_deletion_scope_finds_by_ledger_key(store):
    """删除范围真查库（#6）：传账本 key 也能找到（delete_memory 同款两路查找）。"""
    saved = store.add_memory_text("我喜欢喝茶", source="user")
    assert saved.get("event") == "ADD"
    state = store.ledger_state()
    key = (state or {}).get("adds", [{}])[0].get("key")
    assert key
    scope = store.deletion_scope(key)
    assert scope["found"] is True
    assert scope["memory"]["text"] == "我喜欢喝茶"


def test_forget_instruction_sentences_not_stored(store):
    """“忘记…”指令句不存成记忆（#7）：句首和句中的删除指令都拦下。"""
    cases = [
        "忘记我喜欢喝茶",
        "帮我忘记我喜欢喝茶",
        "请忘记我喜欢喝茶",
        "忘记这个",
        "删掉这条",
        "你能不能忘记这个",
        "删除这条记忆",
    ]
    for i, text in enumerate(cases):
        receipt = store.save_turn(
            [{"role": "user", "content": text}],
            turn_id="t-forget-%d" % i,
        )
        assert receipt["stored"] == [], text
    assert _live_texts(store) == []


def test_recollection_sentences_still_stored(store):
    """回忆句照常存（#7）：句尾带“了”的“忘记带钥匙了”不是删除指令。"""
    for i, text in enumerate(["忘记带钥匙了", "我昨天忘记带钥匙了", "忘记了带钥匙"]):
        receipt = store.save_turn(
            [{"role": "user", "content": text}],
            turn_id="t-recall-%d" % i,
        )
        assert receipt["stored"] == [text], text
    assert len(_live_texts(store)) == 3


def test_strip_remember_prefix():
    """剥“记住”前缀（#8）：只剥句首，句中的“记住”不动。"""
    from memory.sqlite_store import _strip_remember_prefix

    assert _strip_remember_prefix("记住后天要去体检") == "后天要去体检"
    assert _strip_remember_prefix("帮我记住后天要去体检") == "后天要去体检"
    assert _strip_remember_prefix("请记住：后天要去体检") == "后天要去体检"
    assert _strip_remember_prefix("记住：后天要去体检") == "后天要去体检"
    assert _strip_remember_prefix("记住") == ""
    assert _strip_remember_prefix("我记住这件事了") == "我记住这件事了"


def test_remember_prefix_stripped_before_store(store):
    """“记住”两字不进记忆（#8）：“记住后天要去体检”只存“后天要去体检”。"""
    receipt = store.save_turn(
        [{"role": "user", "content": "记住后天要去体检"}],
        turn_id="t-remember-strip",
    )
    assert receipt["stored"] == ["后天要去体检"]
    assert _live_texts(store) == ["后天要去体检"]


def test_remember_prefix_variants_stripped(store):
    """“记住”前缀变体（#8）：“帮我记住/请记住/记住：/记住，”都剥掉再存。"""
    cases = [
        ("帮我记住后天要去体检", "后天要去体检"),
        ("请记住明天要开会", "明天要开会"),
        ("记住：下周三出差", "下周三出差"),
        ("记住，大后天要去游泳", "大后天要去游泳"),
    ]
    for i, (text, expected) in enumerate(cases):
        receipt = store.save_turn(
            [{"role": "user", "content": text}],
            turn_id="t-remember-var-%d" % i,
        )
        assert receipt["stored"] == [expected], text
    assert set(_live_texts(store)) == {expected for _text, expected in cases}


def test_remember_command_not_stored_twice_when_direct_path_already_saved(store):
    """直接路径已存过同一句时，save_turn 不再存第二遍（#8）。

    “记住……”命令走直接路径先存一条（确认版），同一轮收尾的 save_turn
    剥完前缀发现已在库里就跳过——同一文本不该在账本里多一条 ADD。
    """
    store.add_memory_text("后天要去体检", source="user")
    receipt = store.save_turn(
        [{"role": "user", "content": "记住后天要去体检"}],
        turn_id="t-remember-dup",
    )
    assert receipt["stored"] == []
    assert _live_texts(store) == ["后天要去体检"]


def test_bare_remember_command_dropped(store):
    """光秃秃一句“记住”不存（#8）：剥完没内容，直接丢掉。"""
    receipt = store.save_turn(
        [{"role": "user", "content": "记住"}],
        turn_id="t-remember-bare",
    )
    assert receipt["stored"] == []
    assert _live_texts(store) == []


# ---------- 13. correct_memory + decay 接线（2026-11-22） ----------
# Gemini 第二轮 #4（同文去重 / slot 覆盖）+ #5（历史查询）+ decay.py 接线
# + _looks_like_fact 临时纠错过滤移除 + deletion_scope 账本历史 bug 修。


def _meta_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["metadata"]
    raise AssertionError(f"库里没有 {memory_id}")


def _age_memory(store, memory_id, days):
    """把一条记忆的 last_used_at 拨到几天前（模拟“很久没用”）。"""
    import json as _json
    from datetime import datetime, timedelta, timezone

    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE id=?", (memory_id,)
    ).fetchone()
    meta = _json.loads(row[0])
    old = datetime.now(timezone.utc) - timedelta(days=days)
    meta["last_used_at"] = old.isoformat(timespec="microseconds").replace("+00:00", "Z")
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE id=?",
        (_json.dumps(meta, ensure_ascii=False), memory_id),
    )
    store._conn.commit()


def test_add_same_text_same_source_returns_unchanged(store):
    """Gemini #4 同文去重：同一原文、同一来源再存一次，不存第二行，
    返回 UNCHANGED；账本不另写 add。"""
    first = store.add_memory_text("我喜欢吃火锅", source="inferred")
    assert first.get("event") == "ADD"
    second = store.add_memory_text("我喜欢吃火锅", source="inferred")
    assert second.get("event") == "UNCHANGED"
    assert second.get("id") == first.get("id")
    assert _live_texts(store) == ["我喜欢吃火锅"]
    assert len(_ledger_lines(store)) == 1


def test_add_same_text_different_source_keeps_one_row(store):
    """同文、来源不同（已确认版 vs 推断版）时只留一行（2026-11-22
    Bug 2 去重改了行为：同文去重只看归一化文本，不看来源；以前两行
    并存的旧断言已作废）：检索层只剩确认版那一行。"""
    store.add_memory_text("我喜欢吃火锅", source="user")
    second = store.add_memory_text("我喜欢吃火锅", source="inferred")
    assert second.get("event") == "UNCHANGED"
    assert len(_live_texts(store)) == 1
    results = store.search("我喜欢吃什么", top_k=5)
    assert [r["text"] for r in results] == ["我喜欢吃火锅"]
    assert results[0]["confirmed"] is True


def test_add_same_slot_supersedes_old(store):
    """Gemini #4 slot 覆盖：同槽位、文本不同、同级且新的在后时，
    新的取代旧的（旧行标被取代，账本记 supersede）。"""
    store.add_memory_text("我叫小宋", source="user", slot="user_name")
    out = store.add_memory_text("我叫大明", source="user", slot="user_name")
    assert out.get("event") == "ADD"
    assert _live_texts(store) == ["我叫大明"]
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))


def test_add_same_slot_lower_credibility_pending(store):
    """Gemini #4 slot 覆盖：新说法可信度低于旧的（推断 vs 已确认），
    挂起来待确认，不存、不覆盖。"""
    store.add_memory_text("我叫小宋", source="user", slot="user_name")
    out = store.add_memory_text("我叫大明", source="inferred", slot="user_name")
    assert out.get("event") == "PENDING"
    assert _live_texts(store) == ["我叫小宋"]


def test_search_history_returns_superseded_versions(store):
    """Gemini #5 历史查询：默认只返回活记忆；include_history=True 时，
    在活记忆后面补账本里被取代的旧版本（is_history=True）。"""
    store.add_memory_text("我住北京", source="inferred")
    store.correct_memory("我住北京", "我住上海")
    live = store.search("我住", top_k=5)
    assert [h["text"] for h in live] == ["我住上海"]
    assert all(h["is_history"] is False for h in live)
    hist = store.search("我住", top_k=5, include_history=True)
    texts = [h["text"] for h in hist]
    assert texts[0] == "我住上海"
    assert "我住北京" in texts
    old = [h for h in hist if h["text"] == "我住北京"][0]
    assert old["is_history"] is True
    assert old["source"] == "history"
    assert store.search_history("我住", top_k=5) == hist


def test_deletion_scope_history_after_correction(store):
    """修 bug：deletion_scope 的账本历史不再报 AttributeError，
    纠错后能看到 add + supersede 两笔事件。"""
    store.add_memory_text("我住北京", source="inferred")
    out = store.correct_memory("我住北京", "我住上海")
    scope = store.deletion_scope(out.get("memory_id"))
    assert scope["found"] is True
    assert scope["history_count"] == 2
    types = [(e or {}).get("type") for e in scope["history_entries"]]
    assert "add" in types
    assert "supersede" in types


def test_bare_correction_signal_still_dropped(store):
    """_looks_like_fact 的临时宽过滤已移除（2026-11-22），但光秃秃的
    纠错信号（“你记错了”后面没跟新说法）仍不是事实，不存。"""
    r = store.save_turn(
        [{"role": "user", "content": "你记错了"}],
        turn_id="t-bare-corr",
    )
    assert r["stored"] == []
    assert _live_texts(store) == []


def test_decay_weight_ages_episodic_memory_to_floor(store):
    """decay 接线：60 天没用的一次性经历，权重掉到下限 0.10，
    但照常搜得到（权重只管排位、不管资格）。"""
    saved = store.add_memory_text("我喜欢吃火锅", source="inferred")
    _age_memory(store, saved["id"], 60)
    hits = store.search("我喜欢吃火锅", top_k=5)
    assert [h["text"] for h in hits] == ["我喜欢吃火锅"]
    assert hits[0]["decay_weight"] == pytest.approx(0.10)


def test_occurrence_promotes_after_three_distinct_days(store):
    """同文在不同天、不同会话出现 3 次就晋升成稳定事实（decay.py 的规矩）。"""
    saved = store.add_memory_text("我去盐湖城出差了", source="inferred", turn_id="t-day1")
    mid = saved["id"]
    assert _meta_of(store, mid)["decay_kind"] == "episodic"
    store._record_occurrence(mid, _meta_of(store, mid), "2026-11-14", "t-day2")
    assert _meta_of(store, mid)["decay_kind"] == "episodic"
    promoted = store._record_occurrence(mid, _meta_of(store, mid), "2026-11-15", "t-day3")
    assert promoted is True
    assert _meta_of(store, mid)["decay_kind"] == "stable"
    assert any('"type": "promote"' in line for line in _ledger_lines(store))


def test_sweep_decay_marks_dormant_and_delete_due(store):
    """sweep_decay：40 天没用的一次性经历标沉底，200 天的标待删，
    核心记忆不动；只打标、不删，库里条数不变。"""
    episodic = store.add_memory_text("我上周去了一趟盐湖城出差", source="inferred")
    old = store.add_memory_text("我去年去了一趟广州出差", source="inferred")
    core = store.add_memory_text("我爱吃火锅", source="user")
    _age_memory(store, episodic["id"], 40)
    _age_memory(store, old["id"], 200)
    _age_memory(store, core["id"], 200)
    out = store.sweep_decay()
    assert out == {"dormant": 1, "delete_due": 1}
    assert _meta_of(store, episodic["id"])["decay_state"] == "dormant"
    assert _meta_of(store, old["id"])["decay_state"] == "decay_delete_due"
    assert _meta_of(store, core["id"])["decay_state"] == "verbatim"
    assert len(_live_texts(store)) == 3
    assert any('"type": "dormant"' in line for line in _ledger_lines(store))


def test_note_context_revives_dormant_memory(store):
    """沉底的记忆被用上（note_context）就回到原话态，账本记 REVIVE。"""
    saved = store.add_memory_text("我上周去了一趟盐湖城出差", source="inferred")
    _age_memory(store, saved["id"], 40)
    store.sweep_decay()
    assert _meta_of(store, saved["id"])["decay_state"] == "dormant"
    store.note_context([{"id": saved["id"]}])
    assert _meta_of(store, saved["id"])["decay_state"] == "verbatim"
    assert any('"type": "revive"' in line for line in _ledger_lines(store))


# ---------- 14. 待补测试（2026-11-22，第二轮 review） ----------
# 待补清单见 ~/workspace/十二correct-memory-decay-待补测试-2026-11-22.md。
# 第 4 条按 BOOS 已定行为断言：取代失败时返回 failed、新记忆保留、
# 旧记忆仍活着（不回滚删新行）。


def test_correction_repeat_same_fact_is_unchanged_no_dup(store):
    """待补 1a：库里只有“我住广州”，再说“不对，我住广州”，
    报 unchanged，不产生重复行（账本只有最初那一笔 add）。"""
    store.add_memory_text("我住广州", source="inferred")
    r = store.save_turn(
        [{"role": "user", "content": "不对，我住广州"}],
        turn_id="t-corr-repeat-1",
    )
    assert (r.get("correction") or {}).get("status") == "unchanged"
    assert r["stored"] == []
    assert _live_texts(store) == ["我住广州"]
    assert len(_ledger_lines(store)) == 1


def test_correction_to_existing_fact_reuses_row(store):
    """待补 1b：库里同时有“我住上海”和“我住广州”，说“不对，我住广州”，
    最终只有 1 条“我住广州”：复用已有行、不新插，旧的上海被取代。"""
    # 前置态是“库里已有两条同槽位活行”：add_memory_text 现在会按槽位
    # 裁决（S10 修法 2），走正常入口第二条会直接取代第一条、搭不出这个
    # 前置态；所以用 _plant_memory_row 绕开入口直接插行（模拟修复前
    # 工具路直存留下的旧数据）。测的是 correct_memory 的复用逻辑本身。
    _plant_memory_row(store, "我住上海")
    _plant_memory_row(store, "我住广州")
    r = store.save_turn(
        [{"role": "user", "content": "不对，我住广州"}],
        turn_id="t-corr-reuse-1",
    )
    assert (r.get("correction") or {}).get("status") == "updated"
    assert r["stored"] == []
    assert _live_texts(store) == ["我住广州"]
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))
    assert len([t for t in _live_texts(store) if t == "我住广州"]) == 1


def test_correction_safety_net_blocks_cross_topic_supersede(tmp_path, monkeypatch):
    """待补 2：库里只有“我喜欢科幻电影”，用户说“不对，我住广州”
    （save_turn 走 correct_memory(fact, fact)），安全网用库里旧向量
    old_vec 算相似度：科幻 [1, 0] vs 广州 [0.6, 0.8] 相似度 0.6 < 0.70，
    拦截误取代，报 ambiguous，旧行原样不动、不被取代。2026-11-22
    修复 3 之后：ambiguous 时新说法不再丢，当新事实存下来。"""
    def _fake_embed(self, text):
        if "科幻" in (text or ""):
            return np.array([1.0, 0.0], dtype="float32")
        if "广州" in (text or ""):
            return np.array([0.6, 0.8], dtype="float32")
        return np.array([0.0, 1.0], dtype="float32")

    monkeypatch.setattr(SQLiteStore, "_embed_text", _fake_embed)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.add_memory_text("我喜欢科幻电影", source="inferred")
    r = s.save_turn(
        [{"role": "user", "content": "不对，我住广州"}],
        turn_id="t-corr-safetynet-1",
    )
    correction = r.get("correction") or {}
    assert correction.get("status") == "ambiguous"
    assert (correction.get("audit") or {}).get("method") == "safety_net"
    assert (correction.get("audit") or {}).get("similarity") == pytest.approx(0.6)
    # 修复 3：ambiguous 时新说法当新事实存下来，不再丢。
    assert r["stored"] == ["我住广州"]
    assert set(_live_texts(s)) == {"我喜欢科幻电影", "我住广州"}
    # 安全网的本职还在：旧行没被取代，账本里没多写 supersede。
    assert len(_ledger_lines(s)) == 2  # 科幻 add + 广州 add
    assert not any('"type": "supersede"' in line for line in _ledger_lines(s))


def test_decay_delete_due_revive_and_double_sweep_idempotent(store):
    """待补 3：decay_delete_due 的记忆被用上就复活回 verbatim；
    已是 delete_due 的记忆再扫两次也不重复改写、不重复记账。"""
    revive = store.add_memory_text("我去年去了一趟广州出差", source="inferred")
    stale = store.add_memory_text("我去年去了一趟深圳出差", source="inferred")
    _age_memory(store, revive["id"], 200)
    _age_memory(store, stale["id"], 200)
    out1 = store.sweep_decay()
    assert out1 == {"dormant": 0, "delete_due": 2}
    assert _meta_of(store, revive["id"])["decay_state"] == "decay_delete_due"
    assert _meta_of(store, stale["id"])["decay_state"] == "decay_delete_due"
    # 复活：note_context 后回 verbatim，账本记一笔 REVIVE。
    store.note_context([{"id": revive["id"]}])
    assert _meta_of(store, revive["id"])["decay_state"] == "verbatim"
    assert sum(1 for line in _ledger_lines(store) if '"type": "revive"' in line) == 1
    # 再扫两次：已是 delete_due 的那条不再改写、不再记账；
    # 复活过的那条刚被用过，也不再动。
    before = _ledger_lines(store)
    out2 = store.sweep_decay()
    out3 = store.sweep_decay()
    assert out2 == {"dormant": 0, "delete_due": 0}
    assert out3 == {"dormant": 0, "delete_due": 0}
    assert _ledger_lines(store) == before
    assert _meta_of(store, stale["id"])["decay_state"] == "decay_delete_due"
    assert _meta_of(store, revive["id"])["decay_state"] == "verbatim"
    assert len(_live_texts(store)) == 2


def test_correction_supersede_failure_keeps_new_and_old_alive(store, monkeypatch):
    """待补 4（按 BOOS 已定行为断言）：mock _supersede_row 抛异常，
    _apply_correction_result 返回 failed；取代失败时保留新记忆、
    诚实报 failed（新记忆已进账本，不删），旧记忆仍活着。"""
    store.add_memory_text("我住北京", source="inferred")

    def _boom(self, *args, **kwargs):
        raise RuntimeError("supersede 记账炸了")

    monkeypatch.setattr(SQLiteStore, "_supersede_row", _boom)
    out = store.correct_memory("我住北京", "我住上海")
    assert out.get("status") == "failed"
    assert out.get("reason") == "新记忆已存，但旧记忆未被取代"
    assert out.get("memory_id") is not None
    # 新记忆保留：活记忆里新旧两条都在（不回滚删新行）。
    assert set(_live_texts(store)) == {"我住北京", "我住上海"}
    # 新记忆已进账本；旧记忆仍活着：没有 supersede 事件。
    lines = _ledger_lines(store)
    assert any('"type": "add"' in line and "我住上海" in line for line in lines)
    assert not any('"type": "supersede"' in line for line in lines)


# ---------- 15. Bug 2 回归（2026-11-22）：去重只看归一化文本，不看来源 ----------


def test_add_same_text_user_then_inferred_is_one_row(store):
    """同一文本先 user（已确认）再 inferred 存：只留一条活行，仍是已确认。"""
    r1 = store.add_memory_text("我住广州", source="user")
    assert r1.get("event") == "ADD"
    r2 = store.add_memory_text("我住广州", source="inferred")
    assert r2.get("event") == "UNCHANGED"
    assert r2.get("id") == r1["id"]
    assert _live_texts(store) == ["我住广州"]
    assert _meta_of(store, r1["id"]).get("user_confirmed") is True
    assert len(_ledger_lines(store)) == 1  # 没多写 add


def test_add_same_text_inferred_then_user_upgrades_to_confirmed(store):
    """同一文本先 inferred 再 user 存：只留一条活行，且升级成已确认
    （source 改 user、decay_kind 改 stable）；升级不另记账本。"""
    r1 = store.add_memory_text("我住广州", source="inferred")
    assert r1.get("event") == "ADD"
    assert _meta_of(store, r1["id"]).get("user_confirmed") is False
    r2 = store.add_memory_text("我住广州", source="user")
    assert r2.get("event") == "UNCHANGED"
    assert r2.get("id") == r1["id"]
    assert _live_texts(store) == ["我住广州"]
    meta = _meta_of(store, r1["id"])
    assert meta.get("user_confirmed") is True
    assert meta.get("source") == "user"
    assert meta.get("decay_kind") == "stable"
    assert len(_ledger_lines(store)) == 1  # 升级不另记账本，只有最初一笔 add


# ---------- 16. 通用纠正（2026-11-22）：纠正检测扩到所有常见说法，
#       save_turn 纠错路由走通，同槽位隐式更新也取代 ----------


def test_detect_correction_general_phrases():
    """纠正检测通用：任务清单里的每种说法都要认出来，并取对新说法；
    不是纠正的（犹豫/词的一部分/过去时叙述/光秃秃指令）不认。"""
    from memory.correction_text import _detect_correction

    cases = [
        ("不对，我住深圳", "我住深圳"),
        ("错了，我住深圳", "我住深圳"),
        ("不是北京，我住深圳", "我住深圳"),
        ("我住深圳，不是北京", "我住深圳"),
        ("不是北京，是深圳", "深圳"),
        ("记错了，我住深圳", "我住深圳"),
        ("我之前说错了，我住深圳", "我住深圳"),
        ("我记错了，我住深圳", "我住深圳"),
        ("说错了，我住深圳", "我住深圳"),
        ("改一下，我住深圳", "我住深圳"),
        ("纠正一下，我住深圳", "我住深圳"),
        ("更正，我住深圳", "我住深圳"),
        ("其实我住深圳", "我住深圳"),
        ("其实是深圳", "深圳"),
        ("应该是深圳", "深圳"),
        ("不对，应该是深圳", "深圳"),
        ("我现在住深圳了", "我住深圳了"),
        ("搬到深圳了", "深圳了"),
        ("换成深圳了", "深圳了"),
        ("我不是住北京，我住深圳", "我住深圳"),
        ("其实不是北京，我住深圳", "我住深圳"),
        ("不对，不是北京，我住深圳", "我住深圳"),
        ("你记错了，我住深圳", "我住深圳"),
        ("不对，我喜欢喝茶", "我喜欢喝茶"),
    ]
    for message, expected in cases:
        detected = _detect_correction(message)
        assert detected is not None, message
        assert detected["fact"] == expected, message
    negatives = [
        "我不是很确定",
        "我觉得不对劲",
        "我昨天记错了开会时间",
        "我昨天说错了开会时间",
        "改一下名字",
        "你记错了",
        "不对",
        "我之前说错了",
        "我现在喜欢用 Go 写代码。",
        "其实也不是很喜欢吃。",
        "不对，不是北京",
        # “我现在”只认带单值槽位的“……了”：数据库改配置不是槽位事实，
        # 得走待确认守卫，不能短路进纠正分支。
        "我现在把数据库改成 PostgreSQL 了。",
        # 新说法本身是撤回指令时不算纠正（撤回那条路处理）。
        "说错了，前面那句不算",
    ]
    for message in negatives:
        assert _detect_correction(message) is None, message


def _correction_store(tmp_path, monkeypatch):
    """一块干净的 SQLiteStore（向量打桩，和 store fixture 同款）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def test_save_turn_correction_e2e_shenzhen(tmp_path, monkeypatch):
    """端到端：库里有“我住北京”，说“不对，我住深圳”→ 回复带
    correction（updated）、账本有 supersede、新事实入库、旧事实被取代。"""
    store = _correction_store(tmp_path, monkeypatch)
    store.add_memory_text("我住北京", source="inferred")
    receipt = store.save_turn(
        [{"role": "user", "content": "不对，我住深圳"}],
        turn_id="t-corr-e2e",
    )
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated", receipt
    assert correction.get("old_text") == "我住北京"
    assert correction.get("new_text") == "我住深圳"
    assert _live_texts(store) == ["我住深圳"]
    lines = _ledger_lines(store)
    assert sum(1 for line in lines if '"type": "add"' in line) == 2
    assert any('"type": "supersede"' in line for line in lines)


def test_save_turn_correction_general_phrases_e2e(tmp_path, monkeypatch):
    """端到端：每种纠正说法走 save_turn 都要真换掉旧记忆（不只“不对”）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    cases = [
        ("我之前说错了，我住深圳", "我住深圳"),
        ("其实我住深圳", "我住深圳"),
        ("我住深圳，不是北京", "我住深圳"),
        ("不是北京，我住深圳", "我住深圳"),
        ("改一下，我住深圳", "我住深圳"),
        ("说错了，我住深圳", "我住深圳"),
        ("我现在住深圳了", "我住深圳了"),
        ("搬到深圳了", "我住深圳"),
        ("换成深圳了", "我住深圳"),
        ("不对，应该是深圳", "我住深圳"),
    ]
    for i, (message, expected_live) in enumerate(cases):
        store = SQLiteStore(memory_dir=str(tmp_path / ("m%d" % i)))
        store.add_memory_text("我住北京", source="inferred")
        receipt = store.save_turn(
            [{"role": "user", "content": message}],
            turn_id="t-corr-gen-%d" % i,
        )
        correction = receipt.get("correction") or {}
        assert correction.get("status") == "updated", (message, receipt)
        assert _live_texts(store) == [expected_live], message
        assert any(
            '"type": "supersede"' in line for line in _ledger_lines(store)
        ), message


def test_save_turn_implicit_update_supersedes_same_slot(tmp_path, monkeypatch):
    """隐式更新：没说纠正词，直接说新事实“我住深圳”，库里“我住北京”
    照样被取代（同槽位不同值）；不相关的“我喜欢喝咖啡”不动。"""
    store = _correction_store(tmp_path, monkeypatch)
    store.add_memory_text("我住北京", source="inferred")
    store.add_memory_text("我喜欢喝咖啡", source="inferred")
    receipt = store.save_turn(
        [{"role": "user", "content": "我住深圳"}],
        turn_id="t-implicit-1",
    )
    assert set(_live_texts(store)) == {"我住深圳", "我喜欢喝咖啡"}
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))
    assert (receipt.get("correction") or {}).get("status") == "updated"
    assert receipt["stored"] == ["我住深圳"]


def test_save_turn_implicit_update_name_slot(tmp_path, monkeypatch):
    """隐式更新（名字槽）：库里“我叫小宋”，新说“我叫大明”→ 大明
    取代小宋；不同槽位（喝咖啡）不互相取代。"""
    store = _correction_store(tmp_path, monkeypatch)
    store.add_memory_text("我叫小宋", source="inferred")
    store.add_memory_text("我喜欢喝咖啡", source="inferred")
    receipt = store.save_turn(
        [{"role": "user", "content": "我叫大明"}],
        turn_id="t-implicit-name",
    )
    assert set(_live_texts(store)) == {"我叫大明", "我喜欢喝咖啡"}
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))
    assert (receipt.get("correction") or {}).get("status") == "updated"


def test_save_turn_implicit_update_supersedes_confirmed_old(tmp_path, monkeypatch):
    """S10（2026-11-22，BOOS 定选 A）：库里“我叫小宋”是已确认版，
    用户亲口说的新说“我叫大明”同样算已确认，同可信度新的赢，直接
    取代旧的、不挂起。模型推断的新事实仍挂起，见
    test_add_same_slot_lower_credibility_pending（inferred 直存那路）。"""
    store = _correction_store(tmp_path, monkeypatch)
    store.add_memory_text("我叫小宋", source="user")
    receipt = store.save_turn(
        [{"role": "user", "content": "我叫大明"}],
        turn_id="t-implicit-pending",
    )
    assert _live_texts(store) == ["我叫大明"]
    assert _is_superseded(store, "我叫小宋")
    assert receipt["stored"] == ["我叫大明"]
    assert (receipt.get("correction") or {}).get("status") == "updated"
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))


def test_save_turn_correction_ambiguous_slot_retry(tmp_path, monkeypatch):
    """纠错分不清改哪条时按槽位再筛：候选有“我住北京”（住处）和
    “我喜欢深圳菜”（偏好），新说法“我住深圳”带住处槽位，恰好只有
    北京同槽，就改北京那条，不用问 BOOS；深圳菜不动。"""
    store = _correction_store(tmp_path, monkeypatch)
    store.add_memory_text("我住北京", source="inferred")
    store.add_memory_text("我喜欢深圳菜", source="inferred")
    receipt = store.save_turn(
        [{"role": "user", "content": "不对，我住深圳"}],
        turn_id="t-corr-slot-retry",
    )
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated", receipt
    assert correction.get("old_text") == "我住北京"
    assert set(_live_texts(store)) == {"我住深圳", "我喜欢深圳菜"}


def test_bare_new_correction_signals_still_dropped(tmp_path, monkeypatch):
    """光秃秃的新纠错信号（后面没跟新说法）仍不是事实，不存：
    “我之前说错了”“改一下”单独说时不进库。"""
    store = _correction_store(tmp_path, monkeypatch)
    for i, message in enumerate(["我之前说错了", "改一下", "说错了"]):
        receipt = store.save_turn(
            [{"role": "user", "content": message}],
            turn_id="t-bare-new-%d" % i,
        )
        assert receipt["stored"] == [], message
    assert _live_texts(store) == []
    assert _ledger_lines(store) == []

# ---------- 16. S10（2026-11-22，BOOS 定选 A） ----------
# 用户亲口说的新事实直接算已确认（user/1.0），能取代已确认的旧事实；
# 工具路 add_memory_text 没传 slot 时也按文本推断 slot、走槽位裁决，
# 不再出现两条活住址并存。模型推断的事实仍按 inferred/0.7、不直接
# 取代已确认旧行（已覆盖：test_add_same_slot_lower_credibility_pending，
# 这里不再重复造）。


def test_s10_user_spoken_new_fact_supersedes_confirmed_old(tmp_path, monkeypatch):
    """S10(a)：库里“我住长沙”是已确认版，用户直接说新事实“我住郑州”
    （没说纠正词）→ 新的取代旧的：旧的 superseded=1，新的活着，
    回执 correction=updated。修之前新事实按 inferred/0.7 探针，
    被 _resolve_conflict 判 pending_new、进 dropped，郑州没入库。"""
    store = _correction_store(tmp_path, monkeypatch)
    store.add_memory_text("我住长沙", source="user")
    receipt = store.save_turn(
        [{"role": "user", "content": "我住郑州"}],
        turn_id="t-s10-user-1",
    )
    assert _live_texts(store) == ["我住郑州"]
    assert _is_superseded(store, "我住长沙")
    assert receipt["stored"] == ["我住郑州"]
    assert (receipt.get("correction") or {}).get("status") == "updated"
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))


def test_s10_user_spoken_now_living_in_zhengzhou(tmp_path, monkeypatch):
    """S10(a) 原句：库里“我住长沙”是已确认版，用户说“我现在住在郑州了”
    → 走纠正分支，新说法入库、长沙被取代，不丢。"""
    store = _correction_store(tmp_path, monkeypatch)
    store.add_memory_text("我住长沙", source="user")
    receipt = store.save_turn(
        [{"role": "user", "content": "我现在住在郑州了"}],
        turn_id="t-s10-user-2",
    )
    assert _live_texts(store) == ["我住在郑州了"]
    assert _is_superseded(store, "我住长沙")
    assert (receipt.get("correction") or {}).get("status") == "updated"
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))


def test_s10_tool_path_add_memory_text_infers_slot(store):
    """S10(b)：工具路 add_memory_text 不传 slot（模型只传文本），
    source="user" 存“我住西安”，库里已有已确认的“我住长沙”→ 西安
    取代长沙，只剩一条活住址。修之前不推断 slot、裁决不跑，西安直存、
    两条活住址并存。"""
    store.add_memory_text("我住长沙", source="user")
    out = store.add_memory_text("我住西安", source="user")
    assert out.get("event") == "ADD"
    assert _live_texts(store) == ["我住西安"]
    assert _is_superseded(store, "我住长沙")
    assert any('"type": "supersede"' in line for line in _ledger_lines(store))


def test_s10_tool_path_supersedes_unstamped_legacy_row(store):
    """S10(b) 旧行没打 slot 标：修复前工具路存的“我住长沙”行 slot 列
    是空的（西安案现状），工具路再存“我住西安”时只查 slot 列会漏掉
    它；按文本现认槽位补查后，长沙照样被取代，只剩一条活住址。"""
    _plant_memory_row(store, "我住长沙")
    # 旧行改成工具路已确认版、但 slot 列保持为空（修复前的真实状态）：
    # 只改来源/确认位，不动其他字段。
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE text=?", ("我住长沙",)
    ).fetchone()
    meta = json.loads(row[0] or "{}")
    meta["source"] = "user"
    meta["confidence"] = 1.0
    meta["user_confirmed"] = True
    store._conn.execute(
        "UPDATE memories SET source='user', user_confirmed=1,"
        " metadata_json=? WHERE text=?",
        (json.dumps(meta, ensure_ascii=False), "我住长沙"),
    )
    store._conn.commit()
    out = store.add_memory_text("我住西安", source="user")
    assert out.get("event") == "ADD"
    assert _live_texts(store) == ["我住西安"]
    assert _is_superseded(store, "我住长沙")
