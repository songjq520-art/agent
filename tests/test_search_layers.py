# tests/test_search_layers.py（记忆整理：补丁 1 两段式检索 + 补丁 5 BM25）
# 测 search() 的三件事：
#   - 每条结果标 layer：常用层 active / 档案层 archive（沉底态）；
#   - 常用层最好的一条语义分 ≥ 0.6 时档案层不进结果，低于 0.6 时进来；
#   - 关键词补充改走 BM25：原样匹配（基金代码）的排在关键词组最前面，
#     语义漏掉的关键词记忆能被 BM25 捞回来。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。
# 假 search 的分数 = query 窗口命中数 / 该条记忆的窗口总数（见 conftest），
# 下面用例的分数都是按这个算出来的。

from datetime import datetime, timedelta, timezone

from conftest import make_fake_store
from memory.decay import DECAY_STATE_DORMANT, DECAY_STATE_VERBATIM
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    SOURCE_INFERRED,
    _provenance,
)


def _utc_now():
    return datetime.now(timezone.utc)


def _add_inferred(store, text):
    result = store._add_text_with_metadata(
        text, _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    )
    return (result.get("results") or [{}])[0].get("id")


def _age(store, memory_id, days):
    raw = store._raw_metadata_for(memory_id)
    assert raw is not None
    old = _utc_now() - timedelta(days=days)
    raw["last_used_at"] = old.isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    store._memory.update(memory_id=memory_id, metadata=raw)


def _meta_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["metadata"]
    raise AssertionError(f"库里没有 {memory_id}")


def _make_dormant(store, text):
    """存一条记忆、老化 40 天、月扫标沉底，返回 id。"""
    memory_id = _add_inferred(store, text)
    _age(store, memory_id, 40)
    store.sweep_monthly()
    assert _meta_of(store, memory_id)["decay_state"] == DECAY_STATE_DORMANT
    return memory_id


# ---------- 补丁 1：层标记 + 0.6 闸 ----------


def test_layer_marks_active_and_archive(tmp_path):
    """每条结果都有 layer：常用层 active，沉底的 archive。"""
    store = make_fake_store(tmp_path)
    dormant_id = _make_dormant(store, "喝咖啡，每天一杯")
    results = store.search("每天一杯")
    assert [item["id"] for item in results] == [dormant_id]
    assert results[0]["layer"] == "archive"
    assert results[0]["metadata"]["decay_state"] == DECAY_STATE_DORMANT


def test_archive_excluded_when_active_is_strong(tmp_path):
    """常用层最好的一条语义分 1.0（≥0.6）：档案层不进结果。

    常用层“我喜欢喝咖啡”5 个窗口全中，分 5/5=1.0；
    档案层“喝咖啡，每天一杯”中 2 个，分 2/5=0.4，过了 0.4 的
    检索门槛、但被两段式检索的闸拦在外面。"""
    store = make_fake_store(tmp_path)
    dormant_id = _make_dormant(store, "喝咖啡，每天一杯")
    active_id = _add_inferred(store, "我喜欢喝咖啡")
    assert _meta_of(store, active_id)["decay_state"] == DECAY_STATE_VERBATIM
    results = store.search("我喜欢喝咖啡")
    assert [item["id"] for item in results] == [active_id]
    assert dormant_id not in [item["id"] for item in results]
    assert results[0]["layer"] == "active"


def test_archive_included_when_active_is_weak(tmp_path):
    """常用层没有像样的匹配（最好 0 分）：档案层进来，标清是档案。"""
    store = make_fake_store(tmp_path)
    dormant_id = _make_dormant(store, "喝咖啡，每天一杯")
    _add_inferred(store, "我喜欢喝咖啡")
    results = store.search("每天一杯")
    assert [item["id"] for item in results] == [dormant_id]
    assert results[0]["layer"] == "archive"


def test_archive_revived_on_use_still_works(tmp_path):
    """档案层被用上（reinforce）照常回常用层——闸只管进不进结果，
    不管唤醒（见 test_dormant_memory_searchable_and_woken_on_use）。"""
    store = make_fake_store(tmp_path)
    dormant_id = _make_dormant(store, "喝咖啡，每天一杯")
    results = store.search("每天一杯", reinforce=True)
    assert [item["id"] for item in results] == [dormant_id]
    assert _meta_of(store, dormant_id)["decay_state"] == DECAY_STATE_VERBATIM


# ---------- 补丁 5：BM25 + 原样匹配 ----------


def test_exact_fund_code_match_first(tmp_path):
    """问基金代码：含 019441 的那条排最前面，标 exact_match。"""
    store = make_fake_store(tmp_path)
    code_id = _add_inferred(store, "万家纳指 019441 每天定投 5 块")
    _add_inferred(store, "我喜欢喝咖啡")
    _add_inferred(store, "摩根纳指 019172 每天定投 10 块")
    results = store.search("019441 是多少")
    assert results, "BM25 没捞回含代码的记忆"
    assert results[0]["id"] == code_id
    assert results[0]["source"] == "keyword"
    assert results[0]["exact_match"] is True
    assert code_id not in [item["id"] for item in results[1:]]


def test_bm25_supplement_recovers_keyword_only(tmp_path):
    """语义分不够（2/9=0.22 < 0.4）时，BM25 关键词补充捞回来。"""
    store = make_fake_store(tmp_path)
    target_id = _add_inferred(store, "我上次在圆通寄了快递")
    results = store.search("圆通快递怎么查")
    assert [item["id"] for item in results] == [target_id]
    assert results[0]["source"] == "keyword"


def test_keyword_group_keeps_bm25_order(tmp_path):
    """关键词组按 BM25 排名：命中词多的在前（不是按入库顺序）。

    两条都没进语义组（语义分都 < 0.4）；BM25 里含“咖啡”的那条
    命中词更多，排在前面。"""
    store = make_fake_store(tmp_path)
    weak_id = _add_inferred(store, "我喜欢喝牛奶，每天早上都要喝一杯")
    strong_id = _add_inferred(store, "我喜欢喝咖啡，每天早上都要喝一杯")
    results = store.search("每天早上喝咖啡")
    keyword_ids = [
        item["id"] for item in results if item["source"] == "keyword"
    ]
    assert keyword_ids == [strong_id, weak_id]
