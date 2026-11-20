# tests/test_bm25.py（BM25 关键词检索 + RRF 融合 + 原样匹配）
# 方案补丁 5 的测试：全部离线、造假数据，不碰真实 mem0、不联网、不产生费用。

import pytest

from memory.bm25 import bm25_search, extract_exact_matches, reciprocal_rank_fusion

DOCS = [
    {"id": "fund", "text": "我每天定投 5 块买万家纳指基金 019441"},
    {"id": "coffee", "text": "我喜欢喝咖啡"},
    {"id": "fund-news", "text": "万家纳指基金今天涨了"},
]


# ---------- bm25_search：中文词命中与排序 ----------

def test_bm25_hits_chinese_word():
    results = bm25_search("咖啡", DOCS)
    assert [r["id"] for r in results] == ["coffee"]
    assert results[0]["text"] == "我喜欢喝咖啡"
    assert results[0]["score"] > 0


def test_bm25_more_query_terms_ranks_first():
    docs = [
        {"id": "both", "text": "万家基金"},
        {"id": "one-a", "text": "万家好"},
        {"id": "one-b", "text": "基金好"},
        {"id": "none", "text": "我喜欢喝咖啡"},
    ]
    results = bm25_search("万家 基金", docs)
    # 两词都命中的排第一；各中一词的两条分数相同，按 docs 原序。
    assert [r["id"] for r in results] == ["both", "one-a", "one-b"]


def test_bm25_shorter_doc_ranks_first_on_tie():
    docs = [
        {"id": "short", "text": "苹果"},
        {"id": "long", "text": "我喜欢吃苹果香蕉橙子葡萄"},
    ]
    results = bm25_search("苹果", docs)
    assert [r["id"] for r in results] == ["short", "long"]
    assert results[0]["score"] > results[1]["score"]


def test_bm25_identical_texts_keep_original_order():
    docs = [
        {"id": "first", "text": "我喜欢喝咖啡"},
        {"id": "second", "text": "我喜欢喝咖啡"},
    ]
    results = bm25_search("咖啡", docs)
    assert [r["id"] for r in results] == ["first", "second"]
    assert results[0]["score"] == results[1]["score"]


def test_bm25_top_k_truncates():
    docs = [
        {"id": "both", "text": "万家基金"},
        {"id": "one-a", "text": "万家好"},
        {"id": "one-b", "text": "基金好"},
    ]
    assert [r["id"] for r in bm25_search("万家 基金", docs, top_k=2)] == [
        "both",
        "one-a",
    ]
    assert len(bm25_search("万家 基金", docs, top_k=10)) == 3
    assert bm25_search("万家 基金", docs, top_k=0) == []


def test_bm25_no_hit_returns_empty():
    assert bm25_search("火星", DOCS) == []


def test_bm25_empty_query_returns_empty():
    assert bm25_search("", DOCS) == []
    assert bm25_search("   ", DOCS) == []
    # 只有停用词 / 只有一个字：分词结果是空的，也返回空。
    assert bm25_search("怎么", DOCS) == []
    assert bm25_search("我", DOCS) == []


def test_bm25_empty_docs_returns_empty():
    assert bm25_search("咖啡", []) == []


# ---------- reciprocal_rank_fusion ----------

def test_rrf_overlapping_id_ranks_first():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    # b 在两路都出现，融合分最高；a 和 c 各只一路、分数相同，按先出现排。
    assert [doc_id for doc_id, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 62)
    assert fused[1][1] == pytest.approx(1 / 61)


def test_rrf_single_list_keeps_order():
    fused = reciprocal_rank_fusion([["x", "y"]])
    assert fused == [("x", pytest.approx(1 / 61)), ("y", pytest.approx(1 / 62))]


def test_rrf_k_parameter_changes_scores():
    fused = reciprocal_rank_fusion([["x", "y"]], k=0)
    assert fused == [("x", pytest.approx(1.0)), ("y", pytest.approx(0.5))]


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_duplicate_inside_one_list_counts_once():
    fused = reciprocal_rank_fusion([["a", "a", "b"]])
    assert [doc_id for doc_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(1 / 61)


# ---------- extract_exact_matches ----------

def test_exact_match_fund_code():
    assert extract_exact_matches("019441 限购多少", DOCS) == ["fund"]


def test_exact_match_decimal_number():
    docs = [
        {"id": "a", "text": "仲裁总结 1.35 万元"},
        {"id": "b", "text": "仲裁总结 2 万元"},
    ]
    assert extract_exact_matches("收益 1.35 万元", docs) == ["a"]


def test_exact_match_percent_number():
    docs = [
        {"id": "a", "text": "今天涨了 5.5%"},
        {"id": "b", "text": "今天涨了 5.6%"},
    ]
    assert extract_exact_matches("涨了 5.5%", docs) == ["a"]


def test_exact_match_query_phrase_verbatim():
    assert extract_exact_matches("每天定投", DOCS) == ["fund"]
    assert extract_exact_matches("定投", DOCS) == ["fund"]


def test_exact_match_no_match_returns_empty():
    assert (
        extract_exact_matches("019441 限购多少", [{"id": "a", "text": "我喜欢喝咖啡"}])
        == []
    )
    assert extract_exact_matches("", DOCS) == []
    assert extract_exact_matches("我", DOCS) == []


def test_exact_match_number_not_inside_longer_number():
    docs = [
        {"id": "right", "text": "每天定投 5 块买万家纳指"},
        {"id": "wrong", "text": "每天定投 15 块买万家纳指"},
    ]
    # "5" 不能撞上 "15" 里的 5；整句 "定投 5 块" 在 "定投 15 块" 里也不算。
    assert extract_exact_matches("定投 5 块", docs) == ["right"]


def test_exact_match_keeps_docs_order_and_dedups():
    docs = [
        {"id": "second", "text": "019441 限购 5 块"},
        {"id": "first", "text": "019441 今天涨了"},
        {"id": "plain", "text": "我喜欢喝咖啡"},
    ]
    # second 同时命中基金代码和数字，也只出现一次；按 docs 原序返回。
    assert extract_exact_matches("019441 限购 5 块", docs) == ["second", "first"]


def test_exact_match_missing_text_does_not_crash():
    assert extract_exact_matches("019441", [{"id": "x"}, {"id": "y", "text": None}]) == []


# ---------- 确定性 ----------

def test_determinism_same_input_twice():
    query = "万家 基金 019441"
    first = bm25_search(query, DOCS)
    second = bm25_search(query, DOCS)
    assert first == second

    lists = [["a", "b"], ["c", "b", "a"]]
    assert reciprocal_rank_fusion(lists) == reciprocal_rank_fusion(lists)

    assert extract_exact_matches("019441 限购", DOCS) == extract_exact_matches(
        "019441 限购", DOCS
    )
