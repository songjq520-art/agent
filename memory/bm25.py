# memory/bm25.py（BM25 关键词检索 + 排名融合 + 原样匹配）
# 方案补丁 5：记忆检索的第二路信号。
# 语义向量擅长“换说法也能召回”，BM25 擅长“原样词必须命中”
# （基金代码、数字、人名）；两路结果用 RRF 合并，原样匹配的再往前排。
# 纯函数，只用标准库 + memory/keywords.py 的分词，不碰网络、不碰真实 API。

import math
import re
from collections import Counter

try:
    from memory.keywords import _extract_keywords
except ImportError:  # 直接把本文件当脚本跑时，退回同目录导入
    from keywords import _extract_keywords

# BM25 参数（经典默认值）：k1 管词频饱和，b 管长度归一化。
_BM25_K1 = 1.5
_BM25_B = 0.75

# 基金代码：前后都不能再跟数字，避免把 7 位数里的一截当成代码。
_FUND_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
# 数字：整数或小数，后面可直接跟单位（块/元/万/%/美元/人民币……）。
# 单位只认“紧贴着数字”的，中间隔了空格就不算后缀（那是另一个词）。
_NUMBER_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?(?:美元|人民币|[块元万亿%％])?(?!\d)")
# 数字的裸数字部分（不带单位），供 extract_exact_matches 两份都记。
_BARE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def bm25_search(query, docs, top_k=5):
    """用 BM25 给 docs 打分，返回最相关的前 top_k 条。

    query 和 doc 都用 keywords._extract_keywords 分词（中文长词会拆成
    两字窗口，和现有关键词补充信号同一套分词，不另起一套）。
    分数 = sum over query 词：idf * (tf*(k1+1)) / (tf + k1*(1-b+b*dl/avgdl))，
    k1=1.5，b=0.75；idf 取 log(1 + (N-df+0.5)/(df+0.5))，恒为正。
    没命中任何 query 词的 doc 不返回；分数相同按 docs 原序（稳定排序）。
    返回 [{"id", "text", "score"}]，按分降序；空 query / 空 docs /
    没命中都返回 []。
    """
    if not query or not docs:
        return []
    query_terms = _extract_keywords(query)
    if not query_terms:
        return []
    docs = list(docs)
    doc_counts = [
        Counter(_extract_keywords((doc or {}).get("text") or "")) for doc in docs
    ]
    doc_count = len(docs)
    avg_len = sum(sum(counts.values()) for counts in doc_counts) / doc_count
    if avg_len == 0:
        return []
    scored = []
    for index, counts in enumerate(doc_counts):
        if not counts:
            continue
        doc_len = sum(counts.values())
        length_norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg_len)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = sum(1 for other in doc_counts if term in other)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            score += idf * (tf * (_BM25_K1 + 1)) / (tf + length_norm)
        if score > 0:
            scored.append((index, score))
    # 稳定排序：分数降序，分数相同保持 docs 原序。
    scored.sort(key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        if top_k <= 0:
            return []
        scored = scored[:top_k]
    return [
        {"id": docs[index].get("id"), "text": docs[index].get("text"), "score": score}
        for index, score in scored
    ]


def reciprocal_rank_fusion(ranked_id_lists, k=60):
    """把多路排名（每路按好坏排好的 id 列表）融合成一路。

    融合分 = sum(1 / (k + rank))，rank 从 1 开始；k 默认 60（RRF 经典值，
    k 越大，名次差距越不值钱，各路越平等）。
    同一个 id 只出现一次（在同一路里重复出现也只算第一次的名次）。
    返回 [(id, fused_score)]，按分降序；分数相同按 id 第一次出现的顺序
    （确定性，不依赖 dict / set 的遍历顺序）。
    """
    fused = {}
    first_seen = {}
    order = 0
    for ranked_ids in ranked_id_lists or []:
        seen_in_list = set()
        for rank, doc_id in enumerate(ranked_ids or [], start=1):
            if doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)
            if doc_id not in fused:
                fused[doc_id] = 0.0
                first_seen[doc_id] = order
                order += 1
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(
        fused.items(), key=lambda item: (-item[1], first_seen[item[0]])
    )


def _verbatim_in(text, token):
    """token 是否原样出现在 text 里（纯数字 token 不许嵌在更长的数字里）。

    "5" 不能撞上 "15" 里的 5，"019441" 不能撞上 "1019441" 里的一截；
    中文短语直接按子串算，"每天定投" 在 "我每天定投 5 块" 里就是命中。
    """
    if not token or not text:
        return False
    pattern = re.escape(token)
    if token[0].isdigit():
        pattern = r"(?<!\d)" + pattern
    if token[-1].isdigit():
        pattern += r"(?!\d)"
    return re.search(pattern, text) is not None


def extract_exact_matches(query, docs):
    """找出 query 里原样出现在 doc 文本里的 token 对应的 doc id。

    三种 token（只认原样出现，不做模糊/近似匹配）：
      - 基金代码：query 里的 6 位数字（如 019441）；
      - 数字：整数或小数，可带单位（如 5 块、1.35 万、5.5%），
        带单位和裸数字各记一份，只要有一份原样出现就算；
      - query 本身：长度 >= 2 时，整句原样出现在 doc 文本里也算。
    返回按 docs 原序的 doc id 列表（去重，一条 doc 只出现一次）。
    """
    query = query or ""
    docs = docs or []
    if not query.strip():
        return []
    tokens = []
    for match in _FUND_CODE_RE.finditer(query):
        tokens.append(match.group(0))
    for match in _NUMBER_RE.finditer(query):
        full = match.group(0)
        tokens.append(full)
        bare = _BARE_NUMBER_RE.match(full).group(0)
        if bare != full:
            tokens.append(bare)
    stripped = query.strip()
    if len(stripped) >= 2:
        tokens.append(stripped)
    # 去重，保持出现顺序。
    seen = set()
    uniq_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            uniq_tokens.append(token)
    matched = []
    for doc in docs:
        doc_id = (doc or {}).get("id")
        if doc_id is None:
            continue
        text = (doc or {}).get("text") or ""
        if any(_verbatim_in(text, token) for token in uniq_tokens):
            if doc_id not in matched:
                matched.append(doc_id)
    return matched
