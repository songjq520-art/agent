# memory/keywords.py（关键词与向量相似度：中文关键词、余弦相似度）
# 从 memory/mem0_store.py 拆出来，行为不变。

import math
import re

# 以后留的位置（第二步衰减 / 删除时再用，第一步不实现，只把名字占住，
# 免得以后各写各的、又要迁移一次）：
#   - "importance"（重要性，high/normal/low）：和用得多不多分开算——
#     低频不等于低价值（原则②）；衰减速度看 importance，不看调用次数。
#     钉住的核心记忆 importance=high、衰减是零。
#   - "superseded"（被取代标记，bool）：同级新胜出时旧记忆标上，原话保留；
#     第一步里只有 _resolve_conflict 会判，真正标上去要等模型能判断
#     “同一条事实”时再接（见 _resolve_conflict 注释）。
#   - 真删前检查（原则③）：删之前查三样——是不是核心记忆、是不是
#     BOOS 明确让记的、有没有别的记忆连着它；有一条不满足就继续
#     归档、不删。delete_memory() 现在不改。


# ---------- 关键词补充信号（从 v1 main.py 搬过来，只做补充用） ----------
# v1 的关键词检索很小、已测过，留作第二路信号：语义检索漏掉的
# 精确关键词（比如“圆通”“Kimi”这种专有名词），靠它兜底。
# 注意：它只是补充，排在语义结果后面；主召回仍是 mem0 的语义检索。

_STOPWORDS = frozenset(
    [
        "怎么",
        "什么",
        "多少",
        "这个",
        "那个",
        "一下",
        "可以",
        "一个",
        "为什么",
        "怎么办",
        "是不是",
        "有没有",
        "请问",
        "谢谢",
        "the",
        "is",
        "are",
        "do",
        "does",
        "can",
        "could",
        "please",
        "tell",
    ]
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _extract_keywords(prompt, min_len=2):
    """把问题拆成关键词：按空白和标点分词；含中文的长词再拆成两字窗口。

    中文词常是两个字（“预算”“对话”），只按标点分会把“我的预算是多少”
    当成一个词，搜不到记忆；所以含中文的长词要拆成两字窗口才能命中。
    英文长词整词搜，不拆两字母（两字母片段噪音太大）。
    停用词（_STOPWORDS）直接去掉，不参与检索，减少误撞。
    返回去重后的关键词列表（保持出现顺序）。
    """
    parts = re.split(r"[^\w]+", prompt or "")
    keywords = []
    for part in parts:
        part = part.strip()
        if len(part) < min_len:
            continue
        keywords.append(part)
        if len(part) > min_len and _CJK_RE.search(part):
            for i in range(len(part) - 1):
                keywords.append(part[i : i + 2])
    seen = set()
    out = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            if word.lower() in _STOPWORDS:
                continue
            out.append(word)
    return out

def _keyword_weight(text, keywords):
    """给一段记忆文本打关键词权重（供补充信号用）。

    权重 = 匹配到的关键词长度之和：匹配的关键词越长、越多，权重越高。
    返回 (权重, 匹配到的关键词列表)。
    """
    lower = (text or "").lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    return sum(len(kw) for kw in matched), matched

def _cosine_similarity(vec_a, vec_b):
    """两个向量的余弦相似度（纠错安全网用，纯函数）。

    返回 -1 到 1 之间：1 是同一方向、0 是无关、-1 是相反。
    任一向量是空的或零向量时返回 0.0（没法比，就当中立）。
    自己做归一化，不依赖 embedding 模型有没有归一化过。
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in vec_a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

