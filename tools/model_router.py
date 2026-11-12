# tools/model_router.py（模型路由：难问题走 Pro，日常问题走 Flash）
# 作用：BOOS 的问题来了，先判断难易，再决定用哪个 DeepSeek 模型。
#   难问题（多步推理、跨东西比较、写代码/调试、意思复杂）→ deepseek-v4-pro；
#   日常问题（打招呼、查一个数、记一件事）→ deepseek-flash。
# 原理：现在是启发式分类器（heuristic）——关键词 + 结构规则，确定、
#   可测、不花钱；以后想让模型自己判断难易，再接 model-based 分类器
#   （见下面的扩展点）。分类只看 BOOS 原话，不看拼了记忆提示块的
#   model_prompt——记忆块里提到的词不能左右模型选择。
# 关系：router.py 的 route() 调这里；main.py 的 run_question() 把
#   BOOS 原话传给 route()，再按返回的模型名建 Provider。
# 注意：分类是原话的纯函数——同一句话每次都分到同一个模型，
#   任务恢复（resume）时按原话重算也不会换模型，不会触发任务的模型锁定报错。

import logging
import re

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

# CLASSIFIER（分类器种类）：现在只有 "heuristic"（关键词/结构规则）。
# 以后接 model-based 分类器时，把这里改成 "model"，并实现
# _classify_with_model()——入口 classify() 不用改，调用方不用改。
CLASSIFIER = "heuristic"

# 难问题的信号（命中任意一条就是 hard，先命中的先报原因）：
#   写代码/调试：代码、编程、函数、调试、报错、bug、脚本、算法、递归、正则。
_CODE_KEYWORDS = (
    "代码", "编程", "函数", "调试", "报错", "bug", "脚本",
    "算法", "递归", "正则",
)
#   跨东西比较：比较、对比、区别、优劣、哪个好、哪个更好、谁更、哪一个。
_COMPARE_KEYWORDS = (
    "比较", "对比", "区别", "优劣", "哪个好", "哪个更好", "谁更", "哪一个",
)
#   推理/分析：为什么、原因、分析、怎么回事、背后。
_REASON_KEYWORDS = ("为什么", "原因", "分析", "怎么回事", "背后")
#   多步/计划：计划、方案、步骤、流程、先……再……。
_PLAN_KEYWORDS = ("计划", "方案", "步骤", "流程")
_PLAN_SEQUENCE_RE = re.compile(r"先.+再")
#   多个行情对象（纳指、标普、道指……）：两个以上就是跨东西比较。
_MARKET_ENTITIES = (
    "纳指", "纳斯达克", "标普", "道指", "道琼斯",
    "沪深", "上证", "恒生", "黄金", "原油",
)
#   日期写法：11月10日、11月10号、11/10、11.10，以及今天/昨天这类相对日期。
#   注意：BOOS 写日期常带空格（“11 月 10 日”），模式里要容空格。
_DATE_PATTERNS = (
    re.compile(r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]"),
    re.compile(r"\d{1,2}\s*/\s*\d{1,2}"),
    re.compile(r"\d{1,2}\s*\.\s*\d{1,2}"),
)
_RELATIVE_DATES = (
    "今天", "昨天", "明天", "前天", "后天", "上周", "本周", "下周",
    "上个月", "这个月", "下个月",
)
#   长问题 / 连问两个问题：字数多、问号多，通常意思也复杂。
_LONG_QUESTION_CHARS = 100


def _distinct_dates(text):
    """问题里提到的不同日期（去重后）：显式日期 + 相对日期都算。"""
    found = set()
    for pattern in _DATE_PATTERNS:
        found.update(pattern.findall(text))
    for word in _RELATIVE_DATES:
        if word in text:
            found.add(word)
    return found


def _distinct_market_entities(text):
    """问题里提到的不同行情对象（去重后）。"""
    return {entity for entity in _MARKET_ENTITIES if entity in text}


def classify(text, classifier=None):
    """判断一个问题是难（hard）还是日常（easy）。

    返回 (difficulty, reason)：difficulty 是 "hard" / "easy"，
    reason 是一句中文原因（写进日志，BOOS 能看到为什么走这个模型）。
    classifier 不传时用 CLASSIFIER（现在是 "heuristic"）；传 "model"
    时走 _classify_with_model()（还没实现，会明确报错，不静默回落）。
    同一个问题每次都返回同一个结果（纯函数，不调模型、不读状态）。
    """
    kind = classifier or CLASSIFIER
    if kind == "model":
        return _classify_with_model(text)
    if kind != "heuristic":
        raise ValueError(f"未知的分类器：{kind!r}，只认 'heuristic' / 'model'。")
    return _classify_heuristic(text)


def _classify_heuristic(text):
    """启发式分类：关键词 + 结构规则，命中任意一条难信号就是 hard。"""
    text = (text or "").strip()
    if not text:
        return "easy", "空问题按日常算"
    for keyword in _CODE_KEYWORDS:
        if keyword in text:
            return "hard", f"提到“{keyword}”（写代码/调试）"
    for keyword in _COMPARE_KEYWORDS:
        if keyword in text:
            return "hard", f"提到“{keyword}”（跨东西比较）"
    for keyword in _REASON_KEYWORDS:
        if keyword in text:
            return "hard", f"提到“{keyword}”（要推理/分析）"
    for keyword in _PLAN_KEYWORDS:
        if keyword in text:
            return "hard", f"提到“{keyword}”（多步/计划）"
    if _PLAN_SEQUENCE_RE.search(text):
        return "hard", "先……再……（多步）"
    entities = _distinct_market_entities(text)
    if len(entities) >= 2:
        return "hard", f"同时提到{len(entities)}个行情对象（{'、'.join(sorted(entities))}）"
    dates = _distinct_dates(text)
    if len(dates) >= 2:
        return "hard", f"同时提到{len(dates)}个日期（{'、'.join(sorted(dates))}）"
    if text.count("?") + text.count("？") >= 2:
        return "hard", "连问两个问题"
    if len(text) > _LONG_QUESTION_CHARS:
        return "hard", f"问题太长（{len(text)}字）"
    return "easy", "日常问题"


def _classify_with_model(text):
    """模型分类（扩展点，还没实现）。

    以后想让模型自己判断难易时在这里实现：调一个便宜模型问一句
    “这个问题难还是日常”，按回答分。没实现就明确报错，不静默
    回落到启发式——回落会让调用方以为用的是模型判断，其实不是。
    """
    raise NotImplementedError(
        "model-based 分类器还没实现：现在只用启发式分类器。"
        "实现 _classify_with_model() 后再把 CLASSIFIER 改成 'model'。"
    )


def select_model(text):
    """给一个问题选模型。返回 (model, difficulty, reason)。

    hard → deepseek-v4-pro（MODEL_CONFIG 里的 model_pro），
    easy → deepseek-flash（MODEL_CONFIG 里的 model）。
    每次选完记一条日志（难度、模型、原因、问题前 60 字），
    运行输出/日志里能看到每个问题走了哪个模型。
    """
    difficulty, reason = classify(text)
    cfg = MODEL_CONFIG["deepseek"]
    model = cfg["model_pro"] if difficulty == "hard" else cfg["model"]
    logger.info(
        "模型路由：难度=%s，模型=%s，原因=%s，问题=%s",
        difficulty, model, reason, (text or "")[:60],
    )
    return model, difficulty, reason
