# memory/graph_entities.py（图谱推理：实体抽取，免费先行、LLM 兜底）
# 图谱推理第一期（2026-11-18）：从原话里抽实体名，建图用。
# 规矩（设计方案第五节第 3 条 + Graphiti 的抽取规矩）：
#   - 先走免费的（槽位 + 关键词 + 联系人名单），抽不到再调一次 flash；
#   - 不许抽代词、抽象概念、泛词；只抽四类：人 / 基金 / 东西 / 行为；
#   - 实体名要归一化（同一个东西叫法不同算一个节点），归一化表放这里，
#     BOOS 说了新的叫法就往里加；
#   - 抽不到就不抽，不硬凑；抽取失败不阻塞存入。
# 注意：这里只做“抽”这个纯函数，不调模型、不读库、不写库；
# LLM 兜底那一步在 Mem0Store 里（_judge_json），见 graph.py 的用法。

import re

from memory.keywords import _extract_keywords
from memory.money_slots import _extract_money_metadata

ENTITY_KIND_PERSON = "person"
ENTITY_KIND_FUND = "fund"
ENTITY_KIND_THING = "thing"
ENTITY_KIND_HABIT = "habit"
ENTITY_KINDS = (
    ENTITY_KIND_PERSON,
    ENTITY_KIND_FUND,
    ENTITY_KIND_THING,
    ENTITY_KIND_HABIT,
)

# 联系人名单（BOOS 的人，都是已知的名字，配原文子串认）：
# 家人 + 抖音常联系的人 + 正在帮的人。BOOS 说了新名字就往里加。
CONTACT_NAMES = (
    "妈妈",
    "奶奶",
    "姐姐",
    "斌",
    "阿彬",
    "涛儿",
    "彭幺儿",
    "徐不二",
    "且听风吟",
    "裴捷丽",
    "罗雪榛",
    "大哥",
    "雨哥",
    "忱",
    "12-",
    "paranoïa",
    "与海",
    "静静",
    "贵妃",
)

# 基金名（BOOS 持仓 / 常提的，配原文子串认）：
# 定投的几只 + 常问的指数。money_slots 的槽位 target 也会抠出
# 基金名，两边互补——槽位抠出来的是“买/定投的对象”，这里认的是
# “出现在原话里的基金名”，说法不同时两边各认各的，归一化时再并。
FUND_NAMES = (
    "万家纳指",
    "摩根纳指",
    "摩根标普",
    "南方纳指",
    "华安纳指",
    "国富全球科技互联",
    "沪深300",
    "沪深 300",
    "标普500",
    "标普 500",
    "纳斯达克",
    "永赢中短债",
)

# 东西 / 喜好（BOOS 提过的具体食物、饮品，配原文子串认）：
# 只认具体的，不认泛词（“东西”“好吃的”这种不进）。
THING_WORDS = (
    "冒菜",
    "火锅",
    "冰美式",
    "美式",
    "咖啡",
    "拿铁",
    "鸡杂面",
    "拉面",
    "茶",
)

# 行为 / 习惯（BOOS 的具体行为，配原文子串认）。
HABIT_WORDS = (
    "定投",
    "起床",
    "跑步",
    "睡觉",
    "运动",
)

# 基金代码：6 位数字（019441 这种）。注意：代码和基金名是两个节点，
# 不并成一个——同一条记忆里同时出现才连起来（设计方案的例子：
# 定投 5 块 → 万家纳指 → 019441 → 基金，一跳一跳走过去）。
_FUND_CODE_RE = re.compile(r"\d{6}")

# 实体名归一化表：同一个东西叫法不同算一个节点。
# BOOS 说了新的叫法就往里加（key 是原话里的叫法，value 是节点名）。
ENTITY_ALIASES = {
    "万家纳斯达克": "万家纳指",
    "万家纳斯达克指数": "万家纳指",
    "沪深 300": "沪深300",
    "标普 500": "标普500",
    # 家人自称：BOOS 说“我妈”就是妈妈、“我奶”就是奶奶（2026-11-18
    # 真调发现：模型和免费抽取都认不出“我妈”，只认出了“吃饭”）。
    "我妈": "妈妈",
    "我奶": "奶奶",
}


def normalize_entity_name(name):
    """实体名归一化：去首尾空白、走别名表。

    空的、不是字符串的返回 None（不认）。归一化只做“叫法不同算
    一个”，不做语义合并——“万家纳指”和“摩根纳指”是两只基金，
    不能并。
    """
    if not isinstance(name, str):
        return None
    cleaned = name.strip()
    if not cleaned:
        return None
    return ENTITY_ALIASES.get(cleaned, cleaned)


def extract_entities_free(text):
    """免费的实体抽取（不调模型）：槽位 + 基金名/代码 + 联系人 + 东西/习惯。

    返回 [(实体名, 类别)]，按“人、基金、东西、习惯”的顺序排，
    同一个实体只出现一次。抽不到返回 []（不硬凑）。
    钱的槽位（money target）是基金名的重要来源：定投的对象
    （“买万家纳指”里的万家纳指）就是靠它认出来的。
    """
    if not isinstance(text, str) or not text.strip():
        return []
    found = []
    seen = set()

    def _add(name, kind):
        normalized = normalize_entity_name(name)
        if normalized is None or normalized in seen:
            return
        seen.add(normalized)
        found.append((normalized, kind))

    # 人：联系人名单子串认；别名也认（“我妈”就是妈妈——别名表
    # 里值是联系人的那些，键出现在原话里就算认到了）。
    for name in CONTACT_NAMES:
        if name in text:
            _add(name, ENTITY_KIND_PERSON)
    for alias, canonical in ENTITY_ALIASES.items():
        if canonical in CONTACT_NAMES and alias in text:
            _add(canonical, ENTITY_KIND_PERSON)
    # 基金：槽位 target + 基金名 + 6 位代码。
    money = _extract_money_metadata(text)
    if money is not None and money.get("target"):
        _add(money["target"], ENTITY_KIND_FUND)
    for name in FUND_NAMES:
        if name in text:
            _add(name, ENTITY_KIND_FUND)
    for code in _FUND_CODE_RE.findall(text):
        _add(code, ENTITY_KIND_FUND)
    # 东西：具体食物、饮品。
    for word in THING_WORDS:
        if word in text:
            _add(word, ENTITY_KIND_THING)
    # 行为：具体习惯。
    for word in HABIT_WORDS:
        if word in text:
            _add(word, ENTITY_KIND_HABIT)
    return found


# LLM 兜底的提示词（Graphiti 的抽取规矩改过来，只抽四类）：
# 只在免费抽取一个实体都没抽到时才调一次 flash（见 Mem0Store）。
ENTITY_EXTRACT_SYSTEM_PROMPT = (
    "你是一个实体抽取器。只从下面这句话里抽实体名。\n"
    "规矩：\n"
    "1. 不许抽代词（我、你、他、这个、那个、它）；\n"
    "2. 不许抽抽象概念（事情、东西、问题、情况）；\n"
    "3. 不许抽泛词（好吃的、好玩的、那个）；\n"
    "4. 只抽四类：人（具体人名）、基金/钱（基金名、基金代码、金额对象）、"
    "东西/喜好（具体食物、饮品、物品名）、行为/习惯（定投、起床、跑步这类具体行为）；\n"
    "5. 一个实体只抽一次；抽不到就返回空列表，不硬凑。\n"
    "只回 JSON，不要多余的话：{\"entities\": [{\"name\": \"...\", \"kind\": "
    "\"person|fund|thing|habit\"}]}。kind 只能是这四个值之一。"
)


def extract_entities_from_llm_result(data):
    """把 LLM 兜底返回的 JSON 规整成 [(实体名, 类别)]。

    形状不对（不是 dict、entities 不是列表、kind 不在四类里、
    名字空的）就跳过那一项，不猜、不将就；全都不对就返回 []。
    归一化、去重和免费抽取同一套。
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("entities")
    if not isinstance(raw, list):
        return []
    found = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind not in ENTITY_KINDS:
            continue
        normalized = normalize_entity_name(item.get("name"))
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        found.append((normalized, kind))
    return found


def extract_entities(text, llm_judge=None):
    """抽一条原话的实体：免费先行，抽不到再调 LLM 兜底。

    llm_judge：调用方传进来的 _judge_json（system_prompt, user_prompt）
    那个签名；不传（None）时只走免费的。LLM 兜底失败（返回 None、
    形状不对、抛异常）就当没抽到，返回免费抽到的（可能是 []），
    绝不抛异常——抽取失败不阻塞存入。
    返回 [(实体名, 类别)]。
    """
    free = extract_entities_free(text)
    if free:
        return free
    if llm_judge is None:
        return []
    try:
        data = llm_judge(ENTITY_EXTRACT_SYSTEM_PROMPT, text or "")
    except Exception:
        return []
    if data is None:
        return []
    return extract_entities_from_llm_result(data)


def extract_query_entities(query, llm_judge=None):
    """从查询里找种子实体（检索时用，顺序和抽取一样：先免费后 LLM）。

    返回实体名列表（归一化过的、去重过的）。种子实体在图里找不着
    时，调用方跳过图这路、直接走老路的结果（HippoRAG 的兜底）。
    """
    return [name for name, _kind in extract_entities(query, llm_judge=llm_judge)]
