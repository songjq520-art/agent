# tools/date_numbers.py（数字-日期归属：50/54 号的根治）
# 作用：从搜索结果的文本里把“数字”和“日期”配对，程序说了算——
#       模型再也不用从自由文本里猜“这个数是哪天的”。
# 原理：
#   - 日期：英文（November 11, 2026 / Nov 11 / 11/11/2026 / 11/10/26）、
#     中文（2026年11月11日 / 11月11日）、ISO（2026-11-11）都认；
#     没写年份的用当年（default_year）。
#   - 数字：收盘价（25,466.19）、涨跌幅（+1.60%）、涨跌点数（+400.31）。
#     价格要带千分位逗号或小数点，不认裸年份/天数（2026、11 不算数）。
#   - 配对：同一句里，数字归最近的那个日期；句里没日期就不配对
#     （undated，不硬猜）；数字前面 40 字里有 from / previous / 前一
#     这类“参照”词、或后面跟着 the session before 时，这个数是拿来
#     对比的旧数，不配当天的日期。
#   - 注意：published_at 是文章的发布日，不是数的日期，绝不拿它
#     当数的日期——拿它当就又回到 50/54 号的老毛病。
# 关系：tools/search.py 的 SearchResult.to_dict() 调它给每条结果
#       加 dated_numbers；tools/agent_loop.py 调它解析用户问的日期、
#       校验回答里引用的数。
# 误差：0.02（收盘价 25,065.88 和 25,065.87 是同一家来源四舍五入
#       的差别，算同一个数；差 0.02 以上算不同的数）。

import re
from datetime import date as _date
from datetime import datetime, timedelta, timezone

# 同一个数差多少以内算同一个数（来源四舍五入的差别）。
NUMBER_TOLERANCE = 0.02

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_RE = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|"
    r"Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
)


def _make_date(year, month, day):
    """拼一个日期：月份 1-12、日 1-31 才认，非法的（2 月 30 日）返回 None。"""
    try:
        return _date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None


# 日期写法（按组名取年月日；没写年份的用 default_year）。
# 注意：英文“Nov 11”既可能出现在“Nov 11, 2026”里，也可能单独出现——
# 重叠的匹配只留最长的那条（parse_date_mentions 里按跨度去重）。
_DATE_PATTERNS = [
    # 2026-11-11 / 2026/11/11。
    (re.compile(r"(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})"),
     lambda g: (g["y"], g["m"], g["d"])),
    # 11/11/2026 / 11/10/26（两位数年份是 20xx 年）。
    (re.compile(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4}|\d{2})"),
     lambda g: (2000 + int(g["y"]) if len(g["y"]) == 2 else g["y"], g["m"], g["d"])),
    # November 11, 2026 / Nov. 11, 2026。
    (re.compile(_MONTH_RE + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?(?!\d),?\s+(?P<y>\d{4})"),
     lambda g: (g["y"], None, g["d"])),
    # 11 November 2026 / 11 Nov 2026。
    (re.compile(r"(?P<d>\d{1,2})(?:st|nd|rd|th)?(?!\d)\s+" + _MONTH_RE + r",?\s+(?P<y>\d{4})"),
     lambda g: (g["y"], None, g["d"])),
    # 2026年11月11日 / 2026 年 11 月 11 日。
    (re.compile(r"(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日"),
     lambda g: (g["y"], g["m"], g["d"])),
    # 11月11日 / 11 月 11 日（没年份）。
    (re.compile(r"(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日"),
     lambda g: (None, g["m"], g["d"])),
    # November 11 / Nov 11（没年份）。
    (re.compile(_MONTH_RE + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?(?!\d)"),
     lambda g: (None, None, g["d"])),
    # 11 November / 11 Nov（没年份）。
    (re.compile(r"(?P<d>\d{1,2})(?:st|nd|rd|th)?(?!\d)\s+" + _MONTH_RE),
     lambda g: (None, None, g["d"])),
    # 11/11（没年份）。
    (re.compile(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})"),
     lambda g: (None, g["m"], g["d"])),
    # 11.10 / 11.9（BOOS 的说法：“标普的 11.10”“纳指 11.9 那天”）。
    # 守卫：前面不能紧挨数字/字母/逗号/小数点/$——否则是价格“25,065.88”、
    # 版本号“v1.9”“K2.6”的一部分；后面不能紧跟数字/%/小数点——否则是
    # 百分比“1.60%”或更长的数；月 1-12、日 1-31 才认（_make_date 拦
    # 非法的，如“1.60”日 60 不认）。残留风险：裸小数价格“11.9”在没
    # 有别的日期的句子里会被当成 11 月 9 日——指数收盘价是五位数带
    # 逗号，不会撞上；个股价格才可能撞，校验只对行情类问题生效。
    (re.compile(r"(?<![\dA-Za-z,.$])(?P<m>\d{1,2})\.(?P<d>\d{1,2})(?![\d%.])"),
     lambda g: (None, g["m"], g["d"])),
]


def _month_number(name):
    """英文月份名转数字（"Nov." → 11），不认识返回 None。"""
    if not name:
        return None
    return _MONTHS.get(name.rstrip(".").lower())


def _pattern_month(match):
    """从一次英文日期匹配里抠月份名（"November 11, 2026" / "11 November 2026"
    里的月份名可能在开头也可能在中间，逐词找）。"""
    for word in re.split(r"[\s,]+", match.group(0).strip()):
        month = _month_number(word)
        if month is not None:
            return month
    return None


def parse_date_mentions(text, default_year=None):
    """一段文字里出现的所有日期。返回 [{"start", "end", "date", "text"}]，
    按出现顺序排。没写年份的用 default_year（不传时用今年）。
    同一位置重叠的写法只留最长的（"November 11, 2026" 不会再拆出
    一个 "November 11"）。非法的日期（2 月 30 日）直接跳过。"""
    if default_year is None:
        default_year = datetime.now(timezone.utc).year
    found = []
    for pattern, pick in _DATE_PATTERNS:
        for match in pattern.finditer(text or ""):
            groups = pick(match.groupdict())
            year, month, day = groups
            if month is None:
                month = _pattern_month(match)
            if year is None:
                year = default_year
            parsed = _make_date(year, month, day)
            if parsed is None:
                continue
            found.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "date": parsed,
                    "text": match.group(0),
                }
            )
    # 去重：按开始位置排，重叠的只留跨度最长的。
    found.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    kept = []
    for item in found:
        if any(
            item["start"] < other["end"] and other["start"] < item["end"]
            for other in kept
        ):
            continue
        kept.append(item)
    kept.sort(key=lambda item: item["start"])
    return kept


# ---------- 数字 ----------

# 数字的三种样子（按顺序试：百分比 → 带符号的涨跌点数 → 收盘价）。
# 百分比：1.60% / +1.60% / -0.74%。
# 涨跌点数：+400.31 / -187.27（带 + / - 号）。
# 收盘价：25,466.19（千分位逗号）或 25466.19（4 位以上整数 + 小数点）。
# 裸年份（2026）和天数（11）不算数：价格必须带逗号或小数点。
# 开头的 (?<![\d,]) 是关键：数字前面紧挨着数字或逗号时，
# 它是更大数字/日期的一部分（"2026-11-10" 里的 -11、-10 不算数），不认。
_NUMBER_RE = re.compile(
    r"(?<![\d,])"
    r"(?:(?P<percent>[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*%"
    r"|(?P<signed>[+-]\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    r"|(?P<price>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}\.\d{2}))"
)


def parse_number_value(text):
    """一个数字文本转成 (kind, value)：kind 是 price / percent / points。
    认不出来返回 None。"""
    if not text:
        return None
    stripped = text.strip()
    match = _NUMBER_RE.fullmatch(stripped)
    if match is None:
        # 数字后面可能带“点”/“points”：先去掉再试。
        cleaned = re.sub(r"\s*(?:点|points?)\s*$", "", stripped)
        match = _NUMBER_RE.fullmatch(cleaned)
        if match is None:
            return None
    if match.group("percent") is not None:
        return ("percent", float(match.group("percent").replace(",", "")))
    if match.group("signed") is not None:
        return ("points", float(match.group("signed").replace(",", "")))
    return ("price", float(match.group("price").replace(",", "")))


def extract_numbers(text):
    """一段文字里的所有数字。返回 [{"start", "end", "text", "kind", "value"}]。"""
    out = []
    for match in _NUMBER_RE.finditer(text or ""):
        parsed = parse_number_value(match.group(0))
        if parsed is None:
            continue
        kind, value = parsed
        out.append(
            {
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "kind": kind,
                "value": value,
            }
        )
    return out


# ---------- 配对 ----------

# 数字前面出现这些词时，它是拿来对比的旧数（前一收盘、参照位），
# 不是这句主日期的数：from / previous / 前一 / 从 ……
# 英文按单词匹配（from 不能匹配 inform 里的 from），中文按子串匹配。
_REFERENCE_BEFORE_RE = re.compile(
    r"\b(from|vs\.?|versus|previous|prior|compared|against|above|below)\b"
    r"|前一|上个|前收|上一交易日|前一日|相比|较|高于|低于|从",
    re.IGNORECASE,
)
# 数字后面跟着这些短语时，也是旧数（"25,065.87 the session before"）。
_REFERENCE_AFTER_PHRASES = (
    "the session before",
    "the day before",
    "the previous session",
    "the prior session",
    "the previous day",
    "前一交易日",
    "上一交易日",
)


def _is_reference(before_text, after_text):
    """这个数是不是拿来对比的旧数（参照数）？"""
    if _REFERENCE_BEFORE_RE.search(before_text or ""):
        return True
    after_lower = (after_text or "").lower()
    return any(phrase in after_lower for phrase in _REFERENCE_AFTER_PHRASES)


def _split_sentences(text):
    """按句切：中文句号/感叹号/问号、分号、换行、英文句号 + 空格 + 大写。
    分号也要切——"On November 10 it closed at X; on November 11 it closed
    at Y" 是两个分句，各有各的日期，不切开的话后面的数会按字符距离
    错配到前面的日期上。"""
    parts = re.split(r"[。！？；;\n]+|(?<=[.!?])\s+(?=[A-Z\"“])", text or "")
    return [part for part in parts if part.strip()]


def extract_dated_numbers(text, default_year=None):
    """一段文字里的（数字，日期）对照。返回 [{"number", "value", "kind",
    "date", "date_text"}]，date 是 "YYYY-MM-DD" 或 None（句里没日期、
    或这个数是拿来对比的旧数时就是 None，不硬猜）。
    同一句里数字归最近的那个日期；句里没日期就不配对。"""
    if default_year is None:
        default_year = datetime.now(timezone.utc).year
    out = []
    for sentence in _split_sentences(text or ""):
        dates = parse_date_mentions(sentence, default_year)
        numbers = extract_numbers(sentence)
        for number in numbers:
            before = sentence[max(0, number["start"] - 40):number["start"]]
            after = sentence[number["end"]:number["end"] + 30]
            if _is_reference(before, after):
                out.append(
                    {
                        "number": number["text"],
                        "value": number["value"],
                        "kind": number["kind"],
                        "date": None,
                        "date_text": None,
                    }
                )
                continue
            if not dates:
                out.append(
                    {
                        "number": number["text"],
                        "value": number["value"],
                        "kind": number["kind"],
                        "date": None,
                        "date_text": None,
                    }
                )
                continue
            nearest = min(dates, key=lambda d: abs(number["start"] - d["start"]))
            out.append(
                {
                    "number": number["text"],
                    "value": number["value"],
                    "kind": number["kind"],
                    "date": nearest["date"].isoformat(),
                    "date_text": nearest["text"],
                }
            )
    return out


# ---------- 用户问的是哪天 ----------


def resolve_requested_date(question, today=None):
    """用户问的是哪一天（绝对日期）。question 是 BOOS 的原话。
    先找明确写出来的日期（11月11日 / Nov 11 / 2026-11-11），
    没有再换算相对说法：昨天/前天/大前天/今天/明天/后天。
    只认得出一个日期时返回 date 对象；没有、或出现两个以上不同日期
    （比如“从 11 月 10 日到年底”这种范围问题）时返回 None——
    拿不准就不校验，别拿错日期去拦正确答案。
    today 不传时按上海时间算今天。"""
    if today is None:
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    text = question or ""
    # 范围问题（"从 X 到 Y"）没有单一日期，不校验。
    if "从" in text and "到" in text:
        return None
    mentions = parse_date_mentions(text, default_year=today.year)
    unique = {item["date"] for item in mentions}
    if len(unique) == 1:
        return mentions[0]["date"]
    if len(unique) > 1:
        return None
    if "大前天" in text:
        return today - timedelta(days=3)
    if "前天" in text:
        return today - timedelta(days=2)
    if "昨天" in text or "昨日" in text:
        return today - timedelta(days=1)
    if "今天" in text or "今日" in text:
        return today
    if "明天" in text or "明日" in text:
        return today + timedelta(days=1)
    if "后天" in text:
        return today + timedelta(days=2)
    return None


# 问的是不是某天行情/收盘（才需要校验数的日期）：
# 天气问题（"广州今天天气怎么样"）也会解析出"今天"，但它的百分比
# （降水概率 3%）不是收盘数，校验它只会误伤，所以只对行情问题校验。
_MARKET_KEYWORDS = (
    "纳指", "纳斯达克", "标普", "道指", "道琼斯", "恒生", "上证", "深证",
    "指数", "收盘", "涨跌", "股价", "股市", "美股", "黄金", "原油", "期货",
    "nasdaq", "s&p", "dow", "index", "close",
)


def is_market_date_question(question):
    """这是不是在问某天行情/收盘？（校验数的日期只对这类问题开。）"""
    text = (question or "").lower()
    return any(keyword in text for keyword in _MARKET_KEYWORDS)


# ---------- 51 号：先查后说（程序级门禁） ----------

# 行情实体：把中英文说法归到同一个键上——门禁认“（实体，日期）对”时，
# 模型的搜索查询可能是中文（“纳斯达克 11 月 10 日收盘”）也可能是
# 英文（“Nasdaq Index closing on November 10”），归一化后才能对上。
_MARKET_ENTITIES = (
    ("nasdaq", "纳指", ("纳指", "纳斯达克", "nasdaq")),
    ("sp500", "标普", ("标普", "s&p", "standard & poor", "standard and poor")),
    ("dow", "道指", ("道指", "道琼斯", "dow")),
    ("hangseng", "恒生", ("恒生", "hang seng")),
    ("shanghai", "上证", ("上证", "沪指", "sse")),
    ("shenzhen", "深证", ("深证", "szse")),
    ("gold", "黄金", ("黄金", "金价", "gold")),
    ("oil", "原油", ("原油", "油价", "oil", "brent", "wti")),
)


def extract_market_entity(question):
    """问的是哪个行情实体（51 号）：返回 (key, 中文名）。没点名具体实体时
    返回 (None, "美股")——比如“美股昨天是涨还是跌”问的是整个市场。"""
    text = (question or "").lower()
    for key, label, keywords in _MARKET_ENTITIES:
        if any(keyword in text for keyword in keywords):
            return key, label
    return None, "美股"


# 行情实体英文名（51 号重试用）：英文搜索对收盘数的召回比中文好得多
#（实测：Tavily 中文“标普500指数 11月10日 收盘”只搜回盘中快讯和
# 2025 年的表，英文“S&P 500 close November 10 2026”一次就搜到美联社
# 的收盘数）。拼查询不用模型翻——短词组翻译模型有时会原样回中文
#（实测“纳斯达克指数 收盘”翻回来还是“纳斯达克指数收盘”），重试就
# 触发不了；程序拼出来的英文查询是确定的，不花钱、不抖动。
_MARKET_ENTITY_ENGLISH = {
    "nasdaq": "Nasdaq Composite",
    "sp500": "S&P 500",
    "dow": "Dow Jones Industrial Average",
    "hangseng": "Hang Seng Index",
    "shanghai": "Shanghai Composite Index",
    "shenzhen": "Shenzhen Component Index",
    "gold": "Gold",
    "oil": "Crude oil",
}
_MONTH_NAMES_ENGLISH = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def english_market_close_query(question, day):
    """行情问题的英文收盘查询（51 号重试用）：问句里认出实体 + 给出日期
    （date 对象）时，拼出 “<实体英文名> close <Month> <日> <年>”
    （比如 “Nasdaq Composite close November 9 2026”）；认不出实体、
    没给日期时返回 None（不猜）。"""
    if day is None:
        return None
    entity_key, _label = extract_market_entity(question)
    english = _MARKET_ENTITY_ENGLISH.get(entity_key) if entity_key else None
    if not english:
        return None
    month = _MONTH_NAMES_ENGLISH[day.month - 1]
    return f"{english} close {month} {day.day} {day.year}"


# 问的是行情数据（收盘/涨跌/点位）才开门禁，不是算数。
_DATA_KEYWORDS = (
    "收盘", "涨跌", "涨还是跌", "跌还是涨", "涨了还是跌了", "跌了还是涨了",
    "收多少", "收了多少", "多少点", "开盘", "最高", "最低", "涨幅", "跌幅",
    "点位", "收盘价",
)


def is_market_data_question(question):
    """51 号门禁的范围：问的是某天行情数据（收盘/涨跌），不是算数。
    光问“标普的 11.10”这种“实体+日期”也算——省略的问法，问的就是收盘。
    定投算钱（“一共多少钱”）不算，不强制搜；普通聊天也不算。"""
    if not is_market_date_question(question):
        return False
    text = (question or "").lower()
    if any(keyword in text for keyword in _DATA_KEYWORDS):
        return True
    entity_key, _label = extract_market_entity(question)
    if entity_key is not None and resolve_requested_date(question) is not None:
        return True
    return False


# ---------- 判断 (b) 号：动态数字事实（双源印证） ----------

# 动态数字事实 = 一个数配上日期/时间，且这个数每天会变（不是用户说的
# 稳定数）。问的是这类数时，要两个独立来源印证才敢当事实用。
# 类别（判断 b 号）：
#   行情（收盘/价格/指数）：纳指、标普、道指、沪深、上证、恒生、黄金、原油。
#   天气/气温：天气、气温、温度、降雨、湿度。
#   汇率：汇率、美元兑、欧元兑、日元兑。
#   比赛比分：比分、比赛结果。
#   利率：利率、加息、降息。
#   空气质量：空气质量、AQI、PM2.5。
# 用户说的稳定数（“我每天定投 5 块”“我每月预算 5 美元”）不是动态数字
# 事实——那是他的决定/偏好，不用双源印证，也不进这里。
_DYNAMIC_NUMERIC_KEYWORDS = (
    # 行情（复用行情实体 + 数据词）。
    "收盘", "收盘价", "点位", "涨幅", "跌幅", "开盘", "最高", "最低",
    "纳指", "纳斯达克", "标普", "道指", "道琼斯", "沪深", "上证", "恒生",
    "黄金", "金价", "原油", "油价",
    # 天气/气温。
    "气温", "温度", "多少度", "降雨", "湿度",
    # 汇率。
    "汇率", "美元兑", "欧元兑", "日元兑",
    # 比赛比分。
    "比分",
    # 利率。
    "利率", "加息", "降息",
    # 空气质量。
    "空气质量", "AQI", "PM2.5", "PM10",
)


def is_dynamic_numeric_question(question):
    """这是不是在问一个动态数字事实（判断 b 号）？

    动态数字事实 = 一个数配上日期/时间，且这个数每天会变。
    问的是这类数时，要两个独立来源印证才敢当事实用（双源印证）。
    用户说的稳定数（定投金额、预算）不是——那是他的决定/偏好，
    不用印证。定投算钱（“一共多少钱”）也不算，那是算数，不是查数。
    """
    text = (question or "")
    if not text.strip():
        return False
    # 定投算钱、预算：算数，不是查动态数。
    if "一共多少钱" in text or "一共多少" in text:
        return False
    lowered = text.lower()
    if not any(keyword in lowered for keyword in _DYNAMIC_NUMERIC_KEYWORDS):
        return False
    # 行情：走 51 号的口径（实体 + 日期才算问数）。
    if is_market_data_question(question):
        return True
    # 天气/汇率/比分/利率/空气质量：问的就是当天的数（每天会变），
    # 有没有写明日期都算——没写日期就是问今天/现在。
    # 注意：“今天天气怎么样”不算——那是泛问天气，不是要数；
    # 要数得写明气温/温度/多少度/降雨/湿度这些词。
    for keyword in (
        "气温", "温度", "多少度", "降雨", "湿度",
        "汇率", "美元兑", "欧元兑", "日元兑",
        "比分", "利率", "加息", "降息",
        "空气质量", "AQI", "PM2.5", "PM10",
    ):
        if keyword in lowered:
            return True
    return False


def dynamic_numeric_fact_kind(text):
    """动态数字事实的类别（判断 b 号，定 expires_at 用）：
    "market"（行情，24 小时或下一个交易日）、"weather"（天气，12 小时）、
    "other"（其他，24 小时）。不是动态数字事实时返回 None。
    问题和事实原文都能传（“纳指收盘多少”是问题，“纳指收盘 25466.19”
    是事实，类别判断用的都是实体词）。"""
    if not (is_dynamic_numeric_question(text) or is_dynamic_numeric_fact(text)):
        return None
    lowered = (text or "").lower()
    if any(
        keyword in lowered
        for keyword in ("气温", "温度", "多少度", "降雨", "湿度", "空气质量", "AQI", "PM2.5", "PM10")
    ) or re.search(r"\d+\s*(?:度|°c|摄氏度)", lowered):
        return "weather"
    return "market" if is_market_data_question(text) else "other"


def is_dynamic_numeric_fact(text):
    """这句话是不是一个动态数字事实（判断 b 号，记忆入库门禁）？

    动态数字事实 = 一个数配上日期/时间，且这个数每天会变（行情收盘、
    气温、汇率、比分、利率、空气质量）。入库时要双源印证才敢当稳定
    事实用；没印证的只记成未核实、可信度封顶 0.35。
    用户说的稳定数（定投金额、预算、某天花了多少钱）不是——那是他的
    决定/自己的花销，不用双源印证。判断方法：先看有没有配上日期的数
    （没日期的数直接不是），再看有没有动态实体词（行情/天气/汇率/
    比分/利率/空气质量）或气温写法（数字 + 度/°C）。
    """
    if not (text or "").strip():
        return False
    lowered = text.lower()
    # 有没有配上日期：显式日期（11 月 11 日）由 extract_dated_numbers 配；
    # 相对日期（今天/昨天）它不认，这里单独认。
    dated = [
        entry for entry in extract_dated_numbers(text)
        if entry.get("date") is not None
    ]
    has_date = bool(dated) or any(
        word in lowered
        for word in ("今天", "昨天", "前天", "明天", "大前天", "后天", "昨晚", "今晚")
    )
    if not has_date:
        return False
    # 气温写法：数字 + 度/°C（“广州今天 26 度”）——这种数小，
    # _NUMBER_RE（只认带逗号的大数/百分比）抓不到，单独认。
    if re.search(r"\d+(?:\.\d+)?\s*(?:度|°c|摄氏度)", lowered):
        return True
    # 行情/汇率/比分/利率/空气质量：实体词 + 配上日期的数。
    # 没有实体词就不是动态数字事实，只是用户自己的数（定投金额、
    # 预算、某天花了多少钱）。
    if not any(keyword in lowered for keyword in _DYNAMIC_NUMERIC_KEYWORDS):
        return False
    if dated:
        return True
    return bool(extract_numbers(text))


def _search_date_covers(entry, requested_date):
    """一条搜索记录的日期对得上要问的那天吗？
    date_range 起止同一天就是那天；或查询词文本里直接写了那天
    （“纳斯达克 11 月 10 日收盘”没带 date_range 也算）。
    requested_date 是 None 时（问题里没写日期），任何日期都算对得上——
    门禁只查“搜没搜这个实体”。"""
    if requested_date is None:
        return True
    if isinstance(requested_date, str):
        try:
            requested_date = _date.fromisoformat(requested_date)
        except ValueError:
            return False
    date_range = (entry or {}).get("date_range") or {}
    start, end = date_range.get("start"), date_range.get("end")
    if start and start == end:
        try:
            if _date.fromisoformat(start) == requested_date:
                return True
        except ValueError:
            pass
    for text in ((entry or {}).get("query") or "", (entry or {}).get("executed_query") or ""):
        mentions = parse_date_mentions(text, default_year=requested_date.year)
        if any(item["date"] == requested_date for item in mentions):
            return True
    return False


def search_covers_pair(searches, entity_key, requested_date):
    """51 号：这（实体，日期）对搜过没有。
    搜过 = 有一条搜索同时满足：查询词里提到这个实体（没点名具体实体时，
    任何行情搜索都算），日期对得上，而且那次搜索真的拿回了结果
    （ok 且结果非空——搜失败了/没结果不算搜过）。
    老任务记录里没有 ok / result_count 字段时，按搜过算（不拦旧账）。"""
    for entry in searches or []:
        if entry.get("ok", True) is False:
            continue
        if entry.get("result_count", 1) <= 0:
            continue
        if entity_key is not None:
            _key, _label, keywords = next(
                (item for item in _MARKET_ENTITIES if item[0] == entity_key),
                (entity_key, entity_key, (entity_key,)),
            )
            haystack = f"{entry.get('query') or ''} {entry.get('executed_query') or ''}".lower()
            if not any(keyword in haystack for keyword in keywords):
                continue
        if _search_date_covers(entry, requested_date):
            return True
    return False


# ---------- 校验回答 ----------


def find_date_mismatches(reply, dated_numbers, requested_date):
    """回答里引用的数，哪些和要问的那天对不上。
    dated_numbers 是 [{"number", "date"}]（date 是 "YYYY-MM-DD" 或 None），
    requested_date 是 date 对象或 "YYYY-MM-DD" 字符串。
    返回 (mismatches, unverified)：
    - mismatches：[{"number", "kind", "value", "found_dates"}]——
      这个数在结果里只出现在别的日期上，没出现在要问的那天。
    - unverified：[number 文本]——结果里根本没出现过这个数，
      没法按日期验证（不拦，只记下来）。
    同一个数在回答里出现多次只算一次；数对得上（0.02 以内）就算同一个数。"""
    if isinstance(requested_date, str):
        try:
            requested_date = _date.fromisoformat(requested_date)
        except ValueError:
            return [], []
    if requested_date is None:
        return [], []
    wanted = requested_date.isoformat()
    # 按（种类，数值）归集每个数出现过哪些日期（只算带日期的）。
    dates_by_key = {}
    for entry in dated_numbers or []:
        entry_date = entry.get("date")
        if not entry_date:
            continue
        parsed = parse_number_value(entry.get("number") or "")
        if parsed is None:
            continue
        kind, value = parsed
        dates_by_key.setdefault((kind, round(value, 2)), set()).add(entry_date)
    mismatches = []
    unverified = []
    seen = set()
    for number in extract_numbers(reply or ""):
        key = (number["kind"], round(number["value"], 2))
        if key in seen:
            continue
        seen.add(key)
        matched_dates = set()
        found = False
        for (kind, value), dates in dates_by_key.items():
            if kind != number["kind"]:
                continue
            if abs(value - number["value"]) <= NUMBER_TOLERANCE:
                found = True
                matched_dates |= dates
        if not found:
            unverified.append(number["text"])
            continue
        if wanted in matched_dates:
            continue
        mismatches.append(
            {
                "number": number["text"],
                "kind": number["kind"],
                "value": number["value"],
                "found_dates": sorted(matched_dates),
            }
        )
    return mismatches, unverified


# ---------- 追问补全（2026-11-13：上下文接日期） ----------

# “标普呢”接在“道指11.10”后面问，意思就是“标普的 11.10”——
# 但当时的解析只看本句：本句没日期，51 号门禁不开，模型直接说
# 不知道，还反问“要不要联网查”。根因是追问没继承上文的日期/实体。
# 治本：在门禁前面加一层“追问补全”，只补行情实体和日期，不改
# 门禁本身的逻辑；补全后的问题再走正常流程（门禁、校验都认它）。
# 保守起见，只有同时满足下面三条才补：
#   1. 本句缺日期或缺实体（两个都有时不用补）；
#   2. 本句像追问（短 + 带追问词，或光秃秃一个实体）；
#   3. 最近 1–2 轮用户问话里有缺的那块（行情问题才算）。
# 补全格式按 BOOS 自己的说法：“标普的 11.10”（实体 + 的 + M.D），
# 下游解析（resolve_requested_date / extract_market_entity /
# is_market_data_question）都认这个格式。
# 不补的情形：本句有明确的相对日期（今天/昨天）——那是它自己的日期，
# 不是上文的；上文也没有日期/实体；本句不是追问（比如完整的新问题）。

_FOLLOW_UP_TAIL_MARKERS = ("怎么样", "如何", "怎样", "呢", "吗")
_FOLLOW_UP_HEAD_MARKERS = ("那", "那么", "然后", "还有", "另外")


def _looks_like_follow_up(question):
    """这句像不像追问（短 + 带追问词，或光秃秃一个实体）？
    长句子不算——“纳指昨天涨还是跌”这种自己带齐信息的不是追问。"""
    text = (question or "").strip()
    if not text or len(text) > 15:
        return False
    if any(text.startswith(head) for head in _FOLLOW_UP_HEAD_MARKERS):
        return True
    if any(marker in text for marker in _FOLLOW_UP_TAIL_MARKERS):
        return True
    key, _label = extract_market_entity(text)
    return key is not None and len(text) <= 6


def _follow_up_tail(question):
    """追问里要保留的问法尾巴：怎么样/如何/怎样，或“涨了吗/跌了吗”
    这种涨跌问法；光秃秃的“呢/吗”只是语气词，不保留。
    返回要接在补全问题后面的尾巴（带前面的空格），没有返回 ""。"""
    text = (question or "").strip()
    for marker in ("怎么样", "如何", "怎样"):
        if marker in text:
            return f" {marker}"
    lowered = text.lower()
    for word in ("涨", "跌"):
        index = lowered.find(word)
        if index != -1 and "吗" in text[index:]:
            return f" {text[index:].strip()}"
    return ""


def _leftover_after_markers_and_dates(text, today):
    """去掉日期和追问词后，问题里还剩什么实质内容。
    缺实体时只有剩空了才敢补上文的实体——“比特币呢”剩个“比特币”，
    绝不能补成上文的道指；“11.10 怎么样”剩空，才能补。"""
    stripped = text or ""
    for mention in parse_date_mentions(text or "", default_year=today.year):
        start, end = mention["start"], mention["end"]
        stripped = stripped[:start] + " " * (end - start) + stripped[end:]
    for word in (
        "大前天", "前天", "昨天", "昨日", "今天", "今日", "明天", "明日", "后天",
        "怎么样", "如何", "怎样", "呢", "吗", "那", "那么", "然后", "还有", "另外",
    ):
        stripped = stripped.replace(word, "")
    return re.sub(r"[\s，,。？?！!、：:；;]+", "", stripped)


def complete_follow_up_question(question, history, today=None):
    """追问补全：当前问题缺日期/实体、又像追问时，从最近 1–2 轮
    用户问话里把缺的那块补上（只补行情实体和日期，别的不碰）。
    返回补全后的问题；不用补时原样返回。
    history 是之前的对话消息列表（user/assistant），只看 user 的；
    today 不传时按上海时间算今天。
    例子（today=2026-11-12，上轮问过“道指11.10”）：
      “标普呢” → “标普的 11.10”；
      “那道指呢” → “道指的 11.10”；
      “11.10 怎么样” → “道指的 11.10 怎么样”；
      “呢” → “道指的 11.10”。"""
    text = (question or "").strip()
    if not text:
        return question
    if today is None:
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    entity_key, _label = extract_market_entity(text)
    asked_date = resolve_requested_date(text, today)
    if entity_key is not None and asked_date is not None:
        return question
    if not _looks_like_follow_up(text):
        return question
    # 最近 1–2 轮用户问话里找日期和实体（离现在越近越优先）：
    # 先找同一轮里两个都有的，再分开找最近的日期、最近的实体。
    user_turns = [
        msg.get("content") or ""
        for msg in reversed(history or [])
        if isinstance(msg, dict)
        and msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
    ][:2]
    prev_date = None
    prev_entity_key = None
    prev_entity_label = None
    for prev in user_turns:
        prev_asked = resolve_requested_date(prev, today)
        prev_key, prev_label = extract_market_entity(prev)
        if prev_date is None and prev_asked is not None:
            prev_date = prev_asked
        if prev_entity_key is None and prev_key is not None:
            prev_entity_key, prev_entity_label = prev_key, prev_label
        if prev_date is not None and prev_entity_key is not None:
            break
    if asked_date is None and prev_date is None:
        return question
    if entity_key is None and prev_entity_key is None:
        return question
    # 缺实体时，先看本句除了日期和追问词还剩什么：剩了实质内容
    # （比如“比特币”），说明问的是别的东西，别拿上文的实体顶上。
    if entity_key is None and _leftover_after_markers_and_dates(text, today):
        return question
    # 缺哪块补哪块：实体用中文名（“标普”“道指”），日期用 M.D（“11.10”）。
    if entity_key is None:
        _key, entity_label = prev_entity_key, prev_entity_label
    else:
        _key, entity_label = extract_market_entity(text)
    if asked_date is None:
        filled_date = prev_date
    else:
        filled_date = asked_date
    tail = _follow_up_tail(text)
    return f"{entity_label}的 {filled_date.month}.{filled_date.day}{tail}"
