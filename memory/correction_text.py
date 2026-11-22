# memory/correction_text.py（纠错文本判断：撤回、纠正、玩笑、身份陈述）
# 从 memory/mem0_store.py 拆出来，行为不变。

import json
import re
from memory.money_slots import SLOT_AGENT_NAME, SLOT_USER_NAME
from memory.redaction import _normalize_memory_text

# ---------- 聊天纠错的意图识别（供 Mem0Store.add_turn 用） ----------
# 只认明确的纠错说法，宁可漏认、不要误认——普通聊天绝不能触发纠错。
# “我不是”要特别小心：“我不是很确定”只是犹豫，不是纠错；所以
# “我不是”后面必须跟一个替换说法（“而是/其实是/，新说法”），
# 否则不触发（见 _detect_correction）。

# _CORRECTION_PHRASES（纠错说法）：只认这六个明确的说法。
# “记错了”不带“你”也认（“不是，记错了，我住广州”）；分隔符规则会
# 拦住“我昨天记错了开会时间”这种正常陈述（“记错了”前面是“天”，
# 不是分隔符，不触发）。
_CORRECTION_PHRASES = ("你记错了", "记错了", "不对", "纠正一下", "其实是", "我不是")

# _CORRECTION_SEPARATORS（纠错说法前面能跟的分隔符）：纠错说法必须在
# 句首，或在句号/逗号/换行之后——比如“我喜欢科幻电影。不对，我喜欢
# 纪录片。”这种一句里先肯定后纠正的也认；但“我觉得不对劲”里的
# “不对”前面是“得”，不是分隔符，不认。
_CORRECTION_SEPARATORS = set("。！？；\n，、： ")

# _BUDUI_BOUNDARY（“不对”后面能跟的字）：“不对”后面必须是个边界，
# 不然可能是“不对劲/不对头”这种词的一部分——“不对劲，我先走了”
# 不是纠错。标点、语气词（啊/呀/哦/呢/吧）、人称（我/你/他/这/那）
# 都算边界；其他汉字（劲/头/味/路……）不算。
_BUDUI_BOUNDARY = set("，。：、；！？\n \t啊呀哦呢吧我你他她它这那是")

# _CHINESE_NUMERALS（挑候选时认的中文数字）：“第 1 条”里的 1 是阿拉伯
# 数字，“第一个”里的“一”是中文数字，都认。
_CHINESE_NUMERALS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

def _detect_correction(text):
    """从一句话里认出“纠错意图 + 新说法”。

    返回 {"phrase": 触发的说法, "fact": 新说法}，不是纠错时返回 None。
    认法（保守，只认明确的）：
    - 说法必须在句首，或在句号/逗号/换行等分隔符之后；
    - “不对”后面必须是个边界（标点/语气词/人称），排除“不对劲”这种词；
    - 新说法 = 说法后面那一句的话（到第一个句号为止），同一轮里
      后面的句子是别的内容，不算进新说法里；
    - “我不是 X，Y”这种否定句式：Y 才是新说法——先看“而是/其实是”，
      没有就取第一个逗号后面的；两个都没有（比如“我不是很确定”）
      就不是纠错，返回 None；
    - 新说法太短（只剩一个字或空）不算纠错，走正常保存。
    例子：
    - “不对，我喜欢看纪录片。”→ fact “我喜欢看纪录片。”
    - “你记错了，我喜欢用 Go 写代码”→ fact “我喜欢用 Go 写代码”
    - “我不是喜欢科幻电影，我喜欢纪录片。”→ fact “我喜欢纪录片。”
    - “我不是很确定”→ None（只是犹豫，没有替换说法）
    - “我觉得不对劲”→ None（“不对”是词的一部分）
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    best = None  # (出现位置, 说法)：多个说法同时出现时取最早的。
    for phrase in _CORRECTION_PHRASES:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            start = idx + 1
            if idx != 0 and text[idx - 1] not in _CORRECTION_SEPARATORS:
                continue
            if phrase == "不对":
                after = text[idx + len(phrase):idx + len(phrase) + 1]
                if after and after not in _BUDUI_BOUNDARY:
                    continue
            if best is None or idx < best[0]:
                best = (idx, phrase)
            break  # 同一个说法只取第一次出现。
    if best is None:
        return None
    idx, phrase = best
    fact = text[idx + len(phrase):].lstrip("，,：:、；; \t")
    if phrase == "我不是":
        # 否定句式：新说法在“而是/其实是”后面，或在第一个逗号后面。
        for marker in ("而是", "其实是"):
            pos = fact.find(marker)
            if pos >= 0:
                fact = fact[pos + len(marker):].lstrip("，,：:、；; \t")
                break
        else:
            comma = fact.find("，")
            if comma < 0:
                comma = fact.find(",")
            if comma >= 0:
                fact = fact[comma + 1:].lstrip(" \t")
            else:
                return None
    fact = fact.strip()
    # 新说法只取纠错说法所在的那一句——同一轮里后面的句子是别的内容，
    # 不该被纠正的新文本吞进去（2 号：不然“我喜欢喝咖啡”会被存进
    # 名字记忆里，后面待定区提交时又存一条，重复且文本被污染）。
    term = re.search(r"[。！？\n]", fact)
    if term:
        fact = fact[:term.end()].strip()
    # 新说法太短（去掉句末标点只剩一个字、或空）不算纠错，走正常保存。
    if len(fact.rstrip("。！？；，、.!?;,")) < 2:
        return None
    return {"phrase": phrase, "fact": fact}

def _latest_user_content(messages):
    """取消息列表里最后一条 user 消息的内容（本轮用户刚说的那句）。

    add_turn 收到的可能是最近几轮的消息（CLI/网页都只带最近 6 条），
    纠错只看最新那句——以前轮次的纠错在它那轮已经处理过了。
    没有 user 消息时返回 ""。
    """
    for message in reversed(messages or []):
        if (message or {}).get("role") == "user":
            return (message or {}).get("content") or ""
    return ""

def _strip_correction_sentences(latest_user):
    """去掉纠错句，留非纠正部分（Plan C 2 号）。

    一句一句看：含纠错说法的那句，只留说法前面的部分（有实质内容
    才留，比如“我喜欢喝咖啡，你记错了，我叫小宋”里留“我喜欢喝咖啡”），
    说法后面的新说法已由 correct_memory 处理，不再重复存；
    没有纠错说法的句子原样留。
    返回剩下的句子列表，没有剩下时返回 []。
    """
    kept = []
    for sentence in re.split(r"[。！？\n]+", latest_user or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        detected = _detect_correction(sentence)
        if detected is None:
            kept.append(sentence)
            continue
        before = sentence[:sentence.find(detected["phrase"])].strip()
        before = before.strip("，,：:、；; \t")
        if len(before) >= 2:
            kept.append(before)
    return kept

# 碎片纠正：纠正新说法是个碎片（不完整的一句）时，先补成整句再改。
_FRAGMENT_MAX_LEN = 12  # 碎片一般很短，超过这个长度就当完整的一句处理

# 含糊指代：值本身没有内容（“那个”“这个”“它”），补不出来，直接走老路。
_VAGUE_FRAGMENT_VALUES = (
    "那个", "这个", "它", "他", "她", "那", "这",
    "那个东西", "这个东西", "那东西", "这东西",
)

def _detect_correction_fragment(text):
    """认出纠错新说法是不是个“碎片”（不完整的一句）。

    碎片 = “是 + 值”（“是绿色”）或光秃秃一个短语（“绿色”“喜欢绿色”）。
    返回碎片里的“值”（“绿色”），不是碎片时返回 None。
    完整的一句（“我喜欢绿色。”）有主语，不算碎片。
    """
    if not isinstance(text, str):
        return None
    s = text.strip().rstrip("。！？；，、.!?;,").strip()
    if not s or len(s) > _FRAGMENT_MAX_LEN:
        return None
    if s.startswith("是"):
        value = s[1:].strip("，,：:、；; \t")
        return value or None
    # 光秃秃一个短语：不以主语开头——“绿色”“喜欢绿色”。
    if s[0] in ("我", "你", "他", "她", "它", "这", "那"):
        return None
    return s

# _RETRACTION_SIGNALS（撤回信号）：整句只有这些词（加句末标点）时，
# 意思是“刚说的那句撤回、不记了”——直接丢掉待定区里的上一轮，
# 不存、不问。只认“整句只有信号”：“不对，我喜欢看纪录片。”这种
# 带替换内容的不是撤回，走纠错。没有“没有”——光秃秃的“没有”常常
# 是在回答助手刚问的问题（“你喜欢喝茶吗？”“没有。”），不是撤回
# 上一句，误杀会丢掉刚说的重要事实。
_RETRACTION_SIGNALS = frozenset(
    [
        "不不不",
        "不对不对",
        "不不",
        "不对",
        "不是",
        "算了",
        "别记",
        "别记了",
        "不要记",
        "不要记了",
        "不记了",
        "别存",
        "不要存",
        "错了",
        "说错了",
        "收回",
        "撤回",
    ]
)

_RETRACTION_TRAILING_PUNCT = "。！？.!?~～ \t\n"

# 条件句开头：不是既定事实，提取时整句去掉（和 FakeMemory 认同一套）。
_CONDITIONAL_LEADS = ("如果", "假设", "假如", "要是", "万一")

# 玩笑标记：句里出现这些词，这句就是在开玩笑/说着玩，不是既定事实
# （Plan C 闸门 1 的代码兜底；提示词第 13 条仍在，模型判断是第一道，
# 这里是第二道——模型哪天没拦住，代码拦）。
# 写法收紧：只认带明确标记的说法——“我喜欢开玩笑”这种真偏好不带
# 这些标记，不会被误拦。
_JOKE_MARKERS = (
    "开玩笑的", "开玩笑呢", "开玩笑了", "我开玩笑", "只是开玩笑",
    "开个玩笑", "逗你玩", "闹着玩", "说着玩", "别当真",
    "瞎说的", "胡说的",
)

# 明显不可能的数字（只认这两类最稳的，其余交给模型判断）：
# - 年龄超过 120 岁（人类最长寿 122 岁，留一点余量）；
# - 身高超过 3 米（人类最高 2.72 米）。
_JOKE_AGE_LIMIT = 120

_JOKE_HEIGHT_LIMIT_M = 3.0

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "两": 2}

_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

_AGE_RE = re.compile(r"([一二三四五六七八九十百千两\d]+)\s*岁")

# 身高只认两种写法：“身高 3.5 米”“我 3.5 米高”——“我 100 米跑”
# 这种跑步距离不认（“米”后面没有“高”，也不是“身高”开头）。
_HEIGHT_RE = re.compile(
    r"(?:身高\s*([一二三四五六七八九十百千两\d]+(?:\.\d+)?)\s*米"
    r"|我\s*([一二三四五六七八九十百千两\d]+(?:\.\d+)?)\s*米高)"
)

# 身份陈述的触发词：按“先 user、后 agent”试——“你叫我小宋”里的
# “你叫”是叫“我”，不是在给 Agent 起名；每组内按触发词从长到短试。
_IDENTITY_USER_TRIGGERS = ("我的名字是", "我的名字叫", "你叫我", "叫我", "我叫")

_IDENTITY_AGENT_TRIGGERS = ("以后你叫", "现在你叫", "你的名字是", "你的名字叫", "你叫", "助手叫", "助手的名字是", "助手的名字叫")

_IDENTITY_VALUE_MAX_LEN = 12  # 名字值最多 12 字

_IDENTITY_TRAILING_PARTICLES = "了啊呀呢吧哦嘛呗啦喽哈"  # 名字末尾的语气词，去掉

# 问句尾巴：值以这些结尾时整句是问句（“你叫十三好不好”是在问，
# 不是在起名），不当身份陈述认。
_IDENTITY_QUESTION_SUFFIXES = ("好不好", "行不行", "可以吗", "好吗")

def _is_bare_retraction(text):
    """整句只有撤回信号时返回 True（Plan C 待定区撤回）。

    “不不不。”“不对。”“算了。”这种——只有信号加句末标点。
    “不对，我喜欢看纪录片。”带替换内容，不是撤回，返回 False。
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip().strip(_RETRACTION_TRAILING_PUNCT).strip()
    return stripped in _RETRACTION_SIGNALS

def _detect_identity_statement(sentence):
    """从一句话里认出“身份/称呼陈述句”（Plan C 闸门 2，代码定、不是模型定）。

    只认陈述句：带“什么/吗/?”的一律不认（“你叫什么？”是提问，
    不是在告诉名字）。认到返回 {"slot", "value", "text"}：
    - agent_name（十二的名字）：“你叫十二”“以后你叫十二了”
      → text "Agent 的名字是十二，你叫十二"（带“你叫”二字，
      离线关键词召回问“你叫什么”时也能找着）；
    - user_name（BOOS 的称呼）：“我叫小宋”→ 原句存（去句末标点）；
      “叫我小宋”/“你叫我小宋”→ "我叫小宋，请称呼我小宋"。
    名字值最多 12 字，末尾的语气词（了/啊/呀/呢/吧/哦/嘛/呗/啦/喽/哈）去掉；
    值以“好不好/行不行/可以吗/好吗”结尾时整句是问句，返回 None。
    认不到返回 None。
    """
    if not isinstance(sentence, str):
        return None
    text = sentence.strip().strip("。！？；，、.!?;,")
    if not text:
        return None
    if any(q in text for q in ("什么", "吗", "？", "?")):
        return None
    if any(marker in text for marker in _JOKE_MARKERS):
        # 玩笑里的起名不当真（“开玩笑的，你叫十三”不是在改名）。
        return None
    for triggers, slot in (
        (_IDENTITY_USER_TRIGGERS, SLOT_USER_NAME),
        (_IDENTITY_AGENT_TRIGGERS, SLOT_AGENT_NAME),
    ):
        for trigger in triggers:
            idx = text.find(trigger)
            if idx < 0:
                continue
            rest = text[idx + len(trigger):]
            # 值到第一个标点为止。
            value = re.split(r"[，。！？、；;,\s]", rest, maxsplit=1)[0].strip()
            value = value.rstrip(_IDENTITY_TRAILING_PARTICLES)
            # 问句尾巴：整句是在问“这样叫好不好”，不是在起名。
            if any(value.endswith(suffix) for suffix in _IDENTITY_QUESTION_SUFFIXES):
                return None
            if not value or len(value) > _IDENTITY_VALUE_MAX_LEN:
                continue
            if slot == SLOT_USER_NAME:
                if trigger in ("叫我", "你叫我"):
                    canonical = f"我叫{value}，请称呼我{value}"
                else:
                    canonical = text  # “我叫小宋”这类原句直接存
            else:
                canonical = f"Agent 的名字是{value}，你叫{value}"
            return {"slot": slot, "value": value, "text": canonical}
    return None

def _parse_chinese_number(text):
    """把“十/百/千”组合的中文数字转成整数（只做年龄、身高这点小事）。

    一百五十 → 150，二十 → 20，一千二百三十四 → 1234。
    认不出来（混了阿拉伯数字、带“点”这种）返回 None，不猜。
    """
    if not text or not all(ch in _CN_DIGITS or ch in _CN_UNITS for ch in text):
        return None
    total = 0
    current = 0
    seen_unit = False
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        else:
            unit = _CN_UNITS[ch]
            if current == 0:
                current = 1  # “十”就是 10，“百”就是 100
            total += current * unit
            current = 0
            seen_unit = True
    total += current
    if not seen_unit and total == 0:
        return None
    return total

def _is_joke_or_exaggeration(sentence):
    """Plan C 闸门 1 的代码兜底：明显是玩笑/夸张的句子返回 True。

    只认最稳的三类，其余交给提示词第 13 条（模型判断）：
    - 句里带明确玩笑标记（开玩笑的/逗你玩/闹着玩/说着玩/别当真/
      瞎说的/胡说的）——整句当玩笑处理，连着的真话也一起去掉，
      宁可漏记一句（用户会再说），不把玩笑存成事实；
    - 年龄超过 120 岁（阿拉伯数字或中文数字，如 150 岁/一百五十岁）；
    - 身高超过 3 米（身高 3.5 米/我 3.5 米高）。
    “我一百岁了”（100 岁）、“姚明 2.26 米”这种没超线的正常话不拦。
    """
    if not isinstance(sentence, str) or not sentence:
        return False
    if any(marker in sentence for marker in _JOKE_MARKERS):
        return True
    age_match = _AGE_RE.search(sentence)
    if age_match:
        raw = age_match.group(1)
        age = int(raw) if raw.isdigit() else _parse_chinese_number(raw)
        if age is not None and age > _JOKE_AGE_LIMIT:
            return True
    height_match = _HEIGHT_RE.search(sentence)
    if height_match:
        raw = height_match.group(1) or height_match.group(2)
        if raw is None:
            return False
        if "." in raw:
            try:
                height = float(raw)
            except ValueError:
                height = None
        elif raw.isdigit():
            height = float(raw)
        else:
            parsed = _parse_chinese_number(raw)
            height = float(parsed) if parsed is not None else None
        if height is not None and height > _JOKE_HEIGHT_LIMIT_M:
            return True
    return False

def _filter_extraction_messages(messages):
    """Plan C 闸门 1：从待提交的对话里去掉不该进提取的东西，
    返回过滤后的消息列表（assistant 的话原样透传，只当提取上下文）。

    去掉三类 user 句子：
    - 条件/假设句开头的（如果/假设/假如/要是/万一）——不是既定事实；
    - 身份/称呼陈述句——身份事实只走确定性的槽位路径
      （_upsert_slot_memory），不经过模型提取，避免同一事实存两份；
    - 玩笑/夸张句（_is_joke_or_exaggeration）——明确玩笑标记、
      年龄超 120 岁、身高超 3 米；其余玩笑仍靠提示词第 13 条拦。
    一条 user 消息里句子全被去掉时，整条消息不要了。
    """
    kept = []
    for message in messages or []:
        role = (message or {}).get("role")
        content = (message or {}).get("content") or ""
        if role != "user":
            if role == "assistant":
                kept.append({"role": "assistant", "content": content})
            continue
        sentences = re.split(r"[。！？\n]+", content)
        kept_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence.startswith(_CONDITIONAL_LEADS):
                continue
            if _detect_identity_statement(sentence) is not None:
                continue
            if _is_joke_or_exaggeration(sentence):
                continue
            kept_sentences.append(sentence)
        if kept_sentences:
            kept.append(
                {"role": "user", "content": "。".join(kept_sentences) + "。"}
            )
    return kept

def _correction_result(outcome):
    """把 correct_memory 的返回收成调用方好用的样子（只留必要字段）。

    {"status", "memory_id", "old_text", "new_text",
     "candidates": [{"id", "text"}]（ambiguous 时给调用方展示用）,
     "reason": 没改成的原因（给人看的）,
     "layer": 哪一层定的（exact/single/judge/judge_blocked/ask/none，
     见 correct_memory）,
     "resolution": 这次纠正的结果（updated/judge_blocked/ask/
     not_found/unchanged/failed，见 correct_memory）,
     "audit": 审计信息（picked_id/judge_reason/old_new_similarity/
     candidate_scores/context_ids，见 correct_memory）}。
    """
    outcome = outcome or {}
    return {
        "status": outcome.get("status"),
        "memory_id": outcome.get("memory_id"),
        "old_text": outcome.get("old_text"),
        "new_text": outcome.get("new_text"),
        "candidates": [
            {"id": (c or {}).get("id"), "text": (c or {}).get("text")}
            for c in (outcome.get("candidates") or [])
        ],
        "reason": outcome.get("reason") or "",
        "layer": outcome.get("layer") or "none",
        "resolution": outcome.get("resolution") or "failed",
        "audit": outcome.get("audit") or {},
    }

def _parse_pending_selection(text, candidates):
    """用户回“第 1 条/第一个/1”或直接引用某条候选原文时，认出他指的是哪条。

    返回挑中的候选（{"id", "text"}），认不出时返回 None——认不出
    就当放弃这次纠错（add_turn 会清掉待确认、不再问），绝不猜一条。
    认法（按明确程度从高到低）：
    - “第 1 条 / 第1条 / 第一个 / 1”：数字指明第几条；
    - 直接引用候选原文：某条候选里连续 4 个字出现在用户的话里，
      且只有这一条对得上（两条都对得上时不猜，返回 None）。
      比如候选是“我喜欢看科幻电影”，用户说“科幻电影那条”——
      “科幻电影”这 4 个字只在这一条里，就认它。
    """
    if not isinstance(text, str) or not candidates:
        return None
    stripped = text.strip()
    match = re.search(r"第\s*(\d+)\s*条", stripped)
    if match:
        n = int(match.group(1))
        return candidates[n - 1] if 1 <= n <= len(candidates) else None
    match = re.search(r"第\s*([一二三四五六七八九十])\s*个", stripped)
    if match:
        n = _CHINESE_NUMERALS[match.group(1)]
        return candidates[n - 1] if 1 <= n <= len(candidates) else None
    if re.fullmatch(r"\d+", stripped):
        n = int(stripped)
        return candidates[n - 1] if 1 <= n <= len(candidates) else None
    # 直接引用原文：候选里连续 4 个字出现在用户的话里，且只有一条对得上。
    norm_text = _normalize_memory_text(stripped)
    matched = []
    for cand in candidates:
        norm_cand = _normalize_memory_text((cand or {}).get("text") or "")
        if len(norm_cand) < 4:
            continue
        windows = {norm_cand[i:i + 4] for i in range(len(norm_cand) - 3)}
        if any(window in norm_text for window in windows):
            matched.append(cand)
    if len(matched) == 1:
        return matched[0]
    return None

def _parse_judge_choice(content, candidates):
    """从 DeepSeek 纠错判断的返回里抠出它选的候选 id（供 _judge_correction 用）。

    认三种写法：{"choice": "<候选 id>"}（标准）、{"choice": "none"}
    （分不清/没有冲突的）、直接写 id 本身（或第几条的数字序号）。
    返回 (id 或 None, reason)：
    - 选中的 id 时 reason 是 None；
    - choice 是 none 时 reason 是 "similar"（两条都像、分不开）或
      "unrelated"（哪个都不像、没关系），取自返回里的 "reason" 字段；
      reason 缺失或认不出时默认 "similar"——重试一次是安全的，
      最坏也只是多调一次然后问 BOOS。
    分不清、格式不对、选的 id 不在候选里时 id 都是 None，调用方
    走问 BOOS 的流程，绝不猜。模型有时会在 JSON 外面包一层
    ```json 围栏，先剥掉再解析。
    """
    if not isinstance(content, str) or not content.strip():
        return None, "similar"
    text = content.strip()
    # 去掉 ```json ... ``` 这种围栏（模型有时会包一层）。
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    ids = {(c or {}).get("id") for c in candidates or []}
    choice = None
    reason = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            choice = data.get("choice")
            reason = data.get("reason")
        elif isinstance(data, (str, int)) and not isinstance(data, bool):
            # 整段就是答案本身（比如直接写了个 id，或写了个数字序号）。
            choice = data
    except ValueError:
        # 不是 JSON：整段就是它写的答案（比如直接写了个 id）。
        choice = text
    if isinstance(reason, str):
        reason = reason.strip().lower()
        if reason not in ("similar", "unrelated"):
            reason = None
    else:
        reason = None
    if isinstance(choice, int) and not isinstance(choice, bool):
        # 数字序号也认（1 → 第 1 条）。
        if 1 <= choice <= len(candidates or []):
            return (candidates[choice - 1] or {}).get("id"), None
        return None, "similar"
    if not isinstance(choice, str):
        return None, "similar"
    choice = choice.strip().strip("\"'")
    if choice.lower() in ("none", "unclear", "不知道", "分不清", ""):
        return None, reason or "similar"
    if choice in ids:
        return choice, None
    if choice.isdigit():
        n = int(choice)
        if 1 <= n <= len(candidates or []):
            return (candidates[n - 1] or {}).get("id"), None
    return None, "similar"

