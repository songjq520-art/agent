# memory/content_check.py（补丁 4 第一层：程序对原话）
# 作用：十二的记忆没人查内容对不对——一条错记忆没人反对，就会越强化
#       越靠前。这一层不调大模型，只做程序对原话：每条记忆留着出处，
#       对着原话看记忆写得对不对，对不上就标出来；再把最可能错的几条
#       排出来，给月度自检报告用。
# 注意：这里全是纯函数（只用标准库），不读库、不调模型、不改任何东西；
#       自检模块调 rank_most_likely_wrong(items, top_n=5)，签名别改。

import re

__all__ = ["extract_checkables", "grounding_check", "rank_most_likely_wrong"]

# 数字后面可能跟的单位（长的先写：正则按顺序试，“万元”要在“万”前面，
# “美元”要在“元”前面，不然“1.35万元”会被先拆成“1.35万”）。
_UNITS = (
    "美元",
    "人民币",
    "万元",
    "小时",
    "分钟",
    "元",
    "块",
    "%",
    "天",
    "月",
    "年",
    "万",
    "千",
    "秒",
    "次",
)
_UNIT_ALT = "|".join(_UNITS)

# 数字（含小数、千分位逗号，可带一个单位，数字和单位之间可隔一个空格）：
#   “5块”“1.35万元”“50%”“3天”“1,000元”都算一个数，保留原样字符串。
#   注意单位是可选的、且空格只在有单位时才吃掉——“5 个苹果”里的 5
#   只取“5”，不把后面的空格也吞进来。
_NUMBER_RE = re.compile(
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?:" + _UNIT_ALT + r"))?"
)
# 数字部分本身（从一次 _NUMBER_RE 匹配里抠出纯数字，用来判断它是不是
# 6 位代码——代码不进 numbers，只进 codes，免得同一串数在两边各报一次）。
_NUMBER_PART_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

# 6 位数字（基金代码样）：前后不能再跟数字，不然“1234567”会被拆出
# 一个假代码；也不认带小数点的（“123456.7”是个带小数的数，不是代码）。
_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)(?!\.\d)")

# 引号里的内容：中文引号 “...” 或英文引号 "..."。
_QUOTED_RE = re.compile(r"“([^”]*)”|\"([^\"]*)\"")


def _dedup_keep_order(values):
    """去重保序（第一次出现的位置不动，后面重复的不要）。"""
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _normalize_for_compare(text):
    """比对用的规整：去掉逗号（中英文）、顿号和所有空白再比。

    “5块”和“5 块”、“1,000元”和“1000元”说的是同一个数，不能因为
    一个空格、一个逗号就判对不上。只去掉这些分隔符，数字、单位、
    文字本身一个字都不动。
    """
    return re.sub(r"[,\s，、]", "", text or "")


def extract_checkables(text):
    """从一段文本里抠出能拿去跟原话对的项（纯函数）。

    返回 {"numbers": [...], "codes": [...], "quoted": [...]}：
    - numbers：文本里的数字（含小数，可带单位如 元/块/美元/%/天/月/年，
      保留原样字符串，去重保序）。正好 6 位的纯数字不算数，那是代码。
    - codes：6 位数字（基金代码样，如 019441），去重保序。
    - quoted：“...” 或 "..." 引号里的内容（空引号不要），去重保序。
    text 为空或不是字符串时返回三空列表，不抛错。
    """
    if not isinstance(text, str):
        text = ""
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        if not token:
            continue
        # 正好 6 位的纯数字是代码，不进 numbers（见 _CODE_RE 的注释）。
        number_part = _NUMBER_PART_RE.match(token).group(0)
        if re.fullmatch(r"\d{6}", number_part):
            continue
        numbers.append(token)
    codes = _CODE_RE.findall(text)
    quoted = []
    for match in _QUOTED_RE.finditer(text):
        content = match.group(1) if match.group(1) is not None else match.group(2)
        if content is not None and content.strip():
            quoted.append(content)
    return {
        "numbers": _dedup_keep_order(numbers),
        "codes": _dedup_keep_order(codes),
        "quoted": _dedup_keep_order(quoted),
    }


def grounding_check(memory_text, source_text):
    """把一条记忆对着原话查（纯函数，不调大模型）。

    memory 里的每个 number / code 必须原样出现在 source_text 里
    （比之前把两边的逗号、空格去掉再比，见 _normalize_for_compare）；
    quoted 是软检查：只报告在 missing_quoted 里，不判错。
    返回 {"ok": bool, "missing_numbers": [...], "missing_codes": [...],
          "missing_quoted": [...]}；ok 为 True 当且仅当
    missing_numbers 和 missing_codes 都为空。
    source_text 为空时返回 ok=False，且 missing 全是 memory 里的项——
    没出处就是对不上（哪怕 memory 里一项可查的都没有，ok 也是 False）。
    """
    checkables = extract_checkables(memory_text)
    if not isinstance(source_text, str) or not source_text.strip():
        return {
            "ok": False,
            "missing_numbers": list(checkables["numbers"]),
            "missing_codes": list(checkables["codes"]),
            "missing_quoted": list(checkables["quoted"]),
        }
    norm_source = _normalize_for_compare(source_text)
    missing_numbers = [
        number
        for number in checkables["numbers"]
        if _normalize_for_compare(number) not in norm_source
    ]
    missing_codes = [
        code
        for code in checkables["codes"]
        if _normalize_for_compare(code) not in norm_source
    ]
    # 引号是软检查：只报告、不判错——原话换了个说法也算对得上，
    # 这里只把原话里找不到的引文列出来给人看。
    missing_quoted = [
        quoted
        for quoted in checkables["quoted"]
        if _normalize_for_compare(quoted) not in norm_source
    ]
    return {
        "ok": not missing_numbers and not missing_codes,
        "missing_numbers": missing_numbers,
        "missing_codes": missing_codes,
        "missing_quoted": missing_quoted,
    }


def _metadata_field(item, metadata, name):
    """取一条记忆的规整后 metadata 字段。

    调用方传进来的字段可能在 item["metadata"] 字典里（规整过的那份），
    也可能直接平铺在 item 顶层——两处都认，metadata 里的优先。
    """
    if name in metadata:
        return metadata[name]
    return item.get(name)


def _confidence_value(raw):
    """把 confidence 规整成 0–1 的数字或 None（不知道就是不知道）。

    只认真的数字（布尔值不算数）；超出 0–1 的按缺失算，和
    memory/provenance.py 里 _normalize_metadata 的口径一致。
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if 0 <= value <= 1:
        return value
    return None


def rank_most_likely_wrong(items, top_n=5):
    """把最可能错的记忆排出来，给月度自检报告用（纯函数）。

    items 是每条记忆的 dict，键有 id、text 和规整过的 metadata 字段
    （user_confirmed、confidence、needs_review、conflict_for_review、
    grounding_flag、verification_status；metadata 字段可放在
    item["metadata"] 里，也可直接平铺在 item 顶层）。
    打分规则（分数越高越可疑）：
    - 未确认且 confidence 缺失或 <0.5，加 3 分；
    - grounding_flag 为真（跟出处对不上），加 2 分；
    - needs_review 为真（待复核），加 2 分；
    - conflict_for_review 为真（与其他记忆冲突待复核），加 2 分；
    - verification_status == "unverified"（未核实），加 1 分；
    - confidence <0.5（已确认的除外），加 1 分——未确认且可信度
      <0.5 时第一条和这一条都加，共 4 分；已确认的低可信度不加分，
      BOOS 亲口确认过的记忆不靠可信度数字来怀疑。
    返回 [{"id", "text", "score", "reasons": [...]}] 前 top_n 个：
    按分数从高到低排，分数相同按 id 排序（确定性，每次结果一样）；
    得分为 0 的不进榜（没疑点的不用 BOOS 过眼）。
    """
    scored = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        mem_id = item.get("id")
        if mem_id is None or (isinstance(mem_id, str) and not mem_id):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        # user_confirmed 只有明确是 True（布尔值）才算确认，和
        # _normalize_metadata 的口径一致——写成 "yes"/1 不算。
        user_confirmed = _metadata_field(item, metadata, "user_confirmed") is True
        confidence = _confidence_value(_metadata_field(item, metadata, "confidence"))
        score = 0
        reasons = []
        if not user_confirmed and (confidence is None or confidence < 0.5):
            score += 3
            reasons.append("未确认且可信度低")
        if _metadata_field(item, metadata, "grounding_flag") is True:
            score += 2
            reasons.append("跟出处对不上")
        if _metadata_field(item, metadata, "needs_review") is True:
            score += 2
            reasons.append("待复核")
        if _metadata_field(item, metadata, "conflict_for_review") is True:
            score += 2
            reasons.append("存在冲突待复核")
        if _metadata_field(item, metadata, "verification_status") == "unverified":
            score += 1
            reasons.append("未核实")
        if confidence is not None and confidence < 0.5 and not user_confirmed:
            score += 1
            reasons.append("可信度低于 0.5")
        if score <= 0:
            continue
        text = item.get("text")
        if not isinstance(text, str):
            # mem0 返回的条目把文本放 "memory" 键里，这里也认。
            text = item.get("memory") if isinstance(item.get("memory"), str) else ""
        scored.append(
            {"id": mem_id, "text": text, "score": score, "reasons": reasons}
        )
    scored.sort(key=lambda entry: (-entry["score"], str(entry["id"])))
    if top_n is None:
        top_n = 5
    try:
        limit = int(top_n)
    except (TypeError, ValueError):
        limit = 5
    if limit <= 0:
        return []
    return scored[:limit]
