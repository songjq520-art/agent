# memory/redaction.py（隐私脱敏：敏感信息识别、脱敏、去重前规整）
# 从 memory/mem0_store.py 拆出来，行为不变。

import re
import unicodedata

# ---------- 敏感信息过滤（隐私三层过滤：存前脱敏 / 用时拦截 / 导出排除） ----------
# 为什么要有这一层：记忆里可能出现密码、API Key、身份证号、银行卡号——
# 这些东西一旦进了记忆，有三条路会往外漏：提取时发给 DeepSeek、
# 使用时拼进发给模型的问题、导出/迁移时被整批读出来。
# 所以过滤做在三个地方（见 Mem0Store 里各方法的注释），日志里也绝不
# 出现原文。注意：脱敏是按正则模式匹配的，不是万无一失——没见过的
# 密钥格式、中间带空格的多词密码可能漏网；这层是“减少泄露面”，
# 不是“保证绝不泄露”；真的密钥还是不要往对话里发。

# _REDACTED（脱敏占位符）：秘密被替换成的文字。占位符本身不含任何
# 原文信息，提取器看到它也不会把它当成事实存下来。
_REDACTED = "【已隐藏敏感信息】"

# _PRIVATE_KEY_RE（私钥块）：-----BEGIN ... PRIVATE KEY----- 整块都是秘密，
# 整块替换，不留一行。
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

# _JWT_RE（JWT 令牌）：eyJ 开头的三段式（header.payload.signature）。
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
)

# _KNOWN_KEY_RES（已知密钥格式，不带标签也认得）：
# OpenAI/Anthropic 风格的 sk-、GitHub 的 ghp_/gho_/github_pat_、
# AWS 的 AKIA、Google 的 AIza、Slack 的 xox、HuggingFace 的 hf_。
# 前面的 (?<![A-Za-z0-9...]) 是为了不把长单词里的一段误当成密钥
# （比如 task-abcdef... 里的 "sk-..."）。
_KNOWN_KEY_RES = [
    re.compile(r"(?<![A-Za-z0-9-])sk-(?:ant|proj)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9-])sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,}"),
]

# _ID18_RE（18 位身份证号）：6 位地区 + 8 位生日 + 3 位顺序 + 1 位校验。
# 只认生日像生日的（19/20 年开头、月份 01–12、日期 01–31），
# 再加校验位验算（见 _id18_check_ok），避免把 18 位订单号误伤。
# 注意不用 \b：身份证号前面常直接跟中文（“身份证号是11010...”），
# 中文和数字之间没有 \b；改用“前后不能是数字”来定界。
_ID18_RE = re.compile(
    r"(?<![0-9])\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9])"
)

# _ID15_RE（15 位老身份证号）：6 位地区 + 6 位生日（YYMMDD）+ 3 位顺序。
_ID15_RE = re.compile(
    r"(?<![0-9])\d{6}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}(?![0-9])"
)

# _CARD_CANDIDATE_RE（银行卡号候选）：13–19 位数字，中间可带空格或短横。
# 光长度不够，还要过 Luhn 校验（见 _luhn_ok）才算，避免把手机号、
# 订单号这种长数字串误伤。
_CARD_CANDIDATE_RE = re.compile(r"(?<![0-9])\d(?:[ -]?\d){12,18}(?![0-9])")

# _SECRET_LABELS（带标签的秘密）：中文/英文里“某某是/为/：秘密”的说法。
# 标签后面跟的才是秘密值，只替换值、保留标签和前面的正常内容，
# 这样“喜欢喝咖啡，密码是 xxx”脱敏后还剩“喜欢喝咖啡”这句有用的。
# 值至少 3 个字符（“密码是多少”这种问句里的“多少”只有 2 个字，不会误伤），
# 遇到空格、逗号、句号就停（值里带空格的多词密码脱不干净，是已知局限，
# 见文件头“实话”）。
_SECRET_LABELS = (
    r"密码|口令|passwd|password|pwd",
    r"api[_ -]?key|api[_ -]?密钥|密钥|秘钥",
    r"secret|token|令牌|access[_ -]?key|auth[_ -]?token",
    r"身份证(?:号)?|银行卡号|信用卡号|卡号",
)

_LABELED_SECRET_RE = re.compile(
    r"(?:"
    + "|".join(_SECRET_LABELS)
    + r")\s*(?:是|为|is|=|:|：)\s*([^\s，,、；;。！？\n]{3,})",
    re.IGNORECASE,
)

# _GENERIC_TOKEN_RE（没见过的长令牌）：32 位以上、同时含大小写字母和
# 数字的连续字符串（字母数字 _ -），大概率是某种 token/key。
# 要求混搭（纯小写十六进制的 git hash 不算、纯数字不算），且前面不能是
# / . : 或字母数字（网址、路径里的长段不算，避免把链接误伤）。
_GENERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_/.:-])"
    r"(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*\d)"
    r"[A-Za-z0-9_-]{32,}"
)

_ID18_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)

_ID18_CHECK_CHARS = "10X98765432"

def _id18_check_ok(number):
    """验算 18 位身份证号的校验位（GB 11643 加权求和取模）。"""
    if len(number) != 18:
        return False
    total = sum(int(digit) * weight for digit, weight in zip(number[:17], _ID18_WEIGHTS))
    return _ID18_CHECK_CHARS[total % 11] == number[17].upper()

def _luhn_ok(number):
    """Luhn 校验（银行卡号用的校验算法）：从右往左隔位翻倍、减 9，
    各位相加能被 10 整除就是合法卡号。"""
    digits = [int(ch) for ch in number if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0

def _redact_sensitive(text):
    """把一段文本里的敏感信息替换成占位符（只换秘密本身，保留其他内容）。

    顺序有讲究：先换整块/定格式的（私钥块、JWT、已知密钥格式、
    身份证号、银行卡号），再换带标签的（“密码是 xxx”——标签值里
    也可能是一串卡号，先换卡号能把带空格的卡号整段换掉），
    最后换没见过的长令牌。没有敏感信息时原样返回。
    注意：这个函数绝不能记日志、绝不能把原文写进任何地方——
    它处理的就是不能外泄的东西。
    """
    if not isinstance(text, str) or not text:
        return text
    redacted = _PRIVATE_KEY_RE.sub(_REDACTED, text)
    redacted = _JWT_RE.sub(_REDACTED, redacted)
    for pattern in _KNOWN_KEY_RES:
        redacted = pattern.sub(_REDACTED, redacted)
    redacted = _ID18_RE.sub(
        lambda m: _REDACTED if _id18_check_ok(m.group(0)) else m.group(0), redacted
    )
    redacted = _ID15_RE.sub(_REDACTED, redacted)
    redacted = _CARD_CANDIDATE_RE.sub(
        lambda m: _REDACTED if _luhn_ok(m.group(0)) else m.group(0), redacted
    )
    redacted = _LABELED_SECRET_RE.sub(
        lambda m: m.group(0)[: m.start(1) - m.start(0)] + _REDACTED, redacted
    )
    redacted = _GENERIC_TOKEN_RE.sub(_REDACTED, redacted)
    return redacted

def _contains_sensitive(text):
    """一段文本里是否含有敏感信息（按上面的模式判断）。"""
    if not isinstance(text, str) or not text:
        return False
    return _redact_sensitive(text) != text

# _MEANINGLESS_RES（脱敏后判断“还剩不剩有用的内容”时去掉的东西）：
# 占位符、标签词（密码/密钥/……）、虚词和标点都不算内容——
# “我的密码是 xxx”脱敏后只剩“我的密码是”，这不叫“有用的记忆”，
# 不值得存；“我喜欢喝咖啡，我的密码是 xxx”脱敏后还剩“喜欢喝咖啡”，
# 这才叫有用的记忆，要留下来。注意这三个正则只用来判断“还剩不剩”，
# 不用来改要存的文本。
_MEANINGLESS_RES = [
    re.compile(re.escape(_REDACTED)),
    re.compile(
        "|".join(_SECRET_LABELS),
        re.IGNORECASE,
    ),
    re.compile(r"[我的为有在和与了呢吗是：:，,、；;。！？\s]+"),
]

def _meaningful_after_redaction(text):
    """脱敏后的文本里还剩不剩有用的内容（去掉占位符、标签词、虚词后
    还有东西才算有）。纯秘密（“我的密码是 xxx”）脱敏后什么都不剩，
    返回 False，调用方直接不存；混合内容（“喜欢喝咖啡，密码是 xxx”）
    还剩“喜欢喝咖啡”，返回 True，调用方存脱敏后的版本。"""
    if not isinstance(text, str):
        return False
    rest = text
    for pattern in _MEANINGLESS_RES:
        rest = pattern.sub("", rest)
    return bool(rest.strip())

def _normalize_memory_text(text):
    """把一条记忆文本规整成可比的形状（供直接存入的去重用）：
    全角转半角（Ａ→A）、去所有空白、去末尾句末标点、英文转小写。
    “我每天晚上都读半小时书。”和“我每天晚上都读半小时书”
    （只差一个句号）算同一条，不重复存。"""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip("。！？；，、.!?;,")
    return normalized.lower()

def _find_duplicate_memory(text, existing_items):
    """在已有记忆里找和 text 相同（或只差标点/空白/大小写）的那条，
    找到返回那一项（{"id", "text"}），找不到返回 None。"""
    normalized = _normalize_memory_text(text)
    if not normalized:
        return None
    for item in existing_items or []:
        if _normalize_memory_text(item.get("text") or "") == normalized:
            return item
    return None

