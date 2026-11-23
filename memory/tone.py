# memory/tone.py（补丁 3：两段式判语气——本地筛 + 大模型判）
# 背景：十二记不住“玩笑话、测试句”——一句玩笑被当真记下来，
# 出现 3 次就晋升成稳定事实，还会盖掉真的记忆。补丁 3 是两段式：
# 本地先筛明显的，大模型后台判剩下的；拿不准的一律算认真。
# 这里只做本地那段 + 判定接口（大模型调用由别人接，不用这里写）。
# 下面全是纯函数（只算、不读库、不写库、不联网），只用标准库。
# 语气值和 provenance.py 的约定一致：只认 serious / joke / test /
# hypothesis 四个值；tone_evidence 是证据，tone_source 是 local /
# model，voided 是作废标记（见 provenance.py 里 _provenance 的注释）。

import json

# 语气四类：serious（认真）/ joke（玩笑）/ test（测试）/
# hypothesis（假设）。没打标（None）按认真算（见 decay.py
# should_promote 和 provenance.py _normalize_metadata）。
TONE_SERIOUS = "serious"
TONE_JOKE = "joke"
TONE_TEST = "test"
TONE_HYPOTHESIS = "hypothesis"
TONE_VALUES = (TONE_SERIOUS, TONE_JOKE, TONE_TEST, TONE_HYPOTHESIS)

# 本地筛的短语表：按优先级分两轮查——先看明确说法，再看关键词。
# 明确说法（说话人自己点明了语气）：
_EXPLICIT_PHRASES = (
    # 测试：先查 test，再查 joke，最后查 hypothesis。
    (TONE_TEST, ("这是测试", "这句是测试", "测试一下")),
    (TONE_JOKE, ("开玩笑", "玩笑", "闹着玩")),
    (TONE_HYPOTHESIS, ("假设",)),
)

# 关键词（没明说，但词本身就带语气）：
_KEYWORD_PHRASES = (
    (TONE_TEST, ("测试",)),
    (TONE_JOKE, ("开玩笑", "玩笑")),
    (TONE_HYPOTHESIS, ("假设", "假如", "如果", "比如说", "打个比方")),
)


def local_tone_screen(text):
    """本地筛：只对单条记忆文本做判断，不猜上下文。

    返回 (tone, evidence)：tone 是四个语气值之一，evidence 是命中
    到的原样短语（比如“这是测试”）；没命中返回 (None, None)——
    没命中不是“认真”，是“本地筛不出来”，交给大模型再判（拿不准
    的最后按认真算，见 tone_judgment_prompts）。

    优先级：先看明确说法（test → joke → hypothesis），再看关键词
    （test → joke → hypothesis）。明确说法优先于关键词：比如一句
    话里既有“这是测试”又有“如果”，先判 test。
    """
    if not isinstance(text, str) or not text:
        return None, None
    # 第一轮：明确说法。
    for tone, phrases in _EXPLICIT_PHRASES:
        for phrase in phrases:
            if phrase in text:
                return tone, phrase
    # 第二轮：关键词。
    for tone, phrases in _KEYWORD_PHRASES:
        for phrase in phrases:
            if phrase in text:
                return tone, phrase
    return None, None


def tone_judgment_prompts(text):
    """给大模型的判语气提示词，返回 (system_prompt, user_prompt)。

    大模型把一段记忆文本判成语气四类之一，只回 JSON
    {"tone": ..., "evidence": ...}，evidence 写是哪句话看出来的。
    拿不准时判 serious（认真），不要猜成玩笑——猜错成玩笑会把真
    记忆作废，猜错成认真最多是多记一条，宁可后者。
    """
    system_prompt = (
        "你是一名语气分类器，判断一段记忆文本的语气。"
        "语气只能是以下四类之一："
        "serious（认真，BOOS 真这么想/真这么说的）、"
        "joke（玩笑，开玩笑、闹着玩）、"
        "test（测试，BOOS 在测试系统、不是真话）、"
        "hypothesis（假设，假设/假如/如果，只是个设想）。"
        "只返回一个 JSON 对象，不要返回其他任何文字："
        '{"tone": ..., "evidence": ...}。'
        "其中 tone 是上面四类之一，evidence 是原文里让你这么判的那句话"
        "（原样抄，不要改写，必须非空）。"
        "拿不准时判 serious（认真），不要猜成玩笑、测试或假设。"
    )
    user_prompt = (
        "请判断下面这段记忆文本的语气，只返回 JSON "
        '{"tone": ..., "evidence": ...}：\n'
        f"{text}"
    )
    return system_prompt, user_prompt


def parse_tone_judgment(payload):
    """校验大模型回来的判定，返回 (tone, evidence)。

    payload 可以是 dict，也可以是 JSON 字符串（含 ```json 代码块
    也认）。校验不过就返回 (None, None)，不猜：
    - 解析失败（坏 JSON、不是 dict、不是 dict/字符串）；
    - tone 不在 serious / joke / test / hypothesis 四个值里；
    - evidence 不是非空字符串（空串、只有空白、不是字符串都不行）。
    """
    data = payload
    if isinstance(payload, str):
        raw = payload.strip()
        # 代码块：```json {...} ``` 或 ``` {...} ```，先把围栏去掉。
        if raw.startswith("```"):
            lines = raw.splitlines()
            # 第一行是围栏（``` 或 ```json），最后一行是收尾的 ```。
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return None, None
    if not isinstance(data, dict):
        return None, None
    tone = data.get("tone")
    evidence = data.get("evidence")
    if tone not in TONE_VALUES:
        return None, None
    if not isinstance(evidence, str) or not evidence.strip():
        return None, None
    return tone, evidence


def counts_for_promotion(tone):
    """这条语气的记忆算不算进晋升次数。

    joke / test / hypothesis 不算（玩笑、测试、假设说 3 遍也不晋升
    成稳定事实，见 decay.py should_promote）；serious / 没打标
    （None）算。别的非法值也按认真算——不认识的语气不挡晋升，
    挡晋升的只有上面明确的三类。
    """
    if tone in (TONE_JOKE, TONE_TEST, TONE_HYPOTHESIS):
        return False
    return True


def can_override_confirmed(tone):
    """这条语气的记忆能不能盖掉 BOOS 确认过的记忆。

    joke / test / hypothesis 盖不掉（玩笑话不能顶掉真记忆）；
    其余（serious / 没打标 None）可以。别的非法值也按可以算，
    和 counts_for_promotion 一个规矩：只有明确的三类才拦。
    """
    if tone in (TONE_JOKE, TONE_TEST, TONE_HYPOTHESIS):
        return False
    return True
