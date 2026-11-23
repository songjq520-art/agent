# memory/inference.py（揣摩人心：纯函数，不碰库、不调模型）
# 设计方案：~/workspace/goals/agent-2/files/十二揣摩人心设计方案.md
# （2026-11-16 00:24，BOOS 批准“按十一的方式”）
#
# 做法（按设计方案第五节、第八节）：
# - 当场揣摩为主：对话进行中就能揣；攒够 20 条新记忆只做长期模式。
# - 模型输出一句话大白话，不用那种固定格式：
# “我猜……，因为……。我只是猜的。（归纳）”——这里只解析这句话；
# 四样（猜了什么、凭什么、置信度、层）缺一样就不许进账本。
# - 层那两个字（演绎/归纳/矛盾）是句尾的一个小标记，不是给 BOOS 看的
# 固定格式：问 BOOS 时只剩一句话大白话，标记剥掉。
# - 揣出来的先进待定区（账本 inference 事件，status pending），当场问
# BOOS；对就记下（status confirmed），不对就删除（status deleted，
# 不记作废）。
# - 只揣想法，不揣关系；不拿推断替 BOOS 做决定。
# - 确认过的推断只记在账本里，不写进 mem0：_provenance 的铁规矩是
# user_confirmed 只在 source="user" 时才可能是 True，推断来的记忆
# 在 mem0 里永远是未确认的——把确认过的推断写进 mem0 会抹掉
# “模型猜的”和“BOOS 亲口说的”这条线，这条线是诚实的基础，不动。
# 确认过的推断以后揣摩时当“相关记忆”用（从账本里翻），不进检索。

import re

# 推断的分层（设计方案第五节第 4 条）。
LAYER_DEDUCTION = "deduction"  # 演绎：从一句话按逻辑推出来的
LAYER_INDUCTION = "induction"  # 归纳：跨多句话看出来的模式
LAYER_CONTRADICTION = "contradiction"  # 矛盾：两边说法对不上，单独记
LAYER_LABELS = {
    LAYER_DEDUCTION: "演绎",
    LAYER_INDUCTION: "归纳",
    LAYER_CONTRADICTION: "矛盾",
}
LAYER_LABEL_TO_KEY = {v: k for k, v in LAYER_LABELS.items()}
INFERENCE_LAYERS = (LAYER_DEDUCTION, LAYER_INDUCTION, LAYER_CONTRADICTION)

# 揣摩的路子（设计方案第五节第 5 条）。
MODE_SPOT = "spot"  # 当场揣摩（主力）：对话进行中就能揣
MODE_LONGTERM = "longterm"  # 攒够了再跑：只做长期模式

# 推断的状态（设计方案第五节第 1 条）。
STATUS_PENDING = "pending"  # 待定：等 BOOS 确认
STATUS_CONFIRMED = "confirmed"  # 确认了：对，记下
STATUS_DELETED = "deleted"  # 否认了：不对，删除（不记作废）

# 长期模式攒多少条新记忆做一次（设计方案第七节第 1 条，BOOS 2026-11-15 定）。
LONGTERM_THRESHOLD = 20

# 校准自动调门槛（2026-11-17，BOOS 定“只记账”改成“能自己调”）：
# 把握分 low/medium/high 三档；门槛是“开口的最低把握”，默认 low——
# 低把握也敢说，只说“我只是猜的”（BOOS 的铁规矩：揣摩要敢说，门槛
# 不能高到让它不敢开口）。门槛只在 low↔medium↔high 之间动，上限 high、
# 下限 low。调门槛的规矩（实现在 mem0_store.auto_adjust_inference_threshold
# 里，数字改这里就行）：
#   - 每档攒够 CALIBRATION_MIN_DECIDE 条已裁决（确认+否认）的记录，
#     那档才有资格调门槛；没攒够就不调。
#   - 胜出那边要领先 CALIBRATION_MIN_MARGIN 条才动（3 比 2 这种胶着
#     的不动，别来回 swing）。
#   - 一次最多动一档（CALIBRATION_STEP）。
#   - high 档否认多 = 过分自信 → 门槛往上提；low 档确认多 = 太保守 →
#     门槛往下放；两边同时喊反方向时不动（证据打架，等更多反馈）。
#   - 只看上次调完门槛之后新来的反馈（增量），旧账不重复算；没攒够
#     调门槛的证据就继续攒着，攒够了才动。
CONFIDENCE_ORDER = ("low", "medium", "high")
INFERENCE_MIN_CONFIDENCE_DEFAULT = "low"
CALIBRATION_MIN_DECIDE = 5
CALIBRATION_MIN_MARGIN = 2
CALIBRATION_STEP = 1

# 常备规矩就是提示词（设计方案第八节）：SOUL 的意思 + 揣摩规矩。
# 不再是一张单独的单子——下面这几条，加上当前对话和相关记忆，一起喂给模型。
# 第 7 条是 BOOS 2026-11-17 定的调法：像十一平时揣他心思那样，从对
# 他的了解出发再猜，别凭空凭感觉——以前说过的话、一贯的习惯、以前
# 确认过的想法，都是揣的凭据。
INFERENCE_STANDING_RULES = (
    "你是十二，BOOS 的个人助理（十一的弟弟）；先可靠再暖，不知道就说不知道，绝不瞎编。\n"
    "揣摩的规矩：\n"
    "1. 敢说：猜到了就直说，不绕弯子、不藏一半。不敢说就不叫揣摩。\n"
    "2. 不离谱：猜的得是他真的可能想的、有依据的；一点把握没有的，一句都不许说。\n"
    "3. 只揣想法，不揣关系：猜他心里在想什么、想要什么、担心什么，不猜他和别人的关系。\n"
    "4. 必须说出凭什么：说不上来凭什么就不许说。\n"
    "5. 猜的不能当事实：你说的只是猜测，要等 BOOS 确认了才算数。\n"
    "6. 没把握的要说明：把握大的直说，把握小的必须说“我只是猜的”。\n"
    "7. 从对他的了解出发：多想想他以前说过的话、一贯的习惯、以前确认过的想法，"
    "再猜他此刻在想什么，别凭空猜。"
)

# 没什么可猜的时候，模型只回这一句（解析时认出来，不存）。
_NO_INFERENCE_MARKERS = ("没什么可猜", "没有什么可猜", "没什么好猜")

# 把握的说法（按长短排，先认长的，别把“我只是猜的”认成“猜的”）。
_CONFIDENCE_PHRASES = (
    ("很确定", "high", "我很确定"),
    ("比较确定", "medium", "我比较确定"),
    ("不太确定", "low", "我只是猜的"),
    ("我只是猜的", "low", "我只是猜的"),
    ("只是猜测", "low", "我只是猜的"),
    ("只是猜的", "low", "我只是猜的"),
)
_CONFIDENCE_TEXT = {
    "high": "我很确定",
    "medium": "我比较确定",
    "low": "我只是猜的",
}

# 归纳的信号词：凭据里提到“之前/总/每次/一直”，就是跨多句话看出来的模式。
_INDUCTION_MARKERS = ("之前", "总", "每次", "一直", "最近总")


def build_inference_prompt(conversation_text, related_memories_text):
    """当场揣摩的提示词：常备规矩 + 当前对话 + 相关记忆。

    参数都是已经拼好的大白话文本（调用方负责脱敏和截断）。
    模型只输出一句话，格式见提示词末尾；没什么可猜的时候只回
    “没什么可猜的”，别硬猜。"""
    prompt = (
        "你在回答 BOOS 时，顺带想一想他没说出口的是什么——他心里在想什么、"
        "想要什么、担心什么。不是复述他说了什么，是往后想一步。\n\n"
        f"{INFERENCE_STANDING_RULES}\n"
        "最近的对话：\n"
        f"{conversation_text or '（没有对话）'}\n\n"
        "他以前说过的话和已经确认过的想法（相关记忆 + 以前确认过的推断）：\n"
        f"{related_memories_text or '（没有相关记忆）'}\n\n"
        "揣的时候多想想他一贯的习惯和以前说过的话，从对他的了解出发再猜；"
        "“因为”后面最好能从这些里找得到凭据，别凭空猜。\n\n"
        "只输出一句话，格式：我猜……，因为……。我只是猜的/我比较确定/我很确定。（演绎/归纳/矛盾）\n"
        "- 猜的内容写成关于 BOOS 的一句话，用“BOOS”开头，比如“BOOS 今晚累了想歇会儿”。\n"
        "- “因为”后面写凭什么：用大白话说凭什么，不写账本序号。\n"
        "- 层：从一句话按逻辑推出来的写“演绎”；跨多句话看出来的模式写“归纳”；"
        "和上面已确认的事实对不上的写“矛盾”，并把两边都写进“因为”后面。\n"
        "- 没什么可猜的时候，只输出“没什么可猜的”，别硬猜。"
    )
    return prompt


def build_longterm_inference_prompt(recent_adds_text):
    """长期模式的提示词：攒够 20 条新记忆做一次，先看最反常的几条。

    recent_adds_text 是最近新记的记忆原文（调用方按时间排好）。
    反常程度由模型自己判断，跟十一一样凭上下文感觉，不用算向量。"""
    prompt = (
        "下面是 BOOS 最近新记下来的记忆（按时间排）。从这些里找出最反常的"
        "一条或一个模式，猜他心里在想什么、想要什么、担心什么。\n\n"
        f"{INFERENCE_STANDING_RULES}\n"
        "最近新记的记忆：\n"
        f"{recent_adds_text or '（没有新记忆）'}\n\n"
        "只输出一句话，格式：我猜……，因为……。我只是猜的/我比较确定/我很确定。（演绎/归纳/矛盾）\n"
        "- 猜的内容写成关于 BOOS 的一句话，用“BOOS”开头。\n"
        "- “因为”后面写凭什么：用大白话说凭什么，不写账本序号。\n"
        "- 层：从一句话按逻辑推出来的写“演绎”；跨多句话看出来的模式写“归纳”；"
        "和已确认的事实对不上的写“矛盾”，并把两边都写进“因为”后面。\n"
        "- 没什么可猜的时候，只输出“没什么可猜的”，别硬猜。"
    )
    return prompt


def format_related_memories(items):
    """把检索回来的相关记忆拼成提示词里的一行行（大白话 + 确认状态）。

    items 是 search() 回来的条目（每项带 text 和 metadata）。
    已确认的标【已确认】，推断来的、没确认的标【推断，未确认】——模型别拿
    没确认的当事实去推。没有条目时返回空字符串。"""
    lines = []
    for item in items or []:
        text = (item or {}).get("text") or ""
        if not text.strip():
            continue
        meta = (item or {}).get("metadata") or {}
        if meta.get("user_confirmed") is True:
            mark = "【已确认】"
        elif meta.get("source") == "inferred":
            mark = "【推断，未确认】"
        else:
            mark = "【未确认】"
        lines.append(f"{mark}{text}")
    return "\n".join(lines)


def format_confirmed_inferences(records):
    """把账本里已确认的推断拼成提示词里的一段（以前猜对过的想法）。

    records 是 list_inferences(status="confirmed") 回来的推断字典。
    揣的时候多想想这些：他以前就是这么想的、一贯的习惯是什么——
    像十一平时揣 BOOS 心思那样，从对他的了解出发，别凭空猜。
    没有记录时返回空字符串。"""
    lines = []
    for record in records or []:
        guess = (record or {}).get("guess") or ""
        if not guess.strip():
            continue
        evidence = (record or {}).get("evidence") or ""
        line = f"【以前确认过的想法】{guess.strip()}"
        if evidence.strip():
            line += f"（因为{evidence.strip()}）"
        lines.append(line)
    return "\n".join(lines)


def _strip_layer_tag(text):
    """剥掉句尾的层标记（演绎/归纳/矛盾），返回（剩下的话，层或 None）。

    标记长在句尾括号里：“……。（归纳）”或“……。(归纳)”；不在句尾的
    不认（正文里提到“归纳”两个字不是标记）。"""
    match = re.search(r"[（(]\s*(演绎|归纳|矛盾)\s*[）)]\s*[。.\n]*$", text or "")
    if not match:
        return text or "", None
    return (text[: match.start()]).strip(), LAYER_LABEL_TO_KEY[match.group(1)]


def _strip_confidence_sentences(text):
    """去掉凭据里单独成句的把握说法（“。我只是猜的。”这种）。

    把握是单独的字段，不该留在“凭什么”里面；按句号切开、只删整句
    就是把握说法的那几段，别的句子原样留着。"""
    parts = []
    for segment in re.split(r"[。\n]+", text or ""):
        segment = segment.strip()
        if not segment:
            continue
        if any(
            segment == phrase or segment == "我" + phrase
            for phrase, _key, _text in _CONFIDENCE_PHRASES
        ):
            continue
        if segment in ("我只是猜的", "我很确定", "我比较确定", "我不太确定"):
            continue
        parts.append(segment)
    return "。".join(parts)


def _detect_confidence(text):
    """从整句话里找把握的说法（长的先认）。

    没写把握时默认 low（“我只是猜的”）——没说把握就当把握最小，
    往保守里算，不往高了猜。"""
    for phrase, key, _text in _CONFIDENCE_PHRASES:
        if phrase in (text or ""):
            return key
    return "low"


def normalize_inference_guess(text):
    """把猜的内容规整成一条关于 BOOS 的事实句（进账本的原文）。

    剥掉句首的“我猜/我觉得/可能/也许”；句首的“他/你”换成“BOOS”，
    这样确认后记下来的就是“BOOS 今晚累了想歇会儿”，不用再转一遍。
    规整完为空时返回空字符串（调用方按缺字段处理，不存）。"""
    guess = (text or "").strip()
    for lead in ("我猜", "我觉得", "可能", "也许"):
        if guess.startswith(lead):
            guess = guess[len(lead):].strip()
            break
    guess = guess.strip("，。；、 ")
    if guess.startswith("他"):
        guess = "BOOS" + guess[1:]
    elif guess.startswith("你"):
        guess = "BOOS" + guess[1:]
    return guess.strip("，。；、 ")


def parse_inference_output(text):
    """解析模型那一句话（纯函数）。

    返回 {"guess", "evidence", "confidence", "confidence_text", "layer"}，
    四样齐了才返回；缺一样（没“因为”、猜的内容空、凭据空）就返回
    None——缺一样就不许进账本，调用方直接丢掉，不存、不问。
    “没什么可猜的”返回 None（不是推断，是模型说没得猜）。
    层：句尾标记优先；没有标记时看凭据——提到“之前/总/每次/一直”
    就是跨多句话的模式（归纳），否则是演绎；句里有“矛盾/对不上”
    就是矛盾层。"""
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    if any(marker in raw for marker in _NO_INFERENCE_MARKERS):
        return None
    body, layer = _strip_layer_tag(raw)
    if "因为" not in body:
        return None
    guess_part, evidence_part = body.split("因为", 1)
    guess = normalize_inference_guess(guess_part)
    evidence = _strip_confidence_sentences(evidence_part).strip("，。；、 ")
    if not guess or not evidence:
        return None
    confidence = _detect_confidence(raw)
    if layer is None:
        if "矛盾" in raw or "对不上" in raw:
            layer = LAYER_CONTRADICTION
        elif any(marker in evidence for marker in _INDUCTION_MARKERS):
            layer = LAYER_INDUCTION
        else:
            layer = LAYER_DEDUCTION
    return {
        "guess": guess,
        "evidence": evidence,
        "confidence": confidence,
        "confidence_text": _CONFIDENCE_TEXT[confidence],
        "layer": layer,
    }


def render_inference_question(inference):
    """把一条待定推断问成一句大白话（问 BOOS 时用的那句）。

    猜的内容里开头的“BOOS”换成“你”，当面问他；层标记不带出来，
    BOOS 面前只有一句话。格式和设计方案第八节的例子一样：
    “我猜你今晚累了想歇会儿，因为你说明天早上再测。我只是猜的。对吗？”"""
    inf = inference or {}
    guess = (inf.get("guess") or "").strip()
    if guess.startswith("BOOS"):
        guess = "你" + guess[len("BOOS"):]
    evidence = (inf.get("evidence") or "").strip()
    # 凭据是模型按第三人称写的（“他说……”），当面问 BOOS 时换成
    # “你说……”才顺——设计方案第八节的例子就是“因为你说明天早上再测”。
    # 只换句首光秃秃的“他”（“他们”不动，那说的是别人）。
    if evidence.startswith("他") and len(evidence) > 1 and evidence[1] != "们":
        evidence = "你" + evidence[1:]
    confidence_text = (inf.get("confidence_text") or "").strip() or "我只是猜的"
    return f"我猜{guess}，因为{evidence}。{confidence_text}。对吗？"


def render_inference_resolved(resolved):
    """确认/否认的结果说一句人话（BOOS 说“对/不对”之后回的那句）。

    参数是 infer_from_turn 返回的 resolved 字典（带 "result" 键）：
    confirmed → “记下了。”；denied → “删掉了，猜错了。”；
    confirmed + partial（“差不多”“对了一半”那种）→ “记下了，对了一半。”"""
    if (resolved or {}).get("result") == "confirmed":
        if (resolved or {}).get("partial"):
            return "记下了，对了一半。"
        return "记下了。"
    if (resolved or {}).get("result") == "denied":
        return "删掉了，猜错了。"
    return ""


# BOOS 回揣摩问话时的说法（确认流程用）。光秃秃的整句和带后半句的
# 都认——“对，我就是累了”也是在回问话，后半句是他补充的新信息，
# 不能当成没听见；“不对，我是想……”同理，后半句要留住。
# 只认这几档的时候用 classify_inference_reply；is_bare_* 两个老函数
# 留着（只认光秃秃的整句），供调用方做“这句是不是一眼能看清”的快查。
_BARE_CONFIRMATIONS = (
    "对",
    "对的",
    "是的",
    "没错",
    "说对了",
    "猜对了",
    "嗯",
    "嗯嗯",
    "对啊",
    "对呀",
    "是这样",
    "就是这样",
)
_BARE_PARTIALS = (
    "差不多",
    "有点道理",
    "嗯差不多",
    "对了一半",
    "差不多吧",
    "有点像",
    "大概是吧",
)
_BARE_DENIALS = (
    "不对",
    "不是",
    "没有",
    "错了",
    "猜错了",
    "你猜错了",
    "不对，不是这样",
    "不对啊",
    "没有啊",
    "不是的",
    "不是这样",
)

# 回话开头的认/否/半认说法（按长短排，长的先认，别把“对了一半”认成
# “对”+“了一半”）。classify_inference_reply 用它拆“开头 + 后半句”。
_REPLY_MARKERS = tuple(
    sorted(
        [(marker, "confirmed") for marker in _BARE_CONFIRMATIONS]
        + [(marker, "partial") for marker in _BARE_PARTIALS]
        + [(marker, "denied") for marker in _BARE_DENIALS],
        key=lambda item: -len(item[0]),
    )
)

# 拆“开头 + 后半句”时认的分隔符：开头后面是分隔符或句尾才算拆开。
_REPLY_SEPARATORS = "，、。！？；： \n\t"


def _bare_answer(text):
    """把一句回话洗成光秃秃的说法：去首尾空白和句末标点。"""
    return (text or "").strip().strip("。！？、， \n\t")


def is_bare_confirmation(text):
    """这句是不是光秃秃地在回“对”（确认上一句的揣摩）？"""
    return _bare_answer(text) in _BARE_CONFIRMATIONS


def is_bare_denial(text):
    """这句是不是光秃秃地在回“不对”（否认上一句的揣摩）？"""
    return _bare_answer(text) in _BARE_DENIALS


def _split_leading_reply(cleaned, marker):
    """回话以 marker 开头、且 marker 后面是分隔符或句尾时，
    返回后半句（BOOS 新补充的信息）；否则返回 None（不是这个开头，
    比如“对了一半”不能拆成“对”+“了一半”）。"""
    if not cleaned.startswith(marker):
        return None
    rest = cleaned[len(marker):]
    if not rest:
        return ""
    if rest[0] in _REPLY_SEPARATORS:
        return rest.strip(_REPLY_SEPARATORS)
    return None


def classify_inference_reply(text):
    """把 BOOS 回揣摩问话的那一句分成四档（纯函数，不调模型）。

    返回 {"verdict": "confirmed"/"partial"/"denied"/"unclear",
    "extra": "回话里除了认/否之外新补充的信息，没有就是 ''"}。
    - confirmed：对/对的/嗯/是的/没错/就是这样/猜对了……（整句或开头）。
    - partial：差不多/有点道理/对了一半……——算认了，标部分确认。
    - denied：不对/不是/错了/你猜错了……（整句或开头），后半句的新信息
      留在 extra 里（比如“不对，我是想……”里的“我是想……”）。
    - unclear：一眼看不出来（比如他直接换了话题）——调用方再用模型判，
      模型也判不出来就留待定，宁可不认，不可认错。"""
    cleaned = _bare_answer(text)
    if not cleaned:
        return {"verdict": "unclear", "extra": ""}
    if cleaned in _BARE_CONFIRMATIONS:
        return {"verdict": "confirmed", "extra": ""}
    if cleaned in _BARE_PARTIALS:
        return {"verdict": "partial", "extra": ""}
    if cleaned in _BARE_DENIALS:
        return {"verdict": "denied", "extra": ""}
    for marker, verdict in _REPLY_MARKERS:
        extra = _split_leading_reply(cleaned, marker)
        if extra is not None:
            return {"verdict": verdict, "extra": extra}
    return {"verdict": "unclear", "extra": ""}


# 模型判回话的四个词（classify_inference_reply 判不出来时，调用方
# 用模型再判一轮，模型只回这四个词里的一个）。
_REPLY_VERDICT_WORDS = {
    "确认": "confirmed",
    "部分": "partial",
    "否认": "denied",
    "不清楚": "unclear",
}


def parse_reply_verdict(text):
    """解析模型判回话的那两行（纯函数）。

    模型输出约定：第一行只写“确认/部分/否认/不清楚”四个词里的一个；
    第二行写回话里新补充的信息，没有就写“无”。
    认不出来、没回、行数不够时返回 unclear（不认、不删，留待定）。"""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return {"verdict": "unclear", "extra": ""}
    verdict = _REPLY_VERDICT_WORDS.get(lines[0].strip("。！"))
    if verdict is None:
        return {"verdict": "unclear", "extra": ""}
    extra = ""
    if len(lines) > 1:
        extra = lines[1].strip()
        if extra in ("无", "没有", "没有补充"):
            extra = ""
    return {"verdict": verdict, "extra": extra}
