# memory/intake_gate.py（四件套·实测问题三：入口当场拦＋当场判）
#
# 背景（2026-11-24 BOOS 实测＋追加决定七）：十二嘴上说「我就不存
# 了」，「突然想吃螺蛳粉」这类当下念头、问题句、碎片照样落进
# episodic 原始行——旧设计是「先落原始行、保证不晋升」，BOOS 已
# 拍板废除这条路：「别什么都记」要和十一一样，入口当场判断。
#
# 本模块只做判定，不读库、不写库、不联网（落痕/挂起由 store 做）：
#   classify_program(text, fact_check) —— 程序屏：三路裁决
#     store（像事实，放行走原有各道门）/ drop（明确垃圾，当场丢）/
#     hold（拿不准，挂起待判官/待问，绝不当场丢）。
#     判定件全部打底于既有规矩源，不另立一套：fact_check 就是
#     sqlite_store._looks_like_fact（由调用方传入，避免循环导入），
#     语气用 tone.local_tone_screen，槽位用 correction_text 的
#     _infer_fact_slot。
#   validate_mark / combine —— 当场判：对话当场的模型在收尾时给
#     候选打的标记（记/不记/拿不准＋一句话理由）只是提案，combine
#     把它和程序屏合并：程序判明确垃圾的，标记不许放行（门不绕）；
#     程序判是事实、标记说不记的，按护栏不许直接丢——降为挂起；
#     程序拿不准、标记说记的，放行走原门。
#   DictMarker —— 打桩标记器（测试/影子双跑用，不调真模型）。
#   shadow_compare —— 影子双跑对比桩：同一批带标注的用例，程序
#     单判 vs 程序＋当场判各跑一遍，数误判（该扔的记下了、该记的
#     扔了或误挂起了），每条留明细和理由，可追溯。

import re

from memory.correction_text import _infer_fact_slot
from memory.tone import local_tone_screen

VERDICT_STORE = "store"
VERDICT_DROP = "drop"
VERDICT_HOLD = "hold"
VERDICTS = (VERDICT_STORE, VERDICT_DROP, VERDICT_HOLD)

MARK_KEEP = "记"
MARK_DROP = "不记"
MARK_UNSURE = "拿不准"
MARKS = (MARK_KEEP, MARK_DROP, MARK_UNSURE)

CAT_QUESTION = "question"
CAT_HYPOTHESIS = "hypothesis"
CAT_CONFIRMATION = "confirmation"
CAT_FRAGMENT = "fragment"
CAT_TRANSIENT = "transient_want"
CAT_JOKE = "joke"
CAT_TEST = "test"
CAT_SMALLTALK = "smalltalk"
CAT_INSTRUCTION = "instruction"
CAT_OTHER = "other"

# _looks_like_fact 的拒绝原因 → 入口类别（同一个规矩源，不另立）。
_FACT_REASON_CATEGORY = {
    "问句": CAT_QUESTION,
    "假设句": CAT_HYPOTHESIS,
    "当下的想法": CAT_TRANSIENT,
    "半句话": CAT_FRAGMENT,
    "太短": CAT_FRAGMENT,
    "闲聊": CAT_SMALLTALK,
    "确认词": CAT_CONFIRMATION,
    "删除指令": CAT_INSTRUCTION,
    "撤回信号": CAT_INSTRUCTION,
    "纠错指令": CAT_INSTRUCTION,
}

# 光秃秃的确认词/应答词：整句就这点东西，没有事实内容。
_BARE_CONFIRMATIONS = {
    "好", "好的", "行", "嗯", "对", "是", "是的", "对的", "没错",
    "可以", "知道了", "嗯嗯", "好吧", "谢谢", "没事", "不用了",
}

# 未来/计划/习惯标记：带这些的「想 X」是安排或长期意向，不是当下
# 念头（和 _looks_like_fact 当下想法规则的豁免名单同源）。
_FUTURE_OR_PLAN_MARKERS = (
    "以后", "明天", "后天", "下周", "下个月", "打算", "决定", "计划",
    "从今天起", "每天", "每周", "每月", "定期", "习惯",
)

# 当下冲动标记：带这些的「想吃/想喝」是这一会儿的念头（「突然好想
# 吃螺蛳粉」），明确不当偏好记。
_SPUR_MARKERS = ("突然", "好想", "这会儿", "现在就想", "一下子想")

# 当下念头动词（吃喝买玩看睡）：「想/馋」紧跟这些、无当下标记又无
# 未来标记时，是表态式念头——旧路先落行，新路按决定七挂起待裁。
_WANT_RE = re.compile(r"(想|馋)(吃|喝|买|玩|看|睡)")
# 冲动式：再带上「去」（突然想去/好想去是明确的当下冲动）。
_WANT_SPUR_RE = re.compile(r"(想|馋)(吃|喝|买|玩|看|睡|去)")

# 指代半句：只有指示词、没有实义内容的（「就那个吧」）。
_FRAGMENT_ONLY_RE = re.compile(
    r"^(就|就是|那|那个|这|这个|那家|这家|那条|这条)[^，。！？、；：]{0,3}$"
)

# 语气自明短语（说话人自己点明是玩笑/测试的）：命中这些算明确，
# 当场丢；只命中裸关键词（「测试」「玩笑」单字）的拿不准，挂起。
_STRONG_TONE_PHRASES = {
    "这是测试", "这句是测试", "测试一下", "开玩笑", "闹着玩", "假设",
}


def _has_future_or_plan(text):
    return any(m in text for m in _FUTURE_OR_PLAN_MARKERS)


def classify_program(text, fact_check):
    """程序屏：给一条候选记忆文本三路裁决。

    fact_check 是 sqlite_store._looks_like_fact（调用方传入）：
    它以确定性规则拒绝的那几类（问句/假设句/确认词/半句话/当下
    念头/指令句……）在本口径里就是「明确垃圾」，当场丢。
    返回 (verdict, category, reason)。
    """
    s = (text or "").strip()
    if not s:
        return VERDICT_DROP, CAT_FRAGMENT, "空内容，没有可记的东西"
    core = s.rstrip("。！？!?~～…、，,；;：:").strip()
    if core in _BARE_CONFIRMATIONS:
        return VERDICT_DROP, CAT_CONFIRMATION, "光秃秃的确认词，没有事实内容"
    ok, fact_reason = fact_check(s)
    if not ok:
        return (
            VERDICT_DROP,
            _FACT_REASON_CATEGORY.get(fact_reason, CAT_OTHER),
            "程序硬规则判为%s" % (fact_reason or "不像事实"),
        )
    # 语气筛（tone 既有判定件）：自己点明玩笑/测试/假设的当场丢；
    # 只沾裸关键词的拿不准，挂起。
    tone, evidence = local_tone_screen(s)
    if tone in ("joke", "test", "hypothesis"):
        category = {
            "joke": CAT_JOKE, "test": CAT_TEST,
            "hypothesis": CAT_HYPOTHESIS,
        }[tone]
        if evidence in _STRONG_TONE_PHRASES:
            return (
                VERDICT_DROP, category,
                "说话人自己点明了语气（%s），不当事实记" % evidence,
            )
        return (
            VERDICT_HOLD, category,
            "带%s字样（%s），但没点明语气，拿不准，先挂起"
            % (category, evidence),
        )
    # 当下念头：带冲动标记（突然/好想……）的是明确的当下念头，
    # 当场丢；没标记的表态式念头（「我想吃螺蛳粉」）拿不准，挂起
    # ——旧路是先落 episodic 行，决定七废除这条路。
    if not _has_future_or_plan(s) and _infer_fact_slot(s)[0] is None:
        if any(m in s for m in _SPUR_MARKERS) and _WANT_SPUR_RE.search(s):
            return (
                VERDICT_DROP, CAT_TRANSIENT,
                "带当下冲动标记的念头（突然/好想一类），不是长期偏好",
            )
        if _WANT_RE.search(s):
            return (
                VERDICT_HOLD, CAT_TRANSIENT,
                "像当下念头、又没有当下标记，拿不准，先挂起待裁",
            )
    # 指代半句：只有指示词、没实义内容，意思不完整，拿不准挂起
    # （明确垃圾才丢，这类靠上下文才懂，不当场丢）。
    if _FRAGMENT_ONLY_RE.match(core):
        return (
            VERDICT_HOLD, CAT_FRAGMENT,
            "只有指示词的半句话，意思不完整，拿不准，先挂起待裁",
        )
    return VERDICT_STORE, CAT_OTHER, "程序屏看是事实句式，放行走原有门"


def validate_mark(raw):
    """校验一条当场判标记，合法返回归一化 dict，否则 None。

    标记格式：{"text": 候选原文, "mark": 记/不记/拿不准,
    "reason": 一句话理由}。格式不对、标记值不认识的按「没有标记」
    处理（回退程序屏），不猜。
    """
    if not isinstance(raw, dict):
        return None
    text = (raw.get("text") or "").strip()
    mark = (raw.get("mark") or "").strip()
    if not text or mark not in MARKS:
        return None
    return {
        "text": text,
        "mark": mark,
        "reason": (raw.get("reason") or "").strip(),
    }


def combine(program_verdict, program_reason, mark):
    """程序屏裁决＋当场判标记 → 最终裁决。

    合并规矩（契约三、状态变更入口「当场判机制口径」）：
    - 程序判 drop（明确垃圾）：标记不许放行，门不绕；
    - 程序判 store、标记「不记」：护栏——事实句式不许凭标记
      直接丢，降为 hold 挂起待 BOOS 裁；
    - 程序判 store、标记「拿不准」：hold；
    - 程序判 hold、标记「记」：放行 store（仍走原有各道门落账）；
    - 程序判 hold、标记「不记」：仍 hold——程序没判死是垃圾，
      只有一面之词不许丢，挂起让 BOOS 一并裁。
    返回 (verdict, reason, mark_or_None)。
    """
    if mark is None:
        return program_verdict, program_reason, None
    m = mark["mark"]
    if program_verdict == VERDICT_DROP:
        return VERDICT_DROP, program_reason, mark
    if program_verdict == VERDICT_STORE:
        if m == MARK_DROP:
            return (
                VERDICT_HOLD,
                "当场判标了「不记」（%s），但程序看是事实句式——"
                "护栏：不许凭标记直接丢，挂起待裁"
                % (mark.get("reason") or "没给理由"),
                mark,
            )
        if m == MARK_UNSURE:
            return (
                VERDICT_HOLD,
                "当场判拿不准（%s），挂起待裁"
                % (mark.get("reason") or "没给理由"),
                mark,
            )
        return VERDICT_STORE, program_reason, mark
    # program hold
    if m == MARK_KEEP:
        return (
            VERDICT_STORE,
            "程序拿不准，当场判标了「记」（%s），放行走原有门"
            % (mark.get("reason") or "没给理由"),
            mark,
        )
    return VERDICT_HOLD, program_reason, mark


class DictMarker:
    """打桩标记器：按原文查表给标记（测试/影子双跑用，不调模型）。

    mapping：{候选原文: "记"/"不记"/"拿不准" 或 (标记, 理由)}。
    表里没有的候选不标（等于当场判没意见，走程序屏）。
    """

    def __init__(self, mapping):
        self.mapping = dict(mapping or {})

    def __call__(self, candidates, messages=None):
        out = []
        for text in candidates or []:
            hit = self.mapping.get(text)
            if hit is None:
                # 容忍句尾标点差异：按剥标点后的原文再对一次。
                stripped = (text or "").strip().rstrip("。！？!?")
                for key, value in self.mapping.items():
                    if (key or "").strip().rstrip("。！？!?") == stripped:
                        hit = value
                        break
            if hit is None:
                continue
            if isinstance(hit, (tuple, list)):
                mark, reason = hit[0], (hit[1] if len(hit) > 1 else "")
            else:
                mark, reason = hit, "打桩标记"
            out.append({"text": text, "mark": mark, "reason": reason})
        return out


def _verdict_errors(expect, got):
    """一条用例的误判分类：返回错误键或 None。

    误判口径（契约「当场判验收用例」）：该扔的记下了（期望
    drop/hold、实际 store）、该记的扔了（期望 store、实际
    drop）、该记的误挂起（期望 store、实际 hold）。期望 drop、
    实际 hold 不算误判——挂起是安全侧，BOOS 答「不存」即可，
    单独计 held_instead_of_drop 供参考。
    """
    if expect == VERDICT_STORE and got == VERDICT_DROP:
        return "dropped_fact"
    if expect == VERDICT_STORE and got == VERDICT_HOLD:
        return "held_fact"
    if expect in (VERDICT_DROP, VERDICT_HOLD) and got == VERDICT_STORE:
        return "stored_junk"
    return None


def shadow_compare(cases, marker, fact_check):
    """影子双跑对比桩：同一批标注用例，程序单判 vs 程序＋当场判。

    cases：[{"text": ..., "expect": store/drop/hold}]；
    marker：标记器可调用对象（DictMarker 或同签名函数），
    None 表示当场判不可用（对照组就是程序单判本身）。
    返回 {"program": {...}, "marked": {...}, "details": [...]}：
    每组含 errors 总数、分项计数、held_instead_of_drop 参考数；
    details 逐条带程序裁决、标记、最终裁决与理由，可追溯。
    """
    def _run(use_marker):
        counts = {
            "errors": 0, "dropped_fact": 0, "held_fact": 0,
            "stored_junk": 0, "held_instead_of_drop": 0,
        }
        details = []
        texts = [c.get("text") or "" for c in cases]
        marks = {}
        if use_marker and marker is not None:
            try:
                raw_marks = marker(list(texts), None)
            except Exception:
                raw_marks = None
            for raw in raw_marks or []:
                valid = validate_mark(raw)
                if valid is not None:
                    marks[valid["text"]] = valid
        for case in cases:
            text = case.get("text") or ""
            expect = case.get("expect")
            pv, category, preason = classify_program(text, fact_check)
            mark = marks.get(text) if use_marker else None
            verdict, reason, _used = combine(pv, preason, mark)
            err = _verdict_errors(expect, verdict)
            if err:
                counts["errors"] += 1
                counts[err] += 1
            elif expect == VERDICT_DROP and verdict == VERDICT_HOLD:
                counts["held_instead_of_drop"] += 1
            details.append({
                "text": text, "expect": expect,
                "program_verdict": pv, "category": category,
                "mark": (mark or {}).get("mark"),
                "mark_reason": (mark or {}).get("reason"),
                "verdict": verdict, "reason": reason, "error": err,
            })
        return counts, details

    program_counts, program_details = _run(False)
    marked_counts, marked_details = _run(True)
    return {
        "program": program_counts,
        "marked": marked_counts,
        "details": marked_details,
        "program_details": program_details,
    }
