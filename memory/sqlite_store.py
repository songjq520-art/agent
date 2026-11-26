# memory/sqlite_store.py（SQLite 记忆索引，第二步，2026-11-21）
#
# 设计（BOOS 2026-11-20 定，2026-11-21 网页测后改走这条路）：
#   - JSONL 账本（memory_events.jsonl）仍是唯一真相源：先写账本、再写索引。
#   - SQLite（WAL）只做仓库存取：存原文 + metadata + 向量，不替 BOOS 做决定。
#   - mem0/Chroma 不再参与这条路的存取；Mem0Store 的外观（search/save_turn/
#     flush_pending/get_all/add_memory_text/delete_memory）照旧，调用方不用改。
#
# 为什么要有这个模块（病根，见 ~/workspace/十二网页测-2026-11-21.md）：
#   mem0 那条路里，用户亲口说的“一顿饭”要过 DeepSeek 提取 + 价值关卡，
#   提取提示词第 9 条（“只存有长期价值的，一次性小事不值得存，有疑问就不存”）
#   会把饺子、豆浆、豆角判成不值得存，提取返回空列表；而提交那层把“空手回来”
#   当成成功：指纹照样记成已处理、outbox 文件照样删、不留回执、不报错。
#   事实就这样悄悄丢了，日志里什么都不剩。
#   所以这条路反过来：用户亲口说的事实先原样落库（账本 + SQLite），
#   提取/打标只做归类，不再有一票否决不让存的权力；每次提交留回执。
#
# 今天第一部分只做到：说一句新的事实，关对话，新开对话问它，能记住
# （账本和 SQLite 里都有）。还没做的（后面几步再做）：
#   - 纠错 correct_memory（现在只返回明确的 not_yet，不悄悄猜）
#   - 衰减/晋升/沉底（decay.py 的规则还没接进来，权重先恒为 1.0）
#   - 类型权重（稳定偏好 > 一次性事件）已接进检索排序（2026-11-21 Q3
#     修法，见 _type_weight；只管排位、不管 0.4 资格线）
#   - 图谱边、精选清单（检索三路里的图谱路还是空路，精选先查还没接；
#     BM25 + RRF 融合已接，见 _search_inner；衰减权重仍恒为 1.0，只留了接口）
#   - 撤回目前只认光秃秃的“不不不/算了”这类整句信号，带后半句的撤回
#     （要模型判意图）还没接，用时会先存下来、再由 BOOS 删，不悄悄丢。

import copy
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from memory import core_cap as _core_cap
from memory import auth_guard as _auth_guard
from memory import session_auth as _session_auth
from memory import intake_gate as _intake
from memory import reanchor as _reanchor
from memory import persona_shell as _persona_shell
from memory import decay as _decay
from memory import graph as _graph_mod
from memory import ledger as _ledger
from memory.graph_entities import (
    extract_entities_free as _extract_entities_free,
    normalize_entity_name as _normalize_entity_name,
)
from memory import pending_questions as _pq
from memory import provenance as _provenance
from memory.correction_text import (
    _CORRECTION_PHRASES,
    _detect_correction, _strip_correction_sentences,
    _detect_correction_fragment, _VAGUE_FRAGMENT_VALUES,
    _infer_fact_slot, _correction_topic_slot,
    distill_correction_fact, is_correction_meta_sentence,
)
from memory.money_slots import extract_core_slots
from memory.redaction import _contains_sensitive, _normalize_memory_text
from memory.timeutil import _now_epoch, _now_iso, _next_write_seq, _seed_write_seq
from memory.tone import (
    local_tone_screen, parse_tone_judgment, tone_judgment_prompts,
)

logger = logging.getLogger(__name__)

# 相对日期换算用的基准时区：BOOS 的参考时区（MEMORY.md）。
_SHANGHAI = timezone(timedelta(hours=8))

# 向量模型：和 mem0 那条路同一个本地模型，离线加载（启动脚本已设
# HF_HUB_OFFLINE=1 / HF_ENDPOINT=https://hf-mirror.com）。
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# 语义检索的入围线：和 mem0 那条路同一个 0.4（BOOS 定的，留着不动）。
SEARCH_THRESHOLD = 0.4

# 撤回信号：光秃秃的一整句（Plan C 快捷路同款意思，简化版）。
_BARE_RETRACTION = {
    "不", "不不", "不不不", "不是", "没有", "算了", "别", "不要",
    "不对", "错了", "撤回", "取消", "当我没说",
}

# 稳定偏好在排序时的加权倍数（Q3 修法，2026-11-21，见 _type_weight）：
# 纠正行 RRF 0.0152、米线 0.0154，一位之差——1.25 倍足够把偏好行
# 顶回前 5，又小到不会把事件问（“昨天吃了什么”里饺子 RRF 约 0.033，
# 语义 + BM25 两路）里的偏好行顶到第一。
_TYPE_WEIGHT_STABLE_PREFERENCE = 1.25

# 已确认记忆在检索排序里只当平局裁决，不做乘法加权（#2，2026-11-22 修）：
# 同文本聚合已保证确认版赢下自己的槽位；排序里再乘 1.25 会把确认行
# 顶到分更高的未确认行前面，Q4/BM25 钉死的融合顺序就乱了
# （目标测试回归验出）。所以排序键只在 RRF×衰减×类型权重完全相等时，
# 用 confirmed 裁决先后。

# 偏好句的认法：正文里说喜欢 / 不喜欢 / 爱吃 / 爱喝的句子。
# 名单故意收得窄：只认这几个动词，不猜“想 / 希望 / 要”（那些是计划，
# 不是偏好）；“可爱”这类词不含这几个动词，不会误认。
_PREFERENCE_WORDS = ("喜欢", "爱吃", "爱喝", "爱的是", "不爱", "讨厌")

# 语义判断的触发标记（2026-11-22 混合法）：程序推不出槽位的偏好句式。
# “喜欢”已覆盖“喜欢吃/喝/看”，“爱看/爱听/爱玩”补“爱……”系的其余动词。
_JUDGE_PREFERENCE_MARKERS = (
    "喜欢", "爱吃", "爱喝", "爱看", "爱听", "爱玩", "不爱", "讨厌",
    # webfix 一补（复测 #1 实测）：“我最近迷上吃辣子鸡丁”不在表里，
    # 后来“我只吃白灼虾”触发判官时找不到这条候选、顶不掉，
    # 回复却说“不算了”——说的和存的对不上。迷上/爱上和爱吃同族。
    "迷上", "爱上",
)
# 一次判断最多带几条旧偏好候选（提示词要短，超了只带最近的）。
_JUDGE_MAX_CANDIDATES = 10

# 轮次回执缓存上限（2026-11-24）：去重回执要查 _turn_receipts，
# 无上限会一直长；只留最近 200 轮，超了删插入顺序最旧的。
_TURN_RECEIPTS_MAX = 200

# 偏好改变说法（2026-11-23 R2 后续①）：明确表达偏好变了的说法。
# “改吃/只吃/改喝/只喝/改看/只看……”——前缀“改/只/改成”+动词
# （吃/喝/看/听/玩）。不含“换成”等泛词，避免“改成明天”这类
# 非偏好句也触发。触发只决定“问不问判官”，判官才是裁决者；
# 边界说法（如“我只吃了一口”）判官会判不取代，所以宁可多问
# 一次判官，也别漏掉真正的偏好改变。
_JUDGE_CHANGE_MARKERS = tuple(
    prefix + verb
    for prefix in ("改", "只", "改成")
    for verb in ("吃", "喝", "看", "听", "玩")
)

# 判官候选里不要纠正记录（2026-11-23 R2 后续③收窄）：写着“纠正了此前……”
# 的便条是纠正的记录、不是一条活偏好，绝不能当取代目标（之前判官两次
# 把 72cc 那条纠正便条当成同类偏好取代了）。程序过滤是确定性的，
# 不管判官怎么判，这类记忆都不会被它取代。
_JUDGE_SKIP_MARKERS = ("纠正", "更正")

# 主食词（R2 后续③收窄）：天天吃的主食。判官目标合理性检查用——
# 新说法在换主食（米饭/面条这类）时，取代目标也必须是另一条主食记忆；
# 火锅/冒菜/烤鸡这类具体菜式或吃法不是主食，不能当目标。
_STAPLE_FOODS = (
    "米饭", "面条", "米粉", "米线", "馒头", "包子", "饺子", "面包", "粥", "主食",
)


def _judge_target_is_staple_plausible(new_text, target_text):
    """判官目标合理性：新说法含主食词时，目标也必须含主食词。

    主食换主食（面条换米饭）才合理；新说法在换主食、目标却是火锅/
    冒菜/牛肉面这类非主食时，判不取代（返回 False）。新说法不含主食词
    （饮品等）时不受这条限制，返回 True。注意这是判官路的守卫，
    程序纠正路不受影响——守卫拦下后，纠正路仍可正常取代米饭。
    """
    if not any(word in (new_text or "") for word in _STAPLE_FOODS):
        return True
    return any(word in (target_text or "") for word in _STAPLE_FOODS)


def _dedupe_key(text):
    """去重用的比对键：归一化后截掉句尾补充说明、再去掉全部标点。

    模型改写同一事实时标点不固定——“（主食偏好是面条）”和
    “，主食偏好是面条”都是句尾补充说明，截掉后与原句“改吃面条了”
    是同一串字，算同一条。只用于去重比对，不改原文；否定词
    （不/没）是汉字，不受影响。
    逗号只截“改写补充”那种尾巴（2026-11-24 2a，Gemini Bug 3）：
    以前无条件截掉最后一个逗号后的部分，“每天定投5块，买纳指”
    和“买标普”算出同一个 key，第二条被当重复静默吞掉。现在只当
    尾巴是“……是X”、且 X 已在前半句出现过（把前半句的内容换个
    说法重述，如“，主食偏好是面条”）时才截；普通多分句（买纳指/
    买标普、不喜欢吃米饭、也喜欢吃饺子）一律不截，两条都存得下。
    末尾括号和逗号同一条规矩（2026-11-24 R-3）：以前无条件截掉，
    “定投5块（买纳指）”和“（买标普）”算出同一个 key，同样静默
    吞第二条；现在只有括号里是“……是X”、且 X 已在括号前出现过
    （改写重述，如“（主食偏好是面条）”）时才截，内容性括号一律
    保留。
    """
    norm = _normalize_memory_text(text or "")
    # 末尾括号补充：只截改写重述（见 docstring），内容性括号不截。
    if norm.endswith(")"):
        idx = norm.rfind("(")
        if idx > 0:
            inner = norm[idx + 1:-1]
            if "是" in inner:
                stated = inner.rsplit("是", 1)[-1].strip()
                if stated and stated in norm[:idx]:
                    norm = norm[:idx]
    # 逗号补充：只截改写重述尾巴（见 docstring），普通分句不截。
    if "," in norm:
        head, _sep, tail = norm.rpartition(",")
        if "是" in tail:
            stated = tail.rsplit("是", 1)[-1].strip()
            if stated and stated in head:
                norm = head
    return re.sub(r"[^\w]", "", norm, flags=re.UNICODE)

def _extract_note_old_fact(note_text):
    """从模型写的纠正备注行里提被纠正掉的旧说法（2026-11-23 A 组修复）。
    备注行形如“BOOS 喜欢吃面条（纠正：不是喜欢吃米饭）”——
    “纠正：不是”后面就是旧说法的主干（“喜欢吃米饭”），前面那半句
    的主语（“BOOS”/“我”）拼上去就是整句旧说法。
    提不出来返回 None（调用方用整条备注行当 old_text，不硬编）。"""
    if not isinstance(note_text, str) or not note_text:
        return None
    marker_idx = -1
    marker_len = 0
    for marker in ("纠正：不是", "纠正:不是", "纠正：", "纠正:"):
        idx = note_text.find(marker)
        if idx >= 0:
            marker_idx = idx
            marker_len = len(marker)
            break
    if marker_idx < 0:
        return None
    old_core = note_text[marker_idx + marker_len:].strip()
    old_core = old_core.rstrip("）)。！").strip()
    if old_core.startswith("不是"):
        old_core = old_core[len("不是"):].strip()
    if not old_core or len(old_core) > 40:
        return None
    if any(w in old_core for w in ("之前", "是错的", "错了", "不对")):
        return None
    head = note_text[:marker_idx].rstrip("（(").strip()
    subject = ""
    if head:
        if head.startswith("BOOS"):
            subject = "BOOS"
        elif head.startswith("我"):
            subject = "我"
        elif head.startswith("用户"):
            subject = "用户"
    if subject:
        return subject + " " + old_core
    return old_core


# 问句不存：以这些结尾的用户话是提问，不是陈述事实。
_QUESTION_ENDINGS = ("?", "？", "吗", "么", "呢", "吧", "嘛")

# 打招呼/闲聊不存（只有整句是这些时）。
_SMALL_TALK = {
    "你好", "您好", "hi", "hello", "嗨", "在吗", "在么", "谢谢",
    "好的", "好", "嗯", "嗯嗯", "知道了", "收到", "拜拜", "再见",
    "就是这样", "就是这个",
}

# 相对日期词 → 和今天（上海）的天数差。
_RELATIVE_DAY_OFFSETS = {
    "大前天": -3, "前天": -2, "昨天": -1, "今天": 0, "明天": 1, "后天": 2, "大后天": 3,
}


# 中文星期 → 周一为 0 的序号（和 datetime.weekday() 一致）。
_WEEKDAY_CN_TO_NUM = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def _resolve_weekday(text, today):
    """“下周三/本周三/这周三/上周三”这类带星期的相对日期 → 绝对日期。

    周一是一周的开始：本周 = 今天所在周，下周 = 下一周，上周 = 上一周。
    找不到带星期的表达时返回 None。
    """
    m = re.search(r"(下周|本周|这周|上周)(?:星期|周|礼拜)?([一二三四五六日天])", text or "")
    if not m:
        return None
    qualifier, wd = m.group(1), m.group(2)
    target = _WEEKDAY_CN_TO_NUM[wd]
    monday = today - timedelta(days=today.weekday())
    if qualifier == "下周":
        base = monday + timedelta(days=7)
    elif qualifier in ("本周", "这周"):
        base = monday
    else:  # 上周
        base = monday - timedelta(days=7)
    return (base + timedelta(days=target)).isoformat()


def _shanghai_today():
    """今天（Asia/Shanghai 的日期）。"""
    return datetime.now(_SHANGHAI).date()


def resolve_event_date(text, today=None):
    """把一句话里的相对日期换算成绝对日期（YYYY-MM-DD）。

    找不到日期词时返回 None（不是没有日期，是这句话没提日期）。
    “昨天”以今天（上海）为基准算，不以任何记忆的日期为基准——
    饺子记成 11月19日就是因为提示里唯一的日期锚点是牛肉面那天。
    """
    today = today or _shanghai_today()
    for word, offset in _RELATIVE_DAY_OFFSETS.items():
        if word in (text or ""):
            return (today + timedelta(days=offset)).isoformat()
    wd = _resolve_weekday(text, today)
    if wd:
        return wd
    m = re.search(
        r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text or ""
    )
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})月(\d{1,2})日", text or "")
    if m:
        try:
            return date(today.year, int(m.group(1)), int(m.group(2))).isoformat()
        except ValueError:
            return None
    return None


def _split_sentences(text):
    """按句号/感叹号/问号/换行切句，句尾标点留着——问句判断要靠它
    （切掉问号就认不出问句了）；存的时候再把句尾的 。！去掉。"""
    parts = re.findall(r"[^。！？\n]+[。！？]?", text or "")
    return [p.strip() for p in parts if p.strip()]


# 一句话分两条发时的续接碎片判据（缺点清单第 11 条，2026-11-24）。
# 判据保守：单独的语气词、以标点开头、或极短（≤2 字）且不是能独立
# 成句的回复词。像“好/好的/行/嗯/谢谢”这种本身就是完整回复的，
# 永远不算碎片——拼错一句话比漏拼一句代价大。
_FRAGMENT_PARTICLES = "吗呢吧啊么嘛哦呀哇"
_FRAGMENT_TRAILING_PUNCT = "。！？.!?~～…、，,；;：:"
_FRAGMENT_STANDALONE = {
    "好", "好的", "行", "嗯", "对", "是", "可以", "没错", "不", "不要",
    "没事", "谢谢", "知道了", "继续", "嗯嗯", "好吧", "算了", "不用",
    "没了", "就这", "完了",
}


def _is_continuation_fragment(text, prev_text=""):
    """text 是不是上一条 user 消息的续接碎片（判据见上方注释）。

    prev_text 只用于一条护栏：极短规则（≤2 字）要求上一条没以句号/
    问号/感叹号收尾——上一句已经说完了，短消息多半是新话题，不拼。
    语气词和标点开头两条不看 prev：它们本身就成不了句。"""
    t = (text or "").strip()
    if not t:
        return False
    if t[0] in "，。、；：？！,.!?;:":
        return True
    core = t.rstrip(_FRAGMENT_TRAILING_PUNCT).strip()
    if not core:
        return True
    if core in _FRAGMENT_STANDALONE:
        return False
    if len(core) <= 2 and all(ch in _FRAGMENT_PARTICLES for ch in core):
        return True
    if len(core) <= 2:
        prev = (prev_text or "").strip()
        if prev and prev[-1] in "。！？!?":
            return False
        return True
    return False


# “记住”类前缀（2026-11-21 网页测第二轮 #8）：save_turn 逐句循环里
# 先剥掉再存——“记住后天要去体检”要存成“后天要去体检”，不能把
# “记住”两个字也存进记忆。顺序按最长优先；main.py 的
# is_remember_command / remember_content 用同一份名单（两边改了要同步）。
REMEMBER_PREFIXES = ("帮我记住", "请记住", "记住")


def _strip_remember_prefix(sentence):
    """剥掉句首的“记住”类前缀，返回剩下的内容；没有前缀时原样返回。"""
    s = (sentence or "").strip()
    for prefix in REMEMBER_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):].lstrip("：:、，, ").strip()
    return s


# 确认词前缀（2026-11-21 网页测第三轮 #8“别什么都记”）：BOOS 回一句
# “是的/对”再跟半句话（“是的删掉”）时，确认词本身不是记忆内容，
# 先剥掉再判剩下的；剥完没内容就是光秃秃一句确认词，直接丢掉。
# 两个字的前缀（“是的/对的/没错/可以/好的”）直接剥；一个字的
# （“好/嗯/行/对”）只有后面跟了分隔符才剥（“好吃”不能剥成“吃”，
# “对不起”不能剥成“不起”）。
CONFIRMATION_PREFIXES = ("是的", "对的", "没错", "可以", "好的")
_CONFIRMATION_SINGLE = ("好", "嗯", "行", "对")


def _strip_confirmation_prefix(sentence):
    """剥掉句首的确认词，返回剩下的内容；没有确认词前缀时原样返回。

    剥完为空时返回 ""（调用方按确认词丢掉）。只剥句首，句中的“是的”不动。
    """
    s = (sentence or "").strip()
    for prefix in CONFIRMATION_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):].lstrip("：:、，, ").strip()
    for prefix in _CONFIRMATION_SINGLE:
        if s.startswith(prefix) and len(s) > 1 and s[1] in "：:、，, ":
            return s[1:].lstrip("：:、，, ").strip()
    return s


def _looks_like_fact(sentence):
    """一句话像不像用户亲口说的事实（原样先存的候选）。

    不存的：太短的、问句、假设句、确认词、半句话、当下的想法、
    打招呼、光秃秃的撤回、让删/让忘的指令。
    其余的都先存下来——值不值得长期留以后靠衰减沉底，不在这一步判死刑。
    2026-11-21 网页测第三轮 #8（BOOS 定：改成和十一一样，别什么都记）：
    过滤器必须是确定性的程序规则（正则/规则），不能靠模型运气——
    之前就是存不存看运气（有的问句存了、有的没存）。
    """
    s = (sentence or "").strip()
    # 2026-11-23 Phase 2 #2 线上实测根因：分句留着句尾标点（问句判断
    # 要靠 ？），但 endswith 类规则认不出带 。！的句子——“我想喝茶了。”
    # 句尾是 。不是了，当下的想法规则没拦住，线上 seq 494 就这么存进
    # 去了（单测的句子没带句号，所以没抓到）。这里先剥掉句尾的 。！，？
    # 留着（问句判断要靠它）；存的时候调用方还会再剥一遍，不影响原文。
    s = s.rstrip("。！")
    if len(s) < 4:
        return False, "太短"
    if s in _SMALL_TALK:
        return False, "闲聊"
    if s in _BARE_RETRACTION:
        return False, "撤回信号"
    if s.endswith(_QUESTION_ENDINGS):
        return False, "问句"
    # 没打问号的问句：以“怎么样/如何/为什么/谁/哪/多少/怎么办/什么”收尾的，
    # 几乎都是提问（“我不知道他是谁”这类也只是“不知道”的陈述，
    # 本来就不值得当事实存）。2026-11-21 回归抽查抓到的：
    # “定投计划怎么样了”没打问号，被当事实存进去、检索还排第一；
    # 第三轮又抓到“我爱吃什么”（“什么”结尾）、“我的定投现在是什么情况”
    # （“什么情况”结尾）也被当事实存进去。
    if s.endswith(("怎么样", "怎么样了", "如何", "如何了", "怎么了", "为什么",
                   "怎么办", "是什么", "是不是", "谁", "哪", "多少",
                   "几个", "几天", "几点", "什么", "什么情况")):
        return False, "问句"
    # 正反问（A-不-A）和疑问词开头的问句：没打问号、也不以疑问词收尾，
    # 但句式就是提问——“我喜不喜欢吃鸡杂面”“有没有人来过”。
    # “我不知道他是谁”不含这些词，不受影响；“我不喜欢吃辣”含的是
    # “不喜欢”不是“喜不喜欢”，不受影响。
    if re.search(r"(喜不喜欢|爱不爱|想不想|要不要|会不会|能不能|懂不懂|记不记得|知不知道|好不好|行不行|对不对)", s):
        return False, "问句"
    # “记得不”型句尾（2026-11-24 webfix 二）：没打问号、不是 A-不-A
    # 全形——“我的口味你还记得不”句尾就一个“记得不”，旧模式只认
    # “记不记得”，这句漏过去被当事实存了（stable+high 实测）。
    # 只认句尾，句中的“记得不”（“我记得不能吃辣”）不受影响。
    if s.endswith(("记得不", "知道不", "清楚不", "明白不")):
        return False, "问句"
    if re.match(r"^(有没有|是不是|能不能|会不会|要不要|想不想|喜不喜欢|爱不爱|懂不懂|记不记得|知不知道|为什么|怎么|如何|什么时候|什么地方|什么东西|什么人)", s):
        return False, "问句"
    # 假设句（第三轮 #8）：句首的“如果/假设/假如/要是/万一”——条件句、
    # 假设题不是既定事实，不存。mem0 那条路的提取（FakeMemory）本来就
    # 不认句首假设句，这里补齐同一条规矩。句中的“如果”不动
    # （“你说如果下雨就别去”是条件指令，不是假设题）。
    if re.match(r"^(如果|假设|假如|要是|万一)", s):
        return False, "假设句"
    # 半句话（第三轮 #8）：“这几条取消”——指示代词 + 量词 + 动词，
    # 半截指令，没有完整意思，不存。
    if re.match(r"^[这那][几]?[条个只项种]?(取消|删掉|删除|去掉|不要|留着|算了)", s):
        return False, "半句话"
    if "不要记" in s:
        return False, "删除指令"
    # 句中的删除指令（第三轮 #8）：“算了，关于定投的全部删掉吧”
    # 这类没走句首的删指令；句尾带“了”的回忆句除外。
    if any(w in s for w in ("全部删掉", "都删掉", "删掉吧")) and not s.endswith("了"):
        return False, "删除指令"
    # “忘记…”指令句（2026-11-21 网页测第二轮 #7）：“忘记我喜欢喝茶”
    # 是删记忆的指令，不能当事实存。只认句首的忘记/忘掉/删掉/删除
    # （“我昨天忘记带钥匙了”不在句首，不受影响）；句尾带“了”的是
    # 回忆句（“忘记带钥匙了”），不是指令，照常存；“忘记了…”（过去时）
    # 也是回忆，照常存。没带“了”的“忘记带钥匙”会被当成指令丢掉——
    # 故意的取舍：删指令绝不能进库，丢一句含糊的回忆可以再说一遍。
    if re.match(r"^(?:请帮我|帮我|请)?(?:忘记|忘掉|删掉|删除)(?!了)", s) and not s.endswith("了"):
        return False, "删除指令"
    # 句中的“忘记这个/忘掉这个/删掉这条”（比如“你能不能忘记这个”）
    # 同样是指令；句尾带“了”的回忆句除外。
    if any(w in s for w in ("忘记这个", "忘掉这个", "删掉这条")) and not s.endswith("了"):
        return False, "删除指令"
    # 当下的想法（第三轮 #8，BOOS 定：按十一的来）：“我想喝茶了”这种
    # 当下的想法不存成偏好/长期记忆。信号：想/馋 + 当下标记
    # （句尾“了”、句尾“想”、“现在/这会儿”）；带了以后/明天/后天/
    # 下周/打算/决定/计划/每天/每周的，是未来的计划或安排，不拦。
    # “我想去爬山”（没“了”、没未来标记）是表态的意图，拿不准，
    # 先留着（沉底规则管它），不拦。
    if re.search(r"[想馋]", s) and (
        s.endswith("了") or s.endswith("想") or "现在" in s or "这会儿" in s
    ) and not any(
        w in s for w in ("以后", "明天", "后天", "下周", "下个月", "打算", "决定", "计划", "从今天起", "每天", "每周")
    ):
        return False, "当下的想法"
    # 纠错信号本身（2026-11-22 correct_memory 接线后）：带新说法的纠错句
    # 由 save_turn 经 correct_memory 处理，不到这里；这里只拦光秃秃的
    # 纠错信号（“你记错了”后面没跟新说法），那不是一条事实。
    # 原来的宽拦截（句首是纠错说法就整句丢）已随接线移除，否则
    # “你记错了，我住上海”会把后半句纠错新事实一并丢弃。
    if re.match(r"^(?:(?:不对|不是|哎)[，,、\s]*)?(?:你|我)?(?:之前|刚才)?(?:记错了|说错了|搞错了|记岔了)[，,。！!、\s]*$", s):
        return False, "纠错指令"
    if re.match(r"^(?:纠正一下|更正一下|更正|改一下|不对|错了|不是)[，,。！!、\s]*$", s):
        return False, "纠错指令"
    return True, ""



def _is_question_sentence(text):
    """一句话是不是问句（重要性护栏用，2026-11-24 webfix 二）。

    复用 _looks_like_fact 的问句判定（同一个规矩源，不另立一套
    模式）：它以“问句”为原因拒绝的，就是问句；另外带 ？/? 结尾
    的也算（_looks_like_fact 会先剥句尾标点再看，这里直接认）。
    """
    s = (text or "").strip()
    if not s:
        return False
    if s.endswith(("？", "?")):
        return True
    ok, reason = _looks_like_fact(s)
    return (not ok) and reason == "问句"


# 用户明确强调的说法（重要性护栏的例外，2026-11-24 webfix 二）：
# 带这些词的单次提及，判官判 important 程序不封顶——BOOS 亲口
# 说了“别忘了”，那就是要钉住的。光秃秃的“记住/记一下”不算强调
# （那是“记住”命令的触发词，不是重要程度的表态）。
_EMPHASIS_MARKERS = (
    "很重要", "特别重要", "非常重要", "别忘了", "不要忘了",
    "一定要记住", "千万别忘", "牢记", "记牢",
)


def _has_emphasis(text):
    return any(marker in (text or "") for marker in _EMPHASIS_MARKERS)


# 重复性/习惯标记（重要性护栏的另一类例外）：句子本身在陈述一个
# 反复发生的习惯/规律（“我每天下班去游泳”），不是“单次提及”——
# 判官判 important 时程序不封顶（Phase 2 #2 既有行为：习惯类
# 判重要就进核心）。“睡前/晚上”这类时段词不算重复性标记。
_RECURRENCE_MARKERS = (
    "每天", "每日", "天天", "每周", "每月", "每年",
    "总是", "老是", "经常", "习惯", "定期", "日常",
)


def _has_recurrence(text):
    return any(marker in (text or "") for marker in _RECURRENCE_MARKERS)


# 纠正信号词（save_turn 叠剥护栏用）：一截文本剥掉这些和标点后
# 应该什么都不剩，才算“纯纠正信号”——剩了实词（如“不吃牛肉”）
# 就是事实内容，不许当信号剥掉。
_CORRECTION_SIGNAL_TOKENS = (
    "更正一下", "纠正一下", "改一下", "其实是", "应该是",
    "说错了", "记错了", "搞错了", "不对", "其实", "更正",
    "不是", "错了", "我", "你", "他", "她", "这", "那",
)


def _correction_signal_only(text):
    s = text or ""
    for ch in "，。、：；,.:; \t\n":
        s = s.replace(ch, "")
    for token in _CORRECTION_SIGNAL_TOKENS:
        s = s.replace(token, "")
    return not s.strip()


def _normalize_metadata(meta):
    """读时归一化 metadata：补算 confirmed（mem0 的规矩，见
    memory/provenance.py _normalize_metadata：confirmed 就是
    user_confirmed 的短名字，读的时候现算，不是存的时候写）。

    账本/Chroma 来的旧 metadata 里没有 confirmed 这个键——不补算，
    问答提示词（main.py _memory_block）会把 source=user 的已确认记忆
    也当成【来源未确认】、问起来只说不知道。只补缺失的键，不覆盖已有的。
    """
    meta = dict(meta or {})
    if "confirmed" not in meta:
        meta["confirmed"] = bool(meta.get("user_confirmed"))
    return meta


class SQLiteStore:
    """SQLite 记忆库：账本是真相，SQLite 是索引。

    和 Mem0Store 同样的外观，供 web/server.py、main.py 直接用。
    """

    def __init__(self, memory_dir=None, budget_tracker=None,
                 judge_provider=None, **kwargs):
        self.memory_dir = Path(memory_dir or _default_memory_dir())
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.memory_dir / "memory_events.jsonl"
        self.snapshot_path = self.memory_dir / "graph_snapshot.json"
        self.db_path = self.memory_dir / "memories.sqlite3"
        # 图谱二期（2026-11-24）：图是账本的投影（索引），按账本文件
        # 指纹（mtime+大小）缓存，账本一变就从事件重建。注意别和
        # snapshot_path 混了——那个名叫 graph_snapshot.json 的文件
        # 是账本状态快照（load_ledger_state 用），不是图。
        self._graph_cache = None
        self._graph_stamp = None
        self.budget_tracker = budget_tracker
        # 语义判官对象（2026-11-22 混合法）：测试注入 mock 用；
        # None 时每次判断按 config.py 的 SEMANTIC_JUDGE 解析（默认 DeepSeek flash）。
        self._judge_provider = judge_provider
        # 待问 BOOS 的问题（Phase 2 #4）：攒批 + 回答，落盘不丢。
        self.pending_questions = _pq.PendingQuestions(memory_dir=self.memory_dir)
        # 性情区原件在壳（追加决定八）：壳 SOUL.md 路径——默认
        # 仓库根（persona_shell.default_shell_path），测试可经
        # shell_soul_path 注入临时壳。本店只读它、永不写它。
        self._shell_soul_path = Path(
            kwargs.get("shell_soul_path")
            or _persona_shell.default_shell_path()
        )
        # 入口当场拦/当场判（实测问题三）：intake_marker 是对话
        # 当场模型实现的标记器（可调用对象，入参 (候选, messages)，
        # 返回标记列表）；None = 不可用，save_turn 自动回退老管道。
        self.intake_marker = None
        self._intake_log_path = self.memory_dir / "intake_log.jsonl"
        self._intake_log_lock = threading.Lock()
        # 统一受控入口（立项 H1，2026-11-25）：整理员侧状态变更的正门。
        # 回执登记落盘（幂等键 → 回执），重启重放不重复执行；留痕走
        # 独立 append-only 日志（先例：intake_log.jsonl），不进账本折叠。
        self._controlled_log_path = self.memory_dir / "controlled_entry_log.jsonl"
        self._controlled_log_lock = threading.Lock()
        self._controlled_receipts_path = (
            self.memory_dir / "controlled_entry_receipts.json"
        )
        # 生产执行开关：默认关。入口只接受影子/测试调用（走完全部门禁、
        # 出回执、不落状态）；打开的前置（H2 会话认证）已就位——开关
        # 打开后入口强制验会话凭据（见 session_auth）。生产接线仍未
        # 授权（放行四关未过），本开关保持关。
        self.controlled_entry_execution_enabled = False
        # 会话认证（立项 H2，2026-11-25）：开关打开后、执行前验凭据
        # 的签发权威——登记与签名密钥只活在内存里（进程级），不落盘、
        # 不进账本；签发只供宿主内部流程与测试，不存在默认后门凭据；
        # web 层无会话概念（现状保留），接缝以本层的凭据校验为准。
        self._session_authority = _session_auth.SessionAuthority()
        self._controlled_receipts = {}
        self._controlled_receipts_load_error = None
        self._controlled_trace_index = 0
        # 补修轮（2026-11-25）：操作意图登记（提交/恢复协议）、
        # 提案指纹登记、健康冻结标记——都在 _load 里载入/对账。
        self._controlled_intents_path = (
            self.memory_dir / "controlled_entry_intents.json"
        )
        self._controlled_intents = {}
        self._controlled_intents_load_error = None
        self._controlled_fingerprints = {}
        # 撤销接口（立项 H3，2026-11-25）：撤销回执与撤销指纹
        # 登记，与提案回执同文件落盘（见 _controlled_save_receipts）。
        self._controlled_revocations = {}
        self._controlled_revocation_fingerprints = {}
        # 持锁稳定版本快照（立项 H4，2026-11-25）：整理员提案绑
        # 版本用的冻结视图，只活在进程内登记里——不落盘、不进账
        # 本，进程重启即全失效（get/diff 如实 KeyError，不给假数
        # 据）；口径见下方快照节。
        self._stable_snapshots = {}
        # 保护迁移完成证据（立项 H5，2026-11-25）：保护档迁移的
        # 追加式证据日志（先例：controlled_entry_log.jsonl）——只
        # 追加、本类不提供改/删接口；进程内另记「已迁移待补证」
        # 清单，供 pending_notices 露头与核对函数对照。
        self._protection_evidence_path = (
            self.memory_dir / "protection_migration_evidence.jsonl"
        )
        self._protection_evidence_lock = threading.Lock()
        self._protection_evidence_pending = []
        # 恢复健康度（立项 H6，2026-11-25）：最近一次启动恢复的
        # 健康结论落盘于 memory 目录下 recovery_health.json，
        # 跨重启可查（补决定十五的缺口：待问 load_error 只活在
        # 进程里、重启即消失、报警无留存）。构造时先读回上一
        # 份（_load_recovery_health）；上一轮恢复不干净且冻的
        # 是危险写时，本进程先承接冻结、等本轮
        # startup_recovery 重新裁决。口径见 startup_recovery
        # 之后的 H6 节。
        self._recovery_health_path = (
            self.memory_dir / "recovery_health.json"
        )
        self._recovery_health_latest = None
        self._recovery_health_own = False
        self._recovery_health_load_error = None
        self._recovery_health_lock = threading.Lock()
        self._controlled_frozen_reason = None
        self._controlled_health_issues = []
        self._controlled_reconcile_report = None
        self._load_controlled_entry_state()
        self._load_recovery_health()
        # 同轮裁决缓存 {(turn_id, 归一化原文): (verdict, info)}：
        # save_turn 逐句循环里带当场判标记合并裁决过的句子，下游
        # 内部直存（槽位路、收尾直存）复用同一裁决——不许二次屏
        # 在没有标记上下文的情况下把已放行的句子判回挂起。
        self._intake_verdict_cache = {}
        # 核心座位上限（四件套第 1 块）：默认契约定的 150，
        # 测试可调小验证容量门。
        self.core_seat_limit = _core_cap.CORE_SEAT_LIMIT
        self.user_id = kwargs.get("user_id", "boos")
        self._lock = threading.RLock()
        self._ledger_lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        # 写序号播种（2026-11-22，S1 根因）：_WRITE_SEQ 是进程内自增、重启
        # 归零；不播种时新说法拿到的序号比旧行还小，同级裁决会误判成
        # “迟到的旧写入”而 keep_old。启动时抬到库里最大的 write_seq。
        try:
            max_seq = 0
            for (seq,) in self._conn.execute(
                "SELECT write_seq FROM memories"
            ).fetchall():
                if isinstance(seq, int) and not isinstance(seq, bool) and seq > max_seq:
                    max_seq = seq
            if max_seq:
                _seed_write_seq(max_seq)
        except Exception:
            logger.warning("写序号播种失败：本次不播种。", exc_info=True)
        self._embedder = None
        self._embedder_failed = False
        self._seen_turn_ids = set()
        self._turn_receipts = {}
        self._last_context_ids = []
        self._last_stored_turn_id = None
        # 当前轮登记（2026-11-24 1b）：ask() 检索/存轮时把本轮 turn_id
        # 登记在这；本轮内工具调 correct_memory 没带 turn_id 时按它记账，
        # “已纠正过”护栏靠它认本轮（不再靠 5 分钟墙钟）。
        self._active_turn_id = None

    # ---------- 建库 ----------

    def _init_schema(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                write_seq INTEGER,
                slot TEXT,
                source TEXT,
                user_confirmed INTEGER NOT NULL DEFAULT 0,
                decay_kind TEXT,
                importance TEXT,
                event_date TEXT,
                embedding BLOB,
                voided INTEGER NOT NULL DEFAULT 0,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_event_date
                ON memories(event_date);
            CREATE TABLE IF NOT EXISTS vector_cache (
                sha256 TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()
        # superseded 列（2026-11-22 correct_memory 接线）：被取代的记忆
        # 在索引里标 superseded=1，不再进活记忆；旧库没有这列时补上。
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "superseded" not in cols:
            self._conn.execute("ALTER TABLE memories ADD COLUMN superseded INTEGER NOT NULL DEFAULT 0")
            self._conn.commit()

    def available(self):
        return True

    # ---------- 向量（本地 bge，带缓存） ----------

    def _embed_text(self, text):
        """给一句话算向量：先查缓存（key=sha256(text)），没有再调本地模型。

        模型起不来时返回 None——存记忆绝不因为向量失败而失败，
        那条记忆先以无向量入库，检索时靠原样/关键词兜底。
        """
        key = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        # 向量缓存是同一单连接上的读写：和写路径一样加锁（#2）。
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM vector_cache WHERE sha256=?", (key,)
            ).fetchone()
        if row is not None:
            return _blob_to_vector(row[0])
        if self._embedder_failed:
            return None
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                # 本地加载，不碰 Hub：这台电脑离线也能从缓存起模型；
                # 不加 local_files_only 时，新版 transformers 会去 Hub 查
                # chat template，断网时一次次重试、拖几分钟。
                self._embedder = SentenceTransformer(
                    EMBEDDING_MODEL, local_files_only=True
                )
            import numpy as np

            vec = self._embedder.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]
            vec = np.asarray(vec, dtype="float32")
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO vector_cache(sha256, embedding, model, dim)"
                    " VALUES (?, ?, ?, ?)",
                    (key, vec.tobytes(), EMBEDDING_MODEL, int(vec.shape[0])),
                )
                self._conn.commit()
            return vec
        except Exception:
            logger.warning("本地向量模型起不来：本次只存文字、不存向量。", exc_info=True)
            self._embedder_failed = True
            return None

    # ---------- 账本（真相源） ----------

    def _ledger_append_add(self, text, metadata):
        """先写账本 add 事件（写失败就抛，索引那步也别写）。"""
        return _ledger.append_event(
            self.ledger_path,
            {
                "type": _ledger.LEDGER_EVENT_ADD,
                "key": _ledger.ledger_event_key(text, metadata.get("created_at")),
                "text": text,
                "meta": {
                    "source": metadata.get("source"),
                    "confidence": metadata.get("confidence"),
                    "user_confirmed": metadata.get("user_confirmed"),
                    "created_at": metadata.get("created_at"),
                    "write_seq": metadata.get("write_seq"),
                    "turn_id": metadata.get("turn_id"),
                    "slot": metadata.get("slot"),
                    "is_rule": metadata.get("is_rule", False),
                    "curated": metadata.get("curated", False),
                    # 核心上限字段（第 1 块）：保护档与分类抽屉必须
                    # 进账本，不然按账本重算座位时这两样会丢。
                    "protection_tier": metadata.get("protection_tier"),
                    "core_drawer": metadata.get("core_drawer"),
                    "decay_kind": metadata.get("decay_kind"),
                    "importance": metadata.get("importance"),
                    "event_date": metadata.get("event_date"),
                    # 衰减字段（2026-11-22 decay 接线）：账本是真相，
                    # 对账重建（reconcile_ledger_index）直接拿这里的
                    # meta 当 metadata_json，不写进来重建就丢了。
                    "last_used_at": metadata.get("last_used_at"),
                    "decay_state": metadata.get("decay_state"),
                    "seen_dates": metadata.get("seen_dates"),
                    "seen_sessions": metadata.get("seen_sessions"),
                    # 授权标记（实测问题一）：auth_claim 不进账本
                    # 的话，行丢了按账本重建时标记就没了——待问
                    # 文件损坏后按行 meta 标记重建 auth_claim 问题
                    # 时，捞回来的行永远缺这个标记、问题永远漏
                    # 掉（2026-11-25 互查 (b)，同核心字段进账本
                    # 的先例口径）。
                    "auth_claim": metadata.get("auth_claim"),
                    "auth_reason": metadata.get("auth_reason"),
                    "claimed_source": metadata.get("claimed_source"),
                    # 性情投影标记（追加决定八）：不进账本的话，
                    # 按账本重建时标记与重印戳会丢、复印件就没
                    # 人认得出是投影（同核心字段进账本的口径）。
                    "persona_projection": metadata.get(
                        "persona_projection"
                    ),
                    "persona_shell_sha256": metadata.get(
                        "persona_shell_sha256"
                    ),
                },
            },
            lock=self._ledger_lock,
        )

    def ledger_state(self):
        try:
            return _ledger.load_ledger_state(self.ledger_path, self.snapshot_path)
        except Exception:
            logger.warning("读账本状态失败。", exc_info=True)
            return {"by_id": {}, "adds": [], "relations": [],
                    "key_states": {}, "id_to_key": {}, "inferences": {},
                    "seq_max": 0, "snapshot_seq": None}

    # ---------- 图谱二期：图投影与边维护 ----------

    def _ledger_stamp(self):
        """账本文件指纹：(mtime_ns, 大小)；文件不在返回 None。"""
        try:
            st = os.stat(self.ledger_path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _get_graph(self):
        """拿当前图（账本投影）：账本没变用缓存，变了从事件重建。

        重建走 MemoryGraph.rebuild_from_events：关系边由事件重放时
        自动判，同义词边的新增/作废按 graph_edge 事件重放，边的
        有效期按账本状态同步。图建不出来返回空图——检索少一路，
        不拦路（和 BM25 坏掉只丢一路同口径）。
        """
        stamp = self._ledger_stamp()
        if self._graph_cache is not None and stamp == self._graph_stamp:
            return self._graph_cache
        graph = _graph_mod.MemoryGraph()
        _info = None
        try:
            events, _info = _ledger.read_events(self.ledger_path)
            graph.rebuild_from_events(events)
        except Exception:
            logger.warning("图谱重建失败：本次不用图。", exc_info=True)
            graph = _graph_mod.MemoryGraph()
        if _info is not None and self._controlled_ledger_doubtful(_info):
            # 账本残缺（补修 H3 复查 F2）：图是残缺折叠，
            # 本次照用（检索少一路可降级），但不进缓存——
            # 不许把残缺折叠当定案缓存下来，账本修好后下
            # 次调用重建。
            return graph
        self._graph_cache = graph
        self._graph_stamp = self._ledger_stamp()
        return graph

    def add_synonym_edge(self, name_a, name_b, scope, provenance="",
                         evidence=""):
        """落一条同义词边：先写账本 graph_edge 事件，再按账本重建图。

        scope 必填（契约：无作用域不许建；默认只在提出它的上下文
        内生效，全局要明写 global）。审批分级（普通词判官放行、
        人名/称呼双模型核，拍板 5）是调用方的事——这里只负责
        「边带 scope 与出处、账本留痕、检索按 scope 过滤」这套
        机制。实体名先归一化再写账，免得同一实体两种写法对不上。
        返回 {"ok": bool, ...}；ok 只表示账写上了，applied 表示
        重建后的图里这条边确实有效（同对同 scope 已有有效边时
        applied 为 True 且不重复连）。
        """
        a = _normalize_entity_name(name_a)
        b = _normalize_entity_name(name_b)
        if not a or not b or a == b:
            return {"ok": False, "reason": "bad_entities"}
        if not isinstance(scope, str) or not scope.strip():
            return {"ok": False, "reason": "scope_required"}
        scope = scope.strip()
        try:
            record = _ledger.append_event(
                self.ledger_path,
                {
                    "type": _ledger.LEDGER_EVENT_GRAPH_EDGE,
                    "action": "add",
                    "kind": _graph_mod.EDGE_SYNONYM,
                    "entity_a": a,
                    "entity_b": b,
                    "scope": scope,
                    "provenance": provenance or "",
                    "evidence": evidence or "",
                    "valid_at": _now_iso(),
                },
                lock=self._ledger_lock,
            )
        except Exception:
            logger.warning("同义词边写账本失败。", exc_info=True)
            return {"ok": False, "reason": "ledger_failed"}
        self._graph_cache = None
        graph = self._get_graph()
        pair = {_graph_mod.entity_node_id(a), _graph_mod.entity_node_id(b)}
        applied = any(
            edge.get("kind") == _graph_mod.EDGE_SYNONYM
            and not edge.get("invalid_at")
            and edge.get("scope") == scope
            and {edge.get("from"), edge.get("to")} == pair
            for edge in graph.edges
        )
        return {"ok": True, "applied": applied, "seq": record.get("seq")}

    def void_graph_edge(self, kind, name_a=None, name_b=None,
                        from_key=None, to_key=None, reason=""):
        """作废一条边：写账本 graph_edge(invalidate) 事件，边不删。

        同义词边按实体名认（kind="synonym" + name_a/name_b），
        关系边按记忆 key 认（kind 为四种关系之一 + from_key/
        to_key，取代边方向按 from=旧、to=新）。作废后检索不再
        走这条边；边本身留在账本与图里，edge_history 查得到
        （含作废时间与理由）。返回 {"ok": bool, ...}。
        """
        event = {
            "type": _ledger.LEDGER_EVENT_GRAPH_EDGE,
            "action": "invalidate",
            "kind": kind,
            "invalid_at": _now_iso(),
            "reason": reason or "",
        }
        if kind == _graph_mod.EDGE_SYNONYM:
            a = _normalize_entity_name(name_a)
            b = _normalize_entity_name(name_b)
            if not a or not b:
                return {"ok": False, "reason": "entities_required"}
            event["entity_a"] = a
            event["entity_b"] = b
        else:
            if not from_key or not to_key:
                return {"ok": False, "reason": "keys_required"}
            event["from_key"] = from_key
            event["to_key"] = to_key
        try:
            record = _ledger.append_event(
                self.ledger_path, event, lock=self._ledger_lock
            )
        except Exception:
            logger.warning("图边作废写账本失败。", exc_info=True)
            return {"ok": False, "reason": "ledger_failed"}
        self._graph_cache = None
        return {"ok": True, "seq": record.get("seq")}

    def graph_edge_history(self, name_a=None, name_b=None,
                           from_key=None, to_key=None, kind=None):
        """查两个节点之间的边全史（有效 + 作废，带出处与作废理由）。

        同义词边按实体名查，关系边按记忆 key 查；都不给返回 []。
        """
        graph = self._get_graph()
        a = _normalize_entity_name(name_a)
        b = _normalize_entity_name(name_b)
        if a and b:
            return graph.edge_history(
                _graph_mod.entity_node_id(a),
                _graph_mod.entity_node_id(b),
                kind=kind,
            )
        if from_key and to_key:
            return graph.edge_history(
                _graph_mod.memory_node_id(from_key),
                _graph_mod.memory_node_id(to_key),
                kind=kind,
            )
        return []

    def _ledger_overlays(self):
        """扫一遍账本事件，算出账本对每条记忆的最新一笔覆盖：
        (promote_meta, decay_marks, slot_map, revived_keys)。

        promote_meta：每个 key 最新一笔 PROMOTE 的完整 meta（晋升、
        保护档、抽屉都记这笔）；decay_marks：当前还沉底/待删的
        key → decay_state（REVIVE 来了就摘掉）；slot_map：最新一笔
        SLOT 的槽位；revived_keys：沉底过、又被 REVIVE 摘掉的 key
        （账本折叠态已回原话态）。reconcile 重建缺失行和启动修复
        已有行共用这一份口径，不许两边各算一套。读账本失败抛给
        调用方按各自口径兜底。"""
        promote_meta = {}
        decay_marks = {}
        slot_map = {}
        revived_keys = set()
        _events, _info = _ledger.read_events(self.ledger_path)
        if self._controlled_ledger_doubtful(_info):
            # 账本残缺（补修 H3 复查 F2）：覆盖层是折叠结
            # 论，残缺折叠不许当定案——抛给调用方按各自兜
            # 底口径跳过覆盖层（对账/修复的既有 except 路
            # 径；只读复检则向上抛由启动恢复处置）。
            raise ValueError("ledger_doubtful")
        for _event in _events:
            _etype = _event.get("type")
            _key = _event.get("key") or _event.get("memory_id")
            if _etype == _ledger.LEDGER_EVENT_PROMOTE:
                _meta = _event.get("meta") or {}
                if _key and _meta:
                    promote_meta[_key] = _meta
            elif _etype == _ledger.LEDGER_EVENT_SLOT and _key:
                # R-7（2026-11-24）：槽位标的账本事件，重建/修复
                # 时把 slot 一并带回来（最新一笔赢）。
                if _event.get("slot"):
                    slot_map[_key] = _event["slot"]
            elif _etype == _ledger.LEDGER_EVENT_DORMANT and _key:
                # 3b（2026-11-24）：沉底/待删标记在账本里有事件，
                # 重建/修复时把 decay_state 一并带回来。
                decay_marks[_key] = (
                    _decay.DECAY_STATE_DELETE_DUE
                    if _event.get("delete_due")
                    else _decay.DECAY_STATE_DORMANT
                )
                revived_keys.discard(_key)
            elif _etype == _ledger.LEDGER_EVENT_REVIVE and _key:
                if _key in decay_marks:
                    decay_marks.pop(_key, None)
                    revived_keys.add(_key)
        return promote_meta, decay_marks, slot_map, revived_keys


    def reconcile_ledger_index(self):
        """启动对账：账本里有、SQLite 里没有的，按账本重建索引。

        只重建当前态的记忆：被取代（superseded）/删了（deleted）的
        key 不进索引（账本里原话留着，索引里不给搜到——和 mem0 那条路
        的投影过滤同一个意思）。
        """
        with self._lock:
            state = self.ledger_state()
            key_states = (state or {}).get("key_states") or {}
            # P1-4（2026-11-23）：PROMOTE 事件带升级后的完整 meta——
            # 去重升级（add_memory_text）、3 天 3 次晋升、回答问题进核心
            # 都记 PROMOTE。重建缺失行时按最新一笔 PROMOTE 的 meta 覆盖
            # 升级过的字段，否则重建出来的还是 add 那笔的旧 meta
            #（推断版/episodic），升级就丢了。只覆盖这 4 个晋升字段，
            # 不碰 last_used_at / seen_*（那是后来的使用记录，PROMOTE
            # 那一刻的值是旧的）。
            try:
                promote_meta, decay_marks, slot_map, _revived_keys = (
                    self._ledger_overlays()
                )
            except Exception:
                logger.warning("读账本 PROMOTE 事件失败：重建时不应用晋升。", exc_info=True)
                promote_meta, decay_marks, slot_map = {}, {}, {}
            touch_map = (state or {}).get("last_touched") or {}
            added = 0
            for record in (state or {}).get("adds") or []:
                key = record.get("key")
                text = record.get("text") or ""
                meta = record.get("meta") or {}
                if not key or not text.strip():
                    continue
                if key_states.get(key) in (
                    _ledger.LEDGER_STATE_SUPERSEDED,
                    _ledger.LEDGER_STATE_DELETED,
                ):
                    continue
                if self._conn.execute(
                    "SELECT 1 FROM memories WHERE id=?", (key,)
                ).fetchone():
                    continue
                _promoted = promote_meta.get(key) or {}
                _touched = touch_map.get(key)
                _mark = decay_marks.get(key)
                _slot = slot_map.get(key)
                if _promoted or _touched or _mark or _slot:
                    meta = dict(meta)
                if _promoted:
                    for _field in ("source", "user_confirmed", "decay_kind", "importance",
                                   # webfix 二：语气标也走 PROMOTE 写回
                                   # （_tag_memory_tone），重建时一并带回，
                                   # 不然重启对账会把标丢了。
                                   "tone", "tone_evidence", "tone_source"):
                        if _promoted.get(_field) is not None:
                            meta[_field] = _promoted[_field]
                    # 核心上限字段（第 1 块）：以「键在不在」为准——
                    # 显式写 None 是摘档/摘格，重建时照样摘掉。
                    for _field in ("protection_tier", "core_drawer"):
                        if _field in _promoted:
                            meta[_field] = _promoted[_field]
                if _slot:
                    # R-7＋口径对齐（2026-11-25）：账本最新 SLOT 事件赢——
                    # ADD 自带旧槽位也要被后来的 SLOT 事件盖掉，与
                    # repair_projection_from_ledger 同口径
                    # （_insert_memory_row 会把 meta 里的 slot 写进列）。
                    meta["slot"] = _slot
                if _touched and _touched > str(meta.get("last_used_at") or ""):
                    # 3a（2026-11-24）：TOUCH 事件记着最后使用时间，
                    # 重建时带回来，不退回 add 时的值（P1-6 修掉）。
                    meta["last_used_at"] = _touched
                if _mark:
                    meta["decay_state"] = _mark
                vec = self._embed_text(text)
                self._insert_memory_row(key, text, meta, vec)
                added += 1
            return added

    def repair_projection_from_ledger(self):
        """按账本修已有行的半截状态（崩溃恢复，追加决定十五）。

        账本是唯一真相源、SQLite 是投影：两步写之间硬崩溃留下的
        半截状态，一律按账本折叠态修正投影，绝不反过来改账本
        （账本只追加、不改写历史行）。修的窗口：
          - 顶替（_supersede_row 先标 SQLite 再写账本）：进程死在
            两步之间 → SQLite 有 superseded 标记、账本没事件 →
            按账本撤标记；反过来账本有顶替/删除事件、SQLite 没标
            （如恢复了旧备份的库）→ 按事件补标；
          - 晋升/保护档/抽屉/槽位/沉底/最后使用时间：都是账本
            先行、SQLite 后写，死在中间 → 按账本最新一笔覆盖
            补齐（字段口径与 reconcile_ledger_index 重建缺失行
            完全一致，同一份 _ledger_overlays）；
          - 作废（voided）行不碰标记：作废是终局标记，它自己
            也有 delete 账本事件兜底，检索口径只认 voided。
        SQLite 里有、账本里没有 add 的行：不删、不改，只计数
        报警——来历不明删了就是丢记忆，等 BOOS 裁（账本本身缺
        行由 verify_ledger 的序号缺口另行报警）。
        返回报告 dict；内部异常向上抛，调用方（启动恢复）拦。"""
        with self._lock:
            report = {
                "rows_checked": 0,
                "superseded_fixed": 0,
                "deleted_fixed": 0,
                "meta_fixed": 0,
                "extra_rows": 0,
            }
            state = self.ledger_state()
            key_states = (state or {}).get("key_states") or {}
            by_id = (state or {}).get("by_id") or {}
            touch_map = (state or {}).get("last_touched") or {}
            known_keys = set(key_states) | set(by_id)
            for _record in (state or {}).get("adds") or []:
                if _record.get("key"):
                    known_keys.add(_record["key"])
            try:
                promote_meta, decay_marks, slot_map, revived_keys = (
                    self._ledger_overlays()
                )
            except Exception:
                logger.warning(
                    "读账本覆盖层失败：本次只修状态标记。", exc_info=True
                )
                promote_meta, decay_marks, slot_map, revived_keys = (
                    {}, {}, {}, set()
                )
            rows = self._conn.execute(
                "SELECT id, metadata_json, superseded, deleted, voided,"
                " user_confirmed"
                " FROM memories"
            ).fetchall()
            for rid, meta_json, sup, dele, voided, uc in rows:
                report["rows_checked"] += 1
                if rid not in known_keys:
                    report["extra_rows"] += 1
                    logger.warning(
                        "启动修复：账本里没有的行（不删不改，等 BOOS 裁）：%s",
                        rid,
                    )
                    continue
                expected = key_states.get(rid)
                if expected is None:
                    expected = (by_id.get(rid) or {}).get("state")
                new_sup, new_del = sup, dele
                if not voided and expected is not None:
                    # 按折叠终态精确对齐（2026-11-25 互查修复三）：
                    # 终态是哪个，标记就是哪个，另一个必须清掉。
                    # 旧写法不对称——expected=superseded 会清
                    # deleted，反过来 expected=deleted 不清
                    # superseded，能留下 deleted=1 且 superseded=1
                    # 的双标行。voided 行照旧不碰：作废是投影侧
                    # 终局标记，账本折叠态里没有对应状态（见方法
                    # docstring），只认它自己的事件兜底。
                    if expected == _ledger.LEDGER_STATE_SUPERSEDED:
                        new_sup, new_del = 1, 0
                    elif expected == _ledger.LEDGER_STATE_DELETED:
                        new_sup, new_del = 0, 1
                    elif expected in (
                        _ledger.LEDGER_STATE_CURRENT,
                        _ledger.LEDGER_STATE_DORMANT,
                    ):
                        new_sup, new_del = 0, 0
                try:
                    meta = json.loads(meta_json or "{}")
                except (ValueError, TypeError):
                    meta = {}
                new_meta = meta
                _promoted = promote_meta.get(rid) or {}
                _slot = slot_map.get(rid)
                _touched = touch_map.get(rid)
                _mark = decay_marks.get(rid)
                if _promoted or _slot or _touched or _mark or (
                    rid in revived_keys
                ):
                    new_meta = dict(meta)
                    if _promoted:
                        for _field in (
                            "source", "user_confirmed", "decay_kind",
                            "importance", "tone", "tone_evidence",
                            "tone_source",
                        ):
                            if _promoted.get(_field) is not None:
                                new_meta[_field] = _promoted[_field]
                        # 核心上限字段：以「键在不在」为准（同
                        # reconcile）——显式写 None 是摘档/摘格。
                        for _field in ("protection_tier", "core_drawer"):
                            if _field in _promoted:
                                new_meta[_field] = _promoted[_field]
                    if _slot:
                        # 槽位与同函数其他字段同口径：有最新 SLOT
                        # 事件就无条件按最新一笔覆盖（账本为准）。
                        # 旧写法只填空——meta 里先有别的槽位时，
                        # 账本的新标永远盖不上，投影与账本长期
                        # 对不齐（2026-11-25 互查修复一）。
                        new_meta["slot"] = _slot
                    if _touched and _touched > str(
                        new_meta.get("last_used_at") or ""
                    ):
                        new_meta["last_used_at"] = _touched
                    if _mark:
                        new_meta["decay_state"] = _mark
                    elif rid in revived_keys and new_meta.get(
                        "decay_state"
                    ) in (
                        _decay.DECAY_STATE_DORMANT,
                        _decay.DECAY_STATE_DELETE_DUE,
                    ):
                        new_meta["decay_state"] = _decay.DECAY_STATE_VERBATIM
                meta_changed = new_meta != meta
                # user_confirmed 独立列无条件核对（贵妃第十八
                # 份复查 F）：demote 两段写之间硬崩会留下 meta
                # 已与账本一致、列还停在旧值的分叉；旧写法只
                # 在旗或 meta 变化时才连列一起 UPDATE，这种
                # 分叉永远修不到。列与修复后 meta 对不上就强
                # 制进 UPDATE（UPDATE 本来就写这一列），计数
                # 并入 meta_fixed，报告键集不变。voided 行照
                # 旧不碰。
                column_diverged = (
                    not voided
                    and (1 if uc else 0)
                    != (1 if new_meta.get("user_confirmed") else 0)
                )
                if new_sup != sup:
                    report["superseded_fixed"] += 1
                if new_del != dele:
                    report["deleted_fixed"] += 1
                if meta_changed or column_diverged:
                    report["meta_fixed"] += 1
                if (
                    new_sup != sup or new_del != dele
                    or meta_changed or column_diverged
                ):
                    self._conn.execute(
                        "UPDATE memories SET superseded=?, deleted=?,"
                        " metadata_json=?, decay_kind=?, source=?,"
                        " user_confirmed=?, importance=?, slot=?"
                        " WHERE id=?",
                        (
                            new_sup,
                            new_del,
                            json.dumps(new_meta, ensure_ascii=False),
                            new_meta.get("decay_kind"),
                            new_meta.get("source"),
                            1 if new_meta.get("user_confirmed") else 0,
                            new_meta.get("importance"),
                            new_meta.get("slot"),
                            rid,
                        ),
                    )
            self._conn.commit()
            return report

    def rebuild_pending_questions(self):
        """待问文件损坏后按存活记录重建（崩溃恢复，追加决定十五）。

        只在 pending_questions 报告文件坏过（load_error）时由启动
        恢复调用。能重建的只有两类——它们在别处留了痕：
          - 转述授权声称：记忆行 meta 里还带着
            auth_claim=relayed_unverified，按原队列函数重新攒；
          - 入口挂起：intake_log.jsonl 里有 HOLD 留痕、且原文
            还没被存成活记忆的，按原队列函数重新攒。
        其余种类（importance / conflict / core_capacity /
        change_attempt 留痕）的问题、以及「已答过」的裁决痕迹，
        只活在问题文件里，文件没了就没了——这里明说不重建，
        不许假装全恢复了。返回重建计数 dict。"""
        rebuilt = {"auth_claim": 0, "intake_hold": 0}
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall()
            live_texts = set()
            for rid, text, meta_json in rows:
                live_texts.add(_normalize_memory_text(text or ""))
                try:
                    meta = json.loads(meta_json or "{}")
                except (ValueError, TypeError):
                    meta = {}
                if meta.get("auth_claim") == _auth_guard.AUTH_CLAIM_RELAYED:
                    before = self.pending_questions.count_unanswered()
                    self._queue_auth_question(
                        text or "",
                        rid,
                        "待问文件损坏后按记忆行标记重建（崩溃恢复）。",
                    )
                    if self.pending_questions.count_unanswered() > before:
                        rebuilt["auth_claim"] += 1
            try:
                held_entries = []
                if self._intake_log_path.exists():
                    for _line in self._intake_log_path.read_text(
                        encoding="utf-8"
                    ).splitlines():
                        try:
                            _entry = json.loads(_line)
                        except (ValueError, TypeError):
                            continue
                        if (
                            _entry.get("verdict") == _intake.VERDICT_HOLD
                            and _entry.get("text")
                        ):
                            held_entries.append(_entry)
                for _entry in held_entries:
                    _text = _entry["text"]
                    if _normalize_memory_text(_text) in live_texts:
                        continue  # 已存成活记忆：挂起已经裁完了
                    before = self.pending_questions.count_unanswered()
                    self._queue_intake_hold(
                        _text,
                        _entry.get("category") or "",
                        _entry.get("reason")
                        or "待问文件损坏后按入口留痕重建（崩溃恢复）。",
                        turn_id=_entry.get("turn_id"),
                    )
                    if self.pending_questions.count_unanswered() > before:
                        rebuilt["intake_hold"] += 1
            except Exception:
                logger.warning(
                    "按入口留痕重建挂起问题失败。", exc_info=True
                )
        rebuilt["unrecoverable_kinds"] = [
            _pq.QUESTION_KIND_IMPORTANCE,
            _pq.QUESTION_KIND_CONFLICT,
            _pq.QUESTION_KIND_CORE_CAPACITY,
            "auth_claim/change_attempt",
        ]
        return rebuilt

    def startup_recovery(self):
        """启动崩溃恢复（追加决定十五）：verify → repair（账本
        残行）→ 按账本修投影 → 待问文件痕迹与重建，最后才轮到
        create_sqlite_store 里的 reconcile 补缺行。顺序是死规矩：
        账本残行不先修掉，折叠态就少最后一截，后面的修复全对
        不齐。每步失败只记日志、不拦启动（记忆系统出问题不能
        把聊天也堵死）。账本序号缺口只报警不硬修：缺的事件编
        不出来，硬补就是造假账——留给 BOOS/自检裁（同
        verify_ledger 原口径）。返回报告 dict。"""
        report = {
            "verify": None,
            "ledger_repair": None,
            "projection": None,
            "pending": None,
            "pending_rebuilt": None,
        }
        try:
            report["verify"] = _ledger.verify_ledger(self.ledger_path)
            if not (report["verify"] or {}).get("ok"):
                logger.error(
                    "启动恢复：账本校验不干净（缺口/残行/坏行）：%s",
                    report["verify"],
                )
        except Exception:
            logger.warning("启动恢复：账本校验失败。", exc_info=True)
        try:
            report["ledger_repair"] = _ledger.repair_ledger(
                self.ledger_path
            )
        except Exception:
            logger.warning("启动恢复：账本残行修复失败。", exc_info=True)
        try:
            report["projection"] = self.repair_projection_from_ledger()
        except Exception:
            logger.warning("启动恢复：投影修复失败。", exc_info=True)
        try:
            report["pending"] = self.pending_questions.recovery_report()
            if report["pending"]:
                report["pending_rebuilt"] = self.rebuild_pending_questions()
                logger.error(
                    "启动恢复：待问文件曾损坏（原件保全于 %s），"
                    "按存活记录重建 %s；其余种类与已答痕迹无法重建。",
                    report["pending"].get("backup_path"),
                    report["pending_rebuilt"],
                )
        except Exception:
            logger.warning("启动恢复：待问文件恢复失败。", exc_info=True)
        # 恢复健康度收尾（立项 H6）：结论落盘，不干净时冻结
        # 危险写（见 startup_recovery 之后的 H6 节）。收尾永
        # 不抛——健康度落盘失败不能反过来堵住启动。
        try:
            self._finalize_recovery_health(report)
        except Exception:
            logger.warning("启动恢复：健康度收尾失败。", exc_info=True)
        # 性情区一致性校验（追加决定八第 ⑤ 环）：启动时对一遍
        # 壳↔复印件，冲突只挂待问、不覆盖任何一边（校验函数
        # 内处理）；没行/只有种子行不吵。永不抛——校验失败
        # 不能反过来堵住启动。
        try:
            report["persona"] = self._persona_startup_check()
        except Exception:
            logger.warning("启动恢复：性情区校验失败。", exc_info=True)
        return report


    # ---------- 恢复健康度（立项 H6，2026-11-25） ----------
    #
    # 三件：健康度落盘、结论页映射、危险写冻结。
    #
    # 【落盘】每次 startup_recovery 收尾把健康结论写进
    # memory 目录下 recovery_health.json（{version,
    # updated_at, latest, history}，history 留最近 10 份、
    # 最新在前，原子换名不留半截 JSON）。进程重启后读得出
    # 上一份——补决定十五的缺口（待问 load_error 只活在进
    # 程里，重启即消失、报警无留存）。
    #
    # 【结论页映射】沿既有消费通道接，不新造页面：
    #   - recovery_health() 返回最近一份记录（本进程产出的
    #     优先，没有读落盘的上一份），结论页/健康检查从同
    #     一份记录取数；
    #   - pending_notices() 新增 kind=recovery_health：状态
    #     为 frozen 时必出（带 residuals 种类列表与
    #     frozen_reason）；本进程还没产出自己的记录时，上
    #     一进程落盘的未净记录（attention/frozen）也出——
    #     跨重启可见。字段对应：record.status → 总状态；
    #     record.residuals[].kind → 异常项；
    #     record.controlled_entry.frozen/frozen_reason → 冻
    #     结状态与原因；record.recovery.* → 恢复各步明细；
    #     record.controlled_entry.
    #     protection_evidence_pending_count → H5 待补证计数；
    #   - 冻结本身另沿既有 kind=controlled_entry_frozen 露
    #     头（reason=recovery_unclean）。attention 级残余各
    #     有既有 kind 露头（pending_questions_corrupt /
    #     protection_evidence_pending /
    #     controlled_entry_health），不在 recovery_health
    #     条里重复汇总。
    #
    # 【危险写冻结】不另造第二套：仍是
    # _controlled_frozen_reason 这一面旗、submit/revoke 同
    # 一道闸口（执行路径拒 entry_frozen，影子与拒绝照常）。
    # 恢复不干净时经 _controlled_note_health(
    # "recovery_unclean", ...) 置旗（已并入
    # _CONTROLLED_FREEZE_KINDS；置过不覆盖、保留最早原因的
    # 既有口径不变）。解冻只许两种：① 下一次
    # startup_recovery 结论无冻结级残余；②
    # recovery_health_recheck() 显式只读复检通过。两种都
    # 只解 recovery_unclean 这一来源；持久化故障类冻结
    # （receipts_load_error 等）照旧不自动解、复检也不碰。
    # 跨重启承接：构造时读回上一份记录，status=frozen 且
    # frozen_reason=recovery_unclean 的，先承接冻结等本轮
    # 恢复重新裁决（_load_recovery_health）。
    #
    # 【不干净判据（本轮核定）】
    # 冻结级：verify 缺失（步骤抛错）或 ok 非 True（账本序
    # 号缺口/残行/坏行——账本本身不可信）；ledger_repair /
    # projection 步骤抛错（报告字段为 None）；投影修复后
    # 复检仍有分叉（行状态旗对、保护档/抽屉、槽位、
    # user_confirmed 列与 meta 分叉，与账本折叠不符，复检
    # 口径与 repair_projection_from_ledger 对
    # 应段一致，只查受控入口安全所系字段）；账本里没有的
    # 行 extra_rows > 0（repair 明定不删不改、等 BOOS
    # 裁——投影与账本对不上且恢复自己解不开，冻到人工处
    # 置后靠干净恢复/复检解冻）；待问文件损坏且重建没跑
    # 成（pending 非空但 pending_rebuilt 为 None）。
    # 注意级（记录、不冻结）：待问损坏但已重建；H5 待补证
    # 计数 > 0；受控入口非冻结类健康痕迹非空。冻结类痕迹
    # （不变量失败/提交待核实/持久化故障/对账未定案——补
    # 修二轮起同在 _CONTROLLED_FREEZE_KINDS）不算注意级：
    # 它们经 _controlled_frozen_reason 直接定 frozen，并在
    # 残余清单里以 controlled_entry_frozen 冻结级列明。
    # status：无残余 = clean；只有注意级 = attention；有
    # 冻结级残余或入口已冻结 = frozen。

    _RECOVERY_HEALTH_HISTORY_KEEP = 10
    _RECOVERY_FREEZE_REASON = "recovery_unclean"

    def _projection_residual_mismatches(self):
        """投影修复后的只读复检（H6）：拿账本折叠比对行状
        态旗对（superseded/deleted）、保护档/抽屉、槽位——
        口径与 repair_projection_from_ledger 的对应段一致，
        只查不修。返回 {"mismatch_count", "sample_ids",
        "extra_rows"}。voided 行照 repair 口径不碰、不算分
        叉；账本里没有的行单记 extra_rows，不混进分叉数。
        内部异常向上抛，调用方拦。"""
        with self._lock:
            state = self.ledger_state() or {}
            key_states = state.get("key_states") or {}
            by_id = state.get("by_id") or {}
            known_keys = set(key_states) | set(by_id)
            for _record in state.get("adds") or []:
                if _record.get("key"):
                    known_keys.add(_record["key"])
            promote_meta, _decay_marks, slot_map, _revived = (
                self._ledger_overlays()
            )
            mismatched = []
            extra_rows = 0
            rows = self._conn.execute(
                "SELECT id, metadata_json, superseded, deleted, voided,"
                " user_confirmed"
                " FROM memories"
            ).fetchall()
            for rid, meta_json, sup, dele, voided, uc in rows:
                if rid not in known_keys:
                    extra_rows += 1
                    continue
                if voided:
                    continue
                expected = key_states.get(rid)
                if expected is None:
                    expected = (by_id.get(rid) or {}).get("state")
                bad = False
                if expected is not None:
                    if expected == _ledger.LEDGER_STATE_SUPERSEDED:
                        want = (1, 0)
                    elif expected == _ledger.LEDGER_STATE_DELETED:
                        want = (0, 1)
                    elif expected in (
                        _ledger.LEDGER_STATE_CURRENT,
                        _ledger.LEDGER_STATE_DORMANT,
                    ):
                        want = (0, 0)
                    else:
                        want = None
                    if want is not None and (sup, dele) != want:
                        bad = True
                try:
                    meta = json.loads(meta_json or "{}")
                except (ValueError, TypeError):
                    meta = {}
                if not bad and (1 if uc else 0) != (
                    1 if meta.get("user_confirmed") else 0
                ):
                    # user_confirmed 独立列与 meta 分叉（复查
                    # F）：demote 两段写中间窗口的残留，复检必
                    # 须看得见——旧复检根本不查这一列。
                    bad = True
                if not bad:
                    promoted = promote_meta.get(rid) or {}
                    for _field in ("protection_tier", "core_drawer"):
                        if _field in promoted and (
                            meta.get(_field) != promoted[_field]
                        ):
                            bad = True
                            break
                    if not bad:
                        _slot = slot_map.get(rid)
                        if _slot and meta.get("slot") != _slot:
                            bad = True
                if bad:
                    mismatched.append(rid)
            return {
                "mismatch_count": len(mismatched),
                "sample_ids": mismatched[:10],
                "extra_rows": extra_rows,
            }

    def _recovery_residuals(self, report, residual, recheck=False):
        """按 H6 节头判据组装残余异常清单。residual 是
        _projection_residual_mismatches 的返回（None = 复检
        没跑成，本身按冻结级记）。recheck=True 时
        ledger_repair/projection 两步不在复检范围，报告里
        的占位不作数、跳过对应判据。返回 (residuals,
        freezing)：residuals 每条 {kind, detail, freezing}，
        freezing = 有没有冻结级残余。"""
        residuals = []
        verify = (report or {}).get("verify")
        if verify is None:
            residuals.append({
                "kind": "ledger_verify_failed",
                "detail": "账本校验步骤没跑成（报告缺失）",
                "freezing": True,
            })
        elif not verify.get("ok"):
            residuals.append({
                "kind": "ledger_verify_failed",
                "detail": {
                    "gaps": verify.get("gaps"),
                    "truncated": verify.get("truncated"),
                    "bad_lines": verify.get("bad_lines"),
                },
                "freezing": True,
            })
        if not recheck:
            if (report or {}).get("ledger_repair") is None:
                residuals.append({
                    "kind": "ledger_repair_failed",
                    "detail": "账本残行修复步骤没跑成（报告缺失）",
                    "freezing": True,
                })
            if (report or {}).get("projection") is None:
                residuals.append({
                    "kind": "projection_repair_failed",
                    "detail": "投影修复步骤没跑成（报告缺失）",
                    "freezing": True,
                })
        if residual is None:
            residuals.append({
                "kind": "projection_residual_mismatch",
                "detail": "投影修复后复检没跑成",
                "freezing": True,
            })
        elif residual.get("mismatch_count"):
            residuals.append({
                "kind": "projection_residual_mismatch",
                "detail": {
                    "count": residual["mismatch_count"],
                    "sample_ids": residual["sample_ids"],
                },
                "freezing": True,
            })
        extra = None
        projection = (report or {}).get("projection")
        if isinstance(projection, dict) and projection.get(
            "extra_rows"
        ) is not None:
            extra = projection.get("extra_rows") or 0
        elif residual is not None:
            extra = residual.get("extra_rows") or 0
        if extra:
            residuals.append({
                "kind": "projection_extra_rows",
                "detail": {"count": extra},
                "freezing": True,
            })
        pending = (report or {}).get("pending")
        if pending:
            if (report or {}).get("pending_rebuilt") is None:
                residuals.append({
                    "kind": "pending_questions_unrecovered",
                    "detail": {
                        "backup_path": pending.get("backup_path"),
                        "reason": pending.get("reason"),
                    },
                    "freezing": True,
                })
            else:
                residuals.append({
                    "kind": "pending_questions_corrupt",
                    "detail": {
                        "backup_path": pending.get("backup_path"),
                        "reason": pending.get("reason"),
                    },
                    "freezing": False,
                })
        if getattr(self, "_protection_evidence_pending", []):
            residuals.append({
                "kind": "protection_evidence_pending",
                "detail": {
                    "count": len(self._protection_evidence_pending),
                },
                "freezing": False,
            })
        other_issues = [
            i for i in getattr(self, "_controlled_health_issues", [])
            if i.get("kind") != self._RECOVERY_FREEZE_REASON
        ]
        # 补修二轮（复查 A）：冻结类痕迹（不变量失败/提交待
        # 核实/持久化故障/对账未定案）在残余清单里定冻结级，
        # 不再混进注意级的 controlled_entry_health——冻结事
        # 实以 _controlled_frozen_reason 为准，这里只是把同
        # 一事实如实列进恢复结论页。
        freeze_issues = [
            i for i in other_issues
            if i.get("kind") in self._CONTROLLED_FREEZE_KINDS
        ]
        plain_issues = [
            i for i in other_issues
            if i.get("kind") not in self._CONTROLLED_FREEZE_KINDS
        ]
        if freeze_issues:
            residuals.append({
                "kind": "controlled_entry_frozen",
                "detail": {
                    "count": len(freeze_issues),
                    "kinds": [i.get("kind") for i in freeze_issues],
                },
                "freezing": True,
            })
        if plain_issues:
            residuals.append({
                "kind": "controlled_entry_health",
                "detail": {
                    "count": len(plain_issues),
                    "kinds": [i.get("kind") for i in plain_issues],
                },
                "freezing": False,
            })
        freezing = any(r["freezing"] for r in residuals)
        return residuals, freezing

    def _finalize_recovery_health(self, report, source="startup_recovery"):
        """恢复健康度收尾（H6）：复检投影残余 → 组装残余
        → 冻结/解冻裁决 → 组装记录并落盘。返回健康记录
        dict。本函数自己拦异常（落盘失败只记痕、不抛），
        调用方（startup_recovery）外层再兜一层。"""
        try:
            residual = self._projection_residual_mismatches()
        except Exception:
            logger.warning(
                "恢复健康度：投影残余复检失败。", exc_info=True
            )
            residual = None
        residuals, freezing = self._recovery_residuals(
            report, residual, recheck=(source == "recheck")
        )
        unfrozen = None
        if freezing:
            # recovery_unclean 只记恢复自身判出的冻结级残余；
            # 受控入口自身的冻结类痕迹（controlled_entry_
            # frozen）不是恢复的账，别替它背、也别在恢复重
            # 启时替它解冻（它有自己的解冻条件）。
            recovery_caused = any(
                r["freezing"] and r["kind"] != "controlled_entry_frozen"
                for r in residuals
            )
            already = any(
                i.get("kind") == self._RECOVERY_FREEZE_REASON
                for i in self._controlled_health_issues
            )
            if recovery_caused and not already:
                self._controlled_note_health(
                    self._RECOVERY_FREEZE_REASON,
                    {
                        "residual_kinds": [
                            r["kind"] for r in residuals if r["freezing"]
                        ],
                        "source": source,
                    },
                )
        elif (
            self._controlled_frozen_reason
            == self._RECOVERY_FREEZE_REASON
        ):
            # 解冻只解 recovery_unclean 这一来源（判据：本轮
            # 无冻结级残余）；持久化故障类冻结不在这里动。
            self._controlled_frozen_reason = None
            unfrozen = {"previous_reason": self._RECOVERY_FREEZE_REASON}
        frozen_now = bool(self._controlled_frozen_reason)
        if frozen_now:
            status = "frozen"
        elif residuals:
            status = "attention"
        else:
            status = "clean"
        record = {
            "version": 1,
            "recorded_at": _now_iso(),
            "source": source,
            "status": status,
            "clean": status == "clean",
            "recovery": {
                "verify": (report or {}).get("verify"),
                "ledger_repair": (report or {}).get("ledger_repair"),
                "projection": (report or {}).get("projection"),
                "projection_residual_mismatches": (
                    residual.get("mismatch_count")
                    if residual else None
                ),
                "pending_corrupt": (report or {}).get("pending"),
                "pending_rebuilt": (report or {}).get("pending_rebuilt"),
            },
            "residuals": residuals,
            "controlled_entry": {
                "frozen": frozen_now,
                "frozen_reason": self._controlled_frozen_reason,
                "health_issue_count": len(
                    getattr(self, "_controlled_health_issues", [])
                ),
                "protection_evidence_pending_count": len(
                    getattr(self, "_protection_evidence_pending", [])
                ),
            },
        }
        if unfrozen:
            record["unfrozen"] = unfrozen
        self._recovery_health_latest = record
        self._recovery_health_own = True
        self._persist_recovery_health(record)
        return record

    def _persist_recovery_health(self, record):
        """健康记录落盘（H6）：recovery_health.json =
        {version, updated_at, latest, history}，history 留
        最近 10 份（最新在前）。临时文件＋replace 原子换
        名，不留半截 JSON。落盘失败只记日志与 load_error
        痕迹、不抛（不堵启动的既有口径）。"""
        with self._recovery_health_lock:
            try:
                history = []
                try:
                    raw = self._recovery_health_path.read_text(
                        encoding="utf-8"
                    )
                    data = json.loads(raw)
                    if isinstance(data, dict) and isinstance(
                        data.get("history"), list
                    ):
                        history = [
                            h for h in data["history"]
                            if isinstance(h, dict)
                        ]
                except Exception:
                    history = []
                history = [record] + history
                history = history[: self._RECOVERY_HEALTH_HISTORY_KEEP]
                payload = {
                    "version": 1,
                    "updated_at": _now_iso(),
                    "latest": record,
                    "history": history,
                }
                tmp = self._recovery_health_path.parent / (
                    self._recovery_health_path.name + ".tmp"
                )
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                os.replace(tmp, self._recovery_health_path)
            except Exception as exc:
                self._recovery_health_load_error = "%s: %s" % (
                    type(exc).__name__, exc,
                )
                logger.warning(
                    "恢复健康度落盘失败：%s", exc, exc_info=True
                )

    def _load_recovery_health(self):
        """构造时读回上一份健康记录（H6，永不抛）：进程重
        启后上一轮的恢复结论还在（recovery_health() 与
        pending_notices 跨重启取数）。上一份若 status=
        frozen 且冻结来源是 recovery_unclean，先承接冻结
        ——等本进程的 startup_recovery 重新裁决（干净才
        解，见 _finalize_recovery_health）；持久化故障类冻
        结不靠这里承接，它们在 _load_controlled_entry_state
        里按文件现状重新派生。"""
        try:
            raw = self._recovery_health_path.read_text(
                encoding="utf-8"
            )
            data = json.loads(raw)
            latest = (
                data.get("latest") if isinstance(data, dict) else None
            )
            if isinstance(latest, dict):
                self._recovery_health_latest = latest
                entry = latest.get("controlled_entry") or {}
                if (
                    latest.get("status") == "frozen"
                    and entry.get("frozen_reason")
                    == self._RECOVERY_FREEZE_REASON
                ):
                    self._controlled_note_health(
                        self._RECOVERY_FREEZE_REASON,
                        {
                            "carried_from": latest.get("recorded_at"),
                            "residual_kinds": [
                                r.get("kind")
                                for r in latest.get("residuals") or []
                                if r.get("freezing")
                            ],
                        },
                    )
        except FileNotFoundError:
            pass
        except Exception as exc:
            self._recovery_health_load_error = "%s: %s" % (
                type(exc).__name__, exc,
            )
            logger.warning("读回恢复健康记录失败。", exc_info=True)

    def recovery_health(self):
        """最近一份恢复健康记录（H6）：本进程产出的优先，
        没有就读落盘的上一份；从来没有过返回 None。深拷
        贝，调用方改它污染不了登记原件。结论页/健康检查
        从这一份取数（映射见 H6 节头）。"""
        latest = getattr(self, "_recovery_health_latest", None)
        if latest is None:
            try:
                raw = self._recovery_health_path.read_text(
                    encoding="utf-8"
                )
                data = json.loads(raw)
                if isinstance(data, dict) and isinstance(
                    data.get("latest"), dict
                ):
                    latest = data["latest"]
                    self._recovery_health_latest = latest
            except Exception:
                latest = None
        return copy.deepcopy(latest) if latest is not None else None

    def recovery_health_recheck(self):
        """显式只读复检（H6）：不跑修复，只重验账本、复检
        投影残余、看待问损坏痕迹。干净且当前冻的是
        recovery_unclean 时解冻，并落一条 source=recheck
        的健康记录；不干净、或冻的是持久化故障类来源时，
        只落记录、不解冻（那种冻结照旧等对账/人工处置）。
        待问一项复检不重判重建：痕迹在时按最近一份记录的
        重建结果记账（没有记录可依就按未恢复记，宁严勿
        松）。返回健康记录 dict。"""
        report = {
            "verify": None,
            "ledger_repair": {"recheck": True, "repaired": False,
                              "dropped": 0},
            "projection": None,
            "pending": None,
            "pending_rebuilt": None,
        }
        try:
            report["verify"] = _ledger.verify_ledger(self.ledger_path)
        except Exception:
            logger.warning("健康复检：账本校验失败。", exc_info=True)
        try:
            report["pending"] = self.pending_questions.recovery_report()
        except Exception:
            report["pending"] = None
        if report["pending"]:
            latest = getattr(self, "_recovery_health_latest", None)
            if latest:
                report["pending_rebuilt"] = (
                    (latest.get("recovery") or {}).get("pending_rebuilt")
                )
        return self._finalize_recovery_health(
            report, source="recheck"
        )

    def import_from_mem0(self, mem0_store):
        """一次性迁移：把只活在 Chroma 里的旧记忆搬进来（2026-11-21 核实）。

        沙盒账本只有 5 条当前态，Chroma 里有 31 条——2026-11-07 至
        11-17 存的 27 条在账本里根本没有 add 记录（那时还没接账本）。
        只靠账本对账会丢掉 26 条，验收必退化，所以迁移必须多这一步：
        逐条读 mem0 的 get_all，按原文比对——SQLite 里已有同一原文的
        跳过；账本里有同一原文的 delete 事件的跳过（删了的就是删了）；
        其余的先给账本补记一笔 add（删前补记同款做法），再进 SQLite。
        幂等：跑两次也不会重复（原文比对 + 账本 key 去重）。
        返回 {"imported": n, "skipped_existing": n, "skipped_deleted": n, "skipped_sensitive": n}。
        """
        with self._lock:
            deleted_texts = set()
            try:
                events, _info = _ledger.read_events(self.ledger_path)
                if self._controlled_ledger_doubtful(_info):
                    # 账本残缺（补修 H3 复查 F2）：删除过滤
                    # 集合不可信——漏掉的 delete 事件会让已
                    # 删记忆被重新迁入。fail-closed：整笔迁
                    # 移中止、如实回报。
                    logger.warning(
                        "账本残缺：一次性迁移中止（删除过滤不可信）。"
                    )
                    return {
                        "imported": 0,
                        "skipped_existing": 0,
                        "skipped_deleted": 0,
                        "skipped_sensitive": 0,
                        "ledger_doubtful": True,
                    }
                for event in events:
                    if event.get("type") == "delete":
                        text = (event.get("text") or "").strip()
                        if text:
                            deleted_texts.add(text)
            except Exception:
                logger.warning("读账本删记录失败：迁移时不按删除过滤。", exc_info=True)
            existing_texts = {
                (row[0] or "").strip()
                for row in self._conn.execute(
                    "SELECT text FROM memories WHERE voided=0 AND deleted=0 AND superseded=0"
                ).fetchall()
            }
            imported = skipped_existing = skipped_deleted = skipped_sensitive = 0
            items = mem0_store.get_all(include_sensitive=True) or []
            for item in items:
                text = ((item or {}).get("memory") or (item or {}).get("text") or "").strip()
                if not text:
                    continue
                if _contains_sensitive(text):
                    # 迁移时也拦敏感行（2026-11-21 Gemini 第三轮，BOOS 定）：
                    # 旧库里的密码/卡号不进账本、不进 SQLite。
                    skipped_sensitive += 1
                    continue
                if text in existing_texts:
                    skipped_existing += 1
                    continue
                if text in deleted_texts:
                    skipped_deleted += 1
                    continue
                meta_in = (item or {}).get("metadata") or {}
                metadata = {
                    "source": meta_in.get("source", "inferred"),
                    "confidence": meta_in.get("confidence", 0.7),
                    "user_confirmed": bool(meta_in.get("user_confirmed", False)),
                    "created_at": meta_in.get("created_at") or _now_iso(),
                    "write_seq": meta_in.get("write_seq") or _next_write_seq(),
                    "turn_id": meta_in.get("turn_id"),
                    "slot": meta_in.get("slot"),
                    "is_rule": bool(meta_in.get("is_rule", False)),
                    "curated": bool(meta_in.get("curated", False)),
                    "decay_kind": meta_in.get("decay_kind", "episodic"),
                    "importance": meta_in.get("importance", "normal"),
                    "event_date": meta_in.get("event_date")
                    or resolve_event_date(text),
                }
                try:
                    self._ledger_append_add(text, metadata)
                except Exception:
                    logger.warning("迁移补记账本失败：跳过这一条。", exc_info=True)
                    continue
                vec = self._embed_text(text)
                key = _ledger.ledger_event_key(text, metadata.get("created_at"))
                self._insert_memory_row(key, text, metadata, vec)
                existing_texts.add(text)
                imported += 1
            return {
                "imported": imported,
                "skipped_existing": skipped_existing,
                "skipped_deleted": skipped_deleted,
                "skipped_sensitive": skipped_sensitive,
            }

    # ---------- 写（原样先存） ----------

    def _insert_memory_row(self, key, text, metadata, vector):
        import numpy as np

        blob = None
        if vector is not None:
            blob = np.asarray(vector, dtype="float32").tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO memories"
            "(id, text, metadata_json, created_at, write_seq, slot, source,"
            " user_confirmed, decay_kind, importance, event_date, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                text,
                json.dumps(metadata, ensure_ascii=False),
                metadata.get("created_at") or _now_iso(),
                metadata.get("write_seq"),
                metadata.get("slot"),
                metadata.get("source"),
                1 if metadata.get("user_confirmed") else 0,
                metadata.get("decay_kind"),
                metadata.get("importance"),
                metadata.get("event_date"),
                blob,
            ),
        )
        self._conn.commit()

    def add_memory_text(self, text, layers=None, source="user", turn_id=None,
                        event_date=None, **kwargs):
        """原文存一条记忆（“记住”直存、种子、规则都走这里）。

        先写账本、再写 SQLite；账本写不进去就返回 {}、不写索引。
        同文去重（2026-11-22，Bug 2 治本）：同一归一化文本的活记忆
        已在库里时，不再存第二行，返回 UNCHANGED，并记一笔“又出现一次”
        （出现集合 + 晋升检查）；来源不看——已确认版和推断版是同一事实，
        确认状态只往上提（推断版在库、已确认版后到时，把已有行升级成
        已确认；已确认版在库、推断版后到时，只记出现，不降级）。
        同槽位（2026-11-22，Gemini #4）：slot 相同、文本不同的活记忆
        是同一件事的新版本，按 provenance._resolve_conflict 裁决：
        supersede_old 就取代旧行再存新行；pending_new/keep_old/ask
        都不存新行（分别返回 PENDING/STALE/ASK）。
        """
        text = (text or "").strip()
        if not text:
            return {}
        if _contains_sensitive(text):
            # #1 敏感信息拦截：含密码/卡号/密钥的句子不存——账本和 SQLite
            # 都不写。返回非空字典（event=SENSITIVE）："记住"直存的调用方
            # 按"已处理"跳过 add_turn，不让秘密从另一条路再存进去；
            # save_turn 按 dropped 记回执、写明原因（见下面 SENSITIVE 分支）。
            return {
                "event": "SENSITIVE",
                "memory": text,
                "reason": "含敏感信息",
            }
        # 入口当场拦（实测问题三，追加决定七）：所有落行路的共同
        # 入口先过当场拦——明确垃圾当场丢（INTAKE_DROPPED）、拿不准
        # 挂起（INTAKE_HELD、进待问），都不落记忆行，留痕在
        # intake_log。锁内的 auth_guard 授权硬判照常跑，两道门
        # 并存、互不绕过：本门判 store 的才走得到它。
        # intake_bypass 只给已裁决路径（待问答「存」后的回存、
        # 工具路与 CLI 的「记住」命令——BOOS 当轮明确让记的原语
        # 通道），外部调用方不许乱传。另：调用方显式传了
        # decay_kind 的，等于已经定了存储类别（同 judge_importance
        # 「已传 decay_kind 的不问」的既有先例），不当场拦二次
        # 干预——save_turn 的句子在逐句循环里已过合并屏，不漏。
        if (
            not kwargs.get("intake_bypass")
            and not kwargs.get("decay_kind")
            and source in ("user", "inferred")
        ):
            _verdict, _info = self._intake_screen(
                text, turn_id=turn_id, source=source,
                mark=kwargs.get("intake_mark"),
            )
            if _verdict == _intake.VERDICT_DROP:
                return {
                    "event": "INTAKE_DROPPED",
                    "memory": text,
                    "reason": _info["reason"],
                    "intake_category": _info["category"],
                }
            if _verdict == _intake.VERDICT_HOLD:
                return {
                    "event": "INTAKE_HELD",
                    "memory": text,
                    "reason": _info["reason"],
                    "intake_category": _info["category"],
                    "question_id": _info.get("question_id"),
                }
        with self._lock:
            created_at = _now_iso()
            today = _shanghai_today().isoformat()
            slot = kwargs.get("slot")
            # S10（2026-11-22，BOOS 定选 A）：没传 slot 时按文本推断——
            # 模型调 add_memory 只传文本、不传 slot，不推断就走不到下面
            # 的槽位裁决，工具路会直存出两条活住址（西安 + 长沙案）。
            inferred_slot, inferred_value = _infer_fact_slot(text)
            if not slot:
                slot = inferred_slot
            is_rule = bool(kwargs.get("is_rule", False))
            curated = bool(kwargs.get("curated", False))
            user_confirmed = source == "user"
            # 授权硬判（实测问题一，追加决定十二）：转述来的授权
            # 断言只算未核实声称——不管调用方传的 source 是什么，
            # 都不许落已确认、不许走槽位/晋升的已确认路，存完攒
            # 一条待问等 BOOS 本人裁。纯程序判定（auth_guard 正则
            # 信号），不依赖模型当场判断。
            _auth_reason = _auth_guard.detect_relayed_authorization(text)
            if _auth_reason:
                user_confirmed = False
            write_seq = _next_write_seq()
            decay_kind = kwargs.get("decay_kind") or _decay.classify_decay_kind(
                source, False, is_rule
            )
            # 重要性判断（Phase 2 #2，opt-in）：直存路（“记住”/工具）显式
            # 打开 judge_importance=True 时才问——默认不开，直存原语要
            # 确定、不花钱（和 semantic_judge 一个规矩）。有槽位的不问
            # （槽位就是核心类别，见 _decide_importance）；已传 decay_kind
            # 的不问（调用方已经定了）。判不准时先当一次性经历存，问题
            # 攒进 pending（下次聊天顺带问）。
            _importance_decision = None
            _importance_reason = ""
            # 重要性护栏的封顶留痕（webfix 二）：调用方（save_turn）
            # 已经判过重要性时把 clamped 透传进来；本路自己判时从
            # judge_info 里取。落进 meta.importance_clamped。
            _importance_clamped = kwargs.get("importance_clamped")
            if (
                kwargs.get("judge_importance")
                and not kwargs.get("decay_kind")
                and not slot
            ):
                _importance_decision, _importance_reason, _judge_info = (
                    self._decide_importance(text, slot)
                )
                if _importance_clamped is None:
                    _importance_clamped = (_judge_info or {}).get("clamped")
                if _importance_decision == "important":
                    decay_kind = _decay.DECAY_KIND_STABLE
                if _is_question_sentence(text):
                    # 问句护栏（webfix 二）：classify 对 user 来源
                    # 默认 stable，问句走判官路存进来时按护栏压回
                    # episodic——问句永远不许 stable（importance
                    # 只在 important 时才 high，问句到不了那一步）。
                    decay_kind = _decay.DECAY_KIND_EPISODIC
            if _auth_reason:
                # 未核实声称不许 stable：调用方（如 save_turn 的
                # 重要性判断）传进来的 decay_kind 一并压回 episodic。
                decay_kind = _decay.DECAY_KIND_EPISODIC
            # 同文去重（2026-11-22 Bug 2 治本）：只看归一化文本，不看来源——
            # 用户亲口说的（工具直存、save_turn）都记 source="user"（已确认），
            # 模型推断直存的才记 source="inferred"，来源不同不是两条事实。
            # 归一化比对要在 Python 里做
            # （_normalize_memory_text 剥标点/空白/主语，SQL 精确匹配认不出
            # “我住广州。”和“我住广州”是同一句），所以扫一遍活行逐条比。
            norm_text = _normalize_memory_text(text)
            if norm_text:
                for row in self._conn.execute(
                    "SELECT id, text, metadata_json FROM memories"
                    " WHERE voided=0 AND deleted=0 AND superseded=0"
                ).fetchall():
                    # 去重键比对（R2 后续③）：归一化+去全部标点。
                    # 改写句“（主食偏好是面条）”与“，主食偏好是面条”
                    # 去标点后是同一串字，算同一条，不存两份。
                    if _dedupe_key(row[1] or "") != _dedupe_key(text):
                        continue
                    try:
                        old_meta = json.loads(row[2] or "{}")
                    except Exception:
                        old_meta = {}
                    if user_confirmed and not old_meta.get("user_confirmed"):
                        # 确认状态只往上提：推断版在库、已确认版后到时，
                        # 把已有行升级成已确认（source 改 "user"、确认位
                        # 置 1、decay_kind 改 stable——用户亲口确认的事实
                        # 应该是稳定的）。
                        # P1-4（2026-11-23）：升级要补一笔账本 PROMOTE
                        #（复用已有事件类型，meta 带升级后的完整 meta）。
                        # 不记账本的话，对账重建（reconcile_ledger_index）
                        # 会按账本里旧的 add 元数据把这条重建成推断版/
                        # episodic，升级就丢了——账本是唯一真相。
                        new_meta = dict(old_meta)
                        new_meta["source"] = "user"
                        new_meta["user_confirmed"] = True
                        new_meta["decay_kind"] = _decay.DECAY_KIND_STABLE
                        self._conn.execute(
                            "UPDATE memories SET metadata_json=?, source=?,"
                            " user_confirmed=1, decay_kind=? WHERE id=?",
                            (json.dumps(new_meta, ensure_ascii=False), "user",
                             _decay.DECAY_KIND_STABLE, row[0]),
                        )
                        self._conn.commit()
                        try:
                            _ledger.append_event(
                                self.ledger_path,
                                {"type": _ledger.LEDGER_EVENT_PROMOTE,
                                 "memory_id": row[0], "key": row[0],
                                 "text": row[1] or "",
                                 "meta": dict(new_meta)},
                                lock=self._ledger_lock,
                            )
                        except Exception:
                            # 账本写不进去：把 SQLite 的升级撤回来再继续
                            #（只记出现，不升确认），两边要一致，账本是真相。
                            logger.warning("去重升级记账失败：升级已撤回。", exc_info=True)
                            self._conn.execute(
                                "UPDATE memories SET metadata_json=?, source=?,"
                                " user_confirmed=?, decay_kind=? WHERE id=?",
                                (json.dumps(old_meta, ensure_ascii=False),
                                 old_meta.get("source", "inferred"),
                                 1 if old_meta.get("user_confirmed") else 0,
                                 old_meta.get("decay_kind"), row[0]),
                            )
                            self._conn.commit()
                        else:
                            old_meta = new_meta
                    # 同一条事实又出现一次：记出现集合，够 3 天 + 3 会就晋升。
                    self._record_occurrence(row[0], old_meta, today, turn_id, text)
                    if _auth_reason:
                        self._queue_auth_question(
                            text, row[0], _auth_reason, turn_id
                        )
                        return {"id": row[0], "memory": text,
                                "event": "UNCHANGED",
                                "auth_unverified": True,
                                "auth_reason": _auth_reason}
                    return {"id": row[0], "memory": text, "event": "UNCHANGED"}
            if _auth_reason:
                # 未核实声称不进槽位裁决：不许顶掉任何已有行，
                # 也不许被当成某个槽位的新版本落核心。
                slot = None
            # 同槽位裁决：slot 相同、文本不同的活记忆是同一件事的新版本。
            # 先只裁决、不取代：pending/keep/ask 都直接返回不存；全是
            # supersede_old 时才继续存，存好新行再取代旧行（新 key 要
            # 先存在，supersede 事件的 current_key 才有东西可指）。
            to_supersede = []
            # 2026-11-23 model fix #1（D 组根因）：新陈述无纠正词时，不自动
            # supersede 同槽旧记忆——直接存成新行。只带明确纠正词的才算纠正，
            # 那种调用方显式传 allow_slot_supersede=True 才走槽位裁决。
            # 默认 False：直存原语（"记住"/工具/逐句循环）一律只存不顶。
            if slot and kwargs.get("allow_slot_supersede", False):
                # 槽位裁决的候选：slot 列上打过标的行（SQL 查）+ 按文本
                # 现认槽位的行（_find_same_slot_conflicts 扫全部活行）——
                # 修复前工具路存的行没打 slot 标（西安案），只查 SQL 会
                # 漏掉它们，两条活住址就并存了。两边按 id 去重；扫描行
                # 的 metadata 是字典，先转回 JSON 串再进同一条裁决循环。
                conflict_rows = list(self._conn.execute(
                    "SELECT id, text, metadata_json FROM memories"
                    " WHERE slot=? AND voided=0 AND deleted=0 AND superseded=0"
                    " AND text!=?",
                    (slot, text),
                ).fetchall())
                if inferred_value is not None:
                    seen_ids = {row[0] for row in conflict_rows}
                    for row in self._find_same_slot_conflicts(
                        slot, inferred_value
                    ):
                        if row[0] not in seen_ids and row[1] != text:
                            conflict_rows.append(
                                (row[0], row[1],
                                 json.dumps(row[2] or {}, ensure_ascii=False))
                            )
                # R-1 护栏：核心/置顶行不进槽位裁决名单——不顶，
                # 也不让它拦住新行落存，留着走问 BOOS 的路径。
                _kept_rows = []
                for _r in conflict_rows:
                    try:
                        _m = json.loads(_r[2] or "{}")
                    except Exception:
                        _m = {}
                    if not self._is_core_protected_row(_m, slot):
                        _kept_rows.append(_r)
                conflict_rows = _kept_rows
                for row in conflict_rows:
                    old_meta = json.loads(row[2] or "{}")
                    new_meta_probe = {
                        "source": source,
                        "user_confirmed": user_confirmed,
                        "confidence": 1.0 if source == "user" else 0.7,
                        "created_at": created_at,
                        "write_seq": write_seq,
                        "slot": slot,
                    }
                    verdict = _provenance._resolve_conflict(old_meta, new_meta_probe)
                    action = (verdict or {}).get("action")
                    if action == "supersede_old":
                        to_supersede.append((row[0], row[1], old_meta))
                    elif action == "pending_new":
                        return {"event": "PENDING", "memory": text,
                                "reason": (verdict or {}).get("reason", ""),
                                "supersedes_id": row[0]}
                    elif action == "keep_old":
                        return {"event": "STALE", "memory": text,
                                "reason": (verdict or {}).get("reason", ""),
                                "supersedes_id": row[0]}
                    else:  # ask
                        return {"event": "ASK", "memory": text,
                                "reason": (verdict or {}).get("reason", ""),
                                "supersedes_id": row[0]}
            # 混合判断（2026-11-22，BOOS 定）：程序推不出槽位（slot 为 None）
            # 的偏好句，工具直存路也可以问大模型判官——该取代就把目标加进
            # to_supersede（下面存好新行再统一取代）；模型断了/说不取代
            # 就照常直存。默认不开（opt-in，semantic_judge=True 才问）：
            # add_memory_text 是存储原语，测试/迁移/种子直存都要确定、
            # 不花钱——默认调模型会把 Q3 排名测坏（2026-11-23 真回归：
            # 直存“BOOS 喜欢吃鸡杂面。”时模型取代了纠正行，前 5 乱了）。
            # 用户当面让记的两个位置显式打开（见 tools/memory_tool.py
            # 的 add_memory、web/server.py 的“记住”直存）；save_turn
            # 逐句循环里已经判过了，调这里时传 semantic_judge=False，
            # 不重复问。
            if not slot and not _auth_reason and kwargs.get("semantic_judge", False):
                _judged, _verdict, _trig = self._judge_preference_supersede(text)
                if _judged is not None:
                    to_supersede.append(_judged)
            metadata = {
                # 授权硬判命中时来源只许落 unconfirmed（provenance
                # 规整后即最低可信档）；声称的原来源记 claimed_source。
                "source": "unconfirmed" if _auth_reason else source,
                "confidence": (
                    0.5 if _auth_reason
                    else (1.0 if source == "user" else 0.7)
                ),
                # confirmed 不存在这里：它是 user_confirmed 的短名字，
                # 读时由 _normalize_metadata 现算（和 mem0 的规矩一致）；
                # 存两份以后会打架，只留 user_confirmed 一个真相。
                "user_confirmed": user_confirmed,
                "created_at": created_at,
                "write_seq": write_seq,
                "turn_id": turn_id,
                "slot": slot,
                "is_rule": is_rule,
                "curated": curated,
                # 核心上限字段（第 1 块）：只认合法值，传错当没标
                # （普通档 / 未入格），不猜——和 _normalize_metadata
                # 一个规矩；没传记 None，老调用方行为不变。
                "protection_tier": _core_cap.normalize_protection_tier(
                    {"protection_tier": kwargs.get("protection_tier")}
                ),
                "core_drawer": _core_cap.normalize_core_drawer(
                    {"core_drawer": kwargs.get("core_drawer")}
                ),
                "decay_kind": decay_kind,
                "importance": kwargs.get(
                    "importance",
                    "high" if _importance_decision == "important" else "normal",
                ),
                "event_date": event_date or resolve_event_date(text),
                "last_used_at": created_at,
                "decay_state": _decay.DECAY_STATE_VERBATIM,
            }
            if _auth_reason:
                metadata["auth_claim"] = _auth_guard.AUTH_CLAIM_RELAYED
                metadata["auth_reason"] = _auth_reason
                metadata["claimed_source"] = source
                metadata["importance"] = "normal"
            # 语气落账（webfix 二）：本地筛当场能认出的语气（玩笑/
            # 测试/假设的明确说法）直接写进 meta——以前 SQLite 路
            # 从来不写 tone，§16 的“带标记不计晋升”在晋升侧认
            # tone、却没人写，等于空转。本地认不出的（比如反话）
            # 留给实时整理的模型判语气补标（见 _organize_turn_
            # memories 的语气步）。带标记的当场压档：不许 stable、
            # 不许 high（判官/类别判的也不行，语气优先）。
            _tone, _tone_evidence = local_tone_screen(text)
            if _tone:
                metadata["tone"] = _tone
                metadata["tone_evidence"] = _tone_evidence
                metadata["tone_source"] = "local"
                if _tone in ("joke", "test", "hypothesis"):
                    metadata["decay_kind"] = _decay.DECAY_KIND_EPISODIC
                    metadata["importance"] = "normal"
            if _importance_clamped:
                metadata["importance_clamped"] = _importance_clamped
            dates, sessions, _added = _decay.record_seen_occurrence(
                None, None, today, turn_id
            )
            metadata["seen_dates"] = _decay._dump_seen_list(dates)
            metadata["seen_sessions"] = _decay._dump_seen_list(sessions)
            key = self._append_new_memory(text, metadata)
            if key is None:
                return {}
            if _auth_reason:
                self._queue_auth_question(text, key, _auth_reason, turn_id)
            if _importance_decision == "unsure":
                # 判不准的问题攒进 pending，下次聊天顺带问（见 save_turn）。
                self._queue_importance_question(
                    text, key, _importance_reason, turn_id
                )
            if len(to_supersede) > 1:
                to_supersede = sorted(
                    to_supersede,
                    key=lambda c: (
                        (c[2] or {}).get("write_seq") or 0,
                        (c[2] or {}).get("created_at") or "",
                    ),
                    reverse=True,
                )
            # R-1（2026-11-24，BOOS 定“像十一”）：裁决判该顶的不再
            # 只顶最新一条——主目标（最新那条）照旧直接顶；其余走
            # 连带清扫的门槛（只清对话里亲口说过的，防广州式误伤：
            # 旧写法无差别全顶曾连带顶掉真记忆，只顶一条又是矫枉
            # 过正——现在主目标全顶、连带清扫带门槛）。
            for old_id, old_text, old_meta in to_supersede[:1]:
                try:
                    # 推断记忆保护：主目标是推断未确认行时不直接顶，
                    # 攒待问问题（_supersede_or_ask 内部处理）。
                    self._supersede_or_ask(old_id, old_text, old_meta, key, text)
                except Exception:
                    logger.warning("槽位取代记账失败：旧行未取代。", exc_info=True)
            if len(to_supersede) > 1:
                self._sweep_extra_conflicts(
                    slot, to_supersede[1:], key, text, turn_id=turn_id,
                    probe_source=source, probe_confirmed=user_confirmed,
                )
            # 直存不走 save_turn：刚写的新记忆不再属于上一轮，撤回标记清掉，
            # 否则下一次“撤回”会误作废上一轮（2026-11-21 Gemini 第三轮真 Bug）。
            # save_turn 内部也调这里，但它在循环结束后才重设标记，顺序不冲突。
            self._last_stored_turn_id = None
            _add_result = {"id": key, "memory": text, "event": "ADD"}
            if _auth_reason:
                _add_result["auth_unverified"] = True
                _add_result["auth_reason"] = _auth_reason
            return _add_result

    def _append_new_memory(self, text, metadata):
        """写一条新记忆：先写账本 add，再写 SQLite 行。返回 key；
        账本写不进去返回 None（索引那步也不写）。"""
        try:
            self._ledger_append_add(text, metadata)
        except Exception:
            logger.warning("账本写不进去：本次记忆没存。", exc_info=True)
            return None
        try:
            vec = self._embed_text(text)
        except Exception:
            vec = None
        key = _ledger.ledger_event_key(text, metadata.get("created_at"))
        self._insert_memory_row(key, text, metadata, vec)
        return key

    def _supersede_row(self, old_id, old_text, old_meta, new_key, new_text,
                       turn_id=None):
        """把旧行标成被取代：先标 SQLite superseded=1，再写账本 supersede 事件。
        账本写失败时把 SQLite 的标记撤回来再抛——两边要一致，账本是真相。
        old_id 就是旧 key（SQLite id 就是账本 key）；new_key 是新记忆的 key。
        turn_id 传了就记进 supersede 事件（2026-11-24 1b）：“已纠正过”
        护栏按它认本轮，不再靠 5 分钟墙钟猜。
        """
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET superseded=1 WHERE id=?", (old_id,)
            )
            self._conn.commit()
            try:
                _event = {"type": _ledger.LEDGER_EVENT_SUPERSEDE,
                          "memory_id": new_key,
                          "supersedes_id": old_id,
                          "old_text": old_text,
                          "text": new_text,
                          "key": old_id,
                          "current_key": new_key}
                if turn_id is not None:
                    _event["turn_id"] = turn_id
                _ledger.append_event(
                    self.ledger_path, _event, lock=self._ledger_lock,
                )
            except Exception:
                self._conn.execute(
                    "UPDATE memories SET superseded=0 WHERE id=?", (old_id,)
                )
                self._conn.commit()
                raise

    def _record_occurrence(self, memory_id, metadata, today, turn_id, text=None):
        """同一条事实又出现一次：记进出现集合（不同天、不同会话），
        够 3 天 + 3 会就晋升成稳定事实（decay.py 的规矩）。
        同时把 last_used_at 拨到现在（刚被提到 = 刚被用上）。
        返回有没有晋升。调用方在同文去重（UNCHANGED）那条路上调。
        text 没传时从库里取（晋升事件要带原话）。
        """
        meta = dict(metadata or {})
        if text is None:
            try:
                row = self._conn.execute(
                    "SELECT text FROM memories WHERE id=?", (memory_id,)
                ).fetchone()
                text = (row or [""])[0] or ""
            except Exception:
                text = ""
        dates, sessions, _added = _decay.record_seen_occurrence(
            meta.get("seen_dates"), meta.get("seen_sessions"), today, turn_id
        )
        # 晋升聚合（2026-11-24，BOOS 定“晋升规则再优化”）：同一件事
        # 换个说法存成了另一行时，出现证据碎在各行上、谁也凑不满
        # 3 天 3 次——以前只会数自己这行的，是晋升“笨”的根。晋升
        # 判断前，把“同一事实的其他说法”（_fact_cluster_members，
        # 只认硬信号）的出现集合并进来：只并集合、不删不顶；带
        # 语气标记的成员证据不算数（带标记本来就不计晋升）。
        # 已确认成员的证据照常全额计——确认过的事实别被推断版
        # 拖住（同文去重里“确认只往上提”是同一个道理）。
        try:
            for _cid, _ctext, _cmeta in self._fact_cluster_members(
                memory_id, text, meta, allow_judge=True
            ):
                if (_cmeta or {}).get("tone") in (
                    "joke", "test", "hypothesis"
                ):
                    continue
                for _d in _decay._parse_seen_list(
                    (_cmeta or {}).get("seen_dates")
                ):
                    if _d not in dates:
                        dates.append(_d)
                for _s in _decay._parse_seen_list(
                    (_cmeta or {}).get("seen_sessions")
                ):
                    if _s not in sessions:
                        sessions.append(_s)
        except Exception:
            logger.warning("晋升聚合失败：按本行证据继续。", exc_info=True)
        meta["seen_dates"] = _decay._dump_seen_list(dates)
        meta["seen_sessions"] = _decay._dump_seen_list(sessions)
        meta["last_used_at"] = _now_iso()
        promoted = False
        # 授权硬判：转述未核实声称（auth_claim=relayed_unverified）
        # 不许走出现次数晋升路——重复出现再多次也还是未核实，
        # 等 BOOS 在待问里裁过（改成 owner_confirmed）才解禁。
        if meta.get("decay_kind") != _decay.DECAY_KIND_STABLE and meta.get("auth_claim") != _auth_guard.AUTH_CLAIM_RELAYED and _decay.should_promote(
            dates, sessions,
            # 2026-11-23 Phase 2 #2 根因：is_core 不能按“已确认”认——
            # save_turn 存的用户亲口说的事实都是已确认的（含一次性经历），
            # 按 _is_core_memory（confirmed 或有槽位）认，会把已确认的
            # 一次性经历也当核心，3 天 3 次兜底永远走不到。核心记忆本来
            # 就是稳定的（见 should_promote），这里 decay_kind 已保证
            # 不是 stable，is_core 恒为 False。
            is_core=False,
            tone=meta.get("tone"),
        ):
            # R-4（2026-11-24）：账本先行——PROMOTE 写不进去就
            # 不晋升，SQLite 不许跑到账本前面。以前写失败只警告、
            # 照样把 decay_kind 改成 stable，账本里没有这笔晋升，
            # 两边静默分叉（对账只补缺失行、不修已有行）。出现
            # 记录（seen_*）照常写回，下次出现时再试晋升。
            promoted_meta = dict(meta)
            promoted_meta["decay_kind"] = _decay.DECAY_KIND_STABLE
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": _ledger.LEDGER_EVENT_PROMOTE,
                     "memory_id": memory_id, "key": memory_id,
                     "text": text,
                     "meta": promoted_meta},
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("晋升记账失败：本次不晋升（账本为准）。", exc_info=True)
            else:
                meta["decay_kind"] = _decay.DECAY_KIND_STABLE
                promoted = True
        self._conn.execute(
            "UPDATE memories SET metadata_json=?, decay_kind=? WHERE id=?",
            (json.dumps(meta, ensure_ascii=False), meta.get("decay_kind"), memory_id),
        )
        self._conn.commit()
        return promoted

    # 晋升聚合的相似门槛（2026-11-24）：余弦和内容二字组都要够高
    # 才算“同一件事换了个说法”。门槛定高是故意的——认错（把两件
    # 事的证据并到一起）比认漏（晋升慢一点）危险：“每天定投5块
    # 买纳指”和“买标普”只差两个字，二字组重合率 0.75、进不来；
    # “喜欢吃面条”和“爱吃面条”二字组全同、进得来。
    _CLUSTER_MIN_COSINE = 0.85
    _CLUSTER_MIN_BIGRAM_OVERLAP = 0.8
    _CLUSTER_MIN_BIGRAM_JACCARD = 0.5

    # 判官归并的预筛门槛（2026-11-24，BOOS 拍板“让判官多判是不是
    # 同一件事”）：硬信号认不出的候选对，先过这道便宜的预筛才值得
    # 问判官——向量余弦 ≥0.55，或内容二字组重合率 ≥0.2 且 Jaccard
    # ≥0.05（“酸梅汤”三种说法只共享核心词：重合 0.22、Jaccard
    # 0.08–0.09，刚好够问；毫不相干的两句重合是 0，问都不问；
    # “定投纳指/标普”孪生句重合 0.71、照样进得来——是不是同一件
    # 事由判官判，预筛只管“值不值得问”）。
    _SAME_FACT_MIN_COSINE = 0.55
    _SAME_FACT_MIN_BIGRAM_OVERLAP = 0.2
    _SAME_FACT_MIN_BIGRAM_JACCARD = 0.05
    # 一次计数最多问判官几对：防候选一多就烧钱/拖慢；按像的程度
    # 从高到低问，剩下的下次出现再问（结论有缓存，不重复问）。
    _SAME_FACT_MAX_JUDGE_PER_CALL = 3
    # 判官归并结论存行 metadata 的键：{对方 id: {same, provider,
    # model, at, reason, cost_usd}}，两行各存一份——既是缓存
    # （同对不重复问），也是审计留痕（谁判的、何时判的）。
    _SAME_FACT_META_KEY = "same_fact_judgments"

    def _fact_cluster_members(self, memory_id, text, meta, allow_judge=False):
        """和本行是“同一件事、不同说法”的其他活行：[(id, text, meta)]。

        只认硬信号（和 _same_conflict_topic 一个标准，只认不猜）：
        - 同一单值槽 + 同一归一化值（“我住广州”/“我住在广州”）；或
        - 向量余弦 ≥0.85 且内容二字组高度重合（重合率 ≥0.8 且
          Jaccard ≥0.5）。
        排除：和本行正冲突的（同槽不同值、否定式正面冲突）、话题
        粗槽不同且都认得出的（喜欢吃 vs 不喜欢吃）、和本行挂着
        未答冲突问题的（BOOS 没裁之前，两条不共享证据——推断
        保护的规矩）。向量缺失时只有槽位一路可用，照常工作。

        allow_judge=True 时（只有晋升计数入口 _record_occurrence
        传），硬信号认不出、但过了全部护栏的候选对再交给判官判
        “是不是同一件事”（_judge_merge_candidates）：这是计数
        归并——行都留着、不改内容、不触发取代，只把出现证据
        并到一起数。护栏永远优先于判官，护栏挡掉的对问都不问。
        """
        text = text or ""
        if not text.strip():
            return []
        my_slot, my_value = _infer_fact_slot(text)
        my_topic = _correction_topic_slot(text)
        disputed = set()
        try:
            for q in self.pending_questions.list_unanswered():
                if q.get("kind") != _pq.QUESTION_KIND_CONFLICT:
                    continue
                pair = frozenset(
                    (q.get("old_id"), q.get("new_id"))
                )
                if memory_id in pair and len(pair) == 2:
                    disputed.add(pair)
        except Exception:
            disputed = set()
        my_vec = None
        try:
            import numpy as np

            row = self._conn.execute(
                "SELECT embedding FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is not None and row[0] is not None:
                my_vec = _blob_to_vector(row[0])
        except Exception:
            my_vec = None
        my_bigrams = self._content_bigrams(text)
        out = []
        candidates = []
        for row in self._conn.execute(
            "SELECT id, text, metadata_json, embedding FROM memories"
            " WHERE voided=0 AND deleted=0 AND superseded=0 AND id != ?",
            (memory_id,),
        ).fetchall():
            other_id, other_text = row[0], row[1] or ""
            if not other_text.strip():
                continue
            if frozenset((memory_id, other_id)) in disputed:
                continue
            if self._same_conflict_topic(text, other_text):
                continue
            other_slot, other_value = _infer_fact_slot(other_text)
            if (
                my_slot is not None and other_slot == my_slot
                and my_value is not None and other_value is not None
                and _normalize_memory_text(my_value)
                != _normalize_memory_text(other_value)
            ):
                continue
            other_topic = _correction_topic_slot(other_text)
            if my_topic and other_topic and my_topic != other_topic:
                continue
            try:
                other_meta = json.loads(row[2] or "{}")
            except Exception:
                other_meta = {}
            # 相似度三件套（余弦 + 二字组重合率/Jaccard）：硬信号
            # 判定和判官预筛共用，先一次算清。
            cos = None
            overlap = 0.0
            jaccard = 0.0
            if my_bigrams:
                other_bigrams = self._content_bigrams(other_text)
                if other_bigrams:
                    inter = len(my_bigrams & other_bigrams)
                    union = len(my_bigrams | other_bigrams)
                    overlap = inter / min(
                        len(my_bigrams), len(other_bigrams)
                    )
                    jaccard = inter / union if union else 0.0
            if my_vec is not None and row[3] is not None:
                try:
                    other_vec = _blob_to_vector(row[3])
                    if other_vec is not None and other_vec.shape == my_vec.shape:
                        denom = float(
                            np.linalg.norm(my_vec) * np.linalg.norm(other_vec)
                        )
                        if denom > 0:
                            cos = float(np.dot(my_vec, other_vec)) / denom
                except Exception:
                    cos = None
            matched = False
            if (
                my_slot is not None and other_slot == my_slot
                and my_value is not None and other_value is not None
                and _normalize_memory_text(my_value)
                == _normalize_memory_text(other_value)
            ):
                matched = True
            elif (
                cos is not None and cos >= self._CLUSTER_MIN_COSINE
                and overlap >= self._CLUSTER_MIN_BIGRAM_OVERLAP
                and jaccard >= self._CLUSTER_MIN_BIGRAM_JACCARD
            ):
                matched = True
            if matched:
                out.append((other_id, other_text, other_meta))
            else:
                candidates.append(
                    (other_id, other_text, other_meta, cos, overlap, jaccard)
                )
        if allow_judge and candidates:
            out.extend(
                self._judge_merge_candidates(
                    memory_id, text, meta, candidates
                )
            )
        return out

    def _judge_merge_candidates(self, memory_id, text, meta, candidates):
        """硬信号认不出的候选对，问判官“是不是同一件事”（2026-11-24，
        BOOS 拍板的设计扩展）：判为同一件事的，出现证据才并进同一
        簇计数。行都留着、不改内容、不触发取代。

        护栏优先于判官（冲突话题/未答冲突问题的对在
        _fact_cluster_members 里已挡掉，走不到这里）：带语气标记
        （玩笑/测试/假设）的对不问——那种证据本来就不算数。
        预筛不过（余弦和二字组都够不着）的对不问，省钱。
        结论缓存进两行 metadata（_SAME_FACT_META_KEY），同对不
        重复问；判官断了/回不明白 → 不归并、不缓存（下次再试），
        绝不崩、绝不乱并。返回被判为同一件事的 [(id, text, meta)]。
        """
        merged = []
        if (meta or {}).get("tone") in ("joke", "test", "hypothesis"):
            return merged
        my_cache = (meta or {}).get(self._SAME_FACT_META_KEY)
        if not isinstance(my_cache, dict):
            my_cache = {}
        ordered = sorted(
            candidates,
            key=lambda c: (
                c[3] if c[3] is not None else -1.0, c[4], c[5]
            ),
            reverse=True,
        )
        asked = 0
        for other_id, other_text, other_meta, cos, overlap, jaccard in ordered:
            if (other_meta or {}).get("tone") in (
                "joke", "test", "hypothesis"
            ):
                continue
            cached = my_cache.get(other_id)
            if not isinstance(cached, dict):
                other_cache = (other_meta or {}).get(self._SAME_FACT_META_KEY)
                if isinstance(other_cache, dict):
                    cached = other_cache.get(memory_id)
            if isinstance(cached, dict) and "same" in cached:
                if cached.get("same"):
                    merged.append((other_id, other_text, other_meta))
                continue
            plausible = (
                (cos is not None and cos >= self._SAME_FACT_MIN_COSINE)
                or (
                    overlap >= self._SAME_FACT_MIN_BIGRAM_OVERLAP
                    and jaccard >= self._SAME_FACT_MIN_BIGRAM_JACCARD
                )
            )
            if not plausible:
                continue
            if asked >= self._SAME_FACT_MAX_JUDGE_PER_CALL:
                continue
            verdict = self._ask_same_fact_judge(text, other_text)
            asked += 1
            if verdict is None:
                # 判官不可用：这一轮别再问别的对了，全体降级不归并。
                break
            record = {
                "same": bool(verdict.get("same_fact")),
                "provider": verdict.get("provider"),
                "model": verdict.get("model"),
                "at": _now_iso(),
                "reason": verdict.get("reason"),
                "cost_usd": verdict.get("cost_usd"),
            }
            self._store_same_fact_verdict(
                memory_id, meta, other_id, other_meta, record
            )
            if record["same"]:
                merged.append((other_id, other_text, other_meta))
        return merged

    def _store_same_fact_verdict(self, memory_id, meta, other_id,
                                 other_meta, record):
        """把判官归并结论写进两行 metadata（缓存 + 审计留痕）。

        本行写进传进来的 meta dict（_record_occurrence 收尾落盘）；
        对方行现读现写（不提交，跟着调用方的提交一起走）。写失败
        只记日志：结论丢了下次重问，不许拖垮计数主流程。
        """
        try:
            mine = meta.get(self._SAME_FACT_META_KEY)
            if not isinstance(mine, dict):
                mine = {}
                meta[self._SAME_FACT_META_KEY] = mine
            mine[other_id] = record
        except Exception:
            logger.warning("判官归并结论写本行失败。", exc_info=True)
        try:
            theirs = other_meta.get(self._SAME_FACT_META_KEY)
            if not isinstance(theirs, dict):
                theirs = {}
                other_meta[self._SAME_FACT_META_KEY] = theirs
            theirs[memory_id] = record
            self._conn.execute(
                "UPDATE memories SET metadata_json=? WHERE id=?",
                (json.dumps(other_meta, ensure_ascii=False), other_id),
            )
        except Exception:
            logger.warning("判官归并结论写对方行失败。", exc_info=True)

    def _ask_same_fact_judge(self, text_a, text_b):
        """调大模型判官判“是不是同一件事”（晋升计数归并用）。

        模型断了/没接好/回的格式不对时返回 None——调用方降级为
        不归并（走程序老路），别崩、别丢事实、别乱并。判官对象
        同 _ask_semantic_judge：self._judge_provider（测试注入）
        或配置解析（默认 DeepSeek flash，见 semantic_judge.py）。
        """
        try:
            from memory import semantic_judge as _sj
            verdict = _sj.judge_same_fact(
                text_a, text_b, provider=self._judge_provider,
                budget_tracker=self.budget_tracker,
            )
        except Exception as exc:
            logger.warning("同事实判断失败，降级不归并：%s", exc)
            verdict = None
        self._note_judge_outcome(verdict is not None)
        return verdict

    def _inherit_seen_evidence(self, old_meta, new_id, turn_id=None):
        """纠正/取代落盘后，把旧行攒下的出现证据（天数/会话）并给
        新行，再按合并后的证据复核一次晋升（2026-11-24，BOOS 定
        “晋升规则再优化”：新事实不许从零开始数）。

        旧行带语气标记（玩笑/测试/假设）时不继承——那种证据本来
        就不算数。失败只记日志：纠正本身已经落盘，不能被继承
        拖垮。返回有没有并上。"""
        try:
            if not old_meta or not new_id:
                return False
            if old_meta.get("tone") in ("joke", "test", "hypothesis"):
                return False
            old_dates = _decay._parse_seen_list(old_meta.get("seen_dates"))
            old_sessions = _decay._parse_seen_list(
                old_meta.get("seen_sessions")
            )
            if not old_dates and not old_sessions:
                return False
            row = self._conn.execute(
                "SELECT text, metadata_json FROM memories WHERE id=?"
                " AND voided=0 AND deleted=0 AND superseded=0",
                (new_id,),
            ).fetchone()
            if row is None:
                return False
            try:
                new_meta = json.loads(row[1] or "{}")
            except Exception:
                new_meta = {}
            merged_dates = _decay._parse_seen_list(new_meta.get("seen_dates"))
            merged_sessions = _decay._parse_seen_list(
                new_meta.get("seen_sessions")
            )
            for _d in old_dates:
                if _d not in merged_dates:
                    merged_dates.append(_d)
            for _s in old_sessions:
                if _s not in merged_sessions:
                    merged_sessions.append(_s)
            new_meta["seen_dates"] = _decay._dump_seen_list(merged_dates)
            new_meta["seen_sessions"] = _decay._dump_seen_list(merged_sessions)
            # _record_occurrence 把合并后的 meta 写回并复核晋升；
            # 纠正轮本身也算新事实的一次出现（turn 已在集合里时
            # 不重复计，record_seen_occurrence 的规矩）。
            self._record_occurrence(
                new_id, new_meta, _shanghai_today().isoformat(), turn_id,
                row[0] or "",
            )
            return True
        except Exception:
            logger.warning("纠正继承出现证据失败：新行证据未合并。", exc_info=True)
            return False

    # ---------- 第 8 条：实时整理（2026-11-24，BOOS 定“尽量做成和十一一样”） ----------
    def _judge_tone_with_router(self, text):
        """用整理路由的大模型判一条记忆的语气（webfix 二）。

        返回 (tone, evidence)；路由建不起来/模型没回/判不出来
        返回 None——不猜，留着下次整理再判（拿不准按认真算，
        是 tone_judgment_prompts 里给模型的规矩，也是这里的）。
        """
        router = self.organizing_router()
        if router is None:
            return None
        system_prompt, user_prompt = tone_judgment_prompts(text)
        try:
            out = router.judge(
                system_prompt, user_prompt,
                max_tokens=300, purpose="tone", probe_primary=False,
            )
        except Exception:
            logger.warning("判语气调用失败：本次不打标。", exc_info=True)
            return None
        if not out:
            return None
        tone, evidence = parse_tone_judgment(out.get("text"))
        if tone is None:
            return None
        return tone, evidence

    def _tag_memory_tone(self, memory_id, tone, evidence, source):
        """给一条活记忆打语气标（webfix 二，§16 落账）。

        账本先行：PROMOTE 事件带打标后的完整 meta（和晋升/去重
        升级同一个记法，重建时 reconcile 按 PROMOTE 带回 tone
        字段，不丢标）；账本写失败就不打，SQLite 不跑到账本
        前面。判出非认真（joke/test/hypothesis）时同步压档：
        stable → episodic、high → normal——反话/玩笑不许占着
        稳定+高重要（实测“我可太喜欢加班了”就是这么进的核心）。
        返回有没有打上。
        """
        if not memory_id or tone not in (
            "serious", "joke", "test", "hypothesis",
        ):
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT text, metadata_json FROM memories WHERE id=?"
                " AND voided=0 AND deleted=0 AND superseded=0",
                (memory_id,),
            ).fetchone()
            if row is None:
                return False
            text = row[0] or ""
            try:
                meta = json.loads(row[1] or "{}")
            except Exception:
                meta = {}
            meta["tone"] = tone
            meta["tone_evidence"] = evidence
            meta["tone_source"] = source
            if tone in ("joke", "test", "hypothesis"):
                if meta.get("decay_kind") == _decay.DECAY_KIND_STABLE:
                    meta["decay_kind"] = _decay.DECAY_KIND_EPISODIC
                if meta.get("importance") == "high":
                    meta["importance"] = "normal"
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": _ledger.LEDGER_EVENT_PROMOTE,
                     "memory_id": memory_id, "key": memory_id,
                     "text": text, "meta": dict(meta)},
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("语气打标记账失败：本次不打标。", exc_info=True)
                return False
            self._conn.execute(
                "UPDATE memories SET metadata_json=?, decay_kind=?"
                " WHERE id=?",
                (json.dumps(meta, ensure_ascii=False),
                 meta.get("decay_kind"), memory_id),
            )
            self._conn.commit()
            return True

    def _organize_turn_memories(self, memory_ids, turn_id=None):
        """一轮对话的小范围整理：只碰本轮触碰到的记忆（本轮新存的、
        被纠正换上的、被同文引用计数的），不等定时调度：
        - 晋升计数复核：按已存的出现集合复核晋升（_record_occurrence
          同款，账本先行；同一个 turn_id 不重复计数）；
        - 打标：活行还没槽位标的，按原文推断补上（_set_memory_slot，
          账本先行、已有标不动）；
        - 去重检查：集合内、以及集合与集合外活行之间归一化同文的，
          只记进汇总，不删不顶（删改内容要 BOOS 点头）；
        - 冲突扫描：集合内同槽不同值的两两过 _resolve_conflict，
          只记进汇总，不执行（本轮真要顶的，存储路径已经顶了；
          这里是复核网，捞到的留给自检/BOOS 看）。
        每条独立 try/except：单条失败只记日志，不影响其他条。
        幂等：同一批重复跑，结果收敛（标已补、晋升已完成）。
        返回汇总 dict。调用方在后台线程里跑（_schedule_turn_organize）。
        """
        summary = {
            "tagged": [], "promoted": [], "duplicates": [],
            "conflicts": [], "checked": 0, "tone_tagged": [],
        }
        ids = [m for m in dict.fromkeys(memory_ids or []) if m]
        if not ids:
            return summary
        today = _shanghai_today().isoformat()
        with self._lock:
            rows = {}
            for mid in ids:
                try:
                    row = self._conn.execute(
                        "SELECT text, metadata_json, superseded, deleted,"
                        " voided FROM memories WHERE id=?",
                        (mid,),
                    ).fetchone()
                except Exception:
                    continue
                if row is None or row[2] or row[3] or row[4]:
                    continue
                try:
                    meta = json.loads(row[1] or "{}")
                except Exception:
                    meta = {}
                rows[mid] = (row[0] or "", meta)
            summary["checked"] = len(rows)
            # 语气补标（webfix 二）：存的时候本地筛已标过明显的；
            # 没标的（本地认不出，比如反话）在这里问整理路由的
            # 大模型判——聊天/判官判出的语气必须落账，不然晋升
            # 和偏好使用都认不到这个标。判完同步刷新手里的 rows，
            # 后面的晋升复核按新标走（带标记的不计晋升）。
            for mid, (text, meta) in list(rows.items()):
                if meta.get("tone"):
                    continue
                try:
                    judged = self._judge_tone_with_router(text)
                except Exception:
                    judged = None
                if not judged:
                    continue
                _tone_value, _tone_evidence = judged
                try:
                    if self._tag_memory_tone(
                        mid, _tone_value, _tone_evidence, "model"
                    ):
                        summary["tone_tagged"].append(mid)
                        fresh = self._conn.execute(
                            "SELECT metadata_json FROM memories"
                            " WHERE id=?",
                            (mid,),
                        ).fetchone()
                        try:
                            rows[mid] = (
                                text,
                                json.loads((fresh or ["{}"])[0] or "{}"),
                            )
                        except Exception:
                            pass
                except Exception:
                    logger.warning(
                        "实时整理语气补标失败：%s", mid, exc_info=True
                    )
            # 晋升计数复核（先做：它会重写 metadata_json，后面的打标
            # 按库里最新值读，不拿旧 dict 覆盖）。
            for mid, (text, meta) in rows.items():
                try:
                    if self._record_occurrence(
                        mid, meta, today, turn_id, text
                    ):
                        summary["promoted"].append(mid)
                except Exception:
                    logger.warning(
                        "实时整理晋升复核失败：%s", mid, exc_info=True
                    )
            # 打标（槽位标）：按库里当前 metadata 判断有没有标。
            for mid, (text, _meta) in rows.items():
                try:
                    fresh = self._conn.execute(
                        "SELECT metadata_json FROM memories WHERE id=?",
                        (mid,),
                    ).fetchone()
                    fresh_meta = json.loads((fresh or ["{}"])[0] or "{}")
                    if not fresh_meta.get("slot"):
                        slot, _value = _infer_fact_slot(text)
                        if slot:
                            self._set_memory_slot(mid, slot)
                            summary["tagged"].append(mid)
                except Exception:
                    logger.warning(
                        "实时整理打标失败：%s", mid, exc_info=True
                    )
            # 去重检查（只报告，不删不顶）。
            try:
                seen_keys = {}
                for mid, (text, _meta) in rows.items():
                    k = _dedupe_key(text)
                    if not k:
                        continue
                    if k in seen_keys:
                        summary["duplicates"].append(
                            {"id": mid, "duplicate_of": seen_keys[k]}
                        )
                    else:
                        seen_keys[k] = mid
                if seen_keys:
                    for row in self._conn.execute(
                        "SELECT id, text FROM memories WHERE voided=0"
                        " AND deleted=0 AND superseded=0"
                    ).fetchall():
                        if row[0] in rows:
                            continue
                        k = _dedupe_key(row[1] or "")
                        if k and k in seen_keys:
                            summary["duplicates"].append(
                                {"id": seen_keys[k], "duplicate_of": row[0]}
                            )
            except Exception:
                logger.warning("实时整理去重检查失败。", exc_info=True)
            # 冲突扫描（只报告，不执行）。
            try:
                items = list(rows.items())
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        id_a, (text_a, meta_a) = items[i]
                        id_b, (text_b, meta_b) = items[j]
                        slot_a, val_a = _infer_fact_slot(text_a)
                        slot_b, val_b = _infer_fact_slot(text_b)
                        if not slot_a or slot_a != slot_b:
                            continue
                        if _normalize_memory_text(
                            val_a or ""
                        ) == _normalize_memory_text(val_b or ""):
                            continue
                        verdict = _provenance._resolve_conflict(
                            meta_a, meta_b
                        )
                        summary["conflicts"].append(
                            {
                                "a": id_a, "b": id_b, "slot": slot_a,
                                "action": (verdict or {}).get("action"),
                            }
                        )
            except Exception:
                logger.warning("实时整理冲突扫描失败。", exc_info=True)
        return summary

    def _schedule_turn_organize(self, memory_ids, turn_id=None):
        """把本轮整理放后台线程跑：不拖慢回复；整理出任何异常只
        记日志，绝不影响本轮对话。返回线程对象（测试可 join 等它
        跑完验效果）；没有可整理的返回 None。

        线程里不用本 store 的连接：另开一个同目录的 store 实例
        （独立 SQLite 连接，靠 WAL 的文件级并发）——跨线程共用
        一条连接在全套测试里触发过原生层 access violation
        （2026-11-24 全套崩溃定位），独立连接 + 实例被线程帧
        持有（不会被提前回收）才稳。整理线程之间串行：新线程
        先等上一条整理完，不堆积。"""
        ids = [m for m in dict.fromkeys(memory_ids or []) if m]
        if not ids:
            return None
        prev = getattr(self, "_turn_organize_thread", None)

        def _run():
            worker = None
            try:
                if prev is not None and prev.is_alive():
                    prev.join(timeout=30)
                worker = SQLiteStore(memory_dir=str(self.memory_dir))
                self._last_turn_organize_summary = (
                    worker._organize_turn_memories(ids, turn_id=turn_id)
                )
            except Exception:
                logger.warning("实时整理后台执行失败。", exc_info=True)
            finally:
                # 用完立刻关连接：不长期占着库文件（Windows 下开着
                # 的库文件删不掉、搬不走）。
                if worker is not None:
                    try:
                        worker._conn.close()
                    except Exception:
                        pass

        thread = threading.Thread(
            target=_run, daemon=True, name="turn-organize"
        )
        self._turn_organize_thread = thread
        thread.start()
        return thread

    def _find_same_slot_conflicts(self, slot, value):
        """找同槽位、不同值的活记忆：[(id, text, metadata)]。
        按每条原文重新认槽位（老行存的时候还没打 slot 标，只能从文本认）。"""
        with self._lock:
            norm_value = _normalize_memory_text(value or "")
            out = []
            for row in self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall():
                old_slot, old_value = _infer_fact_slot(row[1] or "")
                if old_slot != slot or old_value is None:
                    continue
                if _normalize_memory_text(old_value) == norm_value:
                    continue
                try:
                    old_meta = json.loads(row[2] or "{}")
                except Exception:
                    old_meta = {}
                out.append((row[0], row[1] or "", old_meta))
            return out

    def _is_core_protected_row(self, old_meta, slot=None):
        """核心/置顶记忆：身份槽位（名字/SOUL/电脑配置）、规则、精选
        置顶。这类记忆的内容变更永远要 BOOS 点头（项目红线），自动
        裁决/连带清扫一律不碰——要改只能走 BOOS 点名纠正的路。"""
        meta = old_meta or {}
        if slot in _provenance.KNOWN_SLOTS:
            return True
        if meta.get("slot") in _provenance.KNOWN_SLOTS:
            return True
        if meta.get("persona_projection"):
            # 性情投影行（追加决定八）：哪怕槽位标丢了，投影
            # 标记本身也单独算核心保护——只读投影不许自动动。
            return True
        return bool(meta.get("is_rule") or meta.get("curated"))

    def _sweep_extra_conflicts(self, slot, conflicts, new_key, new_text,
                               turn_id=None, probe_source="user",
                               probe_confirmed=True):
        """R-1 连带清扫（2026-11-24，BOOS 定“像十一”）：一次纠正的
        主目标之外，同槽还有别的真冲突旧值时，逐条一并清掉——不清
        的话单值槽里更早的错误值永远留着（P1 重查 R-1）。

        清扫的门槛（防广州式误伤的护栏，一条都不许放宽）：
        - 核心/置顶不碰（_is_core_protected_row，走问 BOOS 的路径）；
        - 只清 BOOS 在对话里亲口说过的（metadata 带 turn_id）——
          系统直存、迁移、推断来的行不连带：真记忆“用户住在广州”
          就是迁移来的推断行（turn_id 为空），它只能被点名纠正
          （correct_memory 主路），不许被成批清扫误伤；
        - 每条仍走 _resolve_conflict 裁决，不是 supersede_old 不顶；
        - 取代走 _supersede_row：账本先行、记 SUPERSEDE 事件，
          是标记不是删除；记账失败只警告、跳过这条，新记忆不回滚
          （取代失败留新记忆是既定规矩）。
        返回清掉的 id 列表。调用方需持锁。"""
        swept = []
        probe = {
            "source": probe_source,
            "user_confirmed": probe_confirmed,
            "confidence": 1.0 if probe_confirmed else 0.7,
            "created_at": _now_iso(),
            "write_seq": _next_write_seq(),
            "slot": slot,
        }
        for old_id, old_text, old_meta in conflicts or []:
            if not old_id or old_id == new_key:
                continue
            if self._is_core_protected_row(old_meta, slot):
                continue
            if not (old_meta or {}).get("turn_id"):
                continue
            if self._is_inferred_unconfirmed(old_meta):
                # 推断记忆保护（2026-11-24）：带 turn_id 的推断行
                # （模型在一轮里推断直存的）也不许连带悄悄顶——
                # 攒待问问题，BOOS 点头才换。
                self._queue_conflict_question(
                    old_id, old_text, new_key, new_text, turn_id=turn_id,
                )
                continue
            row = self._conn.execute(
                "SELECT superseded, deleted, voided FROM memories"
                " WHERE id=?",
                (old_id,),
            ).fetchone()
            if row is None or row[0] or row[1] or row[2]:
                continue  # 已不是活行（被主路顺带处理了），不重复顶
            verdict = _provenance._resolve_conflict(old_meta, probe)
            if (verdict or {}).get("action") != "supersede_old":
                continue
            try:
                self._supersede_row(
                    old_id, old_text, old_meta, new_key, new_text,
                    turn_id=turn_id,
                )
            except Exception:
                logger.warning(
                    "R-1 连带清扫取代记账失败：旧行未取代。", exc_info=True
                )
                continue
            swept.append(old_id)
        return swept

    def _set_memory_slot(self, memory_id, slot):
        """给一条记忆补槽位标（列 + metadata_json 里都写），以后按槽位
        找它就不用再从文本现认。找不到行、或已有槽位标时什么都不做，不抛异常。"""
        if not memory_id or not slot:
            return
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata_json FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                return
            try:
                meta = json.loads(row[0] or "{}")
            except Exception:
                meta = {}
            if meta.get("slot"):
                return
            # R-7（2026-11-24）：槽位标也记一笔账本（SLOT 事件），
            # 对账重建缺失行时按它把标带回来——以前只写 SQLite，
            # add 事件的 meta 里没有槽位标，重建出来标就丢了。
            # 账本先行：写不进去就先不补标（本函数是尽力而为的
            # 补标、不抛异常，下次用到这条时还会再试）。
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": _ledger.LEDGER_EVENT_SLOT,
                     "memory_id": memory_id, "key": memory_id,
                     "slot": slot},
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("补槽位记账失败：本次先不补标。", exc_info=True)
                return
            meta["slot"] = slot
            self._conn.execute(
                "UPDATE memories SET slot=?, metadata_json=? WHERE id=?",
                (slot, json.dumps(meta, ensure_ascii=False), memory_id),
            )
            self._conn.commit()

    def _complete_correction_fragment(self, fact):
        """碎片纠正补句（确定性版，不调模型）：新说法是个碎片值（“深圳”）
        时，看库里是不是恰好只有一条单值属性记忆——是就把它的旧值换
        成碎片值拼回整句（“我住北京”+“深圳”→“我住深圳”）；不是
        （没有、或有多条、或值含糊）就原样返回，调用方走老路，绝不硬编。
        只认 residence / user_name / agent_name 三个槽。"""
        if not isinstance(fact, str) or not fact.strip():
            return fact
        value = _detect_correction_fragment(fact)
        if value is None:
            return fact
        # 碎片值收尾：去句首的引导词/住/在（“应该是深圳”→“深圳”，
        # “住深圳了”→“深圳了”）、去句尾语气词；带事实动词的不是值
        # （“喜欢喝茶”），不补。
        value = (value or "").strip()
        for prefix in ("应该是", "其实是", "其实"):
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
                break
        if value[:1] in ("住", "在", "是"):
            value = value[1:].strip()
        value = value.rstrip("了呢啊吧哦嘛").strip()
        if not value or value in _VAGUE_FRAGMENT_VALUES:
            return fact
        if len(value) > 20:
            return fact
        if any(ch in value for ch in ("喜", "欢", "爱", "吃", "喝", "叫",
                                       "去", "做", "买", "看", "听", "玩",
                                       "想", "要", "会", "能")):
            return fact
        if "其实是" in value:
            # G7 三道锁之三：值里带着“其实是”，说明它根本不是光秃
            # 秃的值，而是“X 其实是 Y”这种带主语的整句被误认成了
            # 碎片（事故原形：“慢慢其实是巴西红耳龟”）——主语是
            # 第三者，更不许拿去拼任何一行，原样返回走老路。
            return fact
        with self._lock:
            cands = []
            for row in self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall():
                slot, old_value = _infer_fact_slot(row[1] or "")
                if slot is None or old_value is None:
                    continue
                if _normalize_memory_text(old_value) == _normalize_memory_text(value):
                    continue
                # G7 三道锁（一、三）：碎片补句不许拼核心行，也不许
                # 拼名字槽行——碎片本身认不出主语，拿去改名字行必然
                # 有改错对象的风险（事故原形就是把身份行拼了）。
                # 名字纠正走 correct_memory，那里有核心护栏把关。
                # 住处等普通槽位不受影响（“应该是深圳”照旧补）。
                try:
                    _meta = json.loads(row[2] or "{}")
                except Exception:
                    _meta = {}
                if slot in ("agent_name", "user_name"):
                    continue
                if self._is_core_protected_row(_meta, slot):
                    continue
                cands.append((row[1] or "", old_value))
        if len(cands) != 1:
            return fact
        old_text, old_value = cands[0]
        if old_value not in old_text:
            return fact
        return old_text.replace(old_value, value, 1)

    def _complete_plan_change(self, new_value, old_value):
        """计划斜改的整句补全（2026-11-25 实测问题二）：按用户亲口
        否定的旧值在活行里定位要改的那条，把旧值换成新值、拼回
        旧行原文——和碎片补句同一种换值法，只是定位键是旧值本身，
        且只认行程类行（带去/出差/行程/旅标记），不拿食物偏好这
        类同名行开刀。核心/置顶行不在这里排除：定位到之后照样走
        correct_memory，那里的 G7 护栏会挂起问 BOOS，不放宽。
        返回 (fact, old_text, candidates)：
        - 恰好一条候选 → (换好的整句, 旧行原文, [])；
        - 多条候选 → (新值原样, None, [{"id","text"}, ...])，
          调用方记 ambiguous、不猜；
        - 零条 → (新值原样, None, [])，调用方按普通陈述存整句
          （没有可顶的旧行，不存在并存问题）。"""
        new_value = (new_value or "").strip()
        old_value = (old_value or "").strip()
        if not new_value or not old_value or new_value == old_value:
            return new_value, None, []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall()
        cands = []
        for mid, text in rows:
            text = text or ""
            if old_value not in text:
                continue
            if not any(m in text for m in ("去", "出差", "行程", "旅")):
                continue
            if _normalize_memory_text(text) == \
                    _normalize_memory_text(new_value):
                continue
            cands.append((mid, text))
        if len(cands) > 1:
            return new_value, None, [
                {"id": mid, "text": text} for mid, text in cands
            ]
        if not cands:
            return new_value, None, []
        _mid, old_text = cands[0]
        return old_text.replace(old_value, new_value, 1), old_text, []

    def _retry_correction_by_slot(self, correction, fact, turn_id=None):
        """纠错分不清改哪条（ambiguous）时，按单值槽位再筛一次：新说法
        带槽位、且候选里恰好只有一条同槽位（值不同）时，就改那条。
        筛不出来返回 None（调用方保留 ambiguous，等 BOOS 挑）。"""
        slot, value = _infer_fact_slot(fact)
        if slot is None or value is None:
            return None
        cands = (correction or {}).get("candidates") or []
        same_slot = []
        for cand in cands:
            text = (cand or {}).get("text") if isinstance(cand, dict) else None
            if not text:
                continue
            cand_slot, cand_value = _infer_fact_slot(text)
            if cand_slot != slot or cand_value is None:
                continue
            if _normalize_memory_text(cand_value) == _normalize_memory_text(value):
                continue
            same_slot.append(cand)
        if len(same_slot) != 1:
            return None
        old_text = same_slot[0].get("text") or ""
        if not old_text:
            return None
        retry = self.correct_memory(
            old_text, fact, exclude_identical_to_new=True,
            explicit_correction=True, turn_id=turn_id,
        )
        if (retry or {}).get("status") == "updated":
            return retry
        return None

    def _apply_same_slot_update(self, sentence, slot, value, receipt,
                                turn_id, event_date, is_correction=False):
        """同槽位新值进库前的裁决（save_turn 逐句循环里调）：
        - 没同槽位旧值 → 返回 False，调用方照常存；
        - 无纠正词的新陈述（is_correction=False）→ 返回 False，调用方
          照常存成新行、不顶旧的（2026-11-23 model fix #1，D 组根因：
          只有带明确纠正词的才算纠正；不问 BOOS、不顶旧）；
        - 带明确纠正词（is_correction=True）→ 裁决最新的一条同槽位
          旧值（被纠正的那条，主目标）；裁决 supersede_old 时走
          correct_memory 取代，correct_memory 没换成的那条改走直存 +
          _supersede_row 兜底。主目标之外同槽其余真冲突旧值走连带
          清扫（R-1，2026-11-24，_sweep_extra_conflicts，带护栏）。
        """
        conflicts = self._find_same_slot_conflicts(slot, value)
        if not conflicts:
            return False
        if not is_correction:
            # D 组根因：无纠正词的新陈述不自动顶掉同槽旧记忆——直接存成
            # 新行（调用方继续走正常存储），不问、不顶。只带明确纠正词
            # 的才算纠正（is_correction=True 那条路）。
            return False
        # R-1（2026-11-24，BOOS 定“像十一”）：带明确纠正词时不再
        # 只顶最新一条——主目标仍是最新那条同槽冲突（被纠正的那条，
        # 照旧逐条裁决、裁决不过整句不存）；其余冲突旧值在主目标
        # 顶成之后走连带清扫（_sweep_extra_conflicts：只清对话里
        # 亲口说过的、非核心的——旧写法无差别全顶，曾把真记忆广州
        # 连带顶掉，护栏改在清扫的门槛上，不再靠只顶一条）。
        # 核心/置顶行先剔出裁决名单：不顶、不拦，留着走问 BOOS 的路。
        conflicts = [
            c for c in conflicts
            if not self._is_core_protected_row(c[2], slot)
        ]
        if not conflicts:
            return False
        conflicts = sorted(
            conflicts,
            key=lambda c: (
                (c[2] or {}).get("write_seq") or 0,
                (c[2] or {}).get("created_at") or "",
            ),
            reverse=True,
        )
        primary_conflicts = conflicts[:1]
        extra_conflicts = conflicts[1:]
        # S10（2026-11-22，BOOS 定选 A）：_apply_same_slot_update 只在
        # save_turn 的逐句循环里调，句子都来自用户本轮直接说的话——新事实
        # 按已确认算（user/1.0），不再按推断（inferred/0.7）。推断赢不了
        # 已确认的旧行，新事实会被 pending_new 拦掉、进 dropped（郑州案）。
        # 模型自己推断的事实不走这里，仍按 inferred/0.7 直存
        # （add_memory_text(source="inferred")，见低可信度挂起的回归测试）。
        new_probe = {
            "source": "user",
            "user_confirmed": True,
            "confidence": 1.0,
            "created_at": _now_iso(),
            "write_seq": _next_write_seq(),
            "slot": slot,
        }
        to_supersede = []
        for old_id, old_text, old_meta in primary_conflicts:
            verdict = _provenance._resolve_conflict(old_meta, new_probe)
            action = (verdict or {}).get("action")
            if action == "supersede_old":
                to_supersede.append((old_id, old_text, old_meta))
            elif action == "pending_new":
                receipt["dropped"].append(
                    {"text": sentence, "reason": "槽位冲突待确认"}
                )
                return True
            else:  # keep_old / ask：新说法不存，旧的留着。
                receipt["dropped"].append(
                    {"text": sentence, "reason": "槽位冲突未裁决"}
                )
                return True
        # 全是 supersede_old：逐条取代。
        first_outcome = None
        new_key = None
        remaining = []
        for old_id, old_text, old_meta in to_supersede:
            outcome = self.correct_memory(
                old_text, sentence, exclude_identical_to_new=True,
                explicit_correction=True, turn_id=turn_id,
            )
            if (outcome or {}).get("status") == "updated":
                if first_outcome is None:
                    first_outcome = outcome
                new_key = (outcome or {}).get("memory_id") or new_key
            elif (outcome or {}).get("status") == "pending_question":
                # 推断记忆保护：新说法已存、旧的挂待问——这条算
                # 处理完了，不进兜底（兜底会再存一遍/硬顶）。
                if first_outcome is None:
                    first_outcome = outcome
                new_key = (outcome or {}).get("memory_id") or new_key
            else:
                remaining.append((old_id, old_text, old_meta))
        if remaining:
            # correct_memory 没换成的（候选检索没找着旧条等）：新说法先
            # 直存，再按裁决直接取代旧行——裁决已说该换，不并存两条。
            if new_key is None:
                saved = self.add_memory_text(
                    sentence, source="user", turn_id=turn_id,
                    event_date=event_date, decay_kind="episodic", slot=slot,
                    # 下面循环会显式 _supersede_row 顶掉 remaining 这一条，
                    # 这里别再让直存自己裁决顶旧（会顶两遍）。
                    allow_slot_supersede=False,
                )
                if (saved or {}).get("event") != "ADD":
                    receipt["dropped"].append(
                        {"text": sentence, "reason": "账本写失败"}
                    )
                    return True
                new_key = saved.get("id")
            _fb_superseded = False
            for old_id, old_text, old_meta in remaining:
                try:
                    # 推断记忆保护：兜底硬顶也先过护栏——推断未确认
                    # 的旧行改挂待问，不算顶成。
                    if self._supersede_or_ask(
                        old_id, old_text, old_meta, new_key, sentence
                    ) == "superseded":
                        _fb_superseded = True
                except Exception:
                    logger.warning("槽位取代记账失败：旧行未取代。", exc_info=True)
            if first_outcome is None:
                first_outcome = {
                    "status": "updated" if _fb_superseded else "pending_question",
                    "memory_id": new_key,
                    "old_text": remaining[0][1], "new_text": sentence,
                    "candidates": [
                        {"id": oid, "text": ot} for oid, ot, _m in remaining
                    ],
                    "reason": "已更正" if _fb_superseded else "旧记忆是推断来的、还没确认过，等你确认再换",
                    "layer": "slot",
                    "resolution": "updated" if _fb_superseded else "pending_question",
                    "audit": {"source": "user", "method": "slot_conflict"},
                }
        if (
            extra_conflicts and new_key is not None
            and (first_outcome or {}).get("status") == "updated"
        ):
            # 主目标已顶成：其余同槽冲突旧值走连带清扫（门槛见
            # _sweep_extra_conflicts）。清不掉的不影响本次纠正结果。
            # 主目标被推断保护挂起（pending_question）时不扫——
            # 纠正本身还没经 BOOS 确认，其余的更不许先顶。
            self._sweep_extra_conflicts(
                slot, extra_conflicts, new_key, sentence,
                turn_id=turn_id,
            )
        if first_outcome is not None:
            slot2, _v = _infer_fact_slot(sentence)
            if slot2:
                self._set_memory_slot(first_outcome.get("memory_id"), slot2)
            receipt["stored"].append(sentence)
            receipt["results"].append(
                {"id": first_outcome.get("memory_id"), "memory": sentence,
                 "event": (
                     "SUPERSEDE"
                     if first_outcome.get("status") == "updated"
                     else "CONFLICT_PENDING"
                 ),
                 "old_text": first_outcome.get("old_text"),
                 "supersedes_id": ((first_outcome.get("candidates") or [{}])[0] or {}).get("id")}
            )
            if "correction" not in receipt:
                receipt["correction"] = first_outcome
            return True
        return False

    # ---------- 混合语义判断（2026-11-22，BOOS 定） ----------
    # 主控是十二自己（本地程序）：下面的 _needs_semantic_judge 只做分流，
    # 用程序能直接算出来的触发条件（slot 是不是 None、句式是不是偏好、
    # 库里有没有同类活记忆、含不含纠错说法），不调模型、不让模型决定
    # “要不要问模型”。大模型是判官：只在触发后被调来做语义判断。
    # 程序能判定的路（住处/名字/Agent 名）不动，仍走程序，不变慢、不变贵。

    def _preference_judge_candidates(self, text):
        """程序算的判官候选：活记忆里同类偏好句（同文的除外），最近的在前。

        只看句式里含不含偏好标记，不猜意思；同文（归一化后相同）的不算
        候选——那是去重的活，判官不用判。
        """
        norm = _normalize_memory_text(text)
        out = []
        for m in self._all_live_memories():
            cand_text = m["text"] or ""
            if _normalize_memory_text(cand_text) == norm:
                continue
            if not any(marker in cand_text for marker in _JUDGE_PREFERENCE_MARKERS):
                continue
            # 纠正记录不是偏好，不进候选（见 _JUDGE_SKIP_MARKERS）。
            if any(marker in cand_text for marker in _JUDGE_SKIP_MARKERS):
                continue
            # 语气标记（webfix 二）：玩笑/测试/假设不作偏好使用，
            # 不进判官候选（§16：带标记的记忆不当偏好）。
            if (m["metadata"] or {}).get("tone") in (
                "joke", "test", "hypothesis",
            ):
                continue
            out.append({"id": m["id"], "text": cand_text,
                        "meta": m["metadata"]})
            if len(out) >= _JUDGE_MAX_CANDIDATES:
                break
        return out

    def _needs_semantic_judge(self, sentence, slot):
        """主控分流（纯程序逻辑，不调模型）：该不该问大模型判官。

        返回 (trigger, reason, candidates)，trigger 为 True 才问。
        触发条件（至少满足一条，且都要有候选可判）：
        (a) slot 是 None + 偏好句式 + 库里有同类偏好活记忆；
        (c) 含纠错说法 + 有候选（2026-11-23 model fix #4 放宽：不要求
            偏好标记——“我现在喝豆奶了”这种没“喜欢”也没槽位的，程序
            判不定，交给判官；“我现在喜欢喝豆奶了”仍是这条）。
        slot 能推出来时走程序路，这里永远返回不触发。
        (b) 纠正候选多条分不清不在这里——那是 correct_memory 报 ambiguous
        之后的事，见 _judge_ambiguous_correction。
        """
        if slot is not None:
            return False, None, []
        # 2026-11-23 R2 后续①：偏好改变说法（改吃/只吃类）也触发判官。
        # 这类说法没有“喜欢”等偏好标记，但明确表达偏好变了；只认
        # _JUDGE_CHANGE_MARKERS 里的表达，避免过度触发。
        has_marker = any(
            marker in sentence for marker in _JUDGE_PREFERENCE_MARKERS
        )
        has_change = any(
            marker in sentence for marker in _JUDGE_CHANGE_MARKERS
        )
        has_correction_phrase = any(
            phrase in sentence for phrase in _CORRECTION_PHRASES
        )
        # 2026-11-23 model fix #4（模型优先）：程序只判明显 case——
        # 槽位能推出来的走程序路（上面已返回 False）、库里没同类候选的
        # 直接存（下面）；其余（带偏好标记/改变说法/纠正说法、且库里有
        # 同类活记忆）都问模型判官，程序不替模型做语义判断。
        if not has_marker and not has_change and not has_correction_phrase:
            return False, None, []
        cands = self._preference_judge_candidates(sentence)
        if not cands:
            return False, None, []
        # 改变标记是更明确的信号，优先于通用纠正短语匹配（如“我现在”）。
        if has_change and not has_marker:
            return True, "preference_change", cands
        if has_correction_phrase:
            return True, "correction_uncertain", cands
        return True, "slot_none_preference", cands

    def _note_judge_outcome(self, ok):
        """记下最近一次真问判官的结果（G7 三道锁之二的运行时信号）：
        问通了（拿到裁决）清降级标；没问通（None＝模型断了/没接好/
        回的格式不对）置降级标。实例级打桩替换掉 _ask_* 的测试走不
        到这里，不受影响。"""
        self._judge_runtime_failed = not ok

    def _judge_degraded(self):
        """判官现在是不是降级状态（G7 三道锁之二的判定信号）：
        - 配置层解析不出来：没接上/没启用/模型名不在配置里/没 Key
          （resolve_judge_provider 抛 SemanticJudgeUnavailable）；
        - 或本实例最近一次真问判官没问通（_note_judge_outcome 置标）。
        注入了判官对象（测试 mock）时解析直接用它、算不降级，它答
        不答得出走运行时信号。不真调模型，只解析配置。"""
        if getattr(self, "_judge_runtime_failed", False):
            return True
        try:
            from memory import semantic_judge as _sj
            _sj.resolve_judge_provider(
                provider=self._judge_provider,
                budget_tracker=self.budget_tracker,
            )
            return False
        except Exception:
            return True

    def _ask_semantic_judge(self, text, candidates, context=None):
        """调大模型判官（只在触发后调）。返回 verdict dict。

        模型断了（超时、API 报错、没 Key、没接好）或回的格式不对时
        返回 None——调用方按程序 fallback 走（照常存、不崩、不丢事实），
        失败记一条 warning 日志，方便查。
        判官对象：self._judge_provider（测试注入）或配置解析
        （默认 DeepSeek flash，见 memory/semantic_judge.py）。
        context 是整段对话上下文（最近几轮消息，2026-11-23 model fix
        #4）：判官只看一句话分不清“在纠正哪条”时，对话能定锚。
        """
        try:
            from memory import semantic_judge as _sj
            verdict = _sj.judge_semantic(
                text, candidates, provider=self._judge_provider,
                budget_tracker=self.budget_tracker, context=context,
            )
        except Exception as exc:
            logger.warning("语义判断失败，降级走程序：%s", exc)
            verdict = None
        self._note_judge_outcome(verdict is not None)
        return verdict

    def _ask_importance_judge(self, text, context=None):
        """调大模型重要性判官（只在程序判不定时调）。返回 verdict dict。

        模型断了或回的格式不对时返回 None——调用方按程序 fallback 走
        （当一次性经历存，不崩、不丢事实），失败记一条 warning 日志。
        判官对象：self._judge_provider（测试注入）或配置解析
        （默认 DeepSeek flash，见 memory/semantic_judge.py）。
        context 是整段对话上下文（2026-11-23 model fix #4）。
        """
        try:
            from memory import semantic_judge as _sj
            verdict = _sj.judge_importance(
                text, provider=self._judge_provider,
                budget_tracker=self.budget_tracker, context=context,
            )
        except Exception as exc:
            logger.warning("重要性判断失败，降级走程序：%s", exc)
            verdict = None
        self._note_judge_outcome(verdict is not None)
        return verdict

    def _decide_importance(self, text, slot=None, context=None):
        """重要性判定（对外口）：先过程序硬护栏，再走三段判。

        护栏（2026-11-24 webfix 二，BOOS 拍板“判官提议、程序裁决”）：
        - 问句永远不许 important（问句被判成 stable+high 是实测
          毛病：“我的口味你还记得不”）——直接 episodic，不问判官；
        - 判官提议的 important，对“单次提及的非核心内容”封顶成
          episodic：非核心 = 没槽位（槽位在 _decide_importance_inner
          里先行返回、judge_info 为 None，走不到这里）、非规则、
          文本里没有用户明确强调的说法（_EMPHASIS_MARKERS）、
          也不带重复性/习惯标记（_RECURRENCE_MARKERS——“我每天
          去游泳”是习惯陈述，不是单次提及，判重要就进核心，
          是 Phase 2 #2 的既有行为）。
          程序侧类别（槽位/偏好/决定句式）判的 important 不封顶
          （那是既定的核心类别设计，judge_info 同样为 None）。
        封顶留痕：judge_info["clamped"] 写明 from/to/reason，
        存记忆时落进 meta.importance_clamped（见 add_memory_text）。
        """
        if _is_question_sentence(text):
            return "episodic", "问句，不进核心（重要性护栏）", None
        decision, reason, judge_info = self._decide_importance_inner(
            text, slot, context=context
        )
        if (
            decision == "important"
            and judge_info is not None
            and slot is None
            and not _has_emphasis(text)
            and not _has_recurrence(text)
        ):
            judge_info = dict(judge_info)
            judge_info["clamped"] = {
                "from": "important", "to": "episodic",
                "reason": "单次提及的非核心内容，程序封顶",
            }
            return (
                "episodic",
                (reason or "判官判为长期重要")
                + "（程序护栏：单次提及的非核心内容，封顶为一次性经历）",
                judge_info,
            )
        return decision, reason, judge_info

    def _decide_importance_inner(self, text, slot=None, context=None):
        """重要性三段判（Phase 2 #2，混合法）：程序先判，难的问判官。

        返回 (decision, reason, judge_info)：decision 是 "important"
        （进核心，长期钉住）/ "episodic"（当一次性经历，3 天 3 次
        兜底还能晋升）/ "unsure"（拿不准，问 BOOS）；reason 是一句
        话原因；judge_info 是判官回的信息（没问模型时 None）。
        程序段（免费、确定性，先判）：语气非认真（本地筛命中
        joke/test/hypothesis）→ episodic；有槽位（住处/名字，
        核心类别）→ important；偏好句式（喜欢/爱吃/爱喝/爱看/
        爱听/爱玩/不爱/讨厌）→ important；决定句式（决定/打算/
        计划/准备/定投，“我”和动词之间可插时间词）→ important；带过去时间词（昨天/前天/
        今天/刚才/刚刚/上次/这次）→ episodic。都没命中才问判官：
        判 important→important，false→episodic，"unsure"→unsure；
        模型断了→episodic（不问 BOOS，兜底还能晋升）。
        """
        tone, tone_evidence = local_tone_screen(text or "")
        if tone in ("joke", "test", "hypothesis"):
            return (
                "episodic",
                "语气是%s（%s），不进核心" % (tone, tone_evidence),
                None,
            )
        if slot is not None:
            return "important", "带槽位 %s，是核心类别" % slot, None
        if any(m in (text or "") for m in _JUDGE_PREFERENCE_MARKERS):
            return "important", "是偏好句式，长期偏好进核心", None
        # P1-2（2026-11-23）：“我”和动词之间可插时间词——旧正则要求“我”紧贴
        # 动词，“我今天决定去北京”“我明天打算去上海”都漏判，掉到后面的
        # 过去时间词分支里被当成一次性经历。
        if re.search(r"(我(?:今天|明天|下周|下个月|刚才)?(?:决定|打算|计划|准备)|定投)", text or ""):
            return "important", "是决定/计划，长期决定进核心", None
        if re.search(r"(昨天|前天|今天|刚才|刚刚|上次|这次)", text or ""):
            return "episodic", "带过去时间词，是一次性经历", None
        verdict = self._ask_importance_judge(text, context=context)
        if verdict is None:
            return (
                "episodic",
                "判官没接通，先当一次性经历（3 天 3 次兜底还能晋升）",
                None,
            )
        important = verdict.get("important")
        reason = verdict.get("reason") or ""
        judge_info = {
            key: verdict.get(key)
            for key in ("provider", "model", "reason", "cost_usd", "usage")
            if verdict.get(key) is not None
        }
        if important is True:
            return "important", reason or "判官判为长期重要", judge_info
        if important is False:
            return "episodic", reason or "判官判为一次性经历", judge_info
        return "unsure", reason or "判官拿不准", judge_info

    def _queue_importance_question(self, text, memory_id, reason, turn_id=None):
        """为一条拿不准重要性的新记忆攒一个问题（Phase 2 #4）。
        返回问题 dict；同一条记忆已有未回答的问题时返回已有的那条。"""
        return self.pending_questions.add(
            _pq.QUESTION_KIND_IMPORTANCE,
            text,
            _pq.IMPORTANCE_QUESTION_OPTIONS,
            reason,
            memory_id=memory_id,
            turn_id=turn_id,
        )

    def _log_intake(self, text, verdict, category, reason, mark=None,
                    turn_id=None, source="user"):
        """入口当场拦留痕（实测问题三）：被拦/挂起的候选不进账本
        （账本只记记忆事件），另记一份 intake_log.jsonl——原文、
        裁决、类别、理由、当场判标记都在，事后可查可对账。
        写失败只记日志，不拦聊天。"""
        entry = {
            "ts": _now_iso(),
            "text": text,
            "verdict": verdict,
            "category": category,
            "reason": reason,
            "mark": mark,
            "turn_id": turn_id,
            "source": source,
        }
        try:
            with self._intake_log_lock:
                with open(self._intake_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("入口拦截留痕写失败。", exc_info=True)

    def _queue_intake_hold(self, text, category, reason, turn_id=None):
        """拿不准的候选挂起（实测问题三）：不落记忆行，进待问等
        BOOS 裁（存/不存）——挂起不是丢弃，问题和原文都在
        pending_questions 里查得到。同一原文已有未答的挂起问题
        时不重复攒。返回问题 dict。"""
        for q in self.pending_questions.find_questions(
            kind=_pq.QUESTION_KIND_INTAKE_HOLD,
            statuses=(_pq.QUESTION_STATUS_PENDING,
                      _pq.QUESTION_STATUS_ASKED),
        ):
            if (q.get("held_text") or "") == text:
                return q
        return self.pending_questions.add(
            _pq.QUESTION_KIND_INTAKE_HOLD,
            "这句我拿不准要不要记，先挂起了，你裁一下：%s" % text,
            _pq.INTAKE_HOLD_QUESTION_OPTIONS,
            reason,
            memory_id=None,
            turn_id=turn_id,
            extra={"held_text": text, "intake_category": category},
        )

    def _intake_screen(self, text, turn_id=None, mark=None, source="user"):
        """入口当场拦＋当场判合并裁决（实测问题三）。

        返回 (verdict, info)：info 带 category/reason/mark，HOLD
        时带 question_id。DROP/HOLD 都在这里留痕（日志；HOLD 另
        攒待问），调用方只管按 verdict 决定落不落行。"""
        cache_key = None
        if turn_id:
            cache_key = (turn_id, _normalize_memory_text(text))
            cached = self._intake_verdict_cache.get(cache_key)
            if cached is not None:
                return cached
        pv, category, preason = _intake.classify_program(
            text, _looks_like_fact
        )
        verdict, reason, used_mark = _intake.combine(pv, preason, mark)
        info = {
            "category": category, "reason": reason, "mark": used_mark,
            "question_id": None,
        }
        if verdict == _intake.VERDICT_DROP:
            self._log_intake(text, verdict, category, reason,
                             used_mark, turn_id, source)
        elif verdict == _intake.VERDICT_HOLD:
            q = self._queue_intake_hold(text, category, reason, turn_id)
            info["question_id"] = (q or {}).get("id")
            self._log_intake(text, verdict, category, reason,
                             used_mark, turn_id, source)
        if cache_key is not None:
            if len(self._intake_verdict_cache) > 512:
                self._intake_verdict_cache.clear()
            self._intake_verdict_cache[cache_key] = (verdict, info)
        return verdict, info

    def _answer_intake_hold_question(self, question, choice):
        """BOOS 回答入口挂起问题（实测问题三）：「存」→ 把挂起的
        原文按已裁决回存（intake_bypass，不再过当场拦——BOOS 本人
        就是裁决）；「不存」→ 只消耗问题（本来就没落行）。回存
        失败时如实报 ok=False、问题留着可再答（P1-5 同款顺序：
        先执行成功、再消耗问题）。"""
        question_id = question.get("id")
        if choice not in _pq.INTAKE_HOLD_QUESTION_OPTIONS:
            return {
                "ok": False,
                "reason": "选项不对，只能选：存 / 不存。",
                "question_id": question_id,
            }
        if choice == "不存":
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True, "applied": "dropped",
                "question_id": question_id,
            }
        held_text = (question.get("held_text") or "").strip()
        if not held_text:
            return {
                "ok": False,
                "reason": "挂起的原文丢了，没法存。",
                "question_id": question_id,
            }
        saved = self.add_memory_text(
            held_text, source="user",
            turn_id=question.get("turn_id"), intake_bypass=True,
        )
        if (saved or {}).get("event") not in ("ADD", "UNCHANGED"):
            return {
                "ok": False,
                "reason": "没存进去（%s），问题先留着，可以再答。"
                          % ((saved or {}).get("event") or "未知"),
                "question_id": question_id,
            }
        self.pending_questions.answer(question_id, choice)
        return {
            "ok": True, "applied": "stored",
            "question_id": question_id,
            "memory_id": (saved or {}).get("id"),
        }

    def _queue_auth_question(self, text, memory_id, reason, turn_id=None):
        """为一条转述授权声称攒待问（实测问题一）：BOOS 本人裁这话
        是不是他说的。返回问题 dict；同一条记忆已有未回答的授权
        问题时返回已有的那条（PendingQuestions.add 自带去重）。"""
        return self.pending_questions.add(
            _pq.QUESTION_KIND_AUTH_CLAIM,
            "有人转述说你授权过这件事，我没认，只按未核实存了：%s" % text,
            _pq.AUTH_CLAIM_QUESTION_OPTIONS,
            reason,
            memory_id=memory_id,
            turn_id=turn_id,
        )

    def _queue_change_attempt_question(self, action, memory_id,
                                       instruction_source):
        """非本人通道试图改记忆、被受控入口拒绝后攒一条待问留痕
        （实测问题一）。问题本身不挂目标记忆（memory_id=None），
        目标 id 放 extra——BOOS 答这条只表示知悉，不对目标记忆
        做任何变更（见 _answer_auth_claim_question 的
        change_attempt 分支）。返回问题 dict 或 None。"""
        action_cn = {"delete": "删除", "revive": "恢复"}.get(action, action)
        return self.pending_questions.add(
            _pq.QUESTION_KIND_AUTH_CLAIM,
            "有人（非本人通道，来源标记：%s）试图%s一条记忆，"
            "已按授权硬判拒绝、没有执行。目标记忆 id：%s"
            % (instruction_source, action_cn, memory_id),
            _pq.CHANGE_ATTEMPT_QUESTION_OPTIONS,
            "改记忆类动作只认 BOOS 本人指令，转述指令不许执行"
            "（实测问题一）。",
            memory_id=None,
            extra={"auth_action": "change_attempt",
                   "target_memory_id": memory_id,
                   "attempted_action": action},
        )

    def _answer_auth_claim_question(self, question, choice):
        """BOOS 回答授权核实问题（实测问题一）。

        change_attempt 留痕问题：答复只消耗问题，不动任何记忆。
        声称类问题：
          是真的——这话确实出自 BOOS 本人：那条按本人确认升级
            （账本 PROMOTE 先行、再改 SQLite：source 改 user、
            确认位置 1、auth_claim 改 owner_confirmed；decay 保持
            episodic——授权内容本身不是要钉住的事实）；
          不是真的——转述不实：那条走批准删除路删掉（先删成功、
            再消耗问题，和 R-10 同款顺序）。
        别的选项一律不认，什么都不改。
        """
        question_id = question.get("id")
        if question.get("auth_action") in ("change_attempt", "controlled_entry_blocked"):
            if choice not in (question.get("options") or []):
                return {"ok": False, "reason": "选项不对。"}
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True, "applied": "attempt_acknowledged",
                "question_id": question_id, "memory_id": None,
            }
        if choice not in _pq.AUTH_CLAIM_QUESTION_OPTIONS:
            return {"ok": False, "reason": "选项不对，只能选：是真的 / 不是真的。"}
        memory_id = question.get("memory_id")
        if choice == "不是真的":
            if not memory_id:
                self.pending_questions.answer(question_id, choice)
                return {
                    "ok": True, "applied": "rejected_no_memory",
                    "question_id": question_id, "memory_id": memory_id,
                }
            deleted = self.delete_memory(memory_id, approve=True)
            deleted_ok = (
                deleted.get("deleted") is True
                if isinstance(deleted, dict)
                else bool(deleted)
            )
            if not deleted_ok:
                return {
                    "ok": False,
                    "reason": "没删掉（%s），问题先留着，可以再答。"
                    % (
                        (deleted.get("reason") or "未知原因")
                        if isinstance(deleted, dict)
                        else "未知原因"
                    ),
                    "question_id": question_id, "memory_id": memory_id,
                    "deleted": False,
                }
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True, "applied": "rejected_deleted",
                "question_id": question_id, "memory_id": memory_id,
                "deleted": True,
            }
        # 是真的：BOOS 本人认下这句授权，升级为本人确认。
        if not memory_id:
            return {"ok": False, "reason": "这个问题没有对应的记忆。"}
        with self._lock:
            row = self._conn.execute(
                "SELECT text, metadata_json, superseded, deleted, voided"
                " FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return {"ok": False, "reason": "这条记忆已经不在库里了。"}
            if row[2] or row[3] or row[4]:
                return {"ok": False, "reason": "这条记忆已被取代/删除/作废，不能再确认。"}
            meta = json.loads(row[1] or "{}")
            if meta.get("user_confirmed"):
                self.pending_questions.answer(question_id, choice)
                return {
                    "ok": True, "applied": "owner_confirmed",
                    "question_id": question_id, "memory_id": memory_id,
                    "already_confirmed": True,
                }
            new_meta = dict(meta)
            new_meta["source"] = "user"
            new_meta["user_confirmed"] = True
            new_meta["auth_claim"] = _auth_guard.AUTH_CLAIM_OWNER_CONFIRMED
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": _ledger.LEDGER_EVENT_PROMOTE,
                     "memory_id": memory_id, "key": memory_id,
                     "text": row[0] or "", "meta": dict(new_meta)},
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("授权核实确认记账失败：本次不升级。", exc_info=True)
                return {"ok": False, "reason": "账本写不进去，这次没确认成，问题先留着。"}
            self._conn.execute(
                "UPDATE memories SET metadata_json=?, source=?,"
                " user_confirmed=1 WHERE id=?",
                (json.dumps(new_meta, ensure_ascii=False), "user", memory_id),
            )
            self._conn.commit()
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True, "applied": "owner_confirmed",
                "question_id": question_id, "memory_id": memory_id,
            }

    # ---------- 推断记忆保护（2026-11-24，BOOS 裁决） ----------
    # 背景：E 组重跑里，推断来源、未确认的旧记忆（可信度 0.7）被
    # 亲口新事实（1.0）在主目标裁决里直接顶掉。BOOS 定：“类似的
    # 事情别再犯”——这类旧记忆不许再被悄悄顶掉：新事实照常存，
    # 旧的先留着，攒一个待问问题，BOOS 点头才换；说不换就两条
    # 都留、这对冲突标为已裁决，不再反复问。亲口确认过的旧记忆
    # 不受保护，照旧最新赢、不问。

    @staticmethod
    def _is_inferred_unconfirmed(old_meta):
        """旧记忆是不是“推断来源、未确认”：没有 user_confirmed
        标记、且来源不是用户亲口陈述（source != "user"）。"""
        meta = old_meta or {}
        if meta.get("user_confirmed"):
            return False
        return meta.get("source") != "user"

    def _queue_conflict_question(self, old_id, old_text, new_id, new_text,
                                 turn_id=None):
        """为一对“推断旧记忆 vs 新事实”冲突攒待问问题（带新旧原文）。

        同一对已问过未答 → 返回已有的那条；已裁决“不换” → 返回
        None（别反复问）；其余新建。问题正文带新旧两条原文，
        BOOS 看着原文答。
        """
        if not old_id or not new_id or old_id == new_id:
            return None
        for q in self.pending_questions.find_questions(
            kind=_pq.QUESTION_KIND_CONFLICT
        ):
            if q.get("old_id") != old_id or q.get("new_id") != new_id:
                continue
            if q.get("status") == _pq.QUESTION_STATUS_ANSWERED:
                if q.get("answer") == "不换":
                    return None
            else:
                return q
        text = (
            "有一条我之前推断的记忆和你刚说的新事实对不上，"
            "要不要换成新的？\n旧的（推断，未确认）：%s\n新的：%s"
            % (old_text, new_text)
        )
        return self.pending_questions.add(
            _pq.QUESTION_KIND_CONFLICT,
            text,
            _pq.CONFLICT_QUESTION_OPTIONS,
            "旧记忆是推断来的、没有确认过，按规矩不能悄悄顶掉，先问你。",
            memory_id=old_id,
            turn_id=turn_id,
            extra={
                "old_id": old_id, "new_id": new_id,
                "old_text": old_text, "new_text": new_text,
            },
        )

    def _queue_correction_hold_question(self, old_id, old_text, new_id,
                                        new_text, turn_id=None,
                                        hold_reason=None):
        """G7 三道锁的挂起问题：纠正命中核心行 / 判官降级时程序路
        纠正 / 名字主语对不上时，自动纠正停手，攒这条待问问题。
        和推断保护的冲突问题同款（kind=conflict、选项“换成新的 /
        不换”、回答走 _answer_conflict_question 的人工批准路），
        只是正文按挂起原因写，extra 里多一个 hold_reason 供复核。
        同一对已问过未答 → 返回已有的；已裁决“不换” → 返回
        None（别反复问）。"""
        if not old_id or not new_id or old_id == new_id:
            return None
        for q in self.pending_questions.find_questions(
            kind=_pq.QUESTION_KIND_CONFLICT
        ):
            if q.get("old_id") != old_id or q.get("new_id") != new_id:
                continue
            if q.get("status") == _pq.QUESTION_STATUS_ANSWERED:
                if q.get("answer") == "不换":
                    return None
            else:
                return q
        if hold_reason == "core_protected":
            text = (
                "有一条核心记忆和你刚说的新说法对不上。核心记忆按"
                "规矩不能自动改，要不要换成新的？\n"
                "旧的（核心）：%s\n新的：%s" % (old_text, new_text)
            )
            reason = (
                "纠正命中核心/置顶记忆：任何自动纠正都不许碰，"
                "先问你（G7 三道锁之一）。"
            )
        elif hold_reason == "subject_mismatch":
            text = (
                "你刚说的这条纠正，名字对不上对象，我没敢自动改。"
                "要不要把旧的换成新的？\n"
                "旧的：%s\n新的：%s" % (old_text, new_text)
            )
            reason = (
                "名字槽纠正先核对主语：新说法的主语和目标行对不上"
                "（G7 三道锁之三）。"
            )
        else:
            text = (
                "你刚纠正的一条记忆，因为现在判官（大模型）不可用，"
                "我先挂起没有自动改。要不要把旧的换成新的？\n"
                "旧的：%s\n新的：%s" % (old_text, new_text)
            )
            reason = (
                "判官降级期间，程序路纠正只挂起不执行（G7 三道锁"
                "之二）；判官恢复后也可复核。"
            )
        return self.pending_questions.add(
            _pq.QUESTION_KIND_CONFLICT,
            text,
            _pq.CONFLICT_QUESTION_OPTIONS,
            reason,
            memory_id=old_id,
            turn_id=turn_id,
            extra={
                "old_id": old_id, "new_id": new_id,
                "old_text": old_text, "new_text": new_text,
                "hold_reason": hold_reason,
            },
        )

    def _supersede_or_ask(self, old_id, old_text, old_meta, new_key,
                          new_text, turn_id=None):
        """取代的统一护栏：旧行是推断未确认的 → 不顶，攒待问问题，
        返回 "pending"；否则走 _supersede_row（账本先行），返回
        "superseded"。账本失败时 _supersede_row 自己抛，调用方按
        原有的失败处理走。"""
        # G7 三道锁（一、二）：这条兜底取代路同样设防——核心行
        # 不许碰、判官降级时程序路不许执行，都改挂问题等确认。
        _slot = (old_meta or {}).get("slot") or _infer_fact_slot(old_text)[0]
        if self._is_core_protected_row(old_meta, _slot):
            self._queue_correction_hold_question(
                old_id, old_text, new_key, new_text,
                turn_id=turn_id, hold_reason="core_protected",
            )
            return "pending"
        if self._judge_degraded():
            self._queue_correction_hold_question(
                old_id, old_text, new_key, new_text,
                turn_id=turn_id, hold_reason="judge_degraded",
            )
            return "pending"
        if self._is_inferred_unconfirmed(old_meta):
            self._queue_conflict_question(
                old_id, old_text, new_key, new_text, turn_id=turn_id
            )
            return "pending"
        self._supersede_row(
            old_id, old_text, old_meta, new_key, new_text, turn_id=turn_id
        )
        # 晋升证据继承（2026-11-24）：同 _apply_correction_result
        # 的成功路——旧行攒的天数/会话并给新行，别从零开始数。
        self._inherit_seen_evidence(old_meta, new_key, turn_id=turn_id)
        return "superseded"

    def list_pending_questions(self):
        """还没回答的问题（pending + asked），按创建时间从旧到新。"""
        return self.pending_questions.list_unanswered()

    def answer_question(self, question_id, choice):
        """BOOS 回答了一个待问问题，按答案改库（Phase 2 #4）。

        choice 只能是“进核心 / 当一次性经历 / 不存”：
          进核心：把那条记忆的 decay_kind 改 stable、importance 改 high
            （账本记一笔 PROMOTE，和 3 天 3 次晋升同款）；
          当一次性经历：不动（已经是 episodic，兜底还能晋升）；
          不存：delete_memory(approve=True) 删掉那条记忆。
        返回 {"ok", "applied", ...}；问题不存在/已回答/选项不对/记忆
        不在库里时 ok=False，什么都不改。
        """
        # P1-5（2026-11-23）：先只看问题、不消耗——账本写失败时问题
        # 还留着可再答，SQLite 也一动不动；以前是先消耗问题、再先改
        # SQLite 后写账本，账本失败只记 warning，SQLite 的改动不回滚，
        # 重建后记忆又变回 episodic。
        question = self.pending_questions.get(question_id)
        if question is None or question.get("status") == _pq.QUESTION_STATUS_ANSWERED:
            return {"ok": False, "reason": "问题不存在或已经回答过了。"}
        if question.get("kind") == _pq.QUESTION_KIND_CONFLICT:
            # 推断记忆保护的冲突问题（2026-11-24）：选项和处理都
            # 不一样，单独走 _answer_conflict_question。
            return self._answer_conflict_question(question, choice)
        if question.get("kind") == _pq.QUESTION_KIND_CORE_CAPACITY:
            # 核心容量问题（第 1 块）：选项和处理都不同，单独走。
            return self._answer_core_capacity_question(question, choice)
        if question.get("kind") == _pq.QUESTION_KIND_AUTH_CLAIM:
            # 转述授权核实问题（实测问题一）：选项和处理都不同，单独走。
            return self._answer_auth_claim_question(question, choice)
        if question.get("kind") == _pq.QUESTION_KIND_INTAKE_HOLD:
            # 入口挂起问题（实测问题三）：选项和处理都不同，单独走。
            return self._answer_intake_hold_question(question, choice)
        if choice not in ("进核心", "当一次性经历", "不存"):
            return {"ok": False, "reason": "选项不对，只能选：进核心 / 当一次性经历 / 不存。"}
        if question.get("kind") != _pq.QUESTION_KIND_IMPORTANCE:
            return {"ok": False, "reason": "未知的问题种类。"}
        memory_id = question.get("memory_id")
        if choice == "当一次性经历":
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True, "applied": "episodic",
                "question_id": question_id, "memory_id": memory_id,
            }
        if choice == "不存":
            if not memory_id:
                return {"ok": False, "reason": "这个问题没有对应的记忆。"}
            # R-10（2026-11-24）：和「进核心」同款顺序（P1-5 标准）
            # ——先删成功、再消耗问题。以前先把问题标已答再删，删除
            # 因账本失败时问题已经没了、记忆还在，这条问题再也没
            # 机会重答。现在删不掉就如实报 ok=False，问题留着。
            deleted = self.delete_memory(memory_id, approve=True)
            deleted_ok = (
                deleted.get("deleted") is True
                if isinstance(deleted, dict)
                else bool(deleted)
            )
            if not deleted_ok:
                return {
                    "ok": False,
                    "reason": "没删掉（%s），问题先留着，可以再答。"
                    % (
                        (deleted.get("reason") or "未知原因")
                        if isinstance(deleted, dict)
                        else "未知原因"
                    ),
                    "question_id": question_id, "memory_id": memory_id,
                    "deleted": False,
                }
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True, "applied": "dropped",
                "question_id": question_id, "memory_id": memory_id,
                "deleted": True,
            }
        if not memory_id:
            return {"ok": False, "reason": "这个问题没有对应的记忆。"}
        with self._lock:
            row = self._conn.execute(
                "SELECT text, metadata_json, superseded, deleted, voided"
                " FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return {"ok": False, "reason": "这条记忆已经不在库里了。"}
            if row[2] or row[3] or row[4]:
                return {"ok": False, "reason": "这条记忆已被取代/删除/作废，不能进核心。"}
            meta = json.loads(row[1] or "{}")
            if meta.get("auth_claim") == _auth_guard.AUTH_CLAIM_RELAYED:
                # 授权硬判：转述未核实声称不许走重要性路进核心，
                # 先答授权核实问题、由 BOOS 本人裁。
                return {"ok": False, "reason": "这条是转述来的授权声称、还没经你本人核实，不能进核心；请先答授权核实问题。"}
            if meta.get("decay_kind") == _decay.DECAY_KIND_STABLE:
                self.pending_questions.answer(question_id, choice)
                return {
                    "ok": True, "applied": "core",
                    "question_id": question_id, "memory_id": memory_id,
                    "already_core": True,
                }
            # 先写账本、再改 SQLite（和 _append_new_memory 同款顺序）：
            # 账本写不进去就直接返回，SQLite 不动，问题也不消耗。
            new_meta = dict(meta)
            new_meta["decay_kind"] = _decay.DECAY_KIND_STABLE
            new_meta["importance"] = "high"
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {
                        "type": _ledger.LEDGER_EVENT_PROMOTE,
                        "memory_id": memory_id, "key": memory_id,
                        "text": row[0] or "", "meta": dict(new_meta),
                    },
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("回答问题晋升记账失败：记忆没进核心。", exc_info=True)
                return {"ok": False, "reason": "账本写失败，记忆没进核心。"}
            self._conn.execute(
                "UPDATE memories SET metadata_json=?, decay_kind=?,"
                " importance=? WHERE id=?",
                (
                    json.dumps(new_meta, ensure_ascii=False),
                    _decay.DECAY_KIND_STABLE,
                    new_meta.get("importance"),
                    memory_id,
                ),
            )
            self._conn.commit()
            self.pending_questions.answer(question_id, choice)
        return {
            "ok": True, "applied": "core",
            "question_id": question_id, "memory_id": memory_id,
        }

    def _answer_conflict_question(self, question, choice):
        """BOOS 回答推断保护的冲突问题（2026-11-24）：
        “换成新的”→ 走 _supersede_row 真顶（账本先行；写失败不
        消耗问题、如实报 ok=False）；“不换”→ 两条都留着，问题
        记录本身就是“已裁决”的凭据（_queue_conflict_question
        认它、同一对不再重问）。旧行已被别的路处理掉时，问题
        已无意义，消耗掉并如实报 already_gone。"""
        question_id = question.get("id")
        if choice not in _pq.CONFLICT_QUESTION_OPTIONS:
            return {
                "ok": False,
                "reason": "选项不对，只能选：换成新的 / 不换。",
                "question_id": question_id,
            }
        old_id = question.get("old_id") or question.get("memory_id")
        new_id = question.get("new_id")
        if choice == "不换":
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True, "applied": "kept_old",
                "question_id": question_id, "memory_id": old_id,
            }
        if not old_id or not new_id:
            return {
                "ok": False,
                "reason": "这个问题缺新旧记忆的指向，没法执行。",
                "question_id": question_id,
            }
        with self._lock:
            old_row = self._conn.execute(
                "SELECT text, metadata_json, superseded, deleted, voided"
                " FROM memories WHERE id=?",
                (old_id,),
            ).fetchone()
            if old_row is None:
                return {
                    "ok": False, "reason": "旧记忆已经不在库里了。",
                    "question_id": question_id, "memory_id": old_id,
                }
            if old_row[2] or old_row[3] or old_row[4]:
                self.pending_questions.answer(question_id, choice)
                return {
                    "ok": True, "applied": "already_gone",
                    "question_id": question_id, "memory_id": old_id,
                }
            new_row = self._conn.execute(
                "SELECT text, superseded, deleted, voided"
                " FROM memories WHERE id=?",
                (new_id,),
            ).fetchone()
            if new_row is None or new_row[1] or new_row[2] or new_row[3]:
                return {
                    "ok": False, "reason": "新记忆已经不在了，先不换。",
                    "question_id": question_id, "memory_id": old_id,
                }
            try:
                old_meta = json.loads(old_row[1] or "{}")
            except Exception:
                old_meta = {}
            try:
                self._supersede_row(
                    old_id, old_row[0] or "", old_meta,
                    new_id, new_row[0] or "",
                )
            except Exception:
                logger.warning(
                    "冲突问题批准后取代记账失败：没换成。", exc_info=True
                )
                return {
                    "ok": False,
                    "reason": "账本写失败，没换成，问题先留着，可以再答。",
                    "question_id": question_id, "memory_id": old_id,
                }
            self.pending_questions.answer(question_id, choice)
            # 晋升证据继承（2026-11-24）：批准换成新的这一刻才算
            # 真顶——旧行攒的天数/会话这时并给新行（裁“不换”时
            # 走不到这里，一点不并）。
            self._inherit_seen_evidence(
                old_meta, new_id, turn_id=question.get("turn_id")
            )
        return {
            "ok": True, "applied": "superseded",
            "question_id": question_id, "memory_id": old_id,
            "new_memory_id": new_id,
        }

    def _judge_preference_supersede(self, text, context=None):
        """问判官：这句偏好新事实该不该取代某条同类活偏好。

        返回 ((old_id, old_text, old_meta), verdict, trigger)；不取代/
        模型断了/没候选/目标不在候选里时返回 (None, None, None)。
        trigger 是触发原因（slot_none_preference / correction_uncertain）。
        目标校验在判官解析层已做过一次（target_id 必须在候选里），
        这里再按 id 对一次候选，对不上就不取代，不猜。
        """
        _trigger, _reason, cands = self._needs_semantic_judge(text, None)
        if not _trigger:
            return None, None, None
        verdict = self._ask_semantic_judge(text, cands, context=context)
        if not verdict or not verdict.get("should_supersede"):
            return None, None, None
        target = next(
            (c for c in cands if c["id"] == verdict.get("target_id")), None
        )
        if target is None:
            logger.warning("语义判断目标不在候选里，不取代。")
            return None, None, None
        # 主食守卫（R2 后续③）：新说法在换主食、目标却不是主食时不取代——
        # 目标可能已被另一条路取代过，判官硬挑了火锅/冒菜这类当目标。
        if not _judge_target_is_staple_plausible(text, target["text"]):
            logger.warning("语义判断目标不是主食，不取代：%s", target["text"])
            return None, None, None
        return (target["id"], target["text"], target["meta"]), verdict, _reason

    def _apply_semantic_judge_update(self, sentence, receipt, turn_id,
                                     event_date, context=None):
        """save_turn 逐句循环里的模型判官路：触发后问模型，该取代就走
        correct_memory 同款落盘（_apply_correction_result：新说法存一条，
        账本记 supersede，旧行标 superseded=1）。

        回执记 stored + SUPERSEDE 结果（route="model"）+ correction
        （route="model"，带判官的 provider/model/用量/成本），页面和
        /api/chat 的返回里能看到走了模型。返回 True=已处理，
        False=没处理（调用方照常走 add_memory_text 直存，不丢事实）。
        """
        target, verdict, trigger = self._judge_preference_supersede(
            sentence, context=context
        )
        if target is None:
            return False
        old_id, old_text, old_meta = target
        outcome = self._apply_correction_result(
            sentence, old_id, old_text, old_meta, same_text=False,
            candidates=[old_id], status="updated",
            resolution="semantic_judge",
            audit={"source": "user", "method": "semantic_judge",
                   "provider": verdict.get("provider"),
                   "model": verdict.get("model"),
                   "judge_slot": verdict.get("slot")},
            layer="judge", turn_id=turn_id,
        )
        if (outcome or {}).get("status") != "updated":
            return False
        outcome = dict(outcome)
        outcome["route"] = "model"
        outcome["judge"] = {
            key: verdict.get(key)
            for key in ("provider", "model", "slot", "reason",
                        "cost_usd", "usage")
            if verdict.get(key) is not None
        }
        outcome["judge"]["trigger"] = trigger
        receipt["stored"].append(sentence)
        receipt["results"].append(
            {"id": outcome.get("memory_id"), "memory": sentence,
             "event": "SUPERSEDE", "old_text": old_text,
             "supersedes_id": old_id, "route": "model"}
        )
        if "correction" not in receipt:
            receipt["correction"] = outcome
        receipt.setdefault("judge_calls", []).append(outcome["judge"])
        return True

    def _judge_ambiguous_correction(self, correction, fact, context=None,
                                    turn_id=None):
        """触发条件 (b)：correct_memory 报 ambiguous 后，再问大模型判官挑一条。

        挑出来就按那条落盘（_apply_correction_result，route="model"）；
        挑不出来（判官说不取代/目标不在候选里）或模型断了时返回 None——
        调用方保持 ambiguous，不猜（新说法照常当新记忆存）。
        """
        cands = (correction or {}).get("candidates") or []
        cand_list = [
            {"id": c.get("id"), "text": c.get("text")}
            for c in cands if c.get("id") and c.get("text")
        ]
        # 纠正记录不是偏好，不进候选（见 _JUDGE_SKIP_MARKERS）。
        cand_list = [
            c for c in cand_list
            if not any(m in (c.get("text") or "") for m in _JUDGE_SKIP_MARKERS)
        ]
        if len(cand_list) < 2:
            return None
        verdict = self._ask_semantic_judge(fact, cand_list, context=context)
        if not verdict or not verdict.get("should_supersede"):
            return None
        target = next(
            (c for c in cand_list if c["id"] == verdict.get("target_id")), None
        )
        if target is None:
            return None
        # 主食守卫（R2 后续③）：同 _judge_preference_supersede。
        if not _judge_target_is_staple_plausible(fact, target["text"]):
            logger.warning("语义判断目标不是主食，不取代：%s", target["text"])
            return None
        row = self._conn.execute(
            "SELECT metadata_json FROM memories WHERE id=?", (target["id"],)
        ).fetchone()
        old_meta = json.loads((row or [None])[0] or "{}")
        # R-1：判官路只从 save_turn 的槽位型纠正（old==new）进来，
        # 挑中一条落盘后，同槽其余真冲突旧值同样走连带清扫。
        outcome = self._apply_correction_result(
            fact, target["id"], target["text"], old_meta, same_text=False,
            candidates=[target["id"]], status="updated",
            resolution="semantic_judge",
            audit={"source": "user", "method": "semantic_judge",
                   "provider": verdict.get("provider"),
                   "model": verdict.get("model"),
                   "judge_slot": verdict.get("slot")},
            layer="judge", turn_id=turn_id, slot_sweep=True,
        )
        if (outcome or {}).get("status") != "updated":
            return None
        outcome = dict(outcome)
        outcome["route"] = "model"
        outcome["judge"] = {
            key: verdict.get(key)
            for key in ("provider", "model", "slot", "reason",
                        "cost_usd", "usage")
            if verdict.get(key) is not None
        }
        outcome["judge"]["trigger"] = "ambiguous_candidates"
        return outcome

    def _store_turn_receipt(self, turn_id, receipt):
        """存一轮回执进缓存，只留最近 _TURN_RECEIPTS_MAX 条。

        dict 按插入顺序，超了就删最旧的（去重回执要查它，无上限
        会一直长）。
        """
        self._turn_receipts[turn_id] = copy.deepcopy(receipt)
        while len(self._turn_receipts) > _TURN_RECEIPTS_MAX:
            self._turn_receipts.pop(next(iter(self._turn_receipts)))

    def save_turn(self, messages, turn_id=None, intake_marks=None):
        """一轮对话的收尾：同步把用户亲口说的事实原样存下来，留回执。

        和 mem0 那条路最大的不同：没有待定区、没有 60 秒定时器、
        没有“提取返回空就当成功”。调完这个函数，事实要么在账本里、
        要么在回执里写明为什么没存——绝不悄悄丢。
        同一个 turn_id 只存一次（web 的 ask() 和 run_question() 会各调
        一次，第二次直接返回上次的回执（含 correction），不重复存）。
        """
        # 没传 turn_id 时生成内部兜底 ID（2026-11-21 Gemini 第三轮）：多句轮次的
        # 每句都带同一个 turn_id，撤回才能整轮作废，否则退化成只作废最后 1 条。
        # _next_write_seq() 进程内严格递增，每次调用都不同，去重逻辑不受影响。
        turn_id = turn_id or f"auto-{_next_write_seq()}"
        if turn_id and turn_id in self._seen_turn_ids:
            # 根因 a（2026-11-22）：第二次调用不能只回一个空去重回执——
            # 上次的回执（含 correction）要原样还给调用方，否则纠正结果
            # 在第二次调用里丢了，页面拿不到 correction 键。
            cached = self._turn_receipts.get(turn_id)
            if cached is not None:
                receipt = copy.deepcopy(cached)
                receipt["duplicate_turn"] = True
                return receipt
            return {"turn_id": turn_id, "duplicate_turn": True, "results": []}
        if turn_id:
            self._seen_turn_ids.add(turn_id)
            self._active_turn_id = turn_id
        latest_user = ""
        _msgs = list(messages or [])
        for _i in range(len(_msgs) - 1, -1, -1):
            if (_msgs[_i] or {}).get("role") == "user":
                latest_user = (_msgs[_i] or {}).get("content") or ""
                # 缺点清单第 11 条（2026-11-24）：一句话分两条发——
                # 紧挨着的上一条也是 user 消息（中间没有助手回复）、
                # 且本条是明显续接碎片时，拼成整句再走后面的存储/
                # 纠正处理，不让“吗”这类碎片被当成独立消息。只拼
                # 本轮输入，会话历史里的原始两条不动。
                if _i > 0 and (_msgs[_i - 1] or {}).get("role") == "user":
                    _prev_user = (_msgs[_i - 1] or {}).get("content") or ""
                    if _prev_user.strip() and _is_continuation_fragment(
                        latest_user, _prev_user
                    ):
                        latest_user = _prev_user.strip() + latest_user.strip()
                break
        receipt = {
            "turn_id": turn_id,
            "considered": [],
            "stored": [],
            "dropped": [],
            "held": [],
            "retracted": False,
            "dropped_count": 0,
            "results": [],
        }
        if not (latest_user or "").strip():
            self._store_turn_receipt(turn_id, receipt)
            return receipt
        # 撤回：光秃秃的整句信号，把上一轮刚存的那条作废（软删进账本）。
        if latest_user.strip().rstrip("。！？.!?") in _BARE_RETRACTION:
            dropped = self._void_last_stored()
            receipt["retracted"] = True
            receipt["dropped_count"] = dropped
            self._store_turn_receipt(turn_id, receipt)
            return receipt
        today = _shanghai_today()
        # 聊天纠错（2026-11-22 correct_memory 接线）：本轮最后一条 user 消息
        # 里有明确纠错说法时（见 correction_text._detect_correction），先调
        # correct_memory 改旧记忆，结果放进回执的 "correction" 键（web 层
        # 已经在读这个键，供页面展示，见 web/server.py）。
        #   updated/unchanged/failed：纠错已处理，新说法不再当事实重存，
        #     只处理纠错句之外的部分；
        #   not_found/ambiguous：没找着旧记忆、或分不清改哪条时，新说法
        #     不能丢——当新事实存下来（根因 c，2026-11-22）；ambiguous
        #     的状态照实留在回执里，旧的没被取代。
        # 没纠错时走正常逐句循环，原来的键不动。
        correction = None
        correction_fact_norm = None  # 本轮纠正的新说法（归一化），逐句循环
        # 里认“这句就是纠正事实”用——只有它才走槽位取代（model fix #1）。
        sentences = _split_sentences(latest_user)
        detected = _detect_correction(latest_user)
        if detected is not None:
            fact = (detected.get("fact") or "").strip()
            # 叠起来的纠错说法（“不对，你记错了，我住上海”）：_detect_correction
            # 取最早的说法（“不对”），fact 里还带着“你记错了，”——再剥一层，
            # 剥到只剩新说法为止（最多两层，防循环）。
            for _ in range(2):
                inner = _detect_correction(fact)
                if inner is None:
                    break
                inner_fact = (inner.get("fact") or "").strip()
                if not inner_fact or inner_fact == fact:
                    break
                # 叠剥护栏（2026-11-24 webfix 一）：内层再剥只许剥掉
                # 纯纠正信号——被剥掉的那截里带事实内容（否定从句
                # “不吃牛肉”之类）时不许剥。实测根因：“我其实不吃
                # 牛肉，我吃鸡杂面”被内层“其实”剥成“我吃鸡杂面”，
                # 否定从句丢了，纠正只打中推断行、已确认的旧说法
                # 留活。剥掉的前缀只能是信号词+标点+人称。
                if not fact.endswith(inner_fact):
                    break
                dropped_prefix = fact[:len(fact) - len(inner_fact)]
                if not _correction_signal_only(dropped_prefix):
                    break
                fact = inner_fact
            if fact:
                plan_old_text = None
                plan_candidates = []
                if detected.get("plan_change"):
                    # 计划斜改（实测问题二）：不走槽位碎片补句——按
                    # 用户亲口否定的旧值定位旧行、把新值换进旧行
                    # 原文（_complete_plan_change）；定位不到唯一
                    # 旧行时 fact 还是新值碎片，下面的事实门会拦住
                    # 它，多候选时走后面的 ambiguous 留痕路。
                    fact, plan_old_text, plan_candidates = (
                        self._complete_plan_change(
                            fact, detected.get("old_value") or ""
                        )
                    )
                else:
                    # 碎片新说法（“不对，应该是深圳”→“深圳”）先按槽位补成整句；
                    # 补不出来就原样走 correct_memory（找不到旧记忆时当新记忆存）。
                    fact = self._complete_correction_fragment(fact)
                correction_fact_norm = _normalize_memory_text(fact)
                # 当下的想法/问句不纠正（“我现在想喝茶了”不是在改一条记忆）：
                # 检测到了也不走纠正分支，留给下面的正常循环按老规矩处理
                # （当下的想法会被 _looks_like_fact 拦下、不存）。
                if _looks_like_fact(fact)[0]:
                    correction = self.correct_memory(
                        plan_old_text or fact, fact,
                        exclude_identical_to_new=True,
                        explicit_correction=True, turn_id=turn_id,
                    )
                    if (correction or {}).get("status") == "ambiguous":
                        # 分不清改哪条时按槽位再筛一次：新说法带单值槽位、
                        # 且候选里恰好只有一条同槽位时，就改那条，不用问。
                        retry = self._retry_correction_by_slot(
                            correction, fact, turn_id=turn_id
                        )
                        if retry is not None:
                            correction = retry
                    if (correction or {}).get("status") == "ambiguous":
                        # 触发条件 (b)：纠正候选多条、相似度都过线，程序筛不出来——
                        # 再问大模型判官挑一条；挑不出来/模型断了就保持 ambiguous，
                        # 不猜（下面照常把新说法当新记忆存）。
                        judged = self._judge_ambiguous_correction(
                            correction, fact, context=messages,
                            turn_id=turn_id,
                        )
                        if judged is not None:
                            correction = judged
                    if (correction or {}).get("status") == "updated":
                        # 纠正换来的新记忆带上槽位标（以后按槽位找得到它）。
                        slot, _value = _infer_fact_slot(fact)
                        if slot:
                            self._set_memory_slot(
                                (correction or {}).get("memory_id"), slot
                            )
                    kept = _strip_correction_sentences(latest_user)
                    if (correction or {}).get("status") in ("not_found", "ambiguous"):
                        # 没找到旧记忆 / 分不清改哪条：新说法当新记忆存
                        # （带上纠错句之外的部分）。根因 c：ambiguous 时
                        # 以前这里直接丢了新说法，S4 的摩托车就是这么丢的。
                        # 逐句循环里这句是纠正事实，同槽位旧值在时才走
                        # correct_memory 顶掉最新那条（model fix #1）；
                        # 不是纠正事实的句子只存新行、不顶旧的。
                        sentences = kept + [fact]
                    else:
                        sentences = kept
                elif plan_candidates:
                    # 斜改对得上不止一条旧行、分不清改哪条：不猜、
                    # 不顶任何一条，新值碎片也不单独存；回执记
                    # ambiguous，收尾复核会标未落账并留痕
                    # （纠正当轮落账，实测问题二）。
                    correction = {
                        "status": "ambiguous",
                        "memory_id": None,
                        "old_text": detected.get("old_value") or "",
                        "new_text": fact,
                        "candidates": plan_candidates,
                        "reason": "这次改动对得上不止一条旧记忆，"
                                  "分不清要改哪条，先没动",
                        "layer": "plan_change",
                        "resolution": "ambiguous",
                        "audit": {"source": "user",
                                  "method": "plan_change"},
                    }
                    sentences = _strip_correction_sentences(latest_user)
        # 当场判（实测问题三，契约三「当场判机制口径」）：收尾时
        # 由对话当场的模型顺手给候选打标记（记/不记/拿不准＋理由）。
        # 标记来源：调用方显式传 intake_marks，或本库绑的
        # intake_marker（可调用对象，入参 (候选列表, messages)）。
        # 标记器不可用/抛错/格式不对时一律回退老管道（程序屏＋
        # 事后判官），不许比老路差；标记只是提案，逐句循环里经
        # _intake_screen 与程序屏合并，落账仍走原有各道门。
        _marks_by_norm = {}
        _marks_trace = []
        _raw_marks = intake_marks
        if _raw_marks is None and self.intake_marker is not None:
            try:
                _raw_marks = self.intake_marker(list(sentences), messages)
            except Exception:
                logger.warning("当场判标记器不可用，回退老管道。", exc_info=True)
                _raw_marks = None
        for _raw_mark in _raw_marks or []:
            _valid_mark = _intake.validate_mark(_raw_mark)
            if _valid_mark is not None:
                _marks_by_norm[
                    _normalize_memory_text(_valid_mark["text"])
                ] = _valid_mark
        live_texts = None  # 懒加载：只有出现“记住”前缀时才查一次在库文本
        new_questions = []  # 本轮拿不准重要性的问题（Phase 2 #4）
        touched_ids = set()  # 本轮触碰到的记忆 id（第 8 条实时整理用）
        for sentence in sentences:
            receipt["considered"].append(sentence)
            # “记住”前缀先剥（#8）：剥完再判事实、再算日期——“记住”两字
            # 本身不是记忆内容。剥完为空（光秃秃一句“记住”）直接丢掉。
            stripped = _strip_remember_prefix(sentence)
            stripped_remember = stripped != sentence
            sentence = stripped
            if not sentence:
                receipt["dropped"].append(
                    {"text": receipt["considered"][-1], "reason": "记住指令本身，无内容"}
                )
                continue
            # 确认词前缀（第三轮 #8）：剥掉“是的/对的/没错/可以/好的”，
            # 再判剩下的；剥完没内容就是光秃秃一句确认词，直接丢掉。
            # “是的删掉”剥完是“删掉”，走下面的删除指令拦下；
            # “是的，我喜欢喝茶”剥完是“我喜欢喝茶”，照常存。
            stripped = _strip_confirmation_prefix(sentence)
            if stripped != sentence:
                sentence = stripped
                if not sentence:
                    receipt["dropped"].append(
                        {"text": receipt["considered"][-1], "reason": "确认词"}
                    )
                    continue
            if stripped_remember:
                # 直接路径（“记住……”命令）已经存过同一句时不再存第二遍：
                # 同一文本两行是重复根（2026-11-20 根治过），检索层虽会
                # 按文本聚合，账本里也不该多一条 ADD。
                if live_texts is None:
                    live_texts = {m["text"] for m in self._all_live_memories()}
                if sentence in live_texts:
                    receipt["dropped"].append({"text": sentence, "reason": "已在库里"})
                    continue
            # 纠正元句提炼（webfix 一）：整句是“刚才那句说错了，
            # 我只吃鸡杂面”这种元句、又没被 _detect_correction
            # 认成纠正时，先提炼成事实再走后面的存储/判官路——
            # 原话整句不许当事实存进库。提炼不出就原样不动。
            if is_correction_meta_sentence(sentence):
                _distilled = distill_correction_fact(sentence)
                if _distilled:
                    sentence = _distilled
            ok, reason = _looks_like_fact(sentence)
            if not ok:
                receipt["dropped"].append({"text": sentence, "reason": reason})
                continue
            # 入口当场拦＋当场判（实测问题三）：过了 _looks_like_fact
            # 的句子还要过当场拦——程序明确垃圾当场丢、拿不准挂起，
            # 都不落行；当场判标记在这里合并（标记不许绕过任何门，
            # 落账仍走下面的槽位/判官/容量/授权各道原门）。
            _mark = (
                _marks_by_norm.get(_normalize_memory_text(sentence))
                or _marks_by_norm.get(
                    _normalize_memory_text(receipt["considered"][-1])
                )
            )
            _verdict, _info = self._intake_screen(
                sentence, turn_id=turn_id, mark=_mark, source="user"
            )
            if _mark is not None:
                _marks_trace.append({
                    "text": sentence,
                    "mark": _mark["mark"],
                    "reason": _mark.get("reason") or "",
                    "verdict": _verdict,
                })
            if _verdict == _intake.VERDICT_DROP:
                receipt["dropped"].append({
                    "text": sentence,
                    "reason": "入口当场拦：" + _info["reason"],
                    "intake_category": _info["category"],
                })
                continue
            if _verdict == _intake.VERDICT_HOLD:
                receipt["held"].append({
                    "text": sentence,
                    "reason": _info["reason"],
                    "intake_category": _info["category"],
                    "question_id": _info.get("question_id"),
                })
                continue
            # 存原话：去掉句尾的 。！（？的已经当问句 drop 了，不会到这里）。
            sentence = sentence.rstrip("。！")
            event_date = resolve_event_date(sentence, today)
            # 刚被纠正取代的旧说法（同一轮里先肯定、后纠正）不再重存。
            if (
                correction is not None
                and (correction or {}).get("status") == "updated"
                and (correction or {}).get("old_text")
                and _normalize_memory_text(sentence)
                == _normalize_memory_text((correction or {}).get("old_text") or "")
            ):
                receipt["dropped"].append({"text": sentence, "reason": "已被纠正取代"})
                continue
            # 这句是不是本轮的纠正事实（只有它是，才走槽位取代；
            # 普通新陈述只存新行、不顶旧的，2026-11-23 model fix #1）。
            is_correction_sentence = bool(
                correction_fact_norm
                and _normalize_memory_text(sentence) == correction_fact_norm
            )
            # 同槽位更新（2026-11-22 通用纠正 + 2026-11-23 model fix #1）：
            # 只有带明确纠正词的句子才取代同槽位旧值（且只顶最新那条）；
            # 无纠正词的新事实只存新行、不顶旧的。没冲突时照常存，
            # 顺手把槽位标上。
            slot, slot_value = _infer_fact_slot(sentence)
            if slot is not None and self._apply_same_slot_update(
                sentence, slot, slot_value, receipt, turn_id, event_date,
                is_correction=is_correction_sentence,
            ):
                continue
            # 混合判断（2026-11-22，BOOS 定）：程序推不出槽位的偏好句，
            # 主控（十二本地程序）按程序算出的触发条件决定问不问大模型
            # （见 _needs_semantic_judge：slot 为 None + 偏好句式 + 库里
            # 有同类偏好活记忆）；大模型只当判官，不做分流。
            if slot is None and self._apply_semantic_judge_update(
                sentence, receipt, turn_id, event_date, context=messages
            ):
                continue
            # 来源标 user（已确认）：save_turn 只存用户本轮直接说的话，
            # 不是模型推断——2026-11-22 BOOS 定选 A：用户亲口说的新事实
            # 直接算已确认（可信度 1.0）。但已确认不等于自动取代旧事实：
            # 2026-11-23 model fix #1——没纠正词的新陈述只存新行、不顶
            # 旧的，只有带明确纠正词的才取代。unconfirmed
            # 是留给“完全没来源信息”的旧记忆的桶（2026-11-09 之前）。
            # 模型自己推断的事实不走这里，仍按 inferred/0.7 直存
            # （add_memory_text(source="inferred")，见低可信度挂起的回归测试）。
            # 重要性判断（Phase 2 #2）：程序先判（槽位/偏好/决定/语气/
            # 时间词），难的问大模型判官；判不准时先当一次性经历存，
            # 再问 BOOS（#4，见下面的 new_questions）。
            importance_decision, importance_reason, _importance_judge = (
                self._decide_importance(sentence, slot, context=messages)
            )
            saved = self.add_memory_text(
                sentence, source="user", turn_id=turn_id,
                event_date=event_date,
                decay_kind=(
                    _decay.DECAY_KIND_STABLE
                    if importance_decision == "important"
                    else "episodic"
                ),
                importance=(
                    "high" if importance_decision == "important" else "normal"
                ),
                semantic_judge=False,
                # 只有纠正事实才允许直存路的槽位取代（model fix #1）；
                # 普通新陈述只存新行、不顶旧的。
                allow_slot_supersede=is_correction_sentence,
                **({"slot": slot} if slot else {}),
                # 重要性护栏的封顶留痕透传（webfix 二）：判官被程序
                # 压回去时，把 clamped 落进新记忆的 meta。
                **(
                    {"importance_clamped": _importance_judge["clamped"]}
                    if (_importance_judge or {}).get("clamped")
                    else {}
                ),
            )
            if (saved or {}).get("event") == "ADD":
                if live_texts is not None:
                    live_texts.add(sentence)
                receipt["stored"].append(sentence)
                receipt["results"].append(
                    {"id": saved.get("id"), "memory": sentence, "event": "ADD"}
                )
                if importance_decision == "unsure":
                    _q = self._queue_importance_question(
                        sentence, saved.get("id"), importance_reason, turn_id
                    )
                    if _q is not None:
                        new_questions.append(_q)
            elif (saved or {}).get("event") == "UNCHANGED":
                # 同文又出现：晋升计数刚在 add_memory_text 里记过，
                # 这条算本轮触碰到的（实时整理复核晋升用）。
                if (saved or {}).get("id"):
                    touched_ids.add(saved["id"])
                receipt["dropped"].append({"text": sentence, "reason": "已在库里"})
            elif (saved or {}).get("event") == "SENSITIVE":
                receipt["dropped"].append({"text": sentence, "reason": "含敏感信息"})
            elif (saved or {}).get("event") == "PENDING":
                # 同槽位裁决：新说法可信度低于旧的，挂起来待确认，不存。
                receipt["dropped"].append({"text": sentence, "reason": "槽位冲突待确认"})
            elif (saved or {}).get("event") == "INTAKE_DROPPED":
                # 其他路（槽位/判官路内部直存）被当场拦拦下时，
                # 和 SENSITIVE 同款，如实记回执，不落行。
                receipt["dropped"].append({
                    "text": sentence,
                    "reason": "入口当场拦：" + ((saved or {}).get("reason") or ""),
                })
            elif (saved or {}).get("event") == "INTAKE_HELD":
                receipt["held"].append({
                    "text": sentence,
                    "reason": (saved or {}).get("reason") or "",
                    "question_id": (saved or {}).get("question_id"),
                })
            elif (saved or {}).get("event") in ("STALE", "ASK"):
                # 同槽位裁决：新说法比旧的还早（STALE），或分不出先后（ASK），不存。
                receipt["dropped"].append({"text": sentence, "reason": "槽位冲突未裁决"})
            else:
                receipt["dropped"].append({"text": sentence, "reason": "账本写失败"})
        # 已确认冲突同轮清扫（webfix 一）：明确纠正这一轮里，主目标
        # 若被推断保护挂起（pending_question）或候选分不清
        # （ambiguous 后新事实走逐句循环存下），新事实里带否定、
        # 和某条“已确认”旧记忆正面冲突的，不许留活到下一轮再问
        # ——同一轮一并顶掉（指向已存的新行）。实测根因：纠正
        # “不吃牛肉”只打中推断来的“喜欢吃鸡杂面”，已确认的
        # “喜欢吃牛肉面”留活，回复还反问“要我一并处理吗”。
        if detected is not None and correction_fact_norm:
            _new_row_id = None
            _new_row_text = None
            if (correction or {}).get("memory_id"):
                _new_row_id = correction["memory_id"]
            else:
                for _r in receipt["results"]:
                    if (
                        isinstance(_r, dict)
                        and _r.get("event") == "ADD"
                        and _r.get("id")
                        and _normalize_memory_text(_r.get("memory") or "")
                        == correction_fact_norm
                    ):
                        _new_row_id = _r["id"]
                        break
            if _new_row_id:
                try:
                    _row = self._conn.execute(
                        "SELECT text FROM memories WHERE id=?",
                        (_new_row_id,),
                    ).fetchone()
                    _new_row_text = (_row or [""])[0] or ""
                except Exception:
                    _new_row_text = ""
            if _new_row_id and _new_row_text:
                try:
                    _swept = self._supersede_confirmed_contradictions(
                        _new_row_text, _new_row_id, turn_id=turn_id,
                    )
                except Exception:
                    logger.warning("已确认冲突清扫失败。", exc_info=True)
                    _swept = []
                if _swept and correction is not None:
                    correction["also_superseded"] = _swept
                    if correction.get("status") in (
                        "ambiguous", "pending_question",
                    ):
                        correction["reason"] = (
                            (correction.get("reason") or "")
                            + "；另有已确认的旧说法和新说法冲突，"
                            "已一并改掉："
                            + "、".join(
                                s["old_text"] for s in _swept
                            )
                        )
        if receipt["stored"]:
            with self._lock:
                self._last_stored_turn_id = turn_id
        if correction is not None and (
            "correction" not in receipt
            or (correction or {}).get("status") == "updated"
            or (receipt.get("correction") or {}).get("status") != "updated"
        ):
            # 显式纠错优先；但它报 not_found/ambiguous、而逐句循环里的槽位
            # 更新已经真换了旧记忆时，留循环那边的 updated，不拿
            # not_found/ambiguous 盖掉它（2026-11-23 回归修：纠正分支先报
            # ambiguous 时，循环兜底顶掉最新那条后回执仍要是 updated，
            # 回复才说得出“已经改好了”）。
            receipt["correction"] = correction
            if (correction or {}).get("status") == "updated":
                # 纠错存的新记忆属于本轮：撤回标记拨到本轮，
                # 下一句“撤回”作废的是这条新记忆，不是上一轮的。
                with self._lock:
                    self._last_stored_turn_id = turn_id
        # 纠正当轮落账·收尾复核（契约三、状态变更入口；实测
        # 问题二）：回执里的纠正状态只是各路的自报，落没落账要
        # 回查新旧两行在库里的真实状态才算数；没落账的如实标
        # 出来并留痕，不许拿“当场答对了”结案。
        if _marks_trace:
            # 当场判留痕：每条标记的原文、标记、理由、最终裁决都
            # 进回执，抽查时能还原当时为何这样标（契约验收口径）。
            receipt["intake_marks"] = _marks_trace
        if receipt.get("correction") is not None:
            self._verify_correction_landed(
                receipt["correction"], receipt, turn_id=turn_id
            )
        # #4：本轮拿不准的问题先问，顺带把攒着的旧问题一起带出来
        # （一轮最多 3 条，含新问题）。R-2（2026-11-24）：以前只有
        # 本轮有新问题时才带旧问题——本轮没新问题时，攒着的旧问题
        # 连带都不带，永远没机会在聊天里被问。现在每轮都带；重问
        # 的间隔/上限在 drain_for_chat 里管，不会每轮轰炸同一条。
        # 一轮最多问 3 个（QUESTIONS_PER_CHAT_MAX）：只把前 3 个
        # 标成已问、放进回执；多出来的留在 pending 里，下一轮再带。
        new_questions = new_questions[:_pq.QUESTIONS_PER_CHAT_MAX]
        for _q in new_questions:
            self.pending_questions.mark_asked(_q["id"])
        room = max(0, _pq.QUESTIONS_PER_CHAT_MAX - len(new_questions))
        backlog = (
            self.pending_questions.drain_for_chat(max_n=room)
            if room > 0
            else []
        )
        if new_questions or backlog:
            receipt["questions"] = new_questions + backlog
        # 第 8 条实时整理（2026-11-24）：本轮触碰到的记忆（新存的、
        # 纠正换上的、同文引用计数的）立刻在后台整理一小轮——
        # 打标/晋升复核/去重检查/集合内冲突扫描，不等定时调度；
        # 后台线程跑，不拖慢本轮回复，失败只记日志。
        for _r in receipt["results"]:
            if isinstance(_r, dict) and _r.get("id"):
                touched_ids.add(_r["id"])
        if correction is not None and (correction or {}).get("memory_id"):
            touched_ids.add(correction["memory_id"])
        self._schedule_turn_organize(sorted(touched_ids), turn_id=turn_id)
        self._store_turn_receipt(turn_id, receipt)
        return receipt

    def _verify_correction_landed(self, correction, receipt,
                                  turn_id=None):
        """纠正当轮落账·收尾复核（契约三、状态变更入口「纠正当轮
        落账」；2026-11-25 实测问题二）：回执里的纠正状态只是各路
        的自报，落没落账要回查新旧两行在库里的真实状态才算数——
          updated：旧行确已 superseded（至少没有同原文的活行
            残留）、新行确是活行；
          unchanged：新说法确有活行在库（本来就是对的，没啥可改）；
          pending_question：护栏挂起未执行，如实标未落账（待问
            本身就是留痕，不重复挂）；
          ambiguous：候选还活着、新旧并存，未落账；只标不另挂
            待问——conflict 待问按旧行去重，抢先挂一条会把后面
            流程针对同一旧行的精确待问顶掉（webfix 回归实测）；
          not_found：没有可顶的旧行时，新说法已存成活行就算
            到位；同槽位还有别的活行冲突着，就是没落账；
          failed 等其余状态：未落账，只标不挂问。
        复核结论就地写回 correction：landed（True/False）、
        verified=True；未落账时补 verify_note 说明卡在哪。
        返回 landed。"""
        if not isinstance(correction, dict):
            return False
        status = correction.get("status")
        correction["verified"] = True
        new_id = correction.get("memory_id")
        new_text = correction.get("new_text") or ""
        old_text = correction.get("old_text") or ""

        def _is_live(mid):
            if not mid:
                return None
            row = self._conn.execute(
                "SELECT superseded, deleted, voided FROM memories"
                " WHERE id=?", (mid,),
            ).fetchone()
            if row is None:
                return None
            return not (row[0] or row[1] or row[2])

        def _live_id_by_text(text):
            norm = _normalize_memory_text(text or "")
            if not norm:
                return None
            for mid, t, sup, dele, voi in self._conn.execute(
                "SELECT id, text, superseded, deleted, voided"
                " FROM memories"
            ).fetchall():
                if sup or dele or voi:
                    continue
                if _normalize_memory_text(t or "") == norm:
                    return mid
            return None

        def _any_row_by_text(text):
            """按原文找一行（不论死活）：(id, is_live) 或 None。"""
            norm = _normalize_memory_text(text or "")
            if not norm:
                return None
            for mid, t, sup, dele, voi in self._conn.execute(
                "SELECT id, text, superseded, deleted, voided"
                " FROM memories"
            ).fetchall():
                if _normalize_memory_text(t or "") == norm:
                    return mid, not (sup or dele or voi)
            return None

        landed = False
        note = ""
        with self._lock:
            if status == "updated":
                new_live = _is_live(new_id)
                if new_live is None:
                    new_live = _live_id_by_text(new_text) is not None
                # 旧行：优先认 candidates 里原文对得上的那条（按
                # id 查状态）；认不出就按原文全库找——找到活行
                # 就是没顶掉，找不到或找到的是已顶替行都算过。
                old_live = None
                for cand in correction.get("candidates") or []:
                    if not isinstance(cand, dict):
                        continue
                    if _normalize_memory_text(cand.get("text") or "") \
                            == _normalize_memory_text(old_text):
                        old_live = _is_live(cand.get("id"))
                        break
                if old_live is None:
                    found = _any_row_by_text(old_text)
                    if found is not None and found[0] != new_id:
                        old_live = found[1]
                if new_live and old_live is not True:
                    landed = True
                else:
                    parts = []
                    if not new_live:
                        parts.append("新行不是活行")
                    if old_live is True:
                        parts.append("旧行还活着")
                    note = "回查对不上：" + "、".join(parts) + "，纠正未落账"
            elif status == "unchanged":
                landed = (
                    _is_live(new_id) is True
                    or _live_id_by_text(new_text) is not None
                )
                if not landed:
                    note = "回查对不上：新说法在库里没有活行，纠正未落账"
            elif status == "pending_question":
                landed = False
                note = "纠正被护栏挂起、等 BOOS 裁决，未落账（待问已留痕）"
            elif status == "ambiguous":
                landed = False
                note = "分不清要改哪条、新旧并存，未落账"
            elif status == "not_found":
                new_live_id = _live_id_by_text(new_text)
                slot, value = _infer_fact_slot(new_text)
                conflicts = (
                    self._find_same_slot_conflicts(slot, value)
                    if slot is not None and value is not None
                    else []
                )
                if conflicts:
                    landed = False
                    note = "同槽位还有旧说法活着、新旧并存，未落账"
                elif new_live_id:
                    landed = True
                else:
                    landed = False
                    note = "新说法没有存进库，纠正未落账"
            else:  # failed 等其余状态
                landed = False
                note = "纠正未执行（%s），未落账" % (status or "未知状态")
        correction["landed"] = landed
        if not landed:
            correction["verify_note"] = note
        return landed

    def _find_correction_candidates(self, old_fact, new_fact=None, top_k=10,
                                        exclude_identical_to_new=False):
        """找纠错候选：要改的旧记忆（和 mem0_store._find_correction_candidates
        同款两路 + 兜底）。

        1. 语义 + BM25（_search_inner 两路都走了，含 0.4 资格线）；
        2. 兜底（直接检索为空时才走，免得往池子里加噪音）：最近上下文
           （note_context 记的本轮检索到的 id）+ 同核心槽位
           （extract_core_slots 交集，数字/日期/二字词）——“住上海”
           vs“住北京”余弦低、关键词不撞时，靠这条兜底找着旧条
           （mem0 的 Bug 6 场景）。
        返回活记忆的 id 列表（已确认在前、同类里新的在前）。
        exclude_identical_to_new 时跳过与 new_fact 相同的条目：纠错是来
        取代旧说法的，库里已有的新说法本身不是目标；不跳过的话，exact
        会先命中它而直接返回 unchanged，漏掉真正该被取代的旧说法。
        """
        with self._lock:
            old_fact = (old_fact or "").strip()
            if not old_fact:
                return []
            found = []
            seen = set()
            try:
                for hit in self._search_inner(old_fact, top_k=top_k * 4):
                    mid = hit.get("id")
                    if mid and mid not in seen:
                        seen.add(mid)
                        found.append(mid)
            except Exception:
                logger.warning("纠错：语义找候选失败。", exc_info=True)
            if not found:
                # 兜底：最近上下文 + 同核心槽位。
                try:
                    for mid in list(getattr(self, "_last_context_ids", []) or []):
                        if mid and mid not in seen:
                            row = self._conn.execute(
                                "SELECT id FROM memories WHERE id=? AND voided=0"
                                " AND deleted=0 AND superseded=0",
                                (mid,),
                            ).fetchone()
                            if row is not None:
                                seen.add(mid)
                                found.append(mid)
                except Exception:
                    pass
                try:
                    slots = extract_core_slots(new_fact or old_fact)
                    if slots:
                        for row in self._conn.execute(
                            "SELECT id, text FROM memories"
                            " WHERE voided=0 AND deleted=0 AND superseded=0"
                        ).fetchall():
                            if row[0] in seen:
                                continue
                            if extract_core_slots(row[1] or "") & slots:
                                seen.add(row[0])
                                found.append(row[0])
                except Exception:
                    pass
            if exclude_identical_to_new and new_fact:
                # 跳过与新说法相同的条目（见 docstring）：纠错是来取代旧
                # 说法的，库里已有的新说法本身不是目标。
                norm_new = _normalize_memory_text(new_fact)
                kept = []
                for mid in found:
                    row = self._conn.execute(
                        "SELECT text FROM memories WHERE id=?", (mid,)
                    ).fetchone()
                    if row is not None and _normalize_memory_text(
                        row[0] or ""
                    ) == norm_new:
                        continue
                    kept.append(mid)
                found = kept
            # 排：已确认的在前，同类里新的在前。
            # P1-1（2026-11-23）：旧 key 是 (-confirmed, -(created_at 字符串))，
            # 字符串取负号抛 TypeError、被下面的 try/except 吞掉，排序从不执行。
            # ISO 时间字符串字典序即时间序，(confirmed, created_at) 倒序排
            # 就是“已确认在前，同类里新的在前”。
            def _cand_key(mid):
                row = self._conn.execute(
                    "SELECT metadata_json, created_at FROM memories WHERE id=?", (mid,)
                ).fetchone()
                try:
                    meta = json.loads(((row or [None, None])[0]) or "{}")
                except Exception:
                    meta = {}
                confirmed = 1 if meta.get("user_confirmed") else 0
                created_at = (row[1] if row else None) or ""
                return (confirmed, created_at)
            try:
                found.sort(key=_cand_key, reverse=True)
            except Exception:
                pass
            return found

    def _rescue_correction_candidate(self, old_fact, new_fact, exclude_ids):
        """R-9（2026-11-24）：相似度筛全军覆没时补跑一次同槽兜底。

        _find_correction_candidates 的兜底（最近上下文 + 同核心槽位
        扫描）只在语义召回为空时才跑；召回返回了一堆不相关命中时，
        兜底不触发，真目标压根不在候选里，筛完只能报 not_found、
        旧错留着。这里把兜底单独再跑一次：排除已有候选、排除与
        新说法相同的行，兜底里恰好只有一条与新说法相似度 ≥ 0.70
        时返回它，其余情况返回 None（调用方维持原判）。
        只加查找，不放宽任何顶替护栏——返回的候选回主流程后还要
        再过单候选的相似度安全网。调用方需持锁（correct_memory 里）。"""
        seen = set(exclude_ids or [])
        rescued = []
        try:
            for mid in list(getattr(self, "_last_context_ids", []) or []):
                if mid and mid not in seen:
                    row = self._conn.execute(
                        "SELECT id FROM memories WHERE id=? AND voided=0"
                        " AND deleted=0 AND superseded=0",
                        (mid,),
                    ).fetchone()
                    if row is not None:
                        seen.add(mid)
                        rescued.append(mid)
        except Exception:
            pass
        try:
            slots = extract_core_slots(new_fact or old_fact)
            if slots:
                for row in self._conn.execute(
                    "SELECT id, text FROM memories"
                    " WHERE voided=0 AND deleted=0 AND superseded=0"
                ).fetchall():
                    if row[0] in seen:
                        continue
                    if extract_core_slots(row[1] or "") & slots:
                        seen.add(row[0])
                        rescued.append(row[0])
        except Exception:
            pass
        if not rescued:
            return None
        norm_new = _normalize_memory_text(new_fact or "")
        high = []
        for mid in rescued:
            text = self._candidate_text(mid)
            if text is None:
                continue
            if _normalize_memory_text(text) == norm_new:
                continue
            sim = self._candidate_similarity(mid, new_fact)
            if sim is not None and sim >= _CORRECTION_TOPIC_MIN_SIMILARITY:
                high.append(mid)
        if len(high) == 1:
            return high[0]
        return None

    def _cross_slot_net_target(self, old_fact, new_fact):
        """第 5 条跨槽安全网（2026-11-24，BOOS 定“让他概率最小”）。

        纠正靠同槽找对手 + R-9 同槽兜底；两条冲突记忆没分进一个槽
        时，真目标仍可能不在候选里，主路只能报 not_found、旧错留着。
        这里在主路全落空之后，对全部活记忆再扫一遍：
        - 与新说法或旧说法的语义相似度 ≥ 0.80（_CROSS_SLOT_NET_MIN_
          SIMILARITY，高于主路 0.70，宁缺毋滥）才算命中；
        - 命中按相似度取前 5 条打包给判官，判官明确 should_supersede
          且点名其中一条，才返回那条作取代目标。
        护栏（一条不许松）：
        - 判官说不取代 / 模型断了 / 点名的不在命中里 → 返回 None，
          调用方维持 not_found（旧错留着，也不误伤）；
        - 核心/置顶（_is_core_protected_row）命中也不自动顶，
          返回 None——动核心永远走问 BOOS 的路径；
        - 纠正便条行不进判官候选（和判官路同一口径）；
        - 与新说法相同的行跳过（那是新值本身，不是旧错）；
        - 判官目标还要过主食合理性守卫（和判官路同一口径）。
        返回 (memory_id, text, meta, verdict) 或 None。调用方需持锁。"""
        norm_new = _normalize_memory_text(new_fact or "")
        norm_old = _normalize_memory_text(old_fact or "")
        hits = []
        try:
            rows = self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall()
        except Exception:
            return None
        for mid, text, meta_json in rows:
            text = text or ""
            if not text or _normalize_memory_text(text) == norm_new:
                continue
            if any(m in text for m in _JUDGE_SKIP_MARKERS):
                continue
            sims = []
            sim_new = self._candidate_similarity(mid, new_fact)
            if sim_new is not None:
                sims.append(sim_new)
            if norm_old and norm_old != norm_new:
                sim_old = self._candidate_similarity(mid, old_fact)
                if sim_old is not None:
                    sims.append(sim_old)
            if not sims or max(sims) < _CROSS_SLOT_NET_MIN_SIMILARITY:
                continue
            hits.append((max(sims), mid, text, meta_json))
        if not hits:
            return None
        hits.sort(key=lambda h: h[0], reverse=True)
        hits = hits[:5]
        cand_list = [
            {"id": mid, "text": text} for _s, mid, text, _m in hits
        ]
        verdict = self._ask_semantic_judge(new_fact, cand_list)
        if not verdict or not verdict.get("should_supersede"):
            return None
        target = next(
            (h for h in hits if h[1] == verdict.get("target_id")), None
        )
        if target is None:
            logger.warning("跨槽安全网：判官点名的目标不在命中里，不取代。")
            return None
        _best, mid, text, meta_json = target
        try:
            meta = json.loads(meta_json or "{}")
        except Exception:
            meta = {}
        if self._is_core_protected_row(meta, meta.get("slot")):
            logger.info("跨槽安全网：命中核心/置顶记忆，不自动顶：%s", text)
            return None
        if not _judge_target_is_staple_plausible(new_fact, text):
            logger.warning("跨槽安全网：判官目标不是主食，不取代：%s", text)
            return None
        return mid, text, meta, verdict

    def _candidate_text(self, memory_id):
        """按 ID 查一条记忆的原文；查不到返回 None（调用方兜底，不抛异常）。"""
        try:
            row = self._conn.execute(
                "SELECT text FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        except Exception:
            return None
        return (row or [None])[0] if row else None

    def _candidates_with_text(self, candidates):
        """把纠错候选统一成 [{"id", "text"}]（2026-11-22 Bug 1 治本）。

        _find_correction_candidates 内部返回的是内存 ID（字符串）列表，
        调用方（main.py 的打印、网页的候选列表）要的是带文本的字典——
        以前直接把 ID 串交出去，打印层按字典 .get('text') 就崩成 500。
        字典候选原样过（缺 text 时按 ID 补查）；字符串候选按 ID 查库
        补文本，查不到就把 ID 本身当 text；绝不抛异常。
        """
        out = []
        for cand in candidates or []:
            if isinstance(cand, dict):
                cid = cand.get("id")
                text = cand.get("text")
                if not text and cid:
                    text = self._candidate_text(cid)
                out.append({"id": cid, "text": text or (cid or "")})
            elif isinstance(cand, str):
                out.append({"id": cand, "text": self._candidate_text(cand) or cand})
            else:
                out.append({"id": cand, "text": str(cand)})
        return out

    def _candidate_similarity(self, memory_id, new_fact):
        """一条候选旧记忆和新说法的余弦相似度（根因 b 的筛子）。

        优先用库里存着的向量，没有再现算；算不出来返回 None（调用方
        按“算不出来”处理，不当 0 分用——0 分会把候选错误地筛掉）。
        """
        row = self._conn.execute(
            "SELECT text, embedding FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            old_vec = _blob_to_vector(row[1])
            if old_vec is None:
                old_vec = self._embed_text(row[0])
            return _cosine(old_vec, self._embed_text(new_fact))
        except Exception:
            return None

    def _apply_correction_result(self, new_fact, old_id, old_text, old_meta,
                                 *, same_text, candidates, status="updated",
                                 resolution=None, audit=None, source="user",
                                 layer="single", turn_id=None,
                                 slot_sweep=False):
        """纠错收尾（2026-11-22）：按状态把结果写进账本和 SQLite，返回
        回执。纠错是追加式——旧记忆不改原文、不删：新说法当一条新记忆
        存（来源 user、已确认），账本记 supersede，旧行标 superseded=1。
        same_text 时新说法和旧的一样，不用重存：只记一笔出现。
        库里已有和新说法相同的活记忆时直接复用那条（刷新 last_used_at，
        dormant / delete_due 改回 verbatim），不再新插，防重复行。
        layer 哪一层定的（mem0 同款词）：exact（原样对上）/ single
        （单候选过安全网）；调用方显式传，不从 same_text 推。
        slot_sweep（R-1，2026-11-24）：True 时（槽位型纠正，见
        correct_memory）取代成功后把同槽其余真冲突旧值走连带清扫
        （_sweep_extra_conflicts），清掉的 id 记进回执 swept_ids。
        """
        # 提炼收口（webfix 一）：不管哪条路把纠正原话递进来
        # （判官路曾把“我只吃鸡杂面”连着“刚才那句说错了”整句
        # 当新事实），落盘前先提炼——元句不许成为 supersede 的
        # 现行事实。提炼不出（剥完没事实）就保持原样走老流程，
        # 不在这里拦（撤回类文本有自己的路）。
        if any(marker in (new_fact or "") for marker in (
            "说错了", "写错了", "讲错了", "记错了", "搞错了", "记岔了",
        )):
            _distilled = distill_correction_fact(new_fact)
            if _distilled and _distilled != new_fact:
                new_fact = _distilled
        with self._lock:
            # 回执里的 candidates 统一成带文本的字典（2026-11-22 Bug 1）：
            # 传进来的是 _find_correction_candidates 给的 ID 串列表，
            # 这里转完再返回，调用方不用再猜格式。
            candidates = self._candidates_with_text(candidates)
            if status == "unchanged":
                self._record_occurrence(
                    old_id, old_meta, _shanghai_today().isoformat(), turn_id,
                    old_text,
                )
                return {
                    "status": "unchanged", "memory_id": old_id,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": candidates or [],
                    "reason": "已是该内容，无需修改",
                    "layer": layer, "resolution": resolution or "exact",
                    "audit": audit or {"source": "user", "method": "unchanged"},
                }
            # 库里已有和新说法相同的活记忆时直接复用，不再新插一条
            #（否则新旧说法同时在库里时会留下两条一样的活记录）。
            # 复用时刷新 last_used_at，dormant / delete_due 顺便改回
            # verbatim；_supersede_row 的目标就是这条已有记录。
            existing_id = None
            existing_meta = None
            existing_text = None
            for _row in self._conn.execute(
                "SELECT id, text, metadata_json FROM memories WHERE voided=0"
                " AND deleted=0 AND superseded=0 AND id != ?",
                (old_id,),
            ).fetchall():
                if _dedupe_key(_row[1] or "") == _dedupe_key(new_fact):
                    existing_id = _row[0]
                    existing_text = _row[1] or ""
                    existing_meta = json.loads(_row[2] or "{}")
                    break
            if existing_id is not None:
                revived = existing_meta.get("decay_state") in (
                    _decay.DECAY_STATE_DORMANT, _decay.DECAY_STATE_DELETE_DUE,
                )
                existing_meta["last_used_at"] = _now_iso()
                if revived:
                    existing_meta["decay_state"] = _decay.DECAY_STATE_VERBATIM
                self._conn.execute(
                    "UPDATE memories SET metadata_json=? WHERE id=?",
                    (json.dumps(existing_meta, ensure_ascii=False), existing_id),
                )
                if revived:
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {"type": _ledger.LEDGER_EVENT_REVIVE,
                             "memory_id": existing_id, "key": existing_id,
                             "text": existing_text},
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning("复活记账失败。", exc_info=True)
                new_key = existing_id
            else:
                created_at = _now_iso()
                write_seq = _next_write_seq()
                today = _shanghai_today().isoformat()
                metadata = {
                    "source": source,
                    "confidence": 1.0,
                    "user_confirmed": True,
                    "created_at": created_at,
                    "write_seq": write_seq,
                    # 本轮号（2026-11-24 1a）：以前写死 None，撤回按
                    # turn_id 找这条时找不到，纠正出的新记忆撤不掉。
                    "turn_id": turn_id,
                    "slot": (old_meta or {}).get("slot"),
                    "is_rule": bool((old_meta or {}).get("is_rule")),
                    "curated": bool((old_meta or {}).get("curated")),
                    # P1-3（2026-11-23）：新记忆的 decay_kind 继承旧记忆的。
                    # 旧写法按 source="user" 一律算 stable，纠正一条一次性事件
                    #（“昨天我吃了牛肉面”→“鸡杂面”）也会变 stable，违背
                    #“一次性经历是 episodic”的规矩。旧记忆没写 decay_kind
                    # 时才回退到 classify_decay_kind。
                    "decay_kind": (old_meta or {}).get("decay_kind")
                    or _decay.classify_decay_kind(
                        source, False, bool((old_meta or {}).get("is_rule"))
                    ),
                    "importance": (old_meta or {}).get("importance", "normal"),
                    "event_date": resolve_event_date(new_fact),
                    "last_used_at": created_at,
                    "decay_state": _decay.DECAY_STATE_VERBATIM,
                }
                # 出现证据只记本轮（纠正轮本身是新事实的一次
                # 出现）：旧行攒的天数/会话不在这里预支——“取代
                # 后才继承”（2026-11-24 晋升优化），推断保护挂起
                # 时 BOOS 还没裁，预支了裁“不换”就并错了。真顶
                # 成功后由 _inherit_seen_evidence 统一并入+复核。
                dates, sessions, _added = _decay.record_seen_occurrence(
                    None, None, today, turn_id
                )
                metadata["seen_dates"] = _decay._dump_seen_list(dates)
                metadata["seen_sessions"] = _decay._dump_seen_list(sessions)
                try:
                    new_key = self._append_new_memory(new_fact, metadata)
                except Exception:
                    return {
                        "status": "failed", "memory_id": old_id,
                        "old_text": old_text, "new_text": new_fact,
                        "candidates": candidates or [],
                        "reason": "新记忆写入失败", "layer": "none",
                        "resolution": resolution or "failed",
                        "audit": audit or {"source": source, "method": "correct_memory"},
                    }
                if new_key is None:
                    return {
                        "status": "failed", "memory_id": old_id,
                        "old_text": old_text, "new_text": new_fact,
                        "candidates": candidates or [],
                        "reason": "账本写失败", "layer": "none",
                        "resolution": resolution or "failed",
                        "audit": audit or {"source": source, "method": "correct_memory"},
                    }
            if self._is_inferred_unconfirmed(old_meta):
                # 推断记忆保护（2026-11-24，BOOS 裁决“类似的事情
                # 别再犯”）：新说法照常存下（上面已存/已复用），
                # 推断来的旧记忆不直接顶——攒待问问题，BOOS 点头
                # 才换；连带清扫也不跑（主目标都没定，其余的更
                # 不许先顶）。回答走 answer_question 的冲突分支。
                question = self._queue_conflict_question(
                    old_id, old_text, new_key, new_fact, turn_id=turn_id,
                )
                return {
                    "status": "pending_question", "memory_id": new_key,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": candidates or [],
                    "reason": "旧记忆是推断来的、还没确认过：新说法已存，旧的等你确认再换",
                    "layer": layer,
                    "resolution": "pending_question",
                    "audit": audit or {"source": source, "method": "correct_memory"},
                    "question_id": (question or {}).get("id"),
                }
            # G7 三道锁（2026-11-24，BOOS 批准）：取代落盘前的最后
            # 护栏——所有纠正路（程序路、判官路）都从这里过：
            # 一、目标是核心/置顶行：任何自动纠正不许碰。新说法上面
            #     已存（留痕），这里只挂问题等 BOOS 点头；答“换成
            #     新的”才走 _answer_conflict_question 的人工批准路
            #     真顶。（事故形状：判官降级时名字槽纠正把身份行
            #     “Agent 的名字是十二”顶掉了——这条路当时没设防，
            #     其他路如槽位裁决/连带清扫/跨槽安全网都有核心过滤，
            #     唯独纠正落盘口漏了。）
            # 二、判官降级期间（_judge_degraded），程序路解析出的
            #     纠正（layer 不是 judge/net）只挂起不执行；判官路
            #     能走到这里说明判官刚答过，不在降级之列。
            # （三、名字槽主语核对在碎片补句和目标选择处，见
            # _complete_correction_fragment：名字行不许被碎片拼改。）
            _hold_reason = None
            _target_slot = (
                (old_meta or {}).get("slot")
                or _infer_fact_slot(old_text)[0]
            )
            if self._is_core_protected_row(old_meta, _target_slot):
                _hold_reason = "core_protected"
            elif layer not in ("judge", "net") and self._judge_degraded():
                _hold_reason = "judge_degraded"
            if _hold_reason is not None:
                question = self._queue_correction_hold_question(
                    old_id, old_text, new_key, new_fact,
                    turn_id=turn_id, hold_reason=_hold_reason,
                )
                return {
                    "status": "pending_question",
                    "memory_id": new_key,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": candidates or [],
                    "reason": (
                        "目标是核心记忆，按规矩不自动改：新说法已存，"
                        "已挂问题等你确认"
                        if _hold_reason == "core_protected" else
                        "判官暂时不可用，纠正先挂起：新说法已存，"
                        "已挂问题等你确认（或判官恢复后复核）"
                    ),
                    "layer": layer,
                    "resolution": "pending_question",
                    "audit": audit or {
                        "source": source, "method": "correct_memory"
                    },
                    "question_id": (question or {}).get("id"),
                    "hold_reason": _hold_reason,
                }
            try:
                self._supersede_row(
                    old_id, old_text, old_meta, new_key, new_fact,
                    turn_id=turn_id,
                )
            except Exception:
                logger.warning("纠错取代记账失败：旧行未取代。", exc_info=True)
                return {
                    "status": "failed", "memory_id": new_key,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": candidates or [],
                    "reason": "新记忆已存，但旧记忆未被取代",
                    "layer": "none",
                    "resolution": "failed",
                    "audit": audit or {"source": source, "method": "correct_memory"},
                }
            # 晋升证据继承（2026-11-24）：真顶成功了，旧行攒的
            # 天数/会话并给新行并复核晋升——新事实不从零开始数。
            # 失败只记日志（_inherit_seen_evidence 内部处理），
            # 不拖垮已经落盘的纠正。
            self._inherit_seen_evidence(old_meta, new_key, turn_id=turn_id)
            result = {
                "status": status, "memory_id": new_key,
                "old_text": old_text, "new_text": new_fact,
                "candidates": candidates or [],
                "reason": "已更正",
                "layer": layer,
                "resolution": resolution or "updated",
                "audit": audit or {"source": source, "method": "correct_memory"},
            }
            if slot_sweep:
                # R-1：主目标已顶掉，同槽其余真冲突旧值连带清扫
                #（门槛见 _sweep_extra_conflicts，防广州式误伤）。
                _slot, _value = _infer_fact_slot(new_fact)
                if _slot is not None:
                    _extras = [
                        c for c in self._find_same_slot_conflicts(
                            _slot, _value
                        )
                        if c[0] != old_id
                    ]
                    _swept = self._sweep_extra_conflicts(
                        _slot, _extras, new_key, new_fact,
                        turn_id=turn_id,
                    )
                    if _swept:
                        result["swept_ids"] = _swept
            return result

    def _find_correction_note_target(self, candidates, new_fact):
        """找模型写的纠正备注行（2026-11-23 A 组修复）：
        备注行是模型在 save_turn 之前自己调 add_memory 写下的，
        形如“BOOS 喜欢吃面条（纠正：不是喜欢吃米饭）”——它已经顶掉了
        旧说法，但 slot 是 null、相似度又被括号里的旧说法拉低，
        纠正分支认不出它，就会报 not_found。
        认法：文本含“纠正：”标记，且标记前面那半句与新说法归一化后
        相同（主语“我/BOOS”已由 _normalize_memory_text 剥掉）。
        只在带明确纠正词时用；找不到返回 None（调用方走老路，不硬猜）。
        真纠正便条（如“用户表示……纠正了此前‘喜欢吃鸡杂面’的说法”）
        没有“纠正：”冒号标记，不会误认。"""
        norm_new = _normalize_memory_text(new_fact)
        if not norm_new:
            return None
        for mid in candidates or []:
            text = self._candidate_text(mid)
            if not text:
                continue
            marker_idx = text.find("纠正：")
            if marker_idx < 0:
                marker_idx = text.find("纠正:")
            if marker_idx < 0:
                continue
            head = text[:marker_idx].rstrip("（(").strip()
            if not head:
                continue
            if _normalize_memory_text(head) == norm_new:
                return mid
        return None

    def _supersede_note_old_fact(self, old_fact_text, new_key, new_fact,
                                 turn_id=None):
        """备注行里被纠正掉的旧说法若还活着，一并取代（2026-11-23 A 组修复
        的兜底）：线上备注行写下时语义判官通常已顶掉旧行，这时旧行已不
        是活的，什么都不做；万一旧行还活着（判官没顶），按备注行里写的
        旧说法原样找出来顶掉——归一化后完全相同才顶，不猜。"""
        if not old_fact_text or not new_key:
            return
        norm_old = _normalize_memory_text(old_fact_text)
        if not norm_old:
            return
        try:
            rows = self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0 AND id != ?",
                (new_key,),
            ).fetchall()
        except Exception:
            return
        for row in rows:
            if _normalize_memory_text(row[1] or "") != norm_old:
                continue
            try:
                old_meta = json.loads(row[2] or "{}")
            except Exception:
                old_meta = {}
            try:
                # 推断记忆保护：备注兜底要顶的旧行是推断未确认时，
                # 改走待问（_supersede_or_ask 内部处理）。
                self._supersede_or_ask(
                    row[0], row[1] or "", old_meta, new_key, new_fact,
                    turn_id=turn_id,
                )
            except Exception:
                logger.warning("备注旧说法取代记账失败。", exc_info=True)

    def _explicit_correction_slot_target(self, candidates, new_fact):
        """带明确纠正词时按槽位找要改的那条（2026-11-23 model fix #2）：
        新说法和候选的 _correction_topic_slot 相同（住处/名字/喜欢吃……）
        时，取候选顺序里的第一条（候选已按已确认在前、同类里新的在前
        排过）；没有同槽位候选时返回 None（调用方走老路，不硬猜）。
        同槽多条时按相似度差距直选（2026-11-24 1c）：真记忆同槽永远
        多条（火锅/牛肉面/鸡杂面同为“喜欢吃”），原先“恰好 1 条才
        直选”在 live 里从未触发。改成：第一名 ≥ 0.70（沿用纠正安全
        网的线）且第二名 < 0.50（明显不像）才直选第一名；差距不够、
        或相似度算不出来时返回 None，走老路（相似度筛→ambiguous→
        判官），不猜一条改掉（2026-11-23 回归修的底线还在：
        “不对，我喜欢吃米线”对火锅/冒菜两条同槽旧行时相似度都
        不够线，必须报 ambiguous）。"""
        slot_key = _correction_topic_slot(new_fact)
        if slot_key is None:
            return None
        norm_new = _normalize_memory_text(new_fact)
        same_slot = []
        for mid in candidates or []:
            text = self._candidate_text(mid)
            if text is None:
                continue
            if _normalize_memory_text(text) == norm_new:
                continue
            if _correction_topic_slot(text) == slot_key:
                same_slot.append(mid)
        if len(same_slot) == 1:
            return same_slot[0]
        if not same_slot:
            return None
        scored = [
            (mid, self._candidate_similarity(mid, new_fact))
            for mid in same_slot
        ]
        if any(sim is None for _mid, sim in scored):
            return None
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if (scored[0][1] >= _CORRECTION_TOPIC_MIN_SIMILARITY
                and scored[1][1] < 0.50):
            return scored[0][0]
        return None


    def _already_corrected_receipt(self, new_fact, old_fact, turn_id=None):
        """新说法已是活记忆、且它顶掉旧说法的账本事件就是本轮（turn_id
        相同）的，说明纠正已在本轮完成过：模型生成回复时调 update_memory 先改
        了，save_turn 又调一次 correct_memory——两边都改就会误顶同槽位
        的另一条旧记忆（D 组把真记忆广州顶掉；E 组则报 not_found、回复
        丢了“改好了”）。此时不再另找目标，直接回 updated，old_text 就是
        账本里这行顶掉的旧说法。返回 None 表示没撞上这条路。
        只拦两种情况：(a) save_turn 的调用（old_fact == new_fact）；
        (b) 工具的重复调用（旧说法已不在库里活着）。旧说法还活着时不拦
        ——那是真纠正（例如新说法早就在库里、旧说法还活着），正常流程会
        顶掉旧行、复用新行。
        """
        norm_new = _normalize_memory_text(new_fact)
        live_id = None
        for row in self._conn.execute(
            "SELECT id, text FROM memories WHERE voided=0 AND deleted=0"
            " AND superseded=0"
        ).fetchall():
            if _normalize_memory_text(row[1] or "") == norm_new:
                live_id = row[0]
                break
        if live_id is None:
            return None
        # (b) 工具调用：旧说法还活着 → 真纠正，别拦。
        if _normalize_memory_text(old_fact) != norm_new:
            norm_old = _normalize_memory_text(old_fact)
            for row in self._conn.execute(
                "SELECT id, text FROM memories WHERE voided=0 AND deleted=0"
                " AND superseded=0"
            ).fetchall():
                if _normalize_memory_text(row[1] or "") == norm_old:
                    return None
        # 账本里找这行顶掉的旧说法（最近的一条 supersede 事件）。
        old_text = None
        sup_turn_id = None
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            return None
        if self._controlled_ledger_doubtful(_info):
            # 账本残缺（补修 H3 复查 F2）：「本轮已纠正过」
            # 是折叠结论，残缺账本定不了——不拦，交回正常
            # 纠正流程，不许拿残缺折叠挡真纠正。
            return None
        for ev in reversed(events):
            if (ev.get("type") == _ledger.LEDGER_EVENT_SUPERSEDE
                    and ev.get("memory_id") == live_id):
                old_text = ev.get("old_text")
                sup_turn_id = ev.get("turn_id")
                break
        if not old_text:
            return None
        # 只有本轮（turn_id 相同）的 supersede 才算“刚改过”（2026-11-24
        # 1b）。以前靠 5 分钟墙钟猜：上一轮刚改完、这一轮改回旧值时，
        # 最近一条 supersede 还在 5 分钟内，护栏会误拦，真该顶的那条
        # 顶不掉。没带 turn_id 的旧事件、和本轮对不上的，一律不拦，
        # 交给正常流程；turn_id 没传进来（认不出本轮）时也不拦。
        if not turn_id or sup_turn_id != turn_id:
            return None
        return {
            "status": "updated", "memory_id": live_id,
            "old_text": old_text, "new_text": new_fact,
            "candidates": [], "reason": "已改好（本轮已纠正过）",
            "layer": "already", "resolution": "already_corrected",
            "audit": {"source": "user", "method": "already_corrected"},
        }

    @staticmethod
    def _negation_conflicts(new_fact, old_text):
        """新说法和旧说法是不是“否定式正面冲突”（webfix 一）。

        程序信号，不猜语义：两边共享至少一个内容二字组；这个二字组
        在新说法里落在否定辖域内（前面 3 字内有 不/没/别），在旧
        说法里不在否定辖域内——新说法否定的正是旧说法肯定的东西
        （“不吃牛肉” vs “喜欢吃牛肉面”）。旧说法自己也带否定的
        （“不吃牛肉，吃鸡杂面”）不算冲突，那是同义/同向。
        """
        def _bigrams(s):
            return {s[i:i + 2] for i in range(len(s) - 1)}

        new_fact = new_fact or ""
        old_text = old_text or ""
        for bigram in _bigrams(new_fact) & _bigrams(old_text):
            i = new_fact.find(bigram)
            if not any(c in "不没别" for c in new_fact[max(0, i - 3):i]):
                continue
            j = old_text.find(bigram)
            if any(c in "不没别" for c in old_text[max(0, j - 3):j]):
                continue
            return True
        return False

    def _supersede_confirmed_contradictions(self, new_fact_text,
                                            new_key, turn_id=None):
        """同轮清扫已确认冲突（webfix 一）：把和新事实否定式冲突的
        “已确认”活记忆一并顶掉，指向已存好的新行（不另存重复行）。

        只顶 user_confirmed 的行：推断/未确认的行不在这里动——
        它们有推断保护和待问问题管。返回顶掉的清单
        [{"memory_id", "old_text"}]。单条失败只记日志，不拦其余。
        """
        out = []
        if not new_key or not (new_fact_text or "").strip():
            return out
        if not re.search(r"[不没别无]", new_fact_text):
            return out
        with self._lock:
            for m in self._all_live_memories():
                mid = m["id"]
                if mid == new_key:
                    continue
                meta = m["metadata"] or {}
                if not meta.get("user_confirmed"):
                    continue
                # G7 三道锁之一：核心/置顶行不进同轮清扫——自动
                # 纠正路径一律不碰核心行（与槽位裁决同一口径）。
                if self._is_core_protected_row(
                    meta, meta.get("slot") or _infer_fact_slot(m["text"] or "")[0]
                ):
                    continue
                old_text = m["text"] or ""
                if _dedupe_key(old_text) == _dedupe_key(new_fact_text):
                    continue
                if not self._negation_conflicts(new_fact_text, old_text):
                    continue
                try:
                    self._supersede_row(
                        mid, old_text, meta, new_key, new_fact_text,
                        turn_id=turn_id,
                    )
                except Exception:
                    logger.warning(
                        "已确认冲突清扫：取代 %s 失败。", mid, exc_info=True
                    )
                    continue
                out.append({"memory_id": mid, "old_text": old_text})
        return out

    def correct_memory(self, old_fact, new_fact, exclude_identical_to_new=True,
                       explicit_correction=False, turn_id=None):
        """纠错：把 old_fact 指的那条记忆改成 new_fact（整合方案 §6.3）。

        流程（和 mem0_store.correct_memory 一致）：
        先找候选（语义 + 关键词/同槽位 + 近期同一轮）；
        旧说法原样在库里 → 直接改那条；
        带明确纠正词时（explicit_correction=True，2026-11-23 model fix
        #2，A 组阈值）：按槽位找要改的那条，不走相似度 0.70 阈值——
        候选里同槽位的第一条就是目标，找不到同槽位候选时走下面老路；
        只有一条候选 → 过相似度安全网（余弦 ≥ 0.70 才改，否则 ambiguous，
        免得改错不相干的记忆；向量算不出来时跳过安全网，交由调用方兜底）；
        多条候选、没有原样 → 先按相似度筛（根因 b，2026-11-22）：只留
        与新说法相似度 ≥ 0.70 的；筛后剩 1 条就按单候选走，剩 0 条报
        not_found，仍剩多条再报 ambiguous；向量算不出来时不筛，直接
        ambiguous（不猜，留给调用方问）；
        一条没有 → not_found（调用方按新记忆存）；
        新说法和旧的一样 → unchanged；新说法含敏感信息 → failed。
        纠错是追加式（整合方案）：旧记忆不改原文、不删，记 supersede，
        新说法另存一条。返回 mem0 同款回执：
        {status, memory_id, old_text, new_text, candidates, reason,
         layer, resolution, audit}。
        candidates 是带 id 和 text 的字典列表（2026-11-22 Bug 1：以前是
        光秃秃的 ID 串，打印层按字典取 text 就崩成 500）。
        """
        with self._lock:
            new_fact = (new_fact or "").strip()
            old_fact = (old_fact or "").strip()
            if _contains_sensitive(new_fact):
                return {
                    "status": "failed", "memory_id": None,
                    "old_text": old_fact, "new_text": new_fact,
                    "candidates": [],
                    "reason": "新说法含敏感信息", "layer": "none",
                    "resolution": "blocked",
                    "audit": {"source": "user", "method": "correct_memory"},
                }
            if turn_id is None:
                # 工具路（update_memory）没有轮次上下文：用当前轮登记
                # （search / save_turn 登记的 _active_turn_id）兜底，这样
                # 工具先改、save_turn 再调时，护栏和账本都认得出本轮。
                turn_id = getattr(self, "_active_turn_id", None)
            # R-1（2026-11-24）：old 与 new 同文（save_turn 的
            # correct_memory(fact, fact) 路）是一次“槽位型纠正”——
            # BOOS 直接报了槽位的新值、没点名旧值，纠正的是整个
            # 槽位：成功后同槽其余真冲突旧值走连带清扫。点名了
            # 旧值的纠正（old != new）是外科手术式，只改被点名
            # 的那条，不清扫（group1 槽位直选的护栏）。
            slot_sweep = (
                _normalize_memory_text(old_fact)
                == _normalize_memory_text(new_fact)
            )
            already = self._already_corrected_receipt(
                new_fact, old_fact, turn_id=turn_id
            )
            if already is not None:
                return already
            candidates = self._find_correction_candidates(
                old_fact, new_fact=new_fact,
                exclude_identical_to_new=exclude_identical_to_new,
            )
            if not candidates:
                if _normalize_memory_text(old_fact) == _normalize_memory_text(new_fact):
                    # 新旧说法相同：候选过滤已跳过与新说法相同的条目；库里
                    # 已是这句、又没有别的冲突记忆时，无事可做，报 unchanged
                    #（而不是 not_found——“已经是对的”不是“找不到”）。
                    for _row in self._conn.execute(
                        "SELECT id, text, metadata_json FROM memories WHERE voided=0"
                        " AND deleted=0 AND superseded=0"
                    ).fetchall():
                        if _normalize_memory_text(_row[1] or "") == _normalize_memory_text(new_fact):
                            return self._apply_correction_result(
                                new_fact, _row[0], _row[1],
                                json.loads(_row[2] or "{}"),
                                same_text=True, candidates=[], status="unchanged",
                                resolution="exact", layer="exact",
                                audit={"source": "user", "method": "exact_match"},
                                turn_id=turn_id, slot_sweep=slot_sweep,
                            )
                # 第 5 条跨槽安全网（2026-11-24）：主路一条候选都
                # 没找到时，先别下 not_found——对全部活记忆再扫一遍
                # （高门槛 + 判官确认），捞到真冲突就按那条落盘；
                # 捞不到才维持原判。
                net = self._cross_slot_net_target(old_fact, new_fact)
                if net is not None:
                    mid, old_text, old_meta, verdict = net
                    return self._apply_correction_result(
                        new_fact, mid, old_text, old_meta, same_text=False,
                        candidates=[mid], status="updated",
                        resolution="cross_slot_net", layer="net",
                        audit={"source": "user", "method": "cross_slot_net",
                               "provider": verdict.get("provider"),
                               "model": verdict.get("model"),
                               "judge_slot": verdict.get("slot")},
                        turn_id=turn_id, slot_sweep=slot_sweep,
                    )
                return {
                    "status": "not_found", "memory_id": None,
                    "old_text": old_fact, "new_text": new_fact,
                    "candidates": [],
                    "reason": "未找到可纠正的记忆", "layer": "none",
                    "resolution": "not_found",
                    "audit": {"source": "user", "method": "correct_memory"},
                }
            # 原样匹配：旧说法原样在库里，直接改那条（不用过安全网）。
            exact = None
            for mid in candidates:
                row = self._conn.execute(
                    "SELECT text, metadata_json FROM memories WHERE id=?", (mid,)
                ).fetchone()
                if row and (row[0] or "").strip() == old_fact:
                    exact = (mid, row[0], json.loads(row[1] or "{}"))
                    break
            if exact is not None:
                mid, old_text, old_meta = exact
                same = _normalize_memory_text(new_fact) == _normalize_memory_text(old_text)
                if same:
                    return self._apply_correction_result(
                        new_fact, mid, old_text, old_meta, same_text=True,
                        candidates=candidates, status="unchanged",
                        resolution="exact", layer="exact",
                        audit={"source": "user", "method": "exact_match"},
                        turn_id=turn_id, slot_sweep=slot_sweep,
                    )
                return self._apply_correction_result(
                    new_fact, mid, old_text, old_meta, same_text=False,
                    candidates=candidates, status="updated",
                    resolution="exact", layer="exact",
                    audit={"source": "user", "method": "exact_match"},
                    turn_id=turn_id, slot_sweep=slot_sweep,
                )
            # 2026-11-23 model fix #2（A 组阈值）：带明确纠正词的，不走
            # 相似度 0.70 阈值，直接按槽位找要改的那条——候选里同槽位的
            # 第一条（候选已按已确认在前、同类里新的在前排过）。找不到
            # 同槽位候选时走下面的老路（相似度/判官），不硬猜。
            if explicit_correction:
                # 2026-11-23 A 组修复：先认模型已写的纠正备注行——
                # 模型在 save_turn 之前自己调 add_memory 写下
                # “BOOS 喜欢吃面条（纠正：不是喜欢吃米饭）”，已经顶掉
                # 了米饭；但备注行 slot 是 null、相似度又被括号拉低，
                # 槽位直选和相似度筛都认不出它，纠正分支会报 not_found，
                # 新说法又被 _dedupe_key 当成备注行的重复而不存。
                # 认出来就用 exact 新说法取代备注行，回执 updated；
                # 备注行里被纠正掉的旧说法（米饭）若还活着，也一并取代。
                note_target = self._find_correction_note_target(
                    candidates, new_fact
                )
                if note_target is not None:
                    mid = note_target
                    row = self._conn.execute(
                        "SELECT text, metadata_json FROM memories WHERE id=?",
                        (mid,),
                    ).fetchone()
                    if row is not None:
                        old_text, old_meta = row[0], json.loads(row[1] or "{}")
                        same = (
                            _normalize_memory_text(new_fact)
                            == _normalize_memory_text(old_text)
                        )
                        result = self._apply_correction_result(
                            new_fact, mid, old_text, old_meta, same_text=same,
                            candidates=candidates,
                            status="unchanged" if same else "updated",
                            resolution="explicit_note", layer="note",
                            audit={
                                "source": "user",
                                "method": "explicit_correction_note",
                            },
                            turn_id=turn_id, slot_sweep=slot_sweep,
                        )
                        if (result or {}).get("status") == "updated":
                            extracted_old = _extract_note_old_fact(old_text)
                            if extracted_old:
                                result["old_text"] = extracted_old
                                self._supersede_note_old_fact(
                                    extracted_old,
                                    (result or {}).get("memory_id"),
                                    new_fact, turn_id=turn_id,
                                )
                        return result
                slot_target = self._explicit_correction_slot_target(
                    candidates, new_fact
                )
                if slot_target is not None:
                    mid = slot_target
                    row = self._conn.execute(
                        "SELECT text, metadata_json FROM memories WHERE id=?",
                        (mid,),
                    ).fetchone()
                    if row is not None:
                        old_text, old_meta = row[0], json.loads(row[1] or "{}")
                        same = (
                            _normalize_memory_text(new_fact)
                            == _normalize_memory_text(old_text)
                        )
                        return self._apply_correction_result(
                            new_fact, mid, old_text, old_meta, same_text=same,
                            candidates=candidates,
                            status="unchanged" if same else "updated",
                            resolution="explicit_slot", layer="slot",
                            audit={
                                "source": "user",
                                "method": "explicit_correction_slot",
                                "slot": _correction_topic_slot(new_fact),
                            },
                            turn_id=turn_id, slot_sweep=slot_sweep,
                        )
            # 多条候选、没有原样：先按相似度筛（根因 b，2026-11-22）——
            # 0.4 的召回线太宽，“我喜欢…”这种短句会召回一堆无关行；
            # 只留与新说法相似度 ≥ 0.70 的（沿用已有的安全网标准）。
            # 向量算不出来（sim 是 None）时不筛：宁可 ambiguous 问
            # BOOS，也不凭空猜。
            chosen = None
            if len(candidates) > 1:
                scored = [
                    (mid, self._candidate_similarity(mid, new_fact))
                    for mid in candidates
                ]
                high = [
                    mid for mid, sim in scored
                    if sim is not None and sim >= _CORRECTION_TOPIC_MIN_SIMILARITY
                ]
                unknown = [mid for mid, sim in scored if sim is None]
                if len(high) >= 2:
                    return {
                        "status": "ambiguous", "memory_id": None,
                        "old_text": old_fact, "new_text": new_fact,
                        "candidates": self._candidates_with_text(high),
                        "reason": "多条候选与新说法相似度都 ≥ 0.70，无法确定改哪一条",
                        "layer": "none", "resolution": "ambiguous",
                        "audit": {"source": "user", "method": "correct_memory"},
                    }
                if len(high) == 1 and not unknown:
                    chosen = high[0]
                elif len(high) == 0 and not unknown:
                    # R-9（2026-11-24）：下 not_found 结论前补跑一次
                    # 同槽兜底——召回有命中但全不相关时，真目标可能
                    # 不在候选里（兜底只在召回为空时才跑）。兜底恰好
                    # 捞出一条过线的，按单候选继续走（还要过下面的
                    # 安全网）；捞不出就维持原判。
                    rescued = self._rescue_correction_candidate(
                        old_fact, new_fact, candidates
                    )
                    if rescued is not None:
                        chosen = rescued
                    else:
                        # 第 5 条跨槽安全网（2026-11-24）：同槽兜底也
                        # 落空了，对全部活记忆再扫一遍（高门槛 + 判官
                        # 确认）；捞到真冲突就直接按那条落盘（判官
                        # 确认就是这条路的安全网，不再回下面的相似度
                        # 安全网——它过的就是 ≥0.80 的更高门槛），
                        # 捞不到维持原判 not_found。
                        net = self._cross_slot_net_target(old_fact, new_fact)
                        if net is not None:
                            mid, old_text, old_meta, verdict = net
                            return self._apply_correction_result(
                                new_fact, mid, old_text, old_meta,
                                same_text=False, candidates=[mid],
                                status="updated",
                                resolution="cross_slot_net", layer="net",
                                audit={"source": "user",
                                       "method": "cross_slot_net",
                                       "provider": verdict.get("provider"),
                                       "model": verdict.get("model"),
                                       "judge_slot": verdict.get("slot")},
                                turn_id=turn_id, slot_sweep=slot_sweep,
                            )
                        best = max(sim for _mid, sim in scored)
                        return {
                            "status": "not_found", "memory_id": None,
                            "old_text": old_fact, "new_text": new_fact,
                            "candidates": self._candidates_with_text(candidates),
                            "reason": "候选与新说法相似度均低于 0.70（最高 %.2f），未找到可纠正的记忆" % best,
                            "layer": "none", "resolution": "not_found",
                            "audit": {"source": "user", "method": "correct_memory"},
                        }
                else:
                    return {
                        "status": "ambiguous", "memory_id": None,
                        "old_text": old_fact, "new_text": new_fact,
                        "candidates": self._candidates_with_text(candidates),
                        "reason": "多条候选，无法确定改哪一条",
                        "layer": "none", "resolution": "ambiguous",
                        "audit": {"source": "user", "method": "correct_memory"},
                    }
            else:
                chosen = candidates[0]
            # 单条候选（或多候选筛后只剩 1 条）：过相似度安全网。
            mid = chosen
            row = self._conn.execute(
                "SELECT text, metadata_json, embedding FROM memories WHERE id=?", (mid,)
            ).fetchone()
            if row is None:
                return {
                    "status": "not_found", "memory_id": None,
                    "old_text": old_fact, "new_text": new_fact,
                    "candidates": self._candidates_with_text(candidates),
                    "reason": "候选记忆已不在库里", "layer": "none",
                    "resolution": "not_found",
                    "audit": {"source": "user", "method": "correct_memory"},
                }
            old_text, old_meta = row[0], json.loads(row[1] or "{}")
            same = _normalize_memory_text(new_fact) == _normalize_memory_text(old_text)
            if same:
                return self._apply_correction_result(
                    new_fact, mid, old_text, old_meta, same_text=True,
                    candidates=candidates, status="unchanged",
                    resolution="semantic", layer="single",
                    audit={"source": "user", "method": "single_candidate"},
                    turn_id=turn_id, slot_sweep=slot_sweep,
                )
            try:
                # 安全网比的是候选旧记忆和新说法的话题相似度，不是两个入参
                #（save_turn 传的是 correct_memory(fact, fact)，拿入参比
                # 永远是 1.0）。优先用库里存着的向量，没有再现算。
                old_vec = _blob_to_vector(row[2])
                if old_vec is None:
                    old_vec = self._embed_text(old_text)
                sim = _cosine(old_vec, self._embed_text(new_fact))
            except Exception:
                sim = None
            if sim is not None and sim < _CORRECTION_TOPIC_MIN_SIMILARITY:
                return {
                    "status": "ambiguous", "memory_id": None,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": self._candidates_with_text(candidates),
                    "reason": "与候选记忆相似度过低（%.2f），不敢改" % sim,
                    "layer": "none", "resolution": "ambiguous",
                    "audit": {"source": "user", "method": "safety_net",
                              "similarity": sim},
                }
            return self._apply_correction_result(
                new_fact, mid, old_text, old_meta, same_text=False,
                candidates=candidates, status="updated",
                resolution="semantic", layer="single",
                audit={"source": "user", "method": "single_candidate",
                       "similarity": sim},
                turn_id=turn_id, slot_sweep=slot_sweep,
            )

    def _void_last_stored(self):
        """作废上一轮刚存的记忆：账本记 delete，SQLite 标 voided。

        若记录了上一轮的 turn_id（_last_stored_turn_id），则把该轮存下的
        所有条目整轮作废（同一轮多句话拆条写入时不会只作废一半）；
        否则退化为作废最新写入的那 1 条（按 write_seq DESC, rowid DESC）。
        """
        with self._lock:
            candidates = self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
                " ORDER BY write_seq DESC, rowid DESC"
            ).fetchall()
            if not candidates:
                return 0
            rows = []
            if self._last_stored_turn_id:
                rows = [
                    r for r in candidates
                    if json.loads(r[2] or "{}").get("turn_id")
                    == self._last_stored_turn_id
                ]
            if not rows:
                rows = [candidates[0]]
            voided_count = 0
            for row in rows:
                meta = json.loads(row[2] or "{}")
                try:
                    _ledger.append_event(
                        self.ledger_path,
                        {"type": "delete", "memory_id": row[0],
                         "key": row[0], "text": row[1], "meta": meta},
                        lock=self._ledger_lock,
                    )
                except Exception:
                    logger.warning("撤回记账失败。", exc_info=True)
                    continue
                self._conn.execute(
                    "UPDATE memories SET voided=1 WHERE id=?", (row[0],)
                )
                voided_count += 1
            if voided_count:
                self._conn.commit()
                self._last_stored_turn_id = None
            return voided_count

    def flush_pending(self):
        """同步存的路没有待定区：什么都不用等，直接返回 {}（接口兼容）。"""
        return {}

    def ensure_outbox_started(self):
        """同步存的路没有 outbox：什么都不用启动（接口兼容）。"""
        return None

    # ---------- 读 ----------

    def _all_live_memories(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, metadata_json, created_at, write_seq, slot,"
                " source, user_confirmed, decay_kind, importance, event_date,"
                " embedding FROM memories WHERE voided=0 AND deleted=0 AND superseded=0"
                " ORDER BY created_at DESC"
            ).fetchall()
            out = []
            for r in rows:
                meta = _normalize_metadata(json.loads(r[2] or "{}"))
                out.append({
                    "id": r[0], "text": r[1], "metadata": meta,
                    "created_at": r[3], "write_seq": r[4], "slot": r[5],
                    "source": r[6], "user_confirmed": bool(r[7]),
                    "decay_kind": r[8], "importance": r[9],
                    "event_date": r[10], "embedding": r[11],
                })
            return out

    def get_all(self, include_sensitive=False):
        items = [
            {
                "id": m["id"], "memory": m["text"], "text": m["text"],
                "metadata": m["metadata"], "created_at": m["created_at"],
            }
            for m in self._all_live_memories()
        ]
        if include_sensitive:
            return items
        # #1 默认不返回敏感内容：页面列表/前台展示都不该看到密码/卡号。
        # include_sensitive=True 只给内部用（纠错归属校验、迁移/删除校验）。
        return [
            it for it in items
            if not _contains_sensitive(it["text"] or "")
        ]

    def confirmed_facts(self):
        """当前全部「已确认事实」活行（回锚注入用，见 memory/reanchor.py）。

        认定沿现行口径（reanchor.is_anchor_fact）：confirmed
        （user_confirmed，含问题一 owner_confirmed 升级后写回的
        user_confirmed=True）或核心受保护（身份槽位 / is_rule /
        curated）。只返回活行（未作废、未删、未被取代），形状同
        _all_live_memories。这里不截断——挑哪几条、怎么封顶由
        reanchor.select_anchor_facts 按它写明的取舍规矩定。
        """
        return [
            m for m in self._all_live_memories()
            if _reanchor.is_anchor_fact(m)
        ]

    def search(self, query, top_k=5, reinforce=False, include_history=False,
               turn_id=None, scope=None):
        """搜记忆：宽召回（语义 top_k*4 + BM25 全库 top_k*4）→ RRF k=60
        融合 → 原样匹配在前 → 截断 top_k，不够再账本兜底（见 _search_inner，
        旧 mem0 search 的排名逻辑）。

        返回和 Mem0Store.search 一样的形状：
        {"id","text","score","source","metadata","confirmed",
         "decay_weight","is_history"}。阈值 0.4，和原来一样。
        """
        query = (query or "").strip()
        if not query:
            return []
        if turn_id is not None:
            # 登记当前轮（2026-11-24 1b）：ask() 的检索在本轮最前面，
            # 登记后本轮内工具调 correct_memory 没带 turn_id 时认得出本轮。
            self._active_turn_id = turn_id
        try:
            results = self._search_inner(
                query, top_k=top_k, include_history=include_history,
                scope=scope,
            )
        except Exception:
            logger.warning("SQLite 检索失败。", exc_info=True)
            return []
        # 冲突通道（webfix 三）：对外检索（给回答用的）才做——
        # 同话题冲突标注 + pending 新事实换入；内部候选检索
        # （_search_inner 的其他调用方）不走这里，免得干扰纠错
        # 候选逻辑。标注只改 metadata 里的冲突标记，不动返回形状。
        results = self._annotate_search_conflicts(results)
        if reinforce and results:
            # 真用上了才强化（2026-11-24 3a，和 Mem0Store.search 的
            # reinforce 同款语义）：以前 SQLite 的 search 接了
            # reinforce/turn_id 却没往下传，强化从没发生过。SQLite
            # 这边由 note_context 一并做（刷新 last_used_at、沉底
            # 复活、记账本 TOUCH）；历史结果不强化——翻出来看看
            # 不算用上。写回失败不拦检索。
            try:
                self.note_context(
                    [r for r in results if not (r or {}).get("is_history")]
                )
            except Exception:
                logger.warning("检索强化写回失败：记忆本身还在。", exc_info=True)
        return results

    def _search_inner(self, query, top_k=5, include_history=False,
                      scope=None):
        """宽召回 → RRF 融合 → 先筛后截（旧 mem0 search 的排名逻辑，
        commit 7a69699“宽召回 + RRF 融合”，mem0_store.py search）。

        三路排名：语义（原始余弦分 ≥ 0.4，取 top_k*4）、BM25（全库打分，
        取 top_k*4）、图谱（一跳联想扩展，二期接通）。标准 RRF k=60 融合，再乘
        衰减权重 × 类型权重（稳定偏好 1.25，见 _type_weight，2026-11-21
        Q3 修法）；原样匹配（基金代码/数字/整句，见
        memory/bm25.extract_exact_matches）的排在前面。
        和旧逻辑的对应关系：
          - 语义路按原始余弦分排位（旧 mem0 的语义路按向量原始分排，
            不是 0.6*cos+0.4*kw 的混合分——混合分里的关键词成分现在
            由 BM25 那路单独出，再混进语义路等于关键词算两遍；实测
            这么排会把万家那行从语义第 4 挤到第 5、融合后挤出前 5）。
            资格仍看原始余弦 ≥ 0.4；score 字段仍记混合分（第一部分
            的返回形状不动）。
          - 原样整句命中（query 在文本里或文本在 query 里）仍记 1.0 分、
            source="verbatim"，排在语义路最前面（第一部分的规矩，旧逻辑
            里对应 exact_match 置顶）。
          - BM25 命中不是语义分：score 记关键词重合度（旧关键词兜底
            同款 0–1 分），只做展示、不参与排位；source="keyword"。
          - 先筛后截：SQLite 索引里只有当前态（_all_live_memories 只取
            voided=0 AND deleted=0，对账也只重建当前态），投影过滤和
            档案闸在这里是空操作——被取代/删了的行根本不在候选池里，
            也没有沉底态（decay_state）可拦；衰减接进来时闸门加在这里、
            截断之前。精选先查（旧逻辑第 0 步）也没接：库里 curated 行
            是 0 条，等精选清单那步再接。
          - 图谱路（二期）：种子 = 查询抽到的实体 + 语义/BM25 已捞到
            的记忆，顺有效边一跳扩展（同义词边按 scope 过滤、取代边
            定向、冲突边不走、作废边不走）；图上才找得到的记忆以
            source="graph" 进候选，score 只做展示（关键词重合度，
            与 BM25 路同口径），排位交给 RRF。扩展纯读，不改图、
            不改库；图坏了只丢这一路（和旧口径一致）。
        """
        with self._lock:
            import numpy as np

            try:
                from memory import bm25 as _bm25
            except Exception:
                logger.warning("BM25 模块读不到：本次只用语义单路检索。", exc_info=True)
                _bm25 = None

            # #1 检索是拼进提示词的路：含敏感信息的条目不进候选池——已入库的
            # 旧敏感记忆（比如迁移带进来的）也不注入。
            live = [
                m for m in self._all_live_memories()
                if not _contains_sensitive(m["text"] or "")
            ]
            if not live:
                return []
            # 同一文本多行时（比如“包子”既有 BOOS 确认版、又有推断版），
            # 只留一行进检索：确认版优先、再按最新；未确认版不再占名额
            # （2026-11-21 网页测第二轮 #2：确认过的记忆不能输给推断版）。
            # live 已按 created_at 倒序：同文本先出现的是最新的。
            _by_text = {}
            for m in live:
                key = m["text"] or ""
                prev = _by_text.get(key)
                if prev is None:
                    _by_text[key] = m
                elif m["user_confirmed"] and not prev["user_confirmed"]:
                    _by_text[key] = m
            live = list(_by_text.values())
            wide_k = max(top_k * 4, top_k)
            qvec = self._embed_text(query)
            candidates = {}
            semantic_ranked = []
            bm25_ranked = []
            seen_texts = set()
            by_id = {m["id"]: m for m in live}

            def _result_item(m, score, source):
                meta = dict(m["metadata"] or {})
                return {
                    "id": m["id"], "text": m["text"], "score": round(score, 4),
                    "source": source, "metadata": meta,
                    "confirmed": bool(m["user_confirmed"]),
                    "decay_weight": _decay_weight(meta), "is_history": False,
                    # 类型权重只进排序键，不进返回字段（返回形状还是 8 个键，
                    # 见 test_search_result_shape_unchanged）；出结果前 pop 掉。
                    "_type_weight": _type_weight(m["text"]),
                }

            # 语义路：原始余弦分排位，取前 wide_k；原样命中的先放最前面。
            sem_scored = []
            verbatim_first = []
            for m in live:
                text = m["text"] or ""
                verbatim = query in text or text in query
                cos = None
                if qvec is not None and m["embedding"] is not None:
                    vec = _blob_to_vector(m["embedding"])
                    if vec is not None and vec.shape == qvec.shape:
                        cos = float(np.dot(qvec, vec))
                if verbatim:
                    verbatim_first.append(m)
                    continue
                if cos is None or cos < SEARCH_THRESHOLD:
                    continue
                kw = _keyword_overlap(query, text)
                sem_scored.append((cos, 0.6 * cos + 0.4 * kw, m))
            # 同分时确认优先（#2）：余弦相等时确认行在语义路里先占位。
            sem_scored.sort(key=lambda item: (-item[0], 0 if item[2]["user_confirmed"] else 1))
            for m in verbatim_first:
                if m["text"] in seen_texts:
                    continue
                seen_texts.add(m["text"])
                candidates[m["id"]] = _result_item(m, 1.0, "verbatim")
                semantic_ranked.append(m["id"])
            for _cos, score, m in sem_scored[:wide_k]:
                if m["text"] in seen_texts:
                    continue
                seen_texts.add(m["text"])
                candidates[m["id"]] = _result_item(m, score, "semantic")
                semantic_ranked.append(m["id"])

            # BM25 路：全库打分取前 wide_k，和语义平等进 RRF。
            exact_ids = set()
            if _bm25 is not None:
                try:
                    docs = [{"id": m["id"], "text": m["text"] or ""} for m in live]
                    exact_ids = set(_bm25.extract_exact_matches(query, docs))
                    # 同分时确认优先（#2）：BM25 分完全相等时确认行在路里先占位，
                    # 否则 RRF 会把后加的推断行排到确认行前面（live 是倒序）。
                    _bm25_hits = sorted(
                        _bm25.bm25_search(query, docs, top_k=wide_k),
                        key=lambda h: (-h["score"], 0 if (by_id.get(h["id"]) or {}).get("user_confirmed") else 1),
                    )
                    for hit in _bm25_hits:
                        doc_id = hit["id"]
                        m = by_id.get(doc_id)
                        if m is None:
                            continue
                        if doc_id in candidates:
                            candidates[doc_id]["exact_match"] = doc_id in exact_ids
                            bm25_ranked.append(doc_id)
                            continue
                        if m["text"] in seen_texts:
                            continue
                        seen_texts.add(m["text"])
                        candidates[doc_id] = _result_item(
                            m, _keyword_overlap(query, m["text"] or ""), "keyword"
                        )
                        bm25_ranked.append(doc_id)
                except Exception:
                    logger.warning("BM25 检索失败：本次只用语义路结果。", exc_info=True)
            for cid, item in candidates.items():
                item.setdefault("exact_match", cid in exact_ids)

            # 图谱路（二期）：一跳联想扩展，见上面 docstring 的口径。
            graph_ranked = []
            try:
                graph = self._get_graph()
                seed_entities = [
                    name for name, _kind in _extract_entities_free(query)
                ]
                seed_keys = [
                    mid
                    for mid in list(semantic_ranked) + list(bm25_ranked)
                    if mid in by_id
                ]
                expanded = graph.expand_one_hop(
                    seed_keys=seed_keys,
                    seed_entities=seed_entities,
                    scope=scope,
                )
                for gkey, _gscore in expanded.items():
                    m = by_id.get(gkey)
                    if m is None:
                        continue
                    graph_ranked.append(gkey)
                    if gkey in candidates:
                        continue
                    if m["text"] in seen_texts:
                        continue
                    seen_texts.add(m["text"])
                    candidates[gkey] = _result_item(
                        m, _keyword_overlap(query, m["text"] or ""), "graph"
                    )
            except Exception:
                logger.warning("图谱检索失败：本次只用语义/BM25 结果。", exc_info=True)
                graph_ranked = []
            if _bm25 is not None:
                fused = _bm25.reciprocal_rank_fusion(
                    [semantic_ranked, bm25_ranked, graph_ranked], k=60
                )
            else:
                fused = [(doc_id, 0.0) for doc_id in semantic_ranked]
            fused_items = []
            for doc_id, rrf in fused:
                item = candidates.get(doc_id)
                if item is None:
                    continue
                item["_rrf"] = rrf
                fused_items.append(item)
            fused_items.sort(
                key=lambda item: (
                    1 if item.get("exact_match") else 0,
                    (item.get("_rrf") or 0) * (item.get("decay_weight") or 0)
                    * (item.get("_type_weight") or 1.0),
                    # 确认优先只当平局裁决（#2）：同分时确认过的排前面。
                    1 if item.get("confirmed") else 0,
                ),
                reverse=True,
            )
            results = []
            for item in fused_items:
                item.pop("_rrf", None)
                item.pop("_type_weight", None)
                item.pop("exact_match", None)
                results.append(item)
            results = results[:top_k]
            seen = {r["text"] for r in results}
            # 账本兜底：不够 top_k 时翻账本原话按关键词找。
            # 只找当前态的：被取代/删了的不补进来。
            if len(results) < top_k:
                try:
                    state = self.ledger_state()
                    key_states = (state or {}).get("key_states") or {}
                    events = []
                    for rec in (state or {}).get("adds") or []:
                        if key_states.get(rec.get("key")) in (
                            _ledger.LEDGER_STATE_SUPERSEDED,
                            _ledger.LEDGER_STATE_DELETED,
                        ):
                            continue
                        events.append({
                            "type": _ledger.LEDGER_EVENT_ADD,
                            "key": rec.get("key"), "text": rec.get("text") or "",
                            "meta": rec.get("meta") or {},
                        })
                    for hit in _ledger.find_in_ledger(
                        events, query, top_k=top_k - len(results),
                        exclude_keys={r["id"] for r in results},
                    ):
                        text = hit.get("text") or ""
                        if not text or text in seen:
                            continue
                        if _contains_sensitive(text):
                            continue
                        meta = _normalize_metadata(hit.get("meta") or {})
                        results.append({
                            "id": None, "text": text,
                            "score": hit.get("score"), "source": "ledger",
                            "metadata": meta,
                            "confirmed": bool(meta.get("user_confirmed")),
                            "decay_weight": 1.0, "is_history": True,
                            "ledger_key": hit.get("key"),
                        })
                        seen.add(text)
                except Exception:
                    logger.warning("账本兜底检索失败。", exc_info=True)
            # 历史查询（2026-11-22，Gemini #5）：include_history=True 时，
            # 在活记忆后面补账本里被取代/删了的旧版本（翻找 supersede/delete
            # 事件），标 is_history=True、source="history"。活记忆不动、
            # 顺序不动，历史只追加；默认（include_history=False）不补，
            # 免得旧版本混进正常回答。
            if include_history:
                try:
                    state = self.ledger_state()
                    key_states = (state or {}).get("key_states") or {}
                    history_events = []
                    for rec in (state or {}).get("adds") or []:
                        if key_states.get(rec.get("key")) in (
                            _ledger.LEDGER_STATE_SUPERSEDED,
                            _ledger.LEDGER_STATE_DELETED,
                        ):
                            history_events.append({
                                "type": _ledger.LEDGER_EVENT_ADD,
                                "key": rec.get("key"),
                                "text": rec.get("text") or "",
                                "meta": rec.get("meta") or {},
                            })
                    for hit in _ledger.find_in_ledger(
                        history_events, query, top_k=top_k,
                        exclude_keys={r["id"] for r in results},
                    ):
                        text = hit.get("text") or ""
                        if not text or text in seen:
                            continue
                        if _contains_sensitive(text):
                            continue
                        meta = _normalize_metadata(hit.get("meta") or {})
                        results.append({
                            "id": hit.get("key"), "text": text,
                            "score": hit.get("score"), "source": "history",
                            "metadata": meta,
                            "confirmed": bool(meta.get("user_confirmed")),
                            "decay_weight": 1.0, "is_history": True,
                            "ledger_key": hit.get("key"),
                        })
                        seen.add(text)
                except Exception:
                    logger.warning("历史检索失败。", exc_info=True)
            return results

    @staticmethod
    def _content_bigrams(text):
        """内容二字组：两个字都是汉字、且不含偏好脚手架字
        （喜欢/爱/讨厌）——“BOOS 喜欢吃 X”里真正区分内容的是
        X 部分的二字组（鸡杂、杂面），脚手架人人相同、不算数。"""
        s = text or ""
        out = set()
        for i in range(len(s) - 1):
            a, b = s[i], s[i + 1]
            if not ("一" <= a <= "龥" and "一" <= b <= "龥"):
                continue
            if a in "喜欢爱讨厌" or b in "喜欢爱讨厌":
                continue
            out.add(s[i:i + 2])
        return out

    @classmethod
    def _same_conflict_topic(cls, a, b):
        """两条记忆是不是“同一件事的两种说法”（webfix 三成组用）。

        只认硬信号，不猜：单值槽（住处/名字）同槽不同值；或否定式
        正面冲突（_negation_conflicts 任一方向）。同是“喜欢吃”
        槽但吃的东西不一样（冒菜 vs 火锅）不是冲突——那是两个
        偏好，可以并存，不许标冲突。
        """
        slot_a, val_a = _infer_fact_slot(a)
        slot_b, val_b = _infer_fact_slot(b)
        if slot_a is not None and slot_a == slot_b:
            return (
                _normalize_memory_text(val_a or "")
                != _normalize_memory_text(val_b or "")
            )
        return (
            cls._negation_conflicts(a, b)
            or cls._negation_conflicts(b, a)
        )

    def _annotate_search_conflicts(self, results):
        """检索收尾的冲突通道（webfix 三，只改善不根治）。

        0. 槽位二次整理（2026-11-24 漏答优化）：见 _slot_recompose
           ——某槽的现行条没进结果、同槽旧冲突行进了，把现行条
           换进来；现行条在结果里也不许排在旧行后面。
        1. pending 不隐身：未答冲突问题的新事实（new_id）不在
           结果里、但旧行在结果里（或与结果任一条同话题/共享
           内容二字组）时，把它换进来（挤掉最末一条非规则、
           非历史的结果）——新事实已存活，就不许在回答里隐身
           （实测漏答根因之一）。
        2. 冲突标注，成组只认硬信号：
           - 待问问题亲口认定的那一对（old_id/new_id）是种子组，
             再把与新行共享内容二字组的其余结果并进来（同一件
             事的其它说法）；
           - 没有待问问题时，靠 _same_conflict_topic 的硬信号
             （单值槽同槽不同值 / 否定式正面冲突）成组。
           组内以（已确认优先、write_seq/created_at 最新）定
           “现行”：现行标 conflict_current，其余标 conflict_stale
           + conflict_with=<现行文本>。main._memory_block 据此
           给模型加一句“以最新确认的为准”，不让它面对互相
           矛盾的记录把整个话题跳过。
        全程失败只记日志，照常返回原结果；不动返回形状。
        """
        try:
            items = [
                r for r in results
                if not r.get("is_history") and (r.get("text") or "")
            ]
            try:
                questions = self.pending_questions.list_unanswered()
            except Exception:
                questions = []
            conflict_questions = [
                q for q in questions
                if q.get("kind") == _pq.QUESTION_KIND_CONFLICT
            ]
            present = {r.get("id") for r in items}
            by_id = {r.get("id"): r for r in items}
            # --- 0. 槽位二次整理（2026-11-24，BOOS 定“漏答再
            # 优化”）：按话题粗槽分组——某槽的现行条（已确认优先、
            # 同级最新）没进结果、同槽的旧冲突行却进了，就把现行
            # 条换进来占旧行的位置；现行条在结果里但排在旧行后面
            # 的，提到旧行前面。已取代的行本来就不在活行里，这里
            # 管的是“还活着但已经输了”的旧行（待问挂起、没纠正
            # 词的新旧并存）：它们不许压过现行条。失败只记日志，
            # 不连累后面的 pending 换入和冲突标注。
            try:
                self._slot_recompose(
                    results, items, by_id, present, conflict_questions
                )
            except Exception:
                logger.warning("槽位二次整理失败：照常返回结果。", exc_info=True)
            # --- 1. pending 换入 ---
            for q in conflict_questions:
                new_id = q.get("new_id")
                old_id = q.get("old_id")
                if not new_id or new_id in present:
                    continue
                row = self._conn.execute(
                    "SELECT text, metadata_json FROM memories WHERE id=?"
                    " AND voided=0 AND deleted=0 AND superseded=0",
                    (new_id,),
                ).fetchone()
                if row is None:
                    continue
                new_text = row[0] or ""
                try:
                    new_meta = json.loads(row[1] or "{}")
                except Exception:
                    new_meta = {}
                new_bigrams = self._content_bigrams(new_text)
                linked = old_id in present or any(
                    self._same_conflict_topic(new_text, r["text"])
                    or bool(new_bigrams & self._content_bigrams(r["text"]))
                    for r in items
                )
                if not linked:
                    continue
                new_item = {
                    "id": new_id, "text": new_text, "score": 0.0,
                    "source": "pending_conflict",
                    "metadata": _normalize_metadata(new_meta),
                    "confirmed": bool(new_meta.get("user_confirmed")),
                    "decay_weight": 1.0, "is_history": False,
                }
                for idx in range(len(results) - 1, -1, -1):
                    cand = results[idx]
                    if cand.get("is_history"):
                        continue
                    if (cand.get("metadata") or {}).get("is_rule") is True:
                        continue
                    del results[idx]
                    break
                results.append(new_item)
                items.append(new_item)
                by_id[new_id] = new_item
                present.add(new_id)
            # --- 2. 成组 ---
            groups = []
            grouped_ids = set()

            def _add_to_group(group, item):
                if item.get("id") in grouped_ids:
                    return
                grouped_ids.add(item.get("id"))
                group.append(item)

            for q in conflict_questions:
                pair = [
                    by_id[q.get("old_id")], by_id[q.get("new_id")],
                ] if q.get("old_id") in by_id and q.get("new_id") in by_id else []
                if len(pair) < 2:
                    continue
                group = []
                for item in pair:
                    _add_to_group(group, item)
                new_bigrams = self._content_bigrams(pair[1]["text"])
                for r in items:
                    if r.get("id") in grouped_ids:
                        continue
                    if new_bigrams & self._content_bigrams(r["text"]):
                        _add_to_group(group, r)
                if len(group) >= 2:
                    groups.append(group)
            for r in items:
                if r.get("id") in grouped_ids:
                    continue
                placed = False
                for group in groups:
                    if any(
                        self._same_conflict_topic(g["text"], r["text"])
                        for g in group
                    ):
                        _add_to_group(group, r)
                        placed = True
                        break
                if not placed:
                    group = [r]
                    grouped_ids.add(r.get("id"))
                    groups.append(group)

            def _rank(g):
                meta = g.get("metadata") or {}
                return (
                    1 if (g.get("confirmed")
                          or meta.get("user_confirmed")) else 0,
                    meta.get("write_seq") or 0,
                    meta.get("created_at") or "",
                )

            for group in groups:
                if len(group) < 2:
                    continue
                if len({_dedupe_key(g["text"]) for g in group}) < 2:
                    continue
                current = max(group, key=_rank)
                for g in group:
                    meta = g.setdefault("metadata", {})
                    if g is current:
                        meta["conflict_current"] = True
                    else:
                        meta["conflict_stale"] = True
                        meta["conflict_with"] = current["text"]
            return results
        except Exception:
            logger.warning("检索冲突标注失败：照常返回结果。", exc_info=True)
            return results

    def _slot_recompose(self, results, items, by_id, present,
                        conflict_questions):
        """槽位二次整理（2026-11-24）：见 _annotate_search_conflicts
        第 0 步。只就地改 results/items/by_id/present，不改返回形状。

        成组只认硬信号（和 _same_conflict_topic 一个标准）：
        - 单值槽（slot: 住处/名字…）：同槽不同值天然互斥，全槽一组；
        - 偏好话题（pref: 喜欢吃…）：同话题不同食物可以并存，不算
          冲突——只认未答冲突问题亲口认定的对子成组。否定式链接
          不用来成组：“纠正了鸡杂面”那种备注行靠否定词和半个
          话题的二字组就能和并存偏好攀上关系，拿它换入/重排会
          把冒菜这种并存偏好误当现行条（全套回归实测：Q3 排序
          被打乱、webfix 口味场景换错行）。
        组内现行 = 已确认优先、同级按 write_seq/created_at 最新
        （和第 2 步标注同一个 _rank 口径）。现行不在结果里时，
        换掉组内最旧的那条占它的位置（规则行不换）；现行在结果
        里但排在旧行后面的，挪到本组最靠前的旧行前面。
        """
        def _rank_of(item):
            meta = item.get("metadata") or {}
            return (
                1 if (item.get("confirmed")
                      or meta.get("user_confirmed")) else 0,
                meta.get("write_seq") or 0,
                meta.get("created_at") or "",
            )

        linked_pairs = set()
        for q in conflict_questions or []:
            _o, _n = q.get("old_id"), q.get("new_id")
            if _o and _n and _o != _n:
                linked_pairs.add(frozenset((_o, _n)))

        topics = {}
        for item in items:
            topic = _correction_topic_slot(item.get("text") or "")
            if topic:
                topics.setdefault(topic, []).append(item)
        if not topics:
            return
        live_by_topic = {}
        for row in self._conn.execute(
            "SELECT id, text, metadata_json FROM memories"
            " WHERE voided=0 AND deleted=0 AND superseded=0"
        ).fetchall():
            topic = _correction_topic_slot(row[1] or "")
            if not topic or topic not in topics:
                continue
            try:
                _meta = json.loads(row[2] or "{}")
            except Exception:
                _meta = {}
            live_by_topic.setdefault(topic, []).append({
                "id": row[0], "text": row[1] or "", "metadata": _meta,
                "confirmed": bool(_meta.get("user_confirmed")),
            })

        for topic, group_items in topics.items():
            live_rows = live_by_topic.get(topic) or []
            if len(live_rows) < 2:
                continue
            clusters = []
            if topic.startswith("slot:"):
                clusters = [live_rows]
            else:
                # 偏好话题只认待问问题认定的对子（见 docstring）。
                by_id_live = {c["id"]: c for c in live_rows}
                grouped_live = set()
                for pair in linked_pairs:
                    members = [
                        by_id_live[i] for i in pair if i in by_id_live
                    ]
                    if len(members) < 2:
                        continue
                    if any(m["id"] in grouped_live for m in members):
                        continue
                    grouped_live.update(m["id"] for m in members)
                    clusters.append(members)
            for cluster in clusters:
                member_ids = {c["id"] for c in cluster}
                in_results = [
                    it for it in group_items if it.get("id") in member_ids
                ]
                if not in_results:
                    continue
                current = max(cluster, key=_rank_of)
                if current["id"] in present:
                    # 现行在结果里：不许排在任何同组旧行后面。
                    cur_item = by_id.get(current["id"])
                    if cur_item is None or cur_item not in results:
                        continue
                    stale_idxs = [
                        results.index(it) for it in in_results
                        if it.get("id") != current["id"] and it in results
                    ]
                    if stale_idxs and results.index(cur_item) > min(stale_idxs):
                        results.pop(results.index(cur_item))
                        results.insert(min(stale_idxs), cur_item)
                    continue
                # 现行不在结果里：换掉组内最旧的一条（规则行不换；
                # 单值槽里和现行同值的说法不算旧行，不换它）。
                _cur_value = _infer_fact_slot(current["text"])[1]
                victims = []
                for it in in_results:
                    if (it.get("metadata") or {}).get("is_rule") is True:
                        continue
                    if topic.startswith("slot:") and _cur_value is not None:
                        _v_value = _infer_fact_slot(it.get("text") or "")[1]
                        if (
                            _v_value is not None
                            and _normalize_memory_text(_v_value)
                            == _normalize_memory_text(_cur_value)
                        ):
                            continue
                    victims.append(it)
                if not victims:
                    continue
                victim = min(victims, key=_rank_of)
                new_item = {
                    "id": current["id"], "text": current["text"],
                    "score": victim.get("score") or 0.0,
                    "source": "slot_current",
                    "metadata": _normalize_metadata(current["metadata"]),
                    "confirmed": bool(
                        current["metadata"].get("user_confirmed")
                    ),
                    "decay_weight": 1.0, "is_history": False,
                }
                results[results.index(victim)] = new_item
                items[items.index(victim)] = new_item
                group_items[group_items.index(victim)] = new_item
                by_id.pop(victim.get("id"), None)
                by_id[new_item["id"]] = new_item
                present.discard(victim.get("id"))
                present.add(new_item["id"])

    def search_history(self, query, top_k=5):
        return self.search(query, top_k=top_k, include_history=True)

    def note_context(self, memories):
        """记下本轮检索到的记忆 id，并写回 last_used_at（刚被用上）：
        沉底的（dormant）和待删的（decay_delete_due）都复活回 verbatim，
        账本记 REVIVE（2026-11-22 decay 接线）。
        last_used_at 同时记账本（2026-11-24 3a）：每轮一笔 TOUCH
        事件（带本轮全部 id、同一个时间戳），对账重建时按 TOUCH
        把 last_used_at 带回来——P1-6 时期“只写 SQLite、重建退回
        旧值”的已知问题就此修掉；一轮一笔，不逐条记，不吵。
        """
        with self._lock:
            self._last_context_ids = [
                (m or {}).get("id") for m in (memories or []) if (m or {}).get("id")
            ]
            used_at = _now_iso()
            for mid in self._last_context_ids:
                try:
                    row = self._conn.execute(
                        "SELECT text, metadata_json FROM memories WHERE id=? AND voided=0"
                        " AND deleted=0 AND superseded=0",
                        (mid,),
                    ).fetchone()
                    if row is None:
                        continue
                    meta = json.loads(row[1] or "{}")
                    was_dormant = meta.get("decay_state") in (
                        _decay.DECAY_STATE_DORMANT, _decay.DECAY_STATE_DELETE_DUE,
                    )
                    if was_dormant:
                        # R-4（2026-11-24）：复活账本先行——REVIVE
                        # 写不进去就整条不动（last_used_at 也不写），
                        # 下次用上时再试。以前先改 SQLite、记账失败
                        # 只警告，SQLite 已是原话态、账本还认为它
                        # 沉底，两边静默分叉。
                        try:
                            _ledger.append_event(
                                self.ledger_path,
                                {"type": _ledger.LEDGER_EVENT_REVIVE,
                                 "memory_id": mid, "key": mid,
                                 "text": row[0] or ""},
                                lock=self._ledger_lock,
                            )
                        except Exception:
                            logger.warning("复活记账失败：本条先不复活。", exc_info=True)
                            continue
                        meta["decay_state"] = _decay.DECAY_STATE_VERBATIM
                    meta["last_used_at"] = used_at
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), mid),
                    )
                    self._conn.commit()
                except Exception:
                    logger.warning("note_context 写回失败。", exc_info=True)
            if self._last_context_ids:
                try:
                    _ledger.append_event(
                        self.ledger_path,
                        {"type": _ledger.LEDGER_EVENT_TOUCH,
                         "memory_id": self._last_context_ids[0],
                         "key": self._last_context_ids[0],
                         "memory_ids": list(self._last_context_ids),
                         "ts": used_at},
                        lock=self._ledger_lock,
                    )
                except Exception:
                    logger.warning("使用时间记账失败。", exc_info=True)

    def sweep_decay(self, now=None):
        """衰减扫一遍（2026-11-22 decay 接线）：按 decay.py 的规矩，
        闲置够久、权重够低的标沉底（dormant），到期的记 delete_due。
        沉底只是状态（decay_state=dormant + 账本 DORMANT 事件），
        记忆还在、可搜；delete_due 只是到期标记，不删——真删永远只
        走 BOOS 点头的那条路（delete_memory / 忘记这个）。
        不自动跑：由调用方（定时任务/自检）定期调一次。
        标待删也记账本（2026-11-24 3d）：写一笔 DORMANT 事件带
        delete_due=True（账本状态层早就认这个标记）——以前待删只
        写 SQLite，账本里没有，重建就丢了待删状态。
        返回 {"dormant": n, "delete_due": n}。
        """
        now = now or _now_epoch()
        dormant = 0
        delete_due = 0
        with self._lock:
            for row in self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall():
                meta = json.loads(row[2] or "{}")
                state = _decay.decay_state(
                    meta.get("decay_kind") or _decay.DECAY_KIND_EPISODIC,
                    meta.get("last_used_at") or meta.get("created_at"),
                    now=now,
                    is_core=bool(_provenance._is_core_memory(meta)),
                    is_curated=bool(meta.get("curated")),
                )
                # R-4（2026-11-24）：三个分支统一账本先行——账本写
                # 不进去就整条不动、不计数，下次扫再试。以前是先改
                # SQLite、记账失败只警告，账本与 SQLite 静默分叉。
                if state == _decay.DECAY_STATE_DELETE_DUE and meta.get(
                    "decay_state"
                ) != _decay.DECAY_STATE_DELETE_DUE:
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {"type": _ledger.LEDGER_EVENT_DORMANT,
                             "memory_id": row[0], "key": row[0],
                             "text": row[1] or "", "delete_due": True},
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning("待删记账失败：本条先不标待删。", exc_info=True)
                        continue
                    meta["decay_state"] = state
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), row[0]),
                    )
                    delete_due += 1
                elif state == _decay.DECAY_STATE_DORMANT and meta.get(
                    "decay_state"
                ) != _decay.DECAY_STATE_DORMANT:
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {"type": _ledger.LEDGER_EVENT_DORMANT,
                             "memory_id": row[0], "key": row[0],
                             "text": row[1] or ""},
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning("沉底记账失败：本条先不沉底。", exc_info=True)
                        continue
                    meta["decay_state"] = state
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), row[0]),
                    )
                    dormant += 1
                elif state == _decay.DECAY_STATE_VERBATIM and meta.get(
                    "decay_state"
                ) in (
                    _decay.DECAY_STATE_DORMANT,
                    _decay.DECAY_STATE_DELETE_DUE,
                ):
                    # 复活过的记忆：算出来已是原话态，把库里还顶着的旧
                    # 牌子改回 verbatim，别让它一直挂着 dormant / delete_due。
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {"type": _ledger.LEDGER_EVENT_REVIVE,
                             "memory_id": row[0], "key": row[0],
                             "text": row[1] or ""},
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning("复活记账失败：本条先不改状态。", exc_info=True)
                        continue
                    meta["decay_state"] = state
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), row[0]),
                    )
            self._conn.commit()
        return {"dormant": dormant, "delete_due": delete_due}

    def organizing_router(self):
        """整理记忆的大模型路由（懒建，和 Mem0Store.organizing_router
        同款）：主力 deepseek-v4-pro，备选 flash，本地降级；每月
        3 美元上限超了只提醒、不断服务（见 memory/model_router.py）。
        状态文件在账本同目录。建不出来（比如没配 DeepSeek Key）时
        返回 None——调用方走本地降级，不抛异常。"""
        router = getattr(self, "_organizing_router", None)
        if router is not None:
            return router
        try:
            from memory import model_router as _model_router

            router = _model_router.OrganizingModelRouter(
                state_path=str(
                    self.ledger_path.parent / "organizing_router_state.json"
                ),
                budget_tracker=self.budget_tracker,
            )
        except Exception:
            logger.warning("整理记忆路由建不起来：本次走本地降级。", exc_info=True)
            return None
        self._organizing_router = router
        return router

    def run_self_check(self, now=None, sample_size=20,
                       include_model_judge=True):
        """跑一次定期自检（SQLite 版，2026-11-24 3b；第 9 项抽查
        20 条、自检节奏改每周见缺点清单第 4 条）：先扫一遍衰减
        （sweep_decay：该沉底的沉底、该标待删的标待删，都记账本），
        再跑十项自检（memory/self_check.py，只查、只报告，不改库）。
        返回自检报告 dict，多带一个 "decay_sweep" 键记这次衰减扫的
        结果。不在服务启动时自动跑（启动会变慢）：由维护调度按周
        调，和 Mem0Store.run_self_check 同名同节奏，调用方不用分
        后端。模型抽查（第 9/10 项）走整理路由的判官（R-6，
        2026-11-24，和 mem0 同款接法）；路由建不起来时 model_judge
        为 None、判官判不出来时 verdict 为 None，两项都如实
        skipped——判不出来就不猜。"""
        sweep_now = None
        if isinstance(now, bool):
            sweep_now = None
        elif isinstance(now, (int, float)):
            sweep_now = float(now)
        elif isinstance(now, datetime):
            sweep_now = now.timestamp()
        sweep = self.sweep_decay(now=sweep_now)
        from memory import self_check as _self_check_module
        from memory import store_self_check as _store_self_check

        class _SQLiteAdapter(_store_self_check._StoreSelfCheckAdapter):
            """SQLite 版适配器：只有索引口径和 mem0 不同——mem0 对的
            是 history.db vs Chroma，SQLite 索引就是本库，对的是账本
            当前态（current + dormant 的 key）vs SQLite 活行 id；
            两边对不上就是索引丢了/坏了。其余方法照用基类的。"""

            def index_counts(self):
                live_ids = {
                    row[0]
                    for row in self._store._conn.execute(
                        "SELECT id FROM memories WHERE voided=0"
                        " AND deleted=0 AND superseded=0"
                    ).fetchall()
                }
                key_states = (
                    (self._store.ledger_state() or {}).get("key_states")
                    or {}
                )
                ledger_ids = {
                    k
                    for k, v in key_states.items()
                    if v in (
                        _ledger.LEDGER_STATE_CURRENT,
                        _ledger.LEDGER_STATE_DORMANT,
                    )
                }
                return {
                    "sqlite": len(live_ids),
                    "chroma": len(ledger_ids),
                    "mismatched_ids": sorted(live_ids ^ ledger_ids),
                }

        # R-6（2026-11-24）：把整理路由的判官接进来（make_model_judge
        # 和 mem0 同款），第 9/10 项在 SQLite 后端也真跑；路由不可用
        # 时 model_judge=None，两项 skipped，和 mem0 路由不可用时一样。
        # 错过补跑（2026-11-24）：include_model_judge=False 是离线
        # 模式——判官项（第 9/10 项）不接路由、直接 skipped，报告
        # 里标 model_judge_deferred，维护调度据此把判官部分挂成
        # 待跑、等网络恢复补跑；本地项（衰减扫 + 其余八项）照常跑。
        router = self.organizing_router() if include_model_judge else None
        model_judge = None
        if router is not None:
            model_judge = _store_self_check.make_model_judge(router)
        report = _self_check_module.run_self_check(
            _SQLiteAdapter(self),
            now=now,
            model_judge=model_judge,
            sample_size=sample_size,
            router_status=router.status() if router is not None else {},
        )
        report["decay_sweep"] = sweep
        if not include_model_judge:
            report["model_judge_deferred"] = True
        return report

    # ---------- 改 / 删（今天只实现删除；纠错后面再做） ----------

    def delete_memory(self, memory_id, approve=False, instruction_source=None):
        """删一条记忆：要 BOOS 亲口确认（approve=True）才真删。

        账本记 delete，SQLite 标 deleted；原文留在账本里，可恢复。
        SQLite 里没有但账本里有该 key（账本兜底捞出的条目）时，同样
        记账删除，防止重启对账或兜底检索再次捞出。
        """
        # 授权硬判（实测问题一）：改记忆类动作只认 BOOS 本人
        # 通道的指令。调用方显式标了非本人来源（转述/第三方）时
        # 直接拒绝，并攒一条待问留痕——转述指令不许执行。没标
        # 来源的老调用（BOOS 自己的界面、内部路径）行为不变，
        # 仍走下面的 approve 关。
        if not _auth_guard.is_owner_instruction(instruction_source):
            self._queue_change_attempt_question(
                "delete", memory_id, instruction_source
            )
            return {"deleted": False, "reason": "not_owner_instruction"}
        if not approve:
            return {"deleted": False, "reason": "need_approve"}
        if not memory_id:
            return {"deleted": False, "reason": "not_found"}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, metadata_json, deleted FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                state = self.ledger_state()
                key_states = (state or {}).get("key_states") or {}
                rec = next(
                    (
                        r for r in (state or {}).get("adds") or []
                        if r.get("key") == memory_id
                    ),
                    None,
                )
                if rec is None or key_states.get(memory_id) == _ledger.LEDGER_STATE_DELETED:
                    return {"deleted": False, "reason": "not_found"}
                try:
                    _ledger.append_event(
                        self.ledger_path,
                        {
                            "type": "delete",
                            "memory_id": memory_id,
                            "key": memory_id,
                            "text": rec.get("text") or "",
                            "meta": rec.get("meta") or {},
                        },
                        lock=self._ledger_lock,
                    )
                except Exception:
                    logger.warning("删除记账失败。", exc_info=True)
                    return {"deleted": False, "reason": "ledger_failed"}
                return {"deleted": True, "memory_id": memory_id}
            if row[3]:
                return {"deleted": False, "reason": "not_found"}
            meta = json.loads(row[2] or "{}")
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": "delete", "memory_id": row[0],
                     "key": row[0], "text": row[1], "meta": meta},
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("删除记账失败。", exc_info=True)
                return {"deleted": False, "reason": "ledger_failed"}
            self._conn.execute(
                "UPDATE memories SET deleted=1 WHERE id=?", (row[0],)
            )
            self._conn.commit()
            return {"deleted": True, "memory_id": row[0]}

    def revive_memory(self, memory_id, instruction_source=None):
        """恢复一条被取代的记忆（2026-11-22，广州行被杭州错取代后恢复用）。

        只恢复 superseded=1 的行：SQLite 取消取代标记，账本记一笔 REVIVE
        事件（账本是真相源；只改 SQLite 会被对账逻辑弄乱——对账按账本
        的 key_states 认，REVIVE 事件把该 key 拨回 current）。
        已经是活记忆时报 unchanged；行不存在、或已删除/已作废时报
        not_found（删了的恢复走别的路，不在这里顺手复活）。
        账本写失败时把 SQLite 的标记撤回来再报 failed，两边保持一致。
        """
        # 授权硬判（实测问题一）：同 delete_memory——显式标了非
        # 本人来源的恢复指令直接拒绝并攒待问留痕。
        if not _auth_guard.is_owner_instruction(instruction_source):
            self._queue_change_attempt_question(
                "revive", memory_id, instruction_source
            )
            return {"status": "rejected", "reason": "not_owner_instruction",
                    "memory_id": memory_id}
        if not memory_id:
            return {"status": "not_found", "memory_id": memory_id}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, superseded, voided, deleted FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return {"status": "not_found", "memory_id": memory_id}
            if row[3] or row[4]:
                return {"status": "not_found", "memory_id": memory_id,
                        "reason": "已作废或已删除的记忆不能这样恢复"}
            if not row[2]:
                return {"status": "unchanged", "memory_id": row[0],
                        "text": row[1] or ""}
            self._conn.execute(
                "UPDATE memories SET superseded=0 WHERE id=?", (row[0],)
            )
            self._conn.commit()
            try:
                event = _ledger.append_event(
                    self.ledger_path,
                    {"type": _ledger.LEDGER_EVENT_REVIVE,
                     "memory_id": row[0], "key": row[0],
                     "text": row[1] or ""},
                    lock=self._ledger_lock,
                )
            except Exception:
                self._conn.execute(
                    "UPDATE memories SET superseded=1 WHERE id=?", (row[0],)
                )
                self._conn.commit()
                logger.warning("恢复记账失败：取代标记已撤回。", exc_info=True)
                return {"status": "failed", "memory_id": row[0],
                        "reason": "账本写失败"}
            return {"status": "revived", "memory_id": row[0],
                    "text": row[1] or "",
                    "seq": (event or {}).get("seq")}

    def confirm_memory(self, memory_id):
        """BOOS 逐条确认一条记忆（确认后按 provenance 口径成为核心成员）。

        容量门（四件套第 1 块）：确认会让一条记忆新占一个核心
        座位、而座位已满时，不许确认——记忆保持未确认，攒一条
        「核心已满」待问（kind=core_capacity），等 BOOS 拍板或
        腾位后再说。已经是成员的再确认不新增占座，不受拦。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT text, metadata_json, superseded, deleted, voided"
                " FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return {"confirmed": False, "reason": "not_found"}
            meta = json.loads(row[1] or "{}")
            new_meta = dict(meta)
            new_meta["user_confirmed"] = True
            if not (row[2] or row[3] or row[4]):
                before = _core_cap.classify(meta)
                after = _core_cap.classify(new_meta)
                if after["occupies_seat"] and not before["occupies_seat"]:
                    snap = self.core_seat_snapshot()
                    if snap["seats_used"] >= snap["limit"]:
                        question = self._queue_core_capacity_question(
                            memory_id, row[0] or "", snap
                        )
                        return {
                            "confirmed": False,
                            "reason": "core_full",
                            "memory_id": memory_id,
                            "question_id": (question or {}).get("id"),
                            "seats_used": snap["seats_used"],
                            "limit": snap["limit"],
                        }
            self._conn.execute(
                "UPDATE memories SET user_confirmed=1, metadata_json=? WHERE id=?",
                (json.dumps(new_meta, ensure_ascii=False), memory_id),
            )
            self._conn.commit()
            return {"confirmed": True, "memory_id": memory_id}

    # ---------- 核心上限（四件套第 1 块，2026-11-24） ----------

    def _queue_core_capacity_question(self, memory_id, text, snap):
        """满座拦下时攒一条待问：同条记忆已有未答的容量问题时
        PendingQuestions.add 自带去重、返回已有的那条。"""
        return self.pending_questions.add(
            _pq.QUESTION_KIND_CORE_CAPACITY,
            "核心记忆座位已满（%d/%d），这条先没进核心：%s"
            % (snap["seats_used"], snap["limit"], text),
            _pq.CORE_CAPACITY_QUESTION_OPTIONS,
            "核心上限已满，按规矩不许顶掉旧记忆腾座；这条先留作"
            "候选，等整理员腾位或你拍板。",
            memory_id=memory_id,
            extra={
                "seats_used": snap["seats_used"],
                "limit": snap["limit"],
            },
        )

    def core_seat_snapshot(self):
        """核心座位水位快照（只读，结论页/体检用）：持锁一次读全
        活行，同一快照出数——总数、分抽屉、钉死/临时钉死、未入
        格、水位都在里面，分块与总量对账见 count_seats 的
        reconciled 自证。"""
        with self._lock:
            metas = []
            for (mjson,) in self._conn.execute(
                "SELECT metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall():
                try:
                    metas.append(json.loads(mjson or "{}"))
                except Exception:
                    metas.append({})
            snap = _core_cap.count_seats(metas, limit=self.core_seat_limit)
            snap["generated_at"] = _now_iso()
            return snap

    def recount_core_seats_from_ledger(self):
        """按账本事件全量重算座位（对账用）：应与同时刻的
        core_seat_snapshot 逐项相等，不等即缺陷。"""
        events, _info = _ledger.read_events(self.ledger_path)
        state = self.ledger_state()
        snap = _core_cap.count_seats_from_ledger(
            events,
            (state or {}).get("key_states") or {},
            limit=self.core_seat_limit,
        )
        if self._controlled_ledger_doubtful(_info):
            # 账本残缺（补修 H3 复查 F2）：重算结果是残缺
            # 折叠，不许报 reconciled——如实标存疑。
            snap["ledger_doubtful"] = True
            snap["reconciled"] = False
        snap["generated_at"] = _now_iso()
        return snap

    def set_protection_tier(self, memory_id, tier, _op_id=None,
                            _idem_key=None, _source=None):
        """改一条活记忆的保护档位：tier 是 "pinned"（钉死）/
        "temp_pinned"（临时钉死）/ None（摘档回普通）。

        摘档会让成员行新占座：满座时拒绝并攒待问，不许硬摘。
        落库走 _update_core_fields（账本先行）。"""
        if tier is not None and tier not in _core_cap.PROTECTION_TIERS:
            return {"ok": False, "reason": "bad_tier", "memory_id": memory_id}
        return self._update_core_fields(
            memory_id, {_core_cap.PROTECTION_TIER_FIELD: tier},
            _op_id=_op_id, _idem_key=_idem_key, _source=_source,
        )

    def set_core_drawer(self, memory_id, drawer, _op_id=None, _idem_key=None):
        """改一条活记忆的分类抽屉：drawer 是五类键之一 / None
        （摘回未入格）。只影响分块记账、不改变占座总数，不过
        容量门；落库走 _update_core_fields（账本先行）。"""
        if drawer is not None and (
            not isinstance(drawer, str)
            or drawer not in _core_cap.CORE_DRAWERS
        ):
            return {"ok": False, "reason": "bad_drawer", "memory_id": memory_id}
        return self._update_core_fields(
            memory_id, {_core_cap.DRAWER_FIELD: drawer},
            _op_id=_op_id, _idem_key=_idem_key,
        )

    def _update_core_fields(self, memory_id, changes, _op_id=None,
                            _idem_key=None, _source=None):
        """核心上限字段（保护档/抽屉）的统一改法：持锁内复核
        占座变化、过容量门、账本先行（PROMOTE 带改后的完整
        meta，和语气打标同款）、再改 SQLite；账本写失败就
        不改 SQLite，两边不许分叉。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT text, metadata_json, superseded, deleted,"
                " voided, user_confirmed FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return {"ok": False, "reason": "not_found", "memory_id": memory_id}
            if row[2] or row[3] or row[4]:
                return {"ok": False, "reason": "not_live", "memory_id": memory_id}
            meta = json.loads(row[1] or "{}")
            before = _core_cap.classify(meta)
            new_meta = dict(meta)
            for field, value in changes.items():
                new_meta[field] = value
            after = _core_cap.classify(new_meta)
            # 保护迁移证据（立项 H5）：档位变化前后值与变更前
            # 版本戳先算好，账本落笔后在函数尾组装证据。
            _tier_field = _core_cap.PROTECTION_TIER_FIELD
            tier_before = meta.get(_tier_field)
            tier_after = new_meta.get(_tier_field)
            version_before = self._controlled_version_stamp(
                row[0] or "", meta, row[5]
            )
            if after["occupies_seat"] and not before["occupies_seat"]:
                snap = self.core_seat_snapshot()
                if snap["seats_used"] >= snap["limit"]:
                    question = self._queue_core_capacity_question(
                        memory_id, row[0] or "", snap
                    )
                    return {
                        "ok": False,
                        "reason": "core_full",
                        "memory_id": memory_id,
                        "question_id": (question or {}).get("id"),
                        "seats_used": snap["seats_used"],
                        "limit": snap["limit"],
                    }
            event = {
                "type": _ledger.LEDGER_EVENT_PROMOTE,
                "memory_id": memory_id,
                "key": memory_id,
                "text": row[0] or "",
                "meta": dict(new_meta),
            }
            if _op_id:
                # 受控入口执行时带贯通操作 ID（H1 补修）：账本
                # 折叠只读命名字段，这两个字段不影响折叠/对账
                # 口径；提交/恢复协议凭 op_id 在账本里定位这笔
                # 变更，idem_key 与回执/审计三方对照。
                event["op_id"] = _op_id
                event["idem_key"] = _idem_key
            ledger_record = None
            try:
                ledger_record = _ledger.append_event(
                    self.ledger_path, event, lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("核心字段变更记账失败：本次不改。", exc_info=True)
                return {
                    "ok": False,
                    "reason": "ledger_failed",
                    "memory_id": memory_id,
                }
            self._conn.execute(
                "UPDATE memories SET metadata_json=? WHERE id=?",
                (json.dumps(new_meta, ensure_ascii=False), memory_id),
            )
            self._conn.commit()
            result = {"ok": True, "memory_id": memory_id}
            result.update(changes)
            if tier_after != tier_before:
                # 保护档真变了：产一份迁移完成证据（立项 H5）。
                # 证据追加失败不回滚——账本与投影已经一致落地，
                # 回滚投影只会与账本分叉；如实标 pending 待补
                # 证，核对函数（verify_protection_migration）
                # 与 pending_notices 都会把缺证列出来。
                evidence = {
                    "evidence_id": "pme-%s" % (
                        (ledger_record or {}).get("seq"),
                    ),
                    "memory_id": memory_id,
                    "tier_before": tier_before,
                    "tier_after": tier_after,
                    "drawer_before": meta.get(_core_cap.DRAWER_FIELD),
                    "drawer_after": new_meta.get(_core_cap.DRAWER_FIELD),
                    "source": _source or "host_direct",
                    "ts": _now_iso(),
                    "ledger_seq": (ledger_record or {}).get("seq"),
                    "ledger_ts": (ledger_record or {}).get("ts"),
                    "op_id": _op_id,
                    "idem_key": _idem_key,
                    "version_before": version_before,
                    "version_after": self._controlled_version_stamp(
                        row[0] or "", new_meta, row[5]
                    ),
                    "text_hash": hashlib.sha256(
                        (row[0] or "").encode("utf-8")
                    ).hexdigest(),
                }
                recorded = self._append_protection_evidence(evidence)
                result["protection_evidence"] = {
                    "evidence_id": evidence["evidence_id"],
                    "recorded": recorded,
                    "pending": not recorded,
                }
                if not recorded:
                    self._protection_evidence_pending.append(
                        dict(evidence)
                    )
            return result

    def _answer_core_capacity_question(self, question, choice):
        """BOOS 回答核心容量问题（满座候选）：
          先留作候选：只消耗问题，记忆保持原样（还没进核心）；
          不要了：先删成功、再消耗问题（和重要性问题「不存」
          同款顺序——删不掉问题留着、如实报，不许先消耗后删）。"""
        if choice not in _pq.CORE_CAPACITY_QUESTION_OPTIONS:
            return {
                "ok": False,
                "reason": "选项不对，只能选：先留作候选 / 不要了。",
            }
        question_id = question.get("id")
        memory_id = question.get("memory_id")
        if choice == "先留作候选":
            self.pending_questions.answer(question_id, choice)
            return {
                "ok": True,
                "applied": "held",
                "question_id": question_id,
                "memory_id": memory_id,
            }
        if not memory_id:
            return {"ok": False, "reason": "这个问题没有对应的记忆。"}
        deleted = self.delete_memory(memory_id, approve=True)
        deleted_ok = (
            deleted.get("deleted") is True
            if isinstance(deleted, dict)
            else bool(deleted)
        )
        if not deleted_ok:
            return {
                "ok": False,
                "reason": "没删掉（%s），问题先留着，可以再答。"
                % (
                    (deleted.get("reason") or "未知原因")
                    if isinstance(deleted, dict)
                    else "未知原因"
                ),
                "question_id": question_id,
                "memory_id": memory_id,
                "deleted": False,
            }
        self.pending_questions.answer(question_id, choice)
        return {
            "ok": True,
            "applied": "dropped",
            "question_id": question_id,
            "memory_id": memory_id,
            "deleted": True,
        }
    # ---------- 统一受控入口（立项 H1，2026-11-25；补修轮同日） ----------
    #
    # 整理员（及任何程序化调用方）的状态变更只许走这道正门：
    # submit_controlled_proposal 接结构化提案，门内顺序固定——
    # 纯形状/参数校验（补修轮前移：影子路径也先过参数关）→
    # 幂等查重（同键同指纹回 duplicate、同键异指纹拒
    # idempotency_conflict）→ 载行 → 验来源 → 验保护 → 验版本
    # （带了戳必须对上）→ 执行开关 → 健康冻结闸口 → 会话认证
    # （H2）→ 动作前置检查 → 版本必填（仅执行路径）→ 操作意图
    # 落盘 → 走既有 store 路径执行（状态变更的账本事件带贯通
    # op_id）→ 事件关联核验 → 不变量健康闸口 → 回执登记。
    # 不新开一套写库逻辑。删除永远不在动作集里：删除只走
    # delete_memory(approve=True) 的 BOOS 批准路，整理员永无
    # 删除权限（契约三）。
    #
    # 占键语义（补修轮定版）：纯形状/参数校验不过的拒绝不占键
    # （提案本身没成形，改对了同键可再提）；过了校验的每种结局
    # （执行/影子/各道门禁拒绝）都登记占键——同键同指纹重提回
    # 当时的回执副本（duplicate），同键异指纹一律拒
    # idempotency_conflict；被拒或影子之后想换个提法重试，
    # 用新键。
    #
    # 提交/恢复协议（补修轮）：执行前先把操作意图（intent：
    # op_id、提案指纹、目标、绑的版本）落盘为 pending；执行的
    # 账本事件带 op_id＋idem_key，审计日志与回执同带 op_id，
    # 三方凭 op_id＋幂等键对照。提交成功标 committed。启动时
    # （及 controlled_entry_reconcile 手动调时）对账 pending
    # 意图：账本里有这笔 op_id → 补标 committed；没有 → 标
    # aborted（那次执行没落账，不许假装成功过）。回执登记文件
    # 丢了/坏了时，先按意图重建 duplicate 回执，不重复执行；
    # 登记/意图/审计任一持久化失败或损坏 → 入口冻结危险写
    # （_controlled_frozen_reason 非空）：执行类动作一律拒
    # entry_frozen，只许拒绝与影子，已落状态如实标已提交，
    # 不假报 refused、更不许只回滚 SQLite 造成账本分叉。
    #
    # 写路清单（补修轮产出，全仓状态变更写路逐条点名）：
    # 【已收口】submit_controlled_proposal —— 整理员唯一执行面。
    # 【宿主自用，未收口，如实记录】add_memory_text / save_turn
    # （对话写路：新增/晋升/取代）、correct_memory（BOOS 纠正
    # 路）、answer_question（BOOS 答待问，可引发删/确认/恢复）、
    # delete_memory(approve=True)（BOOS 批准删除路）、
    # revive_memory / confirm_memory（BOOS 恢复/确认路）、
    # set_protection_tier / set_core_drawer / _update_core_fields
    # （宿主直改核心字段，无凭据概念——整理员的同名动作必须
    # 走正门，不许直调这三个）、sweep_decay（宿主维护扫描，
    # 直写 DORMANT 事件）、void_joke_memory（宿主作废玩笑路）、
    # add_synonym_edge / void_graph_edge（图谱边写路）、
    # import_from_mem0（一次性迁移）、reconcile_ledger_index /
    # repair_projection_from_ledger / startup_recovery（恢复路：
    # 只按账本修投影，不新造状态）。
    # 【内部，非入口】_insert_memory_row / _supersede_row /
    # _set_memory_slot / _ledger_append_add 等下划线方法，只被
    # 上面各路调用，不单独对外。收口是渐进的：本轮只立正门并
    # 点名其余写路，不封宿主自用路（封了宿主自己先断粮）。

    CONTROLLED_ACTIONS = (
        "demote",
        "set_protection_tier",
        "set_core_drawer",
        "mark_dormant",
    )

    # 这些持久化故障一出现就冻结危险写：登记/意图/审计是提交
    # 协议的地基，地基不稳时宁可只拒不写（见上方协议说明）。
    _CONTROLLED_FREEZE_KINDS = frozenset({
        "receipts_load_error",
        "intents_load_error",
        "receipts_save_failed",
        "intents_save_failed",
        "audit_write_failed",
        # 补修二轮（贵妃第十八份复查 A/D/E）：不变量失败与
        # 提交待核实是危险方向，对账定不了生死（读账本失败/
        # 坏行存疑）同样——全部与持久化故障共用同一面冻结旗。
        "invariant_failed",
        "commit_unverified",
        "reconcile_undetermined",
        # 立项 H6：启动恢复结论不干净（判据见 H6 节）同样冻结
        # 危险写，与持久化故障共用同一面冻结旗、同一道闸口。
        "recovery_unclean",
    })

    def _controlled_note_health(self, kind, detail):
        """记一笔受控入口健康痕迹（store 级标记＋pending_notices
        留痕，与 H6 的落盘/结论页映射在这里留接缝：H6 建成后
        从 _controlled_health_issues 与 _controlled_frozen_reason
        取数落盘，本轮不另建落盘通道）。冻结类故障同时置冻结
        标记，置过不覆盖（保留最早的原因）。"""
        entry = {"kind": kind, "detail": detail, "at": _now_iso()}
        self._controlled_health_issues.append(entry)
        if (
            kind in self._CONTROLLED_FREEZE_KINDS
            and self._controlled_frozen_reason is None
        ):
            self._controlled_frozen_reason = kind
        return entry

    def _load_controlled_entry_state(self):
        """载入受控入口的回执登记、提案指纹、操作意图与留痕
        序号（构造时调，永不抛），并对账未完的操作意图。

        登记/意图文件损坏时按待问文件先例：原文件改名保全、
        load_error 记在实例上、从空开始——不许静默吞掉；且
        损坏即冻结危险写（见 _CONTROLLED_FREEZE_KINDS），等
        对账/人工处置，不许在登记不可信时继续执行。
        """
        try:
            raw = self._controlled_receipts_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            receipts = data.get("receipts") if isinstance(data, dict) else None
            if not isinstance(receipts, dict):
                raise ValueError("receipts is not a dict")
            self._controlled_receipts = {
                str(k): v for k, v in receipts.items() if isinstance(v, dict)
            }
            fingerprints = (
                data.get("fingerprints") if isinstance(data, dict) else None
            )
            if isinstance(fingerprints, dict):
                self._controlled_fingerprints = {
                    str(k): str(v) for k, v in fingerprints.items()
                }
            revocations = (
                data.get("revocations") if isinstance(data, dict) else None
            )
            if isinstance(revocations, dict):
                self._controlled_revocations = {
                    str(k): v
                    for k, v in revocations.items()
                    if isinstance(v, dict)
                }
            rev_fingerprints = (
                data.get("revocation_fingerprints")
                if isinstance(data, dict) else None
            )
            if isinstance(rev_fingerprints, dict):
                self._controlled_revocation_fingerprints = {
                    str(k): str(v)
                    for k, v in rev_fingerprints.items()
                }
        except FileNotFoundError:
            self._controlled_receipts = {}
        except Exception as exc:
            self._controlled_receipts = {}
            self._controlled_receipts_load_error = "%s: %s" % (
                type(exc).__name__, exc,
            )
            self._controlled_note_health(
                "receipts_load_error",
                {"error": self._controlled_receipts_load_error},
            )
            try:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                os.replace(
                    self._controlled_receipts_path,
                    str(self._controlled_receipts_path) + ".corrupt-" + stamp,
                )
            except Exception:
                logger.warning("受控入口回执登记保全失败", exc_info=True)
        try:
            raw = self._controlled_intents_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            intents = data.get("intents") if isinstance(data, dict) else None
            if not isinstance(intents, dict):
                raise ValueError("intents is not a dict")
            self._controlled_intents = {
                str(k): v for k, v in intents.items() if isinstance(v, dict)
            }
        except FileNotFoundError:
            self._controlled_intents = {}
        except Exception as exc:
            self._controlled_intents = {}
            self._controlled_intents_load_error = "%s: %s" % (
                type(exc).__name__, exc,
            )
            self._controlled_note_health(
                "intents_load_error",
                {"error": self._controlled_intents_load_error},
            )
            try:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                os.replace(
                    self._controlled_intents_path,
                    str(self._controlled_intents_path) + ".corrupt-" + stamp,
                )
            except Exception:
                logger.warning("受控入口操作意图保全失败", exc_info=True)
        try:
            with open(self._controlled_log_path, "r", encoding="utf-8") as fh:
                self._controlled_trace_index = sum(1 for _ in fh)
        except FileNotFoundError:
            self._controlled_trace_index = 0
        except Exception:
            logger.warning("受控入口留痕计数失败", exc_info=True)
            self._controlled_trace_index = 0
        # 启动对账：上次进程死在「意图落盘后、标提交前」的
        # pending 意图，按账本里有没有这笔 op_id 定生死。
        try:
            self._controlled_reconcile_intents()
        except Exception:
            logger.warning("受控入口启动对账失败", exc_info=True)
        # 跨重启承接冻结（补修三轮，贵妃第二十二份复查
        # A；补修四轮，第二十三份复查 A 第二条断链）：
        # 已决意图里带着冻结级痕迹的，冻结事实不许随进
        # 程重启蒸发——在新进程里按 status（提交未定案）
        # 与 final_reason（不变量结论）两条来源重建同一
        # 面冻结旗。解冻沿用现有口径：干净恢复/复检只解
        # recovery_unclean 来源，本来源不被顺手解掉（与
        # 同进程语义一致：这类冻结在进程内本来也不自动
        # 解，重启更不许成为意外的解冻键）。
        try:
            self._controlled_carry_intent_freezes()
        except Exception:
            logger.warning("受控入口冻结承接失败", exc_info=True)

    # 哪些已决意图的最终原因（不变量结论）要在重启后
    # 重建冻结（与 _CONTROLLED_FREEZE_KINDS 里执行结论
    # 类的三种对齐）。注意 commit_unverified 的主承接
    # 来源是意图 status 本身（见
    # _controlled_carry_intent_freezes）：生产者从不往
    # final_reason 写这个值，它留在集合里只为带此值的
    # 落盘数据兜底。
    _CONTROLLED_CARRY_FREEZE_REASONS = frozenset({
        "invariant_failed",
        "commit_unverified",
        "reconcile_undetermined",
    })

    def _controlled_carry_intent_freezes(self):
        """按已落盘意图重建受控入口冻结旗（补修三轮，复查
        A；补修四轮，第二十三份复查 A 的第二条断链）：进
        程 1 里执行结论带冻结级痕迹的笔，意图已落盘；进
        程 2 构造时在这里把同一原因的冻结重新置上——失
        败回执保留之外，冻结本身也必须保留。pending 意
        图不归这里（对账管），aborted 是没执行成，都不
        承接。

        两条承接来源分开看、分开标注，不许混：
        - carried_from_status：「提交未定案」这个事实看
          status 本身——status == "committed_unverified"
          的已决意图无条件以 commit_unverified 承接。
          final_reason 只记不变量结论（两个生产者只写
          ""/invariant_failed），未定案那笔的
          final_reason 天生是空串；三轮时拿它当承接闸
          口，等于把这类冻结挡在门外——同进程冻着、重
          启解冻的断链正在这里。
        - carried_from_intent：不变量结论看 final_reason
          ——命中 _CONTROLLED_CARRY_FREEZE_REASONS 的按
          原口径承接。
        同一笔两条都命中时先记 status 来源（与进程内的
        产生顺序一致：未定案留痕在不变量结论之前，冻结
        原因以最早的为准）；status 来源已承接
        commit_unverified 的，final_reason 同值不再重复
        记一笔。"""
        for intent in self._controlled_intents.values():
            if intent.get("status") not in (
                "committed", "committed_unverified",
            ):
                # 只承接已决执行（提交点已落账/待核实）；
                # pending 归对账、aborted 是没执行成，都不承接。
                continue
            status_carried = (
                intent.get("status") == "committed_unverified"
            )
            if status_carried:
                self._controlled_note_health(
                    "commit_unverified",
                    {
                        "carried_from_status": intent.get("op_id"),
                        "action": intent.get("action"),
                        "memory_id": intent.get("memory_id"),
                    },
                )
            reason = intent.get("final_reason")
            if reason in self._CONTROLLED_CARRY_FREEZE_REASONS:
                if status_carried and reason == "commit_unverified":
                    # 同一事实已由 status 来源承接，不重复记。
                    continue
                self._controlled_note_health(
                    reason,
                    {
                        "carried_from_intent": intent.get("op_id"),
                        "action": intent.get("action"),
                        "memory_id": intent.get("memory_id"),
                    },
                )

    @staticmethod
    def _controlled_write_json(path, payload):
        """受控入口持久化落盘（补修二轮，复查第三节第 2 项）：
        写临时文件 → flush + fsync 文件 → os.replace 原子换名
        → 尽力 fsync 目录。旧写法 write_text + replace 只保证
        进程崩溃不丢，断电时新文件可能只剩半截或根本没落盘
        （回执/意图丢了，对账与重建都无从谈起）。目录 fsync
        在 Windows 上可能打不开目录句柄，失败静默跳过——文
        件级 fsync 是断电持久性的主要部分。"""
        tmp = path.with_suffix(".tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=1)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def _controlled_save_receipts(self):
        payload = {
            "version": 3,
            "receipts": self._controlled_receipts,
            "fingerprints": self._controlled_fingerprints,
            "revocations": self._controlled_revocations,
            "revocation_fingerprints":
                self._controlled_revocation_fingerprints,
        }
        self._controlled_write_json(
            self._controlled_receipts_path, payload
        )

    def _controlled_save_intents(self):
        payload = {"version": 1, "intents": self._controlled_intents}
        self._controlled_write_json(
            self._controlled_intents_path, payload
        )

    def _controlled_save_intents_quiet(self):
        """落盘操作意图，失败只记健康痕迹＋冻结、不抛（调用方
        多半正在处理另一笔失败，别让二次失败盖掉原始结论）。"""
        try:
            self._controlled_save_intents()
        except Exception:
            logger.warning("受控入口操作意图落盘失败", exc_info=True)
            self._controlled_note_health("intents_save_failed", {})

    def _controlled_trace(self, entry):
        """往受控入口留痕日志追加一行，返回该行序号。

        留痕通道独立于账本：提案被拒/影子这类没有状态变更的
        事件不许污染账本折叠；真正执行的状态变更由既有路径
        写账本事件（带 op_id），审计行同带 op_id＋提案指纹＋
        参数＋绑的版本＋前后版本戳，三方凭 op_id＋幂等键对照。
        写失败抛给调用方（_finish）按冻结协议处置，不吞。
        """
        with self._controlled_log_lock:
            index = self._controlled_trace_index
            record = {"index": index, "ts": _now_iso()}
            record.update(entry)
            with open(self._controlled_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._controlled_trace_index = index + 1
            return index

    @staticmethod
    def _controlled_version_stamp(text, meta, user_confirmed=None):
        """目标行版本戳：正文＋完整 metadata 的哈希（提案绑版本用）。

        注意（互查口径）：这是内容版本候选，不是单调修订号——
        同一内容改回去会得到同一个戳；G1 再议独立列/授权保护
        的版本覆盖，本轮先把「执行必绑版本」立住。

        补修二轮（复查第三节第 1 项）：戳面补上 user_confirmed
        独立列——demote 在 meta 之外单独写这一列，戳不覆盖它，
        列被单独改动时版本比对看不见。user_confirmed 传 None
        时退回 meta 里的同名字段（历史调用点兼容）。
        """
        if user_confirmed is None:
            user_confirmed = (meta or {}).get("user_confirmed")
        payload = (
            (text or "")
            + "\x00"
            + json.dumps(meta or {}, sort_keys=True, ensure_ascii=False)
            + "\x00uc="
            + ("1" if user_confirmed else "0")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _controlled_fingerprint(action, memory_id, params, source,
                                expected_version):
        """规范提案指纹：动作＋目标＋规范化参数＋来源＋绑的
        版本的哈希。同键重提时比指纹——同指纹才是同一提案，
        异指纹是同键冲突。会话凭据不进指纹：凭据是载体不是
        提案内容，换一张有效凭据重提同一提案仍算同一提案。"""
        payload = json.dumps(
            {
                "action": action,
                "memory_id": memory_id,
                "params": params or {},
                "instruction_source": source,
                "expected_version": expected_version,
            },
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # 按 op 查账的三态结论（补修三轮，贵妃第二十二份复查
    # D/E 收口）：找到 / 确定没落账 / 无法确定。读账本抛异
    # 常、或账本有坏行/被截断（解析出的事件集合不完整）时，
    # 「没找到」不许当成「确定没落账」——submit/revoke/
    # reconcile 三条兄弟路径共用同一套三态，不许再各自加一
    # 个只在部分调用者消费的旗。
    _LEDGER_OP_FOUND = "found"
    _LEDGER_OP_ABSENT = "absent"
    _LEDGER_OP_UNDETERMINED = "undetermined"

    @staticmethod
    def _controlled_ledger_doubtful(info):
        """账本完整性存疑：有坏行或被截断——解析出的事件
        集合不完整，任何基于折叠的结论都不许当定案（补修
        H3 复查 F2 的统一口径，受控入口各消费点共用）。"""
        return bool(
            (info or {}).get("bad_lines")
            or (info or {}).get("truncated")
        )

    def _controlled_ledger_op_state(self, op_id):
        """按 op_id 查账本，三态返回 (state, event)：
        found → (found, 事件)；absent → (absent, None)（账本
        完整且确无此 op）；undetermined → (undetermined,
        None)（读失败或账本存疑，生死定不了）。同步维护
        _controlled_op_read_failed：undetermined 时为 True
        （含坏行/截断存疑，不只是抛异常）。"""
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            logger.warning("受控入口按 op_id 查账本失败", exc_info=True)
            self._controlled_op_read_failed = True
            return (self._LEDGER_OP_UNDETERMINED, None)
        doubtful = self._controlled_ledger_doubtful(_info)
        for event in events:
            if event.get("op_id") == op_id:
                self._controlled_op_read_failed = False
                return (self._LEDGER_OP_FOUND, event)
        if doubtful:
            self._controlled_op_read_failed = True
            return (self._LEDGER_OP_UNDETERMINED, None)
        self._controlled_op_read_failed = False
        return (self._LEDGER_OP_ABSENT, None)

    def _controlled_ledger_op(self, op_id):
        """兼容旧签名：返回事件或 None。None 可能是 absent
        也可能是 undetermined——必须区分生死时用
        _controlled_ledger_op_state；本方法同步维护
        _controlled_op_read_failed（undetermined 时 True）。"""
        _state, event = self._controlled_ledger_op_state(op_id)
        return event

    def _controlled_reconcile_intents(self):
        """对账 pending 操作意图：账本有这笔 op_id → committed
        （补上 ledger_seq）；没有 → aborted。返回对账报告并存
        在实例上（controlled_entry_reconcile 转调这里）。"""
        report = {"checked": 0, "committed": 0, "aborted": 0,
                  "still_pending": 0}
        pending = [
            intent for intent in self._controlled_intents.values()
            if intent.get("status") == "pending"
        ]
        if not pending:
            # 没有待决意图也要跑成对恢复（补修 H3 复查
            # F1）：补偿已落账、原意图已撤销标记没落盘的崩
            # 溃窗口与 pending 无关，只靠 pending 触发会漏。
            report["revocation_pairs_recovered"] = (
                self._controlled_recover_revocation_pairs()
            )
            self._controlled_reconcile_report = report
            return report
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            logger.warning("受控入口对账读账本失败", exc_info=True)
            report["still_pending"] = len(pending)
            # 读账本失败 = fail-closed（复查 D）：冻结危险写，
            # 待决意图全部按未定案，不许按新提案放行。
            self._controlled_note_health(
                "reconcile_undetermined",
                {"error": "ledger_read_failed", "pending": len(pending)},
            )
            self._controlled_reconcile_report = report
            return report
        # 账本有坏行/被截断时（复查 E）：解析出的事件集合不
        # 完整，「没找到 op_id」不能当成确定没落账——待决意
        # 图留 pending 按未定案处理，并留冻结痕迹。
        ledger_doubtful = self._controlled_ledger_doubtful(_info)
        by_op = {}
        for event in events:
            op = event.get("op_id")
            if op:
                by_op[op] = event
        changed = False
        undetermined = 0
        for intent in pending:
            report["checked"] += 1
            event = by_op.get(intent.get("op_id"))
            if event is not None:
                intent["status"] = "committed"
                intent["ledger_seq"] = event.get("seq")
                report["committed"] += 1
                intent["reconciled_at"] = _now_iso()
                changed = True
            elif ledger_doubtful:
                undetermined += 1
                report["still_pending"] += 1
            else:
                intent["status"] = "aborted"
                report["aborted"] += 1
                intent["reconciled_at"] = _now_iso()
                changed = True
        if undetermined:
            self._controlled_note_health(
                "reconcile_undetermined",
                {
                    "bad_lines": (_info or {}).get("bad_lines"),
                    "truncated": (_info or {}).get("truncated"),
                    "pending": undetermined,
                },
            )
        if changed:
            self._controlled_save_intents_quiet()
        # 成对恢复（补修 H3 复查 F1）：本轮刚补成 committed
        # 的撤销意图，若原意图的已撤销标记没落盘，在这里
        # 一并补上——只补提交状态不补成对关系，幂等防线在
        # 崩溃窗口后是断的。
        report["revocation_pairs_recovered"] = (
            self._controlled_recover_revocation_pairs()
        )
        self._controlled_reconcile_report = report
        return report

    def _controlled_recover_revocation_pairs(self):
        """成对恢复（补修 H3 复查 F1）：撤销补偿已落账（撤
        销意图已 committed/committed_unverified）、但原意
        图的 revoked_by 标记没来得及落盘就猝死的窗口——按
        revokes_op_id 回填原意图 revoked_by/revoked_at。
        撤销回执登记不伪造：登记丢了的回执在查询时按撤销
        意图重建（_controlled_revocation_receipt_from_
        intent），与正向同口径。返回恢复的对数；原意图找
        不到、或已被别的撤销标记的，不动。"""
        recovered = 0
        changed = False
        for intent in list(self._controlled_intents.values()):
            if intent.get("kind") != "revocation":
                continue
            if intent.get("status") not in (
                "committed", "committed_unverified",
            ):
                continue
            orig_id = intent.get("revokes_op_id")
            if not orig_id:
                continue
            orig = self._controlled_intents.get(orig_id)
            if orig is None or orig.get("revoked_by"):
                continue
            orig["revoked_by"] = intent.get("op_id")
            orig["revoked_at"] = intent.get("ts") or _now_iso()
            recovered += 1
            changed = True
        if changed:
            self._controlled_save_intents_quiet()
        return recovered

    def controlled_entry_reconcile(self):
        """手动对账未完操作意图（启动时已自动跑一次）；返回
        {checked, committed, aborted, still_pending}。"""
        with self._lock:
            return self._controlled_reconcile_intents()

    def _controlled_receipt_from_intent(self, intent):
        """回执登记丢了时，按已提交的操作意图重建一份
        duplicate 回执（提交点在意图里，账本可复核）——不
        重新执行，只把当时的结论还给调用方。

        最终结论以意图里存的 final_ok/final_reason 为准
        （补修二轮，复查 B）：执行时不变量失败的笔，登记丢
        了重建出来也必须是失败，不许对 committed 一律
        ok=True 洗白。补修前的老意图没有 final_* 字段，属
        于结论未经核实——按未核实交回（ok=False、
        reason=committed_unverified、detail 标
        unverified_reconstruction），不许冒充干净成功。"""
        status = intent.get("status")
        detail = {
            "op_id": intent.get("op_id"),
            "ledger_seq": intent.get("ledger_seq"),
            "commit_state": status,
            "reconstructed_from_intent": True,
        }
        if intent.get("version_after") is None:
            # 执行后版本戳当时没记下（补修三轮，复查小
            # 项①）：如实明示缺失，不许静默留空、让调用方
            # 以为有戳可比；不强填现值、不伪造历史戳。
            detail["version_after_missing"] = True
        if status == "committed" and "final_ok" in intent:
            ok = bool(intent.get("final_ok"))
            reason = intent.get("final_reason") or ""
        elif status == "committed":
            ok = False
            reason = "committed_unverified"
            detail["unverified_reconstruction"] = True
        elif status == "committed_unverified":
            ok = False
            reason = intent.get("final_reason") or "committed_unverified"
        else:
            ok = False
            reason = intent.get("final_reason") or ""
        return {
            "ok": ok,
            "status": "duplicate",
            "reason": reason,
            "action": intent.get("action"),
            "memory_id": intent.get("memory_id"),
            "idempotency_key": intent.get("idempotency_key"),
            "duplicate": True,
            "question_id": None,
            "trace_index": None,
            "version_before": intent.get("expected_version"),
            "version_after": intent.get("version_after"),
            "detail": detail,
            "ts": _now_iso(),
        }

    def _controlled_revocation_receipt_from_intent(self, rev_intent):
        """撤销回执登记丢了时，按已提交的撤销意图重建一份
        duplicate 回执（与正向 _controlled_receipt_from_
        intent 同规）：最终结论以意图里存的 final_ok/
        final_reason 为准；崩溃窗口里的撤销意图没有
        final_*（补偿落账了但核对没跑完）——按未核实交回
        （ok=False、reason=committed_unverified），不许冒
        充干净成功。"""
        status = rev_intent.get("status")
        detail = {
            "revokes_op_id": rev_intent.get("revokes_op_id"),
            "revocation_op_id": rev_intent.get("op_id"),
            "commit_state": status,
            "ledger_seq": rev_intent.get("ledger_seq"),
            "reconstructed_from_intent": True,
        }
        if rev_intent.get("version_after") is None:
            detail["version_after_missing"] = True
        if status == "committed" and "final_ok" in rev_intent:
            ok = bool(rev_intent.get("final_ok"))
            reason = rev_intent.get("final_reason") or ""
        elif status == "committed":
            ok = False
            reason = "committed_unverified"
            detail["unverified_reconstruction"] = True
        elif status == "committed_unverified":
            ok = False
            reason = (
                rev_intent.get("final_reason")
                or "committed_unverified"
            )
        else:
            ok = False
            reason = rev_intent.get("final_reason") or ""
        return {
            "ok": ok,
            "status": "duplicate",
            "reason": reason,
            "revokes_op_id": rev_intent.get("revokes_op_id"),
            "revocation_op_id": rev_intent.get("op_id"),
            "action": rev_intent.get("action"),
            "memory_id": rev_intent.get("memory_id"),
            "idempotency_key": rev_intent.get("idempotency_key"),
            "duplicate": True,
            "question_id": None,
            "trace_index": None,
            "version_before": rev_intent.get("expected_version"),
            "version_after": rev_intent.get("version_after"),
            "detail": detail,
            "ts": _now_iso(),
        }

    def _controlled_prior_revocation(self, rev_key, fingerprint,
                                     target_op_id):
        """按撤销键＋目标查先行撤销（补修 H3 复查 F1）：回
        执登记之外必须查持久化撤销意图——登记没落盘的崩
        溃窗口只查登记，会让同键重入穿到执行段二次补偿。
        返回 (kind, payload)：("duplicate", 重建回执) /
        ("conflict", 先行指纹) / ("unresolved", 撤销意图)
        / ("already_revoked", 撤销意图) / (None, None)。
        与正向 _controlled_prior 同口径：异指纹先判冲突；
        pending 先对账，对账后仍定不了生死按 unresolved，
        绝不放行穿到执行段；aborted 是证明没执行成，不
        拦。target_op_id 给出时还查「同目标、别的键」的已
        决撤销（换键再撤同一笔）。"""
        matches = [
            intent for intent in self._controlled_intents.values()
            if intent.get("kind") == "revocation"
            and intent.get("idempotency_key") == rev_key
        ]
        for intent in matches:
            if intent.get("fingerprint") != fingerprint:
                return ("conflict", intent.get("fingerprint"))
        for intent in matches:
            if intent.get("status") == "pending":
                self._controlled_reconcile_intents()
                intent = self._controlled_intents.get(
                    intent.get("op_id"), intent
                )
            if intent.get("status") == "pending":
                return ("unresolved", intent)
            if intent.get("status") in (
                "committed", "committed_unverified",
            ):
                return (
                    "duplicate",
                    self._controlled_revocation_receipt_from_intent(
                        intent
                    ),
                )
        if target_op_id:
            for intent in self._controlled_intents.values():
                if intent.get("kind") != "revocation":
                    continue
                if intent.get("revokes_op_id") != target_op_id:
                    continue
                if intent.get("status") in (
                    "committed", "committed_unverified",
                ):
                    return ("already_revoked", intent)
                if intent.get("status") == "pending":
                    return ("unresolved", intent)
        return (None, None)

    def _controlled_prior(self, idem_key, fingerprint):
        """按幂等键查先行处理：返回 (kind, payload)——
        ("duplicate", 回执副本) / ("conflict", 先行指纹) /
        (None, None)（没处理过，或只有已中止的同指纹意图，
        可按新提案继续）。回执登记里有就先认登记；登记丢了
        按操作意图找（提交点在意图里）。老登记（补修轮之前
        落的、没指纹）认不出指纹，按 duplicate 处理并在报告
        口径里如实说明——那批键是旧版影子/拒绝攒下的，不许
        因为认不出就放行重执行。"""
        prev = self._controlled_receipts.get(idem_key)
        if prev is not None:
            stored_fp = self._controlled_fingerprints.get(idem_key)
            if stored_fp is not None and stored_fp != fingerprint:
                return ("conflict", stored_fp)
            dup = dict(prev)
            if stored_fp is None:
                # 无指纹的老回执（v1 时代，复查第三节第 6
                # 项）：指纹绑定核对做不了，明示标
                # legacy/unverified，不许冒充已核对的重复。
                dup_detail = dict(dup.get("detail") or {})
                dup_detail["legacy_receipt"] = True
                dup_detail["unverified"] = True
                dup["detail"] = dup_detail
            return ("duplicate", dup)
        matches = [
            intent for intent in self._controlled_intents.values()
            if intent.get("idempotency_key") == idem_key
        ]
        for intent in matches:
            if intent.get("fingerprint") != fingerprint:
                return ("conflict", intent.get("fingerprint"))
        for intent in matches:
            if intent.get("status") == "pending":
                self._controlled_reconcile_intents()
                intent = self._controlled_intents.get(
                    intent.get("op_id"), intent
                )
            if intent.get("status") == "pending":
                # 对账后仍定不了生死（读账本失败/坏行存疑已
                # 在对账里冻结）：这笔到底执行没执行不知道，
                # 绝不许按新提案放行（复查 D）。
                return ("unresolved", intent)
            if intent.get("status") in ("committed", "committed_unverified"):
                return (
                    "duplicate",
                    self._controlled_receipt_from_intent(intent),
                )
        return (None, None)

    def _controlled_members_match_ledger(self, events):
        """成员 ID 集合与账本折叠一致（正向不变量与撤销核
        对共用同一份口径）：投影侧读 SQLite 活行、账本侧
        读折叠后未死状态的 meta，两侧都走 core_cap.
        classify，成员/占座/钉死/临时钉死四组 ID 集合逐组
        相等才算一致。调用方负责账本完整性判定：坏行/截
        断时不许拿本方法的结论下定案。"""
        folded = _core_cap.fold_ledger_metas(events)
        states = _ledger.resolve_state(events).get(
            "key_states", {}
        )

        def _id_sets(pairs):
            members, seats, pinned, temp = set(), set(), set(), set()
            for mid, m in pairs:
                c = _core_cap.classify(m, live=True)
                if c["protection_tier"] == _core_cap.TIER_PINNED:
                    pinned.add(mid)
                elif (
                    c["protection_tier"]
                    == _core_cap.TIER_TEMP_PINNED
                ):
                    temp.add(mid)
                if c["member"]:
                    members.add(mid)
                    if c["occupies_seat"]:
                        seats.add(mid)
            return members, seats, pinned, temp

        ledger_pairs = [
            (key, meta)
            for key, meta in folded.items()
            if states.get(key)
            not in _core_cap._LEDGER_DEAD_STATES
        ]
        proj_rows = self._conn.execute(
            "SELECT id, metadata_json FROM memories"
            " WHERE voided=0 AND deleted=0 AND superseded=0"
        ).fetchall()
        proj_pairs = []
        for rid, mjson in proj_rows:
            try:
                proj_pairs.append(
                    (rid, json.loads(mjson or "{}"))
                )
            except (ValueError, TypeError):
                proj_pairs.append((rid, {}))
        return _id_sets(proj_pairs) == _id_sets(ledger_pairs)

    def _controlled_check_invariants(self, memory_id, before_text,
                                     before_meta, action):
        """执行后不变量健康闸口（补修轮扩项）：逐项查——
        座位账自洽、成员集合与账本重算一致、原文未变、变更
        只落在该动作该改的字段上、保护/抽屉/确认标记与账本
        最新口径一致、SQLite 列与 meta 一致、终态旗没被误动；
        mark_dormant 另查账本折叠态与 meta 的 decay_state 都
        已沉底。返回 {"ok", "checks"}；调用方持锁。任一项
        不过，执行结论不许报干净成功（走 invariant_failed）。
        """
        checks = {}
        snap = self.core_seat_snapshot()
        checks["seats_reconciled"] = bool(snap.get("reconciled"))
        try:
            # 成员按 ID 集合比对（贵妃第十八份复查附带项）：
            # 只比四个计数时，同计数换一批成员也报一致，与
            # 本项注释口径（成员集合一致）不符。两侧都走
            # core_cap.classify 同一口径：投影侧读 SQLite 活
            # 行，账本侧读折叠后未死状态的 meta；档位集合对
            # 全部活行先算（与 count_seats 的计数顺序一致），
            # 成员/占座再按成员资格筛。
            events, _info = _ledger.read_events(self.ledger_path)
            if self._controlled_ledger_doubtful(_info):
                # 账本残缺（补修 H3 复查 F2）：残余事件折
                # 出的成员集合可能恰好与残缺投影自洽，不
                # 许判一致——按不一致（未定案）处理。
                checks["members_match_ledger"] = False
            else:
                checks["members_match_ledger"] = (
                    self._controlled_members_match_ledger(events)
                )
        except Exception:
            checks["members_match_ledger"] = False
        row = self._conn.execute(
            "SELECT text, metadata_json, user_confirmed, superseded,"
            " deleted, voided FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        if row is None:
            checks["row_present"] = False
            return {"ok": False, "checks": checks}
        checks["row_present"] = True
        text_after = row[0] or ""
        try:
            meta_after = json.loads(row[1] or "{}")
        except Exception:
            meta_after = {}
        # 原文未变：四个动作都不许动正文。
        checks["text_unchanged"] = text_after == (before_text or "")
        # 变更只落在该动作该改的字段上，其余字段（槽位/来源/
        # 时间等）一律不许顺手改。
        allowed = {
            "demote": {"user_confirmed"},
            "set_protection_tier": {_core_cap.PROTECTION_TIER_FIELD},
            "set_core_drawer": {_core_cap.DRAWER_FIELD},
            "mark_dormant": {"decay_state"},
        }.get(action, set())
        before_meta = before_meta or {}
        changed = {
            field
            for field in set(before_meta) | set(meta_after)
            if before_meta.get(field) != meta_after.get(field)
        }
        checks["fields_confined"] = changed <= allowed
        # 保护/抽屉/确认标记与账本折叠口径一致（账本是真相源：
        # 以账本折出的 meta 为准比 SQLite 里的 meta）。
        try:
            events, _info = _ledger.read_events(self.ledger_path)
            if self._controlled_ledger_doubtful(_info):
                # 账本残缺（补修 H3 复查 F2）：折叠口径不可
                # 信，保护/抽屉/确认标记不许拿残缺折叠判一
                # 致——走 except 的不一致口径。
                raise ValueError("ledger_doubtful")
            ledger_metas = _core_cap.fold_ledger_metas(events)
            ledger_meta = ledger_metas.get(memory_id)
            if ledger_meta is None:
                checks["protection_matches_ledger"] = False
            else:
                checks["protection_matches_ledger"] = (
                    _core_cap.normalize_protection_tier(ledger_meta)
                    == _core_cap.normalize_protection_tier(meta_after)
                    and _core_cap.normalize_core_drawer(ledger_meta)
                    == _core_cap.normalize_core_drawer(meta_after)
                    and bool(ledger_meta.get("user_confirmed"))
                    == bool(meta_after.get("user_confirmed"))
                )
            if action == "mark_dormant":
                state = _ledger.resolve_state(events)
                key_states = (state or {}).get("key_states") or {}
                checks["dormant_matches_ledger"] = (
                    key_states.get(memory_id)
                    == _ledger.LEDGER_STATE_DORMANT
                    and meta_after.get("decay_state")
                    == _decay.DECAY_STATE_DORMANT
                )
        except Exception:
            checks["protection_matches_ledger"] = False
            if action == "mark_dormant":
                checks["dormant_matches_ledger"] = False
        # SQLite 列与 meta 一致；终态旗（取代/删除/作废）没被
        # 这次变更误动（进来时验过是活行，出去还必须是活行）。
        checks["columns_match_meta"] = (
            bool(row[2]) == bool(meta_after.get("user_confirmed"))
            and not row[3] and not row[4] and not row[5]
        )
        return {"ok": all(checks.values()), "checks": checks}

    def controlled_entry_version(self, memory_id):
        """目标行当前版本戳（提案绑版本用）；行不存在返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT text, metadata_json, user_confirmed"
                " FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            return self._controlled_version_stamp(
                row[0] or "", json.loads(row[1] or "{}"), row[2]
            )

    def controlled_entry_receipt(self, idempotency_key):
        """按幂等键查已登记回执；登记里没有时按已提交的操作
        意图重建一份（提交/恢复协议）；都没有返回 None。"""
        receipt = self._controlled_receipts.get(idempotency_key)
        if receipt is not None:
            out = dict(receipt)
            if idempotency_key not in self._controlled_fingerprints:
                detail = dict(out.get("detail") or {})
                detail["legacy_receipt"] = True
                detail["unverified"] = True
                out["detail"] = detail
            return out
        for intent in self._controlled_intents.values():
            if intent.get("idempotency_key") != idempotency_key:
                continue
            if intent.get("status") in ("committed", "committed_unverified"):
                return self._controlled_receipt_from_intent(intent)
        return None

    def issue_session(self, *, speaker="owner", scope=None,
                      ttl_seconds=3600, owner_direct=False,
                      issuance_source):
        """签发受控入口会话凭据（立项 H2；F1 补修收口）；
        口径不合返回 None。

        只供宿主内部流程与测试：issuance_source 必传、无默认
        值——签发来源必须由调用方显式声明，未标来源在
        session_auth 层一律拒签，不许再靠默认参数裸签出
        「owner 说话人＋全范围＋无来源」的凭据。scope 不传时
        默认受控动作全集，调用方应按最小够用传窄范围。主人
        直达凭据必须显式 issuance_source="owner" 且
        speaker="owner" 才签得出（见 session_auth 口径）；
        web 层不许自签，只许透传宿主签发的凭据。
        """
        return self._session_authority.issue(
            speaker=speaker,
            scope=self.CONTROLLED_ACTIONS if scope is None else scope,
            ttl_seconds=ttl_seconds,
            owner_direct=owner_direct,
            issuance_source=issuance_source,
            allowed_actions=self.CONTROLLED_ACTIONS,
        )

    def revoke_session(self, session_id):
        """撤销一张会话凭据（立项 H2；H3 撤销接口的凭据侧接缝）。

        撤销语义（H2 补修钉死）：撤销与受控入口的执行共用
        本实例的锁串行化——撤销返回时，不可能还有一笔受控
        执行正在半途；撤销返回之后开始的验签必然看到已撤
        销状态，已开始但尚未落笔的执行由落笔前复查阻断
        （见 submit / revoke_controlled_change 的落笔前
        复查）。即：撤销返回后，尚未落笔的执行必须被阻断。
        """
        with self._lock:
            return self._session_authority.revoke(session_id)

    def _queue_controlled_entry_question(self, action, memory_id,
                                         instruction_source, block_reason):
        """受控入口拦下的提案攒一条知情待问（先例：change_attempt）。

        只知情、不执行：BOOS 答「知道了」只是确认知晓，永远不
        因此放行被拦的提案。问题本体 memory_id=None（先例同），
        目标 id、动作与拦截原因记在 extra 里备查。
        """
        action_cn = {
            "demote": "降级",
            "set_protection_tier": "改保护档",
            "set_core_drawer": "改核心抽屉",
            "mark_dormant": "标记休眠",
            "revoke": "撤销受控变更",
        }.get(action, action)
        return self.pending_questions.add(
            _pq.QUESTION_KIND_AUTH_CLAIM,
            "受控入口拦下一条%s提案（来源：%s，目标记忆 id：%s，"
            "拦截原因：%s），已拒绝、未执行。"
            % (action_cn, instruction_source, memory_id, block_reason),
            _pq.CHANGE_ATTEMPT_QUESTION_OPTIONS,
            "受控入口门禁：改记忆类动作只认 BOOS 本人指令，"
            "核心保护行与钉死区不许自动变更（立项 H1）",
            memory_id=None,
            extra={
                "auth_action": "controlled_entry_blocked",
                "target_memory_id": memory_id,
                "attempted_action": action,
                "block_reason": block_reason,
            },
        )

    def submit_controlled_proposal(self, proposal):
        """统一受控入口：提交一条结构化状态变更提案，返回统一回执。

        提案形状（dict）：
          action             demote / set_protection_tier /
                             set_core_drawer / mark_dormant
          memory_id          目标记忆 id（非空 str）
          reason             一句理由（只留痕，不参与判定）
          instruction_source 指令来源（口径同 auth_guard：
                             None / "owner" 才算 BOOS 本人指令）
          idempotency_key    幂等键（必填非空 str；占键语义见
                             本节头部注释）
          params             动作参数（出现就必须是 dict）：
                             set_protection_tier 必须显式给
                             {"tier": "pinned"/"temp_pinned"/None}，
                             set_core_drawer 必须显式给
                             {"drawer": 五类键/None}——缺字段拒
                             bad_params；显式 None 是「摘档/摘格」
                             的显式决定，照办并在回执 detail 标
                             explicit_clear，不许拿缺字段冒充清除
          expected_version   提案时目标行的版本戳（见
                             controlled_entry_version）。执行路径
                             （开关打开时）必填：缺了拒
                             missing_version；带了就必须与执行前
                             锁内现算的戳一致，对不上拒
                             version_stale。影子路径不强制，但
                             带了戳照样验、对不上照样拒。
          session            会话凭据对象（立项 H2，issue_session
                             签发）；影子模式不验，执行开关打开后
                             必验——无凭据/过期/越界/被篡改/被撤销
                             一律拒，敏感动作（保护变更）只认
                             主人直达会话（原因码见 session_auth）

        回执形状（键恒定）：
          {ok, status, reason, action, memory_id, idempotency_key,
           duplicate, question_id, trace_index, version_before,
           version_after, detail, ts}
          status ∈ executed / refused / shadow / duplicate。
          ok 只在真执行且不变量全过时为 True；执行落了状态但
          不变量没过时 status=executed、reason=invariant_failed、
          ok=False——已提交是事实，但不许报干净成功。
          异常路径（留痕/登记落盘失败）回执照常返回，问题记在
          detail 的 audit_write_failed / receipt_persist_failed
          标记里，并冻结后续危险写（entry_frozen）。

        version_before / version_after 的状态语义（补修轮定版）：
          - 校验类早拒（unknown_action / missing_* / bad_* /
            idempotency_conflict）：两者都 None（还没载行）；
          - 载行后的拒绝（not_live 除外——它在算戳之前，两者
            都 None；not_found 同）：version_before 是拒绝当时
            的行戳，version_after 为 None（没落任何变更）；
          - shadow：version_before 是当前行戳，version_after
            为 None（影子不落目标记忆的状态；但会写审计行、
            回执登记与可能的知情待问，别说「完全不落状态」）；
          - duplicate：原回执副本，两戳即原回执当时的值；
          - executed（含 invariant_failed）：两者都有值，
            version_after 是变更后的行戳。
        降级只改核心成员状态（user_confirmed 落 False）：不删原文、
        不清保护标记；删除不在动作集里，永远走 BOOS 批准路。
        """
        if not isinstance(proposal, dict):
            proposal = {}
        action = proposal.get("action")
        memory_id = proposal.get("memory_id")
        idem_key = proposal.get("idempotency_key")
        source = proposal.get("instruction_source")
        expected_version = proposal.get("expected_version")
        params_present = "params" in proposal
        params = proposal.get("params")
        fingerprint = None
        receipt = {
            "ok": False,
            "status": "refused",
            "reason": "",
            "action": action,
            "memory_id": memory_id,
            "idempotency_key": idem_key,
            "duplicate": False,
            "question_id": None,
            "trace_index": None,
            "version_before": None,
            "version_after": None,
            "detail": {},
            "ts": _now_iso(),
        }

        def _trace(event, reason):
            entry = {
                "event": event,
                "action": action,
                "memory_id": memory_id,
                "idempotency_key": idem_key,
                "instruction_source": source,
                "reason": reason,
                "proposal_reason": proposal.get("reason") or "",
                "params": (
                    params if isinstance(params, dict)
                    else {"_raw_type": type(params).__name__}
                ),
                "expected_version": expected_version,
                "fingerprint": fingerprint,
                "version_before": receipt["version_before"],
                "version_after": receipt["version_after"],
            }
            if receipt["detail"].get("op_id"):
                entry["op_id"] = receipt["detail"]["op_id"]
            if receipt["detail"].get("ledger_seq") is not None:
                entry["ledger_seq"] = receipt["detail"]["ledger_seq"]
            return self._controlled_trace(entry)

        def _finish(status, reason, detail=None, question_id=None,
                    register=True, ok_override=None):
            receipt["status"] = status
            receipt["reason"] = reason
            receipt["ok"] = (
                status == "executed" if ok_override is None
                else bool(ok_override)
            )
            if detail:
                receipt["detail"] = detail
            if question_id:
                receipt["question_id"] = question_id
            try:
                receipt["trace_index"] = _trace(status, reason)
            except Exception:
                # 留痕写失败：回执照常返回（已落的结论不许丢），
                # 标记＋冻结后续危险写，重启对账兜底。
                logger.warning("受控入口留痕写失败", exc_info=True)
                receipt["trace_index"] = None
                receipt["detail"]["audit_write_failed"] = True
                self._controlled_note_health(
                    "audit_write_failed",
                    {"idempotency_key": idem_key, "status": status},
                )
            if register and idem_key:
                self._controlled_receipts[idem_key] = dict(receipt)
                if fingerprint is not None:
                    self._controlled_fingerprints[idem_key] = fingerprint
                try:
                    self._controlled_save_receipts()
                except Exception:
                    logger.warning(
                        "受控入口回执登记落盘失败", exc_info=True
                    )
                    receipt["detail"]["receipt_persist_failed"] = True
                    self._controlled_receipts[idem_key] = dict(receipt)
                    self._controlled_note_health(
                        "receipts_save_failed",
                        {"idempotency_key": idem_key},
                    )
            return dict(receipt)

        with self._lock:
            # —— 纯形状/参数校验（补修轮前移：不依赖行状态与
            # 开关，影子路径也先过这一关；这一段拒绝不占键）——
            if not isinstance(action, str) or action not in self.CONTROLLED_ACTIONS:
                # 未知动作（含 delete）一律拒——删除不走这道门。
                return _finish("refused", "unknown_action", register=False)
            if idem_key is None or (
                isinstance(idem_key, str) and not idem_key
            ):
                return _finish(
                    "refused", "missing_idempotency_key", register=False
                )
            if not isinstance(idem_key, str):
                return _finish(
                    "refused", "bad_idempotency_key", register=False
                )
            if memory_id is None or (
                isinstance(memory_id, str) and not memory_id
            ):
                return _finish("refused", "missing_target", register=False)
            if not isinstance(memory_id, str):
                return _finish("refused", "bad_target", register=False)
            if source is not None and not isinstance(source, str):
                return _finish("refused", "bad_source", register=False)
            if expected_version is not None and (
                not isinstance(expected_version, str)
                or not expected_version
            ):
                return _finish(
                    "refused", "bad_expected_version", register=False
                )
            if params_present and not isinstance(params, dict):
                return _finish("refused", "bad_params", register=False)
            if not params_present:
                params = {}
            # 参数值先验类型再查枚举（补修二轮，复查 C）：
            # drawer 枚举是 dict，非字符串值（list/dict）做
            # 成员测试会抛 TypeError——任何 JSON 值都只许走
            # 结构化 bad_params，不许抛异常逃出入口。
            tier_value = None
            drawer_value = None
            explicit_clear = False
            if action == "set_protection_tier":
                if "tier" not in params:
                    return _finish(
                        "refused", "bad_params",
                        detail={"param_error": "missing_field:tier"},
                        register=False,
                    )
                tier_value = params["tier"]
                if tier_value is not None and (
                    not isinstance(tier_value, str)
                    or tier_value not in _core_cap.PROTECTION_TIERS
                ):
                    return _finish(
                        "refused", "bad_params",
                        detail={"param_error": "bad_value:tier"},
                        register=False,
                    )
                explicit_clear = tier_value is None
            elif action == "set_core_drawer":
                if "drawer" not in params:
                    return _finish(
                        "refused", "bad_params",
                        detail={"param_error": "missing_field:drawer"},
                        register=False,
                    )
                drawer_value = params["drawer"]
                if drawer_value is not None and (
                    not isinstance(drawer_value, str)
                    or drawer_value not in _core_cap.CORE_DRAWERS
                ):
                    return _finish(
                        "refused", "bad_params",
                        detail={"param_error": "bad_value:drawer"},
                        register=False,
                    )
                explicit_clear = drawer_value is None
            fingerprint = self._controlled_fingerprint(
                action, memory_id, params, source, expected_version
            )
            # —— 幂等查重：同键同指纹回原结论，同键异指纹拒 ——
            prior_kind, prior_payload = self._controlled_prior(
                idem_key, fingerprint
            )
            if prior_kind == "conflict":
                return _finish(
                    "refused", "idempotency_conflict",
                    detail={
                        "fingerprint": fingerprint,
                        "prior_fingerprint": prior_payload,
                    },
                    register=False,
                )
            if prior_kind == "unresolved":
                return _finish(
                    "refused", "prior_unresolved",
                    detail={
                        "op_id": (prior_payload or {}).get("op_id"),
                        "commit_state": "pending",
                        "frozen_reason": self._controlled_frozen_reason,
                    },
                    register=False,
                )
            if prior_kind == "duplicate":
                dup = prior_payload
                dup["duplicate"] = True
                dup["status"] = "duplicate"
                receipt["detail"] = dup.get("detail") or {}
                try:
                    dup["trace_index"] = _trace(
                        "duplicate", dup.get("reason") or ""
                    )
                except Exception:
                    logger.warning(
                        "受控入口留痕写失败（duplicate）", exc_info=True
                    )
                    dup["trace_index"] = None
                    self._controlled_note_health(
                        "audit_write_failed",
                        {"idempotency_key": idem_key,
                         "status": "duplicate"},
                    )
                return dup
            # —— 载行 ——
            row = self._conn.execute(
                "SELECT text, metadata_json, superseded, deleted,"
                " voided, user_confirmed FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return _finish("refused", "not_found")
            if row[2] or row[3] or row[4]:
                return _finish("refused", "not_live")
            text = row[0] or ""
            meta = json.loads(row[1] or "{}")
            receipt["version_before"] = self._controlled_version_stamp(
                text, meta, row[5]
            )
            # 验来源（授权硬判，实测问题一）：改记忆类动作只认
            # BOOS 本人指令；非本人来源拒绝＋攒知情待问
            # （先例：delete_memory 的 not_owner_instruction）。
            if not _auth_guard.is_owner_instruction(source):
                question = self._queue_controlled_entry_question(
                    action, memory_id, source, "not_owner_instruction"
                )
                return _finish(
                    "refused", "not_owner_instruction",
                    question_id=(question or {}).get("id"),
                )
            # 验保护：核心保护行全动作拒绝；钉死/临时钉死行不许
            # 被降级或标记休眠（降级不清保护标记，契约三）。
            tier = _core_cap.normalize_protection_tier(meta)
            if self._is_core_protected_row(meta) or (
                action in ("demote", "mark_dormant") and tier is not None
            ):
                question = self._queue_controlled_entry_question(
                    action, memory_id, source, "core_protected"
                )
                return _finish(
                    "refused", "core_protected",
                    detail={"protection_tier": tier},
                    question_id=(question or {}).get("id"),
                )
            # 验版本（带了戳的）：提案绑的版本与锁内现算的戳
            # 对不上，提案作废重判，不许拿旧批准改新内容
            # （契约三）。影子路径同样验——带戳影子也不能拿
            # 旧戳蒙混。
            if (
                expected_version is not None
                and expected_version != receipt["version_before"]
            ):
                return _finish("refused", "version_stale")
            # 执行开关：默认关——门禁全走完，只出影子回执，
            # 不落目标记忆的任何状态（立项 H1）。
            if not self.controlled_entry_execution_enabled:
                return _finish("shadow", "execution_disabled")
            # 健康冻结闸口（补修轮）：登记/意图/审计持久化出过
            # 故障时，执行类动作一律拒，只许拒绝与影子。
            if self._controlled_frozen_reason:
                return _finish(
                    "refused", "entry_frozen",
                    detail={
                        "frozen_reason": self._controlled_frozen_reason,
                    },
                )
            # 会话认证（立项 H2）：开关打开后，执行前必须验提案
            # 携带的会话凭据——有效、在范围内、未过期；敏感动作
            # （保护变更）只认主人直达会话。验不过一律拒，原因码
            # 由 session_auth 给（no_session / session_expired /
            # out_of_scope / not_owner_direct / session_tampered /
            # session_revoked），不落任何状态。
            try:
                verdict = self._session_authority.validate(
                    proposal.get("session"), action
                )
            except Exception:
                # 验签绝不许把异常抛穿受控入口（补修 F3）：
                # 畸形凭据按篡改类结构化拒绝，不落任何状态。
                logger.warning("会话验签异常（提交）", exc_info=True)
                verdict = {
                    "ok": False, "reason": "session_tampered",
                    "session_id": None,
                }
            if not verdict["ok"]:
                return _finish(
                    "refused", verdict["reason"],
                    detail={"session_reason": verdict["reason"]},
                )
            # 动作前置检查（纯读，先于落意图：不合的提案不许
            # 在意图登记里留 pending 垃圾）。
            if action == "demote" and not _core_cap.is_core_member(meta):
                return _finish("refused", "not_core_member")
            if (
                action == "mark_dormant"
                and meta.get("decay_state") == _decay.DECAY_STATE_DORMANT
            ):
                return _finish("refused", "already_dormant")
            # 版本必填（补修轮，仅执行路径）：走到这里还没绑
            # 版本的提案不许执行——不许探针式地「先改了再说」。
            if expected_version is None:
                return _finish("refused", "missing_version")
            # —— 操作意图落盘（提交/恢复协议的提交点） ——
            op_id = "op-" + uuid.uuid4().hex
            # 执行前/后字段快照（立项 H3）：撤销接口靠它定恢复
            # 目标与漂移检查；本轮之前的老意图没这两键，撤销时
            # 按账本折叠重建（见 _controlled_derive_fields）。
            if action == "demote":
                _after_fields = {"user_confirmed": False}
            elif action == "set_protection_tier":
                _after_fields = {
                    _core_cap.PROTECTION_TIER_FIELD: tier_value,
                }
            elif action == "set_core_drawer":
                _after_fields = {_core_cap.DRAWER_FIELD: drawer_value}
            else:  # mark_dormant
                _after_fields = {
                    "decay_state": _decay.DECAY_STATE_DORMANT,
                }
            intent = {
                "op_id": op_id,
                "idempotency_key": idem_key,
                "fingerprint": fingerprint,
                "action": action,
                "memory_id": memory_id,
                "expected_version": expected_version,
                "status": "pending",
                "ts": _now_iso(),
                "ledger_seq": None,
                "version_after": None,
                "before_fields": self._controlled_before_fields(meta),
                "after_fields": _after_fields,
                "before_version": receipt["version_before"],
            }
            self._controlled_intents[op_id] = intent
            try:
                self._controlled_save_intents()
            except Exception:
                # 意图落不了盘就不许执行：没有提交点，崩溃后
                # 无从对账——宁可拒，不许裸奔。
                logger.warning(
                    "受控入口操作意图落盘失败", exc_info=True
                )
                self._controlled_intents.pop(op_id, None)
                self._controlled_note_health(
                    "intents_save_failed", {"op_id": op_id},
                )
                return _finish(
                    "refused", "intent_persist_failed", register=False
                )
            # 落笔前撤销复查（H2 补修，撤销语义钉死）：验签
            # 通过之后、真正落笔之前，凭据可能刚被撤销——
            # 撤销返回后尚未落笔的执行必须被阻断，这里复查
            # 一次撤销状态：已撤销按 session_revoked 拒，
            # 意图按 aborted 收尾，不落任何状态变更。
            if self._session_authority.is_revoked(
                verdict["session_id"]
            ):
                intent["status"] = "aborted"
                intent["abort_reason"] = "session_revoked"
                self._controlled_save_intents_quiet()
                return _finish(
                    "refused", "session_revoked",
                    detail={"session_reason": "session_revoked"},
                )
            # —— 执行：只走既有 store 路径，不新开写库逻辑；
            # 状态变更的账本事件带 op_id＋idem_key 贯通对照 ——
            detail = {}
            exec_failure = None
            exec_error = None
            try:
                if action == "demote":
                    result = self._update_core_fields(
                        memory_id, {"user_confirmed": False},
                        _op_id=op_id, _idem_key=idem_key,
                    )
                    if not result.get("ok"):
                        exec_failure = (
                            result.get("reason") or "execute_failed",
                            result,
                        )
                    else:
                        # demote 的第二笔写：user_confirmed 列与
                        # meta 同步（_update_core_fields 只写
                        # meta＋账本）。两笔同属一个 op：账本
                        # PROMOTE 已带 op_id，崩在两笔之间时
                        # 启动对账按 op_id 认 committed，列与
                        # meta 的分叉由不变量闸口与启动修复兜。
                        self._conn.execute(
                            "UPDATE memories SET user_confirmed=0"
                            " WHERE id=?",
                            (memory_id,),
                        )
                        self._conn.commit()
                        detail = dict(result)
                elif action == "set_protection_tier":
                    result = self.set_protection_tier(
                        memory_id, tier_value,
                        _op_id=op_id, _idem_key=idem_key,
                        _source=source,
                    )
                    if not result.get("ok"):
                        exec_failure = (
                            result.get("reason") or "execute_failed",
                            result,
                        )
                    else:
                        detail = dict(result)
                elif action == "set_core_drawer":
                    result = self.set_core_drawer(
                        memory_id, drawer_value,
                        _op_id=op_id, _idem_key=idem_key,
                    )
                    if not result.get("ok"):
                        exec_failure = (
                            result.get("reason") or "execute_failed",
                            result,
                        )
                    else:
                        detail = dict(result)
                elif action == "mark_dormant":
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {
                                "type": _ledger.LEDGER_EVENT_DORMANT,
                                "memory_id": memory_id,
                                "key": memory_id,
                                "text": text,
                                "op_id": op_id,
                                "idem_key": idem_key,
                            },
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning(
                            "受控入口标记休眠写账本失败", exc_info=True
                        )
                        exec_failure = ("ledger_failed", {})
                    else:
                        new_meta = dict(meta)
                        new_meta["decay_state"] = (
                            _decay.DECAY_STATE_DORMANT
                        )
                        self._conn.execute(
                            "UPDATE memories SET metadata_json=?"
                            " WHERE id=?",
                            (
                                json.dumps(
                                    new_meta, ensure_ascii=False
                                ),
                                memory_id,
                            ),
                        )
                        self._conn.commit()
                        detail = {
                            "decay_state": _decay.DECAY_STATE_DORMANT,
                        }
            except Exception as exc:  # 执行半路炸：先对账再定性
                exec_error = exc
            if exec_failure is not None:
                intent["status"] = "aborted"
                intent["abort_reason"] = exec_failure[0]
                self._controlled_save_intents_quiet()
                return _finish(
                    "refused", exec_failure[0], detail=exec_failure[1]
                )
            if exec_error is not None:
                # 已落账就算已提交：不许假报 refused，更不许
                # 只回滚 SQLite 造成账本分叉。
                op_state, event = self._controlled_ledger_op_state(
                    op_id
                )
                if op_state == self._LEDGER_OP_UNDETERMINED:
                    # 账本读失败：到底落没落账定不了，不许按
                    # 没落账判 aborted（复查 D）——按待核实
                    # 落账走完后续核验，冻结由
                    # commit_unverified 负责。
                    intent["status"] = "committed_unverified"
                    detail["commit_state"] = "committed_unverified"
                    detail["ledger_undetermined"] = True
                    detail["error"] = "%s: %s" % (
                        type(exec_error).__name__, exec_error,
                    )
                    self._controlled_note_health(
                        "commit_unverified",
                        {
                            "op_id": op_id,
                            "memory_id": memory_id,
                            "ledger_undetermined": True,
                        },
                    )
                elif op_state == self._LEDGER_OP_ABSENT:
                    intent["status"] = "aborted"
                    intent["abort_reason"] = "execute_failed"
                    self._controlled_save_intents_quiet()
                    return _finish(
                        "refused", "execute_failed",
                        detail={
                            "error": "%s: %s" % (
                                type(exec_error).__name__, exec_error,
                            ),
                        },
                    )
                else:
                    intent["status"] = "committed"
                    intent["ledger_seq"] = event.get("seq")
                    detail["commit_state"] = "committed_after_error"
                    detail["error"] = "%s: %s" % (
                        type(exec_error).__name__, exec_error,
                    )
            else:
                # 事件关联核验：这笔 op 必须在账本里找得到，
                # 找不到如实标「已提交/待核实」，不假报成功
                # 细节、不假报 refused。
                op_state, event = self._controlled_ledger_op_state(
                    op_id
                )
                if op_state != self._LEDGER_OP_FOUND:
                    intent["status"] = "committed_unverified"
                    detail["commit_state"] = "committed_unverified"
                    if op_state == self._LEDGER_OP_UNDETERMINED:
                        detail["ledger_undetermined"] = True
                    self._controlled_note_health(
                        "commit_unverified",
                        {
                            "op_id": op_id, "action": action,
                            "memory_id": memory_id,
                        },
                    )
                else:
                    intent["status"] = "committed"
                    intent["ledger_seq"] = event.get("seq")
                    detail["commit_state"] = "committed"
                    detail["ledger_seq"] = event.get("seq")
            detail["op_id"] = op_id
            if explicit_clear:
                detail["explicit_clear"] = True
            row_after = self._conn.execute(
                "SELECT text, metadata_json, user_confirmed"
                " FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row_after is not None:
                receipt["version_after"] = self._controlled_version_stamp(
                    row_after[0] or "",
                    json.loads(row_after[1] or "{}"),
                    row_after[2],
                )
                intent["version_after"] = receipt["version_after"]
            try:
                self._controlled_save_intents()
            except Exception:
                logger.warning(
                    "受控入口操作意图提交标记落盘失败", exc_info=True
                )
                detail["intent_persist_failed"] = True
                self._controlled_note_health(
                    "intents_save_failed", {"op_id": op_id},
                )
            # —— 不变量健康闸口：执行后逐项验（座位/成员/原文/
            # 字段范围/保护与账本/列与 meta），任一不过不许报
            # 干净成功；不自动回滚（账本为准，回滚只会分叉），
            # 落健康标记等恢复路与 H6 处置 ——
            invariants = self._controlled_check_invariants(
                memory_id, text, meta, action
            )
            detail["invariants"] = invariants["checks"]
            detail["invariants_ok"] = invariants["ok"]
            detail["session_id"] = verdict["session_id"]
            # 最终结论落进意图再返回（补修二轮，复查 B）：
            # 回执登记丢了时重建回执以此为准，失败不许被洗
            # 成成功。落盘失败由 quiet 版自己记冻结痕迹。
            intent["final_ok"] = bool(invariants["ok"])
            intent["final_reason"] = (
                "" if invariants["ok"] else "invariant_failed"
            )
            intent["invariant_checks"] = invariants["checks"]
            self._controlled_save_intents_quiet()
            if not invariants["ok"]:
                self._controlled_note_health(
                    "invariant_failed",
                    {
                        "op_id": op_id, "action": action,
                        "memory_id": memory_id,
                        "checks": invariants["checks"],
                    },
                )
                return _finish(
                    "executed", "invariant_failed", detail=detail,
                    ok_override=False,
                )
            return _finish("executed", "", detail=detail)



    # ---------- 撤销接口（立项 H3，2026-11-25） ----------
    #
    # 「后悔药」：经受控入口（submit_controlled_proposal）已经
    # 执行落账的变更，发现错了能按 op 反向补偿，恢复到执行
    # 前的字段状态。撤销对象范围（本轮核定）：只限受控入口
    # 四动作（demote / set_protection_tier / set_core_drawer /
    # mark_dormant）已提交（committed / committed_unverified）
    # 的执行。宿主自用路（confirm_memory 晋升、delete/revive
    # 批准路等）做出的变更不在本接口范围——那些仍归 BOOS
    # 自己的路处置；凭据撤销是 H2 的 revoke_session（凭据
    # 作废），与这里的变更回滚是两回事，命名不混。
    #
    # 口径：
    #   - 补偿走账本先行的既有路径、带自己的 op_id；撤销意图
    #     记 revokes_op_id 指回原 op，审计行同带两边 op_id；
    #     原 op 的意图、回执、账本事件一律不删不改（追加式）。
    #   - 补偿本身也是一笔受控变更：先落撤销意图（pending），
    #     补偿落账后标 committed、原意图标 revoked_by；启动
    #     对账（_controlled_reconcile_intents）同样管撤销意图。
    #   - 门禁比照原执行从严：只认 BOOS 本人来源；执行开关
    #     打开时必验会话凭据（按原动作验范围，敏感动作——
    #     保护变更——只认主人直达，与 H2 同口径）；当前行已
    #     成核心保护行（_is_core_protected_row）的一律拒，不
    #     许借撤销绕护栏；入口冻结时拒；开关关时只出影子
    #     回执、不落状态（与提案同口径）。
    #   - 栈纪律（防踩踏）：目标字段的当前值必须还等于原 op
    #     落下的值（after），说明没人动过它；被后来的变更改
    #     过就拒 state_drifted——要撤先撤最近的那一笔。
    #   - 幂等：撤销带自己的幂等键；同键同指纹重提回原回执
    #     副本（duplicate）；原 op 已被别的键撤销 → 明确拒
    #     already_revoked，不产生二次补偿。
    #   - 撤销只回滚字段状态：不删记忆、不改原文（四动作本
    #     就不动原文，补偿后逐字核对）、不碰无关字段；不许
    #     借撤销之名走删除或改原文。
    #   - 执行前字段快照：本轮起新执行的意图一律记
    #     before_fields / after_fields（见 submit 处）；更早的
    #     老意图没记时按账本折叠重建（_controlled_derive_fields），
    #     mark_dormant 的旧 decay_state 账本折不出，老意图缺
    #     快照时如实拒 before_state_unavailable，不猜。
    #   - 注意：撤销意图的幂等键与提案键共用一套登记文件，
    #     _controlled_prior 按意图 idem_key 找先行处理时可能
    #     碰到撤销意图——指纹形状不同，只会往 idempotency_
    #     conflict 的拒绝方向走，不会误放行。

    @staticmethod
    def _controlled_before_fields(meta):
        """一份 meta 里受控四动作能动的字段的当前值（规范化
        口径，与 core_cap 的 normalize 一致）：撤销的恢复目标
        与漂移检查都按这份形状比。decay_state 取原值（缺失
        记 None，恢复时按「键不存在」处理）。"""
        meta = meta or {}
        return {
            "user_confirmed": bool(meta.get("user_confirmed")),
            _core_cap.PROTECTION_TIER_FIELD:
                _core_cap.normalize_protection_tier(meta),
            _core_cap.DRAWER_FIELD:
                _core_cap.normalize_core_drawer(meta),
            "decay_state": meta.get("decay_state"),
        }

    @staticmethod
    def _controlled_action_field(action):
        """每个受控动作动的那一个字段（四动作都只动一个）。"""
        return {
            "demote": "user_confirmed",
            "set_protection_tier": _core_cap.PROTECTION_TIER_FIELD,
            "set_core_drawer": _core_cap.DRAWER_FIELD,
            "mark_dormant": "decay_state",
        }.get(action)

    @staticmethod
    def _controlled_revocation_fingerprint(target_op_id,
                                           target_idem_key, source):
        """规范撤销指纹：定位目标＋来源的哈希。同键重提时比
        指纹——同指纹才是同一撤销，异指纹是同键冲突。会话
        凭据不进指纹（同 submit 口径）。"""
        payload = json.dumps(
            {
                "target_op_id": target_op_id,
                "target_idempotency_key": target_idem_key,
                "instruction_source": source,
            },
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _controlled_find_intent(self, op_id=None, idempotency_key=None):
        """按 op_id 或原提案幂等键定位一笔受控入口执行意图；
        找不到返回 None。撤销意图（kind=revocation）不算原
        执行——不许拿这道接口去撤销一笔撤销。"""
        if op_id:
            intent = self._controlled_intents.get(op_id)
            if intent is not None and \
                    intent.get("kind") != "revocation":
                return intent
            return None
        if idempotency_key:
            for intent in self._controlled_intents.values():
                if intent.get("kind") == "revocation":
                    continue
                if intent.get("idempotency_key") == idempotency_key:
                    return intent
        return None

    def _controlled_derive_fields(self, intent):
        """老意图（本轮之前落的）没记 before/after 字段时，按
        账本重建：after 取原 op 那笔账本事件的完整 meta 快照
        折出的字段值，before 取 seq 小于原 op 的事件折出的
        同一记忆的字段值。重建不出（账本序号缺失、对不上、
        mark_dormant 的旧 decay_state 无账本载体）返回 None，
        调用方如实拒，不猜。"""
        action = intent.get("action")
        field = self._controlled_action_field(action)
        if field is None:
            return None
        ledger_seq = intent.get("ledger_seq")
        if ledger_seq is None:
            return None
        if action == "mark_dormant":
            # 旧 decay_state 在账本里没有载体（DORMANT 事件不
            # 带 meta、折叠不折 decay_state），折不出——不猜。
            return None
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            logger.warning("撤销重建字段读账本失败", exc_info=True)
            return None
        if self._controlled_ledger_doubtful(_info):
            # 账本残缺（补修 H3 复查 F2）：残余事件折出的
            # before 可能是错的（缺失的较近 PROMOTE 会改
            # 变真 before）——不猜、如实拒，调用方按
            # before_state_unavailable 回。
            return None
        op_event = None
        for event in events:
            if event.get("seq") == ledger_seq and \
                    event.get("op_id") == intent.get("op_id"):
                op_event = event
                break
        if op_event is None or not op_event.get("meta"):
            return None
        after_fields = self._controlled_before_fields(
            op_event.get("meta")
        )
        prior = [
            e for e in events
            if (e.get("seq") or 0) < ledger_seq
        ]
        folded = _core_cap.fold_ledger_metas(prior)
        before_meta = folded.get(intent.get("memory_id"))
        if before_meta is None:
            return None
        before_fields = self._controlled_before_fields(before_meta)
        return {
            "before_fields": before_fields,
            "after_fields": after_fields,
        }

    def _controlled_verify_revocation(self, memory_id, field,
                                      before_value, before_text,
                                      pre_meta, action):
        """补偿后核对（调用方持锁）：目标字段已恢复到 before
        值、原文逐字未变、补偿只动了那一个字段、账本口径与
        库内一致、座位账自洽。返回 {"ok", "checks"}。

        补修 H3 复查 F3：核对口径与正向
        _controlled_check_invariants 对齐——独立
        user_confirmed 列与 meta 一致（columns_match_meta，
        含终态旗）、成员 ID 集合与账本一致
        （members_match_ledger）必须在场：demote 补偿是两
        段写（meta/账本先落、独立列后写），第二段失败留下
        的列/meta 分叉不许被弱核对洗成干净成功。
        补修 H3 复查 F2：账本有坏行/截断时折叠结论不可
        信，ledger_matches 与 members_match_ledger 直接判
        False 并标 ledger_doubtful，不许让核对被坏行洗过。"""
        checks = {}
        row = self._conn.execute(
            "SELECT text, metadata_json, user_confirmed,"
            " superseded, deleted, voided FROM memories"
            " WHERE id=?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return {"ok": False, "checks": {"row_present": False}}
        checks["row_present"] = True
        checks["text_unchanged"] = (row[0] or "") == (before_text or "")
        try:
            meta_after = json.loads(row[1] or "{}")
        except Exception:
            meta_after = {}
        # 独立确认列与 meta 一致、终态旗没被误动（与正向
        # columns_match_meta 同口径）。
        checks["columns_match_meta"] = (
            bool(row[2]) == bool(meta_after.get("user_confirmed"))
            and not row[3] and not row[4] and not row[5]
        )
        current = self._controlled_before_fields(meta_after)
        checks["field_restored"] = current.get(field) == before_value
        pre_meta = pre_meta or {}
        changed = {
            f
            for f in set(pre_meta) | set(meta_after)
            if pre_meta.get(f) != meta_after.get(f)
        }
        checks["fields_confined"] = changed <= {field}
        ledger_doubtful = False
        try:
            events, _info = _ledger.read_events(self.ledger_path)
            ledger_doubtful = self._controlled_ledger_doubtful(_info)
            if action == "mark_dormant":
                state = _ledger.resolve_state(events)
                key_states = (state or {}).get("key_states") or {}
                checks["ledger_matches"] = (
                    key_states.get(memory_id)
                    != _ledger.LEDGER_STATE_DORMANT
                )
            else:
                ledger_metas = _core_cap.fold_ledger_metas(events)
                ledger_meta = ledger_metas.get(memory_id)
                checks["ledger_matches"] = (
                    ledger_meta is not None
                    and self._controlled_before_fields(
                        ledger_meta
                    ).get(field) == before_value
                )
            # 成员 ID 集合与账本一致（与正向
            # members_match_ledger 同口径、同一份折叠）。
            checks["members_match_ledger"] = (
                self._controlled_members_match_ledger(events)
            )
        except Exception:
            checks["ledger_matches"] = False
            checks["members_match_ledger"] = False
        if ledger_doubtful:
            # 账本残缺：上面折出来的「一致」不可信，一律
            # 判不干净并留痕（F2），不许被坏行洗过。
            checks["ledger_doubtful"] = True
            checks["ledger_matches"] = False
            checks["members_match_ledger"] = False
        snap = self.core_seat_snapshot()
        checks["seats_reconciled"] = bool(snap.get("reconciled"))
        return {"ok": all(checks.values()), "checks": checks}

    def controlled_revocation_receipt(self, idempotency_key):
        """按撤销幂等键查已登记的撤销回执；登记里没有时按
        已提交的撤销意图重建一份（提交/恢复协议，同正向
        controlled_entry_receipt 口径，补修 H3 复查 F1）；
        都没有返回 None。"""
        receipt = self._controlled_revocations.get(idempotency_key)
        if receipt is not None:
            return dict(receipt)
        for intent in self._controlled_intents.values():
            if intent.get("kind") != "revocation":
                continue
            if intent.get("idempotency_key") != idempotency_key:
                continue
            if intent.get("status") in (
                "committed", "committed_unverified",
            ):
                return self._controlled_revocation_receipt_from_intent(
                    intent
                )
        return None

    def revoke_controlled_change(self, request):
        """撤销接口：回滚一笔经受控入口已执行的变更，返回统一回执。

        请求形状（dict）：
          target_op_id / target_idempotency_key
                             定位原执行（二选一，op_id 优先）；
          reason             一句理由（只留痕，不参与判定）；
          instruction_source 指令来源（口径同 auth_guard：
                             只认 BOOS 本人）；
          idempotency_key    撤销自己的幂等键（必填非空 str）；
          session            会话凭据（issue_session 签发）；
                             执行开关打开时必验——按原动作验
                             范围，敏感动作（保护变更）只认
                             主人直达（口径同 H2）。

        回执形状（键恒定）：
          {ok, status, reason, revokes_op_id, revocation_op_id,
           action, memory_id, idempotency_key, duplicate,
           question_id, trace_index, version_before,
           version_after, detail, ts}
          status ∈ revoked / refused / shadow / duplicate。
          ok 只在真补偿且核对全过时为 True；补偿落了账但核对
          没过时 status=revoked、reason=invariant_failed、
          ok=False（已回滚是事实，但不许报干净成功）。
          version_before 是补偿前的行戳，version_after 是补
          偿后的行戳；校验类早拒两者都 None，载行后拒绝
          version_after 为 None。
        """
        if not isinstance(request, dict):
            request = {}
        target_op_id = request.get("target_op_id")
        target_key = request.get("target_idempotency_key")
        rev_key = request.get("idempotency_key")
        source = request.get("instruction_source")
        fingerprint = self._controlled_revocation_fingerprint(
            target_op_id if isinstance(target_op_id, str) else None,
            target_key if isinstance(target_key, str) else None,
            source if isinstance(source, str) else None,
        )
        receipt = {
            "ok": False,
            "status": "refused",
            "reason": "",
            "revokes_op_id": target_op_id,
            "revocation_op_id": None,
            "action": None,
            "memory_id": None,
            "idempotency_key": rev_key,
            "duplicate": False,
            "question_id": None,
            "trace_index": None,
            "version_before": None,
            "version_after": None,
            "detail": {},
            "ts": _now_iso(),
        }

        def _trace(status, reason):
            entry = {
                "event": status,
                "revocation": True,
                "action": receipt["action"],
                "memory_id": receipt["memory_id"],
                "idempotency_key": rev_key,
                "instruction_source": source,
                "reason": reason,
                "proposal_reason": request.get("reason") or "",
                "revokes_op_id": receipt["revokes_op_id"],
                "fingerprint": fingerprint,
                "version_before": receipt["version_before"],
                "version_after": receipt["version_after"],
            }
            if receipt["revocation_op_id"]:
                entry["op_id"] = receipt["revocation_op_id"]
            if receipt["detail"].get("ledger_seq") is not None:
                entry["ledger_seq"] = receipt["detail"]["ledger_seq"]
            return self._controlled_trace(entry)

        def _finish(status, reason, detail=None, question_id=None,
                    register=True, ok_override=None):
            receipt["status"] = status
            receipt["reason"] = reason
            receipt["ok"] = (
                status == "revoked" if ok_override is None
                else bool(ok_override)
            )
            if detail:
                receipt["detail"] = detail
            if question_id:
                receipt["question_id"] = question_id
            try:
                receipt["trace_index"] = _trace(status, reason)
            except Exception:
                logger.warning("撤销接口留痕写失败", exc_info=True)
                receipt["trace_index"] = None
                receipt["detail"]["audit_write_failed"] = True
                self._controlled_note_health(
                    "audit_write_failed",
                    {"idempotency_key": rev_key, "status": status},
                )
            if register and rev_key:
                self._controlled_revocations[rev_key] = dict(receipt)
                self._controlled_revocation_fingerprints[rev_key] = \
                    fingerprint
                try:
                    self._controlled_save_receipts()
                except Exception:
                    logger.warning(
                        "撤销回执登记落盘失败", exc_info=True
                    )
                    receipt["detail"]["receipt_persist_failed"] = True
                    self._controlled_revocations[rev_key] = \
                        dict(receipt)
                    self._controlled_note_health(
                        "receipts_save_failed",
                        {"idempotency_key": rev_key},
                    )
            return dict(receipt)

        with self._lock:
            # —— 纯形状校验（不占键，同 submit 口径）——
            if rev_key is None or (
                isinstance(rev_key, str) and not rev_key
            ):
                return _finish(
                    "refused", "missing_idempotency_key",
                    register=False,
                )
            if not isinstance(rev_key, str):
                return _finish(
                    "refused", "bad_idempotency_key", register=False
                )
            if target_op_id is not None and (
                not isinstance(target_op_id, str) or not target_op_id
            ):
                return _finish(
                    "refused", "bad_target", register=False
                )
            if target_key is not None and (
                not isinstance(target_key, str) or not target_key
            ):
                return _finish(
                    "refused", "bad_target", register=False
                )
            if target_op_id is None and target_key is None:
                return _finish(
                    "refused", "missing_target", register=False
                )
            if source is not None and not isinstance(source, str):
                return _finish("refused", "bad_source", register=False)
            # —— 撤销幂等查重 ——
            prev = self._controlled_revocations.get(rev_key)
            if prev is not None:
                stored_fp = \
                    self._controlled_revocation_fingerprints.get(
                        rev_key
                    )
                if stored_fp is not None and stored_fp != fingerprint:
                    return _finish(
                        "refused", "idempotency_conflict",
                        detail={
                            "fingerprint": fingerprint,
                            "prior_fingerprint": stored_fp,
                        },
                        register=False,
                    )
                dup = dict(prev)
                dup["duplicate"] = True
                dup["status"] = "duplicate"
                receipt["action"] = dup.get("action")
                receipt["memory_id"] = dup.get("memory_id")
                receipt["revokes_op_id"] = dup.get("revokes_op_id")
                receipt["detail"] = dup.get("detail") or {}
                try:
                    dup["trace_index"] = _trace(
                        "duplicate", dup.get("reason") or ""
                    )
                except Exception:
                    logger.warning(
                        "撤销接口留痕写失败（duplicate）",
                        exc_info=True,
                    )
                    dup["trace_index"] = None
                    self._controlled_note_health(
                        "audit_write_failed",
                        {"idempotency_key": rev_key,
                         "status": "duplicate"},
                    )
                return dup
            # —— 撤销幂等查重（意图侧，补修 H3 复查 F1）：
            # 登记里没有时查持久化撤销意图——补偿已落账、
            # 回执登记没落盘的崩溃窗口只查登记会漏，同键
            # 重入会穿到执行段二次补偿。
            prior_kind, prior_payload = \
                self._controlled_prior_revocation(
                    rev_key, fingerprint, None
                )
            if prior_kind == "conflict":
                return _finish(
                    "refused", "idempotency_conflict",
                    detail={
                        "fingerprint": fingerprint,
                        "prior_fingerprint": prior_payload,
                    },
                    register=False,
                )
            if prior_kind == "unresolved":
                return _finish(
                    "refused", "unresolved_revocation",
                    detail={
                        "revocation_op_id":
                            (prior_payload or {}).get("op_id"),
                        "commit_state": "pending",
                        "frozen_reason":
                            self._controlled_frozen_reason,
                    },
                    register=False,
                )
            if prior_kind == "duplicate":
                dup = dict(prior_payload)
                receipt["action"] = dup.get("action")
                receipt["memory_id"] = dup.get("memory_id")
                receipt["revokes_op_id"] = dup.get("revokes_op_id")
                receipt["revocation_op_id"] = \
                    dup.get("revocation_op_id")
                receipt["detail"] = dup.get("detail") or {}
                try:
                    dup["trace_index"] = _trace(
                        "duplicate", dup.get("reason") or ""
                    )
                except Exception:
                    logger.warning(
                        "撤销接口留痕写失败（duplicate）",
                        exc_info=True,
                    )
                    dup["trace_index"] = None
                    self._controlled_note_health(
                        "audit_write_failed",
                        {"idempotency_key": rev_key,
                         "status": "duplicate"},
                    )
                return dup
            # —— 定位原执行 ——
            intent = self._controlled_find_intent(
                op_id=target_op_id, idempotency_key=target_key,
            )
            if intent is None:
                return _finish("refused", "op_not_found")
            if intent.get("status") == "pending":
                # 先对账再定性：可能只是提交标记没落盘。
                self._controlled_reconcile_intents()
                intent = self._controlled_intents.get(
                    intent.get("op_id"), intent
                )
            receipt["revokes_op_id"] = intent.get("op_id")
            receipt["action"] = intent.get("action")
            receipt["memory_id"] = intent.get("memory_id")
            action = intent.get("action")
            memory_id = intent.get("memory_id")
            field = self._controlled_action_field(action)
            if field is None:
                return _finish("refused", "op_not_found")
            if intent.get("status") not in (
                "committed", "committed_unverified",
            ):
                return _finish(
                    "refused", "op_not_committed",
                    detail={"intent_status": intent.get("status")},
                )
            if intent.get("revoked_by"):
                return _finish(
                    "refused", "already_revoked",
                    detail={
                        "revocation_op_id": intent.get("revoked_by"),
                    },
                )
            # 同目标先行撤销意图兜底（换键重提，补修 F1）：
            # 原意图的 revoked_by 一般已由成对恢复补上；万
            # 一撤销意图已决而标记仍缺，这里再兜一层，不许
            # 换个键把同一笔撤第二遍；未定案的同目标撤销在
            # 场时同样不许穿到执行段。
            _pk, _pp = self._controlled_prior_revocation(
                rev_key, fingerprint, intent.get("op_id")
            )
            if _pk == "already_revoked":
                return _finish(
                    "refused", "already_revoked",
                    detail={
                        "revocation_op_id": (_pp or {}).get("op_id"),
                    },
                )
            if _pk == "unresolved":
                return _finish(
                    "refused", "unresolved_revocation",
                    detail={
                        "revocation_op_id": (_pp or {}).get("op_id"),
                        "commit_state": "pending",
                    },
                    register=False,
                )
            # —— 载行 ——
            row = self._conn.execute(
                "SELECT text, metadata_json, superseded, deleted,"
                " voided, user_confirmed FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return _finish("refused", "not_found")
            if row[2] or row[3] or row[4]:
                return _finish("refused", "not_live")
            text = row[0] or ""
            meta = json.loads(row[1] or "{}")
            receipt["version_before"] = \
                self._controlled_version_stamp(text, meta, row[5])
            # —— 验来源：撤销只认 BOOS 本人 ——
            if not _auth_guard.is_owner_instruction(source):
                question = self._queue_controlled_entry_question(
                    "revoke", memory_id, source,
                    "not_owner_instruction",
                )
                return _finish(
                    "refused", "not_owner_instruction",
                    question_id=(question or {}).get("id"),
                )
            # —— 验保护：当前行已成核心保护行的，撤销路不许
            # 绕护栏碰它（要动走 BOOS 自己的路）——
            if self._is_core_protected_row(meta):
                question = self._queue_controlled_entry_question(
                    "revoke", memory_id, source, "core_protected",
                )
                return _finish(
                    "refused", "core_protected",
                    question_id=(question or {}).get("id"),
                )
            # —— 恢复目标：意图里记了 before/after 就用它，
            # 没记（老意图）按账本重建，重建不出如实拒 ——
            before_fields = intent.get("before_fields")
            after_fields = intent.get("after_fields")
            if not isinstance(before_fields, dict) or \
                    not isinstance(after_fields, dict):
                derived = self._controlled_derive_fields(intent)
                if derived is None:
                    return _finish(
                        "refused", "before_state_unavailable",
                        detail={"action": action},
                    )
                before_fields = derived["before_fields"]
                after_fields = derived["after_fields"]
            before_value = before_fields.get(field)
            after_value = after_fields.get(field)
            # —— 栈纪律：字段当前值必须还是原 op 落下的值 ——
            current_fields = self._controlled_before_fields(meta)
            if current_fields.get(field) != after_value:
                return _finish(
                    "refused", "state_drifted",
                    detail={
                        "field": field,
                        "expected_current": after_value,
                        "actual_current": current_fields.get(field),
                    },
                )
            # —— 执行开关：关时只出影子回执（同提案口径）——
            if not self.controlled_entry_execution_enabled:
                return _finish("shadow", "execution_disabled")
            # —— 健康冻结闸口 ——
            if self._controlled_frozen_reason:
                return _finish(
                    "refused", "entry_frozen",
                    detail={
                        "frozen_reason": self._controlled_frozen_reason,
                    },
                )
            # —— 会话认证（H2 口径）：按原动作验范围；敏感
            # 动作（保护变更）只认主人直达 ——
            try:
                verdict = self._session_authority.validate(
                    request.get("session"), action
                )
            except Exception:
                # 同 submit 口径（补修 F3）：验签异常转结构化
                # 回执，不许抛穿撤销入口。
                logger.warning("会话验签异常（撤销）", exc_info=True)
                verdict = {
                    "ok": False, "reason": "session_tampered",
                    "session_id": None,
                }
            if not verdict["ok"]:
                return _finish(
                    "refused", verdict["reason"],
                    detail={"session_reason": verdict["reason"]},
                )
            # —— 撤销意图落盘（补偿的提交点）——
            rev_op_id = "op-" + uuid.uuid4().hex
            receipt["revocation_op_id"] = rev_op_id
            rev_intent = {
                "op_id": rev_op_id,
                "kind": "revocation",
                "revokes_op_id": intent.get("op_id"),
                "idempotency_key": rev_key,
                "fingerprint": fingerprint,
                "action": action,
                "memory_id": memory_id,
                "expected_version": receipt["version_before"],
                "status": "pending",
                "ts": _now_iso(),
                "ledger_seq": None,
                "version_after": None,
            }
            self._controlled_intents[rev_op_id] = rev_intent
            try:
                self._controlled_save_intents()
            except Exception:
                logger.warning(
                    "撤销意图落盘失败", exc_info=True
                )
                self._controlled_intents.pop(rev_op_id, None)
                self._controlled_note_health(
                    "intents_save_failed", {"op_id": rev_op_id},
                )
                return _finish(
                    "refused", "intent_persist_failed",
                    register=False,
                )
            # 落笔前撤销复查（H2 补修，同 submit 口径）：验签
            # 通过后、补偿落笔前凭据被撤销的，本次补偿必须
            # 被阻断——按 session_revoked 拒，撤销意图按
            # aborted 收尾，不落任何补偿。
            if self._session_authority.is_revoked(
                verdict["session_id"]
            ):
                rev_intent["status"] = "aborted"
                rev_intent["abort_reason"] = "session_revoked"
                self._controlled_save_intents_quiet()
                return _finish(
                    "refused", "session_revoked",
                    detail={"session_reason": "session_revoked"},
                )
            # —— 执行补偿：只走既有路径，账本先行 ——
            detail = {
                "revokes_op_id": intent.get("op_id"),
                "field": field,
                "restored_value": before_value,
                "session_id": verdict["session_id"],
            }
            exec_failure = None
            try:
                if action == "demote":
                    result = self._update_core_fields(
                        memory_id, {"user_confirmed": before_value},
                        _op_id=rev_op_id, _idem_key=rev_key,
                    )
                    if not result.get("ok"):
                        exec_failure = (
                            result.get("reason") or "execute_failed",
                            result,
                        )
                    else:
                        self._conn.execute(
                            "UPDATE memories SET user_confirmed=?"
                            " WHERE id=?",
                            (1 if before_value else 0, memory_id),
                        )
                        self._conn.commit()
                elif action == "set_protection_tier":
                    result = self.set_protection_tier(
                        memory_id, before_value,
                        _op_id=rev_op_id, _idem_key=rev_key,
                        _source="revocation",
                    )
                    if not result.get("ok"):
                        exec_failure = (
                            result.get("reason") or "execute_failed",
                            result,
                        )
                elif action == "set_core_drawer":
                    result = self.set_core_drawer(
                        memory_id, before_value,
                        _op_id=rev_op_id, _idem_key=rev_key,
                    )
                    if not result.get("ok"):
                        exec_failure = (
                            result.get("reason") or "execute_failed",
                            result,
                        )
                elif action == "mark_dormant":
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {
                                "type": _ledger.LEDGER_EVENT_REVIVE,
                                "memory_id": memory_id,
                                "key": memory_id,
                                "text": text,
                                "op_id": rev_op_id,
                                "idem_key": rev_key,
                                "revokes_op_id": intent.get("op_id"),
                            },
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning(
                            "撤销休眠写账本失败", exc_info=True
                        )
                        exec_failure = ("ledger_failed", {})
                    else:
                        new_meta = dict(meta)
                        if before_value is None:
                            new_meta.pop("decay_state", None)
                        else:
                            new_meta["decay_state"] = before_value
                        self._conn.execute(
                            "UPDATE memories SET metadata_json=?"
                            " WHERE id=?",
                            (
                                json.dumps(
                                    new_meta, ensure_ascii=False
                                ),
                                memory_id,
                            ),
                        )
                        self._conn.commit()
            except Exception as exc:
                op_state, event = self._controlled_ledger_op_state(
                    rev_op_id
                )
                if op_state == self._LEDGER_OP_ABSENT:
                    rev_intent["status"] = "aborted"
                    rev_intent["abort_reason"] = "execute_failed"
                    self._controlled_save_intents_quiet()
                    return _finish(
                        "refused", "execute_failed",
                        detail={
                            "error": "%s: %s" % (
                                type(exc).__name__, exc,
                            ),
                        },
                    )
                if op_state == self._LEDGER_OP_UNDETERMINED:
                    # 读账失败/账本存疑（补修三轮，复查 D）：补
                    # 偿到底落没落账定不了，不许记 aborted/
                    # execute_failed 假报没执行——按已提交待核
                    # 实落账，冻结由 commit_unverified 负责。
                    rev_intent["status"] = "committed_unverified"
                    detail["commit_state"] = "committed_unverified"
                    detail["ledger_undetermined"] = True
                    detail["error"] = "%s: %s" % (
                        type(exc).__name__, exc,
                    )
                    self._controlled_note_health(
                        "commit_unverified",
                        {
                            "op_id": rev_op_id,
                            "revokes_op_id": intent.get("op_id"),
                            "ledger_undetermined": True,
                        },
                    )
                else:
                    rev_intent["status"] = "committed"
                    rev_intent["ledger_seq"] = event.get("seq")
                    detail["commit_state"] = "committed_after_error"
                    detail["error"] = "%s: %s" % (
                        type(exc).__name__, exc,
                    )
            if exec_failure is not None:
                rev_intent["status"] = "aborted"
                rev_intent["abort_reason"] = exec_failure[0]
                self._controlled_save_intents_quiet()
                return _finish(
                    "refused", exec_failure[0],
                    detail=exec_failure[1],
                )
            if rev_intent["status"] not in (
                "committed", "committed_unverified",
            ):
                # 事件关联核验：补偿必须在账本里找得到。
                op_state, event = self._controlled_ledger_op_state(
                    rev_op_id
                )
                if op_state != self._LEDGER_OP_FOUND:
                    rev_intent["status"] = "committed_unverified"
                    detail["commit_state"] = "committed_unverified"
                    if op_state == self._LEDGER_OP_UNDETERMINED:
                        detail["ledger_undetermined"] = True
                    self._controlled_note_health(
                        "commit_unverified",
                        {
                            "op_id": rev_op_id,
                            "revokes_op_id": intent.get("op_id"),
                        },
                    )
                else:
                    rev_intent["status"] = "committed"
                    rev_intent["ledger_seq"] = event.get("seq")
                    detail["commit_state"] = "committed"
                    detail["ledger_seq"] = event.get("seq")
            detail["op_id"] = rev_op_id
            # 原意图标已撤销（追加式标记，原记录不删不改）。
            intent["revoked_by"] = rev_op_id
            intent["revoked_at"] = _now_iso()
            row_after = self._conn.execute(
                "SELECT text, metadata_json, user_confirmed"
                " FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row_after is not None:
                receipt["version_after"] = \
                    self._controlled_version_stamp(
                        row_after[0] or "",
                        json.loads(row_after[1] or "{}"),
                        row_after[2],
                    )
                rev_intent["version_after"] = \
                    receipt["version_after"]
            try:
                self._controlled_save_intents()
            except Exception:
                logger.warning(
                    "撤销提交标记落盘失败", exc_info=True
                )
                detail["intent_persist_failed"] = True
                self._controlled_note_health(
                    "intents_save_failed", {"op_id": rev_op_id},
                )
            # —— 补偿后核对：字段恢复/原文未变/只动一字段/
            # 账本一致/座位自洽，任一不过不许报干净成功 ——
            verify = self._controlled_verify_revocation(
                memory_id, field, before_value, text, meta, action
            )
            detail["verify"] = verify["checks"]
            detail["verify_ok"] = verify["ok"]
            # 撤销的最终结论同样落进撤销意图（补修二轮，
            # 复查 B 的一致性要求）。
            # 最终结论合成（补修 H3 复查 F4）：提交未定案
            # （committed_unverified）优先于核对结论——补
            # 偿可能已落账是事实（status 保留 revoked），
            # 但生死没定案不许报干净成功；final_* 同口径
            # 落盘，不许 final_ok=True＋空 reason 把未定案
            # 洗白（冻结已由 commit_unverified 在定案时留
            # 痕承接）。
            if rev_intent.get("status") == "committed_unverified":
                detail["commit_state"] = "committed_unverified"
                rev_intent["final_ok"] = False
                rev_intent["final_reason"] = "committed_unverified"
                rev_intent["invariant_checks"] = verify["checks"]
                self._controlled_save_intents_quiet()
                return _finish(
                    "revoked", "committed_unverified",
                    detail=detail, ok_override=False,
                )
            rev_intent["final_ok"] = bool(verify["ok"])
            rev_intent["final_reason"] = (
                "" if verify["ok"] else "invariant_failed"
            )
            rev_intent["invariant_checks"] = verify["checks"]
            self._controlled_save_intents_quiet()
            if not verify["ok"]:
                self._controlled_note_health(
                    "invariant_failed",
                    {
                        "op_id": rev_op_id,
                        "revokes_op_id": intent.get("op_id"),
                        "checks": verify["checks"],
                    },
                )
                return _finish(
                    "revoked", "invariant_failed", detail=detail,
                    ok_override=False,
                )
            return _finish("revoked", "", detail=detail)

    def void_joke_memory(self, memory_id):
        """作废一条玩笑记忆：先记账本 delete（标记 voided），再标 SQLite voided=1。

        账本是唯一真相源——不写账本的话，删库对账重建（reconcile_ledger_index）
        会把作废的玩笑记忆重新复活。返回值同时带 ok 与 voided，对齐 Mem0Store。
        """
        if not memory_id:
            return {"ok": False, "voided": False, "reason": "not_found"}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, metadata_json FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return {
                    "ok": False,
                    "voided": False,
                    "reason": "not_found",
                    "memory_id": memory_id,
                }
            meta = json.loads(row[2] or "{}")
            meta["voided"] = True
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {
                        "type": "delete",
                        "memory_id": row[0],
                        "key": row[0],
                        "text": row[1],
                        "meta": meta,
                    },
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("作废玩笑记忆记账失败。", exc_info=True)
                return {
                    "ok": False,
                    "voided": False,
                    "reason": "ledger_failed",
                    "memory_id": row[0],
                }
            self._conn.execute(
                "UPDATE memories SET voided=1 WHERE id=?", (row[0],)
            )
            self._conn.commit()
            return {"ok": True, "voided": True, "memory_id": row[0]}

    # ---------- 保护迁移完成证据（立项 H5，2026-11-25） ----------
    # 证据形态（十一核定）：双层——
    # ① 当场证据：_update_core_fields 在保护档真变化时（账本
    #    落笔＋投影落库之后、仍在店锁内）往
    #    protection_migration_evidence.jsonl 追加一条完整证据，
    #    evidence_id 与账本 seq 绑死（pme-<seq>）；只追加，
    #    本类不提供任何改/删证据的接口。
    # ② 账本派生：list_protection_events 从账本 ADD/PROMOTE
    #    事件按 presence 口径折档位轨迹，H5 之前发生的迁移也
    #    列得出，标 evidence_recorded=False（待补证）——缺证
    #    只说明没有当场证据，不许据此推断「可降级」。
    # 完成判定以账本为准（verify_protection_migration）：账本
    # 确有这笔且档位对得上、投影现状与账本档位链终点一致、
    # 当场证据在档，三者齐才算完成；对不上逐项列 mismatches，
    # 不许汇总数字掩盖。

    def _append_protection_evidence(self, evidence):
        """把一份迁移证据追加进证据日志（fsync 落盘）；成功
        True、失败 False——失败只记日志不抛：迁移本体已落
        账本，处置口径见 _update_core_fields 尾部注释。"""
        try:
            with self._protection_evidence_lock:
                with open(
                    self._protection_evidence_path, "a",
                    encoding="utf-8",
                ) as fh:
                    fh.write(
                        json.dumps(evidence, ensure_ascii=False) + "\n"
                    )
                    fh.flush()
                    os.fsync(fh.fileno())
            return True
        except Exception:
            logger.warning(
                "保护迁移证据落盘失败（已迁移待补证）：%s",
                (evidence or {}).get("evidence_id"),
                exc_info=True,
            )
            return False

    def _read_protection_evidence(self):
        """读证据日志全量：返回 (records, bad_lines)；文件不
        存在视作空。坏行跳过并计数，不许整份丢弃。"""
        records = []
        bad = 0
        try:
            with open(
                self._protection_evidence_path, "r", encoding="utf-8"
            ) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (ValueError, TypeError):
                        bad += 1
                        continue
                    if isinstance(rec, dict):
                        records.append(rec)
                    else:
                        bad += 1
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("保护迁移证据日志读取失败", exc_info=True)
        return records, bad

    def _protection_tier_history_from_ledger(self):
        """从账本派生全库保护档轨迹：返回 (history, final_tiers)。

        history：迁移条目列表（seq 升序）——ADD 只定初始档、
        不产条目；PROMOTE 起档位真变化产一条（带 evidence_id
        pme-<seq> 与 evidence_recorded 对照当场证据日志）。
        final_tiers：每条记忆账本口径的当前档位。
        档位按 presence 口径读：事件 meta 里有 protection_tier
        键才更新轨迹（键在值为 None = 摘档），与 core_cap 的
        _PROMOTE_PRESENCE_FIELDS 对账口径一致；其他 PROMOTE
        写入点带的完整 meta 里档位没变，自然不产条目。"""
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            logger.warning("保护迁移历史读账本失败", exc_info=True)
            self._protection_history_ledger_doubtful = True
            return [], {}
        # 账本残缺（补修 H3 复查 F2）：派生轨迹是残缺折
        # 叠，存疑标记留给核对路径（verify_protection_
        # migration）判未定案；列表类只读展示照常返回。
        self._protection_history_ledger_doubtful = (
            self._controlled_ledger_doubtful(_info)
        )
        recorded_ids = {
            rec.get("evidence_id")
            for rec in self._read_protection_evidence()[0]
        }
        tiers = {}
        history = []
        for event in sorted(
            (e for e in events if isinstance(e, dict)),
            key=lambda e: e.get("seq") or 0,
        ):
            etype = event.get("type")
            if etype not in (
                _ledger.LEDGER_EVENT_ADD, _ledger.LEDGER_EVENT_PROMOTE,
            ):
                continue
            mid = event.get("memory_id") or event.get("key")
            meta = event.get("meta") or {}
            if not mid or _core_cap.PROTECTION_TIER_FIELD not in meta:
                continue
            tier = meta.get(_core_cap.PROTECTION_TIER_FIELD)
            if etype == _ledger.LEDGER_EVENT_ADD:
                tiers[mid] = tier
                continue
            before = tiers.get(mid)
            if tier != before:
                seq = event.get("seq")
                history.append(
                    {
                        "evidence_id": "pme-%s" % seq,
                        "memory_id": mid,
                        "tier_before": before,
                        "tier_after": tier,
                        "ledger_seq": seq,
                        "ts": event.get("ts"),
                        "op_id": event.get("op_id"),
                        "idem_key": event.get("idem_key"),
                        "source": (
                            "controlled_entry"
                            if event.get("op_id")
                            else "host_direct"
                        ),
                        "evidence_recorded": (
                            ("pme-%s" % seq) in recorded_ids
                        ),
                    }
                )
            tiers[mid] = tier
        return history, tiers

    def list_protection_events(self, memory_id=None, limit=50):
        """保护档迁移事件（立项 H5 做真，原兼容桩返回 []）。

        数据源为账本保护类事件派生轨迹（见
        _protection_tier_history_from_ledger）：带 memory_id
        返回该条的迁移历史（seq 升序）；不带返回全库最近
        limit 条（seq 升序）。每条带 evidence_recorded：
        False = 这笔迁移没有当场证据（H5 之前发生、或证据
        落盘失败），属「待补证」。web/server.py 无参调用，
        签名向后兼容（新增可选 memory_id），返回仍是 list、
        键只增不减。"""
        history, _tiers = self._protection_tier_history_from_ledger()
        if memory_id is not None:
            history = [
                h for h in history if h["memory_id"] == memory_id
            ]
        if limit is not None and len(history) > limit:
            history = history[-limit:]
        return history

    def get_protection_migration_evidence(self, evidence_id):
        """取单份完整迁移证据：先查当场证据日志（完整版，带
        前后版本戳与原文哈希）；没有时按 pme-<seq> 从账本
        派生重建（reconstructed=True，前后版本戳等当场字段
        如实置 None）；两处都对不上返回 None。"""
        if not evidence_id:
            return None
        for rec in self._read_protection_evidence()[0]:
            if rec.get("evidence_id") == evidence_id:
                out = dict(rec)
                out["reconstructed"] = False
                return out
        history, _tiers = self._protection_tier_history_from_ledger()
        for entry in history:
            if entry["evidence_id"] == evidence_id:
                rebuilt = dict(entry)
                rebuilt["reconstructed"] = True
                rebuilt["version_before"] = None
                rebuilt["version_after"] = None
                rebuilt["text_hash"] = None
                return rebuilt
        return None

    def verify_protection_migration(self, memory_id=None,
                                    evidence_id=None):
        """核对一笔保护迁移是否真完成（贵妃 G2 接线用）。

        给 evidence_id：核对指定那一笔——证据与账本原事件逐
        项对照（迁移前后档位），再看投影现状是否与账本档位
        链终点一致；该笔之后又有更新的迁移时在
        detail.superseded_by 注明，不算不一致。
        只给 memory_id：核对它最新一笔迁移；从未迁移过的，
        核对出生档（账本 ADD 口径）与投影一致，
        evidence_id=None、detail.no_migration=True。
        返回 {ok, memory_id, evidence_id, checks, mismatches,
        detail}；ok 只在 mismatches 为空时为 True。"""
        result = {
            "ok": False,
            "memory_id": memory_id,
            "evidence_id": evidence_id,
            "checks": {},
            "mismatches": [],
            "detail": {},
        }
        history, final_tiers = (
            self._protection_tier_history_from_ledger()
        )
        if getattr(
            self, "_protection_history_ledger_doubtful", False
        ):
            # 账本残缺（补修 H3 复查 F2）：轨迹与终点档位
            # 都是残缺折叠，核对不许下「完成」定案。
            result["checks"]["ledger_complete"] = False
            result["mismatches"].append("ledger_doubtful")
        target = None
        if evidence_id is not None:
            target = next(
                (
                    h for h in history
                    if h["evidence_id"] == evidence_id
                ),
                None,
            )
            if target is None:
                result["mismatches"].append(
                    "evidence_not_found_in_ledger"
                )
                return result
            memory_id = target["memory_id"]
            result["memory_id"] = memory_id
        elif memory_id is not None:
            mine = [
                h for h in history if h["memory_id"] == memory_id
            ]
            target = mine[-1] if mine else None
            if target is not None:
                evidence_id = target["evidence_id"]
                result["evidence_id"] = evidence_id
        else:
            result["mismatches"].append("missing_target")
            return result

        with self._lock:
            row = self._conn.execute(
                "SELECT text, metadata_json, user_confirmed"
                " FROM memories"
                " WHERE id=?",
                (memory_id,),
            ).fetchone()
        if row is None:
            result["mismatches"].append("row_missing")
            return result
        row_meta = json.loads(row[1] or "{}")
        row_tier = row_meta.get(_core_cap.PROTECTION_TIER_FIELD)
        ledger_tier = final_tiers.get(memory_id)
        result["checks"]["projection_tier"] = row_tier
        result["checks"]["ledger_tier"] = ledger_tier
        result["checks"]["projection_matches_ledger"] = (
            row_tier == ledger_tier
        )
        if row_tier != ledger_tier:
            result["mismatches"].append("projection_tier_mismatch")

        if target is None:
            # 从未迁移：出生档与投影一致即无迁移可核、算完成。
            result["detail"]["no_migration"] = True
            result["ok"] = not result["mismatches"]
            return result

        evidence = self.get_protection_migration_evidence(
            evidence_id
        )
        recorded = bool(
            evidence and not evidence.get("reconstructed")
        )
        result["checks"]["evidence_recorded"] = recorded
        if not recorded:
            result["mismatches"].append("evidence_record_missing")
        if evidence is not None:
            matches = (
                evidence.get("tier_after") == target["tier_after"]
                and evidence.get("tier_before")
                == target["tier_before"]
            )
            result["checks"]["evidence_matches_ledger"] = matches
            if evidence.get("tier_after") != target["tier_after"]:
                result["mismatches"].append(
                    "evidence_tier_after_mismatch"
                )
            if evidence.get("tier_before") != target["tier_before"]:
                result["mismatches"].append(
                    "evidence_tier_before_mismatch"
                )

        later = [
            h for h in history
            if h["memory_id"] == memory_id
            and (h["ledger_seq"] or 0) > (target["ledger_seq"] or 0)
        ]
        if later:
            result["detail"]["superseded_by"] = (
                later[-1]["evidence_id"]
            )

        # 版本戳只在该笔是这行账本链最后一笔时可比：此后行被
        # 任何事件动过，戳必然不同，硬比只会假报不一致。
        if evidence is not None and evidence.get("version_after"):
            try:
                events, _info = _ledger.read_events(self.ledger_path)
                if self._controlled_ledger_doubtful(_info):
                    # 账本残缺（补修 H3 复查 F2）：事件集合
                    # 不完整，「这笔是不是最后一笔」定不
                    # 了——跳过戳比对（同读失败口径），不许
                    # 拿残缺集合下结论。
                    events = []
            except Exception:
                events = []
            my_seqs = [
                (e.get("seq") or 0)
                for e in events
                if isinstance(e, dict)
                and (e.get("memory_id") or e.get("key")) == memory_id
            ]
            if my_seqs and max(my_seqs) == (
                target["ledger_seq"] or 0
            ):
                stamp_now = self._controlled_version_stamp(
                    row[0] or "", row_meta, row[2]
                )
                stamp_ok = (
                    stamp_now == evidence["version_after"]
                )
                result["checks"]["version_stamp_matches"] = stamp_ok
                if not stamp_ok:
                    result["mismatches"].append(
                        "version_stamp_mismatch"
                    )
            else:
                result["checks"]["version_stamp_matches"] = None
        result["ok"] = not result["mismatches"]
        return result

    def deletion_scope(self, memory_id):
        """删除范围预览（2026-11-21 网页测第二轮 #6）：真查库，不是桩。

        和 mem0_store.deletion_scope 一样的形状：found、memory{id,text,metadata}、
        history_entries/history_count（该条的账本事件）、in_scope/out_of_scope。
        找不到时 found=False（调用方按 404 处理，见 web/server.py）。
        memory_id 既可以是 SQLite 的 id，也可以是账本 key（delete_memory
        同款两路查找）。
        """
        if not memory_id:
            return {"scope": "single", "memory_id": memory_id, "found": False}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, metadata_json, created_at FROM memories"
                " WHERE id=? AND voided=0 AND deleted=0",
                (memory_id,),
            ).fetchone()
            key = memory_id
            if row is None:
                state = self.ledger_state()
                rec = next(
                    (r for r in (state or {}).get("adds") or [] if r.get("key") == memory_id),
                    None,
                )
                key_states = (state or {}).get("key_states") or {}
                if rec is None or key_states.get(memory_id) == _ledger.LEDGER_STATE_DELETED:
                    return {"scope": "single", "memory_id": memory_id, "found": False}
                memory = {
                    "id": memory_id,
                    "text": rec.get("text") or "",
                    "metadata": rec.get("meta") or {},
                }
            else:
                memory = {
                    "id": row[0],
                    "text": row[1],
                    "metadata": json.loads(row[2] or "{}"),
                }
            history = []
            _scope_ledger_doubtful = False
            try:
                # read_events 返回 (events, info) 二元组，不是事件列表本身：
                # 直接 for 遍历会拿到 events 列表再拿 info 字典，dict
                # 的 .get 不会报错但事件列表被当成事件、.get 在 list 上
                # 抛 AttributeError，try 包着所以历史永远是空的
                # （2026-11-22 修：先解包再遍历事件）。
                events, _info = _ledger.read_events(self.ledger_path)
                _scope_ledger_doubtful = (
                    self._controlled_ledger_doubtful(_info)
                )
                for event in events:
                    if (event or {}).get("key") == key or (event or {}).get("memory_id") == key:
                        history.append(event)
            except Exception:
                logger.warning("删除范围读账本历史失败。", exc_info=True)
                # 读不出来＝存疑（H3 补修轮 2 F2-R2）：历史为空是
                # 「没读到」，不是「没有历史」；删除预览必须如实
                # 标存疑，不许让删除审批看着空历史＋不存疑做决定。
                _scope_ledger_doubtful = True
            return {
                "scope": "single",
                "memory_id": memory_id,
                "found": True,
                "memory": memory,
                "history_entries": history,
                "history_count": len(history),
                "ledger_doubtful": _scope_ledger_doubtful,
                "in_scope": [memory],
                "out_of_scope": [],
            }

    def pending_notices(self):
        """待问文件损坏的结论可见痕迹（崩溃恢复，追加决定十五）：
        文件坏过就报一条（原件保全在哪、为什么坏、什么时候），
        没坏过返回空——和以前桩的口径一致，不打扰正常流程。"""
        notices = []
        try:
            info = self.pending_questions.recovery_report()
        except Exception:
            info = None
        if info:
            notices.append(
                {
                    "kind": "pending_questions_corrupt",
                    "backup_path": info.get("backup_path"),
                    "reason": info.get("reason"),
                    "at": info.get("at"),
                }
            )
        # 受控入口健康痕迹（H1 补修）：不变量失败/提交待核实/
        # 持久化故障都在这里露头，与 H6 的落盘/结论页映射在
        # 这里留接缝（H6 从 _controlled_health_issues 与
        # _controlled_frozen_reason 取数）。
        for issue in getattr(self, "_controlled_health_issues", []):
            notices.append(
                {
                    "kind": "controlled_entry_health",
                    "issue": issue.get("kind"),
                    "detail": issue.get("detail"),
                    "at": issue.get("at"),
                }
            )
        # 保护迁移待补证（立项 H5）：迁移已落账本、但当场证据
        # 没落盘的逐条列示——不许静默；账本派生历史里这些条目
        # 同样标 evidence_recorded=False，两边对得上。
        for pending in getattr(
            self, "_protection_evidence_pending", []
        ):
            notices.append(
                {
                    "kind": "protection_evidence_pending",
                    "evidence_id": pending.get("evidence_id"),
                    "memory_id": pending.get("memory_id"),
                    "ledger_seq": pending.get("ledger_seq"),
                    "at": pending.get("ts"),
                }
            )
        frozen = getattr(self, "_controlled_frozen_reason", None)
        if frozen:
            notices.append(
                {
                    "kind": "controlled_entry_frozen",
                    "reason": frozen,
                    "at": _now_iso(),
                }
            )
        # 恢复健康度映射（立项 H6）：状态 frozen 必出（带
        # 残余种类与冻结原因）；本进程还没产出自己的记录
        # 时，上一进程落盘的未净记录也出——跨重启可见。
        # attention 级残余各有上面的既有 kind 露头，这里不
        # 重复汇总（映射全表见 H6 节头）。
        health = getattr(self, "_recovery_health_latest", None)
        if health and (
            health.get("status") == "frozen"
            or (
                health.get("status") != "clean"
                and not getattr(self, "_recovery_health_own", False)
            )
        ):
            notices.append(
                {
                    "kind": "recovery_health",
                    "status": health.get("status"),
                    "residuals": [
                        r.get("kind")
                        for r in health.get("residuals") or []
                    ],
                    "frozen_reason": (
                        health.get("controlled_entry") or {}
                    ).get("frozen_reason"),
                    "at": health.get("recorded_at"),
                }
            )
        return notices

    # ---------- 持锁稳定版本快照（立项 H4，2026-11-25） ----------

    @staticmethod
    def _snapshot_row_state(voided, deleted, superseded, meta):
        """快照里一行的状态名：终态旗优先（deleted > voided >
        superseded），活行再按衰减状态分 current / dormant。
        与账本折叠的 key_states 口径对齐（同一条记忆两边同名）。"""
        if deleted:
            return "deleted"
        if voided:
            return "voided"
        if superseded:
            return "superseded"
        if (meta or {}).get("decay_state") == _decay.DECAY_STATE_DORMANT:
            return "dormant"
        return "current"

    def _frozen_rows_locked(self, memory_ids=None):
        """持锁内一次读全库行，冻成 {id: 行视图}（调用方须已持
        self._lock）。行视图 = 内容版本戳（与受控入口
        controlled_entry_version 同一戳法，提案绑版本/H1 漂移
        检出共用）＋原文＋meta 深拷贝＋状态＋保护档/抽屉。
        同时返回全库活行 meta 列表（座位账从同一批读数算，
        保证快照内部同源、不半新半旧）。"""
        wanted = None
        if memory_ids is not None:
            wanted = {str(m) for m in memory_ids}
        frozen = {}
        live_metas = []
        for mid, text, mjson, voided, deleted, superseded, uc in self._conn.execute(
            "SELECT id, text, metadata_json, voided, deleted, superseded,"
            " user_confirmed"
            " FROM memories"
        ).fetchall():
            try:
                meta = json.loads(mjson or "{}")
            except Exception:
                meta = {}
            if not (voided or deleted or superseded):
                live_metas.append(meta)
            if wanted is not None and mid not in wanted:
                continue
            frozen[mid] = {
                "version": self._controlled_version_stamp(
                    text or "", meta, uc
                ),
                "text": text or "",
                "meta": copy.deepcopy(meta),
                "state": self._snapshot_row_state(
                    voided, deleted, superseded, meta
                ),
                "live": not (voided or deleted or superseded),
                "protection_tier": _core_cap.normalize_protection_tier(meta),
                "core_drawer": _core_cap.normalize_core_drawer(meta),
            }
        return frozen, live_metas

    def take_stable_snapshot(self, memory_ids=None):
        """取一份持锁稳定版本快照（立项 H4）：在 store 锁内一
        次性冻结——行版本戳＋保护版本＋账本序号（ledger_state
        的 seq_max）同一临界区内取齐，生成过程不写账本、不写
        库。整理员拿这份快照出提案、绑版本；之后库里再变，
        快照视图纹丝不动，差异走 diff_stable_snapshot 对账。

        形状：{snapshot_id, generated_at, ledger_seq,
        ledger_snapshot_seq, seats, rows, row_count, scoped}。
        seats 是全库座位账（count_seats 口径，与
        core_seat_snapshot 同源同读）；rows 可按 memory_ids
        缩小行范围（scoped=True），座位账永远全库口径。
        快照只活在进程内登记里，不落盘：进程重启即失效。
        返回的是深拷贝，调用方改它污染不了登记里的原件。"""
        with self._lock:
            frozen, live_metas = self._frozen_rows_locked(memory_ids)
            seats = _core_cap.count_seats(
                live_metas, limit=self.core_seat_limit
            )
            state = self.ledger_state() or {}
            snapshot = {
                "snapshot_id": uuid.uuid4().hex,
                "generated_at": _now_iso(),
                "ledger_seq": state.get("seq_max") or 0,
                "ledger_snapshot_seq": state.get("snapshot_seq"),
                "seats": seats,
                "rows": frozen,
                "row_count": len(frozen),
                "scoped": memory_ids is not None,
            }
            self._stable_snapshots[snapshot["snapshot_id"]] = (
                copy.deepcopy(snapshot)
            )
            return copy.deepcopy(snapshot)

    def get_stable_snapshot(self, snapshot_id):
        """按 id 取回快照（深拷贝）。没取过、已释放或进程重启
        后登记已空——如实抛 KeyError，不给假数据。"""
        with self._lock:
            snap = self._stable_snapshots.get(snapshot_id)
            if snap is None:
                raise KeyError(snapshot_id)
            return copy.deepcopy(snap)

    def release_stable_snapshot(self, snapshot_id):
        """释放一份快照：登记里摘掉，之后 get/diff 一律
        KeyError。摘到了返回 True，本来就没有返回 False。"""
        with self._lock:
            return self._stable_snapshots.pop(snapshot_id, None) is not None

    def diff_stable_snapshot(self, snapshot_id):
        """快照 vs 当前：按 memory_id 给出变化清单。变化判定
        只认版本戳与行状态（不用时间猜）：戳变了就是 changed，
        并带 state_before/state_after——顶替（→superseded）、
        进档案（→dormant）、删除（→deleted）都从状态迁移里
        看得出。added 只对全库快照有意义（scoped 快照的行集
        固定，只报 changed/removed）。快照不存在如实
        KeyError。"""
        with self._lock:
            snap = self._stable_snapshots.get(snapshot_id)
            if snap is None:
                raise KeyError(snapshot_id)
            then_rows = snap["rows"]
            now_rows, _live = self._frozen_rows_locked(
                list(then_rows) if snap.get("scoped") else None
            )
            state = self.ledger_state() or {}
            added = []
            if not snap.get("scoped"):
                added = sorted(set(now_rows) - set(then_rows))
            removed = sorted(set(then_rows) - set(now_rows))
            changed = []
            unchanged = 0
            for mid in sorted(set(then_rows) & set(now_rows)):
                before = then_rows[mid]
                after = now_rows[mid]
                if (
                    before["version"] == after["version"]
                    and before["state"] == after["state"]
                ):
                    unchanged += 1
                    continue
                changed.append(
                    {
                        "memory_id": mid,
                        "version_before": before["version"],
                        "version_after": after["version"],
                        "state_before": before["state"],
                        "state_after": after["state"],
                    }
                )
            return {
                "snapshot_id": snapshot_id,
                "generated_at": snap["generated_at"],
                "ledger_seq_then": snap["ledger_seq"],
                "ledger_seq_now": state.get("seq_max") or 0,
                "added": added,
                "removed": removed,
                "changed": changed,
                "unchanged_count": unchanged,
            }

    # ---------- 启动时要调的（接口兼容，SQLite 版里是空操作） ----------

    # ---------- 性情区原件在壳（追加决定八，2026-11-25 补建） ----------
    #
    # 决定八五环的宿主落点（纯文本活在 memory/persona_shell.py）：
    #   ① 壳原件：仓库根 SOUL.md 标记之间的 17 条（路径可注入
    #      shell_soul_path，测试用临时壳；默认按 persona_shell
    #      的仓库根口径）；
    #   ② 逐字复印件行：slot="soul"、meta 带 persona_projection
    #      只读投影标记、protection_tier="pinned"（钉死区口径，
    #      不占座——core_cap.classify 按「活行+成员+保护档为空」
    #      才占座）；_is_core_protected_row 对投影标记单独认，
    #      受控入口/自动取代/清扫/衰减一律不碰（H1 的
    #      core_protected 拒绝就是这条链）；
    #   ③ 单向同步/重印：sync_persona_projection 只读壳、写记忆，
    #      永远不写壳文件。三方哈希对照分清三种不一致：
    #      复印件哈希≠上次重印戳 → 记忆侧被动过 = 冲突（挂问、
    #      不覆盖）；复印件没变、壳哈希≠重印戳 → 壳侧正常改版，
    #      照印（重印 = 新行落账 + 旧行 supersede，账本留痕）；
    #      两边都对得上 = ok 什么都不做；
    #   ④ 注入从壳原件生成：persona_injection_block 每次现读壳，
    #      不读复印件——壳改了下一轮注入就是新版（main.py 接点）；
    #   ⑤ 一致性校验：verify_persona_consistency（启动恢复收尾
    #      自动跑一遍 _persona_startup_check）；冲突只挂待问
    #      （auth_claim 类知情问，先例同受控入口拦截问），BOOS
    #      裁定后以壳为准时调 sync_persona_projection(force=True)
    #      重印——force 只许人工裁定后用，程序内部不自调。
    #   Plan C 种子行调和：旧 soul 槽行（AGENT_SOUL_TEXT 一句话
    #   概括版，mem0 时代种下、无投影标记）在第一次同步时被
    #   「调和重印」——新投影行落账、种子行走 supersede 退役
    #   （账本留痕、不删），库里从此只有一行活的 soul 行，不
    #   留两行互相竞争的「人格」行。依据：决定八＋BOOS
    #   2026-11-25「替换」拍板——种子本就是指向 SOUL.md 的一句
    #   概括，原件回来后它没有独立存在的理由。

    def _persona_soul_rows(self):
        """活着的 soul 相关行：slot='soul' 或带投影标记的。

        返回 [{"id", "text", "meta"}]，按 write_seq 从新到旧。
        """
        rows = []
        cur = self._conn.execute(
            "SELECT id, text, metadata_json FROM memories"
            " WHERE superseded=0 AND deleted=0 AND voided=0"
        )
        for rid, text, meta_json in cur.fetchall():
            try:
                meta = json.loads(meta_json or "{}")
            except Exception:
                meta = {}
            if meta.get("slot") == "soul" or meta.get("persona_projection"):
                rows.append({"id": rid, "text": text or "", "meta": meta})
        rows.sort(
            key=lambda r: r["meta"].get("write_seq") or 0, reverse=True
        )
        return rows

    def _persona_projection(self):
        """当前活着的投影行（最新的一行）；没有返回 None。"""
        for row in self._persona_soul_rows():
            if row["meta"].get("persona_projection"):
                return row
        return None

    def _queue_persona_conflict_question(self, projection, shell_hash,
                                         copy_hash):
        """壳↔复印件冲突攒一条知情待问（先例：受控入口拦截问）。

        只知情、不裁决：BOOS 答「知道了」只是确认知晓，冲突本身
        等他裁定（以壳为准重印 / 保留记忆这份另议），程序不自决。
        """
        return self.pending_questions.add(
            _pq.QUESTION_KIND_AUTH_CLAIM,
            "性情区壳原件与核心记忆复印件对不上（记忆 id：%s），"
            "已挂起、两边都没动，等 BOOS 裁定。"
            % (projection["id"],),
            _pq.CHANGE_ATTEMPT_QUESTION_OPTIONS,
            "性情区一致性校验（追加决定八）：壳与复印件冲突不许"
            "程序自决或默默覆盖，挂起问 BOOS",
            memory_id=projection["id"],
            extra={
                "auth_action": "persona_conflict",
                "shell_sha256": shell_hash,
                "copy_sha256": copy_hash,
            },
        )

    def _print_persona_projection(self, shell_text, shell_hash):
        """把壳条文逐字印成一行新的投影行（账本先行，走既有落行路）。

        返回新行 key；账本写不进去返回 None（调用方如实报）。
        调用方需持锁（_append_new_memory 的既有口径）。
        """
        created_at = _now_iso()
        meta = {
            "source": "shell",
            "confidence": 1.0,
            "user_confirmed": True,
            "created_at": created_at,
            "write_seq": _next_write_seq(),
            "turn_id": None,
            "slot": "soul",
            "is_rule": False,
            "curated": False,
            "protection_tier": _core_cap.TIER_PINNED,
            "core_drawer": None,
            "decay_kind": _decay.DECAY_KIND_STABLE,
            "importance": "high",
            "event_date": None,
            "last_used_at": created_at,
            "decay_state": _decay.DECAY_STATE_VERBATIM,
            "persona_projection": True,
            "persona_shell_sha256": shell_hash,
        }
        dates, sessions, _added = _decay.record_seen_occurrence(
            None, None, _shanghai_today().isoformat(), None
        )
        meta["seen_dates"] = _decay._dump_seen_list(dates)
        meta["seen_sessions"] = _decay._dump_seen_list(sessions)
        return self._append_new_memory(shell_text, meta)

    def sync_persona_projection(self, force=False):
        """壳→记忆单向同步/重印。返回 {"status", "memory_id", ...}。

        status:
          created    没有任何 soul 行 → 新印投影行；
          reconciled 只有 Plan C 种子行 → 调和重印（种子退役）；
          reprinted  壳改版（复印件未动）→ 照印新版，旧行退役；
          ok         壳、复印件、重印戳三方一致，什么都不做；
          conflict   复印件被记忆侧动过 → 挂待问、不覆盖（force
                     =True 才照印，只许 BOOS 裁定后人工调用）；
          no_shell   壳原件读不到/摘不出条文段 → 不动记忆；
          ledger_failed  账本写不进去 → 没印成，如实报。
        本函数永远不写壳文件（单向是死规矩，测试钉住）。
        """
        shell_text = _persona_shell.read_shell_clauses(
            self._shell_soul_path
        )
        if not shell_text:
            return {"status": "no_shell", "memory_id": None}
        shell_hash = _persona_shell.clauses_sha256(shell_text)
        with self._lock:
            proj = None
            others = []
            for row in self._persona_soul_rows():
                if row["meta"].get("persona_projection") and proj is None:
                    proj = row
                else:
                    others.append(row)
            if proj is not None and not force:
                stored = proj["meta"].get("persona_shell_sha256")
                copy_hash = _persona_shell.clauses_sha256(proj["text"])
                if copy_hash != stored:
                    question = self._queue_persona_conflict_question(
                        proj, shell_hash, copy_hash
                    )
                    return {
                        "status": "conflict",
                        "memory_id": proj["id"],
                        "question_id": (question or {}).get("id"),
                    }
                if shell_hash == stored:
                    return {"status": "ok", "memory_id": proj["id"]}
            if proj is not None:
                status = "reprinted"
            elif others:
                status = "reconciled"
            else:
                status = "created"
            new_id = self._print_persona_projection(shell_text, shell_hash)
            if new_id is None:
                return {"status": "ledger_failed", "memory_id": None}
            retire = ([proj] if proj is not None else []) + others
            for row in retire:
                if row["id"] == new_id:
                    continue
                try:
                    self._supersede_row(
                        row["id"], row["text"], row["meta"],
                        new_id, shell_text,
                    )
                except Exception:
                    logger.warning(
                        "性情投影重印：旧行取代失败（新行已落账）。",
                        exc_info=True,
                    )
            return {"status": status, "memory_id": new_id}

    def verify_persona_consistency(self):
        """壳↔复印件一致性校验（决定八第 ⑤ 环）。

        返回 {"ok", "status", ...}：ok=逐字一致；shell_changed=
        壳改版待重印（正常流程，不挂问）；conflict=复印件被动过
        （挂待问、两边不覆盖）；missing=还没有投影行；
        legacy=只有种子行待调和；no_shell=壳读不到。
        """
        shell_text = _persona_shell.read_shell_clauses(
            self._shell_soul_path
        )
        if not shell_text:
            return {"ok": False, "status": "no_shell"}
        shell_hash = _persona_shell.clauses_sha256(shell_text)
        with self._lock:
            proj = self._persona_projection()
            if proj is None:
                return {
                    "ok": False,
                    "status": "legacy" if self._persona_soul_rows()
                    else "missing",
                }
            if proj["text"] == shell_text:
                return {
                    "ok": True, "status": "ok",
                    "memory_id": proj["id"],
                }
            copy_hash = _persona_shell.clauses_sha256(proj["text"])
            stored = proj["meta"].get("persona_shell_sha256")
            if copy_hash == stored:
                return {
                    "ok": False, "status": "shell_changed",
                    "memory_id": proj["id"],
                }
            question = self._queue_persona_conflict_question(
                proj, shell_hash, copy_hash
            )
            return {
                "ok": False, "status": "conflict",
                "memory_id": proj["id"],
                "question_id": (question or {}).get("id"),
            }

    def _persona_startup_check(self):
        """启动恢复收尾的性情校验：没行不吵、种子行待同步不吵，
        只有真投影在场才三方校验（冲突挂问在 verify 里）。"""
        rows = self._persona_soul_rows()
        if not rows:
            return {"status": "absent"}
        if self._persona_projection() is None:
            return {"status": "legacy_pending_sync"}
        return self.verify_persona_consistency()

    def persona_injection_block(self):
        """每轮注入的性情块：现读壳原件生成，不读记忆复印件。

        壳读不到返回空串（宁可不带，也不拿复印件冒充原件——
        决定八：注入从壳原件生成）。
        """
        shell_text = _persona_shell.read_shell_clauses(
            self._shell_soul_path
        )
        if not shell_text:
            return ""
        return _persona_shell.build_persona_block(shell_text)

    def ensure_identity_memories(self):
        """身份种子接点（web 启动时调）：决定八起承担性情投影同步。

        返回 {"soul": {...}}：新建/调和时 status="added"（沿调用方
        既有口径），精确结果在 persona_status 里（created /
        reconciled / reprinted / ok / conflict / no_shell /
        ledger_failed）。名字/电脑配置种子是 mem0 时代的事、账本
        里已有，本函数不再重复种（旧桩注释的口径不变）。
        """
        result = self.sync_persona_projection()
        outcome = dict(result)
        outcome["persona_status"] = result.get("status")
        if result.get("status") in ("created", "reconciled"):
            outcome["status"] = "added"
        return {"soul": outcome}

    def backfill_timestamps(self):
        """补时间戳是 mem0 索引的事，SQLite 行里本来就有 created_at。"""
        return 0


def _default_memory_dir():
    return Path(__file__).resolve().parent


# _CORRECTION_TOPIC_MIN_SIMILARITY（纠错安全网：旧-新话题相似度下限，2026-11-22 接线）：
# 单候选时，旧说法和新说法的余弦相似度低于这个值就不换，改报 ambiguous
# （问 BOOS），免得改错不相干的记忆。0.70 是 mem0 那边 2026-11-12
# 实测校准出来的分界（同话题 0.757–0.907、跨话题 0.411–0.763，
# 干净空隙 0.656–0.757，取中点约 0.70；见 mem0_store.py 同名常量）。
# 向量算不出来（模型没起来）时跳过这道网，交由调用方兜底。
_CORRECTION_TOPIC_MIN_SIMILARITY = 0.70

# 跨槽安全网的命中门槛（缺点清单第 5 条，2026-11-24）：纠正主路和
# 同槽兜底全落空后，对全部活记忆再扫一遍时用——比主路安全网的
# 0.70 更高（宁缺毋滥），且命中后还必须过判官确认才允许取代。
_CROSS_SLOT_NET_MIN_SIMILARITY = 0.80


def _cosine(a, b):
    """两条向量的余弦相似度；任一向量没有（None）或模为 0 时返回 None
    （调用方按“算不出来”处理，不当 0 分用）。"""
    if a is None or b is None:
        return None
    try:
        import numpy as np

        va = np.asarray(a, dtype="float64")
        vb = np.asarray(b, dtype="float64")
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return None
        return float(np.dot(va, vb) / (na * nb))
    except Exception:
        return None


def _decay_weight(metadata):
    """这条记忆现在的衰减权重（0.10–1.0，核心记忆恒为 1.0）。

    按 memory/decay.py 算（2026-11-22 接线）：指数衰减，
    weight = 0.5 ** (闲置天数 / 半衰期)，下限 0.10；稳定事实半衰期
    180 天、一次性经历 14 天；核心记忆恒为 1.0。
    输入从 metadata 来：decay_kind、last_used_at（没有时按 created_at
    算，再没有按 1.0）；is_core 按 provenance 的口径（已确认或有槽位）。
    权重只管排位、不管能不能搜到（资格仍只看原始余弦分 ≥ 0.4）。
    注意：这里不用 effective_weight 的未确认封顶（0.80）——封顶会改动
    已确认/未确认混排的 RRF 钉死顺序（Q4 等回归），等 BOOS 定了再接。
    """
    meta = metadata or {}
    kind = meta.get("decay_kind") or _decay.DECAY_KIND_EPISODIC
    last_used = meta.get("last_used_at") or meta.get("created_at")
    try:
        is_core = bool(_provenance._is_core_memory(meta))
    except Exception:
        is_core = bool(meta.get("is_rule") or meta.get("curated"))
    return _decay.decay_weight(kind, last_used, now=None, is_core=is_core)


def _type_weight(text):
    """这条记忆的类型权重：稳定偏好 1.25，一次性事件 / 其他 1.0。

    修 Q3 退化（2026-11-21，双写比对 18 问里唯一退化）：
    问“我爱吃什么”时，纠正行“不太喜欢鸡杂面”被新存的饺子/米线
    两行按余弦挤出前 5（RRF 0.0152 vs 米线 0.0154，一位之差）。
    根因不是排序逻辑错——同批旧行的余弦相对顺序新旧一致，是新增的
    真实行按余弦挤出一位。修法是类型权重（稳定偏好 > 一次性事件），
    不是改余弦、不是改阈值。

    为什么按正文认，不按 decay_kind：decay.py 的规矩是“BOOS 亲口说
    的偏好记 stable”，但迁移进来的偏好句在 mem0 里都是 source=inferred，
    迁移时 decay_kind 一律写成了 episodic（火锅/冒菜是 user 亲口说的，
    所以是 stable；牛肉面喜、纠正行、鸡杂面喜都是 inferred，全是
    episodic）。档位没标上，检索时按正文再认一次：说喜欢/不喜欢/
    爱吃/爱喝的句子就是稳定偏好，不管它当初怎么标的。
    权重只管排位：资格仍只看原始余弦分 ≥ 0.4（decay.py 坑 1），
    再低的权重也不会让记忆搜不到，再高的也不会把不够 0.4 的行拉进来。
    """
    if any(word in (text or "") for word in _PREFERENCE_WORDS):
        return _TYPE_WEIGHT_STABLE_PREFERENCE
    return 1.0


def _keyword_overlap(query, text):
    """关键词重合度（0–1）：按字算，简单够用。"""
    q = set(query or "")
    t = set(text or "")
    q = {c for c in q if c.strip() and c not in "，。！？、；：\"'()（）《》"}
    t = {c for c in t if c.strip() and c not in "，。！？、；：\"'()（）《》"}
    if not q:
        return 0.0
    return len(q & t) / len(q)


def _blob_to_vector(blob):
    if blob is None:
        return None
    try:
        import numpy as np

        return np.frombuffer(blob, dtype="float32")
    except Exception:
        return None


def create_sqlite_store(budget_tracker=None, **kwargs):
    """建一个 SQLiteStore；建不好时返回 None（降级成“没有记忆也能聊”）。

    建好后先跑启动崩溃恢复（startup_recovery：校验账本、修残行、
    按账本修投影的半截状态、待问文件损坏保全与重建），再对一次账
    （reconcile_ledger_index）补缺行——两步写之间崩了、或索引坏过，
    重启后靠这两笔把记忆捞回来。都不抛异常、只记日志。
    """
    try:
        store = SQLiteStore(budget_tracker=budget_tracker, **kwargs)
    except Exception as exc:
        logger.warning("SQLite 记忆未启用：%s，本次不带记忆运行。", exc)
        return None
    try:
        store.startup_recovery()
    except Exception:
        logger.warning("启动崩溃恢复失败：继续启动，后面对账照常。", exc_info=True)
    try:
        store.reconcile_ledger_index()
    except Exception:
        logger.warning("启动对账失败：本次不对索引补缺。", exc_info=True)
    try:
        # 待问重建补跑（2026-11-25 互查 (b)）：startup_recovery
        # 里的重建跑在 reconcile 补缺之前——账本里有、SQLite 里
        # 缺的行当时还不存在，靠行 meta 标记重建的 auth_claim
        # 问题会漏掉。补缺完成后再扫一遍：攒问题自带去重
        # （add 按 kind+memory_id 去重、挂起问题按原文去重），
        # 不会重复攒。
        if store.pending_questions.recovery_report():
            store.rebuild_pending_questions()
    except Exception:
        logger.warning("启动对账后待问重建补跑失败。", exc_info=True)
    return store
