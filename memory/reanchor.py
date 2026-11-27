# memory/reanchor.py（四件套·实测问题四：已确认事实回锚，2026-11-24）
#
# 背景（2026-11-24 BOOS 亲测第 ⑦ 组）：长对话越聊越松——已定案的
# 「十二是十一的弟弟」被追问时，十二竟说「当时没逐条确认过，不太
# 确定」。结构性原因有二：
# 1. 每轮注入的只有「与本轮问题相关」的检索 top-5：已确认事实与
# 本轮问题不相关时根本不出现在提示词里；就算出现，也混在写着
# 「历史记录、仅供参考」的检索块里，没有定案的身份。
# 2. 历史超 token 上限后 maybe_compact 把旧轮次压成摘要（只留最近
# 5 轮原文），早期定案的事实随旧轮次一起被稀释。
#
# 定版处置（追加决定十二 · 问题四；BOOS 已知情：这是缓解不是根治）：
# ① 每轮强制注入：已确认事实每轮都进提示词，独立成区、明确标注
# 「已确认」，并写明是定案——直接当事实用，不许再说不确定、
# 不许向 BOOS 重复确认。
# ② 中途回锚：对话达到阈值轮数后，每隔 REANCHOR_EVERY_TURNS 轮
# 用加强版回锚块替换当轮的常规块（不是叠加，避免重复灌）：
# 容量更大、语气更硬，明确重申「聊得再久也不动摇」。
#
# 「已确认」的认定不另立标准，完全沿现行口径：
# - provenance._is_core_memory（confirmed，即 user_confirmed——含
# 问题一 owner_confirmed 升级后写的 user_confirmed=True；或带
# KNOWN_SLOTS 身份槽位）；
# - 或核心受保护行（_is_core_protected_row 的 metadata 面：
# is_rule / curated）。
# 未确认的普通记忆（提取所得、来源未确认）永远进不了这个区。
#
# 本模块只做纯判定与拼装：不读库、不联网。store 侧
# SQLiteStore.confirmed_facts() 负责把活行拿出来喂进来。

from memory import provenance as _provenance
from memory.redaction import _contains_sensitive, _normalize_memory_text

# 常规每轮块的容量：12 条 / 900 字。已确认事实本来就该是少数定案
# （核心座位上限 150 是全核心口径，这里只取其中已确认/受保护的，
# 实际远少于 12）；900 字约 450–900 token，和检索块 500 token 上限
# 同量级，不许把提示词撑爆。超限时的取舍规矩见 select_anchor_facts。
ANCHOR_MAX_ITEMS = 12
ANCHOR_MAX_CHARS = 900

# 回锚加强版的容量：30 条 / 2400 字。回锚轮本来就少（每 8 轮一次），
# 这一轮宁可多带，把定案事实尽量带全；仍封顶，取舍规矩同常规块。
REANCHOR_MAX_ITEMS = 30
REANCHOR_MAX_CHARS = 2400

# 回锚节奏：每 8 轮一次（第 8、16、24…轮）。依据：历史压缩的触发点
# 是 token 上限、留最近 5 轮原文——也就是说聊过 5 轮以上，开头定案
# 的原文就可能已经被压进摘要；8 轮取在「原文大概率已不在场、但对
# 话还没长到回锚本身变成噪音」的位置。每 8 轮周期性重申而不是只
# 在第 8 轮一次，是因为对话可以远长于 16 轮，松动是持续发生的。
REANCHOR_EVERY_TURNS = 8

ANCHOR_TITLE = "【" + "已确认事实" + "】"
REANCHOR_TITLE = "【回锚·" + "已确认事实" + "】"

_ANCHOR_HEADER = (
    ANCHOR_TITLE + "以下是已经由 BOOS 亲口确认过的定案事实，不是待议事项：\n"
    "回答时直接当事实用，不要说「不确定」「没确认过」，也不要向 BOOS 重复确认这些事。"
)

_REANCHOR_HEADER = (
    REANCHOR_TITLE + "对话已经进行了多轮，重申以下已确认事实与当前立场：\n"
    "这些都是 BOOS 亲口确认过的定案，聊得再久也不动摇；不要因为后文没再提起，"
    "就当它们没确认过、改口说不确定。"
)


def is_anchor_fact(row):
    """一行活记忆算不算「已确认事实」（现行口径，见模块头注）。
    
    row 是 _all_live_memories() 形状的字典（带 metadata/slot），
    检索结果形状（只有 metadata）也能判。
    """
    if not row:
        return False
    meta = row.get("metadata") or {}
    if _provenance._is_core_memory(meta):
        return True
    if meta.get("is_rule") or meta.get("curated"):
        return True
    slot = row.get("slot") or meta.get("slot")
    return slot in _provenance.KNOWN_SLOTS


def _fact_order_key(row):
    """取舍/排序键：规则最前，身份槽位次之，其余已确认随后；
    同层里 importance 高的在前，再按 write_seq 新的在前。
    
    规则排最前和 _memory_block 的既有排序一致（先套规则再用事实）；
    身份槽位（名字/SOUL/电脑配置这类「我是谁」的定案）是第 ⑦ 组
    翻车的正是这一类，排在普通已确认之前保住。
    """
    meta = row.get("metadata") or {}
    if meta.get("is_rule"):
        tier = 0
    elif (row.get("slot") or meta.get("slot")) in _provenance.KNOWN_SLOTS:
        tier = 1
    else:
        tier = 2
    importance = row.get("importance")
    if not isinstance(importance, (int, float)) or isinstance(importance, bool):
        importance = 0
    write_seq = row.get("write_seq")
    if not isinstance(write_seq, (int, float)) or isinstance(write_seq, bool):
        write_seq = 0
    return (tier, -importance, -write_seq)


def select_anchor_facts(rows, *, max_items, max_chars):
    """从活行里挑已确认事实：过滤 → 去重 → 排序 → 双封顶。
    
    取舍规矩（超限时按这个顺序丢，写明不许临时改）：
    1. 先按 _fact_order_key 排序（规则 > 身份槽位 > 其余已确认；
    同层 importance 高者优先、再新的优先）；
    2. 按顺序取，条数到 max_items 或累计字数到 max_chars 就停，
    后面的丢——丢的永远是排序靠后的，不随机、不按检索相关性；
    3. 第一条即使单独超字数上限也保留（截断到 max_chars 加省略号），
    一条都不带比带一条截断的更糟（同 _memory_block 的口径）。
    敏感内容（密码/卡号类）不进每轮注入：这类内容本来就不该当定案
    事实反复出现在提示词里，照 get_all 的既有口径排除。
    """
    picked = []
    seen = set()
    for row in rows or []:
        if not is_anchor_fact(row):
            continue
        text = (row.get("text") or "").strip()
        if not text or _contains_sensitive(text):
            continue
        key = _normalize_memory_text(text)
        if key in seen:
            continue
        seen.add(key)
        picked.append(row)
    picked.sort(key=_fact_order_key)
    out = []
    total_chars = 0
    for row in picked:
        if len(out) >= max_items:
            break
        text = (row.get("text") or "").strip()
        if out and total_chars + len(text) > max_chars:
            break
        if not out and len(text) > max_chars:
            text = text[:max_chars] + "…"
            row = dict(row, text=text)
        out.append(row)
        total_chars += len(text)
    return out


def build_anchor_block(facts, *, reanchor=False):
    """把挑好的已确认事实拼成注入块。没有事实时返回空字符串。
    
    每条都带「（已确认）」前缀标注——块头有定案说明还不够，单条
    标注让模型在长提示里扫到任意一条都知道它的身份。
    """
    if not facts:
        return ""
    header = _REANCHOR_HEADER if reanchor else _ANCHOR_HEADER
    lines = [header]
    for row in facts:
        text = (row.get("text") or "").strip()
        if text:
            lines.append(f"- （已确认）{text}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def should_reanchor(turn_number):
    """第几轮（1 起数，本轮是第 turn_number 轮用户发言）用回锚加强版。
    
    第 REANCHOR_EVERY_TURNS 的整数倍轮（8、16、24…）回锚，其余轮
    用常规块。回锚是替换不是叠加：同一轮只出现一个已确认区。
    """
    if not isinstance(turn_number, int) or isinstance(turn_number, bool):
        return False
    return (
        turn_number >= REANCHOR_EVERY_TURNS
        and turn_number % REANCHOR_EVERY_TURNS == 0
    )


def turn_number_from_messages(messages):
    """由本轮之前的历史算当前是第几轮：历史里的 user 条数 + 1。"""
    count = 0
    for m in messages or []:
        if (m or {}).get("role") == "user":
            count += 1
    return count + 1


def anchor_block_for_turn(rows, turn_number):
    """一站式：给本轮的已确认注入块（回锚轮自动用加强版与大容量）。"""
    if should_reanchor(turn_number):
        facts = select_anchor_facts(
            rows, max_items=REANCHOR_MAX_ITEMS, max_chars=REANCHOR_MAX_CHARS
        )
        return build_anchor_block(facts, reanchor=True)
    facts = select_anchor_facts(
        rows, max_items=ANCHOR_MAX_ITEMS, max_chars=ANCHOR_MAX_CHARS
    )
    return build_anchor_block(facts, reanchor=False)
