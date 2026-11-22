# memory/decay.py（连续衰减：纯函数与常量）
# 从 memory/mem0_store.py 拆出来（第二步衰减，2026-11-13，BOOS 定稿
# 《十二记忆整理方案》第二版；2026-11-13 晚最终方案只打标、不自动
# 压摘要、不自动删，见下面 DECAY 常量注释）。
# 下面全是纯函数（只算、不读库、不写库），调用方见 Mem0Store 里
# 的 reinforce_memories / _record_decay_occurrence / sweep_monthly /
# delete_due_memories。行为与 fix/memory-decay 分支（e6d82dc）一致，
# 合并时以那边已测过的版本为准。

import json
import re
from datetime import datetime, timezone

from memory.timeutil import _parse_created_at

# 第二步衰减（连续衰减，2026-11-13，BOOS 定稿《十二记忆整理方案》第二版）。
# 每条记忆自带权重：越久没用、越不重要，自己往下掉；检索里真正用上
# 的才强化一次。三个工作台挑出来的坑，先定死再写代码：
#   1. 衰减绝不能跟检索阈值 0.4 打架——资格只看原始语义分（≥0.4 才进），
#      权重只管进围之后的排序，再低也不会让记忆在观察期内搜不到
#      （见 search：先按原始分过滤，再按“分 × 权重”排）。
#   2. 顺序：先标冲突（旧的标“被取代”），再做检索强化——去重裁决
#      判了“是同一件事”之后，才记一笔“又出现一次”（见
#      _record_decay_occurrence 的调用点，都在去重返回之后）。
#   3. 月扫只标记、不直接删（见 sweep_monthly）。
# 沉底规矩（2026-11-13 晚最终方案，BOOS 定，取代 20:43 的“30 天压摘要、
# 半年删”）：只打标、不自动压摘要、不自动删——
#   30 天没人找 / 没用上 → 打标 dormant（decay_state 从 verbatim 变成
#   dormant，原话和向量文本都不动；被用上就回到原话态、重新强化）；
#   满半年（180 天，从最后一次用上算起）还没人找 → 再打标 decay_delete_due，
#   只做标记，等 BOOS 亲口说删才删；
#   真删是另一件显式的事（见 delete_due_memories：要 BOOS 点头，删前有备份）；
#   核心记忆钉住不动，不走这条（权重恒为 1，月扫直接跳过）。
# 半衰期是指数衰减的“多久掉一半”：稳定事实 180 天（慢）、一次性经历
# 14 天（快）。30 天没用的一次性经历权重 ≈0.23，正好掉到沉底线
# （DECAY_DORMANT_WEIGHT = 0.30）下面；稳定事实 30 天才掉到 ≈0.89，
# 不会被标沉底。权重下限 0.10：再久不用也不会归零，排位垫底而已。
# 晋升：一次性经历在不同天、不同场合出现 3 次就晋升成稳定事实
# （DECAY_PROMOTION_DAYS = 3）；一次聊天里说 3 遍不算——出现次数
# 按“不同日期”数（decay_seen_days），同一天重复提不加数。
DECAY_KIND_STABLE = "stable"
DECAY_KIND_EPISODIC = "episodic"
DECAY_HALF_LIFE_DAYS = {DECAY_KIND_STABLE: 180, DECAY_KIND_EPISODIC: 14}
DECAY_WEIGHT_FLOOR = 0.10
DECAY_DORMANT_WEIGHT = 0.30
DECAY_DORMANT_IDLE_DAYS = 30
DECAY_DELETE_IDLE_DAYS = 180
DECAY_PROMOTION_DAYS = 3
DECAY_PROMOTION_SESSIONS = 3  # 晋升要的会话数：和天数一样，3 个不同会话
DECAY_STATE_VERBATIM = "verbatim"
DECAY_STATE_DORMANT = "dormant"
DECAY_STATE_DELETE_DUE = "decay_delete_due"
DECAY_DELETE_BACKUP_PREFIX = "decay-delete-backup-"
# 未确认权重封顶（补丁 4，第二层）：没经过 BOOS 确认的记忆权重
# 升不到最高——错的也爬不上去。初始值 0.80，以后按回测调（BOOS 定：
# 衰减参数有了真实数据再调，封顶值也是参数，先不动它）。
UNCONFIRMED_WEIGHT_CAP = 0.80


def _decay_now_dt(now=None):
    """把 now 规整成带时区的 datetime：不传就是现在（UTC）；
    传 datetime 就用它（没时区的当 UTC）；传 ISO 时间串就解析，
    解析不出也当现在——衰减算的是“多久没用”，拿不准就按“刚用过”
    算（宁可不沉，不错沉）。"""
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now
    if isinstance(now, str) and now.strip():
        parsed = _parse_created_at(now)
        if parsed is not None:
            return parsed
    return datetime.now(timezone.utc)


def _decay_day(value):
    """取一个时间（datetime / ISO 串 / None）的 UTC 日期 YYYY-MM-DD；
    取不出返回 None（不猜今天——拿不准就当这一天没数过）。
    出现次数按“不同日期”数，就靠它。"""
    if isinstance(value, datetime):
        parsed = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return parsed.date().isoformat()
    if isinstance(value, str) and value.strip():
        parsed = _parse_created_at(value)
        if parsed is not None:
            return parsed.date().isoformat()
    return None


def _decay_iso(moment):
    """把 datetime 格式化成 _now_iso 同款的 ISO 串（6 位微秒，带 Z）。"""
    parsed = _decay_now_dt(moment)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def classify_decay_kind(source, is_core, is_rule):
    """新记忆写进库时定它的衰减档位（stable / episodic）。

    核心（名字、SOUL、决定、钱的规矩、安全权限、项目基础决策）和
    规则记 stable；BOOS 亲口说的稳定事实/偏好/决定也记 stable；
    模型提取的记 episodic（一次性经历多半在这里）。不猜关键词——
    档位只看来源和核心标记，猜错档位顶多是沉得快一点或慢一点，
    不会把记忆弄丢（月扫只压摘要、不直接删，删还要再等半年）。"""
    if is_core or is_rule:
        return DECAY_KIND_STABLE
    if source == "user":
        return DECAY_KIND_STABLE
    return DECAY_KIND_EPISODIC


def _decay_idle_days(last_used_at, now=None):
    """从最后一次用上到 now，闲了多少天（小数天数，负数按 0 算）；
    last_used_at 没有（没时间的老记忆）返回 None——算不出闲了多久，
    就当它没闲过（宁可不沉，不错沉）。"""
    if not isinstance(last_used_at, str) or not last_used_at:
        return None
    last = _parse_created_at(last_used_at)
    if last is None:
        return None
    delta = (_decay_now_dt(now) - last).total_seconds() / 86400.0
    return max(0.0, delta)


def decay_weight(kind, last_used_at, now=None, is_core=False):
    """一条记忆现在的权重（0.10–1.0）：指数衰减，
    weight = 0.5 ** (闲置天数 / 半衰期)，下限 0.10。
    核心记忆恒为 1.0（钉住不动，衰减是零，不是慢）；没时间的老记忆
    也按 1.0 算（算不出闲了多久，就当它没闲过）。
    权重只管排位、不管对错，更不管“能不能搜到”——检索资格只看
    原始语义分（见 search），权重再低也不会让记忆搜不到（坑 1）。"""
    if is_core:
        return 1.0
    idle = _decay_idle_days(last_used_at, now)
    if idle is None:
        return 1.0
    half_life = DECAY_HALF_LIFE_DAYS.get(kind, DECAY_HALF_LIFE_DAYS[DECAY_KIND_EPISODIC])
    weight = 0.5 ** (idle / half_life)
    return max(DECAY_WEIGHT_FLOOR, min(1.0, weight))


def effective_weight(kind, last_used_at, now=None, is_core=False,
                     user_confirmed=False):
    """一条记忆实际用的权重：衰减权重再过一道未确认封顶。

    没经过 BOOS 确认的记忆（user_confirmed=False），权重最高只到
    UNCONFIRMED_WEIGHT_CAP（0.80），升不到 1.0——错的也爬不上去
    （补丁 4 第二层）。已确认的、核心的照常满格。权重只管排位，
    不管“能不能搜到”，和 decay_weight 一个规矩。"""
    weight = decay_weight(kind, last_used_at, now, is_core=is_core)
    if is_core or user_confirmed:
        return weight
    return min(weight, UNCONFIRMED_WEIGHT_CAP)


def promotion_reached(seen_days):
    """一次性经历在不同天出现够 3 次没有——够了就晋升成稳定事实。
    seen_days 不是整数、或没数过时按没够算。"""
    return (
        isinstance(seen_days, int)
        and not isinstance(seen_days, bool)
        and seen_days >= DECAY_PROMOTION_DAYS
    )


def decay_should_go_dormant(kind, last_used_at, now=None, is_core=False,
                             is_curated=False):
    """这条记忆该标沉底了吗：30 天没人找，且权重已经掉到 0.30 下面。
    核心记忆永远不标；精选清单（curated）的记忆也不沉底（2026-11-20，
    最重要的记忆常年待在常用层）；没时间的老记忆不标（闲了多久都算
    不出来）。只判“该不该打标”，标了也只是打标——原话和文本都不动
    （见 sweep_monthly）。"""
    if is_core or is_curated:
        return False
    idle = _decay_idle_days(last_used_at, now)
    if idle is None or idle < DECAY_DORMANT_IDLE_DAYS:
        return False
    return decay_weight(kind, last_used_at, now, is_core=False) < DECAY_DORMANT_WEIGHT


def decay_state(kind, last_used_at, now=None, is_core=False, is_curated=False):
    """这条记忆现在该是什么状态（2026-11-22 接线用）：
    decay_delete_due（180 天没人找）> dormant（30 天没人找且权重 < 0.30）
    > verbatim（原话态）。只判状态、不删不改——核心记忆和精选记忆
    永远是 verbatim（见 decay_should_delete / decay_should_go_dormant）。
    180 天的先判：它同时也满足沉底条件，待删优先。"""
    if decay_should_delete(last_used_at, now=now, is_core=is_core):
        return DECAY_STATE_DELETE_DUE
    if decay_should_go_dormant(kind, last_used_at, now=now, is_core=is_core,
                                is_curated=is_curated):
        return DECAY_STATE_DORMANT
    return DECAY_STATE_VERBATIM


def decay_should_delete(last_used_at, now=None, is_core=False):
    """这条记忆该标待删了吗：从最后一次用上算起满 180 天还没人找。
    核心记忆永远不标；没时间的老记忆不标。只判“该不该打待删标记”，
    不管删——删是 delete_due_memories 的事，要 BOOS 点头才删。"""
    if is_core:
        return False
    idle = _decay_idle_days(last_used_at, now)
    if idle is None:
        return False
    return idle >= DECAY_DELETE_IDLE_DAYS


# ---------- 出现集合：不同天、不同会话的计数（2026-11-19） ----------
# 晋升“一次性经历在不同天、不同会话出现 3 次”要数得清：一次聊天里
# 说 3 遍不算、三次聊天各说一遍才算，跨午夜但同一轮也不算——只数天数
# 分不出来，得把“出现过哪几天、出现过哪几轮”都记下来。
# 存法：Chroma 只接受标量 metadata 值，dict/list 直接存会炸（和
# evidence 一个做法），所以两个集合都以 JSON 字符串存（见
# _dump_seen_list），读回来再解析（见 _parse_seen_list）。
# 下面全是纯函数（只算、不读库、不写库），和本文件其他函数一样。


def _parse_seen_list(value):
    """把出现集合读回列表：库里的 JSON 字符串、规整后的列表都认；
    别的、坏的都按空算。只认非空字符串，去重后按原序返回。"""
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item and item not in out:
                out.append(item)
        return out
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return []
        if isinstance(parsed, list):
            return _parse_seen_list(parsed)
    return []


def _dump_seen_list(values):
    """出现集合写回 metadata 的写法：JSON 字符串。去重、排序后写，
    读回来和写进去的样子一致。"""
    seen = []
    for item in values or []:
        if isinstance(item, str) and item and item not in seen:
            seen.append(item)
    return json.dumps(sorted(seen), ensure_ascii=False)


def record_seen_occurrence(seen_dates, seen_sessions, today, turn_id):
    """一次“出现”记进两个集合，返回 (新日期列表, 新会话列表, 日期是新加的)。

    计数规矩（2026-11-19，BOOS 定，写代码时分清）：
    - 同一轮里说几遍都只算一次：turn_id 已经在会话集合里，这次什么都不加；
    - 新日期且新会话才 +1：会话是新的，才把 turn_id 加进会话集合；
      日期是新的，才把今天加进日期集合；
    - 跨午夜但同一轮不算：日期变了、会话没变——会话集合里已经有它，
      日期也不加（每个会话最多贡献 1 次）。
    turn_id 没传（老路子、没轮次信息）时只按日期数：今天不在日期
    集合里就加上，会话集合不动——老记忆、老调用方照常能晋升。
    today 不是 YYYY-MM-DD 时什么都不加（不猜今天）。"""
    dates = _parse_seen_list(seen_dates)
    sessions = _parse_seen_list(seen_sessions)
    if not isinstance(today, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", today):
        return dates, sessions, False
    if isinstance(turn_id, str) and turn_id:
        if turn_id in sessions:
            return dates, sessions, False
        sessions.append(turn_id)
        if today not in dates:
            dates.append(today)
            return sorted(dates), sorted(sessions), True
        return sorted(dates), sorted(sessions), False
    if today not in dates:
        dates.append(today)
        return sorted(dates), sorted(sessions), True
    return sorted(dates), sorted(sessions), False


def should_promote(seen_dates, seen_sessions, is_core=False, tone=None):
    """这条记忆该晋升成稳定事实了吗：出现过的自然日 ≥3 且出现过的
    会话 ≥3，且不是核心记忆（核心记忆本来就是稳定事实，不用晋升）。
    seen_dates / seen_sessions 传集合、列表或库里的 JSON 字符串都行。
    tone（补丁 3）：玩笑 / 测试 / 假设打标的记忆不晋升——打标的记忆
    照常存、照常能搜，但不算进晋升次数；没打标（None）按认真算。
    只判“该不该晋升”，晋升本身（改 kind、记 PROMOTE 事件）是调用方
    的事（见 Mem0Store._decay_touch）。"""
    if is_core:
        return False
    if tone in ("joke", "test", "hypothesis"):
        return False
    dates = _parse_seen_list(seen_dates)
    sessions = _parse_seen_list(seen_sessions)
    return len(dates) >= DECAY_PROMOTION_DAYS and len(sessions) >= DECAY_PROMOTION_SESSIONS
