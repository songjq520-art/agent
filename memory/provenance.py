# memory/provenance.py（来源与可信度：元数据、来源分层、冲突裁决）
# 从 memory/mem0_store.py 拆出来，行为不变。

import json
import re
from datetime import datetime, timezone
from memory.decay import DECAY_KIND_EPISODIC, DECAY_KIND_STABLE, DECAY_STATE_DORMANT, DECAY_STATE_VERBATIM, _dump_seen_list, _parse_seen_list, classify_decay_kind
from memory.money_slots import KNOWN_SLOTS, _MONEY_UNSET, _flatten_money, _normalize_money_metadata
from memory.timeutil import _next_write_seq, _now_iso, _parse_created_at

# ---------- 来源元数据（provenance） ----------
# 每条记忆存进 mem0 时带的 metadata（见文件头“来源”一节）：
#   source：记忆从哪来。"user"——BOOS 亲口说的/让记住的/亲口纠正的；
#           "inferred"——DeepSeek 从对话里提取的（用户明确说过的话，
#           但 BOOS 没逐条确认过那个写法）。
#   created_at：记忆形成的时间（UTC 的 ISO 时间串，微秒级、严格单调递增，
#           固定 6 位小数、带 Z——见 _now_iso）。
#   write_seq：同一批写进去的序号（进程内严格递增，和 created_at 一起
#           排新旧，见 _memory_order_key）。
#   turn_id：这条记忆是哪一轮对话里写进去的（2026-11-19，save_turn /
#           add_turn 传进来）；没轮次的老路子没有这个字段。
#   created_at_source：这个时间是哪来的——"written"（这次写进去的）、
#           "backfilled"（老记忆后补的，见 backfill_timestamps）、
#           "migrated"（迁移时沿用的旧日期，见 migrate_from_v1）。
#   confidence：可信度 0–1。BOOS 亲口说的记 1.0；提取的记 0.7——
#           不是因为提取得不准，而是因为“模型提的”本身就比“用户
#           亲口确认的”低一档；模型用它来判断这条记忆有多可信。
#   user_confirmed：BOOS 是否逐条确认过。只在 source="user" 时为 True。
#           注意：提取的记忆哪怕存下来了，也绝不会因为“存下来了”就
#           自动变成已确认——source 还是 "inferred"，user_confirmed
#           还是 False；只有 BOOS 亲口说/纠正/让记住，才是已确认。
# 没有 metadata 的旧记忆（2026-11-09 之前存的）：取回时一律按
# "unconfirmed" 处理（见 _normalize_metadata），绝不默认已确认。
#   衰减字段（第二步，2026-11-13）：decay_kind（stable 稳定事实 /
#           episodic 一次性经历）、importance（high/normal/low，重要性
#           和用得多不多分开算）、last_used_at（最后一次用上的时间，
#           写时就是现在）、decay_seen_days（不同天被提到的天数，写时
#           记 1）、decay_last_seen_day（上次被提到的日期 YYYY-MM-DD）、
#           decay_seen_dates（出现过的自然日集合，JSON 字符串，写时记
#           今天）、decay_seen_sessions（出现过的轮次 turn_id 集合，
#           JSON 字符串，写时记这轮）、
#           decay_state（verbatim 原话 / dormant 沉底——30 天没人找
#           只打标，原话和向量文本都不动）、
#           decay_delete_due（月扫标的“满半年没人找、等 BOOS 亲口说删”，
#           只做标记，不直接删）。Chroma 只接受标量值，所以这些全是字符串 /
#           数字 / 布尔，没有字典和列表。

SOURCE_USER = "user"

SOURCE_INFERRED = "inferred"

SOURCE_UNCONFIRMED = "unconfirmed"

CONFIDENCE_USER = 1.0

CONFIDENCE_INFERRED = 0.7

# 可信度分级（覆盖规则用，见 _resolve_conflict）：
#   3 = BOOS 亲口确认的（source=user 且 user_confirmed=True）；
#   2 = 来源是 user、但没逐条确认（旧数据损坏/写坏了才有，正常写不到）；
#   1 = 模型从对话里提取的（inferred）；
#   0 = 没来源信息的旧记忆（unconfirmed）。
# 时间戳决定新旧，可信度决定有没有资格覆盖：级别低的新记忆不能
# 直接覆盖级别高的旧事实。
CREDIBILITY_CONFIRMED = 3

CREDIBILITY_USER = 2

CREDIBILITY_INFERRED = 1

CREDIBILITY_UNCONFIRMED = 0

# created_at 的合法来源（见 _provenance 的 created_at_source 参数）。
CREATED_AT_SOURCES = ("written", "backfilled", "migrated")

def _provenance(source, confidence, user_confirmed, created_at=None, slot=None,
                created_at_source="written", is_rule=False, money=_MONEY_UNSET,
                verification_status=None, requires_confirmation=False,
                expires_at=None, conflict_for_review=False, needs_review=False,
                turn_id=None, tone=None, tone_evidence=None, tone_source=None,
                voided=False):
    """拼一份来源元数据（存记忆时传给 mem0.add 的 metadata 参数）。

    source 不是 "user" 时一律按 "inferred" 存（调用方传错也不让它
    进库）；user_confirmed 只在 source="user" 时才可能是 True。
    slot（Plan C）：身份/称呼类记忆的槽位名（agent_name/user_name/
    soul），只认 KNOWN_SLOTS 里的，其他值不进库——同槽位只留一条，
    新值盖旧值时靠它找旧记忆。
    is_rule（第 3 批 42 号）：纠正后总结的通用规则（比如“定投只在交易日
    扣款”）才记 True——检索时优先捞出来，回答同类问题先套用规则再算。
    created_at 不传时取新鲜的单调时间（_now_iso）；传了就用传的
    （迁移沿用旧日期、纠错回滚恢复旧时间时用）。created_at_source
    说明这个时间是哪来的，只认 CREATED_AT_SOURCES 里的三个值，
    传错了按 "written" 算——时间来源不能靠猜，必须是明确的。
    money（27 号）：金额结构化字段（_extract_money_metadata 的结果）；
    没传（_MONEY_UNSET）就不碰 money 字段；传 None 是显式清掉
    （纠正时新说法没金额，旧的金额字段不能留着）；传了字典就校验后
    展平成四个标量字段存（见 _flatten_money，Chroma 不接受 dict 值）。
    verification_status / requires_confirmation / expires_at（判断 b 号）：
    动态数字事实的核实状态——"verified"（两个独立来源对得上，或一家
    权威来源）或 "unverified"（没印证）；未核实的可信度封顶 0.35，
    须确认（requires_confirmation=True）才当稳定事实用；expires_at
    是这条数的有效期（行情 24 小时、天气 12 小时），过期了要重查，
    不能再当事实引用。只认 "verified"/"unverified"，别的不要。
    conflict_for_review / needs_review（判断 a 号）：按意思去重的裁决
    标记——conflict_for_review 是槽位值冲突（同一件事、说法对着干，
    比如喜欢吃鸡杂面 vs 对鸡杂面过敏），留着让 BOOS 看；
    needs_review 是模型裁决把握不够（confidence <0.80 或判失败），
    等兜底清理时再看。都是 True 才写进 metadata。
    每次调都新拿一个 write_seq（同一批里的先后靠它分）。
    turn_id（2026-11-19）：这轮对话的 id，传了就写进 metadata（没传
    不记）；出现集合写时就算一次——今天进日期集合、这轮进会话集合。
    """
    source = SOURCE_USER if source == SOURCE_USER else SOURCE_INFERRED
    if created_at_source not in CREATED_AT_SOURCES:
        created_at_source = "written"
    metadata = {
        "source": source,
        "created_at": created_at or _now_iso(),
        "created_at_source": created_at_source,
        "write_seq": _next_write_seq(),
        "confidence": confidence,
        "user_confirmed": bool(user_confirmed) and source == SOURCE_USER,
    }
    # turn_id（2026-11-19）：这条记忆是哪一轮对话里写进去的——
    # save_turn/add_turn 传进来，账本 add 事件和这里都记同一份
    # （见 mem0_store._ledger_record_add / _ledger_extraction_adds）。
    # 没传就不记（老路子、手动存的没有轮次）。
    if isinstance(turn_id, str) and turn_id:
        metadata["turn_id"] = turn_id
    # 第二步衰减：新记忆自带衰减字段（见 memory/decay.py 的 DECAY 常量注释）。
    # kind 按“核心 / BOOS 亲口说的 / 模型提取的”分，不猜关键词——
    # 核心（已确认、带槽位、规则）记 stable；BOOS 亲口说的稳定事实/
    # 偏好 / 决定也记 stable；模型提取的记 episodic（一次性经历多半
    # 在这里），反复出现 3 次不同天再晋升。last_used_at 记这次写的
    # 时间（写就是一次“用上”；迁移/恢复沿用旧 created_at 时，last_used
    # 还是现在——刚写回来的记忆不算“很久没用”）。
    _decay_core = (
        (source == SOURCE_USER and bool(user_confirmed))
        or slot in KNOWN_SLOTS
        or bool(is_rule)
    )
    metadata["decay_kind"] = classify_decay_kind(
        source, _decay_core, bool(is_rule)
    )
    metadata["importance"] = "high" if _decay_core else "normal"
    metadata["last_used_at"] = _now_iso()
    metadata["decay_seen_days"] = 1
    _seen_day = datetime.now(timezone.utc).date().isoformat()
    metadata["decay_last_seen_day"] = _seen_day
    # 出现集合（2026-11-19）：写时就算出现过一次——今天进日期集合，
    # 这轮（有就记）进会话集合。存 JSON 字符串（Chroma 不接受 list 值）。
    metadata["decay_seen_dates"] = _dump_seen_list([_seen_day])
    metadata["decay_seen_sessions"] = _dump_seen_list(
        [turn_id] if isinstance(turn_id, str) and turn_id else []
    )
    metadata["decay_state"] = DECAY_STATE_VERBATIM
    if slot in KNOWN_SLOTS:
        metadata["slot"] = slot
    if is_rule:
        metadata["is_rule"] = True
    if verification_status in ("verified", "unverified"):
        metadata["verification_status"] = verification_status
    if requires_confirmation:
        metadata["requires_confirmation"] = True
    if isinstance(expires_at, str) and expires_at:
        metadata["expires_at"] = expires_at
    if conflict_for_review:
        metadata["conflict_for_review"] = True
    if needs_review:
        metadata["needs_review"] = True
    # 语气（补丁 3，两段式判语气）：tone 只认 serious（认真）/ joke
    # （玩笑）/ test（测试）/ hypothesis（假设）四个值；没打标的不写
    # （读回来默认认真）。tone_source 说明是谁判的：local（本地规则）
    # 还是 model（大模型后台判）；tone_evidence 是判的证据（命中了哪句
    # 话），大模型判的必须带证据。voided（补丁 3 撤回）：BOOS 后来说
    # “那句是开玩笑的”，这条连带升上去的一起作废，只打标、不删原话。
    if tone in ("serious", "joke", "test", "hypothesis"):
        metadata["tone"] = tone
        if isinstance(tone_evidence, str) and tone_evidence:
            metadata["tone_evidence"] = tone_evidence
        if tone_source in ("local", "model"):
            metadata["tone_source"] = tone_source
    if voided:
        metadata["voided"] = True
    if money is not _MONEY_UNSET:
        # None 也存：把旧的金额字段清掉（纠正时新说法没金额就清掉，
        # 不然“每天投 5 块”改成“我不投了”，5 块的字段还贴在上面）。
        # 展平成四个标量字段存（见 _flatten_money）：Chroma 不接受
        # dict 值的 metadata，存 dict 会炸掉整条写入。
        metadata.update(_flatten_money(money))
    # 预整理（2026-11-20）：最重要的记忆进精选清单（curated 标记）——
    # 条件照搬十一的 MEMORY.md（见 _should_be_curated）：核心 / BOOS
    # 亲口确认 / 规则 / 晋升成稳定事实；玩笑、测试、假设、已作废的不进。
    # 写的时候满足 (a)/(b)/(c) 就打上；(d) 晋升那条在 _decay_touch 里打。
    if _should_be_curated(metadata):
        metadata["curated"] = True
    return metadata

def _normalize_metadata(raw, item=None):
    """把 mem0 返回的 metadata 规整成统一形状（检索时用）。

    规整规则（安全降级，宁可信度低、不可误认）：
    - raw 不是字典、或缺 source、或 source 不是 user/inferred，
      一律按 "unconfirmed" 处理（旧记忆、新旧版本混用、损坏数据
      都走这条路）；
    - user_confirmed 只有明确是 True（布尔值）才算确认，写成
      "yes"/1 这类非布尔值不算——确认必须是明确的，不能靠猜；
    - confidence 不是 0–1 之间的数字时记成 None（不知道就是不知道）；
    - created_at 不是字符串时记成 None；
    - write_seq 不是整数时记成 None；created_at_source 不是三个合法值
      时记成 None；pending_confirmation 只有明确是 True（布尔值）
      才算待确认——这三样也一样，不能靠猜；
    - slot 不是 KNOWN_SLOTS 里的已知槽位时记成 None（旧记忆、新旧
      版本混用、损坏数据都走这条路——槽位只认这三个，别的都不认）。
    - money（27 号）：金额结构化字段，形状不对就记 None（见
      _normalize_money_metadata），不猜。
    created_at 的第二来源（33 号）：mem0 2.2.1 读回来时把 created_at
    提到条目的顶层（get/get_all/search 的返回里，created_at 是
    MemoryItem 的顶层字段），同时从返回的 metadata 里剔掉——所以
    只看 metadata 永远找不到时间，面板上 14 条全是“未知时间”。
    根因不在写的那头：写的时候 _provenance 每条都带了显式时间，
    mem0 存的时候也留着（add 时缺了它自己还会补一个）。修法：调
    用方把整条 item 也传进来（item 参数），metadata 里没有时再到
    顶层找——metadata 里的优先，顶层是兜底。没传 item 时只看 metadata，
    行为和以前一样。
    返回：{"source", "created_at", "created_at_source", "write_seq",
           "turn_id", "confidence", "user_confirmed", "confirmed", "slot",
           "pending_confirmation", "is_rule", "money",
           "decay_kind", "importance", "last_used_at", "decay_seen_days",
           "decay_last_seen_day", "decay_seen_dates", "decay_seen_sessions",
           "decay_state", "decay_delete_due",
           "superseded", "curated"}，其中 confirmed
    就是 user_confirmed（取了个短名字，方便调用方判断“能不能当作事实用”）。
    衰减字段（第二步）：只认合法值，别的按缺失处理；老记忆没有这些
    字段时走派生默认值（见 return 前的派生段）——不猜、不抬可信度。
    """
    source = SOURCE_UNCONFIRMED
    created_at = None
    created_at_source = None
    write_seq = None
    turn_id = None
    confidence = None
    user_confirmed = False
    slot = None
    pending_confirmation = False
    is_rule = False
    money = None
    verification_status = None
    requires_confirmation = False
    expires_at = None
    conflict_for_review = False
    needs_review = False
    evidence = None
    # 语气（补丁 3）：没打标的默认认真；只认四个值，别的按认真算。
    tone = "serious"
    tone_evidence = None
    tone_source = None
    voided = False
    voided_at = None
    curated = False
    if isinstance(raw, dict):
        raw_source = raw.get("source")
        if raw_source in (SOURCE_USER, SOURCE_INFERRED):
            source = raw_source
        raw_created = raw.get("created_at")
        if isinstance(raw_created, str) and raw_created:
            created_at = raw_created
        raw_created_source = raw.get("created_at_source")
        if raw_created_source in CREATED_AT_SOURCES:
            created_at_source = raw_created_source
        raw_seq = raw.get("write_seq")
        if isinstance(raw_seq, int) and not isinstance(raw_seq, bool):
            write_seq = raw_seq
        raw_turn = raw.get("turn_id")
        if isinstance(raw_turn, str) and raw_turn:
            turn_id = raw_turn
        raw_conf = raw.get("confidence")
        if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool):
            if 0 <= raw_conf <= 1:
                confidence = float(raw_conf)
        user_confirmed = raw.get("user_confirmed") is True and source == SOURCE_USER
        raw_slot = raw.get("slot")
        if raw_slot in KNOWN_SLOTS:
            slot = raw_slot
        pending_confirmation = raw.get("pending_confirmation") is True
        # is_rule 只有明确是 True（布尔值）才算规则，写成 "yes"/1 不算——
        # 规则要在检索时优先捞出来，不能靠猜。
        is_rule = raw.get("is_rule") is True
        # 27 号：money 字典形（旧数据/代码里传的）或扁平形（真正存在
        # Chroma 里的四个 money_* 字段）都认，拼回字典形给调用方。
        money = _normalize_money_metadata(raw.get("money"))
        if money is None:
            money = _normalize_money_metadata(raw)
        # 判断 (b) 号：动态数字事实的核实状态——只认
        # "verified"/"unverified"，别的不要；requires_confirmation
        # 只有明确是 True（布尔值）才算；expires_at 只认非空字符串。
        raw_verification = raw.get("verification_status")
        if raw_verification in ("verified", "unverified"):
            verification_status = raw_verification
        requires_confirmation = raw.get("requires_confirmation") is True
        raw_expires = raw.get("expires_at")
        if isinstance(raw_expires, str) and raw_expires:
            expires_at = raw_expires
        # 判断 (a) 号：裁决标记——只有明确是 True（布尔值）才算。
        conflict_for_review = raw.get("conflict_for_review") is True
        needs_review = raw.get("needs_review") is True
        # 判断 (a) 号：重复证据（JSON 字符串，原样透传，调用方自己解析）。
        raw_evidence = raw.get("evidence")
        if isinstance(raw_evidence, str) and raw_evidence:
            evidence = raw_evidence
        # 语气（补丁 3）：只认四个值；证据和来源只认非空字符串 /
        # local / model；作废标记只有明确是 True（布尔值）才算。
        raw_tone = raw.get("tone")
        if raw_tone in ("serious", "joke", "test", "hypothesis"):
            tone = raw_tone
        raw_tone_evidence = raw.get("tone_evidence")
        if isinstance(raw_tone_evidence, str) and raw_tone_evidence:
            tone_evidence = raw_tone_evidence
        raw_tone_source = raw.get("tone_source")
        if raw_tone_source in ("local", "model"):
            tone_source = raw_tone_source
        voided = raw.get("voided") is True
        # 预整理（2026-11-20）：curated 只有明确是 True（布尔值）才算，
        # 跟 is_rule、pending_confirmation 一个做法。
        curated = raw.get("curated") is True
        raw_voided_at = raw.get("voided_at")
        if isinstance(raw_voided_at, str) and raw_voided_at:
            voided_at = raw_voided_at
    # 第二步衰减：decay_* 字段只认合法值（字符串/数字/布尔要对得上号），
    # 别的按缺失处理。派生默认值（老记忆没有衰减字段时）：
    #   - last_used_at 没有就用 created_at（写时就是最后一次用上）；
    #     两个都没有的老记忆不衰减（记 None，decay_weight 按 1.0 算——
    #     宁可不沉，不错沉）；
    #   - kind 没有就按核心/非核心派生（核心 stable，其余 episodic）；
    #   - importance 没有就按核心 high、其余 normal。
    decay_kind = None
    importance = None
    last_used_at = None
    decay_seen_days = None
    decay_last_seen_day = None
    decay_seen_dates = []
    decay_seen_sessions = []
    decay_state = DECAY_STATE_VERBATIM
    decay_delete_due = False
    superseded = False
    if isinstance(raw, dict):
        raw_kind = raw.get("decay_kind")
        if raw_kind in (DECAY_KIND_STABLE, DECAY_KIND_EPISODIC):
            decay_kind = raw_kind
        raw_importance = raw.get("importance")
        if raw_importance in ("high", "normal", "low"):
            importance = raw_importance
        raw_last_used = raw.get("last_used_at")
        if isinstance(raw_last_used, str) and raw_last_used:
            last_used_at = raw_last_used
        raw_seen = raw.get("decay_seen_days")
        if isinstance(raw_seen, int) and not isinstance(raw_seen, bool) and raw_seen >= 0:
            decay_seen_days = raw_seen
        raw_seen_day = raw.get("decay_last_seen_day")
        if isinstance(raw_seen_day, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", raw_seen_day
        ):
            decay_last_seen_day = raw_seen_day
        # 出现集合（2026-11-19）：库里是 JSON 字符串，规整后是列表；
        # 规整过的再规整（快照比对那条路）传进来的已经是列表，也认。
        decay_seen_dates = _parse_seen_list(raw.get("decay_seen_dates"))
        decay_seen_sessions = _parse_seen_list(raw.get("decay_seen_sessions"))
        raw_state = raw.get("decay_state")
        if raw_state in (DECAY_STATE_VERBATIM, DECAY_STATE_DORMANT):
            decay_state = raw_state
        elif raw_state == "summarized":
            # 旧版（未合进 main 的衰减分支）写过的摘要态：那版没上过真库，
            # 这里只是兜底——摘要态的文本就是原话的短版，按沉底态认，
            # 照常能搜到、被用上回到原话态。
            decay_state = DECAY_STATE_DORMANT
        decay_delete_due = raw.get("decay_delete_due") is True
        superseded = raw.get("superseded") is True
    if last_used_at is None:
        last_used_at = created_at
    _decay_core_derived = bool(user_confirmed) or slot is not None or bool(is_rule)
    if decay_kind is None:
        decay_kind = DECAY_KIND_STABLE if _decay_core_derived else DECAY_KIND_EPISODIC
    if importance is None:
        importance = "high" if _decay_core_derived else "normal"
    if created_at is None and isinstance(item, dict):
        # 33 号兜底：mem0 把 created_at 提到条目顶层了，metadata 里没有
        # 就到顶层找。顶层的时间就是 payload 里存的那个，不是现编的。
        top_created = item.get("created_at")
        if isinstance(top_created, str) and top_created:
            created_at = top_created
    return {
        "source": source,
        "created_at": created_at,
        "created_at_source": created_at_source,
        "write_seq": write_seq,
        "turn_id": turn_id,
        "confidence": confidence,
        "user_confirmed": user_confirmed,
        "confirmed": user_confirmed,
        "slot": slot,
        "pending_confirmation": pending_confirmation,
        "is_rule": is_rule,
        "money": money,
        "verification_status": verification_status,
        "requires_confirmation": requires_confirmation,
        "expires_at": expires_at,
        "conflict_for_review": conflict_for_review,
        "needs_review": needs_review,
        "evidence": evidence,
        "tone": tone,
        "tone_evidence": tone_evidence,
        "tone_source": tone_source,
        "voided": voided,
        "voided_at": voided_at,
        "curated": curated,
        "decay_kind": decay_kind,
        "importance": importance,
        "last_used_at": last_used_at,
        "decay_seen_days": decay_seen_days,
        "decay_last_seen_day": decay_last_seen_day,
        "decay_seen_dates": decay_seen_dates,
        "decay_seen_sessions": decay_seen_sessions,
        "decay_state": decay_state,
        "decay_delete_due": decay_delete_due,
        "superseded": superseded,
    }


def _metadata_for_store(metadata):
    """把规整过的 metadata 转成能写回 Chroma 的形状（写回前用）。

    _normalize_metadata 返回的 money 是字典形（{amount, currency, period,
    target}），Chroma 只接受标量值，存 dict 会炸掉整条写入——2026-11-16
    恒生定投就是这么在守卫恢复（_restore_snapshot_memories）时被拒收的：
    旧记忆按快照原样写回去，money 字典原样传进去，整条 update/add 失败。
    所以写回前把 money 字典展平成四个 money_* 标量字段（见 _flatten_money，
    和 _provenance 写的那头同一个做法），读回来 _normalize_metadata 再拼
    回字典形，来回不丢东西。别的字段本来就是标量，原样留着。
    money 不是字典时（没有金额）直接去掉这个键；形状坏了的字典展平后
    是四个 None（见 _flatten_money），不清掉旧字段、不把坏形状往库里写。
    不改传进来的字典，返回一份新的。只管 money 这一个已知的 dict 字段、
    和 decay_seen_dates / decay_seen_sessions 这两个已知的 list 字段——
    规整后的 metadata 里别的字段都是标量；万一以后多出别的 dict/list 字段，
    Chroma 照常会拒收、调用方的 except 会记日志，不在这里悄悄吞掉。
    """
    meta = dict(metadata or {})
    money = meta.pop("money", None)
    if isinstance(money, dict):
        meta.update(_flatten_money(money))
    # 出现集合（2026-11-19）：规整后是列表，写回 Chroma 前要变回
    # JSON 字符串——和 money 一个道理，list 值直接传 Chroma 拒收
    # （恢复旧记忆那条路走这里，丢了集合就对不上快照）。
    for key in ("decay_seen_dates", "decay_seen_sessions"):
        if isinstance(meta.get(key), list):
            meta[key] = _dump_seen_list(meta[key])
    return meta


def _clean_chroma_metadata(metadata):
    """把一份 metadata 洗成 Chroma 能接受的形状（Bug 4①，2026-11-20）。

    Chroma 只接受标量值（str/int/float/bool）：None、list、dict 直通
    会在底层被拒收（ValueError），而 _mem0_add 的回退只接 TypeError、
    接不住——入口清洗才是主防线，回退补调 update 只是第二道保险。
    洗法：先走 _metadata_for_store（money 字典拍平、两个出现集合变
    JSON 字符串），再剔掉值为 None 的键、把剩下的 list/dict 变 JSON
    字符串、其他非标量值变 str。不改传进来的字典，返回一份新的。
    不是字典时原样返回（调用方自己处理）。
    """
    if not isinstance(metadata, dict):
        return metadata
    meta = _metadata_for_store(metadata)
    out = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = str(value)
    return out


# ---------- 时间戳 + 覆盖规则（第一步，2026-11-10） ----------
# 规则（BOOS 定的）：时间戳决定新旧，来源可信度决定有没有资格覆盖。
# 同级里新的胜出，旧的标“被取代”但原话留着；低可信度的新记忆不能
# 直接覆盖高可信度的旧事实——新记忆挂起来“待确认”，只有真用上时
# 才问 BOOS；时间戳相同或缺失、真分不出先后的，问 BOOS；核心记忆
# 钉住不动，模型推断一律不能顶掉。
# 这几个函数都是纯函数（只“判”，不动库），调用方见 _resolve_conflict
# 的注释——它不管两条记忆是不是同一件事，“是不是同一件事”由调用方
# 按各自的证据定（槽位相同、BOOS 亲口指的、提取后守卫的关键词邻居）。


def _credibility_rank(metadata):
    """一条记忆的可信度级别（覆盖规则用，见 _resolve_conflict）。

    只看规整过的 metadata：confirmed（user 且逐条确认）最高；
    source=user 但没确认次之；inferred 再次；没来源信息的最低。
    raw 和规整过的都能传——里面先过一遍 _normalize_metadata，
    所以两种形状结果一样。
    """
    meta = _normalize_metadata(metadata)
    if meta["confirmed"]:
        return CREDIBILITY_CONFIRMED
    if meta["source"] == SOURCE_USER:
        return CREDIBILITY_USER
    if meta["source"] == SOURCE_INFERRED:
        return CREDIBILITY_INFERRED
    return CREDIBILITY_UNCONFIRMED

def _is_core_memory(metadata):
    """是不是核心记忆（第一步的口径）。

    核心记忆 = 原四类（名字、SOUL、你的决定、钱的规矩）+ 安全权限 +
    项目基础决策。第一步里先按“用户已确认”来认：名字/SOUL 有槽位
    标记，其余几类都是 BOOS 亲口确认过的记忆（记住/纠正/迁移都记
    source=user、user_confirmed=True）；对话里随口说的决定是提取的
    （inferred），要等 BOOS 逐条确认才算核心——这是 Plan C 已有的
    区分，第一步不改它。以后做衰减/删除时，这里换成正式的
    “类别/重要性”字段（见下面“以后留的位置”）。
    raw 和规整过的都能传。
    """
    meta = _normalize_metadata(metadata)
    return bool(meta["confirmed"]) or meta["slot"] in KNOWN_SLOTS


def _should_be_curated(metadata):
    """这条记忆进不进精选清单（预整理，2026-11-20）。

    条件照搬十一的 MEMORY.md（BOOS 定）：进清单（任一满足就进）——
    (a) 核心记忆（_is_core_memory，六类）；(b) BOOS 亲口确认
    （source=user 且 user_confirmed=True，_is_core_memory 已含这一条）；
    (c) 规则（is_rule=True）；(d) 晋升成稳定事实（decay_kind=stable）。
    不进清单（任一满足就不进）——tone 是 joke / test / hypothesis，
    或 voided=True。
    raw 和规整过的都能传（内部走 _normalize_metadata）。
    """
    meta = _normalize_metadata(metadata)
    if meta["voided"] or meta["tone"] in ("joke", "test", "hypothesis"):
        return False
    if _is_core_memory(meta) or meta["is_rule"]:
        return True
    return meta["decay_kind"] == DECAY_KIND_STABLE


# _parse_created_at：实现在 memory/timeutil.py，这里只转引（见上面 import），
# 旧的 from memory.provenance import _parse_created_at 不用改。
# 挪过去的原因：memory/decay.py 也要用它，而 provenance.py 要引
# decay.py 的 classify_decay_kind，放 provenance 里会循环引用。

def _memory_order_key(metadata):
    """一条记忆的新旧排序键：(写入序号, 形成时间)。

    序号是第一顺位：单调递增、单机内一定不会撞，Windows 时钟再粗
    也不影响序号比大小（2026-11-19，排序键第一步：原来是
    (created_at, write_seq)，时间戳当第一顺位，时钟是风险源）。
    时间戳只当第二顺位、同序号时才看（基本撞不上），平时只给人看、
    写审计日志。
    没时间戳的按最早算：缺时间戳的都是时间戳功能上线之前存的，
    一定比新写的旧——这是时间线本身，不是猜；序号缺失按 -1 算。
    后补的时间戳也按没时间算：backfill 补时间时是按库里随手排的
    顺序给的“现在时间”，根本不知道历史上谁先谁后——那个先后是
    假的，不能拿来判新旧。两条后补的排出来键一样，_resolve_conflict
    就会走“分不出先后、问 BOOS”那条；后补的对新写的，新写的赢，
    因为后补的本来就是功能上线前的老记忆，本来就比新写的旧。
    迁移沿用的旧日期（migrated）是旧数据里的真日期，照常用。
    raw 和规整过的都能传。
    """
    meta = _normalize_metadata(metadata)
    if meta.get("created_at_source") == "backfilled":
        return (-1, datetime.min.replace(tzinfo=timezone.utc))
    when = _parse_created_at(meta.get("created_at"))
    if when is None:
        when = datetime.min.replace(tzinfo=timezone.utc)
    seq = meta.get("write_seq")
    if not isinstance(seq, int) or isinstance(seq, bool):
        seq = -1
    return (seq, when)

def _resolve_conflict(old_metadata, new_metadata):
    """新旧两条记忆打架时，时间戳 + 可信度怎么判（纯函数，不动库）。

    判法：
    - 两条的序号 + 时间戳全撞（都没序号、都没时间戳，或都是后补时间戳）：
      真分不出先后，问 BOOS。（只有一边缺时间戳不问：缺的那条一定是
      时间戳功能上线之前存的，一定比新写的旧——这是时间线本身，不是猜。
      序号能比就直接按序号判，不因为缺时间戳就问。）
    - 新的可信度比旧的低：旧的赢，新记忆挂起来“待确认”（pending_new），
      不覆盖——时间戳再新也没用，新旧管不了覆盖资格。
    - 新的可信度比旧的高：新的胜出，旧的标“被取代”（supersede_old）。
    - 同级：新的胜出，旧的标“被取代”；但新记忆比旧的还早（迟到的
      旧写入）时不覆盖，旧的留着（keep_old）。
    - 核心记忆钉住不动：核心记忆都是用户已确认的（见 _is_core_memory），
      可信度最高，模型推断（最低档）永远判成 pending_new——上面
      “新低旧高”那条已经盖住了，这里不用再单列。
    返回 {"winner": "old" | "new" | "ask",
           "action": "supersede_old" | "pending_new" | "keep_old" | "ask",
           "reason": 给人看的一句话}。
    注意：这个函数只“判”，不“认”——它不管两条记忆是不是同一件事。
    “是不是同一件事”由调用方按各自的证据定：同槽位就是同一件事
    （_upsert_slot_memory）；纠正是 BOOS 亲口指的（correct_memory）；
    提取后守卫只凭关键词重合找“同一话题”的邻居（_apply_coverage_guard），
    所以同级的新胜出在那里不自动标被取代——关键词重合分不出
    “同一条事实的新版本”还是“同一话题的不同事实”（“喜欢喝咖啡”
    和“喜欢喝拿铁”能同时是真的），宁可不标，不标错。等以后有模型判
    “同一事实”时，再把 supersede_old 接到提取后守卫上。
    """
    old = _normalize_metadata(old_metadata)
    new = _normalize_metadata(new_metadata)
    old_rank = _credibility_rank(old)
    new_rank = _credibility_rank(new)
    old_key = _memory_order_key(old)
    new_key = _memory_order_key(new)
    if old_key == new_key:
        return {
            "winner": "ask",
            "action": "ask",
            "reason": "两条记忆序号和时间戳都一样，分不出先后，请 BOOS 定。",
        }
    if new_rank < old_rank:
        return {
            "winner": "old",
            "action": "pending_new",
            "reason": "新记忆可信度比旧的低，不能直接覆盖，挂起来待确认。",
        }
    if new_rank > old_rank:
        return {
            "winner": "new",
            "action": "supersede_old",
            "reason": "新记忆可信度比旧的高，新的胜出，旧的标被取代。",
        }
    if new_key > old_key:
        return {
            "winner": "new",
            "action": "supersede_old",
            "reason": "同一级别，新的胜出，旧的标被取代。",
        }
    return {
        "winner": "old",
        "action": "keep_old",
        "reason": "同一级别，但新记忆比旧的还早，不覆盖。",
    }

# 保护失败时结果里附带的列表字段：没恢复回来的旧记忆 id、回滚删不掉
# 的新记忆 id 和原文。保护信息要经过守卫 → 提交 → add_turn 三层往上传，
# 以前每层手写一个键一个键地传，加个字段就得改三处，漏一处就半路丢了
# （回滚残留的原文就是这么丢的：守卫算出来了，只进了保护事件，没进
# 返回结果）。现在三层都调 _carry_protection_fields，以后再加字段只改
# 下面这个元组一行。
_PROTECTION_RESULT_KEYS = (
    "failed_ids",
    "rollback_leftover_ids",
    "rollback_leftover_texts",
)

def _carry_protection_fields(src, dst):
    """把保护失败的字段从一个结果字典带到另一个。

    只在源字典标了 protection_failed 时才带；没标时什么都不动。
    只填空、不覆盖：目标字典里已经有 protection_failed 时，目标
    那份是更新鲜的（比如纠错返回路里，本轮自己的提取也失败了），
    原样留着、一个键都不盖——上一轮的只在目标没有时才补。不能按
    键一个一个填：上一轮的 failed_ids、残留 id/原文都是上一轮那次
    失败的，单拎几个键填进本轮的失败里会串味（比如原因写着快照
    失败，failed_ids 却是上一轮没恢复回来的旧记忆）。
    失败标记和原因照原样带（原因即使是 None 也带，和以前手写时
    一样）；三个列表字段只在非空时带，空的就不写进结果——返回的
    样子不变，同样的顶层键，调用方（web 面板、命令行）不用动。
    """
    src = src or {}
    if not src.get("protection_failed"):
        return
    if dst.get("protection_failed"):
        return
    dst["protection_failed"] = True
    dst["protection_reason"] = src.get("protection_reason")
    for key in _PROTECTION_RESULT_KEYS:
        value = src.get(key)
        if value:
            dst[key] = value

