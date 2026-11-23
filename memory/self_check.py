# memory/self_check.py（十二每月一次自检：只查、只报告，不直接改库）
# 方案里的十项表落成代码（2026-11-19，BOOS 定：每月一次自检；
# 第 10 项 2026-11-23 Phase 2 #2 加：核心记忆还管不管用）。
# 只查不改：要改的写成每项的 actions 列表，由调用方决定做不做；
# 跑两遍结果一致（findings 里不放随机数、不放当前时间戳以外的可变值，
# generated_at 除外）。只用标准库 + memory/decay.py + memory/tone.py
# + memory/content_check.py，不碰 mem0_store.py，不直接读库。
#
# 适配器约定（调用方传一个 adapter 对象，提供下面这些方法）：
#   get_all_items() -> [{"id", "text", "metadata"}]：
#       规整过的全部记忆；metadata 是 _normalize_metadata 那个形状，
#       含 source / confidence / user_confirmed / decay_kind /
#       decay_state / decay_seen_dates / decay_seen_sessions /
#       last_used_at / tone / voided / superseded / needs_review /
#       conflict_for_review / grounding_flag（还有 turn_id / slot /
#       is_rule / decay_delete_due 等，见 provenance.py）。
#   supersede_events_since(days) -> [{"id", "text", "at"}]：
#       最近 days 天内被取代的记忆（账本 supersede 事件）。
#   verify_backup() -> {"ok": bool, "detail": str}：
#       备份恢复演练的结果（真从备份恢复一次、验过才算 ok）。
#   index_counts() -> {"sqlite": int, "chroma": int,
#                      "mismatched_ids": [...]}：
#       两边索引的条数，和对不上的 id。
#   replay_promotion_counts() -> {"checked": int, "corrections":
#       [{"id", "field", "old", "new"}]}：
#       从原始对话日志重放晋升计数的结果（出现次数按不同天、不同会话数）。
#   archive_items() -> [{"id", "text"}]：档案层记忆。
#   embed(text) -> list | None：文本的向量；没有向量模型时返回 None，
#       第 1 项（重复记忆）跳过并注明。
#
# run_self_check(adapter, now=None, model_judge=None, sample_size=10,
#                router_status=None) 返回报告 dict：
#   {"generated_at", "items": {1..10}, "most_likely_wrong_5": [...],
#    "replaced_last_month": [...], "tagged_tone_memories": [...],
#    "organizing_budget": {...}, "summary": str}
# 每项是 {"name", "status", "findings", "actions"}，status 是
# pass / warn / fail / skipped。九项按方案：
#   1 重复记忆、2 过期没沉底、3 冲突未决、4 孤儿记忆、5 备份能恢复吗、
#   6 索引有没有坏、7 晋升计数、8 归档一致性、9 模型抽查、
#   10 核心记忆还管不管用。
# most_likely_wrong_5 调 content_check.rank_most_likely_wrong；
# tagged_tone_memories 是 tone 为 joke / test / hypothesis 的记忆；
# organizing_budget 是传进来的 router_status（没有时 {}）。

from datetime import datetime, timezone

from memory.decay import decay_should_delete, decay_should_go_dormant

# tone.py / content_check.py 由另外的工作在写，按约定的名字调；
# 还没落地时先按字面值顶上，模块照常能跑、测试里用假的顶上。
try:
    from memory.tone import (
        TONE_HYPOTHESIS,
        TONE_JOKE,
        TONE_SERIOUS,
        TONE_TEST,
    )
except ImportError:  # tone.py 还没写好：先用约定的字面值
    TONE_SERIOUS = "serious"
    TONE_JOKE = "joke"
    TONE_TEST = "test"
    TONE_HYPOTHESIS = "hypothesis"

try:
    from memory.content_check import rank_most_likely_wrong
except ImportError:  # content_check.py 还没写好：最可能错的 5 条先空着
    rank_most_likely_wrong = None

# 第 9 项模型抽查用的提示词（写成常量，抽查问法固定、可复核）。
# 问题只有一个：这条记忆跟它的出处对不对得上、有没有内部矛盾。
SELF_CHECK_JUDGE_SYSTEM_PROMPT = (
    "你是十二记忆系统的自检员，只判断、不改库。\n"
    "下面给你一条记忆的原文和它的出处信息，请判断这条记忆准不准：\n"
    "这条记忆跟它的出处对不对得上、有没有内部矛盾。\n"
    "只答这一条，不要猜别的记忆。\n"
    "返回 {\"accurate\": true/false, \"evidence\": \"...\"}：\n"
    "accurate 为 true 表示内容跟出处对得上、没有内部矛盾；\n"
    "false 表示对不上或有矛盾；判不出来时返回 null。\n"
    "evidence 里必须带原文片段（从记忆原文里抄一句），不能只写结论。"
)
SELF_CHECK_JUDGE_USER_PROMPT_TEMPLATE = (
    "请判断下面这条记忆准不准。\n"
    "记忆 id：{id}\n"
    "记忆原文：{text}\n"
    "来源：{source}\n"
    "可信度：{confidence}\n"
    "语气：{tone}\n"
    "出处证据：{evidence}\n"
    "\n"
    "问题：这条记忆跟它的出处对不对得上、有没有内部矛盾？\n"
    "evidence 里必须带原文片段。"
)

# 第 10 项核心记忆有用性抽查用的提示词（写成常量，抽查问法固定、可复核）。
# 复用第 9 项的 model_judge 接口（judge(system_prompt, user_prompt)
# -> {"accurate": bool, "evidence": str} 或 None）：这里的
# "accurate" 读作“这条核心记忆现在还管用”——true 表示还管用，
# false 表示不管用了（过时了、BOOS 的情况变了、留着只会误导），
# 判不出来返回 null。只提议、不自动改：不管用的只进 actions
# （建议沉底/归档），由 BOOS 定，绝不自动降权、绝不自动删。
SELF_CHECK_USEFULNESS_SYSTEM_PROMPT = (
    "你是十二记忆系统的自检员，只判断、不改库。\n"
    "下面给你一条核心记忆的原文和它的基本信息，请判断这条记忆现在还管不管用：\n"
    "BOOS 的情况是不是变了、这条记忆是不是过时了、留着会不会误导以后的回答。\n"
    "只答这一条，不要猜别的记忆。\n"
    "返回 {\"accurate\": true/false, \"evidence\": \"...\"}：\n"
    "accurate 为 true 表示现在还管用；false 表示不管用了；判不出来时返回 null。\n"
    "evidence 里必须带原文片段（从记忆原文里抄一句），不能只写结论。"
)
SELF_CHECK_USEFULNESS_USER_PROMPT_TEMPLATE = (
    "请判断下面这条核心记忆现在还管不管用。\n"
    "记忆 id：{id}\n"
    "记忆原文：{text}\n"
    "来源：{source}\n"
    "可信度：{confidence}\n"
    "上次用过：{last_used_at}\n"
    "\n"
    "问题：BOOS 的情况变了没有？这条记忆现在还管用吗？\n"
    "evidence 里必须带原文片段。"
)

# 第 1 项的相似度线（方案定值）：向量相似度 > 0.95 且有共同关键词才报。
DUPLICATE_SIMILARITY_THRESHOLD = 0.95

# 第 3 项看最近多少天被取代的记忆（方案：上个月）。
REPLACED_LOOKBACK_DAYS = 30

# 语气打标里“不是认真说的”三种（见 tone.py）：只报告、不参与晋升。
NON_SERIOUS_TONES = (TONE_JOKE, TONE_TEST, TONE_HYPOTHESIS)

# 关键词分词、向量相似度、核心记忆判定都用现成的（别各写一份，
# 各写一份迟早走样：_is_core_memory 尤其不能有两个口径）。
from memory.keywords import _cosine_similarity, _extract_keywords
from memory.provenance import _is_core_memory


def _item(name, status, findings, actions):
    """拼一项自检结果（固定形状，调用方按这个读）。"""
    return {
        "name": name,
        "status": status,
        "findings": findings,
        "actions": actions,
    }


def _check_duplicates(items, adapter):
    """第 1 项：重复记忆。
    两两比向量相似度：> 0.95 且有共同关键词才报。核心记忆只报告、
    不建议合并（钉住不动）。没有向量模型（embed 全返回 None）时
    跳过并注明。"""
    name = "1 重复记忆"
    vectors = {}
    for item in items:
        try:
            vector = adapter.embed(item.get("text") or "")
        except Exception:
            vector = None
        if vector is not None:
            try:
                vectors[item.get("id")] = [float(v) for v in vector]
            except (TypeError, ValueError):
                continue
    if not vectors:
        return _item(
            name,
            "skipped",
            [{"reason": "没有向量模型，embed 返回 None，重复检查跳过。"}],
            [],
        )
    findings = []
    actions = []
    ordered = [item for item in items if item.get("id") in vectors]
    for i, left in enumerate(ordered):
        left_keywords = set(_extract_keywords(left.get("text") or ""))
        for right in ordered[i + 1 :]:
            similarity = _cosine_similarity(
                vectors[left.get("id")], vectors[right.get("id")]
            )
            if similarity <= DUPLICATE_SIMILARITY_THRESHOLD:
                continue
            shared = left_keywords & set(
                _extract_keywords(right.get("text") or "")
            )
            if not shared:
                continue
            left_core = _is_core_memory(left.get("metadata") or {})
            right_core = _is_core_memory(right.get("metadata") or {})
            finding = {
                "id": left.get("id"),
                "other_id": right.get("id"),
                "text": left.get("text"),
                "other_text": right.get("text"),
                "similarity": round(similarity, 4),
                "shared_keywords": sorted(shared),
                "reason": "向量相似度超过 0.95 且有共同关键词，可能是重复记忆。",
            }
            if left_core or right_core:
                finding["reason"] += "其中有核心记忆，只报告，不建议合并。"
            else:
                actions.append(
                    f"建议人工核对 {left.get('id')} 和 {right.get('id')} "
                    "是不是同一条记忆，是的话合并。"
                )
            findings.append(finding)
    if not findings:
        return _item(name, "pass", [], [])
    return _item(name, "warn", findings, actions)


def _check_decay(items, now):
    """第 2 项：过期没沉底。
    用 decay_should_go_dormant / decay_should_delete 判：该标沉底、
    该标待删但状态没变的列出来，只建议打标——不压摘要、不直接删，
    删要等 BOOS 亲口说。
    注意（P1-6，2026-11-23，接受 + 注明）：note_context 刷新 last_used_at
    只写 SQLite、不记账本，所以账本重建后 last_used_at 会变回账本里的
    旧值，衰减状态可能随之重置；这是接受了的行为，不是 bug——每次检索
    都记账本事件太吵，故意不记。"""
    name = "2 过期没沉底"
    findings = []
    actions = []
    for item in items:
        meta = item.get("metadata") or {}
        is_core = _is_core_memory(meta)
        if decay_should_go_dormant(
            meta.get("decay_kind") or "episodic",
            meta.get("last_used_at"),
            now,
            is_core=is_core,
        ):
            if meta.get("decay_state") != "dormant":
                findings.append(
                    {
                        "id": item.get("id"),
                        "text": item.get("text"),
                        "reason": "30 天没人找、权重已掉到沉底线下面，"
                        "该标沉底但还是原话态。",
                        "current_state": meta.get("decay_state"),
                    }
                )
                actions.append(f"建议给 {item.get('id')} 打标沉底（只打标）。")
        if decay_should_delete(
            meta.get("last_used_at"), now, is_core=is_core
        ):
            if not meta.get("decay_delete_due"):
                findings.append(
                    {
                        "id": item.get("id"),
                        "text": item.get("text"),
                        "reason": "满 180 天没人找，该标待删但还没打标。",
                        "current_state": meta.get("decay_state"),
                    }
                )
                actions.append(
                    f"建议给 {item.get('id')} 打标待删（只打标，不直接删，"
                    "删要等 BOOS 亲口说）。"
                )
    if not findings:
        return _item(name, "pass", [], [])
    return _item(name, "warn", findings, actions)


def _check_conflicts(items):
    """第 3 项：冲突未决。
    conflict_for_review 或 needs_review 的列出来，等 BOOS 看；
    被取代的记忆不在这一项里（见顶层 replaced_last_month）。"""
    name = "3 冲突未决"
    findings = []
    actions = []
    for item in items:
        meta = item.get("metadata") or {}
        reasons = []
        if meta.get("conflict_for_review") is True:
            reasons.append("说法对着干，等 BOOS 裁决")
        if meta.get("needs_review") is True:
            reasons.append("模型裁决把握不够，等兜底清理时再看")
        if reasons:
            findings.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "reason": "；".join(reasons) + "。",
                }
            )
            actions.append(f"请 BOOS 看一下 {item.get('id')}。")
    if not findings:
        return _item(name, "pass", [], [])
    return _item(name, "warn", findings, actions)


def _check_orphans(items):
    """第 4 项：孤儿记忆。
    metadata 里 source 是 unconfirmed 且没有 turn_id 的列出来——
    不知道哪来的、也找不到是哪轮对话写的。只报告，不建议删。"""
    name = "4 孤儿记忆"
    findings = []
    for item in items:
        meta = item.get("metadata") or {}
        if meta.get("source") == "unconfirmed" and not meta.get("turn_id"):
            findings.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "reason": "来源未确认且没有 turn_id，找不到是哪轮对话写的。"
                    "只报告，不建议删除。",
                }
            )
    if not findings:
        return _item(name, "pass", [], [])
    return _item(name, "warn", findings, [])


def _check_backup(adapter):
    """第 5 项：备份能恢复吗。
    调 adapter.verify_backup()：ok=False 时 status=fail，
    并写明恢复失败就禁止删除（删的前提是备份能恢复）。"""
    name = "5 备份能恢复吗"
    result = adapter.verify_backup() or {}
    detail = result.get("detail") or ""
    if result.get("ok") is True:
        return _item(
            name, "pass", [{"detail": detail, "reason": "备份恢复演练通过。"}], []
        )
    return _item(
        name,
        "fail",
        [
            {
                "detail": detail,
                "reason": "备份恢复演练失败。恢复失败就禁止删除任何记忆，"
                "先修好备份、重新演练通过再说。",
            }
        ],
        ["修复备份并重新做恢复演练，演练通过之前禁止删除任何记忆。"],
    )


def _check_index(adapter):
    """第 6 项：索引有没有坏。
    sqlite 与 chroma 数量对不上、或 mismatched_ids 非空时 warn。"""
    name = "6 索引有没有坏"
    counts = adapter.index_counts() or {}
    sqlite_count = counts.get("sqlite")
    chroma_count = counts.get("chroma")
    mismatched = list(counts.get("mismatched_ids") or [])
    if sqlite_count == chroma_count and not mismatched:
        return _item(
            name,
            "pass",
            [{"sqlite": sqlite_count, "chroma": chroma_count, "reason": "两边数量一致。"}],
            [],
        )
    return _item(
        name,
        "warn",
        [
            {
                "sqlite": sqlite_count,
                "chroma": chroma_count,
                "mismatched_ids": mismatched,
                "reason": "sqlite 与 chroma 数量对不上，或有对不上的 id。",
            }
        ],
        ["建议从账本重建索引后重新对账。"],
    )


def _check_promotion(adapter):
    """第 7 项：晋升计数。
    调 adapter.replay_promotion_counts()：从原始对话日志重放晋升计数，
    有 corrections 时列出来（出现次数要按不同天、不同会话数）。"""
    name = "7 晋升计数"
    result = adapter.replay_promotion_counts() or {}
    corrections = list(result.get("corrections") or [])
    if not corrections:
        return _item(
            name,
            "pass",
            [
                {
                    "checked": result.get("checked"),
                    "reason": "从原始对话日志重放晋升计数，无修正。",
                }
            ],
            [],
        )
    return _item(
        name,
        "warn",
        corrections,
        ["建议按重放结果修正晋升计数（出现次数按不同天、不同会话数）。"],
    )


def _check_archive(items, adapter):
    """第 8 项：归档一致性。
    archive_items 的原话在 get_all_items 里找不着时列出来
    （档案层有、当前库里没了）。"""
    name = "8 归档一致性"
    current_texts = {(item.get("text") or "") for item in items}
    findings = []
    for archived in adapter.archive_items() or []:
        text = archived.get("text") or ""
        if text not in current_texts:
            findings.append(
                {
                    "id": archived.get("id"),
                    "text": text,
                    "reason": "档案层有这条原话，当前库里找不着。",
                }
            )
    if not findings:
        return _item(name, "pass", [], [])
    return _item(
        name, "warn", findings, ["建议核对档案层与当前库，找回缺失的原话。"]
    )


def _deterministic_sample(items, sample_size):
    """确定性取样：按 id 排序后均匀取 sample_size 条。
    同一份库跑两遍，抽到的是同一批——结果可复核。"""
    ordered = sorted(items, key=lambda entry: str(entry.get("id") or ""))
    count = len(ordered)
    if count <= sample_size:
        return list(ordered)
    return [ordered[(i * count) // sample_size] for i in range(sample_size)]


def _check_model_sample(items, model_judge, sample_size):
    """第 9 项：模型抽查。
    model_judge 为 None 时 skipped；否则确定性取样后逐条问
    model_judge(system_prompt, user_prompt)“内容准不准”，judge 返回
    {"accurate": bool, "evidence": str} 或 None（判不出来记 skipped，
    只记在 findings 里）。只报告、不自动改。
    抽到的全都判不出来时整项记 skipped（不是 pass——什么都没查成，
    不能写“通过”）；
    只要有一条判出来了，没发现不准的记 pass，发现不准的记 warn。"""
    name = "9 模型抽查"
    if model_judge is None:
        return _item(
            name,
            "skipped",
            [{"reason": "没传 model_judge，模型抽查跳过。"}],
            [],
        )
    if not items:
        return _item(
            name, "skipped", [{"reason": "库里没有记忆，没得抽。"}], []
        )
    sampled = _deterministic_sample(items, sample_size)
    findings = []
    for item in sampled:
        meta = item.get("metadata") or {}
        user_prompt = SELF_CHECK_JUDGE_USER_PROMPT_TEMPLATE.format(
            id=item.get("id"),
            text=item.get("text") or "",
            source=meta.get("source"),
            confidence=meta.get("confidence"),
            tone=meta.get("tone"),
            evidence=meta.get("evidence") or meta.get("tone_evidence") or "",
        )
        try:
            verdict = model_judge(SELF_CHECK_JUDGE_SYSTEM_PROMPT, user_prompt)
        except Exception:
            verdict = None
        if not isinstance(verdict, dict):
            findings.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "accurate": None,
                    "evidence": None,
                    "reason": "模型判不出来（没返回有效结论），记 skipped。",
                }
            )
            continue
        accurate = verdict.get("accurate")
        if not isinstance(accurate, bool):
            findings.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "accurate": None,
                    "evidence": verdict.get("evidence"),
                    "reason": "模型判不出来（没返回有效结论），记 skipped。",
                }
            )
            continue
        findings.append(
            {
                "id": item.get("id"),
                "text": item.get("text"),
                "accurate": accurate,
                "evidence": verdict.get("evidence"),
                "reason": "模型抽查：内容跟出处对得上。"
                if accurate
                else "模型抽查：内容跟出处对不上或有内部矛盾，只报告，不自动改。",
            }
        )
    if any(entry.get("accurate") is False for entry in findings):
        return _item(
            name,
            "warn",
            findings,
            ["模型抽查发现存疑的记忆，请 BOOS 看一下，确认后再处理。"],
        )
    if findings and all(entry.get("accurate") is None for entry in findings):
        # 抽到的全都判不出来（比如大模型没 Key、调不通）：什么都没查成，
        # 记 skipped，别写 pass 骗人。
        return _item(
            name,
            "skipped",
            findings
            + [{"reason": "抽到的记忆全都没判出来，本月抽查没查成。"}],
            [],
        )
    return _item(name, "pass", findings, [])


def _check_core_usefulness(items, model_judge):
    """第 10 项：核心记忆还管不管用（Phase 2 #2）。

    核心记忆是钉住不动的（权重恒 1、不沉底、不待删），但 BOOS 的情况
    会变——搬家了、换偏好了、不定投了，旧的核心记忆留着只会误导。
    每月一次：把每条核心记忆拿给大模型判“现在还管不管用”，不管用
    的只写进 actions（建议沉底/归档），由 BOOS 定——绝不自动降权、
    绝不自动删、绝不自动改。model_judge 复用第 9 项的接口：
    judge(system_prompt, user_prompt) -> {"accurate": bool,
    "evidence": str} 或 None，这里的 accurate 读作“还管用”。
    没传 model_judge 时 skipped；库里没有核心记忆时 skipped；
    抽到的全都判不出来时整项记 skipped（不是 pass——什么都没查成，
    不能写“通过”）；只要有一条判出来了，没发现不管用的记 pass，
    发现不管用的记 warn。核心记忆按 id 排序后逐条问（确定性，
    跑两遍结果一致）。
    """
    name = "10 核心记忆还管不管用"
    if model_judge is None:
        return _item(
            name,
            "skipped",
            [{"reason": "没传 model_judge，核心记忆有用性抽查跳过。"}],
            [],
        )
    core_items = [
        item
        for item in items
        if _is_core_memory(item.get("metadata") or {})
        and not (item.get("metadata") or {}).get("voided")
    ]
    if not core_items:
        return _item(
            name, "skipped", [{"reason": "库里没有核心记忆，没得查。"}], []
        )
    core_items = sorted(core_items, key=lambda entry: str(entry.get("id") or ""))
    findings = []
    for item in core_items:
        meta = item.get("metadata") or {}
        user_prompt = SELF_CHECK_USEFULNESS_USER_PROMPT_TEMPLATE.format(
            id=item.get("id"),
            text=item.get("text") or "",
            source=meta.get("source"),
            confidence=meta.get("confidence"),
            last_used_at=meta.get("last_used_at"),
        )
        try:
            verdict = model_judge(SELF_CHECK_USEFULNESS_SYSTEM_PROMPT, user_prompt)
        except Exception:
            verdict = None
        if not isinstance(verdict, dict):
            findings.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "accurate": None,
                    "evidence": None,
                    "reason": "模型判不出来（没返回有效结论），记 skipped。",
                }
            )
            continue
        accurate = verdict.get("accurate")
        if not isinstance(accurate, bool):
            findings.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "accurate": None,
                    "evidence": verdict.get("evidence"),
                    "reason": "模型判不出来（没返回有效结论），记 skipped。",
                }
            )
            continue
        findings.append(
            {
                "id": item.get("id"),
                "text": item.get("text"),
                "accurate": accurate,
                "evidence": verdict.get("evidence"),
                "reason": "模型抽查：这条核心记忆现在还管用。"
                if accurate
                else "模型抽查：这条核心记忆现在不管用了，只提议沉底/归档，不自动改。",
            }
        )
    stale = [entry for entry in findings if entry.get("accurate") is False]
    if stale:
        actions = [
            "建议把 %s 沉底/归档（只提议，不自动改），理由：%s"
            % (
                entry.get("id"),
                entry.get("evidence") or entry.get("reason") or "模型判不管用",
            )
            for entry in stale
        ]
        return _item(name, "warn", findings, actions)
    if findings and all(entry.get("accurate") is None for entry in findings):
        return _item(
            name,
            "skipped",
            findings + [{"reason": "核心记忆全都没判出来，本月这项没查成。"}],
            [],
        )
    return _item(name, "pass", findings, [])


def run_self_check(adapter, now=None, model_judge=None, sample_size=10,
                   router_status=None):
    """跑一次十二的每月自检（只查、只报告，不直接改库）。

    adapter：见模块 docstring 的适配器约定。
    now：判衰减用的“现在”（datetime 或 ISO 串，不传就是现在）。
    model_judge：模型抽查用的判官，调 model_judge(system_prompt,
        user_prompt) -> {"accurate": bool, "evidence": str} 或 None；
        不传时第 9 项 skipped。
    sample_size：第 9 项抽几条（确定性取样，按 id 排序后均匀取）。
    router_status：整理预算状态（dict），没有时 {}；里面 over_budget
        为真时，summary 里写一句超预算提醒。
    返回报告 dict（见模块 docstring）。同一份 adapter 状态跑两遍，
    除 generated_at 外报告完全一致。"""
    items = list(adapter.get_all_items() or [])
    # 最近 30 天被取代的记忆（账本 supersede 事件）：顶层单列，方便看。
    replaced_last_month = list(
        adapter.supersede_events_since(REPLACED_LOOKBACK_DAYS) or []
    )
    # 语气打标的记忆：tone 为 joke / test / hypothesis 的单列出来
    # （只报告、不参与晋升，见 decay.should_promote）。
    tagged_tone_memories = []
    for item in items:
        meta = item.get("metadata") or {}
        if meta.get("tone") in NON_SERIOUS_TONES:
            tagged_tone_memories.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "tone": meta.get("tone"),
                    "tone_evidence": meta.get("tone_evidence"),
                }
            )
    # 最可能错的 5 条：调 content_check.rank_most_likely_wrong，
    # grounding_flag 从 metadata 里取；模块还没落地时先空着。
    most_likely_wrong_5 = []
    if rank_most_likely_wrong is not None:
        check_items = [
            {
                "id": item.get("id"),
                "text": item.get("text"),
                "grounding_flag": (item.get("metadata") or {}).get(
                    "grounding_flag"
                ),
                "metadata": item.get("metadata") or {},
            }
            for item in items
        ]
        ranked = rank_most_likely_wrong(check_items, top_n=5)
        if isinstance(ranked, list):
            most_likely_wrong_5 = ranked
    # 整理预算：原样透传（没有时 {}），超预算提醒写进 summary。
    organizing_budget = (
        dict(router_status) if isinstance(router_status, dict) else {}
    )
    over_budget = organizing_budget.get("over_budget") is True

    report_items = {
        1: _check_duplicates(items, adapter),
        2: _check_decay(items, now),
        3: _check_conflicts(items),
        4: _check_orphans(items),
        5: _check_backup(adapter),
        6: _check_index(adapter),
        7: _check_promotion(adapter),
        8: _check_archive(items, adapter),
        9: _check_model_sample(items, model_judge, sample_size),
        10: _check_core_usefulness(items, model_judge),
    }
    statuses = [entry["status"] for entry in report_items.values()]
    passed = statuses.count("pass")
    warned = statuses.count("warn")
    failed = statuses.count("fail")
    skipped = statuses.count("skipped")
    problem_numbers = [
        number
        for number, entry in sorted(report_items.items())
        if entry["status"] in ("warn", "fail")
    ]
    summary = (
        f"本次自检共查 10 项：{passed} 项通过，{warned} 项有问题，"
        f"{failed} 项失败，{skipped} 项跳过。"
    )
    if problem_numbers:
        summary += (
            "有问题的是第"
            + "、第".join(str(number) for number in problem_numbers)
            + "项，建议 BOOS 看一下。"
        )
    else:
        summary += "没有发现需要处理的问题，不用 BOOS 看。"
    if over_budget:
        summary += "提醒：整理预算已超预算，请 BOOS 看一下要不要调整。"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": report_items,
        "most_likely_wrong_5": most_likely_wrong_5,
        "replaced_last_month": replaced_last_month,
        "tagged_tone_memories": tagged_tone_memories,
        "organizing_budget": organizing_budget,
        "summary": summary,
    }
