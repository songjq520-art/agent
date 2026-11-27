"""整理员对接适配层（立项 G1 字段映射，2026-11-27）。

把贵妃侧整理员协议（host_review.Proposal / review_proposal 的
current / curator.Candidate / correction_receipt 证据形状）映射到
宿主真实口：H1 统一受控入口（submit_controlled_proposal）的提案
与回执、H4 持锁快照（take_stable_snapshot）的行视图、core_cap /
provenance 的权威分类与可信度口径。映射表与口径说明见
g1_field_mapping_20261127.md（目标 workspace 存档）。

本层只做字段转换与证据拼装：不执行、不签发授权、不改评估
判定。执行接线（G2）在这层之上组装 H1 提案并提交；开关与
门禁全在 H1/H2 既有口里，本层不许绕。
"""

import copy

from memory import core_cap as _core_cap
from memory import provenance as _provenance

# 整理员对接口径版本：方案/契约口径变更时升版。提案侧与现状侧
# 都由本层盖同一常量——口径一变，两边版本对不上即挂起（有意
# 设计，见映射表第一节）。
POLICY_VERSION = "curator-policy-20261124-v1"

# 她侧 assess 认的判官状态合法集；集外值透传后由她侧归
# hold_judge_unknown，本层不替她改判（映射表第三节）。
JUDGE_STATUSES = frozenset({
    "valid", "unreachable", "budget_exhausted",
    "malformed_or_timeout", "uncertain",
})


def host_protected(store, meta):
    """protected 回调的宿主权威实现：唯一核心保护谓词。"""
    return bool(store._is_core_protected_row(meta or {}))


def host_classify(meta, live=True):
    """classify 回调的宿主权威实现：core_cap 统一分类口径。"""
    return _core_cap.classify(meta or {}, live=live)


def _classify_or_none(meta, live):
    try:
        return _core_cap.classify(meta or {}, live=live)
    except Exception:
        return None


def build_proposal_fields(row):
    """H4 快照行 → 她侧 Proposal 字段（dict，键与 dataclass 同名）。

    宿主只有一个版本戳（正文＋完整 metadata＋user_confirmed 的
    哈希）：content_version 与 protection_version 同绑这一戳——
    保护信息就在 metadata 里，保护一变戳必变，双绑比双戳分离
    更严，不构成字段缺失（映射表第一节）。
    """
    version = row.get("version")
    return {
        "memory_id": row.get("memory_id") or row.get("id"),
        "content_version": version,
        "protection_version": version,
        "policy_version": POLICY_VERSION,
    }


def pending_memory_ids(store):
    """当前有未答待问的 memory_id 集合（pending_classification 用）。"""
    ids = set()
    try:
        questions = store.list_pending_questions()
    except Exception:
        return ids
    for q in questions or []:
        if isinstance(q, dict) and q.get("memory_id"):
            ids.add(q["memory_id"])
    return ids


def build_current(store, row, *, pending_ids=None):
    """H4 快照行 → review_proposal 的 current（现状）dict。

    每个键都由宿主现行口径现算，不许调用方手填（映射表第二
    节）。protection_resolved 的口径：classify 是宿主的正面
    裁定（只认 pinned/temp_pinned、其余即普通），算得出即已
    裁定；meta 损坏算不出 → False，朝安全侧倒。
    """
    meta = row.get("meta") or {}
    live = bool(row.get("live"))
    classification = _classify_or_none(meta, live)
    if pending_ids is None:
        pending_ids = pending_memory_ids(store)
    memory_id = row.get("memory_id") or row.get("id")
    current = build_proposal_fields(row)
    current.update({
        "live": live,
        "protection_resolved": classification is not None,
        "readonly_projection": bool(meta.get("persona_projection")),
        "pending_classification": memory_id in pending_ids,
        "metadata": copy.deepcopy(meta),
    })
    return current


def build_candidate_fields(row, *, judge_status="uncertain"):
    """H4 快照行 → curator.Candidate 字段（dict，键与 dataclass 同名）。"""
    meta = row.get("meta") or {}
    live = bool(row.get("live"))
    classification = _classify_or_none(meta, live) or {}
    tier = classification.get("protection_tier")
    try:
        rank = _provenance._credibility_rank(meta)
    except Exception:
        rank = _provenance.CREDIBILITY_UNCONFIRMED
    fields = build_proposal_fields(row)
    fields.update({
        "core": classification.get("member") is True,
        "protection": "ordinary" if tier is None else tier,
        "source_verified": rank >= _provenance.CREDIBILITY_USER,
        "judge_status": (
            judge_status if isinstance(judge_status, str) and judge_status
            else "uncertain"
        ),
    })
    return fields


def h1_proposal(fields, *, action="demote", reason="",
                instruction_source=None, idempotency_key=None,
                params=None, session=None):
    """她侧提案字段 → H1 submit_controlled_proposal 的提案 dict。

    expected_version 绑提案的 content_version（即宿主戳）。
    instruction_source / idempotency_key / session 是执行层
    供给，不从她侧协议字段推导、本层不伪造（映射表第四节）；
    缺幂等键时如实留 None——H1 会以 missing 口径拒绝，朝安全
    侧倒。
    """
    proposal = {
        "action": action,
        "memory_id": fields.get("memory_id"),
        "reason": reason,
        "instruction_source": instruction_source,
        "idempotency_key": idempotency_key,
        "expected_version": fields.get("content_version"),
    }
    if params is not None:
        proposal["params"] = params
    if session is not None:
        proposal["session"] = session
    return proposal


def receipt_summary(receipt):
    """H1 回执 → 她侧记录用的归一摘要（原回执整份另行留档）。"""
    receipt = receipt or {}
    status = receipt.get("status")
    return {
        "executed": status == "executed" and receipt.get("ok") is True,
        "shadow": status == "shadow",
        "refused": status == "refused",
        "duplicate": bool(receipt.get("duplicate")),
        "reason": receipt.get("reason"),
        "question_id": receipt.get("question_id"),
        "version_before": receipt.get("version_before"),
        "version_after": receipt.get("version_after"),
    }


def correction_evidence(store, receipt, *, turn_id):
    """当轮纠正回执 → correction_receipt 的三件证据（拼装）。

    返回 (receipt_dict, old_row_dict, new_row_dict)。新旧关联
    的宿主证据在账本折叠里：ledger_state()["by_id"][旧行] =
    {state: "superseded", superseded_by: 新行, seq: 顶替事件
    序号}——顶替事件本身一条同时点名新旧两行（事件 key=旧行、
    memory_id=新行），它的 seq 即三方共用的 event_id，写作
    "ledger-seq:<seq>"（映射表第五节）。折叠里查不到这条关联
    时如实留 None，她侧判 incomplete，朝安全侧倒。
    """
    receipt = receipt or {}
    new_id = receipt.get("memory_id")
    evidence_receipt = {
        "turn_id": turn_id,
        "status": receipt.get("status"),
        "old_id": None,
        "new_id": new_id,
        "event_id": None,
    }
    old_row = {}
    new_row = {}
    if not new_id:
        return evidence_receipt, old_row, new_row
    event_id = None
    try:
        by_id = (store.ledger_state() or {}).get("by_id") or {}
    except Exception:
        by_id = {}
    for mid, folded in by_id.items():
        if (isinstance(folded, dict)
                and folded.get("state") == "superseded"
                and folded.get("superseded_by") == new_id):
            evidence_receipt["old_id"] = mid
            if folded.get("seq") is not None:
                event_id = "ledger-seq:%s" % folded["seq"]
            old_row = {
                "id": mid,
                "superseded": True,
                "superseded_by": new_id,
                "event_id": event_id,
            }
            break
    evidence_receipt["event_id"] = event_id
    try:
        snap = store.take_stable_snapshot()
        rows = snap.get("rows") or {}
    except Exception:
        rows = {}
    new_snap = rows.get(new_id)
    if new_snap is not None:
        new_meta = new_snap.get("meta") or {}
        new_row = {
            "id": new_id,
            "live": bool(new_snap.get("live")),
            "turn_id": new_meta.get("turn_id"),
            "event_id": event_id,
        }
    return evidence_receipt, old_row, new_row
