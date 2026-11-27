"""宿主对接的只读复核层；无宿主写入、授权签发或生产执行能力。"""
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class Proposal:
    memory_id: str
    content_version: str
    protection_version: str
    policy_version: str


def review_proposal(proposal, current, *, protected, classify):
    """回调必须由宿主注入权威实现；current 须由宿主持锁捕获。

    只生成复核结果，不是执行期复核，不能拿结果直接执行。
    宿主尚未提供版本/保护迁移完成证据时，一律挂起。
    """
    if not isinstance(current, Mapping):
        return 'hold_missing_snapshot'
    for name in ('memory_id', 'content_version', 'protection_version', 'policy_version'):
        expected = getattr(proposal, name)
        if not isinstance(expected, str) or not expected.strip():
            return 'hold_missing_version'
        if current.get(name) != expected:
            return 'hold_stale_or_missing_version'
    if current.get('live') is not True:
        return 'hold_not_live'
    # 显式宿主证据，不把 metadata 缺标签当普通身份；不猜宿主字段映射。
    if current.get('protection_resolved') is not True:
        return 'hold_unresolved_protection'
    if current.get('readonly_projection') is not False:
        return 'hold_readonly_or_unknown'
    if current.get('pending_classification') is not False:
        return 'hold_pending_or_unknown'
    meta = current.get('metadata')
    if not isinstance(meta, Mapping):
        return 'hold_missing_metadata'
    try:
        # 独立副本，避免回调意外修改传入快照；不触及宿主活数据。
        from copy import deepcopy
        guarded = protected(deepcopy(dict(meta)))
        category = classify(deepcopy(dict(meta)), live=True)
    except Exception:
        return 'hold_host_guard_error'
    if guarded is not False:
        return 'hold_core_protected_or_unknown'
    if not isinstance(category, Mapping):
        return 'hold_bad_classification'
    if category.get('protection_tier') is not None:
        return 'hold_protected'
    if 'protection_tier' not in category:
        return 'hold_bad_classification'
    if category.get('member') is not True or category.get('occupies_seat') is not True:
        return 'hold_not_seated_core'
    return 'shadow_proposal_only'


def correction_receipt(receipt, old_row, new_row, *, turn_id):
    """只核对当轮纠正证据；不调用 correct_memory、不改记忆。

    行结构为待宿主映射的审计协议，不是现有 get_all 返回值的假定。
    """
    incomplete = {'complete': False, 'report_required': True,
                  'reason': 'correction_unverified'}
    if not all(isinstance(v, Mapping) for v in (receipt, old_row, new_row)):
        return incomplete
    if not isinstance(turn_id, str) or not turn_id.strip():
        return incomplete
    if receipt.get('turn_id') != turn_id or receipt.get('status') != 'updated':
        return incomplete
    old_id, new_id = receipt.get('old_id'), receipt.get('new_id')
    if not all(isinstance(v, str) and v.strip() for v in (old_id, new_id)) or old_id == new_id:
        return incomplete
    if old_row.get('id') != old_id or new_row.get('id') != new_id:
        return incomplete
    if old_row.get('superseded') is not True or old_row.get('superseded_by') != new_id:
        return incomplete
    if new_row.get('live') is not True or new_row.get('turn_id') != turn_id:
        return incomplete
    event = receipt.get('event_id')
    # 宿主须给同一已提交快照的事件关联证据，单有 updated 不够。
    if not isinstance(event, str) or not event.strip():
        return incomplete
    if old_row.get('event_id') != event or new_row.get('event_id') != event:
        return incomplete
    return {'complete': True, 'report_required': False,
            'reason': 'readback_verified_only'}
