"""整理员协议生产件（立项 G4 观察窗生产接线，2026-11-27）。

她侧协议类（host_review.Proposal / review_proposal /
shadow_batch.BatchJournal）原本只存在于贵妃 curator-system，
宿主仓里只有 tests/fixtures/curator_protocol/ 的逐字夹具供
离线测试——生产侧没有可注入的真件，影子观察窗无从在真服务
里跑批。本模块是她侧这两份文件的**逐字移植**：判定逻辑、
落盘格式（shadow_batches.sqlite3 的 batches 表＋payload
JSON 形状）、幂等/拒收语义一字不改，宿主只提供运行位置，
不另立口径（G1/G2 共通纪律：不许宿主复刻她的判定另搞一
套）。协议要变，必须她侧先变、夹具与本文件跟着同步，谁都
不许在这边单方面改判定。

移植说明（与原件的全部差异仅此两处，均为装配层、不涉判定）：
  - shadow_batch 原件的 `from host_review import Proposal,
    review_proposal` 改为同模块直引（本文件把两份原件合一）；
  - 文件头与本说明是宿主加的；她侧各段自身的 docstring 原
    样保留——其中「不建立生产观察窗」一句说的是她侧隔离原
    件的定位，本宿主移植件的用途正是按同一协议在宿主侧落
    影子批次，判定与格式与原件完全一致。

correction_receipt 一并移植：它是 host_review 协议面的一部
分（G1 适配层 correction_evidence 拼的就是它的证据形状）。
"""

from collections.abc import Mapping
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3


# ===== 她侧 host_review.py 逐字移植 =====

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


# ===== 她侧 shadow_batch.py 逐字移植 =====

def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


class BatchJournal:
    """调用者必须指定隔离目录；不能传宿主账本或真库。"""
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'shadow_batches.sqlite3'
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS batches ('
                       'batch_id TEXT PRIMARY KEY, evidence_hash TEXT NOT NULL, '
                       'payload TEXT NOT NULL)')

    def collect(self, batch_id, snapshot_id, policy_version, items, *, protected, classify):
        """items为(Proposal, 离线快照)序列；不采集宿主，不建立生产观察窗。"""
        if not all(_text(v) for v in (batch_id, snapshot_id, policy_version)):
            raise ValueError('batch, snapshot and policy versions required')
        copied = deepcopy(list(items))
        evidence = []
        rows = []
        ids = set()
        for proposal, current in copied:
            if not isinstance(proposal, Proposal) or not _text(proposal.memory_id):
                raise ValueError('valid proposal required')
            if proposal.policy_version != policy_version:
                raise ValueError('mixed policy batch rejected')
            if proposal.memory_id in ids:
                raise ValueError('duplicate memory in batch')
            ids.add(proposal.memory_id)
            if not isinstance(current, Mapping):
                raise ValueError('offline snapshot required')
            evidence.append([proposal.__dict__, dict(current)])
        # 摘要仅作冲突检测，不当授权或正确性证据；不保存原文和metadata。
        digest = hashlib.sha256(_json([snapshot_id, policy_version, evidence]).encode()).hexdigest()
        for proposal, current in copied:
            result = review_proposal(proposal, current, protected=protected, classify=classify)
            rows.append(dict(memory_id=proposal.memory_id,
                             content_version=proposal.content_version,
                             protection_version=proposal.protection_version,
                             result=result))
        payload = dict(batch_id=batch_id, snapshot_id=snapshot_id,
                       policy_version=policy_version, rows=rows,
                       scope='isolated_shadow_only', execution_allowed=False)
        encoded = _json(payload)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT evidence_hash, payload FROM batches WHERE batch_id=?',
                             (batch_id,)).fetchone()
            if old is not None:
                if old != (digest, encoded):
                    raise ValueError('batch key reused with different evidence or decisions')
                return json.loads(old[1])
            db.execute('INSERT INTO batches VALUES (?, ?, ?)', (batch_id, digest, encoded))
        return payload

    def read(self, batch_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute('SELECT payload FROM batches WHERE batch_id=?', (batch_id,)).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return json.loads(row[0])

    def compare(self, batch_id, labels):
        """labels来自独立人工，值为candidate/hold/uncertain；不签发放行。"""
        batch = self.read(batch_id)
        if not isinstance(labels, Mapping):
            raise ValueError('independent labels required')
        known = {row['memory_id'] for row in batch['rows']}
        if set(labels) - known or any(v not in ('candidate', 'hold', 'uncertain') for v in labels.values()):
            raise ValueError('unknown memory or label')
        missing, uncertain, differences = [], [], []
        for row in batch['rows']:
            mid = row['memory_id']
            if mid not in labels:
                missing.append(mid)
            elif labels[mid] == 'uncertain':
                uncertain.append(mid)
            else:
                actual = 'candidate' if row['result'] == 'shadow_proposal_only' else 'hold'
                if labels[mid] != actual:
                    differences.append(dict(memory_id=mid, expected=labels[mid], actual=actual))
        return dict(batch_id=batch_id, policy_version=batch['policy_version'],
                    reviewed=len(labels), total=len(known), missing=missing,
                    uncertain=uncertain, differences=differences,
                    comparison_complete=not (missing or uncertain),
                    matched=not (missing or uncertain or differences),
                    execution_allowed=False, scope='isolated_shadow_only')
