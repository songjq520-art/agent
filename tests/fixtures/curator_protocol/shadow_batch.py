"""隔离影子批次：只读复核、原子保存、人工差异对账；无执行能力。"""
from collections.abc import Mapping
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

from host_review import Proposal, review_proposal


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
