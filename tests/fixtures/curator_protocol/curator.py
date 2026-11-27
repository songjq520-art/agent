"""隔离的影子候选评估器；不提供生产执行或授权接口。"""
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class Candidate:
    memory_id: str
    content_version: str
    protection_version: str
    policy_version: str
    core: bool
    protection: str
    source_verified: bool
    judge_status: str


def assess(candidate: Candidate) -> str:
    """只评估影子候选；所有返回值都不代表执行许可。"""
    if not all(isinstance(v, str) and v.strip() for v in (
        candidate.memory_id, candidate.content_version,
        candidate.protection_version, candidate.policy_version,
    )):
        return 'hold_missing_version'
    if candidate.core is not True:
        return 'skip_not_core' if candidate.core is False else 'hold_invalid_core'
    if candidate.protection != 'ordinary':
        return 'hold_protected_or_unknown'
    if candidate.source_verified is not True:
        return 'hold_unverified_source'
    if candidate.judge_status != 'valid':
        return 'hold_judge_' + (candidate.judge_status if candidate.judge_status in {
            'unreachable', 'budget_exhausted', 'malformed_or_timeout', 'uncertain'
        } else 'unknown')
    return 'shadow_candidate_only'


class ShadowJournal:
    """本模块独立影子日志，不是十二正式账本，不写原文或核心状态。"""
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'shadow.sqlite3'
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS observations (
                observation_key TEXT PRIMARY KEY,
                evidence TEXT NOT NULL,
                result TEXT NOT NULL)''')

    def record(self, key: str, candidate: Candidate) -> str:
        if not isinstance(key, str) or not key.strip():
            raise ValueError('observation key required')
        evidence = json.dumps(candidate.__dict__, sort_keys=True, ensure_ascii=False)
        result = assess(candidate)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT evidence, result FROM observations WHERE observation_key=?',
                             (key,)).fetchone()
            if old is not None:
                if old != (evidence, result):
                    raise ValueError('observation key reused with different evidence')
                return old[1]
            db.execute('INSERT INTO observations VALUES (?, ?, ?)', (key, evidence, result))
        return result

    def rows(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute('SELECT observation_key, evidence, result FROM observations '
                              'ORDER BY observation_key').fetchall()


def execute(*args, **kwargs):
    raise PermissionError('execution disabled: locked contract and host adapter required')
