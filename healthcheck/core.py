"""配置、接口、末端脱敏和不可覆盖的逐事件轨迹。仅使用标准库。"""
from __future__ import annotations
import dataclasses
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import uuid
from typing import Any, Protocol

DEFAULT_MODE = "proposal"
RESULTS = {"applied", "still_failed", "gate_rejected", "no_candidate", "not_applied"}
SECRET_FIELD = re.compile(r"secret|password|passwd|token|authorization|credential|api.?key|密钥|密码|凭据", re.I)
SECRET_VALUE = re.compile(r"(?:[Bb][Ee][Aa][Rr][Ee][Rr]\s+[^\s,;]+|(?:sk|pk|ghp|github_pat|AKIA)[-_A-Za-z0-9]{12,}|(?<![A-Za-z0-9_./-])(?=[A-Za-z0-9+_=/-]{40,}(?![A-Za-z0-9_./-]))(?=[A-Za-z0-9+_=/-]*[A-Z])(?=[A-Za-z0-9+_=/-]*[a-z])(?=[A-Za-z0-9+_=/-]*[0-9])[A-Za-z0-9+_=/-]{40,})")
ASSIGNMENT = re.compile(r"(?im)((?:api[_-]?key|password|secret|token|密钥|密码)\s*[:=]\s*)([^\s,;]+)")


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def redact(value: Any) -> Any:
    """证据的唯一出门口；字段和字符串形状同时处理。"""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): "[已脱敏]" if SECRET_FIELD.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return SECRET_VALUE.sub("[已脱敏]", ASSIGNMENT.sub(r"\1[已脱敏]", value))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


def emit(path: Path, data: Any):
    """所有持久化证据必须经过此处；独占新文件，禁止覆盖/删除。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(redact(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


class Trace:
    def __init__(self, directory: Path):
        self.directory = directory
        self.entries = []

    def event(self, step: str, **data):
        entry = redact({"step": step, "time": utcnow().isoformat(), **data})
        emit(self.directory / f"{uuid.uuid4().hex}.json", entry)
        self.entries.append(entry)
        return entry


@dataclasses.dataclass
class Finding:
    item: str
    status: str = "正常"
    severity: str = "info"
    details: Any = dataclasses.field(default_factory=dict)


class Adapter(Protocol):
    """十二提供的可信适配器；不由候选补丁实现。详见 HANDOFF.md。"""
    def memory(self, sample_size: int) -> dict: ...
    def services(self) -> dict: ...
    def models(self) -> dict: ...
    def resources(self) -> dict: ...
    def backups(self) -> dict: ...
    def schedule(self) -> dict: ...
    def approve(self, question_id: str, issue_id: str, action: str) -> bool: ...
    def isolate_dirty(self, repo: Path, status: str) -> bool: ...
    def begin_safe(self, kind: str, issue_id: str) -> Any: ...
    def rollback_safe(self, snapshot: Any) -> bool: ...
    def safe_action(self, kind: str, issue_id: str) -> dict: ...
    def verify_action(self, kind: str, issue_id: str) -> dict: ...
    def weekly_lease(self, key: str) -> bool: ...


class Unconnected:
    def __getattr__(self, name):
        if name in {"approve", "isolate_dirty", "weekly_lease"}:
            return lambda *a, **k: False
        return lambda *a, **k: {"ok": False, "connected": False, "reason": "十二适配器尚未接通"}


@dataclasses.dataclass
class Config:
    repo: Path
    archive: Path
    test_command: list[str]
    allowed_commands: list[list[str]]
    full_suite: bool
    max_seconds: float = 120
    max_steps: int = 8
    max_cost: float = 0.1
    memory_sample: int = 20
    disk_min_bytes: int = 1073741824
    secret_env_names: list[str] = dataclasses.field(default_factory=list)
    permission_paths: list[str] = dataclasses.field(default_factory=list)
    max_attempts: int = 2
    reflection_rounds: int = 3
    max_output_bytes: int = 1048576
    repair_paths: list[str] = dataclasses.field(default_factory=list)
    protected_paths: list[str] = dataclasses.field(default_factory=list)
    weekly_interval_seconds: float = 7 * 24 * 3600

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key in ("repo", "archive"):
            raw[key] = (path.parent / raw[key]).resolve()
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        if not self.repo.is_dir():
            raise ValueError("仓库目录不存在")
        if self.memory_sample != 20:
            raise ValueError("记忆每次必须抽查20条")
        if not 1 <= self.max_attempts <= 2 or not 1 <= self.reflection_rounds <= 3:
            raise ValueError("尝试最多2次，反思最多3轮")
        for value in (self.max_seconds, self.max_steps, self.max_cost,
                      self.max_output_bytes, self.weekly_interval_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("预算必须是有限正数")
        if not isinstance(self.full_suite, bool):
            raise ValueError("full_suite 必须是布尔值")
        if self.archive == self.repo or self.archive.is_relative_to(self.repo):
            raise ValueError("存档目录必须位于目标仓库之外，防止副本递归")
        for path in self.permission_paths + self.repair_paths + self.protected_paths:
            confined(self.repo, path)


def confined(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or ".git" in path.parts:
        raise ValueError("路径必须在仓库内，禁止访问版本库元数据")
    target = root / path
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError("禁止符号链接路径")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("路径越界")
    return target
