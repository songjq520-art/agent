# tools/audit.py（第一批工具共用的操作审计）
# 作用：所有工具的重要操作都记一笔（GPT 的建议）：谁发起的、什么工具、
# 什么操作、哪个路径、确认过没有、成功没有、改前改后的哈希。写成 JSONL，
# 一行一条，删了改了什么以后都能查到。
# 注意：审计日志本身用 tools/permissions.py 挡着，文件工具不能直接改写它。

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


# 密钥特征（审计日志里绝不能出现密钥原文）：Bearer token、sk- 开头的 Key、
# 各家 Key 前缀（tvly-、as_sk_）、key=/token= 这类赋值。
# 注意：SHA-256 这类哈希不是密钥，不打码；下面的规则都避开了纯十六进制哈希。
_REDACTED = "***已打码***"
_SECRET_RES = [
    re.compile(r"(Bearer\s+)[^\s\"']+", re.IGNORECASE),
    re.compile(r"\b(?:sk|tvly)-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bas_sk_[A-Za-z0-9_\-]{8,}\b"),
    re.compile(
        r"((?:api[_-]?key|apikey|token|password|passwd|secret|private[_-]?key)\s*[:=]\s*[\"']?)[^\"'\s,}]{6,}",
        re.IGNORECASE,
    ),
]


def _redact_text(text):
    """给一段文字里的密钥打码：只留个样子，不留原文。"""
    if not isinstance(text, str):
        return text
    out = text
    for pattern in _SECRET_RES:
        out = pattern.sub(
            lambda m: (m.group(1) + _REDACTED) if m.lastindex else _REDACTED, out
        )
    return out


def _redact_value(value):
    """递归给审计内容打码：字符串直接打，字典/列表逐项打，其他原样。"""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def sha256_bytes(data):
    """算一段 bytes 的 SHA-256（十六进制）。"""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    """算一个文件的 SHA-256（十六进制），文件不存在返回 None。"""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditLog:
    """操作审计日志：追加写 JSONL，线程安全。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        tool,
        action,
        path=None,
        actor="agent",
        confirmed=False,
        success=True,
        reason="",
        before_sha256=None,
        after_sha256=None,
        extra=None,
        confirmation_source="",
        origin="",
    ):
        """记一条审计事件。extra 是额外的结构化信息（字典）。
        confirmation_source 是确认的来源（"user-scoped" / "caller-attested" /
        ""），origin 是操作内容的来历（"user" / "agent"）。
        密钥永不进日志：reason、path、extra 在落盘前先打码。"""
        event = {
            "ts": _now_iso(),
            "actor": actor,
            "tool": tool,
            "action": action,
            "path": _redact_text(str(path)) if path is not None else None,
            "confirmed": bool(confirmed),
            "confirmation_source": confirmation_source or "",
            "origin": origin or "",
            "success": bool(success),
            "reason": _redact_text(reason or ""),
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "extra": _redact_value(extra or {}),
        }
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return event

    def read(self, limit=100):
        """读最近 limit 条事件（按时间顺序，最新的在最后）。"""
        if not self.path.is_file():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        events = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def count(self):
        """一共有多少条事件。"""
        if not self.path.is_file():
            return 0
        with self._lock:
            return sum(1 for _ in self.path.read_text(encoding="utf-8").splitlines())
