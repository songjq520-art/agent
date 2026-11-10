# tests/test_tools_audit.py（第一批工具：审计日志测试）
# 作用：验证 tools/audit.py —— 操作记录的追加、读取、哈希工具。
# 审计日志是 GPT 建议加的：谁发起的、什么路径、什么操作、确认过没有、
# 成功没有、改前改后的哈希，删了改了什么以后都能查到。
# 全部离线，不联网、不产生费用。

import hashlib
import json
import threading

from tools.audit import AuditLog, sha256_bytes, sha256_file


def test_record_and_read(tmp_path):
    log = AuditLog(tmp_path / ".audit" / "audit.log")
    log.record(
        tool="files",
        action="write",
        path="a.txt",
        actor="agent",
        confirmed=True,
        success=True,
        before_sha256="abc",
        after_sha256="def",
    )
    events = log.read()
    assert len(events) == 1
    event = events[0]
    assert event["tool"] == "files"
    assert event["action"] == "write"
    assert event["path"] == "a.txt"
    assert event["confirmed"] is True
    assert event["success"] is True
    assert event["before_sha256"] == "abc"
    assert event["after_sha256"] == "def"
    assert "ts" in event


def test_read_limit_and_count(tmp_path):
    log = AuditLog(tmp_path / "audit.log")
    for i in range(5):
        log.record(tool="t", action=f"a{i}")
    assert len(log.read(limit=2)) == 2
    assert log.read(limit=2)[-1]["action"] == "a4"
    assert log.count() == 5


def test_sha_helpers(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()
    assert sha256_bytes(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_concurrent_writes_do_not_corrupt(tmp_path):
    log = AuditLog(tmp_path / "audit.log")

    def worker(n):
        for i in range(20):
            log.record(tool="t", action=f"w{n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    for line in lines:
        json.loads(line)  # 每行都必须是合法 JSON，不能写串行


# ---------- 本次安全修复新增 ----------

def test_audit_redacts_secrets(tmp_path):
    # 密钥、密码、token 一律不进审计日志：哪怕调用方不小心把它写进原因里。
    log = AuditLog(tmp_path / "audit.log")
    log.record(
        tool="search",
        action="query",
        path="",
        actor="agent",
        confirmed=False,
        success=False,
        reason="AnySearch 返回 401：api_key=as_sk_abcdef1234567890 不对",
        extra={"detail": "Authorization: Bearer tvly-xyz1234567890abcdef"},
    )
    text = (tmp_path / "audit.log").read_text(encoding="utf-8")
    assert "as_sk_abcdef1234567890" not in text
    assert "tvly-xyz1234567890abcdef" not in text
    assert "***" in text  # 打码标记还在


def test_audit_redacts_password_in_reason(tmp_path):
    log = AuditLog(tmp_path / "audit.log")
    log.record(
        tool="files",
        action="write",
        path=".env",
        actor="agent",
        confirmed=True,
        success=False,
        reason="写入失败：password=hunter2secret",
    )
    text = (tmp_path / "audit.log").read_text(encoding="utf-8")
    assert "hunter2secret" not in text


def test_audit_keeps_sha256_hashes(tmp_path):
    # SHA-256 哈希不是密钥原文，不用打码（删前删后核对要靠它）。
    log = AuditLog(tmp_path / "audit.log")
    digest = "a" * 64
    log.record(
        tool="files",
        action="write",
        path="a.txt",
        actor="agent",
        confirmed=True,
        success=True,
        before_sha256=digest,
        after_sha256=digest,
    )
    text = (tmp_path / "audit.log").read_text(encoding="utf-8")
    assert digest in text


def test_audit_records_confirmation_source_and_origin(tmp_path):
    log = AuditLog(tmp_path / "audit.log")
    log.record(
        tool="commands",
        action="run",
        path=".",
        actor="agent",
        confirmed=True,
        success=True,
        confirmation_source="user-scoped",
        origin="user",
    )
    events = log.read()
    assert events[0]["confirmation_source"] == "user-scoped"
    assert events[0]["origin"] == "user"
