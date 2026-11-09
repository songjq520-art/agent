# tests/test_tools_files.py（第一批工具：文件测试）
# 作用：验证 tools/files.py —— 固定工作区、真实路径防穿越、密钥黑名单、
# .env 只写不读、读分页、写默认不覆盖、改要求旧串唯一、删默认进回收站、
# 写前备份、原子写、重要操作先确认、操作审计。
# 全部离线，只碰临时目录，不产生费用。

import os

import pytest

from tools.audit import AuditLog
from tools.errors import NeedsConfirmation, PermissionDenied, ToolError
from tools.files import Workspace


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root, audit_log=AuditLog(root / ".audit" / "audit.log"))


def test_write_read_round_trip(ws):
    out = ws.write_text("notes/a.txt", "你好\n世界\n", confirmed=True)
    assert out["sha256"]
    read = ws.read_text("notes/a.txt")
    assert read["lines"] == ["你好", "世界"]
    assert read["total_lines"] == 2
    assert read["truncated"] is False


def test_read_paging(ws):
    ws.write_text("big.txt", "\n".join(f"line{i}" for i in range(10)), confirmed=True)
    page = ws.read_text("big.txt", offset=3, limit=4)
    assert page["lines"] == ["line3", "line4", "line5", "line6"]
    assert page["total_lines"] == 10
    assert page["truncated"] is True


def test_write_no_overwrite_by_default(ws):
    ws.write_text("a.txt", "one", confirmed=True)
    with pytest.raises(ToolError):
        ws.write_text("a.txt", "two", confirmed=True)
    assert ws.read_text("a.txt")["lines"] == ["one"]


def test_overwrite_backups_old_content(ws):
    ws.write_text("a.txt", "one", confirmed=True)
    ws.write_text("a.txt", "two", overwrite=True, confirmed=True)
    assert ws.read_text("a.txt")["lines"] == ["two"]
    backups = list((ws.root / ".backups").glob("a.txt_*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "one"


def test_edit_requires_unique_old_string(ws):
    ws.write_text("a.txt", "foo bar foo", confirmed=True)
    with pytest.raises(ToolError):
        ws.edit_text("a.txt", "foo", "baz", confirmed=True)
    with pytest.raises(ToolError):
        ws.edit_text("a.txt", "missing", "baz", confirmed=True)
    ws.edit_text("a.txt", "foo bar foo", "baz", confirmed=True)
    assert ws.read_text("a.txt")["lines"] == ["baz"]


def test_delete_goes_to_trash_by_default(ws):
    ws.write_text("a.txt", "x", confirmed=True)
    ws.delete("a.txt", confirmed=True)
    assert not (ws.root / "a.txt").exists()
    trashed = list((ws.root / ".trash").glob("a.txt_*"))
    assert len(trashed) == 1
    assert trashed[0].read_text(encoding="utf-8") == "x"


def test_delete_permanent(ws):
    ws.write_text("a.txt", "x", confirmed=True)
    ws.delete("a.txt", permanent=True, confirmed=True)
    assert not (ws.root / "a.txt").exists()
    assert list((ws.root / ".trash").glob("*")) == []


def test_delete_directory_needs_recursive(ws):
    (ws.root / "d").mkdir(parents=True)
    ws.write_text("d/a.txt", "x", confirmed=True)
    with pytest.raises(ToolError):
        ws.delete("d", confirmed=True)
    ws.delete("d", recursive=True, confirmed=True)
    assert not (ws.root / "d").exists()


def test_traversal_rejected(ws):
    with pytest.raises(PermissionDenied):
        ws.write_text("../escape.txt", "x", confirmed=True)
    with pytest.raises(PermissionDenied):
        ws.read_text("../escape.txt")


def test_env_read_blocked_write_allowed(ws):
    with pytest.raises(PermissionDenied):
        ws.read_text(".env")
    ws.write_text(".env", "A=1\n", confirmed=True)
    with pytest.raises(PermissionDenied):
        ws.read_text(".env")


def test_secret_files_blocked(ws):
    with pytest.raises(PermissionDenied):
        ws.write_text("id_rsa", "x", confirmed=True)
    with pytest.raises(PermissionDenied):
        ws.write_text("k.pem", "x", confirmed=True)


def test_confirmation_gate(ws):
    with pytest.raises(NeedsConfirmation):
        ws.write_text("a.txt", "x")
    with pytest.raises(NeedsConfirmation):
        ws.write_text("a.txt", "x", confirmed=False)
    ws.write_text("a.txt", "x", confirmed=True)
    assert ws.read_text("a.txt")["lines"] == ["x"]


def test_audit_records_write_with_hashes(ws):
    ws.write_text("a.txt", "hello", confirmed=True)
    events = ws.audit_log.read()
    assert len(events) == 1
    event = events[0]
    assert event["tool"] == "files"
    assert event["action"] == "write"
    assert event["success"] is True
    assert event["confirmed"] is True
    assert event["after_sha256"]


def test_list_dir_and_file_info(ws):
    ws.write_text("sub/a.txt", "x", confirmed=True)
    ws.write_text("sub/b.txt", "yy", confirmed=True)
    listing = ws.list_dir("sub")
    assert {e["name"] for e in listing["entries"]} == {"a.txt", "b.txt"}
    info = ws.file_info("sub/a.txt")
    assert info["exists"] is True
    assert info["size"] == 1
    assert ws.file_info("nope.txt")["exists"] is False


# ---------- 多场景补充（BOOS 2026-11-10 要求：不只测正常路） ----------


def test_atomic_write_failure_keeps_original(ws, monkeypatch):
    # 原子写：改名那一步失败，原文件必须一字不动，也不能留下临时文件。
    ws.write_text("a.txt", "original", confirmed=True)
    real_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("模拟磁盘失败")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(ToolError):
        ws.write_text("a.txt", "new content", overwrite=True, confirmed=True)
    monkeypatch.setattr(os, "replace", real_replace)
    assert ws.read_text("a.txt")["lines"] == ["original"]
    assert not list((ws.root).glob(".tmp-*"))  # 没留下临时文件
