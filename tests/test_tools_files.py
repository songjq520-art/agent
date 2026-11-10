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


def _symlink_or_skip(link, target):
    """建符号链接；建不起来就跳过，不算失败。
    Windows 普通用户没开开发者模式时，建符号链接报 WinError 1314
    （客户端没有所需的特权）——是这台机器缺权限，不是代码坏了。
    安全断言本身不动，能建链接的机器上照常测。"""
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"建符号链接要管理员权限或开发者模式：{exc}")


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


def test_backup_names_do_not_collide(ws):
    # 16 号回归：a/b.txt 和字面就叫 a__b.txt 的文件，以前备份名
    # 都拼成 a__b.txt_<时间戳>.bak，会撞在一起互相覆盖。
    ws.write_text("a/b.txt", "斜杠那个", confirmed=True)
    ws.write_text("a__b.txt", "下划线那个", confirmed=True)
    ws.write_text("a/b.txt", "斜杠那个 v2", overwrite=True, confirmed=True)
    ws.write_text("a__b.txt", "下划线那个 v2", overwrite=True, confirmed=True)
    backups = sorted((ws.root / ".backups").glob("*.bak"))
    assert len(backups) == 2
    contents = {b.read_text(encoding="utf-8") for b in backups}
    assert contents == {"斜杠那个", "下划线那个"}


def test_edit_refuses_oversize_file(ws, monkeypatch):
    # 10 号：改文件和读文件一样的 10MB 上限，别一次读爆内存。
    monkeypatch.setattr("tools.files._MAX_FILE_BYTES", 100)
    ws.write_text("big.txt", "x" * 200, confirmed=True)
    with pytest.raises(ToolError, match="文件太大"):
        ws.edit_text("big.txt", "x", "y", confirmed=True)


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


# ---------- 本次安全修复新增 ----------

def test_env_file_not_backed_up(ws):
    # .env 覆盖时不备份：旧内容绝不能躺在 .backups 里等着被人读出来。
    ws.write_text(".env", "TAVILY_API_KEY=tvly-fake-old\n", overwrite=True, confirmed=True)
    ws.write_text(".env", "TAVILY_API_KEY=tvly-fake-new\n", overwrite=True, confirmed=True)
    backups = list((ws.root / ".backups").glob("*")) if (ws.root / ".backups").exists() else []
    assert backups == []
    with pytest.raises(PermissionDenied):
        ws.read_text(".env")


def test_env_edit_does_not_back_up(ws):
    # 改 .env 也不备份：edit 走的也是同一份备份逻辑。
    ws.write_text(".env", "TAVILY_API_KEY=tvly-fake-old\n", confirmed=True)
    ws.edit_text(".env", "tvly-fake-old", "tvly-fake-new", confirmed=True)
    backups = list((ws.root / ".backups").glob("*")) if (ws.root / ".backups").exists() else []
    assert backups == []


def test_backup_named_like_env_cannot_be_read(ws):
    # 有人手动把 .env 的备份塞进 .backups：按它备份的原文件判断，照样读不了。
    backup_dir = ws.root / ".backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / ".env_20260101T000000000000.bak").write_text(
        "TAVILY_API_KEY=tvly-fake-leaked\n", encoding="utf-8"
    )
    with pytest.raises(PermissionDenied):
        ws.read_text(".backups/.env_20260101T000000000000.bak")


def test_symlink_to_env_cannot_be_read(ws):
    # 工作区里放一个指到 .env 的符号链接：读链接名就是读 .env，必须拦住。
    target = ws.root / ".env"
    target.write_text("TAVILY_API_KEY=tvly-fake-secret\n", encoding="utf-8")
    link = ws.root / "link_to_env"
    _symlink_or_skip(link, target)
    with pytest.raises(PermissionDenied):
        ws.read_text("link_to_env")


def test_symlink_to_secret_inside_workspace_cannot_be_read(ws):
    target = ws.root / "real_key.pem"
    target.write_text("FAKE-PEM-CONTENT\n", encoding="utf-8")
    link = ws.root / "innocent_name.txt"
    _symlink_or_skip(link, target)
    with pytest.raises(PermissionDenied):
        ws.read_text("innocent_name.txt")


def test_external_origin_write_refused(ws):
    # 外部文件/网页里读到的“写文件”指令不直接执行。
    with pytest.raises(PermissionDenied):
        ws.write_text("evil.txt", "x", origin="external")


def test_external_origin_delete_refused(ws):
    ws.write_text("keep.txt", "别删我", overwrite=True, confirmed=True)
    with pytest.raises(PermissionDenied):
        ws.delete("keep.txt", permanent=True, confirmed=True, origin="external")
    assert (ws.root / "keep.txt").exists()


def test_scoped_confirmation_rejects_wrong_file(ws):
    # 确认是给另一个文件的，顶不了这一次覆盖：必须重新问 BOOS。
    ws.write_text("a.txt", "旧内容", overwrite=True, confirmed=True)
    wrong_scope = {"by": "cli", "scope": "write:b.txt"}
    with pytest.raises(NeedsConfirmation):
        ws.write_text("a.txt", "新内容", overwrite=True, confirmed=wrong_scope)
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "旧内容"


def test_delete_protected_dirs_case_insensitive(ws):
    # Windows 上 .BACKUPS 就是 .backups：大写的重要目录本身一样不让删。
    for name in (".BACKUPS", ".Trash", ".AUDIT"):
        (ws.root / name).mkdir(parents=True, exist_ok=True)
        with pytest.raises(ToolError):
            ws.delete(name, recursive=True, confirmed=True)
        assert (ws.root / name).exists()


def test_recursive_permanent_delete_refuses_protected_inside(ws):
    # 3 号：递归彻底删目录，树里有 .env、id_rsa 时拒绝，报错里列出来是什么，
    # 目录和里面的文件一个都不少。
    (ws.root / "data" / "sub").mkdir(parents=True)
    ws.write_text("data/notes.txt", "hello", confirmed=True)
    (ws.root / "data" / ".env").write_text("K=v\n", encoding="utf-8")
    (ws.root / "data" / "sub" / "id_rsa").write_text("PRIVATE\n", encoding="utf-8")
    with pytest.raises(ToolError, match=".env"):
        ws.delete("data", permanent=True, recursive=True, confirmed=True)
    assert (ws.root / "data" / ".env").exists()
    assert (ws.root / "data" / "sub" / "id_rsa").exists()
    assert (ws.root / "data" / "notes.txt").exists()


def test_recursive_permanent_delete_clean_dir_ok(ws):
    # 3 号对照：树里没有秘密/受保护内容，递归彻底删正常走。
    (ws.root / "data" / "sub").mkdir(parents=True)
    ws.write_text("data/notes.txt", "hello", confirmed=True)
    ws.write_text("data/sub/more.txt", "more", confirmed=True)
    out = ws.delete("data", permanent=True, recursive=True, confirmed=True)
    assert out["protected_inside"] == []
    assert not (ws.root / "data").exists()


def test_recursive_delete_to_trash_notes_protected(ws):
    # 3 号：进回收站那路可恢复，允许继续，但结果和审计里注明含受保护内容。
    (ws.root / "data").mkdir(parents=True)
    ws.write_text("data/notes.txt", "hello", confirmed=True)
    (ws.root / "data" / ".env").write_text("K=v\n", encoding="utf-8")
    out = ws.delete("data", recursive=True, confirmed=True)
    assert not (ws.root / "data").exists()
    assert any(".env" in p for p in out["protected_inside"])
    trashed = list((ws.root / ".trash").glob("data_*"))
    assert len(trashed) == 1
    assert (trashed[0] / ".env").exists()  # 进了回收站，还能找回来
    events = ws.audit_log.read()
    assert any(".env" in str(e.get("extra", {}).get("protected_inside", [])) for e in events)


def test_delete_symlink_to_file_deletes_link_only(ws):
    # 5 号：删指着重要文件的链接，只删链接本身，目标文件完好。
    target = ws.root / "victim.txt"
    target.write_text("important\n", encoding="utf-8")
    link = ws.root / "link.txt"
    _symlink_or_skip(link, target)
    out = ws.delete("link.txt", permanent=True, confirmed=True)
    assert out["is_symlink"] is True
    assert not link.exists() and not link.is_symlink()
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "important\n"


def test_delete_symlink_to_dir_deletes_link_only(ws):
    # 5 号：递归删指着目录的链接，只删链接，目标目录和里面的文件完好，
    # 不管 recursive 是不是 True。
    realdir = ws.root / "realdir"
    realdir.mkdir()
    (realdir / "keep.txt").write_text("keep me\n", encoding="utf-8")
    link = ws.root / "linkdir"
    _symlink_or_skip(link, realdir)
    out = ws.delete("linkdir", permanent=True, recursive=True, confirmed=True)
    assert out["is_symlink"] is True
    assert not link.exists() and not link.is_symlink()
    assert realdir.exists()
    assert (realdir / "keep.txt").exists()


def test_delete_symlink_to_file_goes_to_trash(ws):
    # 5 号：进回收站那路也只移动链接本身，不移动目标。
    target = ws.root / "victim.txt"
    target.write_text("important\n", encoding="utf-8")
    link = ws.root / "link.txt"
    _symlink_or_skip(link, target)
    out = ws.delete("link.txt", confirmed=True)
    assert not link.exists() and not link.is_symlink()
    assert target.exists()
    trashed = list((ws.root / ".trash").glob("link.txt_*"))
    assert len(trashed) == 1
    assert trashed[0].is_symlink()  # 回收站里的是链接本身


def test_delete_symlink_to_outside_refused(ws, tmp_path):
    # 5 号：删指着工作区外文件的链接，拒绝，目标完好。
    outside = tmp_path / "outside.txt"
    outside.write_text("别动我\n", encoding="utf-8")
    link = ws.root / "link.txt"
    _symlink_or_skip(link, outside)
    with pytest.raises(PermissionDenied):
        ws.delete("link.txt", permanent=True, confirmed=True)
    assert outside.exists()
    assert outside.read_text(encoding="utf-8") == "别动我\n"


def test_delete_dangling_symlink(ws):
    # 5 号：目标已经没了的悬空链接，照样能删（不用先报“文件不存在”）。
    link = ws.root / "dangling.txt"
    _symlink_or_skip(link, ws.root / "gone.txt")
    ws.delete("dangling.txt", permanent=True, confirmed=True)
    assert not link.exists() and not link.is_symlink()
