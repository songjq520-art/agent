# tools/cli.py（第一批工具的命令行入口）
# 作用：不用改 main.py、不用接聊天循环，也能直接试这批工具。
#   python -m tools.cli search "今天天气怎么样"
#   python -m tools.cli search "纳斯达克最新收盘" --kind structured --cross-check
#   python -m tools.cli file read notes/a.txt
#   python -m tools.cli file list sub --recursive
#   python -m tools.cli file write notes/a.txt --content "你好" --yes
#   python -m tools.cli file edit notes/a.txt --old "你好" --new "你好呀" --yes
#   python -m tools.cli file delete notes/old.txt --yes
#   python -m tools.cli run -- python -m pytest tests/ -q
#   python -m tools.cli cleanup scan D:/下载 --kind downloads --out preview.json
#   python -m tools.cli cleanup delete --preview preview.json --categories temp,logs --yes
#
# 工作区目录：--workspace 指定，或 TWELVE_WORKSPACE 环境变量，默认当前目录。
# Key 只从环境变量读（ANYSEARCH_API_KEY、TAVILY_API_KEY），绝不在命令行传 Key
# （会进 shell 历史记录）；Parallel 不用 Key。
# 写/改/删和 Tier 1 命令默认要确认，--yes 表示“我确认过了”。

import argparse
import json
import os
import sys
from pathlib import Path

from tools.audit import AuditLog
from tools.cleanup import build_preview, delete_preview, format_summary, scan
from tools.commands import CommandRunner
from tools.errors import NeedsConfirmation, PermissionDenied, ToolError
from tools.files import Workspace
from tools.search import SearchService


def _workspace_root(args):
    root = args.workspace or os.environ.get("TWELVE_WORKSPACE") or os.getcwd()
    return Path(root)


def _print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _fail(message, code=1):
    print(f"失败：{message}", file=sys.stderr)
    raise SystemExit(code)


def cmd_search(args):
    service = SearchService()
    try:
        outcome = service.search(
            args.query,
            kind=args.kind,
            max_results=args.max_results,
            cross_check=args.cross_check,
            force_provider=args.provider,
        )
    except ToolError as exc:
        _fail(str(exc))
    _print_json(outcome.to_dict())
    if outcome.status == "failed":
        raise SystemExit(1)


def cmd_file_read(args):
    workspace = Workspace(_workspace_root(args))
    try:
        result = workspace.read_text(args.path, offset=args.offset, limit=args.limit)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)


def cmd_file_list(args):
    workspace = Workspace(_workspace_root(args))
    try:
        result = workspace.list_dir(args.path, recursive=args.recursive)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)


def cmd_file_write(args):
    workspace = Workspace(_workspace_root(args))
    content = args.content
    if content is None:
        content = sys.stdin.read()
    try:
        result = workspace.write_text(
            args.path, content, overwrite=args.overwrite, confirmed=args.yes
        )
    except NeedsConfirmation as exc:
        _fail(f"{exc.description}（加 --yes 表示确认）", code=2)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)


def cmd_file_edit(args):
    workspace = Workspace(_workspace_root(args))
    try:
        result = workspace.edit_text(args.path, args.old, args.new, confirmed=args.yes)
    except NeedsConfirmation as exc:
        _fail(f"{exc.description}（加 --yes 表示确认）", code=2)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)


def cmd_file_delete(args):
    workspace = Workspace(_workspace_root(args))
    try:
        result = workspace.delete(
            args.path,
            permanent=args.permanent,
            recursive=args.recursive,
            confirmed=args.yes,
        )
    except NeedsConfirmation as exc:
        _fail(f"{exc.description}（加 --yes 表示确认）", code=2)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)


def cmd_run(args):
    workspace = _workspace_root(args)
    runner = CommandRunner(workspace)
    try:
        result = runner.run(args.argv, timeout=args.timeout, confirmed=args.yes)
    except NeedsConfirmation as exc:
        _fail(f"{exc.description}（加 --yes 表示确认）", code=2)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)
    if result["returncode"] not in (0, None):
        raise SystemExit(result["returncode"])


def cmd_cleanup_scan(args):
    try:
        result = scan(args.root, root_kind=args.kind, large_threshold_mb=args.large_mb)
    except ToolError as exc:
        _fail(str(exc))
    preview = build_preview(result)
    print(format_summary(preview))
    if args.out:
        Path(args.out).write_text(
            json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n预览已存到 {args.out}，确认要删的种类后用 cleanup delete 删。")


def cmd_cleanup_delete(args):
    preview_path = Path(args.preview)
    if not preview_path.is_file():
        _fail(f"预览文件不存在：{args.preview}（先跑 cleanup scan --out 存一份）")
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    if not args.yes:
        _fail("删文件必须加 --yes 表示你确认过了（删的是彻底删除，低置信度的先进回收站）", code=2)
    trash_dir = args.trash or (Path(os.environ.get("TWELVE_WORKSPACE", os.getcwd())) / ".trash")
    audit_log = AuditLog(Path(os.environ.get("TWELVE_WORKSPACE", os.getcwd())) / ".audit" / "audit.log")
    report = delete_preview(
        preview,
        [c.strip() for c in args.categories.split(",") if c.strip()],
        trash_dir=trash_dir,
        audit_log=audit_log,
    )
    _print_json(report)


def build_parser():
    parser = argparse.ArgumentParser(prog="tools.cli", description="十二第一批工具")
    parser.add_argument("--workspace", default=None, help="工作区目录（默认 TWELVE_WORKSPACE 或当前目录）")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="上网搜索")
    search.add_argument("query")
    search.add_argument("--kind", choices=["normal", "structured"], default="normal")
    search.add_argument("--provider", choices=["anysearch", "tavily", "parallel"], default=None)
    search.add_argument("--max-results", type=int, default=5)
    search.add_argument("--cross-check", action="store_true")
    search.set_defaults(func=cmd_search)

    file_parser = sub.add_parser("file", help="读写文件")
    file_sub = file_parser.add_subparsers(dest="file_command", required=True)
    read = file_sub.add_parser("read")
    read.add_argument("path")
    read.add_argument("--offset", type=int, default=0)
    read.add_argument("--limit", type=int, default=2000)
    read.set_defaults(func=cmd_file_read)
    list_parser = file_sub.add_parser("list")
    list_parser.add_argument("path", nargs="?", default=".")
    list_parser.add_argument("--recursive", action="store_true")
    list_parser.set_defaults(func=cmd_file_list)
    write = file_sub.add_parser("write")
    write.add_argument("path")
    write.add_argument("--content", default=None)
    write.add_argument("--overwrite", action="store_true")
    write.add_argument("--yes", action="store_true")
    write.set_defaults(func=cmd_file_write)
    edit = file_sub.add_parser("edit")
    edit.add_argument("path")
    edit.add_argument("--old", required=True)
    edit.add_argument("--new", required=True)
    edit.add_argument("--yes", action="store_true")
    edit.set_defaults(func=cmd_file_edit)
    delete = file_sub.add_parser("delete")
    delete.add_argument("path")
    delete.add_argument("--permanent", action="store_true")
    delete.add_argument("--recursive", action="store_true")
    delete.add_argument("--yes", action="store_true")
    delete.set_defaults(func=cmd_file_delete)

    run = sub.add_parser("run", help="跑命令（结构化调用，命令放 -- 后面）")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.add_argument("--timeout", type=int, default=60)
    run.add_argument("--yes", action="store_true")
    run.set_defaults(func=cmd_run)

    cleanup = sub.add_parser("cleanup", help="清理垃圾")
    cleanup_sub = cleanup.add_subparsers(dest="cleanup_command", required=True)
    scan_parser = cleanup_sub.add_parser("scan")
    scan_parser.add_argument("root")
    scan_parser.add_argument(
        "--kind",
        choices=["temp", "downloads", "recycle_bin", "browser_cache", "other"],
        default="other",
    )
    scan_parser.add_argument("--large-mb", type=int, default=500)
    scan_parser.add_argument("--out", default=None)
    scan_parser.set_defaults(func=cmd_cleanup_scan)
    delete_parser = cleanup_sub.add_parser("delete")
    delete_parser.add_argument("--preview", required=True)
    delete_parser.add_argument("--categories", required=True, help="逗号分隔，比如 temp,logs")
    delete_parser.add_argument("--trash", default=None)
    delete_parser.add_argument("--yes", action="store_true")
    delete_parser.set_defaults(func=cmd_cleanup_delete)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]
    args.func(args)


if __name__ == "__main__":
    main()
