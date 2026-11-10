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
#   （清理流程必须用带钥匙的 HMAC 预览：扫描和删除用同一把钥匙，
#   环境变量 TWELVE_CLEANUP_KEY 优先，其次读项目根目录 .cleanup/cleanup_key，
#   都没有时第一次扫描自动生成一把存进去；普通 SHA-256 预览删除时也拒绝。）
#   （搜中文时：设了 DEEPSEEK_API_KEY 就先把查询翻成英文再搜
#   ——AnySearch 中文不准；没设就打印一句提示。）
#
# 工作区目录：--workspace 指定，或 TWELVE_WORKSPACE 环境变量，默认当前目录。
# 搜索的 Key 只从环境变量读（ANYSEARCH_API_KEY、TAVILY_API_KEY、DEEPSEEK_API_KEY），
# 清理钥匙优先读 TWELVE_CLEANUP_KEY 环境变量，其次读项目根目录
# .cleanup/cleanup_key 文件（第一次扫描自动生成，已加进 .gitignore）；
# 绝不在命令行传 Key（会进 shell 历史记录）；Parallel 不用 Key。
# 写/改/删和 Tier 1 命令默认要确认，--yes 表示“我确认过了”。

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

import requests

from tools.audit import AuditLog
from tools.cleanup import build_preview, delete_preview, format_summary, scan
from tools.commands import CommandRunner
from tools.errors import NeedsConfirmation, PermissionDenied, ToolError
from tools.files import Workspace
from tools.permissions import confirmation_scope_for_command, confirmation_scope_for_file
from tools.search import SearchService, _contains_cjk


def _confirmation(action, path_or_argv, yes):
    """--yes 的确认：带作用域的确认，scope 就是本次操作的精确描述。
    命令行是 BOOS 当场敲的 --yes，针对的就是这一条命令/这一个文件，
    所以直接按这个格式写 scope；没加 --yes 就是没确认。"""
    if not yes:
        return False
    if action == "run":
        scope = confirmation_scope_for_command(path_or_argv)
    else:
        scope = confirmation_scope_for_file(action, path_or_argv)
    return {"by": "cli", "scope": scope}


def _workspace_root(args):
    root = args.workspace or os.environ.get("TWELVE_WORKSPACE") or os.getcwd()
    return Path(root)


def _print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _fail(message, code=1):
    print(f"失败：{message}", file=sys.stderr)
    raise SystemExit(code)


def _deepseek_translate(text, target):
    """CLI 用的 DeepSeek 翻译：只翻查询/标题/摘要。Key 从 DEEPSEEK_API_KEY
    环境变量读；翻失败抛错，调用方（SearchService）会回落原文，不影响搜索本身。
    成本：一次搜索约多花 $0.0002，一个月最坏约 $0.5–1。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ToolError("没有 DEEPSEEK_API_KEY，无法翻译")
    direction = "英文" if target == "en" else "中文"
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-flash",
                "messages": [
                    {"role": "system", "content": "你是一个翻译工具。只返回译文，不要解释，不要加引号。"},
                    {"role": "user", "content": f"把下面的文字翻译成{direction}：\n{text}"},
                ],
                "temperature": 0,
                "max_tokens": 1024,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        raise ToolError(f"翻译调用失败：{exc}") from exc
    if response.status_code != 200:
        raise ToolError(f"翻译调用失败：HTTP {response.status_code}")
    try:
        translated = response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise ToolError("翻译返回解析失败") from exc
    if not translated:
        raise ToolError("翻译返回为空")
    return translated


def cmd_search(args):
    # 设了 DEEPSEEK_API_KEY 就接上翻译器：中文查询先翻成英文再搜
    #（AnySearch 中文不准，直接搜中文会返回不相干的法院文书），标题和
    # 摘要前 200 字再翻回中文；没设 Key 就不翻译，但中文查询要先提醒
    # 一句，别让 BOOS 以为搜出来的就是对的。提示打 stderr，stdout
    # 只留 JSON，别把 JSON 打坏。
    translator = None
    if os.environ.get("DEEPSEEK_API_KEY"):
        translator = _deepseek_translate
    elif _contains_cjk(args.query):
        print(
            "提示：没设 DEEPSEEK_API_KEY，中文查询不走翻译，"
            "AnySearch 中文结果可能不准（会返回不相干的法院文书）；"
            "设了 Key 就会先翻成英文再搜。",
            file=sys.stderr,
        )
    service = SearchService(translator=translator)
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
            args.path, content, overwrite=args.overwrite,
            confirmed=_confirmation("write", args.path, args.yes),
        )
    except NeedsConfirmation as exc:
        _fail(f"{exc.description}（加 --yes 表示确认）", code=2)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)


def cmd_file_edit(args):
    workspace = Workspace(_workspace_root(args))
    try:
        result = workspace.edit_text(
            args.path, args.old, args.new,
            confirmed=_confirmation("edit", args.path, args.yes),
        )
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
            confirmed=_confirmation("delete", args.path, args.yes),
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
        result = runner.run(
            args.argv, timeout=args.timeout,
            confirmed=_confirmation("run", args.argv, args.yes),
        )
    except NeedsConfirmation as exc:
        _fail(f"{exc.description}（加 --yes 表示确认）", code=2)
    except (ToolError, PermissionDenied) as exc:
        _fail(str(exc))
    _print_json(result)
    if result["returncode"] not in (0, None):
        raise SystemExit(result["returncode"])


def _cleanup_key_file_path():
    """清理钥匙文件的长期住处：项目根目录下的 .cleanup/cleanup_key
    （按 cli.py 所在位置算，不按当前工作目录算，换个目录跑也找得到）。
    测试时可用 TWELVE_CLEANUP_KEY_FILE 指到别处，别碰真的钥匙文件。"""
    override = os.environ.get("TWELVE_CLEANUP_KEY_FILE")
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parent.parent
    return project_root / ".cleanup" / "cleanup_key"


def _load_cleanup_key():
    """读清理钥匙（不生成）：环境变量优先，其次钥匙文件；都没有返回 None。
    返回 (key, 来源)，来源是 "env" 或 "file"。"""
    env_key = os.environ.get("TWELVE_CLEANUP_KEY")
    if env_key:
        return env_key, "env"
    key_file = _cleanup_key_file_path()
    try:
        file_key = key_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not file_key:
        return None
    return file_key, "file"


def _cleanup_key():
    """拿一把清理钥匙：环境变量优先，其次钥匙文件；都没有就现生成一把
    随机钥匙存进钥匙文件里，以后扫描和删除一直用它。
    返回 (key, 来源)，来源是 "env" / "file" / "generated"。"""
    loaded = _load_cleanup_key()
    if loaded is not None:
        return loaded
    key = secrets.token_hex(16)
    key_file = _cleanup_key_file_path()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key, encoding="utf-8")
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    return key, "generated"


def cmd_cleanup_scan(args):
    # 扫描要生成可删除的预览，必须有钥匙：没钥匙生成的只是普通 SHA-256 预览，
    # 改了清单重算一个就能过，删的时候会被拒绝。钥匙的找法（按顺序）：
    # TWELVE_CLEANUP_KEY 环境变量 → 项目根目录 .cleanup/cleanup_key 文件 →
    # 都没有就现生成一把随机钥匙存进那个文件，以后扫描和删除一直用它。
    # 钥匙绝不在命令行传（会进 shell 历史记录）；钥匙文件已加进 .gitignore，
    # 绝不进仓库。钥匙文件要是丢了，旧预览就验不过，只能重新扫描。
    key, source = _cleanup_key()
    if source == "generated":
        print(
            f"钥匙已自动生成并存到 {_cleanup_key_file_path()}，"
            "扫描和删除都用这把；别删这个文件，删了旧预览就验不过，只能重扫。"
        )
    try:
        result = scan(args.root, root_kind=args.kind, large_threshold_mb=args.large_mb)
    except ToolError as exc:
        _fail(str(exc))
    preview = build_preview(result, key=key)
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
    if not args.root:
        _fail("删文件必须加 --root 指明可信的扫描根目录（当初扫的那几个目录，可重复写），删除范围不从预览文件里读")
    # 删除必须用扫描时那把钥匙验证：环境变量优先，其次钥匙文件；都没
    # 有就拒绝——删除这一步不现生成钥匙（现生成的和扫描时那把对不上，
    # 验不过），请设成扫描时那把钥匙，或重新扫描。
    loaded = _load_cleanup_key()
    if loaded is None:
        _fail("删除必须用扫描时那把钥匙验证：既没设 TWELVE_CLEANUP_KEY，也没有 .cleanup/cleanup_key 文件；请设成扫描时那把钥匙，或重新扫描。")
    key, _source = loaded
    if preview.get("checksum_algo", "sha256") != "hmac-sha256":
        _fail("这份预览是普通 SHA-256 校验和，防不了篡改，拒绝删除；请用带钥匙的 cleanup scan 重新扫一份。")
    try:
        report = delete_preview(
            preview,
            [c.strip() for c in args.categories.split(",") if c.strip()],
            trash_dir=trash_dir,
            roots=args.root,
            key=key,
            audit_log=audit_log,
        )
    except ToolError as exc:
        _fail(str(exc))
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
    delete_parser.add_argument("--root", action="append", default=None,
                               help="可信的扫描根目录（当初扫的那几个目录），可重复写；删除范围以这个为准，不从预览文件里读")
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
