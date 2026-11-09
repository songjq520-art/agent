# tools/files.py（第一批工具：读写文件）
# 作用：十二在固定工作区里读写文件的工具，安全边界按新方案的代码层做：
#   - 固定工作区目录 + 真实路径比对防穿越（../、绝对路径、符号链接逃逸）。
#   - 密钥黑名单（.git/、.ssh/、*.pem、*.key、id_rsa*）一律不可碰；
#     密钥/.env 的判断看解析后的真实路径，不只看用户输入的名字，
#     符号链接指到 .env 也拦得住；.backups/ 下的备份按它备份的原文件判断。
#   - .env 只写不读（Hermes 的做法）：能创建、能改，读不回里面的值；
#     密钥文件和 .env 不备份，备份、审计、报错里都不许带密钥内容。
#   - 读分页（一次最多 2,000 行），超大文件（>10MB）/ 二进制文件先拒绝。
#   - 写默认不覆盖；覆盖必须显式说要覆盖，且覆盖前先备份。
#   - 改要求旧串在文件里只出现一次，不唯一或找不到都失败。
#   - 删默认进回收站（工作区里的 .trash/），彻底删要显式说；删目录默认拒绝，
#     要删目录必须显式说递归。
#   - 写文件用“先写临时文件、再原子改名”，写一半失败原文件不动。
#   - 重要操作（写、改、删）初期默认先问 BOOS 确认：没传 confirmed=True
#     就抛 NeedsConfirmation（带一句人话说明），跑顺了再按路径放宽。
#   - 每次写/改/删都记审计（改前改后哈希），存在工作区 .audit/audit.log。
#
# 诚实说明：代码层防的是误操作和提示注入下的意外损坏，防不住存心绕过
# （终端能绕过文件工具的写保护），真正的隔离靠 OS 层，见 tools/README.md。

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from tools.audit import AuditLog, sha256_bytes, sha256_file
from tools.errors import NeedsConfirmation, PermissionDenied, ToolError
from tools.permissions import (
    _is_env_file,
    _is_secret_blocked,
    classify_path,
    confirmation_scope_for_file,
    resolve_confirmation,
    resolve_in_workspace,
)

_MAX_READ_LINES = 2000
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB：更大的文件先拒绝，别一次读爆上下文
_BACKUPS_TO_KEEP = 5


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


class Workspace:
    """固定工作区里的文件工具。root 之外的路径一律拒绝。"""

    def __init__(self, root, require_confirmation=True, audit_log=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.require_confirmation = require_confirmation
        self.audit_log = audit_log or AuditLog(self.root / ".audit" / "audit.log")

    # ---------- 内部工具 ----------

    def _check(self, rel_path, action):
        """权限检查：不允许就抛 PermissionDenied（带人话原因）。"""
        allowed, reason = classify_path(self.root, rel_path, action)
        if not allowed:
            raise PermissionDenied(reason)

    def _confirm(self, action, rel_path, description, confirmed, actor, origin="agent"):
        """重要操作确认关：初期默认要 BOOS 点头，没确认就拒绝并记审计。
        （写、改、删都要确认——跑顺了再按路径放宽。）
        确认必须当场、当次、来自 BOOS 本人（见 resolve_confirmation）：
        带作用域的确认必须和本次操作完全一致，对不上就拒绝；
        布尔值 True 只是调用方的担保，审计里记为 caller-attested。
        origin 是操作内容的来历："user"（BOOS 原话）或 "agent"
        （agent 自己推理）；"external"（从文件/网页/记忆里读到的）
        绝不能直接拿来删/改/写——外部内容里的指令性文字不能指挥
        agent 做事，路径只能来自 BOOS 原话或 agent 自己的正常推理。
        返回确认来源（"user-scoped" / "caller-attested" / "not-required"）。"""
        if origin not in ("user", "agent"):
            raise PermissionDenied(
                f"外部来源的操作不直接执行：{origin}（路径只能来自 BOOS 原话或 agent 自己的推理）"
            )
        if not self.require_confirmation:
            return "not-required"
        scope = confirmation_scope_for_file(action, rel_path)
        status, reason = resolve_confirmation(confirmed, scope)
        if status is None:
            self.audit_log.record(
                tool="files",
                action=action,
                path=rel_path,
                actor=actor,
                confirmed=False,
                success=False,
                reason=reason,
                origin=origin,
            )
            raise NeedsConfirmation(f"{description}（{reason}）")
        return status

    def _backup(self, path, rel_path):
        """覆盖/改之前，把原文件备份到 .backups/，只留最近 5 份。
        密钥文件和 .env 不备份——备份就是一份能读的密钥副本，
        反而成了漏密钥的口子。返回 (备份路径或 None, 是否因密钥跳过备份)。"""
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            relative = path
        if _is_secret_blocked(relative) or _is_env_file(relative):
            return None, True
        backup_dir = self.root / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        safe_name = str(rel_path).replace("/", "__").replace("\\", "__")
        backup_path = backup_dir / f"{safe_name}_{_timestamp()}.bak"
        shutil.copy2(path, backup_path)
        # 只留最近 5 份旧备份。
        siblings = sorted(backup_dir.glob(f"{safe_name}_*.bak"))
        for stale in siblings[: -_BACKUPS_TO_KEEP]:
            try:
                stale.unlink()
            except OSError:
                pass
        return backup_path, False

    def _atomic_write(self, path, content):
        """先写临时文件、再原子改名：写一半失败，原文件不动。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent), prefix=".tmp-", suffix=".tmp",
            delete=False,
        )
        try:
            handle.write(content)
            handle.close()
            os.replace(handle.name, path)
        except OSError as exc:
            try:
                handle.close()
            except Exception:
                pass
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise ToolError(f"写入失败，原文件未动：{exc}") from exc

    # ---------- 读 ----------

    def read_text(self, rel_path, offset=0, limit=_MAX_READ_LINES):
        """分页读文本文件：一次最多 2,000 行。返回行列表 + 总行数 + 是否截断。"""
        self._check(rel_path, "read")
        path = resolve_in_workspace(self.root, rel_path)
        if not path.is_file():
            raise ToolError(f"文件不存在：{rel_path}")
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ToolError(f"文件太大（>{_MAX_FILE_BYTES // (1024 * 1024)}MB），先拒绝：{rel_path}")
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"不是纯文本文件（读不出 UTF-8），先拒绝：{rel_path}") from exc
        lines = text.splitlines()
        total = len(lines)
        page = lines[offset : offset + limit]
        return {
            "path": str(rel_path),
            "offset": offset,
            "limit": limit,
            "total_lines": total,
            "lines": page,
            "truncated": (offset + len(page)) < total,
        }

    def list_dir(self, rel_path=".", recursive=False, include_hidden=False):
        """列目录：名字、类型、大小、修改时间。默认不列隐藏文件，递归最多 500 条。"""
        self._check(rel_path, "list")
        path = resolve_in_workspace(self.root, rel_path)
        if not path.is_dir():
            raise ToolError(f"目录不存在：{rel_path}")
        entries = []
        truncated = False
        if recursive:
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [
                    d for d in dirnames if include_hidden or not d.startswith(".")
                ]
                for name in sorted(dirnames + filenames):
                    if not include_hidden and name.startswith("."):
                        continue
                    entry_path = Path(dirpath) / name
                    entries.append(self._entry(entry_path, path))
                    if len(entries) >= 500:
                        truncated = True
                        break
                if truncated:
                    break
        else:
            for entry_path in sorted(path.iterdir()):
                if not include_hidden and entry_path.name.startswith("."):
                    continue
                entries.append(self._entry(entry_path, path))
        return {"path": str(rel_path), "entries": entries, "truncated": truncated}

    def _entry(self, entry_path, base):
        try:
            stat = entry_path.stat()
            size = stat.st_size if entry_path.is_file() else None
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
                timespec="seconds"
            )
        except OSError:
            size, mtime = None, None
        return {
            "name": str(entry_path.relative_to(base)),
            "type": "dir" if entry_path.is_dir() else "file",
            "size": size,
            "mtime": mtime,
        }

    def file_info(self, rel_path):
        """看一个文件/目录在不在、多大、什么时候改的。"""
        self._check(rel_path, "read")
        path = resolve_in_workspace(self.root, rel_path)
        if not path.exists():
            return {"path": str(rel_path), "exists": False}
        stat = path.stat()
        return {
            "path": str(rel_path),
            "exists": True,
            "type": "dir" if path.is_dir() else "file",
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
                timespec="seconds"
            ),
        }

    # ---------- 写 / 改 / 删 ----------

    def write_text(self, rel_path, content, overwrite=False, confirmed=False, actor="agent", origin="agent"):
        """写文件：默认不覆盖已存在的文件；覆盖必须显式说，且覆盖前先备份
        （密钥文件和 .env 不备份）。origin 见 _confirm。"""
        self._check(rel_path, "write")
        path = resolve_in_workspace(self.root, rel_path)
        if path.exists() and not path.is_file():
            raise ToolError(f"目标不是文件：{rel_path}")
        if path.exists() and not overwrite:
            raise ToolError(f"文件已存在，默认不覆盖（先读再决定，或显式说要覆盖）：{rel_path}")
        confirm_source = self._confirm(
            "write", rel_path, f"写文件 {rel_path}（{'覆盖' if path.exists() else '新建'}）", confirmed, actor, origin
        )
        before_sha = sha256_file(path) if path.exists() else None
        backup = None
        backup_skipped = False
        if path.exists():
            backup, backup_skipped = self._backup(path, rel_path)
        self._atomic_write(path, content)
        after_sha = sha256_file(path)
        self.audit_log.record(
            tool="files",
            action="write",
            path=rel_path,
            actor=actor,
            confirmed=confirmed,
            success=True,
            before_sha256=before_sha,
            after_sha256=after_sha,
            extra={"bytes": len(content.encode("utf-8")), "backup": str(backup) if backup else None, "backup_skipped": backup_skipped},
            confirmation_source=confirm_source,
            origin=origin,
        )
        return {
            "path": str(rel_path),
            "bytes": len(content.encode("utf-8")),
            "sha256": after_sha,
            "backup": str(backup) if backup else None,
        }

    def edit_text(self, rel_path, old, new, confirmed=False, actor="agent", origin="agent"):
        """按“旧串换新串”改文件：旧串在文件里必须只出现一次。origin 见 _confirm。"""
        self._check(rel_path, "edit")
        path = resolve_in_workspace(self.root, rel_path)
        if not path.is_file():
            raise ToolError(f"文件不存在：{rel_path}")
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"不是纯文本文件（读不出 UTF-8），先拒绝：{rel_path}") from exc
        count = text.count(old)
        if count == 0:
            raise ToolError(f"旧串在文件里找不到，没改：{rel_path}")
        if count > 1:
            raise ToolError(f"旧串在文件里出现了 {count} 次，不唯一，没改（请写更具体的旧串）：{rel_path}")
        confirm_source = self._confirm("edit", rel_path, f"改文件 {rel_path}", confirmed, actor, origin)
        before_sha = sha256_file(path)
        backup, backup_skipped = self._backup(path, rel_path)
        self._atomic_write(path, text.replace(old, new, 1))
        after_sha = sha256_file(path)
        self.audit_log.record(
            tool="files",
            action="edit",
            path=rel_path,
            actor=actor,
            confirmed=confirmed,
            success=True,
            before_sha256=before_sha,
            after_sha256=after_sha,
            extra={"backup": str(backup) if backup else None, "backup_skipped": backup_skipped},
            confirmation_source=confirm_source,
            origin=origin,
        )
        return {"path": str(rel_path), "sha256": after_sha, "backup": str(backup) if backup else None}

    def delete(self, rel_path, permanent=False, recursive=False, confirmed=False, actor="agent", origin="agent"):
        """删文件：默认进回收站（.trash/，能找回来）；彻底删要显式说。
        删目录默认拒绝，要删必须显式说递归。工作区根目录和 .backups/、
        .trash/、.audit/ 本身不让删。origin 见 _confirm——删用的路径只能
        来自 BOOS 原话或 agent 自己的推理，外部文件里读到的路径不直接拿来删。"""
        self._check(rel_path, "delete")
        path = resolve_in_workspace(self.root, rel_path)
        if path == self.root:
            raise ToolError("不能删工作区根目录本身")
        if path.parent == self.root and path.name in (".backups", ".trash", ".audit"):
            raise ToolError(f"不能删 {path.name} 目录本身")
        if not path.exists():
            raise ToolError(f"文件不存在：{rel_path}")
        if path.is_dir() and not recursive:
            raise ToolError(f"删目录必须显式说递归：{rel_path}")
        confirm_source = self._confirm(
            "delete",
            rel_path,
            f"删{'目录' if path.is_dir() else '文件'} {rel_path}（{'彻底删除' if permanent else '进回收站'}）",
            confirmed,
            actor,
            origin,
        )
        before_sha = sha256_file(path) if path.is_file() else None
        if permanent:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            destination = None
        else:
            trash_dir = self.root / ".trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
            destination = trash_dir / f"{path.name}_{_timestamp()}"
            shutil.move(str(path), str(destination))
        self.audit_log.record(
            tool="files",
            action="delete",
            path=rel_path,
            actor=actor,
            confirmed=confirmed,
            success=True,
            before_sha256=before_sha,
            after_sha256=None,
            extra={"permanent": permanent, "destination": str(destination) if destination else None},
            confirmation_source=confirm_source,
            origin=origin,
        )
        return {
            "path": str(rel_path),
            "permanent": permanent,
            "destination": str(destination) if destination else None,
        }
