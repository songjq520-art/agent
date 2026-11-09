# tools/cleanup.py（第一批工具：清理垃圾）
# 作用：BOOS 的新需求——笔记本上垃圾多，让十二帮他清理。这是文件工具的
# 第一个真实用途，按 BOOS 定的规矩做：
#   1. 先扫一遍，按种类报大白话（这是什么、有多大、删了有什么影响），
#      不列原始文件名（BOOS 看不懂文件名）。
#   2. BOOS 按种类点头再删（per-category confirmation），没确认的种类不动。
#   3. 确认要删的直接彻底删除，空间马上腾出来（进回收站不算删干净）。
#   4. 拿不准的东西先进回收站，过几天确认没问题再清空。
#   5. 密钥文件（.env、.pem、.key、.ssh 等）不碰，扫描结果里也不列出来。
#   6. 清理范围是整台电脑，不只工作区——扫描根目录由调用方显式传
#      （下载文件夹、临时文件、回收站是首选），这是单独授权的任务，
#      范围由 BOOS 定，不在清单里的地方不动。
#
# 三道保护（GPT 的建议，采纳）：
#   - 分类置信度：低置信度的种类（大文件、拿不准）绝不直接删，先进回收站。
#   - 删除预览：删除只认冻结的预览清单（build_preview 的那一份），
#     预览里冻的不只是路径，还有每个文件的大小和修改时间；删除时逐个
#     重新验证（仍在扫描根目录内、不是密钥文件、大小/修改时间没变过），
#     对不上的跳过不删；扫描到删除之间新增的文件、改动过的文件都不会被删；
#     .audit、.backups、.trash 这类重要目录不扫、也不删。
#     预览存成 JSON 再读回来时，靠校验和发现有没有被改过。
#   - 删除后核对：实际删了几个、挪进回收站几个、腾出多少空间、
#     失败几个、跳过几个，都如实汇报；删完还要复查文件是不是真的没了。
# 删除清单只认冻结的预览 + BOOS 点过头的种类：绝不认外部文件/网页里
# 读到的路径——调用方必须只把 BOOS 确认过的种类传进来。
#
# 扫文件、算大小、删文件都是代码在本地跑，不花 token；花 token 的只有
# 十二看汇总结果、跟 BOOS 说话那几句。

import hashlib
import hmac
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tools.errors import ToolError
from tools.permissions import _is_env_file, _is_secret_blocked

# 种类定义：id → 大白话名字 / 这是什么 / 删了有什么影响 / 置信度。
# 置信度 high 的，BOOS 点头后直接彻底删；low 的，先进回收站。
CATEGORIES = {
    "temp": {
        "name": "临时文件",
        "description": "系统和软件用完随手扔的临时文件",
        "impact": "删了没事，软件需要时会自己再建",
        "confidence": "high",
    },
    "browser_cache": {
        "name": "浏览器缓存",
        "description": "浏览器存的网页图片和数据",
        "impact": "删了之后打开网页会慢一点，要重新下载",
        "confidence": "high",
    },
    "logs": {
        "name": "日志文件",
        "description": "程序运行留下的记录文件",
        "impact": "删了没事，只是看不到旧的运行记录了",
        "confidence": "high",
    },
    "installers": {
        "name": "旧安装包",
        "description": "下载文件夹里已经装好的软件的安装文件",
        "impact": "删了不影响已经装好的软件，以后重装要重新下载",
        "confidence": "high",
    },
    "recycle_bin": {
        "name": "回收站",
        "description": "已经删掉、还在回收站里的东西",
        "impact": "清空就彻底没了，找不回来",
        "confidence": "high",
    },
    "large_files": {
        "name": "大文件",
        "description": "单个很大的文件",
        "impact": "拿不准，先问你再决定",
        "confidence": "low",
    },
    "unknown": {
        "name": "拿不准的东西",
        "description": "分不清是什么的文件",
        "impact": "先放进回收站，过几天没问题再清空",
        "confidence": "low",
    },
}

_INSTALLER_SUFFIXES = {".msi", ".exe", ".dmg", ".pkg", ".deb", ".rpm", ".apk"}
_MAX_FILES_PER_CATEGORY = 200

# 重要目录：扫描时整个跳过，删除时绝不碰（哪怕预览里有也不删）。
_PROTECTED_DIR_NAMES = (".audit", ".backups", ".trash")


def _is_protected_path(path):
    """路径是不是在 .audit/.backups/.trash 这类重要目录里（或就是它本身）。"""
    return any(part in _PROTECTED_DIR_NAMES for part in Path(path).parts)


def human_size(num_bytes):
    """字节数换成人话：2.3GB、1.1MB、800KB、500B。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            text = f"{size:.1f}".rstrip("0").rstrip(".")
            return f"{text}{unit}"
        size /= 1024
    return f"{num_bytes}B"


def _classify_file(path, root_kind, size, large_threshold_bytes):
    """给一个文件定种类：先看大小（大文件单独列出来问），再看位置和后缀。"""
    if size > large_threshold_bytes:
        return "large_files"
    if root_kind == "temp":
        return "temp"
    if root_kind == "recycle_bin":
        return "recycle_bin"
    if root_kind == "browser_cache":
        return "browser_cache"
    suffix = path.suffix.lower()
    if suffix in (".tmp", ".temp"):
        return "temp"
    if suffix == ".log":
        return "logs"
    if suffix in _INSTALLER_SUFFIXES:
        return "installers"
    if root_kind == "downloads":
        return "unknown"
    return "unknown"


def _is_inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def scan(root, root_kind="other", large_threshold_mb=500, max_files_per_category=_MAX_FILES_PER_CATEGORY):
    """扫一个目录，按种类汇总：每类多少个文件、多大、文件清单（只列前 N 个）。

    root_kind 是 temp / downloads / recycle_bin / browser_cache / other，
    决定按位置分的种类。密钥文件（.env、.pem、.key、.ssh、.git）永不列入，
    只记 skipped_secrets 的个数。符号链接逃出根目录的跳过。
    """
    root = Path(root)
    if not root.is_dir():
        from tools.errors import ToolError

        raise ToolError(f"扫描目录不存在：{root}")
    large_threshold_bytes = large_threshold_mb * 1024 * 1024
    categories = {}
    scanned = 0
    skipped_secrets = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # .git/.ssh/.audit/.backups/.trash 目录整个跳过；符号链接目录逃出根目录的也跳过。
        dirnames[:] = [
            d
            for d in dirnames
            if d not in (".git", ".ssh", *_PROTECTED_DIR_NAMES)
            and _is_inside(Path(dirpath) / d, root)
        ]
        for name in filenames:
            file_path = Path(dirpath) / name
            if _is_secret_blocked(file_path) or _is_env_file(file_path):
                skipped_secrets += 1
                continue
            if file_path.is_symlink() and not _is_inside(file_path, root):
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            scanned += 1
            category = _classify_file(file_path, root_kind, stat.st_size, large_threshold_bytes)
            bucket = categories.setdefault(
                category, {"files": [], "file_count": 0, "total_bytes": 0, "capped": False}
            )
            bucket["file_count"] += 1
            bucket["total_bytes"] += stat.st_size
            if len(bucket["files"]) < max_files_per_category:
                bucket["files"].append(
                    {"path": str(file_path), "size": stat.st_size, "mtime": stat.st_mtime}
                )
            else:
                bucket["capped"] = True
    return {
        "root": str(root),
        "root_kind": root_kind,
        "scanned": scanned,
        "skipped_secrets": skipped_secrets,
        "categories": categories,
    }


def scan_many(roots):
    """扫多个目录（比如下载文件夹 + 临时文件 + 回收站），按种类合并汇总。

    roots 是 [{"path": ..., "kind": ...}]，kind 同 scan 的 root_kind。
    """
    merged = {"roots": [], "scanned": 0, "skipped_secrets": 0, "categories": {}}
    for entry in roots:
        result = scan(entry["path"], root_kind=entry.get("kind", "other"))
        merged["roots"].append(result["root"])
        merged["scanned"] += result["scanned"]
        merged["skipped_secrets"] += result["skipped_secrets"]
        for category, bucket in result["categories"].items():
            target = merged["categories"].setdefault(
                category, {"files": [], "file_count": 0, "total_bytes": 0, "capped": False}
            )
            target["file_count"] += bucket["file_count"]
            target["total_bytes"] += bucket["total_bytes"]
            target["capped"] = target["capped"] or bucket["capped"]
            room = _MAX_FILES_PER_CATEGORY - len(target["files"])
            target["files"].extend(bucket["files"][:room])
            if len(bucket["files"]) > room:
                target["capped"] = True
    return merged


def preview_checksum(preview):
    """算预览的校验和：对去掉 checksum 字段后的预览做规范 JSON 再算 SHA-256。
    预览存成 JSON 再读回来时，靠这个发现有没有被改过。"""
    payload = {key: value for key, value in preview.items() if key != "checksum"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_preview(preview):
    """校验预览完整性：没有 checksum、或 checksum 对不上，都返回 False。
    删之前必须先过这一关——清单要是被改过，宁可不删，让 BOOS 重新扫一遍。"""
    checksum = preview.get("checksum")
    if not isinstance(checksum, str):
        return False
    return hmac.compare_digest(checksum, preview_checksum(preview))


def build_preview(scan_result):
    """把扫描结果冻成一份删除预览：删除只认这一份清单，之后新增、
    改动的文件都不会被删。预览里每个文件都冻了路径、大小和修改时间，
    删除时逐个比对；最后盖上校验和，存成 JSON 再读回来也能发现被改过。
    返回预览字典（可存成 JSON，删的时候再读回来）。"""
    roots = scan_result.get("roots") or [scan_result["root"]]
    categories = {}
    for category_id, bucket in scan_result["categories"].items():
        definition = CATEGORIES[category_id]
        categories[category_id] = {
            "name": definition["name"],
            "description": definition["description"],
            "impact": definition["impact"],
            "confidence": definition["confidence"],
            "file_count": bucket["file_count"],
            "total_bytes": bucket["total_bytes"],
            "files": [
                {"path": f["path"], "size": f["size"], "mtime": f["mtime"]}
                for f in bucket["files"]
            ],
            "capped": bucket.get("capped", False),
        }
    preview = {
        "preview_id": uuid.uuid4().hex[:8],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "roots": roots,
        "categories": categories,
    }
    preview["checksum"] = preview_checksum(preview)
    return preview


def format_summary(preview):
    """把预览写成大白话汇报（给 BOOS 看的）：每类是什么、有多大、
    删了有什么影响。不列原始文件名。"""
    lines = []
    total_files = 0
    total_bytes = 0
    for category_id, bucket in preview["categories"].items():
        if bucket["file_count"] == 0:
            continue
        total_files += bucket["file_count"]
        total_bytes += bucket["total_bytes"]
        confidence_note = "（拿不准，先问你）" if bucket["confidence"] == "low" else ""
        lines.append(
            f"{bucket['name']} {human_size(bucket['total_bytes'])}，"
            f"{bucket['file_count']} 个文件，{bucket['description']}，"
            f"{bucket['impact']}{confidence_note}"
        )
    lines.append(f"一共 {total_files} 个文件，{human_size(total_bytes)}")
    return "\n".join(lines)


def delete_preview(preview, confirmed_categories, trash_dir, actor="agent", audit_log=None):
    """按冻结的预览删：只有 BOOS 点过头的种类才删。

    动手前核对：先验预览的校验和（对不上就拒绝，宁可重新扫），再逐个文件
    重新验证——仍在扫描根目录内、不是密钥文件、不在重要目录里、
    大小和修改时间跟预览时一致；对不上的跳过不删，记进报告。
    高置信度的种类直接彻底删；低置信度的（大文件、拿不准）先进回收站。
    预览之后文件已经没了的，记跳过（不算失败）。
    动手后核对：删完复查文件是不是真的没了，挪进回收站的复查是不是真的
    在回收站里；复查不过的算失败，不算删成功。
    返回删除后核对：删了几个、进回收站几个、腾出多少空间、失败几个、跳过几个。
    删除清单只认冻结的预览 + BOOS 点过头的种类，不认外部文件/网页里
    读到的路径——调用方必须只把 BOOS 确认过的种类传进来。
    """
    if not verify_preview(preview):
        raise ToolError("预览校验和对不上，拒绝删除（预览可能被改过，请重新扫描）")
    trash_dir = Path(trash_dir)
    trash_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(r) for r in preview.get("roots", [])]
    report = {
        "preview_id": preview.get("preview_id"),
        "deleted": 0,
        "moved_to_trash": 0,
        "failed": [],
        "skipped": [],
        "freed_bytes": 0,
        "categories": {},
        "notes": ["只删除文件，空目录留在原地"],
    }
    for category_id in confirmed_categories:
        bucket = preview["categories"].get(category_id)
        if bucket is None:
            report["categories"][category_id] = {
                "deleted": 0,
                "moved_to_trash": 0,
                "failed": 0,
                "skipped": 0,
            }
            continue
        counts = {"deleted": 0, "moved_to_trash": 0, "failed": 0, "skipped": 0}
        for entry in bucket["files"]:
            path_text = entry["path"] if isinstance(entry, dict) else entry
            path = Path(path_text)
            if not any(_is_inside(path, root) for root in roots):
                counts["failed"] += 1
                report["failed"].append({"path": path_text, "reason": "不在扫描范围内，拒绝删除"})
                continue
            if _is_protected_path(path):
                counts["failed"] += 1
                report["failed"].append({"path": path_text, "reason": "重要目录（.audit/.backups/.trash），拒绝删除"})
                continue
            if _is_secret_blocked(path) or _is_env_file(path):
                counts["failed"] += 1
                report["failed"].append({"path": path_text, "reason": "密钥文件，拒绝删除"})
                continue
            if not path.exists():
                counts["skipped"] += 1
                report["skipped"].append(path_text)
                continue
            try:
                stat = path.stat()
            except OSError:
                counts["skipped"] += 1
                report["skipped"].append(path_text)
                continue
            if isinstance(entry, dict):
                if stat.st_size != entry.get("size") or abs(stat.st_mtime - entry.get("mtime", 0)) > 0.01:
                    counts["skipped"] += 1
                    report["skipped"].append(path_text)
                    continue
            size = stat.st_size if path.is_file() else 0
            try:
                if bucket["confidence"] == "low":
                    destination = trash_dir / f"{path.name}_{uuid.uuid4().hex[:8]}"
                    shutil.move(str(path), str(destination))
                    if not destination.exists() or path.exists():
                        counts["failed"] += 1
                        report["failed"].append({"path": path_text, "reason": "挪进回收站后复查不过，算失败"})
                        continue
                    counts["moved_to_trash"] += 1
                    report["moved_to_trash"] += 1
                else:
                    path.unlink()
                    if path.exists():
                        counts["failed"] += 1
                        report["failed"].append({"path": path_text, "reason": "删除后文件还在，算失败"})
                        continue
                    counts["deleted"] += 1
                    report["deleted"] += 1
                    report["freed_bytes"] += size
            except OSError as exc:
                counts["failed"] += 1
                report["failed"].append({"path": path_text, "reason": str(exc)})
        report["categories"][category_id] = counts
    if audit_log is not None:
        audit_log.record(
            tool="cleanup",
            action="delete_preview",
            actor=actor,
            confirmed=True,
            success=len(report["failed"]) == 0,
            reason="",
            extra={
                "preview_id": preview.get("preview_id"),
                "confirmed_categories": list(confirmed_categories),
                "deleted": report["deleted"],
                "moved_to_trash": report["moved_to_trash"],
                "freed_bytes": report["freed_bytes"],
                "failed": len(report["failed"]),
                "skipped": len(report["skipped"]),
            },
        )
    return report
