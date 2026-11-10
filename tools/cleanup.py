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
#     重新验证（仍在调用方给的根目录内、不是密钥文件、大小/修改时间没变过），
#     对不上的跳过不删；扫描到删除之间新增的文件、改动过的文件都不会被删；
#     .audit、.backups、.trash 这类重要目录不扫、也不删。
#     删除范围（roots）由调用方单独传可信的扫描根目录，不从预览里读——
#     预览是能被改的 JSON 文件，里面的 roots 被改了删的范围就跟着跑了；
#     预览里写的 roots 必须和调用方传的一致（比对时过 normcase，
#     Windows 上大小写不敏感，Linux 上原样不动）。
#     预览存成 JSON 再读回来时，靠校验和发现有没有被改过：不带钥匙的
#     普通 SHA-256 只能防手误（改完重算一个就过了）；带钥匙的 HMAC
#     （钥匙只放调用方内存里）改了清单重算也对不上。正常流程里预览
#     只放内存、不落盘；要存成 JSON 跨进程用，必须传一把钥匙——
#     命令行流程（tools/cli.py）缺钥匙时扫描就拒绝，普通 SHA-256 预览
#     删除时也明确拒绝。
#     预览里的文件条目必须是含大小/修改时间的 dict，字符串条目直接拒绝。
#     校验和或范围那关不过、拒绝删除时，也记一笔审计（成功=False），
#     不然事后查不到“有人改过清单”。
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
    """路径是不是在 .audit/.backups/.trash 这类重要目录里（或就是它本身）。
    比对大小写不敏感：Windows 上 .BACKUPS 就是 .backups。"""
    return any(part.lower() in _PROTECTED_DIR_NAMES for part in Path(path).parts)


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
        # 目录名比对大小写不敏感：Windows 上 .GIT 就是 .git。
        dirnames[:] = [
            d
            for d in dirnames
            if d.lower() not in (".git", ".ssh", *_PROTECTED_DIR_NAMES)
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


def _canonical_preview_bytes(preview):
    """预览的规范 JSON 字节（去掉 checksum 字段）：校验和算的就是这一份。"""
    payload = {key: value for key, value in preview.items() if key != "checksum"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return canonical.encode("utf-8")


def preview_checksum(preview, key=None):
    """算预览的校验和。
    key 为 None 时是普通 SHA-256，只能发现手误改坏，不能防有意篡改
    （改完清单自己重算一个就过了）；key 不为 None 时是 HMAC-SHA-256，
    钥匙只放调用方内存里、不写进预览文件，改了清单重算也对不上。
    带钥匙的预览没有钥匙算不出校验和，会直接报错。"""
    algo = preview.get("checksum_algo", "sha256")
    if algo == "hmac-sha256":
        if key is None:
            raise ToolError("这份预览是带钥匙校验的，算校验和必须给钥匙")
        if isinstance(key, str):
            key = key.encode("utf-8")
        return hmac.new(key, _canonical_preview_bytes(preview), hashlib.sha256).hexdigest()
    return hashlib.sha256(_canonical_preview_bytes(preview)).hexdigest()


def verify_preview(preview, key=None):
    """校验预览完整性：没有 checksum、或 checksum 对不上，都返回 False。
    删之前必须先过这一关——清单要是被改过，宁可不删，让 BOOS 重新扫一遍。
    带钥匙的预览（checksum_algo 是 hmac-sha256）必须用同一把钥匙验，
    没钥匙或钥匙不对都算没通过；没钥匙的普通校验和只能防手误，不防篡改。"""
    checksum = preview.get("checksum")
    if not isinstance(checksum, str):
        return False
    try:
        expected = preview_checksum(preview, key=key)
    except ToolError:
        return False
    return hmac.compare_digest(checksum, expected)


def build_preview(scan_result, *, key=None):
    """把扫描结果冻成一份删除预览：删除只认这一份清单，之后新增、
    改动的文件都不会被删。预览里每个文件都冻了路径、大小和修改时间，
    删除时逐个比对；最后盖上校验和，存成 JSON 再读回来也能发现被改过。
    key 不为 None 时校验和是带钥匙的 HMAC（钥匙只放调用方内存里，
    不写进预览文件），改了清单重算也对不上；key 为 None 时是普通
    SHA-256，只能防手误改坏。正常流程里预览只放内存、不落盘；
    要存成 JSON 跨进程用，最好传一把钥匙。
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
        "checksum_algo": "hmac-sha256" if key is not None else "sha256",
    }
    preview["checksum"] = preview_checksum(preview, key=key)
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


def _normalize_root_set(roots):
    """roots 比对用的规范形式：解析成绝对路径再过 normcase。
    normcase 在 Windows 上转小写、统一斜杠（D:\\Downloads 和
    d:\\downloads 是同一个目录），在 Linux 上原样不动（/tmp/A 和
    /tmp/a 是两个不同目录），所以不能直接全转小写。"""
    return {os.path.normcase(str(Path(r).resolve())) for r in roots}


def _record_delete_rejection(audit_log, preview, actor, reason, rejection):
    """拒绝删除时记一笔审计：校验和或范围那关不过，删没删成也要留痕，
    不然事后查不到“有人改过清单”。记不进日志也不影响拒绝本身
    （审计写失败了，拒绝照样生效）。"""
    if audit_log is None:
        return
    try:
        audit_log.record(
            tool="cleanup",
            action="delete_preview",
            actor=actor,
            confirmed=True,
            success=False,
            reason=reason,
            extra={"preview_id": preview.get("preview_id"), "rejection": rejection},
        )
    except Exception:
        pass


def delete_preview(preview, confirmed_categories, trash_dir, *, roots, key=None, actor="agent", audit_log=None):
    """按冻结的预览删：只有 BOOS 点过头的种类才删。

    删除范围 roots 必须由调用方单独传可信的扫描根目录（就是当初扫的那几个
    目录），不从预览里读——预览是能被改的 JSON 文件，里面的 roots 要是被
    改成别的目录，删的范围就跟着跑了。预览里如果也写了 roots，必须和调用方
    传的一致（比对时过 normcase，Windows 上大小写不敏感），对不上就拒绝，
    宁可重新扫；拒绝时如果传了 audit_log，先记一笔拒绝的审计再抛错。
    key 是建预览时那把钥匙（只放调用方内存里）：带钥匙的预览必须用同一把
    钥匙验，没钥匙或钥匙不对都拒绝；没钥匙的普通校验和只能防手误。

    动手前核对：先验预览的校验和（对不上就拒绝，宁可重新扫），再逐个文件
    重新验证——仍在调用方给的根目录内、不是密钥文件、不在重要目录里、
    大小和修改时间跟预览时一致；对不上的跳过不删，记进报告。
    预览里的文件条目必须是含大小/修改时间的 dict，字符串条目直接拒绝
    （它会跳过大小/修改时间检查）。
    高置信度的种类直接彻底删；低置信度的（大文件、拿不准）先进回收站。
    预览之后文件已经没了的，记跳过（不算失败）。
    动手后核对：删完复查文件是不是真的没了，挪进回收站的复查是不是真的
    在回收站里；复查不过的算失败，不算删成功。
    返回删除后核对：删了几个、进回收站几个、腾出多少空间、失败几个、跳过几个。
    删除清单只认冻结的预览 + BOOS 点过头的种类，不认外部文件/网页里
    读到的路径——调用方必须只把 BOOS 确认过的种类传进来。
    """
    if not roots:
        reason = "删除范围必须由调用方单独传入可信的扫描根目录，拒绝从预览里读"
        _record_delete_rejection(audit_log, preview, actor, reason, "no-roots")
        raise ToolError(reason)
    trusted_roots = [Path(r) for r in roots]
    if key is not None and preview.get("checksum_algo") != "hmac-sha256":
        # 调用方传了钥匙，预览却不是带钥匙的 HMAC 校验：要么是调用方拿错了
        # 预览，要么是有人把带钥匙的预览降级改成普通 SHA-256 再重算（校验和
        # 算的是去掉 checksum 字段后的 JSON，algo 字段本身也在被算的内容里，
        # 改了 algo 再重算照样对得上，verify_preview 那关拦不住）。两种情况
        # 都拒绝，宁可重新扫。CLI 那层也有同样的检查，这里是库函数层的防线。
        reason = "调用方传了钥匙，但这份预览不是带钥匙的 HMAC 校验，拒绝删除（预览可能被降级改过，请重新扫描）"
        _record_delete_rejection(audit_log, preview, actor, reason, "algo")
        raise ToolError(reason)
    if not verify_preview(preview, key=key):
        reason = "预览校验和对不上，拒绝删除（预览可能被改过，请重新扫描）"
        _record_delete_rejection(audit_log, preview, actor, reason, "checksum")
        raise ToolError(reason)
    preview_roots = preview.get("roots")
    if preview_roots is not None:
        if _normalize_root_set(preview_roots) != _normalize_root_set(trusted_roots):
            reason = "预览里的删除范围和调用方传入的不一致，拒绝删除（请重新扫描）"
            _record_delete_rejection(audit_log, preview, actor, reason, "roots")
            raise ToolError(reason)
    trash_dir = Path(trash_dir)
    trash_dir.mkdir(parents=True, exist_ok=True)
    roots = trusted_roots
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
            if not isinstance(entry, dict):
                counts["failed"] += 1
                report["failed"].append(
                    {"path": str(entry), "reason": "条目格式不对（缺少大小/修改时间），拒绝删除"}
                )
                continue
            if "size" not in entry or "mtime" not in entry:
                counts["failed"] += 1
                report["failed"].append(
                    {"path": str(entry.get("path", entry)), "reason": "条目缺少大小/修改时间，拒绝删除"}
                )
                continue
            path_text = entry["path"]
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
            if stat.st_size != entry["size"] or abs(stat.st_mtime - entry["mtime"]) > 0.01:
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
