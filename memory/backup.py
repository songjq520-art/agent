# memory/backup.py（备份恢复：账本 + 记忆库一起备份，带校验、真恢复）
# 最终方案里的“自动备份恢复”那块（GPT 第二版的三块之一）：
# 备份不是“写了个文件就完事”——写完要校验、定期要真恢复一次，
# 恢复不出来就是假保险。备份内容：账本全部事件（原话都在里面）+
# 索引里全部记忆（原话 + 原始 metadata），再加一份清单（条数、
# 校验和、写备份的时间），恢复时先验清单、再写回。
#
# 备份文件放哪：默认 memory/backups/（gitignore 覆盖，不进仓库），
# 名字带时间戳。恢复是显式操作（Mem0Store.restore_from_backup），
# 不在启动时自动跑——自动恢复会把坏数据也“恢复”进来，恢复之前
# 一定要先验备份、再让 BOOS 知道。

import hashlib
import json
import os
from pathlib import Path

from memory.ledger import snapshot_checksum
from memory.timeutil import _now_iso

BACKUP_VERSION = 1
BACKUP_FILE_PREFIX = "memory-backup-"


def build_backup(items, events, snapshot=None, snapshot_error=None):
    """拼一份备份字典（纯函数，不碰磁盘）。

    items：索引里的记忆列表，每项 {"id", "text", "metadata"}（原始
    metadata，不是规整过的——恢复时要原样写回去）；events：账本全部
    事件；snapshot：账本快照（压过账本才有，没有就是 None——恢复时
    快照是档案的底子，没它，已压进快照的旧事件就回不来了）。
    snapshot_error：快照读坏了时记一笔（备份照常写，但验备份时会
    判不过——缺了快照的备份是不完整的，不能当“能恢复”来认）。
    备份里附 checksum：除 checksum 外的字段按固定顺序拼起来算，
    恢复时重算一遍，对不上就不认这份备份。
    """
    items = [
        {
            "id": (item or {}).get("id"),
            "text": (item or {}).get("text") or "",
            "metadata": dict((item or {}).get("metadata") or {}),
        }
        for item in items or []
    ]
    events = [dict(event) for event in events or []]
    body = {
        "version": BACKUP_VERSION,
        "created_at": _now_iso(),
        "items": items,
        "ledger_events": events,
        "snapshot": dict(snapshot) if isinstance(snapshot, dict) else None,
        "counts": {"items": len(items), "ledger_events": len(events)},
    }
    if snapshot_error:
        body["snapshot_error"] = str(snapshot_error)
    body["checksum"] = _backup_checksum(body)
    return body


def _backup_checksum(body):
    """备份校验和：除 checksum 本身外的字段按固定顺序拼起来算。"""
    canonical = json.dumps(
        {key: value for key, value in body.items() if key != "checksum"},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_backup(path, backup):
    """把备份字典写到文件（原子写：临时文件 + 替换）。返回文件路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(backup, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
    return str(path)


def read_backup(path):
    """读一份备份文件（不校验，只解析）。文件不存在抛 FileNotFoundError，
    解析不出抛 ValueError。"""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("备份文件不是字典。")
    return payload


def verify_backup(path):
    """验一份备份：文件能不能读、结构对不对、条数对不对、校验和对不对。

    返回 {"ok", "errors", "counts"}。ok 只在“全对”时为 True；
    有任何一项不对都列在 errors 里，不抛异常——验备份就是要把
    问题列出来，不是炸掉调用方。"""
    errors = []
    try:
        payload = read_backup(path)
    except FileNotFoundError:
        return {"ok": False, "errors": ["备份文件不存在。"], "counts": {}}
    except (ValueError, OSError) as exc:
        return {"ok": False, "errors": [f"备份文件读不出来：{exc}。"], "counts": {}}
    if payload.get("version") != BACKUP_VERSION:
        errors.append(f"备份版本不对：{payload.get('version')!r}。")
    items = payload.get("items")
    events = payload.get("ledger_events")
    if not isinstance(items, list):
        errors.append("备份里没有 items 列表。")
        items = []
    if not isinstance(events, list):
        errors.append("备份里没有 ledger_events 列表。")
        events = []
    counts = payload.get("counts") or {}
    if counts.get("items") != len(items):
        errors.append(
            f"items 条数对不上：清单说 {counts.get('items')}，实际 {len(items)}。"
        )
    if counts.get("ledger_events") != len(events):
        errors.append(
            f"ledger_events 条数对不上：清单说 {counts.get('ledger_events')}，"
            f"实际 {len(events)}。"
        )
    checksum = payload.get("checksum")
    body = {key: value for key, value in payload.items() if key != "checksum"}
    if checksum != _backup_checksum(body):
        errors.append("备份校验和对不上，备份可能坏了。")
    if payload.get("snapshot_error"):
        errors.append(f"备份时快照读坏了：{payload.get('snapshot_error')}。")
    snapshot = payload.get("snapshot")
    if isinstance(snapshot, dict):
        snap_checksum = snapshot.get("checksum")
        snap_body = {
            key: value for key, value in snapshot.items() if key != "checksum"
        }
        if snap_checksum != snapshot_checksum(snap_body):
            errors.append("备份里的快照校验和对不上，快照可能坏了。")
    return {"ok": not errors, "errors": errors, "counts": dict(counts)}


def backup_filename(now=None):
    """备份文件名（带时间戳）：memory-backup-YYYYMMDD-HHMMSS.json。"""
    stamp = (_now_iso() if now is None else now).replace(":", "").replace("-", "")
    stamp = stamp[:15]  # YYYYMMDDTHHMMSS
    return f"{BACKUP_FILE_PREFIX}{stamp}.json"
