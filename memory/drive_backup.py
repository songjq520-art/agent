# memory/drive_backup.py（十二自己备份自己：打包 + 上传网盘 + 下载验真）
# 2026-11-24 BOOS 定的自主性要求：备份不能再靠十一的 cron 从外面拉，
# 要十二自己的进程内调度触发、自己打包、自己上传、自己验真。
#
# 和 memory/backup.py 的分工：那边是“账本+记忆的 JSON 备份/恢复”
# （给 restore_from_backup 用）；这边是“记忆目录整包 zip + Google
# Drive 同步”，文件集和十一 cron 时代的 zip-memory.py 完全一致：
# 账本、history.db、5 个 JSON、conversation_transcripts/ 整目录；
# 不含 mem0_chroma（能从账本重建）、.env、.venv。
#
# 红线（都是 BOOS 拍过的板）：
# - 清单里缺哪个文件就跳过并记日志，绝不让整个备份崩掉；
# - 密钥在仓库外（默认 C:\Users\user\.shier\drive_sa.json，可用环境
#   变量 SHIER_DRIVE_KEY 改），密钥不存在时只打包不上传、状态记为
#   待上传，绝不报错崩溃；
# - 上传后必须把网盘文件下载回来比 SHA256，一致才算成功；成功才
#   写 backup_state.json 的 last_success；
# - 本模块所有入口都不抛异常给调用方（调度线程、服务主进程不能被
#   备份拖垮），失败一律落在返回值和状态文件里。

import hashlib
import json
import logging
import os
import zipfile
from datetime import datetime, time as _dtime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo

    SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # 没有 tzdata 时用固定 +08:00（上海无夏令时，等价）
    SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

# 打包文件集：和 zip-memory.py 一字不差，别自己加减。
BACKUP_FILES = [
    "memory_events.jsonl",
    "history.db",
    "conversation_summary.json",
    "graph_snapshot.json",
    "delete-journal.json",
    "mem0_versions.json",
    "memories.json",
]
BACKUP_DIRS = ["conversation_transcripts"]

# 网盘上那份备份文件（十一 cron 时代就一直在更新的同一个文件，
# 原地 update 保持同一 file ID，别新建——BOOS 那边认的是这份）。
DRIVE_FILE_ID = "1SDFAORFfXfxL5ngjkxbdLHZIiAJhEoHZ"
FILE_ID_ENV = "SHIER_DRIVE_FILE_ID"

# 服务账号密钥：仓库外，路径可配。
DEFAULT_KEY_PATH = r"C:\Users\user\.shier\drive_sa.json"
KEY_ENV = "SHIER_DRIVE_KEY"

# 每天 03:51（Asia/Shanghai）跑一次。
BACKUP_HOUR = 3
BACKUP_MINUTE = 51

STATE_FILENAME = "backup_state.json"
BACKUP_SUBDIR = "backups"

DRIVE_UPLOAD_URL = (
    "https://www.googleapis.com/upload/drive/v3/files/{file_id}"
    "?uploadType=media"
)
DRIVE_DOWNLOAD_URL = "https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def resolve_key_path(key_path=None):
    """密钥路径：显式参数 > 环境变量 SHIER_DRIVE_KEY > 默认路径。"""
    if key_path:
        return str(key_path)
    env = os.environ.get(KEY_ENV)
    if env and env.strip():
        return env.strip()
    return DEFAULT_KEY_PATH


def resolve_file_id(file_id=None):
    """网盘文件 ID：显式参数 > 环境变量 > 默认（十一 cron 时代那份）。"""
    if file_id:
        return str(file_id)
    env = os.environ.get(FILE_ID_ENV)
    if env and env.strip():
        return env.strip()
    return DRIVE_FILE_ID


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_shanghai(now=None):
    """现在的上海时间（aware datetime）。now 可传 epoch 或 datetime。"""
    if now is None:
        return datetime.now(SHANGHAI_TZ)
    if isinstance(now, (int, float)):
        return datetime.fromtimestamp(float(now), SHANGHAI_TZ)
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.replace(tzinfo=SHANGHAI_TZ)
        return now.astimezone(SHANGHAI_TZ)
    return datetime.now(SHANGHAI_TZ)


def backup_zip_name(now=None):
    """备份文件名：shier-memory-YYYYMMDD.zip（日期按 Asia/Shanghai）。"""
    return "shier-memory-" + _now_shanghai(now).strftime("%Y%m%d") + ".zip"


def build_backup_zip(memory_dir, backup_dir=None, now=None):
    """把记忆目录按清单打成 zip，算 SHA256。

    清单里缺的文件/目录跳过并记进 skipped（不抛）；zip 先写临时
    文件再原子替换，半截包不会冒充成品。返回
    {"path", "name", "sha256", "size", "included", "skipped"}。
    included/skipped 里是包内相对名（目录记为 "名字/"）。
    """
    memory_dir = Path(memory_dir)
    backup_dir = Path(backup_dir) if backup_dir else memory_dir / BACKUP_SUBDIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    name = backup_zip_name(now)
    target = backup_dir / name
    tmp = backup_dir / (name + ".tmp")
    included = []
    skipped = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in BACKUP_FILES:
            src = memory_dir / fname
            if src.is_file():
                z.write(src, fname)
                included.append(fname)
            else:
                skipped.append(fname)
                logger.info("备份打包跳过（不存在）：%s", src)
        for dname in BACKUP_DIRS:
            src_dir = memory_dir / dname
            if not src_dir.is_dir():
                skipped.append(dname + "/")
                logger.info("备份打包跳过（目录不存在）：%s", src_dir)
                continue
            for root, _dirs, files in os.walk(src_dir):
                for fname in sorted(files):
                    full = Path(root) / fname
                    arc = full.relative_to(memory_dir).as_posix()
                    z.write(full, arc)
            included.append(dname + "/")
    os.replace(tmp, target)
    return {
        "path": str(target),
        "name": name,
        "sha256": sha256_file(target),
        "size": target.stat().st_size,
        "included": included,
        "skipped": skipped,
    }


def load_backup_state(memory_dir):
    """读 memory/backup_state.json；读不出/不存在返回 {}（不抛）。"""
    try:
        data = json.loads(
            (Path(memory_dir) / STATE_FILENAME).read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_backup_state(memory_dir, state):
    """原子写 memory/backup_state.json。写失败只记日志，返回 False。"""
    path = Path(memory_dir) / STATE_FILENAME
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(tmp, path)
        return True
    except Exception:
        logger.warning("备份状态写不进去：%s", path, exc_info=True)
        return False


class DriveClient:
    """Google Drive REST 最小客户端（服务账号密钥）。

    只做两件事：把 zip 原地 update 到已有文件、把文件下载回来。
    http 可注入（默认 requests），token_provider 可注入——单元测试
    靠这两个口子全程 mock，不碰真网络、不碰真密钥。
    google-auth 只在这里懒加载：没密钥、只打包时根本用不上它。
    """

    def __init__(self, key_path, file_id=None, http=None,
                 token_provider=None, timeout=120):
        self._key_path = str(key_path)
        self._file_id = resolve_file_id(file_id)
        self._http = http
        self._token_provider = token_provider
        self._timeout = timeout

    @property
    def file_id(self):
        return self._file_id

    def _session(self):
        if self._http is not None:
            return self._http
        import requests

        return requests

    def _access_token(self):
        if self._token_provider is not None:
            return self._token_provider()
        from google.auth.transport.requests import Request as _GoogleRequest
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            self._key_path, scopes=[DRIVE_SCOPE]
        )
        creds.refresh(_GoogleRequest())
        return creds.token

    def update_file(self, zip_path):
        """把 zip 内容原地更新到网盘文件（PATCH media upload）。"""
        token = self._access_token()
        data = Path(zip_path).read_bytes()
        resp = self._session().patch(
            DRIVE_UPLOAD_URL.format(file_id=self._file_id),
            data=data,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/zip",
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp

    def download_file(self):
        """把网盘文件整份下载回来（bytes），供验真比 SHA256。"""
        token = self._access_token()
        resp = self._session().get(
            DRIVE_DOWNLOAD_URL.format(file_id=self._file_id),
            headers={"Authorization": "Bearer " + token},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.content


def run_backup(memory_dir, key_path=None, file_id=None, drive_client=None,
               allow_upload=True, now=None):
    """跑一次完整备份：打包 →（可传时）上传 → 下载验真 → 写状态。

    任何一步失败都不抛：返回 {"status": ...}，状态文件里记
    pending_upload=True 等下次补跑。status 取值：
    - "success"：上传且验真一致，last_success 已写；
    - "packaged_pending_upload"：只打包没上传（没密钥/被要求只打包/
      有待传的包），或上传、验真没过（另带 reason 区分）；
    - "package_failed"：打包本身失败（error 里有原因）。
    """
    memory_dir = Path(memory_dir)
    now_cst = _now_shanghai(now)
    try:
        zip_info = build_backup_zip(memory_dir, now=now_cst)
    except Exception as exc:
        logger.warning("备份打包失败：%s", exc, exc_info=True)
        return {"status": "package_failed", "error": str(exc)}

    state = load_backup_state(memory_dir)
    state.update({
        "last_packaged_at": now_cst.isoformat(),
        "last_zip": zip_info["name"],
        "last_zip_sha256": zip_info["sha256"],
        "last_zip_size": zip_info["size"],
        "included": zip_info["included"],
        "skipped": zip_info["skipped"],
    })
    result = {"status": "packaged_pending_upload", "zip": zip_info}

    def _pending(reason, error=None):
        state["pending_upload"] = True
        state["pending_reason"] = reason
        if error:
            state["last_error"] = error
        save_backup_state(memory_dir, state)
        result["reason"] = reason
        if error:
            result["error"] = error
        return result

    if not allow_upload:
        return _pending("upload_deferred")

    client = drive_client
    if client is None:
        resolved_key = resolve_key_path(key_path)
        if not resolved_key or not Path(resolved_key).is_file():
            # 没密钥不是故障：包打好了，等密钥到位再补传。
            logger.info(
                "备份只打包不上传（密钥不存在：%s），状态记为待上传。",
                resolved_key,
            )
            return _pending("no_key")
        client = DriveClient(resolved_key, file_id=file_id)

    try:
        client.update_file(zip_info["path"])
        downloaded = client.download_file()
    except Exception as exc:
        logger.warning("备份上传失败，记为待上传：%s", exc, exc_info=True)
        return _pending("upload_error", error=str(exc))

    remote_sha = sha256_bytes(downloaded)
    if remote_sha != zip_info["sha256"]:
        logger.warning(
            "备份验真不过：本地 %s，网盘下载回来 %s。",
            zip_info["sha256"], remote_sha,
        )
        return _pending(
            "verify_mismatch",
            error="downloaded sha256 mismatch: " + remote_sha,
        )

    state.update({
        "last_success": now_cst.isoformat(),
        "last_success_epoch": now_cst.timestamp(),
        "sha256": zip_info["sha256"],
        "size": zip_info["size"],
        "drive_file_id": client.file_id,
        "pending_upload": False,
    })
    state.pop("pending_reason", None)
    state.pop("last_error", None)
    save_backup_state(memory_dir, state)
    result["status"] = "success"
    return result


def last_scheduled_slot(now_epoch, hour=BACKUP_HOUR, minute=BACKUP_MINUTE):
    """最近一个已到的每日备份时点（epoch）：今天 03:51（上海）已过
    就是今天的，没过就是昨天的。"""
    now_cst = datetime.fromtimestamp(float(now_epoch), SHANGHAI_TZ)
    slot = datetime.combine(now_cst.date(), _dtime(hour, minute), SHANGHAI_TZ)
    if now_cst < slot:
        slot -= timedelta(days=1)
    return slot.timestamp()


def _last_success_epoch(*states):
    """从若干状态字典里找最近的 last_success_epoch（数值型）。"""
    best = None
    for state in states:
        if not isinstance(state, dict):
            continue
        for key in ("last_success_epoch", "backup_last_success"):
            value = state.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if best is None or float(value) > best:
                    best = float(value)
    return best


def is_backup_due(maintenance_state, backup_state, now_epoch,
                  hour=BACKUP_HOUR, minute=BACKUP_MINUTE):
    """备份到期判定：上次成功早于最近一个每日时点（或从没成功过）
    就是到期——服务启动时查这个，错过了立刻补跑。"""
    last = _last_success_epoch(maintenance_state, backup_state)
    if last is None:
        return True
    return last < last_scheduled_slot(now_epoch, hour, minute)
