# tests/test_inhouse_backup_20261124.py（十二自己备份自己，2026-11-24）
# BOOS 定的自主性要求：备份由十二自己的进程内调度触发、自己打包、
# 自己上传网盘、下载回来验真。本文件测 memory/drive_backup.py 和
# web/server.py 里 MemoryMaintenanceScheduler 的备份任务挂钩。
# 全程 mock：假 HTTP、假 Drive 客户端、假密钥路径——绝不碰真网络、
# 绝不往真网盘写东西（真实上传验证由十一拿服务账号密钥单独做）。

import json
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from memory import drive_backup
from memory.drive_backup import DriveClient
from web.server import MemoryMaintenanceScheduler

SH = drive_backup.SHANGHAI_TZ


def _epoch(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=SH).timestamp()


def _make_memory_dir(tmp_path, files=("memory_events.jsonl", "memories.json"),
                     with_transcripts=True):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    for name in files:
        (memory_dir / name).write_text("content of " + name, encoding="utf-8")
    if with_transcripts:
        tdir = memory_dir / "conversation_transcripts"
        tdir.mkdir()
        (tdir / "2026-11-24.jsonl").write_text("transcript", encoding="utf-8")
    return memory_dir


# ---------- 打包 ----------

def test_build_zip_includes_present_and_skips_missing(tmp_path):
    memory_dir = _make_memory_dir(tmp_path)
    info = drive_backup.build_backup_zip(
        memory_dir, now=datetime(2026, 11, 24, 12, 0, tzinfo=SH)
    )
    assert info["name"] == "shier-memory-20261124.zip"
    assert Path(info["path"]).is_file()
    assert info["sha256"] == drive_backup.sha256_file(info["path"])
    assert info["size"] == Path(info["path"]).stat().st_size
    with zipfile.ZipFile(info["path"]) as z:
        names = z.namelist()
    assert "memory_events.jsonl" in names
    assert "memories.json" in names
    assert "conversation_transcripts/2026-11-24.jsonl" in names
    # 清单里其余文件没建出来：跳过并记录，不许崩
    for missing in ("history.db", "conversation_summary.json",
                    "graph_snapshot.json", "delete-journal.json",
                    "mem0_versions.json"):
        assert missing in info["skipped"]
        assert missing not in names
    assert "conversation_transcripts/" in info["included"]


def test_build_zip_empty_dir_does_not_crash(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    info = drive_backup.build_backup_zip(memory_dir)
    assert Path(info["path"]).is_file()
    assert info["included"] == []
    assert set(info["skipped"]) == set(drive_backup.BACKUP_FILES) | {
        "conversation_transcripts/"
    }
    with zipfile.ZipFile(info["path"]) as z:
        assert z.namelist() == []


# ---------- run_backup：没密钥只打包 ----------

def test_run_backup_no_key_packages_only(tmp_path, monkeypatch):
    monkeypatch.delenv("SHIER_DRIVE_KEY", raising=False)
    memory_dir = _make_memory_dir(tmp_path)
    result = drive_backup.run_backup(
        memory_dir, key_path=str(tmp_path / "no_such_key.json")
    )
    assert result["status"] == "packaged_pending_upload"
    assert result["reason"] == "no_key"
    assert Path(result["zip"]["path"]).is_file()
    state = drive_backup.load_backup_state(memory_dir)
    assert state["pending_upload"] is True
    assert state["pending_reason"] == "no_key"
    assert "last_success" not in state
    assert state["last_zip_sha256"] == result["zip"]["sha256"]


def test_resolve_key_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIER_DRIVE_KEY", str(tmp_path / "k.json"))
    assert drive_backup.resolve_key_path() == str(tmp_path / "k.json")
    assert drive_backup.resolve_key_path("explicit") == "explicit"


# ---------- run_backup：假客户端 ----------

class _StubClient:
    """假 Drive 客户端：update 时把 zip 字节存下来，download 原样
    还回去（或按 mode 篡改/抛错）。"""

    file_id = "FAKE_FILE_ID"

    def __init__(self, mode="ok"):
        self.mode = mode
        self.uploaded = None
        self.update_calls = 0

    def update_file(self, zip_path):
        self.update_calls += 1
        if self.mode == "boom":
            raise ConnectionError("network down")
        self.uploaded = Path(zip_path).read_bytes()

    def download_file(self):
        if self.mode == "tamper":
            return b"tampered bytes"
        return self.uploaded


def test_run_backup_upload_success_writes_state(tmp_path):
    memory_dir = _make_memory_dir(tmp_path)
    client = _StubClient()
    result = drive_backup.run_backup(memory_dir, drive_client=client)
    assert result["status"] == "success"
    assert client.update_calls == 1
    state = drive_backup.load_backup_state(memory_dir)
    assert state["pending_upload"] is False
    assert state["sha256"] == result["zip"]["sha256"]
    assert state["size"] == result["zip"]["size"]
    assert state["drive_file_id"] == "FAKE_FILE_ID"
    assert state["last_success"]
    assert state["last_success_epoch"] > 0


def test_run_backup_verify_mismatch_not_success(tmp_path):
    memory_dir = _make_memory_dir(tmp_path)
    result = drive_backup.run_backup(
        memory_dir, drive_client=_StubClient(mode="tamper")
    )
    assert result["status"] == "packaged_pending_upload"
    assert result["reason"] == "verify_mismatch"
    state = drive_backup.load_backup_state(memory_dir)
    assert state["pending_upload"] is True
    assert "last_success" not in state


def test_run_backup_upload_error_stays_pending(tmp_path):
    memory_dir = _make_memory_dir(tmp_path)
    result = drive_backup.run_backup(
        memory_dir, drive_client=_StubClient(mode="boom")
    )
    assert result["status"] == "packaged_pending_upload"
    assert result["reason"] == "upload_error"
    state = drive_backup.load_backup_state(memory_dir)
    assert state["pending_upload"] is True
    assert "last_success" not in state


# ---------- DriveClient 本身（假 HTTP） ----------

class _FakeResp:
    def __init__(self, content=b""):
        self.content = content

    def raise_for_status(self):
        pass


class _FakeHTTP:
    def __init__(self):
        self.calls = []
        self.uploaded = None

    def patch(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("PATCH", url, headers))
        self.uploaded = data
        return _FakeResp(b"{}")

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, headers))
        return _FakeResp(self.uploaded)


def test_drive_client_patch_then_download(tmp_path):
    zip_path = tmp_path / "x.zip"
    zip_path.write_bytes(b"zip-bytes")
    http = _FakeHTTP()
    client = DriveClient(
        key_path="unused.json", file_id="FID123", http=http,
        token_provider=lambda: "tok123",
    )
    client.update_file(zip_path)
    assert client.download_file() == b"zip-bytes"
    patch_call, get_call = http.calls
    assert patch_call[0] == "PATCH"
    assert "FID123" in patch_call[1] and "uploadType=media" in patch_call[1]
    assert patch_call[2]["Authorization"] == "Bearer tok123"
    assert get_call[0] == "GET" and "alt=media" in get_call[1]


# ---------- 到期/补跑判定 ----------

def test_last_scheduled_slot():
    # 11-24 中午（上海）：最近的时点是今天 03:51
    assert drive_backup.last_scheduled_slot(_epoch(2026, 11, 24, 12)) == \
        _epoch(2026, 11, 24, 3, 51)
    # 11-24 凌晨 02:00：今天的还没到，最近的是昨天 03:51
    assert drive_backup.last_scheduled_slot(_epoch(2026, 11, 24, 2)) == \
        _epoch(2026, 11, 23, 3, 51)


def test_is_backup_due():
    now = _epoch(2026, 11, 24, 12)
    slot = drive_backup.last_scheduled_slot(now)
    assert drive_backup.is_backup_due({}, {}, now) is True
    assert drive_backup.is_backup_due(
        {}, {"last_success_epoch": slot}, now) is False
    assert drive_backup.is_backup_due(
        {}, {"last_success_epoch": slot - 1}, now) is True
    assert drive_backup.is_backup_due(
        {"backup_last_success": slot}, {}, now) is False


# ---------- 调度挂钩 ----------

class _BareStore:
    pass


class _StubRunner:
    def __init__(self, status="success"):
        self.status = status
        self.calls = []

    def __call__(self, upload=True):
        self.calls.append(upload)
        return {"status": self.status}


def _sched(tmp_path, runner, probe, state_name="maintenance_state.json"):
    # 备份总开关缺省关：测到期/补跑逻辑的用例先把开关打开
    # （开关本身的行为有专门用例在本文件末尾测）。
    (tmp_path / "backup_state.json").write_text(
        json.dumps({"backup_enabled": True}), encoding="utf-8")
    return MemoryMaintenanceScheduler(
        _BareStore(), tmp_path / state_name,
        decay_interval_seconds=100, self_check_interval_seconds=100,
        tick_seconds=60, network_probe=probe,
        probe_interval_seconds=900, backup_runner=runner,
    )


def test_scheduler_backup_due_online_success(tmp_path):
    runner = _StubRunner()
    sched = _sched(tmp_path, runner, probe=lambda: True)
    now = _epoch(2026, 11, 24, 12)
    ran = sched.run_due(now=now)
    assert ran["backup"] is True
    assert runner.calls == [True]
    state = json.loads(
        (tmp_path / "maintenance_state.json").read_text(encoding="utf-8"))
    assert state["backup_last_success"] == now
    assert "backup_pending" not in state


def test_scheduler_backup_offline_packages_then_recovers(tmp_path):
    runner = _StubRunner()
    online = {"value": False}
    sched = _sched(tmp_path, runner, probe=lambda: online["value"])
    now = _epoch(2026, 11, 24, 12)
    ran = sched.run_due(now=now)
    assert ran["backup"] is True
    assert runner.calls == [False]  # 离线只打包
    state = json.loads(
        (tmp_path / "maintenance_state.json").read_text(encoding="utf-8"))
    assert state["backup_pending"] is True
    assert "backup_last_success" not in state
    # 还离线、且没到下次探网（<900 秒）：不重复跑
    sched.run_due(now=now + 100)
    assert runner.calls == [False]
    # 网络恢复、过了探网间隔：补传成功，清挂账
    online["value"] = True
    ran = sched.run_due(now=now + 1000)
    assert ran["backup"] is True
    assert runner.calls == [False, True]
    state = json.loads(
        (tmp_path / "maintenance_state.json").read_text(encoding="utf-8"))
    assert state["backup_last_success"] == now + 1000
    assert "backup_pending" not in state


def test_scheduler_backup_not_due_skips(tmp_path):
    runner = _StubRunner()
    state_path = tmp_path / "maintenance_state.json"
    now = _epoch(2026, 11, 24, 12)
    slot = drive_backup.last_scheduled_slot(now)
    state_path.write_text(
        json.dumps({"backup_last_success": slot}), encoding="utf-8")
    sched = _sched(tmp_path, runner, probe=lambda: True)
    ran = sched.run_due(now=now)
    assert ran["backup"] is False
    assert runner.calls == []


def test_scheduler_without_backup_runner_shape_unchanged(tmp_path):
    sched = MemoryMaintenanceScheduler(
        _BareStore(), tmp_path / "state.json",
        decay_interval_seconds=100, self_check_interval_seconds=100,
        tick_seconds=60,
    )
    ran = sched.run_due(now=1_000)
    assert ran == {"decay": False, "self_check": False}
# ---------- 备份总开关（BOOS 2026-11-24 补定：缺省关，十二能自己开） ----------

def test_backup_switch_absent_means_disabled(tmp_path):
    memory_dir = _make_memory_dir(tmp_path)
    # 没有 backup_state.json → 关
    assert drive_backup.is_backup_enabled(memory_dir) is False
    # 有状态文件但没这个字段 → 还是关
    drive_backup.save_backup_state(memory_dir, {"last_success": "x"})
    assert drive_backup.is_backup_enabled(memory_dir) is False


def test_backup_switch_set_roundtrip_preserves_state(tmp_path):
    memory_dir = _make_memory_dir(tmp_path)
    drive_backup.save_backup_state(memory_dir, {
        "last_success": "2026-11-25T02:48:28+08:00",
        "last_success_epoch": 1795546108.7,
        "sha256": "abc123",
        "pending_upload": False,
    })
    assert drive_backup.set_backup_enabled(memory_dir, True) is True
    state = drive_backup.load_backup_state(memory_dir)
    assert state["backup_enabled"] is True
    assert drive_backup.is_backup_enabled(memory_dir) is True
    # 其余字段原样保留
    assert state["last_success"] == "2026-11-25T02:48:28+08:00"
    assert state["last_success_epoch"] == 1795546108.7
    assert state["sha256"] == "abc123"
    assert state["pending_upload"] is False
    assert drive_backup.set_backup_enabled(memory_dir, False) is True
    state = drive_backup.load_backup_state(memory_dir)
    assert state["backup_enabled"] is False
    assert drive_backup.is_backup_enabled(memory_dir) is False
    assert state["sha256"] == "abc123"


def test_run_backup_preserves_switch_flag(tmp_path):
    memory_dir = _make_memory_dir(tmp_path)
    drive_backup.set_backup_enabled(memory_dir, True)
    result = drive_backup.run_backup(memory_dir, drive_client=_StubClient())
    assert result["status"] == "success"
    state = drive_backup.load_backup_state(memory_dir)
    assert state["backup_enabled"] is True


def _sched_no_switch(tmp_path, runner, probe):
    """不写 backup_state.json 的调度器：开关处于缺省（关）状态。"""
    return MemoryMaintenanceScheduler(
        _BareStore(), tmp_path / "maintenance_state.json",
        decay_interval_seconds=100, self_check_interval_seconds=100,
        tick_seconds=60, network_probe=probe,
        probe_interval_seconds=900, backup_runner=runner,
    )


def test_scheduler_backup_disabled_never_runs(tmp_path):
    # 开关缺省关：哪怕到期 + 在线，runner 一次都不许被叫到
    runner = _StubRunner()
    sched = _sched_no_switch(tmp_path, runner, probe=lambda: True)
    ran = sched.run_due(now=_epoch(2026, 11, 24, 12))
    assert ran["backup"] is False
    assert runner.calls == []
    state = json.loads(
        (tmp_path / "maintenance_state.json").read_text(encoding="utf-8"))
    assert state["backup_skip_reason"] == "disabled"
    assert "backup_pending" not in state
    assert "backup_last_success" not in state


def test_scheduler_backup_disabled_ignores_pending(tmp_path):
    # 开关关着时，之前挂账的 pending 也不许补传
    runner = _StubRunner()
    state_path = tmp_path / "maintenance_state.json"
    state_path.write_text(
        json.dumps({"backup_pending": True}), encoding="utf-8")
    sched = _sched_no_switch(tmp_path, runner, probe=lambda: True)
    ran = sched.run_due(now=_epoch(2026, 11, 24, 12))
    assert ran["backup"] is False
    assert runner.calls == []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["backup_skip_reason"] == "disabled"
