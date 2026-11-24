"""副本保留策略测试（BOOS 2026-11-24 定：体检整仓副本只留最近 3 份）。

- 保留数是 Config.archive_keep_copies（默认 3、正整数校验），工厂可经
  参数/环境变量 SHIER_HEALTHCHECK_KEEP_COPIES 覆盖，缺省仍 3；
- 清理在体检运行（report）收尾于进程内自动执行，只认 sandboxes/ 下
  「标签-32 位十六进制」目录；报告、weekly 记录和其他条目一律不碰；
  本次运行刚建的副本受保护，绝不删到自己；
- 清理失败只记进报告的 sandbox_retention 备注，体检本身照常完成。
"""
import json
import sys
import time
import uuid

import pytest

import healthcheck.checks as checks_mod
from healthcheck import Config, Healthcheck
from healthcheck_integration.factory import build_config

FULL = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]


class FakeAdapter:
    """隔离替身：不是十二实连（与模块自有测试同一套契约形状）。"""
    def memory(self, sample_size):
        return {"ok": True, "sample_count": sample_size, "contradictions": [], "sources": [],
                "duplicates": [], "buried": [], "core_integrity": True}
    def services(self):
        return {"ok": True, "processes": [], "endpoints": [], "recent_logs": []}
    def models(self):
        return {"ok": True, "judge": True, "chat": True, "month_spent": 0, "month_limit": 1}
    def resources(self):
        return {"ok": True, "cpu_percent": 1, "memory_percent": 1, "database_bytes": 0, "ledger_bytes": 0}
    def backups(self):
        return {"ok": True, "last_backup": "2026-01-01", "integrity": True}
    def schedule(self):
        return {"ok": True, "maintenance": [], "pending_questions": 0}
    def weekly_lease(self, key):
        return True
    def approve(self, *args):
        return True
    def isolate_dirty(self, *args):
        return False
    def begin_safe(self, *args):
        return "isolated-snapshot"
    def safe_action(self, *args):
        return {"ok": True}
    def verify_action(self, *args):
        return {"ok": True}
    def rollback_safe(self, snapshot):
        return True


def _config(tmp_path, **kw):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (repo / "tests" / "test_value.py").write_text(
        "import unittest\nfrom module import value\n"
        "class T(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(value(), 1)\n",
        encoding="utf-8")
    return Config(repo, tmp_path / "archive", FULL, [FULL], True, disk_min_bytes=0, **kw)


def _fake_copy(archive, label="check"):
    path = archive / "sandboxes" / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    (path / "marker.txt").write_text("fake", encoding="utf-8")
    time.sleep(0.02)  # 保证创建时间严格递增，排序确定
    return path


def test_config_default_and_validation(tmp_path):
    assert _config(tmp_path).archive_keep_copies == 3
    # 逐个坏值单独建配置验 validate（目录互不干扰）
    for i, bad in enumerate((0, -1, 2.5, True, "3")):
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        config = _config(sub, archive_keep_copies=bad)
        with pytest.raises(ValueError):
            config.validate()


def test_prune_keeps_newest_three(tmp_path):
    config = _config(tmp_path)
    hc = Healthcheck(config, FakeAdapter())
    fakes = [_fake_copy(config.archive) for _ in range(5)]
    result = hc.prune_sandbox_copies()
    remaining = sorted(p.name for p in hc._sandbox_copies())
    assert remaining == sorted(p.name for p in fakes[2:])
    assert result["removed"] == [p.name for p in fakes[:2]]
    assert result["remaining"] == 3 and result["errors"] == [] and result["keep"] == 3
    for p in fakes[:2]:
        assert not p.exists()
    for p in fakes[2:]:
        assert p.is_dir()


def test_prune_keep_copies_configurable(tmp_path):
    config = _config(tmp_path, archive_keep_copies=2)
    hc = Healthcheck(config, FakeAdapter())
    fakes = [_fake_copy(config.archive) for _ in range(5)]
    result = hc.prune_sandbox_copies()
    assert result["keep"] == 2 and result["remaining"] == 2
    assert sorted(p.name for p in hc._sandbox_copies()) == sorted(p.name for p in fakes[3:])


def test_prune_ignores_non_copy_entries(tmp_path):
    config = _config(tmp_path)
    hc = Healthcheck(config, FakeAdapter())
    sandboxes = config.archive / "sandboxes"
    fakes = [_fake_copy(config.archive) for _ in range(5)]
    stray_file = sandboxes / "notes.txt"
    stray_file.write_text("x", encoding="utf-8")
    stray_dir = sandboxes / "random-folder"
    stray_dir.mkdir()
    bad_name = sandboxes / "check-nothex"
    bad_name.mkdir()
    file_with_copy_name = sandboxes / f"check-{uuid.uuid4().hex}"
    file_with_copy_name.write_text("x", encoding="utf-8")
    report_file = config.archive / "reports" / "r.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("{}", encoding="utf-8")
    weekly_file = config.archive / "weekly" / "w.json"
    weekly_file.parent.mkdir(parents=True, exist_ok=True)
    weekly_file.write_text("{}", encoding="utf-8")
    result = hc.prune_sandbox_copies()
    assert result["errors"] == [] and result["remaining"] == 3
    for p in (stray_file, stray_dir, bad_name, file_with_copy_name, report_file, weekly_file):
        assert p.exists(), p


def test_prune_protects_given_copy(tmp_path):
    config = _config(tmp_path)
    hc = Healthcheck(config, FakeAdapter())
    fakes = [_fake_copy(config.archive) for _ in range(5)]
    # 极端时序：受保护的恰好是最旧的一个——跳过它，改删后面最旧的。
    result = hc.prune_sandbox_copies(protect=[fakes[0]])
    assert fakes[0].is_dir()
    assert not fakes[1].exists() and not fakes[2].exists()
    assert fakes[3].is_dir() and fakes[4].is_dir()
    assert result["remaining"] == 3


def test_report_applies_retention_and_protects_own_copy(tmp_path):
    config = _config(tmp_path)
    hc = Healthcheck(config, FakeAdapter())
    fakes = [_fake_copy(config.archive) for _ in range(5)]
    path, data = hc.report()
    assert len(data["findings"]) == 8
    retention = data["sandbox_retention"]
    assert retention["errors"] == [] and retention["remaining"] == 3
    copies = hc._sandbox_copies()
    assert len(copies) == 3
    new_names = {p.name for p in copies} - {p.name for p in fakes}
    assert len(new_names) == 1  # 本次运行新建的 check 副本在保留之列
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["sandbox_retention"]["remaining"] == 3
    assert sorted(on_disk["sandbox_retention"]["removed"]) == sorted(retention["removed"])


def test_prune_failure_recorded_and_report_still_completes(tmp_path, monkeypatch):
    config = _config(tmp_path)
    hc = Healthcheck(config, FakeAdapter())
    for _ in range(5):
        _fake_copy(config.archive)

    def boom(path, *args, **kwargs):
        raise OSError("模拟删除失败")

    monkeypatch.setattr(checks_mod.shutil, "rmtree", boom)
    path, data = hc.report()
    assert path.is_file()
    assert len(data["findings"]) == 8
    retention = data["sandbox_retention"]
    assert retention["errors"], retention
    assert retention["removed"] == []
    assert len(hc._sandbox_copies()) == 6  # 5 假 + 1 本次运行新建，全没删掉


def test_build_config_keep_copies(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "memory").mkdir()
    archive = tmp_path / "archive"
    monkeypatch.delenv("SHIER_HEALTHCHECK_KEEP_COPIES", raising=False)
    assert build_config(repo, archive).archive_keep_copies == 3
    monkeypatch.setenv("SHIER_HEALTHCHECK_KEEP_COPIES", "5")
    assert build_config(repo, archive).archive_keep_copies == 5
    explicit = build_config(repo, archive, archive_keep_copies=2)
    assert explicit.archive_keep_copies == 2
