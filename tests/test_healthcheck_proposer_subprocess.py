# tests/test_healthcheck_proposer_subprocess.py（路线①：子进程隔离提案执行器）
#
# BOOS 2026-11-24 拍板路线①：提案函数改具名 + 独立子进程执行，
# 替代 Windows 上同进程守护线程执行器（杀不掉、无隔离墙）。本文件
# 覆盖：具名提案端到端、挂死超时杀进程树并验尸、崩溃干净失败、
# 未登记/不可序列化在启动前拒绝、超时可配置、工厂装配具名句柄。
# 全部只碰临时目录，不连真服务、不动真库。

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from healthcheck import (
    Config, NamedProposer, ProposerRegistry, ProposerRejected,
    ProposerTimeoutError, Repair, SubprocessProposerExecutor,
)
from healthcheck_integration import build_proposer, build_proposer_registry

UNITTEST_SUITE = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]

FIXTURES = r'''
import os
import time


def good(evidence):
    return {"files": {"module.py": "def value():\n    return 1\n"},
            "explanation": "修正值", "cost": 0.0, "calls": 1}


def hang(evidence):
    time.sleep(600)


def slow_ok(evidence):
    time.sleep(1.0)
    return {"files": {"module.py": "def value():\n    return 1\n"},
            "explanation": "慢但好", "cost": 0.0, "calls": 1}


def boom(evidence):
    raise RuntimeError("提案内部爆炸")


def hard_exit(evidence):
    os._exit(3)


def returns_none(evidence):
    return None
'''


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _registry(tmp_path):
    _write(tmp_path / "prop_fixtures.py", FIXTURES)
    registry = ProposerRegistry()
    for name in ("good", "hang", "slow_ok", "boom", "hard_exit", "returns_none"):
        registry.register(name, f"prop_fixtures:{name}")
    return registry


def _named(tmp_path, registry, name, timeout=None):
    return NamedProposer(name, timeout=timeout, registry=registry,
                         extra_sys_path=(str(tmp_path),))


def _repair_fixture(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "module.py", "def value():\n    return 0\n")
    _write(repo / "reproduce.py",
           "from module import value\nassert value() == 1, '原始故障：值不等于1'\n")
    _write(repo / "tests" / "test_value.py",
           "import unittest\nfrom module import value\n"
           "class T(unittest.TestCase):\n    def test_value(self):\n"
           "        self.assertEqual(value(), 1)\n")
    config = Config(repo, tmp_path / "archive", UNITTEST_SUITE,
                    [UNITTEST_SUITE], True, disk_min_bytes=0,
                    repair_paths=["module.py"])
    config.allowed_commands.append([sys.executable, "reproduce.py"])
    return repo, config


def _python_pids():
    if os.name != "nt":
        return None
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) > 1 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


# ---------- (a) 具名提案端到端 ----------

def test_named_proposer_end_to_end_proposal_ready(tmp_path):
    registry = _registry(tmp_path)
    repo, config = _repair_fixture(tmp_path)
    result = Repair(config).execute(
        "S1", [sys.executable, "reproduce.py"],
        _named(tmp_path, registry, "good"))
    assert result["exit_status"] == "proposal_ready", result
    assert result["slow"]["ok"]
    # 源仓库从未被改（提案只落在副本里）
    assert (repo / "module.py").read_text(encoding="utf-8") == \
        "def value():\n    return 0\n"
    # 隔离证据：父进程从未 import 提案模块
    assert "prop_fixtures" not in sys.modules


def test_executor_returns_candidate_dict(tmp_path):
    registry = _registry(tmp_path)
    executor = SubprocessProposerExecutor()
    candidate = executor.run(_named(tmp_path, registry, "good"),
                             {"issue_id": "S0"}, timeout=30, cwd=tmp_path)
    assert candidate["files"]["module.py"].endswith("return 1\n")
    assert candidate["explanation"] == "修正值"
    assert executor.last_child_pid
    assert executor.process_gone(executor.last_child_pid)


# ---------- (b) 挂死：超时杀树、验尸、父进程立刻再跑 ----------

def test_hang_killed_no_orphan_and_parent_recovers(tmp_path):
    registry = _registry(tmp_path)
    executor = SubprocessProposerExecutor()
    before = _python_pids()
    started = time.monotonic()
    with pytest.raises(ProposerTimeoutError):
        executor.run(_named(tmp_path, registry, "hang"),
                     {"issue_id": "S2"}, timeout=2, cwd=tmp_path)
    assert time.monotonic() - started < 30
    pid = executor.last_child_pid
    assert pid and executor.last_kill_method
    # 验尸：子进程确实没了，进程列表无新增残留 python
    assert executor.process_gone(pid)
    after = _python_pids()
    if before is not None and after is not None:
        assert pid not in after
        assert after - before == set()
    # 父进程无恙：同一执行器立刻再跑正常提案
    candidate = executor.run(_named(tmp_path, registry, "good"),
                             {"issue_id": "S2b"}, timeout=30, cwd=tmp_path)
    assert candidate["explanation"] == "修正值"


def test_named_hang_via_repair_is_time_budget(tmp_path):
    registry = _registry(tmp_path)
    _repo, config = _repair_fixture(tmp_path)
    config.max_seconds = 4.0
    result = Repair(config).execute(
        "S3", [sys.executable, "reproduce.py"],
        _named(tmp_path, registry, "hang"))
    assert result["exit_status"] == "time_budget", result


# ---------- (c) 崩溃：干净失败、父进程无恙 ----------

def test_crash_exception_and_hard_exit(tmp_path):
    registry = _registry(tmp_path)
    executor = SubprocessProposerExecutor()
    with pytest.raises(RuntimeError, match="提案内部爆炸"):
        executor.run(_named(tmp_path, registry, "boom"),
                     {"issue_id": "S4"}, timeout=30, cwd=tmp_path)
    with pytest.raises(RuntimeError, match="exit=3"):
        executor.run(_named(tmp_path, registry, "hard_exit"),
                     {"issue_id": "S5"}, timeout=30, cwd=tmp_path)
    candidate = executor.run(_named(tmp_path, registry, "good"),
                             {"issue_id": "S6"}, timeout=30, cwd=tmp_path)
    assert candidate["explanation"] == "修正值"


def test_named_boom_via_repair_is_executor_error(tmp_path):
    registry = _registry(tmp_path)
    _repo, config = _repair_fixture(tmp_path)
    result = Repair(config).execute(
        "S7", [sys.executable, "reproduce.py"],
        _named(tmp_path, registry, "boom"))
    assert result["exit_status"] == "executor_error", result


# ---------- (d)(e) 启动前拒绝 ----------

def test_unregistered_name_rejected_before_spawn(tmp_path):
    registry = _registry(tmp_path)
    executor = SubprocessProposerExecutor()
    with pytest.raises(ProposerRejected):
        executor.run(_named(tmp_path, registry, "nope"),
                     {"issue_id": "S8"}, timeout=30, cwd=tmp_path)
    assert executor.last_child_pid is None


def test_nonserializable_evidence_rejected_before_spawn(tmp_path):
    registry = _registry(tmp_path)
    executor = SubprocessProposerExecutor()
    with pytest.raises(ProposerRejected):
        executor.run(_named(tmp_path, registry, "good"),
                     {"issue_id": "S9", "bad": object()},
                     timeout=30, cwd=tmp_path)
    assert executor.last_child_pid is None


def test_registry_rejects_bad_spec_and_bad_name():
    registry = ProposerRegistry()
    with pytest.raises(ProposerRejected):
        registry.register("ok_name", "not-a-spec")
    with pytest.raises(ProposerRejected):
        registry.register("坏名字", "mod:func")
    with pytest.raises(ProposerRejected):
        registry.spec_for("missing")


# ---------- (f) 超时可配置 ----------

def test_timeout_is_configurable(tmp_path):
    registry = _registry(tmp_path)
    executor = SubprocessProposerExecutor()
    candidate = executor.run(_named(tmp_path, registry, "slow_ok"),
                             {"issue_id": "S10"}, timeout=10, cwd=tmp_path)
    assert candidate["explanation"] == "慢但好"
    with pytest.raises(ProposerTimeoutError):
        executor.run(_named(tmp_path, registry, "slow_ok"),
                     {"issue_id": "S11"}, timeout=0.2, cwd=tmp_path)


# ---------- 无提案 + 工厂接线 ----------

def test_named_none_via_repair_is_no_candidate(tmp_path):
    registry = _registry(tmp_path)
    _repo, config = _repair_fixture(tmp_path)
    result = Repair(config).execute(
        "S12", [sys.executable, "reproduce.py"],
        _named(tmp_path, registry, "returns_none"))
    assert result["exit_status"] == "no_candidate", result


def test_factory_builds_named_proposer(tmp_path):
    assert build_proposer_registry().names() == ["no_candidate"]
    _repo, config = _repair_fixture(tmp_path)
    result = Repair(config).execute(
        "S13", [sys.executable, "reproduce.py"], build_proposer())
    assert result["exit_status"] == "no_candidate", result
