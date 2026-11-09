# tests/test_tools_commands.py（第一批工具：跑命令测试）
# 作用：验证 tools/commands.py —— 结构化调用（命令列表，不拼 shell 字符串）、
# Tier 2 硬禁绝不执行、Tier 1 需确认、Tier 0 直接跑、超时杀进程、输出截断、
# 工作目录锁在工作区、不把 Key 传给被跑的命令、操作审计。
# 全部离线，只跑本地无害命令，不产生费用。

import os
import sys

import pytest

from tools.audit import AuditLog
from tools.commands import CommandRunner
from tools.errors import NeedsConfirmation, PermissionDenied


@pytest.fixture
def runner(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return CommandRunner(workspace, audit_log=AuditLog(tmp_path / "audit.log"))


def test_tier0_runs(runner):
    out = runner.run([sys.executable, "-c", "print('hi')"])
    assert out["returncode"] == 0
    assert out["stdout"].strip() == "hi"
    assert out["tier"] == 0
    assert out["timed_out"] is False


def test_tier2_never_runs_even_confirmed(runner, tmp_path):
    marker = tmp_path / "pwned.txt"
    with pytest.raises(PermissionDenied):
        runner.run(
            ["sh", "-c", f"echo hi > {marker} && rm -rf /"], confirmed=True
        )
    assert not marker.exists()


def test_tier1_needs_confirmation_then_runs(runner):
    target = runner.workspace / "old.txt"
    target.write_text("x", encoding="utf-8")
    delete_cmd = (
        ["cmd", "/c", "del", "old.txt"]
        if os.name == "nt"
        else ["rm", "old.txt"]
    )
    with pytest.raises(NeedsConfirmation):
        runner.run(delete_cmd)
    assert target.exists()
    out = runner.run(delete_cmd, confirmed=True)
    assert out["returncode"] == 0
    assert not target.exists()


def test_timeout_kills_process(runner):
    out = runner.run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout=1,
    )
    assert out["timed_out"] is True


def test_output_truncated(runner):
    out = runner.run([sys.executable, "-c", "print('x' * 20000)"])
    assert out["truncated"] is True
    assert len(out["stdout"]) <= 10000 + 100  # 截断上限 + 截断提示
    assert "截断" in out["stdout"]


def test_env_secrets_not_passed_to_child(runner, monkeypatch):
    monkeypatch.setenv("FAKE_TEST_API_KEY", "secret123")
    out = runner.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('FAKE_TEST_API_KEY', 'ABSENT'))",
        ]
    )
    assert out["stdout"].strip() == "ABSENT"


def test_working_directory_locked_to_workspace(runner):
    out = runner.run(
        [sys.executable, "-c", "import os; print(os.getcwd())"]
    )
    assert out["stdout"].strip() == str(runner.workspace)


def test_failed_command_returns_result_not_exception(runner):
    out = runner.run([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert out["returncode"] == 3
    assert out["timed_out"] is False


def test_audit_records_execution(runner):
    runner.run([sys.executable, "-c", "print('hi')"])
    events = runner.audit_log.read()
    assert len(events) == 1
    event = events[0]
    assert event["tool"] == "commands"
    assert event["action"] == "run"
    assert event["success"] is True
