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
    # Tier 0 跑已经写好的脚本文件：文件写进去的时候已经要过确认。
    script = runner.workspace / "hello.py"
    script.write_text("print('hi')", encoding="utf-8")
    out = runner.run([sys.executable, "hello.py"])
    assert out["returncode"] == 0
    assert out["stdout"].strip() == "hi"
    assert out["tier"] == 0
    assert out["timed_out"] is False


def test_inline_code_needs_confirmation(runner):
    # python -c 是 Tier 1：现拼的代码，没经过“写文件要确认”那一关。
    with pytest.raises(NeedsConfirmation):
        runner.run([sys.executable, "-c", "print('hi')"])
    out = runner.run([sys.executable, "-c", "print('hi')"], confirmed=True)
    assert out["returncode"] == 0
    assert out["stdout"].strip() == "hi"
    assert out["tier"] == 1


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
        confirmed=True,
    )
    assert out["timed_out"] is True
    assert out["tree_kill"] is not None
    assert out["tree_kill"]["verified"] in ("verified", "assumed")


def test_output_truncated(runner):
    out = runner.run([sys.executable, "-c", "print('x' * 20000)"], confirmed=True)
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
        ],
        confirmed=True,
    )
    assert out["stdout"].strip() == "ABSENT"


def test_working_directory_locked_to_workspace(runner):
    out = runner.run(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        confirmed=True,
    )
    assert out["stdout"].strip() == str(runner.workspace)


def test_failed_command_returns_result_not_exception(runner):
    out = runner.run(
        [sys.executable, "-c", "import sys; sys.exit(3)"], confirmed=True
    )
    assert out["returncode"] == 3
    assert out["timed_out"] is False


def test_audit_records_execution(runner):
    runner.run([sys.executable, "-c", "print('hi')"], confirmed=True)
    events = runner.audit_log.read()
    assert len(events) == 1
    event = events[0]
    assert event["tool"] == "commands"
    assert event["action"] == "run"
    assert event["success"] is True


# ---------- 本次安全修复新增 ----------

def test_scoped_confirmation_rejects_wrong_scope(runner):
    # 确认是给另一条命令的，顶不了这一条：必须重新问 BOOS。
    wrong_scope = {"by": "cli", "scope": "run:pip install requests"}
    with pytest.raises(NeedsConfirmation):
        runner.run(
            [sys.executable, "-c", "print('hi')"],
            confirmed=wrong_scope,
        )
    # 给对了 scope 就能跑，且审计里记的是“用户当场确认”。
    right_scope = {"by": "cli", "scope": f"run:{sys.executable} -c print('hi')"}
    out = runner.run(
        [sys.executable, "-c", "print('hi')"],
        confirmed=right_scope,
    )
    assert out["returncode"] == 0
    events = runner.audit_log.read()
    assert events[-1]["confirmation_source"] == "user-scoped"


def test_bool_confirmation_is_caller_attested_not_user(runner):
    # confirmed=True 只是调用方自己作证，没法冒充 BOOS 当场确认：
    # 能跑（兼容老调用），但审计里如实记 caller-attested。
    out = runner.run(
        [sys.executable, "-c", "print('hi')"],
        confirmed=True,
    )
    assert out["returncode"] == 0
    events = runner.audit_log.read()
    assert events[-1]["confirmation_source"] == "caller-attested"


def test_external_origin_command_refused(runner):
    # 从文件/网页里读到的命令串不直接拿来跑，哪怕它看起来无害。
    with pytest.raises(PermissionDenied):
        runner.run(
            [sys.executable, "-c", "print('hi')"],
            confirmed=True,
            origin="external",
        )


def test_timeout_kills_whole_process_tree(runner, tmp_path):
    # 真实进程树：父进程起一个子进程（sleep 30），父进程也睡；
    # 超时 1 秒后，父子都必须死了，且结果里标 verified。
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(child_pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    out = runner.run(
        [sys.executable, str(script)],
        timeout=1,
    )
    assert out["timed_out"] is True
    assert out["tree_kill"]["verified"] == "verified"
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_windows_tree_kill_builds_taskkill_command(monkeypatch):
    # Linux 上跑不了 taskkill，但 Windows 分支拼的命令必须是对的：
    # taskkill /F /T /PID <pid>，缺一个都可能只杀主进程、留下子进程。
    from tools.commands import _kill_process_tree_windows

    seen = {}

    class FakeResult:
        returncode = 0

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return FakeResult()

    monkeypatch.setattr("tools.commands.subprocess.run", fake_run)
    ok, _reason = _kill_process_tree_windows(1234)
    assert ok is True
    assert seen["argv"] == ["taskkill", "/F", "/T", "/PID", "1234"]


@pytest.mark.skipif(os.name != "nt", reason="taskkill 只在 Windows 上有")
def test_windows_taskkill_kills_tree(runner, tmp_path):
    # Windows 专属：真跑一棵进程树，超时后 taskkill /F /T 必须把整棵树杀掉。
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(child_pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    out = runner.run([sys.executable, str(script)], timeout=1)
    assert out["timed_out"] is True
    assert out["tree_kill"]["method"] == "taskkill /F /T"
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
