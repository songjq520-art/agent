# tools/commands.py（第一批工具：跑程序/跑命令）
# 作用：十二在工作区里跑命令的工具，安全按新方案的两层做：
#   - 结构化调用：只收命令列表（["python", "script.py"]），不用 shell=True、
#     不让模型拼 shell 字符串（GPT 的建议）。
#   - Tier 分级（tools/permissions.py）：Tier 2 硬禁绝不执行（就算确认了
#     也不跑）；Tier 1 需 BOOS 确认；Tier 0 直接跑。校验在执行前跑。
#   - 超时默认 60 秒、上限 300 秒，超时杀掉整个进程组；输出截断（stdout/
#     stderr 各截到 1 万字，标明截断了）；工作目录锁在工作区内；
#     不把 API Key 等环境变量传给被跑的命令（只传最小白名单）。
#   - 每次执行和每次拒绝都记审计。
#
# 诚实说明：分级防的是误操作和提示注入下的灾难命令，不是防有意逃逸——
# 命令本身能读写工作区外的路径（比如 python 脚本里写绝对路径），
# 这是第一批的已知缺口，靠“只跑自己信任的命令 + 危险的先问 BOOS”兜着，
# 真正的隔离靠 OS 层（Box 上先测 gVisor），见 tools/README.md。

import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.audit import AuditLog
from tools.errors import NeedsConfirmation, PermissionDenied, ToolError
from tools.permissions import classify_command, resolve_in_workspace

_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 300
_OUTPUT_MAX_CHARS = 10000

# 传给子命令的最小环境白名单：只要跑命令够用的，不含任何 Key/密钥。
_ENV_ALLOWLIST = {
    "PATH",
    "SYSTEMROOT",
    "SystemRoot",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "USER",
    "LOGNAME",
    "ComSpec",
    "windir",
    "PATHEXT",
}


def clean_env():
    """给子命令的最小环境：白名单里的才传，Key/密钥一律不传。"""
    return {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}


def _truncate(text, limit=_OUTPUT_MAX_CHARS):
    """输出截断：超了只留前 limit 字，并标明截断了。"""
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n…（输出已截断，只保留前 10000 字）", True


class CommandRunner:
    """在固定工作区里跑结构化命令。"""

    def __init__(self, workspace, require_confirmation=True, audit_log=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.require_confirmation = require_confirmation
        self.audit_log = audit_log or AuditLog(self.workspace / ".audit" / "audit.log")

    def run(self, argv, timeout=_DEFAULT_TIMEOUT, confirmed=False, actor="agent", cwd_rel="."):
        """跑一条命令（列表）。返回 returncode / stdout / stderr / 是否超时等。"""
        tier, reason = classify_command(argv)
        try:
            cwd = resolve_in_workspace(self.workspace, cwd_rel)
        except PermissionDenied as exc:
            raise ToolError(f"工作目录超出工作区：{exc}") from exc
        if not cwd.is_dir():
            raise ToolError(f"工作目录不存在：{cwd_rel}")
        command_text = " ".join(str(arg) for arg in argv)
        if tier == 2:
            self.audit_log.record(
                tool="commands",
                action="run",
                path=cwd_rel,
                actor=actor,
                confirmed=False,
                success=False,
                reason=f"硬禁命令，拒绝执行：{reason}",
                extra={"argv": [str(arg) for arg in argv], "tier": 2},
            )
            raise PermissionDenied(f"硬禁命令，绝不执行：{reason}（{command_text}）")
        if tier == 1 and self.require_confirmation and not confirmed:
            self.audit_log.record(
                tool="commands",
                action="run",
                path=cwd_rel,
                actor=actor,
                confirmed=False,
                success=False,
                reason="需要 BOOS 确认后才能跑",
                extra={"argv": [str(arg) for arg in argv], "tier": 1},
            )
            raise NeedsConfirmation(f"需要确认后才能跑：{command_text}（{reason}）")
        timeout = max(1, min(_MAX_TIMEOUT, timeout or _DEFAULT_TIMEOUT))
        started = time.monotonic()
        timed_out = False
        returncode = None
        stdout = ""
        stderr = ""
        try:
            process = subprocess.Popen(
                [str(arg) for arg in argv],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env=clean_env(),
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except (OSError, ProcessLookupError):
                    pass
                stdout, stderr = process.communicate()
                returncode = None
        except FileNotFoundError as exc:
            raise ToolError(f"命令不存在：{argv[0]}") from exc
        except OSError as exc:
            raise ToolError(f"命令启动失败：{exc}") from exc
        duration = time.monotonic() - started
        stdout, stdout_truncated = _truncate(stdout or "")
        stderr, stderr_truncated = _truncate(stderr or "")
        truncated = stdout_truncated or stderr_truncated
        self.audit_log.record(
            tool="commands",
            action="run",
            path=cwd_rel,
            actor=actor,
            confirmed=confirmed if tier == 1 else False,
            success=(returncode == 0 and not timed_out),
            reason="超时，已杀掉" if timed_out else "",
            extra={
                "argv": [str(arg) for arg in argv],
                "tier": tier,
                "returncode": returncode,
                "timed_out": timed_out,
                "duration_s": round(duration, 2),
            },
        )
        return {
            "argv": [str(arg) for arg in argv],
            "tier": tier,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
            "timed_out": timed_out,
            "duration_s": round(duration, 2),
        }
