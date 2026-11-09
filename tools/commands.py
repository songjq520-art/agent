# tools/commands.py（第一批工具：跑程序/跑命令）
# 作用：十二在工作区里跑命令的工具，安全按新方案的两层做：
#   - 结构化调用：只收命令列表（["python", "script.py"]），不用 shell=True、
#     不让模型拼 shell 字符串（GPT 的建议）。
#   - Tier 分级（tools/permissions.py）：Tier 2 硬禁绝不执行（就算确认了
#     也不跑）；Tier 1 需 BOOS 确认；Tier 0 直接跑。校验在执行前跑。
#   - 超时默认 60 秒、上限 300 秒，超时杀掉整个进程组（POSIX 上 killpg，
#     Windows 上 taskkill /F /T），杀完复查进程是不是真的死了（结果里标
#     verified 还是 assumed）；子进程占着管道会让收输出挂住，限时等、
#     超时就关掉管道。输出截断（stdout/stderr 各截到 1 万字，标明截断了）；
#     工作目录锁在工作区内；不把 API Key 等环境变量传给被跑的命令（只传最小白名单）。
#   - 确认必须当场、当次、来自 BOOS 本人：Tier 1 的确认带作用域，
#     必须和本次操作完全一致，一次确认只顶一次操作；审计里记确认来源。
#   - 命令的来历只能是 BOOS 原话或 agent 自己的推理，外部文件/网页里读到
#     的命令串不直接拿来跑。
#   - 每次执行和每次拒绝都记审计；审计日志里密钥打码，绝不出现密钥原文。
#
# 诚实说明：分级防的是误操作和提示注入下的灾难命令，不是防有意逃逸——
# 命令本身能读写工作区外的路径（比如 python 脚本里写绝对路径），
# 这是第一批的已知缺口，靠“只跑自己信任的命令 + 危险的先问 BOOS”兜着，
# 真正的隔离靠 OS 层（Box 上先测 gVisor），见 tools/README.md。
# 内联代码（python -c / -m 等）升到 Tier 1 后，写进文件的脚本还是 Tier 0：
# 文件写进去的时候已经要过一次确认；但如果以后放宽了写确认，跑脚本
# 也要重新评估，见 tools/README.md。

import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.audit import AuditLog
from tools.errors import NeedsConfirmation, PermissionDenied, ToolError
from tools.permissions import (
    classify_command,
    confirmation_scope_for_command,
    resolve_confirmation,
    resolve_in_workspace,
)

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


def _kill_process_tree_windows(pid):
    """Windows 上杀整棵进程树：taskkill /F /T。
    返回 (ok, reason)。taskkill 是 Windows 自带的，不用另外装；
    它失败了（比如 taskkill 本身不在）就返回 False，调用方再补一刀
    只杀主进程，并把“树不一定死了”写进结果里。"""
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return result.returncode == 0, f"taskkill 退出码 {result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"taskkill 调用失败：{exc}"


def _pid_alive_windows(pid):
    """Windows 上查一个进程还在不在：tasklist 里找得到就是还活着。
    查不出来返回 None（不知道）。"""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == str(pid):
                return True
        return False
    except (OSError, subprocess.TimeoutExpired):
        return None


def _pid_alive(pid):
    """查一个进程还在不在（跨平台）：True 活着、False 死了、None 不知道。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _process_group_alive(pid):
    """POSIX：进程组里还有没有活着的进程（killpg 发 0 信号探一下）。
    True 还有活着的、False 组里死光了、None 不知道。"""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


class CommandRunner:
    """在固定工作区里跑结构化命令。"""

    def __init__(self, workspace, require_confirmation=True, audit_log=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.require_confirmation = require_confirmation
        self.audit_log = audit_log or AuditLog(self.workspace / ".audit" / "audit.log")

    def run(self, argv, timeout=_DEFAULT_TIMEOUT, confirmed=False, actor="agent", cwd_rel=".", origin="agent"):
        """跑一条命令（列表）。返回 returncode / stdout / stderr / 是否超时等。
        origin 是命令的来历："user"（BOOS 原话）或 "agent"（agent 自己推理）；
        "external"（从文件/网页/记忆里读到的命令串）绝不直接拿来跑——
        外部内容里的指令性文字不能指挥 agent 做事，命令只能来自 BOOS 原话
        或 agent 自己的正常推理。
        超时后返回结果里有 tree_kill：杀整棵进程树的方法和核对结果，
        verified 表示复查过确实死了，assumed 表示只信了 kill 的返回值。"""
        if origin not in ("user", "agent"):
            raise PermissionDenied(
                f"外部来源的命令不直接执行：{origin}（命令只能来自 BOOS 原话或 agent 自己的推理）"
            )
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
                origin=origin,
            )
            raise PermissionDenied(f"硬禁命令，绝不执行：{reason}（{command_text}）")
        confirm_source = "not-required"
        if tier == 1 and self.require_confirmation:
            scope = confirmation_scope_for_command(argv)
            confirm_source, confirm_reason = resolve_confirmation(confirmed, scope)
            if confirm_source is None:
                self.audit_log.record(
                    tool="commands",
                    action="run",
                    path=cwd_rel,
                    actor=actor,
                    confirmed=False,
                    success=False,
                    reason=confirm_reason,
                    extra={"argv": [str(arg) for arg in argv], "tier": 1},
                    origin=origin,
                )
                raise NeedsConfirmation(f"需要确认后才能跑：{command_text}（{reason}；{confirm_reason}）")
        timeout = max(1, min(_MAX_TIMEOUT, timeout or _DEFAULT_TIMEOUT))
        started = time.monotonic()
        timed_out = False
        output_discarded = False
        tree_kill = None
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
                tree_kill = self._kill_tree(process)
                # 子进程占着管道会让 communicate() 挂住：限时等，超时就关掉管道，
                # 别在这儿一直等下去；输出丢掉，记下来。
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    for stream in (process.stdout, process.stderr):
                        try:
                            if stream is not None:
                                stream.close()
                        except Exception:
                            pass
                    stdout, stderr = "", ""
                    output_discarded = True
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                returncode = process.returncode
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
            confirmed=bool(confirmed) if tier == 1 else False,
            success=(returncode == 0 and not timed_out),
            reason="超时，已杀掉" if timed_out else "",
            extra={
                "argv": [str(arg) for arg in argv],
                "tier": tier,
                "returncode": returncode,
                "timed_out": timed_out,
                "duration_s": round(duration, 2),
                "tree_kill": tree_kill,
                "output_discarded": output_discarded,
            },
            confirmation_source=confirm_source if tier == 1 else "",
            origin=origin,
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
            "tree_kill": tree_kill,
            "output_discarded": output_discarded,
        }

    def _kill_tree(self, process):
        """超时后杀整棵进程树，杀完再核对是不是真的死了。
        返回 dict：method（怎么杀的）、parent_dead（主进程死了没）、
        group_dead（POSIX 上进程组死光了没）、verified（verified 是复查过
        确实死了，assumed 是只信了 kill 的返回值，failed 是核对发现还活着）。
        别只信 kill 的返回值——杀完必须复查。"""
        pid = process.pid
        result = {"method": None, "parent_dead": None, "group_dead": None, "verified": "failed"}
        try:
            if os.name == "posix":
                result["method"] = "killpg"
                os.killpg(pid, signal.SIGKILL)
            else:
                result["method"] = "taskkill /F /T"
                ok, reason = _kill_process_tree_windows(pid)
                result["taskkill_ok"] = ok
                result["taskkill_reason"] = reason
                if not ok:
                    process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            # 先把主进程收回来（wait），不然死了的进程还以僵尸形态占着 PID，
            # 复查时会误判“还活着”。
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.2)  # 给杀信号一点时间生效，再复查
        if os.name == "posix":
            result["parent_dead"] = _pid_alive(pid) is False
            group_alive = _process_group_alive(pid)
            result["group_dead"] = group_alive is False
            if result["parent_dead"] and result["group_dead"]:
                result["verified"] = "verified"
            elif result["parent_dead"]:
                result["verified"] = "assumed"
            # 主进程还活着：verified 保持默认的 failed。
        else:
            alive = _pid_alive_windows(pid)
            result["parent_dead"] = alive is False
            # Windows 上没有进程组这一说：taskkill /T 报成功 + 主进程死了，
            # 树大概率死了，但没法逐个复查孙子进程，只能算 assumed。
            if result["parent_dead"]:
                result["verified"] = "assumed"
        return result
