"""路线①：子进程隔离的提案执行器（BOOS 2026-11-24 拍板）。

背景：原 proposer 契约传任意 callable，Windows 没有 fork，移植时
只能退到同进程守护线程 + 硬超时——线程杀不掉、没有隔离墙。本文件
把 proposer 改成「具名函数 + 独立子进程」：

- 白名单注册表：提案方只能用注册名点名，注册名背后的真实落点是
  「模块路径:函数名」字符串。父进程只持有名字与字符串，**绝不在
  自己进程里 import 或执行提案代码**；未登记的名字在启动子进程
  之前就拒绝（ProposerRejected）。
- 子进程执行：用当前解释器（sys.executable，服务里就是服务自己
  的 venv python）起一个独立子进程，任务说明（落点 + JSON 参数 +
  工作目录 + sys.path）经临时文件交给它；子进程 import 具名函数、
  执行，把结果 JSON 写回结果文件后退出（见 proposer_child.py）。
- 硬超时可杀：父进程带可配置超时等结果；超时在 Windows 用
  taskkill /T /F 杀整个进程树（POSIX 用进程组 SIGKILL），并回查
  进程确实已消失才返回；子进程崩溃/非零退出变成干净的失败
  （RuntimeError），异常语义与原 fork 路径一致。父进程在任何情况
  下保持健康，杀完能立刻再跑下一个提案。
- 参数（evidence）不可 JSON 序列化时在启动前干净拒绝。

本模块不 import repair（避免循环）：超时抛 ProposerTimeoutError，
由 Repair.call_proposer 转成 BudgetExit("time_budget")；候选以纯
dict 返回，由 Repair 转成 Candidate。
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")
_SPEC_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r":([A-Za-z_][A-Za-z0-9_]*)")


class ProposerRejected(ValueError):
    """启动子进程之前就拒绝：未登记的名字 / 落点格式坏 / 参数不可序列化。"""


class ProposerTimeoutError(RuntimeError):
    """子进程超时且进程树已被杀净；Repair 层转 BudgetExit("time_budget")。"""


class ProposerRegistry:
    """具名提案函数白名单：注册名 → 「模块路径:函数名」。

    注册只存字符串，父进程永不因注册而 import 提案模块。
    """

    def __init__(self):
        self._specs = {}

    def register(self, name, spec):
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ProposerRejected(f"提案注册名格式错误：{name!r}")
        if not isinstance(spec, str) or not _SPEC_RE.fullmatch(spec):
            raise ProposerRejected(f"提案落点必须是「模块路径:函数名」：{spec!r}")
        self._specs[name] = spec
        return name

    def spec_for(self, name):
        try:
            return self._specs[name]
        except KeyError:
            raise ProposerRejected(f"提案未登记，拒绝启动子进程：{name!r}") from None

    def names(self):
        return sorted(self._specs)

    def __contains__(self, name):
        return name in self._specs


default_registry = ProposerRegistry()


def register_proposer(name, spec, registry=None):
    return (registry or default_registry).register(name, spec)


@dataclasses.dataclass(frozen=True)
class NamedProposer:
    """传给 Repair.execute 的具名提案句柄：只有名字与执行参数，不含代码。

    timeout=None 表示用 Repair 的剩余预算作硬超时；extra_sys_path 是
    提案模块所在的额外目录（子进程侧插入 sys.path，父进程不 import）。
    """

    name: str
    timeout: float | None = None
    registry: ProposerRegistry | None = None
    extra_sys_path: tuple = ()

    def resolve_spec(self):
        return (self.registry or default_registry).spec_for(self.name)


_CHILD_SCRIPT = Path(__file__).resolve().with_name("proposer_child.py")
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


class SubprocessProposerExecutor:
    """把一次提案调用放进独立子进程执行；父进程只收结果 dict / None。"""

    def __init__(self, python_executable=None):
        self.python_executable = python_executable or sys.executable
        self.last_child_pid = None
        self.last_kill_method = None

    @staticmethod
    def process_gone(pid):
        """回查 pid 是否确实已不存在（杀后验尸用）。"""
        if pid is None:
            return True
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=30)
            return not re.search(rf"\b{pid}\b", result.stdout or "")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _kill_tree(self, proc):
        """杀掉子进程整个进程树，并等它真的消失。"""
        pid = proc.pid
        if os.name == "nt":
            self.last_kill_method = "taskkill /T /F"
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=30)
        else:
            self.last_kill_method = "killpg SIGKILL"
            try:
                os.killpg(os.getpgid(pid), 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        deadline = time.monotonic() + 10
        while not self.process_gone(pid):
            if time.monotonic() > deadline:
                raise RuntimeError(f"提案子进程 pid={pid} 超时后仍未确认消失")
            time.sleep(0.1)

    def run(self, named, evidence, *, timeout, cwd):
        """执行一次具名提案。返回候选 dict 或 None；失败抛异常（语义见模块头）。

        未登记名字 / 参数不可序列化 → ProposerRejected（不启动子进程）；
        超时 → ProposerTimeoutError（进程树已杀净）；子进程内异常或
        崩溃/非零退出 → RuntimeError。
        """
        spec = named.resolve_spec()
        sys_path = [str(cwd), *(str(p) for p in named.extra_sys_path),
                    _PACKAGE_PARENT]
        task = {"spec": spec, "evidence": evidence, "cwd": str(cwd),
                "sys_path": sys_path}
        try:
            task_json = json.dumps(task, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ProposerRejected(
                f"提案参数不可 JSON 序列化，拒绝启动子进程：{exc}") from None
        workdir = Path(tempfile.mkdtemp(prefix="hc-proposer-"))
        task_file = workdir / "task.json"
        result_file = workdir / "result.json"
        task_file.write_text(task_json, encoding="utf-8")
        popen_kwargs = {}
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [self.python_executable, str(_CHILD_SCRIPT),
             str(task_file), str(result_file)],
            cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", **popen_kwargs)
        self.last_child_pid = proc.pid
        try:
            try:
                _stdout, stderr = proc.communicate(timeout=max(0.01, timeout))
            except subprocess.TimeoutExpired:
                self._kill_tree(proc)
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
                raise ProposerTimeoutError(
                    f"提案子进程超时（{timeout:.2f}s），进程树已杀 "
                    f"pid={proc.pid} method={self.last_kill_method}") from None
            if proc.returncode != 0:
                tail = (stderr or "").strip()[-500:]
                raise RuntimeError(
                    f"提案子进程异常退出（exit={proc.returncode}）"
                    + (f"：{tail}" if tail else ""))
            if not result_file.exists():
                raise RuntimeError("提案子进程未产出结果文件（exit=0）")
            outcome = json.loads(result_file.read_text(encoding="utf-8"))
            if outcome.get("status") == "error":
                raise RuntimeError(f"提案函数异常：{outcome.get('error')}")
            if outcome.get("status") != "ok":
                raise RuntimeError(f"提案子进程结果格式错误：{outcome!r}")
            return outcome.get("candidate")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
