"""线性自修：复现→诊断→提案→快门→全套慢门→结论；默认只提案。

并入十二时的移植改动（2026-11-24，十一）：
- proposer 执行器：有 fork 的平台仍走子进程；Windows 无 fork，
  改走同进程守护线程 + 硬超时（提案代码是受信代码，边界见
  _call_proposer_thread 的 docstring），实测见集成测试。
- 路线①（2026-11-24 BOOS 拍板，十一实现）：proposer 若是
  NamedProposer（具名句柄），一律走 proposer_subprocess 的子进程
  隔离执行器——父进程只点名、不 import 提案代码，超时杀整个进程
  树。裸 callable 的 fork/线程路径保留兼容（既有测试与 POSIX 用），
  但 Windows 工厂装配只发具名句柄，不再用线程执行器跑提案。
- git() 子进程显式 encoding="utf-8", errors="replace"：原版
  text=True 用系统区域编码，中文 Windows（GBK）下解 git 的
  UTF-8 输出会让读取线程抛 UnicodeDecodeError、stdout 变
  None，整条自修路径直接死——这是移植实测抓到的真 bug。
- git() 的 PATH/HOME 隔离在 Windows 下改用模块存档目录内的
  空 gitconfig 与空 hooks 目录代替 /dev/null。
"""
from __future__ import annotations
import dataclasses
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from .core import DEFAULT_MODE, Trace, Unconnected, confined, emit, redact
from .gates import Gates, GateConfigError, capability_diff, copy_repo, run, command_check
from .proposer_subprocess import (
    NamedProposer, ProposerTimeoutError, SubprocessProposerExecutor)


@dataclasses.dataclass
class Candidate:
    files: dict[str, str]
    explanation: str
    cost: float = 0.0
    calls: int = 1


class BudgetExit(RuntimeError):
    pass


class Repair:
    def __init__(self, config, adapter=None):
        config.validate()
        config.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
        config.archive.chmod(0o700)
        self.config = config
        self.adapter = adapter or Unconnected()
        self.trace = Trace(config.archive / "traces" / uuid.uuid4().hex)
        self.ledger = []
        self.steps = 0
        self.cost = 0.0
        self.calls = 0
        self.started = time.monotonic()

    def event(self, step, **data):
        return self.trace.event(step, cost=self.cost, calls=self.calls,
                                elapsed_seconds=time.monotonic() - self.started, **data)

    def attempt(self, result, **data):
        entry = self.event("尝试台账", result=result, **data)
        self.ledger.append(entry)

    def budget(self, reserve):
        import math
        if not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
            raise GateConfigError("模型单次花费上界必须为有限非负数")
        if self.steps >= self.config.max_steps:
            raise BudgetExit("step_budget")
        if self.cost + reserve > self.config.max_cost:
            raise BudgetExit("cost_budget")
        if self.remaining() <= 0:
            raise BudgetExit("time_budget")

    def call_proposer(self, proposer, evidence):
        # 具名句柄（路线①）：子进程隔离执行，父进程从不 import 提案代码。
        if isinstance(proposer, NamedProposer):
            return self._call_proposer_subprocess(proposer, evidence)
        # 独立子进程硬超时；POSIX 用 fork，不在父进程执行不可取消回调。
        # 移植注记（2026-11-24）：十二真服务跑在 Windows 上，没有 fork——
        # 无 fork 的平台走 _call_proposer_thread（守护线程 + 硬超时，
        # 超时后放弃结果；线程不可强杀是已知边界，proposer 是可信集成
        # 代码而非不可信代码，预算闸的语义保持不变）。
        import multiprocessing
        if "fork" not in multiprocessing.get_all_start_methods():
            return self._call_proposer_thread(proposer, evidence)
        context = multiprocessing.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        def invoke():
            receiver.close()
            try:
                sender.send((True, proposer(evidence)))
            except Exception as exc:
                sender.send((False, str(exc)))
            finally:
                sender.close()
        process = context.Process(target=invoke)
        process.start()
        sender.close()
        try:
            if not receiver.poll(max(0, self.remaining())):
                raise BudgetExit("time_budget")
            ok, value = receiver.recv()
            if not ok:
                raise RuntimeError(value)
            return value
        finally:
            receiver.close()
            process.join(timeout=0.05)
            if process.is_alive():
                process.kill()
                process.join()

    def _call_proposer_subprocess(self, named, evidence):
        """路线①执行器接线：子进程隔离跑具名提案函数。

        硬超时 = min(句柄自带 timeout（若有）, 本轮剩余预算)；超时在
        执行器内杀净进程树后转 BudgetExit("time_budget")，与 fork /
        线程路径的超时语义一致；未登记名字 / 参数不可序列化在启动
        子进程前被执行器拒绝（ProposerRejected，属 ValueError，
        execute 归 executor_error）；子进程内异常/崩溃为 RuntimeError。
        候选以 dict 回来，在这里转 Candidate，后续三道闸原样不变。"""
        cap = max(0.0, self.remaining())
        timeout = cap if named.timeout is None else min(cap, named.timeout)
        if timeout <= 0:
            raise BudgetExit("time_budget")
        executor = SubprocessProposerExecutor()
        try:
            candidate = executor.run(named, evidence, timeout=timeout,
                                     cwd=self.config.repo)
        except ProposerTimeoutError as exc:
            raise BudgetExit("time_budget") from exc
        if candidate is None:
            return None
        return Candidate(files=candidate["files"],
                         explanation=candidate.get("explanation", ""),
                         cost=float(candidate.get("cost", 0.0)),
                         calls=int(candidate.get("calls", 1)))

    def _call_proposer_thread(self, proposer, evidence):
        """无 fork 平台的回调执行器：守护线程里跑 proposer，join 硬超时。

        超时抛 BudgetExit("time_budget")，结果被放弃；异常按 fork 路径
        同样包成 RuntimeError。守护线程随进程退出而结束，不阻塞主流程。"""
        import threading
        box = {}
        def invoke():
            try:
                box["value"] = proposer(evidence)
            except Exception as exc:
                box["error"] = str(exc)
        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        thread.join(timeout=max(0, self.remaining()))
        if thread.is_alive():
            raise BudgetExit("time_budget")
        if "error" in box:
            raise RuntimeError(box["error"])
        return box.get("value")

    def remaining(self):
        return self.config.max_seconds - (time.monotonic() - self.started)

    def finish(self, reason, original=None, **data):
        report = {"exit_status": reason, "original_fault": original, "ledger": self.ledger,
                  "trace_directory": str(self.trace.directory), **data}
        self.event("结论", exit_status=reason, final_artifact=data.get("artifact"), report=report)
        return redact(report)

    def execute(self, issue_id, reproduction, proposer, *, mode=DEFAULT_MODE,
                question_id=None, max_call_cost=0.0):
        """proposer 接收脱敏证据及剩余硬预算；必须为受信任可取消/计费上界回调。"""
        original = None
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", issue_id):
            return self.finish("configuration_error", error="问题id格式错误")
        if mode not in {"proposal", "apply"}:
            return self.finish("configuration_error", error="未知档位")
        self.event("诊断开始", issue_id=issue_id, mode=mode, message="先复现，不改源文件")
        if mode == "apply":
            if not question_id or not self.adapter.approve(question_id, issue_id, "code_branch"):
                return self.finish("approval_required")
            self.event("档位变更", before=DEFAULT_MODE, after=mode, approved=True, issue_id=issue_id)
        try:
            gate = Gates(self.config)
            gate.preflight()
            command_check(self.config, reproduction)
            if len(reproduction) != 2 or not reproduction[1].endswith(".py"):
                raise GateConfigError("复现必须是仓库内独立Python脚本命令")
            reproduction_source = confined(self.config.repo, reproduction[1]).read_text(encoding="utf-8")
            if self.remaining() <= 0:
                raise BudgetExit("time_budget")
            baseline = copy_repo(self.config, "baseline")
            self.event("复现开始", command=reproduction)
            original = run(self.config, baseline, reproduction, max(0.01, self.remaining()))
            original["reproduction_script"] = redact(reproduction_source)
            self.event("诊断", reproduction_command=reproduction, reproduction_script=reproduction_source, before=original)
            if original["ok"] or original["timeout"] or original["output_limit"]:
                return self.finish("observation_only", original, explanation="未稳定复现明确失败；不进入自动修复")
            feedback = original["output"]
            format_errors = 0
            attempts = 0
            rounds = 0
            while rounds < self.config.reflection_rounds and attempts < self.config.max_attempts:
                self.budget(max_call_cost)
                rounds += 1
                self.steps += 1
                self.event("提案开始", round=rounds, feedback=feedback)
                # 预留最大花费，异常/错误格式同样计费，不信任模型返回的自报成本。
                self.cost += max_call_cost
                self.calls += 1
                candidate = self.call_proposer(proposer, redact({"issue_id": issue_id, "original_fault": original,
                    "failure_output_verbatim": feedback, "round": rounds,
                    "remaining_seconds": max(0, self.remaining()),
                    "max_cost": max_call_cost}))
                if self.remaining() <= 0:
                    raise BudgetExit("time_budget")
                if candidate is None:
                    self.attempt("no_candidate", round=rounds)
                    return self.finish("no_candidate", original)
                if not isinstance(candidate, Candidate) or not isinstance(candidate.files, dict) or not candidate.files:
                    format_errors += 1
                    self.attempt("no_candidate", round=rounds, format_error=True)
                    self.event("格式错误", consecutive=format_errors)
                    if format_errors >= 3:
                        return self.finish("format_error_limit", original)
                    continue
                format_errors = 0
                attempts += 1
                sandbox = copy_repo(self.config, "candidate")
                changes = {}
                originals = {}
                try:
                    for name, source in candidate.files.items():
                        if name not in self.config.repair_paths or any(
                                name == protected or name.startswith(protected.rstrip("/") + "/")
                                for protected in self.config.protected_paths):
                            raise ValueError("文件不在显式修复白名单或属于核心记忆保护范围")
                        path = confined(sandbox, name)
                        if path.suffix != ".py" or not path.is_file() or not isinstance(source, str):
                            raise ValueError("只允许现有Python文件；禁止新增、删除、记忆与规矩变动")
                        old = path.read_text(encoding="utf-8")
                        originals[name] = path.read_bytes()
                        changes[name] = capability_diff(old, source)
                        if not changes[name]["ok"]:
                            raise ValueError("候选新增危险或未经证明安全的能力")
                    # 原始副本首次备份，供失败还原；不把候选源码或备份当模型证据外发。
                    backup_dir = sandbox.parent / (sandbox.name + "-first-backup")
                    for name, raw in originals.items():
                        backup = confined(backup_dir, name)
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, "wb") as stream:
                            stream.write(raw)
                    for name, source in candidate.files.items():
                        confined(sandbox, name).write_text(source, encoding="utf-8")
                    self.event("验证开始", attempt=attempts, capability_changes=changes)
                    validation = gate.validate(sandbox, list(candidate.files), max(0.01, self.remaining()))
                    self.event("验证", quick=validation["quick"], slow=validation["slow"], capability_changes=changes)
                    if (validation.get("slow") or {}).get("configuration_error"):
                        raise GateConfigError(validation["slow"]["configuration_error"])
                    after = run(self.config, sandbox, reproduction, max(0.01, self.remaining())) if validation["quick"]["ok"] else None
                    passed = bool(validation["quick"]["ok"] and (validation["slow"] or {}).get("ok") and after and after["ok"])
                    if not passed:
                        self.attempt("gate_rejected", attempt=attempts, quick=validation["quick"], slow=validation["slow"], after=after,
                                     capability_changes=changes)
                        feedback = ((validation.get("slow") or {}).get("output") or
                                    (after or {}).get("output") or str(validation["quick"]))
                        continue
                    self.event("候选通过", explanation=candidate.explanation, before=original, after=after,
                               capability_changes=changes, quick=validation["quick"], slow=validation["slow"])
                    if mode == "proposal":
                        self.attempt("not_applied", attempt=attempts, artifact=str(sandbox), capability_changes=changes)
                        # 通过验证的副本保留为提案，源仓库从未被修改。
                        originals = {}
                        return self.finish("proposal_ready", original, artifact=str(sandbox), after=after,
                                           quick=validation["quick"], slow=validation["slow"], capability_changes=changes)
                    artifact = self.commit_branch(issue_id, candidate, sandbox, reproduction)
                    self.attempt("applied", attempt=attempts, artifact=artifact, capability_changes=changes)
                    originals = {}
                    return self.finish("branch_ready", original, artifact=artifact, after=after,
                                       quick=validation["quick"], slow=validation["slow"], capability_changes=changes)
                except GateConfigError:
                    raise
                except (ValueError, SyntaxError) as exc:
                    self.attempt("gate_rejected", attempt=attempts, error=str(exc), capability_changes=changes)
                    feedback = str(exc)
                finally:
                    for name, raw in originals.items():
                        confined(sandbox, name).write_bytes(raw)
                    if originals:
                        self.event("失败还原", files=list(originals), scope="仅候选副本；源仓库未改")
            return self.finish("attempt_limit", original)
        except GateConfigError as exc:
            return self.finish("gate_configuration_error", original, error=str(exc))
        except BudgetExit as exc:
            return self.finish(str(exc), original)
        except Exception as exc:
            self.attempt("not_applied", error=str(exc))
            return self.finish("executor_error", original, error=str(exc))

    def git(self, root, *args):
        # 内部固定Git命令，不暴露给候选；禁钩子，不继承宿主凭据环境。
        # 移植注记（2026-11-24）：PATH 用 gates.safe_path()（Windows 的
        # defpath 里没有 git）；GIT_CONFIG_GLOBAL 与 hooksPath 在
        # Windows 上不能用 /dev/null 与 nul 设备，改指存档区内的空
        # 文件/目录，语义相同（无全局配置、无钩子）。
        from .gates import safe_path
        if os.name == "nt":
            empty_config = self.config.archive / "gitconfig-empty"
            empty_config.parent.mkdir(parents=True, exist_ok=True)
            if not empty_config.exists():
                empty_config.write_text("", encoding="utf-8")
            global_config = str(empty_config)
            hooks_path = str(self.config.archive)
        else:
            global_config = os.devnull
            hooks_path = "/dev/null"
        git_args = ["git", "-c", "core.hooksPath=" + hooks_path]
        if os.name == "nt":
            # Windows：隔离环境丢了 Git for Windows 系统配置里的
            # core.autocrlf=true 后，状态比较不再做 CRLF→LF 归一，
            # 所有文本文件都会被误判为已改动（移植实测：干净仓库
            # 报三文件 M、自修被脏改动护栏误杀）。显式补回平台默认。
            git_args += ["-c", "core.autocrlf=true"]
        result = subprocess.run([*git_args, *args], cwd=root,
            env={"PATH": safe_path(), "HOME": str(self.config.archive), "GIT_CONFIG_NOSYSTEM": "1",
                 "GIT_CONFIG_GLOBAL": global_config, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=max(0.01, self.remaining()))
        if result.returncode:
            raise RuntimeError("Git失败：" + result.stderr)
        return result.stdout.strip()

    def commit_branch(self, issue_id, candidate, sandbox, reproduction):
        status = self.git(self.config.repo, "status", "--porcelain")
        if status:
            if not self.adapter.isolate_dirty(self.config.repo, status):
                raise RuntimeError("存在他人未提交改动；需待问批准隔离，不混入AI提交")
            if self.git(self.config.repo, "status", "--porcelain"):
                raise RuntimeError("人工改动尚未隔离干净")
        clone = self.config.archive / "branches" / uuid.uuid4().hex
        clone.parent.mkdir(parents=True, exist_ok=True)
        self.git(self.config.repo, "clone", "--no-hardlinks", "--no-local", str(self.config.repo), str(clone))
        branch = "healthcheck/" + issue_id + "-" + uuid.uuid4().hex[:8]
        self.git(clone, "checkout", "-b", branch)
        for name, source in candidate.files.items():
            original = confined(self.config.repo, name).read_bytes()
            if original != confined(sandbox.parent / (sandbox.name + "-first-backup"), name).read_bytes():
                raise RuntimeError("源文件在验证期间已变化，拒绝提交")
            confined(clone, name).write_text(source, encoding="utf-8")
        # 从最终提交副本再跑全套，防止clone基线与验证副本不一致。
        final_gate = Gates(self.config).validate(clone, list(candidate.files), max(0.01, self.remaining()))
        self.event("提交前复检", validation=final_gate)
        if not final_gate["quick"]["ok"] or not (final_gate["slow"] or {}).get("ok"):
            raise RuntimeError("最终分支全套复检未通过；不提交")
        final_reproduction = run(self.config, clone, reproduction, max(0.01, self.remaining()))
        self.event("分支同脚本复检", after=final_reproduction)
        if not final_reproduction["ok"]:
            raise RuntimeError("最终分支同一复现脚本未通过；不提交")
        self.git(clone, "add", "--", *candidate.files)
        message = f"体检[代码] {issue_id} 修复已复现故障\n\n台账：候选副本两级门及同一复现已通过；独立分支等待合并。"
        self.git(clone, "-c", "user.name=healthcheck", "-c", "user.email=healthcheck@localhost",
                 "commit", "-m", message)
        commit = self.git(clone, "rev-parse", "HEAD")
        self.event("版本成果", branch=branch, commit=commit, repository=str(clone))
        return {"repository": str(clone), "branch": branch, "commit": commit}


SAFE_ACTIONS = {"rebuild_projection", "expire_lock", "restart_component", "catchup_maintenance"}


def safe_repair(config, adapter, kind, issue_id, *, mode=DEFAULT_MODE, question_id=None):
    """业务小修由十二适配器事务执行；任何删除/核心记忆行为必须另行待问批准。"""
    trace = Trace(config.archive / "safe-traces" / uuid.uuid4().hex)
    def event(step, **data):
        return trace.event(step, **data)
    if kind not in SAFE_ACTIONS or mode not in {"proposal", "apply"}:
        return event("结论", result="not_applied", reason="未知小修或档位")
    if mode == "proposal":
        return event("结论", result="not_applied", proposal=kind, issue_id=issue_id)
    if not question_id or not adapter.approve(question_id, issue_id, kind):
        return event("结论", result="not_applied", reason="待问批准缺失")
    event("档位变更", before=DEFAULT_MODE, after=mode, approved=True)
    snapshot = None
    success = False
    try:
        event("备份开始", kind=kind)
        snapshot = adapter.begin_safe(kind, issue_id)
        if snapshot is None:
            raise RuntimeError("没有可回滚备份，拒绝执行")
        for number in range(1, config.max_attempts + 1):
            event("执行开始", attempt=number, kind=kind)
            action = adapter.safe_action(kind, issue_id)
            verify = adapter.verify_action(kind, issue_id)
            success = action.get("ok") is True and verify.get("ok") is True
            event("尝试台账", result="applied" if success else "still_failed", action=action, verify=verify)
            if success:
                return event("结论", result="applied")
        return event("结论", result="still_failed", reason="尝试上限，恢复原始故障")
    except Exception as exc:
        return event("结论", result="still_failed", error=str(exc))
    finally:
        if snapshot is not None and not success:
            event("还原开始")
            restored = adapter.rollback_safe(snapshot)
            event("还原结果", restored=restored)
            if restored is not True:
                raise RuntimeError("严重异常：适配器回滚失败，必须人工处理")
