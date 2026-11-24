"""仓库内白名单命令、两级验证、候选 AST 能力围栏。

并入十二时的移植改动（2026-11-24，十一）：
- 慢门除 unittest discover 外，接受 pytest 全套入口（见 preflight 的
  精确白名单），并从 pytest 汇总行解析测试计数——HANDOFF 要求的
  「有测试计数的可信全套入口」，零计数仍判配置错误。
- 子进程环境在原有净化基础上放行少数非密钥变量（HF 离线开关、
  Windows 系统目录/临时目录），PATH 改为只含解释器与 git 所在
  目录 + 系统默认路径，Windows 下也能找到 git/python。
- 超时/超量杀进程：POSIX 用进程组 SIGKILL，Windows 用 taskkill /T。
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from .core import confined, redact


class GateConfigError(ValueError):
    pass


def safe_path():
    """子进程可用的 PATH：当前解释器目录、git 目录、系统默认路径。

    不继承宿主完整 PATH（避免把宿主工具链和潜在注入点带进门内），
    但保证 Windows 下 defpath 里没有的 git/python 也能定位。"""
    parts = []
    exe_dir = os.path.dirname(sys.executable)
    if exe_dir:
        parts.append(exe_dir)
    git = shutil.which("git")
    if git:
        parts.append(os.path.dirname(git))
    parts.append(os.defpath)
    return os.pathsep.join(part for part in parts if part)


# 允许透传给子进程的非密钥环境变量：HF 离线/镜像开关是十二全套
# 测试加载本地向量模型的前提；SYSTEMROOT/TEMP/TMP 是 Windows 子
# 进程（git/python）正常工作所需。密钥类变量一律不透传。
ENV_PASSTHROUGH = ("HF_HUB_OFFLINE", "HF_ENDPOINT", "SYSTEMROOT", "TEMP", "TMP")


def command_check(config, command):
    if not command or command not in config.allowed_commands:
        raise GateConfigError("命令不在精确白名单内")
    if any(not isinstance(x, str) or "\x00" in x for x in command):
        raise GateConfigError("命令参数格式错误")
    if any(x in {"-c", "-e", "--eval"} for x in command):
        raise GateConfigError("禁止内联解释器命令")
    if Path(command[0]).name.lower() in {"sh", "bash", "zsh", "cmd", "powershell"}:
        raise GateConfigError("禁止shell")
    if any(Path(x).is_absolute() or ".." in Path(x).parts for x in command[1:]):
        raise GateConfigError("命令参数不可越出仓库")


def run(config, root, command, timeout=None):
    command_check(config, command)
    if not root.resolve().is_dir():
        raise GateConfigError("命令工作目录不存在")
    # 不继承宿主密钥和 Python 注入环境；白名单由部署方审核，不由候选控制。
    env = {"PATH": safe_path(), "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ENV_PASSTHROUGH:
        if name in os.environ:
            env[name] = os.environ[name]
    chunks = bytearray()
    exceeded = threading.Event()
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)

    def kill():
        if os.name == "nt":
            # Windows 没有进程组信号：taskkill /T 连子进程一并结束。
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def collect():
        while True:
            block = proc.stdout.read(4096)
            if not block:
                break
            remaining = config.max_output_bytes - len(chunks)
            chunks.extend(block[:max(0, remaining)])
            if len(block) > remaining:
                exceeded.set()
                kill()
                break
    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout if timeout is not None else config.max_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill()
        proc.wait()
    reader.join(timeout=2)
    if reader.is_alive():
        kill()
        reader.join(timeout=2)
    proc.stdout.close()
    return redact({"command": command, "returncode": proc.returncode,
                   "output": chunks.decode("utf-8", errors="replace"),
                   "timeout": timed_out, "output_limit": exceeded.is_set(),
                   "ok": proc.returncode == 0 and not timed_out and not exceeded.is_set(),
                   "elapsed_seconds": time.monotonic() - started})


DANGEROUS = {"subprocess", "socket", "pickle", "marshal", "ctypes", "multiprocessing",
             "importlib", "builtins", "__builtins__", "eval", "exec", "compile", "__import__",
             "getattr", "setattr", "globals", "locals", "vars", "open", "os", "shutil",
             "pathlib", "requests", "urllib", "http", "aiohttp", "sys"}


def capabilities(source):
    tree = ast.parse(source)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom):
                names += [node.module or ""]
            for name in names:
                if name.split(".")[0] in DANGEROUS:
                    found.add("import:" + name)
        if isinstance(node, ast.Name) and node.id in DANGEROUS:
            found.add("name:" + node.id)
        if isinstance(node, ast.Attribute):
            # 对所有属性名保守记录，防止已有危险别名新增方法。
            found.add("attribute:" + node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            found.add("call:" + node.func.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in DANGEROUS or node.value.startswith("__"):
                found.add("dynamic:" + node.value)
    return found


def capability_diff(original, candidate):
    before, after = capabilities(original), capabilities(candidate)
    return {"added": sorted(after - before), "removed": sorted(before - after),
            "ok": not bool(after - before), "note": "保守静态围栏，不是恶意代码沙箱"}


def copy_repo(config, label):
    destination = config.archive / "sandboxes" / (label + "-" + uuid.uuid4().hex)
    # 不复制 Git、密钥常见文件及符号链接；副本保留供追溯，不删除。
    # 移植补（2026-11-24）：.pytest_cache 一并跳过——它是测试运行
    # 的临时缓存（门里会重新全量跑，不依赖它），而 Windows 上它
    # 可能被其他会话的进程锁住/带特殊 ACL，实测整仓复制被它一个
    # 目录堵死（WinError 5），代码项直接报 adapter_error。
    def ignore(directory, names):
        return [name for name in names if name == ".git" or name == ".env" or
                name == ".pytest_cache" or
                name.endswith((".pem", ".key")) or (Path(directory) / name).is_symlink()]
    shutil.copytree(config.repo, destination, ignore=ignore)
    return destination


class Gates:
    def __init__(self, config):
        self.config = config

    def preflight(self):
        command_check(self.config, self.config.test_command)
        # 明确要求发现式全套入口，拒绝单文件与筛选；自定义runner需十一封装发现入口。
        command = self.config.test_command
        unittest_form = (len(command) >= 4 and command[1:4] == ["-m", "unittest", "discover"])
        # 十二仓库用 pytest：只接受 -m pytest + 无筛选的安全参数集，
        # 任何路径参数、-k、-p、--deselect 等一律拒绝（不许缩小范围）。
        pytest_args = {"-q", "-s", "--no-header", "--disable-warnings", "-ra",
                       "--tb=short", "--tb=long", "--tb=line", "--tb=no"}
        pytest_form = (len(command) >= 3 and command[1:3] == ["-m", "pytest"]
                       and all(arg in pytest_args for arg in command[3:]))
        if not self.config.full_suite or not (unittest_form or pytest_form):
            raise GateConfigError("慢门必须声明全套，并使用 python -m unittest discover 或 python -m pytest 全套入口，禁止单文件验证")
        if any(x in {"-k", "-p", "--pattern"} or x.startswith("--pattern=") for x in command[3:]):
            raise GateConfigError("全套慢门禁止过滤测试")
        if shutil.which(command[0], path=safe_path()) is None:
            raise GateConfigError("缺少测试解释器，等待BOOS批准安装")

    def validate(self, root, files, timeout=None):
        self.preflight()
        quick = {"ok": True, "files": []}
        for name in files:
            path = confined(root, name)
            if path.suffix != ".py":
                return {"quick": {"ok": False, "error": "候选仅接受Python文件"}, "slow": None}
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=name)
                quick["files"].append(name)
            except (SyntaxError, UnicodeError) as exc:
                quick = {"ok": False, "error": str(exc), "files": quick["files"]}
                return redact({"quick": quick, "slow": None})
        slow = run(self.config, root, self.config.test_command, timeout)
        # 零测试不能当全套通过（unittest 与 pytest 两种汇总口径）。
        total, runner = suite_counts(slow["output"])
        slow["test_count"] = total
        slow["suite_runner"] = runner
        if total == 0:
            slow["ok"] = False
            slow["configuration_error"] = "测试发现数为零或无法确认全套执行数量"
        return redact({"quick": quick, "slow": slow})


def suite_counts(output):
    """从全套输出里解析实际执行的测试数；认不出返回 (0, None)。

    unittest：取最后一个 "Ran N tests"；pytest：汇总行里
    passed/failed/error/skipped/deselected/xfailed/xpassed 计数求和
    （"no tests ran" 天然求和为 0）。"""
    counts = re.findall(r"Ran (\d+) tests?", output or "")
    if counts:
        return int(counts[-1]), "unittest"
    # pytest 只认最后一行汇总（逐行找带 passed/failed 的行取最后一行），
    # 防止测试自身输出里的同类字样把计数算重。
    total = None
    for line in (output or "").splitlines():
        if " passed" not in line and " failed" not in line and " error" not in line:
            continue
        line_total = sum(int(num) for num in re.findall(
            r"(\d+) (?:passed|failed|errors?|skipped|deselected|xfailed|xpassed)", line))
        if line_total or "no tests ran" in line:
            total = line_total
    if total is not None:
        return total, "pytest"
    return 0, None
