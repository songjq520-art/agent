# healthcheck_integration/adapter.py（十二可信适配器：体检模块的实连接口）
#
# 把 healthcheck 模块 Adapter 协议的每个方法接到十二的真实系统上：
#   memory()    —— store.run_self_check（十二原有十项自检，抽样 20，
#                  只做结果映射，不替换、不改动原算法；注意它按生产
#                  入口语义先跑一遍衰减扫，见方法内注记）
#   services()  —— 进程（PowerShell CIM 查 web.server 进程）+ HTTP
#                  只读探活 + logs/agent.log 近期异常行扫描
#   models()    —— 判官（config.SEMANTIC_JUDGE）与聊天（默认模型）
#                  各一次极小真实调用（max_tokens=1）；当月花费读
#                  logs/budget_usage.json（BudgetTracker 的账本），
#                  上限 config.MONTHLY_BUDGET_USD。外加两件（2026-
#                  11-28 并入本项证据）：记账自对账——accounting
#                  持久账本当月总额 vs BudgetTracker 当月总额跑
#                  reconcile，对不上本项判异常；DeepSeek 平台余额
#                  （U9）——只读官方余额接口，查不到标未知、不影响
#                  本项结论。密钥只经 config.get_api_key 交给
#                  Provider/请求头，不落日志、不回传。
#   resources() —— Windows ctypes 读系统 CPU/内存；库与账本体积读
#                  文件大小（磁盘剩余由模块自己实测）
#   backups()   —— 只读 memory/drive_backup 的状态：最近成功时间 +
#                  本地 zip 的 SHA256 与状态记录对账。**只读**：不调
#                  run_backup、不碰 backup_enabled 开关（生产开关现
#                  为关，体检只报告状态，永不翻开关）
#   schedule()  —— maintenance_state.json + 待问问题积压数
#   approve()   —— 真防重放：批准必须是待问问题库里一条 kind=
#                  healthcheck_approval 的问题，extra 绑定 issue_id
#                  与 action，BOOS 已回答「批准」，且从未被消费过；
#                  消费用 O_EXCL 独占建文件落盘，一次性、跨进程有效
#   isolate_dirty() —— git stash push 把人的改动单独持久化一笔
#   begin_safe / safe_action / verify_action / rollback_safe
#               —— 四类小修的真事务（作用域只限本适配器管辖的沙盒
#                  资源：记忆库 SQLite 投影、体检租约锁、受管子进程
#                  组件、维护状态；绝不碰真服务进程与核心记忆）
#   weekly_lease() —— 记忆目录下带过期时间的独占锁文件
#
# 全部方法遵守模块契约：返回 dict、ok 严格 True 才算好、返回值不带
# 密钥原值；接不通/异常时如实报 ok=False 并给原因，不冒充健康。

import base64
import ctypes
import json
import os
import sqlite3
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 体检批准问题的种类与选项：批准走十二待问问题库的正常通道——
# 体检要 apply 前先用 request_approval() 攒一条问题，BOOS 在待问
# 列表里回答「批准」后，approve() 才可能消费成功。
APPROVAL_KIND = "healthcheck_approval"
APPROVAL_OPTIONS = ("批准", "不批准")
APPROVAL_ANSWER_YES = "批准"

# 待问积压超过这个数，调度项判异常（提醒 BOOS 攒太多没答）。
PENDING_QUESTIONS_MAX = 20
# 租约默认有效期：要盖住一次完整体检（含全套测试）的大致时长。
LEASE_SECONDS = 3600


def _iso_from_epoch(value):
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    return None


class SubprocessComponent:
    """受管子进程组件（restart_component 的作用对象）。

    十二本体是单进程服务，没有可独立重启的内部组件注册表；体检的
    「重启卡死组件」在集成层落成这个显式受管组件：调用方把允许被
    体检重启的子进程（例如沙盒服务、辅助进程）登记进来，重启 =
    结束进程树并按原命令重新拉起。真服务主进程永不登记进来。"""

    def __init__(self, argv, cwd=None):
        self.argv = list(argv)
        self.cwd = cwd
        self._proc = None

    def start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(self.argv, cwd=self.cwd)

    def stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                    capture_output=True, timeout=15)
            else:
                self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)

    def restart(self):
        self.stop()
        self.start()

    def is_alive(self):
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self):
        return self._proc.pid if self._proc is not None else None


class ShierHealthcheckAdapter:
    """healthcheck 模块的十二可信适配器（契约见 healthcheck/HANDOFF.md）。"""

    def __init__(self, repo_root, memory_dir=None, store=None,
                 server_base_url=None, component=None, budget_tracker=None):
        self.repo_root = Path(repo_root)
        self.memory_dir = Path(memory_dir) if memory_dir else self.repo_root / "memory"
        self._store = store
        self.server_base_url = server_base_url
        self.component = component
        self._budget_tracker = budget_tracker

    # ---------- 内部工具 ----------

    def store(self):
        """拿记忆库：外部注入优先（测试/服务内共用），否则按工厂新建。

        工厂建的是本仓库 memory 目录的库；沙盒里就是沙盒库。"""
        if self._store is None:
            from memory.sqlite_store import SQLiteStore

            self._store = SQLiteStore(memory_dir=str(self.memory_dir))
        return self._store

    def _locks_dir(self):
        return self.memory_dir / "healthcheck_locks"

    def _consumed_dir(self):
        return self.memory_dir / "healthcheck_consumed"

    def _snapshots_dir(self):
        return self.memory_dir / "healthcheck_snapshots"

    # ---------- 1. 记忆 ----------

    def memory(self, sample_size=20):
        """调十二原有自检并映射到体检契约（不替换原算法）。

        走 store.run_self_check 生产入口：它会先跑一遍衰减扫（该沉
        底/标待删的记账本——这是十二每周维护本来就在做的事，体检
        只是搭同一班车），再跑十项自检；第 9/10 项的模型抽查走整理
        路由的判官。映射规则：
          contradictions ← 第 3 项（冲突未决）findings
          sources        ← 第 4 项（孤儿/出处）findings
          duplicates     ← 第 1 项（重复记忆）findings
          buried         ← 第 2 项（衰减）+ 第 8 项（归档）findings
          core_integrity ← 第 6 项（索引一致）通过
          sample_count   ← 第 9 项实际抽到并给出结论条目的条数
          ok             ← 1-8 项全 pass（第 1 项无向量时 skipped 算
                           通过口径，属环境限制）且第 9 项 pass、
                           第 10 项非 warn；任一 fail 直接异常。
        """
        store = self.store()
        report = store.run_self_check(sample_size=sample_size)
        items = report.get("items") or {}

        def status(number):
            return (items.get(number) or {}).get("status")

        def findings(number):
            return list((items.get(number) or {}).get("findings") or [])

        sample_count = sum(
            1 for entry in findings(9)
            if isinstance(entry, dict) and entry.get("id"))
        statuses = {str(n): status(n) for n in range(1, 11)}
        has_fail = any(status(n) == "fail" for n in range(1, 11))
        has_warn = any(status(n) == "warn" for n in range(1, 11))
        ok = (
            all(status(n) == "pass" for n in (2, 3, 4, 6, 7, 8))
            and status(1) in ("pass", "skipped")
            and status(9) == "pass"
            and status(10) in ("pass", "skipped")
        )
        return {
            "ok": ok,
            "severity": "critical" if has_fail else ("warning" if has_warn else "info"),
            "sample_count": sample_count,
            "contradictions": findings(3)[:20],
            "sources": findings(4)[:20],
            "duplicates": findings(1)[:20],
            "buried": (findings(2) + findings(8))[:20],
            "core_integrity": status(6) == "pass",
            "self_check_statuses": statuses,
            "self_check_summary": report.get("summary"),
        }

    # ---------- 2. 服务状态 ----------

    def _server_processes(self):
        """查在跑的 web.server 进程（Windows：PowerShell CIM）。

        只认命令行里带 web.server 的 python 进程；查不到/查不动都
        如实返回空列表 + 原因，不猜。"""
        if os.name != "nt":
            return [], "仅 Windows 做了进程枚举"
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                 "| Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
            data = json.loads(proc.stdout or "[]")
        except Exception as exc:
            return [], "进程枚举失败：%s" % exc
        if isinstance(data, dict):
            data = [data]
        found = []
        for entry in data or []:
            cmdline = (entry or {}).get("CommandLine") or ""
            if "web.server" in cmdline:
                found.append({"pid": entry.get("ProcessId"),
                              "command_line": cmdline[:200]})
        return found, None

    def _probe_endpoint(self):
        if not self.server_base_url:
            return {"url": None, "ok": False, "reason": "未配置服务地址"}
        url = self.server_base_url.rstrip("/") + "/"
        started = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                resp.read(64)
                return {"url": url, "ok": resp.status == 200,
                        "status": resp.status,
                        "elapsed_ms": round((time.monotonic() - started) * 1000)}
        except Exception as exc:
            return {"url": url, "ok": False, "error": str(exc)[:200],
                    "elapsed_ms": round((time.monotonic() - started) * 1000)}

    def _recent_log_lines(self, limit_bytes=262144, max_lines=10):
        path = self.repo_root / "logs" / "agent.log"
        if not path.exists():
            return [], "无日志文件"
        try:
            size = path.stat().st_size
            with open(path, "rb") as fh:
                if size > limit_bytes:
                    fh.seek(size - limit_bytes)
                text = fh.read().decode("utf-8", errors="replace")
        except OSError as exc:
            return [], "日志读不了：%s" % exc
        hits = [line.strip()[:200] for line in text.splitlines()
                if ("Traceback" in line or "ERROR" in line or "CRITICAL" in line)]
        return hits[-max_lines:], None

    def services(self):
        processes, proc_error = self._server_processes()
        endpoint = self._probe_endpoint()
        log_lines, log_error = self._recent_log_lines()
        endpoint_ok = endpoint.get("ok") is True
        ok = bool(processes) and endpoint_ok
        return {
            "ok": ok,
            "severity": "info" if ok else "critical",
            "processes": processes,
            "process_error": proc_error,
            "endpoints": [endpoint],
            "recent_logs": log_lines,
            "recent_logs_error": log_error,
        }

    # ---------- 3. 模型接口 ----------

    def _accounting_check(self, month, counterpart_total_usd):
        """当月记账自对账（只读）：持久账本 vs BudgetTracker 当月总额。

        对方口径：DeepSeek 没有逐笔用量接口，平台账单拉不到自动版；
        BudgetTracker 是服务内所有模型调用的记账汇合点，它的当月
        总额本该与 accounting 持久账本一分不差——哪边漏记、哪条路
        绕过记账，差额就会冒出来（阈值与 accounting.reconcile 的
        既定口径一致：max(2%, $0.07)，有成本未知的笔直接判对不上）。
        与真平台账单的人工对账仍走 accounting.py 的 reconcile 命令行。
        账本坏行数（写一半崩溃的半行）一并报出，不静默吞。本方法
        任何异常都收进结果返回，绝不让体检整体崩。
        """
        try:
            import accounting
            verdict = accounting.reconcile(month, counterpart_total_usd)
            summary = accounting.monthly_summary(month)
            return {
                "month": month,
                "calls": summary["calls"],
                "ledger_total_usd": verdict["ledger_total_usd"],
                "counterpart": "BudgetTracker 当月账本总额（budget_usage.json）",
                "counterpart_total_usd": verdict["platform_total_usd"],
                "diff_usd": verdict["diff_usd"],
                "unknown_cost_calls": verdict["unknown_cost_calls"],
                "corrupt_lines": verdict["corrupt_lines"],
                "ok": verdict["ok"],
            }
        except Exception as exc:
            return {"month": month, "ok": False,
                    "error": f"记账对账执行失败：{str(exc)[:200]}"}

    def _deepseek_balance(self, api_key):
        """查 DeepSeek 平台余额（U9）：只读官方接口，不产生费用。

        返回 (balances, is_available, error)：balances 是 balance_infos
        净化后的列表（只留币种与三类余额），查不到时 balances=None、
        error 给原因。Key 只进 Authorization 请求头，不落日志、
        不进返回值。
        """
        request = urllib.request.Request(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {api_key}",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return None, None, f"余额接口不通：{str(exc)[:200]}"
        infos = payload.get("balance_infos") if isinstance(payload, dict) else None
        if not isinstance(infos, list):
            return None, None, "余额接口返回结构异常（无 balance_infos）"
        balances = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            balances.append({
                "currency": info.get("currency"),
                "total_balance": info.get("total_balance"),
                "granted_balance": info.get("granted_balance"),
                "topped_up_balance": info.get("topped_up_balance"),
            })
        return balances, payload.get("is_available"), None

    def models(self):
        """判官/聊天各一次极小真实探活 + 当月真实花费。

        花费口径：logs/budget_usage.json 是 BudgetTracker 的账本
        （judge/chat/tool 所有模型调用都记在里面），月份对不上按
        0 算（新月还没花）。上限是 config.MONTHLY_BUDGET_USD。
        整理路由自己的当月花费（上限 3 美元）附在证据里一起报。
        另附记账自对账（accounting 持久账本 vs BudgetTracker 当月
        总额，对不上本项判异常）与 DeepSeek 平台余额（查不到标
        未知、不影响结论），细口径见 _accounting_check /
        _deepseek_balance。"""
        import config as repo_config

        result = {"judge": False, "chat": False}
        # 当月花费（只读账本文件，不经 BudgetTracker 写路径）
        spent = 0.0
        budget_path = self.repo_root / "logs" / "budget_usage.json"
        try:
            data = json.loads(budget_path.read_text(encoding="utf-8"))
            if data.get("month") == datetime.now().strftime("%Y-%m"):
                spent = float(data.get("total_cost_usd") or 0.0)
        except Exception:
            spent = 0.0
        result["month_spent"] = round(spent, 6)
        result["month_limit"] = float(repo_config.MONTHLY_BUDGET_USD)
        router_state = self.memory_dir / "organizing_router_state.json"
        try:
            rstate = json.loads(router_state.read_text(encoding="utf-8"))
            result["organizing_spent_usd"] = rstate.get("spent_usd")
            result["organizing_cap_usd"] = 3.0
        except Exception:
            pass

        # 记账自对账（当月）：在探活写账之前算，读到的就是探活前
        # 的账面；对方总额用上面刚读的 BudgetTracker 当月花费。
        month = datetime.now().strftime("%Y-%m")
        result["accounting"] = self._accounting_check(
            month, result["month_spent"])
        result["accounting_ok"] = result["accounting"].get("ok") is True

        api_key = repo_config.get_api_key("deepseek")
        if not api_key:
            result["reason"] = "未配置 DEEPSEEK_API_KEY（只查在不在，未读值）"
            result["deepseek_balance"] = {
                "status": "跳过", "reason": "未配置 DEEPSEEK_API_KEY"}
            result["ok"] = False
            return result

        # 平台余额（U9）：查不到标未知并说明原因，不影响本项结论，
        # 更不许假装查到。
        balances, is_available, balance_error = self._deepseek_balance(api_key)
        if balance_error:
            result["deepseek_balance"] = {
                "status": "未知", "reason": balance_error}
        else:
            result["deepseek_balance"] = {
                "status": "已查", "is_available": is_available,
                "balances": balances}
        from providers.deepseek import DeepSeekProvider

        judge_cfg = repo_config.SEMANTIC_JUDGE or {}
        targets = []
        if judge_cfg.get("provider", "deepseek") == "deepseek":
            targets.append(("judge", judge_cfg.get("model") or "deepseek-flash"))
        else:
            result["judge_error"] = "判官 provider 不是 deepseek，未接探活"
        targets.append(("chat", repo_config.MODEL_CONFIG["deepseek"]["model"]))
        for label, model in targets:
            try:
                provider = DeepSeekProvider(
                    api_key=api_key,
                    budget_tracker=self._budget_tracker,
                    model=model)
                out = provider.chat_messages(
                    [{"role": "user", "content": "ping"}],
                    max_tokens=1, ledger="chat", thinking="disabled",
                    purpose="探针", caller="healthcheck")
                result[label] = bool(
                    out and (out.get("text") is not None or out.get("usage")))
                result[label + "_model"] = model
                result[label + "_cost_usd"] = (out or {}).get("cost_usd")
            except Exception as exc:
                result[label + "_error"] = str(exc)[:200]
        result["ok"] = bool(
            result["judge"] and result["chat"]
            and 0 <= result["month_spent"] < result["month_limit"]
            and result["accounting_ok"])
        return result

    # ---------- 4. 身体资源 ----------

    def _cpu_percent(self, sample_seconds=0.25):
        if os.name == "nt":
            class FILETIME(ctypes.Structure):
                _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]

                def value(self):
                    return (self.hi << 32) | self.lo

            def read_times():
                idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
                ok = ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
                if not ok:
                    raise OSError("GetSystemTimes 失败")
                return idle.value(), kernel.value() + user.value()

            idle1, total1 = read_times()
            time.sleep(sample_seconds)
            idle2, total2 = read_times()
            total = total2 - total1
            if total <= 0:
                return 0.0
            return round(100.0 * (1 - (idle2 - idle1) / total), 1)
        load = os.getloadavg()[0]
        return round(min(100.0, 100.0 * load / (os.cpu_count() or 1)), 1)

    def _memory_percent(self):
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                raise OSError("GlobalMemoryStatusEx 失败")
            return float(stat.dwMemoryLoad)
        return 0.0

    def resources(self):
        cpu = self._cpu_percent()
        memory_percent = self._memory_percent()
        db_path = self.memory_dir / "memories.sqlite3"
        database_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(db_path) + suffix)
            if candidate.exists():
                database_bytes += candidate.stat().st_size
        ledger_path = self.memory_dir / "memory_events.jsonl"
        ledger_bytes = ledger_path.stat().st_size if ledger_path.exists() else 0
        return {
            "ok": cpu < 98.0 and memory_percent < 95.0,
            "cpu_percent": cpu,
            "memory_percent": memory_percent,
            "database_bytes": database_bytes,
            "ledger_bytes": ledger_bytes,
        }

    # ---------- 5. 备份（只读） ----------

    def backups(self):
        """只读备份状态：最近一次成功 + 本地包摘要对账。

        完整性 = 状态里记的 sha256 与本地 zip 实算一致；新鲜度 =
        最近成功在 48 小时内。生产开关（backup_enabled）只读着报，
        本方法永不调用 run_backup / set_backup_enabled。"""
        from memory import drive_backup

        state = drive_backup.load_backup_state(self.memory_dir)
        enabled = drive_backup.is_backup_enabled(self.memory_dir)
        last_success = state.get("last_success")
        recorded_sha = state.get("sha256")
        integrity = False
        detail = ""
        zip_name = state.get("last_zip")
        if last_success and recorded_sha and zip_name:
            zip_path = self.memory_dir / "backups" / zip_name
            if zip_path.exists():
                integrity = drive_backup.sha256_file(zip_path) == recorded_sha
                detail = "本地包与状态记录的 SHA256 对账" + ("一致" if integrity else "不一致")
            else:
                detail = "状态有记录，但本地备份包不在"
        else:
            detail = "尚无成功备份记录"
        fresh = False
        if state.get("last_success_epoch"):
            fresh = (time.time() - float(state["last_success_epoch"])) < 48 * 3600
        return {
            "ok": bool(integrity and fresh),
            "last_backup": last_success,
            "integrity": integrity,
            "fresh_within_48h": fresh,
            "backup_enabled": enabled,
            "detail": detail,
        }

    # ---------- 6. 调度 ----------

    def schedule(self):
        state = {}
        state_path = self.memory_dir / "maintenance_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
        now = time.time()
        specs = [
            ("decay", "decay_last_run", 24 * 3600),
            ("self_check", "self_check_last_run", 7 * 24 * 3600),
            ("backup", "backup_last_success", None),
        ]
        maintenance = []
        self_check_overdue = False
        for task, key, interval in specs:
            ts = state.get(key)
            entry = {"task": task, "last_run": _iso_from_epoch(ts)}
            if interval and isinstance(ts, (int, float)):
                entry["overdue"] = (now - float(ts)) > interval + 24 * 3600
                if task == "self_check":
                    self_check_overdue = entry["overdue"]
            maintenance.append(entry)
        if state.get("backup_skip_reason"):
            maintenance.append({"task": "backup_skip_reason",
                                "last_run": None,
                                "reason": state["backup_skip_reason"]})
        pending = self.store().pending_questions.count_unanswered()
        return {
            "ok": pending <= PENDING_QUESTIONS_MAX and not self_check_overdue,
            "maintenance": maintenance,
            "pending_questions": pending,
        }

    # ---------- 批准（防重放） ----------

    def request_approval(self, issue_id, action, text, reason=""):
        """攒一条体检批准问题进十二的待问问题库（BOOS 在正常通道答）。

        extra 里绑定 issue_id 与 action：approve() 时逐项核对，问题
        本身答错对象、答错动作都消费不了。返回问题 dict。"""
        questions = self.store().pending_questions
        return questions.add(
            APPROVAL_KIND,
            text,
            list(APPROVAL_OPTIONS),
            reason or "体检模块申请执行需要批准的动作",
            extra={"issue_id": issue_id, "action": action},
        )

    def approve(self, question_id, issue_id, action):
        """一次性批准校验 + 消费（真防重放）。

        全部满足才算数：问题存在、种类是体检批准、extra 绑定的
        issue_id/action 与本次申请逐字一致、状态是已回答且答案是
        「批准」、并且从未被消费过。消费 = 在 healthcheck_consumed/
        里以问题 id 独占建文件（O_EXCL，跨进程原子），建成功才返
        回 True；任何人第二次拿同一问题来，一律 False。"""
        question = self.store().pending_questions.get(question_id)
        if not question:
            return False
        if question.get("kind") != APPROVAL_KIND:
            return False
        if question.get("issue_id") != issue_id or question.get("action") != action:
            return False
        if question.get("status") != "answered":
            return False
        if question.get("answer") != APPROVAL_ANSWER_YES:
            return False
        consumed_dir = self._consumed_dir()
        consumed_dir.mkdir(parents=True, exist_ok=True)
        marker = consumed_dir / (question_id + ".json")
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"question_id": question_id, "issue_id": issue_id,
                       "action": action,
                       "consumed_at": datetime.now(timezone.utc).isoformat()},
                      fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        return True

    # ---------- 脏改动隔离 ----------

    def isolate_dirty(self, repo, status):
        """把人的未提交改动单独 stash 成一笔，工作区干净了才 True。

        不删除、不混进 AI 提交：改动完整留在 stash 里，随时能找回。"""
        if not (status or "").strip():
            return True
        repo = Path(repo)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # stash 会产生提交，需要身份；用体检自己的身份，账面上一眼
        # 能认出这是机器隔离动作，不是人的提交。
        proc = subprocess.run(
            ["git", "-C", str(repo),
             "-c", "user.name=shier-healthcheck",
             "-c", "user.email=healthcheck@localhost",
             "stash", "push", "-u",
             "-m", "healthcheck-isolate-" + stamp],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if proc.returncode != 0:
            return False
        after = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return after.returncode == 0 and not after.stdout.strip()

    # ---------- 每周租约 ----------

    def weekly_lease(self, key):
        """带过期的独占锁文件租约：抢到 True，被占/没过期 False。

        锁里记 pid 与过期时间；过期锁可被接管（先删再独占建，竞争
        失败方自然拿 False）。有效期 LEASE_SECONDS，盖得住一次完整
        体检；模块只在开跑前抢一次，跑动中不再复查。"""
        locks = self._locks_dir()
        locks.mkdir(parents=True, exist_ok=True)
        path = locks / (key + ".lock")
        now = time.time()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if float(data.get("expires_at") or 0) > now:
                    return False
            except Exception:
                pass  # 坏锁按过期处理，允许接管
            try:
                path.unlink()
            except OSError:
                return False
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "pid": os.getpid(),
                       "acquired_at": now, "expires_at": now + LEASE_SECONDS},
                      fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        return True

    # ---------- 小修事务（四类） ----------

    def begin_safe(self, kind, issue_id):
        """动手前备份全部受影响资源，返回不含凭据的快照句柄。

        快照放 memory/healthcheck_snapshots/ 下，按 kind 分工：
        rebuild_projection 备 SQLite 库（sqlite3 backup API，WAL 安全）；
        expire_lock 备租约锁目录内容；restart_component 只记组件当时
        状态（进程没法快照，靠 rollback 保证它最后是活着的）；
        catchup_maintenance 备维护状态文件原文。备不下来返回 None，
        模块按契约拒绝执行。"""
        try:
            snap_dir = self._snapshots_dir() / (
                "%s-%s-%s" % (kind, issue_id, uuid.uuid4().hex[:8]))
            snap_dir.mkdir(parents=True, exist_ok=False)
            snapshot = {"kind": kind, "issue_id": issue_id, "dir": str(snap_dir)}
            if kind == "rebuild_projection":
                db_path = self.memory_dir / "memories.sqlite3"
                target = snap_dir / "memories.sqlite3"
                src = sqlite3.connect(str(db_path))
                try:
                    dst = sqlite3.connect(str(target))
                    try:
                        src.backup(dst)
                    finally:
                        dst.close()
                finally:
                    src.close()
                snapshot["db_backup"] = str(target)
            elif kind == "expire_lock":
                locks = {}
                locks_dir = self._locks_dir()
                if locks_dir.exists():
                    for entry in locks_dir.iterdir():
                        if entry.is_file():
                            locks[entry.name] = base64.b64encode(
                                entry.read_bytes()).decode("ascii")
                snapshot["locks"] = locks
            elif kind == "restart_component":
                snapshot["was_alive"] = bool(
                    self.component and self.component.is_alive())
            elif kind == "catchup_maintenance":
                state_path = self.memory_dir / "maintenance_state.json"
                snapshot["state_text"] = (
                    state_path.read_text(encoding="utf-8")
                    if state_path.exists() else None)
            else:
                return None
            return snapshot
        except Exception:
            return None

    def safe_action(self, kind, issue_id):
        """四类小修的实际动作（作用域见模块头，绝不越界）。"""
        if kind == "rebuild_projection":
            # 按账本重建/补齐 SQLite 投影：十二自己的对账入口。
            self.store().reconcile_ledger_index()
            return {"ok": True, "action": "reconcile_ledger_index"}
        if kind == "expire_lock":
            removed, kept = [], []
            locks_dir = self._locks_dir()
            now = time.time()
            if locks_dir.exists():
                for entry in sorted(locks_dir.iterdir()):
                    if not entry.is_file():
                        continue
                    try:
                        data = json.loads(entry.read_text(encoding="utf-8"))
                        expired = float(data.get("expires_at") or 0) <= now
                    except Exception:
                        expired = False  # 读不懂的锁不动，留给 verify 报
                    if expired:
                        entry.unlink()
                        removed.append(entry.name)
                    else:
                        kept.append(entry.name)
            return {"ok": True, "removed": removed, "kept": kept}
        if kind == "restart_component":
            if self.component is None:
                return {"ok": False, "reason": "未登记受管组件"}
            self.component.restart()
            return {"ok": self.component.is_alive(),
                    "pid": self.component.pid}
        if kind == "catchup_maintenance":
            from web.server import MemoryMaintenanceScheduler

            state_path = self.memory_dir / "maintenance_state.json"
            # backup_runner 不注入：体检补跑只补衰减/自检，备份走它
            # 自己的开关与调度（开关关着时这里更不会碰它）。
            scheduler = MemoryMaintenanceScheduler(
                self.store(), state_path, backup_runner=None)
            ran = scheduler.run_due()
            return {"ok": True, "ran": ran}
        return {"ok": False, "reason": "未知小修种类"}

    def verify_action(self, kind, issue_id):
        """独立复检：修完必须再对一次账才算数。"""
        if kind == "rebuild_projection":
            store = self.store()
            live_ids = {
                row[0] for row in store._conn.execute(
                    "SELECT id FROM memories WHERE voided=0 "
                    "AND deleted=0 AND superseded=0").fetchall()}
            from memory import ledger as _ledger

            key_states = (store.ledger_state() or {}).get("key_states") or {}
            ledger_ids = {
                k for k, v in key_states.items()
                if v in (_ledger.LEDGER_STATE_CURRENT,
                         _ledger.LEDGER_STATE_DORMANT)}
            mismatched = sorted(live_ids ^ ledger_ids)
            return {"ok": not mismatched, "mismatched_ids": mismatched[:20],
                    "live": len(live_ids), "ledger": len(ledger_ids)}
        if kind == "expire_lock":
            remaining = []
            locks_dir = self._locks_dir()
            now = time.time()
            if locks_dir.exists():
                for entry in locks_dir.iterdir():
                    if not entry.is_file():
                        continue
                    try:
                        data = json.loads(entry.read_text(encoding="utf-8"))
                        if float(data.get("expires_at") or 0) <= now:
                            remaining.append(entry.name)
                    except Exception:
                        remaining.append(entry.name)
            return {"ok": not remaining, "expired_remaining": remaining}
        if kind == "restart_component":
            return {"ok": bool(self.component and self.component.is_alive())}
        if kind == "catchup_maintenance":
            state_path = self.memory_dir / "maintenance_state.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                return {"ok": False, "reason": "维护状态读不回来"}
            return {"ok": "decay_last_run" in state
                    or "self_check_last_run" in state,
                    "state_keys": sorted(state.keys())}
        return {"ok": False, "reason": "未知小修种类"}

    def rollback_safe(self, snapshot):
        """失败还原：把 begin_safe 的快照原样恢复，成功才返回 True。"""
        kind = snapshot.get("kind")
        if kind == "rebuild_projection":
            store = self.store()
            src = sqlite3.connect(snapshot["db_backup"])
            try:
                src.backup(store._conn)
            finally:
                src.close()
            return True
        if kind == "expire_lock":
            locks_dir = self._locks_dir()
            locks_dir.mkdir(parents=True, exist_ok=True)
            for name, encoded in (snapshot.get("locks") or {}).items():
                (locks_dir / name).write_bytes(base64.b64decode(encoded))
            return True
        if kind == "restart_component":
            if snapshot.get("was_alive") and self.component is not None:
                self.component.start()
            return bool(self.component is None or self.component.is_alive())
        if kind == "catchup_maintenance":
            state_path = self.memory_dir / "maintenance_state.json"
            if snapshot.get("state_text") is None:
                if state_path.exists():
                    state_path.unlink()
            else:
                state_path.write_text(snapshot["state_text"], encoding="utf-8")
            return True
        return False
