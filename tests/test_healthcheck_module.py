"""体检模块自有测试（贵妃原 53 项的并入移植版，2026-11-24 十一移植）。

仍是标准库真实子进程/临时 Git 仓库测试，业务适配器为明确隔离替身；
与原版的差异只有平台适配：解释器用 sys.executable（Windows 无
python3 别名）、临时目录用系统 temp、POSIX 权限位与符号链接两项
按平台跳过、模型硬超时的时间常数按 Windows 进程启动开销放宽。
集成层（真适配器）的测试在 tests/test_healthcheck_integration.py。"""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from healthcheck import Config, Healthcheck, Repair, Candidate, safe_repair
from healthcheck.core import DEFAULT_MODE, confined, emit, redact, Trace
from healthcheck.gates import Gates, GateConfigError, capability_diff, run

SCRATCH = Path(tempfile.gettempdir()) / "shier-healthcheck-tests"
SCRATCH.mkdir(parents=True, exist_ok=True)
FULL = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
REPRO = [sys.executable, "reproduce.py"]


class FakeAdapter:
    """隔离替身：不是十二实连。"""
    def __init__(self):
        self.actions = 0
        self.rollback = 0
        self.approved = True
        self.verify_ok = True
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
        return self.approved
    def isolate_dirty(self, *args):
        return False
    def begin_safe(self, *args):
        return "isolated-snapshot"
    def safe_action(self, *args):
        self.actions += 1
        return {"ok": True}
    def verify_action(self, *args):
        return {"ok": self.verify_ok}
    def rollback_safe(self, snapshot):
        self.rollback += 1
        return True


class Case(unittest.TestCase):
    def setUp(self):
        # 不清理/删除任何文件；隔离目录供失败追溯，由宿主保留策略管理。
        self.base = Path(tempfile.mkdtemp(prefix="healthcheck-test-", dir=SCRATCH))
        self.repo = self.base / "repo"
        self.repo.mkdir()
        (self.repo / "tests").mkdir()
        self.write("module.py", "def value():\n    return 0\n")
        self.write("reproduce.py", "from module import value\nassert value() == 1, '原始故障：值不等于1'\n")
        self.write("tests/test_value.py", "import unittest\nfrom module import value\nclass T(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(value(), 1)\n")
        self.config = Config(self.repo, self.base / "archive", FULL, [FULL, REPRO], True, disk_min_bytes=0,
                             repair_paths=["module.py"])
        self.adapter = FakeAdapter()
    def write(self, name, text):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    def good(self, evidence):
        return Candidate({"module.py": "def value():\n    return 1\n"}, "修正值")
    def git(self, *args):
        result = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()
    def init_git(self):
        self.git("init")
        self.git("add", ".")
        self.git("-c", "user.name=测试", "-c", "user.email=test@localhost", "commit", "-m", "基线")

    def test_default_proposal(self):
        self.assertEqual(DEFAULT_MODE, "proposal")
        result = Repair(self.config).execute("Q1", REPRO, self.good)
        self.assertEqual(result["exit_status"], "proposal_ready", result)
        self.assertEqual(result["ledger"][-1]["result"], "not_applied")
        self.assertIn("return 0", (self.repo / "module.py").read_text(encoding="utf-8"))
        self.assertTrue(result["after"]["ok"])
        self.assertTrue(result["slow"]["ok"])
        self.assertFalse(result["original_fault"]["ok"])

    def test_report_eight_and_diff(self):
        self.good_source()
        check = Healthcheck(self.config, self.adapter)
        p1, r1 = check.report()
        p2, r2 = check.report()
        self.assertEqual(len(r1["findings"]), 8)
        self.assertEqual(len(r2["findings"]), 8)
        self.assertTrue(all(f["status"] == "正常" for f in r1["findings"]))
        self.assertEqual(r2["previous_time"], r1["time"])
        self.assertTrue(p1.is_file() and p2.is_file())

    def good_source(self):
        self.write("module.py", "def value():\n    return 1\n")

    def test_unconnected_explicit(self):
        _, result = Healthcheck(self.config).report()
        memory = result["findings"][0]
        self.assertEqual(memory["status"], "异常")
        self.assertFalse(memory["details"]["connected"])

    def test_memory_exactly_twenty(self):
        self.adapter.memory = lambda sample_size: {"ok": True, "sample_count": 19}
        self.assertEqual(Healthcheck(self.config, self.adapter).check()[0]["status"], "异常")

    def test_trace_immediate_redaction(self):
        synthetic_marker = "sk-" + "x" * 40
        trace = Trace(self.base / "trace")
        trace.event("测试", api_key=synthetic_marker, output="token=" + synthetic_marker)
        content = next(trace.directory.glob("*.json")).read_text(encoding="utf-8")
        self.assertNotIn(synthetic_marker, content)
        self.assertIn("已脱敏", content)

    def test_redact_nested_fields_values(self):
        text = "Bearer " + "a" * 40
        data = redact({"nested": [{"password": "short-secret", "value": text}]})
        self.assertNotIn("short-secret", json.dumps(data))
        self.assertNotIn("a" * 40, json.dumps(data))

    def test_emit_no_overwrite(self):
        p = self.base / "a.json"
        emit(p, {"ok": True})
        with self.assertRaises(FileExistsError):
            emit(p, {})
        if os.name != "nt":  # Windows 无 POSIX 权限位，只验独占创建
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_config_limits(self):
        self.config.max_attempts = 3
        with self.assertRaises(ValueError):
            self.config.validate()

    def test_config_load_relative(self):
        p = self.base / "config.json"
        p.write_text(json.dumps({"repo": "repo", "archive": "archive", "test_command": FULL,
                                "allowed_commands": [FULL, REPRO], "full_suite": True}))
        self.assertEqual(Config.load(p).repo, self.repo)

    def test_confined_paths(self):
        for name in ("../x", "/x", ".git/config"):
            with self.assertRaises(ValueError):
                confined(self.repo, name)

    def test_symlink_rejected(self):
        try:
            (self.repo / "link.py").symlink_to(self.base / "outside")
        except OSError:
            self.skipTest("当前环境不允许创建符号链接")
        with self.assertRaises(ValueError):
            confined(self.repo, "link.py")

    def test_capability_subprocess(self):
        result = capability_diff("x=1", "import subprocess as s\ns.run(['id'])")
        self.assertFalse(result["ok"])
        self.assertIn("import:subprocess", result["added"])

    def test_capability_dynamic_builtins(self):
        self.assertFalse(capability_diff("x=1", "b=globals()['__builtins__']\nb['eval']('1')")["ok"])

    def test_capability_alias_existing_module(self):
        self.assertFalse(capability_diff("import os as x\nx.getcwd()", "import os as x\nx.remove('a')")["ok"])

    def test_capability_literal_change_allowed(self):
        self.assertTrue(capability_diff("def f(): return 0", "def f(): return 1")["ok"])

    def test_dangerous_candidate_rejected(self):
        def proposal(evidence):
            return Candidate({"module.py": "import socket\ndef value(): return 1\n"}, "不安全")
        result = Repair(self.config).execute("Q2", REPRO, proposal)
        self.assertEqual(result["exit_status"], "attempt_limit")
        self.assertEqual(len(result["ledger"]), 2)
        self.assertTrue(all(e["result"] == "gate_rejected" for e in result["ledger"]))

    def test_gate_configuration_not_candidate_failure(self):
        self.config.full_suite = False
        result = Repair(self.config).execute("Q3", REPRO, self.good)
        self.assertEqual(result["exit_status"], "gate_configuration_error")
        self.assertEqual(result["ledger"], [])

    def test_single_file_gate_rejected(self):
        command = [sys.executable, "-m", "unittest", "tests.test_value"]
        self.config.test_command = command
        self.config.allowed_commands.append(command)
        with self.assertRaises(GateConfigError):
            Gates(self.config).preflight()

    def test_filter_gate_rejected(self):
        self.config.test_command = FULL + ["-k", "value"]
        self.config.allowed_commands.append(self.config.test_command)
        with self.assertRaises(GateConfigError):
            Gates(self.config).preflight()

    def test_quick_gate_separate(self):
        self.write("module.py", "def f(\n")
        result = Gates(self.config).validate(self.repo, ["module.py"])
        self.assertFalse(result["quick"]["ok"])
        self.assertIsNone(result["slow"])

    def test_full_suite_zero_is_configuration_error(self):
        empty = self.base / "empty"
        (empty / "tests").mkdir(parents=True)
        result = Gates(self.config).validate(empty, [])
        self.assertIn("configuration_error", result["slow"])
        self.assertFalse(result["slow"]["ok"])

    def test_reflection_verbatim_and_bounded(self):
        def proposal(evidence):
            if evidence["round"] == 2:
                assert "FAIL" in evidence["failure_output_verbatim"]
            return Candidate({"module.py": "def value():\n    return 2\n"}, "仍错误")
        result = Repair(self.config).execute("Q4", REPRO, proposal)
        self.assertEqual(len(result["ledger"]), 2)
        self.assertEqual(result["exit_status"], "attempt_limit")
        self.assertIn("原始故障", result["original_fault"]["output"])
        for p in (self.config.archive / "sandboxes").glob("candidate-*/module.py"):
            self.assertIn("return 0", p.read_text(encoding="utf-8"))

    def test_no_candidate(self):
        result = Repair(self.config).execute("Q5", REPRO, lambda e: None)
        self.assertEqual(result["ledger"][0]["result"], "no_candidate")

    def test_three_format_errors(self):
        result = Repair(self.config).execute("Q6", REPRO, lambda e: "bad")
        self.assertEqual(result["exit_status"], "format_error_limit")
        self.assertEqual(len(result["ledger"]), 3)

    def test_step_budget(self):
        self.config.max_steps = 1
        result = Repair(self.config).execute("Q7", REPRO, lambda e: "bad")
        self.assertEqual(result["exit_status"], "step_budget")

    def test_cost_budget_before_call(self):
        called = []
        result = Repair(self.config).execute("Q8", REPRO, lambda e: called.append(e), max_call_cost=1)
        self.assertEqual(result["exit_status"], "cost_budget")
        self.assertEqual(called, [])

    def test_time_budget(self):
        repair = Repair(self.config)
        repair.started -= 1000
        result = repair.execute("Q9", REPRO, self.good)
        self.assertIn(result["exit_status"], {"time_budget", "observation_only"})
        self.assertEqual(result["ledger"], [])

    def test_unreproduced_observe_only(self):
        self.good_source()
        result = Repair(self.config).execute("Q10", REPRO, self.good)
        self.assertEqual(result["exit_status"], "observation_only")

    def test_apply_requires_question(self):
        result = Repair(self.config, self.adapter).execute("Q11", REPRO, self.good, mode="apply")
        self.assertEqual(result["exit_status"], "approval_required")

    def test_approved_branch_commit(self):
        self.init_git()
        old_head = self.git("rev-parse", "HEAD")
        result = Repair(self.config, self.adapter).execute("Q12", REPRO, self.good,
                                                            mode="apply", question_id="pending-12")
        self.assertEqual(result["exit_status"], "branch_ready", result)
        self.assertEqual(self.git("rev-parse", "HEAD"), old_head)
        self.assertEqual(self.git("status", "--porcelain"), "")
        artifact = result["artifact"]
        clone = Path(artifact["repository"])
        self.assertIn("return 1", (clone / "module.py").read_text(encoding="utf-8"))
        count = subprocess.check_output(["git", "rev-list", "--count", old_head + "..HEAD"], cwd=clone, text=True, encoding="utf-8", errors="replace").strip()
        self.assertEqual(count, "1")

    def test_dirty_worktree_not_mixed(self):
        self.init_git()
        self.write("human.txt", "人工改动")
        result = Repair(self.config, self.adapter).execute("Q13", REPRO, self.good,
                                                            mode="apply", question_id="pending-13")
        self.assertEqual(result["exit_status"], "executor_error")
        self.assertEqual((self.repo / "human.txt").read_text(encoding="utf-8"), "人工改动")
        self.assertIn("return 0", (self.repo / "module.py").read_text(encoding="utf-8"))

    def test_shell_and_unlisted_command(self):
        with self.assertRaises(GateConfigError):
            run(self.config, self.repo, ["sh", "-c", "true"])
        with self.assertRaises(GateConfigError):
            run(self.config, self.repo, [sys.executable, "unlisted.py"])

    def test_timeout(self):
        self.write("hang.py", "import time\ntime.sleep(5)\n")
        command = [sys.executable, "hang.py"]
        self.config.allowed_commands.append(command)
        result = run(self.config, self.repo, command, 0.02)
        self.assertTrue(result["timeout"])
        self.assertFalse(result["ok"])

    def test_output_limit(self):
        self.write("big.py", "print('z' * 10000)\n")
        command = [sys.executable, "big.py"]
        self.config.allowed_commands.append(command)
        self.config.max_output_bytes = 1024
        result = run(self.config, self.repo, command)
        self.assertTrue(result["output_limit"])
        self.assertFalse(result["ok"])

    def test_no_secret_environment_in_child(self):
        self.write("env.py", "import os\nassert 'TEST_PRIVATE_VALUE' not in os.environ\n")
        command = [sys.executable, "env.py"]
        self.config.allowed_commands.append(command)
        with patch.dict(os.environ, {"TEST_PRIVATE_VALUE": "not-to-inherit"}):
            self.assertTrue(run(self.config, self.repo, command)["ok"])

    def test_security_presence_only(self):
        self.config.secret_env_names = ["TEST_PRIVATE_VALUE"]
        with patch.dict(os.environ, {"TEST_PRIVATE_VALUE": "never-read-this-value"}):
            security = Healthcheck(self.config, self.adapter).check()[6]
        self.assertEqual(security["status"], "正常")
        self.assertNotIn("never-read-this-value", json.dumps(security))

    @unittest.skipIf(os.name == "nt", "POSIX 权限位在 Windows 无对应语义")
    def test_security_permissions(self):
        self.config.permission_paths = ["module.py"]
        (self.repo / "module.py").chmod(0o644)
        self.assertEqual(Healthcheck(self.config, self.adapter).check()[6]["status"], "异常")

    def test_weekly_catchup_once(self):
        check = Healthcheck(self.config, self.adapter)
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        self.assertTrue(check.weekly(now)["ran"])
        self.assertFalse(check.weekly(now + dt.timedelta(days=1))["ran"])
        self.assertTrue(check.weekly(now + dt.timedelta(days=30))["ran"])
        self.assertEqual(len(list((self.config.archive / "weekly").glob("*.json"))), 2)

    def test_weekly_default_interval_is_seven_days(self):
        self.assertEqual(self.config.weekly_interval_seconds, 7 * 24 * 3600)
        check = Healthcheck(self.config, self.adapter)
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        self.assertTrue(check.weekly(now)["ran"])
        self.assertFalse(check.weekly(now + dt.timedelta(days=6))["ran"])
        self.assertTrue(check.weekly(now + dt.timedelta(days=7))["ran"])

    def test_weekly_custom_interval_monthly(self):
        self.config.weekly_interval_seconds = 30 * 24 * 3600
        check = Healthcheck(self.config, self.adapter)
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        self.assertTrue(check.weekly(now)["ran"])
        self.assertFalse(check.weekly(now + dt.timedelta(days=29))["ran"])
        self.assertTrue(check.weekly(now + dt.timedelta(days=30))["ran"])
        self.assertEqual(len(list((self.config.archive / "weekly").glob("*.json"))), 2)

    def test_weekly_unconnected_no_claim(self):
        self.assertFalse(Healthcheck(self.config).weekly()["ran"])

    def test_safe_default_does_not_execute(self):
        result = safe_repair(self.config, self.adapter, "rebuild_projection", "S1")
        self.assertEqual(result["result"], "not_applied")
        self.assertEqual(self.adapter.actions, 0)

    def test_safe_approved_recheck(self):
        result = safe_repair(self.config, self.adapter, "restart_component", "S2", mode="apply", question_id="Q")
        self.assertEqual(result["result"], "applied")
        self.assertEqual(self.adapter.actions, 1)

    def test_safe_failed_rollback(self):
        self.adapter.verify_ok = False
        result = safe_repair(self.config, self.adapter, "catchup_maintenance", "S3", mode="apply", question_id="Q")
        self.assertEqual(result["result"], "still_failed")
        self.assertEqual(self.adapter.actions, 2)
        self.assertEqual(self.adapter.rollback, 1)

    def test_safe_delete_not_supported(self):
        result = safe_repair(self.config, self.adapter, "delete_memory", "S4", mode="apply", question_id="Q")
        self.assertEqual(result["result"], "not_applied")
        self.assertEqual(self.adapter.actions, 0)

    def test_archive_inside_repo_rejected(self):
        self.config.archive = self.repo / "archives"
        with self.assertRaises(ValueError):
            self.config.validate()

    def test_model_hard_timeout(self):
        import time
        def hanging(evidence):
            time.sleep(10)
        self.config.max_seconds = 2.0  # Windows 进程启动较慢，0.3 不够复现先行
        result = Repair(self.config).execute("Qtime", REPRO, hanging)
        self.assertEqual(result["exit_status"], "time_budget")

    def test_protected_code_rejected(self):
        self.config.protected_paths = ["module.py"]
        result = Repair(self.config).execute("Qprotected", REPRO, self.good)
        self.assertEqual(result["exit_status"], "attempt_limit")
        self.assertTrue(all(e["result"] == "gate_rejected" for e in result["ledger"]))

    def test_empty_repair_allowlist(self):
        self.config.repair_paths = []
        result = Repair(self.config).execute("Qempty", REPRO, self.good)
        self.assertEqual(result["exit_status"], "attempt_limit")

    def test_gate_runtime_configuration_separate(self):
        self.write("tests/test_value.py", "# 无测试\n")
        result = Repair(self.config).execute("Qzero", REPRO, self.good)
        self.assertEqual(result["exit_status"], "gate_configuration_error")
        self.assertFalse(any(e["result"] == "gate_rejected" for e in result["ledger"]))

    def test_model_budget_and_backup_integrity(self):
        self.adapter.models = lambda: {"ok": True, "judge": True, "chat": True, "month_spent": 2, "month_limit": 1}
        self.adapter.backups = lambda: {"ok": True, "last_backup": "2026-01-01", "integrity": False}
        report = Healthcheck(self.config, self.adapter).check()
        self.assertEqual(report[3]["status"], "异常")
        self.assertEqual(report[5]["status"], "异常")

    def test_incomplete_adapter_cannot_claim_healthy(self):
        self.adapter.services = lambda: {"ok": True}
        self.assertEqual(Healthcheck(self.config, self.adapter).check()[2]["status"], "异常")

    def test_original_report_when_quick_syntax_fails(self):
        result = Repair(self.config).execute("Qsyntax", REPRO,
            lambda evidence: Candidate({"module.py": "def broken(\n"}, "语法错"))
        self.assertEqual(result["exit_status"], "attempt_limit")
        self.assertIn("原始故障", result["original_fault"]["output"])

    def test_severe_headline(self):
        self.adapter.services = lambda: {"ok": False, "severity": "critical", "processes": [],
                                        "endpoints": [], "recent_logs": ["crash"]}
        _, report = Healthcheck(self.config, self.adapter).report()
        self.assertEqual(report["headlines"][0]["item"], "服务状态")

    def test_unknown_mode_rejected(self):
        result = Repair(self.config).execute("Q14", REPRO, self.good, mode="automatic")
        self.assertEqual(result["exit_status"], "configuration_error")


if __name__ == "__main__":
    unittest.main()
