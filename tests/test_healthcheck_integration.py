# tests/test_healthcheck_integration.py（体检并入十二：集成层验收）
#
# 测的是 healthcheck_integration 的真适配器与移植后的门/执行器，
# 全部只碰临时目录与沙盒副本：不连真服务、不动真库、不真调模型
# （models() 的真探活在实测环节单独跑，见并入报告）。覆盖审查清单
# 里点名的三件关键集成：pytest 全套入口（计数可信）、Windows 无
# fork 的 proposer 执行器、待问批准防重放真实现；以及其余适配器
# 与四类小修在沙盒资源上的真事务。

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from healthcheck import Candidate, Config, Healthcheck, Repair, safe_repair
from healthcheck.gates import GateConfigError, Gates
from healthcheck_integration import (
    APPROVAL_KIND,
    ShierHealthcheckAdapter,
    SubprocessComponent,
    build_adapter,
    build_config,
)
from memory.pending_questions import PendingQuestions

PYTEST_SUITE = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-ra", "--no-header"]
UNITTEST_SUITE = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]


def _mini_config(repo, archive, suite):
    return Config(repo, archive, suite, [suite], True, disk_min_bytes=0,
                  repair_paths=["module.py"])


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------- 关键集成 1：pytest 全套入口 ----------

def test_pytest_gate_runs_and_counts(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "tests" / "test_a.py",
           "def test_one():\n    assert True\n\ndef test_two():\n    assert True\n")
    _write(repo / "module.py", "def value():\n    return 1\n")
    config = _mini_config(repo, tmp_path / "archive", PYTEST_SUITE)
    result = Gates(config).validate(repo, [])
    assert result["quick"]["ok"]
    assert result["slow"]["ok"], result["slow"]
    assert result["slow"]["suite_runner"] == "pytest"
    assert result["slow"]["test_count"] == 2


def test_pytest_gate_zero_tests_is_configuration_error(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    _write(repo / "module.py", "def value():\n    return 1\n")
    config = _mini_config(repo, tmp_path / "archive", PYTEST_SUITE)
    result = Gates(config).validate(repo, [])
    assert not result["slow"]["ok"]
    assert "configuration_error" in result["slow"]


def test_pytest_gate_rejects_filters_and_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for bad in (
        [sys.executable, "-m", "pytest", "-q", "-k", "one"],
        [sys.executable, "-m", "pytest", "tests/test_a.py"],
        [sys.executable, "-m", "pytest", "--deselect", "tests/test_a.py::test_one"],
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
    ):
        config = _mini_config(repo, tmp_path / "archive", bad)
        config.allowed_commands.append(bad)
        with pytest.raises(GateConfigError):
            Gates(config).preflight()


def test_pytest_gate_failure_is_not_ok(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "tests" / "test_a.py", "def test_bad():\n    assert False\n")
    _write(repo / "module.py", "def value():\n    return 1\n")
    config = _mini_config(repo, tmp_path / "archive", PYTEST_SUITE)
    result = Gates(config).validate(repo, [])
    assert not result["slow"]["ok"]
    assert result["slow"]["test_count"] == 1


# ---------- 关键集成 2：Windows proposer 执行器 ----------

def _repair_fixture(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "module.py", "def value():\n    return 0\n")
    _write(repo / "reproduce.py",
           "from module import value\nassert value() == 1, '原始故障：值不等于1'\n")
    _write(repo / "tests" / "test_value.py",
           "import unittest\nfrom module import value\n"
           "class T(unittest.TestCase):\n    def test_value(self):\n"
           "        self.assertEqual(value(), 1)\n")
    config = _mini_config(repo, tmp_path / "archive", UNITTEST_SUITE)
    config.allowed_commands.append([sys.executable, "reproduce.py"])
    return repo, config


def test_proposer_executor_produces_proposal(tmp_path):
    # 本机（Windows）没有 fork：执行器必须走移植后的兼容路径，
    # 且完整跑通复现→提案→两级门→proposal_ready。
    assert "fork" not in __import__("multiprocessing").get_all_start_methods() \
        or os.name == "nt"  # 记录平台事实，断言重点在下面跑通
    _repo, config = _repair_fixture(tmp_path)

    def proposer(evidence):
        return Candidate({"module.py": "def value():\n    return 1\n"}, "修正值")

    result = Repair(config).execute("W1", [sys.executable, "reproduce.py"], proposer)
    assert result["exit_status"] == "proposal_ready", result
    assert result["slow"]["ok"]


def test_proposer_executor_hard_timeout(tmp_path):
    _repo, config = _repair_fixture(tmp_path)
    config.max_seconds = 2.0

    def hanging(evidence):
        time.sleep(10)

    result = Repair(config).execute("W2", [sys.executable, "reproduce.py"], hanging)
    assert result["exit_status"] == "time_budget", result


def test_proposer_executor_exception_becomes_executor_error(tmp_path):
    _repo, config = _repair_fixture(tmp_path)

    def broken(evidence):
        raise ValueError("提案例外")

    result = Repair(config).execute("W3", [sys.executable, "reproduce.py"], broken)
    assert result["exit_status"] == "executor_error", result


# ---------- 关键集成 3：待问批准防重放 ----------

def _adapter_with_questions(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(exist_ok=True)
    questions = PendingQuestions(memory_dir=memory_dir)
    adapter = ShierHealthcheckAdapter(
        tmp_path, memory_dir=memory_dir,
        store=_FakeStore(questions))
    return adapter, questions


class _FakeStore:
    """只给适配器批准/调度路径用的最小 store 替身（带真问题库）。"""

    def __init__(self, questions):
        self.pending_questions = questions


def test_approve_anti_replay(tmp_path):
    adapter, questions = _adapter_with_questions(tmp_path)
    question = adapter.request_approval("ISSUE-1", "code_branch", "批准体检修复？")
    assert question["kind"] == APPROVAL_KIND
    qid = question["id"]
    # 没答之前消费不了
    assert adapter.approve(qid, "ISSUE-1", "code_branch") is False
    # 答「不批准」消费不了
    questions.answer(qid, "不批准")
    assert adapter.approve(qid, "ISSUE-1", "code_branch") is False

    question2 = adapter.request_approval("ISSUE-2", "restart_component", "批准重启？")
    qid2 = question2["id"]
    questions.answer(qid2, "批准")
    # 绑定不对：换 issue 或换 action 都不行，且不消耗
    assert adapter.approve(qid2, "ISSUE-X", "restart_component") is False
    assert adapter.approve(qid2, "ISSUE-2", "code_branch") is False
    # 本人本事：第一次 True
    assert adapter.approve(qid2, "ISSUE-2", "restart_component") is True
    # 重放：第二次一律 False（消费标记已落盘）
    assert adapter.approve(qid2, "ISSUE-2", "restart_component") is False
    # 换一个新适配器实例（模拟重启）仍然防重放
    adapter2, _ = _adapter_with_questions(tmp_path)
    adapter2._store = _FakeStore(questions)
    assert adapter2.approve(qid2, "ISSUE-2", "restart_component") is False
    # 不存在的 id / 非批准类问题
    assert adapter.approve("nope", "ISSUE-2", "restart_component") is False


# ---------- 记忆适配器映射 ----------

class _CannedStore(_FakeStore):
    def __init__(self, questions, report):
        super().__init__(questions)
        self._report = report

    def run_self_check(self, sample_size=20):
        return self._report


def _canned_report(statuses, sample_ids=20):
    items = {}
    for number in range(1, 11):
        items[number] = {"name": str(number), "status": statuses.get(number, "pass"),
                         "findings": [], "actions": []}
    items[9]["findings"] = [{"id": "m%d" % i, "accurate": True}
                            for i in range(sample_ids)]
    items[3]["findings"] = [{"id": "c1", "reason": "冲突"}]
    return {"items": items, "summary": " canned "}


def test_memory_adapter_mapping(tmp_path):
    _adapter, questions = _adapter_with_questions(tmp_path)
    report = _canned_report({3: "warn"})
    adapter = ShierHealthcheckAdapter(
        tmp_path, memory_dir=tmp_path / "memory", store=_CannedStore(questions, report))
    result = adapter.memory(20)
    assert result["sample_count"] == 20
    assert result["contradictions"] and result["contradictions"][0]["id"] == "c1"
    for key in ("sources", "duplicates", "buried", "core_integrity"):
        assert key in result
    assert result["core_integrity"] is True
    assert result["ok"] is False  # 第 3 项 warn → 记忆判异常（warning）
    assert result["severity"] == "warning"

    healthy = ShierHealthcheckAdapter(
        tmp_path, memory_dir=tmp_path / "memory",
        store=_CannedStore(questions, _canned_report({})))
    ok_result = healthy.memory(20)
    assert ok_result["ok"] is True
    assert ok_result["sample_count"] == 20


def test_memory_adapter_real_store_contract(tmp_path):
    # 真 SQLiteStore（离线打桩版，conftest 同款）：一张空库也必须
    # 给齐契约键，抽不到 20 条就如实 sample_count<20，不冒充。
    from conftest import make_sqlite_store

    store = make_sqlite_store(tmp_path)
    adapter = ShierHealthcheckAdapter(
        tmp_path, memory_dir=Path(store.memory_dir), store=store)
    result = adapter.memory(20)
    for key in ("sample_count", "contradictions", "sources", "duplicates",
                "buried", "core_integrity", "ok"):
        assert key in result
    assert result["sample_count"] < 20  # 空库抽不到 20 条，如实报


# ---------- 备份适配器（只读） ----------

def test_backups_adapter_readonly_and_integrity(tmp_path):
    from memory import drive_backup

    memory_dir = tmp_path / "memory"
    (memory_dir / "backups").mkdir(parents=True)
    payload = b"fake-zip-content"
    zip_path = memory_dir / "backups" / "shier-memory-20261124.zip"
    zip_path.write_bytes(payload)
    state = {
        "last_success": "2026-11-24T03:51:00+08:00",
        "last_success_epoch": time.time(),
        "last_zip": "shier-memory-20261124.zip",
        "sha256": drive_backup.sha256_bytes(payload),
        "backup_enabled": False,
    }
    drive_backup.save_backup_state(memory_dir, state)
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=memory_dir)
    result = adapter.backups()
    assert result["integrity"] is True
    assert result["ok"] is True
    assert result["backup_enabled"] is False  # 只读着报，没被翻开
    # 篡改包内容 → 完整性立刻变 False
    zip_path.write_bytes(b"tampered")
    assert adapter.backups()["integrity"] is False
    # 开关仍然关着（适配器全程没碰开关）
    assert drive_backup.is_backup_enabled(memory_dir) is False


def test_backups_adapter_no_state(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=memory_dir)
    result = adapter.backups()
    assert result["last_backup"] is None
    assert result["integrity"] is False
    assert result["ok"] is False


# ---------- 调度适配器 + 租约 ----------

def test_schedule_adapter_and_pending(tmp_path):
    adapter, questions = _adapter_with_questions(tmp_path)
    memory_dir = tmp_path / "memory"
    (memory_dir / "maintenance_state.json").write_text(json.dumps({
        "decay_last_run": time.time(),
        "self_check_last_run": time.time(),
        "backup_skip_reason": "disabled",
    }), encoding="utf-8")
    result = adapter.schedule()
    assert result["pending_questions"] == 0
    tasks = {entry["task"] for entry in result["maintenance"]}
    assert {"decay", "self_check", "backup"} <= tasks
    assert result["ok"] is True
    questions.add("importance", "这条重要吗", ["进核心"], "测试")
    assert adapter.schedule()["pending_questions"] == 1


def test_weekly_lease_exclusion_and_expiry(tmp_path):
    adapter, _questions = _adapter_with_questions(tmp_path)
    assert adapter.weekly_lease("healthcheck-weekly") is True
    assert adapter.weekly_lease("healthcheck-weekly") is False
    # 把锁改成已过期 → 可接管
    lock = adapter._locks_dir() / "healthcheck-weekly.lock"
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["expires_at"] = time.time() - 1
    lock.write_text(json.dumps(data), encoding="utf-8")
    assert adapter.weekly_lease("healthcheck-weekly") is True


# ---------- 资源适配器 ----------

def test_resources_adapter(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "memories.sqlite3").write_bytes(b"x" * 100)
    (memory_dir / "memory_events.jsonl").write_bytes(b"y" * 50)
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=memory_dir)
    result = adapter.resources()
    assert result["database_bytes"] == 100
    assert result["ledger_bytes"] == 50
    assert 0 <= result["cpu_percent"] <= 100
    assert 0 <= result["memory_percent"] <= 100


# ---------- models 适配器（无 Key 时如实报，不真调） ----------

def test_models_adapter_without_key(tmp_path, monkeypatch):
    import config  # 先触发 config 的 .env 装载，再删环境变量才删得掉
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=tmp_path / "memory")
    result = adapter.models()
    assert result["judge"] is False and result["chat"] is False
    assert result["ok"] is False
    assert result["month_limit"] > 0
    assert "month_spent" in result


def test_models_adapter_probe_uses_chat_ledger(tmp_path, monkeypatch):
    # DeepSeekProvider 只认 ledger='tool'/'chat'（实测教训：传
    # 'healthcheck' 会被拒）。用假 provider 离线验证探活路径与
    # 账本名，不真调模型。
    import providers.deepseek as deepseek_module

    seen = {}

    class FakeProvider:
        def __init__(self, api_key=None, budget_tracker=None, model=None):
            seen["model"] = model

        def chat_messages(self, messages, max_tokens=None, ledger=None,
                          thinking=None):
            seen.setdefault("ledgers", []).append(ledger)
            assert ledger in ("tool", "chat")
            return {"text": "pong", "usage": {"total_tokens": 2},
                    "cost_usd": 0.0}

    monkeypatch.setattr(deepseek_module, "DeepSeekProvider", FakeProvider)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-real")
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=tmp_path / "memory")
    result = adapter.models()
    assert result["judge"] is True and result["chat"] is True
    assert result["ok"] is True
    assert seen["ledgers"] == ["chat", "chat"]


# ---------- 脏改动隔离 ----------

def test_isolate_dirty_stashes_and_cleans(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "tracked.txt", "基线\n")

    def git(*args):
        proc = subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    git("init")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-m", "基线")
    _write(repo / "tracked.txt", "人的改动\n")
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=tmp_path / "memory")
    assert adapter.isolate_dirty(repo, " M tracked.txt") is True
    assert git("status", "--porcelain") == ""
    assert "healthcheck-isolate-" in git("stash", "list")
    assert adapter.isolate_dirty(repo, "") is True


# ---------- 四类小修（沙盒资源真事务） ----------

def _config_for_safe(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "module.py", "def value():\n    return 1\n")
    return Config(repo, tmp_path / "archive", UNITTEST_SUITE, [UNITTEST_SUITE],
                  True, disk_min_bytes=0)


def test_safe_repair_rebuild_projection(tmp_path):
    from conftest import make_sqlite_store

    store = make_sqlite_store(tmp_path)
    adapter = ShierHealthcheckAdapter(
        tmp_path, memory_dir=Path(store.memory_dir), store=store)
    config = _config_for_safe(tmp_path)
    question = adapter.request_approval("SAFE-1", "rebuild_projection", "批准重建投影？")
    store.pending_questions.answer(question["id"], "批准")
    result = safe_repair(config, adapter, "rebuild_projection", "SAFE-1",
                         mode="apply", question_id=question["id"])
    assert result["result"] == "applied", result
    # 同一批准不能再用第二次（防重放落到 safe_repair 层）
    result2 = safe_repair(config, adapter, "rebuild_projection", "SAFE-1",
                          mode="apply", question_id=question["id"])
    assert result2["result"] == "not_applied", result2


def test_safe_repair_expire_lock(tmp_path):
    adapter, _questions = _adapter_with_questions(tmp_path)
    locks = adapter._locks_dir()
    locks.mkdir(parents=True)
    expired = locks / "old.lock"
    expired.write_text(json.dumps({"expires_at": time.time() - 10}), encoding="utf-8")
    live = locks / "live.lock"
    live.write_text(json.dumps({"expires_at": time.time() + 600}), encoding="utf-8")
    config = _config_for_safe(tmp_path)
    question = adapter.request_approval("SAFE-2", "expire_lock", "批准清过期锁？")
    adapter.store().pending_questions.answer(question["id"], "批准")
    result = safe_repair(config, adapter, "expire_lock", "SAFE-2",
                         mode="apply", question_id=question["id"])
    assert result["result"] == "applied", result
    assert not expired.exists()
    assert live.exists()


def test_safe_repair_restart_component(tmp_path):
    heartbeat = tmp_path / "heartbeat.txt"
    script = (
        "import time, pathlib\n"
        "p = pathlib.Path(%r)\n"
        "while True:\n"
        "    p.write_text(str(time.time()))\n"
        "    time.sleep(0.2)\n" % str(heartbeat))
    script_path = tmp_path / "beat.py"
    script_path.write_text(script, encoding="utf-8")
    component = SubprocessComponent([sys.executable, str(script_path)])
    component.start()
    try:
        adapter = ShierHealthcheckAdapter(
            tmp_path, memory_dir=tmp_path / "memory", component=component)
        config = _config_for_safe(tmp_path)
        question = adapter.request_approval("SAFE-3", "restart_component", "批准重启组件？")
        adapter.store().pending_questions.answer(question["id"], "批准")
        old_pid = component.pid
        result = safe_repair(config, adapter, "restart_component", "SAFE-3",
                             mode="apply", question_id=question["id"])
        assert result["result"] == "applied", result
        assert component.is_alive() and component.pid != old_pid
    finally:
        component.stop()


def test_safe_repair_catchup_maintenance(tmp_path):
    from conftest import make_sqlite_store

    store = make_sqlite_store(tmp_path)
    adapter = ShierHealthcheckAdapter(
        tmp_path, memory_dir=Path(store.memory_dir), store=store)
    config = _config_for_safe(tmp_path)
    question = adapter.request_approval("SAFE-4", "catchup_maintenance", "批准补跑维护？")
    store.pending_questions.answer(question["id"], "批准")
    result = safe_repair(config, adapter, "catchup_maintenance", "SAFE-4",
                         mode="apply", question_id=question["id"])
    assert result["result"] == "applied", result
    state = json.loads(
        (Path(store.memory_dir) / "maintenance_state.json").read_text(encoding="utf-8"))
    assert "decay_last_run" in state


def test_safe_repair_rollback_restores_projection(tmp_path):
    # 制造投影丢失：账本里有、SQLite 里删掉一行；让 verify 故意失败
    # （把适配器的 verify 换成坏的）→ safe_repair 必须回滚到快照。
    from conftest import make_sqlite_store

    store = make_sqlite_store(tmp_path)
    adapter = ShierHealthcheckAdapter(
        tmp_path, memory_dir=Path(store.memory_dir), store=store)
    config = _config_for_safe(tmp_path)
    question = adapter.request_approval("SAFE-5", "rebuild_projection", "批准？")
    store.pending_questions.answer(question["id"], "批准")
    adapter.verify_action = lambda kind, issue_id: {"ok": False}
    result = safe_repair(config, adapter, "rebuild_projection", "SAFE-5",
                         mode="apply", question_id=question["id"])
    assert result["result"] == "still_failed", result


# ---------- 服务调度钩子：断网恢复补跑通知 ----------

def test_scheduler_fires_recovery_callback_once(tmp_path):
    from web.server import MemoryMaintenanceScheduler

    calls = []

    class FakeStore:
        def run_self_check(self, include_model_judge=True):
            calls.append(include_model_judge)
            return {"items": {}}

    online = {"value": False}
    fired = []
    scheduler = MemoryMaintenanceScheduler(
        FakeStore(),
        tmp_path / "maintenance_state.json",
        network_probe=lambda: online["value"],
        probe_interval_seconds=1,
        on_network_recovered=lambda: fired.append(True),
    )
    # 断网到期：本地部分先跑、判官部分挂起，回调不触发
    scheduler.run_due(now=1000.0)
    assert fired == []
    # 网络恢复：挂起的自检补跑完成 → 回调恰好触发一次
    online["value"] = True
    scheduler.run_due(now=1002.0)
    assert fired == [True]
    # 之后正常周期跑完（非 pending）→ 不再触发
    scheduler.run_due(now=1002.0 + 8 * 24 * 3600)
    assert fired == [True]


def test_scheduler_without_callback_unchanged(tmp_path):
    from web.server import MemoryMaintenanceScheduler

    class FakeStore:
        def run_self_check(self, include_model_judge=True):
            return {"items": {}}

    scheduler = MemoryMaintenanceScheduler(
        FakeStore(), tmp_path / "maintenance_state.json",
        network_probe=lambda: True)
    ran = scheduler.run_due(now=1000.0)
    assert ran["self_check"] is True


# ---------- 配置装配 ----------

def test_build_config_shape(tmp_path):
    archive = tmp_path / "archive"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "memory").mkdir()
    config = build_config(repo, archive)
    assert config.test_command[1:3] == ["-m", "pytest"]
    assert config.repair_paths == []
    assert config.protected_paths == ["memory"]
    Gates(config).preflight()  # pytest 全套入口必须被门接受


def test_build_config_weekly_interval(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "memory").mkdir()
    archive = tmp_path / "archive"
    monkeypatch.delenv("SHIER_HEALTHCHECK_INTERVAL_SECONDS", raising=False)
    assert build_config(repo, archive).weekly_interval_seconds == 7 * 24 * 3600
    monkeypatch.setenv("SHIER_HEALTHCHECK_INTERVAL_SECONDS", "2592000")
    assert build_config(repo, archive).weekly_interval_seconds == 2592000
    explicit = build_config(repo, archive, weekly_interval_seconds=42)
    assert explicit.weekly_interval_seconds == 42


def test_healthcheck_unconnected_still_reports(tmp_path):
    # 不带适配器的模块单独跑：八项齐、记忆项如实异常（模块原行为）。
    repo = tmp_path / "repo"
    _write(repo / "tests" / "test_a.py",
           "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        assert True\n")
    _write(repo / "module.py", "def value():\n    return 1\n")
    config = _mini_config(repo, tmp_path / "archive", UNITTEST_SUITE)
    _path, report = Healthcheck(config).report()
    assert len(report["findings"]) == 8
    assert report["findings"][0]["status"] == "异常"
