# tests/test_accounting_healthcheck_20261128.py（记账对账＋余额进体检）
# 作用：验证 2026-11-28 并入体检「模型接口」项的两件证据——
#       ①记账自对账：accounting 持久账本当月总额 vs BudgetTracker
#       当月总额跑 reconcile，对不上（差额超阈值或有成本未知笔）
#       时模型接口项判异常，且体检整体不崩、其余项不受牵连；
#       ②DeepSeek 平台余额（U9）：查到写进报告；接口不通标
#       「未知」并给原因、不影响本项结论；没 Key 标「跳过」。
#       Key 只进请求头：任何返回值里都不许出现 Key 原值。
# 全部离线：假 Provider、假余额接口、临时账本与临时预算文件，
# 不真调模型、不真查余额、不碰真实账本。

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

import accounting
from healthcheck import Config, Healthcheck
from healthcheck_integration import ShierHealthcheckAdapter

_FAKE_KEY = "sk-test-not-real-for-accounting-check"


def _month():
    return datetime.now().strftime("%Y-%m")


class _FakeProvider:
    """签名与适配器调用完全对齐的假 Provider（离线探活）。"""

    def __init__(self, api_key=None, budget_tracker=None, model=None):
        self.model = model

    def chat_messages(self, messages, max_tokens=None, ledger=None,
                      thinking=None, purpose=None, caller=None):
        return {"text": "pong", "usage": {"total_tokens": 2},
                "cost_usd": 0.0}


def _fake_balance_ok(self, api_key):
    assert api_key == _FAKE_KEY  # Key 只应出现在这一跳，不许外泄
    return ([{"currency": "CNY", "total_balance": "110.00",
              "granted_balance": "10.00", "topped_up_balance": "100.00"}],
            True, None)


def _fake_balance_down(self, api_key):
    return None, None, "余额接口不通：模拟网络故障"


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """把适配器的三本账都接到临时目录：持久账本（env 路径）、
    预算账本（repo/logs/budget_usage.json）、假 Provider、假 Key。"""
    ledger = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("SHIER_LLM_CALLS_PATH", str(ledger))
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    import providers.deepseek as deepseek_module
    monkeypatch.setattr(deepseek_module, "DeepSeekProvider", _FakeProvider)
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=tmp_path / "memory")
    return adapter, ledger, tmp_path


def _write_budget(repo_root, total):
    logs = Path(repo_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "budget_usage.json").write_text(
        json.dumps({"month": _month(), "total_cost_usd": total}),
        encoding="utf-8")


def _record(ledger, cost, status="ok"):
    accounting.record_call(
        provider="deepseek", model="deepseek-flash", cost_usd=cost,
        status=status, purpose="测试", caller="test.healthcheck",
        path=ledger)


# ---------- 1. 对账对得上 + 余额查到 ----------

def test_models_accounting_ok_and_balance_reported(wired, monkeypatch):
    adapter, ledger, repo = wired
    monkeypatch.setattr(
        ShierHealthcheckAdapter, "_deepseek_balance", _fake_balance_ok)
    _record(ledger, 0.60)
    _record(ledger, 0.40)
    _write_budget(repo, 1.00)
    result = adapter.models()
    assert result["judge"] is True and result["chat"] is True
    block = result["accounting"]
    assert block["ok"] is True
    assert block["calls"] == 2
    assert block["ledger_total_usd"] == pytest.approx(1.00)
    assert block["counterpart_total_usd"] == pytest.approx(1.00)
    assert block["diff_usd"] == pytest.approx(0.0)
    assert block["unknown_cost_calls"] == 0
    assert result["accounting_ok"] is True
    assert result["ok"] is True
    balance = result["deepseek_balance"]
    assert balance["status"] == "已查"
    assert balance["is_available"] is True
    assert balance["balances"][0]["currency"] == "CNY"
    assert balance["balances"][0]["total_balance"] == "110.00"
    # Key 不许出现在任何返回值里
    assert _FAKE_KEY not in json.dumps(result, ensure_ascii=False)


# ---------- 2. 对不上 → 本项异常 ----------

def test_models_accounting_unknown_row_flags_abnormal(wired, monkeypatch):
    """账本里有成本未知的笔：总额就算对得上也判对不上 → 本项异常。"""
    adapter, ledger, repo = wired
    monkeypatch.setattr(
        ShierHealthcheckAdapter, "_deepseek_balance", _fake_balance_ok)
    _record(ledger, 1.00)
    _record(ledger, None, status="unknown_cost")
    _write_budget(repo, 1.00)
    result = adapter.models()
    assert result["judge"] is True and result["chat"] is True
    assert result["accounting"]["unknown_cost_calls"] == 1
    assert result["accounting"]["ok"] is False
    assert result["accounting_ok"] is False
    assert result["ok"] is False


def test_models_accounting_diff_flags_abnormal(wired, monkeypatch):
    """两本账差额超阈值（账本 1.00 vs 预算 2.00）：本项异常。"""
    adapter, ledger, repo = wired
    monkeypatch.setattr(
        ShierHealthcheckAdapter, "_deepseek_balance", _fake_balance_ok)
    _record(ledger, 1.00)
    _write_budget(repo, 2.00)
    result = adapter.models()
    assert result["accounting"]["diff_usd"] == pytest.approx(-1.00)
    assert result["accounting"]["ok"] is False
    assert result["ok"] is False


# ---------- 3. 余额查不到：标未知、不冤枉本项 ----------

def test_models_balance_failure_marks_unknown_not_abnormal(wired, monkeypatch):
    adapter, ledger, repo = wired
    monkeypatch.setattr(
        ShierHealthcheckAdapter, "_deepseek_balance", _fake_balance_down)
    _record(ledger, 1.00)
    _write_budget(repo, 1.00)
    result = adapter.models()
    assert result["deepseek_balance"]["status"] == "未知"
    assert "余额接口不通" in result["deepseek_balance"]["reason"]
    assert result["ok"] is True  # 余额查不到不连坐探活与对账


def test_models_no_key_balance_skipped_accounting_present(
        tmp_path, monkeypatch):
    """没 Key：余额标跳过、模型项照旧异常，但对账证据仍然在。"""
    import config  # 先触发 .env 装载，删环境变量才删得掉
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    ledger = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("SHIER_LLM_CALLS_PATH", str(ledger))
    _record(ledger, 0.50)
    _write_budget(tmp_path, 0.50)
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=tmp_path / "memory")
    result = adapter.models()
    assert result["ok"] is False
    assert result["deepseek_balance"]["status"] == "跳过"
    assert result["accounting"]["calls"] == 1
    assert result["accounting"]["ok"] is True


# ---------- 4. 余额接口解析（真方法 + 假网络） ----------

class _FakeHTTPResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_deepseek_balance_parses_official_shape(tmp_path, monkeypatch):
    """按 DeepSeek 官方返回结构解析；请求头带 Key、结果不带 Key。"""
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        return _FakeHTTPResponse({
            "is_available": True,
            "balance_infos": [{
                "currency": "USD", "total_balance": "25.50",
                "granted_balance": "0.00", "topped_up_balance": "25.50",
            }],
        })

    monkeypatch.setattr(
        "healthcheck_integration.adapter.urllib.request.urlopen",
        fake_urlopen)
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=tmp_path / "memory")
    balances, is_available, error = adapter._deepseek_balance(_FAKE_KEY)
    assert error is None
    assert is_available is True
    assert balances == [{"currency": "USD", "total_balance": "25.50",
                         "granted_balance": "0.00",
                         "topped_up_balance": "25.50"}]
    assert seen["url"] == "https://api.deepseek.com/user/balance"
    assert seen["auth"] == f"Bearer {_FAKE_KEY}"
    assert _FAKE_KEY not in json.dumps(balances)


def test_deepseek_balance_http_error_returns_reason(tmp_path, monkeypatch):
    """接口报错：返回 (None, None, 原因)，不抛、不编造余额。"""
    def fake_urlopen(request, timeout=None):
        raise OSError("连接超时")

    monkeypatch.setattr(
        "healthcheck_integration.adapter.urllib.request.urlopen",
        fake_urlopen)
    adapter = ShierHealthcheckAdapter(tmp_path, memory_dir=tmp_path / "memory")
    balances, is_available, error = adapter._deepseek_balance(_FAKE_KEY)
    assert balances is None and is_available is None
    assert "余额接口不通" in error


# ---------- 5. 模块层：对账异常只倒一项，体检不崩 ----------

class _StubAdapter:
    """模块层替身：模型项带对账失败证据，其余项契约齐全。"""

    def memory(self, sample_size):
        return {"ok": True, "sample_count": sample_size,
                "contradictions": [], "sources": [], "duplicates": [],
                "buried": [], "core_integrity": True}

    def services(self):
        return {"ok": True, "processes": [], "endpoints": [],
                "recent_logs": []}

    def models(self):
        return {"ok": False, "judge": True, "chat": True,
                "month_spent": 1.0, "month_limit": 20.0,
                "accounting_ok": False,
                "accounting": {"month": _month(), "calls": 3,
                               "ledger_total_usd": 1.0,
                               "counterpart_total_usd": 2.0,
                               "diff_usd": -1.0, "unknown_cost_calls": 0,
                               "corrupt_lines": 0, "ok": False}}

    def resources(self):
        return {"ok": True, "cpu_percent": 1, "memory_percent": 1,
                "database_bytes": 0, "ledger_bytes": 0}

    def backups(self):
        return {"ok": True, "last_backup": "2026-01-01", "integrity": True}

    def schedule(self):
        return {"ok": True, "maintenance": [], "pending_questions": 0}


def test_healthcheck_report_flags_models_item_only(tmp_path):
    """模型接口项因对账失败标异常：八项齐、其余正常项不被牵连、
    对账证据原样进报告。"""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_a.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n"
        "    def test_x(self):\n        assert True\n", encoding="utf-8")
    (repo / "module.py").write_text("def value():\n    return 1\n",
                                    encoding="utf-8")
    suite = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    config = Config(repo, tmp_path / "archive", suite, [suite], True,
                    disk_min_bytes=0, repair_paths=["module.py"])
    _path, report = Healthcheck(config, _StubAdapter()).report()
    findings = {f["item"]: f for f in report["findings"]}
    assert len(findings) == 8
    assert findings["模型接口"]["status"] == "异常"
    assert findings["模型接口"]["details"]["accounting"]["diff_usd"] == -1.0
    assert findings["记忆"]["status"] == "正常"
    assert findings["备份"]["status"] == "正常"
