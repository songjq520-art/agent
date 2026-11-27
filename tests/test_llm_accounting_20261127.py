# tests/test_llm_accounting_20261127.py（大模型逐笔记账 + 自对账测试）
# 作用：验证 accounting.py 持久账本（一笔不丢）、BudgetTracker 转写、
#       两家 provider 在有/无 tracker 时都落账且不重复、体检探针用途、
#       tools/cli.py 翻译直连补记（堵绕过）、按月汇总与平台对账，
#       以及全仓防绕过扫描（不许再出现未登记的直连大模型 HTTP 调用）。
# 全部离线：网络一律用 Mock 假装 requests 返回固定数据，不产生费用；
#       账本文件用 SHIER_LLM_CALLS_PATH 指到临时目录，不碰真实账本。

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import accounting
from config import BudgetTracker
from providers.deepseek import DeepSeekProvider
from providers.muse import MuseProvider
from tools.cli import _deepseek_translate


@pytest.fixture()
def ledger_file(tmp_path, monkeypatch):
    """把持久账本指到临时文件，返回它的路径。"""
    path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("SHIER_LLM_CALLS_PATH", str(path))
    return path


def _rows(path):
    return accounting.read_calls(path=path)


def _fake_response(payload, status_code=200):
    fake = Mock()
    fake.json.return_value = payload
    fake.raise_for_status.return_value = None
    fake.status_code = status_code
    return fake


_USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "total_tokens": 1500,
    "prompt_tokens_details": {"cached_tokens": 200},
}
# Flash 价：(1000-200)/1e6*0.15 + 200/1e6*0.003 + 500/1e6*0.60
_FLASH_COST = 0.0004206
# Muse 价：(1000-200)/1e6*1.25 + 200/1e6*0.15 + 500/1e6*4.25
_MUSE_COST = 0.003155


# ---------- 1. record_call 基本行为 ----------

def test_record_call_writes_full_row(ledger_file):
    """record_call 落一行：字段齐全、Token 从 usage 提取、金额取整 6 位。"""
    ok = accounting.record_call(
        provider="deepseek", model="deepseek-flash", cost_usd=0.123456789,
        usage=_USAGE, purpose="测试", caller="test.case",
        ledger_book="tool", task_id="t1", trace_id="tr1",
        path=ledger_file)
    assert ok is True
    rows = _rows(ledger_file)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "deepseek"
    assert row["model"] == "deepseek-flash"
    assert row["purpose"] == "测试"
    assert row["caller"] == "test.case"
    assert row["ledger_book"] == "tool"
    assert row["task_id"] == "t1"
    assert row["trace_id"] == "tr1"
    assert row["prompt_tokens"] == 1000
    assert row["completion_tokens"] == 500
    assert row["cached_tokens"] == 200
    assert row["cost_usd"] == pytest.approx(0.123457, abs=1e-9)
    assert row["status"] == "ok"
    assert row["estimated"] is False
    assert row["ts"]  # 有时间戳


def test_record_call_rejects_bad_purpose(ledger_file):
    """非法 purpose 是编程错误，必须响（ValueError），不能悄悄记错账。"""
    with pytest.raises(ValueError):
        accounting.record_call(
            provider="deepseek", model="deepseek-flash", cost_usd=0.1,
            purpose="随便", path=ledger_file)
    assert _rows(ledger_file) == []


def test_record_call_unknown_and_estimated_shapes(ledger_file):
    """成本未知记 null + unknown_cost；估算行 status 与 estimated 对齐。"""
    accounting.record_call(
        provider="deepseek", model="deepseek-flash", cost_usd=None,
        status="unknown_cost", path=ledger_file)
    accounting.record_call(
        provider="deepseek", model="deepseek-flash", cost_usd=0.5,
        status="estimated", path=ledger_file)
    rows = _rows(ledger_file)
    assert rows[0]["cost_usd"] is None
    assert rows[0]["status"] == "unknown_cost"
    assert rows[0]["prompt_tokens"] == 0  # 无 usage 不编造 Token
    assert rows[1]["status"] == "estimated"
    assert rows[1]["estimated"] is True


def test_read_calls_skips_bad_lines(ledger_file):
    """账本里混了坏行也要能读：坏行跳过，好行照常返回。"""
    accounting.record_call(
        provider="deepseek", model="deepseek-flash", cost_usd=0.1,
        path=ledger_file)
    with ledger_file.open("a", encoding="utf-8") as fh:
        fh.write("这不是一行 JSON\n")
        fh.write('{"ts": "2026-11-01T00:00:00", "model": "x", "cost_usd": 1}\n')
    rows = _rows(ledger_file)
    assert len(rows) == 2


# ---------- 2~4. BudgetTracker 转写 ----------

def test_tracker_add_cost_writes_ledger_row(tmp_path, ledger_file):
    """add_cost 后持久账本出现对应行：金额/模型/Token/用途都对得上。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(
        _FLASH_COST, model="deepseek-flash", ledger="chat",
        task_id="task-1", trace_id="trace-1", usage=_USAGE,
        purpose="测试", caller="test.tracker", provider="deepseek")
    rows = _rows(ledger_file)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "deepseek"
    assert row["model"] == "deepseek-flash"
    assert row["cost_usd"] == pytest.approx(_FLASH_COST, abs=1e-6)  # 账本金额取整 6 位
    assert row["prompt_tokens"] == 1000
    assert row["completion_tokens"] == 500
    assert row["cached_tokens"] == 200
    assert row["purpose"] == "测试"
    assert row["caller"] == "test.tracker"
    assert row["ledger_book"] == "chat"
    assert row["task_id"] == "task-1"
    assert row["status"] == "ok"


def test_501_calls_none_lost(tmp_path, ledger_file):
    """连记 501 笔：tracker.records 只剩最近 500，但持久账本一笔不丢。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    for _ in range(501):
        tracker.add_cost(0.000001, model="deepseek-flash")
    assert len(tracker.records) == 500
    rows = _rows(ledger_file)
    assert len(rows) == 501


def test_tracker_mark_unknown_cost_writes_unknown_row(tmp_path, ledger_file):
    """mark_unknown_cost 在持久账本留一笔 unknown 行（成本 null）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.mark_unknown_cost(
        provider="deepseek", model="deepseek-flash", caller="test.unknown")
    assert tracker.unknown_cost is True
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["status"] == "unknown_cost"
    assert rows[0]["cost_usd"] is None
    assert rows[0]["model"] == "deepseek-flash"
    assert rows[0]["caller"] == "test.unknown"


# ---------- 5~6. DeepSeekProvider 落账 ----------

def _deepseek_payload(with_usage=True):
    payload = {"choices": [{"message": {"content": "你好"}}]}
    if with_usage:
        payload["usage"] = dict(_USAGE)
    return payload


def test_deepseek_no_tracker_with_usage_writes_row(ledger_file):
    """无 tracker + 有 usage：provider 直接写持久账本一行。"""
    provider = DeepSeekProvider(api_key="fake-key-for-test")
    with patch("providers.deepseek.requests.post",
               return_value=_fake_response(_deepseek_payload())):
        result = provider.chat("你好")
    assert result["cost_usd"] == pytest.approx(_FLASH_COST)
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["provider"] == "deepseek"
    assert rows[0]["model"] == "deepseek-flash"
    assert rows[0]["caller"] == "deepseek.chat_messages"
    assert rows[0]["cost_usd"] == pytest.approx(_FLASH_COST, abs=1e-6)
    assert rows[0]["status"] == "ok"


def test_deepseek_with_tracker_writes_exactly_one_row(tmp_path, ledger_file):
    """有 tracker + 有 usage：只走 tracker 转写，恰好一行，不重复。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post",
               return_value=_fake_response(_deepseek_payload())):
        provider.chat("你好")
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == pytest.approx(_FLASH_COST, abs=1e-6)


def test_deepseek_no_usage_writes_unknown_row_no_tracker(ledger_file):
    """无 tracker + 无 usage：直接写一笔 unknown 行。"""
    provider = DeepSeekProvider(api_key="fake-key-for-test")
    with patch("providers.deepseek.requests.post",
               return_value=_fake_response(_deepseek_payload(with_usage=False))):
        result = provider.chat("你好")
    assert result["cost_usd"] is None
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["status"] == "unknown_cost"
    assert rows[0]["cost_usd"] is None


def test_deepseek_no_usage_writes_unknown_row_with_tracker(tmp_path, ledger_file):
    """有 tracker + 无 usage：经 mark_unknown_cost 转写一笔 unknown 行。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post",
               return_value=_fake_response(_deepseek_payload(with_usage=False))):
        provider.chat("你好")
    assert tracker.unknown_cost is True
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["status"] == "unknown_cost"
    assert rows[0]["model"] == "deepseek-flash"


def test_deepseek_chat_with_tools_writes_row(ledger_file):
    """chat_with_tools（无 tracker）也落账，caller 标明是工具路。"""
    provider = DeepSeekProvider(api_key="fake-key-for-test")
    payload = {
        "choices": [{"message": {"content": "", "tool_calls": []}}],
        "usage": dict(_USAGE),
    }
    tools = [{"type": "function",
              "function": {"name": "noop", "description": "无操作",
                           "parameters": {"type": "object", "properties": {}}}}]
    with patch("providers.deepseek.requests.post",
               return_value=_fake_response(payload)):
        provider.chat_with_tools(
            [{"role": "user", "content": "你好"}], tools)
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["caller"] == "deepseek.chat_with_tools"
    assert rows[0]["ledger_book"] == "tool"


# ---------- 7. MuseProvider 落账 ----------

def test_muse_no_tracker_writes_row(ledger_file):
    """Muse 无 tracker + 有 usage：直接写持久账本一行（provider=muse）。"""
    provider = MuseProvider(api_key="fake-key-for-test")
    payload = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": dict(_USAGE),
    }
    with patch("providers.muse.requests.post",
               return_value=_fake_response(payload)):
        result = provider.chat("你好")
    assert result["cost_usd"] == pytest.approx(_MUSE_COST)
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["provider"] == "muse"
    assert rows[0]["model"] == "muse-spark-1.3"
    assert rows[0]["caller"] == "muse.chat_messages"
    assert rows[0]["cost_usd"] == pytest.approx(_MUSE_COST, abs=1e-6)


def test_muse_with_tracker_writes_exactly_one_row(tmp_path, ledger_file):
    """Muse 有 tracker：add_cost 补齐 usage 后转写，恰好一行。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    payload = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": dict(_USAGE),
    }
    with patch("providers.muse.requests.post",
               return_value=_fake_response(payload)):
        provider.chat("你好")
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["prompt_tokens"] == 1000  # usage 已透传进账本
    assert tracker.cost_for_model("muse-spark-1.3") == pytest.approx(_MUSE_COST)


# ---------- 8. tools/cli.py 翻译直连补记 ----------

def test_translate_records_call(monkeypatch, ledger_file):
    """_deepseek_translate 成功后补记一行：caller 标明翻译路，金额按 Flash 价。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")
    payload = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": dict(_USAGE),
    }
    with patch("tools.cli.requests.post",
               return_value=_fake_response(payload)):
        assert _deepseek_translate("你好", "en") == "hello"
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["caller"] == "tools.cli.translate"
    assert rows[0]["provider"] == "deepseek"
    assert rows[0]["model"] == "deepseek-flash"
    assert rows[0]["cost_usd"] == pytest.approx(_FLASH_COST, abs=1e-6)


def test_translate_no_usage_records_unknown(monkeypatch, ledger_file):
    """翻译响应没 usage：补记一笔 unknown 行，翻译结果照常返回。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")
    payload = {"choices": [{"message": {"content": "hello"}}]}
    with patch("tools.cli.requests.post",
               return_value=_fake_response(payload)):
        assert _deepseek_translate("你好", "en") == "hello"
    rows = _rows(ledger_file)
    assert len(rows) == 1
    assert rows[0]["status"] == "unknown_cost"
    assert rows[0]["cost_usd"] is None


# ---------- 9. 汇总与对账 ----------

def _write_row(path, ts, model, purpose, cost, status="ok"):
    row = {
        "ts": ts, "provider": "deepseek", "model": model,
        "purpose": purpose, "caller": "test", "ledger_book": "chat",
        "task_id": "", "trace_id": "", "prompt_tokens": 0,
        "completion_tokens": 0, "cached_tokens": 0,
        "cost_usd": cost, "status": status, "estimated": False,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_monthly_summary_cross_month(tmp_path):
    """手造跨月数据：汇总只算当月，按模型/用途分项正确。"""
    path = tmp_path / "llm_calls.jsonl"
    _write_row(path, "2026-10-05T10:00:00", "deepseek-flash", "日常", 0.10)
    _write_row(path, "2026-10-06T10:00:00", "deepseek-flash", "日常", 0.05)
    _write_row(path, "2026-10-07T10:00:00", "deepseek-v4-pro", "测试", 0.20)
    _write_row(path, "2026-11-01T10:00:00", "deepseek-flash", "日常", 9.99)
    summary = accounting.monthly_summary("2026-10", path=path)
    assert summary["calls"] == 3
    assert summary["total_usd"] == pytest.approx(0.35)
    assert summary["unknown_cost_calls"] == 0
    assert summary["by_model"]["deepseek-flash"] == {"calls": 2, "total_usd": pytest.approx(0.15)}
    assert summary["by_model"]["deepseek-v4-pro"]["calls"] == 1
    assert summary["by_purpose"]["日常"]["calls"] == 2
    assert summary["by_purpose"]["测试"] == {"calls": 1, "total_usd": pytest.approx(0.20)}


def test_reconcile_thresholds_and_unknown(tmp_path):
    """对账：差额在阈值内 ok；超阈值不对；有 unknown 行永远不对。"""
    path = tmp_path / "llm_calls.jsonl"
    _write_row(path, "2026-11-03T10:00:00", "deepseek-flash", "日常", 1.00)
    # 差额 0.01 <= max(1.01*2%, 0.07) → 对得上
    result = accounting.reconcile("2026-11", 1.01, path=path)
    assert result["ok"] is True
    assert result["ledger_total_usd"] == pytest.approx(1.00)
    assert result["diff_usd"] == pytest.approx(-0.01)
    # 差额 1.00 远超阈值 → 对不上
    result = accounting.reconcile("2026-11", 2.00, path=path)
    assert result["ok"] is False
    # 12 月：总额对得上，但有一笔成本未知 → 对不上
    _write_row(path, "2026-12-01T10:00:00", "deepseek-flash", "日常", 0.50)
    _write_row(path, "2026-12-01T11:00:00", "deepseek-flash", "日常",
               None, status="unknown_cost")
    result = accounting.reconcile("2026-12", 0.50, path=path)
    assert result["unknown_cost_calls"] == 1
    assert result["ok"] is False


# ---------- 10. 防绕过扫描 ----------

def test_no_bypass_llm_http_calls():
    """全仓扫描：同时出现 requests.post 与 chat/completions 的文件，
    只许是已登记并已接记账的三家（两 provider + 翻译直连）。
    发现新文件先停下来报告，不许擅自放行。"""
    root = Path(__file__).resolve().parent.parent
    allowed = {"providers/deepseek.py", "providers/muse.py", "tools/cli.py"}
    skip_dirs = {".git", ".venv", "tests", "logs", "__pycache__"}
    offenders = set()
    for py_file in root.rglob("*.py"):
        rel = py_file.relative_to(root)
        if any(part in skip_dirs for part in rel.parts):
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        if "requests.post" in text and "chat/completions" in text:
            offenders.add(rel.as_posix())
    assert offenders <= allowed, (
        f"发现未登记的直连大模型调用文件：{sorted(offenders - allowed)}，"
        "先报告，不要擅自放行。")
