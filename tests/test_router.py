# tests/test_router.py（任务路由测试文件）
# 作用：用 pytest（Python 自动化测试工具）验证基础功能。
# 测试分两类，已明确区分：
#   - 模拟测试（Mock Test）：不需要 API Key，不联网，不产生费用，默认运行。
#   - 真实测试（Real API Test）：需要 API Key，可能产生实际费用，默认跳过（skip），
#     只有设置了 MODEL_API_KEY（或 MUSE_API_KEY）并且显式设置 RUN_REAL_API_TESTS=1 时才运行。
#     没有用户确认，不要进行产生费用的真实 API 测试。
# 原理：模拟测试用 unittest.mock（模拟工具）假装 requests 返回了固定数据，
#       只验证“我们的代码逻辑对不对”，不验证“模型真的回答了什么”。

import json
import os
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker, get_api_key, get_model_status, load_env_file
from main import run_question
from providers.muse import MuseProvider
from router import TASK_TYPES, route


# ---------- 模拟测试：配置 ----------
def test_config_can_be_read():
    """配置能否正常读取：Muse 已启用，其他模型未启用，默认预算 $5。"""
    from config import MODEL_CONFIG, MONTHLY_BUDGET_USD

    assert MODEL_CONFIG["muse"]["enabled"] is True
    assert MODEL_CONFIG["muse"]["model"] == "muse-spark-1.3"
    assert MODEL_CONFIG["gpt"]["enabled"] is False
    assert MODEL_CONFIG["gemini"]["enabled"] is False
    assert MODEL_CONFIG["grok"]["enabled"] is False
    assert MONTHLY_BUDGET_USD == 5.0


def test_model_status_no_key(monkeypatch):
    """模型状态检查：没有 Key 时，状态里应明确显示 has_api_key 为 False。"""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    status = get_model_status()
    assert status["muse"]["has_api_key"] is False
    assert status["muse"]["enabled"] is True


def test_api_key_official_name_first(monkeypatch):
    """Key 命名兼容：MODEL_API_KEY（官方名）优先于 MUSE_API_KEY（旧名）。"""
    monkeypatch.setenv("MODEL_API_KEY", "official-key")
    monkeypatch.setenv("MUSE_API_KEY", "old-key")
    assert get_api_key("muse") == "official-key"
    # 只有旧名时也能用。
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert get_api_key("muse") == "old-key"
    # 两个都没有时返回 None。
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    assert get_api_key("muse") is None


def test_load_env_file(tmp_path, monkeypatch):
    """加载 .env：忽略注释和空行，支持 export 前缀和引号值，不覆盖已有环境变量。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 这是一行注释\n"
        "\n"
        "MODEL_API_KEY=key-from-file\n"
        "export MUSE_API_KEY='old-key-from-file'\n"
        'MY_TEST_VALUE="带 空格 的值"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    monkeypatch.delenv("MY_TEST_VALUE", raising=False)
    # 已有的环境变量不应被 .env 文件覆盖。
    monkeypatch.setenv("MUSE_API_KEY", "already-set")
    load_env_file(env_file)
    assert os.environ.get("MODEL_API_KEY") == "key-from-file"
    assert os.environ.get("MUSE_API_KEY") == "already-set"
    assert os.environ.get("MY_TEST_VALUE") == "带 空格 的值"


# ---------- 模拟测试：路由 ----------
def test_router_question():
    """路由：普通问答任务应交给 Muse。"""
    result = route("你好", "question")
    assert result["provider"] == "muse"
    assert result["model"] == "muse-spark-1.3"
    assert result["task_type"] == "question"


def test_router_code_and_general():
    """路由：代码相关和通用任务目前也交给 Muse。"""
    assert route("帮我写一段代码", "code")["provider"] == "muse"
    assert route("随便聊聊", "general")["provider"] == "muse"


def test_router_unknown_type_raises():
    """路由：不支持的任务类型应报错，而不是默默交给某个模型。"""
    with pytest.raises(ValueError):
        route("你好", "not_a_type")


def test_router_empty_text_raises():
    """路由：空任务内容应报错。"""
    with pytest.raises(ValueError):
        route("   ", "question")


def test_task_types_are_expected():
    """任务类型集合应包含 question、code、general 三种。"""
    assert set(TASK_TYPES) == {"question", "code", "general"}


# ---------- 模拟测试：Muse 接口（无 Key 时正确报错） ----------
def test_muse_provider_no_key_raises():
    """模型接口：无 Key 时应明确报错“未配置 API Key，真实连接未完成”，不返回假成功。"""
    provider = MuseProvider(api_key=None)
    with pytest.raises(RuntimeError, match="未配置 API Key"):
        provider.chat("你好")


def test_muse_provider_cost_only_when_usage_returned(tmp_path):
    """模型接口：只有模型返回了 usage 才计算成本；用模拟的 requests，不联网。

    模拟返回：prompt 100 Token、completion 50 Token，
    成本应为 100/1e6*1.25 + 50/1e6*4.25 = 0.0003375 美元。
    """
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好，我是 Muse。"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post", return_value=fake_response):
        result = provider.chat("你好")
    assert result["text"] == "你好，我是 Muse。"
    assert result["usage"]["total_tokens"] == 150
    assert result["cost_usd"] == pytest.approx(0.0003375)
    assert tracker.total_cost_usd == pytest.approx(0.0003375)


def test_muse_provider_no_usage_means_cost_unknown(tmp_path):
    """模型接口：模型没返回 usage 时，成本为 None，绝不编造。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        # 注意：这里故意没有 usage 字段
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post", return_value=fake_response):
        result = provider.chat("你好")
    assert result["cost_usd"] is None
    assert tracker.total_cost_usd == 0.0


def test_budget_tracker_warn_and_block(tmp_path):
    """预算控制：达 80%（$4）警告，达 100%（$5）拒绝后续付费调用。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    assert tracker.can_call() is True
    tracker.add_cost(4.0)
    assert tracker.is_over_warn() is True
    assert tracker.can_call() is True  # 80% 只警告，不拦截
    tracker.add_cost(1.0)
    assert tracker.is_over_budget() is True
    assert tracker.can_call() is False  # 100% 拦截


def test_budget_tracker_monthly_reset(tmp_path):
    """预算每月重置：上个月的文件（累计 $4.5）新建 Tracker 后应重置为 0 且月份为当前月。"""
    now = datetime.now()
    if now.month == 1:
        last_month = f"{now.year - 1}-12"
    else:
        last_month = f"{now.year}-{now.month - 1:02d}"
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps({"month": last_month, "total_cost_usd": 4.5}, ensure_ascii=False),
        encoding="utf-8",
    )
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.total_cost_usd == 0.0
    assert tracker.month == now.strftime("%Y-%m")
    assert tracker.can_call() is True


def test_budget_tracker_old_file_without_month_keeps_total(tmp_path):
    """预算旧文件兼容：没有 month 字段的旧文件，保留原累计值，只补上当前月份。"""
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps({"total_cost_usd": 2.5}, ensure_ascii=False),
        encoding="utf-8",
    )
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.total_cost_usd == 2.5
    assert tracker.month == datetime.now().strftime("%Y-%m")


def test_budget_block_happens_before_api_request(tmp_path):
    """预算拦截在请求前：累计到 $5 后，chat() 应抛错，且 requests.post 一次也不能被调用。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(5.0)
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post") as mock_post:
        with pytest.raises(RuntimeError, match="预算上限"):
            provider.chat("你好")
        mock_post.assert_not_called()


def test_muse_provider_cached_cost(tmp_path):
    """成本区分缓存 Token：prompt 100、缓存 80、completion 50 时，
    成本应为 20/1e6*1.25 + 80/1e6*0.15 + 50/1e6*4.25 = 0.0002495 美元。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post", return_value=fake_response):
        result = provider.chat("你好")
    assert result["cost_usd"] == pytest.approx(0.0002495)
    assert tracker.total_cost_usd == pytest.approx(0.0002495)


# ---------- 模拟测试：主入口（无 Key 演示模式） ----------
def test_main_no_key_demo_mode(monkeypatch, tmp_path):
    """主入口：无 Key 时应为演示模式，不调用模型，不产生费用。"""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    outcome = run_question("你好", budget_tracker=tracker)
    assert outcome["mode"] == "demo"
    assert outcome["result"] is None
    assert "未接通真实 API" in outcome["message"]


# ---------- 真实测试：默认跳过，需要 Key + 显式开启 ----------
RUN_REAL = bool(os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY")) and os.environ.get("RUN_REAL_API_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_REAL,
    reason="真实测试需要 API Key 且需显式设置 RUN_REAL_API_TESTS=1，未开启时跳过，不产生费用",
)
def test_real_muse_chat():
    """真实测试：用真实 Key 调一次 Muse API（会产生少量费用，需用户确认才运行）。"""
    provider = MuseProvider(api_key=get_api_key("muse"))
    result = provider.chat("请用一句话回答：你好吗？")
    assert result["text"]
    assert result["model"] == "muse-spark-1.3"
