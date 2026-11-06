# tests/test_deepseek.py（DeepSeek 接入测试文件）
# 作用：验证 DeepSeek 作为第一任主控的接入：配置、路由、接口、费用、预算。
# 测试分两类，已明确区分：
#   - 模拟测试（Mock Test）：不需要 API Key，不联网，不产生费用，默认运行。
#   - 真实测试（Real API Test）：需要 DeepSeek API Key，可能产生实际费用，
#     默认跳过（skip），只有设置了 DEEPSEEK_API_KEY 并且显式设置
#     RUN_REAL_API_TESTS=1 时才运行。没有用户确认，不要进行产生费用的真实 API 测试。
# 原理：模拟测试用 unittest.mock（模拟工具）假装 requests 返回了固定数据，
#       只验证“我们的代码逻辑对不对”，不验证“模型真的回答了什么”。

import json
import os
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from config import (
    CONTROLLER,
    DEFAULT_MODEL,
    MODEL_CONFIG,
    BudgetTracker,
    get_api_key,
    get_model_status,
)
from main import run_question
from providers.deepseek import DeepSeekProvider
from router import route


# ---------- 模拟测试：配置 ----------
def test_deepseek_config():
    """DeepSeek 配置：已启用，默认模型 deepseek-flash，地址和闲时价格正确。"""
    cfg = MODEL_CONFIG["deepseek"]
    assert cfg["enabled"] is True
    assert cfg["model"] == "deepseek-flash"
    assert cfg["base_url"] == "https://api.deepseek.com"
    assert cfg["price_input_per_1m"] == 0.15
    assert cfg["price_output_per_1m"] == 0.60
    assert cfg["price_cached_input_per_1m"] == 0.003
    # Pro 模型可切换：名字和约 4 倍的价格都配好了。
    assert cfg["model_pro"] == "deepseek-v4-pro"
    assert cfg["price_input_per_1m_pro"] == 0.66
    assert cfg["price_output_per_1m_pro"] == 1.98
    assert cfg["price_cached_input_per_1m_pro"] == 0.022


def test_default_model_and_controller_are_deepseek():
    """默认模型和主控都是 DeepSeek；Muse 保留为启用的工作模型。"""
    assert DEFAULT_MODEL == "deepseek"
    assert CONTROLLER == "deepseek"
    assert MODEL_CONFIG["muse"]["enabled"] is True
    assert MODEL_CONFIG["muse"]["model"] == "muse-spark-1.3"


def test_deepseek_api_key_env_name(monkeypatch):
    """DeepSeek 的 Key 读官方变量名 DEEPSEEK_API_KEY，没有时返回 None。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    assert get_api_key("deepseek") == "sk-test-key"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert get_api_key("deepseek") is None


def test_model_status_includes_deepseek(monkeypatch):
    """模型状态检查：DeepSeek 在状态里，有 Key 时 has_api_key 为 True。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    status = get_model_status()
    assert status["deepseek"]["enabled"] is True
    assert status["deepseek"]["has_api_key"] is True
    assert status["deepseek"]["model"] == "deepseek-flash"


# ---------- 模拟测试：路由 ----------
def test_router_defaults_to_deepseek():
    """路由：普通问答、代码相关、通用任务默认都交给 DeepSeek 主控。"""
    assert route("你好", "question")["provider"] == "deepseek"
    assert route("你好", "question")["model"] == "deepseek-flash"
    assert route("帮我写一段代码", "code")["provider"] == "deepseek"
    assert route("随便聊聊", "general")["provider"] == "deepseek"


# ---------- 模拟测试：DeepSeek 接口（无 Key 时正确报错） ----------
def test_deepseek_provider_no_key_raises():
    """模型接口：无 Key 时应明确报错，并提示填 DEEPSEEK_API_KEY，不返回假成功。"""
    provider = DeepSeekProvider(api_key=None)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        provider.chat("你好")


def test_deepseek_provider_sends_correct_request(tmp_path):
    """模型接口：请求应发到 DeepSeek 官方地址，模型 deepseek-flash，
    输出上限 max_tokens=2048，用模拟的 requests，不联网。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好，我是 DeepSeek。"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post", return_value=fake_response) as mock_post:
        result = provider.chat("你好")
        mock_post.assert_called_once()
    assert result["text"] == "你好，我是 DeepSeek。"
    assert result["model"] == "deepseek-flash"
    called_url = mock_post.call_args.args[0]
    assert called_url == "https://api.deepseek.com/chat/completions"
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["model"] == "deepseek-flash"
    assert sent_payload["max_tokens"] == 2048
    sent_headers = mock_post.call_args.kwargs["headers"]
    assert sent_headers["Authorization"] == "Bearer fake-key-for-test"


def test_deepseek_provider_cost_only_when_usage_returned(tmp_path):
    """模型接口：只有模型返回了 usage 才计算成本；用模拟的 requests，不联网。

    模拟返回：prompt 1000 Token（其中缓存 200）、completion 500 Token，
    成本应为 800/1e6*0.15 + 200/1e6*0.003 + 500/1e6*0.60 = 0.0004206 美元。
    费用应同时记到总数和 deepseek-flash 的小计里。
    """
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 200},
        },
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post", return_value=fake_response):
        result = provider.chat("你好")
    assert result["cost_usd"] == pytest.approx(0.0004206)
    assert tracker.total_cost_usd == pytest.approx(0.0004206)
    assert tracker.cost_for_model("deepseek-flash") == pytest.approx(0.0004206)


def test_deepseek_provider_no_usage_means_cost_unknown(tmp_path):
    """模型接口：模型没返回 usage 时，成本为 None，绝不编造；
    同时要标记成本未知、暂停后续付费调用，不能当成零成本继续调用。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        # 注意：这里故意没有 usage 字段
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post", return_value=fake_response):
        result = provider.chat("你好")
    assert result["cost_usd"] is None
    assert tracker.total_cost_usd == 0.0
    assert tracker.unknown_cost is True
    assert tracker.can_call() is False  # 已暂停，等人工核对


def test_deepseek_provider_pro_model_switch(tmp_path):
    """模型切换：指定 deepseek-v4-pro 时用 Pro 的名字和价格记账。

    模拟返回：prompt 1000、completion 500（无缓存），
    成本应为 1000/1e6*0.66 + 500/1e6*1.98 = 0.00165 美元，
    记在 deepseek-v4-pro 的小计里。
    """
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker, model="deepseek-v4-pro")
    assert provider.model == "deepseek-v4-pro"
    with patch("providers.deepseek.requests.post", return_value=fake_response) as mock_post:
        result = provider.chat("你好")
        assert mock_post.call_args.kwargs["json"]["model"] == "deepseek-v4-pro"
    assert result["cost_usd"] == pytest.approx(0.00165)
    assert tracker.cost_for_model("deepseek-v4-pro") == pytest.approx(0.00165)
    assert tracker.cost_for_model("deepseek-flash") == 0.0


def test_deepseek_provider_unknown_model_raises():
    """模型切换：传一个不存在的 DeepSeek 模型名时应直接报错，不用错的价格记账。"""
    with pytest.raises(ValueError, match="不支持的 DeepSeek 模型"):
        DeepSeekProvider(api_key="fake-key-for-test", model="deepseek-chat")


def test_deepseek_budget_block_happens_before_api_request(tmp_path):
    """预算拦截在请求前：累计到 $5 后，chat() 应抛错，且 requests.post 一次也不能被调用。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(5.0)
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post") as mock_post:
        with pytest.raises(RuntimeError, match="预算上限"):
            provider.chat("你好")
        mock_post.assert_not_called()


def test_deepseek_estimated_cost_blocks_before_api_request(tmp_path):
    """单次预估拦截：累计 $4.998、只剩 $0.002 时，长 prompt 的预估成本
    超过剩余预算，chat() 应抛错且 requests.post 一次也不能被调用。

    DeepSeek 闲时价预估：10000 个汉字估 15000 Token，
    15000/1e6*0.15 + 2048/1e6*0.60 = 0.0034788 美元，超过剩余 $0.002。
    """
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(4.998)
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    long_prompt = "你好" * 5000  # 10000 个汉字
    assert tracker.estimate_cost(long_prompt, provider="deepseek") == pytest.approx(0.0034788)
    with patch("providers.deepseek.requests.post") as mock_post:
        with pytest.raises(RuntimeError, match="预估成本可能超过"):
            provider.chat(long_prompt)
        mock_post.assert_not_called()


# ---------- 模拟测试：按模型记账 ----------
def test_budget_per_model_breakdown(tmp_path):
    """按模型记账：不同模型的费用分别记小计，总数是两者之和，且总数仍是唯一封顶依据。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(0.1, model="deepseek-flash")
    tracker.add_cost(0.2, model="muse-spark-1.3")
    assert tracker.total_cost_usd == pytest.approx(0.3)
    assert tracker.cost_for_model("deepseek-flash") == pytest.approx(0.1)
    assert tracker.cost_for_model("muse-spark-1.3") == pytest.approx(0.2)
    assert tracker.cost_for_model("deepseek-v4-pro") == 0.0
    assert tracker.model_breakdown() == {
        "deepseek-flash": pytest.approx(0.1),
        "muse-spark-1.3": pytest.approx(0.2),
    }
    # 封顶只看总数：总数 0.3，没到 $4 警告线，也没到 $5 上限。
    assert tracker.is_over_warn() is False
    assert tracker.can_call() is True
    # 总数到 $5 就拦截，哪怕单个模型的小计都没到 $5。
    tracker.add_cost(4.7, model="deepseek-flash")
    assert tracker.total_cost_usd == pytest.approx(5.0)
    assert tracker.is_over_budget() is True
    assert tracker.can_call() is False


def test_budget_file_with_breakdown_round_trips(tmp_path):
    """预算文件带按模型小计：保存后再读回来，总数和小计都不丢。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(0.1, model="deepseek-flash")
    tracker.add_cost(0.2, model="muse-spark-1.3")
    saved = json.loads((tmp_path / "budget_usage.json").read_text(encoding="utf-8"))
    assert saved["total_cost_usd"] == pytest.approx(0.3)
    assert saved["cost_by_model"]["deepseek-flash"] == pytest.approx(0.1)
    assert saved["cost_by_model"]["muse-spark-1.3"] == pytest.approx(0.2)
    tracker2 = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    assert tracker2.corrupted is False
    assert tracker2.total_cost_usd == pytest.approx(0.3)
    assert tracker2.cost_for_model("deepseek-flash") == pytest.approx(0.1)


def test_budget_old_file_without_breakdown_loads_fine(tmp_path):
    """旧预算文件没有 cost_by_model 字段：不算损坏，能正常读出累计，小计默认为空。"""
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps(
            {
                "month": datetime.now().strftime("%Y-%m"),
                "total_cost_usd": 2.5,
                "unknown_cost": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.corrupted is False
    assert tracker.can_call() is True
    assert tracker.total_cost_usd == 2.5
    assert tracker.cost_for_model("deepseek-flash") == 0.0
    assert tracker.model_breakdown() == {}


def test_budget_breakdown_resets_on_month_rollover(tmp_path):
    """按模型小计随月份重置：跨月后总数和小计都清零，但未知成本标记不清除。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(0.1, model="deepseek-flash")
    tracker.add_cost(0.2, model="muse-spark-1.3")
    tracker.mark_unknown_cost()
    tracker.month = "2000-01"  # 模拟程序一直运行、跨过自然月
    assert tracker.can_call() is False  # 未知成本未核对，仍暂停
    assert tracker.total_cost_usd == 0.0
    assert tracker.model_breakdown() == {}
    assert tracker.unknown_cost is True  # 跨月不清除


def test_budget_corrupted_breakdown_pauses(tmp_path):
    """按模型小计损坏也暂停：cost_by_model 里有负数时进入损坏状态，原文件不被覆盖。"""
    budget_file = tmp_path / "budget_usage.json"
    raw = json.dumps(
        {
            "month": datetime.now().strftime("%Y-%m"),
            "total_cost_usd": 1.0,
            "unknown_cost": False,
            "cost_by_model": {"deepseek-flash": -0.5},
        },
        ensure_ascii=False,
    )
    budget_file.write_text(raw, encoding="utf-8")
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.corrupted is True
    assert tracker.can_call() is False
    assert budget_file.read_text(encoding="utf-8") == raw


# ---------- 模拟测试：主入口（无 Key 演示模式） ----------
def test_main_no_deepseek_key_demo_mode(monkeypatch, tmp_path):
    """主入口：无 DeepSeek Key 时应为演示模式，提示里写 DEEPSEEK_API_KEY，不调用模型，不产生费用。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    outcome = run_question("你好", budget_tracker=tracker)
    assert outcome["mode"] == "demo"
    assert outcome["result"] is None
    assert outcome["route"]["provider"] == "deepseek"
    assert "DEEPSEEK_API_KEY" in outcome["message"]
    assert "未接通真实 API" in outcome["message"]


def test_main_passes_routing_model_to_deepseek_flash(monkeypatch, tmp_path):
    """主入口传模型：默认路由算出 Flash 时，main.py 传给 Provider 的也是 Flash，
    成本按 Flash 价格算。用模拟的 requests，不联网，不产生费用。

    模拟返回：prompt 1000、completion 500（无缓存），
    成本应为 1000/1e6*0.15 + 500/1e6*0.60 = 0.00045 美元。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch("providers.deepseek.requests.post", return_value=fake_response) as mock_post:
        outcome = run_question("你好", budget_tracker=tracker)
    assert outcome["mode"] == "real"
    assert outcome["result"]["model"] == "deepseek-flash"
    assert mock_post.call_args.kwargs["json"]["model"] == "deepseek-flash"
    assert outcome["result"]["cost_usd"] == pytest.approx(0.00045)


def test_main_passes_routing_model_to_deepseek_pro(monkeypatch, tmp_path):
    """主入口传模型：路由算出 Pro 时，main.py 必须把 Pro 传给 Provider，
    不能丢了 routing["model"]、让 Provider 又按默认 Flash 自己猜；
    成本必须按 Pro 价格算。用模拟的 requests，不联网，不产生费用。

    模拟返回：prompt 1000、completion 500（无缓存），
    成本应为 1000/1e6*0.66 + 500/1e6*1.98 = 0.00165 美元。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setattr(
        "main.route",
        lambda prompt, task_type="question": {
            "task_type": task_type,
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
        },
    )
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch("providers.deepseek.requests.post", return_value=fake_response) as mock_post:
        outcome = run_question("你好", budget_tracker=tracker)
    assert outcome["mode"] == "real"
    assert outcome["result"]["model"] == "deepseek-v4-pro"
    assert mock_post.call_args.kwargs["json"]["model"] == "deepseek-v4-pro"
    assert outcome["result"]["cost_usd"] == pytest.approx(0.00165)
    assert tracker.cost_for_model("deepseek-v4-pro") == pytest.approx(0.00165)
    assert tracker.cost_for_model("deepseek-flash") == 0.0


# ---------- 真实测试：默认跳过，需要 Key + 显式开启 ----------
RUN_REAL_DEEPSEEK = bool(os.environ.get("DEEPSEEK_API_KEY")) and os.environ.get("RUN_REAL_API_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_REAL_DEEPSEEK,
    reason="真实测试需要 DeepSeek API Key 且需显式设置 RUN_REAL_API_TESTS=1，未开启时跳过，不产生费用",
)
def test_real_deepseek_chat():
    """真实测试：用真实 Key 调一次 DeepSeek API（会产生少量费用，需用户确认才运行）。"""
    provider = DeepSeekProvider(api_key=get_api_key("deepseek"))
    result = provider.chat("请用一句话回答：你好吗？")
    assert result["text"]
    assert result["model"] == "deepseek-flash"
