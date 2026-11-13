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
    """路由：普通问答任务应交给 DeepSeek（第一任主控）。"""
    result = route("你好", "question")
    assert result["provider"] == "deepseek"
    assert result["model"] == "deepseek-flash"
    assert result["task_type"] == "question"


def test_router_code_and_general():
    """路由：代码相关和通用任务目前也交给 DeepSeek。"""
    assert route("帮我写一段代码", "code")["provider"] == "deepseek"
    assert route("随便聊聊", "general")["provider"] == "deepseek"


def test_routing_matches_controller():
    """路由一致性检查：所有任务类型路由到的模型必须等于 CONTROLLER。
    注意：这只是配置一致性检查，不代表已经实现动态主控——
    当前路由仍是固定映射，CONTROLLER 只是角色名。"""
    from config import CONTROLLER, MODEL_CONFIG

    for task_type in TASK_TYPES:
        result = route("你好", task_type)
        assert result["provider"] == CONTROLLER
        assert result["model"] == MODEL_CONFIG[CONTROLLER]["model"]


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


def test_muse_provider_default_model_is_spark(tmp_path):
    """Muse 模型参数：不传 model 时默认用 muse-spark-1.3。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="sk-test-key", budget_tracker=tracker)
    assert provider.model == "muse-spark-1.3"


def test_muse_provider_accepts_explicit_model(tmp_path):
    """Muse 模型参数：显式传 muse-spark-1.3 时正常使用，请求里也是这个模型。
    用模拟的 requests，不联网，不产生费用。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="sk-test-key", budget_tracker=tracker, model="muse-spark-1.3")
    assert provider.model == "muse-spark-1.3"
    with patch("providers.muse.requests.post", return_value=fake_response) as mock_post:
        result = provider.chat("你好")
    assert result["model"] == "muse-spark-1.3"
    assert mock_post.call_args.kwargs["json"]["model"] == "muse-spark-1.3"


def test_muse_provider_unknown_model_raises():
    """Muse 模型参数：传一个不存在的模型名时应直接报错，不用错的价格记账。"""
    with pytest.raises(ValueError, match="不支持的 Muse 模型"):
        MuseProvider(api_key="sk-test-key", model="muse-spark-9.9")


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
    """模型接口：模型没返回 usage 时，成本为 None，绝不编造；
    同时要标记成本未知、只打印提醒，不暂停后续调用（BOOS 2026-11-08 定：
    成本未知只提醒、不停）；标记还在，不能当成零成本继续调用，
    要去官方后台核对实际消费。"""
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
    assert tracker.unknown_cost is True
    assert tracker.can_call() is True  # 只提醒，不暂停


def test_budget_tracker_warn_at_full_budget_no_block(tmp_path):
    """预算控制：工具账本到 100%（$5）提醒，永不自动停（BOOS 2026-11-08 定）。

    提醒比例已从 80% 改成 100%：$4 时不提醒、也不拦截；到 $5 时
    提醒一次，之后调用照常进行。提醒只看工具账本（BOOS 2026-11-11 定：
    $5 只算工具的钱，聊天账本不设限）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    assert tracker.can_call() is True
    tracker.add_cost(4.0, ledger="tool")
    assert tracker.is_over_warn() is False  # 未到 $5，不提醒
    assert tracker.can_call() is True
    tracker.add_cost(1.0, ledger="tool")
    assert tracker.is_over_budget() is True
    assert tracker.is_over_warn() is True  # 到 $5，提醒
    assert tracker.can_call() is True  # 只提醒，永不拦截


def test_budget_tracker_monthly_reset(tmp_path):
    """预算每月重置：上个月的文件（累计 $4.5）新建 Tracker 后应重置为 0 且月份为当前月。"""
    now = datetime.now()
    if now.month == 1:
        last_month = f"{now.year - 1}-12"
    else:
        last_month = f"{now.year}-{now.month - 1:02d}"
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps(
            {"month": last_month, "total_cost_usd": 4.5, "unknown_cost": False},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.total_cost_usd == 0.0
    assert tracker.month == now.strftime("%Y-%m")
    assert tracker.can_call() is True


def test_budget_tracker_old_file_without_month_is_corrupted(tmp_path):
    """预算旧文件缺少 month 字段：按关键字段缺失处理，进入损坏暂停状态，
    不能发起付费调用，原文件不被覆盖。"""
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps({"total_cost_usd": 2.5}, ensure_ascii=False),
        encoding="utf-8",
    )
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.corrupted is True
    assert tracker.can_call() is False
    # 原文件不被覆盖，内容仍是只有 total_cost_usd 的旧文件。
    assert json.loads(budget_file.read_text(encoding="utf-8")) == {"total_cost_usd": 2.5}
    # 人工清除也不允许，必须先修好预算文件。
    with pytest.raises(RuntimeError, match="预算文件已损坏"):
        tracker.clear_unknown_cost()


def test_corrupted_budget_file_pauses(tmp_path):
    """预算文件损坏暂停：unknown_cost 为 True 后，文件被改成损坏的 JSON，
    Agent 仍然不能发起付费请求，原文件不被覆盖。"""
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps(
            {
                "month": datetime.now().strftime("%Y-%m"),
                "total_cost_usd": 1.0,
                "unknown_cost": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # 模拟文件损坏：改成不是合法 JSON 的内容。
    budget_file.write_text("{不是合法 JSON", encoding="utf-8")
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.corrupted is True
    assert tracker.unknown_cost is True
    assert tracker.can_call() is False
    assert "预算记录异常" in tracker.status_reason()
    # 原损坏文件不被覆盖。
    assert budget_file.read_text(encoding="utf-8") == "{不是合法 JSON"
    # 付费请求在请求前被拦截，requests.post 一次也不被调用。
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post") as mock_post:
        with pytest.raises(RuntimeError, match="预算记录异常"):
            provider.chat("你好")
        mock_post.assert_not_called()


def test_corrupted_budget_file_missing_fields_pauses(tmp_path):
    """预算文件缺少关键字段也暂停：合法 JSON 但只有 total_cost_usd，
    缺少 month 和 unknown_cost，can_call() 为 False，文件内容不变。"""
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps({"total_cost_usd": 1.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.corrupted is True
    assert tracker.can_call() is False
    assert json.loads(budget_file.read_text(encoding="utf-8")) == {"total_cost_usd": 1.0}


def test_invalid_total_cost_values_pause(tmp_path):
    """非法金额进损坏暂停：NaN、Infinity、负数、布尔值都拒绝。

    注意：JSON 本身不允许 NaN/Infinity 字面量，json.dumps 默认会写出
    NaN/Infinity，Python 的 json.loads 能读回，所以这里用文本直接写，
    覆盖这四种非法值。"""
    import math

    cases = [
        '{"month": "2026-11", "total_cost_usd": NaN, "unknown_cost": false}',
        '{"month": "2026-11", "total_cost_usd": Infinity, "unknown_cost": false}',
        '{"month": "2026-11", "total_cost_usd": -1.5, "unknown_cost": false}',
        '{"month": "2026-11", "total_cost_usd": true, "unknown_cost": false}',
        '{"month": "2026-11", "total_cost_usd": false, "unknown_cost": false}',
    ]
    for raw in cases:
        budget_file = tmp_path / "budget_usage.json"
        budget_file.write_text(raw, encoding="utf-8")
        tracker = BudgetTracker(file_path=budget_file)
        assert tracker.corrupted is True, raw
        assert tracker.can_call() is False, raw
        # 原文件不被覆盖。
        assert budget_file.read_text(encoding="utf-8") == raw
    # 0 是合法值，不应判损坏。
    budget_file = tmp_path / "budget_usage.json"
    budget_file.write_text(
        json.dumps(
            {"month": datetime.now().strftime("%Y-%m"), "total_cost_usd": 0, "unknown_cost": False},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tracker = BudgetTracker(file_path=budget_file)
    assert tracker.corrupted is False
    assert tracker.can_call() is True
    assert tracker.total_cost_usd == 0.0


def test_normal_budget_file_reads_fine(tmp_path):
    """正常预算文件读取不受影响：三个字段齐全合法，累计能正常读出和记账，
    文件也能正常更新。"""
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
    tracker.add_cost(0.5)
    assert tracker.total_cost_usd == 3.0
    saved = json.loads(budget_file.read_text(encoding="utf-8"))
    assert saved["total_cost_usd"] == 3.0
    assert saved["unknown_cost"] is False


def test_budget_no_block_after_reaching_budget(tmp_path):
    """到 $5 只提醒、不拦截：累计到 $5 后，chat() 仍正常发请求、拿到回答。

    BOOS 2026-11-08 定：到 5 美元（100%）提醒，永不自动停。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(5.0)
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response.raise_for_status.return_value = None
    with patch("providers.muse.requests.post", return_value=fake_response) as mock_post:
        result = provider.chat("你好")
        mock_post.assert_called_once()
    assert result["text"] == "你好"


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


def test_budget_tracker_monthly_reset_while_running(tmp_path):
    """预算运行期间跨月重置：程序一直运行时，把月份手动改成上个月，
    再调用 can_call() 时应重置为 0、恢复可调用，文件里的月份也已更新。
    注意：如果之前没有未知成本，unknown_cost 保持 False；如果有未知成本，
    跨月后它必须保留（见 test_unknown_cost_survives_month_rollover），
    不能在这里断言它被清掉。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(4.5)
    assert tracker.total_cost_usd == 4.5
    # 模拟“程序一直运行、跨过自然月”：只改内存里的月份，不重建 Tracker。
    tracker.month = "2000-01"
    assert tracker.can_call() is True
    assert tracker.total_cost_usd == 0.0
    assert tracker.unknown_cost is False  # 本来就没有未知成本，不是被清掉的
    assert tracker.month == datetime.now().strftime("%Y-%m")
    saved = json.loads((tmp_path / "budget_usage.json").read_text(encoding="utf-8"))
    assert saved["month"] == datetime.now().strftime("%Y-%m")
    assert saved["total_cost_usd"] == 0.0


def test_unknown_cost_survives_month_rollover(tmp_path):
    """未知成本跨月保留：上个月产生未知成本，进入新月份后标记仍在，
    但只提醒、不暂停（BOOS 2026-11-08 定：成本未知只提醒、不停）。
    跨月只重置当月累计，不自动清除标记。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(4.5)
    tracker.mark_unknown_cost()
    assert tracker.unknown_cost is True
    assert tracker.can_call() is True  # 只提醒，不暂停
    # 模拟跨月：只改内存里的月份，不重建 Tracker。
    tracker.month = "2000-01"
    assert tracker.can_call() is True  # 仍可调用
    assert tracker.total_cost_usd == 0.0  # 当月累计已重置
    assert tracker.month == datetime.now().strftime("%Y-%m")
    assert tracker.unknown_cost is True  # 未核对的未知成本必须保留
    saved = json.loads((tmp_path / "budget_usage.json").read_text(encoding="utf-8"))
    assert saved["total_cost_usd"] == 0.0
    assert saved["unknown_cost"] is True


def test_unknown_cost_cleared_only_by_manual_confirmation(tmp_path):
    """未知成本只能人工清除：接上一个场景，人工核对并清除标记后，
    unknown_cost 才恢复为 False；跨月本身不会清除它。
    注意：新规矩下成本未知只提醒、不暂停，所以跨月前后 can_call()
    一直是 True；这个测试只验证“标记只能人工清除”。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.mark_unknown_cost()
    tracker.month = "2000-01"
    assert tracker.can_call() is True  # 不暂停
    assert tracker.unknown_cost is True  # 跨月后标记仍在
    # 人工核对后清除标记，才恢复。
    tracker.clear_unknown_cost()
    assert tracker.unknown_cost is False
    assert tracker.can_call() is True


def test_estimated_cost_no_block_before_api_request(tmp_path):
    """单次预估只作参考、不拦截：累计 $4.998、只剩 $0.002 时，长 prompt 的
    预估成本超过剩余预算，chat() 仍正常发请求（BOOS 2026-11-08 定：
    到 $5 提醒、永不自动停，预估也不再拦截）；累计为 0、短 prompt 时
    也能正常走到 post。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(4.998)
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    long_prompt = "你好" * 5000  # 10000 个汉字，预估约 $0.027，超过剩余 $0.002
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response.raise_for_status.return_value = None
    with patch("providers.muse.requests.post", return_value=fake_response) as mock_post:
        result = provider.chat(long_prompt)
        mock_post.assert_called_once()
    assert result["text"] == "你好"

    # 累计为 0、短 prompt 时，预估应通过。
    tracker2 = BudgetTracker(file_path=tmp_path / "budget2.json")
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response.raise_for_status.return_value = None
    provider2 = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker2)
    with patch("providers.muse.requests.post", return_value=fake_response) as mock_post:
        result = provider2.chat("你好")
        mock_post.assert_called_once()
    assert result["text"] == "你好"


def test_chinese_prompt_estimate_not_underestimated(tmp_path):
    """中文长提示词预估不低估：10000 个汉字按旧算法（字符数/4）只估 2500 Token，
    实际可能接近 10000 Token，低估约 4 倍；新算法按每个汉字 1.5 Token，
    应估为 15000 Token，明显大于 2500，且预估成本也大于旧算法的成本。"""
    from config import MAX_OUTPUT_TOKENS

    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    long_chinese = "你好" * 5000  # 10000 个汉字
    assert tracker.estimate_input_tokens(long_chinese) == 15000
    assert tracker.estimate_input_tokens(long_chinese) >= 10000
    assert tracker.estimate_input_tokens(long_chinese) > 2500  # 旧算法的值
    # 新预估成本 = 15000/1e6*1.25 + 4096/1e6*4.25 = 0.036158 美元，
    # 旧算法成本 = 2500/1e6*1.25 + 512/1e6*4.25 = 0.005301 美元。
    assert tracker.estimate_cost(long_chinese) == pytest.approx(0.036158)
    assert tracker.estimate_cost(long_chinese) > 0.005301
    # 短中文 prompt 的预估仍在 $5 预算内，能通过预估检查。
    assert tracker.estimate_input_tokens("你好") == 3  # 2 个汉字 * 1.5
    assert tracker.can_afford_estimated("你好") is True
    assert MAX_OUTPUT_TOKENS == 4096


def test_max_output_tokens_is_sent_in_request(tmp_path):
    """输出上限实际生效：调 chat() 时，请求体里必须带 max_completion_tokens=4096，
    而不是只在预估里假设 4096、实际请求不限输出。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post", return_value=fake_response) as mock_post:
        result = provider.chat("你好")
        mock_post.assert_called_once()
    assert result["text"] == "你好"
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["max_completion_tokens"] == 4096
    assert sent_payload["model"] == "muse-spark-1.3"


def test_unknown_cost_reminds_and_clears(tmp_path):
    """usage 缺失只提醒、不暂停：模拟返回无 usage，chat() 返回 cost None，
    标记未知成本、只打印提醒，之后 can_call() 仍为 True；
    再调一次照常发请求；人工清除标记后 unknown_cost 恢复为 False。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        # 注意：这里故意没有 usage 字段
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post", return_value=fake_response) as mock_post:
        result = provider.chat("你好")
        assert result["cost_usd"] is None
        assert mock_post.call_count == 1
        # 成本未知只提醒、不暂停：再调一次照常发请求，不会被拦截。
        result2 = provider.chat("你好")
        assert result2["cost_usd"] is None
        assert mock_post.call_count == 2
    assert tracker.unknown_cost is True
    assert tracker.can_call() is True
    # 人工核对后清除标记，恢复可调用。
    tracker.clear_unknown_cost()
    assert tracker.unknown_cost is False
    assert tracker.can_call() is True


# ---------- 模拟测试：主入口（无 Key 演示模式） ----------
def test_main_no_key_demo_mode(monkeypatch, tmp_path):
    """主入口：无 Key 时应为演示模式，不调用模型，不产生费用。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    outcome = run_question("你好", budget_tracker=tracker)
    assert outcome["mode"] == "demo"
    assert outcome["result"] is None
    assert "未接通真实 API" in outcome["message"]


def test_main_passes_routing_model_to_muse(monkeypatch, tmp_path):
    """主入口传模型：路由算出 Muse 时，main.py 必须把 muse-spark-1.3 传给 Provider，
    不能丢了 routing["model"]、让 Provider 又按默认模型自己猜。
    用模拟的 requests，不联网，不产生费用。"""
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-key")
    monkeypatch.setattr(
        "main.route",
        lambda prompt, task_type="question": {
            "task_type": task_type,
            "provider": "muse",
            "model": "muse-spark-1.3",
        },
    )
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response.raise_for_status.return_value = None
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch("providers.muse.requests.post", return_value=fake_response) as mock_post:
        outcome = run_question("你好", budget_tracker=tracker)
    assert outcome["mode"] == "real"
    assert outcome["result"]["model"] == "muse-spark-1.3"
    assert mock_post.call_args.kwargs["json"]["model"] == "muse-spark-1.3"


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
