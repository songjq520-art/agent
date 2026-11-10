# tests/test_conversation.py（连续对话能力测试文件）
# 作用：验证 Agent 的连续对话升级：多轮历史、预算覆盖完整历史、
#       共用 BudgetTracker、退出命令、错误处理。
# 测试全部是模拟测试（Mock Test）：不需要 API Key，不联网，不产生费用。
# 原理：用 unittest.mock 假装 requests 返回固定数据，并假装用户输入，
#       只验证“我们的代码逻辑对不对”，不验证“模型真的回答了什么”。

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import requests

from config import BudgetTracker
from conftest import make_fake_store
from main import chat_loop, is_exit_command, main, run_question
from providers.deepseek import DeepSeekProvider
from providers.muse import MuseProvider


def _fake_ok_response(text="你好，我是 DeepSeek。", prompt_tokens=100, completion_tokens=50):
    """构造一个模拟的成功 API 返回（带 usage），供 tests 复用。"""
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    fake_response.raise_for_status.return_value = None
    return fake_response


# ---------- 旧版 chat(prompt) 仍然正常工作 ----------
def test_deepseek_chat_still_sends_single_user_message(tmp_path):
    """旧版 chat(prompt) 兼容：只发一条 user 消息，行为与原来一致。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post:
        result = provider.chat("你好")
    assert result["text"] == "你好，我是 DeepSeek。"
    sent = mock_post.call_args.kwargs["json"]
    assert sent["messages"] == [{"role": "user", "content": "你好"}]
    assert sent["model"] == "deepseek-flash"


def test_muse_chat_still_sends_single_user_message(tmp_path):
    """旧版 chat(prompt) 兼容（Muse）：只发一条 user 消息，行为与原来一致。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post", return_value=_fake_ok_response()) as mock_post:
        result = provider.chat("你好")
    assert result["text"] == "你好，我是 DeepSeek。"
    sent = mock_post.call_args.kwargs["json"]
    assert sent["messages"] == [{"role": "user", "content": "你好"}]
    assert sent["model"] == "muse-spark-1.3"


# ---------- 多轮对话确实发送完整历史，顺序正确 ----------
def test_deepseek_chat_messages_sends_full_history_in_order(tmp_path):
    """多轮对话：chat_messages 把完整历史原样发给 API，user / assistant 顺序正确。"""
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是 DeepSeek。"},
        {"role": "user", "content": "今天天气怎么样？"},
    ]
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post:
        result = provider.chat_messages(history)
    assert result["text"] == "你好，我是 DeepSeek。"
    sent = mock_post.call_args.kwargs["json"]
    assert sent["messages"] == history
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
    assert sent["model"] == "deepseek-flash"


def test_muse_chat_messages_sends_full_history_in_order(tmp_path):
    """多轮对话（Muse）：chat_messages 把完整历史原样发给 API，顺序正确。"""
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好。"},
        {"role": "user", "content": "帮我写段代码"},
    ]
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = MuseProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.muse.requests.post", return_value=_fake_ok_response()) as mock_post:
        result = provider.chat_messages(history)
    assert result["text"] == "你好，我是 DeepSeek。"
    sent = mock_post.call_args.kwargs["json"]
    assert sent["messages"] == history
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]


def test_chat_messages_rejects_bad_history(tmp_path):
    """消息历史校验：空列表或格式错误的项应直接报错，不发请求。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    with patch("providers.deepseek.requests.post") as mock_post:
        with pytest.raises(ValueError, match="消息历史不能为空"):
            provider.chat_messages([])
        with pytest.raises(ValueError, match="消息历史格式错误"):
            provider.chat_messages([{"role": "user"}])
        with pytest.raises(ValueError, match="消息历史格式错误"):
            provider.chat_messages(["你好"])
        mock_post.assert_not_called()


# ---------- 预算预估覆盖完整历史 ----------
def test_deepseek_estimate_covers_full_history_not_just_latest(tmp_path):
    """预算预估覆盖完整历史：历史很长、最新问题很短时，预估仍按完整历史算。

    累计 $4.998、只剩 $0.002。历史里第一轮有 10000 个汉字（约 15000 Token），
    完整历史预估约 $0.00348，超过剩余 $0.002；如果只算最新问题
    “你好吗”（3 个汉字），预估只有约 $0.00123。
    新规矩（BOOS 2026-11-08 定）：预估只作参考、不再拦截——所以这里
    只验证预估数本身算的是完整历史，并验证 chat_messages 照常发请求。
    """
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(4.998)
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    history = [
        {"role": "user", "content": "你好" * 5000},  # 10000 个汉字
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "你好吗"},  # 最新问题很短
    ]
    # 只算最新问题时预估约 $0.00123，远小于剩余 $0.002，不会拦截。
    assert tracker.estimate_cost("你好吗", provider="deepseek") < 0.002
    # 算完整历史时预估约 $0.00348，超过剩余 $0.002——证明预估算的是
    # 完整历史，不是最新问题（这里把历史内容拼起来算，和旧版
    # chat_messages 内部的算法一致）。
    history_text = "\n".join(item["content"] for item in history)
    assert tracker.estimate_cost(history_text, provider="deepseek") == pytest.approx(0.00348)
    # 预估超了也不拦截：chat_messages 照常发请求。
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post:
        result = provider.chat_messages(history)
        mock_post.assert_called_once()
    assert result["text"] == "你好，我是 DeepSeek。"


def test_deepseek_chat_messages_no_block_after_reaching_budget(tmp_path):
    """到 $5 只提醒、不拦截（多轮）：累计到 $5 后，chat_messages 仍正常
    发请求、拿到回答（BOOS 2026-11-08 定：到 5 美元提醒、永不自动停）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    tracker.add_cost(5.0)
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好。"},
        {"role": "user", "content": "再见"},
    ]
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post:
        result = provider.chat_messages(history)
        mock_post.assert_called_once()
    assert result["text"] == "你好，我是 DeepSeek。"


# ---------- API 错误不会伪造成功回答 ----------
def test_deepseek_chat_messages_api_error_raises_no_fake_answer(tmp_path):
    """API 报错时：chat_messages 应把错误抛出去，不返回假回答。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    provider = DeepSeekProvider(api_key="fake-key-for-test", budget_tracker=tracker)
    history = [{"role": "user", "content": "你好"}]
    with patch(
        "providers.deepseek.requests.post",
        side_effect=requests.RequestException("网络断了"),
    ) as mock_post:
        # requests 异常会被包成 RuntimeError 抛出，不会返回假成功。
        with pytest.raises(RuntimeError, match="DeepSeek API 调用失败"):
            provider.chat_messages(history)
        mock_post.assert_called_once()


# ---------- 多轮共用同一个 BudgetTracker，费用统计正常 ----------
def test_run_question_multi_turn_shares_one_tracker_and_grows_history(monkeypatch, tmp_path):
    """多轮共用一个 BudgetTracker：两轮 run_question 用同一个 tracker，
    费用累计进同一个 tracker；第二轮发送的消息包含第一轮的 user + assistant。

    每轮模拟返回 prompt 100、completion 50，成本为
    100/1e6*0.15 + 50/1e6*0.60 = 0.000045 美元。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    messages = []
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post:
        # 第一轮：历史为空，只发本轮问题。
        outcome1 = run_question("你好", budget_tracker=tracker, messages=messages)
        assert outcome1["mode"] == "real"
        assert mock_post.call_args.kwargs["json"]["messages"] == [
            {"role": "user", "content": "你好"}
        ]
        # 成功后调用方把本轮 user + assistant 追加进历史（与 chat_loop 一致）。
        messages.append({"role": "user", "content": "你好"})
        messages.append({"role": "assistant", "content": outcome1["result"]["text"]})
        # 第二轮：发送完整历史 + 新问题。
        outcome2 = run_question("今天天气怎么样", budget_tracker=tracker, messages=messages)
        assert outcome2["mode"] == "real"
        assert mock_post.call_args.kwargs["json"]["messages"] == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是 DeepSeek。"},
            {"role": "user", "content": "今天天气怎么样"},
        ]
        assert mock_post.call_count == 2
    # 两轮费用都记进同一个 tracker：总数和小计都是 0.00009 美元。
    assert tracker.total_cost_usd == pytest.approx(0.00009)
    assert tracker.cost_for_model("deepseek-flash") == pytest.approx(0.00009)
    # 路由仍是原来那套：都交给 DeepSeek Flash。
    assert outcome1["route"]["provider"] == "deepseek"
    assert outcome1["route"]["model"] == "deepseek-flash"
    assert outcome2["route"]["provider"] == "deepseek"


def test_run_question_with_messages_does_not_mutate_history(monkeypatch, tmp_path):
    """run_question 不修改传入的 messages 列表：历史只由调用方在成功后追加。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    messages = [{"role": "user", "content": "你好"}]
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()):
        run_question("再见", budget_tracker=tracker, messages=messages)
    assert messages == [{"role": "user", "content": "你好"}]


def test_multi_turn_unknown_cost_reminds_only(monkeypatch, tmp_path):
    """多轮未知成本只提醒、不暂停：第二轮模型没返回 usage 时，成本为 None，
    标记未知成本、只打印提醒，之后仍可调用（BOOS 2026-11-08 定：
    成本未知只提醒、不停），不能当成零成本继续——标记还在，
    要去官方后台核对。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    messages = []
    first = _fake_ok_response()
    second_no_usage = Mock()
    second_no_usage.json.return_value = {
        "choices": [{"message": {"content": "你好"}}],
        # 故意没有 usage 字段
    }
    second_no_usage.raise_for_status.return_value = None
    with patch(
        "providers.deepseek.requests.post", side_effect=[first, second_no_usage]
    ):
        outcome1 = run_question("你好", budget_tracker=tracker, messages=messages)
        assert outcome1["result"]["cost_usd"] == pytest.approx(0.000045)
        messages.append({"role": "user", "content": "你好"})
        messages.append({"role": "assistant", "content": outcome1["result"]["text"]})
        outcome2 = run_question("再见", budget_tracker=tracker, messages=messages)
        assert outcome2["result"]["cost_usd"] is None
    assert tracker.unknown_cost is True
    assert tracker.can_call() is True  # 只提醒，不暂停


# ---------- 退出命令不会调用 API ----------
def test_is_exit_command():
    """退出命令判断：exit / quit / 退出，不区分大小写，前后空格忽略。"""
    assert is_exit_command("exit") is True
    assert is_exit_command("EXIT") is True
    assert is_exit_command("  Exit  ") is True
    assert is_exit_command("quit") is True
    assert is_exit_command("QUIT") is True
    assert is_exit_command("退出") is True
    assert is_exit_command(" 退出 ") is True
    assert is_exit_command("你好") is False
    assert is_exit_command("") is False


def test_chat_loop_exit_immediately_calls_no_api(monkeypatch, tmp_path):
    """退出命令不调 API：刚进循环就输入 exit，直接正常结束，requests 一次也不调用。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=["exit"]),
        patch("providers.deepseek.requests.post") as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
        mock_post.assert_not_called()


@pytest.mark.parametrize("cmd", ["quit", "退出", "EXIT", "  quit  "])
def test_chat_loop_all_exit_commands_call_no_api(monkeypatch, tmp_path, cmd):
    """退出命令不调 API：quit / 退出 / 大写 EXIT / 带空格的 quit 都一样。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=[cmd]),
        patch("providers.deepseek.requests.post") as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
        mock_post.assert_not_called()


def test_chat_loop_two_turns_then_exit_sends_growing_history(monkeypatch, tmp_path):
    """连续对话：两轮对话后输入 exit 结束；第二轮发送的消息包含第一轮的完整历史。

    每轮模拟返回 prompt 100、completion 50，成本 0.000045 美元，
    两轮共 0.00009 美元，记进同一个 tracker。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=["你好", "今天天气怎么样", "exit"]),
        patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
        assert mock_post.call_count == 2
    # 第一轮只发本轮问题。
    first_sent = mock_post.call_args_list[0].kwargs["json"]["messages"]
    assert first_sent == [{"role": "user", "content": "你好"}]
    # 第二轮发完整历史 + 新问题，顺序为 user、assistant、user。
    second_sent = mock_post.call_args_list[1].kwargs["json"]["messages"]
    assert second_sent == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是 DeepSeek。"},
        {"role": "user", "content": "今天天气怎么样"},
    ]
    assert tracker.total_cost_usd == pytest.approx(0.00009)


def test_chat_loop_api_error_does_not_fake_answer_and_continues(monkeypatch, tmp_path):
    """API 报错不伪造回答：第一轮调用失败时，循环不崩溃、不追加假历史，
    继续等下一句；第二轮成功后正常结束。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    failing = Mock()
    failing.raise_for_status.side_effect = requests.HTTPError("网络断了")
    with (
        patch("builtins.input", side_effect=["你好", "再见", "exit"]),
        patch(
            "providers.deepseek.requests.post",
            side_effect=[failing, _fake_ok_response()],
        ) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
        assert mock_post.call_count == 2
    # 第一轮失败，没追加历史；第二轮只发本轮问题（没有第一轮的假历史）。
    second_sent = mock_post.call_args_list[1].kwargs["json"]["messages"]
    assert second_sent == [{"role": "user", "content": "再见"}]
    # 只有第二轮成功记了账。
    assert tracker.total_cost_usd == pytest.approx(0.000045)


def test_chat_loop_empty_input_does_not_call_api(monkeypatch, tmp_path):
    """空输入不调 API：直接按回车时跳过，继续等下一句。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=["", "  ", "exit"]),
        patch("providers.deepseek.requests.post") as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
        mock_post.assert_not_called()


def test_chat_loop_demo_mode_does_not_call_api(monkeypatch, tmp_path):
    """演示模式不调 API：无 Key 时循环只做演示，不调用模型，可正常退出。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=["你好", "exit"]),
        patch("providers.deepseek.requests.post") as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
        mock_post.assert_not_called()


# ---------- main() 入口分发 ----------
def test_main_with_arg_is_single_question(monkeypatch, tmp_path):
    """main() 带参数时为单次问答：python main.py "你好" 只问一次，不进入循环。

    注意：新版 main() 的命令行入口不带记忆库（B 号工人的设计：
    命令行默认不加载 mem0，记忆只在网页服务里启用，或“记住……”时
    才懒加载；main() 的 memory_store 参数只是兼容旧调用，传了也不用）。
    所以这里不传 store，只验证单次问答的分发。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    # 摘要文件指到临时目录：单次问答提问前会读摘要，别让它读到真实文件。
    monkeypatch.setenv("AGENT_SUMMARY_FILE", str(tmp_path / "conversation_summary.json"))
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post:
        assert main(["你好"]) == 0
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["json"]["messages"] == [
            {"role": "user", "content": "你好"}
        ]


def test_main_without_arg_enters_chat_loop(monkeypatch, tmp_path):
    """main() 不带参数时进入连续对话：python main.py 后可多轮对话，exit 结束。

    同上：命令行入口不带记忆库，这里不传 store，只验证进入循环的分发。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    # 摘要文件指到临时目录：退出时总结会调一次模型、写一次文件，
    # 别让它碰真实的 memory/conversation_summary.json。
    monkeypatch.setenv("AGENT_SUMMARY_FILE", str(tmp_path / "conversation_summary.json"))
    with (
        patch("builtins.input", side_effect=["你好", "exit"]),
        patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post,
    ):
        assert main([]) == 0
        # 两次调用：一次是“你好”的问答，一次是退出时总结对话。
        assert mock_post.call_count == 2
        first_sent = mock_post.call_args_list[0].kwargs["json"]["messages"]
        assert first_sent == [{"role": "user", "content": "你好"}]
        second_sent = mock_post.call_args_list[1].kwargs["json"]["messages"]
        assert "对话记录员" in second_sent[0]["content"]
    # 退出时总结的那段对话记进了最近列表（标题取第一句用户话）。
    saved = json.loads((tmp_path / "conversation_summary.json").read_text(encoding="utf-8"))
    assert len(saved["conversations"]) == 1
    assert saved["conversations"][0]["title"] == "你好"


# ---------- 第二阶段补充验收：会话隔离 ----------
def test_session_isolation_new_chat_loop_starts_with_empty_history(monkeypatch, tmp_path):
    """会话隔离：每次启动新的 chat_loop() 都会创建独立的空 messages 历史，
    上一轮会话的 user/assistant 消息不会传入下一轮会话。

    第一轮会话问“你好”，第二轮会话问“再见”。第二轮发送的消息应只有
    本轮的“再见”，不含第一轮的“你好”和模型的回答。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    # 两个会话各用独立的预算文件，只隔离消息历史这一个变量。
    tracker1 = BudgetTracker(file_path=tmp_path / "budget1.json")
    tracker2 = BudgetTracker(file_path=tmp_path / "budget2.json")
    with patch(
        "providers.deepseek.requests.post", return_value=_fake_ok_response()
    ) as mock_post:
        # 第一轮会话：问“你好”，然后 exit。
        with patch("builtins.input", side_effect=["你好", "exit"]):
            assert chat_loop(budget_tracker=tracker1) == 0
        # 第二轮会话（全新启动）：问“再见”，然后 exit。
        with patch("builtins.input", side_effect=["再见", "exit"]):
            assert chat_loop(budget_tracker=tracker2) == 0
        assert mock_post.call_count == 2
    # 第一轮只发本轮问题。
    first_sent = mock_post.call_args_list[0].kwargs["json"]["messages"]
    assert first_sent == [{"role": "user", "content": "你好"}]
    # 第二轮也只发本轮问题，不含第一轮的任何消息，证明历史是独立的空历史。
    second_sent = mock_post.call_args_list[1].kwargs["json"]["messages"]
    assert second_sent == [{"role": "user", "content": "再见"}]
    assert all(m["content"] != "你好" for m in second_sent)


# ---------- 第二阶段补充验收：预算连续累计 ----------
def test_budget_accumulates_across_turns_in_one_chat_loop(monkeypatch, tmp_path):
    """预算连续累计：同一个 chat_loop() 中的多轮调用共用同一个 BudgetTracker，
    连续两轮 API 调用产生的成本会累计到同一个预算记录中，
    且预算文件保存的累计金额正确。

    每轮模拟返回 prompt 100、completion 50，成本为
    100/1e6*0.15 + 50/1e6*0.60 = 0.000045 美元，两轮共 0.00009 美元。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    budget_file = tmp_path / "budget_usage.json"
    tracker = BudgetTracker(file_path=budget_file)
    with (
        patch("builtins.input", side_effect=["你好", "再见", "exit"]),
        patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
        assert mock_post.call_count == 2
    # 内存中的 tracker：两轮成本累计到同一个记录。
    assert tracker.total_cost_usd == pytest.approx(0.00009)
    assert tracker.cost_for_model("deepseek-flash") == pytest.approx(0.00009)
    # 预算文件：保存的累计金额正确。
    saved = json.loads(budget_file.read_text(encoding="utf-8"))
    assert saved["total_cost_usd"] == pytest.approx(0.00009)
    assert saved["cost_by_model"]["deepseek-flash"] == pytest.approx(0.00009)
    assert saved["month"] == datetime.now().strftime("%Y-%m")
    assert saved["unknown_cost"] is False
    # 重新读文件的新 tracker：累计金额持久化正确，不是只存在内存里。
    tracker2 = BudgetTracker(file_path=budget_file)
    assert tracker2.corrupted is False
    assert tracker2.total_cost_usd == pytest.approx(0.00009)
    assert tracker2.cost_for_model("deepseek-flash") == pytest.approx(0.00009)


# ---------- 连续对话接 mem0（新接口，见 tests/conftest.py 的 make_fake_store） ----------


def test_chat_loop_with_mem0_store_saves_turn_after_answer(monkeypatch, tmp_path):
    """连续对话 + mem0：成功回答后把本轮对话存进长期记忆（add_turn），
    FakeMemory 能从 user 消息里提出“我叫小宋”这条事实。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    with (
        patch("builtins.input", side_effect=["我叫小宋。", "exit"]),
        patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker, mem0_store=store) == 0
        mock_post.assert_called_once()
    texts = [item["text"] for item in store.get_all()]
    assert any("小宋" in text for text in texts)


def test_chat_loop_remember_command_saves_text_via_mem0(monkeypatch, tmp_path):
    """连续对话“记住：内容”：懒加载 get_mem0_store 后调 add_memory_text，
    原文存进长期记忆，不用再确认（mem0 自动存，BOOS 已同意）。

    输入里多给一个 "exit"：旧版 main.py 的“记住”还要 input() 问一次 y/n，
    会把第二个 "exit" 当确认回答吃掉；新版不问，直接存，第二个 "exit" 退出循环。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    # main.py 里是 `from memory import mem0_store`（模块），再调模块上的
    # get_mem0_store() 懒加载；所以补丁要打在模块属性上，不是 main.get_mem0_store。
    monkeypatch.setattr(
        "main.mem0_store.get_mem0_store", lambda budget_tracker=None: store
    )
    with patch("builtins.input", side_effect=["记住：我喜欢喝咖啡。", "exit", "exit"]):
        assert chat_loop(budget_tracker=tracker) == 0
    texts = [item["text"] for item in store.get_all()]
    assert "我喜欢喝咖啡。" in texts
