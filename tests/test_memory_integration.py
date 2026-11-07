# tests/test_memory_integration.py（长期记忆接入的测试）
# 作用：验证记忆系统 v1 真的接进了对话（不是只存在模块里）。
# 原理：用临时记忆文件（tmp_path）做记忆库，不碰真实的 memory/memories.json；
#       模型调用全部用 Mock 假装，不联网、不产生费用。
# 覆盖：
#   - 关键词拆分（中文两字词、英文整词）
#   - 相关记忆检索（找得到、找不到、没传库、库损坏时都不拦聊天）
#   - run_question 带记忆提问（发给模型的问题里有记忆，历史里仍记原话）
#   - chat_loop 启动时种下前三条记忆、“记住”命令先问再存
#   - 网页 ChatState 接记忆（启动时种下、提问时带上）
# 注意：这些测试不依赖网络、不依赖 API Key；需要真实调用的场景用 Mock。

from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker
from main import (
    _extract_keywords,
    _find_relevant_memories,
    _memory_block,
    chat_loop,
    run_question,
)
from memory.memories import MemoryStore, ensure_first_memories
from web.server import ChatState, ServerConfig, create_server


def _fake_ok_response(text="你好，我是 DeepSeek。", prompt_tokens=100, completion_tokens=50):
    """假装 DeepSeek 返回成功：固定回答 + 固定 Token 用量。"""
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    response.raise_for_status.return_value = None
    return response


def _seeded_store(tmp_path):
    """一个已种下前三条记忆的临时记忆库。"""
    store = MemoryStore(file_path=tmp_path / "memories.json")
    assert ensure_first_memories(store) == 3
    return store


# ---------- 关键词拆分 ----------


def test_extract_keywords_chinese_two_char_windows():
    """中文长词拆成两字窗口：“我的预算是多少”能拆出“预算”，才能命中记忆。"""
    keywords = _extract_keywords("我的预算是多少")
    assert "预算" in keywords
    assert "我的预算是多少" in keywords


def test_extract_keywords_english_whole_words():
    """英文整词搜，不拆两字母：“What is my budget?” 能拆出 budget。"""
    keywords = _extract_keywords("What is my budget?")
    assert "budget" in keywords
    assert "What" in keywords


def test_extract_keywords_empty_and_greeting():
    """空问题没关键词；“你好”只有一个词，搜不到记忆是正常的。"""
    assert _extract_keywords("") == []
    assert _extract_keywords("你好") == ["你好"]


# ---------- 相关记忆检索 ----------


def test_find_relevant_memories_budget_question(tmp_path):
    """问预算找到预算记忆：关键词“预算”命中“每月预算 5 美元……”那条。"""
    store = _seeded_store(tmp_path)
    found = _find_relevant_memories(store, "我的预算是多少")
    assert len(found) == 1
    assert "每月预算 5 美元" in found[0]["text"]


def test_find_relevant_memories_deepseek_question(tmp_path):
    """问 DeepSeek 找到 DeepSeek 记忆。"""
    store = _seeded_store(tmp_path)
    found = _find_relevant_memories(store, "DeepSeek 是哪个模型")
    assert len(found) == 1
    assert "DeepSeek 已接通" in found[0]["text"]


def test_find_relevant_memories_unrelated_question_finds_nothing(tmp_path):
    """问天气找不到记忆：返回空列表，不报错。"""
    store = _seeded_store(tmp_path)
    assert _find_relevant_memories(store, "今天天气怎么样") == []


def test_find_relevant_memories_no_store_returns_empty():
    """没传记忆库时返回空列表，不报错。"""
    assert _find_relevant_memories(None, "我的预算是多少") == []


def test_find_relevant_memories_corrupted_store_returns_empty(tmp_path):
    """记忆文件损坏时返回空列表，不抛错、不拦聊天。"""
    bad_file = tmp_path / "memories.json"
    bad_file.write_text("{ 这不是合法 JSON", encoding="utf-8")
    store = MemoryStore(file_path=bad_file)
    assert store.corrupted is True
    assert _find_relevant_memories(store, "我的预算是多少") == []


def test_memory_block_empty_returns_empty_string():
    """没有记忆时拼出来的提示是空字符串。"""
    assert _memory_block([]) == ""


def test_memory_block_formats_memories(tmp_path):
    """有记忆时拼成“仅供参考”的提示，列出每条记忆。"""
    store = _seeded_store(tmp_path)
    found = _find_relevant_memories(store, "我的预算是多少")
    block = _memory_block(found)
    assert "仅供参考" in block
    assert "每月预算 5 美元" in block


# ---------- run_question 带记忆 ----------


def test_run_question_with_memory_sends_memory_to_model(monkeypatch, tmp_path):
    """带记忆提问：发给模型的问题里有相关记忆（预算那条），且以原问题结尾。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    store = _seeded_store(tmp_path)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch(
        "providers.deepseek.requests.post", return_value=_fake_ok_response()
    ) as mock_post:
        outcome = run_question(
            "我的预算是多少", budget_tracker=tracker, memory_store=store
        )
    assert outcome["mode"] == "real"
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert len(sent) == 1
    assert "每月预算 5 美元" in sent[0]["content"]
    assert sent[0]["content"].endswith("我的预算是多少")


def test_run_question_without_memory_sends_original_prompt(monkeypatch, tmp_path):
    """不带记忆提问：发给模型的就是原问题，不多也不少。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch(
        "providers.deepseek.requests.post", return_value=_fake_ok_response()
    ) as mock_post:
        outcome = run_question("我的预算是多少", budget_tracker=tracker)
    assert outcome["mode"] == "real"
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert sent == [{"role": "user", "content": "我的预算是多少"}]


def test_run_question_with_memory_keeps_history_original(monkeypatch, tmp_path):
    """多轮对话带记忆：发给模型的是“历史 + 带记忆提示的新问题”，
    传入的 history 列表本身不被改动（调用方仍按原话记历史）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    store = _seeded_store(tmp_path)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}]
    with patch(
        "providers.deepseek.requests.post", return_value=_fake_ok_response()
    ) as mock_post:
        outcome = run_question(
            "我的预算是多少",
            budget_tracker=tracker,
            messages=history,
            memory_store=store,
        )
    assert outcome["mode"] == "real"
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert sent[0] == {"role": "user", "content": "你好"}
    assert sent[1] == {"role": "assistant", "content": "你好呀"}
    assert "每月预算 5 美元" in sent[2]["content"]
    assert sent[2]["content"].endswith("我的预算是多少")
    # 传入的 history 没被改动。
    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]


def test_run_question_with_corrupted_memory_still_works(monkeypatch, tmp_path):
    """记忆文件损坏时提问照常：不带记忆、只发原问题，不报错。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    bad_file = tmp_path / "memories.json"
    bad_file.write_text("{ 这不是合法 JSON", encoding="utf-8")
    store = MemoryStore(file_path=bad_file)
    assert store.corrupted is True
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch(
        "providers.deepseek.requests.post", return_value=_fake_ok_response()
    ) as mock_post:
        outcome = run_question(
            "我的预算是多少", budget_tracker=tracker, memory_store=store
        )
    assert outcome["mode"] == "real"
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert sent == [{"role": "user", "content": "我的预算是多少"}]


# ---------- chat_loop 接记忆 ----------


def test_chat_loop_seeds_first_memories_and_loads_them(monkeypatch, tmp_path):
    """连续对话启动时种下前三条记忆；问预算时，发给模型的问题里带上预算记忆。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    store = MemoryStore(file_path=tmp_path / "memories.json")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=["我的预算是多少", "exit"]),
        patch(
            "providers.deepseek.requests.post", return_value=_fake_ok_response()
        ) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker, memory_store=store) == 0
    # 启动时种下了前三条记忆。
    assert len(store.list_all()) == 3
    # 发给模型的问题里带上了预算记忆。
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert "每月预算 5 美元" in sent[0]["content"]
    assert sent[0]["content"].endswith("我的预算是多少")


def test_chat_loop_without_memory_store_behaves_as_before(monkeypatch, tmp_path):
    """不传记忆库时连续对话和原来一样：只发原问题，不带记忆。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=["我的预算是多少", "exit"]),
        patch(
            "providers.deepseek.requests.post", return_value=_fake_ok_response()
        ) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker) == 0
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert sent == [{"role": "user", "content": "我的预算是多少"}]


def test_chat_loop_remember_command_saves_after_confirm(monkeypatch, tmp_path):
    """“记住”命令：先拟一条给 BOOS 看，输入 y 才存；不调用模型。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    store = MemoryStore(file_path=tmp_path / "memories.json")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch(
            "builtins.input",
            side_effect=["记住：我喜欢喝咖啡", "y", "exit"],
        ),
        patch("providers.deepseek.requests.post") as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker, memory_store=store) == 0
        mock_post.assert_not_called()
    # 前三条初始记忆 + 新记的 1 条。
    texts = [m["text"] for m in store.list_all()]
    assert "我喜欢喝咖啡" in texts
    assert len(texts) == 4


def test_chat_loop_remember_command_declined_does_not_save(monkeypatch, tmp_path):
    """“记住”命令输入 n 就不存；不调用模型。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    store = MemoryStore(file_path=tmp_path / "memories.json")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch(
            "builtins.input",
            side_effect=["记住：我喜欢喝咖啡", "n", "exit"],
        ),
        patch("providers.deepseek.requests.post") as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker, memory_store=store) == 0
        mock_post.assert_not_called()
    texts = [m["text"] for m in store.list_all()]
    assert "我喜欢喝咖啡" not in texts
    # 只有启动时种下的前三条。
    assert len(texts) == 3


def test_chat_loop_remembered_memory_is_recalled_later(monkeypatch, tmp_path):
    """刚记住的能马上想起来：记住“我喜欢喝咖啡”后，问“我喜欢喝什么”，
    发给模型的问题里带上这条记忆。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    store = MemoryStore(file_path=tmp_path / "memories.json")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch(
            "builtins.input",
            side_effect=["记住：我喜欢喝咖啡", "y", "我喜欢喝什么", "exit"],
        ),
        patch(
            "providers.deepseek.requests.post", return_value=_fake_ok_response()
        ) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker, memory_store=store) == 0
        assert mock_post.call_count == 1
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert "我喜欢喝咖啡" in sent[0]["content"]
    assert sent[0]["content"].endswith("我喜欢喝什么")


def test_chat_loop_remember_empty_content_does_not_save(monkeypatch, tmp_path):
    """“记住”后面没写内容时：提示用法，不保存，不调用模型。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    store = MemoryStore(file_path=tmp_path / "memories.json")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with (
        patch("builtins.input", side_effect=["记住", "exit"]),
        patch("providers.deepseek.requests.post") as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker, memory_store=store) == 0
        mock_post.assert_not_called()
    # 只有启动时种下的前三条。
    assert len(store.list_all()) == 3


# ---------- 网页 ChatState 接记忆 ----------


def test_chat_state_seeds_first_memories_on_init(tmp_path):
    """网页服务启动时种下前三条记忆（只在空库时写一次）。"""
    store = MemoryStore(file_path=tmp_path / "memories.json")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    state = ChatState(budget_tracker=tracker, memory_store=store)
    assert len(store.list_all()) == 3
    assert state.memory_store is store


def test_chat_state_without_memory_store_has_no_memory(tmp_path):
    """不传记忆库时网页服务不带记忆（保持原来行为，不碰真实记忆文件）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    state = ChatState(budget_tracker=tracker)
    assert state.memory_store is None


def test_chat_state_ask_passes_memory_store_to_run_question(monkeypatch, tmp_path):
    """网页提问时把记忆库传给 run_question；历史里记的仍是原话。"""
    store = _seeded_store(tmp_path)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    state = ChatState(budget_tracker=tracker, memory_store=store)
    seen = {}

    def _fake_run_question(prompt, budget_tracker=None, messages=None, memory_store=None):
        seen["prompt"] = prompt
        seen["memory_store"] = memory_store
        seen["messages"] = list(messages or [])
        return {
            "mode": "real",
            "route": {"model": "deepseek-flash"},
            "result": {
                "model": "deepseek-flash",
                "text": "每月预算 5 美元。",
                "usage": {"total_tokens": 10},
                "cost_usd": 0.00001,
            },
            "message": "真实调用完成。",
            "model_status": {},
        }

    monkeypatch.setattr("web.server.run_question", _fake_run_question)
    reply = state.ask("我的预算是多少")
    assert reply["mode"] == "real"
    assert reply["reply"] == "每月预算 5 美元。"
    # run_question 收到的是这个记忆库，不是 None。
    assert seen["memory_store"] is store
    # 历史里记的是原话。
    assert state.messages == [
        {"role": "user", "content": "我的预算是多少"},
        {"role": "assistant", "content": "每月预算 5 美元。"},
    ]


def test_create_server_passes_memory_store(tmp_path):
    """create_server 把记忆库传给 ChatState。"""
    store = MemoryStore(file_path=tmp_path / "memories.json")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="127.0.0.1", port=0)
    server, state = create_server(
        config=config, budget_tracker=tracker, memory_store=store
    )
    try:
        assert state.memory_store is store
        # 启动时种下了前三条记忆。
        assert len(store.list_all()) == 3
    finally:
        server.server_close()


def test_chat_state_corrupted_memory_does_not_break_ask(monkeypatch, tmp_path):
    """网页记忆文件损坏时提问照常：run_question 收到记忆库，
    检索不到记忆就原样提问，不报错。"""
    bad_file = tmp_path / "memories.json"
    bad_file.write_text("{ 这不是合法 JSON", encoding="utf-8")
    store = MemoryStore(file_path=bad_file)
    assert store.corrupted is True
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    state = ChatState(budget_tracker=tracker, memory_store=store)
    seen = {}

    def _fake_run_question(prompt, budget_tracker=None, messages=None, memory_store=None):
        seen["memory_store"] = memory_store
        return {
            "mode": "real",
            "route": {"model": "deepseek-flash"},
            "result": {
                "model": "deepseek-flash",
                "text": "你好。",
                "usage": {"total_tokens": 10},
                "cost_usd": 0.00001,
            },
            "message": "真实调用完成。",
            "model_status": {},
        }

    monkeypatch.setattr("web.server.run_question", _fake_run_question)
    reply = state.ask("你好")
    assert reply["mode"] == "real"
    assert seen["memory_store"] is store
