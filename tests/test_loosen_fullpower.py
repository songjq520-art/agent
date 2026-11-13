# tests/test_loosen_fullpower.py（2026-11-13 放宽 + 满血：BOOS 批准）
# 作用：钉住这次改的 7 件事——
#   1. 最后回答那轮开思考（中间调工具那几轮还关着）；
#   2. 规则 6 放宽（先给结论、再给必要的解释）；
#   3. 工具次数 5→8、历史 3 轮→5 轮；
#   4. 工具结果截断 4000→8000；
#   5. 输出上限 2048→4096；
#   6. 明确“记住”时同步直存（不走后台 outbox）；
#   7. 满血路由：难问题走 deepseek-v4-pro，日常问题走 deepseek-flash。
# 全部离线模拟，不调真实 API、不花钱。最后回答开思考的真机验证
# （2026-11-13 实测：不带 tool_calls 的历史能过，带 tool_calls 的
# 历史缺 reasoning_content 会报 400，补空字符串可过）见本文件末尾
# 的三条补丁测试，钉住 chat_messages 的补齐行为。

import copy
import json
from unittest.mock import Mock, patch

import pytest

from conftest import make_fake_store
from router import route
from tools.agent_loop import (
    LOOP_HISTORY_MAX_MESSAGES,
    MAX_TOOL_CALLS,
    TOOL_RESULT_MAX_CHARS,
    AgentLoop,
)
from tools.memory_tool import _run_add_memory, bound_memory_store
from tools.model_router import classify, select_model
from tools.search import SearchOutcome, SearchResult


# ---------- 7. 满血路由：难问题走 Pro，日常问题走 Flash ----------

# 难问题（8 条）：写代码/调试、跨东西比较、要推理、做计划、
# 两个日期、多步、报错分析、代码跑不起来。
HARD_QUESTIONS = [
    ("帮我写一个 Python 函数判断回文字符串", "写代码"),
    ("纳指和标普 11 月 10 日收盘哪个涨得多", "跨东西比较"),
    ("为什么纳指 11 月 11 日涨了这么多", "要推理"),
    ("帮我做一个定投计划，从今天到年底，每天 5 块", "做计划"),
    ("纳指 11 月 9 日和 11 月 10 日收盘各是多少", "两个日期"),
    ("帮我分析一下这个报错", "报错分析"),
    ("先帮我查纳指收盘，再算一下定投到年底一共多少钱", "多步"),
    ("这个代码跑不起来，帮我看看哪里错了", "调试"),
]

# 日常问题（8 条）：打招呼、查一个数、查天气、记住、回忆、问定义、闲聊、问时间。
EASY_QUESTIONS = [
    "你好",
    "纳指昨天收盘多少",
    "今天广州天气怎么样",
    "记住：我喜欢喝美式",
    "你记得我叫什么吗",
    "纳指是什么",
    "我爱吃拉面",
    "现在几点了",
]


def test_router_classifies_hard_questions_as_hard():
    """满血路由：难问题分到 hard（走 Pro）。"""
    assert len(HARD_QUESTIONS) >= 6
    for question, _why in HARD_QUESTIONS:
        difficulty, reason = classify(question)
        assert difficulty == "hard", f"{question!r} 应分 hard，实际 {difficulty}（{reason}）"
        assert reason, "hard 必须带一句原因（写进日志）"


def test_router_classifies_easy_questions_as_easy():
    """满血路由：日常问题分到 easy（走 Flash）。"""
    assert len(EASY_QUESTIONS) >= 6
    for question in EASY_QUESTIONS:
        difficulty, reason = classify(question)
        assert difficulty == "easy", f"{question!r} 应分 easy，实际 {difficulty}（{reason}）"


def test_router_is_deterministic():
    """满血路由：分类是原话的纯函数，同一句话每次都分到同一个模型
    （任务恢复时按原话重算也不会换模型，不触发模型锁定报错）。"""
    for question, _why in HARD_QUESTIONS + [(q, "") for q in EASY_QUESTIONS]:
        first = classify(question)
        second = classify(question)
        assert first == second


def test_select_model_picks_pro_for_hard_and_flash_for_easy():
    """满血路由：hard 选 deepseek-v4-pro，easy 选 deepseek-flash。"""
    model, difficulty, _reason = select_model("帮我写一个 Python 函数判断回文字符串")
    assert difficulty == "hard"
    assert model == "deepseek-v4-pro"
    model, difficulty, _reason = select_model("纳指昨天收盘多少")
    assert difficulty == "easy"
    assert model == "deepseek-flash"


def test_select_model_logs_the_choice(caplog):
    """满血路由：每次选完记一条日志（难度、模型、原因），
    运行输出/日志里能看到每个问题走了哪个模型。"""
    import logging

    with caplog.at_level(logging.INFO, logger="tools.model_router"):
        select_model("帮我写一个 Python 函数判断回文字符串")
    assert "deepseek-v4-pro" in caplog.text
    assert "hard" in caplog.text


def test_route_uses_model_router_for_deepseek():
    """满血路由：route() 对 DeepSeek 按难易选模型（Pro/Flash），
    provider 不变，还是 deepseek。"""
    result = route("帮我写一个 Python 函数判断回文字符串", "question")
    assert result["provider"] == "deepseek"
    assert result["model"] == "deepseek-v4-pro"
    result = route("你好", "question")
    assert result["provider"] == "deepseek"
    assert result["model"] == "deepseek-flash"


def test_route_task_type_does_not_force_hard():
    """满血路由：任务类型不强制难易——“你好”哪怕 task_type 是 code
    也是 easy（只按原话分；test_routing_matches_controller 靠这个）。"""
    assert route("你好", "code")["model"] == "deepseek-flash"
    assert route("你好", "general")["model"] == "deepseek-flash"


def test_model_classifier_is_an_explicit_extension_point():
    """满血路由：model-based 分类器还没实现，调了就明确报错，
    不静默回落到启发式（回落会让调用方误以为用的是模型判断）。"""
    with pytest.raises(NotImplementedError):
        classify("你好", classifier="model")
    with pytest.raises(ValueError):
        classify("你好", classifier="不存在的分类器")


# ---------- 1. 最后回答那轮开思考，中间调工具那几轮还关着 ----------


class _RecordingProvider:
    """记下每次调用思考开关的假模型：chat_with_tools 按剧本回，
    chat_messages 回最终回答。"""

    def __init__(self, model="deepseek-flash", script=None):
        self.model = model
        self.script = list(script or [])
        self.tool_calls_log = []
        self.final_calls_log = []

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        return {"text": "translated", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        self.tool_calls_log.append(
            {"messages": copy.deepcopy(messages), "thinking": thinking}
        )
        if not self.script:
            raise AssertionError("剧本用完了，模型又要调工具——循环该停了")
        return dict(self.script.pop(0))

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None, model=None, max_tokens=None):
        self.final_calls_log.append(
            {"messages": copy.deepcopy(messages), "thinking": thinking, "model": model, "max_tokens": max_tokens}
        )
        return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}


class _FakeSearchService:
    """回写死结果的假搜索服务。"""

    def __init__(self):
        self.translator = None
        self.queries = []

    def search(self, query, kind="normal", max_results=5, cross_check=False,
               force_provider=None, date_range=None):
        self.queries.append(query)
        return SearchOutcome(
            query=query,
            provider="fake",
            status="ok",
            results=[SearchResult(title="广州天气", url="https://example.com/weather",
                                  snippet="晴，23度")],
        )


def _tool_call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _tool_response(text="", tool_calls=None):
    return {
        "text": text,
        "tool_calls": tool_calls or [],
        "reasoning_content": None,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "cost_usd": 0.001,
        "model": "deepseek-flash",
    }


def test_final_answer_enables_thinking_while_tool_rounds_disable_it(tmp_path):
    """最后回答那轮开思考（2026-11-13 放宽），中间调工具那几轮还关着。
    触发最后回答的办法：工具次数用完（max_tool_calls=1，模型一直要调工具）。"""
    from tasks.task_store import TaskStore
    from tools.audit import AuditLog
    from tools.tool_gateway import ToolGateway

    provider = _RecordingProvider(script=[
        _tool_response(tool_calls=[_tool_call("call_1", "search", {"query": "广州天气"})]),
        _tool_response(tool_calls=[_tool_call("call_2", "search", {"query": "广州天气"})]),
    ])
    loop = AgentLoop(
        provider,
        gateway=ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log"))),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=_FakeSearchService(),
        max_tool_calls=1,
    )
    outcome = loop.run("一直查下去的问题，帮我分析一下为什么")
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "最终回答"
    # 中间调工具的轮次思考关着。
    assert len(provider.tool_calls_log) == 2
    assert all(entry["thinking"] == "disabled" for entry in provider.tool_calls_log)
    # 最后回答那轮思考开着（难问题走 c 档深想；提速 2 号后，
    # 日常问题走 a/b 档，思考是关着的——见 test_speed_judgment_r2.py）。
    assert len(provider.final_calls_log) == 1
    assert provider.final_calls_log[0]["thinking"] == "enabled"


# ---------- 2/3/4/5. 放宽后的常量 ----------


def test_loosened_constants():
    """放宽后的常量：工具次数 5→8、历史 3 轮→5 轮（10 条消息）、
    工具结果截断 4000→8000、输出上限 2048→4096。"""
    assert MAX_TOOL_CALLS == 8
    assert LOOP_HISTORY_MAX_MESSAGES == 10
    assert TOOL_RESULT_MAX_CHARS == 8000
    from config import MAX_OUTPUT_TOKENS

    assert MAX_OUTPUT_TOKENS == 4096


# ---------- 6. 明确“记住”时同步直存，不走后台 outbox ----------


def test_add_memory_returns_only_after_entry_is_stored(tmp_path):
    """明确“记住”时同步直存：_run_add_memory 返回时，条目必须已经在
    库里（不用等、不 flush、不等后台线程）；而且根本没经过 outbox
    （直存那条路不建 outbox，正常聊天的后台提取才走 outbox）。"""
    store = make_fake_store(tmp_path)
    with bound_memory_store(store):
        result = _run_add_memory({"text": "我喜欢喝美式。"})
    assert result["ok"] is True
    memory_id = result["data"]["memory_id"]
    # 返回时条目已经在库里：不用等。
    items = store.get_all(include_sensitive=True)
    texts = [item["text"] for item in items]
    assert "我喜欢喝美式。" in texts
    ids = [item["id"] for item in items]
    assert memory_id in ids
    # 直存没经过 outbox：outbox 是正常聊天收尾（save_turn）才懒建的。
    assert store._outbox is None


# ---------- 1 补丁：最后回答开思考时，工具轮的 assistant 消息要带 reasoning_content ----------


def _tool_history():
    """最后回答那轮要发给接口的历史：工具那轮调过 search，assistant
    消息带 tool_calls、没有 reasoning_content（思考关着，模型没产生）。"""
    return [
        {"role": "system", "content": "你是 BOOS 的个人助理十二。"},
        {"role": "user", "content": "广州今天天气怎么样"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_1", "search", {"query": "广州今天天气"})],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "广州今天晴，23度。"},
        {"role": "user", "content": "明天呢"},
    ]


def _fake_chat_response():
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "广州今天晴，23度。"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    fake_response.raise_for_status.return_value = None
    return fake_response


def test_chat_messages_fills_empty_reasoning_for_tool_call_history_when_thinking_enabled():
    """最后回答开思考：带 tool_calls 的 assistant 消息发出去时必须带
    reasoning_content（接口硬性要求，缺了报 400），工具那轮思考关着、
    模型没产生，补空字符串。纯文本 assistant 消息不用补。
    2026-11-13 真机核实：补空字符串接口接受，两轮工具历史两条都要补。"""
    from providers.deepseek import DeepSeekProvider

    provider = DeepSeekProvider(api_key="fake-key-for-test")
    history = _tool_history()
    with patch("providers.deepseek.requests.post",
               return_value=_fake_chat_response()) as mock_post:
        result = provider.chat_messages(history, thinking="enabled")
    assert result["text"] == "广州今天晴，23度。"
    sent = mock_post.call_args.kwargs["json"]["messages"]
    tool_assistant = [m for m in sent if m.get("tool_calls")]
    assert len(tool_assistant) == 1
    assert tool_assistant[0]["reasoning_content"] == ""
    # 纯文本 assistant 消息不用补 reasoning_content。
    plain = [m for m in sent if m.get("role") == "assistant" and not m.get("tool_calls")]
    assert len(plain) == 1
    assert "reasoning_content" not in plain[0]


def test_chat_messages_does_not_fill_reasoning_when_thinking_disabled():
    """思考关着时不补：原来怎么发还怎么发，不改请求体。"""
    from providers.deepseek import DeepSeekProvider

    provider = DeepSeekProvider(api_key="fake-key-for-test")
    history = _tool_history()
    with patch("providers.deepseek.requests.post",
               return_value=_fake_chat_response()) as mock_post:
        provider.chat_messages(history, thinking="disabled")
    sent = mock_post.call_args.kwargs["json"]["messages"]
    tool_assistant = [m for m in sent if m.get("tool_calls")]
    assert len(tool_assistant) == 1
    assert "reasoning_content" not in tool_assistant[0]


def test_chat_messages_does_not_mutate_caller_history():
    """补 reasoning_content 不能改调用方传进来的历史（循环还要复用它）。"""
    from providers.deepseek import DeepSeekProvider

    provider = DeepSeekProvider(api_key="fake-key-for-test")
    history = _tool_history()
    with patch("providers.deepseek.requests.post",
               return_value=_fake_chat_response()):
        provider.chat_messages(history, thinking="enabled")
    tool_assistant = [m for m in history if m.get("tool_calls")]
    assert len(tool_assistant) == 1
    assert "reasoning_content" not in tool_assistant[0]
