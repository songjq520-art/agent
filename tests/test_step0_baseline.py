# tests/test_step0_baseline.py（第 0 步地基：31 项基线 + 回归测试架子）
# 作用：把 31 项的基线状态钉死在测试里，修好一批、测试就跟着绿一批。
# 基线表是 2026-11-12 对代码逐项核实过的（HEAD 248b9b1），状态说明：
#   能复现 = 代码里问题还在，笔记本上实测也出现过；
#   要实测 = 代码层面无法确认，必须到 BOOS 笔记本上实测才能定；
#   已修好 = 代码里已经没这个问题了（本次基线里是 0 项）。
# 复现测试的规矩：已修好的项测通过；没修的项标 xfail（预期失败），
# 修好那批时把 xfail 拿掉、测通过——套件一直是绿的。
# 注意：全部离线模拟，不调真实 API、不花钱；真调在 /tmp 外面另做。

import re
import time
from pathlib import Path

import pytest

import main as main_module
from config import BudgetTracker
from main import run_question
from tools.agent_loop import AgentLoop
from tools.permissions import _find_shell_code_flag
from tools.timing import Timer
from tools.tool_gateway import ToolGateway
from tasks.task_store import TaskStore

from conftest import make_fake_store


# ---------- 31 项基线表 ----------
# 格式：号：(组， 状态）。组：A=修一次就根除，B=治本靠机制，X=横跨两组。
# 20 号状态是“能复现”（启动从不调迁移），但恢复本身要先在 BOOS 笔记本上
# 确认两个前提（旧 memories.json 在不在、“DeepSeek 已接通”那条真被删了没）。
# 状态是活的：修好一批就把那批改成“已修好”，计数跟着改——
# 第 0 步：44 号已修好；第 1 批：25、29、30 号已修好；
# 第 2 批：12、38、41 号已修好；第 3 批：4、26、34、42、43、46 号已修好；
# 第 4 批：5、24、27、31、32、33、35、36、39、40、47、48、49 号已修好；
# 第 5 批：17、37、45 号已修好；
# 28 号机制已修（date_range 过滤），根因（来源给错还是模型读错）待 BOOS 笔记本实测。

BASELINE = {
    4: ("B", "已修好"),
    5: ("A", "已修好"),
    12: ("X", "已修好"),
    17: ("A", "已修好"),
    20: ("A", "能复现"),
    24: ("B", "已修好"),
    25: ("A", "已修好"),
    26: ("A", "已修好"),
    27: ("B", "已修好"),
    28: ("B", "要实测"),
    29: ("A", "已修好"),
    30: ("A", "已修好"),
    31: ("B", "已修好"),
    32: ("B", "已修好"),
    33: ("A", "已修好"),
    34: ("B", "已修好"),
    35: ("A", "已修好"),
    36: ("A", "已修好"),
    37: ("B", "已修好"),
    38: ("A", "已修好"),
    39: ("A", "已修好"),
    40: ("A", "已修好"),
    41: ("A", "已修好"),
    42: ("B", "已修好"),
    43: ("B", "已修好"),
    44: ("A", "已修好"),
    45: ("A", "已修好"),
    46: ("B", "已修好"),
    47: ("B", "已修好"),
    48: ("A", "已修好"),
    49: ("B", "已修好"),
}

EXPECTED_ITEMS = {
    4, 5, 12, 17, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
}


def test_baseline_covers_all_31_items():
    """基线表必须正好是这 31 项，一项不多、一项不少。"""
    assert set(BASELINE) == EXPECTED_ITEMS
    assert len(BASELINE) == 31
    statuses = [status for _group, status in BASELINE.values()]
    assert statuses.count("能复现") == 1
    assert statuses.count("要实测") == 1
    assert statuses.count("已修好") == 29
    assert all(group in ("A", "B", "X") for group, _ in BASELINE.values())


# ---------- 44 号：发送按钮（已修好，第 0 步修的） ----------


def test_44_send_button_click_calls_send_without_event():
    """44 号：点发送按钮不再把鼠标事件当 presetText 传进去。"""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "addEventListener('click', () => send())" in html
    assert "addEventListener('click', send)" not in html
    # 第二道防线：send() 只认字符串 presetText，不是字符串就读输入框。
    assert "if (typeof presetText !== 'string') presetText = undefined;" in html


# ---------- 计时（第 0 步新加的） ----------


def test_timer_measures_block():
    """Timer：量一段代码花的时间，毫秒数要对得上。"""
    timer = Timer()
    with timer.measure("model_ms"):
        time.sleep(0.05)
    assert timer.as_dict()["model_ms"] >= 40
    assert timer.total_ms() >= 40


def test_timer_accumulates_same_phase():
    """同一截量多次时取累计和（循环里多轮模型调用都记进 model_ms）。"""
    timer = Timer()
    with timer.measure("model_ms"):
        time.sleep(0.02)
    with timer.measure("model_ms"):
        time.sleep(0.02)
    assert timer.as_dict()["model_ms"] >= 30


class _DirectAnswerProvider:
    """直接回答的假模型：不调工具，记下每次收到的消息（供测试检查）。

    每次 new 一个就记成 _LAST：run_question 自己 new provider，测试从
    这里拿它实际用的那只实例、看它收到了什么。”"""

    _LAST = None

    def __init__(self, api_key=None, budget_tracker=None, model=None):
        self.model = model or "deepseek-flash"
        self.tool_calls_log = []
        _DirectAnswerProvider._LAST = self

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        return {"text": "译文", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        self.tool_calls_log.append({"messages": list(messages), "thinking": thinking})
        return {
            "text": "你好，我是十二。",
            "tool_calls": [],
            "reasoning_content": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost_usd": 0.001,
            "model": self.model,
        }

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None):
        return {"text": "你好，我是十二。", "usage": None, "cost_usd": 0.0,
                "model": self.model}


class _StubSearchService:
    """占位搜索服务：本文件测试里模型直接回答，搜索一次也不调。"""


class _SleepyMem0Stub:
    """假记忆库：搜和存都睡一会儿，让计时能测到这两截花的时间。"""

    def search(self, prompt, top_k=5):
        time.sleep(0.01)
        return []

    def note_context(self, memories):
        return None

    def add_turn(self, recent):
        time.sleep(0.02)
        return {}

    def save_turn(self, recent, turn_id=None):
        """26 号后台存：假对象不模拟后台线程，直接当场存（和 add_turn 一样
        睡 20 毫秒，让计时能测到这一截）。"""
        time.sleep(0.02)
        return {"pending": True, "results": [], "queued": True, "turn_id": turn_id}


def test_run_question_returns_timings(tmp_path, monkeypatch):
    """计时：run_question 返回里带 timings，四截（搜记忆、读摘要、调模型、
    存记忆）量到的都记上，没量的不出现。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setitem(main_module.PROVIDER_CLASSES, "deepseek", _DirectAnswerProvider)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    outcome = run_question(
        "你好",
        budget_tracker=tracker,
        mem0_store=_SleepyMem0Stub(),
        tool_gateway=ToolGateway(),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=_StubSearchService(),
    )
    assert outcome["mode"] == "real"
    timings = outcome["timings"]
    assert timings["model_ms"] >= 0
    assert timings["memory_search_ms"] >= 0
    # 存记忆那截假对象睡了 20 毫秒，量出来必须对得上（证明真在量、不是写死的 0）。
    assert timings["memory_save_ms"] >= 10
    # 没传 summary_store，摘要那截没量、就不该出现。
    assert "summary_ms" not in timings


def test_agent_loop_returns_timings(tmp_path):
    """计时：工具循环的返回里带 timings，模型那轮花了多久在里面。”"""
    provider = _DirectAnswerProvider()
    loop = AgentLoop(
        provider,
        gateway=ToolGateway(),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=_StubSearchService(),
    )
    reply = loop.run("你好")
    assert reply["mode"] == "done"
    assert "timings" in reply
    assert reply["timings"]["model_ms"] >= 0


# ---------- 复现测试（没修的项，预期失败；修好那批时拿掉 xfail） ----------


def test_05_shell_abbrev_flag_recognized():
    """5 号：-ep Bypass -Command、-Com、-CommandWithArgs 都要被认作代码标志。
    第 4 批已修好：PowerShell 缩写走 _resolve_powershell_param 解析
    （全写/短写法/无歧义前缀），不再只认全写；白名单 + 先问 BOOS 的
    默认见 tests/test_tools_permissions.py::test_whitelist_default_asks_first。
    """
    assert _find_shell_code_flag(["powershell", "-ep", "Bypass", "-Command", "Get-Date"]) is not None
    assert _find_shell_code_flag(["powershell", "-Com", "Get-Date"]) is not None
    assert _find_shell_code_flag(["powershell", "-CommandWithArgs", "Get-Date"]) is not None
    # 缩写绕过去的硬禁命令，现在要被拦住（Tier 2）。
    from tools.permissions import classify_command

    assert classify_command(["powershell", "-ep", "Bypass", "-Command", "shutdown /s"])[0] == 2
    assert classify_command(["powershell", "-Com", "shutdown /s"])[0] == 2
    # 无害的缩写命令照样只 Tier 1（跑 shell 自由代码，要确认），不误杀。
    assert classify_command(["powershell", "-ep", "Bypass", "-Command", "echo hi"])[0] == 1


def test_29_history_forwarded_to_loop(tmp_path, monkeypatch):
    """29 号：上一轮聊过的“广州今天天气怎么样？”，下一轮只发一个“查”，
    循环里必须能看到上一轮——现在只能看到本轮这一句。”"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setitem(main_module.PROVIDER_CLASSES, "deepseek", _DirectAnswerProvider)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    outcome = run_question(
        "查",
        budget_tracker=tracker,
        messages=[
            {"role": "user", "content": "广州今天天气怎么样？"},
            {"role": "assistant", "content": "要查吗？"},
        ],
        tool_gateway=ToolGateway(),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=_StubSearchService(),
    )
    assert outcome["mode"] == "real"
    # 循环第一轮发给模型的各条消息里，必须有上一轮的“广州”——现在没有。
    provider = _DirectAnswerProvider._LAST
    first_messages = provider.tool_calls_log[0]["messages"]
    assert any("广州" in (m.get("content") or "") for m in first_messages)


def test_30_date_injected_into_loop(tmp_path):
    """30 号：循环发给模型的第一条系统消息里，必须有今天的日期——
    现在全文只有 4 条规矩，没有任何日期。”"""
    provider = _DirectAnswerProvider()
    loop = AgentLoop(
        provider,
        gateway=ToolGateway(),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=_StubSearchService(),
    )
    reply = loop.run("今天几号")
    assert reply["mode"] == "done"
    first_messages = provider.tool_calls_log[0]["messages"]
    system_text = next(m["content"] for m in first_messages if m["role"] == "system")
    assert re.search(r"20\d\d", system_text) is not None
    # 日期必须是对的、不是写死的：和现在上海时间同一天。
    from datetime import datetime
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("Asia/Shanghai"))
    assert f"{today.year} 年 {today.month} 月 {today.day} 日" in system_text


def _install_fake_retraction_judge(store, retract=True):
    """给假 store 装一个假撤回判（34 号测试用）：模型判永远回
    {"retract": retract}。FakeMemory 本来没有 llm 属性，这里是测试
    给它临时装一个，不影响其他测试。"""
    import json
    from types import SimpleNamespace

    class _FakeCompletions:
        def create(self, *args, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {"retract": retract}, ensure_ascii=False
                            )
                        ),
                        finish_reason=None,
                    )
                ]
            )

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeLLM:
        client = SimpleNamespace(chat=_FakeChat())

    store._memory.llm = _FakeLLM()


def test_34_retraction_with_clause_drops_pending(tmp_path):
    """34 号：BOOS 说“我喜欢蓝色”，下一句“不不不，刚刚那句不算”，
    蓝色那条绝不能进库——现在撤回先请模型判意图（假判回 retract=true），
    带后半句的撤回也认得出来，待定区丢掉、已进库的软删除。”"""
    store = make_fake_store(tmp_path)
    _install_fake_retraction_judge(store, retract=True)
    result = store.add_turn(
        [
            {"role": "user", "content": "我喜欢蓝色。"},
            {"role": "assistant", "content": "记下了。"},
        ]
    )
    assert result.get("pending") is True
    result = store.add_turn(
        [
            {"role": "user", "content": "不不不，刚刚那句不算"},
            {"role": "assistant", "content": "好的，我不记了。"},
        ]
    )
    assert result.get("retracted") is True
    assert result.get("dropped") == 1
    assert not any("蓝色" in text for text in store._memory.all_texts())
    store.flush_pending()
    assert not any("蓝色" in text for text in store._memory.all_texts())
