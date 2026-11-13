# tests/test_search_first_51.py（51 号：先查后说落到程序上）
# 作用：验证 tools/agent_loop.py 的 51 号门禁——行情数据问题没搜过
#       （实体，日期）这一对就不能直接答：第一次提醒模型去搜，
#       提醒过还直接答就如实回不知道。
# 全部离线模拟：ScriptedProvider 按剧本回话，FakeSearchService 回写死的
#       搜索结果，不调真实 API、不花钱。
# 关键回归（2026-11-12 实测）：BOOS 连问三个收盘问题（纳指 11/10、标普 11/8、
#       纳指 11/9）都搜了也答了，第四个“标普的 11.10”却没搜，直接说
#       “没有查到可靠数据，不能给出具体数字”，还反问要不要联网查——
#       标普 11/10 明明有（7,520.93）。短问句被当成不用搜的接续问。

import copy
import json

import pytest

from tasks.task_store import TaskStore
from tools.agent_loop import AgentLoop
from tools.audit import AuditLog
from tools.date_numbers import (
    extract_market_entity,
    is_market_data_question,
    resolve_requested_date,
    search_covers_pair,
)
from tools.search import SearchOutcome, SearchResult
from tools.tool_gateway import ToolGateway


def _tool_call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _tool_response(text="", tool_calls=None, cost=0.001):
    return {
        "text": text,
        "tool_calls": tool_calls or [],
        "reasoning_content": None,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "cost_usd": cost,
        "model": "deepseek-flash",
    }


class ScriptedProvider:
    """按剧本回话的假模型：chat_with_tools 按 script 顺序回，
    chat_messages 按 final_script 顺序回（兜底那轮走这里）。"""

    def __init__(self, model="deepseek-flash", script=None, final_script=None):
        self.model = model
        self.script = list(script or [])
        self.final_script = list(final_script or [])
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
        if self.final_script:
            return dict(self.final_script.pop(0))
        return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}


class FakeSearchService:
    """回写死结果的假搜索服务。"""

    def __init__(self, outcomes=None):
        self.translator = None
        self.queries = []
        self.date_ranges = []
        self.outcomes = list(outcomes or [])

    def search(self, query, kind="normal", max_results=5, cross_check=False,
               force_provider=None, date_range=None):
        self.queries.append(query)
        self.date_ranges.append(date_range)
        if self.outcomes:
            return self.outcomes.pop(0)
        return SearchOutcome(query=query, provider="fake", status="ok", results=[])


def _sp_nov10_outcome():
    """标普 11 月 10 日收盘 7,520.93（51 号实测里那个“明明有”的数）。"""
    return SearchOutcome(
        query="标普 11 月 10 日收盘",
        provider="fake",
        status="ok",
        results=[
            SearchResult(
                title="S&P 500 closes at 7,520.93 on November 10, 2026",
                url="https://example.com/sp-nov10",
                snippet="The S&P 500 closed at 7,520.93 on November 10, 2026, down 0.38%.",
                published_at="2026-11-11",
            ),
        ],
    )


@pytest.fixture
def store(tmp_path):
    return TaskStore(file_path=str(tmp_path / "tasks.json"))


@pytest.fixture
def gateway(tmp_path):
    return ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))


def _make_loop(provider, search_service, store, gateway):
    return AgentLoop(
        provider, gateway=gateway, task_store=store, search_service=search_service
    )


# ---------- 51 号正测：没搜就直接答，拦住、提醒、再搜 ----------


def test_51_replay_sp_11_10_searches_after_reminder(store, gateway):
    """51 号原样重放：“标普的 11.10”——模型第一次没搜就直接答，
    门禁提醒它去搜；搜完再答 7,520.93，正常交出。"""
    provider = ScriptedProvider(
        script=[
            # 第一轮：没调工具，直接给了个数（凭记忆编的）。
            _tool_response(text="标普 11 月 10 日收于 7,500 点左右。"),
            # 第二轮：被提醒后去搜。
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {
                    "query": "标普 11 月 10 日收盘",
                    "date_range": {"start": "2026-11-10", "end": "2026-11-10"},
                })
            ]),
            # 第三轮：搜完再答。
            _tool_response(text="标普 11 月 10 日收于 7,520.93，跌 0.38%。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([_sp_nov10_outcome()]), store, gateway)
    outcome = loop.run("标普的 11.10", goal="标普的 11.10")
    assert outcome["mode"] == "done"
    assert "7,520.93" in outcome["reply"]
    # 提醒那轮确实跑了：任务消息里有系统提醒。
    messages = outcome["task"]["messages"]
    assert any("先联网查" in (m.get("content") or "") for m in messages)
    # 模型第一轮凭记忆给的错数没留在任务记录里（被换掉了）。
    assistant_texts = [m["content"] for m in messages if m.get("role") == "assistant"]
    assert all("7,500" not in text for text in assistant_texts)
    # 搜过（标普，11 月 10 日）这一对，任务里有记录。
    assert outcome["task"]["searches"][0]["query"] == "标普 11 月 10 日收盘"


def test_51_second_direct_answer_returns_dont_know(store, gateway):
    """51 号兜底：提醒过一次还直接答（还是没搜），就回不知道，
    不把没查过的数交出去。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(text="标普 11 月 10 日收于 7,500 点左右。"),
            _tool_response(text="我没查到，大概是 7,500 点。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([]), store, gateway)
    outcome = loop.run("标普的 11.10", goal="标普的 11.10")
    assert outcome["mode"] == "done"
    assert "7,500" not in outcome["reply"]
    assert "不敢给数" in outcome["reply"]
    assert "2026-11-10" in outcome["reply"]


def test_51_wrong_entity_search_does_not_cover(store, gateway):
    """51 号的（实体，日期）对：搜了纳指 11/10，问的是标普 11/10——
    搜过别的实体不算搜过，门禁照样拦；搜对实体后才放行。"""
    nasdaq_outcome = SearchOutcome(
        query="纳斯达克 11 月 10 日收盘",
        provider="fake",
        status="ok",
        results=[
            SearchResult(
                title="Nasdaq falls on November 10",
                url="https://example.com/nov10",
                snippet="The Nasdaq Composite declined 0.7% to close at 25,065.87 on November 10, 2026.",
                published_at="2026-11-11",
            ),
        ],
    )
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {
                    "query": "纳斯达克 11 月 10 日收盘",
                    "date_range": {"start": "2026-11-10", "end": "2026-11-10"},
                })
            ]),
            # 搜了纳指，却直接答标普——门禁要拦。
            _tool_response(text="标普 11 月 10 日收于 7,500 点左右。"),
            # 被提醒后搜标普。
            _tool_response(tool_calls=[
                _tool_call("call_2", "search", {
                    "query": "标普 11 月 10 日收盘",
                    "date_range": {"start": "2026-11-10", "end": "2026-11-10"},
                })
            ]),
            _tool_response(text="标普 11 月 10 日收于 7,520.93，跌 0.38%。"),
        ],
    )
    loop = _make_loop(
        provider,
        FakeSearchService([nasdaq_outcome, _sp_nov10_outcome()]),
        store, gateway,
    )
    outcome = loop.run("标普的 11.10", goal="标普的 11.10")
    assert outcome["mode"] == "done"
    assert "7,520.93" in outcome["reply"]
    assert "7,500" not in outcome["reply"]
    assert len(outcome["task"]["searches"]) == 2


# ---------- 51 号不该拦的情形 ----------


def test_51_dca_question_not_gated(store, gateway):
    """定投算钱不强制搜：模型直接算出来交出，不提醒、不多花一轮。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(text="从 11 月 10 日到年底，每天定投 5 块，一共 52 天，260 元。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([]), store, gateway)
    outcome = loop.run(
        "从 11 月 10 日到年底，每天定投 5 块买万家纳指，一共投多少钱",
        goal="从 11 月 10 日到年底，每天定投 5 块买万家纳指，一共投多少钱",
    )
    assert outcome["mode"] == "done"
    assert "260" in outcome["reply"]
    assert len(provider.tool_calls_log) == 1  # 只问了一轮，没被提醒


def test_51_future_date_not_gated(store, gateway):
    """未来日期不强制搜：要问的那天还没到，模型按日期逻辑答“还没到”就行。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(text="今天才 2026 年 11 月 12 日，2030 年 1 月 2 日还没到，没法给收盘数。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([]), store, gateway)
    outcome = loop.run(
        "纳斯达克 2030 年 1 月 2 日收盘多少",
        goal="纳斯达克 2030 年 1 月 2 日收盘多少",
    )
    assert outcome["mode"] == "done"
    assert len(provider.tool_calls_log) == 1


def test_51_greeting_and_weather_not_gated(store, gateway):
    """普通聊天不强制联网：问好直接答，不提醒。"""
    provider = ScriptedProvider(
        script=[_tool_response(text="你好 BOOS，我是十二。有什么要处理的说一声就行。")],
    )
    loop = _make_loop(provider, FakeSearchService([]), store, gateway)
    outcome = loop.run("你好", goal="你好")
    assert outcome["mode"] == "done"
    assert len(provider.tool_calls_log) == 1


def test_51_reasoning_only_answer_not_gated(store, gateway):
    """纯推脱的回答不拦：模型没给数、没说涨跌、没说“没查到”，
    只是讲“具体数我得查一下”——没给数据主张，门禁不拦。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(text="11 月 10 日是周二，正常开市，具体收盘数我得查一下。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([]), store, gateway)
    outcome = loop.run(
        "标普 11 月 10 日收盘多少",
        goal="标普 11 月 10 日收盘多少",
    )
    assert outcome["mode"] == "done"
    assert len(provider.tool_calls_log) == 1


# ---------- 配对判断（单元级） ----------


def test_51_search_covers_pair_unit():
    """（实体，日期）对的判定：实体对不上、日期对不上、搜失败了、
    没结果，都不算搜过；查询词文本里写了日期、没带 date_range 也算。"""
    from datetime import date

    requested = date(2026, 11, 10)
    base = {
        "query": "标普 11 月 10 日收盘",
        "executed_query": "S&P 500 closing on November 10",
        "date_range": {"start": "2026-11-10", "end": "2026-11-10"},
        "ok": True,
        "result_count": 3,
    }
    assert search_covers_pair([base], "sp500", requested) is True
    # 实体不对：搜的是纳指。
    assert search_covers_pair(
        [{**base, "query": "纳斯达克 11 月 10 日收盘",
          "executed_query": "Nasdaq closing on November 10"}],
        "sp500", requested,
    ) is False
    # 日期不对：搜的是 11 月 9 日。
    assert search_covers_pair(
        [{**base, "date_range": {"start": "2026-11-09", "end": "2026-11-09"},
          "query": "标普 11 月 9 日收盘",
          "executed_query": "S&P 500 closing on November 9"}],
        "sp500", requested,
    ) is False
    # 搜失败了不算搜过。
    assert search_covers_pair([{**base, "ok": False}], "sp500", requested) is False
    # 没结果不算搜过。
    assert search_covers_pair([{**base, "result_count": 0}], "sp500", requested) is False
    # 没带 date_range，但查询词文本里写了日期，也算搜过。
    assert search_covers_pair(
        [{**base, "date_range": None, "query": "标普 11 月 10 日收盘"}],
        "sp500", requested,
    ) is True
    # 英文查询词也认（AnySearch 翻译后的英文查询）。
    assert search_covers_pair(
        [{**base, "date_range": None, "query": "标普收盘",
          "executed_query": "S&P 500 closing on November 10"}],
        "sp500", requested,
    ) is True


# ---------- B 组风格新问法（51 号 5 条） ----------

SEARCH_FIRST_PHRASES = [
    # （问法，要问的那天，实体）
    ("道指 11.10 收了多少", "2026-11-10", "dow"),
    ("纳指 11.9 那天呢", "2026-11-09", "nasdaq"),
    ("标普昨天涨了还是跌了", "2026-11-11", "sp500"),
    ("黄金 11 月 11 日收盘价多少", "2026-11-11", "gold"),
    ("道琼斯前天收多少点", "2026-11-10", "dow"),
]

SEARCH_FIRST_NEGATIVES = [
    # 不该开门禁的：定投算钱、问好、天气、闲聊。
    "每天定投 5 块买万家纳指，一共投多少钱",
    "你好",
    "广州今天天气怎么样",
    "我喜欢吃拉面",
    "从 11 月 10 日到年底，每天定投 5 块，一共投多少钱",
]


def test_51_phrases_cover_search_first():
    """51 号 5 条新问法：条条都要开门禁（短问句、口语、连续追问都得触发），
    实体和日期都要认对；下面 5 条不该开的，一条都不能开。"""
    from datetime import date

    today = date(2026, 11, 12)
    for phrase, wanted_date, wanted_entity in SEARCH_FIRST_PHRASES:
        assert is_market_data_question(phrase), phrase
        assert resolve_requested_date(phrase, today).isoformat() == wanted_date, phrase
        assert extract_market_entity(phrase)[0] == wanted_entity, phrase
    for phrase in SEARCH_FIRST_NEGATIVES:
        assert not is_market_data_question(phrase), phrase
