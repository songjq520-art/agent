# tests/test_date_attribution.py（50/54 号根治：循环层的日期校验）
# 作用：验证 tools/agent_loop.py 的数字-日期校验——模型引用了别的日子的数时，
#       循环在交出回答前拦住它、让它重答一次；重答还错就直接回“不知道”。
# 全部离线模拟：ScriptedProvider 按剧本回话，FakeSearchService 回写死的
#       搜索结果，不调真实 API、不花钱。
# 关键回归（2026-11-12 实测）：
#   54 号：问“纳指 11 月 11 日是涨还是跌”，模型拿 11 月 10 日的 25,065.88 /
#     -0.74% 当答案——必须拦住、纠正成 25,466.19 / +1.60%（涨）。
#   50 号：问 11 月 9 日，模型答对 25,253.14，结果里混着 25,466.19
#     （11 月 11 日的数）——不能因为它就降信心、不能触发纠正。

import copy
import json

import pytest

from tasks.task_store import TaskStore
from tools.agent_loop import AgentLoop
from tools.audit import AuditLog
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


def _final_response(text, cost=0.001):
    return {
        "text": text,
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        "cost_usd": cost,
        "model": "deepseek-flash",
    }


class ScriptedProvider:
    """按剧本回话的假模型：chat_with_tools 按 script 顺序回，
    chat_messages 按 final_script 顺序回（纠正那轮走这里）。"""

    def __init__(self, model="deepseek-flash", script=None, final_script=None):
        self.model = model
        self.script = list(script or [])
        self.final_script = list(final_script or [])
        self.tool_calls_log = []
        self.final_calls_log = []
        self.chat_calls = []

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        self.chat_calls.append({"prompt": prompt, "ledger": ledger})
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


def _nov_outcome():
    """11 月 9/10/11 三天的纳指收盘混在一起的结果（50/54 号的真实情形）。"""
    return SearchOutcome(
        query="纳斯达克收盘",
        provider="fake",
        status="ok",
        results=[
            SearchResult(
                title="Nasdaq Composite closed at 25,253.14 on November 9, 2026",
                url="https://example.com/nov9",
                snippet="The Nasdaq Composite closed at 25,253.14 on November 9, 2026, up 2.74%.",
                published_at="2026-11-10",
            ),
            SearchResult(
                title="Nasdaq falls on November 10",
                url="https://example.com/nov10",
                snippet="The Nasdaq Composite declined 0.7% to close at 25,065.87 on November 10, 2026.",
                published_at="2026-11-11",
            ),
            SearchResult(
                title="Nasdaq rallies on November 11",
                url="https://example.com/nov11",
                snippet="The Nasdaq Composite closed at 25,466.19 on November 11, 2026, up 1.60% "
                        "from 25,065.87 the session before.",
                published_at="2026-11-12",
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


# ---------- 54 号：拿错日子的数，拦住、纠正 ----------


def test_54_wrong_date_number_is_corrected(store, gateway):
    """54 号：问 11 月 11 日，模型拿 11 月 10 日的 25,065.88 当答案——
    循环在交出前拦住它，纠正一轮，返回 25,466.19（涨）。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {
                    "query": "纳斯达克 11 月 11 日收盘涨跌",
                    "date_range": {"start": "2026-11-11", "end": "2026-11-11"},
                })
            ]),
            # 模型直接回答，却引用了 11 月 10 日的数（25,065.88 是结果里
            # 25,065.87 四舍五入的写法，算同一个数）。
            _tool_response(text="昨天（11月11日）纳指是跌的：收于 25,065.88 点，跌 0.74%。"),
        ],
        final_script=[
            _final_response("昨天（11月11日）纳指是涨的：收于 25,466.19，涨 1.60%。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([_nov_outcome()]), store, gateway)
    outcome = loop.run(
        "纳斯达克 11 月 11 日是涨还是跌",
        goal="纳斯达克 11 月 11 日是涨还是跌",
    )
    assert outcome["mode"] == "done"
    # 纠正过的回答：涨、25,466.19，错数 25,065.88 不见了。
    assert "25,466.19" in outcome["reply"]
    assert "涨" in outcome["reply"]
    assert "25,065.88" not in outcome["reply"]
    # 纠正那轮确实跑了（chat_messages 被调了一次），纠正消息里点名了错数和正确日期。
    assert len(provider.final_calls_log) == 1
    correction_messages = provider.final_calls_log[0]["messages"]
    correction = correction_messages[-1]
    assert correction["role"] == "user"
    assert "25,065.88" in correction["content"]
    assert "2026-11-11" in correction["content"]
    # 任务记录里只留纠正过的回答，不留错的那份。
    saved_messages = outcome["task"]["messages"]
    assistant_texts = [m["content"] for m in saved_messages if m.get("role") == "assistant"]
    assert all("25,065.88" not in text for text in assistant_texts)
    # 搜索结果的数字-日期对照记进了任务。
    dated = outcome["task"].get("search_dated_numbers") or []
    pairs = {(item["number"], item["date"]) for item in dated}
    assert ("25,466.19", "2026-11-11") in pairs
    assert ("25,065.87", "2026-11-10") in pairs
    assert ("25,253.14", "2026-11-09") in pairs


def test_54_correction_still_wrong_returns_dont_know(store, gateway):
    """54 号的兜底：纠正一轮后模型还拿错数，就直接回“不知道”，
    不把错数交给 BOOS。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {
                    "query": "纳斯达克 11 月 11 日收盘涨跌",
                    "date_range": {"start": "2026-11-11", "end": "2026-11-11"},
                })
            ]),
            _tool_response(text="昨天（11月11日）纳指是跌的：收于 25,065.88 点。"),
        ],
        final_script=[
            _final_response("昨天（11月11日）纳指是跌的：收于 25,065.88 点。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([_nov_outcome()]), store, gateway)
    outcome = loop.run(
        "纳斯达克 11 月 11 日是涨还是跌",
        goal="纳斯达克 11 月 11 日是涨还是跌",
    )
    assert outcome["mode"] == "done"
    assert "25,065.88" not in outcome["reply"]
    assert "不敢给数" in outcome["reply"]
    assert "2026-11-11" in outcome["reply"]


# ---------- 50 号：答对了就别动 ----------


def test_50_correct_answer_passes_without_correction(store, gateway):
    """50 号：问 11 月 9 日，模型答对 25,253.14——结果里虽然混着
    25,466.19（11 月 11 日的数），但回答没引用它，就不触发纠正、
    不多花一次模型调用。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {
                    "query": "纳斯达克 11 月 9 日收盘",
                    "date_range": {"start": "2026-11-09", "end": "2026-11-09"},
                })
            ]),
            _tool_response(text="纳指 11 月 9 日收于 25,253.14，涨 2.74%。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([_nov_outcome()]), store, gateway)
    outcome = loop.run(
        "纳斯达克 11 月 9 日收盘多少",
        goal="纳斯达克 11 月 9 日收盘多少",
    )
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "纳指 11 月 9 日收于 25,253.14，涨 2.74%。"
    assert provider.final_calls_log == []  # 没触发纠正，没多花钱


# ---------- 不该校验的情形 ----------


def test_54_dot_date_question_is_corrected(store, gateway):
    """50/54 的缺口（2026-11-12 补）：BOOS 原话“标普的 11.10”——点分日期
    之前解析不出来，requested_date 是 None，校验根本不生效，模型拿
    11 月 11 日的数当 11 月 10 日的答案也没人拦。现在点分日期能换算，
    拿错日子的数必须照样拦住、纠正成 7,520.93。"""
    sp_outcome = SearchOutcome(
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
            SearchResult(
                title="S&P 500 rallies on November 11",
                url="https://example.com/sp-nov11",
                snippet="The S&P 500 closed at 7,619.38 on November 11, 2026, up 1.31%.",
                published_at="2026-11-12",
            ),
        ],
    )
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {
                    "query": "标普 11 月 10 日收盘",
                    "date_range": {"start": "2026-11-10", "end": "2026-11-10"},
                })
            ]),
            # 模型拿了 11 月 11 日的数当答案。
            _tool_response(text="标普 11 月 10 日收于 7,619.38，涨 1.31%。"),
        ],
        final_script=[
            _final_response("标普 11 月 10 日收于 7,520.93，跌 0.38%。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([sp_outcome]), store, gateway)
    outcome = loop.run("标普的 11.10", goal="标普的 11.10")
    assert outcome["mode"] == "done"
    assert "7,520.93" in outcome["reply"]
    assert "7,619.38" not in outcome["reply"]
    assert len(provider.final_calls_log) == 1  # 纠正那轮确实跑了


def test_non_market_question_skips_validation(store, gateway):
    """天气问题不校验数的日期（降水概率 3% 不是收盘数，校验它只会误伤）。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {"query": "广州今天天气"})
            ]),
            _tool_response(text="广州今天多云，13~26°C，降水概率约 3%。"),
        ],
    )
    weather_outcome = SearchOutcome(
        query="广州今天天气",
        provider="fake",
        status="ok",
        results=[
            SearchResult(
                title="广州天气",
                url="https://example.com/weather",
                snippet="广州今天多云，13~26°C，降水概率约 3%。明天转雨，可能有中雨。",
                published_at="2026-11-12",
            )
        ],
    )
    loop = _make_loop(provider, FakeSearchService([weather_outcome]), store, gateway)
    outcome = loop.run("广州今天天气怎么样", goal="广州今天天气怎么样")
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "广州今天多云，13~26°C，降水概率约 3%。"
    assert provider.final_calls_log == []


def test_no_date_in_question_skips_validation(store, gateway):
    """问题里没写日期、也换算不出相对日期时，不校验（别拿错日期拦正确答案）。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {"query": "纳指收盘"})
            ]),
            _tool_response(text="纳指收于 25,466.19。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([_nov_outcome()]), store, gateway)
    outcome = loop.run("纳指收盘多少", goal="纳指收盘多少")
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "纳指收于 25,466.19。"
    assert provider.final_calls_log == []


def test_answer_without_numbers_skips_validation(store, gateway):
    """回答里没数时不用校验。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {"query": "纳斯达克 11 月 11 日收盘涨跌"})
            ]),
            _tool_response(text="昨天纳指涨了。"),
        ],
    )
    loop = _make_loop(provider, FakeSearchService([_nov_outcome()]), store, gateway)
    outcome = loop.run(
        "纳斯达克 11 月 11 日是涨还是跌",
        goal="纳斯达克 11 月 11 日是涨还是跌",
    )
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "昨天纳指涨了。"
    assert provider.final_calls_log == []


# ---------- 搜索工具结果里带 dated_numbers ----------


def test_search_tool_result_carries_dated_numbers(store, gateway):
    """搜索工具返回的每条结果都带 dated_numbers（数字-日期对照，程序标的）。"""
    from tools.search_tool import _run_search, set_search_service

    set_search_service(FakeSearchService([_nov_outcome()]))
    try:
        result = _run_search({"query": "纳斯达克收盘"})
    finally:
        set_search_service(None)
    assert result["ok"] is True
    by_url = {item["url"]: item for item in result["data"]["results"]}
    assert ("25,466.19", "2026-11-11") in [
        (dn["number"], dn["date"]) for dn in by_url["https://example.com/nov11"]["dated_numbers"]
    ]
    assert ("25,065.87", "2026-11-10") in [
        (dn["number"], dn["date"]) for dn in by_url["https://example.com/nov10"]["dated_numbers"]
    ]
    assert ("25,253.14", "2026-11-09") in [
        (dn["number"], dn["date"]) for dn in by_url["https://example.com/nov9"]["dated_numbers"]
    ]
    # "from 25,065.87 the session before" 那个旧数不配日期，不进对照表。
    nov11_numbers = {dn["number"] for dn in by_url["https://example.com/nov11"]["dated_numbers"]}
    assert nov11_numbers == {"25,466.19", "1.60%"}


# ---------- B 组风格新问法（50/54 号各 5 条，和实测用过的不一样） ----------

DATE_ATTRIBUTION_PHRASES = {
    # 50 号：不同日期的数混在一起，答对的不能被别的日期的数拖低把握。
    50: [
        "标普 11 月 9 日收盘多少？结果里有个 7,619.38 是 11 月 11 日的，别搞混。",
        "道指 11 月 10 日收了多少？搜出来有个 52,486.49 是 11 月 11 日的。",
        "纳指 11 月 6 日收盘多少？别把 11 月 9 日的 25,253.14 当成它的。",
        "昨天标普是涨是跌？注意别拿前天的数当昨天的。",
        "11 月 9 日和 11 月 11 日纳指各收多少？两个数别串了。",
    ],
    # 54 号：问“昨天/前天”这种相对日期，数必须对上换算后的那天。
    54: [
        "纳指前天是涨还是跌？",
        "标普昨天收盘多少？",
        "道指昨天涨了还是跌了？",
        "纳指昨天收了多少点？",
        "昨天美股三大指数各涨跌多少？",
    ],
}


def test_date_attribution_phrases_cover_50_and_54():
    """50/54 号各备 5 条新问法，条条不一样（实测换说法验泛化用）。"""
    assert set(DATE_ATTRIBUTION_PHRASES) == {50, 54}
    for item, phrases in DATE_ATTRIBUTION_PHRASES.items():
        assert len(phrases) == 5, item
        assert len(set(phrases)) == 5, item
        assert all(isinstance(phrase, str) and phrase.strip() for phrase in phrases)
