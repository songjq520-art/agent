# tests/test_followup_completion.py（追问补全：上下文接日期）
# 作用：验证 tools/date_numbers.py 的 complete_follow_up_question——
#   BOOS 先问“道指11.10”，接着问“标普呢”，意思就是“标普的 11.10”；
#   当时只看本句，没日期就认不出来，模型直接说不知道。根因是追问
#   没继承上文的日期/实体。治本：在 51 号门禁前面补全问题，不改
#   门禁本身的逻辑。全部离线模拟，不调真实 API、不花钱。

import json
from datetime import date

import pytest

from tasks.task_store import TaskStore
from tools.agent_loop import AgentLoop
from tools.audit import AuditLog
from tools.date_numbers import complete_follow_up_question
from tools.search import SearchOutcome, SearchResult
from tools.tool_gateway import ToolGateway

TODAY = date(2026, 11, 12)


def _history(*user_turns):
    """对话历史：只给 user 的问话（assistant 的回答不影响补全）。"""
    messages = []
    for turn in user_turns:
        messages.append({"role": "user", "content": turn})
        messages.append({"role": "assistant", "content": "（回答）"})
    return messages


# ---------- 单元级：补全函数本身 ----------


def test_followup_sp_after_dow():
    """原样重放：上轮“道指11.10”，这轮“标普呢”→“标普的 11.10”。"""
    assert (
        complete_follow_up_question("标普呢", _history("道指11.10"), TODAY)
        == "标普的 11.10"
    )


def test_followup_head_marker():
    """带“那”的追问也补：“那道指呢”→“道指的 11.10”。"""
    assert (
        complete_follow_up_question("那道指呢", _history("标普的 11.10"), TODAY)
        == "道指的 11.10"
    )


def test_followup_missing_entity():
    """缺实体：上轮“道指11.10”，这轮“11.10 怎么样”→“道指的 11.10 怎么样”。"""
    assert (
        complete_follow_up_question("11.10 怎么样", _history("道指11.10"), TODAY)
        == "道指的 11.10 怎么样"
    )


def test_followup_bare_particle():
    """光秃秃一个“呢”：两个都从上轮拿→“道指的 11.10”。"""
    assert (
        complete_follow_up_question("呢", _history("道指11.10"), TODAY)
        == "道指的 11.10"
    )


def test_followup_bare_entity():
    """光秃秃一个实体也补：“道指”→“道指的 11.10”。"""
    assert (
        complete_follow_up_question("道指", _history("标普的 11.10"), TODAY)
        == "道指的 11.10"
    )


def test_followup_keeps_direction_question():
    """涨跌问法保留尾巴：“标普涨了吗”→“标普的 11.10 涨了吗”。"""
    assert (
        complete_follow_up_question("标普涨了吗", _history("道指11.10"), TODAY)
        == "标普的 11.10 涨了吗"
    )


def test_followup_relative_date_in_history():
    """上轮是相对日期也认：“道指昨天”（今天 11 月 12 日）→ 11 月 11 日。"""
    assert (
        complete_follow_up_question("标普呢", _history("道指昨天"), TODAY)
        == "标普的 11.11"
    )


def test_followup_uses_second_last_turn():
    """最近一轮是闲聊也不怕：往前一轮找日期和实体。"""
    history = _history("道指11.10", "你好")
    assert complete_follow_up_question("标普呢", history, TODAY) == "标普的 11.10"


def test_followup_no_history_no_completion():
    """没有上文时不补：原样返回，别凭空编一个日期。"""
    assert complete_follow_up_question("标普呢", [], TODAY) == "标普呢"
    assert complete_follow_up_question("标普呢", None, TODAY) == "标普呢"


def test_followup_complete_question_unchanged():
    """本句自己带齐日期实体时不补：“标普的 11.10”原样返回。"""
    assert (
        complete_follow_up_question("标普的 11.10", _history("道指11.10"), TODAY)
        == "标普的 11.10"
    )


def test_followup_not_a_followup_unchanged():
    """不是追问时不补：问好、完整的新问题原样返回。"""
    assert complete_follow_up_question("你好", _history("道指11.10"), TODAY) == "你好"
    assert (
        complete_follow_up_question("纳指昨天涨还是跌", _history("道指11.10"), TODAY)
        == "纳指昨天涨还是跌"
    )


def test_followup_other_entity_not_completed():
    """问的是别的东西时不补：“比特币呢”不能补成上文的道指。"""
    assert (
        complete_follow_up_question("比特币呢", _history("道指11.10"), TODAY)
        == "比特币呢"
    )
    assert (
        complete_follow_up_question("比特币11.10涨了吗", _history("道指11.10"), TODAY)
        == "比特币11.10涨了吗"
    )


def test_followup_history_without_date_no_completion():
    """上轮也没有日期时不补：别拿更早的、没影的日期顶上。"""
    assert (
        complete_follow_up_question("标普呢", _history("你好"), TODAY) == "标普呢"
    )


# ---------- 循环级：两轮对话，第二轮触发搜索 ----------


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
    """按剧本回话的假模型。"""

    def __init__(self, model="deepseek-flash", script=None):
        self.model = model
        self.script = list(script or [])

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        return {"text": "translated", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        if not self.script:
            raise AssertionError("剧本用完了，模型又要调工具——循环该停了")
        return dict(self.script.pop(0))

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None, model=None, max_tokens=None):
        return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}


class FakeSearchService:
    """回写死结果的假搜索服务。"""

    def __init__(self, outcomes=None):
        self.translator = None
        self.queries = []
        self.outcomes = list(outcomes or [])

    def search(self, query, kind="normal", max_results=5, cross_check=False,
               force_provider=None, date_range=None):
        self.queries.append(query)
        if self.outcomes:
            return self.outcomes.pop(0)
        return SearchOutcome(query=query, provider="fake", status="ok", results=[])


def _sp_nov10_outcome():
    """标普 11 月 10 日收盘 7,520.93。"""
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


def test_loop_followup_triggers_search(tmp_path):
    """两轮对话：第一轮问过“道指11.10”，第二轮只问“标普呢”——
    补全成“标普的 11.10”后，51 号门禁认出来、提醒模型去搜，
    搜完答 7,520.93。任务里 goal 是补全后的，原话在 goal_original。"""
    provider = ScriptedProvider(
        script=[
            # 第一轮：没搜就直接答（凭记忆编的）。
            _tool_response(text="标普 11 月 10 日收于 7,500 点左右。"),
            # 第二轮：被门禁提醒后去搜。
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
    search_service = FakeSearchService([_sp_nov10_outcome()])
    loop = AgentLoop(
        provider,
        gateway=ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log"))),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=search_service,
    )
    history = [
        {"role": "user", "content": "道指11.10"},
        {"role": "assistant", "content": "道指 11 月 10 日收于 52,231.89，跌 0.30%。"},
    ]
    outcome = loop.run("标普呢", goal="标普呢", history=history)
    assert outcome["mode"] == "done"
    assert "7,520.93" in outcome["reply"]
    assert outcome["task"]["goal"] == "标普的 11.10"
    assert outcome["task"]["goal_original"] == "标普呢"
    assert search_service.queries == ["标普 11 月 10 日收盘"]
