# tests/test_batch2_session.py（第 2 批：41、38、12 号——会话地基剩下的）
# 作用：刷新恢复当前对话、对话全文存盘 + 保留期流转、任务状态机 +
# 进度 + 恢复卡住的任务。全部离线模拟，不调真实 API、不花钱；
# 真调在 /tmp 外面另做。
# 规矩和第 0 步一样：已修好的项测通过；没修的项标 xfail（预期失败），
# 修好那批时把 xfail 拿掉、测通过——套件一直是绿的。

import http.client
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from config import BudgetTracker
from memory.conversation_summary import ConversationSummaryStore
from memory.conversation_transcript import (
    TRANSCRIPT_CONDENSE_DAYS,
    TRANSCRIPT_FULL_DAYS,
    TranscriptStore,
    message_fingerprint,
)
from tasks.task_store import TASK_STATUSES, TaskStore, make_task
from tools.agent_loop import AgentLoop
from tools.tool_gateway import ToolGateway
from web.server import ChatState, ServerConfig, create_server
import web.server as server_module


def _request(port, method, path, body=None):
    """向测试服务发一次请求，返回 (状态码, 解析后的 JSON 或原始文本)。"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {}
    payload = None
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    content_type = resp.getheader("Content-Type") or ""
    if "application/json" in content_type:
        return resp.status, json.loads(raw.decode("utf-8"))
    return resp.status, raw.decode("utf-8", errors="replace")


def _start_batch2_server(tmp_path, mem0_store=None):
    """用临时端口、临时预算/摘要/任务/全文文件启动一个测试服务。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="127.0.0.1", port=0)
    summary_store = ConversationSummaryStore(
        file_path=tmp_path / "summary.json", summarizer=None
    )
    task_store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    transcript_store = TranscriptStore(directory=tmp_path / "transcripts")
    server, state = create_server(
        config=config,
        budget_tracker=tracker,
        mem0_store=mem0_store,
        summary_store=summary_store,
        task_store=task_store,
        transcript_store=transcript_store,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], state


def _stop_server(server):
    server.shutdown()
    server.server_close()


@pytest.fixture
def batch2_server(tmp_path):
    server, port, state = _start_batch2_server(tmp_path)
    try:
        yield port, state
    finally:
        _stop_server(server)


def _fake_done_run_question(prompt, budget_tracker=None, messages=None, **kwargs):
    """假的 run_question：直接返回一轮成功的假回答，不调用真实模型。"""
    return {
        "mode": "real",
        "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
        "result": {"text": "假回答", "model": "deepseek-flash", "cost_usd": 0.0001, "usage": None},
        "message": "真实调用完成。",
        "model_status": {},
        "timings": {},
    }


def _fake_waiting_run_question(prompt, budget_tracker=None, messages=None, **kwargs):
    """假的 run_question：停在确认卡上（12 号计时日志测试用）。"""
    task = make_task(goal=prompt, model="deepseek-flash")
    task["status"] = "WAITING_APPROVAL"
    return {
        "mode": "waiting_approval",
        "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
        "result": None,
        "message": "等 BOOS 确认工具调用。",
        "model_status": {},
        "task": task,
        "approval": {
            "approval_id": "appr_1",
            "tool": "search",
            "args": {"query": "x"},
            "risk_level": "low",
            "expires_at": "2099-01-01T00:00:00Z",
        },
        "cost_usd": 0.001,
        "timings": {"model_ms": 12},
    }


def _two_rounds():
    """两轮对话（user + assistant 各两条），供全文/当前对话测试用。"""
    return [
        {"role": "user", "content": "今天天气怎么样"},
        {"role": "assistant", "content": "广州晴，24 度。"},
        {"role": "user", "content": "纳斯达克昨天收盘多少"},
        {"role": "assistant", "content": "25,065.88，跌 0.74%。"},
    ]


# ---------- 12 号：任务状态机 ----------


def test_task_statuses_include_paused_and_unknown():
    """12 号：状态机是显式的九态，PAUSED（干净关机停下）和 UNKNOWN
    （结果未知）都在里面；写错状态名还是直接报错。"""
    assert "PAUSED" in TASK_STATUSES
    assert "UNKNOWN" in TASK_STATUSES
    assert len(TASK_STATUSES) == 9
    store = TaskStore(file_path="/tmp/does-not-matter-batch2.json")
    bad = make_task(goal="x", model="deepseek-flash")
    bad["status"] = "乱写的状态"
    with pytest.raises(ValueError):
        store.save(bad)


def test_reconcile_on_startup_marks_running_unknown(tmp_path):
    """12 号：启动核对——还在 RUNNING 的改成 UNKNOWN，记下原因，
    不悄悄重执行；跑完了的、没过期的等确认卡不动；过期的等确认卡
    改成 EXPIRED。"""
    store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    running = make_task(goal="查一下纳斯达克", model="deepseek-flash")
    running["status"] = "RUNNING"
    store.save(running)
    done = make_task(goal="你好", model="deepseek-flash")
    done["status"] = "COMPLETED"
    store.save(done)
    waiting = make_task(goal="搜一下", model="deepseek-flash")
    waiting["status"] = "WAITING_APPROVAL"
    waiting["approval"] = {
        "approval_id": "appr_old",
        "tool": "search",
        "args": {"query": "x"},
        "risk_level": "low",
        "created_at": "2020-01-01T00:00:00+00:00",
        "expires_at": "2020-01-01T00:10:00+00:00",
    }
    store.save(waiting)
    fresh_waiting = make_task(goal="再搜一下", model="deepseek-flash")
    fresh_waiting["status"] = "WAITING_APPROVAL"
    fresh_waiting["approval"] = {
        "approval_id": "appr_new",
        "tool": "search",
        "args": {"query": "y"},
        "risk_level": "low",
        "created_at": "2099-01-01T00:00:00+00:00",
        "expires_at": "2099-01-01T00:10:00+00:00",
    }
    store.save(fresh_waiting)

    affected = store.reconcile_on_startup()
    assert {item["task_id"] for item in affected} == {
        running["task_id"],
        waiting["task_id"],
    }
    assert store.load(running["task_id"])["status"] == "UNKNOWN"
    assert store.load(running["task_id"])["reconciled_reason"]
    assert store.load(done["task_id"])["status"] == "COMPLETED"
    assert store.load(waiting["task_id"])["status"] == "EXPIRED"
    assert store.load(fresh_waiting["task_id"])["status"] == "WAITING_APPROVAL"
    assert store.list_unresolved()[0]["task_id"] == running["task_id"]


def test_pause_running_on_shutdown(tmp_path):
    """12 号：干净关机时还在跑的标成 PAUSED，和崩溃的 UNKNOWN 区分开；
    跑完了的不动。"""
    store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    running = make_task(goal="查一下", model="deepseek-flash")
    running["status"] = "RUNNING"
    store.save(running)
    done = make_task(goal="你好", model="deepseek-flash")
    done["status"] = "COMPLETED"
    store.save(done)
    affected = store.pause_running_on_shutdown()
    assert [item["task_id"] for item in affected] == [running["task_id"]]
    assert store.load(running["task_id"])["status"] == "PAUSED"
    assert store.load(done["task_id"])["status"] == "COMPLETED"


# ---------- 38 号：全文存盘 + 保留期 ----------


def test_transcript_append_and_get(tmp_path):
    """38 号：全文按对话 id 存，一个文件一场对话；追加只加新消息，
    恢复回来的旧消息不重复写（调用方只传新消息）。"""
    store = TranscriptStore(directory=tmp_path / "transcripts")
    record = store.append("conv1", "天气", _two_rounds())
    assert record["status"] == "full"
    assert len(record["messages"]) == 4
    record = store.append(
        "conv1", "天气", [{"role": "user", "content": "明天呢"}, {"role": "assistant", "content": "多云。"}]
    )
    assert len(record["messages"]) == 6
    got = store.get("conv1")
    assert got["messages"][0]["content"] == "今天天气怎么样"
    assert got["messages"][-1]["content"] == "多云。"
    assert store.get("没有这场") is None
    with pytest.raises(ValueError):
        store.append("../跑出去", "坏", _two_rounds())


def test_transcript_insert_before_fingerprint(tmp_path):
    """38 号：恢复的那轮插在历史中间时，按后面那条消息的指纹定位，
    不排到最后；指纹找不到时追加到最后，不丢消息。"""
    store = TranscriptStore(directory=tmp_path / "transcripts")
    store.append("conv1", "天气", _two_rounds())
    inserted = [
        {"role": "user", "content": "等一下，先问个别的"},
        {"role": "assistant", "content": "你问吧。"},
    ]
    following = message_fingerprint({"role": "user", "content": "纳斯达克昨天收盘多少"})
    record = store.append("conv1", "天气", inserted, insert_before_fingerprint=following)
    contents = [m["content"] for m in record["messages"]]
    assert contents == [
        "今天天气怎么样",
        "广州晴，24 度。",
        "等一下，先问个别的",
        "你问吧。",
        "纳斯达克昨天收盘多少",
        "25,065.88，跌 0.74%。",
    ]
    record = store.append(
        "conv1", "天气", [{"role": "user", "content": "最后补一句"}],
        insert_before_fingerprint="找不到这个指纹",
    )
    assert record["messages"][-1]["content"] == "最后补一句"


def test_transcript_prune_full_to_summary_after_30_days(tmp_path):
    """38 号：满 30 天的全文变摘要——摘要优先用摘要文件里那份；
    两样都没有（没摘要、没模型）就不动，宁可多留全文、不丢数据。"""
    store = TranscriptStore(directory=tmp_path / "transcripts")
    old = datetime.now(timezone.utc) - timedelta(days=TRANSCRIPT_FULL_DAYS + 1)
    record = store.append("conv1", "天气", _two_rounds())
    path = tmp_path / "transcripts" / "conv1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["updated_at"] = old.isoformat(timespec="microseconds").replace("+00:00", "Z")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # 有现成摘要：转成摘要，消息清空。
    actions = store.prune(summary_lookup=lambda cid: "广州天气和纳指收盘。" if cid == "conv1" else "")
    assert [a["action"] for a in actions] == ["full_to_summary"]
    got = store.get("conv1")
    assert got["status"] == "summary"
    assert got["messages"] == []
    assert got["summary"] == "广州天气和纳指收盘。"
    assert record["id"] == "conv1"

    # 没摘要、没模型：不动。
    store2 = TranscriptStore(directory=tmp_path / "transcripts2")
    store2.append("conv2", "天气", _two_rounds())
    path2 = tmp_path / "transcripts2" / "conv2.json"
    data2 = json.loads(path2.read_text(encoding="utf-8"))
    data2["updated_at"] = old.isoformat(timespec="microseconds").replace("+00:00", "Z")
    path2.write_text(json.dumps(data2, ensure_ascii=False), encoding="utf-8")
    assert store2.prune() == []
    assert store2.get("conv2")["status"] == "full"


def test_transcript_prune_condenses_summary_after_a_year(tmp_path):
    """38 号：满 1 年的摘要再精简一次——调模型压短，压出来的比原来
    短才换；没模型就不动。"""
    store = TranscriptStore(directory=tmp_path / "transcripts")
    old = datetime.now(timezone.utc) - timedelta(days=TRANSCRIPT_CONDENSE_DAYS + 1)
    store.append("conv1", "天气", _two_rounds())
    path = tmp_path / "transcripts" / "conv1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "summary"
    data["messages"] = []
    data["summary"] = "这是一段很长很长的摘要，讲了广州天气和纳斯达克收盘，还有定投的事。"
    data["updated_at"] = old.isoformat(timespec="microseconds").replace("+00:00", "Z")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _short_summarizer(turns, previous_summary=None):
        return "天气和纳指。"

    actions = store.prune(summarizer=_short_summarizer)
    assert [a["action"] for a in actions] == ["summary_to_condensed"]
    got = store.get("conv1")
    assert got["status"] == "condensed"
    assert got["summary"] == "天气和纳指。"
    # 没模型时不动。
    assert store.prune() == []


# ---------- 41 号：当前对话恢复 ----------


def test_current_conversation_returns_live_messages(tmp_path):
    """41 号：正在聊的对话能恢复——服务端内存里的完整历史原样回来；
    刚开新对话还没说话时返回 None，页面显示空白欢迎页。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    summary_store = ConversationSummaryStore(
        file_path=tmp_path / "summary.json", summarizer=None
    )
    state = ChatState(
        budget_tracker=tracker,
        summary_store=summary_store,
        transcript_store=TranscriptStore(directory=tmp_path / "transcripts"),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
    )
    assert state.current_conversation() is None
    state.messages = [dict(m) for m in _two_rounds()]
    current = state.current_conversation()
    assert current is not None
    assert [m["content"] for m in current["messages"]] == [
        "今天天气怎么样",
        "广州晴，24 度。",
        "纳斯达克昨天收盘多少",
        "25,065.88，跌 0.74%。",
    ]
    state.reset()
    assert state.current_conversation() is None


def test_open_conversation_returns_full_transcript(tmp_path):
    """38 号：点开旧对话时有全文记录就恢复全文，不只最近 5 轮；
    全文已转摘要时仍是最近几轮 + transcript_status 告诉页面。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    summary_store = ConversationSummaryStore(
        file_path=tmp_path / "summary.json", summarizer=None
    )
    transcript_store = TranscriptStore(directory=tmp_path / "transcripts")
    state = ChatState(
        budget_tracker=tracker,
        summary_store=summary_store,
        transcript_store=transcript_store,
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
    )
    # 7 轮对话：摘要记录里 recent 只留最近 5 轮，全文里 7 轮都在。
    seven = []
    for i in range(7):
        seven.append({"role": "user", "content": f"第{i + 1}问"})
        seven.append({"role": "assistant", "content": f"第{i + 1}答"})
    summary_store.sync(seven)
    conv_id = summary_store.current_id
    transcript_store.append(conv_id, "七轮对话", seven)
    opened = state.open_conversation(conv_id)
    assert opened is not None
    assert len(opened["messages"]) == 14
    assert opened["messages"][0]["content"] == "第1问"
    assert opened["transcript_status"] == "full"
    # 全文转摘要后：仍是最近 5 轮，状态告诉页面。
    record = transcript_store.get(conv_id)
    record["status"] = "summary"
    record["messages"] = []
    record["summary"] = "七轮对话的摘要。"
    transcript_store._save(record)
    opened = state.open_conversation(conv_id)
    assert len(opened["messages"]) == 10
    assert opened["transcript_status"] == "summary"


# ---------- 12 号：续跑 / 作废 + 进度 + 计时日志 ----------


def test_resume_unresolved_task_runs_and_completes(tmp_path, monkeypatch):
    """12 号：UNKNOWN 的任务能续跑——跑完写历史、标 COMPLETED；
    已经跑完了的任务不用恢复，直接回绝。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    task_store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    state = ChatState(
        budget_tracker=tracker,
        task_store=task_store,
        summary_store=ConversationSummaryStore(
            file_path=tmp_path / "summary.json", summarizer=None
        ),
        transcript_store=TranscriptStore(directory=tmp_path / "transcripts"),
    )
    monkeypatch.setattr(server_module, "run_question", _fake_done_run_question)
    task = make_task(goal="查一下纳斯达克", model="deepseek-flash")
    task["status"] = "UNKNOWN"
    task_store.save(task)
    result = state.resume_unresolved_task(task["task_id"])
    assert result["ok"] is True
    assert result["reply"] == "假回答"
    assert [m["content"] for m in state.messages] == ["查一下纳斯达克", "假回答"]
    assert task_store.load(task["task_id"])["status"] == "COMPLETED"

    done = make_task(goal="你好", model="deepseek-flash")
    done["status"] = "COMPLETED"
    task_store.save(done)
    result = state.resume_unresolved_task(done["task_id"])
    assert result["ok"] is False


def test_resume_failure_restores_status(tmp_path, monkeypatch):
    """12 号：续跑没跑起来（模型调用报错）时，任务放回原来的状态，
    不挂着 RUNNING。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    task_store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    state = ChatState(
        budget_tracker=tracker,
        task_store=task_store,
        summary_store=ConversationSummaryStore(
            file_path=tmp_path / "summary.json", summarizer=None
        ),
        transcript_store=TranscriptStore(directory=tmp_path / "transcripts"),
    )

    def _boom(prompt, **kwargs):
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(server_module, "run_question", _boom)
    task = make_task(goal="查一下", model="deepseek-flash")
    task["status"] = "PAUSED"
    task_store.save(task)
    with pytest.raises(RuntimeError):
        state.resume_unresolved_task(task["task_id"])
    assert task_store.load(task["task_id"])["status"] == "PAUSED"


def test_resume_task_without_messages_runs_fresh(tmp_path, monkeypatch):
    """12 号：没存下循环消息的任务（老记录、或存盘时正好断了），
    续跑按新任务从头跑（resume_task=None），不拿空历史去问模型。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    task_store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    state = ChatState(
        budget_tracker=tracker,
        task_store=task_store,
        summary_store=ConversationSummaryStore(
            file_path=tmp_path / "summary.json", summarizer=None
        ),
        transcript_store=TranscriptStore(directory=tmp_path / "transcripts"),
    )
    seen = {}

    def _capture(prompt, budget_tracker=None, messages=None, **kwargs):
        seen.update(kwargs)
        return _fake_done_run_question(prompt, budget_tracker=budget_tracker, messages=messages, **kwargs)

    monkeypatch.setattr(server_module, "run_question", _capture)
    task = make_task(goal="查一下", model="deepseek-flash")
    task["status"] = "UNKNOWN"
    task["messages"] = []
    task_store.save(task)
    result = state.resume_unresolved_task(task["task_id"])
    assert result["ok"] is True
    assert result["reply"] == "假回答"
    assert seen.get("resume_task") is None


def test_discard_unresolved_task(tmp_path):
    """12 号：UNKNOWN / PAUSED 的任务能作废（改成 CANCELLED）；
    已经跑完了的不用作废，直接回绝。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    task_store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    state = ChatState(budget_tracker=tracker, task_store=task_store)
    task = make_task(goal="查一下", model="deepseek-flash")
    task["status"] = "UNKNOWN"
    task_store.save(task)
    assert state.discard_unresolved_task(task["task_id"]) == {"ok": True}
    assert task_store.load(task["task_id"])["status"] == "CANCELLED"
    assert state.discard_unresolved_task(task["task_id"])["ok"] is False
    assert state.discard_unresolved_task("没有这个任务")["ok"] is False


def test_active_tasks_lists_running_with_progress(tmp_path):
    """12 号：正在跑的任务带进度读出来；跑完了的不在里面。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    task_store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    state = ChatState(budget_tracker=tracker, task_store=task_store)
    running = make_task(goal="查一下新鲜事", model="deepseek-flash")
    running["status"] = "RUNNING"
    running["progress"] = {"stage": "tool", "detail": "正在执行搜索…", "round": 1, "tool_calls": 0}
    task_store.save(running)
    done = make_task(goal="你好", model="deepseek-flash")
    done["status"] = "COMPLETED"
    task_store.save(done)
    active = state.active_tasks()
    assert [t["task_id"] for t in active] == [running["task_id"]]
    assert active[0]["progress"]["detail"] == "正在执行搜索…"


def test_waiting_approval_round_logs_timing(tmp_path, monkeypatch, caplog):
    """第 0 步留的小缺口：停在确认卡上那轮的计时也要进服务端日志
    （以前只进返回、不进日志，慢在哪查不到）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    state = ChatState(
        budget_tracker=tracker,
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
    )
    monkeypatch.setattr(server_module, "run_question", _fake_waiting_run_question)
    with caplog.at_level(logging.INFO, logger="agent"):
        result = state.ask("查一下今天天气")
    assert result["mode"] == "waiting_approval"
    assert "timing |" in caplog.text


class _StubOutcome:
    """搜索服务的假返回：to_dict() 是 _validate_search_data 要的形状。"""

    def __init__(self):
        self.error = None

    def to_dict(self):
        return {
            "status": "ok",
            "results": [
                {"title": "今天天气", "url": "http://example.com", "snippet": "晴"}
            ],
            "answer": None,
            "provider": "stub",
        }


class _StubSearchService:
    """占位搜索服务：search 一次就回一条结果，不联网。"""

    def __init__(self):
        self.calls = 0

    def search(self, query, kind="normal", max_results=5):
        self.calls += 1
        return _StubOutcome()


class _ToolCallProvider:
    """先调一次搜索工具、再直接回答的假模型（进度测试用）。"""

    def __init__(self, api_key=None, budget_tracker=None, model=None):
        self.model = model or "deepseek-flash"
        self.calls = 0

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        return {"text": "译文", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "text": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "今天天气"}',
                        },
                    }
                ],
                "reasoning_content": None,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "cost_usd": 0.001,
                "model": self.model,
            }
        return {
            "text": "今天晴。",
            "tool_calls": [],
            "reasoning_content": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost_usd": 0.001,
            "model": self.model,
        }

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None):
        return {"text": "今天晴。", "usage": None, "cost_usd": 0.0, "model": self.model}


def test_agent_loop_writes_progress(tmp_path):
    """12 号：循环跑起来时，进度写进任务记录——问模型第几轮、
    正在执行哪个工具，前端轮询能看到；跑完了进度清掉。"""
    store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    seen = []
    orig_save = store.save

    def _recording_save(task):
        seen.append(dict(task.get("progress") or {}))
        return orig_save(task)

    store.save = _recording_save
    loop = AgentLoop(
        _ToolCallProvider(),
        gateway=ToolGateway(),
        task_store=store,
        search_service=_StubSearchService(),
    )
    outcome = loop.run("查一下今天天气")
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "今天晴。"
    details = [p.get("detail", "") for p in seen if p]
    assert any("第 1 轮" in d for d in details)
    assert any("搜索" in d for d in details)
    assert any("等模型回话" in d for d in details)
    final = store.load(outcome["task"]["task_id"])
    assert final["status"] == "COMPLETED"
    assert not final.get("progress")


def test_resumed_crash_retries_interrupted_search(tmp_path):
    """12 号：崩溃时正在跑的搜索，续跑要重调一次、拿到结果再回答，
    不能只填一句“跳过了”就答“没查到”（真调 2026-11-12 抓到的毛病）。
    有确认卡的恢复不受影响：那条还是“跳过了”，因为确认流程里
    同轮另一项本来就是跳过的。"""
    store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    search_service = _StubSearchService()
    loop = AgentLoop(
        _ToolCallProvider(),
        gateway=ToolGateway(),
        task_store=store,
        search_service=search_service,
    )
    # 崩溃现场：模型已经发了搜索调用、结果还没回来服务就停了。
    # 任务里只有调用、没有结果、没有确认卡。
    crashed = make_task(goal="查一下今天广州天气怎么样", model="deepseek-flash")
    crashed["status"] = "UNKNOWN"
    crashed["messages"] = [
        {"role": "system", "content": "你是十二。"},
        {"role": "user", "content": "查一下今天广州天气怎么样"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_crash1",
                    "function": {"name": "search", "arguments": '{"query": "广州天气"}'},
                }
            ],
        },
    ]
    crashed["tool_calls"] = []
    outcome = loop.run("查一下今天广州天气怎么样", resume_task=crashed)
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "今天晴。"
    assert search_service.calls == 1
    resumed_messages = store.load(outcome["task"]["task_id"])["messages"]
    placeholders = [
        m["content"]
        for m in resumed_messages
        if m.get("role") == "tool" and m.get("tool_call_id") == "call_crash1"
    ]
    assert placeholders and "重新调一次" in placeholders[0]
    assert "跳过了" not in placeholders[0]


# ---------- 端口测试：新接口 ----------


def test_current_endpoint_returns_live_messages(batch2_server, monkeypatch):
    """41 号：/api/conversations/current 返回正在聊的对话；
    还没聊时 messages 为空。"""
    port, state = batch2_server
    status, body = _request(port, "GET", "/api/conversations/current")
    assert status == 200
    assert body["messages"] == []
    monkeypatch.setattr(server_module, "run_question", _fake_done_run_question)
    status, body = _request(port, "POST", "/api/chat", {"message": "你好"})
    assert status == 200
    assert body["mode"] == "real"
    status, body = _request(port, "GET", "/api/conversations/current")
    assert status == 200
    assert [m["content"] for m in body["messages"]] == ["你好", "假回答"]


def test_unresolved_and_active_endpoints(batch2_server):
    """12 号：/api/tasks/unresolved 列出没交代的任务，
    /api/tasks/active 列出正在跑的任务 + 进度。"""
    port, state = batch2_server
    task = make_task(goal="查一下纳斯达克", model="deepseek-flash")
    task["status"] = "UNKNOWN"
    state.task_store.save(task)
    running = make_task(goal="查一下新鲜事", model="deepseek-flash")
    running["status"] = "RUNNING"
    running["progress"] = {"stage": "model", "detail": "正在问模型（第 2 轮）…", "round": 2, "tool_calls": 1}
    state.task_store.save(running)
    status, body = _request(port, "GET", "/api/tasks/unresolved")
    assert status == 200
    assert [t["task_id"] for t in body["tasks"]] == [task["task_id"]]
    status, body = _request(port, "GET", "/api/tasks/active")
    assert status == 200
    assert [t["task_id"] for t in body["tasks"]] == [running["task_id"]]
    assert body["tasks"][0]["progress"]["detail"] == "正在问模型（第 2 轮）…"


def test_unresolved_payload_shows_tool_call_states(batch2_server):
    """GPT 第一轮审查 P1 第 4 条：未解决任务的返回里要带每条工具调用的
    状态——BOOS 得看得出哪条执行过了结果丢了（executing）、哪条结果
    已保存（ok / failed）、哪条还没执行（waiting_approval），
    光说任务“结果未知”不够。"""
    port, state = batch2_server
    task = make_task(goal="查一下纳斯达克", model="deepseek-flash")
    task["status"] = "UNKNOWN"
    task["tool_calls"] = [
        {"tool": "search", "tool_call_id": "call_1", "status": "ok"},
        {"tool": "search", "tool_call_id": "call_2", "status": "executing"},
        {"tool": "write_file", "tool_call_id": "call_3", "status": "waiting_approval"},
    ]
    state.task_store.save(task)
    status, body = _request(port, "GET", "/api/tasks/unresolved")
    assert status == 200
    assert [t["task_id"] for t in body["tasks"]] == [task["task_id"]]
    assert body["tasks"][0]["tool_calls"] == [
        {"tool": "search", "status": "ok"},
        {"tool": "search", "status": "executing"},
        {"tool": "write_file", "status": "waiting_approval"},
    ]


def test_discard_endpoint(batch2_server):
    """12 号：POST /api/tasks/<id>/discard 作废没交代的任务；
    已经跑完了的回 409。"""
    port, state = batch2_server
    task = make_task(goal="查一下", model="deepseek-flash")
    task["status"] = "PAUSED"
    state.task_store.save(task)
    status, body = _request(port, "POST", f"/api/tasks/{task['task_id']}/discard")
    assert status == 200
    assert body == {"ok": True}
    assert state.task_store.load(task["task_id"])["status"] == "CANCELLED"
    status, body = _request(port, "POST", f"/api/tasks/{task['task_id']}/discard")
    assert status == 409


def test_resume_endpoint(batch2_server, monkeypatch):
    """12 号：POST /api/tasks/<id>/resume 续跑没交代的任务，
    跑完返回回答、任务标 COMPLETED；找不到的任务回 409。"""
    port, state = batch2_server
    monkeypatch.setattr(server_module, "run_question", _fake_done_run_question)
    task = make_task(goal="查一下纳斯达克", model="deepseek-flash")
    task["status"] = "UNKNOWN"
    state.task_store.save(task)
    status, body = _request(port, "POST", f"/api/tasks/{task['task_id']}/resume")
    assert status == 200
    assert body["ok"] is True
    assert body["reply"] == "假回答"
    assert state.task_store.load(task["task_id"])["status"] == "COMPLETED"
    status, body = _request(port, "POST", "/api/tasks/task_not_exist/resume")
    assert status == 409


def test_current_endpoint_html_has_unresolved_box_and_progress_poll():
    """12 号 + 41 号：页面里有卡住任务的盒子样式、当前对话恢复和
    进度轮询的代码（静态检查，防手滑删掉）。"""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    assert ".unresolved {" in html
    assert "refreshCurrentConversation" in html
    assert "refreshUnresolvedTasks" in html
    assert "/api/conversations/current" in html
    assert "/api/tasks/unresolved" in html
    assert "/api/tasks/active" in html
    assert "pollProgress" in html
    # GPT 第一轮审查 P1 第 4 条：页面上要把每条工具调用的状态
    # 也显示出来（执行中结果未知 / 结果已保存 / 还没执行）。
    assert "unresolvedToolStatusLabel" in html
    assert "工具：" in html
