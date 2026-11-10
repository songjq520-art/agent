# tests/test_conversation_summary.py（对话摘要 + 最近对话列表，2026-11-10）
# 覆盖 BOOS 定的摘要规则和最近列表：
#   - 到上限（32,000 token）触发总结，最近 5 轮原样保留；
#   - 对话结束时（CLI 退出、网页开新对话）也总结一次；
#   - 新对话进列表、点一条恢复上下文、重启后列表还在；
#   - 摘要拼进提问、不喂给 mem0。
# 全部离线：总结用假的摘要器（stub），不联网、不产生费用；
# 需要真实模型调用的场景只测 make_deepseek_summarizer 的接线
# （用 mock 的 requests.post，不花钱）。

import http.client
import json
import threading
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker
from conftest import make_fake_store
from main import chat_loop, run_question
from memory.conversation_summary import (
    CONVERSATION_SUMMARY_TOKEN_LIMIT,
    ConversationSummaryStore,
    build_summary_prompt,
    estimate_tokens,
    make_deepseek_summarizer,
    summary_block,
)
from web.server import ChatState, ServerConfig, create_server


class StubSummarizer:
    """假的摘要器： deterministic，不调模型；记下每次调用，供核对。"""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, turns, previous_summary=None):
        self.calls.append((list(turns), previous_summary))
        if self.fail:
            return None
        topics = "、".join(
            (t.get("content") or "")[:6] for t in turns if t.get("role") == "user"
        )
        base = f"摘要[{topics}]"
        if previous_summary:
            return previous_summary + " | " + base
        return base


def _messages(n_rounds, topic="今天天气真好"):
    """造 n 轮短对话（每轮一问一答）。"""
    out = []
    for i in range(n_rounds):
        out.append({"role": "user", "content": f"{topic}第{i + 1}句"})
        out.append({"role": "assistant", "content": f"好的，第{i + 1}个回答"})
    return out


def _long_messages(n_rounds, chars=60):
    """造 n 轮长对话（每句 chars 个字），用来快速超过一个很小的 token 上限。"""
    out = []
    for i in range(n_rounds):
        out.append({"role": "user", "content": f"问题{i + 1}：" + "啊" * chars})
        out.append({"role": "assistant", "content": f"回答{i + 1}：" + "嗯" * chars})
    return out


def _store(tmp_path, summarizer=None, **kwargs):
    """造一个指到临时文件的摘要库（不碰真实的 memory/conversation_summary.json）。"""
    return ConversationSummaryStore(
        file_path=tmp_path / "conversation_summary.json",
        summarizer=summarizer,
        **kwargs,
    )


def _fake_ok_response(text="你好，我是 DeepSeek。", prompt_tokens=100, completion_tokens=50):
    """模拟的成功 API 返回（带 usage），和 test_conversation.py 里同一个做法。"""
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


# ---------- 估算和提示块 ----------

def test_estimate_tokens_chinese_chars_count_double_and_english_words():
    """估算：中文按字×2，英文按词×1（和 main._estimate_tokens 同一套）。"""
    assert estimate_tokens("你好") == 4
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好 hello") == 4 + 1


def test_summary_block_empty_returns_empty_and_with_text_labels_summary():
    """摘要提示块：没摘要时是空串；有时标明是梗概、不是逐字记录。"""
    assert summary_block("") == ""
    assert summary_block("   ") == ""
    block = summary_block("之前聊过天气")
    assert "之前聊过天气" in block
    assert "摘要" in block
    assert "不是逐字记录" in block


def test_build_summary_prompt_includes_turns_and_previous_summary():
    """总结提示：含对话记录；有旧摘要时要求合并、不编造。"""
    prompt = build_summary_prompt(_messages(1), previous_summary="旧摘要")
    assert "BOOS：今天天气真好第1句" in prompt
    assert "十二：好的，第1个回答" in prompt
    assert "旧摘要" in prompt
    assert "不要编造" in prompt


def test_build_summary_prompt_redacts_secrets_in_turns():
    """隐私：轮次里的密码/密钥不进提示（只留占位符），和 mem0 提取同一套脱敏。"""
    turns = [
        {"role": "user", "content": "我的密码是 hunter2ABCdef12345678，帮我记一下"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "我的 API 密钥是 sk-testkey1234567890abcdefgh"},
        {"role": "user", "content": "令牌 Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4z 也别外传"},
    ]
    prompt = build_summary_prompt(turns)
    assert "hunter2ABCdef12345678" not in prompt
    assert "sk-testkey1234567890abcdefgh" not in prompt
    assert "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4z" not in prompt
    assert "【已隐藏敏感信息】" in prompt


def test_build_summary_prompt_redacts_previous_summary():
    """隐私：旧摘要里带的秘密也要脱敏——修复前生成的旧摘要可能已经带了秘密，
    下次合并时还会再发出去，不能只脱敏新轮次。"""
    prompt = build_summary_prompt(
        _messages(1),
        previous_summary="之前聊过广州出差，我的密码是 hunter2ABCdef12345678",
    )
    assert "hunter2ABCdef12345678" not in prompt
    assert "【已隐藏敏感信息】" in prompt
    assert "广州出差" in prompt


def test_build_summary_prompt_keeps_normal_content():
    """脱敏不伤正常内容：话题、决定、人名/地名原样留在提示里。"""
    turns = [
        {"role": "user", "content": "我喜欢喝咖啡，明天去广州出差，决定坐高铁"},
        {"role": "assistant", "content": "好的，广州出差坐高铁"},
    ]
    prompt = build_summary_prompt(turns)
    assert "我喜欢喝咖啡，明天去广州出差，决定坐高铁" in prompt
    assert "广州出差坐高铁" in prompt


# ---------- 存取：重启后还在 ----------

def test_store_save_load_round_trip_restart(tmp_path):
    """重启后能读回：记一笔、建新实例，列表和标题都在。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    store.sync(_messages(2))
    assert store.list_conversations()[0]["title"] == "今天天气真好第1句"

    store2 = _store(tmp_path, summarizer=stub)
    items = store2.list_conversations()
    assert len(items) == 1
    assert items[0]["title"] == "今天天气真好第1句"
    assert items[0]["started_at"]
    record = store2.get_conversation(items[0]["id"])
    assert len(record["recent"]) == 4


def test_corrupt_file_and_unknown_version_start_empty(tmp_path):
    """摘要文件坏了、版本不认识时按空处理，不 crash。"""
    bad = tmp_path / "conversation_summary.json"
    bad.write_text("{不是 json", encoding="utf-8")
    assert ConversationSummaryStore(file_path=bad).list_conversations() == []
    bad.write_text(json.dumps({"version": 99, "conversations": []}), encoding="utf-8")
    assert ConversationSummaryStore(file_path=bad).list_conversations() == []


def test_v1_single_summary_file_converts_to_one_record(tmp_path):
    """v1 单条摘要文件读进来转成一条记录（标题叫“之前的对话”）。"""
    old = tmp_path / "conversation_summary.json"
    old.write_text(
        json.dumps(
            {"version": 1, "summary": "之前聊过天气", "updated_at": "2026-11-10T00:00:00Z"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = ConversationSummaryStore(file_path=old)
    items = store.list_conversations()
    assert len(items) == 1
    assert items[0]["title"] == "之前的对话"
    assert store.current_summary() == "之前聊过天气"


# ---------- 列表：新对话进来、标题、最近几轮 ----------

def test_new_conversation_appears_in_list_after_first_turn(tmp_path):
    """新对话进列表：第一轮后出现，标题取第一句用户话，时间有值。"""
    store = _store(tmp_path, summarizer=StubSummarizer())
    store.sync(_messages(1))
    items = store.list_conversations()
    assert len(items) == 1
    assert items[0]["title"] == "今天天气真好第1句"
    assert items[0]["started_at"] and items[0]["updated_at"]


def test_sync_keeps_only_last_five_rounds_as_recent(tmp_path):
    """最近几轮只留 5 轮：12 轮对话记下来，recent 是最后 10 条消息。"""
    store = _store(tmp_path, summarizer=StubSummarizer())
    store.sync(_messages(12))
    record = store.get_conversation(store.list_conversations()[0]["id"])
    assert len(record["recent"]) == 10
    assert record["recent"][0]["content"] == "今天天气真好第8句"
    assert record["recent"][-1]["content"] == "好的，第12个回答"


def test_title_falls_back_to_summary_first_sentence_then_default(tmp_path):
    """标题没有用户话时取摘要头一句；再没有叫“新对话”。"""
    store = _store(tmp_path, summarizer=StubSummarizer())
    store.sync([{"role": "assistant", "content": "我是十二。"}])
    # 只有助手消息、没有用户话、没有摘要：标题是默认的“新对话”。
    assert store.list_conversations()[0]["title"] == "新对话"
    # 有摘要时取摘要头一句。
    store2 = _store(tmp_path, summarizer=StubSummarizer())
    store2.begin_conversation()
    store2.flush([{"role": "user", "content": "广州天气怎么样？明天出门。"}])
    assert store2.list_conversations()[0]["title"] == "广州天气怎么样"


def test_title_truncates_long_first_sentence(tmp_path):
    """标题太长时截掉加…，不超过 24 字 + 省略号。"""
    store = _store(tmp_path, summarizer=StubSummarizer())
    store.sync([{"role": "user", "content": "这是一个特别特别长的第一句话，超过二十四个字了还要继续说下去"}])
    title = store.list_conversations()[0]["title"]
    assert title.endswith("…")
    assert len(title) == 24 + 1


def test_list_capped_at_twenty_oldest_dropped(tmp_path):
    """列表最多 20 条：第 21 段对话进来时，最旧的那条被删掉。"""
    store = _store(tmp_path, summarizer=StubSummarizer())
    for i in range(21):
        store.begin_conversation()
        store.sync([{"role": "user", "content": f"第{i + 1}段对话"}])
    items = store.list_conversations()
    assert len(items) == 20
    titles = [item["title"] for item in items]
    assert "第1段对话" not in titles
    assert "第21段对话" in titles
    assert items[0]["title"] == "第21段对话"


# ---------- 压缩：到上限才总结、最近 5 轮原样 ----------

def test_maybe_compact_triggers_over_limit_keeps_last_five_rounds(tmp_path):
    """到上限触发总结：旧的被总结进摘要，最近 5 轮原样留着。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub, token_limit=100)
    messages = _long_messages(8)  # 16 条消息，每条约 120 token，远超 100。
    assert store.maybe_compact(messages) is True
    assert len(stub.calls) == 1
    # 总结的是除最近 5 轮之外的旧消息（前 6 条）。
    assert len(stub.calls[0][0]) == 6
    # 历史只剩最近 5 轮（10 条），原样保留。
    assert len(messages) == 10
    assert messages[0]["content"].startswith("问题4：")
    assert messages[-1]["content"].startswith("回答8：")
    # 摘要进了正在聊的记录，存硬盘了。
    assert "摘要[" in store.current_summary()
    assert store.list_conversations()[0]["title"].startswith("问题1：")


def test_maybe_compact_below_limit_does_nothing(tmp_path):
    """没到上限不总结：短对话原样不动，摘要器一次也不调。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    messages = _messages(2)
    assert store.maybe_compact(messages) is False
    assert stub.calls == []
    assert len(messages) == 4


def test_maybe_compact_without_summarizer_does_nothing(tmp_path):
    """没配摘要器时不总结（比如没 Key 的环境），历史不动。"""
    store = _store(tmp_path, summarizer=None, token_limit=1)
    messages = _long_messages(8)
    assert store.maybe_compact(messages) is False
    assert len(messages) == 16


def test_maybe_compact_summarizer_failure_keeps_history(tmp_path):
    """总结失败时历史不动、旧摘要不动，下一轮再试。"""
    store = _store(tmp_path, summarizer=StubSummarizer(fail=True), token_limit=100)
    messages = _long_messages(8)
    assert store.maybe_compact(messages) is False
    assert len(messages) == 16
    assert store.current_summary() == ""


def test_default_token_limit_is_32000():
    """默认上限是 BOOS 定的 32,000 token。"""
    assert CONVERSATION_SUMMARY_TOKEN_LIMIT == 32000


# ---------- 结束时总结：短对话也要总结 ----------

def test_flush_summarizes_short_conversation(tmp_path):
    """结束时总结：短对话（没到上限）也总结进摘要、进列表。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    messages = _messages(2)
    assert store.flush(messages) is True
    assert len(stub.calls) == 1
    assert "摘要[" in store.current_summary()
    record = store.get_conversation(store.list_conversations()[0]["id"])
    assert len(record["recent"]) == 4
    assert record["turns_summarized"] == 4


def test_flush_empty_messages_does_nothing(tmp_path):
    """空消息不总结，摘要器不调、列表还是空的。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    assert store.flush([]) is False
    assert stub.calls == []
    assert store.list_conversations() == []


def test_flush_same_messages_twice_summarizes_once(tmp_path):
    """同一份消息重复 flush 只总结一次（指纹拦着）。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    messages = _messages(2)
    assert store.flush(messages) is True
    assert store.flush(messages) is False
    assert len(stub.calls) == 1


def test_new_conversation_prompt_block_uses_latest_summary(tmp_path):
    """新对话读回最新摘要：上一段 flush 后、开新对话，提示块里有它。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    store.flush(_messages(2, topic="广州天气"))
    store.begin_conversation()
    block = store.prompt_block()
    assert "广州天气" in block
    assert "摘要" in block


# ---------- 回到旧对话 ----------

def test_open_conversation_restores_summary_and_recent(tmp_path):
    """点一条回旧对话：恢复它的摘要 + 最近几轮，找不到返回 None。"""
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    store.flush(_messages(2, topic="广州天气"))
    first_id = store.list_conversations()[0]["id"]
    store.begin_conversation()
    store.sync(_messages(1, topic="改名十三"))
    assert store.list_conversations()[0]["title"].startswith("改名十三")

    record = store.open_conversation(first_id)
    assert record is not None
    assert "广州天气" in record["summary"]
    assert len(record["recent"]) == 4
    # 正在聊的换成旧的了：提示块用的是旧对话的摘要。
    assert "广州天气" in store.prompt_block()
    assert store.open_conversation("不存在的id") is None


def test_now_iso_strictly_increases_when_clock_frozen(tmp_path, monkeypatch):
    """时钟钉死不动时，_now_iso() 每次发的仍严格递增（Windows 上粒度粗会撞时间）。"""
    import memory.conversation_summary as cs

    fixed = datetime(2026, 11, 10, 3, 0, 0, tzinfo=timezone.utc)

    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return fixed

    monkeypatch.setattr(cs, "datetime", FrozenClock)
    monkeypatch.setattr(cs, "_NOW_LAST", None, raising=False)
    values = [cs._now_iso() for _ in range(5)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_frozen_clock_still_orders_new_conversation_first(tmp_path, monkeypatch):
    """时钟钉死不动（模拟 Windows 上两次读到同一个时间戳），新对话仍排列表第一。"""
    import memory.conversation_summary as cs

    fixed = datetime(2026, 11, 10, 3, 0, 0, tzinfo=timezone.utc)

    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return fixed

    monkeypatch.setattr(cs, "datetime", FrozenClock)
    monkeypatch.setattr(cs, "_NOW_LAST", None, raising=False)
    stub = StubSummarizer()
    store = _store(tmp_path, summarizer=stub)
    store.flush(_messages(2, topic="广州天气"))
    store.begin_conversation()
    store.sync(_messages(1, topic="改名十三"))
    assert store.list_conversations()[0]["title"].startswith("改名十三")


# ---------- 摘要器接线 ----------

def test_make_deepseek_summarizer_without_key_returns_none(monkeypatch):
    """没 Key 时摘要器返回 None（不调用、不花钱），调用方按跳过处理。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    summarizer = make_deepseek_summarizer()
    assert summarizer(_messages(1), None) is None


def test_make_deepseek_summarizer_calls_provider(monkeypatch, tmp_path):
    """有 Key 时摘要器走 DeepSeek：提示里有对话记录，返回模型写的摘要。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    summarizer = make_deepseek_summarizer(budget_tracker=tracker)
    with patch(
        "providers.deepseek.requests.post",
        return_value=_fake_ok_response(text="这段对话聊了天气。"),
    ) as mock_post:
        result = summarizer(_messages(1), None)
    assert result == "这段对话聊了天气。"
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert "今天天气真好第1句" in sent[0]["content"]
    assert "对话记录员" in sent[0]["content"]


# ---------- run_question：摘要进提示、不进 mem0 ----------

def test_run_question_injects_summary_but_not_into_mem0(monkeypatch, tmp_path):
    """提问时摘要拼进发给模型的问题；存 mem0 的消息里没有摘要原文。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    fake_store = make_fake_store(tmp_path=tmp_path, budget_tracker=tracker)
    stub = StubSummarizer()
    summary_store = _store(tmp_path, summarizer=stub)
    summary_store.flush(_messages(1, topic="之前聊过广州天气"))
    summary_text = summary_store.current_summary()
    # 假摘要器只留每句用户话的前 6 个字，摘要里有“广州”就够了。
    assert "广州" in summary_text

    with patch(
        "providers.deepseek.requests.post", return_value=_fake_ok_response()
    ) as mock_post:
        outcome = run_question(
            "我叫BOOS",
            budget_tracker=tracker,
            messages=[],
            mem0_store=fake_store,
            summary_store=summary_store,
        )
    assert outcome["mode"] == "real"
    sent = mock_post.call_args.kwargs["json"]["messages"]
    sent_text = "\n".join(m["content"] for m in sent)
    assert "之前对话的摘要" in sent_text
    assert "广州" in sent_text
    # mem0 里存的记忆没有摘要原文（FakeMemory 的提取只认原话）。
    stored = fake_store.get_all()
    assert all("广州" not in (item.get("text") or "") for item in stored)


# ---------- chat_loop：退出时总结 ----------

def test_chat_loop_exit_flushes_summary_to_file(monkeypatch, tmp_path):
    """CLI 退出时总结：两轮对话后 exit，文件里有这段对话的记录；问答只调一次 API。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _store(tmp_path, summarizer=StubSummarizer())
    with (
        patch("builtins.input", side_effect=["今天天气真好", "exit"]),
        patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post,
    ):
        assert chat_loop(budget_tracker=tracker, summary_store=store) == 0
        # 只有问答调了一次 API（摘要器是 stub，不走网络）。
        assert mock_post.call_count == 1
    items = store.list_conversations()
    assert len(items) == 1
    assert items[0]["title"] == "今天天气真好"
    assert "摘要[" in store.current_summary()


# ---------- 网页：reset / open / 列表接口 ----------

def test_chat_state_reset_flushes_and_lists(tmp_path):
    """网页开新对话：先总结保存当前对话，再清空历史，列表里有它。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _store(tmp_path, summarizer=StubSummarizer())
    state = ChatState(budget_tracker=tracker, summary_store=store)
    state.messages = _messages(2, topic="广州天气")
    state.reset()
    assert state.messages == []
    items = state.list_conversations()
    assert len(items) == 1
    assert items[0]["title"] == "广州天气第1句"
    assert "摘要[" in store.current_summary()


def test_chat_state_open_restores_messages(tmp_path):
    """网页回到旧对话：恢复那段的最近几轮，接着聊。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _store(tmp_path, summarizer=StubSummarizer())
    state = ChatState(budget_tracker=tracker, summary_store=store)
    state.messages = _messages(2, topic="广州天气")
    state.reset()
    first_id = state.list_conversations()[0]["id"]
    state.messages = _messages(1, topic="改名十三")
    state.reset()

    result = state.open_conversation(first_id)
    assert result is not None
    assert result["title"] == "广州天气第1句"
    assert len(result["messages"]) == 4
    assert result["messages"][0]["content"] == "广州天气第1句"
    assert state.messages == result["messages"]
    assert state.open_conversation("不存在的id") is None


def _fake_success_run_question(prompt, budget_tracker=None, messages=None, **kwargs):
    """假的 run_question：直接返回一轮成功的假回答，不调用真实模型。"""
    return {
        "mode": "real",
        "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
        "result": {"text": "假回答", "model": "deepseek-flash", "cost_usd": 0.0001, "usage": {"total_tokens": 10}},
        "message": "真实调用完成。",
        "model_status": {},
    }


def _start_summary_server(tmp_path):
    """启动一个带摘要库（假摘要器）的测试服务，返回 (server, port, state, store)。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _store(tmp_path, summarizer=StubSummarizer())
    config = ServerConfig(host="127.0.0.1", port=0)
    server, state = create_server(
        config=config, budget_tracker=tracker, summary_store=store
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], state, store


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


def test_http_conversations_list_and_open(monkeypatch, tmp_path):
    """HTTP 端到端：聊两段对话，列表里有两条，点一条能恢复上下文。"""
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    server, port, state, store = _start_summary_server(tmp_path)
    try:
        # 第一段对话：聊一句。
        status, data = _request(port, "POST", "/api/chat", {"message": "你好"})
        assert status == 200
        # 开新对话：第一段被总结保存。
        status, _ = _request(port, "POST", "/api/reset")
        assert status == 200
        # 第二段对话：聊一句。
        status, data = _request(port, "POST", "/api/chat", {"message": "再见"})
        assert status == 200
        # 列表里有两条，最新的在顶部。
        status, data = _request(port, "GET", "/api/conversations")
        assert status == 200
        items = data["conversations"]
        assert len(items) == 2
        assert items[0]["title"] == "再见"
        assert items[1]["title"] == "你好"
        first_id = items[1]["id"]
        # 点第一段：恢复它的最近几轮，接着聊。
        status, data = _request(port, "POST", "/api/conversations/open", {"id": first_id})
        assert status == 200
        assert data["title"] == "你好"
        assert [m["content"] for m in data["messages"]] == ["你好", "假回答"]
        assert state.messages == data["messages"]
        # 再聊一句：记回同一条记录，它回到列表顶部。
        status, _ = _request(port, "POST", "/api/chat", {"message": "又聊了一句"})
        assert status == 200
        status, data = _request(port, "GET", "/api/conversations")
        assert data["conversations"][0]["id"] == first_id
    finally:
        server.shutdown()
        server.server_close()


def test_http_conversations_open_bad_requests(monkeypatch, tmp_path):
    """HTTP：open 没给 id 回 400，找不到的 id 回 404；没对话时列表是空的。"""
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    server, port, state, store = _start_summary_server(tmp_path)
    try:
        status, data = _request(port, "GET", "/api/conversations")
        assert status == 200
        assert data["conversations"] == []
        status, data = _request(port, "POST", "/api/conversations/open", {})
        assert status == 400
        status, data = _request(port, "POST", "/api/conversations/open", {"id": "没有这条"})
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()


# ---------- 摘要卡片：点开旧对话 / 开新对话时顶部看得见 ----------

def test_chat_state_current_summary_after_reset(tmp_path):
    """开新对话后，current_summary() 是刚结束那段的摘要（和存下来的一致）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _store(tmp_path, summarizer=StubSummarizer())
    state = ChatState(budget_tracker=tracker, summary_store=store)
    assert state.current_summary() == ""
    state.messages = _messages(2, topic="广州天气")
    state.reset()
    assert state.messages == []
    # 刚结束的那段对话的摘要：和存下来的那份一字不差。
    assert state.current_summary() == store.current_summary()
    assert "广州天气" in state.current_summary()


def test_chat_state_current_summary_without_store_is_empty(tmp_path):
    """没启用摘要库时，current_summary() 是空串（页面不显示卡片）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    state = ChatState(budget_tracker=tracker, summary_store=None)
    assert state.current_summary() == ""


def test_chat_state_open_without_summary_returns_empty(tmp_path):
    """没摘要的对话点开时 summary 是空串（页面不显示卡片）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _store(tmp_path, summarizer=None)  # 没配摘要器：flush 不产生摘要。
    state = ChatState(budget_tracker=tracker, summary_store=store)
    state.messages = _messages(1, topic="改名十三")
    store.sync(state.messages)
    conv_id = state.list_conversations()[0]["id"]
    result = state.open_conversation(conv_id)
    assert result is not None
    assert result["summary"] == ""
    assert [m["content"] for m in result["messages"]] == ["改名十三第1句", "好的，第1个回答"]


def test_http_reset_returns_previous_summary(monkeypatch, tmp_path):
    """HTTP：开新对话时 reset 回包里带上上一段的摘要（卡片就显示这份）。"""
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    server, port, state, store = _start_summary_server(tmp_path)
    try:
        # 还没聊过就开新对话：没有摘要，回空串。
        status, data = _request(port, "POST", "/api/reset")
        assert status == 200
        assert data["ok"] is True
        assert data["summary"] == ""
        # 聊一句、再开新对话：回包里是上一段的摘要，和存下来的一致。
        status, _ = _request(port, "POST", "/api/chat", {"message": "你好"})
        assert status == 200
        status, data = _request(port, "POST", "/api/reset")
        assert status == 200
        assert data["summary"] == store.current_summary()
        assert "你好" in data["summary"]
    finally:
        server.shutdown()
        server.server_close()


def test_http_open_returns_stored_summary_exact(monkeypatch, tmp_path):
    """HTTP：点开旧对话时 open 回包里的摘要和存下来的一字不差。"""
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    server, port, state, store = _start_summary_server(tmp_path)
    try:
        status, _ = _request(port, "POST", "/api/chat", {"message": "你好"})
        assert status == 200
        status, _ = _request(port, "POST", "/api/reset")
        assert status == 200
        status, data = _request(port, "GET", "/api/conversations")
        first_id = data["conversations"][0]["id"]
        stored = store.get_conversation(first_id)["summary"]
        assert stored  # 存下来的摘要不是空的。
        status, data = _request(port, "POST", "/api/conversations/open", {"id": first_id})
        assert status == 200
        assert data["summary"] == stored
        assert "你好" in data["summary"]
    finally:
        server.shutdown()
        server.server_close()


def test_served_html_contains_summary_card(monkeypatch, tmp_path):
    """下发的页面里有摘要卡片的样式和逻辑：点开/开新对话时用 data.summary 显示。"""
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    server, port, state, store = _start_summary_server(tmp_path)
    try:
        status, html = _request(port, "GET", "/")
        assert status == 200
        assert "summary-card" in html
        assert "之前聊过" in html
        assert "renderSummaryCard" in html
        # 两处都用回包里的 summary：点开旧对话、开新对话。
        assert html.count("data.summary") >= 2
    finally:
        server.shutdown()
        server.server_close()
