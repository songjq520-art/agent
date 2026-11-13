# tests/test_memory_receipts_52.py（52 号：落袋慢 + 回执；55 号：“记住”卡得太死）
# 作用：验证记忆写入链的可靠性——提取失败不能悄悄丢、失败的句子
#       下一轮要重试、后台 outbox 要给每轮一个 committed / pending /
#       failed 回执；以及 add_memory 的触发条件放宽到“明确表达的
#       事实/偏好/决定/习惯”，不只认“记住”两个字。
# 全部离线：FakeMemory + 注入失败的假对象，不调真实 API、不花钱。
# 关键回归（2026-11-12 实测）：
#   52 号：BOOS 说“我爱吃拉面”，十二回“记下了”，面板里却没有——
#     提取在后台排队、失败了也没人知道，模型却先说了“记下了”。
#   55 号：BOOS 说“我想以后每天7点起床”（没说“记住”），十二说
#     按规矩得明确说“记住”才能存——触发条件写死在工具描述和第 8 条里。

import pytest

from conftest import FakeMemory, make_fake_store
from memory.outbox import MemoryOutbox


class FlakyFakeMemory(FakeMemory):
    """提取会炸几次的假 mem0：add 前 _fail_times 次抛错（模拟 DeepSeek /
    网络抖一下），之后正常存。只在 infer=True（提取路）炸，
    infer=False（原文直存）不炸。"""

    def __init__(self, fail_times=1):
        super().__init__()
        self._fail_times = fail_times
        self.add_failures = 0

    def add(self, messages, user_id=None, infer=True, metadata=None):
        if infer and self._fail_times > 0:
            self._fail_times -= 1
            self.add_failures += 1
            raise RuntimeError("模拟提取失败：mem0.add 炸了")
        return super().add(messages, user_id=user_id, infer=infer, metadata=metadata)


def _texts(store):
    return store._memory.all_texts()


# ---------- 52 号：提取失败要能区分、能重试、不丢 ----------


def test_52_extraction_failure_is_flagged_not_silent(tmp_path):
    """52 号：提取炸了不能装成“没什么可存”——返回里要带 extraction_failed，
    失败的那句不能被记成已处理。"""
    store = make_fake_store(tmp_path, memory=FlakyFakeMemory(fail_times=1))
    result = store.add_turn([{"role": "user", "content": "我喜欢吃拉面。"}])
    assert result.get("pending") is True
    # 下一轮提交上一轮：提取炸一次。
    result = store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    assert result.get("extraction_failed") is True
    assert "我喜欢吃拉面。" in (result.get("failed_messages") or [])
    # 拉面还没进库（这次没存成），但也没被记成“已处理”。
    assert not any("拉面" in text for text in _texts(store))


def test_52_failed_turn_retried_next_round_and_lands(tmp_path):
    """52 号：失败的那句跟着下一轮再试一次——服务好了就落袋，不丢。"""
    store = make_fake_store(tmp_path, memory=FlakyFakeMemory(fail_times=1))
    store.add_turn([{"role": "user", "content": "我喜欢吃拉面。"}])
    result = store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    assert result.get("extraction_failed") is True
    # 再来一轮：上一轮失败的拉面跟着本轮进待定区，提交时重试，落袋。
    store.add_turn([{"role": "user", "content": "明天想去跑步。"}])
    store.flush_pending()
    assert any("拉面" in text for text in _texts(store))


def test_52_fingerprint_not_recorded_on_failure(tmp_path):
    """52 号的根因：指纹不能在提取之前记——提取失败后，同一句
    下一轮还得是“新鲜的”，不然重试时会被当成已处理过、跳过。"""
    store = make_fake_store(tmp_path, memory=FlakyFakeMemory(fail_times=1))
    store.add_turn([{"role": "user", "content": "我喜欢吃拉面。"}])
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    assert "我喜欢吃拉面。" not in store._committed_user_keys
    # 重试成功后，指纹才记下来。
    store.add_turn([{"role": "user", "content": "明天想去跑步。"}])
    store.flush_pending()
    assert "我喜欢吃拉面。" in store._committed_user_keys


# ---------- 52 号：outbox 回执 ----------


class _RaisingStore:
    """add_turn 一直抛错的假 store（outbox 重试用）。"""

    def __init__(self):
        self.calls = 0

    def add_turn(self, messages):
        self.calls += 1
        raise RuntimeError("模拟后台存记忆一直失败")


class _RecordingStore:
    """add_turn 正常返回的假 store（outbox 成功回执用）。"""

    def __init__(self):
        self.calls = 0

    def add_turn(self, messages):
        self.calls += 1
        return {"pending": True, "results": []}


def test_52_outbox_retries_then_reports_failed(tmp_path):
    """52 号：后台存记忆抛异常要重试（3 次），3 次都失败就进失败清单，
    turn_status 是 failed，别悄悄丢了。"""
    outbox = MemoryOutbox(_RaisingStore(), pending_dir=tmp_path / "outbox_pending")
    outbox._SAVE_RETRY_DELAY = 0  # 测试里不睡
    outbox.start()
    turn_id = outbox.submit([{"role": "user", "content": "我喜欢吃拉面。"}])
    assert outbox.turn_status(turn_id) == "pending"
    assert outbox.flush(timeout=10) is True
    assert outbox.turn_status(turn_id) == "failed"
    failed = outbox.failed_turns()
    assert len(failed) == 1
    assert failed[0]["turn_id"] == turn_id
    assert failed[0]["messages"][0]["content"] == "我喜欢吃拉面。"
    outbox.stop()


def test_52_outbox_success_reports_committed(tmp_path):
    """52 号：后台存完的轮次，turn_status 是 committed，失败清单是空的。"""
    store = _RecordingStore()
    outbox = MemoryOutbox(store, pending_dir=tmp_path / "outbox_pending")
    outbox.start()
    turn_id = outbox.submit([{"role": "user", "content": "我喜欢吃拉面。"}])
    assert outbox.flush(timeout=10) is True
    assert outbox.turn_status(turn_id) == "committed"
    assert outbox.failed_turns() == []
    assert store.calls == 1
    outbox.stop()


def test_52_outbox_unknown_turn_id(tmp_path):
    """没见过的 turn_id 查回执是 unknown，不是报错。"""
    outbox = MemoryOutbox(_RecordingStore(), pending_dir=tmp_path / "outbox_pending")
    assert outbox.turn_status("no-such-turn") == "unknown"


def test_57_outbox_persists_and_recovers(tmp_path):
    """57 号：submit 当场写盘，存完删盘；进程死前没存完的，重启捞回重存。"""
    pending_dir = tmp_path / "outbox_pending"
    store = _RecordingStore()
    outbox = MemoryOutbox(store, pending_dir=pending_dir)
    turn_id = outbox.submit([{"role": "user", "content": "我爱吃冒菜。"}])
    assert (pending_dir / f"{turn_id}.json").exists()
    outbox.start()
    assert outbox.flush(timeout=10) is True
    assert outbox.turn_status(turn_id) == "committed"
    assert not (pending_dir / f"{turn_id}.json").exists()
    outbox.stop()
    # 模拟进程死：提交一轮但不启动、不刷盘，直接建新 outbox 捞回。
    store2 = _RecordingStore()
    outbox2 = MemoryOutbox(store2, pending_dir=pending_dir)
    turn_id2 = outbox2.submit([{"role": "user", "content": "我想每天7点起床。"}])
    assert (pending_dir / f"{turn_id2}.json").exists()
    store3 = _RecordingStore()
    outbox3 = MemoryOutbox(store3, pending_dir=pending_dir)
    assert outbox3.recover_pending() == 1
    outbox3.start()
    assert outbox3.flush(timeout=10) is True
    assert store3.calls == 1
    assert not (pending_dir / f"{turn_id2}.json").exists()
    outbox3.stop()


# ---------- 55 号：触发条件放宽 ----------


def test_55_add_memory_description_no_longer_requires_the_word():
    """55 号：add_memory 的工具描述不能再写“只有明确说‘记住’才调”——
    明确表达的事实/偏好/决定/习惯都算存。"""
    from tools.memory_tool import register_memory_tools
    from tools.tool_registry import get_tool

    register_memory_tools()
    tool = get_tool("add_memory")
    description = tool["description"]
    assert "只在 BOOS 明确说" not in description
    assert "记住" in description  # “记住”还是最稳的一条路，只是唯一
    assert "事实" in description and "习惯" in description


def test_55_system_message_rule_8_allows_plain_statements():
    """55 号：系统消息第 8 条也不能再卡“记住”两个字；并且只有
    add_memory 成功返回后才能说“记下了”，没调时不能先说。"""
    from tools.agent_loop import build_loop_system_message

    system = build_loop_system_message()
    assert "不只认“记住”两个字" in system
    assert "每天7点起床" in system
    assert "记下了" in system
    # 旧的卡死说法不能再出现。
    assert "只有 BOOS 明确说“记住”时才调 add_memory" not in system


# ---------- 55 号：习惯直存，成功后才能说“记下了” ----------


def test_55_loop_adds_habit_via_tool_and_replies_recorded(tmp_path):
    """55 号：BOOS 说“我想以后每天7点起床”——模型调 add_memory 直存，
    成功返回后才回“记下了”，库里当天就有这条，不用等下一轮提取。"""
    import json

    from tasks.task_store import TaskStore
    from tools.agent_loop import AgentLoop
    from tools.audit import AuditLog
    from tools.tool_gateway import ToolGateway

    mem_store = make_fake_store(tmp_path)
    gateway = ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))
    from tools.memory_tool import bound_memory_store, register_memory_tools

    register_memory_tools()

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

    class ScriptedProvider:
        model = "deepseek-flash"

        def __init__(self, script):
            self.script = list(script)
            self.tool_calls_log = []

        def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                            task_id=None, trace_id=None):
            self.tool_calls_log.append({"messages": list(messages)})
            return dict(self.script.pop(0))

        def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                          thinking=None, model=None, max_tokens=None):
            return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}

    provider = ScriptedProvider([
        _tool_response(tool_calls=[
            _tool_call("call_1", "add_memory", {"text": "BOOS 想以后每天7点起床"})
        ]),
        _tool_response(text="记下了，每天7点起床。"),
    ])
    loop = AgentLoop(
        provider, gateway=gateway,
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=None,
    )
    with bound_memory_store(mem_store):
        outcome = loop.run("我想以后每天7点起床")
    assert outcome["mode"] == "done"
    assert "记下了" in outcome["reply"]
    assert any("7点起床" in text for text in mem_store._memory.all_texts())
