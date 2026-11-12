# tests/test_memory_tools.py（第 5 批：45 号记忆管理工具接进聊天）
# 作用：把第 5 批 45 号的修法钉死在测试里，全部离线模拟，不调真实 API、
# 不花钱；真调在 /tmp 外面另做（见第 5 批报告）。
# 原理：记忆库用 conftest 的 make_fake_store（FakeMemory）；模型用
# ScriptedProvider（按剧本回 tool_calls）；网关、任务存储用真的，
# 都指向 tmp_path 下的临时文件，不碰真实数据。
# 注意：记忆工具名是正式名字（list_memories / add_memory / update_memory /
# delete_memory），和 test_ 开头的测试工具区分开；登记只做一次，重复登记
# 不报错，所以不要在单个测试里重复登记同一个名字。

import json

import pytest

from conftest import make_fake_store
from tasks.task_store import TaskStore
from tools.agent_loop import LOOP_SYSTEM_MESSAGE, AgentLoop, build_loop_system_message
from tools.audit import AuditLog
from tools.memory_tool import (
    bound_memory_store,
    get_memory_store,
    register_memory_tools,
    set_memory_store,
)
from tools.tool_gateway import ToolGateway
from tools.tool_registry import get_tool


# ---------- 假模型、假搜索 ----------

def _tool_call(call_id, name, args):
    """拼一条模型返回的 tool_call（arguments 是 JSON 字符串）。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _tool_response(text="", tool_calls=None):
    """拼一次 chat_with_tools 的返回（剧本用）。"""
    return {
        "text": text,
        "tool_calls": tool_calls or [],
        "reasoning_content": None,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "cost_usd": 0.001,
        "model": "deepseek-flash",
    }


class ScriptedProvider:
    """按剧本回话的假模型：chat_with_tools 按 script 顺序回，
    chat_messages 回固定回答。每次调用都记下来。"""

    def __init__(self, model="deepseek-flash", script=None):
        self.model = model
        self.script = list(script or [])
        self.tool_calls_log = []

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        return {"text": "translated", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        self.tool_calls_log.append({"messages": list(messages), "thinking": thinking})
        if not self.script:
            raise AssertionError("剧本用完了，模型又要调工具——循环该停了")
        return dict(self.script.pop(0))

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None):
        return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}


class _NoSearchService:
    """这批测试不搜：调搜索就报错（证明记忆工具走自己的路）。"""

    def search(self, *args, **kwargs):
        raise AssertionError("这批测试不该调搜索")


# ---------- 固定装置 ----------

@pytest.fixture
def mem_store(tmp_path):
    """一个有两条记忆的假记忆库（带名字种子，方便列/改/删）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("BOOS 喜欢喝冰美式", source="user")
    store.add_memory_text("BOOS 每月工具预算 5 美元", source="user")
    return store


@pytest.fixture
def store(tmp_path):
    """指向临时文件的任务存储（每个测试独立）。"""
    return TaskStore(file_path=str(tmp_path / "tasks.json"))


@pytest.fixture
def gateway(tmp_path):
    """指向临时审计文件的网关（每个测试独立）。"""
    return ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))


def _make_loop(provider, gateway, store):
    register_memory_tools()
    return AgentLoop(
        provider,
        gateway=gateway,
        task_store=store,
        search_service=_NoSearchService(),
    )


def _mem_id(mem_store, text):
    """按原文在库里找一条记忆的 id（改/删工具要传 id + 原文）。"""
    for item in mem_store.get_all(include_sensitive=True):
        if item["text"] == text:
            return item["id"]
    raise AssertionError(f"库里没有这条记忆：{text}")


def _texts(mem_store):
    return [item["text"] for item in mem_store.get_all(include_sensitive=True)]


# ---------- 登记：四个工具都在，危险程度和确认卡分得清 ----------

def test_memory_tools_registered_with_right_flags():
    """45 号：四个记忆工具都登记了——列是低风险只读、可重跑；
    记是中风险不重跑；改和删是中风险、要确认卡、不重跑。"""
    register_memory_tools()
    listed = get_tool("list_memories")
    assert listed is not None
    assert listed["risk_level"] == "low"
    assert listed["needs_confirmation"] is False
    assert listed["replay_safe"] is True
    assert listed["side_effect_free"] is True
    added = get_tool("add_memory")
    assert added["risk_level"] == "medium"
    assert added["needs_confirmation"] is False
    assert added["replay_safe"] is False
    updated = get_tool("update_memory")
    assert updated["risk_level"] == "medium"
    assert updated["needs_confirmation"] is True
    assert updated["replay_safe"] is False
    deleted = get_tool("delete_memory")
    assert deleted["risk_level"] == "medium"
    assert deleted["needs_confirmation"] is True
    assert deleted["replay_safe"] is False
    # 工具的说明里写明了：改删之前先列出来、按原文复制，别凭印象改。
    assert "list_memories" in updated["description"]
    assert "list_memories" in deleted["description"]


def test_register_memory_tools_idempotent():
    """重复登记不报错、不覆盖（循环每次启动都会登记一次）。"""
    register_memory_tools()
    first = get_tool("list_memories")
    register_memory_tools()
    assert get_tool("list_memories") is first


def test_loop_system_message_has_memory_rule():
    """45 号：循环的系统消息里有第 8 条——长期记忆用记忆工具，
    改删过确认卡；55 号：记新事不再只认“记住”两个字——BOOS 明确表达的
    事实/偏好/决定/习惯都调 add_memory 存。"""
    assert "8. BOOS 让你看/改/删长期记忆时，用记忆工具" in LOOP_SYSTEM_MESSAGE
    assert "改和删要先过确认卡" in LOOP_SYSTEM_MESSAGE
    assert "明确表达的事实/偏好/决定/习惯" in LOOP_SYSTEM_MESSAGE
    assert "不只认“记住”两个字" in LOOP_SYSTEM_MESSAGE
    # 追加的那条（相对日期换算）现在是第 9 条。
    assert "\n9. 用户说“今天/昨天/最新”时" in build_loop_system_message()


# ---------- 绑定：上下文变量，不串到别的请求 ----------

def test_memory_store_binding_scoped():
    """绑定只在 with 块里有效，出块就恢复；绑 None 不覆盖外层的。"""
    assert get_memory_store() is None
    first = object()
    token = set_memory_store(first)
    assert get_memory_store() is first
    with bound_memory_store(None):
        assert get_memory_store() is first
    with bound_memory_store(object()) as inner:
        assert get_memory_store() is not first
    assert get_memory_store() is first
    token = set_memory_store(None)
    assert get_memory_store() is None


# ---------- 工具函数本身：绑上库能干活，没绑报错 ----------

def test_list_memories_returns_texts(mem_store):
    """列记忆：返回库里的全部记忆（按原来 get_all 的口径）。"""
    from tools.memory_tool import _run_list_memories

    with bound_memory_store(mem_store):
        result = _run_list_memories({})
    assert result["ok"] is True
    data = result["data"]
    texts = [entry["text"] for entry in data["memories"]]
    assert "BOOS 喜欢喝冰美式" in texts
    assert "BOOS 每月工具预算 5 美元" in texts
    assert data["count"] >= 2


def test_list_memories_counts_sensitive_without_showing(mem_store):
    """列记忆：敏感记忆不列出来，但 count 是库里真正的总数（只计数、不露内容）。
    敏感记忆直接从假库底层塞进去（正常存会脱敏，存不进去这种原文）。"""
    from tools.memory_tool import _run_list_memories

    with mem_store._memory._lock:
        mem_store._memory._store("我的密码是 abc123")
    with bound_memory_store(mem_store):
        result = _run_list_memories({})
    assert result["ok"] is True
    data = result["data"]
    texts = [entry["text"] for entry in data["memories"]]
    assert not any("abc123" in text for text in texts)
    assert not any("密码" in text for text in texts)
    # 库里 3 条（2 条普通 + 1 条敏感），列出 2 条，count 是 3。
    assert len(data["memories"]) == 2
    assert data["count"] == 3


def test_add_memory_stores_and_dedups(mem_store):
    """记记忆：新事存进去；再记一模一样的不重复存。"""
    from tools.memory_tool import _run_add_memory

    with bound_memory_store(mem_store):
        first = _run_add_memory({"text": "BOOS 喜欢吃拉面"})
        second = _run_add_memory({"text": "BOOS 喜欢吃拉面"})
    assert first["ok"] is True
    assert first["data"]["memory_id"]
    assert second["ok"] is True
    assert second["data"]["duplicate"] is True
    assert "BOOS 喜欢吃拉面" in _texts(mem_store)
    assert _texts(mem_store).count("BOOS 喜欢吃拉面") == 1


def test_update_memory_changes_text(mem_store):
    """改记忆：按原文找到旧条，换成新条。"""
    from tools.memory_tool import _run_update_memory

    with bound_memory_store(mem_store):
        result = _run_update_memory(
            {
                "memory_id": _mem_id(mem_store, "BOOS 每月工具预算 5 美元"),
                "old_text": "BOOS 每月工具预算 5 美元",
                "new_text": "BOOS 每月工具预算 10 美元",
            }
        )
    assert result["ok"] is True
    assert "BOOS 每月工具预算 10 美元" in _texts(mem_store)
    assert "BOOS 每月工具预算 5 美元" not in _texts(mem_store)


def test_update_memory_refuses_mismatched_text(mem_store):
    """改记忆：传的原文和库里对不上，拒绝改——不许凭印象改。"""
    from tools.memory_tool import _run_update_memory

    with bound_memory_store(mem_store):
        result = _run_update_memory(
            {
                "memory_id": _mem_id(mem_store, "BOOS 每月工具预算 5 美元"),
                "old_text": "BOOS 每月工具预算 99 美元",
                "new_text": "BOOS 每月工具预算 10 美元",
            }
        )
    assert result["ok"] is False
    assert "对不上" in result["error"]
    assert "BOOS 每月工具预算 5 美元" in _texts(mem_store)


def test_delete_memory_requires_approval_guard(mem_store):
    """删记忆：传的原文对得上才能删；删完库里没了，守卫没绕过
    （这里 approve=True 的意思是“网关确认卡已经通过了”，不是跳过守卫）。"""
    from tools.memory_tool import _run_delete_memory

    with bound_memory_store(mem_store):
        result = _run_delete_memory(
            {
                "memory_id": _mem_id(mem_store, "BOOS 喜欢喝冰美式"),
                "text": "BOOS 喜欢喝冰美式",
                "anchor_text": "冰美式",
            }
        )
    assert result["ok"] is True
    assert "BOOS 喜欢喝冰美式" not in _texts(mem_store)
    assert "BOOS 每月工具预算 5 美元" in _texts(mem_store)


def test_delete_memory_refuses_mismatched_text(mem_store):
    """删记忆：传的原文和库里对不上，拒绝删。"""
    from tools.memory_tool import _run_delete_memory

    with bound_memory_store(mem_store):
        result = _run_delete_memory(
            {
                "memory_id": _mem_id(mem_store, "BOOS 喜欢喝冰美式"),
                "text": "BOOS 喜欢喝热美式",
                "anchor_text": "冰美式",
            }
        )
    assert result["ok"] is False
    assert "对不上" in result["error"]
    assert "BOOS 喜欢喝冰美式" in _texts(mem_store)


def test_memory_tool_without_store_reports_not_enabled():
    """没绑记忆库时，工具报“记忆库没启用”，不崩溃、不瞎编。"""
    from tools.memory_tool import _run_list_memories

    assert get_memory_store() is None
    result = _run_list_memories({})
    assert result["ok"] is False
    assert "没启用" in result["error"]


# ---------- 网关：改删要确认卡，列和记不用 ----------

def test_gateway_needs_approval_for_update_delete_not_for_list_add(mem_store, gateway):
    """45 号：网关层面——改和删要确认卡（没确认不执行），列和记不用。"""
    register_memory_tools()
    budget_id = _mem_id(mem_store, "BOOS 每月工具预算 5 美元")
    coffee_id = _mem_id(mem_store, "BOOS 喜欢喝冰美式")
    with bound_memory_store(mem_store):
        assert gateway.execute("list_memories", {})["decision"] == "ok"
        assert gateway.execute("add_memory", {"text": "BOOS 喜欢吃拉面"})["decision"] == "ok"
        update = gateway.execute(
            "update_memory",
            {
                "memory_id": budget_id,
                "old_text": "BOOS 每月工具预算 5 美元",
                "new_text": "BOOS 每月工具预算 10 美元",
            },
        )
        assert update["decision"] == "needs_approval"
        delete = gateway.execute(
            "delete_memory",
            {"memory_id": coffee_id, "text": "BOOS 喜欢喝冰美式", "anchor_text": "冰美式"},
        )
        assert delete["decision"] == "needs_approval"
    # 没确认之前，库里一条没动。
    assert "BOOS 每月工具预算 5 美元" in _texts(mem_store)
    assert "BOOS 喜欢喝冰美式" in _texts(mem_store)


# ---------- 循环：模型能调记忆工具，改删走确认卡全流程 ----------

def test_loop_lists_memories_via_tool(mem_store, gateway, store):
    """45 号：聊天里模型调 list_memories，拿到库里的记忆再回答。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[_tool_call("call_1", "list_memories", {})]),
            _tool_response(text="你喜欢喝冰美式。"),
        ]
    )
    loop = _make_loop(provider, gateway, store)
    with bound_memory_store(mem_store):
        outcome = loop.run("我喜欢喝什么？")
    assert outcome["mode"] == "done"
    assert outcome["reply"] == "你喜欢喝冰美式。"
    # 工具结果喂给模型时，记忆文本在里面。
    second_call = provider.tool_calls_log[1]["messages"]
    assert any("BOOS 喜欢喝冰美式" in str(m.get("content", "")) for m in second_call)


def test_loop_adds_memory_via_tool(mem_store, gateway, store):
    """45 号：聊天里模型调 add_memory，BOOS 说“记住”的事当天就存进库里。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(
                tool_calls=[_tool_call("call_1", "add_memory", {"text": "BOOS 喜欢吃拉面"})]
            ),
            _tool_response(text="记住了，你喜欢吃拉面。"),
        ]
    )
    loop = _make_loop(provider, gateway, store)
    with bound_memory_store(mem_store):
        outcome = loop.run("记住，我喜欢吃拉面")
    assert outcome["mode"] == "done"
    assert "BOOS 喜欢吃拉面" in _texts(mem_store)


def test_loop_delete_via_approval_then_resume(mem_store, gateway, store):
    """45 号全流程：模型调 delete_memory → 循环停在确认卡（没执行）→
    BOOS 点确认 → 真删了 → 恢复循环 → 模型回答。"""
    coffee_id = _mem_id(mem_store, "BOOS 喜欢喝冰美式")
    provider = ScriptedProvider(
        script=[
            _tool_response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "delete_memory",
                        {"memory_id": coffee_id, "text": "BOOS 喜欢喝冰美式", "anchor_text": "冰美式"},
                    )
                ]
            ),
            _tool_response(text="删掉了。"),
        ]
    )
    loop = _make_loop(provider, gateway, store)
    with bound_memory_store(mem_store):
        outcome = loop.run("把“我喜欢喝冰美式”这条记忆删掉")
    # 第一步：停在确认卡，库里还没删。
    assert outcome["mode"] == "waiting_approval"
    assert outcome["approval"]["tool"] == "delete_memory"
    assert "BOOS 喜欢喝冰美式" in _texts(mem_store)
    task = outcome["task"]
    # 第二步：BOOS 点确认，网关执行删除。
    with bound_memory_store(mem_store):
        resolved = gateway.resolve_approval(
            store, task["task_id"], outcome["approval"]["approval_id"], "confirm"
        )
    assert resolved["decision"] == "ok"
    assert resolved["result"]["ok"] is True
    assert "BOOS 喜欢喝冰美式" not in _texts(mem_store)
    # 第三步：恢复循环，模型回答。
    with bound_memory_store(mem_store):
        resumed = loop.run(
            "把“我喜欢喝冰美式”这条记忆删掉",
            resume_task=resolved["task"],
            pending_tool_result=resolved["result"],
        )
    assert resumed["mode"] == "done"
    assert resumed["reply"] == "删掉了。"


def test_loop_update_via_approval_then_resume(mem_store, gateway, store):
    """45 号全流程（改）：模型调 update_memory → 循环停在确认卡（没改）→
    BOOS 点确认 → 真改了 → 恢复循环 → 模型回答。"""
    budget_id = _mem_id(mem_store, "BOOS 每月工具预算 5 美元")
    provider = ScriptedProvider(
        script=[
            _tool_response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "update_memory",
                        {
                            "memory_id": budget_id,
                            "old_text": "BOOS 每月工具预算 5 美元",
                            "new_text": "BOOS 每月工具预算 10 美元",
                        },
                    )
                ]
            ),
            _tool_response(text="改好了。"),
        ]
    )
    loop = _make_loop(provider, gateway, store)
    with bound_memory_store(mem_store):
        outcome = loop.run("把每月工具预算改成 10 美元")
    # 第一步：停在确认卡，库里还没改。
    assert outcome["mode"] == "waiting_approval"
    assert outcome["approval"]["tool"] == "update_memory"
    assert "BOOS 每月工具预算 5 美元" in _texts(mem_store)
    task = outcome["task"]
    # 第二步：BOOS 点确认，网关执行修改。
    with bound_memory_store(mem_store):
        resolved = gateway.resolve_approval(
            store, task["task_id"], outcome["approval"]["approval_id"], "confirm"
        )
    assert resolved["decision"] == "ok"
    assert resolved["result"]["ok"] is True
    assert "BOOS 每月工具预算 10 美元" in _texts(mem_store)
    assert "BOOS 每月工具预算 5 美元" not in _texts(mem_store)
    # 第三步：恢复循环，模型回答。
    with bound_memory_store(mem_store):
        resumed = loop.run(
            "把每月工具预算改成 10 美元",
            resume_task=resolved["task"],
            pending_tool_result=resolved["result"],
        )
    assert resumed["mode"] == "done"
    assert resumed["reply"] == "改好了。"


def test_loop_delete_cancel_keeps_memory(mem_store, gateway, store):
    """45 号：确认卡上点取消，记忆一条没删，循环继续回答。"""
    coffee_id = _mem_id(mem_store, "BOOS 喜欢喝冰美式")
    provider = ScriptedProvider(
        script=[
            _tool_response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "delete_memory",
                        {"memory_id": coffee_id, "text": "BOOS 喜欢喝冰美式", "anchor_text": "冰美式"},
                    )
                ]
            ),
            _tool_response(text="好，不删了。"),
        ]
    )
    loop = _make_loop(provider, gateway, store)
    with bound_memory_store(mem_store):
        outcome = loop.run("把“我喜欢喝冰美式”这条记忆删掉")
    assert outcome["mode"] == "waiting_approval"
    task = outcome["task"]
    with bound_memory_store(mem_store):
        resolved = gateway.resolve_approval(
            store, task["task_id"], outcome["approval"]["approval_id"], "cancel"
        )
    assert resolved["decision"] == "cancelled"
    assert "BOOS 喜欢喝冰美式" in _texts(mem_store)
    with bound_memory_store(mem_store):
        resumed = loop.run(
            "把“我喜欢喝冰美式”这条记忆删掉",
            resume_task=resolved["task"],
            user_declined=True,
        )
    assert resumed["mode"] == "done"
    assert resumed["reply"] == "好，不删了。"
    assert "BOOS 喜欢喝冰美式" in _texts(mem_store)


def test_loop_delete_replay_not_rerun(mem_store, gateway, store):
    """45 号：删记忆不可重跑——确认执行后，模型拿同一个工具调用号再发一次，
    循环直接跳过，不会再删一次（也没有第二条可删）。"""
    coffee_id = _mem_id(mem_store, "BOOS 喜欢喝冰美式")
    provider = ScriptedProvider(
        script=[
            _tool_response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "delete_memory",
                        {"memory_id": coffee_id, "text": "BOOS 喜欢喝冰美式", "anchor_text": "冰美式"},
                    )
                ]
            ),
            # 恢复后模型又发了一次同一个调用号（重放攻击 / 模型重复）。
            _tool_response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "delete_memory",
                        {"memory_id": coffee_id, "text": "BOOS 喜欢喝冰美式", "anchor_text": "冰美式"},
                    )
                ]
            ),
            _tool_response(text="已经删掉了。"),
        ]
    )
    loop = _make_loop(provider, gateway, store)
    with bound_memory_store(mem_store):
        outcome = loop.run("把“我喜欢喝冰美式”这条记忆删掉")
    assert outcome["mode"] == "waiting_approval"
    task = outcome["task"]
    with bound_memory_store(mem_store):
        resolved = gateway.resolve_approval(
            store, task["task_id"], outcome["approval"]["approval_id"], "confirm"
        )
    assert resolved["result"]["ok"] is True
    with bound_memory_store(mem_store):
        resumed = loop.run(
            "把“我喜欢喝冰美式”这条记忆删掉",
            resume_task=resolved["task"],
            pending_tool_result=resolved["result"],
        )
    assert resumed["mode"] == "done"
    # 只执行了一次（第一次那 1 次），重放的那次没算。
    assert resumed["tool_calls"] == 1
    assert "BOOS 每月工具预算 5 美元" in _texts(mem_store)
