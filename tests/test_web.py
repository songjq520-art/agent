# tests/test_web.py（本地网页入口的测试）
# 作用：验证 web/server.py 的接口、对话历史逻辑和 mem0 记忆接线。
# 原理：用临时端口（port 0，系统自动分配）和临时预算文件启动真实服务，
#       再用标准库 http.client 发请求验证，不调用真实模型 API。
#       需要“真实调用成功”的场景时，用 monkeypatch 把 run_question 换成
#       返回假结果的函数，不产生费用。
#       记忆用下面就地定义的 FakeMem0Store（最小的假 store），不碰真实 mem0。
# 注意：这些测试不依赖网络、不依赖 API Key；无 Key 时走演示模式。

import http.client
import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker
from conftest import make_fake_store
from web.server import ChatState, ServerConfig, create_server


def _fake_ok_response(text="你好，我是 DeepSeek。", prompt_tokens=100, completion_tokens=50):
    """模拟的成功 API 返回（带 usage），供真实 run_question 的测试用。"""
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


class FakeMem0Store:
    """最小的 mem0 假 store：只实现 web 层用到的四个方法，不碰真实 mem0。

    memories 是 search() 固定返回的记忆文本列表；saved_turns 记下每次
    add_turn() 收到的消息，供测试核对“回答后存了本轮对话”；
    noted_context 记下每次 note_context() 收到的记忆 id，供测试核对
    “提问前记了最近对话上下文”。
    """

    def __init__(self, memories=None):
        self.memories = list(memories or [])
        self.saved_turns = []
        self.noted_context = []
        # 25 号直存记下来的（每项 {"text", "source"}），供测试核对
        # “记住”有没有立刻存、记的来源对不对。
        self.direct_stored = []
        # 第二步衰减：每次 search() 有没有要求强化（reinforce=True），
        # 供测试核对“真把记忆用上了才强化”的接线。
        self.reinforce_calls = []

    def add_memory_text(self, text, source="user"):
        """25 号直存：记下来，放进 memories 里让下一轮 search() 能找到
        （和真实库“存完下一句就能查到”一致），返回非空表示存成了。"""
        self.direct_stored.append({"text": text, "source": source})
        if text not in self.memories:
            self.memories.append(text)
        return {"results": [{"id": f"d{len(self.direct_stored)}", "memory": text, "event": "ADD"}]}

    def search(self, query, top_k=5, reinforce=False):
        """返回固定的记忆列表（每项和真实 Mem0Store.search() 的格式一致）。
        reinforce 跟真实签名对齐（默认 False），只记下来、不真写回。"""
        self.reinforce_calls.append(bool(reinforce))
        return [
            {"id": f"m{i}", "text": text, "score": 0.9, "source": "semantic"}
            for i, text in enumerate(self.memories[:top_k])
        ]

    def add_turn(self, messages):
        """记下收到的消息（复制一份，免得调用方之后改了历史影响核对）。"""
        self.saved_turns.append(list(messages))
        return {}

    def save_turn(self, messages, turn_id=None):
        """26 号后台存：假 store 不模拟后台线程，直接走 add_turn 当场存
        （和真实库“纠错/撤回走同步”的那条路一致），返回 add_turn 的结果。
        这样 web 层“回答后存了本轮对话”的核对（saved_turns）不用改。"""
        return self.add_turn(messages)

    def note_context(self, memories):
        """记下本轮检索到的记忆 id（和真实 Mem0Store.note_context 一致，
        假对象只记下来，供测试核对“提问前记了上下文”）。"""
        self.noted_context = [
            (item.get("id") if isinstance(item, dict) else item)
            for item in (memories or [])
            if (item.get("id") if isinstance(item, dict) else item)
        ]

    def get_all(self):
        return [{"id": f"m{i}", "text": text} for i, text in enumerate(self.memories)]


def _request(port, method, path, body=None):
    """向测试服务发一次请求，返回 (状态码, 响应头 Content-Type, 解析后的 JSON 或原始文本)。"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {}
    payload = None
    if body is not None:
        # 请求体要先编码成 UTF-8 字节再发，http.client 不接受含中文的 str。
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    content_type = resp.getheader("Content-Type") or ""
    if "application/json" in content_type:
        return resp.status, content_type, json.loads(raw.decode("utf-8"))
    return resp.status, content_type, raw.decode("utf-8", errors="replace")


def _start_test_server(tmp_path, authenticator=None, mem0_store=None):
    """用临时端口和临时预算文件启动一个测试服务，返回 (server, port, state)。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="127.0.0.1", port=0)
    server, state = create_server(
        config=config, budget_tracker=tracker, authenticator=authenticator, mem0_store=mem0_store
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], state


def _stop_test_server(server):
    server.shutdown()
    server.server_close()


@pytest.fixture
def running_server(tmp_path):
    """启动一个用临时预算文件的测试服务，测试完关掉。

    返回 (port, state)：port 是实际监听的端口，state 是 ChatState，
    测试里可以直接看 state.messages 验证历史有没有追加。
    默认不带记忆库（mem0_store=None）。
    """
    server, port, state = _start_test_server(tmp_path)
    try:
        yield port, state
    finally:
        _stop_test_server(server)


@pytest.fixture
def running_server_with_mem0(tmp_path):
    """启动一个带假 mem0 记忆库的测试服务，测试完关掉。

    返回 (port, state, mem0_store)：记忆库里已有一条“用户叫 BOOS。”，
    供测试核对提问时有没有把记忆带给模型、回答后有没有存本轮对话。
    """
    mem0_store = FakeMem0Store(memories=["用户叫 BOOS。"])
    server, port, state = _start_test_server(tmp_path, mem0_store=mem0_store)
    try:
        yield port, state, mem0_store
    finally:
        _stop_test_server(server)


def _fake_success_run_question(prompt, budget_tracker=None, messages=None, **kwargs):
    """假的 run_question：直接返回一轮成功的假回答，不调用真实模型。"""
    return {
        "mode": "real",
        "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
        "result": {"text": "假回答", "model": "deepseek-flash", "cost_usd": 0.0001, "usage": {"total_tokens": 10}},
        "message": "真实调用完成。",
        "model_status": {},
    }


def test_get_root_serves_chat_page(running_server):
    """GET / 返回聊天页面：200、HTML、里面有标题和输入框，没有提议侧边栏。"""
    port, _state = running_server
    status, content_type, body = _request(port, "GET", "/")
    assert status == 200
    assert "text/html" in content_type
    assert "我的 Agent" in body
    assert 'id="input"' in body
    assert 'id="send"' in body
    # 提议侧边栏已删除：页面里不该再有提议区块和指向 /api/proposals 的请求。
    assert "待确认的记忆提议" not in body
    assert "proposalList" not in body
    assert "/api/proposals" not in body


def test_get_status_returns_models_and_budget(running_server):
    """GET /api/status 返回模型状态和本月预算：有 deepseek、预算 5 美元、剩余 5 美元。

    默认不带记忆库，mem0_enabled 是 False。
    """
    port, _state = running_server
    status, content_type, data = _request(port, "GET", "/api/status")
    assert status == 200
    assert "application/json" in content_type
    assert "deepseek" in data["model_status"]
    assert data["monthly_budget_usd"] == 5.0
    assert data["total_cost_usd"] == 0.0
    assert data["remaining_usd"] == 5.0
    assert data["budget_corrupted"] is False
    assert data["mem0_enabled"] is False


def test_get_status_reports_mem0_enabled(running_server_with_mem0):
    """带假 mem0 记忆库启动时：GET /api/status 返回 mem0_enabled 为 True。"""
    port, _state, _mem0_store = running_server_with_mem0
    status, _ct, data = _request(port, "GET", "/api/status")
    assert status == 200
    assert data["mem0_enabled"] is True


def test_chat_demo_mode_returns_demo_message_and_keeps_history_empty(running_server, monkeypatch):
    """无 Key 时 POST /api/chat 走演示模式：返回演示提示，历史不动。"""
    port, state = running_server
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "你好"})
    assert status == 200
    assert data["mode"] == "demo"
    assert "模拟模式" in data["reply"] or "演示" in data["reply"]
    assert data["cost_usd"] is None
    assert state.messages == []


def test_chat_success_appends_history(running_server, monkeypatch):
    """真实调用成功（用假结果模拟）时：返回回答，历史追加一问一答。"""
    port, state = running_server
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "你好"})
    assert status == 200
    assert data["mode"] == "real"
    assert data["reply"] == "假回答"
    assert data["model"] == "deepseek-flash"
    assert data["cost_usd"] == 0.0001
    assert state.messages == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "假回答"},
    ]


def test_chat_failure_returns_error_and_keeps_history(running_server, monkeypatch):
    """调用抛 RuntimeError 时：返回 502 和通用错误，不泄露内部信息，历史不动。"""
    port, state = running_server

    def _boom(prompt, budget_tracker=None, messages=None, **kwargs):
        # 模拟服务商报错里夹带内部信息：接口地址、内部请求号、假密钥片段。
        raise RuntimeError(
            "DeepSeek API 调用失败：401 Client Error for url: "
            "https://api.deepseek.com/chat/completions，内部请求号 req-abc123，"
            "密钥片段 sk-fake-secret-xyz"
        )

    monkeypatch.setattr("web.server.run_question", _boom)
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "你好"})
    assert status == 502
    assert data["error"] == "调用失败，请稍后再试。"
    assert "req-abc123" not in data["error"]
    assert "sk-fake-secret-xyz" not in data["error"]
    assert "api.deepseek.com" not in data["error"]
    assert "DeepSeek" not in data["error"]
    assert state.messages == []


def test_chat_empty_message_returns_400(running_server):
    """消息为空（或只有空格）时：返回 400，不调用模型。"""
    port, state = running_server
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "   "})
    assert status == 400
    assert "不能为空" in data["error"]
    assert state.messages == []


def test_chat_bad_json_returns_400(running_server):
    """请求体不是合法 JSON 时：返回 400。"""
    port, _state = running_server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/api/chat", body="{不是 JSON".encode("utf-8"), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    conn.close()
    assert resp.status == 400
    assert "JSON" in data["error"]


def test_reset_clears_history(running_server, monkeypatch):
    """POST /api/reset 清空对话历史：先攒一轮问答，再清空，历史变回空。"""
    port, state = running_server
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    _request(port, "POST", "/api/chat", {"message": "你好"})
    assert len(state.messages) == 2
    status, _ct, data = _request(port, "POST", "/api/reset")
    assert status == 200
    assert data["ok"] is True
    assert state.messages == []


def test_unknown_path_returns_404(running_server):
    """不存在的地址：返回 404（路径用 ASCII，http.client 发不了中文路径）。"""
    port, _state = running_server
    status, _ct, _body = _request(port, "GET", "/no-such-page")
    assert status == 404
    status, _ct, _body = _request(port, "POST", "/api/no-such", {"message": "你好"})
    assert status == 404


# ---------- mem0 长期记忆 ----------


def test_ask_without_mem0_store_chats_normally(tmp_path, monkeypatch):
    """mem0_store=None 时 ask() 仍能正常聊天：不带记忆，原样提问，历史正常追加。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    state = ChatState(budget_tracker=tracker, mem0_store=None)
    seen = {}

    def _capturing_fake(prompt, budget_tracker=None, messages=None, **kwargs):
        seen["prompt"] = prompt
        return _fake_success_run_question(prompt, budget_tracker, messages, **kwargs)

    monkeypatch.setattr("web.server.run_question", _capturing_fake)
    result = state.ask("你好")
    assert result["mode"] == "real"
    assert result["reply"] == "假回答"
    # 没带记忆库：发给模型的和用户原话一样，没拼记忆块。
    assert seen["prompt"] == "你好"
    assert state.messages == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "假回答"},
    ]


def test_ask_with_mem0_injects_memory_and_saves_turn(tmp_path, monkeypatch):
    """有 mem0 记忆库时：提问前把搜到的记忆拼进发给模型的问题里，回答后存本轮对话。

    历史里仍记用户原话（不带记忆块）；存记忆时带的是本轮及最近的消息，
    含刚追加的新问答。
    """
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    mem0_store = FakeMem0Store(memories=["用户叫 BOOS。"])
    state = ChatState(budget_tracker=tracker, mem0_store=mem0_store)
    seen = {}

    def _capturing_fake(prompt, budget_tracker=None, messages=None, **kwargs):
        seen["prompt"] = prompt
        return _fake_success_run_question(prompt, budget_tracker, messages, **kwargs)

    monkeypatch.setattr("web.server.run_question", _capturing_fake)
    result = state.ask("你记得我叫什么吗？")
    assert result["mode"] == "real"
    assert result["reply"] == "假回答"
    # 发给模型的问题里带上了记忆块（main._memory_block 拼的格式）。
    assert "用户叫 BOOS。" in seen["prompt"]
    assert "仅供参考" in seen["prompt"]
    assert seen["prompt"].endswith("你记得我叫什么吗？")
    # 历史里记的是原话，不带记忆块。
    assert state.messages == [
        {"role": "user", "content": "你记得我叫什么吗？"},
        {"role": "assistant", "content": "假回答"},
    ]
    # 回答成功后调了一次 add_turn，带上了刚追加的新问答。
    assert len(mem0_store.saved_turns) == 1
    assert mem0_store.saved_turns[0] == state.messages
    # 提问前记了一次上下文：搜到的那条记忆的 id（纠错时当“在聊什么”的提示）。
    assert mem0_store.noted_context == ["m0"]
    # 第二步衰减：提问前那次检索要求了强化（记忆真拼进问题里了）。
    assert mem0_store.reinforce_calls == [True]


def test_ask_with_mem0_but_no_results_asks_plainly(tmp_path, monkeypatch):
    """记忆库里搜不到相关记忆时：原样提问，不拼空的记忆块；回答后仍存本轮对话。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    mem0_store = FakeMem0Store(memories=[])
    state = ChatState(budget_tracker=tracker, mem0_store=mem0_store)
    seen = {}

    def _capturing_fake(prompt, budget_tracker=None, messages=None, **kwargs):
        seen["prompt"] = prompt
        return _fake_success_run_question(prompt, budget_tracker, messages, **kwargs)

    monkeypatch.setattr("web.server.run_question", _capturing_fake)
    result = state.ask("今天天气怎么样？")
    assert result["mode"] == "real"
    assert seen["prompt"] == "今天天气怎么样？"
    assert len(mem0_store.saved_turns) == 1
    assert mem0_store.saved_turns[0] == state.messages


def test_ask_remember_stores_immediately_and_skips_add_turn(tmp_path, monkeypatch):
    """25 号：网页说“记住……”要立刻直存（不用等下一轮），收尾跳过
    add_turn（别让提取再存一条推断版）；下一句问起来就能查到。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    mem0_store = FakeMem0Store(memories=[])
    state = ChatState(budget_tracker=tracker, mem0_store=mem0_store)
    seen = {}

    def _capturing_fake(prompt, budget_tracker=None, messages=None, **kwargs):
        seen["prompt"] = prompt
        return _fake_success_run_question(prompt, budget_tracker, messages, **kwargs)

    monkeypatch.setattr("web.server.run_question", _capturing_fake)
    result = state.ask("记住，我喜欢喝冰美式")
    assert result["mode"] == "real"
    assert result["reply"] == "假回答"
    # 直存了：内容是“记住”后面的文字，来源是 user（已确认）。
    assert mem0_store.direct_stored == [
        {"text": "我喜欢喝冰美式", "source": "user"}
    ]
    # 直存那轮不调 add_turn（已经在库里了，再提取会重复存）。
    assert len(mem0_store.saved_turns) == 0
    # 直存的记忆在当轮搜记忆时就已经能找到，发给模型的提示块里有它。
    assert "我喜欢喝冰美式" in seen["prompt"]
    # 下一句马上问，搜到的记忆块里有这条（不用等一轮）。
    result = state.ask("你记不记得我喜欢喝什么")
    assert result["mode"] == "real"
    assert "我喜欢喝冰美式" in seen["prompt"]
    # 普通那轮照常走 add_turn（只跳过“记住”那轮）。
    assert len(mem0_store.saved_turns) == 1
    # 第二步衰减：两轮提问前的检索都要求了强化。
    assert mem0_store.reinforce_calls == [True, True]


def test_proposals_endpoint_is_gone(running_server):
    """提议端点已删除：GET /api/proposals 和确认/拒绝地址都回 404。"""
    port, _state = running_server
    status, _ct, _body = _request(port, "GET", "/api/proposals")
    assert status == 404
    status, _ct, _body = _request(port, "POST", "/api/proposals/abc123/confirm", {"ok": True})
    assert status == 404
    status, _ct, _body = _request(port, "POST", "/api/proposals/abc123/reject", {"ok": True})
    assert status == 404


def test_ask_with_real_mem0_store_searches_and_saves(tmp_path, monkeypatch):
    """ChatState + 真实 Mem0Store（注了 FakeMemory，见 tests/conftest.py）：
    提问前经 Mem0Store.search 检索到名字记忆并拼进发给模型的问题里，
    回答后经 add_turn 把本轮对话存进记忆（FakeMemory 提出“我叫小宋”）。

    和上面用 FakeMem0Store 的测试不同，这里走的是真实适配层逻辑
    （阈值过滤 + 关键词补充 + 提取），只把底下的 mem0 换成假的。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋，昵称小宋，请称呼我小宋。")
    state = ChatState(budget_tracker=tracker, mem0_store=store)
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()) as mock_post:
        result = state.ask("我叫小宋。")
    assert result["mode"] == "real"
    assert result["reply"] == "你好，我是 DeepSeek。"
    # 发给模型的问题里带上了检索到的名字记忆。
    sent = mock_post.call_args.kwargs["json"]["messages"]
    assert "小宋" in sent[-1]["content"]
    # 历史里记的是用户原话，不带记忆块。
    assert state.messages == [
        {"role": "user", "content": "我叫小宋。"},
        {"role": "assistant", "content": "你好，我是 DeepSeek。"},
    ]
    # 回答后本轮对话先待定（Plan C），flush 时才提交：
    # “我叫小宋”说的就是种下的那条名字记忆（同一槽位、同一值），
    # upsert 判定为 unchanged、不重复存——库里还是原来那 1 条。
    store.flush_pending()
    texts = [item["text"] for item in store.get_all()]
    assert len([text for text in texts if "小宋" in text]) == 1
    assert texts == ["我叫小宋，昵称小宋，请称呼我小宋。"]
    # 再问一句非身份事实，flush 后正常存下来（待定区本身没坏）。
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()):
        state.ask("我喜欢喝咖啡。")
    store.flush_pending()
    assert "我喜欢喝咖啡" in [item["text"] for item in store.get_all()]


def test_ask_demo_mode_with_mem0_does_not_save_turn(tmp_path, monkeypatch):
    """演示模式（无 Key）+ 有 mem0 记忆库：返回演示提示，历史不动，
    也不调 add_turn 存记忆。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    state = ChatState(budget_tracker=tracker, mem0_store=store)
    result = state.ask("我叫小宋。")
    assert result["mode"] == "demo"
    assert state.messages == []
    assert store._memory.add_calls == []


# ---------- ServerConfig（监听配置） ----------


def test_server_config_defaults():
    """ServerConfig 默认值：只监听本机 127.0.0.1:8000。"""
    config = ServerConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 8000


def test_server_config_from_env(monkeypatch):
    """环境变量能改监听地址和端口。"""
    monkeypatch.setenv("AGENT_WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("AGENT_WEB_PORT", "8080")
    config = ServerConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.port == 8080


def test_server_config_from_env_bad_port_falls_back(monkeypatch):
    """环境变量里的端口不是数字时：用默认 8000，不崩。"""
    monkeypatch.setenv("AGENT_WEB_PORT", "not-a-number")
    config = ServerConfig.from_env()
    assert config.port == 8000


# ---------- JSON 类型校验 ----------


@pytest.mark.parametrize("bad_body", ["[]", "123", '"你好"', "null", "true"])
def test_chat_non_object_json_returns_400(running_server, bad_body):
    """请求 JSON 是数组/数字/字符串/null/布尔值时：返回 400，不崩，历史不动。"""
    port, state = running_server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/api/chat", body=bad_body.encode("utf-8"), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    conn.close()
    assert resp.status == 400
    assert "JSON 对象" in data["error"]
    assert state.messages == []


def test_chat_bad_content_length_returns_400(running_server):
    """请求头 Content-Length 不是数字时：返回 400，不崩。"""
    port, _state = running_server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        "POST",
        "/api/chat",
        body=b'{"message": "hello"}',
        headers={"Content-Type": "application/json", "Content-Length": "abc"},
    )
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    conn.close()
    assert resp.status == 400
    assert "Content-Length" in data["error"]


# ---------- 通用异常处理 ----------


def test_unexpected_error_returns_generic_500(running_server, monkeypatch):
    """run_question 抛非 RuntimeError 时：返回 500 通用错误，不泄露内部信息。"""
    port, state = running_server

    def _boom(prompt, budget_tracker=None, messages=None, **kwargs):
        raise ValueError("内部秘密：数据库连接串 abc123")

    monkeypatch.setattr("web.server.run_question", _boom)
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "你好"})
    assert status == 500
    assert data["error"] == "服务内部出错，请稍后再试。"
    assert "abc123" not in data["error"]
    assert "数据库" not in data["error"]
    assert state.messages == []


# ---------- 身份验证钩子（未来预留） ----------


def test_authenticator_deny_returns_401(tmp_path):
    """authenticator 返回 False 时：所有接口都回 401，不放行。"""
    server, port, _state = _start_test_server(tmp_path, authenticator=lambda handler: False)
    try:
        status, _ct, data = _request(port, "GET", "/api/status")
        assert status == 401
        assert "未授权" in data["error"]
        status, _ct, data = _request(port, "POST", "/api/chat", {"message": "你好"})
        assert status == 401
        status, _ct, _body = _request(port, "GET", "/")
        assert status == 401
    finally:
        _stop_test_server(server)


def test_authenticator_allow_passes(tmp_path):
    """authenticator 返回 True 时：请求正常放行。"""
    server, port, _state = _start_test_server(tmp_path, authenticator=lambda handler: True)
    try:
        status, _ct, data = _request(port, "GET", "/api/status")
        assert status == 200
        assert "deepseek" in data["model_status"]
    finally:
        _stop_test_server(server)


def test_authenticator_error_is_treated_as_deny(tmp_path):
    """authenticator 自己抛错时：按拒绝处理，回 401，不让请求继续。"""
    def _broken(handler):
        raise RuntimeError("校验函数坏了")

    server, port, _state = _start_test_server(tmp_path, authenticator=_broken)
    try:
        status, _ct, data = _request(port, "GET", "/api/status")
        assert status == 401
    finally:
        _stop_test_server(server)


# ---------- 并发安全 ----------


def _slow_fake_run_question(prompt, budget_tracker=None, messages=None, **kwargs):
    """假的 run_question：故意拖 0.2 秒，让两个请求重叠；回答里带上问题本身，方便核对配对。"""
    time.sleep(0.2)
    return {
        "mode": "real",
        "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
        "result": {"text": f"答：{prompt}", "model": "deepseek-flash", "cost_usd": 0.0001, "usage": {"total_tokens": 10}},
        "message": "真实调用完成。",
        "model_status": {},
    }


def test_concurrent_chats_keep_history_consistent(running_server, monkeypatch):
    """两个 /api/chat 同时跑：历史不乱，每条回答紧跟自己的问题，不交叉。"""
    port, state = running_server
    monkeypatch.setattr("web.server.run_question", _slow_fake_run_question)

    # 循环 3 次：没有锁时，追加顺序容易乱；有锁时，每次都稳。
    for _ in range(3):
        state.reset()
        results = {}

        def _ask(name, prompt):
            _status, _ct, data = _request(port, "POST", "/api/chat", {"message": prompt})
            results[name] = data

        t1 = threading.Thread(target=_ask, args=("a", "问题A"))
        t2 = threading.Thread(target=_ask, args=("b", "问题B"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        assert results["a"]["reply"] == "答：问题A"
        assert results["b"]["reply"] == "答：问题B"
        assert len(state.messages) == 4
        # 每两条一组（user 紧跟 assistant），且每组的回答属于同一问题。
        pairs = [(state.messages[i]["content"], state.messages[i + 1]["content"]) for i in range(0, 4, 2)]
        assert sorted(pairs) == sorted([("问题A", "答：问题A"), ("问题B", "答：问题B")])
        for i in range(0, 4, 2):
            assert state.messages[i]["role"] == "user"
            assert state.messages[i + 1]["role"] == "assistant"


def test_reset_during_chat_leaves_consistent_history(running_server, monkeypatch):
    """聊天进行中调用 /api/reset：历史要么是完整的一问一答，要么是空，不会只剩半条。"""
    port, state = running_server
    monkeypatch.setattr("web.server.run_question", _slow_fake_run_question)

    def _ask():
        _request(port, "POST", "/api/chat", {"message": "你好"})

    t = threading.Thread(target=_ask)
    t.start()
    time.sleep(0.05)  # 等聊天开始（还在假调用里拖着），再发清空。
    status, _ct, data = _request(port, "POST", "/api/reset")
    assert status == 200
    assert data["ok"] is True
    t.join(timeout=10)
    assert len(state.messages) in (0, 2)
    if len(state.messages) == 2:
        assert state.messages[0]["role"] == "user"
        assert state.messages[1]["role"] == "assistant"


def test_concurrent_chat_calls_are_isolated(running_server, monkeypatch):
    """慢请求进行时，快请求不能把它的消息塞进慢请求的上下文。

    原理：慢请求睡 0.4 秒、快请求睡 0.1 秒，快请求先结束并追加历史。
    有锁时，慢请求全程拿着锁，快请求进不来，慢请求看到的历史还是开始时的空；
    没锁时，慢请求在睡，快请求的追加会出现在慢请求看到的历史里。
    用 Event 保证慢请求先进了调用，快请求后放进来，不靠运气。
    """
    port, state = running_server
    seen = {}
    started = threading.Event()

    def _timed_fake(prompt, budget_tracker=None, messages=None, **kwargs):
        if prompt == "慢":
            started.set()  # 告诉测试：慢请求已经进来了，锁已拿到。
        time.sleep(0.4 if prompt == "慢" else 0.1)
        seen[prompt] = len(messages)
        return {
            "mode": "real",
            "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
            "result": {"text": f"答：{prompt}", "model": "deepseek-flash", "cost_usd": 0.0001, "usage": {"total_tokens": 10}},
            "message": "真实调用完成。",
            "model_status": {},
        }

    monkeypatch.setattr("web.server.run_question", _timed_fake)

    def _ask(name, prompt):
        _status, _ct, data = _request(port, "POST", "/api/chat", {"message": prompt})
        return data

    results = {}
    t_slow = threading.Thread(target=lambda: results.update(slow=_ask("slow", "慢")))
    t_fast = threading.Thread(target=lambda: results.update(fast=_ask("fast", "快")))
    t_slow.start()
    assert started.wait(timeout=5)  # 等慢请求真正开始，再放快请求进来。
    t_fast.start()
    t_slow.join(timeout=10)
    t_fast.join(timeout=10)
    assert results["slow"]["reply"] == "答：慢"
    assert results["fast"]["reply"] == "答：快"
    # 慢请求开始时历史是空的；有锁时，它全程看不到快请求的消息。
    assert seen["慢"] == 0
    # 历史最终是 4 条，配对正确。
    assert len(state.messages) == 4


# ---------- 监听地址安全保护 ----------


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),  # 127.0.0.0/8 整个段都是回环
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("192.168.1.5", False),
        ("10.0.0.1", False),
        ("not-an-ip", False),  # 写错的地址按非回环处理，宁可严一点
    ],
)
def test_is_loopback_host(host, expected):
    """回环地址判断：127 段、::1、localhost 算回环，其他不算。"""
    from web.server import _is_loopback_host

    assert _is_loopback_host(host) is expected


def test_non_loopback_without_auth_refuses_to_start(tmp_path):
    """监听地址不是回环、又没配身份验证时：拒绝启动，抛 ValueError，不建服务。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="0.0.0.0", port=0)
    with pytest.raises(ValueError, match="不是本机回环地址"):
        create_server(config=config, budget_tracker=tracker)


def test_non_loopback_lan_address_without_auth_refuses_to_start(tmp_path):
    """局域网地址（192.168.x.x）没配身份验证时：同样拒绝启动。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="192.168.1.5", port=0)
    with pytest.raises(ValueError, match="不是本机回环地址"):
        create_server(config=config, budget_tracker=tracker)


def test_non_loopback_with_auth_starts(tmp_path):
    """非回环地址配了身份验证函数时：允许启动（绑定 0.0.0.0，临时端口）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="0.0.0.0", port=0)
    server, state = create_server(
        config=config, budget_tracker=tracker, authenticator=lambda handler: True
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        status, _ct, data = _request(port, "GET", "/api/status")
        assert status == 200
        assert "deepseek" in data["model_status"]
    finally:
        _stop_test_server(server)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_without_auth_starts(tmp_path, host):
    """回环地址没配身份验证时：默认行为不变，允许启动。
    （::1 只测 _is_loopback_host 的判断逻辑，不在这里绑定——标准 socket 默认是 IPv4。）"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host=host, port=0)
    server, state = create_server(config=config, budget_tracker=tracker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert port > 0
        assert state is not None
        status, _ct, data = _request(port, "GET", "/api/status")
        assert status == 200
    finally:
        _stop_test_server(server)


def test_main_refuses_non_loopback_without_auth(capsys, monkeypatch):
    """命令行 --host 0.0.0.0 且没配身份验证时：main 打印清楚原因，返回 1，不启动。"""
    monkeypatch.delenv("AGENT_WEB_HOST", raising=False)
    monkeypatch.delenv("AGENT_WEB_PORT", raising=False)
    from web.server import main

    assert main(["--host", "0.0.0.0"]) == 1
    captured = capsys.readouterr()
    assert "启动失败" in captured.err
    assert "不是本机回环地址" in captured.err
    assert "身份验证" in captured.err


# ---------- 长期记忆管理（忘记 / 纠正的网页界面） ----------


@pytest.fixture
def running_server_with_real_mem0(tmp_path):
    """启动一个带真实 Mem0Store（底下是 FakeMemory，见 tests/conftest.py）
    的测试服务，供记忆管理接口的测试用。

    库里已种好三条记忆（都是 BOOS 亲口说的、已确认）：
    “我喜欢喝咖啡。”（fake-1）、“我喜欢喝茶。”（fake-2）、
    “我叫小宋。”（fake-3）。
    返回 (port, state, store)：测试里可以直接看 store.get_all()
    核对记忆到底变没变。
    """
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    store.add_memory_text("我喜欢喝茶。")
    store.add_memory_text("我叫小宋。")
    server, port, state = _start_test_server(tmp_path, mem0_store=store)
    try:
        yield port, state, store
    finally:
        _stop_test_server(server)


def _memory_texts(port):
    """GET /api/memories，返回当前库里的记忆文本列表（按返回顺序）。"""
    status, _ct, data = _request(port, "GET", "/api/memories")
    assert status == 200
    return [item["text"] for item in data["memories"]]


def test_memories_list_returns_memories_with_metadata(running_server_with_real_mem0):
    """GET /api/memories：返回全部三条记忆，每条带 id、文本和来源元数据
    （source/created_at/confidence/user_confirmed/confirmed）。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(port, "GET", "/api/memories")
    assert status == 200
    assert data["mem0_enabled"] is True
    assert [item["text"] for item in data["memories"]] == [
        "我喜欢喝咖啡。",
        "我喜欢喝茶。",
        "我叫小宋。",
    ]
    assert [item["id"] for item in data["memories"]] == ["fake-1", "fake-2", "fake-3"]
    for item in data["memories"]:
        meta = item["metadata"]
        assert meta["source"] == "user"
        assert meta["user_confirmed"] is True
        assert meta["confirmed"] is True
        assert meta["confidence"] == 1.0
        assert meta["created_at"]  # 形成时间有值


def test_memories_list_without_store_returns_empty_and_disabled(running_server):
    """没带记忆库启动时：GET /api/memories 返回 mem0_enabled False、列表为空，
    不报错，页面据此显示“记忆未启用”。"""
    port, _state = running_server
    status, _ct, data = _request(port, "GET", "/api/memories")
    assert status == 200
    assert data["mem0_enabled"] is False
    assert data["memories"] == []


def test_protection_events_returns_recorded_events(running_server_with_real_mem0):
    """GET /api/protection-events：返回记下来的保护事件（新的在前）。

    保护事件是失败关闭的可见留痕：快照失败、恢复失败时记一条，
    面板上“保护记录”那块读这个接口展示。"""
    port, _state, store = running_server_with_real_mem0
    store._record_protection_event(
        "snapshot_failed", "提取前快照两次都失败，本轮没有存记忆。"
    )
    status, _ct, data = _request(port, "GET", "/api/protection-events")
    assert status == 200
    assert data["mem0_enabled"] is True
    assert len(data["events"]) == 1
    assert data["events"][0]["kind"] == "snapshot_failed"
    assert data["events"][0]["reason"]


def test_protection_events_without_store_returns_empty_and_disabled(running_server):
    """没带记忆库启动时：GET /api/protection-events 返回 mem0_enabled
    False、列表为空，不报错，页面据此显示“暂无保护记录”。"""
    port, _state = running_server
    status, _ct, data = _request(port, "GET", "/api/protection-events")
    assert status == 200
    assert data["mem0_enabled"] is False
    assert data["events"] == []


def test_memory_scope_returns_deletion_scope(running_server_with_real_mem0):
    """GET /api/memories/<id>/scope：返回删除范围——找得到这条记忆、
    有 1 条 history 审计行、删的范围和不删的范围都列出来了。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(port, "GET", "/api/memories/fake-1/scope")
    assert status == 200
    assert data["found"] is True
    assert data["memory"]["id"] == "fake-1"
    assert data["memory"]["text"] == "我喜欢喝咖啡。"
    assert data["history_count"] == 1
    assert len(data["history_entries"]) == 1
    assert data["history_entries"][0]["event"] == "ADD"
    # 删的范围：向量库 + history 审计行。
    assert len(data["in_scope"]) == 2
    assert "向量库" in data["in_scope"][0]["where"]
    # 不删的范围：原始对话记录、日志、BOOS 自己的备份、Chroma 文件碎片。
    assert len(data["out_of_scope"]) == 4
    assert "messages" in data["out_of_scope"][0]["where"]
    assert "备份" in data["out_of_scope"][2]["where"]


def test_memory_scope_unknown_id_returns_404(running_server_with_real_mem0):
    """删除范围预览查一个不存在的 id：返回 404，说明这条记忆已经不在库里了。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(port, "GET", "/api/memories/fake-999/scope")
    assert status == 404
    assert "找不到这条记忆" in data["error"]


def test_memory_scope_without_store_returns_503(running_server):
    """没带记忆库时：删除范围预览返回 503“记忆未启用”，不崩。"""
    port, _state = running_server
    status, _ct, data = _request(port, "GET", "/api/memories/fake-1/scope")
    assert status == 503
    assert "记忆未启用" in data["error"]


def test_forget_without_confirmation_returns_400_and_keeps_memory(
    running_server_with_real_mem0,
):
    """忘一条记忆但没确认（confirmed 不是 true / 干脆没传）时：
    返回 400，记忆还在库里，一条都没少——没确认就不删。"""
    port, _state, store = running_server_with_real_mem0
    status, _ct, data = _request(
        port, "POST", "/api/memories/fake-1/forget", {"confirmed": False}
    )
    assert status == 400
    assert "确认" in data["error"]
    status, _ct, data = _request(port, "POST", "/api/memories/fake-1/forget", {})
    assert status == 400
    assert "确认" in data["error"]
    assert "我喜欢喝咖啡。" in _memory_texts(port)
    assert len(store.get_all()) == 3


def test_forget_with_confirmation_deletes_memory(running_server_with_real_mem0):
    """忘一条记忆且确认了（confirmed 为 true）时：返回 deleted True，
    列表里没这条了，删除范围预览也查不到了（404）。"""
    port, _state, store = running_server_with_real_mem0
    status, _ct, data = _request(
        port, "POST", "/api/memories/fake-1/forget", {"confirmed": True}
    )
    assert status == 200
    assert data["ok"] is True
    assert data["deleted"] is True
    assert "我喜欢喝咖啡。" not in _memory_texts(port)
    assert "我喜欢喝茶。" in _memory_texts(port)
    assert "我叫小宋。" in _memory_texts(port)
    assert len(store.get_all()) == 2
    # 删完验干净了：history 也清了，scope 查不到。
    assert store.history("fake-1") == []
    status, _ct, _body = _request(port, "GET", "/api/memories/fake-1/scope")
    assert status == 404


def test_forget_unknown_memory_returns_not_deleted(running_server_with_real_mem0):
    """忘一条已经不在库里的记忆：返回 ok True、deleted False，
    不报错，其他记忆不受影响。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(
        port, "POST", "/api/memories/fake-999/forget", {"confirmed": True}
    )
    assert status == 200
    assert data["ok"] is True
    assert data["deleted"] is False
    assert len(_memory_texts(port)) == 3


def test_forget_without_store_returns_503(running_server):
    """没带记忆库时：忘记接口返回 503“记忆未启用”，不删任何东西。"""
    port, _state = running_server
    status, _ct, data = _request(
        port, "POST", "/api/memories/fake-1/forget", {"confirmed": True}
    )
    assert status == 503
    assert "记忆未启用" in data["error"]


def test_correct_success_updates_memory(running_server_with_real_mem0):
    """纠正一条记忆（页面上选了 id、给了新说法）：返回 updated，
    列表里旧文本没了、新文本在同一条 id 上。"""
    port, _state, store = running_server_with_real_mem0
    status, _ct, data = _request(
        port,
        "POST",
        "/api/memories/correct",
        {"memory_id": "fake-1", "new_text": "我喜欢喝白开水。"},
    )
    assert status == 200
    assert data["status"] == "updated"
    assert data["memory_id"] == "fake-1"
    assert data["old_text"] == "我喜欢喝咖啡。"
    assert data["new_text"] == "我喜欢喝白开水。"
    texts = _memory_texts(port)
    assert "我喜欢喝咖啡。" not in texts
    assert "我喜欢喝白开水。" in texts
    assert "我喜欢喝茶。" in texts
    # 纠正后来源变成 user、已确认（BOOS 亲口纠正的）。
    corrected = [item for item in store.get_all() if item["id"] == "fake-1"][0]
    assert corrected["metadata"]["source"] == "user"
    assert corrected["metadata"]["confirmed"] is True


def test_correct_ambiguous_returns_candidates_and_changes_nothing(
    running_server_with_real_mem0,
):
    """只给旧内容、没选 id，且旧内容对上两条相似记忆时：
    返回 ambiguous、带两条候选，库里一条都没改——不猜、不改。"""
    port, _state, store = running_server_with_real_mem0
    status, _ct, data = _request(
        port,
        "POST",
        "/api/memories/correct",
        {"old_text": "我喜欢喝点热的", "new_text": "我喜欢喝白开水。"},
    )
    assert status == 200
    assert data["status"] == "ambiguous"
    assert {cand["id"] for cand in data["candidates"]} == {"fake-1", "fake-2"}
    assert data["memory_id"] is None
    # 一条都没改：三条原文都在。
    assert _memory_texts(port) == ["我喜欢喝咖啡。", "我喜欢喝茶。", "我叫小宋。"]
    assert len(store.get_all()) == 3


def test_correct_with_memory_id_resolves_ambiguity(running_server_with_real_mem0):
    """ambiguous 之后，页面让 BOOS 挑了一条、按 id 再提交：
    只改挑中的那条，另一条相似记忆不动。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(
        port,
        "POST",
        "/api/memories/correct",
        {"old_text": "我喜欢喝点热的", "new_text": "我喜欢喝白开水。"},
    )
    assert data["status"] == "ambiguous"
    # 挑候选里的“喝茶”那条改。
    tea_id = next(
        cand["id"] for cand in data["candidates"] if "茶" in cand["text"]
    )
    status, _ct, data = _request(
        port,
        "POST",
        "/api/memories/correct",
        {"memory_id": tea_id, "new_text": "我喜欢喝白开水。"},
    )
    assert status == 200
    assert data["status"] == "updated"
    assert data["memory_id"] == tea_id
    texts = _memory_texts(port)
    assert "我喜欢喝茶。" not in texts
    assert "我喜欢喝白开水。" in texts
    # 没挑中的“喝咖啡”那条原样不动。
    assert "我喜欢喝咖啡。" in texts


def test_correct_not_found_returns_status_and_changes_nothing(
    running_server_with_real_mem0,
):
    """纠正一个库里没有的旧内容：返回 not_found，库里一条都没改。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(
        port,
        "POST",
        "/api/memories/correct",
        {"old_text": "今天天气真不错", "new_text": "今天下雨了。"},
    )
    assert status == 200
    assert data["status"] == "not_found"
    assert _memory_texts(port) == ["我喜欢喝咖啡。", "我喜欢喝茶。", "我叫小宋。"]


def test_correct_unchanged_returns_status(running_server_with_real_mem0):
    """新内容和旧内容一样：返回 unchanged，不用改，库里不动。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(
        port,
        "POST",
        "/api/memories/correct",
        {"memory_id": "fake-1", "new_text": "我喜欢喝咖啡。"},
    )
    assert status == 200
    assert data["status"] == "unchanged"
    assert _memory_texts(port) == ["我喜欢喝咖啡。", "我喜欢喝茶。", "我叫小宋。"]


def test_correct_missing_new_text_returns_400(running_server_with_real_mem0):
    """纠正时没给新内容：返回 400，库里不动。"""
    port, _state, _store = running_server_with_real_mem0
    status, _ct, data = _request(
        port, "POST", "/api/memories/correct", {"memory_id": "fake-1", "new_text": "  "}
    )
    assert status == 400
    assert "纠正后的内容" in data["error"]
    status, _ct, data = _request(
        port, "POST", "/api/memories/correct", {"old_text": "我喜欢喝咖啡。"}
    )
    assert status == 400
    assert _memory_texts(port) == ["我喜欢喝咖啡。", "我喜欢喝茶。", "我叫小宋。"]


def test_correct_without_store_returns_503(running_server):
    """没带记忆库时：纠正接口返回 503“记忆未启用”。"""
    port, _state = running_server
    status, _ct, data = _request(
        port,
        "POST",
        "/api/memories/correct",
        {"old_text": "我喜欢喝咖啡。", "new_text": "我喜欢喝茶。"},
    )
    assert status == 503
    assert "记忆未启用" in data["error"]


def test_root_page_contains_memory_panel(running_server):
    """聊天页面里有长期记忆面板：侧边栏按钮、面板容器、
    指向 /api/memories 的加载逻辑都在，页面样式和原来一致。"""
    port, _state = running_server
    status, content_type, body = _request(port, "GET", "/")
    assert status == 200
    assert "长期记忆" in body
    assert 'id="memoriesBtn"' in body
    assert 'id="memoriesPanel"' in body
    assert 'id="backToChat"' in body
    assert "/api/memories" in body
    assert "/scope" in body
    assert "/forget" in body
    assert "/api/memories/correct" in body


# ---------- 聊天里的纠错提示（correction 字段的展示） ----------
# 后端 ChatState.ask 在纠错轮里给 /api/chat 的返回多带一个
# "correction" 键；页面按它展示“改好了 / 请挑一条 / 改失败了”。
# 下面是 HTTP 层的覆盖：返回形状、候选列表、挑候选的回复走正常
# 聊天通道。页面 JS 的渲染逻辑本身没有浏览器跑不了，
# 只做静态检查（关键文案和按钮逻辑在页面源码里）+ node 语法检查。


class CorrectionFakeStore(FakeMem0Store):
    """假 mem0 store：add_turn 返回预设的 correction 字典，
    模拟一次聊天纠错的结果（updated / ambiguous 等），不碰真实 mem0。

    correction_to_return 是完整的 correction 字典
    （{"status", "memory_id", "old_text", "new_text", "candidates",
    "reason"}）；设成 None 时按普通轮次处理（返回 {}，不带 correction）。
    """

    def __init__(self, correction=None, memories=None):
        super().__init__(memories=memories)
        self.correction_to_return = correction

    def add_turn(self, messages):
        self.saved_turns.append(list(messages))
        if self.correction_to_return is None:
            return {}
        return {"correction": self.correction_to_return}


_UPDATED_CORRECTION = {
    "status": "updated",
    "memory_id": "fake-1",
    "old_text": "我喜欢喝咖啡。",
    "new_text": "我喜欢喝茶。",
    "candidates": [],
    "reason": "",
}


@pytest.fixture
def running_server_with_correction_store(tmp_path):
    """启动一个带 CorrectionFakeStore 的测试服务，默认下一轮纠错
    返回 updated（咖啡→茶）。测试里改 store.correction_to_return
    就能换别的纠错结果。返回 (port, state, store)。"""
    store = CorrectionFakeStore(correction=dict(_UPDATED_CORRECTION))
    server, port, state = _start_test_server(tmp_path, mem0_store=store)
    try:
        yield port, state, store
    finally:
        _stop_test_server(server)


def test_chat_response_carries_correction_updated(
    running_server_with_correction_store, monkeypatch
):
    """纠错成功时：/api/chat 返回里带 correction（updated），
    旧文本、新文本、id 都在，回答本身不受影响。"""
    port, _state, _store = running_server_with_correction_store
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    status, _ct, data = _request(
        port, "POST", "/api/chat", {"message": "你记错了，我喜欢喝茶。"}
    )
    assert status == 200
    assert data["mode"] == "real"
    assert data["reply"] == "假回答"
    corr = data["correction"]
    assert corr["status"] == "updated"
    assert corr["memory_id"] == "fake-1"
    assert corr["old_text"] == "我喜欢喝咖啡。"
    assert corr["new_text"] == "我喜欢喝茶。"
    assert corr["candidates"] == []


def test_chat_response_carries_correction_ambiguous(
    running_server_with_correction_store, monkeypatch
):
    """纠错分不清改哪条时：返回 correction（ambiguous），带两条候选
    （每条有 id 和文本），库和回答本身不受影响。"""
    port, _state, store = running_server_with_correction_store
    store.correction_to_return = {
        "status": "ambiguous",
        "memory_id": None,
        "old_text": "我喜欢喝点热的",
        "new_text": "我喜欢喝白开水。",
        "candidates": [
            {"id": "fake-1", "text": "我喜欢喝咖啡。"},
            {"id": "fake-2", "text": "我喜欢喝茶。"},
        ],
        "reason": "对上了多条相似记忆。",
    }
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    status, _ct, data = _request(
        port, "POST", "/api/chat", {"message": "你记错了，我喜欢喝白开水。"}
    )
    assert status == 200
    corr = data["correction"]
    assert corr["status"] == "ambiguous"
    assert corr["memory_id"] is None
    assert corr["new_text"] == "我喜欢喝白开水。"
    assert [cand["id"] for cand in corr["candidates"]] == ["fake-1", "fake-2"]
    assert [cand["text"] for cand in corr["candidates"]] == [
        "我喜欢喝咖啡。",
        "我喜欢喝茶。",
    ]


@pytest.mark.parametrize(
    "status,reason",
    [
        ("failed", "改的时候出错了。"),
        ("unchanged", ""),
        ("not_found", "库里没有这条记忆。"),
    ],
)
def test_chat_response_carries_other_correction_statuses(
    running_server_with_correction_store, monkeypatch, status, reason
):
    """failed / unchanged / not_found 的纠错结果也原样带回给页面，
    页面按 status 分别展示（失败时带 reason）。"""
    port, _state, store = running_server_with_correction_store
    store.correction_to_return = {
        "status": status,
        "memory_id": None,
        "old_text": "我喜欢喝咖啡。",
        "new_text": "我喜欢喝茶。",
        "candidates": [],
        "reason": reason,
    }
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    _status, _ct, data = _request(
        port, "POST", "/api/chat", {"message": "你记错了，我喜欢喝茶。"}
    )
    assert data["correction"]["status"] == status
    assert data["correction"]["reason"] == reason


def test_chat_response_without_correction_has_no_correction_key(
    running_server_with_mem0, monkeypatch
):
    """普通轮次（add_turn 没返回 correction）时：返回里没有
    correction 键，页面和原来一模一样，不多出任何提示。"""
    port, _state, _store = running_server_with_mem0
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "你好"})
    assert status == 200
    assert data["mode"] == "real"
    assert "correction" not in data


def test_chat_correction_selection_reply_flows_through_chat(
    running_server_with_correction_store, monkeypatch
):
    """挑候选的回复走正常聊天通道：ambiguous 之后，BOOS 回“第 1 条”，
    它就是一条普通的 /api/chat 消息——模型照常回答，
    add_turn 也照常收到它（后端在那里认出这是挑候选）。
    这是候选按钮实际发的文本（页面点“选第 1 条”发的也是这一句）。"""
    port, _state, store = running_server_with_correction_store
    store.correction_to_return = {
        "status": "ambiguous",
        "memory_id": None,
        "old_text": "我喜欢喝点热的",
        "new_text": "我喜欢喝白开水。",
        "candidates": [
            {"id": "fake-1", "text": "我喜欢喝咖啡。"},
            {"id": "fake-2", "text": "我喜欢喝茶。"},
        ],
        "reason": "",
    }
    monkeypatch.setattr("web.server.run_question", _fake_success_run_question)
    status, _ct, data = _request(
        port, "POST", "/api/chat", {"message": "你记错了，我喜欢喝白开水。"}
    )
    assert status == 200
    assert data["correction"]["status"] == "ambiguous"
    # 下一轮直接回“第 1 条”：走正常聊天，回答照常，add_turn 收到这句。
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "第 1 条"})
    assert status == 200
    assert data["mode"] == "real"
    assert data["reply"] == "假回答"
    assert len(store.saved_turns) == 2
    assert store.saved_turns[-1][-2] == {"role": "user", "content": "第 1 条"}


def test_root_page_renders_correction_field(running_server):
    """聊天页面里有纠错提示的渲染逻辑：读到 correction 键才展示；
    updated / ambiguous / failed 各有对应的中文文案；
    候选按钮点下去发的是“第 N 条”（走已有的 send() 聊天通道，
    不走新接口），并留了直接在输入框回复的文字提示。"""
    port, _state = running_server
    status, _ct, body = _request(port, "GET", "/")
    assert status == 200
    assert "renderCorrection" in body
    assert "data.correction" in body
    assert "已更正：把" in body
    assert "这条已经记对了，不用改。" in body
    assert "没找到要改的旧记忆" in body
    assert "没改成：" in body
    assert "分不清改哪条，请选一条" in body
    assert "选第 ' + (i + 1) + ' 条" in body
    assert "send('第 ' + (i + 1) + ' 条')" in body
    assert "回复“第 1 条”" in body


def test_page_script_syntax_is_valid(tmp_path):
    """页面里 <script> 的 JS 语法能过 node --check。

    这只是语法检查，不是浏览器行为测试：按钮点下去是不是真发
    消息、候选列表是不是真显示出来，还得 BOOS 在本地浏览器里点一遍。
    没装 node 的机器上跳过（不拦测试）。"""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("没装 node，跳过页面 JS 语法检查。")
    html = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    assert match is not None, "页面里没找到 <script> 块"
    script_path = tmp_path / "page.js"
    script_path.write_text(match.group(1), encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(script_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"页面 JS 语法错误：{result.stderr}"


def _extract_js_function(script, name):
    """从页面 <script> 里按花括号配对抠出一个 function 的完整源码。"""
    start = script.index("function " + name + "(")
    i = script.index("{", start)
    depth = 0
    for j in range(i, len(script)):
        if script[j] == "{":
            depth += 1
        elif script[j] == "}":
            depth -= 1
            if depth == 0:
                return script[start : j + 1]
    raise AssertionError("没找到函数 " + name)


def test_page_markdown_renders_bold_italic_list_and_escapes_html(tmp_path):
    """页面渲染 Markdown：**加粗**、*斜体*、- 列表出来，HTML 先转义。

    做法和语法检查一样：把 renderMarkdown 相关函数从页面里抠出来，
    用 node 真跑一遍断言。没装 node 的机器上跳过（不拦测试）。
    """
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("没装 node，跳过页面 Markdown 渲染检查。")
    html = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    assert match is not None, "页面里没找到 <script> 块"
    script = match.group(1)
    funcs = _extract_js_function(script, "inlineMarkdown") + "\n" + _extract_js_function(
        script, "renderMarkdown"
    )
    checks = """
const assert = require('assert');
assert.strictEqual(renderMarkdown('**加粗**'), '<strong>加粗</strong>');
assert.strictEqual(renderMarkdown('*斜体*'), '<em>斜体</em>');
assert.strictEqual(renderMarkdown('**加粗**和*斜体*'), '<strong>加粗</strong>和<em>斜体</em>');
assert.strictEqual(renderMarkdown('第一行\\n第二行'), '第一行<br>第二行');
assert.strictEqual(renderMarkdown('- 苹果\\n- 香蕉'), '<ul><li>苹果</li><li>香蕉</li></ul>');
assert.strictEqual(renderMarkdown('- **加粗**苹果'), '<ul><li><strong>加粗</strong>苹果</li></ul>');
assert.strictEqual(renderMarkdown('<script>alert(1)</script>'), '&lt;script&gt;alert(1)&lt;/script&gt;');
assert.strictEqual(renderMarkdown('**<b>hi</b>**'), '<strong>&lt;b&gt;hi&lt;/b&gt;</strong>');
assert.strictEqual(renderMarkdown('5*3=15'), '5*3=15');
assert.strictEqual(renderMarkdown(''), '');
console.log('markdown ok');
"""
    js_path = tmp_path / "md.js"
    js_path.write_text(funcs + "\n" + checks, encoding="utf-8")
    result = subprocess.run(
        [node, str(js_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"页面 Markdown 渲染不对：{result.stderr}"


def test_page_chat_bubble_uses_markdown_renderer():
    """聊天区气泡用 renderMarkdown 渲染（** 加粗不再原样显示 * 号）。"""
    html = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "bubble.innerHTML = renderMarkdown(text)" in html
    assert "function renderMarkdown" in html


# ---------- 第 4 批界面：35 号排队、36 号倒序+返回按钮、39 号宽面板、40 号后台总结 ----------
# 页面 JS 没有浏览器跑不了，只做静态检查（关键结构和文案在页面源码里）
# + node 语法检查（见 test_page_script_syntax_is_valid）。行为本身
# （排队真不丢、点新对话真不卡）靠 BOOS 在本地浏览器里点一遍。


def _read_page_html():
    return (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(
        encoding="utf-8"
    )


def test_page_send_queue_keeps_input_when_busy():
    """35 号：忙的时候输入不丢——排队结构在，排队文案在，忙完接着发。"""
    html = _read_page_html()
    assert "const sendQueue = []" in html
    assert "sendQueue.push({ text, wrap })" in html
    assert "排队中，忙完就发" in html
    assert "已排队 ' + sendQueue.length + ' 条，忙完就发" in html
    assert "sendQueue.shift()" in html
    assert "async function _sendNow(text, shownWrap)" in html


def test_page_memories_newest_first_and_back_button_bottom():
    """36 号：记忆列表新的在前（按 created_at 倒序），返回按钮上下各一个。"""
    html = _read_page_html()
    assert "return tb - ta;" in html
    assert 'id="backToChat"' in html
    assert 'id="backToChatBottom"' in html
    assert "← 返回对话" in html


def test_page_memories_panel_wider_than_chat():
    """39 号：记忆面板比对话宽、布局分开——面板 1100px，对话 860px。"""
    html = _read_page_html()
    assert "max-width: 1100px" in html
    assert "max-width: 860px" in html


def test_page_new_chat_opens_immediately_and_summarizes_in_background():
    """40 号：点新对话立刻开新的（先清聊天区、显示“已开始新对话”），
    总结放后台跑（pollClosedSummary 轮询，等好了再补摘要卡片），
    重复点不重复开（newChatResetting 防重入）。"""
    html = _read_page_html()
    assert "let newChatResetting = false" in html
    assert "已开始新对话" in html
    assert "async function pollClosedSummary(closedId, triesLeft, anchorWrap)" in html
    assert "pollClosedSummary(data.closed_id, 30, startedWrap)" in html
