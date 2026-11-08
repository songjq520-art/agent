# tests/test_web.py（本地网页入口的测试）
# 作用：验证 web/server.py 的四个接口和对话历史逻辑。
# 原理：用临时端口（port 0，系统自动分配）和临时预算文件启动真实服务，
#       再用标准库 http.client 发请求验证，不调用真实模型 API。
#       需要“真实调用成功”的场景时，用 monkeypatch 把 run_question 换成
#       返回假结果的函数，不产生费用。
# 注意：这些测试不依赖网络、不依赖 API Key；无 Key 时走演示模式。

import http.client
import json
import threading
import time

import pytest

from config import BudgetTracker
from web.server import ServerConfig, create_server


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


def _start_test_server(tmp_path, authenticator=None):
    """用临时端口和临时预算文件启动一个测试服务，返回 (server, port, state)。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="127.0.0.1", port=0)
    server, state = create_server(
        config=config, budget_tracker=tracker, authenticator=authenticator
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
    """
    server, port, state = _start_test_server(tmp_path)
    try:
        yield port, state
    finally:
        _stop_test_server(server)


def test_get_root_serves_chat_page(running_server):
    """GET / 返回聊天页面：200、HTML、里面有标题和输入框。"""
    port, _state = running_server
    status, content_type, body = _request(port, "GET", "/")
    assert status == 200
    assert "text/html" in content_type
    assert "我的 Agent" in body
    assert 'id="input"' in body
    assert 'id="send"' in body


def test_get_status_returns_models_and_budget(running_server):
    """GET /api/status 返回模型状态和本月预算：有 deepseek、预算 5 美元、剩余 5 美元。"""
    port, _state = running_server
    status, content_type, data = _request(port, "GET", "/api/status")
    assert status == 200
    assert "application/json" in content_type
    assert "deepseek" in data["model_status"]
    assert data["monthly_budget_usd"] == 5.0
    assert data["total_cost_usd"] == 0.0
    assert data["remaining_usd"] == 5.0
    assert data["budget_corrupted"] is False


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

    def _fake_run_question(prompt, budget_tracker=None, messages=None, memory_store=None, **kwargs):
        return {
            "mode": "real",
            "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
            "result": {"text": "假回答", "model": "deepseek-flash", "cost_usd": 0.0001, "usage": {"total_tokens": 10}},
            "message": "真实调用完成。",
            "model_status": {},
        }

    monkeypatch.setattr("web.server.run_question", _fake_run_question)
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

    def _boom(prompt, budget_tracker=None, messages=None, memory_store=None, **kwargs):
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

    def _fake_run_question(prompt, budget_tracker=None, messages=None, memory_store=None, **kwargs):
        return {
            "mode": "real",
            "route": {"task_type": "question", "provider": "deepseek", "model": "deepseek-flash"},
            "result": {"text": "假回答", "model": "deepseek-flash", "cost_usd": 0.0001, "usage": {"total_tokens": 10}},
            "message": "真实调用完成。",
            "model_status": {},
        }

    monkeypatch.setattr("web.server.run_question", _fake_run_question)
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

    def _boom(prompt, budget_tracker=None, messages=None, memory_store=None, **kwargs):
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


def _slow_fake_run_question(prompt, budget_tracker=None, messages=None, memory_store=None, **kwargs):
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

    def _timed_fake(prompt, budget_tracker=None, messages=None, memory_store=None, **kwargs):
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
