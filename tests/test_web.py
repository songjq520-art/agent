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

import pytest

from config import BudgetTracker
from web.server import create_server


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


@pytest.fixture
def running_server(tmp_path):
    """启动一个用临时预算文件的测试服务，测试完关掉。

    返回 (port, state)：port 是实际监听的端口，state 是 ChatState，
    测试里可以直接看 state.messages 验证历史有没有追加。
    """
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    server, state = create_server(host="127.0.0.1", port=0, budget_tracker=tracker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], state
    finally:
        server.shutdown()
        server.server_close()


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

    def _fake_run_question(prompt, budget_tracker=None, messages=None):
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
    """调用抛错时：返回 502 和错误信息，历史不动。"""
    port, state = running_server

    def _boom(prompt, budget_tracker=None, messages=None):
        raise RuntimeError("模拟调用失败")

    monkeypatch.setattr("web.server.run_question", _boom)
    status, _ct, data = _request(port, "POST", "/api/chat", {"message": "你好"})
    assert status == 502
    assert "调用失败" in data["error"]
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

    def _fake_run_question(prompt, budget_tracker=None, messages=None):
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
