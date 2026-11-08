# tests/test_memory_proposals.py（模型驱动记忆提议的测试）
# 作用：验证 Memory V1 的完整提议流程——模型提取 → 后端创建真实提议 →
#       用户确认/拒绝 → 确认后写入长期记忆。
# 原理：用临时记忆文件（tmp_path）做记忆库，不碰真实的 memory/memories.json；
#       模型调用全部用 Mock 假装，不联网、不产生费用。
# 覆盖：
#   - 提议对象：真实 ID、pending 状态、字段校验
#   - 提议暂存：创建、列表、确认（写入文件）、拒绝（不保存）、去重、保存失败时保留
#   - 提取解析：合法 JSON、代码块包裹的 JSON、非法 JSON、非法条目、调用失败
#   - run_question 提取：聊天成功后提取建提议；提取失败不影响聊天；默认不提取
#   - 记忆去重：相同内容不重复写文件
#   - 网页接口：提议列表、确认、拒绝、404、保存失败、端到端（聊天→提取→确认）
# 注意：这些测试不依赖网络、不依赖 API Key；需要真实调用的场景用 Mock。
# 安全：模型文字绝不能直接被当成真实提议——所有提议必须由后端创建；
#       测试里用 Mock 返回的 JSON 也要经过解析与校验才建提议。

import http.client
import json
import threading
from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker
from main import (
    _parse_extraction_json,
    extract_memory_candidates,
    run_question,
)
from memory.memories import MemoryStore
from memory.proposals import ProposalStore, create_proposal
from web.server import ServerConfig, create_server


def _fake_ok_response(text="你好，我是 DeepSeek。", prompt_tokens=100, completion_tokens=50):
    """假装 DeepSeek 返回成功：固定回答 + 固定 Token 用量。"""
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    response.raise_for_status.return_value = None
    return response


def _fake_extraction_response(candidates, prompt_tokens=200, completion_tokens=100):
    """假装 DeepSeek 返回提取结果：JSON 文本 + 固定 Token 用量。"""
    return _fake_ok_response(
        text=json.dumps(candidates, ensure_ascii=False),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _store(tmp_path):
    """一个用临时文件的空记忆库。"""
    return MemoryStore(file_path=tmp_path / "memories.json")


# ---------- 提议对象 ----------


def test_create_proposal_has_real_id_and_pending_status():
    """提议有后端生成的真实 ID（prop- 开头）和 pending 状态，不是模型给的。"""
    proposal = create_proposal("每月预算 5 美元。", ["core"], "inferred", "长期预算规则")
    assert proposal["id"].startswith("prop-")
    assert len(proposal["id"]) == 13  # "prop-" + 8 位
    assert proposal["status"] == "pending"
    assert proposal["text"] == "每月预算 5 美元。"
    assert proposal["layers"] == ["core"]
    assert proposal["source"] == "inferred"
    assert proposal["reason"] == "长期预算规则"
    assert proposal["date"]  # 有创建日期


def test_create_proposal_ids_are_unique():
    """每个提议的 ID 都不一样。"""
    first = create_proposal("记忆一。", ["core"], "inferred", "")
    second = create_proposal("记忆二。", ["core"], "inferred", "")
    assert first["id"] != second["id"]


def test_create_proposal_rejects_bad_content():
    """提议内容不合法时不创建（空内容、非法层、非法来源都报 ValueError）。"""
    with pytest.raises(ValueError):
        create_proposal("", ["core"], "inferred", "")
    with pytest.raises(ValueError):
        create_proposal("记忆。", ["not_a_layer"], "inferred", "")
    with pytest.raises(ValueError):
        create_proposal("记忆。", ["core"], "not_a_source", "")


# ---------- 提议暂存 ----------


def test_proposal_store_add_and_list_pending():
    """创建提议后出现在待确认列表里；列表返回的是深拷贝。"""
    store = ProposalStore()
    created = store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0]["id"] == created["id"]
    assert pending[0]["status"] == "pending"
    # 改返回的列表不影响存着的提议。
    pending[0]["text"] = "改掉了"
    assert store.get(created["id"])["text"] == "每月预算 5 美元。"


def test_proposal_store_add_duplicate_text_returns_existing():
    """内容完全相同的提议不重复创建，返回已有的那条。"""
    store = ProposalStore()
    first = store.add("每月预算 5 美元。", ["core"], "inferred", "理由一")
    second = store.add("每月预算 5 美元。", ["core"], "inferred", "理由二")
    assert second["id"] == first["id"]
    assert len(store.list_pending()) == 1


def test_proposal_store_confirm_writes_to_memory_file(tmp_path):
    """确认提议后，记忆写进长期记忆文件，提议从待确认列表移除。"""
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")
    saved = proposal_store.confirm(proposal["id"], memory_store)
    assert saved["text"] == "每月预算 5 美元。"
    assert len(memory_store.list_all()) == 1
    assert memory_store.list_all()[0]["text"] == "每月预算 5 美元。"
    assert proposal_store.list_pending() == []
    assert proposal_store.get(proposal["id"]) is None


def test_proposal_store_confirm_missing_raises(tmp_path):
    """确认不存在的提议时报 ValueError，不写文件。"""
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    with pytest.raises(ValueError):
        proposal_store.confirm("prop-doesnotexist", memory_store)
    assert memory_store.list_all() == []


def test_proposal_store_confirm_twice_raises(tmp_path):
    """同一提议确认两次时，第二次报 ValueError（已处理，不能重复确认）。"""
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "")
    proposal_store.confirm(proposal["id"], memory_store)
    with pytest.raises(ValueError):
        proposal_store.confirm(proposal["id"], memory_store)
    # 只写了一条，没有重复。
    assert len(memory_store.list_all()) == 1


def test_proposal_store_reject_removes_without_saving(tmp_path):
    """拒绝提议后，提议从列表移除，记忆文件里没有这条。"""
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "")
    assert proposal_store.reject(proposal["id"]) is True
    assert proposal_store.list_pending() == []
    assert memory_store.list_all() == []


def test_proposal_store_reject_missing_returns_false():
    """拒绝不存在的提议时返回 False，不报错。"""
    assert ProposalStore().reject("prop-doesnotexist") is False


def test_proposal_store_confirm_save_failure_keeps_proposal(tmp_path, monkeypatch):
    """保存失败时提议保留在待确认列表（用户可稍后重试），并报异常。"""
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "")

    def _boom(self, _memories):
        raise RuntimeError("磁盘写不进去")

    monkeypatch.setattr(MemoryStore, "_write", _boom)
    with pytest.raises(RuntimeError):
        proposal_store.confirm(proposal["id"], memory_store)
    # 提议还在，记忆文件里没有。
    assert len(proposal_store.list_pending()) == 1
    assert memory_store.list_all() == []


# ---------- 提取解析 ----------


def test_parse_extraction_json_plain():
    """直接返回的 JSON 数组能解析。"""
    text = '[{"text": "预算 5 美元。", "layers": ["core"], "reason": "长期规则"}]'
    data = _parse_extraction_json(text)
    assert isinstance(data, list)
    assert data[0]["text"] == "预算 5 美元。"


def test_parse_extraction_json_fenced():
    """包在 ```json 代码块里的 JSON 能解析。"""
    text = '```json\n[{"text": "预算 5 美元。", "layers": ["core"], "reason": "长期规则"}]\n```'
    data = _parse_extraction_json(text)
    assert isinstance(data, list)
    assert data[0]["text"] == "预算 5 美元。"


def test_parse_extraction_json_with_surrounding_text():
    """前后带几句废话的 JSON 也能解析（取第一个 [ 到最后一个 ]）。"""
    text = '好的，我提取到了：\n[{"text": "预算 5 美元。", "layers": ["core"], "reason": "长期规则"}]\n以上就是结果。'
    data = _parse_extraction_json(text)
    assert isinstance(data, list)
    assert data[0]["text"] == "预算 5 美元。"


def test_parse_extraction_json_invalid_returns_none():
    """不是合法 JSON 时返回 None（调用方按“没有候选”处理）。"""
    assert _parse_extraction_json("这不是 JSON") is None
    assert _parse_extraction_json("") is None
    assert _parse_extraction_json('{"不是": "数组"}') is None
    # 文本里夹着一个合法空数组时，能把它抠出来（容错）。
    assert _parse_extraction_json("[] 不是") == []


def test_parse_extraction_json_empty_array():
    """空数组 [] 是合法的（表示没有值得保存的信息），解析出空列表。"""
    assert _parse_extraction_json("[]") == []


# ---------- 提取候选 ----------


def _mock_provider(text):
    """一个假 Provider：chat_messages 返回给定文本（不联网）。"""
    provider = Mock()
    provider.chat_messages.return_value = {"text": text, "model": "deepseek-flash"}
    return provider


def test_extract_candidates_valid_json():
    """模型返回合法 JSON 时，提取出已校验的候选。"""
    provider = _mock_provider(
        json.dumps(
            [
                {"text": "预算 5 美元。", "layers": ["core"], "reason": "长期规则"},
                {"text": "喜欢喝咖啡。", "layers": ["experience"], "reason": "个人偏好"},
            ],
            ensure_ascii=False,
        )
    )
    candidates = extract_memory_candidates(
        provider, [{"role": "user", "content": "你好"}], BudgetTracker()
    )
    assert len(candidates) == 2
    assert candidates[0]["text"] == "预算 5 美元。"
    assert candidates[0]["layers"] == ["core"]
    assert candidates[0]["reason"] == "长期规则"
    # source 固定为 inferred，不信任模型给的 source。
    assert all("source" not in c for c in candidates)


def test_extract_candidates_invalid_json_returns_empty():
    """模型返回的不是合法 JSON 时，返回空列表（不建提议、不报错）。"""
    provider = _mock_provider("我觉得预算很重要，但没有结构化数据。")
    candidates = extract_memory_candidates(
        provider, [{"role": "user", "content": "你好"}], BudgetTracker()
    )
    assert candidates == []


def test_extract_candidates_invalid_items_skipped():
    """JSON 里不合法的条目被丢弃，合法的照常保留。"""
    provider = _mock_provider(
        json.dumps(
            [
                {"text": "预算 5 美元。", "layers": ["core"], "reason": "长期规则"},
                {"text": "", "layers": ["core"], "reason": "空内容，不合法"},
                {"text": "坏层。", "layers": ["not_a_layer"], "reason": "层不合法"},
                {"text": "缺层。", "reason": "没有 layers 字段"},
            ],
            ensure_ascii=False,
        )
    )
    candidates = extract_memory_candidates(
        provider, [{"role": "user", "content": "你好"}], BudgetTracker()
    )
    assert len(candidates) == 1
    assert candidates[0]["text"] == "预算 5 美元。"


def test_extract_candidates_provider_raises_returns_empty():
    """提取调用失败（没 Key、超预算、网络报错）时返回空列表，不抛异常。"""
    provider = Mock()
    provider.chat_messages.side_effect = RuntimeError("DeepSeek API 调用失败")
    candidates = extract_memory_candidates(
        provider, [{"role": "user", "content": "你好"}], BudgetTracker()
    )
    assert candidates == []


def test_extract_candidates_empty_messages_returns_empty():
    """没有对话消息时不提取，返回空列表。"""
    provider = _mock_provider("[]")
    assert extract_memory_candidates(provider, [], BudgetTracker()) == []
    assert extract_memory_candidates(None, [{"role": "user", "content": "你好"}], BudgetTracker()) == []


# ---------- run_question 提取 ----------


def _run_question_with_extract(monkeypatch, tmp_path, chat_text, extraction_text):
    """用 Mock 的 DeepSeek 调用跑一次带提取的 run_question。

    第一次 post 返回聊天回答，第二次 post 返回提取 JSON。
    返回 (outcome, memory_store, proposal_store)。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    responses = [
        _fake_ok_response(text=chat_text),
        _fake_extraction_response(
            json.loads(extraction_text) if extraction_text.strip().startswith("[") else [],
        ),
    ]
    # 提取返回的文本直接用给定的 extraction_text（可能是非法 JSON）。
    responses[1] = _fake_ok_response(text=extraction_text)
    with patch("providers.deepseek.requests.post", side_effect=responses):
        outcome = run_question(
            "我的预算是多少",
            budget_tracker=tracker,
            messages=[],
            memory_store=memory_store,
            extract_proposals=True,
            proposal_store=proposal_store,
        )
    return outcome, memory_store, proposal_store


def test_run_question_with_extract_creates_proposals(monkeypatch, tmp_path):
    """聊天成功后提取建提议：有真实 ID、pending 状态；聊天回答本身不受影响。"""
    extraction = json.dumps(
        [{"text": "每月预算 5 美元。", "layers": ["core"], "reason": "长期预算规则"}],
        ensure_ascii=False,
    )
    outcome, _memory_store, proposal_store = _run_question_with_extract(
        monkeypatch, tmp_path, "你的预算是每月 5 美元。", extraction
    )
    assert outcome["mode"] == "real"
    assert outcome["result"]["text"] == "你的预算是每月 5 美元。"
    assert len(outcome["new_proposals"]) == 1
    proposal = outcome["new_proposals"][0]
    assert proposal["id"].startswith("prop-")
    assert proposal["status"] == "pending"
    assert proposal["text"] == "每月预算 5 美元。"
    # 提议在待确认列表里，记忆文件里还没有（未经确认不写）。
    assert len(proposal_store.list_pending()) == 1
    assert _store(tmp_path).list_all() == []


def test_run_question_extract_failure_does_not_affect_chat(monkeypatch, tmp_path):
    """提取失败时聊天回答照常返回，没有新提议（解耦）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch(
        "providers.deepseek.requests.post",
        side_effect=[
            _fake_ok_response(text="你的预算是每月 5 美元。"),
            RuntimeError("DeepSeek API 调用失败"),
        ],
    ):
        outcome = run_question(
            "我的预算是多少",
            budget_tracker=tracker,
            messages=[],
            memory_store=memory_store,
            extract_proposals=True,
            proposal_store=proposal_store,
        )
    assert outcome["mode"] == "real"
    assert outcome["result"]["text"] == "你的预算是每月 5 美元。"
    assert outcome["new_proposals"] == []
    assert proposal_store.list_pending() == []


def test_run_question_extract_invalid_json_creates_no_proposals(monkeypatch, tmp_path):
    """提取返回非法 JSON 时不建提议，聊天回答照常返回。"""
    outcome, _memory_store, proposal_store = _run_question_with_extract(
        monkeypatch, tmp_path, "你的预算是每月 5 美元。", "这不是 JSON"
    )
    assert outcome["mode"] == "real"
    assert outcome["new_proposals"] == []
    assert proposal_store.list_pending() == []


def test_run_question_without_extract_creates_no_proposals(monkeypatch, tmp_path):
    """默认不提取：不传 extract_proposals 时不跑提取、不建提议。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    with patch(
        "providers.deepseek.requests.post",
        return_value=_fake_ok_response(text="你的预算是每月 5 美元。"),
    ) as mock_post:
        outcome = run_question(
            "我的预算是多少",
            budget_tracker=tracker,
            messages=[],
            memory_store=memory_store,
            proposal_store=proposal_store,
        )
    assert outcome["mode"] == "real"
    # 只调用了一次模型（聊天），没有第二次提取调用。
    assert mock_post.call_count == 1
    assert outcome.get("new_proposals") == []
    assert proposal_store.list_pending() == []


# ---------- 记忆去重 ----------


def test_add_memory_duplicate_text_not_saved_twice(tmp_path):
    """内容完全相同的记忆只存一条，第二次返回已有记忆，不重复写文件。"""
    store = _store(tmp_path)
    first = store.add_memory("每月预算 5 美元。", ["core"], "user", "2026-11-07")
    second = store.add_memory("每月预算 5 美元。", ["core"], "user", "2026-11-07")
    assert second["text"] == first["text"]
    assert len(store.list_all()) == 1


def test_add_memory_different_text_still_saved(tmp_path):
    """内容不同的记忆照常存（去重只拦完全相同的内容）。"""
    store = _store(tmp_path)
    store.add_memory("每月预算 5 美元。", ["core"], "user", "2026-11-07")
    store.add_memory("每月预算 10 美元。", ["core"], "user", "2026-11-07")
    assert len(store.list_all()) == 2


def test_confirm_duplicate_proposal_does_not_duplicate_memory(tmp_path):
    """确认一条与已有记忆内容相同的提议时，不重复写文件，只留一条。"""
    memory_store = _store(tmp_path)
    memory_store.add_memory("每月预算 5 美元。", ["core"], "user", "2026-11-07")
    proposal_store = ProposalStore()
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")
    saved = proposal_store.confirm(proposal["id"], memory_store)
    assert saved["text"] == "每月预算 5 美元。"
    assert len(memory_store.list_all()) == 1
    assert proposal_store.list_pending() == []


# ---------- 网页接口 ----------


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


@pytest.fixture
def proposal_server(tmp_path):
    """启动一个带记忆库和提议暂存的测试服务，返回 (port, state, memory_store, proposal_store)。"""
    memory_store = MemoryStore(file_path=tmp_path / "memories.json")
    proposal_store = ProposalStore()
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    config = ServerConfig(host="127.0.0.1", port=0)
    server, state = create_server(
        config=config,
        budget_tracker=tracker,
        memory_store=memory_store,
        proposal_store=proposal_store,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], state, memory_store, proposal_store
    finally:
        server.shutdown()
        server.server_close()


def test_web_proposals_list_empty(proposal_server):
    """没有提议时，GET /api/proposals 返回空列表。"""
    port, _state, _memory_store, _proposal_store = proposal_server
    status, body = _request(port, "GET", "/api/proposals")
    assert status == 200
    assert body == {"proposals": []}


def test_web_proposals_list_shows_pending(proposal_server):
    """创建提议后，GET /api/proposals 能看到它（含真实 ID 与 pending 状态）。"""
    port, _state, _memory_store, proposal_store = proposal_server
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")
    status, body = _request(port, "GET", "/api/proposals")
    assert status == 200
    assert len(body["proposals"]) == 1
    assert body["proposals"][0]["id"] == proposal["id"]
    assert body["proposals"][0]["status"] == "pending"
    assert body["proposals"][0]["text"] == "每月预算 5 美元。"


def test_web_proposals_confirm_writes_memory(proposal_server):
    """POST 确认后，记忆写进文件，提议从列表消失。

    注意：测试服务启动时已种下前三条记忆，所以文件里原来就有 3 条；
    确认后应为 4 条，且包含新确认的这条。
    """
    port, _state, memory_store, proposal_store = proposal_server
    assert len(memory_store.list_all()) == 3  # 启动时种下的前三条
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")
    status, body = _request(port, "POST", f"/api/proposals/{proposal['id']}/confirm")
    assert status == 200
    assert body["ok"] is True
    assert body["memory"]["text"] == "每月预算 5 美元。"
    assert len(memory_store.list_all()) == 4
    assert "每月预算 5 美元。" in [m["text"] for m in memory_store.list_all()]
    status, body = _request(port, "GET", "/api/proposals")
    assert body == {"proposals": []}


def test_web_proposals_reject_does_not_write(proposal_server):
    """POST 拒绝后，提议消失，记忆文件里没有这条（仍只有启动时种下的 3 条）。"""
    port, _state, memory_store, proposal_store = proposal_server
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")
    status, body = _request(port, "POST", f"/api/proposals/{proposal['id']}/reject")
    assert status == 200
    assert body["ok"] is True
    assert "每月预算 5 美元。" not in [m["text"] for m in memory_store.list_all()]
    assert len(memory_store.list_all()) == 3
    status, body = _request(port, "GET", "/api/proposals")
    assert body == {"proposals": []}


def test_web_proposals_confirm_missing_returns_404(proposal_server):
    """确认不存在的提议时返回 404。"""
    port, _state, _memory_store, _proposal_store = proposal_server
    status, body = _request(port, "POST", "/api/proposals/prop-doesnotexist/confirm")
    assert status == 404
    assert "error" in body


def test_web_proposals_reject_missing_returns_404(proposal_server):
    """拒绝不存在的提议时返回 404。"""
    port, _state, _memory_store, _proposal_store = proposal_server
    status, body = _request(port, "POST", "/api/proposals/prop-doesnotexist/reject")
    assert status == 404


def test_web_proposals_confirm_save_failure_keeps_proposal(proposal_server, monkeypatch):
    """保存失败时返回 502，提议保留在待确认列表（可重试）。"""
    port, _state, memory_store, proposal_store = proposal_server
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")

    def _boom(self, _memories):
        raise RuntimeError("磁盘写不进去")

    monkeypatch.setattr(MemoryStore, "_write", _boom)
    status, body = _request(port, "POST", f"/api/proposals/{proposal['id']}/confirm")
    assert status == 502
    assert "error" in body
    # 提议还在，记忆文件里没有这条（仍只有启动时种下的 3 条）。
    status, body = _request(port, "GET", "/api/proposals")
    assert len(body["proposals"]) == 1
    assert "每月预算 5 美元。" not in [m["text"] for m in memory_store.list_all()]


def test_web_end_to_end_chat_extract_confirm(monkeypatch, proposal_server):
    """端到端：网页聊天 → 提取建提议 → 列表可见 → 确认 → 记忆进文件。

    模型调用全部用 Mock：第一次 post 是聊天回答，第二次 post 是提取 JSON。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    port, state, memory_store, proposal_store = proposal_server
    extraction = json.dumps(
        [{"text": "每月预算 5 美元。", "layers": ["core"], "reason": "长期预算规则"}],
        ensure_ascii=False,
    )
    with patch(
        "providers.deepseek.requests.post",
        side_effect=[
            _fake_ok_response(text="你的预算是每月 5 美元。"),
            _fake_ok_response(text=extraction),
        ],
    ):
        status, body = _request(port, "POST", "/api/chat", {"message": "我的预算是多少"})
    assert status == 200
    assert body["mode"] == "real"
    assert body["reply"] == "你的预算是每月 5 美元。"
    # 提取建了提议，列表里能看到。
    assert len(body["new_proposals"]) == 1
    proposal_id = body["new_proposals"][0]["id"]
    assert proposal_id.startswith("prop-")
    status, body = _request(port, "GET", "/api/proposals")
    assert len(body["proposals"]) == 1
    assert body["proposals"][0]["id"] == proposal_id
    # 确认后记忆进文件，列表清空。
    # 注意：测试服务启动时已种下前三条记忆，确认后文件里共 4 条。
    status, body = _request(port, "POST", f"/api/proposals/{proposal_id}/confirm")
    assert status == 200
    assert body["ok"] is True
    assert len(memory_store.list_all()) == 4
    assert "每月预算 5 美元。" in [m["text"] for m in memory_store.list_all()]
    status, body = _request(port, "GET", "/api/proposals")
    assert body == {"proposals": []}
