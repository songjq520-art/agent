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


# ---------- 提取提示词 ----------


def test_extraction_instruction_requires_user_explicit_content():
    """提取提示词必须明确：只提用户明确表达的事实、偏好和决定。

    明确排除：AI 自己提出的建议、假设、举例，不能当成用户事实；
    第三方信息，不能当成用户个人记忆。这是防止模型把 AI 的话
    或通用知识记成用户记忆的关键防线，提示词被改弱时这条测试会失败。
    """
    from main import EXTRACTION_INSTRUCTION

    assert "用户自己明确表达" in EXTRACTION_INSTRUCTION
    assert "事实" in EXTRACTION_INSTRUCTION
    assert "偏好" in EXTRACTION_INSTRUCTION
    assert "决定" in EXTRACTION_INSTRUCTION
    # 明确排除 AI 的建议、假设、举例。
    assert "建议" in EXTRACTION_INSTRUCTION
    assert "假设" in EXTRACTION_INSTRUCTION
    assert "举例" in EXTRACTION_INSTRUCTION
    assert "不能当成用户事实" in EXTRACTION_INSTRUCTION
    # 明确排除第三方信息。
    assert "第三方" in EXTRACTION_INSTRUCTION
    assert "不能把第三方信息当成用户个人记忆" in EXTRACTION_INSTRUCTION


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


def test_run_question_duplicate_candidates_appear_once_in_new_proposals(monkeypatch, tmp_path):
    """同一轮提取里内容重复的候选，在 new_proposals 里只出现一次。

    回归测试：之前 _run_extraction_and_propose() 会把 ProposalStore.add()
    返回的已有提议也追加进返回列表，导致同一条提议在 new_proposals 里
    出现两次（同一个 ID）。待确认列表本身只有一条（add() 的去重是好的），
    但返回列表重复会让前端误以为新建了两条。
    """
    extraction = json.dumps(
        [
            {"text": "每月预算 5 美元。", "layers": ["core"], "reason": "长期预算规则"},
            {"text": "每月预算 5 美元。", "layers": ["core"], "reason": "长期预算规则"},
        ],
        ensure_ascii=False,
    )
    outcome, _memory_store, proposal_store = _run_question_with_extract(
        monkeypatch, tmp_path, "你的预算是每月 5 美元。", extraction
    )
    assert outcome["mode"] == "real"
    assert len(outcome["new_proposals"]) == 1
    assert outcome["new_proposals"][0]["id"].startswith("prop-")
    assert len(proposal_store.list_pending()) == 1
    # 返回列表里的 ID 不重复。
    ids = [p["id"] for p in outcome["new_proposals"]]
    assert len(ids) == len(set(ids))


def test_run_question_candidate_matching_existing_proposal_not_in_new_proposals(monkeypatch, tmp_path):
    """提取出的候选与已在待确认列表里的提议内容相同时，不重复放进 new_proposals。

    new_proposals 的含义是“本轮新创建的提议”，已在列表里的不算新提议。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    # 先放一条已在待确认列表里的提议。
    existing = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
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
        outcome = run_question(
            "我的预算是多少",
            budget_tracker=tracker,
            messages=[],
            memory_store=memory_store,
            extract_proposals=True,
            proposal_store=proposal_store,
        )
    assert outcome["mode"] == "real"
    # 本轮没有新创建提议（候选与已有的重复），返回列表为空。
    assert outcome["new_proposals"] == []
    # 待确认列表里仍只有原来那一条。
    assert len(proposal_store.list_pending()) == 1
    assert proposal_store.list_pending()[0]["id"] == existing["id"]


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


# ---------- 正式记忆与提议去重（BOOS 本地验收发现的问题） ----------
# 问题：已经保存的正式记忆，再次被模型提取出来时，又出现在待确认列表。
# 原因：创建提议时只检查了 Pending 提议，没检查已保存的正式记忆。
# 修复：ProposalStore.add() 增加 memory_store 参数，创建前同时检查
# 待确认提议和正式记忆（相似度去重，不只是完全相同）。


def test_saved_memory_reextracted_creates_no_proposal(tmp_path):
    """已保存的记忆再次被提取时，不生成重复提议（核心修复）。

    复现 BOOS 的本地测试：先保存一条长期目标，后续聊天再次提取出
    同一条（措辞略有不同），待确认列表里不应再出现它。
    """
    from main import _run_extraction_and_propose

    memory_store = _store(tmp_path)
    memory_store.add_memory(
        "长期目标：做出真正属于自己的多模型 Agent。", ["core"], "user", "2026-11-08"
    )
    proposal_store = ProposalStore()
    extraction = json.dumps(
        [
            {
                "text": "用户长期目标是做出真正属于自己的多模型 Agent。",
                "layers": ["core"],
                "reason": "长期目标",
            }
        ],
        ensure_ascii=False,
    )
    provider = Mock()
    provider.chat_messages.return_value = {"text": extraction, "model": "deepseek-flash"}
    created = _run_extraction_and_propose(
        provider,
        [{"role": "user", "content": "我的目标是什么"}],
        None,
        proposal_store,
        memory_store=memory_store,
    )
    assert created == []
    assert proposal_store.list_pending() == []
    # 正式记忆只有一条，没被修改。
    assert len(memory_store.list_all()) == 1


def test_saved_memory_reextracted_via_run_question_creates_no_proposal(monkeypatch, tmp_path):
    """端到端：已保存的记忆再次被提取，run_question 的 new_proposals 为空。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    memory_store = _store(tmp_path)
    memory_store.add_memory(
        "每月预算 5 美元，4 美元提醒，5 美元停。", ["core"], "user", "2026-11-08"
    )
    proposal_store = ProposalStore()
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    extraction = json.dumps(
        [
            {
                "text": "用户每月预算 5 美元，4 美元提醒，5 美元停。",
                "layers": ["core"],
                "reason": "长期预算规则",
            }
        ],
        ensure_ascii=False,
    )
    with patch(
        "providers.deepseek.requests.post",
        side_effect=[
            _fake_ok_response(text="你的预算是每月 5 美元。"),
            _fake_ok_response(text=extraction),
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
    assert outcome["new_proposals"] == []
    assert proposal_store.list_pending() == []
    assert len(memory_store.list_all()) == 1


def test_pending_proposal_similar_text_not_duplicated():
    """Pending 中已有措辞略有不同的提议时，不重复生成。"""
    store = ProposalStore()
    first = store.add("每月预算 5 美元，4 美元提醒，5 美元停。", ["core"], "inferred", "长期规则")
    # 措辞略有不同（加“用户”前缀），不应再创建一条。
    second = store.add("用户每月预算 5 美元，4 美元提醒，5 美元停。", ["core"], "inferred", "长期规则")
    assert second["id"] == first["id"]
    assert len(store.list_pending()) == 1


def test_substantively_different_memory_still_creates_proposal(tmp_path):
    """内容相似但有实质区别的记忆，仍允许生成新提议（不过度去重）。"""
    memory_store = _store(tmp_path)
    memory_store.add_memory("每月预算 5 美元。", ["core"], "user", "2026-11-08")
    proposal_store = ProposalStore()
    # 关键数字不同（5 美元 → 10 美元），是实质区别，应允许生成。
    created = proposal_store.add(
        "每月预算 10 美元。", ["core"], "inferred", "预算调整", memory_store=memory_store
    )
    assert created is not None
    assert created["id"].startswith("prop-")
    assert len(proposal_store.list_pending()) == 1
    # 正式记忆仍只有原来那一条，没被修改。
    assert len(memory_store.list_all()) == 1


def test_dedup_does_not_affect_confirm_and_reject(tmp_path):
    """去重不影响正常的确认、拒绝和保存流程。"""
    memory_store = _store(tmp_path)
    proposal_store = ProposalStore()
    # 一条与正式记忆重复的候选不生成提议，一条全新的照常生成。
    memory_store.add_memory("每月预算 5 美元。", ["core"], "user", "2026-11-08")
    dup = proposal_store.add(
        "每月预算 5 美元。", ["core"], "inferred", "", memory_store=memory_store
    )
    assert dup is None
    new = proposal_store.add(
        "我喜欢喝咖啡。", ["experience"], "inferred", "个人偏好", memory_store=memory_store
    )
    assert new is not None
    # 确认新提议：照常写入。
    saved = proposal_store.confirm(new["id"], memory_store)
    assert saved["text"] == "我喜欢喝咖啡。"
    assert len(memory_store.list_all()) == 2
    # 再建一条并拒绝：照常丢弃，不写文件。
    another = proposal_store.add(
        "我喜欢喝茶。", ["experience"], "inferred", "", memory_store=memory_store
    )
    assert proposal_store.reject(another["id"]) is True
    assert len(memory_store.list_all()) == 2
    assert proposal_store.list_pending() == []


# ---------- 相似度函数单元测试 ----------


def test_is_duplicate_memory_text_cases():
    """相似度函数的正反案例：完全相同/轻微措辞差异算重复，实质区别不算。"""
    from memory.memories import is_duplicate_memory_text

    # 算重复的。
    assert is_duplicate_memory_text("每月预算 5 美元。", "每月预算 5 美元。") is True
    assert is_duplicate_memory_text("每月预算5美元", "每月预算 5 美元。") is True
    assert (
        is_duplicate_memory_text(
            "用户长期目标是做出真正属于自己的多模型 Agent。",
            "长期目标：做出真正属于自己的多模型 Agent。",
        )
        is True
    )
    assert (
        is_duplicate_memory_text(
            "每月预算 5 美元，4 美元提醒，5 美元停。",
            "用户每月预算 5 美元，4 美元提醒，5 美元停。",
        )
        is True
    )
    # 不算重复的（实质区别）。
    assert is_duplicate_memory_text("每月预算 5 美元", "每月预算 10 美元") is False
    assert is_duplicate_memory_text("我喜欢喝咖啡", "我喜欢喝茶") is False
    assert (
        is_duplicate_memory_text(
            "DeepSeek 已接通，是第一任主控。", "DeepSeek 已接通，是第二任主控。"
        )
        is False
    )
    assert (
        is_duplicate_memory_text("我喜欢喝咖啡", "我喜欢喝咖啡，也喜欢喝茶") is False
    )
    assert (
        is_duplicate_memory_text(
            "2026 年的目标是做出 Agent。", "2027 年的目标是做出 Agent。"
        )
        is False
    )


def test_find_duplicate_memory_returns_first_match():
    """find_duplicate_memory 在列表里找到重复的那一条，没找到返回 None。"""
    from memory.memories import find_duplicate_memory

    memories = [
        {"text": "每月预算 5 美元。", "layers": ["core"], "source": "user"},
        {"text": "我喜欢喝咖啡。", "layers": ["experience"], "source": "user"},
    ]
    found = find_duplicate_memory("用户每月预算 5 美元。", memories)
    assert found is not None
    assert found["text"] == "每月预算 5 美元。"
    assert find_duplicate_memory("我喜欢喝茶。", memories) is None
    assert find_duplicate_memory("", memories) is None


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
    """保存失败时返回 500（不是 502：这是本地磁盘写入失败，不是网关错误），
    提议保留在待确认列表（可重试），并明确提示是保存失败。"""
    port, _state, memory_store, proposal_store = proposal_server
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")

    def _boom(self, _memories):
        raise RuntimeError("磁盘写不进去")

    monkeypatch.setattr(MemoryStore, "_write", _boom)
    status, body = _request(port, "POST", f"/api/proposals/{proposal['id']}/confirm")
    assert status == 500
    assert "error" in body
    assert "保存失败" in body["error"]
    # 提议还在，记忆文件里没有这条（仍只有启动时种下的 3 条）。
    status, body = _request(port, "GET", "/api/proposals")
    assert len(body["proposals"]) == 1
    assert "每月预算 5 美元。" not in [m["text"] for m in memory_store.list_all()]


def test_web_proposals_confirm_real_io_error_returns_500_and_keeps_proposal(proposal_server, monkeypatch):
    """真实磁盘 I/O 错误（OSError，如权限不足、磁盘满）时也返回 500，
    提议保留在待确认列表（可重试），并明确提示是保存失败。

    回归测试：之前 _do_proposal_action 只 catch RuntimeError，
    OSError 会落到通用 500 处理（“服务内部出错”），不是明确的“保存失败”。
    """
    port, _state, memory_store, proposal_store = proposal_server
    proposal = proposal_store.add("每月预算 5 美元。", ["core"], "inferred", "长期规则")

    def _boom(self, _memories):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(MemoryStore, "_write", _boom)
    status, body = _request(port, "POST", f"/api/proposals/{proposal['id']}/confirm")
    assert status == 500
    assert "error" in body
    assert "保存失败" in body["error"]
    # 提议还在，记忆文件里没有这条。
    status, body = _request(port, "GET", "/api/proposals")
    assert len(body["proposals"]) == 1
    assert "每月预算 5 美元。" not in [m["text"] for m in memory_store.list_all()]


def test_web_end_to_end_saved_memory_not_reextracted_as_proposal(monkeypatch, proposal_server):
    """端到端：保存过的记忆再次被提取，不再出现在待确认列表（BOOS 本地验收场景）。

    流程：聊天提取出提议 → 确认保存进文件 → 再聊天，模型又提取出同一条
    → 待确认列表里不应再出现它，正式记忆仍只有一条。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    port, state, memory_store, proposal_store = proposal_server
    extraction = json.dumps(
        [{"text": "每月预算 5 美元。", "layers": ["core"], "reason": "长期预算规则"}],
        ensure_ascii=False,
    )
    # 第一轮：聊天提取出提议。
    with patch(
        "providers.deepseek.requests.post",
        side_effect=[
            _fake_ok_response(text="你的预算是每月 5 美元。"),
            _fake_ok_response(text=extraction),
        ],
    ):
        status, body = _request(port, "POST", "/api/chat", {"message": "我的预算是多少"})
    assert status == 200
    assert len(body["new_proposals"]) == 1
    proposal_id = body["new_proposals"][0]["id"]
    # 确认保存。
    status, body = _request(port, "POST", f"/api/proposals/{proposal_id}/confirm")
    assert status == 200
    # 第二轮：再聊天，模型又提取出同一条（措辞略有不同）。
    extraction2 = json.dumps(
        [{"text": "用户每月预算 5 美元。", "layers": ["core"], "reason": "长期预算规则"}],
        ensure_ascii=False,
    )
    with patch(
        "providers.deepseek.requests.post",
        side_effect=[
            _fake_ok_response(text="你的预算还是每月 5 美元。"),
            _fake_ok_response(text=extraction2),
        ],
    ):
        status, body = _request(port, "POST", "/api/chat", {"message": "我的预算变了吗"})
    assert status == 200
    # 不应再生成提议。
    assert body["new_proposals"] == []
    status, body = _request(port, "GET", "/api/proposals")
    assert body == {"proposals": []}
    # 正式记忆里只有这一条（加上启动时种下的 3 条，共 4 条），没有重复。
    texts = [m["text"] for m in memory_store.list_all()]
    assert texts.count("每月预算 5 美元。") == 1


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
