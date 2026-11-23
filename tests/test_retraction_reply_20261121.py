# tests/test_retraction_reply_20261121.py（2026-11-21 网页测第二轮 #1 + #3 + #4）
# #1：撤回后回消息崩——run_question 收尾读 save_turn 回执里的 dropped
# （句子列表）时 list > 0 抛 TypeError → 500。修法：dropped_count > 0
# 或 len(dropped) > 0（回执里 dropped 恒为 []、真正记数的是
# dropped_count；只按长度判断会让“刚说的那句已经撤回”提示永远打不出来）。
# 全部离线模拟：模型用假 requests.post，记忆库用只返回指定回执的假 store。

from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker
from main import _memory_block, _should_offer_mind_read, run_question


class _ReceiptStore:
    """只返回指定 save_turn 回执的假记忆库（#1 测试用）。"""

    def __init__(self, receipt):
        self.receipt = receipt
        self.saved = []

    def search(self, prompt, top_k=5, reinforce=False, turn_id=None):
        return []

    def note_context(self, memories):
        return None

    def save_turn(self, messages, turn_id=None):
        self.saved.append(messages)
        return dict(self.receipt)


def _fake_ok_response(text="你好，我是 DeepSeek。"):
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    fake_response.raise_for_status.return_value = None
    return fake_response


def _run_with_receipt(monkeypatch, tmp_path, receipt, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _ReceiptStore(receipt)
    with patch("providers.deepseek.requests.post", return_value=_fake_ok_response()):
        outcome = run_question(
            "你好", budget_tracker=tracker, messages=[], mem0_store=store
        )
    return outcome, store, capsys.readouterr()


def test_retraction_reply_with_dropped_list_does_not_crash(monkeypatch, tmp_path, capsys):
    """#1：撤回那条路的回执里 dropped 是句子列表、dropped_count 是 1——
    以前 list > 0 直接抛 TypeError → 500，现在正常回话，还打出提示。"""
    outcome, _store, captured = _run_with_receipt(
        monkeypatch, tmp_path,
        {"retracted": True, "dropped": ["上一句"], "dropped_count": 1, "stored": []},
        capsys,
    )
    assert outcome["mode"] == "real"
    assert outcome["result"]["text"] == "你好，我是 DeepSeek。"
    assert "已经撤回" in captured.out


def test_retraction_reply_with_nothing_dropped_does_not_crash(monkeypatch, tmp_path, capsys):
    """#1：撤回了但什么都没丢（dropped 空列表、dropped_count 0）——不崩，也不打提示。"""
    outcome, _store, captured = _run_with_receipt(
        monkeypatch, tmp_path,
        {"retracted": True, "dropped": [], "dropped_count": 0, "stored": []},
        capsys,
    )
    assert outcome["mode"] == "real"
    assert "已经撤回" not in captured.out


def test_retraction_reply_with_dropped_list_only(monkeypatch, tmp_path, capsys):
    """#1：只有 dropped 列表非空（dropped_count 是 0）——不崩，提示也打得出来。"""
    outcome, _store, captured = _run_with_receipt(
        monkeypatch, tmp_path,
        {"retracted": True, "dropped": ["上一句"], "dropped_count": 0, "stored": []},
        capsys,
    )
    assert outcome["mode"] == "real"
    assert "已经撤回" in captured.out


# ---------- #4：揣摩门控 ----------

@pytest.mark.parametrize("prompt,expected", [
    ("我喜欢喝茶", False),
    ("周末去爬山", False),
    ("好久没吃火锅了", True),
    ("手机快没电了", True),
    ("帮我查一下天气", False),
    ("可以", False),
    ("忘记我喜欢喝茶", False),
    ("好累啊", True),
])
def test_should_offer_mind_read_gate(prompt, expected):
    """#4：揣摩门控——简单表态、交代做事、随口一提的喜好不递提示；
    只有带想要信号的才递。"""
    assert _should_offer_mind_read(prompt) is expected


# ---------- #3：回答口吻锚点 ----------

def test_memory_block_bans_audit_phrasing():
    """#3：记忆提示块里常驻“别念记录”的要求（回答口吻的程序锚点；
    模型最终怎么答以 BOOS 再测为准）。"""
    block = _memory_block([{"text": "我每天定投 5 块买万家纳指", "confirmed": True}])
    assert "别念记录" in block
    assert "别编造保存、工具之类的说法" in block
