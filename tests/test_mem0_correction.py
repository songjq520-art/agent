# tests/test_mem0_correction.py（聊天纠错接线测试）
# 作用：验证 add_turn 里的“你记错了”接线（2026-11-09）——聊天里说
# 纠错时，add_turn 自己调 correct_memory 改旧记忆，而不是把新说法
# 当成一条新记忆存下来（2026-11-09 生命周期复测证实的已知缺口：
# “不对，我喜欢看纪录片。”之后，库里科幻和纪录片两条都在）。
# 测试全部离线（FakeMemory，不联网、不产生费用）；真实提取质量
# （DeepSeek 听不听得懂“不对，”）靠真实 API 的生命周期复测覆盖。
# 覆盖的情形（任务清单）：
#   1. 聊天纠错替换旧记忆（Go/Python + 生命周期原案科幻/纪录片）；
#   2. 多条相似记忆时问 BOOS、不猜不改，下句挑中了再改；
#   3. 普通聊天不误触发；
#   4. 纠正不存在的记忆时当新记忆存下来；
#   5. 新说法已经存了时报 unchanged、不重复存；
#   6. 纠错失败时原记忆不动、如实报失败；
#   7. 两条聊天入口都把结果告诉 BOOS（CLI 打印、网页 ask 返回字段）。

from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker
from conftest import FakeMemory, make_fake_store
from main import _print_correction_result, run_question
from memory.mem0_store import Mem0Store, _detect_correction
from web.server import ChatState


# ---------- 1. 识别：只认明确的纠错说法 ----------


def test_detect_correction_prefixes():
    """五个明确说法都认，新说法抠得对；普通话不认。"""
    assert _detect_correction("不对，我喜欢看纪录片。") == {
        "phrase": "不对",
        "fact": "我喜欢看纪录片。",
    }
    assert _detect_correction("你记错了，我喜欢用 Go 写代码") == {
        "phrase": "你记错了",
        "fact": "我喜欢用 Go 写代码",
    }
    assert _detect_correction("纠正一下，我住在北京。") == {
        "phrase": "纠正一下",
        "fact": "我住在北京。",
    }
    assert _detect_correction("其实是纪录片。") == {
        "phrase": "其实是",
        "fact": "纪录片。",
    }
    assert _detect_correction("我不是喜欢科幻电影，我喜欢纪录片。") == {
        "phrase": "我不是",
        "fact": "我喜欢纪录片。",
    }
    assert _detect_correction("我不是喜欢科幻电影，而是喜欢纪录片。") == {
        "phrase": "我不是",
        "fact": "喜欢纪录片。",
    }
    # 一句里先肯定后纠正（“不对”在句号后面）也认。
    assert _detect_correction("我喜欢看科幻电影。不对，我喜欢看纪录片。") == {
        "phrase": "不对",
        "fact": "我喜欢看纪录片。",
    }


def test_detect_correction_rejects_lookalikes():
    """容易误认的说法不触发：犹豫、词的一部分、只有否定没有新说法。"""
    # “我不是很确定”只是犹豫，没有替换说法。
    assert _detect_correction("我不是很确定") is None
    assert _detect_correction("我不是故意的") is None
    # “不对劲/不对头”是词的一部分，不是“不对，”这个说法。
    assert _detect_correction("我觉得不对劲") is None
    assert _detect_correction("不对劲，我先走了") is None
    # 只有否定、没有新说法（“我不是程序员。”没说到底是什么）。
    assert _detect_correction("我不是程序员。") is None
    # 普通聊天、问句、单纯的偏好陈述都不触发。
    assert _detect_correction("今天天气真好。") is None
    assert _detect_correction("我喜欢喝咖啡。") is None
    assert _detect_correction("你记得我叫什么吗？") is None
    # 新说法太短（只剩一个字）不算纠错。
    assert _detect_correction("不对，是。") is None


# ---------- 2. 聊天纠错替换旧记忆 ----------


def test_chat_correction_replaces_old_memory(tmp_path):
    """聊天说“你记错了，我喜欢用 Go 写代码”→ 旧 Python 记忆被替换，
    新 Go 记忆存下来，只剩 Go 能检索到（同一 id 原地替换）。"""
    store = make_fake_store(tmp_path)
    store.add_turn([
        {"role": "user", "content": "我喜欢用 Python 写代码。"},
        {"role": "assistant", "content": "明白，你喜欢用 Python 写代码，我记下了。"},
    ])
    py_id = store.get_all()[0]["id"]
    result = store.add_turn([
        {"role": "user", "content": "你记错了，我喜欢用 Go 写代码。"},
        {"role": "assistant", "content": "明白，你现在喜欢用 Go 写代码，我更新一下。"},
    ])
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["memory_id"] == py_id
    assert result["correction"]["old_text"] == "我喜欢用 Python 写代码"
    assert result["correction"]["new_text"] == "我喜欢用 Go 写代码。"
    assert [item["text"] for item in store.get_all()] == ["我喜欢用 Go 写代码。"]
    assert store.get_all()[0]["id"] == py_id
    found = store.search("我喜欢用什么语言写代码？", top_k=5)
    assert any("Go" in item["text"] for item in found)
    assert not any("Python" in item["text"] for item in found)
    # 提取没再跑一遍（新说法只存了一份，没变重复）。
    assert len(store._memory.add_calls) == 1


def test_chat_correction_lifecycle_case(tmp_path):
    """生命周期复测的原案：先说喜欢科幻电影，再说“不对，我喜欢看纪录片。”，
    旧的科幻记忆必须没了、只剩纪录片——这正是之前没接线的缺口。"""
    store = make_fake_store(tmp_path)
    store.add_turn([
        {"role": "user", "content": "我喜欢看科幻电影。"},
        {"role": "assistant", "content": "明白，你喜欢看科幻电影，我记下了。"},
    ])
    result = store.add_turn([
        {"role": "user", "content": "不对，我喜欢看纪录片。"},
        {"role": "assistant", "content": "明白，你现在喜欢看纪录片，我更新一下。"},
    ])
    assert result["correction"]["status"] == "updated"
    texts = [item["text"] for item in store.get_all()]
    assert texts == ["我喜欢看纪录片。"]
    assert not any("科幻" in text for text in texts)


# ---------- 3. 多条相似记忆：问 BOOS，下句挑中再改 ----------


def test_ambiguous_correction_asks_user_then_resolves(tmp_path):
    """两条相似记忆（科幻电影/科幻小说）+“不对，我喜欢看纪录片。”→
    ambiguous：列出候选，什么都不改、不存重复；下句回“第 1 条”→
    按挑中的那条改，另一条原封不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    result = store.add_turn([
        {"role": "user", "content": "不对，我喜欢看纪录片。"},
        {"role": "assistant", "content": "明白，你现在喜欢看纪录片，我更新一下。"},
    ])
    assert result["correction"]["status"] == "ambiguous"
    assert len(result["correction"]["candidates"]) == 2
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看科幻电影。",
        "我喜欢看科幻小说。",
    ]
    assert store._memory.update_calls == []
    # 下句挑“第 1 条”：按挑中的改，另一条不动。
    resolved = store.add_turn([
        {"role": "user", "content": "第 1 条"},
        {"role": "assistant", "content": "好的，改第一条。"},
    ])
    assert resolved["correction"]["status"] == "updated"
    assert resolved["correction"]["old_text"] == "我喜欢看科幻电影。"
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我喜欢看科幻小说。",
    ]


def test_ambiguous_correction_resolves_by_quoting_text(tmp_path):
    """ambiguous 之后直接引用候选原文（“科幻小说那条”）也能挑中。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    store.add_turn([
        {"role": "user", "content": "不对，我喜欢看纪录片。"},
        {"role": "assistant", "content": "明白，我更新一下。"},
    ])
    resolved = store.add_turn([
        {"role": "user", "content": "科幻小说那条"},
        {"role": "assistant", "content": "好的，改科幻小说那条。"},
    ])
    assert resolved["correction"]["status"] == "updated"
    assert resolved["correction"]["old_text"] == "我喜欢看科幻小说。"
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看科幻电影。",
        "我喜欢看纪录片。",
    ]


def test_ambiguous_correction_abandoned_when_user_says_something_else(tmp_path):
    """ambiguous 之后用户说了别的（不是挑候选）：当作放弃这次纠错，
    待确认清掉，本轮按正常聊天处理，不再追着问。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    store.add_turn([
        {"role": "user", "content": "不对，我喜欢看纪录片。"},
        {"role": "assistant", "content": "明白，我更新一下。"},
    ])
    assert store._pending_correction is not None
    result = store.add_turn([
        {"role": "user", "content": "算了，先不改了。"},
        {"role": "assistant", "content": "好的，先不改。"},
    ])
    # 没再报纠错结果，待确认已清掉，两条旧记忆原封不动。
    assert "correction" not in result
    assert store._pending_correction is None
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看科幻电影。",
        "我喜欢看科幻小说。",
    ]


# ---------- 4. 普通聊天不误触发 ----------


def test_ordinary_message_no_false_trigger(tmp_path):
    """普通聊天走正常保存：没有 "correction" 键，记忆正常存下来。"""
    store = make_fake_store(tmp_path)
    result = store.add_turn([
        {"role": "user", "content": "今天天气真好。我喜欢喝咖啡。"},
        {"role": "assistant", "content": "是啊，喝杯咖啡挺好的。"},
    ])
    assert "correction" not in result
    assert [item["text"] for item in store.get_all()] == ["我喜欢喝咖啡"]


# ---------- 5. 纠正不存在的记忆：当新记忆存下来 ----------


def test_correction_of_missing_memory_saves_new(tmp_path):
    """库里没有冲突的旧记忆：“不对，我喜欢喝茶。”→ not_found，
    走正常提取保存（新说法当新记忆存下来），不崩、不谎称改过。"""
    store = make_fake_store(tmp_path)
    before_calls = len(store._memory.add_calls)
    result = store.add_turn([
        {"role": "user", "content": "不对，我喜欢喝茶。"},
        {"role": "assistant", "content": "明白，你喜欢喝茶，我记下了。"},
    ])
    assert result["correction"]["status"] == "not_found"
    # 正常提取确实跑了（add 被调了），新说法存下来了。
    assert len(store._memory.add_calls) == before_calls + 1
    assert [item["text"] for item in store.get_all()] == ["我喜欢喝茶"]


# ---------- 6. 新说法已经存了：unchanged，不重复 ----------


def test_correction_already_stored_is_unchanged(tmp_path):
    """新说法库里已经有了：“你记错了，我喜欢用 Go 写代码。”→
    unchanged，只有一条 Go 记忆，不重复存。"""
    store = make_fake_store(tmp_path)
    store.add_turn([
        {"role": "user", "content": "我喜欢用 Go 写代码。"},
        {"role": "assistant", "content": "明白，你喜欢用 Go 写代码，我记下了。"},
    ])
    before_calls = len(store._memory.add_calls)
    result = store.add_turn([
        {"role": "user", "content": "你记错了，我喜欢用 Go 写代码。"},
        {"role": "assistant", "content": "明白，你喜欢用 Go 写代码，我记下了。"},
    ])
    assert result["correction"]["status"] == "unchanged"
    assert [item["text"] for item in store.get_all()] == ["我喜欢用 Go 写代码"]
    # 没再调提取保存（不重复存）。
    assert len(store._memory.add_calls) == before_calls


# ---------- 7. 纠错失败：原记忆不动、如实报失败 ----------


def test_correction_failure_preserves_original(tmp_path):
    """底层 update 抛错时：聊天纠错返回 failed，库里还是旧文本，
    没有新记忆被存下来（不走提取兜底、不谎称改好了）。"""

    class _FailingUpdateMemory(FakeMemory):
        def update(self, memory_id, text=None, metadata=None):
            raise RuntimeError("向量库炸了")

    failing = _FailingUpdateMemory()
    failing.add([{"role": "user", "content": "我喜欢喝咖啡。"}], infer=False)
    store = Mem0Store(
        memory=failing,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    result = store.add_turn([
        {"role": "user", "content": "不对，我喜欢喝茶。"},
        {"role": "assistant", "content": "明白，我更新一下。"},
    ])
    assert result["correction"]["status"] == "failed"
    assert result["correction"]["old_text"] == "我喜欢喝咖啡。"
    assert [item["text"] for item in store.get_all()] == ["我喜欢喝咖啡。"]


# ---------- 8. 两条聊天入口都把结果告诉 BOOS ----------


def _fake_ok_response(text="你好，我是 DeepSeek。", prompt_tokens=100, completion_tokens=50):
    """模拟的成功 API 返回（带 usage），只给 CLI 入口测试用。"""
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


def test_cli_run_question_prints_correction(monkeypatch, tmp_path, capsys):
    """CLI 入口：run_question 里纠错改好后，打印一行人话、返回里带
    correction，库里旧记忆已被替换。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    store.add_turn([
        {"role": "user", "content": "我喜欢用 Python 写代码。"},
        {"role": "assistant", "content": "明白，我记下了。"},
    ])
    with patch(
        "providers.deepseek.requests.post",
        return_value=_fake_ok_response(text="好的，我帮你改。"),
    ):
        outcome = run_question(
            "你记错了，我喜欢用 Go 写代码。",
            budget_tracker=tracker,
            mem0_store=store,
        )
    assert outcome["correction"]["status"] == "updated"
    printed = capsys.readouterr().out
    assert "已纠正" in printed
    assert "Python" in printed and "Go" in printed
    assert [item["text"] for item in store.get_all()] == ["我喜欢用 Go 写代码。"]


def test_cli_print_correction_lines(capsys):
    """_print_correction_result：四种要说话的状态各打印一行人话；
    没纠错时什么都不打印、返回 None。"""
    assert _print_correction_result({}) is None
    assert _print_correction_result({"results": []}) is None
    assert capsys.readouterr().out == ""
    _print_correction_result({
        "correction": {
            "status": "updated",
            "old_text": "我喜欢用 Python 写代码",
            "new_text": "我喜欢用 Go 写代码。",
        }
    })
    _print_correction_result({
        "correction": {
            "status": "ambiguous",
            "candidates": [
                {"id": "a", "text": "我喜欢喝黑咖啡。"},
                {"id": "b", "text": "我喜欢喝拿铁咖啡。"},
            ],
        }
    })
    _print_correction_result({
        "correction": {"status": "failed", "reason": "更新失败，原记忆未动。"}
    })
    printed = capsys.readouterr().out
    assert "已纠正" in printed
    assert "我喜欢喝黑咖啡。" in printed and "我喜欢喝拿铁咖啡。" in printed
    assert "第 1 条" in printed
    assert "纠正失败，原记忆没动" in printed


def test_web_ask_returns_correction_field(monkeypatch, tmp_path):
    """网页入口：ChatState.ask() 的返回里带 "correction" 键，
    原来的 reply/model/mode/cost_usd/usage 五个键不动。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    store.add_turn([
        {"role": "user", "content": "我喜欢用 Python 写代码。"},
        {"role": "assistant", "content": "明白，我记下了。"},
    ])
    state = ChatState(budget_tracker=tracker, mem0_store=store)

    def _fake_run_question(prompt, budget_tracker=None, messages=None, **kwargs):
        return {
            "mode": "real",
            "result": {
                "text": "好的，我帮你改。",
                "model": "deepseek-flash",
                "cost_usd": 0.0001,
                "usage": {"total_tokens": 10},
            },
        }

    monkeypatch.setattr("web.server.run_question", _fake_run_question)
    result = state.ask("你记错了，我喜欢用 Go 写代码。")
    assert result["mode"] == "real"
    assert result["reply"] == "好的，我帮你改。"
    assert result["model"] == "deepseek-flash"
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["old_text"] == "我喜欢用 Python 写代码"
    assert result["correction"]["new_text"] == "我喜欢用 Go 写代码。"
    assert [item["text"] for item in store.get_all()] == ["我喜欢用 Go 写代码。"]


def test_web_ask_without_correction_has_no_correction_field(monkeypatch, tmp_path):
    """网页入口：普通聊天时 ask() 的返回里没有 "correction" 键
    （原来的五个键原样返回，不多不少）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    state = ChatState(budget_tracker=tracker, mem0_store=store)

    def _fake_run_question(prompt, budget_tracker=None, messages=None, **kwargs):
        return {
            "mode": "real",
            "result": {
                "text": "假回答",
                "model": "deepseek-flash",
                "cost_usd": 0.0001,
                "usage": {"total_tokens": 10},
            },
        }

    monkeypatch.setattr("web.server.run_question", _fake_run_question)
    result = state.ask("今天天气真好。")
    assert result == {
        "reply": "假回答",
        "model": "deepseek-flash",
        "mode": "real",
        "cost_usd": 0.0001,
        "usage": {"total_tokens": 10},
    }
