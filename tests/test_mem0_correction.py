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
#   7. 两条聊天入口都把结果告诉 BOOS（CLI 打印、网页 ask 返回字段）；
#   8. 纠错判断（2026-11-09 修正轮）：多个候选分不清时请 DeepSeek
#      按意思 + 对话上下文判断改哪条（离线用假判断客户端模拟模型
#      的选择，模型真懂不懂语义靠真实 API 的生命周期复测覆盖）；
#      模型分不清/调用失败时改问 BOOS，不崩、不谎称改好了。

import json
import math
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from config import BudgetTracker
from conftest import FakeMemory, make_fake_store
from main import _print_correction_result, run_question
from memory.mem0_store import (
    Mem0Store,
    _cosine_similarity,
    _detect_correction,
    _parse_judge_choice,
)
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
    # Plan C：第一轮先待定，flush 后才落库，纠错改的才是落库后的那条。
    store.flush_pending()
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
    assert result["correction"]["layer"] == "ask"
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
    assert resolved["correction"]["layer"] == "exact"
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
    """普通聊天走正常保存：没有 "correction" 键，记忆正常存下来。

    Plan C：本轮先待定，flush 后才提交。"""
    store = make_fake_store(tmp_path)
    result = store.add_turn([
        {"role": "user", "content": "今天天气真好。我喜欢喝咖啡。"},
        {"role": "assistant", "content": "是啊，喝杯咖啡挺好的。"},
    ])
    assert "correction" not in result
    store.flush_pending()
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
    assert result["correction"]["layer"] == "none"
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
    # Plan C：第一轮先待定，flush 后才落库；before_calls 在落库后量，
    # 纠错轮次本身不调提取（只改旧记忆），总数才不变。
    store.flush_pending()
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
    assert result["correction"]["layer"] == "single"
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
    """网页入口：普通聊天时 ask() 的返回里没有 "correction" 键。
    第 0 步地基后返回是六个键：原来的五个 + timings_ms（本轮各截耗时）；
    没有 correction 键（普通聊天没纠错），timings_ms 里是本轮量到的两截
    （ask 的搜记忆 + 收尾的存记忆，假 run_question 不带计时）。"""
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
    assert result["reply"] == "假回答"
    assert result["model"] == "deepseek-flash"
    assert result["mode"] == "real"
    assert result["cost_usd"] == 0.0001
    assert result["usage"] == {"total_tokens": 10}
    assert "correction" not in result
    assert set(result["timings_ms"]) == {"memory_search_ms", "memory_save_ms"}


# ---------- 9. 纠错判断：模型按意思 + 对话上下文裁决（2026-11-09 修正轮） ----------
# 背景：生命周期 v3 复测里，“不对，我喜欢看纪录片。”同时撞上科幻电影
# （0.574）和 Go 写代码（0.516）两条候选，分数分不开，纠错直接
# ambiguous、改不成。修正：多个候选时请 DeepSeek 判断改哪条，
# 候选带上“在不在最近对话上下文里”当提示；模型分不清/调用失败
# 时才问 BOOS。下面用假判断客户端模拟模型的选择——离线测的是
# “上下文 → 提示 → 选择 → 替换”这条线通没通，模型真懂不懂语义
# （纪录片是电影、Go 是编程）靠真实 API 的生命周期复测覆盖。


def _install_fake_judge(store, choice="none", error=None, finish_reason=None):
    """给假 store 装一个假 DeepSeek 判断客户端（供纠错判断测试用）。

    _judge_correction 调 client.chat.completions.create 时：
    - error 给了就抛 error（模拟网络/限流失败）；
    - 否则返回 {"choice": choice}，choice 可以是：
      - 固定的候选 id / "none" / "unclear"；
      - 一个 dict（比如 {"choice": "none", "reason": "similar"}，
        原样当作模型返回的 JSON）；
      - 一个函数：收到 user 提示后现场算一个 choice（模拟模型按意思
        或按上下文挑，见下面的 _pick_in_context / _pick_movie）；
      - 一个列表：按调用次数依次取（最后一位重复），模拟“第一次
        回 none、重试才挑中”这类多轮行为。
    finish_reason：模拟每次调用的 finish_reason（比如 "length" 表示
    截断）；同样可以是单个值、函数或列表，默认 None（没截断）。
    返回 judge_calls：每次调用的 (args, kwargs) 记录，供测试断言
    “判断被调了 / 没被调 / 提示里带了什么 / 重试用了多少 Token”。
    FakeMemory 本来没有 llm 属性，这里是测试给它临时装一个，
    不影响其他测试。
    """
    calls = []

    def _resolve(value, prompt, index):
        if isinstance(value, list):
            picked = value[min(index, len(value) - 1)] if value else None
        else:
            picked = value
        return picked(prompt) if callable(picked) else picked

    class _FakeCompletions:
        def create(self, *args, **kwargs):
            calls.append((args, kwargs))
            if error is not None:
                raise error
            messages = kwargs.get("messages") or []
            prompt = ""
            for msg in messages:
                if (msg or {}).get("role") == "user":
                    prompt = (msg or {}).get("content") or ""
            index = len(calls) - 1
            picked = _resolve(choice, prompt, index)
            if isinstance(picked, dict):
                content = json.dumps(picked, ensure_ascii=False)
            else:
                content = json.dumps({"choice": picked}, ensure_ascii=False)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                        finish_reason=_resolve(finish_reason, prompt, index),
                    )
                ]
            )

    store._memory.llm = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    )
    return calls


def _judge_prompt(calls):
    """取假判断客户端收到的 user 提示（第一次调用的那份）。"""
    messages = calls[0][1].get("messages") or []
    for msg in messages:
        if (msg or {}).get("role") == "user":
            return (msg or {}).get("content") or ""
    return ""


def _correction_judge_calls(judge_calls):
    """只挑出纠错判的调用（42 号规则提炼是另一路调用，不算在这里）。

    judge_calls 是 _install_fake_judge 记的 (args, kwargs) 列表；纠错判的
    系统提示是“你是记忆纠错的判断助手”，规则提炼的是“纠正规则的提炼助手”。
    纠正成功后 42 号会再调一次模型提炼规则，所以数“纠错判调了几次”
    时必须过滤，不然每个成功纠正的测试都会多算一次。
    """
    return [
        call
        for call in judge_calls
        if "记忆纠错的判断助手" in str(call)
    ]


def _pick_in_context(prompt):
    """假“模型”：挑提示里标了“本轮对话刚提到过”的那条
    （模拟模型跟了上下文提示；括号里还可能跟个相似度， regex 要容得下）。"""
    match = re.search(r"id=(\S+?)（本轮对话刚提到过[^）]*）", prompt)
    return match.group(1) if match else "none"


def _pick_movie(prompt):
    """假“模型”：挑文本里有“电影”的那条（模拟模型按意思挑：
    纪录片是电影，Go 写代码是编程）。"""
    for match in re.finditer(r"id=(\S+?)（[^）]*）：([^\n]*)", prompt):
        if "电影" in match.group(2):
            return match.group(1)
    return "none"


def _pick_novel_in_context(prompt):
    """假“模型”：挑提示里标了“本轮对话刚提到过”、文本里又有“小说”
    的那条（模拟模型按意思 + 上下文决胜：纪录片和小说都是“看”的，
    刚聊的是小说就改小说那条）。"""
    for match in re.finditer(r"id=(\S+?)（本轮对话刚提到过[^）]*）：([^\n]*)", prompt):
        if "小说" in match.group(2):
            return match.group(1)
    return "none"


def _install_fake_embedder(store, angles):
    """给假 store 装一个假本地 embedding 模型（纠错安全网用）。

    angles：{文本: 夹角（弧度）}——embed 返回二维单位向量
    [cos(夹角), sin(夹角)]，于是两段文本的余弦相似度就是
    cos(夹角差)，测试能精确控制“话题像不像”：
    - 夹角差小（比如 0.44 弧度）→ 相似度约 0.90，高于阈值，通过；
    - 夹角差大（比如 1.36 弧度）→ 相似度约 0.21，低于阈值，拦下。
    没在 angles 里的文本默认夹角 0。FakeMemory 本来没有
    embedding_model 属性，这里是测试给它临时装一个，不影响其他测试。
    返回 embed 的调用记录 [(text, memory_action)]。
    """
    calls = []

    class _FakeEmbedder:
        def embed(self, text, memory_action=None):
            calls.append((text, memory_action))
            angle = angles.get(text, 0.0)
            return [math.cos(angle), math.sin(angle)]

    store._memory.embedding_model = _FakeEmbedder()
    return calls


def _store_with_scifi_and_go(tmp_path):
    """造一个有两条记忆的假 store：科幻电影 + Go 写代码
    （生命周期 v3 的原案，两条都是“我喜欢…”句式）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("用户喜欢看科幻电影。")
    store.add_memory_text("我现在喜欢用 Go 写代码。")
    return store


def _correct_to_documentary(store):
    """跑一次原案纠错：“不对，我喜欢看纪录片。”"""
    return store.add_turn([
        {"role": "user", "content": "不对，我喜欢看纪录片。"},
        {"role": "assistant", "content": "明白，我更新一下。"},
    ])


def test_chat_correction_lifecycle_case_judged_with_context(tmp_path):
    """v3 原案 + 话题在最近对话里：先聊科幻电影（上下文记下它），
    再说“不对，我喜欢看纪录片。”→ 模型跟上下文提示挑中科幻那条，
    只改它，Go 那条原封不动——之前这一步是 ambiguous、改不成的。
    注意：上下文里两条候选都在，所以这里测的是模型跟上下文提示
    挑——第 1 层直接定已去掉（2026-11-09 新方案），恰好 1 条在
    上下文里时也不直接定，同样走模型判，见
    test_judge_called_with_single_in_context。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    go_id = store.get_all()[1]["id"]
    # 模拟前面那轮提问：两条都被检索到、拼进提示了。
    store.note_context([{"id": scifi_id}, {"id": go_id}])
    judge_calls = _install_fake_judge(store, choice=_pick_in_context)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == scifi_id
    assert result["correction"]["old_text"] == "用户喜欢看科幻电影。"
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我现在喜欢用 Go 写代码。",
    ]
    # 提示里两条都标了“刚提到过”（上下文里两条都在）。
    prompt = _judge_prompt(judge_calls)
    assert prompt.count("（本轮对话刚提到过") == 2
    assert "最近对话没提到" not in prompt
    assert "用户喜欢看科幻电影。" in prompt
    assert "我现在喜欢用 Go 写代码。" in prompt
    # 找候选的那次检索没有污染上下文：上下文里还是只有之前记的那两条。
    assert list(store._recent_context_ids) == [scifi_id, go_id]


def test_chat_correction_fallback_judges_by_meaning_without_context(tmp_path):
    """v3 原案 + 话题最近没聊：上下文是空的，模型按意思挑（纪录片是
    电影）→ 还是改科幻那条，Go 不动；提示里两条都标“没提到”。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    judge_calls = _install_fake_judge(store, choice=_pick_movie)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == scifi_id
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我现在喜欢用 Go 写代码。",
    ]
    prompt = _judge_prompt(judge_calls)
    assert "最近对话没提到" in prompt
    # 没有任何一条候选被标成“刚提到过”（括号里的标记；提示语正文里
    # 本来就有“本轮对话刚提到过”这几个字，那不算标记）。
    assert "（本轮对话刚提到过" not in prompt


def test_context_hint_does_not_mislead_judge(tmp_path):
    """上下文只是提示、模型才是裁判：上下文记的是 Go（前面在聊编程），
    但纠错说的是电影——按意思挑的模型不会因为 Go“刚提到过”就改 Go，
    还是改科幻电影那条。第 1 层直接定已去掉（2026-11-09 新方案），
    恰好只有 Go 在上下文里时也不直接定，同样走模型判——这正是 C3
    要防的情形，见下面的安全网测试。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    go_id = store.get_all()[1]["id"]
    store.note_context([{"id": scifi_id}, {"id": go_id}])
    _install_fake_judge(store, choice=_pick_movie)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["old_text"] == "用户喜欢看科幻电影。"
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我现在喜欢用 Go 写代码。",
    ]


def test_same_topic_correction_judge_uses_context(tmp_path):
    """同一话题两条记忆（科幻电影/科幻小说）：光按意思分不开，
    上下文说刚才在聊小说 → 模型按上下文提示挑中小说那条改，电影
    那条不动。第 1 层直接定已去掉（2026-11-09 新方案）：恰好只有
    小说在上下文里也不直接定，照样调模型判，上下文是提示里的决胜
    依据。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    novel_id = store.get_all()[1]["id"]
    store.note_context([{"id": novel_id}])
    judge_calls = _install_fake_judge(store, choice=_pick_in_context)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == novel_id
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看科幻电影。",
        "我喜欢看纪录片。",
    ]
    # 纠错判被调了一次（第 1 层没了，不再“不调模型直接定”）。
    # 注意：只数纠错判（系统提示是“记忆纠错的判断助手”）——纠正成功后
    # 42 号还会再调一次模型提炼规则（系统提示是“纠正规则的提炼助手”），
    # 那是另一回事，不算在这里。
    correction_judge_calls = [
        call
        for call in judge_calls
        if "记忆纠错的判断助手" in str(call)
    ]
    assert len(correction_judge_calls) == 1


def test_judge_unsure_asks_user_then_resolves(tmp_path):
    """模型也分不清（回 none，没写 reason → 默认按 "similar" 处理，
    用 2000 重试一次；重试还是 none）→ 还是 ambiguous：什么都不改、
    不存重复，等 BOOS 下句挑；挑“第 2 条”后按挑中的改，判断不再重调。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    judge_calls = _install_fake_judge(store, choice="none")
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "ambiguous"
    assert result["correction"]["layer"] == "ask"
    assert len(result["correction"]["candidates"]) == 2
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看科幻电影。",
        "我喜欢看科幻小说。",
    ]
    assert store._memory.update_calls == []
    # 第一次 none（默认 similar）重试了一次，一共调了两次。
    assert len(judge_calls) == 2
    assert judge_calls[0][1]["max_tokens"] == 1000
    assert judge_calls[1][1]["max_tokens"] == 2000
    resolved = store.add_turn([
        {"role": "user", "content": "第 2 条"},
        {"role": "assistant", "content": "好的，改第二条。"},
    ])
    assert resolved["correction"]["status"] == "updated"
    assert resolved["correction"]["old_text"] == "我喜欢看科幻小说。"
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看科幻电影。",
        "我喜欢看纪录片。",
    ]
    # 挑中后走原文对上的路，不用再请模型判断（纠错判还是两次；
    # 纠正成功后 42 号提炼规则又调了一次模型，那是另一路，不算在这里）。
    assert len(_correction_judge_calls(judge_calls)) == 2


def test_judge_failure_falls_back_to_asking_user(tmp_path):
    """判断调用失败（模拟网络/限流抛错）→ 不崩、不谎称改好了：
    回 ambiguous 问 BOOS，两条旧记忆原封不动；调用异常不重试，
    只调了一次。"""
    store = _store_with_scifi_and_go(tmp_path)
    judge_calls = _install_fake_judge(store, error=RuntimeError("网络炸了"))
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "ambiguous"
    assert result["correction"]["layer"] == "ask"
    assert len(result["correction"]["candidates"]) == 2
    assert [item["text"] for item in store.get_all()] == [
        "用户喜欢看科幻电影。",
        "我现在喜欢用 Go 写代码。",
    ]
    assert store._memory.update_calls == []
    assert len(judge_calls) == 1


def test_judge_unknown_choice_asks_user(tmp_path):
    """模型回了一个不在候选里的 id → 当作分不清，问 BOOS，不猜。
    认不出的回答按 "similar" 处理（重试一次是安全的），所以一共
    调了两次，重试还是认不出才问 BOOS。"""
    store = _store_with_scifi_and_go(tmp_path)
    judge_calls = _install_fake_judge(store, choice="fake-999")
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "ambiguous"
    assert result["correction"]["layer"] == "ask"
    assert [item["text"] for item in store.get_all()] == [
        "用户喜欢看科幻电影。",
        "我现在喜欢用 Go 写代码。",
    ]
    assert len(judge_calls) == 2


def test_judge_not_called_for_single_candidate(tmp_path):
    """只有一个候选时直接改，不用请模型（省一次调用）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    judge_calls = _install_fake_judge(store, choice="none")
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    # 纠错判一次没调（42 号提炼规则会调一次模型，那是另一路，不算在这里）。
    assert _correction_judge_calls(judge_calls) == []
    assert [item["text"] for item in store.get_all()] == ["我喜欢看纪录片。"]


def test_ordinary_message_does_not_call_judge(tmp_path):
    """普通聊天不触发纠错判断：没有 "correction" 键，判断一次都没调，
    记忆正常存下来。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("用户喜欢看科幻电影。")
    store.add_memory_text("我现在喜欢用 Go 写代码。")
    judge_calls = _install_fake_judge(store, choice="none")
    result = store.add_turn([
        {"role": "user", "content": "今天天气真好。我喜欢喝咖啡。"},
        {"role": "assistant", "content": "是啊，喝杯咖啡挺好的。"},
    ])
    assert "correction" not in result
    assert judge_calls == []
    # Plan C：本轮先待定，flush 后才提交。
    store.flush_pending()
    assert "我喜欢喝咖啡" in [item["text"] for item in store.get_all()]


def test_parse_judge_choice():
    """_parse_judge_choice：标准 JSON、围栏 JSON、直接写 id、数字序号
    都认，返回 (id, None)；none/unclear/格式不对/不在候选里的 id 都
    返回 (None, reason)，reason 分 "similar" 和 "unrelated" 两种，
    缺失或认不出时默认 "similar"（不猜，只定重试还是不重试）。"""
    candidates = [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}]
    assert _parse_judge_choice('{"choice": "a"}', candidates) == ("a", None)
    assert _parse_judge_choice('```json\n{"choice": "b"}\n```', candidates) == ("b", None)
    assert _parse_judge_choice("a", candidates) == ("a", None)
    assert _parse_judge_choice('{"choice": 2}', candidates) == ("b", None)
    assert _parse_judge_choice("2", candidates) == ("b", None)
    # none：reason 原样认，缺失默认 similar。
    assert _parse_judge_choice('{"choice": "none"}', candidates) == (None, "similar")
    assert _parse_judge_choice(
        '{"choice": "none", "reason": "similar"}', candidates
    ) == (None, "similar")
    assert _parse_judge_choice(
        '{"choice": "none", "reason": "unrelated"}', candidates
    ) == (None, "unrelated")
    assert _parse_judge_choice(
        '{"choice": "none", "reason": "不知道"}', candidates
    ) == (None, "similar")
    assert _parse_judge_choice('{"choice": "unclear"}', candidates) == (None, "similar")
    assert _parse_judge_choice('{"choice": "zzz"}', candidates) == (None, "similar")
    assert _parse_judge_choice("我不知道", candidates) == (None, "similar")
    assert _parse_judge_choice("", candidates) == (None, "similar")
    assert _parse_judge_choice('{"choice": 9}', candidates) == (None, "similar")
    assert _parse_judge_choice("not json at all {{{", candidates) == (None, "similar")


def test_note_context_records_ids_and_caps(tmp_path):
    """note_context：记 search 返回的 id，也认直接给的 id 字符串；
    最多留 20 条，旧的自动挤掉；传空/None 不崩。"""
    store = make_fake_store(tmp_path)
    store.note_context(None)
    store.note_context([])
    assert list(store._recent_context_ids) == []
    store.note_context([{"id": "a"}, {"id": "b"}])
    store.note_context(["c"])
    assert list(store._recent_context_ids) == ["a", "b", "c"]
    store.note_context([{"id": f"m-{i}"} for i in range(25)])
    assert len(store._recent_context_ids) == 20
    assert list(store._recent_context_ids)[:1] == ["m-5"]
    assert list(store._recent_context_ids)[-1] == "m-24"


def test_cli_run_question_records_context(monkeypatch, tmp_path):
    """CLI 入口：提问前 search 到的记忆 id 被记进上下文
    （之后纠错时当“当前在聊什么”的提示用）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫 BOOS。")
    mem_id = store.get_all()[0]["id"]
    with patch(
        "providers.deepseek.requests.post",
        return_value=_fake_ok_response(),
    ):
        run_question(
            "你记得我叫什么吗？",
            budget_tracker=tracker,
            mem0_store=store,
        )
    assert mem_id in list(store._recent_context_ids)


def test_lifecycle_case_via_chat_turns_with_judge(tmp_path):
    """v3 原案走完整聊天轮次：先聊科幻电影、再聊 Go（两条都经提取存下），
    期间科幻电影被检索进过上下文；再说“不对，我喜欢看纪录片。”→
    模型挑中科幻那条替换，Go 不动——这正是 v3 复测里失败的那一步。
    注意：上下文里两条候选都在，第 1 层直接定不触发，走第 2 层模型判。"""
    store = make_fake_store(tmp_path)
    store.add_turn([
        {"role": "user", "content": "我喜欢看科幻电影。"},
        {"role": "assistant", "content": "明白，你喜欢看科幻电影，我记下了。"},
    ])
    store.add_turn([
        {"role": "user", "content": "我现在喜欢用 Go 写代码。"},
        {"role": "assistant", "content": "明白，你喜欢用 Go 写代码，我记下了。"},
    ])
    # Plan C：两轮都在待定区，flush 后才都落库（第二轮调 add_turn 时
    # 第一轮已提交，第二轮在 flush 时提交）。
    store.flush_pending()
    scifi_id = store.get_all()[0]["id"]
    go_id = store.get_all()[1]["id"]
    store.note_context([{"id": scifi_id}, {"id": go_id}])
    _install_fake_judge(store, choice=_pick_in_context)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    texts = [item["text"] for item in store.get_all()]
    assert "我喜欢看纪录片。" in texts
    assert "我现在喜欢用 Go 写代码" in texts
    assert not any("科幻" in text for text in texts)

# ---------- 10. 三层纠正（BOOS 2026-11-09 定） ----------
# 第 1 层直接定：≥2 个候选里恰好只有 1 条在最近对话上下文里 →
# 直接选它，不调模型；0 条或 ≥2 条在里面 → 走第 2 层模型判。
# 第 2 层模型判：上下文是意思分不开时的决胜依据（提示里写明），
# 模型回 none 要分 "similar"（分不清）和 "unrelated"（不知道）；
# 第一次截断或 "similar" 的 none → 用 2000 同一个提示再试一次，
# "unrelated" 的 none 和调用异常不重试；重试还是不行 → 问 BOOS。
# 第 3 层问 BOOS：现有 ambiguous/pending 流程不变。
# correct_memory 的返回里多一个 "layer" 字段标明哪一层定的。


def test_judge_called_with_single_in_context(tmp_path):
    """第 1 层直接定已去掉（2026-11-09 新方案）：两条候选里恰好只有
    科幻在上下文里，也不再直接定——照样调模型按槽位判，layer 是
    "judge"；模型按意思挑中科幻，Go 不动。C3 的教训：上下文里只有
    1 条就直接定，会静默改错（上下文只有 Go 时就把 Go 改了）。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    store.note_context([{"id": scifi_id}])
    judge_calls = _install_fake_judge(store, choice=_pick_movie)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == scifi_id
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我现在喜欢用 Go 写代码。",
    ]
    # 第 1 层没了，模型判必须被调一次（1000 上限）。
    # 只数纠错判——纠正成功后 42 号提炼规则又调了一次模型，不算在这里。
    correction_calls = _correction_judge_calls(judge_calls)
    assert len(correction_calls) == 1
    assert correction_calls[0][1]["max_tokens"] == 1000


def test_judge_called_with_zero_in_context(tmp_path):
    """上下文里 0 条候选：走模型判，模型按意思挑中科幻，layer 是 "judge"。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    judge_calls = _install_fake_judge(store, choice=_pick_movie)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == scifi_id
    # 只数纠错判——纠正成功后 42 号提炼规则又调了一次模型，不算在这里。
    correction_calls = _correction_judge_calls(judge_calls)
    assert len(correction_calls) == 1
    assert correction_calls[0][1]["max_tokens"] == 1000


def test_judge_called_with_two_in_context(tmp_path):
    """上下文里 ≥2 条候选：走模型判，模型按意思挑中科幻，layer 是 "judge"。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    go_id = store.get_all()[1]["id"]
    store.note_context([{"id": scifi_id}, {"id": go_id}])
    judge_calls = _install_fake_judge(store, choice=_pick_movie)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == scifi_id
    # 只数纠错判——纠正成功后 42 号提炼规则又调了一次模型，不算在这里。
    assert len(_correction_judge_calls(judge_calls)) == 1


def test_judge_truncation_retries_once_at_2000(tmp_path):
    """第 2 层重试（截断）：第一次 finish_reason="length" → 用 2000
    同一个提示再试一次，挑中科幻；一共调两次，第二次 max_tokens=2000。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    judge_calls = _install_fake_judge(
        store, choice=scifi_id, finish_reason=["length", None]
    )
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == scifi_id
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我现在喜欢用 Go 写代码。",
    ]
    # 只数纠错判——纠正成功后 42 号提炼规则又调了一次模型，不算在这里。
    correction_calls = _correction_judge_calls(judge_calls)
    assert len(correction_calls) == 2
    assert correction_calls[0][1]["max_tokens"] == 1000
    assert correction_calls[1][1]["max_tokens"] == 2000
    # 重试用的是同一个提示。
    messages_0 = correction_calls[0][1]["messages"]
    messages_1 = correction_calls[1][1]["messages"]
    assert messages_0 == messages_1


def test_judge_similar_none_retries_and_picks(tmp_path):
    """第 2 层重试（分不清）：第一次回 {"choice": "none",
    "reason": "similar"} → 用 2000 再试一次，挑中科幻；一共调两次。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    judge_calls = _install_fake_judge(
        store,
        choice=[{"choice": "none", "reason": "similar"}, scifi_id],
    )
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == scifi_id
    # 只数纠错判——纠正成功后 42 号提炼规则又调了一次模型，不算在这里。
    correction_calls = _correction_judge_calls(judge_calls)
    assert len(correction_calls) == 2
    assert correction_calls[0][1]["max_tokens"] == 1000
    assert correction_calls[1][1]["max_tokens"] == 2000


def test_judge_unrelated_none_does_not_retry(tmp_path):
    """第 2 层不重试（不知道）：第一次回 {"choice": "none",
    "reason": "unrelated"} → 只调一次，直接 ambiguous 问 BOOS。"""
    store = _store_with_scifi_and_go(tmp_path)
    judge_calls = _install_fake_judge(
        store, choice={"choice": "none", "reason": "unrelated"}
    )
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "ambiguous"
    assert result["correction"]["layer"] == "ask"
    assert len(result["correction"]["candidates"]) == 2
    assert [item["text"] for item in store.get_all()] == [
        "用户喜欢看科幻电影。",
        "我现在喜欢用 Go 写代码。",
    ]
    assert len(judge_calls) == 1


def test_judge_retry_still_fails_after_similar_none(tmp_path):
    """重试也失败（两次都回 similar 的 none）→ ambiguous 问 BOOS，
    总共最多调两次，不会无限重试。"""
    store = _store_with_scifi_and_go(tmp_path)
    judge_calls = _install_fake_judge(
        store,
        choice=[
            {"choice": "none", "reason": "similar"},
            {"choice": "none", "reason": "similar"},
        ],
    )
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "ambiguous"
    assert result["correction"]["layer"] == "ask"
    assert store._memory.update_calls == []
    assert len(judge_calls) == 2
    assert judge_calls[1][1]["max_tokens"] == 2000


def test_judge_retry_still_truncated_asks_user(tmp_path):
    """重试也失败（两次都被截断）→ ambiguous 问 BOOS，总共调两次。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    judge_calls = _install_fake_judge(
        store, choice=scifi_id, finish_reason=["length", "length"]
    )
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "ambiguous"
    assert result["correction"]["layer"] == "ask"
    assert store._memory.update_calls == []
    assert len(judge_calls) == 2
    assert judge_calls[1][1]["max_tokens"] == 2000


def test_judge_prompt_tiebreaker_rule(tmp_path):
    """提示决胜依据：同一话题两条记忆（科幻电影/科幻小说）都在上下
    文里（第 1 层不触发），提示里写明上下文是意思分不开时的决胜
    依据、且有“别硬套上下文”的 guard；假模型按“刚聊的是小说”
    挑中小说那条，电影那条不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    movie_id = store.get_all()[0]["id"]
    novel_id = store.get_all()[1]["id"]
    store.note_context([{"id": movie_id}, {"id": novel_id}])
    judge_calls = _install_fake_judge(store, choice=_pick_novel_in_context)
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["layer"] == "judge"
    assert result["correction"]["memory_id"] == novel_id
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看科幻电影。",
        "我喜欢看纪录片。",
    ]
    prompt = _judge_prompt(judge_calls)
    # 槽位判断的文字在提示里：新说法和旧记忆必须在同一个槽位上
    # 才构成替换（2026-11-09 新方案；提示里的换行位置和这里一致）。
    assert "槽位" in prompt
    assert "同一个\n槽位上才构成替换" in prompt
    # 还有“别选和新说法几乎一样的那条”这句（v7 复测补的）：
    # 和新说法意思几乎一样的那条不用改，要改的是被取代的那条。
    assert "和新说法意思几乎一样的那条不用改" in prompt
    # 决胜规则的文字在提示里。
    assert "决胜依据" in prompt
    assert "刚聊的那条如果和新说法是同一话题，就选刚聊的那条" in prompt
    assert "只有刚聊的也分不出来" in prompt
    # guard 还在：新说法明明在说另一件旧事时别硬选刚提到的那条。
    assert "不要因为\n它最近没提到就硬选刚提到的那条" in prompt
    # none 要分 similar / unrelated 两种，格式写在提示里。
    assert '"reason": "similar" 或 "unrelated"' in prompt


def test_judge_none_without_reason_defaults_to_similar_retry(tmp_path):
    """none 兜底：模型回 {"choice": "none"} 没写 reason → 默认按
    "similar" 处理（重试一次是安全的），重试还是 none → ambiguous，
    一共调两次。"""
    store = _store_with_scifi_and_go(tmp_path)
    judge_calls = _install_fake_judge(
        store,
        choice=[{"choice": "none"}, {"choice": "none"}],
    )
    result = _correct_to_documentary(store)
    assert result["correction"]["status"] == "ambiguous"
    assert result["correction"]["layer"] == "ask"
    assert len(judge_calls) == 2
    assert judge_calls[1][1]["max_tokens"] == 2000


def test_correction_layer_field_values(tmp_path):
    """layer 字段各状态的取值（直接调 correct_memory）：
    exact（原文完全相同）、single（单候选）、none（not_found）、
    unchanged 沿用选了目标的那一层。judge/ask/judge_blocked 见
    上面各测试——"direct" 已经没有了（2026-11-09 新方案去掉
    第 1 层直接定）。"""
    # exact：原文对上了就不用猜。
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    outcome = store.correct_memory("我喜欢看科幻电影。", "我喜欢看纪录片。")
    assert outcome["status"] == "updated"
    assert outcome["layer"] == "exact"
    # single：只有一个候选、又不是原文完全相同。
    single = make_fake_store(tmp_path)
    single.add_memory_text("我喜欢喝咖啡。")
    outcome = single.correct_memory("我喜欢喝咖啡吗", "我喜欢喝茶。")
    assert outcome["status"] == "updated"
    assert outcome["layer"] == "single"
    # not_found：没有目标，layer 是 none。
    empty = make_fake_store(tmp_path)
    outcome = empty.correct_memory("我喜欢喝茶", "我喜欢喝咖啡")
    assert outcome["status"] == "not_found"
    assert outcome["layer"] == "none"
    # unchanged：新旧一样，layer 是实际选了目标的那一层（exact）。
    stored = make_fake_store(tmp_path)
    stored.add_memory_text("我喜欢用 Go 写代码。")
    outcome = stored.correct_memory("我喜欢用 Go 写代码", "我喜欢用 Go 写代码。")
    assert outcome["status"] == "unchanged"
    assert outcome["layer"] == "exact"


# ---------- 9. 安全网（2026-11-09 新方案）：模型判挑中后、真正替换之前，
# 用本地向量算旧记忆和新说法的相似度；话题不像就不换、改问 BOOS。
# 第 1 层直接定已去掉，C3 的情形（上下文只有 Go、要改科幻）现在走
# 模型判 + 安全网：模型按槽位挑对了就换，挑错了（挑中 Go）安全网
# 拦下。假 embedding 模型用夹角控制相似度（见 _install_fake_embedder）：
# 纪录片 0 弧度、科幻电影 0.44 弧度（相似度约 0.90，通过）、Go 写代码
# 1.36 弧度（相似度约 0.21，低于 0.56 阈值，拦下）。


def test_judge_pick_blocked_by_topic_check(tmp_path):
    """安全网拦下：模型判错挑中了 Go（模拟 C3 里模型没按槽位挑），
    旧记忆（Go 写代码）和新说法（纪录片）相似度只有约 0.21，
    低于 0.56 阈值 → 不换，status="ambiguous"、layer="judge_blocked"、
    resolution="judge_blocked"，Go 那条原封不动，等 BOOS 挑。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    go_id = store.get_all()[1]["id"]
    _install_fake_judge(store, choice=go_id)
    _install_fake_embedder(
        store,
        {
            "我喜欢看纪录片。": 0.0,
            "用户喜欢看科幻电影。": 0.44,
            "我现在喜欢用 Go 写代码。": 1.36,
        },
    )
    result = _correct_to_documentary(store)
    correction = result["correction"]
    assert correction["status"] == "ambiguous"
    assert correction["layer"] == "judge_blocked"
    assert correction["resolution"] == "judge_blocked"
    assert correction["memory_id"] is None
    assert len(correction["candidates"]) == 2
    # 什么都没改：Go 和科幻都在原位。
    assert [item["text"] for item in store.get_all()] == [
        "用户喜欢看科幻电影。",
        "我现在喜欢用 Go 写代码。",
    ]
    assert store._memory.update_calls == []
    # 审计里记着：模型挑的是 Go、相似度约 0.21（低于阈值）。
    audit = correction["audit"]
    assert audit["picked_id"] == go_id
    assert audit["old_new_similarity"] is not None
    assert abs(audit["old_new_similarity"] - 0.21) < 0.02
    assert audit["old_new_similarity"] < 0.56
    assert set(audit["candidate_scores"]) == {scifi_id, go_id}


def test_judge_pick_passes_topic_check(tmp_path):
    """安全网通过：模型按槽位挑中科幻，旧记忆（科幻电影）和新说法
    （纪录片）相似度约 0.90，高于阈值 → 正常换，layer="judge"、
    resolution="updated"，审计里记着相似度。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    _install_fake_judge(store, choice=_pick_movie)
    _install_fake_embedder(
        store,
        {
            "我喜欢看纪录片。": 0.0,
            "用户喜欢看科幻电影。": 0.44,
            "我现在喜欢用 Go 写代码。": 1.36,
        },
    )
    result = _correct_to_documentary(store)
    correction = result["correction"]
    assert correction["status"] == "updated"
    assert correction["layer"] == "judge"
    assert correction["resolution"] == "updated"
    assert correction["memory_id"] == scifi_id
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我现在喜欢用 Go 写代码。",
    ]
    audit = correction["audit"]
    assert audit["picked_id"] == scifi_id
    assert audit["old_new_similarity"] is not None
    assert abs(audit["old_new_similarity"] - 0.90) < 0.02
    assert audit["old_new_similarity"] >= 0.56


def test_topic_check_skipped_when_embedder_unavailable(tmp_path):
    """安全网坏了不堵死纠正：没装 embedding 模型（离线假对象默认
    就没有，和真实环境里模型加载失败一样），相似度算不出来 →
    跳过校验、按模型判的结果换，layer="judge"。主防线是模型判，
    安全网只是第二层。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    _install_fake_judge(store, choice=_pick_movie)
    assert not hasattr(store._memory, "embedding_model")
    result = _correct_to_documentary(store)
    correction = result["correction"]
    assert correction["status"] == "updated"
    assert correction["layer"] == "judge"
    assert correction["resolution"] == "updated"
    assert correction["memory_id"] == scifi_id
    assert correction["audit"]["old_new_similarity"] is None


def test_single_candidate_blocked_by_topic_check(tmp_path):
    """单候选也走安全网：库里只有 Go 写代码一条，旧说法找回来它
    （单候选、不是原文完全相同），新说法是纪录片，相似度约 0.21、
    低于 0.56 阈值 → 不换，status="ambiguous"、layer="judge_blocked"、
    resolution="judge_blocked"，Go 那条原封不动，等 BOOS 挑。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我现在喜欢用 Go 写代码。")
    go_id = store.get_all()[0]["id"]
    _install_fake_embedder(
        store,
        {
            "我喜欢看纪录片。": 0.0,
            "我现在喜欢用 Go 写代码。": 1.36,
        },
    )
    outcome = store.correct_memory("我现在喜欢用 Go 写代码吗", "我喜欢看纪录片。")
    assert outcome["status"] == "ambiguous"
    assert outcome["layer"] == "judge_blocked"
    assert outcome["resolution"] == "judge_blocked"
    assert outcome["memory_id"] is None
    assert [item["text"] for item in store.get_all()] == ["我现在喜欢用 Go 写代码。"]
    assert store._memory.update_calls == []
    audit = outcome["audit"]
    assert audit["picked_id"] == go_id
    assert audit["old_new_similarity"] is not None
    assert abs(audit["old_new_similarity"] - 0.21) < 0.02
    assert audit["old_new_similarity"] < 0.56


def test_single_candidate_passes_topic_check(tmp_path):
    """单候选安全网通过：库里只有喝咖啡一条，旧说法找回来它，
    新说法是喝茶，相似度约 0.90、高于阈值 → 正常换，layer 还是
    "single"、resolution="updated"，审计里记着相似度。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    _install_fake_embedder(
        store,
        {
            "我喜欢喝茶。": 0.0,
            "我喜欢喝咖啡。": 0.44,
        },
    )
    outcome = store.correct_memory("我喜欢喝咖啡吗", "我喜欢喝茶。")
    assert outcome["status"] == "updated"
    assert outcome["layer"] == "single"
    assert outcome["resolution"] == "updated"
    assert [item["text"] for item in store.get_all()] == ["我喜欢喝茶。"]
    audit = outcome["audit"]
    assert audit["old_new_similarity"] is not None
    assert abs(audit["old_new_similarity"] - 0.90) < 0.02
    assert audit["old_new_similarity"] >= 0.56


def test_correction_audit_marking(tmp_path):
    """审计标记：每次纠正的返回里都有 resolution + audit。
    updated 的审计里有挑中的 id、候选分数、当时上下文里的 id；
    ambiguous（模型分不清问 BOOS）的审计里记着 judge_reason。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    go_id = store.get_all()[1]["id"]
    store.note_context([{"id": scifi_id}])
    _install_fake_judge(store, choice=_pick_movie)
    result = _correct_to_documentary(store)
    correction = result["correction"]
    assert correction["resolution"] == "updated"
    audit = correction["audit"]
    assert audit["picked_id"] == scifi_id
    assert audit["judge_reason"] is None
    assert set(audit["candidate_scores"]) == {scifi_id, go_id}
    assert audit["context_ids"] == [scifi_id]
    # ambiguous 的那路也有 resolution + audit（judge_reason 记着
    # 模型为什么分不清）。
    store2 = _store_with_scifi_and_go(tmp_path)
    _install_fake_judge(
        store2, choice={"choice": "none", "reason": "unrelated"}
    )
    result2 = _correct_to_documentary(store2)
    correction2 = result2["correction"]
    assert correction2["status"] == "ambiguous"
    assert correction2["layer"] == "ask"
    assert correction2["resolution"] == "ask"
    assert correction2["audit"]["judge_reason"] == "unrelated"
    assert correction2["audit"]["picked_id"] is None


def test_cosine_similarity():
    """_cosine_similarity 纯函数：相同方向 1、无关 0、相反 -1、
    零向量/长度不一 0（不崩）。"""
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert _cosine_similarity([1.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert _cosine_similarity([], []) == pytest.approx(0.0)
    # 长度不影响余弦：[2, 0] 和 [1, 0] 同向。
    assert _cosine_similarity([2.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    # 45 度夹角约 0.707。
    assert _cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(
        0.7071, abs=1e-3
    )
