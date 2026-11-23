# tests/test_mind_reading_20261116.py（揣摩人心 + 批量删路由，2026-11-16）
# 作用：把 2026-11-16 的揣摩人心功能和批量删路由修法钉死在测试里，
# 全部离线模拟，不调真实 API、不花钱。
# 揣摩（设计方案 ~/workspace/goals/agent-2/files/十二揣摩人心设计方案.md）：
# 当场揣摩为主（对话进行中就能揣），置信度模型自己判断不数条数，
# 证据用大白话、范围只揣想法不揣关系，揣出来当场问，对就记下、不对就删除。
# 原理：记忆库用 conftest 的 make_fake_store（FakeMemory，没 llm 客户端，
# 所以 _call_llm_text 默认返回 None——没模型就不揣、不崩）；要揣时用
# monkeypatch 把 store._call_llm_text 换成按剧本回话的假模型。
# 批量删路由（2026-11-16 10:08 真机）：BOOS 说“把拉面、冒菜、麻辣烫三条
# 记忆一起删掉”，十二先单删了拉面、又批量删了另外两条，弹了两张卡。
# 工具本身没坏，是模型挑工具时没把三条一起走批量——修法是把两条工具的
# 说明开头第一句就写死路由规则（delete_memory 一次只删一条，2 条及以上
# 走 delete_memories），并用网关端到端钉死“三条一起删只弹一张卡”。

import json
import re

import pytest

from conftest import FakeMemory, make_fake_store
from memory import inference as inference_mod
from memory import ledger as ledger_mod
from tasks.task_store import TaskStore
from tools.agent_loop import AgentLoop
from tools.audit import AuditLog
from tools.memory_tool import (
    bound_memory_store,
    register_memory_tools,
)
from tools.tool_gateway import ToolGateway
from tools.tool_registry import get_tool


# ---------- 假模型：按剧本回揣摩的那一句话 ----------


class ScriptedInferenceLLM:
    """按剧本回揣摩输出的假模型：每次调按顺序回一句，调了几次都记下来。
    剧本用完了就回“没什么可猜的。”——模型没把握时就是这么说的。"""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = 0

    def __call__(self, system_prompt, user_prompt, max_tokens=500):
        self.calls += 1
        if self.script:
            return self.script.pop(0)
        return "没什么可猜的。"


def _spot_script(guess="BOOS今晚累了想歇会儿", evidence="他说今晚好累、明天再说"):
    return [f"我猜{guess}，因为{evidence}。我只是猜的。（归纳）"]


def _texts(mem_store):
    return [item["text"] for item in mem_store.get_all(include_sensitive=True)]


def _mem_id(mem_store, text):
    for item in mem_store.get_all(include_sensitive=True):
        if item["text"] == text:
            return item["id"]
    raise AssertionError(f"库里没有这条记忆：{text}")


def _ledger_events(mem_store):
    events, _info = ledger_mod.read_events(mem_store.ledger_path)
    return events or []


# ---------- 1. 解析：模型那一句话怎么认 ----------


def test_parse_normal_spot_sentence():
    parsed = inference_mod.parse_inference_output(
        "我猜BOOS今晚累了想歇会儿，因为他说今晚好累、明天再说。"
        "我只是猜的。（归纳）"
    )
    assert parsed is not None
    assert parsed["guess"] == "BOOS今晚累了想歇会儿"
    assert parsed["evidence"] == "他说今晚好累、明天再说"
    assert parsed["confidence"] == "low"
    assert parsed["confidence_text"] == "我只是猜的"
    assert parsed["layer"] == "induction"


def test_parse_no_inference_marker_is_none():
    assert inference_mod.parse_inference_output("没什么可猜的。") is None
    assert inference_mod.parse_inference_output("没什么可猜的") is None


def test_parse_missing_evidence_is_none():
    # 没有“因为”——猜了没给凭据，不记（设计方案第八节：每条推断都要有证据）。
    assert inference_mod.parse_inference_output("我猜BOOS累了。我只是猜的。") is None
    assert inference_mod.parse_inference_output("我猜，因为他很累。") is None


def test_parse_confidence_phrases():
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS想换个手机，因为他总抱怨现在的慢。我很确定。（演绎）"
        )["confidence"]
        == "high"
    )
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS想换个手机，因为他总抱怨现在的慢。我比较确定。（演绎）"
        )["confidence"]
        == "medium"
    )
    # 没写确定程度时默认 low——宁可往没把握里算。
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS想换个手机，因为他总抱怨现在的慢。（演绎）"
        )["confidence"]
        == "low"
    )


def test_parse_layer_tag_and_heuristic():
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS想歇会儿，因为他说累了。我只是猜的。（演绎）"
        )["layer"]
        == "deduction"
    )
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS想歇会儿，因为他说累了。我只是猜的。（矛盾）"
        )["layer"]
        == "contradiction"
    )
    # 没贴层标签时按证据里的词猜：提到“之前/总/每次/一直”就是归纳。
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS最近总在加班，因为他之前每次都这么晚才回。我只是猜的。"
        )["layer"]
        == "induction"
    )
    # 证据里说“对不上/矛盾”就是矛盾层。
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS其实不想去，因为他嘴上说想去、行动上总在拖，对不上。"
            "我只是猜的。"
        )["layer"]
        == "contradiction"
    )
    # 都不像时默认演绎。
    assert (
        inference_mod.parse_inference_output(
            "我猜BOOS想歇会儿，因为他说累了。我只是猜的。"
        )["layer"]
        == "deduction"
    )


def test_render_question_and_resolved():
    inference = {
        "guess": "BOOS今晚累了想歇会儿",
        "evidence": "他说今晚好累、明天再说",
        "confidence": "low",
        "confidence_text": "我只是猜的",
        "layer": "induction",
    }
    question = inference_mod.render_inference_question(inference)
    # 问 BOOS 时主语换成“你”，不写账本序号、不写层标签。
    assert question == (
        "我猜你今晚累了想歇会儿，因为你说今晚好累、明天再说。"
        "我只是猜的。对吗？"
    )
    assert inference_mod.render_inference_resolved({"result": "confirmed"}) == "记下了。"
    assert inference_mod.render_inference_resolved({"result": "denied"}) == "删掉了，猜错了。"
    # “差不多”“对了一半”：算认了，但只认了一半，回的话也留一半。
    assert (
        inference_mod.render_inference_resolved({"result": "confirmed", "partial": True})
        == "记下了，对了一半。"
    )


def test_bare_answers_only():
    # is_bare_* 只认光秃秃的整句（带后半句的不算，那是 classify 的活）：
    # “对，我就是累了”不是光秃秃的回答；“嗯”单独一句就是在回“对吗？”。
    for text in ("对", "对的", "是的", "没错", "说对了", "猜对了", "嗯", "嗯嗯", "对啊", "就是这样"):
        assert inference_mod.is_bare_confirmation(text) is True
        assert inference_mod.is_bare_denial(text) is False
    for text in ("不对", "不是", "没有", "错了", "猜错了", "你猜错了", "不对，不是这样"):
        assert inference_mod.is_bare_denial(text) is True
        assert inference_mod.is_bare_confirmation(text) is False
    assert inference_mod.is_bare_confirmation("对，我就是累了") is False
    assert inference_mod.is_bare_denial("不对，我没那个意思") is False
    assert inference_mod.is_bare_denial("嗯") is False


def test_classify_inference_reply():
    # 像人一样听懂 BOOS 的回话：整句和带后半句的都认，后半句的新信息留住。
    classify = inference_mod.classify_inference_reply
    # 确认：任务里列的说法。
    for text in ("对", "对的", "嗯", "是的", "没错", "就是这样", "猜对了"):
        assert classify(text) == {"verdict": "confirmed", "extra": ""}
    assert classify("对，我就是累了") == {"verdict": "confirmed", "extra": "我就是累了"}
    assert classify("嗯，明天早上再测") == {"verdict": "confirmed", "extra": "明天早上再测"}
    # 部分确认：算认了，标 partial。
    for text in ("差不多", "有点道理", "嗯差不多", "对了一半"):
        assert classify(text) == {"verdict": "partial", "extra": ""}
    assert classify("差不多，不过也没那么累") == {
        "verdict": "partial",
        "extra": "不过也没那么累",
    }
    # 否认：整句和带后半句的都认，后半句的新信息留住。
    for text in ("不对", "不是", "错了", "你猜错了"):
        assert classify(text)["verdict"] == "denied"
    assert classify("不对，我是想明天再测") == {
        "verdict": "denied",
        "extra": "我是想明天再测",
    }
    assert classify("不对，我没那个意思") == {
        "verdict": "denied",
        "extra": "我没那个意思",
    }
    # 看不出来：直接换了话题，既没认也没否——留待定，让模型再判。
    assert classify("明天早上再测") == {"verdict": "unclear", "extra": ""}
    assert classify("") == {"verdict": "unclear", "extra": ""}
    # “对了一半”不能拆成“对”+“了一半”。
    assert classify("对了一半")["verdict"] == "partial"


def test_parse_reply_verdict():
    # 模型判回话的两行输出：第一行四词里的一个，第二行是新补充的信息。
    parse = inference_mod.parse_reply_verdict
    assert parse("确认\n无") == {"verdict": "confirmed", "extra": ""}
    assert parse("部分\n对了一半吧") == {"verdict": "partial", "extra": "对了一半吧"}
    assert parse("否认\n我是想明天再测") == {"verdict": "denied", "extra": "我是想明天再测"}
    assert parse("不清楚\n无") == {"verdict": "unclear", "extra": ""}
    assert parse("没什么可猜的。") == {"verdict": "unclear", "extra": ""}
    assert parse("") == {"verdict": "unclear", "extra": ""}
    assert parse("确认") == {"verdict": "confirmed", "extra": ""}


# ---------- 2. 记/确认/否认：账本是推断的家 ----------


def test_record_inference_validates_and_dedupes_pending(tmp_path):
    store = make_fake_store(tmp_path)
    assert store.record_inference("", "凭据", "low", "induction") is None
    assert store.record_inference("猜", "", "low", "induction") is None
    assert store.record_inference("猜", "凭据", "sure", "induction") is None
    assert store.record_inference("猜", "凭据", "low", "瞎猜") is None
    recorded = store.record_inference(
        "BOOS今晚累了想歇会儿", "他说今晚好累", "low", "induction"
    )
    assert recorded is not None
    assert recorded["status"] == "pending"
    assert recorded["mode"] == "spot"
    # 同一句已经待定了就不重复记——问一次就够了。
    again = store.record_inference(
        "BOOS今晚累了想歇会儿", "他说今晚好累", "low", "induction"
    )
    assert again["key"] == recorded["key"]
    assert len(store.list_pending_inferences()) == 1
    # 账本里就是一笔 inference 事件。
    events = _ledger_events(store)
    assert [e for e in events if e["type"] == "inference"]


def test_infer_confirm_flow_stays_ledger_only(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", llm)
    turn1 = [
        {"role": "user", "content": "我今晚好累，明天再说吧"},
        {"role": "assistant", "content": "好，明天再说。"},
    ]
    result = store.infer_from_turn(turn1)
    inference = result.get("inference")
    assert inference is not None
    assert inference["status"] == "pending"
    assert inference["guess"] == "BOOS今晚累了想歇会儿"
    assert llm.calls == 1
    # 下一轮 BOOS 光秃秃地回“对”：记下。
    resolved = store.infer_from_turn([{"role": "user", "content": "对"}])
    assert resolved.get("resolved", {}).get("result") == "confirmed"
    assert resolved["resolved"]["status"] == "confirmed"
    assert store.list_pending_inferences() == []
    # 确认过的推断只在账本里，不进 mem0——mem0 里永远是“模型猜的”和
    # “BOOS 亲口说的”分得清清楚楚。
    assert not any("累了想歇会儿" in text for text in _texts(store))
    assert store.inference_feedback_summary() == {"induction": {"confirmed": 1, "denied": 0}}


def test_infer_deny_flow_deletes(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", llm)
    turn1 = [
        {"role": "user", "content": "我今晚好累，明天再说吧"},
        {"role": "assistant", "content": "好，明天再说。"},
    ]
    assert store.infer_from_turn(turn1).get("inference") is not None
    # 下一轮 BOOS 光秃秃地回“不对”：删除（不记作废）。
    resolved = store.infer_from_turn([{"role": "user", "content": "不对"}])
    assert resolved.get("resolved", {}).get("result") == "denied"
    assert resolved["resolved"]["status"] == "deleted"
    assert store.list_pending_inferences() == []
    assert store.inference_feedback_summary() == {"induction": {"confirmed": 0, "denied": 1}}
    # 否认的推断也不进 mem0。
    assert not any("累了想歇会儿" in text for text in _texts(store))


def test_answer_with_extra_resolves_and_keeps_extra(tmp_path, monkeypatch):
    # 带后半句的回答也认（2026-11-17 改法：以前只认光秃秃的“对/不对”，
    # “对，我就是累了”这种不认，推断一直挂着——现在认了，后半句的新
    # 信息留在 resolved["extra"] 里）。
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", llm)
    turn1 = [
        {"role": "user", "content": "我今晚好累，明天再说吧"},
        {"role": "assistant", "content": "好，明天再说。"},
    ]
    assert store.infer_from_turn(turn1).get("inference") is not None
    result = store.infer_from_turn([{"role": "user", "content": "对，我就是累了"}])
    assert result.get("resolved", {}).get("result") == "confirmed"
    assert result["resolved"]["extra"] == "我就是累了"
    assert store.list_pending_inferences() == []
    assert llm.calls == 1  # 词表一眼看清，没多调模型


def test_denial_with_extra_keeps_new_info(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", llm)
    turn1 = [
        {"role": "user", "content": "我今晚好累，明天再说吧"},
        {"role": "assistant", "content": "好，明天再说。"},
    ]
    assert store.infer_from_turn(turn1).get("inference") is not None
    # “不对，我是想明天再测”：否认 + 后半句的新信息留住。
    result = store.infer_from_turn([{"role": "user", "content": "不对，我是想明天再测"}])
    assert result.get("resolved", {}).get("result") == "denied"
    assert result["resolved"]["extra"] == "我是想明天再测"
    assert store.list_pending_inferences() == []
    assert llm.calls == 1


def test_unclear_reply_stays_pending(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", llm)
    turn1 = [
        {"role": "user", "content": "我今晚好累，明天再说吧"},
        {"role": "assistant", "content": "好，明天再说。"},
    ]
    assert store.infer_from_turn(turn1).get("inference") is not None
    # “明天早上再测”既没认也没否：词表看不出来，模型（剧本用完回
    # “没什么可猜的”）也判不出来——推断还待定着，不认、不删。
    result = store.infer_from_turn([{"role": "user", "content": "明天早上再测"}])
    assert "resolved" not in result
    assert len(store.list_pending_inferences()) == 1


def test_llm_classifies_fuzzy_reply(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(_spot_script() + ["否认\n我是想明天再测"])
    monkeypatch.setattr(store, "_call_llm_text", llm)
    turn1 = [
        {"role": "user", "content": "我今晚好累，明天再说吧"},
        {"role": "assistant", "content": "好，明天再说。"},
    ]
    assert store.infer_from_turn(turn1).get("inference") is not None
    # “你想多了”词表一眼看不出来，模型判了“否认”+ 新信息——否认 + 留住。
    result = store.infer_from_turn([{"role": "user", "content": "你想多了"}])
    assert result.get("resolved", {}).get("result") == "denied"
    assert result["resolved"]["extra"] == "我是想明天再测"
    assert store.list_pending_inferences() == []
    assert llm.calls == 2  # 当场揣 1 次 + 模型判回话 1 次


def test_partial_confirmation_marks_partial(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", llm)
    turn1 = [
        {"role": "user", "content": "我今晚好累，明天再说吧"},
        {"role": "assistant", "content": "好，明天再说。"},
    ]
    assert store.infer_from_turn(turn1).get("inference") is not None
    # “差不多”：算认了，状态 confirmed，但档案里标 partial。
    result = store.infer_from_turn([{"role": "user", "content": "差不多"}])
    resolved = result.get("resolved", {})
    assert resolved.get("result") == "confirmed"
    assert resolved.get("partial") is True
    assert inference_mod.render_inference_resolved(resolved) == "记下了，对了一半。"
    assert store.list_pending_inferences() == []
    records = store.list_inferences(status="confirmed")
    assert len(records) == 1
    assert records[0]["partial"] is True
    # 压完快照、重读账本，partial 标记还在。
    store.compact_ledger()
    after = store.list_inferences(status="confirmed")
    assert len(after) == 1 and after[0]["partial"] is True


def test_inference_calibration_summary(tmp_path):
    # 校准记账：按模型自己说的把握数确认/否认/待定——以后看 high 那档
    # 否认多还是确认多，就知道它是不是经常过分自信。
    store = make_fake_store(tmp_path)
    high_ok = store.record_inference("BOOS想换手机", "他总抱怨慢", "high", "deduction")
    high_bad = store.record_inference("BOOS想去跑步", "他说过要运动", "high", "induction")
    low_ok = store.record_inference("BOOS今晚累了", "他说好累", "low", "induction")
    store.record_inference("BOOS想吃拉面", "他说饿了", "medium", "deduction")
    store.confirm_inference(high_ok["key"])
    store.deny_inference(high_bad["key"])
    store.confirm_inference(low_ok["key"])
    summary = store.inference_calibration_summary()
    # high 那档 1 确认 1 否认——把握说得很大，猜得却一半错，过分自信的苗头。
    assert summary["high"] == {"confirmed": 1, "denied": 1, "pending": 0}
    assert summary["low"] == {"confirmed": 1, "denied": 0, "pending": 0}
    assert summary["medium"] == {"confirmed": 0, "denied": 0, "pending": 1}


# ---------- 校准自动调门槛（2026-11-17：只记账 → 能自己调） ----------


def _record_decided(store, confidence, count, confirm=True, prefix="猜"):
    """记 count 条某把握的推断，再全确认或全否认（每条猜的内容不同，
    不触发待定去重）。"""
    for i in range(count):
        recorded = store.record_inference(
            f"BOOS{prefix}{confidence}{i}", "他说过", confidence, "deduction"
        )
        assert recorded is not None
        if confirm:
            assert store.confirm_inference(recorded["key"]) is not None
        else:
            assert store.deny_inference(recorded["key"]) is not None


def test_auto_adjust_raises_threshold_when_high_overconfident(tmp_path):
    # high 档 5 条全否认（过分自信）→ 门槛从 low 往上提一档到 medium，
    # 只提一档（不直接跳到 high）；再调一次也不动（同一批反馈只看一次）。
    store = make_fake_store(tmp_path)
    assert store._inference_threshold == "low"
    _record_decided(store, "high", 5, confirm=False)
    assert store._inference_threshold == "medium"
    again = store.auto_adjust_inference_threshold()
    assert again == {
        "threshold": "medium",
        "changed": False,
        "reason": "no_new_feedback",
    }


def test_auto_adjust_lowers_threshold_when_low_too_conservative(tmp_path):
    # 先把门槛抬到 medium（high 档 5 否认），再来 5 条 low 确认
    # （太保守）→ 门槛放回 low。旧账不重复算：抬门槛那 5 条否认不会
    # 在新一批 low 确认来的时候又把门槛顶回去。
    store = make_fake_store(tmp_path)
    _record_decided(store, "high", 5, confirm=False)
    assert store._inference_threshold == "medium"
    _record_decided(store, "low", 5, confirm=True)
    assert store._inference_threshold == "low"


def test_auto_adjust_skips_when_not_enough_samples(tmp_path):
    # high 档只攒了 4 条否认（没够 5 条）→ 门槛不动；再来 1 条否认
    # 攒够 5 条 → 提一档。
    store = make_fake_store(tmp_path)
    _record_decided(store, "high", 4, confirm=False)
    assert store._inference_threshold == "low"
    result = store.auto_adjust_inference_threshold()
    assert result["changed"] is False
    _record_decided(store, "high", 1, confirm=False, prefix="补")
    assert store._inference_threshold == "medium"


def test_auto_adjust_skips_close_call(tmp_path):
    # high 档 3 否认 2 确认（胶着，领先只有 1 条）→ 不动；再来 1 条
    # 否认，变成 4 否认 2 确认（领先 2 条）→ 提一档。
    store = make_fake_store(tmp_path)
    _record_decided(store, "high", 3, confirm=False)
    _record_decided(store, "high", 2, confirm=True)
    assert store._inference_threshold == "low"
    _record_decided(store, "high", 2, confirm=False, prefix="补")
    assert store._inference_threshold == "medium"


def test_auto_adjust_conflicting_signals_stay_put(tmp_path, monkeypatch):
    # 同一批新反馈里 high 档 5 否认、low 档 5 确认（两边喊反方向）→ 不动。
    # 调门槛的触发在每次确认/否认后跑，这里先把它按住，一批记完再手动调，
    # 这样两边的信号才落在同一批里。
    store = make_fake_store(tmp_path)
    held = {"threshold": "low", "changed": False, "reason": "held"}
    orig = store.auto_adjust_inference_threshold
    monkeypatch.setattr(store, "auto_adjust_inference_threshold", lambda: held)
    _record_decided(store, "high", 5, confirm=False)
    _record_decided(store, "low", 5, confirm=True)
    monkeypatch.setattr(store, "auto_adjust_inference_threshold", orig)
    result = store.auto_adjust_inference_threshold()
    assert result["changed"] is False
    assert result["reason"] == "conflicting_signals"
    assert store._inference_threshold == "low"


def test_threshold_bounds_at_high_and_low(tmp_path):
    # 上限：high 档连续否认，门槛最多到 high，不会再往上；下限：low 档
    # 连续确认，门槛到 low 就停，不会再往下。
    store = make_fake_store(tmp_path)
    _record_decided(store, "high", 5, confirm=False)
    assert store._inference_threshold == "medium"
    _record_decided(store, "high", 5, confirm=False, prefix="再")
    assert store._inference_threshold == "high"
    _record_decided(store, "high", 5, confirm=False, prefix="又")
    assert store._inference_threshold == "high"
    _record_decided(store, "low", 5, confirm=True)
    assert store._inference_threshold == "medium"
    _record_decided(store, "low", 5, confirm=True, prefix="再")
    assert store._inference_threshold == "low"
    _record_decided(store, "low", 5, confirm=True, prefix="又")
    assert store._inference_threshold == "low"


def test_threshold_persists_across_restart(tmp_path):
    # 门槛存在 inference_calibration.json 里：新起一个 store（同一份
    # 账本目录），门槛还是 medium，不是从默认 low 重来。
    store = make_fake_store(tmp_path)
    _record_decided(store, "high", 5, confirm=False)
    assert store._inference_threshold == "medium"
    store2 = make_fake_store(tmp_path)
    assert store2._inference_threshold == "medium"


def test_threshold_gates_low_confidence_guess(tmp_path, monkeypatch):
    # 门槛 medium 时，low 把握的猜不问、不记；medium 把握的照常问。
    store = make_fake_store(tmp_path)
    _record_decided(store, "high", 5, confirm=False)
    assert store._inference_threshold == "medium"
    low_llm = ScriptedInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", low_llm)
    assert store._run_spot_inference(
        [{"role": "user", "content": "今天好累"}]
    ) is None
    assert store.list_pending_inferences() == []
    medium_script = [
        "我猜BOOS今晚累了想歇会儿，因为他说今晚好累、明天再说。"
        "我比较确定。（归纳）"
    ]
    medium_llm = ScriptedInferenceLLM(medium_script)
    monkeypatch.setattr(store, "_call_llm_text", medium_llm)
    result = store._run_spot_inference(
        [{"role": "user", "content": "今天好累"}]
    )
    assert result is not None
    assert result["confidence"] == "medium"
    assert result["status"] == "pending"


class CapturingInferenceLLM(ScriptedInferenceLLM):
    """按剧本回话、同时把最后一次的提示词记下来（看提示词里带了什么）。"""

    def __init__(self, script=None):
        super().__init__(script)
        self.last_prompt = ""

    def __call__(self, system_prompt, user_prompt, max_tokens=500):
        self.last_prompt = user_prompt
        return super().__call__(system_prompt, user_prompt, max_tokens=max_tokens)


def test_spot_prompt_includes_confirmed_inferences(tmp_path, monkeypatch):
    # 当场揣摩的提示词里带上以前确认过的推断（从账本里翻，不进检索）：
    # 像十一平时揣 BOOS 心思那样，从对他的了解出发，别凭空猜。
    store = make_fake_store(tmp_path)
    recorded = store.record_inference(
        "BOOS累了想歇会儿", "他说过想早点睡", "low", "induction"
    )
    store.confirm_inference(recorded["key"])
    llm = CapturingInferenceLLM(_spot_script())
    monkeypatch.setattr(store, "_call_llm_text", llm)
    result = store.infer_from_turn(
        [
            {"role": "user", "content": "我今晚好累，明天再说吧"},
            {"role": "assistant", "content": "好，明天再说。"},
        ]
    )
    assert result.get("inference") is not None
    assert "以前确认过的想法" in llm.last_prompt
    assert "BOOS累了想歇会儿" in llm.last_prompt
    assert "从对他的了解出发" in llm.last_prompt


def test_no_llm_means_no_inference_no_crash(tmp_path):
    # FakeMemory 没有 llm 客户端（和没接模型的真机一样）：_call_llm_text
    # 返回 None，infer_from_turn 返回 {}，不崩、不揣。
    store = make_fake_store(tmp_path)
    result = store.infer_from_turn(
        [
            {"role": "user", "content": "我今天很累"},
            {"role": "assistant", "content": "歇会儿吧。"},
        ]
    )
    assert result == {}
    assert store.list_pending_inferences() == []


def test_contradiction_layer_recorded(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    llm = ScriptedInferenceLLM(
        [
            "我猜BOOS其实不想去，因为他嘴上说想去、行动上总在拖，对不上。"
            "我只是猜的。（矛盾）"
        ]
    )
    monkeypatch.setattr(store, "_call_llm_text", llm)
    result = store.infer_from_turn(
        [
            {"role": "user", "content": "明天那个会啊，我去吧……算了再说"},
            {"role": "assistant", "content": "行，你定。"},
        ]
    )
    inference = result.get("inference")
    assert inference is not None
    assert inference["layer"] == "contradiction"
    # 矛盾层两边都留着、不站队：推断只记一句，证据里两边都在。
    assert "想去" in inference["evidence"] and "拖" in inference["evidence"]


# ---------- 3. 长期模式：攒够 20 条新记忆才跑 ----------


def test_longterm_triggers_at_20_and_does_not_retrigger(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    for i in range(20):
        store.add_memory_text(f"BOOS 第{i + 1}天都去跑步", source="user")
    assert store.count_new_memories_since_longterm() == 20
    llm = ScriptedInferenceLLM(
        ["我猜BOOS最近在坚持运动，因为最近20条记忆里总在说跑步。我只是猜的。（归纳）"]
    )
    monkeypatch.setattr(store, "_call_llm_text", llm)
    result = store.infer_from_turn(
        [
            {"role": "user", "content": "今天也跑了"},
            {"role": "assistant", "content": "加油。"},
        ]
    )
    inference = result.get("inference")
    assert inference is not None
    assert inference["mode"] == "longterm"
    assert llm.calls == 1
    # 长期揣摩本身就是标记：再跑一轮，没攒够新的 20 条就不重跑。
    assert store.count_new_memories_since_longterm() == 0
    result2 = store.infer_from_turn(
        [
            {"role": "user", "content": "明天继续"},
            {"role": "assistant", "content": "好。"},
        ]
    )
    assert "inference" not in result2
    longterms = [
        item for item in store.list_inferences() if item["mode"] == "longterm"
    ]
    assert len(longterms) == 1


def test_longterm_no_guess_sets_skip_line(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    for i in range(20):
        store.add_memory_text(f"BOOS 第{i + 1}天都去跑步", source="user")
    llm = ScriptedInferenceLLM(["没什么可猜的。"])
    monkeypatch.setattr(store, "_call_llm_text", llm)
    # 第一轮：长期试了一次（没什么可猜的，解析失败又重试一次），
    # 又当场试了一次（同样重试一次）——一共 4 次调用，都没揣出来。
    assert store.infer_from_turn(
        [
            {"role": "user", "content": "今天也跑了"},
            {"role": "assistant", "content": "加油。"},
        ]
    ) == {}
    assert llm.calls == 4
    # 跳过线记下来了：第二轮不再为长期多调一次模型，只当场试一次（2 次调用）。
    assert store.count_new_memories_since_longterm() == 0
    assert store.infer_from_turn(
        [
            {"role": "user", "content": "明天继续"},
            {"role": "assistant", "content": "好。"},
        ]
    ) == {}
    assert llm.calls == 6


# ---------- 4. 账本：推断进快照，压完还在 ----------


def test_inference_survives_snapshot(tmp_path):
    store = make_fake_store(tmp_path)
    recorded = store.record_inference(
        "BOOS今晚累了想歇会儿", "他说今晚好累", "low", "induction"
    )
    store.confirm_inference(recorded["key"])
    before = store.ledger_state()
    assert before["inferences"][recorded["key"]]["status"] == "confirmed"
    store.compact_ledger()
    after = store.ledger_state()
    assert after["inferences"][recorded["key"]]["status"] == "confirmed"
    # 账本原始形态里猜的内容叫 "text"（guess 是 list_inferences 对外
    # 用的名字，见 _inference_record_to_dict）。
    assert after["inferences"][recorded["key"]]["text"] == "BOOS今晚累了想歇会儿"


def test_confirm_unknown_key_is_none(tmp_path):
    store = make_fake_store(tmp_path)
    assert store.confirm_inference("no-such-key") is None
    assert store.deny_inference("no-such-key") is None


# ---------- 5. 提取守卫：问句里的猜，别存进库里 ----------


class QuestionExtractingFakeMemory(FakeMemory):
    """FakeMemory + 真实 mem0 的一个坏毛病：会把助手问句里的猜当成
    用户事实提出来（2026-11-16 真机：助手复述的“不再爱吃冒菜”被存
    成一条记忆）。只认“我猜你……，因为……”这个形状——问句就是
    render_inference_question 拼出来的，形状固定。"""

    def add(self, messages, user_id=None, infer=True, metadata=None):
        result = super().add(messages, user_id=user_id, infer=infer, metadata=metadata)
        if not infer:
            return result
        extra = []
        with self._lock:
            for message in messages or []:
                if (message or {}).get("role") != "assistant":
                    continue
                content = (message or {}).get("content") or ""
                for sentence in re.split(r"[。！？\n]+", content):
                    sentence = sentence.strip()
                    if not sentence.startswith("我猜你"):
                        continue
                    guess = sentence[len("我猜你"):].split("，因为")[0].strip()
                    if not guess:
                        continue
                    mem_id = self._store(guess, metadata=metadata)
                    extra.append({"id": mem_id, "memory": guess, "event": "ADD"})
        return {"results": list((result or {}).get("results") or []) + extra}


def test_extraction_guard_drops_asked_guess(tmp_path):
    store = make_fake_store(
        tmp_path, memory=QuestionExtractingFakeMemory()
    )
    store.record_inference("BOOS现在累了想歇会儿", "他说喜欢晚上早点睡", "low", "induction")
    question = inference_mod.render_inference_question(
        {
            "guess": "BOOS现在累了想歇会儿",
            "evidence": "他说喜欢晚上早点睡",
            "confidence": "low",
            "confidence_text": "我只是猜的",
            "layer": "induction",
        }
    )
    # 本轮问句里有这句猜：提取时问句里的猜拦掉，BOOS 自己说的事实留下。
    result = store._extract_and_save_turn(
        [
            {"role": "user", "content": "我喜欢晚上早点睡。"},
            {"role": "assistant", "content": "早点睡好啊。\n" + question},
        ]
    )
    kept = [item.get("memory") or "" for item in (result or {}).get("results") or []]
    assert any("喜欢晚上早点睡" in text for text in kept)
    assert not any("累了想歇会儿" in text for text in kept)
    assert (result or {}).get("inference_dropped") == 1
    assert not any("累了想歇会儿" in text for text in _texts(store))
    # BOOS 以后自己重新说起同一句（问句不在本轮里）：提取照常存。
    result2 = store._extract_and_save_turn(
        [{"role": "user", "content": "我现在累了想歇会儿。"}]
    )
    kept2 = [item.get("memory") or "" for item in (result2 or {}).get("results") or []]
    assert any("现在累了想歇会儿" in text for text in kept2)


def test_asked_guesses_only_when_question_present(tmp_path):
    store = make_fake_store(tmp_path)
    store.record_inference("BOOS现在累了想歇会儿", "他说喜欢晚上早点睡", "low", "induction")
    question = inference_mod.render_inference_question(
        {
            "guess": "BOOS现在累了想歇会儿",
            "evidence": "他说喜欢晚上早点睡",
            "confidence": "low",
            "confidence_text": "我只是猜的",
            "layer": "induction",
        }
    )
    asked = store._asked_inference_guesses(
        [{"role": "assistant", "content": "早点睡好啊。\n" + question}]
    )
    assert "现在累了想歇会儿" in asked
    # 问句不在本轮里时不拦——哪怕猜的内容和用户刚说的话一模一样。
    assert store._asked_inference_guesses(
        [{"role": "user", "content": "我现在累了想歇会儿。"}]
    ) == set()


# ---------- 6. 批量删路由：说明第一句就写死 + 一张卡删三条 ----------


def test_delete_tool_descriptions_route_first():
    register_memory_tools()
    delete_one = get_tool("delete_memory")["description"]
    assert delete_one.startswith("一次只删一条记忆。")
    assert "要删 2 条及以上时别调这个，调 delete_memories" in delete_one
    delete_many = get_tool("delete_memories")["description"]
    assert delete_many.startswith("一次删 2 条及以上记忆时调这个；只删一条时调 delete_memory。")


class ScriptedProvider:
    """按剧本回话的假模型：chat_with_tools 按 script 顺序回，chat_messages 回固定回答。"""

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
                      thinking=None, model=None, max_tokens=None):
        return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}


class _NoSearchService:
    def search(self, *args, **kwargs):
        raise AssertionError("这批测试不该调搜索")


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


@pytest.fixture
def mem_store(tmp_path):
    store = make_fake_store(tmp_path)
    store.add_memory_text("BOOS 爱吃拉面。", source="user")
    store.add_memory_text("BOOS 爱吃冒菜。", source="user")
    store.add_memory_text("BOOS 爱吃麻辣烫。", source="user")
    return store


@pytest.fixture
def task_store(tmp_path):
    return TaskStore(file_path=str(tmp_path / "tasks.json"))


@pytest.fixture
def gateway(tmp_path):
    return ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))


def _make_loop(provider, gateway, store):
    register_memory_tools()
    return AgentLoop(
        provider,
        gateway=gateway,
        task_store=store,
        search_service=_NoSearchService(),
    )


def test_batch_delete_three_items_one_approval_card(mem_store, gateway, task_store):
    """2026-11-16 10:08 真机重演：三条一起删，只弹一张卡、点一次确认就全删。
    锚点“爱吃”三条原文里都有（和真机那次三条都是“爱吃……”一个形状）。"""
    items = [
        {"memory_id": _mem_id(mem_store, text), "text": text}
        for text in ("BOOS 爱吃拉面。", "BOOS 爱吃冒菜。", "BOOS 爱吃麻辣烫。")
    ]
    provider = ScriptedProvider(
        script=[
            _tool_response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "delete_memories",
                        {"items": items, "anchor_text": "爱吃"},
                    )
                ]
            ),
            _tool_response(text="删掉了。"),
        ]
    )
    loop = _make_loop(provider, gateway, task_store)
    with bound_memory_store(mem_store):
        outcome = loop.run("把拉面、冒菜、麻辣烫三条记忆一起删掉")
    # 只调了一次工具、只停在一张卡上，库里还没删。
    assert len(provider.tool_calls_log) == 1
    assert outcome["mode"] == "waiting_approval"
    assert outcome["approval"]["tool"] == "delete_memories"
    assert outcome["approval"]["approval_id"]
    assert len(_texts(mem_store)) == 3
    task = outcome["task"]
    # BOOS 点一次确认，三条全删。
    with bound_memory_store(mem_store):
        resolved = gateway.resolve_approval(
            task_store, task["task_id"], outcome["approval"]["approval_id"], "confirm"
        )
    assert resolved["decision"] == "ok"
    assert resolved["result"]["ok"] is True
    assert _texts(mem_store) == []
    # 恢复循环，模型回答。
    with bound_memory_store(mem_store):
        resumed = loop.run(
            "把拉面、冒菜、麻辣烫三条记忆一起删掉",
            resume_task=resolved["task"],
            pending_tool_result=resolved["result"],
        )
    assert resumed["mode"] == "done"
    assert resumed["reply"] == "删掉了。"
