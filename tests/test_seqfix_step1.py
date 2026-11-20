# tests/test_seqfix_step1.py（第一步修复 6 个 Bug 的回归测试，2026-11-20）
# 每个 Bug 修完补在这里；全部离线（FakeMemory），不碰真实 mem0、不联网。
# 依据：~/workspace/十二记忆代码审查核实-2026-11-20.md。

from conftest import make_fake_store


# ---------- Bug 1：作废的玩笑记忆不再被检索注入 ----------


def test_bug1_voided_memory_not_searchable(tmp_path):
    """Bug 1：作废的玩笑记忆不再被检索注入——语义、关键词、账本兜底
    三路都拦。修之前 search 会把它当成事实带回来。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我喜欢喝可乐", source="user")
    mem_id = (result.get("results") or [{}])[0].get("id")
    assert mem_id
    assert store.void_joke_memory(mem_id)["ok"] is True
    found = store.search("我喜欢喝什么")
    assert not any("可乐" in item["text"] for item in found), found
    # 问“以前”也不放出来：作废不是历史。
    found_history = store.search("我喜欢喝什么", include_history=True)
    assert not any("可乐" in item["text"] for item in found_history)


def test_bug1_project_and_filter_drops_voided_even_with_history():
    """Bug 1（筛子层）：project_and_filter 永远丢掉 voided 的项，
    include_history=True 也不放出来；正常的照常放行。"""
    from memory.ledger import project_and_filter

    state = {"by_id": {}, "key_states": {}}
    voided = [{"id": "m1", "text": "玩笑话", "metadata": {"voided": True}}]
    assert project_and_filter(voided, state) == []
    assert project_and_filter(voided, state, include_history=True) == []
    normal = [{"id": "m2", "text": "正常记忆", "metadata": {}}]
    out = project_and_filter(normal, state)
    assert len(out) == 1
    assert out[0]["is_history"] is False


# ---------- Bug 5：撤回要删掉 outbox 落盘文件 ----------


def _bug5_parked_turn(store, tmp_path, turn_id="turn-1"):
    """先让一轮经 outbox 写盘、再进待定区（文件还留着）：复现撤回前
    的状态。返回 (outbox, pending_dir)。"""
    from memory.outbox import MemoryOutbox

    pending_dir = tmp_path / "outbox_pending"
    outbox = MemoryOutbox(store, pending_dir=pending_dir)
    store._outbox = outbox
    messages = [
        {"role": "user", "content": "我喜欢吃拉面。"},
        {"role": "assistant", "content": "记下了。"},
    ]
    outbox.submit(messages, turn_id=turn_id)
    assert (pending_dir / f"{turn_id}.json").exists()
    # 后台存一遍：只进待定区（pending），文件还留着（没真正落库）。
    outbox._save_one(turn_id, messages)
    assert outbox.turn_status(turn_id) == "pending"
    assert (pending_dir / f"{turn_id}.json").exists()
    return outbox, pending_dir


def test_bug5_bare_retraction_drops_outbox_file(tmp_path):
    """Bug 5：光秃秃撤回（“不不不。”）时，待定轮的 outbox 落盘文件
    必须删掉——否则重启后 recover_pending 会把它捞回来重存，
    已撤回的内容就被悄悄存回去了。回执标 dropped，不是 committed。"""
    from memory.outbox import MemoryOutbox

    store = make_fake_store(tmp_path)
    outbox, pending_dir = _bug5_parked_turn(store, tmp_path)
    result = store.add_turn([
        {"role": "user", "content": "不不不。"},
        {"role": "assistant", "content": "好的，不算。"},
    ])
    assert result.get("retracted") is True
    assert not (pending_dir / "turn-1.json").exists()
    assert outbox.turn_status("turn-1") == "dropped"
    # 重启后捞不回来：文件没了，recover_pending 不该再交出这一轮。
    outbox2 = MemoryOutbox(store, pending_dir=pending_dir)
    assert outbox2.recover_pending() == 0


def test_bug5_judged_retraction_drops_outbox_file(tmp_path):
    """Bug 5（模型判意图那路撤回）：“其实也不是很喜欢吃。”判撤回后，
    同样删盘、标 dropped。"""
    store = make_fake_store(tmp_path)
    # 只测“判撤回之后”的程序行为，判本身靠模型（同 test_plan_c）。
    store._judge_retraction_intent = lambda *a, **k: True
    outbox, pending_dir = _bug5_parked_turn(store, tmp_path)
    result = store.add_turn([
        {"role": "user", "content": "其实也不是很喜欢吃。"},
        {"role": "assistant", "content": "好的。"},
    ])
    assert result.get("retracted") is True
    assert not (pending_dir / "turn-1.json").exists()
    assert outbox.turn_status("turn-1") == "dropped"


# ---------- Bug 2：keep_old 分支（旧条赢，新条当证据，不并存） ----------


def _bug2_old_money_memory(store):
    """种一条同槽位旧金额记忆：推断来源、写序号故意给得很大（比新批
    的还晚），让 _resolve_conflict 判 keep_old。返回旧条 id。"""
    from memory.mem0_store import (
        _provenance,
        CONFIDENCE_INFERRED,
        SOURCE_INFERRED,
    )

    old_meta = _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    old_meta["write_seq"] = 10 ** 9
    result = store._add_text_with_metadata("每天定投5块买万家纳指", old_meta)
    return (result.get("results") or [{}])[0].get("id")


def _bug2_evidence_texts(store, memory_id):
    import json

    raw = store._raw_metadata_for(memory_id)
    assert raw is not None
    evidence = raw.get("evidence") or ""
    if isinstance(evidence, str) and evidence:
        return [entry.get("text") for entry in json.loads(evidence)]
    return []


def test_bug2_dedupe_keep_old_appends_evidence(tmp_path):
    """Bug 2（提取路）：同槽位新金额判成 keep_old 时，旧条不动、新条
    当证据挂到旧条下、不并存两条。修之前新条照常留下来，新旧并存。"""
    store = make_fake_store(tmp_path)
    old_id = _bug2_old_money_memory(store)
    before = store.get_all()
    added = store._memory.add(
        [{"role": "user", "content": "每天定投10块买万家纳指"}],
        user_id=store.user_id,
        infer=False,
        metadata=None,
    )
    new_id = (added.get("results") or [{}])[0].get("id")
    out = store._dedupe_extraction_results(
        {"results": [{"id": new_id, "memory": "每天定投10块买万家纳指", "event": "ADD"}]},
        before,
        turn_id="t1",
    )
    assert out.get("money_kept_old") == 1
    assert store._memory.all_texts() == ["每天定投5块买万家纳指"]
    assert "每天定投10块买万家纳指" in _bug2_evidence_texts(store, old_id)


def test_bug2_add_memory_text_keep_old_appends_evidence(tmp_path):
    """Bug 2（手动存路）：同槽位新金额判成 keep_old 时，旧条不动、
    新说法当证据挂到旧条下、返回已有那条，不并存两条。"""
    store = make_fake_store(tmp_path)
    old_id = _bug2_old_money_memory(store)
    result = store.add_memory_text("每天定投10块买万家纳指", source="inferred")
    assert result.get("duplicate") is True
    assert (result.get("results") or [{}])[0].get("id") == old_id
    assert store._memory.all_texts() == ["每天定投5块买万家纳指"]
    assert "每天定投10块买万家纳指" in _bug2_evidence_texts(store, old_id)
