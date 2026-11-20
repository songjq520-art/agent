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
