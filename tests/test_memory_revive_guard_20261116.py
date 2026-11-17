# tests/test_memory_revive_guard_20261116.py（2026-11-16 真机恢复测试的第二轮回归）
# 背景：BOOS 在笔记本上（新版 9cbdc909）测恢复：说“记住，我爱吃拉面”→
# 删 → 让十二恢复。十二先说“删掉了”，恢复时改口“没有真正删掉，现在还在
# 记忆里”。账本 memory_events.jsonl 定死：删（seq 11，01:18:42）在前、
# 记（seq 12，01:18:52）在后——删完 10 秒，提取又把同一条记回来了；
# 删掉那份的 key（70b4200b）≠ 后记那份的 key（4c09e828），且账本里从来
# 没有 33307618 的 add 记录；同一句话经显式记和提取各存一份，去重没并掉。
# 这 3 个根子 + 2 个小项都在这里钉死（见 ~/workspace/goals/agent-2/files/
# 今晚整合分支测试单.md 末尾“2026-11-16 真机测试结果”一节）。
# 运行：~/workspace/mem0-spike/.venv/bin/python -m pytest tests/test_memory_revive_guard_20261116.py -q

import pytest

from conftest import make_fake_store
from memory import ledger as _ledger
from memory.provenance import _provenance
from memory.redaction import _normalize_memory_text
from tools.memory_tool import (
    _check_delete_anchor,
    _run_delete_memory,
    _run_revive_memory,
    bound_memory_store,
)


# ---------- 辅助 ----------


def _add(store, text):
    """直存一条（source=user、已确认），返回新 id。"""
    result = store.add_memory_text(text, source="user")
    return ((result or {}).get("results") or [{}])[0].get("id")


def _add_raw(store, text):
    """绕过去重直接存一条（造“相同说法两条”或“账本落地前的老记忆”时用）。"""
    result = store._add_text_with_metadata(
        text, _provenance("user", 1.0, True)
    )
    return ((result or {}).get("results") or [{}])[0].get("id")


def _texts(store):
    return [item["text"] for item in store.get_all(include_sensitive=True)]


def _mem_id(store, text):
    for item in store.get_all(include_sensitive=True):
        if item["text"] == text:
            return item["id"]
    raise AssertionError(f"库里没有这条记忆：{text}")


def _ledger_events(store, event_type=None):
    events, _info = _ledger.read_events(store.ledger_path)
    if event_type is None:
        return events or []
    return [e for e in (events or []) if e.get("type") == event_type]


# ---------- 根子 1：提取把刚删的记忆又记回来 ----------


def test_deleted_text_is_not_reextracted_next_turn(tmp_path):
    """删完 10 秒，下一轮提交待定区时提取不再把同一条记回来；
    BOOS 以后重新说起时，照样能记回去（删在源轮次说话之前的不拦）。
    真机原样：第 1 轮“记住，我喜欢吃拉面。”→ 第 2 轮删掉 → 第 2 轮的
    add_turn 提交第 1 轮的待定区，提取把“喜欢吃拉面”又记回来了。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢吃拉面。")
    store.add_turn([
        {"role": "user", "content": "记住。我喜欢吃拉面。"},
        {"role": "assistant", "content": "记下了。"},
    ])
    assert store.delete_memory(mem_id, approve=True) is True
    assert _texts(store) == []
    # 第 2 轮：删掉——提交第 1 轮待定区时，提取不许再记回来。
    store.add_turn([
        {"role": "user", "content": "把'我喜欢吃拉面'这条记忆删掉"},
        {"role": "assistant", "content": "删掉了。"},
    ])
    assert _texts(store) == []
    # 重新说起：BOOS 又说一遍喜欢吃拉面，这回要能记回去。
    store.add_turn([
        {"role": "user", "content": "我喜欢吃拉面。"},
        {"role": "assistant", "content": "嗯。"},
    ])
    store.add_turn([
        {"role": "user", "content": "还有什么？"},
        {"role": "assistant", "content": "没了。"},
    ])
    assert any(
        _normalize_memory_text(t) == "喜欢吃拉面" for t in _texts(store)
    )


# ---------- 根子 2：显式记的记忆没写账本 add 记录 ----------


def test_explicit_add_ledgers_add_and_preledger_delete_is_revivable(tmp_path):
    """显式记写账本 add、删的 key 与记的 key 一致；账本落地之前存的
    老记忆（没有 add 记录）删时补记一笔 add，删了进得了已删列表、恢复得了。
    真机原样：账本里从来没有 33307618 的 add 记录，删的 key（70b4200b）
    对不上后记那份的 key（4c09e828），已删列表按 key 认、找不到它；
    早上“不再爱吃冒菜”那条删和记的 key 是一致的（29d093f9），能找到。"""
    store = make_fake_store(tmp_path)
    # 显式记：账本里有 add，删的 key 就是记的 key。
    mem_id = _add(store, "BOOS 爱吃冒菜。")
    adds = [
        e for e in _ledger_events(store, "add")
        if e.get("text") == "BOOS 爱吃冒菜。"
    ]
    assert len(adds) == 1
    assert store.delete_memory(mem_id, approve=True) is True
    deletes = [
        e for e in _ledger_events(store, "delete")
        if e.get("memory_id") == mem_id
    ]
    assert len(deletes) == 1
    assert deletes[0].get("key") == adds[0].get("key")
    assert any(d["text"] == "BOOS 爱吃冒菜。" for d in store.list_deleted_memories())
    assert isinstance(store.revive_memory(text="BOOS 爱吃冒菜。"), str)
    # 账本落地前的老记忆：没有 add 记录，删时补记，照样能恢复。
    _add_raw(store, "我爱吃火锅。")
    assert not [
        e for e in _ledger_events(store, "add") if e.get("text") == "我爱吃火锅。"
    ]
    old_id = _mem_id(store, "我爱吃火锅。")
    assert store.delete_memory(old_id, approve=True) is True
    assert any(d["text"] == "我爱吃火锅。" for d in store.list_deleted_memories())
    adds = [
        e for e in _ledger_events(store, "add")
        if e.get("text") == "我爱吃火锅。"
    ]
    deletes = [
        e for e in _ledger_events(store, "delete")
        if e.get("memory_id") == old_id
    ]
    assert len(adds) == 1 and len(deletes) == 1
    assert adds[0].get("key") == deletes[0].get("key")
    assert isinstance(store.revive_memory(text="我爱吃火锅。"), str)
    assert "我爱吃火锅。" in _texts(store)


# ---------- 根子 3：一句话两份、去重没并掉 ----------


def test_explicit_plus_extraction_same_sentence_keeps_one_copy(tmp_path):
    """同一句话，显式记是第一人称“我爱吃拉面。”，提取出来是第三人称
    “BOOS 爱吃拉面。”，只留一份——归一化剥掉“我/BOOS”主语后，
    去重认得出来。真机原样：33307618 与 ce549127 各存一份。"""
    # 归一化本身：主语不同算同一句；“我们/我家”主语不同，不算。
    assert _normalize_memory_text("我爱吃拉面。") == _normalize_memory_text(
        "BOOS 爱吃拉面。"
    )
    assert _normalize_memory_text("我爱吃拉面。") != _normalize_memory_text(
        "我们爱吃拉面。"
    )
    assert _normalize_memory_text("我爱吃拉面。") != _normalize_memory_text(
        "我家爱吃拉面。"
    )
    store = make_fake_store(tmp_path)
    _add(store, "我爱吃拉面。")

    # 模拟真实 mem0 的提取：把“我爱吃拉面”抽成第三人称“BOOS 爱吃拉面。”，
    # 并真实存进假库（走 FakeMemory._store），再按 mem0 的返回格式交回。
    def fake_mem0_add(messages, user_id=None, infer=None, metadata=None):
        new_id = store._memory._store("BOOS 爱吃拉面。", metadata or {})
        return {"results": [{"id": new_id, "memory": "BOOS 爱吃拉面。", "event": "ADD"}]}

    store._mem0_add = fake_mem0_add
    store.add_turn([
        {"role": "user", "content": "记住，我爱吃拉面。"},
        {"role": "assistant", "content": "记下了。"},
    ])
    store.flush_pending()
    assert _texts(store) == ["我爱吃拉面。"]


# ---------- 小项 4：单条删除只删指定的 id，不顺手清副本 ----------


def test_text_delete_removes_all_live_copies(tmp_path):
    """同一句话存了两份（显式记“我爱吃拉面。”+ 提取“BOOS 爱吃拉面。”），
    单条删除只删指定的那一个 id，另一份活着不动——“删重复、留一条”时，
    多删的那个正是该留下的（2026-11-17 火锅：留 00:29、删 00:41 结果
    两条都没了）。要删多份时用批量删，把每条 id 都列在确认卡上。"""
    store = make_fake_store(tmp_path)
    first = _add_raw(store, "我爱吃拉面。")
    second = _add_raw(store, "BOOS 爱吃拉面。")
    assert first != second
    with bound_memory_store(store):
        error, augment = _check_delete_anchor(
            {"memory_id": first, "text": "我爱吃拉面。", "anchor_text": "拉面"}
        )
        assert error is None
        note = (augment or {}).get("_anchor_match", "")
        assert "另外还有 1 条同一句话的副本" in note
        assert "BOOS 爱吃拉面。" in note
        result = _run_delete_memory(
            {"memory_id": first, "text": "我爱吃拉面。", "anchor_text": "拉面"}
        )
    assert result["ok"] is True
    assert result["data"]["extra_deleted"] == []
    assert _texts(store) == ["BOOS 爱吃拉面。"]


# ---------- 小项 5：恢复失败的两种原因分开说 ----------


def test_revive_failure_reasons_are_distinct(tmp_path):
    """恢复失败分两种说法，不合成一句让模型猜：“已删里没有它”和
    “它已经在库里了”。真机原样：十二恢复时改口“没有真正删掉，现在
    还在记忆里”——那条其实是删完又被提取记回来的，已在库里了。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "BOOS 爱吃拉面。")
    assert store.delete_memory(mem_id, approve=True) is True
    # 删完又记回来了（提取复活的活口）：恢复时要说“已经在库里了”。
    _add_raw(store, "BOOS 爱吃拉面。")
    with bound_memory_store(store):
        result = _run_revive_memory(
            {"memory_id": mem_id, "text": "BOOS 爱吃拉面。", "anchor_text": "拉面"}
        )
    assert result["ok"] is False
    assert "已经在库里了" in result["error"]
    assert "已删里没有" not in result["error"]
    # 已删里根本没有：恢复时要说“已删里没有这条”。
    with bound_memory_store(store):
        result = _run_revive_memory(
            {"memory_id": "no-such-id", "text": "BOOS 爱吃火锅。", "anchor_text": "火锅"}
        )
    assert result["ok"] is False
    assert "已删的记忆里没有这条" in result["error"]
    assert "已经在库里了" not in result["error"]
