# tests/test_memory_revive_batch.py（2026-11-16 真机测试的回归测试）
# 背景：2026-11-16 早上 BOOS 在笔记本上测整合分支，第 3 步真机验里
# 恢复没过、批量删要逐条点确认、长任务没进度、模型脑补了删除动机。
# 这四个问题 + tzdata 缺件都在这里钉死，修完重测恢复那一步、没问题
# 再合 main（见 ~/workspace/goals/agent-2/files/今晚整合分支测试单.md
# 末尾“2026-11-16 真机测试结果”一节）。
# 运行：~/workspace/mem0-spike/.venv/bin/python -m pytest tests/test_memory_revive_batch.py -q

from pathlib import Path

import pytest

from conftest import make_fake_store
from memory import ledger as _ledger
from memory.mem0_store import CUSTOM_INSTRUCTIONS, _describes_memory_management
from memory.provenance import _provenance
from tools.agent_loop import build_loop_system_message
from tools.memory_tool import (
    _check_batch_delete_anchor,
    _check_revive_anchor,
    _run_delete_memories,
    _run_list_deleted_memories,
    _run_revive_memory,
    bound_memory_store,
    register_memory_tools,
)
from tools.tool_registry import get_tool


# ---------- 辅助 ----------


def _add(store, text):
    """直存一条（source=user、已确认），返回新 id。"""
    result = store.add_memory_text(text, source="user")
    return ((result or {}).get("results") or [{}])[0].get("id")


def _add_raw(store, text):
    """绕过去重直接存一条（造“相同说法两条”时用）。"""
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


def _ledger_types(store):
    events, _info = _ledger.read_events(store.ledger_path)
    return [e.get("type") for e in events or []]


def _search_texts(store, query):
    return [(h or {}).get("text") or "" for h in store.search(query)]


# ---------- 问题 1：恢复工具缺失 ----------


def test_revive_deleted_memory_searchable_again_and_ledged(tmp_path):
    """删掉的记忆能恢复回来：恢复后搜得到、账本里有一笔 revive 事件、
    已删清单里不再有它——2026-11-16 真机测试第 4 步的原样复现
    （删冒菜 → 恢复 → 又能搜到）。"""
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    _add(store, "BOOS 爱吃火锅。")
    mem_id = _mem_id(store, "BOOS 爱吃冒菜。")
    assert store.delete_memory(mem_id, approve=True) is True
    assert "BOOS 爱吃冒菜。" not in _search_texts(store, "我爱吃冒菜")
    assert "BOOS 爱吃冒菜。" not in _texts(store)
    deleted = store.list_deleted_memories()
    assert any(d["text"] == "BOOS 爱吃冒菜。" for d in deleted)
    new_id = store.revive_memory(text="BOOS 爱吃冒菜。")
    assert isinstance(new_id, str) and new_id
    assert new_id != mem_id
    assert "BOOS 爱吃冒菜。" in _texts(store)
    assert "BOOS 爱吃冒菜。" in _search_texts(store, "我爱吃冒菜")
    assert "revive" in _ledger_types(store)
    assert not any(d["text"] == "BOOS 爱吃冒菜。" for d in store.list_deleted_memories())


def test_revive_by_old_id_keeps_original_provenance(tmp_path):
    """按旧 id 恢复：来源/确认状态/形成时间都按原样回来（不是新存一条
    推断记忆），账本里 revive 事件的 key 就是删掉时标 deleted 的那个 key。"""
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    mem_id = _mem_id(store, "BOOS 爱吃冒菜。")
    assert store.delete_memory(mem_id, approve=True) is True
    new_id = store.revive_memory(memory_id=mem_id)
    assert isinstance(new_id, str) and new_id
    revived = next(i for i in store.get_all(include_sensitive=True) if i["id"] == new_id)
    meta = revived.get("metadata") or {}
    assert meta.get("source") == "user"
    assert meta.get("user_confirmed") is True
    events, _info = _ledger.read_events(store.ledger_path)
    revive_events = [e for e in events if e.get("type") == "revive"]
    assert len(revive_events) == 1
    assert revive_events[0].get("memory_id") == new_id


def test_revive_refuses_unknown_already_back_and_wrong_text(tmp_path):
    """恢复不了的三种情况都返回 False，不谎称恢复了：已删里没有这条、
    原话已经在库里了、传来的原文和已删档案对不上。"""
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    assert store.revive_memory(text="BOOS 爱吃拉面。") is False
    assert store.revive_memory(memory_id="no-such-id") is False
    assert store.revive_memory(text="BOOS 爱吃冒菜。") is False
    mem_id = _mem_id(store, "BOOS 爱吃冒菜。")
    assert store.delete_memory(mem_id, approve=True) is True
    assert store.revive_memory(text="BOOS 爱吃拉面。") is False
    assert "BOOS 爱吃冒菜。" not in _texts(store)
    assert isinstance(store.revive_memory(text="BOOS 爱吃冒菜。"), str)
    assert store.revive_memory(text="BOOS 爱吃冒菜。") is False


def test_revive_soft_deleted_from_trash(tmp_path):
    """软删除（回收站）的也能恢复：走已有的 restore_memory 那条路，
    回收站里清掉、账本里照样记一笔 revive。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "BOOS 爱吃冒菜。")
    assert store.soft_delete_memory(mem_id) is True
    assert "BOOS 爱吃冒菜。" not in _texts(store)
    assert any(e["text"] == "BOOS 爱吃冒菜。" for e in store.list_retracted())
    new_id = store.revive_memory(text="BOOS 爱吃冒菜。")
    assert isinstance(new_id, str) and new_id
    assert "BOOS 爱吃冒菜。" in _texts(store)
    assert store.list_retracted() == []
    assert "revive" in _ledger_types(store)


def test_revive_tools_registered_and_end_to_end(tmp_path):
    """恢复工具已登记、走通全流程：列已删 → 锚点校验 → 恢复 → 搜得到。
    锚点不对（恢复目标和原文对不上）时拒绝，不建卡、不执行。"""
    register_memory_tools()
    assert get_tool("revive_memory") is not None
    assert get_tool("list_deleted_memories") is not None
    assert get_tool("delete_memories") is not None
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    _add(store, "BOOS 爱吃火锅。")
    mem_id = _mem_id(store, "BOOS 爱吃冒菜。")
    assert store.delete_memory(mem_id, approve=True) is True
    with bound_memory_store(store):
        listed = _run_list_deleted_memories({})
        assert listed["ok"] is True
        assert any(m["text"] == "BOOS 爱吃冒菜。" for m in listed["data"]["memories"])
        error, _aug = _check_revive_anchor(
            {"memory_id": mem_id, "text": "BOOS 爱吃冒菜。", "anchor_text": "拉面"}
        )
        assert error
        error, augment = _check_revive_anchor(
            {"memory_id": mem_id, "text": "BOOS 爱吃冒菜。", "anchor_text": "冒菜"}
        )
        assert error is None
        assert "BOOS 爱吃冒菜。" in (augment or {}).get("_anchor_match", "")
        result = _run_revive_memory(
            {"memory_id": mem_id, "text": "BOOS 爱吃冒菜。", "anchor_text": "冒菜"}
        )
        assert result["ok"] is True
        assert result["data"]["revived"] is True
        assert result["data"]["text"] == "BOOS 爱吃冒菜。"
    assert "BOOS 爱吃冒菜。" in _texts(store)
    assert "BOOS 爱吃冒菜。" in _search_texts(store, "我爱吃冒菜")


def test_revive_tool_refuses_wrong_text(tmp_path):
    """恢复工具：传来的原文和已删档案对不上时拒绝，什么都不恢复。"""
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    mem_id = _mem_id(store, "BOOS 爱吃冒菜。")
    assert store.delete_memory(mem_id, approve=True) is True
    with bound_memory_store(store):
        result = _run_revive_memory(
            {"memory_id": mem_id, "text": "BOOS 爱吃拉面。", "anchor_text": "冒菜"}
        )
        assert result["ok"] is False
    assert "BOOS 爱吃冒菜。" not in _texts(store)
    assert "BOOS 爱吃拉面。" not in _texts(store)


# ---------- 问题 2：批量删要逐条点确认 + “删了一半”误报 ----------


def test_delete_one_of_duplicate_texts_passes_verification(tmp_path):
    """“删了一半”的误报根因：库里有两条相同说法时，删一条后搜还回来
    另一条，旧的按文本验永远验不过。按 id 认之后，删一条就通过，
    另一条原封不动——2026-11-16 真机测试第一条“只清理了一半”
    就是这么误报的（当时库里两条“我爱吃冒菜”）。"""
    store = make_fake_store(tmp_path)
    first = _add_raw(store, "BOOS 爱吃冒菜。")
    second = _add_raw(store, "BOOS 爱吃冒菜。")
    assert first != second
    assert store.delete_memory(first, approve=True) is True
    assert _texts(store) == ["BOOS 爱吃冒菜。"]
    assert store.verify_forgotten(first, "BOOS 爱吃冒菜。")["ok"] is True
    assert store.delete_memory(second, approve=True) is True
    assert _texts(store) == []


def test_batch_delete_deletes_all_with_one_call(tmp_path):
    """批量删：一次调用删 3 条，火锅和无关的都还在；账本里 3 笔 delete 事件。
    确认卡的备注里 3 条原文全列出来（点一次确认之前看得清删的是哪几条）。
    注意：这里不用“我爱吃冒菜。”当第二条——2026-11-16 拉面根子 3 修完后
    它和“BOOS 爱吃冒菜。”归一化相同（剥掉“我/BOOS”主语），add 时会当
    重复并掉，根本存不进去；用“配米饭”那句意思才不同。"""
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    _add(store, "我爱吃冒菜配米饭。")
    _add(store, "我爱的是冒菜，不是麻辣烫。")
    _add(store, "BOOS 爱吃火锅。")
    _add(store, "BOOS 爱吃拉面。")
    items = [
        {"memory_id": _mem_id(store, t), "text": t}
        for t in ("BOOS 爱吃冒菜。", "我爱吃冒菜配米饭。", "我爱的是冒菜，不是麻辣烫。")
    ]
    with bound_memory_store(store):
        error, augment = _check_batch_delete_anchor(
            {"items": items, "anchor_text": "冒菜"}
        )
        assert error is None
        note = (augment or {}).get("_anchor_match", "")
        assert "确认一次全删" in note
        for t in ("BOOS 爱吃冒菜。", "我爱吃冒菜配米饭。", "我爱的是冒菜，不是麻辣烫。"):
            assert t in note
        result = _run_delete_memories({"items": items, "anchor_text": "冒菜"})
        assert result["ok"] is True
        assert len(result["data"]["deleted"]) == 3
        assert result["data"]["failed"] == []
    assert _texts(store) == ["BOOS 爱吃火锅。", "BOOS 爱吃拉面。"]
    assert _ledger_types(store).count("delete") == 3


def test_batch_delete_rejects_whole_batch_on_one_bad_item(tmp_path):
    """批量删：有一条挑错了（原文对不上），整批都不删——卡上是一次确认，
    验不过的条目绝不能混进这一批里。"""
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    _add(store, "我爱吃冒菜配米饭。")
    _add(store, "BOOS 爱吃火锅。")
    items = [
        {"memory_id": _mem_id(store, "BOOS 爱吃冒菜。"), "text": "BOOS 爱吃冒菜。"},
        {"memory_id": _mem_id(store, "我爱吃冒菜配米饭。"), "text": "BOOS 爱吃拉面。"},
    ]
    with bound_memory_store(store):
        result = _run_delete_memories({"items": items, "anchor_text": "冒菜"})
        assert result["ok"] is False
    assert _texts(store) == ["BOOS 爱吃冒菜。", "我爱吃冒菜配米饭。", "BOOS 爱吃火锅。"]
    assert "delete" not in _ledger_types(store)


def test_batch_delete_partial_failure_reported_honestly(tmp_path):
    """批量删：一半删成了、一半没删成时，不谎称全删完了——ok 是 False，
    data 里删成的和没删成的两份清单都在，错误里写清哪条没删成。"""
    store = make_fake_store(tmp_path)
    _add(store, "BOOS 爱吃冒菜。")
    _add(store, "我爱吃冒菜配米饭。")
    _add(store, "BOOS 爱吃火锅。")
    bad_id = _mem_id(store, "我爱吃冒菜配米饭。")
    good_id = _mem_id(store, "BOOS 爱吃冒菜。")
    real_delete = store.delete_memory

    def _flaky(memory_id, approve=False):
        if memory_id == bad_id:
            raise RuntimeError("模拟失败：向量删了，校验没过。")
        return real_delete(memory_id, approve=approve)

    store.delete_memory = _flaky
    items = [
        {"memory_id": good_id, "text": "BOOS 爱吃冒菜。"},
        {"memory_id": bad_id, "text": "我爱吃冒菜配米饭。"},
    ]
    try:
        with bound_memory_store(store):
            result = _run_delete_memories({"items": items, "anchor_text": "冒菜"})
    finally:
        store.delete_memory = real_delete
    assert result["ok"] is False
    assert len(result["data"]["deleted"]) == 1
    assert result["data"]["deleted"][0]["memory_id"] == good_id
    assert len(result["data"]["failed"]) == 1
    assert result["data"]["failed"][0]["memory_id"] == bad_id
    assert "我爱吃冒菜配米饭。" in result["error"]
    assert "BOOS 爱吃冒菜。" not in _texts(store)
    assert "我爱吃冒菜配米饭。" in _texts(store)


# ---------- 问题 3 + 4b：循环提示词里的两条新规矩 ----------


def test_system_message_multistep_rule_and_inference_guardrail():
    """第 14 条（多步任务没做完不许只回一句话就结束、每步报进度）和
    第 15 条（没确认的推断不能当理由拒绝 BOOS 明确的指令）都在
    循环系统消息里——2026-11-16 真机测试问题 3、问题 4 的规矩层。"""
    text = build_loop_system_message()
    assert "多步任务没做完" in text
    assert "不许只回“收到”" in text
    assert "每做完一步，先报一声进度" in text
    assert "没确认的推断不能当事实用" in text
    assert "拿它当理由拒绝" in text
    assert "拿不准就问他" in text


# ---------- 问题 4a：删记忆的动作不许脑补成偏好变了 ----------


def test_describes_memory_management():
    """管记忆动作的复述长什么样：删/改 + 记忆/记录同时出现才算；
    真凭原话说的偏好变化（“我不爱吃冒菜了”）不提删记忆，不拦；
    “修改了会议记录”是正经工作事实，也不拦。"""
    assert _describes_memory_management(
        "用户于 2026 年 11 月 16 日表示自己不再爱吃冒菜了，并要求删除此前记录的相关偏好记忆。"
    )
    assert _describes_memory_management("把这条从记忆里删掉")
    assert _describes_memory_management("修改了记忆里的定投金额")
    assert not _describes_memory_management("我不爱吃冒菜了。")
    assert not _describes_memory_management("我喜欢吃火锅。")
    assert not _describes_memory_management("用户删掉了手机里的旧照片。")
    assert not _describes_memory_management("用户修改了会议记录。")
    assert not _describes_memory_management("")


def test_extraction_drops_management_action_fact(tmp_path, monkeypatch):
    """提取管线：把删记忆的动作复述成事实的那条，在提取后过滤里拦掉
    （从结果里去掉、从库里删掉、不进账本）；同一批里真凭原话的事实
    照常留下来——2026-11-16 真机测试账本 seq 5 那条的原样复现。"""
    store = make_fake_store(tmp_path)
    bad_text = "用户于 2026 年 11 月 16 日表示自己不再爱吃冒菜了，并要求删除此前记录的相关偏好记忆。"
    good_text = "我喜欢吃火锅。"
    # 坏的那条直接塞进库里（绕开 add_memory_text，不记账本 add 事件，
    # 这样断言“坏文本没进账本”才是过滤本身的功劳）；_mem0_add 打桩，
    # 假装提取刚从这轮对话里提了这两条。
    bad_id = _add_raw(store, bad_text)
    canned = {
        "results": [
            {"id": bad_id, "memory": bad_text, "event": "ADD"},
            {"id": "fake-999", "memory": good_text, "event": "ADD"},
        ]
    }
    monkeypatch.setattr(store, "_mem0_add", lambda *a, **k: canned)
    result = store._extract_and_save_turn(
        [{"role": "user", "content": "把'我爱吃冒菜'这条删掉"}]
    )
    kept = [(item or {}).get("memory") or "" for item in (result or {}).get("results") or []]
    assert bad_text not in kept
    assert good_text in kept
    assert bad_text not in _texts(store)
    assert bad_text not in [
        (e or {}).get("text") or "" for e in _ledger.read_events(store.ledger_path)[0]
    ]


def test_custom_instructions_bans_management_facts():
    """提取提示词第 14 条：删/改记忆的请求不是新事实，不许把“删掉 X”
    存成“不再喜欢 X”——2026-11-16 真机测试问题 4a 的提示词层。"""
    assert "NEVER turn a request to manage memories" in CUSTOM_INSTRUCTIONS
    assert "Delete X" in CUSTOM_INSTRUCTIONS
    assert "does NOT mean the user no longer likes" in CUSTOM_INSTRUCTIONS


# ---------- 附带：tzdata 进 requirements ----------


def test_requirements_include_tzdata():
    """Windows 的 Python 没有系统时区库，zoneinfo 要靠 tzdata 才能认
    Asia/Shanghai——2026-11-16 笔记本跑套件 13 个文件收集报错，
    装了 tzdata 才过。写进 requirements，新检出在 Windows 上直接能用。"""
    text = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "tzdata" in text
