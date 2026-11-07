# tests/test_memory.py（长期记忆测试文件）
# 作用：用 pytest 验证记忆存储（memory/memories.py）的基本功能。
# 测试都是模拟测试（Mock Test）：只用临时文件，不联网，不产生费用，默认运行。
# 原理：用 pytest 自带的 tmp_path（临时文件夹）在隔离环境里测读写，
#       不碰真实的 memory/memories.json，也不改仓库里的任何文件。

import json

import pytest

from memory.memories import (
    FIRST_MEMORIES,
    LAYERS,
    SOURCES,
    MemoryStore,
    ensure_first_memories,
    propose_memory,
)


# ---------- 拟记忆：propose_memory ----------

def test_propose_memory_ok():
    """拟一条记忆：字段齐全，内容去掉了首尾空格，日期默认是今天。"""
    memory = propose_memory("DeepSeek 已接通", ["core"], "user")
    assert memory["text"] == "DeepSeek 已接通"
    assert memory["layers"] == ["core"]
    assert memory["source"] == "user"
    assert len(memory["date"]) == 10  # YYYY-MM-DD


def test_propose_memory_keeps_layers():
    """拟一条记忆：可以属于多个层，layers 原样保留。"""
    memory = propose_memory("DeepSeek 已接通", ["experience", "core"], "user")
    assert memory["layers"] == ["experience", "core"]


def test_propose_memory_rejects_bad_layers():
    """拟一条记忆：layers 非法时要报错，不能悄悄存错东西。"""
    with pytest.raises(ValueError):
        propose_memory("test", [], "user")
    with pytest.raises(ValueError):
        propose_memory("test", ["not-a-layer"], "user")
    with pytest.raises(ValueError):
        propose_memory("test", "core", "user")  # 不是列表也不行


def test_propose_memory_rejects_bad_source():
    """拟一条记忆：source 必须是三档之一，不能把猜测写成事实还不标清楚。"""
    with pytest.raises(ValueError):
        propose_memory("test", ["core"], "maybe")
    for source in SOURCES:
        assert propose_memory("test", ["core"], source)["source"] == source


def test_propose_memory_rejects_empty_text():
    """拟一条记忆：内容不能为空。"""
    with pytest.raises(ValueError):
        propose_memory("   ", ["core"], "user")


# ---------- 读写：MemoryStore ----------

def test_store_starts_empty_when_no_file(tmp_path):
    """新文件不存在时，记忆列表是空的，不报错。"""
    store = MemoryStore(tmp_path / "memories.json")
    assert store.memories == []
    assert store.corrupted is False


def test_add_and_save_round_trip(tmp_path):
    """存一条再读出来：内容、层、来源、日期都在，写进文件的是 JSON。"""
    path = tmp_path / "memories.json"
    store = MemoryStore(path)
    added = store.add_memory("每月预算 5 美元", ["core"], "user", "2026-11-07")
    assert added["text"] == "每月预算 5 美元"
    assert json.loads(path.read_text(encoding="utf-8"))[0]["text"] == "每月预算 5 美元"
    # 重新读一遍，内容还在。
    store2 = MemoryStore(path)
    assert len(store2.memories) == 1
    assert store2.memories[0]["source"] == "user"


def test_propose_does_not_save(tmp_path):
    """拟一条记忆不会自动存：先问再存，拟只是草稿。"""
    path = tmp_path / "memories.json"
    store = MemoryStore(path)
    candidate = propose_memory("候选记忆", ["knowledge"], "inferred")
    assert store.memories == []
    assert not path.exists()
    # 确认后用 add_proposed 存进去，才真的有。
    store.add_proposed(candidate)
    assert len(store.memories) == 1
    assert store.memories[0]["source"] == "inferred"


def test_search_keyword(tmp_path):
    """关键词检索：找到包含关键词的记忆，不区分大小写；找不到返回空列表。"""
    store = MemoryStore(tmp_path / "memories.json")
    store.add_memory("DeepSeek 已接通", ["experience", "core"], "user", "2026-11-07")
    store.add_memory("每月预算 5 美元", ["core"], "user", "2026-11-07")
    assert len(store.search("deepseek")) == 1
    assert store.search("DeepSeek")[0]["text"].startswith("DeepSeek")
    assert store.search("预算")[0]["text"].startswith("每月预算")
    assert store.search("不存在的词") == []
    assert store.search("   ") == []


def test_list_all_returns_copy(tmp_path):
    """list_all 返回的是拷贝，改它不影响存着的记忆。"""
    store = MemoryStore(tmp_path / "memories.json")
    store.add_memory("测试记忆", ["core"], "user", "2026-11-07")
    copied = store.list_all()
    copied[0]["text"] = "改掉了"
    assert store.memories[0]["text"] == "测试记忆"


def test_corrupted_file_is_not_overwritten(tmp_path):
    """文件损坏时：读出来是安全状态（corrupted=True），原文件不被覆盖，不能再存。"""
    path = tmp_path / "memories.json"
    path.write_text("{不是合法的 JSON", encoding="utf-8")
    store = MemoryStore(path)
    assert store.corrupted is True
    assert store.memories == []
    assert path.read_text(encoding="utf-8") == "{不是合法的 JSON"
    with pytest.raises(RuntimeError):
        store.save()
    with pytest.raises(RuntimeError):
        store.add_memory("新记忆", ["core"], "user")


def test_non_list_file_is_corrupted(tmp_path):
    """文件里不是一个列表时，同样进安全状态。"""
    path = tmp_path / "memories.json"
    path.write_text('{"text": "不是列表"}', encoding="utf-8")
    store = MemoryStore(path)
    assert store.corrupted is True


def test_bad_entries_trigger_corrupted_not_skip(tmp_path):
    """文件里混着格式不对的条目：不再跳过，而是整份文件进安全状态，原文件保留。

    原因：跳过再保存会永久丢掉坏条目；必须保留原文件，等人工检查修复。
    """
    path = tmp_path / "memories.json"
    raw = json.dumps(
        [
            {"text": "好记忆", "date": "2026-11-07", "layers": ["core"], "source": "user"},
            {"text": "   ", "date": "2026-11-07", "layers": ["core"], "source": "user"},
            {"text": "层不对", "date": "2026-11-07", "layers": ["nope"], "source": "user"},
            {"text": "来源不对", "date": "2026-11-07", "layers": ["core"], "source": "maybe"},
        ],
        ensure_ascii=False,
    )
    path.write_text(raw, encoding="utf-8")
    store = MemoryStore(path)
    assert store.corrupted is True
    assert store.memories == []
    assert path.read_text(encoding="utf-8") == raw


# ---------- 首批三条记忆 ----------

def test_first_memories_shape():
    """首批三条记忆：正好三条，每条都有内容、层、来源、日期，层和来源都合法。"""
    assert len(FIRST_MEMORIES) == 3
    for item in FIRST_MEMORIES:
        assert item["text"].strip()
        assert item["layers"]
        assert all(layer in LAYERS for layer in item["layers"])
        assert item["source"] in SOURCES
        assert len(item["date"]) == 10


def test_ensure_first_memories_only_once(tmp_path):
    """首批记忆只在空文件时写一次：第一次写 3 条，第二次什么也不做，不重复。"""
    store = MemoryStore(tmp_path / "memories.json")
    assert ensure_first_memories(store) == 3
    assert len(store.memories) == 3
    assert ensure_first_memories(store) == 0
    assert len(store.memories) == 3
    # 三条都在，来源都是用户确认过的。
    assert all(m["source"] == "user" for m in store.memories)
    texts = [m["text"] for m in store.memories]
    assert any("DeepSeek" in t for t in texts)
    assert any("连续对话" in t for t in texts)
    assert any("5 美元" in t for t in texts)


# ---------- 修复 1：格式异常条目进安全状态 ----------

def test_bad_entry_triggers_corrupted_state(tmp_path):
    """文件里只要有一条格式异常，就进安全状态，原文件保留，不被覆盖。"""
    path = tmp_path / "memories.json"
    raw = json.dumps(
        [
            {"text": "好记忆", "date": "2026-11-07", "layers": ["core"], "source": "user"},
            {"text": "坏记忆", "date": "2026-11-07", "layers": ["nope"], "source": "user"},
        ],
        ensure_ascii=False,
    )
    path.write_text(raw, encoding="utf-8")
    store = MemoryStore(path)
    assert store.corrupted is True
    assert "第 2 条" in store.corruption_reason
    assert store.memories == []
    # 原文件一字没动，坏条目还在。
    assert path.read_text(encoding="utf-8") == raw
    # 保存和新增都被禁止，不会覆盖原文件。
    with pytest.raises(RuntimeError):
        store.save()
    with pytest.raises(RuntimeError):
        store.add_memory("新记忆", ["core"], "user")


def test_bad_entry_each_kind_triggers_corrupted(tmp_path):
    """每种格式异常（空内容、坏日期、坏层、坏来源）都会进安全状态。"""
    good = {"text": "好记忆", "date": "2026-11-07", "layers": ["core"], "source": "user"}
    bad_variants = [
        {**good, "text": "   "},
        {**good, "date": "2026-13-01"},
        {**good, "date": "不是日期"},
        {**good, "layers": []},
        {**good, "layers": ["core", "nope"]},
        {**good, "source": "maybe"},
        {"text": "缺字段"},
    ]
    for bad in bad_variants:
        path = tmp_path / "memories.json"
        path.write_text(json.dumps([good, bad], ensure_ascii=False), encoding="utf-8")
        store = MemoryStore(path)
        assert store.corrupted is True, f"这条坏数据没进安全状态：{bad}"
        assert store.memories == []


# ---------- 修复 2：返回深拷贝 ----------

def test_search_returns_deep_copy(tmp_path):
    """search 返回的是深拷贝：改返回结果的 text 和 layers，都不影响内部记忆。"""
    store = MemoryStore(tmp_path / "memories.json")
    store.add_memory("测试记忆", ["core"], "user", "2026-11-07")
    found = store.search("测试")
    found[0]["text"] = "被改掉了"
    found[0]["layers"].append("knowledge")
    assert store.memories[0]["text"] == "测试记忆"
    assert store.memories[0]["layers"] == ["core"]


def test_list_all_returns_deep_copy(tmp_path):
    """list_all 返回的是深拷贝：改返回结果的 text 和 layers，都不影响内部记忆。"""
    store = MemoryStore(tmp_path / "memories.json")
    store.add_memory("测试记忆", ["core"], "user", "2026-11-07")
    listed = store.list_all()
    listed[0]["text"] = "被改掉了"
    listed[0]["layers"].append("relationship")
    assert store.memories[0]["text"] == "测试记忆"
    assert store.memories[0]["layers"] == ["core"]


# ---------- 顺手检查：日期严格验证 ----------

def test_propose_memory_rejects_bad_dates():
    """拟记忆时日期要严格验证：不是 YYYY-MM-DD 的真实日期就报错。"""
    with pytest.raises(ValueError):
        propose_memory("test", ["core"], "user", "2026-13-01")
    with pytest.raises(ValueError):
        propose_memory("test", ["core"], "user", "2026-02-30")
    with pytest.raises(ValueError):
        propose_memory("test", ["core"], "user", "不是日期")
    with pytest.raises(ValueError):
        propose_memory("test", ["core"], "user", "20261107")
    assert propose_memory("test", ["core"], "user", "2026-11-07")["date"] == "2026-11-07"


# ---------- 顺手检查：保存失败时内存和磁盘一致 ----------

def test_add_memory_keeps_consistent_when_save_fails(tmp_path, monkeypatch):
    """保存失败时，内存里的记忆不变，和磁盘保持一致（不会内存有、磁盘没有）。"""
    store = MemoryStore(tmp_path / "memories.json")
    store.add_memory("已有记忆", ["core"], "user", "2026-11-07")
    assert len(store.memories) == 1

    def _boom(*args, **kwargs):
        raise OSError("磁盘写不进去（模拟）")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)
    with pytest.raises(OSError):
        store.add_memory("新记忆", ["core"], "user", "2026-11-07")
    # 内存里还是只有原来那一条，新记忆没存进去。
    assert len(store.memories) == 1
    assert store.memories[0]["text"] == "已有记忆"
