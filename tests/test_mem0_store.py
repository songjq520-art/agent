# tests/test_mem0_store.py（mem0 适配层 Mem0Store 的离线测试）
# 作用：验证 memory/mem0_store.py 的行为：阈值过滤、关键词补充、
#       写入、删除两道关、迁移、种子、版本校验、用量包装、遥测关闭。
# 测试全部是离线假对象测试（FakeMemory，见 tests/conftest.py）：
# 不碰真实 mem0、不联网、不产生费用，默认运行。
# 注意：mem0 没装在 .venv 里，这里一律不 import mem0。

import json
import os
import threading
from types import SimpleNamespace

import pytest

from config import MODEL_CONFIG, BudgetTracker
from memory.mem0_store import (
    CUSTOM_INSTRUCTIONS,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    FIRST_MEMORIES,
    MEM0_VERSION,
    RETRIEVAL_THRESHOLD,
    Mem0Store,
    _check_version_file,
    _write_version_file,
)
from conftest import FakeMemory, make_fake_store


# ---------- 遥测 ----------


def test_mem0_telemetry_is_off_after_import():
    """遥测默认关：import memory.mem0_store 后 MEM0_TELEMETRY 必须是 False。

    mem0 默认会把使用情况发到 PostHog；适配层在 import mem0 之前就关掉，
    这里断言环境变量确实是 False（不是“没设”，是明确关了）。"""
    assert os.environ.get("MEM0_TELEMETRY") == "False"


def test_threshold_default_is_0_4():
    """检索阈值默认值是 0.4（试跑实测：答对的记忆分数在 0.40–0.66 之间，
    0.5 会把“小宋”那条正确的也滤掉）。"""
    assert RETRIEVAL_THRESHOLD == 0.4
    assert make_fake_store().threshold == 0.4


def test_injected_memory_skips_version_check(tmp_path):
    """注入假 memory 时跳过版本校验和真实构建：不写版本文件、不碰 Chroma，
    临时目录里什么也不产生。"""
    store = make_fake_store(tmp_path)
    assert store.available is True
    assert not (tmp_path / "mem0_versions.json").exists()
    assert not (tmp_path / "mem0_chroma").exists()


# ---------- search：阈值过滤 ----------


class _FixedSearchMemory(FakeMemory):
    """search 返回固定结果的 FakeMemory（绕开假打分，直接测适配层的阈值逻辑）。"""

    def __init__(self, fixed_results):
        super().__init__()
        self._fixed = list(fixed_results)

    def search(self, query, filters=None, top_k=5):
        return {"results": list(self._fixed)}


def _store_with_fixed_search(tmp_path, fixed_results, seed_texts=()):
    """造一个 search 返回固定结果、get_all 返回 seed_texts 的 store。"""
    memory = _FixedSearchMemory(fixed_results)
    for text in seed_texts:
        memory.add([{"role": "user", "content": text}], infer=False)
    return Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )


def test_search_filters_out_below_threshold(tmp_path):
    """语义结果按阈值过滤：分数 0.39 的不返回，0.41 的返回。

    固定结果里的记忆文本和问题没有关键词重合，避免关键词补充信号
    把它捞回来，测的就是纯阈值过滤。"""
    store = _store_with_fixed_search(
        tmp_path,
        [
            {"id": "fake-low", "memory": "完全不相关的旧记录甲", "score": 0.39},
            {"id": "fake-high", "memory": "完全不相关的旧记录乙", "score": 0.41},
        ],
    )
    found = store.search("你记得我叫什么吗？")
    assert [item["id"] for item in found] == ["fake-high"]
    assert found[0]["source"] == "semantic"
    assert found[0]["score"] == pytest.approx(0.41)


def test_search_drops_missing_score(tmp_path):
    """分数缺失的语义结果不注入：不知道它有多相关，就当它不相关。"""
    store = _store_with_fixed_search(
        tmp_path,
        [{"id": "fake-noscore", "memory": "完全不相关的旧记录丙", "score": None}],
    )
    assert store.search("你记得我叫什么吗？") == []


def test_search_threshold_is_configurable(tmp_path):
    """阈值可调：threshold=0.3 时，0.39 分的结果就能回来。"""
    memory = _FixedSearchMemory(
        [{"id": "fake-low", "memory": "完全不相关的旧记录甲", "score": 0.39}]
    )
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
        threshold=0.3,
    )
    found = store.search("你记得我叫什么吗？")
    assert len(found) == 1 and found[0]["id"] == "fake-low"


# ---------- search：关键词补充信号 ----------


class _EmptySearchMemory(FakeMemory):
    """search 永远返回空的 FakeMemory：模拟语义检索漏掉专有名词。"""

    def search(self, query, filters=None, top_k=5):
        return {"results": []}


def test_search_keyword_rescue_for_proper_noun(tmp_path):
    """关键词补充能捞回语义漏掉的专有名词：FakeMemory.search 故意不返回
    “圆通”那条，但关键词“圆通”能命中它，结果里以 source=keyword 出现。"""
    memory = _EmptySearchMemory()
    memory.add([{"role": "user", "content": "我目前在圆通工作，做快递员。"}], infer=False)
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    found = store.search("圆通是做什么的？")
    assert len(found) == 1
    assert "圆通" in found[0]["text"]
    assert found[0]["source"] == "keyword"
    assert found[0]["score"] is None


def test_search_keyword_supplement_goes_after_semantic(tmp_path):
    """关键词补充排在语义结果后面：语义命中的在前，关键词捞回的在后。"""
    memory = _EmptySearchMemory()
    memory.add([{"role": "user", "content": "我叫小宋。"}], infer=False)
    memory.add([{"role": "user", "content": "我目前在圆通工作。"}], infer=False)
    # search 只返回小宋那条（高分），圆通那条靠关键词补充。
    memory.search = lambda query, filters=None, top_k=5: {
        "results": [{"id": "fake-1", "memory": "我叫小宋。", "score": 0.9}]
    }
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    found = store.search("我叫什么？在圆通工作的是谁？")
    assert [item["source"] for item in found] == ["semantic", "keyword"]
    assert found[0]["text"] == "我叫小宋。"
    assert "圆通" in found[1]["text"]


def test_search_top_k_cap(tmp_path):
    """top_k 上限：库里 6 条都命中关键词，top_k=3 时只返回 3 条。"""
    store = make_fake_store(tmp_path)
    for text in [
        "小宋喜欢喝咖啡。",
        "小宋在圆通工作。",
        "小宋的目标是创业。",
        "小宋用 Kimi。",
        "小宋学英语。",
        "小宋住在广州。",
    ]:
        store.add_memory_text(text)
    found = store.search("小宋是谁？", top_k=3)
    assert len(found) == 3


def test_search_error_returns_what_was_got(tmp_path):
    """search 出错时不抛异常：语义检索炸了，关键词补充还能捞回结果。"""
    memory = _EmptySearchMemory()
    memory.add([{"role": "user", "content": "我目前在圆通工作。"}], infer=False)

    def _boom(query, filters=None, top_k=5):
        raise RuntimeError("向量库炸了")

    memory.search = _boom
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    found = store.search("圆通是做什么的？")
    assert len(found) == 1 and "圆通" in found[0]["text"]


def test_search_both_steps_fail_returns_empty(tmp_path):
    """语义检索和关键词补充都炸了：返回空列表，不抛异常、不拦聊天。"""
    memory = _EmptySearchMemory()

    def _boom(*args, **kwargs):
        raise RuntimeError("全炸了")

    memory.search = _boom
    memory.get_all = _boom
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    assert store.search("你好") == []


# ---------- add_turn / add_memory_text ----------


def test_add_turn_calls_mem0_add_with_infer(tmp_path):
    """add_turn 调一次 mem0.add（infer=True，自动提取），本轮对话存进记忆。

    Plan C：本轮先待定，flush 时才提交——这里用 flush_pending()
    主动提交，再断言提取调了一次、infer=True。"""
    store = make_fake_store(tmp_path)
    result = store.add_turn(
        [
            {"role": "user", "content": "我喜欢喝咖啡。"},
            {"role": "assistant", "content": "好的，记下了。"},
        ]
    )
    assert result.get("pending") is True
    assert store._memory.add_calls == []
    store.flush_pending()
    assert "我喜欢喝咖啡" in store._memory.all_texts()
    # 只记了一次 add 调用。
    assert len(store._memory.add_calls) == 1
    _messages, _user_id, infer = store._memory.add_calls[0]
    assert infer is True


def test_add_turn_does_not_store_assistant_suggestions(tmp_path):
    """add_turn 不把 assistant 的建议存成用户事实：assistant 说“你可以试试
    每天运动”，FakeMemory 不提它（CUSTOM_INSTRUCTIONS 第 3 条）。"""
    store = make_fake_store(tmp_path)
    store.add_turn(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "我喜欢每天运动，你可以试试每天运动。"},
        ]
    )
    assert store._memory.all_texts() == []


def test_add_turn_empty_messages_does_nothing(tmp_path):
    """add_turn 传空消息列表时返回 {}，不调 mem0.add。"""
    store = make_fake_store(tmp_path)
    assert store.add_turn([]) == {}
    assert store._memory.add_calls == []


def test_add_turn_failure_returns_empty_dict(tmp_path):
    """add 失败时返回 {} 不抛异常：存记忆失败不能把已经拿到的聊天回答搞崩。

    Plan C：失败发生在提交时（本轮先待定），所以 flush_pending() 返回 {}。"""
    memory = FakeMemory()

    def _boom(messages, user_id=None, infer=True):
        raise RuntimeError("Chroma 写不进去了")

    memory.add = _boom
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    result = store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。"}])
    assert result.get("pending") is True
    assert store.flush_pending() == {}


def test_add_turn_concurrent_writes_do_not_crash(tmp_path):
    """写锁存在：两个线程同时 add_turn，最终不崩、库还能正常读。

    Plan C：两轮说的是同一句话，第二轮提交第一轮的待定内容（1 次 add），
    再把自己待定；flush 时按指纹去重、不再重复提——同一句话最终只存 1 条。
    只断言“不崩”+ 最终库里有这条：去重是提取那侧 + 待定区指纹做的事。
    """
    store = make_fake_store(tmp_path)
    messages = [{"role": "user", "content": "我叫小宋。"}]
    errors = []

    def _write():
        try:
            store.add_turn(messages)
        except Exception as exc:  # noqa: BLE001 - 测试里要把异常记下来断言
            errors.append(exc)

    threads = [threading.Thread(target=_write) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    assert len(store._memory.add_calls) == 1
    store.flush_pending()
    assert len(store._memory.add_calls) == 1
    assert store.get_all()  # 库还能正常读


def test_add_memory_text_stores_verbatim(tmp_path):
    """add_memory_text 用 infer=False 原文存入，不经过提取、不改写。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我喜欢喝咖啡，不加糖。")
    assert result.get("results")
    assert store._memory.all_texts() == ["我喜欢喝咖啡，不加糖。"]
    _messages, _user_id, infer = store._memory.add_calls[0]
    assert infer is False


def test_add_memory_text_empty_returns_empty_dict(tmp_path):
    """add_memory_text 空文本（或只有空白）返回 {}，不存。"""
    store = make_fake_store(tmp_path)
    assert store.add_memory_text("") == {}
    assert store.add_memory_text("   ") == {}
    assert store.add_memory_text(None) == {}
    assert store._memory.all_texts() == []


def test_add_memory_text_failure_returns_empty_dict(tmp_path):
    """手动存记忆失败时返回 {} 不抛异常。"""
    memory = FakeMemory()

    def _boom(messages, user_id=None, infer=True):
        raise RuntimeError("写不进去")

    memory.add = _boom
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    assert store.add_memory_text("我喜欢喝咖啡。") == {}


# ---------- get_all / history ----------


def test_get_all_returns_id_and_text(tmp_path):
    """get_all 正常返回：每项有 id 和 text。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋。")
    store.add_memory_text("我喜欢喝咖啡。")
    items = store.get_all()
    assert [item["text"] for item in items] == ["我叫小宋。", "我喜欢喝咖啡。"]
    assert all(item["id"] for item in items)


def test_get_all_failure_returns_empty_list(tmp_path):
    """get_all 出错时返回 []，不抛异常。"""
    memory = FakeMemory()

    def _boom(filters=None):
        raise RuntimeError("读不出来")

    memory.get_all = _boom
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    assert store.get_all() == []


def test_history_returns_audit_list(tmp_path):
    """history 正常返回：存入后能查到 ADD 审计记录。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我叫小宋。")
    mem_id = result["results"][0]["id"]
    entries = store.history(mem_id)
    assert isinstance(entries, list)
    assert entries and entries[0]["event"] == "ADD"
    assert entries[0]["memory"] == "我叫小宋。"


def test_history_failure_returns_empty_list(tmp_path):
    """history 出错时返回 []，不抛异常；查不到的 id 也返回 []。"""
    memory = FakeMemory()

    def _boom(memory_id):
        raise RuntimeError("history.db 坏了")

    memory.history = _boom
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    assert store.history("fake-1") == []
    good_store = make_fake_store(tmp_path)
    assert good_store.history("不存在的id") == []


# ---------- delete_memory：两道关 ----------


def test_delete_memory_requires_approval(tmp_path):
    """删除第一道关：approve=False 时抛 PermissionError，哪怕 id 是自己的。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我叫小宋。")
    mem_id = result["results"][0]["id"]
    with pytest.raises(PermissionError, match="需要 BOOS 同意"):
        store.delete_memory(mem_id, approve=False)
    # 没删掉：还在库里。
    assert any(item["id"] == mem_id for item in store.get_all())


def test_delete_memory_owned_with_approval_succeeds(tmp_path):
    """删除两道关都过：approve=True 且 id 属于当前用户时删除成功返回 True。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我叫小宋。")
    mem_id = result["results"][0]["id"]
    assert store.delete_memory(mem_id, approve=True) is True
    assert store.get_all() == []


def test_delete_memory_not_owned_returns_false(tmp_path):
    """删除归属校验：id 不属于当前用户时返回 False、不删，库里原有的不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋。")
    assert store.delete_memory("fake-999", approve=True) is False
    assert len(store.get_all()) == 1


def test_delete_memory_empty_id_returns_false(tmp_path):
    """删除空 id 返回 False（不抛异常、不删东西）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋。")
    assert store.delete_memory("", approve=True) is False
    assert store.delete_memory(None, approve=True) is False
    assert len(store.get_all()) == 1


# ---------- migrate_from_v1 ----------


def _write_old_memories(path, texts):
    """写一个假的旧 memories.json（v1 格式：列表，每项有 text）。"""
    path.write_text(
        json.dumps(
            [{"text": text, "layers": ["experience"], "source": "user", "date": "2026-11-07"} for text in texts],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_migrate_from_v1_rewrites_outdated_budget(tmp_path):
    """迁移：3 条旧记忆全迁进新库，其中旧预算文本被改成新规矩
    “每月预算 5 美元，到 5 美元提醒，不自动停。”，其余原文不动。"""
    old_path = tmp_path / "memories.json"
    _write_old_memories(
        old_path,
        [
            "DeepSeek 已接通（2026-11-07），真实调用成功，是第一任主控。",
            "连续对话已做好，进 main 了（PR #4、PR #5），多轮能记住上下文，能退出。",
            "每月预算 5 美元，4 美元提醒，5 美元停。",
        ],
    )
    store = make_fake_store(tmp_path)
    assert store.migrate_from_v1(old_path) == 3
    texts = store._memory.all_texts()
    assert len(texts) == 3
    assert "每月预算 5 美元，到 5 美元提醒，不自动停。" in texts
    assert "每月预算 5 美元，4 美元提醒，5 美元停。" not in texts
    assert texts[0].startswith("DeepSeek 已接通")


def test_migrate_from_v1_second_run_is_noop(tmp_path):
    """迁移去重：第二次迁移返回 0，不重复存。"""
    old_path = tmp_path / "memories.json"
    _write_old_memories(old_path, ["我叫小宋。", "我喜欢喝咖啡。", "每月预算 5 美元，4 美元提醒，5 美元停。"])
    store = make_fake_store(tmp_path)
    assert store.migrate_from_v1(old_path) == 3
    assert store.migrate_from_v1(old_path) == 0
    assert len(store.get_all()) == 3


def test_migrate_from_v1_missing_file_returns_zero(tmp_path):
    """迁移：文件不存在时返回 0，不抛异常；文件内容不是列表时也返回 0。"""
    store = make_fake_store(tmp_path)
    assert store.migrate_from_v1(tmp_path / "不存在.json") == 0
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"不是": "列表"}, ensure_ascii=False), encoding="utf-8")
    assert store.migrate_from_v1(bad_path) == 0
    assert store.get_all() == []


# ---------- ensure_first_memories ----------


def test_ensure_first_memories_seeds_three_on_empty_store(tmp_path, monkeypatch):
    """空库时种下 3 条首批记忆，第 3 条是新预算文本（到 5 美元提醒，不自动停）。

    迁移步骤在这里 stub 掉（返回 0，表示没有旧文件可迁），只测“空库种 3 条”
    这一路；迁移本身在上面的 migrate 测试里覆盖。
    """
    monkeypatch.setattr(Mem0Store, "migrate_from_v1", lambda self, memories_json_path=None: 0)
    store = make_fake_store(tmp_path)
    assert store.ensure_first_memories() == 3
    texts = [item["text"] for item in store.get_all()]
    assert texts == [item["text"] for item in FIRST_MEMORIES]
    assert texts[2] == "每月预算 5 美元，到 5 美元提醒，不自动停。"


def test_ensure_first_memories_noop_when_not_empty(tmp_path):
    """已有记忆时返回 0，什么也不种。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋。")
    assert store.ensure_first_memories() == 0
    assert len(store.get_all()) == 1


# ---------- _check_version_file ----------


def test_check_version_file_fresh_writes_file(tmp_path):
    """版本文件不存在 + Chroma 为空：通过，并写下版本文件（模型、维度、mem0 版本都对）。"""
    version_path = tmp_path / "mem0_versions.json"
    chroma_path = tmp_path / "mem0_chroma"
    _check_version_file(version_path, chroma_path, EMBEDDING_MODEL, EMBEDDING_DIMS, MEM0_VERSION)
    data = json.loads(version_path.read_text(encoding="utf-8"))
    assert data["embedding_model"] == EMBEDDING_MODEL
    assert data["embedding_dims"] == EMBEDDING_DIMS
    assert data["mem0_version"] == MEM0_VERSION


def test_check_version_file_matching_passes(tmp_path):
    """版本文件存在且模型/维度/mem0 版本都对得上：通过。"""
    version_path = tmp_path / "mem0_versions.json"
    chroma_path = tmp_path / "mem0_chroma"
    _write_version_file(version_path, EMBEDDING_MODEL, EMBEDDING_DIMS, MEM0_VERSION)
    _check_version_file(version_path, chroma_path, EMBEDDING_MODEL, EMBEDDING_DIMS, MEM0_VERSION)


@pytest.mark.parametrize(
    "model,dims,version",
    [
        ("另一个模型", EMBEDDING_DIMS, MEM0_VERSION),  # 模型名对不上
        (EMBEDDING_MODEL, 768, MEM0_VERSION),  # 维度对不上
        (EMBEDDING_MODEL, EMBEDDING_DIMS, "9.9.9"),  # mem0 版本对不上
    ],
)
def test_check_version_file_mismatch_raises(tmp_path, model, dims, version):
    """版本文件存在但模型名/维度/mem0 版本对不上：抛 RuntimeError，
    拒绝混用旧库（换 embedding 模型必须全量重嵌）。"""
    version_path = tmp_path / "mem0_versions.json"
    chroma_path = tmp_path / "mem0_chroma"
    _write_version_file(version_path, model, dims, version)
    with pytest.raises(RuntimeError, match="版本不匹配"):
        _check_version_file(version_path, chroma_path, EMBEDDING_MODEL, EMBEDDING_DIMS, MEM0_VERSION)


def test_check_version_file_chroma_data_without_version_raises(tmp_path):
    """Chroma 有数据但无版本文件：来历不明的旧库，抛 RuntimeError 拒绝启动。"""
    version_path = tmp_path / "mem0_versions.json"
    chroma_path = tmp_path / "mem0_chroma"
    chroma_path.mkdir(parents=True)
    (chroma_path / "chroma.sqlite3").write_text("假数据", encoding="utf-8")
    with pytest.raises(RuntimeError, match="版本不明"):
        _check_version_file(version_path, chroma_path, EMBEDDING_MODEL, EMBEDDING_DIMS, MEM0_VERSION)


# ---------- 用量包装 _attach_usage_wrapper ----------


def _fake_llm_memory(prompt_tokens=1000, completion_tokens=500):
    """带假 LLM 客户端的 memory：client.chat.completions.create 返回带 usage 的假响应。"""
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )
    completions = SimpleNamespace(create=lambda *args, **kwargs: response)
    chat = SimpleNamespace(completions=completions)
    client = SimpleNamespace(chat=chat)
    return SimpleNamespace(llm=SimpleNamespace(client=client))


def test_attach_usage_wrapper_records_cost(tmp_path):
    """用量包装：包上后调一次 create，BudgetTracker 的花费按
    config.MODEL_CONFIG["deepseek"] 的价格增加了。

    1000 输入 Token、500 输出 Token，DeepSeek Flash 闲时价：
    1000/1e6*0.15 + 500/1e6*0.60 = 0.00045 美元，记在 deepseek-flash 小计里。
    """
    price_in = MODEL_CONFIG["deepseek"]["price_input_per_1m"]
    price_out = MODEL_CONFIG["deepseek"]["price_output_per_1m"]
    expected = 1000 / 1_000_000 * price_in + 500 / 1_000_000 * price_out
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path, budget_tracker=tracker)
    store._memory = _fake_llm_memory()
    store._attach_usage_wrapper()
    store._memory.llm.client.chat.completions.create(model="deepseek-flash", messages=[])
    assert tracker.total_cost_usd == pytest.approx(expected)
    assert tracker.cost_for_model("deepseek-flash") == pytest.approx(expected)


def test_attach_usage_wrapper_without_tracker_is_noop(tmp_path):
    """没传 budget_tracker 时不包：调 _attach_usage_wrapper 不报错，
    create 还是原来的（不记账也不崩）。"""
    store = make_fake_store(tmp_path)
    store._memory = _fake_llm_memory()
    store._attach_usage_wrapper()  # 不应抛异常
    resp = store._memory.llm.client.chat.completions.create()
    assert resp.usage.prompt_tokens == 1000


def test_attach_usage_wrapper_tolerates_missing_llm(tmp_path):
    """包不住（假对象没有 llm 结构）时只记日志、不抛异常、不影响使用。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = make_fake_store(tmp_path, budget_tracker=tracker)
    store._memory = SimpleNamespace()  # 没有 .llm
    store._attach_usage_wrapper()
    assert tracker.total_cost_usd == 0.0


# ---------- CUSTOM_INSTRUCTIONS 文本（规则别被悄悄改没） ----------


def test_custom_instructions_mentions_key_rules():
    """自定义提取规则文本里必须有关键句（防止以后有人把规则改没了）：
    只存用户明确说的、绝不把推断当事实、条件句/假设句/闲聊不存。"""
    assert "ONLY store information the USER explicitly stated" in CUSTOM_INSTRUCTIONS
    assert "NEVER store an inference" in CUSTOM_INSTRUCTIONS
    assert "conditionals" in CUSTOM_INSTRUCTIONS
    assert "hypotheticals" in CUSTOM_INSTRUCTIONS
    assert "small talk" in CUSTOM_INSTRUCTIONS
