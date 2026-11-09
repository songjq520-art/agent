# tests/test_mem0_provenance.py（来源元数据 + 价值判断 + 纠错 + 检索来源）
# 作用：守住 2026-11-09 这一轮的三条线——
#   1. 每条记忆都带来源（source / 形成时间 / 置信度 / 是否用户确认），
#      CLI“记住”真的把 source="user" 传给 mem0，迁移给旧记忆补来源；
#   2. 纠错（correct_memory）原地替换、找不到/分不清时不猜不改、
#      改失败原记忆不动；
#   3. 检索带来源注入：已确认的才能当事实，未确认的不能当事实，
#      阈值 0.4 不动。
# 测试全部是离线假对象测试（FakeMemory，见 tests/conftest.py）：
# 不碰真实 mem0、不联网、不产生费用。
# 注意：FakeMemory 的提取是规则不是模型（只认句首前缀），所以这里的
# 提取测试只证明“流程通了、来源标对了”，不证明真实提取得准；
# 真实提取质量靠 tests/test_mem0_regression.py 里默认跳过的真实 API 测试。

import json

import pytest

from main import _memory_block
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    CONFIDENCE_USER,
    CUSTOM_INSTRUCTIONS,
    CUSTOM_INSTRUCTIONS_VERSION,
    RETRIEVAL_THRESHOLD,
    Mem0Store,
    _normalize_metadata,
)
from conftest import FakeMemory, make_fake_store


# ---------- 来源元数据：两条写入路径 ----------


def test_add_memory_text_stores_user_provenance(tmp_path):
    """CLI“记住”存的记忆带来源：source=user、user_confirmed=True、
    confidence=1.0、形成时间 present（lifecycle 里 REMEMBER 那步
    metadata=None 的缺口就在这里补上）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋。")
    items = store.get_all()
    assert len(items) == 1
    metadata = items[0]["metadata"]
    assert metadata["source"] == "user"
    assert metadata["user_confirmed"] is True
    assert metadata["confirmed"] is True
    assert metadata["confidence"] == CONFIDENCE_USER == 1.0
    assert metadata["created_at"]


def test_add_memory_text_passes_source_to_mem0(tmp_path):
    """回归：add_memory_text 接受 source 参数，必须真的传给 mem0.add，
    不能像以前那样接了又丢掉。直接看 FakeMemory 里存下来的 metadata。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋。", source="user")
    raw = store._memory._items[0]["metadata"]
    assert raw is not None and raw["source"] == "user"
    store.add_memory_text("我喜欢喝茶。", source="inferred")
    raw2 = store._memory._items[1]["metadata"]
    assert raw2["source"] == "inferred"
    assert raw2["user_confirmed"] is False


def test_add_turn_stores_inferred_provenance(tmp_path):
    """提取路径存的记忆带来源：source=inferred、user_confirmed=False、
    confidence=0.7——用户明确说过的话，但 BOOS 没逐条确认过写法。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我叫小宋。"}])
    items = store.get_all()
    assert len(items) == 1
    metadata = items[0]["metadata"]
    assert metadata["source"] == "inferred"
    assert metadata["user_confirmed"] is False
    assert metadata["confirmed"] is False
    assert metadata["confidence"] == CONFIDENCE_INFERRED == 0.7
    assert metadata["created_at"]


def test_inferred_not_auto_upgraded_by_saving(tmp_path):
    """推断来的信息不会因为“存下来了”就自动变成已确认：
    同一句话提取两次，存下来的两条来源都还是 inferred、未确认。"""
    store = make_fake_store(tmp_path)
    for _ in range(2):
        store.add_turn([{"role": "user", "content": "我叫小宋。"}])
    items = store.get_all()
    assert len(items) == 2
    for item in items:
        assert item["metadata"]["source"] == "inferred"
        assert item["metadata"]["confirmed"] is False


def test_search_returns_provenance(tmp_path):
    """检索结果带来源：每项都有 metadata 和 confirmed；用户确认的
    那条 confirmed=True，提取的那条 confirmed=False。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋。")
    store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。"}])
    found = store.search("我叫什么？喜欢喝什么？")
    by_text = {item["text"]: item for item in found}
    assert by_text["我叫小宋。"]["confirmed"] is True
    assert by_text["我喜欢喝咖啡"]["confirmed"] is False
    assert by_text["我喜欢喝咖啡"]["metadata"]["source"] == "inferred"


# ---------- 迁移与种子：旧记忆补来源 ----------


def _write_old_memories(path, items):
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def test_migration_assigns_user_source(tmp_path):
    """迁移给旧记忆补来源：v1 时代是先确认再存，所以迁过来的都记
    source=user、user_confirmed=True；旧条目有 date 就用它当形成时间。"""
    store = make_fake_store(tmp_path)
    old_path = tmp_path / "memories.json"
    _write_old_memories(
        old_path,
        [
            {"text": "我叫小宋。", "layers": ["core"], "source": "user", "date": "2026-11-07"},
            {"text": "我喜欢喝咖啡。", "layers": ["experience"], "source": "user"},
        ],
    )
    assert store.migrate_from_v1(old_path) == 2
    by_text = {item["text"]: item for item in store.get_all()}
    assert by_text["我叫小宋。"]["metadata"]["source"] == "user"
    assert by_text["我叫小宋。"]["metadata"]["confirmed"] is True
    assert by_text["我叫小宋。"]["metadata"]["created_at"] == "2026-11-07"
    assert by_text["我喜欢喝咖啡。"]["metadata"]["confirmed"] is True
    assert by_text["我喜欢喝咖啡。"]["metadata"]["created_at"]


def test_first_memories_seeded_as_user(tmp_path, monkeypatch):
    """首批三条种子记忆也是用户确认的（BOOS 2026-11-07 确认过）。

    迁移步骤在这里 stub 掉（返回 0，表示没有旧文件可迁），只测“空库种 3 条、
    来源都是 user”这一路：ensure_first_memories() 会先试着从项目
    memory/memories.json 迁移，本机跑过迁移后那个文件真实存在（6 条），
    不隔离就会迁出 6 条、断言 3 条失败（BOOS 2026-11-09 本机实测）。
    迁移本身在上面的 test_migration_assigns_user_source 里覆盖。
    """
    monkeypatch.setattr(Mem0Store, "migrate_from_v1", lambda self, memories_json_path=None: 0)
    store = make_fake_store(tmp_path)
    assert store.ensure_first_memories() == 3
    for item in store.get_all():
        assert item["metadata"]["source"] == "user"
        assert item["metadata"]["confirmed"] is True


# ---------- 旧记忆 / 坏来源：安全降级 ----------


def test_old_memory_without_metadata_defaults_unconfirmed(tmp_path):
    """没有 metadata 的旧记忆，取回时一律按“未确认”处理，绝不默认
    已确认：直接调底层 FakeMemory 存一条（不带 metadata，模拟
    2026-11-09 之前存的旧记忆），search 回来的来源是 unconfirmed，
    提示块里打【来源未确认】、写明不能当作事实。"""
    store = make_fake_store(tmp_path)
    store._memory.add([{"role": "user", "content": "我叫小宋。"}], infer=False)
    found = store.search("你记得我叫什么吗？")
    assert found
    assert found[0]["metadata"]["source"] == "unconfirmed"
    assert found[0]["confirmed"] is False
    block = _memory_block(found)
    assert "【来源未确认】" in block
    assert "不能当作事实" in block


@pytest.mark.parametrize(
    "raw, expected_source, expected_confirmed, expected_confidence",
    [
        (None, "unconfirmed", False, None),
        ({}, "unconfirmed", False, None),
        ({"source": "??? "}, "unconfirmed", False, None),
        ({"source": "user"}, "user", False, None),  # 有来源、没确认字段：不算确认
        ({"source": "user", "user_confirmed": "yes"}, "user", False, None),  # "yes" 不是布尔 True：不算确认
        ({"source": "user", "user_confirmed": 1}, "user", False, None),  # 1 不是布尔 True：不算确认
        ({"source": "user", "user_confirmed": True}, "user", True, None),
        ({"source": "inferred", "user_confirmed": True}, "inferred", False, None),  # 推断来的不能靠一个 True 字段升级
        ({"source": "user", "user_confirmed": True, "confidence": 2}, "user", True, None),  # 越界置信度记 None
        ({"source": "user", "user_confirmed": True, "confidence": 0.9}, "user", True, 0.9),
        ("not-a-dict", "unconfirmed", False, None),
    ],
)
def test_normalize_metadata_handles_malformed(
    raw, expected_source, expected_confirmed, expected_confidence
):
    """来源规整是安全降级：缺失/损坏/含糊的来源一律按未确认处理，
    确认必须是明确的布尔 True，推断来的不能靠字段升级成已确认。"""
    metadata = _normalize_metadata(raw)
    assert metadata["source"] == expected_source
    assert metadata["confirmed"] is expected_confirmed
    assert metadata["confidence"] == expected_confidence


# ---------- 价值判断：提示词文本 + 行为边界 ----------
# 测试 6 和测试 7 方向相反（宁缺毋滥 vs 宁滥毋缺），边界定在这里：
# 过滤按“每条事实”做，不按“每轮对话”做——一轮里混着闲聊和真事实时，
# 真事实存下来、闲聊那句丢掉；整轮只有闲聊时，一条也不存。


def test_custom_instructions_value_judgement():
    """价值判断规则写进了提取提示（DeepSeek 自己判断，不另设机制、
    不多调一次模型）：只存值得长期记的；没确认/不值得记的直接不存；
    别人的事不是用户的事实；没被认可的 AI 观点不是用户立场；
    过滤按事实做、不按整轮做。版本号也跟着改了。"""
    assert CUSTOM_INSTRUCTIONS_VERSION == "2026-11-09-provenance"
    assert "VALUE JUDGEMENT" in CUSTOM_INSTRUCTIONS
    assert "long-term goals" in CUSTOM_INSTRUCTIONS
    assert "Do NOT store trivial" in CUSTOM_INSTRUCTIONS
    assert "unconfirmed / not-worth-remembering" in CUSTOM_INSTRUCTIONS
    assert "other people's experiences" in CUSTOM_INSTRUCTIONS
    assert "explicitly endorses" in CUSTOM_INSTRUCTIONS
    assert "per fact, not per turn" in CUSTOM_INSTRUCTIONS


def test_trivial_chat_not_stored_but_fact_in_same_turn_is(tmp_path):
    """边界测试（6+7）：整轮只有闲聊（“今天天气真好。”）时一条也不存；
    同一轮里闲聊混着真事实（“今天天气真好。我叫小宋。”）时，
    真事实存下来、闲聊那句丢掉——不因一半是闲聊就整轮丢掉，
    也不因沾了真事实就把闲聊也存下来。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    assert store.get_all() == []
    store.add_turn([{"role": "user", "content": "今天天气真好。我叫小宋。"}])
    texts = [item["text"] for item in store.get_all()]
    assert texts == ["我叫小宋"]  # 提取按句号拆句，存下来的句子没有句末句号


def test_others_experiences_not_attributed_to_user(tmp_path):
    """别人的事不记成用户的事实：“我朋友喜欢爬山”说的是朋友，
    不是 BOOS；“我朋友说圆通很好”也一样——都不存。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我朋友喜欢爬山。"}])
    store.add_turn([{"role": "user", "content": "我朋友说圆通很好。"}])
    assert store.get_all() == []


def test_unendorsed_ai_view_not_saved_as_user_stance(tmp_path):
    """没被认可的 AI 观点不存成用户立场：assistant 的建议（“我建议你
    每天运动”）不提取；用户只问了一句、没表态，也不存。"""
    store = make_fake_store(tmp_path)
    store.add_turn(
        [
            {"role": "user", "content": "你觉得我该多运动吗？"},
            {"role": "assistant", "content": "我建议你每天运动。"},
        ]
    )
    assert store.get_all() == []
    assert not any("运动" in item["text"] for item in store.get_all())


# ---------- 纠错（correct_memory）：6 个场景 ----------


def test_correction_replaces_old_memory(tmp_path):
    """纠错场景 1（lifecycle 缺口复现）：Python→Go，改完库里只有 Go，
    id 不变（原地替换，不是新存一条），旧文本搜不到。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我喜欢用 Python 写代码。")
    mem_id = result["results"][0]["id"]
    outcome = store.correct_memory("我喜欢用 Python 写代码", "我喜欢用 Go 写代码。")
    assert outcome["status"] == "updated"
    assert outcome["memory_id"] == mem_id
    assert [item["text"] for item in store.get_all()] == ["我喜欢用 Go 写代码。"]
    found = store.search("我喜欢用什么语言写代码？")
    assert found and all("Go" in item["text"] for item in found)
    assert not any("Python" in item["text"] for item in found)


def test_correction_decision_changed_then_changed_again(tmp_path):
    """纠错场景 2（决定改了又改）：Python→Go→Rust，最后库里只有 Rust，
    中间态的 Go 也搜不到——每次纠错都替换同一条，不留历史版本当事实。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢用 Python 写代码。")
    assert store.correct_memory("我喜欢用 Python 写代码", "我喜欢用 Go 写代码。")["status"] == "updated"
    assert store.correct_memory("我喜欢用 Go 写代码", "我喜欢用 Rust 写代码。")["status"] == "updated"
    assert [item["text"] for item in store.get_all()] == ["我喜欢用 Rust 写代码。"]
    assert not any("Go" in item["text"] for item in store.search("我喜欢用什么语言写代码？"))


def test_correction_supersedes_old_in_search(tmp_path):
    """纠错场景 3（纠正 supersedes 旧内容）：直接拿旧问法搜，
    回来的只有新内容，旧内容一条都没有。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢用 Python 写代码。")
    store.correct_memory("我喜欢用 Python 写代码", "我喜欢用 Go 写代码。")
    found = store.search("我喜欢用 Python 写代码吗？")
    assert not any("Python" in item["text"] for item in found)


def test_multiple_consecutive_corrections(tmp_path):
    """纠错场景 4（连续多次纠错）：广州→深圳→上海→北京，同一 id 改三次，
    最后只有北京；history 里留着 ADD + 3 条 UPDATE 审计（改过什么有据可查）。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我住在广州。")
    mem_id = result["results"][0]["id"]
    for old, new in [
        ("我住在广州", "我住在深圳。"),
        ("我住在深圳", "我住在上海。"),
        ("我住在上海", "我住在北京。"),
    ]:
        assert store.correct_memory(old, new)["status"] == "updated"
    assert [item["text"] for item in store.get_all()] == ["我住在北京。"]
    assert store.get_all()[0]["id"] == mem_id
    entries = store.history(mem_id)
    assert [entry["event"] for entry in entries] == ["ADD", "UPDATE", "UPDATE", "UPDATE"]
    assert entries[-1]["old_memory"] == "我住在上海。"
    assert entries[-1]["new_memory"] == "我住在北京。"


def test_ambiguous_correction_asks_do_not_overwrite(tmp_path):
    """纠错场景 5（分不清改哪条）：库里有黑咖啡和拿铁两条相似记忆，
    只说“你记错了，我喜欢喝咖啡”分不清是哪条——返回 ambiguous、
    列出候选，什么都不改、不猜一条改掉。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝黑咖啡。")
    store.add_memory_text("我喜欢喝拿铁咖啡。")
    outcome = store.correct_memory("我喜欢喝咖啡", "我喜欢喝茶。")
    assert outcome["status"] == "ambiguous"
    assert len(outcome["candidates"]) == 2
    assert store._memory.update_calls == []
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢喝黑咖啡。",
        "我喜欢喝拿铁咖啡。",
    ]


def test_correction_exact_match_wins_over_similar(tmp_path):
    """纠错补充：两条相似记忆里，有一条和旧内容原文完全对得上，
    就改那条（不用问）；另一条原封不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝黑咖啡。")
    store.add_memory_text("我喜欢喝拿铁咖啡。")
    outcome = store.correct_memory("我喜欢喝黑咖啡", "我喜欢喝茶。")
    assert outcome["status"] == "updated"
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢喝茶。",
        "我喜欢喝拿铁咖啡。",
    ]


def test_correction_not_found_changes_nothing(tmp_path):
    """纠错场景 6（库里没有）：旧内容对不上任何记忆时返回 not_found，
    不新存一条、不改任何东西。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    outcome = store.correct_memory("我住在广州", "我住在深圳。")
    assert outcome["status"] == "not_found"
    assert outcome["candidates"] == []
    assert [item["text"] for item in store.get_all()] == ["我喜欢喝咖啡。"]
    assert store._memory.update_calls == []


def test_correction_failure_preserves_original(tmp_path):
    """纠错失败时原记忆不动、报失败：底层 update 抛错，返回 failed，
    库里还是旧文本，绝不谎称改好了。"""

    class _FailingUpdateMemory(FakeMemory):
        def update(self, memory_id, text=None, metadata=None):
            raise RuntimeError("向量库炸了")

    failing = _FailingUpdateMemory()
    failing.add([{"role": "user", "content": "我喜欢喝咖啡。"}], infer=False)
    failing_store = Mem0Store(
        memory=failing,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    outcome = failing_store.correct_memory("我喜欢喝咖啡", "我喜欢喝茶。")
    assert outcome["status"] == "failed"
    assert "原记忆未动" in outcome["reason"]
    assert [item["text"] for item in failing_store.get_all()] == ["我喜欢喝咖啡。"]


def test_correction_marks_user_confirmed(tmp_path):
    """纠错完的记忆来源是 user（BOOS 亲口纠正的）、已确认——
    纠正过的事实以后可以直接当作事实回答。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。"}])
    assert store.get_all()[0]["metadata"]["confirmed"] is False
    store.correct_memory("我喜欢喝咖啡", "我喜欢喝茶。")
    metadata = store.get_all()[0]["metadata"]
    assert metadata["source"] == "user"
    assert metadata["confirmed"] is True


def test_correction_unchanged_when_same(tmp_path):
    """旧内容和新内容一样时不用改：返回 unchanged，不调 update。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝茶。")
    outcome = store.correct_memory("我喜欢喝茶", "我喜欢喝茶。")
    assert outcome["status"] == "unchanged"
    assert store._memory.update_calls == []


# ---------- 用户表达 vs 推断：冲突时用户说了算 ----------


def test_user_expressed_beats_inferred_on_conflict(tmp_path):
    """用户明确说的压过推断来的：提取的“我喜欢喝咖啡”（inferred）和
    用户亲口说的“我喜欢喝茶”（user）同时在库里时，检索结果里
    茶那条是已确认、咖啡那条不是；提示块里已确认的排在前面、
    标签也不一样，模型先看到用户亲口说的。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。"}])
    store.add_memory_text("我喜欢喝茶。")
    found = store.search("我喜欢喝什么？")
    by_text = {item["text"]: item for item in found}
    assert by_text["我喜欢喝茶。"]["confirmed"] is True
    assert by_text["我喜欢喝咖啡"]["confirmed"] is False
    block = _memory_block(found)
    assert block.index("【用户确认】我喜欢喝茶。") < block.index("【提取所得，未经逐条确认】我喜欢喝咖啡")


# ---------- 检索来源 + 负向测试 ----------


def test_threshold_still_0_4(tmp_path):
    """阈值 0.4 不动（BOOS 的决定）：模块常量和 store 默认值都是 0.4。"""
    assert RETRIEVAL_THRESHOLD == 0.4
    assert make_fake_store(tmp_path).threshold == 0.4


def test_keyword_rescued_item_is_weak_evidence(tmp_path):
    """关键词捞回的记忆是弱证据：语义没命中时，关键词补充能捞回来，
    但结果里 source=keyword、score=None，调用方知道它不是语义命中的。"""

    class _EmptySearchMemory(FakeMemory):
        def search(self, query, filters=None, top_k=5):
            return {"results": []}

    empty = _EmptySearchMemory()
    store = Mem0Store(
        memory=empty,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    # 经适配器存（不是直接调底层）：这样记忆才带上来源元数据。
    store.add_memory_text("我目前在圆通工作，做快递员。")
    found = store.search("圆通是做什么的？")
    assert len(found) == 1
    assert found[0]["source"] == "keyword"
    assert found[0]["score"] is None
    assert found[0]["metadata"]["source"] == "user"


def test_same_keyword_different_meaning_only_weak_signal(tmp_path):
    """同一关键词、不同意思（“苹果”水果 vs “苹果”手机）：语义分
    只有 0.2、过不了 0.4 的阈值；关键词补充会把它捞回来（“苹果”
    这个词确实撞上了），但只作为 source=keyword、score=None 的
    弱证据返回，并且提示块里写着“只用真正回答了问题的记忆”——
    语义上分不分得清是模型的事，适配器保证的是：不把弱证据
    包装成语义命中、不把来源藏起来。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢吃苹果。")
    found = store.search("苹果手机值得买吗？")
    assert len(found) == 1
    assert found[0]["source"] == "keyword"
    assert found[0]["score"] is None
    block = _memory_block(found)
    assert "只用真正回答了问题的记忆" in block


def test_same_proper_noun_different_topic_not_semantic(tmp_path):
    """同一专有名词、不同话题（在圆通工作 vs 圆通客服电话）：
    语义检索给低分时不过阈值、不注入；关键词捞回的也只能是弱证据，
    不能冒充语义命中。反过来，没共享关键词的问题（“香蕉好吃吗？”）
    一条也捞不到——不知道就是不知道。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我在圆通工作。")
    found = store.search("圆通快递的客服电话是多少？")
    assert all(item["source"] == "keyword" for item in found)
    assert all(item["score"] is None for item in found)
    assert store.search("香蕉好吃吗？") == []


def test_memory_block_labels_all_three_tiers(tmp_path):
    """提示块按来源分三层打标签：已确认的可以直接当事实，
    提取所得只能当背景参考，未确认的不能当事实；三层标签都在块里，
    且已确认的排在最前面。"""
    block = _memory_block(
        [
            {"id": "1", "text": "旧记忆，没有来源。", "score": 0.9, "source": "semantic",
             "metadata": {"source": "unconfirmed", "confirmed": False}},
            {"id": "2", "text": "我喜欢喝咖啡。", "score": 0.9, "source": "semantic",
             "metadata": {"source": "inferred", "confirmed": False}},
            {"id": "3", "text": "我叫小宋。", "score": 0.9, "source": "semantic",
             "metadata": {"source": "user", "confirmed": True}},
        ]
    )
    assert "【用户确认】我叫小宋。" in block
    assert "【提取所得，未经逐条确认】我喜欢喝咖啡。" in block
    assert "【来源未确认】旧记忆，没有来源。" in block
    assert "只有【用户确认】的记忆可以直接当作事实回答" in block
    assert block.index("我叫小宋") < block.index("我喜欢喝咖啡") < block.index("旧记忆")


def test_memory_block_legacy_dict_without_metadata(tmp_path):
    """没带来源信息的旧格式 dict（比如以前的调用方直接拼的），
    按“来源未确认”处理、文本照常列出——老测试和老调用不断。"""
    block = _memory_block([{"id": "fake-1", "text": "我叫小宋。", "score": 0.9, "source": "semantic"}])
    assert "我叫小宋" in block
    assert "【来源未确认】" in block
    assert "仅供参考" in block
