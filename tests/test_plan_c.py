# tests/test_plan_c.py（Plan C：十二记忆信任修复，2026-11-09）
# 覆盖准备稿第五节的四项必测场景 + 辅助函数的单元测试。
# 全部离线（FakeMemory），不联网、不产生费用；真实 API 的多场景
# 测试另跑（见本文件末尾的说明），记 Token 和成本。

from memory.mem0_store import (
    AGENT_NAME_TEXT,
    AGENT_SOUL_TEXT,
    _detect_identity_statement,
    _filter_extraction_messages,
    _is_bare_retraction,
)
from conftest import make_fake_store


# ---------- 辅助函数：身份识别 / 撤回信号 ----------


def test_detect_identity_agent_name_forms():
    """身份识别：Agent 起名的几种说法都认，问句不认。"""
    assert _detect_identity_statement("你叫十二")["value"] == "十二"
    assert _detect_identity_statement("以后你叫十二了")["value"] == "十二"
    assert _detect_identity_statement("现在你叫十三")["value"] == "十三"
    assert _detect_identity_statement("你的名字是十二")["value"] == "十二"
    assert _detect_identity_statement("你叫什么？") is None
    assert _detect_identity_statement("你叫什么名字") is None
    assert _detect_identity_statement("你的名字是什么？") is None


def test_detect_identity_user_name_forms():
    """身份识别：BOOS 称呼的几种说法都认；“你叫我小宋”是叫“我”，
    不是给 Agent 起名（先认 user、后认 agent）。"""
    assert _detect_identity_statement("我叫小宋")["value"] == "小宋"
    assert _detect_identity_statement("我的名字是小宋")["value"] == "小宋"
    assert _detect_identity_statement("叫我小宋")["value"] == "小宋"
    detected = _detect_identity_statement("你叫我小宋")
    assert detected["slot"] == "user_name"
    assert detected["value"] == "小宋"
    assert _detect_identity_statement("我叫什么来着？") is None


def test_is_bare_retraction_only_bare_signal():
    """撤回信号：整句只有信号才算撤回；带替换内容的不算（走纠错）。"""
    assert _is_bare_retraction("不不不。") is True
    assert _is_bare_retraction("不对") is True
    assert _is_bare_retraction("算了。") is True
    assert _is_bare_retraction("不对，我喜欢看纪录片。") is False
    assert _is_bare_retraction("不是，我记错了。") is False
    assert _is_bare_retraction("我喜欢喝咖啡。") is False


# ---------- 场景 1：撤回——“现在你叫十三”→“不不不”→沉默 ----------


def test_retraction_thirteen_not_stored_original_name_kept(tmp_path):
    """必测 1（0 错取标准）：BOOS 说“现在你叫十三”，下一句“不不不”，
    然后沉默——十三绝不能进库，原名十二保留；即使后来 flush，
    被撤回的那轮也不会回来。"""
    store = make_fake_store(tmp_path)
    seeded = store.ensure_identity_memories()
    assert seeded["agent_name"]["status"] == "added"
    result = store.add_turn(
        [
            {"role": "user", "content": "现在你叫十三。"},
            {"role": "assistant", "content": "好的，以后我就叫十三。"},
        ]
    )
    assert result.get("pending") is True
    assert "十三" not in " ".join(store._memory.all_texts())
    result = store.add_turn(
        [
            {"role": "user", "content": "不不不。"},
            {"role": "assistant", "content": "好的，我不记了。"},
        ]
    )
    assert result.get("retracted") is True
    assert result.get("dropped") == 1
    # 沉默：没有下一轮，也不 flush——十三绝不能进库。
    texts = store._memory.all_texts()
    assert not any("十三" in text for text in texts)
    # 原名保留：十二还在，搜“你叫什么”找得到它。
    assert any("十二" in text for text in texts)
    found = store.search("你叫什么")
    assert any("十二" in item["text"] for item in found)
    # 即使后来 flush（比如服务重启时），被撤回的那轮也不会回来。
    store.flush_pending()
    assert not any("十三" in text for text in store._memory.all_texts())


def test_retraction_does_not_drop_committed_memory(tmp_path):
    """撤回只丢待定区里的那轮：已经落库的事实不受影响。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。"}])
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    # 咖啡已在上一轮提交时落库；本轮只有闲聊，撤回也只丢本轮。
    result = store.add_turn([{"role": "user", "content": "不对。"}])
    assert result.get("retracted") is True
    assert any("我喜欢喝咖啡" in item["text"] for item in store.get_all())


def test_bare_no_answering_assistant_question_does_not_retract(tmp_path):
    """撤回信号的例外：助手刚问了问题时，光秃秃的“不是。”更可能
    是在回答那个问题，不是撤回上一句——上一轮照常提交。"""
    store = make_fake_store(tmp_path)
    store.add_turn(
        [
            {"role": "user", "content": "我喜欢喝咖啡。"},
            {"role": "assistant", "content": "好的。你喜欢喝茶吗？"},
        ]
    )
    result = store.add_turn([{"role": "user", "content": "不是。"}])
    assert "retracted" not in result
    assert result.get("pending") is True
    store.flush_pending()
    assert any("我喜欢喝咖啡" in item["text"] for item in store.get_all())


# ---------- 场景 2：亲口命名落库，新对话问名字和 SOUL.md ----------


def test_seeded_name_and_soul_recall_in_fresh_conversation(tmp_path):
    """必测 2：种子种下后，新对话问名字（三种问法）和 SOUL.md，
    都找得到、都是已确认——不再说“未经确认”/“没有任何记录”；
    种子只加一次，再种全部 unchanged、不重复。"""
    store = make_fake_store(tmp_path)
    seeded = store.ensure_identity_memories()
    assert seeded["agent_name"]["status"] == "added"
    assert seeded["soul"]["status"] == "added"
    for question in ("你叫什么", "你叫什么名字", "你的名字是什么"):
        found = store.search(question)
        assert found, question
        assert any("十二" in item["text"] for item in found), question
        assert any(
            item["confirmed"] is True for item in found if "十二" in item["text"]
        ), question
    found = store.search("你的灵魂是什么")
    assert any("SOUL.md" in item["text"] for item in found)
    assert any(
        item["confirmed"] is True for item in found if "SOUL.md" in item["text"]
    )
    seeded_again = store.ensure_identity_memories()
    assert seeded_again["agent_name"]["status"] == "unchanged"
    assert seeded_again["soul"]["status"] == "unchanged"
    assert len(store._memory.all_texts()) == 2


def test_chat_naming_stores_user_confirmed_name(tmp_path):
    """必测 2 的另一半：BOOS 在聊天里亲口说“以后你叫十二了”，
    下一轮提交后名字以已确认存下来；再问一次名字，不重复存。"""
    store = make_fake_store(tmp_path)
    store.add_turn(
        [
            {"role": "user", "content": "以后你叫十二了。"},
            {"role": "assistant", "content": "好的，以后我就叫十二。"},
        ]
    )
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    texts = store._memory.all_texts()
    assert texts == ["Agent 的名字是十二，你叫十二"]
    item = store.get_all()[0]
    assert item["metadata"]["source"] == "user"
    assert item["metadata"]["confirmed"] is True
    assert item["metadata"]["slot"] == "agent_name"
    # 再说一遍名字：同一槽位同一值，不重复存。
    store.add_turn([{"role": "user", "content": "你叫十二。"}])
    store.flush_pending()
    assert store._memory.all_texts() == ["Agent 的名字是十二，你叫十二"]


def test_seed_does_not_overwrite_chat_renamed_name(tmp_path):
    """种子只加没有的：BOOS 在聊天里改了名字之后，下次启动再种，
    不能把名字改回去。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "现在你叫十三。"}])
    store.flush_pending()
    assert "十三" in " ".join(store._memory.all_texts())
    seeded = store.ensure_identity_memories()
    assert seeded["agent_name"]["status"] == "unchanged"
    assert not any("十二" in text for text in store._memory.all_texts())


def test_identity_seed_texts_are_the_confirmed_ones():
    """种子文本本身：名字和 SOUL.md 的内容就是准备稿里定的那两条。"""
    assert "十二" in AGENT_NAME_TEXT and "十一的弟弟" in AGENT_NAME_TEXT
    assert "SOUL.md" in AGENT_SOUL_TEXT and "十条原则" in AGENT_SOUL_TEXT


# ---------- 场景 3：玩笑 / 假设不存 ----------


def test_joke_and_hypothetical_not_stored(tmp_path):
    """必测 3：“我一百岁了”“如果我去了英国”都不存；同一轮里
    条件句之外的真事实照常存。

    离线说明：“我一百岁了”在假提取里本来就提不出来（没有事实前缀），
    真正的防线是 CUSTOM_INSTRUCTIONS 第 13 条（模型判断玩笑不存），
    真实 API 测试里验证那条；这里测的是代码层：条件句整句去掉、
    身份句不进提取、真事实不受影响。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我一百岁了。"}])
    store.add_turn(
        [{"role": "user", "content": "如果我去了英国，我就去看大本钟。我喜欢喝咖啡。"}]
    )
    store.flush_pending()
    texts = store._memory.all_texts()
    assert not any("一百岁" in text for text in texts)
    assert not any("英国" in text for text in texts)
    assert not any("大本钟" in text for text in texts)
    assert any("我喜欢喝咖啡" in text for text in texts)


def test_filter_extraction_messages_drops_conditionals_and_identity():
    """闸门 1 的单元测试：条件句、身份句不进提取，真事实留下来。"""
    filtered = _filter_extraction_messages(
        [
            {"role": "user", "content": "如果我去了英国，我就去看大本钟。我喜欢喝咖啡。我叫小宋。"},
            {"role": "assistant", "content": "好的。"},
        ]
    )
    user_texts = [m["content"] for m in filtered if m["role"] == "user"]
    assert user_texts == ["我喜欢喝咖啡。"]
    assert not any("英国" in text for text in user_texts)
    assert not any("小宋" in text for text in user_texts)


# ---------- 场景 4：纠正能覆盖旧值（同槽位 UPDATE，不并存） ----------


def test_rename_updates_same_slot_no_coexistence(tmp_path):
    """必测 4：改名是同槽位 UPDATE——“现在你叫十三”盖掉十二，
    库里只剩一条名字记忆；再改回十二，同样只剩一条。"""
    store = make_fake_store(tmp_path)
    store.ensure_identity_memories()
    store.add_turn([{"role": "user", "content": "现在你叫十三。"}])
    store.flush_pending()
    name_texts = [
        text for text in store._memory.all_texts() if "Agent 的名字是" in text
    ]
    assert len(name_texts) == 1
    assert "十三" in name_texts[0]
    assert not any("十二" in text for text in name_texts)
    store.add_turn([{"role": "user", "content": "现在你叫十二。"}])
    store.flush_pending()
    name_texts = [
        text for text in store._memory.all_texts() if "Agent 的名字是" in text
    ]
    assert len(name_texts) == 1
    assert "十二" in name_texts[0]


def test_rename_updates_legacy_unmarked_name_no_coexistence(tmp_path):
    """Plan C 之前存的旧名字记忆（没打 slot 标记）：改名时也要原地
    更新，不并存——槽位是从旧文本里认出来的，不是猜的。
    旧记忆用 add_memory_text 模拟（来源是 user、已确认，但没有 slot，
    就是 Plan C 之前存下来的样子）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("Agent 的名字是十二，你叫十二")
    legacy = store.get_all()[0]
    # 规整后的 metadata 总带 slot 键，没标记时是 None。
    assert (legacy["metadata"] or {}).get("slot") is None
    legacy_id = legacy["id"]
    store.add_turn([{"role": "user", "content": "现在你叫十三。"}])
    store.flush_pending()
    name_texts = [
        text for text in store._memory.all_texts() if "Agent 的名字是" in text
    ]
    assert len(name_texts) == 1
    assert "十三" in name_texts[0]
    assert not any("十二" in text for text in name_texts)
    item = store.get_all()[0]
    assert item["id"] == legacy_id  # 原地更新，id 不变
    assert item["metadata"]["slot"] == "agent_name"


def test_seed_marks_legacy_unmarked_name(tmp_path):
    """种子遇到 Plan C 之前存的同一槽位同一值旧记忆时，不新增，
    顺手给它打上槽位标记（文本不动）——以后改名走槽位找得到它。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("Agent 的名字是十二，你叫十二")
    seeded = store.ensure_identity_memories()
    assert seeded["agent_name"]["status"] == "unchanged"
    items = store.get_all()
    assert len(items) == 2  # 名字（旧的）+ SOUL（新种的），没重复
    name_item = [i for i in items if "十二" in i["text"]][0]
    assert name_item["metadata"]["slot"] == "agent_name"
    assert name_item["text"] == "Agent 的名字是十二，你叫十二"
    # 打过标记之后，改名走槽位原地更新，不并存。
    legacy_id = name_item["id"]
    store.add_turn([{"role": "user", "content": "现在你叫十三。"}])
    store.flush_pending()
    renamed = [i for i in store.get_all() if "十三" in i["text"]]
    assert len(renamed) == 1
    assert renamed[0]["id"] == legacy_id
    assert not any("十二" in t for t in store._memory.all_texts())


def test_rename_updates_legacy_unmarked_user_name(tmp_path):
    """称呼也一样：Plan C 之前存的“我叫小宋”（没打标记），
    改成“我叫大明”时原地更新，不并存。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我叫小宋")
    legacy = store.get_all()[0]
    assert (legacy["metadata"] or {}).get("slot") is None
    legacy_id = legacy["id"]
    store.add_turn([{"role": "user", "content": "我叫大明。"}])
    store.flush_pending()
    texts = store._memory.all_texts()
    assert len(texts) == 1
    assert "大明" in texts[0]
    assert not any("小宋" in t for t in texts)
    item = store.get_all()[0]
    assert item["id"] == legacy_id
    assert item["metadata"]["slot"] == "user_name"


def test_correction_of_name_keeps_slot(tmp_path):
    """纠错路径也不丢槽位：“不对，你叫十二。”走纠错改名之后，
    槽位标记还在，下次再改名仍是同槽位 UPDATE、不并存。"""
    store = make_fake_store(tmp_path)
    store.ensure_identity_memories()
    result = store.add_turn(
        [
            {"role": "user", "content": "不对，你叫十三。"},
            {"role": "assistant", "content": "好的，我叫十三。"},
        ]
    )
    assert result["correction"]["status"] == "updated"
    item = store.get_all()[0]
    assert item["metadata"]["slot"] == "agent_name"
    store.add_turn([{"role": "user", "content": "现在你叫十二。"}])
    store.flush_pending()
    name_texts = [
        text for text in store._memory.all_texts() if "Agent 的名字是" in text
    ]
    # 纠错改过的那条是“你叫十三。”（不是“Agent 的名字是…”句式），
    # 但槽位还在，upsert 按槽位找到它、原地改回十二，不并存。
    assert len([t for t in store._memory.all_texts() if "十二" in t]) == 1
    assert len([t for t in store._memory.all_texts() if "十三" in t]) == 0
    assert len(name_texts) <= 1


# ---------- 待定区本身：沉默不存、下一轮才提交 ----------


def test_pending_turn_not_stored_until_next_turn(tmp_path):
    """待定区：只有一轮对话、没有下一轮时，什么都不存；
    下一轮来了，上一轮才提交。"""
    store = make_fake_store(tmp_path)
    result = store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。"}])
    assert result.get("pending") is True
    assert store._memory.all_texts() == []
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    assert store._memory.all_texts() == ["我喜欢喝咖啡"]


def test_flush_pending_commits_last_turn(tmp_path):
    """flush_pending：主动提交待定区（测试用、退出时用），
    没待定内容时返回 {}。"""
    store = make_fake_store(tmp_path)
    assert store.flush_pending() == {}
    store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。"}])
    result = store.flush_pending()
    assert [item["memory"] for item in result.get("results", [])] == ["我喜欢喝咖啡"]
    assert store.flush_pending() == {}
    assert store._memory.all_texts() == ["我喜欢喝咖啡"]
