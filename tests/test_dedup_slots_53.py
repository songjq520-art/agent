# tests/test_dedup_slots_53.py（53 号：完善去重——同槽位/同主题先归并再判）
# 作用：验证金额记忆按（对象，周期）槽位归并：同槽位同金额是重复、
#       同槽位新金额是更新（旧条原地改成新说法）、不同槽位分别保留、
#       别只靠文本相似度（“5 块”和“10 块”相似度 0.934，比真重复还高，
#       拿去给模型判可能把新值当重复丢掉）。
# 全部离线：FakeMemory + 注入的提取，模型判那层离线没有 embedding、
#       本来就走“判不出来→不合并”；槽位归并是纯程序，不依赖模型。
# 关键回归（2026-11-12 实测，BOOS 定要修）：46 号去重是按相似度判的，
#       说法换得大的近义重复、同个话题换个角度的几条（定投的规则、问题、
#       场景各一条）还可能认不出来、照样存进去。

from conftest import make_fake_store
from memory.mem0_store import (
    _extract_money_metadata,
    _money_slot_relation,
)


def _texts(store):
    return store._memory.all_texts()


def _money_of(store, text):
    for item in store.get_all():
        if item["text"] == text:
            return (item.get("metadata") or {}).get("money")
    return None


# ---------- 槽位关系（单元级） ----------

# 53 号 5 条新说法（关系表）：新说法 vs 库里已有的，期望的关系。
SLOT_53_CASES = [
    # 同槽位新金额 → 更新。
    ("每天定投10块买万家纳指", "每天定投5块买万家纳指", "supersede"),
    # 同槽位同金额 → 重复。
    ("每天定投5块买万家纳指", "每天定投5块买万家纳指", "duplicate"),
    # 不同对象 → 分别保留。
    ("每天定投5刀的沪深300", "每天定投5块买万家纳指", "different"),
    # 同对象、不同周期 → 分别保留（每天一笔、每月一笔是两笔计划）。
    ("每月定投5块买万家纳指", "每天定投5块买万家纳指", "different"),
    # 没金额的 → 不归槽位管。
    ("我喜欢吃拉面", "每天定投5块买万家纳指", None),
]


def test_53_money_slot_relation_cases():
    """53 号：槽位关系表——同槽位新值更新、同值重复、不同槽位保留、
    没金额的不管。"""
    for new_text, known_text, wanted in SLOT_53_CASES:
        relation = _money_slot_relation(
            _extract_money_metadata(new_text),
            _extract_money_metadata(known_text),
        )
        assert relation == wanted, (new_text, known_text, relation)
    # 对象不明时不归槽位（不猜）：“每天定投5刀”不知道投哪只。
    assert _money_slot_relation(
        _extract_money_metadata("每天定投5刀"),
        _extract_money_metadata("每天定投5块买万家纳指"),
    ) is None


# ---------- 提取路：同槽位新值更新旧条，不留两条 ----------


def test_53_extraction_supersedes_same_slot_new_amount(tmp_path):
    """53 号：提取出“每天定投10块买万家纳指”时，库里已有的“每天定投
    5块买万家纳指”原地更新成新说法——库里只留一条（10块的），
    旧的5块不能和新的10块并存，也不能把新值当重复丢掉。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我的定投是每天 5 块买万家纳指。"}])
    store.add_turn([{"role": "user", "content": "今天天气怎么样？"}])
    assert len(_texts(store)) == 1
    assert "5 块" in _texts(store)[0]
    store.add_turn([{"role": "user", "content": "我现在每天定投10块买万家纳指。"}])
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    store.flush_pending()
    texts = _texts(store)
    assert len(texts) == 1, texts
    assert "10块" in texts[0] or "10 块" in texts[0]
    assert "5 块" not in texts[0]
    money = _money_of(store, texts[0])
    assert money is not None and money["amount"] == 10


def test_53_extraction_keeps_different_slots(tmp_path):
    """53 号：同主题不同槽位分别保留——万家纳指每天5块和沪深300每天
    5刀是两笔不同的定投，都得留着。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我的定投是每天 5 块买万家纳指。"}])
    store.add_turn([{"role": "user", "content": "今天天气怎么样？"}])
    store.add_turn([{"role": "user", "content": "我现在每天定投5刀的沪深300。"}])
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    store.flush_pending()
    texts = _texts(store)
    assert len(texts) == 2, texts
    assert any("万家纳指" in t for t in texts)
    assert any("沪深300" in t for t in texts)


def test_53_extraction_same_slot_same_amount_is_duplicate(tmp_path):
    """53 号：同槽位同金额是重复——换个说法再说一遍，不多存一条。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我的定投是每天 5 块买万家纳指。"}])
    store.add_turn([{"role": "user", "content": "今天天气怎么样？"}])
    store.add_turn([{"role": "user", "content": "我每天投 5 块买万家纳指。"}])
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    store.flush_pending()
    assert len(_texts(store)) == 1


# ---------- 直存路：同槽位新值更新旧条 ----------


def test_53_direct_store_supersedes_same_slot(tmp_path):
    """53 号：直存“每天定投10块买万家纳指”时，库里已有的5块那条
    原地更新成新说法，返回 superseded，库里只留一条。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("每天定投 5 块买万家纳指")
    result = store.add_memory_text("每天定投10块买万家纳指")
    assert result.get("superseded") is True
    texts = _texts(store)
    assert len(texts) == 1, texts
    assert "10块" in texts[0] or "10 块" in texts[0]
    money = _money_of(store, texts[0])
    assert money is not None and money["amount"] == 10


def test_53_direct_store_same_slot_same_amount_is_duplicate(tmp_path):
    """53 号：直存同槽位同金额是重复——返回 duplicate，不多存。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("每天定投 5 块买万家纳指")
    result = store.add_memory_text("每天定投5块买万家纳指")
    assert result.get("duplicate") is True
    assert len(_texts(store)) == 1


def test_53_direct_store_keeps_different_slots(tmp_path):
    """53 号：直存不同槽位分别保留——沪深300那笔是新计划，不是重复。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("每天定投 5 块买万家纳指")
    result = store.add_memory_text("每天定投5刀的沪深300")
    assert not result.get("duplicate")
    assert not result.get("superseded")
    assert len(_texts(store)) == 2
