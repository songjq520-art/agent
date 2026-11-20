# tests/test_conflict_rules.py（记忆整理第 2 步：冲突裁决 + 两道闸）
# 测三件事：
#   - 第 1 步的排序键：(write_seq, created_at)，序号第一顺位——
#     时间更早、序号更大的那条更新（时钟再粗也不怕）。
#   - 补丁 2 两道闸：推断出的新说法盖不掉 BOOS 确认过的旧条（金额
#     同槽位新值也一样），新条挂起来待确认，旧条原封不动。
#   - 用户亲口说的（最高可信度）照常能顶掉旧条，闸不拦 BOOS 自己。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。

from conftest import make_fake_store
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    CONFIDENCE_USER,
    SOURCE_INFERRED,
    SOURCE_USER,
    _memory_order_key,
    _provenance,
    _resolve_conflict,
)


def _meta(source, confirmed, created_at, seq, created_at_source="written"):
    """拼一份测试用的 metadata（raw 形状，裁决内部会规整）。"""
    return {
        "source": source,
        "created_at": created_at,
        "created_at_source": created_at_source,
        "write_seq": seq,
        "confidence": CONFIDENCE_USER if confirmed else CONFIDENCE_INFERRED,
        "user_confirmed": confirmed,
        "slot": None,
    }


def _user_meta(created_at, seq):
    return _meta(SOURCE_USER, True, created_at, seq)


def _inferred_meta(created_at, seq):
    return _meta(SOURCE_INFERRED, False, created_at, seq)


# ---------- 第 1 步：排序键 (write_seq, created_at) ----------


def test_order_key_seq_beats_clock():
    """序号第一顺位：时间更早、序号更大的那条更新。

    Windows 时钟粒度粗，时间戳会撞、会乱——序号单调递增不会撞。
    旧顺序 (created_at, write_seq) 下这组会判反（时间晚的赢）。"""
    old = _inferred_meta("2026-11-10T10:00:02.000000Z", 5)
    new = _inferred_meta("2026-11-10T10:00:01.000000Z", 6)
    assert _memory_order_key(new) > _memory_order_key(old)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "new"
    assert result["action"] == "supersede_old"


def test_order_key_backfilled_still_oldest():
    """后补时间戳的还是按最旧算：序号再大也不顶用（补的时间是假的）。"""
    old = _meta(SOURCE_USER, True, "2026-11-10T10:00:00.000000Z", 99,
                created_at_source="backfilled")
    new = _inferred_meta("2026-11-10T10:00:00.000000Z", 1)
    assert _memory_order_key(old) < _memory_order_key(new)


def test_order_key_tie_asks_boos():
    """两条键全撞（都没序号、都没时间戳）：分不出先后，问 BOOS。"""
    old = _inferred_meta(None, None)
    new = _inferred_meta(None, None)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "ask"
    assert result["action"] == "ask"


# ---------- 补丁 2：两道闸（纯函数层） ----------


def test_gate_unconfirmed_cannot_overwrite_confirmed():
    """可信度闸：新记忆可信度≤1（推断）盖不掉旧的≥3（已确认）。"""
    old = _user_meta("2026-11-10T10:00:00.000000Z", 1)
    new = _inferred_meta("2026-11-10T10:00:01.000000Z", 2)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "old"
    assert result["action"] == "pending_new"


def test_gate_confirmed_can_overwrite_inferred():
    """反过来：已确认的新说法能顶掉推断的旧条。"""
    old = _inferred_meta("2026-11-10T10:00:00.000000Z", 1)
    new = _user_meta("2026-11-10T10:00:01.000000Z", 2)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "new"
    assert result["action"] == "supersede_old"


# ---------- 补丁 2：金额同槽位去重（集成，FakeMemory） ----------


def _seed_confirmed_money(store):
    """库里先有一条 BOOS 确认过的金额记忆：每天定投 5 块万家纳指。"""
    out = store.add_memory_text("每天定投 5 块万家纳指")
    assert out.get("results"), out
    return out["results"][0]["id"]


def _seed_inferred_money(store, text):
    """库里先有一条推断的金额记忆（提取来的，没确认过）。"""
    meta = _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    added = store._memory.add(
        [{"role": "user", "content": text}], infer=False, metadata=meta
    )
    return added["results"][0]["id"]


def _extracted_money_item(store, text):
    """模拟“提取刚存了一条金额新说法”：先原样写进假库，再交给去重。"""
    batch_meta = _provenance(
        SOURCE_INFERRED, CONFIDENCE_INFERRED, False, turn_id="turn-1"
    )
    added = store._memory.add(
        [{"role": "user", "content": text}], infer=False, metadata=batch_meta
    )
    new_id = added["results"][0]["id"]
    return new_id, batch_meta


def _texts(store):
    return {item["text"] for item in store.get_all()}


def _item_by_id(store, memory_id):
    for item in store.get_all():
        if item["id"] == memory_id:
            return item
    return None


def test_dedupe_money_gate_holds_inferred_over_confirmed(tmp_path):
    """提取的推断新值（10 块）盖不掉已确认的旧值（5 块）：

    旧条原封不动；新条留着，标待确认 + 冲突待审；结果里记一笔。"""
    store = make_fake_store(tmp_path)
    old_id = _seed_confirmed_money(store)
    before = store.get_all()
    new_id, batch_meta = _extracted_money_item(store, "每天定投 10 块万家纳指")
    out = store._dedupe_extraction_results(
        {"results": [{"id": new_id, "memory": "每天定投 10 块万家纳指",
                      "event": "ADD"}]},
        before,
        turn_id="turn-1",
        batch_metadata=batch_meta,
    )
    assert out.get("money_held_pending") == 1
    assert out.get("money_superseded") is None
    # 旧条没被顶掉：5 块还在，10 块是另外一条。
    assert "每天定投 5 块万家纳指" in _texts(store)
    assert "每天定投 10 块万家纳指" in _texts(store)
    old = _item_by_id(store, old_id)
    assert old["text"] == "每天定投 5 块万家纳指"
    new = _item_by_id(store, new_id)
    assert (new["metadata"] or {}).get("pending_confirmation") is True
    assert (new["metadata"] or {}).get("conflict_for_review") is True
    kept = [item for item in out["results"] if item.get("id") == new_id]
    assert kept and kept[0].get("held_pending") is True


def test_dedupe_money_gate_allows_inferred_over_inferred(tmp_path):
    """对照：旧条也是推断的，同级、新的更新——照常顶掉。"""
    store = make_fake_store(tmp_path)
    old_id = _seed_inferred_money(store, "每天定投 5 块万家纳指")
    before = store.get_all()
    new_id, batch_meta = _extracted_money_item(store, "每天定投 10 块万家纳指")
    out = store._dedupe_extraction_results(
        {"results": [{"id": new_id, "memory": "每天定投 10 块万家纳指",
                      "event": "ADD"}]},
        before,
        turn_id="turn-1",
        batch_metadata=batch_meta,
    )
    assert out.get("money_superseded") == 1
    assert out.get("money_held_pending") is None
    assert "每天定投 5 块万家纳指" not in _texts(store)
    assert _item_by_id(store, old_id)["text"] == "每天定投 10 块万家纳指"


def test_dedupe_money_gate_respects_coverage_guard_flag(tmp_path):
    """守卫已经标过待确认的那条，去重也不能翻过来顶掉旧条。

    旧条只是推断的（裁决本会放行），但守卫先标了待确认——去重照样挂起。"""
    store = make_fake_store(tmp_path)
    _seed_inferred_money(store, "每天定投 5 块万家纳指")
    before = store.get_all()
    new_id, batch_meta = _extracted_money_item(store, "每天定投 10 块万家纳指")
    out = store._dedupe_extraction_results(
        {"results": [{"id": new_id, "memory": "每天定投 10 块万家纳指",
                      "event": "ADD", "pending_confirmation": True}]},
        before,
        turn_id="turn-1",
        batch_metadata=batch_meta,
    )
    assert out.get("money_held_pending") == 1
    assert "每天定投 5 块万家纳指" in _texts(store)


# ---------- 补丁 2：真用上时再问 ----------


def test_search_flags_pending_only_when_reinforced(tmp_path):
    """真用上（reinforce=True）时，待确认的记忆在返回项上标出来；
    只是找找（reinforce=False）时不标——不打扰。"""
    store = make_fake_store(tmp_path)
    _seed_confirmed_money(store)
    store.add_memory_text("每天定投 10 块万家纳指", source="inferred")
    used = store.search("万家纳指", reinforce=True)
    held = [item for item in used if item["text"] == "每天定投 10 块万家纳指"]
    assert held, used
    assert held[0].get("needs_boos_decision") is True
    assert held[0].get("decision_reason")
    looked = store.search("万家纳指", reinforce=False)
    held_looked = [item for item in looked if item["text"] == "每天定投 10 块万家纳指"]
    assert held_looked, looked
    assert held_looked[0].get("needs_boos_decision") is None


def test_list_pending_decisions(tmp_path):
    """待定清单：只列挂起来的那条，已确认的旧条不在里面；按序号排。"""
    store = make_fake_store(tmp_path)
    _seed_confirmed_money(store)
    out = store.add_memory_text("每天定投 10 块万家纳指", source="inferred")
    new_id = out["results"][0]["id"]
    pending = store.list_pending_decisions()
    assert [entry["id"] for entry in pending] == [new_id]
    assert pending[0]["text"] == "每天定投 10 块万家纳指"
    assert "确认" in pending[0]["reason"]
    assert store.list_pending_decisions() == pending  # 只列不改，跑两遍一样



def test_add_memory_text_gate_holds_inferred_over_confirmed(tmp_path):
    """手动存：非用户来源的新值盖不掉已确认的旧值，另存一条待确认的。"""
    store = make_fake_store(tmp_path)
    old_id = _seed_confirmed_money(store)
    out = store.add_memory_text("每天定投 10 块万家纳指", source="inferred")
    assert out.get("held_pending") is True
    assert out.get("superseded") is None
    assert "每天定投 5 块万家纳指" in _texts(store)
    assert "每天定投 10 块万家纳指" in _texts(store)
    old = _item_by_id(store, old_id)
    assert old["text"] == "每天定投 5 块万家纳指"
    new_id = out["results"][0]["id"]
    new = _item_by_id(store, new_id)
    assert (new["metadata"] or {}).get("pending_confirmation") is True
    assert (new["metadata"] or {}).get("conflict_for_review") is True


def test_add_memory_text_user_still_supersedes(tmp_path):
    """对照：BOOS 亲口说的（默认 source=user）照常顶掉旧值，闸不拦。"""
    store = make_fake_store(tmp_path)
    old_id = _seed_confirmed_money(store)
    out = store.add_memory_text("每天定投 10 块万家纳指")
    assert out.get("superseded") is True
    assert out.get("held_pending") is None
    assert "每天定投 5 块万家纳指" not in _texts(store)
    assert _item_by_id(store, old_id)["text"] == "每天定投 10 块万家纳指"
