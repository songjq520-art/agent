# tests/test_memory_timestamp.py（记忆整理第一步：时间戳 + 覆盖规则）
# 测三件事：时间戳够细不撞车、覆盖规则按“时间戳定新旧、可信度定资格”判、
# 提取后守卫拦得住“模型推断顶掉 BOOS 确认的事实”。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。

from datetime import datetime, timezone

from conftest import make_fake_store
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    CONFIDENCE_USER,
    SLOT_AGENT_NAME,
    SOURCE_INFERRED,
    SOURCE_USER,
    _credibility_rank,
    _is_core_memory,
    _memory_order_key,
    _normalize_metadata,
    _provenance,
    _resolve_conflict,
)


def _meta(source, confirmed, created_at, seq):
    """拼一份测试用的 metadata（raw 形状，_resolve_conflict 内部会规整）。"""
    return {
        "source": source,
        "created_at": created_at,
        "created_at_source": "written",
        "write_seq": seq,
        "confidence": CONFIDENCE_USER if confirmed else CONFIDENCE_INFERRED,
        "user_confirmed": confirmed,
        "slot": None,
    }


def _user_meta(created_at, seq):
    """BOOS 亲口确认的记忆（最高可信度）。"""
    return _meta(SOURCE_USER, True, created_at, seq)


def _inferred_meta(created_at, seq):
    """模型从对话里提取的记忆（最低档里的一档）。"""
    return _meta(SOURCE_INFERRED, False, created_at, seq)


# ---------- 时间戳 ----------


def test_now_iso_strictly_increases_when_clock_frozen(monkeypatch):
    """时钟钉死不动时，mem0_store._now_iso() 每次发的仍严格递增。

    和 conversation_summary 的 _now_iso 是同一套算法（PR #13 修过的
    那类问题：Windows 上粒度粗会撞时间），这里各自一份状态。
    """
    import memory.mem0_store as ms

    fixed = datetime(2026, 11, 10, 3, 0, 0, tzinfo=timezone.utc)

    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return fixed

    monkeypatch.setattr(ms, "datetime", FrozenClock)
    monkeypatch.setattr(ms, "_NOW_LAST", None, raising=False)
    values = [ms._now_iso() for _ in range(5)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)
    # 固定 6 位微秒、带 Z：字符串顺序就是时间顺序。
    for value in values:
        assert value.endswith("Z")
        assert len(value.rsplit(".", 1)[1]) == 7  # 6 位微秒 + Z


def test_provenance_assigns_distinct_timestamps_and_seq():
    """同一批连着调 _provenance，每条的 (时间戳, 序号) 都不一样、且递增。"""
    metas = [_provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False) for _ in range(5)]
    keys = [_memory_order_key(meta) for meta in metas]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    seqs = [meta["write_seq"] for meta in metas]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_normalize_metadata_keeps_new_fields():
    """规整 metadata 时，新字段（序号/时间来源/待确认）照实留着，脏值记成 None。"""
    raw = {
        "source": "user",
        "created_at": "2026-11-10T10:00:00.000001Z",
        "created_at_source": "backfilled",
        "write_seq": 7,
        "confidence": 1.0,
        "user_confirmed": True,
        "slot": None,
        "pending_confirmation": True,
    }
    meta = _normalize_metadata(raw)
    assert meta["write_seq"] == 7
    assert meta["created_at_source"] == "backfilled"
    assert meta["pending_confirmation"] is True
    dirty = {
        "source": "user",
        "created_at": "2026-11-10T10:00:00.000001Z",
        "created_at_source": "yesterday",
        "write_seq": "7",
        "confidence": 1.0,
        "user_confirmed": True,
        "pending_confirmation": "yes",
    }
    meta = _normalize_metadata(dirty)
    assert meta["write_seq"] is None
    assert meta["created_at_source"] is None
    assert meta["pending_confirmation"] is False


# ---------- 覆盖规则（纯函数） ----------


def test_resolve_conflict_same_level_newer_wins():
    """同级冲突：新的胜出，旧的标被取代。"""
    old = _inferred_meta("2026-11-10T10:00:00.000001Z", 1)
    new = _inferred_meta("2026-11-10T10:00:00.000002Z", 2)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "new"
    assert result["action"] == "supersede_old"


def test_resolve_conflict_same_level_stale_write_does_not_win():
    """同级但新记忆比旧的还早（迟到的旧写入）：不覆盖，旧的留着。"""
    old = _inferred_meta("2026-11-10T10:00:00.000002Z", 2)
    new = _inferred_meta("2026-11-10T10:00:00.000001Z", 1)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "old"
    assert result["action"] == "keep_old"


def test_resolve_conflict_inferred_cannot_overwrite_confirmed():
    """模型推断顶不掉 BOOS 确认的事实：旧的赢，新记忆挂待确认。"""
    old = _user_meta("2026-11-10T10:00:00.000001Z", 1)
    new = _inferred_meta("2026-11-10T10:00:00.000002Z", 2)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "old"
    assert result["action"] == "pending_new"


def test_resolve_conflict_confirmed_overwrites_inferred():
    """BOOS 亲口确认的新说法可以覆盖模型推断的旧记忆。"""
    old = _inferred_meta("2026-11-10T10:00:00.000001Z", 1)
    new = _user_meta("2026-11-10T10:00:00.000002Z", 2)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "new"
    assert result["action"] == "supersede_old"


def test_resolve_conflict_both_missing_timestamps_asks():
    """两条都没有时间戳：真分不出先后，问 BOOS。"""
    old = _inferred_meta(None, None)
    new = _inferred_meta(None, None)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "ask"
    assert result["action"] == "ask"


def test_resolve_conflict_identical_timestamp_and_seq_asks():
    """时间戳和序号全撞：真分不出先后，问 BOOS。"""
    old = _inferred_meta("2026-11-10T10:00:00.000001Z", 5)
    new = _inferred_meta("2026-11-10T10:00:00.000001Z", 5)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "ask"
    assert result["action"] == "ask"


def test_resolve_conflict_one_side_missing_timestamp_is_not_a_tie():
    """只有一边缺时间戳：缺的那条一定是时间戳功能上线之前存的，
    一定比新写的旧——这是时间线本身，不是猜，不问 BOOS。"""
    old = _inferred_meta(None, None)
    new = _inferred_meta("2026-11-10T10:00:00.000001Z", 5)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "new"
    assert result["action"] == "supersede_old"


def test_core_memory_never_overwritten_by_inference():
    """核心记忆（槽位 + 已确认）钉住不动，模型推断一律不能顶掉。"""
    old = _user_meta("2026-11-10T10:00:00.000001Z", 1)
    old["slot"] = SLOT_AGENT_NAME
    assert _is_core_memory(old) is True
    new = _inferred_meta("2026-11-10T10:00:00.000002Z", 2)
    result = _resolve_conflict(old, new)
    assert result["winner"] == "old"
    assert result["action"] == "pending_new"
    assert _credibility_rank(old) > _credibility_rank(new)


# ---------- 提取后守卫（端到端，SQLite 例子） ----------


def test_sqlite_guard_holds_inferred_change_as_pending(tmp_path):
    """SQLite 例子：先确认“数据库用 SQLite”，再来一条模型推断的
    “改成 PostgreSQL”——旧的必须还在，新推断只能待确认。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    assert len(before) == 1
    old_id = next(iter(before))
    old_created = before[old_id]["metadata"]["created_at"]
    # 第一轮只进待定区（Plan C），第二轮提交第一轮。
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    items = {item["text"]: item for item in store.get_all()}
    # 旧的确认记忆原封不动（文本、时间戳都没变）。
    assert "数据库用 SQLite。" in items
    assert items["数据库用 SQLite。"]["metadata"]["confirmed"] is True
    assert items["数据库用 SQLite。"]["metadata"]["pending_confirmation"] is False
    assert items["数据库用 SQLite。"]["metadata"]["created_at"] == old_created
    # 新的推断记忆存下来了，但标了待确认，不是正常事实。
    assert "我现在把数据库改成 PostgreSQL 了" in items
    new_meta = items["我现在把数据库改成 PostgreSQL 了"]["metadata"]
    assert new_meta["source"] == "inferred"
    assert new_meta["confirmed"] is False
    assert new_meta["pending_confirmation"] is True


def test_guard_ignores_unrelated_topic(tmp_path):
    """不同话题的新推断不拦：确认了 SQLite，聊拿铁照常存，不标待确认。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    store.add_turn([{"role": "user", "content": "我喜欢喝拿铁。"}])
    store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    items = {item["text"]: item for item in store.get_all()}
    assert "我喜欢喝拿铁" in items
    assert items["我喜欢喝拿铁"]["metadata"]["pending_confirmation"] is False


def test_guard_does_not_flag_siblings_in_same_turn(tmp_path):
    """同一批提取出的兄弟不互相认成邻居：快照在提取前拍，
    同一批里的两条不会因为关键词重合就互相标待确认。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我喜欢喝咖啡。我喜欢喝拿铁。"}])
    store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    items = {item["text"]: item for item in store.get_all()}
    assert "我喜欢喝咖啡" in items
    assert "我喜欢喝拿铁" in items
    assert items["我喜欢喝咖啡"]["metadata"]["pending_confirmation"] is False
    assert items["我喜欢喝拿铁"]["metadata"]["pending_confirmation"] is False


def test_user_remember_is_not_held_pending(tmp_path):
    """BOOS 亲口让记住的，不走守卫：可信度最高，不该被自己拦下来。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    store.add_memory_text("数据库改成 PostgreSQL。", source="user")
    items = {item["text"]: item for item in store.get_all()}
    assert "数据库改成 PostgreSQL。" in items
    assert items["数据库改成 PostgreSQL。"]["metadata"]["pending_confirmation"] is False
    assert items["数据库改成 PostgreSQL。"]["metadata"]["confirmed"] is True


# ---------- 槽位：同级新胜出，旧的留原话 ----------


def test_slot_rename_keeps_old_text_and_orders_by_timestamp(tmp_path):
    """改名是“同槽位就是同一件事”的地方：新值胜出，旧文本留审计，
    新记忆的时间戳比旧的晚。"""
    store = make_fake_store(tmp_path)
    added = store._upsert_slot_memory(SLOT_AGENT_NAME, "十三", "以后你叫十三。")
    assert added["status"] == "added"
    mem_id = added["memory_id"]
    old_meta = next(
        item["metadata"] for item in store.get_all() if item["id"] == mem_id
    )
    updated = store._upsert_slot_memory(SLOT_AGENT_NAME, "十二", "以后你叫十二。")
    assert updated["status"] == "updated"
    assert updated["memory_id"] == mem_id
    items = {item["id"]: item for item in store.get_all()}
    assert items[mem_id]["text"] == "以后你叫十二。"
    new_meta = items[mem_id]["metadata"]
    assert _memory_order_key(new_meta) > _memory_order_key(old_meta)
    assert _resolve_conflict(old_meta, new_meta)["action"] == "supersede_old"
    # 旧文本留审计（history 里有 UPDATE 行，旧文本还在）。
    history = store._memory.history(mem_id)
    assert any(entry.get("old_memory") == "以后你叫十三。" for entry in history)


# ---------- 老记忆补时间戳 ----------


def test_backfill_marks_old_memories_honestly(tmp_path):
    """没时间戳的老记忆补一次：时间戳补上了，标明是后补的，
    文本不动、来源不升级（没确认的还是没确认）。"""
    store = make_fake_store(tmp_path)
    store._memory.add(
        [{"role": "user", "content": "我喜欢看纪录片。"}],
        user_id=store.user_id,
        infer=False,
        metadata=None,
    )
    assert store.backfill_timestamps() == 1
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["text"] == "我喜欢看纪录片。"
    meta = items[0]["metadata"]
    assert meta["created_at"]
    assert meta["created_at_source"] == "backfilled"
    assert meta["source"] == "unconfirmed"
    assert meta["confirmed"] is False
    # 幂等：补过的不会再补。
    assert store.backfill_timestamps() == 0


def test_backfill_leaves_timestamped_memories_alone(tmp_path):
    """已经有时间戳的记忆，补时间戳时不动它。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    created = store.get_all()[0]["metadata"]["created_at"]
    assert store.backfill_timestamps() == 0
    assert store.get_all()[0]["metadata"]["created_at"] == created
