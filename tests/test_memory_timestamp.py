# tests/test_memory_timestamp.py（记忆整理第一步：时间戳 + 覆盖规则）
# 测五件事：时间戳够细不撞车、覆盖规则按“时间戳定新旧、可信度定资格”判、
# 提取后守卫拦得住“模型推断顶掉 BOOS 确认的事实”、守卫按快照恢复被改动
# 的旧记忆（用会捣乱的假 mem0 模拟 add 里改、删旧记忆）、恢复和回滚做完
# 都要再读一遍核对（没报错不等于做完了），残留要写进结果和保护事件。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。

from datetime import datetime, timezone

from conftest import FakeMemory, _windows, make_fake_store
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


# ---------- 会捣乱的假 mem0：add 里改、删旧记忆 ----------


class MutatingFakeMemory(FakeMemory):
    """会捣乱的假 mem0：add 里会改、会删旧记忆。

    真实 mem0 2.2.1 读代码是只插不改，但说明书上写着 add 会决定
    增改删——守卫不能只靠“现在不会发生”。这个假对象模拟的就是
    说明书上那种行为：新推断存进来时，把同一话题的旧记忆改掉
    （mode="update"）或直接删掉（mode="delete"），看守卫能不能
    按提取前的快照把它恢复回来。
    同一话题的判定：和 FakeMemory.search 同一套两字窗口拆法，有
    一个窗口重合就算同一话题。只动本次 add 之前就有的旧记忆，
    同一批新存的兄弟不动。
    只在 infer=True（走提取）时捣乱：infer=False 是原文直存，
    真实 mem0 那条路也是只插不改——守卫恢复被删的旧记忆时走的
    就是 infer=False，不能让恢复本身再触发一次捣乱。
    """

    def __init__(self, mode="update"):
        super().__init__()
        self.mode = mode

    def add(self, messages, user_id=None, infer=True, metadata=None):
        old_ids = {item["id"] for item in self._items}
        result = super().add(messages, user_id=user_id, infer=infer, metadata=metadata)
        if not infer:
            return result
        for entry in result.get("results", []):
            new_text = entry.get("memory") or ""
            windows = _windows(new_text)
            if not windows:
                continue
            for item in list(self._items):
                if item["id"] not in old_ids:
                    continue
                if not (windows & _windows(item["memory"])):
                    continue
                if self.mode == "update":
                    self.update(memory_id=item["id"], text=new_text)
                elif self.mode == "delete":
                    self.delete(item["id"])
                break
        return result


class NoopMetadataUpdateFakeMemory(FakeMemory):
    """update 带 metadata 时假装成功、实际什么都不改。

    模拟“update 不报错但根本没恢复”：底层接口返回了成功，库里的
    文本和 metadata 动都没动。只拦带 metadata 的 update——恢复和
    打标记都带 metadata；捣乱假对象改旧记忆时不带，不受影响。
    noop_update_ids 不为空时，只对里面的 id 假装成功，其余的照常
    真改——用来模拟“多条旧记忆部分恢复：一半真恢复、一半悄悄失败”。
    """

    def __init__(self, noop_update_ids=None):
        super().__init__()
        self.noop_update_ids = set(noop_update_ids or [])

    def update(self, memory_id, text=None, metadata=None):
        if metadata is not None and (
            not self.noop_update_ids or memory_id in self.noop_update_ids
        ):
            self.update_calls.append((memory_id, text, metadata))
            return {"message": "Memory updated successfully!"}
        return super().update(memory_id, text=text, metadata=metadata)


class IncompleteMetadataFakeMemory(FakeMemory):
    """原文直存（infer=False）时把 metadata 存残。

    模拟“重新 add 不报错但 metadata 不完整”：add 返回了成功，文本
    存进去了，但 write_seq 丢了——读回来规整后对不上快照里的原样。
    只动恢复重存那条路：恢复传的是规整过的 metadata（带 confirmed
    键），种子和手动存传的是 _provenance 拼的（没有 confirmed 键），
    靠这个区分——种子那条正常存，不受影响。
    """

    def add(self, messages, user_id=None, infer=True, metadata=None):
        if not infer and metadata and "confirmed" in metadata:
            metadata = {k: v for k, v in metadata.items() if k != "write_seq"}
        return super().add(messages, user_id=user_id, infer=infer, metadata=metadata)


class FailDeleteMutatingFakeMemory(MutatingFakeMemory):
    """捣乱（add 里改旧记忆）+ delete 一律抛错。

    模拟“新记忆删除失败、仍然残留”：回滚删的时候抛错，记录还在。
    """

    def delete(self, memory_id):
        raise RuntimeError("delete 删不掉（测试注入的失败）。")


class NoopDeleteMutatingFakeMemory(MutatingFakeMemory):
    """捣乱（add 里改旧记忆）+ delete 假装成功、实际不删。

    模拟“delete 不报错但记录还在”：回滚删的时候返回了成功，记录
    还躺在库里。只靠返回值看不出来，只能靠删完再读一遍核对抓住。
    """

    def delete(self, memory_id):
        return {"deleted": True}


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


def test_guard_restores_old_memory_when_add_rewrites_it(tmp_path):
    """假 mem0 在 add 里把旧记忆改掉了，守卫要按快照恢复回来。

    模拟 mem0 说明书上那种行为：新推断存进来，旧的确认记忆在 add
    里面就被改写成新说法。断言：跑完提取保存，旧记忆原封不动
    （文本、时间戳、来源、确认状态都没变，id 也没变），新推断照样
    标待确认。只打标记、不恢复的守卫过不了这条——旧的已经被改了。
    """
    store = make_fake_store(tmp_path, memory=MutatingFakeMemory("update"))
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    assert len(before) == 1
    old_id = next(iter(before))
    old_created = before[old_id]["metadata"]["created_at"]
    # 第一轮只进待定区（Plan C），第二轮提交第一轮。
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    items = {item["text"]: item for item in store.get_all()}
    # 旧记忆被假 mem0 改掉过，但守卫按快照恢复了。
    assert "数据库用 SQLite。" in items
    restored = items["数据库用 SQLite。"]
    assert restored["id"] == old_id
    assert restored["metadata"]["created_at"] == old_created
    assert restored["metadata"]["source"] == "user"
    assert restored["metadata"]["confirmed"] is True
    assert restored["metadata"]["pending_confirmation"] is False
    # 新的推断记忆存下来了，但标了待确认。
    assert "我现在把数据库改成 PostgreSQL 了" in items
    new_meta = items["我现在把数据库改成 PostgreSQL 了"]["metadata"]
    assert new_meta["source"] == "inferred"
    assert new_meta["confirmed"] is False
    assert new_meta["pending_confirmation"] is True


def test_guard_restores_old_memory_when_add_deletes_it(tmp_path):
    """假 mem0 在 add 里把旧记忆删掉了，守卫要按快照重新存回来。

    删没了的恢复不回来旧 id（mem0 的 id 是它自己生成的），但文本、
    时间戳、来源、确认状态都要原样回来；新推断照样标待确认。
    """
    store = make_fake_store(tmp_path, memory=MutatingFakeMemory("delete"))
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    assert len(before) == 1
    old_created = before[next(iter(before))]["metadata"]["created_at"]
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    items = {item["text"]: item for item in store.get_all()}
    assert "数据库用 SQLite。" in items
    restored = items["数据库用 SQLite。"]
    assert restored["metadata"]["created_at"] == old_created
    assert restored["metadata"]["source"] == "user"
    assert restored["metadata"]["confirmed"] is True
    assert restored["metadata"]["pending_confirmation"] is False
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


def test_backfilled_memories_tie_and_ask():
    """两条后补的记忆分不出先后：排序键撞上，判成问 BOOS。

    后补的时间是补的时候随手给的“现在时间”，根本不知道历史上
    谁先谁后——排新旧时按没时间算，两条后补的键一样，就走问 BOOS
    那条，不能分出假胜负。
    """
    old = _user_meta("2026-11-10T12:00:00.000001Z", 1)
    old["created_at_source"] = "backfilled"
    new = _user_meta("2026-11-10T12:00:00.000002Z", 2)
    new["created_at_source"] = "backfilled"
    assert _memory_order_key(old) == _memory_order_key(new)
    assert _resolve_conflict(old, new)["action"] == "ask"


def test_backfilled_loses_to_written():
    """后补的对新写的，新写的赢：后补的本来就是功能上线前的老记忆，
    本来就比新写的旧，排在后面是对的。"""
    old = _user_meta("2026-11-10T12:00:00.000001Z", 1)
    old["created_at_source"] = "backfilled"
    new = _user_meta("2026-11-10T12:00:00.000002Z", 2)
    assert _resolve_conflict(old, new)["action"] == "supersede_old"


def test_backfilled_memories_have_no_order(tmp_path):
    """补完时间的两条老记忆，排序键一样：后补的先后是假的，不能拿来判。

    补的时候明明给了越来越新的时间和递增的序号，但排新旧时一律
    按没时间算——键撞上，真打架时就问 BOOS。
    """
    store = make_fake_store(tmp_path)
    store._memory.add(
        [{"role": "user", "content": "我喜欢看纪录片。"}],
        user_id=store.user_id,
        infer=False,
        metadata=None,
    )
    store._memory.add(
        [{"role": "user", "content": "我喜欢喝咖啡。"}],
        user_id=store.user_id,
        infer=False,
        metadata=None,
    )
    assert store.backfill_timestamps() == 2
    items = store.get_all()
    assert len(items) == 2
    assert all(
        item["metadata"]["created_at_source"] == "backfilled" for item in items
    )
    keys = [_memory_order_key(item["metadata"]) for item in items]
    assert keys[0] == keys[1]

# ---------- 失败关闭：快照失败不写、恢复失败回滚 ----------


def test_snapshot_retry_succeeds_on_second_try(tmp_path):
    """快照第一次读不出来、重试成功：照常写，不留保护记录。

    很多时候库只是抖一下，重试一次就好了。这条证明重试那一下
    真的发生了、写也没少：新推断存下来了、标了待确认、没留账。
    """
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    store._memory.get_all_failures = [RuntimeError("库抖了一下。")]
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    out = store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    assert "protection_failed" not in out
    items = {item["text"]: item for item in store.get_all()}
    assert "数据库用 SQLite。" in items
    assert "我现在把数据库改成 PostgreSQL 了" in items
    assert items["我现在把数据库改成 PostgreSQL 了"]["metadata"]["pending_confirmation"] is True
    assert store.list_protection_events() == []


def test_snapshot_both_attempts_fail_skips_write(tmp_path):
    """快照两次都读不出来：本轮不写，mem0.add 没被调，跳过有账可查。

    快照是守卫的眼睛：读不出来就不知道旧记忆长什么样，写了也
    没法保护，干脆不写。断言三点：add 没被调（add_calls 里没有
    infer=True 那条）、新说法没进库、保护事件记下来了。
    """
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    store._memory.get_all_failures = [
        RuntimeError("库读不出来。"),
        RuntimeError("库还是读不出来。"),
    ]
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    out = store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    assert out.get("protection_failed") is True
    assert out.get("protection_reason") == "snapshot_failed"
    # add 没被调：add_calls 里只有种子那条 infer=False，没有写新记忆那条。
    assert all(infer is not True for (_, _, infer) in store._memory.add_calls)
    items = {item["text"]: item for item in store.get_all()}
    assert "我现在把数据库改成 PostgreSQL 了" not in items
    assert "数据库用 SQLite。" in items
    events = store.list_protection_events()
    assert len(events) == 1
    assert events[0]["kind"] == "snapshot_failed"


def test_restore_unverified_rolls_back_new_memories(tmp_path):
    """恢复没法核对（库读不出来）：新存的几条全删掉回滚。

    假 mem0 在 add 里把旧的 SQLite 改成了 PostgreSQL，守卫想按
    快照恢复，可库读不出来、根本没法核对——保护状态未知，不能
    当成成功。本轮新存的几条全删掉，库回到写之前那个样子。
    旧的 SQLite 文本没恢复回来（恢复没法核对），这点如实记在
    账里、不是假装恢复了。
    """
    store = make_fake_store(tmp_path, memory=MutatingFakeMemory("update"))
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    old_id = next(iter(before))
    # 第一次读（快照）正常，第二次读（恢复时核对）读不出来。
    store._memory.get_all_failures = [None, RuntimeError("库读不出来。")]
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    out = store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    assert out.get("protection_failed") is True
    assert out.get("protection_reason") == "restore_unverified"
    items = store.get_all()
    # 新存的那条被删掉了：库里只有一条 PostgreSQL 意思的记忆，
    # 就是被假 mem0 改过的那条旧的（id 还是旧的）。
    pg_items = [item for item in items if item["text"] == "我现在把数据库改成 PostgreSQL 了"]
    assert len(pg_items) == 1
    assert pg_items[0]["id"] == old_id
    # 旧的 SQLite 文本没恢复回来（恢复没法核对，如实保留现状）。
    assert all(item["text"] != "数据库用 SQLite。" for item in items)
    events = store.list_protection_events()
    assert len(events) == 1
    assert events[0]["kind"] == "restore_unverified"
    assert "数据库用 SQLite。" in events[0]["failed_texts"]


def test_restore_partial_rolls_back_only_same_topic(tmp_path):
    """恢复个别失败：只回滚跟没恢复回来的旧记忆同一话题的新记忆。

    同一轮里两条新记忆：PostgreSQL（跟旧的 SQLite 同一话题）
    和拿铁（不同话题）。假 mem0 把旧的 SQLite 改了，守卫恢复时
    写不进去——PostgreSQL 那条新记忆删掉回滚，拿铁照常留下、
    不标待确认；结果和账里都写明是哪条旧记忆没恢复回来。
    """
    store = make_fake_store(tmp_path, memory=MutatingFakeMemory("update"))
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    old_id = next(iter(before))
    # 恢复时写不进去：守卫原地 update 旧的 SQLite 那条就抛错。
    store._memory.fail_update_ids = {old_id}
    store.add_turn(
        [{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。我喜欢喝拿铁。"}]
    )
    out = store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    assert out.get("protection_failed") is True
    assert out.get("protection_reason") == "restore_partial"
    assert out.get("failed_ids") == [old_id]
    items = store.get_all()
    # PostgreSQL 那条新存的被删掉了：只剩被假 mem0 改过的旧的那条。
    pg_items = [item for item in items if item["text"] == "我现在把数据库改成 PostgreSQL 了"]
    assert len(pg_items) == 1
    assert pg_items[0]["id"] == old_id
    # 拿铁照常留下，不标待确认。
    latte_items = [item for item in items if item["text"] == "我喜欢喝拿铁"]
    assert len(latte_items) == 1
    assert latte_items[0]["metadata"]["pending_confirmation"] is False
    # 旧的 SQLite 文本没恢复回来（恢复写不进去，如实保留现状）。
    assert all(item["text"] != "数据库用 SQLite。" for item in items)
    events = store.list_protection_events()
    assert len(events) == 1
    assert events[0]["kind"] == "restore_partial"
    assert events[0]["failed_ids"] == [old_id]
    assert "数据库用 SQLite。" in events[0]["failed_texts"]


def test_restore_result_tells_verified_and_failed(tmp_path):
    """_restore_snapshot_memories 的返回值要说清楚三种情况。

    没什么要恢复时：verified=True、restored=0、failed_ids 为空；
    读库失败时：verified=False（不是 -1、不是个数，调用方分得
    出“没核对上”和“核对上了但没什么可恢复”）。
    """
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = store.get_all()
    assert store._restore_snapshot_memories(before) == {
        "verified": True,
        "restored": 0,
        "failed_ids": [],
    }
    store._memory.get_all_failures = [RuntimeError("库读不出来。")]
    assert store._restore_snapshot_memories(before) == {
        "verified": False,
        "restored": 0,
        "failed_ids": [],
    }


# ---------- 恢复/回滚核对：没报错不等于做完了 ----------


def test_restore_verify_catches_update_that_did_nothing(tmp_path):
    """update 不报错但根本没恢复：核对要抓住它，标失败。

    底层 update 返回了成功，实际文本和 metadata 动都没动——以前
    “没报错就数成功”会把它算成恢复了。现在恢复完读回来比对，
    对不上就进 failed_ids；verified 还是 True（核对本身成功了，
    是恢复没成）。
    """
    store = make_fake_store(tmp_path, memory=NoopMetadataUpdateFakeMemory())
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = store.get_all()
    old_id = before[0]["id"]
    # 假装 add 里把旧记忆改了：不带 metadata 改文本（捣乱的口径）。
    store._memory.update(memory_id=old_id, text="我现在把数据库改成 PostgreSQL 了。")
    result = store._restore_snapshot_memories(before)
    assert result == {"verified": True, "restored": 0, "failed_ids": [old_id]}
    # 旧记忆还是被改过的样子（没恢复回来，如实保留现状）。
    items = {item["id"]: item for item in store.get_all()}
    assert items[old_id]["text"] == "我现在把数据库改成 PostgreSQL 了。"


def test_restore_verify_catches_readd_with_incomplete_metadata(tmp_path):
    """重新 add 不报错但 metadata 不完整：核对要抓住它，标失败。

    旧记忆被删了，恢复时重新存了一条，add 返回了成功、文本也对，
    但 write_seq 丢了——读回来规整后对不上快照里的原样。以前只看
    add 有没有报错，会把它算成恢复了。
    """
    store = make_fake_store(tmp_path, memory=IncompleteMetadataFakeMemory())
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = store.get_all()
    old_id = before[0]["id"]
    old_seq = before[0]["metadata"]["write_seq"]
    assert old_seq is not None
    # 假装 add 里把旧记忆删了。
    store._memory.delete(old_id)
    result = store._restore_snapshot_memories(before)
    assert result == {"verified": True, "restored": 0, "failed_ids": [old_id]}
    # 那条是重新存回来了，但 metadata 不全（write_seq 没了）。
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["text"] == "数据库用 SQLite。"
    assert items[0]["metadata"]["write_seq"] is None


def test_restore_verify_read_failure_marks_unverified(tmp_path):
    """恢复后核对读库失败：verified=False，不能算恢复成功。

    update 写进去了，但核对那一下读不出来——到底恢复没恢复根本
    不知道，不能返回 verified=True。failed_ids 是本轮试着恢复的 id。
    """
    store = make_fake_store(tmp_path)
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = store.get_all()
    old_id = before[0]["id"]
    store._memory.update(memory_id=old_id, text="我现在把数据库改成 PostgreSQL 了。")
    # 第一次读（恢复前核对）正常，第二次读（恢复后核对）读不出来。
    store._memory.get_all_failures = [None, RuntimeError("库读不出来。")]
    result = store._restore_snapshot_memories(before)
    assert result == {"verified": False, "restored": 0, "failed_ids": [old_id]}


def test_restore_verify_read_failure_rolls_back_new_memories(tmp_path):
    """恢复后核对读库失败：走没法核对那支，本轮新存的全删掉回滚。

    端到端：假 mem0 在 add 里把旧的改了，守卫恢复时写进去了，但
    核对那一下读不出来——恢复到底成没成不知道，按失败关闭：新存
    的几条全删掉回滚，add_turn 返回里标 protection_failed；回滚
    删干净了，事件里写“已删掉回滚”是真的，没有残留要报。
    注意：恢复的 update 本身是写进去了的（旧记忆改回了 SQLite，
    只是核对读不到、没法确认），所以库里最后是 SQLite 在、
    新存的 PostgreSQL 不在——不是旧记忆还坏着。
    """
    store = make_fake_store(tmp_path, memory=MutatingFakeMemory("update"))
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    old_id = next(iter(before))
    # 快照正常、恢复前核对正常，恢复后核对读不出来。
    store._memory.get_all_failures = [None, None, RuntimeError("库读不出来。")]
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    out = store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    assert out.get("protection_failed") is True
    assert out.get("protection_reason") == "restore_unverified"
    assert "rollback_leftover_ids" not in out
    items = store.get_all()
    # 旧记忆改回了 SQLite（update 写进去了，只是没法核对确认）。
    assert "数据库用 SQLite。" in {item["text"] for item in items}
    # 新存的那条被删掉回滚了：PostgreSQL 不在库里了。
    assert all(item["text"] != "我现在把数据库改成 PostgreSQL 了" for item in items)
    events = store.list_protection_events()
    assert len(events) == 1
    assert events[0]["kind"] == "restore_unverified"
    assert "已删掉回滚" in events[0]["reason"]
    assert "rollback_leftover_ids" not in events[0]


def test_rollback_delete_failure_leaves_visible_leftover(tmp_path):
    """新记忆删除失败、仍然残留：残留要记下来，不能写成“已回滚”。

    端到端：恢复没法核对（库读不出来），本轮新存的全删掉回滚，
    可 delete 抛错、新记忆还在库里。add_turn 返回里要有
    rollback_leftover_ids，保护事件里要有残留的 id 和原文，
    事件文案不能写“已删掉回滚”。
    """
    store = make_fake_store(tmp_path, memory=FailDeleteMutatingFakeMemory("update"))
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    old_id = next(iter(before))
    # 第一次读（快照）正常，第二次读（恢复前核对）读不出来。
    store._memory.get_all_failures = [None, RuntimeError("库读不出来。")]
    store.add_turn([{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。"}])
    out = store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    assert out.get("protection_failed") is True
    assert out.get("protection_reason") == "restore_unverified"
    leftover_ids = out.get("rollback_leftover_ids") or []
    assert len(leftover_ids) == 1
    # 残留的那条就是新存的 PostgreSQL，旧的 id 不是它。
    assert leftover_ids[0] != old_id
    items = {item["id"]: item for item in store.get_all()}
    assert leftover_ids[0] in items
    assert items[leftover_ids[0]]["text"] == "我现在把数据库改成 PostgreSQL 了"
    events = store.list_protection_events()
    assert len(events) == 1
    assert events[0]["kind"] == "restore_unverified"
    assert events[0].get("rollback_leftover_ids") == leftover_ids
    assert "我现在把数据库改成 PostgreSQL 了" in (
        events[0].get("rollback_leftover_texts") or []
    )
    assert "已删掉回滚" not in events[0]["reason"]
    assert "没删掉" in events[0]["reason"]


def test_restore_partial_success_and_failure(tmp_path):
    """多条旧记忆部分恢复：成功的数成恢复，失败的进 failed_ids。

    两条旧记忆都被改了，恢复时一条真写进去了、一条 update 不报错
    但悄悄没改——以前“没报错就数成功”会把两条都算成恢复了，现在
    核对读回来比对，成功的数 1 条，悄悄失败的那条进 failed_ids，
    verified 还是 True（核对本身成功了）。
    """
    store = make_fake_store(tmp_path, memory=NoopMetadataUpdateFakeMemory())
    store.add_memory_text("数据库用 SQLite。", source="user")
    store.add_memory_text("我喜欢喝咖啡。", source="user")
    before = store.get_all()
    ids = {item["text"]: item["id"] for item in before}
    old_sqlite = ids["数据库用 SQLite。"]
    old_coffee = ids["我喜欢喝咖啡。"]
    # 两条都被改了（假装 add 里动了旧记忆）。
    store._memory.update(memory_id=old_sqlite, text="我现在把数据库改成 PostgreSQL 了。")
    store._memory.update(memory_id=old_coffee, text="我现在把咖啡改成拿铁了。")
    # 恢复时咖啡那条 update 不报错、但悄悄没改。
    store._memory.noop_update_ids = {old_coffee}
    result = store._restore_snapshot_memories(before)
    assert result == {"verified": True, "restored": 1, "failed_ids": [old_coffee]}
    items = {item["id"]: item for item in store.get_all()}
    assert items[old_sqlite]["text"] == "数据库用 SQLite。"
    assert items[old_coffee]["text"] == "我现在把咖啡改成拿铁了。"


def test_rollback_delete_noop_leaves_visible_leftover(tmp_path):
    """delete 不报错但记录还在：回滚核对要抓住它。

    端到端：恢复个别失败，只回滚跟没恢复回来的旧记忆同一话题的
    新记忆；可 delete 返回了成功、记录还躺在库里。光看返回值
    看不出来，删完再读一遍才抓住：add_turn 返回里要有
    rollback_leftover_ids，保护事件里要有残留，拿铁那条照常留下。
    """
    store = make_fake_store(tmp_path, memory=NoopDeleteMutatingFakeMemory("update"))
    store.add_memory_text("数据库用 SQLite。", source="user")
    before = {item["id"]: item for item in store.get_all()}
    old_id = next(iter(before))
    # 恢复时写不进去：守卫原地 update 旧的 SQLite 那条就抛错。
    store._memory.fail_update_ids = {old_id}
    store.add_turn(
        [{"role": "user", "content": "我现在把数据库改成 PostgreSQL 了。我喜欢喝拿铁。"}]
    )
    out = store.add_turn([{"role": "user", "content": "今天天气不错。"}])
    assert out.get("protection_failed") is True
    assert out.get("protection_reason") == "restore_partial"
    assert out.get("failed_ids") == [old_id]
    leftover_ids = out.get("rollback_leftover_ids") or []
    assert len(leftover_ids) == 1
    items = {item["id"]: item for item in store.get_all()}
    assert leftover_ids[0] in items
    assert items[leftover_ids[0]]["text"] == "我现在把数据库改成 PostgreSQL 了"
    # 拿铁照常留下，不标待确认。
    latte_items = [item for item in items.values() if item["text"] == "我喜欢喝拿铁"]
    assert len(latte_items) == 1
    assert latte_items[0]["metadata"]["pending_confirmation"] is False
    events = store.list_protection_events()
    assert len(events) == 1
    assert events[0]["kind"] == "restore_partial"
    assert events[0].get("rollback_leftover_ids") == leftover_ids
    assert "我现在把数据库改成 PostgreSQL 了" in (
        events[0].get("rollback_leftover_texts") or []
    )
    assert "已删掉回滚" not in events[0]["reason"]
    assert "没删掉" in events[0]["reason"]
