# tests/test_write_seq_persistence_session.py（写序号持久化 + 会话计数，2026-11-19）
# 测两件事：
#   1. write_seq 持久化：启动时扫一遍库里最大的 write_seq 当起点，
#      重启（全新进程）后新序号接着往上数，不归零。
#   2. 会话计数：turn_id 从 save_turn/add_turn 一路带进 mem0 metadata
#      和账本 add 事件；出现集合（日期 + 会话）按规矩数——同一轮说几遍
#      只算一次，新日期且新会话才 +1，跨午夜但同一轮不算；两个集合都够
#      3 才晋升。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from conftest import FakeMemory, make_fake_store
from memory import ledger as _ledger
from memory.decay import (
    DECAY_KIND_EPISODIC,
    DECAY_KIND_STABLE,
    record_seen_occurrence,
    should_promote,
)
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    CONFIDENCE_USER,
    SOURCE_INFERRED,
    SOURCE_USER,
    _metadata_for_store,
    _normalize_metadata,
    _provenance,
)
from memory.timeutil import _next_write_seq, _seed_write_seq


@pytest.fixture
def _reset_write_seq(monkeypatch):
    """每个序号测试前把写序号归零、测完恢复——序号是进程内全局，
    不隔离会互相串味。"""
    import memory.timeutil as timeutil

    monkeypatch.setattr(timeutil, "_WRITE_SEQ", 0)


def _meta_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["metadata"]
    raise AssertionError(f"库里没有 {memory_id}")


def _add_episodic(store, text):
    """存一条模型提取的记忆（episodic），返回 id。"""
    result = store._add_text_with_metadata(
        text, _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    )
    return (result.get("results") or [{}])[0].get("id")


def _day_str(dt):
    return dt.date().isoformat()


def _at(day_iso, hour=12, minute=0):
    """某天几点（UTC），供 _record_decay_occurrence 的 now 用。"""
    return datetime.fromisoformat(f"{day_iso}T{hour:02d}:{minute:02d}:00+00:00")


# ---------- 1. write_seq 持久化 ----------


def test_seed_write_seq_only_goes_up(_reset_write_seq):
    """播种只往上抬、不往下压：小了、不是整数的都不动；
    发出去的序号绝不能重发。"""
    assert _seed_write_seq(5) == 5
    assert _next_write_seq() == 6
    assert _seed_write_seq(3) == 6
    assert _seed_write_seq("7") == 6
    assert _seed_write_seq(True) == 6
    assert _next_write_seq() == 7


def test_store_init_seeds_from_existing_memories(tmp_path, _reset_write_seq):
    """建 store 时扫库播种：库里最大的 write_seq 是 25，新序号从 26 起。"""
    fake = FakeMemory()
    fake._store("BOOS 爱吃冒菜。", metadata={"source": "user", "write_seq": 10})
    fake._store("BOOS 喜欢看纪录片。", metadata={"source": "inferred", "write_seq": 25})
    make_fake_store(tmp_path=tmp_path, memory=fake)
    assert _next_write_seq() == 26


def test_store_init_empty_library_starts_at_one(tmp_path, _reset_write_seq):
    """库是空的：播种不到东西，从 1 开始，和以前一样。"""
    make_fake_store(tmp_path=tmp_path)
    assert _next_write_seq() == 1


def test_restart_new_process_continues_seq(tmp_path):
    """模拟重启：全新进程里，序号接着库里最大的 write_seq 往上数，不归零。

    进程 1 先把一条 write_seq=41 的记忆写进假库、把库快照落盘，记下
    新序号；进程 2 从同一份快照重建假库、再建 store，新序号必须还是
    接着 41 往上数——两个全新进程都返回 1、2，就是没修好的老样子。"""
    repo_root = Path(__file__).resolve().parent.parent
    lib_path = tmp_path / "lib.json"
    workdir = tmp_path / "work"
    workdir.mkdir()
    setup = (
        "import json, sys; "
        f"sys.path.insert(0, r'{repo_root}'); "
        f"sys.path.insert(0, r'{repo_root / 'tests'}'); "
        "from conftest import FakeMemory, make_fake_store; "
        "from memory.timeutil import _next_write_seq; "
        "fake = FakeMemory(); "
        "fake._store('BOOS 爱吃冒菜。', metadata={'source': 'user', 'write_seq': 41}); "
        f"make_fake_store(tmp_path=r'{workdir}', memory=fake); "
        f"json.dump(fake._items, open(r'{lib_path}', 'w', encoding='utf-8')); "
        "print(_next_write_seq())"
    )
    first = subprocess.run(
        [sys.executable, "-c", setup],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert first.returncode == 0, first.stderr[-2000:]
    assert first.stdout.strip() == "42"
    rebuild = (
        "import json, sys; "
        f"sys.path.insert(0, r'{repo_root}'); "
        f"sys.path.insert(0, r'{repo_root / 'tests'}'); "
        "from conftest import FakeMemory, make_fake_store; "
        "from memory.timeutil import _next_write_seq; "
        f"items = json.load(open(r'{lib_path}', encoding='utf-8')); "
        "fake = FakeMemory(); "
        "[fake._store(it['memory'], metadata=it['metadata']) for it in items]; "
        f"make_fake_store(tmp_path=r'{workdir}', memory=fake); "
        "print(_next_write_seq())"
    )
    second = subprocess.run(
        [sys.executable, "-c", rebuild],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert second.returncode == 0, second.stderr[-2000:]
    # 全新进程，没播种会从 1 开始；播种了还是 42。
    assert second.stdout.strip() == "42"


# ---------- 2a. turn_id 进 metadata 和账本 ----------


def test_provenance_turn_id_round_trip():
    """_provenance 带 turn_id：metadata 里有它，出现集合里这轮也记上了；
    规整后还是列表，写回 Chroma 前变回 JSON 字符串（不是 list）。"""
    meta = _provenance(SOURCE_USER, CONFIDENCE_USER, True, turn_id="turn-9")
    assert meta["turn_id"] == "turn-9"
    assert json.loads(meta["decay_seen_sessions"]) == ["turn-9"]
    norm = _normalize_metadata(meta)
    assert norm["turn_id"] == "turn-9"
    assert norm["decay_seen_sessions"] == ["turn-9"]
    assert isinstance(norm["decay_seen_dates"], list)
    back = _metadata_for_store(norm)
    assert back["turn_id"] == "turn-9"
    assert back["decay_seen_sessions"] == '["turn-9"]'
    assert isinstance(back["decay_seen_dates"], str)


def test_provenance_without_turn_id_has_empty_sessions():
    """没传 turn_id：metadata 里没有它，会话集合是空的（老路子照常）。"""
    meta = _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    assert "turn_id" not in meta
    assert json.loads(meta["decay_seen_sessions"]) == []
    assert len(json.loads(meta["decay_seen_dates"])) == 1


def test_save_turn_records_turn_id(tmp_path):
    """turn_id 从 save_turn 一路带进 mem0 metadata 和账本 add 事件。

    save_turn（turn-abc）先进 outbox，下一轮 add_turn 提交待定区时
    提取存入——新记忆的 metadata 和账本 add 事件里都得是 turn-abc，
    不能丢成别的轮、也不能没了。"""
    store = make_fake_store(tmp_path=tmp_path)
    result = store.save_turn(
        [
            {"role": "user", "content": "我喜欢吃冒菜。"},
            {"role": "assistant", "content": "记下了，你喜欢吃冒菜。"},
        ],
        turn_id="turn-abc",
    )
    assert result.get("turn_id") == "turn-abc"
    store.add_turn(
        [
            {"role": "user", "content": "我今天想吃点辣的。"},
            {"role": "assistant", "content": "好，冒菜怎么样？"},
        ],
        turn_id="turn-def",
    )
    store.flush_pending()
    items = store.get_all(include_sensitive=True)
    assert len(items) == 1
    assert items[0]["text"] == "我喜欢吃冒菜"
    assert items[0]["metadata"]["turn_id"] == "turn-abc"
    events, _info = _ledger.read_events(store.ledger_path)
    adds = [
        event
        for event in events
        if event.get("type") == _ledger.LEDGER_EVENT_ADD
        and "冒菜" in (event.get("text") or "")
    ]
    assert len(adds) == 1
    assert (adds[0].get("meta") or {}).get("turn_id") == "turn-abc"


# ---------- 2b. 出现集合纯函数 ----------


def test_should_promote_helper():
    """晋升判断：3 天且 3 会话、不是核心才晋升；少一样就不够；
    JSON 字符串和列表都认。"""
    days3 = ["2026-11-13", "2026-11-14", "2026-11-15"]
    turns3 = ["turn-1", "turn-2", "turn-3"]
    assert should_promote(days3, turns3) is True
    assert should_promote(days3[:2], turns3) is False
    assert should_promote(days3, turns3[:2]) is False
    assert should_promote(days3, turns3, is_core=True) is False
    assert should_promote(json.dumps(days3), json.dumps(turns3)) is True
    assert should_promote(None, None) is False
    assert should_promote("不是 JSON", turns3) is False


def test_record_seen_occurrence_rules():
    """计数规矩：同一轮说几遍只算一次；跨午夜但同一轮不算；
    没传 turn_id 时只按日期数。"""
    # 同一轮说 3 遍：第一遍记上，后两遍什么都不加。
    dates, sessions, added = record_seen_occurrence([], [], "2026-11-13", "turn-1")
    assert (dates, sessions, added) == (["2026-11-13"], ["turn-1"], True)
    dates, sessions, added = record_seen_occurrence(dates, sessions, "2026-11-13", "turn-1")
    assert (dates, sessions, added) == (["2026-11-13"], ["turn-1"], False)
    # 跨午夜但同一轮：日期变了、会话没变，日期也不加。
    dates, sessions, added = record_seen_occurrence(dates, sessions, "2026-11-14", "turn-1")
    assert (dates, sessions, added) == (["2026-11-13"], ["turn-1"], False)
    # 新日期且新会话：两个都加。
    dates, sessions, added = record_seen_occurrence(dates, sessions, "2026-11-14", "turn-2")
    assert (dates, sessions, added) == (["2026-11-13", "2026-11-14"], ["turn-1", "turn-2"], True)
    # 没传 turn_id：只按日期数。
    dates, sessions, added = record_seen_occurrence(dates, sessions, "2026-11-15", None)
    assert (dates, sessions, added) == (
        ["2026-11-13", "2026-11-14", "2026-11-15"],
        ["turn-1", "turn-2"],
        True,
    )
    # 日期不是 YYYY-MM-DD：不猜，什么都不加。
    assert record_seen_occurrence(dates, sessions, "昨天", "turn-3")[2] is False


# ---------- 2c. 会话计数走完整流程 ----------


def test_same_turn_three_times_does_not_promote(tmp_path):
    """同一轮里说 3 遍只算一次：3 遍都是 turn-1，会话集合只有 1 个，
    不晋升。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_episodic(store, "BOOS 去盐湖城出差了。")
    day0 = _meta_of(store, memory_id)["decay_last_seen_day"]
    day1 = (datetime.fromisoformat(day0) + timedelta(days=1)).date().isoformat()
    for _ in range(3):
        assert store._record_decay_occurrence(memory_id, now=_at(day1), turn_id="turn-1") is True
    meta = _meta_of(store, memory_id)
    assert meta["decay_seen_dates"] == [day0, day1]
    assert meta["decay_seen_sessions"] == ["turn-1"]
    assert meta["decay_seen_days"] == 2
    assert meta["decay_kind"] == DECAY_KIND_EPISODIC


def test_three_days_three_turns_promotes(tmp_path):
    """三天三轮各一次：日期 3 个以上、会话 3 个，晋升成稳定事实，
    账本里记一条 PROMOTE 事件。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_episodic(store, "BOOS 去盐湖城出差了。")
    day0 = _meta_of(store, memory_id)["decay_last_seen_day"]
    base = datetime.fromisoformat(day0)
    for i, turn in enumerate(("turn-1", "turn-2", "turn-3"), start=1):
        day = (base + timedelta(days=i)).date().isoformat()
        assert store._record_decay_occurrence(memory_id, now=_at(day), turn_id=turn) is True
    meta = _meta_of(store, memory_id)
    assert len(meta["decay_seen_dates"]) == 4
    assert meta["decay_seen_sessions"] == ["turn-1", "turn-2", "turn-3"]
    assert meta["decay_kind"] == DECAY_KIND_STABLE
    events, _info = _ledger.read_events(store.ledger_path)
    promotes = [
        event
        for event in events
        if event.get("type") == _ledger.LEDGER_EVENT_PROMOTE
        and event.get("memory_id") == memory_id
    ]
    assert len(promotes) == 1


def test_midnight_same_turn_does_not_promote(tmp_path):
    """跨午夜但同一轮不算：23:30 提一次、00:30 又提一次，都是 turn-A，
    日期不加、会话不加，不晋升。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_episodic(store, "BOOS 去盐湖城出差了。")
    day0 = _meta_of(store, memory_id)["decay_last_seen_day"]
    day1 = (datetime.fromisoformat(day0) + timedelta(days=1)).date().isoformat()
    assert store._record_decay_occurrence(memory_id, now=_at(day0, 23, 30), turn_id="turn-A") is True
    assert store._record_decay_occurrence(memory_id, now=_at(day1, 0, 30), turn_id="turn-A") is True
    meta = _meta_of(store, memory_id)
    assert meta["decay_seen_dates"] == [day0]
    assert meta["decay_seen_sessions"] == ["turn-A"]
    assert meta["decay_seen_days"] == 1
    assert meta["decay_kind"] == DECAY_KIND_EPISODIC


def test_no_turn_still_promotes_by_days(tmp_path):
    """没传 turn_id 的老路子：只按不同天数晋升——3 个不同天就晋升，
    会话集合还是空的。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_episodic(store, "BOOS 去盐湖城出差了。")
    day0 = _meta_of(store, memory_id)["decay_last_seen_day"]
    base = datetime.fromisoformat(day0)
    for i in (1, 2):
        day = (base + timedelta(days=i)).date().isoformat()
        assert store._record_decay_occurrence(memory_id, now=_at(day)) is True
    meta = _meta_of(store, memory_id)
    assert len(meta["decay_seen_dates"]) == 3
    assert meta["decay_seen_sessions"] == []
    assert meta["decay_kind"] == DECAY_KIND_STABLE

# ---------- 2d. 强化按会话记（2026-11-19：turn_id 提前到检索之前） ----------


def test_reinforce_same_turn_twice_counts_once(tmp_path):
    """同一轮强化两次只算一次：两次 reinforce 都是 turn-1，
    会话集合只有 1 个，日期也只有创建那天。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_episodic(store, "BOOS 去盐湖城出差了。")
    assert store.reinforce_memories([memory_id], turn_id="turn-1")["reinforced"] == [memory_id]
    assert store.reinforce_memories([memory_id], turn_id="turn-1")["reinforced"] == [memory_id]
    meta = _meta_of(store, memory_id)
    assert meta["decay_seen_sessions"] == ["turn-1"]
    assert len(meta["decay_seen_dates"]) == 1
    assert meta["decay_kind"] == DECAY_KIND_EPISODIC


def test_reinforce_different_turns_counts_each(tmp_path):
    """不同轮强化各算一次：三天三轮各强化一次，日期 3 个、会话 3 个，
    晋升成稳定事实，账本里记一条 PROMOTE 事件。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_episodic(store, "BOOS 去盐湖城出差了。")
    day0 = _meta_of(store, memory_id)["decay_last_seen_day"]
    base = datetime.fromisoformat(day0)
    for i, turn in enumerate(("turn-1", "turn-2", "turn-3")):
        day = (base + timedelta(days=i)).date().isoformat()
        result = store.reinforce_memories([memory_id], now=_at(day), turn_id=turn)
        assert result["reinforced"] == [memory_id]
    meta = _meta_of(store, memory_id)
    assert len(meta["decay_seen_dates"]) == 3
    assert meta["decay_seen_sessions"] == ["turn-1", "turn-2", "turn-3"]
    assert meta["decay_kind"] == DECAY_KIND_STABLE
    events, _info = _ledger.read_events(store.ledger_path)
    promotes = [
        event
        for event in events
        if event.get("type") == _ledger.LEDGER_EVENT_PROMOTE
        and event.get("memory_id") == memory_id
    ]
    assert len(promotes) == 1
