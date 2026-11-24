# tests/test_fx8prot_20261124.py（最后一轮修复，2026-11-24）
# 全部离线（向量打桩、判官打桩成不可用、探网打桩，不调真 API、不花钱）。
#
# 推断记忆保护（BOOS 裁决“类似的事情别再犯”）：推断来源、未确认
#    的旧记忆被亲口新事实顶掉之前，先攒待问问题，BOOS 点头才换；
#    说不换就两条都留、这对冲突已裁决不再问。亲口确认过的旧记忆
#    照旧立即顶掉（回归不误伤）。
# 第 8 条实时整理：save_turn 后本轮触碰到的记忆立刻后台整理
#    （晋升复核/打标/去重检查/集合内冲突扫描），不等定时调度；
#    整理失败不影响对话；不碰本轮没触碰的记忆。
# 错过补跑（catch-up，BOOS 补定）：维护自检到期遇断网——本地
#    部分照常跑，判官部分挂待跑、不记已跑、不推满周期；每
#    15 分钟探网，恢复立刻补跑完整自检。衰减是本地活，断网照跑。

import json

import pytest

from memory import decay as _decay
from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso
from web.server import MemoryMaintenanceScheduler


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词
    路），判官打桩成不可用（模型断了降级走程序，是既定生产模式）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _turn(store, text, turn_id):
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的"}],
        turn_id=turn_id,
    )


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _meta_of(store, memory_id):
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE id=?", (memory_id,)
    ).fetchone()
    return json.loads(row[0] or "{}")


def _plant_row(store, text, **meta_over):
    """绕过 add 入口直接插一行（模拟迁移/直存来的旧行），meta 可覆盖。"""
    created_at = _now_iso()
    key = _ledger.ledger_event_key(text, created_at)
    meta = {
        "source": "inferred", "confidence": 0.7, "user_confirmed": False,
        "created_at": created_at, "write_seq": 1, "turn_id": None,
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "episodic", "importance": "normal",
        "event_date": None,
    }
    meta.update(meta_over)
    store._insert_memory_row(key, text, meta, None)
    return key


def _conflict_questions(store):
    return [
        q for q in store.list_pending_questions()
        if q.get("kind") == "conflict"
    ]


# ---------- 推断记忆保护 ----------


def test_prot_inferred_main_target_held_and_question_queued(store):
    """点名纠正一条推断旧记忆：新说法照常存，旧的不顶，攒一条
    带新旧原文的待问问题（E 组广州事件的防复发主场景）。"""
    gz = _plant_row(store, "用户住在广州")
    outcome = store.correct_memory("用户住在广州", "用户住在西安")
    assert outcome["status"] == "pending_question", outcome
    assert "用户住在广州" in _live_texts(store)
    assert not _is_superseded(store, "用户住在广州")
    assert "用户住在西安" in _live_texts(store)
    qs = _conflict_questions(store)
    assert len(qs) == 1
    assert "用户住在广州" in qs[0]["text"]
    assert "用户住在西安" in qs[0]["text"]
    assert qs[0]["old_id"] == gz


def test_prot_direct_store_slot_path_also_asks(store):
    """直存路的槽位裁决（E 组事件的实际触发路）：新陈述存入时
    同槽只有一条推断旧记忆——不许当主目标直接顶，先问。"""
    _plant_row(store, "用户住在广州")
    result = store.add_memory_text(
        "用户住在西安", source="user", allow_slot_supersede=True
    )
    assert result.get("event") == "ADD"
    assert not _is_superseded(store, "用户住在广州")
    assert "用户住在广州" in _live_texts(store)
    assert "用户住在西安" in _live_texts(store)
    assert len(_conflict_questions(store)) == 1


def test_prot_answer_yes_supersedes(store):
    """BOOS 答“换成新的”：旧的才被顶掉（账本 supersede 标记），
    问题消耗掉、不再挂着。"""
    _plant_row(store, "用户住在广州")
    store.correct_memory("用户住在广州", "用户住在西安")
    q = _conflict_questions(store)[0]
    res = store.answer_question(q["id"], "换成新的")
    assert res["ok"] and res["applied"] == "superseded", res
    assert _is_superseded(store, "用户住在广州")
    assert "用户住在西安" in _live_texts(store)
    assert _conflict_questions(store) == []


def test_prot_answer_no_keeps_both_and_never_reasks(store):
    """BOOS 答“不换”：两条都留；同一对冲突再来一次也不再问
    （已裁决）；拿重要性问题的选项答冲突问题要被拒。"""
    _plant_row(store, "用户住在广州")
    store.correct_memory("用户住在广州", "用户住在西安")
    q = _conflict_questions(store)[0]
    bad = store.answer_question(q["id"], "进核心")
    assert bad["ok"] is False
    res = store.answer_question(q["id"], "不换")
    assert res["ok"] and res["applied"] == "kept_old", res
    assert "用户住在广州" in _live_texts(store)
    assert "用户住在西安" in _live_texts(store)
    assert not _is_superseded(store, "用户住在广州")
    store.correct_memory("用户住在广州", "用户住在西安")
    assert _conflict_questions(store) == []
    assert not _is_superseded(store, "用户住在广州")


@pytest.mark.requires_judge
def test_prot_confirmed_old_superseded_immediately(store):
    """回归不误伤：亲口确认过的旧记忆撞上纠正，照旧立即顶掉，
    不攒问题（保护只管推断未确认的）。"""
    _plant_row(
        store, "我住在武汉",
        source="user", confidence=1.0, user_confirmed=True,
        turn_id="t0",
    )
    outcome = store.correct_memory("我住在武汉", "我住在郑州")
    assert outcome["status"] == "updated", outcome
    assert _is_superseded(store, "我住在武汉")
    assert "我住在郑州" in _live_texts(store)
    assert _conflict_questions(store) == []


# ---------- 第 8 条：实时整理 ----------


def test_organize_promotes_and_tags_without_scheduler(store):
    """存量行被本轮触碰到（模拟引用计数）：整理立刻补槽位标、
    按出现集合复核晋升——不等每天/每周的定时调度。"""
    key = _plant_row(
        store, "我住在深圳",
        source="user", confidence=1.0, user_confirmed=True,
        seen_dates=_decay._dump_seen_list(["2026-11-20", "2026-11-21"]),
        seen_sessions=_decay._dump_seen_list(["s1", "s2"]),
    )
    summary = store._organize_turn_memories([key], turn_id="s3")
    assert key in summary["promoted"]
    assert key in summary["tagged"]
    meta = _meta_of(store, key)
    assert meta.get("decay_kind") == "stable"
    assert meta.get("slot")


def test_organize_failure_isolated_and_save_turn_ok(store, monkeypatch):
    """整理里单条炸了不许冒泡；save_turn 的后台整理炸了，
    本轮该存的照样存、回执照样回。"""
    def _boom(self, memory_id, slot):
        raise RuntimeError("打标炸了（测试打桩）")

    monkeypatch.setattr(SQLiteStore, "_set_memory_slot", _boom)
    key = _plant_row(
        store, "我住在深圳",
        source="user", confidence=1.0, user_confirmed=True,
    )
    summary = store._organize_turn_memories([key], turn_id="sx")
    assert summary["checked"] == 1

    def _boom_organize(self, ids, turn_id=None):
        raise RuntimeError("整理炸了（测试打桩）")

    monkeypatch.setattr(SQLiteStore, "_organize_turn_memories", _boom_organize)
    receipt = _turn(store, "我喜欢吃烤鸭", "t-org-1")
    assert "我喜欢吃烤鸭" in receipt["stored"]
    thread = getattr(store, "_turn_organize_thread", None)
    if thread is not None:
        thread.join(timeout=10)


def test_organize_leaves_untouched_alone(store):
    """整理只碰本轮集合：集合外的记忆一个字节都不许动。"""
    a = _plant_row(
        store, "我住在深圳",
        source="user", confidence=1.0, user_confirmed=True,
    )
    b = _plant_row(
        store, "我住在成都",
        source="user", confidence=1.0, user_confirmed=True,
    )
    before_b = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE id=?", (b,)
    ).fetchone()[0]
    summary = store._organize_turn_memories([a], turn_id="sz")
    assert a in summary["tagged"]
    after_b = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE id=?", (b,)
    ).fetchone()[0]
    assert after_b == before_b
    assert _meta_of(store, a).get("slot")


# ---------- 错过补跑（catch-up） ----------


class _CatchupStore:
    """维护调度测试打桩：记衰减次数和自检每次的判官开关。"""

    def __init__(self):
        self.decay_calls = 0
        self.check_calls = []

    def sweep_decay(self):
        self.decay_calls += 1
        return {"dormant": 0, "delete_due": 0}

    def run_self_check(self, include_model_judge=True):
        self.check_calls.append(include_model_judge)
        return {"summary": "ok"}


def _catchup_scheduler(store, state_path, net, **kw):
    args = {
        "decay_interval_seconds": 100000,
        "self_check_interval_seconds": 1000,
        "tick_seconds": 60,
        "network_probe": lambda: net["online"],
        "probe_interval_seconds": 900,
    }
    args.update(kw)
    return MemoryMaintenanceScheduler(store, state_path, **args)


def test_catchup_offline_defers_judge_then_catches_up(tmp_path):
    """到期 + 离线：本地部分先跑（判官关）、判官部分挂待跑，
    self_check_last_run 不许写（没记已跑、没推满周期）；探网
    间隔没到不重复探；网络一恢复立刻补跑完整自检并清待跑。"""
    state_path = tmp_path / "maintenance_state.json"
    state_path.write_text(
        json.dumps({"decay_last_run": 10000}), encoding="utf-8"
    )
    fake = _CatchupStore()
    net = {"online": False}
    probes = {"n": 0}

    def probe():
        probes["n"] += 1
        return net["online"]

    sched = MemoryMaintenanceScheduler(
        fake, state_path,
        decay_interval_seconds=100000, self_check_interval_seconds=1000,
        tick_seconds=60, network_probe=probe, probe_interval_seconds=900,
    )
    ran = sched.run_due(now=10000)
    assert fake.check_calls == [False]  # 只跑了本地部分（判官关）
    assert ran["self_check"] is False
    assert ran["self_check_local"] is True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["self_check_pending"] is True
    assert "self_check_last_run" not in state  # 没记已跑

    ran = sched.run_due(now=10500)  # 还离线、且没到下次探网（<900 秒）
    assert fake.check_calls == [False]
    assert probes["n"] == 1  # 没重复探

    net["online"] = True
    ran = sched.run_due(now=10900)  # 到了探网时间、网络恢复
    assert fake.check_calls == [False, True]  # 完整自检补跑（判官开）
    assert ran["self_check"] is True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["self_check_last_run"] == 10900
    assert "self_check_pending" not in state


def test_catchup_decay_runs_while_offline(tmp_path):
    """衰减是纯本地活：断网照常补跑，不受探网结果影响；
    自检没到期时不许因为离线凭空挂出待跑。"""
    fake = _CatchupStore()
    net = {"online": False}
    state_path = tmp_path / "maintenance_state.json"
    state_path.write_text(
        json.dumps({"self_check_last_run": 10000}), encoding="utf-8"
    )
    sched = _catchup_scheduler(
        fake, state_path, net,
        decay_interval_seconds=100, self_check_interval_seconds=100000,
    )
    ran = sched.run_due(now=10000)
    assert fake.decay_calls == 1
    assert fake.check_calls == []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "self_check_pending" not in state


def test_catchup_not_due_no_run(tmp_path):
    """未到期不跑：有网也不跑、不挂待跑（离线失败不改变到期
    判定的另一面——没到期就是没到期）。"""
    fake = _CatchupStore()
    net = {"online": True}
    state_path = tmp_path / "maintenance_state.json"
    state_path.write_text(
        json.dumps(
            {"decay_last_run": 10000, "self_check_last_run": 10000}
        ),
        encoding="utf-8",
    )
    sched = _catchup_scheduler(fake, state_path, net)
    ran = sched.run_due(now=10500)
    assert ran == {"decay": False, "self_check": False}
    assert fake.decay_calls == 0 and fake.check_calls == []


def test_catchup_online_normal_path(tmp_path):
    """在线正常路径回归：到期就完整跑一次（判官开）、记
    last_run、不留待跑标记。"""
    fake = _CatchupStore()
    net = {"online": True}
    state_path = tmp_path / "maintenance_state.json"
    sched = _catchup_scheduler(fake, state_path, net)
    ran = sched.run_due(now=20000)
    assert fake.check_calls == [True]
    assert ran["self_check"] is True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["self_check_last_run"] == 20000
    assert "self_check_pending" not in state
