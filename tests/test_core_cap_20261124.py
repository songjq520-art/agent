# tests/test_core_cap_20261124.py（四件套第 1 块：核心上限）
#
# 覆盖契约「一、核心契约」的本轮口径：
#   - 占座只数「有效核心且非钉死」：钉死/临时钉死不占座、
#     未确认/推断行不是成员不占座；
#   - 五类抽屉分账：分块合计 + 未入格 == 总座位（对账）；
#     分类字段缺失/标错的行按未入格计入总量、不进分块；
#   - 容量门：满座时 confirm 晋升被拒、摘档被拒，都攒
#     core_capacity 待问且不重复攒；回答「先留作候选」只
#     消耗问题，「不要了」走批准删除；
#   - 座位可由账本事件全量重算，与库内快照逐项相等；
#   - 水位线纯函数：120 条亮水位、比例、140 回差目标。

import pytest

from memory import core_cap
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _next_write_seq, _now_iso


@pytest.fixture
def store(tmp_path, monkeypatch):
    """干净的离线 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _add_user(store, text, **kwargs):
    saved = store.add_memory_text(text, source="user", **kwargs)
    assert saved.get("event") == "ADD", saved
    return saved["id"]


def _add_unconfirmed_user_row(store, text):
    """种一行 source=user 但还没逐条确认的活记忆（迁移/导入的形状）：
    还不是核心成员，confirm_memory 确认后才占座——容量门的考题。"""
    meta = {
        "source": "user",
        "confidence": 1.0,
        "user_confirmed": False,
        "created_at": _now_iso(),
        "write_seq": _next_write_seq(),
        "turn_id": None,
        "slot": None,
        "is_rule": False,
        "curated": False,
        "decay_kind": "episodic",
        "importance": "normal",
        "event_date": None,
        "last_used_at": None,
        "decay_state": "verbatim",
    }
    key = store._append_new_memory(text, meta)
    assert key
    return key


def _capacity_questions(store):
    return [
        q for q in store.list_pending_questions()
        if q.get("kind") == "core_capacity"
    ]


# ---------- 占座口径 ----------

def test_pinned_and_temp_pinned_do_not_occupy_seats(store):
    a = _add_user(store, "条目甲：长期事实一")
    b = _add_user(store, "条目乙：长期事实二")
    _add_user(store, "条目丙：长期事实三")
    assert store.set_protection_tier(a, "pinned")["ok"] is True
    assert store.set_protection_tier(b, "temp_pinned")["ok"] is True
    snap = store.core_seat_snapshot()
    assert snap["members"] == 3
    assert snap["pinned"] == 1
    assert snap["temp_pinned"] == 1
    assert snap["seats_used"] == 1
    assert snap["limit"] == 150
    assert snap["reconciled"] is True


def test_non_members_do_not_occupy_seats(store):
    store.add_memory_text("条目丁：推断来的事", source="inferred")
    _add_unconfirmed_user_row(store, "条目戊：还没确认的事")
    snap = store.core_seat_snapshot()
    assert snap["members"] == 0
    assert snap["seats_used"] == 0


def test_born_pinned_via_kwargs(store):
    _add_user(store, "条目己：生来钉住的", protection_tier="pinned")
    snap = store.core_seat_snapshot()
    assert snap["pinned"] == 1
    assert snap["seats_used"] == 0


# ---------- 五类分账对账 ----------

def test_drawer_breakdown_reconciles_with_total(store):
    _add_user(store, "条目一：本人的事", core_drawer="person")
    _add_user(store, "条目二：家人的事", core_drawer="people")
    _add_user(store, "条目三：又一件家人的事", core_drawer="people")
    _add_user(store, "条目四：没入格的事")
    _add_user(store, "条目五：乱标格的事", core_drawer="bogus")
    snap = store.core_seat_snapshot()
    assert snap["seats_used"] == 5
    assert snap["drawers"]["person"] == 1
    assert snap["drawers"]["people"] == 2
    # 没标的和标错的都按未入格：计入总量、不进分块
    assert snap["unclassified_seats"] == 2
    assert (
        sum(snap["drawers"].values()) + snap["unclassified_seats"]
        == snap["seats_used"]
    )
    assert snap["reconciled"] is True


def test_drawer_change_moves_between_buckets(store):
    a = _add_user(store, "条目一：先没入格", core_drawer=None)
    assert store.core_seat_snapshot()["unclassified_seats"] == 1
    assert store.set_core_drawer(a, "environment")["ok"] is True
    snap = store.core_seat_snapshot()
    assert snap["drawers"]["environment"] == 1
    assert snap["unclassified_seats"] == 0
    assert snap["seats_used"] == 1


# ---------- 容量门 ----------

def test_capacity_gate_blocks_confirm_when_full(store):
    store.core_seat_limit = 2
    _add_user(store, "条目一：占座一")
    _add_user(store, "条目二：占座二")
    cand = _add_unconfirmed_user_row(store, "条目三：等确认的候选")
    out = store.confirm_memory(cand)
    assert out["confirmed"] is False
    assert out["reason"] == "core_full"
    # 没确认上：座位没变
    assert store.core_seat_snapshot()["seats_used"] == 2
    # 待问生成了，同条再确认不重复攒
    qs = _capacity_questions(store)
    assert len(qs) == 1 and qs[0]["memory_id"] == cand
    assert store.confirm_memory(cand)["confirmed"] is False
    assert len(_capacity_questions(store)) == 1
    # 回答「先留作候选」：只消耗问题，记忆还在、仍未确认
    ans = store.answer_question(qs[0]["id"], "先留作候选")
    assert ans["ok"] is True and ans["applied"] == "held"
    assert _capacity_questions(store) == []
    assert store.core_seat_snapshot()["seats_used"] == 2


def test_confirm_passes_when_seats_available(store):
    store.core_seat_limit = 3
    _add_user(store, "条目一：占座一")
    cand = _add_unconfirmed_user_row(store, "条目二：等确认")
    out = store.confirm_memory(cand)
    assert out["confirmed"] is True
    assert store.core_seat_snapshot()["seats_used"] == 2
    assert _capacity_questions(store) == []


def test_capacity_gate_drop_answer_deletes_candidate(store):
    store.core_seat_limit = 1
    _add_user(store, "条目一：占满的座")
    cand = _add_unconfirmed_user_row(store, "条目二：满座候选")
    assert store.confirm_memory(cand)["reason"] == "core_full"
    qs = _capacity_questions(store)
    assert len(qs) == 1
    ans = store.answer_question(qs[0]["id"], "不要了")
    assert ans["ok"] is True and ans["applied"] == "dropped"
    snap = store.core_seat_snapshot()
    assert snap["live_rows"] == 1
    assert _capacity_questions(store) == []


def test_unpin_refused_when_full(store):
    store.core_seat_limit = 1
    _add_user(store, "条目一：占满的座")
    b = _add_user(store, "条目二：钉住的", protection_tier="pinned")
    assert store.core_seat_snapshot()["seats_used"] == 1
    out = store.set_protection_tier(b, None)
    assert out["ok"] is False and out["reason"] == "core_full"
    snap = store.core_seat_snapshot()
    assert snap["pinned"] == 1 and snap["seats_used"] == 1
    qs = _capacity_questions(store)
    assert len(qs) == 1 and qs[0]["memory_id"] == b


def test_setters_reject_bad_values(store):
    a = _add_user(store, "条目一：正常条目")
    assert store.set_protection_tier(a, "gold")["ok"] is False
    assert store.set_core_drawer(a, "不存在的格")["ok"] is False
    assert store.set_protection_tier("nope-id", "pinned")["reason"] == "not_found"
    snap = store.core_seat_snapshot()
    assert snap["seats_used"] == 1 and snap["pinned"] == 0


# ---------- 账本重算对账 ----------

def test_recount_from_ledger_matches_snapshot(store):
    _add_user(store, "条目一：本人的事", core_drawer="person")
    _add_user(store, "条目二：家人的事", core_drawer="people")
    c = _add_user(store, "条目三：没入格的事")
    d = _add_user(store, "条目四：要钉住的事")
    assert store.set_protection_tier(d, "pinned")["ok"] is True
    assert store.set_core_drawer(c, "environment")["ok"] is True
    store.add_memory_text("条目五：推断的事", source="inferred")
    snap = store.core_seat_snapshot()
    recount = store.recount_core_seats_from_ledger()
    for key in (
        "seats_used", "members", "pinned", "temp_pinned",
        "unclassified_seats", "live_rows",
    ):
        assert recount[key] == snap[key], key
    assert recount["drawers"] == snap["drawers"]


# ---------- 纯函数：水位与归一 ----------

def test_count_seats_watermark_and_ratio():
    metas = [{"source": "user", "user_confirmed": True} for _ in range(120)]
    snap = core_cap.count_seats(metas)
    assert snap["seats_used"] == 120
    assert snap["over_watermark"] is True
    assert snap["ratio"] == pytest.approx(0.8)
    assert snap["limit"] == 150
    assert snap["hysteresis_target"] == 140
    snap2 = core_cap.count_seats(metas[:119])
    assert snap2["over_watermark"] is False
    assert snap2["seats_free"] == 31


def test_normalize_rejects_bad_values():
    assert core_cap.normalize_protection_tier({"protection_tier": "pinned"}) == "pinned"
    assert core_cap.normalize_protection_tier({"protection_tier": "temp_pinned"}) == "temp_pinned"
    assert core_cap.normalize_protection_tier({"protection_tier": "yes"}) is None
    assert core_cap.normalize_protection_tier({}) is None
    assert core_cap.normalize_protection_tier(None) is None
    assert core_cap.normalize_core_drawer({"core_drawer": "people"}) == "people"
    assert core_cap.normalize_core_drawer({"core_drawer": "不存在"}) is None
    assert core_cap.normalize_core_drawer(None) is None
