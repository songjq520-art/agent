# tests/test_promo_recall_fix_20261124.py（晋升 + 漏答优化，2026-11-24）
# BOOS 定：晋升规则再优化（换说法聚合计数、纠正后继承晋升证据）、
# 漏答再优化（槽位二次整理、被取代/输了的旧行不许压过现行条）。
# 全部离线（向量按需打桩、语义判官打桩成不可用，不调真 API）。
#
# 晋升侧根因：出现证据（seen_dates/seen_sessions）只记在单行上——
# 同一件事换个说法存成另一行，证据碎成几份、谁也凑不满 3 天 3 次；
# 纠正出的新行更是从零开始数。修法：_record_occurrence 晋升判断前
# 按硬信号聚合“同一事实的其他说法”的证据（_fact_cluster_members）；
# 纠正/取代真落盘后旧行证据并给新行（_inherit_seen_evidence）。
# 召回侧根因：_search_inner 截断 top_k 后，冲突标注只看结果内——
# 某槽的现行条落在 top_k 外、同槽旧行在里面时，回答只能用旧行。
# 修法：_annotate_search_conflicts 第 0 步槽位二次整理
# （_slot_recompose）：现行条换进来 / 排到旧行前面。

import json

import numpy as np
import pytest

from memory import decay as _decay
from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso


@pytest.fixture
def store(tmp_path, monkeypatch):
    """干净的 SQLiteStore；向量打桩成 None（槽位/关键词路照常），
    语义判官打桩成不可用（模型断了降级走程序，是既定生产模式）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _meta_of(store, memory_id):
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE id=?", (memory_id,)
    ).fetchone()
    return json.loads(row[0] or "{}")


def _seen(meta):
    return (
        set(_decay._parse_seen_list(meta.get("seen_dates"))),
        set(_decay._parse_seen_list(meta.get("seen_sessions"))),
    )


def _plant(store, text, vector=None, **meta_over):
    """直接插一行（模拟旧行/迁移行），meta 可覆盖；seen 集合传
    seen_dates/seen_sessions 列表即可（这里转成账本同款 JSON 串）。"""
    created_at = meta_over.pop("created_at", _now_iso())
    key = _ledger.ledger_event_key(text, created_at + text)
    meta = {
        "source": "inferred", "confidence": 0.7, "user_confirmed": False,
        "created_at": created_at, "write_seq": 1, "turn_id": None,
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "episodic", "importance": "normal",
        "event_date": None,
    }
    meta.update(meta_over)
    if isinstance(meta.get("seen_dates"), list):
        meta["seen_dates"] = _decay._dump_seen_list(meta["seen_dates"])
    if isinstance(meta.get("seen_sessions"), list):
        meta["seen_sessions"] = _decay._dump_seen_list(meta["seen_sessions"])
    store._insert_memory_row(key, text, meta, vector)
    return key


def _unit_vec(dim0=1.0):
    vec = np.zeros(512, dtype="float32")
    vec[0] = dim0
    return vec


def _item(store, memory_id):
    """按 search 结果项的形状造一条（_annotate_search_conflicts 用）。"""
    row = store._conn.execute(
        "SELECT text, metadata_json FROM memories WHERE id=?", (memory_id,)
    ).fetchone()
    meta = json.loads(row[1] or "{}")
    return {
        "id": memory_id, "text": row[0], "score": 0.5, "source": "semantic",
        "metadata": meta, "confirmed": bool(meta.get("user_confirmed")),
        "decay_weight": 1.0, "is_history": False,
    }


def _filler(text):
    return {
        "id": None, "text": text, "score": 0.4, "source": "keyword",
        "metadata": {}, "confirmed": False, "decay_weight": 1.0,
        "is_history": False,
    }


# ---------- 一、晋升聚合：换说法不碎证据 ----------


def test_promotion_aggregates_same_slot_rephrasings(store):
    """同一住处换三种说法（我住广州 / 我住在广州），证据碎在两行上：
    任一行单独都只有 1 天 1 会话，聚合后第 3 天第 3 会话应晋升。"""
    a = _plant(store, "我住广州", write_seq=1,
               seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, "我住在广州", write_seq=2,
               seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c"
    )
    assert promoted is True
    meta_b = _meta_of(store, b)
    assert meta_b["decay_kind"] == "stable"
    dates, sessions = _seen(meta_b)
    assert dates == {"2026-11-18", "2026-11-19", "2026-11-20"}
    assert sessions == {"s-a", "s-b", "s-c"}
    # 被并的那行本身没被动：还是 episodic，下次它被提到时再轮到它。
    assert _meta_of(store, a)["decay_kind"] == "episodic"


def test_promotion_aggregates_similar_rephrasing_with_vectors(store):
    """没有槽位的偏好句靠向量 + 内容二字组认亲：“喜欢吃面条”和
    “爱吃面条”向量相同、二字组全同，算同一件事，证据聚合晋升。"""
    vec = _unit_vec()
    _plant(store, "我喜欢吃面条", vector=vec, write_seq=1,
           seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, "我爱吃面条", vector=vec, write_seq=2,
               seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c"
    )
    assert promoted is True
    dates, sessions = _seen(_meta_of(store, b))
    assert dates == {"2026-11-18", "2026-11-19", "2026-11-20"}
    assert sessions == {"s-a", "s-b", "s-c"}


def test_promotion_no_merge_template_twins(store):
    """反例护栏：模板相同只换宾语的两件事（定投买纳指 / 买标普），
    就算向量一模一样也不许并证据——二字组重合率不够。"""
    vec = _unit_vec()
    _plant(store, "我想每天定投5块买纳指", vector=vec, write_seq=1,
           seen_dates=["2026-11-08", "2026-11-09"],
           seen_sessions=["s-8", "s-9"])
    b = _plant(store, "我想每天定投5块买标普", vector=vec, write_seq=2,
               seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-19", "s-b"
    )
    assert promoted is False
    dates, sessions = _seen(_meta_of(store, b))
    assert dates == {"2026-11-18", "2026-11-19"}
    assert sessions == {"s-a", "s-b"}


def test_promotion_no_merge_across_slot_conflict(store):
    """反例护栏：同槽不同值是冲突、不是换说法（广州 vs 深圳），
    证据不许互并。"""
    _plant(store, "我住广州", write_seq=1,
           seen_dates=["2026-11-08", "2026-11-09"],
           seen_sessions=["s-8", "s-9"])
    b = _plant(store, "我住深圳", write_seq=2,
               seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-19", "s-b"
    )
    assert promoted is False
    dates, sessions = _seen(_meta_of(store, b))
    assert dates == {"2026-11-18", "2026-11-19"}
    assert sessions == {"s-a", "s-b"}


def test_promotion_joke_member_evidence_not_counted(store):
    """带玩笑标记的成员：照常是“同一件事”，但它的出现不算数
    （带标记不计晋升是既定规矩），不能靠它凑满。"""
    _plant(store, "我住广州", write_seq=1, tone="joke",
           seen_dates=["2026-11-08", "2026-11-09"],
           seen_sessions=["s-8", "s-9"])
    b = _plant(store, "我住在广州", write_seq=2,
               seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-19", "s-b"
    )
    assert promoted is False
    dates, sessions = _seen(_meta_of(store, b))
    assert dates == {"2026-11-18", "2026-11-19"}
    assert sessions == {"s-a", "s-b"}


def test_promotion_confirmed_member_evidence_counts(store):
    """已确认成员的证据全额计（任务 1c）：推断版被提到时，把已确认
    同事实攒的天数/会话算进来，别被推断版的身份卡住晋升。"""
    _plant(store, "我住广州", write_seq=1, source="user",
           user_confirmed=True, confidence=1.0,
           seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, "我住在广州", write_seq=2,
               seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c"
    )
    assert promoted is True
    dates, sessions = _seen(_meta_of(store, b))
    assert dates == {"2026-11-18", "2026-11-19", "2026-11-20"}
    assert sessions == {"s-a", "s-b", "s-c"}


# ---------- 二、纠正继承：新事实不从零开始 ----------


def test_correction_inherits_evidence_and_promotes(store):
    """旧行攒了 2 天 2 会话，纠正出的新行应继承：加上纠正轮就是
    3 天 3 会话，当场晋升成稳定，不许从零开始。"""
    old = _plant(
        store, "我喜欢吃米线", write_seq=1, source="user",
        user_confirmed=True, confidence=1.0,
        seen_dates=["2026-11-20", "2026-11-21"],
        seen_sessions=["s-old-1", "s-old-2"],
    )
    receipt = store.save_turn(
        [{"role": "user", "content": "不对，我喜欢吃米粉"},
         {"role": "assistant", "content": "好的"}],
        turn_id="t-corr-1",
    )
    assert (receipt.get("correction") or {}).get("status") == "updated"
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE id=?", (old,)
    ).fetchone()
    assert row[0] == 1
    new_row = store._conn.execute(
        "SELECT id, metadata_json FROM memories WHERE voided=0"
        " AND deleted=0 AND superseded=0 AND text LIKE '%米粉%'"
    ).fetchone()
    assert new_row is not None
    meta = json.loads(new_row[1] or "{}")
    dates, sessions = _seen(meta)
    assert {"2026-11-20", "2026-11-21"} <= dates
    assert {"s-old-1", "s-old-2", "t-corr-1"} <= sessions
    assert len(dates) >= 3 and len(sessions) >= 3
    assert meta["decay_kind"] == "stable"


def test_correction_no_inherit_from_joke_old(store):
    """旧行是玩笑话（带 tone 标）：它的“出现”本来就不算数，
    纠正出的新行不许继承那份证据。"""
    _plant(
        store, "我喜欢吃螺蛳粉", write_seq=1, source="user",
        user_confirmed=True, confidence=1.0, tone="joke",
        seen_dates=["2026-11-18", "2026-11-19", "2026-11-20"],
        seen_sessions=["s-j-1", "s-j-2", "s-j-3"],
    )
    receipt = store.save_turn(
        [{"role": "user", "content": "不对，我喜欢吃桂林米粉"},
         {"role": "assistant", "content": "好的"}],
        turn_id="t-corr-joke",
    )
    assert (receipt.get("correction") or {}).get("status") == "updated"
    new_row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE voided=0"
        " AND deleted=0 AND superseded=0 AND text LIKE '%桂林米粉%'"
    ).fetchone()
    assert new_row is not None
    meta = json.loads(new_row[0] or "{}")
    dates, sessions = _seen(meta)
    assert not ({"s-j-1", "s-j-2", "s-j-3"} & sessions)
    assert not ({"2026-11-18", "2026-11-19", "2026-11-20"} & dates)
    assert sessions == {"t-corr-joke"}


def test_slot_supersede_inherits_evidence(store):
    """槽位直存取代（add_memory_text 的 allow_slot_supersede 路）
    落盘后同样继承：旧住处攒的天数/会话并给新住处行。"""
    old = _plant(
        store, "我住广州", write_seq=1, source="user",
        user_confirmed=True, confidence=1.0,
        seen_dates=["2026-11-20", "2026-11-21"],
        seen_sessions=["s-old-1", "s-old-2"],
    )
    saved = store.add_memory_text(
        "我住深圳", source="user", turn_id="t-new-home",
        allow_slot_supersede=True,
    )
    assert saved.get("event") == "ADD"
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE id=?", (old,)
    ).fetchone()
    assert row[0] == 1
    dates, sessions = _seen(_meta_of(store, saved["id"]))
    assert {"2026-11-20", "2026-11-21"} <= dates
    assert {"s-old-1", "s-old-2", "t-new-home"} <= sessions


def test_answer_conflict_question_inherits_on_swap(store):
    """待问问题裁“换成新的”这一刻才算真顶：旧行证据这时并给
    新行；裁之前（挂起状态）不许提前并。"""
    old = _plant(
        store, "我住广州", write_seq=1,
        seen_dates=["2026-11-20", "2026-11-21"],
        seen_sessions=["s-old-1", "s-old-2"],
    )
    new = _plant(
        store, "我住深圳", write_seq=2, source="user",
        user_confirmed=True, confidence=1.0,
        seen_dates=["2026-11-22"], seen_sessions=["s-new-1"],
    )
    question = store._queue_conflict_question(
        old, "我住广州", new, "我住深圳", turn_id="t-ask",
    )
    assert question and question.get("id")
    # 还没裁：新行证据不许动。
    dates, sessions = _seen(_meta_of(store, new))
    assert dates == {"2026-11-22"} and sessions == {"s-new-1"}
    result = store.answer_question(question["id"], "换成新的")
    assert result.get("ok") is True
    assert result.get("applied") == "superseded"
    dates, sessions = _seen(_meta_of(store, new))
    assert {"2026-11-20", "2026-11-21", "2026-11-22"} <= dates
    assert {"s-old-1", "s-old-2", "s-new-1"} <= sessions
    # 2 天旧证据 + 新行自己的 1 天 = 3 天，会话同理 3 个：晋升。
    assert _meta_of(store, new)["decay_kind"] == "stable"


# ---------- 三、槽位二次整理：现行条不许缺席、不许垫底 ----------


def test_slot_recompose_swaps_current_in(store):
    """住处槽：现行条（最新已确认）没进结果、旧行进了——把现行
    条换进来占旧行的位置，结果条数不变。"""
    old = _plant(store, "我住西安", write_seq=1, source="user",
                 user_confirmed=True, confidence=1.0,
                 created_at="2026-11-18T00:00:00Z")
    new = _plant(store, "我住深圳", write_seq=9, source="user",
                 user_confirmed=True, confidence=1.0,
                 created_at="2026-11-22T00:00:00Z")
    results = [
        _item(store, old),
        _filler("用户养了一只猫，名字叫汤圆。"),
        _filler("用户想购买碧桂园的房子。"),
        _filler("每月预算 5 美元，到 5 美元提醒。"),
    ]
    out = store._annotate_search_conflicts(results)
    texts = [r["text"] for r in out]
    assert "我住深圳" in texts
    assert "我住西安" not in texts
    assert len(out) == 4


def test_slot_recompose_current_before_stale(store):
    """现行条和旧行都在结果里时，现行必须排在旧行前面，并被标
    conflict_current、旧行标 conflict_stale（第 2 步标注接上）。"""
    old = _plant(store, "我住西安", write_seq=1, source="user",
                 user_confirmed=True, confidence=1.0,
                 created_at="2026-11-18T00:00:00Z")
    new = _plant(store, "我住深圳", write_seq=9, source="user",
                 user_confirmed=True, confidence=1.0,
                 created_at="2026-11-22T00:00:00Z")
    results = [
        _item(store, old),
        _filler("用户养了一只猫，名字叫汤圆。"),
        _item(store, new),
    ]
    out = store._annotate_search_conflicts(results)
    texts = [r["text"] for r in out]
    assert texts.index("我住深圳") < texts.index("我住西安")
    by_text = {r["text"]: r for r in out}
    assert (by_text["我住深圳"].get("metadata") or {}).get("conflict_current")
    assert (by_text["我住西安"].get("metadata") or {}).get("conflict_stale")


def test_pending_new_fact_not_hidden(store):
    """待问问题挂着的新事实不许在结果里隐身（webfix 三的规矩，
    本轮保持并加测）：旧行在结果里、新行不在，新行必须被换进来。"""
    old = _plant(store, "用户喜欢喝燕麦奶", write_seq=1,
                 created_at="2026-11-18T00:00:00Z")
    new = _plant(store, "用户喜欢喝豆奶", write_seq=2, source="user",
                 user_confirmed=True, confidence=1.0,
                 created_at="2026-11-22T00:00:00Z")
    question = store._queue_conflict_question(
        old, "用户喜欢喝燕麦奶", new, "用户喜欢喝豆奶", turn_id="t-ask",
    )
    assert question and question.get("id")
    results = [
        _item(store, old),
        _filler("用户养了一只猫，名字叫汤圆。"),
        _filler("用户想购买碧桂园的房子。"),
        _filler("每月预算 5 美元，到 5 美元提醒。"),
        _filler("用户是男性。"),
    ]
    out = store._annotate_search_conflicts(results)
    texts = [r["text"] for r in out]
    assert "用户喜欢喝豆奶" in texts


def test_slot_recompose_ignores_negation_note_rows(store):
    """防回归（全套实测教训）：偏好话题里带否定词的纠正备注行
    （“纠正了此前喜欢吃鸡杂面的说法”）不许拿否定链接成组——
    否则它会把同话题的并存行拖成一组、按序号选出“现行条”
    重排结果，把好好的排序打乱。没有待问问题认定的对子时，
    二次整理对偏好话题什么都不做。"""
    beef = _plant(store, "BOOS 喜欢吃牛肉面。", write_seq=4,
                  created_at="2026-11-18T00:00:00Z")
    note = _plant(
        store,
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。",
        write_seq=6, created_at="2026-11-19T00:00:00Z",
    )
    noodle = _plant(store, "BOOS 喜欢吃鸡杂面。", write_seq=7,
                    created_at="2026-11-20T00:00:00Z")
    results = [_item(store, beef), _item(store, note), _item(store, noodle)]
    out = store._annotate_search_conflicts(results)
    assert [r["text"] for r in out] == [
        "BOOS 喜欢吃牛肉面。",
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。",
        "BOOS 喜欢吃鸡杂面。",
    ]
    assert all(r.get("source") != "slot_current" for r in out)


def test_slot_recompose_pref_coexist_not_swapped(store):
    """护栏：偏好话题里不同食物可以并存（火锅和冒菜都是真的），
    没有硬链接时不许把“最新那条”当现行条换进来赶走旧条。"""
    old = _plant(store, "我爱吃火锅", write_seq=1, source="user",
                 user_confirmed=True, confidence=1.0,
                 created_at="2026-11-18T00:00:00Z")
    _plant(store, "我爱吃冒菜", write_seq=9, source="user",
           user_confirmed=True, confidence=1.0,
           created_at="2026-11-22T00:00:00Z")
    results = [
        _item(store, old),
        _filler("用户养了一只猫，名字叫汤圆。"),
    ]
    out = store._annotate_search_conflicts(results)
    texts = [r["text"] for r in out]
    assert "我爱吃火锅" in texts
    assert "我爱吃冒菜" not in texts
