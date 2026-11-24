# tests/test_judge_same_fact_20261124.py（判官判“是不是同一件事”，2026-11-24）
# BOOS 拍板的设计扩展：晋升计数时，硬信号认不出的候选对交给判官
# 判“同一件事 / 不是同一件事”，判同一件的才把出现证据并入同簇。
# 本文件只测这一项（BOOS 定：改完单独跑这一项，其他的不跑）。
# 全部离线：判官用假的（注入 judge_provider），不调真 API、不烧钱。
#
# 规矩回顾（不许破）：
# - 这是计数归并：不重写、不删除、不触发取代，行都留着；
# - 护栏优先于判官：冲突话题/挂未答冲突问题/带语气标的对不问；
# - 判官断了/回垃圾 → 降级不归并、不崩、不丢事实、不乱并；
# - 判了留痕：结论（same、provider、model、时间）写进两行 metadata，
#   同时当缓存——同对不重复问。

import json

import numpy as np
import pytest

from memory import decay as _decay
from memory import ledger as _ledger
from memory import pending_questions as _pq
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso


class FakeJudgeProvider:
    """假判官 Provider：responder(prompt) 返回 dict 当判词、返回
    字符串当原始回复（测垃圾回复）、抛异常模拟断线。记录被问次数。"""

    provider_name = "fake"
    model = "fake-judge-1"

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def chat_messages(self, messages, ledger=None, thinking=None,
                      max_tokens=None):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        out = self.responder(prompt)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(
            out, ensure_ascii=False)
        return {
            "text": text,
            "usage": {"prompt_tokens": 120, "completion_tokens": 20},
            "cost_usd": 0.0,
        }


def _always_same(prompt):
    return {"same_fact": True, "reason": "核心事实等价"}


def _always_diff(prompt):
    return {"same_fact": False, "reason": "对象不同"}


@pytest.fixture
def make_store(tmp_path, monkeypatch):
    """干净的 SQLiteStore 工厂：向量打桩成 None（靠 _plant 显式给）、
    语义判官（取代那个）打桩成不可用；同事实判官注入假的。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )

    def _make(responder):
        judge = FakeJudgeProvider(responder)
        store = SQLiteStore(
            memory_dir=str(tmp_path / "memory"), judge_provider=judge)
        return store, judge

    return _make


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
    """直接插一行（模拟旧行），meta 可覆盖；seen 集合传列表。"""
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


def _unit_vec():
    vec = np.zeros(512, dtype="float32")
    vec[0] = 1.0
    return vec


# 松散改述对（上一轮 A 组的真实形状）：只共享核心词“酸梅汤”，
# 骨架全变——硬信号过不了（无槽位、二字组重合 0.22 < 0.8），
# 预筛刚好够问判官（重合 ≥0.2 且 Jaccard ≥0.05）。
TEXT_A = "我平时最喜欢喝的就是酸梅汤"
TEXT_B = "酸梅汤一直是我的心头好，喝了好多年了"


def _plant_loose_pair(store):
    a = _plant(store, TEXT_A, write_seq=1,
               seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, TEXT_B, write_seq=2,
               seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    return a, b


# ---------- 一、判官判同一件事 → 证据并入同簇计数 ----------


def test_judge_same_fact_merges_loose_paraphrase(make_store):
    """松散改述：硬信号认不出，判官判同一件事 → 两行证据并起来，
    第 3 天第 3 会话晋升；判官只问一次；结论留痕在两行 metadata。"""
    store, judge = make_store(_always_same)
    a, b = _plant_loose_pair(store)
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is True
    meta_b = _meta_of(store, b)
    assert meta_b["decay_kind"] == "stable"
    dates, sessions = _seen(meta_b)
    assert dates == {"2026-11-18", "2026-11-19", "2026-11-20"}
    assert sessions == {"s-a", "s-b", "s-c"}
    assert len(judge.calls) == 1
    # 留痕 + 缓存：两行各存一份，带 provider/model/时间。
    rec_b = meta_b["same_fact_judgments"][a]
    assert rec_b["same"] is True
    assert rec_b["provider"] == "fake"
    assert rec_b["model"] == "fake-judge-1"
    assert rec_b["at"]
    rec_a = _meta_of(store, a)["same_fact_judgments"][b]
    assert rec_a["same"] is True
    # 被并的那行本身没被动：还是 episodic，没删没顶。
    assert _meta_of(store, a)["decay_kind"] == "episodic"


def test_judge_same_fact_via_vectors_only(make_store):
    """二字组完全不重合（骑行通勤的两种说法）、只有向量像：硬信号
    过不了（重合 0），靠余弦预筛进判官，判同一件事后并证据晋升。"""
    store, judge = make_store(_always_same)
    vec = _unit_vec()
    a = _plant(store, "我每天上班都是骑自行车去的", vector=vec, write_seq=1,
               seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, "这几年通勤我一直蹬车子，风雨无阻", vector=vec,
               write_seq=2, seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is True
    assert len(judge.calls) == 1
    assert _meta_of(store, b)["same_fact_judgments"][a]["same"] is True


# ---------- 二、判官判不是同一件事 → 不并 ----------


def test_judge_different_fact_no_merge(make_store):
    """判官判不是同一件事：证据不并、不晋升；结论缓存，
    再出现也不重复问。"""
    store, judge = make_store(_always_diff)
    a, b = _plant_loose_pair(store)
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is False
    dates, sessions = _seen(_meta_of(store, b))
    assert dates == {"2026-11-19", "2026-11-20"}
    assert sessions == {"s-b", "s-c"}
    assert len(judge.calls) == 1
    assert _meta_of(store, b)["same_fact_judgments"][a]["same"] is False
    # 缓存命中：再记一次出现，判官不再被问。
    store._record_occurrence(b, _meta_of(store, b), "2026-11-21", "s-d")
    assert len(judge.calls) == 1


def test_judge_verdict_cached_no_reask_on_merge(make_store):
    """判同一件事后缓存生效：第二次计数直接用缓存并证据，
    判官总共只被问一次。"""
    store, judge = make_store(_always_same)
    a, b = _plant_loose_pair(store)
    store._record_occurrence(b, _meta_of(store, b), "2026-11-20", "s-c")
    assert len(judge.calls) == 1
    store._record_occurrence(b, _meta_of(store, b), "2026-11-21", "s-d")
    assert len(judge.calls) == 1
    dates, sessions = _seen(_meta_of(store, b))
    assert dates == {"2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"}
    assert sessions == {"s-a", "s-b", "s-c", "s-d"}


# ---------- 三、护栏对：根本不问判官 ----------


def test_guard_conflict_pair_never_asks_judge(make_store):
    """同槽不同值（我住广州 / 我住西安）是冲突对：哪怕预筛能过、
    判官会说同一件事，也不许问、不许并。"""
    store, judge = make_store(_always_same)
    _plant(store, "我住广州", write_seq=1,
           seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, "我住西安", write_seq=2,
               seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is False
    assert judge.calls == []
    assert "same_fact_judgments" not in _meta_of(store, b)


def test_guard_pending_question_pair_never_asks_judge(make_store, monkeypatch):
    """两人之间挂着未答冲突问题的对：BOOS 没裁之前不共享证据，
    判官也不许问。"""
    store, judge = make_store(_always_same)
    a, b = _plant_loose_pair(store)
    monkeypatch.setattr(
        store.pending_questions, "list_unanswered",
        lambda: [{
            "kind": _pq.QUESTION_KIND_CONFLICT,
            "old_id": a, "new_id": b,
        }],
    )
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is False
    assert judge.calls == []


def test_guard_tone_pair_never_asks_judge(make_store):
    """带语气标（玩笑）的行证据本来就不算数：这一对不问判官、
    不并证据、不留判官痕。"""
    store, judge = make_store(_always_same)
    _plant(store, TEXT_A, write_seq=1, tone="joke",
           seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, TEXT_B, write_seq=2,
               seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is False
    assert judge.calls == []
    assert "same_fact_judgments" not in _meta_of(store, b)


def test_hard_signal_pair_never_asks_judge(make_store):
    """硬信号能认的对（同槽同值）照老路并证据，判官一次都不问
    ——哪怕判官会说“不是同一件事”，也轮不到它管。"""
    store, judge = make_store(_always_diff)
    _plant(store, "我住广州", write_seq=1,
           seen_dates=["2026-11-18"], seen_sessions=["s-a"])
    b = _plant(store, "我住在广州", write_seq=2,
               seen_dates=["2026-11-19"], seen_sessions=["s-b"])
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is True
    assert judge.calls == []


# ---------- 四、判官不可用 → 降级不并、不崩 ----------


def test_judge_unavailable_degrades_no_merge(make_store):
    """判官调用抛错（断线）：计数照常走完、本行证据照记，
    不并、不晋升、不留缓存（下次还能再试），不许崩。"""
    store, judge = make_store(
        lambda prompt: RuntimeError("API 断了"))
    a, b = _plant_loose_pair(store)
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is False
    assert len(judge.calls) == 1
    meta_b = _meta_of(store, b)
    dates, sessions = _seen(meta_b)
    assert dates == {"2026-11-19", "2026-11-20"}
    assert sessions == {"s-b", "s-c"}
    assert "same_fact_judgments" not in meta_b
    assert "same_fact_judgments" not in _meta_of(store, a)


def test_judge_garbage_response_degrades_no_merge(make_store):
    """判官回的不是 JSON：同样降级不并、不崩、不留缓存。"""
    store, judge = make_store(lambda prompt: "我拿不准，你们看着办")
    a, b = _plant_loose_pair(store)
    promoted = store._record_occurrence(
        b, _meta_of(store, b), "2026-11-20", "s-c")
    assert promoted is False
    assert len(judge.calls) == 1
    assert "same_fact_judgments" not in _meta_of(store, b)
