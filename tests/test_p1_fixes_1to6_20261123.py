# tests/test_p1_fixes_1to6_20261123.py（P1-1..P1-5 的回归测试，2026-11-23）
# P1-6 是注释类改动，不用单测。
# 全部离线（向量打桩成 None），不碰真实 mem0、不联网。
# 辅助对象（FakeImportanceJudge / _judge_store / _turn）照搬
# tests/test_phase2_importance_questions.py 同名帮手，只为造“拿不准”
# 的问题；#4 的行为回归仍以那边为准。

import json
import re

import pytest

import memory.sqlite_store as store_mod
from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _row(store, text):
    """库里某条记忆的 (id, source, user_confirmed, decay_kind, importance)。"""
    return store._conn.execute(
        "SELECT id, source, user_confirmed, decay_kind, importance"
        " FROM memories WHERE text=?",
        (text,),
    ).fetchone()


def _meta(store, text):
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE text=?", (text,)
    ).fetchone()
    return json.loads(row[0]) if row else {}


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


class FakeImportanceJudge:
    """mock 重要性判官：按句子查表回 verdict，查不到回 False。"""

    def __init__(self, verdicts=None):
        self.verdicts = verdicts or {}
        self.calls = []
        self.provider_name = "deepseek"
        self.model = "deepseek-flash"

    def chat_messages(self, messages, ledger=None, thinking=None,
                      max_tokens=None):
        prompt = (messages or [{}])[0].get("content") or ""
        self.calls.append(prompt)
        match = re.search(r"^句子：(.+)$", prompt, re.M)
        sentence = (match.group(1).strip() if match else "").rstrip("。！？；,. ")
        verdict = self.verdicts.get(
            sentence, {"important": False, "reason": "默认一次性经历"}
        )
        return {
            "text": json.dumps(verdict, ensure_ascii=False),
            "usage": {"prompt_tokens": 400, "completion_tokens": 80},
            "cost_usd": None,
            "model": "deepseek-flash",
        }


def _judge_store(tmp_path, monkeypatch, judge):
    """带 mock 重要性判官的 store。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"),
                       judge_provider=judge)


def _turn(store, text, turn_id):
    """一轮对话：用户说一句话，助手回一句“好的”。"""
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的。"}],
        turn_id=turn_id,
    )


def _unsure_setup(tmp_path, monkeypatch):
    """造一个拿不准的问题：判官对这句回 unsure，存成 episodic + 攒问题。"""
    judge = FakeImportanceJudge({
        "我表弟下个月可能来广州玩": {
            "important": "unsure", "reason": "既像长期事实又像一次性经历",
        },
    })
    store = _judge_store(tmp_path, monkeypatch, judge)
    receipt = _turn(store, "我表弟下个月可能来广州玩。", "t1")
    assert len(receipt["stored"]) == 1
    questions = receipt.get("questions")
    assert questions is not None and len(questions) == 1
    return store, questions[0], receipt["stored"][0]


# ---------- P1-1：纠正候选排序死代码 ----------


def test_p1_1_correction_candidates_sorted_confirmed_first_newest_first(store):
    """已确认的在前，同类里新的在前——修之前 _cand_key 里字符串取负号
    抛 TypeError、被 try/except 吞掉，排序从不执行，返回的是检索原序。"""
    store.add_memory_text("我喜欢吃牛肉面", source="inferred")
    store.add_memory_text("我喜欢吃鸡杂面", source="user")
    store.add_memory_text("我喜欢吃担担面", source="user")
    # 控制创建时间：推断版最新、已确认版一新一旧。
    store._conn.execute(
        "UPDATE memories SET created_at=? WHERE text=?",
        ("2026-11-20T10:00:00+08:00", "我喜欢吃牛肉面"))
    store._conn.execute(
        "UPDATE memories SET created_at=? WHERE text=?",
        ("2026-11-10T10:00:00+08:00", "我喜欢吃鸡杂面"))
    store._conn.execute(
        "UPDATE memories SET created_at=? WHERE text=?",
        ("2026-11-22T10:00:00+08:00", "我喜欢吃担担面"))
    store._conn.commit()
    ids = {t: _row(store, t)[0] for t in
           ("我喜欢吃牛肉面", "我喜欢吃鸡杂面", "我喜欢吃担担面")}
    found = store._find_correction_candidates("我喜欢吃炸酱面")
    assert found == [ids["我喜欢吃担担面"], ids["我喜欢吃鸡杂面"],
                     ids["我喜欢吃牛肉面"]]


# ---------- P1-2：决定正则漏时间词 ----------


@pytest.mark.parametrize("text", [
    "我今天决定去北京",
    "我明天打算去上海",
    "我下周计划去广州",
    "我下个月准备买车",
    "我刚才决定退掉那张票",
    "我决定去北京",  # 不带时间词的对照：本来就判得对
])
def test_p1_2_decision_with_time_words_is_important(store, text):
    """“我”和决定动词之间插了时间词，也要判成 important——修之前
    正则要求“我”紧贴动词，这几句都漏判，掉进过去时间词分支当成
    一次性经历。"""
    decision, _reason, _judge = store._decide_importance(text)
    assert decision == "important"


# ---------- P1-3：纠正新记忆一律变 stable ----------


def test_p1_3_correction_of_episodic_event_stays_episodic(store):
    """纠正一条一次性事件（“昨天我吃了牛肉面”→“鸡杂面”），新记忆
    还是 episodic——修之前按 source="user" 一律算 stable。"""
    store.add_memory_text("昨天我吃了牛肉面", source="user",
                          decay_kind="episodic")
    old_id, _s, _c, old_kind, _i = _row(store, "昨天我吃了牛肉面")
    assert old_kind == "episodic"
    old_meta = _meta(store, "昨天我吃了牛肉面")
    out = store._apply_correction_result(
        "昨天我吃了鸡杂面", old_id, "昨天我吃了牛肉面", old_meta,
        same_text=False, candidates=[old_id], status="updated",
    )
    assert out.get("status") == "updated"
    new_kind = store._conn.execute(
        "SELECT decay_kind FROM memories WHERE id=?",
        (out.get("memory_id"),)).fetchone()[0]
    assert new_kind == "episodic"
    assert "昨天我吃了鸡杂面" in _live_texts(store)
    assert store._conn.execute(
        "SELECT superseded FROM memories WHERE id=?",
        (old_id,)).fetchone()[0] == 1


def test_p1_3_correction_of_stable_preference_stays_stable(store):
    """对照：纠正一条稳定偏好，新记忆继承 stable，不降级。"""
    store.add_memory_text("我喜欢吃牛肉面", source="user")
    old_id, _s, _c, old_kind, _i = _row(store, "我喜欢吃牛肉面")
    assert old_kind == "stable"
    old_meta = _meta(store, "我喜欢吃牛肉面")
    out = store._apply_correction_result(
        "我喜欢吃鸡杂面", old_id, "我喜欢吃牛肉面", old_meta,
        same_text=False, candidates=[old_id], status="updated",
    )
    assert out.get("status") == "updated"
    new_kind = store._conn.execute(
        "SELECT decay_kind FROM memories WHERE id=?",
        (out.get("memory_id"),)).fetchone()[0]
    assert new_kind == "stable"


# ---------- P1-4：同文去重升级不记账本 ----------


def test_p1_4_dedupe_upgrade_survives_rebuild(store):
    """推断版在库、已确认版后到：升级成已确认/stable，账本补一笔
    PROMOTE；删掉 SQLite 行再对账重建，升级不丢——修之前升级不记
    账本，重建后变回推断版/episodic。"""
    store.add_memory_text("我喜欢吃牛肉面", source="inferred")
    row = _row(store, "我喜欢吃牛肉面")
    assert row[1] == "inferred" and row[2] == 0 and row[3] == "episodic"
    out = store.add_memory_text("我喜欢吃牛肉面", source="user")
    assert out.get("event") == "UNCHANGED"
    row = _row(store, "我喜欢吃牛肉面")
    assert row[1] == "user" and row[2] == 1 and row[3] == "stable"
    events, _info = _ledger.read_events(store.ledger_path)
    promotes = [e for e in events
                if e.get("type") == _ledger.LEDGER_EVENT_PROMOTE
                and e.get("memory_id") == row[0]]
    assert len(promotes) == 1
    assert (promotes[0].get("meta") or {}).get("decay_kind") == "stable"
    # 模拟索引丢失：删掉 SQLite 行，再对账重建。
    store._conn.execute("DELETE FROM memories WHERE id=?", (row[0],))
    store._conn.commit()
    assert store.reconcile_ledger_index() == 1
    row2 = _row(store, "我喜欢吃牛肉面")
    assert row2 is not None
    assert row2[1] == "user" and row2[2] == 1 and row2[3] == "stable"


# ---------- P1-5：回答“进核心”先改库再写账本 ----------


def test_p1_5_answer_core_writes_ledger_first(tmp_path, monkeypatch):
    """正常进核心：账本 PROMOTE 和 SQLite 改动两者一致。"""
    store, question, text = _unsure_setup(tmp_path, monkeypatch)
    mid = _row(store, text)[0]
    result = store.answer_question(question["id"], "进核心")
    assert result["ok"] is True and result["applied"] == "core"
    assert _row(store, text)[3:] == ("stable", "high")
    events, _info = _ledger.read_events(store.ledger_path)
    promotes = [e for e in events
                if e.get("type") == _ledger.LEDGER_EVENT_PROMOTE
                and e.get("memory_id") == mid]
    assert len(promotes) == 1
    assert (promotes[0].get("meta") or {}).get("decay_kind") == "stable"
    assert store.list_pending_questions() == []


def test_p1_5_answer_core_ledger_failure_leaves_sqlite_untouched(
        tmp_path, monkeypatch):
    """账本写失败时 SQLite 不动，问题也不消耗——修之前先改 SQLite
    再写账本，账本失败只记 warning，SQLite 的改动不回滚。"""
    store, question, text = _unsure_setup(tmp_path, monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("账本写不进去（测试注入的失败）。")

    monkeypatch.setattr(store_mod._ledger, "append_event", _boom)
    result = store.answer_question(question["id"], "进核心")
    assert result["ok"] is False
    assert _row(store, text)[3:] == ("episodic", "normal")
    assert len(store.list_pending_questions()) == 1  # 问题还在，可再答
    # 账本写失败后放开，再答一次能成功（问题没被烧掉）。
    monkeypatch.undo()
    result2 = store.answer_question(question["id"], "进核心")
    assert result2["ok"] is True and result2["applied"] == "core"
    assert _row(store, text)[3:] == ("stable", "high")
