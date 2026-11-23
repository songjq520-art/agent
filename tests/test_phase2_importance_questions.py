# tests/test_phase2_importance_questions.py（Phase 2 #2 + #4 回归测试，2026-11-23）
# 只测 memory/sqlite_store.py + memory/pending_questions.py + memory/self_check.py
# 的 Phase 2 接线；全部离线（向量打桩 + mock 重要性判官，不加载 bge 模型、
# 不调真 API、不花钱）。
# 依据：BOOS 2026-11-22 定——#2 留什么/丢什么改成和十一一样（重要的出现
#   一次就进核心，3 天 3 次留着当兜底；拿不准就问）；#4 问你的方式改成和
#   十一一样（聊天里直接问：带原文、给选项、说原因；后台的攒一批再问）。
#
# 覆盖：
#   #2-1 偏好句一次进核心（程序判，不调模型）
#   #2-2 判官说重要，一次进核心
#   #2-3 不重要的靠 3 天 3 次兜底晋升（第 2 天还不晋升）
#   #2-4 问题句/假设句/确认词/半句话/当下想法不存（复用“别什么都记”）
#   #2-5 判官断了降级：当一次性经历存，不崩、不丢、不问
#   #4-1 聊天里拿不准时，回执带原文+选项+原因
#   #4-2 后台问题攒批：直存路的攒着，下次聊天有新问题时顺带问（最多 3 条）
#   #4-3 没新问题时不打扰（回执里没有 questions）
#   #4-4 回答“进核心”/“当一次性经历”/“不存”各改对库；选项不对不消耗问题
#   #2-6 自检第 10 项：核心记忆还管不管用（只提议，不自动改）

import copy
import json
import re
from datetime import date, datetime, timedelta, timezone

import pytest

import memory.sqlite_store as store_mod
from memory import self_check
from memory.self_check import run_self_check
from memory.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


class FakeImportanceJudge:
    """mock 重要性判官：按句子查表回 verdict，查不到回 False；fail=True 模拟断了。

    provider_name / model 属性供分发层认名字；calls 记下每次收到的提示词，
    断言“调没调模型、调了几次”就看它。查表时剥掉句尾标点（save_turn 分句
    后没标点、直存路有标点，两边都对得上）。
    """

    def __init__(self, verdicts=None, fail=False):
        self.verdicts = verdicts or {}
        self.fail = fail
        self.calls = []
        self.provider_name = "deepseek"
        self.model = "deepseek-flash"

    def chat_messages(self, messages, ledger=None, thinking=None,
                      max_tokens=None):
        prompt = (messages or [{}])[0].get("content") or ""
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("API 断了")
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


def _row(store, text):
    """库里某条记忆的 (decay_kind, importance)，没有返回 None。"""
    return store._conn.execute(
        "SELECT decay_kind, importance FROM memories WHERE text=?", (text,)
    ).fetchone()


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


# ---------- #2-1：重要的出现一次就进核心（程序判，不调模型） ----------

def test_important_preference_enters_core_at_once_no_judge_call(
        tmp_path, monkeypatch):
    judge = FakeImportanceJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    receipt = _turn(store, "我喜欢吃牛肉面。", "t1")
    assert len(receipt["stored"]) == 1
    assert _row(store, receipt["stored"][0]) == ("stable", "high")
    assert judge.calls == []  # 偏好句式程序就判了，没调模型


# ---------- #2-2：判官说重要，一次进核心 ----------

def test_important_model_judge_enters_core_at_once(tmp_path, monkeypatch):
    judge = FakeImportanceJudge({
        "我每周三下午去游泳": {"important": True, "reason": "固定习惯，长期"},
    })
    store = _judge_store(tmp_path, monkeypatch, judge)
    receipt = _turn(store, "我每周三下午去游泳。", "t1")
    assert len(receipt["stored"]) == 1
    assert _row(store, receipt["stored"][0]) == ("stable", "high")
    assert len(judge.calls) == 1  # 程序判不定，问了一次模型


# ---------- #2-3：不重要的靠 3 天 3 次兜底晋升 ----------

def test_unimportant_promotes_after_three_days_three_sessions(
        tmp_path, monkeypatch):
    store = _judge_store(tmp_path, monkeypatch, FakeImportanceJudge())
    day1, day2, day3 = date(2026, 11, 20), date(2026, 11, 21), date(2026, 11, 22)
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: day1)
    receipt = _turn(store, "昨天去了一趟超市。", "t1")
    assert len(receipt["stored"]) == 1
    text = receipt["stored"][0]
    assert _row(store, text) == ("episodic", "normal")  # 第 1 天：一次性经历
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: day2)
    _turn(store, "昨天去了一趟超市。", "t2")
    assert _row(store, text) == ("episodic", "normal")  # 才 2 天 2 会，不晋升
    monkeypatch.setattr(store_mod, "_shanghai_today", lambda: day3)
    _turn(store, "昨天去了一趟超市。", "t3")
    assert _row(store, text)[0] == "stable"  # 3 天 3 会，兜底晋升


# ---------- #2-4：问题句/假设句/确认词/半句话/当下想法不存 ----------

@pytest.mark.parametrize("text", [
    "今天天气怎么样？",      # 问题句
    "今天天气怎么样。",      # 问题句（句尾是 。，也不能存）
    "如果每天定投5块会怎样",  # 假设句
    "好的",                 # 确认词
    "这几条取消",           # 半句话
    "我想喝茶了",           # 当下想法
    "我想喝茶了。",         # 当下想法（句尾是 。，也不能存）
])
def test_do_not_store(store, text):
    receipt = _turn(store, text, "t1")
    assert receipt["stored"] == []
    assert _live_texts(store) == []


# ---------- #2-5：判官断了降级，不崩、不丢、不问 ----------

def test_judge_down_degrades_to_episodic(tmp_path, monkeypatch):
    judge = FakeImportanceJudge(fail=True)
    store = _judge_store(tmp_path, monkeypatch, judge)
    receipt = _turn(store, "我每周三下午去游泳。", "t1")
    assert len(receipt["stored"]) == 1
    assert _row(store, receipt["stored"][0]) == ("episodic", "normal")
    assert "questions" not in receipt  # 模型断了不问 BOOS，兜底还能晋升


# ---------- #4-1：聊天里拿不准时，回执带原文+选项+原因 ----------

def _unsure_store(tmp_path, monkeypatch):
    judge = FakeImportanceJudge({
        "我表弟下个月可能来广州玩": {
            "important": "unsure", "reason": "既像长期事实又像一次性经历",
        },
    })
    return _judge_store(tmp_path, monkeypatch, judge)


def _unsure_turn(tmp_path, monkeypatch):
    store = _unsure_store(tmp_path, monkeypatch)
    receipt = _turn(store, "我表弟下个月可能来广州玩。", "t1")
    assert len(receipt["stored"]) == 1
    questions = receipt.get("questions")
    assert questions is not None and len(questions) == 1
    return store, receipt, questions[0]


def test_unsure_question_in_receipt(tmp_path, monkeypatch):
    store, receipt, question = _unsure_turn(tmp_path, monkeypatch)
    text = receipt["stored"][0]
    assert _row(store, text) == ("episodic", "normal")  # 拿不准先存着，不丢
    assert question["text"] == text  # 原文
    assert list(question["options"]) == ["进核心", "当一次性经历", "不存"]
    assert question["reason"]  # 原因非空
    assert question["kind"] == "importance"
    assert question["memory_id"] == receipt["results"][0]["id"]
    assert len(store.list_pending_questions()) == 1  # 问过、等回答


# ---------- #4-4：回答按答案改库 ----------

def test_answer_question_core(tmp_path, monkeypatch):
    store, receipt, question = _unsure_turn(tmp_path, monkeypatch)
    text = receipt["stored"][0]
    result = store.answer_question(question["id"], "进核心")
    assert result["ok"] is True and result["applied"] == "core"
    assert _row(store, text) == ("stable", "high")
    assert store.list_pending_questions() == []


def test_answer_question_episodic(tmp_path, monkeypatch):
    store, receipt, question = _unsure_turn(tmp_path, monkeypatch)
    text = receipt["stored"][0]
    result = store.answer_question(question["id"], "当一次性经历")
    assert result["ok"] is True and result["applied"] == "episodic"
    assert _row(store, text) == ("episodic", "normal")
    assert store.list_pending_questions() == []


def test_answer_question_drop(tmp_path, monkeypatch):
    store, receipt, question = _unsure_turn(tmp_path, monkeypatch)
    text = receipt["stored"][0]
    result = store.answer_question(question["id"], "不存")
    assert result["ok"] is True and result["applied"] == "dropped"
    assert result["deleted"] is True
    assert _live_texts(store) == []
    assert store.list_pending_questions() == []


def test_answer_question_bad_choice_does_not_consume(tmp_path, monkeypatch):
    store, receipt, question = _unsure_turn(tmp_path, monkeypatch)
    result = store.answer_question(question["id"], "随便")
    assert result["ok"] is False
    assert len(store.list_pending_questions()) == 1  # 选项不对，问题还在
    assert store.answer_question(question["id"], "进核心")["ok"] is True


# ---------- #4-2/#4-3：后台攒批 + 没新问题不打扰 ----------

def test_pending_questions_batch(tmp_path, monkeypatch):
    judge = FakeImportanceJudge({
        "表弟下个月可能来广州玩": {"important": "unsure", "reason": "拿不准1"},
        "朋友下周可能来上海出差": {"important": "unsure", "reason": "拿不准2"},
        "同事下个月可能去深圳培训": {"important": "unsure", "reason": "拿不准3"},
    })
    store = _judge_store(tmp_path, monkeypatch, judge)
    # 后台直存两条拿不准的：只攒着，不问
    store.add_memory_text("表弟下个月可能来广州玩", judge_importance=True)
    store.add_memory_text("朋友下周可能来上海出差", judge_importance=True)
    assert len(store.list_pending_questions()) == 2
    # 下次聊天有新问题时，顺带一起问（最多 3 条）
    receipt = _turn(store, "同事下个月可能去深圳培训。", "t1")
    questions = receipt.get("questions")
    assert questions is not None and len(questions) == 3
    assert questions[0]["text"] == "同事下个月可能去深圳培训"  # 本轮新的在前
    assert [q["text"] for q in questions[1:]] == [
        "表弟下个月可能来广州玩", "朋友下周可能来上海出差",
    ]
    # 没新问题时不打扰：回执里没有 questions
    receipt2 = _turn(store, "我喜欢吃牛肉面。", "t2")
    assert "questions" not in receipt2


# ---------- #2-6：自检第 10 项：核心记忆还管不管用 ----------
# 下面的 _meta / FakeAdapter / _judge_factory 照搬
# tests/test_self_check.py 同名帮手（自检适配器约定的最小实现），
# 只为测第 10 项；那边的 9 项回归仍以 test_self_check.py 为准。

_NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)


def _sc_meta(**over):
    base = {
        "source": "user",
        "confidence": 1.0,
        "user_confirmed": True,
        "slot": None,
        "is_rule": False,
        "turn_id": "turn-1",
        "decay_kind": "stable",
        "decay_state": "verbatim",
        "decay_delete_due": False,
        "decay_seen_dates": ["2026-11-23"],
        "decay_seen_sessions": ["turn-1"],
        "last_used_at": "2026-11-23T12:00:00Z",
        "tone": "serious",
        "tone_evidence": None,
        "tone_source": None,
        "voided": False,
        "superseded": False,
        "needs_review": False,
        "conflict_for_review": False,
        "grounding_flag": None,
    }
    base.update(over)
    return base


def _sc_item(item_id, text, **meta_over):
    return {"id": item_id, "text": text, "metadata": _sc_meta(**meta_over)}


class _ScFakeAdapter:
    """实现 self_check 适配器约定的假对象（离线）。"""

    def __init__(self, items=None):
        self._items = copy.deepcopy(items or [])

    def get_all_items(self):
        return copy.deepcopy(self._items)

    def supersede_events_since(self, days):
        return []

    def verify_backup(self):
        return {"ok": True, "detail": "恢复演练通过。"}

    def index_counts(self):
        return {"sqlite": 0, "chroma": 0, "mismatched_ids": []}

    def replay_promotion_counts(self):
        return {"checked": 0, "corrections": []}

    def archive_items(self):
        return []

    def embed(self, text):
        return None


def _sc_judge_factory(results):
    """按记忆 id 返回 canned 结论的假 judge（第 9 项同款写法）。"""

    def judge(system_prompt, user_prompt):
        for item_id, verdict in results.items():
            if "记忆 id：%s" % item_id in user_prompt:
                return verdict
        return None

    return judge


def test_self_check_item10_stale_proposes_only():
    items = [
        _sc_item("m1", "我住在广州"),
        _sc_item("m2", "我喜欢吃牛肉面"),
    ]
    judge = _sc_judge_factory({
        "m1": {"accurate": False, "evidence": "原文“我住在广州”：BOOS 已经搬走了。"},
        "m2": {"accurate": True, "evidence": "原文“我喜欢吃牛肉面”：还管用。"},
    })
    report = run_self_check(_ScFakeAdapter(items), now=_NOW, model_judge=judge)
    check = report["items"][10]
    assert check["status"] == "warn"
    assert any("m1" in action for action in check["actions"])  # 只提议沉底/归档
    stale = [f for f in check["findings"] if f.get("id") == "m1"]
    assert stale and stale[0]["accurate"] is False
    # 没自动改：输入的两条核心记忆原样还是 stable
    assert all(
        item["metadata"]["decay_kind"] == "stable" for item in items
    )


def test_self_check_item10_all_useful_pass():
    items = [_sc_item("m1", "我住在广州"), _sc_item("m2", "我喜欢吃牛肉面")]
    judge = _sc_judge_factory({
        "m1": {"accurate": True, "evidence": "原文“我住在广州”：还管用。"},
        "m2": {"accurate": True, "evidence": "原文“我喜欢吃牛肉面”：还管用。"},
    })
    report = run_self_check(_ScFakeAdapter(items), now=_NOW, model_judge=judge)
    assert report["items"][10]["status"] == "pass"
    assert report["items"][10]["actions"] == []


def test_self_check_item10_no_judge_skipped():
    items = [_sc_item("m1", "我住在广州")]
    report = run_self_check(_ScFakeAdapter(items), now=_NOW)
    assert report["items"][10]["status"] == "skipped"


def test_self_check_item10_no_core_skipped():
    items = [_sc_item("m1", "昨天去了一趟超市", user_confirmed=False,
                      decay_kind="episodic")]
    judge = _sc_judge_factory({
        "m1": {"accurate": True, "evidence": "原文“昨天去了一趟超市”：还管用。"},
    })
    report = run_self_check(_ScFakeAdapter(items), now=_NOW, model_judge=judge)
    assert report["items"][10]["status"] == "skipped"


def test_self_check_item10_all_undecided_skipped():
    items = [_sc_item("m1", "我住在广州")]
    judge = _sc_judge_factory({"m1": None})
    report = run_self_check(_ScFakeAdapter(items), now=_NOW, model_judge=judge)
    assert report["items"][10]["status"] == "skipped"
