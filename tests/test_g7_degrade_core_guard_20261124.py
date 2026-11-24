# tests/test_g7_degrade_core_guard_20261124.py（G7 修复轮，2026-11-24）
# BOOS 批准的三道锁，堵“判官降级时程序路纠正误伤核心记忆”：
#   一、核心/置顶行不许被任何自动纠正路径取代（程序路、判官路
#       都不行）——命中就挂待答问题，新说法留痕存下；
#   二、判官不可用/降级期间，程序路解析出的纠正只挂起不执行
#       （add 直存路径不受影响、不丢事实）；
#   三、名字槽纠正先核对主语：带主语的整句不许被误认成碎片值
#       拼进任何一行，碎片补句不许拼核心行/名字行。
# 事故原形（上一轮 G7 实测，题材已换新）：判官被指到坏模型后，
# “慢慢其实是巴西红耳龟”被按名字槽解析，把真记忆“Agent 的名字
# 是十二，你叫十二”取代成了拼接行。本文件把事故形状编码成回归。
#
# 全部离线：向量打桩成 None；“判官关”用配置层解析失败模拟
# （resolve_judge_provider 抛 SemanticJudgeUnavailable——正是
# G7 实测里坏模型名触发的同款降级信号）；不调真 API、不烧钱。

import json

import pytest

from memory import ledger as _ledger
from memory import semantic_judge as _sj
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso

IDENTITY_TEXT = "Agent 的名字是十二，你叫十二"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """干净的 SQLiteStore：向量/语义判官问答打桩（同 fx8prot 口径）。
    注意：这只打桩“问答”层，不动配置层解析——三道锁之二的降级
    信号走配置层（见 judge_off），所以本夹具下判官算“可用”。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


@pytest.fixture
def judge_off(monkeypatch):
    """判官关：配置层解析失败（没接上/模型名不在配置里同款）。"""
    def _raise(*args, **kwargs):
        raise _sj.SemanticJudgeUnavailable("测试：判官不可用")
    monkeypatch.setattr(_sj, "resolve_judge_provider", _raise)


def _plant_row(store, text, **meta_over):
    """直接插一行（模拟已在库的旧行），meta 可覆盖。"""
    created_at = _now_iso()
    key = _ledger.ledger_event_key(text, created_at + text)
    meta = {
        "source": "user", "confidence": 1.0, "user_confirmed": True,
        "created_at": created_at, "write_seq": 1, "turn_id": None,
        "slot": None, "is_rule": False, "curated": False,
        "decay_kind": "episodic", "importance": "normal",
        "event_date": None,
    }
    meta.update(meta_over)
    store._insert_memory_row(key, text, meta, None)
    return key


def _plant_identity(store):
    return _plant_row(
        store, IDENTITY_TEXT,
        slot="agent_name", decay_kind="stable", importance="high",
    )


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _conflict_questions(store):
    return [
        q for q in store.list_pending_questions()
        if q.get("kind") == "conflict"
    ]


def _turn(store, text, turn_id):
    receipt = store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的"}],
        turn_id=turn_id,
    )
    # 等本轮的后台整理线程收尾，断言时状态已定。
    thread = getattr(store, "_turn_organize_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=30)
    return receipt


# ---------- 一、核心行上锁 ----------


def test_g7_core_correction_held_even_with_judge_on(store):
    """判官开着，纠正命中核心身份行：不取代、挂问题、新说法留痕；
    BOOS 答“换成新的”才真顶（人工批准路照常能改核心）。"""
    _plant_identity(store)
    outcome = store.correct_memory(
        IDENTITY_TEXT, "Agent 的名字是十三，你叫十三",
        explicit_correction=True,
    )
    assert outcome["status"] == "pending_question", outcome
    assert outcome.get("hold_reason") == "core_protected", outcome
    assert IDENTITY_TEXT in _live_texts(store)
    assert not _is_superseded(store, IDENTITY_TEXT)
    assert "Agent 的名字是十三，你叫十三" in _live_texts(store)
    qs = _conflict_questions(store)
    assert len(qs) == 1
    assert qs[0].get("hold_reason") == "core_protected"
    assert IDENTITY_TEXT in qs[0]["text"]
    # 人工批准：答“换成新的”后才真顶。
    answered = store.answer_question(qs[0]["id"], "换成新的")
    assert answered["ok"] and answered["applied"] == "superseded", answered
    assert _is_superseded(store, IDENTITY_TEXT)
    assert "Agent 的名字是十三，你叫十三" in _live_texts(store)


# ---------- 二、降级时只挂起不执行 ----------


def test_g7_degraded_program_correction_held_not_executed(store, judge_off):
    """判官关：普通非核心行（住处）的程序路纠正也不许直接取代——
    挂起 + 挂问题，新说法照常存下（不丢事实）。"""
    _plant_row(store, "我住北京", slot="residence")
    outcome = store.correct_memory(
        "我住北京", "我住深圳", explicit_correction=True,
    )
    assert outcome["status"] == "pending_question", outcome
    assert outcome.get("hold_reason") == "judge_degraded", outcome
    assert "我住北京" in _live_texts(store)
    assert not _is_superseded(store, "我住北京")
    assert "我住深圳" in _live_texts(store)
    qs = _conflict_questions(store)
    assert len(qs) == 1
    assert qs[0].get("hold_reason") == "judge_degraded"
    assert "我住北京" in qs[0]["text"] and "我住深圳" in qs[0]["text"]


def test_g7_degraded_hold_answer_buhuan_keeps_both(store, judge_off):
    """降级挂起后 BOOS 答“不换”：新旧两条都留，问题消耗掉。"""
    _plant_row(store, "我住北京", slot="residence")
    outcome = store.correct_memory(
        "我住北京", "我住深圳", explicit_correction=True,
    )
    answered = store.answer_question(outcome["question_id"], "不换")
    assert answered["ok"] and answered["applied"] == "kept_old", answered
    assert "我住北京" in _live_texts(store)
    assert "我住深圳" in _live_texts(store)
    assert not _is_superseded(store, "我住北京")
    assert store.list_pending_questions() == []


def test_g7_judge_on_normal_correction_still_updates(store):
    """对照组：判官可用时，普通纠正照旧立即取代（锁二不误伤）。"""
    _plant_row(store, "我住北京", slot="residence")
    outcome = store.correct_memory(
        "我住北京", "我住深圳", explicit_correction=True,
    )
    assert outcome["status"] == "updated", outcome
    assert _is_superseded(store, "我住北京")
    assert "我住深圳" in _live_texts(store)
    assert _conflict_questions(store) == []


def test_g7_degraded_add_path_unaffected(store, judge_off):
    """判官关：add 直存路径不受挂起影响，新事实照常落账。"""
    saved = store.add_memory_text("BOOS 喜欢喝酸梅汤", source="user")
    assert (saved or {}).get("event") == "ADD", saved
    assert "BOOS 喜欢喝酸梅汤" in _live_texts(store)


def test_g7_judge_runtime_failure_marks_degraded(tmp_path, monkeypatch):
    """运行时信号：真问判官失败（抛错）后本实例算降级；问通了
    （拿到裁决）自动清标。不打桩 _ask_semantic_judge 本身。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    raw = SQLiteStore(memory_dir=str(tmp_path / "memory"))

    def _boom(*args, **kwargs):
        raise RuntimeError("测试：判官调用失败")

    monkeypatch.setattr(_sj, "judge_semantic", _boom)
    assert raw._ask_semantic_judge("随便", []) is None
    assert raw._judge_degraded() is True

    monkeypatch.setattr(
        _sj, "judge_semantic",
        lambda *a, **k: {"should_supersede": False},
    )
    assert raw._ask_semantic_judge("随便", []) is not None
    assert raw._judge_runtime_failed is False
    assert raw._judge_degraded() is False


# ---------- 三、名字槽主语核对（事故形状，换新题材） ----------


def test_g7_pet_correction_never_touches_identity(store, judge_off):
    """事故复刻（猫题材）：判官关，存宠物事实后发“小橘其实是
    英短，不是橘猫”——身份行纹丝不动、不许出现拼接行、新说法
    留痕存下、旧宠物行也不许被悄悄顶掉。"""
    _plant_identity(store)
    _plant_row(store, "BOOS 新养了一只猫，叫小橘")
    _turn(store, "小橘其实是英短，不是橘猫", "g7-pet-turn")
    assert IDENTITY_TEXT in _live_texts(store)
    assert not _is_superseded(store, IDENTITY_TEXT)
    for text in _live_texts(store):
        assert "名字是小橘其实是英短" not in text
    assert "小橘其实是英短" in _live_texts(store)
    assert "BOOS 新养了一只猫，叫小橘" in _live_texts(store)
    assert not _is_superseded(store, "BOOS 新养了一只猫，叫小橘")


def test_g7_fragment_completion_guards(store):
    """碎片补句三道口径：
    - 库里只有身份行（事故当时的库形）：碎片“深圳”不许拼进
      身份行，原样返回；
    - 带主语的整句（“小橘其实是英短”）不许被当碎片值拼进任何
      一行（住处行也不行）；
    - 正经碎片补住处照旧能用（“深圳”→“我住深圳”）。"""
    _plant_identity(store)
    assert store._complete_correction_fragment("深圳") == "深圳"
    _plant_row(store, "我住北京", slot="residence")
    assert (
        store._complete_correction_fragment("小橘其实是英短")
        == "小橘其实是英短"
    )
    assert store._complete_correction_fragment("深圳") == "我住深圳"
    assert IDENTITY_TEXT in _live_texts(store)
    assert not _is_superseded(store, IDENTITY_TEXT)
