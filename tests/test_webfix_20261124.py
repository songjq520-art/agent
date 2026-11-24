# tests/test_webfix_20261124.py（网页实测三类毛病修复，2026-11-24）
# 全部离线（向量打桩、判官按需打桩，不调真 API、不花钱）。
#
# 一、纠正流程：明确纠正命中已确认旧记忆时同一轮 supersede，
#    不许留活、不许把纠正原话/元句当现行事实存（实测：牛肉面
#    没当场顶掉 + “刚才那句说错了，我只吃鸡杂面”整句成了现行
#    事实）。根因三处：save_turn 叠剥把否定从句剥掉、判官路把
#    原话整句递给落盘、主目标被推断保护挂起时没人管已确认行。
# 二、重要性护栏 + 语气落账：问句永不 stable/high（程序硬上限）；
#    判官提议的 important 对单次提及的非核心内容封顶并留痕；
#    语气（本地筛 + 整理时模型判）必须写进 meta，带标记的压档、
#    不作偏好使用（实测：反话“太喜欢加班”stable+high 无标记、
#    问句“我的口味你还记得不”stable+high、猫/牛奶被判 high）。
# 三、冲突感知回忆（只改善）：pending 问题的新事实不许在检索里
#    隐身；同话题冲突要在给模型的记忆块里标明以哪条为准
#    （实测：问口味漏答刚纠正完的鸡杂面）。

import json

import pytest

from memory import ledger as _ledger
from memory.correction_text import (
    distill_correction_fact,
    is_correction_meta_sentence,
)
from memory.sqlite_store import SQLiteStore
from memory.timeutil import _now_iso
from main import _memory_block


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词
    路），语义判官打桩成不可用（模型断了降级走程序，是既定生产模式）。"""
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


def _id_of(store, text):
    row = store._conn.execute(
        "SELECT id FROM memories WHERE text=?", (text,)
    ).fetchone()
    return row[0] if row else None


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


# ---------- 一、纠正流程 ----------


def test_distill_correction_fact_unit():
    """提炼函数本身：元句剥成事实、普通事实原样、光秃秃元句返回 None。"""
    assert distill_correction_fact("刚才那句说错了，我只吃鸡杂面") == "我只吃鸡杂面"
    assert distill_correction_fact("我说错了，我住深圳") == "我住深圳"
    assert distill_correction_fact("BOOS 不吃牛肉，吃鸡杂面") == "BOOS 不吃牛肉，吃鸡杂面"
    assert distill_correction_fact("他记错了时间") == "他记错了时间"
    assert distill_correction_fact("刚才那句说错了") is None
    assert is_correction_meta_sentence("刚才那句说错了，我只吃鸡杂面")
    assert not is_correction_meta_sentence("我住深圳")
    assert not is_correction_meta_sentence("他记错了时间")


def test_correction_sequence_confirmed_superseded_same_turn(store):
    """复刻实测 1)+2) 全序列：
    种入已确认的“喜欢吃牛肉面”和推断来的“喜欢吃鸡杂面”；
    第一轮纠正后，已确认的牛肉面必须同轮被顶掉（不许留活等
    下一轮、不许靠回复反问），新事实带完整否定从句；
    第二轮元句纠正后，库里不许有整句原话当现行事实，落盘的
    必须是提炼后的“我只吃鸡杂面”。"""
    beef = _plant_row(
        store, "BOOS 喜欢吃牛肉面。",
        source="user", confidence=1.0, user_confirmed=True,
        decay_kind="stable", importance="high", write_seq=1,
    )
    _plant_row(store, "BOOS 喜欢吃鸡杂面。", write_seq=2)

    receipt = _turn(store, "不对，我其实不吃牛肉，我吃鸡杂面", "w1")
    # 已确认旧记忆同轮顶掉。
    assert _is_superseded(store, "BOOS 喜欢吃牛肉面。")
    # 新事实存活且带完整否定从句（叠剥护栏：不许只剩“我吃鸡杂面”）。
    live = _live_texts(store)
    assert any("不吃牛肉" in t and "鸡杂面" in t for t in live), live
    # 回执里留了同轮清扫的痕。
    correction = receipt.get("correction") or {}
    assert correction.get("also_superseded"), correction
    assert any(
        s["old_text"] == "BOOS 喜欢吃牛肉面。"
        for s in correction["also_superseded"]
    )

    # 第二轮：元句。判官打桩成“该顶推断那条”（复刻线上判官路）。
    inferred_id = _id_of(store, "BOOS 喜欢吃鸡杂面。")
    store._ask_semantic_judge = (
        lambda text, candidates, context=None: {
            "should_supersede": True, "target_id": inferred_id,
            "slot": "preference", "provider": "test",
            "model": "fake", "reason": "打桩判官",
        }
    )
    _turn(store, "刚才那句说错了，我只吃鸡杂面", "w2")
    live = _live_texts(store)
    # 原话整句不许成为现行事实；提炼后的事实必须在。
    assert not any("说错了" in t for t in live), live
    assert "我只吃鸡杂面" in live, live
    # 推断那条走保护：没被悄悄顶掉，攒了待问问题。
    assert not _is_superseded(store, "BOOS 喜欢吃鸡杂面。")
    qs = [
        q for q in store.list_pending_questions()
        if q.get("kind") == "conflict"
    ]
    assert any(q.get("new_id") == _id_of(store, "我只吃鸡杂面") for q in qs), qs


def test_apply_correction_result_distills_meta_sentence(store):
    """落盘收口：任何路把纠正原话递进 _apply_correction_result，
    存下来的现行事实必须是提炼后的，原话不许成为 current。"""
    target = _plant_row(store, "BOOS 喜欢吃鸡杂面。")
    outcome = store._apply_correction_result(
        "刚才那句说错了，我只吃鸡杂面",
        target, "BOOS 喜欢吃鸡杂面。", _meta_of(store, target),
        same_text=False, candidates=[target], status="updated",
        resolution="semantic_judge",
        audit={"source": "user", "method": "semantic_judge"},
        layer="judge", turn_id="cx",
    )
    assert outcome["new_text"] == "我只吃鸡杂面", outcome
    live = _live_texts(store)
    assert "我只吃鸡杂面" in live, live
    assert not any("说错了" in t for t in live), live


def test_negation_conflict_signal(store):
    """清扫的程序信号：否定辖域共享词才算正面冲突；同向否定不算。"""
    f = SQLiteStore._negation_conflicts
    assert f("我不吃牛肉，我吃鸡杂面", "BOOS 喜欢吃牛肉面。")
    assert not f("BOOS 不吃牛肉，吃鸡杂面", "BOOS 不吃牛肉，吃鸡杂面")
    assert not f("我吃鸡杂面", "BOOS 喜欢吃鸡杂面。")


# ---------- 二、重要性护栏 + 语气落账 ----------


def test_question_sentence_dropped_and_never_important(store):
    """问句“我的口味你还记得不”：不落账（问句闸补“记得不”句尾）；
    万一漏进重要性判定，护栏也保证判不出 important。"""
    receipt = _turn(store, "我的口味你还记得不", "q1")
    assert not any("口味" in t for t in _live_texts(store))
    assert any(
        d.get("reason") == "问句" for d in receipt["dropped"]
    ), receipt["dropped"]
    decision, _reason, _info = store._decide_importance("我的口味你还记得不")
    assert decision != "important"
    # 直存路开判官也一样压得住。
    saved = store.add_memory_text(
        "我的口味你还记得不", source="user", judge_importance=True
    )
    if saved.get("id"):
        meta = _meta_of(store, saved["id"])
        assert meta.get("decay_kind") != "stable"
        assert meta.get("importance") != "high"


def test_single_mention_judge_important_clamped(store, monkeypatch):
    """判官把单次提及的非核心内容判 important：程序封顶成
    episodic 并留痕；带明确强调的同类句不封顶（护栏不误伤）。"""
    monkeypatch.setattr(
        SQLiteStore, "_ask_importance_judge",
        lambda self, text, context=None: {
            "important": True, "reason": "判官判重要",
            "provider": "test", "model": "fake",
        },
    )
    receipt = _turn(store, "我买了一辆自行车", "c1")
    assert receipt["stored"], receipt
    meta = _meta_of(store, receipt["results"][0]["id"])
    assert meta["decay_kind"] == "episodic", meta
    assert meta["importance"] == "normal", meta
    assert (meta.get("importance_clamped") or {}).get("from") == "important", meta

    receipt2 = _turn(store, "我新买的自行车很重要", "c2")
    assert receipt2["stored"], receipt2
    meta2 = _meta_of(store, receipt2["results"][0]["id"])
    assert meta2["decay_kind"] == "stable", meta2
    assert meta2["importance"] == "high", meta2
    assert "importance_clamped" not in meta2, meta2


class _FakeToneRouter:
    """整理路由打桩：语气一律判 joke（模拟模型听出反话）。"""

    def judge(self, system_prompt, user_prompt, max_tokens=300,
              purpose=None, probe_primary=True):
        return {
            "text": json.dumps(
                {"tone": "joke", "evidence": "太喜欢加班"},
                ensure_ascii=False,
            )
        }


def test_sarcasm_tone_tagged_and_demoted_in_organize(store):
    """反话“我可太喜欢加班了”：先按实测那样存成 stable+high
    （类别判的），实时整理里模型判语气后必须落账——meta 带
    tone=joke/model，打标同步压档（不许再 stable/high），
    且不再作偏好候选使用。"""
    saved = store.add_memory_text(
        "我可太喜欢加班了", source="user",
        decay_kind="stable", importance="high",
    )
    mid = saved["id"]
    meta0 = _meta_of(store, mid)
    assert meta0["decay_kind"] == "stable" and meta0["importance"] == "high"
    assert "tone" not in meta0  # 本地筛认不出反话，存时没标（符合预期）

    store.organizing_router = lambda: _FakeToneRouter()
    summary = store._organize_turn_memories([mid], turn_id="s1")
    assert mid in summary["tone_tagged"], summary
    meta = _meta_of(store, mid)
    assert meta["tone"] == "joke", meta
    assert meta["tone_source"] == "model", meta
    assert meta["decay_kind"] == "episodic", meta
    assert meta["importance"] == "normal", meta
    # 账本里留了打标事件（PROMOTE 带 tone），重建不丢标。
    ledger_text = store.ledger_path.read_text(encoding="utf-8")
    assert '"tone": "joke"' in ledger_text
    # 不作偏好使用：判官候选里不再有它。
    cands = store._preference_judge_candidates("我喜欢吃火锅")
    assert all(c["id"] != mid for c in cands), cands


def test_local_tone_written_at_add(store):
    """本地筛当场认得出的语气（测试句）：存的时候就落账，
    且强制 episodic、不许 high。"""
    saved = store.add_memory_text(
        "这是测试，我喜欢吃火锅", source="user",
        decay_kind="stable", importance="high",
    )
    meta = _meta_of(store, saved["id"])
    assert meta["tone"] == "test", meta
    assert meta["tone_source"] == "local", meta
    assert meta["decay_kind"] == "episodic", meta
    assert meta["importance"] == "normal", meta


# ---------- 三、冲突感知回忆 ----------


def _plant_noodle_conflict_scene(store):
    """种入实测那组面条冲突状态（纠正完成、问题还挂着时）。"""
    inferred = _plant_row(store, "BOOS 喜欢吃鸡杂面。", write_seq=1)
    _plant_row(store, "BOOS 其实不是很喜欢吃鸡杂面。", write_seq=2)
    _plant_row(store, "BOOS 喜欢吃冒菜。", source="user",
               user_confirmed=True, write_seq=3)
    _plant_row(store, "BOOS 喜欢喝牛奶。", source="user",
               user_confirmed=True, write_seq=4)
    _plant_row(store, "BOOS 不吃牛肉，吃鸡杂面", source="user",
               user_confirmed=True, decay_kind="stable", write_seq=5)
    new_fact = _plant_row(store, "我吃鸡杂面", source="user",
                          user_confirmed=True, write_seq=6)
    store._queue_conflict_question(
        inferred, "BOOS 喜欢吃鸡杂面。", new_fact, "我吃鸡杂面",
    )
    return inferred, new_fact


def test_recall_conflict_surfaces_current_fact(store):
    """问口味类问题：鸡杂面必须出现在检索结果和给模型的记忆块里，
    冲突组要标明现行说法，块里要带“以最新确认的为准”的提示；
    无关的并存偏好（冒菜）不许被误标成冲突旧说法。"""
    _plant_noodle_conflict_scene(store)
    results = store.search("鸡杂面 牛肉面 我的口味", top_k=5)
    texts = [r["text"] for r in results]
    assert any("鸡杂面" in t for t in texts), texts
    currents = [
        r for r in results
        if (r.get("metadata") or {}).get("conflict_current") is True
    ]
    assert len(currents) == 1, results
    assert "鸡杂面" in currents[0]["text"], currents
    for r in results:
        if "冒菜" in r["text"]:
            assert (r.get("metadata") or {}).get("conflict_stale") is not True
    block = _memory_block(results)
    assert "鸡杂面" in block
    assert "互相矛盾的记录" in block, block


def test_pending_new_fact_swapped_into_results(store):
    """pending 问题的新事实本来排不进结果时（旧行在、新行不在），
    要被换进来——不许隐身。"""
    inferred, new_fact = _plant_noodle_conflict_scene(store)
    # 只问旧行能命中的话：结果里先只有推断旧行和填充偏好。
    results = store.search("喜欢吃鸡杂面", top_k=2)
    ids = [r.get("id") for r in results]
    assert new_fact in ids, results


def test_coexisting_preferences_not_flagged_as_conflict(store):
    """并存的两个偏好（冒菜 + 火锅）不是冲突：不许标
    conflict_current/stale，免得回答时胡说“以哪个为准”。"""
    _plant_row(store, "BOOS 喜欢吃冒菜。", source="user",
               user_confirmed=True, write_seq=1)
    _plant_row(store, "BOOS 喜欢吃火锅。", source="user",
               user_confirmed=True, write_seq=2)
    results = store.search("冒菜 火锅 喜欢吃什么", top_k=5)
    assert len(results) >= 2, results
    for r in results:
        meta = r.get("metadata") or {}
        assert meta.get("conflict_current") is not True, r
        assert meta.get("conflict_stale") is not True, r
    block = _memory_block(results)
    assert "互相矛盾的记录" not in block, block
