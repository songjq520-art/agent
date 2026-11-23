# tests/test_curated_and_permissions.py（预整理 + 记忆访问权限，2026-11-20）
# 测两条新规矩：
#   - 预整理：写时打 curated 标记、晋升时打上、玩笑/作废时摘掉、BOOS 确认
#     时打上；search() 精选先查（最前面、source="curated"、最多 3 条、
#     总数 <= top_k、同一文本不占两个位置）；精选记忆不沉底。
#   - 记忆访问权限：_authorize_content_change 各动作的允许/拒绝、拒绝时
#     记 warning 日志（不记原文）；BOOS 的删/纠正/撤回/槽位更新/金额
#     supersede 照常走通，system 顶不掉没判赢的旧条。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。

import logging
from datetime import datetime, timedelta, timezone

from conftest import make_fake_store
from memory.decay import DECAY_KIND_EPISODIC, DECAY_KIND_STABLE, decay_should_go_dormant
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    CONFIDENCE_USER,
    SOURCE_INFERRED,
    SOURCE_USER,
    _authorize_content_change,
    _extract_money_metadata,
    _provenance,
)
from memory.provenance import _normalize_metadata, _should_be_curated


def _utc_now():
    return datetime.now(timezone.utc)


def _add(store, text, metadata):
    result = store._add_text_with_metadata(text, metadata)
    return (result.get("results") or [{}])[0].get("id")


def _add_inferred(store, text):
    return _add(store, text, _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False))


def _add_user(store, text):
    return _add(store, text, _provenance(SOURCE_USER, CONFIDENCE_USER, True))


def _meta_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["metadata"]
    raise AssertionError("库里没有 %s" % memory_id)


def _text_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["text"]
    raise AssertionError("库里没有 %s" % memory_id)


def _age(store, memory_id, days):
    raw = store._raw_metadata_for(memory_id)
    assert raw is not None
    old = _utc_now() - timedelta(days=days)
    raw["last_used_at"] = old.isoformat(timespec="microseconds").replace("+00:00", "Z")
    store._memory.update(memory_id=memory_id, metadata=raw)


# ---------- 预整理：条件（纯函数 + 写时） ----------


def test_should_be_curated_conditions():
    """进清单的四条路 + 两条排除，照搬十一的 MEMORY.md。"""
    assert _should_be_curated({"source": "user", "user_confirmed": True}) is True
    assert (
        _should_be_curated(
            {"source": "inferred", "user_confirmed": False, "is_rule": True}
        )
        is True
    )
    assert (
        _should_be_curated(
            {
                "source": "inferred",
                "user_confirmed": False,
                "decay_kind": "stable",
            }
        )
        is True
    )
    assert (
        _should_be_curated({"source": "inferred", "user_confirmed": False})
        is False
    )
    assert (
        _should_be_curated(
            {"source": "user", "user_confirmed": True, "tone": "joke"}
        )
        is False
    )
    assert (
        _should_be_curated(
            {"source": "user", "user_confirmed": True, "tone": "test"}
        )
        is False
    )
    assert (
        _should_be_curated(
            {"source": "user", "user_confirmed": True, "voided": True}
        )
        is False
    )
    assert (
        _should_be_curated(
            {"source": "user", "user_confirmed": True, "slot": "agent_name"}
        )
        is True
    )


def test_provenance_marks_curated_on_write():
    """写时打标记：BOOS 亲口确认的、规则的进清单；模型提取的不进；
    玩笑的不进。"""
    assert (
        _provenance(SOURCE_USER, CONFIDENCE_USER, True).get("curated") is True
    )
    assert (
        _provenance(
            SOURCE_INFERRED, CONFIDENCE_INFERRED, False, is_rule=True
        ).get("curated")
        is True
    )
    assert (
        _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False).get("curated")
        is not True
    )
    assert (
        _provenance(SOURCE_USER, CONFIDENCE_USER, True, tone="joke").get(
            "curated"
        )
        is not True
    )


def test_write_marks_curated_for_user_confirmed(tmp_path):
    """BOOS 亲口说的写进去就带 curated，读回来还在（_normalize 不丢）。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_user(store, "BOOS 喜欢喝咖啡")
    assert _meta_of(store, memory_id)["curated"] is True
    assert _normalize_metadata(_meta_of(store, memory_id))["curated"] is True


def test_write_no_curated_for_plain_inferred(tmp_path):
    """模型提取的一次性经历写进去不带 curated。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_inferred(store, "BOOS 去盐湖城出差了。")
    assert _meta_of(store, memory_id)["curated"] is False


# ---------- 预整理：晋升打上、玩笑/作废摘掉、确认打上 ----------


def test_promotion_marks_curated(tmp_path):
    """不同天出现 3 次晋升成稳定事实时，curated 一起打上。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_inferred(store, "BOOS 去盐湖城出差了。")
    assert _meta_of(store, memory_id)["curated"] is False
    day2 = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
    day3 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert store._record_decay_occurrence(memory_id, now=day2) is True
    assert _meta_of(store, memory_id)["curated"] is False
    assert store._record_decay_occurrence(memory_id, now=day3) is True
    meta = _meta_of(store, memory_id)
    assert meta["decay_kind"] == DECAY_KIND_STABLE
    assert meta["curated"] is True


def test_void_joke_removes_curated(tmp_path):
    """作废时摘掉 curated，kind 也打回一次性。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_user(store, "BOOS 喜欢喝咖啡")
    assert _meta_of(store, memory_id)["curated"] is True
    assert store.void_joke_memory(memory_id)["ok"] is True
    meta = _meta_of(store, memory_id)
    assert meta["curated"] is False
    assert meta["voided"] is True
    assert meta["decay_kind"] == DECAY_KIND_EPISODIC


def test_tag_joke_removes_curated(tmp_path):
    """打标玩笑/测试/假设时摘掉 curated。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_user(store, "BOOS 喜欢喝咖啡")
    assert store._tag_memory_tone(memory_id, "joke", "开玩笑", "local") is True
    assert _meta_of(store, memory_id)["curated"] is False
    memory_id2 = _add_user(store, "BOOS 喜欢喝拿铁")
    assert store._tag_memory_tone(memory_id2, "test", "测试", "local") is True
    assert _meta_of(store, memory_id2)["curated"] is False


def test_confirm_memory_marks_curated(tmp_path):
    """BOOS 确认一条已存的推断记忆：user_confirmed 置 True，进精选清单。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_inferred(store, "BOOS 喜欢喝咖啡")
    assert _meta_of(store, memory_id)["curated"] is False
    assert store.confirm_memory(memory_id)["ok"] is True
    meta = _meta_of(store, memory_id)
    assert meta["user_confirmed"] is True
    assert meta["curated"] is True


def test_confirm_memory_refuses_joke(tmp_path):
    """玩笑/已作废的记忆确认不了。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_inferred(store, "BOOS 喜欢喝咖啡")
    assert store._tag_memory_tone(memory_id, "joke", "开玩笑", "local") is True
    assert store.confirm_memory(memory_id)["ok"] is False


# ---------- 预整理：检索精选先查 ----------


def test_search_curated_first(tmp_path):
    """精选在最前面、标 source="curated"；语义分 1.0 的也在它后面。"""
    store = make_fake_store(tmp_path)
    curated_id = _add_user(store, "我喜欢喝咖啡，每天早上都要喝一杯")
    semantic_id = _add_inferred(store, "我喜欢喝咖啡")
    results = store.search("我喜欢喝咖啡")
    assert results, "精选没捞回记忆"
    assert results[0]["id"] == curated_id
    assert results[0]["source"] == "curated"
    assert results[0]["layer"] == "active"
    assert semantic_id in [item["id"] for item in results]
    assert [item["id"] for item in results].index(curated_id) < [
        item["id"] for item in results
    ].index(semantic_id)


def test_search_curated_capped_at_three(tmp_path):
    """精选一次最多占 3 条。"""
    store = make_fake_store(tmp_path)
    for text in (
        "我喜欢喝咖啡",
        "我喜欢喝咖啡加糖",
        "我喜欢喝咖啡不加糖",
        "我喜欢喝咖啡配面包",
    ):
        _add_user(store, text)
    results = store.search("我喜欢喝咖啡", top_k=10)
    curated = [item for item in results if item["source"] == "curated"]
    assert len(curated) == 3
    assert len(results) <= 10


def test_search_curated_total_within_top_k(tmp_path):
    """总数不超过 top_k，精选在前、其余往后排。"""
    store = make_fake_store(tmp_path)
    _add_user(store, "我喜欢喝咖啡，每天早上都要喝一杯")
    _add_inferred(store, "我喜欢喝咖啡")
    _add_inferred(store, "我喜欢喝牛奶，每天早上都要喝一杯")
    results = store.search("我喜欢喝咖啡", top_k=2)
    assert len(results) == 2
    assert results[0]["source"] == "curated"


def test_search_same_text_single_seat(tmp_path):
    """同一文本不占两个位置：精选里有的，语义/关键词不再重复占位。"""
    store = make_fake_store(tmp_path)
    curated_id = _add_user(store, "我喜欢喝咖啡")
    _add_inferred(store, "我喜欢喝咖啡")
    results = store.search("我喜欢喝咖啡")
    texts = [item["text"] for item in results]
    assert texts.count("我喜欢喝咖啡") == 1
    assert results[0]["id"] == curated_id
    assert results[0]["source"] == "curated"


def test_search_no_curated_falls_back(tmp_path):
    """没有精选记忆时老路不变：没有 source="curated" 的结果。"""
    store = make_fake_store(tmp_path)
    target_id = _add_inferred(store, "我上次在圆通寄了快递")
    results = store.search("圆通快递怎么查")
    assert [item["id"] for item in results] == [target_id]
    assert all(item["source"] != "curated" for item in results)


# ---------- 预整理：精选不沉底 ----------


def test_curated_never_goes_dormant(tmp_path):
    """精选记忆（晋升进来的、非核心）40 天没人找也不标沉底。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_inferred(store, "BOOS 去盐湖城出差了。")
    day2 = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
    day3 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
    store._record_decay_occurrence(memory_id, now=day2)
    store._record_decay_occurrence(memory_id, now=day3)
    assert _meta_of(store, memory_id)["curated"] is True
    _age(store, memory_id, 40)
    ledger = store.sweep_monthly()
    assert ledger["dormant"] == []
    from memory.decay import DECAY_STATE_VERBATIM

    assert _meta_of(store, memory_id)["decay_state"] == DECAY_STATE_VERBATIM


def test_decay_should_go_dormant_curated():
    """纯函数：curated 的不沉底，没标记的照常沉。"""
    old = (_utc_now() - timedelta(days=40)).isoformat()
    assert (
        decay_should_go_dormant(DECAY_KIND_EPISODIC, old, is_curated=True)
        is False
    )
    assert (
        decay_should_go_dormant(DECAY_KIND_EPISODIC, old, is_curated=False)
        is True
    )


# ---------- 记忆访问权限：helper 各动作 ----------


def test_authorize_organize_always_allowed():
    """整理类永远允许。"""
    allowed, _reason = _authorize_content_change("organize", {}, "system")
    assert allowed is True


def test_authorize_delete_only_boos(caplog):
    """system 删已立记忆被拒绝，记 warning 日志（不记原文）。"""
    with caplog.at_level(logging.WARNING, logger="agent"):
        allowed, reason = _authorize_content_change(
            "delete", {"id": "fake-1", "text": "秘密原文"}, "system"
        )
    assert allowed is False
    assert "BOOS" in reason
    assert any("记忆权限拒绝" in r.getMessage() for r in caplog.records)
    assert all("秘密原文" not in r.getMessage() for r in caplog.records)
    allowed, _reason = _authorize_content_change(
        "delete", {"id": "fake-1"}, "boos"
    )
    assert allowed is True


def test_authorize_rewrite_only_boos():
    """system 改原文被拒绝，BOOS 允许。"""
    allowed, _reason = _authorize_content_change(
        "rewrite_text", {"id": "fake-1"}, "system"
    )
    assert allowed is False
    allowed, _reason = _authorize_content_change(
        "rewrite_text", {"id": "fake-1"}, "boos"
    )
    assert allowed is True


def test_authorize_overwrite_confirmed_needs_boos_and_confidence():
    """拿没确认的盖已确认的：BOOS + 新可信度不低于旧的才允许。"""
    meta = {"id": "fake-1", "confidence": 0.8}
    allowed, _reason = _authorize_content_change(
        "overwrite_confirmed", meta, "boos", new_confidence=0.9
    )
    assert allowed is True
    allowed, _reason = _authorize_content_change(
        "overwrite_confirmed", meta, "boos", new_confidence=0.5
    )
    assert allowed is False
    allowed, _reason = _authorize_content_change(
        "overwrite_confirmed", meta, "system", new_confidence=0.9
    )
    assert allowed is False
    allowed, _reason = _authorize_content_change(
        "overwrite_confirmed", meta, "boos", new_confidence=None
    )
    assert allowed is False


def test_authorize_restore_only_verbatim():
    """恢复只有文本逐字一样才允许。"""
    allowed, _reason = _authorize_content_change(
        "restore", {"id": "fake-1", "text": "原话"}, "system", new_text="原话"
    )
    assert allowed is True
    allowed, _reason = _authorize_content_change(
        "restore", {"id": "fake-1", "text": "原话"}, "system", new_text="改过的"
    )
    assert allowed is False
    allowed, _reason = _authorize_content_change(
        "restore", {"id": "fake-1", "text": "原话"}, "system", new_text=None
    )
    assert allowed is False


def test_authorize_supersede_only_when_new_wins():
    """supersede 只有 _resolve_conflict 判了新胜出才允许。"""
    allowed, _reason = _authorize_content_change(
        "supersede", {"id": "fake-1"}, "system", conflict_outcome="supersede_old"
    )
    assert allowed is True
    allowed, _reason = _authorize_content_change(
        "supersede", {"id": "fake-1"}, "system", conflict_outcome="pending_new"
    )
    assert allowed is False
    allowed, _reason = _authorize_content_change(
        "supersede", {"id": "fake-1"}, "system", conflict_outcome=None
    )
    assert allowed is False


def test_authorize_delete_just_written_allowed():
    """删本轮刚写的允许（系统，安全过滤/回滚）。"""
    allowed, _reason = _authorize_content_change(
        "delete_just_written", {"id": "fake-1"}, "system"
    )
    assert allowed is True


def test_authorize_unknown_action_denied(caplog):
    """未知动作默认拒绝。"""
    with caplog.at_level(logging.WARNING, logger="agent"):
        allowed, _reason = _authorize_content_change(
            "explode", {"id": "fake-1"}, "system"
        )
    assert allowed is False


# ---------- 记忆访问权限：接入点 ----------


def test_delete_memory_boos_allowed(tmp_path):
    """BOOS 删记忆照常走通（helper 不拦 BOOS）。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_user(store, "BOOS 喜欢喝咖啡")
    assert store.delete_memory(memory_id, approve=True) is True
    assert all(
        item["id"] != memory_id for item in store.get_all(include_sensitive=True)
    )


def test_soft_delete_memory_boos_allowed(tmp_path):
    """BOOS 撤回照常走通（helper 不拦 BOOS）。"""
    store = make_fake_store(tmp_path)
    memory_id = _add_user(store, "BOOS 喜欢喝咖啡")
    assert store.soft_delete_memory(memory_id) is True
    assert all(
        item["id"] != memory_id for item in store.get_all(include_sensitive=True)
    )


def test_supersede_money_needs_conflict_win(tmp_path):
    """金额 supersede：没判新胜出就不顶（旧条不动），判了才顶。"""
    store = make_fake_store(tmp_path)
    known_id = _add_user(store, "每天定投5块买万家纳指")
    known_item = {
        "id": known_id,
        "text": "每天定投5块买万家纳指",
        "metadata": _meta_of(store, known_id),
    }
    new_money = _extract_money_metadata("每天定投10块买万家纳指")
    assert (
        store._supersede_money_memory(
            known_item, "每天定投10块买万家纳指", new_money
        )
        is None
    )
    assert _text_of(store, known_id) == "每天定投5块买万家纳指"
    updated = store._supersede_money_memory(
        known_item,
        "每天定投10块买万家纳指",
        new_money,
        conflict_action="supersede_old",
    )
    assert updated is not None
    assert _text_of(store, known_id) == "每天定投10块买万家纳指"


def test_rollback_deletes_just_written(tmp_path):
    """回滚删本轮刚写的（delete_just_written，系统，记日志）。"""
    store = make_fake_store(tmp_path)
    first_id = _add_inferred(store, "BOOS 去盐湖城出差了。")
    second_id = _add_inferred(store, "BOOS 喜欢喝咖啡")
    leftover = store._rollback_new_memories(
        [{"id": first_id}, {"id": second_id}]
    )
    assert leftover == []
    assert store.get_all(include_sensitive=True) == []
