# tests/test_seqfix_step1.py（第一步修复 6 个 Bug 的回归测试，2026-11-20）
# 每个 Bug 修完补在这里；全部离线（FakeMemory），不碰真实 mem0、不联网。
# 依据：~/workspace/十二记忆代码审查核实-2026-11-20.md。

from datetime import datetime, timedelta, timezone

from conftest import make_fake_store
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    SOURCE_INFERRED,
    _provenance,
)


def _utc_now():
    return datetime.now(timezone.utc)


# ---------- Bug 1：作废的玩笑记忆不再被检索注入 ----------


def test_bug1_voided_memory_not_searchable(tmp_path):
    """Bug 1：作废的玩笑记忆不再被检索注入——语义、关键词、账本兜底
    三路都拦。修之前 search 会把它当成事实带回来。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我喜欢喝可乐", source="user")
    mem_id = (result.get("results") or [{}])[0].get("id")
    assert mem_id
    assert store.void_joke_memory(mem_id)["ok"] is True
    found = store.search("我喜欢喝什么")
    assert not any("可乐" in item["text"] for item in found), found
    # 问“以前”也不放出来：作废不是历史。
    found_history = store.search("我喜欢喝什么", include_history=True)
    assert not any("可乐" in item["text"] for item in found_history)


def test_bug1_project_and_filter_drops_voided_even_with_history():
    """Bug 1（筛子层）：project_and_filter 永远丢掉 voided 的项，
    include_history=True 也不放出来；正常的照常放行。"""
    from memory.ledger import project_and_filter

    state = {"by_id": {}, "key_states": {}}
    voided = [{"id": "m1", "text": "玩笑话", "metadata": {"voided": True}}]
    assert project_and_filter(voided, state) == []
    assert project_and_filter(voided, state, include_history=True) == []
    normal = [{"id": "m2", "text": "正常记忆", "metadata": {}}]
    out = project_and_filter(normal, state)
    assert len(out) == 1
    assert out[0]["is_history"] is False


# ---------- Bug 5：撤回要删掉 outbox 落盘文件 ----------


def _bug5_parked_turn(store, tmp_path, turn_id="turn-1"):
    """先让一轮经 outbox 写盘、再进待定区（文件还留着）：复现撤回前
    的状态。返回 (outbox, pending_dir)。"""
    from memory.outbox import MemoryOutbox

    pending_dir = tmp_path / "outbox_pending"
    outbox = MemoryOutbox(store, pending_dir=pending_dir)
    store._outbox = outbox
    messages = [
        {"role": "user", "content": "我喜欢吃拉面。"},
        {"role": "assistant", "content": "记下了。"},
    ]
    outbox.submit(messages, turn_id=turn_id)
    assert (pending_dir / f"{turn_id}.json").exists()
    # 后台存一遍：只进待定区（pending），文件还留着（没真正落库）。
    outbox._save_one(turn_id, messages)
    assert outbox.turn_status(turn_id) == "pending"
    assert (pending_dir / f"{turn_id}.json").exists()
    return outbox, pending_dir


def test_bug5_bare_retraction_drops_outbox_file(tmp_path):
    """Bug 5：光秃秃撤回（“不不不。”）时，待定轮的 outbox 落盘文件
    必须删掉——否则重启后 recover_pending 会把它捞回来重存，
    已撤回的内容就被悄悄存回去了。回执标 dropped，不是 committed。"""
    from memory.outbox import MemoryOutbox

    store = make_fake_store(tmp_path)
    outbox, pending_dir = _bug5_parked_turn(store, tmp_path)
    result = store.add_turn([
        {"role": "user", "content": "不不不。"},
        {"role": "assistant", "content": "好的，不算。"},
    ])
    assert result.get("retracted") is True
    assert not (pending_dir / "turn-1.json").exists()
    assert outbox.turn_status("turn-1") == "dropped"
    # 重启后捞不回来：文件没了，recover_pending 不该再交出这一轮。
    outbox2 = MemoryOutbox(store, pending_dir=pending_dir)
    assert outbox2.recover_pending() == 0


def test_bug5_judged_retraction_drops_outbox_file(tmp_path):
    """Bug 5（模型判意图那路撤回）：“其实也不是很喜欢吃。”判撤回后，
    同样删盘、标 dropped。"""
    store = make_fake_store(tmp_path)
    # 只测“判撤回之后”的程序行为，判本身靠模型（同 test_plan_c）。
    store._judge_retraction_intent = lambda *a, **k: True
    outbox, pending_dir = _bug5_parked_turn(store, tmp_path)
    result = store.add_turn([
        {"role": "user", "content": "其实也不是很喜欢吃。"},
        {"role": "assistant", "content": "好的。"},
    ])
    assert result.get("retracted") is True
    assert not (pending_dir / "turn-1.json").exists()
    assert outbox.turn_status("turn-1") == "dropped"


# ---------- Bug 2：keep_old 分支（旧条赢，新条当证据，不并存） ----------


def _bug2_old_money_memory(store):
    """种一条同槽位旧金额记忆：推断来源、写序号故意给得很大（比新批
    的还晚），让 _resolve_conflict 判 keep_old。返回旧条 id。"""
    from memory.mem0_store import (
        _provenance,
        CONFIDENCE_INFERRED,
        SOURCE_INFERRED,
    )

    old_meta = _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    old_meta["write_seq"] = 10 ** 9
    result = store._add_text_with_metadata("每天定投5块买万家纳指", old_meta)
    return (result.get("results") or [{}])[0].get("id")


def _bug2_evidence_texts(store, memory_id):
    import json

    raw = store._raw_metadata_for(memory_id)
    assert raw is not None
    evidence = raw.get("evidence") or ""
    if isinstance(evidence, str) and evidence:
        return [entry.get("text") for entry in json.loads(evidence)]
    return []


def test_bug2_dedupe_keep_old_appends_evidence(tmp_path):
    """Bug 2（提取路）：同槽位新金额判成 keep_old 时，旧条不动、新条
    当证据挂到旧条下、不并存两条。修之前新条照常留下来，新旧并存。"""
    store = make_fake_store(tmp_path)
    old_id = _bug2_old_money_memory(store)
    before = store.get_all()
    added = store._memory.add(
        [{"role": "user", "content": "每天定投10块买万家纳指"}],
        user_id=store.user_id,
        infer=False,
        metadata=None,
    )
    new_id = (added.get("results") or [{}])[0].get("id")
    out = store._dedupe_extraction_results(
        {"results": [{"id": new_id, "memory": "每天定投10块买万家纳指", "event": "ADD"}]},
        before,
        turn_id="t1",
    )
    assert out.get("money_kept_old") == 1
    assert store._memory.all_texts() == ["每天定投5块买万家纳指"]
    assert "每天定投10块买万家纳指" in _bug2_evidence_texts(store, old_id)


def test_bug2_add_memory_text_keep_old_appends_evidence(tmp_path):
    """Bug 2（手动存路）：同槽位新金额判成 keep_old 时，旧条不动、
    新说法当证据挂到旧条下、返回已有那条，不并存两条。"""
    store = make_fake_store(tmp_path)
    old_id = _bug2_old_money_memory(store)
    result = store.add_memory_text("每天定投10块买万家纳指", source="inferred")
    assert result.get("duplicate") is True
    assert (result.get("results") or [{}])[0].get("id") == old_id
    assert store._memory.all_texts() == ["每天定投5块买万家纳指"]
    assert "每天定投10块买万家纳指" in _bug2_evidence_texts(store, old_id)


# ---------- Bug 6：纠错候选池（上下文 + 同槽位，排除新说法自己） ----------


def test_bug6_correction_replaces_conflicting_not_identical(tmp_path):
    """Bug 6 场景 1：库里有“喜欢喝茶”（和新说法同义）和“喜欢喝咖啡”
    （冲突），用户说“你记错了，我喜欢喝茶”→ 该改的是咖啡那条，
    不是报 unchanged。修之前精确匹配命中了茶自己，咖啡纹丝不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝茶", source="user")
    store.add_memory_text("我喜欢喝咖啡", source="user")
    result = store.add_turn([
        {"role": "user", "content": "你记错了，我喜欢喝茶。"},
        {"role": "assistant", "content": "明白，我更新一下。"},
    ])
    assert result["correction"]["status"] == "updated"
    texts = [item["text"] for item in store.get_all()]
    assert not any("咖啡" in text for text in texts), texts
    assert any("喝茶" in text for text in texts)


def test_bug6_correction_finds_conflict_via_recent_context(tmp_path):
    """Bug 6 场景 2：新旧句字面重合低（“住上海” vs “住北京”），直接
    检索找不着冲突的旧条；但旧条在最近对话上下文里，兜底 recall
    找回来、照常纠正。修之前报 not_found，新说法单独存一条、旧的还在。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我家在北京", source="user")
    home_id = store.get_all()[0]["id"]
    store.note_context([{"id": home_id}])
    result = store.add_turn([
        {"role": "user", "content": "你记错了，我住上海。"},
        {"role": "assistant", "content": "明白，我更新一下。"},
    ])
    assert result["correction"]["status"] == "updated"
    assert result["correction"]["memory_id"] == home_id
    texts = [item["text"] for item in store.get_all()]
    assert not any("北京" in text for text in texts), texts


def test_bug6_fallback_candidates_from_context_and_slot(tmp_path):
    """Bug 6（兜底 recall 本身）：直接检索为空时，上下文 id 进候选池；
    同槽位（共同核心槽位）的旧记忆也进候选池。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我家在北京", source="user")
    home_id = store.get_all()[0]["id"]
    store.note_context([{"id": home_id}])
    cands = store._find_correction_candidates("我住上海", new_text="我住上海")
    assert [cand["id"] for cand in cands] == [home_id]
    assert cands[0]["source"] == "context"
    store2 = make_fake_store(tmp_path)
    store2.add_memory_text("我喜欢喝咖啡", source="user")
    coffee_id = store2.get_all()[0]["id"]
    slot_cands = store2._correction_fallback_candidates("我喜欢喝茶")
    assert any(
        cand["id"] == coffee_id and cand["source"] == "slot"
        for cand in slot_cands
    ), slot_cands


# ---------- Bug 4①：元数据入口清洗（_clean_chroma_metadata） ----------


def test_bug4a_clean_chroma_metadata_drops_none_and_flattens(tmp_path):
    """Bug 4①（清洗函数本身）：None 剔掉、list/dict 变 JSON 字符串、
    money 字典拍平——洗完只剩 Chroma 接受的标量值。"""
    from memory.provenance import _clean_chroma_metadata

    cleaned = _clean_chroma_metadata({
        "source": "user",
        "turn_id": None,
        "slot": None,
        "money": {
            "amount": 5,
            "currency": "CNY",
            "period": "day",
            "target": "万家纳指",
        },
        "decay_seen_dates": ["2026-11-20"],
        "custom_list": [1, 2],
        "custom_dict": {"a": 1},
        "confidence": 0.7,
        "flag": True,
    })
    assert cleaned["source"] == "user"
    assert "turn_id" not in cleaned and "slot" not in cleaned
    assert cleaned["money_amount"] == 5
    assert cleaned["money_target"] == "万家纳指"
    assert isinstance(cleaned["decay_seen_dates"], str)
    assert isinstance(cleaned["custom_list"], str)
    assert isinstance(cleaned["custom_dict"], str)
    assert cleaned["confidence"] == 0.7
    assert cleaned["flag"] is True
    assert all(
        isinstance(value, (str, int, float, bool))
        for value in cleaned.values()
    )


def test_bug4a_mem0_add_cleans_none_metadata(tmp_path):
    """Bug 4①（入口）：_mem0_add 之前先洗——底层收到的 metadata 里
    没有 None、没有 list/dict（对账重建那条路传进来的 turn_id/slot
    经常是 None）。"""
    from conftest import FakeMemory

    class _RecordingFakeMemory(FakeMemory):
        def __init__(self):
            super().__init__()
            self.seen_metadata = []

        def add(self, messages, user_id=None, infer=True, metadata=None):
            self.seen_metadata.append(metadata)
            return super().add(
                messages, user_id=user_id, infer=infer, metadata=metadata
            )

    mem = _RecordingFakeMemory()
    store = make_fake_store(tmp_path, memory=mem)
    store._mem0_add(
        [{"role": "user", "content": "我喜欢喝茶"}],
        store.user_id,
        False,
        {"source": "user", "turn_id": None, "slot": None, "custom": [1]},
    )
    assert mem.seen_metadata
    got = mem.seen_metadata[0]
    assert got is not None
    assert "turn_id" not in got and "slot" not in got
    assert all(
        isinstance(value, (str, int, float, bool)) for value in got.values()
    )


def test_bug4a_mem0_add_fallback_rewrites_clean_metadata(tmp_path):
    """Bug 4①（第二道保险）：底层不支持 metadata 时回退重存，随后把
    清洗过的 metadata 补写回去——新记忆不丢来源（write_seq 还在）。"""
    from conftest import FakeMemory
    from memory.mem0_store import (
        _provenance,
        CONFIDENCE_USER,
        SOURCE_USER,
    )

    class _NoMetadataFakeMemory(FakeMemory):
        def add(self, messages, user_id=None, infer=True, metadata=None):
            if metadata is not None:
                raise TypeError(
                    "add() got an unexpected keyword argument 'metadata'"
                )
            return super().add(
                messages, user_id=user_id, infer=infer, metadata=None
            )

    mem = _NoMetadataFakeMemory()
    store = make_fake_store(tmp_path, memory=mem)
    store._add_text_with_metadata(
        "我喜欢喝茶",
        _provenance(SOURCE_USER, CONFIDENCE_USER, True),
    )
    stored = store.get_all()
    assert [item["text"] for item in stored] == ["我喜欢喝茶"]
    meta = stored[0]["metadata"]
    assert meta["source"] == "user"
    assert isinstance(meta.get("write_seq"), int)


# ---------- Bug 3：search 管线重构（宽召回 + RRF + 先筛后截 + 条件档案闸） ----------


def _bug3_add_inferred(store, text):
    result = store._add_text_with_metadata(
        text, _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    )
    return (result.get("results") or [{}])[0].get("id")


def _bug3_age(store, memory_id, days):
    raw = store._raw_metadata_for(memory_id)
    assert raw is not None
    old = _utc_now() - timedelta(days=days)
    raw["last_used_at"] = old.isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    store._memory.update(memory_id=memory_id, metadata=raw)


def _bug3_meta_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["metadata"]
    raise AssertionError(f"库里没有 {memory_id}")


def _bug3_make_dormant(store, text):
    """存一条记忆、老化 40 天、月扫标沉底，返回 id。"""
    from memory.decay import DECAY_STATE_DORMANT

    memory_id = _bug3_add_inferred(store, text)
    _bug3_age(store, memory_id, 40)
    store.sweep_monthly()
    assert _bug3_meta_of(store, memory_id)["decay_state"] == DECAY_STATE_DORMANT
    return memory_id


def test_bug3_filter_before_truncation_keeps_good_item(tmp_path):
    """Bug 3（先筛后截）：语义前 5 全是被取代的，好的那条排第 6——
    旧流程语义只取 top_k=5，好的那条进不来、只能靠账本兜底捡回来
    （source 是 ledger、没 id）；新流程宽召回 20 条再筛，好的那条
    走主路回来（source 是 semantic、有 id）。"""
    store = make_fake_store(tmp_path)
    for suffix in ("呀", "呢", "啊", "哦", "吧"):
        old_id = _bug3_add_inferred(store, f"我喜欢喝咖啡{suffix}")
        outcome = store.correct_memory(
            f"我喜欢喝咖啡{suffix}", f"我现在不喝这个了{suffix}"
        )
        assert outcome["status"] == "updated", outcome
    good_id = _bug3_add_inferred(store, "我喜欢喝茶")
    results = store.search("我喜欢喝咖啡")
    assert [item["id"] for item in results] == [good_id]
    assert results[0]["source"] == "semantic"
    assert not any(
        "喝咖啡" in item["text"] for item in results
    ), [item["text"] for item in results]


def test_bug3_rrf_fusion_math():
    """Bug 3（RRF 融合分）：三路排名按 1/(60+rank) 相加，多路都有的
    在前——标准 RRF，k=60，b 在三路里都靠前，排第一。"""
    from memory.bm25 import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "a", "d"], ["c", "b"]], k=60
    )
    assert [doc_id for doc_id, _score in fused] == ["b", "a", "c", "d"]


def test_bug3_archive_gate_skipped_for_dated_query(tmp_path):
    """Bug 3（条件档案闸：日期）：常用层最好语义分 1.0 ≥ 0.6，但查询
    带“昨天”——问的就是旧账，档案层不屏蔽，沉底的也进来。"""
    store = make_fake_store(tmp_path)
    dormant_id = _bug3_make_dormant(store, "喝咖啡，每天一杯")
    active_id = _bug3_add_inferred(store, "昨天我喜欢喝咖啡")
    results = store.search("昨天我喜欢喝咖啡")
    ids = [item["id"] for item in results]
    assert active_id in ids
    assert dormant_id in ids
    dormant = next(item for item in results if item["id"] == dormant_id)
    assert dormant["layer"] == "archive"


def test_bug3_archive_gate_skipped_for_fund_entity(tmp_path):
    """Bug 3（条件档案闸：实体）：常用层最好语义分 0.8 ≥ 0.6，但查询
    点了基金“万家纳指”——问的就是这只基金，档案层不屏蔽。
    咖啡这类东西不算明确实体，别把闸翻过来（见
    test_archive_excluded_when_active_is_strong）。"""
    store = make_fake_store(tmp_path)
    dormant_id = _bug3_make_dormant(store, "万家纳指的基金代码是 019441")
    active_id = _bug3_add_inferred(store, "定投万家纳指")
    results = store.search("万家纳指定投怎么样")
    ids = [item["id"] for item in results]
    assert active_id in ids
    assert dormant_id in ids
    dormant = next(item for item in results if item["id"] == dormant_id)
    assert dormant["layer"] == "archive"


# ---------- Bug 4③：key 时间基准统一（correct_memory 复用旧 created_at） ----------


def test_bug4c_correction_reuses_old_created_at(tmp_path):
    """Bug 4③：纠正后新条 metadata 的 created_at 还是旧条那个时间
    （不是纠正那一刻的新时间），SUPERSEDE 事件的 current_key 等于
    ledger_event_key(new_text, old_created_at)——和
    _supersede_money_memory 一个基准。"""
    from memory import ledger as _ledger

    old_created_at = "2026-11-10T08:00:00.000000Z"
    store = make_fake_store(tmp_path)
    result = store._add_text_with_metadata(
        "我喜欢喝咖啡",
        _provenance(
            SOURCE_INFERRED, CONFIDENCE_INFERRED, False,
            created_at=old_created_at,
        ),
    )
    memory_id = (result.get("results") or [{}])[0].get("id")
    assert _bug3_meta_of(store, memory_id)["created_at"] == old_created_at
    outcome = store.correct_memory("我喜欢喝咖啡", "我喜欢喝茶")
    assert outcome["status"] == "updated", outcome
    assert _bug3_meta_of(store, memory_id)["created_at"] == old_created_at
    events, _info = _ledger.read_events(store.ledger_path)
    supersedes = [
        event for event in events
        if event.get("type") == _ledger.LEDGER_EVENT_SUPERSEDE
        and event.get("memory_id") == memory_id
    ]
    assert supersedes, "账本里没有这条纠正的 SUPERSEDE 事件"
    assert supersedes[-1].get("current_key") == _ledger.ledger_event_key(
        "我喜欢喝茶", old_created_at
    )
    assert supersedes[-1].get("key") == _ledger.ledger_event_key(
        "我喜欢喝咖啡", old_created_at
    )
