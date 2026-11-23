# tests/test_memory_ledger.py（新方案整合：账本 + 投影过滤 + 关系 + 备份恢复 + 快照）
# 测这些事：账本追加/序号/读/验/修（残行）、折叠（全部事件类型、
# key_states、同 id 取代 vs 跨 id 取代）、筛子（默认只拦取代和删、
# 睡着的不拦、问历史才给、metadata 兜底）、历史兜底（翻账本、
# search_history）、对账重建（丢了能回来、取代/删了的不复活）、
# 备份恢复（真恢复一轮、坏备份不认）、快照压缩（压完状态一样、
# 序号接着走）、只打标（月扫记 dormant 事件、待删翻转再记一笔、
# 删要 BOOS 点头）、关系（四种 + None、relate 往返）、先写账本
# 再写索引（账本坏了不写索引）、提取补记不重复。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。

import json
from datetime import datetime, timedelta, timezone

import pytest

from conftest import make_fake_store
from memory import backup as backup_mod
from memory import ledger as ledger_mod
from memory import relations as relations_mod
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    CONFIDENCE_USER,
    DECAY_STATE_DORMANT,
    SOURCE_INFERRED,
    SOURCE_USER,
    _provenance,
)


def _utc_now():
    return datetime.now(timezone.utc)


def _add(store, text):
    """add_memory_text 存一条，返回 id。"""
    result = store.add_memory_text(text)
    items = (result or {}).get("results") or []
    assert items, f"存不进去：{text}"
    return items[0].get("id")


def _texts(store):
    return [item["text"] for item in store.get_all(include_sensitive=True)]


def _add_inferred(store, text):
    """存一条模型提取的记忆（episodic，一次性经历，半衰期 14 天），
    记账本（best_effort）。睡着/月扫测试用它——add_memory_text 是
    BOOS 亲口说的（stable，半衰期 180 天），40 天沉不下去。"""
    result = store._add_text_with_metadata(
        text,
        _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False),
        ledger_mode="best_effort",
    )
    items = (result or {}).get("results") or []
    assert items, f"存不进去：{text}"
    return items[0].get("id")


def _age(store, memory_id, days):
    """把一条记忆的 last_used_at 拨到几天前（模拟“很久没用”）。"""
    raw = store._raw_metadata_for(memory_id)
    assert raw is not None
    old = _utc_now() - timedelta(days=days)
    raw["last_used_at"] = old.isoformat(timespec="microseconds").replace("+00:00", "Z")
    store._memory.update(memory_id=memory_id, metadata=raw)


def _meta_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["metadata"]
    raise AssertionError(f"库里没有 {memory_id}")


def _ledger_events(store):
    events, _info = ledger_mod.read_events(store.ledger_path)
    return events


# ---------- 1. 账本：追加、序号、读、验、修 ----------


def test_ledger_append_assigns_seq_and_reads_back(tmp_path):
    """追加两笔，序号 1、2；读回来原话、key 都在；验账本通过。"""
    path = tmp_path / "events.jsonl"
    first = ledger_mod.append_event(
        path,
        {"type": "add", "key": "k1", "text": "我喜欢喝冰美式。"},
    )
    second = ledger_mod.append_event(
        path,
        {"type": "add", "key": "k2", "text": "我爱吃冒菜。"},
    )
    assert first["seq"] == 1
    assert second["seq"] == 2
    events, info = ledger_mod.read_events(path)
    assert [e["text"] for e in events] == ["我喜欢喝冰美式。", "我爱吃冒菜。"]
    assert info == {"truncated": False, "bad_lines": 0}
    assert ledger_mod.verify_ledger(path)["ok"] is True


def test_ledger_rejects_unknown_event_type(tmp_path):
    """没名没姓的事件类型不让进账本：早点炸，别等查账时才发现。"""
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        ledger_mod.append_event(path, {"type": "whatever", "text": "x"})


def test_ledger_truncated_tail_is_dropped_and_repaired(tmp_path):
    """写一半没结尾的行：读的时候认出来扔掉，不算数；修完序号接着走。"""
    path = tmp_path / "events.jsonl"
    ledger_mod.append_event(path, {"type": "add", "key": "k1", "text": "第一条。"})
    ledger_mod.append_event(path, {"type": "add", "key": "k2", "text": "第二条。"})
    # 手动接一行没写完的（没换行符）：模拟写到一半服务崩了。
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "add", "key": "k3", "text": "写一半')
    events, info = ledger_mod.read_events(path)
    assert [e["text"] for e in events] == ["第一条。", "第二条。"]
    assert info["truncated"] is True
    check = ledger_mod.verify_ledger(path)
    assert check["ok"] is False
    assert check["truncated"] is True
    repaired = ledger_mod.repair_ledger(path)
    assert repaired == {"repaired": True, "dropped": 1}
    events, info = ledger_mod.read_events(path)
    assert len(events) == 2
    assert info["truncated"] is False
    third = ledger_mod.append_event(
        path, {"type": "add", "key": "k3", "text": "第三条。"}
    )
    assert third["seq"] == 3


def test_ledger_verify_flags_bad_middle_line(tmp_path):
    """中间坏了一行（不是末尾）：验出来报 bad_lines，不悄悄跳过。"""
    path = tmp_path / "events.jsonl"
    ledger_mod.append_event(path, {"type": "add", "key": "k1", "text": "第一条。"})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("这行不是 JSON\n")
    ledger_mod.append_event(path, {"type": "add", "key": "k2", "text": "第二条。"})
    check = ledger_mod.verify_ledger(path)
    assert check["ok"] is False
    assert check["bad_lines"] == 1
    events, _info = ledger_mod.read_events(path)
    assert [e["text"] for e in events] == ["第一条。", "第二条。"]


# ---------- 2. 折叠：全部事件类型、key_states ----------


def test_resolve_state_folds_all_event_types(tmp_path):
    """折叠认全部事件：add 进档案、跨 id 取代标旧 id、同 id 取代只记
    history、dormant 带 delete_due、revive 回 current、delete 标删、
    relate 进关系；key_states 跟着走（旧 key 取代、新 key 当前）。"""
    events = [
        {"type": "add", "key": "k-old", "text": "我喜欢喝冰美式。", "seq": 1},
        {"type": "add", "key": "k-new", "text": "我喜欢喝白开水。", "seq": 2},
        {
            "type": "supersede",
            "memory_id": "m-new",
            "supersedes_id": "m-old",
            "key": "k-old",
            "current_key": "k-new",
            "seq": 3,
        },
        {"type": "dormant", "memory_id": "m-new", "key": "k-new", "seq": 4},
        {"type": "revive", "memory_id": "m-new", "key": "k-new", "seq": 5},
        {"type": "delete", "memory_id": "m-new", "key": "k-new", "seq": 6},
        {
            "type": "relate",
            "memory_id": "m-new",
            "related_id": "m-other",
            "relation": "related",
            "seq": 7,
        },
    ]
    state = ledger_mod.resolve_state(events)
    assert state["by_id"]["m-old"]["state"] == "superseded"
    assert state["by_id"]["m-new"]["state"] == "deleted"
    assert state["key_states"]["k-old"] == "superseded"
    assert state["key_states"]["k-new"] == "deleted"
    assert [a["key"] for a in state["adds"]] == ["k-old", "k-new"]
    assert len(state["relations"]) == 1
    assert state["seq_max"] == 7


def test_resolve_state_same_id_supersede_keeps_current(tmp_path):
    """同 id 取代（原地纠正、原地改金额）：状态还是 current，旧话只
    记一条 history；但旧原话的 key 标 superseded——旧说法不再是
    当前的，对账不会把它复活回来。"""
    state = ledger_mod.resolve_state(
        [
            {"type": "add", "key": "k-old", "text": "我喜欢喝冰美式。", "seq": 1},
            {
                "type": "supersede",
                "memory_id": "m1",
                "supersedes_id": "m1",
                "key": "k-old",
                "current_key": "k-new",
                "old_text": "我喜欢喝冰美式。",
                "seq": 2,
            },
        ]
    )
    assert state["by_id"]["m1"]["state"] == "current"
    assert len(state["by_id"]["m1"]["history"]) == 1
    assert state["key_states"]["k-old"] == "superseded"
    assert state["key_states"]["k-new"] == "current"
    assert state["id_to_key"]["m1"] == "k-new"


def test_resolve_state_revive_without_key_uses_id_map(tmp_path):
    """只有 id 的 revive（_decay_touch 手头没原话）：按 id_to_key 找到
    之前 dormant 事件留下的 key，把它标回 current——唤醒不能因为
    事件没带 key 就失效。"""
    state = ledger_mod.resolve_state(
        [
            {"type": "add", "key": "k1", "text": "我喜欢喝冰美式。", "seq": 1},
            {"type": "dormant", "memory_id": "m1", "key": "k1", "seq": 2},
            {"type": "revive", "memory_id": "m1", "seq": 3},
        ]
    )
    assert state["key_states"]["k1"] == "current"
    assert state["by_id"]["m1"]["state"] == "current"


# ---------- 3. 筛子 ----------


def test_project_and_filter_hides_superseded_and_deleted_only(tmp_path):
    """筛子默认只拦取代和删：睡着的不拦（它是当前记忆，只是很久没用，
    衰减分支的定法）；问“以前”时才把取代/删了的放出来，并标清状态。"""
    state = {
        "by_id": {
            "m-sup": {"state": "superseded"},
            "m-dor": {"state": "dormant"},
            "m-del": {"state": "deleted"},
        }
    }
    items = [
        {"id": "m-sup", "text": "旧说法。"},
        {"id": "m-dor", "text": "睡着的说法。"},
        {"id": "m-del", "text": "删了的说法。"},
        {"id": "m-cur", "text": "当前说法。"},
    ]
    out = ledger_mod.project_and_filter(items, state)
    assert [i["id"] for i in out] == ["m-dor", "m-cur"]
    out_all = ledger_mod.project_and_filter(items, state, include_history=True)
    assert [i["id"] for i in out_all] == ["m-sup", "m-dor", "m-del", "m-cur"]
    assert out_all[0]["is_history"] is True
    assert out_all[0]["history_state"] == "superseded"
    assert out_all[2]["history_state"] == "deleted"
    assert out_all[1].get("is_history") is not True


def test_project_and_filter_metadata_fallback(tmp_path):
    """账本里没记录的老记忆，退到 mem0 metadata 上的 superseded 标记——
    账本启用之前就被标过的，不能因为账本没记就放出来。"""
    state = {"by_id": {}}
    items = [
        {"id": "m1", "text": "旧说法。", "metadata": {"superseded": True}},
        {"id": "m2", "text": "当前说法。", "metadata": {}},
    ]
    out = ledger_mod.project_and_filter(items, state)
    assert [i["id"] for i in out] == ["m2"]


# ---------- 4. 历史兜底 ----------


def test_find_in_ledger_keyword_fallback(tmp_path):
    """翻账本按关键词找：命中线 ≥2，和检索里的关键词补充信号一致；
    已经给过的 key 不重复给。"""
    events = [
        {"type": "add", "key": "k1", "text": "我喜欢喝冰美式。", "seq": 1},
        {"type": "add", "key": "k2", "text": "我爱吃冒菜。", "seq": 2},
        {"type": "add", "key": "k3", "text": "我每天早上 7 点起床。", "seq": 3},
    ]
    hits = ledger_mod.find_in_ledger(events, "冒菜好吃吗", top_k=5)
    assert [h["key"] for h in hits] == ["k2"]
    hits = ledger_mod.find_in_ledger(events, "冒菜好吃吗", top_k=5, exclude_keys={"k2"})
    assert hits == []


def test_search_history_returns_ledger_only(tmp_path):
    """search_history 只翻账本：返回的都是历史（is_history=True，
    source=ledger），索引里没有也能找到；关键词一个没命中返回空。"""
    store = make_fake_store(tmp_path)
    _add(store, "我爱吃冒菜。")
    store._memory.delete(store.get_all()[0]["id"])
    assert store.get_all() == []
    out = store.search_history("冒菜")
    assert len(out) == 1
    assert out[0]["text"] == "我爱吃冒菜。"
    assert out[0]["is_history"] is True
    assert out[0]["source"] == "ledger"
    assert store.search_history("火锅") == []


# ---------- 5. 检索集成：纠正、删除、睡着、兜底不复活 ----------


def test_search_after_correction_hides_old_text_but_history_keeps_it(tmp_path):
    """纠正后：默认检索里旧说法不见了（筛子拦的，不是索引里还有）；
    翻账本还能找回“以前记错的那句”。"""
    store = make_fake_store(tmp_path)
    _add(store, "我喜欢喝冰美式。")
    outcome = store.correct_memory("我喜欢喝冰美式。", "我喜欢喝白开水。")
    assert outcome["status"] == "updated"
    found = store.search("我喜欢喝什么")
    assert not any("冰美式" in item["text"] for item in found)
    assert any("白开水" in item["text"] for item in found)
    history = store.search_history("冰美式")
    assert any("冰美式" in item["text"] for item in history)


def test_search_after_soft_delete_hides_text_but_history_keeps_it(tmp_path):
    """软删除后：默认检索里不见了；翻账本还能找回（删了什么、什么
    时候删的，账本里有数）；恢复后又能搜到了。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我爱吃冒菜。")
    assert store.soft_delete_memory(mem_id) is True
    assert not any("冒菜" in item["text"] for item in store.search("冒菜"))
    assert any("冒菜" in item["text"] for item in store.search_history("冒菜"))
    assert store.restore_memory(mem_id) is True
    assert any("冒菜" in item["text"] for item in store.search("冒菜"))


def test_search_fallback_does_not_resurface_superseded_text(tmp_path):
    """兜底不复活：旧说法纠正后、索引里也没了，翻账本也不把它当现况
    给（默认检索里没有它）；问历史时才给。"""
    store = make_fake_store(tmp_path)
    _add(store, "我喜欢喝冰美式。")
    store.correct_memory("我喜欢喝冰美式。", "我喜欢喝白开水。")
    # 索引里把当前那条也弄丢（模拟索引坏了）：兜底只能翻到旧说法，
    # 但旧说法是 superseded，不能当现况给。
    store._memory.delete(store.get_all()[0]["id"])
    assert store.search("冰美式") == []
    assert any("冰美式" in item["text"] for item in store.search_history("冰美式"))


def test_dormant_memory_still_searchable_and_wakes_on_use(tmp_path):
    """睡着的照常能搜到（筛子不拦它），被用上就唤醒回原话态——
    和衰减分支的定法一致，账本不能把衰减的行为改掉。"""
    store = make_fake_store(tmp_path)
    mem_id = _add_inferred(store, "BOOS 上周去了一趟盐湖城出差。")
    _age(store, mem_id, 40)
    store.sweep_monthly()
    assert _meta_of(store, mem_id)["decay_state"] == DECAY_STATE_DORMANT
    results = store.search("盐湖城出差", reinforce=True)
    assert [item["id"] for item in results] == [mem_id]
    assert _meta_of(store, mem_id)["decay_state"] != DECAY_STATE_DORMANT


# ---------- 6. 对账重建 ----------


def test_reconcile_rebuilds_index_from_ledger(tmp_path):
    """索引丢了能从账本重建：直接删掉向量（不走 delete_memory，不记
    delete 事件），对账按账本原话写回来；再对一次就没得补了。"""
    store = make_fake_store(tmp_path)
    _add(store, "我喜欢喝冰美式。")
    _add(store, "我爱吃冒菜。")
    store._memory.delete(store.get_all()[0]["id"])
    assert len(store.get_all()) == 1
    result = store.reconcile_ledger_index()
    assert result["reconciled"] == 1
    assert sorted(_texts(store)) == ["我喜欢喝冰美式。", "我爱吃冒菜。"]
    again = store.reconcile_ledger_index()
    assert again["reconciled"] == 0



def test_reconcile_skips_when_same_text_already_in_store(tmp_path):
    """key 对不上但相同文本已在索引里时，对账不重建（不造重复）。

    2026-11-20 冒菜第 21 份的根因：账本 seq 3（正规冒菜记忆）的 key
    永远对不上库里的副本（重建时 metadata 回退丢了 created_at，
    新副本的 key 和账本 key 不同），每次启动对账都把它当“缺了”
    又重建一份。修法是重建前按内容再查一次：相同文本已在就跳过。
    """
    store = make_fake_store(tmp_path)
    _add(store, "我爱的是冒菜，不是麻辣烫")
    before = store.get_all()
    old_id = before[0]["id"]
    # 模拟 key 失配：索引里的副本换了个新 created_at（像重建/恢复
    # 时 metadata 回退丢了原 created_at），账本的 key 再也对不上。
    store._memory.delete(old_id)
    store._add_text_with_metadata(
        "我爱的是冒菜，不是麻辣烫",
        _provenance(SOURCE_USER, CONFIDENCE_USER, True),
    )
    assert len(store.get_all()) == 1
    result = store.reconcile_ledger_index()
    assert result["reconciled"] == 0
    # 没重建：库里还是只有 1 份冒菜。
    assert len(store.get_all()) == 1
    assert _texts(store) == ["我爱的是冒菜，不是麻辣烫"]


def test_reconcile_does_not_resurrect_superseded_or_deleted(tmp_path):
    """对账不复活：纠正过的旧说法、删了的，索引丢了也不补回来；
    纠正后的新说法能补回来（它有自己的 add 事件）。"""
    store = make_fake_store(tmp_path)
    _add(store, "我喜欢喝冰美式。")
    store.correct_memory("我喜欢喝冰美式。", "我喜欢喝白开水。")
    _add(store, "我爱吃冒菜。")
    doomed = store.get_all()[-1]["id"]
    store.soft_delete_memory(doomed)
    # 索引全丢：当前那条（白开水）靠账本回来，旧说法和删了的不回来。
    for item in list(store.get_all()):
        store._memory.delete(item["id"])
    assert store.get_all() == []
    result = store.reconcile_ledger_index()
    assert result["reconciled"] == 1
    assert _texts(store) == ["我喜欢喝白开水。"]


def test_reconcile_is_idempotent_and_never_raises_on_empty_ledger(tmp_path):
    """账本是空的：对账返回 0，不抛异常——刚启用的 store 也能对账。"""
    store = make_fake_store(tmp_path)
    assert store.reconcile_ledger_index() == {
        "reconciled": 0,
        "missing": [],
        "failed": [],
    }


# ---------- 7. 备份恢复 ----------


def test_backup_and_real_restore_round_trip(tmp_path):
    """真恢复一轮：A 店备份 → 全新的 B 店恢复 → 原话、账本都在；
    再恢复一次，重复的都跳过（不重复存）。"""
    store_a = make_fake_store(tmp_path / "a")
    _add(store_a, "我喜欢喝冰美式。")
    _add(store_a, "我爱吃冒菜。")
    backup = store_a.backup_memories(backup_dir=tmp_path / "backups")
    assert backup_mod.verify_backup(backup["backup_path"])["ok"] is True
    store_b = make_fake_store(tmp_path / "b")
    assert store_b.get_all() == []
    result = store_b.restore_from_backup(backup["backup_path"])
    assert len(result["restored_items"]) == 2
    assert sorted(_texts(store_b)) == ["我喜欢喝冰美式。", "我爱吃冒菜。"]
    assert len(_ledger_events(store_b)) == 2
    again = store_b.restore_from_backup(backup["backup_path"])
    assert again["restored_items"] == []
    assert len(again["skipped_items"]) == 2
    assert sorted(_texts(store_b)) == ["我喜欢喝冰美式。", "我爱吃冒菜。"]


def test_backup_verify_rejects_tampered_backup(tmp_path):
    """备份被改过：校验和对不上，验不过；恢复直接拒绝，一个也不恢复。"""
    store = make_fake_store(tmp_path)
    _add(store, "我喜欢喝冰美式。")
    backup = store.backup_memories(backup_dir=tmp_path / "backups")
    payload = json.loads(open(backup["backup_path"], encoding="utf-8").read())
    payload["items"][0]["text"] = "我喜欢喝白开水。"
    open(backup["backup_path"], "w", encoding="utf-8").write(
        json.dumps(payload, ensure_ascii=False)
    )
    assert backup_mod.verify_backup(backup["backup_path"])["ok"] is False
    fresh = make_fake_store(tmp_path / "fresh")
    with pytest.raises(ValueError):
        fresh.restore_from_backup(backup["backup_path"])
    assert fresh.get_all() == []


def test_backup_with_broken_snapshot_is_not_trusted(tmp_path):
    """备份时快照读坏了：备份照常写，但验备份判不过——缺了快照的
    备份是不完整的，不能当“能恢复”来认。"""
    payload = backup_mod.build_backup(
        [], [], snapshot=None, snapshot_error="快照校验和对不上"
    )
    path = tmp_path / "b.json"
    backup_mod.write_backup(path, payload)
    check = backup_mod.verify_backup(path)
    assert check["ok"] is False
    assert any("快照" in e for e in check["errors"])


# ---------- 8. 快照压缩 ----------


def test_compact_ledger_keeps_state_and_seq_continues(tmp_path):
    """压一年：压完状态一模一样（by_id、原话、关系、key_states 都在
    快照里），账本只剩一行标记；再记一笔，序号接着往下走、不断档。"""
    store = make_fake_store(tmp_path)
    _add(store, "我喜欢喝冰美式。")
    _add(store, "我爱吃冒菜。")
    store.correct_memory("我喜欢喝冰美式。", "我喜欢喝白开水。")
    before = store.ledger_state()
    result = store.compact_ledger()
    assert result["snapshot_seq"] == before["seq_max"]
    after = store.ledger_state()
    assert after["by_id"] == before["by_id"]
    assert after["adds"] == before["adds"]
    assert after["key_states"] == before["key_states"]
    assert after["relations"] == before["relations"]
    events, _info = ledger_mod.read_events(store.ledger_path)
    assert len(events) == 1
    assert events[0]["type"] == "snapshot"
    nxt = ledger_mod.append_event(
        store.ledger_path,
        {"type": "add", "key": "k-after", "text": "压完再记一条。"},
    )
    assert nxt["seq"] == before["seq_max"] + 2
    state = store.ledger_state()
    assert any(a["key"] == "k-after" for a in state["adds"])
    assert state["key_states"]["k-after"] == "current"


def test_snapshot_checksum_mismatch_is_rejected(tmp_path):
    """快照被改过：读的时候校验和对不上，直接抛，不悄悄认一个坏快照。"""
    store = make_fake_store(tmp_path)
    _add(store, "我喜欢喝冰美式。")
    store.compact_ledger()
    payload = json.loads(
        open(store.snapshot_path, encoding="utf-8").read()
    )
    payload["adds"][0]["text"] = "改过的原话。"
    open(store.snapshot_path, "w", encoding="utf-8").write(
        json.dumps(payload, ensure_ascii=False)
    )
    with pytest.raises(ValueError):
        ledger_mod.read_snapshot(store.snapshot_path)


# ---------- 9. 只打标 ----------


def test_sweep_marks_dormant_in_ledger_and_delete_due_flips(tmp_path):
    """月扫只打标：40 天标 dormant、账本记一笔（delete_due=False）；
    再老到 180 天，待删翻成 True 时账本再记一笔（delete_due=True）；
    原话和文本都不动、不压摘要、不删。"""
    store = make_fake_store(tmp_path)
    mem_id = _add_inferred(store, "BOOS 上周去了一趟盐湖城出差。")
    _age(store, mem_id, 40)
    store.sweep_monthly()
    assert _meta_of(store, mem_id)["decay_state"] == DECAY_STATE_DORMANT
    assert _meta_of(store, mem_id).get("decay_delete_due") is not True
    dormant_events = [
        e for e in _ledger_events(store) if e["type"] == "dormant"
    ]
    assert len(dormant_events) == 1
    assert dormant_events[0]["delete_due"] is False
    _age(store, mem_id, 200)
    store.sweep_monthly()
    assert _meta_of(store, mem_id).get("decay_delete_due") is True
    dormant_events = [
        e for e in _ledger_events(store) if e["type"] == "dormant"
    ]
    assert len(dormant_events) == 2
    assert dormant_events[-1]["delete_due"] is True
    assert _meta_of(store, mem_id)["decay_state"] == DECAY_STATE_DORMANT
    assert "盐湖城" in _texts(store)[0]


def test_delete_due_needs_approval(tmp_path):
    """删的入口默认不开：没 BOOS 点头就删，抛 PermissionError；
    删之前先写备份，删完账本记一笔 delete。"""
    store = make_fake_store(tmp_path)
    mem_id = _add_inferred(store, "BOOS 上周去了一趟盐湖城出差。")
    _age(store, mem_id, 200)
    store.sweep_monthly()
    with pytest.raises(PermissionError):
        store.delete_due_memories()
    assert _texts(store) == ["BOOS 上周去了一趟盐湖城出差。"]


# ---------- 10. 关系 ----------


def test_classify_relation_four_types_and_none(tmp_path):
    """关系只留四种：同槽位新金额→取代；喜欢 vs 过敏→冲突；
    原文相同→补充；关键词重合≥2→关联；话题不挨着→None。"""
    assert (
        relations_mod.classify_relation(
            "我每月定投 5 块买万家纳指。",
            {"money_amount": 5, "money_fund": "万家纳指", "money_period": "daily"},
            "我每月定投 10 块买万家纳指。",
            {"money_amount": 10, "money_fund": "万家纳指", "money_period": "daily"},
        )
        == "supersede"
    )
    assert (
        relations_mod.classify_relation(
            "我喜欢吃鸡杂面。",
            {},
            "我对鸡杂面过敏。",
            {},
        )
        == "conflict"
    )
    assert (
        relations_mod.classify_relation(
            "我喜欢喝冰美式。",
            {},
            "我喜欢喝冰美式。",
            {},
        )
        == "supplement"
    )
    assert (
        relations_mod.classify_relation(
            "我喜欢喝冰美式。",
            {},
            "冰美式要加冰才好喝。",
            {},
        )
        == "related"
    )
    assert (
        relations_mod.classify_relation(
            "我喜欢喝冰美式。",
            {},
            "今天盐湖城下雪了。",
            {},
        )
        is None
    )


def test_relate_memories_round_trip(tmp_path):
    """relate 往返：记一条冲突关系，get_relations 双向都认得到；
    is_linked 是沉底前查三道里的第三道；四种之外的关系不让记。"""
    store = make_fake_store(tmp_path)
    first = _add(store, "我喜欢吃鸡杂面。")
    second = _add(store, "我对鸡杂面过敏。")
    event = store.relate_memories(first, second, "conflict", note="等 BOOS 定。")
    assert event is not None
    state = store.ledger_state()
    assert relations_mod.is_linked(state, first) is True
    assert relations_mod.is_linked(state, second) is True
    rels = relations_mod.get_relations(state, second)
    assert len(rels) == 1
    assert rels[0]["relation"] == "conflict"
    with pytest.raises(ValueError):
        store.relate_memories(first, second, "whatever")


# ---------- 11. 先写账本再写索引、提取补记 ----------


def test_add_memory_text_refuses_when_ledger_is_broken(tmp_path):
    """账本写不进去：这次不写索引、返回 {}——不能让索引里多一条
    账本里没有的记忆（对账时认不出来）。"""
    broken_dir = tmp_path / "ledger-dir"
    broken_dir.mkdir()
    store = make_fake_store(tmp_path)
    store.ledger_path = broken_dir
    assert store.add_memory_text("我喜欢喝冰美式。") == {}
    assert store.get_all() == []


def test_extraction_ledger_adds_are_not_double_appended(tmp_path):
    """提取补记不重复：同一批结果调两遍，第二遍按 key 去重，一条
    原话在账本里只留一笔 add 事件。"""
    store = make_fake_store(tmp_path)
    result = {"results": [{"memory": "我喜欢喝冰美式。", "event": "ADD"}]}
    metadata = {"created_at": "2026-11-13T10:00:00Z"}
    store._ledger_extraction_adds(result, metadata)
    store._ledger_extraction_adds(result, metadata)
    adds = [e for e in _ledger_events(store) if e["type"] == "add"]
    assert len(adds) == 1
    assert adds[0]["text"] == "我喜欢喝冰美式。"


def test_direct_add_is_ledger_first(tmp_path):
    """直接存（add_memory_text）先写账本再写索引：账本里有这条的
    add 事件，key 和索引里的原话 + created_at 对得上。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢喝冰美式。")
    adds = [e for e in _ledger_events(store) if e["type"] == "add"]
    assert len(adds) == 1
    created_at = _meta_of(store, mem_id)["created_at"]
    assert adds[0]["key"] == ledger_mod.ledger_event_key(
        "我喜欢喝冰美式。", created_at
    )


def test_identity_seed_is_recorded_in_ledger(tmp_path):
    """身份种子也进账本：ensure_identity_memories 种下的名字、SOUL、
    电脑，账本里都有 add 事件——账本是完整的档案，不能漏了种子。"""
    store = make_fake_store(tmp_path)
    store.ensure_identity_memories()
    adds = [e for e in _ledger_events(store) if e["type"] == "add"]
    assert len(adds) >= 3
    assert any("十二" in e["text"] for e in adds)
