# tests/test_group3_fixes_20261124.py（第三组修复，2026-11-24）
# 全部离线（向量打桩，不加载 bge 模型、不调真 API、不花钱）。
#
# 3a 检索强化没传 + 时间只写 SQLite（GPT #2）：SQLiteStore.search
#    接了 reinforce/turn_id 却没往下传，强化从没发生；note_context
#    刷 last_used_at 只写 SQLite，账本重建退回旧值（P1-6 已知问题）。
#    修法：search(reinforce=True) 真用上的结果走 note_context 强化；
#    note_context 每轮记一笔账本 TOUCH 事件，对账重建时按 TOUCH
#    把 last_used_at 带回来。
# 3b 衰减没人跑 + SQLite 没有自检（GPT #3）：sweep_decay 只有测试
#    在调，SQLiteStore 没有 run_self_check。修法：加 run_self_check
#    （先 sweep_decay 再跑十项自检），和 Mem0Store 同名同节奏，由
#    每月定时任务调，不在服务启动时自动跑。
# 3c 网页和命令行各用各的库（GPT #4）：web 看环境变量、main 写死
#    mem0。修法：memory/store_factory 统一入口，两边都走它。
# 3d 标“待删”不写账本：sweep_decay 的 delete_due 只写 SQLite。
#    修法：补一笔 DORMANT 事件带 delete_due=True，重建不丢。

import json
from datetime import datetime, timedelta, timezone

import pytest

from memory import ledger as _ledger
from memory import store_factory
from memory import decay as _decay
from memory.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _meta_of(store, text):
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE text=?", (text,)
    ).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def _set_meta(store, text, **fields):
    meta = _meta_of(store, text)
    meta.update(fields)
    store._conn.execute(
        "UPDATE memories SET metadata_json=? WHERE text=?",
        (json.dumps(meta, ensure_ascii=False), text),
    )
    store._conn.commit()


def _drop_row(store, memory_id):
    store._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    store._conn.commit()


# ---------- 3a：search reinforce + TOUCH 记账 ----------


def test_search_reinforce_refreshes_last_used(store):
    saved = store.add_memory_text("我喜欢吃火锅", source="user")
    _set_meta(store, "我喜欢吃火锅", last_used_at="2020-01-01T00:00:00+00:00")
    results = store.search("火锅", reinforce=True)
    assert results and results[0]["id"] == saved["id"]
    assert _meta_of(store, "我喜欢吃火锅")["last_used_at"] > "2020-01-01"


def test_search_without_reinforce_does_not_refresh(store):
    store.add_memory_text("我喜欢吃火锅", source="user")
    _set_meta(store, "我喜欢吃火锅", last_used_at="2020-01-01T00:00:00+00:00")
    results = store.search("火锅")
    assert results
    assert _meta_of(store, "我喜欢吃火锅")["last_used_at"].startswith("2020-01-01")


def test_note_context_touch_survives_rebuild(store):
    saved = store.add_memory_text("我喜欢吃火锅", source="user")
    mid = saved["id"]
    store.note_context([{"id": mid}])
    touched = _meta_of(store, "我喜欢吃火锅")["last_used_at"]
    # 账本里有一笔 TOUCH 事件。
    events, _info = _ledger.read_events(store.ledger_path)
    touches = [e for e in events if e.get("type") == "touch"]
    assert touches and touches[-1].get("memory_ids") == [mid]
    # 删行 + 对账重建：last_used_at 按 TOUCH 带回来，不退回 add 时的值。
    _drop_row(store, mid)
    assert store.reconcile_ledger_index() == 1
    rebuilt = _meta_of(store, "我喜欢吃火锅")
    assert rebuilt["last_used_at"] == touched


# ---------- 3b/3d：sweep 待删记账本 + run_self_check ----------


def _aged_store(store):
    """一条 episodic 记忆，last_used_at 拨到 200 天前（该标待删了）。"""
    saved = store.add_memory_text("我去爬了玉龙雪山", source="user")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    # 注意：source=user 的已确认记忆算核心（_is_core_memory），核心不
    # 衰减——这里要测的是普通 episodic 记忆，改成未确认的推断来源。
    _set_meta(
        store, "我去爬了玉龙雪山",
        decay_kind="episodic", created_at=old, last_used_at=old,
        decay_state="verbatim", source="inferred", user_confirmed=False,
        slot=None,
    )
    return saved


def test_sweep_delete_due_writes_ledger_and_survives_rebuild(store):
    saved = _aged_store(store)
    mid = saved["id"]
    out = store.sweep_decay()
    assert out["delete_due"] == 1
    assert _meta_of(store, "我去爬了玉龙雪山")["decay_state"] == _decay.DECAY_STATE_DELETE_DUE
    # 账本里有 DORMANT 事件带 delete_due=True。
    events, _info = _ledger.read_events(store.ledger_path)
    marks = [
        e for e in events
        if e.get("type") == "dormant" and e.get("memory_id") == mid
    ]
    assert marks and marks[-1].get("delete_due") is True
    # 幂等：再扫一遍不动账本、不重复标。
    lines_before = len(events)
    out2 = store.sweep_decay()
    assert out2 == {"dormant": 0, "delete_due": 0}
    events2, _info2 = _ledger.read_events(store.ledger_path)
    assert len(events2) == lines_before
    # 删行 + 对账重建：待删状态不丢。
    _drop_row(store, mid)
    assert store.reconcile_ledger_index() == 1
    assert _meta_of(store, "我去爬了玉龙雪山")["decay_state"] == _decay.DECAY_STATE_DELETE_DUE


def test_run_self_check_runs_and_sweeps(store, monkeypatch):
    _aged_store(store)
    # R-6（2026-11-24）后 run_self_check 接了整理路由的判官；本测测的
    # 是没判官时的口径（第 9/10 项 skipped），把路由打桩成不可用
    # （和 test_self_check_store.py 同款：笔记本有真 Key，不拦会真调）。
    monkeypatch.setattr(store, "organizing_router", lambda: None)
    report = store.run_self_check()
    assert set(report["items"].keys()) == set(range(1, 11))
    # 先扫衰减：那条老记忆这次被标待删。
    assert report["decay_sweep"]["delete_due"] == 1
    assert _meta_of(store, "我去爬了玉龙雪山")["decay_state"] == _decay.DECAY_STATE_DELETE_DUE
    # 没接模型路由时第 9/10 项 skipped，不猜。
    assert report["items"][9]["status"] == "skipped"
    assert report["items"][10]["status"] == "skipped"


def test_run_self_check_empty_store(store):
    report = store.run_self_check()
    assert set(report["items"].keys()) == set(range(1, 11))
    assert report["decay_sweep"] == {"dormant": 0, "delete_due": 0}


# ---------- 3c：store 工厂 ----------




def test_factory_sqlite_backend(monkeypatch):
    monkeypatch.setenv("SHIER_MEMORY_BACKEND", "sqlite")
    monkeypatch.setattr(
        "memory.sqlite_store.create_sqlite_store", lambda bt=None: "SQLITE"
    )
    assert store_factory.get_store() == "SQLITE"
    assert store_factory.create_store() == "SQLITE"


def test_main_remember_goes_through_factory(monkeypatch, capsys):
    """main.handle_remember 不再写死 mem0：工厂给什么库就用什么库。"""
    import main as _main

    class _FakeStore:
        def __init__(self):
            self.added = []

        def add_memory_text(self, content):
            self.added.append(content)
            return {"id": "x", "memory": content, "event": "ADD"}

    fake = _FakeStore()
    monkeypatch.setattr(store_factory, "get_store", lambda bt=None: fake)
    _main.handle_remember(None, "记住：我喜欢喝咖啡")
    assert fake.added == ["我喜欢喝咖啡"]
    assert "已记住" in capsys.readouterr().out
