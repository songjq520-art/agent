# tests/test_rfix_20261124.py（P1 重查清单修复，2026-11-24）
# 全部离线（向量打桩，不加载 bge 模型、不调真 API、不花钱）。
#
# R-2 问过没人答的问题永不再问：PendingQuestions.drain_for_chat 只取
#    pending、save_turn 只有本轮有新问题时才带旧问题。修法：asked 超过
#    7 天未答且没问满 3 次的，drain 前回滚成 pending 可重问（记
#    asked_at/ask_count）；save_turn 每轮都带攒着的旧问题（仍受一轮
#    3 条上限管）。
# R-3 _dedupe_key 无条件剥结尾括号：改成与逗号版同一规则——只有括号
#    里是“……是X”、且 X 已在括号前出现过（改写重述）才截，内容性
#    括号（买纳指/买标普）保留。
# R-4 衰减/晋升/复活先改 SQLite、账本失败只警告：三处统一账本先行
#    ——账本写失败就不改 SQLite、不计数/不晋升/不复活，下次再试。
# R-6 SQLite 版 run_self_check 写死 model_judge=None：按 mem0 同款接
#    整理路由（organizing_router + make_model_judge），第 9/10 项真跑；
#    路由不可用时如实 skipped。
# R-7 _set_memory_slot 只写 SQLite 不记账本：补一笔 SLOT 账本事件，
#    对账重建时按它恢复槽位标；记账失败先不补标。
# R-9 纠正候选同槽兜底只在候选为空时触发：多候选相似度筛全军覆没
#    （0 条过线）时补跑一次同槽兜底，恰好一条过线才按单候选继续，
#    护栏（0.70 安全网）不放宽。
# R-10 回答问题选「不存」先消耗问题再删记忆：改成先删成功再消耗
#    问题（与 P1-5「进核心」同款顺序），删不掉如实报 ok=False。

import json
from datetime import datetime, timedelta, timezone

import pytest

from memory import ledger as _ledger
from memory import pending_questions as _pq
from memory.sqlite_store import SQLiteStore, _dedupe_key


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


def _fail_ledger_on(monkeypatch, fail_types):
    """把账本 append 打桩：fail_types 里的事件类型抛错，其余照常写。
    返回 state dict，把 state["fail"] 置 False 即恢复正常写。"""
    orig = _ledger.append_event
    state = {"fail": True}

    def fake(path, event, lock=None, seq=None):
        if state["fail"] and (event or {}).get("type") in fail_types:
            raise RuntimeError("账本写失败（测试打桩）")
        return orig(path, event, lock=lock, seq=seq)

    monkeypatch.setattr(_ledger, "append_event", fake)
    return state


# ---------- R-2：问题重问（间隔 + 上限）+ save_turn 每轮带旧问题 ----------


def _make_pq(tmp_path):
    return _pq.PendingQuestions(path=str(tmp_path / "pq.json"))


def _backdate_asked(pq, qid, days):
    for item in pq._questions:
        if item["id"] == qid:
            item["asked_at"] = (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).isoformat().replace("+00:00", "Z")
    pq._save()


def test_asked_question_reasked_after_interval(tmp_path):
    pq = _make_pq(tmp_path)
    q = pq.add("importance", "这条重要吗？", ["进核心", "不存"], "拿不准")
    first = pq.drain_for_chat()
    assert [x["id"] for x in first] == [q["id"]]
    assert pq.get(q["id"])["ask_count"] == 1
    # 刚问过：不再带出来。
    assert pq.drain_for_chat() == []
    # 8 天没人答：回滚重问（第 2 次）。
    _backdate_asked(pq, q["id"], 8)
    second = pq.drain_for_chat()
    assert [x["id"] for x in second] == [q["id"]]
    assert pq.get(q["id"])["ask_count"] == 2


def test_asked_question_reask_cap(tmp_path):
    pq = _make_pq(tmp_path)
    q = pq.add("importance", "这条重要吗？", ["进核心", "不存"], "拿不准")
    for _round in range(_pq.QUESTION_MAX_ASKS):
        assert len(pq.drain_for_chat()) == 1
        _backdate_asked(pq, q["id"], _pq.REASK_AFTER_DAYS + 1)
    # 问满 3 次：再隔多久也不往聊天里带了（待答列表里还能答）。
    assert pq.drain_for_chat() == []
    assert pq.get(q["id"])["status"] == "asked"
    assert pq.count_unanswered() == 1


def test_save_turn_brings_backlog_without_new_questions(store):
    saved = store.add_memory_text("我朋友下周可能来上海出差", source="user")
    q = store.pending_questions.add(
        _pq.QUESTION_KIND_IMPORTANCE,
        "我朋友下周可能来上海出差",
        _pq.IMPORTANCE_QUESTION_OPTIONS,
        "拿不准",
        memory_id=saved["id"],
    )
    assert q is not None
    # 本轮只闲聊（提问句不存记忆、不产生新问题）：以前旧问题带不出来，
    # 现在回执里要有它、且标成已问。
    receipt = store.save_turn(
        [{"role": "user", "content": "我喜欢吃什么？"},
         {"role": "assistant", "content": "你喜欢吃面条。"}],
        turn_id="t-r2-backlog",
    )
    asked_ids = [x["id"] for x in (receipt.get("questions") or [])]
    assert q["id"] in asked_ids
    assert store.pending_questions.get(q["id"])["status"] == "asked"


# ---------- R-3：去重键括号规则 ----------


def test_dedupe_key_parenthetical_content_kept():
    assert _dedupe_key("每天定投5块（买纳指）") != _dedupe_key("每天定投5块（买标普）")
    assert _dedupe_key("每天定投5块(买纳指)") != _dedupe_key("每天定投5块(买标普)")


def test_dedupe_key_parenthetical_restatement_still_stripped():
    assert _dedupe_key("我改吃面条了（主食偏好是面条）") == _dedupe_key("我改吃面条了")
    assert _dedupe_key("我改吃面条了(主食偏好是面条)") == _dedupe_key("我改吃面条了")


def test_parenthetical_facts_both_stored(store):
    store.add_memory_text("每天定投5块（买纳指）", source="user")
    store.add_memory_text("每天定投5块（买标普）", source="user")
    rows = store._conn.execute(
        "SELECT text FROM memories WHERE voided=0 AND deleted=0"
        " AND superseded=0 AND text LIKE '每天定投5块%'"
    ).fetchall()
    assert {r[0] for r in rows} == {"每天定投5块（买纳指）", "每天定投5块（买标普）"}


# ---------- R-4：账本先行（晋升 / 衰减 / 复活） ----------


def test_promote_ledger_failure_keeps_episodic(store, monkeypatch):
    saved = store.add_memory_text("我喜欢吃火锅", source="user")
    mid = saved["id"]
    # 用户亲口说的事实存入时 decay_kind 就是 stable（设计如此）；这里
    # 测的是一次性经历靠多次出现晋升的路，先把它拨回 episodic。
    _set_meta(store, "我喜欢吃火锅", decay_kind="episodic")
    assert _meta_of(store, "我喜欢吃火锅")["decay_kind"] == "episodic"
    state = _fail_ledger_on(monkeypatch, {_ledger.LEDGER_EVENT_PROMOTE})

    def _occur(day, sess):
        row = store._conn.execute(
            "SELECT metadata_json FROM memories WHERE id=?", (mid,)
        ).fetchone()
        meta = json.loads(row[0] or "{}")
        return store._record_occurrence(mid, meta, day, sess, "我喜欢吃火锅")

    # 凑满 3 天 3 次：每次都够格晋升，但账本写不进去就不许晋升。
    for i, (day, sess) in enumerate(
        [("2026-11-21", "s1"), ("2026-11-22", "s2"),
         ("2026-11-23", "s3"), ("2026-11-24", "s4")]
    ):
        assert _occur(day, sess) is False
    assert _meta_of(store, "我喜欢吃火锅")["decay_kind"] != "stable"
    # 账本恢复：第 5 次出现，晋升成功。
    state["fail"] = False
    assert _occur("2026-11-25", "s5") is True
    assert _meta_of(store, "我喜欢吃火锅")["decay_kind"] == "stable"


def _aged_memory(store):
    """一条普通 episodic 记忆，last_used_at 拨到 200 天前（该标待删）。"""
    saved = store.add_memory_text("我去爬了玉龙雪山", source="user")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    _set_meta(
        store, "我去爬了玉龙雪山",
        decay_kind="episodic", created_at=old, last_used_at=old,
        decay_state="verbatim", source="inferred", user_confirmed=False,
        slot=None,
    )
    return saved


def test_sweep_ledger_failure_keeps_state(store, monkeypatch):
    _aged_memory(store)
    state = _fail_ledger_on(monkeypatch, {_ledger.LEDGER_EVENT_DORMANT})
    out = store.sweep_decay()
    assert out == {"dormant": 0, "delete_due": 0}
    assert _meta_of(store, "我去爬了玉龙雪山")["decay_state"] == "verbatim"
    state["fail"] = False
    out = store.sweep_decay()
    assert out["delete_due"] == 1
    assert _meta_of(store, "我去爬了玉龙雪山")["decay_state"] == "decay_delete_due"


def test_note_context_revive_ledger_failure(store, monkeypatch):
    saved = store.add_memory_text("我爱吃火锅", source="user")
    mid = saved["id"]
    _set_meta(store, "我爱吃火锅", decay_state="dormant")
    state = _fail_ledger_on(monkeypatch, {_ledger.LEDGER_EVENT_REVIVE})
    store.note_context([{"id": mid}])
    # REVIVE 写不进账本：SQLite 也不许先复活。
    assert _meta_of(store, "我爱吃火锅")["decay_state"] == "dormant"
    state["fail"] = False
    store.note_context([{"id": mid}])
    assert _meta_of(store, "我爱吃火锅")["decay_state"] == "verbatim"


# ---------- R-6：自检接整理路由判官 ----------


class _FakeRouter:
    def judge(self, system_prompt, user_prompt, max_tokens=500,
              purpose=None, **kwargs):
        return {
            "text": '{"accurate": true, "evidence": "跟出处对得上"}',
            "cost_usd": 0.0,
            "model": "fake",
        }

    def status(self):
        return {"over_budget": False, "spent_usd": 0.0, "cap_usd": 3.0}


def test_self_check_sqlite_uses_router_judge(store, monkeypatch):
    store.add_memory_text("我喜欢喝咖啡", source="user")
    core = store.add_memory_text("我每天定投5块买纳指", source="user")
    _set_meta(store, "我每天定投5块买纳指", decay_kind="stable",
              user_confirmed=True)
    monkeypatch.setattr(store, "organizing_router", lambda: _FakeRouter())
    report = store.run_self_check()
    # 第 9 项真抽查、第 10 项真查核心：都不再 skipped。
    assert report["items"][9]["status"] == "pass"
    assert report["items"][10]["status"] == "pass"
    assert report["organizing_budget"]["spent_usd"] == 0.0


def test_self_check_sqlite_no_router_still_skips(store, monkeypatch):
    store.add_memory_text("我喜欢喝咖啡", source="user")
    monkeypatch.setattr(store, "organizing_router", lambda: None)
    report = store.run_self_check()
    assert report["items"][9]["status"] == "skipped"
    assert report["items"][10]["status"] == "skipped"


# ---------- R-7：槽位标记账本 ----------


def test_set_memory_slot_writes_ledger_and_survives_rebuild(store):
    saved = store.add_memory_text("我喜欢吃火锅", source="user")
    mid = saved["id"]
    assert not _meta_of(store, "我喜欢吃火锅").get("slot")
    store._set_memory_slot(mid, "residence")
    assert _meta_of(store, "我喜欢吃火锅")["slot"] == "residence"
    events, _info = _ledger.read_events(store.ledger_path)
    slots = [e for e in events if e.get("type") == "slot"]
    assert slots and slots[-1].get("memory_id") == mid
    assert slots[-1].get("slot") == "residence"
    # 删行 + 对账重建：槽位标按 SLOT 事件带回来（列和 meta 都有）。
    store._conn.execute("DELETE FROM memories WHERE id=?", (mid,))
    store._conn.commit()
    assert store.reconcile_ledger_index() == 1
    row = store._conn.execute(
        "SELECT slot, metadata_json FROM memories WHERE id=?", (mid,)
    ).fetchone()
    assert row[0] == "residence"
    assert json.loads(row[1] or "{}")["slot"] == "residence"


def test_set_memory_slot_ledger_failure_skips(store, monkeypatch):
    saved = store.add_memory_text("我喜欢吃火锅", source="user")
    mid = saved["id"]
    _fail_ledger_on(monkeypatch, {_ledger.LEDGER_EVENT_SLOT})
    store._set_memory_slot(mid, "residence")  # 不抛异常
    assert not _meta_of(store, "我喜欢吃火锅").get("slot")


# ---------- R-9：纠正同槽兜底补跑 ----------


def _rig_bad_recall(store, monkeypatch, exclude_ids, sims):
    """把语义召回打桩成只返回不相关命中（exclude_ids 里的真目标
    不在候选里），相似度按 sims 表给（id -> 分数，其余 0.30）。"""
    junk = [
        row[0]
        for row in store._conn.execute("SELECT id FROM memories").fetchall()
        if row[0] not in exclude_ids
    ]
    monkeypatch.setattr(
        store, "_search_inner",
        lambda query, top_k=10: [{"id": j} for j in junk],
    )
    monkeypatch.setattr(
        store, "_candidate_similarity",
        lambda mid, new_fact: sims.get(mid, 0.30),
    )
    return junk


def test_correction_rescued_by_slot_fallback(store, monkeypatch):
    target = store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text("我开货车", source="user")
    store.add_memory_text("今天去公园散步", source="user")
    _rig_bad_recall(store, monkeypatch, {target["id"]}, {target["id"]: 0.85})
    result = store.correct_memory("我喜欢吃米饭", "我喜欢吃面条")
    assert result["status"] == "updated"
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE id=?", (target["id"],)
    ).fetchone()
    assert row[0] == 1
    live = store._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE text='我喜欢吃面条'"
        " AND superseded=0 AND deleted=0 AND voided=0"
    ).fetchone()
    assert live[0] == 1


def test_correction_rescue_ambiguous_stays_not_found(store, monkeypatch):
    t1 = store.add_memory_text("我喜欢吃米饭", source="user")
    t2 = store.add_memory_text("我喜欢吃米线", source="user")
    store.add_memory_text("我开货车", source="user")
    store.add_memory_text("今天去公园散步", source="user")
    # 2026-11-24 适配：本测管的是同槽兜底的含糊判定；第 5 条跨槽
    # 安全网（Round A 后加）不在本测范围——它会真调在线判官，
    # 裁决有随机性、还会让本测违背文件头“全部离线”的前提。
    # 安全网有它自己的专项测试，这里把它打桩关掉。
    monkeypatch.setattr(store, "_cross_slot_net_target", lambda *a, **k: None)
    _rig_bad_recall(
        store, monkeypatch, {t1["id"], t2["id"]},
        {t1["id"]: 0.85, t2["id"]: 0.85},
    )
    result = store.correct_memory("我喜欢吃米饭", "我喜欢吃面条")
    # 兜底捞出两条都过线：不猜，维持 not_found，两条都不动。
    assert result["status"] == "not_found"
    for t in (t1, t2):
        row = store._conn.execute(
            "SELECT superseded FROM memories WHERE id=?", (t["id"],)
        ).fetchone()
        assert row[0] == 0


# ---------- R-10：回答「不存」先删后消耗 ----------


def _importance_question(store, text="我朋友下周可能来上海出差"):
    saved = store.add_memory_text(text, source="user")
    q = store.pending_questions.add(
        _pq.QUESTION_KIND_IMPORTANCE, text,
        _pq.IMPORTANCE_QUESTION_OPTIONS, "拿不准",
        memory_id=saved["id"],
    )
    return saved, q


def test_answer_drop_delete_failure_keeps_question(store, monkeypatch):
    saved, q = _importance_question(store)
    orig_delete = store.delete_memory
    monkeypatch.setattr(
        store, "delete_memory",
        lambda mid, approve=False: {"deleted": False, "reason": "ledger_failed"},
    )
    result = store.answer_question(q["id"], "不存")
    assert result["ok"] is False
    assert result["deleted"] is False
    # 问题没被消耗，还能再答；记忆也还在。
    assert store.pending_questions.get(q["id"])["status"] != "answered"
    monkeypatch.setattr(store, "delete_memory", orig_delete)
    result2 = store.answer_question(q["id"], "不存")
    assert result2["ok"] is True and result2["deleted"] is True
    assert store.pending_questions.get(q["id"])["status"] == "answered"
