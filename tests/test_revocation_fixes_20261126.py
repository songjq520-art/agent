# tests/test_revocation_fixes_20261126.py（H3 补修轮：贵妃第二
# 十八份复查 F1–F4，2026-11-26）
#
# 覆盖口径（裁决存档 TWENTY_EIGHTH_REVIEW_adjudication.md）：
#   F1 撤销崩溃恢复成对性：补偿已落账、原意图 revoked_by 与
#      撤销回执登记没落盘的窗口——重启后成对恢复（回填标记、
#      回执按意图重建），同键重入回 duplicate 不二次补偿，
#      合法新变更把字段写回 after 后换键再撤仍 already_revoked；
#      未定案（pending＋账本存疑）的先行撤销意图在场时同键
#      重入只许 unresolved_revocation，不许穿到执行段。
#   F2 账本完整性消费：derive 在坏行/截断时拒
#      before_state_unavailable（不拿残缺折叠折 before）；
#      撤销核对在账本存疑时标 ledger_doubtful 判不干净；
#      座位重算标 ledger_doubtful 不报 reconciled；账本压缩
#      遇坏行拒绝。
#   F3 撤销核对补齐：独立 user_confirmed 列与 meta 分叉
#      （columns_match_meta）、成员 ID 集合与账本不一致
#      （members_match_ledger）都判不干净；demote 补偿第二
#      段列写被触发器打断时顶层不许报干净成功。
#   F4 提交未定案：op 核验存疑（committed_unverified）＋核
#      对通过时，顶层 ok=False、reason=committed_unverified，
#      意图 final_* 同口径落盘，重启后回执与冻结都不洗白。
#
# 全程离线：嵌入/判官打桩为 None，同 test_revocation 口径。
# 故障只注入在边界上（落盘文件回滚模拟猝死窗口、账本行写
# 坏/写一半、数据库触发器打断第二段写、账本读单次存疑）。

import json
import uuid

import pytest

from memory import ledger as _ledger
from memory.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.organizing_router = lambda: None
    return s


def _add_user(store, text, **kwargs):
    saved = store.add_memory_text(text, source="user", **kwargs)
    assert saved.get("event") == "ADD", saved
    return saved["id"]


def _row(store, mid):
    r = store._conn.execute(
        "SELECT text, metadata_json, user_confirmed FROM memories"
        " WHERE id=?",
        (mid,),
    ).fetchone()
    assert r is not None, mid
    return {
        "text": r[0],
        "meta": json.loads(r[1] or "{}"),
        "user_confirmed": bool(r[2]),
    }


def _session(store, direct=True):
    sess = store.issue_session(
        speaker="owner", issuance_source="owner", owner_direct=direct
    )
    assert sess is not None
    return sess


def _execute(store, action, mid, key=None, **params):
    store.controlled_entry_execution_enabled = True
    prop = {
        "action": action,
        "memory_id": mid,
        "reason": "测试执行",
        "instruction_source": "owner",
        "idempotency_key": key or uuid.uuid4().hex,
        "expected_version": store.controlled_entry_version(mid),
        "session": _session(store),
    }
    if params:
        prop["params"] = params
    receipt = store.submit_controlled_proposal(prop)
    assert receipt["status"] == "executed", receipt
    return receipt


def _revoke_req(op_id=None, target_key=None, source="owner",
                rev_key=None, session=None):
    req = {
        "reason": "测试撤销",
        "instruction_source": source,
        "idempotency_key": rev_key or uuid.uuid4().hex,
    }
    if op_id is not None:
        req["target_op_id"] = op_id
    if target_key is not None:
        req["target_idempotency_key"] = target_key
    if session is not None:
        req["session"] = session
    return req


def _ledger_events(store):
    events, _info = _ledger.read_events(store.ledger_path)
    return events


def _comp_events(store, rev_op_id):
    return [
        e for e in _ledger_events(store)
        if e.get("op_id") == rev_op_id
    ]


# ---------- F1：崩溃窗口成对恢复 ----------

def test_f1_crash_window_pairing_recovered(store, tmp_path):
    mid = _add_user(store, "BOOS 每天定投五块钱买纳指")
    receipt = _execute(store, "demote", mid)
    op_id = receipt["detail"]["op_id"]
    req = _revoke_req(op_id=op_id, rev_key="rev-f1-1",
                      session=_session(store))
    rev = store.revoke_controlled_change(req)
    assert rev["status"] == "revoked" and rev["ok"] is True, rev
    rev_op_id = rev["revocation_op_id"]

    # 模拟猝死窗口：补偿已落账，但落盘状态停在「撤销意图
    # 还是 pending、原意图没标 revoked_by、撤销回执登记没
    # 有」——把三个落盘文件回滚到那一刻的样子。
    intents_path = store._controlled_intents_path
    data = json.loads(intents_path.read_text(encoding="utf-8"))
    orig = data["intents"][op_id]
    orig.pop("revoked_by", None)
    orig.pop("revoked_at", None)
    rev_intent = data["intents"][rev_op_id]
    rev_intent["status"] = "pending"
    for k in ("final_ok", "final_reason", "invariant_checks"):
        rev_intent.pop(k, None)
    intents_path.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    receipts_path = store._controlled_receipts_path
    rdata = json.loads(receipts_path.read_text(encoding="utf-8"))
    rdata.get("revocations", {}).pop("rev-f1-1", None)
    rdata.get("revocation_fingerprints", {}).pop("rev-f1-1", None)
    receipts_path.write_text(
        json.dumps(rdata, ensure_ascii=False), encoding="utf-8"
    )

    # 重启：对账补提交状态之外，成对关系必须一起恢复。
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    orig2 = store2._controlled_intents[op_id]
    assert orig2.get("revoked_by") == rev_op_id
    # 回执登记丢了，按撤销意图重建：补偿落账了但核对没跑
    # 完（final_* 没落盘）——按未核实交回，不许冒充干净。
    got = store2.controlled_revocation_receipt("rev-f1-1")
    assert got is not None
    assert got["ok"] is False
    assert got["reason"] == "committed_unverified"
    assert got["detail"]["unverified_reconstruction"] is True

    # 同键重入：duplicate，不二次补偿。
    store2.controlled_entry_execution_enabled = True
    dup = store2.revoke_controlled_change(
        _revoke_req(op_id=op_id, rev_key="rev-f1-1",
                    session=_session(store2))
    )
    assert dup["status"] == "duplicate" and dup["duplicate"] is True
    assert len(_comp_events(store2, rev_op_id)) == 1
    assert _row(store2, mid)["user_confirmed"] is True

    # 合法新 demote 把字段写回原 after 值后，换个键撤原
    # op：already_revoked，不许把新变更静默踩掉。
    receipt2 = _execute(store2, "demote", mid)
    assert _row(store2, mid)["user_confirmed"] is False
    rev2 = store2.revoke_controlled_change(
        _revoke_req(op_id=op_id, session=_session(store2))
    )
    assert rev2["status"] == "refused", rev2
    assert rev2["reason"] == "already_revoked"
    assert _row(store2, mid)["user_confirmed"] is False
    assert len(_comp_events(store2, rev_op_id)) == 1
    assert receipt2["detail"]["op_id"] != op_id


def test_f1_pending_revocation_blocks_reentry(store):
    mid = _add_user(store, "BOOS 有一台安卓手机")
    receipt = _execute(store, "demote", mid)
    op_id = receipt["detail"]["op_id"]
    # 先行撤销意图只落了 pending（补偿没落账进程就死了），
    # 且账本被写坏一行让对账定不了生死。
    fp = store._controlled_revocation_fingerprint(
        op_id, None, "owner"
    )
    store._controlled_intents["op-rev-pending-1"] = {
        "op_id": "op-rev-pending-1",
        "kind": "revocation",
        "revokes_op_id": op_id,
        "idempotency_key": "rev-f1-pending",
        "fingerprint": fp,
        "action": "demote",
        "memory_id": mid,
        "expected_version": None,
        "status": "pending",
        "ts": "2026-11-26T00:00:00+00:00",
        "ledger_seq": None,
        "version_after": None,
    }
    store._controlled_save_intents()
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("this-line-is-not-json\n")

    rev = store.revoke_controlled_change(
        _revoke_req(op_id=op_id, rev_key="rev-f1-pending",
                    session=_session(store))
    )
    assert rev["status"] == "refused", rev
    assert rev["reason"] == "unresolved_revocation"
    # 没穿到执行段：没有补偿事件，字段没动。
    assert _comp_events(store, "op-rev-pending-1") == []
    assert _row(store, mid)["user_confirmed"] is False


# ---------- F2：账本完整性消费 ----------

def _old_intent_demote(store, text="BOOS 喜欢喝茶"):
    """造一笔「老意图」demote：add(uc=False) → PROMOTE
    (uc=True) → 受控 demote，意图里的 before/after 剥掉，
    只剩账本可重建。返回 (mid, op_id)。"""
    saved = store.add_memory_text(text, source="assistant")
    assert saved.get("id"), saved
    mid = saved["id"]
    row = _row(store, mid)
    assert row["user_confirmed"] is False
    meta_true = dict(row["meta"])
    meta_true["user_confirmed"] = True
    # 确认口径（provenance._is_core_memory）：confirmed 只认
    # source=user 且 user_confirmed=True——真实确认路走完后
    # 行的 meta 就是这个形状，这里按同形状造已确认行。
    meta_true["source"] = "user"
    _ledger.append_event(
        store.ledger_path,
        {
            "type": _ledger.LEDGER_EVENT_PROMOTE,
            "memory_id": mid,
            "key": mid,
            "text": row["text"],
            "meta": meta_true,
        },
        lock=store._ledger_lock,
    )
    store._conn.execute(
        "UPDATE memories SET metadata_json=?, user_confirmed=1"
        " WHERE id=?",
        (json.dumps(meta_true, ensure_ascii=False), mid),
    )
    store._conn.commit()
    receipt = _execute(store, "demote", mid)
    op_id = receipt["detail"]["op_id"]
    intent = store._controlled_intents[op_id]
    intent.pop("before_fields", None)
    intent.pop("after_fields", None)
    store._controlled_save_intents()
    return mid, op_id


def test_f2_derive_refuses_on_bad_lines(store, tmp_path):
    mid, op_id = _old_intent_demote(store)
    # 把中间那笔 PROMOTE 写坏：残余折叠会折出错误 before
    # (uc=False)——不许拿它去补偿。
    lines = store.ledger_path.read_text(
        encoding="utf-8"
    ).splitlines()
    out = []
    corrupted = False
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            ev = None
        if (
            ev is not None
            and ev.get("type") == _ledger.LEDGER_EVENT_PROMOTE
            and not ev.get("op_id")
        ):
            out.append("CORRUPTED-LINE")
            corrupted = True
        else:
            out.append(line)
    assert corrupted
    store.ledger_path.write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )

    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    intent2 = store2._controlled_intents[op_id]
    assert store2._controlled_derive_fields(intent2) is None
    store2.controlled_entry_execution_enabled = True
    rev = store2.revoke_controlled_change(
        _revoke_req(op_id=op_id, session=_session(store2))
    )
    assert rev["status"] == "refused", rev
    assert rev["reason"] == "before_state_unavailable"
    assert _row(store2, mid)["user_confirmed"] is False


def test_f2_derive_refuses_on_truncated(store, tmp_path):
    mid, op_id = _old_intent_demote(store, "BOOS 夜班作息")
    # 账本末尾写一半的残行：truncated 同样判未定案。
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "promote", "seq": 99, "mem')
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    intent2 = store2._controlled_intents[op_id]
    assert store2._controlled_derive_fields(intent2) is None
    store2.controlled_entry_execution_enabled = True
    rev = store2.revoke_controlled_change(
        _revoke_req(op_id=op_id, session=_session(store2))
    )
    assert rev["status"] == "refused", rev
    assert rev["reason"] == "before_state_unavailable"


def test_f2_verify_marks_ledger_doubtful(store, monkeypatch):
    mid = _add_user(store, "BOOS 的邮箱是谷歌的")
    receipt = _execute(store, "demote", mid)
    pre_meta = _row(store, mid)["meta"]
    # 宿主路把字段恢复到位（真账本 PROMOTE＋投影），模拟
    # 补偿已落账后的行状态。
    store._update_core_fields(mid, {"user_confirmed": True})
    store._conn.execute(
        "UPDATE memories SET user_confirmed=1 WHERE id=?", (mid,)
    )
    store._conn.commit()
    row = _row(store, mid)
    clean = store._controlled_verify_revocation(
        mid, "user_confirmed", True, row["text"], pre_meta,
        "demote",
    )
    assert clean["ok"] is True, clean

    real_read = _ledger.read_events

    def doubtful_read(path):
        events, info = real_read(path)
        info = dict(info)
        info["bad_lines"] = (info.get("bad_lines") or 0) + 1
        return events, info

    monkeypatch.setattr(_ledger, "read_events", doubtful_read)
    doubted = store._controlled_verify_revocation(
        mid, "user_confirmed", True, row["text"], pre_meta,
        "demote",
    )
    assert doubted["ok"] is False
    assert doubted["checks"]["ledger_doubtful"] is True
    assert doubted["checks"]["ledger_matches"] is False
    assert receipt["detail"]["op_id"]


def test_f2_recount_marks_doubtful(store):
    _add_user(store, "BOOS 的姐姐在西安")
    snap0 = store.recount_core_seats_from_ledger()
    assert snap0.get("ledger_doubtful") is not True
    assert snap0["reconciled"] is True
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("broken-line\n")
    snap = store.recount_core_seats_from_ledger()
    assert snap["ledger_doubtful"] is True
    assert snap["reconciled"] is False


def test_f2_compact_refuses_bad_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    snap = tmp_path / "snapshot.json"
    _ledger.append_event(
        path,
        {
            "type": _ledger.LEDGER_EVENT_ADD,
            "memory_id": "m1",
            "key": "m1",
            "text": "测试记忆",
            "meta": {},
        },
    )
    out = _ledger.compact_ledger(path, snap)
    assert out["events_archived"] == 1
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("not-json-at-all\n")
    with pytest.raises(ValueError):
        _ledger.compact_ledger(path, snap)


# ---------- F3：撤销核对补齐 ----------

def _restored_row(store, mid):
    """执行 demote 后走宿主路把字段恢复到位，返回
    (补偿前 meta, 恢复后 row)。"""
    receipt = _execute(store, "demote", mid)
    pre_meta = _row(store, mid)["meta"]
    store._update_core_fields(mid, {"user_confirmed": True})
    store._conn.execute(
        "UPDATE memories SET user_confirmed=1 WHERE id=?", (mid,)
    )
    store._conn.commit()
    return receipt, pre_meta, _row(store, mid)


def test_f3_verify_columns_mismatch(store):
    mid = _add_user(store, "BOOS 在美国出差用现金")
    _receipt, pre_meta, row = _restored_row(store, mid)
    # 只把独立列改回 0：meta 与列分叉（两段写第二段失败
    # 留下的正是这个形状）。
    store._conn.execute(
        "UPDATE memories SET user_confirmed=0 WHERE id=?", (mid,)
    )
    store._conn.commit()
    verify = store._controlled_verify_revocation(
        mid, "user_confirmed", True, row["text"], pre_meta,
        "demote",
    )
    assert verify["ok"] is False
    assert verify["checks"]["columns_match_meta"] is False
    assert verify["checks"]["ledger_matches"] is True


def test_f3_verify_members_mismatch(store):
    mid = _add_user(store, "BOOS 用微信联系朋友")
    _receipt, pre_meta, row = _restored_row(store, mid)
    # 投影里多出一行账本没有的活成员（列/meta 分叉之外的
    # 另一种分叉）：成员集合与账本对不上。
    store._conn.execute(
        "INSERT INTO memories (id, text, metadata_json,"
        " created_at, user_confirmed) VALUES (?, ?, ?, ?, 1)",
        (
            "phantom-row-1",
            "账本里没有的一行",
            json.dumps({"source": "user", "user_confirmed": True}),
            "2026-11-26T00:00:00+00:00",
        ),
    )
    store._conn.commit()
    verify = store._controlled_verify_revocation(
        mid, "user_confirmed", True, row["text"], pre_meta,
        "demote",
    )
    assert verify["ok"] is False
    assert verify["checks"]["members_match_ledger"] is False
    assert verify["checks"]["ledger_matches"] is True


def test_f3_second_stage_failure_not_clean(store):
    mid = _add_user(store, "BOOS 的笔记本是华硕的")
    receipt = _execute(store, "demote", mid)
    # 触发器打断 demote 补偿的第二段写（独立列回写）。
    store._conn.execute(
        "CREATE TRIGGER fail_uc_restore"
        " BEFORE UPDATE OF user_confirmed ON memories"
        " WHEN NEW.user_confirmed=1"
        " BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    store._conn.commit()
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=receipt["detail"]["op_id"],
                    session=_session(store))
    )
    # 补偿事件落了账（status=revoked 是事实），但列/meta
    # 分叉真实存在——不许报干净成功。
    assert rev["status"] == "revoked", rev
    assert rev["ok"] is False
    assert rev["reason"] == "invariant_failed"
    assert rev["detail"]["verify"]["columns_match_meta"] is False
    row = _row(store, mid)
    assert row["meta"].get("user_confirmed") is True
    assert row["user_confirmed"] is False


# ---------- F4：提交未定案不许干净成功 ----------

def test_f4_undetermined_not_clean_success(store, tmp_path,
                                           monkeypatch):
    mid = _add_user(store, "BOOS 每天睡白天")
    receipt = _execute(store, "demote", mid)
    op_id = receipt["detail"]["op_id"]
    demote_seq = store._controlled_intents[op_id].get("ledger_seq")
    assert demote_seq

    real_read = _ledger.read_events
    state = {"armed": True, "calls": 0}

    def flaky_read(path):
        events, info = real_read(path)
        if state["armed"]:
            state["calls"] += 1
            if state["calls"] == 1:
                # 只在「补偿落账后按 op 核验」那一次读上注
                # 入存疑：事件集合截到原 op 为止＋坏行标记
                # ——补偿生死定不了（undetermined）。
                events = [
                    e for e in events
                    if (e.get("seq") or 0) <= demote_seq
                ]
                info = dict(info)
                info["bad_lines"] = (info.get("bad_lines") or 0) + 1
        return events, info

    monkeypatch.setattr(_ledger, "read_events", flaky_read)
    rev = store.revoke_controlled_change(
        _revoke_req(op_id=op_id, rev_key="rev-f4-1",
                    session=_session(store))
    )
    state["armed"] = False
    # 补偿其实落了账、核对也过，但提交未定案——顶层不许
    # 干净成功，事实（已恢复）不许丢。
    assert rev["status"] == "revoked", rev
    assert rev["ok"] is False
    assert rev["reason"] == "committed_unverified"
    assert rev["detail"]["commit_state"] == "committed_unverified"
    assert _row(store, mid)["user_confirmed"] is True
    assert store._controlled_frozen_reason == "commit_unverified"
    rev_intent = store._controlled_intents[rev["revocation_op_id"]]
    assert rev_intent["status"] == "committed_unverified"
    assert rev_intent["final_ok"] is False
    assert rev_intent["final_reason"] == "committed_unverified"

    # 重启后回执与冻结都不洗白。
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    got = store2.controlled_revocation_receipt("rev-f4-1")
    assert got is not None
    assert got["ok"] is False
    assert got["reason"] == "committed_unverified"
    assert store2._controlled_frozen_reason == "commit_unverified"
