# tests/test_revocation_fixes2_20261126.py（H3 补修轮 2：贵妃
# 第二十九份复查 F2 残留 R1/R2，2026-11-26）
#
# 覆盖口径（裁决存档 TWENTY_NINTH_REVIEW_adjudication.md）：
#   R1-a 自检适配器 verify_backup：源账本存疑（坏行/截断）时
#        必须 ok=False，不许拿残余事件建备份再报演练通过；
#        完好账本仍 ok=True（原行为不回退）。
#   R1-b 自检适配器 supersede_events_since：读异常或账本存疑
#        必须把存疑沿自检契约传出去（抛 RuntimeError，框架外层
#        把整轮自检记为失败），不许静默回 [] 冒充无事件；
#        完好时原结果不变。并经真自检框架 run_self_check 验
#        一次：存疑信号确实传到框架层、不被吞成空报告。
#   R2 deletion_scope：读账本抛异常的分支必须置
#        ledger_doubtful=True（坏行路径上一轮已修，本轮补异常
#        路径；完好/坏行两组不回退）。
#
# 全程真路径：真 SQLiteStore 构造器、真自检适配器
# （build_self_check_adapter）、真账本/备份函数；故障只注入
# 在边界上（往真账本文件尾加坏行、把末行截半、把 ledger_path
# 指向目录让真 read_events 抛异常）。嵌入/判官打桩为 None，
# 同 test_revocation_fixes 口径。

import pytest

from memory import ledger as _ledger
from memory import self_check as _self_check
from memory.sqlite_store import SQLiteStore
from memory.store_self_check import build_self_check_adapter


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


def _append_supersede(store, mid):
    """往真账本追加一笔取代事件（走真 append_event，ts 自动落）。"""
    return _ledger.append_event(
        store.ledger_path,
        {
            "type": _ledger.LEDGER_EVENT_SUPERSEDE,
            "memory_id": mid,
            "old_text": "旧说法",
        },
    )


def _append_garbage_line(store):
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("this-line-is-not-json\n")


def _truncate_last_line(store):
    data = open(store.ledger_path, "rb").read()
    assert data.endswith(b"\n")
    cut = data[:-9]
    assert not cut.endswith(b"\n")
    open(store.ledger_path, "wb").write(cut)


def _ledger_info(store):
    _events, info = _ledger.read_events(store.ledger_path)
    return info


# ---------- R1-a：verify_backup 账本存疑必须 ok=False ----------

def test_r1a_verify_backup_intact_ok(store):
    _add_user(store, "BOOS 每天定投五块钱买纳指")
    adapter = build_self_check_adapter(store)
    result = adapter.verify_backup()
    assert result.get("ok") is True, result


def test_r1a_verify_backup_bad_line_fails(store):
    _add_user(store, "BOOS 每天定投五块钱买纳指")
    _append_garbage_line(store)
    assert _ledger_info(store)["bad_lines"] >= 1
    adapter = build_self_check_adapter(store)
    result = adapter.verify_backup()
    assert result.get("ok") is False, result
    assert "不完整" in (result.get("detail") or ""), result


def test_r1a_verify_backup_truncated_fails(store):
    _add_user(store, "BOOS 每天定投五块钱买纳指")
    _truncate_last_line(store)
    assert _ledger_info(store)["truncated"] is True
    adapter = build_self_check_adapter(store)
    result = adapter.verify_backup()
    assert result.get("ok") is False, result
    assert "不完整" in (result.get("detail") or ""), result


# ---------- R1-b：supersede_events_since 存疑必须传出 ----------

def test_r1b_supersede_events_intact(store):
    mid = _add_user(store, "BOOS 以前住洛杉矶")
    _append_supersede(store, mid)
    adapter = build_self_check_adapter(store)
    out = adapter.supersede_events_since(7)
    assert len(out) == 1, out
    assert out[0].get("id") == mid, out


def test_r1b_supersede_events_bad_line_raises(store):
    mid = _add_user(store, "BOOS 以前住洛杉矶")
    _append_supersede(store, mid)
    _append_garbage_line(store)
    assert _ledger_info(store)["bad_lines"] >= 1
    adapter = build_self_check_adapter(store)
    with pytest.raises(RuntimeError):
        adapter.supersede_events_since(7)


def test_r1b_supersede_events_truncated_raises(store):
    mid = _add_user(store, "BOOS 以前住洛杉矶")
    _append_supersede(store, mid)
    _truncate_last_line(store)
    assert _ledger_info(store)["truncated"] is True
    adapter = build_self_check_adapter(store)
    with pytest.raises(RuntimeError):
        adapter.supersede_events_since(7)


def test_r1b_supersede_events_read_error_raises(store, tmp_path):
    _add_user(store, "BOOS 以前住洛杉矶")
    # 边界注入：ledger_path 指向目录，真 read_events 打开即抛异常。
    store.ledger_path = tmp_path / "memory"
    adapter = build_self_check_adapter(store)
    with pytest.raises(RuntimeError):
        adapter.supersede_events_since(7)


def test_r1b_doubt_reaches_self_check_framework(store, tmp_path):
    """存疑信号要真传到自检框架层：run_self_check 不许把读失败
    吞成一份 replaced_last_month 为空的干净报告。"""
    _add_user(store, "BOOS 以前住洛杉矶")
    store.ledger_path = tmp_path / "memory"
    adapter = build_self_check_adapter(store)
    with pytest.raises(RuntimeError):
        _self_check.run_self_check(adapter)


# ---------- R2：deletion_scope 读异常必须标存疑 ----------

def test_r2_deletion_scope_intact_not_doubtful(store):
    mid = _add_user(store, "BOOS 每天定投五块钱买纳指")
    scope = store.deletion_scope(mid)
    assert scope.get("found") is True, scope
    assert scope.get("history_count", 0) >= 1, scope
    assert scope.get("ledger_doubtful") is False, scope


def test_r2_deletion_scope_bad_line_doubtful(store):
    mid = _add_user(store, "BOOS 每天定投五块钱买纳指")
    _append_garbage_line(store)
    assert _ledger_info(store)["bad_lines"] >= 1
    scope = store.deletion_scope(mid)
    assert scope.get("found") is True, scope
    assert scope.get("ledger_doubtful") is True, scope


def test_r2_deletion_scope_read_error_doubtful(store, tmp_path):
    mid = _add_user(store, "BOOS 每天定投五块钱买纳指")
    # 边界注入：ledger_path 指向目录，真 read_events 抛异常。
    store.ledger_path = tmp_path / "memory"
    scope = store.deletion_scope(mid)
    assert scope.get("found") is True, scope
    assert scope.get("history_count") == 0, scope
    assert scope.get("ledger_doubtful") is True, scope
