# tests/test_f2r2_repro_20261127.py（整包验收第三批：F2-R1/R2 定向复现，
# 2026-11-27）
#
# 原声称（贵妃第二十九份复查 F2-R1，裁决存档
# TWENTY_NINTH_REVIEW_adjudication.md，十一真链复现于修复前代码）：
# memory/store_self_check.py 的 verify_backup 不看 read_events 的
# _info，拿残余事件建自洽备份再验副本——源账本有坏行（bad_lines）
# 或截断（truncated）时，「备份演练」仍报 ok=True 假报通过；
# supersede_events_since 同样吞 _info、读异常静默回 []。
#
# 本次定向复现（当前整包栈，含 H3 补修轮 2 的 57c2481）结论分两半：
#
# 【原声称两模式：已不成立】坏行、截断两组 verify_backup 均正确
# 返回 ok=False（detail 明说源账本不完整），完好账本仍 ok=True。
# 原因：57c2481 的 F2-R1 修复已在本栈落地——verify_backup 在
# build_backup 之前先查 _info，坏行/截断直接判败；
# supersede_events_since 存疑改抛 RuntimeError。下方三条正向
# 测试断言这一现状正确行为，与 test_revocation_fixes2 的
# test_r1a_* 同口径、互为印证（那份是施工方修复自证，本份是
# 整包验收侧的独立复跑）。
#
# 【同类残留：仍成立，strict xfail 挂账】read_events 的 _info 只报
# 坏行与截断，不报「整行事件丢失」——把账本中间一整行事件删掉，
# 其余行全部合法、seq 出现缺口时：ledger.verify_ledger 明确判
# ok=False（gaps=[被删 seq]，其 docstring 定性「某几行事件丢了…
# 这种事不该发生」），但 verify_backup 从不查 seq 连续性，照样
# 拿残缺事件集合建备份并报 ok=True「备份演练通过」；
# supersede_events_since 同样静默少报一条、不抛存疑。
# 即「账本损坏时备份演练假报通过」这一缺陷类未被 F2-R1 修复
# 完全覆盖：修复只堵了坏行/截断两种损坏形态，整行丢失形态仍漏。
# 下方两条 xfail(strict=True) 断言正确期望（损坏时应判败/应抛
# 存疑）；机制一旦修好，XPASS 会让全套变红逼复核，不许改断言
# 迁就现状。修法等 BOOS 在整包总结时拍板，本文件只复现不修。
#
# 全程真路径：真 SQLiteStore 构造器、真适配器、真账本函数；
# 故障只注入在真账本文件上（尾加坏行、末行截半、整行删除）。

import json

import pytest

from memory import ledger as _ledger
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


def _add_user(store, text):
    saved = store.add_memory_text(text, source="user")
    assert saved.get("event") == "ADD", saved
    return saved["id"]


def _seed_three(store):
    return [
        _add_user(store, "BOOS 每天定投五块钱买纳指"),
        _add_user(store, "BOOS 下个月去成都出差"),
        _add_user(store, "BOOS 喜欢喝乌龙茶"),
    ]


def _ledger_lines(store):
    return open(store.ledger_path, encoding="utf-8").read().splitlines()


def _rewrite_lines(store, lines):
    open(store.ledger_path, "w", encoding="utf-8").write(
        "\n".join(lines) + "\n"
    )


# ---------- 原声称模式：已修复，断言现状正确行为 ----------

def test_verify_backup_intact_ok(store):
    _seed_three(store)
    result = build_self_check_adapter(store).verify_backup()
    assert result.get("ok") is True, result


def test_verify_backup_bad_line_fails(store):
    """原声称模式 A（尾加坏行）：修复前假报通过，现应判败。"""
    _seed_three(store)
    with open(store.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("this-line-is-not-json\n")
    _events, info = _ledger.read_events(store.ledger_path)
    assert info["bad_lines"] >= 1, info
    result = build_self_check_adapter(store).verify_backup()
    assert result.get("ok") is False, result
    assert "不完整" in (result.get("detail") or ""), result


def test_verify_backup_truncated_fails(store):
    """原声称模式 B（末行截半）：修复前假报通过，现应判败。"""
    _seed_three(store)
    data = open(store.ledger_path, "rb").read()
    assert data.endswith(b"\n")
    open(store.ledger_path, "wb").write(data[:-9])
    _events, info = _ledger.read_events(store.ledger_path)
    assert info["truncated"] is True, info
    result = build_self_check_adapter(store).verify_backup()
    assert result.get("ok") is False, result
    assert "不完整" in (result.get("detail") or ""), result


# ---------- 同类残留：整行丢失（seq 缺口）仍假报，strict xfail ----------

def _delete_middle_line(store):
    lines = _ledger_lines(store)
    assert len(lines) >= 3, lines
    del lines[1]  # 第二行（seq=2 的 ADD 事件）整行消失
    _rewrite_lines(store, lines)
    verdict = _ledger.verify_ledger(store.ledger_path)
    assert verdict["ok"] is False and verdict["gaps"] == [2], verdict
    _events, info = _ledger.read_events(store.ledger_path)
    # 缺口形态下 _info 干净——这正是它能绕过 F2-R1 前置检查的原因
    assert info["bad_lines"] == 0 and info["truncated"] is False, info


@pytest.mark.xfail(
    strict=True,
    reason="F2-R1 残留：账本整行丢失（seq 缺口）时 verify_backup 仍报 "
    "ok=True 假报通过——它只查 bad_lines/truncated、从不查 seq "
    "连续性。修法待 BOOS 拍板（整包验收第三批定向复现，2026-11-27）。",
)
def test_verify_backup_seq_gap_must_fail(store):
    _seed_three(store)
    _delete_middle_line(store)
    result = build_self_check_adapter(store).verify_backup()
    assert result.get("ok") is False, result


@pytest.mark.xfail(
    strict=True,
    reason="F2-R1 残留：账本整行丢失（seq 缺口）吞掉一笔取代事件时，"
    "supersede_events_since 静默少报、不抛 RuntimeError 存疑。"
    "修法待 BOOS 拍板（整包验收第三批定向复现，2026-11-27）。",
)
def test_supersede_events_seq_gap_must_raise(store):
    ids = _seed_three(store)
    for mid in ids[:2]:
        _ledger.append_event(
            store.ledger_path,
            {
                "type": _ledger.LEDGER_EVENT_SUPERSEDE,
                "memory_id": mid,
                "old_text": "旧说法",
            },
        )
    adapter = build_self_check_adapter(store)
    assert len(adapter.supersede_events_since(7)) == 2
    kept = [
        ln for ln in _ledger_lines(store)
        if not (
            json.loads(ln).get("type") == _ledger.LEDGER_EVENT_SUPERSEDE
            and json.loads(ln).get("memory_id") == ids[0]
        )
    ]
    assert len(kept) == len(_ledger_lines(store)) - 1
    _rewrite_lines(store, kept)
    verdict = _ledger.verify_ledger(store.ledger_path)
    assert verdict["ok"] is False and verdict["gaps"], verdict
    with pytest.raises(RuntimeError):
        adapter.supersede_events_since(7)
