# tests/test_accounting_crash_recovery_20261128.py（记账崩溃恢复测试）
# 作用：模拟进程在写账本中途崩溃留下的各种残局——尾行是截断的半行
#       JSON、半行把多字节字符写断（非法 UTF-8）、文件中间有整行
#       损坏、空文件——验证 accounting 的读取/汇总/对账：不崩、
#       完好行一笔不丢、坏行被明确计数（corrupt_lines，不静默吞）、
#       恢复后继续追加不把文件越写越坏（新行不与半行粘连）。
# 背景：本测试在旧实现上实测暴露了两个真问题并已在同分支修复——
#       ①半行无换行时，下一次追加的新行会和半行粘成一坨、连丢两笔
#       （record_call 现先补换行给半行封口）；②半行写断多字节字符
#       时，read_calls 抛 UnicodeDecodeError 整本读不出（现容错解码，
#       坏行计数）。全部离线，只碰临时文件。

import json
from datetime import datetime

import pytest

import accounting


def _month():
    return datetime.now().strftime("%Y-%m")


def _record(path, cost, status="ok"):
    return accounting.record_call(
        provider="deepseek", model="deepseek-flash", cost_usd=cost,
        status=status, purpose="测试", caller="test.crash", path=path)


def _append_raw(path, data):
    mode = "ab" if isinstance(data, bytes) else "a"
    with path.open(mode, encoding=None if mode == "ab" else "utf-8") as fh:
        fh.write(data)


# ---------- 1. 尾行截断（半行 JSON、无换行） ----------

def test_truncated_tail_skipped_and_counted(tmp_path):
    """尾巴是半行 JSON：好行全在、半行跳过且计入 corrupt_lines。"""
    path = tmp_path / "llm_calls.jsonl"
    _record(path, 0.10)
    _record(path, 0.20)
    _append_raw(path, '{"ts": "2026-11-28T00:00:00", "provider": "deep')
    rows = accounting.read_calls(path=path)
    assert len(rows) == 2
    assert [r["cost_usd"] for r in rows] == [0.10, 0.20]
    summary = accounting.monthly_summary(_month(), path=path)
    assert summary["calls"] == 2
    assert summary["total_usd"] == pytest.approx(0.30)
    assert summary["corrupt_lines"] == 1
    # 对账照常出结论，不被坏行拖崩
    verdict = accounting.reconcile(_month(), 0.30, path=path)
    assert verdict["ok"] is True
    assert verdict["corrupt_lines"] == 1


# ---------- 2. 半行写断多字节字符（非法 UTF-8） ----------

def test_truncated_multibyte_char_does_not_crash(tmp_path):
    """半行把中文字符写断：读取不许抛 UnicodeDecodeError。

    旧实现此处实测抛异常、整本账读不出；修复后好行照常读、
    断行计入坏行数。"""
    path = tmp_path / "llm_calls.jsonl"
    _record(path, 0.10)
    full = json.dumps(
        {"ts": "2026-11-28T00:00:00", "purpose": "日常", "cost_usd": 0.5},
        ensure_ascii=False).encode("utf-8")
    cut = full.index("日".encode("utf-8")) + 1  # 断在三字节字符中间
    _append_raw(path, full[:cut])
    rows = accounting.read_calls(path=path)
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == 0.10
    summary = accounting.monthly_summary(_month(), path=path)
    assert summary["calls"] == 1
    assert summary["corrupt_lines"] == 1


# ---------- 3. 空文件 / 只有坏行 / 空白行 ----------

def test_empty_file_is_clean(tmp_path):
    """空文件：读出空列表、汇总全零、对账（对方 0）判对得上。"""
    path = tmp_path / "llm_calls.jsonl"
    path.write_bytes(b"")
    assert accounting.read_calls(path=path) == []
    summary = accounting.monthly_summary(_month(), path=path)
    assert summary["calls"] == 0
    assert summary["total_usd"] == 0.0
    assert summary["corrupt_lines"] == 0
    assert accounting.reconcile(_month(), 0.0, path=path)["ok"] is True


def test_only_corrupt_lines(tmp_path):
    """整本只有坏行：读出空列表，坏行一笔笔记数，不当好行吞掉。"""
    path = tmp_path / "llm_calls.jsonl"
    path.write_text("这不是 JSON\n{\"half\": \n", encoding="utf-8")
    assert accounting.read_calls(path=path) == []
    assert accounting.monthly_summary(_month(), path=path)["corrupt_lines"] == 2


def test_blank_lines_are_not_corrupt(tmp_path):
    """空白行（含纯空格行）是无害的，不计坏行。"""
    path = tmp_path / "llm_calls.jsonl"
    _record(path, 0.10)
    _append_raw(path, "\n   \n")
    summary = accounting.monthly_summary(_month(), path=path)
    assert summary["calls"] == 1
    assert summary["corrupt_lines"] == 0


# ---------- 4. 文件中间的整行损坏 ----------

def test_mid_file_corrupt_line(tmp_path):
    """中间一行整行损坏：前后好行都在、顺序不变、坏行计数。"""
    path = tmp_path / "llm_calls.jsonl"
    _record(path, 0.10)
    _append_raw(path, "这一行彻底坏了\n")
    _record(path, 0.20)
    rows = accounting.read_calls(path=path)
    assert [r["cost_usd"] for r in rows] == [0.10, 0.20]
    assert accounting.monthly_summary(_month(), path=path)["corrupt_lines"] == 1


def test_non_dict_json_lines_count_corrupt(tmp_path):
    """能解析但不是字典的行（数组/数字）不是账，也计坏行。"""
    path = tmp_path / "llm_calls.jsonl"
    _record(path, 0.10)
    _append_raw(path, "[1, 2, 3]\n42\n")
    rows = accounting.read_calls(path=path)
    assert len(rows) == 1
    assert accounting.monthly_summary(_month(), path=path)["corrupt_lines"] == 2


# ---------- 5. 恢复后继续追加：不许越写越坏 ----------

def test_append_after_truncated_tail_keeps_new_rows(tmp_path):
    """半行之后继续记账：新行必须独立成行可读，不能和半行粘连。

    旧实现此处实测连丢：追加的新行和半行粘成一坨、两行都解析
    不了。修复后（追加前先补换行封口）：新行完好、半行仍只算
    一行坏行，再追加也不会更坏。"""
    path = tmp_path / "llm_calls.jsonl"
    _record(path, 0.10)
    _append_raw(path, '{"ts": "2026-11-28T00:00:00", "provider": "deep')
    assert _record(path, 0.20) is True
    rows = accounting.read_calls(path=path)
    assert [r["cost_usd"] for r in rows] == [0.10, 0.20]
    assert accounting.monthly_summary(_month(), path=path)["corrupt_lines"] == 1
    # 再追加一笔：文件只增好行，坏行数不涨
    assert _record(path, 0.30) is True
    rows = accounting.read_calls(path=path)
    assert [r["cost_usd"] for r in rows] == [0.10, 0.20, 0.30]
    assert accounting.monthly_summary(_month(), path=path)["corrupt_lines"] == 1
    assert path.read_bytes().endswith(b"\n")


def test_append_after_complete_row_without_newline(tmp_path):
    """尾行是完整 JSON 只是没换行（换行字节丢失）：追加后两行都在。"""
    path = tmp_path / "llm_calls.jsonl"
    row = {"ts": f"{_month()}-01T00:00:00", "provider": "deepseek",
           "model": "deepseek-flash", "purpose": "测试", "caller": "t",
           "ledger_book": "", "task_id": "", "trace_id": "",
           "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
           "cost_usd": 0.40, "status": "ok", "estimated": False}
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    assert _record(path, 0.50) is True
    rows = accounting.read_calls(path=path)
    assert [r["cost_usd"] for r in rows] == [0.40, 0.50]
    assert accounting.monthly_summary(_month(), path=path)["corrupt_lines"] == 0
