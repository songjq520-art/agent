# tests/test_pending_questions_save_warn_20261124.py — 修复 1 回归：
# PendingQuestions._save() 写盘失败时不再静默吞掉，要记一条 warning 日志。
import logging
from unittest import mock

from memory.pending_questions import PendingQuestions


def test_save_failure_logs_warning(tmp_path, caplog):
    pq = PendingQuestions(memory_dir=str(tmp_path / "memory"))
    pq.add("importance", "第一个测试问题", ["进核心", "当一次性经历", "不存"], "原因")
    assert len(pq.list_unanswered()) == 1
    with mock.patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        with caplog.at_level(logging.WARNING, logger="memory.pending_questions"):
            # 写盘失败也不能抛错：问题还在内存里，调用方无感。
            pq.add("importance", "第二个测试问题", ["进核心", "当一次性经历", "不存"], "原因")
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "memory.pending_questions"
    ]
    assert warnings, "写盘失败时应记一条 warning 日志"
    assert "保存失败" in warnings[0].getMessage()
    assert any(q["text"] == "第二个测试问题" for q in pq.list_unanswered())
