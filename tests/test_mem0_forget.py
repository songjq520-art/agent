# tests/test_mem0_forget.py（深度删除 + 改/忘区分 + 删除范围展示）
# 作用：守住 2026-11-09 这一轮的深度删除线——
#   删一条记忆时，向量和 history.db 里的审计行一起清掉，验三处
#   （向量、历史、检索）都干净才算删完；删之前先展示删除范围给
#   BOOS 看、拿到确认才动手；删失败/删一半时报清楚状态，绝不谎称删干净；
#   改（correct_memory）不是忘：改完 history 里还留着审计记录。
#   忘掉的记忆不会“复活”：提取只对新轮次跑，不回头重提旧对话，
#   只有 BOOS 重新说起它，才会被重新记下来。
# 测试全部是离线假对象测试（FakeMemory，见 tests/conftest.py）：
# 不碰真实 mem0、不联网、不产生费用。history.db 的 sqlite 清理
# 那一路用一个按真实 schema 建的临时 sqlite 文件测（见
# test_delete_purges_real_sqlite_history_rows）。

import json
import sqlite3

import pytest

from memory.mem0_store import Mem0Store
from conftest import FakeMemory, make_fake_store


def _add(store, text):
    """存一条记忆，返回它的 id。"""
    result = store.add_memory_text(text)
    return result["results"][0]["id"]


# ---------- 深度删除：向量 + 历史都干净 ----------


def test_delete_removes_vector_and_history(tmp_path):
    """深度删除：向量删掉、history 审计行也清掉，验三处都干净；
    另一条记忆原封不动。"""
    store = make_fake_store(tmp_path)
    go_id = _add(store, "我喜欢用 Go 写代码。")
    _add(store, "我喜欢喝咖啡。")
    assert len(store.history(go_id)) == 1  # 删之前有 ADD 审计行
    assert store.delete_memory(go_id, approve=True) is True
    assert [item["text"] for item in store.get_all()] == ["我喜欢喝咖啡。"]
    assert store.history(go_id) == []
    verified = store.verify_forgotten(go_id, "我喜欢用 Go 写代码。")
    assert verified["ok"] is True
    assert verified["vector_gone"] is True
    assert verified["history_clean"] is True
    assert verified["search_empty"] is True
    assert store.list_pending_deletes() == []  # 删完验干净，删除日志清掉了


def test_delete_scope_preview_does_not_delete(tmp_path):
    """删之前先展示删除范围：哪条记忆、几行历史、在范围内的是什么、
    不在范围内的是什么（备份、日志、messages 表、Chroma 文件碎片），
    展示本身只读不删，库里原封不动。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Go 写代码。")
    scope = store.deletion_scope(mem_id)
    assert scope["found"] is True
    assert scope["memory"]["text"] == "我喜欢用 Go 写代码。"
    assert scope["history_count"] == 1
    assert len(scope["history_entries"]) == 1
    in_scope = " ".join(entry["where"] for entry in scope["in_scope"])
    assert "Chroma" in in_scope
    assert "history" in in_scope
    out_of_scope = " ".join(entry["where"] for entry in scope["out_of_scope"])
    assert "备份" in out_of_scope
    assert "messages" in out_of_scope
    assert "日志" in out_of_scope
    # 展示完还没删：还在库里。
    assert any(item["id"] == mem_id for item in store.get_all())
    assert store.deletion_scope("不存在的id")["found"] is False


def test_delete_requires_approval(tmp_path):
    """没确认不删：approve=False 抛 PermissionError，记忆还在；
    删之前展示范围也不需要 approve（展示只是给人看）。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我叫小宋。")
    with pytest.raises(PermissionError, match="需要 BOOS 同意"):
        store.delete_memory(mem_id, approve=False)
    assert any(item["id"] == mem_id for item in store.get_all())


def test_research_after_forget_finds_nothing(tmp_path):
    """忘掉之后再搜：几种问法都搜不到被忘的文本（lifecycle 步骤 9a 的复现）。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Go 写代码。")
    store.delete_memory(mem_id, approve=True)
    for question in [
        "我喜欢用什么语言写代码？",
        "我现在喜欢用 Go 写代码吗？",
        "我喜欢用 Go 写代码吗？",
    ]:
        assert not any("Go" in item["text"] for item in store.search(question)), question


def test_restart_does_not_resurrect(tmp_path):
    """重启不复活：换一个 Mem0Store 实例（同一底层 FakeMemory、同一路径，
    模拟服务重启后重新加载），被忘的记忆还是没有——向量没了、历史没了，
    没地方能把它捞回来。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Go 写代码。")
    _add(store, "我喜欢喝咖啡。")
    store.delete_memory(mem_id, approve=True)
    restarted = Mem0Store(
        memory=store._memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    assert [item["text"] for item in restarted.get_all()] == ["我喜欢喝咖啡。"]
    assert restarted.history(mem_id) == []
    assert not any("Go" in item["text"] for item in restarted.search("我喜欢用什么语言写代码？"))


def test_delete_purges_real_sqlite_history_rows(tmp_path):
    """history.db 那一路是真清：按真实 mem0 的 schema 建一个临时
    sqlite 文件，里面给这条记忆留两行审计（ADD 行 + DELETE 行，
    DELETE 行的 old_memory 里还留着被删记忆的全文——lifecycle 步骤
    9c 验出的缺口），删完这两行都没了，文件里再也搜不到那段文本。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Go 写代码。")
    db_path = tmp_path / "history.db"
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE history (id TEXT PRIMARY KEY, memory_id TEXT,"
        " old_memory TEXT, new_memory TEXT, event TEXT,"
        " created_at DATETIME, updated_at DATETIME,"
        " is_deleted INTEGER, actor_id TEXT, role TEXT)"
    )
    con.execute(
        "INSERT INTO history VALUES (?, ?, NULL, ?, 'ADD', '2026-11-09T00:00:00+00:00', NULL, 0, NULL, NULL)",
        ("row-add", mem_id, "我喜欢用 Go 写代码。"),
    )
    con.execute(
        "INSERT INTO history VALUES (?, ?, ?, NULL, 'DELETE', '2026-11-09T00:01:00+00:00', NULL, 1, NULL, NULL)",
        ("row-del", mem_id, "我喜欢用 Go 写代码。"),
    )
    con.commit()
    con.close()
    assert store.delete_memory(mem_id, approve=True) is True
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT * FROM history WHERE memory_id = ?", (mem_id,)
        ).fetchall()
        assert rows == []
        leaked = con.execute(
            "SELECT * FROM history WHERE old_memory LIKE '%Go%' OR new_memory LIKE '%Go%'"
        ).fetchall()
        assert leaked == []
    finally:
        con.close()


def test_purge_history_rows_direct(tmp_path):
    """直接测 _purge_history_rows：sqlite 文件里有行就清掉、返回行数；
    没有文件/没有表时返回 0 行，不算失败（FakeMemory 测试就是这样）。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Go 写代码。")
    assert store._purge_history_rows(mem_id)["rows_removed"] == 1  # 假对象里的 ADD 行
    assert store.history(mem_id) == []
    assert store._purge_history_rows("不存在的id")["rows_removed"] == 0


# ---------- 失败 / 崩溃：报清楚状态，不谎称成功 ----------


def test_delete_failure_reports_clear_state(tmp_path):
    """向量删失败时报清楚：抛 RuntimeError、说明“向量没删掉、记忆还在”，
    库里原封不动；删除日志留着（能查到这次没删完），里面只有 id 和
    文本哈希、没有原文。"""
    store = make_fake_store(tmp_path)

    class _FailingDeleteMemory(FakeMemory):
        def delete(self, memory_id):
            raise RuntimeError("向量库炸了")

    failing = _FailingDeleteMemory()
    failing.add([{"role": "user", "content": "我喜欢用 Go 写代码。"}], infer=False)
    failing_store = Mem0Store(
        memory=failing,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    mem_id = failing_store.get_all()[0]["id"]
    with pytest.raises(RuntimeError, match="向量没删掉"):
        failing_store.delete_memory(mem_id, approve=True)
    assert [item["text"] for item in failing_store.get_all()] == ["我喜欢用 Go 写代码。"]
    pending = failing_store.list_pending_deletes()
    assert len(pending) == 1
    assert pending[0]["memory_id"] == mem_id
    assert "text" not in pending[0]  # 日志里没有原文
    assert pending[0]["text_sha256"]  # 只有哈希


def test_history_purge_failure_reports_partial_state(tmp_path):
    """删了一半时报清楚：向量已删、清 history 失败——抛 RuntimeError、
    说明“向量已删、history 可能还剩行”，history 行确实还在，
    删除日志也留着，调用方知道要补删、不会误以为删干净了。"""

    class _FailingPurgeMemory(FakeMemory):
        def purge_history(self, memory_id):
            raise RuntimeError("history.db 锁住了")

    failing = _FailingPurgeMemory()
    failing.add([{"role": "user", "content": "我喜欢用 Go 写代码。"}], infer=False)
    failing_store = Mem0Store(
        memory=failing,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    mem_id = failing_store.get_all()[0]["id"]
    with pytest.raises(RuntimeError, match="删除了一半"):
        failing_store.delete_memory(mem_id, approve=True)
    assert failing_store.get_all() == []  # 向量确实删了
    assert failing_store.history(mem_id) != []  # history 行确实还在
    assert any(entry["memory_id"] == mem_id for entry in failing_store.list_pending_deletes())


def test_verify_forgotten_reports_each_location(tmp_path):
    """verify_forgotten 逐项报：还没删的记忆三项全是 False、ok 是 False；
    删完的三项全是 True、ok 是 True。调用方不用猜是哪一处没干净。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Go 写代码。")
    before = store.verify_forgotten(mem_id, "我喜欢用 Go 写代码。")
    assert before["ok"] is False
    assert before["vector_gone"] is False
    assert before["history_clean"] is False
    store.delete_memory(mem_id, approve=True)
    after = store.verify_forgotten(mem_id, "我喜欢用 Go 写代码。")
    assert after["ok"] is True
    assert after["history_rows_remaining"] == 0


def test_delete_does_not_log_text(tmp_path, caplog):
    """删记忆不记原文进日志：日志里只出现 id 和条数，不出现被删的文本。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用独角兽写代码。")
    with caplog.at_level("INFO", logger="agent"):
        store.delete_memory(mem_id, approve=True)
    assert not any("独角兽" in record.getMessage() for record in caplog.records)


# ---------- 删一条不牵连相似的；改不是忘 ----------


def test_deleting_one_does_not_delete_similar(tmp_path):
    """删一条不牵连相似的：黑咖啡和拿铁都在库里，删黑咖啡那条，
    拿铁的向量和 history 都原封不动。"""
    store = make_fake_store(tmp_path)
    black_id = _add(store, "我喜欢喝黑咖啡。")
    latte_id = _add(store, "我喜欢喝拿铁咖啡。")
    assert store.delete_memory(black_id, approve=True) is True
    assert [item["text"] for item in store.get_all()] == ["我喜欢喝拿铁咖啡。"]
    assert store.history(latte_id) != []
    assert store.history(black_id) == []
    assert any("拿铁" in item["text"] for item in store.search("我喜欢喝什么？"))


def test_modify_is_not_forget(tmp_path):
    """改不是忘：纠错换文本后，history 里还留着 UPDATE 审计行
    （旧文本在 old_memory 里，有据可查）；搜旧内容只回来新文本；
    verify_forgotten 对这条是 False——改过的记忆没有被“忘掉”。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Python 写代码。")
    assert store.correct_memory("我喜欢用 Python 写代码", "我喜欢用 Go 写代码。")["status"] == "updated"
    entries = store.history(mem_id)
    assert [entry["event"] for entry in entries] == ["ADD", "UPDATE"]
    assert entries[-1]["old_memory"] == "我喜欢用 Python 写代码。"
    assert [item["text"] for item in store.get_all()] == ["我喜欢用 Go 写代码。"]
    assert store.verify_forgotten(mem_id, "我喜欢用 Python 写代码。")["ok"] is False


# ---------- 忘掉的不复活：提取只对新轮次跑 ----------


def test_forgotten_memory_not_reextracted_by_unrelated_turns(tmp_path):
    """忘掉的记忆不会被无关的新轮次捞回来：删掉 Go 那条之后，
    再聊一轮不相关的话（“今天天气真好。我叫小宋。”），库里还是
    没有 Go——提取只处理传进来的新消息，不回头读旧对话、旧 history。
    但 BOOS 重新说起它（“我喜欢用 Go 写代码。”）时，会被重新记下来——
    忘是忘掉记录，不是禁止以后再提。"""
    store = make_fake_store(tmp_path)
    mem_id = _add(store, "我喜欢用 Go 写代码。")
    store.delete_memory(mem_id, approve=True)
    store.add_turn([{"role": "user", "content": "今天天气真好。我叫小宋。"}])
    assert not any("Go" in item["text"] for item in store.get_all())
    store.add_turn([{"role": "user", "content": "我喜欢用 Go 写代码。"}])
    # Plan C：上一轮先待定，Go 这轮在 flush 时才提交（上一轮的“我叫小宋”
    # 在 Go 这轮调 add_turn 时已提交，不会把删掉的 Go 捞回来）。
    store.flush_pending()
    assert any("Go" in item["text"] for item in store.get_all())
