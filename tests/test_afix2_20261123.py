# tests/test_afix2_20261123.py（双重纠正护栏，2026-11-23）
# 全部离线（向量打桩，不加载 bge 模型、不调真 API、不花钱）。
# 背景：afix 重测 D/E 组实锤（账本 seq 560/561）：模型生成 T2 回复时先调
# update_memory 工具改了（武汉→长沙，seq 560），随后 save_turn 又调一次
# correct_memory——武汉已不在候选里，槽位直选只剩真记忆广州，于是把广州
# 顶了（seq 561，只隔 4.5 秒）；E 组同理，西安已被模型顶掉，save_turn 报
# not_found，回复丢了“改好了”。
# 修法：correct_memory 开头加“已纠正过”护栏——新说法已是活行、且它顶掉
# 旧说法的账本事件就是本轮（5 分钟内）的，说明纠正已完成过，不再另找目
# 标，直接回 updated，old_text 取账本里这行顶掉的旧说法。
# 只拦两种情况：(a) save_turn 的调用（old_fact == new_fact）；(b) 工具的
# 重复调用（旧说法已不在库里活着）。旧说法还活着时不拦——那是真纠正。

import pytest

from memory.sqlite_store import SQLiteStore
from web.server import _correction_reply_text


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _turn(store, text, turn_id):
    """一轮对话：用户说一句话，助手回一句“好的”。"""
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的"}],
        turn_id=turn_id,
    )


def test_guard_blocks_double_correction_save_turn(store):
    """D 组复现：模型工具先改武汉→长沙，save_turn 再调 correct_memory
    （长沙，长沙）时必须拦住，广州不能被顶。"""
    store.add_memory_text("用户住在广州", source="user")
    store.add_memory_text("我住在武汉", source="user")
    # 模型工具调用：原样匹配，武汉→长沙。
    first = store.correct_memory("我住在武汉", "我住在长沙",
                                 explicit_correction=True)
    assert first["status"] == "updated"
    assert first["old_text"] == "我住在武汉"
    assert _is_superseded(store, "我住在武汉")
    assert "我住在长沙" in _live_texts(store)
    # save_turn 的第二次调用：新说法已是活行、刚顶过武汉。
    second = store.correct_memory("我住在长沙", "我住在长沙",
                                  explicit_correction=True)
    assert second["status"] == "updated"
    assert second["memory_id"] == first["memory_id"]
    assert second["old_text"] == "我住在武汉"
    assert second["new_text"] == "我住在长沙"
    assert second["layer"] == "already"
    assert second["resolution"] == "already_corrected"
    # 真记忆广州必须还活着。
    assert "用户住在广州" in _live_texts(store)
    assert not _is_superseded(store, "用户住在广州")


def test_guard_blocks_duplicate_tool_call(store):
    """工具重复调用：correct_memory(武汉，长沙）调两次，第二次拦住。"""
    store.add_memory_text("用户住在广州", source="user")
    store.add_memory_text("我住在武汉", source="user")
    first = store.correct_memory("我住在武汉", "我住在长沙",
                                 explicit_correction=True)
    assert first["status"] == "updated"
    second = store.correct_memory("我住在武汉", "我住在长沙",
                                  explicit_correction=True)
    assert second["status"] == "updated"
    assert second["old_text"] == "我住在武汉"
    assert second["layer"] == "already"
    assert "用户住在广州" in _live_texts(store)


def test_guard_does_not_block_genuine_correction(store):
    """真纠正不拦：长沙早就在库里（刚顶过武汉），现在纠正郑州→长沙，
    旧说法郑州还活着，正常流程顶掉郑州、复用长沙。"""
    store.add_memory_text("我住在武汉", source="user")
    first = store.correct_memory("我住在武汉", "我住在长沙",
                                 explicit_correction=True)
    assert first["status"] == "updated"
    store.add_memory_text("我住在郑州", source="user")
    second = store.correct_memory("我住在郑州", "我住在长沙",
                                  explicit_correction=True)
    assert second["status"] == "updated"
    assert second["old_text"] == "我住在郑州"
    assert second["layer"] != "already"
    assert _is_superseded(store, "我住在郑州")
    assert "我住在长沙" in _live_texts(store)


def test_guard_does_not_block_when_new_fact_added_directly(store):
    """新说法是直接存的（没顶过谁、账本无 supersede 事件），不拦。"""
    store.add_memory_text("我住在长沙", source="user")
    store.add_memory_text("我住在郑州", source="user")
    out = store.correct_memory("我住在郑州", "我住在长沙",
                               explicit_correction=True)
    assert out["status"] == "updated"
    assert out["old_text"] == "我住在郑州"
    assert out["layer"] != "already"
    assert _is_superseded(store, "我住在郑州")


def test_guard_save_turn_end_to_end(store):
    """save_turn 端到端：T1 武汉 → 模型工具改武汉→长沙 → T2“错了，
    我住在长沙”——回复必须是“已经改好了：把‘我住在武汉’改成了
    ‘我住在长沙’”，广州还活着。"""
    store.add_memory_text("用户住在广州", source="user")
    _turn(store, "我住在武汉", "t1")
    assert "我住在武汉" in _live_texts(store)
    first = store.correct_memory("我住在武汉", "我住在长沙",
                                 explicit_correction=True)
    assert first["status"] == "updated"
    receipt = _turn(store, "错了，我住在长沙", "t2")
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated"
    assert correction.get("old_text") == "我住在武汉"
    reply = _correction_reply_text(correction)
    assert "已经改好了" in reply
    assert "我住在武汉" in reply
    assert "我住在长沙" in reply
    assert "用户住在广州" in _live_texts(store)
    assert "我住在长沙" in _live_texts(store)
