# tests/test_turn_receipts_cap_20261124.py — 修复 2 回归：
# SQLiteStore._turn_receipts 只留最近 200 条，超了删插入顺序最旧的。
import pytest

from memory.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每个测试一个干净的 SQLiteStore，embedder 桩成 None，只走本地路。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def test_turn_receipts_capped_at_200(store):
    for i in range(250):
        store._store_turn_receipt(f"turn-{i}", {"turn_id": f"turn-{i}", "results": []})
    assert len(store._turn_receipts) == 200
    assert "turn-0" not in store._turn_receipts
    assert "turn-49" not in store._turn_receipts
    assert "turn-50" in store._turn_receipts
    assert "turn-249" in store._turn_receipts
    assert list(store._turn_receipts)[0] == "turn-50"
    assert list(store._turn_receipts)[-1] == "turn-249"


def test_turn_receipts_cap_via_save_turn(store):
    # 走真实 save_turn 路：空消息轮次也进缓存，250 轮后同样只剩 200 条。
    for i in range(250):
        store.save_turn([{"role": "user", "content": ""}], turn_id=f"empty-{i}")
    assert len(store._turn_receipts) == 200
    assert "empty-0" not in store._turn_receipts
    assert "empty-49" not in store._turn_receipts
    assert "empty-50" in store._turn_receipts
    assert "empty-249" in store._turn_receipts
