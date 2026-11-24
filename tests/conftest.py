# tests/conftest.py（测试共用的离线记忆库构造）
# 2026-11-24 Step 2 之后：mem0 已彻底删除，原 FakeMemory / make_fake_store
# （模拟 mem0 的 Memory 对象、造 Mem0Store 的那套）随 mem0 一起退役删除。
# 现在只提供 make_sqlite_store：造一个离线 SQLiteStore（临时目录、
# 向量和判官都打桩——判官返回 None 即走程序降级路，不联网、不花钱），
# 供还需要真实记忆库的流程测试（对话、工具、网页层）使用。

import tempfile
from pathlib import Path

from memory.sqlite_store import SQLiteStore


def make_sqlite_store(tmp_path=None, budget_tracker=None):
    """造一个离线 SQLiteStore（向量/判官打桩，见模块头）。

    tmp_path 没传时用系统临时目录；每个 store 一份自己的账本和
    SQLite 文件，互不污染。
    """
    base = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = SQLiteStore(
        memory_dir=str(base / "memory"), budget_tracker=budget_tracker
    )
    store._embed_text = lambda text: None
    store._ask_semantic_judge = lambda text, candidates, context=None: None
    return store
