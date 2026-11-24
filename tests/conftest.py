# tests/conftest.py（测试共用的离线记忆库构造）
# 2026-11-24 Step 2 之后：mem0 已彻底删除，原 FakeMemory / make_fake_store
# （模拟 mem0 的 Memory 对象、造 Mem0Store 的那套）随 mem0 一起退役删除。
# 现在只提供 make_sqlite_store：造一个离线 SQLiteStore（临时目录、
# 向量和判官都打桩——判官返回 None 即走程序降级路，不联网、不花钱），
# 供还需要真实记忆库的流程测试（对话、工具、网页层）使用。

import tempfile
from pathlib import Path

import pytest

from memory.sqlite_store import SQLiteStore

# 本工厂造过的 store 登记表：见下方 autouse 夹具的说明。
_CREATED_STORES = []


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
    # 整理路由（语气判读走它）也关掉：不关的话，落账后的异步语气判读会
    # 真调模型，还可能漏进下一个测试的网络计数里（2026-11-24 全套实测）。
    store.organizing_router = lambda: None
    _CREATED_STORES.append(store)
    return store


@pytest.fixture(autouse=True)
def _drain_turn_organize_threads():
    """等本工厂造的 store 的后台整理线程在测试结束时收尾。

    SQLiteStore 每次落账会起一条后台“实时整理”线程，线程里另开一个
    store 实例干活（语气判读等），实例上的打桩管不到它。线程要是拖到
    下一个测试里，它调模型会被下一个测试的网络 mock 数进去
    （2026-11-24 全套实测：test_run_question_injects_summary_* 的整理
    线程漏进了 test_chat_loop_exit_flushes_* 的 API 计数）。这里在每个
    测试收尾时把还活着的整理线程 join 掉，让异步工作落在自己的测试里。
    """
    yield
    for store in _CREATED_STORES:
        thread = getattr(store, "_turn_organize_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=60)
    _CREATED_STORES.clear()
