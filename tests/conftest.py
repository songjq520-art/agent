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
# ---------- 判官在场标记 requires_judge（2026-11-24 修②） ----------
#
# 有一批纠正类测试的前提是「语义判官在配置层可用」：判官没接好
# （没 Key 等）时，store 按 G7 规矩把程序路纠正挂成待问问题
# （pending_question），纠正不执行——这是生产行为本身没错，但这批
# 测试断言的是判官在场时的纠正结果，在无 Key 环境（体检副本、CI）
# 里只会假失败。给它们打 requires_judge 标记：判官未配置时正确
# 跳过并写明缺哪项配置；配置在场时必须真跑、坏了响亮失败。
# 判定只认「判官配置」这个事实（memory.semantic_judge
# .judge_available：只查接没接上/启用没/Key 在不在，不调模型、
# 不读 Key 值），不认「在不在副本里」这类环境嗅探。


def judge_skip_reason():
    """判官没配置好返回缺哪项（str），配置好了返回 None。"""
    try:
        from memory.semantic_judge import judge_available

        ok, reason = judge_available()
    except Exception as exc:  # 判官模块本身坏了也不能连累全套收集
        return f"判官可用性检查失败：{exc}"
    return None if ok else reason


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_judge: 需要语义判官已配置（Key 到位）；未配置时跳过并写明原因",
    )


@pytest.fixture(autouse=True)
def _skip_when_judge_unconfigured(request):
    if request.node.get_closest_marker("requires_judge") is not None:
        reason = judge_skip_reason()
        if reason is not None:
            pytest.skip(f"判官未配置（{reason}），本测试需要判官在场")
    yield
