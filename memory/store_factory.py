# memory/store_factory.py（记忆后端工厂，2026-11-24 3c）
# 网页服务和命令行共用一个入口选记忆后端：看环境变量
# SHIER_MEMORY_BACKEND（默认 mem0，和以前一样；sqlite 时走
# memory/sqlite_store.py，JSONL 账本仍是真相源）。
# 以前 web/server.py 自己看环境变量、main.py 写死 get_mem0_store，
# 两边各选各的——网页切了 sqlite、命令行还往 mem0 里存，两个库
# 分叉。选后端只在这里，别再各写一份。
# 两个入口对应两种原有语义，别混：
#   get_store()：命令行“记住……”用——mem0 是懒加载单例
#       （get_mem0_store，建好后一直复用）；
#   create_store()：网页服务启动用——mem0 每次新建
#       （create_mem0_store，服务启动时建一次）。
# sqlite 两边都是 create_sqlite_store（建好顺手对账）。
# 建不好都返回 None：调用方按“没有记忆也能聊”处理，和两个后端
# 各自工厂的同款语义。

import os


def _backend_name():
    return os.environ.get("SHIER_MEMORY_BACKEND", "mem0").strip().lower()


def get_store(budget_tracker=None):
    """按 SHIER_MEMORY_BACKEND 拿记忆库（懒加载语义，见模块头）。"""
    if _backend_name() == "sqlite":
        from memory.sqlite_store import create_sqlite_store

        return create_sqlite_store(budget_tracker)
    from memory.mem0_store import get_mem0_store

    return get_mem0_store(budget_tracker)


def create_store(budget_tracker=None):
    """按 SHIER_MEMORY_BACKEND 建记忆库（新建语义，见模块头）。"""
    if _backend_name() == "sqlite":
        from memory.sqlite_store import create_sqlite_store

        return create_sqlite_store(budget_tracker)
    from memory.mem0_store import create_mem0_store

    return create_mem0_store(budget_tracker)
