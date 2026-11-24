# memory/store_factory.py（记忆后端工厂）
# 网页服务和命令行共用一个入口拿记忆库。2026-11-24 Step 2 之后
# 只剩一条路：memory/sqlite_store.py（JSONL 账本是真相源，
# SQLite 是投影）。mem0 后端已彻底删除，不再有后端开关
# （SHIER_MEMORY_BACKEND 不再被读取）。
# 两个入口对应两种原有语义，别混：
#   get_store()：命令行“记住……”用——懒加载语义（要用时才建）；
#   create_store()：网页服务启动用——新建语义（启动时建一次）。
# sqlite 建库不贵，两个入口都直接新建并顺手对账（见
# create_sqlite_store）。建不好都返回 None：调用方按“没有记忆
# 也能聊”处理。


def get_store(budget_tracker=None):
    """拿记忆库（懒加载语义，见模块头）。"""
    from memory.sqlite_store import create_sqlite_store

    return create_sqlite_store(budget_tracker)


def create_store(budget_tracker=None):
    """建记忆库（新建语义，见模块头）。"""
    from memory.sqlite_store import create_sqlite_store

    return create_sqlite_store(budget_tracker)
