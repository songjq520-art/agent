# memory/__init__.py（记忆包）
# 作用：让 memory/ 成为一个 Python 包，和 providers/ 一样，
#       这样 main.py 可以用 from memory.sqlite_store import SQLiteStore 来用它。
# 注意：v1 的 MemoryStore（memory/memories.py）和 ProposalStore
#       （memory/proposals.py）已删；mem0 适配层（memory/mem0_store.py）
#       也已于 2026-11-24 Step 2 彻底删除。长期记忆现在只走
#       memory/store_factory.py → memory/sqlite_store.py
#       （账本 memory_events.jsonl 是真相源，SQLite 是投影），
#       不要再引旧模块。
