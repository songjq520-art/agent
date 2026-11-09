# memory/__init__.py（记忆包）
# 作用：让 memory/ 成为一个 Python 包，和 providers/ 一样，
#       这样 main.py 可以用 from memory.mem0_store import Mem0Store 来用它。
# 注意：v1 的 MemoryStore（memory/memories.py）和 ProposalStore
#       （memory/proposals.py）已删，长期记忆现在走 mem0
#       （memory/mem0_store.py），不要再引旧模块。
