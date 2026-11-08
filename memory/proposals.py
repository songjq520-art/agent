# memory/proposals.py（记忆提议管理模块）
# 作用：管理模型从对话中提取出的待确认记忆提议。
# 原理：提议是“建议保存”的候选，不是记忆本身——只有用户确认后，
#       才调 MemoryStore.add_proposed() 写进长期记忆文件（memory/memories.json）。
#       提议只存在内存里（ProposalStore 的字典），不写文件；重启服务后，
#       未确认的提议会丢弃（v1 不做提议持久化），已确认的记忆不受影响。
# 安全规则（重要）：模型生成的文字绝不能直接被当成真实提议。
#       所有提议必须由后端创建和管理：提取出的候选要经过解析、校验，
#       再由 create_proposal() 生成真实 ID 和 pending 状态；模型文字本身
#       从不直接进记忆文件。确认时也必须经过 ProposalStore.confirm()，
#       由后端写文件，不接受前端直接提交的记忆内容。
# 关系：main.py 的提取函数产出候选，调 ProposalStore.add() 建提议；
#       web/server.py 的 ChatState 拿着一个 ProposalStore，给页面提供
#       提议列表、确认、拒绝三个接口。

import copy
import logging
import threading
import uuid
from datetime import date

from .memories import find_duplicate_memory, propose_memory

logger = logging.getLogger("agent")

# 提议状态（三档）。
#   pending：待确认（刚创建，等用户在网页上确认或拒绝）
#   confirmed：已确认（已写进长期记忆，提议随即从待确认列表移除）
#   rejected：已拒绝（不保存，提议随即从待确认列表移除）
PROPOSAL_STATUS_PENDING = "pending"
PROPOSAL_STATUS_CONFIRMED = "confirmed"
PROPOSAL_STATUS_REJECTED = "rejected"


def _today():
    """今天的日期，格式 YYYY-MM-DD，例如 2026-11-08。"""
    return date.today().isoformat()


def _new_proposal_id():
    """生成一个提议 ID：prop- + 8 位随机十六进制。

    每个提议的 ID 都不一样，靠 uuid 保证不重复；ID 本身不含记忆内容，
    记进日志也不会泄露隐私。
    """
    return "prop-" + uuid.uuid4().hex[:8]


def create_proposal(text, layers, source, reason):
    """创建一个记忆提议对象（只创建、不保存到任何文件）。

    参数 text：记忆内容，不能是空字符串。
    参数 layers：属于哪几层，例如 ["core"] 或 ["experience"]。
    参数 source：来源，三档之一（user / inferred / unconfirmed）；
        模型提取出的提议固定为 "inferred"（是 Agent 推断值得记的）。
    参数 reason：为什么值得保存（模型给的理由，供用户判断用，可空）。

    返回：提议字典，有 id、text、layers、source、reason、status、date。
    校验：先调 propose_memory() 校验（text 非空、layers 合法、source 合法），
    不合法就报 ValueError，不建提议。ID 由后端生成，不是模型给的。
    注意：提议不是记忆——它只是“建议保存”的候选，必须经用户确认后
    再调 MemoryStore.add_proposed() 存进 memories.json；未经确认的提议
    不进长期记忆文件。
    """
    memory = propose_memory(text, layers, source)
    return {
        "id": _new_proposal_id(),
        "text": memory["text"],
        "layers": memory["layers"],
        "source": memory["source"],
        "reason": (reason or "").strip() if isinstance(reason, str) else "",
        "status": PROPOSAL_STATUS_PENDING,
        "date": _today(),
    }


class ProposalStore:
    """待确认记忆提议的暂存（只在内存里，不写文件）。

    作用：保存模型提取出的候选记忆提议，等用户在网页上确认或拒绝。
    重启服务后，未确认的提议会丢弃（v1 不做提议持久化）；已确认的记忆
    已经写进 memories.json，不受影响。

    线程安全：网页服务是 ThreadingHTTPServer，多个聊天请求可能并发执行。
    本类用一把可重入锁（RLock）保护所有读写操作：add() 的“查重 + 创建”
    是原子操作，两个请求同时检查时不会都发现“没有重复”然后各自创建一条；
    confirm()/reject()/list_pending()/get() 也在同一把锁内，与 add() 并发时
    状态一致。用 RLock 而不用 Lock，是因为 confirm() 等方法内部可能调
    用本类的其他方法（如 get()），同一线程重复加锁不会死锁。
    注意：这把锁只保护提议字典本身；ChatState 另有一把锁串行化整个聊天
    操作（模型调用、历史更新等），两把锁的加锁顺序始终是 ChatState → 
    ProposalStore，不会死锁。
    """

    def __init__(self):
        self._proposals = {}  # id -> 提议字典
        self._lock = threading.RLock()

    def add(self, text, layers, source, reason, memory_store=None):
        """创建一个提议并加入待确认列表。返回提议字典（深拷贝）。

        去重（两层）：
        1. 待确认提议：内容与已有待确认提议完全相同或仅有轻微措辞差异的，
           不重复创建，返回已有的那条（并记一条日志）。
        2. 正式记忆：memory_store 传了时，内容与已保存的正式记忆完全相同
           或仅有轻微措辞差异的，不生成提议，返回 None（并记一条日志）。
           这样已经保存过的记忆再次被提取时，不会又出现在待确认列表。
        内容有实质区别的（比如关键数字、对象不同），不受影响，照常创建。

        线程安全：查重和创建在同一把锁内完成，是原子操作；多个线程同时
        提交相同候选时，最终只会存在一条 Pending 提议。
        """
        clean_text = text.strip() if isinstance(text, str) else ""
        with self._lock:
            # 1. 先查正式记忆：已经保存过的，不再生成提议。
            if memory_store is not None:
                existing_memory = find_duplicate_memory(
                    clean_text, memory_store.list_all()
                )
                if existing_memory is not None:
                    logger.info(
                        "已保存过：内容与正式记忆重复，不生成提议。"
                    )
                    return None
            # 2. 再查待确认提议：内容重复的不重复创建，返回已有的那条。
            for existing in self._proposals.values():
                if existing["status"] != PROPOSAL_STATUS_PENDING:
                    continue
                if find_duplicate_memory(clean_text, [existing]) is not None:
                    logger.info(
                        "重复提议：id=%s 内容已在待确认列表，不重复创建。",
                        existing["id"],
                    )
                    return copy.deepcopy(existing)
            proposal = create_proposal(text, layers, source, reason)
            self._proposals[proposal["id"]] = proposal
            logger.info(
                "提议创建：id=%s status=pending layers=%s source=%s",
                proposal["id"],
                proposal["layers"],
                proposal["source"],
            )
            return copy.deepcopy(proposal)

    def list_pending(self):
        """返回所有待确认的提议（按创建时间顺序，返回深拷贝）。

        线程安全：在锁内读取字典，返回的是创建完成时的完整快照，
        不会读到一半被并发的 add()/confirm()/reject() 改掉。
        """
        with self._lock:
            return [
                copy.deepcopy(p)
                for p in self._proposals.values()
                if p["status"] == PROPOSAL_STATUS_PENDING
            ]

    def get(self, proposal_id):
        """按 ID 取一条提议（深拷贝），没有返回 None。

        线程安全：在锁内读取，与并发的 add()/confirm()/reject() 状态一致。
        """
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            return copy.deepcopy(proposal) if proposal is not None else None

    def confirm(self, proposal_id, memory_store):
        """确认一条提议：把记忆写进长期记忆文件，并从待确认列表移除。

        参数 proposal_id：提议 ID；memory_store：长期记忆库（MemoryStore）。
        返回：存进去的记忆字典（内容与已有记忆完全相同时，返回已有记忆，
        不重复写文件——去重在 MemoryStore.add_memory() 里做）。
        找不到提议时报 ValueError；提议已处理过时也报 ValueError。
        保存失败时报异常，提议保留在待确认列表（用户可稍后重试），
        并记一条保存失败日志。

        线程安全：查找、写文件、删除提议在同一把锁内完成；两个线程同时
        确认同一条提议时，只有一个能成功，另一个报 ValueError（提议已处理）。
        写文件期间持有锁，确认操作的磁盘写入也是串行的。
        """
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise ValueError(f"没有这个提议：{proposal_id}")
            if proposal["status"] != PROPOSAL_STATUS_PENDING:
                raise ValueError(
                    f"提议 {proposal_id} 已处理（{proposal['status']}），不能重复确认"
                )
            try:
                saved = memory_store.add_proposed(proposal)
            except Exception:
                logger.warning("提议保存失败：id=%s", proposal_id, exc_info=True)
                raise
            # 保存成功才移除（保存失败时提议保留，用户可重试）。
            del self._proposals[proposal_id]
            logger.info("提议确认：id=%s 已存进长期记忆。", proposal_id)
            return saved

    def reject(self, proposal_id):
        """拒绝一条提议：从待确认列表移除，不保存。返回 True；找不到返回 False。

        线程安全：在锁内删除，与并发的 add()/confirm()/list_pending() 状态一致。
        """
        with self._lock:
            proposal = self._proposals.pop(proposal_id, None)
            if proposal is None:
                return False
            logger.info("提议拒绝：id=%s 未保存。", proposal_id)
            return True
