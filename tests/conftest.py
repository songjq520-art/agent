# tests/conftest.py（mem0 测试共用的假对象）
# 作用：给所有 mem0 相关测试提供一个 FakeMemory（模拟 mem0 2.2.1 的
#       Memory 对象）和 make_fake_store（造一个注了假 memory 的 Mem0Store）。
# 测试一律用假对象，不碰真实 mem0、不联网、不产生费用。
# 注意：FakeMemory 只是“够测流程”的近似，不是真实 mem0——差异见文件末尾。

import re
import tempfile
import threading
from pathlib import Path

from memory.mem0_store import Mem0Store

# _CJK_RUN_RE：一段连续的中文字符。
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _windows(text):
    """把一段文本拆成“两字窗口 + 英文整词”的集合（供 search 打分用）。

    中文按连续中文段切两字窗口（和 mem0_store 里关键词补充信号的拆法一致）；
    英文/数字按整词（小写）算，不拆两字母。
    """
    out = set()
    for run in _CJK_RUN_RE.findall(text or ""):
        for i in range(len(run) - 1):
            out.add(run[i : i + 2])
    for word in _ASCII_WORD_RE.findall(text or ""):
        out.add(word.lower())
    return out


class FakeMemory:
    """模拟 mem0 2.2.1 的 Memory 对象（离线假对象）。

    只实现 Mem0Store 用到的几个方法，签名和返回形状和真实 mem0 对齐：
      - add(messages, user_id=None, infer=True, metadata=None) -> {"results": [{"id", "memory", "event"}]}
      - search(query, filters=None, top_k=5) -> {"results": [{"id", "memory", "score", "metadata"}]}
      - get_all(filters=None) -> {"results": [{"id", "memory", "metadata"}]}
      - update(memory_id, text=None, metadata=None) -> {"message": "Memory updated successfully!"}
      - delete(memory_id) -> {"deleted": True}
      - history(memory_id) -> [审计条目]（ADD / UPDATE / DELETE 事件）
      - purge_history(memory_id) -> 删掉的审计行数（适配器做深度删除时调，
        对应真实 mem0 里直接清 history.db 里该 memory_id 的行——真实
        mem0 没有这个方法，见 _purge_history_rows）
    每条记忆都带一份 metadata（来源/时间/置信度/是否用户确认），
    add 时传进来的 metadata 原样存下来，search/get_all 原样返回；
    没传 metadata 就存 None（模拟 2026-11-09 之前存的旧记忆）。
    另有测试辅助属性：
      - add_calls：每次 add() 的 (messages, user_id, infer) 记录，
        供测试断言“某流程确实调了 add / 没调 add”。
      - all_texts()：当前所有记忆文本的列表（按存入顺序）。

    提取规则（infer=True 时，只从 user 消息里提，不碰 assistant 的话）：
    把 user 消息按句号/问号/感叹号/换行拆句，每句满足以下条件就存一条：
      - 以“我是 / 我叫 / 我在 / 我喜欢 / 我的 / 我希望 / 我目前 / 我现在”
        其中之一开头——都是用户明确说的、关于自己的现在时事实或偏好；
      - 不以“如果 / 假设 / 假如 / 要是 / 万一”开头——条件句、假设句
        不是既定事实，不存。
    任务原文只列了前五个前缀；“我希望”是为了 Test 2 第 4 句
    （“我希望 AI 和英语帮助我实现这个目标。”），“我目前 / 我现在”
    是为了 Test 3-A（“我目前在圆通工作，做快递员。”）——这两句
    都是用户明确说的现在时事实，按规则应当存，见
    tests/test_mem0_regression.py。
    其余句子（小 talk、条件句、假设句、问句）一律不存。
    """

    # 用户明确说的、关于自己的现在时事实/偏好，常以这些词开头。
    _EXTRACT_PREFIXES = (
        "我是",
        "我叫",
        "我在",
        "我喜欢",
        "我的",
        "我希望",
        "我目前",
        "我现在",
    )
    # 条件/假设句的开头：不是既定事实，不存。
    _CONDITIONAL_LEADS = ("如果", "假设", "假如", "要是", "万一")

    def __init__(self):
        self._lock = threading.Lock()
        self._items = []  # 每项 {"id", "memory", "metadata"}，按存入顺序
        self._history = {}  # id -> [审计条目]
        self._next_id = 1
        self.add_calls = []  # 每次 add() 的 (messages, user_id, infer)
        self.update_calls = []  # 每次 update() 的 (memory_id, text, metadata) 记录

    # ---------- 内部工具 ----------

    def _store(self, text, metadata=None):
        """存一条记忆文本，返回新 id。调用方需已持有锁。

        metadata 原样存下来（不做校验——适配器负责拼好它）；
        没传就是 None，模拟旧记忆“没有来源信息”的样子。
        """
        mem_id = f"fake-{self._next_id}"
        self._next_id += 1
        self._items.append(
            {"id": mem_id, "memory": text, "metadata": dict(metadata) if metadata else None}
        )
        self._history.setdefault(mem_id, []).append({"event": "ADD", "memory": text})
        return mem_id

    @classmethod
    def _extract_sentences(cls, content):
        """从一段 user 消息里按提取规则提句子（见类 docstring）。"""
        sentences = re.split(r"[。！？\n]+", content or "")
        out = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence.startswith(cls._CONDITIONAL_LEADS):
                continue
            if sentence.startswith(cls._EXTRACT_PREFIXES):
                out.append(sentence)
        return out

    # ---------- mem0 Memory 接口 ----------

    def add(self, messages, user_id=None, infer=True, metadata=None):
        """存记忆。infer=False 时把消息内容原文各存一条；infer=True 时
        按上面的提取规则从 user 消息里提事实句再存。

        metadata（有就原样存到每条新记忆上，没传就存 None）：
        真实 mem0 2.2.1 的 add 接受 metadata 字典，提取出的每条记忆
        都带上它，search 再把它原样返回（见 mem0_store 的来源说明）。
        """
        with self._lock:
            self.add_calls.append((list(messages or []), user_id, infer))
            stored = []
            for message in messages or []:
                content = (message or {}).get("content") or ""
                if not infer:
                    # 原文存入：infer=False 不提取、不改写，内容是什么就存什么。
                    if content.strip():
                        mem_id = self._store(content.strip(), metadata=metadata)
                        stored.append({"id": mem_id, "memory": content.strip(), "event": "ADD"})
                    continue
                if (message or {}).get("role") != "user":
                    # assistant 的话不是用户事实，不提取（CUSTOM_INSTRUCTIONS
                    # 第 3 条：不要把模型的建议/判断存成用户事实）。
                    continue
                for sentence in self._extract_sentences(content):
                    mem_id = self._store(sentence, metadata=metadata)
                    stored.append({"id": mem_id, "memory": sentence, "event": "ADD"})
            return {"results": stored}

    def search(self, query, filters=None, top_k=5):
        """在已存记忆里按“关键词重合度”打分，返回 0.0–1.0 的分数，按分数降序。

        分数 = query 的两字窗口与某条记忆的两字窗口的重合数 / 该条记忆的
        窗口总数。分数低的也照常返回（真实 mem0 也是按相似度返回 top_k，
        再由 Mem0Store 按阈值过滤）；没有任何记忆时返回空结果。
        """
        with self._lock:
            query_windows = _windows(query)
            scored = []
            for item in self._items:
                mem_windows = _windows(item["memory"])
                denom = max(1, len(mem_windows))
                hits = len(query_windows & mem_windows)
                scored.append(
                    {
                        "id": item["id"],
                        "memory": item["memory"],
                        "score": hits / denom,
                        "metadata": dict(item["metadata"]) if item.get("metadata") else None,
                    }
                )
            scored.sort(key=lambda entry: entry["score"], reverse=True)
            return {"results": scored[:top_k]}

    def get_all(self, filters=None):
        """返回全部已存记忆（id + 文本 + metadata）。"""
        with self._lock:
            return {
                "results": [
                    {
                        "id": item["id"],
                        "memory": item["memory"],
                        "metadata": dict(item["metadata"]) if item.get("metadata") else None,
                    }
                    for item in self._items
                ]
            }

    def update(self, memory_id, text=None, metadata=None):
        """改一条记忆的文本（和/或 metadata），保留同一 id。

        行为对齐真实 mem0 2.2.1（试跑 Test 5 实测）：改完检索到的是
        新文本；history 里多一条 UPDATE 审计（old_memory 是旧文本，
        new_memory 是新文本）；找不到的 id 抛 ValueError。
        metadata 传了就替换整份（不是合并），不传就保留旧的。
        """
        with self._lock:
            self.update_calls.append((memory_id, text, metadata))
            for item in self._items:
                if item["id"] == memory_id:
                    old_text = item["memory"]
                    new_text = text if text is not None else old_text
                    item["memory"] = new_text
                    if metadata is not None:
                        item["metadata"] = dict(metadata)
                    self._history.setdefault(memory_id, []).append(
                        {
                            "event": "UPDATE",
                            "old_memory": old_text,
                            "new_memory": new_text,
                        }
                    )
                    return {"message": "Memory updated successfully!"}
            raise ValueError(f"Memory with id {memory_id} not found")

    def delete(self, memory_id):
        """删掉一条记忆（按 id）。

        对齐真实 mem0 2.2.1（试跑 Test 5、lifecycle 实测）：向量删掉、
        检索查不到，但 history 里留一条 DELETE 审计（old_memory 是
        被删的全文，is_deleted=True）——这正是深度删除要补的缺口：
        光调 delete()，文本还留在 history.db 里。所以适配器的
        delete_memory() 删完向量还要再清 history（见 purge_history
        和 mem0_store._purge_history_rows）。找不到的 id 抛 ValueError
        （真实 mem0 也是这样，见 Test 5 步骤 7c）；适配器在调之前
        先做归属校验，不会把不存在的 id 传进来。
        """
        with self._lock:
            for item in list(self._items):
                if item["id"] == memory_id:
                    self._items.remove(item)
                    self._history.setdefault(memory_id, []).append(
                        {"event": "DELETE", "old_memory": item["memory"], "is_deleted": True}
                    )
                    return {"deleted": True}
            raise ValueError(f"Memory with id {memory_id} not found")

    def purge_history(self, memory_id):
        """删掉一条记忆的全部审计行（模拟适配器清 history.db 的那一步）。

        真实 mem0 没有这个方法：真实的清法是直接对 history.db 的
        history 表 DELETE WHERE memory_id=?（见 mem0_store
        的 _purge_history_rows）。假对象没有 sqlite 文件，所以用
        这个方法模拟“审计行已被清掉”的结果，返回值是删掉的行数。
        """
        with self._lock:
            removed = len(self._history.pop(memory_id, []))
            return removed

    def history(self, memory_id):
        """返回一条记忆的审计记录（ADD / UPDATE / DELETE 事件）；没有时返回 []。"""
        with self._lock:
            return list(self._history.get(memory_id, []))

    # ---------- 测试辅助 ----------

    def all_texts(self):
        """当前所有记忆文本的列表（按存入顺序）。"""
        with self._lock:
            return [item["memory"] for item in self._items]


def make_fake_store(tmp_path=None, budget_tracker=None, threshold=0.4):
    """造一个注了 FakeMemory 的 Mem0Store（离线测试用）。

    Mem0Store 注入 memory 时跳过版本校验和真实构建，所以 tmp_path
    随便给也行；没传时用系统临时目录。返回的 store 和真实 Mem0Store
    同一个类，只是底下的 memory 是假的，行为（阈值过滤、关键词补充、
    写锁、迁移、删除两道关）走的都是真实代码。
    """
    base = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    return Mem0Store(
        memory=FakeMemory(),
        chroma_path=base / "mem0_chroma",
        history_db_path=base / "history.db",
        version_path=base / "mem0_versions.json",
        budget_tracker=budget_tracker,
        threshold=threshold,
    )


# ---------- FakeMemory 和真实 mem0 的行为差异（测试写死了这些，升级 mem0 时重看） ----------
# 1. 提取是规则不是模型：真实 mem0 调 DeepSeek 按 CUSTOM_INSTRUCTIONS 提取，
#    能理解语义、处理各种句式；FakeMemory 只认句首前缀（我是/我叫/…），
#    换个说法（比如“我叫小宋”写成“大家都叫我小宋”）它就提不出来。
#    所以这里的提取测试只证明“流程通了、规则文本没丢”，不证明真实提取得准；
#    真实提取质量靠 tests/test_mem0_regression.py 里默认跳过的真实 API 测试。
# 2. 检索是关键词重合不是向量相似：真实 mem0 用 bge 向量算语义相似度，
#    同义词、换说法也能召回；FakeMemory 只算两字窗口重合，换说法就召不回。
#    阈值过滤、关键词补充这些“适配器逻辑”能被它测到，语义召回本身测不到。
# 3. 去重是假的：真实 mem0 提取时会看已有记忆、去重近重复；FakeMemory
#    不去重，同一句话 add 两次就存两条。所以并发测试只断言“不崩”，不断言去重。
# 4. 没有持久化：FakeMemory 只活在内存里，重启就没了；真实 mem0 落在
#    Chroma + history.db 里，重启还在。持久化靠真实 API 测试（试跑 Test 6）覆盖。
# 5. 没有用量：FakeMemory.add 不调 LLM、不产生 Token；用量包装测试用的
#    是另一个带假 LLM 客户端的对象（见 tests/test_mem0_store.py）。
# 6. history 记 ADD / UPDATE / DELETE 三种事件：真实 mem0 的 history.db
#    也记这三种（UPDATE 行留 old_memory，DELETE 行留 old_memory 全文、
#    is_deleted=True）；适配器的深度删除（delete_memory）就是要把这些
#    行清掉，假对象用 purge_history() 模拟那一步（真实 mem0 没有这个
#    方法，清法是直接对 history.db 做 DELETE，见 mem0_store）。
# 7. delete 找不到的 id 时抛 ValueError（对齐真实 mem0，见试跑 Test 5
#    步骤 7c）；适配器在调之前先做归属校验，不会把不存在的 id 传进来。
# 8. update / metadata 是假的“够测流程”的近似：真实 mem0 的 update
#    会重新算向量、写 UPDATE 审计行，metadata 是随记忆一起存进 Chroma
#    的；假对象只改内存里的文本和 metadata 字典，不算向量。
