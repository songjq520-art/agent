# memory/memories.py（长期记忆存储模块）
# 作用：把记忆存在本地 JSON 文件里，支持读写、检索、先问再存。
# 原理：记忆存在项目根目录下的 memory/memories.json（JSON 格式），
#       这个文件会被 .gitignore 忽略，不上传到 GitHub（仓库是公开的，记忆只保存在本地）。
#       每条记忆有 text（内容）、date（日期，YYYY-MM-DD）、layers（属于哪几层）、
#       source（来源：user 用户明确说的 / inferred Agent 推断的 / unconfirmed 尚未确认的）。
#       先问再存：propose_memory() 只拟一条候选记忆，不保存；add_memory() 才是真正存进去。
#       检索优先：search() 用关键词搜，找到相关的再回答，不把全部记忆塞进上下文。
# 关系：main.py 以后会用到它（问问题前先搜记忆）；现在先把读写、检索做可靠，
#       向量数据库、知识图谱这些复杂的以后再说。先把记忆做准，再把记忆做大。
# 安全规则：记忆文件损坏、无法解析、或里面任何一条格式异常时，
#       都进入安全状态（corrupted=True），原文件会被保留、不被覆盖，
#       只能人工检查修复后才能恢复；保存失败时内存状态不变，和磁盘保持一致。
#       这和 config.py 里 BudgetTracker 对预算文件的处理是一样的思路。

import copy
import json
from datetime import date
from pathlib import Path


# 记忆分层（五层，用来判断什么值得记）。
#   core：核心记忆（你是谁、长期目标、重要偏好）
#   experience：经历记忆（重要事件、做过的决定、经验教训）
#   working：工作记忆（当前项目进度、待办事项、最近对话）——注意：工作记忆是临时的，
#             一般不进长期记忆文件，这里列出来是为了分类完整。
#   knowledge：知识记忆（学到的知识、技术资料、解决方案）
#   relationship：关系记忆（相处习惯、沟通方式、重要约定）
LAYERS = ("core", "experience", "working", "knowledge", "relationship")

# 记忆来源（三档，必须分开标，不能把猜测写成事实）。
#   user：用户明确说过的（可信度最高）
#   inferred：Agent 推断的（要标清楚是推测）
#   unconfirmed：尚未确认的（还没经用户点头）
SOURCES = ("user", "inferred", "unconfirmed")


# 首批三条记忆（2026-11-07，BOOS 确认过）。
# 注意：这三条是 Agent 项目本身的事实，已经在 PROJECT_PRINCIPLES.md 里公开过，
#       所以写进代码里没问题。真正的私人记忆存在 memories.json 里，只保存在本地，不上传。
FIRST_MEMORIES = [
    {
        "text": "DeepSeek 已接通（2026-11-07），真实调用成功，是第一任主控。",
        "layers": ["experience", "core"],
        "source": "user",
        "date": "2026-11-07",
    },
    {
        "text": "连续对话已做好，进 main 了（PR #4、PR #5），多轮能记住上下文，能退出。",
        "layers": ["experience"],
        "source": "user",
        "date": "2026-11-07",
    },
    {
        "text": "每月预算 5 美元，4 美元提醒，5 美元停。",
        "layers": ["core"],
        "source": "user",
        "date": "2026-11-07",
    },
]


def _today():
    """今天的日期，格式 YYYY-MM-DD，例如 2026-11-07。"""
    return date.today().isoformat()


def _check_date(date_str):
    """检查日期格式：必须是 YYYY-MM-DD 的真实日期（例如 2026-11-07）。

    不合法就报 ValueError。记忆要做准，日期不能是假的。
    """
    if not isinstance(date_str, str):
        raise ValueError("日期必须是字符串")
    try:
        date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"日期格式不对：{date_str!r}，要是 YYYY-MM-DD 的真实日期")
    # fromisoformat 在 Python 3.11+ 还会接受 YYYYMMDD 这种写法，再卡一道格式。
    if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
        raise ValueError(f"日期格式不对：{date_str!r}，要是 YYYY-MM-DD 的真实日期")


def _check_layers(layers):
    """检查 layers 是否合法：必须是非空列表，里面每一项都得是 LAYERS 里的五层之一。

    不合法就报 ValueError，告诉调用方哪里错了，不悄悄存错东西。
    """
    if not isinstance(layers, list) or not layers:
        raise ValueError("layers 必须是非空列表，例如 ['core']")
    for layer in layers:
        if layer not in LAYERS:
            raise ValueError(f"layers 里不能有 {layer!r}，只能是 {LAYERS} 里的层")


def _check_source(source):
    """检查 source 是否合法：必须是 SOURCES 里的三档之一。

    不合法就报 ValueError。来源必须分开标，不能把猜测写成事实。
    """
    if source not in SOURCES:
        raise ValueError(f"source 必须是 {SOURCES} 里的一档，不能是 {source!r}")


def _entry_problem(item):
    """检查一条记忆字典的格式。格式正确返回 None；不对返回中文原因。

    用途：读文件时逐条校验，只要有一条不对，整个文件就进安全状态。
    """
    if not isinstance(item, dict):
        return "不是一个字典"
    if not isinstance(item.get("text"), str) or not item["text"].strip():
        return "内容不是非空字符串"
    if not isinstance(item.get("date"), str):
        return "日期不是字符串"
    try:
        _check_date(item["date"])
    except ValueError:
        return f"日期格式不对（{item['date']!r}，要是 YYYY-MM-DD 的真实日期）"
    if not isinstance(item.get("layers"), list):
        return "layers 不是列表"
    try:
        _check_layers(item["layers"])
    except ValueError as exc:
        return f"layers 不对：{exc}"
    if item.get("source") not in SOURCES:
        return f"source 不对（{item.get('source')!r}，只能是 {SOURCES} 里的档）"
    return None


def propose_memory(text, layers, source, mem_date=None):
    """拟一条候选记忆（先问再存的“拟”这一步），不保存，只返回记忆字典。

    参数 text：记忆内容，不能是空字符串。
    参数 layers：属于哪几层，例如 ["core"] 或 ["experience", "core"]。
    参数 source：来源，三档之一（user / inferred / unconfirmed）。
    参数 mem_date：日期，不传时用今天；传了就要是 YYYY-MM-DD 的真实日期。

    返回：一条记忆字典，有 text、date、layers、source 四个字段。
    这条记忆还没存进文件，等 BOOS 点头确认后，再用 add_memory() 存进去。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text 必须是非空字符串")
    _check_layers(layers)
    _check_source(source)
    mem_date = mem_date or _today()
    _check_date(mem_date)
    return {
        "text": text.strip(),
        "date": mem_date,
        "layers": list(layers),
        "source": source,
    }


class MemoryStore:
    """长期记忆存储（简单版，先做准，再做大）。

    作用：把记忆存在本地 JSON 文件里，支持读写、检索、先问再存。
    原理：记忆存在 memory/memories.json（JSON 格式），这个文件会被 .gitignore 忽略，
          不上传到 GitHub。文件里就是一个列表，每一项是一条记忆字典。
          文件不存在时当作空列表；文件损坏、无法解析、或里面任何一条格式异常时，
          都进入安全状态（corrupted=True），原文件保留、不被覆盖，
          只能人工检查修复后才能恢复。
    注意：工作记忆（当前项目进度、待办、最近对话）是临时的，一般不进这个文件；
          这个文件存的是长期记忆（核心、经历、知识、关系四层）。
    """

    def __init__(self, file_path=None):
        # 默认存在项目根目录下的 memory/memories.json
        self.file_path = (
            Path(file_path)
            if file_path
            else Path(__file__).parent / "memories.json"
        )
        self.memories = []  # 记忆列表，每一项是一条记忆字典
        self.corrupted = False  # 记忆文件是否损坏/异常，损坏时不自动覆盖
        self.corruption_reason = ""  # 损坏原因，中文简短说明
        self._load()

    def _load(self):
        """读取记忆文件。

        文件不存在时当作空列表；损坏、无法解析、或里面任何一条格式异常时，
        都进安全状态（corrupted=True），不加载、不覆盖原文件。
        原因：跳过坏条目再保存会永久丢掉它；必须保留原文件，等人工检查修复。
        """
        try:
            raw = self.file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.memories = []
            return
        except OSError as exc:
            self.corrupted = True
            self.corruption_reason = f"记忆文件读不出来：{exc}"
            self.memories = []
            return
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self.corrupted = True
            self.corruption_reason = f"记忆文件损坏，解析失败：{exc}"
            self.memories = []
            return
        if not isinstance(data, list):
            self.corrupted = True
            self.corruption_reason = "记忆文件损坏：里面不是一个列表"
            self.memories = []
            return
        # 逐条校验：只要发现任何一条格式异常，就进安全状态，不加载、不覆盖。
        for index, item in enumerate(data):
            reason = _entry_problem(item)
            if reason:
                self.corrupted = True
                self.corruption_reason = f"记忆文件损坏：第 {index + 1} 条记忆{reason}"
                self.memories = []
                return
        self.memories = data

    def _write(self, memories):
        """把给定的记忆列表写到 JSON 文件（不改内存状态）。

        安全规则：文件损坏（corrupted=True）时不写，等人工检查修复。
        """
        if self.corrupted:
            raise RuntimeError(f"记忆文件已损坏，不保存：{self.corruption_reason}")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save(self):
        """把当前记忆列表保存到 JSON 文件。"""
        self._write(self.memories)

    def add_memory(self, text, layers, source, mem_date=None):
        """存一条记忆（先问再存的“存”这一步）。

        调用前应该已经用 propose_memory() 拟好、并经 BOOS 确认过。
        存完自动保存到文件。返回存进去的那条记忆字典。

        一致性：先写磁盘（含新记忆），写成功了再更新内存；
        写失败时内存不变，和磁盘保持一致。
        """
        memory = propose_memory(text, layers, source, mem_date)
        if self.corrupted:
            raise RuntimeError(f"记忆文件已损坏，不能存：{self.corruption_reason}")
        new_memories = self.memories + [memory]
        self._write(new_memories)
        self.memories = new_memories
        return memory

    def add_proposed(self, memory):
        """把一条已经拟好（propose_memory 返回）的候选记忆存进去。

        用途：先拟、后确认、再存的流程里，确认后调这个存。同样自动保存到文件。
        """
        if not isinstance(memory, dict):
            raise ValueError("memory 必须是一条记忆字典")
        return self.add_memory(
            memory["text"], memory["layers"], memory["source"], memory.get("date")
        )

    def search(self, keyword):
        """关键词检索：返回内容里包含关键词的记忆（不区分大小写）。

        检索优先：回答问题前先搜记忆，找到相关的再回答，不把全部记忆塞进上下文。
        一开始关键词搜就够了，向量搜索以后再说。

        返回的是深拷贝，改它不影响存着的记忆。
        """
        if not isinstance(keyword, str) or not keyword.strip():
            return []
        word = keyword.strip().lower()
        return [
            copy.deepcopy(m) for m in self.memories if word in m["text"].lower()
        ]

    def list_all(self):
        """返回全部记忆（返回深拷贝，改它不影响存着的记忆）。"""
        return copy.deepcopy(self.memories)


def ensure_first_memories(store):
    """如果记忆文件是空的，把首批三条记忆写进去（只在第一次时做）。

    参数 store：一个 MemoryStore。文件里已经有记忆时什么也不做，不重复写。
    返回：写进去的条数（0 或 3）。
    """
    if store.memories:
        return 0
    for item in FIRST_MEMORIES:
        store.add_memory(
            item["text"], item["layers"], item["source"], item["date"]
        )
    return len(FIRST_MEMORIES)
