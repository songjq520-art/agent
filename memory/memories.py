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
import logging
import os
import re
import uuid
from datetime import date
from pathlib import Path

logger = logging.getLogger("agent")


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


# ---------- 记忆去重（正式记忆与新提议之间的去重） ----------
# 问题：已经保存的正式记忆，再次被模型提取出来时，会重复生成提议
# （待确认列表里又出现同一条），虽然正式记忆文件里不会重复写，
# 但用户每次都要再确认一次，体验差。
# 解决：创建提议前，同时检查待确认提议和已保存的正式记忆；内容完全相同
# 或仅有轻微措辞差异的，不重复生成提议；内容有实质区别的，仍允许生成。
# 注意：去重靠的是下面的相似度比较（归一化比对 + 关键词重合度），
# 不是扩大停用词表——停用词表（_MEMORY_DUP_STOPWORDS）与 main.py 的
# _STOPWORDS 完全一致，不扩大。

# _DUPLICATE_SIMILARITY_THRESHOLD（去重相似度阈值）：
# 候选记忆与已有记忆的关键词 Jaccard 相似度达到这个值，就认为是同一条
# （完全相同或仅有轻微措辞差异），不重复生成提议。
# 0.75 的意思是 75% 的关键词重合；内容有实质区别时（如关键数字、对象不同），
# 相似度通常远低于 0.75，不会被误拦；数字不同时直接判定为不同条（见下）。
_DUPLICATE_SIMILARITY_THRESHOLD = 0.75

# 去重用的停用词：与 main.py 的 _STOPWORDS 完全一致（中文通用词 + 英文通用词），
# 不扩大。只去掉“怎么/什么/多少/the/is”这类没信息量的词，不参与相似度计算。
_MEMORY_DUP_STOPWORDS = frozenset(
    [
        "怎么",
        "什么",
        "多少",
        "这个",
        "那个",
        "一下",
        "可以",
        "一个",
        "为什么",
        "怎么办",
        "是不是",
        "有没有",
        "请问",
        "谢谢",
        "the",
        "is",
        "are",
        "do",
        "does",
        "can",
        "could",
        "please",
        "tell",
    ]
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _strip_user_prefix(text):
    """去掉开头的“用户”前缀（模型提取时常加，如“用户每月预算 5 美元”）。

    “用户”本身不携带事实信息，去掉后比对更准确。
    只去掉开头的“用户”及后面的冒号/逗号/空白，不动中间的“用户”
    （比如“用户手册”里的“用户”是内容的一部分，但出现在开头时
    通常是模型加的前缀；为去重准确，这里仍去掉——“用户手册”这类
    记忆本身极少作为用户事实被提取）。
    """
    if not isinstance(text, str):
        return text
    return re.sub(r"^\s*用户\s*[：:，、]?\s*", "", text)


def _normalize_memory_text(text):
    """归一化记忆文本：去掉空白和常见标点，英文转小写。

    用于“完全相同或仅有标点/空白差异”的快速判断。
    比如“每月预算 5 美元。”和“每月预算5美元”归一化后完全相同。
    """
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[\s，。！？、；：\"\"''（）()\[\]【】《》<>·~～!?,.;:]", "", text)
    return cleaned.lower()


def _memory_text_keywords(text):
    """提取记忆文本的关键词（供去重时的相似度计算用）。

    规则（与 main.py 的 _extract_keywords 同款思路，但为去重做了两处调整）：
    - 按空白和标点分词；含中文的长词只拆成两字窗口，**不保留整词**——
      整词太具体，措辞稍有不同就对不上，会把“同一条”误判成“不同条”。
    - 数字（哪怕只有一位）也要保留——数字常常是关键区别
      （比如预算 5 美元和 10 美元，区别就在于数字）。
    - 英文按整词；去掉停用词。
    返回去重后的关键词集合（用于计算两条记忆的关键词重合度）。
    """
    parts = re.split(r"[^\w]+", text or "")
    keywords = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 数字（哪怕只有一位）也要保留——数字常常是关键区别。
        if part.isdigit():
            keywords.add(part)
            continue
        if len(part) < 2:
            continue
        if part.lower() in _MEMORY_DUP_STOPWORDS:
            continue
        # 含中文的长词：只拆成两字窗口，不保留整词。
        if len(part) > 2 and _CJK_RE.search(part):
            for i in range(len(part) - 1):
                word = part[i : i + 2]
                if word.lower() not in _MEMORY_DUP_STOPWORDS:
                    keywords.add(word)
        else:
            keywords.add(part)
    return keywords


def _memory_text_numbers(text):
    """提取记忆文本里的全部数字（整数序列，如 5、10、2026）。

    用途：两条记忆的数字集合不同时，直接判定为不同条——
    数字不同通常意味着实质区别（比如预算 5 美元和 10 美元、2026 年和 2027 年），
    即使关键词重合度很高，也不能当成同一条。
    """
    return set(re.findall(r"\d+", text or ""))


def is_duplicate_memory_text(candidate_text, existing_text, threshold=_DUPLICATE_SIMILARITY_THRESHOLD):
    """判断候选记忆与已有记忆是不是同一条（完全相同或仅有轻微措辞差异）。

    参数 candidate_text：候选记忆内容；existing_text：已有记忆内容；
    threshold：关键词 Jaccard 相似度阈值，默认 0.75。
    返回 True 表示是同一条（不应重复生成提议/不应重复保存），False 表示
    内容有实质区别（允许生成新提议）。
    判断分三步：
    1. 归一化比对：去掉空白、标点后完全相同 → 同一条。
    2. 数字集合不同 → 不是同一条（数字不同通常是实质区别，直接放行）。
    3. 关键词 Jaccard 相似度：关键词集合的交集/并集 ≥ 阈值 → 同一条。
       比如“每月预算 5 美元”与“每月预算 10 美元”，因数字不同直接判定
       为不同条；“我喜欢喝咖啡”与“我喜欢喝茶”相似度约 0.5，不会被误判。
    """
    if not isinstance(candidate_text, str) or not isinstance(existing_text, str):
        return False
    if not candidate_text.strip() or not existing_text.strip():
        return False
    # 先去掉开头的“用户”前缀（模型提取时常加，不携带事实信息）。
    candidate_text = _strip_user_prefix(candidate_text)
    existing_text = _strip_user_prefix(existing_text)
    # 1. 归一化后完全相同 → 同一条。
    if _normalize_memory_text(candidate_text) == _normalize_memory_text(existing_text):
        return True
    # 2. 数字集合不同 → 不是同一条（数字不同通常是实质区别）。
    if _memory_text_numbers(candidate_text) != _memory_text_numbers(existing_text):
        return False
    # 3. 关键词 Jaccard 相似度 ≥ 阈值 → 同一条。
    candidate_keywords = _memory_text_keywords(candidate_text)
    existing_keywords = _memory_text_keywords(existing_text)
    if not candidate_keywords or not existing_keywords:
        return False
    union = candidate_keywords | existing_keywords
    if not union:
        return False
    similarity = len(candidate_keywords & existing_keywords) / len(union)
    return similarity >= threshold


def find_duplicate_memory(text, memories, threshold=_DUPLICATE_SIMILARITY_THRESHOLD):
    """在已有记忆列表里找与给定内容重复的那一条。

    参数 text：候选记忆内容；memories：已有记忆列表（每项为记忆字典，
    至少有 "text" 字段）；threshold：相似度阈值，默认 0.75。
    返回：找到的第一条重复记忆字典（原对象，不复制），没找到返回 None。
    用途：创建提议前检查正式记忆、检查待确认提议时共用。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    for memory in memories or []:
        if not isinstance(memory, dict):
            continue
        existing_text = memory.get("text") or ""
        if is_duplicate_memory_text(text, existing_text, threshold):
            return memory
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
        原子写入：先写临时文件，写成功后再用 os.replace() 原子替换正式文件。
        这样写一半时程序突然退出或磁盘出错，原文件也不会只剩半截坏 JSON；
        临时文件没用上就删掉，不留垃圾。
        """
        if self.corrupted:
            raise RuntimeError(f"记忆文件已损坏，不保存：{self.corruption_reason}")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(memories, ensure_ascii=False, indent=2)
        tmp_path = self.file_path.with_name(
            self.file_path.name + f".tmp-{uuid.uuid4().hex[:8]}"
        )
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, self.file_path)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def save(self):
        """把当前记忆列表保存到 JSON 文件。"""
        self._write(self.memories)

    def add_memory(self, text, layers, source, mem_date=None):
        """存一条记忆（先问再存的“存”这一步）。

        调用前应该已经用 propose_memory() 拟好、并经 BOOS 确认过。
        存完自动保存到文件。返回存进去的那条记忆字典。

        去重：内容与已有记忆完全相同的，不重复写文件，直接返回已有记忆。
        这样同一条记忆确认两次、或模型两次提取出同一条时，文件里只留一条。

        一致性：先写磁盘（含新记忆），写成功了再更新内存；
        写失败时内存不变，和磁盘保持一致。
        """
        memory = propose_memory(text, layers, source, mem_date)
        if self.corrupted:
            raise RuntimeError(f"记忆文件已损坏，不能存：{self.corruption_reason}")
        for existing in self.memories:
            if existing["text"] == memory["text"]:
                logger.info("重复记忆：内容已在长期记忆里，不重复保存。")
                return existing
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

    原子初始化：三条记忆先拟好，再一次写进文件（一次原子写入）。
    写失败时一条也不存，文件仍是空的，下次启动可干净重试，
    不会留下“只存了一条/两条”的半初始化状态。
    以前是逐条调 add_memory() 写三次，中途失败会留半套记忆；
    现在三条一起写，要么全有，要么全无。

    参数 store：一个 MemoryStore。文件里已经有记忆时什么也不做，不重复写。
    返回：写进去的条数（0 或 3）。
    """
    if store.memories:
        return 0
    # 先把三条都拟好（校验不过会直接抛错，一条也不写）。
    memories = [
        propose_memory(item["text"], item["layers"], item["source"], item["date"])
        for item in FIRST_MEMORIES
    ]
    # 一次原子写入：_write() 本身就是临时文件 + os.replace()，
    # 写失败时抛错，store.memories 不变（仍是空），文件里什么也没存，
    # 下次启动可干净重试，不留半初始化状态。
    new_memories = store.memories + memories
    store._write(new_memories)
    store.memories = new_memories
    return len(memories)
