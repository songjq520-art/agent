# memory/mem0_store.py（mem0 长期记忆适配层）
# 作用：项目里唯一直接碰 mem0 的模块，其他代码（main.py、web/server.py）
#       都只调这里，不直接 import mem0。以后换记忆后端，只改这一个文件。
# 原理：mem0 负责三件事——每轮对话后自动提取记忆、存进本地 Chroma 向量库、
#       提问时语义检索。这里只做薄薄一层包装：配好 DeepSeek + 本地
#       embedding + 本地 Chroma、关掉遥测、加阈值过滤、加写锁、记账。
# 关系：main.py 的 run_question() 提问前调 search()、回答后调 add_turn()；
#       web/server.py 常驻服务里建一个 Mem0Store；命令行默认不建（见下）。
# 安全与隐私：
#   - 遥测默认是开的（mem0 会把使用情况发到 PostHog），这里在 import mem0
#     之前就把 MEM0_TELEMETRY 设成 False，彻底关掉，一点都不往外发。
#     注意顺序：mem0 在 import 时就读这个环境变量，建好 Memory 之后再关
#     就晚了，所以必须在 import 之前设。
#   - API Key 只从环境变量读（config.get_api_key，.env 里那个），绝不写进代码。
#   - 记忆数据只保存在本地（memory/mem0_chroma/ + memory/history.db），
#     .gitignore 已忽略，不上传。
#   - 删除记忆要过两道关：先校验这条记忆确实属于当前用户，再要 BOOS
#     明确同意（approve=True），缺一个都不删。
#
# 性能（试跑实测 2026-11-08）：
#   - 每个进程第一次加载 embedding 模型约 25 秒，第一次提取调用还特别慢
#     （12–46 秒），稳态后添加约 1.2–2.7 秒、检索约 0.09 秒。
#     所以 mem0 只在常驻的网页服务里启用；命令行单次问答默认不加载，
#     需要时（比如用户说“记住……”）才懒加载一次。

import json
import logging
import os
import re
import threading
from datetime import date
from pathlib import Path

logger = logging.getLogger("agent")

# ---------- 遥测开关（必须在 import mem0 之前设置） ----------
# mem0 的 telemetry.py 在 import 时读 MEM0_TELEMETRY，默认 True。
# 这里先设成 False，再 import mem0，确保整个进程里遥测都是关的。
# setdefault：用户自己显式设了 MEM0_TELEMETRY=1 时不覆盖（方便调试），
# 但项目正常使用时永远是 False（create_mem0_store 里也会再强制设一次）。
os.environ.setdefault("MEM0_TELEMETRY", "False")

# ---------- 版本与配置常量（钉死的版本号，升级 = 重新试跑） ----------
# mem0ai==2.2.1、chromadb==1.5.9、sentence-transformers==6.2.0、
# torch==2.15.0+cpu、transformers==5.22.2、openai==3.46.0
# （完整清单见 requirements.txt；升级时先重跑试跑，通过了再改这里）。
MEM0_VERSION = "2.2.1"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 本地 embedding 模型
EMBEDDING_DIMS = 512  # bge-small-zh-v1.5 的向量维度
LLM_MODEL = "deepseek-flash"  # 提取用的模型（DeepSeek V4.1 Flash）
LLM_TEMPERATURE = 0.1  # 提取要稳定，温度低一点
LLM_MAX_TOKENS = 4000  # 提取调用的输出上限
COLLECTION_NAME = "agent_memories"  # Chroma 里集合的名字
DEFAULT_USER_ID = "boos"  # 单用户 Agent，记忆都归 BOOS

# RETRIEVAL_THRESHOLD（检索得分阈值）：语义检索只取分数 ≥ 这个值的记忆。
# 为什么是 0.4 不是 0.5：试跑实测里答对的记忆分数在 0.40–0.66 之间，
# 比如“你记得我叫什么吗？”召回“小宋”那条只有 0.4793；阈值设 0.5 会把
# 这条正确的也滤掉，v1 那个 bug 就又回来了。答错的记忆分数在 0.39–0.57，
# 和答对的重叠很大，光靠分数分不开——所以阈值只是第一道关，
# 真正的“不知道就说不知道”靠提示块里的相关性要求（见 main.py 的
# _memory_block）和模型自己判断。阈值是个可调的常量，观测后可以再改。
RETRIEVAL_THRESHOLD = 0.4

# KEYWORD_MIN_WEIGHT（关键词补充信号的最低权重）：语义检索之外，
# 再用 v1 那套中文两字窗口关键词做补充。权重 = 匹配到的关键词长度之和，
# ≥ 2 才算命中（至少一个两字关键词）。补充信号只排在语义结果后面，
# 且不超过 top_k 总数；召回不够时再调，不够时再考虑融合。
KEYWORD_MIN_WEIGHT = 2

# CUSTOM_INSTRUCTIONS_VERSION（自定义提取规则的版本号）：
# 规则文本改了就改这个版本号，方便以后追踪“哪版规则存的记忆”。
CUSTOM_INSTRUCTIONS_VERSION = "2026-11-08-spike"

# CUSTOM_INSTRUCTIONS（自定义提取规则，BOOS 定的，最高优先级）：
# 原文照抄试跑验证过的版本（~/workspace/mem0-spike/spike_common.py）。
# mem0 2.2.1 里没有 custom_fact_extraction_prompt 这个字段，
# 自定义规则只能走 MemoryConfig 的 custom_instructions。
# 注意：这是提示词层面的服从，不是机制保证——换模型、换 mem0 版本、
# 改措辞都可能动摇它，所以 tests/test_mem0_regression.py 里有
# 对应的回归测试，每次升级 mem0 都要重跑。
CUSTOM_INSTRUCTIONS = """CUSTOM EXTRACTION RULES — HIGHEST PRIORITY, these override the general
"extract everything / when in doubt, extract" guidance above:

1. ONLY store information the USER explicitly stated about themselves. A fact
   must be something the user said, in their own words' meaning.
2. User-explicit long-term preferences, goals and hard constraints are the
   highest priority to keep (e.g. "I want...", "my long-term goal is...",
   "I use X as my daily model", "my name is ...", "I work at ...").
3. NEVER store an inference, guess, hypothesis, conditional or assumption as a
   settled user fact. In particular, DO NOT store:
   - conditionals: "if I learn Rust later, it might help" — do NOT store that
     the user learns Rust, nor that Rust is helpful for them;
   - hypotheticals: "suppose I work in Australia later" — do NOT store that
     the user works or will work in Australia;
   - small talk / transient states: "the weather is nice today" — do NOT store
     this as a long-term fact about the user;
   - the assistant's suggestions, recommendations or the model's own
     judgements — do NOT store them as user facts.
4. NEVER promote an inference to a user fact just because it seems important
   or memorable. If the user did not explicitly say it, do not write it.
5. Keep each memory atomic: one fact per item, do not merge distinct facts.
6. Preserve the original meaning exactly; do not add details the user did not
   give, and do not change the meaning by rephrasing.
7. If a message contains no user-explicit long-term fact, output an empty
   memory list for it.
8. LANGUAGE: write every extracted memory in the SAME LANGUAGE as the user's
   messages. Chinese input -> Chinese memory text. Preserve proper nouns,
   brand names and model names exactly as written."""

# FIRST_MEMORIES（首批三条记忆，2026-11-07 BOOS 确认过）：
# 从 memory/memories.py 原样搬过来，只改了一处：第 3 条的预算规则
# 按 2026-11-08 BOOS 改的新规矩写（“到 5 美元提醒，不自动停”），
# 旧的“4 美元提醒，5 美元停”已经作废。迁移旧 memories.json 时，
# 旧的第 3 条会被改成这条新文本（见 migrate_from_v1，迁移时会明确记日志）。
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
        "text": "每月预算 5 美元，到 5 美元提醒，不自动停。",
        "layers": ["core"],
        "source": "user",
        "date": "2026-11-08",
    },
]

# _OUTDATED_BUDGET_TEXT（已作废的旧预算记忆文本）：
# 旧 memories.json 里的第 3 条写的是这句，迁移时改成新规矩
# （FIRST_MEMORIES 里的第 3 条），并记一条日志说明改过。
# 只改这一句，其他记忆一律原文迁移，不改一个字。
_OUTDATED_BUDGET_TEXT = "每月预算 5 美元，4 美元提醒，5 美元停。"

# _WRITE_LOCK（全局写锁）：所有写记忆的操作（add_turn、add_memory_text、
# 迁移、删除）都走这一把锁，一次只让一个写。mem0 的去重依赖“提取时
# 看到的已有记忆”，并发写时互相看不见，会存出重复记忆（试跑测试 8
# 实测：4 个线程同时写同一句，存出 4 条近重复）。网页服务和命令行
# 共用同一把锁（模块级单例），跨进程不共享——同机多进程同时写仍需人工
# 避免，正常使用时只有一个服务进程在跑。
_WRITE_LOCK = threading.RLock()


def _today():
    """今天的日期，格式 YYYY-MM-DD。"""
    return date.today().isoformat()


# ---------- 关键词补充信号（从 v1 main.py 搬过来，只做补充用） ----------
# v1 的关键词检索很小、已测过，留作第二路信号：语义检索漏掉的
# 精确关键词（比如“圆通”“Kimi”这种专有名词），靠它兜底。
# 注意：它只是补充，排在语义结果后面；主召回仍是 mem0 的语义检索。

_STOPWORDS = frozenset(
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


def _extract_keywords(prompt, min_len=2):
    """把问题拆成关键词：按空白和标点分词；含中文的长词再拆成两字窗口。

    中文词常是两个字（“预算”“对话”），只按标点分会把“我的预算是多少”
    当成一个词，搜不到记忆；所以含中文的长词要拆成两字窗口才能命中。
    英文长词整词搜，不拆两字母（两字母片段噪音太大）。
    停用词（_STOPWORDS）直接去掉，不参与检索，减少误撞。
    返回去重后的关键词列表（保持出现顺序）。
    """
    parts = re.split(r"[^\w]+", prompt or "")
    keywords = []
    for part in parts:
        part = part.strip()
        if len(part) < min_len:
            continue
        keywords.append(part)
        if len(part) > min_len and _CJK_RE.search(part):
            for i in range(len(part) - 1):
                keywords.append(part[i : i + 2])
    seen = set()
    out = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            if word.lower() in _STOPWORDS:
                continue
            out.append(word)
    return out


def _keyword_weight(text, keywords):
    """给一段记忆文本打关键词权重（供补充信号用）。

    权重 = 匹配到的关键词长度之和：匹配的关键词越长、越多，权重越高。
    返回 (权重, 匹配到的关键词列表)。
    """
    lower = (text or "").lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    return sum(len(kw) for kw in matched), matched


# ---------- 版本文件（embedding 模型 + 维度记录） ----------
# 换 embedding 模型必须全量重嵌：维度不同时 Chroma 直接报硬错，
# 旧库不能混用。所以每次建库时，把用的模型名、维度、mem0 版本
# 写进版本文件；启动时先对一下，对不上就拒绝启动、不悄悄混用，
# 让用户走重嵌迁移（tools/reembed_memories.py），而不是等 Chroma 报错。

def _read_version_file(version_path):
    """读版本文件，读不到或格式不对时返回 None（当作“没有版本记录”）。"""
    try:
        data = json.loads(Path(version_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_version_file(version_path, embedding_model, embedding_dims, mem0_version):
    """写版本文件：模型名、维度、mem0 版本、写入日期。"""
    path = Path(version_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "embedding_dims": embedding_dims,
                "mem0_version": mem0_version,
                "custom_instructions_version": CUSTOM_INSTRUCTIONS_VERSION,
                "updated": _today(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _check_version_file(version_path, chroma_path, embedding_model, embedding_dims, mem0_version):
    """校验版本文件与当前配置是否一致。

    规则：
    - 版本文件不存在、且 Chroma 目录也不存在或为空：全新库，写版本文件，通过。
    - 版本文件不存在、但 Chroma 目录里已有数据：来历不明的旧库，拒绝启动
      （不知道它是用哪个模型嵌的，混用会坏数据）。
    - 版本文件存在、模型名或维度或 mem0 版本对不上：拒绝启动，提示走重嵌迁移。
    - 对得上：通过。
    拒绝时抛 RuntimeError，信息里写清楚原因和该怎么做。
    """
    version_path = Path(version_path)
    chroma_path = Path(chroma_path)
    recorded = _read_version_file(version_path)
    chroma_has_data = chroma_path.exists() and any(chroma_path.iterdir())
    if recorded is None:
        if chroma_has_data:
            raise RuntimeError(
                "记忆库版本不明：Chroma 目录里已有数据，但没有版本文件，"
                "不知道它是用哪个 embedding 模型建的，不能直接混用。"
                "请先备份 memory/，再用 tools/reembed_memories.py 重嵌，"
                "或删掉 memory/mem0_chroma 后重新迁移。"
            )
        _write_version_file(version_path, embedding_model, embedding_dims, mem0_version)
        return
    if (
        recorded.get("embedding_model") != embedding_model
        or recorded.get("embedding_dims") != embedding_dims
        or recorded.get("mem0_version") != mem0_version
    ):
        raise RuntimeError(
            "记忆库版本不匹配：记录的是 "
            f"{recorded.get('embedding_model')}（{recorded.get('embedding_dims')} 维，"
            f"mem0 {recorded.get('mem0_version')}），当前配置是 "
            f"{embedding_model}（{embedding_dims} 维，mem0 {mem0_version}）。"
            "换 embedding 模型必须全量重嵌，不能直接混用。"
            "请用 tools/reembed_memories.py 重嵌，或删掉 memory/mem0_chroma 后重新迁移。"
        )


class Mem0Store:
    """mem0 长期记忆的适配器（项目里唯一直接碰 mem0 的类）。

    用法：
      store = create_mem0_store(budget_tracker)  # 建好就用，建不好返回 None
      store.search("你记得我叫什么吗？")  # 提问前检索，返回过阈值的记忆
      store.add_turn(messages)  # 回答后自动存本轮对话（加全局写锁）
      store.add_memory_text("我喜欢喝咖啡。")  # CLI“记住：内容”手动存原文
      store.delete_memory(mem_id, approve=True)  # 删除：先校验归属，再要 BOOS 同意
    记忆库出问题（没 Key、mem0 没装、Chroma 目录坏了）时，建 store 失败
    返回 None，调用方照常聊天、不拦路（见 create_mem0_store）。
    测试时可传一个假的 memory 对象（FakeMemory），不碰真实 mem0。
    """

    def __init__(
        self,
        memory=None,
        chroma_path=None,
        history_db_path=None,
        version_path=None,
        user_id=DEFAULT_USER_ID,
        budget_tracker=None,
        threshold=RETRIEVAL_THRESHOLD,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_dims=EMBEDDING_DIMS,
        mem0_version=MEM0_VERSION,
    ):
        # 路径默认值：项目根目录下的 memory/（按本文件位置算，不依赖当前工作目录）。
        project_root = Path(__file__).resolve().parent.parent
        self.chroma_path = Path(chroma_path) if chroma_path else project_root / "memory" / "mem0_chroma"
        self.history_db_path = (
            Path(history_db_path) if history_db_path else project_root / "memory" / "history.db"
        )
        self.version_path = (
            Path(version_path) if version_path else project_root / "memory" / "mem0_versions.json"
        )
        self.user_id = user_id
        self.budget_tracker = budget_tracker
        self.threshold = threshold
        self.collection_name = collection_name
        # embedding 模型/维度/mem0 版本：默认用模块常量；重嵌脚本
        # （tools/reembed_memories.py）会传旧模型来读旧库。
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.mem0_version = mem0_version
        self._memory = memory
        if self._memory is None:
            # 真实 mem0：先校验版本文件，再建 Memory（建的时候会加载
            # embedding 模型，约 25 秒，常驻服务只付一次）。
            _check_version_file(
                self.version_path,
                self.chroma_path,
                self.embedding_model,
                self.embedding_dims,
                self.mem0_version,
            )
            self._memory = self._build_memory()
            self._attach_usage_wrapper()

    def _build_memory(self):
        """建一个真实的 mem0 Memory（DeepSeek + 本地 bge + 本地 Chroma）。

        Key 从环境变量读（config.get_api_key，.env 里那个），没有 Key
        就报 RuntimeError，调用方（create_mem0_store）会降级成“没有记忆”。
        注意：这里才 import mem0，模块 import 时不碰它——这样没装 mem0
        的机器也能 import 本模块、跑离线测试（测试注入假 memory 对象）。
        """
        # 遥测必须在 import mem0 之前关掉（模块级已设过一次，这里再强制设一次，
        # 防止调用方在 import 本模块之前改了环境变量）。
        os.environ["MEM0_TELEMETRY"] = "False"
        from config import get_api_key

        api_key = get_api_key("deepseek")
        if not api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY，mem0 记忆未启用："
                "请在 .env 里填写 DEEPSEEK_API_KEY 后再运行。"
            )
        from mem0 import Memory

        config = {
            "llm": {
                "provider": "deepseek",
                "config": {
                    "model": LLM_MODEL,
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "api_key": api_key,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": self.embedding_model},
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": self.collection_name,
                    "path": str(self.chroma_path),
                },
            },
            "history_db_path": str(self.history_db_path),
            # mem0 2.2.1 里没有 custom_fact_extraction_prompt 这个字段，
            # 自定义规则只能走 custom_instructions（试跑已核实）。
            "custom_instructions": CUSTOM_INSTRUCTIONS,
        }
        return Memory.from_config(config)

    def _attach_usage_wrapper(self):
        """包住 mem0 的 LLM 客户端，记下每次提取调用的 Token，喂给 BudgetTracker。

        做法（试跑验证过）：mem0 的 DeepSeek LLM 在 memory.llm.client
        上挂了一个 OpenAI 兼容客户端，包住它的 chat.completions.create，
        每次返回后从 usage 里读 prompt/completion Token，按 DeepSeek
        Flash 闲时价算成本，记进 BudgetTracker（模型名记 deepseek-flash）。
        只记 Token 数和耗时，不记请求/回答内容。
        没传 budget_tracker 时不包；包不住（假对象、结构变了）时只记日志，
        不影响正常使用——记账失败不能把聊天搞崩。
        """
        if self.budget_tracker is None:
            return
        try:
            from config import MODEL_CONFIG

            client = self._memory.llm.client
            orig_create = client.chat.completions.create
            tracker = self.budget_tracker
            price_in = MODEL_CONFIG["deepseek"]["price_input_per_1m"]
            price_out = MODEL_CONFIG["deepseek"]["price_output_per_1m"]

            def wrapped(*args, **kwargs):
                resp = orig_create(*args, **kwargs)
                try:
                    usage = resp.usage
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    cost = (
                        prompt_tokens / 1_000_000 * price_in
                        + completion_tokens / 1_000_000 * price_out
                    )
                    tracker.add_cost(cost, model=LLM_MODEL)
                except Exception:
                    logger.warning("mem0 用量记录失败：本次提取调用未记账。", exc_info=True)
                return resp

            client.chat.completions.create = wrapped
        except Exception:
            logger.warning("mem0 用量包装失败：提取调用将不记账。", exc_info=True)

    @property
    def available(self):
        """记忆库是否可用（建好了就是 True）。"""
        return self._memory is not None

    # ---------- 检索（提问前） ----------

    def search(self, query, top_k=5):
        """搜和问题相关的记忆（语义检索 + 关键词补充信号）。

        返回：记忆列表，每项为 {"id", "text", "score", "source"}，
        source 是 "semantic"（语义）或 "keyword"（关键词补充）。
        流程：
        1. mem0 语义检索（top_k 条），只留分数 ≥ 阈值（默认 0.4）的。
           分数不够的直接丢掉，不注入——宁可不带记忆，也别把看着像的
           记忆当答案（BOOS 的硬规矩：不知道一定要说不知道）。
        2. 关键词补充：把库里所有记忆拿出来（get_all），用 v1 的两字窗口
           关键词再筛一遍，命中（权重 ≥ 2）且语义结果里没有的，追加在后面。
           这是给精确专有名词（“圆通”“Kimi”）兜底，Chroma 没有 BM25，
           语义万一漏了，关键词能捞回来。
        3. 总数不超过 top_k；任何一步出错都只记日志、返回已拿到的结果，
           不抛异常、不拦聊天。
        """
        results = []
        seen_texts = set()
        try:
            raw = self._memory.search(query, filters={"user_id": self.user_id}, top_k=top_k)
            for item in (raw or {}).get("results", []):
                score = item.get("score")
                # 分数缺失时不注入（不知道它有多相关，就当它不相关）。
                if score is None or score < self.threshold:
                    continue
                text = item.get("memory") or ""
                if not text:
                    continue
                results.append(
                    {
                        "id": item.get("id"),
                        "text": text,
                        "score": score,
                        "source": "semantic",
                    }
                )
                seen_texts.add(text)
        except Exception:
            logger.warning("mem0 语义检索失败：本次只用关键词补充信号。", exc_info=True)
        # 关键词补充信号（排在语义结果后面，不超过 top_k 总数）。
        if len(results) < top_k:
            try:
                keywords = _extract_keywords(query)
                if keywords:
                    all_items = self._memory.get_all(filters={"user_id": self.user_id})
                    for item in (all_items or {}).get("results", []):
                        if len(results) >= top_k:
                            break
                        text = item.get("memory") or ""
                        if not text or text in seen_texts:
                            continue
                        weight, _matched = _keyword_weight(text, keywords)
                        if weight >= KEYWORD_MIN_WEIGHT:
                            results.append(
                                {
                                    "id": item.get("id"),
                                    "text": text,
                                    "score": None,
                                    "source": "keyword",
                                }
                            )
                            seen_texts.add(text)
            except Exception:
                logger.warning("mem0 关键词补充检索失败。", exc_info=True)
        return results

    def get_all(self):
        """返回当前用户的所有记忆（列表，每项 {"id", "text"}）。

        供迁移、删除校验、种子检查用。出问题时返回空列表，不抛异常。
        """
        try:
            raw = self._memory.get_all(filters={"user_id": self.user_id})
            out = []
            for item in (raw or {}).get("results", []):
                text = item.get("memory") or ""
                if text:
                    out.append({"id": item.get("id"), "text": text})
            return out
        except Exception:
            logger.warning("mem0 取全部记忆失败。", exc_info=True)
            return []

    # ---------- 写入（回答后自动存 / 手动存） ----------

    def add_turn(self, messages):
        """把本轮对话存进长期记忆（回答后调一次）。

        参数 messages：本轮及最近几轮的消息列表（user/assistant 交替）。
        mem0 自己调 DeepSeek 提取（自动存，BOOS 已同意不用确认），
        再去重、存进向量库。整个写入走全局写锁，一次只让一个写，
        防止并发写出重复记忆。
        返回：mem0.add() 的结果（字典），失败时返回 {}，绝不抛异常——
        存记忆失败不能把已经拿到的聊天回答搞崩。
        """
        if not messages:
            return {}
        with _WRITE_LOCK:
            try:
                result = self._memory.add(messages, user_id=self.user_id)
                return result or {}
            except Exception:
                logger.warning("mem0 自动存记忆失败：本轮对话未存入长期记忆。", exc_info=True)
                return {}

    def add_memory_text(self, text, layers=None, source="user"):
        """手动存一条记忆原文（CLI“记住：内容”用，不经过提取）。

        参数 text：记忆内容原文；layers/source：保留给以后用，
        mem0 的 infer=False 路径不吃这两个字段，这里只记日志。
        走全局写锁。返回：mem0.add() 的结果，失败返回 {}，不抛异常。
        layers/source 现在不用（mem0 的 infer=False 直接存原文，
        不分层、不标来源），参数留着是为了调用方不用改签名。
        """
        if not isinstance(text, str) or not text.strip():
            return {}
        with _WRITE_LOCK:
            try:
                result = self._memory.add(
                    [{"role": "user", "content": text.strip()}],
                    user_id=self.user_id,
                    infer=False,
                )
                return result or {}
            except Exception:
                logger.warning("mem0 手动存记忆失败。", exc_info=True)
                return {}

    # ---------- 迁移与种子 ----------

    def migrate_from_v1(self, memories_json_path=None):
        """把旧 memories.json 里的记忆一次性迁进 mem0（原文迁移，不重新提取）。

        参数 memories_json_path：旧文件路径，不传时用项目 memory/memories.json。
        每条用 add(infer=False) 原文存入，旧记忆默认全部有效。
        例外：旧的第 3 条预算记忆写的是已作废的旧规矩
        （“每月预算 5 美元，4 美元提醒，5 美元停。”），迁移时改成新规矩
        （FIRST_MEMORIES 里的第 3 条），并记一条日志说明改过；
        其他记忆一律一字不改。
        已在库里的（按原文完全相同算）跳过，不重复存。
        返回：本次新存入的条数。文件不存在或读不出时返回 0，不抛异常。
        """
        path = (
            Path(memories_json_path)
            if memories_json_path
            else Path(__file__).resolve().parent / "memories.json"
        )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("迁移：旧记忆文件 %s 不存在或读不出来，跳过迁移。", path)
            return 0
        if not isinstance(data, list):
            logger.warning("迁移：旧记忆文件里不是一个列表，跳过迁移。")
            return 0
        existing_texts = {item["text"] for item in self.get_all()}
        added = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            # 已作废的旧预算规则：迁移时改成新规矩，并记日志。
            if text == _OUTDATED_BUDGET_TEXT:
                new_text = FIRST_MEMORIES[2]["text"]
                logger.info("迁移：旧预算记忆已按新规矩改写（%s → %s）。", text, new_text)
                text = new_text
            if text in existing_texts:
                continue
            result = self.add_memory_text(text)
            if result:
                added += 1
                existing_texts.add(text)
        logger.info("迁移完成：从 %s 新存入 %d 条记忆。", path, added)
        return added

    def ensure_first_memories(self):
        """启动时确保首批三条记忆已存（只在空库时做一次）。

        流程：库里已有记忆时什么也不做；空库时先试着从旧 memories.json
        迁移（有文件就有记忆）；迁完还是空（没有旧文件）时，把
        FIRST_MEMORIES 三条用 infer=False 原文存进去。
        返回：本次新存入的条数（0 或 3，或迁移的条数）。
        记忆库出问题时返回 0，不抛异常、不影响服务启动。
        """
        try:
            if self.get_all():
                return 0
        except Exception:
            logger.warning("检查已有记忆失败，本次不种初始记忆。", exc_info=True)
            return 0
        migrated = self.migrate_from_v1()
        if migrated:
            return migrated
        added = 0
        for item in FIRST_MEMORIES:
            result = self.add_memory_text(item["text"])
            if result:
                added += 1
        if added:
            logger.info("已记录 %d 条初始记忆。", added)
        return added

    # ---------- 删除与历史（带守卫） ----------

    def delete_memory(self, memory_id, approve=False):
        """删一条记忆（两道关，缺一个都不删）。

        第一道：归属校验——这条记忆必须属于当前用户（在 get_all() 里
        能找到这个 id），别人的记忆删不掉（mem0 的 delete(id) 本身
        不校验归属，这里补上）。
        第二道：BOOS 明确同意——调用方必须传 approve=True，否则报
        PermissionError。删除记忆必须经过 BOOS 同意，不能悄悄删。
        两道都过了才调 mem0.delete(id)。返回 True；找不到记忆时
        返回 False（已经没了，也算删干净了）；mem0 报错时抛异常，
        调用方决定怎么处理。
        注意：现在还没有删除的网页界面/命令行入口，这个方法是给以后
        的删除入口准备的；任何入口调它之前，都必须先问 BOOS。
        """
        if not approve:
            raise PermissionError("删除记忆需要 BOOS 同意：请先确认再删。")
        if not memory_id:
            return False
        with _WRITE_LOCK:
            owned_ids = {item["id"] for item in self.get_all()}
            if memory_id not in owned_ids:
                logger.warning("删除被拒：记忆 %s 不属于当前用户。", memory_id)
                return False
            self._memory.delete(memory_id)
            logger.info("已删除记忆：id=%s。", memory_id)
            return True

    def history(self, memory_id):
        """取一条记忆的修改历史（mem0 自带的 SQLite 审计记录）。

        返回：历史条目列表（每项含 old/new 内容与时间戳），
        查不到或出错时返回空列表，不抛异常。
        删除过的记忆，历史里仍能查到（is_deleted=True），有据可查。
        """
        try:
            result = self._memory.history(memory_id)
            return result if isinstance(result, list) else []
        except Exception:
            logger.warning("取记忆历史失败：id=%s。", memory_id, exc_info=True)
            return []


# ---------- 懒加载单例（命令行默认不加载，网页服务启动时建） ----------

_mem0_store_singleton = None
_mem0_store_lock = threading.Lock()


def create_mem0_store(budget_tracker=None, **kwargs):
    """建一个 Mem0Store；建不好时返回 None（降级成“没有记忆也能聊”）。

    建不好的情况：没配 DEEPSEEK_API_KEY、mem0 没装、版本文件不匹配、
    Chroma 目录坏了。只记一条警告日志，不抛异常——记忆库出问题
    不能把聊天搞崩。
    """
    try:
        return Mem0Store(budget_tracker=budget_tracker, **kwargs)
    except Exception as exc:
        logger.warning("mem0 记忆未启用：%s，本次不带记忆运行。", exc)
        return None


def get_mem0_store(budget_tracker=None):
    """懒加载的 Mem0Store 单例（命令行“记住……”时按需加载）。

    第一次调用时才建（建的时候要加载 embedding 模型，约 25 秒），
    建好后一直复用。建不好时返回 None（并记日志），调用方按“没有记忆”处理。
    注意：网页服务不要用这个——网页服务在启动时就用 create_mem0_store()
    建好、传给 ChatState；这里的单例是给命令行准备的。
    """
    global _mem0_store_singleton
    if _mem0_store_singleton is not None:
        return _mem0_store_singleton
    with _mem0_store_lock:
        if _mem0_store_singleton is None:
            _mem0_store_singleton = create_mem0_store(budget_tracker=budget_tracker)
        return _mem0_store_singleton


def reset_mem0_store_singleton():
    """清掉懒加载单例（只给测试用，正式代码不要调）。

    测试里换了临时目录/假 memory 对象后，单例还指着旧的，
    调这个清掉，下次 get_mem0_store() 会重新建。
    """
    global _mem0_store_singleton
    with _mem0_store_lock:
        _mem0_store_singleton = None
