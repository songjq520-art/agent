# memory/sqlite_store.py（SQLite 记忆索引，第二步，2026-11-21）
#
# 设计（BOOS 2026-11-20 定，2026-11-21 网页测后改走这条路）：
#   - JSONL 账本（memory_events.jsonl）仍是唯一真相源：先写账本、再写索引。
#   - SQLite（WAL）只做仓库存取：存原文 + metadata + 向量，不替 BOOS 做决定。
#   - mem0/Chroma 不再参与这条路的存取；Mem0Store 的外观（search/save_turn/
#     flush_pending/get_all/add_memory_text/delete_memory）照旧，调用方不用改。
#
# 为什么要有这个模块（病根，见 ~/workspace/十二网页测-2026-11-21.md）：
#   mem0 那条路里，用户亲口说的“一顿饭”要过 DeepSeek 提取 + 价值关卡，
#   提取提示词第 9 条（“只存有长期价值的，一次性小事不值得存，有疑问就不存”）
#   会把饺子、豆浆、豆角判成不值得存，提取返回空列表；而提交那层把“空手回来”
#   当成成功：指纹照样记成已处理、outbox 文件照样删、不留回执、不报错。
#   事实就这样悄悄丢了，日志里什么都不剩。
#   所以这条路反过来：用户亲口说的事实先原样落库（账本 + SQLite），
#   提取/打标只做归类，不再有一票否决不让存的权力；每次提交留回执。
#
# 今天第一部分只做到：说一句新的事实，关对话，新开对话问它，能记住
# （账本和 SQLite 里都有）。还没做的（后面几步再做）：
#   - 纠错 correct_memory（现在只返回明确的 not_yet，不悄悄猜）
#   - 衰减/晋升/沉底（decay.py 的规则还没接进来，权重先恒为 1.0）
#   - 类型权重（稳定偏好 > 一次性事件）已接进检索排序（2026-11-21 Q3
#     修法，见 _type_weight；只管排位、不管 0.4 资格线）
#   - 图谱边、精选清单（检索三路里的图谱路还是空路，精选先查还没接；
#     BM25 + RRF 融合已接，见 _search_inner；衰减权重仍恒为 1.0，只留了接口）
#   - 撤回目前只认光秃秃的“不不不/算了”这类整句信号，带后半句的撤回
#     （要模型判意图）还没接，用时会先存下来、再由 BOOS 删，不悄悄丢。

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from memory import decay as _decay
from memory import ledger as _ledger
from memory import provenance as _provenance
from memory.correction_text import (
    _detect_correction, _strip_correction_sentences,
    _detect_correction_fragment, _VAGUE_FRAGMENT_VALUES,
    _infer_fact_slot,
)
from memory.money_slots import extract_core_slots
from memory.redaction import _contains_sensitive, _normalize_memory_text
from memory.timeutil import _now_epoch, _now_iso, _next_write_seq

logger = logging.getLogger(__name__)

# 相对日期换算用的基准时区：BOOS 的参考时区（MEMORY.md）。
_SHANGHAI = timezone(timedelta(hours=8))

# 向量模型：和 mem0 那条路同一个本地模型，离线加载（启动脚本已设
# HF_HUB_OFFLINE=1 / HF_ENDPOINT=https://hf-mirror.com）。
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# 语义检索的入围线：和 mem0 那条路同一个 0.4（BOOS 定的，留着不动）。
SEARCH_THRESHOLD = 0.4

# 撤回信号：光秃秃的一整句（Plan C 快捷路同款意思，简化版）。
_BARE_RETRACTION = {
    "不", "不不", "不不不", "不是", "没有", "算了", "别", "不要",
    "不对", "错了", "撤回", "取消", "当我没说",
}

# 稳定偏好在排序时的加权倍数（Q3 修法，2026-11-21，见 _type_weight）：
# 纠正行 RRF 0.0152、米线 0.0154，一位之差——1.25 倍足够把偏好行
# 顶回前 5，又小到不会把事件问（“昨天吃了什么”里饺子 RRF 约 0.033，
# 语义 + BM25 两路）里的偏好行顶到第一。
_TYPE_WEIGHT_STABLE_PREFERENCE = 1.25

# 已确认记忆在检索排序里只当平局裁决，不做乘法加权（#2，2026-11-22 修）：
# 同文本聚合已保证确认版赢下自己的槽位；排序里再乘 1.25 会把确认行
# 顶到分更高的未确认行前面，Q4/BM25 钉死的融合顺序就乱了
# （目标测试回归验出）。所以排序键只在 RRF×衰减×类型权重完全相等时，
# 用 confirmed 裁决先后。

# 偏好句的认法：正文里说喜欢 / 不喜欢 / 爱吃 / 爱喝的句子。
# 名单故意收得窄：只认这几个动词，不猜“想 / 希望 / 要”（那些是计划，
# 不是偏好）；“可爱”这类词不含这几个动词，不会误认。
_PREFERENCE_WORDS = ("喜欢", "爱吃", "爱喝", "爱的是", "不爱", "讨厌")

# 问句不存：以这些结尾的用户话是提问，不是陈述事实。
_QUESTION_ENDINGS = ("?", "？", "吗", "么", "呢", "吧", "嘛")

# 打招呼/闲聊不存（只有整句是这些时）。
_SMALL_TALK = {
    "你好", "您好", "hi", "hello", "嗨", "在吗", "在么", "谢谢",
    "好的", "好", "嗯", "嗯嗯", "知道了", "收到", "拜拜", "再见",
    "就是这样", "就是这个",
}

# 相对日期词 → 和今天（上海）的天数差。
_RELATIVE_DAY_OFFSETS = {
    "大前天": -3, "前天": -2, "昨天": -1, "今天": 0, "明天": 1, "后天": 2, "大后天": 3,
}


# 中文星期 → 周一为 0 的序号（和 datetime.weekday() 一致）。
_WEEKDAY_CN_TO_NUM = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def _resolve_weekday(text, today):
    """“下周三/本周三/这周三/上周三”这类带星期的相对日期 → 绝对日期。

    周一是一周的开始：本周 = 今天所在周，下周 = 下一周，上周 = 上一周。
    找不到带星期的表达时返回 None。
    """
    m = re.search(r"(下周|本周|这周|上周)(?:星期|周|礼拜)?([一二三四五六日天])", text or "")
    if not m:
        return None
    qualifier, wd = m.group(1), m.group(2)
    target = _WEEKDAY_CN_TO_NUM[wd]
    monday = today - timedelta(days=today.weekday())
    if qualifier == "下周":
        base = monday + timedelta(days=7)
    elif qualifier in ("本周", "这周"):
        base = monday
    else:  # 上周
        base = monday - timedelta(days=7)
    return (base + timedelta(days=target)).isoformat()


def _shanghai_today():
    """今天（Asia/Shanghai 的日期）。"""
    return datetime.now(_SHANGHAI).date()


def resolve_event_date(text, today=None):
    """把一句话里的相对日期换算成绝对日期（YYYY-MM-DD）。

    找不到日期词时返回 None（不是没有日期，是这句话没提日期）。
    “昨天”以今天（上海）为基准算，不以任何记忆的日期为基准——
    饺子记成 11月19日就是因为提示里唯一的日期锚点是牛肉面那天。
    """
    today = today or _shanghai_today()
    for word, offset in _RELATIVE_DAY_OFFSETS.items():
        if word in (text or ""):
            return (today + timedelta(days=offset)).isoformat()
    wd = _resolve_weekday(text, today)
    if wd:
        return wd
    m = re.search(
        r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text or ""
    )
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})月(\d{1,2})日", text or "")
    if m:
        try:
            return date(today.year, int(m.group(1)), int(m.group(2))).isoformat()
        except ValueError:
            return None
    return None


def _split_sentences(text):
    """按句号/感叹号/问号/换行切句，句尾标点留着——问句判断要靠它
    （切掉问号就认不出问句了）；存的时候再把句尾的 。！去掉。"""
    parts = re.findall(r"[^。！？\n]+[。！？]?", text or "")
    return [p.strip() for p in parts if p.strip()]


# “记住”类前缀（2026-11-21 网页测第二轮 #8）：save_turn 逐句循环里
# 先剥掉再存——“记住后天要去体检”要存成“后天要去体检”，不能把
# “记住”两个字也存进记忆。顺序按最长优先；main.py 的
# is_remember_command / remember_content 用同一份名单（两边改了要同步）。
REMEMBER_PREFIXES = ("帮我记住", "请记住", "记住")


def _strip_remember_prefix(sentence):
    """剥掉句首的“记住”类前缀，返回剩下的内容；没有前缀时原样返回。"""
    s = (sentence or "").strip()
    for prefix in REMEMBER_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):].lstrip("：:、，, ").strip()
    return s


# 确认词前缀（2026-11-21 网页测第三轮 #8“别什么都记”）：BOOS 回一句
# “是的/对”再跟半句话（“是的删掉”）时，确认词本身不是记忆内容，
# 先剥掉再判剩下的；剥完没内容就是光秃秃一句确认词，直接丢掉。
# 两个字的前缀（“是的/对的/没错/可以/好的”）直接剥；一个字的
# （“好/嗯/行/对”）只有后面跟了分隔符才剥（“好吃”不能剥成“吃”，
# “对不起”不能剥成“不起”）。
CONFIRMATION_PREFIXES = ("是的", "对的", "没错", "可以", "好的")
_CONFIRMATION_SINGLE = ("好", "嗯", "行", "对")


def _strip_confirmation_prefix(sentence):
    """剥掉句首的确认词，返回剩下的内容；没有确认词前缀时原样返回。

    剥完为空时返回 ""（调用方按确认词丢掉）。只剥句首，句中的“是的”不动。
    """
    s = (sentence or "").strip()
    for prefix in CONFIRMATION_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):].lstrip("：:、，, ").strip()
    for prefix in _CONFIRMATION_SINGLE:
        if s.startswith(prefix) and len(s) > 1 and s[1] in "：:、，, ":
            return s[1:].lstrip("：:、，, ").strip()
    return s


def _looks_like_fact(sentence):
    """一句话像不像用户亲口说的事实（原样先存的候选）。

    不存的：太短的、问句、假设句、确认词、半句话、当下的想法、
    打招呼、光秃秃的撤回、让删/让忘的指令。
    其余的都先存下来——值不值得长期留以后靠衰减沉底，不在这一步判死刑。
    2026-11-21 网页测第三轮 #8（BOOS 定：改成和十一一样，别什么都记）：
    过滤器必须是确定性的程序规则（正则/规则），不能靠模型运气——
    之前就是存不存看运气（有的问句存了、有的没存）。
    """
    s = (sentence or "").strip()
    if len(s) < 4:
        return False, "太短"
    if s in _SMALL_TALK:
        return False, "闲聊"
    if s in _BARE_RETRACTION:
        return False, "撤回信号"
    if s.endswith(_QUESTION_ENDINGS):
        return False, "问句"
    # 没打问号的问句：以“怎么样/如何/为什么/谁/哪/多少/怎么办/什么”收尾的，
    # 几乎都是提问（“我不知道他是谁”这类也只是“不知道”的陈述，
    # 本来就不值得当事实存）。2026-11-21 回归抽查抓到的：
    # “定投计划怎么样了”没打问号，被当事实存进去、检索还排第一；
    # 第三轮又抓到“我爱吃什么”（“什么”结尾）、“我的定投现在是什么情况”
    # （“什么情况”结尾）也被当事实存进去。
    if s.endswith(("怎么样", "怎么样了", "如何", "如何了", "怎么了", "为什么",
                   "怎么办", "是什么", "是不是", "谁", "哪", "多少",
                   "几个", "几天", "几点", "什么", "什么情况")):
        return False, "问句"
    # 正反问（A-不-A）和疑问词开头的问句：没打问号、也不以疑问词收尾，
    # 但句式就是提问——“我喜不喜欢吃鸡杂面”“有没有人来过”。
    # “我不知道他是谁”不含这些词，不受影响；“我不喜欢吃辣”含的是
    # “不喜欢”不是“喜不喜欢”，不受影响。
    if re.search(r"(喜不喜欢|爱不爱|想不想|要不要|会不会|能不能|懂不懂|记不记得|知不知道)", s):
        return False, "问句"
    if re.match(r"^(有没有|是不是|能不能|会不会|要不要|想不想|喜不喜欢|爱不爱|懂不懂|记不记得|知不知道|为什么|怎么|如何|什么时候|什么地方|什么东西|什么人)", s):
        return False, "问句"
    # 假设句（第三轮 #8）：句首的“如果/假设/假如/要是/万一”——条件句、
    # 假设题不是既定事实，不存。mem0 那条路的提取（FakeMemory）本来就
    # 不认句首假设句，这里补齐同一条规矩。句中的“如果”不动
    # （“你说如果下雨就别去”是条件指令，不是假设题）。
    if re.match(r"^(如果|假设|假如|要是|万一)", s):
        return False, "假设句"
    # 半句话（第三轮 #8）：“这几条取消”——指示代词 + 量词 + 动词，
    # 半截指令，没有完整意思，不存。
    if re.match(r"^[这那][几]?[条个只项种]?(取消|删掉|删除|去掉|不要|留着|算了)", s):
        return False, "半句话"
    if "不要记" in s:
        return False, "删除指令"
    # 句中的删除指令（第三轮 #8）：“算了，关于定投的全部删掉吧”
    # 这类没走句首的删指令；句尾带“了”的回忆句除外。
    if any(w in s for w in ("全部删掉", "都删掉", "删掉吧")) and not s.endswith("了"):
        return False, "删除指令"
    # “忘记…”指令句（2026-11-21 网页测第二轮 #7）：“忘记我喜欢喝茶”
    # 是删记忆的指令，不能当事实存。只认句首的忘记/忘掉/删掉/删除
    # （“我昨天忘记带钥匙了”不在句首，不受影响）；句尾带“了”的是
    # 回忆句（“忘记带钥匙了”），不是指令，照常存；“忘记了…”（过去时）
    # 也是回忆，照常存。没带“了”的“忘记带钥匙”会被当成指令丢掉——
    # 故意的取舍：删指令绝不能进库，丢一句含糊的回忆可以再说一遍。
    if re.match(r"^(?:请帮我|帮我|请)?(?:忘记|忘掉|删掉|删除)(?!了)", s) and not s.endswith("了"):
        return False, "删除指令"
    # 句中的“忘记这个/忘掉这个/删掉这条”（比如“你能不能忘记这个”）
    # 同样是指令；句尾带“了”的回忆句除外。
    if any(w in s for w in ("忘记这个", "忘掉这个", "删掉这条")) and not s.endswith("了"):
        return False, "删除指令"
    # 当下的想法（第三轮 #8，BOOS 定：按十一的来）：“我想喝茶了”这种
    # 当下的想法不存成偏好/长期记忆。信号：想/馋 + 当下标记
    # （句尾“了”、句尾“想”、“现在/这会儿”）；带了以后/明天/后天/
    # 下周/打算/决定/计划/每天/每周的，是未来的计划或安排，不拦。
    # “我想去爬山”（没“了”、没未来标记）是表态的意图，拿不准，
    # 先留着（沉底规则管它），不拦。
    if re.search(r"[想馋]", s) and (
        s.endswith("了") or s.endswith("想") or "现在" in s or "这会儿" in s
    ) and not any(
        w in s for w in ("以后", "明天", "后天", "下周", "下个月", "打算", "决定", "计划", "从今天起", "每天", "每周")
    ):
        return False, "当下的想法"
    # 纠错信号本身（2026-11-22 correct_memory 接线后）：带新说法的纠错句
    # 由 save_turn 经 correct_memory 处理，不到这里；这里只拦光秃秃的
    # 纠错信号（“你记错了”后面没跟新说法），那不是一条事实。
    # 原来的宽拦截（句首是纠错说法就整句丢）已随接线移除，否则
    # “你记错了，我住上海”会把后半句纠错新事实一并丢弃。
    if re.match(r"^(?:(?:不对|不是|哎)[，,、\s]*)?(?:你|我)?(?:之前|刚才)?(?:记错了|说错了|搞错了)[，,。！!、\s]*$", s):
        return False, "纠错指令"
    if re.match(r"^(?:纠正一下|更正一下|更正|改一下|不对|错了|不是)[，,。！!、\s]*$", s):
        return False, "纠错指令"
    return True, ""



def _normalize_metadata(meta):
    """读时归一化 metadata：补算 confirmed（mem0 的规矩，见
    memory/provenance.py _normalize_metadata：confirmed 就是
    user_confirmed 的短名字，读的时候现算，不是存的时候写）。

    账本/Chroma 来的旧 metadata 里没有 confirmed 这个键——不补算，
    问答提示词（main.py _memory_block）会把 source=user 的已确认记忆
    也当成【来源未确认】、问起来只说不知道。只补缺失的键，不覆盖已有的。
    """
    meta = dict(meta or {})
    if "confirmed" not in meta:
        meta["confirmed"] = bool(meta.get("user_confirmed"))
    return meta


class SQLiteStore:
    """SQLite 记忆库：账本是真相，SQLite 是索引。

    和 Mem0Store 同样的外观，供 web/server.py、main.py 直接用。
    """

    def __init__(self, memory_dir=None, budget_tracker=None, **kwargs):
        self.memory_dir = Path(memory_dir or _default_memory_dir())
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.memory_dir / "memory_events.jsonl"
        self.snapshot_path = self.memory_dir / "graph_snapshot.json"
        self.db_path = self.memory_dir / "memories.sqlite3"
        self.budget_tracker = budget_tracker
        self.user_id = kwargs.get("user_id", "boos")
        self._lock = threading.RLock()
        self._ledger_lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._embedder = None
        self._embedder_failed = False
        self._seen_turn_ids = set()
        self._last_context_ids = []
        self._last_stored_turn_id = None

    # ---------- 建库 ----------

    def _init_schema(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                write_seq INTEGER,
                slot TEXT,
                source TEXT,
                user_confirmed INTEGER NOT NULL DEFAULT 0,
                decay_kind TEXT,
                importance TEXT,
                event_date TEXT,
                embedding BLOB,
                voided INTEGER NOT NULL DEFAULT 0,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_event_date
                ON memories(event_date);
            CREATE TABLE IF NOT EXISTS vector_cache (
                sha256 TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()
        # superseded 列（2026-11-22 correct_memory 接线）：被取代的记忆
        # 在索引里标 superseded=1，不再进活记忆；旧库没有这列时补上。
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "superseded" not in cols:
            self._conn.execute("ALTER TABLE memories ADD COLUMN superseded INTEGER NOT NULL DEFAULT 0")
            self._conn.commit()

    def available(self):
        return True

    # ---------- 向量（本地 bge，带缓存） ----------

    def _embed_text(self, text):
        """给一句话算向量：先查缓存（key=sha256(text)），没有再调本地模型。

        模型起不来时返回 None——存记忆绝不因为向量失败而失败，
        那条记忆先以无向量入库，检索时靠原样/关键词兜底。
        """
        key = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        # 向量缓存是同一单连接上的读写：和写路径一样加锁（#2）。
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM vector_cache WHERE sha256=?", (key,)
            ).fetchone()
        if row is not None:
            return _blob_to_vector(row[0])
        if self._embedder_failed:
            return None
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                # 本地加载，不碰 Hub：这台电脑离线也能从缓存起模型；
                # 不加 local_files_only 时，新版 transformers 会去 Hub 查
                # chat template，断网时一次次重试、拖几分钟。
                self._embedder = SentenceTransformer(
                    EMBEDDING_MODEL, local_files_only=True
                )
            import numpy as np

            vec = self._embedder.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]
            vec = np.asarray(vec, dtype="float32")
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO vector_cache(sha256, embedding, model, dim)"
                    " VALUES (?, ?, ?, ?)",
                    (key, vec.tobytes(), EMBEDDING_MODEL, int(vec.shape[0])),
                )
                self._conn.commit()
            return vec
        except Exception:
            logger.warning("本地向量模型起不来：本次只存文字、不存向量。", exc_info=True)
            self._embedder_failed = True
            return None

    # ---------- 账本（真相源） ----------

    def _ledger_append_add(self, text, metadata):
        """先写账本 add 事件（写失败就抛，索引那步也别写）。"""
        return _ledger.append_event(
            self.ledger_path,
            {
                "type": _ledger.LEDGER_EVENT_ADD,
                "key": _ledger.ledger_event_key(text, metadata.get("created_at")),
                "text": text,
                "meta": {
                    "source": metadata.get("source"),
                    "confidence": metadata.get("confidence"),
                    "user_confirmed": metadata.get("user_confirmed"),
                    "created_at": metadata.get("created_at"),
                    "write_seq": metadata.get("write_seq"),
                    "turn_id": metadata.get("turn_id"),
                    "slot": metadata.get("slot"),
                    "is_rule": metadata.get("is_rule", False),
                    "curated": metadata.get("curated", False),
                    "decay_kind": metadata.get("decay_kind"),
                    "importance": metadata.get("importance"),
                    "event_date": metadata.get("event_date"),
                    # 衰减字段（2026-11-22 decay 接线）：账本是真相，
                    # 对账重建（reconcile_ledger_index）直接拿这里的
                    # meta 当 metadata_json，不写进来重建就丢了。
                    "last_used_at": metadata.get("last_used_at"),
                    "decay_state": metadata.get("decay_state"),
                    "seen_dates": metadata.get("seen_dates"),
                    "seen_sessions": metadata.get("seen_sessions"),
                },
            },
            lock=self._ledger_lock,
        )

    def ledger_state(self):
        try:
            return _ledger.load_ledger_state(self.ledger_path, self.snapshot_path)
        except Exception:
            logger.warning("读账本状态失败。", exc_info=True)
            return {"by_id": {}, "adds": [], "relations": [],
                    "key_states": {}, "id_to_key": {}, "inferences": {},
                    "seq_max": 0, "snapshot_seq": None}

    def reconcile_ledger_index(self):
        """启动对账：账本里有、SQLite 里没有的，按账本重建索引。

        只重建当前态的记忆：被取代（superseded）/删了（deleted）的
        key 不进索引（账本里原话留着，索引里不给搜到——和 mem0 那条路
        的投影过滤同一个意思）。
        """
        with self._lock:
            state = self.ledger_state()
            key_states = (state or {}).get("key_states") or {}
            added = 0
            for record in (state or {}).get("adds") or []:
                key = record.get("key")
                text = record.get("text") or ""
                meta = record.get("meta") or {}
                if not key or not text.strip():
                    continue
                if key_states.get(key) in (
                    _ledger.LEDGER_STATE_SUPERSEDED,
                    _ledger.LEDGER_STATE_DELETED,
                ):
                    continue
                if self._conn.execute(
                    "SELECT 1 FROM memories WHERE id=?", (key,)
                ).fetchone():
                    continue
                vec = self._embed_text(text)
                self._insert_memory_row(key, text, meta, vec)
                added += 1
            return added

    def import_from_mem0(self, mem0_store):
        """一次性迁移：把只活在 Chroma 里的旧记忆搬进来（2026-11-21 核实）。

        沙盒账本只有 5 条当前态，Chroma 里有 31 条——2026-11-07 至
        11-17 存的 27 条在账本里根本没有 add 记录（那时还没接账本）。
        只靠账本对账会丢掉 26 条，验收必退化，所以迁移必须多这一步：
        逐条读 mem0 的 get_all，按原文比对——SQLite 里已有同一原文的
        跳过；账本里有同一原文的 delete 事件的跳过（删了的就是删了）；
        其余的先给账本补记一笔 add（删前补记同款做法），再进 SQLite。
        幂等：跑两次也不会重复（原文比对 + 账本 key 去重）。
        返回 {"imported": n, "skipped_existing": n, "skipped_deleted": n, "skipped_sensitive": n}。
        """
        with self._lock:
            deleted_texts = set()
            try:
                events, _info = _ledger.read_events(self.ledger_path)
                for event in events:
                    if event.get("type") == "delete":
                        text = (event.get("text") or "").strip()
                        if text:
                            deleted_texts.add(text)
            except Exception:
                logger.warning("读账本删记录失败：迁移时不按删除过滤。", exc_info=True)
            existing_texts = {
                (row[0] or "").strip()
                for row in self._conn.execute(
                    "SELECT text FROM memories WHERE voided=0 AND deleted=0 AND superseded=0"
                ).fetchall()
            }
            imported = skipped_existing = skipped_deleted = skipped_sensitive = 0
            items = mem0_store.get_all(include_sensitive=True) or []
            for item in items:
                text = ((item or {}).get("memory") or (item or {}).get("text") or "").strip()
                if not text:
                    continue
                if _contains_sensitive(text):
                    # 迁移时也拦敏感行（2026-11-21 Gemini 第三轮，BOOS 定）：
                    # 旧库里的密码/卡号不进账本、不进 SQLite。
                    skipped_sensitive += 1
                    continue
                if text in existing_texts:
                    skipped_existing += 1
                    continue
                if text in deleted_texts:
                    skipped_deleted += 1
                    continue
                meta_in = (item or {}).get("metadata") or {}
                metadata = {
                    "source": meta_in.get("source", "inferred"),
                    "confidence": meta_in.get("confidence", 0.7),
                    "user_confirmed": bool(meta_in.get("user_confirmed", False)),
                    "created_at": meta_in.get("created_at") or _now_iso(),
                    "write_seq": meta_in.get("write_seq") or _next_write_seq(),
                    "turn_id": meta_in.get("turn_id"),
                    "slot": meta_in.get("slot"),
                    "is_rule": bool(meta_in.get("is_rule", False)),
                    "curated": bool(meta_in.get("curated", False)),
                    "decay_kind": meta_in.get("decay_kind", "episodic"),
                    "importance": meta_in.get("importance", "normal"),
                    "event_date": meta_in.get("event_date")
                    or resolve_event_date(text),
                }
                try:
                    self._ledger_append_add(text, metadata)
                except Exception:
                    logger.warning("迁移补记账本失败：跳过这一条。", exc_info=True)
                    continue
                vec = self._embed_text(text)
                key = _ledger.ledger_event_key(text, metadata.get("created_at"))
                self._insert_memory_row(key, text, metadata, vec)
                existing_texts.add(text)
                imported += 1
            return {
                "imported": imported,
                "skipped_existing": skipped_existing,
                "skipped_deleted": skipped_deleted,
                "skipped_sensitive": skipped_sensitive,
            }

    # ---------- 写（原样先存） ----------

    def _insert_memory_row(self, key, text, metadata, vector):
        import numpy as np

        blob = None
        if vector is not None:
            blob = np.asarray(vector, dtype="float32").tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO memories"
            "(id, text, metadata_json, created_at, write_seq, slot, source,"
            " user_confirmed, decay_kind, importance, event_date, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                text,
                json.dumps(metadata, ensure_ascii=False),
                metadata.get("created_at") or _now_iso(),
                metadata.get("write_seq"),
                metadata.get("slot"),
                metadata.get("source"),
                1 if metadata.get("user_confirmed") else 0,
                metadata.get("decay_kind"),
                metadata.get("importance"),
                metadata.get("event_date"),
                blob,
            ),
        )
        self._conn.commit()

    def add_memory_text(self, text, layers=None, source="user", turn_id=None,
                        event_date=None, **kwargs):
        """原文存一条记忆（“记住”直存、种子、规则都走这里）。

        先写账本、再写 SQLite；账本写不进去就返回 {}、不写索引。
        同文去重（2026-11-22，Bug 2 治本）：同一归一化文本的活记忆
        已在库里时，不再存第二行，返回 UNCHANGED，并记一笔“又出现一次”
        （出现集合 + 晋升检查）；来源不看——已确认版和推断版是同一事实，
        确认状态只往上提（推断版在库、已确认版后到时，把已有行升级成
        已确认；已确认版在库、推断版后到时，只记出现，不降级）。
        同槽位（2026-11-22，Gemini #4）：slot 相同、文本不同的活记忆
        是同一件事的新版本，按 provenance._resolve_conflict 裁决：
        supersede_old 就取代旧行再存新行；pending_new/keep_old/ask
        都不存新行（分别返回 PENDING/STALE/ASK）。
        """
        text = (text or "").strip()
        if not text:
            return {}
        if _contains_sensitive(text):
            # #1 敏感信息拦截：含密码/卡号/密钥的句子不存——账本和 SQLite
            # 都不写。返回非空字典（event=SENSITIVE）："记住"直存的调用方
            # 按"已处理"跳过 add_turn，不让秘密从另一条路再存进去；
            # save_turn 按 dropped 记回执、写明原因（见下面 SENSITIVE 分支）。
            return {
                "event": "SENSITIVE",
                "memory": text,
                "reason": "含敏感信息",
            }
        with self._lock:
            created_at = _now_iso()
            today = _shanghai_today().isoformat()
            slot = kwargs.get("slot")
            is_rule = bool(kwargs.get("is_rule", False))
            curated = bool(kwargs.get("curated", False))
            user_confirmed = source == "user"
            write_seq = _next_write_seq()
            decay_kind = kwargs.get("decay_kind") or _decay.classify_decay_kind(
                source, False, is_rule
            )
            # 同文去重（2026-11-22 Bug 2 治本）：只看归一化文本，不看来源——
            # 模型工具写 source="user"（已确认）、save_turn 写 source="inferred"
            # （未确认），来源不同不是两条事实。归一化比对要在 Python 里做
            # （_normalize_memory_text 剥标点/空白/主语，SQL 精确匹配认不出
            # “我住广州。”和“我住广州”是同一句），所以扫一遍活行逐条比。
            norm_text = _normalize_memory_text(text)
            if norm_text:
                for row in self._conn.execute(
                    "SELECT id, text, metadata_json FROM memories"
                    " WHERE voided=0 AND deleted=0 AND superseded=0"
                ).fetchall():
                    if _normalize_memory_text(row[1] or "") != norm_text:
                        continue
                    try:
                        old_meta = json.loads(row[2] or "{}")
                    except Exception:
                        old_meta = {}
                    if user_confirmed and not old_meta.get("user_confirmed"):
                        # 确认状态只往上提：推断版在库、已确认版后到时，
                        # 把已有行升级成已确认（source 改 "user"、确认位
                        # 置 1、decay_kind 改 stable——用户亲口确认的事实
                        # 应该是稳定的）。升级不记账本：复用那条路
                        # （_apply_correction_result）也没给复用记账本，
                        # 只在 decay_state 复活时记一笔 REVIVE，这里不复活，
                        # 不自作主张加新事件类型。
                        new_meta = dict(old_meta)
                        new_meta["source"] = "user"
                        new_meta["user_confirmed"] = True
                        new_meta["decay_kind"] = _decay.DECAY_KIND_STABLE
                        self._conn.execute(
                            "UPDATE memories SET metadata_json=?, source=?,"
                            " user_confirmed=1, decay_kind=? WHERE id=?",
                            (json.dumps(new_meta, ensure_ascii=False), "user",
                             _decay.DECAY_KIND_STABLE, row[0]),
                        )
                        self._conn.commit()
                        old_meta = new_meta
                    # 同一条事实又出现一次：记出现集合，够 3 天 + 3 会就晋升。
                    self._record_occurrence(row[0], old_meta, today, turn_id, text)
                    return {"id": row[0], "memory": text, "event": "UNCHANGED"}
            # 同槽位裁决：slot 相同、文本不同的活记忆是同一件事的新版本。
            # 先只裁决、不取代：pending/keep/ask 都直接返回不存；全是
            # supersede_old 时才继续存，存好新行再取代旧行（新 key 要
            # 先存在，supersede 事件的 current_key 才有东西可指）。
            to_supersede = []
            if slot:
                for row in self._conn.execute(
                    "SELECT id, text, metadata_json FROM memories"
                    " WHERE slot=? AND voided=0 AND deleted=0 AND superseded=0"
                    " AND text!=?",
                    (slot, text),
                ).fetchall():
                    old_meta = json.loads(row[2] or "{}")
                    new_meta_probe = {
                        "source": source,
                        "user_confirmed": user_confirmed,
                        "confidence": 1.0 if source == "user" else 0.7,
                        "created_at": created_at,
                        "write_seq": write_seq,
                        "slot": slot,
                    }
                    verdict = _provenance._resolve_conflict(old_meta, new_meta_probe)
                    action = (verdict or {}).get("action")
                    if action == "supersede_old":
                        to_supersede.append((row[0], row[1], old_meta))
                    elif action == "pending_new":
                        return {"event": "PENDING", "memory": text,
                                "reason": (verdict or {}).get("reason", ""),
                                "supersedes_id": row[0]}
                    elif action == "keep_old":
                        return {"event": "STALE", "memory": text,
                                "reason": (verdict or {}).get("reason", ""),
                                "supersedes_id": row[0]}
                    else:  # ask
                        return {"event": "ASK", "memory": text,
                                "reason": (verdict or {}).get("reason", ""),
                                "supersedes_id": row[0]}
            metadata = {
                "source": source,
                "confidence": 1.0 if source == "user" else 0.7,
                # confirmed 不存在这里：它是 user_confirmed 的短名字，
                # 读时由 _normalize_metadata 现算（和 mem0 的规矩一致）；
                # 存两份以后会打架，只留 user_confirmed 一个真相。
                "user_confirmed": user_confirmed,
                "created_at": created_at,
                "write_seq": write_seq,
                "turn_id": turn_id,
                "slot": slot,
                "is_rule": is_rule,
                "curated": curated,
                "decay_kind": decay_kind,
                "importance": kwargs.get("importance", "normal"),
                "event_date": event_date or resolve_event_date(text),
                "last_used_at": created_at,
                "decay_state": _decay.DECAY_STATE_VERBATIM,
            }
            dates, sessions, _added = _decay.record_seen_occurrence(
                None, None, today, turn_id
            )
            metadata["seen_dates"] = _decay._dump_seen_list(dates)
            metadata["seen_sessions"] = _decay._dump_seen_list(sessions)
            key = self._append_new_memory(text, metadata)
            if key is None:
                return {}
            for old_id, old_text, old_meta in to_supersede:
                try:
                    self._supersede_row(old_id, old_text, old_meta, key, text)
                except Exception:
                    logger.warning("槽位取代记账失败：旧行未取代。", exc_info=True)
            # 直存不走 save_turn：刚写的新记忆不再属于上一轮，撤回标记清掉，
            # 否则下一次“撤回”会误作废上一轮（2026-11-21 Gemini 第三轮真 Bug）。
            # save_turn 内部也调这里，但它在循环结束后才重设标记，顺序不冲突。
            self._last_stored_turn_id = None
            return {"id": key, "memory": text, "event": "ADD"}

    def _append_new_memory(self, text, metadata):
        """写一条新记忆：先写账本 add，再写 SQLite 行。返回 key；
        账本写不进去返回 None（索引那步也不写）。"""
        try:
            self._ledger_append_add(text, metadata)
        except Exception:
            logger.warning("账本写不进去：本次记忆没存。", exc_info=True)
            return None
        try:
            vec = self._embed_text(text)
        except Exception:
            vec = None
        key = _ledger.ledger_event_key(text, metadata.get("created_at"))
        self._insert_memory_row(key, text, metadata, vec)
        return key

    def _supersede_row(self, old_id, old_text, old_meta, new_key, new_text):
        """把旧行标成被取代：先标 SQLite superseded=1，再写账本 supersede 事件。
        账本写失败时把 SQLite 的标记撤回来再抛——两边要一致，账本是真相。
        old_id 就是旧 key（SQLite id 就是账本 key）；new_key 是新记忆的 key。
        """
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET superseded=1 WHERE id=?", (old_id,)
            )
            self._conn.commit()
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": _ledger.LEDGER_EVENT_SUPERSEDE,
                     "memory_id": new_key,
                     "supersedes_id": old_id,
                     "old_text": old_text,
                     "text": new_text,
                     "key": old_id,
                     "current_key": new_key},
                    lock=self._ledger_lock,
                )
            except Exception:
                self._conn.execute(
                    "UPDATE memories SET superseded=0 WHERE id=?", (old_id,)
                )
                self._conn.commit()
                raise

    def _record_occurrence(self, memory_id, metadata, today, turn_id, text=None):
        """同一条事实又出现一次：记进出现集合（不同天、不同会话），
        够 3 天 + 3 会就晋升成稳定事实（decay.py 的规矩）。
        同时把 last_used_at 拨到现在（刚被提到 = 刚被用上）。
        返回有没有晋升。调用方在同文去重（UNCHANGED）那条路上调。
        text 没传时从库里取（晋升事件要带原话）。
        """
        meta = dict(metadata or {})
        if text is None:
            try:
                row = self._conn.execute(
                    "SELECT text FROM memories WHERE id=?", (memory_id,)
                ).fetchone()
                text = (row or [""])[0] or ""
            except Exception:
                text = ""
        dates, sessions, _added = _decay.record_seen_occurrence(
            meta.get("seen_dates"), meta.get("seen_sessions"), today, turn_id
        )
        meta["seen_dates"] = _decay._dump_seen_list(dates)
        meta["seen_sessions"] = _decay._dump_seen_list(sessions)
        meta["last_used_at"] = _now_iso()
        promoted = False
        if meta.get("decay_kind") != _decay.DECAY_KIND_STABLE and _decay.should_promote(
            dates, sessions,
            is_core=bool(_provenance._is_core_memory(meta)),
            tone=meta.get("tone"),
        ):
            meta["decay_kind"] = _decay.DECAY_KIND_STABLE
            promoted = True
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": _ledger.LEDGER_EVENT_PROMOTE,
                     "memory_id": memory_id, "key": memory_id,
                     "text": text,
                     "meta": dict(meta)},
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("晋升记账失败。", exc_info=True)
        self._conn.execute(
            "UPDATE memories SET metadata_json=?, decay_kind=? WHERE id=?",
            (json.dumps(meta, ensure_ascii=False), meta.get("decay_kind"), memory_id),
        )
        self._conn.commit()
        return promoted

    def _find_same_slot_conflicts(self, slot, value):
        """找同槽位、不同值的活记忆：[(id, text, metadata)]。
        按每条原文重新认槽位（老行存的时候还没打 slot 标，只能从文本认）。"""
        with self._lock:
            norm_value = _normalize_memory_text(value or "")
            out = []
            for row in self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall():
                old_slot, old_value = _infer_fact_slot(row[1] or "")
                if old_slot != slot or old_value is None:
                    continue
                if _normalize_memory_text(old_value) == norm_value:
                    continue
                try:
                    old_meta = json.loads(row[2] or "{}")
                except Exception:
                    old_meta = {}
                out.append((row[0], row[1] or "", old_meta))
            return out

    def _set_memory_slot(self, memory_id, slot):
        """给一条记忆补槽位标（列 + metadata_json 里都写），以后按槽位
        找它就不用再从文本现认。找不到行、或已有槽位标时什么都不做，不抛异常。"""
        if not memory_id or not slot:
            return
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata_json FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                return
            try:
                meta = json.loads(row[0] or "{}")
            except Exception:
                meta = {}
            if meta.get("slot"):
                return
            meta["slot"] = slot
            self._conn.execute(
                "UPDATE memories SET slot=?, metadata_json=? WHERE id=?",
                (slot, json.dumps(meta, ensure_ascii=False), memory_id),
            )
            self._conn.commit()

    def _complete_correction_fragment(self, fact):
        """碎片纠正补句（确定性版，不调模型）：新说法是个碎片值（“深圳”）
        时，看库里是不是恰好只有一条单值属性记忆——是就把它的旧值换
        成碎片值拼回整句（“我住北京”+“深圳”→“我住深圳”）；不是
        （没有、或有多条、或值含糊）就原样返回，调用方走老路，绝不硬编。
        只认 residence / user_name / agent_name 三个槽。"""
        if not isinstance(fact, str) or not fact.strip():
            return fact
        value = _detect_correction_fragment(fact)
        if value is None:
            return fact
        # 碎片值收尾：去句首的引导词/住/在（“应该是深圳”→“深圳”，
        # “住深圳了”→“深圳了”）、去句尾语气词；带事实动词的不是值
        # （“喜欢喝茶”），不补。
        value = (value or "").strip()
        for prefix in ("应该是", "其实是", "其实"):
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
                break
        if value[:1] in ("住", "在", "是"):
            value = value[1:].strip()
        value = value.rstrip("了呢啊吧哦嘛").strip()
        if not value or value in _VAGUE_FRAGMENT_VALUES:
            return fact
        if len(value) > 20:
            return fact
        if any(ch in value for ch in ("喜", "欢", "爱", "吃", "喝", "叫",
                                       "去", "做", "买", "看", "听", "玩",
                                       "想", "要", "会", "能")):
            return fact
        with self._lock:
            cands = []
            for row in self._conn.execute(
                "SELECT id, text FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall():
                slot, old_value = _infer_fact_slot(row[1] or "")
                if slot is None or old_value is None:
                    continue
                if _normalize_memory_text(old_value) == _normalize_memory_text(value):
                    continue
                cands.append((row[1] or "", old_value))
        if len(cands) != 1:
            return fact
        old_text, old_value = cands[0]
        if old_value not in old_text:
            return fact
        return old_text.replace(old_value, value, 1)

    def _retry_correction_by_slot(self, correction, fact):
        """纠错分不清改哪条（ambiguous）时，按单值槽位再筛一次：新说法
        带槽位、且候选里恰好只有一条同槽位（值不同）时，就改那条。
        筛不出来返回 None（调用方保留 ambiguous，等 BOOS 挑）。"""
        slot, value = _infer_fact_slot(fact)
        if slot is None or value is None:
            return None
        cands = (correction or {}).get("candidates") or []
        same_slot = []
        for cand in cands:
            text = (cand or {}).get("text") if isinstance(cand, dict) else None
            if not text:
                continue
            cand_slot, cand_value = _infer_fact_slot(text)
            if cand_slot != slot or cand_value is None:
                continue
            if _normalize_memory_text(cand_value) == _normalize_memory_text(value):
                continue
            same_slot.append(cand)
        if len(same_slot) != 1:
            return None
        old_text = same_slot[0].get("text") or ""
        if not old_text:
            return None
        retry = self.correct_memory(old_text, fact, exclude_identical_to_new=True)
        if (retry or {}).get("status") == "updated":
            return retry
        return None

    def _apply_same_slot_update(self, sentence, slot, value, receipt,
                                turn_id, event_date):
        """同槽位新值进库前的裁决（save_turn 逐句循环里调）：
        - 没同槽位旧值 → 返回 False，调用方照常存；
        - 有旧值：逐条按 provenance._resolve_conflict 裁决；只要有一条
          不是 supersede_old，就不存新句（挂起/留旧），返回 True；
        - 全是 supersede_old → 逐条调 correct_memory 取代；都换好了
          返回 True（回执记 stored + correction）；correct_memory 没换
          成的那条，改走直存 + _supersede_row 兜底（裁决已说该换，
          不能因为候选检索没找着就并存两条）。
        """
        conflicts = self._find_same_slot_conflicts(slot, value)
        if not conflicts:
            return False
        new_probe = {
            "source": "inferred",
            "user_confirmed": False,
            "confidence": 0.7,
            "created_at": _now_iso(),
            "write_seq": _next_write_seq(),
            "slot": slot,
        }
        to_supersede = []
        for old_id, old_text, old_meta in conflicts:
            verdict = _provenance._resolve_conflict(old_meta, new_probe)
            action = (verdict or {}).get("action")
            if action == "supersede_old":
                to_supersede.append((old_id, old_text, old_meta))
            elif action == "pending_new":
                receipt["dropped"].append(
                    {"text": sentence, "reason": "槽位冲突待确认"}
                )
                return True
            else:  # keep_old / ask：新说法不存，旧的留着。
                receipt["dropped"].append(
                    {"text": sentence, "reason": "槽位冲突未裁决"}
                )
                return True
        # 全是 supersede_old：逐条取代。
        first_outcome = None
        new_key = None
        remaining = []
        for old_id, old_text, old_meta in to_supersede:
            outcome = self.correct_memory(
                old_text, sentence, exclude_identical_to_new=True
            )
            if (outcome or {}).get("status") == "updated":
                if first_outcome is None:
                    first_outcome = outcome
                new_key = (outcome or {}).get("memory_id") or new_key
            else:
                remaining.append((old_id, old_text, old_meta))
        if remaining:
            # correct_memory 没换成的（候选检索没找着旧条等）：新说法先
            # 直存，再按裁决直接取代旧行——裁决已说该换，不并存两条。
            if new_key is None:
                saved = self.add_memory_text(
                    sentence, source="inferred", turn_id=turn_id,
                    event_date=event_date, decay_kind="episodic", slot=slot,
                )
                if (saved or {}).get("event") != "ADD":
                    receipt["dropped"].append(
                        {"text": sentence, "reason": "账本写失败"}
                    )
                    return True
                new_key = saved.get("id")
            for old_id, old_text, old_meta in remaining:
                try:
                    self._supersede_row(old_id, old_text, old_meta, new_key, sentence)
                except Exception:
                    logger.warning("槽位取代记账失败：旧行未取代。", exc_info=True)
            if first_outcome is None:
                first_outcome = {
                    "status": "updated", "memory_id": new_key,
                    "old_text": remaining[0][1], "new_text": sentence,
                    "candidates": [
                        {"id": oid, "text": ot} for oid, ot, _m in remaining
                    ],
                    "reason": "已更正", "layer": "slot",
                    "resolution": "updated",
                    "audit": {"source": "user", "method": "slot_conflict"},
                }
        if first_outcome is not None:
            slot2, _v = _infer_fact_slot(sentence)
            if slot2:
                self._set_memory_slot(first_outcome.get("memory_id"), slot2)
            receipt["stored"].append(sentence)
            receipt["results"].append(
                {"id": first_outcome.get("memory_id"), "memory": sentence,
                 "event": "SUPERSEDE",
                 "old_text": first_outcome.get("old_text"),
                 "supersedes_id": ((first_outcome.get("candidates") or [{}])[0] or {}).get("id")}
            )
            if "correction" not in receipt:
                receipt["correction"] = first_outcome
            return True
        return False

    def save_turn(self, messages, turn_id=None):
        """一轮对话的收尾：同步把用户亲口说的事实原样存下来，留回执。

        和 mem0 那条路最大的不同：没有待定区、没有 60 秒定时器、
        没有“提取返回空就当成功”。调完这个函数，事实要么在账本里、
        要么在回执里写明为什么没存——绝不悄悄丢。
        同一个 turn_id 只存一次（web 的 ask() 和 run_question() 会各调
        一次，第二次直接返回上次的回执，不重复存）。
        """
        # 没传 turn_id 时生成内部兜底 ID（2026-11-21 Gemini 第三轮）：多句轮次的
        # 每句都带同一个 turn_id，撤回才能整轮作废，否则退化成只作废最后 1 条。
        # _next_write_seq() 进程内严格递增，每次调用都不同，去重逻辑不受影响。
        turn_id = turn_id or f"auto-{_next_write_seq()}"
        if turn_id and turn_id in self._seen_turn_ids:
            return {"turn_id": turn_id, "duplicate_turn": True, "results": []}
        if turn_id:
            self._seen_turn_ids.add(turn_id)
        latest_user = ""
        for m in reversed(messages or []):
            if (m or {}).get("role") == "user":
                latest_user = (m or {}).get("content") or ""
                break
        receipt = {
            "turn_id": turn_id,
            "considered": [],
            "stored": [],
            "dropped": [],
            "retracted": False,
            "dropped_count": 0,
            "results": [],
        }
        if not (latest_user or "").strip():
            return receipt
        # 撤回：光秃秃的整句信号，把上一轮刚存的那条作废（软删进账本）。
        if latest_user.strip().rstrip("。！？.!?") in _BARE_RETRACTION:
            dropped = self._void_last_stored()
            receipt["retracted"] = True
            receipt["dropped_count"] = dropped
            return receipt
        today = _shanghai_today()
        # 聊天纠错（2026-11-22 correct_memory 接线）：本轮最后一条 user 消息
        # 里有明确纠错说法时（见 correction_text._detect_correction），先调
        # correct_memory 改旧记忆，结果放进回执的 "correction" 键（web 层
        # 已经在读这个键，供页面展示，见 web/server.py）。
        #   updated/unchanged/ambiguous/failed：纠错已处理，新说法不再当
        #     事实重存，只处理纠错句之外的部分；
        #   not_found：没找到要改的旧记忆，新说法当新记忆存下来。
        # 没纠错时走正常逐句循环，原来的键不动。
        correction = None
        sentences = _split_sentences(latest_user)
        detected = _detect_correction(latest_user)
        if detected is not None:
            fact = (detected.get("fact") or "").strip()
            # 叠起来的纠错说法（“不对，你记错了，我住上海”）：_detect_correction
            # 取最早的说法（“不对”），fact 里还带着“你记错了，”——再剥一层，
            # 剥到只剩新说法为止（最多两层，防循环）。
            for _ in range(2):
                inner = _detect_correction(fact)
                if inner is None:
                    break
                inner_fact = (inner.get("fact") or "").strip()
                if not inner_fact or inner_fact == fact:
                    break
                fact = inner_fact
            if fact:
                # 碎片新说法（“不对，应该是深圳”→“深圳”）先按槽位补成整句；
                # 补不出来就原样走 correct_memory（找不到旧记忆时当新记忆存）。
                fact = self._complete_correction_fragment(fact)
                # 当下的想法/问句不纠正（“我现在想喝茶了”不是在改一条记忆）：
                # 检测到了也不走纠正分支，留给下面的正常循环按老规矩处理
                # （当下的想法会被 _looks_like_fact 拦下、不存）。
                if _looks_like_fact(fact)[0]:
                    correction = self.correct_memory(
                        fact, fact, exclude_identical_to_new=True
                    )
                    if (correction or {}).get("status") == "ambiguous":
                        # 分不清改哪条时按槽位再筛一次：新说法带单值槽位、
                        # 且候选里恰好只有一条同槽位时，就改那条，不用问。
                        retry = self._retry_correction_by_slot(correction, fact)
                        if retry is not None:
                            correction = retry
                    if (correction or {}).get("status") == "updated":
                        # 纠正换来的新记忆带上槽位标（以后按槽位找得到它）。
                        slot, _value = _infer_fact_slot(fact)
                        if slot:
                            self._set_memory_slot(
                                (correction or {}).get("memory_id"), slot
                            )
                    kept = _strip_correction_sentences(latest_user)
                    if (correction or {}).get("status") == "not_found":
                        # 没找到旧记忆：新说法当新记忆存（带上纠错句之外的部分）。
                        # 逐句循环里的槽位更新还会再拦一次（同槽位旧值在时
                        # 替它走 correct_memory，不并存两条）。
                        sentences = kept + [fact]
                    else:
                        sentences = kept
        live_texts = None  # 懒加载：只有出现“记住”前缀时才查一次在库文本
        for sentence in sentences:
            receipt["considered"].append(sentence)
            # “记住”前缀先剥（#8）：剥完再判事实、再算日期——“记住”两字
            # 本身不是记忆内容。剥完为空（光秃秃一句“记住”）直接丢掉。
            stripped = _strip_remember_prefix(sentence)
            stripped_remember = stripped != sentence
            sentence = stripped
            if not sentence:
                receipt["dropped"].append(
                    {"text": receipt["considered"][-1], "reason": "记住指令本身，无内容"}
                )
                continue
            # 确认词前缀（第三轮 #8）：剥掉“是的/对的/没错/可以/好的”，
            # 再判剩下的；剥完没内容就是光秃秃一句确认词，直接丢掉。
            # “是的删掉”剥完是“删掉”，走下面的删除指令拦下；
            # “是的，我喜欢喝茶”剥完是“我喜欢喝茶”，照常存。
            stripped = _strip_confirmation_prefix(sentence)
            if stripped != sentence:
                sentence = stripped
                if not sentence:
                    receipt["dropped"].append(
                        {"text": receipt["considered"][-1], "reason": "确认词"}
                    )
                    continue
            if stripped_remember:
                # 直接路径（“记住……”命令）已经存过同一句时不再存第二遍：
                # 同一文本两行是重复根（2026-11-20 根治过），检索层虽会
                # 按文本聚合，账本里也不该多一条 ADD。
                if live_texts is None:
                    live_texts = {m["text"] for m in self._all_live_memories()}
                if sentence in live_texts:
                    receipt["dropped"].append({"text": sentence, "reason": "已在库里"})
                    continue
            ok, reason = _looks_like_fact(sentence)
            if not ok:
                receipt["dropped"].append({"text": sentence, "reason": reason})
                continue
            # 存原话：去掉句尾的 。！（？的已经当问句 drop 了，不会到这里）。
            sentence = sentence.rstrip("。！")
            event_date = resolve_event_date(sentence, today)
            # 刚被纠正取代的旧说法（同一轮里先肯定、后纠正）不再重存。
            if (
                correction is not None
                and (correction or {}).get("status") == "updated"
                and (correction or {}).get("old_text")
                and _normalize_memory_text(sentence)
                == _normalize_memory_text((correction or {}).get("old_text") or "")
            ):
                receipt["dropped"].append({"text": sentence, "reason": "已被纠正取代"})
                continue
            # 同槽位隐式更新（2026-11-22 通用纠正）：新事实带单值槽位、
            # 库里已有同槽位不同值时，按 _resolve_conflict 裁决、走
            # correct_memory 取代旧的；没冲突时照常存，顺手把槽位标上。
            slot, slot_value = _infer_fact_slot(sentence)
            if slot is not None and self._apply_same_slot_update(
                sentence, slot, slot_value, receipt, turn_id, event_date
            ):
                continue
            # 来源标 inferred（提取所得）：原样存的事实是从对话里提出来的，
            # 出处完整（turn_id + created_at），和 DeepSeek 提取同一档，
            # 只是没经过模型改写、保真度更高。unconfirmed 是留给“完全没来源
            # 信息”的旧记忆的桶（2026-11-09 之前）；标错了问答提示词会把它
            # 当没来源处理、问起来只说不知道——这就是第一轮验收回答卡住的原因。
            saved = self.add_memory_text(
                sentence, source="inferred", turn_id=turn_id,
                event_date=event_date, decay_kind="episodic",
                **({"slot": slot} if slot else {}),
            )
            if (saved or {}).get("event") == "ADD":
                if live_texts is not None:
                    live_texts.add(sentence)
                receipt["stored"].append(sentence)
                receipt["results"].append(
                    {"id": saved.get("id"), "memory": sentence, "event": "ADD"}
                )
            elif (saved or {}).get("event") == "UNCHANGED":
                receipt["dropped"].append({"text": sentence, "reason": "已在库里"})
            elif (saved or {}).get("event") == "SENSITIVE":
                receipt["dropped"].append({"text": sentence, "reason": "含敏感信息"})
            elif (saved or {}).get("event") == "PENDING":
                # 同槽位裁决：新说法可信度低于旧的，挂起来待确认，不存。
                receipt["dropped"].append({"text": sentence, "reason": "槽位冲突待确认"})
            elif (saved or {}).get("event") in ("STALE", "ASK"):
                # 同槽位裁决：新说法比旧的还早（STALE），或分不出先后（ASK），不存。
                receipt["dropped"].append({"text": sentence, "reason": "槽位冲突未裁决"})
            else:
                receipt["dropped"].append({"text": sentence, "reason": "账本写失败"})
        if receipt["stored"]:
            with self._lock:
                self._last_stored_turn_id = turn_id
        if correction is not None and (
            "correction" not in receipt
            or (correction or {}).get("status") != "not_found"
        ):
            # 显式纠错优先；但它报 not_found、而逐句循环里的槽位更新已经
            # 真换了旧记忆时，留循环那边的 updated，不拿 not_found 盖掉它。
            receipt["correction"] = correction
            if (correction or {}).get("status") == "updated":
                # 纠错存的新记忆属于本轮：撤回标记拨到本轮，
                # 下一句“撤回”作废的是这条新记忆，不是上一轮的。
                with self._lock:
                    self._last_stored_turn_id = turn_id
        return receipt

    def _find_correction_candidates(self, old_fact, new_fact=None, top_k=10,
                                        exclude_identical_to_new=False):
        """找纠错候选：要改的旧记忆（和 mem0_store._find_correction_candidates
        同款两路 + 兜底）。

        1. 语义 + BM25（_search_inner 两路都走了，含 0.4 资格线）；
        2. 兜底（直接检索为空时才走，免得往池子里加噪音）：最近上下文
           （note_context 记的本轮检索到的 id）+ 同核心槽位
           （extract_core_slots 交集，数字/日期/二字词）——“住上海”
           vs“住北京”余弦低、关键词不撞时，靠这条兜底找着旧条
           （mem0 的 Bug 6 场景）。
        返回活记忆的 id 列表（已确认在前、同类里新的在前）。
        exclude_identical_to_new 时跳过与 new_fact 相同的条目：纠错是来
        取代旧说法的，库里已有的新说法本身不是目标；不跳过的话，exact
        会先命中它而直接返回 unchanged，漏掉真正该被取代的旧说法。
        """
        with self._lock:
            old_fact = (old_fact or "").strip()
            if not old_fact:
                return []
            found = []
            seen = set()
            try:
                for hit in self._search_inner(old_fact, top_k=top_k * 4):
                    mid = hit.get("id")
                    if mid and mid not in seen:
                        seen.add(mid)
                        found.append(mid)
            except Exception:
                logger.warning("纠错：语义找候选失败。", exc_info=True)
            if not found:
                # 兜底：最近上下文 + 同核心槽位。
                try:
                    for mid in list(getattr(self, "_last_context_ids", []) or []):
                        if mid and mid not in seen:
                            row = self._conn.execute(
                                "SELECT id FROM memories WHERE id=? AND voided=0"
                                " AND deleted=0 AND superseded=0",
                                (mid,),
                            ).fetchone()
                            if row is not None:
                                seen.add(mid)
                                found.append(mid)
                except Exception:
                    pass
                try:
                    slots = extract_core_slots(new_fact or old_fact)
                    if slots:
                        for row in self._conn.execute(
                            "SELECT id, text FROM memories"
                            " WHERE voided=0 AND deleted=0 AND superseded=0"
                        ).fetchall():
                            if row[0] in seen:
                                continue
                            if extract_core_slots(row[1] or "") & slots:
                                seen.add(row[0])
                                found.append(row[0])
                except Exception:
                    pass
            if exclude_identical_to_new and new_fact:
                # 跳过与新说法相同的条目（见 docstring）：纠错是来取代旧
                # 说法的，库里已有的新说法本身不是目标。
                norm_new = _normalize_memory_text(new_fact)
                kept = []
                for mid in found:
                    row = self._conn.execute(
                        "SELECT text FROM memories WHERE id=?", (mid,)
                    ).fetchone()
                    if row is not None and _normalize_memory_text(
                        row[0] or ""
                    ) == norm_new:
                        continue
                    kept.append(mid)
                found = kept
            # 排：已确认的在前，同类里新的在前。
            def _cand_key(mid):
                row = self._conn.execute(
                    "SELECT metadata_json, created_at FROM memories WHERE id=?", (mid,)
                ).fetchone()
                meta = json.loads((row or [None, None])[0] or "{}")
                confirmed = 1 if meta.get("user_confirmed") else 0
                return (-confirmed, -(row[1] or ""))
            try:
                found.sort(key=_cand_key)
            except Exception:
                pass
            return found

    def _candidate_text(self, memory_id):
        """按 ID 查一条记忆的原文；查不到返回 None（调用方兜底，不抛异常）。"""
        try:
            row = self._conn.execute(
                "SELECT text FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        except Exception:
            return None
        return (row or [None])[0] if row else None

    def _candidates_with_text(self, candidates):
        """把纠错候选统一成 [{"id", "text"}]（2026-11-22 Bug 1 治本）。

        _find_correction_candidates 内部返回的是内存 ID（字符串）列表，
        调用方（main.py 的打印、网页的候选列表）要的是带文本的字典——
        以前直接把 ID 串交出去，打印层按字典 .get('text') 就崩成 500。
        字典候选原样过（缺 text 时按 ID 补查）；字符串候选按 ID 查库
        补文本，查不到就把 ID 本身当 text；绝不抛异常。
        """
        out = []
        for cand in candidates or []:
            if isinstance(cand, dict):
                cid = cand.get("id")
                text = cand.get("text")
                if not text and cid:
                    text = self._candidate_text(cid)
                out.append({"id": cid, "text": text or (cid or "")})
            elif isinstance(cand, str):
                out.append({"id": cand, "text": self._candidate_text(cand) or cand})
            else:
                out.append({"id": cand, "text": str(cand)})
        return out

    def _apply_correction_result(self, new_fact, old_id, old_text, old_meta,
                                 *, same_text, candidates, status="updated",
                                 resolution=None, audit=None, source="user",
                                 layer="single"):
        """纠错收尾（2026-11-22）：按状态把结果写进账本和 SQLite，返回
        回执。纠错是追加式——旧记忆不改原文、不删：新说法当一条新记忆
        存（来源 user、已确认），账本记 supersede，旧行标 superseded=1。
        same_text 时新说法和旧的一样，不用重存：只记一笔出现。
        库里已有和新说法相同的活记忆时直接复用那条（刷新 last_used_at，
        dormant / delete_due 改回 verbatim），不再新插，防重复行。
        layer 哪一层定的（mem0 同款词）：exact（原样对上）/ single
        （单候选过安全网）；调用方显式传，不从 same_text 推。
        """
        with self._lock:
            # 回执里的 candidates 统一成带文本的字典（2026-11-22 Bug 1）：
            # 传进来的是 _find_correction_candidates 给的 ID 串列表，
            # 这里转完再返回，调用方不用再猜格式。
            candidates = self._candidates_with_text(candidates)
            if status == "unchanged":
                self._record_occurrence(
                    old_id, old_meta, _shanghai_today().isoformat(), None, old_text
                )
                return {
                    "status": "unchanged", "memory_id": old_id,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": candidates or [],
                    "reason": "已是该内容，无需修改",
                    "layer": layer, "resolution": resolution or "exact",
                    "audit": audit or {"source": "user", "method": "unchanged"},
                }
            # 库里已有和新说法相同的活记忆时直接复用，不再新插一条
            #（否则新旧说法同时在库里时会留下两条一样的活记录）。
            # 复用时刷新 last_used_at，dormant / delete_due 顺便改回
            # verbatim；_supersede_row 的目标就是这条已有记录。
            existing_id = None
            existing_meta = None
            existing_text = None
            for _row in self._conn.execute(
                "SELECT id, text, metadata_json FROM memories WHERE voided=0"
                " AND deleted=0 AND superseded=0 AND id != ?",
                (old_id,),
            ).fetchall():
                if _normalize_memory_text(_row[1] or "") == _normalize_memory_text(new_fact):
                    existing_id = _row[0]
                    existing_text = _row[1] or ""
                    existing_meta = json.loads(_row[2] or "{}")
                    break
            if existing_id is not None:
                revived = existing_meta.get("decay_state") in (
                    _decay.DECAY_STATE_DORMANT, _decay.DECAY_STATE_DELETE_DUE,
                )
                existing_meta["last_used_at"] = _now_iso()
                if revived:
                    existing_meta["decay_state"] = _decay.DECAY_STATE_VERBATIM
                self._conn.execute(
                    "UPDATE memories SET metadata_json=? WHERE id=?",
                    (json.dumps(existing_meta, ensure_ascii=False), existing_id),
                )
                if revived:
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {"type": _ledger.LEDGER_EVENT_REVIVE,
                             "memory_id": existing_id, "key": existing_id,
                             "text": existing_text},
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning("复活记账失败。", exc_info=True)
                new_key = existing_id
            else:
                created_at = _now_iso()
                write_seq = _next_write_seq()
                today = _shanghai_today().isoformat()
                metadata = {
                    "source": source,
                    "confidence": 1.0,
                    "user_confirmed": True,
                    "created_at": created_at,
                    "write_seq": write_seq,
                    "turn_id": None,
                    "slot": (old_meta or {}).get("slot"),
                    "is_rule": bool((old_meta or {}).get("is_rule")),
                    "curated": bool((old_meta or {}).get("curated")),
                    "decay_kind": _decay.classify_decay_kind(
                        source, False, bool((old_meta or {}).get("is_rule"))
                    ),
                    "importance": (old_meta or {}).get("importance", "normal"),
                    "event_date": resolve_event_date(new_fact),
                    "last_used_at": created_at,
                    "decay_state": _decay.DECAY_STATE_VERBATIM,
                }
                dates, sessions, _added = _decay.record_seen_occurrence(None, None, today, None)
                metadata["seen_dates"] = _decay._dump_seen_list(dates)
                metadata["seen_sessions"] = _decay._dump_seen_list(sessions)
                try:
                    new_key = self._append_new_memory(new_fact, metadata)
                except Exception:
                    return {
                        "status": "failed", "memory_id": old_id,
                        "old_text": old_text, "new_text": new_fact,
                        "candidates": candidates or [],
                        "reason": "新记忆写入失败", "layer": "none",
                        "resolution": resolution or "failed",
                        "audit": audit or {"source": source, "method": "correct_memory"},
                    }
                if new_key is None:
                    return {
                        "status": "failed", "memory_id": old_id,
                        "old_text": old_text, "new_text": new_fact,
                        "candidates": candidates or [],
                        "reason": "账本写失败", "layer": "none",
                        "resolution": resolution or "failed",
                        "audit": audit or {"source": source, "method": "correct_memory"},
                    }
            try:
                self._supersede_row(old_id, old_text, old_meta, new_key, new_fact)
            except Exception:
                logger.warning("纠错取代记账失败：旧行未取代。", exc_info=True)
                return {
                    "status": "failed", "memory_id": new_key,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": candidates or [],
                    "reason": "新记忆已存，但旧记忆未被取代",
                    "layer": "none",
                    "resolution": "failed",
                    "audit": audit or {"source": source, "method": "correct_memory"},
                }
            return {
                "status": status, "memory_id": new_key,
                "old_text": old_text, "new_text": new_fact,
                "candidates": candidates or [],
                "reason": "已更正",
                "layer": layer,
                "resolution": resolution or "updated",
                "audit": audit or {"source": source, "method": "correct_memory"},
            }

    def correct_memory(self, old_fact, new_fact, exclude_identical_to_new=True):
        """纠错：把 old_fact 指的那条记忆改成 new_fact（整合方案 §6.3）。

        流程（和 mem0_store.correct_memory 一致）：
        先找候选（语义 + 关键词/同槽位 + 近期同一轮）；
        旧说法原样在库里 → 直接改那条；
        只有一条候选 → 过相似度安全网（余弦 ≥ 0.70 才改，否则 ambiguous，
        免得改错不相干的记忆；向量算不出来时跳过安全网，交由调用方兜底）；
        多条候选、没有原样 → ambiguous（不猜，留给调用方问）；
        一条没有 → not_found（调用方按新记忆存）；
        新说法和旧的一样 → unchanged；新说法含敏感信息 → failed。
        纠错是追加式（整合方案）：旧记忆不改原文、不删，记 supersede，
        新说法另存一条。返回 mem0 同款回执：
        {status, memory_id, old_text, new_text, candidates, reason,
         layer, resolution, audit}。
        candidates 是带 id 和 text 的字典列表（2026-11-22 Bug 1：以前是
        光秃秃的 ID 串，打印层按字典取 text 就崩成 500）。
        """
        with self._lock:
            new_fact = (new_fact or "").strip()
            old_fact = (old_fact or "").strip()
            if _contains_sensitive(new_fact):
                return {
                    "status": "failed", "memory_id": None,
                    "old_text": old_fact, "new_text": new_fact,
                    "candidates": [],
                    "reason": "新说法含敏感信息", "layer": "none",
                    "resolution": "blocked",
                    "audit": {"source": "user", "method": "correct_memory"},
                }
            candidates = self._find_correction_candidates(
                old_fact, new_fact=new_fact,
                exclude_identical_to_new=exclude_identical_to_new,
            )
            if not candidates:
                if _normalize_memory_text(old_fact) == _normalize_memory_text(new_fact):
                    # 新旧说法相同：候选过滤已跳过与新说法相同的条目；库里
                    # 已是这句、又没有别的冲突记忆时，无事可做，报 unchanged
                    #（而不是 not_found——“已经是对的”不是“找不到”）。
                    for _row in self._conn.execute(
                        "SELECT id, text, metadata_json FROM memories WHERE voided=0"
                        " AND deleted=0 AND superseded=0"
                    ).fetchall():
                        if _normalize_memory_text(_row[1] or "") == _normalize_memory_text(new_fact):
                            return self._apply_correction_result(
                                new_fact, _row[0], _row[1],
                                json.loads(_row[2] or "{}"),
                                same_text=True, candidates=[], status="unchanged",
                                resolution="exact", layer="exact",
                                audit={"source": "user", "method": "exact_match"},
                            )
                return {
                    "status": "not_found", "memory_id": None,
                    "old_text": old_fact, "new_text": new_fact,
                    "candidates": [],
                    "reason": "未找到可纠正的记忆", "layer": "none",
                    "resolution": "not_found",
                    "audit": {"source": "user", "method": "correct_memory"},
                }
            # 原样匹配：旧说法原样在库里，直接改那条（不用过安全网）。
            exact = None
            for mid in candidates:
                row = self._conn.execute(
                    "SELECT text, metadata_json FROM memories WHERE id=?", (mid,)
                ).fetchone()
                if row and (row[0] or "").strip() == old_fact:
                    exact = (mid, row[0], json.loads(row[1] or "{}"))
                    break
            if exact is not None:
                mid, old_text, old_meta = exact
                same = _normalize_memory_text(new_fact) == _normalize_memory_text(old_text)
                if same:
                    return self._apply_correction_result(
                        new_fact, mid, old_text, old_meta, same_text=True,
                        candidates=candidates, status="unchanged",
                        resolution="exact", layer="exact",
                        audit={"source": "user", "method": "exact_match"},
                    )
                return self._apply_correction_result(
                    new_fact, mid, old_text, old_meta, same_text=False,
                    candidates=candidates, status="updated",
                    resolution="exact", layer="exact",
                    audit={"source": "user", "method": "exact_match"},
                )
            if len(candidates) > 1:
                return {
                    "status": "ambiguous", "memory_id": None,
                    "old_text": old_fact, "new_text": new_fact,
                    "candidates": self._candidates_with_text(candidates),
                    "reason": "多条候选，无法确定改哪一条",
                    "layer": "none", "resolution": "ambiguous",
                    "audit": {"source": "user", "method": "correct_memory"},
                }
            # 只有一条候选：过相似度安全网。
            mid = candidates[0]
            row = self._conn.execute(
                "SELECT text, metadata_json, embedding FROM memories WHERE id=?", (mid,)
            ).fetchone()
            if row is None:
                return {
                    "status": "not_found", "memory_id": None,
                    "old_text": old_fact, "new_text": new_fact,
                    "candidates": self._candidates_with_text(candidates),
                    "reason": "候选记忆已不在库里", "layer": "none",
                    "resolution": "not_found",
                    "audit": {"source": "user", "method": "correct_memory"},
                }
            old_text, old_meta = row[0], json.loads(row[1] or "{}")
            same = _normalize_memory_text(new_fact) == _normalize_memory_text(old_text)
            if same:
                return self._apply_correction_result(
                    new_fact, mid, old_text, old_meta, same_text=True,
                    candidates=candidates, status="unchanged",
                    resolution="semantic", layer="single",
                    audit={"source": "user", "method": "single_candidate"},
                )
            try:
                # 安全网比的是候选旧记忆和新说法的话题相似度，不是两个入参
                #（save_turn 传的是 correct_memory(fact, fact)，拿入参比
                # 永远是 1.0）。优先用库里存着的向量，没有再现算。
                old_vec = _blob_to_vector(row[2])
                if old_vec is None:
                    old_vec = self._embed_text(old_text)
                sim = _cosine(old_vec, self._embed_text(new_fact))
            except Exception:
                sim = None
            if sim is not None and sim < _CORRECTION_TOPIC_MIN_SIMILARITY:
                return {
                    "status": "ambiguous", "memory_id": None,
                    "old_text": old_text, "new_text": new_fact,
                    "candidates": self._candidates_with_text(candidates),
                    "reason": "与候选记忆相似度过低（%.2f），不敢改" % sim,
                    "layer": "none", "resolution": "ambiguous",
                    "audit": {"source": "user", "method": "safety_net",
                              "similarity": sim},
                }
            return self._apply_correction_result(
                new_fact, mid, old_text, old_meta, same_text=False,
                candidates=candidates, status="updated",
                resolution="semantic", layer="single",
                audit={"source": "user", "method": "single_candidate",
                       "similarity": sim},
            )

    def _void_last_stored(self):
        """作废上一轮刚存的记忆：账本记 delete，SQLite 标 voided。

        若记录了上一轮的 turn_id（_last_stored_turn_id），则把该轮存下的
        所有条目整轮作废（同一轮多句话拆条写入时不会只作废一半）；
        否则退化为作废最新写入的那 1 条（按 write_seq DESC, rowid DESC）。
        """
        with self._lock:
            candidates = self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
                " ORDER BY write_seq DESC, rowid DESC"
            ).fetchall()
            if not candidates:
                return 0
            rows = []
            if self._last_stored_turn_id:
                rows = [
                    r for r in candidates
                    if json.loads(r[2] or "{}").get("turn_id")
                    == self._last_stored_turn_id
                ]
            if not rows:
                rows = [candidates[0]]
            voided_count = 0
            for row in rows:
                meta = json.loads(row[2] or "{}")
                try:
                    _ledger.append_event(
                        self.ledger_path,
                        {"type": "delete", "memory_id": row[0],
                         "key": row[0], "text": row[1], "meta": meta},
                        lock=self._ledger_lock,
                    )
                except Exception:
                    logger.warning("撤回记账失败。", exc_info=True)
                    continue
                self._conn.execute(
                    "UPDATE memories SET voided=1 WHERE id=?", (row[0],)
                )
                voided_count += 1
            if voided_count:
                self._conn.commit()
                self._last_stored_turn_id = None
            return voided_count

    def flush_pending(self):
        """同步存的路没有待定区：什么都不用等，直接返回 {}（接口兼容）。"""
        return {}

    def ensure_outbox_started(self):
        """同步存的路没有 outbox：什么都不用启动（接口兼容）。"""
        return None

    # ---------- 读 ----------

    def _all_live_memories(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, metadata_json, created_at, write_seq, slot,"
                " source, user_confirmed, decay_kind, importance, event_date,"
                " embedding FROM memories WHERE voided=0 AND deleted=0 AND superseded=0"
                " ORDER BY created_at DESC"
            ).fetchall()
            out = []
            for r in rows:
                meta = _normalize_metadata(json.loads(r[2] or "{}"))
                out.append({
                    "id": r[0], "text": r[1], "metadata": meta,
                    "created_at": r[3], "write_seq": r[4], "slot": r[5],
                    "source": r[6], "user_confirmed": bool(r[7]),
                    "decay_kind": r[8], "importance": r[9],
                    "event_date": r[10], "embedding": r[11],
                })
            return out

    def get_all(self, include_sensitive=False):
        items = [
            {
                "id": m["id"], "memory": m["text"], "text": m["text"],
                "metadata": m["metadata"], "created_at": m["created_at"],
            }
            for m in self._all_live_memories()
        ]
        if include_sensitive:
            return items
        # #1 默认不返回敏感内容：页面列表/前台展示都不该看到密码/卡号。
        # include_sensitive=True 只给内部用（纠错归属校验、迁移/删除校验）。
        return [
            it for it in items
            if not _contains_sensitive(it["text"] or "")
        ]

    def search(self, query, top_k=5, reinforce=False, include_history=False,
               turn_id=None):
        """搜记忆：宽召回（语义 top_k*4 + BM25 全库 top_k*4）→ RRF k=60
        融合 → 原样匹配在前 → 截断 top_k，不够再账本兜底（见 _search_inner，
        旧 mem0 search 的排名逻辑）。

        返回和 Mem0Store.search 一样的形状：
        {"id","text","score","source","metadata","confirmed",
         "decay_weight","is_history"}。阈值 0.4，和原来一样。
        """
        query = (query or "").strip()
        if not query:
            return []
        try:
            return self._search_inner(query, top_k=top_k, include_history=include_history)
        except Exception:
            logger.warning("SQLite 检索失败。", exc_info=True)
            return []

    def _search_inner(self, query, top_k=5, include_history=False):
        """宽召回 → RRF 融合 → 先筛后截（旧 mem0 search 的排名逻辑，
        commit 7a69699“宽召回 + RRF 融合”，mem0_store.py search）。

        三路排名：语义（原始余弦分 ≥ 0.4，取 top_k*4）、BM25（全库打分，
        取 top_k*4）、图谱（还没建，空路）。标准 RRF k=60 融合，再乘
        衰减权重 × 类型权重（稳定偏好 1.25，见 _type_weight，2026-11-21
        Q3 修法）；原样匹配（基金代码/数字/整句，见
        memory/bm25.extract_exact_matches）的排在前面。
        和旧逻辑的对应关系：
          - 语义路按原始余弦分排位（旧 mem0 的语义路按向量原始分排，
            不是 0.6*cos+0.4*kw 的混合分——混合分里的关键词成分现在
            由 BM25 那路单独出，再混进语义路等于关键词算两遍；实测
            这么排会把万家那行从语义第 4 挤到第 5、融合后挤出前 5）。
            资格仍看原始余弦 ≥ 0.4；score 字段仍记混合分（第一部分
            的返回形状不动）。
          - 原样整句命中（query 在文本里或文本在 query 里）仍记 1.0 分、
            source="verbatim"，排在语义路最前面（第一部分的规矩，旧逻辑
            里对应 exact_match 置顶）。
          - BM25 命中不是语义分：score 记关键词重合度（旧关键词兜底
            同款 0–1 分），只做展示、不参与排位；source="keyword"。
          - 先筛后截：SQLite 索引里只有当前态（_all_live_memories 只取
            voided=0 AND deleted=0，对账也只重建当前态），投影过滤和
            档案闸在这里是空操作——被取代/删了的行根本不在候选池里，
            也没有沉底态（decay_state）可拦；衰减接进来时闸门加在这里、
            截断之前。精选先查（旧逻辑第 0 步）也没接：库里 curated 行
            是 0 条，等精选清单那步再接。
          - 图谱路是空列表：融合时等于没这一路（旧逻辑图坏了只丢图这路
            的同款处理）；图上才找得到的记忆（source="graph"）现在没有。
        """
        with self._lock:
            import numpy as np

            try:
                from memory import bm25 as _bm25
            except Exception:
                logger.warning("BM25 模块读不到：本次只用语义单路检索。", exc_info=True)
                _bm25 = None

            # #1 检索是拼进提示词的路：含敏感信息的条目不进候选池——已入库的
            # 旧敏感记忆（比如迁移带进来的）也不注入。
            live = [
                m for m in self._all_live_memories()
                if not _contains_sensitive(m["text"] or "")
            ]
            if not live:
                return []
            # 同一文本多行时（比如“包子”既有 BOOS 确认版、又有推断版），
            # 只留一行进检索：确认版优先、再按最新；未确认版不再占名额
            # （2026-11-21 网页测第二轮 #2：确认过的记忆不能输给推断版）。
            # live 已按 created_at 倒序：同文本先出现的是最新的。
            _by_text = {}
            for m in live:
                key = m["text"] or ""
                prev = _by_text.get(key)
                if prev is None:
                    _by_text[key] = m
                elif m["user_confirmed"] and not prev["user_confirmed"]:
                    _by_text[key] = m
            live = list(_by_text.values())
            wide_k = max(top_k * 4, top_k)
            qvec = self._embed_text(query)
            candidates = {}
            semantic_ranked = []
            bm25_ranked = []
            seen_texts = set()
            by_id = {m["id"]: m for m in live}

            def _result_item(m, score, source):
                meta = dict(m["metadata"] or {})
                return {
                    "id": m["id"], "text": m["text"], "score": round(score, 4),
                    "source": source, "metadata": meta,
                    "confirmed": bool(m["user_confirmed"]),
                    "decay_weight": _decay_weight(meta), "is_history": False,
                    # 类型权重只进排序键，不进返回字段（返回形状还是 8 个键，
                    # 见 test_search_result_shape_unchanged）；出结果前 pop 掉。
                    "_type_weight": _type_weight(m["text"]),
                }

            # 语义路：原始余弦分排位，取前 wide_k；原样命中的先放最前面。
            sem_scored = []
            verbatim_first = []
            for m in live:
                text = m["text"] or ""
                verbatim = query in text or text in query
                cos = None
                if qvec is not None and m["embedding"] is not None:
                    vec = _blob_to_vector(m["embedding"])
                    if vec is not None and vec.shape == qvec.shape:
                        cos = float(np.dot(qvec, vec))
                if verbatim:
                    verbatim_first.append(m)
                    continue
                if cos is None or cos < SEARCH_THRESHOLD:
                    continue
                kw = _keyword_overlap(query, text)
                sem_scored.append((cos, 0.6 * cos + 0.4 * kw, m))
            # 同分时确认优先（#2）：余弦相等时确认行在语义路里先占位。
            sem_scored.sort(key=lambda item: (-item[0], 0 if item[2]["user_confirmed"] else 1))
            for m in verbatim_first:
                if m["text"] in seen_texts:
                    continue
                seen_texts.add(m["text"])
                candidates[m["id"]] = _result_item(m, 1.0, "verbatim")
                semantic_ranked.append(m["id"])
            for _cos, score, m in sem_scored[:wide_k]:
                if m["text"] in seen_texts:
                    continue
                seen_texts.add(m["text"])
                candidates[m["id"]] = _result_item(m, score, "semantic")
                semantic_ranked.append(m["id"])

            # BM25 路：全库打分取前 wide_k，和语义平等进 RRF。
            exact_ids = set()
            if _bm25 is not None:
                try:
                    docs = [{"id": m["id"], "text": m["text"] or ""} for m in live]
                    exact_ids = set(_bm25.extract_exact_matches(query, docs))
                    # 同分时确认优先（#2）：BM25 分完全相等时确认行在路里先占位，
                    # 否则 RRF 会把后加的推断行排到确认行前面（live 是倒序）。
                    _bm25_hits = sorted(
                        _bm25.bm25_search(query, docs, top_k=wide_k),
                        key=lambda h: (-h["score"], 0 if (by_id.get(h["id"]) or {}).get("user_confirmed") else 1),
                    )
                    for hit in _bm25_hits:
                        doc_id = hit["id"]
                        m = by_id.get(doc_id)
                        if m is None:
                            continue
                        if doc_id in candidates:
                            candidates[doc_id]["exact_match"] = doc_id in exact_ids
                            bm25_ranked.append(doc_id)
                            continue
                        if m["text"] in seen_texts:
                            continue
                        seen_texts.add(m["text"])
                        candidates[doc_id] = _result_item(
                            m, _keyword_overlap(query, m["text"] or ""), "keyword"
                        )
                        bm25_ranked.append(doc_id)
                except Exception:
                    logger.warning("BM25 检索失败：本次只用语义路结果。", exc_info=True)
            for cid, item in candidates.items():
                item.setdefault("exact_match", cid in exact_ids)

            # 图谱路：还没建（已知缺口），空路进融合等于没这一路。
            graph_ranked = []
            if _bm25 is not None:
                fused = _bm25.reciprocal_rank_fusion(
                    [semantic_ranked, bm25_ranked, graph_ranked], k=60
                )
            else:
                fused = [(doc_id, 0.0) for doc_id in semantic_ranked]
            fused_items = []
            for doc_id, rrf in fused:
                item = candidates.get(doc_id)
                if item is None:
                    continue
                item["_rrf"] = rrf
                fused_items.append(item)
            fused_items.sort(
                key=lambda item: (
                    1 if item.get("exact_match") else 0,
                    (item.get("_rrf") or 0) * (item.get("decay_weight") or 0)
                    * (item.get("_type_weight") or 1.0),
                    # 确认优先只当平局裁决（#2）：同分时确认过的排前面。
                    1 if item.get("confirmed") else 0,
                ),
                reverse=True,
            )
            results = []
            for item in fused_items:
                item.pop("_rrf", None)
                item.pop("_type_weight", None)
                item.pop("exact_match", None)
                results.append(item)
            results = results[:top_k]
            seen = {r["text"] for r in results}
            # 账本兜底：不够 top_k 时翻账本原话按关键词找。
            # 只找当前态的：被取代/删了的不补进来。
            if len(results) < top_k:
                try:
                    state = self.ledger_state()
                    key_states = (state or {}).get("key_states") or {}
                    events = []
                    for rec in (state or {}).get("adds") or []:
                        if key_states.get(rec.get("key")) in (
                            _ledger.LEDGER_STATE_SUPERSEDED,
                            _ledger.LEDGER_STATE_DELETED,
                        ):
                            continue
                        events.append({
                            "type": _ledger.LEDGER_EVENT_ADD,
                            "key": rec.get("key"), "text": rec.get("text") or "",
                            "meta": rec.get("meta") or {},
                        })
                    for hit in _ledger.find_in_ledger(
                        events, query, top_k=top_k - len(results),
                        exclude_keys={r["id"] for r in results},
                    ):
                        text = hit.get("text") or ""
                        if not text or text in seen:
                            continue
                        if _contains_sensitive(text):
                            continue
                        meta = _normalize_metadata(hit.get("meta") or {})
                        results.append({
                            "id": None, "text": text,
                            "score": hit.get("score"), "source": "ledger",
                            "metadata": meta,
                            "confirmed": bool(meta.get("user_confirmed")),
                            "decay_weight": 1.0, "is_history": True,
                            "ledger_key": hit.get("key"),
                        })
                        seen.add(text)
                except Exception:
                    logger.warning("账本兜底检索失败。", exc_info=True)
            # 历史查询（2026-11-22，Gemini #5）：include_history=True 时，
            # 在活记忆后面补账本里被取代/删了的旧版本（翻找 supersede/delete
            # 事件），标 is_history=True、source="history"。活记忆不动、
            # 顺序不动，历史只追加；默认（include_history=False）不补，
            # 免得旧版本混进正常回答。
            if include_history:
                try:
                    state = self.ledger_state()
                    key_states = (state or {}).get("key_states") or {}
                    history_events = []
                    for rec in (state or {}).get("adds") or []:
                        if key_states.get(rec.get("key")) in (
                            _ledger.LEDGER_STATE_SUPERSEDED,
                            _ledger.LEDGER_STATE_DELETED,
                        ):
                            history_events.append({
                                "type": _ledger.LEDGER_EVENT_ADD,
                                "key": rec.get("key"),
                                "text": rec.get("text") or "",
                                "meta": rec.get("meta") or {},
                            })
                    for hit in _ledger.find_in_ledger(
                        history_events, query, top_k=top_k,
                        exclude_keys={r["id"] for r in results},
                    ):
                        text = hit.get("text") or ""
                        if not text or text in seen:
                            continue
                        if _contains_sensitive(text):
                            continue
                        meta = _normalize_metadata(hit.get("meta") or {})
                        results.append({
                            "id": hit.get("key"), "text": text,
                            "score": hit.get("score"), "source": "history",
                            "metadata": meta,
                            "confirmed": bool(meta.get("user_confirmed")),
                            "decay_weight": 1.0, "is_history": True,
                            "ledger_key": hit.get("key"),
                        })
                        seen.add(text)
                except Exception:
                    logger.warning("历史检索失败。", exc_info=True)
            return results

    def search_history(self, query, top_k=5):
        return self.search(query, top_k=top_k, include_history=True)

    def note_context(self, memories):
        """记下本轮检索到的记忆 id，并写回 last_used_at（刚被用上）：
        沉底的（dormant）和待删的（decay_delete_due）都复活回 verbatim，
        账本记 REVIVE（2026-11-22 decay 接线）。
        """
        with self._lock:
            self._last_context_ids = [
                (m or {}).get("id") for m in (memories or []) if (m or {}).get("id")
            ]
            for mid in self._last_context_ids:
                try:
                    row = self._conn.execute(
                        "SELECT text, metadata_json FROM memories WHERE id=? AND voided=0"
                        " AND deleted=0 AND superseded=0",
                        (mid,),
                    ).fetchone()
                    if row is None:
                        continue
                    meta = json.loads(row[1] or "{}")
                    was_dormant = meta.get("decay_state") in (
                        _decay.DECAY_STATE_DORMANT, _decay.DECAY_STATE_DELETE_DUE,
                    )
                    meta["last_used_at"] = _now_iso()
                    if was_dormant:
                        meta["decay_state"] = _decay.DECAY_STATE_VERBATIM
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), mid),
                    )
                    self._conn.commit()
                    if was_dormant:
                        try:
                            _ledger.append_event(
                                self.ledger_path,
                                {"type": _ledger.LEDGER_EVENT_REVIVE,
                                 "memory_id": mid, "key": mid,
                                 "text": row[0] or ""},
                                lock=self._ledger_lock,
                            )
                        except Exception:
                            logger.warning("复活记账失败。", exc_info=True)
                except Exception:
                    logger.warning("note_context 写回失败。", exc_info=True)

    def sweep_decay(self, now=None):
        """衰减扫一遍（2026-11-22 decay 接线）：按 decay.py 的规矩，
        闲置够久、权重够低的标沉底（dormant），到期的记 delete_due。
        沉底只是状态（decay_state=dormant + 账本 DORMANT 事件），
        记忆还在、可搜；delete_due 只是到期标记，不删——真删永远只
        走 BOOS 点头的那条路（delete_memory / 忘记这个）。
        不自动跑：由调用方（定时任务/自检）定期调一次。
        返回 {"dormant": n, "delete_due": n}。
        """
        now = now or _now_epoch()
        dormant = 0
        delete_due = 0
        with self._lock:
            for row in self._conn.execute(
                "SELECT id, text, metadata_json FROM memories"
                " WHERE voided=0 AND deleted=0 AND superseded=0"
            ).fetchall():
                meta = json.loads(row[2] or "{}")
                state = _decay.decay_state(
                    meta.get("decay_kind") or _decay.DECAY_KIND_EPISODIC,
                    meta.get("last_used_at") or meta.get("created_at"),
                    now=now,
                    is_core=bool(_provenance._is_core_memory(meta)),
                    is_curated=bool(meta.get("curated")),
                )
                if state == _decay.DECAY_STATE_DELETE_DUE and meta.get(
                    "decay_state"
                ) != _decay.DECAY_STATE_DELETE_DUE:
                    meta["decay_state"] = state
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), row[0]),
                    )
                    delete_due += 1
                elif state == _decay.DECAY_STATE_DORMANT and meta.get(
                    "decay_state"
                ) != _decay.DECAY_STATE_DORMANT:
                    meta["decay_state"] = state
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), row[0]),
                    )
                    dormant += 1
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {"type": _ledger.LEDGER_EVENT_DORMANT,
                             "memory_id": row[0], "key": row[0],
                             "text": row[1] or ""},
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning("沉底记账失败。", exc_info=True)
                elif state == _decay.DECAY_STATE_VERBATIM and meta.get(
                    "decay_state"
                ) in (
                    _decay.DECAY_STATE_DORMANT,
                    _decay.DECAY_STATE_DELETE_DUE,
                ):
                    # 复活过的记忆：算出来已是原话态，把库里还顶着的旧
                    # 牌子改回 verbatim，别让它一直挂着 dormant / delete_due。
                    meta["decay_state"] = state
                    self._conn.execute(
                        "UPDATE memories SET metadata_json=? WHERE id=?",
                        (json.dumps(meta, ensure_ascii=False), row[0]),
                    )
                    try:
                        _ledger.append_event(
                            self.ledger_path,
                            {"type": _ledger.LEDGER_EVENT_REVIVE,
                             "memory_id": row[0], "key": row[0],
                             "text": row[1] or ""},
                            lock=self._ledger_lock,
                        )
                    except Exception:
                        logger.warning("复活记账失败。", exc_info=True)
            self._conn.commit()
        return {"dormant": dormant, "delete_due": delete_due}

    # ---------- 改 / 删（今天只实现删除；纠错后面再做） ----------

    def delete_memory(self, memory_id, approve=False):
        """删一条记忆：要 BOOS 亲口确认（approve=True）才真删。

        账本记 delete，SQLite 标 deleted；原文留在账本里，可恢复。
        SQLite 里没有但账本里有该 key（账本兜底捞出的条目）时，同样
        记账删除，防止重启对账或兜底检索再次捞出。
        """
        if not approve:
            return {"deleted": False, "reason": "need_approve"}
        if not memory_id:
            return {"deleted": False, "reason": "not_found"}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, metadata_json, deleted FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                state = self.ledger_state()
                key_states = (state or {}).get("key_states") or {}
                rec = next(
                    (
                        r for r in (state or {}).get("adds") or []
                        if r.get("key") == memory_id
                    ),
                    None,
                )
                if rec is None or key_states.get(memory_id) == _ledger.LEDGER_STATE_DELETED:
                    return {"deleted": False, "reason": "not_found"}
                try:
                    _ledger.append_event(
                        self.ledger_path,
                        {
                            "type": "delete",
                            "memory_id": memory_id,
                            "key": memory_id,
                            "text": rec.get("text") or "",
                            "meta": rec.get("meta") or {},
                        },
                        lock=self._ledger_lock,
                    )
                except Exception:
                    logger.warning("删除记账失败。", exc_info=True)
                    return {"deleted": False, "reason": "ledger_failed"}
                return {"deleted": True, "memory_id": memory_id}
            if row[3]:
                return {"deleted": False, "reason": "not_found"}
            meta = json.loads(row[2] or "{}")
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {"type": "delete", "memory_id": row[0],
                     "key": row[0], "text": row[1], "meta": meta},
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("删除记账失败。", exc_info=True)
                return {"deleted": False, "reason": "ledger_failed"}
            self._conn.execute(
                "UPDATE memories SET deleted=1 WHERE id=?", (row[0],)
            )
            self._conn.commit()
            return {"deleted": True, "memory_id": row[0]}

    def confirm_memory(self, memory_id):
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata_json FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                return {"confirmed": False, "reason": "not_found"}
            meta = json.loads(row[0] or "{}")
            meta["user_confirmed"] = True
            self._conn.execute(
                "UPDATE memories SET user_confirmed=1, metadata_json=? WHERE id=?",
                (json.dumps(meta, ensure_ascii=False), memory_id),
            )
            self._conn.commit()
            return {"confirmed": True, "memory_id": memory_id}

    def void_joke_memory(self, memory_id):
        """作废一条玩笑记忆：先记账本 delete（标记 voided），再标 SQLite voided=1。

        账本是唯一真相源——不写账本的话，删库对账重建（reconcile_ledger_index）
        会把作废的玩笑记忆重新复活。返回值同时带 ok 与 voided，对齐 Mem0Store。
        """
        if not memory_id:
            return {"ok": False, "voided": False, "reason": "not_found"}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, metadata_json FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return {
                    "ok": False,
                    "voided": False,
                    "reason": "not_found",
                    "memory_id": memory_id,
                }
            meta = json.loads(row[2] or "{}")
            meta["voided"] = True
            try:
                _ledger.append_event(
                    self.ledger_path,
                    {
                        "type": "delete",
                        "memory_id": row[0],
                        "key": row[0],
                        "text": row[1],
                        "meta": meta,
                    },
                    lock=self._ledger_lock,
                )
            except Exception:
                logger.warning("作废玩笑记忆记账失败。", exc_info=True)
                return {
                    "ok": False,
                    "voided": False,
                    "reason": "ledger_failed",
                    "memory_id": row[0],
                }
            self._conn.execute(
                "UPDATE memories SET voided=1 WHERE id=?", (row[0],)
            )
            self._conn.commit()
            return {"ok": True, "voided": True, "memory_id": row[0]}

    def list_protection_events(self, limit=50):
        return []

    def deletion_scope(self, memory_id):
        """删除范围预览（2026-11-21 网页测第二轮 #6）：真查库，不是桩。

        和 mem0_store.deletion_scope 一样的形状：found、memory{id,text,metadata}、
        history_entries/history_count（该条的账本事件）、in_scope/out_of_scope。
        找不到时 found=False（调用方按 404 处理，见 web/server.py）。
        memory_id 既可以是 SQLite 的 id，也可以是账本 key（delete_memory
        同款两路查找）。
        """
        if not memory_id:
            return {"scope": "single", "memory_id": memory_id, "found": False}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, metadata_json, created_at FROM memories"
                " WHERE id=? AND voided=0 AND deleted=0",
                (memory_id,),
            ).fetchone()
            key = memory_id
            if row is None:
                state = self.ledger_state()
                rec = next(
                    (r for r in (state or {}).get("adds") or [] if r.get("key") == memory_id),
                    None,
                )
                key_states = (state or {}).get("key_states") or {}
                if rec is None or key_states.get(memory_id) == _ledger.LEDGER_STATE_DELETED:
                    return {"scope": "single", "memory_id": memory_id, "found": False}
                memory = {
                    "id": memory_id,
                    "text": rec.get("text") or "",
                    "metadata": rec.get("meta") or {},
                }
            else:
                memory = {
                    "id": row[0],
                    "text": row[1],
                    "metadata": json.loads(row[2] or "{}"),
                }
            history = []
            try:
                # read_events 返回 (events, info) 二元组，不是事件列表本身：
                # 直接 for 遍历会拿到 events 列表再拿 info 字典，dict
                # 的 .get 不会报错但事件列表被当成事件、.get 在 list 上
                # 抛 AttributeError，try 包着所以历史永远是空的
                # （2026-11-22 修：先解包再遍历事件）。
                events, _info = _ledger.read_events(self.ledger_path)
                for event in events:
                    if (event or {}).get("key") == key or (event or {}).get("memory_id") == key:
                        history.append(event)
            except Exception:
                logger.warning("删除范围读账本历史失败。", exc_info=True)
            return {
                "scope": "single",
                "memory_id": memory_id,
                "found": True,
                "memory": memory,
                "history_entries": history,
                "history_count": len(history),
                "in_scope": [memory],
                "out_of_scope": [],
            }

    def pending_notices(self):
        return []

    # ---------- 启动时要调的（接口兼容，SQLite 版里是空操作） ----------

    def ensure_identity_memories(self):
        """身份记忆已经在账本里了，SQLite 版启动时不用再种。"""
        return {}

    def backfill_timestamps(self):
        """补时间戳是 mem0 索引的事，SQLite 行里本来就有 created_at。"""
        return 0


def _default_memory_dir():
    return Path(__file__).resolve().parent


# _CORRECTION_TOPIC_MIN_SIMILARITY（纠错安全网：旧-新话题相似度下限，2026-11-22 接线）：
# 单候选时，旧说法和新说法的余弦相似度低于这个值就不换，改报 ambiguous
# （问 BOOS），免得改错不相干的记忆。0.70 是 mem0 那边 2026-11-12
# 实测校准出来的分界（同话题 0.757–0.907、跨话题 0.411–0.763，
# 干净空隙 0.656–0.757，取中点约 0.70；见 mem0_store.py 同名常量）。
# 向量算不出来（模型没起来）时跳过这道网，交由调用方兜底。
_CORRECTION_TOPIC_MIN_SIMILARITY = 0.70


def _cosine(a, b):
    """两条向量的余弦相似度；任一向量没有（None）或模为 0 时返回 None
    （调用方按“算不出来”处理，不当 0 分用）。"""
    if a is None or b is None:
        return None
    try:
        import numpy as np

        va = np.asarray(a, dtype="float64")
        vb = np.asarray(b, dtype="float64")
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return None
        return float(np.dot(va, vb) / (na * nb))
    except Exception:
        return None


def _decay_weight(metadata):
    """这条记忆现在的衰减权重（0.10–1.0，核心记忆恒为 1.0）。

    按 memory/decay.py 算（2026-11-22 接线）：指数衰减，
    weight = 0.5 ** (闲置天数 / 半衰期)，下限 0.10；稳定事实半衰期
    180 天、一次性经历 14 天；核心记忆恒为 1.0。
    输入从 metadata 来：decay_kind、last_used_at（没有时按 created_at
    算，再没有按 1.0）；is_core 按 provenance 的口径（已确认或有槽位）。
    权重只管排位、不管能不能搜到（资格仍只看原始余弦分 ≥ 0.4）。
    注意：这里不用 effective_weight 的未确认封顶（0.80）——封顶会改动
    已确认/未确认混排的 RRF 钉死顺序（Q4 等回归），等 BOOS 定了再接。
    """
    meta = metadata or {}
    kind = meta.get("decay_kind") or _decay.DECAY_KIND_EPISODIC
    last_used = meta.get("last_used_at") or meta.get("created_at")
    try:
        is_core = bool(_provenance._is_core_memory(meta))
    except Exception:
        is_core = bool(meta.get("is_rule") or meta.get("curated"))
    return _decay.decay_weight(kind, last_used, now=None, is_core=is_core)


def _type_weight(text):
    """这条记忆的类型权重：稳定偏好 1.25，一次性事件 / 其他 1.0。

    修 Q3 退化（2026-11-21，双写比对 18 问里唯一退化）：
    问“我爱吃什么”时，纠正行“不太喜欢鸡杂面”被新存的饺子/米线
    两行按余弦挤出前 5（RRF 0.0152 vs 米线 0.0154，一位之差）。
    根因不是排序逻辑错——同批旧行的余弦相对顺序新旧一致，是新增的
    真实行按余弦挤出一位。修法是类型权重（稳定偏好 > 一次性事件），
    不是改余弦、不是改阈值。

    为什么按正文认，不按 decay_kind：decay.py 的规矩是“BOOS 亲口说
    的偏好记 stable”，但迁移进来的偏好句在 mem0 里都是 source=inferred，
    迁移时 decay_kind 一律写成了 episodic（火锅/冒菜是 user 亲口说的，
    所以是 stable；牛肉面喜、纠正行、鸡杂面喜都是 inferred，全是
    episodic）。档位没标上，检索时按正文再认一次：说喜欢/不喜欢/
    爱吃/爱喝的句子就是稳定偏好，不管它当初怎么标的。
    权重只管排位：资格仍只看原始余弦分 ≥ 0.4（decay.py 坑 1），
    再低的权重也不会让记忆搜不到，再高的也不会把不够 0.4 的行拉进来。
    """
    if any(word in (text or "") for word in _PREFERENCE_WORDS):
        return _TYPE_WEIGHT_STABLE_PREFERENCE
    return 1.0


def _keyword_overlap(query, text):
    """关键词重合度（0–1）：按字算，简单够用。"""
    q = set(query or "")
    t = set(text or "")
    q = {c for c in q if c.strip() and c not in "，。！？、；：\"'()（）《》"}
    t = {c for c in t if c.strip() and c not in "，。！？、；：\"'()（）《》"}
    if not q:
        return 0.0
    return len(q & t) / len(q)


def _blob_to_vector(blob):
    if blob is None:
        return None
    try:
        import numpy as np

        return np.frombuffer(blob, dtype="float32")
    except Exception:
        return None


def create_sqlite_store(budget_tracker=None, **kwargs):
    """建一个 SQLiteStore；建不好时返回 None（降级成“没有记忆也能聊”）。

    建好后顺手对一次账（reconcile_ledger_index）：账本里有、索引里
    没有的，按账本重建索引——写索引那步崩了、或索引坏过，重启后
    靠这一笔把记忆捞回来。对账本身不抛异常、只记日志。
    """
    try:
        store = SQLiteStore(budget_tracker=budget_tracker, **kwargs)
    except Exception as exc:
        logger.warning("SQLite 记忆未启用：%s，本次不带记忆运行。", exc)
        return None
    try:
        store.reconcile_ledger_index()
    except Exception:
        logger.warning("启动对账失败：本次不对索引补缺。", exc_info=True)
    return store
