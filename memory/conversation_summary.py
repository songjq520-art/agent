# memory/conversation_summary.py（对话摘要 + 最近对话列表：GPT 式上下文压缩）
# 作用：让十二记住“上一段对话聊了什么”，并在网页左边列出最近的对话。
# 背景（2026-11-10，BOOS 本地测试后定的）：
#   长期记忆（mem0）只记提炼出来的事，不记原话逐句；新对话问“上一段对话
#   聊了什么”，十二老实说不知道——这是设计不是 bug。BOOS 要它能记住上一段
#   对话，定了用 GPT 那种做法：对话历史到上限就总结成摘要、开新对话，
#   摘要代替旧历史。后来又加了：网页左侧像 ChatGPT 那样列“最近”的对话，
#   点一条回到那段对话、接着聊。
# 做法（BOOS 定的，照做）：
#   - 上限 32,000 token（约 100 多轮短对话），见 CONVERSATION_SUMMARY_TOKEN_LIMIT；
#     总结时最近 5 轮原样留着，只把更早的压成摘要。
#   - 摘要存在硬盘上（本地 JSON 文件，memory/conversation_summary.json，
#     .gitignore 的 memory/*.json* 已覆盖，不上传）；重启后读回来，
#     新对话开始时放进提示里当上下文。
#   - 对话正常结束时也要总结一次（CLI 退出、网页开新对话、服务关闭时，
#     和 Plan C 的 flush_pending 同样的落点），不只到上限才总结——
#     不然 BOOS 的短测试对话永远触发不到，功能等于没有。
#   - 摘要用 DeepSeek flash 走现有 provider 接口生成，简短、省钱；
#     摘要本身是对话记录、单独存放，绝不喂给 mem0 当事实提取
#     （摘要只拼进发给模型的问题里，不进 add_turn 的消息）。
#   - 摘要发给模型之前先脱敏：对话记录和旧摘要都过
#     memory.mem0_store._redact_sensitive（和 mem0 提取同一套，
#     密码/密钥/令牌换成占位符），秘密不发到外部 API；本地存的
#     最近几轮和标题是本地存储，不上传，不在这次脱敏范围里。
#   - 摘要失败（没 Key、网络断、模型返回空）时不丢历史：不裁剪、不覆盖，
#     下一轮再试；聊天本身不受影响。
#   - 一个对话一条记录：标题（第一句用户话，没有时取摘要头一句）、
#     开始时间、摘要、最近几轮（原样留着，供点开后接着聊）。
#     列表最多留最近 20 条，多的从旧的开始删。
#   - 点开旧对话时恢复它的摘要 + 最近几轮，接着聊；之后的新轮次
#     记回同一条记录，更新时间后它回到列表顶部。
# 关系：main.py 的 run_question() 提问前把摘要块拼进问题（和 _memory_block
#       一样的位置）；对话历史的拥有者（chat_loop、web/server.py 的 ChatState）
#       在每轮后调 maybe_compact() + sync()、结束时调 flush()。
# Token 估算：用本文件的 estimate_tokens()，和 main._estimate_tokens()
#       同一套启发式（中文按字算、英文按词算，宁可高估）；注释里写明是
#       估算值，上限本身就是约数，不是模型返回的真实 usage。

import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from memory.mem0_store import _redact_sensitive

logger = logging.getLogger("agent")

# CONVERSATION_SUMMARY_TOKEN_LIMIT（对话历史的 Token 上限）：
# 历史估算超过这个数就触发一次总结。BOOS 2026-11-10 定：32,000，
# 约 100 多轮短对话；8000 太少、总结来得太勤，32000 记得更细、
# 费用基本一样（总结一次贵一点、但来得少，摊下来差不多）。
CONVERSATION_SUMMARY_TOKEN_LIMIT = 32000

# CONVERSATION_SUMMARY_KEEP_LAST_ROUNDS（总结时原样保留的最近轮数）：
# 一轮 = 一问一答（2 条消息）。总结时最近 5 轮留在历史里，只压更早的，
# 聊起来不会断；最近列表里恢复的也是这 5 轮。
CONVERSATION_SUMMARY_KEEP_LAST_ROUNDS = 5

# CONVERSATION_SUMMARY_MAX_TOKENS（存下来的摘要本身的 Token 上限，估算值）：
# 摘要是给以后翻看用的梗概，不是逐字记录；合并后超了就只留后半段
# （最近的内容），防止摘要本身越长越长、每次提问都带一大段。
CONVERSATION_SUMMARY_MAX_TOKENS = 2000

# MAX_CONVERSATIONS（最近列表最多留几条对话）：BOOS 定 10–20 条，取 20；
# 多的从最旧的开始删（只删列表记录，不动 mem0 里的长期记忆）。
MAX_CONVERSATIONS = 20

# TITLE_MAX_CHARS（列表标题最多几个字）：取对话第一句用户话，超了截掉加…。
TITLE_MAX_CHARS = 24

# SUMMARY_FILE_VERSION（摘要文件的格式版本）：以后改格式时加一，
# 旧文件读到不认识的版本时按空处理（不 crash、不猜）。
# v1 是单条摘要的老格式（没发出去过，只在这里兼容一下）；
# v2 是一个对话一条记录（现在用的）。
SUMMARY_FILE_VERSION = 2


def _is_cjk_char(ch):
    """判断一个字符是不是中文字符（CJK 统一表意文字区）。"""
    return "\u4e00" <= ch <= "\u9fff"


def _now_iso():
    """现在时间的 UTC ISO 串（带 Z），存进文件、前端按 Asia/Shanghai 显示。"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def estimate_tokens(text):
    """估算一段文本的 Token 数（启发式，和 main._estimate_tokens() 同一套）。

    中文字符按 2 Token/字算，英文按单词算（1 Token/词），其他字符忽略。
    这是粗略估计，宁可高估、不低估；上限本身就是约数（32,000），
    差一点不影响功能。真实用量以模型返回的 usage 为准。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk_char(ch))
    # 非中文部分按空白分词算单词数。
    non_cjk = "".join(" " if _is_cjk_char(ch) else ch for ch in text)
    words = len(non_cjk.split())
    return cjk * 2 + words


def estimate_messages_tokens(messages):
    """估算一段对话历史的 Token 数（只算消息内容，不算系统提示和记忆块）。"""
    return sum(estimate_tokens((msg or {}).get("content") or "") for msg in (messages or []))


def format_turns(turns):
    """把一轮轮消息排成给模型看的对话记录（BOOS 说的话标 BOOS，回答标十二）。"""
    lines = []
    for msg in turns or []:
        role = (msg or {}).get("role")
        content = (msg or {}).get("content") or ""
        if role == "user":
            lines.append(f"BOOS：{content}")
        elif role == "assistant":
            lines.append(f"十二：{content}")
        else:
            lines.append(content)
    return "\n".join(lines)


def _redacted_turns(turns):
    """发给模型前脱敏：每轮消息内容过一遍 _redact_sensitive（和 mem0
    提取同一套，密码/密钥/令牌换成占位符），秘密不出本机。
    只改要发出去的副本，原消息不动；本地存的最近几轮和标题不受影响。"""
    return [
        {
            "role": (msg or {}).get("role"),
            "content": _redact_sensitive((msg or {}).get("content") or ""),
        }
        for msg in (turns or [])
    ]


def build_summary_prompt(turns, previous_summary=None):
    """给 DeepSeek 的总结提示：把一段对话压成简短摘要。

    要求模型只写对话里有的内容、不编造；留关键信息（话题、重要的事、
    决定、人名/地名/数字/待办）。有旧摘要时让模型合并进去，仍保持简短。
    发给模型之前先脱敏：对话记录和旧摘要都过 _redact_sensitive，
    密码/密钥等秘密只留占位符，不发到外部 API（mem0 提取也是同一套，
    两边一致；修复前生成的旧摘要里可能已经带了秘密，合并时也要脱敏）。
    """
    parts = [
        "你是十二的对话记录员。把下面这段对话压缩成一段简短摘要。",
        "要求：",
        "- 用中文写，300 字以内；",
        "- 保留关键信息：聊了什么话题、BOOS 说了什么重要的事、做了什么决定、"
        "提到的人名/地名/数字/待办；",
        "- 只写对话里有的内容，不要编造，不要加自己的推测；",
        "- 摘要是给以后翻看用的，写清楚、别写空话。",
    ]
    if previous_summary:
        parts.append(
            "下面是这段对话之前的摘要，请把新内容合并进去，仍保持在 500 字以内：\n"
            f"{_redact_sensitive(previous_summary)}"
        )
    parts.append("对话记录：\n" + format_turns(_redacted_turns(turns)))
    return "\n".join(parts)


def summary_block(summary_text):
    """把摘要拼成一段提示，塞进发给模型的问题里（和 _memory_block 一个位置）。

    没有摘要时返回空字符串。摘要标明身份——它是过去对话的梗概，
    不是逐字记录、不是已确认的事实；没有提到就说不知道，不要编。
    """
    if not (summary_text or "").strip():
        return ""
    return "\n".join(
        [
            "以下是之前对话的摘要（上一次对话的梗概，不是逐字记录）：",
            summary_text.strip(),
            "注意：摘要只是梗概，可能有遗漏；如果摘要里没有提到，"
            "就说不知道，不要编。",
        ]
    )


def make_deepseek_summarizer(api_key=None, budget_tracker=None):
    """造一个用 DeepSeek flash 生成摘要的函数（生产环境用）。

    返回的函数签名是 summarizer(turns, previous_summary) -> str | None：
    turns 是要总结的消息列表，previous_summary 是已有的旧摘要（没有时传 None）。
    生成失败（没 Key、网络断、模型返回空）时返回 None，调用方按“跳过”
    处理，不丢历史、不覆盖旧摘要。
    Key 在每次调用时才从环境变量读（不写死、不提前缓存），
    预算走传进来的 BudgetTracker（和正常问答同一套记账）。
    """
    from config import get_api_key

    def summarizer(turns, previous_summary=None):
        key = api_key or get_api_key("deepseek")
        if not key:
            logger.warning("对话摘要未生成：未配置 DEEPSEEK_API_KEY。")
            return None
        try:
            # 延迟 import：平时只用摘要存取的测试不用拉起 requests。
            from providers.deepseek import DeepSeekProvider

            provider = DeepSeekProvider(api_key=key, budget_tracker=budget_tracker)
            result = provider.chat_messages(
                [{"role": "user", "content": build_summary_prompt(turns, previous_summary)}]
            )
        except Exception:
            logger.warning("对话摘要生成失败：本次跳过，历史不动。", exc_info=True)
            return None
        text = (result or {}).get("text") or ""
        if not text.strip():
            logger.warning("对话摘要生成失败：模型返回了空内容，本次跳过。")
            return None
        return text.strip()

    return summarizer


def _truncate_summary(text, token_max=CONVERSATION_SUMMARY_MAX_TOKENS):
    """摘要太长时只留后半段（最近的内容），从前往后截。

    按字符从后往前累计估算 Token，到上限就停；摘要本来就短，
    正常走不到这里，只是防止它越长越长。
    """
    if estimate_tokens(text) <= token_max:
        return text
    total = 0
    cut = len(text)
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        total += 2 if _is_cjk_char(ch) else (0 if ch.isspace() else 1)
        if total > token_max:
            cut = i + 1
            break
    return text[cut:].lstrip()


def _first_sentence(text):
    """取一段文字的第一句：先按行、再按最早出现的句号/问号/感叹号切。"""
    if not (text or "").strip():
        return ""
    first_line = (text or "").strip().split("\n", 1)[0].strip()
    cut = len(first_line)
    for sep in ("。", "？", "！", "?", "!"):
        idx = first_line.find(sep)
        if idx != -1:
            cut = min(cut, idx)
    return first_line[:cut].strip()


def _make_title(messages, summary=None):
    """给一条对话起标题：对话第一句用户话；没有时取摘要头一句；再没有叫“新对话”。

    标题只在对话第一次记下来时定一次，之后不改（第一句就是这段对话的题）。
    """
    for msg in messages or []:
        if (msg or {}).get("role") == "user":
            title = _first_sentence((msg or {}).get("content") or "")
            if title:
                break
    else:
        title = _first_sentence(summary or "")
    if not title:
        return "新对话"
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS] + "…"
    return title


def _copy_messages(messages):
    """复制一段消息列表（只留 role/content，调用方改了原列表也不影响存档）。"""
    return [
        {"role": (msg or {}).get("role"), "content": (msg or {}).get("content") or ""}
        for msg in (messages or [])
        if (msg or {}).get("role") in ("user", "assistant")
    ]


class ConversationSummaryStore:
    """对话摘要 + 最近对话列表的存取（存在硬盘上，重启还在）。

    文件默认在项目 memory/ 目录下（conversation_summary.json），
    和 mem0 的本地数据放一起；.gitignore 的 memory/*.json* 已覆盖，
    不会上传到 GitHub。文件格式 {"version": 2, "conversations": [...]}，
    一条记录 {"id", "title", "started_at", "updated_at", "summary",
    "recent", "turns_summarized"}。
    损坏、缺字段、版本不认识时按空处理，不 crash、不猜。
    保存用临时文件 + os.replace（原子写），写坏了旧文件还在。
    current_id 只放内存里，不存文件——服务重启后没有“正在聊”的对话，
    新对话从最新的那条记录里取摘要当上下文（见 current_summary）。
    线程：本类自己不加锁——调用方（ChatState 的锁、单线程的 chat_loop）
    保证一次只有一个地方在调；不要两个线程同时调同一个 store。
    """

    def __init__(
        self,
        file_path=None,
        summarizer=None,
        token_limit=CONVERSATION_SUMMARY_TOKEN_LIMIT,
        keep_last_rounds=CONVERSATION_SUMMARY_KEEP_LAST_ROUNDS,
        max_conversations=MAX_CONVERSATIONS,
    ):
        # 文件位置：显式传了 file_path 就用它；没传时看环境变量
        # AGENT_SUMMARY_FILE（测试指到临时目录用），再没有就用默认的
        # memory/conversation_summary.json。
        if file_path is None:
            file_path = os.environ.get("AGENT_SUMMARY_FILE")
        self.file_path = (
            Path(file_path)
            if file_path
            else Path(__file__).resolve().parent / "conversation_summary.json"
        )
        # summarizer：summarizer(turns, previous_summary) -> str | None。
        # 不传时不总结（maybe_compact / flush 直接跳过），只做存取。
        self.summarizer = summarizer
        self.token_limit = token_limit
        self.keep_last_rounds = keep_last_rounds
        self.max_conversations = max_conversations
        # conversations：对话记录列表，新的在前（按 updated_at 从新到旧）。
        self.conversations = []
        # current_id：正在聊的对话是哪条（内存里才有，重启后是 None）。
        # _current_record：正在聊的那条记录本身（和 current_id 一起换）。
        self.current_id = None
        self._current_record = None
        # _last_flush_fingerprint：上一次 flush 总结过的消息指纹；
        # 同一份消息重复 flush 时跳过，不把同一段对话总结两遍。
        self._last_flush_fingerprint = None
        self._load()

    # ---------- 存取 ----------

    def _load(self):
        """启动时从硬盘读回对话记录；没有文件或文件坏了时按空处理。"""
        try:
            raw = self.file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("对话摘要文件读失败（%s），按空处理。", self.file_path)
            return
        if not isinstance(data, dict):
            return
        version = data.get("version")
        if version == SUMMARY_FILE_VERSION:
            records = data.get("conversations") or []
        elif version == 1:
            # v1 是单条摘要的老格式：转成一条记录（标题叫“之前的对话”）。
            summary = data.get("summary") or ""
            records = (
                [
                    {
                        "id": "legacy",
                        "title": "之前的对话",
                        "started_at": data.get("updated_at"),
                        "updated_at": data.get("updated_at"),
                        "summary": summary,
                        "recent": [],
                        "turns_summarized": data.get("turns_summarized") or 0,
                    }
                ]
                if isinstance(summary, str) and summary.strip()
                else []
            )
        else:
            logger.warning("对话摘要文件版本不认识，按空处理。")
            return
        cleaned = []
        for rec in records:
            if not isinstance(rec, dict) or not rec.get("id"):
                continue
            cleaned.append(
                {
                    "id": str(rec["id"]),
                    "title": str(rec.get("title") or "新对话"),
                    "started_at": rec.get("started_at"),
                    "updated_at": rec.get("updated_at"),
                    "summary": rec.get("summary") if isinstance(rec.get("summary"), str) else "",
                    "recent": _copy_messages(rec.get("recent") or []),
                    "turns_summarized": rec.get("turns_summarized") or 0,
                }
            )
        cleaned.sort(key=lambda r: (r.get("updated_at") or ""), reverse=True)
        self.conversations = cleaned[: self.max_conversations]

    def save(self):
        """把当前对话记录写回硬盘（临时文件 + os.replace，写坏了旧文件还在）。"""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "version": SUMMARY_FILE_VERSION,
                    "conversations": self.conversations,
                },
                ensure_ascii=False,
            )
            fd, tmp_name = tempfile.mkstemp(
                prefix=self.file_path.name + ".tmp-",
                dir=str(self.file_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp_name, self.file_path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except Exception:
            # 摘要存不下来不能把聊天搞崩：记一条日志，内存里的记录还在，
            # 下次 save 再试。
            logger.warning("对话摘要保存失败。", exc_info=True)

    def _sort(self):
        """记录按更新时间从新到旧排，多了的从旧的开始删。"""
        self.conversations.sort(key=lambda r: (r.get("updated_at") or ""), reverse=True)
        del self.conversations[self.max_conversations :]

    def _new_record(self):
        """一条空的对话记录（还没进列表，进列表是 sync/flush 的事）。"""
        return {
            "id": uuid.uuid4().hex[:12],
            "title": "新对话",
            "started_at": None,
            "updated_at": None,
            "summary": "",
            "recent": [],
            "turns_summarized": 0,
        }

    def _ensure_current(self):
        """保证有一条“正在聊”的记录（没有时新建一条，不进列表）。"""
        record = self._current_record
        if record is not None and record["id"] == self.current_id:
            return record
        if self.current_id is not None:
            record = self._find_record(self.current_id)
            if record is not None:
                self._current_record = record
                return record
        # current_id 指的那条已经不在列表里了（比如被 20 条上限挤掉），
        # 就当新对话开一条，不拿旧 id 继续写。
        record = self._new_record()
        self.current_id = record["id"]
        self._current_record = record
        return record

    def _find_record(self, conv_id):
        """按 id 找一条记录（找不到返回 None）。"""
        for rec in self.conversations:
            if rec["id"] == conv_id:
                return rec
        return None

    def _ensure_current_in_list(self, record):
        """正在聊的记录进列表（第一次记下来时），再按时间排好。"""
        if self._find_record(record["id"]) is None:
            self.conversations.append(record)
        self._sort()

    # ---------- 列表 ----------

    def list_conversations(self):
        """最近对话列表（给网页左边用）：每条只有 id、标题、时间，不带正文。"""
        return [
            {
                "id": rec["id"],
                "title": rec.get("title") or "新对话",
                "started_at": rec.get("started_at"),
                "updated_at": rec.get("updated_at"),
            }
            for rec in self.conversations
        ]

    def get_conversation(self, conv_id):
        """按 id 取一条记录的完整内容（标题、摘要、最近几轮），找不到返回 None。"""
        rec = self._find_record(conv_id)
        if rec is None:
            return None
        return {
            "id": rec["id"],
            "title": rec.get("title") or "新对话",
            "started_at": rec.get("started_at"),
            "updated_at": rec.get("updated_at"),
            "summary": rec.get("summary") or "",
            "recent": _copy_messages(rec.get("recent") or []),
            "turns_summarized": rec.get("turns_summarized") or 0,
        }

    def begin_conversation(self):
        """开始一段新对话：正在聊的换成一条空记录（不进列表，等第一轮再记）。

        调用方在调之前要先把上一段对话 flush 掉（总结保存），
        不然上一段的尾巴就丢了。
        """
        record = self._new_record()
        self.current_id = record["id"]
        self._current_record = record
        self._last_flush_fingerprint = None
        return record["id"]

    def open_conversation(self, conv_id):
        """回到一段旧对话：正在聊的换成它，返回它的完整内容（找不到返回 None）。

        调用方在调之前要先把当前对话 flush 掉；调之后把返回的 recent
        当成新的对话历史，接着聊。返回的 recent 是原样留着的最近几轮，
        摘要在 "summary" 里。
        """
        rec = self._find_record(conv_id)
        if rec is None:
            return None
        self.current_id = rec["id"]
        self._current_record = rec
        self._last_flush_fingerprint = None
        return self.get_conversation(conv_id)

    # ---------- 摘要 ----------

    def current_summary(self):
        """新对话提问时用的摘要：正在聊的那条有摘要就用它的；
        正在聊的是空记录（刚开新对话）时，用列表里最新那条的摘要——
        那就是“上一段对话”，让十二能回答“上一段对话聊了什么”。"""
        current = getattr(self, "_current_record", None)
        if current is not None and (current.get("summary") or "").strip():
            return current["summary"]
        if self.conversations:
            return self.conversations[0].get("summary") or ""
        return ""

    def has_summary(self):
        """有没有可用的摘要（新对话开始时要不要拼进提示）。"""
        return bool(self.current_summary().strip())

    def prompt_block(self):
        """给提问用的摘要提示块（没有摘要时返回空字符串）。"""
        return summary_block(self.current_summary())

    def _fingerprint(self, messages):
        """一段消息的指纹（角色 + 内容拼起来算 SHA-256），防重复 flush。"""
        joined = "\n".join(
            f"{(msg or {}).get('role')}:{(msg or {}).get('content') or ''}"
            for msg in (messages or [])
        )
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def sync(self, messages):
        """每轮后记一笔：标题、更新时间、最近几轮写进正在聊的记录，存硬盘。

        只记元数据和最近几轮，不总结（总结是 maybe_compact / flush 的事）。
        新对话第一轮后它自动出现在列表顶部。空消息时什么都不做。
        """
        if not messages:
            return False
        record = self._ensure_current()
        if not record.get("started_at"):
            record["started_at"] = _now_iso()
        if not (record.get("title") or "").strip() or record.get("title") == "新对话":
            record["title"] = _make_title(messages, record.get("summary"))
        record["updated_at"] = _now_iso()
        record["recent"] = _copy_messages(messages[-self.keep_last_rounds * 2 :])
        self._ensure_current_in_list(record)
        self.save()
        return True

    def maybe_compact(self, messages):
        """历史超上限时总结一次（最近 5 轮原样留着，只压更早的）。

        参数 messages：调用方持有的对话历史列表，本函数会原地裁剪它
        （删掉已总结的旧消息，只留最近 keep_last_rounds 轮）。
        返回 True（总结过、历史已裁剪）或 False（没到上限、没法总结、
        或总结失败——失败时历史不动，下一轮再试）。
        没配 summarizer 时直接返回 False，不做任何事。
        """
        if self.summarizer is None:
            return False
        if not messages:
            return False
        if estimate_messages_tokens(messages) <= self.token_limit:
            return False
        keep = self.keep_last_rounds * 2
        if len(messages) <= keep:
            # 历史已经只剩最近几轮，没什么可压的（多半是几条特别长的消息）；
            # 下一轮消息多了再压，不硬压。
            return False
        old_turns = list(messages[:-keep])
        record = self._ensure_current()
        new_summary = self.summarizer(old_turns, record.get("summary") or None)
        if not (new_summary or "").strip():
            # 总结失败：历史不动、旧摘要不动，下一轮再试。
            return False
        record["summary"] = _truncate_summary(new_summary.strip())
        record["turns_summarized"] += len(old_turns)
        record["updated_at"] = _now_iso()
        if not record.get("started_at"):
            record["started_at"] = record["updated_at"]
        if not (record.get("title") or "").strip() or record.get("title") == "新对话":
            record["title"] = _make_title(old_turns, record.get("summary"))
        del messages[:-keep]
        record["recent"] = _copy_messages(messages)
        self._ensure_current_in_list(record)
        self.save()
        return True

    def flush(self, messages):
        """对话结束时把剩下的历史总结进摘要、存硬盘。

        总结的是调用方手里剩下的全部消息（压缩后留下的最近几轮，
        或从没触发过压缩的整段对话）。空消息、重复 flush 同一份消息、
        没配 summarizer、总结失败时都返回 False，什么都不改。
        注意：恢复回来的最近几轮也会在这里再总结一次（它们在摘要里
        可能已经有一份梗概），合并时模型会去重，摘要只是梗概、
        轻微重复不影响用；但同一份消息绝不会被 flush 两遍
        （指纹拦着）。
        """
        if self.summarizer is None:
            return False
        if not messages:
            return False
        fingerprint = self._fingerprint(messages)
        if fingerprint == self._last_flush_fingerprint:
            # 同一份消息已经总结过一次了，不再总结第二遍。
            return False
        record = self._ensure_current()
        new_summary = self.summarizer(list(messages), record.get("summary") or None)
        if not (new_summary or "").strip():
            return False
        record["summary"] = _truncate_summary(new_summary.strip())
        record["turns_summarized"] += len(messages)
        record["updated_at"] = _now_iso()
        if not record.get("started_at"):
            record["started_at"] = record["updated_at"]
        if not (record.get("title") or "").strip() or record.get("title") == "新对话":
            record["title"] = _make_title(messages, record.get("summary"))
        record["recent"] = _copy_messages(messages[-self.keep_last_rounds * 2 :])
        self._ensure_current_in_list(record)
        self._last_flush_fingerprint = fingerprint
        self.save()
        return True
