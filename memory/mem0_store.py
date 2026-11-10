# memory/mem0_store.py（mem0 长期记忆适配层）
# 作用：项目里唯一直接碰 mem0 的模块，其他代码（main.py、web/server.py）
#       都只调这里，不直接 import mem0。以后换记忆后端，只改这一个文件。
# 原理：mem0 负责三件事——每轮对话后自动提取记忆、存进本地 Chroma 向量库、
#       提问时语义检索。这里只做薄薄一层包装：配好 DeepSeek + 本地
#       embedding + 本地 Chroma、强制关遥测、加阈值过滤、加写锁、记账、
#       存前脱敏（隐私过滤）、直接存入时去重、每条记忆带来源元数据、
#       纠错（改）与深度删除（忘）。
# 关系：main.py 的 run_question() 提问前调 search()、回答后调 add_turn()；
#       web/server.py 常驻服务里建一个 Mem0Store；命令行默认不建（见下）。
# 来源（provenance）：每条记忆存进 mem0 时都带一份 metadata——
#       source（user：BOOS 亲口说的/让记住的；inferred：DeepSeek 从对话
#       里提取的）、created_at（形成时间）、confidence（可信度）、
#       user_confirmed（BOOS 是否逐条确认过）。没有 metadata 的旧记忆，
#       取回时一律按“未确认”处理，绝不默认当成已确认的事实（安全降级）。
#       只有 user_confirmed=True 的记忆才能在回答里直接当作事实；
#       inferred 的只能当背景参考；未确认的不能当作事实（见 main.py
#       的 _memory_block，那里按来源给每条记忆打标签、分层）。
# 纠错（改）：用户说“你记错了”时调 correct_memory()——先找到唯一对得上
#       的那条，找不到/对不上多个时不猜、不改（返回状态让调用方问用户），
#       找准了才用 mem0.update() 原地替换（同一 id），再验旧文本确实
#       检索不到；改失败时原记忆不动、报失败，绝不谎称改好了。改过的
#       记忆来源变成 user（BOOS 亲口纠正的）、user_confirmed=True。
#       多个候选分不清时（比如“都喜欢…”的句式分数都差不多），
#       请 DeepSeek 按槽位判断改哪条（见 _judge_correction）：新说法
#       是在替换哪个语义槽位里的内容就选那条，上下文是意思分不开
#       时的决胜依据，第一次截断或“分不清”时用 2000 再试一次；
#       模型判选中后、真正替换之前，还有一道本地安全网：用 bge 向量
#       算旧记忆和新说法的相似度，话题不像就不换、改问 BOOS（见
#       _old_new_similarity、CORRECTION_TOPIC_MIN_SIMILARITY）；
#       前面都定不下来时才问 BOOS。第 1 层直接定已经去掉（2026-11-09
#       新方案，BOOS 定的）：上下文里恰好只有 1 条也不直接定，C3
#       证明那样会静默改错。correct_memory 的返回里有 "layer" 字段
#       标明哪一层定的，还有 "resolution" 和 "audit"（审计信息）。
#       注意：改只换向量库里的当前文本，history.db 里的 UPDATE 审计行
#       会留着旧文本——那是“改过什么”的审计记录，有意保留，和“忘”不同。
# 深度删除（忘）：delete_memory() 删的不只是向量——向量删掉之后，还要
#       把 history.db 里这条记忆的全部审计行（ADD/UPDATE/DELETE 行，
#       DELETE 行里还留着被删记忆的全文）清掉，再验向量、历史、检索
#       三处都干净。删之前先调 deletion_scope() 把删除范围（删哪条、
#       删几行历史、哪些地方不在删除范围内）展示给 BOOS 看，拿到
#       approve=True 才动手。删的过程中先写一个只含 id 和文本哈希
#       （不含原文）的删除日志，验完干净再清掉；中途崩了，日志留着，
#       下次能查到这次删了一半（见 list_pending_deletes）。
#       不在删除范围内的东西（实话）：history.db 里 messages 表的原始
#       对话记录（那是“说过什么话”的记录，不是记忆本身，删记忆不动它，
#       但它也不会被重新提取——提取只对新轮次跑）；BOOS 自己在 Agent
#       之外做的备份；Chroma 底层文件碎片（delete 后查不到，但文件层
#       的清理归 Chroma 管）。删除范围展示里会把这些逐项列出来。
# 关系：main.py 的 run_question() 提问前调 search()、回答后调 add_turn()；
#       web/server.py 常驻服务里建一个 Mem0Store；命令行默认不建（见下）。
# 安全与隐私：
#   - 遥测是强制关的（mem0 会把使用情况发到 PostHog），这里在 import mem0
#     之前就把 MEM0_TELEMETRY 强制设成 False，一点都不往外发；就算环境
#     里事先把它设成了 True，也会被盖掉。注意顺序：mem0 在 import 时就读
#     这个环境变量，建好 Memory 之后再关就晚了，所以必须在 import 之前设。
#     遥测和“提取要调 DeepSeek”是两回事：遥测是 mem0 自己往 PostHog
#     发使用情况（已关）；提取是把最近几轮对话发给 DeepSeek 做提取
#     （这是记忆功能本身，必须调 API，见下）。
#   - 敏感信息三层过滤（密码、API Key、令牌、身份证号、银行卡号）：
#     存之前先脱敏（秘密换成占位符再发给 DeepSeek 提取，见
#     _redact_sensitive）；用之前再拦一道（旧库里已有的敏感记忆不注入
#     提示，见 search）；导出/迁移时不带敏感内容（见 get_all、
#     migrate_from_v1）。日志里也绝不出现敏感原文。
#   - 什么会离开本地（实话）：每轮对话后做提取的那几轮消息会发给
#     DeepSeek（这是唯一一次记忆内容离开本地）；提问时拼进问题的相关
#     记忆也会发给模型。Chroma 向量库、history.db、embedding 模型
#     都在本地，不往外发。脱敏是按正则模式匹配的，不是万无一失——
#     没见过的密钥格式、中间带空格的多词密码可能漏网，所以这层是
#     “减少泄露面”，不是“保证绝不泄露”；真的密钥还是不要往对话里发。
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

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import unicodedata
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger("agent")

# ---------- 遥测开关（必须在 import mem0 之前设置） ----------
# mem0 的 telemetry.py 在 import 时读 MEM0_TELEMETRY，默认 True。
# 这里强制设成 False，再 import mem0，确保整个进程里遥测都是关的。
# 注意是强制设（不是 setdefault）：就算环境里事先有人把
# MEM0_TELEMETRY 设成了 True，也照样盖掉——遥测没有“调试时开一下”
# 的例外；_build_memory 里建 Memory 之前还会再强制设一次（见下）。
os.environ["MEM0_TELEMETRY"] = "False"

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

# CONTEXT_IDS_MAX（最近对话上下文记多少条记忆 id）：note_context 记的
# 最近几轮提问时检索到的记忆 id 上限（deque，满了旧的自动挤掉）。
# 20 条够覆盖最近几轮对话（每轮最多 5 条），又不会把很久以前的
# 话题一直留在“当前在聊什么”里。
CONTEXT_IDS_MAX = 20

# JUDGE_MAX_TOKENS（纠错判断调用的输出上限）：模型只回一个 JSON
# （{"choice": "<id>" 或 "none"}），答案本身只要约 30 Token；但
# deepseek-flash 默认开着思考，思考先占 100–200 个 Token，200 的
# 上限经常在答案写出来之前就截断（2026-11-09 实测判断调用 5 次只
# 对了 2 次）。提到 1000，留足思考的余量；思考保持开启，判断哪条
# 旧记忆冲突需要想一想。模型用多少算多少，上限只是防截断，不多花钱。
JUDGE_MAX_TOKENS = 1000

# JUDGE_RETRY_MAX_TOKENS（纠错判断重试的输出上限）：第一次判断调用
# （JUDGE_MAX_TOKENS = 1000）被截断（finish_reason == "length"），
# 或者模型回 none 但原因是 "similar"（两条都像、分不清）时，用
# 2000 再试一次——同一个提示，只重试一次，绝不无限循环。第一次回
# none 且原因是 "unrelated"（哪个都不像、没关系）时不重试，直接问
# BOOS；调用异常（网络等）也不重试。2000 再试还是不行（截断或
# none）→ 返回 None，走问 BOOS。思考保持开启，temperature=0 不变。
JUDGE_RETRY_MAX_TOKENS = 2000

# CORRECTION_TOPIC_MIN_SIMILARITY（纠错安全网：旧-新话题相似度下限）：
# 模型判选中一条候选后、真正替换之前，用本地 bge 模型算旧记忆文本
# 和新说法的余弦相似度；低于这个值就不换，改问 BOOS（见
# Mem0Store._old_new_similarity）。2026-11-09 实测校准
# （bge-small-zh-v1.5，短句都是“我喜欢…”结构，所以分数整体偏高）：
# 同话题都在 0.59 以上——科幻电影→纪录片 0.629、纪录片→恐怖电影
# 0.719、恐怖电影→纪录片 0.595、喜剧电影→纪录片 0.641；跨话题都在
# 0.54 以下——Go 写代码→纪录片 0.531、Go 写代码→恐怖电影 0.443。
# 0.56 是两边的中点（最坏边距约 0.03）。分得开但边距很窄——这道网
# 只是第二层，主防线是去掉第 1 层直接定 + 模型按槽位判；阈值偏高
# 只会多问 BOOS 一次（安全方向），偏低才可能漏过。换 embedding
# 模型要重测这个值。
CORRECTION_TOPIC_MIN_SIMILARITY = 0.56

# CUSTOM_INSTRUCTIONS_VERSION（自定义提取规则的版本号）：
# 规则文本改了就改这个版本号，方便以后追踪“哪版规则存的记忆”。
# 2026-11-09 改到第二版：加了“价值判断”（只存值得长期记的）和
# “不存未确认/不存别人的事/不存没被认可的 AI 观点”（见下文第 9–12 条）。
# 2026-11-09 Plan C 改到第三版：加第 13 条（玩笑/夸张/不可能的说法不存）。
CUSTOM_INSTRUCTIONS_VERSION = "2026-11-09-plan-c"

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
   brand names and model names exactly as written.
9. VALUE JUDGEMENT — only store information with lasting value: long-term goals,
   stable preferences, important decisions, completed items, and
   facts the user will still need weeks from now. Do NOT store trivial
   chat, passing remarks, transient states, or one-off small talk, even if
   it is about the user. When in doubt whether something is worth keeping
   for the long term, do NOT store it: unconfirmed / not-worth-remembering
   content must not be stored at all (there is no "unconfirmed" storage
   state and no review queue — it simply is not saved).
10. NEVER store other people's experiences, preferences or words as facts
    about the user. "My friend likes hiking" is a fact about the friend,
    not about the user — do not store that the user likes hiking.
11. NEVER store the assistant's own suggestions, recommendations or views
    as the user's stance. An AI view only becomes a user fact when the user
    explicitly endorses it in their own words ("I agree", "yes, do that").
12. Apply rules 9-11 per fact, not per turn: a turn may mix trivial chat
    with a real long-term fact — store the fact, drop the trivial part.
    Never drop a whole turn just because part of it is trivial, and never
    keep a trivial sentence just because it shares a turn with a real fact.
13. NEVER store jokes, exaggerations, or claims that are plainly not
    serious or not possible as settled facts (e.g. "I am 100 years old",
    "I am a billionaire", "I can fly"). If a statement reads as a joke,
    a boast, or an impossible claim rather than a sincere fact about the
    user, do NOT store it. When in doubt whether the user is joking, do
    NOT store it — a missed joke is harmless, a stored joke becomes a
    false memory the user has to correct later."""

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


# ---------- 来源元数据（provenance） ----------
# 每条记忆存进 mem0 时带的 metadata（见文件头“来源”一节）：
#   source：记忆从哪来。"user"——BOOS 亲口说的/让记住的/亲口纠正的；
#           "inferred"——DeepSeek 从对话里提取的（用户明确说过的话，
#           但 BOOS 没逐条确认过那个写法）。
#   created_at：记忆形成的时间（UTC 的 ISO 时间串，精确到秒）。
#   confidence：可信度 0–1。BOOS 亲口说的记 1.0；提取的记 0.7——
#           不是因为提取得不准，而是因为“模型提的”本身就比“用户
#           亲口确认的”低一档；模型用它来判断这条记忆有多可信。
#   user_confirmed：BOOS 是否逐条确认过。只在 source="user" 时为 True。
#           注意：提取的记忆哪怕存下来了，也绝不会因为“存下来了”就
#           自动变成已确认——source 还是 "inferred"，user_confirmed
#           还是 False；只有 BOOS 亲口说/纠正/让记住，才是已确认。
# 没有 metadata 的旧记忆（2026-11-09 之前存的）：取回时一律按
# "unconfirmed" 处理（见 _normalize_metadata），绝不默认已确认。

SOURCE_USER = "user"
SOURCE_INFERRED = "inferred"
SOURCE_UNCONFIRMED = "unconfirmed"
CONFIDENCE_USER = 1.0
CONFIDENCE_INFERRED = 0.7


def _now_iso():
    """现在的时间，UTC 的 ISO 时间串（精确到秒），供 metadata 用。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _provenance(source, confidence, user_confirmed, created_at=None, slot=None):
    """拼一份来源元数据（存记忆时传给 mem0.add 的 metadata 参数）。

    source 不是 "user" 时一律按 "inferred" 存（调用方传错也不让它
    进库）；user_confirmed 只在 source="user" 时才可能是 True。
    slot（Plan C）：身份/称呼类记忆的槽位名（agent_name/user_name/
    soul），只认 KNOWN_SLOTS 里的，其他值不进库——同槽位只留一条，
    新值盖旧值时靠它找旧记忆。
    """
    source = SOURCE_USER if source == SOURCE_USER else SOURCE_INFERRED
    metadata = {
        "source": source,
        "created_at": created_at or _now_iso(),
        "confidence": confidence,
        "user_confirmed": bool(user_confirmed) and source == SOURCE_USER,
    }
    if slot in KNOWN_SLOTS:
        metadata["slot"] = slot
    return metadata


def _normalize_metadata(raw):
    """把 mem0 返回的 metadata 规整成统一形状（检索时用）。

    规整规则（安全降级，宁可信度低、不可误认）：
    - raw 不是字典、或缺 source、或 source 不是 user/inferred，
      一律按 "unconfirmed" 处理（旧记忆、新旧版本混用、损坏数据
      都走这条路）；
    - user_confirmed 只有明确是 True（布尔值）才算确认，写成
      "yes"/1 这类非布尔值不算——确认必须是明确的，不能靠猜；
    - confidence 不是 0–1 之间的数字时记成 None（不知道就是不知道）；
    - created_at 不是字符串时记成 None；
    - slot 不是 KNOWN_SLOTS 里的已知槽位时记成 None（旧记忆、新旧
      版本混用、损坏数据都走这条路——槽位只认这三个，别的都不认）。
    返回：{"source", "created_at", "confidence", "user_confirmed",
           "confirmed", "slot"}，其中 confirmed 就是 user_confirmed
    （取了个短名字，方便调用方判断“能不能当作事实用”）。
    """
    source = SOURCE_UNCONFIRMED
    created_at = None
    confidence = None
    user_confirmed = False
    slot = None
    if isinstance(raw, dict):
        raw_source = raw.get("source")
        if raw_source in (SOURCE_USER, SOURCE_INFERRED):
            source = raw_source
        raw_created = raw.get("created_at")
        if isinstance(raw_created, str) and raw_created:
            created_at = raw_created
        raw_conf = raw.get("confidence")
        if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool):
            if 0 <= raw_conf <= 1:
                confidence = float(raw_conf)
        user_confirmed = raw.get("user_confirmed") is True and source == SOURCE_USER
        raw_slot = raw.get("slot")
        if raw_slot in KNOWN_SLOTS:
            slot = raw_slot
    return {
        "source": source,
        "created_at": created_at,
        "confidence": confidence,
        "user_confirmed": user_confirmed,
        "confirmed": user_confirmed,
        "slot": slot,
    }


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


# ---------- 敏感信息过滤（隐私三层过滤：存前脱敏 / 用时拦截 / 导出排除） ----------
# 为什么要有这一层：记忆里可能出现密码、API Key、身份证号、银行卡号——
# 这些东西一旦进了记忆，有三条路会往外漏：提取时发给 DeepSeek、
# 使用时拼进发给模型的问题、导出/迁移时被整批读出来。
# 所以过滤做在三个地方（见 Mem0Store 里各方法的注释），日志里也绝不
# 出现原文。注意：脱敏是按正则模式匹配的，不是万无一失——没见过的
# 密钥格式、中间带空格的多词密码可能漏网；这层是“减少泄露面”，
# 不是“保证绝不泄露”；真的密钥还是不要往对话里发。

# _REDACTED（脱敏占位符）：秘密被替换成的文字。占位符本身不含任何
# 原文信息，提取器看到它也不会把它当成事实存下来。
_REDACTED = "【已隐藏敏感信息】"

# _PRIVATE_KEY_RE（私钥块）：-----BEGIN ... PRIVATE KEY----- 整块都是秘密，
# 整块替换，不留一行。
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

# _JWT_RE（JWT 令牌）：eyJ 开头的三段式（header.payload.signature）。
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
)

# _KNOWN_KEY_RES（已知密钥格式，不带标签也认得）：
# OpenAI/Anthropic 风格的 sk-、GitHub 的 ghp_/gho_/github_pat_、
# AWS 的 AKIA、Google 的 AIza、Slack 的 xox、HuggingFace 的 hf_。
# 前面的 (?<![A-Za-z0-9...]) 是为了不把长单词里的一段误当成密钥
# （比如 task-abcdef... 里的 "sk-..."）。
_KNOWN_KEY_RES = [
    re.compile(r"(?<![A-Za-z0-9-])sk-(?:ant|proj)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9-])sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,}"),
]

# _ID18_RE（18 位身份证号）：6 位地区 + 8 位生日 + 3 位顺序 + 1 位校验。
# 只认生日像生日的（19/20 年开头、月份 01–12、日期 01–31），
# 再加校验位验算（见 _id18_check_ok），避免把 18 位订单号误伤。
# 注意不用 \b：身份证号前面常直接跟中文（“身份证号是11010...”），
# 中文和数字之间没有 \b；改用“前后不能是数字”来定界。
_ID18_RE = re.compile(
    r"(?<![0-9])\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9])"
)
# _ID15_RE（15 位老身份证号）：6 位地区 + 6 位生日（YYMMDD）+ 3 位顺序。
_ID15_RE = re.compile(
    r"(?<![0-9])\d{6}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}(?![0-9])"
)
# _CARD_CANDIDATE_RE（银行卡号候选）：13–19 位数字，中间可带空格或短横。
# 光长度不够，还要过 Luhn 校验（见 _luhn_ok）才算，避免把手机号、
# 订单号这种长数字串误伤。
_CARD_CANDIDATE_RE = re.compile(r"(?<![0-9])\d(?:[ -]?\d){12,18}(?![0-9])")

# _SECRET_LABELS（带标签的秘密）：中文/英文里“某某是/为/：秘密”的说法。
# 标签后面跟的才是秘密值，只替换值、保留标签和前面的正常内容，
# 这样“喜欢喝咖啡，密码是 xxx”脱敏后还剩“喜欢喝咖啡”这句有用的。
# 值至少 3 个字符（“密码是多少”这种问句里的“多少”只有 2 个字，不会误伤），
# 遇到空格、逗号、句号就停（值里带空格的多词密码脱不干净，是已知局限，
# 见文件头“实话”）。
_SECRET_LABELS = (
    r"密码|口令|passwd|password|pwd",
    r"api[_ -]?key|api[_ -]?密钥|密钥|秘钥",
    r"secret|token|令牌|access[_ -]?key|auth[_ -]?token",
    r"身份证(?:号)?|银行卡号|信用卡号|卡号",
)
_LABELED_SECRET_RE = re.compile(
    r"(?:"
    + "|".join(_SECRET_LABELS)
    + r")\s*(?:是|为|is|=|:|：)\s*([^\s，,、；;。！？\n]{3,})",
    re.IGNORECASE,
)

# _GENERIC_TOKEN_RE（没见过的长令牌）：32 位以上、同时含大小写字母和
# 数字的连续字符串（字母数字 _ -），大概率是某种 token/key。
# 要求混搭（纯小写十六进制的 git hash 不算、纯数字不算），且前面不能是
# / . : 或字母数字（网址、路径里的长段不算，避免把链接误伤）。
_GENERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_/.:-])"
    r"(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*\d)"
    r"[A-Za-z0-9_-]{32,}"
)

_ID18_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID18_CHECK_CHARS = "10X98765432"


def _id18_check_ok(number):
    """验算 18 位身份证号的校验位（GB 11643 加权求和取模）。"""
    if len(number) != 18:
        return False
    total = sum(int(digit) * weight for digit, weight in zip(number[:17], _ID18_WEIGHTS))
    return _ID18_CHECK_CHARS[total % 11] == number[17].upper()


def _luhn_ok(number):
    """Luhn 校验（银行卡号用的校验算法）：从右往左隔位翻倍、减 9，
    各位相加能被 10 整除就是合法卡号。"""
    digits = [int(ch) for ch in number if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _redact_sensitive(text):
    """把一段文本里的敏感信息替换成占位符（只换秘密本身，保留其他内容）。

    顺序有讲究：先换整块/定格式的（私钥块、JWT、已知密钥格式、
    身份证号、银行卡号），再换带标签的（“密码是 xxx”——标签值里
    也可能是一串卡号，先换卡号能把带空格的卡号整段换掉），
    最后换没见过的长令牌。没有敏感信息时原样返回。
    注意：这个函数绝不能记日志、绝不能把原文写进任何地方——
    它处理的就是不能外泄的东西。
    """
    if not isinstance(text, str) or not text:
        return text
    redacted = _PRIVATE_KEY_RE.sub(_REDACTED, text)
    redacted = _JWT_RE.sub(_REDACTED, redacted)
    for pattern in _KNOWN_KEY_RES:
        redacted = pattern.sub(_REDACTED, redacted)
    redacted = _ID18_RE.sub(
        lambda m: _REDACTED if _id18_check_ok(m.group(0)) else m.group(0), redacted
    )
    redacted = _ID15_RE.sub(_REDACTED, redacted)
    redacted = _CARD_CANDIDATE_RE.sub(
        lambda m: _REDACTED if _luhn_ok(m.group(0)) else m.group(0), redacted
    )
    redacted = _LABELED_SECRET_RE.sub(
        lambda m: m.group(0)[: m.start(1) - m.start(0)] + _REDACTED, redacted
    )
    redacted = _GENERIC_TOKEN_RE.sub(_REDACTED, redacted)
    return redacted


def _contains_sensitive(text):
    """一段文本里是否含有敏感信息（按上面的模式判断）。"""
    if not isinstance(text, str) or not text:
        return False
    return _redact_sensitive(text) != text


# _MEANINGLESS_RES（脱敏后判断“还剩不剩有用的内容”时去掉的东西）：
# 占位符、标签词（密码/密钥/……）、虚词和标点都不算内容——
# “我的密码是 xxx”脱敏后只剩“我的密码是”，这不叫“有用的记忆”，
# 不值得存；“我喜欢喝咖啡，我的密码是 xxx”脱敏后还剩“喜欢喝咖啡”，
# 这才叫有用的记忆，要留下来。注意这三个正则只用来判断“还剩不剩”，
# 不用来改要存的文本。
_MEANINGLESS_RES = [
    re.compile(re.escape(_REDACTED)),
    re.compile(
        "|".join(_SECRET_LABELS),
        re.IGNORECASE,
    ),
    re.compile(r"[我的为有在和与了呢吗是：:，,、；;。！？\s]+"),
]


def _meaningful_after_redaction(text):
    """脱敏后的文本里还剩不剩有用的内容（去掉占位符、标签词、虚词后
    还有东西才算有）。纯秘密（“我的密码是 xxx”）脱敏后什么都不剩，
    返回 False，调用方直接不存；混合内容（“喜欢喝咖啡，密码是 xxx”）
    还剩“喜欢喝咖啡”，返回 True，调用方存脱敏后的版本。"""
    if not isinstance(text, str):
        return False
    rest = text
    for pattern in _MEANINGLESS_RES:
        rest = pattern.sub("", rest)
    return bool(rest.strip())


def _normalize_memory_text(text):
    """把一条记忆文本规整成可比的形状（供直接存入的去重用）：
    全角转半角（Ａ→A）、去所有空白、去末尾句末标点、英文转小写。
    “我每天晚上都读半小时书。”和“我每天晚上都读半小时书”
    （只差一个句号）算同一条，不重复存。"""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip("。！？；，、.!?;,")
    return normalized.lower()


def _find_duplicate_memory(text, existing_items):
    """在已有记忆里找和 text 相同（或只差标点/空白/大小写）的那条，
    找到返回那一项（{"id", "text"}），找不到返回 None。"""
    normalized = _normalize_memory_text(text)
    if not normalized:
        return None
    for item in existing_items or []:
        if _normalize_memory_text(item.get("text") or "") == normalized:
            return item
    return None


# ---------- 聊天纠错的意图识别（供 Mem0Store.add_turn 用） ----------
# 只认明确的纠错说法，宁可漏认、不要误认——普通聊天绝不能触发纠错。
# “我不是”要特别小心：“我不是很确定”只是犹豫，不是纠错；所以
# “我不是”后面必须跟一个替换说法（“而是/其实是/，新说法”），
# 否则不触发（见 _detect_correction）。

# _CORRECTION_PHRASES（纠错说法）：只认这五个明确的说法。
_CORRECTION_PHRASES = ("你记错了", "不对", "纠正一下", "其实是", "我不是")

# _CORRECTION_SEPARATORS（纠错说法前面能跟的分隔符）：纠错说法必须在
# 句首，或在句号/逗号/换行之后——比如“我喜欢科幻电影。不对，我喜欢
# 纪录片。”这种一句里先肯定后纠正的也认；但“我觉得不对劲”里的
# “不对”前面是“得”，不是分隔符，不认。
_CORRECTION_SEPARATORS = set("。！？；\n，、： ")

# _BUDUI_BOUNDARY（“不对”后面能跟的字）：“不对”后面必须是个边界，
# 不然可能是“不对劲/不对头”这种词的一部分——“不对劲，我先走了”
# 不是纠错。标点、语气词（啊/呀/哦/呢/吧）、人称（我/你/他/这/那）
# 都算边界；其他汉字（劲/头/味/路……）不算。
_BUDUI_BOUNDARY = set("，。：、；！？\n \t啊呀哦呢吧我你他她它这那是")

# _CHINESE_NUMERALS（挑候选时认的中文数字）：“第 1 条”里的 1 是阿拉伯
# 数字，“第一个”里的“一”是中文数字，都认。
_CHINESE_NUMERALS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _detect_correction(text):
    """从一句话里认出“纠错意图 + 新说法”。

    返回 {"phrase": 触发的说法, "fact": 新说法}，不是纠错时返回 None。
    认法（保守，只认明确的）：
    - 说法必须在句首，或在句号/逗号/换行等分隔符之后；
    - “不对”后面必须是个边界（标点/语气词/人称），排除“不对劲”这种词；
    - 新说法 = 说法后面那一句的话（到第一个句号为止），同一轮里
      后面的句子是别的内容，不算进新说法里；
    - “我不是 X，Y”这种否定句式：Y 才是新说法——先看“而是/其实是”，
      没有就取第一个逗号后面的；两个都没有（比如“我不是很确定”）
      就不是纠错，返回 None；
    - 新说法太短（只剩一个字或空）不算纠错，走正常保存。
    例子：
    - “不对，我喜欢看纪录片。”→ fact “我喜欢看纪录片。”
    - “你记错了，我喜欢用 Go 写代码”→ fact “我喜欢用 Go 写代码”
    - “我不是喜欢科幻电影，我喜欢纪录片。”→ fact “我喜欢纪录片。”
    - “我不是很确定”→ None（只是犹豫，没有替换说法）
    - “我觉得不对劲”→ None（“不对”是词的一部分）
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    best = None  # (出现位置, 说法)：多个说法同时出现时取最早的。
    for phrase in _CORRECTION_PHRASES:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            start = idx + 1
            if idx != 0 and text[idx - 1] not in _CORRECTION_SEPARATORS:
                continue
            if phrase == "不对":
                after = text[idx + len(phrase):idx + len(phrase) + 1]
                if after and after not in _BUDUI_BOUNDARY:
                    continue
            if best is None or idx < best[0]:
                best = (idx, phrase)
            break  # 同一个说法只取第一次出现。
    if best is None:
        return None
    idx, phrase = best
    fact = text[idx + len(phrase):].lstrip("，,：:、；; \t")
    if phrase == "我不是":
        # 否定句式：新说法在“而是/其实是”后面，或在第一个逗号后面。
        for marker in ("而是", "其实是"):
            pos = fact.find(marker)
            if pos >= 0:
                fact = fact[pos + len(marker):].lstrip("，,：:、；; \t")
                break
        else:
            comma = fact.find("，")
            if comma < 0:
                comma = fact.find(",")
            if comma >= 0:
                fact = fact[comma + 1:].lstrip(" \t")
            else:
                return None
    fact = fact.strip()
    # 新说法只取纠错说法所在的那一句——同一轮里后面的句子是别的内容，
    # 不该被纠正的新文本吞进去（2 号：不然“我喜欢喝咖啡”会被存进
    # 名字记忆里，后面待定区提交时又存一条，重复且文本被污染）。
    term = re.search(r"[。！？\n]", fact)
    if term:
        fact = fact[:term.end()].strip()
    # 新说法太短（去掉句末标点只剩一个字、或空）不算纠错，走正常保存。
    if len(fact.rstrip("。！？；，、.!?;,")) < 2:
        return None
    return {"phrase": phrase, "fact": fact}


def _latest_user_content(messages):
    """取消息列表里最后一条 user 消息的内容（本轮用户刚说的那句）。

    add_turn 收到的可能是最近几轮的消息（CLI/网页都只带最近 6 条），
    纠错只看最新那句——以前轮次的纠错在它那轮已经处理过了。
    没有 user 消息时返回 ""。
    """
    for message in reversed(messages or []):
        if (message or {}).get("role") == "user":
            return (message or {}).get("content") or ""
    return ""


def _strip_correction_sentences(latest_user):
    """去掉纠错句，留非纠正部分（Plan C 2 号）。

    一句一句看：含纠错说法的那句，只留说法前面的部分（有实质内容
    才留，比如“我喜欢喝咖啡，你记错了，我叫小宋”里留“我喜欢喝咖啡”），
    说法后面的新说法已由 correct_memory 处理，不再重复存；
    没有纠错说法的句子原样留。
    返回剩下的句子列表，没有剩下时返回 []。
    """
    kept = []
    for sentence in re.split(r"[。！？\n]+", latest_user or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        detected = _detect_correction(sentence)
        if detected is None:
            kept.append(sentence)
            continue
        before = sentence[:sentence.find(detected["phrase"])].strip()
        before = before.strip("，,：:、；; \t")
        if len(before) >= 2:
            kept.append(before)
    return kept


# ---------- Plan C：记忆信任修复（2026-11-09，BOOS 批准动手） ----------
# 问题：十二不敢信 BOOS 亲口说的名字和 SOUL.md——新对话里问名字，
# 它说“未经逐条确认的信息”、SOUL.md“没有任何记录”。根因是确认
# 模型太严，连用户亲口说的都不敢信。治本办法（三闸门 + 待定区 +
# 冲突兜底），落地顺序见 add_turn / _commit_pending_turn：
#   闸门 1（话语类型过滤）：假设句、玩笑夸张不存（见
#     _filter_extraction_messages + CUSTOM_INSTRUCTIONS 第 13 条）；
#   闸门 2（槽位分级）：名字/称呼/重要偏好是 BOOS 亲口说的，代码
#     直接标 user_explicit（source="user"、user_confirmed=True、
#     confidence=1.0），来源由代码定、不由模型定（见
#     _detect_identity_statement + _upsert_slot_memory）；
#   闸门 3（待定区 + 冲突检查）：新记忆先在内存里待一轮对话，
#     下一轮不撤回才真正存入；“不不不”这类撤回信号直接丢掉待定
#     的那轮，沉默（没有下一轮）则待定区清空、不存；同槽位新值
#     来时原地 UPDATE，不并存（mem0 本身是 ADD-only，冲突处理
#     必须在应用层做，见 _upsert_slot_memory）。
# 记忆面板只做事后修正兜底，不当确认入口。

# 槽位（slot）：身份/称呼类记忆在 metadata 里带的槽位名，
# 同槽位只留一条（新值盖旧值）。KNOWN_SLOTS 之外的 slot 一律不认。
SLOT_AGENT_NAME = "agent_name"  # Agent 自己的名字（十二）
SLOT_USER_NAME = "user_name"  # BOOS 希望被怎么称呼
SLOT_SOUL = "soul"  # Agent 的人格（灵魂，SOUL.md）
KNOWN_SLOTS = frozenset([SLOT_AGENT_NAME, SLOT_USER_NAME, SLOT_SOUL])

# 身份种子（Plan C 种进长期记忆的两条“已确认”记忆）：
# 名字是 BOOS 2026-11-08 亲口定的、2026-11-09 亲口告诉十二的；
# SOUL.md 是 BOOS 逐条定的十条原则。种子只在“没有时”才加，
# 绝不覆盖已有的槽位值（比如 BOOS 以后在聊天里改了名字，
# 下次启动不能把名字改回去）。
AGENT_NAME_VALUE = "十二"
# 种子文本里特意留了“你叫十二”：_detect_identity_statement 要能从
# 这句里认出 agent_name=十二——以后 BOOS 在聊天里再说“你叫十二”，
# upsert 比对旧值对得上，才不会误判成改名、把“是十一的弟弟”盖掉。
AGENT_NAME_TEXT = "Agent 的名字是十二，你叫十二，是十一的弟弟。"
AGENT_SOUL_TEXT = (
    "Agent 的人格（灵魂）写在 SOUL.md 里：十条原则——"
    "真诚帮忙不装样子、有主见、自己先想办法再问、重要的事先问、"
    "亦师亦友、温暖不冷淡、记住并不断学习、诚实绝不编造、有原则、"
    "BOOS 明确要求无条件服从时就服从。"
)

# _RETRACTION_SIGNALS（撤回信号）：整句只有这些词（加句末标点）时，
# 意思是“刚说的那句撤回、不记了”——直接丢掉待定区里的上一轮，
# 不存、不问。只认“整句只有信号”：“不对，我喜欢看纪录片。”这种
# 带替换内容的不是撤回，走纠错。没有“没有”——光秃秃的“没有”常常
# 是在回答助手刚问的问题（“你喜欢喝茶吗？”“没有。”），不是撤回
# 上一句，误杀会丢掉刚说的重要事实。
_RETRACTION_SIGNALS = frozenset(
    [
        "不不不",
        "不对不对",
        "不不",
        "不对",
        "不是",
        "算了",
        "别记",
        "别记了",
        "不要记",
        "不要记了",
        "不记了",
        "别存",
        "不要存",
        "错了",
        "说错了",
        "收回",
        "撤回",
    ]
)
_RETRACTION_TRAILING_PUNCT = "。！？.!?~～ \t\n"

# 条件句开头：不是既定事实，提取时整句去掉（和 FakeMemory 认同一套）。
_CONDITIONAL_LEADS = ("如果", "假设", "假如", "要是", "万一")

# 玩笑标记：句里出现这些词，这句就是在开玩笑/说着玩，不是既定事实
# （Plan C 闸门 1 的代码兜底；提示词第 13 条仍在，模型判断是第一道，
# 这里是第二道——模型哪天没拦住，代码拦）。
# 写法收紧：只认带明确标记的说法——“我喜欢开玩笑”这种真偏好不带
# 这些标记，不会被误拦。
_JOKE_MARKERS = (
    "开玩笑的", "开玩笑呢", "开玩笑了", "我开玩笑", "只是开玩笑",
    "开个玩笑", "逗你玩", "闹着玩", "说着玩", "别当真",
    "瞎说的", "胡说的",
)

# 明显不可能的数字（只认这两类最稳的，其余交给模型判断）：
# - 年龄超过 120 岁（人类最长寿 122 岁，留一点余量）；
# - 身高超过 3 米（人类最高 2.72 米）。
_JOKE_AGE_LIMIT = 120
_JOKE_HEIGHT_LIMIT_M = 3.0
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "两": 2}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_AGE_RE = re.compile(r"([一二三四五六七八九十百千两\d]+)\s*岁")
# 身高只认两种写法：“身高 3.5 米”“我 3.5 米高”——“我 100 米跑”
# 这种跑步距离不认（“米”后面没有“高”，也不是“身高”开头）。
_HEIGHT_RE = re.compile(
    r"(?:身高\s*([一二三四五六七八九十百千两\d]+(?:\.\d+)?)\s*米"
    r"|我\s*([一二三四五六七八九十百千两\d]+(?:\.\d+)?)\s*米高)"
)

# 身份陈述的触发词：按“先 user、后 agent”试——“你叫我小宋”里的
# “你叫”是叫“我”，不是在给 Agent 起名；每组内按触发词从长到短试。
_IDENTITY_USER_TRIGGERS = ("我的名字是", "我的名字叫", "你叫我", "叫我", "我叫")
_IDENTITY_AGENT_TRIGGERS = ("以后你叫", "现在你叫", "你的名字是", "你的名字叫", "你叫")
_IDENTITY_VALUE_MAX_LEN = 12  # 名字值最多 12 字
_IDENTITY_TRAILING_PARTICLES = "了啊呀呢吧哦嘛呗啦喽哈"  # 名字末尾的语气词，去掉
# 问句尾巴：值以这些结尾时整句是问句（“你叫十三好不好”是在问，
# 不是在起名），不当身份陈述认。
_IDENTITY_QUESTION_SUFFIXES = ("好不好", "行不行", "可以吗", "好吗")


def _is_bare_retraction(text):
    """整句只有撤回信号时返回 True（Plan C 待定区撤回）。

    “不不不。”“不对。”“算了。”这种——只有信号加句末标点。
    “不对，我喜欢看纪录片。”带替换内容，不是撤回，返回 False。
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip().strip(_RETRACTION_TRAILING_PUNCT).strip()
    return stripped in _RETRACTION_SIGNALS


def _detect_identity_statement(sentence):
    """从一句话里认出“身份/称呼陈述句”（Plan C 闸门 2，代码定、不是模型定）。

    只认陈述句：带“什么/吗/?”的一律不认（“你叫什么？”是提问，
    不是在告诉名字）。认到返回 {"slot", "value", "text"}：
    - agent_name（十二的名字）：“你叫十二”“以后你叫十二了”
      → text "Agent 的名字是十二，你叫十二"（带“你叫”二字，
      离线关键词召回问“你叫什么”时也能找着）；
    - user_name（BOOS 的称呼）：“我叫小宋”→ 原句存（去句末标点）；
      “叫我小宋”/“你叫我小宋”→ "我叫小宋，请称呼我小宋"。
    名字值最多 12 字，末尾的语气词（了/啊/呀/呢/吧/哦/嘛/呗/啦/喽/哈）去掉；
    值以“好不好/行不行/可以吗/好吗”结尾时整句是问句，返回 None。
    认不到返回 None。
    """
    if not isinstance(sentence, str):
        return None
    text = sentence.strip().strip("。！？；，、.!?;,")
    if not text:
        return None
    if any(q in text for q in ("什么", "吗", "？", "?")):
        return None
    if any(marker in text for marker in _JOKE_MARKERS):
        # 玩笑里的起名不当真（“开玩笑的，你叫十三”不是在改名）。
        return None
    for triggers, slot in (
        (_IDENTITY_USER_TRIGGERS, SLOT_USER_NAME),
        (_IDENTITY_AGENT_TRIGGERS, SLOT_AGENT_NAME),
    ):
        for trigger in triggers:
            idx = text.find(trigger)
            if idx < 0:
                continue
            rest = text[idx + len(trigger):]
            # 值到第一个标点为止。
            value = re.split(r"[，。！？、；;,\s]", rest, maxsplit=1)[0].strip()
            value = value.rstrip(_IDENTITY_TRAILING_PARTICLES)
            # 问句尾巴：整句是在问“这样叫好不好”，不是在起名。
            if any(value.endswith(suffix) for suffix in _IDENTITY_QUESTION_SUFFIXES):
                return None
            if not value or len(value) > _IDENTITY_VALUE_MAX_LEN:
                continue
            if slot == SLOT_USER_NAME:
                if trigger in ("叫我", "你叫我"):
                    canonical = f"我叫{value}，请称呼我{value}"
                else:
                    canonical = text  # “我叫小宋”这类原句直接存
            else:
                canonical = f"Agent 的名字是{value}，你叫{value}"
            return {"slot": slot, "value": value, "text": canonical}
    return None


def _parse_chinese_number(text):
    """把“十/百/千”组合的中文数字转成整数（只做年龄、身高这点小事）。

    一百五十 → 150，二十 → 20，一千二百三十四 → 1234。
    认不出来（混了阿拉伯数字、带“点”这种）返回 None，不猜。
    """
    if not text or not all(ch in _CN_DIGITS or ch in _CN_UNITS for ch in text):
        return None
    total = 0
    current = 0
    seen_unit = False
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        else:
            unit = _CN_UNITS[ch]
            if current == 0:
                current = 1  # “十”就是 10，“百”就是 100
            total += current * unit
            current = 0
            seen_unit = True
    total += current
    if not seen_unit and total == 0:
        return None
    return total


def _is_joke_or_exaggeration(sentence):
    """Plan C 闸门 1 的代码兜底：明显是玩笑/夸张的句子返回 True。

    只认最稳的三类，其余交给提示词第 13 条（模型判断）：
    - 句里带明确玩笑标记（开玩笑的/逗你玩/闹着玩/说着玩/别当真/
      瞎说的/胡说的）——整句当玩笑处理，连着的真话也一起去掉，
      宁可漏记一句（用户会再说），不把玩笑存成事实；
    - 年龄超过 120 岁（阿拉伯数字或中文数字，如 150 岁/一百五十岁）；
    - 身高超过 3 米（身高 3.5 米/我 3.5 米高）。
    “我一百岁了”（100 岁）、“姚明 2.26 米”这种没超线的正常话不拦。
    """
    if not isinstance(sentence, str) or not sentence:
        return False
    if any(marker in sentence for marker in _JOKE_MARKERS):
        return True
    age_match = _AGE_RE.search(sentence)
    if age_match:
        raw = age_match.group(1)
        age = int(raw) if raw.isdigit() else _parse_chinese_number(raw)
        if age is not None and age > _JOKE_AGE_LIMIT:
            return True
    height_match = _HEIGHT_RE.search(sentence)
    if height_match:
        raw = height_match.group(1) or height_match.group(2)
        if raw is None:
            return False
        if "." in raw:
            try:
                height = float(raw)
            except ValueError:
                height = None
        elif raw.isdigit():
            height = float(raw)
        else:
            parsed = _parse_chinese_number(raw)
            height = float(parsed) if parsed is not None else None
        if height is not None and height > _JOKE_HEIGHT_LIMIT_M:
            return True
    return False


def _filter_extraction_messages(messages):
    """Plan C 闸门 1：从待提交的对话里去掉不该进提取的东西，
    返回过滤后的消息列表（assistant 的话原样透传，只当提取上下文）。

    去掉三类 user 句子：
    - 条件/假设句开头的（如果/假设/假如/要是/万一）——不是既定事实；
    - 身份/称呼陈述句——身份事实只走确定性的槽位路径
      （_upsert_slot_memory），不经过模型提取，避免同一事实存两份；
    - 玩笑/夸张句（_is_joke_or_exaggeration）——明确玩笑标记、
      年龄超 120 岁、身高超 3 米；其余玩笑仍靠提示词第 13 条拦。
    一条 user 消息里句子全被去掉时，整条消息不要了。
    """
    kept = []
    for message in messages or []:
        role = (message or {}).get("role")
        content = (message or {}).get("content") or ""
        if role != "user":
            if role == "assistant":
                kept.append({"role": "assistant", "content": content})
            continue
        sentences = re.split(r"[。！？\n]+", content)
        kept_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence.startswith(_CONDITIONAL_LEADS):
                continue
            if _detect_identity_statement(sentence) is not None:
                continue
            if _is_joke_or_exaggeration(sentence):
                continue
            kept_sentences.append(sentence)
        if kept_sentences:
            kept.append(
                {"role": "user", "content": "。".join(kept_sentences) + "。"}
            )
    return kept


def _correction_result(outcome):
    """把 correct_memory 的返回收成调用方好用的样子（只留必要字段）。

    {"status", "memory_id", "old_text", "new_text",
     "candidates": [{"id", "text"}]（ambiguous 时给调用方展示用）,
     "reason": 没改成的原因（给人看的）,
     "layer": 哪一层定的（exact/single/judge/judge_blocked/ask/none，
     见 correct_memory）,
     "resolution": 这次纠正的结果（updated/judge_blocked/ask/
     not_found/unchanged/failed，见 correct_memory）,
     "audit": 审计信息（picked_id/judge_reason/old_new_similarity/
     candidate_scores/context_ids，见 correct_memory）}。
    """
    outcome = outcome or {}
    return {
        "status": outcome.get("status"),
        "memory_id": outcome.get("memory_id"),
        "old_text": outcome.get("old_text"),
        "new_text": outcome.get("new_text"),
        "candidates": [
            {"id": (c or {}).get("id"), "text": (c or {}).get("text")}
            for c in (outcome.get("candidates") or [])
        ],
        "reason": outcome.get("reason") or "",
        "layer": outcome.get("layer") or "none",
        "resolution": outcome.get("resolution") or "failed",
        "audit": outcome.get("audit") or {},
    }


def _parse_pending_selection(text, candidates):
    """用户回“第 1 条/第一个/1”或直接引用某条候选原文时，认出他指的是哪条。

    返回挑中的候选（{"id", "text"}），认不出时返回 None——认不出
    就当放弃这次纠错（add_turn 会清掉待确认、不再问），绝不猜一条。
    认法（按明确程度从高到低）：
    - “第 1 条 / 第1条 / 第一个 / 1”：数字指明第几条；
    - 直接引用候选原文：某条候选里连续 4 个字出现在用户的话里，
      且只有这一条对得上（两条都对得上时不猜，返回 None）。
      比如候选是“我喜欢看科幻电影”，用户说“科幻电影那条”——
      “科幻电影”这 4 个字只在这一条里，就认它。
    """
    if not isinstance(text, str) or not candidates:
        return None
    stripped = text.strip()
    match = re.search(r"第\s*(\d+)\s*条", stripped)
    if match:
        n = int(match.group(1))
        return candidates[n - 1] if 1 <= n <= len(candidates) else None
    match = re.search(r"第\s*([一二三四五六七八九十])\s*个", stripped)
    if match:
        n = _CHINESE_NUMERALS[match.group(1)]
        return candidates[n - 1] if 1 <= n <= len(candidates) else None
    if re.fullmatch(r"\d+", stripped):
        n = int(stripped)
        return candidates[n - 1] if 1 <= n <= len(candidates) else None
    # 直接引用原文：候选里连续 4 个字出现在用户的话里，且只有一条对得上。
    norm_text = _normalize_memory_text(stripped)
    matched = []
    for cand in candidates:
        norm_cand = _normalize_memory_text((cand or {}).get("text") or "")
        if len(norm_cand) < 4:
            continue
        windows = {norm_cand[i:i + 4] for i in range(len(norm_cand) - 3)}
        if any(window in norm_text for window in windows):
            matched.append(cand)
    if len(matched) == 1:
        return matched[0]
    return None


def _parse_judge_choice(content, candidates):
    """从 DeepSeek 纠错判断的返回里抠出它选的候选 id（供 _judge_correction 用）。

    认三种写法：{"choice": "<候选 id>"}（标准）、{"choice": "none"}
    （分不清/没有冲突的）、直接写 id 本身（或第几条的数字序号）。
    返回 (id 或 None, reason)：
    - 选中的 id 时 reason 是 None；
    - choice 是 none 时 reason 是 "similar"（两条都像、分不开）或
      "unrelated"（哪个都不像、没关系），取自返回里的 "reason" 字段；
      reason 缺失或认不出时默认 "similar"——重试一次是安全的，
      最坏也只是多调一次然后问 BOOS。
    分不清、格式不对、选的 id 不在候选里时 id 都是 None，调用方
    走问 BOOS 的流程，绝不猜。模型有时会在 JSON 外面包一层
    ```json 围栏，先剥掉再解析。
    """
    if not isinstance(content, str) or not content.strip():
        return None, "similar"
    text = content.strip()
    # 去掉 ```json ... ``` 这种围栏（模型有时会包一层）。
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    ids = {(c or {}).get("id") for c in candidates or []}
    choice = None
    reason = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            choice = data.get("choice")
            reason = data.get("reason")
        elif isinstance(data, (str, int)) and not isinstance(data, bool):
            # 整段就是答案本身（比如直接写了个 id，或写了个数字序号）。
            choice = data
    except ValueError:
        # 不是 JSON：整段就是它写的答案（比如直接写了个 id）。
        choice = text
    if isinstance(reason, str):
        reason = reason.strip().lower()
        if reason not in ("similar", "unrelated"):
            reason = None
    else:
        reason = None
    if isinstance(choice, int) and not isinstance(choice, bool):
        # 数字序号也认（1 → 第 1 条）。
        if 1 <= choice <= len(candidates or []):
            return (candidates[choice - 1] or {}).get("id"), None
        return None, "similar"
    if not isinstance(choice, str):
        return None, "similar"
    choice = choice.strip().strip("\"'")
    if choice.lower() in ("none", "unclear", "不知道", "分不清", ""):
        return None, reason or "similar"
    if choice in ids:
        return choice, None
    if choice.isdigit():
        n = int(choice)
        if 1 <= n <= len(candidates or []):
            return (candidates[n - 1] or {}).get("id"), None
    return None, "similar"


def _cosine_similarity(vec_a, vec_b):
    """两个向量的余弦相似度（纠错安全网用，纯函数）。

    返回 -1 到 1 之间：1 是同一方向、0 是无关、-1 是相反。
    任一向量是空的或零向量时返回 0.0（没法比，就当中立）。
    自己做归一化，不依赖 embedding 模型有没有归一化过。
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in vec_a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


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
      store.delete_memory(mem_id, approve=True)  # 深度删除：先校验归属，再要 BOOS 同意
      store.correct_memory("我喜欢喝咖啡", "我喜欢喝茶")  # 纠错：原地替换旧记忆
      store.deletion_scope(mem_id)  # 删之前展示删除范围，给 BOOS 确认用
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
        # _pending_correction（待确认的纠错）：上一轮纠错分不清改哪条
        # （ambiguous）时留下的，新说法 + 候选列表，等 BOOS 下一句挑
        # 哪条（见 add_turn / _parse_pending_selection）。
        # 只放内存里，重启就没了——对话里的追问本来就是临时的。
        self._pending_correction = None
        # _pending_turn（Plan C 待定区）：上一轮对话先放这里，等下一轮
        # 不撤回才真正存入（见 add_turn / _commit_pending_turn）。
        # 结构 {"messages": [...], "identity": [{"slot","value","text"}]}，
        # identity 是本轮最新一句 user 消息里认出的身份陈述（代码定的，
        # 见 _detect_identity_statement）。只放内存里，重启就没了——
        # 没来得及提交的对话本来就是临时的；沉默（没有下一轮）时，
        # 待定区清空、不存（这是设计，不是丢数据：用户没给下一句，
        # 就当那句还没说定）。
        self._pending_turn = None
        # _committed_user_keys（已提交过的 user 消息指纹）：add_turn
        # 每次收到最近 6 条消息（约 3 轮），提交待定轮时只提没提交过
        # 的 user 消息——指纹是内容去掉首尾空白后的哈希，deque 最多
        # 200 条，旧的自动挤掉（重启后清空，旧轮次不会重复提交，
        # 因为待定区本身也只放内存）。
        self._committed_user_keys = deque(maxlen=200)
        # _recent_context_ids（最近对话上下文）：最近几轮提问时检索到、
        # 拼进提示的记忆 id（deque，最多 CONTEXT_IDS_MAX 条，旧的自动挤掉）。
        # 纠错时拿它当“当前在聊什么”的提示（见 note_context /
        # _judge_correction）。只放内存里，重启就没了——和
        # _pending_correction 一样，对话上下文本来就是临时的。
        self._recent_context_ids = deque(maxlen=CONTEXT_IDS_MAX)
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

        返回：记忆列表，每项为 {"id", "text", "score", "source",
        "metadata", "confirmed"}，source 是 "semantic"（语义）或
        "keyword"（关键词补充），metadata 是规整过的来源信息
        （见 _normalize_metadata），confirmed 是“BOOS 是否逐条确认过”
        （只有 source="user" 的记忆才可能是 True）。
        注意：search 不按来源过滤——提取的（inferred）记忆和未确认的
        旧记忆也会返回，调用方（main.py 的 _memory_block）按来源决定
        怎么用：已确认的才能当作事实，未确认的不能当作事实。
        这样设计的原因：提取的记忆是记忆功能的主力（自动存的都在这），
        过滤掉它们等于记忆功能作废；安全靠“注入时打标签、分层”
        来保证，不靠检索时偷偷丢掉。
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
        4. 敏感过滤（用时拦截）：不管哪路信号拿到的记忆，只要含有敏感
           信息（密码、密钥、身份证号、银行卡号……）就整条跳过、不注入——
           旧库里在过滤加上之前就已存下的敏感记忆，也绝不能拼进发给模型
           的问题里。跳过几条只记个数，不记内容。
        """
        results = []
        seen_texts = set()
        skipped_sensitive = 0
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
                if _contains_sensitive(text):
                    skipped_sensitive += 1
                    continue
                metadata = _normalize_metadata(item.get("metadata"))
                results.append(
                    {
                        "id": item.get("id"),
                        "text": text,
                        "score": score,
                        "source": "semantic",
                        "metadata": metadata,
                        "confirmed": metadata["confirmed"],
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
                        if _contains_sensitive(text):
                            skipped_sensitive += 1
                            continue
                        weight, _matched = _keyword_weight(text, keywords)
                        if weight >= KEYWORD_MIN_WEIGHT:
                            metadata = _normalize_metadata(item.get("metadata"))
                            results.append(
                                {
                                    "id": item.get("id"),
                                    "text": text,
                                    "score": None,
                                    "source": "keyword",
                                    "metadata": metadata,
                                    "confirmed": metadata["confirmed"],
                                }
                            )
                            seen_texts.add(text)
            except Exception:
                logger.warning("mem0 关键词补充检索失败。", exc_info=True)
        if skipped_sensitive:
            logger.info("检索：%d 条记忆含敏感信息，已跳过，不注入。", skipped_sensitive)
        return results

    def note_context(self, memories):
        """记下本轮提问时检索到的记忆 id（最近对话上下文）。

        main.py / web/server.py 提问前 search() 完调一次，传 search
        返回的列表（或 id 列表）。纠错时拿它当“当前在聊什么”的提示：
        候选里有在上下文里的，模型判断时优先考虑它（见 _judge_correction）。
        只放内存里（deque，最多 CONTEXT_IDS_MAX 条，旧的自动挤掉，
        重启就没了——和 _pending_correction 一样，对话上下文本来就是临时的）。
        注意：_find_correction_candidates 内部的 search 不调这里——
        找候选的那次检索不能污染上下文（不然候选自己就成了“上下文”，
        提示就没意义了）。传空列表/None 时什么都不记；记失败
        （比如传了奇怪的东西）也不抛异常、不拦聊天。
        """
        try:
            ids = []
            for item in memories or []:
                if isinstance(item, dict):
                    mem_id = item.get("id")
                else:
                    mem_id = item
                if mem_id:
                    ids.append(mem_id)
            self._recent_context_ids.extend(ids)
        except Exception:
            logger.warning("记对话上下文失败：本次上下文未记录。", exc_info=True)

    def get_all(self, include_sensitive=False):
        """返回当前用户的所有记忆（列表，每项 {"id", "text", "metadata"}，
        metadata 是规整过的来源信息，见 _normalize_metadata）。

        默认不含敏感信息：导出、迁移、展示用的都是这个默认口径，
        旧库里已有的敏感记忆不会从这里漏出去。
        include_sensitive=True 只给内部用：删除归属校验（敏感记忆
        也要能删掉）和重嵌备份（本地维护，不能悄悄丢数据）——
        这两个地方不把记忆发给模型、也不导出，只是在本地处理。
        出问题时返回空列表，不抛异常。
        """
        try:
            raw = self._memory.get_all(filters={"user_id": self.user_id})
            out = []
            for item in (raw or {}).get("results", []):
                text = item.get("memory") or ""
                if not text:
                    continue
                if not include_sensitive and _contains_sensitive(text):
                    continue
                out.append(
                    {
                        "id": item.get("id"),
                        "text": text,
                        "metadata": _normalize_metadata(item.get("metadata")),
                    }
                )
            return out
        except Exception:
            logger.warning("mem0 取全部记忆失败。", exc_info=True)
            return []

    # ---------- 写入（回答后自动存 / 手动存） ----------

    def _mem0_add(self, messages, user_id, infer, metadata):
        """调底层 mem0.add() 存记忆（带来源元数据）。

        mem0 2.2.1 的 add 接受 metadata 参数（试跑已核实）；但万一
        底下的对象是旧版本 mem0、或测试里的子类没写 metadata 参数，
        传 metadata 会报 TypeError（unexpected keyword argument），
        这时退一步：不带 metadata 再调一次（记忆照常存，只是没来源；
        取回时按“未确认”处理，见 _normalize_metadata）。
        注意：只有报错信息里提到 metadata 时才退；别的 TypeError
        （比如底层实现内部出错）不重试——重试可能把一条记忆存两遍。
        """
        try:
            return self._memory.add(
                messages, user_id=user_id, infer=infer, metadata=metadata
            )
        except TypeError as exc:
            if "metadata" not in str(exc):
                raise
            logger.warning("底层 add 不支持 metadata 参数：本次记忆没带来源信息。")
            return self._memory.add(messages, user_id=user_id, infer=infer)

    # ---------- 聊天纠错接线（“你记错了”走 correct_memory） ----------
    # 背景：correct_memory() 本身能改，但聊天里说“你记错了”时没人调它——
    # add_turn 只会把新说法当成一条新记忆存下来，旧的错的还留在库里
    # （2026-11-09 生命周期复测证实的已知缺口：用户说“不对，我喜欢看
    # 纪录片。”，库里最后科幻和纪录片两条都在）。
    # 所以纠错的判断做在 add_turn 里：CLI、网页、直接调 add_turn 的
    # 调用方都走同一条路，不用每个调用方各写一份。流程：
    #   1. 看本轮最后一条 user 消息里有没有明确的纠错说法；
    #   2. 有，就把新说法抠出来，当成 old_text 和 new_text 一起交给
    #      correct_memory(new_fact, new_fact)——它自己会找候选、判断
    #      唯一/多个/没有、原地替换并校验；
    #   3. 按 correct_memory 的 status 决定本轮还做不做正常提取保存。
    # 冲突判断：不另建一套关键词规则引擎（BOOS 的规矩：价值/冲突判断
    # 由模型在提示里做，不硬编码规则）。这里的判断就是两点：
    # 用户亲口说“记错了”——这是他本人断言旧记忆错了，不用模型再猜；
    # 哪条是旧记忆——交给 correct_memory 的候选检索（语义 + 关键词）
    # 和它的唯一/多个/没有逻辑。候选不止一条时不猜，问 BOOS。
    # 检索验证（2026-11-09，离线 FakeMemory 实测）：
    # “我喜欢看纪录片”搜“用户喜欢看科幻电影。”，语义分 0.43（≥ 0.4
    # 阈值）且关键词权重 6（≥ 2），两路都找得到，不用改候选检索、
    # 也不动全局 0.4 阈值（BOOS 定的，留着不动）。

    def _handle_correction_outcome(self, outcome):
        """按 correct_memory 的 status 决定本轮怎么收尾（供 add_turn 用）。

        返回最终要返回给调用方的字典；返回 None 表示“没冲突可改，
        走正常提取保存”（not_found 时，调用方继续走正常保存）。
        规矩：
        - updated：改好了，新说法已经存进去了（同一 id 原地替换），
          不要再走正常提取——提取会把新说法又存一条，变重复。
        - unchanged：新说法库里已经有了，不用再存。
        - ambiguous：分不清改哪条：把这次纠错（新说法 + 候选）留着，
          等 BOOS 下一句挑，不猜、不存重复。
        - failed：改失败了，原记忆没动（correct_memory 已回滚），
          不走提取兜底——兜底会把新说法存成重复记忆；如实报失败。
        - not_found：库里没有冲突的旧记忆，返回 None，让调用方走
          正常提取保存（新说法当成一条新记忆存下来）。
        - 未知 status：保守处理——不猜、不存重复，只把结果报出来。
        """
        status = (outcome or {}).get("status")
        correction = _correction_result(outcome)
        if status == "updated":
            return {
                "correction": correction,
                "results": [
                    {
                        "id": outcome.get("memory_id"),
                        "memory": outcome.get("new_text"),
                        "event": "UPDATE",
                    }
                ],
            }
        if status in ("unchanged", "ambiguous", "failed"):
            if status == "ambiguous":
                # 留着等 BOOS 下一句挑（只放内存里，重启就没了——
                # 对话里的追问本来就是临时的，可以接受）。
                self._pending_correction = {
                    "new_text": outcome.get("new_text"),
                    "candidates": outcome.get("candidates") or [],
                }
            return {"correction": correction}
        if status == "not_found":
            return None
        return {"correction": correction}

    def _extract_and_save_turn(self, safe_messages):
        """正常提取保存（add_turn 的老路子，只做这一件事）。

        调 mem0.add 提取本轮对话、去重、存入；提取出的记忆里万一
        还带着敏感信息，_drop_sensitive_results 会再删掉。
        失败时返回 {}，绝不抛异常——存记忆失败不能把聊天回答搞崩。
        """
        try:
            result = self._mem0_add(
                safe_messages,
                self.user_id,
                True,
                _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False),
            )
        except Exception:
            logger.warning("mem0 自动存记忆失败：本轮对话未存入长期记忆。", exc_info=True)
            return {}
        return self._drop_sensitive_results(result or {})

    # ---------- Plan C：待定区提交 + 同槽位 UPDATE + 身份种子 ----------

    def _commit_pending_turn(self):
        """提交待定区里上一轮对话（Plan C 闸门 3）。

        只在写锁里调（add_turn / flush_pending 都持有锁）。
        顺序：先把待定轮里的身份陈述按槽位 upsert（新值盖旧值），
        再把剩下的、没提交过的 user 句子过闸门 1 过滤、调提取存入。
        提交过的 user 消息指纹记进 _committed_user_keys——add_turn
        每次收到最近 6 条消息（约 3 轮的滑动窗口），下次提交同一批
        消息时，旧轮次的 user 消息不再重复提。
        返回提交结果 {"results": [...], "identity": [...]}，
        没提交任何东西时返回 {}。
        """
        pending = self._pending_turn
        self._pending_turn = None
        if pending is None:
            return {}
        identity_outcomes = []
        for item in pending.get("identity") or []:
            outcome = self._upsert_slot_memory(
                item["slot"], item["value"], item["text"]
            )
            identity_outcomes.append(outcome)
        # 提取：只留没提交过的 user 消息，及其后面跟着的那条
        # assistant 回复（当提取上下文）；指纹先记下来——过滤后
        # 什么都不剩、提取没调时，下次也不再重复处理这句。
        messages = pending.get("messages") or []
        fresh = []
        for i, message in enumerate(messages):
            if (message or {}).get("role") != "user":
                continue
            content = (message or {}).get("content") or ""
            key = content.strip()
            if key in self._committed_user_keys:
                continue
            self._committed_user_keys.append(key)
            fresh.append({"role": "user", "content": content})
            if i + 1 < len(messages) and (messages[i + 1] or {}).get(
                "role"
            ) == "assistant":
                fresh.append(messages[i + 1])
        results = []
        filtered = _filter_extraction_messages(fresh)
        if filtered:
            extracted = self._extract_and_save_turn(filtered)
            results = (extracted or {}).get("results") or []
        if not results and not identity_outcomes:
            return {}
        return {"results": results, "identity": identity_outcomes}

    def flush_pending(self):
        """立刻提交待定区（Plan C）。

        正常对话里待定轮由下一轮 add_turn 提交；这个方法给两处用：
        测试里主动提交、服务停之前把最后一轮落库（否则最后一轮
        永远待在内存里）。返回提交结果，没待定内容时返回 {}。
        """
        with _WRITE_LOCK:
            return self._commit_pending_turn()

    def _find_slot_memory_by_metadata(self, slot):
        """按 metadata.slot 找同槽位记忆（Plan C 冲突兜底的主路）。

        只认 _normalize_metadata 规整过的 slot（KNOWN_SLOTS 里的
        三个）。找不到返回 None。内部用，含敏感也照常找（不导出）。
        """
        for item in self.get_all(include_sensitive=True):
            if (item.get("metadata") or {}).get("slot") == slot:
                return item
        return None

    def _find_slot_memories_by_text(self, slot):
        """按文本找同槽位旧记忆（Plan C 之前存的、没打 slot 标记）：
        文本里认出同一槽位就算——槽位是从文本里认出来的
        （_detect_identity_statement），不是猜的。soul 槽位没有“值”
        的说法：文本里同时有 "SOUL.md" 和“十条原则”时算同一条。
        返回按检索顺序排好的候选列表（可能多条），找不到返回 []。
        """
        queries = {
            SLOT_AGENT_NAME: "Agent 的名字",
            SLOT_USER_NAME: "我叫",
            SLOT_SOUL: "SOUL.md",
        }
        query = queries.get(slot)
        if not query:
            return []
        try:
            candidates = self.search(query, top_k=5) or []
        except Exception:
            logger.warning("找同槽位记忆失败，本次按没有处理。", exc_info=True)
            return []
        found = []
        for cand in candidates:
            text = (cand or {}).get("text") or ""
            if slot == SLOT_SOUL:
                if "SOUL.md" in text and "十条原则" in text:
                    found.append(cand)
                continue
            detected = _detect_identity_statement(text)
            if detected is not None and detected["slot"] == slot:
                found.append(cand)
        return found

    def _find_slot_memory_by_value(self, slot, value):
        """兜底：Plan C 之前存的同槽位记忆没有 slot 标记——按关键词
        找回，文本里认出同一槽位、同一值时才算（只做去重用：值相同
        就不用再存）。

        soul 槽位没有“值”的说法：文本里同时有 "SOUL.md" 和
        “十条原则”时算同一条。找不到返回 None。
        """
        for cand in self._find_slot_memories_by_text(slot):
            if slot == SLOT_SOUL:
                return cand
            detected = _detect_identity_statement((cand or {}).get("text") or "")
            if detected is not None and detected["value"] == value:
                return cand
        return None

    def _find_slot_memory_by_slot_text(self, slot):
        """找同槽位旧记忆，不管值新旧（改名时用）：文本里认出同一槽位
        就算，返回第一条；找不到返回 None。只认第一条——同槽位旧记忆
        有多条时（极少见），只原地更新第一条，其余的要走面板手动清理。
        """
        found = self._find_slot_memories_by_text(slot)
        return found[0] if found else None

    def _upsert_slot_memory(self, slot, value, text):
        """同槽位 UPDATE（Plan C 闸门 3 的冲突兜底）。

        mem0 是 ADD-only，同一槽位的新值必须在应用层盖掉旧值，
        不能并存。只在写锁里调。
        - 库里已有同槽位记忆（metadata.slot 对得上）：值相同→
          不动（unchanged，留着旧文本）；值不同→ 原地 update
          成新文本 + 用户已确认的来源（updated）。SOUL 种子文本
          是固定的：已有 soul 槽位记忆就不动。
        - 没有 slot 标记、但按关键词找回同一槽位同一值的旧记忆：
          也不动（unchanged，只是没标记——Plan C 之前存的）。
        - 没有 slot 标记、但文本认得出是同一槽位、值是旧的：
          原地 update 成新值（updated）——槽位是从文本里认出来的
          （_detect_identity_statement），不是猜的，所以敢更新；
          soul 不走这里（种子文本固定，upsert 本来就不改 soul）。
        - 都没有→ 新增一条（added），来源记 user、已确认、置信度 1.0。
        返回 {"slot", "value", "status", "memory_id"}，
        存失败时 status="failed"。
        """
        if slot not in KNOWN_SLOTS or not value:
            return {
                "slot": slot,
                "value": value,
                "status": "failed",
                "memory_id": None,
            }
        existing = self._find_slot_memory_by_metadata(slot)
        if existing is not None:
            if slot == SLOT_SOUL:
                # SOUL 种子文本是固定的：已有 soul 槽位记忆就不动。
                return {
                    "slot": slot,
                    "value": value,
                    "status": "unchanged",
                    "memory_id": existing["id"],
                }
            old_detected = _detect_identity_statement(existing["text"])
            old_value = old_detected["value"] if old_detected else None
            if old_value == value:
                return {
                    "slot": slot,
                    "value": value,
                    "status": "unchanged",
                    "memory_id": existing["id"],
                }
            try:
                self._memory.update(
                    memory_id=existing["id"],
                    text=text,
                    metadata=_provenance(
                        SOURCE_USER, CONFIDENCE_USER, True, slot=slot
                    ),
                )
            except Exception:
                logger.warning("同槽位记忆更新失败：旧值保留。", exc_info=True)
                return {
                    "slot": slot,
                    "value": value,
                    "status": "failed",
                    "memory_id": existing["id"],
                }
            return {
                "slot": slot,
                "value": value,
                "status": "updated",
                "memory_id": existing["id"],
            }
        duplicate = self._find_slot_memory_by_value(slot, value)
        if duplicate is not None:
            return {
                "slot": slot,
                "value": value,
                "status": "unchanged",
                "memory_id": duplicate["id"],
            }
        # Plan C 之前存的同槽位旧记忆（没打 slot 标记、值是旧的）：
        # 文本里认得出是同一槽位，就原地更新成新值，不并存。
        if slot != SLOT_SOUL:
            legacy = self._find_slot_memory_by_slot_text(slot)
            if legacy is not None:
                try:
                    self._memory.update(
                        memory_id=legacy["id"],
                        text=text,
                        metadata=_provenance(
                            SOURCE_USER, CONFIDENCE_USER, True, slot=slot
                        ),
                    )
                except Exception:
                    logger.warning("同槽位旧记忆更新失败：旧值保留。", exc_info=True)
                    return {
                        "slot": slot,
                        "value": value,
                        "status": "failed",
                        "memory_id": legacy["id"],
                    }
                return {
                    "slot": slot,
                    "value": value,
                    "status": "updated",
                    "memory_id": legacy["id"],
                }
        result = self._add_text_with_metadata(
            text, _provenance(SOURCE_USER, CONFIDENCE_USER, True, slot=slot)
        )
        results = (result or {}).get("results") or []
        if not results:
            return {
                "slot": slot,
                "value": value,
                "status": "failed",
                "memory_id": None,
            }
        return {
            "slot": slot,
            "value": value,
            "status": "added",
            "memory_id": results[0].get("id"),
        }

    def _seed_slot_memory(self, slot, value, text):
        """种子专用：只加没有的，绝不覆盖已有值（和 _upsert_slot_memory
        的区别：upsert 是“新值盖旧值”，种子是“有旧值就不动”——
        BOOS 在聊天里改了名字之后，下次启动再种，不能把名字改回去）。

        已有同槽位记忆（带 slot 标记，或按关键词找回同一槽位同一值）
        时返回 unchanged；都没有时新增一条（added），来源记 user、
        已确认、置信度 1.0。按关键词找回的旧记忆（Plan C 之前存的、
        没打标记）顺手打上槽位标记，文本不动——以后改名走槽位找得到它。
        只在写锁里调。
        """
        if slot not in KNOWN_SLOTS or not value:
            return {
                "slot": slot,
                "value": value,
                "status": "failed",
                "memory_id": None,
            }
        existing = self._find_slot_memory_by_metadata(slot)
        if existing is not None:
            return {
                "slot": slot,
                "value": value,
                "status": "unchanged",
                "memory_id": existing["id"],
            }
        duplicate = self._find_slot_memory_by_value(slot, value)
        if duplicate is not None:
            # Plan C 之前存的同一槽位同一值旧记忆（没打 slot 标记）：
            # 文本不动，只给它打上槽位标记——以后改名走槽位找得到它，
            # 不会并存。打标记失败也不影响种子本身，照常返回 unchanged。
            try:
                self._memory.update(
                    memory_id=duplicate["id"],
                    text=duplicate["text"],
                    metadata=_provenance(
                        SOURCE_USER, CONFIDENCE_USER, True, slot=slot
                    ),
                )
            except Exception:
                logger.warning("给同槽位旧记忆打标记失败：保持原样。", exc_info=True)
            return {
                "slot": slot,
                "value": value,
                "status": "unchanged",
                "memory_id": duplicate["id"],
            }
        result = self._add_text_with_metadata(
            text, _provenance(SOURCE_USER, CONFIDENCE_USER, True, slot=slot)
        )
        results = (result or {}).get("results") or []
        if not results:
            return {
                "slot": slot,
                "value": value,
                "status": "failed",
                "memory_id": None,
            }
        return {
            "slot": slot,
            "value": value,
            "status": "added",
            "memory_id": results[0].get("id"),
        }

    def ensure_identity_memories(self):
        """把十二的名字和 SOUL.md 以“已确认”种进长期记忆（Plan C）。

        只加没有的，绝不覆盖已有的槽位值（见 _seed_slot_memory）。
        网页服务启动时调一次（web/server.py）；CLI 不自动调
        （命令行默认不加载 mem0）。失败时记日志、不抛异常——
        种子没种上不能把服务搞崩。
        返回 {"agent_name": {...}, "soul": {...}}，每项是
        _seed_slot_memory 的返回（added/unchanged/failed）。
        """
        outcomes = {}
        with _WRITE_LOCK:
            try:
                outcomes["agent_name"] = self._seed_slot_memory(
                    SLOT_AGENT_NAME, AGENT_NAME_VALUE, AGENT_NAME_TEXT
                )
            except Exception:
                logger.warning("种 Agent 名字记忆失败。", exc_info=True)
                outcomes["agent_name"] = {
                    "slot": SLOT_AGENT_NAME,
                    "value": AGENT_NAME_VALUE,
                    "status": "failed",
                    "memory_id": None,
                }
            try:
                outcomes["soul"] = self._seed_slot_memory(
                    SLOT_SOUL, "soul", AGENT_SOUL_TEXT
                )
            except Exception:
                logger.warning("种 SOUL 记忆失败。", exc_info=True)
                outcomes["soul"] = {
                    "slot": SLOT_SOUL,
                    "value": "soul",
                    "status": "failed",
                    "memory_id": None,
                }
        return outcomes

    def _pend_correction_remainder(self, safe_messages, latest_user):
        """纠正轮里非纠正部分的句子进待定区（Plan C 2 号）。

        safe_messages 是本轮及最近几轮的完整消息；把其中最后一条
        user 消息换成去掉纠错句后剩下的句子，其余原样保留，按正常
        轮次放进待定区（下一轮提交）。没有剩下的句子时不放。
        身份陈述也按正常轮次认（只在剩下的句子里认）。
        """
        kept = _strip_correction_sentences(latest_user)
        if not kept:
            return
        remainder = "。".join(kept) + "。"
        messages = [dict(m) for m in safe_messages]
        for i in range(len(messages) - 1, -1, -1):
            if (messages[i] or {}).get("role") == "user":
                messages[i] = {"role": "user", "content": remainder}
                break
        else:
            return
        identity_items = []
        for sentence in kept:
            detected = _detect_identity_statement(sentence)
            if detected is not None:
                identity_items.append(detected)
        self._pending_turn = {"messages": messages, "identity": identity_items}

    def add_turn(self, messages):
        """把本轮对话存进长期记忆（回答后调一次）。

        Plan C（2026-11-09）：本轮不立即存，先放进待定区（_pending_turn），
        下一轮 add_turn 时不撤回才真正提交（_commit_pending_turn）；
        沉默（没有下一轮）时待定区清空、不存；“不不不”这类撤回信号
        直接丢掉待定区里的上一轮。提交时身份陈述走槽位 upsert
        （新值盖旧值），其余句子过闸门 1 过滤再提取。纠错轮次不进
        待定区（避免重复存），见下面的接线说明。
        返回里多两个键：正常轮次被待定时 {"pending": True, ...}；
        撤回时 {"retracted": True, "dropped": n, "results": []}；
        其余和以前一样（纠错时带 "correction" 键）。

        参数 messages：本轮及最近几轮的消息列表（user/assistant 交替）。
        mem0 自己调 DeepSeek 提取（自动存，BOOS 已同意不用确认），
        再去重、存进向量库。整个写入走全局写锁，一次只让一个写，
        防止并发写出重复记忆。
        来源：提取出的记忆记 source="inferred"、user_confirmed=False、
        confidence=0.7（见文件头“来源”一节）——用户明确说过的话，
        但 BOOS 没逐条确认过那个写法；存下来了也不会自动变成已确认。
        价值判断在提取提示里（CUSTOM_INSTRUCTIONS 第 9–12 条，DeepSeek
        自己判断）：只存值得长期记的，不值得记的/没确认的直接不存。
        隐私（存前脱敏 + 存后复查，两道）：
        提取要把对话原文发给 DeepSeek，秘密不能跟着原文一起发出去——
        所以先把每条消息里的敏感信息换成占位符，再调提取；脱敏后
        什么有用的内容都不剩（比如整轮只说了一个密码）时，直接不调
        提取，省一次调用。提取回来之后再复查一遍：万一哪条提取出的
        记忆还带着敏感信息（比如模型把没见过的密钥格式复述出来了），
        立刻删掉、不留库里（见 _drop_sensitive_results）。
        聊天纠错（“你记错了”接线，2026-11-09）：
        本轮最后一条 user 消息里有明确纠错说法时（见 _detect_correction），
        不走正常提取——先调 correct_memory(new_fact, new_fact) 改旧记忆，
        按它的 status 收尾（见 _handle_correction_outcome）：
        updated/unchanged/ambiguous/failed 都直接返回纠错结果（不重复存），
        只有 not_found 才走正常提取保存（新说法当新记忆存下来）。
        上一轮 ambiguous 留下的待确认纠错（_pending_correction），
        本轮先看 BOOS 是不是在挑候选（见 _parse_pending_selection），
        挑中了就按他挑的那条改；没挑中就当放弃这次纠错，继续正常处理。
        返回：mem0.add() 的结果（字典），失败时返回 {}，绝不抛异常——
        存记忆失败不能把已经拿到的聊天回答搞崩。
        纠错路径返回的字典里多一个 "correction" 键：
        {"status", "memory_id", "old_text", "new_text", "candidates",
         "reason", "layer"}，调用方（main.py / web/server.py）按它告诉 BOOS
        改好了、还是要他挑一条、还是改失败了。
        """
        if not messages:
            return {}
        with _WRITE_LOCK:
            safe_messages = [
                {
                    "role": (m or {}).get("role"),
                    "content": _redact_sensitive((m or {}).get("content") or ""),
                }
                for m in messages
            ]
            if not any(
                _meaningful_after_redaction(m["content"]) for m in safe_messages
            ):
                return {}
            latest_user = _latest_user_content(safe_messages)
            # 0. Plan C 待定区撤回：本轮最新一句只有撤回信号时，丢掉
            #    待定区里的上一轮（不存、不问），本轮自己也不存。
            #    待确认的纠错也一并清掉（“算了”就是放弃这次纠错）。
            #    注意：这一步在提交待定区之前——撤回就是不要提交。
            #    例外：助手刚问了问题时，光秃秃的否定（“不是。”）更可能
            #    是在回答那个问题，不是撤回上一句——这时不丢待定区，
            #    正常提交上一轮、本轮照常待定。
            if latest_user and _is_bare_retraction(latest_user):
                pending = self._pending_turn
                assistant_asked = False
                if pending is not None:
                    for m in reversed(pending.get("messages") or []):
                        if (m or {}).get("role") == "assistant":
                            content = (m or {}).get("content") or ""
                            assistant_asked = "？" in content or "?" in content
                            break
                if pending is not None and not assistant_asked:
                    self._pending_turn = None
                    self._pending_correction = None
                    dropped = sum(
                        1
                        for m in pending.get("messages") or []
                        if (m or {}).get("role") == "user"
                    )
                    return {"retracted": True, "dropped": dropped, "results": []}
                # 否则：当作普通轮次继续（下面先提交待定区、再待定本轮）。
            # 0b. 提交上一轮待定区（在纠错判断之前——刚说的事实先落库，
            #     再处理本轮的纠错；纠错改的才是落库后的那条）。
            committed = self._commit_pending_turn()
            # 1. 先处理上一轮留下的待确认纠错：BOOS 这句是不是在挑候选？
            if self._pending_correction is not None and latest_user:
                pending = self._pending_correction
                # 无论成不成，这轮都先清掉——不留着反复问。
                self._pending_correction = None
                selected = _parse_pending_selection(
                    latest_user, pending.get("candidates") or []
                )
                if selected is not None:
                    # 按他挑的那条改：old_text 传挑中的原文，
                    # correct_memory 里原文对上了就不用猜。
                    outcome = self.correct_memory(
                        selected.get("text") or "", pending.get("new_text") or ""
                    )
                    handled = self._handle_correction_outcome(outcome)
                    if handled is not None:
                        return handled
                    # not_found：挑中的那条已经不在库里了（被删了），
                    # 新说法还是他确认过的，直接当新记忆存下来。
                    saved = self.add_memory_text(
                        pending.get("new_text") or "", source=SOURCE_USER
                    )
                    if isinstance(saved, dict):
                        saved = dict(saved)
                        saved["correction"] = _correction_result(outcome)
                    return saved
                # 没挑中（说了别的、或说“都不是”）：当作放弃这次纠错，
                # 继续按本轮消息正常处理（下面可能又是一次新的纠错）。
            # 2. 本轮是不是纠错（“你记错了……”）？
            detected = _detect_correction(latest_user) if latest_user else None
            if detected is not None:
                outcome = self.correct_memory(
                    detected["fact"], detected["fact"]
                )
                handled = self._handle_correction_outcome(outcome)
                if handled is not None:
                    # 2 号：纠正那一轮里别的句子别丢——去掉纠错句，
                    # 剩下的按正常轮次进待定区，下一轮正常存。
                    self._pend_correction_remainder(safe_messages, latest_user)
                    return handled
                # not_found：库里没有冲突的旧记忆——新说法当成一条新记忆，
                # 走正常提取保存（下面），结果里附上这次纠错的尝试记录。
                normal = self._extract_and_save_turn(safe_messages)
                if isinstance(normal, dict):
                    normal = dict(normal)
                    normal["correction"] = _correction_result(outcome)
                return normal
            # 3. 正常轮次：不立即存，放进待定区等下一轮。
            #    身份陈述（名字/称呼）只在最新一句里认——旧轮次的
            #    身份在它那轮已经认过、提交时已 upsert，不重复认。
            identity_items = []
            if latest_user:
                for sentence in re.split(r"[。！？\n]+", latest_user):
                    detected = _detect_identity_statement(sentence.strip())
                    if detected is not None:
                        identity_items.append(detected)
            self._pending_turn = {
                "messages": safe_messages,
                "identity": identity_items,
            }
            return {
                "pending": True,
                "results": (committed or {}).get("results") or [],
                "identity": [item["slot"] for item in identity_items],
            }

    def _drop_sensitive_results(self, result):
        """提取后二次过滤：检查本次提取出的记忆，万一哪条还带着敏感信息
        （比如模型把没见过的密钥格式复述出来了），立刻删掉、不留库里，
        也不返回给调用方。删失败时只记日志（不记内容），不影响其他记忆。
        返回过滤后的结果（删掉的条目不在 results 里）。"""
        items = (result or {}).get("results") or []
        if not items:
            return result
        kept = []
        dropped = 0
        for item in items:
            text = (item or {}).get("memory") or ""
            mem_id = (item or {}).get("id")
            if text and _contains_sensitive(text):
                dropped += 1
                if mem_id:
                    try:
                        self._memory.delete(mem_id)
                    except Exception:
                        logger.warning(
                            "提取后过滤：删除含敏感信息的记忆失败。", exc_info=True
                        )
                continue
            kept.append(item)
        if dropped:
            logger.info("提取后过滤：%d 条提取出的记忆含敏感信息，已删除。", dropped)
        if len(kept) == len(items):
            return result
        filtered = dict(result)
        filtered["results"] = kept
        return filtered

    def _add_text_with_metadata(self, text, metadata):
        """原文存一条记忆（infer=False），带上来源元数据。

        只做“调 mem0.add”这一件事：脱敏、去重由调用方先做好。
        返回前过一遍提取后二次过滤（_drop_sensitive_results）：
        底层 add 的返回里万一混进一条含敏感信息的记忆（比如
        实现 bug 多存了一条），立刻删掉、不留库里——Plan C 的
        槽位 upsert 也走这里，和提取路径同一道防线。
        mem0 报错时记日志、返回 {}，不抛异常（存记忆失败不能把
        聊天搞崩，见 add_turn 的约定）。
        """
        try:
            result = self._mem0_add(
                [{"role": "user", "content": text}],
                self.user_id,
                False,
                metadata,
            )
            return self._drop_sensitive_results(result or {})
        except Exception:
            logger.warning("mem0 手动存记忆失败。", exc_info=True)
            return {}

    def add_memory_text(self, text, layers=None, source="user"):
        """手动存一条记忆原文（CLI“记住：内容”用，不经过提取）。

        参数 text：记忆内容原文；layers：保留给以后用；
        source：这条记忆的来源，默认 "user"——CLI“记住”是 BOOS
        亲口让记住的，所以记 source="user"、user_confirmed=True、
        confidence=1.0，并把这份 metadata 传给 mem0.add（以前这里
        接受了 source 参数却没传给 mem0，是个 bug：库里存的记忆
        metadata 是 None，查不到来源——2026-11-09 修好）。
        不是 "user" 的 source 一律按 "inferred" 存（见 _provenance）。
        走全局写锁。
        隐私（存前脱敏）：直接存的原文不经过提取，没有模型把关，
        秘密会原样进库——所以存之前先把秘密换成占位符；脱敏后
        什么有用的内容都不剩（整条就是一个密码）时，直接不存。
        混合内容（“喜欢喝咖啡，密码是 xxx”）只脱敏秘密部分，
        有用的那句照常存下来，不整条丢掉。
        去重：库里已有相同（或只差标点/空白/大小写）的记忆时不重复存，
        返回已有的那条（event=NONE，和 mem0“没变化”的意思一致，
        顶层再带一个 duplicate=True，方便迁移脚本数数）。
        返回：mem0.add() 的结果，失败返回 {}，不抛异常。
        """
        if not isinstance(text, str) or not text.strip():
            return {}
        with _WRITE_LOCK:
            safe_text = _redact_sensitive(text.strip())
            if not _meaningful_after_redaction(safe_text):
                return {}
            duplicate = _find_duplicate_memory(safe_text, self.get_all())
            if duplicate is not None:
                logger.info("手动存记忆：库里已有相同记忆，本次未重复存。")
                return {
                    "results": [
                        {
                            "id": duplicate["id"],
                            "memory": duplicate["text"],
                            "event": "NONE",
                        }
                    ],
                    "duplicate": True,
                }
            metadata = _provenance(
                source,
                CONFIDENCE_USER if source == SOURCE_USER else CONFIDENCE_INFERRED,
                source == SOURCE_USER,
            )
            return self._add_text_with_metadata(safe_text, metadata)

    # ---------- 迁移与种子 ----------

    def migrate_from_v1(self, memories_json_path=None):
        """把旧 memories.json 里的记忆一次性迁进 mem0（原文迁移，不重新提取）。

        参数 memories_json_path：旧文件路径，不传时用项目 memory/memories.json。
        每条用 add(infer=False) 原文存入，旧记忆默认全部有效。
        来源：v1 时代是“先确认再存”（confirm-first），所以迁过来的旧记忆
        都记 source="user"、user_confirmed=True、confidence=1.0，并把
        旧条目里的 date（有的话）当作形成时间；没有 date 的用今天。
        例外：旧的第 3 条预算记忆写的是已作废的旧规矩
        （“每月预算 5 美元，4 美元提醒，5 美元停。”），迁移时改成新规矩
        （FIRST_MEMORIES 里的第 3 条），并记一条日志说明改过；
        其他记忆一律一字不改。
        已在库里的（按原文完全相同算）跳过，不重复存。
        隐私（导出过滤）：旧文件里的记忆先脱敏再迁——混合内容只脱敏
        秘密部分、有用的那句照常迁；整条只有敏感信息（比如整条就是
        一个密码）时跳过不迁。日志里不出现记忆原文。
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
        existing_normalized = {_normalize_memory_text(t) for t in existing_texts}
        added = 0
        skipped_sensitive = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            # 已作废的旧预算规则：迁移时改成新规矩，并记日志。
            # 日志里不写原文（这句是固定文本，但规矩是“日志不记记忆内容”，
            # 以后万一有人把别的记忆也记进日志，至少这一处是干净的）。
            if text == _OUTDATED_BUDGET_TEXT:
                text = FIRST_MEMORIES[2]["text"]
                logger.info("迁移：旧预算记忆已按新规矩改写。")
            # 先脱敏再迁：混合内容只脱敏秘密部分，整条只有秘密时跳过。
            text = _redact_sensitive(text)
            if not _meaningful_after_redaction(text):
                skipped_sensitive += 1
                continue
            if _normalize_memory_text(text) in existing_normalized:
                continue
            # v1 记忆是 BOOS 确认过才存的，来源记 user；旧条目有 date
            # 就用它当形成时间（add_memory_text 自己会拼 metadata，
            # 但形成时间它不知道，这里直接传一份拼好的 metadata——
            # 所以这里不走 add_memory_text，改走 _add_text_with_metadata。
            # 注意：直接调 mem0.add 之前，脱敏/去重上面已经做过了。）
            created_at = item.get("date") if isinstance(item.get("date"), str) else None
            result = self._add_text_with_metadata(
                text,
                _provenance(SOURCE_USER, CONFIDENCE_USER, True, created_at=created_at),
            )
            if result and not result.get("duplicate"):
                added += 1
                existing_texts.add(text)
                existing_normalized.add(_normalize_memory_text(text))
        logger.info(
            "迁移完成：从 %s 新存入 %d 条记忆（%d 条只有敏感信息，已跳过）。",
            path,
            added,
            skipped_sensitive,
        )
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
            result = self.add_memory_text(
                item["text"], source=item.get("source", SOURCE_USER)
            )
            if result:
                added += 1
        if added:
            logger.info("已记录 %d 条初始记忆。", added)
        return added

    # ---------- 纠错（改）与深度删除（忘） ----------
    # 改和忘是两件不同的事，别搞混：
    #   - 改（correct_memory）：旧内容错了，换成新的；同一 id 原地替换，
    #     history.db 里留一条 UPDATE 审计（旧文本还在审计行里，那是
    #     “改过什么”的记录，有意保留）；改完旧文本检索不到、新文本能检索到。
    #   - 忘（delete_memory）：这条记忆不要了，向量删掉，history.db 里
    #     这条记忆的全部审计行也清掉，验三处都干净才算删完。
    # 两者共用一道规矩：绝不谎称成功——改失败原记忆不动、报失败；
    # 删失败（或删了一半）报清楚哪一步没成，日志/返回值里不写“已删除”。

    def _find_correction_candidates(self, old_text, top_k=5):
        """找“纠错要改的那条记忆”的候选（供 correct_memory 用）。

        两路信号，和 search 一样：
        1. 语义检索 old_text（过阈值的才算）；
        2. 关键词：库里所有记忆里，和 old_text 有 ≥2 权重关键词重合的。
        两路合并、按 id 去重，返回 [{"id", "text", "score", "metadata",
        "confirmed"}]，按分数从高到低排（关键词捞回的分数记 None，
        排在有分数的后面）。调用方按这个结果判断“唯一/多个/没有”。
        出错时返回 []（找不到就报找不到，不猜）。
        """
        candidates = []
        seen_ids = set()
        try:
            for item in self.search(old_text, top_k=top_k):
                mem_id = item.get("id")
                if mem_id and mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    candidates.append(item)
        except Exception:
            logger.warning("纠错：语义找候选失败，只用关键词找。", exc_info=True)
        try:
            keywords = _extract_keywords(old_text)
            if keywords:
                for item in self.get_all(include_sensitive=True):
                    mem_id = item.get("id")
                    if not mem_id or mem_id in seen_ids:
                        continue
                    weight, _matched = _keyword_weight(item.get("text") or "", keywords)
                    if weight >= KEYWORD_MIN_WEIGHT:
                        seen_ids.add(mem_id)
                        candidates.append(
                            {
                                "id": mem_id,
                                "text": item.get("text") or "",
                                "score": None,
                                "source": "keyword",
                                "metadata": item.get("metadata")
                                or _normalize_metadata(None),
                                "confirmed": (item.get("metadata") or {}).get(
                                    "confirmed", False
                                ),
                            }
                        )
        except Exception:
            logger.warning("纠错：关键词找候选失败。", exc_info=True)
        return candidates

    def _judge_correction(self, new_text, candidates):
        """请 DeepSeek 判断：新说法到底在纠正候选里的哪一条。

        什么时候调：correct_memory 里候选 ≥2 条、又没有原文完全相同的
        时候——光靠相似度分数分不开（比如“都喜欢…”的句式互相都像），
        请模型按意思判断。第 1 层直接定已经去掉（2026-11-09 新方案：
        上下文里恰好只有 1 条也不直接定，C3 证明那样会静默改错），
        这里是 ≥2 候选时唯一的定夺路径。模型是最终裁判：
        - 判断按“槽位”做：每条旧记忆占一个语义槽位（比如“电影喜好”
          是一个槽位、“编程语言喜好”是另一个槽位）；新说法是在替换
          哪个槽位里的内容，就选那条。必须在同一个槽位上才构成替换，
          不在同一个槽位的候选即使最近聊过、分数再高也不选。
          还有一条：和新说法意思几乎一样的那条不用改（它说的已经
          是新说法的意思了），别选它；要选的是被新说法取代的那条
          ——和新说法意思不同、但在同一个槽位上的那条（2026-11-09
          v7 复测发现：不加这句，模型会把和新说法最像的那条（比如
          已有“用户喜欢看恐怖电影。”时说“我喜欢看恐怖电影。”）
          当成要改的那条，其实该改的是同一槽位里意思不同的那条）。
        - 每条候选都带上文本、分数、和“在不在最近对话上下文里”
          （note_context 记的，本轮刚提到过的标出来）当提示；
        - 上下文是决胜依据：按槽位分不开、两条都像时，看刚聊的是
          哪一个，刚聊的那个又是同一槽位，就选它；只有刚聊的也分
          不出来，才说 none。但别硬套：如果新说法明明在说另一件
          旧事，不要因为它最近没提到就硬选刚提到的那条。
        - 重试：第一次被截断（finish_reason == "length"）或回 none
          且原因是 "similar"（分不清）时，用 2000 同一个提示再试
          一次；回 none 且原因是 "unrelated"（不知道）时不重试；
          调用异常也不重试；重试还是不行就回 (None, reason)。最多调两次。
        - 不加任何关键词/启发式打分层：价值判断由 DeepSeek 自己做
          （BOOS 的规矩），这里只把候选原样递给模型。
        返回 (挑中的候选原字典, None)，或 (None, reason)：
        reason 是 "similar"（分不清）或 "unrelated"（不知道），
        没做判断时（候选里有敏感内容、调用失败、重试也被截断、
        模型选的 id 不在候选里）reason 是 None。调用方拿到 None
        就走问 BOOS 的流程，绝不猜、绝不谎称改好了。
        调用：走 mem0 的 LLM 客户端（self._memory.llm.client，和提取
        同一路，模型 deepseek-flash，温度 0，输出只够回一个 JSON）；
        用量包装（_attach_usage_wrapper）建库时已经包住了它的 create，
        所以这次调用的 Token 也会记进 BudgetTracker；没包住
        （假对象、没传 tracker）时至少记一条日志，不崩。
        日志里只记条数和结果（挑中/没挑中），不记候选原文——
        日志不记记忆内容（见文件头“安全与隐私”）。
        """
        if not candidates:
            return None, None
        # 候选里有敏感内容时不判断：敏感文本不能发给模型做判断，
        # 直接回 None，让 BOOS 自己挑（问 BOOS 的流程不经过模型）。
        for cand in candidates:
            if _contains_sensitive((cand or {}).get("text") or ""):
                logger.info("纠错判断：候选里有敏感内容，本次不请模型判断，改问 BOOS。")
                return None, None
        context_ids = set(self._recent_context_ids)
        lines = []
        for i, cand in enumerate(candidates, start=1):
            cand = cand or {}
            in_context = cand.get("id") in context_ids
            hint = "本轮对话刚提到过" if in_context else "最近对话没提到"
            score = cand.get("score")
            score_text = f"，相似度 {score:.3f}" if isinstance(score, (int, float)) else ""
            lines.append(
                f"{i}. id={cand.get('id')}（{hint}{score_text}）：{cand.get('text') or ''}"
            )
        prompt = (
            "用户纠正了一条旧记忆，请判断新说法到底在纠正下面哪一条旧记忆。\n"
            "按“槽位”判断：每条旧记忆都占一个语义槽位（比如“电影喜好”\n"
            "是一个槽位、“编程语言喜好”是另一个槽位）；新说法是在替换\n"
            "哪个槽位里的内容，就选那条——新说法和旧记忆必须在同一个\n"
            "槽位上才构成替换，不在同一个槽位的候选即使最近聊过、分\n"
            "数再高也不选。\n"
            "注意：和新说法意思几乎一样的那条不用改（它说的已经是新说\n"
            "法的意思了），别选它；要选的是被新说法取代的那条——和新\n"
            "说法意思不同、但在同一个槽位上的那条。\n"
            "如果按槽位分不开、两条以上都像：看‘本轮对话刚提到过’——\n"
            "刚聊的那条如果和新说法是同一话题，就选刚聊的那条；上下\n"
            "文就是这时候的决胜依据。只有刚聊的也分不出来（比如刚聊\n"
            "的和新说法不是同一话题、或者两条刚聊的都像），才选 none。\n"
            "但别硬套上下文：如果新说法明明在说另一件旧事，不要因为\n"
            "它最近没提到就硬选刚提到的那条。\n"
            "选 none 时要说明是哪种：两条都像、分不开是 \"similar\"；\n"
            "哪个都不像、没关系是 \"unrelated\"。只返回一个 JSON，不要解释，\n"
            "格式：\n"
            '{"choice": "<候选 id>" 或 "none", "reason": "similar" 或 "unrelated"（只有 choice 是 none 时才写）}\n'
            f"新说法：{new_text}\n"
            "候选：\n" + "\n".join(lines)
        )
        # 最多调两次：第一次 JUDGE_MAX_TOKENS（1000），思考保持开启，
        # temperature=0；第一次被截断（finish_reason == "length"）或
        # 回 none 且原因是 "similar"（分不清）时，用
        # JUDGE_RETRY_MAX_TOKENS（2000）同一个提示再试一次。第一次
        # 回 none 且原因是 "unrelated"（不知道）时不重试；调用异常
        # 也不重试。重试还是不行（截断或 none）→ 返回 None，走问 BOOS。
        max_tokens = JUDGE_MAX_TOKENS
        for attempt in (1, 2):
            try:
                client = self._memory.llm.client
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "你是记忆纠错的判断助手，只返回 JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                content = (resp.choices[0].message.content or "").strip()
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
            except Exception:
                logger.warning("纠错判断：调 DeepSeek 失败，改走问 BOOS 的流程。", exc_info=True)
                return None, None
            if finish_reason == "length":
                if attempt == 1:
                    logger.info("纠错判断：第一次调用被截断，用 2000 再试一次。")
                    max_tokens = JUDGE_RETRY_MAX_TOKENS
                    continue
                logger.info("纠错判断：重试也被截断，改问 BOOS。")
                return None, None
            choice, reason = _parse_judge_choice(content, candidates)
            if choice is not None:
                for cand in candidates:
                    if (cand or {}).get("id") == choice:
                        logger.info("纠错判断：模型挑中一条，继续按它改。")
                        return cand, None
                return None, None
            if reason == "unrelated":
                logger.info("纠错判断：模型说哪个都不像，改问 BOOS（不重试）。")
                return None, "unrelated"
            # reason == "similar"（分不清）：第一次就用 2000 再试一次。
            if attempt == 1:
                logger.info("纠错判断：模型分不清（两条都像），用 2000 再试一次。")
                max_tokens = JUDGE_RETRY_MAX_TOKENS
                continue
            logger.info("纠错判断：重试还是分不清，改问 BOOS。")
            return None, "similar"
        return None, None

    def _embed_text(self, text):
        """用本地 embedding 模型把一段文本嵌成向量（纠错安全网用）。

        走 mem0 的本地 embedding 模型（self._memory.embedding_model，
        bge-small-zh-v1.5，不联网、不花钱），和检索是同一路。返回
        list[float]；模型不可用（比如离线测试的假对象没装 embedding
        模型）或出错时返回 None——调用方跳过话题校验、不拦纠正：
        安全网坏了不该把纠错也堵死，主防线是模型判本身。
        """
        try:
            embedder = getattr(self._memory, "embedding_model", None)
            if embedder is None:
                return None
            vec = embedder.embed(text, "search")
            if not isinstance(vec, (list, tuple)) or not vec:
                return None
            return [float(x) for x in vec]
        except Exception:
            logger.warning("纠错安全网：本地嵌文本失败，跳过话题校验。", exc_info=True)
            return None

    def _old_new_similarity(self, old_text, new_text):
        """旧记忆文本 vs 新说法的余弦相似度（纠错安全网用）。

        返回 -1 到 1 之间的 float；嵌不出来（模型不可用/出错）时
        返回 None，调用方跳过校验。相似度高说明新说法和旧记忆是
        同一话题（一次正常的纠正本来就该在同一个槽位里换），低
        说明模型可能挑错了槽位（比如把 Go 写代码改成纪录片）。
        """
        vec_old = self._embed_text(old_text)
        vec_new = self._embed_text(new_text)
        if vec_old is None or vec_new is None:
            return None
        return _cosine_similarity(vec_old, vec_new)

    def correct_memory(self, old_text, new_text):
        """纠错：把一条记错的记忆换成正确的（原地替换，同一 id）。

        什么时候调：BOOS 说“你记错了”并给出正确说法时（或以后网页端/
        命令行的纠错入口调）。old_text 是“错的那句大概是什么”
        （调用方从刚检索到的那条记忆里取），new_text 是正确的说法。
        流程：
        1. 先找候选（_find_correction_candidates）。
           - 候选里有和 old_text 规整后完全相同的，先认它（哪怕还有
             别的相似记忆——原文对上了就不用猜）。
           - 恰好 1 个候选：改它。
           - 0 个候选：status="not_found"，什么都不改（调用方可以问
             BOOS、或改走“记住”新存一条，绝不凭空造一条“纠正”）。
           - ≥2 个候选、又没有原文完全相同的：请模型按槽位判（见
             _judge_correction，模型按意思 + 槽位挑，上下文是意思
             分不开时的决胜依据）；分不清/调用失败时
             status="ambiguous"，把候选列表原样返回，让调用方问
             BOOS 到底是哪条——绝不猜一条改掉。
           - 第 1 层直接定已经去掉（2026-11-09 新方案，BOOS 定的）：
             上下文里恰好只有 1 条也不直接定——C3 证明那样会静默
             改错（上下文只有 Go、要改的是科幻，直接定就把 Go 改
             成了纪录片，科幻没动还报成功）。慢一点没关系，安全优先。
        2. 新文本和旧文本规整后相同：status="unchanged"，不用改。
        3. 脱敏新文本：新文本里带秘密时只存脱敏后的版本；脱敏后
           什么都不剩时 status="failed"（不存一条空记忆）。
        4. 安全网（拦模型判和单候选挑中的目标）：真正替换之前，用本地
           bge 向量算旧记忆和新说法的相似度
           （_old_new_similarity）；低于
           CORRECTION_TOPIC_MIN_SIMILARITY 说明新说法和这条旧记忆
           不是同一话题——可能挑错了槽位，不换，改问 BOOS
           （status="ambiguous"、layer="judge_blocked"）。嵌不出来
           时不拦（安全网坏了不该堵死纠正，主防线是模型判本身）。
           exact 不走这道网：原文对上了没歧义。single 也走这道网——
           单候选也可能是错的（同话题长得像的另一条），直接改会
           静默改错，和 C3 一个毛病。
        5. 用 mem0.update() 原地替换（同一 id），新记忆来源记 user、
           user_confirmed=True（BOOS 亲口纠正的）、confidence=1.0。
           mem0.update 不支持 metadata 参数的老版本会报 TypeError，
           这时退一步：只传 text 再改（来源还是旧的，记一条警告；
           FakeMemory 和 mem0 2.2.1 都支持 metadata，不会走到这步）。
        6. 验：重新读库——id 还在、新文本对得上、搜 old_text 找不回
           旧文本，三条都过才算改好。验不过时，把旧文本和旧 metadata
           写回去（回滚），status="failed"，并在结果里写清哪一步没过。
        返回字典（不抛异常，调用方按 status 处理）：
          {"status": "updated" | "ambiguous" | "not_found" |
                     "unchanged" | "failed",
           "memory_id": id 或 None,
           "old_text": 旧文本或 None, "new_text": 新文本或 None,
           "candidates": 候选列表（ambiguous / not_found 时给调用方看）,
           "reason": 失败/未改的原因（给人看的，不含敏感内容）,
           "layer": 哪一层定的——"exact"（原文完全相同）、"single"
             （单候选）、"judge"（模型判中并通过了安全网）、
             "judge_blocked"（模型判挑中了、但安全网拦下来了，
             ambiguous 时用这个）、"ask"（模型分不清、走问 BOOS，
             ambiguous 时用这个）、"none"（not_found 时用这个）；
             failed/unchanged 时填实际选了目标的那一层，没有目标
             就填 "none"。
           "resolution": 这次纠正的结果——"updated"（换好了）、
             "judge_blocked"（安全网拦下、问 BOOS）、"ask"（分不清、
             问 BOOS）、"not_found"、"unchanged"、"failed"。
             layer 说“靠什么定的”，resolution 说“最后怎么样了”。
           "audit": 审计信息（每次纠正都有，改错了靠它查、靠它回滚）——
             {"picked_id": 挑中的那条 id、"judge_reason": 模型说
              none 时的原因（"similar"/"unrelated"，挑中时是 None）、
              "old_new_similarity": 安全网算的旧-新相似度（没算时
              是 None）、"candidate_scores": 各候选的检索分数
              （关键词捞回的是 None）、"context_ids": 当时最近对话
              上下文里的记忆 id}。}
        注意：改不经过提取（新文本是 BOOS 给的原话，不用模型再判断
        一遍）；但多个候选分不清时，会请 DeepSeek 判断改哪条
        （只判断、不改写新文本，见 _judge_correction）。改只换当前文本，
        history.db 的 UPDATE 审计行会留着旧文本——那是审计记录，
        不是“当前事实”，和“忘”要求的清历史是两回事（见 delete_memory）。
        """
        def _audit(picked_id=None, judge_reason=None, similarity=None, cands=None):
            """拼一份审计信息（每次纠正的返回里都带，见上面 docstring）。
            cands 不传时用本次找出的候选（早于找候选的返回用空列表）。"""
            return {
                "picked_id": picked_id,
                "judge_reason": judge_reason,
                "old_new_similarity": similarity,
                "candidate_scores": {
                    (c or {}).get("id"): (c or {}).get("score")
                    for c in (cands if cands is not None else [])
                },
                "context_ids": list(self._recent_context_ids),
            }
        if not isinstance(old_text, str) or not old_text.strip():
            return {
                "status": "not_found",
                "memory_id": None,
                "old_text": None,
                "new_text": new_text,
                "candidates": [],
                "reason": "没给旧内容，不知道改哪条。",
                "layer": "none",
                "resolution": "not_found",
                "audit": _audit(),
            }
        if not isinstance(new_text, str) or not new_text.strip():
            return {
                "status": "failed",
                "memory_id": None,
                "old_text": old_text,
                "new_text": None,
                "candidates": [],
                "reason": "没给新内容，没法改。",
                "layer": "none",
                "resolution": "failed",
                "audit": _audit(),
            }
        with _WRITE_LOCK:
            candidates = self._find_correction_candidates(old_text)
            target = None
            layer = "none"
            judge_reason = None
            similarity = None
            normalized_old = _normalize_memory_text(old_text)
            for item in candidates:
                if _normalize_memory_text(item.get("text") or "") == normalized_old:
                    target = item
                    layer = "exact"
                    break
            if target is None:
                if len(candidates) == 1:
                    target = candidates[0]
                    layer = "single"
                elif len(candidates) == 0:
                    return {
                        "status": "not_found",
                        "memory_id": None,
                        "old_text": old_text,
                        "new_text": new_text,
                        "candidates": [],
                        "reason": "库里没有和旧内容对得上的记忆，没改任何东西。",
                        "layer": "none",
                        "resolution": "not_found",
                        "audit": _audit(cands=candidates),
                    }
                else:
                    # ≥2 个候选、又没有原文完全相同的：请模型按槽位判
                    # （见 _judge_correction）——模型按意思 + 槽位挑，
                    # 上下文是意思分不开时的决胜依据；分不清/调用失败
                    # 时走问 BOOS。第 1 层直接定已经去掉：上下文里恰好
                    # 只有 1 条也不直接定（C3 教训：上下文只有 Go、要
                    # 改的是科幻，直接定会静默改错 Go）。慢一点没关系。
                    judged, judge_reason = self._judge_correction(new_text, candidates)
                    if judged is not None:
                        target = judged
                        layer = "judge"
                    else:
                        return {
                            "status": "ambiguous",
                            "memory_id": None,
                            "old_text": old_text,
                            "new_text": new_text,
                            "candidates": candidates,
                            "reason": (
                                f"有 {len(candidates)} 条相似记忆，分不清改哪条，"
                                "请 BOOS 指明是哪条再改。"
                            ),
                            "layer": "ask",
                            "resolution": "ask",
                            "audit": _audit(judge_reason=judge_reason, cands=candidates),
                        }
            mem_id = target["id"]
            old_stored_text = target["text"]
            old_metadata = target.get("metadata") or _normalize_metadata(None)
            if _normalize_memory_text(old_stored_text) == _normalize_memory_text(new_text):
                return {
                    "status": "unchanged",
                    "memory_id": mem_id,
                    "old_text": old_stored_text,
                    "new_text": new_text,
                    "candidates": [],
                    "reason": "旧内容和新内容一样，不用改。",
                    "layer": layer,
                    "resolution": "unchanged",
                    "audit": _audit(picked_id=mem_id, judge_reason=judge_reason, cands=candidates),
                }
            # 脱敏新文本（纠正的话是 BOOS 亲口说的，但万一里面带了秘密，
            # 还是只存脱敏后的版本；脱敏后什么都不剩就不存）。
            safe_new = _redact_sensitive(new_text.strip())
            if not _meaningful_after_redaction(safe_new):
                return {
                    "status": "failed",
                    "memory_id": mem_id,
                    "old_text": old_stored_text,
                    "new_text": new_text,
                    "candidates": [],
                    "reason": "新内容脱敏后什么都不剩，没改。",
                    "layer": layer,
                    "resolution": "failed",
                    "audit": _audit(picked_id=mem_id, judge_reason=judge_reason, cands=candidates),
                }
            # 安全网（拦模型判和单候选挑中的目标）：真正替换之前，用本地
            # bge 向量算旧记忆和新说法的相似度。低于
            # CORRECTION_TOPIC_MIN_SIMILARITY 说明新说法和这条旧记忆
            # 不是同一话题——可能挑错了槽位（C3 就是这样：把 Go
            # 写代码改成纪录片），不换，改问 BOOS。嵌不出来
            # （similarity 是 None）时不拦：安全网坏了不该堵死纠正，
            # 主防线是模型判本身。exact 不走这道网（原文对上了没歧义）；
            # single 也走这道网——单候选也可能是错的。
            if layer in ("judge", "single"):
                similarity = self._old_new_similarity(old_stored_text, safe_new)
                if similarity is not None and similarity < CORRECTION_TOPIC_MIN_SIMILARITY:
                    logger.info(
                        "纠错安全网：挑中的旧记忆和新说法话题不像"
                        "（相似度 %.3f，低于 %.2f），不换，改问 BOOS。",
                        similarity,
                        CORRECTION_TOPIC_MIN_SIMILARITY,
                    )
                    return {
                        "status": "ambiguous",
                        "memory_id": None,
                        "old_text": old_stored_text,
                        "new_text": safe_new,
                        "candidates": candidates,
                        "reason": (
                            "挑中的那条记忆和新说法不是同一话题，"
                            "可能是挑错了，请 BOOS 指明改哪条再改。"
                        ),
                        "layer": "judge_blocked",
                        "resolution": "judge_blocked",
                        "audit": _audit(
                            picked_id=mem_id,
                            judge_reason=judge_reason,
                            similarity=similarity,
                            cands=candidates,
                        ),
                    }
            # Plan C：纠错换文本时保留旧记忆的槽位（slot）——名字/称呼
            # 记忆被“不对，你叫十二”改过之后，槽位标记不能丢，不然
            # 下次改名时按槽位找不到旧记忆，就会并存两条。
            new_metadata = _provenance(
                SOURCE_USER,
                CONFIDENCE_USER,
                True,
                slot=old_metadata.get("slot"),
            )
            try:
                try:
                    self._memory.update(mem_id, text=safe_new, metadata=new_metadata)
                except TypeError:
                    logger.warning(
                        "纠错：底层 update 不支持 metadata 参数，只改文本。"
                    )
                    self._memory.update(mem_id, text=safe_new)
            except Exception as exc:
                logger.warning("纠错失败：记忆 %s 未改动。", mem_id, exc_info=True)
                return {
                    "status": "failed",
                    "memory_id": mem_id,
                    "old_text": old_stored_text,
                    "new_text": new_text,
                    "candidates": [],
                    "reason": f"更新失败（{type(exc).__name__}），原记忆未动。",
                    "layer": layer,
                    "resolution": "failed",
                    "audit": _audit(
                        picked_id=mem_id,
                        judge_reason=judge_reason,
                        similarity=similarity,
                        cands=candidates,
                    ),
                }
            # 验：id 还在、新文本对得上、搜旧内容找不回旧文本。
            verified = self._verify_correction(mem_id, safe_new, old_stored_text)
            if not verified["ok"]:
                # 回滚：把旧文本和旧 metadata 写回去，再报失败。
                try:
                    self._memory.update(
                        mem_id, text=old_stored_text, metadata=_provenance(
                            old_metadata.get("source") or SOURCE_INFERRED,
                            old_metadata.get("confidence")
                            if isinstance(old_metadata.get("confidence"), (int, float))
                            else CONFIDENCE_INFERRED,
                            bool(old_metadata.get("user_confirmed")),
                            created_at=old_metadata.get("created_at"),
                            slot=old_metadata.get("slot"),
                        )
                    )
                except Exception:
                    logger.warning("纠错回滚失败：记忆 %s 可能处于新旧混合状态。", mem_id, exc_info=True)
                return {
                    "status": "failed",
                    "memory_id": mem_id,
                    "old_text": old_stored_text,
                    "new_text": new_text,
                    "candidates": [],
                    "reason": "更新后校验没过（%s），已尝试恢复原记忆。"
                    % verified.get("reason", "未知原因"),
                    "layer": layer,
                    "resolution": "failed",
                    "audit": _audit(
                        picked_id=mem_id,
                        judge_reason=judge_reason,
                        similarity=similarity,
                        cands=candidates,
                    ),
                }
            logger.info("已纠正记忆：id=%s。", mem_id)
            return {
                "status": "updated",
                "memory_id": mem_id,
                "old_text": old_stored_text,
                "new_text": safe_new,
                "candidates": [],
                "reason": "",
                "layer": layer,
                "resolution": "updated",
                "audit": _audit(
                    picked_id=mem_id,
                    judge_reason=judge_reason,
                    similarity=similarity,
                    cands=candidates,
                ),
            }

    def _verify_correction(self, mem_id, new_text, old_text):
        """验纠错结果：id 还在、新文本对得上、搜旧内容找不回旧文本。

        返回 {"ok": bool, "reason": 没过的原因}。搜旧内容时只看
        “旧文本本身”还在不在结果里（新文本里可能也提到旧词，
        比如“以前喜欢 Python，现在喜欢 Go”这种写法，那不算旧记忆还在）。
        """
        try:
            current = {
                item["id"]: item
                for item in self.get_all(include_sensitive=True)
            }
        except Exception:
            return {"ok": False, "reason": "校验时读库失败"}
        item = current.get(mem_id)
        if item is None:
            return {"ok": False, "reason": "改完记忆不见了"}
        if _normalize_memory_text(item.get("text") or "") != _normalize_memory_text(new_text):
            return {"ok": False, "reason": "改完文本和新内容对不上"}
        try:
            found = self.search(old_text, top_k=5)
        except Exception:
            return {"ok": False, "reason": "校验时检索失败"}
        for hit in found:
            if _normalize_memory_text(hit.get("text") or "") == _normalize_memory_text(old_text):
                return {"ok": False, "reason": "旧文本还能检索到"}
        return {"ok": True, "reason": ""}

    # ---------- 删除范围展示 / 删除日志（深度删除的配套工具） ----------

    def deletion_scope(self, memory_id):
        """展示一条记忆的删除范围（删之前给 BOOS 看、让他确认用）。

        返回字典：
          {"found": 是否找得到这条记忆,
           "memory": {"id", "text", "metadata"} 或 None,
           "history_entries": 这条记忆的审计行（history() 的原样返回）,
           "history_count": 审计行数,
           "in_scope": [在删除范围内的位置：向量库、history 表审计行],
           "out_of_scope": [不在删除范围内的位置及原因：
             history.db 的 messages 表（原始对话记录，不是记忆本身，
             删记忆不动它；它也不会被重新提取——提取只对新轮次跑）；
             日志文件（适配器日志只记 id 和条数，不记记忆原文，
             所以没什么可删的）；BOOS 自己在 Agent 之外做的备份
             （Agent 管不着，得 BOOS 自己处理）；Chroma 底层文件碎片
             （delete 后查不到，但文件层的清理归 Chroma 管）]}
        这个方法只读不删；删不删、删哪条，由 BOOS 看了范围之后决定，
        再调 delete_memory(approve=True)。
        """
        owned = {
            item["id"]: item
            for item in self.get_all(include_sensitive=True)
        }
        memory = owned.get(memory_id) if memory_id else None
        entries = self.history(memory_id) if memory_id else []
        return {
            "found": memory is not None,
            "memory": memory,
            "history_entries": entries,
            "history_count": len(entries),
            "in_scope": [
                {
                    "where": "向量库（Chroma，memory/mem0_chroma）",
                    "what": "这条记忆的向量（1 条）",
                },
                {
                    "where": "history.db 的 history 表",
                    "what": f"这条记忆的审计行（{len(entries)} 条，含 ADD/UPDATE/DELETE 行，DELETE 行里还留着被删记忆的全文）",
                },
            ],
            "out_of_scope": [
                {
                    "where": "history.db 的 messages 表",
                    "what": "原始对话记录（说过什么话）",
                    "why": "那是对话记录，不是记忆本身；删记忆不动它。它也不会被重新提取——提取只对新轮次跑，不会回头读旧对话。",
                },
                {
                    "where": "日志文件（logs/）",
                    "what": "运行日志",
                    "why": "适配器日志只记记忆 id 和条数，不记记忆原文，所以日志里没有这条记忆的内容可删。",
                },
                {
                    "where": "BOOS 自己在 Agent 之外做的备份",
                    "what": "比如手动拷走的 memory/ 副本、旧 memories.json",
                    "why": "Agent 管不着这些文件，删不掉；要清得 BOOS 自己处理。",
                },
                {
                    "where": "Chroma 底层文件碎片",
                    "what": "向量文件删除后留下的磁盘碎片",
                    "why": "delete 后 get/search 都查不到，但文件层的彻底清理归 Chroma 管，适配器只能保证“查不到”，不能保证磁盘碎片里一个字节都不剩。",
                },
            ],
        }

    def _journal_path(self):
        """删除日志的路径：history.db 同目录下的 delete-journal.json。

        日志里只记 memory_id、文本的 SHA-256（不记原文）、审计行数
        和开始时间——删了一半时能凭它查到“哪条没删完”，又不会因为
        记日志反而把要忘掉的文本又留一份。
        """
        return Path(self.history_db_path).parent / "delete-journal.json"

    def _journal_add(self, memory_id, text, history_count):
        """删之前记一笔删除日志（删完验干净了再由 _journal_remove 清掉）。"""
        try:
            path = self._journal_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            pending = data.get("pending") or []
            pending = [e for e in pending if e.get("memory_id") != memory_id]
            pending.append(
                {
                    "memory_id": memory_id,
                    "text_sha256": hashlib.sha256(
                        (text or "").encode("utf-8")
                    ).hexdigest(),
                    "history_rows": history_count,
                    "started_at": _now_iso(),
                }
            )
            path.write_text(
                json.dumps({"pending": pending}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("删除日志写入失败：继续删，但崩了会查不到这次删除。", exc_info=True)

    def _journal_remove(self, memory_id):
        """删完验干净了，清掉这条的删除日志。清不掉时只记日志，不报错
        （删本身已经验干净了，日志残留不影响“已删干净”这个结论；
        残留的日志里也没有原文，只有 id 和哈希）。"""
        try:
            path = self._journal_path()
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            pending = [e for e in (data.get("pending") or []) if e.get("memory_id") != memory_id]
            path.write_text(
                json.dumps({"pending": pending}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("删除日志清理失败。", exc_info=True)

    def list_pending_deletes(self):
        """列出“删了一半”的删除日志（上次删除中途崩了、或删失败了留下的）。

        返回列表，每项 {"memory_id", "text_sha256", "history_rows",
        "started_at"}——只有 id 和哈希，没有原文。空列表就是没有
        未完成的删除。调用方看到有未完成的，可以对那个 id 再调一次
        delete_memory(approve=True) 补删（已删干净的会返回 False，
        表示“已经没了”；history 还没清干净的会继续清）。
        读不到/格式不对时返回 []，不抛异常。
        """
        try:
            path = self._journal_path()
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            pending = data.get("pending") or []
            return [e for e in pending if isinstance(e, dict) and e.get("memory_id")]
        except (OSError, ValueError):
            return []

    def _purge_history_rows(self, memory_id):
        """清掉 history.db 里这条记忆的全部审计行（深度删除的第二步）。

        分两路（都要做，缺一路都不算清干净）：
        1. 底层 memory 对象如果有 purge_history 方法（测试用的
           FakeMemory 有），调它——返回删掉的行数。
        2. 真实的 history.db 是个 sqlite 文件（表 history，列 memory_id /
           old_memory / new_memory / event / is_deleted……）：直接连上去
           DELETE WHERE memory_id=?，再 VACUUM 一下（把删掉的文本从
           文件空闲页里也清掉，不然文本还躺在文件碎片里）。VACUUM
           失败时（比如别的连接正占着库）只记日志、不报错，但结果里
           vacuum_ok 记 False，调用方知道“行没了，但文件碎片不保证”。
        history.db 文件不存在、或里面没有 history 表时，sqlite 这路
        返回 0 行（没什么可清的，不算失败——FakeMemory 测试就是这样）。
        返回 {"rows_removed": int, "rows_remaining": int, "vacuum_ok": bool}。
        删 sqlite 行时万一报错，抛 RuntimeError（调用方按“删失败”处理，
        绝不谎称删干净了）。
        """
        removed = 0
        try:
            purge = getattr(self._memory, "purge_history", None)
            if callable(purge):
                removed += int(purge(memory_id) or 0)
        except Exception as exc:
            raise RuntimeError(f"清审计行失败（内存里的审计记录）：{exc}") from exc
        vacuum_ok = True
        remaining = 0
        db_path = Path(self.history_db_path)
        if db_path.is_file():
            try:
                con = sqlite3.connect(str(db_path), timeout=5)
                try:
                    tables = {
                        row[0]
                        for row in con.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    }
                    if "history" in tables:
                        cols = {
                            row[1]
                            for row in con.execute("PRAGMA table_info(history)").fetchall()
                        }
                        if "memory_id" in cols:
                            cur = con.execute(
                                "DELETE FROM history WHERE memory_id = ?",
                                (memory_id,),
                            )
                            removed += cur.rowcount or 0
                            con.commit()
                            remaining = con.execute(
                                "SELECT COUNT(*) FROM history WHERE memory_id = ?",
                                (memory_id,),
                            ).fetchone()[0]
                finally:
                    con.close()
                # VACUUM：把删掉的文本从文件空闲页里也清掉。失败时
                # （比如别的连接占着库）不报错，只记 vacuum_ok=False。
                try:
                    con = sqlite3.connect(str(db_path), timeout=5)
                    try:
                        con.execute("VACUUM")
                        con.commit()
                    finally:
                        con.close()
                except Exception:
                    vacuum_ok = False
                    logger.warning("深度删除：history.db VACUUM 失败，文件碎片里可能还留着文本。", exc_info=True)
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"清审计行失败（history.db）：{exc}") from exc
        return {
            "rows_removed": removed,
            "rows_remaining": remaining,
            "vacuum_ok": vacuum_ok,
        }

    def verify_forgotten(self, memory_id, old_text=None):
        """验一条记忆是不是真的忘干净了（深度删除的第三步）。

        查三处：向量库里没有这个 id；history() 查不到审计行；
        用旧内容搜，搜不到旧文本本身。返回
        {"ok": bool, "vector_gone": bool, "history_clean": bool,
         "search_empty": bool, "history_rows_remaining": int}。
        搜旧内容时只看“旧文本本身”在不在（新文本里提到旧词不算）。
        出错时对应项记 False（验不出来就当没干净，不谎称干净）。
        """
        vector_gone = True
        try:
            vector_gone = not any(
                item["id"] == memory_id
                for item in self.get_all(include_sensitive=True)
            )
        except Exception:
            vector_gone = False
        history_clean = False
        history_rows = 0
        try:
            history_rows = len(self.history(memory_id))
            history_clean = history_rows == 0
        except Exception:
            history_rows = -1
        search_empty = True
        if old_text:
            try:
                for hit in self.search(old_text, top_k=5):
                    if _normalize_memory_text(hit.get("text") or "") == _normalize_memory_text(old_text):
                        search_empty = False
                        break
            except Exception:
                search_empty = False
        return {
            "ok": bool(vector_gone and history_clean and search_empty),
            "vector_gone": vector_gone,
            "history_clean": history_clean,
            "search_empty": search_empty,
            "history_rows_remaining": history_rows,
        }

    def delete_memory(self, memory_id, approve=False):
        """删一条记忆（深度删除：向量 + history 审计行都清掉）。

        三道关，缺一个都不删：
        第一道：归属校验——这条记忆必须属于当前用户（在 get_all() 里
        能找到这个 id），别人的记忆删不掉（mem0 的 delete(id) 本身
        不校验归属，这里补上）。
        第二道：BOOS 明确同意——调用方必须传 approve=True，否则报
        PermissionError。删除记忆必须经过 BOOS 同意，不能悄悄删。
        调用方在传 approve=True 之前，必须先调 deletion_scope()
        把删除范围展示给 BOOS 看（删哪条、删几行历史、哪些地方
        不在删除范围内），BOOS 看过、点了确认，才算同意。
        第三道：验干净——向量删掉、history 行清掉之后，验三处都干净
        （见 verify_forgotten）才返回 True；验不过、或中间哪步报错，
        抛 RuntimeError（并在异常信息里写清“向量删了没有、history
        还剩几行”），绝不返回 True 谎称删干净了。
        流程（删之前先写删除日志，见 _journal_add）：
        1. 归属校验 + 记删除日志（只含 id 和文本哈希，不含原文）；
        2. 调 mem0.delete(id) 删向量；
        3. 清 history 行（内存审计 + history.db，见 _purge_history_rows）；
        4. 验三处都干净，清删除日志，返回 True。
        返回 True（删干净了）；找不到记忆时返回 False（已经没了，
        也算删干净了）；mem0 报错、清 history 失败、验不过时抛异常，
        调用方决定怎么处理。
        注意：归属校验要用 include_sensitive=True 的 get_all——
        旧库里已有的敏感记忆也要能删掉，不能因为它含敏感信息
        就查不到、删不掉。
        注意：现在还没有删除的网页界面/命令行入口，这个方法是给以后
        的删除入口准备的；任何入口调它之前，都必须先问 BOOS，
        并先展示 deletion_scope()。
        注意：删完不会“复活”——提取只对新轮次的对话跑（add_turn 只
        处理传进来的最近几轮消息），不会回头把旧对话、旧 history
        再提取一遍；所以一条被忘掉的记忆，只有 BOOS 重新说起它，
        才会被重新记下来。不要另建一个后台整理任务去重提旧内容，
        那会把忘掉的东西捞回来。
        """
        if not approve:
            raise PermissionError("删除记忆需要 BOOS 同意：请先确认再删。")
        if not memory_id:
            return False
        with _WRITE_LOCK:
            owned = {
                item["id"]: item
                for item in self.get_all(include_sensitive=True)
            }
            if memory_id not in owned:
                logger.warning("删除被拒：记忆 %s 不属于当前用户。", memory_id)
                return False
            old_text = owned[memory_id].get("text") or ""
            history_count = len(self.history(memory_id))
            self._journal_add(memory_id, old_text, history_count)
            try:
                self._memory.delete(memory_id)
            except Exception as exc:
                raise RuntimeError(
                    f"删除失败：向量没删掉（{type(exc).__name__}），"
                    "记忆还在库里，history 也没动。"
                ) from exc
            logger.info("已删除记忆向量：id=%s。", memory_id)
            try:
                purge = self._purge_history_rows(memory_id)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"删除了一半：向量已删，但清 history 失败（{exc}），"
                    f"history 可能还剩审计行。请对 {memory_id} 再删一次补清，"
                    "或手动检查 history.db。"
                ) from exc
            verified = self.verify_forgotten(memory_id, old_text)
            if not verified["ok"]:
                raise RuntimeError(
                    "删除了一半：向量已删，但校验没过——"
                    f"向量已清={verified['vector_gone']}，"
                    f"history 已清={verified['history_clean']}，"
                    f"检索已空={verified['search_empty']}，"
                    f"history 还剩 {verified['history_rows_remaining']} 行。"
                    f"请对 {memory_id} 再删一次补清，或手动检查 history.db。"
                )
            if not purge["vacuum_ok"]:
                logger.warning(
                    "删除已验干净，但 history.db VACUUM 没成，文件碎片里可能还留着文本。"
                )
            self._journal_remove(memory_id)
            logger.info(
                "已深度删除记忆：id=%s，history 清掉 %d 行。",
                memory_id,
                purge["rows_removed"],
            )
            return True

    def history(self, memory_id):
        """取一条记忆的修改历史（mem0 自带的 SQLite 审计记录）。

        返回：历史条目列表（每项含 old/new 内容与时间戳），
        查不到或出错时返回空列表，不抛异常。
        注意：用 delete_memory() 深度删过的记忆，历史行也被清掉了，
        这里查不到（返回 []）——“删过的记忆历史里还能查到”是旧行为，
        深度删除改掉了它。只改过（correct_memory）的记忆，历史里
        还留着 UPDATE 审计行（旧文本在 old_memory 里），有据可查。
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
