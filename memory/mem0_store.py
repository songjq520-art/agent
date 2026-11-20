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
#       时间戳（第一步，2026-11-10）：created_at 是微秒级、严格单调递增
#       的 UTC ISO 时间串（带 Z），同一批写进去的几条各拿一个 write_seq
#       序号分先后；created_at_source 说明这个时间是哪来的
#       （written：这次写进去的；backfilled：老记忆后补的；
#       migrated：迁移时沿用的旧日期）。覆盖规则：时间戳决定新旧，
#       来源可信度决定有没有资格覆盖——低可信度的新记忆不能直接覆盖
#       高可信度的旧事实，改标 pending_confirmation（待确认）挂起来，
#       只有真用上时才问 BOOS（见 _resolve_conflict、_apply_coverage_guard）。
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from memory.outbox import MemoryOutbox

# （拆分，2026-11-13：纯函数按功能拆进 memory/ 下的小模块，这里只做
#  re-export——行为不变，测试和调用方继续从本模块 import，一切照旧。）
from memory.timeutil import (
    _today,
    _now_iso,
    _now_epoch,
    _parse_created_at,
    _next_write_seq,
    _seed_write_seq,
    _NOW_LOCK,
    _NOW_LAST,
    _WRITE_SEQ_LOCK,
    _WRITE_SEQ,
)
from memory.money_slots import (
    _extract_money_metadata,
    _known_money,
    _money_slot_relation,
    _normalize_money_metadata,
    _flatten_money,
    extract_core_slots,
    _slot_values_conflict,
    _MONEY_CURRENCIES,
    _MONEY_PERIODS,
    _MONEY_AMOUNT_RE,
    _MONEY_SYMBOL_AMOUNT_RE,
    _MONEY_PERIOD_WORDS,
    _MONEY_BUY_TARGET_RE,
    _MONEY_DCA_TARGET_RE,
    _MONEY_DCA_AMOUNT_TARGET_RE,
    _MONEY_DCA_AMOUNT_TARGET_NO_DE_RE,
    _MONEY_NON_TARGET_LEADS,
    _MONEY_FLAT_KEYS,
    _MONEY_UNSET,
    _SLOT_STOPWORDS,
    _SLOT_RELATIVE_DATES,
    _SLOT_DATE_RE,
    SLOT_AGENT_NAME,
    SLOT_USER_NAME,
    SLOT_SOUL,
    SLOT_COMPUTER,
    KNOWN_SLOTS,
    AGENT_NAME_VALUE,
    AGENT_NAME_TEXT,
    AGENT_SOUL_TEXT,
    COMPUTER_VALUE,
    COMPUTER_TEXT,
)
from memory.provenance import (
    _provenance,
    _normalize_metadata,
    _metadata_for_store,
    _credibility_rank,
    _is_core_memory,
    _parse_created_at,
    _memory_order_key,
    _resolve_conflict,
    _carry_protection_fields,
    SOURCE_USER,
    SOURCE_INFERRED,
    SOURCE_UNCONFIRMED,
    CONFIDENCE_USER,
    CONFIDENCE_INFERRED,
    CREDIBILITY_CONFIRMED,
    CREDIBILITY_USER,
    CREDIBILITY_INFERRED,
    CREDIBILITY_UNCONFIRMED,
    CREATED_AT_SOURCES,
    _PROTECTION_RESULT_KEYS,
)
from memory.redaction import (
    _id18_check_ok,
    _luhn_ok,
    _redact_sensitive,
    _contains_sensitive,
    _meaningful_after_redaction,
    _normalize_memory_text,
    _find_duplicate_memory,
    _REDACTED,
    _PRIVATE_KEY_RE,
    _JWT_RE,
    _KNOWN_KEY_RES,
    _ID18_RE,
    _ID15_RE,
    _CARD_CANDIDATE_RE,
    _SECRET_LABELS,
    _LABELED_SECRET_RE,
    _GENERIC_TOKEN_RE,
    _ID18_WEIGHTS,
    _ID18_CHECK_CHARS,
    _MEANINGLESS_RES,
)
from memory.keywords import (
    _extract_keywords,
    _keyword_weight,
    _cosine_similarity,
    _STOPWORDS,
    _CJK_RE,
)
from memory.correction_text import (
    _detect_correction,
    _latest_user_content,
    _strip_correction_sentences,
    _detect_correction_fragment,
    _is_bare_retraction,
    _detect_identity_statement,
    _parse_chinese_number,
    _is_joke_or_exaggeration,
    _filter_extraction_messages,
    _correction_result,
    _parse_pending_selection,
    _parse_judge_choice,
    _CORRECTION_PHRASES,
    _CORRECTION_SEPARATORS,
    _BUDUI_BOUNDARY,
    _CHINESE_NUMERALS,
    _FRAGMENT_MAX_LEN,
    _VAGUE_FRAGMENT_VALUES,
    _RETRACTION_SIGNALS,
    _RETRACTION_TRAILING_PUNCT,
    _CONDITIONAL_LEADS,
    _JOKE_MARKERS,
    _JOKE_AGE_LIMIT,
    _JOKE_HEIGHT_LIMIT_M,
    _CN_DIGITS,
    _CN_UNITS,
    _AGE_RE,
    _HEIGHT_RE,
    _IDENTITY_USER_TRIGGERS,
    _IDENTITY_AGENT_TRIGGERS,
    _IDENTITY_VALUE_MAX_LEN,
    _IDENTITY_TRAILING_PARTICLES,
    _IDENTITY_QUESTION_SUFFIXES,
)
from memory.versioning import (
    _read_version_file,
    _write_version_file,
    _check_version_file,
    CUSTOM_INSTRUCTIONS_VERSION,
)
# （整合，2026-11-13：账本 / 衰减 / 关系 / 备份四个新模块。
# 衰减的名字直接引进来重导出——tests/test_memory_decay.py 从这里
# import 它们；账本、关系、备份用模块名前缀调，避免和 Mem0Store
# 的同名方法撞上。）
from memory import backup as _backup
from memory import ledger as _ledger
from memory import relations as _relations
from memory import inference as _inference
# （记忆整理，2026-11-19：补丁 5 BM25 关键词索引 + 原样匹配，补丁 1
# 两段式检索——都只在 search 里用，见 search 的注释。）
from memory import bm25 as _bm25
# （记忆整理，2026-11-19：补丁 3 语气、补丁 4 对原话、十三节模型路由——
# 语气本地筛 + 对原话检查在写流程里调，路由在后台判语气和自检抽查时调。）
from memory import content_check as _content_check
from memory import model_router as _model_router
from memory import self_check as _self_check
from memory import tone as _tone
# （图谱推理第一期，2026-11-18：图快照 + 实体抽取 + PPR 检索。
# 图是索引不是账本，快照放 memory/graph_snapshot.json，和账本同目录，
# .gitignore 里 memory/*.json* 已覆盖，不进仓库。）
from memory import graph as _graph_mod
from memory.graph_entities import extract_entities, extract_query_entities
from memory.decay import (
    DECAY_DELETE_BACKUP_PREFIX,
    DECAY_DELETE_IDLE_DAYS,
    DECAY_DORMANT_IDLE_DAYS,
    DECAY_DORMANT_WEIGHT,
    DECAY_HALF_LIFE_DAYS,
    DECAY_KIND_EPISODIC,
    DECAY_KIND_STABLE,
    DECAY_PROMOTION_DAYS,
    DECAY_STATE_DORMANT,
    DECAY_STATE_VERBATIM,
    DECAY_WEIGHT_FLOOR,
    _decay_day,
    _decay_idle_days,
    _decay_iso,
    _decay_now_dt,
    classify_decay_kind,
    decay_should_delete,
    decay_should_go_dormant,
    decay_weight,
    promotion_reached,
    record_seen_occurrence,
    should_promote,
    _dump_seen_list,
    _parse_seen_list,
)

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

# FETCH_ALL_TOP_K（“取全库”时传给 mem0 get_all 的 top_k）：
# mem0 2.2.1 的 get_all 默认 top_k=20，不传就只返回前 20 条——
# 2026-11-15 上午就是这么把库“看少”的：火锅、冒菜一直在库里，
# 直查只返回前 20 条，看着像没存进去；去重时快照也只看到前 20 条，
# 刚存的那条看不见，又被重复存了一遍。凡是要“全部记忆”的地方
# （快照、去重、守卫核对、按 id 找原始 metadata、补时间戳、
# 关键词补充）都必须走 _fetch_all_raw()，不许再裸调 get_all。
# 现在库里 30 多条，1000 留足余量；真到 1000 那天再改这个数。
# （mem0 内部按 top_k*4 从向量库取，1000 就是取 4000 条，个人库没压力；
# 遥测已关，top_k ≥ 50 的 mem0 提示也不会打出来。）
FETCH_ALL_TOP_K = 1000

# KEYWORD_MIN_WEIGHT（关键词补充信号的最低权重）：语义检索之外，
# 再用 v1 那套中文两字窗口关键词做补充。权重 = 匹配到的关键词长度之和，
# ≥ 2 才算命中（至少一个两字关键词）。补充信号只排在语义结果后面，
# 且不超过 top_k 总数；召回不够时再调，不够时再考虑融合。
KEYWORD_MIN_WEIGHT = 2

# ARCHIVE_SEARCH_THRESHOLD（补丁 1 两段式检索的门槛，初始值 0.6，
# 以后按回测调）：常用层里最好的一条语义分 ≥ 这个数，就只给常用层，
# 档案层的不进结果；低于它，档案层的也进来（标清是档案）。
# 语义分是 mem0 给的 0–1 分；没有语义结果时按 0 算（档案层进来）。
ARCHIVE_SEARCH_THRESHOLD = 0.6

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
# 2026-11-12 第 3 批重测（4 号，同样本是整句原文、带“用户喜欢…”前缀，
# 18 对，见 /tmp/batch3-probe.py）：同话题 0.757–0.907，跨话题
# 0.411–0.763。0.56 只拦住 3/8 个跨话题（网太松）；0.70 拦住 7/8、
# 同话题 10/10 放行，是量出来的最好分界（干净空隙在 0.656–0.757 之间，
# 取中点约 0.70）。唯一漏网的是“助手名字叫十二”vs“十一是助手的大哥”
# （0.763，名字长得太像， embedding 分不开）——这道网本来就是第二层，
# 主防线是模型判 + 问 BOOS，漏网的这一类由模型判兜底。
CORRECTION_TOPIC_MIN_SIMILARITY = 0.70

# DEDUP_SIMILARITY_MIN（去重预筛：新记忆和旧记忆的向量相似度下限）：
# 46 号：相似度 ≥ 这个值才值得请模型判“是不是同一件事”；低于它
# 直接当不同事实，不请模型。2026-11-12 实测（bge-small-zh-v1.5）：
# 同一事实换说法 0.735–0.978，同话题不同事实 0.783–0.934——只看
# 相似度分不开（“每天定投 5 块”vs“每天定投 10 块”高达 0.934），
# 所以这里只是预筛（把明显不同话题的先滤掉），真正的定夺是模型判；
# 0.70 是预筛线，不是合并线——过了线也不直接合并，必须模型点头。
# 0.70 兜住了全部 10 对真重复（最低 0.735）。
DEDUP_SIMILARITY_MIN = 0.70

# 判断 (a) 号：按意思去重（宽召回 + 核心槽位 + 结构化裁决）。
# DEDUP_RECALL_THRESHOLD（宽召回下限 = 检索阈值 0.40）：
# 新记忆先按 0.40 召回 top_k=20 个候选（以前只看 ≥0.70），
# 换说法多、相似度被拉低的真重复也能捞回来。
# DEDUP_HARD_THRESHOLD（硬合并线 0.90）：
# 相似度 ≥0.90 且没有槽位冲突时，直接算重复，不请模型。
# DEDUP_SLOT_SIMILARITY_FLOOR（槽位过滤线 0.55）：
# 候选和新记忆没有共同核心槽位、且相似度 <0.55 时直接丢掉，
# 不送模型判（省一次模型调用）。
DEDUP_RECALL_THRESHOLD = 0.40
DEDUP_HARD_THRESHOLD = 0.90
DEDUP_SLOT_SIMILARITY_FLOOR = 0.55
DEDUP_RECALL_TOP_K = 20

# 第二步衰减（连续衰减，2026-11-13，BOOS 定稿《十二记忆整理方案》第二版）。
# 每条记忆自带权重：越久没用、越不重要，自己往下掉；检索里真正用上
# 的才强化一次。三个工作台挑出来的坑，先定死再写代码：
#   1. 衰减绝不能跟检索阈值 0.4 打架——资格只看原始语义分（≥0.4 才进），
#      权重只管进围之后的排序，再低也不会让记忆在观察期内搜不到
#      （见 search：先按原始分过滤，再按“分 × 权重”排）。
#   2. 顺序：先标冲突（旧的标“被取代”），再做检索强化——去重裁决
#      判了“是同一件事”之后，才记一笔“又出现一次”（见
#      _record_decay_occurrence 的调用点，都在去重返回之后）。
#   3. 月扫只标记、不直接删（见 sweep_monthly）。
# 沉底规矩（2026-11-13 晚最终方案，BOOS 定，取代 20:43 的“30 天压摘要、
# 半年删”）：只打标、不自动压摘要、不自动删——
#   30 天没人找 / 没用上 → 打标 dormant（decay_state 从 verbatim 变成
#   dormant，原话和向量文本都不动；被用上就回到原话态、重新强化）；
#   满半年（180 天，从最后一次用上算起）还没人找 → 再打标 decay_delete_due，
#   只做标记，等 BOOS 亲口说删才删；
#   真删是另一件显式的事（见 delete_due_memories：要 BOOS 点头，删前有备份）；
#   核心记忆钉住不动，不走这条（权重恒为 1，月扫直接跳过）。
# 半衰期是指数衰减的“多久掉一半”：稳定事实 180 天（慢）、一次性经历
# 14 天（快）。30 天没用的一次性经历权重 ≈0.23，正好掉到沉底线
# （DECAY_DORMANT_WEIGHT = 0.30）下面；稳定事实 30 天才掉到 ≈0.89，
# 不会被标沉底。权重下限 0.10：再久不用也不会归零，排位垫底而已。
# 晋升：一次性经历在不同天、不同场合出现 3 次就晋升成稳定事实
# （DECAY_PROMOTION_DAYS = 3）；一次聊天里说 3 遍不算——出现次数
# 按“不同日期”数（decay_seen_days），同一天重复提不加数。
DECAY_KIND_STABLE = "stable"
DECAY_KIND_EPISODIC = "episodic"
DECAY_HALF_LIFE_DAYS = {DECAY_KIND_STABLE: 180, DECAY_KIND_EPISODIC: 14}
DECAY_WEIGHT_FLOOR = 0.10
DECAY_DORMANT_WEIGHT = 0.30
DECAY_DORMANT_IDLE_DAYS = 30
DECAY_DELETE_IDLE_DAYS = 180
DECAY_PROMOTION_DAYS = 3
DECAY_STATE_VERBATIM = "verbatim"
DECAY_STATE_DORMANT = "dormant"
DECAY_DELETE_BACKUP_PREFIX = "decay-delete-backup-"

# 判断 (a) 号：裁决的可信度线。
# 模型说重复且 confidence ≥0.80 时才合并（新条当证据挂到旧条下）；
# 低于 0.80 或模型判不出来时，新条照常留，标 needs_review=true
#（等兜底清理时再看）。

# VALUE_THRESHOLD（价值关卡：提取出的记忆值不值得长期记的分界线）：
# 43 号：模型给每条新提取的记忆打 0–1 分，低于这个值不进库。
# 0.5 是适中的默认（中点）：稳定事实/明确偏好/重要决定通常 ≥0.7，
# 一次性请求/闲聊/临时状态通常 ≤0.3；拿不准的往低打（提示词里写明），
# 所以 0.5 不会把真事实拦在外面。BOOS 看了可以调（见第 3 批报告）。
VALUE_THRESHOLD = 0.5

# PENDING_IDLE_SECONDS（待定区闲置提交：正常轮次先进待定区，等下一轮或等闲置
# 这么久才提交提取——给“不不不”撤回留窗口，又保证最后一轮没人接话时也
# 能落库（2026-11-20 游泳那句就是没下一轮才丢的）。测试里可以改小
# store._pending_idle_seconds，不用真等 60 秒。
PENDING_IDLE_SECONDS = 60.0

# RETRACTION_CANDIDATE_MARKERS（撤回候选触发词）：34 号。
# 撤回判断分两步：先用这些词便宜地筛出“可能是撤回”的句子，再请模型
# 判意图——不是每轮都请模型（贵），也不是只认整句信号词（漏）。
# 注意：这里只是触发器，不是判决——带这些词的不一定是撤回（比如
# “不对，是绿色”就是纠正），最终由模型判；没带这些词的也不直接放行，
# 光秃秃的撤回（“不不不。”）走已有的 _is_bare_retraction 快捷路。
RETRACTION_CANDIDATE_MARKERS = (
    "不不", "不对", "算了", "收回", "别记", "说错", "撤回", "取消",
    "不算", "当我没说", "忘掉", "删掉", "不要记", "别存", "不记",
    "作废", "刚才那句", "上面那句", "上一句",
    # 34 号补（2026-11-12）：自然说法的撤回——“其实也不是很喜欢吃”
    # 里没有上面这些词，0a 根本不跑，撤回句就被当成新事实提出来了。
    # “也不是”“不是很”只是请模型判意图的触发词，判不是撤回就走正常路；
    # 新偏好（“我不是很喜欢吃辣”）判完不是撤回，照常存，不会丢。
    "也不是", "不是很",
)

# RETRACTED_TRASH_FILENAME（撤回软删除的回收站文件）：34 号。
# 软删除 = 从向量库删掉 + 全文记进这个文件（memory/ 目录下，
# .gitignore 已覆盖，不上传），随时能恢复。文件里只放被撤回的记忆
# （id、原文、元数据、撤回时间），不放别的。
RETRACTED_TRASH_FILENAME = "retracted_memories.json"


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
    false memory the user has to correct later.
14. NEVER turn a request to manage memories (delete / update / correct a
    stored memory) into a new fact about the user. "Delete X" only means
    delete the stored record — it does NOT mean the user no longer likes
    X, changed their mind, or changed a preference, and "the user asked
    to delete X" is not a long-term fact worth storing (the deletion
    itself is already recorded in the ledger). A preference change is
    only stored when the user explicitly states the new preference in
    their own words (e.g. "I don't like X anymore")."""

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





# 判断 (b) 号：动态数字事实的核实可信度上限。
#   未核实（没两个独立来源印证）：0.35——不进稳定事实层，须确认才用。
#   已核实（两个独立来源对得上）：0.75——网上来的数再对也比不上官方。
#   权威来源一家验证（官方交易所、政府气象局、央行）：0.85。
# 有效期：行情 24 小时、天气 12 小时、其他 24 小时——过期了要重查。
VERIFICATION_CAP_UNVERIFIED = 0.35
VERIFICATION_CAP_VERIFIED = 0.75
VERIFICATION_CAP_AUTHORITATIVE = 0.85
VERIFICATION_EXPIRY_HOURS = {"market": 24, "weather": 12, "other": 24}




































































































































def _describes_memory_management(text):
    """这段提取出的文本是不是在复述“管记忆的动作”（删/改一条记忆）。

    删/改记忆的动作本身已经记在账本里（delete/update 事件），复述成
    一条长期记忆没有价值，还容易把动作脑补成偏好变了（“不再喜欢 X”）。
    只认两种形状：
    - “删”+“记忆/记录”：比如“要求删除此前记录的相关偏好记忆”、
      “把这条从记忆里删掉”；
    - “改”+“记忆”：比如“修改了记忆里的定投金额”。
    “改”+“记录”不拦（比如“修改了会议记录”是正经工作事实，不是
    管记忆的动作）。真凭 BOOS 原话说的偏好变化（“我不爱吃冒菜了”）
    不提删/改记忆，不在这个形状里，不拦。
    """
    if not text:
        return False
    if "删" in text and ("记忆" in text or "记录" in text):
        return True
    if "改" in text and "记忆" in text:
        return True
    return False


# 中文数字短语（补丁 4 对原话用）：后面跟了量词/单位、或在句末的才转，
# “一起”“一般”“十分喜欢”里的“一/十”不转（不是数字，别乱碰）。
# 转法复用 correction_text._parse_chinese_number（十/百/千组合都认，
# “两”算 2）；“万”它不认（“一万块”转不出来，留原样——对不上时只标
# 待复核，不判错，模型抽查是兜底）。
_CN_NUMERAL_PHRASE_RE = re.compile(
    r"[零一二三四五六七八九十百千两]{1,8}"
    r"(?=[块元万亿%％个只支天月年次]|小时|分钟|秒|$)"
)


def _cn_numerals_to_digits(text):
    """把文本里的中文数字短语转成阿拉伯数字（只转数字短语，别的不动）。
    对原话检查之前，记忆文本和出处原话都先过一遍这个：BOOS 原话常说
    “五块”，提取出来的记忆常写“5 块”——不转的话每个数都对不上，
    全被误标待复核。转不出来的（比如带“万”的）留原样。"""
    if not text:
        return text or ""

    def _replace(match):
        parsed = _parse_chinese_number(match.group(0))
        return str(parsed) if parsed is not None else match.group(0)

    return _CN_NUMERAL_PHRASE_RE.sub(_replace, text)


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
        protection_events_path=None,
        ledger_path=None,
        snapshot_path=None,
        graph_path=None,
        inference_calibration_path=None,
        backup_dir=None,
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
        # 保护事件记录（失败关闭的可见留痕）：快照失败、恢复失败这些事
        # 不能只写日志——记到本地 JSONL 里（memory/protection-events.jsonl，
        # gitignore 覆盖，不进仓库），面板上有接口能读到，BOOS 看得见。
        self.protection_events_path = (
            Path(protection_events_path)
            if protection_events_path
            else project_root / "memory" / "protection-events.jsonl"
        )
        # 事件账本（最终方案，2026-11-13）：memory_events.jsonl 是家，
        # mem0 只是可丢弃的索引。先写账本、再写索引；账本写不进去，
        # 索引那步也别写。快照（一年压一次）和备份（带校验、真恢复）
        # 也都放 memory/ 下，gitignore 覆盖，不进仓库。
        self.ledger_path = (
            Path(ledger_path)
            if ledger_path
            else project_root / "memory" / "memory_events.jsonl"
        )
        self.snapshot_path = (
            Path(snapshot_path)
            if snapshot_path
            else project_root / "memory" / "memory_events_snapshot.json"
        )
        # 图谱推理第一期：图快照放 memory/graph_snapshot.json（和账本
        # 同目录）。默认跟账本走——测试里账本指到临时目录时，图也跟
        # 过去，不污染工作区。_graph 懒建：第一次存入/检索时才读快照
        # （没有就从账本重建），启动不为它多花时间；_graph_lock 管
        # 这把懒建 + 写快照，别让两个线程同时建两张图。
        self.graph_path = (
            Path(graph_path)
            if graph_path
            else self.ledger_path.parent / "graph_snapshot.json"
        )
        self._graph = None
        self._graph_lock = threading.Lock()
        self.backup_dir = (
            Path(backup_dir)
            if backup_dir
            else project_root / "memory" / "backups"
        )
        # _ledger_lock：账本追加的锁——序号是“读最后一行 + 1”算出来的，
        # 两个线程同时追加会编出同一个号；追加那一下串起来，别的写操作
        # 不走这把锁（写记忆本来就串在 _WRITE_LOCK 里）。
        self._ledger_lock = threading.Lock()
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
        # 见 _detect_identity_statement）。
        # 2026-11-20 游泳事故之后：待定区不再只等下一轮——闲置满
        # _pending_idle_seconds（默认 60 秒）也提交（_on_pending_idle），
        # 保证最后一轮没人接话时也能落库；outbox 落盘文件在真正落库前
        # 一直留着（见 _commit_pending_turn），重启能捞回来重交。
        self._pending_turn = None
        # _pending_idle_seconds / _pending_timer（待定区闲置提交）：
        # 停车后（重）启一个守护定时器，到点仍是同一轮就提交；新轮次
        # 来了就取消旧的、重设。测试里可以改小 _pending_idle_seconds。
        self._pending_idle_seconds = PENDING_IDLE_SECONDS
        self._pending_timer = None
        # _pending_inference（待定的揣摩）：上一轮揣出来、当场问了 BOOS
        # 的那条推断（见 memory/inference.py + infer_from_turn）。
        # 只放内存里，重启就没了——问话本身就是当场的；账本里那份
        # （inference 事件，status pending）才是 durable 的记录。
        # 下一轮 BOOS 光秃秃地回“对/不对”时，就认这一条；下一轮没回，
        # 它就过期了（账本里还是 pending，以后对账/自检能翻出来）。
        self._pending_inference = None
        # _longterm_skip_until_seq（长期揣摩的跳过线）：攒够 20 条新记忆
        # 时跑一次长期揣摩；那次模型说“没什么可猜的”时，把当前账本序号
        # 记下来，下次要再攒够 20 条新的才重跑——不然每轮都多调一次模型。
        self._longterm_skip_until_seq = 0
        # 揣摩开口门槛（2026-11-17，校准自动调）：当前门槛是“开口的最
        # 低把握”（low/medium/high），默认 low——低把握也敢说，只说
        # “我只是猜的”（BOOS 的铁规矩：揣摩要敢说）。门槛存在
        # inference_calibration.json 里（默认和账本同目录，gitignore
        # 覆盖、不进仓库），重启还在；文件没了/读坏了就回默认 low。
        # _inference_calibration_baseline：上次调门槛时各档的裁决数
        # （{把握: {confirmed, denied}}）——调门槛只看这之后新来的反馈
        # （增量），旧账不重复算，不然同一批过分自信的数据每轮都推着
        # 门槛往上走；新反馈来了也只按新反馈的信号动。
        self.inference_calibration_path = (
            Path(inference_calibration_path)
            if inference_calibration_path
            else self.ledger_path.parent / "inference_calibration.json"
        )
        (
            self._inference_threshold,
            self._inference_calibration_baseline,
        ) = self._load_inference_calibration_state()
        # _just_deleted_texts（本进程里刚删掉的原文）：{归一化原文: 删的时间戳}
        # 2026-11-16 真机测试（拉面）：删完 10 秒，下一轮提交待定区时提取
        # 又把同一条记回来了——账本是兜底，内存里这份是第一道（账本写失败
        # 时也有据）。恢复/还原一条时把它清掉，删掉的才不会拦着以后重新说。
        self._just_deleted_texts = {}
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
        # _outbox（26 号后台存记忆）：懒建——第一次调 save_turn 时才建
        # 并启动后台线程；只调 save_turn 的进程才有它，测试里不调
        # save_turn 就不建、不启动线程。
        self._outbox = None
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
        # 写序号持久化（2026-11-19）：_WRITE_SEQ 是进程内自增、重启
        # 归零，新进程写的序号会比库里已有的小，排序键平局时就分不
        # 出新旧。启动时扫一遍库里现有的 write_seq，取最大那个当起点
        # （见 _seed_write_seq_from_store）——之后新写的序号一定比库
        # 里的大。扫不出来（库空、读失败）就从 0 开始，和以前一样；
        # 扫失败绝不能把启动搞崩。
        self._seed_write_seq_from_store()

    def _seed_write_seq_from_store(self):
        """启动时把写序号抬到库里最大的 write_seq（见 __init__）。

        扫一遍现有记忆的 metadata，取最大的 write_seq 喂给
        _seed_write_seq（只往上抬）。库空、或读失败时返回 0、不播种，
        启动照常进行——播种失败绝不能把启动搞崩。
        返回播种到的最大值（没播种时是 0）。"""
        try:
            snapshot = self._snapshot_all()
        except Exception:
            logger.warning("写序号播种：读库失败，本次不播种。", exc_info=True)
            return 0
        max_seq = 0
        for item in snapshot or []:
            seq = ((item or {}).get("metadata") or {}).get("write_seq")
            if isinstance(seq, int) and not isinstance(seq, bool) and seq > max_seq:
                max_seq = seq
        if max_seq:
            _seed_write_seq(max_seq)
        return max_seq

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
                    tracker.add_cost(cost, model=LLM_MODEL, ledger="chat")
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

    def search(self, query, top_k=5, reinforce=False, include_history=False, turn_id=None):
        """搜和问题相关的记忆（语义检索 + 关键词补充信号）。

        返回：记忆列表，每项为 {"id", "text", "score", "source",
        "metadata", "confirmed", "decay_weight", "is_history"}，source 是
        "semantic"（语义）、"keyword"（关键词补充）或 "ledger"（账本兜底），
        metadata 是规整过的来源信息（见 _normalize_metadata），confirmed
        是“BOOS 是否逐条确认过”（只有 source="user" 的记忆才可能是 True），
        decay_weight 是这条记忆现在的衰减权重（0.10–1.0，核心记忆恒为 1.0），
        is_history 是不是“以前的”（被取代/睡着的，默认不返回，见下）。
        补丁 2：reinforce=True（真用上）时，被用上的记忆里有待确认 /
        冲突待审的，项上多两个键——needs_boos_decision=True 和
        decision_reason（一句话原因），调用方看到就问 BOOS 定。
        补丁 1 两段式检索：每项多一个 layer 键（"active" 常用层 /
        "archive" 档案层 = 沉底态）；常用层最好的一条语义分 ≥ 0.6 时
        档案层不进结果，低于 0.6 时进来（标清是档案）。补丁 5：关键词
        补充改走 BM25 排名，原样匹配（基金代码/数字/人名）的那几条在
        关键词组里排最前面（项上 exact_match=True）。
        注意：search 不按来源过滤——提取的（inferred）记忆和未确认的
        旧记忆也会返回，调用方（main.py 的 _memory_block）按来源决定
        怎么用：已确认的才能当作事实，未确认的不能当作事实。
        这样设计的原因：提取的记忆是记忆功能的主力（自动存的都在这），
        过滤掉它们等于记忆功能作废；安全靠“注入时打标签、分层”
        来保证，不靠检索时偷偷丢掉。
        第二步衰减（2026-11-13）：权重只管排位，不管“能不能搜到”——
        资格还是只看原始语义分 ≥0.4（工作台坑 1 的定法：衰减再狠，
        也不让记忆在 30 天观察期内搜不到）；进了围的，语义结果按
        “原始分 × 权重”排，关键词补充按权重排。压成摘要的记忆照常
        能搜到（摘要文本在向量里），被用上就恢复原话（见 reinforce）。
        reinforce（默认 False）：只有真正把记忆用上了（拼进发给模型
        的问题里）才传 True——这时返回的这几条才刷新 last_used_at，
        没返回的候选一条也不抬。内部检索（纠错找候选、守卫找邻居）
        一律不传，绝不能把“找候选”也算成“用上了”（好几个纠错测试
        都断言了“检索不写回”，见 tests/test_mem0_correction.py）。
        turn_id（2026-11-19）：提问这轮的 id——强化时记进被用上记忆
        的会话集合（见 reinforce_memories）。提问流程（main.py
        run_question、web/server.py ask）在检索之前就生成、传进来；
        不传时只按日期数（老路子）。
        投影过滤（最终方案，2026-11-13）：检索回来之后再过筛子——
        被取代的、删了的默认不给（is_history=False 的才是当前记忆）；
        问“以前”时传 include_history=True 才放出来，并标
        is_history=True（“这是以前的”）。睡着的不屏蔽：它是衰减的
        标签，不是“不是当前了”（见 ledger.py project_and_filter）。
        账本是空的（刚启用、还没记过）时筛子不过滤，行为和以前一模一样。
        账本兜底：当前结果不够 top_k 时，翻账本原话按关键词找，
        补在后面（source="ledger"，score 是关键词权重，不是语义分）。
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
        now_dt = datetime.now(timezone.utc)
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
                metadata = _normalize_metadata(item.get("metadata"), item)
                weight = decay_weight(
                    metadata["decay_kind"],
                    metadata["last_used_at"],
                    now_dt,
                    is_core=_is_core_memory(metadata),
                )
                results.append(
                    {
                        "id": item.get("id"),
                        "text": text,
                        "score": score,
                        "source": "semantic",
                        "metadata": metadata,
                        "confirmed": metadata["confirmed"],
                        "decay_weight": weight,
                    }
                )
                seen_texts.add(text)
        except Exception:
            logger.warning("mem0 语义检索失败：本次只用关键词补充信号。", exc_info=True)
        # 关键词补充信号（补丁 5：BM25 + 原样匹配，排在语义结果后面，
        # 不超过 top_k 总数）。v1 的两字窗口权重（_keyword_weight）换成
        # BM25 排名：中文长词一样拆两字窗口，但加了词频饱和和长度归一化，
        # 常见词不再靠长度刷分。原样匹配（基金代码、数字、人名原样命中）
        # 的排在关键词组最前面——按 BM25 顺序排（整句命中的 BM25 分高，
        # 只含一个裸数字的排后面，不拿裸数字冒充精确命中）。
        if len(results) < top_k:
            try:
                all_items = self._fetch_all_raw()
                docs = [
                    {"id": item.get("id"), "text": item.get("memory") or ""}
                    for item in (all_items or {}).get("results", [])
                ]
                by_id = {doc["id"]: item for doc, item in zip(docs, (all_items or {}).get("results", []))}
                exact_ids = set(_bm25.extract_exact_matches(query, docs))
                ranked = _bm25.bm25_search(query, docs, top_k=top_k)
                ranked_ids = [hit["id"] for hit in ranked]
                exact_first = [doc_id for doc_id in ranked_ids if doc_id in exact_ids]
                rest = [doc_id for doc_id in ranked_ids if doc_id not in exact_ids]
                for doc_id in exact_first + rest:
                    if len(results) >= top_k:
                        break
                    item = by_id.get(doc_id)
                    if item is None:
                        continue
                    text = item.get("memory") or ""
                    if not text or text in seen_texts:
                        continue
                    if _contains_sensitive(text):
                        skipped_sensitive += 1
                        continue
                    metadata = _normalize_metadata(item.get("metadata"), item)
                    results.append(
                        {
                            "id": item.get("id"),
                            "text": text,
                            "score": None,
                            "source": "keyword",
                            "metadata": metadata,
                            "confirmed": metadata["confirmed"],
                            "exact_match": doc_id in exact_ids,
                            "decay_weight": decay_weight(
                                metadata["decay_kind"],
                                metadata["last_used_at"],
                                now_dt,
                                is_core=_is_core_memory(metadata),
                            ),
                        }
                    )
                    seen_texts.add(text)
            except Exception:
                logger.warning("mem0 关键词补充检索失败。", exc_info=True)
        if skipped_sensitive:
            logger.info("检索：%d 条记忆含敏感信息，已跳过，不注入。", skipped_sensitive)
        # 衰减排位：语义组按“原始分 × 权重”排，关键词组按“原样匹配优先、
        # 再按权重”排——都是稳定排序，权重全是 1.0 时顺序和原来一模一样；
        # 关键词组还是排在语义组后面（补充信号不抢主信号的位置）。
        semantic_end = sum(1 for item in results if item.get("source") == "semantic")
        results[:semantic_end] = sorted(
            results[:semantic_end],
            key=lambda item: (item.get("score") or 0) * (item.get("decay_weight") or 0),
            reverse=True,
        )
        results[semantic_end:] = sorted(
            results[semantic_end:],
            key=lambda item: (
                1 if item.get("exact_match") else 0,
                item.get("decay_weight") or 0,
            ),
            reverse=True,
        )
        # 账本状态先读出来：图合并和投影过滤共用一份，不读两遍。
        ledger_state = self.ledger_state()
        # 图检索（图谱推理第一期）：种子实体 → PPR 一步多跳，两路分数
        # 归一化相加。插在关键词补充之后、投影过滤之前；图是空的、
        # 种子找不着、PPR 没命中时什么都不做，老路结果原样。
        # 失败只记日志、不抛异常（_graph_merge 内部全包住了）。
        self._graph_merge(
            results, seen_texts, query, top_k, include_history, ledger_state
        )
        # 投影过滤之前先把这一批的 key 都记下来：被筛掉的（睡着的、
        # 被取代的、删了的）原话，兜底那步也不许再从账本里捞回来——
        # 筛子刚扔掉的，兜底不能再捡回来当现况（账本兜底只补“索引
        # 里丢了”的，不补“筛子判了旧的”）。图上找回来的也算在里头，
        # 一样不许兜底再捡。
        pre_filter_keys = {
            _ledger.ledger_event_key(
                item.get("text") or "",
                (item.get("metadata") or {}).get("created_at"),
            )
            for item in results
        }
        # 投影过滤：被取代的、睡着的默认不给；问“以前”才放出来。
        results = _ledger.project_and_filter(
            results, ledger_state, include_history=include_history
        )
        # 补丁 1 两段式检索：先给常用层，不够再翻档案。
        # 每条结果标 layer（active/archive，档案 = 沉底态 dormant）；
        # 常用层里最好的一条语义分 ≥ 0.6 时，档案层的不进结果（够用了
        # 就别翻旧账）；低于 0.6 时档案层的也进来（标清是档案），被用上
        # 就回常用层（_decay_touch 唤醒，见 reinforce）。门槛只看语义分
        # （mem0 的 0–1 分）：关键词补充没有可比的分数，不参与算分；
        # 没有语义结果时按 0 算，档案层进来。include_history=True
        # （问“以前”）时不过这道闸——问的就是旧账，全给。
        if not include_history:
            best_active = 0.0
            for item in results:
                if item.get("source") != "semantic":
                    continue
                meta = (item or {}).get("metadata") or {}
                if meta.get("decay_state") == DECAY_STATE_DORMANT:
                    continue
                score = item.get("score")
                if isinstance(score, (int, float)) and score > best_active:
                    best_active = score
            for item in results:
                meta = (item or {}).get("metadata") or {}
                item["layer"] = (
                    "archive"
                    if meta.get("decay_state") == DECAY_STATE_DORMANT
                    else "active"
                )
            if best_active >= ARCHIVE_SEARCH_THRESHOLD:
                results = [
                    item for item in results if item.get("layer") == "active"
                ]
        else:
            for item in results:
                meta = (item or {}).get("metadata") or {}
                item["layer"] = (
                    "archive"
                    if meta.get("decay_state") == DECAY_STATE_DORMANT
                    else "active"
                )
        # 账本兜底：当前结果不够时，翻账本原话按关键词找，补在后面。
        # 只补“当前”的原话：账本里标了 superseded / deleted 的 key
        # 不当现况（除非问的就是以前，include_history=True 时才放
        # 出来，并标清它是什么状态）；睡着的（dormant）是当前记忆，
        # 照常补（见 project_and_filter）。兜底项没有 mem0 id
        # （id=None）：它只是档案里的原话，拿来回答问题可以，拿去
        # update / 纠错 / 打标记不行——内部流程认 id 的地方要跳过它
        # （见 _find_slot_memories_by_text、_find_correction_candidates）。
        if len(results) < top_k:
            try:
                events, _info = _ledger.read_events(self.ledger_path)
                seen_keys = {
                    _ledger.ledger_event_key(
                        item.get("text") or "",
                        (item.get("metadata") or {}).get("created_at"),
                    )
                    for item in results
                } | pre_filter_keys
                key_states = ledger_state.get("key_states") or {}
                for hit in _ledger.find_in_ledger(
                    events, query, top_k=top_k - len(results), exclude_keys=seen_keys
                ):
                    text = hit.get("text") or ""
                    if not text or text in seen_texts:
                        continue
                    hit_state = key_states.get(hit.get("key"))
                    if not include_history and hit_state in (
                        _ledger.LEDGER_STATE_SUPERSEDED,
                        _ledger.LEDGER_STATE_DELETED,
                    ):
                        continue
                    if _contains_sensitive(text):
                        skipped_sensitive += 1
                        continue
                    meta = _normalize_metadata(hit.get("meta") or {})
                    results.append(
                        {
                            "id": None,
                            "text": text,
                            "score": hit.get("score"),
                            "source": "ledger",
                            "metadata": meta,
                            "confirmed": meta["confirmed"],
                            "decay_weight": 1.0,
                            "is_history": True,
                            "history_state": hit_state or "ledger",
                            "ledger_key": hit.get("key"),
                        }
                    )
                    seen_texts.add(text)
            except Exception:
                logger.warning("账本兜底检索失败：本次只用索引里的结果。", exc_info=True)
        if reinforce and results:
            # 补丁 2 真用上时再问：被用上的记忆里有“待确认 / 冲突待审”的，
            # 在返回项上标出来（needs_boos_decision + 一句话原因），
            # 调用方（main.py / web 面板）看到就问 BOOS 定——平时不打扰，
            # 只在真用上时问。只标、不写库，问不问由调用方定，检索不拦路。
            for item in results:
                meta = (item or {}).get("metadata") or {}
                if meta.get("pending_confirmation"):
                    item["needs_boos_decision"] = True
                    item["decision_reason"] = (
                        "这条是后来推断出来的新说法，跟 BOOS 确认过的旧说法"
                        "对着干，挂在这等你定。"
                    )
                elif meta.get("conflict_for_review"):
                    item["needs_boos_decision"] = True
                    item["decision_reason"] = (
                        "这条跟库里另一条说法对着干，等你定哪条是对的。"
                    )
            # 真正用上了才强化：只强化返回的、不是历史的那几条——
            # 历史记忆被翻出来看看不算“用上了”，不能把它唤醒回原话态。
            # 写回失败也不拦聊天——记忆本身还在，只是这次没刷新时间。
            try:
                self.reinforce_memories(
                    [
                        item.get("id")
                        for item in results
                        if item.get("id") and not item.get("is_history")
                    ],
                    now=now_dt,
                    turn_id=turn_id,
                )
            except Exception:
                logger.warning("检索强化写回失败：记忆本身还在。", exc_info=True)
        return results

    def search_history(self, query, top_k=5):
        """直接翻账本找“以前”的记忆（问“以前想去哪”这类问题时调）。

        不走 mem0 索引：翻账本原话按关键词找，返回的都是历史记忆
        （is_history=True，source="ledger"，score 是关键词权重）。
        账本里没有、或关键词一个没命中时返回空列表，不抛异常。
        """
        try:
            events, _info = _ledger.read_events(self.ledger_path)
            key_states = (self.ledger_state().get("key_states") or {})
            out = []
            for hit in _ledger.find_in_ledger(events, query, top_k=top_k):
                text = hit.get("text") or ""
                if not text or _contains_sensitive(text):
                    continue
                meta = _normalize_metadata(hit.get("meta") or {})
                out.append(
                    {
                        "id": None,
                        "text": text,
                        "score": hit.get("score"),
                        "source": "ledger",
                        "metadata": meta,
                        "confirmed": meta["confirmed"],
                        "decay_weight": 1.0,
                        "is_history": True,
                        # 历史也要分清状态：被取代的、睡着的、删了的，
                        # 问“以前”时都该给，但得标清它是什么状态。
                        "history_state": key_states.get(hit.get("key"))
                        or "ledger",
                        "ledger_key": hit.get("key"),
                    }
                )
            return out
        except Exception:
            logger.warning("翻账本找历史失败。", exc_info=True)
            return []

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

    def _fetch_all_raw(self):
        """取全库原始记录（mem0 get_all 的返回口径）。

        唯一允许裸调 self._memory.get_all 的地方：top_k 必须传
        FETCH_ALL_TOP_K。mem0 默认只给前 20 条，漏了后半个库，
        去重、守卫、找 metadata 全都会错（2026-11-15 上午的教训：
        库里 33 条，直查只看到前 20 条）。
        """
        return self._memory.get_all(
            filters={"user_id": self.user_id}, top_k=FETCH_ALL_TOP_K
        )

    def _snapshot_all(self):
        """拍一张库快照（提取前、恢复核对时用）。

        内容口径和 get_all(include_sensitive=True) 一样：全部记忆，
        空文本跳过，metadata 规整过。区别只有一点：读库失败时抛
        异常、不吞掉——调用方要分得出“库是空的”还是“库读不出来”，
        分不出来就没法失败关闭（快照读不出来还往里写，就是没保护
        硬写，见 _extract_and_save_turn）。
        """
        raw = self._fetch_all_raw()
        out = []
        for item in (raw or {}).get("results", []):
            text = item.get("memory") or ""
            if not text:
                continue
            out.append(
                {
                    "id": item.get("id"),
                    "text": text,
                    "metadata": _normalize_metadata(item.get("metadata"), item),
                }
            )
        return out

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
            out = self._snapshot_all()
        except Exception:
            logger.warning("mem0 取全部记忆失败。", exc_info=True)
            return []
        if not include_sensitive:
            out = [
                item
                for item in out
                if not _contains_sensitive(item.get("text") or "")
            ]
        return out

    def _record_protection_event(self, kind, reason, failed_ids=None, failed_texts=None,
                                 rollback_leftover_ids=None, rollback_leftover_texts=None):
        """记一条保护事件（失败关闭的可见留痕）。

        快照失败、恢复失败这些事不能只写日志——日志 BOOS 看不到，
        悄悄跳过一轮他也不知道。记到本地 JSONL 里
        （memory/protection-events.jsonl，gitignore 覆盖，不进仓库），
        面板上有接口能读到（GET /api/protection-events）。
        kind：snapshot_failed（快照两次都失败，本轮没写）、
        restore_unverified（恢复时读库失败，没法核对，本轮新存的已回滚）、
        restore_partial（个别旧记忆没恢复回来，跟它们同一话题的新记忆已回滚）。
        failed_ids 是旧记忆的 id，failed_texts 是它们的原文——面板上
        BOOS 要看得出是哪条记忆，光给 id 他认不出来。文件只在本地，
        和 history.db 一个待遇（里面本来就有全文），不上传、不导出；
        logger 里的警告照旧不带内容。
        rollback_leftover_ids/texts 是回滚删不掉、还留在库里的新记忆
        （id 和原文）：有残留时才记，没有就不写这两个键——保护事件
        不能写成“已回滚”，而实际仍有记录残留。
        记不下来时只记日志、不抛异常——留痕本身坏了不能把聊天搞崩。
        """
        try:
            event = {
                "time": _now_iso(),
                "kind": kind,
                "reason": reason,
                "failed_ids": list(failed_ids or []),
                "failed_texts": list(failed_texts or []),
            }
            if rollback_leftover_ids:
                event["rollback_leftover_ids"] = list(rollback_leftover_ids)
            if rollback_leftover_texts:
                event["rollback_leftover_texts"] = list(rollback_leftover_texts)
            self.protection_events_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.protection_events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("保护事件没记下来。", exc_info=True)

    def list_protection_events(self, limit=50):
        """返回最近的保护事件（给面板看，新的在前，最多 limit 条）。

        文件没有、读不出来、某一行写坏了，都跳过不报错——面板上
        显示“暂无保护记录”，不拦聊天。
        """
        try:
            path = self.protection_events_path
            if not path.exists():
                return []
            events = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(event, dict) and event.get("kind"):
                        events.append(event)
            events.reverse()
            return events[:limit]
        except Exception:
            logger.warning("读保护事件失败。", exc_info=True)
            return []

    # ---------- 账本（事件账本：只追加、不修改，原话逐字存） ----------
    # 账本是家、mem0 只是索引（最终方案，2026-11-13）：先写账本、再写
    # 索引；索引坏了从账本重建；筛子在检索回来之后过滤。下面这几个
    # 帮手只管“记账本 / 读账本状态”，不管业务判断——判断在调用方。

    def _ledger_append(self, event_type, **fields):
        """往账本记一笔（只追加，写完 + 刷盘才算数）。

        写失败只记日志、返回 None，不抛异常——账本是第二道防线，
        不能因为账本写不进去就把聊天搞崩；但“先写账本再写索引”的
        那一步不用这个，用 _ledger_append_or_raise（账本写不进去，
        索引那步也别写）。
        """
        try:
            return _ledger.append_event(
                self.ledger_path,
                {"type": event_type, **fields},
                lock=self._ledger_lock,
            )
        except Exception:
            logger.warning("账本记一笔失败：%s。", event_type, exc_info=True)
            return None

    def _ledger_append_or_raise(self, event_type, seq=None, **fields):
        """往账本记一笔，写失败就抛——先写账本再写索引，账本写不进去，
        索引那步也别写（add_memory_text 用这个；写记忆失败返回 {}，
        不抛到聊天里，见 _add_text_with_metadata 的约定）。
        seq 只有恢复时才传（按备份里的原序号写回来）。"""
        return _ledger.append_event(
            self.ledger_path,
            {"type": event_type, **fields},
            lock=self._ledger_lock,
            seq=seq,
        )

    def ledger_state(self):
        """当前账本折叠出的状态（快照打底 + 快照之后的新事件）。

        账本读坏了返回空状态、不抛异常——筛子没状态就不过滤，
        检索照常跑，绝不能因为账本读不出来就搜不到记忆。"""
        try:
            return _ledger.load_ledger_state(self.ledger_path, self.snapshot_path)
        except Exception:
            logger.warning("读账本状态失败：本次检索不过滤。", exc_info=True)
            return {
                "by_id": {},
                "adds": [],
                "relations": [],
                "key_states": {},
                "id_to_key": {},
                "inferences": {},
                "seq_max": 0,
                "snapshot_seq": None,
            }

    def ledger_verify(self):
        """查账本：序号缺没缺、最后一行写没写完、有没有坏行
        （见 memory/ledger.py verify_ledger）。"""
        return _ledger.verify_ledger(self.ledger_path)

    def ledger_repair(self):
        """修账本：扔掉末尾写一半的残行（见 memory/ledger.py repair_ledger）。"""
        return _ledger.repair_ledger(self.ledger_path)

    # ---------- 图谱推理第一期（2026-11-18） ----------

    def _get_graph(self):
        """拿图（懒建）：快照在就读快照，没有/读坏了就从账本重建。

        图是索引，坏了从账本重建，不拦路；重建失败返回一张空图，
        检索照常走老路。_graph_lock 管懒建 + 写快照，别让两个线程
        同时建两张图。
        """
        with self._graph_lock:
            if self._graph is not None:
                return self._graph
            graph = _graph_mod.MemoryGraph(self.graph_path)
            if not graph.load():
                try:
                    events, _info = _ledger.read_events(self.ledger_path)
                except Exception:
                    events = []
                try:
                    graph.rebuild_from_events(events)
                    # 重建完就把快照写盘（2026-11-19 真机教训：之前
                    # 这里只重建不写盘，快照一直没有，服务每次启动
                    # 都现重建）。写失败只记日志、不拦路——图是索引。
                    try:
                        graph.save()
                    except Exception:
                        logger.warning("图快照写盘失败：下次启动再重建。", exc_info=True)
                except Exception:
                    logger.warning("图从账本重建失败：本次用空图。", exc_info=True)
                    graph = _graph_mod.MemoryGraph(self.graph_path)
            self._graph = graph
            return graph

    def _graph_add_memory(self, key, text, metadata, seq):
        """把一条新记忆接进图（抽实体、连边、判关系、写快照）。

        只在 _ledger_record_add 成功后调（新 add 事件落盘了）。
        实体先走免费的，抽不到再调一次 _judge_json 兜底（flash，
        温度 0）；抽取失败、关系判不出来、快照写不进去都只记日志、
        不抛异常——图是索引，坏了从账本重建，绝不能因为图写不进去
        就把存入搞崩。
        关系只和“连着同一实体”的旧记忆判（图上邻居），不全库扫；
        判出来的写成记忆—记忆边（只在图里，不另写 relate 事件——
        重建时按账本重判一遍，同一对关系不会重复进账本）。
        """
        if not key or not (text or "").strip():
            return
        try:
            graph = self._get_graph()
        except Exception:
            logger.warning("图接新记忆失败：图没建起来。", exc_info=True)
            return
        try:
            entities = extract_entities(text, llm_judge=self._judge_json)
            with self._graph_lock:
                graph.add_memory(
                    key,
                    text,
                    entities,
                    valid_at=(metadata or {}).get("created_at"),
                    source_seq=seq if isinstance(seq, int) else None,
                )
                # 关系：只和连着同一实体的旧记忆判（图上邻居）。
                try:
                    state = self.ledger_state()
                except Exception:
                    state = {}
                adds_by_key = {
                    (record or {}).get("key"): record
                    for record in (state or {}).get("adds") or []
                    if (record or {}).get("key")
                }
                candidates = set()
                for name, _kind in entities:
                    ent_id = _graph_mod.entity_node_id(name)
                    for edge in graph.edges:
                        if edge.get("kind") != _graph_mod.EDGE_MENTIONS:
                            continue
                        if edge.get("invalid_at"):
                            continue
                        if edge.get("from") == ent_id:
                            other = edge.get("to")
                        elif edge.get("to") == ent_id:
                            other = edge.get("from")
                        else:
                            continue
                        if isinstance(other, str) and other.startswith("mem:"):
                            candidates.add(other[4:])
                candidates.discard(key)
                for old_key in sorted(candidates):
                    old_record = adds_by_key.get(old_key)
                    if old_record is None:
                        continue
                    relation = _relations.classify_relation(
                        old_record.get("text") or "",
                        old_record.get("meta") or {},
                        text,
                        metadata or {},
                    )
                    if relation is not None:
                        graph.add_relation(old_key, key, relation, source_seq=seq if isinstance(seq, int) else None)
                try:
                    graph.save()
                except Exception:
                    logger.warning("图快照写盘失败：记忆本身已经进账本了。", exc_info=True)
        except Exception:
            logger.warning("图接新记忆失败：本次只记账本、不记图。", exc_info=True)

    def _graph_merge(self, results, seen_texts, query, top_k, include_history, ledger_state):
        """图检索合并：种子实体 → PPR 一步多跳，两路分数归一化相加。

        就地改 results（按合并分重排、截到 top_k），seen_texts 同步更新。
        图是空的、种子找不着、PPR 没命中时什么都不做，老路结果原样。
        失败只记日志、不抛异常——图是加分项，不是必经之路。
        合并分 = 归一化向量分 + 归一化 PPR 分：向量分是老路语义检索
        的分数（关键词补充的没有，就当 0）；PPR 分是图上跑出来的。
        图上找回来的新记忆按账本原话拼成结果项（source="graph"，
        id 能对上就对上，对不上就是 None，和账本兜底项一个待遇——
        拿来回答问题可以，拿去 update / 纠错不行）。
        """
        try:
            graph = self._get_graph()
        except Exception:
            logger.warning("图检索失败：图没建起来。", exc_info=True)
            return
        if not graph.nodes or not graph.edges:
            return
        try:
            seeds = extract_query_entities(query, llm_judge=self._judge_json)
        except Exception:
            logger.warning("图检索失败：种子实体没抽出来。", exc_info=True)
            return
        seeds = [
            name for name in (seeds or [])
            if _graph_mod.entity_node_id(name) in graph.nodes
        ]
        if not seeds:
            return
        try:
            key_states = (ledger_state or {}).get("key_states") or {}
            # 失效时间从账本事件里拿（supersede / delete 那笔的 ts），
            # 拿不到就按已有的 invalid_at 留着——sync_validity 只看
            # invalid_at 有没有值，时间本身不影响 PPR。
            invalid_ts = {}
            try:
                events, _info = _ledger.read_events(self.ledger_path)
                for event in events or []:
                    if event.get("type") in (
                        _ledger.LEDGER_EVENT_SUPERSEDE,
                        _ledger.LEDGER_EVENT_DELETE,
                    ) and event.get("key"):
                        invalid_ts[event["key"]] = event.get("ts")
            except Exception:
                invalid_ts = {}
            with self._graph_lock:
                graph.sync_validity(key_states, invalid_ts)
                # 记忆种子的分：老路语义结果的归一化分 × 0.05。
                vector_scores = {}
                for item in results:
                    if item.get("source") != "semantic":
                        continue
                    score = item.get("score")
                    if score is None:
                        continue
                    item_key = _ledger.ledger_event_key(
                        item.get("text") or "",
                        (item.get("metadata") or {}).get("created_at"),
                    )
                    vector_scores[item_key] = float(score)
                max_vector = max(vector_scores.values()) if vector_scores else 0.0
                memory_seeds = (
                    {key: score / max_vector for key, score in vector_scores.items()}
                    if max_vector > 0
                    else {}
                )
                ppr_scores = graph.ppr(
                    seeds, memory_seeds, include_history=include_history
                )
            if not ppr_scores:
                return
            max_ppr = max(ppr_scores.values())
            if max_ppr <= 0:
                return
            norm_ppr = {key: score / max_ppr for key, score in ppr_scores.items()}
            norm_vector = (
                {key: score / max_vector for key, score in vector_scores.items()}
                if max_vector > 0
                else {}
            )
            # 账本原话档案（key → text/meta），图上找回来的新记忆靠它
            # 拼成结果项。id 固定是 None，和账本兜底项一个待遇：它是
            # 账本档案里的原话，拿来回答问题可以，拿去 update / 纠错 /
            # 打标记不行——账本里的 id_to_key 可能过期（索引换过、条目
            # 删过），按它给 id 会让纠正拿着一个不存在的 id 去改库
            # （2026-11-18 测出来的：空库纠正错拿了 fake-1 去 update）。
            adds_by_key = {
                (record or {}).get("key"): record
                for record in (ledger_state or {}).get("adds") or []
                if (record or {}).get("key")
            }
            now_dt = datetime.now(timezone.utc)
            by_key = {}
            for item in results:
                item_key = _ledger.ledger_event_key(
                    item.get("text") or "",
                    (item.get("metadata") or {}).get("created_at"),
                )
                by_key[item_key] = item
            merged = []
            for item in results:
                item_key = _ledger.ledger_event_key(
                    item.get("text") or "",
                    (item.get("metadata") or {}).get("created_at"),
                )
                combined = norm_vector.get(item_key, 0.0) + norm_ppr.get(item_key, 0.0)
                item["graph_score"] = norm_ppr.get(item_key, 0.0)
                item["combined_score"] = combined
                merged.append(item)
            for key, ppr_norm in norm_ppr.items():
                if key in by_key:
                    continue
                record = adds_by_key.get(key)
                if record is None:
                    continue
                text = record.get("text") or ""
                if not text or text in seen_texts:
                    continue
                if _contains_sensitive(text):
                    continue
                hit_state = key_states.get(key)
                if not include_history and hit_state in (
                    _ledger.LEDGER_STATE_SUPERSEDED,
                    _ledger.LEDGER_STATE_DELETED,
                ):
                    continue
                meta = _normalize_metadata(record.get("meta") or {})
                merged.append(
                    {
                        "id": None,
                        "text": text,
                        "score": None,
                        "source": "graph",
                        "metadata": meta,
                        "confirmed": meta["confirmed"],
                        "decay_weight": decay_weight(
                            meta["decay_kind"],
                            meta["last_used_at"],
                            now_dt,
                            is_core=_is_core_memory(meta),
                        ),
                        "graph_score": ppr_norm,
                        "combined_score": ppr_norm,
                        "is_history": bool(
                            include_history
                            and hit_state
                            in (
                                _ledger.LEDGER_STATE_SUPERSEDED,
                                _ledger.LEDGER_STATE_DELETED,
                            )
                        ),
                        "history_state": hit_state
                        if include_history
                        and hit_state
                        in (
                            _ledger.LEDGER_STATE_SUPERSEDED,
                            _ledger.LEDGER_STATE_DELETED,
                        )
                        else None,
                    }
                )
                seen_texts.add(text)
            merged.sort(
                key=lambda item: (item.get("combined_score") or 0)
                * (item.get("decay_weight") or 0),
                reverse=True,
            )
            results[:] = merged[:top_k]
        except Exception:
            logger.warning("图检索合并失败：本次只用老路结果。", exc_info=True)
            return

    def relate_memories(self, memory_id, related_id, relation, note=""):
        """记一条记忆关系（只留四种：取代 / 补充 / 冲突 / 关联）。

        关系存在账本里（relate 事件），不另建表。relation 不在四种
        里时抛 ValueError——账本里不许出现没名没姓的关系。
        返回记好的事件，写失败返回 None。"""
        if relation not in _relations.RELATION_TYPES:
            raise ValueError(f"记忆关系不对：{relation!r}。")
        return self._ledger_append(
            _ledger.LEDGER_EVENT_RELATE,
            memory_id=memory_id,
            related_id=related_id,
            relation=relation,
            note=note or "",
        )

    def reconcile_ledger_index(self):
        """启动对序号补缺：账本里有、索引里没有的，按账本重建索引。

        对账按 key 认（原话 + created_at 的哈希，见 ledger_event_key）：
        账本里有、索引里没有的 key，就是“账本写了、索引没写上”（写
        索引那步崩了、或索引坏过），按账本里的原话 + metadata 原样
        写回索引。索引里有、账本里没有的不动（提取路径的老记忆本来
        就没有账本记录，不算缺）。返回 {"reconciled", "missing",
        "failed"}。读账本失败时返回 reconciled=0，不抛异常——对账
        是启动时的兜底，不能因为对不上就把服务搞崩。
        注意：重建写回走 _add_text_with_metadata，含敏感信息的原文
        会被用时拦截那道防线再删掉（和正常写入同一道防线），这种
        记在 failed 里，不算“补上了”。
        """
        try:
            state = self.ledger_state()
            adds = state.get("adds") or []
            if not adds:
                return {"reconciled": 0, "missing": [], "failed": []}
            # 只补“当前”的原话：账本里标了 superseded / deleted 的
            # 不补——纠正过的旧说法、删了的，补回索引就是复活（原地
            # 纠正后重启会把错的那句再存一条；删了的重启又回来）。
            # 睡着的（dormant）照常补：它丢了只是索引坏了，补回来时
            # 按账本里的原 metadata（last_used_at 还是旧的），下次
            # 月扫还会把它标回 dormant，不会“洗白”成常用记忆。
            # 没标过的（老事件）默认当 current。
            key_states = state.get("key_states") or {}
            adds = [
                entry
                for entry in adds
                if key_states.get(entry.get("key"))
                not in (
                    _ledger.LEDGER_STATE_SUPERSEDED,
                    _ledger.LEDGER_STATE_DELETED,
                )
            ]
            existing_keys = set()
            for item in self.get_all(include_sensitive=True):
                meta = (item or {}).get("metadata") or {}
                existing_keys.add(
                    _ledger.ledger_event_key(
                        item.get("text") or "", meta.get("created_at")
                    )
                )
            missing = [
                entry for entry in adds if entry.get("key") not in existing_keys
            ]
            reconciled = []
            failed = []
            for entry in missing:
                text = entry.get("text") or ""
                meta = dict(entry.get("meta") or {})
                if not text:
                    failed.append(
                        {"key": entry.get("key"), "reason": "账本里这条没原话"}
                    )
                    continue
                try:
                    result = self._add_text_with_metadata(text, meta)
                    if result and (result.get("results") or []):
                        reconciled.append(entry.get("key"))
                    else:
                        failed.append(
                            {"key": entry.get("key"), "reason": "写索引没返回结果"}
                        )
                except Exception as exc:
                    failed.append({"key": entry.get("key"), "reason": str(exc)})
            return {
                "reconciled": len(reconciled),
                "missing": [entry.get("key") for entry in missing],
                "failed": failed,
            }
        except Exception:
            logger.warning("账本对账失败：本次不对索引补缺。", exc_info=True)
            return {
                "reconciled": 0,
                "missing": [],
                "failed": [{"reason": "对账读账本失败"}],
            }

    def backup_memories(self, backup_dir=None):
        """备份：账本全部事件 + 索引全部记忆（原话 + 原始 metadata）+
        快照（压过账本才有），带校验和。返回 {"backup_path", "counts",
        "snapshot_error"}；写失败抛 OSError。备份是显式操作，不在
        启动时自动跑——写备份之前要先想清楚放哪、留几份。
        快照读坏了时备份照常写，但结果里带 snapshot_error，验备份
        时会判不过（缺了快照的备份是不完整的，不能当“能恢复”来认）。"""
        events, _info = _ledger.read_events(self.ledger_path)
        items = []
        for item in self.get_all(include_sensitive=True):
            items.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text") or "",
                    "metadata": self._raw_metadata_for(item.get("id")) or {},
                }
            )
        snapshot = None
        snapshot_error = None
        try:
            snapshot = _ledger.read_snapshot(self.snapshot_path)
        except ValueError as exc:
            snapshot_error = str(exc)
            logger.warning("备份时快照读坏了：%s。", exc, exc_info=True)
        backup = _backup.build_backup(
            items, events, snapshot=snapshot, snapshot_error=snapshot_error
        )
        directory = Path(backup_dir) if backup_dir else self.backup_dir
        path = directory / _backup.backup_filename()
        _backup.write_backup(path, backup)
        return {
            "backup_path": str(path),
            "counts": backup["counts"],
            "snapshot_error": snapshot_error,
        }

    def verify_backup(self, backup_path):
        """验一份备份（见 memory/backup.py verify_backup）。"""
        return _backup.verify_backup(backup_path)

    def restore_from_backup(self, backup_path):
        """真恢复：先验备份，再把缺的东西补回来（不是回滚）。

        验不过（文件读不出、条数对不上、校验和对不上、快照坏了）
        就抛 ValueError，一个也不恢复。验过了：
          1. 快照：备份里的快照比现有的新（或现有没有）才写回去；
             现有的更新时不动（恢复是“补齐”，不是“回滚”）。
          2. 账本事件：按原 seq 排好、缺的才追加（已有的跳过）；
             snapshot 标记事件也按原 seq 写回去（快照的底子靠它认）。
          3. 索引记忆：按 key 认，已有的跳过，缺的按备份里的原话 +
             原始 metadata 原样写回（created_at、decay 字段都原样，
             不重新编时间）。
        含敏感信息的原文写回时会被用时拦截那道防线再删掉（和正常
        写入同一道防线），记在 skipped 里。返回 {"restored_items",
        "skipped_items", "restored_events", "skipped_events",
        "snapshot_restored", "backup_path"}。
        """
        check = _backup.verify_backup(backup_path)
        if not check["ok"]:
            raise ValueError(
                f"备份验不过，不恢复：{'; '.join(check['errors'])}。"
            )
        payload = _backup.read_backup(backup_path)
        snapshot_restored = False
        backup_snapshot = payload.get("snapshot")
        if isinstance(backup_snapshot, dict):
            current_snapshot = None
            try:
                current_snapshot = _ledger.read_snapshot(self.snapshot_path)
            except ValueError:
                current_snapshot = None
            current_seq = (
                current_snapshot.get("snapshot_seq")
                if isinstance(current_snapshot, dict)
                else None
            )
            backup_seq = backup_snapshot.get("snapshot_seq")
            if current_seq is None or (
                isinstance(backup_seq, int) and backup_seq > current_seq
            ):
                _ledger.write_snapshot_raw(self.snapshot_path, backup_snapshot)
                snapshot_restored = True
        existing_events, _info = _ledger.read_events(self.ledger_path)
        existing_seqs = {
            e.get("seq") for e in existing_events if isinstance(e.get("seq"), int)
        }
        restored_events = 0
        skipped_events = 0
        for event in sorted(
            payload.get("ledger_events") or [],
            key=lambda e: e.get("seq") if isinstance(e.get("seq"), int) else 0,
        ):
            seq = event.get("seq")
            if seq in existing_seqs:
                skipped_events += 1
                continue
            record = {k: v for k, v in event.items() if k != "seq"}
            self._ledger_append_or_raise(event.get("type"), seq=seq, **record)
            restored_events += 1
        existing_keys = set()
        for item in self.get_all(include_sensitive=True):
            meta = (item or {}).get("metadata") or {}
            existing_keys.add(
                _ledger.ledger_event_key(
                    item.get("text") or "", meta.get("created_at")
                )
            )
        restored_items = []
        skipped_items = []
        for item in payload.get("items") or []:
            text = (item or {}).get("text") or ""
            meta = dict((item or {}).get("metadata") or {})
            key = _ledger.ledger_event_key(text, meta.get("created_at"))
            if key in existing_keys:
                skipped_items.append({"id": item.get("id"), "reason": "索引里已有"})
                continue
            if not text:
                skipped_items.append({"id": item.get("id"), "reason": "备份里这条没原话"})
                continue
            result = self._add_text_with_metadata(text, meta)
            if result and (result.get("results") or []):
                restored_items.append(
                    ((result.get("results") or [{}])[0] or {}).get("id")
                )
                existing_keys.add(key)
            else:
                skipped_items.append(
                    {"id": item.get("id"), "reason": "写索引没返回结果"}
                )
        return {
            "restored_items": restored_items,
            "skipped_items": skipped_items,
            "restored_events": restored_events,
            "skipped_events": skipped_events,
            "snapshot_restored": snapshot_restored,
            "backup_path": str(backup_path),
        }

    def compact_ledger(self):
        """一年压缩一次：账本压成快照（原话都在快照里），账本截断
        到只剩一行 snapshot 标记（见 memory/ledger.py compact_ledger）。
        显式调用，不自动跑；压完就不用删了——快照就是档案，真删
        记忆还是等 BOOS 亲口说，压缩本身不删任何记忆。"""
        return _ledger.compact_ledger(self.ledger_path, self.snapshot_path)

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
        33 号：写入口显式给时间——metadata 里没有非空 created_at
        字符串时，这里补一个新鲜的单调时间（_now_iso），不改调用方
        传进来的字典（复制一份再补，覆盖守卫后面还要用原来那份）。
        正常调用方都走 _provenance，本来就有时间，这只是兜底。
        """
        if isinstance(metadata, dict):
            created = metadata.get("created_at")
            if not (isinstance(created, str) and created):
                metadata = dict(metadata)
                metadata["created_at"] = _now_iso()
                metadata.setdefault("created_at_source", "written")
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

    def _extract_and_save_turn(self, safe_messages, extract_since=None, turn_id=None):
        """正常提取保存（add_turn 的老路子，只做这一件事）。

        调 mem0.add 提取本轮对话、去重、存入；提取出的记忆里万一
        还带着敏感信息，_drop_sensitive_results 会再删掉。
        再过一道覆盖守卫（_apply_coverage_guard）：刚提取出的推断记忆
        别把 BOOS 确认过的事实顶掉——守卫要拿提取前的库快照找邻居，
        所以快照在调 mem0.add 之前拍（快照里没有本次新存的，就不会
        把新记忆和自己/同一批的兄弟认成邻居）；快照还有第二个用处：
        万一 add 在里面动了旧记忆，守卫按快照原样恢复（见
        _restore_snapshot_memories）。
        extract_since（2026-11-16 真机测试，拉面）：源轮次说话的
        时间戳（待定区里那轮的 created_at）。提取出的原文如果在
        extract_since 之后被删过，_drop_deleted_text_results 会把它
        从库里清掉、不进账本——删完 10 秒提取又记回来，就是这么来的
        （待定轮的提取跑在删除那轮的 add_turn 里，删除时库里还没它，
        去重看不见它）。删在 extract_since 之前的是 BOOS 重新说起，
        不拦。直接提取（纠错没找到时的那一处）不传 extract_since，
        那一轮是当场纠错、没经过删除，不拦。
        失败关闭：快照是保护的前提——快照读不出来，就不知道旧记忆
        长什么样，写进去就是没保护硬写。所以快照先重试一次（很多
        时候就是抖一下，重试之间不睡、保持简单），还失败就不调
        mem0.add，本轮不写，记一条保护事件、结果里带上
        protection_failed 标记（调用方和面板能读到）。代价只是这轮
        的事实这次没存进长期记忆，对话摘要里还留着，还能再说——
        比悄悄顶掉已确认的事实强得多。
        失败时返回带 protection_failed（快照失败）或 extraction_failed
        （mem0.add 失败）的字典，绝不抛异常——存记忆失败不能把聊天
        回答搞崩；但失败和“没什么可存”要分得清（52 号），不然调用方
        以为本轮没什么可存、其实是事实丢了。
        """
        before = None
        for _attempt in range(2):
            try:
                before = self._snapshot_all()
                break
            except Exception:
                before = None
        if before is None:
            logger.warning("提取前快照两次都失败：本轮不写记忆。", exc_info=True)
            self._record_protection_event(
                "snapshot_failed", "提取前快照两次都失败，本轮没有存记忆。"
            )
            return {
                "protection_failed": True,
                "protection_reason": "snapshot_failed",
                "results": [],
            }
        metadata = _provenance(
            SOURCE_INFERRED, CONFIDENCE_INFERRED, False, turn_id=turn_id
        )
        try:
            result = self._mem0_add(
                safe_messages,
                self.user_id,
                True,
                metadata,
            )
        except Exception:
            # 52 号：提取失败和“没什么可存”必须分得清——以前这里也返回 {}，
            # 调用方以为本轮没什么可存，其实是 mem0.add 炸了、本轮的事实
            # 丢了。带上 extraction_failed，让提交那层能重试这一轮，
            # 而不是把它记成已处理、再也不管。
            logger.warning("mem0 自动存记忆失败：本轮对话未存入长期记忆。", exc_info=True)
            return {
                "results": [],
                "extraction_failed": True,
                "extraction_reason": "mem0_add_failed",
            }
        result = self._drop_sensitive_results(result or {})
        # 2026-11-16 真机测试（问题 4）：管记忆的动作复述成事实的，
        # 在这里拦掉——动作本身已经记在账本里了，不用再存一条。
        result = self._drop_management_action_results(result)
        # 揣摩守卫（2026-11-16）：本轮问句里的猜，别让提取存进库里
        # （见 _drop_pending_inference_results）——只拦本轮问过的那句。
        result = self._drop_pending_inference_results(result, safe_messages)
        # 2026-11-16 真机测试（拉面，根子 1）：刚删的原文别再记回来。
        # 放在去重和账本补记之前——去重看不见已删的（快照是删之后的），
        # 账本补记只对留下的新条记，拦在这里就不会多一笔 add。
        if extract_since is not None:
            result = self._drop_deleted_text_results(result, extract_since)
        result = self._apply_coverage_guard(result, before, metadata)
        # 43 号价值关卡先走（一次判全场，低价值的先清掉），再 46 号去重
        # （对照面是提取前的库 + 本轮已留下的新记忆）。
        result = self._value_gate_extraction_results(result)
        result = self._dedupe_extraction_results(
            result, before, turn_id=turn_id, batch_metadata=metadata
        )
        # 27 号：提取出的新记忆里带金额的，把结构化金额字段贴上。
        # 提取时 metadata 是整批共用的、不知道每条写了什么，只能事后
        # 按文本贴；放在去重之后（被删掉的就不用贴了）。
        result = self._attach_money_metadata(result)
        # 补丁 3 语气 + 补丁 4 对原话：对留下的每条新记忆——
        # 本地筛出玩笑/测试/假设的直接打标；筛不出来的排队等大模型
        # 后台判；数字/代码跟本轮原话对不上的标待复核 + 对原话标记。
        # 放在最后做：前面几步（守卫、去重、贴金额字段）都可能改
        # metadata，语气标和对原话标最后写才不会被盖掉。
        self._tag_extraction_results_tone_and_grounding(result, safe_messages)
        # 提取路径补记账本：mem0 自己拆句，事前不知道文本，只能事后
        # 对留下的新条补记 add 事件（UPDATE 的是同槽位新值，
        # _supersede_money_memory 里已经记了 supersede，不重复记）。
        self._ledger_extraction_adds(result, metadata)
        return result

    def _tag_extraction_results_tone_and_grounding(self, result, safe_messages):
        """提取收尾：语气打标 + 对原话检查（补丁 3、补丁 4 第一层）。
        只对留下的新条做（results 里的每条，有 id 和文本的都算，
        包括同槽位新值 UPDATE 的那条——新说法本身也是这轮的话）。
        语气：本地筛出玩笑/测试/假设的直接打标（tone + 证据 +
        source=local）；筛不出来的排进 tone_pending.jsonl，等大模型
        后台判。拿不准不猜，排队就是“还没判”，不是“认真”。
        对原话：数字/代码跟本轮用户原话对不上的，标 needs_review +
        grounding_flag（只标、不删，月自检“最可能错的 5 条”会捞它）；
        记进 results["grounding_flagged"]，调用方看得见。原话为空
        （这轮没用户说话）时不查——没出处就是没法对，不标。
        中文数字先转阿拉伯数字再比（“五块”对“5 块”算对上）。
        失败只记日志、不拦路：打标是对原话的辅助，不能把提取搞崩。"""
        items = (result or {}).get("results") or []
        if not items:
            return
        source_text = "\n".join(
            (msg or {}).get("content") or ""
            for msg in safe_messages or []
            if (msg or {}).get("role") == "user"
        )
        source_norm = _cn_numerals_to_digits(source_text) if source_text.strip() else ""
        grounded_flagged = []
        for item in items:
            mem_id = (item or {}).get("id")
            mem_text = (item or {}).get("memory") or ""
            if not mem_id or not mem_text:
                continue
            tone, evidence = self._local_tone_for(mem_text)
            if tone is not None:
                if self._tag_memory_tone(mem_id, tone, evidence, "local"):
                    item["tone"] = tone
            else:
                self._tone_queue_append(mem_id, mem_text)
            if not source_norm:
                continue
            try:
                check = _content_check.grounding_check(
                    _cn_numerals_to_digits(mem_text), source_norm
                )
            except Exception:
                logger.warning("对原话检查失败：这条先不标。", exc_info=True)
                continue
            if not check.get("ok"):
                self._mark_memory_review(mem_id, needs_review=True)
                self._mark_memory_grounding(mem_id)
                grounded_flagged.append(mem_id)
        if grounded_flagged:
            result["grounding_flagged"] = grounded_flagged

    def _deleted_text_times(self, cutoff_epoch):
        """cutoff 之后被删掉的原文（归一化原文 → 删的时间戳，epoch 秒）。

        两处来源并在一起：
        - 内存里刚删的（_just_deleted_texts）：账本写失败时也有据；
        - 账本：按 key 折叠 add/delete/revive，取最后一次 delete 的
          时间——revive 过的就不算删了，add 的原话用来认是哪条。
        只认“删在源轮次说话之后”的：源轮次之前就删了的，是 BOOS
        重新说起，不拦（2026-11-16 拉面：删 01:18:42、源轮次
        01:18 之前，拦；周三重新说、周一删的，不拦）。
        账本读不出来时只用内存那份，不抛异常——拦不住也别把提取搞崩。
        """
        out = {}
        try:
            cutoff = float(cutoff_epoch)
        except (TypeError, ValueError):
            return out
        for norm, when in (self._just_deleted_texts or {}).items():
            try:
                when_f = float(when)
            except (TypeError, ValueError):
                continue
            if when_f > cutoff:
                out[norm] = when_f
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            return out
        key_text = {}
        key_deleted_at = {}
        for event in events or []:
            etype = (event or {}).get("type")
            key = (event or {}).get("key")
            if not key:
                continue
            if etype == _ledger.LEDGER_EVENT_ADD:
                key_text.setdefault(key, (event or {}).get("text") or "")
            elif etype == _ledger.LEDGER_EVENT_DELETE:
                parsed = _parse_created_at((event or {}).get("ts"))
                if parsed is not None:
                    key_deleted_at[key] = parsed.timestamp()
            elif etype == _ledger.LEDGER_EVENT_REVIVE:
                key_deleted_at.pop(key, None)
        for key, when in key_deleted_at.items():
            if when <= cutoff:
                continue
            text = key_text.get(key) or ""
            if not text:
                continue
            norm = _normalize_memory_text(text)
            if norm and norm not in out:
                out[norm] = when
        return out

    def _drop_deleted_text_results(self, result, cutoff_epoch):
        """从提取结果里清掉刚删的原文（2026-11-16 真机测试，拉面）。

        删掉的原文不再自己记回来，除非 BOOS 重新说起——重新说起
        的轮次说话时间在删之后，cutoff 一过就不拦（见
        _deleted_text_times）。删的办法和 _drop_sensitive_results /
        _dedupe_extraction_results 同一套：先从向量库清掉、再读一遍
        核对，清不掉的留着（results 里原样返回，调用方看得见），
        清掉的记在 results["deleted_text_dropped"] 里。
        放在账本补记之前调——拦在这里，账本就不会多一笔 add。
        """
        items = (result or {}).get("results") or []
        if not items:
            return result
        deleted_after = self._deleted_text_times(cutoff_epoch)
        if not deleted_after:
            return result
        dropped = []
        kept = []
        for item in items:
            mem_text = (item or {}).get("memory") or ""
            if mem_text and _normalize_memory_text(mem_text) in deleted_after:
                dropped.append(item)
            else:
                kept.append(item)
        if not dropped:
            return result
        survivors = self._rollback_new_memories(dropped)
        survivor_ids = {(item or {}).get("id") for item in survivors}
        actually_dropped = [
            item for item in dropped if (item or {}).get("id") not in survivor_ids
        ]
        result["results"] = kept + survivors
        result["deleted_text_dropped"] = len(actually_dropped)
        if survivors:
            logger.warning(
                "提取复活拦截：%d 条刚删的记忆没清掉（仍在库里）。",
                len(survivors),
            )
        return result

    def _ledger_extraction_adds(self, result, batch_metadata):
        """提取路径补记账本（见 _extract_and_save_turn）。

        只记 event == "ADD" 的新条；同一批共用一份 metadata，
        key 按“原话 + 这批的 created_at”算。同一条重复记时
        （同一轮被提交两遍）按 key 去重，不记两遍。
        记失败不拦路：记忆已经在索引里了，账本缺这一笔只是档案
        不全，不丢事实。
        """
        try:
            items = (result or {}).get("results") or []
            if not items:
                return
            created_at = (batch_metadata or {}).get("created_at")
            try:
                state = self.ledger_state()
                known_keys = {
                    entry.get("key") for entry in state.get("adds") or []
                }
            except Exception:
                known_keys = set()
            for item in items:
                if (item or {}).get("event") != "ADD":
                    continue
                text = (item or {}).get("memory") or ""
                if not text:
                    continue
                key = _ledger.ledger_event_key(text, created_at)
                if key in known_keys:
                    continue
                known_keys.add(key)
                self._ledger_append(
                    _ledger.LEDGER_EVENT_ADD,
                    key=key,
                    text=text,
                    meta={
                        "source": SOURCE_INFERRED,
                        "confidence": CONFIDENCE_INFERRED,
                        "user_confirmed": False,
                        "created_at": created_at,
                        "turn_id": (batch_metadata or {}).get("turn_id"),
                    },
                )
        except Exception:
            logger.warning("提取补记账本失败：索引里的记忆还在。", exc_info=True)

    def _attach_money_metadata(self, results):
        """27 号：给提取出的新记忆贴金额结构化字段。

        提取出的每条新记忆，看文本里有没有金额，有就把 money 字段
        （金额/币种/周期/对象，见 _extract_money_metadata）update 进
        它的 metadata——mem0 的 update 是合并，不动原有字段（时间、
        来源、确认状态都还在）。贴不上（update 失败）只记日志、
        不拦聊天：字段是辅助，文本本身还在，回答时走兜底句。
        """
        items = (results or {}).get("results") or []
        for item in items:
            mem_id = (item or {}).get("id")
            text = (item or {}).get("memory") or ""
            money = _extract_money_metadata(text)
            if not mem_id or not money:
                continue
            try:
                # 扁平四个标量字段贴上去（见 _flatten_money）：Chroma
                # 不接受 dict 值的 metadata，直接传 {"money": {...}}
                # 会炸掉整条 update。
                # 先拿原始整份 metadata 再合并着写：只传 money 字段时，
                # 替换式 update（测试用的 FakeMemory）会把之前打上的
                # 标记（待确认、语气）全丢掉；真实 mem0 是合并式、不会
                # 丢，但两边都按整份传才稳（见 _mark_memory_review 注释）。
                raw = self._raw_metadata_for(mem_id)
                if raw is None:
                    logger.warning("金额字段：这条记忆读不出来，没贴上。")
                    continue
                raw.update(_flatten_money(money))
                self._memory.update(memory_id=mem_id, metadata=raw)
            except TypeError:
                logger.warning("金额字段：底层 update 不支持 metadata 参数，本条没贴上。")
            except Exception:
                logger.warning("金额字段：贴结构化金额失败。", exc_info=True)
        return results

    # ---------- Plan C：待定区提交 + 同槽位 UPDATE + 身份种子 ----------

    def _arm_pending_timer(self, turn_id):
        """给待定区停车后（重）启闲置定时器：到点仍是同一轮就提交。

        每来一个新 add_turn 就取消旧定时器重设——撤回窗口从“等到下一轮”
        变成“等闲置这么久”，“不不不”在窗口内仍能丢掉待定区；窗口过了
        没人接话，定时器自己提交，保证最后一轮也能落库（2026-11-20
        游泳事故）。定时器是守护线程，不拦进程退出；起不来只记日志、
        不拦提交（退回成“只等下一轮”的老行为）。
        """
        self._cancel_pending_timer()
        if not turn_id:
            return
        try:
            timer = threading.Timer(
                self._pending_idle_seconds, self._on_pending_idle, args=(turn_id,)
            )
            timer.daemon = True
            timer.start()
            self._pending_timer = timer
        except Exception:
            logger.warning("待定区闲置定时器没起来：本轮要等下一轮或关机才提交。", exc_info=True)

    def _cancel_pending_timer(self):
        timer = getattr(self, "_pending_timer", None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
        self._pending_timer = None

    def _on_pending_idle(self, turn_id):
        """闲置定时器到点：待定区还是同一轮时提交它。"""
        with _WRITE_LOCK:
            pending = self._pending_turn
            if pending is None or pending.get("turn_id") != turn_id:
                return
            try:
                self._commit_pending_turn()
            except Exception:
                logger.warning("闲置提交待定记忆失败：等下一轮或关机时再试。", exc_info=True)

    def ensure_outbox_started(self):
        """启动后台 outbox（服务器启动时调，别等第一次聊天才懒建）。

        启动时顺带捞回盘上没存完的轮次（recover_pending）：重启后就算
        没人马上聊天，落盘的轮次也能被消费、进待定区、再由闲置定时器
        提交（2026-11-20 游泳事故：重启后没人接话，轮次就没人管）。
        """
        if self._outbox is None:
            self._outbox = MemoryOutbox(self)
            self._outbox.start()
        return self._outbox

    def _drop_committed_outbox_turn(self, pending, out=None):
        """待定轮真正落库后：删掉它的 outbox 落盘文件，回执改成已提交。

        只在提交成功（没失败）时调——失败的留着文件、等下一轮重试。
        _commit_pending_turn 里调；outbox 不存在（直接调 add_turn 的
        测试）时什么都不做。
        """
        tid = (pending or {}).get("turn_id")
        if not tid or self._outbox is None:
            return
        # 删盘在 mark_committed 里一起做（落库了才不用留着）。
        try:
            self._outbox.mark_committed(tid, out)
        except Exception:
            logger.warning("提交待定记忆后改回执失败：turn_id=%s。", tid, exc_info=True)

    def _commit_pending_turn(self):
        """提交待定区里上一轮对话（Plan C 闸门 3）。

        只在写锁里调（add_turn / flush_pending 都持有锁）。
        顺序：先把待定轮里的身份陈述按槽位 upsert（新值盖旧值），
        再把剩下的、没提交过的 user 句子过闸门 1 过滤、调提取存入。
        提交过的 user 消息指纹记进 _committed_user_keys——add_turn
        每次收到最近 6 条消息（约 3 轮的滑动窗口），下次提交同一批
        消息时，旧轮次的 user 消息不再重复提。
        指纹只在提取成功（或没什么可提）后才记：提取失败时不记，
        失败的那几句原样退回给调用方（failed_messages），下一轮
        提交时再试一次——以前指纹先记、提取后跑，提取一炸，那几句
        就被记成“已处理”、再也不会回来（52 号）。
        返回提交结果 {"results": [...], "identity": [...]}，
        没提交任何东西时返回 {}；提取/快照失败时多带
        extraction_failed / protection_failed 和 failed_messages
        （没存成的那几句 user 原话）。
        """
        pending = self._pending_turn
        self._pending_turn = None
        if pending is None:
            return {}
        # 本轮正在提交，闲置定时器不用再等它（新轮次来了会重设）。
        self._cancel_pending_timer()
        identity_outcomes = []
        for item in pending.get("identity") or []:
            outcome = self._upsert_slot_memory(
                item["slot"], item["value"], item["text"]
            )
            identity_outcomes.append(outcome)
        # 提取：只留没提交过的 user 消息，及其后面跟着的那条
        # assistant 回复（当提取上下文）；指纹先别记——提取成功
        # 后再记（见下面），失败时不记、留待重试。
        messages = pending.get("messages") or []
        fresh = []
        fresh_keys = []
        for i, message in enumerate(messages):
            if (message or {}).get("role") != "user":
                continue
            content = (message or {}).get("content") or ""
            key = content.strip()
            if key in self._committed_user_keys:
                continue
            fresh_keys.append(key)
            fresh.append({"role": "user", "content": content})
            if i + 1 < len(messages) and (messages[i + 1] or {}).get(
                "role"
            ) == "assistant":
                fresh.append(messages[i + 1])
        results = []
        protection_failed = False
        extraction_failed = False
        grounding_flagged = []
        extracted = {}
        filtered = _filter_extraction_messages(fresh)
        if filtered:
            # extract_since：待定轮说话的时间——这轮之后删掉的原文，
            # 提取时拦掉，不再记回来（2026-11-16 拉面）。
            extracted = self._extract_and_save_turn(
                filtered,
                extract_since=pending.get("created_at"),
                turn_id=pending.get("turn_id"),
            )
            results = (extracted or {}).get("results") or []
            # 保护失败的标记要往上传：本轮没写成、或回滚过，调用方
            # （add_turn / 面板）得知道，不能只留在提取那一层。
            # 用共用的 _carry_protection_fields 一次带过去——以前这里
            # 手写一个键一个键地传，残留原文就是这么漏掉的。
            protection_failed = bool((extracted or {}).get("protection_failed"))
            extraction_failed = bool((extracted or {}).get("extraction_failed"))
            # 补丁 4：对原话标出来的那几条也往上传——调用方得知道这轮
            # 有记忆跟原话对不上（只标待复核、不拦路），不然只留在
            # 提取那一层，谁也看不见。
            grounding_flagged = (extracted or {}).get("grounding_flagged") or []
        failed = protection_failed or extraction_failed
        if not failed:
            # 提取成功（或过滤后什么都不剩，本来就没什么可提）：指纹记下来，
            # 下次提交同一批消息时不再重复处理。
            for key in fresh_keys:
                if key and key not in self._committed_user_keys:
                    self._committed_user_keys.append(key)
        if not results and not identity_outcomes and not protection_failed and not extraction_failed:
            self._drop_committed_outbox_turn(pending)
            return {}
        out = {"results": results, "identity": identity_outcomes}
        if grounding_flagged:
            out["grounding_flagged"] = grounding_flagged
        if protection_failed:
            _carry_protection_fields(extracted, out)
        if extraction_failed:
            out["extraction_failed"] = True
            out["extraction_reason"] = (extracted or {}).get(
                "extraction_reason", "mem0_add_failed"
            )
        if failed:
            # 没存成的那几句 user 原话退回给调用方，下一轮再试；
            # 身份 upsert 已经成功的（identity_outcomes）不退回——
            # upsert 按槽位幂等，重试也不会重复存。
            out["failed_messages"] = [
                (m or {}).get("content") or "" for m in fresh
                if (m or {}).get("role") == "user"
            ]
        else:
            self._drop_committed_outbox_turn(pending, out)
        return out

    def flush_pending(self):
        """立刻提交待定区（Plan C）。

        正常对话里待定轮由下一轮 add_turn 提交；这个方法给两处用：
        测试里主动提交、服务停之前把最后一轮落库（否则最后一轮
        永远待在内存里）。返回提交结果，没待定内容时返回 {}。
        26 号：先等后台 outbox 里的轮次都存完（它们会把待定区里
        上一轮的提交掉），再把最后一轮的待定区当场提交。
        """
        if self._outbox is not None:
            self._outbox.flush()
        with _WRITE_LOCK:
            return self._commit_pending_turn()

    def save_turn(self, messages, turn_id=None):
        """后台存一轮记忆（26 号）：先回答、后台再存。

        正常轮次：把本轮消息放进 outbox 就返回
        （{"pending": True, "results": [], "queued": True, "turn_id": ...}），
        不等 add_turn 跑完；后台线程按 turn_id 顺序调 add_turn
        （幂等：同一 turn_id 只存一次）。关机/退出前调 flush_pending()
        等它存完。
        需要同步的轮次（纠错、撤回、挑候选）：直接调 add_turn 当场跑完——
        纠错/撤回的结果要立刻回给页面展示，不能等后台。
        """
        if self._needs_sync_save(messages):
            return self.add_turn(messages, turn_id=turn_id)
        self.ensure_outbox_started()
        tid = self._outbox.submit(messages, turn_id=turn_id)
        return {"pending": True, "results": [], "queued": True, "turn_id": tid}

    def _needs_sync_save(self, messages):
        """这轮要不要同步存（26 号）：纠错、撤回、挑候选要当场出结果。

        有待确认的纠错等 BOOS 挑时、最新一句是纠错时、是撤回时
        （光秃秃的或带触发词），都同步——结果要立刻回给页面。
        其他轮次走后台。
        """
        if self._pending_correction is not None:
            return True
        latest_user = _latest_user_content(messages or [])
        if not latest_user:
            return False
        if _detect_correction(latest_user) is not None:
            return True
        if _is_bare_retraction(latest_user):
            return True
        return any(
            marker in latest_user for marker in RETRACTION_CANDIDATE_MARKERS
        )

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
            # 账本兜底项没有 mem0 id（id=None）：它只是档案里的原话，
            # 不能拿去 update / 打标记——槽位逻辑要的是索引里真实的
            # 那条，兜底的原话不算（索引里丢了它，对账会按账本重建）。
            if (cand or {}).get("source") == "ledger":
                continue
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
        # 46 号：提取先存下的同一事实（没打 slot 标记、触发词也认不出它，
        # 比如提取把“助手叫十二”改写成“用户给助手取名为「十二」”）——
        # 别再新存一条，原地更新成槽位记忆（打上 slot 标记）。
        if slot != SLOT_SOUL:
            near = self._find_near_duplicate_memory(text, self.get_all())
            if near is not None:
                try:
                    self._memory.update(
                        memory_id=near["id"],
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
                        "memory_id": near["id"],
                    }
                return {
                    "slot": slot,
                    "value": value,
                    "status": "updated",
                    "memory_id": near["id"],
                }
        # 种子是新记忆：记账本（best_effort——记失败不拦种子本身，
        # 记忆本身还在索引里，只是账本里缺这一笔档案）。
        result = self._add_text_with_metadata(
            text,
            _provenance(SOURCE_USER, CONFIDENCE_USER, True, slot=slot),
            ledger_mode="best_effort",
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
        # 同槽位新增是新记忆：记账本（best_effort——记失败不拦，
        # 记忆本身还在索引里，只是账本里缺这一笔档案）。
        result = self._add_text_with_metadata(
            text,
            _provenance(SOURCE_USER, CONFIDENCE_USER, True, slot=slot),
            ledger_mode="best_effort",
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
        """把十二的名字、SOUL.md 和 BOOS 的电脑配置以“已确认”种进长期记忆（Plan C + 17 号）。

        只加没有的，绝不覆盖已有的槽位值（见 _seed_slot_memory）。
        网页服务启动时调一次（web/server.py）；CLI 不自动调
        （命令行默认不加载 mem0）。失败时记日志、不抛异常——
        种子没种上不能把服务搞崩。
        返回 {"agent_name": {...}, "soul": {...}, "computer": {...}}，每项是
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
            try:
                outcomes["computer"] = self._seed_slot_memory(
                    SLOT_COMPUTER, COMPUTER_VALUE, COMPUTER_TEXT
                )
            except Exception:
                logger.warning("种电脑配置记忆失败。", exc_info=True)
                outcomes["computer"] = {
                    "slot": SLOT_COMPUTER,
                    "value": COMPUTER_VALUE,
                    "status": "failed",
                    "memory_id": None,
                }
        return outcomes

    def _pend_correction_remainder(self, safe_messages, latest_user, turn_id=None):
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
        # created_at：这轮话是什么时候说的——提取复活拦截的分水岭
        # （2026-11-16 拉面）：删在这之后才拦，删在之前的是 BOOS 重新说起，不拦。
        # 必须走单调的 _now_epoch()，别用 raw datetime.now()：账本删记录的
        # ts 是 _now_iso()（同一单调源），raw 墙钟在 Windows 粗时钟下会
        # 落后于它，删在前的事反而显得晚（2026-11-19 全套 flaky 的根子）。
        self._pending_turn = {
            "messages": messages,
            "identity": identity_items,
            "created_at": _now_epoch(),
            "turn_id": turn_id,
        }

    def add_turn(self, messages, turn_id=None):
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
        参数 turn_id：本轮的 id（save_turn/outbox 传进来）；存进待定区，
        提交时写进新记忆的 metadata 和账本 add 事件（见
        _commit_pending_turn）；不传就没有轮次信息，老路子照常走。
        mem0 自己调 DeepSeek 提取（自动存，BOOS 已同意不用确认），
        再去重、存进向量库。整个写入走全局写锁，一次只让一个写，
        防止并发写出重复记忆。
        来源：提取出的记忆记 source="inferred"、user_confirmed=False、
        confidence=0.7（见文件头“来源”一节）——用户明确说过的话，
        但 BOOS 没逐条确认过那个写法；存下来了也不会自动变成已确认。
        价值判断在提取提示里（CUSTOM_INSTRUCTIONS 第 9–12、14 条，DeepSeek
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
            bare_retraction = bool(latest_user) and _is_bare_retraction(latest_user)
            assistant_asked = False
            assistant_asked_text = None
            pending = self._pending_turn
            if pending is not None:
                for m in reversed(pending.get("messages") or []):
                    if (m or {}).get("role") == "assistant":
                        content = (m or {}).get("content") or ""
                        assistant_asked = "？" in content or "?" in content
                        assistant_asked_text = content if assistant_asked else None
                        break
            if bare_retraction and pending is not None and not assistant_asked:
                self._pending_turn = None
                self._pending_correction = None
                # 34 号：撤回这句本身也别留给以后提取——把它记成已处理，
                # 否则下一轮提交滑动窗口时，“不不不”这类撤回句会被当成
                # 新事实提出来（鸡杂面那次就是撤回句被存成了第二条记忆）。
                if latest_user:
                    key = latest_user.strip()
                    if key and key not in self._committed_user_keys:
                        self._committed_user_keys.append(key)
                dropped = sum(
                    1
                    for m in pending.get("messages") or []
                    if (m or {}).get("role") == "user"
                )
                return {"retracted": True, "dropped": dropped, "results": []}
            # 0a. 模型判意图的撤回（34 号）：光秃秃的撤回走上面的快捷路；
            #     带后半句的（“不不不，刚刚那句不算”）走这里——先看是不是
            #     纠正（带新事实的纠正绝不走撤回），再看有没有撤回触发词，
            #     有就请模型判意图。光秃秃的撤回 + 助手刚问了问题时不判
            #     （上面已判定为“在回答问题”，不再翻案）。
            #     判是撤回：丢掉待定区 + 把已进库的那句软删除（能恢复）。
            #     判不是 / 判不出来：当普通轮次继续，绝不因为判不出来就丢记忆。
            if (
                latest_user
                and not (bare_retraction and assistant_asked)
                and _detect_correction(latest_user) is None
                and any(
                    marker in latest_user
                    for marker in RETRACTION_CANDIDATE_MARKERS
                )
            ):
                prev_user = None
                if pending is not None:
                    for m in reversed(pending.get("messages") or []):
                        if (m or {}).get("role") == "user":
                            prev_user = (m or {}).get("content") or ""
                            break
                if prev_user is None:
                    user_msgs = [
                        (m or {}).get("content") or ""
                        for m in safe_messages
                        if (m or {}).get("role") == "user"
                    ]
                    if len(user_msgs) >= 2:
                        prev_user = user_msgs[-2]
                if prev_user:
                    retract = self._judge_retraction_intent(
                        latest_user, prev_user, assistant_asked_text
                    )
                    if retract is True:
                        dropped = 0
                        if pending is not None:
                            dropped = sum(
                                1
                                for m in pending.get("messages") or []
                                if (m or {}).get("role") == "user"
                            )
                            self._pending_turn = None
                        self._pending_correction = None
                        # 34 号：撤回这句本身也别留给以后提取——把它记成
                        # 已处理，否则下一轮提交滑动窗口时，撤回句
                        # （“其实也不是很喜欢吃”）会被当成新事实提出来，
                        # 存成第二条记忆（鸡杂面那次就是这么多出来的）。
                        key = latest_user.strip()
                        if key and key not in self._committed_user_keys:
                            self._committed_user_keys.append(key)
                        soft_deleted = self._soft_delete_same_fact(prev_user)
                        return {
                            "retracted": True,
                            "dropped": dropped,
                            "soft_deleted": soft_deleted,
                            "results": [],
                        }
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
                        # 上一轮提交的保护失败不能丢在半路：纠错结果
                        # 里没有保护字段，上一轮的才补进来；只填空、
                        # 不覆盖（见 _carry_protection_fields）。
                        _carry_protection_fields(committed, handled)
                        return handled
                    # not_found：挑中的那条已经不在库里了（被删了），
                    # 新说法还是他确认过的，直接当新记忆存下来。
                    saved = self.add_memory_text(
                        pending.get("new_text") or "",
                        source=SOURCE_USER,
                        turn_id=turn_id,
                    )
                    if isinstance(saved, dict):
                        saved = dict(saved)
                        saved["correction"] = _correction_result(outcome)
                        _carry_protection_fields(committed, saved)
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
                    self._pend_correction_remainder(
                        safe_messages, latest_user, turn_id=turn_id
                    )
                    # 上一轮提交的保护失败不能丢在半路：纠错结果里
                    # 没有保护字段，上一轮的才补进来；只填空、不覆盖。
                    _carry_protection_fields(committed, handled)
                    return handled
                # not_found：库里没有冲突的旧记忆——新说法当成一条新记忆，
                # 走正常提取保存（下面），结果里附上这次纠错的尝试记录。
                normal = self._extract_and_save_turn(safe_messages, turn_id=turn_id)
                if isinstance(normal, dict):
                    normal = dict(normal)
                    normal["correction"] = _correction_result(outcome)
                    # 上一轮提交的保护失败不能丢在半路：本轮自己的
                    # 提取也可能带保护失败，留本轮的、只填空不覆盖。
                    _carry_protection_fields(committed, normal)
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
            # 52 号：上一轮提取失败的那几句（committed["failed_messages"]），
            # 跟着本轮一起进待定区，下一轮提交时再试一次；超过 5 句时
            # 只留最近的 5 句（服务一直坏时不让待定区无限长大，
            # 丢掉的最老的几句只记日志）。
            pending_messages = list(safe_messages)
            failed_messages = (committed or {}).get("failed_messages") or []
            if failed_messages:
                carried = [
                    {"role": "user", "content": text}
                    for text in failed_messages[-5:]
                    if (text or "").strip()
                ]
                if carried:
                    if len(failed_messages) > len(carried):
                        logger.warning(
                            "提取失败重试：%d 句里只带回最近 %d 句，丢掉 %d 句旧的。",
                            len(failed_messages), len(carried),
                            len(failed_messages) - len(carried),
                        )
                    logger.warning(
                        "上一轮 %d 句没存成，跟着本轮再试一次。",
                        len(carried),
                    )
                    pending_messages = carried + pending_messages
            self._pending_turn = {
                "messages": pending_messages,
                "identity": identity_items,
                # created_at：本轮话是什么时候说的——提取复活拦截的分水岭
                # （2026-11-16 拉面）：删在这之后才拦，删在之前的是 BOOS 重新说起，不拦。
                # 走单调的 _now_epoch()（见上面那处 _pending_turn 的注释）。
                "created_at": _now_epoch(),
                "turn_id": turn_id,
            }
            # 闲置提交（2026-11-20 游泳事故）：停车后（重）启定时器，闲置满
            # _pending_idle_seconds 仍是这轮就提交；新轮次来了会重设。
            self._arm_pending_timer(turn_id)
            out = {
                "pending": True,
                "results": (committed or {}).get("results") or [],
                "identity": [item["slot"] for item in identity_items],
            }
            if (committed or {}).get("extraction_failed"):
                out["extraction_failed"] = True
                out["extraction_reason"] = (committed or {}).get("extraction_reason")
                out["failed_messages"] = (committed or {}).get("failed_messages") or []
            # 保护失败的字段用共用的 _carry_protection_fields 一次带过去，
            # 和 _commit_pending_turn 里是同一个函数——以后再加字段只改
            # 它里面那一行，不会再漏。
            _carry_protection_fields(committed, out)
            return out

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

    def _drop_management_action_results(self, result):
        """提取后过滤：删掉“把管记忆的动作复述成一条事实”的提取结果。

        2026-11-16 真机测试（问题 4）：BOOS 只说了“把'我爱吃冒菜'这条
        删掉”，提取却存了一条“用户表示自己不再爱吃冒菜了，并要求删除
        此前记录的相关偏好记忆”——把删记忆的动作脑补成偏好变了，还拿
        它当理由拒绝恢复。管记忆的动作（删/改一条记忆）本身已经记在
        账本里（delete/update 事件），不需要再存一条“用户要求删除/修改
        记忆”当长期记忆。
        只拦复述动作本身的条目：文本里同时提到动作词（删/改）和
        “记忆/记录”。真凭 BOOS 原话说的偏好变化（“我不爱吃冒菜了”，
        不提删记忆）不在这个形状里，不拦——那是真事实，照常存。
        和 _drop_sensitive_results 一样的处理：条目已经在库里了，
        从库里删掉、也不返回给调用方；账本那步在后面，只对留下的
        条目补记，拦掉的不会进账本。
        返回过滤后的结果（拦掉的条目不在 results 里）。
        """
        items = (result or {}).get("results") or []
        if not items:
            return result
        kept = []
        dropped = 0
        for item in items:
            text = (item or {}).get("memory") or ""
            if text and _describes_memory_management(text):
                dropped += 1
                mem_id = (item or {}).get("id")
                if mem_id:
                    try:
                        self._memory.delete(mem_id)
                    except Exception:
                        logger.warning(
                            "提取后过滤：删除管记忆动作的复述失败。", exc_info=True
                        )
                continue
            kept.append(item)
        if dropped:
            logger.info("提取后过滤：%d 条是管记忆动作的复述，已删除。", dropped)
        if len(kept) == len(items):
            return result
        filtered = dict(result)
        filtered["results"] = kept
        return filtered

    # 提取后守卫用的关键词门槛：和检索补充信号（KEYWORD_MIN_WEIGHT=2）
    # 不是同一把尺子——守卫一旦拦下就要挂起来、以后真用上时问 BOOS，
    # 多拦的代价是以后多问一句，所以宁严不宽：新记忆和旧记忆至少
    # 共享两个两字词（或一个更长的共有词）才算“同一话题”。
    # 例子：新记忆“数据库改成 PostgreSQL”和旧记忆“数据库用 SQLite”
    # 共享“数据”“据库”，权重 2+2=4，刚好够；只共享一个词（比如只
    # 共享“预算”）的，不拦——那点重合说明不了是同一件事。
    _COVERAGE_GUARD_MIN_WEIGHT = 4

    def _verify_after_repair(self, expect_present, expect_absent_ids):
        """改完库再读一遍核对（恢复和回滚共用的核对办法）。

        病根（2026-11-10 GPT 再审）：恢复和回滚以前都信“没报错就
        是成功了”——update/add/delete 没抛异常就数成做完了。没报错
        只能说明调用没炸，不能说明库里的东西真对了（update 可能假装
        成功根本没改，delete 可能假装成功根本没删）。所以凡是动手改
        了库的动作，做完都要再读一遍、按期望的样子逐条核对，对上了
        才算数——恢复、回滚都调这一个办法，以后第三处改库也调它，
        不会再犯同一个错。
        只读本地库、只做本地比对，不调模型、不花钱；调用方已经持有
        写锁（提取那条路），这里不另起锁。
        参数：
          expect_present：应在库里的条目，每项 {"lookup_id", "report_id",
            "text", "metadata"}——lookup_id 是按 id 找的凭据（update
            恢复的旧记忆，id 没变）；重新 add 的那条 id 变了，
            lookup_id 是 None，改按内容找（文本 + 规整后的 metadata
            都对上）。report_id 是核对不上时要报的 id（重新 add 的
            也记旧 id——调用方只认得旧 id，新 id 它不知道）。
            metadata 只比我们自己的字段（created_at、created_at_source、
            write_seq、source、confidence、user_confirmed、slot、
            pending_confirmation），用 _normalize_metadata 规整后比——
            mem0 自己加的字段（比如它内部的 hash、更新时间）不比，
            不然会误判。
            含糊点（写明）：库里有多条内容一样的，有一条对得上就算
            恢复了——重新 add 的那条和库里本来就有的重复项分不开，
            但这条路只在“旧记忆被删了、按快照重存”时走，重存的文本
            和 metadata 和快照一致就是恢复的目的；多一条重复顶多是
            多占一条，不会把错的当成对的。
          expect_absent_ids：应在库里消失的 id（回滚删掉的新记忆）。
        返回 {"read_ok", "missing", "still_present"}：
          read_ok False 表示核对那一下读库失败，什么都不知道——
            调用方按“没法核对”失败关闭，不能按成功处理；这时
            missing 是全部 expect_present 的 report_id、
            still_present 是全部 expect_absent_ids（一个都没确认，
            保守全算没对上）。
          missing：expect_present 里没对上的 report_id；
          still_present：expect_absent_ids 里读回来还在的 id。
        """
        try:
            current = self._snapshot_all()
        except Exception:
            logger.warning("改完库核对时读库失败，没法确认结果。", exc_info=True)
            return {
                "read_ok": False,
                "missing": [
                    (item or {}).get("report_id")
                    for item in (expect_present or [])
                    if (item or {}).get("report_id")
                ],
                "still_present": list(expect_absent_ids or []),
            }
        by_id = {}
        contents = []
        for item in current:
            mem_id = (item or {}).get("id")
            text = (item or {}).get("text") or ""
            meta = _normalize_metadata((item or {}).get("metadata"), item)
            if mem_id:
                by_id[mem_id] = {"text": text, "metadata": meta}
            contents.append({"text": text, "metadata": meta})
        missing = []
        for expected in expect_present or []:
            text = (expected or {}).get("text") or ""
            meta = _normalize_metadata((expected or {}).get("metadata"), expected)
            lookup_id = (expected or {}).get("lookup_id")
            found = False
            if lookup_id:
                now = by_id.get(lookup_id)
                found = (
                    now is not None
                    and now["text"] == text
                    and now["metadata"] == meta
                )
            else:
                found = any(
                    entry["text"] == text and entry["metadata"] == meta
                    for entry in contents
                )
            if not found:
                report_id = (expected or {}).get("report_id")
                if report_id:
                    missing.append(report_id)
        still_present = [
            mem_id for mem_id in (expect_absent_ids or []) if mem_id in by_id
        ]
        return {"read_ok": True, "missing": missing, "still_present": still_present}

    def _restore_snapshot_memories(self, before_items):
        """按提取前的快照，把被改动过的旧记忆恢复原样（守卫的第二道）。

        背景：守卫本来只给新记忆打“待确认”标记——这够用的前提，是
        mem0.add 只插新记忆、不动旧记忆。实际装着的 mem0 2.2.1 读
        代码确实是只插不改，但说明书上写着 add 会决定增改删，升级
        就可能变；万一哪天 add 在里面把旧记忆改了、删了，事后只打
        标记就救不回来了。所以这里加一道事后核对：add 完拿快照跟
        现在的库比对，快照里有、现在文本被改了或者没了的旧记忆，
        都按快照原样恢复（文本 + metadata 全套，包括时间戳、时间
        来源、序号、来源和确认状态）。恢复的对象是快照里所有的旧
        记忆，不只核心记忆——提取这条路只会存推断事实，纠正走
        correct_memory 不走这里，所以恢复全部被改的旧记忆是安全、
        对的。恢复的办法：还在库里的，原地 update 回去（保住原 id）；
        已经没了的，重新存一条（mem0 的 id 是它自己生成的，恢复
        不回来旧 id；文本和 metadata 原样带上，一样不少）。
        读库失败时不猜：分不清“全被删了”还是“读失败了”，宁可本次
        不恢复（记一条警告），也别把没删的东西重复存一遍。
        恢复只记日志、不抛异常——守卫坏了不能把聊天回答搞崩。
        恢复完再读一遍核对（见 _verify_after_repair）：update 或重新
        add 没报错不等于恢复了，读回来文本 + metadata 都对上了才算；
        底层 update 不支持 metadata、只恢复了文本的那种，metadata 对
        不上，数成失败。restored 只数核对上了的。核对那一下读库失败
        时 verified=False、failed_ids 是本轮试着恢复的 id（一个都没
        确认，保守全算没恢复）；没什么要恢复时不读第二遍——第一遍
        读已经比过、都没动过，没必要再读。
        返回一个小结构（调用方要分得出三种情况，不能只给个数）：
        {"verified": 本次到底核没核对上（读库失败时是 False）、
         "restored": 核对上了几条、
         "failed_ids": 没恢复回来的旧记忆 id（空表示都恢复了、
         或本来就没什么要恢复的）}。
        """
        if not before_items:
            return {"verified": True, "restored": 0, "failed_ids": []}
        try:
            raw = self._fetch_all_raw()
        except Exception:
            logger.warning("覆盖守卫：核对快照时读库失败，本次不恢复。", exc_info=True)
            return {"verified": False, "restored": 0, "failed_ids": []}
        current = {}
        for item in (raw or {}).get("results", []):
            mem_id = (item or {}).get("id")
            text = (item or {}).get("memory") or ""
            if not mem_id or not text:
                continue
            current[mem_id] = {
                "text": text,
                "metadata": _normalize_metadata((item or {}).get("metadata"), item),
            }
        attempted = []
        for old in before_items:
            mem_id = (old or {}).get("id")
            old_text = (old or {}).get("text") or ""
            if not mem_id or not old_text:
                continue
            old_meta = _normalize_metadata((old or {}).get("metadata"))
            now = current.get(mem_id)
            if now is None:
                # 没了：重新存一条，文本和 metadata 原样带上。
                # 存完不数成功——成没成等下面统一核对。
                # 写回前先展平（见 _metadata_for_store）：规整后的
                # metadata 里 money 是字典形，直接传 Chroma 拒收
                # （2026-11-16 恒生定投就是这么恢复失败的）。
                try:
                    self._mem0_add(
                        [{"role": "user", "content": old_text}],
                        self.user_id,
                        False,
                        _metadata_for_store(old_meta),
                    )
                except Exception:
                    logger.warning("覆盖守卫：恢复被删的旧记忆失败。", exc_info=True)
                # 新存的 id 是 mem0 生成的，核对时按内容找，失败报旧 id。
                attempted.append(
                    {
                        "lookup_id": None,
                        "report_id": mem_id,
                        "text": old_text,
                        "metadata": old_meta,
                    }
                )
                continue
            if now["text"] == old_text and now["metadata"] == old_meta:
                continue
            # 文本被改了、或者 metadata 被动了：原地恢复，保住原 id。
            # 改完不数成功——成没成等下面统一核对。
            # 写回前先展平（见 _metadata_for_store）：规整后的 metadata
            # 里 money 是字典形，直接传 Chroma 拒收（2026-11-16 恒生定投
            # 就是这么恢复失败的）。
            try:
                try:
                    self._memory.update(
                        memory_id=mem_id, text=old_text,
                        metadata=_metadata_for_store(old_meta),
                    )
                except TypeError:
                    logger.warning(
                        "覆盖守卫：底层 update 不支持 metadata 参数，"
                        "只恢复了文本、没恢复 metadata。"
                    )
                    self._memory.update(memory_id=mem_id, text=old_text)
            except Exception:
                logger.warning("覆盖守卫：恢复被改的旧记忆失败。", exc_info=True)
            attempted.append(
                {
                    "lookup_id": mem_id,
                    "report_id": mem_id,
                    "text": old_text,
                    "metadata": old_meta,
                }
            )
        if not attempted:
            return {"verified": True, "restored": 0, "failed_ids": []}
        # 改完再读一遍核对：没报错不等于恢复了，读回来对上了才算。
        # 核对读库失败时一个都没确认，全算没恢复（失败关闭）。
        check = self._verify_after_repair(attempted, [])
        if not check["read_ok"]:
            return {
                "verified": False,
                "restored": 0,
                "failed_ids": [item["report_id"] for item in attempted],
            }
        failed_ids = check["missing"]
        restored = len(attempted) - len(failed_ids)
        if restored:
            logger.info("提取后守卫：按快照恢复了 %d 条被改动的旧记忆。", restored)
        return {"verified": True, "restored": restored, "failed_ids": failed_ids}

    def _rollback_new_memories(self, items):
        """删掉本轮新存的几条记忆（恢复失败时的回滚）。

        删的办法复用 _drop_sensitive_results 里现成的
        self._memory.delete——刚存进来的记录，删掉就是回到写之前
        的样子。删失败只记日志、不抛异常。
        删完再读一遍核对（见 _verify_after_repair）：delete 没报错
        不等于删掉了，读回来不在才算删掉；delete 抛了错也不等于还
        在——到底在不在，以读回来的为准，残留的判定只看核对结果。
        没 id 的条目删不掉、也核对不了，保守算残留。核对那一下读
        库失败时，一个都确认不了，全算残留（失败关闭）。
        返回删完还留着的条目（删失败的还在里面）。
        """
        no_id = []
        with_id = []
        for item in items or []:
            if (item or {}).get("id"):
                with_id.append(item)
            else:
                no_id.append(item)
        for item in with_id:
            try:
                self._memory.delete(item["id"])
            except Exception:
                logger.warning("回滚新记忆失败：id=%s。", item["id"], exc_info=True)
        if not with_id:
            return list(no_id)
        check = self._verify_after_repair([], [item["id"] for item in with_id])
        still = set(check["still_present"])
        return list(no_id) + [item for item in with_id if item["id"] in still]

    def _apply_coverage_guard(self, results, before_items, new_metadata):
        """提取后覆盖守卫：刚提取出的推断记忆，别把 BOOS 确认过的事实顶掉。

        只处理一种情况：新记忆是推断的（inferred），而库里已经有一条
        同一话题（关键词重合 ≥ _COVERAGE_GUARD_MIN_WEIGHT，和检索
        补充信号同一类尺子、但更严）的用户已确认记忆——这时新记忆
        不能直接当成事实，改标“待确认”挂起来（metadata 里
        pending_confirmation=True，时间戳和序号原样保留），旧记忆
        原封不动；只有真用上时才问 BOOS（问的那一步以后再接，
        先挂起来——挂起来本身就有用：面板上看得到，主流程里它
        本来就是 inferred、不会被当成确认过的事实）。
        同级冲突在这里不自动标“被取代”：关键词重合分不出“同一条事实
        的新版本”还是“同一话题的不同事实”，宁可不标，不标错（见
        _resolve_conflict 注释）。用户亲口说的（记住/纠正/槽位）不走
        这里——那是 BOOS 本人的话，可信度最高，不该被守卫拦。
        before_items：提取前库里的记忆快照（get_all 的结果），用来找邻居、
        也防止新记忆和自己/同一批的兄弟互相认成邻居。
        new_metadata：本次提取传给 mem0.add 的那份 metadata（兄弟们共用
        的同一份），打待确认标记时在它的基础上加 pending_confirmation，
        时间戳和序号原样保留。
        返回：标注过的结果字典（没拦下任何东西时原样返回；结果里被拦下
        的那条多一个 "pending_confirmation": True，调用方不用再查库）。
        拦下/失败只记日志（不记内容），绝不抛异常——守卫坏了不能把
        聊天回答搞崩。标记打不上（底层 update 不支持 metadata）时，
        记忆本身还在，只丢了标记，记一条警告。
        打标记之前先按快照核对恢复（见 _restore_snapshot_memories）：
        万一 add 在里面动了旧记忆，先恢复原样，再谈新记忆待确认——
        旧事实丢了，打再多标记也没用。
        恢复失败时失败关闭，不当成保护成功（恢复的返回值不能丢掉）：
        (a) 连库都读不出来、根本没法核对——本轮新存的都删掉回滚，
        结果里标 protection_failed=True、原因 restore_unverified，
        记一条保护事件，提前返回，不再打标记；回滚删不掉、还留在
        库里的新记忆，id 写进结果（rollback_leftover_ids）、原文写
        进保护事件，事件里不写“已删掉回滚”；
        (b) 个别旧记忆没恢复回来——只回滚跟它们同一话题的新记忆，
        其他新记忆照常走打标记的流程，结果里同样标 protection_failed、
        原因 restore_partial、并写明是哪几条旧记忆（failed_ids），
        也记一条保护事件；回滚的残留同样写进结果和保护事件。
        没恢复回来的旧记忆本身还是坏的，这事标出来、记下来，
        不能当成保护成功。
        """
        if not before_items:
            return results
        results = dict(results or {})
        restore = self._restore_snapshot_memories(before_items)
        items = (results or {}).get("results") or []
        if not restore.get("verified", False):
            # (a) 没法核对：旧记忆到底被动了没有根本不知道。光标个
            #     失败不够——新记忆已经存进去了，得把本轮新存的都
            #     删掉回滚，让库回到写之前的样子。删完还剩下的（回滚
            #     核对出来的残留）要写进结果和保护事件，不能写成
            #     “已删掉回滚”——没删掉就是没删掉。
            survivors = self._rollback_new_memories(items)
            results["results"] = survivors
            results["protection_failed"] = True
            results["protection_reason"] = "restore_unverified"
            leftover_ids = [
                (item or {}).get("id")
                for item in survivors
                if (item or {}).get("id")
            ]
            leftover_texts = [(item or {}).get("memory") or "" for item in survivors]
            if leftover_ids or leftover_texts:
                results["rollback_leftover_ids"] = leftover_ids
                results["rollback_leftover_texts"] = leftover_texts
                reason = (
                    "恢复时读库失败，没法核对 %d 条旧记忆有没有被改动，"
                    "本轮新存的 %d 条记忆里删掉了 %d 条，还有 %d 条没删掉"
                    "（仍在库里）。"
                    % (
                        len(before_items),
                        len(items),
                        len(items) - len(survivors),
                        len(survivors),
                    )
                )
            else:
                reason = (
                    "恢复时读库失败，没法核对 %d 条旧记忆有没有被改动，"
                    "本轮新存的记忆已删掉回滚。" % len(before_items)
                )
            self._record_protection_event(
                "restore_unverified",
                reason,
                failed_ids=[
                    (old or {}).get("id")
                    for old in before_items
                    if (old or {}).get("id")
                ],
                failed_texts=[
                    (old or {}).get("text") or "" for old in before_items
                ],
                rollback_leftover_ids=leftover_ids or None,
                rollback_leftover_texts=leftover_texts or None,
            )
            return results
        failed_ids = restore.get("failed_ids") or []
        if failed_ids:
            # (b) 个别旧记忆没恢复回来：只回滚跟它们同一话题的新记忆，
            #     其他新记忆照常走打标记的流程。
            failed_set = set(failed_ids)
            failed_old = [
                old for old in before_items if (old or {}).get("id") in failed_set
            ]
            doomed = []
            kept = []
            for item in items:
                new_text = (item or {}).get("memory") or ""
                keywords = _extract_keywords(new_text) if new_text else set()
                same_topic = False
                for old in failed_old:
                    old_text = (old or {}).get("text") or ""
                    weight, _matched = _keyword_weight(old_text, keywords)
                    if weight >= self._COVERAGE_GUARD_MIN_WEIGHT:
                        same_topic = True
                        break
                if same_topic:
                    doomed.append(item)
                else:
                    kept.append(item)
            survivors = self._rollback_new_memories(doomed)
            items = kept
            results["results"] = items
            results["protection_failed"] = True
            results["protection_reason"] = "restore_partial"
            results["failed_ids"] = list(failed_ids)
            leftover_ids = [
                (item or {}).get("id")
                for item in survivors
                if (item or {}).get("id")
            ]
            leftover_texts = [(item or {}).get("memory") or "" for item in survivors]
            if leftover_ids or leftover_texts:
                # 回滚不完整：删不掉的那条还留在库里（results 里没有
                # 它），id 和原文都写进结果往上传，原文也写进保护事件——
                # 调用方和面板都得看得见，不能假装删掉了。
                results["rollback_leftover_ids"] = leftover_ids
                results["rollback_leftover_texts"] = leftover_texts
                reason = (
                    "有旧记忆没恢复回来，跟它们同一话题的新记忆里删掉了 %d 条，"
                    "还有 %d 条没删掉（仍在库里）。"
                    % (len(doomed) - len(survivors), len(survivors))
                )
            else:
                reason = "有旧记忆没恢复回来，跟它们同一话题的新记忆已删掉回滚。"
            self._record_protection_event(
                "restore_partial",
                reason,
                failed_ids=list(failed_ids),
                failed_texts=[
                    (old or {}).get("text") or "" for old in failed_old
                ],
                rollback_leftover_ids=leftover_ids or None,
                rollback_leftover_texts=leftover_texts or None,
            )
        if not items:
            return results
        base_meta = dict(new_metadata or {})
        held = []
        for item in items:
            new_text = (item or {}).get("memory") or ""
            mem_id = (item or {}).get("id")
            if not new_text or not mem_id:
                continue
            keywords = _extract_keywords(new_text)
            if not keywords:
                continue
            for old in before_items:
                old_meta = (old or {}).get("metadata") or {}
                if not _is_core_memory(old_meta):
                    continue
                old_text = (old or {}).get("text") or ""
                weight, _matched = _keyword_weight(old_text, keywords)
                if weight < self._COVERAGE_GUARD_MIN_WEIGHT:
                    continue
                resolution = _resolve_conflict(old_meta, base_meta)
                if resolution.get("action") != "pending_new":
                    continue
                try:
                    try:
                        self._memory.update(
                            memory_id=mem_id,
                            text=new_text,
                            metadata={
                                **base_meta,
                                "pending_confirmation": True,
                            },
                        )
                    except TypeError:
                        logger.warning(
                            "覆盖守卫：底层 update 不支持 metadata 参数，"
                            "待确认标记没打上。"
                        )
                        self._memory.update(memory_id=mem_id, text=new_text)
                except Exception:
                    logger.warning("覆盖守卫：打待确认标记失败。", exc_info=True)
                    continue
                item["pending_confirmation"] = True
                held.append(mem_id)
                break
        if held:
            logger.info(
                "提取后守卫：%d 条推断记忆撞上 BOOS 确认过的事实，已标待确认。",
                len(held),
            )
        return results

    def _has_event_date(self, text):
        """文本里有没有“具体日期”——能把一次性的事件和一般喜好分开的信号。

        绝对日期（2026年11月19日这种带数字的）一定算；相对日期
        （今天/昨天/昨晚……）也算，但“我喜欢……”这种喜好句不算——
        “我喜欢今天天气真好”里的“今天”只是说今天天气，不是一件
        发生过的事（2026-11-20 test_43：别把闲话也护下来）。
        """
        if not isinstance(text, str) or not text.strip():
            return False
        has_relative = False
        for slot in extract_core_slots(text):
            if not slot.startswith("date:"):
                continue
            val = slot[5:]
            if any(ch.isdigit() for ch in val):
                return True
            has_relative = True
        if has_relative and "喜欢" not in text.lower():
            return True
        return False

    def _value_gate_protected(self, item):
        """价值关卡不得删的条（修 1 + 修 2，2026-11-20 牛肉面事故）：
        守卫已标待确认的，或带具体日期的事件记忆。"""
        if (item or {}).get("pending_confirmation"):
            return True
        text = (item or {}).get("memory") or ""
        return self._has_event_date(text)

    def _value_gate_extraction_results(self, results):
        """价值关卡（43 号）：提取出的新记忆，值不值得长期记的再判一次。

        提取提示词是第一道防线（只抽“值得长期记的”），这里是机制上的
        第二道：请模型给每条新记忆打 0–1 分，低于 VALUE_THRESHOLD 的
        从库里删掉、不进 results。拿不准的往低打（提示词里写明），
        所以 0.5 的档位不会把真事实拦在外面。
        判不出来（没客户端、调用失败、回的不是 JSON）时全留——关卡
        失败关闭，别把真事实拦在外面。results 里没有新记忆时直接返回。
        删掉的条数记在 results["value_dropped"] 里（调用方和面板看得见）。
        守卫已标待确认的、带具体日期的事件记忆不删（修 1、修 2，2026-11-20
        牛肉面事故）：前者是挂起来等 BOOS，后者是 episodic 记忆靠衰减沉底。
        """
        items = (results or {}).get("results") or []
        if not items:
            return results
        # 修 1 + 修 2（2026-11-20 牛肉面事故）：守卫已标待确认的、带具体
        # 日期的一次性事件，不送去打分——前者是“挂起来等 BOOS”，后者是
        # episodic 记忆该靠衰减沉底；两道都不该被价值关卡删掉。
        # （去重里金额分支已有“守卫在前、关卡在后”的同款保护，这里补上。）
        score_items = [
            {"id": (item or {}).get("id"), "text": (item or {}).get("memory") or ""}
            for item in items
            if (item or {}).get("id")
            and (item or {}).get("memory")
            and not self._value_gate_protected(item)
        ]
        scored = self._judge_value_scores(score_items) if score_items else {}
        if scored is None:
            return results
        dropped = []
        kept = []
        for item in items:
            if self._value_gate_protected(item):
                kept.append(item)
                continue
            mem_id = (item or {}).get("id")
            score = scored.get(mem_id)
            if score is not None and score < VALUE_THRESHOLD:
                dropped.append(item)
            else:
                kept.append(item)
        if not dropped:
            return results
        survivors = self._rollback_new_memories(dropped)
        survivor_ids = {(item or {}).get("id") for item in survivors}
        actually_dropped = [
            item for item in dropped if (item or {}).get("id") not in survivor_ids
        ]
        results["results"] = kept + survivors
        results["value_dropped"] = len(actually_dropped)
        if survivors:
            logger.warning("价值关卡：%d 条低价值记忆没删掉（仍在库里）。", len(survivors))
        return results

    def _supersede_money_memory(self, known_item, new_text, new_money):
        """53 号：同槽位新值——旧条原地更新成新说法（id 不变，审计里留
        UPDATE 行），返回更新后的条目字典；更新失败返回 None（调用方
        按“没合并”处理，新条照常留，别丢事实）。
        metadata 用旧条的整份 + 新的金额扁平字段（去掉旧的 money 字典，
        不然读回来时旧字典会盖掉新字段）；只传金额字段在假库（替换式
        update）里会丢掉来源/时间，真实 mem0 是合并式、不会丢，但两边
        都按整份传才稳。"""
        known_id = (known_item or {}).get("id")
        if not known_id or not new_text:
            return None
        metadata = dict((known_item or {}).get("metadata") or {})
        metadata.pop("money", None)
        metadata.update(_flatten_money(new_money))
        try:
            self._memory.update(memory_id=known_id, text=new_text, metadata=metadata)
        except Exception:
            logger.warning("金额槽位更新失败：旧条没改成新说法。", exc_info=True)
            return None
        updated = dict(known_item)
        updated["text"] = new_text
        updated["metadata"] = metadata
        # 账本记一笔取代：旧说法被新说法换掉了（原地更新，id 不变，
        # 状态还是 current，旧话留在账本里——resolve_state 里同 id
        # 的 supersede 只记 history，不标 superseded）。key 带上：
        # 旧原话的 key 标 superseded（对账、兜底按 key 认旧话，不会
        # 把旧说法当现况捞回来）；新原话的 key 标 current。
        old_text = (known_item or {}).get("text") or ""
        created_at = ((known_item or {}).get("metadata") or {}).get("created_at")
        self._ledger_append(
            _ledger.LEDGER_EVENT_SUPERSEDE,
            memory_id=known_id,
            supersedes_id=known_id,
            old_text=old_text,
            text=new_text,
            key=_ledger.ledger_event_key(old_text, created_at),
            current_key=_ledger.ledger_event_key(new_text, created_at),
        )
        # 新说法也要进档案：不然账本里只有旧金额那句的 add 事件，
        # 对账重建、兜底都找不到新金额这句。记失败不拦路——更新
        # 本身已经成功了，索引里的新说法还在。
        self._ledger_record_add(new_text, metadata)
        return updated

    def _raw_metadata_for(self, memory_id):
        """一条记忆的原始 metadata（判断 a 号）。

        get_all(include_sensitive=True) 返回的是规整过的 metadata，
        会丢掉 evidence 这类字段——拿它当整份传回 update 会把原始
        字段弄丢。所以改 metadata 之前，先到库里拿原始的那份。
        找不到/读不出来返回 None。
        """
        if not memory_id:
            return None
        try:
            raw = self._fetch_all_raw()
        except Exception:
            return None
        for item in (raw or {}).get("results", []):
            if item.get("id") == memory_id:
                return dict(item.get("metadata") or {})
        return None

    def _append_duplicate_evidence(self, canonical_item, new_text,
                                   new_source=None, new_confidence=None):
        """判断 (a) 号：新记忆是旧条的重复时，把新条当证据挂到旧条下
        （不另建主条）。证据存在旧条 metadata 的 "evidence" 里，
        是一串 JSON（Chroma 不接受 dict/list 值的 metadata，所以
        证据列表先 json.dumps 成字符串再存；读的时候再 json.loads）。
        每条证据留原文/来源/时间/可信度——旧条是 BOOS 亲口说的、
        新条是提取出来的时，旧条当主条、新条只当证据。
        返回 True（挂上了）或 False（挂失败，调用方按“没合并”处理，
        新条照常留，别丢事实）。"""
        import json

        canonical_id = (canonical_item or {}).get("id")
        if not canonical_id or not new_text:
            return False
        metadata = self._raw_metadata_for(canonical_id)
        if metadata is None:
            return False
        evidence = []
        raw_evidence = metadata.get("evidence")
        if isinstance(raw_evidence, str) and raw_evidence:
            try:
                parsed = json.loads(raw_evidence)
                if isinstance(parsed, list):
                    evidence = parsed
            except (ValueError, TypeError):
                evidence = []
        evidence.append(
            {
                "text": new_text,
                "source": new_source or SOURCE_INFERRED,
                "created_at": _now_iso(),
                "confidence": new_confidence,
            }
        )
        metadata["evidence"] = json.dumps(evidence, ensure_ascii=False)
        try:
            self._memory.update(memory_id=canonical_id, metadata=metadata)
        except Exception:
            logger.warning("重复证据挂到旧条失败：旧条没改。", exc_info=True)
            return False
        return True

    def _mark_memory_review(self, memory_id, conflict_for_review=False,
                            needs_review=False):
        """判断 (a) 号：给一条已存的记忆打上裁决标记
        （conflict_for_review / needs_review）。
        提取路径里新记忆已经存进去了，裁决完才知道要标——这里直接改
        它的 metadata（拿原始的那份改，不拿规整过的，见 _raw_metadata_for）。
        返回 True（标上了）或 False（标失败，不拦路，记忆本身还在）。
        """
        if not memory_id or (not conflict_for_review and not needs_review):
            return False
        metadata = self._raw_metadata_for(memory_id)
        if metadata is None:
            return False
        if conflict_for_review:
            metadata["conflict_for_review"] = True
        if needs_review:
            metadata["needs_review"] = True
        try:
            self._memory.update(memory_id=memory_id, metadata=metadata)
        except Exception:
            logger.warning("裁决标记写回失败：记忆本身还在。", exc_info=True)
            return False
        return True

    # ---------- 第二步衰减：强化、月扫、到期删除 ----------
    # 锁的规矩：下面这几个方法自己不拿 _WRITE_LOCK——_decay_touch /
    # _record_decay_occurrence 是从写流程里（add_memory_text、提取去重，
    # 它们已经拿着锁）调的，reinforce 是从读流程里（search）调的；
    # 只有 sweep_monthly 自己拿锁（它是显式维护操作，不在聊天流程里）。
    # 写回都走 _raw_metadata_for 拿原始整份 metadata 再改（见
    # _mark_memory_review 的注释：拿规整过的当整份传回，会把 evidence
    # 这类字段弄丢）。状态变化顺带记一笔账本（revive / promote /
    # dormant）：账本是家，索引上的标记也要在账本里留痕，记账失败
    # 不拦路（记忆本身的标记已经写回去了）。

    def _decay_touch(self, memory_id, now=None, record_occurrence=False, turn_id=None):
        """用上了一条记忆：刷新 last_used_at，顺带把沉底态唤醒回原话态。

        沉底态的唤醒：decay_state 从 dormant 回到 verbatim，清掉
        decay_dormant_at；文本从没动过，不用从备份里恢复原话。
        被用上也顺带清掉 decay_delete_due（用上了就不在待删之列）。
        核心记忆不写回：权重恒为 1，也不会被月扫收起来，没什么可刷新的。
        record_occurrence=True 时再记一笔“出现”：日期和会话各记一个
        集合（见 decay.record_seen_occurrence）——同一轮里说 3 遍只算
        一次，新日期且新会话才 +1，跨午夜但同一轮不算；两个集合都够
        3 个，一次性经历就晋升成稳定事实（kind 改 stable，衰减变慢；
        importance 不动——晋升是“沉得慢一点”，不是变成核心）。
        turn_id 是这轮的 id（调用方传进来）；没传时只按日期数（老路子）。
        只在去重已经判了“是同一件事”之后调（先标冲突、再强化）。
        返回 {"ok", "restored", "promoted"}；写回失败返回 ok=False，
        不抛异常（记忆本身还在，只是这次没刷新时间）。"""
        if not memory_id:
            return {"ok": False, "restored": False, "promoted": False}
        now_dt = _decay_now_dt(now)
        today = now_dt.date().isoformat()
        metadata = self._raw_metadata_for(memory_id)
        if metadata is None:
            return {"ok": False, "restored": False, "promoted": False}
        meta = _normalize_metadata(metadata)
        if _is_core_memory(meta):
            return {"ok": True, "restored": False, "promoted": False}
        text = None
        restored = False
        if meta["decay_state"] == DECAY_STATE_DORMANT:
            metadata["decay_state"] = DECAY_STATE_VERBATIM
            metadata.pop("decay_dormant_at", None)
            restored = True
        metadata["last_used_at"] = _decay_iso(now_dt)
        metadata.pop("decay_delete_due", None)
        promoted = False
        # 补丁 3：玩笑 / 测试 / 假设打标的、或已作废的记忆，不计出现次数、
        # 不晋升——打标的照常存、照常能搜、照常唤醒，只是不算进晋升次数。
        # （should_promote 里也有一道同样的闸：万一这条在打标之前已经
        # 攒够了次数，晋升那步也拦得住；两道闸各管各的，别只留一道。）
        tone = meta.get("tone")
        counts_for_promotion = (
            tone not in ("joke", "test", "hypothesis")
            and not bool(meta.get("voided"))
        )
        if record_occurrence and counts_for_promotion:
            # 出现集合（2026-11-19）：日期和会话各记一个集合，晋升要
            # 两个都够 3（见 decay.should_promote）。老记忆没有这两个
            # 集合时，从 decay_last_seen_day 起步——只有最后一次被提到
            # 的日期可考，算 1 个日期；会话一个也不可考，从空起步。
            old_dates = _parse_seen_list(metadata.get("decay_seen_dates"))
            if not old_dates:
                _last_seen_day = metadata.get("decay_last_seen_day")
                if isinstance(_last_seen_day, str) and _last_seen_day:
                    old_dates = [_last_seen_day]
            old_sessions = _parse_seen_list(metadata.get("decay_seen_sessions"))
            old_seen = metadata.get("decay_seen_days")
            if not isinstance(old_seen, int) or isinstance(old_seen, bool):
                old_seen = 0
            new_dates, new_sessions, date_added = record_seen_occurrence(
                old_dates, old_sessions, today, turn_id
            )
            metadata["decay_seen_dates"] = _dump_seen_list(new_dates)
            metadata["decay_seen_sessions"] = _dump_seen_list(new_sessions)
            if date_added:
                # 老字段跟着集合走：不同天数就是日期集合的大小；老记忆
                # 以前数出来的天数比集合大时，按大的留（不往回降）。
                metadata["decay_seen_days"] = max(old_seen, len(new_dates))
                metadata["decay_last_seen_day"] = today
            if meta["decay_kind"] == DECAY_KIND_EPISODIC:
                if should_promote(new_dates, new_sessions, tone=tone):
                    metadata["decay_kind"] = DECAY_KIND_STABLE
                    promoted = True
                elif not new_sessions and promotion_reached(len(new_dates)):
                    # 没会话信息可考的老路子（老记忆、没传 turn_id 的
                    # 调用）：只按不同天数晋升，老行为不变。
                    metadata["decay_kind"] = DECAY_KIND_STABLE
                    promoted = True
        try:
            self._memory.update(memory_id=memory_id, text=text, metadata=metadata)
        except Exception:
            logger.warning("衰减刷新写回失败：记忆本身还在。", exc_info=True)
            return {"ok": False, "restored": False, "promoted": False}
        if restored:
            self._ledger_append(_ledger.LEDGER_EVENT_REVIVE, memory_id=memory_id)
        if promoted:
            self._ledger_append(_ledger.LEDGER_EVENT_PROMOTE, memory_id=memory_id)
        return {"ok": True, "restored": restored, "promoted": promoted}

    def reinforce_memories(self, memory_ids, now=None, turn_id=None):
        """把检索里真正用上的几条记忆强化一次（只强化传进来的这几条，
        不抬整个候选池——search(reinforce=True) 传的是返回结果的 id）。
        沉底态的被用上就回到原话态（见 _decay_touch）。核心记忆不写回。
        强化也记一笔“出现”（见 _decay_touch 的 record_occurrence）：
        被拼进回答就是一次出现，日期和会话都记进集合。turn_id 是提问
        这轮的 id（search 传进来）；没传时只按日期数。
        返回 {"reinforced", "restored", "failed"}；写回失败只记在 failed
        里，不抛异常。"""
        reinforced = []
        restored = []
        failed = []
        for memory_id in memory_ids or []:
            if not memory_id:
                continue
            # 强化就是一次“出现”：日期和会话都记进集合——同一轮里强化
            # 几次只算一次（record_seen_occurrence 按 turn_id 去重），
            # 不同轮各算一次。没传 turn_id 时只按日期数（老路子）。
            outcome = self._decay_touch(
                memory_id, now=now, record_occurrence=True, turn_id=turn_id
            )
            if outcome["ok"]:
                reinforced.append(memory_id)
                if outcome["restored"]:
                    restored.append(memory_id)
            else:
                failed.append(memory_id)
        return {"reinforced": reinforced, "restored": restored, "failed": failed}

    def _record_decay_occurrence(self, memory_id, now=None, turn_id=None):
        """同一件事又被提到一次：记一笔“出现”，够 3 个不同天且 3 个
        不同会话就晋升。调用点都在去重返回“重复”之后——裁决已经说了
        这是同一件事、不是冲突（先标冲突、再强化，坑 2 的定法）。
        turn_id 是这轮的 id（调用方传进来）；没传时只按日期数（老路子）。
        返回 True（记上了）或 False（记失败，记忆本身还在）。"""
        try:
            outcome = self._decay_touch(
                memory_id, now=now, record_occurrence=True, turn_id=turn_id
            )
            return bool(outcome["ok"])
        except Exception:
            logger.warning("衰减出现次数写回失败：记忆本身还在。", exc_info=True)
            return False

    # ---------- 记忆整理：语气判标 + 大模型路由 ----------
    # 补丁 3 两段式判语气 + 十三节主力/备选模型 + BOOS 的预算规矩
    # （2026-11-19：超了只提醒、不断服务）。
    # 语气标在 metadata 里（tone / tone_evidence / tone_source /
    # voided，见 provenance._provenance）：打标的照常存、照常能搜，
    # 只不算进晋升、盖不掉已确认的记忆。
    # 本地筛（local_tone_screen）只认明确的说法和关键词；筛不出来的
    # 排进 tone_pending.jsonl（账本同目录），等后台 process_tone_queue
    # 请大模型判——不堵聊天。判不出来的留着下次再判，不猜成认真、
    # 也不猜成玩笑；大模型说拿不准时按认真算（见 tone.py）。

    def _tone_queue_path(self):
        return self.ledger_path.parent / "tone_pending.jsonl"

    def _tone_queue_read(self):
        """读出待判语气的队（每行 {"memory_id", "text", "queued_at"}）。
        文件没有、读坏了返回 []，不抛异常。"""
        path = self._tone_queue_path()
        entries = []
        try:
            if not path.exists():
                return []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict) and entry.get("memory_id"):
                    entries.append(entry)
        except OSError:
            logger.warning("语气待判队读失败。", exc_info=True)
        return entries

    def _tone_queue_write(self, entries):
        """把待判语气的队整个写回去（原子替换，写坏了旧队还在）。
        返回 True（写好了）或 False。"""
        path = self._tone_queue_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text(
                "".join(
                    json.dumps(entry, ensure_ascii=False) + "\n"
                    for entry in entries or []
                ),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            return True
        except OSError:
            logger.warning("语气待判队写失败。", exc_info=True)
            return False

    def _tone_queue_append(self, memory_id, text):
        """一条记忆排进待判语气的队（已在队里的不重复排）。
        写失败只记日志、不拦路（判语气是后台活，不能把写记忆搞崩）。"""
        if not memory_id:
            return False
        entries = self._tone_queue_read()
        if any(entry.get("memory_id") == memory_id for entry in entries):
            return True
        entries.append(
            {
                "memory_id": memory_id,
                "text": text or "",
                "queued_at": _now_iso(),
            }
        )
        return self._tone_queue_write(entries)

    def _tag_memory_tone(self, memory_id, tone, evidence, source):
        """给一条已存的记忆打上语气标（tone / tone_evidence / tone_source）。
        先拿原始整份 metadata 再合并着写（见 _mark_memory_review 注释）。
        返回 True（打上了）或 False。"""
        if not memory_id or tone not in _tone.TONE_VALUES:
            return False
        metadata = self._raw_metadata_for(memory_id)
        if metadata is None:
            return False
        metadata["tone"] = tone
        if isinstance(evidence, str) and evidence:
            metadata["tone_evidence"] = evidence
        if source in ("local", "model"):
            metadata["tone_source"] = source
        try:
            self._memory.update(memory_id=memory_id, metadata=metadata)
        except Exception:
            logger.warning("语气标记写回失败：记忆本身还在。", exc_info=True)
            return False
        return True

    def _mark_memory_grounding(self, memory_id):
        """给一条记忆打上 grounding_flag（补丁 4 第一层：跟出处对不上）。
        和 _mark_memory_review 同一套写法。返回 True / False。"""
        if not memory_id:
            return False
        metadata = self._raw_metadata_for(memory_id)
        if metadata is None:
            return False
        metadata["grounding_flag"] = True
        try:
            self._memory.update(memory_id=memory_id, metadata=metadata)
        except Exception:
            logger.warning("对原话标记写回失败：记忆本身还在。", exc_info=True)
            return False
        return True

    def _local_tone_for(self, text):
        """本地筛一段记忆文本（补丁 3 第一段）：返回 (tone, evidence)
        或 (None, None)。只认明确的说法和关键词，不猜。"""
        try:
            return _tone.local_tone_screen(text)
        except Exception:
            logger.warning("本地语气筛失败：这条先不打标。", exc_info=True)
            return (None, None)

    def organizing_router(self):
        """整理记忆的大模型路由（懒建）：主力 deepseek-v4-pro，备选
        flash，本地降级；每月 3 美元上限超了只提醒、不断服务
        （见 memory/model_router.py）。状态文件在账本同目录。
        建不出来（比如没配 DeepSeek Key）时返回 None——调用方走
        本地降级，不抛异常。"""
        router = getattr(self, "_organizing_router", None)
        if router is not None:
            return router
        try:
            router = _model_router.OrganizingModelRouter(
                state_path=str(
                    self.ledger_path.parent / "organizing_router_state.json"
                ),
                budget_tracker=self.budget_tracker,
            )
        except Exception:
            logger.warning("整理记忆路由建不起来：本次走本地降级。", exc_info=True)
            return None
        self._organizing_router = router
        return router

    def pending_notices(self):
        """等着告诉 BOOS 的事（目前只有一条：整理记忆的大模型超预算了）。
        返回 [{"kind", "message", ...}]，没有时 []。对话层（main.py /
        web 面板）在下一轮对话开头调它，告诉 BOOS、由他决定要不要停；
        BOOS 回话后调 organizing_router().acknowledge_notice() 认领，
        同一月里不再提醒。"""
        router = self.organizing_router()
        if router is None:
            return []
        try:
            notice = router.pending_notice()
        except Exception:
            return []
        if notice is None:
            return []
        return [{"kind": "organizing_over_budget", **notice}]

    def process_tone_queue(self, router=None):
        """后台判语气：把 tone_pending.jsonl 里的队一批一批判完。
        每条拿 tone_judgment_prompts 请大模型判（走 organizing_router，
        主力 + 备选 + 本地降级，费用记进整理记忆的 3 美元预算）；
        判出来的打标（tone / 证据 / source=model）；判不出来的（模型
        没回、没判官、路由建不起来）留队里下次再判，不猜。
        记忆已经没了、或已经打过标的跳过（队里那行直接清掉）。
        返回 {"judged", "tagged", "remaining", "skipped"}，不抛异常。
        调用方（后台任务/月扫/手动）按这个看进度；不在聊天流程里调。"""
        entries = self._tone_queue_read()
        if not entries:
            return {"judged": 0, "tagged": 0, "remaining": 0, "skipped": 0}
        if router is None:
            router = self.organizing_router()
        judged = 0
        tagged = 0
        skipped = 0
        remaining = []
        probe_primary = True  # 每批开头探一次主力（恢复了就切回去）
        for entry in entries:
            memory_id = entry.get("memory_id")
            text = entry.get("text") or ""
            if not memory_id or not text:
                skipped += 1
                continue
            raw = self._raw_metadata_for(memory_id)
            if raw is None:
                skipped += 1  # 记忆已经没了，队里这行不要了
                continue
            if (raw.get("tone") in _tone.TONE_VALUES
                    and raw.get("tone") != _tone.TONE_SERIOUS):
                skipped += 1  # 已经打过非认真的标了，不用再判
                continue
            if raw.get("voided"):
                skipped += 1
                continue
            if router is None:
                remaining.append(entry)
                continue
            system_prompt, user_prompt = _tone.tone_judgment_prompts(text)
            try:
                out = router.judge(
                    system_prompt,
                    user_prompt,
                    max_tokens=300,
                    purpose="tone",
                    probe_primary=probe_primary,
                )
            except Exception:
                out = None
            probe_primary = False
            if out is None:
                remaining.append(entry)  # 大模型没回，留队里下次再判
                continue
            judged += 1
            tone, evidence = _tone.parse_tone_judgment(out.get("text"))
            if tone is None:
                remaining.append(entry)  # 判不出来，留队里下次再判
                continue
            if self._tag_memory_tone(memory_id, tone, evidence, "model"):
                tagged += 1
            else:
                remaining.append(entry)
        self._tone_queue_write(remaining)
        return {
            "judged": judged,
            "tagged": tagged,
            "remaining": len(remaining),
            "skipped": skipped,
        }

    def void_joke_memory(self, memory_id):
        """BOOS 后来说“那句是开玩笑的”：这条记忆作废。
        打 voided 标记（原话留着，不删）、语气记成 joke（已经打过
        test / hypothesis 的保留原标——反正都不计晋升），已经晋升成
        稳定事实的打回一次性（decay_kind 回 episodic）。由它晋升的
        也一起作废：出现次数是按条记的，别条的晋升不靠它，所以打回
        它自己就够了。返回 {"ok", ...}，不抛异常。"""
        if not memory_id:
            return {"ok": False, "reason": "no_id"}
        metadata = self._raw_metadata_for(memory_id)
        if metadata is None:
            return {"ok": False, "reason": "not_found"}
        metadata["voided"] = True
        metadata["voided_at"] = _now_iso()
        tone = metadata.get("tone")
        if tone not in ("joke", "test", "hypothesis"):
            metadata["tone"] = "joke"
            metadata["tone_source"] = "local"
        if metadata.get("decay_kind") == DECAY_KIND_STABLE:
            metadata["decay_kind"] = DECAY_KIND_EPISODIC
        try:
            self._memory.update(memory_id=memory_id, metadata=metadata)
        except Exception:
            logger.warning("作废标记写回失败：记忆本身还在。", exc_info=True)
            return {"ok": False, "reason": "update_failed"}
        return {"ok": True, "memory_id": memory_id}

    def run_self_check(self, now=None, sample_size=10):
        """跑一次每月自检（见 memory/self_check.py 的九项表）。
        只查、只报告，不直接改库；要改的在报告每项的 actions 里，
        由调用方决定做不做。第 9 项模型抽查走 organizing_router
        （判官是大模型，费用记进整理记忆的预算）；路由不可用时
        第 9 项 skipped。返回自检报告 dict。"""
        from memory import store_self_check as _store_self_check

        adapter = _store_self_check.build_self_check_adapter(self)
        router = self.organizing_router()
        model_judge = None
        if router is not None:
            model_judge = _store_self_check.make_model_judge(router)
        return _self_check.run_self_check(
            adapter,
            now=now,
            model_judge=model_judge,
            sample_size=sample_size,
            router_status=router.status() if router is not None else {},
        )

    def sweep_monthly(self, now=None):
        """每月兜底整理（显式调用，不在聊天流程里自动跑）：
        只打标，不改文本、不压摘要、不删任何东西——
          1. 标沉底：原话态、30 天没人找、权重已掉到 0.30 下面的，
             打上 dormant 标记（decay_state 改 dormant，记
             decay_dormant_at；原话和向量文本都不动）。
          2. 标待删：从最后一次用上算起满 180 天还没人找的，
             打上 decay_delete_due 标记，等 BOOS 亲口说删才删（见
             delete_due_memories）。沉底和待删可以在同一次月扫里
             一起标，不用等下次。
        核心记忆直接跳过（钉住不动）。返回台账
        {"checked", "dormant", "delete_due", "skipped_core", "errors"}，
        每笔都带 id 和原因，BOOS 扫一眼就能对上。读库失败时返回
        checked=0、errors 里记一笔，不抛异常。"""
        now_dt = _decay_now_dt(now)
        ledger = {
            "checked": 0,
            "dormant": [],
            "delete_due": [],
            "skipped_core": 0,
            "errors": [],
        }
        try:
            items = self.get_all(include_sensitive=True)
        except Exception:
            logger.warning("月扫：读记忆库失败，本次不整理。", exc_info=True)
            ledger["errors"].append({"reason": "读记忆库失败"})
            return ledger
        with _WRITE_LOCK:
            for item in items:
                memory_id = (item or {}).get("id")
                text = (item or {}).get("text") or ""
                meta = (item or {}).get("metadata") or {}
                if not memory_id or not text:
                    continue
                ledger["checked"] += 1
                if _is_core_memory(meta):
                    ledger["skipped_core"] += 1
                    continue
                state = meta.get("decay_state") or DECAY_STATE_VERBATIM
                last_used_at = meta.get("last_used_at")
                kind = meta.get("decay_kind") or DECAY_KIND_EPISODIC
                mark_dormant = state == DECAY_STATE_VERBATIM and decay_should_go_dormant(
                    kind, last_used_at, now_dt, is_core=False
                )
                mark_delete_due = decay_should_delete(last_used_at, now_dt, is_core=False)
                if not mark_dormant and not mark_delete_due:
                    continue
                raw = self._raw_metadata_for(memory_id)
                if raw is None:
                    ledger["errors"].append(
                        {"id": memory_id, "reason": "读原始 metadata 失败"}
                    )
                    continue
                was_delete_due = bool(raw.get("decay_delete_due"))
                if mark_dormant:
                    raw["decay_state"] = DECAY_STATE_DORMANT
                    raw["decay_dormant_at"] = _decay_iso(now_dt)
                if mark_delete_due:
                    raw["decay_delete_due"] = True
                try:
                    self._memory.update(memory_id=memory_id, metadata=raw)
                except Exception:
                    ledger["errors"].append(
                        {"id": memory_id, "reason": "打标写回失败"}
                    )
                    continue
                # 账本记一笔睡着：刚睡着时记一次；已经睡着、这次待删
                # 刚翻成 True 时也记一次（账本里的 delete_due 要跟上，
                # 不然账本还以为这条只是睡着、没到待删）。已经睡着又
                # 已经待删的就不重复记了——月扫每月跑一次，别刷屏。
                # 注意：台账里的 dormant 列表只列“这次新睡着的”
                # （mark_dormant），待删翻转的那条不进这个列表——它
                # 早就睡着了，这次只是待删翻了个标记。
                if mark_dormant or (mark_delete_due and not was_delete_due):
                    # key 带上：这条原话标 dormant——兜底、筛子按 key 认，
                    # 睡着的原话不会再从账本里被当现况捞回来；唤醒时
                    # revive 再把它标回 current。
                    self._ledger_append(
                        _ledger.LEDGER_EVENT_DORMANT,
                        memory_id=memory_id,
                        delete_due=bool(mark_delete_due),
                        key=_ledger.ledger_event_key(
                            text, meta.get("created_at")
                        ),
                    )
                if mark_dormant:
                    ledger["dormant"].append(
                        {
                            "id": memory_id,
                            "text": text,
                            "weight": decay_weight(kind, last_used_at, now_dt, is_core=False),
                            "idle_days": _decay_idle_days(last_used_at, now_dt),
                        }
                    )
                if mark_delete_due:
                    ledger["delete_due"].append(
                        {
                            "id": memory_id,
                            "text": text,
                            "last_used_at": last_used_at,
                            "idle_days": _decay_idle_days(last_used_at, now_dt),
                        }
                    )
        return ledger

    def delete_due_memories(self, now=None, backup_path=None, approve=False):
        """删满半年没人找的记忆（显式调用，BOOS 亲口说删才删）：
        只删同时满足两条的——从最后一次用上算起满 180 天还没人找、
        不是核心记忆。月扫只打标、不删，删的入口只在这里，默认不开。
        删前先写备份（原话 + 原始 metadata，全放一个 JSON 里，
        临时文件 + 原子替换）：备份写失败，一个也不删。备份好了再一条
        一条调 delete_memory（approve=True，走深度删除的老路：向量 +
        history 审计行都清、验三处都干净才算删完）。返回台账
        {"deleted", "backup_path", "kept", "errors"}。
        approve=False 时抛 PermissionError（删记忆必须 BOOS 同意，
        和 delete_memory 一样的规矩）。"""
        if not approve:
            raise PermissionError("删除记忆需要 BOOS 同意：请先确认再删。")
        now_dt = _decay_now_dt(now)
        candidates = []
        for item in self.get_all(include_sensitive=True):
            memory_id = (item or {}).get("id")
            meta = (item or {}).get("metadata") or {}
            if not memory_id:
                continue
            if _is_core_memory(meta):
                continue
            if not decay_should_delete(
                meta.get("last_used_at"),
                now_dt,
                is_core=False,
            ):
                continue
            candidates.append(item)
        if not candidates:
            return {"deleted": [], "backup_path": None, "kept": [], "errors": []}
        backup_entries = []
        for item in candidates:
            memory_id = item["id"]
            backup_entries.append(
                {
                    "id": memory_id,
                    "summary": item.get("text") or "",
                    # 文本从没被换过（月扫只打标、不压摘要），库里的文本就是原话。
                    "full_text": item.get("text") or "",
                    "metadata": self._raw_metadata_for(memory_id) or {},
                    "delete_at": _decay_iso(now_dt),
                }
            )
        if backup_path is not None:
            path = Path(backup_path)
        else:
            path = (
                Path(self.chroma_path).parent
                / (
                    DECAY_DELETE_BACKUP_PREFIX
                    + now_dt.strftime("%Y%m%d-%H%M%S")
                    + ".json"
                )
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(backup_entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as exc:
            raise RuntimeError(
                f"待删备份写失败，本次一个没删（{exc}）。"
            ) from exc
        deleted = []
        kept = []
        errors = []
        for item in candidates:
            memory_id = item["id"]
            try:
                if self.delete_memory(memory_id, approve=True):
                    deleted.append(memory_id)
                else:
                    kept.append({"id": memory_id, "reason": "库里已没有"})
            except Exception as exc:
                kept.append({"id": memory_id, "reason": str(exc)})
                errors.append({"id": memory_id, "reason": str(exc)})
        return {
            "deleted": deleted,
            "backup_path": str(path),
            "kept": kept,
            "errors": errors,
        }

    def _dedupe_extraction_results(self, results, before_items, turn_id=None,
                                   batch_metadata=None):
        """提取路径去重（46 号）：新提取的记忆里，库里已有同一件事的删掉。

        顺序：先精确匹配（完全相同），再近似匹配（向量预筛 + 模型判同一件事）。
        对照面是两部分：提取前的库（before_items）+ 本轮已留下的新记忆
        （同一轮里模型抽出两条同一件事时也只留一条）。
        删掉的条数记在 results["dedup_dropped"] 里。判不出来时只做精确匹配，
        近似的照常留（宁漏勿错）。删除失败的（还在库里）留在 results 里，
        不能假装删掉了。
        batch_metadata：本轮提取共用的那份 metadata（推断来源）——
        金额同槽位新值要不要顶掉旧条，先拿它跟旧条过一遍
        _resolve_conflict（补丁 2 两道闸）；不传时按推断来源现拼一份。
        """
        items = (results or {}).get("results") or []
        if not items:
            return results
        known = list(before_items or [])
        embed_cache = {}
        dropped = []
        kept = []
        superseded_count = 0
        held_count = 0
        if batch_metadata is None:
            batch_metadata = _provenance(
                SOURCE_INFERRED, CONFIDENCE_INFERRED, False, turn_id=turn_id
            )
        for item in items:
            mem_text = (item or {}).get("memory") or ""
            mem_id = (item or {}).get("id")
            if not mem_text or not mem_id:
                kept.append(item)
                continue
            # 53 号：金额记忆先按（对象，周期）槽位归并，再判。
            # 同槽位新值直接更新旧条（不经过模型判——向量相似度分不开
            # “5 块”和“10 块”，交给模型判可能把新值当重复丢掉）；
            # 不同槽位的排除在近重复候选之外（同主题不同用途，分别保留）。
            new_money = _extract_money_metadata(mem_text)
            supersede_target = None
            slot_duplicate = False
            judge_pool = []
            for known_item in known:
                relation = _money_slot_relation(new_money, _known_money(known_item))
                if relation == "supersede" and supersede_target is None:
                    supersede_target = known_item
                elif relation == "duplicate":
                    slot_duplicate = True
                elif relation == "different":
                    continue
                judge_pool.append(known_item)
            if slot_duplicate:
                # 同槽位已有相同金额的那条：新条就是重复，丢掉。
                dropped.append(item)
                continue
            if supersede_target is not None:
                # 补丁 2 两道闸：金额同槽位新值先判可信度，再决定顶不顶。
                # 守卫已经标过 pending_confirmation 的那条，闸也一样——
                # 守卫在前、去重在后，去重不能把守卫的决定翻过来。
                # 新说法可信度比旧条低（提取的推断 vs BOOS 确认过的），
                # 或两条分不出先后（同一批里先后抽出的两条）时，不顶掉
                # 旧条：新条挂起来待确认（标待确认 + 冲突待审、跟旧条
                # 连一条冲突关系），旧条原封不动，等 BOOS 真用上时再定。
                resolution = None
                if (item or {}).get("pending_confirmation"):
                    resolution = {
                        "action": "pending_new",
                        "reason": "提取后守卫已标待确认",
                    }
                else:
                    resolution = _resolve_conflict(
                        (supersede_target or {}).get("metadata") or {},
                        batch_metadata,
                    )
                if resolution.get("action") in ("pending_new", "ask"):
                    try:
                        self._memory.update(
                            memory_id=mem_id,
                            text=mem_text,
                            metadata={
                                **batch_metadata,
                                "pending_confirmation": True,
                            },
                        )
                    except Exception:
                        logger.warning(
                            "金额去重：待确认标记没打上，新条照常留着。",
                            exc_info=True,
                        )
                    self._mark_memory_review(mem_id, conflict_for_review=True)
                    try:
                        self.relate_memories(
                            mem_id,
                            supersede_target.get("id"),
                            _relations.RELATION_CONFLICT,
                            note="金额新值可信度不够，盖不掉已确认的旧值，等 BOOS 定。",
                        )
                    except Exception:
                        logger.warning(
                            "金额去重：冲突关系没连上，新条照常留着。",
                            exc_info=True,
                        )
                    kept.append(
                        {
                            **item,
                            "pending_confirmation": True,
                            "held_pending": True,
                        }
                    )
                    known.append({"id": mem_id, "text": mem_text})
                    held_count += 1
                    continue
                updated = self._supersede_money_memory(
                    supersede_target, mem_text, new_money
                )
                if updated is not None:
                    supersede_target.update(updated)
                    kept.append(
                        {
                            "id": supersede_target.get("id"),
                            "memory": mem_text,
                            "event": "UPDATE",
                        }
                    )
                    superseded_count += 1
                    dropped.append(item)
                    continue
                # 更新失败：新条照常留（别丢事实），也不再拿去给模型判重复
                # （模型可能把它当重复丢掉——新值就真没了）。
                kept.append(item)
                known.append({"id": mem_id, "text": mem_text})
                continue
            # 精确：和已知的某条完全相同（只差标点/空白/大小写）。
            if _find_duplicate_memory(mem_text, known) is not None:
                dropped.append(item)
                continue
            # 近似：宽召回 + 按意思裁决（判断 a 号，不同槽位的已排除在外）。
            # 精确匹配上面已经查过，这里只处理近似。
            scored = self._score_candidates(mem_text, judge_pool, embed_cache)
            verdict = (
                self._adjudicate_duplicate(mem_text, scored)
                if scored
                else {
                    "verdict": "new", "confidence": 0.0,
                    "reason": "没有候选", "candidate": None, "judged": False,
                }
            )
            if verdict["verdict"] == "duplicate":
                # 修 3（2026-11-20 牛肉面事故的防守）：守卫已挂起的条不能删——
                # 像金额分支那样留着、不挂证据、不进 dropped，等 BOOS 定。
                if (item or {}).get("pending_confirmation"):
                    kept.append(
                        {
                            **item,
                            "pending_confirmation": True,
                            "held_pending": True,
                        }
                    )
                    known.append({"id": mem_id, "text": mem_text})
                    continue
                # 重复：新条当证据挂到旧条下（旧条是主条——BOOS 亲口说的
                # 留主条，提取出来的当证据），新条删掉。
                canonical = verdict["candidate"]
                if canonical is not None:
                    self._append_duplicate_evidence(
                        canonical, mem_text,
                        new_source=SOURCE_INFERRED,
                        new_confidence=None,
                    )
                    # 第二步衰减：同一件事又出现一次，记一笔“不同天出现”
                    # （裁决已经说了这是重复、不是冲突——先标冲突、再强化）。
                    try:
                        self._record_decay_occurrence(
                            canonical.get("id"), turn_id=turn_id
                        )
                    except Exception:
                        logger.warning(
                            "衰减出现次数写回失败：记忆本身还在。", exc_info=True
                        )
                dropped.append(item)
                continue
            if verdict["verdict"] == "conflict":
                # 槽位值冲突：不是重复，新条留着，标 conflict_for_review，
                # 等 BOOS 看（比如喜欢吃鸡杂面 vs 对鸡杂面过敏）。
                self._mark_memory_review(mem_id, conflict_for_review=True)
                candidate = verdict.get("candidate") or {}
                if candidate.get("id"):
                    # 账本记一条冲突关系：两条都留着，谁也没顶掉谁。
                    self.relate_memories(
                        mem_id,
                        candidate["id"],
                        _relations.RELATION_CONFLICT,
                        note="提取去重：同一件事、说法对着干，等 BOOS 定。",
                    )
                kept.append(item)
                known.append({"id": mem_id, "text": mem_text})
                continue
            # 新事实：模型裁决过、但把握不够（confidence <0.80 或判失败）
            # 时，新条留着，标 needs_review=true，等兜底清理时再看。
            # 模型明确说不是重复（confidence ≥0.80）或根本没候选时不标。
            if verdict.get("judged") and verdict["confidence"] < 0.80:
                self._mark_memory_review(mem_id, needs_review=True)
            kept.append(item)
            known.append({"id": mem_id, "text": mem_text})
        if held_count:
            results["money_held_pending"] = held_count
        if not dropped:
            # 没删掉任何东西时也别忘了把 kept 写回去：挂起的那条在
            # kept 里，带了 pending_confirmation / held_pending 标记，
            # 原样返回 results 的话调用方看不见它被挂起了。
            results["results"] = kept
            return results
        survivors = self._rollback_new_memories(dropped)
        survivor_ids = {(item or {}).get("id") for item in survivors}
        actually_dropped = [
            item for item in dropped if (item or {}).get("id") not in survivor_ids
        ]
        results["results"] = kept + survivors
        results["dedup_dropped"] = len(actually_dropped)
        if superseded_count:
            results["money_superseded"] = superseded_count
        if held_count:
            results["money_held_pending"] = held_count
        if survivors:
            logger.warning("提取去重：%d 条重复记忆没删掉（仍在库里）。", len(survivors))
        return results

    def _ledger_record_add(self, text, metadata, or_raise=False):
        """给一条新原话记 add 事件进账本（原话进档案）。

        or_raise=True 时写失败就抛——“先写账本再写索引”的那一步
        （add_memory_text）；否则写失败只记日志、不拦路（种子、
        规则、纠正后的新说法这类“索引里已经有了”的补记）。
        meta 只放标量字段（来源/置信度/是否确认/时间/写序号/轮次/槽位/
        是否规则/衰减种类/重要性），和 add 事件的档案格式一致。
        返回记好的事件，写失败返回 None（or_raise 时抛异常）。
        图谱推理第一期：add 事件落盘后，顺手把这条记忆接进图
        （抽实体、连边、判关系、写快照）——图是索引，失败只记日志、
        不拦路（_graph_add_memory 内部全包住了）。
        """
        meta = metadata or {}
        if or_raise:
            event = self._ledger_append_or_raise(
                _ledger.LEDGER_EVENT_ADD,
                key=_ledger.ledger_event_key(text, meta.get("created_at")),
                text=text,
                meta={
                    "source": meta.get("source"),
                    "confidence": meta.get("confidence"),
                    "user_confirmed": meta.get("user_confirmed"),
                    "created_at": meta.get("created_at"),
                    "write_seq": meta.get("write_seq"),
                    "turn_id": meta.get("turn_id"),
                    "slot": meta.get("slot"),
                    "is_rule": meta.get("is_rule", False),
                    "decay_kind": meta.get("decay_kind"),
                    "importance": meta.get("importance"),
                },
            )
        else:
            event = self._ledger_append(
                _ledger.LEDGER_EVENT_ADD,
                key=_ledger.ledger_event_key(text, meta.get("created_at")),
                text=text,
                meta={
                    "source": meta.get("source"),
                    "confidence": meta.get("confidence"),
                    "user_confirmed": meta.get("user_confirmed"),
                    "created_at": meta.get("created_at"),
                    "write_seq": meta.get("write_seq"),
                    "turn_id": meta.get("turn_id"),
                    "slot": meta.get("slot"),
                    "is_rule": meta.get("is_rule", False),
                    "decay_kind": meta.get("decay_kind"),
                    "importance": meta.get("importance"),
                },
            )
        if event is not None:
            self._graph_add_memory(
                event.get("key"),
                text,
                meta,
                event.get("seq") if isinstance(event.get("seq"), int) else None,
            )
        return event

    def _ledger_ensure_add(self, text, metadata, key):
        """删之前先确认账本里有这条的 add 档案：没有就补记一笔
        （2026-11-16 真机测试，拉面根子 2）。

        账本落地之前存的老记忆没有 add 记录，删的时候只记了 delete，
        _deleted_entries 认不出来——删了进不了已删列表、恢复不了。
        补记的 key 和 delete 用的是同一个（text + mem0 metadata 的
        created_at，ledger_event_key），删和记能对上；meta 用原条目
        的那份，来源/确认状态/形成时间原样。已有 add 时什么都不做。
        补记失败只记日志、不拦路——删本身还要继续，缺这一笔只是
        恢复时还认不出来，和以前一样，不比以前差。
        """
        if not key or not (text or "").strip():
            return
        try:
            state = self.ledger_state()
        except Exception:
            state = {}
        key_states = (state or {}).get("key_states") or {}
        if key in key_states:
            return
        if any(
            (record or {}).get("key") == key
            for record in (state or {}).get("adds") or []
        ):
            return
        try:
            self._ledger_record_add(text, metadata or {}, or_raise=False)
        except Exception:
            logger.warning("删前补记 add 失败：key=%s。", key, exc_info=True)

    def _add_text_with_metadata(self, text, metadata, ledger_mode=None):
        """原文存一条记忆（infer=False），带上来源元数据。

        只做“调 mem0.add”这一件事：脱敏、去重由调用方先做好。
        返回前过一遍提取后二次过滤（_drop_sensitive_results）：
        底层 add 的返回里万一混进一条含敏感信息的记忆（比如
        实现 bug 多存了一条），立刻删掉、不留库里——Plan C 的
        槽位 upsert 也走这里，和提取路径同一道防线。
        mem0 报错时记日志、返回 {}，不抛异常（存记忆失败不能把
        聊天搞崩，见 add_turn 的约定）。
        ledger_mode：这次写要不要记账本——None 不记（从账本/备份/
        回收站重建时，那条原话的账本记录已经有了，再记就重复）；
        "best_effort" 记、记失败只记日志不拦路（种子、规则、迁移
        这类新记忆）；"required" 先记账本、记失败就抛（add_memory_text：
        账本写不进去，索引那步也别写）。
        """
        if ledger_mode:
            if ledger_mode == "required":
                self._ledger_record_add(text, metadata, or_raise=True)
            elif ledger_mode == "best_effort":
                self._ledger_record_add(text, metadata)
            else:
                raise ValueError(f"ledger_mode 不对：{ledger_mode!r}。")
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

    # ---------- 判断 (b) 号：动态数字事实的入库门禁 ----------

    def verify_numeric_fact(self, text, sources=None):
        """动态数字事实的入库门禁（判断 b 号）：这句话要不要当稳定事实存。

        text：要存的记忆原文。sources：这条数的来源（可选），每条是
        {"url": 来源链接, "number": 数的文本, "value": 数值,
        "kind": "price"/"percent"/"points", "date": "YYYY-MM-DD"}——
        一般是本轮搜索结果里 dated_numbers 配上 url 的那几条。
        返回 {"is_dynamic_numeric": bool, "status": "verified" /
        "unverified" / None, "confidence_cap": float / None,
        "requires_confirmation": bool, "expires_at": str / None,
        "source_count": int, "consensus": bool}。
        不是动态数字事实时 is_dynamic_numeric=False，其余为 None/False
        （门禁不拦，正常存——用户说的稳定数、偏好、决定都走这里）。
        是动态数字事实时：
        - 两个独立来源对得上（注册域名不同、数在 tolerance 内一致）→
          status="verified"，可信度封顶 0.75，expires_at 按类别定。
        - 只有一家来源，但是权威来源（官方交易所、政府气象局、央行）→
          status="verified"，可信度封顶 0.85。
        - 没来源、只有一家普通来源、两家对不上 → status="unverified"，
          可信度封顶 0.35，requires_confirmation=True，不进稳定事实层。
        气温 tolerance ±1°C，收盘价精确到小数点后两位。
        纯函数（不读库、不写库），调用方拿着 verdict 再决定怎么存。
        """
        from tools.date_numbers import (
            dynamic_numeric_fact_kind,
            extract_dated_numbers,
            is_dynamic_numeric_fact,
            resolve_requested_date,
        )
        from tools.search import (
            check_numeric_consensus,
            is_authoritative_domain,
            registrable_domain,
        )

        not_dynamic = {
            "is_dynamic_numeric": False,
            "status": None,
            "confidence_cap": None,
            "requires_confirmation": False,
            "expires_at": None,
            "source_count": 0,
            "consensus": False,
        }
        if not is_dynamic_numeric_fact(text):
            return not_dynamic
        kind = dynamic_numeric_fact_kind(text) or "other"
        expiry_hours = VERIFICATION_EXPIRY_HOURS.get(kind, 24)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
        ).isoformat()
        # 目标日期：事实原文里配上日期的那个数是哪天的。
        # 显式日期（11 月 11 日）从 extract_dated_numbers 拿；
        # 相对日期（今天/昨天）它不认，用 resolve_requested_date 换算。
        dated = [
            entry for entry in extract_dated_numbers(text)
            if entry.get("date") is not None
        ]
        target_date = dated[0]["date"] if dated else None
        if target_date is None:
            resolved = resolve_requested_date(text)
            if resolved is not None:
                target_date = resolved.isoformat()
        # 来源按注册域名去重（finance.yahoo.com 和 yahoo.com 是一家）。
        domains = []
        for src in sources or []:
            domain = registrable_domain((src or {}).get("url") or "")
            if domain and domain not in domains:
                domains.append(domain)
        source_count = len(domains)
        consensus = False
        if target_date and len(sources or []) >= 2:
            tolerance = 1.0 if kind == "weather" else 0.01
            results = [
                {
                    "url": (src or {}).get("url") or "",
                    "dated_numbers": [
                        {
                            "number": (src or {}).get("number"),
                            "value": (src or {}).get("value"),
                            "kind": (src or {}).get("kind") or "price",
                            "date": (src or {}).get("date"),
                        }
                    ]
                    if (src or {}).get("date")
                    else [],
                }
                for src in sources or []
            ]
            verdict = check_numeric_consensus(results, target_date, tolerance=tolerance)
            consensus = bool(verdict.get("consensus"))
        if consensus:
            return {
                "is_dynamic_numeric": True,
                "status": "verified",
                "confidence_cap": VERIFICATION_CAP_VERIFIED,
                "requires_confirmation": False,
                "expires_at": expires_at,
                "source_count": source_count,
                "consensus": True,
            }
        # 一家权威来源（官方交易所、政府气象局、央行）单独就能验证。
        if source_count == 1 and is_authoritative_domain(domains[0]):
            return {
                "is_dynamic_numeric": True,
                "status": "verified",
                "confidence_cap": VERIFICATION_CAP_AUTHORITATIVE,
                "requires_confirmation": False,
                "expires_at": expires_at,
                "source_count": 1,
                "consensus": False,
            }
        return {
            "is_dynamic_numeric": True,
            "status": "unverified",
            "confidence_cap": VERIFICATION_CAP_UNVERIFIED,
            "requires_confirmation": True,
            "expires_at": expires_at,
            "source_count": source_count,
            "consensus": False,
        }

    def add_memory_text(self, text, layers=None, source="user", turn_id=None):
        """手动存一条记忆原文（CLI“记住：内容”用，不经过提取）。

        参数 text：记忆内容原文；layers：保留给以后用；
        turn_id：这条记忆是哪一轮里写进去的（add_turn 传进来；
        网页“记住”那路传本轮的 id）；不传时只按日期数。
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
                # 第二步衰减：同一件事又被提到一次，记一笔“不同天出现”
                # （去重已经判了这是同一条，不是冲突——先标冲突、再强化）。
                # 记失败也不拦路：记忆本身还在，只是这次没数上。
                try:
                    self._record_decay_occurrence(duplicate["id"], turn_id=turn_id)
                except Exception:
                    logger.warning("衰减出现次数写回失败：记忆本身还在。", exc_info=True)
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
            # 53 号：金额记忆先按（对象，周期）槽位归并，再判近重复。
            # 同槽位同金额是重复（返回已有那条）；同槽位新金额是更新
            # （旧条原地改成新说法，id 不变）；不同槽位分别保留，
            # 不拿去给模型判重复（向量相似度分不开“万家纳指”和“沪深300”）。
            new_money = _extract_money_metadata(safe_text)
            # 27 号：带金额的记忆，把结构化金额字段贴上（金额/币种/周期/对象），
            # 回答时 _memory_block 标在每条后面，模型分得清两笔钱；没金额
            # 就不传 money 参数（不往 payload 里写 money=None 的空字段）。
            # 判断 (b) 号：动态数字事实入库先过门禁——没印证的数只记成
            # 未核实（可信度封顶 0.35），须确认才当稳定事实用，带有效期。
            # 用户亲口说的稳定数（定投金额、偏好）不受影响，门禁只拦
            # 动态数字（收盘价、气温、汇率……）。
            # 这份 metadata 提前拼好：下面的金额同槽位分支要先拿它跟旧条
            # 过一遍冲突裁决（补丁 2 两道闸），再决定顶不顶。
            money = new_money
            confidence = CONFIDENCE_USER if source == SOURCE_USER else CONFIDENCE_INFERRED
            verification_status = None
            requires_confirmation = False
            expires_at = None
            verdict = self.verify_numeric_fact(safe_text)
            if verdict["is_dynamic_numeric"]:
                confidence = min(confidence, verdict["confidence_cap"])
                verification_status = verdict["status"]
                requires_confirmation = verdict["requires_confirmation"]
                expires_at = verdict["expires_at"]
            # 补丁 3 语气：本地筛出玩笑/测试/假设的，存的时候就打标；
            # 筛不出来的（tone 是 None），存好后排队等大模型后台判。
            tone, tone_evidence = self._local_tone_for(safe_text)
            metadata = _provenance(
                source,
                confidence,
                source == SOURCE_USER,
                money=money if money is not None else _MONEY_UNSET,
                verification_status=verification_status,
                requires_confirmation=requires_confirmation,
                expires_at=expires_at,
                turn_id=turn_id,
                tone=tone,
                tone_evidence=tone_evidence,
                tone_source="local" if tone is not None else None,
            )
            existing = self.get_all()
            supersede_target = None
            slot_duplicate = None
            judge_pool = []
            for known_item in existing:
                relation = _money_slot_relation(new_money, _known_money(known_item))
                if relation == "duplicate" and slot_duplicate is None:
                    slot_duplicate = known_item
                elif relation == "supersede" and supersede_target is None:
                    supersede_target = known_item
                elif relation == "different":
                    continue
                judge_pool.append(known_item)
            if slot_duplicate is not None:
                logger.info("手动存记忆：同槽位已有相同金额的记忆，本次未重复存。")
                # 第二步衰减：同槽位同金额又被提到一次，记一笔“不同天出现”。
                try:
                    self._record_decay_occurrence(slot_duplicate["id"], turn_id=turn_id)
                except Exception:
                    logger.warning("衰减出现次数写回失败：记忆本身还在。", exc_info=True)
                return {
                    "results": [
                        {
                            "id": slot_duplicate["id"],
                            "memory": slot_duplicate["text"],
                            "event": "NONE",
                        }
                    ],
                    "duplicate": True,
                }
            if supersede_target is not None:
                # 补丁 2 两道闸：先判可信度，再决定顶不顶。
                # BOOS 亲口说的（source=user、已确认）顶掉旧条没问题；
                # 非用户来源的新说法可信度比旧条低时，不顶掉旧条：新条
                # 另存一条，标待确认 + 冲突待审，旧条原封不动，等 BOOS
                # 真用上时再定（见 list_pending_decisions）。
                resolution = _resolve_conflict(
                    (supersede_target or {}).get("metadata") or {}, metadata
                )
                if resolution.get("action") in ("pending_new", "ask"):
                    held_metadata = {
                        **metadata,
                        "pending_confirmation": True,
                        "conflict_for_review": True,
                    }
                    try:
                        held = self._add_text_with_metadata(
                            safe_text, held_metadata, ledger_mode="required"
                        )
                    except Exception:
                        logger.warning(
                            "手动存记忆：待确认的新说法没存下来。", exc_info=True
                        )
                        return {}
                    held_results = (held or {}).get("results") or []
                    new_id = (
                        held_results[0].get("id") if held_results else None
                    )
                    if new_id:
                        try:
                            self.relate_memories(
                                new_id,
                                supersede_target.get("id"),
                                _relations.RELATION_CONFLICT,
                                note="手动存的金额新值可信度不够，盖不掉已确认的旧值，等 BOOS 定。",
                            )
                        except Exception:
                            logger.warning(
                                "手动存记忆：冲突关系没连上，新条照常留着。",
                                exc_info=True,
                            )
                    logger.info(
                        "手动存记忆：新说法可信度不够，旧条没动，新条已挂待确认。"
                    )
                    return {
                        "results": [
                            {
                                "id": new_id,
                                "memory": safe_text,
                                "event": "ADD",
                            }
                        ],
                        "held_pending": True,
                    }
                updated = self._supersede_money_memory(
                    supersede_target, safe_text, new_money
                )
                if updated is not None:
                    logger.info("手动存记忆：同槽位新金额，旧条已更新成新说法。")
                    return {
                        "results": [
                            {
                                "id": updated["id"],
                                "memory": safe_text,
                                "event": "UPDATE",
                            }
                        ],
                        "superseded": True,
                    }
                # 更新失败：继续往下存新条，别丢事实。
            # 46 号：近似去重——说法不同、事实相同也算重复（比如“助手叫十二”
            # 和“你的名字是十二”）。先按向量相似度预筛，再请模型判是不是
            # 同一件事（只看相似度分不开：5 块和 10 块的相似度比真重复还高）。
            # 判是重复时不存新条，返回已有那条；判不出来时照常存（宁漏勿错）。
            near = self._find_near_duplicate_memory(safe_text, judge_pool)
            if near is not None:
                logger.info("手动存记忆：库里已有同一件事的记忆，本次未重复存。")
                # 第二步衰减：同一件事又被提到一次，记一笔“不同天出现”
                # （模型已经裁决了这是同一件事、不是冲突——先标冲突、再强化）。
                try:
                    self._record_decay_occurrence(near["id"], turn_id=turn_id)
                except Exception:
                    logger.warning("衰减出现次数写回失败：记忆本身还在。", exc_info=True)
                return {
                    "results": [
                        {
                            "id": near["id"],
                            "memory": near["text"],
                            "event": "NONE",
                        }
                    ],
                    "duplicate": True,
                }
            # metadata 已在上面拼好（金额分支要先拿它过冲突裁决）。
            # 先写账本、再写索引（最终方案）：账本是家，mem0 只是索引。
            # 账本写不进去时这次不写索引、返回 {}——写记忆失败不能把
            # 聊天搞崩，但也不能让索引里多一条账本里没有的记忆
            # （对账时认不出来）。记账本在 _add_text_with_metadata 里
            # （ledger_mode="required"，记失败就抛、这里接住返回 {}）。
            try:
                stored = self._add_text_with_metadata(
                    safe_text, metadata, ledger_mode="required"
                )
            except Exception:
                logger.warning("账本先写失败：本次不写索引。", exc_info=True)
                return {}
            # 补丁 3：本地没筛出语气的，排队等大模型后台判。
            if tone is None and stored:
                stored_results = (stored or {}).get("results") or []
                if stored_results and stored_results[0].get("id"):
                    self._tone_queue_append(stored_results[0]["id"], safe_text)
            return stored

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
            # 迁移进来的也是新记忆：记账本（best_effort），不然账本
            # 里没有它们，对账重建、兜底都找不到这些老记忆。
            result = self._add_text_with_metadata(
                text,
                _provenance(
                    SOURCE_USER,
                    CONFIDENCE_USER,
                    True,
                    created_at=created_at,
                    created_at_source="migrated",
                ),
                ledger_mode="best_effort",
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

    def backfill_timestamps(self):
        """给没有时间戳的老记忆补时间戳（第一步的一次性迁移）。

        只补缺的：created_at 不是非空字符串的才补；补的时候时间戳用
        新鲜的单调时间（_now_iso），created_at_source 记 "backfilled"——
        诚实标明这是后补的，不是原来就有的；其他 metadata（来源、
        置信度、是否确认、槽位）原样保留，绝不因为补时间就把可信度
        抬上去（后补的时间戳不能让一条没确认的记忆变成已确认）。
        幂等：补过的不会再补，重复调也没事。失败只记日志、不抛异常，
        返回补了几条（-1 表示连库都读不出来）。
        网页服务启动时调一次（web/server.py）；命令行“记住”不调——
        后补是一次性的事，服务启动时补完就够了，平时不用反复跑。
        33 号修过的两点：
        - 查时间要看条目顶层：mem0 读回来时把 created_at 提到顶层，
          metadata 里没有——以前只查 metadata，永远以为“缺时间”，
          每次启动把所有记忆都“补”一遍，白白把 created_at_source
          改写成 "backfilled"（来源标错了）。现在顶层有了就跳过。
        - 真缺时间的（顶层也没有）：mem0 的 update() 会保留 payload
          里原有的 created_at，缺了就是 None，传什么都写不进去——
          所以这种不补，只记个数；重存会丢 id，不值得。面板上照实
          显示“未知时间”。实际上写入口都带时间、mem0 的 add 缺了也
          会自己补一个，顶层没有的情况本就不该出现。
        """
        with _WRITE_LOCK:
            try:
                raw = self._fetch_all_raw()
            except Exception:
                logger.warning("补时间戳：读记忆库失败，本次不补。", exc_info=True)
                return -1
            items = (raw or {}).get("results") or []
            backfilled = 0
            skipped_missing = 0
            for item in items:
                mem_id = (item or {}).get("id")
                text = (item or {}).get("memory") or ""
                if not mem_id or not text:
                    continue
                # 33 号：先看条目顶层——mem0 把 created_at 提到顶层了，
                # 顶层有就说明 payload 里有，不用补。
                top_created = (item or {}).get("created_at")
                if isinstance(top_created, str) and top_created:
                    continue
                raw_meta = (item or {}).get("metadata")
                if (
                    isinstance(raw_meta, dict)
                    and isinstance(raw_meta.get("created_at"), str)
                    and raw_meta.get("created_at")
                ):
                    continue
                # 顶层和 metadata 里都没有：mem0 的 update() 保留 payload
                # 原有的 created_at，缺了写不进去，不补、只记数。
                skipped_missing += 1
                logger.warning(
                    "补时间戳：记忆 %s 没有形成时间，update 写不进去，"
                    "本次不补（面板上照实显示未知时间）。",
                    mem_id,
                )
                continue
            if skipped_missing:
                logger.info("补时间戳：%d 条记忆没有形成时间，未补。", skipped_missing)
            return backfilled

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

    def _judge_json(self, system_prompt, user_prompt, max_tokens=1000):
        """请 DeepSeek 做一次 JSON 判断（第 3 批四个判共用的小助手）。

        撤回判意图（34 号）、去重判同一件事（46 号）、价值打分（43 号）、
        纠正提炼规则（42 号）都是“一问一答、只回 JSON”——共用这一套：
        temperature=0、思考保持开启（和 _judge_correction 一路）、
        第一次被截断（finish_reason == "length"）时用 2000 再试一次。
        返回解析好的 dict；没 LLM 客户端、调用失败、截断两次、回的不是
        JSON 时返回 None——调用方按“判不出来”失败关闭（不猜、不拦路），
        绝不抛异常。日志里只记“判了/没判出来”，不记提示和回答原文。
        """
        try:
            client = self._memory.llm.client
        except Exception:
            logger.info("模型判断：没有 LLM 客户端，本次不判断。")
            return None
        max_tokens_try = max_tokens
        for attempt in (1, 2):
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=max_tokens_try,
                )
                content = (resp.choices[0].message.content or "").strip()
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
            except Exception:
                logger.warning("模型判断：调 DeepSeek 失败。", exc_info=True)
                return None
            if finish_reason == "length":
                if attempt == 1:
                    max_tokens_try = JUDGE_RETRY_MAX_TOKENS
                    continue
                logger.info("模型判断：重试也被截断。")
                return None
            if not content:
                return None
            text = content.strip()
            fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if fence:
                text = fence.group(1).strip()
            try:
                data = json.loads(text)
            except ValueError:
                logger.info("模型判断：返回的不是 JSON。")
                return None
            if isinstance(data, dict):
                return data
            return None
        return None

    def _call_llm_text(self, system_prompt, user_prompt, max_tokens=500):
        """请 DeepSeek 回一句话大白话（揣摩人心用，_judge_json 的纯文本兄弟）。

        和 _judge_json 同一路：走 mem0 的 LLM 客户端（self._memory.llm.client，
        和提取同一路，模型 deepseek-flash，温度 0），用量包装建库时已经
        包住了它的 create，所以这次调用的 Token 也会记进 BudgetTracker；
        没包住（假对象、没传 tracker）时至少记一条日志，不崩。
        不同的是不解析 JSON——揣摩的输出就是一句话（设计方案第八节），
        原样返回调用方解析；第一次被截断（finish_reason == "length"）
        时用 1000 再试一次。没 LLM 客户端、调用失败、截断两次、回空
        时返回 None——调用方按“没揣出来”处理，不猜、不拦路，绝不抛异常。
        """
        try:
            client = self._memory.llm.client
        except Exception:
            logger.info("揣摩：没有 LLM 客户端，本次不揣。")
            return None
        max_tokens_try = max_tokens
        for attempt in (1, 2):
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=max_tokens_try,
                )
                content = (resp.choices[0].message.content or "").strip()
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
            except Exception:
                logger.warning("揣摩：调 DeepSeek 失败。", exc_info=True)
                return None
            if finish_reason == "length":
                if attempt == 1:
                    max_tokens_try = 1000
                    continue
                logger.info("揣摩：重试也被截断。")
                return None
            return content or None
        return None

    def _expand_correction_fragment(self, fragment_value, old_text):
        """把纠正碎片补成一整句（碎片纠正）。

        请模型结合旧记忆补句；补不出来、拿不准、调用失败、补好的句子
        不含碎片里的值时返回 None——调用方走老路（安全网大概率拦住、
        改问 BOOS），绝不硬编一句去改。
        返回补好的整句，或 None。
        """
        if not fragment_value or not old_text:
            return None
        # 值本身就是个含糊指代（“那个”“这个”“它”）时，补不出来——
        # 没有内容可补，硬补只能是把旧值原样抄回来，白改一趟。
        # 直接走老路（问 BOOS），不调模型。
        if fragment_value in _VAGUE_FRAGMENT_VALUES:
            logger.info(
                "碎片补句：值只是个含糊指代（%s），不补句，走老路。",
                fragment_value,
            )
            return None
        # 敏感内容不发给模型判（和 _judge_correction 一路）：直接走老路。
        if _contains_sensitive(old_text) or _contains_sensitive(fragment_value):
            logger.info("碎片补句：旧记忆或碎片里有敏感内容，不补句，走老路。")
            return None
        prompt = (
            "用户在纠正一条旧记忆，但新说法是个碎片（不完整的一句）。\n"
            "请结合旧记忆，把碎片补成一整句完整的新说法。\n"
            f"旧记忆：{old_text}\n"
            f"碎片里的值：{fragment_value}\n"
            "只返回一个 JSON，不要解释，格式："
            '{"expanded": "<补好的整句>"}；补不出来或拿不准时返回 {"expanded": null}。\n'
            "注意：补好的整句必须包含碎片里的值，不能改掉它；不能编造"
            "旧记忆里没有的新信息；只是把碎片补完整，不改变纠正的意思。"
        )
        data = self._judge_json(
            "你是记忆纠错的补句助手，只返回 JSON。", prompt
        )
        if not isinstance(data, dict):
            return None
        expanded = data.get("expanded")
        if not isinstance(expanded, str) or not expanded.strip():
            return None
        expanded = expanded.strip()
        # 补好的句子必须含着碎片里的值——值被改掉/丢了就是补错了，不用。
        if fragment_value not in expanded:
            logger.info("碎片补句：补好的句子不含碎片里的值，不用。")
            return None
        return expanded

    def _judge_retraction_intent(self, user_text, prev_user_text,
                                 assistant_asked_text=None):
        """请 DeepSeek 判断：这句话是不是在撤回上一句（34 号）。

        只认纯撤回——不要上一句了，那句不算、不记了，而且没有补上新的说法。
        以下都不是撤回：纠正（给出了新的正确说法，比如“不对，是绿色”）、
        回答助手刚问的问题、闲聊、换话题、继续说别的事。
        判不出来（没客户端、调用失败、回的不是 JSON）时返回 None——调用方
        按“不是撤回”处理，绝不因为判不出来就把上一轮丢了。
        """
        if not user_text or not prev_user_text:
            return None
        if _contains_sensitive(user_text) or _contains_sensitive(prev_user_text):
            return None
        asked_part = ""
        if assistant_asked_text:
            asked_part = (
                "注意：助手上一句刚问了一个问题——"
                f"“{assistant_asked_text}”；如果这句话只是在回答那个问题，"
                "就不是撤回。\n"
            )
        prompt = (
            "判断下面这句话是不是在撤回上一句说过的话。\n"
            "撤回的意思是：不要上一句了，那句不算、不记了，而且没有补上新的说法。\n"
            "以下都不是撤回：\n"
            "- 纠正：给出了新的正确说法（比如“不对，是绿色”——这是纠正，不是撤回）；\n"
            "- 回答问题、闲聊、换话题、继续说别的事。\n"
            + asked_part +
            "只返回一个 JSON，不要解释，格式：{\"retract\": true 或 false}\n"
            f"上一句：{prev_user_text}\n"
            f"这句话：{user_text}"
        )
        data = self._judge_json(
            "你是撤回意图的判断助手，只返回 JSON。", prompt
        )
        if not isinstance(data, dict):
            return None
        retract = data.get("retract")
        if isinstance(retract, bool):
            return retract
        return None

    def _judge_same_fact(self, new_text, candidates):
        """请 DeepSeek 判断：新记忆和候选里的哪条是同一件事（46 号）。

        同一件事 = 同一个事实、只是说法不同（比如“助手叫十二”和
        “你的名字是十二”）。数字不同就是不同事实（“每天定投 5 块”和
        “每天定投 10 块”不是同一件事）；意思不同也是不同事实
        （“喜欢看科幻电影”和“喜欢看纪录片”不是同一件事）。
        返回挑中的候选原字典，或 None（没有同一件事 / 判不出来）。
        判不出来时调用方按“不是重复”处理——漏合并只是多一条记忆，
        错合并是丢一条事实，宁漏勿错。
        """
        if not new_text or not candidates:
            return None
        for cand in candidates:
            if _contains_sensitive((cand or {}).get("text") or ""):
                return None
        if _contains_sensitive(new_text):
            return None
        lines = []
        for i, cand in enumerate(candidates, start=1):
            lines.append(f"{i}. id={cand.get('id')}：{cand.get('text') or ''}")
        prompt = (
            "判断下面这条新记忆，和候选里的哪一条说的是同一件事。\n"
            "同一件事 = 同一个事实，只是说法不同。\n"
            "数字不同就是不同事实（“每天定投 5 块”和“每天定投 10 块”不是同一件事）；\n"
            "意思不同也是不同事实（“喜欢看科幻电影”和“喜欢看纪录片”不是同一件事）。\n"
            "只返回一个 JSON，不要解释，格式：\n"
            '{"choice": "<候选 id>" 或 "none"}\n'
            f"新记忆：{new_text}\n"
            "候选：\n" + "\n".join(lines)
        )
        data = self._judge_json(
            "你是记忆去重的判断助手，只返回 JSON。", prompt
        )
        if not isinstance(data, dict):
            return None
        choice = data.get("choice")
        if not isinstance(choice, str):
            return None
        choice = choice.strip()
        if choice.lower() in ("none", "不知道", "分不清", ""):
            return None
        for cand in candidates:
            if (cand or {}).get("id") == choice:
                return cand
        return None

    def _judge_available(self):
        """模型裁决能不能调（判断 a 号，硬合并线用）：
        有 LLM 客户端就是能调。硬合并线（相似度 ≥0.90 不请模型）
        只在能调时才走——模型挂了的时候，宁漏勿错（46 号安全网），
        不靠硬合并线直接合并，免得错合并丢事实。"""
        try:
            return self._memory.llm.client is not None
        except Exception:
            return False

    def _judge_duplicate_verdict(self, new_text, candidates):
        """请 Flash 裁决：新记忆和候选里的是不是同一件事（判断 a 号）。

        输入带旧记忆的全文/来源/时间/可信度和新记忆的同样字段，
        输出固定 JSON {"is_duplicate": bool, "confidence": 0–1,
        "reason": "...", "conflict": bool}。
        conflict 为真 = 槽位值冲突（同一件事、说法却对着干，比如一个
        说喜欢吃鸡杂面、一个说对鸡杂面过敏）——不是重复，要留着。
        判不出来（没客户端、调用失败、回的不是 JSON）时返回 None，
        调用方按“新记忆”处理（宁漏勿错）。
        """
        if not new_text or not candidates:
            return None
        for cand in candidates:
            if _contains_sensitive((cand or {}).get("text") or ""):
                return None
        if _contains_sensitive(new_text):
            return None
        lines = []
        for i, cand in enumerate(candidates, start=1):
            meta = _normalize_metadata((cand or {}).get("metadata"), cand)
            lines.append(
                f"{i}. id={cand.get('id')}：{cand.get('text') or ''}"
                f"（来源={meta['source']}，时间={meta['created_at'] or '未知'}，"
                f"可信度={meta['confidence'] if meta['confidence'] is not None else '未知'}）"
            )
        prompt = (
            "判断下面这条新记忆，和候选里的哪一条说的是同一件事。\n"
            "同一件事 = 同一个事实，只是说法不同。\n"
            "数字不同就是不同事实（“每天定投 5 块”和“每天定投 10 块”不是同一件事）；\n"
            "意思不同也是不同事实（“喜欢看科幻电影”和“喜欢看纪录片”不是同一件事）；\n"
            "槽位值冲突也不是重复（“喜欢吃鸡杂面”和“对鸡杂面过敏”是同一话题、"
            "说法对着干，要标 conflict=true）。\n"
            "只返回一个 JSON，不要解释，格式：\n"
            '{"is_duplicate": true/false, "confidence": 0.0-1.0, '
            '"reason": "一句话原因", "conflict": true/false}\n'
            "confidence 是你对这个判断的把握（0–1）。\n"
            f"新记忆：{new_text}\n"
            "候选：\n" + "\n".join(lines)
        )
        data = self._judge_json(
            "你是记忆去重的判断助手，只返回 JSON。", prompt
        )
        if not isinstance(data, dict):
            return None
        is_duplicate = data.get("is_duplicate")
        confidence = data.get("confidence")
        if not isinstance(is_duplicate, bool):
            return None
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return None
        confidence = max(0.0, min(1.0, float(confidence)))
        return {
            "is_duplicate": is_duplicate,
            "confidence": confidence,
            "reason": data.get("reason") or "",
            "conflict": data.get("conflict") is True,
        }

    def _adjudicate_duplicate(self, new_text, scored):
        """按意思裁决：新记忆和候选里的是不是同一件事（判断 a 号）。

        scored：[(相似度， 候选条目)]（已按相似度排好序，最高在前）。
        返回 {"verdict": "duplicate" | "conflict" | "new",
        "confidence": float, "reason": str, "candidate": dict | None}。
        流程：
        1. 槽位过滤：没共同核心槽位、且相似度 <0.55 的候选直接丢掉，
           不送模型判（省一次模型调用）。
        2. 硬合并：剩下的最高分 ≥0.90 且和新记忆没有槽位值冲突时，
           直接算重复（confidence=0.95），不请模型。
        3. 否则请 Flash 裁决（固定 JSON），输入带旧记忆的全文/来源/
           时间/可信度和新记忆的同样字段。
        4. 映射：is_duplicate 且 confidence ≥0.80 → duplicate；
           conflict 为真 → conflict（不是重复，留着让 BOOS 看）；
           其余 → new（模型判不出来/把握不够时，宁漏勿错）。
        钱的记忆不走这里（53 号已按（对象，周期）槽位归并过）。
        """
        if not new_text or not scored:
            return {
                "verdict": "new", "confidence": 0.0,
                "reason": "没有候选", "candidate": None, "judged": False,
            }
        new_slots = extract_core_slots(new_text)
        kept = []
        for sim, item in scored:
            item_text = (item or {}).get("text") or ""
            if not item_text:
                continue
            shared = new_slots & extract_core_slots(item_text)
            if not shared and sim < DEDUP_SLOT_SIMILARITY_FLOOR:
                continue
            kept.append((sim, item))
        if not kept:
            return {
                "verdict": "new", "confidence": 0.0,
                "reason": "没有共同核心槽位的候选", "candidate": None,
                "judged": False,
            }
        kept.sort(key=lambda pair: pair[0], reverse=True)
        top_sim, top_item = kept[0]
        top_text = (top_item or {}).get("text") or ""
        # 硬合并：最高分 ≥0.90 且没有槽位值冲突时，直接算重复，不请模型。
        # 但只在模型能调时才走这条——模型挂了的时候宁漏勿错（46 号），
        # 不靠硬合并线直接合并（见 _judge_available）。
        # 57 号：_slot_values_conflict 现在也认“宾语不同”（冒菜≠拉面），
        # 不同事实不会再被硬合并线误并；同义改写（“喜欢喝冰美式”vs
        # “喜欢冰美式”，宾语都是冰美式）照常走硬合并。
        # 修 4（2026-11-20 牛肉面事故的防守）：新记忆带具体日期、候选最高分
        # 那条没有时，是“事件 vs 喜好”，不是同一件事——硬合并和模型判重复
        # 都要否决，留着。程序规则兜底，不依赖模型每次都判对。
        _new_has_date = self._has_event_date(new_text)
        _top_has_date = self._has_event_date(top_text)
        _dated_event_vs_undated = _new_has_date and not _top_has_date
        if (
            top_sim >= DEDUP_HARD_THRESHOLD
            and not _slot_values_conflict(new_text, top_text)
            and self._judge_available()
            and not _dated_event_vs_undated
        ):
            return {
                "verdict": "duplicate", "confidence": 0.95,
                "reason": f"相似度 {top_sim:.2f} 且没有槽位值冲突，直接合并",
                "candidate": top_item, "judged": False,
            }
        candidates = [item for _, item in kept[:5]]
        verdict = self._judge_duplicate_verdict(new_text, candidates)
        if verdict is None:
            return {
                "verdict": "new", "confidence": 0.0,
                "reason": "模型裁决失败", "candidate": None, "judged": True,
            }
        if verdict["conflict"]:
            return {
                "verdict": "conflict", "confidence": verdict["confidence"],
                "reason": verdict["reason"] or "槽位值冲突",
                "candidate": candidates[0], "judged": True,
            }
        if verdict["is_duplicate"] and verdict["confidence"] >= 0.80:
            if _dated_event_vs_undated:
                return {
                    "verdict": "new", "confidence": 1.0,
                    "reason": "新记忆带具体日期、候选没有：事件 vs 喜好，不是同一件事",
                    "candidate": None, "judged": False,
                }
            return {
                "verdict": "duplicate", "confidence": verdict["confidence"],
                "reason": verdict["reason"] or "模型裁决为重复",
                "candidate": candidates[0], "judged": True,
            }
        return {
            "verdict": "new", "confidence": verdict["confidence"],
            "reason": verdict["reason"] or "模型裁决为不同事实",
            "candidate": None, "judged": True,
        }

    def _score_candidates(self, text, existing_items, embed_cache=None):
        """按意思去重的宽召回（判断 a 号）：新记忆和已有记忆的向量
        相似度，≥ DEDUP_RECALL_THRESHOLD（0.40）的取最高的 20 条，
        按相似度排好序返回 [(相似度， 条目)]。
        完全相同先认（不用 embedding）。embedding 拿不到时返回 []。
        embed_cache（可选）：{归一化文本: 向量}，一次验多条新记忆时共用。
        """
        if not text or not existing_items:
            return []
        cache = embed_cache if isinstance(embed_cache, dict) else {}
        key = _normalize_memory_text(text)
        vec = cache.get(key)
        if vec is None:
            vec = self._embed_text(text)
            if vec is None:
                return []
            cache[key] = vec
        scored = []
        for item in existing_items:
            item_text = (item or {}).get("text") or ""
            if not item_text:
                continue
            item_key = _normalize_memory_text(item_text)
            other_vec = cache.get(item_key)
            if other_vec is None:
                other_vec = self._embed_text(item_text)
                if other_vec is None:
                    continue
                cache[item_key] = other_vec
            sim = _cosine_similarity(vec, other_vec)
            if sim >= DEDUP_RECALL_THRESHOLD:
                scored.append((sim, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:DEDUP_RECALL_TOP_K]

    def _find_near_duplicate_memory(self, text, existing_items,
                                    embed_cache=None):
        """在已有记忆里找和 text 是同一件事的那条（46 号 + 判断 a 号）。

        两步：先按向量相似度宽召回（≥ DEDUP_RECALL_THRESHOLD 0.40，
        取最高的 20 条——以前只看 ≥0.70，换说法多、相似度被拉低的
        真重复捞不回来），再按意思裁决（_adjudicate_duplicate：
        核心槽位过滤 + 硬合并线 + Flash 结构化裁决）。
        找到返回那一项（{"id", "text", ...}），找不到返回 None。
        只在裁决是 "duplicate" 时返回候选——"conflict"（槽位值冲突）
        和 "new" 都返回 None（调用方按“不是重复”处理，宁漏勿错：
        漏合并只是多一条记忆，错合并是丢一条事实）。
        模型判不出来时也返回 None。embedding 拿不到时返回 None——
        这里只做近似，完全相同是 _find_duplicate_memory 的活儿。
        embed_cache（可选）：{归一化文本: 向量}，提取路径一次验多条
        新记忆时共用，同一条文本只算一次。
        钱的记忆不走这里（53 号已按（对象，周期）槽位归并过）。
        """
        if not text or not existing_items:
            return None
        # 完全相同是 _find_duplicate_memory 的活儿，这里先认（不用 embedding）——
        # embedding 拿不到时（比如离线测试的假对象没装 embedding 模型），
        # 精确匹配照样能走，不能因为嵌不出来就连完全相同的都认不出。
        exact = _find_duplicate_memory(text, existing_items)
        if exact is not None:
            return exact
        scored = self._score_candidates(text, existing_items, embed_cache)
        if not scored:
            return None
        verdict = self._adjudicate_duplicate(text, scored)
        if verdict["verdict"] != "duplicate":
            return None
        return verdict["candidate"]

    def _judge_value_scores(self, items):
        """请 DeepSeek 给新提取的记忆打价值分（43 号）。

        items：[{"id":..., "text":...}]。每条打 0–1 分：
        稳定事实/明确偏好/重要决定/以后还会用到的 = 高分；
        一次性请求/临时状态/闲聊/琐事 = 低分；拿不准的往低打。
        返回 {id: 分数}；判不出来（没客户端、调用失败、回的不是 JSON）
        时返回 None——调用方按“全留”处理（提取提示词是第一道防线，
        判不出来不拦路，别把真事实拦在外面）。
        """
        if not items:
            return {}
        for item in items:
            if _contains_sensitive((item or {}).get("text") or ""):
                return None
        lines = []
        for i, item in enumerate(items, start=1):
            lines.append(f"{i}. id={item.get('id')}：{item.get('text') or ''}")
        prompt = (
            "给下面每条新提取的记忆打价值分（0–1 分），判断它值不值得长期记住。\n"
            "高分（0.7 以上）：稳定事实、明确偏好、重要决定、以后还会用到的事；\n"
            "低分（0.3 以下）：一次性请求、临时状态、闲聊、琐事、问完就没用的事；\n"
            "带具体日期/地点的一次性经历是值得记的 episodic 记忆，不是琐事，不打低分；\n"
            "拿不准的往低打。\n"
            "只返回一个 JSON，不要解释，格式：\n"
            '{"scores": [{"id": "<记忆 id>", "score": 0.8}, ...]}\n'
            "候选：\n" + "\n".join(lines)
        )
        data = self._judge_json(
            "你是记忆价值的判断助手，只返回 JSON。", prompt
        )
        if not isinstance(data, dict):
            return None
        scores = data.get("scores")
        if not isinstance(scores, list):
            return None
        out = {}
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            mem_id = entry.get("id")
            score = entry.get("score")
            if mem_id and isinstance(score, (int, float)) and not isinstance(score, bool):
                out[mem_id] = max(0.0, min(1.0, float(score)))
        return out

    def _derive_correction_rule(self, old_text, new_text):
        """从一次纠正里提炼通用规则（42 号）。

        比如旧“每月定投 5 块，一年 1825 块”→新“一个月约 21 个交易日、
        月投约 105 块”，规则是“定投只在交易日扣款，一个月约 21 个交易日”。
        只是改了一个具体事实、背后没有通用规则时（比如“喜欢蓝色”改成
        “喜欢绿色”），返回 None，不存规则。
        返回规则文本（一句话）或 None。提炼失败（没客户端、调用失败、
        回的不是 JSON）时返回 None——纠正本身已经成功了，规则只是添头，
        提炼不出来不影响纠正。
        """
        if not old_text or not new_text:
            return None
        if _contains_sensitive(old_text) or _contains_sensitive(new_text):
            return None
        prompt = (
            "下面是一次纠正：旧说法被改成了新说法。\n"
            "如果这次纠正背后有一条通用规则（能套用到同类问题上的），用一句话写出来。\n"
            "如果只是改了一个具体事实、背后没有通用规则，返回 null。\n"
            "规则要写成以后能直接套用的样子（比如“定投只在交易日扣款，一个月约 21 个交易日”），\n"
            "不要写“这次纠正了某某”这种只针对这一次的话。\n"
            "只返回一个 JSON，不要解释，格式：{\"rule\": \"规则文本\" 或 null}\n"
            f"旧说法：{old_text}\n"
            f"新说法：{new_text}"
        )
        data = self._judge_json(
            "你是纠正规则的提炼助手，只返回 JSON。", prompt
        )
        if not isinstance(data, dict):
            return None
        rule = data.get("rule")
        if not isinstance(rule, str):
            return None
        rule = rule.strip().strip("。！？；，、.!?;,")
        if len(rule) < 4:
            return None
        return rule

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

    def _derive_and_store_rule(self, old_text, new_text):
        """从一次纠正里提炼通用规则并单独存一条规则记忆（42 号）。

        规则 = 一句话，能套用到同类问题上（比如“定投只在交易日扣款，
        一个月约 21 个交易日”）。提炼不出（没有通用规则 / 模型判失败）
        时返回 None，不存。
        规则记忆的来源记 "inferred"、user_confirmed=False、confidence=0.7
        （和提取出的记忆一样），但多一个 is_rule=True 标记——检索时优先
        捞出来，回答同类问题先套用规则再算。
        同一条规则已经存过时不重复存（先精确、再近似判）。
        返回规则记忆的 id，或 None。
        """
        rule = self._derive_correction_rule(old_text, new_text)
        if not rule:
            return None
        existing = self.get_all()
        if _find_duplicate_memory(rule, existing) is not None:
            return None
        if self._find_near_duplicate_memory(rule, existing) is not None:
            return None
        metadata = _provenance(
            SOURCE_INFERRED, CONFIDENCE_INFERRED, False, is_rule=True
        )
        # 规则也是新记忆：记账本（best_effort）。
        result = self._add_text_with_metadata(
            rule, metadata, ledger_mode="best_effort"
        )
        items = (result or {}).get("results") or []
        if items:
            return (items[0] or {}).get("id")
        return None

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
            # 碎片纠正：新说法是个碎片（“是绿色”）时，先结合旧记忆补成
            # 整句（“我喜欢绿色”），再过安全网、再改。补不出来、拿不准
            # 时走老路（下面安全网大概率拦住、改问 BOOS），绝不硬编一句
            # 去改。0.70 的阈值本身不动——补句是绕开它，不是改它。
            fragment_value = _detect_correction_fragment(safe_new)
            if fragment_value is not None:
                expanded = self._expand_correction_fragment(
                    fragment_value, old_stored_text
                )
                if expanded is not None:
                    safe_new = expanded
                    # 补完和旧的一样 = 不用改（比如“是蓝色”纠“我喜欢蓝色”）。
                    if _normalize_memory_text(
                        old_stored_text
                    ) == _normalize_memory_text(safe_new):
                        return {
                            "status": "unchanged",
                            "memory_id": mem_id,
                            "old_text": old_stored_text,
                            "new_text": safe_new,
                            "candidates": [],
                            "reason": "旧内容和新内容一样，不用改。",
                            "layer": layer,
                            "resolution": "unchanged",
                            "audit": _audit(
                                picked_id=mem_id,
                                judge_reason=judge_reason,
                                cands=candidates,
                            ),
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
            # 27 号：新说法带金额就把金额字段贴上；没金额就传 None
            # （显式清掉——旧说法带金额、新说法不带时，旧字段不能留着）。
            new_metadata = _provenance(
                SOURCE_USER,
                CONFIDENCE_USER,
                True,
                slot=old_metadata.get("slot"),
                money=_extract_money_metadata(safe_new),
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
                # 27 号：旧的金额字段也写回去（旧说法带金额时恢复它；
                # 旧说法没金额时传 None，把纠正那步贴上的清掉）。
                try:
                    self._memory.update(
                        mem_id, text=old_stored_text, metadata=_provenance(
                            old_metadata.get("source") or SOURCE_INFERRED,
                            old_metadata.get("confidence")
                            if isinstance(old_metadata.get("confidence"), (int, float))
                            else CONFIDENCE_INFERRED,
                            bool(old_metadata.get("user_confirmed")),
                            created_at=old_metadata.get("created_at"),
                            created_at_source=old_metadata.get("created_at_source")
                            or "written",
                            slot=old_metadata.get("slot"),
                            money=old_metadata.get("money"),
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
            # 42 号：从这次纠正里提炼通用规则（比如“定投只在交易日扣款”），
            # 单独存一条规则记忆——下次回答同类问题时先套用规则再算。
            # 提炼不出来（没有通用规则 / 模型判失败）时不存，纠正本身
            # 已经成功了，规则只是添头。
            rule_id = self._derive_and_store_rule(old_stored_text, safe_new)
            # 账本记一笔取代：错的那句被正确的换掉了（原地替换，id 不变，
            # 旧话留在账本里，以后翻账本还能找回“以前记错的那句”）。
            # key 带上：旧原话的 key 标 superseded，新原话的 key 标
            # current——对账、兜底按 key 认旧话，不会把错的那句复活。
            # 注意：new_metadata 是新编的 created_at（纠正发生在现在），
            # current_key 必须按新时间算，和下面补记的 add 事件是同一个 key。
            old_created_at = (old_metadata or {}).get("created_at")
            self._ledger_append(
                _ledger.LEDGER_EVENT_SUPERSEDE,
                memory_id=mem_id,
                supersedes_id=mem_id,
                old_text=old_stored_text,
                text=safe_new,
                key=_ledger.ledger_event_key(old_stored_text, old_created_at),
                current_key=_ledger.ledger_event_key(
                    safe_new, new_metadata.get("created_at")
                ),
            )
            # 新说法也要进档案：不然账本里只有错的那句的 add 事件，
            # 对账重建、兜底都找不到正确的这句。记失败不拦路——纠正
            # 本身已经成功了，索引里的新说法还在。
            self._ledger_record_add(safe_new, new_metadata)
            return {
                "status": "updated",
                "memory_id": mem_id,
                "old_text": old_stored_text,
                "new_text": safe_new,
                "candidates": [],
                "reason": "",
                "layer": layer,
                "resolution": "updated",
                "rule_id": rule_id,
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
            # 历史项不算：账本天生留着旧原话（“以前记错的那句”），
            # 它能翻出来是档案的功能，不是“旧文本还在索引里”。
            if (hit or {}).get("is_history"):
                continue
            # 图那路的命中也不算：图是账本的索引（图谱推理第一期），
            # 校验时取代事件还没写进账本，账本还认旧原话是“当前”的，
            # 图自然还会把它捡回来——那不是“索引里还有旧文本”，是
            # 图滞后账本一步。校验通过、取代事件写进账本之后，下次
            # 检索图就把它筛掉了。索引里还有没有旧文本，只看语义/
            # 关键词那两路（它们带 mem0 id，图那路没有）。
            if (hit or {}).get("source") == "graph":
                continue
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
                    # 历史项不算：账本天生留着删掉的原话（删了什么、
                    # 什么时候删的，账本里有数），它能翻出来是档案的
                    # 功能，不是“向量里还留着”。
                    if (hit or {}).get("is_history"):
                        continue
                    # 只认同一条记忆：库里还有另一条相同说法的记忆时，
                    # 搜回来的是那条，不是这条——按 id 认，别按文本认。
                    # （2026-11-16 真机测试：两条相同的“我爱吃冒菜”，
                    # 删掉一条后搜还回来另一条，旧的按文本验永远验不
                    # 过，谎报“删除了一半”。）
                    if (hit or {}).get("id") != memory_id:
                        continue
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
            # 账本记一笔删：深度删除是真删（向量 + history 都清了），
            # 但账本里的原话留着——删了什么、什么时候删的，账本里有数，
            # 以后翻账本还能找回“以前有过这条”。
            # key 带上：这条原话标 deleted——兜底、对账按 key 认，
            # 删了的原话不会再被当现况捞回来、也不会被对账复活。
            # 2026-11-16 真机测试（拉面根子 2）：先补记 add——账本落地
            # 之前存的老记忆没有 add 记录，不补这一笔，删了进不了已删
            # 列表、恢复不了。补记的 key 和下面这笔 delete 是同一个。
            delete_key = _ledger.ledger_event_key(
                old_text,
                (owned[memory_id].get("metadata") or {}).get("created_at"),
            )
            self._ledger_ensure_add(
                old_text,
                owned[memory_id].get("metadata") or {},
                delete_key,
            )
            self._ledger_append(
                _ledger.LEDGER_EVENT_DELETE,
                memory_id=memory_id,
                key=delete_key,
            )
            # 内存里也记一笔刚删（提取复活拦截的第一道，见
            # _drop_deleted_text_results；账本是兜底）。时间戳走单调的
            # _now_epoch()：要和待定轮 created_at、账本 ts 在同一条
            # 时间线上比先后，raw datetime.now() 在 Windows 粗时钟下
            # 会落后（2026-11-19 全套 flaky 的根子）。
            norm_old = _normalize_memory_text(old_text)
            if norm_old:
                self._just_deleted_texts[norm_old] = _now_epoch()
            return True

    # ---------- 34 号：撤回软删除（能恢复） ----------

    def _retracted_trash_path(self):
        """撤回回收站文件的路径：跟 chroma 数据同一个目录下
        （memory/retracted_memories.json；测试时跟临时 chroma 走，
        不污染正式库）。文件只放被撤回的记忆（id、原文、元数据、撤回时间）。"""
        return Path(self.chroma_path).parent / RETRACTED_TRASH_FILENAME

    def _read_retracted_trash(self):
        """读回收站文件：没有文件、读不出、不是列表时返回 []，不抛异常。"""
        try:
            raw = json.loads(
                self._retracted_trash_path().read_text(encoding="utf-8")
            )
            return raw if isinstance(raw, list) else []
        except (OSError, ValueError):
            return []

    def _write_retracted_trash(self, entries):
        """写回收站文件（临时文件 + 原子替换，写一半不留坏文件）。
        写失败只记日志，不抛异常——回收站写不进去时，向量已经删了
        也不能把调用方搞崩（最坏是恢复不了，但记忆本身已撤回）。"""
        try:
            path = self._retracted_trash_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(entries, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(path)
        except OSError:
            logger.warning("撤回回收站：写文件失败。", exc_info=True)

    def _soft_delete_same_fact(self, text):
        """找到库里和 text 是同一件事的记忆，软删除它（34 号）。
        只删模型判中的那一条；找不到 / 判不出来时不删任何东西。
        返回删掉的 id 列表（空 = 没删）。
        """
        if not text:
            return []
        near = self._find_near_duplicate_memory(text, self.get_all())
        if near is None:
            return []
        mem_id = (near or {}).get("id")
        if not mem_id:
            return []
        if self.soft_delete_memory(mem_id):
            return [mem_id]
        return []

    def soft_delete_memory(self, memory_id):
        """软删除一条记忆（34 号撤回用）：从向量库删掉，全文记进回收站文件，
        随时能恢复（restore_memory）。

        不需要 BOOS 另外确认——调它之前调用方已经确认过这是撤回
        （模型判过意图，或整句信号词已认出）。
        归属校验：id 不在当前用户的库里时返回 False。删完验读不到才算数。
        返回 True（已软删）/ False（没找到 / 删失败）。
        注意：只删向量，不清 history 审计行——撤回不是“忘掉”，
        审计里还留着这条记忆来过（和纠错的 UPDATE 审计一个道理）。
        """
        if not memory_id:
            return False
        with _WRITE_LOCK:
            owned = {
                item["id"]: item
                for item in self.get_all(include_sensitive=True)
            }
            if memory_id not in owned:
                return False
            item = owned[memory_id]
            trash = self._read_retracted_trash()
            # 已经在回收站里的不重复记。
            # 判断 c 号：软删除的记录带 deleted_at（删掉的时间）和
            # status="deleted"（这条是已删状态），恢复时认它——
            # 面板/清理流程读回收站时，一眼看得出哪条删了、什么时候删的。
            if not any((e or {}).get("id") == memory_id for e in trash):
                trash.append({
                    "id": memory_id,
                    "text": item.get("text") or "",
                    "metadata": item.get("metadata") or {},
                    "retracted_at": _now_iso(),
                    "deleted_at": _now_iso(),
                    "status": "deleted",
                })
                self._write_retracted_trash(trash)
            try:
                self._memory.delete(memory_id)
            except Exception:
                logger.warning("软删除：向量删除失败。", exc_info=True)
                return False
            # 验：读回来不在才算删掉。
            try:
                after = {
                    i["id"] for i in self.get_all(include_sensitive=True)
                }
            except Exception:
                return False
            deleted = memory_id not in after
            if deleted:
                # 账本记一笔删：软删除也是删（能恢复，但当前不在索引里了），
                # 原话在回收站文件和账本里都留着，恢复时再记一笔 revive。
                # key 带上：这条原话标 deleted——兜底、对账按 key 认，
                # 删了的原话不会再被当现况捞回来、也不会被对账复活。
                # 2026-11-16 真机测试（拉面根子 2）：先补记 add——没有
                # add 记录的老记忆，不补这一笔，删了进不了已删列表。
                delete_key = _ledger.ledger_event_key(
                    item.get("text") or "",
                    (item.get("metadata") or {}).get("created_at"),
                )
                self._ledger_ensure_add(
                    item.get("text") or "",
                    item.get("metadata") or {},
                    delete_key,
                )
                self._ledger_append(
                    _ledger.LEDGER_EVENT_DELETE,
                    memory_id=memory_id,
                    key=delete_key,
                )
                # 内存里也记一笔刚删（提取复活拦截的第一道）。
                # 时间戳走单调的 _now_epoch()（见上面那处 _just_deleted_texts 的注释）。
                norm_item = _normalize_memory_text(item.get("text") or "")
                if norm_item:
                    self._just_deleted_texts[norm_item] = _now_epoch()
            return deleted

    def restore_memory(self, memory_id):
        """从回收站恢复一条软删除的记忆（34 号）。

        按回收站里记的原文 + 元数据重新存入（新 id），再把回收站里那条清掉。
        来源/确认状态/形成时间都按原样恢复（时间戳用原来的，别写成恢复时间）。
        返回 True（恢复了）/ False（回收站里没有这条 / 存失败）。
        """
        if not memory_id:
            return False
        with _WRITE_LOCK:
            trash = self._read_retracted_trash()
            entry = next(
                (e for e in trash if (e or {}).get("id") == memory_id), None
            )
            if entry is None:
                return False
            text = (entry or {}).get("text") or ""
            if not text:
                return False
            metadata = _normalize_metadata((entry or {}).get("metadata"))
            result = self._add_text_with_metadata(
                text,
                _provenance(
                    metadata["source"],
                    metadata["confidence"]
                    if isinstance(metadata["confidence"], (int, float))
                    else CONFIDENCE_INFERRED,
                    bool(metadata["user_confirmed"]),
                    created_at=metadata["created_at"],
                    created_at_source=metadata["created_at_source"] or "written",
                    slot=metadata["slot"],
                    is_rule=bool(metadata["is_rule"]),
                ),
            )
            if not result:
                return False
            self._write_retracted_trash(
                [e for e in trash if (e or {}).get("id") != memory_id]
            )
            # 账本记一笔唤醒：恢复是重新存一条（新 id），旧 id 在账本里
            # 还是 deleted——旧 id 不在索引里了，不影响筛子；新 id 是
            # 当前的，记 revive 让账本知道这条记忆回来了。
            new_id = ((result or {}).get("results") or [{}])[0].get("id")
            if new_id:
                # key 带上：恢复的这条原话标回 current——软删除时记的
                # delete 把它的 key 标成 deleted 了，恢复要把它标回来，
                # 不然兜底、对账还以为这条是删了的。
                self._ledger_append(
                    _ledger.LEDGER_EVENT_REVIVE,
                    memory_id=new_id,
                    key=_ledger.ledger_event_key(
                        text, metadata.get("created_at")
                    ),
                    note=f"从回收站恢复，旧 id {memory_id}。",
                )
                # 内存里刚删的记录也清掉——这条回来了，以后重新说起
                # 不再拦（2026-11-16 拉面）。
                self._just_deleted_texts.pop(
                    _normalize_memory_text(text), None
                )
            return True

    def list_retracted(self):
        """列出回收站里的软删除记忆（34 号）：每项 {"id", "text", "retracted_at"}。"""
        return [
            {
                "id": (e or {}).get("id"),
                "text": (e or {}).get("text") or "",
                "retracted_at": (e or {}).get("retracted_at"),
            }
            for e in self._read_retracted_trash()
        ]

    # ---------- 恢复（2026-11-16 真机测试：恢复工具缺失） ----------

    def _deleted_entries(self):
        """能恢复的已删记忆（内部用，带恢复要用的全部字段）。

        两处来源并在一起，同一条原文只留一条（回收站的优先，
        它带旧 id 和完整元数据）：
        - 回收站（软删除的）：_read_retracted_trash()；
        - 账本（深度删除的）：ledger_state() 里 key_states 标了
          deleted 的 add 档案（原话 + 形成时间 + 元数据都在），
          旧 id 按 id_to_key 反查，查不到就是 None（账本的 add
          事件本来就没有 mem0 id，靠原话认）。
        每项 {"id", "text", "metadata", "key", "deleted_at",
        "source"}，source 是 "trash" 或 "ledger"。只读，不改库。
        """
        entries = []
        seen_texts = set()
        for item in reversed(self._read_retracted_trash()):
            text = (item or {}).get("text") or ""
            if not text:
                continue
            norm = _normalize_memory_text(text)
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
            entries.append(
                {
                    "id": (item or {}).get("id"),
                    "text": text,
                    "metadata": dict((item or {}).get("metadata") or {}),
                    "key": None,
                    "deleted_at": (item or {}).get("deleted_at")
                    or (item or {}).get("retracted_at"),
                    "source": "trash",
                }
            )
        try:
            state = self.ledger_state()
        except Exception:
            state = {}
        key_states = (state or {}).get("key_states") or {}
        id_to_key = (state or {}).get("id_to_key") or {}
        key_to_id = {}
        for mem_id, key in id_to_key.items():
            key_to_id.setdefault(key, mem_id)
        # 旧 id 反查 key 不能只靠 id_to_key：add 事件本来就没有 mem0 id，
        # 纯 add→delete 的那条 id 只在 delete 事件里（memory_id + key
        # 都带），折叠时 delete 分支又不写 id_to_key——所以这里直接读
        # 一遍原始事件，把 delete 事件里的 id→key 也补进来。
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            events = []
        for event in events or []:
            if (event or {}).get("type") != _ledger.LEDGER_EVENT_DELETE:
                continue
            mem_id = (event or {}).get("memory_id")
            key = (event or {}).get("key")
            if mem_id and key:
                key_to_id.setdefault(key, mem_id)
        for record in reversed((state or {}).get("adds") or []):
            key = (record or {}).get("key")
            if not key or key_states.get(key) != _ledger.LEDGER_STATE_DELETED:
                continue
            text = (record or {}).get("text") or ""
            if not text:
                continue
            norm = _normalize_memory_text(text)
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
            meta = dict((record or {}).get("meta") or {})
            entries.append(
                {
                    "id": key_to_id.get(key),
                    "text": text,
                    "metadata": meta,
                    "key": key,
                    "deleted_at": None,
                    "source": "ledger",
                }
            )
        return entries

    def list_deleted_memories(self):
        """列出能恢复的已删记忆：每项 {"id", "text", "deleted_at",
        "source"}，回收站的在前（新的在前），账本的在后（新的在前）。
        只读，不改库——给 revive_memory 工具列清单用，模型按原文挑。"""
        return [
            {
                "id": entry.get("id"),
                "text": entry.get("text") or "",
                "deleted_at": entry.get("deleted_at"),
                "source": entry.get("source"),
            }
            for entry in self._deleted_entries()
        ]

    def revive_scope(self, memory_id=None, text=None):
        """恢复的确认前核对（只读）：要恢复的那条在不在已删里。

        memory_id（旧 id）和 text（原文）至少给一个；两个都给时
        必须对得上同一条（id 对上的那条原文要和 text 一致）。
        返回 {"found": bool, "entry": {...} | None, "reason": str}，
        entry 是 _deleted_entries() 里那一项（带 metadata、key）。
        """
        if not memory_id and not (text or "").strip():
            return {
                "found": False,
                "entry": None,
                "reason": "要恢复哪条没说：传 memory_id（旧 id）或 text（原文）。",
            }
        entries = self._deleted_entries()
        picked = None
        if memory_id:
            picked = next(
                (e for e in entries if (e or {}).get("id") == memory_id), None
            )
            if picked is None:
                return {
                    "found": False,
                    "entry": None,
                    "reason": f"已删的记忆里没有这条（id 对不上）：{memory_id}。",
                }
            if (text or "").strip() and _normalize_memory_text(
                picked.get("text") or ""
            ) != _normalize_memory_text(text):
                return {
                    "found": False,
                    "entry": None,
                    "reason": "id 对上的那条原文和传来的 text 不一致，挑错了。",
                }
        else:
            norm = _normalize_memory_text(text)
            picked = next(
                (
                    e
                    for e in entries
                    if _normalize_memory_text(e.get("text") or "") == norm
                ),
                None,
            )
            if picked is None:
                return {
                    "found": False,
                    "entry": None,
                    "reason": f"已删的记忆里没有这条原文：{text}。",
                }
        return {"found": True, "entry": picked, "reason": ""}

    def revive_memory(self, memory_id=None, text=None):
        """恢复一条已删的记忆（2026-11-16 真机测试：恢复工具缺失）。

        账本是恢复的凭据：删掉的原话在账本档案里留着（add 事件），
        回收站里留着软删除的原文 + 元数据。恢复就是按原话 + 原元数据
        重新存一条（新 id），再记一笔 revive 事件——旧 id 在账本里
        还是 deleted，不影响筛子；新 id 是当前的。
        来源/确认状态/形成时间都按原样恢复（时间戳用原来的，别写成
        恢复时间），和 restore_memory（34 号）同一条规矩。
        原话已经在库里时不重复存，返回 False（调用方按“已经在库里了”
        回 BOOS，别谎称恢复了）。
        返回新 id（恢复了）/ False（已删里没有这条 / 存失败）。
        """
        if not memory_id and not (text or "").strip():
            return False
        with _WRITE_LOCK:
            scope = self.revive_scope(memory_id, text)
            if not scope.get("found"):
                return False
            entry = scope.get("entry") or {}
            entry_text = (entry.get("text") or "").strip()
            if not entry_text:
                return False
            # 已经在库里就别重复存。
            try:
                current = {
                    _normalize_memory_text((item or {}).get("text") or "")
                    for item in self.get_all(include_sensitive=True)
                }
            except Exception:
                return False
            if _normalize_memory_text(entry_text) in current:
                return False
            if (entry.get("source") or "") == "trash":
                old_id = entry.get("id")
                if not old_id:
                    return False
                if not self.restore_memory(old_id):
                    return False
                # restore_memory 只回 True/False：按原文找回新 id。
                try:
                    for item in self.get_all(include_sensitive=True):
                        if _normalize_memory_text(
                            (item or {}).get("text") or ""
                        ) == _normalize_memory_text(entry_text):
                            return (item or {}).get("id") or True
                except Exception:
                    pass
                return True
            meta = dict(entry.get("metadata") or {})
            result = self._add_text_with_metadata(
                entry_text,
                _provenance(
                    meta.get("source") or SOURCE_INFERRED,
                    meta.get("confidence")
                    if isinstance(meta.get("confidence"), (int, float))
                    else CONFIDENCE_INFERRED,
                    bool(meta.get("user_confirmed")),
                    created_at=meta.get("created_at"),
                    created_at_source="written",
                    slot=meta.get("slot"),
                    is_rule=bool(meta.get("is_rule")),
                ),
            )
            results = (result or {}).get("results") or []
            if not results:
                return False
            new_id = ((results[0] or {}).get("id")) or ""
            if not new_id:
                return False
            # 账本记一笔恢复：恢复的这条原话标回 current——删掉时记的
            # delete 把它的 key 标成 deleted 了，恢复要把它标回来，
            # 不然兜底、对账还以为这条是删了的。
            self._ledger_append(
                _ledger.LEDGER_EVENT_REVIVE,
                memory_id=new_id,
                key=entry.get("key")
                or _ledger.ledger_event_key(
                    entry_text, meta.get("created_at")
                ),
                note=f"从账本恢复，旧 id {entry.get('id')}。"
                if entry.get("id")
                else "从账本恢复。",
            )
            # 内存里刚删的记录也清掉——这条回来了，以后重新说起
            # 不再拦（2026-11-16 拉面）。
            self._just_deleted_texts.pop(
                _normalize_memory_text(entry_text), None
            )
            return new_id

    # ---------- 揣摩人心（2026-11-16，设计方案见 memory/inference.py） ----------

    def record_inference(self, guess, evidence, confidence, layer,
                         mode=_inference.MODE_SPOT, confidence_text=None):
        """记一条揣摩进账本（待定区），返回推断字典；缺字段就不记。

        推断不是记忆——它只进账本（inference 事件，status pending），
        不进 mem0。BOOS 当场说“对”才记下（status confirmed，仍只在
        账本里，不写进 mem0——见 confirm_inference 的说明）；说“不对”
        就删除（status deleted，不记作废）。
        同一句已经待定了就不重复记——问一次就够了，问两遍是打扰。
        猜的内容里有敏感信息时不记。账本写失败时返回 None。
        """
        if not isinstance(guess, str) or not guess.strip():
            return None
        if not isinstance(evidence, str) or not evidence.strip():
            return None
        if confidence not in ("low", "medium", "high"):
            return None
        if layer not in _inference.INFERENCE_LAYERS:
            return None
        if mode not in (_inference.MODE_SPOT, _inference.MODE_LONGTERM):
            return None
        guess = guess.strip()
        evidence = evidence.strip()
        if _contains_sensitive(guess) or _contains_sensitive(evidence):
            logger.info("揣摩：猜的内容里有敏感信息，本次不记。")
            return None
        for pending in self.list_pending_inferences():
            if _normalize_memory_text(
                pending.get("guess") or ""
            ) == _normalize_memory_text(guess):
                return dict(pending)
        created_at = _now_iso()
        key = _ledger.ledger_event_key(guess, created_at)
        conf_text = confidence_text or _inference._CONFIDENCE_TEXT[confidence]
        event = self._ledger_append(
            _ledger.LEDGER_EVENT_INFERENCE,
            key=key,
            text=guess,
            evidence=evidence,
            confidence=confidence,
            confidence_text=conf_text,
            layer=layer,
            mode=mode,
            status=_inference.STATUS_PENDING,
            created_at=created_at,
        )
        if event is None:
            return None
        return {
            "key": key,
            "guess": guess,
            "evidence": evidence,
            "confidence": confidence,
            "confidence_text": conf_text,
            "layer": layer,
            "mode": mode,
            "status": _inference.STATUS_PENDING,
            "seq": event.get("seq"),
            "created_at": created_at,
        }

    @staticmethod
    def _inference_record_to_dict(record):
        """账本里折叠出的推断档案 → 对外用的推断字典（guess 是猜的内容）。
        partial：BOOS 只认了一半（“差不多”“对了一半”），确认时标 True。"""
        record = record or {}
        return {
            "key": record.get("key") or "",
            "guess": record.get("text") or "",
            "evidence": record.get("evidence") or "",
            "confidence": record.get("confidence") or "low",
            "confidence_text": record.get("confidence_text") or "我只是猜的",
            "layer": record.get("layer") or _inference.LAYER_INDUCTION,
            "mode": record.get("mode") or _inference.MODE_SPOT,
            "status": record.get("status") or _inference.STATUS_PENDING,
            "partial": bool(record.get("partial")),
            "seq": record.get("seq"),
            "created_at": record.get("created_at") or "",
        }

    def list_inferences(self, status=None):
        """列出账本里的推断（按 seq 排）；status 给了就只列那个状态的。
        账本读不出来时返回 []，不抛异常。"""
        try:
            state = self.ledger_state()
        except Exception:
            return []
        records = (state or {}).get("inferences") or {}
        out = [self._inference_record_to_dict(r) for r in records.values()]
        if status is not None:
            out = [r for r in out if r.get("status") == status]
        out.sort(
            key=lambda r: r.get("seq") if isinstance(r.get("seq"), int) else 0
        )
        return out

    def list_pending_inferences(self):
        """列出待定的推断（问了、还没回的那几条）。"""
        return self.list_inferences(status=_inference.STATUS_PENDING)

    def list_pending_decisions(self):
        """列出“挂起来等 BOOS 定”的记忆（补丁 2 两道闸的待定清单）。

        两类都算：pending_confirmation（新说法可信度不够、盖不掉已
        确认的旧条，挂着等确认）和 conflict_for_review（两条说法
        对着干，等 BOOS 定哪条是对的）。返回列表，每项 {"id", "text",
        "reason", "metadata"}，按写入序号排（先挂的先问）。
        面板/聊天层拿它问 BOOS；问完由调用方调 correct_memory /
        delete_memory / 确认接口处理，这里只列、不改库。
        读不到库时返回 []，不抛异常。"""
        try:
            items = self.get_all()
        except Exception:
            logger.warning("待定清单读库失败。", exc_info=True)
            return []
        out = []
        for item in items or []:
            meta = (item or {}).get("metadata") or {}
            if meta.get("pending_confirmation"):
                reason = "新说法可信度不够，盖不掉已确认的旧条，等 BOOS 确认。"
            elif meta.get("conflict_for_review"):
                reason = "跟库里另一条说法对着干，等 BOOS 定哪条是对的。"
            else:
                continue
            out.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text") or "",
                    "reason": reason,
                    "metadata": meta,
                }
            )
        out.sort(key=lambda entry: (
            (entry.get("metadata") or {}).get("write_seq")
            if isinstance((entry.get("metadata") or {}).get("write_seq"), int)
            else -1
        ))
        return out

    def _set_inference_status(self, key, status, partial=False):
        """把一条待定推断的状态改掉（确认/否认共用）：账本里追加一笔
        同 key 的 inference 事件，只带状态，折叠时最新状态赢。
        partial：BOOS 只认了一半（“差不多”“对了一半”）时 True，账本
        事件里带上，折叠后档案里留着这个标记。
        找不到这条待定推断时返回 None，什么都不写。"""
        if not key:
            return None
        pending = next(
            (r for r in self.list_pending_inferences() if r.get("key") == key),
            None,
        )
        if pending is None:
            return None
        event = self._ledger_append(
            _ledger.LEDGER_EVENT_INFERENCE,
            key=key,
            status=status,
            **({"partial": True} if partial else {}),
        )
        if event is None:
            return None
        # 新反馈落地了（确认/否认都算）：顺手跑一次校准，看门槛要不要动。
        # 校准坏了不拦路——确认/否认本身已经记好了。
        try:
            self.auto_adjust_inference_threshold()
        except Exception:
            logger.warning("揣摩校准：确认/否认后调门槛失败，门槛不动。", exc_info=True)
        updated = dict(pending)
        updated["status"] = status
        if partial:
            updated["partial"] = True
        updated["seq"] = event.get("seq")
        return updated

    def confirm_inference(self, key, partial=False):
        """BOOS 说“对”：这条推断记下（status confirmed）。
        partial=True：BOOS 只认了一半（“差不多”“对了一半”）——算认了，
        状态还是 confirmed，档案里多标一个 partial=True，以后校准和
        复盘能分得清“全认了”和“只认了一半”。
        只记账本，不写进 mem0——_provenance 的铁规矩是 user_confirmed
        只在 source="user" 时才可能是 True，推断来的记忆在 mem0 里
        永远是未确认的；把确认过的推断写进 mem0 会抹掉“模型猜的”和
        “BOOS 亲口说的”这条线（诚实底线）。确认过的推断以后揣摩时
        当“相关记忆”用（从账本里翻），不进检索。内存里那份待定也清掉。
        找不到这条待定推断时返回 None。"""
        updated = self._set_inference_status(
            key, _inference.STATUS_CONFIRMED, partial=partial
        )
        if updated is not None and (self._pending_inference or {}).get("key") == key:
            self._pending_inference = None
        return updated

    def deny_inference(self, key):
        """BOOS 说“不对”：这条推断删除（status deleted，不记作废）。
        账本是 append-only 的，原话留着当审计（只改状态，不抹行）；
        内存里那份待定也清掉。找不到这条待定推断时返回 None。"""
        updated = self._set_inference_status(key, _inference.STATUS_DELETED)
        if updated is not None and (self._pending_inference or {}).get("key") == key:
            self._pending_inference = None
        return updated

    def inference_feedback_summary(self):
        """校准的底子：每次确认/否认都记在账本里，这里按层数一数
        （设计方案第五节第 7 条只定了“记下来”，校准结果存在哪、
        怎么调门槛，待细化时定——这里只数数，不调门槛）。
        返回 {层: {"confirmed": n, "denied": m}}，没有的层不出现。"""
        summary = {}
        for record in self.list_inferences():
            layer = record.get("layer") or _inference.LAYER_INDUCTION
            entry = summary.setdefault(layer, {"confirmed": 0, "denied": 0})
            if record.get("status") == _inference.STATUS_CONFIRMED:
                entry["confirmed"] += 1
            elif record.get("status") == _inference.STATUS_DELETED:
                entry["denied"] += 1
        return summary

    def inference_calibration_summary(self):
        """校准记账：按模型自己说的把握（low/medium/high）数确认/否认/待定。
        猜的时候模型说的把握记在 inference 事件的 confidence 里，BOOS
        最后认不认记在 status 里——账本里本来就有，这里只是按把握
        聚合一下，不改账本结构。
        读法：high 那档否认比确认多，就是过分自信（把握说得很大、猜得
        却老错）；low 那档确认多，就是太保守（没把握的其实猜得挺准）。
        返回 {把握: {"confirmed": n, "denied": m, "pending": p}}，
        没有的把握不出现。"""
        summary = {}
        for record in self.list_inferences():
            confidence = record.get("confidence") or "low"
            entry = summary.setdefault(
                confidence, {"confirmed": 0, "denied": 0, "pending": 0}
            )
            status = record.get("status")
            if status == _inference.STATUS_CONFIRMED:
                entry["confirmed"] += 1
            elif status == _inference.STATUS_DELETED:
                entry["denied"] += 1
            elif status == _inference.STATUS_PENDING:
                entry["pending"] += 1
        return summary

    def _load_inference_calibration_state(self):
        """读校准状态文件（门槛 + 上次调门槛时各档的裁决数）。

        文件没了/读坏了/形状不对时回默认（门槛 low、各档 0），不抛异常——
        校准是辅助，坏了就从默认重来，别把揣摩搞崩。
        """
        default = (_inference.INFERENCE_MIN_CONFIDENCE_DEFAULT, {})
        try:
            raw = json.loads(
                Path(self.inference_calibration_path).read_text(encoding="utf-8")
            )
        except Exception:
            return default
        if not isinstance(raw, dict):
            return default
        threshold = raw.get("threshold")
        if threshold not in _inference.CONFIDENCE_ORDER:
            threshold = _inference.INFERENCE_MIN_CONFIDENCE_DEFAULT
        baseline = raw.get("baseline")
        if not isinstance(baseline, dict):
            return threshold, {}
        clean = {}
        for conf, entry in baseline.items():
            if conf not in _inference.CONFIDENCE_ORDER:
                continue
            if not isinstance(entry, dict):
                continue
            confirmed = entry.get("confirmed")
            denied = entry.get("denied")
            if (
                isinstance(confirmed, int)
                and not isinstance(confirmed, bool)
                and confirmed >= 0
                and isinstance(denied, int)
                and not isinstance(denied, bool)
                and denied >= 0
            ):
                clean[conf] = {"confirmed": confirmed, "denied": denied}
        return threshold, clean

    def _save_inference_calibration_state(self, threshold, baseline):
        """写校准状态文件。写失败只记日志、不抛异常——门槛丢了下次按
        账本重算，揣摩本身不受影响。"""
        try:
            path = Path(self.inference_calibration_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"threshold": threshold, "baseline": baseline},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return True
        except Exception:
            logger.warning("揣摩校准：门槛状态写不进去，下次重算。", exc_info=True)
            return False

    def _meets_inference_threshold(self, confidence):
        """这条推断的把握够不够开口：把握档位 >= 当前门槛才说。
        把握不在 low/medium/high 里时按 low 算（往保守里算）。"""
        order = _inference.CONFIDENCE_ORDER
        conf = confidence if confidence in order else "low"
        threshold = self._inference_threshold
        if threshold not in order:
            threshold = _inference.INFERENCE_MIN_CONFIDENCE_DEFAULT
        return order.index(conf) >= order.index(threshold)

    def auto_adjust_inference_threshold(self):
        """校准：按已裁决的反馈自动调揣摩的开口门槛（2026-11-17）。

        门槛是“开口的最低把握”（low/medium/high，默认 low——低把握也敢说，
        只说“我只是猜的”）。调门槛的规矩（数字在 memory/inference.py 里，
        BOOS 改那里就行）：
          - 只看上次调完之后新来的反馈（增量）：每档新裁决（确认+否认）
            攒够 5 条，那档才有资格调门槛；旧账不重复算。
          - 基线（上次看完的那批）什么时候更新：这批有了明确结论才更新——
            动了、到顶/到底了、两边打架了；证据不够、胜负不明时不更新，
            继续攒着（到顶了不更新，旧信号会一直留着、以后反方向的证据
            来了还堵在那儿）。
          - 胜出那边要领先至少 2 条才动（3 比 2 这种胶着的不动）。
          - 一次最多动一档，上限 high、下限 low。
          - high 档否认多 = 过分自信 → 门槛往上提；low 档确认多 = 太保守
            → 门槛往下放；两边同时喊反方向时不动（证据打架，等更多反馈）。
        注意：数的是“上次调完之后新来的”那一批，不是累计总数——累计数
        会让旧信号一直推着门槛走（比如 high 档以前否认多，即使后来猜准
        了也还在往上推）。增量只看新反馈，信号过时了就不再推。
        触发点：确认/否认落账本时（_set_inference_status）顺手跑一次。
        返回 {"threshold": 当前门槛, "changed": 是否动了, "reason": 原因}。
        绝不抛异常——校准坏了不能把揣摩和聊天搞崩。
        """
        try:
            summary = self.inference_calibration_summary()
            baseline = self._inference_calibration_baseline or {}
            order = _inference.CONFIDENCE_ORDER
            min_decide = _inference.CALIBRATION_MIN_DECIDE
            min_margin = _inference.CALIBRATION_MIN_MARGIN
            step = _inference.CALIBRATION_STEP
            # 新来的反馈（增量）：每档现在数减掉上次调完时的数；账本被压
            # 缩/重置导致数倒退时按 0 算（不拿负数当信号）。
            delta = {}
            for conf in order:
                entry = summary.get(conf) or {}
                base = baseline.get(conf) or {}
                delta[conf] = {
                    "confirmed": max(
                        0, entry.get("confirmed", 0) - base.get("confirmed", 0)
                    ),
                    "denied": max(
                        0, entry.get("denied", 0) - base.get("denied", 0)
                    ),
                }
            new_decided = sum(
                entry["confirmed"] + entry["denied"] for entry in delta.values()
            )
            threshold = self._inference_threshold
            if threshold not in order:
                threshold = _inference.INFERENCE_MIN_CONFIDENCE_DEFAULT
            if new_decided == 0:
                return {
                    "threshold": threshold,
                    "changed": False,
                    "reason": "no_new_feedback",
                }
            high = delta.get("high") or {}
            low = delta.get("low") or {}
            high_decided = high.get("confirmed", 0) + high.get("denied", 0)
            low_decided = low.get("confirmed", 0) + low.get("denied", 0)
            raise_it = (
                high_decided >= min_decide
                and high.get("denied", 0) - high.get("confirmed", 0) >= min_margin
            )
            lower_it = (
                low_decided >= min_decide
                and low.get("confirmed", 0) - low.get("denied", 0) >= min_margin
            )
            new_threshold = threshold
            reason = "no_clear_signal"
            if raise_it and lower_it:
                reason = "conflicting_signals"
            elif raise_it:
                idx = min(order.index(threshold) + step, len(order) - 1)
                new_threshold = order[idx]
                reason = (
                    "overconfident_high"
                    if new_threshold != threshold
                    else "already_at_ceiling"
                )
            elif lower_it:
                idx = max(order.index(threshold) - step, 0)
                new_threshold = order[idx]
                reason = (
                    "too_conservative_low"
                    if new_threshold != threshold
                    else "already_at_floor"
                )
            changed = new_threshold != threshold
            # 基线什么时候记到现在这批：只有这批反馈已经有了明确结论才记——
            # 动了、到顶/到底了、两边打架了。证据不够、胜负不明时不记，继续
            # 攒着（比如 high 档 3 否认 2 确认，再来几条否认攒够了就动）。
            # 到顶了不记的话，旧信号会一直留着、以后反方向的证据来了还堵
            # 在那儿；打架了不记的话，同一批打架的证据会一直堵着。
            consume = changed or reason in (
                "already_at_ceiling",
                "already_at_floor",
                "conflicting_signals",
            )
            if consume:
                new_baseline = {
                    conf: {
                        "confirmed": (summary.get(conf) or {}).get("confirmed", 0),
                        "denied": (summary.get(conf) or {}).get("denied", 0),
                    }
                    for conf in order
                }
                self._inference_threshold = new_threshold
                self._inference_calibration_baseline = new_baseline
                self._save_inference_calibration_state(new_threshold, new_baseline)
            return {
                "threshold": new_threshold,
                "changed": changed,
                "reason": reason,
            }
        except Exception:
            logger.warning("揣摩校准：自动调门槛失败，门槛不动。", exc_info=True)
            return {
                "threshold": self._inference_threshold,
                "changed": False,
                "reason": "error",
            }

    def count_new_memories_since_longterm(self):
        """自上次长期揣摩以来新记了几条记忆（账本 add 事件按 seq 数）。
        长期揣摩本身就是标记（它的 seq）；上次长期揣摩说“没什么可猜的”
        时记下的跳过线（_longterm_skip_until_seq）也算数——不然每轮都
        多调一次模型。账本读不出来时返回 0（不揣，别把聊天搞崩）。"""
        try:
            events, _info = _ledger.read_events(self.ledger_path)
        except Exception:
            return 0
        baseline = self._longterm_skip_until_seq or 0
        for event in events or []:
            if (event or {}).get("type") != _ledger.LEDGER_EVENT_INFERENCE:
                continue
            if (event or {}).get("mode") != _inference.MODE_LONGTERM:
                continue
            seq = (event or {}).get("seq")
            if isinstance(seq, int):
                baseline = max(baseline, seq)
        count = 0
        for event in events or []:
            if (event or {}).get("type") != _ledger.LEDGER_EVENT_ADD:
                continue
            seq = (event or {}).get("seq")
            if isinstance(seq, int) and seq > baseline:
                count += 1
        return count

    def infer_from_turn(self, messages):
        """一轮答完后的揣摩（当场揣摩的主力入口，run_question 调它）。
        顺序：先处理上一轮的待定推断（本轮光秃秃地回“对/不对”就是在
        回上一句的问话）→ 攒够 20 条新记忆就做长期、不做当场 →
        否则当场揣摩本轮对话。一轮最多问一句。
        返回 {"resolved": {"result": "confirmed"/"denied", ...推断字典},
        "inference": 推断字典}；没揣出来、没模型、出错时返回 {}，
        绝不抛异常——揣摩失败不能把已经拿到的回答搞崩。
        """
        try:
            return self._infer_from_turn_inner(messages)
        except Exception:
            logger.warning("揣摩失败，本轮不问。", exc_info=True)
            return {}

    def _infer_from_turn_inner(self, messages):
        result = {}
        latest_user = _latest_user_content(messages or [])
        # 1. 先处理上一轮的待定推断。只认紧接着问话的下一轮
        # （_pending_inference 是上一轮刚问的那条）；本轮没回，
        # 它就过期了（账本里还是 pending，以后对账/自检能翻出来）。
        # 回话先按词表一眼看清（对/差不多/不对，带后半句的也认，后半句
        # 是他补充的新信息，留在 resolved["extra"] 里）；一眼看不清的
        # 再用模型判一轮；模型也判不出来就留待定，宁可不认，不可认错。
        pending = self._pending_inference
        self._pending_inference = None
        if not latest_user:
            return result
        classified = _inference.classify_inference_reply(latest_user)
        if pending is not None and classified["verdict"] == "unclear":
            classified = self._classify_reply_with_llm(pending, latest_user)
        verdict = classified["verdict"]
        extra = classified.get("extra") or ""
        if pending is not None and verdict in ("confirmed", "partial", "denied"):
            if verdict == "confirmed":
                resolved = self.confirm_inference(pending.get("key"))
                resolved_result = "confirmed"
            elif verdict == "partial":
                # “差不多”“对了一半”：算认了，状态还是 confirmed，
                # 档案里多标 partial=True（以后校准分得清全认和半认）。
                resolved = self.confirm_inference(pending.get("key"), partial=True)
                resolved_result = "confirmed"
            else:
                resolved = self.deny_inference(pending.get("key"))
                resolved_result = "denied"
            if resolved is not None:
                entry = {"result": resolved_result, "extra": extra, **resolved}
                if verdict == "partial":
                    entry["partial"] = True
                result["resolved"] = entry
            # 这一轮就是在回上一句揣摩，别再当场揣——拿一句回答揣只会
            # 揣出“你确认了刚才的推断”这种循环话。
            return result
        if verdict in ("confirmed", "partial", "denied"):
            # 没待定推断时，“对/不对”这种回答也别拿来揣（只有一句回答，
            # 没东西可揣）。
            return result
        # 2. 长期模式攒够了就做长期，不做当场（一轮只问一句）。
        # 长期那次模型说“没什么可猜的”，就把跳过线记到当前序号，
        # 下次再攒够 20 条新的才重跑。
        if self.count_new_memories_since_longterm() >= _inference.LONGTERM_THRESHOLD:
            longterm = self._run_longterm_inference()
            if longterm is not None:
                self._pending_inference = longterm
                result["inference"] = longterm
                return result
            try:
                state = self.ledger_state()
                self._longterm_skip_until_seq = (state or {}).get("seq_max") or 0
            except Exception:
                pass
        # 3. 当场揣摩：本轮对话 + 相关记忆喂给模型，有把握的才说。
        spot = self._run_spot_inference(messages)
        if spot is not None:
            self._pending_inference = spot
            result["inference"] = spot
        return result

    def _classify_reply_with_llm(self, pending, latest_user):
        """回话不是一眼能看清时，用模型判一判：确认/部分/否认/不清楚。
        模型只回两行（第一行四词里的一个，第二行是新补充的信息，没有
        就写“无”）。没模型、模型回的认不出来时返回 unclear——不认、
        不删，推断留待定。绝不抛异常。"""
        try:
            question = _inference.render_inference_question(pending or {})
            prompt = (
                "BOOS 刚被问了一句揣摩，他回了一句。请判断他这句是在回那个揣摩吗，"
                "是确认、部分确认、否认，还是看不出来？\n"
                f"问的是：{question}\n"
                f"他回的是：{latest_user}\n"
                "只输出两行：第一行只写“确认”“部分”“否认”“不清楚”四个词里的一个；"
                "第二行写他这句里除了确认/否认之外新补充的信息，没有就写“无”。"
            )
            raw = self._call_llm_text("你是十二的分类助手，只输出两行。", prompt)
            if not raw:
                return {"verdict": "unclear", "extra": ""}
            return _inference.parse_reply_verdict(raw)
        except Exception:
            logger.warning("揣摩：模型判回话失败，本次不认。", exc_info=True)
            return {"verdict": "unclear", "extra": ""}

    def _call_inference(self, prompt):
        """调一次模型、解析一次揣摩（一句话）。解析失败时用更严的
        要求再试一次（只输出那一句话，不要加解释），两次都解析不出来
        返回 None——不猜、不拦路。"""
        raw = self._call_llm_text(
            "你是十二的揣摩助手，只输出一句话。", prompt
        )
        if raw:
            parsed = _inference.parse_inference_output(raw)
            if parsed is not None:
                return parsed
        raw = self._call_llm_text(
            "你是十二的揣摩助手，只输出一句话。",
            prompt + "\n只输出那一句话，不要加任何解释。",
        )
        if not raw:
            return None
        return _inference.parse_inference_output(raw)

    def _run_spot_inference(self, messages):
        """当场揣摩：本轮对话 + 相关记忆喂给模型，有把握的才说。
        返回记好的推断字典，没揣出来时返回 None。
        对话是脱敏后发给模型的（和 add_turn 一路）；相关记忆按最新
        一句搜，不强化（揣摩只是“用上”看看，不是真用上，别因为揣摩
        一次就把一条记忆的衰减计数刷上去）。"""
        latest_user = _latest_user_content(messages or [])
        if not latest_user:
            return None
        lines = []
        for msg in messages or []:
            role = (msg or {}).get("role")
            content = (msg or {}).get("content") or ""
            if role not in ("user", "assistant") or not content.strip():
                continue
            who = "BOOS" if role == "user" else "十二"
            lines.append(f"{who}：{_redact_sensitive(content)}")
        conversation = "\n".join(lines[-6:])
        if not conversation.strip():
            return None
        related_text = ""
        try:
            related = self.search(latest_user, top_k=5, reinforce=False) or []
            related_text = _inference.format_related_memories(related)
        except Exception:
            logger.warning(
                "揣摩：相关记忆没搜到，本次只按对话揣。", exc_info=True
            )
        # 以前确认过的推断也喂进去（从账本里翻，不进检索）：他以前就是
        # 这么想的、一贯的习惯是什么——揣的时候从对他的了解出发，别凭空猜。
        try:
            confirmed = self.list_inferences(
                status=_inference.STATUS_CONFIRMED
            )[-5:]
            confirmed_text = _inference.format_confirmed_inferences(confirmed)
            if confirmed_text:
                related_text = (
                    f"{related_text}\n{confirmed_text}".strip()
                    if related_text
                    else confirmed_text
                )
        except Exception:
            logger.warning(
                "揣摩：已确认的推断没翻到，本次只按对话和相关记忆揣。",
                exc_info=True,
            )
        prompt = _inference.build_inference_prompt(conversation, related_text)
        parsed = self._call_inference(prompt)
        if parsed is None:
            return None
        if not self._meets_inference_threshold(parsed["confidence"]):
            # 门槛拦住了：把握不够，不问、不记——没说出口的猜不用进待定区。
            logger.info(
                "揣摩：把握 %s 没到门槛 %s，本次不问。",
                parsed["confidence"],
                self._inference_threshold,
            )
            return None
        return self.record_inference(
            parsed["guess"],
            parsed["evidence"],
            parsed["confidence"],
            parsed["layer"],
            mode=_inference.MODE_SPOT,
            confidence_text=parsed["confidence_text"],
        )

    def _run_longterm_inference(self):
        """长期模式：最近 20 条新记的记忆里找最反常的，猜一个模式。
        返回记好的推断字典，没揣出来时返回 None。"""
        try:
            state = self.ledger_state()
            adds = (state or {}).get("adds") or []
        except Exception:
            return None
        recent = adds[-_inference.LONGTERM_THRESHOLD:]
        lines = []
        for entry in recent:
            text = (entry or {}).get("text") or ""
            if text.strip():
                lines.append(text)
        if not lines:
            return None
        prompt = _inference.build_longterm_inference_prompt("\n".join(lines))
        parsed = self._call_inference(prompt)
        if parsed is None:
            return None
        if not self._meets_inference_threshold(parsed["confidence"]):
            # 门槛拦住了：把握不够，不问、不记——没说出口的猜不用进待定区。
            logger.info(
                "揣摩：把握 %s 没到门槛 %s，本次不问。",
                parsed["confidence"],
                self._inference_threshold,
            )
            return None
        return self.record_inference(
            parsed["guess"],
            parsed["evidence"],
            parsed["confidence"],
            parsed["layer"],
            mode=_inference.MODE_LONGTERM,
            confidence_text=parsed["confidence_text"],
        )

    def _asked_inference_guesses(self, messages):
        """本轮对话里问过（assistant 消息里出现过）的推断原文（归一化）。
        只拦这一轮问过的——BOOS 以后自己重新说起同一句时，提取照常存。
        任何状态的推断都算（待定/已确认/已删除都拦）：问句里的猜要是被
        提取存进 mem0，待定的会变成没确认的事实，已确认的会丢了“模型
        猜的”这条线，已删除的会死灰复燃。"""
        asked = set()
        assistant_texts = []
        for msg in messages or []:
            if (msg or {}).get("role") == "assistant":
                content = (msg or {}).get("content") or ""
                if content:
                    assistant_texts.append(_normalize_memory_text(content))
        if not assistant_texts:
            return asked
        joined = "\n".join(assistant_texts)
        for record in self.list_inferences():
            norm = _normalize_memory_text((record or {}).get("guess") or "")
            if norm and norm in joined:
                asked.add(norm)
        return asked

    def _drop_pending_inference_results(self, result, safe_messages):
        """提取后过滤：刚问出口的揣摩，别让提取把它当事实存进库里。

        问句是十二自己说的（assistant 消息），但提取时 assistant 的话
        原样透传当上下文（见 _filter_extraction_messages），模型会把
        问句里的猜当成 BOOS 的事实提出来——2026-11-16 删记忆那次，
        助手复述的“不再爱吃冒菜”就是这么被存成一条记忆的。
        只拦本轮问过的那句（问句在 safe_messages 里才拦）：BOOS 以后
        自己重新说起同一句时，提取照常存。
        条目已经在库里了，和 _drop_deleted_text_results 一路回滚；
        回滚不掉的照实记日志、不藏着。返回过滤后的结果（拦掉的条目
        不在 results 里），拦了几条记在 inference_dropped 里。
        """
        items = (result or {}).get("results") or []
        if not items:
            return result
        asked = self._asked_inference_guesses(safe_messages)
        if not asked:
            return result
        dropped = []
        kept = []
        for item in items:
            mem_text = (item or {}).get("memory") or ""
            if mem_text and _normalize_memory_text(mem_text) in asked:
                dropped.append(item)
            else:
                kept.append(item)
        if not dropped:
            return result
        survivors = self._rollback_new_memories(dropped)
        survivor_ids = {(item or {}).get("id") for item in survivors}
        actually_dropped = [
            item for item in dropped if (item or {}).get("id") not in survivor_ids
        ]
        result["results"] = kept + survivors
        result["inference_dropped"] = len(actually_dropped)
        if survivors:
            logger.warning(
                "提取揣摩拦截：%d 条刚问的推断没清掉（仍在库里）。",
                len(survivors),
            )
        return result

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

    建好后顺手对一次账（reconcile_ledger_index）：账本里有、索引里
    没有的，按账本重建索引——写索引那步崩了、或索引坏过，重启后
    靠这一笔把记忆捞回来。对账本身不抛异常、只记日志：对不上也
    不能把服务搞崩，索引里现有的记忆照常能用。
    """
    try:
        store = Mem0Store(budget_tracker=budget_tracker, **kwargs)
    except Exception as exc:
        logger.warning("mem0 记忆未启用：%s，本次不带记忆运行。", exc)
        return None
    try:
        store.reconcile_ledger_index()
    except Exception:
        logger.warning("启动对账失败：本次不对索引补缺。", exc_info=True)
    return store


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
