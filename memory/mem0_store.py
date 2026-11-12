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

# VALUE_THRESHOLD（价值关卡：提取出的记忆值不值得长期记的分界线）：
# 43 号：模型给每条新提取的记忆打 0–1 分，低于这个值不进库。
# 0.5 是适中的默认（中点）：稳定事实/明确偏好/重要决定通常 ≥0.7，
# 一次性请求/闲聊/临时状态通常 ≤0.3；拿不准的往低打（提示词里写明），
# 所以 0.5 不会把真事实拦在外面。BOOS 看了可以调（见第 3 批报告）。
VALUE_THRESHOLD = 0.5

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
#   created_at：记忆形成的时间（UTC 的 ISO 时间串，微秒级、严格单调递增，
#           固定 6 位小数、带 Z——见 _now_iso）。
#   write_seq：同一批写进去的序号（进程内严格递增，和 created_at 一起
#           排新旧，见 _memory_order_key）。
#   created_at_source：这个时间是哪来的——"written"（这次写进去的）、
#           "backfilled"（老记忆后补的，见 backfill_timestamps）、
#           "migrated"（迁移时沿用的旧日期，见 migrate_from_v1）。
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

# 可信度分级（覆盖规则用，见 _resolve_conflict）：
#   3 = BOOS 亲口确认的（source=user 且 user_confirmed=True）；
#   2 = 来源是 user、但没逐条确认（旧数据损坏/写坏了才有，正常写不到）；
#   1 = 模型从对话里提取的（inferred）；
#   0 = 没来源信息的旧记忆（unconfirmed）。
# 时间戳决定新旧，可信度决定有没有资格覆盖：级别低的新记忆不能
# 直接覆盖级别高的旧事实。
CREDIBILITY_CONFIRMED = 3
CREDIBILITY_USER = 2
CREDIBILITY_INFERRED = 1
CREDIBILITY_UNCONFIRMED = 0

# created_at 的合法来源（见 _provenance 的 created_at_source 参数）。
CREATED_AT_SOURCES = ("written", "backfilled", "migrated")

# _NOW_LOCK / _NOW_LAST：保证 _now_iso() 发出的时间戳严格递增。
# 秒级精度时，两轮对话同一秒写进去，created_at 一撞就分不出新旧——
# 和 conversation_summary 的 _now_iso 修过的是同一类问题（PR #13，
# BOOS 笔记本上跑全套时撞上过一次）。这里不能直接 import 那边
# （conversation_summary 反过来 import 本模块的 _redact_sensitive，
# 循环了），所以同一套算法、各自一份状态，各管各的域。
# 每次发值时如果新值 <= 上一次发的值，就在上一次的值上加 1 微秒再发。
# 时钟往回调也不会乱：每次只比上一次大 1 微秒，墙上时间追上来就恢复正常。
# 写记忆是多线程的（网页服务），读-改-写上锁，别让两个线程同时拿到同一个值。
_NOW_LOCK = threading.Lock()
_NOW_LAST = None


def _now_iso():
    """现在的时间，UTC 的 ISO 时间串（固定 6 位微秒，带 Z），供 metadata 用。

    小数部分固定 6 位：isoformat() 默认会省略微秒为 0 时的小数点，
    '...T03:00:00Z' 按字符串排会比 '...T03:00:00.000001Z' 大——
    字符串顺序就和时间顺序反了。固定 6 位后，字符串顺序就是时间顺序。
    """
    global _NOW_LAST
    now = datetime.now(timezone.utc)
    with _NOW_LOCK:
        if _NOW_LAST is not None and now <= _NOW_LAST:
            now = _NOW_LAST + timedelta(microseconds=1)
        _NOW_LAST = now
    return now.isoformat(timespec="microseconds").replace("+00:00", "Z")


# _WRITE_SEQ_LOCK / _WRITE_SEQ：同一批写进去的序号（进程内严格递增）。
# 同一批写进去的几条记忆，时间戳也可能撞在同一微秒——单调 +1 微秒能
# 分开绝大多数，但序号是更稳的第二道：同一批里每条各拿一个号，先后
# 一目了然。序号只管同一进程里的先后，跨进程重启从 0 再来——新旧判断
# 先看 created_at，序号只是平局时的决胜（见 _memory_order_key）。
# 真正的写操作本来就串在 _WRITE_LOCK 里，这里这把锁只是兜底。
_WRITE_SEQ_LOCK = threading.Lock()
_WRITE_SEQ = 0


def _next_write_seq():
    """下一条写入序号（进程内严格递增）。"""
    global _WRITE_SEQ
    with _WRITE_SEQ_LOCK:
        _WRITE_SEQ += 1
        return _WRITE_SEQ


# ---------- 金额结构化字段（27 号） ----------
# 背景：BOOS 的两笔钱——每月 5 美元工具预算、每天定投 5 块买万家纳指——
# 数字都是 5，模型只看文字经常揉成一笔。修法：金额类记忆存的时候，
# 把金额、币种、周期、对象拆成结构化字段贴在 metadata 上（money），
# 回答时 _memory_block（main.py）把字段标在每条记忆后面，模型一眼
# 分得清两笔钱；提示词那句“两笔钱没关系”只当兜底（没字段的老记忆用）。
# 只做金额类、不全量：别的记忆不贴字段，不动现有结构。
# 提取是确定性正则、不调模型、不花钱；一条记忆里有两笔钱时只取第一笔
# （两笔钱 normally 各是一条记忆，混在一条里的极少，真混了就靠兜底句）。

_MONEY_CURRENCIES = ("CNY", "USD")
_MONEY_PERIODS = ("day", "week", "month", "year", "once")
# 金额：数字 + 单位（块/元/人民币/¥ 是元，美元/美金/刀/$ 是美元），
# 万是乘数（1 万块 = 10000 元）；$ / ¥ 也可以写在数字前面（$5）。
_MONEY_AMOUNT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万)?\s*(块|元|人民币|美元|美金|刀|\$|＄|¥)"
)
_MONEY_SYMBOL_AMOUNT_RE = re.compile(r"(\$|＄|¥)\s*(\d+(?:\.\d+)?)")
_MONEY_PERIOD_WORDS = (
    ("每天", "day"), ("每日", "day"),
    ("每周", "week"), ("每星期", "week"), ("每个星期", "week"),
    ("每月", "month"), ("每个月", "month"),
    ("每年", "year"), ("每一年", "year"),
    ("一次", "once"), ("一次性", "once"),
)
# 对象：买/定投后面的名词（“买万家纳指”→“万家纳指”）；“投”后面常跟
# 数字（“投 5 块”），数字开头的不算对象；“预算”出现就是工具预算。
# “定投”后面跟“是/的/了”时是名词用法（“我的定投是每天 5 块”），
# 不算对象——先找“买”，再找后面直接跟名词的“定投”。
_MONEY_BUY_TARGET_RE = re.compile(r"买\s*([^\s，。,、；;：:！!？?]{1,12})")
_MONEY_DCA_TARGET_RE = re.compile(
    r"定投\s*([^是的了\s，。,、；;：:！!？?][^\s，。,、；;：:！!？?]{0,11})"
)
# 56 号：定投金额后面的“的+目标”（“每天定投5刀的沪深300”→“沪深300”）——
# 金额夹在“定投”和“的”中间时，上面那条会抠出“5刀的沪深300”又因数字
# 开头被丢掉；这条直接认“定投<金额>的<目标>”。
_MONEY_DCA_AMOUNT_TARGET_RE = re.compile(
    r"定投\s*\d+(?:\.\d+)?\s*(?:万)?\s*(?:块|元|人民币|美元|美金|刀|\$|＄|¥)?\s*的\s*([^\s，。,、；;：:！!？?]{1,12})"
)
# 56 号补（2026-11-12 实测）：定投金额后面直接跟目标、没有“的”也没有
# “买”——模型复述时常把“的”丢了（“每天定投5美元沪深300”→“沪深300”），
# 上面三条都认不出来，target 存成 None，53 号的槽位比对就认不出是同一
# 对象、同值重复的定投就漏进来了。这条认“定投<金额><目标>”。
_MONEY_DCA_AMOUNT_TARGET_NO_DE_RE = re.compile(
    r"定投\s*\d+(?:\.\d+)?\s*(?:万)?\s*(?:块|元|人民币|美元|美金|刀|\$|＄|¥)?\s*([^是的了钱块元美刀\$＄¥\s，。,、；;：:！!？?][^\s，。,、；;：:！!？?]{0,11})"
)
# 上面那条抠出来的目标如果是时间/状语开头（“到年底”“从今天起”“以后”
# 这类），不算目标——宁可 target 为空（53 号走兜底），也别把时间当对象。
_MONEY_NON_TARGET_LEADS = (
    "到", "从", "以后", "之前", "开始", "结束", "为止",
    "每天", "每月", "每年", "每周", "一次",
)


def _extract_money_metadata(text):
    """从一条记忆文本里抠金额结构化字段（27 号）。

    返回 {"amount", "currency", "period", "target"}，没金额返回 None。
    amount 是 int（整数）或 float；currency 是 "CNY"（元）或 "USD"（美元）；
    period 是 "day"/"week"/"month"/"year"/"once"，没写周期就是 None；
    target 是对象（比如“万家纳指”“工具预算”），没写就是 None。
    纯正则，不调模型；抠不出来（写法太怪）就返回 None，不猜。
    “刀”就是美元（56 号：BOOS 说“5刀”就是 5 美元）。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    amount = None
    currency = None
    match = _MONEY_AMOUNT_RE.search(text)
    if match:
        num, wan, unit = match.groups()
        amount = float(num)
        if wan:
            amount *= 10000
        if amount == int(amount):
            amount = int(amount)
        currency = "USD" if unit in ("美元", "美金", "刀", "$", "＄") else "CNY"
    else:
        symbol_match = _MONEY_SYMBOL_AMOUNT_RE.search(text)
        if symbol_match:
            symbol, num = symbol_match.groups()
            amount = float(num)
            if amount == int(amount):
                amount = int(amount)
            currency = "USD" if symbol in ("$", "＄") else "CNY"
    if amount is None:
        return None
    period = None
    for word, code in _MONEY_PERIOD_WORDS:
        if word in text:
            period = code
            break
    target = None
    if "预算" in text:
        target = "工具预算"
    else:
        buy_match = _MONEY_BUY_TARGET_RE.search(text)
        if buy_match:
            candidate = buy_match.group(1).lstrip("的")
            if candidate and not candidate[0].isdigit():
                target = candidate
        if target is None:
            # 56 号：先认“定投<金额>的<目标>”，再认“定投<金额><目标>”
            #（没有“的”的，模型复述常丢“的”），最后认“定投<名词>”。
            dca_amount_match = _MONEY_DCA_AMOUNT_TARGET_RE.search(text)
            if dca_amount_match:
                candidate = dca_amount_match.group(1)
                if candidate and not candidate[0].isdigit():
                    target = candidate
        if target is None:
            dca_no_de_match = _MONEY_DCA_AMOUNT_TARGET_NO_DE_RE.search(text)
            if dca_no_de_match:
                candidate = dca_no_de_match.group(1)
                if (
                    candidate
                    and not candidate[0].isdigit()
                    and not candidate.startswith(_MONEY_NON_TARGET_LEADS)
                ):
                    target = candidate
        if target is None:
            dca_match = _MONEY_DCA_TARGET_RE.search(text)
            if dca_match:
                candidate = dca_match.group(1)
                if (
                    candidate
                    and not candidate[0].isdigit()
                    # 56 号补：时间/状语开头（“定投到年底”里的“到年底”）
                    # 不是目标，和上面那条同一个拦截。
                    and not candidate.startswith(_MONEY_NON_TARGET_LEADS)
                ):
                    target = candidate
    return {
        "amount": amount,
        "currency": currency,
        "period": period,
        "target": target,
    }


def _known_money(item):
    """一条已有记忆的金额字段（53 号）：metadata 里的 money 字典优先，
    没有就从文本里现抠（快照/老数据可能没贴字段）。"""
    money = _normalize_money_metadata(((item or {}).get("metadata") or {}).get("money"))
    if money is not None:
        return money
    return _extract_money_metadata((item or {}).get("text") or "")


def _money_slot_relation(new_money, known_money):
    """53 号：两条金额记忆的槽位关系。
    - 任一边没金额字段 → None（不归槽位管，走老路：向量+模型判）。
    - 对象不明（任一边 target 为空）→ None（没法归槽位，不猜）。
    - （对象，周期）不同 → "different"（同主题不同槽位，分别保留，
      别拿去给模型判重复——向量相似度分不开“万家纳指”和“沪深300”）。
    - （对象，周期）相同：金额+币种相同 → "duplicate"（重复，不存新的）；
      不同 → "supersede"（同槽位新值，旧条更新成新说法）。
    """
    if not new_money or not known_money:
        return None
    new_target = new_money.get("target")
    known_target = known_money.get("target")
    if not new_target or not known_target:
        return None
    if new_target != known_target or new_money.get("period") != known_money.get("period"):
        return "different"
    if (
        new_money.get("amount") == known_money.get("amount")
        and new_money.get("currency") == known_money.get("currency")
    ):
        return "duplicate"
    return "supersede"


def _normalize_money_metadata(raw):
    """校验 money 字段的形状（读回来时用，见 _normalize_metadata）。

    形状不对（比如旧数据、手写坏了）返回 None，不猜、不将就——
    字段是辅助，文本本身还在，丢了字段最多走兜底句。
    接受两种形状：字典形 {"amount", "currency", "period", "target"}
    （代码里传来传去的形状）和扁平形 {"money_amount", ...}
    （真正存进 Chroma 的形状，见 _flatten_money）。
    """
    if not isinstance(raw, dict):
        return None
    amount = raw.get("amount", raw.get("money_amount"))
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return None
    currency = raw.get("currency", raw.get("money_currency"))
    if currency not in _MONEY_CURRENCIES:
        return None
    period = raw.get("period", raw.get("money_period"))
    if period is not None and period not in _MONEY_PERIODS:
        return None
    target = raw.get("target", raw.get("money_target"))
    if target is not None and (not isinstance(target, str) or not target):
        return None
    return {
        "amount": amount,
        "currency": currency,
        "period": period,
        "target": target,
    }


# Chroma（BOOS 笔记本和这里用的向量库）只接受标量 metadata 值
# （str/int/float/bool/list/None），dict 值会直接炸掉整条 add/update——
# 27 号第一版把 money 存成一个 dict，真调时 add_memory_text 在 Chroma
# 那层就 ValueError，金额记忆根本没存进去（第 4 批真调抓到的）。
# 所以 money 存成四个扁平标量字段，读回来（_normalize_metadata）
# 再拼回字典形。字段名都带 money_ 前缀，别和别的字段撞上。
_MONEY_FLAT_KEYS = ("money_amount", "money_currency", "money_period", "money_target")


def _flatten_money(money):
    """把 money 字典展平成 Chroma 能存的四个标量字段。

    money 是 None 时四个字段都显式给 None——纠正时新说法没金额，
    旧的金额字段要清掉（mem0 的 update 是合并，不传就留着旧值）。
    形状不对时当 None 处理（清掉），不把坏形状往库里写。
    """
    flat = dict.fromkeys(_MONEY_FLAT_KEYS)
    clean = _normalize_money_metadata(money)
    if clean is not None:
        flat["money_amount"] = clean["amount"]
        flat["money_currency"] = clean["currency"]
        flat["money_period"] = clean["period"]
        flat["money_target"] = clean["target"]
    return flat


# _MONEY_UNSET：_provenance 的 money 参数没传——不碰 money 字段。
# 传 None 是“显式清掉”（纠正时新说法没金额，旧的金额字段不能留着）。
_MONEY_UNSET = object()


def _provenance(source, confidence, user_confirmed, created_at=None, slot=None,
                created_at_source="written", is_rule=False, money=_MONEY_UNSET):
    """拼一份来源元数据（存记忆时传给 mem0.add 的 metadata 参数）。

    source 不是 "user" 时一律按 "inferred" 存（调用方传错也不让它
    进库）；user_confirmed 只在 source="user" 时才可能是 True。
    slot（Plan C）：身份/称呼类记忆的槽位名（agent_name/user_name/
    soul），只认 KNOWN_SLOTS 里的，其他值不进库——同槽位只留一条，
    新值盖旧值时靠它找旧记忆。
    is_rule（第 3 批 42 号）：纠正后总结的通用规则（比如“定投只在交易日
    扣款”）才记 True——检索时优先捞出来，回答同类问题先套用规则再算。
    created_at 不传时取新鲜的单调时间（_now_iso）；传了就用传的
    （迁移沿用旧日期、纠错回滚恢复旧时间时用）。created_at_source
    说明这个时间是哪来的，只认 CREATED_AT_SOURCES 里的三个值，
    传错了按 "written" 算——时间来源不能靠猜，必须是明确的。
    money（27 号）：金额结构化字段（_extract_money_metadata 的结果）；
    没传（_MONEY_UNSET）就不碰 money 字段；传 None 是显式清掉
    （纠正时新说法没金额，旧的金额字段不能留着）；传了字典就校验后
    展平成四个标量字段存（见 _flatten_money，Chroma 不接受 dict 值）。
    每次调都新拿一个 write_seq（同一批里的先后靠它分）。
    """
    source = SOURCE_USER if source == SOURCE_USER else SOURCE_INFERRED
    if created_at_source not in CREATED_AT_SOURCES:
        created_at_source = "written"
    metadata = {
        "source": source,
        "created_at": created_at or _now_iso(),
        "created_at_source": created_at_source,
        "write_seq": _next_write_seq(),
        "confidence": confidence,
        "user_confirmed": bool(user_confirmed) and source == SOURCE_USER,
    }
    if slot in KNOWN_SLOTS:
        metadata["slot"] = slot
    if is_rule:
        metadata["is_rule"] = True
    if money is not _MONEY_UNSET:
        # None 也存：把旧的金额字段清掉（纠正时新说法没金额就清掉，
        # 不然“每天投 5 块”改成“我不投了”，5 块的字段还贴在上面）。
        # 展平成四个标量字段存（见 _flatten_money）：Chroma 不接受
        # dict 值的 metadata，存 dict 会炸掉整条写入。
        metadata.update(_flatten_money(money))
    return metadata


def _normalize_metadata(raw, item=None):
    """把 mem0 返回的 metadata 规整成统一形状（检索时用）。

    规整规则（安全降级，宁可信度低、不可误认）：
    - raw 不是字典、或缺 source、或 source 不是 user/inferred，
      一律按 "unconfirmed" 处理（旧记忆、新旧版本混用、损坏数据
      都走这条路）；
    - user_confirmed 只有明确是 True（布尔值）才算确认，写成
      "yes"/1 这类非布尔值不算——确认必须是明确的，不能靠猜；
    - confidence 不是 0–1 之间的数字时记成 None（不知道就是不知道）；
    - created_at 不是字符串时记成 None；
    - write_seq 不是整数时记成 None；created_at_source 不是三个合法值
      时记成 None；pending_confirmation 只有明确是 True（布尔值）
      才算待确认——这三样也一样，不能靠猜；
    - slot 不是 KNOWN_SLOTS 里的已知槽位时记成 None（旧记忆、新旧
      版本混用、损坏数据都走这条路——槽位只认这三个，别的都不认）。
    - money（27 号）：金额结构化字段，形状不对就记 None（见
      _normalize_money_metadata），不猜。
    created_at 的第二来源（33 号）：mem0 2.2.1 读回来时把 created_at
    提到条目的顶层（get/get_all/search 的返回里，created_at 是
    MemoryItem 的顶层字段），同时从返回的 metadata 里剔掉——所以
    只看 metadata 永远找不到时间，面板上 14 条全是“未知时间”。
    根因不在写的那头：写的时候 _provenance 每条都带了显式时间，
    mem0 存的时候也留着（add 时缺了它自己还会补一个）。修法：调
    用方把整条 item 也传进来（item 参数），metadata 里没有时再到
    顶层找——metadata 里的优先，顶层是兜底。没传 item 时只看 metadata，
    行为和以前一样。
    返回：{"source", "created_at", "created_at_source", "write_seq",
           "confidence", "user_confirmed", "confirmed", "slot",
           "pending_confirmation", "is_rule", "money"}，其中 confirmed
    就是 user_confirmed（取了个短名字，方便调用方判断“能不能当作事实用”）。
    """
    source = SOURCE_UNCONFIRMED
    created_at = None
    created_at_source = None
    write_seq = None
    confidence = None
    user_confirmed = False
    slot = None
    pending_confirmation = False
    is_rule = False
    money = None
    if isinstance(raw, dict):
        raw_source = raw.get("source")
        if raw_source in (SOURCE_USER, SOURCE_INFERRED):
            source = raw_source
        raw_created = raw.get("created_at")
        if isinstance(raw_created, str) and raw_created:
            created_at = raw_created
        raw_created_source = raw.get("created_at_source")
        if raw_created_source in CREATED_AT_SOURCES:
            created_at_source = raw_created_source
        raw_seq = raw.get("write_seq")
        if isinstance(raw_seq, int) and not isinstance(raw_seq, bool):
            write_seq = raw_seq
        raw_conf = raw.get("confidence")
        if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool):
            if 0 <= raw_conf <= 1:
                confidence = float(raw_conf)
        user_confirmed = raw.get("user_confirmed") is True and source == SOURCE_USER
        raw_slot = raw.get("slot")
        if raw_slot in KNOWN_SLOTS:
            slot = raw_slot
        pending_confirmation = raw.get("pending_confirmation") is True
        # is_rule 只有明确是 True（布尔值）才算规则，写成 "yes"/1 不算——
        # 规则要在检索时优先捞出来，不能靠猜。
        is_rule = raw.get("is_rule") is True
        # 27 号：money 字典形（旧数据/代码里传的）或扁平形（真正存在
        # Chroma 里的四个 money_* 字段）都认，拼回字典形给调用方。
        money = _normalize_money_metadata(raw.get("money"))
        if money is None:
            money = _normalize_money_metadata(raw)
    if created_at is None and isinstance(item, dict):
        # 33 号兜底：mem0 把 created_at 提到条目顶层了，metadata 里没有
        # 就到顶层找。顶层的时间就是 payload 里存的那个，不是现编的。
        top_created = item.get("created_at")
        if isinstance(top_created, str) and top_created:
            created_at = top_created
    return {
        "source": source,
        "created_at": created_at,
        "created_at_source": created_at_source,
        "write_seq": write_seq,
        "confidence": confidence,
        "user_confirmed": user_confirmed,
        "confirmed": user_confirmed,
        "slot": slot,
        "pending_confirmation": pending_confirmation,
        "is_rule": is_rule,
        "money": money,
    }


# ---------- 时间戳 + 覆盖规则（第一步，2026-11-10） ----------
# 规则（BOOS 定的）：时间戳决定新旧，来源可信度决定有没有资格覆盖。
# 同级里新的胜出，旧的标“被取代”但原话留着；低可信度的新记忆不能
# 直接覆盖高可信度的旧事实——新记忆挂起来“待确认”，只有真用上时
# 才问 BOOS；时间戳相同或缺失、真分不出先后的，问 BOOS；核心记忆
# 钉住不动，模型推断一律不能顶掉。
# 这几个函数都是纯函数（只“判”，不动库），调用方见 _resolve_conflict
# 的注释——它不管两条记忆是不是同一件事，“是不是同一件事”由调用方
# 按各自的证据定（槽位相同、BOOS 亲口指的、提取后守卫的关键词邻居）。


def _credibility_rank(metadata):
    """一条记忆的可信度级别（覆盖规则用，见 _resolve_conflict）。

    只看规整过的 metadata：confirmed（user 且逐条确认）最高；
    source=user 但没确认次之；inferred 再次；没来源信息的最低。
    raw 和规整过的都能传——里面先过一遍 _normalize_metadata，
    所以两种形状结果一样。
    """
    meta = _normalize_metadata(metadata)
    if meta["confirmed"]:
        return CREDIBILITY_CONFIRMED
    if meta["source"] == SOURCE_USER:
        return CREDIBILITY_USER
    if meta["source"] == SOURCE_INFERRED:
        return CREDIBILITY_INFERRED
    return CREDIBILITY_UNCONFIRMED


def _is_core_memory(metadata):
    """是不是核心记忆（第一步的口径）。

    核心记忆 = 原四类（名字、SOUL、你的决定、钱的规矩）+ 安全权限 +
    项目基础决策。第一步里先按“用户已确认”来认：名字/SOUL 有槽位
    标记，其余几类都是 BOOS 亲口确认过的记忆（记住/纠正/迁移都记
    source=user、user_confirmed=True）；对话里随口说的决定是提取的
    （inferred），要等 BOOS 逐条确认才算核心——这是 Plan C 已有的
    区分，第一步不改它。以后做衰减/删除时，这里换成正式的
    “类别/重要性”字段（见下面“以后留的位置”）。
    raw 和规整过的都能传。
    """
    meta = _normalize_metadata(metadata)
    return bool(meta["confirmed"]) or meta["slot"] in KNOWN_SLOTS


def _parse_created_at(value):
    """把 created_at 字符串解析成带时区的 datetime，解析不出返回 None。

    认三种写法：新版微秒 + Z（"2026-11-10T12:00:00.000001Z"）、
    旧版秒级 +00:00（"2026-11-09T10:00:00+00:00"）、迁移带来的
    纯日期（"2026-11-07"，算当天 0 点）。没时区的当 UTC。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        if len(text) == 10:
            parsed = datetime.strptime(text, "%Y-%m-%d")
            return parsed.replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _memory_order_key(metadata):
    """一条记忆的新旧排序键：(形成时间, 写入序号)。

    没时间戳的按最早算：缺时间戳的都是时间戳功能上线之前存的，
    一定比新写的旧——这是时间线本身，不是猜；序号缺失按 -1 算。
    后补的时间戳也按没时间算：backfill 补时间时是按库里随手排的
    顺序给的“现在时间”，根本不知道历史上谁先谁后——那个先后是
    假的，不能拿来判新旧。两条后补的排出来键一样，_resolve_conflict
    就会走“分不出先后、问 BOOS”那条；后补的对新写的，新写的赢，
    因为后补的本来就是功能上线前的老记忆，本来就比新写的旧。
    迁移沿用的旧日期（migrated）是旧数据里的真日期，照常用。
    raw 和规整过的都能传。
    """
    meta = _normalize_metadata(metadata)
    if meta.get("created_at_source") == "backfilled":
        return (datetime.min.replace(tzinfo=timezone.utc), -1)
    when = _parse_created_at(meta.get("created_at"))
    if when is None:
        when = datetime.min.replace(tzinfo=timezone.utc)
    seq = meta.get("write_seq")
    if not isinstance(seq, int) or isinstance(seq, bool):
        seq = -1
    return (when, seq)


def _resolve_conflict(old_metadata, new_metadata):
    """新旧两条记忆打架时，时间戳 + 可信度怎么判（纯函数，不动库）。

    判法：
    - 两条都没有时间戳、或时间戳 + 序号全撞：真分不出先后，问 BOOS。
      （只有一边缺时间戳不问：缺的那条一定是时间戳功能上线之前存的，
      一定比新写的旧——这是时间线本身，不是猜。后补的时间戳在排序
      时按没时间算（见 _memory_order_key），所以两条后补的也走问 BOOS。）
    - 新的可信度比旧的低：旧的赢，新记忆挂起来“待确认”（pending_new），
      不覆盖——时间戳再新也没用，新旧管不了覆盖资格。
    - 新的可信度比旧的高：新的胜出，旧的标“被取代”（supersede_old）。
    - 同级：新的胜出，旧的标“被取代”；但新记忆比旧的还早（迟到的
      旧写入）时不覆盖，旧的留着（keep_old）。
    - 核心记忆钉住不动：核心记忆都是用户已确认的（见 _is_core_memory），
      可信度最高，模型推断（最低档）永远判成 pending_new——上面
      “新低旧高”那条已经盖住了，这里不用再单列。
    返回 {"winner": "old" | "new" | "ask",
           "action": "supersede_old" | "pending_new" | "keep_old" | "ask",
           "reason": 给人看的一句话}。
    注意：这个函数只“判”，不“认”——它不管两条记忆是不是同一件事。
    “是不是同一件事”由调用方按各自的证据定：同槽位就是同一件事
    （_upsert_slot_memory）；纠正是 BOOS 亲口指的（correct_memory）；
    提取后守卫只凭关键词重合找“同一话题”的邻居（_apply_coverage_guard），
    所以同级的新胜出在那里不自动标被取代——关键词重合分不出
    “同一条事实的新版本”还是“同一话题的不同事实”（“喜欢喝咖啡”
    和“喜欢喝拿铁”能同时是真的），宁可不标，不标错。等以后有模型判
    “同一事实”时，再把 supersede_old 接到提取后守卫上。
    """
    old = _normalize_metadata(old_metadata)
    new = _normalize_metadata(new_metadata)
    old_rank = _credibility_rank(old)
    new_rank = _credibility_rank(new)
    old_key = _memory_order_key(old)
    new_key = _memory_order_key(new)
    if old.get("created_at") is None and new.get("created_at") is None:
        return {
            "winner": "ask",
            "action": "ask",
            "reason": "两条记忆都没有时间戳，分不出先后，请 BOOS 定。",
        }
    if old_key == new_key:
        return {
            "winner": "ask",
            "action": "ask",
            "reason": "两条记忆时间戳和序号都一样，分不出先后，请 BOOS 定。",
        }
    if new_rank < old_rank:
        return {
            "winner": "old",
            "action": "pending_new",
            "reason": "新记忆可信度比旧的低，不能直接覆盖，挂起来待确认。",
        }
    if new_rank > old_rank:
        return {
            "winner": "new",
            "action": "supersede_old",
            "reason": "新记忆可信度比旧的高，新的胜出，旧的标被取代。",
        }
    if new_key > old_key:
        return {
            "winner": "new",
            "action": "supersede_old",
            "reason": "同一级别，新的胜出，旧的标被取代。",
        }
    return {
        "winner": "old",
        "action": "keep_old",
        "reason": "同一级别，但新记忆比旧的还早，不覆盖。",
    }


# 以后留的位置（第二步衰减 / 删除时再用，第一步不实现，只把名字占住，
# 免得以后各写各的、又要迁移一次）：
#   - "importance"（重要性，high/normal/low）：和用得多不多分开算——
#     低频不等于低价值（原则②）；衰减速度看 importance，不看调用次数。
#     钉住的核心记忆 importance=high、衰减是零。
#   - "superseded"（被取代标记，bool）：同级新胜出时旧记忆标上，原话保留；
#     第一步里只有 _resolve_conflict 会判，真正标上去要等模型能判断
#     “同一条事实”时再接（见 _resolve_conflict 注释）。
#   - 真删前检查（原则③）：删之前查三样——是不是核心记忆、是不是
#     BOOS 明确让记的、有没有别的记忆连着它；有一条不满足就继续
#     归档、不删。delete_memory() 现在不改。


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


# 碎片纠正：纠正新说法是个碎片（不完整的一句）时，先补成整句再改。
_FRAGMENT_MAX_LEN = 12  # 碎片一般很短，超过这个长度就当完整的一句处理
# 含糊指代：值本身没有内容（“那个”“这个”“它”），补不出来，直接走老路。
_VAGUE_FRAGMENT_VALUES = (
    "那个", "这个", "它", "他", "她", "那", "这",
    "那个东西", "这个东西", "那东西", "这东西",
)


def _detect_correction_fragment(text):
    """认出纠错新说法是不是个“碎片”（不完整的一句）。

    碎片 = “是 + 值”（“是绿色”）或光秃秃一个短语（“绿色”“喜欢绿色”）。
    返回碎片里的“值”（“绿色”），不是碎片时返回 None。
    完整的一句（“我喜欢绿色。”）有主语，不算碎片。
    """
    if not isinstance(text, str):
        return None
    s = text.strip().rstrip("。！？；，、.!?;,").strip()
    if not s or len(s) > _FRAGMENT_MAX_LEN:
        return None
    if s.startswith("是"):
        value = s[1:].strip("，,：:、；; \t")
        return value or None
    # 光秃秃一个短语：不以主语开头——“绿色”“喜欢绿色”。
    if s[0] in ("我", "你", "他", "她", "它", "这", "那"):
        return None
    return s


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
SLOT_COMPUTER = "computer"  # BOOS 的电脑配置（17 号种子）
KNOWN_SLOTS = frozenset([SLOT_AGENT_NAME, SLOT_USER_NAME, SLOT_SOUL, SLOT_COMPUTER])

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

# 电脑配置种子（17 号：BOOS 的电脑配置也种进长期记忆当已确认，
# 评审技术方案、回答“我的电脑能不能跑这个”时先拿这条当依据）。
# 配置以 BOOS 2026-11-06 发来的系统信息为准：Windows 11 Home 23H2、
# ASUS Vivobook K3402ZA、i5-12500H、16GB 内存、Python 3.12.10、
# Git、VS Code；仓库在 D:\代码\agent，文件工具的工作区围栏是
# D:\代码\十二的工作区。硬盘容量截图里没显示，不要编，就不写。
# 和身份种子一样，只在“没有时”才加，绝不覆盖已有值。
COMPUTER_VALUE = "windows-pc"
COMPUTER_TEXT = (
    "BOOS 的电脑配置：Windows 11 Home 23H2，ASUS Vivobook K3402ZA，"
    "i5-12500H，16GB 内存，Python 3.12.10，Git，VS Code。"
    "仓库在 D:\\代码\\agent，文件工具的工作区围栏是 D:\\代码\\十二的工作区。"
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
_IDENTITY_AGENT_TRIGGERS = ("以后你叫", "现在你叫", "你的名字是", "你的名字叫", "你叫", "助手叫", "助手的名字是", "助手的名字叫")
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


# 保护失败时结果里附带的列表字段：没恢复回来的旧记忆 id、回滚删不掉
# 的新记忆 id 和原文。保护信息要经过守卫 → 提交 → add_turn 三层往上传，
# 以前每层手写一个键一个键地传，加个字段就得改三处，漏一处就半路丢了
# （回滚残留的原文就是这么丢的：守卫算出来了，只进了保护事件，没进
# 返回结果）。现在三层都调 _carry_protection_fields，以后再加字段只改
# 下面这个元组一行。
_PROTECTION_RESULT_KEYS = (
    "failed_ids",
    "rollback_leftover_ids",
    "rollback_leftover_texts",
)


def _carry_protection_fields(src, dst):
    """把保护失败的字段从一个结果字典带到另一个。

    只在源字典标了 protection_failed 时才带；没标时什么都不动。
    只填空、不覆盖：目标字典里已经有 protection_failed 时，目标
    那份是更新鲜的（比如纠错返回路里，本轮自己的提取也失败了），
    原样留着、一个键都不盖——上一轮的只在目标没有时才补。不能按
    键一个一个填：上一轮的 failed_ids、残留 id/原文都是上一轮那次
    失败的，单拎几个键填进本轮的失败里会串味（比如原因写着快照
    失败，failed_ids 却是上一轮没恢复回来的旧记忆）。
    失败标记和原因照原样带（原因即使是 None 也带，和以前手写时
    一样）；三个列表字段只在非空时带，空的就不写进结果——返回的
    样子不变，同样的顶层键，调用方（web 面板、命令行）不用动。
    """
    src = src or {}
    if not src.get("protection_failed"):
        return
    if dst.get("protection_failed"):
        return
    dst["protection_failed"] = True
    dst["protection_reason"] = src.get("protection_reason")
    for key in _PROTECTION_RESULT_KEYS:
        value = src.get(key)
        if value:
            dst[key] = value


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
                metadata = _normalize_metadata(item.get("metadata"), item)
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
                            metadata = _normalize_metadata(item.get("metadata"), item)
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

    def _snapshot_all(self):
        """拍一张库快照（提取前、恢复核对时用）。

        内容口径和 get_all(include_sensitive=True) 一样：全部记忆，
        空文本跳过，metadata 规整过。区别只有一点：读库失败时抛
        异常、不吞掉——调用方要分得出“库是空的”还是“库读不出来”，
        分不出来就没法失败关闭（快照读不出来还往里写，就是没保护
        硬写，见 _extract_and_save_turn）。
        """
        raw = self._memory.get_all(filters={"user_id": self.user_id})
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

    def _extract_and_save_turn(self, safe_messages):
        """正常提取保存（add_turn 的老路子，只做这一件事）。

        调 mem0.add 提取本轮对话、去重、存入；提取出的记忆里万一
        还带着敏感信息，_drop_sensitive_results 会再删掉。
        再过一道覆盖守卫（_apply_coverage_guard）：刚提取出的推断记忆
        别把 BOOS 确认过的事实顶掉——守卫要拿提取前的库快照找邻居，
        所以快照在调 mem0.add 之前拍（快照里没有本次新存的，就不会
        把新记忆和自己/同一批的兄弟认成邻居）；快照还有第二个用处：
        万一 add 在里面动了旧记忆，守卫按快照原样恢复（见
        _restore_snapshot_memories）。
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
        metadata = _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
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
        result = self._apply_coverage_guard(result, before, metadata)
        # 43 号价值关卡先走（一次判全场，低价值的先清掉），再 46 号去重
        # （对照面是提取前的库 + 本轮已留下的新记忆）。
        result = self._value_gate_extraction_results(result)
        result = self._dedupe_extraction_results(result, before)
        # 27 号：提取出的新记忆里带金额的，把结构化金额字段贴上。
        # 提取时 metadata 是整批共用的、不知道每条写了什么，只能事后
        # 按文本贴；放在去重之后（被删掉的就不用贴了）。
        return self._attach_money_metadata(result)

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
                self._memory.update(memory_id=mem_id, metadata=_flatten_money(money))
            except TypeError:
                logger.warning("金额字段：底层 update 不支持 metadata 参数，本条没贴上。")
            except Exception:
                logger.warning("金额字段：贴结构化金额失败。", exc_info=True)
        return results

    # ---------- Plan C：待定区提交 + 同槽位 UPDATE + 身份种子 ----------

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
        extracted = {}
        filtered = _filter_extraction_messages(fresh)
        if filtered:
            extracted = self._extract_and_save_turn(filtered)
            results = (extracted or {}).get("results") or []
            # 保护失败的标记要往上传：本轮没写成、或回滚过，调用方
            # （add_turn / 面板）得知道，不能只留在提取那一层。
            # 用共用的 _carry_protection_fields 一次带过去——以前这里
            # 手写一个键一个键地传，残留原文就是这么漏掉的。
            protection_failed = bool((extracted or {}).get("protection_failed"))
            extraction_failed = bool((extracted or {}).get("extraction_failed"))
        failed = protection_failed or extraction_failed
        if not failed:
            # 提取成功（或过滤后什么都不剩，本来就没什么可提）：指纹记下来，
            # 下次提交同一批消息时不再重复处理。
            for key in fresh_keys:
                if key and key not in self._committed_user_keys:
                    self._committed_user_keys.append(key)
        if not results and not identity_outcomes and not protection_failed and not extraction_failed:
            return {}
        out = {"results": results, "identity": identity_outcomes}
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
            return self.add_turn(messages)
        if self._outbox is None:
            self._outbox = MemoryOutbox(self)
            self._outbox.start()
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
                        pending.get("new_text") or "", source=SOURCE_USER
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
                    self._pend_correction_remainder(safe_messages, latest_user)
                    # 上一轮提交的保护失败不能丢在半路：纠错结果里
                    # 没有保护字段，上一轮的才补进来；只填空、不覆盖。
                    _carry_protection_fields(committed, handled)
                    return handled
                # not_found：库里没有冲突的旧记忆——新说法当成一条新记忆，
                # 走正常提取保存（下面），结果里附上这次纠错的尝试记录。
                normal = self._extract_and_save_turn(safe_messages)
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
            }
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
            raw = self._memory.get_all(filters={"user_id": self.user_id})
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
                try:
                    self._mem0_add(
                        [{"role": "user", "content": old_text}],
                        self.user_id,
                        False,
                        dict(old_meta),
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
            try:
                try:
                    self._memory.update(
                        memory_id=mem_id, text=old_text, metadata=dict(old_meta)
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

    def _value_gate_extraction_results(self, results):
        """价值关卡（43 号）：提取出的新记忆，值不值得长期记的再判一次。

        提取提示词是第一道防线（只抽“值得长期记的”），这里是机制上的
        第二道：请模型给每条新记忆打 0–1 分，低于 VALUE_THRESHOLD 的
        从库里删掉、不进 results。拿不准的往低打（提示词里写明），
        所以 0.5 的档位不会把真事实拦在外面。
        判不出来（没客户端、调用失败、回的不是 JSON）时全留——关卡
        失败关闭，别把真事实拦在外面。results 里没有新记忆时直接返回。
        删掉的条数记在 results["value_dropped"] 里（调用方和面板看得见）。
        """
        items = (results or {}).get("results") or []
        if not items:
            return results
        scored = self._judge_value_scores([
            {"id": (item or {}).get("id"), "text": (item or {}).get("memory") or ""}
            for item in items
            if (item or {}).get("id") and (item or {}).get("memory")
        ])
        if scored is None:
            return results
        dropped = []
        kept = []
        for item in items:
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
        return updated

    def _dedupe_extraction_results(self, results, before_items):
        """提取路径去重（46 号）：新提取的记忆里，库里已有同一件事的删掉。

        顺序：先精确匹配（完全相同），再近似匹配（向量预筛 + 模型判同一件事）。
        对照面是两部分：提取前的库（before_items）+ 本轮已留下的新记忆
        （同一轮里模型抽出两条同一件事时也只留一条）。
        删掉的条数记在 results["dedup_dropped"] 里。判不出来时只做精确匹配，
        近似的照常留（宁漏勿错）。删除失败的（还在库里）留在 results 里，
        不能假装删掉了。
        """
        items = (results or {}).get("results") or []
        if not items:
            return results
        known = list(before_items or [])
        embed_cache = {}
        dropped = []
        kept = []
        superseded_count = 0
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
            # 近似：向量预筛 + 模型判同一件事（不同槽位的已排除在外）。
            near = self._find_near_duplicate_memory(mem_text, judge_pool, embed_cache)
            if near is not None:
                dropped.append(item)
                continue
            kept.append(item)
            known.append({"id": mem_id, "text": mem_text})
        if not dropped:
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
        if survivors:
            logger.warning("提取去重：%d 条重复记忆没删掉（仍在库里）。", len(survivors))
        return results

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
            # 53 号：金额记忆先按（对象，周期）槽位归并，再判近重复。
            # 同槽位同金额是重复（返回已有那条）；同槽位新金额是更新
            # （旧条原地改成新说法，id 不变）；不同槽位分别保留，
            # 不拿去给模型判重复（向量相似度分不开“万家纳指”和“沪深300”）。
            new_money = _extract_money_metadata(safe_text)
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
            # 27 号：带金额的记忆，把结构化金额字段贴上（金额/币种/周期/对象），
            # 回答时 _memory_block 标在每条后面，模型分得清两笔钱；没金额
            # 就不传 money 参数（不往 payload 里写 money=None 的空字段）。
            money = _extract_money_metadata(safe_text)
            metadata = _provenance(
                source,
                CONFIDENCE_USER if source == SOURCE_USER else CONFIDENCE_INFERRED,
                source == SOURCE_USER,
                money=money if money is not None else _MONEY_UNSET,
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
                _provenance(
                    SOURCE_USER,
                    CONFIDENCE_USER,
                    True,
                    created_at=created_at,
                    created_at_source="migrated",
                ),
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
                raw = self._memory.get_all(filters={"user_id": self.user_id})
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

    def _find_near_duplicate_memory(self, text, existing_items,
                                    embed_cache=None):
        """在已有记忆里找和 text 是同一件事的那条（46 号）。

        两步：先按向量相似度预筛（≥ DEDUP_SIMILARITY_MIN 的才值得判），
        再请模型判是不是同一件事——只看相似度分不开（“5 块”vs“10 块”
        相似度 0.934，比真重复还高），必须模型点头才算重复。
        找到返回那一项（{"id", "text", ...}），找不到返回 None。
        模型判不出来时也返回 None（宁漏勿错：漏合并只是多一条记忆，
        错合并是丢一条事实）。embedding 拿不到时返回 None——这里只做
        近似，完全相同是 _find_duplicate_memory 的活儿。
        embed_cache（可选）：{归一化文本: 向量}，提取路径一次验多条
        新记忆时共用，同一条文本只算一次。
        """
        if not text or not existing_items:
            return None
        # 完全相同是 _find_duplicate_memory 的活儿，这里先认（不用 embedding）——
        # embedding 拿不到时（比如离线测试的假对象没装 embedding 模型），
        # 精确匹配照样能走，不能因为嵌不出来就连完全相同的都认不出。
        exact = _find_duplicate_memory(text, existing_items)
        if exact is not None:
            return exact
        cache = embed_cache if isinstance(embed_cache, dict) else {}
        key = _normalize_memory_text(text)
        vec = cache.get(key)
        if vec is None:
            vec = self._embed_text(text)
            if vec is None:
                return None
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
            if sim >= DEDUP_SIMILARITY_MIN:
                scored.append((sim, item))
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0], reverse=True)
        candidates = [item for _, item in scored[:5]]
        return self._judge_same_fact(text, candidates)

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
        result = self._add_text_with_metadata(rule, metadata)
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
            if not any((e or {}).get("id") == memory_id for e in trash):
                trash.append({
                    "id": memory_id,
                    "text": item.get("text") or "",
                    "metadata": item.get("metadata") or {},
                    "retracted_at": _now_iso(),
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
            return memory_id not in after

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
