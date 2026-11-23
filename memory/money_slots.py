# memory/money_slots.py（钱与槽位：金额元数据、核心槽位、槽位冲突）
# 从 memory/mem0_store.py 拆出来，行为不变。

import re

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

# 判断 (a) 号：核心槽位（轻量，不请模型）。
# 实体/数字/日期/动作词/类别词——两条记忆有没有共同槽位，
# 决定它值不值得送模型判“是不是同一件事”。
# 做法（纯程序，便宜）：
#   数字：\d+（带小数也认）；
#   日期：显式日期文本（11 月 11 日 / 2026-11-11），
#         相对日期（今天/昨天）也算；
#   内容词：中文没空格，取去停用词后的相邻二字当词
#         （“鸡杂面”出“鸡杂”“杂面”，“喜欢吃”出“喜欢”“欢吃”——
#         同一件事换说法时，至少有一两个二字词能对上）。
# 返回一组槽位 token（"num:5"、"date:11 月 11 日"、"word:鸡杂"），
# 调用方拿两组求交集：有交集 = 有共同槽位。
_SLOT_STOPWORDS = frozenset(
    "的了是在和我你他她它这那有个不没也都就还把被给让对从到为以及或但若如果因为所以虽然然而可是于是然后接着"
)

_SLOT_RELATIVE_DATES = ("今天", "昨天", "前天", "明天", "大前天", "后天", "昨晚", "今晚")

_SLOT_DATE_RE = re.compile(
    r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?"
    r"|\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?"
    r"|\d{4}-\d{2}-\d{2}"
)

def extract_core_slots(text):
    """一条记忆文本的核心槽位（判断 a 号）。
    返回 set（"num:..." / "date:..." / "word:..."）。
    空文本返回空 set。"""
    slots = set()
    if not isinstance(text, str) or not text.strip():
        return slots
    lowered = text.lower()
    for match in re.finditer(r"\d(?:\d|,\d)*(?:\.\d+)?", lowered):
        slots.add(f"num:{match.group(0)}")
    for match in _SLOT_DATE_RE.finditer(text):
        slots.add(f"date:{match.group(0)}")
    for word in _SLOT_RELATIVE_DATES:
        if word in lowered:
            slots.add(f"date:{word}")
    cleaned = "".join(ch for ch in lowered if ch not in _SLOT_STOPWORDS and ch.strip())
    for i in range(len(cleaned) - 1):
        bigram = cleaned[i:i + 2]
        if len(bigram) == 2:
            slots.add(f"word:{bigram}")
    return slots

def _slot_values_conflict(new_text, old_text):
    """两条记忆是不是槽位值冲突（判断 a 号，硬合并线用）：
    同一个槽位（比如都提“鸡杂面”），值却对着干（一个说喜欢吃、
    一个说过敏）——这种不能直接算重复，得请模型判或留着让 BOOS 看。
    做法（启发式，宁严勿松）：两边有共同内容词，且一边带肯定动作词
    （喜欢/爱/爱吃/要），另一边带否定/排斥动作词（过敏/讨厌/不喜欢/
    不吃/不能吃/不要），就当冲突。没把握时返回 False（不拦硬合并，
    模型判那道关还在）。
    """
    new_lower = (new_text or "").lower()
    old_lower = (old_text or "").lower()
    new_slots = extract_core_slots(new_text)
    old_slots = extract_core_slots(old_text)
    shared_words = {
        slot for slot in (new_slots & old_slots) if slot.startswith("word:")
    }
    if not shared_words:
        return False
    positive = ("喜欢", "爱吃", "爱喝", "想要", "要吃", "要喝")
    negative = ("过敏", "讨厌", "不喜欢", "不吃", "不能吃", "不喝", "不要")
    new_pos = any(word in new_lower for word in positive)
    new_neg = any(word in new_lower for word in negative)
    old_pos = any(word in old_lower for word in positive)
    old_neg = any(word in old_lower for word in negative)
    if (new_pos and old_neg) or (new_neg and old_pos):
        return True
    # 57 号：宾语不同也是槽位值冲突——只对偏好句（喜欢/爱吃/爱喝）用：
    # “BOOS 爱吃冒菜”vs“BOOS 爱吃拉面”，都是肯定偏好、没数字，旧检查
    # 认不出冲突，硬合并线会把两条不同事实并成一条。做法：把两边的泛词
    # 都去掉，剩下实义词再比——冒菜 vs 拉面剩下不同，冲突；“喜欢喝冰美式”
    # vs“喜欢冰美式”剩下都是冰美式，不冲突，照常合并。
    # 注意：只在两边都是偏好句时才比——“助手叫十二”vs“你的名字是十二”
    # 这种换句式的身份陈述，泛词去掉后剩下的也不同，但它是同一件事，
    # 不能拦，留给模型判。
    # （rebase 补记：这段是 main 上 c8b6b05 的去重根治，拆分提交切出来时
    # 还没有，rebase 到新 main 后移植进来，别再丢了。）
    _PREF = ("喜欢", "爱吃", "爱喝")
    if any(w in (new_text or "") for w in _PREF) and any(
        w in (old_text or "") for w in _PREF
    ):
        _FILLER = (
            "BOOS", "我", "你", "他", "她", "它", "这", "那", "有", "个", "不",
            "没", "也", "都", "就", "还", "把", "被", "给", "让", "对", "从",
            "到", "为", "以及", "或", "但", "若", "如果", "因为", "所以", "虽然",
            "然而", "可是", "于是", "然后", "接着", "喜欢", "爱吃", "爱喝",
            "喝", "吃", "想", "要", "的", "了", "是", "在", "和", "助手", "叫",
            "名字", "你的", "我的",
        )
        def _core_text(t):
            s = t or ""
            for w in _FILLER:
                s = s.replace(w, "")
            # 数字里的逗号是格式，不是内容：“25,466.19”和“25466.19”是同一个数。
            s = s.replace(",", "")
            return s.strip(" 。，、")
        new_core = _core_text(new_text)
        old_core = _core_text(old_text)
        if new_core and old_core and new_core != old_core:
            return True
    # 数字不同也是槽位值冲突（“每天喝 2 杯咖啡”vs“每天喝 3 杯咖啡”——
    # 同一话题、数不同，不是同一件事）。去掉逗号后比数值，
    # 25,466.19 和 25466.19 是同一个数，不算冲突。
    def _num_values(slots):
        values = set()
        for slot in slots:
            if not slot.startswith("num:"):
                continue
            raw = slot[4:].replace(",", "")
            try:
                values.add(float(raw))
            except ValueError:
                values.add(raw)
        return values

    new_nums = _num_values(new_slots)
    old_nums = _num_values(old_slots)
    if new_nums and old_nums and new_nums != old_nums:
        return True
    return False

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

