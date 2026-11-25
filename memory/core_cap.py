# memory/core_cap.py（四件套第 1 块：核心上限——座位记账 / 容量门 / 水位）
#
# BOOS 2026-11-24 定的规矩（四件套方案追加决定一/三/十一、施工契约
# 「一、核心契约」）：
#   - 核心座位上限 150，只数「有效核心且非钉死」的成员；
#   - 钉死区（铁律＋性情投影）不占座；临时钉死（拍板 4：散落规矩先
#     挂临时钉死，保护与真钉死同级）同样不占座；
#   - 水位 80%＝120 条，只在月度体检结论页提一句，不单独打扰；
#     140 条是降级腾座的回差目标线（整理员那块用），不是准入线；
#   - 满座时拒绝新晋升、生成候选＋待问，不许自动删任何记忆腾座，
#     更不许「顶掉最旧一条」或整表替换。
#
# 三个概念分开表示（契约钉死，不许再用一个布尔同时表达三者）：
#   - 成员（membership）：是不是核心记忆——唯一口径是
#     provenance._is_core_memory（已确认或带身份槽位），本模块只
#     转调它，不另立第二份定义；
#   - 保护（protection）：metadata 的 protection_tier 字段
#     （"pinned" 钉死 / "temp_pinned" 临时钉死 / 缺失或 None=普通）。
#     它只管占不占座；「能不能被自动改动」是另一回事，仍归
#     sqlite_store._is_core_protected_row 那个护栏谓词管；
#   - 占座（seat）：活行 + 成员 + 保护档为空，三样同时成立才占座。
#
# 分类抽屉（追加决定十一，暂定五类）：metadata 的 core_drawer
# 字段，只认 CORE_DRAWERS 里的键；没标或标错的按「未入格」——
# 计入座位总量、不进任何分块；分块合计 + 未入格必须等于总座位。
# 抽屉只是记账与分类，进抽屉不等于钉死。
#
# 本模块是纯函数层：分类、计数、按账本事件重算，都不读库不写库；
# 持锁读快照、容量门落库在 sqlite_store 里（core_seat_snapshot /
# recount_core_seats_from_ledger / confirm_memory / set_*）。

from memory import provenance as _provenance

# 座位上限（追加决定：座位先定 150 条）。
CORE_SEAT_LIMIT = 150

# 水位提醒线：80%＝120 条（只在月度体检结论页提一句）。
CORE_WATERMARK_SEATS = 120

# 回差目标线：降级腾座腾到 140 条为止（整理员口径），不是准入线。
CORE_HYSTERESIS_TARGET = 140

# 保护档字段与合法值：只认这两个字符串，其余一律按普通（不猜，
# 和 provenance._normalize_metadata 一个规矩）。
PROTECTION_TIER_FIELD = "protection_tier"
TIER_PINNED = "pinned"
TIER_TEMP_PINNED = "temp_pinned"
PROTECTION_TIERS = (TIER_PINNED, TIER_TEMP_PINNED)

# 分类抽屉字段与暂定五类（追加决定十一；键是存库用的稳定标识，
# 值是给人看的名字）。不设 Agent 自身格；以后新开抽屉在这里加键，
# 并按决定十一在结论页/体检报告留痕报 BOOS。
DRAWER_FIELD = "core_drawer"
CORE_DRAWERS = {
    "person": "本人",
    "people": "重要的人",
    "long_term": "长期安排",
    "environment": "环境家当",
    "money_decisions": "钱与决定",
}

# 账本里这两个状态的 key 不是活行（dormant 仍是活行：沉底只是
# 衰减标记，行还在库里、还占座——和 SQLite 活行口径一致）。
_LEDGER_DEAD_STATES = ("superseded", "deleted")

# PROMOTE 事件折分类字段时照常带的字段（PROMOTE 带完整 meta 快照，
# 这些字段以最新一笔为准、但写 None 时不当真——快照里 None 多半
# 是「当时没这字段」，不是显式清掉）。
_PROMOTE_CARRY_FIELDS = (
    "source", "user_confirmed", "slot", "decay_kind", "importance",
    "curated", "is_rule",
)

# 这两个字段以「键在不在」为准：显式写 None 是摘档/摘格，是真
# 动作，不能当没写过（set_protection_tier(None) 靠它生效）。
_PROMOTE_PRESENCE_FIELDS = (PROTECTION_TIER_FIELD, DRAWER_FIELD)


def normalize_protection_tier(meta):
    """读一份 metadata 的保护档：只认 "pinned" / "temp_pinned"，
    其余（缺失、None、写错的字符串）一律 None（普通），不猜。"""
    if isinstance(meta, dict):
        tier = meta.get(PROTECTION_TIER_FIELD)
        if tier in PROTECTION_TIERS:
            return tier
    return None


def normalize_core_drawer(meta):
    """读一份 metadata 的分类抽屉：只认 CORE_DRAWERS 里的键，
    其余一律 None（未入格），不猜。"""
    if isinstance(meta, dict):
        drawer = meta.get(DRAWER_FIELD)
        if drawer in CORE_DRAWERS:
            return drawer
    return None


def is_core_member(meta):
    """是不是核心成员：唯一口径转调 provenance._is_core_memory，
    本模块不另立第二份定义（self_check 早有钉子：这个口径不能有两个）。"""
    return _provenance._is_core_memory(meta)


def classify(meta, live=True):
    """给一份 metadata 分类：成员？保护档？占座？哪个抽屉？

    live=False（已被取代/删除/作废的行）一律不算成员、不占座。
    返回 {"member", "protection_tier", "occupies_seat", "drawer"}；
    drawer 只对占座行有意义（不占座的行不进分块记账），其余为 None。
    """
    tier = normalize_protection_tier(meta)
    member = bool(live) and is_core_member(meta)
    occupies = bool(member and tier is None)
    return {
        "member": member,
        "protection_tier": tier,
        "occupies_seat": occupies,
        "drawer": normalize_core_drawer(meta) if occupies else None,
    }


def count_seats(metas, limit=CORE_SEAT_LIMIT):
    """按一组活行的 metadata 数座位，返回水位快照（纯计数部分）。

    同一快照出数：总数、分抽屉、钉死/临时钉死、未入格、水位比例
    都从这同一批输入算，不许各算各的。reconciled 是自证：
    分块合计 + 未入格 == 总座位，对不上即缺陷。
    """
    metas = list(metas or [])
    drawers = {key: 0 for key in CORE_DRAWERS}
    seats = 0
    members = 0
    pinned = 0
    temp_pinned = 0
    unclassified = 0
    for meta in metas:
        result = classify(meta, live=True)
        if result["protection_tier"] == TIER_PINNED:
            pinned += 1
        elif result["protection_tier"] == TIER_TEMP_PINNED:
            temp_pinned += 1
        if not result["member"]:
            continue
        members += 1
        if not result["occupies_seat"]:
            continue
        seats += 1
        if result["drawer"] is None:
            unclassified += 1
        else:
            drawers[result["drawer"]] += 1
    return {
        "limit": limit,
        "live_rows": len(metas),
        "members": members,
        "pinned": pinned,
        "temp_pinned": temp_pinned,
        "seats_used": seats,
        "seats_free": max(0, limit - seats),
        "ratio": (seats / limit) if limit else 0.0,
        "watermark_seats": CORE_WATERMARK_SEATS,
        "over_watermark": seats >= CORE_WATERMARK_SEATS,
        "hysteresis_target": CORE_HYSTERESIS_TARGET,
        "drawers": drawers,
        "unclassified_seats": unclassified,
        "reconciled": sum(drawers.values()) + unclassified == seats,
    }


def fold_ledger_metas(events):
    """把账本事件折成 {key: 最终 meta}（只折分类要用的字段）。

    ADD 带的是白名单 meta 子集；PROMOTE 带完整 meta 快照，分类
    字段以最新一笔 PROMOTE 为准（protection_tier / core_drawer
    以「键在不在」为准，见 _PROMOTE_PRESENCE_FIELDS）。生死不在
    这里判，由调用方拿 ledger 的 key_states 过滤。
    """
    metas = {}
    for event in events or []:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        key = event.get("key") or event.get("memory_id")
        if not key:
            continue
        if etype == "add":
            metas[key] = dict(event.get("meta") or {})
        elif etype == "promote":
            meta = event.get("meta") or {}
            if not meta:
                continue
            current = dict(metas.get(key) or {})
            for field in _PROMOTE_CARRY_FIELDS:
                if meta.get(field) is not None:
                    current[field] = meta[field]
            for field in _PROMOTE_PRESENCE_FIELDS:
                if field in meta:
                    current[field] = meta[field]
            metas[key] = current
    return metas


def count_seats_from_ledger(events, key_states, limit=CORE_SEAT_LIMIT):
    """按账本事件全量重算座位：应与同时刻的库内快照逐项相等，
    不等即缺陷（契约「一、核心契约」：座位数可由账本事件全量重算）。

    key_states 是 ledger.resolve_state 折出来的 {key: state}；
    superseded / deleted 不算活行，其余（current / dormant /
    没记录）都按活行算——和账本「没说它有问题就别擅自当旧的」
    的老规矩一致。
    """
    metas = fold_ledger_metas(events)
    states = key_states or {}
    live = []
    for key, meta in metas.items():
        if states.get(key) in _LEDGER_DEAD_STATES:
            continue
        live.append(meta)
    return count_seats(live, limit=limit)
