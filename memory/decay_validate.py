# memory/decay_validate.py（衰减参数离线验证：纯函数，只算、不改参数）
# 背景（2026-11-19，BOOS 定）：十二的衰减参数（稳定事实半衰期 180 天、
# 一次性经历 14 天、权重下限 0.10）全是猜的，没拿真实数据验过；有了
# 真实数据再调，代码先只做验证工具、不改参数。本文件就是这套验证工具。
# 只用标准库 + decay.py；不读库、不写库、不联网、不调任何真实 API。
# 注意：decay.py 里的 decay_weight() 把当前常量写死了（不接受参数），
# 所以这里按它的指数衰减公式原样重算，只是把半衰期和下限换成每组候选
# 参数——公式和 decay_weight 一致（0.5 ** (闲置天数 / 半衰期)，下限托底，
# 核心记忆恒 1.0，没时间的老记忆按 1.0 算），不是另起一套公式。
# 两个函数都只给结论和建议，绝不自动改参数：调不调、调成哪组，由 BOOS
# 拿了真实数据、看了验证结果之后亲口定。

import statistics
from datetime import datetime, timezone

from memory.decay import _decay_idle_days
from memory.timeutil import _parse_created_at

# 默认扫的参数网格：稳定半衰期 × 一次性半衰期 × 下限 = 3 × 3 × 3 = 27 组。
# 中间那组（180 天、14 天、0.10）就是现在线上在用的值。
DEFAULT_STABLE_HALF_LIVES = (90, 180, 365)
DEFAULT_EPISODIC_HALF_LIVES = (7, 14, 30)
DEFAULT_FLOORS = (0.05, 0.10, 0.20)

# 样本量门槛：记忆不到 100 条、或观测到的检索次数不到 100 次，
# 结论都站不住，直接报样本不够，不硬算。
MIN_MEMORIES = 100
MIN_RETRIEVAL_SAMPLES = 100

# 四个指标的通过线（BOOS 定的验证标准，见任务）：
#   有用记忆进前十 ≥0.90；过期记忆还留在前十 ≤0.10；
#   核心记忆掉出前五 ==0 次；有用记忆衰减到下限以下 ≤0.05。
THRESHOLD_USEFUL_TOP10 = 0.90
THRESHOLD_EXPIRED_TOP10 = 0.10
THRESHOLD_CORE_DROPPED_TOP5 = 0
THRESHOLD_USEFUL_BELOW_FLOOR = 0.05

TOP_N = 10
TOP_CORE_N = 5


def _default_param_sets():
    """默认 27 组参数：稳定半衰期 [90,180,365] × 一次性半衰期
    [7,14,30] × 下限 [0.05,0.10,0.20]。"""
    return [
        {
            "stable_half_life": stable,
            "episodic_half_life": episodic,
            "floor": floor,
        }
        for stable in DEFAULT_STABLE_HALF_LIVES
        for episodic in DEFAULT_EPISODIC_HALF_LIVES
        for floor in DEFAULT_FLOORS
    ]


def _reference_now(memories):
    """验证的参照时间 now：取全部记忆里最晚的 last_used_at。

    不用墙上的现在时间——离线验证要可复现：同一批数据、同一天跑
    和下个月跑，结果得一样。最晚那条记忆就是“现在”，它之前的闲置
    天数全是相对值。没有一条能解析出时间的，就按当前 UTC 算。"""
    latest = None
    for mem in memories or []:
        parsed = _parse_created_at((mem or {}).get("last_used_at"))
        if parsed is not None and (latest is None or parsed > latest):
            latest = parsed
    if latest is None:
        return datetime.now(timezone.utc)
    return latest


def _weight_with_params(kind, last_used_at, now, is_core, params):
    """按 decay_weight 的公式、用给定参数算一条记忆的权重。

    核心记忆恒 1.0；没时间的老记忆按 1.0（算不出闲了多久，就当没闲过）；
    其余按 0.5 ** (闲置天数 / 半衰期)，再用该组下限托底、上限 1.0。
    kind 不是 stable 就按 episodic 算（和 decay_weight 的缺省一致）。"""
    if is_core:
        return 1.0
    idle = _decay_idle_days(last_used_at, now)
    if idle is None:
        return 1.0
    half_life = (
        params["stable_half_life"]
        if kind == "stable"
        else params["episodic_half_life"]
    )
    raw = 0.5 ** (idle / half_life)
    return max(params["floor"], min(1.0, raw))


def _raw_weight_with_params(kind, last_used_at, now, is_core, params):
    """下限托底之前的原始衰减值（0.5 ** (闲置天数 / 半衰期））。

    只用来算“有用记忆衰减到下限以下”的指标：decay_weight 托底之后
    权重永远 ≥ 下限，直接比永远是 0——真正要看的是原始值已经掉到
    下限下面、被下限托住的那些（已经沉到底了）。核心记忆和没时间的
    老记忆没有衰减，原始值按 1.0 算。"""
    if is_core:
        return 1.0
    idle = _decay_idle_days(last_used_at, now)
    if idle is None:
        return 1.0
    half_life = (
        params["stable_half_life"]
        if kind == "stable"
        else params["episodic_half_life"]
    )
    return 0.5 ** (idle / half_life)


def _evaluate_param_set(memories, now, params):
    """一组参数下的四个指标。返回 (metrics, passed)：

    - useful_top10_ratio：权重前十里有用记忆的占比（≥0.90 通过）。
      分母是前十的槽位数（10）——不是有用记忆总数：有用记忆有
      100 条时，前十最多也只能装下 10 条，按总数算比例最高才 0.1，
      通过线 0.90 就永远达不到了。前十为空时按 0.0 算。
    - expired_top10_ratio：前十里过期记忆（useful=False）的占比
      （≤0.10 通过）。前十为空时按 0.0 算。
    - core_dropped_top5：核心记忆里掉出权重前五的条数（==0 通过）。
      这是条数不是比例——前五只能装五条，核心记忆超过五条时
      注定有人掉出去，这个数就是要把这种情况摆上台面。
    - useful_below_floor_ratio：有用记忆（不含核心）里原始衰减值
      已经掉到下限下面、被下限托住的比例（≤0.05 通过）。
      一条有用的非核心记忆都没有时按 0.0 算。"""
    weighted = [
        (mem, _weight_with_params(mem.get("kind"), mem.get("last_used_at"),
                                  now, bool(mem.get("is_core")), params))
        for mem in memories
    ]
    # 权重降序排，权重一样时按 id 排——同一批数据跑两次，顺序得一样。
    weighted.sort(key=lambda item: (-item[1], str(item[0].get("id"))))
    top10 = weighted[:TOP_N]
    top5_ids = {str(mem.get("id")) for mem, _ in weighted[:TOP_CORE_N]}

    useful = [mem for mem in memories if mem.get("useful")]
    useful_in_top10 = sum(1 for mem, _ in top10 if mem.get("useful"))
    useful_top10_ratio = (useful_in_top10 / len(top10)) if top10 else 0.0

    expired_in_top10 = sum(1 for mem, _ in top10 if not mem.get("useful"))
    expired_top10_ratio = (expired_in_top10 / len(top10)) if top10 else 0.0

    core_dropped_top5 = sum(
        1 for mem in memories
        if mem.get("is_core") and str(mem.get("id")) not in top5_ids
    )

    useful_non_core = [mem for mem in useful if not mem.get("is_core")]
    below_floor = sum(
        1 for mem in useful_non_core
        if _raw_weight_with_params(mem.get("kind"), mem.get("last_used_at"),
                                   now, False, params) < params["floor"]
    )
    useful_below_floor_ratio = (
        (below_floor / len(useful_non_core)) if useful_non_core else 0.0
    )

    metrics = {
        "useful_top10_ratio": useful_top10_ratio,
        "expired_top10_ratio": expired_top10_ratio,
        "core_dropped_top5": core_dropped_top5,
        "useful_below_floor_ratio": useful_below_floor_ratio,
    }
    passed = sum([
        useful_top10_ratio >= THRESHOLD_USEFUL_TOP10,
        expired_top10_ratio <= THRESHOLD_EXPIRED_TOP10,
        core_dropped_top5 == THRESHOLD_CORE_DROPPED_TOP5,
        useful_below_floor_ratio <= THRESHOLD_USEFUL_BELOW_FLOOR,
    ])
    return metrics, passed


def validate_decay_params(memories, retrieval_samples, param_sets=None):
    """离线验证衰减参数：拿一批带“现在还应不应该排前”标注的记忆，
    把每组候选参数都跑一遍，看哪组排得最准。

    memories：[{"id","kind","last_used_at","is_core","useful"}]，
    kind 是 stable / episodic，last_used_at 是 ISO 时间串，useful 是
    bool（True = 现在还应排前，False = 已经过期该沉底）。
    retrieval_samples：观测到的检索次数（int）——样本量够不够的
    另一半证据，只看记忆条数不够。
    param_sets：不传就默认扫 27 组（稳定半衰期 [90,180,365] ×
    一次性半衰期 [7,14,30] × 下限 [0.05,0.10,0.20]）；传了就按传
    的扫，每组是 {"stable_half_life","episodic_half_life","floor"}。

    返回 {"sufficient","reason","results","best"}：
    - 样本不够（记忆 <100 条或检索次数 <100）时 sufficient=False，
      reason 是中文原因，results=[]，best=None——不硬算、不猜结论。
    - 样本够时 results 是每组 {"params","metrics","passed"}，按 passed
      降序、再按有用进前十的比例降序排；best 是排第一的那组（原样）。
    只验证、不改参数：best 再好看，也只是“建议候选”，调不调由 BOOS 定。
    """
    memories = list(memories or [])
    if len(memories) < MIN_MEMORIES:
        return {
            "sufficient": False,
            "reason": f"记忆样本不够：只有 {len(memories)} 条，需要至少 "
                      f"{MIN_MEMORIES} 条才有验证意义。",
            "results": [],
            "best": None,
        }
    if not isinstance(retrieval_samples, int) or isinstance(retrieval_samples, bool) \
            or retrieval_samples < MIN_RETRIEVAL_SAMPLES:
        return {
            "sufficient": False,
            "reason": f"检索样本不够：只有 {retrieval_samples} 次检索，"
                      f"需要至少 {MIN_RETRIEVAL_SAMPLES} 次才有验证意义。",
            "results": [],
            "best": None,
        }

    param_sets = list(param_sets) if param_sets is not None else _default_param_sets()
    now = _reference_now(memories)
    results = []
    for params in param_sets:
        params = {
            "stable_half_life": params["stable_half_life"],
            "episodic_half_life": params["episodic_half_life"],
            "floor": params["floor"],
        }
        metrics, passed = _evaluate_param_set(memories, now, params)
        results.append({"params": params, "metrics": metrics, "passed": passed})
    # passed 多的在前；passed 一样时，有用记忆进前十比例高的在前；
    # 再一样时按参数字典排——顺序全确定，跑两次结果一样。
    results.sort(key=lambda r: (
        -r["passed"],
        -r["metrics"]["useful_top10_ratio"],
        r["params"]["stable_half_life"],
        r["params"]["episodic_half_life"],
        r["params"]["floor"],
    ))
    return {
        "sufficient": True,
        "reason": None,
        "results": results,
        "best": results[0] if results else None,
    }


def biweekly_retest(recent_used_weights):
    """双周复测：看最近 14 天真正被拼进问题的记忆权重，判断衰减
    是快了还是慢了。只给建议，绝不自动改参数——调不调、往哪调，
    由 BOOS 看了建议之后亲口定；这个函数连参数的影子都不碰。

    recent_used_weights：最近 14 天真正被拼进问题的记忆的权重列表
    （每条是 decay_weight 算出来的 0.10–1.0 之间的数）。
    返回 {"median","recommendation","reason"}：
    - 空列表：median=None，recommendation="insufficient"——没数据，
      不瞎建议。
    - 中位数 <0.2：recommendation="slow_down"——最近真正在用的记忆
      权重都快沉到底了，衰减太快，半衰期该调大。
    - 中位数 >0.6：recommendation="speed_up"——最近在用的记忆权重
      还高高在上，衰减太慢，半衰期该调小。
    - 其余：recommendation="keep"——中位数在 0.2–0.6 之间，衰减速度
      看着合适，先不动。"""
    weights = list(recent_used_weights or [])
    if not weights:
        return {
            "median": None,
            "recommendation": "insufficient",
            "reason": "最近 14 天没有真正被拼进问题的记忆，样本为空，无法判断衰减快慢。",
        }
    median = float(statistics.median(weights))
    if median < 0.2:
        return {
            "median": median,
            "recommendation": "slow_down",
            "reason": f"最近真正在用的记忆权重中位数只有 {median:.2f}，"
                      "衰减太快、半衰期该调大（只建议，不自动改）。",
        }
    if median > 0.6:
        return {
            "median": median,
            "recommendation": "speed_up",
            "reason": f"最近真正在用的记忆权重中位数高达 {median:.2f}，"
                      "衰减太慢、半衰期该调小（只建议，不自动改）。",
        }
    return {
        "median": median,
        "recommendation": "keep",
        "reason": f"最近真正在用的记忆权重中位数为 {median:.2f}，"
                  "在合理区间内，衰减速度先不动。",
    }
