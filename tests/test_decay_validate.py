# tests/test_decay_validate.py（衰减参数离线验证工具的测试）
# 全是纯函数测试：数据全是程序生成的合成数据，不读库、不写库、
# 不联网、不调任何真实 API。
# 合成数据的设计思路（写死在下面的 _GOOD_VS_BAD_DATA 里）：
#   参照时间是 2026-11-19（取最晚那条记忆的 last_used_at 当 now）；
#   有用记忆：5 条稳定事实闲了 300 天、45 条一次性经历闲了 40 天；
#   过期记忆：60 条一次性经历闲了 200 天、40 条稳定事实闲了 900 天；
#   核心记忆：3 条，权重恒 1.0。
#   好参数（180 天、14 天、0.10）：有用记忆权重 0.315 / 0.138，
#   都压过期记忆的下限 0.10 一头，前十全是有用的、四个指标全过；
#   坏参数（90 天、7 天、0.20）：有用记忆全沉到下限 0.20，
#   和过期记忆平起平坐，前十被过期记忆挤占、指标只过 1 个。
#   所以 best 必须选好参数——不是碰巧，是算出来的。

from datetime import datetime, timedelta, timezone

from memory.decay_validate import biweekly_retest, validate_decay_params

# 合成数据的参照时间：最晚那条记忆（核心记忆，闲置 0 天）就是它，
# validate_decay_params 拿它当 now，所以闲置天数全是确定值。
_NOW = datetime(2026, 11, 19, tzinfo=timezone.utc)


def _mem(mem_id, kind, idle_days, is_core, useful):
    """造一条合成记忆：last_used_at = 参照时间往前推 idle_days 天。"""
    last_used = _NOW - timedelta(days=idle_days)
    return {
        "id": mem_id,
        "kind": kind,
        "last_used_at": last_used.isoformat().replace("+00:00", "Z"),
        "is_core": is_core,
        "useful": useful,
    }


def _good_vs_bad_data():
    """“好参数明显赢”的合成数据（见文件头注释）：153 条记忆，
    53 条有用（含 3 条核心）、100 条过期。"""
    memories = []
    memories += [_mem(f"c{i:02d}", "stable", 0, True, True) for i in range(1, 4)]
    memories += [_mem(f"u-s{i:02d}", "stable", 300, False, True) for i in range(1, 6)]
    memories += [_mem(f"u-e{i:02d}", "episodic", 40, False, True) for i in range(1, 46)]
    memories += [_mem(f"e-e{i:02d}", "episodic", 200, False, False) for i in range(1, 61)]
    memories += [_mem(f"e-s{i:02d}", "stable", 900, False, False) for i in range(1, 41)]
    return memories


def _generic_data(n=120):
    """凑数的通用数据：n 条记忆，一半有用一半过期，闲置天数拉开，
    只用来测“样本够时能跑通”，不指望哪组参数赢。"""
    memories = []
    for i in range(n):
        if i % 2 == 0:
            memories.append(_mem(f"g-u{i:03d}", "stable", 10 + i, False, True))
        else:
            memories.append(_mem(f"g-e{i:03d}", "episodic", 200 + i, False, False))
    return memories


# ---------- 样本不够 ----------


def test_insufficient_too_few_memories():
    """记忆不到 100 条：sufficient=False，results 为空，best 为 None。"""
    result = validate_decay_params(_generic_data(50), 200)
    assert result["sufficient"] is False
    assert result["results"] == []
    assert result["best"] is None
    assert result["reason"]  # 中文原因，不能是空串


def test_insufficient_too_few_retrievals():
    """记忆够了、检索次数不到 100：一样报样本不够。"""
    result = validate_decay_params(_generic_data(150), 99)
    assert result["sufficient"] is False
    assert result["results"] == []
    assert result["best"] is None
    assert result["reason"]


def test_insufficient_both_too_few():
    """两边都不够：也是 sufficient=False（先报记忆不够）。"""
    result = validate_decay_params(_generic_data(10), 10)
    assert result["sufficient"] is False
    assert result["best"] is None


def test_sufficient_exactly_at_threshold():
    """正好卡在门槛上（100 条记忆、100 次检索）：算够，能跑。"""
    result = validate_decay_params(_generic_data(100), 100)
    assert result["sufficient"] is True
    assert result["reason"] is None
    assert len(result["results"]) == 27
    assert result["best"] is not None


# ---------- 样本够时能跑通 ----------


def test_enough_samples_runs_27_param_sets():
    """样本够时默认扫 27 组、best 不为空、三个比例指标都在 0~1 之间，
    核心掉出前五是条数（非负整数），结果按 passed 降序排。"""
    result = validate_decay_params(_generic_data(120), 150)
    assert result["sufficient"] is True
    assert len(result["results"]) == 27
    assert result["best"] is not None
    assert result["best"] is result["results"][0]
    passed_list = [r["passed"] for r in result["results"]]
    assert passed_list == sorted(passed_list, reverse=True)
    for entry in result["results"]:
        assert set(entry["params"]) == {
            "stable_half_life", "episodic_half_life", "floor",
        }
        metrics = entry["metrics"]
        assert 0.0 <= metrics["useful_top10_ratio"] <= 1.0
        assert 0.0 <= metrics["expired_top10_ratio"] <= 1.0
        assert 0.0 <= metrics["useful_below_floor_ratio"] <= 1.0
        assert isinstance(metrics["core_dropped_top5"], int)
        assert metrics["core_dropped_top5"] >= 0
        assert 0 <= entry["passed"] <= 4


def test_default_grid_covers_expected_values():
    """默认网格确实是 3×3×3：稳定半衰期 {90,180,365}、
    一次性半衰期 {7,14,30}、下限 {0.05,0.10,0.20}。"""
    result = validate_decay_params(_generic_data(120), 150)
    stable = {r["params"]["stable_half_life"] for r in result["results"]}
    episodic = {r["params"]["episodic_half_life"] for r in result["results"]}
    floors = {r["params"]["floor"] for r in result["results"]}
    assert stable == {90, 180, 365}
    assert episodic == {7, 14, 30}
    assert floors == {0.05, 0.10, 0.20}


# ---------- 好参数明显赢 ----------


def test_best_picks_good_params():
    """好参数（180 天、14 天、0.10）四个指标全过，best 必须选它；
    两个坏参数至多过 2 个，排不到前面。"""
    good = {"stable_half_life": 180, "episodic_half_life": 14, "floor": 0.10}
    bad_fast = {"stable_half_life": 90, "episodic_half_life": 7, "floor": 0.20}
    bad_slow_floor = {"stable_half_life": 365, "episodic_half_life": 7, "floor": 0.20}
    result = validate_decay_params(
        _good_vs_bad_data(), 200, param_sets=[good, bad_fast, bad_slow_floor]
    )
    assert result["sufficient"] is True
    assert len(result["results"]) == 3
    assert result["best"]["params"] == good
    assert result["best"]["passed"] == 4
    assert result["best"]["metrics"]["useful_top10_ratio"] == 1.0
    assert result["best"]["metrics"]["expired_top10_ratio"] == 0.0
    assert result["best"]["metrics"]["core_dropped_top5"] == 0
    assert result["best"]["metrics"]["useful_below_floor_ratio"] == 0.0
    for entry in result["results"][1:]:
        assert entry["passed"] < 4


def test_best_picks_good_params_in_default_grid():
    """同一批数据、不传 param_sets 走默认 27 组：best 仍是
    通过数最高的那组（好参数在其中），且排第一。"""
    result = validate_decay_params(_good_vs_bad_data(), 200)
    assert result["sufficient"] is True
    assert result["best"]["passed"] == 4
    assert result["best"]["metrics"]["useful_top10_ratio"] == 1.0


# ---------- biweekly_retest 四个分支 ----------


def test_biweekly_retest_empty_is_insufficient():
    """空列表：median 是 None，recommendation 是 insufficient。"""
    result = biweekly_retest([])
    assert result["median"] is None
    assert result["recommendation"] == "insufficient"
    assert result["reason"]


def test_biweekly_retest_low_median_slows_down():
    """中位数 0.1（<0.2）：slow_down——衰减太快、半衰期该调大。"""
    result = biweekly_retest([0.05, 0.10, 0.15])
    assert result["median"] == 0.10
    assert result["recommendation"] == "slow_down"


def test_biweekly_retest_high_median_speeds_up():
    """中位数 0.8（>0.6）：speed_up——衰减太慢、半衰期该调小。"""
    result = biweekly_retest([0.70, 0.80, 0.90])
    assert result["median"] == 0.80
    assert result["recommendation"] == "speed_up"


def test_biweekly_retest_middle_median_keeps():
    """中位数 0.4（在 0.2–0.6 之间）：keep——先不动。"""
    result = biweekly_retest([0.30, 0.40, 0.50])
    assert result["median"] == 0.40
    assert result["recommendation"] == "keep"


def test_biweekly_retest_boundaries_keep():
    """边界值 0.2 和 0.6 本身都算 keep（<0.2 才 slow、>0.6 才 speed）。"""
    assert biweekly_retest([0.20])["recommendation"] == "keep"
    assert biweekly_retest([0.60])["recommendation"] == "keep"


def test_biweekly_retest_even_count_median():
    """偶数个权重时中位数是中间两数的平均值。"""
    result = biweekly_retest([0.10, 0.20, 0.30, 0.40])
    assert result["median"] == 0.25
    assert result["recommendation"] == "keep"
