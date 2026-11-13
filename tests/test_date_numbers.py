# tests/test_date_numbers.py（50/54 号根治：数字-日期归属）
# 作用：验证 tools/date_numbers.py——从搜索结果文本里把数字和日期配对、
#       用户问的日期换算、回答里引用的数按日期校验。
# 全部离线：纯文本解析，不调真实接口、不花钱。
# 关键回归（2026-11-12 实测的两个真问题）：
#   50 号：纳指 11 月 9 日收 25,253.14，结果里混着 25,466.19（11 月 11 日的数）
#     ——25,466.19 必须认成 11 月 11 日的，不能拉低 11 月 9 日答案的把握。
#   54 号：问“纳指昨天（11 月 11 日）是涨还是跌”，模型拿了 11 月 10 日的
#     25,065.88 / -0.74% 当答案——校验必须拦住它，方向都说反了。

from datetime import date

from tools.date_numbers import (
    NUMBER_TOLERANCE,
    extract_dated_numbers,
    extract_numbers,
    find_date_mismatches,
    is_market_date_question,
    parse_date_mentions,
    parse_number_value,
    resolve_requested_date,
)


def _dates(text, year=2026):
    return [(item["date"].isoformat(), item["text"]) for item in parse_date_mentions(text, year)]


# ---------- 日期解析 ----------


def test_parse_english_long_date():
    assert _dates("closed on November 11, 2026") == [("2026-11-11", "November 11, 2026")]
    assert _dates("closed on Nov 11, 2026") == [("2026-11-11", "Nov 11, 2026")]
    assert _dates("on Nov. 11, 2026") == [("2026-11-11", "Nov. 11, 2026")]


def test_parse_english_date_no_year_uses_default_year():
    assert _dates("closed on November 11", 2026) == [("2026-11-11", "November 11")]
    assert _dates("on Nov 11", 2026) == [("2026-11-11", "Nov 11")]


def test_parse_english_day_first_date():
    assert _dates("Wednesday, November 11, 2026") == [("2026-11-11", "November 11, 2026")]
    assert _dates("11 November 2026") == [("2026-11-11", "11 November 2026")]


def test_parse_iso_and_numeric_dates():
    assert _dates("2026-11-11") == [("2026-11-11", "2026-11-11")]
    assert _dates("11/11/2026") == [("2026-11-11", "11/11/2026")]
    # WSJ 的“前收盘（11/10/26）”：两位数年份是 20xx 年。
    assert _dates("前收盘（11/10/26）") == [("2026-11-10", "11/10/26")]
    assert _dates("11/11") == [("2026-11-11", "11/11")]


def test_parse_chinese_dates():
    assert _dates("2026年11月11日") == [("2026-11-11", "2026年11月11日")]
    assert _dates("11月11日") == [("2026-11-11", "11月11日")]
    assert _dates("11 月 10 日") == [("2026-11-10", "11 月 10 日")]


def test_parse_dot_dates_boos_phrasings():
    # 50/54 的缺口：BOOS 原话就是点分日期——“标普的 11.10”“纳指 11.9 那天”。
    assert _dates("标普的 11.10") == [("2026-11-10", "11.10")]
    assert _dates("纳指 11.9 那天") == [("2026-11-09", "11.9")]
    assert _dates("11.10 那天收盘多少") == [("2026-11-10", "11.10")]


def test_parse_dot_dates_do_not_eat_prices_or_percents():
    # 价格和百分比不能被当成日期：25,065.88 里的 .88、1.60% 的 .60。
    assert _dates("收于 25,065.88 点") == []
    assert _dates("涨了 1.60%") == []
    assert _dates("跌 0.74%") == []
    # 版本号也不能当日期。
    assert _dates("v1.9 发布了") == []
    assert _dates("K2.6 模型") == []
    # 非法日期不认：月 13、日 32。
    assert _dates("13.10") == []
    assert _dates("11.32") == []


def test_parse_overlapping_keeps_longest():
    # "November 11, 2026" 里不能再拆出一个 "November 11"。
    assert _dates("November 11, 2026") == [("2026-11-11", "November 11, 2026")]
    assert _dates("2026年11月11日") == [("2026-11-11", "2026年11月11日")]


def test_parse_may_2026_is_not_a_date():
    # "May 2026" 里 "20" 不是日子（后面还跟着数字），不能认成 5 月 20 日。
    assert _dates("May 2026", 2026) == []


def test_parse_invalid_date_skipped():
    assert _dates("February 30, 2026") == []


def test_parse_multiple_dates_in_order():
    assert [d for d, _ in _dates("from November 10 to November 11, 2026")] == [
        "2026-11-10",
        "2026-11-11",
    ]


# ---------- 数字解析 ----------


def test_parse_number_value_kinds():
    assert parse_number_value("25,466.19") == ("price", 25466.19)
    assert parse_number_value("7,619.38") == ("price", 7619.38)
    assert parse_number_value("+1.60%") == ("percent", 1.6)
    assert parse_number_value("-0.74%") == ("percent", -0.74)
    assert parse_number_value("1.60%") == ("percent", 1.6)
    assert parse_number_value("+400.31") == ("points", 400.31)
    assert parse_number_value("-187.27") == ("points", -187.27)
    assert parse_number_value("25,065.88 点") == ("price", 25065.88)


def test_parse_number_value_rejects_years_and_days():
    # 裸年份和天数不算数。
    assert parse_number_value("2026") is None
    assert parse_number_value("11") is None
    assert parse_number_value("abc") is None
    assert parse_number_value("") is None


def test_extract_numbers_skips_numbers_inside_dates():
    # "2026-11-10" 里的 -11、-10 不是涨跌点数。
    assert extract_numbers("2026-11-10") == []
    assert extract_numbers("收盘 25,065.88，跌 0.74%（-187.27 点）") == [
        {"start": 3, "end": 12, "text": "25,065.88", "kind": "price", "value": 25065.88},
        {"start": 15, "end": 20, "text": "0.74%", "kind": "percent", "value": 0.74},
        {"start": 21, "end": 28, "text": "-187.27", "kind": "points", "value": -187.27},
    ]


# ---------- 数字-日期配对 ----------


def test_dated_numbers_basic():
    dns = extract_dated_numbers(
        "The Nasdaq Composite closed at 25,466.19 on November 11, 2026, up 1.60%.",
        default_year=2026,
    )
    assert dns == [
        {"number": "25,466.19", "value": 25466.19, "kind": "price",
         "date": "2026-11-11", "date_text": "November 11, 2026"},
        {"number": "1.60%", "value": 1.6, "kind": "percent",
         "date": "2026-11-11", "date_text": "November 11, 2026"},
    ]


def test_dated_numbers_previous_close_not_paired():
    # "from 25,065.87 the session before"：这个数是前一收盘，
    # 不能配成 11 月 11 日的数。
    dns = extract_dated_numbers(
        "The Nasdaq Composite closed at 25,466.19 on Wednesday, November 11, 2026, "
        "up 1.60% from 25,065.87 the session before.",
        default_year=2026,
    )
    by_number = {item["number"]: item["date"] for item in dns}
    assert by_number["25,466.19"] == "2026-11-11"
    assert by_number["1.60%"] == "2026-11-11"
    assert by_number["25,065.87"] is None


def test_dated_numbers_previous_close_from_marker():
    # "rose from 25,065.87"：from 后面那个数是起点（旧数），不配当天。
    dns = extract_dated_numbers(
        "The Nasdaq rose 1.60% to 25,466.19 on Nov 11 from 25,065.87.",
        default_year=2026,
    )
    by_number = {item["number"]: item["date"] for item in dns}
    assert by_number["25,466.19"] == "2026-11-11"
    assert by_number["25,065.87"] is None


def test_dated_numbers_nov10_close():
    dns = extract_dated_numbers(
        "The Nasdaq Composite declined 0.7% to close at 25,065.87 on November 10, 2026.",
        default_year=2026,
    )
    by_number = {item["number"]: item["date"] for item in dns}
    assert by_number["0.7%"] == "2026-11-10"
    assert by_number["25,065.87"] == "2026-11-10"


def test_dated_numbers_undated_when_no_date_in_sentence():
    # 句里没日期就不配对，不硬猜（published_at 也不拿来顶）。
    dns = extract_dated_numbers("纳斯达克综合指数收于 25,065.88 点，跌 0.74%。", default_year=2026)
    assert all(item["date"] is None for item in dns)
    assert {item["number"] for item in dns} == {"25,065.88", "0.74%"}


def test_dated_numbers_nearest_date_wins():
    # 一句里两个日期，数字归最近的那个。
    dns = extract_dated_numbers(
        "On November 10 it closed at 25,065.88; on November 11 it closed at 25,466.19.",
        default_year=2026,
    )
    by_number = {item["number"]: item["date"] for item in dns}
    assert by_number["25,065.88"] == "2026-11-10"
    assert by_number["25,466.19"] == "2026-11-11"


# ---------- 用户问的是哪天 ----------


def test_resolve_relative_yesterday():
    today = date(2026, 11, 12)
    assert resolve_requested_date("纳指昨天是涨还是跌", today) == date(2026, 11, 11)
    assert resolve_requested_date("昨天纳指收盘多少", today) == date(2026, 11, 11)
    assert resolve_requested_date("前天标普多少", today) == date(2026, 11, 10)
    assert resolve_requested_date("广州今天天气怎么样", today) == date(2026, 11, 12)
    assert resolve_requested_date("明天广州天气怎么样", today) == date(2026, 11, 13)


def test_resolve_absolute_date():
    today = date(2026, 11, 12)
    assert resolve_requested_date("纳斯达克 11 月 9 日收盘多少", today) == date(2026, 11, 9)
    assert resolve_requested_date("纳斯达克 11 月 10 日收盘多少", today) == date(2026, 11, 10)
    assert resolve_requested_date("Nasdaq close on November 11", today) == date(2026, 11, 11)
    assert resolve_requested_date("2026-11-11 纳指多少", today) == date(2026, 11, 11)
    # 点分日期（BOOS 原话）也要能换算，否则 50/54 的校验对它不生效。
    assert resolve_requested_date("标普的 11.10", today) == date(2026, 11, 10)
    assert resolve_requested_date("纳指 11.9 那天", today) == date(2026, 11, 9)


def test_resolve_no_date_or_ambiguous_returns_none():
    today = date(2026, 11, 12)
    assert resolve_requested_date("你好", today) is None
    assert resolve_requested_date("纳指收盘多少", today) is None
    # 范围问题（两个日期）拿不准，返回 None，不校验。
    assert resolve_requested_date("从 11 月 10 日到年底每天定投 5 块", today) is None


# ---------- 是不是行情问题 ----------


def test_is_market_date_question():
    assert is_market_date_question("纳指昨天是涨还是跌") is True
    assert is_market_date_question("纳斯达克 11 月 10 日收盘多少") is True
    assert is_market_date_question("标普 11 月 10 日收盘多少") is True
    assert is_market_date_question("昨天黄金收盘价多少") is True
    assert is_market_date_question("广州今天天气怎么样") is False
    assert is_market_date_question("你好") is False


# ---------- 校验回答 ----------


def _dated_50_54():
    """50/54 号的真实结果集：11 月 9 日、10 日、11 日的数混在一起。"""
    dated = []
    for text in [
        "The Nasdaq Composite closed at 25,253.14 on November 9, 2026, up 2.74%.",
        "The Nasdaq Composite declined 0.7% to close at 25,065.87 on November 10, 2026.",
        "The Nasdaq Composite closed at 25,466.19 on November 11, 2026, up 1.60% "
        "from 25,065.87 the session before.",
    ]:
        dated += [
            {"number": item["number"], "date": item["date"]}
            for item in extract_dated_numbers(text, default_year=2026)
        ]
    return dated


def test_mismatch_54_nov10_number_for_nov11_question():
    # 54 号：问 11 月 11 日，回答引用 25,065.88（11 月 10 日的数）——必须拦住。
    # 25,065.88 和结果里的 25,065.87 差 0.01（来源四舍五入），算同一个数。
    assert NUMBER_TOLERANCE >= 0.01
    mismatches, _ = find_date_mismatches(
        "昨天（11月11日）纳指是跌的：收于 25,065.88 点，跌 0.74%。",
        _dated_50_54(),
        date(2026, 11, 11),
    )
    assert [item["number"] for item in mismatches] == ["25,065.88"]
    assert mismatches[0]["found_dates"] == ["2026-11-10"]


def test_mismatch_54_correct_nov11_answer_passes():
    # 11 月 11 日的正确答案：25,466.19、+1.60%——必须放行。
    mismatches, _ = find_date_mismatches(
        "昨天（11月11日）纳指是涨的：收于 25,466.19，涨 1.60%。",
        _dated_50_54(),
        date(2026, 11, 11),
    )
    assert mismatches == []


def test_mismatch_50_conflicting_number_from_other_date_ignored():
    # 50 号：问 11 月 9 日，回答 25,253.14 是对的；结果里虽然混着
    # 25,466.19（11 月 11 日的数），但回答没引用它，就不算冲突。
    mismatches, _ = find_date_mismatches(
        "纳指 11 月 9 日收于 25,253.14，涨 2.74%。",
        _dated_50_54(),
        date(2026, 11, 9),
    )
    assert mismatches == []


def test_mismatch_50_quoting_other_date_number_is_caught():
    # 50 号的反面：如果回答引用了 25,466.19 却说是 11 月 9 日的数，拦住。
    mismatches, _ = find_date_mismatches(
        "纳指 11 月 9 日收于 25,466.19。",
        _dated_50_54(),
        date(2026, 11, 9),
    )
    assert [item["number"] for item in mismatches] == ["25,466.19"]
    assert mismatches[0]["found_dates"] == ["2026-11-11"]


def test_mismatch_unknown_number_is_unverified_not_blocked():
    # 结果里根本没出现过的数：没法按日期验证，不拦（只记下来）。
    mismatches, unverified = find_date_mismatches(
        "纳指 11 月 9 日收于 99,999.99。",
        _dated_50_54(),
        date(2026, 11, 9),
    )
    assert mismatches == []
    assert unverified == ["99,999.99"]


def test_mismatch_same_number_twice_counted_once():
    mismatches, _ = find_date_mismatches(
        "25,065.88，25,065.88。",
        _dated_50_54(),
        date(2026, 11, 11),
    )
    assert len(mismatches) == 1


def test_mismatch_no_numbers_in_reply_passes():
    mismatches, unverified = find_date_mismatches(
        "昨天纳指涨了。",
        _dated_50_54(),
        date(2026, 11, 11),
    )
    assert mismatches == []
    assert unverified == []


def test_mismatch_rounding_tolerance():
    # 25,065.88 和 25,065.87 差 0.01，算同一个数；差 0.05 以上算不同的数。
    dated = [{"number": "25,065.87", "date": "2026-11-10"}]
    mismatches, _ = find_date_mismatches("收于 25,065.88。", dated, date(2026, 11, 11))
    assert len(mismatches) == 1
    mismatches2, _ = find_date_mismatches("收于 25,066.19。", dated, date(2026, 11, 11))
    assert mismatches2 == []
    assert find_date_mismatches("收于 25,066.19。", dated, date(2026, 11, 11))[1] == ["25,066.19"]


# ---------- 51 号重试：行情问题程序拼英文收盘查询 ----------

# 51 号 5 条新说法（英文收盘查询拼法）：实体 + 日期 → 确定的英文查询；
# 认不出实体、没给日期时返回 None（不猜）。
ENGLISH_CLOSE_QUERY_51_CASES = [
    ("纳斯达克指数 收盘", date(2026, 11, 9), "Nasdaq Composite close November 9 2026"),
    ("标普500指数 11月10日 收盘", date(2026, 11, 10), "S&P 500 close November 10 2026"),
    ("道琼斯指数 收盘 点数", date(2026, 11, 10), "Dow Jones Industrial Average close November 10 2026"),
    ("黄金 11月11日 收盘价", date(2026, 11, 11), "Gold close November 11 2026"),
    ("广州天气 11月10日", date(2026, 11, 10), None),
]


def test_51_english_market_close_query_cases():
    """51 号：行情问题的英文收盘查询是程序拼出来的（实体英文名 + close +
    日期），不依赖模型翻译；非行情问题返回 None。"""
    from tools.date_numbers import english_market_close_query

    for question, day, wanted in ENGLISH_CLOSE_QUERY_51_CASES:
        assert english_market_close_query(question, day) == wanted, question
    assert english_market_close_query("纳斯达克指数 收盘", None) is None
