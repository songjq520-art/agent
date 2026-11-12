# tests/test_batch4_prompts_ui.py（第 4 批：提示词规矩 24/31/32/47/49）
# 24 号：先查后说（系统消息第 3 条那个“不调的借口”堵死）；
# 31 号：别每句追问；32 号：有证据就敢下结论；47 号：不许编日期；
# 49 号：先给结论、再给必要的解释（2026-11-13 放宽，原来“一两句就够”
# 太干巴）。全部离线检查，不调真实 API、不花钱。

from datetime import datetime
from zoneinfo import ZoneInfo

from tools.agent_loop import LOOP_SYSTEM_MESSAGE, build_loop_system_message


def test_loop_system_message_has_batch4_rules():
    """第 4 批的五条提示词规矩都在系统消息里，一条不能少。"""
    # 24 号：先查后说，不许不搜就答、不许先反问“要查吗”。
    assert "先调搜索查了再回答" in LOOP_SYSTEM_MESSAGE
    assert "不许先反问" in LOOP_SYSTEM_MESSAGE
    # 31 号：别每句追问。
    assert "回答完就停，不要每句追问" in LOOP_SYSTEM_MESSAGE
    assert "一次只问一个" in LOOP_SYSTEM_MESSAGE
    # 32 号：有证据就敢下结论，别动不动含糊。
    assert "给最可能的结论" in LOOP_SYSTEM_MESSAGE
    assert "别怕说错" in LOOP_SYSTEM_MESSAGE
    # 47 号：不许编日期。
    assert "日期不许编" in LOOP_SYSTEM_MESSAGE
    assert "不许凭自己的“现在”推断日期" in LOOP_SYSTEM_MESSAGE
    # 49 号：先给结论、再给必要的解释（2026-11-13 放宽）。
    assert "先给结论，再给必要的解释" in LOOP_SYSTEM_MESSAGE
    assert "结论在前，把道理说清楚" in LOOP_SYSTEM_MESSAGE


def test_build_loop_system_message_adds_date_and_date_range_rule():
    """循环的系统消息 = 基础规矩 + 今天的日期 + 第 8 条（28 号 date_range）。"""
    text = build_loop_system_message()
    today = datetime.now(ZoneInfo("Asia/Shanghai"))
    assert f"{today.year} 年 {today.month} 月 {today.day} 日" in text
    assert "date_range" in text
    assert "换算成具体日期再查" in text
    # 基础规矩本身没被改写：开头还是 LOOP_SYSTEM_MESSAGE 原文。
    assert text.startswith(LOOP_SYSTEM_MESSAGE)


# ---------- 42 号：定投默认按交易日算（2026-11-12 BOOS 定） ----------

# 42 号 5 条新问法（实测换说法验默认口径用）：前 3 条没点明口径，
# 默认按交易日算；后 2 条明确说了日历天，按日历天算。
DCA_42_PHRASES = [
    # （问法，期望口径）
    ("从今天定投到年底，每天 5 块，一共投多少钱", "trading"),
    ("从今天起一个月，每天定投 5 块，一共多少钱", "trading"),
    ("从今天起一年，每天定投 5 块，一共多少钱", "trading"),
    ("从今天定投到年底，每天 5 块，按日历天算，一共投多少钱", "calendar"),
    ("从今天起一个月，每天定投 5 块，按自然日算，一共多少钱", "calendar"),
]


def test_42_system_message_defaults_to_trading_days():
    """42 号：第 11 条写明定投默认按交易日算，只有明确说日历天/
    自然日时才按日历天算；回答里要说明口径。"""
    text = build_loop_system_message()
    assert "默认按交易日算" in text
    assert "定投只在美股交易日扣款" in text
    assert "按日历天/自然日算时" in text
    assert "回答里说明是按交易日算的" in text
    # 42 号实测（2026-11-12）：“从今天起一年”那轮模型先给了 1825（日历天）
    # 再补交易日——第 11 条加一句：先给交易日的数，别先给日历天再补充；
    # 不确定就先搜，别先给大概数。
    assert "先给交易日的数" in text
    assert "别先给日历天的数再补充" in text
    assert "别先给个大概数" in text


def test_42_phrases_cover_default_and_calendar():
    """42 号 5 条新问法：3 条默认口径（交易日）+ 2 条明确日历天，
    条条不一样（实测换说法验泛化用）。"""
    assert len(DCA_42_PHRASES) == 5
    assert len({phrase for phrase, _ in DCA_42_PHRASES}) == 5
    assert [basis for _, basis in DCA_42_PHRASES].count("trading") == 3
    assert [basis for _, basis in DCA_42_PHRASES].count("calendar") == 2
