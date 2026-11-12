# tests/test_batch4_prompts_ui.py（第 4 批：提示词规矩 24/31/32/47/49）
# 24 号：先查后说（系统消息第 3 条那个“不调的借口”堵死）；
# 31 号：别每句追问；32 号：有证据就敢下结论；47 号：不许编日期；
# 49 号：回答要短。全部离线检查，不调真实 API、不花钱。

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
    # 49 号：回答要短。
    assert "回答要短，先给结论" in LOOP_SYSTEM_MESSAGE
    assert "一两句就够" in LOOP_SYSTEM_MESSAGE


def test_build_loop_system_message_adds_date_and_date_range_rule():
    """循环的系统消息 = 基础规矩 + 今天的日期 + 第 8 条（28 号 date_range）。"""
    text = build_loop_system_message()
    today = datetime.now(ZoneInfo("Asia/Shanghai"))
    assert f"{today.year} 年 {today.month} 月 {today.day} 日" in text
    assert "date_range" in text
    assert "换算成具体日期再查" in text
    # 基础规矩本身没被改写：开头还是 LOOP_SYSTEM_MESSAGE 原文。
    assert text.startswith(LOOP_SYSTEM_MESSAGE)
