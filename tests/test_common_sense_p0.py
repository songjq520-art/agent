# tests/test_common_sense_p0.py（常识 P0：本地确定性工具 + 强制调用规则）
# 作用：验证 tools/calc_tool.py 的两个本地工具和它们的接线——
#   42 号的根因是模型在文本里心算交易日（50 天 − 2 节假日 = 49，
#   错；对的是 33 天）。治本：精确数字必须调工具算，工具给的数
#   就是答案。全部离线，不调真实 API、不花钱。
# 覆盖：
#   - calculate：基本算术、拒绝不安全表达式；
#   - calendar_count：42 号原题（2026-11-13 到年底美股交易日 = 33）、
#     闰年、节假日落在周末、起止端点、CN 暂不支持要明说；
#   - 接线：两个工具登记进注册表、循环里能调、系统消息里有第 0 条规矩。

import json

import pytest

from tasks.task_store import TaskStore
from tools.agent_loop import LOOP_SYSTEM_MESSAGE, AgentLoop
from tools.audit import AuditLog
from tools.calc_tool import (
    _run_calculate,
    _run_calendar_count,
    calculate_expression,
    calendar_count,
    register_calc_tools,
)
from tools.tool_gateway import ToolGateway
from tools.tool_registry import get_tool


# ---------- calculate ----------


def test_calculate_basic():
    """基本算术：42 号原题 5*33=165，还有括号、除法、round。"""
    assert calculate_expression("5 * 33") == 165
    assert calculate_expression("(100 - 20) / 4") == 20
    assert calculate_expression("25466.19 - 25065.88") == pytest.approx(400.31)
    assert calculate_expression("round(3.14159, 2)") == 3.14
    assert calculate_expression("min(3, 7)") == 3
    assert calculate_expression("abs(0 - 5)") == 5


def test_calculate_rejects_unsafe():
    """不安全表达式一律拒绝，不执行：导入、变量名、属性、除零、幂过大。"""
    for bad in (
        "__import__('os')",
        "import os",
        "a + 1",
        "1 / 0",
        "10 ** 400",
        "[1, 2, 3][0]",
        "open('/etc/passwd').read()",
        "",
        "eval('1+1')",
    ):
        result = _run_calculate({"expression": bad})
        assert result["ok"] is False, bad
        assert result["error"]


def test_calculate_ok_envelope():
    """执行包络：成功时 ok=True，data 里带原表达式和结果。"""
    result = _run_calculate({"expression": "5 * 33"})
    assert result == {
        "ok": True,
        "data": {"expression": "5 * 33", "result": 165},
        "error": None,
    }


# ---------- calendar_count ----------


def test_calendar_count_42_original():
    """42 号原题：2026-11-13 到 2026-12-31，美股交易日 = 33 天、
    工作日 = 35 天、自然日 = 49 天；区间里的休市日是感恩节和圣诞节。"""
    data = calendar_count("US", "2026-11-13", "2026-12-31", "trading_days")
    assert data["count"] == 33
    assert data["trading_days"] == 33
    assert data["workdays"] == 35
    assert data["natural_days"] == 49
    assert [h["date"] for h in data["holidays"]] == ["2026-11-26", "2026-12-25"]
    assert calendar_count("US", "2026-11-13", "2026-12-31", "workdays")["count"] == 35
    assert calendar_count("US", "2026-11-13", "2026-12-31", "natural_days")["count"] == 49


def test_calendar_count_leap_year():
    """闰年：2028-02-28 到 2028-03-01 是 3 天（28、29、3 月 1 日），
    都是工作日、都是交易日——2 月 29 日不能漏掉。"""
    data = calendar_count("US", "2028-02-28", "2028-03-01", "trading_days")
    assert data["natural_days"] == 3
    assert data["workdays"] == 3
    assert data["count"] == 3


def test_calendar_count_holiday_on_weekend():
    """节假日落在周末：2026 年 7 月 4 日是周六，独立日补休在周五 7 月 3 日——
    7 月 3 日那天不算交易日，也不能重复减（周六日本来就不算）。"""
    data = calendar_count("US", "2026-07-02", "2026-07-06", "trading_days")
    assert data["natural_days"] == 5
    assert data["workdays"] == 3  # 周四、周五、周一。
    assert data["count"] == 2  # 周四、周一；周五补休。
    assert data["holidays"] == [{"date": "2026-07-03", "name": "独立日（补休）"}]
    # 单独一天：补休的周五工作日是 1、天是 1，交易日是 0。
    single = calendar_count("US", "2026-07-03", "2026-07-03", "trading_days")
    assert single["workdays"] == 1
    assert single["count"] == 0


def test_calendar_count_endpoints():
    """起止端点都包含在内：单个交易日（周五）是 1 天；感恩节当天
    工作日是 1、交易日是 0；周末单天工作日和交易日都是 0。"""
    assert calendar_count("US", "2026-11-13", "2026-11-13", "trading_days")["count"] == 1
    thanksgiving = calendar_count("US", "2026-11-26", "2026-11-26", "trading_days")
    assert thanksgiving["workdays"] == 1
    assert thanksgiving["count"] == 0
    saturday = calendar_count("US", "2026-11-14", "2026-11-14", "trading_days")
    assert saturday["workdays"] == 0
    assert saturday["count"] == 0


def test_calendar_count_bad_args():
    """参数不对直接报错，不猜：市场只认 US/CN、日期要写对、起止别反了。"""
    for args in (
        {"market": "JP", "start": "2026-11-13", "end": "2026-12-31", "type": "trading_days"},
        {"market": "US", "start": "2026-13-01", "end": "2026-12-31", "type": "trading_days"},
        {"market": "US", "start": "2026-12-31", "end": "2026-11-13", "type": "trading_days"},
        {"market": "US", "start": "2026-11-13", "end": "2026-12-31", "type": "days"},
        {"market": "", "start": "2026-11-13", "end": "2026-12-31", "type": "trading_days"},
    ):
        result = _run_calendar_count(args)
        assert result["ok"] is False, args
        assert result["error"]


def test_calendar_count_cn_not_supported_without_library():
    """中国市场：chinese-calendar 库没装时直接说暂不支持，不许猜节假日
    和调休、更不许编一个数出来。库装上后这条测试要改成真算。"""
    try:
        import chinese_calendar  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("chinese-calendar 已装，CN 走真算，不走这条")
    result = _run_calendar_count(
        {"market": "CN", "start": "2026-11-13", "end": "2026-12-31", "type": "trading_days"}
    )
    assert result["ok"] is False
    assert "暂不支持" in result["error"]


# ---------- 接线 ----------


def test_calc_tools_registered():
    """两个工具登记进注册表：只读、风险 low、不用确认、重复执行安全。"""
    register_calc_tools()
    for name in ("calculate", "calendar_count"):
        tool = get_tool(name)
        assert tool is not None, name
        assert tool["risk_level"] == "low"
        assert tool["needs_confirmation"] is False
        assert tool["replay_safe"] is True
        assert tool["side_effect_free"] is True


def test_loop_system_message_forbids_mental_math():
    """系统消息第 0 条：精确数字不许心算，必须调工具；第 11 条的
    交易日算法改成调 calendar_count，不再是“先搜一下”。"""
    assert "不许心算" in LOOP_SYSTEM_MESSAGE
    assert "calendar_count" in LOOP_SYSTEM_MESSAGE
    assert "交易日 = 工作日" in LOOP_SYSTEM_MESSAGE
    assert "自然日 − 节假日" in LOOP_SYSTEM_MESSAGE
    from tools.agent_loop import build_loop_system_message

    assert "先搜一下当年美股休市日历" not in build_loop_system_message()


def _tool_call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _tool_response(text="", tool_calls=None, cost=0.001):
    return {
        "text": text,
        "tool_calls": tool_calls or [],
        "reasoning_content": None,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "cost_usd": cost,
        "model": "deepseek-flash",
    }


class ScriptedProvider:
    """按剧本回话的假模型（只够循环测试用）。"""

    def __init__(self, model="deepseek-flash", script=None):
        self.model = model
        self.script = list(script or [])

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        return {"text": "translated", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        if not self.script:
            raise AssertionError("剧本用完了，模型又要调工具——循环该停了")
        return dict(self.script.pop(0))

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None, model=None, max_tokens=None):
        return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}


def test_loop_42_uses_calc_tools(tmp_path):
    """42 号端到端：模型调 calendar_count 算出 33 天、再调 calculate
    算出 165 元，最后回答里是工具给的数——不是心算的。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "calendar_count", {
                    "market": "US", "start": "2026-11-13",
                    "end": "2026-12-31", "type": "trading_days",
                }),
                _tool_call("call_2", "calculate", {"expression": "5 * 33"}),
            ]),
            _tool_response(text="从今天到年底是 33 个交易日，每天 5 块，一共 165 元。"),
        ],
    )
    store_task = TaskStore(file_path=str(tmp_path / "tasks.json"))
    loop = AgentLoop(
        provider,
        gateway=ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log"))),
        task_store=store_task,
    )
    outcome = loop.run(
        "每天定投 5 块，到年底大概投多少",
        goal="每天定投 5 块，到年底大概投多少",
    )
    assert outcome["mode"] == "done"
    assert "165" in outcome["reply"]
    assert "33" in outcome["reply"]
    called = [rec.get("tool") for rec in outcome["task"].get("tool_calls") or []]
    assert "calendar_count" in called
    assert "calculate" in called
    # 没调搜索：本地工具不花搜索预算。
    assert "search" not in called
