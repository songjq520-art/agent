# tools/calc_tool.py（本地确定性计算工具：常识 P0，2026-11-13）
# 作用：模型要算精确数字时调这两个本地工具，不许心算。
# 背景：42 号“每天定投 5 块，到年底大概投多少”（2026-11-13 起算），
#   十二在文本里心算：50 个日历天减 2 个节假日 = 49 天、245 元；
#   对的是 33 个交易日、165 元。交易日 = 工作日 − 该市场节假日，
#   不是日历天数减节假日——心算必错，程序算才对。
# 原理：
#   - calculate：安全算术（ast 白名单解析，不用 eval 跑任意代码）；
#   - calendar_count：本地日历计数。美股交易日按纽交所休市规则现算
#     （固定假日 + 第 N 个星期几 + 复活节推耶稣受难日，周末顺延），
#     不依赖外部库、不联网、不花搜索预算、不猜。
#     中国市场：库没装时直接报错说暂不支持，别瞎编（见 _cn_*）。
# 关系：tools/tool_registry.py 管登记；tools/tool_gateway.py 调 func 执行；
#       tools/agent_loop.py 在 __init__ 里登记这两个工具，系统消息
#       里立规矩“精确数字必须调工具，禁止心算”。
# 端点约定：start 和 end 都包含在内（2026-11-13 到 2026-12-31，
#   自然日 = 49 天、工作日 = 35 天、美股交易日 = 33 天）。

import ast
import operator
from datetime import date as _date
from datetime import timedelta

from tools.tool_registry import get_tool, register_tool

# ---------- calculate：安全算术 ----------

# 白名单：四则 + 取整/取余/幂 + 括号 + 正负号 + round/min/max/abs。
# 不认变量名、属性、下标、lambda、导入——表达式里出现这些直接拒绝。
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_SAFE_FUNCS = {"round": round, "min": min, "max": max, "abs": abs}
# 幂指数别太大（10**400 这种会把内存撑爆）：指数绝对值超过 100 就拒绝。
_MAX_POW_EXPONENT = 100


def _eval_node(node):
    """递归算一个 ast 节点：白名单以外的节点类型直接报 ValueError。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只认数字常量")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不认这个运算符：{type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("幂指数太大了，不算")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不认这个正负号：{type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
            raise ValueError("只认 round/min/max/abs 这四个函数")
        if node.keywords:
            raise ValueError("函数不认关键字参数")
        return _SAFE_FUNCS[node.func.id](*[_eval_node(arg) for arg in node.args])
    raise ValueError(f"表达式里不认 {type(node).__name__} 这种写法")


def calculate_expression(expression):
    """算一条算术表达式（字符串）。返回数字；表达式不合法、
    除零、结果不是有限数时抛 ValueError（调用方转成 ok=False）。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式是空的")
    if len(expression) > 200:
        raise ValueError("表达式太长了")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式写错了：{exc}") from exc
    result = _eval_node(tree)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError("结果不是数字")
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError("结果不是有限数")
    return result


def _run_calculate(args):
    """calculate 的执行函数（签名 func(args: dict) -> dict，网关调它）。
    args：{"expression": "5 * 33"}。"""
    expression = (args or {}).get("expression") or ""
    try:
        result = calculate_expression(expression)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return {"ok": False, "data": None, "error": f"算不出来：{exc}"}
    return {
        "ok": True,
        "data": {"expression": expression.strip(), "result": result},
        "error": None,
    }


# ---------- calendar_count：本地日历计数 ----------

# 美股休市规则（纽交所 NYSE，规则现算，不写死某一年）：
#   固定假日：元旦 1/1、六月节 6/19、独立日 7/4、圣诞节 12/25；
#   第 N 个星期几：马丁·路德·金 1 月第 3 个周一、总统日 2 月第 3 个周一、
#     阵亡将士纪念日 5 月最后一个周一、劳动节 9 月第 1 个周一、
#     感恩节 11 月第 4 个周四；
#   耶稣受难日 = 复活节星期日 − 2 天（复活节用格里高利算法现算）。
#   周末顺延：假日落在周六 → 提前到周五休市；落在周日 → 顺延到周一。
#   提前收盘日（感恩节次日、圣诞前夜等）照样是交易日，不排除。
#   临时全天休市（比如全国哀悼日）极少发生，不在规则里——真遇上了，
#   那个数以交易所公告为准，工具不负责。
# 2026 年核对：1/1、1/19、2/16、4/3（复活节 4/5）、5/25、6/19、
#   7/3（7/4 是周六，顺延到周五）、9/7、11/26、12/25——和任务
#   给的 2026 年美股休市日表一致。

def _nth_weekday(year, month, weekday, n):
    """某年某月第 n 个星期几（weekday：周一=0 … 周日=6；
    n 为正数第 n 个，n=-1 是最后一个）。
    注意：取“最后一个”时必须从当月真正的最后一天往回找——
    从 28 号往回找是错的（比如 2028 年 5 月的最后一个周一是 29 号，
    从 28 号往回找会落到 22 号）。"""
    if n > 0:
        day = _date(year, month, 1)
        offset = (weekday - day.weekday()) % 7
        return day + timedelta(days=offset + 7 * (n - 1))
    if month == 12:
        day = _date(year + 1, 1, 1) - timedelta(days=1)
    else:
        day = _date(year, month + 1, 1) - timedelta(days=1)
    offset = (day.weekday() - weekday) % 7
    return day - timedelta(days=offset)


def _easter_sunday(year):
    """复活节星期日（格里高利算法，Anonymous Gregorian computus）。"""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return _date(year, month, day)


def _us_market_holidays(year):
    """某年美股休市日（已按周末顺延），返回 {date: 名字}。"""
    holidays = {
        _date(year, 1, 1): "元旦",
        _nth_weekday(year, 1, 0, 3): "马丁·路德·金纪念日",
        _nth_weekday(year, 2, 0, 3): "总统日",
        _easter_sunday(year) - timedelta(days=2): "耶稣受难日",
        _nth_weekday(year, 5, 0, -1): "阵亡将士纪念日",
        _date(year, 6, 19): "六月节",
        _date(year, 7, 4): "独立日",
        _nth_weekday(year, 9, 0, 1): "劳动节",
        _nth_weekday(year, 11, 3, 4): "感恩节",
        _date(year, 12, 25): "圣诞节",
    }
    observed = {}
    for day, name in holidays.items():
        if day.weekday() == 5:  # 周六 → 提前到周五。
            observed[day - timedelta(days=1)] = name + "（补休）"
        elif day.weekday() == 6:  # 周日 → 顺延到周一。
            observed[day + timedelta(days=1)] = name + "（补休）"
        else:
            observed[day] = name
    return observed


def _parse_day(value, field):
    """YYYY-MM-DD 转 date；写错了抛 ValueError（调用方转成 ok=False）。"""
    if not isinstance(value, str):
        raise ValueError(f"{field} 要写成 YYYY-MM-DD，比如 2026-12-31")
    try:
        return _date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(f"{field} 要写成 YYYY-MM-DD，比如 2026-12-31") from None


def _cn_calendar():
    """中国日历库（chinese-calendar）：没装时返回 None，调用方直接
    报错说暂不支持——别用猜的节假日表顶上。"""
    try:
        import chinese_calendar
    except ImportError:
        return None
    return chinese_calendar


def calendar_count(market, start, end, count_type):
    """数两个日期之间（起止都包含）的天数。
    market："US"（美股）或 "CN"（中国）；count_type：
    natural_days（自然日）、workdays（工作日，周一到周五；
    CN 时按 chinese-calendar 的调休算）、trading_days（交易日 =
    工作日 − 该市场休市日）。
    返回 {"count", "workdays", "natural_days", "holidays": [{"date", "name"}]}。
    参数不对、CN 库没装、年份超出规则范围时抛 ValueError。"""
    market = (market or "").strip().upper()
    if market not in ("US", "CN"):
        raise ValueError("market 只认 US（美股）或 CN（中国）")
    count_type = (count_type or "").strip()
    if count_type not in ("natural_days", "workdays", "trading_days"):
        raise ValueError(
            "type 只认 natural_days（自然日）/ workdays（工作日）"
            "/ trading_days（交易日）"
        )
    start_day = _parse_day(start, "start")
    end_day = _parse_day(end, "end")
    if start_day > end_day:
        raise ValueError("start 不能比 end 晚")
    if start_day.year < 2000 or end_day.year > 2100:
        raise ValueError("只算 2000–2100 年之间的日期")
    natural_days = (end_day - start_day).days + 1
    holidays = {}
    if market == "US":
        for year in range(start_day.year, end_day.year + 1):
            holidays.update(_us_market_holidays(year))
        workdays = 0
        trading_days = 0
        for offset in range(natural_days):
            day = start_day + timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            workdays += 1
            if day not in holidays:
                trading_days += 1
    else:
        lib = _cn_calendar()
        if lib is None:
            raise ValueError(
                "中国市场日历暂不支持：chinese-calendar 库没装，"
                "不要猜节假日和调休"
            )
        workdays = 0
        trading_days = 0
        for offset in range(natural_days):
            day = start_day + timedelta(days=offset)
            # 工作日按库里的调休算（含补班的周六、扣掉的调休假）；
            # 交易日只认周一到周五、且不是法定节假日（补班的周末
            # 交易所不开市，绝不能算成交易日）。
            if lib.is_workday(day):
                workdays += 1
            if day.weekday() < 5 and not lib.is_holiday(day):
                trading_days += 1
    count = {
        "natural_days": natural_days,
        "workdays": workdays,
        "trading_days": trading_days,
    }[count_type]
    in_range_holidays = [
        {"date": day.isoformat(), "name": name}
        for day, name in sorted(holidays.items())
        if start_day <= day <= end_day and day.weekday() < 5
    ]
    return {
        "market": market,
        "start": start_day.isoformat(),
        "end": end_day.isoformat(),
        "type": count_type,
        "count": count,
        "natural_days": natural_days,
        "workdays": workdays,
        "trading_days": trading_days,
        "holidays": in_range_holidays,
    }


def _run_calendar_count(args):
    """calendar_count 的执行函数（签名 func(args: dict) -> dict，网关调它）。
    args：{"market": "US", "start": "2026-11-13", "end": "2026-12-31",
    "type": "trading_days"}。market 不给默认值——不知道是哪个市场
    时让模型反问 BOOS，不猜。"""
    args = args or {}
    try:
        data = calendar_count(
            args.get("market"), args.get("start"), args.get("end"), args.get("type")
        )
    except (ValueError, TypeError) as exc:
        return {"ok": False, "data": None, "error": f"数不出来：{exc}"}
    return {"ok": True, "data": data, "error": None}


def register_calc_tools():
    """把 "calculate" 和 "calendar_count" 登记进工具注册表
    （已登记过时什么都不做，重复调不会报错）。
    两个都是纯本地计算：只读、风险 low、不用确认、重复执行安全、
    不改任何东西、不花搜索预算。"""
    if get_tool("calculate") is None:
        register_tool(
            name="calculate",
            description=(
                "本地算术：算一条算术表达式（加减乘除、取整、取余、幂、"
                "括号、round/min/max/abs），返回精确结果。"
                "凡是涉及精确数字、金额、涨跌幅、单位换算、概率计算，"
                "必须调这个工具算，禁止在心里算。表达式只认数字和"
                "运算符，不认变量、函数调用（除 round/min/max/abs）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要算的算术表达式，比如 \"5 * 33\"、\"(25466.19 - 25065.88) / 25065.88 * 100\"",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                },
                "required": ["expression"],
            },
            output=(
                "成功时 data 是 {expression: 原表达式, result: 算出来的数}；"
                "失败时 ok 为 false，error 里写原因（表达式写错了、除零等）。"
            ),
            risk_level="low",
            needs_confirmation=False,
            tier2=False,
            replay_safe=True,
            side_effect_free=True,
            timeout=10,
        )(_run_calculate)
    if get_tool("calendar_count") is None:
        register_tool(
            name="calendar_count",
            description=(
                "本地日历计数：数两个日期之间（起止都包含）有几天。"
                "type 三种：natural_days（自然日）、workdays（工作日，"
                "周一到周五）、trading_days（交易日 = 工作日 − 该市场"
                "休市日）。market 必须写 US（美股）或 CN（中国），"
                "不知道是哪个市场时先问 BOOS，不猜。"
                "凡是涉及日期差、工作日、交易日（比如定投到年底有几天、"
                "两个日期之间隔几天），必须调这个工具算，禁止在心里算；"
                "交易日绝不能用“自然日 − 节假日”算。"
                "返回 count（要的那种天数）和明细（自然日/工作日/交易日"
                "各多少、区间里的休市日是哪些）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "哪个市场：US（美股）或 CN（中国），不知道时先问 BOOS，不猜",
                        "enum": ["US", "CN"],
                    },
                    "start": {
                        "type": "string",
                        "description": "开始日期，YYYY-MM-DD，包含在内",
                    },
                    "end": {
                        "type": "string",
                        "description": "结束日期，YYYY-MM-DD，包含在内",
                    },
                    "type": {
                        "type": "string",
                        "description": "数哪种天：natural_days（自然日）/ workdays（工作日）/ trading_days（交易日）",
                        "enum": ["natural_days", "workdays", "trading_days"],
                    },
                },
                "required": ["market", "start", "end", "type"],
            },
            output=(
                "成功时 data 是 {market, start, end, type, count, "
                "natural_days, workdays, trading_days, "
                "holidays: [{date, name}]}；count 就是要的那种天数。"
                "失败时 ok 为 false，error 里写原因（市场不对、日期写错了、"
                "中国市场库没装等）。"
            ),
            risk_level="low",
            needs_confirmation=False,
            tier2=False,
            replay_safe=True,
            side_effect_free=True,
            timeout=10,
        )(_run_calendar_count)
