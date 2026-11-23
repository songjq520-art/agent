# memory/calculator.py（本地计算器，2026-11-21 网页测第二轮 #11）
#
# 病根：十二没有计算工具，难问题（每天定投 5 块、年化 8%、30 年）只能靠
# 模型心算——模型算成 128 万，还编了个“工具调用次数到上限”的借口。
# 修法：程序先算好、把结果拼进提示，模型照着用；算不出来的就说算不出来。
# 安全：只用 ast 白名单求值（数字、括号、四则、乘方、取余），不用 eval/exec；
# 出现名字、调用、属性、下标等一律不算（返回 None），没有注入口子。
# 生产代码里现在没有任何 eval/exec，也没有任何计算工具——这个模块是第一个。

import ast
import re

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}

# 乘方指数上限：防 "2**999999999" 这种写法把内存撑爆（提示词来自 BOOS，
# 不是攻击者，但上限本身不要钱）。
_POW_EXP_LIMIT = 1000

# 每年交易日数（A 股约 250 天，这里按 252 个交易日算，和核实结论一致）。
TRADING_DAYS_PER_YEAR = 252


def safe_eval(expr):
    """白名单表达式求值：只允许数字、括号、四则、乘方、取余。

    出现名字、调用、属性、下标、字符串等一律返回 None（不抛异常）。
    """
    try:
        tree = ast.parse(str(expr), mode="eval")
    except Exception:
        return None

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("not a number")
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if abs(right) > _POW_EXP_LIMIT:
                    raise ValueError("exponent too big")
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError("bad op")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
            value = _eval(node.operand)
            return +value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("not allowed")

    try:
        return _eval(tree)
    except Exception:
        return None


def _parse_cn_int(text):
    """中文整数转 int：八、十、十二、二十、二十五、一百二十三。"""
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        return int(text)
    total = 0
    if "百" in text:
        before, after = text.split("百", 1)
        total += (_parse_cn_int(before) if before else 1) * 100
        text = after
        if not text:
            return total
    if "十" in text:
        before, after = text.split("十", 1)
        total += (_parse_cn_int(before) if before else 1) * 10
        if after:
            rest = _parse_cn_int(after)
            if rest is None:
                return None
            total += rest
        return total
    if len(text) == 1 and text in _CN_DIGITS:
        return total + _CN_DIGITS[text]
    return None


def _parse_cn_number(text):
    """中文/阿拉伯数字转 float：8、8.5、八、八点五、十二点五。"""
    text = (text or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if "点" in text:
        whole, frac = text.split("点", 1)
        whole_v = _parse_cn_int(whole)
        if whole_v is None:
            return None
        digits = []
        for ch in frac:
            if ch in _CN_DIGITS:
                digits.append(str(_CN_DIGITS[ch]))
            elif ch.isdigit():
                digits.append(ch)
            else:
                return None
        if not digits:
            return None
        return whole_v + int("".join(digits)) / (10 ** len(digits))
    v = _parse_cn_int(text)
    return float(v) if v is not None else None


def _fv_annuity(pmt, annual_rate, years, periods_per_year):
    """期末年金终值：每期投 pmt，年化 annual_rate，投 years 年。"""
    r = annual_rate / periods_per_year
    n = int(round(years * periods_per_year))
    if r == 0:
        return pmt * n
    return pmt * (((1 + r) ** n - 1) / r)


def _fmt_wan(value, decimals=1):
    """元转“万”：223554 → "22.4 万"，37800 → "3.78 万"。"""
    s = f"{value / 10000:.{decimals}f}".rstrip("0").rstrip(".")
    return s + " 万"


def _fmt_number(value):
    """算式结果的写法：整数不带小数点，小数最多 4 位。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _dca_block(text):
    """定投复利：每天 X 块/美元、年化 Y%、N 年 → 两种口径的终值。

    按每月投 30X 块算，和按每年 252 个交易日、每年投 252X 块算——
    BOOS 那道题的核实结论是 14.3 万（交易日）到 22.4 万（按月），
    两种都给，模型不用自己选口径、不用心算。
    单位跟问题走（第三轮 #11，Rule 13）：问题说块就按块算，
    说美元就按美元算，不换算；两种口径的算法也写清（每月投一次是
    月末、每年投一次是年末），模型别自己编“每月月初定投”。
    精确数也给出来（万后面的括号里），模型直接抄，不用自己换算。
    """
    if "定投" not in text and "基金" not in text:
        return None
    m_amt = re.search(
        r"每天\s*(?:定投)?\s*(\d+(?:\.\d+)?)\s*(块|元|美元|美金|\$)", text
    )
    if m_amt:
        unit = "美元" if m_amt.group(2) in ("美元", "美金", "$") else "块"
        daily = float(m_amt.group(1))
    else:
        # “每天定投 $5”这种单位在数字前面的写法。
        m_amt = re.search(r"每天\s*(?:定投)?\s*\$\s*(\d+(?:\.\d+)?)", text)
        if not m_amt:
            return None
        unit = "美元"
        daily = float(m_amt.group(1))
    m_rate = re.search(r"年化(?:百分之)?\s*([零一二三四五六七八九十百点\d\.]+)", text)
    if not m_rate:
        return None
    rate = _parse_cn_number(m_rate.group(1))
    if rate is None or not (0 < rate < 100):
        return None
    years_matches = re.findall(r"(\d+(?:\.\d+)?)\s*年", text)
    if not years_matches:
        return None
    years = float(years_matches[-1])
    if not (0 < years <= 60):
        return None
    annual_rate = rate / 100.0
    monthly_pmt = daily * 30
    fv_monthly = _fv_annuity(monthly_pmt, annual_rate, years, 12)
    principal_monthly = monthly_pmt * int(round(years * 12))
    yearly_pmt = daily * TRADING_DAYS_PER_YEAR
    fv_yearly = _fv_annuity(yearly_pmt, annual_rate, years, 1)
    principal_yearly = yearly_pmt * int(round(years))
    fv_daily = _fv_annuity(daily, annual_rate, years, 365)
    principal_daily = daily * int(round(years * 365))
    return (
        "【本地计算结果（已算好，直接用这个数回答，别自己再算）】\n"
        f"每天定投 {daily:g} {unit}、年化 {rate:g}%、{years:g} 年：\n"
        f"- 按每天投 {daily:g} {unit}算（每天投一次，按日复利，日利率 年化/365）："
        f"{years:g} 年后约 {_fmt_wan(fv_daily)}"
        f"（{fv_daily:,.0f} {unit}，本金 {_fmt_wan(principal_daily, 2)}）\n"
        f"- 按每月投 {monthly_pmt:g} {unit}算（每月投一次，月末）："
        f"{years:g} 年后约 {_fmt_wan(fv_monthly)}"
        f"（{fv_monthly:,.0f} {unit}，本金 {_fmt_wan(principal_monthly, 2)}）\n"
        f"- 按每年 {TRADING_DAYS_PER_YEAR} 个交易日、每年投 {yearly_pmt:g} {unit}算"
        f"（每年投一次，年末）：{years:g} 年后约 {_fmt_wan(fv_yearly)}"
        f"（{fv_yearly:,.0f} {unit}，本金 {_fmt_wan(principal_yearly, 2)}）\n"
        "问题说每天定投，优先用上面每天投（按日复利）的数，直接抄，别自己再算；"
        "别说成每月月初定投，也别自己编别的口径；单位是块就按块算，"
        "是美元就按美元算，不要自己换算。年化是假设，实际收益有波动。"
    )


_ARITH_RE = re.compile(r"[\d\.\+\-\*/\(\)×÷（）\s]{3,}")
_ARITH_CUES = ("算", "等于", "多少", "总共", "一共", "合计")


def _arithmetic_block(text):
    """普通四则算式：只在问题带了“算/等于/多少……”这类算数口吻时才算。

    日期（2026-11-21）不算——那是日子不是减法；没有算数口吻时也不算，
    免得把“5-10 块”这种区间当成减法。
    """
    if not any(cue in text for cue in _ARITH_CUES):
        return ""
    normalized = (text.replace("×", "*").replace("÷", "/")
                  .replace("（", "(").replace("）", ")"))
    found = []
    for m in _ARITH_RE.finditer(normalized):
        expr = m.group(0).strip()
        if not any(op in expr for op in ("+", "-", "*", "/", "%")):
            continue
        if not re.search(r"\d", expr):
            continue
        if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", expr):
            continue  # 日期不是减法
        value = safe_eval(expr)
        if value is None:
            continue
        found.append((expr, value))
    if not found:
        return ""
    lines = ["【本地计算结果（已算好，直接用这个数回答，别自己再算）】"]
    for expr, value in found[:3]:
        lines.append(f"{expr} = {_fmt_number(value)}")
    return "\n".join(lines)


def calculator_block(prompt):
    """识别问题里的算式，返回拼进提示的计算结果块；识别不到返回 ""。"""
    text = str(prompt or "")
    if not text.strip():
        return ""
    return _dca_block(text) or _arithmetic_block(text)
