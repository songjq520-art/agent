# tests/test_calculator.py（本地计算器，2026-11-21 网页测第二轮 #11）
# 只测 memory/calculator.py；全部离线，不碰账本、不联网。

from memory.calculator import calculator_block, safe_eval, _parse_cn_number


def test_safe_eval_arithmetic():
    """四则、括号、乘方、取余都算对。"""
    assert safe_eval("1+2*3") == 7
    assert safe_eval("(1+2)*3") == 9
    assert safe_eval("2**10") == 1024
    assert safe_eval("10/4") == 2.5
    assert safe_eval("10%3") == 1
    assert safe_eval("-5+3") == -2
    assert safe_eval("128+5") == 133


def test_safe_eval_rejects_non_arithmetic():
    """名字、调用、属性、下标、字符串、除零、超大乘方一律不算。"""
    assert safe_eval("__import__('os')") is None
    assert safe_eval("open('/etc/passwd')") is None
    assert safe_eval("a+1") is None
    assert safe_eval("[1, 2]") is None
    assert safe_eval("{'a': 1}") is None
    assert safe_eval("1/0") is None
    assert safe_eval("2**999999999") is None
    assert safe_eval("'hello'") is None
    assert safe_eval("") is None


def test_parse_cn_number():
    """中文数字：八、八点五、十二、二十五。"""
    assert _parse_cn_number("八") == 8.0
    assert _parse_cn_number("八点五") == 8.5
    assert _parse_cn_number("十二") == 12.0
    assert _parse_cn_number("二十五") == 25.0
    assert _parse_cn_number("8.5") == 8.5
    assert _parse_cn_number("八点二五") == 8.25


def test_dca_question_returns_verified_range():
    """BOOS 那道题：每天 5 块、年化 8%、30 年 → 14.3 万到 22.4 万，
    不再是模型心算的 128 万（#11）。"""
    block = calculator_block("如果我每天定投5块，年化百分之八，30年后能有多少钱")
    assert "22.4 万" in block
    assert "14.3 万" in block
    assert "本金 5.4 万" in block
    assert "本金 3.78 万" in block
    assert "128" not in block


def test_dca_rate_variants():
    """年化写法变体：8%、百分之八点五都认。"""
    block = calculator_block("每天定投5块，年化8%，30年后能有多少钱")
    assert "22.4 万" in block and "14.3 万" in block
    block = calculator_block("每天定投5块，年化百分之八点五，30年后能有多少钱")
    assert "24.8 万" in block


def test_arithmetic_question():
    """普通算式：128+5 是多少 → 133。"""
    block = calculator_block("128+5 是多少")
    assert "128+5 = 133" in block


def test_date_is_not_subtraction():
    """日期不是减法：2026-11-21 不算（#11 的护栏）。"""
    assert calculator_block("2026-11-21 是哪天") == ""
    assert calculator_block("我今天早上吃了什么") == ""
    assert calculator_block("") == ""
