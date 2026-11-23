# tests/test_tone.py（补丁 3：两段式判语气——本地筛 + 判定接口）
# 只测 memory/tone.py 里的纯函数，全部离线，不碰真实 mem0、
# 不联网、不产生费用。

import json

import pytest

from memory.tone import (
    TONE_HYPOTHESIS,
    TONE_JOKE,
    TONE_SERIOUS,
    TONE_TEST,
    TONE_VALUES,
    can_override_confirmed,
    counts_for_promotion,
    local_tone_screen,
    parse_tone_judgment,
    tone_judgment_prompts,
)


# ---------- 常量 ----------


def test_tone_values_come_in_four():
    """四个语气值齐全，和 provenance.py / decay.py 认的值一致。"""
    assert TONE_VALUES == ("serious", "joke", "test", "hypothesis")
    assert TONE_SERIOUS == "serious"
    assert TONE_JOKE == "joke"
    assert TONE_TEST == "test"
    assert TONE_HYPOTHESIS == "hypothesis"


# ---------- local_tone_screen：明确说法 ----------


@pytest.mark.parametrize(
    "phrase",
    ["这是测试", "这句是测试", "测试一下"],
)
def test_explicit_test_phrases(phrase):
    """明确说法 test：三个短语各判 test，evidence 是原文里的短语。"""
    text = f"每天定投 5 美元沪深 300，{phrase}，别当真"
    tone, evidence = local_tone_screen(text)
    assert tone == "test"
    assert evidence == phrase
    assert evidence in text


@pytest.mark.parametrize(
    "phrase",
    ["开玩笑", "玩笑", "闹着玩"],
)
def test_explicit_joke_phrases(phrase):
    """明确说法 joke：三个短语各判 joke，evidence 是原文里的短语。"""
    text = f"我喜欢吃辣，{phrase}的"
    tone, evidence = local_tone_screen(text)
    assert tone == "joke"
    assert evidence == phrase
    assert evidence in text


def test_explicit_hypothesis_phrase():
    """明确说法 hypothesis：“假设”判 hypothesis。"""
    text = "假设我每天定投 5 块，十年后有多少"
    tone, evidence = local_tone_screen(text)
    assert tone == "hypothesis"
    assert evidence == "假设"
    assert evidence in text


# ---------- local_tone_screen：关键词 ----------


def test_keyword_test():
    """关键词 test：只含“测试”（没有明确说法）也判 test。"""
    text = "这是一个测试，看看你记不记得"
    tone, evidence = local_tone_screen(text)
    assert tone == "test"
    assert evidence == "测试"
    assert evidence in text


@pytest.mark.parametrize(
    "phrase",
    ["开玩笑", "玩笑"],
)
def test_keyword_joke_phrases(phrase):
    """关键词 joke：“开玩笑”/“玩笑”判 joke。"""
    text = f"别信，{phrase}话而已"
    tone, evidence = local_tone_screen(text)
    assert tone == "joke"
    assert evidence == phrase
    assert evidence in text


@pytest.mark.parametrize(
    "phrase",
    ["假设", "假如", "如果", "比如说", "打个比方"],
)
def test_keyword_hypothesis_phrases(phrase):
    """关键词 hypothesis：五个短语各判 hypothesis。"""
    text = f"{phrase}我每天定投 5 块，十年后有多少"
    tone, evidence = local_tone_screen(text)
    assert tone == "hypothesis"
    assert evidence == phrase
    assert evidence in text


# ---------- local_tone_screen：没命中 / 优先级 / 边界 ----------


def test_no_match_returns_none_none():
    """没命中的认真句子返回 (None, None)，不硬判成语气。"""
    tone, evidence = local_tone_screen("我每天定投 5 块买万家纳指")
    assert (tone, evidence) == (None, None)


@pytest.mark.parametrize("text", ["", None, 123])
def test_empty_or_non_string_returns_none_none(text):
    """空串 / 非字符串不猜，返回 (None, None)。"""
    assert local_tone_screen(text) == (None, None)


def test_explicit_beats_keyword():
    """明确说法优先于关键词：既有“这是测试”又有“如果”，判 test。"""
    tone, evidence = local_tone_screen("这是测试，如果我每天投 5 块会怎样")
    assert tone == "test"
    assert evidence == "这是测试"


def test_keyword_test_beats_keyword_hypothesis():
    """关键词内部也有优先级：只有关键词“测试”（无明确说法）又有
    “如果”时，判 test。"""
    tone, evidence = local_tone_screen("做个测试，如果每天投 5 块会怎样")
    assert tone == "test"
    assert evidence == "测试"


# ---------- tone_judgment_prompts ----------


def test_tone_judgment_prompts_shape():
    """提示词返回 (system, user)：system 要求只回 JSON 四类、拿不准判
    serious；user 里带上原文。"""
    system_prompt, user_prompt = tone_judgment_prompts("每天定投 5 块")
    assert isinstance(system_prompt, str) and isinstance(user_prompt, str)
    for value in TONE_VALUES:
        assert value in system_prompt
    assert '{"tone":' in system_prompt
    assert "拿不准" in system_prompt and "serious" in system_prompt
    assert "每天定投 5 块" in user_prompt


# ---------- parse_tone_judgment ----------


def test_parse_valid_dict():
    """合法 dict：原样返回 (tone, evidence)。"""
    assert parse_tone_judgment({"tone": "joke", "evidence": "开玩笑的"}) == (
        "joke",
        "开玩笑的",
    )


def test_parse_valid_json_string():
    """JSON 字符串：解析后返回 (tone, evidence)。"""
    payload = json.dumps({"tone": "test", "evidence": "这是测试"}, ensure_ascii=False)
    assert parse_tone_judgment(payload) == ("test", "这是测试")


def test_parse_json_code_block():
    """```json 代码块：去掉围栏后照样认。"""
    payload = '```json\n{"tone": "hypothesis", "evidence": "假设我每天投 5 块"}\n```'
    assert parse_tone_judgment(payload) == ("hypothesis", "假设我每天投 5 块")


def test_parse_plain_code_block():
    """不带 json 标记的代码块也认。"""
    payload = '```\n{"tone": "serious", "evidence": "我每天定投 5 块"}\n```'
    assert parse_tone_judgment(payload) == ("serious", "我每天定投 5 块")


def test_parse_bad_json():
    """坏 JSON：返回 (None, None)。"""
    assert parse_tone_judgment('{"tone": "joke", "evidence": ') == (None, None)
    assert parse_tone_judgment("不是 JSON") == (None, None)


def test_parse_illegal_tone():
    """tone 值非法：返回 (None, None)。"""
    assert parse_tone_judgment({"tone": "funny", "evidence": "开玩笑的"}) == (
        None,
        None,
    )
    assert parse_tone_judgment({"tone": "SERIOUS", "evidence": "认真的"}) == (
        None,
        None,
    )


def test_parse_empty_evidence():
    """evidence 为空：返回 (None, None)。"""
    assert parse_tone_judgment({"tone": "joke", "evidence": ""}) == (None, None)
    assert parse_tone_judgment({"tone": "joke", "evidence": "   "}) == (None, None)
    assert parse_tone_judgment({"tone": "joke"}) == (None, None)
    assert parse_tone_judgment({"tone": "joke", "evidence": 123}) == (None, None)


def test_parse_non_dict_payload():
    """解析出来不是 dict / 本来就不是 dict 或字符串：返回 (None, None)。"""
    assert parse_tone_judgment('["joke"]') == (None, None)
    assert parse_tone_judgment(123) == (None, None)
    assert parse_tone_judgment(None) == (None, None)


# ---------- counts_for_promotion / can_override_confirmed ----------


@pytest.mark.parametrize(
    "tone, expected",
    [
        ("serious", True),
        (None, True),
        ("joke", False),
        ("test", False),
        ("hypothesis", False),
    ],
)
def test_counts_for_promotion(tone, expected):
    """四种语气全覆盖：joke/test/hypothesis 不算晋升次数，
    serious/没打标算。"""
    assert counts_for_promotion(tone) is expected


@pytest.mark.parametrize(
    "tone, expected",
    [
        ("serious", True),
        (None, True),
        ("joke", False),
        ("test", False),
        ("hypothesis", False),
    ],
)
def test_can_override_confirmed(tone, expected):
    """四种语气全覆盖：joke/test/hypothesis 盖不掉 BOOS 确认过的记忆，
    serious/没打标可以。"""
    assert can_override_confirmed(tone) is expected
