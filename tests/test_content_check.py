# tests/test_content_check.py（补丁 4 第一层：程序对原话，离线纯函数测试）
# 作用：测 memory/content_check.py 的三个纯函数——extract_checkables、
#       grounding_check、rank_most_likely_wrong。
# 注意：全是纯函数测试，不碰 mem0、不联网、不调真实 API。

from memory.content_check import extract_checkables, grounding_check, rank_most_likely_wrong


# ---------- extract_checkables：数字 ----------


def test_extract_numbers_plain_decimal_and_units():
    result = extract_checkables("每天定投5块，目标1.35万元，涨了50%")
    assert result["numbers"] == ["5块", "1.35万元", "50%"]
    assert result["codes"] == []
    assert result["quoted"] == []


def test_extract_numbers_time_units():
    result = extract_checkables("等3天，每月10号，2026年1月1日")
    # “号”“日”不是单位，只取数字本身。
    assert result["numbers"] == ["3天", "10", "2026年", "1月", "1"]


def test_extract_numbers_thousands_separator():
    result = extract_checkables("投1,000元")
    assert result["numbers"] == ["1,000元"]


def test_extract_numbers_dedup_preserves_order():
    result = extract_checkables("5块 5块 10块 5块")
    assert result["numbers"] == ["5块", "10块"]


def test_extract_numbers_no_numbers():
    result = extract_checkables("没有数字，只有文字")
    assert result == {"numbers": [], "codes": [], "quoted": []}


def test_extract_numbers_empty_text():
    assert extract_checkables("") == {"numbers": [], "codes": [], "quoted": []}
    assert extract_checkables(None) == {"numbers": [], "codes": [], "quoted": []}


# ---------- extract_checkables：基金代码 ----------


def test_extract_codes_fund_code_not_a_number():
    result = extract_checkables("买019441")
    assert result["codes"] == ["019441"]
    # 6 位纯数字是代码，不进 numbers，免得同一串数在两边各报一次。
    assert result["numbers"] == []


def test_extract_codes_not_five_or_seven_digits():
    result = extract_checkables("12345和1234567")
    assert result["codes"] == []
    assert result["numbers"] == ["12345", "1234567"]


def test_extract_codes_dedup_preserves_order():
    result = extract_checkables("019441和019172和019441")
    assert result["codes"] == ["019441", "019172"]


# ---------- extract_checkables：引号 ----------


def test_extract_quoted_chinese_quotes():
    result = extract_checkables("他说“每天定投5块”")
    assert result["quoted"] == ["每天定投5块"]


def test_extract_quoted_ascii_quotes():
    result = extract_checkables('他说"hello world"')
    assert result["quoted"] == ["hello world"]


def test_extract_quoted_none_and_empty():
    assert extract_checkables("没有引号")["quoted"] == []
    assert extract_checkables("空引号“”不算")["quoted"] == []


# ---------- grounding_check ----------


def test_grounding_all_match():
    result = grounding_check("每天定投5块买019441", "我每天定投5块买万家纳指019441")
    assert result == {
        "ok": True,
        "missing_numbers": [],
        "missing_codes": [],
        "missing_quoted": [],
    }


def test_grounding_number_mismatch():
    result = grounding_check("每天定投10块", "每天定投5块")
    assert result["ok"] is False
    assert result["missing_numbers"] == ["10块"]
    assert result["missing_codes"] == []


def test_grounding_code_mismatch():
    result = grounding_check("买019441", "买019172")
    assert result["ok"] is False
    assert result["missing_codes"] == ["019441"]
    assert result["missing_numbers"] == []


def test_grounding_empty_source_reports_everything():
    result = grounding_check('每天定投5块买019441，他说“我是小宋”', "")
    assert result["ok"] is False
    assert result["missing_numbers"] == ["5块"]
    assert result["missing_codes"] == ["019441"]
    assert result["missing_quoted"] == ["我是小宋"]


def test_grounding_quoted_is_soft_only():
    # 引号对不上只报告，不判错：没有数字/代码缺失时 ok 还是 True。
    result = grounding_check('他说“我是小宋”', "他说我是小张")
    assert result["ok"] is True
    assert result["missing_quoted"] == ["我是小宋"]
    assert result["missing_numbers"] == []
    assert result["missing_codes"] == []


def test_grounding_ignores_commas_and_spaces():
    # 逗号、空格去掉再比：“1,000元”对“1000元”，“5块”对“5 块”。
    assert grounding_check("投1,000元", "投1000元")["ok"] is True
    assert grounding_check("每天定投5块", "每天定投 5 块")["ok"] is True


# ---------- rank_most_likely_wrong：打分规则 ----------


def _item(mem_id, text, **metadata):
    return {"id": mem_id, "text": text, "metadata": metadata}


def test_rank_scoring_rules_add_up():
    items = [
        _item(
            "m1",
            "每天定投10块",
            user_confirmed=False,
            confidence=0.3,
            grounding_flag=True,
            needs_review=True,
            conflict_for_review=True,
            verification_status="unverified",
        )
    ]
    # 未确认且可信度<0.5 是 3 分，可信度<0.5（未确认）再加 1 分，
    # 跟出处对不上 2 分，待复核 2 分，冲突 2 分，未核实 1 分，共 11 分。
    ranked = rank_most_likely_wrong(items)
    assert len(ranked) == 1
    assert ranked[0]["id"] == "m1"
    assert ranked[0]["score"] == 11
    assert ranked[0]["reasons"] == [
        "未确认且可信度低",
        "跟出处对不上",
        "待复核",
        "存在冲突待复核",
        "未核实",
        "可信度低于 0.5",
    ]


def test_rank_missing_confidence_unconfirmed_scores_three():
    ranked = rank_most_likely_wrong([_item("m1", "某事", user_confirmed=False)])
    assert ranked[0]["score"] == 3
    assert ranked[0]["reasons"] == ["未确认且可信度低"]


def test_rank_confirmed_low_confidence_scores_zero():
    # 已确认的低可信度不加分：BOOS 亲口确认过的记忆不靠可信度数字怀疑。
    ranked = rank_most_likely_wrong(
        [_item("m1", "某事", user_confirmed=True, confidence=0.3)]
    )
    assert ranked == []


def test_rank_ordering_by_score_then_id():
    items = [
        # b：未确认但 confidence 0.9（不缺失也不<0.5），0 分，不进榜。
        _item("b", "低分", user_confirmed=False, confidence=0.9),
        _item("a", "高分", user_confirmed=False, grounding_flag=True),  # 3+2=5
        _item("c", "中分", user_confirmed=False, confidence=0.4),  # 3+1=4
    ]
    ranked = rank_most_likely_wrong(items)
    assert [entry["id"] for entry in ranked] == ["a", "c"]
    assert [entry["score"] for entry in ranked] == [5, 4]


def test_rank_tie_break_by_id():
    items = [
        _item("m2", "第二", user_confirmed=False, grounding_flag=True),
        _item("m1", "第一", user_confirmed=False, grounding_flag=True),
        _item("m10", "第十", user_confirmed=False, grounding_flag=True),
    ]
    ranked = rank_most_likely_wrong(items)
    # 分数相同按 id 排序（字符串序，确定性）。
    assert [entry["id"] for entry in ranked] == ["m1", "m10", "m2"]


def test_rank_zero_score_excluded():
    items = [
        _item("ok1", "已确认", user_confirmed=True, confidence=1.0,
              verification_status="verified"),
        _item("ok2", "未确认但可信度高", user_confirmed=False, confidence=0.9),
        _item("bad", "可疑", user_confirmed=False, confidence=0.4),
    ]
    ranked = rank_most_likely_wrong(items)
    assert [entry["id"] for entry in ranked] == ["bad"]


def test_rank_reasons_nonempty():
    items = [
        _item("a", "甲", user_confirmed=False, grounding_flag=True),
        _item("b", "乙", user_confirmed=False, needs_review=True),
    ]
    for entry in rank_most_likely_wrong(items):
        assert entry["reasons"], "每条上榜记忆都要有中文原因"
        assert all(isinstance(reason, str) and reason for reason in entry["reasons"])


def test_rank_top_n_truncation():
    items = [
        _item(f"m{i}", f"第{i}条", user_confirmed=False, grounding_flag=True)
        for i in range(7)
    ]
    assert len(rank_most_likely_wrong(items)) == 5
    assert len(rank_most_likely_wrong(items, top_n=2)) == 2
    assert len(rank_most_likely_wrong(items, top_n=10)) == 7


def test_rank_flat_metadata_fields_also_work():
    # metadata 字段直接平铺在 item 顶层时也要认。
    items = [
        {"id": "m1", "text": "某事", "user_confirmed": False,
         "confidence": 0.3, "grounding_flag": True}
    ]
    ranked = rank_most_likely_wrong(items)
    assert ranked[0]["score"] == 3 + 1 + 2
    assert ranked[0]["text"] == "某事"
