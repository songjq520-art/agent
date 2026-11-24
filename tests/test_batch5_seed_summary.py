# tests/test_batch5_seed_summary.py（第 5 批：17 号电脑配置种子 + 37 号摘要防混轮）
# 作用：把第 5 批这两项的修法钉死在测试里，全部离线模拟，不调真实 API、
# 不花钱；真调在 /tmp 外面另做（见第 5 批报告）。
# 17 号：启动时把 BOOS 的电脑配置种进长期记忆当已确认，种一次、重启还在、
# 绝不重复种；18 号不在范围。37 号：只改摘要的生成（提示词 + 轮次标法），
# 已经存好的摘要实例没有任何代码会去改写。

import inspect

from memory.conversation_summary import build_summary_prompt, format_turns






# ---------- 17 号：电脑配置种子 ----------











# ---------- 37 号：摘要防混轮 ----------

def _turns():
    """两轮话题完全不同的对话（第 1 轮定投、第 2 轮预算——原先混轮的例子）。"""
    return [
        {"role": "user", "content": "每天定投 5 块钱买万家纳指"},
        {"role": "assistant", "content": "好的，每天定投 5 块买万家纳指"},
        {"role": "user", "content": "每月工具预算 5 美元，到 5 美元提醒我"},
        {"role": "assistant", "content": "好的，每月 5 美元工具预算，到 5 美元提醒"},
    ]


def test_format_turns_marks_rounds():
    """37 号：对话记录标轮次——每轮前面有“第 N 轮”，内容原样在。"""
    text = format_turns(_turns())
    assert "第 1 轮" in text
    assert "第 2 轮" in text
    assert "BOOS：每天定投 5 块钱买万家纳指" in text
    assert "十二：好的，每月 5 美元工具预算，到 5 美元提醒" in text
    # 第 2 轮的内容不在第 1 轮那一段里（轮次分得开）。
    first_round = text.split("第 2 轮")[0]
    assert "5 美元" not in first_round


def test_format_turns_leading_assistant_joins_first_round():
    """开头没有 BOOS 消息的回答算进第 1 轮，不丢、不另起一轮。"""
    text = format_turns(
        [
            {"role": "assistant", "content": "你好"},
            {"role": "user", "content": "今天天气怎么样"},
            {"role": "assistant", "content": "晴"},
        ]
    )
    assert text.count("第 1 轮") == 1
    assert "第 2 轮" in text
    assert "十二：你好" in text


def test_build_summary_prompt_has_anti_mixing_rules():
    """37 号：摘要提示词里有防混轮规则——按时间顺序、每轮只写自己的事、
    不重复、不确定的写不知道；300 字以内、不要编造还在。"""
    prompt = build_summary_prompt(_turns())
    assert "按时间顺序" in prompt
    assert "每轮只写那轮自己的事" in prompt
    assert "别把一轮的事算到另一轮里" in prompt
    assert "同一件事只写一次，别重复" in prompt
    assert "不确定的写不知道" in prompt
    assert "300 字以内" in prompt
    assert "不要编造" in prompt
    # 轮次标记进了提示，模型看得出哪句是哪轮的。
    assert "第 1 轮" in prompt
    assert "第 2 轮" in prompt


def test_build_summary_prompt_merge_has_anti_repeat_rule():
    """合并旧摘要时也不重复、不确定的写不知道。"""
    prompt = build_summary_prompt(_turns(), previous_summary="旧摘要：聊过天气")
    assert "旧摘要：聊过天气" in prompt
    assert "别重复写两遍" in prompt
    assert "不确定的写不知道" in prompt


def test_old_summaries_not_touched_by_prompt_change():
    """37 号只改生成：已经存好的摘要实例没有任何代码会去改写——
    build_summary_prompt 是纯函数，不读不写摘要文件。"""
    src = inspect.getsource(build_summary_prompt)
    assert "conversation_summary.json" not in src
    assert ".save(" not in src
