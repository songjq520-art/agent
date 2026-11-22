# tests/test_round4_fixes_20261121.py（第四轮网页测 5 项修复，2026-11-21）
# 只测修复点本身；全部离线，不碰真实账本、不联网、不调真实模型。
from datetime import date as _date
from pathlib import Path

from config import BudgetTracker
from main import _memory_block, _mentions_date_or_weekday
from memory.calculator import calculator_block
from memory.conversation_summary import _copy_messages as _summary_copy_messages
from memory.conversation_transcript import _copy_messages as _transcript_copy_messages
from memory.sqlite_store import resolve_event_date
from web.server import ChatState, _human_approval_summary, _summarize_delete_texts


# ---------- #3 先说结论 + 下周三 ----------


def test_resolve_event_date_weekday_expressions():
    """“下周三/本周三/下周六/上周三”这类带星期的相对日期要算对。"""
    assert resolve_event_date("下周三我要出差", _date(2026, 11, 22)) == "2026-11-25"
    assert resolve_event_date("下周三我要出差", _date(2026, 11, 21)) == "2026-11-25"
    assert resolve_event_date("本周三开会", _date(2026, 11, 22)) == "2026-11-18"
    assert resolve_event_date("下周六要去爬山", _date(2026, 11, 22)) == "2026-11-28"
    assert resolve_event_date("下周星期日回家", _date(2026, 11, 22)) == "2026-11-29"
    assert resolve_event_date("上周三开会", _date(2026, 11, 22)) == "2026-11-11"
    assert resolve_event_date("我喜欢蓝色", _date(2026, 11, 22)) is None
    assert resolve_event_date("后天去体检", _date(2026, 11, 22)) == "2026-11-24"


def test_mentions_date_or_weekday():
    assert _mentions_date_or_weekday("下周三我要出差") is True
    assert _mentions_date_or_weekday("公司临时又不要我去了") is False
    assert _mentions_date_or_weekday("我喜欢蓝色") is False
    assert _mentions_date_or_weekday("11月25日开会") is True
    assert _mentions_date_or_weekday("明天去体检") is False  # 相对日子词走 event_date，不触发文本兜底


def test_memory_block_outdated_memories_still_get_week_hint():
    """已经存进去的老记忆没有 event_date，但正文提到“下周三”，
    星期对照也要给出来——不然模型会把 11 月 25 日说成“这周三”。"""
    mems = [
        {"text": "下周三我要出差", "metadata": {"source": "inferred"}},
        {"text": "公司临时又不要我去了", "metadata": {"source": "inferred"}},
    ]
    block = _memory_block(mems)
    assert "星期对照" in block
    assert "2026-11-25（下周三）" in block
    assert "这周三" not in block
    # 结论先行的收尾句在块的最后（模型读完记忆立刻看到它）。
    assert block.strip().endswith(
        "回答时第一句必须是结论（一句话），不要先复述上面的记忆、不要先说“我这边记录是……”。"
    )


def test_memory_block_no_date_mention_still_no_week_hint():
    """没提日期的记忆：星期对照不出现（老行为不变）。"""
    block = _memory_block([{"text": "我喜欢蓝色", "metadata": {}}])
    assert "事件日期" not in block
    assert "今天是" not in block
    assert "星期对照" not in block


# ---------- #8 “记住”回执只说记下了 ----------


class _MiniMem0Store:
    """最小的假记忆库：只实现 ask() 的“记住”短路用到的三个方法。"""

    def __init__(self):
        self.direct_stored = []
        self.memories = []

    def add_memory_text(self, text, source="user", turn_id=None, **kwargs):
        self.direct_stored.append({"text": text, "source": source})
        if text not in self.memories:
            self.memories.append(text)
        return {"id": "d1", "memory": text, "event": "ADD"}

    def search(self, query, top_k=5, reinforce=False, turn_id=None):
        return [
            {"id": f"m{i}", "text": text, "score": 0.9, "source": "semantic"}
            for i, text in enumerate(self.memories[:top_k])
        ]

    def note_context(self, memories):
        return None


def test_ask_remember_receipt_is_one_sentence_no_model_call(tmp_path, monkeypatch):
    """第四轮 #8：回执只说记下了（带解析出的日期），不调模型——
    模型会多嘴补一段无关内容（比如出差）。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _MiniMem0Store()
    state = ChatState(budget_tracker=tracker, mem0_store=store)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("记住回执不该调模型")

    monkeypatch.setattr("web.server.run_question", _must_not_be_called)
    result = state.ask("记住，我后天去体检")
    assert result["mode"] == "real"
    assert result["model"] == "程序直存"
    assert result["cost_usd"] == 0.0
    # 只有一句回执，没有第二段、没有出差。
    reply = result["reply"]
    assert "\n\n" not in reply
    assert "出差" not in reply
    assert reply.startswith("记下了：我后天去体检")
    # 日期按当天算出来，格式是“11月24日，周二”这种。
    iso = resolve_event_date("我后天去体检")
    d = _date.fromisoformat(iso)
    weekday = "一二三四五六日"[d.weekday()]
    assert f"（{d.month}月{d.day}日，周{weekday}）" in reply
    # 记忆确实存了。
    assert store.direct_stored == [{"text": "我后天去体检", "source": "user"}]


# ---------- #11 算数要算准 ----------


def test_dca_daily_method_matches_question():
    """6 块、6%、20 年：每天投（按日复利）是 84,672，不是模型手算的 78,842。"""
    block = calculator_block("如果每天定投6块，年化百分之6，20年后有多少钱")
    assert "84,672" in block
    assert "按每天投 6 块算" in block
    assert "按日复利" in block
    assert "78,842" not in block
    # 每天投那行排在最前面（问题说每天定投，优先用它）。
    assert block.index("按每天投") < block.index("按每月投")
    assert block.index("按每天投") < block.index("按每年")
    # 每月/交易日两行还在（老口径没丢）。
    assert "83,167" in block
    assert "55,620" in block


def test_dca_old_pinned_numbers_still_present():
    """老测试钉死的数还在：5 块、6%、35 年是 140,408 / 213,707。"""
    block = calculator_block("如果我每天定投5块，年化百分之6，35年以后有多少钱")
    assert "140,408" in block
    assert "213,707" in block
    assert "每月投一次，月末" in block
    assert "每年投一次，年末" in block


# ---------- #4 模型/成本行统一 ----------


def test_copy_messages_keeps_model_and_cost_for_assistant():
    """_copy_messages：assistant 带 model/cost_usd，user 不带，其余键丢掉。"""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "model": "deepseek-flash",
         "cost_usd": 0.000123, "extra": 1},
        {"role": "assistant", "content": "receipt", "model": "程序直存", "cost_usd": 0.0},
        {"role": "system", "content": "skip me"},
    ]
    for copy_fn in (_transcript_copy_messages, _summary_copy_messages):
        copied = copy_fn(msgs)
        assert copied == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok", "model": "deepseek-flash",
             "cost_usd": 0.000123},
            {"role": "assistant", "content": "receipt", "model": "程序直存",
             "cost_usd": 0.0},
        ]
    # 老形状（没有 model/cost）照常复制，不多加键。
    assert _transcript_copy_messages([{"role": "assistant", "content": "old"}]) == [
        {"role": "assistant", "content": "old"}
    ]


def test_history_render_passes_meta():
    """前端两处恢复对话的渲染也要传 meta（静态检查：接线在）。"""
    html = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "function metaForMessage(m)" in html
    assert html.count("metaForMessage(m)") >= 3  # 定义 1 次 + 两处恢复渲染各 1 次
    assert "addMsg(m.role === 'user' ? 'user' : 'ai', m.content || '', metaForMessage(m));" in html


# ---------- #5 确认卡列出全部要删的记忆 ----------


def test_approval_summary_lists_all_delete_items():
    """第四轮 #5：3 条全列出来，相同原文标 ×2，不再只列第一条加“等”。"""
    summary = _human_approval_summary(
        "delete_memories",
        {"items": [
            {"memory_id": "a", "text": "后天要去体检"},
            {"memory_id": "b", "text": "后天要去体检"},
            {"memory_id": "c", "text": "我后天去体检"},
        ]},
    )
    assert summary == "删除 3 条记忆：「后天要去体检」×2、「我后天去体检」"


def test_summarize_delete_texts_shortens_only_when_long():
    """太长了再精简：单条截短；实在放不下才用“等 N 条”收尾。"""
    assert _summarize_delete_texts(["第一条", "第二条"]) == "「第一条」、「第二条」"
    long3 = ["x" * 40, "y" * 40, "z" * 40]
    assert _summarize_delete_texts(long3) == "「xxxxxxxxxxxx…」、「yyyyyyyyyyyy…」、「zzzzzzzzzzzz…」"
    many = [f"记忆第{i}号内容很长很长很长" for i in range(8)]
    result = _summarize_delete_texts(many)
    assert result.endswith("等 5 条")
    assert "记忆第0号" in result
