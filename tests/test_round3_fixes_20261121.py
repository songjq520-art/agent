# tests/test_round3_fixes_20261121.py（第三轮网页测 12 项的回归测试，2026-11-22）
# 全部离线：不调真实模型、不碰真实账本、不联网。
# 覆盖：#8 过滤器（问句/假设句/确认词/半句话/当下的想法）+ #8 add_memory
# 谎报与双存 + #7 删除剩余数 + #11 计算器单位/口径/原话 + #1 审计口吻与
# 星期对照 + #11 预算设置标注 + #4 对话时间 + #5 确认卡人话摘要 +
# #6/#7 删除流程规矩 + #9 显示层脱敏。

import time
from datetime import datetime, timedelta, timezone

import pytest

from config import BudgetTracker
from main import _memory_block, run_question
from memory.calculator import calculator_block
from memory.conversation_summary import ConversationSummaryStore
from memory.sqlite_store import (
    SQLiteStore,
    _looks_like_fact,
    _strip_confirmation_prefix,
)
from tools.agent_loop import build_loop_system_message
from tools.memory_tool import (
    _run_add_memory,
    _run_delete_memories,
    _run_delete_memory,
    bound_memory_store,
)
from web.server import (
    ChatState,
    _clean_approval_args,
    _human_approval_summary,
    _redacted_messages,
)


@pytest.fixture
def store(tmp_path):
    """指向临时目录的真 SQLiteStore（每个测试独立，不碰真实账本）。"""
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


# ---------- #8 过滤器：问句 ----------


@pytest.mark.parametrize("sentence", [
    "我爱吃什么",  # “什么”结尾
    "我喜不喜欢吃鸡杂面",  # 正反问 A-不-A
    "我的定投现在是什么情况",  # “什么情况”结尾
    "定投计划怎么样了",
    "有没有人来过",  # 疑问词开头
    "你知不知道他是谁",  # 正反问
])
def test_filter_questions_not_stored(sentence):
    ok, reason = _looks_like_fact(sentence)
    assert ok is False
    assert reason == "问句"


# ---------- #8 过滤器：假设句 ----------


@pytest.mark.parametrize("sentence", [
    "如果每天定投5块，年化百分之6，35年以后有多少钱",
    "假设明天下雨",
    "要是下周六不下雨就去爬山",
    "万一他不来我就自己去",
])
def test_filter_hypotheticals_not_stored(sentence):
    ok, reason = _looks_like_fact(sentence)
    assert ok is False
    assert reason == "假设句"


# ---------- #8 过滤器：半句话 ----------


@pytest.mark.parametrize("sentence", ["这几条取消", "那条不要", "这个留着"])
def test_filter_fragments_not_stored(sentence):
    ok, reason = _looks_like_fact(sentence)
    assert ok is False
    assert reason == "半句话"


# ---------- #8 过滤器：当下的想法 ----------


@pytest.mark.parametrize("sentence", [
    "我想喝茶了",
    "还有爬山我也想",
    "我现在想喝茶",
    "我馋火锅了",
])
def test_filter_transient_wants_not_stored(sentence):
    ok, reason = _looks_like_fact(sentence)
    assert ok is False
    assert reason == "当下的想法"


# ---------- #8 过滤器：真事实和计划照存 ----------


@pytest.mark.parametrize("sentence", [
    "我喜欢喝茶",
    "下周六要去爬山",
    "我想以后每天7点起床",  # 带“以后/每天”，是未来的计划，不是当下
    "我昨天中午吃了饺子",
    "我不喜欢吃辣",
    "忘记带钥匙了",  # 句尾带“了”的回忆句
    "我想去爬山",  # 表态的意图，拿不准先留着（沉底规则管它）
])
def test_filter_keeps_facts_and_plans(sentence):
    ok, _reason = _looks_like_fact(sentence)
    assert ok is True


# ---------- #8 确认词前缀 ----------


def test_strip_confirmation_prefix():
    assert _strip_confirmation_prefix("是的删掉") == "删掉"
    assert _strip_confirmation_prefix("是的，我喜欢喝茶") == "我喜欢喝茶"
    assert _strip_confirmation_prefix("对，没错") == "没错"
    assert _strip_confirmation_prefix("好，我喜欢喝茶") == "我喜欢喝茶"
    assert _strip_confirmation_prefix("可以删掉") == "删掉"
    # 不剥：句中、或一个字后面没跟分隔符（“好吃”不能剥成“吃”，
    # “对不起”不能剥成“不起”）。
    assert _strip_confirmation_prefix("好吃") == "好吃"
    assert _strip_confirmation_prefix("对不起") == "对不起"
    assert _strip_confirmation_prefix("对面那家店很好吃") == "对面那家店很好吃"
    assert _strip_confirmation_prefix("我喜欢喝茶") == "我喜欢喝茶"


def test_save_turn_confirmation_prefix(store):
    """“是的删掉”剥完是“删掉”，丢掉；“是的，我喜欢喝茶”剥完是事实，照存。"""
    receipt = store.save_turn(
        [{"role": "user", "content": "是的删掉"}], turn_id="t-confirm-delete"
    )
    assert receipt["stored"] == []
    assert [m["text"] for m in store.get_all()] == []
    receipt = store.save_turn(
        [{"role": "user", "content": "是的，我喜欢喝茶"}], turn_id="t-confirm-fact"
    )
    assert receipt["stored"] == ["我喜欢喝茶"]
    assert [m["text"] for m in store.get_all()] == ["我喜欢喝茶"]


def test_save_turn_round3_garbage_all_dropped(store):
    """第三轮账本里的垃圾句：save_turn 一句都不存。"""
    garbage = [
        "我爱吃什么",
        "如果每天定投5块，年化百分之6，35年以后有多少钱",
        "我想喝茶了",
        "还有爬山我也想",
        "这几条取消",
        "我喜不喜欢吃鸡杂面",
        "我的定投现在是什么情况",
    ]
    for i, text in enumerate(garbage):
        receipt = store.save_turn(
            [{"role": "user", "content": text}], turn_id=f"t-garbage-{i}"
        )
        assert receipt["stored"] == [], text
    assert store.get_all() == []


# ---------- #8 add_memory：谎报 + 双存 ----------


def test_add_memory_returns_ok_on_success(store):
    """SQLite 的 add_memory_text 返回 {"id", "memory", "event"}，
    没有 "results"——以前这里读 "results" 读不到，每次都谎报“没存进去”。"""
    with bound_memory_store(store):
        first = _run_add_memory({"text": "后天要去体检"})
        second = _run_add_memory({"text": "后天要去体检"})
    assert first["ok"] is True
    assert first["data"]["duplicate"] is False
    assert first["data"]["memory_id"]
    assert second["ok"] is True
    assert second["data"]["duplicate"] is True
    assert second["data"]["memory_id"] == first["data"]["memory_id"]
    assert [m["text"] for m in store.get_all()].count("后天要去体检") == 1


def test_add_memory_strips_date_annotation(store):
    """模型在句尾补的日期标注先剥掉再存，不存第二遍。"""
    with bound_memory_store(store):
        _run_add_memory({"text": "后天要去体检"})
        annotated = _run_add_memory({"text": "后天要去体检（2026-11-24，周二）"})
    assert annotated["ok"] is True
    assert annotated["data"]["duplicate"] is True
    assert annotated["data"]["text"] == "后天要去体检"
    assert [m["text"] for m in store.get_all()].count("后天要去体检") == 1


def test_add_memory_sensitive_is_truthful(store):
    """含敏感信息是真拦，不是“没存进去”那种含糊说法。"""
    with bound_memory_store(store):
        result = _run_add_memory({"text": "我的密码是 abc12345"})
    assert result["ok"] is False
    assert "敏感" in (result.get("error") or "")
    assert store.get_all() == []


# ---------- #7 删除剩余数 ----------


def _mem_id(store, text):
    for item in store.get_all(include_sensitive=True):
        if item["text"] == text:
            return item["id"]
    raise AssertionError(f"库里没有这条记忆：{text}")


def test_delete_memory_reports_remaining(store):
    """删一条后，工具要告诉模型还剩几条同目标的（第三轮 seq 137 的教训）。"""
    with bound_memory_store(store):
        _run_add_memory({"text": "每天定投5块买万家纳指"})
        _run_add_memory({"text": "我的定投现在是什么情况"})
        first = _run_delete_memory({
            "memory_id": _mem_id(store, "每天定投5块买万家纳指"),
            "text": "每天定投5块买万家纳指",
            "anchor_text": "定投",
        })
    assert first["ok"] is True
    assert first["data"]["remaining_count"] == 1
    assert first["data"]["remaining_texts"] == ["我的定投现在是什么情况"]


def test_delete_memories_reports_zero_remaining_when_all_gone(store):
    with bound_memory_store(store):
        _run_add_memory({"text": "每天定投5块买万家纳指"})
        _run_add_memory({"text": "我的定投现在是什么情况"})
        result = _run_delete_memories({
            "items": [
                {"memory_id": _mem_id(store, "每天定投5块买万家纳指"),
                 "text": "每天定投5块买万家纳指"},
                {"memory_id": _mem_id(store, "我的定投现在是什么情况"),
                 "text": "我的定投现在是什么情况"},
            ],
            "anchor_text": "定投",
        })
    assert result["ok"] is True
    assert result["data"]["remaining_count"] == 0
    assert result["data"]["remaining_texts"] == []
    assert store.get_all() == []


# ---------- #11 计算器：单位、口径、精确数 ----------


def test_dca_usd_keeps_dollars():
    block = calculator_block("如果我每天定投5美元，年化百分之6，35年以后有多少钱")
    assert "每天定投 5 美元" in block
    assert "140,408 美元" in block
    assert "213,707 美元" in block
    assert "按每月投 150 美元算" in block
    assert "别说成每月月初定投" in block


def test_dca_rmb_exact_numbers_and_method_labels():
    block = calculator_block("如果我每天定投5块，年化百分之6，35年以后有多少钱")
    assert "每天定投 5 块" in block
    assert "140,408 块" in block
    assert "213,707 块" in block
    assert "每月投一次，月末" in block
    assert "每年投一次，年末" in block


def test_dca_dollar_sign_form():
    block = calculator_block("每天定投 $5，年化6%，35年后有多少钱")
    assert "每天定投 5 美元" in block
    assert "140,408 美元" in block


# ---------- #11 计算器用原话，不用拼了记忆块的 prompt ----------


class _DirectAnswerProvider:
    """直接回答的假模型：记下收到的消息（供测试检查计算器块用）。"""

    _LAST = None

    def __init__(self, api_key=None, budget_tracker=None, model=None):
        self.model = model or "deepseek-flash"
        self.tool_calls_log = []
        _DirectAnswerProvider._LAST = self

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        return {"text": "你好。", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        self.tool_calls_log.append({"messages": list(messages), "thinking": thinking})
        return {
            "text": "你好。",
            "tool_calls": [],
            "reasoning_content": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost_usd": 0.001,
            "model": self.model,
        }

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None, model=None, max_tokens=None):
        return {"text": "你好。", "usage": None, "cost_usd": 0.0, "model": self.model}


class _EightPercentMemory:
    """假记忆库：搜到一条含“年化百分之八”的记忆（不是问题本身的年化）。"""

    def search(self, prompt, top_k=5, reinforce=False, turn_id=None):
        return [{
            "text": "每天定投5块，年化百分之八，30年后能有多少钱",
            "metadata": {"source": "inferred", "confirmed": False},
        }]

    def note_context(self, memories):
        return None

    def add_turn(self, recent):
        return {}

    def save_turn(self, recent, turn_id=None):
        return {"pending": True, "results": [], "queued": True, "turn_id": turn_id}

    def infer_from_turn(self, messages):
        return {}


class _StubSearchService:
    """占位搜索服务：本测试里模型直接回答，搜索一次也不调。"""


def test_run_question_calculator_uses_original_prompt(tmp_path, monkeypatch):
    """run_question 里计算器用原话算：网页调进来时 prompt 已经拼了记忆块，
    块里的记忆也可能含“年化/定投”——用原话，第一个“年化”才不会来自记忆。"""
    import main as main_module
    from tasks.task_store import TaskStore
    from tools.tool_gateway import ToolGateway

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setitem(main_module.PROVIDER_CLASSES, "deepseek", _DirectAnswerProvider)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    question = "如果我每天定投5块，年化百分之6，35年以后有多少钱"
    # 网页调进来时 prompt 是拼了记忆块的 model_prompt（块里有年化 8% 的记忆）。
    prefixed = (
        "以下是与你的问题相关的记忆（历史记录，仅供参考）：\n"
        "- 【提取所得，未经逐条确认】每天定投5块，年化百分之八，30年后能有多少钱\n\n"
        + question
    )
    outcome = run_question(
        prefixed,
        budget_tracker=tracker,
        mem0_store=_EightPercentMemory(),
        tool_gateway=ToolGateway(),
        task_store=TaskStore(file_path=str(tmp_path / "tasks.json")),
        search_service=_StubSearchService(),
        original_prompt=question,
    )
    assert outcome["mode"] == "real"
    joined = "\n".join(
        (m.get("content") or "")
        for call in _DirectAnswerProvider._LAST.tool_calls_log
        for m in call["messages"]
    )
    assert "每天定投 5 块、年化 6%、35 年" in joined
    assert "140,408" in joined
    assert "年化 8%、35 年" not in joined


# ---------- #1 审计口吻 + 星期对照，#11 预算设置标注 ----------


def test_memory_block_bans_audit_speak_variants():
    block = _memory_block([
        {"text": "我喜欢喝茶", "metadata": {"source": "user", "confirmed": True}},
    ])
    for banned in ["我不敢当成确定的结论", "根据我这边记录的信息",
                   "这两条都只是背景参考", "按这条线索看"]:
        assert banned in block
    assert "先说结论再说依据" in block


def test_memory_block_week_hint_labels_next_week():
    """今天是周日时，3 天后（周三）是“下周三”，不是“本周三”。"""
    shanghai = timezone(timedelta(hours=8))
    today = datetime.now(shanghai).date()
    target = today + timedelta(days=3)
    monday = today - timedelta(days=today.weekday())
    label = "本周" if monday <= target < monday + timedelta(days=7) else "下周"
    weekday_cn = ("一", "二", "三", "四", "五", "六", "日")[target.weekday()]
    event_date = (today + timedelta(days=2)).isoformat()
    block = _memory_block([
        {"text": "后天要去体检",
         "metadata": {"source": "user", "confirmed": True, "event_date": event_date}},
    ])
    assert f"今天是 {today.isoformat()}" in block
    assert f"{target.isoformat()}（{label}{weekday_cn}）" in block
    assert "星期对照" in block


def test_memory_block_budget_setting_label():
    block = _memory_block([
        {"text": "每月预算 5 美元，到 5 美元提醒，不自动停。",
         "metadata": {"source": "user", "confirmed": True}},
        {"text": "每天定投 5 块买万家纳指",
         "metadata": {"source": "user", "confirmed": True,
                      "money": {"amount": 5, "currency": "CNY", "period": "day",
                                "target": "万家纳指"}}},
    ])
    lines = block.split("\n")
    budget_line = next(ln for ln in lines if "每月预算 5 美元" in ln)
    dca_line = next(ln for ln in lines if "每天定投 5 块" in ln)
    assert "这是工具调用的预算设置，不是投资/定投计划" in budget_line
    assert "这是工具调用的预算设置" not in dca_line


def test_memory_block_no_event_date_no_week_hint():
    block = _memory_block([
        {"text": "我喜欢蓝色", "metadata": {"source": "user", "confirmed": True}},
    ])
    assert "事件日期" not in block
    assert "今天是" not in block
    assert "星期对照" not in block


# ---------- #4 对话时间：只看不说话，时间不该变 ----------


class _StubSummarizer:
    """假摘要器：deterministic，不调模型。"""

    def __call__(self, turns, previous_summary=None):
        topics = "、".join(
            (t.get("content") or "")[:6] for t in turns if t.get("role") == "user"
        )
        return f"摘要[{topics}]"


def _summary_store(tmp_path, name="conversation_summary.json", **kwargs):
    return ConversationSummaryStore(
        file_path=tmp_path / name,
        summarizer=_StubSummarizer(),
        **kwargs,
    )


def _msgs(n_rounds, topic="今天天气真好"):
    out = []
    for i in range(n_rounds):
        out.append({"role": "user", "content": f"{topic}第{i + 1}句"})
        out.append({"role": "assistant", "content": f"好的，第{i + 1}个回答"})
    return out


def test_sync_same_messages_keeps_timestamp(tmp_path):
    """切对话/双击：消息没变时 sync 不更新时间；消息多了才更新。"""
    store = _summary_store(tmp_path)
    messages = _msgs(2)
    store.sync(messages)
    t1 = store.list_conversations()[0]["updated_at"]
    assert t1
    time.sleep(0.05)
    store.sync(messages)
    assert store.list_conversations()[0]["updated_at"] == t1
    time.sleep(0.05)
    grown = messages + [
        {"role": "user", "content": "新问题"},
        {"role": "assistant", "content": "新回答"},
    ]
    store.sync(grown)
    assert store.list_conversations()[0]["updated_at"] != t1


def test_flush_and_compact_do_not_touch_timestamp(tmp_path):
    """总结不加消息：flush / maybe_compact 不更新时间，但摘要照写。"""
    store = _summary_store(tmp_path)
    messages = _msgs(2)
    store.sync(messages)
    t1 = store.list_conversations()[0]["updated_at"]
    time.sleep(0.05)
    assert store.flush(messages) is True
    assert store.list_conversations()[0]["updated_at"] == t1
    assert store.current_record()["summary"]
    # 压历史：很小的 token 上限，逼出一次 maybe_compact。
    store2 = _summary_store(tmp_path, name="cs2.json", token_limit=50)
    long_messages = _msgs(10)
    store2.sync(long_messages)
    t2 = store2.list_conversations()[0]["updated_at"]
    time.sleep(0.05)
    assert store2.maybe_compact(long_messages) is True
    assert store2.list_conversations()[0]["updated_at"] == t2
    assert len(long_messages) == 10  # 只留最近 5 轮


# ---------- #5 确认卡人话摘要 ----------


def test_flush_first_time_stamps_new_record(tmp_path):
    """新记录第一次 flush（没 sync 过）：补 updated_at，排到列表最前面。

    回归：全量测试里 test_title_falls_back_to_summary_first_sentence_then_default
    曾因 flush 不写 updated_at 而失败——新记录的 updated_at 是 None，
    _sort() 把它排到旧对话后面，list[0] 读到旧对话。
    """
    store = _summary_store(tmp_path)
    store.sync([{"role": "assistant", "content": "我是十二。"}])
    store2 = _summary_store(tmp_path)
    store2.begin_conversation()
    assert store2.flush([{"role": "user", "content": "广州天气怎么样？明天出门。"}]) is True
    first = store2.list_conversations()[0]
    assert first["title"] == "广州天气怎么样"
    assert first["updated_at"]


def test_maybe_compact_first_time_stamps_new_record(tmp_path):
    """新记录第一次 maybe_compact（没 sync 过）：同样补 updated_at。"""
    store = _summary_store(tmp_path, token_limit=50)
    messages = _msgs(8)
    assert store.maybe_compact(messages) is True
    assert store.list_conversations()[0]["updated_at"]


def test_human_approval_summary():
    assert _human_approval_summary(
        "delete_memory",
        {"memory_id": "abc", "text": "我喜欢喝茶", "anchor_text": "喝茶"},
    ) == "删除记忆「我喜欢喝茶」"
    assert _human_approval_summary(
        "delete_memories",
        {"items": [{"memory_id": "a", "text": "第一条"},
                   {"memory_id": "b", "text": "第二条"}],
         "anchor_text": "定投"},
    ) == "删除 2 条记忆：第一条 等"
    assert _human_approval_summary(
        "update_memory", {"old_text": "我爱吃拉面", "new_text": "我爱吃米线"}
    ) == "把记忆「我爱吃拉面」改成「我爱吃米线」"
    assert _human_approval_summary(
        "revive_memory", {"text": "我喜欢喝茶"}
    ) == "恢复记忆「我喜欢喝茶」"
    assert _human_approval_summary("unknown_tool", {}) == "确认执行这个操作"


def test_clean_approval_args_strips_internal_keys():
    cleaned = _clean_approval_args({
        "memory_id": "abc",
        "text": "我喜欢喝茶",
        "anchor_text": "喝茶",
        "_anchor_match": "程序已核对：……",
        "items": [{"memory_id": "a", "text": "第一条"}],
    })
    assert "memory_id" not in cleaned
    assert "_anchor_match" not in cleaned
    assert cleaned["text"] == "我喜欢喝茶"
    assert cleaned["anchor_text"] == "喝茶"
    assert cleaned["items"] == [{"text": "第一条"}]


# ---------- #9 显示层脱敏 ----------


def test_redacted_messages_hide_password_and_keep_original():
    messages = [
        {"role": "user", "content": "我的密码是 abc12345，电话 13812345678"},
        {"role": "assistant", "content": "好的。"},
    ]
    redacted = _redacted_messages(messages)
    assert "abc12345" not in redacted[0]["content"]
    assert "【已隐藏敏感信息】" in redacted[0]["content"]
    assert redacted[1]["content"] == "好的。"
    # 原来的消息没被改（只改返回的副本）。
    assert messages[0]["content"] == "我的密码是 abc12345，电话 13812345678"


def test_open_conversation_redacts_password_in_bubbles(tmp_path):
    """#9：回到旧对话时，气泡里的密码脱敏显示；内存历史和存着的记录不动。"""
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    store = _summary_store(tmp_path)
    state = ChatState(budget_tracker=tracker, summary_store=store)
    state.messages = [
        {"role": "user", "content": "我的密码是 abc12345"},
        {"role": "assistant", "content": "好的。"},
    ]
    state.reset()
    conv_id = state.list_conversations()[0]["id"]
    state.messages = _msgs(1, topic="另一段对话")
    state.reset()
    result = state.open_conversation(conv_id)
    assert result is not None
    assert "abc12345" not in result["messages"][0]["content"]
    assert "【已隐藏敏感信息】" in result["messages"][0]["content"]
    # 内存里的历史还是原文（只改返回的副本）。
    assert state.messages[0]["content"] == "我的密码是 abc12345"
    # 存着的记录还是原文。
    assert store._find_record(conv_id)["recent"][0]["content"] == "我的密码是 abc12345"


# ---------- #6/#7 删除流程规矩 ----------


def test_loop_system_message_has_delete_rules():
    text = build_loop_system_message()
    assert "删记忆别问得太烦" in text
    assert "别在聊天里把候选一条一条列出来问" in text
    assert "全部删掉" in text
    assert "删完照实报" in text
    assert "remaining_count" in text
    assert "都清干净了" in text
