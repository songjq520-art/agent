# tests/test_reanchor_20261124.py（四件套·实测问题四：已确认事实回锚）
#
# 覆盖契约「六、验收」回锚观察项与方案追加决定十二问题四的定版：
# ① 已确认事实每轮强制注入提示词，独立成区、标注「已确认」、
# 写明是定案（不许再说不确定、不许重复向 BOOS 确认）；
# ② 长对话中途回锚：第 8、16、24…轮用加强版回锚块替换常规块
# （重申已确认事实与当前立场），其余轮只有常规块；
# ③ 未确认的普通记忆（提取所得/来源未确认）不许混进已确认区；
# ④ 注入封顶生效（常规 12 条/900 字、回锚 30 条/2400 字，
# 取舍顺序：规则 > 身份槽位 > 其余已确认，同层 importance
# 高者优先、再新的优先）；
# ⑤ 实测第 ⑦ 组桩用例：身份事实「十二是十一的弟弟」每轮都在
# 已确认区里、带「（已确认）」标注——被追问时不许再摇摆的
# 机制面证据（模型当场怎么答是模型的事，本测试只验机制把
# 定案事实摆到了它每轮眼前）。
#
# 全部离线：向量/判官打桩成 None（同 intake/g7 口径），模型用
# _DirectAnswerProvider 打桩（同 step0 口径），不调真 API。

import pytest

import main as main_module
from config import BudgetTracker
from main import run_question
from memory import reanchor as _reanchor
from memory.sqlite_store import SQLiteStore
from tasks.task_store import TaskStore
from tools.tool_gateway import ToolGateway

_ANCHOR_TITLE = "【" + "已确认事实" + "】"
_REANCHOR_TITLE = "【回锚·" + "已确认事实" + "】"


# ---------------- 通用件 ----------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    """干净的离线 SQLiteStore：向量/判官打桩，整理路由关掉。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    monkeypatch.setattr(
        SQLiteStore, "_ask_importance_judge",
        lambda self, text, context=None: None,
    )
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.organizing_router = lambda: None
    return s


def _row(text, *, confirmed=False, slot=None, is_rule=False, curated=False,
        source=None, importance=None, write_seq=None):
    """_all_live_memories() 形状的假行。"""
    meta = {}
    if confirmed:
        meta["source"] = "user"
        meta["user_confirmed"] = True
        meta["confirmed"] = True
    elif source:
        meta["source"] = source
    if slot:
        meta["slot"] = slot
    if is_rule:
        meta["is_rule"] = True
    if curated:
        meta["curated"] = True
    return {
        "id": f"id-{text[:8]}", "text": text, "metadata": meta,
        "slot": slot, "source": meta.get("source"),
        "user_confirmed": bool(confirmed),
        "importance": importance, "write_seq": write_seq,
    }


# ---------------- 纯判定：认不认 ----------------


def test_is_anchor_fact_uses_existing_criteria():
    # BOOS 亲口确认（user_confirmed）——含问题一 owner_confirmed
    # 升级后的形状（source=user + user_confirmed=True）。
    assert _reanchor.is_anchor_fact(_row("十二是十一的弟弟", confirmed=True))
    # 身份槽位行（即便来源是提取的，槽位本身就是定案口径）。
    assert _reanchor.is_anchor_fact(
        _row("名字是十二", slot="agent_name", source="inferred"))
    # 规则行、精选置顶行（核心受保护）。
    assert _reanchor.is_anchor_fact(_row("删记忆前必须先问用户", is_rule=True))
    assert _reanchor.is_anchor_fact(_row("精选的一条", curated=True))
    # 未确认的普通记忆一律不算。
    assert not _reanchor.is_anchor_fact(_row("用户喜欢吃冒菜", source="inferred"))
    assert not _reanchor.is_anchor_fact(_row("没有来源信息的旧记忆"))
    assert not _reanchor.is_anchor_fact(None)
    assert not _reanchor.is_anchor_fact({})


# ---------------- 纯判定：挑与封顶 ----------------


def test_select_excludes_unconfirmed_and_dedupes():
    rows = [
        _row("十二是十一的弟弟", confirmed=True),
        _row("十二是十一的弟弟", confirmed=True), # 同文重复只留一条
        _row("用户养了一只叫汤圆的猫", source="inferred"),
        _row("用户住在广州"),
    ]
    facts = _reanchor.select_anchor_facts(rows, max_items=12, max_chars=900)
    assert [f["text"] for f in facts] == ["十二是十一的弟弟"]


def test_select_excludes_sensitive_text():
    rows = [_row("我的银行卡号是6222021234567890", confirmed=True)]
    assert _reanchor.select_anchor_facts(rows, max_items=12, max_chars=900) == []


def test_select_order_rules_then_slots_then_confirmed():
    rows = [
        _row("普通已确认事实", confirmed=True, write_seq=9),
        _row("名字是十二", slot="agent_name", source="inferred", write_seq=1),
        _row("删记忆前必须先问用户", is_rule=True, write_seq=2),
    ]
    facts = _reanchor.select_anchor_facts(rows, max_items=12, max_chars=900)
    assert [f["text"] for f in facts] == [
        "删记忆前必须先问用户", "名字是十二", "普通已确认事实",
    ]


def test_select_caps_items():
    rows = [_row(f"已确认事实第{i}条", confirmed=True, write_seq=i)
        for i in range(15)]
    facts = _reanchor.select_anchor_facts(
        rows, max_items=_reanchor.ANCHOR_MAX_ITEMS,
        max_chars=_reanchor.ANCHOR_MAX_CHARS)
    assert len(facts) == _reanchor.ANCHOR_MAX_ITEMS


def test_select_caps_chars_keeps_first_even_if_oversized():
    big = "长" * 2000
    rows = [
        _row(big, is_rule=True),
        _row("第二条已确认事实", confirmed=True),
    ]
    facts = _reanchor.select_anchor_facts(rows, max_items=12, max_chars=900)
    # 第一条超限也保留（截断），后面的因预算已满不再带。
    assert len(facts) == 1
    assert facts[0]["text"].endswith("…")
    assert len(facts[0]["text"]) <= 901


# ---------------- 纯判定：拼块与回锚节奏 ----------------


def test_build_block_labels_each_fact_confirmed():
    facts = [_row("十二是十一的弟弟", confirmed=True)]
    block = _reanchor.build_anchor_block(facts)
    assert _ANCHOR_TITLE in block
    assert "（已确认）十二是十一的弟弟" in block
    assert "定案" in block
    assert _reanchor.build_anchor_block([]) == ""


def test_build_reanchor_block_is_strengthened_variant():
    facts = [_row("十二是十一的弟弟", confirmed=True)]
    block = _reanchor.build_anchor_block(facts, reanchor=True)
    assert _REANCHOR_TITLE in block
    assert "（已确认）十二是十一的弟弟" in block


def test_should_reanchor_every_eight_turns():
    assert not any(_reanchor.should_reanchor(n) for n in range(1, 8))
    assert _reanchor.should_reanchor(8)
    assert not any(_reanchor.should_reanchor(n) for n in range(9, 16))
    assert _reanchor.should_reanchor(16)
    assert _reanchor.should_reanchor(24)
    assert not _reanchor.should_reanchor(0)
    assert not _reanchor.should_reanchor(None)


def test_turn_number_from_messages():
    assert _reanchor.turn_number_from_messages(None) == 1
    assert _reanchor.turn_number_from_messages([]) == 1
    history = [
        {"role": "user", "content": "一"}, {"role": "assistant", "content": "好"},
        {"role": "user", "content": "二"}, {"role": "assistant", "content": "好"},
    ]
    assert _reanchor.turn_number_from_messages(history) == 3


def test_anchor_block_for_turn_switches_variant():
    rows = [_row("十二是十一的弟弟", confirmed=True)]
    normal = _reanchor.anchor_block_for_turn(rows, 3)
    assert _ANCHOR_TITLE in normal and "【回锚" not in normal
    anchored = _reanchor.anchor_block_for_turn(rows, 8)
    assert _REANCHOR_TITLE in anchored


# ---------------- store 层：只拿已确认的活行 ----------------


def test_store_confirmed_facts_only_confirmed_rows(store):
    store.add_memory_text("十二是十一的弟弟", source="user")
    store.add_memory_text("用户称呼是BOOS", source="user")
    store.add_memory_text("用户喜欢吃冒菜", source="inferred")
    texts = [m["text"] for m in store.confirmed_facts()]
    assert "十二是十一的弟弟" in texts
    assert "用户称呼是BOOS" in texts
    assert "用户喜欢吃冒菜" not in texts


# ---------------- 集成：多轮对话里每轮都带锚 ----------------


class _DirectAnswerProvider:
    """直接回答的假模型（同 step0 口径）：记下每次收到的消息。"""
    
    _LAST = None
    
    def __init__(self, api_key=None, budget_tracker=None, model=None):
        self.model = model or "deepseek-flash"
        self.tool_calls_log = []
        _DirectAnswerProvider._LAST = self
    
    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None,
            thinking=None):
        return {"text": "好的。", "usage": None, "cost_usd": 0.0,
            "model": self.model}
    
    def chat_with_tools(self, messages, tools, thinking="disabled",
            ledger="tool", task_id=None, trace_id=None):
        self.tool_calls_log.append({"messages": list(messages)})
        return {
            "text": "好的。",
            "tool_calls": [],
            "reasoning_content": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                "total_tokens": 15},
            "cost_usd": 0.001,
            "model": self.model,
        }
    
    def chat_messages(self, messages, ledger="chat", task_id=None,
            trace_id=None, thinking=None, model=None,
            max_tokens=None):
        return {"text": "好的。", "usage": None, "cost_usd": 0.0,
            "model": self.model}


class _StubSearchService:
    """占位搜索服务：模型直接回答，搜索一次也不调。"""


class _ProbeStore:
    """包一层真 store：检索与已确认事实走真的，收尾存记忆关掉——
    多轮模拟里每轮的问答不该反过来污染被测的事实集。"""
    
    def __init__(self, inner):
        self._inner = inner
    
    def search(self, prompt, **kwargs):
        return self._inner.search(prompt, **kwargs)
    
    def note_context(self, memories):
        return self._inner.note_context(memories)
    
    def confirmed_facts(self):
        return self._inner.confirmed_facts()
    
    def save_turn(self, recent, turn_id=None):
        return {"pending": True, "results": [], "queued": True,
            "turn_id": turn_id}
    
    def flush_pending(self):
        return None


def _run_turns(store, tmp_path, monkeypatch, n_turns):
    """模拟 n 轮对话，返回每轮实际发给模型的全部消息文本。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setitem(
        main_module.PROVIDER_CLASSES, "deepseek", _DirectAnswerProvider)
    probe = _ProbeStore(store)
    tracker = BudgetTracker(file_path=tmp_path / "budget_usage.json")
    gateway = ToolGateway()
    tasks = TaskStore(file_path=str(tmp_path / "tasks.json"))
    history = []
    sent_per_turn = []
    for i in range(1, n_turns + 1):
        outcome = run_question(
            f"随便聊聊第{i}轮",
            budget_tracker=tracker,
            messages=history,
            memory_store=probe,
            tool_gateway=gateway,
            task_store=tasks,
            search_service=_StubSearchService(),
        )
        assert outcome["mode"] == "real"
        provider = _DirectAnswerProvider._LAST
        calls = provider.tool_calls_log
        assert calls, "模型应当被调用过"
        joined = "\n".join(
            str(m.get("content") or "") for m in calls[-1]["messages"])
        sent_per_turn.append(joined)
        history = history + [
            {"role": "user", "content": f"随便聊聊第{i}轮"},
            {"role": "assistant", "content": "好的。"},
        ]
    return sent_per_turn


def _anchor_zone(sent_text):
    """从发给模型的文本里切出已确认区（块头到下一个空行分隔）。"""
    start = sent_text.find(_REANCHOR_TITLE)
    if start < 0:
        start = sent_text.find(_ANCHOR_TITLE)
    assert start >= 0, "这一轮的提示词里没有已确认区"
    end = sent_text.find("\n\n", start)
    return sent_text[start:] if end < 0 else sent_text[start:end]


def test_every_turn_carries_confirmed_anchor(store, tmp_path, monkeypatch):
    # 实测第 ⑦ 组桩用例：身份定案事实 + 一条未确认的普通记忆。
    store.add_memory_text("十二是十一的弟弟", source="user")
    store.add_memory_text("用户养了一只叫汤圆的猫", source="inferred")
    sent = _run_turns(store, tmp_path, monkeypatch, 17)
    assert len(sent) == 17
    for turn_text in sent:
        zone = _anchor_zone(turn_text)
        assert "（已确认）十二是十一的弟弟" in zone
        # 未确认的普通记忆不许混进已确认区。
        assert "汤圆" not in zone


def test_reanchor_turns_use_strengthened_block(store, tmp_path, monkeypatch):
    store.add_memory_text("十二是十一的弟弟", source="user")
    sent = _run_turns(store, tmp_path, monkeypatch, 17)
    for idx, turn_text in enumerate(sent, start=1):
        zone = _anchor_zone(turn_text)
        if idx in (8, 16):
            assert zone.startswith(_REANCHOR_TITLE), f"第{idx}轮应回锚"
        else:
            assert zone.startswith(_ANCHOR_TITLE), f"第{idx}轮应是常规块"
            assert "【回锚" not in zone


def test_anchor_cap_applies_in_prompt(store, tmp_path, monkeypatch):
    for i in range(15):
        store.add_memory_text(f"已确认的规矩第{i}条内容", source="user")
    sent = _run_turns(store, tmp_path, monkeypatch, 1)
    zone = _anchor_zone(sent[0])
    lines = [ln for ln in zone.splitlines() if ln.startswith("- （已确认）")]
    assert 0 < len(lines) <= _reanchor.ANCHOR_MAX_ITEMS
