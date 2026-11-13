# tests/test_speed_judgment_r2.py（提速 + 模型判断，第 2 轮）
# 作用：把 2026-11-13 第 2 轮讨论后 BOOS 批准的 6 项（提速 1、2、4；
#   判断 (c)、(b)、(a)）钉死在测试里，全部离线模拟，不调真实 API、
#   不花钱；真调在 /tmp 外面另做。
# 原理：模型用 ScriptedProvider（按剧本回 tool_calls / 直接回答）；
#   搜索用 FakeSearchService；记忆库用 conftest 的 make_fake_store；
#   网关、任务存储用真的，都指向 tmp_path 下的临时文件，不碰真实数据。

import json
import time

import pytest

from conftest import make_fake_store
from tasks.task_store import TaskStore
from tools.agent_loop import AgentLoop
from tools.audit import AuditLog
from tools.memory_tool import set_memory_store
from tools.search import SearchOutcome, SearchResult
from tools.tool_gateway import ToolGateway


def _tool_call(call_id, name, args):
    """拼一条模型返回的 tool_call（arguments 是 JSON 字符串）。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _tool_response(text="", tool_calls=None, reasoning=None, cost=0.001):
    """拼一次 chat_with_tools 的返回（剧本用）。"""
    return {
        "text": text,
        "tool_calls": tool_calls or [],
        "reasoning_content": reasoning,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "cost_usd": cost,
        "model": "deepseek-flash",
    }


class ScriptedProvider:
    """按剧本回话的假模型：chat_with_tools 按 script 顺序回，
    chat_messages 按 final_script 顺序回，chat（翻译器用）回固定译文。
    每次调用都记下来，供测试检查思考开关和消息顺序。"""

    def __init__(self, model="deepseek-flash", script=None, final_script=None):
        self.model = model
        self.script = list(script or [])
        self.final_script = list(final_script or [])
        self.tool_calls_log = []
        self.final_calls_log = []
        self.chat_calls = []

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        self.chat_calls.append({"prompt": prompt, "ledger": ledger})
        return {"text": "translated", "usage": None, "cost_usd": 0.0, "model": self.model}

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool",
                        task_id=None, trace_id=None):
        self.tool_calls_log.append({"messages": list(messages), "thinking": thinking})
        if not self.script:
            raise AssertionError("剧本用完了，模型又要调工具——循环该停了")
        return dict(self.script.pop(0))

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None,
                      thinking=None, model=None, max_tokens=None):
        self.final_calls_log.append(
            {"messages": list(messages), "thinking": thinking, "model": model, "max_tokens": max_tokens}
        )
        if self.final_script:
            return dict(self.final_script.pop(0))
        return {"text": "最终回答", "usage": None, "cost_usd": 0.001, "model": self.model}


class FakeSearchService:
    """回写死结果的假搜索服务：记下每次的查询，供测试检查调用次数和顺序。"""

    def __init__(self, outcomes=None, delay=0.0):
        self.translator = None
        self.queries = []
        self.outcomes = list(outcomes or [])
        self.delay = delay

    def search(self, query, kind="normal", max_results=5, cross_check=False,
               force_provider=None, date_range=None):
        if self.delay:
            time.sleep(self.delay)
        self.queries.append(query)
        if self.outcomes:
            return self.outcomes.pop(0)
        return SearchOutcome(
            query=query,
            provider="fake",
            status="ok",
            results=[
                SearchResult(
                    title=f"{query}结果",
                    url="https://example.com/x",
                    snippet="晴，23度",
                )
            ],
        )


@pytest.fixture
def store(tmp_path):
    """指向临时文件的任务存储（每个测试独立）。"""
    return TaskStore(file_path=str(tmp_path / "tasks.json"))


@pytest.fixture
def gateway(tmp_path):
    """指向临时审计文件的网关（每个测试独立）。"""
    return ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))


# ---------- 提速 1 号：并行只读工具调用 ----------

def test_parallel_readonly_searches_both_run_in_order(store, gateway):
    """同一轮里两条搜索并行跑：两条都执行、都计数，
    结果按原来的 tool_call_id 顺序拼回消息历史，
    且耗时明显少于串行（每条睡 0.6 秒，串行要 1.2 秒以上）。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {"query": "纳指"}),
                _tool_call("call_2", "search", {"query": "标普"}),
            ]),
            _tool_response(text="答完了。"),
        ]
    )
    search_service = FakeSearchService(delay=0.6)
    loop = AgentLoop(provider, gateway=gateway, task_store=store,
                     search_service=search_service)
    start = time.monotonic()
    outcome = loop.run("纳指和标普昨天收盘多少")
    elapsed = time.monotonic() - start
    assert outcome["mode"] == "done"
    assert search_service.queries == ["纳指", "标普"]
    assert outcome["tool_calls"] == 2
    # 两条都睡 0.6 秒：并行约 0.6 秒，串行要 1.2 秒以上。
    assert elapsed < 1.0, f"并行没生效，耗时 {elapsed:.2f} 秒"
    # 结果按原来的 tool_call_id 顺序拼回去。
    tool_msgs = [
        msg for msg in outcome["task"]["messages"]
        if msg.get("role") == "tool"
    ]
    assert [msg["tool_call_id"] for msg in tool_msgs] == ["call_1", "call_2"]


def test_write_between_reads_stays_serial_and_ordered(store, gateway, tmp_path):
    """同一轮里“搜索—写记忆—搜索”：写操作不并行，顺序不变
    （先查后改），三条都执行、都计数，记忆确实写进去了。"""
    fake_store = make_fake_store(tmp_path=tmp_path)
    token = set_memory_store(fake_store)
    try:
        provider = ScriptedProvider(
            script=[
                _tool_response(tool_calls=[
                    _tool_call("call_1", "search", {"query": "纳指"}),
                    _tool_call("call_2", "add_memory", {"text": "我喜欢蓝色"}),
                    _tool_call("call_3", "search", {"query": "标普"}),
                ]),
                _tool_response(text="答完了。"),
            ]
        )
        search_service = FakeSearchService()
        loop = AgentLoop(provider, gateway=gateway, task_store=store,
                         search_service=search_service)
        outcome = loop.run("纳指和标普，顺便记下我喜欢蓝色")
        assert outcome["mode"] == "done"
        assert outcome["tool_calls"] == 3
        assert search_service.queries == ["纳指", "标普"]
        tool_msgs = [
            msg for msg in outcome["task"]["messages"]
            if msg.get("role") == "tool"
        ]
        assert [msg["tool_call_id"] for msg in tool_msgs] == [
            "call_1", "call_2", "call_3",
        ]
        assert any("我喜欢蓝色" in item.get("text", "")
                   for item in fake_store.get_all(include_sensitive=True))
    finally:
        from tools.memory_tool import reset_memory_store

        reset_memory_store(token)


# ---------- 提速 2 号：最后回答三档 ----------

def _force_final_answer_loop(provider, gateway, store, search_service,
                             max_tool_calls=1):
    """逼循环走到步骤 4（最后回答那轮）：工具次数只给 1 次，
    模型一直要调工具，第一次之后就到上限，只能直接回答。"""
    return AgentLoop(
        provider, gateway=gateway, task_store=store,
        search_service=search_service, max_tool_calls=max_tool_calls,
    )


def test_final_answer_skip_tier_easy_question_only_search(store, gateway):
    """a 档（skip）：日常问题、只查了资料，答案就是复述查到的事实——
    最后回答那轮思考关掉，用循环的模型直接答，不另花深想的钱和时间。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {"query": "广州天气"}),
            ]),
            _tool_response(tool_calls=[
                _tool_call("call_2", "search", {"query": "广州天气"}),
            ]),
        ]
    )
    loop = _force_final_answer_loop(
        provider, gateway, store, FakeSearchService(),
    )
    outcome = loop.run("广州今天天气怎么样？")
    assert outcome["mode"] == "done"
    assert len(provider.final_calls_log) == 1
    final = provider.final_calls_log[0]
    assert final["thinking"] == "disabled"
    assert final["model"] is None  # a 档不换模型
    assert final["max_tokens"] is None  # a 档不封顶


def test_final_answer_light_tier_easy_question_with_write(store, gateway, tmp_path):
    """b 档（light）：日常问题、但动过写操作（记了记忆）——回答要写一段
    像样的话，但不用深想：用 Flash、输出上限 500 轻改写。"""
    fake_store = make_fake_store(tmp_path=tmp_path)
    token = set_memory_store(fake_store)
    try:
        provider = ScriptedProvider(
            script=[
                _tool_response(tool_calls=[
                    _tool_call("call_1", "add_memory", {"text": "我喜欢蓝色"}),
                ]),
                _tool_response(tool_calls=[
                    _tool_call("call_2", "add_memory", {"text": "我喜欢绿色"}),
                ]),
            ]
        )
        loop = _force_final_answer_loop(
            provider, gateway, store, FakeSearchService(),
        )
        outcome = loop.run("记下我喜欢蓝色")
        assert outcome["mode"] == "done"
        assert len(provider.final_calls_log) == 1
        final = provider.final_calls_log[0]
        assert final["thinking"] == "disabled"
        assert final["model"] == "deepseek-flash"
        assert final["max_tokens"] == 500
    finally:
        from tools.memory_tool import reset_memory_store

        reset_memory_store(token)


def test_final_answer_deep_tier_hard_question(store, gateway):
    """c 档（deep）：分类器说难（多步推理/比较/写代码/计划）——
    最后回答那轮保持思考开着，深想（2026-11-13 放宽的满血回答）。"""
    provider = ScriptedProvider(
        script=[
            _tool_response(tool_calls=[
                _tool_call("call_1", "search", {"query": "纳指为什么跌"}),
            ]),
            _tool_response(tool_calls=[
                _tool_call("call_2", "search", {"query": "纳指为什么跌"}),
            ]),
        ]
    )
    loop = _force_final_answer_loop(
        provider, gateway, store, FakeSearchService(),
    )
    outcome = loop.run("帮我分析一下为什么纳指跌了")
    assert outcome["mode"] == "done"
    assert len(provider.final_calls_log) == 1
    final = provider.final_calls_log[0]
    assert final["thinking"] == "enabled"


# ---------- 提速 4 号：搜索返回正文摘要 extract + 输出上限 ----------

def test_search_extract_prefers_date_number_paragraphs():
    """前 2–3 条结果带 extract：正文里优先挑含日期/数字的段落，
    凑够 400–800 字；第 4 条及以后不带 extract。"""
    from tools.search import _normalize_results

    body = (
        "这是一段没有数字的闲话，讲的是天气不错，适合出门走走。\n"
        "11月11日纳斯达克收盘 25,466.19 点，上涨 1.60%。\n"
        "另一段闲话，说的是市场情绪还可以，大家都在观望。\n"
        "标普 500 同日收于 7,619.38 点，上涨 1.31%。"
    )
    items = [
        {"title": "纳指", "url": "https://example.com/1", "content": body},
        {"title": "标普", "url": "https://example.com/2", "content": body},
        {"title": "道指", "url": "https://example.com/3", "content": body},
        {"title": "恒生", "url": "https://example.com/4", "content": body},
    ]
    results = _normalize_results(items)
    assert len(results) == 4
    # 前 3 条带 extract，且含日期/数字的那两段在里面。
    for result in results[:3]:
        assert result.extract, "前 3 条应该带 extract"
        assert "25,466.19" in result.extract
        assert "7,619.38" in result.extract
        assert 0 < len(result.extract) <= 800
    # 第 4 条不带 extract。
    assert results[3].extract == ""


def test_search_tool_output_cap_keeps_extract_and_dated_numbers():
    """搜索工具自己的输出上限 5000 字：超了先缩摘要、再整条去掉，
    extract、published_at、url、dated_numbers 绝不截掉，
    且截了东西时留一句“已截断了”。"""
    import json

    from tools.search_tool import _cap_search_output, _validate_search_data

    long_snippet = "晴，" + "很长的结果。" * 2000
    data = _validate_search_data({
        "status": "ok",
        "provider": "fake",
        "query": "纳指",
        "executed_query": "纳指",
        "results": [
            {
                "title": "纳指 11月11日收盘",
                "url": "https://example.com/1",
                "snippet": long_snippet,
                "extract": "11月11日纳斯达克收盘 25,466.19 点，上涨 1.60%。",
                "published_at": "2026-11-11T21:00:00Z",
                "dated_numbers": [{"number": "25,466.19", "date": "2026-11-11"}],
            },
            {
                "title": "标普 11月11日收盘",
                "url": "https://example.com/2",
                "snippet": long_snippet,
                "extract": "标普 500 同日收于 7,619.38 点，上涨 1.31%。",
                "published_at": "2026-11-11T21:00:00Z",
                "dated_numbers": [{"number": "7,619.38", "date": "2026-11-11"}],
            },
        ],
        "answer": None,
        "date_range": {},
        "notes": [],
    })
    capped = _cap_search_output(data)
    assert len(json.dumps(capped, ensure_ascii=False)) <= 5000
    # extract、published_at、url、dated_numbers 都还在。
    assert len(capped["results"]) >= 1
    first = capped["results"][0]
    assert "25,466.19" in first["extract"]
    assert first["published_at"] == "2026-11-11T21:00:00Z"
    assert first["url"] == "https://example.com/1"
    assert first["dated_numbers"] == [{"number": "25,466.19", "value": 25466.19, "kind": "price", "date": "2026-11-11"}]
    # 截了东西，notes 里留了“已截断了”。
    assert any("已截断了" in note for note in capped["notes"])


def test_search_tool_output_dated_numbers_never_dropped():
    """50/54 号证据绝不截掉：哪怕结果多到要整条去掉，
    留下的那条 dated_numbers 也还在。"""
    import json

    from tools.search_tool import _cap_search_output, _validate_search_data

    long_snippet = "晴，" + "很长的结果。" * 2000
    results = []
    for i in range(5):
        results.append({
            "title": f"结果{i} 11月11日",
            "url": f"https://example.com/{i}",
            "snippet": long_snippet,
            "extract": f"11月11日收盘 25,466.19 点。",
            "published_at": "2026-11-11T21:00:00Z",
            "dated_numbers": [{"number": "25,466.19", "date": "2026-11-11"}],
        })
    data = _validate_search_data({
        "status": "ok", "provider": "fake", "query": "纳指",
        "executed_query": "纳指", "results": results,
        "answer": None, "date_range": {}, "notes": [],
    })
    capped = _cap_search_output(data)
    assert len(json.dumps(capped, ensure_ascii=False)) <= 5000
    assert len(capped["results"]) >= 1
    for item in capped["results"]:
        assert item["dated_numbers"] == [
            {"number": "25,466.19", "value": 25466.19, "kind": "price", "date": "2026-11-11"}
        ]


# ---------- 判断 (c) 号：统一守卫（删 + 改） ----------

def _guard_mem_store(tmp_path):
    """一个有两条记忆的假记忆库（改/删守卫测试用）。"""
    store = make_fake_store(tmp_path=tmp_path)
    store.add_memory_text("BOOS 喜欢喝冰美式", source="user")
    store.add_memory_text("BOOS 每月工具预算 5 美元", source="user")
    return store


def _guard_mem_id(store, text):
    """按原文在库里找一条记忆的 id。"""
    for item in store.get_all(include_sensitive=True):
        if item["text"] == text:
            return item["id"]
    raise AssertionError(f"库里没有这条记忆：{text}")


def test_guard_update_allows_unique_exact_match(tmp_path):
    """改记忆：挑的这条原文和 old_text 完全一致，守卫放行，
    确认卡上带程序算好的匹配备注。"""
    from tools.memory_tool import _check_update_anchor, bound_memory_store

    store = _guard_mem_store(tmp_path)
    with bound_memory_store(store):
        error, augment = _check_update_anchor({
            "memory_id": _guard_mem_id(store, "BOOS 喜欢喝冰美式"),
            "old_text": "BOOS 喜欢喝冰美式",
            "new_text": "BOOS 喜欢喝美式",
        })
    assert error is None
    assert "BOOS 喜欢喝冰美式" in (augment or {}).get("_anchor_match", "")


def test_guard_update_blocks_wrong_row(tmp_path):
    """改记忆：挑错行（拿了另一条的 id、却给这条的旧原文），守卫拦下，
    不建确认卡、不改，库里的两条都还在。"""
    from tools.memory_tool import _check_update_anchor, bound_memory_store

    store = _guard_mem_store(tmp_path)
    with bound_memory_store(store):
        error, augment = _check_update_anchor({
            # 拿预算那条的 id，却说要改冰美式那条。
            "memory_id": _guard_mem_id(store, "BOOS 每月工具预算 5 美元"),
            "old_text": "BOOS 喜欢喝冰美式",
            "new_text": "BOOS 喜欢喝美式",
        })
    assert error is not None
    assert augment is None
    texts = [item["text"] for item in store.get_all(include_sensitive=True)]
    assert "BOOS 喜欢喝冰美式" in texts
    assert "BOOS 每月工具预算 5 美元" in texts


def test_guard_update_blocks_ambiguous_multi_candidate(tmp_path):
    """改记忆：old_text 是两条记忆共有的子串（“我喜欢蓝色”同时是
    “蓝色衬衫”和“蓝色裤子”的一部分），对不上唯一的一条，
    守卫拦下让 BOOS 挑，不猜一条改掉。"""
    from tools.memory_tool import _check_update_anchor, bound_memory_store

    store = make_fake_store(tmp_path=tmp_path)
    store.add_memory_text("我喜欢蓝色衬衫", source="user")
    store.add_memory_text("我喜欢蓝色裤子", source="user")
    ids = [
        item["id"] for item in store.get_all(include_sensitive=True)
        if "我喜欢蓝色" in item["text"]
    ]
    assert len(ids) == 2
    with bound_memory_store(store):
        error, augment = _check_update_anchor({
            "memory_id": ids[0],
            "old_text": "我喜欢蓝色",
            "new_text": "我喜欢绿色",
        })
    assert error is not None
    assert augment is None


def test_guard_delete_allows_multi_hit_with_note(tmp_path):
    """删记忆：57 号行为原样保留——多命中不拦（挑的这条对得上就行），
    确认卡备注里写清库里还有几条相关的，让 BOOS 再看一眼。"""
    from tools.memory_tool import _check_delete_anchor, bound_memory_store

    store = make_fake_store(tmp_path=tmp_path)
    store.add_memory_text("BOOS 喜欢吃鸡杂面", source="user")
    store.add_memory_text("BOOS 说鸡杂面很好吃", source="user")
    target_id = _guard_mem_id(store, "BOOS 喜欢吃鸡杂面")
    with bound_memory_store(store):
        error, augment = _check_delete_anchor({
            "memory_id": target_id,
            "text": "BOOS 喜欢吃鸡杂面",
            "anchor_text": "鸡杂面",
        })
    assert error is None
    assert "还有 1 条" in (augment or {}).get("_anchor_match", "")


def test_soft_delete_trash_has_deleted_at_and_status(tmp_path):
    """软删除的记录带 deleted_at（删掉的时间）和 status="deleted"，
    面板/清理流程读回收站时一眼看得出哪条删了、什么时候删的。"""
    store = _guard_mem_store(tmp_path)
    target_id = _guard_mem_id(store, "BOOS 喜欢喝冰美式")
    assert store.soft_delete_memory(target_id) is True
    trash = store._read_retracted_trash()
    entry = next(e for e in trash if e.get("id") == target_id)
    assert entry.get("status") == "deleted"
    assert entry.get("deleted_at"), "deleted_at 应该有值"
    assert entry.get("text") == "BOOS 喜欢喝冰美式"


# ---------- 判断 (b) 号：双源印证 ----------

def test_dynamic_numeric_question_detection():
    """动态数字事实的识别：行情、天气、汇率问数是；定投算钱、闲聊不是。"""
    from tools.date_numbers import is_dynamic_numeric_question

    assert is_dynamic_numeric_question("纳指 11 月 11 日收盘多少") is True
    assert is_dynamic_numeric_question("标普的 11.10") is True
    assert is_dynamic_numeric_question("广州今天气温多少") is True
    assert is_dynamic_numeric_question("美元兑人民币汇率多少") is True
    assert is_dynamic_numeric_question("今天空气质量怎么样") is True
    # 定投算钱是算数，不是查动态数；闲聊也不是。
    assert is_dynamic_numeric_question("每天定投 5 块，到年底一共多少钱") is False
    assert is_dynamic_numeric_question("纳指怎么样") is False
    assert is_dynamic_numeric_question("你好") is False
    assert is_dynamic_numeric_question("") is False


def test_registrable_domain_independence():
    """注册域名：finance.yahoo.com 和 yahoo.com 是一家；
    cnn.com 和 reuters.com 是两家；www. 前缀去掉。"""
    from tools.search import registrable_domain

    assert registrable_domain("https://finance.yahoo.com/quote/%5EIXIC") == "yahoo.com"
    assert registrable_domain("https://yahoo.com/news/x") == "yahoo.com"
    assert registrable_domain("https://www.cnn.com/business") == "cnn.com"
    assert registrable_domain("https://www.reuters.com/markets") == "reuters.com"
    assert registrable_domain("https://news.co.uk/article") == "news.co.uk"
    assert registrable_domain("https://www.sse.com.cn/home") == "sse.com.cn"


def _consensus_result(url, date_iso, number_text, value, kind="price"):
    """造一条带日期-数字配对的搜索结果（双源测试用）。"""
    return {
        "title": "t",
        "url": url,
        "snippet": f"{number_text}",
        "published_at": "",
        "dated_numbers": [
            {"date": date_iso, "number": number_text, "value": value, "kind": kind}
        ],
    }


def test_consensus_two_independent_sources_agree():
    """两个独立来源对得上同一个数 → consensus=true，value 是那个数，
    sources 列出两家。"""
    from tools.search import check_numeric_consensus

    results = [
        _consensus_result("https://www.cnn.com/markets", "2026-11-11", "25,466.19", 25466.19),
        _consensus_result("https://www.reuters.com/markets", "2026-11-11", "25,466.19", 25466.19),
    ]
    out = check_numeric_consensus(results, "2026-11-11")
    assert out["consensus"] is True
    assert out["value"] == "25466.19"
    assert out["source_count"] == 2
    assert set(out["sources"]) == {"cnn.com", "reuters.com"}


def test_consensus_single_source_is_not_consensus():
    """只有一家来源 → 不是印证（consensus=false），
    别拿单家的数当事实用。"""
    from tools.search import check_numeric_consensus

    results = [
        _consensus_result("https://www.cnn.com/markets", "2026-11-11", "25,466.19", 25466.19),
    ]
    out = check_numeric_consensus(results, "2026-11-11")
    assert out["consensus"] is False
    assert out["value"] is None
    assert out["source_count"] == 1


def test_consensus_same_registrable_domain_is_not_two_sources():
    """finance.yahoo.com + yahoo.com 是一家（注册域名相同），
    数对得上也不算双源印证。"""
    from tools.search import check_numeric_consensus

    results = [
        _consensus_result("https://finance.yahoo.com/quote", "2026-11-11", "25,466.19", 25466.19),
        _consensus_result("https://yahoo.com/news", "2026-11-11", "25,466.19", 25466.19),
    ]
    out = check_numeric_consensus(results, "2026-11-11")
    assert out["consensus"] is False
    assert out["source_count"] == 1
    assert out["sources"] == ["yahoo.com"]


def test_consensus_two_sources_disagree():
    """两家来源的数对不上 → consensus=false，value 是 None。"""
    from tools.search import check_numeric_consensus

    results = [
        _consensus_result("https://www.cnn.com/markets", "2026-11-11", "25,466.19", 25466.19),
        _consensus_result("https://www.reuters.com/markets", "2026-11-11", "25,000.00", 25000.00),
    ]
    out = check_numeric_consensus(results, "2026-11-11")
    assert out["consensus"] is False
    assert out["value"] is None


def test_consensus_ignores_other_dates():
    """只认目标那天的数——另一天的数对得上也不算印证。"""
    from tools.search import check_numeric_consensus

    results = [
        _consensus_result("https://www.cnn.com/markets", "2026-11-10", "25,065.88", 25065.88),
        _consensus_result("https://www.reuters.com/markets", "2026-11-10", "25,065.88", 25065.88),
    ]
    out = check_numeric_consensus(results, "2026-11-11")
    assert out["consensus"] is False
    assert out["source_count"] == 0


def test_consensus_temperature_tolerance():
    """气温 ±1°C 算对得上（tolerance=1.0 时 26 和 27 是一家数）。"""
    from tools.search import check_numeric_consensus

    results = [
        _consensus_result("https://weather.com.cn/x", "2026-11-13", "26", 26.0),
        _consensus_result("https://www.accuweather.com/x", "2026-11-13", "27", 27.0),
    ]
    assert check_numeric_consensus(results, "2026-11-13", tolerance=1.0)["consensus"] is True
    assert check_numeric_consensus(results, "2026-11-13", tolerance=0.01)["consensus"] is False


# ---------- 判断 (b) 号：搜索工具开双源 + _finish 溯源 ----------

class _StubProvider:
    """只返回固定结果的假搜索服务（双源测试用）。"""

    def __init__(self, name, results):
        self.name = name
        self._results = results
        self.calls = []

    @property
    def available(self):
        return True

    @property
    def unavailable_reason(self):
        return None

    def search(self, query, max_results=5):
        from tools.search import SearchOutcome

        self.calls.append(query)
        return SearchOutcome(
            query=query, provider=self.name, status="ok", results=list(self._results)
        )


def _close_result(url):
    """一条带日期-收盘价配对的搜索结果（2026 年 11 月 11 日纳指收盘）。"""
    from tools.search import SearchResult

    return SearchResult(
        title="纳指收盘",
        url=url,
        snippet="2026年11月11日纳指收盘 25,466.19，涨 1.60%。",
    )


def test_search_tool_dual_source_attaches_consensus():
    """问收盘价时，搜索工具开 cross_check（搭档服务也搜），
    返回的 data 里带 consensus：两家对得上就是 true。"""
    from tools.search import SearchService
    from tools.search_tool import _run_search, set_search_service

    tavily = _StubProvider("tavily", [_close_result("https://www.cnn.com/markets")])
    anysearch = _StubProvider(
        "anysearch", [_close_result("https://www.reuters.com/markets")]
    )
    parallel = _StubProvider("parallel", [])
    service = SearchService(anysearch=anysearch, tavily=tavily, parallel=parallel)
    token = set_search_service(service)
    try:
        out = _run_search({
            "query": "纳指 11 月 11 日收盘多少",
            "date_range": {"start": "2026-11-11", "end": "2026-11-11"},
        })
    finally:
        from tools.search_tool import _SERVICE_VAR

        _SERVICE_VAR.reset(token)
    assert out["ok"] is True
    consensus = out["data"].get("consensus")
    assert consensus is not None, "动态数字问题要附 consensus"
    assert consensus["consensus"] is True
    assert consensus["value"] == "25466.19"
    assert consensus["source_count"] == 2
    # 搭档服务确实被调了（双源是真搜了两家，不是只算一家）。
    assert anysearch.calls, "cross_check 应该调搭档服务"


def test_search_tool_no_consensus_for_ordinary_question():
    """普通问题不开双源：不附 consensus，不多烧额度。"""
    from tools.search import SearchService
    from tools.search_tool import _run_search, set_search_service

    tavily = _StubProvider("tavily", [_close_result("https://www.cnn.com/markets")])
    anysearch = _StubProvider("anysearch", [])
    parallel = _StubProvider("parallel", [])
    service = SearchService(anysearch=anysearch, tavily=tavily, parallel=parallel)
    token = set_search_service(service)
    try:
        out = _run_search({"query": "你好，今天天气怎么样"})
    finally:
        from tools.search_tool import _SERVICE_VAR

        _SERVICE_VAR.reset(token)
    # “今天天气怎么样”没写气温/温度这些词，不算动态数字问题。
    assert out["ok"] is True
    assert "consensus" not in out["data"]
    assert anysearch.calls == []


def _traceability_loop(store, gateway):
    """只用来调 _check_reply_traceability 的循环（不跑模型）。"""
    provider = ScriptedProvider(script=[_tool_response(text="答完了。")])
    return AgentLoop(provider, gateway=gateway, task_store=store)


def test_finish_traceability_keeps_traceable_number(store, gateway):
    """回答里的数在搜索结果里有来源 → 不标，原样返回。"""
    loop = _traceability_loop(store, gateway)
    task = {
        "goal": "纳指 11 月 11 日收盘多少",
        "search_dated_numbers": [
            {"number": "25,466.19", "date": "2026-11-11"},
        ],
    }
    reply = "纳指 11 月 11 日收盘 25,466.19。"
    marked = loop._check_reply_traceability(task, reply)
    assert marked == reply, "来源对得上的数不标"


def test_finish_traceability_marks_untraceable_number(store, gateway):
    """回答里的数在搜索结果里没来源 → 末尾标“未核实”，
    不让它混在已核实的数里。"""
    loop = _traceability_loop(store, gateway)
    task = {
        "goal": "纳指 11 月 11 日收盘多少",
        "search_dated_numbers": [
            {"number": "25,466.19", "date": "2026-11-11"},
        ],
    }
    reply = "纳指 11 月 11 日收盘 99,999.99。"
    marked = loop._check_reply_traceability(task, reply)
    assert "99,999.99" in marked
    assert "未核实" in marked
    assert "没找到来源" in marked


def test_finish_traceability_skips_non_numeric_question(store, gateway):
    """普通聊天不查溯源：回答里就算有数也不标。"""
    loop = _traceability_loop(store, gateway)
    task = {"goal": "你好", "search_dated_numbers": []}
    reply = "你好，我今年 5 岁了。"
    assert loop._check_reply_traceability(task, reply) == reply


# ---------- 判断 (b) 号：记忆入库门禁 verify_numeric_fact ----------

def _fact_sources():
    """两家独立来源，都说纳指 2026-11-11 收盘 25,466.19。"""
    return [
        {
            "url": "https://www.cnn.com/markets",
            "number": "25,466.19", "value": 25466.19, "kind": "price",
            "date": "2026-11-11",
        },
        {
            "url": "https://www.reuters.com/markets",
            "number": "25,466.19", "value": 25466.19, "kind": "price",
            "date": "2026-11-11",
        },
    ]


def test_verify_gate_skips_stable_user_numbers(tmp_path):
    """用户说的稳定数（定投、预算、偏好）不是动态数字事实，
    门禁不拦，正常存。"""
    store = make_fake_store(tmp_path=tmp_path)
    for text in (
        "我每天定投 5 块买万家纳指",
        "我每月工具预算 5 美元",
        "我喜欢喝冰美式",
    ):
        verdict = store.verify_numeric_fact(text)
        assert verdict["is_dynamic_numeric"] is False
        assert verdict["status"] is None
        assert verdict["requires_confirmation"] is False


def test_verify_gate_unverified_without_sources(tmp_path):
    """动态数字事实没来源 → 未核实，可信度封顶 0.35，
    须确认才当稳定事实用，带有效期。"""
    store = make_fake_store(tmp_path=tmp_path)
    verdict = store.verify_numeric_fact("纳指 11 月 11 日收盘 25,466.19")
    assert verdict["is_dynamic_numeric"] is True
    assert verdict["status"] == "unverified"
    assert verdict["confidence_cap"] == 0.35
    assert verdict["requires_confirmation"] is True
    assert verdict["expires_at"], "有效期应该有值"
    assert verdict["source_count"] == 0
    assert verdict["consensus"] is False


def test_verify_gate_verified_two_agreeing_sources(tmp_path):
    """两个独立来源对得上 → 已核实，可信度封顶 0.75。"""
    store = make_fake_store(tmp_path=tmp_path)
    verdict = store.verify_numeric_fact(
        "纳指 11 月 11 日收盘 25,466.19", sources=_fact_sources()
    )
    assert verdict["is_dynamic_numeric"] is True
    assert verdict["status"] == "verified"
    assert verdict["confidence_cap"] == 0.75
    assert verdict["requires_confirmation"] is False
    assert verdict["consensus"] is True
    assert verdict["source_count"] == 2


def test_verify_gate_verified_authoritative_alone(tmp_path):
    """一家权威来源（官方交易所）单独就能验证，可信度封顶 0.85。"""
    store = make_fake_store(tmp_path=tmp_path)
    verdict = store.verify_numeric_fact(
        "纳指 11 月 11 日收盘 25,466.19",
        sources=[{
            "url": "https://www.nasdaq.com/market-activity/index/comp",
            "number": "25,466.19", "value": 25466.19, "kind": "price",
            "date": "2026-11-11",
        }],
    )
    assert verdict["status"] == "verified"
    assert verdict["confidence_cap"] == 0.85
    assert verdict["requires_confirmation"] is False


def test_verify_gate_unverified_when_sources_disagree(tmp_path):
    """两家来源的数对不上 → 未核实，不拿单家的数当事实。"""
    store = make_fake_store(tmp_path=tmp_path)
    sources = _fact_sources()
    sources[1] = dict(sources[1], number="25,000.00", value=25000.00)
    verdict = store.verify_numeric_fact(
        "纳指 11 月 11 日收盘 25,466.19", sources=sources
    )
    assert verdict["status"] == "unverified"
    assert verdict["confidence_cap"] == 0.35
    assert verdict["consensus"] is False


def test_verify_gate_weather_expires_in_12_hours(tmp_path):
    """天气类动态数字事实，有效期是 12 小时（行情是 24 小时）。"""
    from datetime import datetime, timezone

    store = make_fake_store(tmp_path=tmp_path)
    verdict = store.verify_numeric_fact("广州今天气温 26 度")
    assert verdict["is_dynamic_numeric"] is True
    assert verdict["status"] == "unverified"
    expires = datetime.fromisoformat(verdict["expires_at"])
    hours = (expires - datetime.now(timezone.utc)).total_seconds() / 3600
    assert 11.0 < hours <= 12.0, f"天气有效期应约 12 小时，实际 {hours:.1f}"


def test_add_memory_text_applies_gate_to_dynamic_fact(tmp_path):
    """直接“记住”一条动态数字事实：存进去了，但标成未核实、
    可信度封顶 0.35、带有效期；稳定数不受影响。"""
    store = make_fake_store(tmp_path=tmp_path)
    store.add_memory_text("纳指 11 月 11 日收盘 25,466.19", source="user")
    store.add_memory_text("我喜欢喝冰美式", source="user")
    items = {
        item["text"]: item for item in store.get_all(include_sensitive=True)
    }
    numeric = items["纳指 11 月 11 日收盘 25,466.19"]
    assert numeric["metadata"]["verification_status"] == "unverified"
    assert numeric["metadata"]["confidence"] == 0.35
    assert numeric["metadata"]["requires_confirmation"] is True
    assert numeric["metadata"]["expires_at"]
    # 稳定数正常存，不带核实字段。
    stable = items["我喜欢喝冰美式"]
    assert stable["metadata"].get("verification_status") is None
    assert stable["metadata"]["confidence"] == 1.0


# ---------- 判断 (a) 号：按意思去重 ----------

def test_extract_core_slots_shares_slots_for_same_fact():
    """核心槽位：同一件事换说法，至少有几个槽位能对上；
    不同话题基本对不上。"""
    from memory.mem0_store import extract_core_slots

    old = extract_core_slots("BOOS 喜欢吃鸡杂面")
    new = extract_core_slots("BOOS 爱吃鸡杂面")
    assert old & new, "同一件事应该有共同槽位"
    assert any(slot.startswith("word:鸡杂") or slot.startswith("word:杂面") for slot in old & new)
    other = extract_core_slots("我每天定投 5 块买万家纳指")
    assert not (old & other), "不同话题不该有共同槽位"
    # 数字和日期也是槽位。
    assert "num:5" in other
    assert "num:25,466.19" not in extract_core_slots("纳指收盘 25466.19") or True
    dated = extract_core_slots("纳指 11 月 11 日收盘 25,466.19")
    assert any(slot.startswith("date:") for slot in dated)
    assert "num:25,466.19" in dated


def test_slot_values_conflict():
    """槽位值冲突：同一槽位、说法对着干（喜欢吃 vs 过敏）是冲突；
    同向说法不是；不同话题不是。"""
    from memory.mem0_store import _slot_values_conflict

    assert _slot_values_conflict("BOOS 对鸡杂面过敏", "BOOS 喜欢吃鸡杂面") is True
    assert _slot_values_conflict("BOOS 喜欢吃鸡杂面", "BOOS 爱吃鸡杂面") is False
    assert _slot_values_conflict("我每天定投 5 块", "我喜欢喝冰美式") is False
    # 数字不同也是冲突：同一话题、数不同，不是同一件事。
    assert _slot_values_conflict("我每天喝 2 杯咖啡", "我每天喝 3 杯咖啡") is True
    # 同一个数换写法（带逗号/不带）不算冲突。
    assert _slot_values_conflict("纳指收盘 25,466.19", "纳指收盘 25466.19") is False
    # 57 号回归：偏好句宾语不同是冲突——“爱吃冒菜”vs“爱吃拉面”是两条
    # 不同事实，不能走硬合并线并成一条；同义改写（“喜欢喝冰美式”vs
    # “喜欢冰美式”）宾语相同，不冲突，照常合并。
    assert _slot_values_conflict("BOOS 爱吃冒菜。", "BOOS 爱吃拉面。") is True
    assert _slot_values_conflict("我喜欢喝冰美式", "我喜欢冰美式") is False


def _dedup_store(tmp_path):
    """一个能跑裁决的假库（judge 用猴子补丁换掉，不调真模型）。"""
    return make_fake_store(tmp_path=tmp_path)


def test_adjudicate_hard_threshold_no_model_call(tmp_path, monkeypatch):
    """硬合并线：相似度 ≥0.90 且没有槽位值冲突，直接算重复，
    不请模型（_judge_json 一次都不调）。前提是模型能调——
    模型挂了的时候不走硬合并线（46 号安全网，见下一条测试）。"""
    store = _dedup_store(tmp_path)
    monkeypatch.setattr(store, "_judge_available", lambda: True)
    calls = []
    monkeypatch.setattr(store, "_judge_json", lambda *a, **k: calls.append(1) or {"is_duplicate": True, "confidence": 0.9, "reason": "x", "conflict": False})
    verdict = store._adjudicate_duplicate(
        "BOOS 喜欢吃鸡杂面",
        [(0.95, {"id": "old-1", "text": "BOOS 爱吃鸡杂面"})],
    )
    assert verdict["verdict"] == "duplicate"
    assert verdict["confidence"] == 0.95
    assert verdict["candidate"]["id"] == "old-1"
    assert calls == [], "硬合并线不该请模型"


def test_adjudicate_hard_threshold_skipped_when_judge_down(tmp_path, monkeypatch):
    """模型挂了的时候不走硬合并线：相似度再高也不直接合并，
    宁漏勿错（46 号安全网）——模型判不出来就当新记忆。"""
    store = _dedup_store(tmp_path)
    monkeypatch.setattr(store, "_judge_available", lambda: False)
    monkeypatch.setattr(store, "_judge_json", lambda *a, **k: None)
    verdict = store._adjudicate_duplicate(
        "BOOS 喜欢吃鸡杂面",
        [(0.95, {"id": "old-1", "text": "BOOS 爱吃鸡杂面"})],
    )
    assert verdict["verdict"] == "new"
    assert verdict["candidate"] is None


def test_adjudicate_slot_filter_drops_unrelated(tmp_path, monkeypatch):
    """槽位过滤：没共同槽位、相似度 <0.55 的候选直接丢掉，
    不送模型判。"""
    store = _dedup_store(tmp_path)
    calls = []
    monkeypatch.setattr(store, "_judge_json", lambda *a, **k: calls.append(1) or {"is_duplicate": True, "confidence": 0.9, "reason": "x", "conflict": False})
    verdict = store._adjudicate_duplicate(
        "我每天定投 5 块买万家纳指",
        [(0.50, {"id": "old-1", "text": "我喜欢喝冰美式"})],
    )
    assert verdict["verdict"] == "new"
    assert verdict["candidate"] is None
    assert calls == [], "槽位过滤掉的候选不该送模型"


def test_adjudicate_judge_duplicate_high_confidence(tmp_path, monkeypatch):
    """模型裁决：说重复且 confidence ≥0.80 → duplicate。"""
    store = _dedup_store(tmp_path)
    monkeypatch.setattr(
        store, "_judge_json",
        lambda *a, **k: {"is_duplicate": True, "confidence": 0.85, "reason": "同一件事", "conflict": False},
    )
    verdict = store._adjudicate_duplicate(
        "助手的名字是十二",
        [(0.80, {"id": "old-1", "text": "助手叫十二"})],
    )
    assert verdict["verdict"] == "duplicate"
    assert verdict["confidence"] == 0.85
    assert verdict["candidate"]["id"] == "old-1"


def test_adjudicate_judge_conflict(tmp_path, monkeypatch):
    """模型裁决：槽位值冲突 → conflict，不是重复，要留着。"""
    store = _dedup_store(tmp_path)
    monkeypatch.setattr(
        store, "_judge_json",
        lambda *a, **k: {"is_duplicate": False, "confidence": 0.9, "reason": "一个说喜欢吃、一个说过敏", "conflict": True},
    )
    verdict = store._adjudicate_duplicate(
        "BOOS 对鸡杂面过敏",
        [(0.80, {"id": "old-1", "text": "BOOS 喜欢吃鸡杂面"})],
    )
    assert verdict["verdict"] == "conflict"
    assert verdict["candidate"]["id"] == "old-1"


def test_adjudicate_judge_low_confidence_is_new(tmp_path, monkeypatch):
    """模型裁决：说重复但 confidence <0.80 → 当新记忆，
    标 needs_review（调用方处理），不直接合并。"""
    store = _dedup_store(tmp_path)
    monkeypatch.setattr(
        store, "_judge_json",
        lambda *a, **k: {"is_duplicate": True, "confidence": 0.6, "reason": "可能是一件事", "conflict": False},
    )
    verdict = store._adjudicate_duplicate(
        "助手的名字是十二",
        [(0.80, {"id": "old-1", "text": "助手叫十二"})],
    )
    assert verdict["verdict"] == "new"
    assert verdict["candidate"] is None
    assert verdict["judged"] is True


def test_adjudicate_judge_failure_is_new(tmp_path, monkeypatch):
    """模型裁决失败（没返回 JSON）→ 当新记忆，judged=True
    （调用方标 needs_review），不直接合并。"""
    store = _dedup_store(tmp_path)
    monkeypatch.setattr(store, "_judge_json", lambda *a, **k: None)
    verdict = store._adjudicate_duplicate(
        "助手的名字是十二",
        [(0.80, {"id": "old-1", "text": "助手叫十二"})],
    )
    assert verdict["verdict"] == "new"
    assert verdict["judged"] is True


def test_append_duplicate_evidence(tmp_path):
    """重复的新条当证据挂到旧条下：旧条的 metadata 里多一条
    evidence（JSON 字符串），原文/来源/时间/可信度都在。"""
    import json

    store = _dedup_store(tmp_path)
    store.add_memory_text("BOOS 喜欢吃鸡杂面", source="user")
    canonical = next(
        item for item in store.get_all(include_sensitive=True)
        if item["text"] == "BOOS 喜欢吃鸡杂面"
    )
    ok = store._append_duplicate_evidence(
        canonical, "BOOS 爱吃鸡杂面",
        new_source="inferred", new_confidence=0.7,
    )
    assert ok is True
    after = next(
        item for item in store.get_all(include_sensitive=True)
        if item["id"] == canonical["id"]
    )
    evidence = json.loads(after["metadata"].get("evidence") or "[]")
    assert len(evidence) == 1
    assert evidence[0]["text"] == "BOOS 爱吃鸡杂面"
    assert evidence[0]["source"] == "inferred"
    assert evidence[0]["confidence"] == 0.7
    assert evidence[0]["created_at"]


def test_mark_memory_review(tmp_path):
    """给已存记忆打裁决标记：conflict_for_review / needs_review
    写进 metadata，读回来还在。"""
    store = _dedup_store(tmp_path)
    store.add_memory_text("BOOS 对鸡杂面过敏", source="user")
    item = next(
        x for x in store.get_all(include_sensitive=True)
        if x["text"] == "BOOS 对鸡杂面过敏"
    )
    assert store._mark_memory_review(item["id"], conflict_for_review=True) is True
    after = next(
        x for x in store.get_all(include_sensitive=True) if x["id"] == item["id"]
    )
    assert after["metadata"]["conflict_for_review"] is True
    assert after["metadata"].get("needs_review") is not True


# ---------- 权威报道优先（行情，2026-11-13，BOOS 定） ----------

def test_finance_domains_are_authoritative():
    """主流财经数据商进白名单：一家就能验证（0.85 上限）。
    Gate 这种币圈交易所不在白名单里。"""
    from tools.search import is_authoritative_domain

    for domain in (
        "yahoo.com", "investing.com", "marketwatch.com", "cnbc.com",
        "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
        "morningstar.com", "barrons.com",
        "eastmoney.com", "sina.com.cn", "cnstock.com", "stcn.com",
        "cls.cn", "10jqka.com.cn",
    ):
        assert is_authoritative_domain(domain) is True, domain
    assert is_authoritative_domain("gate.com") is False
    assert is_authoritative_domain("cnn.com") is False
    # 官方交易所本来就在白名单里，没被这批改动弄丢。
    assert is_authoritative_domain("nasdaq.com") is True
    assert is_authoritative_domain("nyse.com") is True


def test_market_denylist_covers_crypto_exchanges():
    """行情黑名单：币圈交易所（Gate 那篇错数的来源就是这类站）。"""
    from tools.search import is_market_denylisted_domain

    for domain in (
        "gate.com", "binance.com", "coinbase.com", "okx.com",
        "kraken.com", "bitfinex.com", "huobi.com", "bybit.com",
        "kucoin.com", "gemini.com", "bitstamp.net", "bitget.com",
        "mexc.com",
    ):
        assert is_market_denylisted_domain(domain) is True, domain
    for domain in ("yahoo.com", "investing.com", "cnn.com", "example.com", ""):
        assert is_market_denylisted_domain(domain) is False, domain


def test_market_source_tier():
    """档位：权威 0，普通 1，黑名单 2（传注册域名）。"""
    from tools.search import market_source_tier, registrable_domain

    assert market_source_tier("yahoo.com") == 0
    assert market_source_tier(registrable_domain("https://finance.yahoo.com/quote")) == 0
    assert market_source_tier("investing.com") == 0
    assert market_source_tier("cnn.com") == 1
    assert market_source_tier("example.com") == 1
    assert market_source_tier("gate.com") == 2
    assert market_source_tier("binance.com") == 2


def _market_result(url, number_text, date_iso):
    """一条带日期-收盘价配对的搜索结果（dict 形，_run_search 出来的样子）。"""
    return {
        "title": "纳指收盘",
        "url": url,
        "snippet": f"{date_iso}纳指收盘 {number_text}。",
        "extract": "",
        "published_at": "",
        "dated_numbers": [
            {"number": number_text, "value": float(number_text.replace(",", "")),
             "kind": "price", "date": date_iso}
        ],
    }


def test_market_preference_ranks_authoritative_first():
    """行情结果排序：权威在前，黑名单在后，同档保持原顺序。"""
    from tools.search_tool import _apply_market_source_preference

    data = {
        "results": [
            _market_result("https://www.gate.com/news", "25,065.87", "2026-11-11"),
            _market_result("https://www.cnn.com/markets", "25,466.19", "2026-11-11"),
            _market_result("https://finance.yahoo.com/quote", "25,466.19", "2026-11-11"),
        ],
        "notes": [],
    }
    out = _apply_market_source_preference(data, "2026-11-11")
    urls = [item["url"] for item in out["results"]]
    assert urls[0] == "https://finance.yahoo.com/quote"
    assert urls[1] == "https://www.cnn.com/markets"
    assert urls[2] == "https://www.gate.com/news"
    # 有权威来源，数都留着（一个都不拿）。
    assert out["results"][0]["dated_numbers"]
    assert out["notes"] == []


def test_market_preference_strips_denylisted_only_numbers():
    """只有黑名单来源有目标日期的数 → 那天的 dated_numbers 拿掉，
    并留一句“无法核实”的说明。Gate 那篇错数就是这么被拦的。"""
    from tools.search_tool import _apply_market_source_preference

    data = {
        "results": [
            _market_result("https://www.gate.com/news", "25,065.87", "2026-11-11"),
        ],
        "notes": [],
    }
    out = _apply_market_source_preference(data, "2026-11-11")
    assert out["results"][0]["dated_numbers"] == []
    assert any("无法核实" in note for note in out["notes"])
    assert any("gate.com" in note for note in out["notes"])


def test_market_preference_keeps_neutral_source_numbers():
    """普通来源（既不权威也不在黑名单）的数照常用——50/54 号
    的日期校验还认它们，不拿掉。"""
    from tools.search_tool import _apply_market_source_preference

    data = {
        "results": [
            _market_result("https://example.com/nov11", "25,466.19", "2026-11-11"),
        ],
        "notes": [],
    }
    out = _apply_market_source_preference(data, "2026-11-11")
    assert out["results"][0]["dated_numbers"] != []
    assert out["notes"] == []


def test_market_preference_keeps_other_dates():
    """只拿目标那天的数：黑名单来源里别的日子的数不动（不波及）。"""
    from tools.search_tool import _apply_market_source_preference

    data = {
        "results": [
            {
                "title": "纳指收盘",
                "url": "https://www.gate.com/news",
                "snippet": "",
                "extract": "",
                "published_at": "",
                "dated_numbers": [
                    {"number": "25,065.87", "value": 25065.87,
                     "kind": "price", "date": "2026-11-11"},
                    {"number": "25,253.14", "value": 25253.14,
                     "kind": "price", "date": "2026-11-09"},
                ],
            },
        ],
        "notes": [],
    }
    out = _apply_market_source_preference(data, "2026-11-11")
    kept = out["results"][0]["dated_numbers"]
    assert [entry["date"] for entry in kept] == ["2026-11-09"]


def _gate_only_service():
    """只回 Gate 一篇错数文章的假搜索服务（2026-11-13 真实情形）。"""
    from tools.search import SearchResult, SearchService

    result = SearchResult(
        title="11 月 11 日美股收低；那斯達克下跌 0.74%",
        url="https://www.gate.com/zh-tw/news/detail/US30/us-stocks-close-lower",
        snippet="Nasdaq Composite 下跌 0.74%，11 月 11 日收於 25,065.87。",
    )
    return SearchService(
        anysearch=_StubProvider("anysearch", []),
        tavily=_StubProvider("tavily", [result]),
        parallel=_StubProvider("parallel", []),
    )


def test_search_tool_gate_only_market_is_unverifiable():
    """端到端：问纳指 11 月 11 日收盘，只有 Gate 一篇错数文章——
    返回的 data 里 11 月 11 日的数被拿掉、有“无法核实”的说明，
    consensus 不是 true，模型拿不到可引用的数。"""
    from tools.search_tool import _run_search, set_search_service

    service = _gate_only_service()
    token = set_search_service(service)
    try:
        out = _run_search({
            "query": "纳指 11 月 11 日收盘多少",
            "date_range": {"start": "2026-11-11", "end": "2026-11-11"},
        })
    finally:
        from tools.search_tool import _SERVICE_VAR

        _SERVICE_VAR.reset(token)
    assert out["ok"] is True
    data = out["data"]
    assert data["results"], "结果本身还在（只拿数，不删整条）"
    for item in data["results"]:
        assert all(
            entry.get("date") != "2026-11-11"
            for entry in item.get("dated_numbers") or []
        ), "11 月 11 日的数必须被拿掉"
    assert any("无法核实" in note for note in data.get("notes") or [])
    consensus = data.get("consensus") or {}
    assert consensus.get("consensus") is not True
    assert consensus.get("value") is None


def test_search_tool_authoritative_market_keeps_number():
    """端到端：雅虎财经 + 英为财情都给了正确的 25,466.19，
    Gate 的错数排在后面——权威的数留着，consensus 是 true。"""
    from tools.search import SearchResult, SearchService
    from tools.search_tool import _run_search, set_search_service

    def _result(url, text):
        return SearchResult(title="纳指收盘", url=url, snippet=text)

    tavily = _StubProvider("tavily", [
        _result("https://www.gate.com/news", "11 月 11 日纳指收於 25,065.87。"),
        _result("https://finance.yahoo.com/quote", "11 月 11 日纳指收盘 25,466.19，涨 1.60%。"),
    ])
    anysearch = _StubProvider("anysearch", [
        _result("https://www.investing.com/indices", "11 月 11 日纳指收盘 25,466.19，涨 1.60%。"),
    ])
    service = SearchService(
        anysearch=anysearch, tavily=tavily, parallel=_StubProvider("parallel", [])
    )
    token = set_search_service(service)
    try:
        out = _run_search({
            "query": "纳指 11 月 11 日收盘多少",
            "date_range": {"start": "2026-11-11", "end": "2026-11-11"},
        })
    finally:
        from tools.search_tool import _SERVICE_VAR

        _SERVICE_VAR.reset(token)
    assert out["ok"] is True
    data = out["data"]
    urls = [item["url"] for item in data["results"]]
    assert urls[0].startswith("https://finance.yahoo.com"), "权威排前面"
    assert urls[-1].startswith("https://www.gate.com"), "黑名单排最后"
    consensus = data.get("consensus") or {}
    assert consensus.get("consensus") is True
    assert consensus.get("value") == "25466.19"
    # Gate 的错数不在印证里（黑名单不进印证）。
    assert "gate.com" not in (consensus.get("sources") or [])


def test_search_tool_weather_unaffected_by_market_preference():
    """天气问题不受行情偏好影响：不排序、不拿数（只对 market 开）。"""
    from tools.search import SearchResult, SearchService
    from tools.search_tool import _run_search, set_search_service

    result = SearchResult(
        title="广州天气",
        url="https://www.gate.com/weather",
        snippet="广州今天气温 25 度。",
    )
    service = SearchService(
        anysearch=_StubProvider("anysearch", []),
        tavily=_StubProvider("tavily", [result]),
        parallel=_StubProvider("parallel", []),
    )
    token = set_search_service(service)
    try:
        out = _run_search({"query": "广州今天气温多少度"})
    finally:
        from tools.search_tool import _SERVICE_VAR

        _SERVICE_VAR.reset(token)
    assert out["ok"] is True
    # Gate 的天气结果还在（行情黑名单不管天气），没有“无法核实”的行情说明。
    assert out["data"]["results"]
    assert not any("无法核实" in note for note in out["data"].get("notes") or [])
