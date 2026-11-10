# tests/test_tools_search.py（第一批工具：搜索测试）
# 作用：验证 tools/search.py —— AnySearch / Tavily / Parallel MCP 三个服务
# 的请求形状与结果解析、402/429 额度耗尽、失败与无结果的区分、调度器的
# 备选链路（直接可测：注入假服务模拟失败，不等真实失败）与交叉验证。
# 全部离线：用假 HTTP 会话和假服务对象，不碰真实接口、不产生费用。
# 真实接口验证（AnySearch/Tavily 需 BOOS 注册 Key；Parallel 可直接测）
# 不在 pytest 里，见 tools/README.md。

import json

import pytest

from tools.errors import QuotaExhausted, ToolError
from tools.search import (
    AnySearchProvider,
    ParallelMCPClient,
    ParallelProvider,
    SearchOutcome,
    SearchResult,
    SearchService,
    TavilyProvider,
)


class FakeResponse:
    """requests.Response 的最小替身：状态码 + JSON + 文本 + 头。"""

    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class FakeSession:
    """requests.Session 的替身：按 handler 决定返回什么，记下每次调用。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        return self.handler(url, kwargs)


# ---------- AnySearch ----------


def anysearch_ok_payload():
    return {
        "data": {
            "results": [
                {"title": "标题一", "url": "https://example.com/1", "snippet": "摘要一"},
                {"title": "标题二", "url": "https://example.com/2", "content": "正文二"},
            ],
            "total_results": 2,
        }
    }


def test_anysearch_ok_parses_results():
    session = FakeSession(lambda url, kw: FakeResponse(200, anysearch_ok_payload()))
    provider = AnySearchProvider(api_key="test-key", session=session)
    outcome = provider.search("测试查询")
    assert outcome.status == "ok"
    assert outcome.provider == "anysearch"
    assert len(outcome.results) == 2
    assert outcome.results[0].url == "https://example.com/1"
    assert outcome.results[0].snippet == "摘要一"
    assert outcome.results[1].snippet == "正文二"
    # 请求形状：Bearer Key、general.general、max_results。
    call = session.calls[0]
    assert call["url"] == "https://api.anysearch.com/v1/search"
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer test-key"
    body = call["kwargs"]["json"]
    assert body["query"] == "测试查询"
    assert body["tag"] == "general.general"


def test_anysearch_402_is_quota_exhausted():
    session = FakeSession(lambda url, kw: FakeResponse(402, {"error": "quota exceeded"}))
    provider = AnySearchProvider(api_key="k", session=session)
    with pytest.raises(QuotaExhausted) as exc:
        provider.search("q")
    assert exc.value.status == 402
    assert exc.value.provider == "anysearch"


def test_anysearch_500_is_failure():
    session = FakeSession(lambda url, kw: FakeResponse(500, {"error": "boom"}))
    provider = AnySearchProvider(api_key="k", session=session)
    with pytest.raises(ToolError):
        provider.search("q")


def test_anysearch_no_results_status():
    session = FakeSession(
        lambda url, kw: FakeResponse(200, {"data": {"results": [], "total_results": 0}})
    )
    provider = AnySearchProvider(api_key="k", session=session)
    outcome = provider.search("查不到的东西")
    assert outcome.status == "no_results"
    assert outcome.results == []


# ---------- Tavily ----------


def tavily_ok_payload():
    return {
        "query": "q",
        "results": [
            {"title": "T1", "url": "https://a.com/1", "content": "内容一", "score": 0.9},
            {"title": "T2", "url": "https://b.com/2", "content": "内容二", "score": 0.8},
        ],
        "response_time": 0.5,
    }


def test_tavily_requires_key():
    provider = TavilyProvider(
        api_key=None, session=FakeSession(lambda u, k: FakeResponse(200, {}))
    )
    assert provider.available is False
    assert "TAVILY_API_KEY" in (provider.unavailable_reason or "")


def test_tavily_ok_uses_basic_depth():
    session = FakeSession(lambda url, kw: FakeResponse(200, tavily_ok_payload()))
    provider = TavilyProvider(api_key="tvly-test", session=session)
    outcome = provider.search("q")
    assert outcome.status == "ok"
    assert outcome.provider == "tavily"
    assert len(outcome.results) == 2
    body = session.calls[0]["kwargs"]["json"]
    assert body["search_depth"] == "basic"  # 最便宜的普通搜索，绝不用 advanced
    assert body["max_results"] == 5
    assert session.calls[0]["kwargs"]["headers"]["Authorization"] == "Bearer tvly-test"


def test_tavily_429_is_quota_exhausted():
    session = FakeSession(lambda url, kw: FakeResponse(429, {"detail": "rate limited"}))
    provider = TavilyProvider(api_key="tvly-test", session=session)
    with pytest.raises(QuotaExhausted) as exc:
        provider.search("q")
    assert exc.value.provider == "tavily"


# ---------- Parallel MCP ----------


def parallel_search_text():
    return json.dumps(
        {
            "results": [
                {
                    "title": "P1",
                    "url": "https://p.com/1",
                    "excerpts": ["摘录一", "摘录二"],
                },
                {"title": "P2", "url": "https://p.com/2", "excerpts": ["摘录三"]},
            ],
            "session_id": "sess-1",
        }
    )


def make_parallel_session(mode="json"):
    def handler(url, kwargs):
        payload = kwargs["json"]
        method = payload.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "serverInfo": {"name": "parallel", "version": "1"},
            }
            body = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
            if mode == "sse":
                return FakeResponse(
                    200,
                    None,
                    text="event: message\ndata: " + json.dumps(body) + "\n\n",
                    headers={"Content-Type": "text/event-stream"},
                )
            return FakeResponse(200, body)
        if method == "notifications/initialized":
            return FakeResponse(202, None, text="")
        if method == "tools/call":
            assert payload["params"]["name"] == "web_search"
            args = payload["params"]["arguments"]
            assert "objective" in args and "search_queries" in args
            assert len(args["search_queries"]) >= 1
            result = {"content": [{"type": "text", "text": parallel_search_text()}]}
            body = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
            if mode == "sse":
                return FakeResponse(
                    200,
                    None,
                    text="event: message\ndata: " + json.dumps(body) + "\n\n",
                    headers={"Content-Type": "text/event-stream"},
                )
            return FakeResponse(200, body)
        raise AssertionError(f"unexpected method {method}")

    return FakeSession(handler)


def test_parallel_web_search_json_response():
    session = make_parallel_session("json")
    outcome = ParallelProvider(client=ParallelMCPClient(session=session)).search(
        "今天天气怎么样"
    )
    assert outcome.status == "ok"
    assert outcome.provider == "parallel"
    assert len(outcome.results) == 2
    assert outcome.results[0].url == "https://p.com/1"
    assert "摘录一" in outcome.results[0].snippet
    # MCP 地址：没有末尾斜杠（带斜杠会 404），不用 Key。
    assert session.calls[0]["url"] == "https://mcp.parallel.ai/mcp"
    assert "Authorization" not in session.calls[0]["kwargs"]["headers"]


def test_parallel_web_search_sse_response():
    session = make_parallel_session("sse")
    outcome = ParallelProvider(client=ParallelMCPClient(session=session)).search(
        "今天天气怎么样"
    )
    assert outcome.status == "ok"
    assert len(outcome.results) == 2


def test_parallel_search_queries_derived():
    # 长查询要拆成 2-3 条 3-6 词的关键词查询。
    session = make_parallel_session("json")
    ParallelProvider(client=ParallelMCPClient(session=session)).search(
        "latest artificial intelligence model rankings news today"
    )
    call = [
        c
        for c in session.calls
        if c["kwargs"]["json"].get("method") == "tools/call"
    ][0]
    queries = call["kwargs"]["json"]["params"]["arguments"]["search_queries"]
    assert 1 <= len(queries) <= 3
    for query in queries:
        assert 1 <= len(query.split()) <= 6


def test_parallel_429_is_quota_exhausted():
    def handler(url, kwargs):
        payload = kwargs["json"]
        if payload.get("method") == "initialize":
            return FakeResponse(
                200,
                {"jsonrpc": "2.0", "id": payload["id"], "result": {"protocolVersion": "2025-03-26"}},
            )
        if payload.get("method") == "notifications/initialized":
            return FakeResponse(202, None, text="")
        return FakeResponse(429, {"error": "too many requests"})

    session = FakeSession(handler)
    with pytest.raises(QuotaExhausted) as exc:
        ParallelProvider(client=ParallelMCPClient(session=session)).search("q")
    assert exc.value.provider == "parallel"


# ---------- 调度器（SearchService） ----------


class FakeProvider:
    """可注入的假搜索服务：按剧本返回结果或抛错。"""

    def __init__(self, name, script, available=True, unavailable_reason=None):
        self.name = name
        self._script = list(script)
        self.calls = []
        self._available = available
        self._unavailable_reason = unavailable_reason

    @property
    def available(self):
        return self._available

    @property
    def unavailable_reason(self):
        return self._unavailable_reason

    def search(self, query, max_results=5):
        self.calls.append(query)
        if not self._script:
            return SearchOutcome(query=query, provider=self.name, status="no_results")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok_outcome(query, provider, urls):
    return SearchOutcome(
        query=query,
        provider=provider,
        status="ok",
        results=[
            SearchResult(title=f"t{i}", url=url, snippet=f"s{i}")
            for i, url in enumerate(urls)
        ],
    )


def test_scheduler_falls_back_on_402():
    anysearch = FakeProvider(
        "anysearch", [QuotaExhausted("anysearch", 402, "当日额度用完")]
    )
    parallel = FakeProvider(
        "parallel", [ok_outcome("q", "parallel", ["https://p.com/1"])]
    )
    tavily = FakeProvider("tavily", [])
    service = SearchService(anysearch=anysearch, tavily=tavily, parallel=parallel)
    outcome = service.search("q")
    assert outcome.status == "ok"
    assert outcome.provider == "parallel"
    assert len(outcome.results) == 1
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0]["provider"] == "anysearch"
    assert outcome.attempts[0]["status"] == "failed"
    assert outcome.attempts[1]["provider"] == "parallel"
    assert outcome.attempts[1]["status"] == "ok"
    assert tavily.calls == []  # 备选成功了就不再往下试


def test_scheduler_all_fail_is_honest():
    service = SearchService(
        anysearch=FakeProvider("anysearch", [ToolError("anysearch 挂了")]),
        tavily=FakeProvider("tavily", [ToolError("tavily 挂了")]),
        parallel=FakeProvider("parallel", [ToolError("parallel 挂了")]),
    )
    outcome = service.search("q")
    assert outcome.status == "failed"
    assert outcome.results == []  # 绝不编结果
    assert outcome.error
    assert "anysearch" in outcome.error
    assert "tavily" in outcome.error
    assert "parallel" in outcome.error


def test_scheduler_all_empty_is_no_results():
    service = SearchService(
        anysearch=FakeProvider("anysearch", []),
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
    )
    outcome = service.search("查不到的东西")
    assert outcome.status == "no_results"
    assert outcome.results == []
    assert len(outcome.attempts) == 3


def test_scheduler_force_provider_direct():
    anysearch = FakeProvider("anysearch", [ok_outcome("q", "anysearch", ["https://a.com"])])
    tavily = FakeProvider("tavily", [])
    parallel = FakeProvider("parallel", [])
    service = SearchService(anysearch=anysearch, tavily=tavily, parallel=parallel)
    outcome = service.search("q", force_provider="tavily")
    assert outcome.provider == "tavily"
    assert outcome.status == "no_results"  # tavily 剧本是空的，只调它一个
    assert anysearch.calls == [] and parallel.calls == []


def test_scheduler_cross_check_merges_and_corrobates():
    anysearch = FakeProvider(
        "anysearch",
        [ok_outcome("q", "anysearch", ["https://a.com/1", "https://b.com/2"])],
    )
    tavily = FakeProvider(
        "tavily",
        [ok_outcome("q", "tavily", ["https://b.com/2", "https://c.com/3"])],
    )
    parallel = FakeProvider("parallel", [])
    service = SearchService(anysearch=anysearch, tavily=tavily, parallel=parallel)
    outcome = service.search("q", cross_check=True)
    assert outcome.status == "ok"
    assert len(outcome.results) == 3  # 按 URL 去重后合并
    assert "b.com" in outcome.quality.get("corroborated_domains", [])


def test_scheduler_skips_unavailable_provider():
    anysearch = FakeProvider("anysearch", [ok_outcome("q", "anysearch", ["https://a.com"])])
    tavily = FakeProvider(
        "tavily", [], available=False, unavailable_reason="未配置 TAVILY_API_KEY"
    )
    parallel = FakeProvider("parallel", [])
    service = SearchService(anysearch=anysearch, tavily=tavily, parallel=parallel)
    outcome = service.search("q", kind="structured")
    assert outcome.status == "ok"
    assert outcome.provider == "anysearch"  # Tavily 没 Key，跳过它走 AnySearch
    assert tavily.calls == []
    assert any(a.get("reason") == "未配置 TAVILY_API_KEY" for a in outcome.attempts)


def test_scheduler_quality_notes_present():
    anysearch = FakeProvider(
        "anysearch", [ok_outcome("q", "anysearch", ["https://a.com/1", "https://b.com/2"])]
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
    )
    outcome = service.search("q")
    assert outcome.quality["results"] == 2
    assert outcome.quality["with_url"] == 2
    assert outcome.quality["notes"]
    assert outcome.fetched_at


# ---------- 多场景补充（BOOS 2026-11-10 要求：不只测正常路） ----------


def test_scheduler_structured_routes_to_tavily_first():
    # 结构化查询先走 Tavily，Tavily 行了就不再试别的。
    tavily = FakeProvider(
        "tavily", [ok_outcome("q", "tavily", ["https://t.com/1"])]
    )
    anysearch = FakeProvider("anysearch", [])
    parallel = FakeProvider("parallel", [])
    service = SearchService(anysearch=anysearch, tavily=tavily, parallel=parallel)
    outcome = service.search("q", kind="structured")
    assert outcome.status == "ok"
    assert outcome.provider == "tavily"
    assert anysearch.calls == [] and parallel.calls == []


def test_scheduler_provider_down_falls_back():
    # 主力挂了（不是 402，是直接报错），也要切备选，不能直接认栽。
    anysearch = FakeProvider("anysearch", [ToolError("anysearch 连不上")])
    parallel = FakeProvider(
        "parallel", [ok_outcome("q", "parallel", ["https://p.com/1"])]
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=parallel,
    )
    outcome = service.search("q")
    assert outcome.status == "ok"
    assert outcome.provider == "parallel"
    assert outcome.attempts[0]["status"] == "failed"
    assert "连不上" in outcome.attempts[0]["reason"]


def test_scheduler_no_results_falls_back():
    # 主力“没找到结果”不等于失败，但也值得换一家再试试。
    anysearch = FakeProvider("anysearch", [])
    parallel = FakeProvider(
        "parallel", [ok_outcome("q", "parallel", ["https://p.com/1"])]
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=parallel,
    )
    outcome = service.search("q")
    assert outcome.status == "ok"
    assert outcome.provider == "parallel"
    assert outcome.attempts[0]["status"] == "empty"


def test_quality_stale_source_flagged():
    # 带发布时间的结果：30 天以前的要标“可能过时了”，新的不标。
    stale_result = SearchResult(
        title="旧闻", url="https://old.com/1", snippet="s", published_at="2020-01-01"
    )
    fresh_result = SearchResult(
        title="新闻",
        url="https://new.com/1",
        snippet="s",
        published_at="2026-11-10",
    )
    anysearch = FakeProvider(
        "anysearch",
        [
            SearchOutcome(
                query="q",
                provider="anysearch",
                status="ok",
                results=[stale_result, fresh_result],
            )
        ],
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
    )
    outcome = service.search("q")
    notes = " ".join(outcome.quality["notes"])
    assert "可能过时了" in notes
    assert "1 条是 30 天以前的" in notes


def test_quality_keeps_url_and_time():
    # 结果必须保留来源 URL 和时间，调度器不丢信息。
    result = SearchResult(
        title="t",
        url="https://a.com/1",
        snippet="s",
        published_at="2026-11-09T10:00:00",
        score=0.9,
    )
    anysearch = FakeProvider(
        "anysearch",
        [SearchOutcome(query="q", provider="anysearch", status="ok", results=[result])],
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
    )
    outcome = service.search("q")
    assert outcome.results[0].url == "https://a.com/1"
    assert outcome.results[0].published_at == "2026-11-09T10:00:00"
    assert outcome.fetched_at  # 抓取时间也留着
    assert outcome.quality["with_url"] == 1


# ---------- 真实 SSE 格式回归（Parallel 实测发现：JSON 会跨行） ----------


def test_parallel_sse_multiline_json_parses():
    # 真实响应：JSON 很长，data: 行断成两截，续行不带 data: 前缀。
    client = ParallelMCPClient(session=FakeSession(lambda u, k: FakeResponse(200, {})))
    body = (
        "event: message\n"
        'data: {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-03-26",\n'
        ' "serverInfo": {"name": "Parallel MCP"}}}\n\n'
    )
    response = FakeResponse(
        200, None, text=body, headers={"Content-Type": "text/event-stream"}
    )
    parsed = client._parse_response(response)
    assert parsed["result"]["protocolVersion"] == "2025-03-26"
    assert parsed["result"]["serverInfo"]["name"] == "Parallel MCP"


def test_parallel_sse_literal_newline_in_string_parses():
    # 真实响应：JSON 字符串里带真实换行（instructions 很长），
    # 严格解析失败时要放宽（strict=False）再试。
    client = ParallelMCPClient(session=FakeSession(lambda u, k: FakeResponse(200, {})))
    body = (
        "event: message\n"
        'data: {"jsonrpc": "2.0", "id": 1, "result": {"instructions": "line one\n'
        'line two"}}\n\n'
    )
    response = FakeResponse(
        200, None, text=body, headers={"Content-Type": "text/event-stream"}
    )
    parsed = client._parse_response(response)
    assert parsed["result"]["instructions"] == "line one\nline two"


def test_parallel_sse_multiple_data_lines_joined():
    # 多个 data: 行按 SSE 规范用换行拼起来。
    client = ParallelMCPClient(session=FakeSession(lambda u, k: FakeResponse(200, {})))
    body = (
        "event: message\n"
        'data: {"jsonrpc": "2.0", "id": 1,\n'
        'data: "result": {"ok": true}}\n\n'
    )
    response = FakeResponse(
        200, None, text=body, headers={"Content-Type": "text/event-stream"}
    )
    parsed = client._parse_response(response)
    assert parsed["result"] == {"ok": True}


# ---------- 真实 web_search 格式回归（Parallel 实测发现） ----------


def make_parallel_session_with_text(text):
    """假 MCP 会话：tools/call 直接返回给定的结果文本（走真实客户端的解析）。"""

    def handler(url, kwargs):
        payload = kwargs["json"]
        method = payload.get("method")
        if method == "initialize":
            return FakeResponse(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"protocolVersion": "2025-03-26"},
                },
            )
        if method == "notifications/initialized":
            return FakeResponse(202, None, text="")
        if method == "tools/call":
            result = {"content": [{"type": "text", "text": text}]}
            return FakeResponse(
                200, {"jsonrpc": "2.0", "id": payload["id"], "result": result}
            )
        raise AssertionError(f"unexpected method {method}")

    return FakeSession(handler)


def test_parallel_web_search_result_with_literal_newline():
    # 真实返回：JSON 字符串里带真实换行，严格解析失败时要放宽再试。
    import json as _json

    payload = {
        "search_id": "search_x",
        "results": [
            {
                "url": "https://tianqi.2345.com/",
                "title": "天气",
                "publish_date": "2026-11-10",
                "excerpts": ["今天晴\n明天多云"],
            }
        ],
    }
    text = _json.dumps(payload, ensure_ascii=False)
    # 把摘要里的 \n 换成真实换行（模拟服务端原样输出）。
    text = text.replace("\\n", "\n")
    session = make_parallel_session_with_text(text)
    outcome = ParallelProvider(client=ParallelMCPClient(session=session)).search(
        "今天天气怎么样"
    )
    assert outcome.status == "ok"
    assert outcome.results[0].url == "https://tianqi.2345.com/"
    assert outcome.results[0].published_at == "2026-11-10"


def test_parallel_web_search_result_real_shape():
    # 真实返回的形状：search_id + results[{url, title, publish_date, excerpts}]。
    import json as _json

    payload = {
        "search_id": "search_x",
        "results": [
            {
                "url": "https://example.com/1",
                "title": "å¤©æ°",  # 双重编码的“天气”，要修回来
                "publish_date": None,
                "excerpts": ["摘录一", "摘录二"],
            },
            {
                "url": "https://example.com/2",
                "title": "正常标题",
                "publish_date": "2026-11-09",
                "excerpts": ["摘录三"],
            },
        ],
    }
    session = make_parallel_session_with_text(_json.dumps(payload, ensure_ascii=False))
    outcome = ParallelProvider(client=ParallelMCPClient(session=session)).search(
        "今天天气怎么样"
    )
    assert outcome.status == "ok"
    assert len(outcome.results) == 2
    assert outcome.results[0].title == "天气"
    assert "摘录一" in outcome.results[0].snippet
    assert outcome.results[1].title == "正常标题"
    assert outcome.results[1].published_at == "2026-11-09"




def _garble(s):
    """把正常中文转成双重编码的样子（UTF-8 字节被当成 Latin-1 读了）。"""
    return s.encode("utf-8").decode("latin-1")


def test_mojibake_repair_mixed_chinese_and_garbled():
    # 真实返回：标题/摘要里正确中文和双重编码混在一起，也要逐段修。
    from tools.search import _normalize_results

    items = [
        {
            "url": "https://example.com/1",
            "title": "福州天气_" + _garble("【福州今天天气预报】"),
            "excerpts": [_garble("下载APP"), "扫码下载app，天气随时看"],
        }
    ]
    results = _normalize_results(items)
    assert results[0].title == "福州天气_【福州今天天气预报】"
    assert results[0].snippet.startswith("下载APP")
    assert "扫码下载app" in results[0].snippet


def test_mojibake_repair_keeps_normal_text():
    # 正常文本不动：单独的 °、é、法语都不修。
    from tools.search import _repair_mojibake

    assert _repair_mojibake("今天 2~15°，晴") == "今天 2~15°，晴"
    assert _repair_mojibake("café naïve") == "café naïve"
    assert _repair_mojibake("hello world") == "hello world"
    assert _repair_mojibake("") == ""


# ---------- 翻译（中文查询翻成英文再搜，结果翻回中文） ----------


class FakeTranslator:
    """假翻译器：按词典翻，没在词典里的原样返回。记下每次调用。"""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, text, target):
        self.calls.append((text, target))
        return self.mapping.get((text, target), text)


def test_translator_translates_chinese_query_for_anysearch_only():
    # 中文查询：AnySearch 那路翻成英文再搜；Tavily 不翻。
    translator = FakeTranslator({("广州天气", "en"): "Guangzhou weather"})
    anysearch = FakeProvider(
        "anysearch", [ok_outcome("Guangzhou weather", "anysearch", ["https://a.com/1"])]
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
        translator=translator,
    )
    outcome = service.search("广州天气")
    assert outcome.status == "ok"
    assert outcome.provider == "anysearch"
    assert anysearch.calls == ["Guangzhou weather"]
    assert ("广州天气", "en") in translator.calls
    assert outcome.attempts[0]["translated_query"] == "Guangzhou weather"
    assert outcome.query == "广州天气"
    assert outcome.quality["translated_back"] is True


def test_translator_skips_tavily_chinese_query():
    # 结构化查询走 Tavily 优先：Tavily 中文直接搜，不翻。
    translator = FakeTranslator({("广州天气", "en"): "Guangzhou weather"})
    tavily = FakeProvider(
        "tavily", [ok_outcome("广州天气", "tavily", ["https://t.com/1"])]
    )
    service = SearchService(
        anysearch=FakeProvider("anysearch", []),
        tavily=tavily,
        parallel=FakeProvider("parallel", []),
        translator=translator,
    )
    outcome = service.search("广州天气", kind="structured")
    assert outcome.provider == "tavily"
    assert tavily.calls == ["广州天气"]
    assert translator.calls == []


def test_translator_skips_english_query():
    # 英文查询不用翻。
    translator = FakeTranslator({})
    anysearch = FakeProvider(
        "anysearch", [ok_outcome("Guangzhou weather", "anysearch", ["https://a.com/1"])]
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
        translator=translator,
    )
    outcome = service.search("Guangzhou weather")
    assert anysearch.calls == ["Guangzhou weather"]
    assert translator.calls == []
    assert outcome.quality["translated_back"] is False


def test_translator_translates_results_back_to_chinese():
    # 翻成英文搜回来的结果，标题和摘要翻回中文。
    translator = FakeTranslator(
        {
            ("广州天气", "en"): "Guangzhou weather",
            ("Sunny day", "zh"): "晴天",
            ("Warm and clear", "zh"): "温暖晴朗",
        }
    )
    results = SearchOutcome(
        query="Guangzhou weather",
        provider="anysearch",
        status="ok",
        results=[
            SearchResult(title="Sunny day", url="https://a.com/1", snippet="Warm and clear"),
        ],
    )
    anysearch = FakeProvider("anysearch", [results])
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
        translator=translator,
    )
    outcome = service.search("广州天气")
    assert outcome.results[0].title == "晴天"
    assert outcome.results[0].snippet == "温暖晴朗"
    assert outcome.quality["translated_back"] is True
    assert any("翻回中文" in note for note in outcome.quality["notes"])


def test_translator_failure_falls_back_to_original_query():
    # 翻译器炸了也不影响搜索：用原查询照常搜。
    def broken_translator(text, target):
        raise RuntimeError("翻译服务挂了")

    anysearch = FakeProvider(
        "anysearch", [ok_outcome("广州天气", "anysearch", ["https://a.com/1"])]
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=FakeProvider("tavily", []),
        parallel=FakeProvider("parallel", []),
        translator=broken_translator,
    )
    outcome = service.search("广州天气")
    assert outcome.status == "ok"
    assert anysearch.calls == ["广州天气"]
    assert outcome.quality["translated_back"] is False


def test_translator_cross_check_partner_gets_original_query():
    # 交叉验证的搭档拿原查询（中文），不拿翻过的英文。
    translator = FakeTranslator({("广州天气", "en"): "Guangzhou weather"})
    anysearch = FakeProvider(
        "anysearch", [ok_outcome("Guangzhou weather", "anysearch", ["https://a.com/1"])]
    )
    tavily = FakeProvider(
        "tavily", [ok_outcome("广州天气", "tavily", ["https://a.com/1"])]
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=tavily,
        parallel=FakeProvider("parallel", []),
        translator=translator,
    )
    outcome = service.search("广州天气", cross_check=True)
    assert tavily.calls == ["广州天气"]
    assert len(outcome.results) == 1


def test_translator_cross_check_partner_anysearch_gets_translated_query():
    # Bug A 回归：交叉验证的搭档是 AnySearch 且查询含中文时，搭档也要拿
    # 翻过的英文查询去搜，不能拿中文原文（否则 AnySearch 返回法院文书，
    # 烂结果混进好结果里）。搭档那路搜回来的结果也要翻回中文。
    translator = FakeTranslator(
        {
            ("广州天气", "en"): "Guangzhou weather",
            ("Sunny in Guangzhou", "zh"): "广州晴",
        }
    )
    tavily = FakeProvider(
        "tavily", [ok_outcome("广州天气", "tavily", ["https://t.com/1"])]
    )
    anysearch = FakeProvider(
        "anysearch",
        [
            SearchOutcome(
                query="Guangzhou weather",
                provider="anysearch",
                status="ok",
                results=[
                    SearchResult(
                        title="Sunny in Guangzhou",
                        url="https://a.com/1",
                        snippet="Clear sky",
                    )
                ],
            )
        ],
    )
    service = SearchService(
        anysearch=anysearch,
        tavily=tavily,
        parallel=FakeProvider("parallel", []),
        translator=translator,
    )
    outcome = service.search("广州天气", kind="structured", cross_check=True)
    assert outcome.provider == "tavily"
    assert tavily.calls == ["广州天气"]  # Tavily 中文直接搜，不翻
    assert anysearch.calls == ["Guangzhou weather"]  # 搭档 AnySearch 拿翻过的英文
    assert ("广州天气", "en") in translator.calls
    # 搭档新并进来的结果翻回了中文
    assert outcome.results[1].title == "广州晴"
    assert outcome.quality["translated_back"] is True
