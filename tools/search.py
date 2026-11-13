# tools/search.py（第一批工具：上网搜索）
# 作用：三个搜索服务 + 一个统一调度器（GPT 的想法），按任务路由、不一股脑全试：
#   - AnySearch：日常主力，每天 1,000 次免费，超量返回 402（不自动扣费）。
#     一般搜索 tag 用 general.general（官网文档原话：可省略 tag，这里显式写上）。
#   - Tavily：结构化交叉验证，每月 1,000 credits 免费；只用 basic（1 credit），
#     绝不用 advanced（2 credits）和 Research 端点。需要 Key（TAVILY_API_KEY），
#     没 Key 时这个服务不可用，调度器跳过它。
#   - Parallel MCP：零门槛入口兼兜底，不用注册、不用 Key、不用绑卡；
#     走托管 MCP（https://mcp.parallel.ai/mcp，没有末尾斜杠），调 web_search。
#     默认 fast 模式（中文查询用 fast，Turbo 只支持英文和日文）。
#   - SearchService：统一调度器——先试不用翻译的（Tavily → Parallel），
#     前面免费额度用完或失败了，才走 AnySearch（中文要翻译）；
#     失败/额度用完/没结果 → 顺着链路切备选；cross_check 时两个服务
#     的结果互相印证（多源印证）；失败和无结果区分开；保留来源 URL 和时间；
#     所有服务都失败/都没结果时老实说失败，绝不编结果。
#
# Key 只从环境变量读（ANYSEARCH_API_KEY、TAVILY_API_KEY），绝不进代码、
# 绝不进聊天、绝不打进日志。AnySearch 没 Key 也能用匿名额度（官网说的），
# Tavily 没 Key 就不可用。

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import requests

from tools.date_numbers import (
    english_market_close_query,
    extract_dated_numbers,
    extract_numbers,
    parse_date_mentions,
)
from tools.errors import QuotaExhausted, ToolError

ANYSEARCH_API_URL = "https://api.anysearch.com/v1/search"
ANYSEARCH_GENERAL_TAG = "general.general"
TAVILY_API_URL = "https://api.tavily.com/search"
PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"

_DEFAULT_TIMEOUT = 30
_SNIPPET_MAX_CHARS = 1000
# 提速 4 号（2026-11-13，BOOS 批准）：正文摘要 extract 的长度和条数——
#   前 2–3 条结果带 extract（正文里挑出来的 400–800 字），让模型
#   少跑一轮“再去读正文”；摘要（snippet）还是照旧，不动。
_EXTRACT_MIN_CHARS = 400
_EXTRACT_MAX_CHARS = 800
_EXTRACT_TOP_RESULTS = 3
# 中文及相关字符区（含 CJK 标点、全角形式）：用来判断一段文字是不是中文。
_CJK_RE = re.compile(r"[\u2e80-\ua4cf\uff00-\uffef]")
# 一段连续的 Latin-1 字符（0x80–0xFF）：双重编码的 UTF-8 字节读出来就是这个样子。
_MOJIBAKE_RUN_RE = re.compile(r"[\x80-\xff]+")

# 只有 AnySearch 需要先把中文查询翻成英文：它中文相关性差（实测“广州天气”
# 返回的全是法院文书），英文结果才对题。Tavily 和 Parallel 中文结果是好的，
# 不用翻。翻译器由调用方注入（接进聊天时用 DeepSeek 翻）；不传就不翻译。
_TRANSLATE_TO_ENGLISH_PROVIDERS = {"anysearch"}
# 翻回来时只翻标题和摘要的前 200 字：全文太长，翻全了费 token；十二自己
# 看英文就行，翻回来是给 BOOS 看的。
_TRANSLATE_BACK_SNIPPET_CHARS = 200


def _contains_cjk(text):
    """这段文字里有没有中文（中文查询才需要翻成英文再搜）。"""
    return bool(_CJK_RE.search(text or ""))


def _repair_mojibake(text):
    """修双重编码的中文：UTF-8 字节被当成 Latin-1 读了（比如“天气”变成“å¤©æ°”）。

    按连续 Latin-1 字符分段修：每段转回 UTF-8，能转出中文就替换，转不出
    （比如单独的 °、é）就原样保留。正常中文、ASCII 不动。
    """
    if not text:
        return text

    def _fix_run(match):
        run = match.group(0)
        try:
            fixed = run.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return run
        if _CJK_RE.search(fixed):
            return fixed
        return run

    return _MOJIBAKE_RUN_RE.sub(_fix_run, text)


@dataclass
class SearchResult:
    """一条搜索结果：标题、来源链接、摘要、发布时间（有就留，没有就空）。
    extract（提速 4 号）：正文里挑出来的摘要（400–800 字），
    优先含日期/数字/关键事实的段落；只有前 2–3 条结果带它，
    其余的为空字符串。"""

    title: str
    url: str
    snippet: str = ""
    published_at: str = None
    score: float = None
    extract: str = ""

    def to_dict(self):
        # 50/54 号：每条结果带 dated_numbers（数字-日期对照，只留配上日期的；
        # 没配上日期的数不放进来，模型就知道“这个数没日期，别当某天的数用”。
        # 注意：这里只看标题 + 摘要 + extract 里的文字日期，绝不拿 published_at 当数的日期。）
        # 判断 (b) 号：value（数值）和 kind（价格/百分比/点位）也留着，
        # 双源印证要拿它比两个来源的数对不对得上。
        dated = [
            {
                "number": item["number"],
                "value": item["value"],
                "kind": item["kind"],
                "date": item["date"],
            }
            for item in extract_dated_numbers(
                f"{self.title}. {self.snippet}. {self.extract}"
            )
            if item["date"] is not None
        ]
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "score": self.score,
            "extract": self.extract,
            "dated_numbers": dated,
        }


@dataclass
class SearchOutcome:
    """一次搜索的完整结果：状态、结果列表、走了哪些服务、质量检查。"""

    query: str
    provider: str
    status: str = "ok"  # ok / no_results / failed
    results: list = field(default_factory=list)
    answer: str = None
    error: str = None
    attempts: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    fetched_at: str = ""
    # 48 号：实际发出去的查询词（走 AnySearch 翻成英文时是英文那份，
    # 不是模型写的中文原文；没翻译时就等于 query）。界面上显示它，
    # BOOS 才看得到“到底拿什么词去搜的”。
    executed_query: str = ""
    # 28 号：按日期范围过滤的情况（{"start", "end", "dropped", "undated"}），
    # 没给日期范围时是空字典。
    date_range: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "query": self.query,
            "provider": self.provider,
            "status": self.status,
            "results": [r.to_dict() for r in self.results],
            "answer": self.answer,
            "error": self.error,
            "attempts": self.attempts,
            "quality": self.quality,
            "fetched_at": self.fetched_at,
            "executed_query": self.executed_query or self.query,
            "date_range": self.date_range,
        }


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_extract(body):
    """从正文里挑一段摘要（提速 4 号）：优先要含日期/数字/关键事实的段落，
    凑够 400–800 字。段落按换行切；含日期或数字的段落先取（按原文顺序），
    凑不够 400 字时再按原文顺序补别的段落；正文太短时有几字算几字。
    正文为空时返回 ""。"""
    text = (body or "").strip()
    if not text:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    if not paragraphs:
        return text[:_EXTRACT_MAX_CHARS]

    def _informative(paragraph):
        """这段有没有日期或数字（信息密度高，优先要）。"""
        if parse_date_mentions(paragraph):
            return True
        if extract_numbers(paragraph):
            return True
        return False

    chosen = set()
    total = 0
    # 先取含日期/数字的段落（原文顺序）。
    for index, paragraph in enumerate(paragraphs):
        if total >= _EXTRACT_MAX_CHARS:
            break
        if _informative(paragraph):
            chosen.add(index)
            total += len(paragraph)
    # 凑不够 400 字时，按原文顺序补别的段落。
    if total < _EXTRACT_MIN_CHARS:
        for index, paragraph in enumerate(paragraphs):
            if total >= _EXTRACT_MAX_CHARS:
                break
            if index in chosen:
                continue
            chosen.add(index)
            total += len(paragraph)
    if not chosen:
        return text[:_EXTRACT_MAX_CHARS]
    # 按原文顺序拼。
    return " ".join(paragraphs[index] for index in sorted(chosen))[
        :_EXTRACT_MAX_CHARS
    ]


def _normalize_results(items):
    """把各家形状不一的结果条目归一化成 SearchResult 列表。

    认的字段：title / url（或 link）/ snippet / content / text / excerpts
    （列表，拼成摘要）/ published_at / score。没有 URL 的条目直接丢掉——
    没有来源的结果不能用。
    """
    out = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link") or ""
        if not url:
            continue
        snippet = item.get("snippet") or item.get("content") or item.get("text") or ""
        excerpts = item.get("excerpts")
        if not snippet and isinstance(excerpts, list):
            snippet = " ".join(str(e) for e in excerpts)
        published_at = (
            item.get("published_at") or item.get("publish_date") or item.get("publishedDate")
        )
        # 提速 4 号：前 2–3 条结果带 extract（正文里挑出来的摘要）。
        # 正文用 content / text（没给时用 snippet 本身）；只给前
        # _EXTRACT_TOP_RESULTS 条，省 token。
        body = item.get("content") or item.get("text") or snippet
        extract = (
            _make_extract(_repair_mojibake(str(body)))
            if len(out) < _EXTRACT_TOP_RESULTS and body
            else ""
        )
        out.append(
            SearchResult(
                title=_repair_mojibake(item.get("title") or url),
                url=url,
                snippet=_repair_mojibake(str(snippet))[:_SNIPPET_MAX_CHARS],
                published_at=published_at,
                score=item.get("score"),
                extract=extract,
            )
        )
    return out


def _extract_items(payload):
    """从各家返回的 JSON 里抠出结果条目列表（data.results / results / data）。"""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("results", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("results"), list):
            return value["results"]
    return []


def _check_http_error(response, provider):
    """把 HTTP 错误翻成对应的异常：402/429 是额度用完，其他是普通失败。"""
    status = response.status_code
    if status in (402, 429):
        raise QuotaExhausted(
            provider,
            status,
            f"{provider} 额度用完了（HTTP {status}），切备选服务",
        )
    if status >= 400:
        raise ToolError(f"{provider} 请求失败（HTTP {status}）")


class AnySearchProvider:
    """AnySearch：日常主力。POST /v1/search，Bearer Key，匿名也能用（额度更低）。

    一般搜索显式用 tag general.general（官网文档：可省略，这里写清楚）。
    """

    name = "anysearch"

    def __init__(self, api_key=None, session=None, timeout=_DEFAULT_TIMEOUT):
        self.api_key = api_key if api_key is not None else os.environ.get("ANYSEARCH_API_KEY")
        self.session = session or requests.Session()
        self.timeout = timeout

    @property
    def available(self):
        return True  # 有 Key 更好，没 Key 走匿名额度，也能用

    @property
    def unavailable_reason(self):
        return None

    def search(self, query, max_results=5):
        if not (query or "").strip():
            raise ToolError("查询不能为空")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "query": query,
            "tag": ANYSEARCH_GENERAL_TAG,
            "max_results": max(1, min(10, max_results)),
        }
        try:
            response = self.session.post(
                ANYSEARCH_API_URL, json=body, headers=headers, timeout=self.timeout
            )
        except Exception as exc:
            raise ToolError(f"anysearch 请求失败：{exc}") from exc
        _check_http_error(response, "anysearch")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ToolError("anysearch 返回了无法解析的结果") from exc
        results = _normalize_results(_extract_items(payload))
        return SearchOutcome(
            query=query,
            provider="anysearch",
            status="ok" if results else "no_results",
            results=results,
            fetched_at=_now_iso(),
        )


class TavilyProvider:
    """Tavily：结构化交叉验证。POST /v1/search，Bearer Key。

    只用 search_depth basic（1 credit/次），绝不用 advanced（2 credits）
    和 Research 端点。没 Key 时不可用（调度器会跳过它）。
    """

    name = "tavily"

    def __init__(self, api_key=None, session=None, timeout=_DEFAULT_TIMEOUT):
        self.api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY")
        self.session = session or requests.Session()
        self.timeout = timeout

    @property
    def available(self):
        return bool(self.api_key)

    @property
    def unavailable_reason(self):
        if self.api_key:
            return None
        return "未配置 TAVILY_API_KEY"

    def search(self, query, max_results=5):
        if not self.api_key:
            raise ToolError("Tavily 需要 Key：未配置 TAVILY_API_KEY")
        if not (query or "").strip():
            raise ToolError("查询不能为空")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "query": query,
            "search_depth": "basic",
            "max_results": max(1, min(20, max_results)),
            "include_answer": False,
            "topic": "general",
        }
        try:
            response = self.session.post(
                TAVILY_API_URL, json=body, headers=headers, timeout=self.timeout
            )
        except Exception as exc:
            raise ToolError(f"tavily 请求失败：{exc}") from exc
        _check_http_error(response, "tavily")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ToolError("tavily 返回了无法解析的结果") from exc
        results = _normalize_results(_extract_items(payload))
        return SearchOutcome(
            query=query,
            provider="tavily",
            status="ok" if results else "no_results",
            results=results,
            fetched_at=_now_iso(),
        )


class ParallelMCPClient:
    """Parallel 托管 MCP 的最小客户端（streamable HTTP + JSON-RPC 2.0）。

    只实现够用的三步：initialize → notifications/initialized → tools/call。
    不用 Key。响应可能是纯 JSON，也可能是 SSE（text/event-stream），两种都认。
    """

    def __init__(self, endpoint=PARALLEL_MCP_URL, session=None, timeout=60):
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self.timeout = timeout
        self._initialized = False
        self._next_id = 0
        self._session_id = None
        self._lock = threading.RLock()
        # 提速 1 号：同一轮里多条搜索并行跑时，两个线程可能同时走到
        # Parallel MCP——_next_id、_session_id、_initialized 都是
        # 共享的可变状态，不加锁会撞号、丢会话。RLock 串行化每一次
        # JSON-RPC 调用（initialize 里调 _rpc 要重入，所以用 RLock）；
        # Tavily / AnySearch 是纯 HTTP、无共享可变状态，不用锁，
        # 仍可并行。

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _parse_sse_event(self, event_lines):
        """把一个 SSE 事件拼回 JSON 文本。

        按 SSE 规范，多个 data: 行用换行拼起来；实测 Parallel 的响应里
        JSON 字符串自带真实换行（instructions 很长），续行不带 data: 前缀，
        也要拼回去。event:/id:/retry: 行跳过。
        """
        parts = []
        for line in event_lines:
            if line.startswith("data:"):
                parts.append(line[5:].lstrip(" "))
            elif line.startswith(("event:", "id:", "retry:", ":")):
                continue
            else:
                # data: 行里 JSON 字符串自带的真实换行（续行）。
                parts.append(line)
        return "\n".join(parts)

    def _parse_response(self, response):
        """解析 MCP 响应：纯 JSON 或 SSE（取最后一个能解析出 JSON-RPC 消息的事件）。"""
        content_type = response.headers.get("Content-Type", "") if response.headers else ""
        text = response.text or ""
        is_sse = (
            "text/event-stream" in content_type
            or "\ndata:" in text
            or text.startswith("data:")
        )
        if not is_sse:
            try:
                return response.json()
            except ValueError as exc:
                raise ToolError("Parallel MCP 返回了无法解析的响应") from exc
        # 按空行切事件，从后往前找第一个能解析出 JSON-RPC 消息的事件。
        events = []
        current = []
        for line in text.splitlines():
            if line.strip() == "":
                if current:
                    events.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            events.append(current)
        for event_lines in reversed(events):
            payload = self._parse_sse_event(event_lines)
            if not payload.strip():
                continue
            for strict in (True, False):
                try:
                    parsed = json.loads(payload, strict=strict)
                except ValueError:
                    continue
                if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
                    return parsed
                break  # 解析出来了但不是 JSON-RPC 消息，找上一个事件
        raise ToolError("Parallel MCP 返回了无法解析的 SSE 响应")

    def _post(self, payload):
        try:
            response = self.session.post(
                self.endpoint, json=payload, headers=self._headers(), timeout=self.timeout
            )
        except Exception as exc:
            raise ToolError(f"parallel 请求失败：{exc}") from exc
        if response.status_code in (402, 429):
            raise QuotaExhausted(
                "parallel", response.status_code, "parallel 被限流了（匿名额度），切备选服务"
            )
        if response.status_code >= 400:
            raise ToolError(f"parallel MCP 请求失败（HTTP {response.status_code}）")
        session_id = (response.headers or {}).get("Mcp-Session-Id") or (
            response.headers or {}
        ).get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        return response

    def _rpc(self, method, params=None):
        # 提速 1 号：整段上锁——id 分配、发请求、读会话号必须原子，
        # 不然两个并行搜索会拿到同一个 id 或互相覆盖会话。
        with self._lock:
            self._next_id += 1
            payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
            if params is not None:
                payload["params"] = params
            parsed = self._parse_response(self._post(payload))
            if isinstance(parsed, dict) and "error" in parsed:
                error = parsed["error"]
                raise ToolError(
                    f"Parallel MCP 报错：{error.get('message', error)}"
                )
            return parsed.get("result") if isinstance(parsed, dict) else None

    def initialize(self):
        """握手一次：initialize + notifications/initialized，之后复用。"""
        # 提速 1 号：整段上锁（RLock，里面调 _rpc 会重入）——
        # _initialized 的“查了没有再握手”必须原子，不然两个线程
        # 会各握手一次，第二个覆盖第一个的会话号。
        with self._lock:
            if self._initialized:
                return
            self._rpc(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "twelve-tools", "version": "0.1.0"},
                },
            )
            try:
                self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            except ToolError:
                pass  # 通知发出去就行，回不回无所谓
            self._initialized = True

    def call_tool(self, name, arguments):
        """调一个 MCP 工具，返回它的文本内容（拼成一段）。"""
        self.initialize()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise ToolError(f"Parallel MCP 工具 {name} 返回了空结果")
        if result.get("isError"):
            texts = [
                item.get("text", "")
                for item in result.get("content", [])
                if isinstance(item, dict)
            ]
            raise ToolError(f"Parallel MCP 工具 {name} 报错：{' '.join(texts)[:500]}")
        texts = [
            item.get("text", "")
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(texts)

    def web_search(self, objective, search_queries):
        """调 web_search：objective 是自然语言目标，search_queries 是 2-3 条
        3-6 词的关键词查询。返回解析后的 JSON（字典）。

        实测返回的是带 search_id 的大 JSON，字符串里可能带真实换行，
        严格解析失败时放宽（strict=False）再试。
        """
        text = self.call_tool(
            "web_search", {"objective": objective, "search_queries": search_queries}
        )
        for strict in (True, False):
            try:
                return json.loads(text, strict=strict)
            except ValueError:
                continue
        raise ToolError("Parallel web_search 返回了无法解析的结果")


def _to_search_queries(query):
    """把自然语言查询拆成 2-3 条 3-6 词的关键词查询（Parallel 的要求）。

    中文没空格就整句当一条；英文长句按 5 个词一切，最多 3 条。
    """
    words = (query or "").split()
    if not words:
        raise ToolError("查询不能为空")
    if len(words) <= 6:
        return [query.strip()]
    chunks = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
    return chunks[:3]


class ParallelProvider:
    """Parallel MCP：零门槛入口兼兜底。不用注册、不用 Key、不用绑卡。

    默认 fast 模式（中文查询用 fast；Turbo 只支持英文和日文）。
    """

    name = "parallel"

    def __init__(self, client=None, session=None, timeout=60):
        self.client = client or ParallelMCPClient(session=session, timeout=timeout)

    @property
    def available(self):
        return True

    @property
    def unavailable_reason(self):
        return None

    def search(self, query, max_results=5):
        if not (query or "").strip():
            raise ToolError("查询不能为空")
        payload = self.client.web_search(query.strip(), _to_search_queries(query))
        results = _normalize_results(_extract_items(payload))
        return SearchOutcome(
            query=query,
            provider="parallel",
            status="ok" if results else "no_results",
            results=results[:max_results],
            fetched_at=_now_iso(),
        )


# ---------- 统一调度器 ----------

# 路由表（按任务类型，不是一股脑全试）：
#   普通查询和结构化信息都先走不用翻译的：Tavily → Parallel → AnySearch。
#   Tavily（中文直接搜，一个月 1000 次免费额度）先用，Parallel
#   （不用注册、不用 Key）兜底，前面两个额度用完或失败了，才走
#   AnySearch（每天 1000 次免费，但中文要翻译，多花翻译钱）。
#   失败回落按链路顺序走（402 超额、429、报错都算失败、自动下一个），
#   不用另加额度计数逻辑。
_CHAINS = {
    "normal": ["tavily", "parallel", "anysearch"],
    "structured": ["tavily", "parallel", "anysearch"],
}
# 交叉验证的搭档：主服务是谁，就用另一个服务的结果互相印证。
_CROSS_CHECK_PARTNER = {"anysearch": "tavily", "tavily": "anysearch", "parallel": "anysearch"}


def _domain_of(url):
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc.lower()
    except Exception:
        return ""


# ---------- 判断 (b) 号：双源印证 ----------

# 权威来源白名单：官方交易所、政府气象局、央行——一家就能验证
#（可信度上限 0.85），不用等第二家。注意：这里是注册域名
#（finance.yahoo.com 和 yahoo.com 都是 yahoo.com，不算两家）。
# 2026-11-13（BOOS 定：行情搜索要权威报道）：再加主流财经数据商——
# 雅虎财经、英为财情、MarketWatch、CNBC、路透、彭博、华尔街日报、
# 金融时报、晨星、巴伦周刊、东方财富、新浪财经、中国证券网、
# 证券时报、财联社、同花顺。行情收盘数一家权威来源就能验证
#（0.85 上限）；这些站的天气/汇率页同样可靠，不另设白名单。
AUTHORITATIVE_DOMAINS = frozenset(
    {
        # 官方交易所。
        "nasdaq.com", "nyse.com", "spglobal.com", "dowjones.com",
        "sse.com.cn", "szse.cn", "hkex.com.hk",
        # 政府气象局。
        "weather.gov", "weather.com.cn", "metoffice.gov.uk", "bom.gov.au",
        # 央行/官方汇率。
        "federalreserve.gov", "ecb.europa.eu", "boj.or.jp", "pbc.gov.cn",
        # 主流财经数据商（行情）。
        "yahoo.com", "investing.com", "marketwatch.com", "cnbc.com",
        "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
        "morningstar.com", "barrons.com",
        "eastmoney.com", "sina.com.cn", "cnstock.com", "stcn.com",
        "cls.cn", "10jqka.com.cn",
    }
)

# 行情黑名单（2026-11-13，BOOS 定）：这些站不是股票/指数收盘的
# 权威来源——绝不能只凭它们一家就把数当收盘价给出去。
# 主要是币圈交易所的新闻（Gate 那篇把 11 月 10 日的收盘写成
# 11 月 11 日的，就是这类来源）。注册域名比对，和白名单同口径。
MARKET_DENYLIST_DOMAINS = frozenset(
    {
        "gate.com", "binance.com", "coinbase.com", "okx.com",
        "kraken.com", "bitfinex.com", "huobi.com", "bybit.com",
        "kucoin.com", "gemini.com", "bitstamp.net", "bitget.com",
        "mexc.com",
    }
)

# 两段式公共后缀（注册域名要取最后三段，比如 news.co.uk 的注册域名是
# news.co.uk，不是 co.uk；com.cn 同理）。
_TWO_PART_SUFFIXES = frozenset(
    {
        "co.uk", "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
        "co.jp", "com.au", "co.kr", "com.br", "com.hk", "com.tw",
        "com.sg", "co.in", "com.mx", "co.nz",
    }
)


def registrable_domain(url):
    """一个 URL 的注册域名（判断 b 号：独立来源按注册域名算）。
    finance.yahoo.com 和 yahoo.com 都是 yahoo.com——不算两家；
    cnn.com 和 reuters.com 是两家。www. 前缀去掉。"""
    host = _domain_of(url or "")
    if host.startswith("www."):
        host = host[4:]
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return host
    if ".".join(parts[-2:]) in _TWO_PART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_authoritative_domain(domain):
    """这个注册域名是不是权威来源（判断 b 号白名单）？"""
    return (domain or "").lower() in AUTHORITATIVE_DOMAINS


def is_market_denylisted_domain(domain):
    """这个注册域名是不是行情黑名单（2026-11-13）？
    黑名单站不是股票/指数收盘的权威来源，绝不能只凭它一家
    就把数当收盘价给出去。"""
    return (domain or "").lower() in MARKET_DENYLIST_DOMAINS


def market_source_tier(domain):
    """行情来源档位（2026-11-13，权威报道优先）：
    0 = 权威财经来源（白名单里的交易所/财经数据商），
    1 = 普通来源（既不权威也不在黑名单），
    2 = 黑名单来源（币圈交易所等）。排序用：权威在前，黑名单在后。"""
    domain = (domain or "").lower()
    if domain in AUTHORITATIVE_DOMAINS:
        return 0
    if domain in MARKET_DENYLIST_DOMAINS:
        return 2
    return 1


def check_numeric_consensus(results, date_iso, tolerance=0.01):
    """双源印证（判断 b 号）：给定一组搜索结果（to_dict 后的 dict，
    含 url + dated_numbers）和目标日期，判断两个独立来源的数对不对得上。

    只认 date == date_iso 且配上日期的数（50/54 的日期-数字配对已经
    算好了）；同一注册域名的多个结果算一家（finance.yahoo.com 和
    yahoo.com 不算两家）。数按 kind 分组（收盘价和涨跌幅不混着比），
    同 kind 里两个独立来源的数在 tolerance 内对得上才算印证。
    收盘价 tolerance 默认 0.01（精确到小数点后两位）；气温传 1.0
    （±1°C 算对得上）。

    返回 {"consensus": bool, "value": str|None, "kind": str|None,
    "sources": [domain, ...], "source_count": int,
    "by_kind": {kind: {"consensus": bool, "value": ..., "sources": [...]}},
    "date": date_iso}。
    consensus 为真时 value 是两个来源都认的数（字符串，原样）；为假时
    value 是 None（别拿单家的数当事实用）。
    """
    by_kind = {}
    for result in results or []:
        domain = registrable_domain((result or {}).get("url") or "")
        if not domain:
            continue
        for entry in (result or {}).get("dated_numbers") or []:
            if (entry or {}).get("date") != date_iso:
                continue
            kind = (entry or {}).get("kind") or "price"
            value = (entry or {}).get("value")
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            by_kind.setdefault(kind, {}).setdefault(domain, number)
    verdicts = {}
    for kind, per_domain in by_kind.items():
        domains = sorted(per_domain)
        agreed_value = None
        agreed_sources = []
        # 找“最多独立来源认的那个数”：每个数当一回候选，数它有多少家
        # 在 tolerance 内认它；两家以上认同一个数才算印证。
        for candidate_domain in domains:
            candidate = per_domain[candidate_domain]
            supporters = [
                domain
                for domain in domains
                if abs(per_domain[domain] - candidate) <= tolerance
            ]
            if len(supporters) >= 2 and len(supporters) > len(agreed_sources):
                agreed_sources = supporters
                agreed_value = candidate
        if agreed_value is not None:
            verdicts[kind] = {
                "consensus": True,
                "value": f"{agreed_value:.2f}",
                "sources": agreed_sources,
                "source_count": len(agreed_sources),
            }
        else:
            verdicts[kind] = {
                "consensus": False,
                "value": None,
                "sources": domains,
                "source_count": len(domains),
            }
    # 主结论认收盘价（price）那组——行情问题最关心收盘价；
    # 没收盘价时认数最多的那组。
    main_kind = "price" if "price" in verdicts else None
    if main_kind is None and verdicts:
        main_kind = max(verdicts, key=lambda k: verdicts[k]["source_count"])
    main = verdicts.get(main_kind) if main_kind else None
    return {
        "consensus": bool(main and main["consensus"]),
        "value": main["value"] if main else None,
        "kind": main_kind,
        "sources": main["sources"] if main else [],
        "source_count": main["source_count"] if main else 0,
        "by_kind": verdicts,
        "date": date_iso,
    }


def _parse_date(value):
    """把发布时间解析成日期（ISO / YYYY-MM-DD 都认），解析不了返回 None。"""
    if not value:
        return None
    text = str(value).strip()
    try:
        from datetime import datetime as _dt

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return _dt.fromisoformat(text[:19]).date() if "T" in text else _dt.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        from datetime import date as _date

        return _date.fromisoformat(text[:10])
    except ValueError:
        return None


def _is_stale(published_at, days=30):
    """发布时间是不是 30 天以前的（过时了的结果要标出来）。"""
    parsed = _parse_date(published_at)
    if parsed is None:
        return False
    from datetime import date as _date

    return (_date.today() - parsed).days > days


def _parse_date_range(date_range):
    """把 date_range 参数解析成 (开始日期, 结束日期)，都是 date 对象。

    认 {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}；只给一边时另一边
    就等于那一边（只查当天）；开始晚于结束、格式不对、不是字典时
    返回 None——调用方按“没给日期范围”处理，不静默用错范围查。
    """
    if not isinstance(date_range, dict):
        return None
    start = _parse_date(date_range.get("start"))
    end = _parse_date(date_range.get("end"))
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start > end:
        return None
    return start, end


def _apply_date_range(results, start, end):
    """按日期范围过滤结果：返回 (留下的结果, 掉出去的条数, 没带时间的条数)。

    28 号：搜回来的数分属不同日期时，只留要的那几天的——带发布时间的，
    日期在 [start, end] 里才留；没带时间、或时间解析不出来的，留着：
    没法按日期判断就别硬丢，丢了可能把对的结果也丢了，数量记下来、
    让模型知道这几条是没时间信息的（回答时标明“没带时间”）。
    """
    kept = []
    dropped = 0
    undated = 0
    for result in results or []:
        parsed = _parse_date(result.published_at)
        if parsed is None:
            undated += 1
            kept.append(result)
        elif start <= parsed <= end:
            kept.append(result)
        else:
            dropped += 1
    return kept, dropped, undated


def _quality_notes(results, corroborated_domains):
    """结果质量检查（GPT 的建议）：来源、时间（新旧）、多源印证，写成人话。"""
    notes = []
    with_url = sum(1 for r in results if r.url)
    notes.append(f"{len(results)} 条结果里有 {with_url} 条带来源链接")
    domains = {_domain_of(r.url) for r in results if r.url}
    notes.append(f"来源分布在 {len(domains)} 个网站")
    if corroborated_domains:
        notes.append(
            f"{len(corroborated_domains)} 个来源在两个服务里都出现了，互相印证："
            + "、".join(sorted(corroborated_domains)[:5])
        )
    else:
        notes.append("没有两个服务都出现的来源，没法互相印证")
    dated = [r for r in results if r.published_at]
    if not dated:
        notes.append("结果里没有时间信息，无法判断新旧")
    else:
        notes.append(f"{len(dated)} 条结果带发布时间")
        stale = [r for r in dated if _is_stale(r.published_at)]
        if stale:
            notes.append(f"其中 {len(stale)} 条是 30 天以前的，可能过时了")
    return notes


def _outcome_has_close_candidate(outcome, day_iso):
    """这一批结果里有没有可能是收盘价的数：plain number（kind=price，
    不是百分比）且配上了要的那天的日期。百分比不算——财联社“跌幅达
    0.9%”配上了 2026-11-10，但那是盘中跌幅，不是收盘价；2025 年表里
    的数配的是 2025 年的日期，也不算。"""
    for result in (outcome.results or []):
        for item in extract_dated_numbers(f"{result.title}. {result.snippet}"):
            if item.get("kind") == "price" and item.get("date") == day_iso:
                return True
    return False


def _outcome_has_undated_price_for_day(outcome, day_iso):
    """这一批结果里有没有没配上日期的 plain number，同时整条结果提到了
    要的那天（日期在标题里、数在正文里这种——句内配对够不着，但整条
    结果就是讲那天的，比如美联社“Tuesday 11/10/2026”标题 + 正文里的
    7,520.93）。重试的接受条件用这个：模型自己会读正文判断哪个数是
    收盘价，重试只要把像样的 raw 材料带回来就行。"""
    day = date.fromisoformat(day_iso)
    for result in (outcome.results or []):
        if _result_rank_for_day(result, day_iso) == 1:
            return True
    return False


def _result_rank_for_day(result, day_iso):
    """一条结果对要的那天的有用程度（51 号排序用）：0 = 有配上那天的
    plain number（收盘价候选）；1 = 有没配上日期的 plain number、同时
    整条结果提到了那天（日期在标题里、数在正文里）；2 = 其他。
    排序时 0 在前、1 其次，2 垫底。"""
    text = f"{result.title}. {result.snippet}"
    dated = extract_dated_numbers(text)
    if any(
        item.get("kind") == "price" and item.get("date") == day_iso
        for item in dated
    ):
        return 0
    day = date.fromisoformat(day_iso)
    if any(m["date"] == day for m in parse_date_mentions(text)) and any(
        item.get("kind") == "price" and item.get("date") is None for item in dated
    ):
        return 1
    return 2


class SearchService:
    """统一搜索调度器：按任务路由、失败切备选、必要时交叉验证。

    providers 可注入（测试用假服务模拟失败，直接测备选链路）；
    不传就用默认的三个真实服务（Key 从环境变量读）。
    translator(text, target)：把 text 翻成 target（"en" 或 "zh"）。
    接进聊天时由调用方传一个 DeepSeek 翻译器；不传就不翻译，行为不变。
    """

    def __init__(self, anysearch=None, tavily=None, parallel=None, translator=None):
        self.providers = {
            "anysearch": anysearch or AnySearchProvider(),
            "tavily": tavily or TavilyProvider(),
            "parallel": parallel or ParallelProvider(),
        }
        self.translator = translator

    def _send_query_for(self, provider_name, query):
        """主链路和交叉验证共用的发查询规则：AnySearch + 中文查询时，
        先翻成英文再搜（AnySearch 中文相关性差，直接搜中文会返回不相干的
        法院文书）；翻译失败就回落原文。Tavily、Parallel 中文直接搜，不翻。
        """
        if (
            self.translator is not None
            and provider_name in _TRANSLATE_TO_ENGLISH_PROVIDERS
            and _contains_cjk(query)
        ):
            try:
                translated = self.translator(query, "en")
            except Exception:
                translated = None
            if translated and translated.strip() and translated.strip() != query.strip():
                return translated.strip()
        return query

    def _translate_results_back(self, results):
        """把一批结果的标题和摘要翻回中文给 BOOS 看。翻不成功不影响结果本身。"""
        if self.translator is None:
            return False
        for result in results:
            if result.title:
                try:
                    translated = self.translator(result.title, "zh")
                except Exception:
                    translated = None
                if translated:
                    result.title = translated
            if result.snippet:
                head = result.snippet[:_TRANSLATE_BACK_SNIPPET_CHARS]
                try:
                    translated = self.translator(head, "zh")
                except Exception:
                    translated = None
                if translated:
                    result.snippet = translated + result.snippet[_TRANSLATE_BACK_SNIPPET_CHARS:]
        return True

    def _translate_back(self, outcome):
        """把翻成英文搜回来的结果，标题和摘要翻回中文给 BOOS 看。

        只翻中过英文的那一路（AnySearch）赢来的结果；Tavily、Parallel
        中文直接搜回来的不用翻。翻不成功不影响结果本身。
        """
        return self._translate_results_back(outcome.results)

    def available_providers(self):
        """每个服务是否可用、不可用是什么原因。"""
        return [
            {
                "name": name,
                "available": provider.available,
                "reason": provider.unavailable_reason,
            }
            for name, provider in self.providers.items()
        ]

    def search(self, query, kind="normal", max_results=5, cross_check=False, force_provider=None, date_range=None):
        """搜一次。kind 是 normal（普通查询）或 structured（结构化信息）；
        force_provider 直接指定只用某一个服务（备选链路测试用）；
        cross_check 为真时，再用搭档服务的结果互相印证。
        date_range 是 {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}（28 号）：
        结果按 published_at 过滤到这几天——带时间的只留范围内的，没带时间
        的留着（没法按日期判断）；一家服务过滤后没结果就顺着链路切下一家，
        都没结果时报错里写明是日期过滤掉的，让模型放宽范围或去掉它再查。"""
        if kind not in _CHAINS:
            raise ToolError(f"未知的搜索类型：{kind}（只认 normal / structured）")
        if not (query or "").strip():
            raise ToolError("查询不能为空")
        # 28 号：日期范围先解析好（格式不对按没给处理，不静默用错范围查）。
        parsed_range = _parse_date_range(date_range)
        range_start = parsed_range[0] if parsed_range else None
        range_end = parsed_range[1] if parsed_range else None
        date_filter_stats = {"dropped": 0, "undated": 0}
        if force_provider is not None:
            if force_provider not in self.providers:
                raise ToolError(f"未知的搜索服务：{force_provider}")
            chain = [force_provider]
        else:
            chain = _CHAINS[kind]
        attempts = []
        outcome = None
        winning_translated = False
        for name in chain:
            provider = self.providers[name]
            if not provider.available:
                attempts.append(
                    {
                        "provider": name,
                        "status": "skipped",
                        "reason": provider.unavailable_reason or "不可用",
                    }
                )
                continue
            send_query = self._send_query_for(name, query)
            try:
                candidate = provider.search(send_query, max_results=max_results)
            except QuotaExhausted as exc:
                attempts.append(
                    {"provider": name, "status": "failed", "reason": exc.message}
                )
                continue
            except ToolError as exc:
                attempts.append(
                    {"provider": name, "status": "failed", "reason": str(exc)}
                )
                continue
            if candidate.status == "no_results":
                attempts.append(
                    {"provider": name, "status": "empty", "reason": "这个服务没找到结果"}
                )
                continue
            # 28 号：按日期范围过滤——一家服务过滤后没结果就切下一家，
            # 别拿着别的日子的数当答案。没带时间的留着，数量记下来。
            if range_start is not None:
                kept, dropped, undated = _apply_date_range(
                    candidate.results, range_start, range_end
                )
                if not kept:
                    attempts.append(
                        {
                            "provider": name,
                            "status": "empty",
                            "reason": f"按日期过滤（{range_start} 至 {range_end}）后没有结果",
                            "date_filtered": True,
                        }
                    )
                    continue
                candidate.results = kept
                date_filter_stats["dropped"] += dropped
                date_filter_stats["undated"] += undated
            outcome = candidate
            outcome.query = query
            # 48 号：实际发出去的查询词（翻成英文时是英文那份）。
            outcome.executed_query = send_query
            attempt = {"provider": name, "status": "ok"}
            if send_query != query:
                attempt["translated_query"] = send_query
                winning_translated = True
            attempts.append(attempt)
            break
        if outcome is None:
            failed = [a for a in attempts if a["status"] == "failed"]
            # 报错只说实际试过的服务：--provider 只指定一个时，别写成
            #“三个服务都试过了”（以前写死了这句，只试一个也这么说）。
            # 括号里也只列真的失败了的——“没找到结果”不是失败，别和
            # 失败混在一个“都失败了”后面。
            attempted = [a for a in attempts if a["status"] in ("failed", "empty")]
            if failed:
                failed_names = "、".join(a["provider"] for a in failed)
                error = f"试过的服务都失败了（{failed_names}）： " + "；".join(
                    f"{a['provider']}（{a['reason']}）" for a in failed
                )
                empty = [a for a in attempted if a["status"] == "empty"]
                if empty:
                    empty_names = "、".join(a["provider"] for a in empty)
                    error += f"；{empty_names} 没找到结果"
                status = "failed"
            elif attempted:
                tried = "、".join(a["provider"] for a in attempted)
                if any(a.get("date_filtered") for a in attempted):
                    # 28 号：不是没搜到，是日期过滤把结果都滤掉了——
                    # 报错里写明，让模型放宽范围或去掉 date_range 再查，
                    # 别让它以为“没这个东西”就说不知道。
                    error = (
                        f"按日期过滤（{range_start} 至 {range_end}）后没有结果："
                        "搜到了，但都不是这几天的——"
                        "可以放宽日期范围或去掉 date_range 再查"
                    )
                else:
                    error = f"试过的服务都没找到结果（{tried}）"
                status = "no_results"
            else:
                skipped = "；".join(
                    f"{a['provider']}（{a['reason']}）" for a in attempts
                )
                if not skipped:
                    error = "没有可用的搜索服务"
                elif len(chain) == 1:
                    # 链路只有一个服务（--provider 指定的那一个被跳过），
                    # 别写“三个服务都没法用”，实际只试了一个。
                    error = f"指定的服务没法用：{skipped}"
                else:
                    error = f"三个服务都没法用：{skipped}"
                status = "failed"
            return SearchOutcome(
                query=query,
                provider=force_provider or "",
                status=status,
                results=[],
                error=error,
                attempts=attempts,
                quality={"results": 0, "with_url": 0, "notes": [error]},
                fetched_at=_now_iso(),
            )
        # 51 号实测（2026-11-12）：Tavily 中文查询召回差——“标普500指数
        # 11月10日 收盘”只搜回一则盘中快讯（0.9% 是百分比，不是收盘价）
        # 和一张 2025 年的表，没有 2026-11-10 的收盘数；同一问题翻成英文
        # 一次就搜到美联社的收盘数（7,520.93）。链路只在“没结果”时才切
        # 备选，“有结果、但没有要的那天的收盘价”就停了，模型只能说没查到。
        # 所以：带单日 date_range 且结果里一个收盘价候选都没有时，翻成英文
        # 整条链重跑一次；重跑后有了收盘价候选（配上那天的 plain number，
        # 或整条结果讲那天、数没配上日期）就用重跑的，还是没有就留第一份
        #（别把已有的结果弄丢）。只重试一次——翻出来的查询不再含中文，
        # 不会再触发。
        # 英文查询优先用程序拼的（行情问题：实体英文名 + close + 日期，
        # 确定的、不花钱）：模型翻译器对短词组不稳，实测“纳斯达克指数 收盘”
        # 翻回来还是中文，重试就触发不了；非行情问题才走模型翻译器。
        if (
            range_start is not None
            and range_start == range_end
            and _contains_cjk(query)
            and not _outcome_has_close_candidate(outcome, range_start.isoformat())
        ):
            retry_query = english_market_close_query(query, range_start)
            if retry_query is None and self.translator is not None:
                translated = None
                try:
                    translated = self.translator(query, "en")
                except Exception:
                    translated = None
                if (
                    translated
                    and translated.strip()
                    and translated.strip() != query.strip()
                    and not _contains_cjk(translated)
                ):
                    retry_query = translated.strip()
            if retry_query:
                retry = self.search(
                    retry_query,
                    kind=kind,
                    max_results=max_results,
                    cross_check=cross_check,
                    force_provider=force_provider,
                    date_range=date_range,
                )
                if retry.status == "ok" and (
                    _outcome_has_close_candidate(retry, range_start.isoformat())
                    or _outcome_has_undated_price_for_day(
                        retry, range_start.isoformat()
                    )
                ):
                    attempts.append(
                        {
                            "provider": retry.provider,
                            "status": "ok",
                            "reason": "中文查询没有要的那天的收盘价，翻成英文重搜后找到",
                        }
                    )
                    outcome = retry
        # 51 号实测（2026-11-12）：道指 11.10 那轮，第一轮搜回来的结果里
        # 本来就有 2026-11-10 的收盘数（英为财情历史表：收 52,231.89），
        # 但排在维基百科那条巨型摘要后面，工具结果 4000 字截断把它截掉了，
        # 模型根本没看到、只能说没查到。所以：带单日 date_range 时，把
        # 有那天收盘价候选的结果排前面（稳定排序，只提顺序、不丢结果），
        # 其次是整条结果讲那天、数没配上日期的，让模型先看到能用的数。
        if range_start is not None and range_start == range_end:
            day_iso = range_start.isoformat()
            outcome.results.sort(key=lambda r: _result_rank_for_day(r, day_iso))
        if not outcome.fetched_at:
            outcome.fetched_at = _now_iso()
        outcome.attempts = attempts
        translated_back = False
        if winning_translated:
            translated_back = self._translate_back(outcome)
        corroborated = []
        if cross_check:
            partner_name = _CROSS_CHECK_PARTNER.get(outcome.provider)
            partner = self.providers.get(partner_name) if partner_name else None
            if partner is not None and partner.available:
                try:
                    # 搭档也走同一套翻译规则：搭档是 AnySearch 且查询含中文时，
                    # 先翻成英文再搜，不然交叉验证会把 AnySearch 的中文烂结果
                    #（法院文书）混进好结果里。
                    partner_query = self._send_query_for(partner_name, query)
                    partner_outcome = partner.search(partner_query, max_results=max_results)
                    if partner_outcome.status == "ok":
                        # 28 号：搭档的结果也要按日期过滤（搭档那路没过滤），
                        # 过滤完再合并、再算互相印证——印证只算范围内的来源。
                        if range_start is not None:
                            filtered, dropped, undated = _apply_date_range(
                                partner_outcome.results, range_start, range_end
                            )
                            partner_outcome.results = filtered
                            date_filter_stats["dropped"] += dropped
                            date_filter_stats["undated"] += undated
                        seen_urls = {r.url for r in outcome.results}
                        merged = list(outcome.results)
                        new_results = []
                        for result in partner_outcome.results:
                            if result.url not in seen_urls:
                                merged.append(result)
                                new_results.append(result)
                                seen_urls.add(result.url)
                        primary_domains = {_domain_of(r.url) for r in outcome.results}
                        partner_domains = {
                            _domain_of(r.url) for r in partner_outcome.results
                        }
                        corroborated = sorted(primary_domains & partner_domains)
                        outcome.results = merged
                        # 搭档那路翻成英文搜的，新并进来的结果也要翻回中文。
                        if partner_query != query and new_results:
                            if self._translate_results_back(new_results):
                                translated_back = True
                        attempts.append(
                            {"provider": partner_name, "status": "ok", "reason": "交叉验证"}
                        )
                except (ToolError, QuotaExhausted) as exc:
                    attempts.append(
                        {
                            "provider": partner_name,
                            "status": "failed",
                            "reason": f"交叉验证失败：{exc}",
                        }
                    )
        outcome.quality = {
            "results": len(outcome.results),
            "with_url": sum(1 for r in outcome.results if r.url),
            "unique_domains": len({_domain_of(r.url) for r in outcome.results if r.url}),
            "corroborated_domains": corroborated,
            "translated_back": translated_back,
            "notes": _quality_notes(outcome.results, corroborated),
        }
        # 28 号：日期过滤的情况记进结果里（模型看得到、界面也看得到），
        # 告诉模型哪些去掉了、哪些没带时间——回答时标明每组数是哪天的。
        if range_start is not None:
            outcome.date_range = {
                "start": str(range_start),
                "end": str(range_end),
                "dropped": date_filter_stats["dropped"],
                "undated": date_filter_stats["undated"],
            }
            note = f"已按日期过滤到 {range_start} 至 {range_end}"
            if date_filter_stats["dropped"]:
                note += f"，去掉了 {date_filter_stats['dropped']} 条其他日期的结果"
            if date_filter_stats["undated"]:
                note += f"，{date_filter_stats['undated']} 条没带时间（没法按日期判断，留着）"
            outcome.quality["notes"].append(note)
        if translated_back:
            outcome.quality["notes"].append(
                "查询先翻成英文再搜（AnySearch 中文不准），标题和摘要已翻回中文"
            )
        outcome.attempts = attempts
        return outcome
