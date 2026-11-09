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
#   - SearchService：统一调度器——普通查询 → AnySearch；结构化 → Tavily；
#     主力失败/额度用完/没结果 → 顺着链路切备选；cross_check 时两个服务
#     的结果互相印证（多源印证）；失败和无结果区分开；保留来源 URL 和时间；
#     所有服务都失败/都没结果时老实说失败，绝不编结果。
#
# Key 只从环境变量读（ANYSEARCH_API_KEY、TAVILY_API_KEY），绝不进代码、
# 绝不进聊天、绝不打进日志。AnySearch 没 Key 也能用匿名额度（官网说的），
# Tavily 没 Key 就不可用。

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from tools.errors import QuotaExhausted, ToolError

ANYSEARCH_API_URL = "https://api.anysearch.com/v1/search"
ANYSEARCH_GENERAL_TAG = "general.general"
TAVILY_API_URL = "https://api.tavily.com/search"
PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"

_DEFAULT_TIMEOUT = 30
_SNIPPET_MAX_CHARS = 1000
# 中文及相关字符区（含 CJK 标点、全角形式）：用来判断一段文字是不是中文。
_CJK_RE = re.compile(r"[\u2e80-\ua4cf\uff00-\uffef]")
# 一段连续的 Latin-1 字符（0x80–0xFF）：双重编码的 UTF-8 字节读出来就是这个样子。
_MOJIBAKE_RUN_RE = re.compile(r"[\x80-\xff]+")


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

ANYSEARCH_API_URL = "https://api.anysearch.com/v1/search"
ANYSEARCH_GENERAL_TAG = "general.general"
TAVILY_API_URL = "https://api.tavily.com/search"
PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"

_DEFAULT_TIMEOUT = 30
_SNIPPET_MAX_CHARS = 1000


@dataclass
class SearchResult:
    """一条搜索结果：标题、来源链接、摘要、发布时间（有就留，没有就空）。"""

    title: str
    url: str
    snippet: str = ""
    published_at: str = None
    score: float = None

    def to_dict(self):
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "score": self.score,
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
        }


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        out.append(
            SearchResult(
                title=_repair_mojibake(item.get("title") or url),
                url=url,
                snippet=_repair_mojibake(str(snippet))[:_SNIPPET_MAX_CHARS],
                published_at=published_at,
                score=item.get("score"),
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
#   普通查询 → AnySearch → Parallel → Tavily
#   结构化信息 → Tavily → AnySearch → Parallel
_CHAINS = {
    "normal": ["anysearch", "parallel", "tavily"],
    "structured": ["tavily", "anysearch", "parallel"],
}
# 交叉验证的搭档：主服务是谁，就用另一个服务的结果互相印证。
_CROSS_CHECK_PARTNER = {"anysearch": "tavily", "tavily": "anysearch", "parallel": "anysearch"}


def _domain_of(url):
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc.lower()
    except Exception:
        return ""


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


class SearchService:
    """统一搜索调度器：按任务路由、失败切备选、必要时交叉验证。

    providers 可注入（测试用假服务模拟失败，直接测备选链路）；
    不传就用默认的三个真实服务（Key 从环境变量读）。
    """

    def __init__(self, anysearch=None, tavily=None, parallel=None):
        self.providers = {
            "anysearch": anysearch or AnySearchProvider(),
            "tavily": tavily or TavilyProvider(),
            "parallel": parallel or ParallelProvider(),
        }

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

    def search(self, query, kind="normal", max_results=5, cross_check=False, force_provider=None):
        """搜一次。kind 是 normal（普通查询）或 structured（结构化信息）；
        force_provider 直接指定只用某一个服务（备选链路测试用）；
        cross_check 为真时，再用搭档服务的结果互相印证。"""
        if kind not in _CHAINS:
            raise ToolError(f"未知的搜索类型：{kind}（只认 normal / structured）")
        if not (query or "").strip():
            raise ToolError("查询不能为空")
        if force_provider is not None:
            if force_provider not in self.providers:
                raise ToolError(f"未知的搜索服务：{force_provider}")
            chain = [force_provider]
        else:
            chain = _CHAINS[kind]
        attempts = []
        outcome = None
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
            try:
                candidate = provider.search(query, max_results=max_results)
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
            outcome = candidate
            attempts.append({"provider": name, "status": "ok"})
            break
        if outcome is None:
            failed = [a for a in attempts if a["status"] == "failed"]
            if failed:
                error = "三个搜索服务都试过了，都失败了： " + "；".join(
                    f"{a['provider']}（{a['reason']}）" for a in failed
                )
                status = "failed"
            else:
                error = "三个搜索服务都试过了，都没找到结果"
                status = "no_results"
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
        if not outcome.fetched_at:
            outcome.fetched_at = _now_iso()
        outcome.attempts = attempts
        corroborated = []
        if cross_check:
            partner_name = _CROSS_CHECK_PARTNER.get(outcome.provider)
            partner = self.providers.get(partner_name) if partner_name else None
            if partner is not None and partner.available:
                try:
                    partner_outcome = partner.search(query, max_results=max_results)
                    if partner_outcome.status == "ok":
                        seen_urls = {r.url for r in outcome.results}
                        merged = list(outcome.results)
                        for result in partner_outcome.results:
                            if result.url not in seen_urls:
                                merged.append(result)
                                seen_urls.add(result.url)
                        primary_domains = {_domain_of(r.url) for r in outcome.results}
                        partner_domains = {
                            _domain_of(r.url) for r in partner_outcome.results
                        }
                        corroborated = sorted(primary_domains & partner_domains)
                        outcome.results = merged
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
            "notes": _quality_notes(outcome.results, corroborated),
        }
        outcome.attempts = attempts
        return outcome
