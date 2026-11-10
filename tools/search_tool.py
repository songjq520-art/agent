# tools/search_tool.py（搜索工具：把现成的搜索服务登记进工具注册表）
# 作用：模型要查东西时调的就是这个工具（名字 "search"）。
# 原理：登记时绑一个 SearchService（tools/search.py）；真正执行时调
#       它的 search()，再把 SearchOutcome 转成网关要的包络
#       {"ok": ..., "data": ..., "error": ...}。
#       翻译器（translator）由 tools/agent_loop.py 在每轮循环开始时
#       绑进来：用当轮的模型把中文查询翻成英文再搜（只翻 AnySearch
#       那一路，翻译费记工具账本）；没绑翻译器时直接用原文搜。
# 关系：tools/tool_registry.py 管登记；tools/tool_gateway.py 调 func 执行；
#       搜索服务本身（Tavily / Parallel / AnySearch 路由、质量检查）
#       都在 tools/search.py 里，这里不重复实现。
# 结果校验（执行层 + 格式层，目标层第二批再做）：
#   status 为 "failed" 时算执行失败（ok=False，error 里写清原因）；
#   status 为 "ok" / "no_results" 时算执行成功，data 里带 status、
#   结果列表（每条只留 title / url / snippet / published_at，
#   保证形状固定，模型不用猜）和 answer；结果条目缺 title 或 url
#   时整条丢弃，不把脏数据喂给模型。

import contextvars

from tools.search import SearchService
from tools.tool_registry import get_tool, register_tool

# 绑定的搜索服务：用 ContextVar 存，不用模块全局字典——每个任务
# （每个线程 / 每个上下文）有自己的服务，两个任务并行时各用各的，
# 不会串线。翻译器是绑在服务实例上的（见 agent_loop._bind_search_service，
# 每轮新建服务 + 全新翻译器），所以服务串了，翻译和记账上下文也会
# 跟着串——这就是为什么不能用全局。
# 注意：新线程默认是空上下文，读不到调用方绑的服务；网关跑工具时
# 会把调用方的上下文拷进工作线程（见 tool_gateway._run_with_timeout），
# 所以线程里跑搜索也能读到同一份绑定。
_SERVICE_VAR = contextvars.ContextVar("tools_search_service", default=None)


def set_search_service(service):
    """换绑定的搜索服务（循环开始绑带翻译器的，测试绑假的）。
    传 None 时恢复成默认的 SearchService（不带翻译器）。
    只改当前上下文里的绑定，不影响别的任务。"""
    _SERVICE_VAR.set(service or SearchService())


def get_search_service():
    """当前上下文绑定的搜索服务（没绑过时建一个默认的）。"""
    service = _SERVICE_VAR.get()
    if service is None:
        service = SearchService()
        _SERVICE_VAR.set(service)
    return service


def _validate_search_data(data):
    """格式层校验：搜索结果的 data 必须是字典，带 status；
    results 必须是列表，每条带 title 和 url（缺了就整条丢弃）。
    返回整理好的 data（脏条目已去掉）。"""
    if not isinstance(data, dict):
        return {"status": "failed", "results": [], "error": "搜索返回的形状不对"}
    status = data.get("status")
    if status not in ("ok", "no_results", "failed"):
        return {"status": "failed", "results": [], "error": f"搜索状态不明：{status!r}"}
    clean_results = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or ""
        url = item.get("url") or ""
        if not title or not url:
            continue
        clean_results.append(
            {
                "title": title,
                "url": url,
                "snippet": item.get("snippet") or "",
                "published_at": item.get("published_at"),
            }
        )
    return {
        "status": status,
        "results": clean_results,
        "answer": data.get("answer"),
        "provider": data.get("provider", ""),
    }


def _run_search(args):
    """真正执行搜索的函数（签名 func(args: dict) -> dict，网关调它）。
    args：{"query": 要查的话（必填）, "kind": "normal"（默认）或
    "structured"（结构化问题）, "max_results": 最多几条（默认 5）}。
    """
    query = (args or {}).get("query") or ""
    kind = (args or {}).get("kind") or "normal"
    max_results = (args or {}).get("max_results") or 5
    try:
        outcome = get_search_service().search(
            query, kind=kind, max_results=max_results
        )
    except Exception as exc:  # noqa: BLE001 - 搜索服务抛错也算执行失败
        return {"ok": False, "data": None, "error": f"搜索服务出错：{exc}"}
    data = _validate_search_data(outcome.to_dict())
    if data["status"] == "failed":
        return {
            "ok": False,
            "data": None,
            "error": outcome.error or data.get("error") or "搜索失败，原因不明",
        }
    return {"ok": True, "data": data, "error": None}


def register_search_tool():
    """把 "search" 登记进工具注册表（已登记过时什么都不做，
    重复调不会报错）。搜索是只读操作：风险 low，不用确认，
    重复执行同一份参数也安全。"""
    if get_tool("search") is not None:
        return
    register_tool(
        name="search",
        description=(
            "上网搜索实时信息（新闻、天气、股价、百科等）。"
            "模型自己的知识可能过时了，需要最新信息、或者用户明确问了"
            "“今天/现在/最新”这类问题时，用这个工具查。"
            "返回结果列表（标题、链接、摘要、发布时间）和一句话答案。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查的问题或关键词，用中文写就行",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "kind": {
                    "type": "string",
                    "description": "normal（普通问题）或 structured（需要结构化答案的问题，如对比、列表）",
                    "enum": ["normal", "structured"],
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回几条结果，默认 5",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
        output=(
            "成功时 data 是 {status: ok/no_results, results: "
            "[{title, url, snippet, published_at}], answer, provider}；"
            "失败时 ok 为 false，error 里写原因。"
        ),
        risk_level="low",
        needs_confirmation=False,
        tier2=False,
        replay_safe=True,
        side_effect_free=True,
        timeout=60,
    )(_run_search)
