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
import copy
import json

from tools.date_numbers import (
    dynamic_numeric_fact_kind,
    extract_dated_numbers,
    is_dynamic_numeric_question,
    parse_number_value,
    resolve_requested_date,
)
from tools.search import (
    SearchService,
    check_numeric_consensus,
    is_market_denylisted_domain,
    market_source_tier,
    registrable_domain,
)
from tools.tool_registry import get_tool, register_tool

# 提速 4 号（2026-11-13，BOOS 批准）：搜索工具自己的输出上限——
#   整份 data 序列化后不超过 5000 字（在 4000–6000 的区间里取中）。
#   循环那边的通用上限是 8000，这里更严一档：搜索结果带 extract 后
#   变长了，先在这里截，模型一次少读几千字，少跑一轮“再去读正文”。
SEARCH_TOOL_OUTPUT_MAX_CHARS = 5000


def _cap_search_output(data, max_chars=SEARCH_TOOL_OUTPUT_MAX_CHARS):
    """搜索工具自己的输出上限（提速 4 号）：整份 data 序列化后
    不超过 max_chars（默认 5000）。截的时候优先保 extract、
    published_at、url、dated_numbers（50/54 号证据绝不能截掉）；
    先缩摘要（从后往前，每条先缩到 200 字、再清掉），
    还超就从后往前整条去掉（至少留 1 条）。
    保的字段本身就超了上限时（比如单条的 extract + dated_numbers
    就很长），截不下来——这时原样返回，循环那边的 8000 上限是兜底，
    绝不为了凑上限去砍证据。
    截了东西时，在 data["notes"] 里留一句“已截断到 5000 字以内”，
    模型和 BOOS 都看得到——静默截断会让模型以为结果就这么点。
    返回截好的 data（原 data 不动，复制一份再改）。"""
    capped = copy.deepcopy(data)

    def _size(payload):
        return len(json.dumps(payload, ensure_ascii=False))

    if _size(capped) <= max_chars:
        return capped
    truncated = False
    results = capped.get("results") or []
    # 先缩摘要：从最后一条往前，每条先缩到 200 字，再清掉。
    for limit in (200, 0):
        for item in reversed(results):
            if _size(capped) <= max_chars:
                break
            snippet = item.get("snippet") or ""
            if len(snippet) > limit:
                item["snippet"] = snippet[:limit]
                truncated = True
        if _size(capped) <= max_chars:
            break
    # 还超：从后往前整条去掉，至少留 1 条。
    while len(results) > 1 and _size(capped) > max_chars:
        results.pop()
        truncated = True
    if truncated:
        notes = capped.get("notes") or []
        notes.append(
            "搜索结果太长，已截断了（按 5000 字上限，先缩摘要、再整条去掉；"
            "extract、发布时间、链接、数字-日期对照没动；"
            "被截掉的数据不可执行，也不要猜）"
        )
        capped["notes"] = notes
    return capped

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
    只改当前上下文里的绑定，不影响别的任务。
    返回 ContextVar 的 Token（测试里用它复位）。"""
    return _SERVICE_VAR.set(service or SearchService())


def get_search_service():
    """当前上下文绑定的搜索服务（没绑过时建一个默认的）。"""
    service = _SERVICE_VAR.get()
    if service is None:
        service = SearchService()
        _SERVICE_VAR.set(service)
    return service


def _dated_number_entry(dn):
    """一条 dated_numbers 的规范形状：number / value / kind / date。
    进来的数据没带 value/kind 时，从 number 文本里现算
   （parse_number_value），不留 None——双源印证要拿它比数。"""
    number = dn.get("number")
    value = dn.get("value")
    kind = dn.get("kind")
    if value is None or kind is None:
        parsed = parse_number_value(number or "")
        if parsed is not None:
            parsed_kind, parsed_value = parsed
            kind = kind if kind is not None else parsed_kind
            value = value if value is not None else parsed_value
    return {"number": number, "value": value, "kind": kind, "date": dn.get("date")}


def _validate_search_data(data):
    """格式层校验：搜索结果的 data 必须是字典，带 status；
    results 必须是列表，每条带 title 和 url（缺了就整条丢弃）。
    返回整理好的 data（脏条目已去掉）。
    28/48 号：查询词、实际发出的查询词、日期过滤情况、质量备注也留下来——
    模型回答时要知道“拿什么词搜的、结果是哪天的、哪些去掉了、哪些没带时间”，
    以前质量备注（比如“30 天以前的可能过时了”）在这里被丢掉，模型看不到。
    """
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
                # 提速 4 号：正文摘要（前 2–3 条才有），截输出时优先保它。
                "extract": item.get("extract") or "",
                "published_at": item.get("published_at"),
                # 50/54 号：数字-日期对照（程序标的，只留配上日期的）。
                # 没配上的数不在里面——模型引用数字时只认这里面的，
                # 对不上要问的那天就不算证据。
                # 判断 (b) 号：value（数值）和 kind（价格/百分比/点位）也留着，
                # 双源印证要拿它比两个来源的数对不对得上。
                # 进来的 dated_numbers 没带 value/kind 时（比如测试手写的数据），
                # 从 number 文本里现算，不留 None。
                "dated_numbers": [
                    _dated_number_entry(dn)
                    for dn in (item.get("dated_numbers") or [])
                    if isinstance(dn, dict) and dn.get("number") and dn.get("date")
                ] or [
                    {
                        "number": entry["number"],
                        "value": entry["value"],
                        "kind": entry["kind"],
                        "date": entry["date"],
                    }
                    for entry in extract_dated_numbers(
                        f"{title}. {item.get('snippet') or ''}. "
                        f"{item.get('extract') or ''}"
                    )
                    if entry["date"] is not None
                ],
            }
        )
    quality = data.get("quality") or {}
    notes = quality.get("notes") or []
    return {
        "status": status,
        "results": clean_results,
        "answer": data.get("answer"),
        "provider": data.get("provider", ""),
        "query": data.get("query") or "",
        "executed_query": data.get("executed_query") or data.get("query") or "",
        "date_range": data.get("date_range") or {},
        "notes": [str(note) for note in notes if note],
    }


def _apply_market_source_preference(data, target_date):
    """行情问题：权威财经来源优先，黑名单来源不做唯一证据
    （2026-11-13，BOOS 定：以后类似的搜索要权威报道）。
    起因：问“纳指 11 月 11 日收盘多少”时，搜索只回来 Gate
    （gate.com，币圈交易所新闻）一篇，它自己把 11 月 10 日的
    收盘 25,065.87 写成 11 月 11 日的——来源自己错，日期配对
    再对也拦不住。只对行情（market）问题开；天气/汇率/其他
    问题不受影响（不排序、不拿数）。
    - 结果按来源档位排序：权威财经域名在前，黑名单域名在后
      （稳定排序，同档保持原顺序；截断从后往前截，黑名单先被截掉）。
    - 目标日期的数如果只有黑名单来源有（没有任何权威来源、也
      没有普通来源），就把那天的 dated_numbers 从结果里拿掉，
      并留一句说明——模型没有可引用的数，只能说无法核实，不许把
      黑名单来源的数当收盘价给。普通来源（既不权威也不在黑名单，
      比如一般新闻站）照常用，50/54 号的日期校验还认它们。
      51 号门禁不受影响（它认搜索记录，不认 dated_numbers）；
      50/54 校验没数可对时不拦，溯源会把回答里的数标成未核实。
    返回改好的 data（原 data 不动，复制一份再改）。"""
    out = copy.deepcopy(data)
    results = out.get("results") or []
    # 排序：权威在前，黑名单在后。
    results.sort(
        key=lambda item: market_source_tier(
            registrable_domain(item.get("url") or "")
        )
    )
    out["results"] = results
    if not target_date:
        return out
    # 目标日期的数，是不是只有黑名单来源有？
    # 只有黑名单来源有（没有任何权威来源、也没有普通来源）时，
    # 才把那天的 dated_numbers 拿掉——黑名单站绝不能只凭它一家
    # 就把数当收盘价给出去。普通来源（既不权威也不在黑名单，
    # 比如一般新闻站）照常用，50/54 号的日期校验还认它们；
    # 有权威来源时更不用拿（权威排前面，印证照算）。
    has_non_denylisted = False
    has_any = False
    for item in results:
        domain = registrable_domain(item.get("url") or "")
        for entry in item.get("dated_numbers") or []:
            if entry.get("date") != target_date:
                continue
            has_any = True
            if not is_market_denylisted_domain(domain):
                has_non_denylisted = True
                break
        if has_non_denylisted:
            break
    if not has_any or has_non_denylisted:
        return out
    # 只有黑名单来源有那天的数：把目标日期的 dated_numbers 拿掉，并留说明。
    stripped_domains = set()
    for item in results:
        kept = []
        for entry in item.get("dated_numbers") or []:
            if entry.get("date") == target_date:
                stripped_domains.add(
                    registrable_domain(item.get("url") or "")
                )
                continue
            kept.append(entry)
        item["dated_numbers"] = kept
    if stripped_domains:
        notes = out.get("notes") or []
        names = "、".join(sorted(stripped_domains))
        notes.append(
            f"注意：{target_date} 的收盘数没有任何权威财经来源"
            "（雅虎财经、英为财情、MarketWatch、CNBC、路透、彭博、"
            "华尔街日报等），只有黑名单来源（"
            f"{names}）有这个数，无法核实——不要把这个数当收盘价给，"
            "要说无法核实/不知道。"
        )
        out["notes"] = notes
    return out


def _run_search(args):
    """真正执行搜索的函数（签名 func(args: dict) -> dict，网关调它）。
    args：{"query": 要查的话（必填）, "kind": "normal"（默认）或
    "structured"（结构化问题）, "max_results": 最多几条（默认 5）,
    "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}（可选，
    28 号：问具体某天时把日期放这里，结果按 published_at 过滤到这几天）}。
    """
    query = (args or {}).get("query") or ""
    kind = (args or {}).get("kind") or "normal"
    max_results = (args or {}).get("max_results") or 5
    date_range = (args or {}).get("date_range") or None
    # 没日期就不传 date_range 参数：不是所有搜索服务都认它（比如测试里
    # 的假服务），没传就别硬塞一个 None 过去；有日期时再传。
    kwargs = {"kind": kind, "max_results": max_results}
    if date_range is not None:
        kwargs["date_range"] = date_range
    # 判断 (b) 号：问的是动态数字事实（收盘价、气温、汇率……）时，
    # 开 cross_check——再用搭档服务搜一次，两家来源互相印证。
    # 代价是多烧一份搜索额度（Tavily 每月 1000 credits，搭档那路
    # 另外算），只对动态数字问题开，普通问题不开。
    is_numeric = is_dynamic_numeric_question(query)
    if is_numeric:
        kwargs["cross_check"] = True
    try:
        outcome = get_search_service().search(query, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 搜索服务抛错也算执行失败
        return {"ok": False, "data": None, "error": f"搜索服务出错：{exc}"}
    data = _validate_search_data(outcome.to_dict())
    if data["status"] == "failed":
        return {
            "ok": False,
            "data": None,
            "error": outcome.error or data.get("error") or "搜索失败，原因不明",
        }
    # 目标日期：date_range 是单日就用它；否则从查询词里解析
    #（“昨天”“11 月 10 日”）；解析不出就不算（consensus 不附）。
    # 先算出来——行情来源偏好和双源印证都要用它。
    target_date = None
    if is_numeric:
        if date_range and date_range.get("start") == date_range.get("end"):
            target_date = date_range.get("start")
        else:
            resolved = resolve_requested_date(query)
            if resolved is not None:
                target_date = resolved.isoformat()
    # 权威报道优先（行情问题才开）：权威财经来源排前面，黑名单排最后；
    # 目标日期的数如果没有任何权威来源，就把那天的数拿掉、留说明。
    if dynamic_numeric_fact_kind(query) == "market":
        data = _apply_market_source_preference(data, target_date)
    # 提速 4 号：搜索工具自己的输出上限（5000 字），截的时候
    # 优先保 extract、published_at、url、dated_numbers。
    data = _cap_search_output(data)
    # 判断 (b) 号：动态数字问题，算一算两个独立来源对不对得上。
    # 放在截断之后算——印证只认模型实际看到的那几条结果。
    # 气温 tolerance ±1°C，收盘价精确到小数点后两位。
    # 行情问题：黑名单来源不进印证（两家黑名单对上同一个错数
    # 也不算印证——Gate 那种来源自己错，互相印证不了）。
    if is_numeric and target_date:
        kind_label = dynamic_numeric_fact_kind(query)
        tolerance = 1.0 if kind_label == "weather" else 0.01
        consensus_results = data.get("results") or []
        if kind_label == "market":
            consensus_results = [
                item
                for item in consensus_results
                if not is_market_denylisted_domain(
                    registrable_domain(item.get("url") or "")
                )
            ]
        data["consensus"] = check_numeric_consensus(
            consensus_results, target_date, tolerance=tolerance
        )
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
            "问具体某天（比如“昨天纳指收盘多少”“11 月 10 日纳指多少”）时，"
            "先把“昨天/11 月 10 日”换算成具体日期（系统消息里给了今天是哪天），"
            "再把日期放进 date_range 里查——结果会按 published_at 过滤到这几天。"
            "每条结果带 dated_numbers（数字-日期对照，程序标的）：引用收盘价/"
            "涨跌幅时，只用日期和要问的那天对得上的那条；不同日期的数不算冲突，"
            "别让它拉低你对正确答案的把握；没有要问的那天的数就说不知道，"
            "不许拿别的日子的数顶上。"
            "问动态数字（收盘价、气温、汇率……）时，结果里还带 consensus"
            "（双源印证：两个独立来源对得上才是 consensus=true；只有一家"
            "来源或两家对不上时，数只能当未核实引用，要写明未核实）。"
            "问行情收盘时，优先用权威财经来源（雅虎财经、英为财情、"
            "MarketWatch、CNBC、路透、彭博、华尔街日报等）的数；结果里"
            "如果注明没有权威来源，就说无法核实，不要把非权威来源的数"
            "当收盘价给。"
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
                "date_range": {
                    "type": "object",
                    "description": "只查这几天：{\"start\": \"YYYY-MM-DD\", \"end\": \"YYYY-MM-DD\"}，问某一天时两边写同一天；不问具体日期时别传",
                    "properties": {
                        "start": {"type": "string", "description": "开始日期，YYYY-MM-DD"},
                        "end": {"type": "string", "description": "结束日期，YYYY-MM-DD"},
                    },
                },
            },
            "required": ["query"],
        },
        output=(
            "成功时 data 是 {status: ok/no_results, results: "
            "[{title, url, snippet, extract, published_at, dated_numbers}], "
            "answer, provider, query, executed_query, date_range, notes}；"
            "extract 是前 2–3 条结果的正文摘要（400–800 字，优先含日期/"
            "数字的段落）；dated_numbers 是数字-日期对照（50/54 号证据）；"
            "整份 data 序列化后不超过 5000 字（截的时候先缩摘要、再整条去掉，"
            "extract、published_at、url、dated_numbers 绝不截掉）；"
            "失败时 ok 为 false，error 里写原因。"
        ),
        risk_level="low",
        needs_confirmation=False,
        tier2=False,
        replay_safe=True,
        side_effect_free=True,
        timeout=60,
    )(_run_search)
