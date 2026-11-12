# tools/tool_registry.py（工具注册表）
# 作用：模型能调的工具都在这里登记，循环只认登记过的工具。
# 原理：一个字典 + 一个装饰器。新工具想进聊天，不用改循环，
#       在自己的模块里写好定义、调 register_tool() 登记就行；
#       没登记的工具，模型叫了也不执行（网关会拦）。
# 关系：tools/agent_loop.py 从这里取工具定义发给模型；
#       tools/tool_gateway.py 从这里取工具执行函数和风险等级。
# 字段说明（每条登记都要写全）：
#   name：工具名，模型调的就是这个名字；
#   description：给模型看的说明，写清楚工具是干什么的、什么时候用；
#   parameters：JSON Schema 格式的参数说明（OpenAI 工具定义同款写法），
#     模型按这个填参数，网关按这个校验；
#   output：返回结果的形状说明（给模型看，知道拿到的是什么）；
#   risk_level：风险等级，"low"（只读、可自动跑，如查天气）、
#     "medium"（会改东西或花钱，要先问 BOOS，如写文件）、
#     "high"（危险操作，永远不跑，如删系统文件）；
#   needs_confirmation：要不要先问 BOOS 再执行（True/False），
#     和 risk_level 一起决定网关放不放行；
#   tier2：是不是第二级禁令里的操作（True = 永远不许执行，
#     哪怕 BOOS 点了确认也不行；见 tools/permissions.py）；
#   replay_safe：重复执行同一份参数安不安全（True = 查天气这类只读操作，
#     确认超时重跑也没事；False = 写文件这类，重复跑会重复写）。
#     重复执行规则（网关按这个拦，见 tools/tool_gateway.py 的 can_replay）：
#     同一条工具调用（同一个 tool_call_id）执行过一次后，模型又发同一条
#     调用时，True 的允许再执行一次，False 的直接拒绝、不执行；
#     恢复/续跑时一律喂存下来的结果，不重执行——这条规则是真的关口，
#     不是只登记不用的摆设（见 tools/agent_loop.py 的 _execute_one）。
#   side_effect_free：这工具会不会改东西（True = 纯只读，什么都不改；
#     False = 会改东西，或者不确定）。和 replay_safe 是两个意思，
#     不能互相代替：只读操作重跑一次结果可能变了（比如实时搜索），
#     所以 replay_safe 未必是 True；反过来幂等的写操作重跑没事，
#     replay_safe 可以是 True，但它照样会改东西。网关拿这个字段
#     决定超时的跑法（见 tools/tool_gateway.py 的 execute）：
#     True 的走线程超时（超时了后台还在跑也没关系，反正什么都不改），
#     False 的同步跑（线程杀不掉，超时后后台接着改等于改两次）。
#     默认 False：没声明无副作用时，一律按会改东西处理，宁可慢，
#     不可错。replay_safe 的默认值还是 True，别动它。
#   timeout：执行超时秒数（默认 60 秒，见 tools/commands.py）；
#   func：真正干活的函数，签名 func(args: dict) -> dict，
#     返回 {"ok": True/False, "data": ..., "error": ...}。

# _REGISTRY：工具登记表，名字 -> 登记信息（不要在外面直接改它，
#   用 register_tool() 登记、用 get_tool() / list_tools() 读）。

_REGISTRY = {}


def register_tool(name, description, parameters, output, risk_level,
                  needs_confirmation, tier2=False, replay_safe=True,
                  side_effect_free=False, timeout=60, precheck=None):
    """登记一个工具。返回一个装饰器，装饰真正干活的函数。

    参数说明见文件头。risk_level 只认 "low" / "medium" / "high"，
    写错了直接报错，不让含糊的登记混进去。
    同一个名字登记两次也报错——重名会让模型分不清调的是哪个。
    precheck（57 号）：可选的确认前校验函数，签名 precheck(args)，
    返回 (error, augment)：error 是字符串时拒绝这次调用（不建确认卡、
    不执行，错误直接回给模型，让它重新列出再动手）；error 为 None
    时通过，augment 是要补进参数里的程序算好的字段（比如删记忆的
    匹配结果，确认卡上原样显示——卡只显示 args，不改卡片渲染）。
    precheck 只读、不改库，确认前和执行前各跑一次（执行前再验一遍，
    防卡建好后库变了）。
    """

    if risk_level not in ("low", "medium", "high"):
        raise ValueError(
            f"工具 {name} 的风险等级只认 'low' / 'medium' / 'high'，"
            f"不是 {risk_level!r}。"
        )
    if name in _REGISTRY:
        raise ValueError(f"工具 {name} 已经登记过了，不能重复登记。")

    def decorator(func):
        _REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "output": output,
            "risk_level": risk_level,
            "needs_confirmation": bool(needs_confirmation),
            "tier2": bool(tier2),
            "replay_safe": bool(replay_safe),
            "side_effect_free": bool(side_effect_free),
            "timeout": timeout,
            "precheck": precheck,
            "func": func,
        }
        return func

    return decorator


def get_tool(name):
    """按名字取一条登记，没有返回 None（网关用它判断“认不认这个工具”）。"""
    return _REGISTRY.get(name)


def list_tools():
    """所有已登记工具的列表（按登记顺序），返回副本，改它不影响登记表。"""
    return list(_REGISTRY.values())


def tool_definitions():
    """给模型看的工具定义列表（OpenAI 工具定义格式），循环发请求时用。

    只含 name / description / parameters 三项——模型填参数只需要这些，
    风险等级、要不要确认这些是网关的事，不用告诉模型。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in _REGISTRY.values()
    ]
