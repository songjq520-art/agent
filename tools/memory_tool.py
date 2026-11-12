# tools/memory_tool.py（记忆管理工具：把长期记忆的查/记/改/删接进聊天）
# 作用：45 号——聊天里只接了搜索，BOOS 让删没用的记忆时十二只能说
#       “没有能直接删长期记忆的工具”。现在把记忆管理也接进来：
#       列出全部记忆、记一条、改一条、删一条。
# 原理：和搜索工具同一套路（见 tools/search_tool.py）——
#       ContextVar 绑记忆库（每轮/每次请求有自己的绑定，不串线），
#       register_memory_tools() 登记四个工具，网关按登记执行。
#       列是只读（low、不确认）；记是 BOOS 当轮明确让记的（medium、
#       不确认，和“记住”直存同一条路）；改和删是 medium、要确认——
#       模型调了先弹确认卡，BOOS 点了确认才执行，取消了就不执行。
#       删的守卫（mem0_store.delete_memory 的三道关：归属校验、
#       BOOS 同意、验干净）一个不少——工具里 approve=True 只是
#       “网关确认过了”的意思，确认卡没点之前网关根本不会调到这里；
#       删之前先拿 deletion_scope 对一遍范围，删完把结果写清楚。
#       改走 correct_memory（BOOS 亲口纠正的那条路，带安全网和验）。
#       删/改都要先调 list_memories 列出全部、把原文一字不差地抄来——
#       原文对不上库里存的就不动手，防列表旧了、抄错了改错删错；
#       确认卡上显示的就是这份原文，BOOS 点之前看得清删改的是哪条。
# 关系：tools/tool_registry.py 管登记；tools/tool_gateway.py 调 func；
#       记忆库本身（mem0_store）由调用方绑进来——网页 ask() /
#       resolve_approval() 绑 ChatState 的库，命令行 run_question
#       绑传进来的库；没绑时工具报“记忆库没启用”，不崩、不瞎编。

import contextvars

from tools.tool_registry import get_tool, register_tool

# 绑定的记忆库：用 ContextVar 存，不用模块全局字典——每个任务
# （每个线程 / 每个上下文）有自己的绑定，两个任务并行时各用各的，
# 不会串线。网关跑工具时把调用方的上下文拷进工作线程
# （见 tool_gateway._run_with_timeout），所以线程里跑 list_memories
# 这类只读工具也能读到同一份绑定；改/删是同步跑的，直接在调用方
# 线程里执行，绑定一样在。
# 注意：新线程默认是空上下文，读不到调用方绑的库；没绑时工具直接
# 报错，不猜、不自己新建一个库。
_STORE_VAR = contextvars.ContextVar("tools_memory_store", default=None)


def set_memory_store(store):
    """换绑定的记忆库（网页每次请求绑 ChatState 的库，命令行绑传进来的库）。
    传 None 时解绑。返回 token，调 reset_memory_store(token) 恢复。
    只改当前上下文里的绑定，不影响别的任务。"""
    return _STORE_VAR.set(store)


def reset_memory_store(token):
    """恢复 set_memory_store 之前的绑定（配对使用，别漏了）。"""
    _STORE_VAR.reset(token)


def get_memory_store():
    """当前上下文绑定的记忆库（没绑过返回 None）。"""
    return _STORE_VAR.get()


class bound_memory_store:
    """with 语句绑记忆库：进块时绑上，出块时恢复（没传库时等于没绑，
    不覆盖外面已有的绑定——网页 ask() 绑了库之后，里面 run_question
    再绑一次 None 也不会把外面的顶掉）。

    用法：with bound_memory_store(store): ...
    """

    def __init__(self, store):
        self._store = store
        self._token = None

    def __enter__(self):
        if self._store is not None:
            self._token = set_memory_store(self._store)
        return self

    def __exit__(self, *exc_info):
        if self._token is not None:
            reset_memory_store(self._token)
            self._token = None
        return False


def _same_text(a, b):
    """两段记忆文本算不算同一条：去首尾空白和句末标点再比。

    删/改之前拿模型传来的原文和库里存的原文对一遍——对不上说明
    模型拿的是旧列表、或抄错了，拒掉，让它重新列出再动手，绝不猜一条改掉。
    """
    def _norm(text):
        return (text or "").strip().strip("。！？；，、.!?;, \t\n")

    norm_a = _norm(a)
    norm_b = _norm(b)
    return bool(norm_a) and norm_a == norm_b


def _store_or_error():
    """取绑定的记忆库，没绑时返回（None，错误包络），调用方直接返回它。"""
    store = get_memory_store()
    if store is None:
        return None, {
            "ok": False,
            "data": None,
            "error": "记忆库没启用：这次对话没带长期记忆。",
        }
    return store, None


def _run_list_memories(args):
    """列出全部长期记忆（只读）。返回 {count, memories: [{id, text,
    created_at, source, confirmed}]}。敏感记忆不列出来（get_all 的默认
    口径，不含敏感信息），但 count 是库里真正的总数（含敏感记忆，只
    计数、不列出——模型回答“一共几条”时不能少算）。"""
    store, error = _store_or_error()
    if error is not None:
        return error
    try:
        items = store.get_all()
        total = len(store.get_all(include_sensitive=True))
    except Exception as exc:  # noqa: BLE001 - 读库失败也算执行失败
        return {"ok": False, "data": None, "error": f"读记忆库失败：{exc}"}
    entries = []
    for item in items or []:
        meta = (item or {}).get("metadata") or {}
        entries.append(
            {
                "id": (item or {}).get("id") or "",
                "text": (item or {}).get("text") or "",
                "created_at": meta.get("created_at"),
                "source": meta.get("source"),
                "confirmed": bool(meta.get("user_confirmed")),
            }
        )
    return {
        "ok": True,
        "data": {"count": total, "memories": entries},
        "error": None,
    }


def _run_add_memory(args):
    """往长期记忆里加一条 BOOS 明确让记的事（add_memory_text 直存，
    来源记 user、已确认，自带去重）。返回 {memory_id, text, duplicate}
    （duplicate 表示库里已经有同一条，没重复存）。"""
    store, error = _store_or_error()
    if error is not None:
        return error
    text = ((args or {}).get("text") or "").strip()
    if not text:
        return {"ok": False, "data": None, "error": "要记的内容是空的，没法记。"}
    try:
        result = store.add_memory_text(text, source="user")
    except Exception as exc:  # noqa: BLE001 - 存失败也算执行失败
        return {"ok": False, "data": None, "error": f"存记忆失败：{exc}"}
    results = (result or {}).get("results") or []
    if not results:
        return {
            "ok": False,
            "data": None,
            "error": "没存进去（可能内容脱敏后什么都不剩，或存的时候出了问题）。",
        }
    first = results[0] or {}
    return {
        "ok": True,
        "data": {
            "memory_id": first.get("id"),
            "text": text,
            "duplicate": bool((result or {}).get("duplicate")),
        },
        "error": None,
    }


def _run_update_memory(args):
    """改一条长期记忆的说法（走 correct_memory，BOOS 亲口纠正的那条路，
    带安全网和改后验）。调用方必须先调 list_memories，把 id、旧原文、
    新说法一起传来；旧原文对不上库里的就不改。"""
    store, error = _store_or_error()
    if error is not None:
        return error
    memory_id = (args or {}).get("memory_id") or ""
    old_text = (args or {}).get("old_text") or ""
    new_text = ((args or {}).get("new_text") or "").strip()
    if not memory_id or not old_text.strip() or not new_text:
        return {
            "ok": False,
            "data": None,
            "error": "改记忆要传 memory_id、old_text（列出来的旧原文）、"
            "new_text（新说法），缺了没法改。",
        }
    try:
        owned = {
            item["id"]: item
            for item in store.get_all(include_sensitive=True)
        }
    except Exception as exc:  # noqa: BLE001 - 读库失败也算执行失败
        return {"ok": False, "data": None, "error": f"读记忆库失败：{exc}"}
    item = owned.get(memory_id)
    if item is None:
        return {
            "ok": False,
            "data": None,
            "error": f"找不到这条记忆（id 对不上）：{memory_id}。请先列出全部记忆再改。",
        }
    stored_text = item.get("text") or ""
    if not _same_text(stored_text, old_text):
        return {
            "ok": False,
            "data": None,
            "error": "旧原文对不上库里存的那条（可能列表旧了、或抄错了），没改。"
            "请先列出全部记忆，照原文再改。",
        }
    try:
        outcome = store.correct_memory(stored_text, new_text)
    except Exception as exc:  # noqa: BLE001 - 改失败也算执行失败
        return {"ok": False, "data": None, "error": f"改记忆失败：{exc}"}
    status = (outcome or {}).get("status")
    if status == "updated":
        return {
            "ok": True,
            "data": {
                "memory_id": (outcome or {}).get("memory_id") or memory_id,
                "old_text": stored_text,
                "new_text": new_text,
            },
            "error": None,
        }
    if status == "unchanged":
        return {
            "ok": True,
            "data": {
                "memory_id": memory_id,
                "old_text": stored_text,
                "new_text": new_text,
                "unchanged": True,
            },
            "error": None,
        }
    reason = (outcome or {}).get("reason") or f"改不了（{status}）"
    return {"ok": False, "data": None, "error": reason}


def _run_delete_memory(args):
    """删一条长期记忆（深度删除：向量 + history 审计行都清掉）。

    调用方必须先调 list_memories，把 id、原文一起传来；原文对不上
    库里的就不删。删之前先拿 deletion_scope 对一遍范围（删哪条、
    多少行审计记录），删完把结果写清楚。
    approve=True 只是“网关确认过了”的意思：这个工具登记了
    needs_confirmation=True，确认卡没点之前网关根本不会调到这里；
    归属校验、验干净那两道关还在 delete_memory 里面，一道都不少。
    """
    store, error = _store_or_error()
    if error is not None:
        return error
    memory_id = (args or {}).get("memory_id") or ""
    text = (args or {}).get("text") or ""
    if not memory_id or not text.strip():
        return {
            "ok": False,
            "data": None,
            "error": "删记忆要传 memory_id 和 text（列出来的原文），缺了没法删。",
        }
    try:
        scope = store.deletion_scope(memory_id)
    except Exception as exc:  # noqa: BLE001 - 读范围失败也算执行失败
        return {"ok": False, "data": None, "error": f"读删除范围失败：{exc}"}
    if not (scope or {}).get("found"):
        return {
            "ok": False,
            "data": None,
            "error": f"找不到这条记忆（id 对不上）：{memory_id}。请先列出全部记忆再删。",
        }
    stored_text = ((scope or {}).get("memory") or {}).get("text") or ""
    if not _same_text(stored_text, text):
        return {
            "ok": False,
            "data": None,
            "error": "原文对不上库里存的那条（可能列表旧了、或抄错了），没删。"
            "请先列出全部记忆，照原文再删。",
        }
    try:
        deleted = store.delete_memory(memory_id, approve=True)
    except PermissionError:
        return {
            "ok": False,
            "data": None,
            "error": "删除需要 BOOS 同意，这次没通过确认，删不了。",
        }
    except RuntimeError as exc:
        return {"ok": False, "data": None, "error": f"删了一半或删失败：{exc}"}
    except Exception as exc:  # noqa: BLE001 - 删失败也算执行失败
        return {"ok": False, "data": None, "error": f"删记忆失败：{exc}"}
    if not deleted:
        return {
            "ok": False,
            "data": None,
            "error": "这条记忆已经不在了（可能刚才已经删过）。",
        }
    return {
        "ok": True,
        "data": {
            "memory_id": memory_id,
            "text": stored_text,
            "history_cleared": (scope or {}).get("history_count") or 0,
        },
        "error": None,
    }


def _register_one(name, description, parameters, output, risk_level,
                  needs_confirmation, func, replay_safe, side_effect_free):
    """登记一个记忆工具（已登记过时跳过，重复调不报错）。

    replay_safe / side_effect_free 每项显式写清楚，不推导：
    只读的列（list_memories）才能重跑、才能走线程超时；会改东西的
    记/改/删一律不可重跑、同步跑（见 tool_gateway.py 的执行规则）。
    """
    if get_tool(name) is not None:
        return
    register_tool(
        name=name,
        description=description,
        parameters=parameters,
        output=output,
        risk_level=risk_level,
        needs_confirmation=needs_confirmation,
        tier2=False,
        replay_safe=replay_safe,
        side_effect_free=side_effect_free,
        timeout=60,
    )(func)


def register_memory_tools():
    """把四个记忆管理工具登记进工具注册表（已登记过时什么都不做，
    重复调不会报错）。列是只读（low、不确认、可重跑）；记是 BOOS
    当轮明确让记的（medium、不确认、会改东西、不可重跑）；改和删
    是 medium、要确认、会改东西、不可重跑——确认卡没点之前不执行，
    执行过一次也绝不重执行（恢复时喂存下来的结果）。"""
    _register_one(
        name="list_memories",
        description=(
            "列出 BOOS 长期记忆里的全部记忆（每条带 id、原文、形成时间、来源）。"
            "BOOS 让你看看他记了什么、有哪些没用的、或者要改/删记忆之前，"
            "先调这个列出全部再动手，别只凭提问前检索到的几条就改删。"
            "返回 {count, memories: [{id, text, created_at, source, confirmed}]}。"
        ),
        parameters={"type": "object", "properties": {}},
        output=(
            "成功时 data 是 {count: 一共几条, memories: "
            "[{id, text, created_at, source, confirmed}]}；"
            "失败时 ok 为 false，error 里写原因。"
        ),
        risk_level="low",
        needs_confirmation=False,
        func=_run_list_memories,
        replay_safe=True,
        side_effect_free=True,
    )
    _register_one(
        name="add_memory",
        description=(
            "往长期记忆里加一条 BOOS 明确让记的事。只在 BOOS 明确说"
            "“记住…”、“记下来”、“帮我记住”时用；别自己决定把对话内容"
            "存成长久记忆（对话结束会自动提取）。text 写 BOOS 让记的原话。"
            "返回 {memory_id, text, duplicate}（duplicate 表示库里已经有"
            "同一条，没重复存）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "BOOS 让记的原话",
                    "minLength": 1,
                    "maxLength": 1000,
                },
            },
            "required": ["text"],
        },
        output=(
            "成功时 data 是 {memory_id, text, duplicate}；"
            "失败时 ok 为 false，error 里写原因。"
        ),
        risk_level="medium",
        needs_confirmation=False,
        func=_run_add_memory,
        replay_safe=False,
        side_effect_free=False,
    )
    _register_one(
        name="update_memory",
        description=(
            "改一条长期记忆里的说法。先调 list_memories 列出全部，找到要改的"
            "那条，把 memory_id、old_text（列出来的旧原文，一字不差地抄）、"
            "new_text（新说法）一起传来；旧原文对不上库里的就不改。"
            "改之前会先弹确认卡，BOOS 点了确认才执行，取消了就不改。"
            "返回 {memory_id, old_text, new_text}。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "要改的那条记忆的 id（list_memories 里拿到的）",
                    "minLength": 1,
                },
                "old_text": {
                    "type": "string",
                    "description": "列出来的旧原文，一字不差地抄",
                    "minLength": 1,
                    "maxLength": 1000,
                },
                "new_text": {
                    "type": "string",
                    "description": "新说法",
                    "minLength": 1,
                    "maxLength": 1000,
                },
            },
            "required": ["memory_id", "old_text", "new_text"],
        },
        output=(
            "成功时 data 是 {memory_id, old_text, new_text}；"
            "失败时 ok 为 false，error 里写原因。"
        ),
        risk_level="medium",
        needs_confirmation=True,
        func=_run_update_memory,
        replay_safe=False,
        side_effect_free=False,
    )
    _register_one(
        name="delete_memory",
        description=(
            "删一条长期记忆。先调 list_memories 列出全部，找到要删的那条，"
            "把 memory_id、text（列出来的原文，一字不差地抄）一起传来；"
            "原文对不上库里的就不删。删之前会先弹确认卡，BOOS 点了确认才"
            "执行，取消了就不删；删了就彻底没了（向量和审计记录都清掉），"
            "别删错。返回 {memory_id, text, history_cleared}。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "要删的那条记忆的 id（list_memories 里拿到的）",
                    "minLength": 1,
                },
                "text": {
                    "type": "string",
                    "description": "列出来的原文，一字不差地抄",
                    "minLength": 1,
                    "maxLength": 1000,
                },
            },
            "required": ["memory_id", "text"],
        },
        output=(
            "成功时 data 是 {memory_id, text, history_cleared}；"
            "失败时 ok 为 false，error 里写原因。"
        ),
        risk_level="medium",
        needs_confirmation=True,
        func=_run_delete_memory,
        replay_safe=False,
        side_effect_free=False,
    )
