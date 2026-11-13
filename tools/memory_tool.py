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


def _norm_anchor_text(text):
    """anchor/原文归一化：去首尾空白和句末标点，子串比对用。"""
    return (text or "").strip().strip("。！？；，、.!?;, \t\n")


def _find_anchor_matches(store, anchor):
    """库里哪些记忆的原文含有 anchor（归一化后子串比对）。
    返回 [(id, text)]，按库里的顺序排。anchor 为空时返回 []。"""
    norm = _norm_anchor_text(anchor)
    if not norm:
        return []
    out = []
    try:
        items = store.get_all(include_sensitive=True) or []
    except Exception:
        return []
    for item in items:
        text = (item or {}).get("text") or ""
        if norm in _norm_anchor_text(text):
            out.append(((item or {}).get("id") or "", text))
    return out


def guard_memory_target(action, memory_id, anchor_text, candidates):
    """统一守卫（判断 c 号）：删/改记忆时，程序核对模型挑的这条
    是不是用户要动的那条。模型只能在候选里挑，挑出范围就不行。

    参数：
    - action："delete" 或 "update"（别的动作直接拦）。
    - memory_id：模型挑的那条的 id。
    - anchor_text：用户原话里的目标（delete 时是 anchor_text 参数，
      比如“鸡杂面”；update 时是 old_text，就是要改的那条旧原文）。
    - candidates：[(id, text)]，库里原文含 anchor 的候选（调用方
      用 _find_anchor_matches 现读出来——执行前要重读，不能拿旧的）。
    返回 {"ok", "reason", "target_id", "needs_user", "choices"}：
    - ok=True：放行，target_id 是核对过的那条 id，reason 是一句
      通过原因。
    - ok=False：拦下，reason 是一句中文原因；needs_user=True 时
      是“有多条候选、要 BOOS 挑一个”，choices 是候选 [(id, text)]；
      needs_user=False 时是“程序就认得出不对”，不用问 BOOS。
    规则（57 号 delete 的行为原样保留，只把“核对”抽出来）：
    - action / memory_id / anchor 非法 → 拦。
    - 候选为空（anchor 在库里没命中）→ 拦。
    - 模型没挑 / 挑的 id 不在候选里 → 拦，附上候选让模型重新列出再挑。
    - delete：挑的这条在候选里就放行（多命中不拦，确认卡上会注明，
      BOOS 再看一眼——这是 57 号定下的行为，不改）。
    - update：anchor 就是旧原文，要求挑的这条原文和 anchor 完全
      一致（exact，用 _same_text 比）才放行；只有一条候选且是挑的
      这条时也放行（非 delete 允许一条部分匹配——执行时还会再验
      一次原文，不对就拒掉）；多条候选、挑的不是唯一的 exact 时，
      拦下让 BOOS 挑。
    """
    norm_anchor = _norm_anchor_text(anchor_text)
    if action not in ("delete", "update"):
        return {
            "ok": False, "reason": f"不支持的动作：{action}。",
            "target_id": None, "needs_user": False, "choices": [],
        }
    if not memory_id:
        return {
            "ok": False, "reason": "没给 memory_id，没法核对挑的是哪条。",
            "target_id": None, "needs_user": False, "choices": [],
        }
    if not norm_anchor:
        return {
            "ok": False, "reason": "没给 anchor，没法核对目标。",
            "target_id": None, "needs_user": False, "choices": [],
        }
    cand = list(candidates or [])
    if not cand:
        return {
            "ok": False,
            "reason": f"库里没找到含“{anchor_text}”的记忆。",
            "target_id": None, "needs_user": False, "choices": [],
        }
    picked = next((c for c in cand if c[0] == memory_id), None)
    if picked is None:
        return {
            "ok": False,
            "reason": "挑的这条不在候选里（原文里没有目标），挑错了。",
            "target_id": None, "needs_user": True, "choices": cand,
        }
    if action == "delete":
        return {
            "ok": True, "reason": "挑的这条原文里含目标。",
            "target_id": picked[0], "needs_user": False, "choices": cand,
        }
    # update：anchor 就是旧原文，要求精确一致。
    exact = [c for c in cand if _same_text(c[1], anchor_text)]
    if len(exact) == 1 and exact[0][0] == memory_id:
        return {
            "ok": True, "reason": "挑的这条原文和旧原文完全一致。",
            "target_id": picked[0], "needs_user": False, "choices": cand,
        }
    if len(cand) == 1 and cand[0][0] == memory_id:
        return {
            "ok": True,
            "reason": "只有一条候选，且是挑的这条（部分匹配）。",
            "target_id": picked[0], "needs_user": False, "choices": cand,
        }
    return {
        "ok": False,
        "reason": "多条候选、挑的不是唯一精确的那条，要 BOOS 挑一个。",
        "target_id": None, "needs_user": True, "choices": cand,
    }


def _check_delete_anchor(args):
    """删记忆的确认前校验（57 号 precheck）：程序核对模型挑的这条
    是不是用户要删的那条。
    返回 (error, augment)：error 非空时拒绝（不建卡、不执行，错误
    直接回给模型，让它重新列出再动手）；通过时 augment 带程序算好
    的匹配结果（确认卡上原样显示）。
    只做一件事：模型挑的那条（memory_id + text）的原文里必须含 anchor。
    09:01（拉面）和 09:03（Agent 目标）就是这么露的——挑的条目真实存在、
    原文逐字对得上，但原文里根本没有用户说的“鸡杂面”。
    多命中不拦：库里常有“关于同一件事的说法”（比如撤回句本身也含“鸡杂面”），
    命中多条不代表挑错了——只要挑的这条对得上 anchor 就放行，卡上会注明
    库里还有几条相关记忆，让 BOOS 再看一眼；挑的这条对不上才拒绝。
    只读、不改库，确认前和执行前各跑一次。
    核对走统一守卫 guard_memory_target（判断 c 号），这里只补 delete
    特有的两句：挑的条目原文必须和模型传来的 text 对得上（_same_text），
    以及确认卡上显示的匹配备注（57 号定的格式，测试钉死了，别改）。
    """
    store, error = _store_or_error()
    if error is not None:
        return error["error"], None
    memory_id = (args or {}).get("memory_id") or ""
    text = (args or {}).get("text") or ""
    anchor = (args or {}).get("anchor_text") or ""
    if not _norm_anchor_text(anchor):
        return (
            "删记忆要写 anchor_text：用户原话里的删除目标"
            "（比如“鸡杂面”），没写没法核对。",
            None,
        )
    matches = _find_anchor_matches(store, anchor)
    guard = guard_memory_target("delete", memory_id, anchor, matches)
    if not guard["ok"]:
        if not matches:
            return (
                f"库里没找到含“{anchor}”的记忆，没删。"
                "请先列出全部记忆，确认要删的是哪条。",
                None,
            )
        shown = "；".join(t for _i, t in matches[:3])
        more = f"（等共 {len(matches)} 条）" if len(matches) > 3 else ""
        return (
            f"你要删的是“{anchor}”，但挑的这条原文里没有它，挑错了，"
            f"没删：{text}。库里提到“{anchor}”的是：{shown}{more}"
            "请先列出全部记忆，照原文再删。",
            None,
        )
    picked_text = next(
        (text for _id, text in matches if _id == guard["target_id"]), ""
    )
    if not _same_text(picked_text, text):
        return (
            "原文对不上库里存的那条（可能列表旧了、或抄错了），没删。"
            "请先列出全部记忆，照原文再删。",
            None,
        )
    note = f"程序已核对：anchor“{anchor}”对得上“{picked_text}”"
    if len(matches) > 1:
        note += f"（库里还有 {len(matches) - 1} 条提到“{anchor}”的记忆，确认删的是这一条。）"
    return None, {"_anchor_match": note}


def _check_update_anchor(args):
    """改记忆的确认前校验（判断 c 号 precheck）：程序核对模型挑的这条
    是不是用户要改的那条。anchor 就是 old_text（要改的那条旧原文）。
    返回 (error, augment)，和 _check_delete_anchor 一样的形状。
    统一守卫 guard_memory_target 验：挑的这条原文和 old_text 完全
    一致才放行；多条候选、挑的不是唯一精确的那条时，拦下让 BOOS 挑。
    只读、不改库，确认前和执行前各跑一次。
    """
    store, error = _store_or_error()
    if error is not None:
        return error["error"], None
    memory_id = (args or {}).get("memory_id") or ""
    old_text = (args or {}).get("old_text") or ""
    if not _norm_anchor_text(old_text):
        return (
            "改记忆要写 old_text：列出来的旧原文，没写没法核对。",
            None,
        )
    matches = _find_anchor_matches(store, old_text)
    guard = guard_memory_target("update", memory_id, old_text, matches)
    if not guard["ok"]:
        if not matches:
            return (
                f"库里没找到含“{old_text}”的记忆，没改。"
                "请先列出全部记忆，确认要改的是哪条。",
                None,
            )
        shown = "；".join(t for _i, t in matches[:3])
        more = f"（等共 {len(matches)} 条）" if len(matches) > 3 else ""
        return (
            f"你要改的是“{old_text}”，但挑的这条对不上，挑错了，"
            f"没改。库里提到“{old_text}”的是：{shown}{more}"
            "请先列出全部记忆，照原文再改。",
            None,
        )
    picked_text = next(
        (text for _id, text in matches if _id == guard["target_id"]), ""
    )
    note = f"程序已核对：旧原文对得上“{picked_text}”"
    if len(matches) > 1:
        note += f"（库里还有 {len(matches) - 1} 条提到它的记忆，确认改的是这一条。）"
    return None, {"_anchor_match": note}


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
    # 57 号截断悬崖：工具结果一律截到前 8000 字（2026-11-13 从 4000 放宽），列表按存入顺序排时，
    # 尾巴上最新、最可能要删的那几条会被静默截掉，模型连挑的机会都没有。
    # 新在前：最新的条目排在前面，截断也先截掉最老的（和面板 36 号一致）。
    entries.reverse()
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
    # 判断 c 号：执行前再验一次统一守卫（确认卡建好后库可能变了，
    # 防卡上的条目在确认前被改掉/删掉）。precheck 已经验过一次，
    # 这里是第二道；只有守卫放行、且 target_id 就是要改的这条才动手。
    anchor_error, _augment = _check_update_anchor(args)
    if anchor_error:
        return {"ok": False, "data": None, "error": anchor_error}
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
    # 57 号：执行前再验一次 anchor（确认卡建好后库可能变了，防卡上的
    # 条目在确认前被改掉/删掉）。precheck 已经验过一次，这里是第二道。
    anchor_error, _augment = _check_delete_anchor(args)
    if anchor_error:
        return {"ok": False, "data": None, "error": anchor_error}
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
                  needs_confirmation, func, replay_safe, side_effect_free,
                  precheck=None):
    """登记一个记忆工具（已登记过时跳过，重复调不报错）。

    replay_safe / side_effect_free 每项显式写清楚，不推导：
    只读的列（list_memories）才能重跑、才能走线程超时；会改东西的
    记/改/删一律不可重跑、同步跑（见 tool_gateway.py 的执行规则）。
    precheck（57 号）：确认前校验，见 tools/tool_registry.py。
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
        precheck=precheck,
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
            "往长期记忆里加一条 BOOS 的事实/偏好/决定/习惯。BOOS 明确表达"
            "出来的就调——不只认“记住”两个字：他说“我想以后每天7点起床”"
            "“我爱吃拉面”“以后每天定投5块”这种明确的事实/偏好/决定/习惯"
            "都算，直接存；只有一次性的请求（“帮我查一下”“发个消息”）"
            "和闲聊不存。text 写 BOOS 原话里的那条事实。"
            "返回 {memory_id, text, duplicate}（duplicate 表示库里已经有"
            "同一条，没重复存）。成功返回 ok=True 才算存进去了，才能跟"
            "BOOS 说“记下了”；没调这个工具时别说“记下了”。"
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
            "程序会先核对：挑的这条原文和 old_text 完全一致才建确认卡，"
            "对不上、多条候选挑的不是唯一精确的那条时直接拒绝、不建卡；"
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
                "_anchor_match": {
                    "type": "string",
                    "description": "程序算好的匹配结果，确认卡上显示用；模型不用填",
                    "maxLength": 500,
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
        precheck=_check_update_anchor,
    )
    _register_one(
        name="delete_memory",
        description=(
            "删一条长期记忆。先调 list_memories 列出全部，找到要删的那条，"
            "把 memory_id、text（列出来的原文，一字不差地抄）和 anchor_text"
            "（用户原话里的删除目标，比如 BOOS 说“删掉鸡杂面”时 anchor_text"
            "就是“鸡杂面”）一起传来；原文对不上库里的就不删。"
            "程序会先核对 anchor：挑的这条原文里含 anchor 才建确认卡，"
            "对不上、没找到时直接拒绝、不建卡；库里还有别的提到 anchor 的，"
            "卡上会注明，确认删的是这一条；"
            "先问清楚再删，别自己挑。删之前会先弹确认卡，卡上显示 anchor"
            "和程序算好的匹配结果，BOOS 点了确认才执行，取消了就不删；"
            "删了就彻底没了（向量和审计记录都清掉），别删错。"
            "返回 {memory_id, text, history_cleared}。"
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
                "anchor_text": {
                    "type": "string",
                    "description": "用户原话里的删除目标（比如“鸡杂面”）",
                    "minLength": 1,
                    "maxLength": 200,
                },
                "_anchor_match": {
                    "type": "string",
                    "description": "程序算好的匹配结果，确认卡上显示用；模型不用填",
                    "maxLength": 500,
                },
            },
            "required": ["memory_id", "text", "anchor_text"],
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
        precheck=_check_delete_anchor,
    )
