# tools/tool_gateway.py（工具网关）
# 作用：模型调工具的唯一入口。所有工具调用都走这里，不走直通车。
# 原理：按顺序过六道关——登记检查 → 参数校验 → 权限检查 →
#       确认检查 → 执行 → 结果格式检查；每一步的结果都记审计日志。
#       执行那关：无副作用的工具按登记的 timeout 掐表（线程跑、
#       超时只算“等不及了”，结果未知——见 _run_with_timeout）；
#       会改东西的工具同步执行，不走线程超时（线程杀不掉，超时后
#       后台还会接着跑，等于一次调用可能改两次）；重复执行按登记的
#       replay_safe 拦
#       （见 can_replay：只读的能重跑，会改东西的执行过一次就绝不
#       再执行，恢复时一律喂存下来的结果）。
#       预算不由网关拦：工具执行本身不直接花钱，花的钱是模型每一轮
#       的调用费，已经在 providers 里按账本记了（工具循环传
#       ledger="tool"）；网关只负责“该不该跑、跑没跑成”，绝不因为
#       预算拦住一次已经确认过的执行（BOOS 定的规矩：到 $5 提醒，
#       永不自动停——网关这里连提醒都不发，提醒只在记账时发一次）。
# 关系：tools/agent_loop.py 调 execute() 跑工具；
#       web/server.py 的确认接口调 resolve_approval() 完成确认后执行；
#       工具定义和风险等级来自 tools/tool_registry.py；
#       审计写 tools/audit.py 的 AuditLog；确认状态存 tasks/task_store.py。
# 确认流程（两步走）：模型想调一个要确认的工具时，execute() 不执行，
#       只返回 needs_approval + 一条确认记录（approval_id、工具名、
#       完整参数 + 参数 sha256）；循环把任务存盘、把确认卡推给 BOOS；
#       BOOS 点了确认后，resolve_approval() 用 TaskStore.cas 原子抢占，
#       只执行存着的那份参数——页面新传来的参数一律不认，
#       这样中途改参数、重复点确认都执行不了第二次。
#       确认执行完马上把结果存盘：对应的 tool_calls 记录改成最终状态
#       （ok/failed），工具结果也存进任务（pending_tool_result），
#       恢复时优先读存下来的结果；执行完了、还没存盘时进程崩了，
#       结果才可能丢——保存发生在返回之前，窗口只剩执行完到存盘
#       这一小段（中间没有模型调用和网络），不是执行完还隔着一整轮。

import contextvars
import json
import threading

from tasks.task_store import approval_expired, approval_sha256, make_approval
from tools.audit import AuditLog
from tools.tool_registry import get_tool

# 参数校验支持的 JSON Schema 类型（第一版只用得到这些，够搜索工具用了；
# 第二批工具需要更多类型时再加，不要提前造一堆用不上的）。
_SCHEMA_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _check_value(value, schema, where):
    """按一份 JSON Schema 校验一个值。返回错误原因字符串，通过返回 ""。

    只校验第一版用得到的约束：type、enum、minimum / maximum、
    minLength / maxLength、minItems / maxItems、required（对象内）、
    properties（对象内，未知字段直接报错，模型不能自造参数）。
    注意：JSON 里的 true/false 在 Python 里是 bool 的实例，
    而 bool 又是 int 的子类——校验 integer 和 number 时都要先把
    bool 排除掉，不然 True 会被当成整数 1（或数字 1.0）蒙混过去。
    """
    expected = schema.get("type")
    if expected:
        python_type = _SCHEMA_TYPES.get(expected)
        if python_type is None:
            return f"{where} 的类型 {expected!r} 不支持校验"
        if expected == "integer" and isinstance(value, bool):
            return f"{where} 应为整数，不能是 true/false"
        if expected == "number" and isinstance(value, bool):
            return f"{where} 应为数字，不能是 true/false"
        if not isinstance(value, python_type):
            return f"{where} 应为 {expected}，实际是 {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} 只认 {schema['enum']} 里的值，不是 {value!r}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{where} 不能小于 {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{where} 不能大于 {schema['maximum']}"
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{where} 太短，至少 {schema['minLength']} 个字符"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{where} 太长，最多 {schema['maxLength']} 个字符"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"{where} 至少要有 {schema['minItems']} 项"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{where} 最多只能有 {schema['maxItems']} 项"
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in value:
            if key not in properties:
                return f"{where} 里有多余参数 {key!r}，工具定义里没有它"
        for key in schema.get("required") or []:
            if key not in value:
                return f"{where} 缺少必填参数 {key!r}"
        for key, sub_schema in properties.items():
            if key in value:
                error = _check_value(value[key], sub_schema, f"{where}.{key}")
                if error:
                    return error
    return ""


def validate_args(args, parameters_schema):
    """按工具登记的 parameters（JSON Schema）校验模型传来的参数。
    返回错误原因字符串，通过返回 ""。args 不是字典时直接报错。"""
    if not isinstance(args, dict):
        return "工具参数应为对象（字典），不是别的类型"
    schema = parameters_schema or {"type": "object", "properties": {}}
    return _check_value(args, schema, "参数")


def _validate_result_envelope(result, tool_name):
    """校验工具返回的包络：必须是字典，带 "ok"（布尔值）。
    ok 为 True 时要有 "data"；ok 为 False 时要有 "error"（字符串）。
    形状不对的返回直接按执行失败处理，不把脏结果喂给模型。"""
    if not isinstance(result, dict):
        return f"工具 {tool_name} 返回的不是字典，无法使用"
    if "ok" not in result or not isinstance(result["ok"], bool):
        return f"工具 {tool_name} 返回里没有 ok（应为 true/false）"
    if result["ok"] and "data" not in result:
        return f"工具 {tool_name} 成功了却没给 data"
    if not result["ok"] and not isinstance(result.get("error"), str):
        return f"工具 {tool_name} 失败了却没给 error（字符串）"
    return ""


def _run_with_timeout(func, args, timeout):
    """按登记的超时秒数跑一次工具函数：线程跑、主线程等 timeout 秒。

    注意：这只是“等待超时”，不是“硬超时”——超时的线程杀不掉，
    会在后台跑完自己结束，主线程只是不等它了。所以超时返回不等
    于工具没执行：工具可能晚几秒就成功了，也可能正在改东西。
    超时时返回 ok=False + timeout=True（结果未知），调用方不许把它
    当成普通失败、不许自动重试（见 tools/agent_loop.py）。
    工作线程里跑 func 之前，先把调用方的上下文拷过去
    （contextvars.copy_context），这样线程里也能读到循环绑的
    搜索服务等上下文绑定——新线程默认是空上下文，不拷就读不到。
    Windows 上 signal 那套行不通，所以用线程，笔记本和 Linux
    都能跑。timeout 不是正数时按 60 秒算。
    工具函数自己抛的错原样抛出去，由调用方按执行失败记。"""
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = 60.0
    if timeout <= 0:
        timeout = 60.0
    box = {}

    def _target():
        try:
            box["result"] = func(args)
        except Exception as exc:  # noqa: BLE001 - 记下来由主线程处理
            box["error"] = exc

    # 把调用方的上下文拷进工作线程：func 里读到的 ContextVar
    # （比如搜索服务）才和主线程是同一份绑定。
    ctx = contextvars.copy_context()
    worker = threading.Thread(target=lambda: ctx.run(_target), daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return {
            "ok": False,
            "data": None,
            "error": (
                f"工具执行超时（超过 {timeout:g} 秒还没返回）：这只是等不及了，"
                "结果未知，执行可能仍在后台继续——不要当成没执行，"
                "也不要重复调用。"
            ),
            "timeout": True,
        }
    if "error" in box:
        raise box["error"]
    return box.get("result")


class ToolGateway:
    """工具网关：模型调工具的唯一入口。

    参数 audit_log：AuditLog 实例，不传时用默认路径
    （工作区下的 .audit/audit.log，和 tools/cli.py 同一个位置）。
    """

    def __init__(self, audit_log=None):
        self.audit_log = audit_log or AuditLog(_default_audit_path())

    def can_replay(self, tool_name):
        """这工具能不能重复执行同一份参数（登记的 replay_safe）。

        重复执行规则：只读操作（replay_safe=True，如搜索）可以再执行一次；
        会改东西的操作（replay_safe=False，如写文件）执行过一次就绝不
        再执行——恢复/续跑时模型又发同一条调用（同一个 tool_call_id），
        循环直接拒绝，不执行（见 tools/agent_loop.py 的 _execute_one）。
        恢复时一律喂存下来的结果，不重执行。没登记的工具一律按不能
        重跑处理（宁可不跑，不可错跑）。"""
        tool = get_tool(tool_name)
        if tool is None:
            return False
        return bool(tool.get("replay_safe", False))

    # ---------- 执行入口 ----------

    def execute(self, tool_name, args, task_id=None, trace_id=None,
                tool_call_id=None, confirmed=False, run_id=None):
        """执行一次工具调用，返回决定字典：
        {"decision": "ok" / "rejected" / "needs_approval",
         "result": 工具返回（ok 时才有）,
         "reason": 拒绝或失败的原因（ok 时为 ""）,
         "approval": 确认记录（needs_approval 时才有）}。

        参数 confirmed：True 表示这次是 BOOS 点了确认后执行的
        （resolve_approval 调进来），跳过“要确认”的检查，但参数校验、
        禁令、风险等级照样过一遍——确认不能绕过这些。
        参数 run_id：所属任务的本次运行编号，和 task_id 一起写进确认记录，
        执行校验时核对归属用；不传时确认记录里记 None。
        """
        extra = {"task_id": task_id or "", "trace_id": trace_id or ""}

        # 第 1 关：登记检查——没登记的工具一律不执行。
        tool = get_tool(tool_name)
        if tool is None:
            reason = f"未登记的工具：{tool_name}，不能执行。"
            self._audit(tool_name, "execute", False, reason, extra)
            return {"decision": "rejected", "result": None, "reason": reason, "approval": None}

        # 第 2 关：参数校验——按登记的 parameters（JSON Schema）逐项校验。
        error = validate_args(args, tool["parameters"])
        if error:
            reason = f"工具 {tool_name} 的参数不对：{error}。"
            self._audit(tool_name, "execute", False, reason, extra)
            return {"decision": "rejected", "result": None, "reason": reason, "approval": None}

        # 第 3 关：权限检查——第二级禁令（tier2）永远不执行，哪怕确认过；
        #   高风险（high）第一版也不执行（高风险的定义就是“危险操作，
        #   永远不跑”，见 tool_registry.py）。
        if tool["tier2"]:
            reason = f"工具 {tool_name} 在第二级禁令里，永远不执行。"
            self._audit(tool_name, "execute", False, reason, extra)
            return {"decision": "rejected", "result": None, "reason": reason, "approval": None}
        if tool["risk_level"] == "high":
            reason = f"工具 {tool_name} 是高风险操作，第一版不执行。"
            self._audit(tool_name, "execute", False, reason, extra)
            return {"decision": "rejected", "result": None, "reason": reason, "approval": None}

        # 第 4 关：确认检查——要确认的工具，没确认就不执行，只返回确认记录。
        if tool["needs_confirmation"] and not confirmed:
            approval = make_approval(
                tool_name, args, tool_call_id or "", tool["risk_level"],
                task_id=task_id, run_id=run_id,
            )
            self._audit(
                tool_name, "needs_approval", False,
                "等 BOOS 确认后才执行。", {**extra, "approval_id": approval["approval_id"]},
            )
            return {
                "decision": "needs_approval",
                "result": None,
                "reason": "",
                "approval": approval,
            }

        # 第 5 关：执行。无副作用的工具（replay_safe=True，如搜索）按登记的
        #   timeout 掐表跑（见 _run_with_timeout）：超时只算“等不及了”，
        #   结果未知，不按普通失败处理、不自动重试。
        #   会改东西的工具（replay_safe=False）不用线程超时，直接同步跑：
        #   线程杀不掉，超时后后台还会接着跑，等于一次调用可能改两次；
        #   同步跑就没有“超时了但还在改”这回事，跑完才有结果。
        #   工具函数自己抛错也一样算执行失败。
        try:
            if tool.get("replay_safe", True):
                result = _run_with_timeout(tool["func"], args, tool.get("timeout", 60))
            else:
                result = tool["func"](args)
        except Exception as exc:  # noqa: BLE001 - 工具内部错统一记成失败
            reason = f"工具 {tool_name} 执行时出错：{exc}"
            self._audit(tool_name, "execute", False, reason, extra)
            return {
                "decision": "ok",
                "result": {"ok": False, "data": None, "error": reason},
                "reason": "",
                "approval": None,
            }

        # 第 6 关：结果格式检查——包络不对的按执行失败处理。
        error = _validate_result_envelope(result, tool_name)
        if error:
            reason = f"工具 {tool_name} 的返回格式不对：{error}。"
            self._audit(tool_name, "execute", False, reason, extra)
            return {
                "decision": "ok",
                "result": {"ok": False, "data": None, "error": reason},
                "reason": "",
                "approval": None,
            }
        self._audit(
            tool_name, "execute", bool(result["ok"]),
            "" if result["ok"] else str(result.get("error") or ""),
            extra, confirmed=confirmed,
        )
        return {"decision": "ok", "result": result, "reason": "", "approval": None}

    # ---------- 确认后执行 ----------

    def _persist_confirmed_execution(self, task_store, task_id, tool_name,
                                     tool_call_id, result):
        """确认后执行完，马上把结果存盘（返回之前存好）：
        对应 tool_call_id 的那条 tool_calls 记录改成最终状态
        （ok/failed），工具结果也写进记录、并存到任务的
        pending_tool_result 里（恢复时优先读它）。
        找不到对应记录时（老任务）补一条，不崩。
        结果先转一道 JSON（default=str），保证存进去的东西一定能
        写进任务文件；转不过来时只存一句“结果存不下来”，不让存盘
        本身炸掉整个确认流程。
        返回存好后的任务记录；存不下来返回 None。"""
        try:
            stored_result = json.loads(
                json.dumps(result, ensure_ascii=False, default=str)
            )
        except (TypeError, ValueError):
            stored_result = {"ok": False, "data": None,
                             "error": "工具结果不是能存盘的形状。"}
        status = "ok" if (stored_result or {}).get("ok") else "failed"

        def mutate(current):
            records = list(current.get("tool_calls") or [])
            for rec in records:
                if rec.get("tool_call_id") == tool_call_id:
                    rec["status"] = status
                    rec["result"] = stored_result
                    break
            else:
                records.append({
                    "tool": tool_name,
                    "args": None,
                    "tool_call_id": tool_call_id,
                    "status": status,
                    "result": stored_result,
                })
            current["tool_calls"] = records
            current["pending_tool_result"] = stored_result
            current["pending_tool_call_id"] = tool_call_id
            return current

        ok, updated = task_store.cas(task_id, "RUNNING", mutate)
        if ok:
            return updated
        # 状态已经不是 RUNNING（极少见：有人同时动了任务）：读出来
        # 改了再存一次，别让执行完的结果丢了。
        current = task_store.load(task_id)
        if current is None:
            return None
        try:
            return task_store.save(mutate(dict(current)))
        except Exception:
            return None

    def resolve_approval(self, task_store, task_id, approval_id, decision):
        """处理 BOOS 对一张确认卡的决定。返回决定字典：
        {"decision": "ok" / "rejected" / "cancelled",
         "result": 工具返回（确认并执行成功时才有）,
         "reason": 拒绝或取消的原因,
         "task": 处理完的任务记录（有就带上）}。

        参数 decision："confirm"（确认执行）或 "cancel"（取消）。
        只执行任务里存着的那份参数：先按 approval_id 找到任务里的确认记录，
        再核对参数 sha256（对不上说明存着的参数被改过，直接拒绝），
        最后用 TaskStore.cas 原子抢占——只有状态还是 WAITING_APPROVAL
        的任务才能被处理，重复点确认、超时后才点、别人先处理了，
        都会在这里被拦住，不会执行第二次。
        确认执行完、返回之前，马上把结果存盘（见
        _persist_confirmed_execution）：对应记录改成 ok/failed，
        结果存进任务，恢复时优先读存下来的结果——防重跑检查认的是
        ok/failed 记录，不存盘它就看不见这次执行，模型再发同一条
        调用会被当成没执行过而重跑一次。
        """
        task = task_store.load(task_id)
        if task is None:
            return {
                "decision": "rejected",
                "result": None,
                "reason": f"找不到任务：{task_id}。",
                "task": None,
            }
        approval = task.get("approval") or {}
        if approval.get("approval_id") != approval_id:
            return {
                "decision": "rejected",
                "result": None,
                "reason": "确认编号对不上，这张卡可能已经被处理过了。",
                "task": task,
            }

        # 归属核对：确认记录里写着 task_id/run_id 时，必须和这条任务对上，
        # 不然就是拿错了卡，不能执行。旧记录没有这两个字段时跳过核对
        # （按没有处理，不崩、不拦）。
        if approval.get("task_id") and approval["task_id"] != task_id:
            return {
                "decision": "rejected",
                "result": None,
                "reason": "这张确认卡不属于这条任务，不能执行。",
                "task": task,
            }
        if approval.get("run_id") and approval["run_id"] != task.get("run_id"):
            return {
                "decision": "rejected",
                "result": None,
                "reason": "这张确认卡不是本次运行的，不能执行。",
                "task": task,
            }

        # 超时：10 分钟没点就当取消，之后再点也不执行。
        # 确认记录留在任务上（恢复循环时还要靠它找 tool_call_id，
        # 告诉模型那次调用超时取消了；列表只看状态，不会再显示这张卡）。
        if approval_expired(approval):
            ok, updated = task_store.cas(
                task_id, "WAITING_APPROVAL",
                lambda current: {**current, "status": "EXPIRED"},
            )
            return {
                "decision": "rejected",
                "result": None,
                "reason": "这张确认卡已经超时（10 分钟没点），按取消处理了。",
                "task": updated if ok else task,
            }

        # 取消：状态改成 CANCELLED，工具不执行。确认记录留在任务上
        # （恢复循环时还要靠它找 tool_call_id，告诉模型用户没同意）。
        if decision == "cancel":
            ok, updated = task_store.cas(
                task_id, "WAITING_APPROVAL",
                lambda current: {**current, "status": "CANCELLED"},
            )
            if not ok:
                return {
                    "decision": "rejected",
                    "result": None,
                    "reason": "这张卡已经被处理过了，不用再点。",
                    "task": updated,
                }
            self._audit(
                approval.get("tool", ""), "cancelled", False, "BOOS 取消了这次工具调用。",
                {"task_id": task_id, "trace_id": task.get("trace_id", ""),
                 "approval_id": approval_id},
            )
            return {"decision": "cancelled", "result": None, "reason": "", "task": updated}

        if decision != "confirm":
            return {
                "decision": "rejected",
                "result": None,
                "reason": f"确认决定只认 'confirm' 或 'cancel'，不是 {decision!r}。",
                "task": task,
            }

        # 确认：先核对存着的参数有没有被改过，再原子抢占。
        stored_args = approval.get("args") or {}
        if approval_sha256(stored_args) != approval.get("args_sha256"):
            return {
                "decision": "rejected",
                "result": None,
                "reason": "存着的工具参数被改过（和确认时的哈希对不上），不能执行。",
                "task": task,
            }
        ok, updated = task_store.cas(
            task_id, "WAITING_APPROVAL",
            lambda current: {**current, "status": "RUNNING"},
        )
        if not ok:
            return {
                "decision": "rejected",
                "result": None,
                "reason": "这张卡已经被处理过了，不会执行第二次。",
                "task": updated,
            }
        # 抢到了：只执行存着的那份参数，页面传来的参数一律不认。
        outcome = self.execute(
            approval.get("tool", ""),
            stored_args,
            task_id=task_id,
            trace_id=task.get("trace_id"),
            tool_call_id=approval.get("tool_call_id", ""),
            confirmed=True,
            run_id=task.get("run_id"),
        )
        if outcome["decision"] == "needs_approval":
            # 不应该发生：confirmed=True 时不会再要确认。万一发生了，
            # 把任务放回等确认，不让它卡在 RUNNING。
            task_store.cas(
                task_id, "RUNNING",
                lambda current: {**current, "status": "WAITING_APPROVAL"},
            )
            return {
                "decision": "rejected",
                "result": None,
                "reason": "确认后执行时又要求确认，任务已放回等确认状态。",
                "task": task_store.load(task_id),
            }
        if outcome["decision"] == "rejected":
            # 确认后执行被拦（参数、禁令等）：工具没执行，任务不能卡在
            # RUNNING——放回 WAITING_APPROVAL，卡还在，BOOS 可以再点
            # 确认或取消。
            task_store.cas(
                task_id, "RUNNING",
                lambda current: {**current, "status": "WAITING_APPROVAL"},
            )
            return {
                "decision": "rejected",
                "result": None,
                "reason": outcome["reason"],
                "task": task_store.load(task_id),
            }
        # 执行完了：先把记录和结果存盘，再返回——恢复时读盘上的，
        # 不只靠内存里传回去的那份。
        saved = self._persist_confirmed_execution(
            task_store, task_id, approval.get("tool", ""),
            approval.get("tool_call_id", ""), outcome["result"],
        )
        return {
            "decision": "ok",
            "result": outcome["result"],
            "reason": "",
            "task": saved if saved is not None else task_store.load(task_id),
        }

    # ---------- 内部 ----------

    def _audit(self, tool_name, action, success, reason, extra, confirmed=False):
        """记一条审计事件。审计写失败不影响主流程（记不下来就跳过，
        不能因为审计坏了就把一次工具调用也弄丢）。"""
        try:
            self.audit_log.record(
                tool=tool_name,
                action=action,
                actor="agent",
                confirmed=confirmed,
                success=success,
                reason=reason,
                extra=extra,
            )
        except Exception:  # noqa: BLE001 - 审计失败不挡主流程
            pass


def _default_audit_path():
    """默认审计日志路径：工作区下的 .audit/audit.log，
    和 tools/cli.py 用的同一个位置。"""
    import os

    workspace = os.environ.get("TWELVE_WORKSPACE") or os.getcwd()
    return os.path.join(workspace, ".audit", "audit.log")
