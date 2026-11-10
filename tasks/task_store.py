# tasks/task_store.py（任务存储）
# 作用：工具循环的任务状态存在磁盘上，不是只放内存里。
# 原理：一个 JSON 文件（tasks/tasks.json）+ 一把线程锁 + 原子写
#       （先写临时文件再 os.replace，保证写一半断电也不会留半个文件）。
#       只暴露 save / load / cas / list_waiting 四个方法，
#       以后想换成 SQLite 时只改这个文件里面，不用改调用方。
# 关系：tools/agent_loop.py 每次循环前后调 save 存状态；
#       web/server.py 的确认接口用 cas 原子抢占，防止重复执行。
# 任务状态（7 种，第一版没有 PAUSED）：
#   CREATED：刚建好，还没跑；RUNNING：正在循环里跑；
#   WAITING_APPROVAL：停在确认卡上，等 BOOS 点确认/取消；
#   COMPLETED：跑完了；FAILED：出错了；CANCELLED：BOOS 取消了
#   或超时没确认；EXPIRED：确认超时（10 分钟没点）转成的状态，
#   和 CANCELLED 分开记，方便以后查“超时了几次”。
# 任务记录字段：task_id（任务编号）、run_id（本次运行编号）、
#   trace_id（串联记账和审计的编号）、model（本任务锁定的模型名，
#   中途换模型就另开新任务）、goal（BOOS 的原话）、status（上面 7 种）、
#   messages（循环消息历史，assistant 的完整 dict，含 reasoning_content
#   和 tool_calls，JSON 能直接存）、tool_calls（调过的工具记录）、
#   failures（同一工具同一参数失败次数）、approval（等确认的信息：
#   approval_id、工具名、完整参数 + 参数 sha256、tool_call_id、
#   风险等级、创建时间、过期时间 10 分钟）、result（最终回答）。
# 安全：确认时只执行这里存着的参数，不执行页面新传来的参数
#       （见 web/server.py）；过期、重复确认都由 cas 拦住。

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

# 确认卡有效期：10 分钟。超时没点就当取消，之后再点也不执行。
APPROVAL_TIMEOUT = timedelta(minutes=10)

# 任务状态（7 种）。写错状态名直接报错，不让脏状态混进文件。
TASK_STATUSES = (
    "CREATED",
    "RUNNING",
    "WAITING_APPROVAL",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "EXPIRED",
)


def new_task_id():
    """生成一个任务编号（task_ + 随机 16 位十六进制）。"""
    return "task_" + uuid.uuid4().hex[:16]


def new_approval_id():
    """生成一个确认编号（appr_ + 随机 16 位十六进制）。"""
    return "appr_" + uuid.uuid4().hex[:16]


def new_trace_id():
    """生成一个追踪编号（trace_ + 随机 16 位十六进制），用来把
    模型请求、工具请求、授权、执行结果和费用串起来查。"""
    return "trace_" + uuid.uuid4().hex[:16]


def make_task(goal, model, task_id=None, run_id=None, trace_id=None):
    """建一条新任务记录（纯拼字典，不写文件）。model 是本任务锁定的
    模型名，恢复时换模型就另开新任务，不接着跑旧的。"""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "task_id": task_id or new_task_id(),
        "run_id": run_id or uuid.uuid4().hex[:8],
        "trace_id": trace_id or new_trace_id(),
        "model": model,
        "goal": goal,
        "status": "CREATED",
        "messages": [],
        "tool_calls": [],
        "failures": {},
        "approval": None,
        "result": None,
        "created_at": now,
        "updated_at": now,
    }


def approval_sha256(args):
    """算一份工具参数的 sha256（确认时比对用，防止参数被中途改过）。"""
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_approval(tool, args, tool_call_id, risk_level):
    """拼一条等确认的信息：确认编号、工具名、完整参数 + 参数 sha256、
    tool_call_id、风险等级、创建时间和过期时间（10 分钟后）。"""
    now = datetime.now(timezone.utc)
    return {
        "approval_id": new_approval_id(),
        "tool": tool,
        "args": args,
        "args_sha256": approval_sha256(args),
        "tool_call_id": tool_call_id,
        "risk_level": risk_level,
        "created_at": now.isoformat(timespec="seconds"),
        "expires_at": (now + APPROVAL_TIMEOUT).isoformat(timespec="seconds"),
    }


def approval_expired(approval, now=None):
    """这条确认是否已过期（超过 10 分钟没点）。"""
    if not approval:
        return True
    now = now or datetime.now(timezone.utc)
    expires = datetime.fromisoformat(approval["expires_at"])
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return now >= expires


class TaskStore:
    """任务存储：JSON 文件 + 线程锁 + 原子写。

    参数 file_path：任务文件路径，不传时用项目根目录下的
    tasks/tasks.json（这个文件会被 .gitignore 忽略，不上传到 GitHub，
    因为里面是进行中的任务状态，不是代码）。
    """

    def __init__(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tasks",
                "tasks.json",
            )
        self.file_path = file_path
        self._lock = threading.RLock()

    def _read_all(self):
        """读出全部任务（字典：task_id -> 任务记录）。文件不存在时返回空字典。
        文件损坏（JSON 解析失败）时直接报错，不猜、不清空——任务状态丢了
        就没法恢复确认，宁可让调用方知道出错了。"""
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"任务文件损坏，读不出来：{self.file_path}（{exc}）") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"任务文件损坏，顶层不是对象：{self.file_path}")
        return data

    def _write_all(self, data):
        """原子写回全部任务：先写同目录的临时文件，再 os.replace 换过去，
        写一半断电也不会留半个文件。"""
        directory = os.path.dirname(self.file_path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp_path = self.file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.file_path)

    def save(self, task):
        """保存一条任务记录（新建或更新）。task_id 缺了、状态名不在 7 种里，
        直接报错。返回存进去的那条记录。"""
        if not isinstance(task, dict) or not task.get("task_id"):
            raise ValueError("任务记录必须带 task_id。")
        if task.get("status") not in TASK_STATUSES:
            raise ValueError(f"任务状态只认 {TASK_STATUSES}，不是 {task.get('status')!r}。")
        with self._lock:
            data = self._read_all()
            task = dict(task)
            task["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            data[task["task_id"]] = task
            self._write_all(data)
            return dict(task)

    def load(self, task_id):
        """按 task_id 取一条任务记录，没有返回 None。返回的是副本，
        改它不影响文件里存着的那份。"""
        with self._lock:
            task = self._read_all().get(task_id)
            return dict(task) if task is not None else None

    def cas(self, task_id, expected_status, mutate):
        """原子比较并交换（Compare-And-Swap）：只有任务当前状态等于
        expected_status 时才执行 mutate 并保存，否则什么都不做。

        参数 mutate：一个函数，接收当前任务记录（副本），返回改好的
        任务记录；返回 None 表示“不用改了”，也不保存。
        返回 (True, 改好的任务记录) 表示抢到了；返回 (False, 当前任务
        记录) 表示状态已经变了（别人先抢到了、超时了、取消了），
        调用方不要再执行。任务不存在时返回 (False, None)。
        整个“读-比-改-写”都在同一把锁里做，不会两个人同时抢到。
        """
        with self._lock:
            data = self._read_all()
            task = data.get(task_id)
            if task is None:
                return False, None
            if task.get("status") != expected_status:
                return False, dict(task)
            updated = mutate(dict(task))
            if updated is None:
                return False, dict(task)
            if updated.get("status") not in TASK_STATUSES:
                raise ValueError(f"任务状态只认 {TASK_STATUSES}，不是 {updated.get('status')!r}。")
            updated["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            data[task_id] = updated
            self._write_all(data)
            return True, dict(updated)

    def find_by_approval(self, approval_id):
        """按确认编号找任务（确认卡只带 approval_id 时用它定位任务）。
        找不到返回 None。"""
        with self._lock:
            for task in self._read_all().values():
                approval = task.get("approval") or {}
                if approval.get("approval_id") == approval_id:
                    return dict(task)
        return None

    def list_waiting(self):
        """所有等确认的任务（状态 WAITING_APPROVAL），按创建时间排序。
        面板刷新时用它列出“还有哪些确认卡没点”。"""
        with self._lock:
            tasks = [
                dict(task)
                for task in self._read_all().values()
                if task.get("status") == "WAITING_APPROVAL"
            ]
        tasks.sort(key=lambda task: task.get("created_at", ""))
        return tasks
