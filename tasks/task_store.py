# tasks/task_store.py（任务存储）
# 作用：工具循环的任务状态存在磁盘上，不是只放内存里。
# 原理：一个 JSON 文件（tasks/tasks.json）+ 进程内共享锁 + 跨进程锁
#       + 原子写（每次写用唯一临时文件再 os.replace，写一半断电也
#       不会留半个文件；两个进程同时写也不会互相踩坏临时文件）。
#       锁是模块级的：同一个文件路径的所有 TaskStore 实例（不管谁
#       建的）共用一把锁，所以实例级竞争在进程内不存在。
#       跨进程写（2026-11-13 修，Windows 并发测试失败的根因）：
#       进程内 threading.RLock 管不到别的进程；Windows 上 os.replace
#       还会被“目标文件正被别的进程打开读”挡住——Python 的 open()
#       不带 FILE_SHARE_DELETE，读着的进程会让替换报 WinError 5。
#       所以写操作再加两层：① 跨进程锁文件（tasks.json.lock），
#       POSIX 用 fcntl.flock、Windows 用 msvcrt.locking，save/cas/
#       reconcile/pause 的整个“读-改-写”都在锁里串行化，两个进程
#       同时写也不会丢更新、文件始终是完整 JSON；② os.replace 本身
#       带重试（退避、有限次数），防读进程正好开着文件的瞬间。
#       对外提供 save / load / cas / find_by_approval /
#       list_waiting / list_by_status / list_unresolved /
#       reconcile_on_startup / pause_running_on_shutdown，
#       以后想换成 SQLite 时只改这个文件里面，不用改调用方。
# 关系：tools/agent_loop.py 每次循环前后调 save 存状态；
#       web/server.py 的确认接口用 cas 原子抢占，防止重复执行。
# 任务状态（9 种）。写错状态名直接报错，不让脏状态混进文件。
# 12 号状态机（GPT 的七态 + 取消/过期两个终态）：
#   CREATED：刚建好，还没跑（等）；
#   RUNNING：正在循环里跑（正在）；
#   PAUSED：干净关机时停下的（暂停）——知道它停在哪，续跑是安全的，
#     和崩溃的 UNKNOWN 区分开；
#   WAITING_APPROVAL：停在确认卡上，等 BOOS 点确认/取消（等确认）；
#   COMPLETED：跑完了；FAILED：出错了；
#   UNKNOWN：启动时还在 RUNNING，结果未知（结果未知）——不知道它
#     执行到哪了，绝不悄悄重执行，等 BOOS 决定续跑还是作废；
#   CANCELLED：BOOS 取消了或作废了；EXPIRED：确认超时（10 分钟没点）
#     转成的状态，和 CANCELLED 分开记，方便以后查“超时了几次”。
# 任务记录字段：task_id（任务编号）、run_id（本次运行编号）、
#   trace_id（串联记账和审计的编号）、model（本任务锁定的模型名，
#   中途换模型就另开新任务）、goal（BOOS 的原话）、status（上面 9 种）、
#   messages（循环消息历史，assistant 的完整 dict，含 reasoning_content
#   和 tool_calls，JSON 能直接存）、tool_calls（调过的工具记录）、
#   failures（同一工具同一参数失败次数）、approval（等确认的信息：
#   approval_id、工具名、完整参数 + 参数 sha256、tool_call_id、
#   风险等级、创建时间、过期时间 10 分钟，所属任务的 task_id 和 run_id）、
#   result（最终回答）。
# 安全：确认时只执行这里存着的参数，不执行页面新传来的参数
#       （见 web/server.py）；过期、重复确认都由 cas 拦住。

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

try:
    import fcntl
except ImportError:  # Windows 上没有 fcntl，走 msvcrt。
    fcntl = None

try:
    import msvcrt
except ImportError:  # POSIX 上没有 msvcrt，走 fcntl。
    msvcrt = None

# 确认卡有效期：10 分钟。超时没点就当取消，之后再点也不执行。
APPROVAL_TIMEOUT = timedelta(minutes=10)

# 任务状态（9 种）。写错状态名直接报错，不让脏状态混进文件。
TASK_STATUSES = (
    "CREATED",
    "RUNNING",
    "PAUSED",
    "WAITING_APPROVAL",
    "COMPLETED",
    "FAILED",
    "UNKNOWN",
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


def make_approval(tool, args, tool_call_id, risk_level, task_id=None, run_id=None):
    """拼一条等确认的信息：确认编号、工具名、完整参数 + 参数 sha256、
    tool_call_id、风险等级、创建时间和过期时间（10 分钟后），以及
    它所属任务的 task_id 和 run_id（执行校验时核对归属用）。
    task_id/run_id 不传时记 None——老代码和旧记录没有这两个字段，
    读的时候按没有处理，不崩。"""
    now = datetime.now(timezone.utc)
    return {
        "approval_id": new_approval_id(),
        "tool": tool,
        "args": args,
        "args_sha256": approval_sha256(args),
        "tool_call_id": tool_call_id,
        "risk_level": risk_level,
        "task_id": task_id,
        "run_id": run_id,
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


# 模块级锁登记：同一个绝对路径的所有 TaskStore 实例共用一把 RLock。
# 实例各建各的锁等于没锁——服务器传自己的实例给循环、命令行又建一个，
# 两个实例同时写同一文件会互相盖掉。锁本身只在进程内有效，跨进程
# 不起作用（见 TaskStore 的文档）。
_PATH_LOCKS = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(file_path):
    """取这个文件路径共用的那把锁（没有就建一把）。"""
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(file_path)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[file_path] = lock
        return lock


# ---------- 跨进程锁 + 替换重试（2026-11-13，Windows 并发失败的根因） ----------

# 跨进程锁最多等 30 秒：写操作本身是毫秒级，等这么久还没拿到锁，
# 说明有进程卡住了（或锁文件被异常状态占住），直接报错、不悄悄写。
_XPROC_LOCK_TIMEOUT = 30.0
# os.replace 重试：最多 10 次，退避 10ms → 500ms（上限），总共最多等约 2 秒。
# 只重试 PermissionError（Windows 上目标文件正被读着就是这个错）；
# 别的 OSError（比如磁盘满了、路径没了）立刻抛，不重试。
_REPLACE_MAX_ATTEMPTS = 10
_REPLACE_BACKOFF = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


@contextmanager
def _cross_process_lock(lock_path):
    """跨进程互斥锁（上下文管理器）：锁文件 + 平台文件锁。

    锁文件（<任务文件>.lock）一直留在盘上，不删——删锁文件本身就会
    制造竞态（A 删了、B 正要锁）。锁的是文件里的第 0 个字节：
    POSIX 用 fcntl.flock（LOCK_EX | LOCK_NB，轮询等锁），Windows 用
    msvcrt.locking（LK_NBLCK 轮询）。拿不到锁时不死等，超时报
    RuntimeError，让调用方知道写没写进去，别假装写成功了。
    只在写操作的“读-改-写”期间持有（save/cas/reconcile/pause），
    读操作不加锁——读到旧版或新版都是完整 JSON。
    拿锁顺序永远是“先进程内锁、再跨进程锁”，别反过来，不死锁。
    """
    directory = os.path.dirname(lock_path) or "."
    os.makedirs(directory, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + _XPROC_LOCK_TIMEOUT
    acquired = False
    try:
        attempt = 0
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    # 两个平台锁都没有：没法跨进程互斥，别假装锁上了——
                    # 直接报错，让调用方知道这环境保不住跨进程写。
                    raise RuntimeError("跨进程锁不可用：既没有 fcntl 也没有 msvcrt。")
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"等跨进程锁超时（{_XPROC_LOCK_TIMEOUT:.0f} 秒）：{lock_path}"
                    ) from None
                time.sleep(min(0.01 * (2 ** attempt), 0.1))
                attempt += 1
        yield
    finally:
        if acquired:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif msvcrt is not None:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        os.close(fd)


def _replace_with_retry(tmp_path, file_path):
    """os.replace 带重试：Windows 上目标文件正被别的进程打开读时，
    替换会报 PermissionError（WinError 5）——读进程的 open() 不带
    FILE_SHARE_DELETE。读完就几毫秒的事，等一等再换，一般一次就成；
    10 次还换不过去就原样抛错，不吞。"""
    delay_index = 0
    for attempt in range(_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(tmp_path, file_path)
            return
        except PermissionError:
            if attempt == _REPLACE_MAX_ATTEMPTS - 1:
                raise
            delay = _REPLACE_BACKOFF[min(delay_index, len(_REPLACE_BACKOFF) - 1)]
            delay_index += 1
            time.sleep(delay)


class TaskStore:
    """任务存储：JSON 文件 + 进程内共享锁 + 原子写。

    参数 file_path：任务文件路径，不传时用项目根目录下的
    tasks/tasks.json（这个文件会被 .gitignore 忽略，不上传到 GitHub，
    因为里面是进行中的任务状态，不是代码）。

    并发范围（GPT 第一轮审查 P2，2026-11-13 修后）：
    - 同一进程里，不管建了几个 TaskStore 实例，只要指着同一个文件，
      就共用同一把锁（模块级按路径登记）——CAS 的“读-比-改-写”
      在进程内是原子的，两个实例抢同一条任务只有一个能抢到。
    - 跨进程也保证：写操作（save/cas/reconcile/pause）的整个
      “读-改-写”都在跨进程锁文件里串行化，两个进程同时写不会
      丢更新、文件始终是完整 JSON；CAS 在跨进程下也是原子的
      （走 TaskStore 的写操作都过这把锁）。锁文件是
      <任务文件>.lock，一直留在盘上。
    - 读操作（load/list_*）不加跨进程锁：读到旧版或新版都是完整
      JSON；替换那一瞬读进程正开着文件时，写的那边靠 os.replace
      重试兜底。
    """

    def __init__(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tasks",
                "tasks.json",
            )
        self.file_path = os.path.abspath(file_path)
        self._lock = _lock_for(self.file_path)
        # 跨进程锁文件（和任务文件放同一目录，一直留在盘上，不删）。
        self._lock_path = self.file_path + ".lock"

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
        """原子写回全部任务：每次用唯一临时文件（带进程号和随机数），
        写完再 os.replace 换过去。固定 .tmp 文件两个进程同时写会互相
        踩——第 3 批出过真事故，文件被写坏过。唯一临时文件保证写坏
        的只可能是单个临时文件，目标文件要么是旧的完整版、要么是
        新的完整版，绝不会是写了一半的。写失败时尽力删掉临时文件，
        不留垃圾。"""
        directory = os.path.dirname(self.file_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(self.file_path) + ".tmp.",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            # Windows 上目标文件正被别的进程打开读时，替换会报
            # PermissionError（WinError 5）——带重试兜底。
            _replace_with_retry(tmp_path, self.file_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save(self, task):
        """保存一条任务记录（新建或更新）。task_id 缺了、状态名不在 9 种里，
        直接报错。返回存进去的那条记录。"""
        if not isinstance(task, dict) or not task.get("task_id"):
            raise ValueError("任务记录必须带 task_id。")
        if task.get("status") not in TASK_STATUSES:
            raise ValueError(f"任务状态只认 {TASK_STATUSES}，不是 {task.get('status')!r}。")
        with self._lock, _cross_process_lock(self._lock_path):
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
        进程内是 threading.RLock，跨进程是锁文件（见 _cross_process_lock），
        两层都拿着才写。
        """
        with self._lock, _cross_process_lock(self._lock_path):
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

    def list_by_status(self, status):
        """按状态取任务（12 号：进度轮询读 RUNNING 的），按创建时间排序。"""
        with self._lock:
            tasks = [
                dict(task)
                for task in self._read_all().values()
                if task.get("status") == status
            ]
        tasks.sort(key=lambda task: task.get("created_at", ""))
        return tasks

    def list_unresolved(self):
        """没交代的任务（12 号：UNKNOWN + PAUSED），按创建时间排序。
        给页面上“恢复卡住的任务”入口用——这些是重启/关机后留下的
        悬案，等 BOOS 决定续跑还是作废；只读，不改状态。"""
        with self._lock:
            tasks = [
                dict(task)
                for task in self._read_all().values()
                if task.get("status") in ("UNKNOWN", "PAUSED")
            ]
        tasks.sort(key=lambda task: task.get("created_at", ""))
        return tasks

    def reconcile_on_startup(self):
        """服务启动时核对一遍盘上的任务（12 号）：

        - 还在 RUNNING 的：改成 UNKNOWN（结果未知），记下核对时间和原因，
          绝不悄悄重执行——不知道它执行到哪了，重跑可能重复改东西。
        - WAITING_APPROVAL 但确认已过期的：改成 EXPIRED（点了也没用，
          网关的 resolve_approval 本来就会拦）。
        - 其他状态（COMPLETED / FAILED / CANCELLED / EXPIRED / PAUSED /
          没过期的 WAITING_APPROVAL）不动。
        返回受影响的任务摘要 [{task_id, goal, old_status, new_status}]，
        没有时返回空列表。读文件失败时直接报错（和 _read_all 一致），
        调用方（服务启动）记日志、不拦启动。
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        affected = []
        with self._lock, _cross_process_lock(self._lock_path):
            data = self._read_all()
            changed = False
            for task in data.values():
                status = task.get("status")
                if status == "RUNNING":
                    task["status"] = "UNKNOWN"
                    task["reconciled_at"] = now
                    task["reconciled_reason"] = (
                        "服务启动时仍在 RUNNING，结果未知，未重新执行。"
                    )
                    task["updated_at"] = now
                    affected.append(
                        {
                            "task_id": task.get("task_id"),
                            "goal": task.get("goal") or "",
                            "old_status": "RUNNING",
                            "new_status": "UNKNOWN",
                        }
                    )
                    changed = True
                elif status == "WAITING_APPROVAL":
                    approval = task.get("approval") or {}
                    try:
                        expired = approval_expired(approval)
                    except (KeyError, TypeError, ValueError):
                        # 确认记录里没有能用的过期时间：宁可不动，
                        # 等 BOOS 点了再由网关判，也不在启动时悄悄改掉。
                        expired = False
                    if expired:
                        task["status"] = "EXPIRED"
                        task["reconciled_at"] = now
                        task["reconciled_reason"] = "确认已超时（10 分钟没点）。"
                        task["updated_at"] = now
                        affected.append(
                            {
                                "task_id": task.get("task_id"),
                                "goal": task.get("goal") or "",
                                "old_status": "WAITING_APPROVAL",
                                "new_status": "EXPIRED",
                            }
                        )
                        changed = True
            if changed:
                self._write_all(data)
        return affected

    def pause_running_on_shutdown(self):
        """干净关机时把还在跑的任务标成 PAUSED（12 号）：和崩溃的 UNKNOWN
        区分开——干净关机知道它停在哪（服务停了、循环没跑完），
        续跑是安全的。返回受影响的任务摘要，没有时返回空列表。"""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        affected = []
        with self._lock, _cross_process_lock(self._lock_path):
            data = self._read_all()
            changed = False
            for task in data.values():
                if task.get("status") == "RUNNING":
                    task["status"] = "PAUSED"
                    task["reconciled_at"] = now
                    task["reconciled_reason"] = "服务干净关机时停下，续跑是安全的。"
                    task["updated_at"] = now
                    affected.append(
                        {
                            "task_id": task.get("task_id"),
                            "goal": task.get("goal") or "",
                            "old_status": "RUNNING",
                            "new_status": "PAUSED",
                        }
                    )
                    changed = True
            if changed:
                self._write_all(data)
        return affected
