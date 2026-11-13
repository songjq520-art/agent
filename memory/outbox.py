"""后台存记忆的 outbox（第 3 批 26 号）。

回答先回、记忆后存：run_question / web ask() 在回答完后把本轮消息
放进 outbox 就返回，不等记忆存完；后台线程按 turn_id 顺序慢慢存
（调 Mem0Store.add_turn，幂等：同一 turn_id 只存一次）。
outbox 保持精简：一个队列 + 一个后台线程 + turn_id 幂等 + 关机时
flush——别为架构漂亮加东西（GPT 第二份“机制够简单”原则）。

57 号根因修（2026-11-13）：待存轮次只放内存里，进程一死就丢——
11-13 早上冒菜、7 点起床、恒生定投三条就是这么丢的（服务重启前
发件箱没刷完）。修法：submit 时当场把轮次写盘（memory/outbox_pending/
<turn_id>.json，原子写），存完删文件；启动时把盘上没存完的捞回来重交。
"""

import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryOutbox:
    """一轮消息的后台存记忆队列。

    store：Mem0Store 实例（只调它的 add_turn）。
    线程是守护线程（进程退出时自己会走），但关机/退出前必须调 flush()
    等它存完——不然最后一轮的记忆可能没存上。
    57 号：submit 时轮次当场写盘，进程死了下次启动能捞回来重存。
    """

    def __init__(self, store, pending_dir=None):
        self._store = store
        # 待存轮次落盘目录：默认项目根目录下的 memory/outbox_pending/
        # （按本文件位置算，不依赖当前工作目录）。
        if pending_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            pending_dir = project_root / "memory" / "outbox_pending"
        self._pending_dir = Path(pending_dir)
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue()
        self._seen = set()
        self._seen_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._pending = 0
        self._pending_lock = threading.Lock()
        # 52 号回执：每轮的存完状态（turn_id → {"status": "committed" /
        # "failed", "result": add_turn 的返回}），调用方按 turn_id 查；
        # _failed 是 3 次重试都抛异常、真的丢了的轮次（带原消息）。
        self._outcomes = {}
        self._failed = {}
        self._outcomes_lock = threading.Lock()

    # 后台存记忆抛异常时重试几次（52 号）：add_turn 抛异常才到这里——
    # 提取失败（mem0.add 炸了）走返回字典里的 extraction_failed，
    # 由提交那层带着下一轮重试，不在这里重试。
    _SAVE_RETRIES = 3
    _SAVE_RETRY_DELAY = 1.0

    def _pending_path(self, turn_id):
        return self._pending_dir / f"{turn_id}.json"

    def _write_pending(self, turn_id, messages):
        """把待存轮次当场写盘（原子写：先写临时文件再 os.replace，
        Windows 上也不会写出一半的文件）。写盘失败只记日志、不拦提交——
        内存队列还在，进程不死就没事；死了才丢，和以前一样。"""
        try:
            tmp = self._pending_dir / f".{turn_id}.tmp"
            tmp.write_text(
                json.dumps(
                    {"turn_id": turn_id, "messages": list(messages)},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self._pending_path(turn_id))
        except Exception:
            logger.warning("发件箱写盘失败：turn_id=%s。", turn_id, exc_info=True)

    def _drop_pending(self, turn_id):
        """存完了，删掉落盘文件。删失败只记日志。"""
        try:
            self._pending_path(turn_id).unlink(missing_ok=True)
        except Exception:
            logger.warning("发件箱删盘文件失败：turn_id=%s。", turn_id, exc_info=True)

    def recover_pending(self):
        """启动时捞回盘上没存完的轮次（57 号）：按文件名排序重交进队列，
        返回捞回的轮数。文件坏了的跳过并记日志，不拦启动。"""
        recovered = 0
        try:
            files = sorted(self._pending_dir.glob("*.json"))
        except Exception:
            logger.warning("发件箱恢复：列盘文件失败。", exc_info=True)
            return 0
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                turn_id = data.get("turn_id") or path.stem
                messages = data.get("messages") or []
                if not messages:
                    self._drop_pending(turn_id)
                    continue
                with self._seen_lock:
                    if turn_id in self._seen:
                        continue
                    self._seen.add(turn_id)
                with self._pending_lock:
                    self._pending += 1
                self._queue.put((turn_id, list(messages)))
                recovered += 1
            except Exception:
                logger.warning("发件箱恢复：跳过坏文件 %s。", path.name, exc_info=True)
        if recovered:
            logger.info("发件箱恢复：捞回 %d 轮待存记忆。", recovered)
        return recovered

    # 后台存记忆抛异常时重试几次（52 号）：add_turn 抛异常才到这里——
    # 提取失败（mem0.add 炸了）走返回字典里的 extraction_failed，
    # 由提交那层带着下一轮重试，不在这里重试。
    _SAVE_RETRIES = 3
    _SAVE_RETRY_DELAY = 1.0

    def turn_status(self, turn_id):
        """一轮记忆的回执（52 号）：pending（还在队列里/正在存）、
        committed（存完了）、failed（存失败了）、unknown（没见过这个 turn_id）。"""
        with self._outcomes_lock:
            outcome = self._outcomes.get(turn_id)
        if outcome is not None:
            return outcome["status"]
        with self._seen_lock:
            return "pending" if turn_id in self._seen else "unknown"

    def failed_turns(self):
        """真的丢了的轮次（52 号）：重试 3 次都抛异常的，带原消息，
        调用方能查到、能重发；提取失败但带着下一轮重试的那些不在这里
        （它们没丢，只是晚一轮）。"""
        with self._outcomes_lock:
            return list(self._failed.values())

    def submit(self, messages, turn_id=None):
        """把一轮消息放进 outbox（不阻塞、不等存完）。
        turn_id 不传时现生成。同一 turn_id 只提交一次（重复提交直接丢掉）。
        57 号：提交时当场写盘，进程死了下次启动能捞回来。
        返回 turn_id。
        """
        turn_id = turn_id or uuid.uuid4().hex
        with self._seen_lock:
            if turn_id in self._seen:
                return turn_id
            self._seen.add(turn_id)
        self._write_pending(turn_id, messages)
        with self._pending_lock:
            self._pending += 1
        self._queue.put((turn_id, list(messages)))
        return turn_id

    def start(self):
        """启动后台线程（没启动时 submit 只攒着，flush 时当场存）。
        57 号：启动时先捞回盘上没存完的轮次。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self.recover_pending()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._drain, name="memory-outbox", daemon=True
        )
        self._thread.start()

    def _save_one(self, turn_id, messages):
        outcome = None
        raised = None
        for attempt in range(1, self._SAVE_RETRIES + 1):
            try:
                outcome = self._store.add_turn(messages)
                break
            except Exception as exc:  # noqa: BLE001 - 重试完还失败才记失败清单
                raised = exc
                logger.warning(
                    "后台存记忆失败（第 %d/%d 次）：turn_id=%s。",
                    attempt, self._SAVE_RETRIES, turn_id, exc_info=True,
                )
                if attempt < self._SAVE_RETRIES:
                    time.sleep(self._SAVE_RETRY_DELAY)
        with self._pending_lock:
            self._pending -= 1
        with self._outcomes_lock:
            if outcome is None:
                # 3 次都抛异常：记进失败清单（带原消息，别悄悄丢了）。
                # 57 号：盘文件留着，下次启动还能捞回来重存。
                self._failed[turn_id] = {
                    "turn_id": turn_id,
                    "messages": list(messages),
                    "error": str(raised) if raised is not None else "add_turn 失败",
                }
                self._outcomes[turn_id] = {"status": "failed", "result": None}
            else:
                # add_turn 的返回里带 extraction_failed / protection_failed
                # 时，是上一轮的提交没写成——那几句带着下一轮重试，
                # 这轮记 failed（不是丢了，是“这次没写成、等重试”）。
                failed = bool(
                    (outcome or {}).get("extraction_failed")
                    or (outcome or {}).get("protection_failed")
                )
                self._outcomes[turn_id] = {
                    "status": "failed" if failed else "committed",
                    "result": outcome,
                }
                if not failed:
                    # 存成了，删掉落盘文件（57 号）。
                    self._drop_pending(turn_id)

    def _drain(self):
        """后台线程主循环：按顺序取一轮、存一轮。"""
        while not self._stop.is_set():
            try:
                turn_id, messages = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._save_one(turn_id, messages)
            self._queue.task_done()

    def pending_count(self):
        """还没存完的轮数（队列里 + 正在存的）。"""
        with self._pending_lock:
            return self._pending

    def flush(self, timeout=30):
        """等队列里的都存完（关机/退出前调）。超时后先停后台线程、
        再在调用线程里把剩下的当场存完——不丢轮次。
        返回 True（都存完了）/ False（超时且当场存也超时了——
        基本不会发生，除非 add_turn 本身卡住）。
        """
        deadline = time.monotonic() + timeout
        while self.pending_count() > 0:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if self.pending_count() <= 0:
            return True
        # 超时了：停后台线程（等它把手头那轮存完），剩下的当场存。
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        while True:
            try:
                turn_id, messages = self._queue.get_nowait()
            except queue.Empty:
                break
            self._save_one(turn_id, messages)
            self._queue.task_done()
        return self.pending_count() <= 0

    def stop(self):
        """停后台线程（先 flush 再停）。"""
        self.flush()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
