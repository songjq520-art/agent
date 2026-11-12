"""后台存记忆的 outbox（第 3 批 26 号）。

回答先回、记忆后存：run_question / web ask() 在回答完后把本轮消息
放进 outbox 就返回，不等记忆存完；后台线程按 turn_id 顺序慢慢存
（调 Mem0Store.add_turn，幂等：同一 turn_id 只存一次）。
outbox 保持精简：一个队列 + 一个后台线程 + turn_id 幂等 + 关机时
flush——别为架构漂亮加东西（GPT 第二份“机制够简单”原则）。
"""

import logging
import queue
import threading
import time
import uuid

logger = logging.getLogger(__name__)


class MemoryOutbox:
    """一轮消息的后台存记忆队列。

    store：Mem0Store 实例（只调它的 add_turn）。
    线程是守护线程（进程退出时自己会走），但关机/退出前必须调 flush()
    等它存完——不然最后一轮的记忆可能没存上。
    """

    def __init__(self, store):
        self._store = store
        self._queue = queue.Queue()
        self._seen = set()
        self._seen_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._pending = 0
        self._pending_lock = threading.Lock()

    def submit(self, messages, turn_id=None):
        """把一轮消息放进 outbox（不阻塞、不等存完）。
        turn_id 不传时现生成。同一 turn_id 只提交一次（重复提交直接丢掉）。
        返回 turn_id。
        """
        turn_id = turn_id or uuid.uuid4().hex
        with self._seen_lock:
            if turn_id in self._seen:
                return turn_id
            self._seen.add(turn_id)
        with self._pending_lock:
            self._pending += 1
        self._queue.put((turn_id, list(messages)))
        return turn_id

    def start(self):
        """启动后台线程（没启动时 submit 只攒着，flush 时当场存）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._drain, name="memory-outbox", daemon=True
        )
        self._thread.start()

    def _save_one(self, turn_id, messages):
        try:
            self._store.add_turn(messages)
        except Exception:
            logger.warning("后台存记忆失败：turn_id=%s。", turn_id, exc_info=True)
        finally:
            with self._pending_lock:
                self._pending -= 1

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
