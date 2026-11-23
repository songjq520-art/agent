# memory/pending_questions.py（Phase 2 #4：待问 BOOS 的问题，攒批 + 回答）
#
# 十二拿不准的事（比如一条新记忆重不重要）不能自己瞎猜、也不能悄悄
# 丢——记成一个问题，等 BOOS 回答。分两种问法：
#   聊天里问：本轮 save_turn 拿不准且这事重要时，问题进回执的
#     "questions"，页面在回答下面展示（带原文、给选项、说原因）；
#   后台攒批：不急的问题先攒着（pending），下次聊天有新问题时顺带
#     一起问（drain_for_chat 最多 3 条），别一条一条打扰。
# 只存问题、不改记忆：回答了才按答案改库（见 sqlite_store.answer_question）。
# 落盘文件 memory/pending_questions.json，原子写（先写临时文件再
# os.replace），进程死了问题不丢；启动时照常读回来。
# 纯标准库，不读库、不调模型、不联网。

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 问题种类：目前只有 importance（这条新记忆重不重要、进不进核心）。
QUESTION_KIND_IMPORTANCE = "importance"

# 问题状态：pending（攒着、还没问过）、asked（已在某轮回复里问过、
# 等回答）、answered（已回答、按答案处理完）。
QUESTION_STATUS_PENDING = "pending"
QUESTION_STATUS_ASKED = "asked"
QUESTION_STATUS_ANSWERED = "answered"

# 重要性问题的选项（固定三个，页面按这个展示）：
#   进核心：长期钉住记住（decay_kind 改 stable）；
#   当一次性经历：先当一次性经历存着（3 天 3 次兜底还能晋升）；
#   不存：这条不要了，删掉。
IMPORTANCE_QUESTION_OPTIONS = ("进核心", "当一次性经历", "不存")

# 一轮回复里最多顺带问几条（含本轮新问题）：多了 BOOS 看不过来。
QUESTIONS_PER_CHAT_MAX = 3


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


class PendingQuestions:
    """待问 BOOS 的问题（一个 JSON 文件 + 一把锁）。

    path 不传时默认 memory_dir/pending_questions.json；
    memory_dir 也不传时默认当前目录下的 memory/pending_questions.json
    （测试传 tmp_path 更方便）。文件坏了/读不出来时按空算，不崩——
    问题丢了可以再攒，绝不能因为问题文件坏了拦住聊天。
    """

    def __init__(self, path=None, memory_dir=None):
        if path is not None:
            self._path = Path(path)
        elif memory_dir is not None:
            self._path = Path(memory_dir) / "pending_questions.json"
        else:
            self._path = Path("memory") / "pending_questions.json"
        self._lock = threading.RLock()
        self._questions = self._load()

    def _load(self):
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return []
        if not isinstance(data, dict):
            return []
        items = data.get("questions") or []
        if not isinstance(items, list):
            return []
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("id") or not item.get("text"):
                continue
            out.append(dict(item))
        return out

    def _save(self):
        """原子写盘：先写临时文件再 os.replace。写失败只记不抛——
        问题存不下来不拦聊天，下次还能再攒。"""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(".%s.tmp" % self._path.name)
            tmp.write_text(
                json.dumps(
                    {"questions": self._questions}, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except Exception:
            pass

    def add(self, kind, text, options, reason, memory_id=None, turn_id=None):
        """攒一个新问题，返回问题 dict。

        同一条记忆、同一种问题还没回答时不重复攒（返回已有的那条）——
        一轮里同一句问两遍是打扰。options 传列表/元组，原样存。
        """
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            for item in self._questions:
                if (
                    item.get("status") in (QUESTION_STATUS_PENDING, QUESTION_STATUS_ASKED)
                    and item.get("kind") == kind
                    and (memory_id is not None and item.get("memory_id") == memory_id)
                ):
                    return dict(item)
            question = {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "text": text,
                "options": list(options or []),
                "reason": (reason or "").strip(),
                "memory_id": memory_id,
                "turn_id": turn_id,
                "created_at": _now_iso(),
                "status": QUESTION_STATUS_PENDING,
                "answer": None,
            }
            self._questions.append(question)
            self._save()
            return dict(question)

    def get(self, question_id):
        """按 id 取一条问题（没有返回 None）。"""
        with self._lock:
            for item in self._questions:
                if item.get("id") == question_id:
                    return dict(item)
        return None

    def list_unanswered(self):
        """还没回答的问题（pending + asked），按创建时间从旧到新。"""
        with self._lock:
            return [
                dict(item)
                for item in sorted(
                    self._questions, key=lambda q: q.get("created_at") or ""
                )
                if item.get("status")
                in (QUESTION_STATUS_PENDING, QUESTION_STATUS_ASKED)
            ]

    def count_unanswered(self):
        """还没回答的问题数。"""
        return len(self.list_unanswered())

    def drain_for_chat(self, max_n=QUESTIONS_PER_CHAT_MAX):
        """取一批攒着的问题去聊天里问：只取 pending 的（asked 的已经
        问过一遍了，不重复问），按创建时间从旧到新，最多 max_n 条；
        取出来的标成 asked（等回答）。返回问题 dict 列表。"""
        with self._lock:
            pending = [
                item
                for item in sorted(
                    self._questions, key=lambda q: q.get("created_at") or ""
                )
                if item.get("status") == QUESTION_STATUS_PENDING
            ][:max_n]
            for item in pending:
                item["status"] = QUESTION_STATUS_ASKED
            if pending:
                self._save()
            return [dict(item) for item in pending]

    def mark_asked(self, question_id):
        """把一条问题标成已问过（它正在某轮回复里被问，等回答）。
        找不到、或已经回答过时返回 False，不重复标。"""
        with self._lock:
            for item in self._questions:
                if item.get("id") != question_id:
                    continue
                if item.get("status") == QUESTION_STATUS_ANSWERED:
                    return False
                if item.get("status") == QUESTION_STATUS_ASKED:
                    return True
                item["status"] = QUESTION_STATUS_ASKED
                self._save()
                return True
        return False

    def answer(self, question_id, choice):
        """BOOS 回答了一个问题：标成 answered、记下答案，返回问题 dict；
        找不到、或已经回答过时返回 None（不重复处理）。"""
        with self._lock:
            for item in self._questions:
                if item.get("id") != question_id:
                    continue
                if item.get("status") == QUESTION_STATUS_ANSWERED:
                    return None
                item["status"] = QUESTION_STATUS_ANSWERED
                item["answer"] = choice
                self._save()
                return dict(item)
        return None
