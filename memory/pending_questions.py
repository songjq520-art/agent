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
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# 问题种类：importance（这条新记忆重不重要、进不进核心）；
# conflict（推断记忆保护，2026-11-24）：一条推断来的、未确认的旧
# 记忆和新事实冲突时，要不要换成新的——不许悄悄顶掉，先问 BOOS。
QUESTION_KIND_IMPORTANCE = "importance"
QUESTION_KIND_CONFLICT = "conflict"
QUESTION_KIND_CORE_CAPACITY = "core_capacity"
QUESTION_KIND_AUTH_CLAIM = "auth_claim"
# 入口挂起问题（四件套·实测问题三，追加决定七）：入口当场拦
# 拿不准的候选不落记忆行、先挂起，BOOS 裁这句存还是不存。
QUESTION_KIND_INTAKE_HOLD = "intake_hold"

# 冲突保护问题的选项（固定两个）：换成新的 = 按答案取代旧记忆；
# 不换 = 两条都留着，这对冲突记为已裁决，不再重复问。
CONFLICT_QUESTION_OPTIONS = ("换成新的", "不换")
# 核心容量问题的选项（四件套第 1 块）：满座被容量门拦下的候选，
# 先留作候选 = 只记下问题、记忆保持原样（还没进核心）；不要了 =
# 按 BOOS 的回答删掉那条（走 delete_memory 的批准删除路）。
CORE_CAPACITY_QUESTION_OPTIONS = ("先留作候选", "不要了")
# 授权核实问题的选项（四件套·实测问题一）：是真的 = 这话确实是
# BOOS 本人授权，那条按本人确认升级；不是真的 = 转述不实，
# 那条删掉。非本人通道改记忆被拒的留痕问题只用「知道了」。
AUTH_CLAIM_QUESTION_OPTIONS = ("是真的", "不是真的")
CHANGE_ATTEMPT_QUESTION_OPTIONS = ("知道了",)
# 入口挂起问题的选项：存 = 按 BOOS 裁决把原文回存；不存 =
# 这句就不要了（本来就没落行，只消耗问题）。
INTAKE_HOLD_QUESTION_OPTIONS = ("存", "不存")

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

# 问过没人答的问题，隔多久可以在聊天里再带出来问一次（R-2，
# 2026-11-24）：问一次没答不等于这事就放下了，但也不能每轮都
# 追着问——隔 7 天才重问，且同一个问题最多在聊天里问
# QUESTION_MAX_ASKS 次（第一次 + 两次重问），到顶就只留在待答
# 列表里（/api/questions 还能答），不再往聊天里带。
REASK_AFTER_DAYS = 7
QUESTION_MAX_ASKS = 3


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_iso(value):
    """解析 _now_iso 写的时间串；解析不了返回 None（调用方按没时间算）。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class PendingQuestions:
    """待问 BOOS 的问题（一个 JSON 文件 + 一把锁）。

    path 不传时默认 memory_dir/pending_questions.json；
    memory_dir 也不传时默认当前目录下的 memory/pending_questions.json
    （测试传 tmp_path 更方便）。文件坏了不崩，但不许静默按空算
    （追加决定十五）：原件先改名保全、load_error 报警，store 启动恢复按存活记录重建能重建的部分；文件不存在是第一次跑，不算坏。
    """

    def __init__(self, path=None, memory_dir=None):
        if path is not None:
            self._path = Path(path)
        elif memory_dir is not None:
            self._path = Path(memory_dir) / "pending_questions.json"
        else:
            self._path = Path("memory") / "pending_questions.json"
        self._lock = threading.RLock()
        self.load_error = None
        self._questions = self._load()

    def _preserve_corrupt(self, reason):
        """保全损坏的问题文件原件并报警（崩溃恢复，追加决定十五）。

        原件改名为 pending_questions.json.corrupt-<时间戳> 留在同
        目录——坏成什么样就是什么样，死无对证的事不许干。改名
        失败（权限之类）也照样把 load_error 记下来：报警不能丢，
        store 的 pending_notices 会把它带到结论可见处。"""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        backup = self._path.with_name(
            "%s.corrupt-%s" % (self._path.name, stamp)
        )
        n = 2
        while backup.exists():
            backup = self._path.with_name(
                "%s.corrupt-%s-%d" % (self._path.name, stamp, n)
            )
            n += 1
        moved = False
        try:
            os.replace(self._path, backup)
            moved = True
        except OSError:
            logger.error(
                "pending_questions 文件损坏且保全改名失败：%s",
                self._path, exc_info=True,
            )
        self.load_error = {
            "path": str(self._path),
            "backup_path": str(backup) if moved else None,
            "reason": reason,
            "at": _now_iso(),
        }
        logger.error(
            "pending_questions 文件损坏（%s）：原件保全到 %s，"
            "本次按空加载，待启动恢复按存活记录重建。",
            reason, str(backup) if moved else "（改名失败，原地保留）",
        )

    def recovery_report(self):
        """文件损坏的痕迹：没坏过返回 None；坏过返回 load_error
        副本（原件保全在哪、坏在哪、什么时候）。"""
        if not self.load_error:
            return None
        return dict(self.load_error)

    def _load(self):
        # 文件不存在是第一次跑，不算损坏；文件在却读不出/解析
        # 不了/形状不对 = 损坏（追加决定十五）：不许再静默按空
        # 算——先保全原件、记 load_error 报警，再按空起步，问题
        # 由 store 启动恢复按账本/入口留痕重建能重建的部分。
        self.load_error = None
        if not self._path.exists():
            return []
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self._preserve_corrupt("read_failed: %s" % exc)
            return []
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            self._preserve_corrupt("bad_json: %s" % exc)
            return []
        if not isinstance(data, dict):
            self._preserve_corrupt("bad_shape: top level is not an object")
            return []
        items = data.get("questions") or []
        if not isinstance(items, list):
            self._preserve_corrupt("bad_shape: questions is not a list")
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
        except Exception as exc:
            logger.warning("pending_questions 保存失败，问题可能丢失：%s", exc, exc_info=True)

    def add(self, kind, text, options, reason, memory_id=None, turn_id=None,
            extra=None):
        """攒一个新问题，返回问题 dict。

        同一条记忆、同一种问题还没回答时不重复攒（返回已有的那条）——
        一轮里同一句问两遍是打扰。options 传列表/元组，原样存。
        extra 是可选的附加字段（冲突保护问题带 old_id/new_id/新旧
        原文，回答时按它执行取代）；只补没有的键，不许盖掉问题
        本身的固有字段。
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
                # 在聊天里问过几次、最近一次是什么时候问的（R-2 重问
                # 间隔/上限靠这两个字段；老文件里没有时按未问过算）。
                "asked_at": None,
                "ask_count": 0,
            }
            if extra:
                for _k, _v in extra.items():
                    question.setdefault(_k, _v)
            self._questions.append(question)
            self._save()
            return dict(question)

    def find_questions(self, kind=None, statuses=None):
        """按种类/状态找问题（返回副本列表，含已回答的）。

        冲突保护按 (old_id, new_id) 配对查用：同一对冲突是不是
        已经问过、是不是已经裁决过（答了"不换"），都认这里。
        """
        with self._lock:
            out = []
            for item in self._questions:
                if kind is not None and item.get("kind") != kind:
                    continue
                if statuses is not None and item.get("status") not in statuses:
                    continue
                out.append(dict(item))
            return out

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

    def _recycle_stale_asked(self, now):
        """把问过太久没人答的问题回滚成 pending（R-2，调用方持锁）。

        条件：状态是 asked、上次问的时间（asked_at，老数据没有时按
        created_at 算）距今 ≥ REASK_AFTER_DAYS 天、且总共问过的次数
        还没到 QUESTION_MAX_ASKS。返回有没有改动。"""
        changed = False
        for item in self._questions:
            if item.get("status") != QUESTION_STATUS_ASKED:
                continue
            if int(item.get("ask_count") or 1) >= QUESTION_MAX_ASKS:
                continue
            asked_at = _parse_iso(item.get("asked_at") or item.get("created_at"))
            if asked_at is None:
                continue
            if (now - asked_at).days >= REASK_AFTER_DAYS:
                item["status"] = QUESTION_STATUS_PENDING
                changed = True
        return changed

    def drain_for_chat(self, max_n=QUESTIONS_PER_CHAT_MAX):
        """取一批攒着的问题去聊天里问：pending 的按创建时间从旧到新，
        最多 max_n 条；取出来的标成 asked（等回答），并记 asked_at /
        ask_count。返回问题 dict 列表。

        R-2（2026-11-24）：取之前先把「问过超过 REASK_AFTER_DAYS 天
        还没人答、且没问满 QUESTION_MAX_ASKS 次」的 asked 问题回滚
        成 pending，让它们能被再带出来问一次——以前 asked 的永远
        不重问，问一次没答，这事就等于被悄悄放下了。"""
        with self._lock:
            changed = self._recycle_stale_asked(datetime.now(timezone.utc))
            pending = [
                item
                for item in sorted(
                    self._questions, key=lambda q: q.get("created_at") or ""
                )
                if item.get("status") == QUESTION_STATUS_PENDING
            ][:max_n]
            for item in pending:
                item["status"] = QUESTION_STATUS_ASKED
                item["asked_at"] = _now_iso()
                item["ask_count"] = int(item.get("ask_count") or 0) + 1
            if pending or changed:
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
                item["asked_at"] = _now_iso()
                item["ask_count"] = int(item.get("ask_count") or 0) + 1
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
