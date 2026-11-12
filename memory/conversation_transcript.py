# memory/conversation_transcript.py（对话全文存盘 + 保留期流转：38 号）
# 作用：每场对话存一份全文记录，点开旧对话能从头看到尾；
# 保留期按 BOOS 2026-11-12 定的来：完整对话保存 1 个月，1 个月后
# 变摘要，1 年后再精简摘要。摘要只当提示，不当全文的替代。
# 做法：每场对话一个文件（memory/conversation_transcripts/<对话 id>.json），
# 和摘要文件（conversation_summary.json）分开——摘要每轮都重写，
# 全文放一起会让那个文件越长越长；分开后摘要还是管列表和提示，
# 全文只在这里管。记录只放本机数据，不上传（见 .gitignore）。
# 写入：每轮结束时由 web/server.py 的 _finish_turn_locked 把本轮两条
# 消息 append 进来（只追加新消息，恢复回来的旧消息不重复写）；
# 摘要压缩裁掉历史前面时不影响这里——全文已经在文件里了。
# 保留期：prune() 在服务启动时跑一遍——满 30 天的全文变摘要
# （摘要优先用摘要文件里那份，没有时调模型现生成；都拿不到就不动，
# 宁可多留全文、不丢数据）；满 1 年的摘要再精简一次（调模型压短）。
# 什么都不删：最老的也只留精简版摘要，全文转摘要后消息清空，但
# 标题、时间、摘要本身一直留着。
# 线程：本类自己不加锁——调用方（ChatState 的锁、服务启动时单线程）
# 保证一次只有一个地方在调；不要两个线程同时调同一个 store。

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("agent")

# TRANSCRIPT_FULL_DAYS（全文保留期）：完整对话保存 1 个月（30 天），
# 到了就变摘要（BOOS 2026-11-12 定）。
TRANSCRIPT_FULL_DAYS = 30
# TRANSCRIPT_CONDENSE_DAYS（摘要再精简期）：摘要满 1 年（365 天）后
# 再精简一次（BOOS 2026-11-12 定）。
TRANSCRIPT_CONDENSE_DAYS = 365

# TRANSCRIPT_STATUSES（全文记录的状态）：full=全文还在，summary=已变摘要，
# condensed=摘要已精简过一次。只往前走（full → summary → condensed），
# 不往回走；状态名写错不报错——读到不认识的状态按 full 处理（不丢数据）。
TRANSCRIPT_STATUSES = ("full", "summary", "condensed")


def message_fingerprint(message):
    """一条消息的指纹（角色 + 内容算 SHA-256）：恢复的那轮插回全文
    中间时，按后面那条消息的指纹定位插到哪，不靠下标（下标在摘要
    压缩裁掉前面后就对不上了）。"""
    text = f"{(message or {}).get('role')}:{(message or {}).get('content') or ''}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso():
    """现在时间的 UTC ISO 串（带 Z），和 conversation_summary._now_iso
    同一个格式（固定 6 位微秒，字符串顺序就是时间顺序）。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_time(value):
    """解析一条 ISO 时间串（带 Z 或不带）；解析不了返回 None。"""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _copy_messages(messages):
    """复制一段消息列表（只留 role/content，调用方改了原列表也不影响存档）。"""
    return [
        {"role": (msg or {}).get("role"), "content": (msg or {}).get("content") or ""}
        for msg in (messages or [])
        if (msg or {}).get("role") in ("user", "assistant")
    ]


class TranscriptStore:
    """对话全文记录的存取（每场对话一个文件）。

    文件默认在项目 memory/ 目录下（conversation_transcripts/<对话 id>.json），
    和 mem0 的本地数据放一起；.gitignore 已覆盖，不会上传到 GitHub。
    文件格式 {"id", "title", "started_at", "updated_at", "status",
    "messages", "summary", "summarized_at", "condensed_at"}。
    对话 id 只认字母数字、下划线、横线（防路径穿越）；文件损坏、
    缺字段时按空处理（get 返回 None），不 crash、不猜。
    保存用临时文件 + os.replace（原子写），写坏了旧文件还在。
    """

    def __init__(self, directory=None):
        # 目录位置：显式传了 directory 就用它；没传时看环境变量
        # AGENT_TRANSCRIPT_DIR（测试指到临时目录用），再没有就用默认的
        # memory/conversation_transcripts/。
        if directory is None:
            directory = os.environ.get("AGENT_TRANSCRIPT_DIR")
        self.directory = (
            Path(directory)
            if directory
            else Path(__file__).resolve().parent / "conversation_transcripts"
        )

    # ---------- 存取 ----------

    def _path(self, conv_id):
        """一场对话的文件路径：id 只认字母数字、下划线、横线，
        别的字符（比如 ../）直接报错，不拼路径。"""
        conv_id = str(conv_id or "")
        if not conv_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for ch in conv_id
        ):
            raise ValueError(f"对话 id 只认字母数字、下划线、横线：{conv_id!r}。")
        return self.directory / f"{conv_id}.json"

    def _load(self, conv_id):
        """读一条全文记录；没有文件或文件坏了时返回 None。"""
        try:
            raw = self._path(conv_id).read_text(encoding="utf-8")
            data = json.loads(raw)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logger.warning("对话全文记录读失败（%s），按没有处理。", conv_id)
            return None
        if not isinstance(data, dict) or not data.get("id"):
            return None
        status = data.get("status")
        return {
            "id": str(data["id"]),
            "title": str(data.get("title") or "新对话"),
            "started_at": data.get("started_at"),
            "updated_at": data.get("updated_at"),
            "status": status if status in TRANSCRIPT_STATUSES else "full",
            "messages": _copy_messages(data.get("messages") or []),
            "summary": data.get("summary") if isinstance(data.get("summary"), str) else "",
            "summarized_at": data.get("summarized_at"),
            "condensed_at": data.get("condensed_at"),
        }

    def _save(self, record):
        """把一条记录写回硬盘（临时文件 + os.replace，写坏了旧文件还在）。
        id 非法时直接报错——那是调用方传错了，不该悄悄丢数据；
        只有真的写不进盘才记日志、不拦聊天。"""
        path = self._path(record["id"])
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(record, ensure_ascii=False)
            fd, tmp_name = tempfile.mkstemp(
                prefix=record["id"] + ".tmp-",
                dir=str(self.directory),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except Exception:
            # 全文存不下来不能把聊天搞崩：记一条日志，内存里的记录还在，
            # 下次 append 再试。
            logger.warning("对话全文记录保存失败。", exc_info=True)
            return False
        return True

    def get(self, conv_id):
        """按 id 取一条全文记录的完整内容，找不到返回 None。"""
        try:
            return self._load(conv_id)
        except ValueError:
            return None

    def list_records(self):
        """全部全文记录（按更新时间从新到旧），给保留期流转用。"""
        records = []
        try:
            files = sorted(self.directory.glob("*.json"))
        except OSError:
            return []
        for path in files:
            record = self._load(path.stem)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (r.get("updated_at") or ""), reverse=True)
        return records

    def append(self, conv_id, title, messages, insert_before_fingerprint=None):
        """把几条新消息追加进一场对话的全文记录（没有就新建一条）。

        只追加调用方传进来的新消息——恢复回来的旧消息不经过这里，
        不重复写。insert_before_fingerprint 给了时（恢复的那轮插在
        历史中间），按后面那条消息的指纹定位、插回原来的位置；
        找不到那条指纹时追加到最后（宁可顺序差一点、不丢消息）。
        标题只在新建时、或原来是“新对话”时更新，不覆盖已有标题。
        返回存下来的记录（副本）；写失败时返回 None。
        """
        record = self._load(conv_id)
        now = _now_iso()
        if record is None:
            record = {
                "id": str(conv_id),
                "title": title or "新对话",
                "started_at": now,
                "updated_at": now,
                "status": "full",
                "messages": [],
                "summary": "",
                "summarized_at": None,
                "condensed_at": None,
            }
        new_messages = _copy_messages(messages)
        if not new_messages:
            return dict(record)
        if insert_before_fingerprint:
            index = None
            for i, msg in enumerate(record["messages"]):
                if message_fingerprint(msg) == insert_before_fingerprint:
                    index = i
                    break
            if index is None:
                record["messages"].extend(new_messages)
            else:
                record["messages"][index:index] = new_messages
        else:
            record["messages"].extend(new_messages)
        if title and title != "新对话" and record.get("title") in (None, "", "新对话"):
            record["title"] = title
        record["updated_at"] = now
        if not record.get("started_at"):
            record["started_at"] = now
        return dict(record) if self._save(record) else None

    # ---------- 保留期流转 ----------

    def prune(self, summarizer=None, summary_lookup=None, now=None):
        """按保留期流转全部全文记录（服务启动时跑一遍）：

        - 满 30 天的全文（status=full）变摘要：摘要优先用 summary_lookup
          （摘要文件里那份）给的，没有时调 summarizer 现生成；
          两样都没有就不动（宁可多留全文、不丢数据），下次启动再试。
        - 满 1 年的摘要（status=summary）再精简一次：调 summarizer 把
          摘要压短，压出来的比原来短才换；没有 summarizer 就不动。
        - 什么都不删：最老的也只留精简版摘要，标题、时间、摘要本身
          一直留着。返回流转的动作 [{conv_id, action, ...}]，没有时返回 []。
        参数 summarizer：summarizer(turns, previous_summary) -> str | None，
        和 ConversationSummaryStore 用的同一个签名；不传时只做“有现成
        摘要就能转”的那部分。summary_lookup：summary_lookup(conv_id) -> str，
        查摘要文件里那份摘要；不传时没有现成摘要。
        参数 now：现在时间（datetime，默认取当前时间；测试传固定值）。
        """
        now_dt = now or datetime.now(timezone.utc)
        actions = []
        for record in self.list_records():
            updated_at = _parse_time(record.get("updated_at")) or _parse_time(
                record.get("started_at")
            )
            if updated_at is None:
                continue
            age_days = (now_dt - updated_at).days
            status = record.get("status")
            if status == "full" and age_days > TRANSCRIPT_FULL_DAYS:
                summary = ""
                if summary_lookup is not None:
                    try:
                        summary = (summary_lookup(record["id"]) or "").strip()
                    except Exception:
                        logger.warning(
                            "全文转摘要时查摘要文件失败：%s。", record["id"], exc_info=True
                        )
                        summary = ""
                if not summary and summarizer is not None:
                    try:
                        summary = (summarizer(list(record["messages"]), None) or "").strip()
                    except Exception:
                        logger.warning(
                            "全文转摘要时生成摘要失败：%s。", record["id"], exc_info=True
                        )
                        summary = ""
                if not summary:
                    # 摘要拿不到：不动，宁可多留全文、不丢数据，下次启动再试。
                    continue
                record["messages"] = []
                record["summary"] = summary
                record["status"] = "summary"
                record["summarized_at"] = _now_iso()
                record["updated_at"] = _now_iso()
                if self._save(record):
                    actions.append(
                        {
                            "conv_id": record["id"],
                            "action": "full_to_summary",
                            "title": record.get("title") or "",
                        }
                    )
            elif status == "summary" and age_days > TRANSCRIPT_CONDENSE_DAYS:
                if summarizer is None:
                    continue
                old_summary = record.get("summary") or ""
                if not old_summary.strip():
                    continue
                try:
                    condensed = (
                        summarizer(
                            [{"role": "assistant", "content": old_summary}], None
                        )
                        or ""
                    ).strip()
                except Exception:
                    logger.warning(
                        "摘要再精简失败：%s。", record["id"], exc_info=True
                    )
                    continue
                if not condensed or len(condensed) >= len(old_summary):
                    # 没压短（模型原样回了、或失败了）：不动，下次启动再试。
                    continue
                record["summary"] = condensed
                record["status"] = "condensed"
                record["condensed_at"] = _now_iso()
                record["updated_at"] = _now_iso()
                if self._save(record):
                    actions.append(
                        {
                            "conv_id": record["id"],
                            "action": "summary_to_condensed",
                            "title": record.get("title") or "",
                        }
                    )
        return actions
