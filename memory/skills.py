# memory/skills.py（四件套第 4 块：技能自动生成——技能登记/生命周期/哈希账本/遥测）
#
# 技能库的唯一登记处。设计口径（定版，不重议）：
#
# - 形态借 Hermes：标准 SKILL.md（name + Use when 描述 + 触发条件/
#   步骤/坑点/适用边界/验证方法五段）+ 每技能一个目录（正文、meta、
#   usage 遥测、validation 验证记录）；渐进披露靠 name/description。
# - 门用自家的：生成侧过结晶门（skill_crystallize，MemOS 机制），
#   验证侧过留出集干跑＋两次一致（skill_crystallize 验证台），
#   转正侧必须人工点头（追加决定九）——本模块不实现、也不默认
#   启用任何自动转正。
# - 生命周期：candidate（候选）→ probation（试用）→ active（转正）
#   → stale（14 天未用）→ archived（再 30 天未用）；归档不删除、
#   可恢复；pinned 钉住的技能不进时钟流转。试用期被纠正一次即
#   sent_back（停用打回，追加决定十四）：旧版立即停用，修复优先
#   打补丁——补丁作为新候选版本，重跑完整验收才可再转正。
# - 账本：技能的状态变更事件（提案/生成/补丁/状态流转/打回/转正/
#   淘汰/恢复）写进主账本（type=skill，带前后内容哈希），与记忆
#   事件同流、可回溯；遥测是高频计数，留在技能自己的 usage.jsonl，
#   不灌主账本。哈希只证内容一致、不证对错（verify_integrity）。
# - 淘汰留痕：eliminate 只标状态＋记原因（可带被谁取代），文件与
#   记录都留着，不许无声消失。
# - 未过门的技能正式检索不可见（借 SkillWeaver 硬开关）：
#   retrievable_skills() 只返回 active 的。
# - 遥测失败不打断技能调用：usage 写失败只吞掉，计数照常更新。
#
# 本模块不认识任何模型：结晶的输入是做法字典，验证的执行器由
# 外面注入（见 skill_crystallize）。也不直接写记忆——与整理员的
# 接口只有提案入口（submit_proposal / accept_proposal）。

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from . import ledger as _ledger
from .skill_crystallize import (
    build_source_chain,
    check_crystallization_gate,
    check_proposal,
    practice_fingerprint,
)

# ---------------------------------------------------------------------------
# 状态与常量
# ---------------------------------------------------------------------------

STATE_CANDIDATE = "candidate"
STATE_PROBATION = "probation"
STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
STATE_SENT_BACK = "sent_back"
STATE_ELIMINATED = "eliminated"

ALL_STATES = (
    STATE_CANDIDATE,
    STATE_PROBATION,
    STATE_ACTIVE,
    STATE_STALE,
    STATE_ARCHIVED,
    STATE_SENT_BACK,
    STATE_ELIMINATED,
)

#: active 多久没用转 stale、再多久转 archived（借 Hermes Curator 阈值）。
STALE_AFTER_DAYS = 14
ARCHIVE_AFTER_DAYS = 14 + 30

_SECTION_KEYS = ("description", "trigger", "steps", "pitfalls", "boundary", "verification")


def _now_iso(dt):
    return dt.isoformat(timespec="seconds")


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return slug


def skill_id_for_practice(practice):
    """技能 id：名字 slug + 做法 id 的短哈希（同一做法重复结晶认得出）。"""
    base = _slugify(practice.get("name") or practice.get("title")) or "skill"
    suffix = hashlib.sha256(str(practice.get("id") or "").encode("utf-8")).hexdigest()[:8]
    return f"{base}-{suffix}"


def render_skill_md(name, sections, version, state):
    """把结构化内容渲染成标准 SKILL.md 文本（frontmatter + 五段）。"""
    lines = [
        "---",
        f"name: {name}",
        f"description: Use when {sections.get('description') or sections.get('trigger') or ''}",
        f"version: {version}",
        f"state: {state}",
        "---",
        "",
        "## 触发条件",
        str(sections.get("trigger") or ""),
        "",
        "## 步骤",
    ]
    steps = sections.get("steps") or []
    if isinstance(steps, str):
        steps = [steps]
    for idx, step in enumerate(steps, 1):
        lines.append(f"{idx}. {step}")
    lines += ["", "## 坑点"]
    pitfalls = sections.get("pitfalls") or []
    if isinstance(pitfalls, str):
        pitfalls = [pitfalls]
    for pit in pitfalls:
        lines.append(f"- {pit}")
    lines += [
        "",
        "## 适用边界",
        str(sections.get("boundary") or ""),
        "",
        "## 验证方法",
        str(sections.get("verification") or ""),
        "",
    ]
    return "\n".join(lines)


def _sections_from_practice(practice):
    verification = practice.get("verification_method") or practice.get("verification")
    if isinstance(verification, dict):
        verification = verification.get("method") or verification.get("text") or ""
    return {
        "description": practice.get("description") or "",
        "trigger": practice.get("trigger") or "",
        "steps": list(practice.get("steps") or []),
        "pitfalls": list(practice.get("pitfalls") or []),
        "boundary": practice.get("boundary") or practice.get("applicability") or "",
        "verification": verification or "",
    }


# ---------------------------------------------------------------------------
# 登记处
# ---------------------------------------------------------------------------


class SkillRegistry:
    """技能登记处：一个 memory_dir 一个实例。

    memory_dir 是记忆目录（与 SQLiteStore 同一个）：技能住
    memory_dir/skills/<skill-id>/，状态事件写 memory_dir/
    memory_events.jsonl 主账本（type=skill）。
    lock：外面有账本锁（如 store 的 RLock）就传进来共用，没有就
    靠 append_event 自身的读改写（技能操作低频，可接受）。
    now_fn：可注入的时钟（测试用），返回带时区的 datetime。
    """

    def __init__(self, memory_dir, *, lock=None, now_fn=None):
        self.memory_dir = Path(memory_dir)
        self.skills_dir = self.memory_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.memory_dir / "memory_events.jsonl"
        self._lock = lock
        self._now_fn = now_fn or (lambda: datetime.now().astimezone())

    # ---------- 基础设施 ----------

    def _now(self):
        return self._now_fn()

    def _now_str(self):
        return _now_iso(self._now())

    def _skill_dir(self, skill_id):
        return self.skills_dir / skill_id

    def _meta_path(self, skill_id):
        return self._skill_dir(skill_id) / "meta.json"

    def _content_path(self, skill_id):
        return self._skill_dir(skill_id) / "SKILL.md"

    def _read_meta(self, skill_id):
        path = self._meta_path(skill_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_meta(self, meta):
        path = self._meta_path(meta["skill_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _write_content(self, skill_id, text):
        path = self._content_path(skill_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _ledger_event(self, action, skill_id, **fields):
        event = {
            "type": _ledger.LEDGER_EVENT_SKILL,
            "action": action,
            "skill_id": skill_id,
        }
        event.update(fields)
        return _ledger.append_event(self.ledger_path, event, lock=self._lock)

    def _telemetry(self, skill_id, kind, detail=None):
        """写一条 usage 遥测。失败只吞掉返回 False——遥测不许打断正事。"""
        try:
            line = json.dumps(
                {"ts": self._now_str(), "kind": kind, "detail": detail or {}},
                ensure_ascii=False,
            )
            path = self._skill_dir(skill_id) / "usage.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return True
        except Exception:
            return False

    def _require(self, skill_id):
        meta = self._read_meta(skill_id)
        if meta is None:
            raise ValueError(f"技能 {skill_id!r} 不存在。")
        return meta

    # ---------- 生成侧：结晶与提案 ----------

    def crystallize(self, practice, *, actor=""):
        """从已验证做法结晶出候选技能。过不了结晶门抛 ValueError（带拒因）。

        同一做法重复结晶是幂等的：已存在（未淘汰）就返回现有记录，
        不重复生成、不重复落账。
        """
        ok, reasons = check_crystallization_gate(practice)
        if not ok:
            raise ValueError("结晶门不过：" + "；".join(reasons))
        skill_id = skill_id_for_practice(practice)
        existing = self._read_meta(skill_id)
        if existing is not None and existing.get("state") != STATE_ELIMINATED:
            return copy.deepcopy(existing)
        sections = _sections_from_practice(practice)
        name = practice.get("name") or practice.get("title") or skill_id
        content = render_skill_md(name, sections, version=1, state=STATE_CANDIDATE)
        content_hash = _sha256_text(content)
        now = self._now_str()
        meta = {
            "skill_id": skill_id,
            "name": name,
            "version": 1,
            "state": STATE_CANDIDATE,
            "pinned": False,
            "created_at": now,
            "updated_at": now,
            "last_used_at": None,
            "sections": sections,
            "content_hash": content_hash,
            "source_chain": build_source_chain(practice, skill_id),
            "practice_fingerprint": practice_fingerprint(practice),
            "versions": [
                {
                    "version": 1,
                    "hash": content_hash,
                    "created_at": now,
                    "mode": "create",
                    "outcome": "open",
                    "patch": None,
                }
            ],
            "counters": {
                "use_count": 0,
                "success_count": 0,
                "correction_count": 0,
                "patch_count": 0,
            },
            "probation": None,
            "approval": None,
            "validation": {},
            "eliminated": None,
        }
        self._write_content(skill_id, content)
        self._write_meta(meta)
        self._ledger_event(
            "create",
            skill_id,
            version=1,
            before_hash=None,
            after_hash=content_hash,
            actor=actor,
            evidence={"practice_id": practice.get("id")},
        )
        return copy.deepcopy(meta)

    def _proposals_path(self):
        return self.skills_dir / "proposals.json"

    def _read_proposals(self):
        path = self._proposals_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_proposals(self, proposals):
        path = self._proposals_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def submit_proposal(self, proposal, *, proposer=""):
        """收一份结晶候选提案（外部唯一入口）。按幂等键去重。

        形状不过（check_proposal）抛 ValueError；幂等键重复时返回
        已存的那份，不重复落账。
        """
        ok, reasons = check_proposal(proposal)
        if not ok:
            raise ValueError("提案形状不过：" + "；".join(reasons))
        key = proposal["idempotency_key"]
        proposals = self._read_proposals()
        if key in proposals:
            return copy.deepcopy(proposals[key])
        proposal_id = "prop-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        record = {
            "proposal_id": proposal_id,
            "idempotency_key": key,
            "practice": copy.deepcopy(proposal["practice"]),
            "expected_state": proposal.get("expected_state") or "candidate",
            "proposer": proposer or proposal.get("proposer") or "",
            "status": "pending",
            "reasons": [],
            "skill_id": None,
            "ts": self._now_str(),
        }
        proposals[key] = record
        self._write_proposals(proposals)
        self._ledger_event(
            "proposal",
            None,
            proposal_id=proposal_id,
            idempotency_key=key,
            actor=record["proposer"],
            evidence={"practice_id": record["practice"].get("id")},
        )
        return copy.deepcopy(record)

    def accept_proposal(self, proposal_id):
        """处理一份提案：过结晶门生成候选，或拒收并留痕。

        返回 {"status": "accepted", "skill": meta} 或
        {"status": "rejected", "reasons": [...]}。重复处理已接受的
        提案是幂等的（返回已生成的技能）。
        """
        proposals = self._read_proposals()
        record = None
        for item in proposals.values():
            if item.get("proposal_id") == proposal_id or item.get("idempotency_key") == proposal_id:
                record = item
                break
        if record is None:
            raise ValueError(f"提案 {proposal_id!r} 不存在。")
        if record["status"] == "accepted" and record.get("skill_id"):
            return {"status": "accepted", "skill": self.get_skill(record["skill_id"])}
        if record["status"] == "rejected":
            return {"status": "rejected", "reasons": list(record.get("reasons") or [])}
        try:
            meta = self.crystallize(record["practice"], actor=record.get("proposer") or "")
        except ValueError as exc:
            record["status"] = "rejected"
            record["reasons"] = [str(exc)]
            self._write_proposals(proposals)
            self._ledger_event(
                "proposal_rejected",
                None,
                proposal_id=record["proposal_id"],
                reason=str(exc),
                actor=record.get("proposer") or "",
            )
            return {"status": "rejected", "reasons": record["reasons"]}
        record["status"] = "accepted"
        record["skill_id"] = meta["skill_id"]
        self._write_proposals(proposals)
        return {"status": "accepted", "skill": meta}

    # ---------- 查询 ----------

    def get_skill(self, skill_id):
        meta = self._read_meta(skill_id)
        return copy.deepcopy(meta) if meta is not None else None

    def list_skills(self, state=None):
        out = []
        if self.skills_dir.exists():
            for child in sorted(self.skills_dir.iterdir()):
                if not child.is_dir():
                    continue
                meta = self._read_meta(child.name)
                if meta is None:
                    continue
                if state is None or meta.get("state") == state:
                    out.append(copy.deepcopy(meta))
        return out

    def retrievable_skills(self):
        """正式检索可见的技能：只有转正（active）的。未过门的不可见。"""
        return self.list_skills(state=STATE_ACTIVE)

    def verify_integrity(self, skill_id):
        """验当前 SKILL.md 与账本/meta 记录的哈希是否一致（只证一致）。"""
        meta = self._require(skill_id)
        path = self._content_path(skill_id)
        if not path.exists():
            return False
        actual = _sha256_text(path.read_text(encoding="utf-8"))
        return actual == meta.get("content_hash")

    # ---------- 验证记录（验证台的结果落这里） ----------

    def record_validation(self, skill_id, record):
        """落一条验证记录（干跑/两次一致），并更新本版本的过门摘要。

        记录由 skill_crystallize.run_dry_run / run_consistency 产出；
        这里补上 version 与时间，存 validation.jsonl，并把「留出池
        干跑通过」「留出池两次一致」两个过门标志按版本记进 meta。
        练习池的记录只留档，不动过门标志。
        """
        meta = self._require(skill_id)
        entry = dict(record)
        entry["version"] = meta["version"]
        entry["ts"] = self._now_str()
        path = self._skill_dir(skill_id) / "validation.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if entry.get("pool") == "exam":
            summary = meta.setdefault("validation", {}).setdefault(
                str(meta["version"]),
                {"exam_dry_run_passed": False, "exam_consistency_passed": False},
            )
            if entry.get("run_kind") == "dry_run" and entry.get("passed"):
                summary["exam_dry_run_passed"] = True
            if entry.get("run_kind") == "consistency" and entry.get("consistent"):
                summary["exam_consistency_passed"] = True
            meta["updated_at"] = self._now_str()
            self._write_meta(meta)
        return entry

    def validation_gate(self, skill_id):
        """进试用的验证门：本版本留出池干跑通过＋两次一致。返回 (ok, reasons)。"""
        meta = self._require(skill_id)
        summary = (meta.get("validation") or {}).get(str(meta["version"])) or {}
        reasons = []
        if not summary.get("exam_dry_run_passed"):
            reasons.append("本版本还没有留出池干跑通过记录（练习池干跑不算）。")
        if not summary.get("exam_consistency_passed"):
            reasons.append("本版本还没有留出池两次一致记录。")
        return (len(reasons) == 0), reasons

    # ---------- 状态机 ----------

    def _set_state(self, meta, new_state, action, **event_fields):
        old_state = meta["state"]
        meta["state"] = new_state
        meta["updated_at"] = self._now_str()
        # SKILL.md frontmatter 里带 state，状态变了正文要跟着变、哈希跟着变。
        content = render_skill_md(meta["name"], meta["sections"], meta["version"], new_state)
        before_hash = meta.get("content_hash")
        after_hash = _sha256_text(content)
        meta["content_hash"] = after_hash
        self._write_content(meta["skill_id"], content)
        self._write_meta(meta)
        self._ledger_event(
            action,
            meta["skill_id"],
            version=meta["version"],
            from_state=old_state,
            to_state=new_state,
            before_hash=before_hash,
            after_hash=after_hash,
            **event_fields,
        )
        return meta

    def begin_probation(self, skill_id):
        """候选 → 试用。先过验证门（留出集），不过抛 ValueError。"""
        meta = self._require(skill_id)
        if meta["state"] != STATE_CANDIDATE:
            raise ValueError(f"技能当前是 {meta['state']}，只有 candidate 能进试用。")
        ok, reasons = self.validation_gate(skill_id)
        if not ok:
            raise ValueError("验证门不过，不许进试用：" + "；".join(reasons))
        meta["probation"] = {
            "started_at": self._now_str(),
            "completed_at": None,
            "corrections": [],
        }
        return copy.deepcopy(self._set_state(meta, STATE_PROBATION, "probation_start"))

    def complete_probation(self, skill_id):
        """标记试用走完（无纠正）。只记完成，转正还差人工点头。"""
        meta = self._require(skill_id)
        if meta["state"] != STATE_PROBATION:
            raise ValueError(f"技能当前是 {meta['state']}，不在试用期。")
        probation = meta.get("probation") or {}
        if probation.get("corrections"):
            raise ValueError("试用期有纠正记录，不能标完成。")
        probation["completed_at"] = self._now_str()
        meta["probation"] = probation
        meta["updated_at"] = self._now_str()
        self._write_meta(meta)
        self._ledger_event(
            "probation_complete", skill_id, version=meta["version"]
        )
        return copy.deepcopy(meta)

    def promote(self, skill_id, *, approval):
        """试用 → 转正。必须带人工点头标记（追加决定九），没有拒。

        approval: {"approved_by": 谁点头, "note": ...}。自动转正未
        授权——本方法没有任何绕过 approval 的路径。
        """
        meta = self._require(skill_id)
        if meta["state"] != STATE_PROBATION:
            raise ValueError(f"技能当前是 {meta['state']}，只有试用期能转正。")
        probation = meta.get("probation") or {}
        if not probation.get("completed_at"):
            raise ValueError("试用还没走完（complete_probation 未标），不能转正。")
        if not isinstance(approval, dict) or not str(approval.get("approved_by") or "").strip():
            raise PermissionError(
                "转正必须带人工点头标记（approval.approved_by）——追加决定九，自动转正未授权。"
            )
        meta["approval"] = {
            "approved_by": approval["approved_by"],
            "note": approval.get("note") or "",
            "ts": self._now_str(),
        }
        return copy.deepcopy(self._set_state(meta, STATE_ACTIVE, "promote",
                                             actor=approval["approved_by"]))

    def record_correction(self, skill_id, correction):
        """记一次纠正。试用期内：立即停用打回（追加决定十四）。

        correction: {"source": 谁纠正的, "text": 纠正内容}。
        返回更新后的 meta。非试用期只记计数与遥测，不动状态。
        """
        meta = self._require(skill_id)
        correction = dict(correction or {})
        correction.setdefault("ts", self._now_str())
        meta["counters"]["correction_count"] = meta["counters"].get("correction_count", 0) + 1
        self._telemetry(skill_id, "correction", correction)
        if meta["state"] == STATE_PROBATION:
            probation = meta.get("probation") or {}
            probation.setdefault("corrections", []).append(correction)
            probation["sent_back_at"] = self._now_str()
            meta["probation"] = probation
            for ver in meta["versions"]:
                if ver["version"] == meta["version"]:
                    ver["outcome"] = "sent_back"
            meta = self._set_state(
                meta,
                STATE_SENT_BACK,
                "send_back",
                reason="试用期被纠正，停用打回（追加决定十四）",
                evidence={"correction": correction},
            )
        else:
            meta["updated_at"] = self._now_str()
            self._write_meta(meta)
        return copy.deepcopy(meta)

    def _apply_new_version(self, skill_id, new_sections, *, mode, correction, reason=""):
        meta = self._require(skill_id)
        if meta["state"] != STATE_SENT_BACK:
            raise ValueError(
                f"技能当前是 {meta['state']}，只有打回（sent_back）的版本能修了重来。"
            )
        old_content = render_skill_md(
            meta["name"], meta["sections"], meta["version"], STATE_SENT_BACK
        )
        merged = dict(meta["sections"])
        for key in _SECTION_KEYS:
            if key in new_sections and new_sections[key] is not None:
                merged[key] = new_sections[key]
        new_version = meta["version"] + 1
        new_content = render_skill_md(meta["name"], merged, new_version, STATE_CANDIDATE)
        diff = "\n".join(
            difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"SKILL.md v{meta['version']}",
                tofile=f"SKILL.md v{new_version}",
                lineterm="",
            )
        )
        before_hash = meta.get("content_hash")
        after_hash = _sha256_text(new_content)
        meta["sections"] = merged
        meta["version"] = new_version
        meta["content_hash"] = after_hash
        meta["state"] = STATE_CANDIDATE
        meta["probation"] = None
        meta["approval"] = None
        meta["counters"]["patch_count"] = meta["counters"].get("patch_count", 0) + 1
        meta["versions"].append(
            {
                "version": new_version,
                "hash": after_hash,
                "created_at": self._now_str(),
                "mode": mode,
                "outcome": "open",
                "patch": {
                    "from_version": new_version - 1,
                    "correction": correction or {},
                    "diff": diff,
                    "reason": reason,
                },
            }
        )
        meta["updated_at"] = self._now_str()
        self._write_content(skill_id, new_content)
        self._write_meta(meta)
        self._ledger_event(
            "patch" if mode == "patch" else "rewrite",
            skill_id,
            version=new_version,
            from_state=STATE_SENT_BACK,
            to_state=STATE_CANDIDATE,
            before_hash=before_hash,
            after_hash=after_hash,
            evidence={"correction": correction or {}},
            reason=reason,
        )
        return copy.deepcopy(meta)

    def apply_patch(self, skill_id, patch_sections, *, correction=None):
        """打补丁：改指定段落，生成新候选版本（追加决定十四的首选修法）。"""
        return self._apply_new_version(
            skill_id, patch_sections, mode="patch", correction=correction
        )

    def rewrite(self, skill_id, sections, *, correction=None, reason=""):
        """整份重写：只在补丁修不好时用，同样生成新候选版本重跑全验。"""
        return self._apply_new_version(
            skill_id, sections, mode="rewrite", correction=correction, reason=reason
        )

    # ---------- 使用与遥测 ----------

    def record_use(self, skill_id, *, success=True):
        """记一次真实使用：计数、最后使用时间、遥测（失败不打断）。

        stale 的技能被用上就恢复 active；archived/eliminated 不许
        直接用（先 restore / 已淘汰的不能用）。
        """
        meta = self._require(skill_id)
        if meta["state"] in (STATE_ARCHIVED, STATE_ELIMINATED):
            raise ValueError(f"技能当前是 {meta['state']}，不能直接使用。")
        meta["counters"]["use_count"] = meta["counters"].get("use_count", 0) + 1
        if success:
            meta["counters"]["success_count"] = meta["counters"].get("success_count", 0) + 1
        meta["last_used_at"] = self._now_str()
        meta["updated_at"] = self._now_str()
        self._write_meta(meta)
        self._telemetry(skill_id, "use", {"success": bool(success)})
        if meta["state"] == STATE_STALE:
            meta = self._set_state(meta, STATE_ACTIVE, "restore", reason="被再次使用，恢复 active")
        return copy.deepcopy(meta)

    def set_pinned(self, skill_id, pinned):
        meta = self._require(skill_id)
        meta["pinned"] = bool(pinned)
        meta["updated_at"] = self._now_str()
        self._write_meta(meta)
        self._ledger_event("pin" if pinned else "unpin", skill_id, version=meta["version"])
        return copy.deepcopy(meta)

    def restore(self, skill_id):
        """stale/archived → active（归档不删除、可恢复）。"""
        meta = self._require(skill_id)
        if meta["state"] not in (STATE_STALE, STATE_ARCHIVED):
            raise ValueError(f"技能当前是 {meta['state']}，只有 stale/archived 能恢复。")
        meta["last_used_at"] = self._now_str()
        return copy.deepcopy(self._set_state(meta, STATE_ACTIVE, "restore"))

    def eliminate(self, skill_id, *, reason, replaced_by=None):
        """淘汰：标状态＋记原因，文件与记录都留着，不许无声消失。"""
        meta = self._require(skill_id)
        if meta["state"] == STATE_ELIMINATED:
            return copy.deepcopy(meta)
        meta["eliminated"] = {
            "reason": reason,
            "replaced_by": replaced_by,
            "ts": self._now_str(),
        }
        return copy.deepcopy(
            self._set_state(meta, STATE_ELIMINATED, "eliminate",
                            reason=reason, replaced_by=replaced_by)
        )

    # ---------- 时钟流转 ----------

    def advance_lifecycle(self, now=None):
        """按时钟推进 stale/archived（借 Hermes Curator 阈值）。

        active 未用满 14 天 → stale；未用满 44 天（stale 后再 30 天）
        → archived。pinned 豁免。归档只是状态——文件、账本、来源链
        都在，随时可 restore，不存在「判死删除」。返回流转清单
        [{"skill_id", "from", "to"}...]。
        """
        now = now or self._now()
        transitions = []
        for meta in self.list_skills():
            if meta.get("pinned"):
                continue
            state = meta.get("state")
            if state not in (STATE_ACTIVE, STATE_STALE):
                continue
            anchor = meta.get("last_used_at") or meta.get("created_at")
            try:
                anchor_dt = datetime.fromisoformat(str(anchor))
            except (TypeError, ValueError):
                continue
            unused_days = (now - anchor_dt).days
            if state == STATE_ACTIVE and unused_days >= STALE_AFTER_DAYS:
                self._set_state(self._require(meta["skill_id"]), STATE_STALE, "stale",
                                reason=f"未用 {unused_days} 天，转 stale 待复核")
                transitions.append(
                    {"skill_id": meta["skill_id"], "from": STATE_ACTIVE, "to": STATE_STALE}
                )
            elif state == STATE_STALE and unused_days >= ARCHIVE_AFTER_DAYS:
                self._set_state(self._require(meta["skill_id"]), STATE_ARCHIVED, "archive",
                                reason=f"未用 {unused_days} 天，归档（不删除，可恢复）")
                transitions.append(
                    {"skill_id": meta["skill_id"], "from": STATE_STALE, "to": STATE_ARCHIVED}
                )
        return transitions
