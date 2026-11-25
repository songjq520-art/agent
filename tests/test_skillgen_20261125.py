# tests/test_skillgen_20261125.py（四件套第 4 块：技能自动生成）
#
# 覆盖契约「六、验收·技能验收三条」＋追加决定九/十四与施工设计草稿：
# ① 结晶门（MemOS 机制）：原始经历/未验证做法直接结晶被拒；已验证
#    做法结晶通过，来源链逐级完整（L1 证据 → L2 做法 → 技能），
#    每级晋升有主账本事件可回溯；
# ② 验证台（EvoSkill 留出集）：干跑产出与预设答案逐项对上才过、
#    不过如实记录；练习池与留出池不重叠（同源题组跨池登记被拒）；
#    同题独立两次产出一致才算稳；练习池干跑不算过门证据；
# ③ 试用期打回（追加决定十四）：试用中被纠正一次 → 旧版立即停用
#    打回；打补丁留痕（纠正来源＋补丁差异）生成新候选版本，重跑
#    完整验收（干跑＋两次一致＋重新试用＋人工点头）才可再转正；
# ④ 转正须人工点头（追加决定九）：无人工批准标记的转正一律拒，
#    本模块没有自动转正路径；
# ⑤ 生命周期：stale（14 天未用）/archived（再 30 天）按时钟推进、
#    归档不删除可恢复、pinned 豁免；淘汰留记录与原因不许无声消失；
# ⑥ 哈希账本：正文被篡改能检出（哈希只证一致不证对错）；
# ⑦ 遥测失败不打断技能调用本身；
# ⑧ 提案入口：外部只许提交结晶候选提案（绑来源/预期状态/幂等键），
#    幂等去重，技能文件不自带权限。
#
# 全部离线：执行器一律打桩（不调真 API、不花钱），时钟注入假时钟。

from datetime import datetime, timedelta, timezone

import pytest

from memory import ledger as _ledger
from memory.skill_crystallize import (
    QuestionBank,
    check_crystallization_gate,
    run_consistency,
    run_dry_run,
)
from memory.skills import SkillRegistry

_TZ = timezone(timedelta(hours=8))
_T0 = datetime(2026, 11, 25, 12, 0, tzinfo=_TZ)


# ---------------- 通用件 ----------------


class _Clock:
    def __init__(self):
        self.now = _T0

    def __call__(self):
        return self.now

    def advance(self, days):
        self.now = self.now + timedelta(days=days)


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def registry(tmp_path, clock):
    return SkillRegistry(tmp_path / "memory", now_fn=clock)


def _practice(**over):
    p = {
        "id": "practice-001",
        "kind": "practice",
        "name": "fund-limit-check",
        "title": "基金限额核查做法",
        "description": "需要核对某只基金当期限额时",
        "trigger": "用户问某只基金今天能买多少时",
        "steps": ["打开基金页", "核对限额公告", "按公告口径回答"],
        "pitfalls": ["别拿昨天的限额当今天的"],
        "boundary": "只适用于公募基金限额查询，不适用于赎回",
        "verification_method": "拿公告原文逐项对",
        "evidence": [{"id": "exp-1"}, {"id": "exp-2"}],
        "benefit_estimate": 0.7,
        "verified": True,
        "success_count": 3,
        "version": 2,
    }
    p.update(over)
    return p


def _bank():
    bank = QuestionBank()
    bank.add_task(
        {
            "task_id": "exam-1",
            "pool": "exam",
            "group": "limit-basics",
            "input": "查万家纳指今天限额",
            "expected": ["打开基金页", "核对限额公告"],
        }
    )
    bank.add_task(
        {
            "task_id": "prac-1",
            "pool": "practice",
            "group": "limit-drill",
            "input": "练习：查限额",
            "expected": ["打开基金页", "核对限额公告"],
        }
    )
    return bank


def _good_executor(content, task):
    return list(task["expected"])


def _wrong_executor(content, task):
    return ["全错的答案"]


def _pass_exam_gate(registry, skill_id, bank):
    """给当前版本补齐留出池干跑通过＋两次一致两项过门记录。"""
    skill = registry.get_skill(skill_id)
    content = (registry.skills_dir / skill_id / "SKILL.md").read_text(encoding="utf-8")
    assert skill is not None and content
    task = bank.get("exam-1")
    registry.record_validation(skill_id, run_dry_run(content, task, _good_executor))
    registry.record_validation(skill_id, run_consistency(content, task, _good_executor))


def _make_active(registry, bank, practice=None):
    meta = registry.crystallize(practice or _practice())
    skill_id = meta["skill_id"]
    _pass_exam_gate(registry, skill_id, bank)
    registry.begin_probation(skill_id)
    registry.complete_probation(skill_id)
    registry.promote(skill_id, approval={"approved_by": "BOOS", "note": "首批点头"})
    return skill_id


# ---------------- ① 结晶门 ----------------


def test_gate_rejects_raw_experience():
    raw = _practice(kind="experience")
    ok, reasons = check_crystallization_gate(raw)
    assert not ok and reasons


def test_gate_rejects_unverified_and_missing_requirements():
    for over in (
        {"verified": False},
        {"success_count": 0, "verification": None},
        {"evidence": []},
        {"benefit_estimate": 0},
        {"benefit_estimate": None},
        {"trigger": ""},
        {"steps": []},
        {"boundary": "", "applicability": ""},
    ):
        ok, reasons = check_crystallization_gate(_practice(**over))
        assert not ok, over
        assert reasons, over


def test_crystallize_raw_experience_raises(registry):
    with pytest.raises(ValueError):
        registry.crystallize(_practice(kind="trajectory"))
    assert registry.list_skills() == []


def test_crystallize_verified_practice_and_source_chain(registry):
    meta = registry.crystallize(_practice())
    assert meta["state"] == "candidate"
    assert meta["version"] == 1
    chain = meta["source_chain"]
    levels = [lvl["level"] for lvl in chain["levels"]]
    assert levels == ["L1", "L2", "skill"]
    assert chain["levels"][0]["refs"] == [{"id": "exp-1"}, {"id": "exp-2"}]
    assert chain["levels"][1]["practice_id"] == "practice-001"
    assert chain["levels"][2]["skill_id"] == meta["skill_id"]
    # 正文是标准 SKILL.md 形状：frontmatter + 五段
    content = (registry.skills_dir / meta["skill_id"] / "SKILL.md").read_text(encoding="utf-8")
    assert "name: fund-limit-check" in content
    assert "Use when" in content
    for section in ("## 触发条件", "## 步骤", "## 坑点", "## 适用边界", "## 验证方法"):
        assert section in content
    # 生成落主账本，事件可回溯且哈希对得上
    events, _info = _ledger.read_events(registry.ledger_path)
    skill_events = [e for e in events if e.get("type") == "skill"]
    assert len(skill_events) == 1
    assert skill_events[0]["action"] == "create"
    assert skill_events[0]["after_hash"] == meta["content_hash"]
    state = _ledger.resolve_state(events)
    folded = [e for e in state["skill_events"] if e["skill_id"] == meta["skill_id"]]
    assert len(folded) == 1 and folded[0]["action"] == "create"
    # 快照携带：压进快照再折叠，技能事件还在
    snap_path = registry.memory_dir / "snapshot.json"
    _ledger.write_snapshot(snap_path, state, state["seq_max"])
    payload = _ledger.read_snapshot(snap_path)
    state2 = _ledger.resolve_state([], snapshot=payload)
    assert len(state2["skill_events"]) == 1


def test_crystallize_is_idempotent(registry):
    first = registry.crystallize(_practice())
    second = registry.crystallize(_practice())
    assert first["skill_id"] == second["skill_id"]
    assert len(registry.list_skills()) == 1


# ---------------- ② 验证台 ----------------


def test_question_bank_pools_do_not_overlap():
    bank = _bank()
    assert bank.pools_overlap() is False
    # 同源改写题（同 group）想进另一个池：拒
    with pytest.raises(ValueError):
        bank.add_task(
            {
                "task_id": "prac-leak",
                "pool": "practice",
                "group": "limit-basics",
                "input": "改写版",
                "expected": ["打开基金页"],
            }
        )
    # 同 task_id 重复登记：拒
    with pytest.raises(ValueError):
        bank.add_task(
            {"task_id": "exam-1", "pool": "exam", "group": "other",
             "input": "x", "expected": []}
        )
    assert bank.pools_overlap() is False


def test_dry_run_pass_and_fail_recorded_truthfully(registry):
    meta = registry.crystallize(_practice())
    skill_id = meta["skill_id"]
    content = (registry.skills_dir / skill_id / "SKILL.md").read_text(encoding="utf-8")
    bank = _bank()
    exam = bank.get("exam-1")
    good = run_dry_run(content, exam, _good_executor)
    assert good["passed"] is True and good["mismatches"] == []
    bad = run_dry_run(content, exam, _wrong_executor)
    assert bad["passed"] is False and bad["mismatches"]
    # 记录落盘如实：过与不过都在 validation.jsonl 里
    registry.record_validation(skill_id, good)
    registry.record_validation(skill_id, bad)
    log = (registry.skills_dir / skill_id / "validation.jsonl").read_text(encoding="utf-8")
    assert '"passed": true' in log and '"passed": false' in log


def test_consistency_two_runs(registry):
    content = "技能正文"
    task = _bank().get("exam-1")
    stable = run_consistency(content, task, _good_executor)
    assert stable["consistent"] is True and stable["matches_expected"] is True

    calls = {"n": 0}

    def flaky(c, t):
        calls["n"] += 1
        return list(t["expected"]) if calls["n"] % 2 == 1 else ["另一个答案"]

    unstable = run_consistency(content, task, flaky)
    assert unstable["consistent"] is False


def test_practice_pool_does_not_open_gate(registry):
    meta = registry.crystallize(_practice())
    skill_id = meta["skill_id"]
    content = (registry.skills_dir / skill_id / "SKILL.md").read_text(encoding="utf-8")
    bank = _bank()
    prac = bank.get("prac-1")
    registry.record_validation(skill_id, run_dry_run(content, prac, _good_executor))
    registry.record_validation(skill_id, run_consistency(content, prac, _good_executor))
    ok, reasons = registry.validation_gate(skill_id)
    assert not ok and reasons
    with pytest.raises(ValueError):
        registry.begin_probation(skill_id)


def test_exam_failure_does_not_open_gate(registry):
    meta = registry.crystallize(_practice())
    skill_id = meta["skill_id"]
    content = (registry.skills_dir / skill_id / "SKILL.md").read_text(encoding="utf-8")
    exam = _bank().get("exam-1")
    registry.record_validation(skill_id, run_dry_run(content, exam, _wrong_executor))
    registry.record_validation(skill_id, run_consistency(content, exam, _good_executor))
    ok, _reasons = registry.validation_gate(skill_id)
    assert not ok


# ---------------- ③④ 试用、打回与转正 ----------------


def test_promote_requires_probation_and_human_approval(registry):
    bank = _bank()
    meta = registry.crystallize(_practice())
    skill_id = meta["skill_id"]
    _pass_exam_gate(registry, skill_id, bank)
    registry.begin_probation(skill_id)
    # 试用没走完不许转正
    with pytest.raises(ValueError):
        registry.promote(skill_id, approval={"approved_by": "BOOS"})
    registry.complete_probation(skill_id)
    # 没有人工点头标记一律拒（决定九）
    with pytest.raises(PermissionError):
        registry.promote(skill_id, approval=None)
    with pytest.raises(PermissionError):
        registry.promote(skill_id, approval={"note": "没写谁点头"})
    # 干跑通过本身不是转正证据：带点头才转正
    promoted = registry.promote(skill_id, approval={"approved_by": "BOOS"})
    assert promoted["state"] == "active"
    assert promoted["approval"]["approved_by"] == "BOOS"
    assert [s["skill_id"] for s in registry.retrievable_skills()] == [skill_id]


def test_probation_correction_sendback_patch_and_repromote(registry):
    bank = _bank()
    meta = registry.crystallize(_practice())
    skill_id = meta["skill_id"]
    _pass_exam_gate(registry, skill_id, bank)
    registry.begin_probation(skill_id)
    # 试用中被纠正一次：立即停用打回
    sent = registry.record_correction(
        skill_id, {"source": "BOOS", "text": "第二步说错了，要先看公告日期"}
    )
    assert sent["state"] == "sent_back"
    assert sent["counters"]["correction_count"] == 1
    assert sent["versions"][0]["outcome"] == "sent_back"
    assert registry.retrievable_skills() == []
    # 打回后不许直接再进试用（旧版已停用，得先修出新版本）
    with pytest.raises(ValueError):
        registry.begin_probation(skill_id)
    # 打补丁：留痕（纠正来源＋补丁差异），生成新候选版本
    patched = registry.apply_patch(
        skill_id,
        {"steps": ["打开基金页", "先核对公告日期", "核对限额公告", "按公告口径回答"]},
        correction={"source": "BOOS", "text": "第二步说错了，要先看公告日期"},
    )
    assert patched["state"] == "candidate"
    assert patched["version"] == 2
    assert patched["counters"]["patch_count"] == 1
    patch_info = patched["versions"][1]["patch"]
    assert patch_info["from_version"] == 1
    assert patch_info["correction"]["source"] == "BOOS"
    assert "先核对公告日期" in patch_info["diff"]
    # 新版本没重跑验收前不许进试用（旧版本的过门记录不许冒充新版本的）
    with pytest.raises(ValueError):
        registry.begin_probation(skill_id)
    # 重跑完整验收：干跑＋两次一致＋重新试用＋人工点头
    _pass_exam_gate(registry, skill_id, bank)
    registry.begin_probation(skill_id)
    registry.complete_probation(skill_id)
    repromoted = registry.promote(skill_id, approval={"approved_by": "BOOS"})
    assert repromoted["state"] == "active" and repromoted["version"] == 2
    # 全程账本可查：create → probation_start → send_back → patch → … → promote
    events, _info = _ledger.read_events(registry.ledger_path)
    actions = [e["action"] for e in events if e.get("type") == "skill"]
    assert actions[0] == "create"
    for expected_action in ("probation_start", "send_back", "patch", "promote"):
        assert expected_action in actions


def test_correction_outside_probation_only_counts(registry):
    bank = _bank()
    skill_id = _make_active(registry, bank)
    after = registry.record_correction(skill_id, {"source": "BOOS", "text": "一处措辞"})
    assert after["state"] == "active"
    assert after["counters"]["correction_count"] == 1


# ---------------- ⑤ 生命周期与淘汰 ----------------


def test_lifecycle_stale_archive_restore(registry, clock):
    bank = _bank()
    skill_id = _make_active(registry, bank)
    registry.record_use(skill_id, success=True)
    clock.advance(13)
    assert registry.advance_lifecycle() == []
    clock.advance(1)  # 未用满 14 天
    moved = registry.advance_lifecycle()
    assert moved == [{"skill_id": skill_id, "from": "active", "to": "stale"}]
    assert registry.get_skill(skill_id)["state"] == "stale"
    assert registry.retrievable_skills() == []
    clock.advance(29)
    assert registry.advance_lifecycle() == []
    clock.advance(1)  # stale 后再 30 天（共 44 天未用）
    moved = registry.advance_lifecycle()
    assert moved == [{"skill_id": skill_id, "from": "stale", "to": "archived"}]
    # 归档不删除：文件与记录都在，可恢复
    assert (registry.skills_dir / skill_id / "SKILL.md").exists()
    with pytest.raises(ValueError):
        registry.record_use(skill_id)
    restored = registry.restore(skill_id)
    assert restored["state"] == "active"


def test_pinned_skill_exempt_from_lifecycle(registry, clock):
    bank = _bank()
    skill_id = _make_active(registry, bank)
    registry.set_pinned(skill_id, True)
    clock.advance(60)
    assert registry.advance_lifecycle() == []
    assert registry.get_skill(skill_id)["state"] == "active"


def test_use_reactivates_stale(registry, clock):
    bank = _bank()
    skill_id = _make_active(registry, bank)
    registry.record_use(skill_id)
    clock.advance(20)
    registry.advance_lifecycle()
    assert registry.get_skill(skill_id)["state"] == "stale"
    used = registry.record_use(skill_id, success=True)
    assert used["state"] == "active"
    assert used["counters"]["use_count"] == 2


def test_eliminate_leaves_record(registry):
    bank = _bank()
    skill_id = _make_active(registry, bank)
    gone = registry.eliminate(skill_id, reason="被新技能取代", replaced_by="other-skill")
    assert gone["state"] == "eliminated"
    assert gone["eliminated"]["reason"] == "被新技能取代"
    assert gone["eliminated"]["replaced_by"] == "other-skill"
    # 不许无声消失：列表里还在、文件还在、账本有淘汰事件
    assert [s["skill_id"] for s in registry.list_skills()] == [skill_id]
    assert (registry.skills_dir / skill_id / "SKILL.md").exists()
    events, _info = _ledger.read_events(registry.ledger_path)
    assert any(
        e.get("type") == "skill" and e.get("action") == "eliminate" for e in events
    )


# ---------------- ⑥ 哈希账本验篡改 ----------------


def test_integrity_detects_tampering(registry):
    meta = registry.crystallize(_practice())
    skill_id = meta["skill_id"]
    assert registry.verify_integrity(skill_id) is True
    path = registry.skills_dir / skill_id / "SKILL.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n偷偷加的一行\n", encoding="utf-8")
    assert registry.verify_integrity(skill_id) is False


# ---------------- ⑦ 遥测失败不打断 ----------------


def test_telemetry_failure_does_not_break_use(registry, tmp_path):
    bank = _bank()
    skill_id = _make_active(registry, bank)
    # 把 usage.jsonl 占成目录，遥测写必失败——record_use 仍要成功
    usage_path = registry.skills_dir / skill_id / "usage.jsonl"
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    if usage_path.exists():
        usage_path.unlink()
    usage_path.mkdir()
    used = registry.record_use(skill_id, success=True)
    assert used["counters"]["use_count"] == 1
    assert used["counters"]["success_count"] == 1


# ---------------- ⑧ 提案入口 ----------------


def test_proposal_flow_and_idempotency(registry):
    proposal = {
        "idempotency_key": "curator-20261125-001",
        "practice": _practice(),
        "expected_state": "candidate",
    }
    first = registry.submit_proposal(proposal, proposer="curator")
    again = registry.submit_proposal(proposal, proposer="curator")
    assert first["proposal_id"] == again["proposal_id"]
    result = registry.accept_proposal(first["proposal_id"])
    assert result["status"] == "accepted"
    assert result["skill"]["state"] == "candidate"
    # 重复处理已接受提案：幂等，不再生成第二个技能
    result2 = registry.accept_proposal(first["proposal_id"])
    assert result2["status"] == "accepted"
    assert len(registry.list_skills()) == 1


def test_proposal_shape_rejections(registry):
    with pytest.raises(ValueError):
        registry.submit_proposal({"practice": _practice()})  # 没幂等键
    with pytest.raises(ValueError):
        registry.submit_proposal(
            {"idempotency_key": "k1", "practice": _practice(), "expected_state": "active"}
        )
    with pytest.raises(ValueError):
        registry.submit_proposal(
            {"idempotency_key": "k2", "practice": _practice(), "permissions": ["execute"]}
        )


def test_proposal_rejected_at_gate_leaves_record(registry):
    proposal = {
        "idempotency_key": "curator-raw-001",
        "practice": _practice(kind="experience"),
        "expected_state": "candidate",
    }
    record = registry.submit_proposal(proposal, proposer="curator")
    result = registry.accept_proposal(record["proposal_id"])
    assert result["status"] == "rejected" and result["reasons"]
    assert registry.list_skills() == []
    # 拒收留痕：再问一次还是 rejected，不会无声消失
    result2 = registry.accept_proposal(record["proposal_id"])
    assert result2["status"] == "rejected"
