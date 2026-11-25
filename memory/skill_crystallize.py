# memory/skill_crystallize.py（四件套第 4 块：技能自动生成——结晶门＋提案校验＋验证台）
#
# 这一层只做三件事，全是纯判定/纯记录，不读库、不联网、不调模型：
#
# 1. 结晶门（借 MemOS/MSCE 机制，只借机制）：技能只许从「已验证的
#    做法」（L2）结晶，原始经历/对话（L1 轨迹）直接结晶一律拒。
#    准入三要件（缺一不可）：
#      (a) 留有 L1 证据链——evidence 非空，每条能回指一条经历/事件；
#      (b) 估计收益为正——benefit_estimate 是数字且 > 0；
#      (c) 触发条件/步骤/适用边界三者自洽——三样都写明了。
#    另外要求做法本身已被验证：verified 为真，且有成功证据
#    （success_count >= 1，或 verification 记录 passed）。
#    门只判「许不许生成候选」，生成之后还要过验证台＋试用＋人工
#    点头（见 skills.py 的状态机），两道门不互相替代。
#
# 2. 提案校验：外部（如整理员）只能提交结晶候选提案——绑来源
#    做法、预期状态（只许 candidate）、幂等键。提案形状在这儿验，
#    收不收由 skills.SkillRegistry 按幂等键去重后落账。
#
# 3. 验证台（借 EvoSkill 留出集机制）：干跑题分练习池（practice）
#    与留出池（exam）两池——同一来源的改写题共享 group，一个
#    group 只许属于一个池，跨池登记直接拒；留出题不进生成/练习
#    流程（生成侧的输入只有做法本身，本模块没有任何把考题喂给
#    生成的路）。干跑 = 拿预设任务在沙箱里跑，产出与预设答案逐项
#    对上才算过；同题独立跑两次、两次产出逐项一致才算稳。干跑
#    结果只算练习/验证记录，不算转正证据（转正证据是试用无错＋
#    人工点头，见契约技能验收三条与追加决定九）。
#
# 执行器（executor）是注入进来的可调用对象：executor(技能正文,
# 题目) -> 产出。生产侧接判官/模型时在外面包一层传进来，本模块
# 不认识任何模型——测试全部用打桩执行器，不真调 API、不花钱。

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# 结晶门
# ---------------------------------------------------------------------------

#: 做法的层级标记：只有 L2（已归纳的做法）能进结晶门。
PRACTICE_KIND = "practice"

#: 原始轨迹的层级标记（出现这些值一律拒——不许从原始轨迹直提技能）。
RAW_KINDS = ("experience", "trajectory", "raw", "conversation", "L1")


def _nonempty(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return bool(value)


def check_crystallization_gate(practice):
    """结晶门判定（纯函数）。返回 (ok, reasons)。

    practice 是做法的字典形状（由记忆层/整理员给出）：
      id, kind, name/title, trigger, steps, boundary/applicability,
      evidence: [L1 证据引用...], benefit_estimate,
      verified, success_count / verification。
    reasons 是拒因列表（中文，给人看的），ok=True 时为空。
    """
    reasons = []
    if not isinstance(practice, dict):
        return False, ["做法不是字典形状，无法过结晶门。"]
    kind = practice.get("kind")
    if kind in RAW_KINDS or kind == "L1":
        reasons.append(
            "来源是原始经历/轨迹（L1），不许直接结晶成技能——"
            "先归纳成已验证的做法（L2）再来。"
        )
    elif kind != PRACTICE_KIND:
        reasons.append(f"来源 kind={kind!r} 不是已归纳的做法（practice/L2）。")
    if not _nonempty(practice.get("id")):
        reasons.append("做法没有 id，来源链断头。")
    # (a) L1 证据链
    evidence = practice.get("evidence")
    if not isinstance(evidence, (list, tuple)) or len(evidence) == 0:
        reasons.append("没有 L1 证据链（evidence 为空）——结晶门要件 (a) 不过。")
    else:
        for idx, ref in enumerate(evidence):
            if isinstance(ref, dict):
                if not _nonempty(ref.get("id") or ref.get("event_id") or ref.get("ref")):
                    reasons.append(f"证据链第 {idx + 1} 条没有可回指的 id。")
            elif not _nonempty(ref):
                reasons.append(f"证据链第 {idx + 1} 条是空引用。")
    # (b) 估计收益为正
    benefit = practice.get("benefit_estimate")
    if not isinstance(benefit, (int, float)) or isinstance(benefit, bool) or benefit <= 0:
        reasons.append("估计收益不是正数（benefit_estimate）——结晶门要件 (b) 不过。")
    # (c) 触发/步骤/边界三者自洽（都写明）
    if not _nonempty(practice.get("trigger")):
        reasons.append("触发条件没写明——结晶门要件 (c) 不过。")
    if not _nonempty(practice.get("steps")):
        reasons.append("步骤没写明——结晶门要件 (c) 不过。")
    if not _nonempty(practice.get("boundary") or practice.get("applicability")):
        reasons.append("适用边界没写明——结晶门要件 (c) 不过。")
    # 已验证：verified 为真 + 有成功证据
    verified = practice.get("verified") is True
    success_count = practice.get("success_count")
    has_success = (
        isinstance(success_count, (int, float))
        and not isinstance(success_count, bool)
        and success_count >= 1
    )
    verification = practice.get("verification")
    if isinstance(verification, dict) and verification.get("passed") is True:
        has_success = True
    if not verified:
        reasons.append("做法未经验证（verified 不为真），不许结晶。")
    elif not has_success:
        reasons.append("做法没有成功证据（success_count>=1 或 verification.passed），不许结晶。")
    return (len(reasons) == 0), reasons


def build_source_chain(practice, skill_id):
    """把做法的来源链整理成技能要带的那份（逐级可回溯）。

    形状：{"levels": [{"level": "L1", "refs": [...]},
                      {"level": "L2", "practice_id": ..., "practice_version": ...},
                      {"level": "skill", "skill_id": ...}],
           "evidence": [原始证据引用...]}
    """
    evidence = list(practice.get("evidence") or [])
    return {
        "levels": [
            {"level": "L1", "refs": evidence},
            {
                "level": "L2",
                "practice_id": practice.get("id"),
                "practice_version": practice.get("version"),
            },
            {"level": "skill", "skill_id": skill_id},
        ],
        "evidence": evidence,
        "practice_id": practice.get("id"),
    }


def practice_fingerprint(practice):
    """做法内容的指纹（同一做法重复结晶/重复提案靠它认）。"""
    canonical = json.dumps(practice, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 提案校验（整理员等外部入口的唯一通道形状）
# ---------------------------------------------------------------------------

#: 提案只许期望生成候选——外部不许直接指定试用/转正等后续状态。
PROPOSAL_EXPECTED_STATES = ("candidate",)


def check_proposal(proposal):
    """提案形状校验（纯函数）。返回 (ok, reasons)。

    提案形状：{idempotency_key, practice: {...}, expected_state,
    proposer}。技能模块只从提案收「结晶候选」这一件事：绑了来源
    做法、带幂等键、预期状态只许 candidate。技能文件不自带权限——
    提案里出现权限/执行类字段直接拒。
    """
    reasons = []
    if not isinstance(proposal, dict):
        return False, ["提案不是字典形状。"]
    if not _nonempty(proposal.get("idempotency_key")):
        reasons.append("提案没有幂等键（idempotency_key），不收——重复提交会造出重复技能。")
    expected = proposal.get("expected_state") or "candidate"
    if expected not in PROPOSAL_EXPECTED_STATES:
        reasons.append(
            f"提案预期状态 {expected!r} 不许——外部提案只许期望 candidate，"
            "试用/转正走技能自己的状态机和人工点头。"
        )
    if not isinstance(proposal.get("practice"), dict):
        reasons.append("提案没有绑来源做法（practice）。")
    for bad_key in ("permissions", "permission", "execute", "executor", "grants"):
        if bad_key in proposal:
            reasons.append(f"提案带了权限/执行字段 {bad_key!r}——技能文件不自带权限，不收。")
    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# 验证台：题库（双池）＋干跑＋两次一致
# ---------------------------------------------------------------------------

POOL_PRACTICE = "practice"
POOL_EXAM = "exam"
POOLS = (POOL_PRACTICE, POOL_EXAM)


def _normalize_items(output):
    """把执行器产出规整成可逐项比的列表。

    str -> [str]；list/tuple 原样逐项转成规范 JSON 串；dict -> 按键
    排序的 "k=v" 串列表；其余 -> [规范串]。逐项比的是断言与产物，
    不比推理逐字（与第七封信答贵妃的口径一致）。
    """
    def canon(item):
        if isinstance(item, str):
            return item
        return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)

    if output is None:
        return []
    if isinstance(output, str):
        return [output]
    if isinstance(output, dict):
        return [f"{k}={canon(output[k])}" for k in sorted(output)]
    if isinstance(output, (list, tuple)):
        return [canon(item) for item in output]
    return [canon(output)]


def compare_items(expected, output):
    """逐项比对：返回 (passed, mismatches)。

    mismatches 是 [{"index", "expected", "actual"}...]，长度对不上时
    缺/多的项以 None 占位。passed 只在项数与逐项都对上时为真。
    """
    exp = _normalize_items(expected)
    act = _normalize_items(output)
    mismatches = []
    for idx in range(max(len(exp), len(act))):
        e = exp[idx] if idx < len(exp) else None
        a = act[idx] if idx < len(act) else None
        if e != a:
            mismatches.append({"index": idx, "expected": e, "actual": a})
    return (len(mismatches) == 0), mismatches


class QuestionBank:
    """干跑题库：练习池与留出池严格分开。

    分池规矩：
      - 一道题只属于一个池（task_id 全库唯一）；
      - 同源改写题共享 group，一个 group 只许登记在一个池里——
        想把同组题放进另一个池直接拒（防留出题被同源题泄漏）；
      - 留出池的题只供验证台在验收时取用，本类不提供任何把考题
        交给生成侧的方法。
    """

    def __init__(self):
        self._tasks = {}
        self._group_pool = {}

    def add_task(self, task):
        """登记一道题。task: {task_id, pool, group, input, expected}。"""
        if not isinstance(task, dict):
            raise ValueError("题目不是字典形状。")
        task_id = task.get("task_id")
        pool = task.get("pool")
        group = task.get("group") or task_id
        if not _nonempty(task_id):
            raise ValueError("题目没有 task_id。")
        if pool not in POOLS:
            raise ValueError(f"题目 pool={pool!r} 不对，只许 practice / exam。")
        if task_id in self._tasks:
            raise ValueError(f"task_id {task_id!r} 已登记过。")
        bound_pool = self._group_pool.get(group)
        if bound_pool is not None and bound_pool != pool:
            raise ValueError(
                f"题组 {group!r} 已属于 {bound_pool} 池，不许再进 {pool} 池"
                "——同源改写题必须同组同池，防留出集泄漏。"
            )
        self._group_pool[group] = pool
        self._tasks[task_id] = dict(task)
        return dict(task)

    def get(self, task_id):
        task = self._tasks.get(task_id)
        return dict(task) if task else None

    def tasks(self, pool=None):
        out = []
        for task in self._tasks.values():
            if pool is None or task.get("pool") == pool:
                out.append(dict(task))
        return out

    def pools_overlap(self):
        """两池是否有交叠（task_id 或 group 层面）。正常应恒为 False。"""
        practice_ids = {t["task_id"] for t in self.tasks(POOL_PRACTICE)}
        exam_ids = {t["task_id"] for t in self.tasks(POOL_EXAM)}
        if practice_ids & exam_ids:
            return True
        practice_groups = {t.get("group") or t["task_id"] for t in self.tasks(POOL_PRACTICE)}
        exam_groups = {t.get("group") or t["task_id"] for t in self.tasks(POOL_EXAM)}
        return bool(practice_groups & exam_groups)


def run_dry_run(skill_content, task, executor):
    """干跑一题（纯执行＋记录，不落任何状态）。

    executor(技能正文, 题目) -> 产出。返回记录：
    {task_id, pool, group, passed, mismatches, run_kind: "dry_run"}。
    执行器抛错记为不过（passed=False, error 带原因），不往外炸——
    干跑失败是验收结论，不是程序事故。
    """
    record = {
        "task_id": task.get("task_id"),
        "pool": task.get("pool"),
        "group": task.get("group") or task.get("task_id"),
        "run_kind": "dry_run",
    }
    try:
        output = executor(skill_content, task)
    except Exception as exc:  # 干跑执行失败 = 不过，如实记录
        record.update({"passed": False, "mismatches": [], "error": f"{type(exc).__name__}: {exc}"})
        return record
    passed, mismatches = compare_items(task.get("expected"), output)
    record.update({"passed": passed, "mismatches": mismatches})
    return record


def run_consistency(skill_content, task, executor):
    """同题独立跑两次，比两次产出是否逐项一致。

    返回 {task_id, pool, group, consistent, matches_expected,
    mismatches, run_kind: "consistency"}。consistent 只看两次产出
    互比；matches_expected 是附带信息（两次都跑偏但一致时
    consistent=True、matches_expected=False，进试用门时干跑过
    不过另有记录管，不在这里混判）。
    """
    record = {
        "task_id": task.get("task_id"),
        "pool": task.get("pool"),
        "group": task.get("group") or task.get("task_id"),
        "run_kind": "consistency",
    }
    try:
        first = executor(skill_content, task)
        second = executor(skill_content, task)
    except Exception as exc:
        record.update(
            {"consistent": False, "matches_expected": False,
             "mismatches": [], "error": f"{type(exc).__name__}: {exc}"}
        )
        return record
    consistent, mismatches = compare_items(first, second)
    matches_expected, _ = compare_items(task.get("expected"), first)
    record.update(
        {"consistent": consistent, "matches_expected": matches_expected,
         "mismatches": mismatches}
    )
    return record
