"""整理员执行接线（立项 G2，2026-11-27）。

把贵妃侧 Curator 的 execute 从「无条件 PermissionError」改为按
已就绪的宿主项逐段接线（接线材料：G1 适配层
memory/curator_adapter.py）。纪律（立项 G2 原义）：

  - 提案只经 H1 统一受控入口提交，执行只认 H1 回执——本层
    不直接改任何记忆状态，不绕入口；
  - 已接线段只有一条：普通核心成员的降级（demote）候选，
    且必须先过她侧 assess 与 review_proposal 两道评估（都得
    shadow_*_only 才许组装提案），任一挂起即停在该段、不提交；
  - 未就绪段继续拒绝：demote 以外的动作（改保护档/改抽屉/
    标记休眠/删除等）一律 PermissionError，与她侧原口径一致；
  - 执行开关（store.controlled_entry_execution_enabled）归
    宿主/BOOS，本层只读不拨：关时提交只出影子回执，本层如实
    报 shadow，绝不冒充已执行；开时会话凭据（H2）由调用方
    供给，本层不签发、不伪造——没凭据 H1 自会拒，本层如实
    转 refused；
  - 幂等键必须由调用方给：整理员执行必须可重放安全，缺键
    在提交前就拒（ValueError），不许本层现编随机键。
"""

from memory import curator_adapter as _adapter

# 她侧评估模块不在宿主仓内（原件在贵妃 curator-system）；本层
# 需要她的判定函数时由调用方注入（assess_fn / review_fn），
# 缺注入即该段未接线、在评估段拒绝——不许宿主侧复刻她的判定
# 逻辑另立口径（立项 G1/G2 共通纪律）。


class SegmentNotWired(PermissionError):
    """该执行段尚未接线：与她侧原 execute 的拒绝同义。"""


def execute_demote(store, row, *, judge_status, instruction_source=None,
                   idempotency_key=None, session=None, reason="",
                   assess_fn=None, review_fn=None):
    """一条降级候选的接线执行：评估两段 → H1 提交 → 只认回执。

    row 是 H4 快照行（take_stable_snapshot 的 rows 值，可带
    memory_id 键）。返回结果 dict：
      {outcome, stage, assess, review, receipt, summary}
      outcome ∈ held / shadow / executed / refused / duplicate
      held 时 stage 指出停在哪段（assess / review），后段字段
      为 None 且保证没有提交过 H1。
    """
    if not idempotency_key:
        raise ValueError("idempotency_key required: 整理员执行必须由 "
                         "调用方给幂等键，不许现编")
    if assess_fn is None or review_fn is None:
        raise SegmentNotWired(
            "execution disabled: 评估判定未注入（assess/review），"
            "该段未接线")
    candidate = _adapter.build_candidate_fields(
        row, judge_status=judge_status)
    verdict = assess_fn(candidate)
    if verdict != "shadow_candidate_only":
        return {"outcome": "held", "stage": "assess", "assess": verdict,
                "review": None, "receipt": None, "summary": None}
    proposal_fields = _adapter.build_proposal_fields(row)
    current = _adapter.build_current(store, row)
    review = review_fn(proposal_fields, current)
    if review != "shadow_proposal_only":
        return {"outcome": "held", "stage": "review", "assess": verdict,
                "review": review, "receipt": None, "summary": None}
    proposal = _adapter.h1_proposal(
        proposal_fields, action="demote", reason=reason,
        instruction_source=instruction_source,
        idempotency_key=idempotency_key, session=session)
    receipt = store.submit_controlled_proposal(proposal)
    summary = _adapter.receipt_summary(receipt)
    if summary["duplicate"]:
        # 同键同指纹重提：H1 不重复执行，回执 status=duplicate；
        # 如实报 duplicate，不许冒充 refused 或二次执行。
        outcome = "duplicate"
    elif summary["executed"]:
        outcome = "executed"
    elif summary["shadow"]:
        outcome = "shadow"
    else:
        outcome = "refused"
    return {"outcome": outcome, "stage": "submitted", "assess": verdict,
            "review": review, "receipt": receipt, "summary": summary}


def execute(store, action, *args, **kwargs):
    """与她侧 curator.execute 同名的接线总口：只认已接线段。

    demote 走 execute_demote；其余动作该段未接线，继续
    PermissionError（SegmentNotWired）——包括删除：整理员
    永远无删除权限，删除只走 BOOS 批准路，不在本口里。
    """
    if action == "demote":
        return execute_demote(store, *args, **kwargs)
    raise SegmentNotWired(
        "execution disabled: action %r 该段未接线（G2 只接 demote）"
        % (action,))
