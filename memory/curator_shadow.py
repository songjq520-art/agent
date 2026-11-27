"""G4 影子观察跑批（2026-11-27，立项 G4）。

把宿主活行经 G1 适配层组装成她侧批次协议（Proposal＋离线
快照），喂她侧 BatchJournal 落影子批次：只观察、只落判定，
execution_allowed 恒 False，不提交受控入口、不动宿主账本。

设计口径：
  - 批次人口＝宿主账本折叠里的全部活行（按 memory_id 排序
    保证同快照同批次可复现）；挑不挑得动由她侧 review 判定
    决定，本模块不预筛、不改判。
  - 她侧协议类（Proposal / BatchJournal）由调用方注入，
    与 G1/G2 同规：协议所有权在她侧代码，本模块只组装。
  - 观察窗规则（G3-U4 定版）：policy_version 一变，旧批次
    证据对不上、新批次必须另开 batch_id 从头观察；本模块
    每批都带适配层 POLICY_VERSION，混版本由她侧 collect
    当场拒收。
  - 差异复核（关四）走她侧 compare：标签来自独立人工，
    compare 只报差异、不签发放行。
"""

from __future__ import annotations

from memory import curator_adapter as _adapter


class SegmentNotWired(PermissionError):
    """她侧协议件未注入时拒跑，不自造协议。"""


def build_batch_items(store, *, proposal_cls, memory_ids=None):
    """组装一批 (Proposal, current)：人口＝全部活行。

    memory_ids 给定时只取其中活行（顺序按账本折叠排序，
    不按传入顺序）。行不存在或已非活行直接报错，不静默跳过。
    """
    if proposal_cls is None:
        raise SegmentNotWired("proposal class not injected")
    snap = store.take_stable_snapshot()
    rows = snap["rows"]
    if memory_ids is None:
        ids = sorted(rows)
    else:
        ids = sorted(set(memory_ids))
        missing = [mid for mid in ids if mid not in rows]
        if missing:
            raise ValueError(f"memory not live in snapshot: {missing}")
    items = []
    for mid in ids:
        row = dict(rows[mid])
        row["memory_id"] = mid
        proposal = proposal_cls(**_adapter.build_proposal_fields(row))
        current = _adapter.build_current(store, row)
        items.append((proposal, current))
    return items


def run_shadow_batch(store, journal, *, batch_id, snapshot_id,
                     proposal_cls, memory_ids=None):
    """落一期影子批次，返回她侧 payload（execution_allowed=False）。"""
    if journal is None:
        raise SegmentNotWired("batch journal not injected")
    items = build_batch_items(store, proposal_cls=proposal_cls,
                              memory_ids=memory_ids)
    return journal.collect(
        batch_id, snapshot_id, _adapter.POLICY_VERSION, items,
        protected=lambda meta: _adapter.host_protected(store, meta),
        classify=_adapter.host_classify,
    )


def compare_batch(journal, batch_id, labels):
    """关四差异复核：独立人工标签 vs 批次判定，只报差异。"""
    if journal is None:
        raise SegmentNotWired("batch journal not injected")
    return journal.compare(batch_id, labels)
