"""G4 影子观察跑批（2026-11-27，立项 G4）。

把宿主活行经 G1 适配层组装成她侧批次协议（Proposal＋离线
快照），喂她侧 BatchJournal 落影子批次：只观察、只落判定，
execution_allowed 恒 False，不提交受控入口、不动宿主账本。

设计口径：
  - 批次人口＝快照里的全部活行（按 live 标记筛，按 memory_id
    排序保证同快照同批次可复现）；挑不挑得动由她侧 review
    判定决定，本模块不预筛判定、不改判。人口筛活行与快照绑
    定是贵妃互查 F3/F4（2026-11-27）的修复口径：快照 rows
    本含死行，旧实现不筛 live、且落盘的 snapshot_id 收调用
    方任意值与实际组装所用快照对不上号，都已改掉。
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


def _resolve_snapshot(store, snapshot_id):
    """取本批要用的那份快照（互查 F4）。

    给了 id：按 id 取回宿主已登记快照——批次内容必须出自这
    个 id 名下的那份快照，取不回如实报错，不许另取一份顶替
    冒充。没给 id：现取一份新快照，其真实 id 即本批绑定 id。
    """
    if snapshot_id is None:
        return store.take_stable_snapshot()
    try:
        return store.get_stable_snapshot(snapshot_id)
    except KeyError:
        raise ValueError(
            f"snapshot not registered: {snapshot_id}") from None


def _items_from_snapshot(store, snap, *, proposal_cls, memory_ids=None):
    """从一份已定快照组装 (Proposal, current)：人口只算活行。

    快照 rows 含死行（已顶替/删除/作废），人口按 live 标记
    筛（互查 F3）。memory_ids 显式点名时（顺序按账本折叠排
    序，不按传入顺序），点到不存在或已非活行的直接报错，
    不静默接受、也不静默跳过。
    """
    rows = snap["rows"]
    if memory_ids is None:
        ids = sorted(
            mid for mid, row in rows.items() if row.get("live"))
    else:
        ids = sorted(set(memory_ids))
        bad = [mid for mid in ids
               if mid not in rows or not rows[mid].get("live")]
        if bad:
            raise ValueError(f"memory not live in snapshot: {bad}")
    items = []
    for mid in ids:
        row = dict(rows[mid])
        row["memory_id"] = mid
        proposal = proposal_cls(**_adapter.build_proposal_fields(row))
        current = _adapter.build_current(store, row)
        items.append((proposal, current))
    return items


def build_batch_items(store, *, proposal_cls, memory_ids=None):
    """组装一批 (Proposal, current)：人口＝全部活行。

    现取一份新快照组装。要把组装绑到某份已登记快照上，走
    run_shadow_batch 并传 snapshot_id。
    """
    if proposal_cls is None:
        raise SegmentNotWired("proposal class not injected")
    snap = store.take_stable_snapshot()
    return _items_from_snapshot(
        store, snap, proposal_cls=proposal_cls, memory_ids=memory_ids)


def run_shadow_batch(store, journal, *, batch_id, snapshot_id=None,
                     proposal_cls, memory_ids=None):
    """落一期影子批次，返回她侧 payload（execution_allowed=False）。

    snapshot_id 给定：必须是宿主已登记快照的 id，批次 items
    就从那份快照组装、落盘 id 与之一字不差；取不回直接拒绝
    （互查 F4：不许落一个跟实际内容对不上号的 id）。没给：
    现取一份新快照，落盘其真实 id。
    """
    if journal is None:
        raise SegmentNotWired("batch journal not injected")
    if proposal_cls is None:
        raise SegmentNotWired("proposal class not injected")
    snap = _resolve_snapshot(store, snapshot_id)
    items = _items_from_snapshot(
        store, snap, proposal_cls=proposal_cls, memory_ids=memory_ids)
    return journal.collect(
        batch_id, snap["snapshot_id"], _adapter.POLICY_VERSION, items,
        protected=lambda meta: _adapter.host_protected(store, meta),
        classify=_adapter.host_classify,
    )


def compare_batch(journal, batch_id, labels):
    """关四差异复核：独立人工标签 vs 批次判定，只报差异。"""
    if journal is None:
        raise SegmentNotWired("batch journal not injected")
    return journal.compare(batch_id, labels)
