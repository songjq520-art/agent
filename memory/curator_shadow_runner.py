"""G4 观察窗生产跑批器（2026-11-27，观察窗生产接线）。

G4 沙盒交付的是影子机制（memory/curator_shadow.py）＋四关证
据；要在真服务上把两周日历观察窗真正跑起来，还缺生产这一
层——本模块补齐三件：

  - 跑批：拿活 store、用生产协议件（memory/curator_protocol，
    她侧协议的逐字移植）按「一天一批」调 run_shadow_batch，
    批次落进固定隔离目录 shadow_journal/（她侧 BatchJournal
    的 sqlite 格式，她侧 collect/compare 原样能读）；
  - 批次号：shadow-YYYYMMDD，按北京时间日期。同日重跑幂等
    （台账里已有当天批次号就不再跑；就算绕过台账直调，她侧
    collect 对同 id 同证据也只回既有 payload、同 id 异证据
    当场拒收）；
  - 窗口台账：shadow_window_state.json 与维护状态同目录，
    记窗口起始日（北京时间）、钉住的 POLICY_VERSION、已跑
    批次清单（批次号/日期/行数/提议数/挂起数）。

观察窗规矩（G3-U4 定版）：跑批前先校验当前 POLICY_VERSION
与窗口钉住值——不一致＝口径已变、旧窗作废：台账标记 voided
并停跑，只记日志，绝不混版本接着跑。新口径下要不要重开新
窗是另行拍板的事，本模块不自作主张重开。

红线：只观察——execution_allowed 恒 False（她侧 payload 写
死）、不提交受控入口、不动宿主账本；批次日志与台账只记 id、
版本戳与判定计数，不落记忆原文。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory import curator_adapter as _adapter
from memory import curator_shadow as _shadow
from memory.curator_protocol import BatchJournal, Proposal

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo

    SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # 没有 tzdata 时用固定 +08:00（上海无夏令时，等价）
    SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

JOURNAL_DIR_NAME = "shadow_journal"
WINDOW_STATE_NAME = "shadow_window_state.json"
# 观察窗长度：14 个日历日（G3-U4／决定十）。起始日算第 1 天。
WINDOW_DAYS = 14


def _to_shanghai(now):
    """now 归一成上海时区 datetime：可传 None/epoch/datetime。"""
    if now is None:
        return datetime.now(SHANGHAI_TZ)
    if isinstance(now, (int, float)):
        return datetime.fromtimestamp(float(now), SHANGHAI_TZ)
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.replace(tzinfo=SHANGHAI_TZ)
        return now.astimezone(SHANGHAI_TZ)
    return datetime.now(SHANGHAI_TZ)


def beijing_date(now=None):
    """now 对应的北京时间日期（date 对象）。"""
    return _to_shanghai(now).date()


def batch_id_for(day):
    """某北京时间日期的批次号：shadow-YYYYMMDD。"""
    return "shadow-" + day.strftime("%Y%m%d")


def journal_dir(base_dir):
    """批次日志隔离目录：<记忆目录>/shadow_journal/。"""
    return Path(base_dir) / JOURNAL_DIR_NAME


def window_state_path(base_dir):
    return Path(base_dir) / WINDOW_STATE_NAME


def _read_window_state(base_dir):
    """读窗口台账：文件不存在返回 None；坏了抛 ValueError。

    台账读不出＝窗口状态未知，朝安全侧倒：抛错让调度层记
    日志停跑，绝不把未知洗成「还没开窗」从头再来。
    """
    path = window_state_path(base_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("shadow window state unreadable: %r" % (exc,)) from exc
    if not isinstance(data, dict):
        raise ValueError("shadow window state is not an object")
    return data


def _save_window_state(base_dir, state):
    path = window_state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def load_window_state(base_dir):
    """给状态查询用的读口：没有台账返回 None，坏了抛 ValueError。"""
    return _read_window_state(base_dir)


def window_status(base_dir, now=None):
    """观察窗现状摘要（只读）：开没开窗、钉的哪个口径、跑了
    几批、第几天、作没作废。台账不存在＝还没开窗。"""
    state = _read_window_state(base_dir)
    if state is None:
        return {"opened": False, "voided": False, "batches": 0}
    start = state.get("window_start")
    days_elapsed = None
    if isinstance(start, str):
        try:
            start_day = datetime.strptime(start, "%Y-%m-%d").date()
            days_elapsed = (beijing_date(now) - start_day).days + 1
        except ValueError:
            days_elapsed = None
    voided = bool(state.get("voided"))
    return {
        "opened": True,
        "window_start": start,
        "pinned_policy_version": state.get("pinned_policy_version"),
        "voided": voided,
        "batches": len(state.get("batches") or []),
        "days_elapsed": days_elapsed,
        "window_days": WINDOW_DAYS,
        "window_complete": bool(
            not voided and days_elapsed is not None
            and days_elapsed >= WINDOW_DAYS),
    }


def run_daily_shadow_batch(store, base_dir, now=None):
    """跑当天这一批影子观察（或给出不跑的理由），返回摘要 dict。

    status ∈ ran / already_ran / voided：
      - ran：真跑了，带 batch_id、date、rows/proposals/holds；
      - already_ran：当天批次台账里已有，幂等跳过；
      - voided：口径已变、窗口作废，停跑（台账已标记）。
    异常（台账坏、组装/落盘失败）如实抛给调用方（调度层会吞
    掉记日志、下个间隔再试），不在这里洗成某个 status。
    """
    base_dir = Path(base_dir)
    day = beijing_date(now)
    batch_id = batch_id_for(day)
    current_policy = _adapter.POLICY_VERSION
    state = _read_window_state(base_dir)
    if state is None:
        # 开窗：先把台账落盘再跑批——跑批中途崩了，窗口起点
        # 与钉住的口径也不丢，下次接着同一窗继续。
        state = {
            "window_start": day.isoformat(),
            "pinned_policy_version": current_policy,
            "voided": False,
            "batches": [],
        }
        _save_window_state(base_dir, state)
        logger.info(
            "影子观察窗开启：起始日 %s（北京时间），钉住口径 %s。",
            state["window_start"], current_policy)
    if state.get("pinned_policy_version") != current_policy:
        # G3-U4：口径一变旧窗作废。只标记、只记日志、停跑，
        # 绝不拿新口径混进旧窗继续攒批次。
        if not state.get("voided"):
            state["voided"] = True
            state["voided_at"] = _to_shanghai(now).isoformat()
            state["voided_reason"] = "policy_version_changed"
            _save_window_state(base_dir, state)
        logger.warning(
            "影子观察窗作废：口径已由 %s 变为 %s，停跑待另行拍板。",
            state.get("pinned_policy_version"), current_policy)
        return {
            "status": "voided",
            "batch_id": batch_id,
            "date": day.isoformat(),
            "pinned_policy_version": state.get("pinned_policy_version"),
            "policy_version": current_policy,
        }
    if state.get("voided"):
        return {
            "status": "voided",
            "batch_id": batch_id,
            "date": day.isoformat(),
            "pinned_policy_version": state.get("pinned_policy_version"),
            "policy_version": current_policy,
        }
    batches = state.setdefault("batches", [])
    if any(isinstance(b, dict) and b.get("batch_id") == batch_id
           for b in batches):
        return {
            "status": "already_ran",
            "batch_id": batch_id,
            "date": day.isoformat(),
            "pinned_policy_version": current_policy,
        }
    journal = BatchJournal(journal_dir(base_dir))
    payload = _shadow.run_shadow_batch(
        store, journal, batch_id=batch_id, proposal_cls=Proposal)
    rows = payload.get("rows") or []
    holds = sum(
        1 for r in rows if r.get("result") != "shadow_proposal_only")
    record = {
        "batch_id": batch_id,
        "date": day.isoformat(),
        "snapshot_id": payload.get("snapshot_id"),
        "rows": len(rows),
        "proposals": len(rows) - holds,
        "holds": holds,
        "ran_at": _to_shanghai(now).isoformat(),
    }
    batches.append(record)
    _save_window_state(base_dir, state)
    logger.info(
        "影子观察批次 %s 已落盘：%d 行，提议 %d、挂起 %d。",
        batch_id, record["rows"], record["proposals"], record["holds"])
    return {"status": "ran", **record}
