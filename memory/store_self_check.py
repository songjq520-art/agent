# memory/store_self_check.py（自检适配器：把记忆库接到自检的九项表上）
# self_check.run_self_check 只认一个 adapter 对象（见 memory/self_check.py
# 的适配器约定），这里把 store 的现成能力包成那七个方法。
# 只查不改：这个文件里不写库、不删东西；verify_backup 的演练写临时目录，
# 演练完就删，不留备份文件。
# 晋升计数重放（replay_promotion_counts）目前只做“内部一致”检查：
# 出现次数只写在 metadata 的集合里，账本里没有逐次出现事件，所以
# 从账本重放不出完整的出现记录——这里对的是集合大小 vs decay_seen_days
# 字段、日期格式、会话集合里的空串。真要从原始对话日志全量重放，
# 得另接 history.db 的对话记录（以后再接，见方法 docstring）。

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from memory import backup as _backup
from memory import ledger as _ledger
from memory.decay import DECAY_STATE_DORMANT, _parse_seen_list
from memory.timeutil import _now_dt, _parse_created_at


class _StoreSelfCheckAdapter:
    """把一个记忆库包成自检要的 adapter。"""

    def __init__(self, store):
        self._store = store

    def get_all_items(self):
        items = []
        for item in self._store.get_all(include_sensitive=True) or []:
            items.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text") or "",
                    "metadata": (item or {}).get("metadata") or {},
                }
            )
        return items

    def supersede_events_since(self, days):
        """最近 days 天内被取代的记忆（账本 supersede 事件）。
        事件没时间戳的跳过（定不了是哪天的，不猜）。

        账本读失败或不完整（坏行/截断）时抛 RuntimeError 把存疑
        传出去（H3 补修轮 2 F2-R1）；窗口内有 seq 缺口（整行
        丢失）同样抛——丢的那条没法证明不在窗口内（F2-R2 残留，
        2026-11-27）：这份结果要进自检报告的
        replaced_last_month——残缺结果冒充完整、读失败冒充
        「最近没有取代事件」，都会让报告看着健康实则没查。
        自检框架的承接方式就是异常：run_self_check 不吞适配器
        异常，外层维护调度/体检适配器会把整轮自检记为失败，
        不会记成一份干净的空报告。"""
        try:
            events, _info = _ledger.read_events(self._store.ledger_path)
        except Exception as exc:
            raise RuntimeError(
                "账本读取失败，取代事件查不了：%s" % exc
            ) from exc
        if (_info or {}).get("bad_lines") or (_info or {}).get("truncated"):
            raise RuntimeError(
                "账本不完整（坏行 %s 行%s），取代事件只能读出一部分，"
                "不冒充完整结果。"
                % (
                    (_info or {}).get("bad_lines") or 0,
                    "、末尾有写一半的残行"
                    if (_info or {}).get("truncated") else "",
                )
            )
        try:
            days = float(days)
        except (TypeError, ValueError):
            return []
        # 截止线用 _now_dt()（单调时钟），不用 datetime.now()：
        # 账本事件的 ts 是 _now_iso() 写的（同一单调源）；Windows 粗时钟
        # 下 raw now() 可能比刚写的事件 ts 还早（同一滴答里 _now_dt
        # 会往前推 1 微秒），那样 days=0 也会把刚发生的事件算进来。
        # 同一条时间线比先后才不会跑反（2026-11-19 全套 flaky 的根子）。
        cutoff = _now_dt().timestamp() - days * 86400.0
        # seq 缺口存疑（F2-R2 残留，2026-11-27）：_info 只报坏行
        # 和截断，整行丢失它看不见——被删的若恰是窗口内一笔取代
        # 事件，结果就静默少一条、冒充完整。缺口紧挨着的前/后
        # 一条事件只要有一条落在窗口内，就没法证明丢的那条在
        # 窗口外（事件时间戳随 seq 单调），按存疑抛出去，口径
        # 同上面的坏行/截断；前后邻居都在窗口外时，丢的那条
        # 夹在两条窗口外事件之间，才确证与本窗口无关。
        _seq_ts = sorted(
            (
                (event.get("seq"), _parse_created_at(event.get("ts")))
                for event in events or []
                if isinstance((event or {}).get("seq"), int)
            ),
            key=lambda pair: pair[0],
        )
        if _seq_ts:
            _present = {seq for seq, _when in _seq_ts}
            for _gap in range(1, _seq_ts[-1][0] + 1):
                if _gap in _present:
                    continue
                _neighbors = []
                _before = [w for s, w in _seq_ts if s < _gap]
                _after = [w for s, w in _seq_ts if s > _gap]
                if _before:
                    _neighbors.append(_before[-1])
                if _after:
                    _neighbors.append(_after[0])
                if any(
                    w is None or w.timestamp() >= cutoff
                    for w in _neighbors
                ):
                    raise RuntimeError(
                        "账本序号有缺口（第 %d 号事件整行丢了），"
                        "丢的事件没法证明不在最近 %s 天内，取代"
                        "事件可能少报，不冒充完整结果。"
                        % (_gap, days)
                    )
        out = []
        for event in events or []:
            if (event or {}).get("type") != _ledger.LEDGER_EVENT_SUPERSEDE:
                continue
            when = _parse_created_at((event or {}).get("ts"))
            if when is None or when.timestamp() < cutoff:
                continue
            out.append(
                {
                    "id": (event or {}).get("memory_id")
                    or (event or {}).get("supersedes_id"),
                    "text": (event or {}).get("old_text")
                    or (event or {}).get("text")
                    or "",
                    "at": (event or {}).get("ts"),
                }
            )
        return out

    def verify_backup(self):
        """备份恢复演练：从当前库建一份备份、写临时文件、读回来验。
        演练完临时文件就删，不留备份。返回 {"ok", "detail"}。
        源账本不完整（坏行/截断）时直接 ok=False：读出的事件集合
        是残的，拿残账本建的备份再自洽，也证明不了源账本完整
        （H3 补修轮 2 F2-R1）。整行丢失（seq 缺口）同理判败：
        read_events 的 _info 报不出缺口，另走 verify_ledger 查
        序号连续性（F2-R2 残留，2026-11-27）。"""
        try:
            items = self._store.get_all(include_sensitive=True) or []
            events, _info = _ledger.read_events(self._store.ledger_path)
            if (_info or {}).get("bad_lines") or (_info or {}).get("truncated"):
                parts = []
                if (_info or {}).get("bad_lines"):
                    parts.append("坏行 %d 行" % _info["bad_lines"])
                if (_info or {}).get("truncated"):
                    parts.append("末尾有写一半的残行")
                return {
                    "ok": False,
                    "detail": "备份演练没通过：源账本不完整（%s），"
                    "读出的事件集合不全，不拿残缺账本建备份。"
                    % "、".join(parts),
                }
            _verdict = _ledger.verify_ledger(self._store.ledger_path)
            if _verdict.get("gaps"):
                return {
                    "ok": False,
                    "detail": "备份演练没通过：源账本不完整（序号"
                    "缺口：第 %s 号事件整行丢了），读出的事件集合"
                    "不全，不拿残缺账本建备份。"
                    % "、".join(
                        str(g) for g in _verdict["gaps"]
                    ),
                }
            backup = _backup.build_backup(items, events)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "self-check-backup.json"
                _backup.write_backup(path, backup)
                result = _backup.verify_backup(path)
        except Exception as exc:  # noqa: BLE001 - 演练跑不起来也要报出来
            return {"ok": False, "detail": "备份演练跑不起来：%s" % exc}
        if result.get("ok"):
            return {"ok": True, "detail": "备份演练通过：建、写、读、验都过了。"}
        errors = result.get("errors") or ["原因不明"]
        return {"ok": False, "detail": "备份演练没通过：" + "；".join(errors)}

    def index_counts(self):
        """两边索引的条数：sqlite（history.db 的 memory_id）对
        chroma（get_all 的 id）。history.db 没有、没 history 表时
        sqlite 按 0 条算（不是失败——FakeMemory 测试就是这样）。
        mismatched_ids 是只在一边的 id（排好序，确定性）。"""
        chroma_ids = {
            item["id"] for item in self.get_all_items() if item.get("id")
        }
        sqlite_ids = set()
        try:
            con = sqlite3.connect(str(self._store.history_db_path), timeout=5)
            try:
                rows = con.execute(
                    "SELECT DISTINCT memory_id FROM history"
                ).fetchall()
                sqlite_ids = {row[0] for row in rows if row and row[0]}
            except sqlite3.Error:
                sqlite_ids = set()
            finally:
                con.close()
        except sqlite3.Error:
            sqlite_ids = set()
        return {
            "sqlite": len(sqlite_ids),
            "chroma": len(chroma_ids),
            "mismatched_ids": sorted(sqlite_ids ^ chroma_ids),
        }

    def replay_promotion_counts(self):
        """晋升计数内部一致检查（见模块 docstring：账本里没有逐次
        出现事件，全量重放得另接对话日志，这里只对内部一致）。
        每条记忆：decay_seen_dates 集合大小应该等于 decay_seen_days
        字段；日期应该是 YYYY-MM-DD；会话集合里不该有空串。
        返回 {"checked", "corrections": [{id, field, old, new}]}。"""
        checked = 0
        corrections = []
        for item in self.get_all_items():
            meta = item.get("metadata") or {}
            checked += 1
            dates = _parse_seen_list(meta.get("decay_seen_dates"))
            sessions = _parse_seen_list(meta.get("decay_seen_sessions"))
            seen_days = meta.get("decay_seen_days")
            if (
                isinstance(seen_days, int)
                and not isinstance(seen_days, bool)
                and seen_days != len(dates)
            ):
                corrections.append(
                    {
                        "id": item.get("id"),
                        "field": "decay_seen_days",
                        "old": seen_days,
                        "new": len(dates),
                    }
                )
            for day in dates:
                try:
                    datetime.strptime(day, "%Y-%m-%d")
                except (TypeError, ValueError):
                    corrections.append(
                        {
                            "id": item.get("id"),
                            "field": "decay_seen_dates",
                            "old": day,
                            "new": None,
                        }
                    )
                    break
            if any(not session for session in sessions):
                corrections.append(
                    {
                        "id": item.get("id"),
                        "field": "decay_seen_sessions",
                        "old": "含空会话",
                        "new": [s for s in sessions if s],
                    }
                )
        return {"checked": checked, "corrections": corrections}

    def archive_items(self):
        """档案层记忆（沉底态 dormant 的那几条）——档案就是打标，
        不另建一份 JSONL（账本里有全部原话，不再造第二个真相来源）。"""
        return [
            {"id": item["id"], "text": item["text"]}
            for item in self.get_all_items()
            if (item.get("metadata") or {}).get("decay_state")
            == DECAY_STATE_DORMANT
        ]

    def embed(self, text):
        """本地向量（bge，不联网、不花钱）；模型不可用时返回 None，
        第 1 项（重复记忆）跳过并注明，不拦自检。"""
        try:
            return self._store._embed_text(text)
        except Exception:
            return None


def build_self_check_adapter(store):
    """给一个记忆库建自检适配器。"""
    return _StoreSelfCheckAdapter(store)


def make_model_judge(router):
    """把整理记忆的路由包成自检第 9 项要的 model_judge。
    judge(system_prompt, user_prompt) -> {"accurate": bool, "evidence": str}
    或 None（判不出来）。路由调不通、回的不是合法 JSON、accurate 不是
    布尔值时都返回 None——抽查判不出来就记 skipped，不猜。"""
    import re

    def judge(system_prompt, user_prompt):
        try:
            out = router.judge(
                system_prompt,
                user_prompt,
                max_tokens=500,
                purpose="self_check",
            )
        except Exception:
            return None
        if out is None:
            return None
        text = (out.get("text") or "").strip()
        payload = None
        try:
            payload = json.loads(text)
        except ValueError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except ValueError:
                    payload = None
        if not isinstance(payload, dict):
            return None
        accurate = payload.get("accurate")
        if not isinstance(accurate, bool):
            return None
        evidence = payload.get("evidence")
        return {
            "accurate": accurate,
            "evidence": evidence if isinstance(evidence, str) else "",
        }

    return judge
