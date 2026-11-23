# memory/ledger.py（事件账本：只追加、不修改，原话逐字存）
# 最终方案（2026-11-13 晚 BOOS 定稿）的做法层：DeepSeek 的账本 +
# 投影过滤，mem0 只当可丢弃的语义索引。
#
# 账本是家、筛子是门、备份是保险：
#   - 账本（memory_events.jsonl）：每行一个事件 JSON，只追加不改。
#     先写账本、再写 mem0 索引；记账本中间关服务器也不怕——一行写完
#     才算数，写一半没结尾的、重启认出来扔掉，账本还是整的；索引坏了
#     从账本重建（见 Mem0Store.reconcile_ledger_index）。
#   - 筛子（project_and_filter，纯函数）：检索回来之后再过滤，被取代的、
#     睡着的、删了的默认不给；问“以前”时才放出来，并标“这是以前的”。
#   - 兜底（find_in_ledger）：mem0 里找不够，直接翻账本原话按关键词找。
#
# 事件类型（BOOS 定了 6 种，都是记忆本身的状态变化）：
#   add（新记一条，带原话逐字 + 写时的整份 metadata）、supersede
#   （旧的被新说法取代，旧 id 标上，原话留着）、dormant（30 天没人找
#   只打标沉底）、revive（沉底的被用上、回到原话态）、delete（真删了，
#   只等 BOOS 亲口说）、promote（一次性经历出现够 3 个不同天、晋升成
#   稳定事实）。揣摩人心又加了一种（2026-11-16，见 memory/inference.py）：
#   inference（猜到了 BOOS 心里想什么，带猜了什么、凭什么、把握、层、
#   状态 pending/confirmed/deleted；确认/否认是同一类型的新事件，
#   按 key 折叠、最新状态赢）。另外两种是机器用的，不是记忆事实：
#   relate（两条记忆的关系记录，见 memory/relations.py）、snapshot
#   （一年压缩一次的快照标记，见 compact_ledger）。
#
# 序号：每行带 seq，严格递增；启动时对序号（verify_ledger 查缺口，
# reconcile 时按 key 补索引）。“add”事件没有 mem0 id（写账本时索引
# 还没建出来），用 key（原话 + created_at 的哈希）当稳定键——对账、
# 去重、重建都按 key 认，不靠 id。
#
# 快照压缩（BOOS 21:54 定）：账本一年以后压缩一次，每年压一次；压缩
# 是把算好的状态存成快照、原话都在快照里，压完就不用删了。compact_ledger
# 把当前状态写进 memory_events_snapshot.json（带校验和），账本截断
# 到只剩一行 snapshot 标记，序号照旧往下走，不断档。

import hashlib
import json
import os
from pathlib import Path

from memory.keywords import _extract_keywords, _keyword_weight
from memory.timeutil import _now_iso

# 账本事件类型：6 种记忆事实 + 推断 + 2 种机器标记。
LEDGER_EVENT_ADD = "add"
LEDGER_EVENT_SUPERSEDE = "supersede"
LEDGER_EVENT_DORMANT = "dormant"
LEDGER_EVENT_REVIVE = "revive"
LEDGER_EVENT_DELETE = "delete"
LEDGER_EVENT_PROMOTE = "promote"
LEDGER_EVENT_INFERENCE = "inference"
LEDGER_EVENT_RELATE = "relate"
LEDGER_EVENT_SNAPSHOT = "snapshot"
LEDGER_EVENT_TOUCH = "touch"
# R-7（2026-11-24）：补槽位标事件——槽位标以前只写 SQLite 索引，
# 对账重建缺失行时无从恢复；记一笔 slot 事件（带 memory_id/key 和
# slot），重建时按它把标带回来。只是索引提示，不改变记忆状态，
# resolve_state 不处理它（状态层不认这个类型，无副作用）。
LEDGER_EVENT_SLOT = "slot"
LEDGER_EVENT_TYPES = (
    LEDGER_EVENT_ADD,
    LEDGER_EVENT_SUPERSEDE,
    LEDGER_EVENT_DORMANT,
    LEDGER_EVENT_REVIVE,
    LEDGER_EVENT_DELETE,
    LEDGER_EVENT_PROMOTE,
    LEDGER_EVENT_INFERENCE,
    LEDGER_EVENT_RELATE,
    LEDGER_EVENT_SNAPSHOT,
    LEDGER_EVENT_TOUCH,
    LEDGER_EVENT_SLOT,
)

# 账本里的状态名（resolve_state 的 by_id 值里 "state" 字段的取值）。
LEDGER_STATE_CURRENT = "current"
LEDGER_STATE_SUPERSEDED = "superseded"
LEDGER_STATE_DORMANT = "dormant"
LEDGER_STATE_DELETED = "deleted"

# 账本兜底关键词的命中线，和 mem0_store.KEYWORD_MIN_WEIGHT 同值（2）——
# 账本里翻出来的和检索里关键词补充信号的标准一致，不各说各话。
LEDGER_KEYWORD_MIN_WEIGHT = 2

SNAPSHOT_VERSION = 1


def ledger_event_key(text, created_at):
    """一条“add”事件的稳定键：原话 + created_at 的哈希。

    写账本时 mem0 id 还没建出来，key 就是这条记忆在账本里的身份——
    对账（reconcile）、去重、重建都按 key 认。同一句话在不同时间
    各记一次时，created_at 不同、key 也不同，不会串。"""
    payload = f"{text or ''}\n{created_at or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_lines(path):
    """读账本文件：返回 (完整行列表, 末尾有没有没写完的残行）。

    一行写完才算数：文件末尾没换行符的那行就是写一半断掉的（记账本
    中间关了服务器），不算事件，调用方修掉它（见 repair_ledger）。
    文件不存在按空账本算（刚启用，还没记过）。"""
    path = Path(path)
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    if not raw:
        return [], False
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    truncated = False
    if lines and lines[-1] != "":
        # 最后一行没换行符：写一半的残行。
        truncated = True
        lines = lines[:-1]
    return [line for line in lines if line.strip()], truncated


def read_events(path):
    """读出账本里的全部事件（按 seq 排好序）。

    返回 (events, info)：events 是事件字典列表；info 是
    {"truncated": 末尾有没有写一半的残行, "bad_lines": 解析不出的
    行数}。坏行（JSON 解析不出）跳过、不拦路，但记在 info 里——
    账本坏了一行不能把整本账都判死刑，也不能悄悄当没看见。
    """
    lines, truncated = _read_lines(path)
    events = []
    bad_lines = 0
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            bad_lines += 1
            continue
        if not isinstance(event, dict) or event.get("type") not in LEDGER_EVENT_TYPES:
            bad_lines += 1
            continue
        events.append(event)
    events.sort(key=lambda e: e.get("seq") if isinstance(e.get("seq"), int) else 0)
    return events, {"truncated": truncated, "bad_lines": bad_lines}


def verify_ledger(path):
    """查账本：序号缺没缺、最后一行写没写完、有没有坏行。

    返回 {"ok", "events", "seq_max", "gaps", "truncated", "bad_lines"}。
    ok 只在“序号连续、没残行、没坏行”时为 True；有缺口也不抛异常，
    缺口列出来、调用方按死规矩补（见 Mem0Store.reconcile_ledger_index）。
    序号缺口的意思：某几行事件丢了（不是索引缺，是账本本身缺）——
    这种事不该发生，查出来是给自检技能留的口子（自检以后再加）。
    """
    events, info = read_events(path)
    seqs = [e["seq"] for e in events if isinstance(e.get("seq"), int)]
    gaps = []
    if seqs:
        seq_max = max(seqs)
        expected = set(range(1, seq_max + 1))
        gaps = sorted(expected - set(seqs))
    else:
        seq_max = 0
    return {
        "ok": not gaps and not info["truncated"] and info["bad_lines"] == 0,
        "events": len(events),
        "seq_max": seq_max,
        "gaps": gaps,
        "truncated": info["truncated"],
        "bad_lines": info["bad_lines"],
    }


def repair_ledger(path):
    """修账本：扔掉末尾写一半的残行（没结尾的那行不算数）。

    只动残行，别的行原样不动；修完返回 {"repaired", "dropped"}。
    坏行（JSON 解析不出的整行）不自动修——那可能是写坏了，也可能是
    人为改的，自动扔了就死无对证，留给 BOOS 定。
    """
    path = Path(path)
    lines, truncated = _read_lines(path)
    if not truncated:
        return {"repaired": False, "dropped": 0}
    dropped = 1
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, path)
    return {"repaired": True, "dropped": dropped}


def append_event(path, event, lock=None, seq=None):
    """往账本追加一个事件（原子写：写完 + 刷盘才算数）。

    event 里不用带 seq：这里按账本最后一行的 seq + 1 自动编，调用方
    只填 type 和业务字段（memory_id / text / meta / 原因）。
    seq 只有恢复（restore_from_backup）时才显式传——备份里的事件按
    原序号写回来，顺序和序号空间都不变；平时别传，自动编才不会撞号。
    type 不在 LEDGER_EVENT_TYPES 里时抛 ValueError——账本里不许出现
    没名没姓的事件类型，错了早点炸、别等查账时才发现。
    返回带 seq 的事件（原字典的拷贝，调用方传进来的那份不动）。
    写失败时抛 OSError，不吞掉——先写账本再写索引，账本写不进去，
    索引那步也别写（调用方 Mem0Store._ledger_append 会记日志、但
    不让这次写记忆静默“成功”）。
    """
    if not isinstance(event, dict) or event.get("type") not in LEDGER_EVENT_TYPES:
        raise ValueError(f"账本事件类型不对：{(event or {}).get('type')!r}。")
    if seq is not None and (not isinstance(seq, int) or isinstance(seq, bool) or seq < 1):
        raise ValueError(f"账本序号不对：{seq!r}。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _do_append():
        lines, _truncated = _read_lines(path)
        last_seq = 0
        for line in lines:
            try:
                parsed = json.loads(line)
            except (ValueError, TypeError):
                continue
            parsed_seq = parsed.get("seq") if isinstance(parsed, dict) else None
            if isinstance(parsed_seq, int) and not isinstance(parsed_seq, bool):
                last_seq = max(last_seq, parsed_seq)
        record = dict(event)
        record["seq"] = seq if seq is not None else last_seq + 1
        record.setdefault("ts", _now_iso())
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    if lock is not None:
        with lock:
            return _do_append()
    return _do_append()


def resolve_state(events, snapshot=None):
    """把一串账本事件折叠成当前状态（纯函数，不读库、不写库）。

    snapshot：compact_ledger 写过的快照（见 read_snapshot），有就先
    以它为底子、再折叠快照之后的新事件；没有就从空开始。
    返回 {"by_id": {mem0_id: {"state", ...}}, "adds": [add 事件的
    原话档案，按 seq 排], "relations": [关系记录], "key_states":
    {add 的 key: "current"/"superseded"/"dormant"/"deleted"},
    "id_to_key": {mem0_id: 当前原话的 key}, "inferences": {推断的 key:
    {"key", "text", "evidence", "confidence", "confidence_text",
    "layer", "mode", "status", "seq", "created_at", "partial"}，按 key 认},
    "seq_max": 最大序号,
    "snapshot_seq": 快照截止到的序号（没有快照就是 None）}。
    折叠规则（只认账本，不猜）：
      - add：进原话档案（key/text/meta/seq），不碰 by_id——写账本时
        mem0 id 还没建出来，id 要靠对账（reconcile）认；key 标 current；
      - inference：按 key 认一条推断——第一次出现时建档案（猜了什么 /
        凭什么 / 把握 / 层 / 路子 / 状态，缺的字段按空字符串 / pending
        补）；同一 key 再出现时只更新状态（和带过来的字段），原话、
        凭据这些不重写——确认/否认就是同一 key 的新事件，最新状态赢；
      - supersede：supersedes_id 和 memory_id 不同时，旧 id 标
        superseded（原话留着，档案里还有）；同一 id 时是原地纠正，
        状态还是 current，旧话记一条 history。两种都把旧原话的 key
        标 superseded（key 在事件里，没带时按 id_to_key 找）；新原话
        的 key（current_key，有就带）标 current，id 指向新 key；
      - dormant：标 dormant（delete_due 一起带上）；原话的 key 也标
        dormant（key 在事件里，没带时按 id_to_key 找）；
      - revive：回 current，清掉 dormant/delete_due；原话的 key 也
        回 current（key 在事件里，没带时按 id_to_key 找）；
      - delete：标 deleted；原话的 key 也标 deleted；
      - promote：kind 记 stable，状态不动；
      - touch：检索真用上了（2026-11-24 3a）——只记每个 id 的最后
        使用时间（last_touched，取事件 ts，最新赢），不改状态；
        对账重建时按它恢复 last_used_at（快照不带这个字段，压缩后
        丢了只是时间退回 add 时的值，不丢记忆）；
      - relate：进 relations 列表（双向都认）；
      - snapshot：只记 snapshot_seq，不改状态（状态在快照文件里）。
    key_states 是给“按原话认”的流程用的：检索兜底和对账认的都是
    key（原话 + created_at 的哈希），不是 id——id 只在索引里有意义，
    账本里的原话没有 id。没有 key 记录的 add（老事件、或事件没带
    key）默认当 current：账本没说它有问题，就别擅自把它当旧的。
    """
    by_id = {}
    adds = []
    relations = []
    key_states = {}
    id_to_key = {}
    inferences = {}
    last_touched = {}
    seq_max = 0
    snapshot_seq = None
    if snapshot is not None:
        snapshot_seq = snapshot.get("snapshot_seq")
        for mem_id, entry in (snapshot.get("by_id") or {}).items():
            by_id[mem_id] = dict(entry)
        for entry in snapshot.get("adds") or []:
            adds.append(dict(entry))
        for entry in snapshot.get("relations") or []:
            relations.append(dict(entry))
        for key, state in (snapshot.get("key_states") or {}).items():
            key_states[key] = state
        for mem_id, key in (snapshot.get("id_to_key") or {}).items():
            id_to_key[mem_id] = key
        for key, entry in (snapshot.get("inferences") or {}).items():
            inferences[key] = dict(entry)
        for _tid, _ts in (snapshot.get("last_touched") or {}).items():
            last_touched[_tid] = _ts
        if isinstance(snapshot_seq, int):
            seq_max = snapshot_seq
    for event in events or []:
        seq = event.get("seq") if isinstance(event.get("seq"), int) else 0
        seq_max = max(seq_max, seq)
        etype = event.get("type")
        mem_id = event.get("memory_id")
        key = event.get("key")
        if etype == LEDGER_EVENT_ADD:
            adds.append(
                {
                    "key": event.get("key"),
                    "text": event.get("text") or "",
                    "meta": dict(event.get("meta") or {}),
                    "seq": seq,
                }
            )
            if key:
                key_states[key] = LEDGER_STATE_CURRENT
        elif etype == LEDGER_EVENT_INFERENCE:
            # 推断按 key 认：第一次出现建档案，之后同一 key 的事件只
            # 更新状态（确认/否认就记一笔同 key 的新事件，最新状态赢）。
            if key:
                record = inferences.get(key)
                if record is None:
                    record = {
                        "key": key,
                        "text": event.get("text") or "",
                        "evidence": event.get("evidence") or "",
                        "confidence": event.get("confidence") or "low",
                        "confidence_text": event.get("confidence_text") or "我只是猜的",
                        "layer": event.get("layer") or "induction",
                        "mode": event.get("mode") or "spot",
                        "status": event.get("status") or "pending",
                        "seq": seq,
                        "created_at": event.get("created_at") or "",
                        # 部分确认（“差不多”“对了一半”）：只认了，没全认。
                        # 建档案时默认 False，确认那一笔事件带了 True 才变 True。
                        "partial": bool(event.get("partial")),
                    }
                    inferences[key] = record
                else:
                    for field in (
                        "text",
                        "evidence",
                        "confidence",
                        "confidence_text",
                        "layer",
                        "mode",
                        "status",
                        "partial",
                    ):
                        if event.get(field):
                            record[field] = event.get(field)
                    record["seq"] = seq
        elif etype == LEDGER_EVENT_SUPERSEDE:
            old_id = event.get("supersedes_id")
            if old_id and old_id != mem_id:
                by_id[old_id] = {
                    "state": LEDGER_STATE_SUPERSEDED,
                    "superseded_by": mem_id,
                    "seq": seq,
                }
            elif mem_id:
                entry = by_id.setdefault(mem_id, {"state": LEDGER_STATE_CURRENT})
                entry.setdefault("history", []).append(
                    {
                        "type": "supersede",
                        "old_text": event.get("old_text") or "",
                        "seq": seq,
                    }
                )
            old_key = key or id_to_key.get(old_id or mem_id)
            if old_key:
                key_states[old_key] = LEDGER_STATE_SUPERSEDED
            current_key = event.get("current_key")
            if current_key:
                key_states[current_key] = LEDGER_STATE_CURRENT
                if mem_id:
                    id_to_key[mem_id] = current_key
        elif etype == LEDGER_EVENT_DORMANT:
            if mem_id:
                by_id[mem_id] = {
                    "state": LEDGER_STATE_DORMANT,
                    "delete_due": bool(event.get("delete_due")),
                    "seq": seq,
                }
            dormant_key = key or id_to_key.get(mem_id)
            if dormant_key:
                key_states[dormant_key] = LEDGER_STATE_DORMANT
                if mem_id:
                    id_to_key[mem_id] = dormant_key
        elif etype == LEDGER_EVENT_REVIVE:
            if mem_id:
                by_id[mem_id] = {"state": LEDGER_STATE_CURRENT, "seq": seq}
            revive_key = key or id_to_key.get(mem_id)
            if revive_key:
                key_states[revive_key] = LEDGER_STATE_CURRENT
                if mem_id:
                    id_to_key[mem_id] = revive_key
        elif etype == LEDGER_EVENT_DELETE:
            if mem_id:
                by_id[mem_id] = {"state": LEDGER_STATE_DELETED, "seq": seq}
            delete_key = key or id_to_key.get(mem_id)
            if delete_key:
                key_states[delete_key] = LEDGER_STATE_DELETED
        elif etype == LEDGER_EVENT_PROMOTE:
            if mem_id:
                entry = by_id.setdefault(mem_id, {"state": LEDGER_STATE_CURRENT})
                entry["kind"] = "stable"
                entry["promoted"] = True
                entry["seq"] = seq
            if key and mem_id:
                id_to_key[mem_id] = key
        elif etype == LEDGER_EVENT_RELATE:
            relations.append(
                {
                    "memory_id": mem_id,
                    "related_id": event.get("related_id"),
                    "relation": event.get("relation"),
                    "note": event.get("note") or "",
                    "seq": seq,
                }
            )
        elif etype == LEDGER_EVENT_TOUCH:
            _touch_ts = event.get("ts")
            if _touch_ts:
                for _tid in event.get("memory_ids") or []:
                    if _tid and _touch_ts >= str(last_touched.get(_tid) or ""):
                        last_touched[_tid] = _touch_ts
                if key and _touch_ts >= str(last_touched.get(key) or ""):
                    last_touched[key] = _touch_ts
        elif etype == LEDGER_EVENT_SNAPSHOT:
            snapshot_seq = event.get("snapshot_seq")
    return {
        "by_id": by_id,
        "adds": adds,
        "relations": relations,
        "key_states": key_states,
        "id_to_key": id_to_key,
        "inferences": inferences,
        "last_touched": last_touched,
        "seq_max": seq_max,
        "snapshot_seq": snapshot_seq,
    }


def project_and_filter(items, state, include_history=False):
    """筛子：检索回来之后再过滤（纯函数）。

    默认只放“当前”的：被取代的（superseded）、删了的（deleted）
    不给；问“以前”时（include_history=True）才放出来，并标
    is_history=True 和 history_state（“这是以前的”）。
    已作废的（voided）永远不给：作废不是“以前”，问“以前”也不放
    出来（Bug 1，2026-11-20）——作废的玩笑原话不能再被当成事实注入。
    睡着的（dormant）不屏蔽：它是衰减的标签（30 天没人找），不是
    “不是当前了”——衰减分支的定法是权重只管排位、不管能不能搜到，
    睡着的照常能搜到、被用上就唤醒；筛子把它屏蔽掉就和衰减分支
    打架了（test_dormant_memory_searchable_and_woken_on_use）。
    账本里没记录的，退到 mem0 metadata 上的标记（superseded）——
    老记忆在账本启用之前就被标过，不能因为账本没记就放出来；
    metadata 上的 dormant 同样不屏蔽（同上）。
    返回的新列表里每项都是原字典的拷贝，不动输入。
    """
    by_id = (state or {}).get("by_id") or {}
    key_states = (state or {}).get("key_states") or {}
    out = []
    for item in items or []:
        mem_id = (item or {}).get("id")
        entry = by_id.get(mem_id) if mem_id else None
        ledger_state = (entry or {}).get("state") if entry else None
        meta = (item or {}).get("metadata") or {}
        if ledger_state is None:
            if meta.get("superseded") is True:
                ledger_state = LEDGER_STATE_SUPERSEDED
            else:
                # id 查不着时再按 key 查（key = 原话 + created_at 的哈希，
                # 见 ledger_event_key）：图上找回来的记忆没有 mem0 id
                # （id=None，账本档案里的原话），by_id 永远查不着它——
                # 不按 key 查，筛子就认不出它是被取代/删了的，问“以前”
                # 时 is_history 会错标成 False。图是索引不是账本，筛子
                # 的口径还是账本说了算，这里只是多一条查法。
                try:
                    item_key = ledger_event_key(
                        (item or {}).get("text") or "",
                        meta.get("created_at"),
                    )
                except Exception:
                    item_key = None
                if item_key:
                    ledger_state = key_states.get(item_key)
        if meta.get("voided") is True:
            # Bug 1（2026-11-20）：已作废的玩笑记忆永远不返回——作废
            # 不是“以前”，问“以前”也不放出来；作废的原话不能再被
            # 当成事实注入。账本里没有 voided 状态（void_joke_memory
            # 只写 Chroma metadata），这里只认 metadata 上的标记。
            continue
        if ledger_state in (
            LEDGER_STATE_SUPERSEDED,
            LEDGER_STATE_DELETED,
        ):
            if not include_history:
                continue
            kept = dict(item)
            kept["is_history"] = True
            kept["history_state"] = ledger_state
            out.append(kept)
        else:
            kept = dict(item)
            kept["is_history"] = False
            out.append(kept)
    return out


def find_in_ledger(events, query, top_k=5, exclude_keys=None):
    """账本兜底：mem0 里找不够，直接翻账本原话按关键词找（纯函数）。

    只翻 "add" 事件的原话（原话逐字存，提炼句不在账本里）；按关键词
    权重排，命中线和检索里的关键词补充信号一致（≥2）。exclude_keys
    是已经在检索结果里的 key，翻出来的不重复给。
    返回 [{"key", "text", "score", "seq", "meta"}]，score 是关键词
    权重（不是语义分，调用方别把它和语义分放一起比）。
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return []
    excluded = set(exclude_keys or [])
    scored = []
    for event in events or []:
        if event.get("type") != LEDGER_EVENT_ADD:
            continue
        key = event.get("key")
        if key in excluded:
            continue
        text = event.get("text") or ""
        if not text:
            continue
        weight, _matched = _keyword_weight(text, keywords)
        if weight >= LEDGER_KEYWORD_MIN_WEIGHT:
            scored.append(
                {
                    "key": key,
                    "text": text,
                    "score": weight,
                    "seq": event.get("seq") if isinstance(event.get("seq"), int) else 0,
                    "meta": dict(event.get("meta") or {}),
                }
            )
    scored.sort(key=lambda entry: (entry["score"], entry["seq"]), reverse=True)
    return scored[:top_k]


def snapshot_checksum(payload):
    """快照内容的校验和：除 checksum 本身外的字段按固定顺序拼起来算。
    备份也要验快照（见 memory/backup.py），所以公开出来。"""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()





def write_snapshot(path, state, seq_max):
    """把算好的状态写成快照（原子写：临时文件 + 替换）。

    快照里有：截止序号 snapshot_seq、by_id 状态、全部原话档案 adds、
    关系 relations、原话状态 key_states、id 到原话的映射 id_to_key、
    推断档案 inferences、写快照的时间；最后附 checksum，读的时候验。
    原话都在快照里——压完账本截断了，快照就是档案。
    返回快照文件路径（字符串）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SNAPSHOT_VERSION,
        "snapshot_seq": seq_max,
        "created_at": _now_iso(),
        "by_id": (state or {}).get("by_id") or {},
        "adds": (state or {}).get("adds") or [],
        "relations": (state or {}).get("relations") or [],
        "key_states": (state or {}).get("key_states") or {},
        "id_to_key": (state or {}).get("id_to_key") or {},
        "inferences": (state or {}).get("inferences") or {},
    }
    payload["checksum"] = snapshot_checksum(payload)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
    return str(path)


def read_snapshot(path):
    """读快照并验校验和：对不上就抛 ValueError，不悄悄认一个坏快照。

    文件不存在返回 None（还没压过快照）。"""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("快照文件不是字典。")
    checksum = payload.get("checksum")
    body = {key: value for key, value in payload.items() if key != "checksum"}
    if checksum != snapshot_checksum(body):
        raise ValueError("快照校验和对不上，快照可能坏了。")
    return payload


def write_snapshot_raw(path, payload):
    """把一份现成的快照字典原样写到文件（恢复时用）。

    先验校验和：对不上就抛 ValueError，不把坏快照写回去。
    原子写（临时文件 + 替换）。返回文件路径（字符串）。"""
    if not isinstance(payload, dict):
        raise ValueError("快照不是字典。")
    checksum = payload.get("checksum")
    body = {key: value for key, value in payload.items() if key != "checksum"}
    if checksum != snapshot_checksum(body):
        raise ValueError("快照校验和对不上，快照可能坏了。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
    return str(path)


def load_ledger_state(ledger_path, snapshot_path):
    """读整本账：快照（有的话）打底，再折叠快照之后的新事件。

    返回 resolve_state 的结果。快照读坏了抛 ValueError——坏快照
    不能当底子，调用方要先修快照（从账本重算一份）再继续。
    """
    events, _info = read_events(ledger_path)
    snapshot = None
    if any(e.get("type") == LEDGER_EVENT_SNAPSHOT for e in events):
        snapshot = read_snapshot(snapshot_path)
    return resolve_state(events, snapshot=snapshot)


def compact_ledger(ledger_path, snapshot_path):
    """一年压缩一次：把账本压成快照（原话都在快照里），账本截断。

    步骤：读全部事件 → 折叠成当前状态 → 写快照（带校验和）→ 账本
    重写成只剩一行 snapshot 标记（seq 接着往下走，不断档）。
    压之前先修残行（repair_ledger）：写一半的那行不算数，别把它
    压进快照里。返回 {"snapshot_path", "snapshot_seq",
    "events_archived", "truncated"}。
    压完就不用删了——快照就是档案；真想删记忆还是等 BOOS 亲口说，
    压缩本身不删任何记忆。
    """
    ledger_path = Path(ledger_path)
    repaired = repair_ledger(ledger_path)
    events, _info = read_events(ledger_path)
    state = resolve_state(events)
    seq_max = state["seq_max"]
    write_snapshot(snapshot_path, state, seq_max)
    marker = {
        "type": LEDGER_EVENT_SNAPSHOT,
        "snapshot_seq": seq_max,
        "seq": seq_max + 1,
        "ts": _now_iso(),
    }
    tmp = ledger_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, ledger_path)
    return {
        "snapshot_path": str(snapshot_path),
        "snapshot_seq": seq_max,
        "events_archived": len(events),
        "truncated": repaired["repaired"],
    }
