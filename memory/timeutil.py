# memory/timeutil.py（时间小工具：单调时间串、写序号）
# 从 memory/mem0_store.py 拆出来，行为不变。

import threading
from datetime import date, datetime, timedelta, timezone

def _today():
    """今天的日期，格式 YYYY-MM-DD。"""
    return date.today().isoformat()

# _NOW_LOCK / _NOW_LAST：保证 _now_iso() 发出的时间戳严格递增。
# 秒级精度时，两轮对话同一秒写进去，created_at 一撞就分不出新旧——
# 和 conversation_summary 的 _now_iso 修过的是同一类问题（PR #13，
# BOOS 笔记本上跑全套时撞上过一次）。这里不能直接 import 那边
# （conversation_summary 反过来 import 本模块的 _redact_sensitive，
# 循环了），所以同一套算法、各自一份状态，各管各的域。
# 每次发值时如果新值 <= 上一次发的值，就在上一次的值上加 1 微秒再发。
# 时钟往回调也不会乱：每次只比上一次大 1 微秒，墙上时间追上来就恢复正常。
# 写记忆是多线程的（网页服务），读-改-写上锁，别让两个线程同时拿到同一个值。
_NOW_LOCK = threading.Lock()

_NOW_LAST = None

def _now_dt():
    """单调递增的现在（datetime，UTC）：同一进程里每次调用都比上一次大。

    _now_iso() 和 _now_epoch() 共用这一个源——删记忆的时间戳、
    待定轮的 created_at、账本的 ts 必须在同一条单调时间线上比
    先后，否则 Windows 粗时钟下会撞车或跑反：raw datetime.now()
    在同一滴答里返回值相同，_now_iso() 的 +1 微秒又会跑到墙钟
    前面去，删在前的事反而显得比后说的轮次晚（2026-11-19 全套
    flaky 的根子，mem0_forget / revive_guard 在笔记本上时挂时过）。
    """
    global _NOW_LAST
    now = datetime.now(timezone.utc)
    with _NOW_LOCK:
        if _NOW_LAST is not None and now <= _NOW_LAST:
            now = _NOW_LAST + timedelta(microseconds=1)
        _NOW_LAST = now
    return now


def _now_epoch():
    """单调递增的现在（epoch 秒，float）：和 _now_iso() 同一个源，
    供删记忆的时间戳、待定轮的 created_at 用——它们要和账本 ts
    （_now_iso 写）比先后，必须走同一条单调线，别用 raw
    datetime.now().timestamp()（见 _now_dt）。"""
    return _now_dt().timestamp()


def _now_iso():
    """现在的时间，UTC 的 ISO 时间串（固定 6 位微秒，带 Z），供 metadata 用。

    小数部分固定 6 位：isoformat() 默认会省略微秒为 0 时的小数点，
    '...T03:00:00Z' 按字符串排会比 '...T03:00:00.000001Z' 大——
    字符串顺序就和时间顺序反了。固定 6 位后，字符串顺序就是时间顺序。
    """
    return _now_dt().isoformat(timespec="microseconds").replace("+00:00", "Z")

def _parse_created_at(value):
    """把 created_at 字符串解析成带时区的 datetime，解析不出返回 None。

    认三种写法：新版微秒 + Z（"2026-11-10T12:00:00.000001Z"）、
    旧版秒级 +00:00（"2026-11-09T10:00:00+00:00"）、迁移带来的
    纯日期（"2026-11-07"，算当天 0 点）。没时区的当 UTC。
    原来在 memory/provenance.py，2026-11-13 挪到这里：memory/decay.py
    也要用它，而 provenance.py 要引 decay.py 的 classify_decay_kind，
    放 provenance 里会循环引用；时间解析本来就是时间小工具，放这里
    最干净。provenance.py 里留一个转引，旧的 import 不用改。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        if len(text) == 10:
            parsed = datetime.strptime(text, "%Y-%m-%d")
            return parsed.replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# _WRITE_SEQ_LOCK / _WRITE_SEQ：同一批写进去的序号（进程内严格递增）。
# 同一批写进去的几条记忆，时间戳也可能撞在同一微秒——单调 +1 微秒能
# 分开绝大多数，但序号是更稳的第二道：同一批里每条各拿一个号，先后
# 一目了然。新旧判断先看 created_at，序号只是平局时的决胜
# （见 _memory_order_key）。序号本来只管同一进程里的先后，重启从 0
# 再来——2026-11-19 改成持久化：Mem0Store 启动时扫一遍库里已有的
# write_seq，用最大的那个喂给 _seed_write_seq（见下面），之后新写
# 的序号一定比库里的大，重启也不归零。真正的写操作本来就串在
# _WRITE_LOCK 里，这里这把锁只是兜底。
_WRITE_SEQ_LOCK = threading.Lock()

_WRITE_SEQ = 0

def _next_write_seq():
    """下一条写入序号（进程内严格递增）。"""
    global _WRITE_SEQ
    with _WRITE_SEQ_LOCK:
        _WRITE_SEQ += 1
        return _WRITE_SEQ

def _seed_write_seq(value):
    """把写序号的起点抬到至少 value（启动时用库里最大的 write_seq 喂一次）。

    只往上抬、不往下压：value 不是整数、或比当前值小，都不动——
    进程里已经发出去的序号绝不能重发（重发会让新旧判断撞车）。
    返回抬完之后的当前值。"""
    global _WRITE_SEQ
    with _WRITE_SEQ_LOCK:
        if isinstance(value, int) and not isinstance(value, bool) and value > _WRITE_SEQ:
            _WRITE_SEQ = value
        return _WRITE_SEQ

