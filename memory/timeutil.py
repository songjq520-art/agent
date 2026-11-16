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

def _now_iso():
    """现在的时间，UTC 的 ISO 时间串（固定 6 位微秒，带 Z），供 metadata 用。

    小数部分固定 6 位：isoformat() 默认会省略微秒为 0 时的小数点，
    '...T03:00:00Z' 按字符串排会比 '...T03:00:00.000001Z' 大——
    字符串顺序就和时间顺序反了。固定 6 位后，字符串顺序就是时间顺序。
    """
    global _NOW_LAST
    now = datetime.now(timezone.utc)
    with _NOW_LOCK:
        if _NOW_LAST is not None and now <= _NOW_LAST:
            now = _NOW_LAST + timedelta(microseconds=1)
        _NOW_LAST = now
    return now.isoformat(timespec="microseconds").replace("+00:00", "Z")

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
# 一目了然。序号只管同一进程里的先后，跨进程重启从 0 再来——新旧判断
# 先看 created_at，序号只是平局时的决胜（见 _memory_order_key）。
# 真正的写操作本来就串在 _WRITE_LOCK 里，这里这把锁只是兜底。
_WRITE_SEQ_LOCK = threading.Lock()

_WRITE_SEQ = 0

def _next_write_seq():
    """下一条写入序号（进程内严格递增）。"""
    global _WRITE_SEQ
    with _WRITE_SEQ_LOCK:
        _WRITE_SEQ += 1
        return _WRITE_SEQ

