# tests/test_chat_extraction_runs.py（回归：正常聊天一轮后，提取必须跑、记忆必须落库）
# 根因（2026-11-20 游泳那句）：
#   正常轮次 save_turn -> outbox 后台线程 -> add_turn，add_turn 按 Plan C
#   把本轮先放进内存待定区（_pending_turn），提取只在“下一轮 add_turn 提交
#   上一轮”或“关机 flush_pending”时才跑。最后一轮没有下一轮、又没走到
#   干净关机时，提取永远不跑；更糟的是 outbox 的落盘文件在 add_turn 只
#   是“待定”返回后就被删了，重启也捞不回来——2026-11-20 09:04 游泳那句
#   就是这么丢的（日志 3 分钟无 mem0 操作、库里没有游泳内容、outbox_pending 为空）。
# 全部离线（FakeMemory + 隔离的 outbox 落盘目录），不联网、不产生费用。
# 注意 FakeMemory 只提取“我喜欢/我是/…”开头的句子（见 tests/conftest.py），
# 所以这里用“我喜欢游泳，每周都去游半小时。”复现游泳场景——真实 mem0 调
# DeepSeek 提取时“我昨天去游泳了，游了半小时”同样可提，这里只验流程。

import time

from conftest import make_fake_store
from memory.outbox import MemoryOutbox

# 游泳场景的复现句：FakeMemory 能提取（“我喜欢”开头），且能过闸门 1
# （不是条件句、不是身份陈述、不是玩笑句）。
_SWIM_USER = "我喜欢游泳，每周都去游半小时。"
_SWIM_STORED = "我喜欢游泳，每周都去游半小时"  # 假提取会把句末“。”去掉


def _round(user, assistant="记下了。"):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _drain(store, timeout=10):
    """只等后台 outbox 把这轮消费完（不发下一轮、不调 flush_pending）。"""
    deadline = time.monotonic() + timeout
    while store._outbox.pending_count() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert store._outbox.pending_count() == 0, "后台 outbox 在 10 秒内没消费完这轮"


def test_single_normal_turn_is_extracted_without_next_turn(tmp_path):
    """一轮正常聊天（不走“记住”命令）后，不等下一轮、不调 flush，
    提取就要跑、记忆就要进库——靠闲置定时器提交（生产默认 60 秒，
    这里改小成 0.2 秒，不用真等）。"""
    store = make_fake_store(tmp_path)
    fake = store._memory
    # 闲置定时器改小：生产默认 60 秒，这里 0.2 秒就提交，不用真等。
    store._pending_idle_seconds = 0.2
    # 落盘目录走隔离的 tmp 目录，别写进真实仓库的 memory/outbox_pending。
    store._outbox = MemoryOutbox(store, pending_dir=tmp_path / "outbox_pending")
    store._outbox.start()
    try:
        result = store.save_turn(_round(_SWIM_USER), turn_id="turn-swim-1")
        assert result.get("queued") is True
        assert result.get("turn_id") == "turn-swim-1"
        _drain(store)
        # 闲置定时器到点后提交提取：等一会儿（最多 5 秒），不等下一轮。
        deadline = time.monotonic() + 5
        while _SWIM_STORED not in fake.all_texts() and time.monotonic() < deadline:
            time.sleep(0.05)
        # 提取确实被触发：假 mem0 的 add 至少被调过一次。
        assert fake.add_calls, "后台线程消费完这轮后，提取一次都没调（add 没被调过）"
        # 记忆确实进了库。
        assert _SWIM_STORED in fake.all_texts()
    finally:
        store._cancel_pending_timer()
        store._outbox.stop()


def test_single_normal_turn_stays_durable_until_stored(tmp_path):
    """上面那轮在没真正落库之前不能丢：要么已经进了库，要么落盘文件还在、
    重启能捞回来。当前代码两头都不占——add_turn 只返回待定就删了落盘文件，
    记忆又没进库，进程一死这轮就彻底没了。"""
    store = make_fake_store(tmp_path)
    fake = store._memory
    pending_dir = tmp_path / "outbox_pending"
    store._outbox = MemoryOutbox(store, pending_dir=pending_dir)
    store._outbox.start()
    try:
        store.save_turn(_round(_SWIM_USER), turn_id="turn-swim-1")
        _drain(store)
        stored = _SWIM_STORED in fake.all_texts()
        pending_files = list(pending_dir.glob("*.json"))
        assert stored or pending_files, (
            "这轮既没进库、落盘文件也没了——重启后捞不回来"
        )
    finally:
        store._cancel_pending_timer()
        store._outbox.stop()
