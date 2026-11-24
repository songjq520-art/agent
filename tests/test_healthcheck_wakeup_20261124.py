# tests/test_healthcheck_wakeup_20261124.py（修①：体检定时唤醒，2026-11-24）
#
# 原来服务内只有「断网恢复」一个触发点：干净开机、长期不断网就
# 没人去问 healthcheck.weekly() 到期没，月度体检不会自己跑。
# 修①在 web.server 里补 _HealthcheckWakeup：断网恢复回调与进程内
# 定时器共用同一个唤醒器、同一把锁；定时器启动先查一次（错过
# 补跑），之后每隔设定间隔再查。未设 SHIER_HEALTHCHECK_CONFIG
# 时唤醒器不存在，服务行为与之前完全一致。
# 全部离线：run_weekly 注入假实现，不真跑体检、不调模型。

import json
import threading
import time

import web.server as server_module
from web.server import (
    _healthcheck_recovery_callback,
    _healthcheck_wakeup,
    start_memory_maintenance,
)


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _set_config(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SHIER_HEALTHCHECK_CONFIG",
        json.dumps({
            "archive": str(tmp_path / "hc-archive"),
            "base_url": "http://127.0.0.1:8000",
        }),
    )


def test_no_config_no_wakeup(monkeypatch, tmp_path):
    """(c) 未设 SHIER_HEALTHCHECK_CONFIG：不装配唤醒器，老入口也返回 None。"""
    monkeypatch.delenv("SHIER_HEALTHCHECK_CONFIG", raising=False)
    assert _healthcheck_wakeup(object(), tmp_path) is None
    assert _healthcheck_recovery_callback(object(), tmp_path) is None


def test_bad_config_no_wakeup(monkeypatch, tmp_path):
    """配置解析失败同样不启用（与并入时的门控语义一致）。"""
    monkeypatch.setenv("SHIER_HEALTHCHECK_CONFIG", "{not json")
    assert _healthcheck_wakeup(object(), tmp_path) is None
    assert _healthcheck_recovery_callback(object(), tmp_path) is None


def test_timer_triggers_at_startup_and_on_interval(monkeypatch, tmp_path):
    """(a) 定时器：启动立刻触发一次，之后按短间隔持续触发；停了就不再触发。"""
    _set_config(monkeypatch, tmp_path)
    calls = []
    wakeup = _healthcheck_wakeup(
        object(), tmp_path,
        run_weekly=lambda: calls.append(1),
        timer_interval_seconds=0.05,
    )
    assert wakeup is not None
    wakeup.start_timer()
    try:
        # 启动触发 + 至少一次间隔触发
        assert _wait_until(lambda: len(calls) >= 2)
    finally:
        wakeup.stop_timer()
    time.sleep(0.2)  # 让已在途的触发收尾
    settled = len(calls)
    time.sleep(0.25)
    assert len(calls) == settled, "stop_timer 后不许再触发"


def test_triggers_share_one_lock(monkeypatch, tmp_path):
    """(b) 防重入：一体检在跑时，恢复回调与定时器触发都被跳过；跑完可再触发。"""
    _set_config(monkeypatch, tmp_path)
    entered = []
    release = threading.Event()
    inside = threading.Event()

    def blocking_run():
        entered.append(1)
        inside.set()
        release.wait(timeout=10)

    wakeup = _healthcheck_wakeup(
        object(), tmp_path,
        run_weekly=blocking_run,
        timer_interval_seconds=3600,
    )
    wakeup.on_network_recovered()
    assert _wait_until(inside.is_set)
    # 第一个还堵在里面：两个触发点各再触发一次，都必须被锁挡住
    wakeup.trigger("timer")
    wakeup.on_network_recovered()
    time.sleep(0.3)
    assert len(entered) == 1, "并发触发不许跑出第二份体检"
    release.set()
    assert _wait_until(lambda: not wakeup._lock.locked())
    wakeup.trigger("timer")
    assert _wait_until(lambda: len(entered) == 2)


class _FakeStore:
    def __init__(self, ledger_path):
        self.ledger_path = ledger_path

    def sweep_decay(self):
        return {}

    def run_self_check(self):
        return {}


def test_start_memory_maintenance_wires_wakeup(monkeypatch, tmp_path):
    """接线：装配出唤醒器时，回调挂上调度器、定时器被启动。"""
    events = []

    class _Recorder:
        def on_network_recovered(self):
            events.append("cb")

        def start_timer(self):
            events.append("timer")

    recorder = _Recorder()
    monkeypatch.setattr(
        server_module, "_healthcheck_wakeup",
        lambda store, memory_dir: recorder)
    store = _FakeStore(str(tmp_path / "memory" / "memory_events.jsonl"))
    sched = start_memory_maintenance(store)
    try:
        assert events == ["timer"]
        assert sched._on_network_recovered is not None
    finally:
        sched.stop()
        if sched._thread is not None:
            sched._thread.join(timeout=10)


def test_start_memory_maintenance_without_wakeup_unchanged(monkeypatch, tmp_path):
    """接线：没有唤醒器（未配置）时，回调为 None、不启动定时器，行为不变。"""
    monkeypatch.setattr(
        server_module, "_healthcheck_wakeup",
        lambda store, memory_dir: None)
    store = _FakeStore(str(tmp_path / "memory" / "memory_events.jsonl"))
    sched = start_memory_maintenance(store)
    try:
        assert sched._on_network_recovered is None
    finally:
        sched.stop()
        if sched._thread is not None:
            sched._thread.join(timeout=10)
