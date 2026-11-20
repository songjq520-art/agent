# tests/test_model_router.py（整理记忆的大模型路由：主力 + 备选 + 预算提醒）
# 测 memory/model_router.py。离线测：provider 全是假对象，不联网、不花钱。
# 关键规矩（2026-11-19）：
#   - 主力 deepseek-v4-pro 连续 3 次调不通才切 flash；
#   - 每月 3 美元上限超了只提醒、不断服务（BOOS 定），继续正常调；
#   - 只有 BOOS 亲口说停才停（paused_by_user）。

import json

import pytest

from memory.model_router import (
    FAILURES_TO_FALLBACK,
    FALLBACK_MODEL,
    ORGANIZE_MONTHLY_CAP_USD,
    PRIMARY_MODEL,
    OrganizingModelRouter,
)


class FakeProvider:
    """假 provider：按剧本回（"ok", 文本, 花费）或抛错（"fail", 原因）。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat_messages(self, messages, ledger="chat", thinking=None, max_tokens=None):
        self.calls.append(
            {"messages": messages, "ledger": ledger, "thinking": thinking,
             "max_tokens": max_tokens}
        )
        if not self.script:
            raise RuntimeError("剧本用完了")
        entry = self.script.pop(0)
        if entry[0] == "fail":
            raise RuntimeError(entry[1] if len(entry) > 1 else "boom")
        text = entry[1]
        cost = entry[2] if len(entry) > 2 else 0.001
        return {"text": text, "cost_usd": cost, "model": "fake-model"}


class FakeFactory:
    """假工厂：按模型名给 provider，记下每次要的是哪个模型。"""

    def __init__(self):
        self.by_model = {}
        self.requested = []

    def add(self, model, provider):
        self.by_model[model] = provider
        return self

    def __call__(self, model_name, budget_tracker=None):
        self.requested.append(model_name)
        return self.by_model.get(model_name)


def _router(tmp_path, factory):
    return OrganizingModelRouter(
        state_path=str(tmp_path / "router-state.json"),
        provider_factory=factory,
    )


def test_primary_success_records_cost_and_tier(tmp_path):
    factory = FakeFactory().add(
        PRIMARY_MODEL, FakeProvider([("ok", '{"tone": "serious"}', 0.002)])
    )
    router = _router(tmp_path, factory)
    out = router.judge("sys", "user")
    assert out["text"] == '{"tone": "serious"}'
    assert out["tier"] == "primary"
    assert router.monthly_spent() == pytest.approx(0.002)
    assert router.is_over_budget() is False
    assert router.pending_notice() is None
    # 短判断关思考、不烧推理 Token；记账走 chat 账本。
    call = factory.by_model[PRIMARY_MODEL].calls[0]
    assert call["thinking"] == "disabled"
    assert call["ledger"] == "chat"


def test_three_primary_failures_switch_to_fallback(tmp_path):
    primary = FakeProvider([("fail", "x")] * 3 + [("ok", "p", 0.001)])
    fallback = FakeProvider([("ok", "f", 0.001)] * 5)
    factory = FakeFactory().add(PRIMARY_MODEL, primary).add(FALLBACK_MODEL, fallback)
    router = _router(tmp_path, factory)
    # 前两次：主力不通，当场用备选顶上，但 tier 还没切（不到 3 次）。
    assert router.judge("s", "u")["tier"] == "fallback"
    assert router.status()["tier"] == "primary"
    assert router.judge("s", "u")["tier"] == "fallback"
    assert router.status()["tier"] == "primary"
    # 第三次：连续 3 次不通，切到备选。
    assert router.judge("s", "u")["tier"] == "fallback"
    status = router.status()
    assert status["tier"] == "fallback"
    assert status["tier_model"] == FALLBACK_MODEL
    assert len(status["switch_log"]) == 1
    assert status["switch_log"][0]["to"] == "fallback"
    # 切过去之后，不再探主力，直接用备选。
    factory.requested.clear()
    router.judge("s", "u")
    assert factory.requested == [FALLBACK_MODEL]


def test_fallback_failure_means_local_degrade(tmp_path):
    factory = FakeFactory().add(
        PRIMARY_MODEL, FakeProvider([("fail", "x")] * 10)
    ).add(
        FALLBACK_MODEL, FakeProvider([("fail", "y")] * 10)
    )
    router = _router(tmp_path, factory)
    assert router.judge("s", "u") is None
    assert router.judge("s", "u") is None


def test_over_budget_keeps_calling_and_notifies(tmp_path):
    big = ORGANIZE_MONTHLY_CAP_USD + 1.0
    factory = FakeFactory().add(
        PRIMARY_MODEL,
        FakeProvider([("ok", "a", big), ("ok", "b", 0.001), ("ok", "c", 0.001)]),
    )
    router = _router(tmp_path, factory)
    out = router.judge("s", "u")
    assert out is not None  # 超了也不停调
    assert router.is_over_budget() is True
    notice = router.pending_notice()
    assert notice is not None
    assert notice["cap_usd"] == ORGANIZE_MONTHLY_CAP_USD
    assert notice["spent_usd"] == pytest.approx(big)
    assert "3" in notice["message"] and "没停" in notice["message"]
    # 超了之后继续调，照常返回。
    assert router.judge("s", "u")["text"] == "b"
    assert router.judge("s", "u")["text"] == "c"
    # BOOS 认领后，同一月里不再提醒；状态里还留着超预算的事实。
    router.acknowledge_notice()
    assert router.pending_notice() is None
    assert router.is_over_budget() is True
    assert router.status()["over_budget"] is True


def test_paused_by_user_stops_calls(tmp_path):
    provider = FakeProvider([("ok", "a", 0.001)])
    factory = FakeFactory().add(PRIMARY_MODEL, provider)
    router = _router(tmp_path, factory)
    router.set_paused_by_user(True)
    assert router.judge("s", "u") is None
    assert provider.calls == []
    router.set_paused_by_user(False)
    assert router.judge("s", "u")["text"] == "a"


def test_probe_primary_switches_back(tmp_path):
    primary = FakeProvider([("fail", "x")] * 3 + [("ok", "back", 0.001)])
    fallback = FakeProvider([("ok", "f", 0.001)] * 5)
    factory = FakeFactory().add(PRIMARY_MODEL, primary).add(FALLBACK_MODEL, fallback)
    router = _router(tmp_path, factory)
    for _ in range(3):
        router.judge("s", "u")
    assert router.status()["tier"] == "fallback"
    # 下一批任务开头探一次主力：通了就切回主力。
    out = router.judge("s", "u", probe_primary=True)
    assert out["tier"] == "primary"
    assert out["text"] == "back"
    assert router.status()["tier"] == "primary"
    assert router.status()["switch_log"][-1]["to"] == "primary"


def test_probe_primary_failure_stays_on_fallback(tmp_path):
    primary = FakeProvider([("fail", "x")] * 10)
    fallback = FakeProvider([("ok", "f", 0.001)] * 5)
    factory = FakeFactory().add(PRIMARY_MODEL, primary).add(FALLBACK_MODEL, fallback)
    router = _router(tmp_path, factory)
    for _ in range(3):
        router.judge("s", "u")
    out = router.judge("s", "u", probe_primary=True)
    assert out["text"] == "f"
    assert router.status()["tier"] == "fallback"


def test_no_provider_means_local_degrade(tmp_path):
    router = _router(tmp_path, FakeFactory())
    assert router.judge("s", "u") is None
    assert router.monthly_spent() == pytest.approx(0.0)


def test_cost_none_does_not_break_accounting(tmp_path):
    factory = FakeFactory().add(
        PRIMARY_MODEL, FakeProvider([("ok", "a", None), ("ok", "b", 0.002)])
    )
    router = _router(tmp_path, factory)
    assert router.judge("s", "u")["text"] == "a"
    assert router.monthly_spent() == pytest.approx(0.0)
    assert router.judge("s", "u")["text"] == "b"
    assert router.monthly_spent() == pytest.approx(0.002)


def test_month_rollover_resets_spending(tmp_path):
    state_path = tmp_path / "router-state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "month": "2020-01",
                "spent_usd": 99.0,
                "over_budget": True,
                "first_over_at": "2020-01-15T00:00:00Z",
                "notice_acknowledged": False,
                "paused_by_user": False,
                "tier": "fallback",
                "consecutive_failures": 2,
                "switch_log": [],
            }
        ),
        encoding="utf-8",
    )
    router = OrganizingModelRouter(
        state_path=str(state_path), provider_factory=FakeFactory()
    )
    assert router.monthly_spent() == pytest.approx(0.0)
    assert router.is_over_budget() is False
    assert router.pending_notice() is None
    assert router.status()["tier"] == "primary"


def test_corrupt_state_file_starts_fresh(tmp_path):
    state_path = tmp_path / "router-state.json"
    state_path.write_text("{not json", encoding="utf-8")
    router = OrganizingModelRouter(
        state_path=str(state_path), provider_factory=FakeFactory()
    )
    assert router.monthly_spent() == pytest.approx(0.0)
    assert router.status()["tier"] == "primary"
    assert router.judge("s", "u") is None  # 没 provider，本地降级，不崩


def test_state_survives_reload(tmp_path):
    factory = FakeFactory().add(
        PRIMARY_MODEL, FakeProvider([("ok", "a", 0.5), ("ok", "b", 0.5)])
    )
    router = _router(tmp_path, factory)
    router.judge("s", "u")
    again = _router(tmp_path, factory)
    assert again.monthly_spent() == pytest.approx(0.5)
    assert again.judge("s", "u")["text"] == "b"
    assert again.monthly_spent() == pytest.approx(1.0)
