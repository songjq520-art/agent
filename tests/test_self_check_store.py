# tests/test_self_check_store.py（自检接进 store：run_self_check +
# store_self_check 适配器 + make_model_judge）
# 离线测试：FakeMemory + tmp_path，不联网、不调真实大模型。
# run_self_check 里的大模型路由在没 Key 的环境里建不起来/判不出来，
# 第 9 项走 skipped——断言的就是这个（不依赖真模型）。

from memory import store_self_check as _store_self_check
from tests.conftest import make_fake_store


def _store_with_memories(tmp_path):
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    stored = store.add_memory_text("我每天定投 5 块，开玩笑的。")
    return store, stored["results"][0]["id"]


# ---------- run_self_check 端到端 ----------


def test_run_self_check_report_shape(tmp_path, monkeypatch):
    store, joke_id = _store_with_memories(tmp_path)
    # 强制走“没大模型”那条路：笔记本的 .env 里有真 Key，不拦的话
    # 第 9 项会真去调模型、结果随模型走，测试就不确定了。
    monkeypatch.setattr(store, "organizing_router", lambda: None)
    report = store.run_self_check()
    assert set(report.keys()) == {
        "generated_at",
        "items",
        "most_likely_wrong_5",
        "replaced_last_month",
        "tagged_tone_memories",
        "organizing_budget",
        "summary",
    }
    assert set(report["items"].keys()) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    for item in report["items"].values():
        assert set(item.keys()) == {"name", "status", "findings", "actions"}
        assert item["status"] in ("pass", "warn", "fail", "skipped")
    # 第 9 项：没大模型路由，跳过
    assert report["items"][9]["status"] == "skipped"
    # 第 10 项：没大模型路由，跳过
    assert report["items"][10]["status"] == "skipped"
    # 第 5 项：备份演练（建、写、读、验）在临时目录跑，通过
    assert report["items"][5]["status"] == "pass"
    # 打标的玩笑记忆列在报告里
    tagged = report["tagged_tone_memories"]
    assert len(tagged) == 1
    assert tagged[0]["id"] == joke_id
    assert tagged[0]["tone"] == "joke"
    # 摘要是中文一句话
    assert isinstance(report["summary"], str) and report["summary"]


def test_run_self_check_with_fake_router_judges(tmp_path, monkeypatch):
    # 有模型时第 9 项真抽查：假路由回 accurate=true → pass，
    # 每条 findings 里都有结论（不是“判不出来”）。
    store, _joke_id = _store_with_memories(tmp_path)

    class _FakeRouter:
        def judge(self, system_prompt, user_prompt, max_tokens=500,
                  purpose=None, **kwargs):
            return {
                "text": '{"accurate": true, "evidence": "跟出处对得上"}',
                "cost_usd": 0.0,
                "model": "fake",
            }

        def status(self):
            return {"over_budget": False, "spent_usd": 0.0, "cap_usd": 3.0}

    monkeypatch.setattr(store, "organizing_router", lambda: _FakeRouter())
    report = store.run_self_check()
    check = report["items"][9]
    assert check["status"] == "pass"
    assert len(check["findings"]) == 2
    assert all(entry.get("accurate") is True for entry in check["findings"])


def test_run_self_check_clean_store_passes_most_items(tmp_path, monkeypatch):
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    monkeypatch.setattr(store, "organizing_router", lambda: None)
    report = store.run_self_check()
    # 干净的库：过期、冲突、孤儿、晋升计数这几项都该通过
    for num in (2, 3, 4, 7):
        assert report["items"][num]["status"] == "pass", num
    # 索引那项在 FakeMemory 下是 warn：history.db 里没有记录，
    # chroma 里有 1 条——两边对不上，报告要写出来（不是静默通过）
    assert report["items"][6]["status"] == "warn"
    assert report["items"][6]["findings"]


# ---------- 适配器：被取代事件 ----------


def test_adapter_supersede_events_since(tmp_path):
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    out = store.correct_memory("我喜欢喝咖啡。", "我喜欢喝茶。")
    assert out.get("status") == "updated"
    adapter = _store_self_check.build_self_check_adapter(store)
    events = adapter.supersede_events_since(30)
    assert len(events) == 1
    assert "咖啡" in events[0]["text"] or "茶" in events[0]["text"]
    # 30 天前没有事件
    assert adapter.supersede_events_since(0) == []


def test_adapter_replay_promotion_counts_clean(tmp_path):
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    adapter = _store_self_check.build_self_check_adapter(store)
    result = adapter.replay_promotion_counts()
    assert result["checked"] == 1
    assert result["corrections"] == []


def test_adapter_archive_items(tmp_path):
    store = make_fake_store(tmp_path)
    stored = store.add_memory_text("我喜欢喝咖啡。")
    mem_id = stored["results"][0]["id"]
    adapter = _store_self_check.build_self_check_adapter(store)
    assert adapter.archive_items() == []
    # 直接把 decay_state 打成 dormant（月扫就是这么标的）
    meta = store._raw_metadata_for(mem_id)
    meta["decay_state"] = "dormant"
    store._memory.update(memory_id=mem_id, metadata=meta)
    archived = adapter.archive_items()
    assert len(archived) == 1
    assert archived[0]["id"] == mem_id


def test_adapter_verify_backup_drill(tmp_path):
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    adapter = _store_self_check.build_self_check_adapter(store)
    result = adapter.verify_backup()
    assert result["ok"] is True
    assert "通过" in result["detail"]


# ---------- make_model_judge ----------


class _ScriptRouter:
    def __init__(self, text=None, exc=None):
        self.text = text
        self.exc = exc

    def judge(self, system_prompt, user_prompt, max_tokens=500, purpose=None):
        if self.exc is not None:
            raise self.exc
        return {"text": self.text, "cost_usd": 0.0, "model": "fake"}


def test_make_model_judge_parses_json():
    judge = _store_self_check.make_model_judge(
        _ScriptRouter(text='{"accurate": true, "evidence": "出处一致"}')
    )
    out = judge("sys", "user")
    assert out == {"accurate": True, "evidence": "出处一致"}


def test_make_model_judge_parses_fenced_json():
    judge = _store_self_check.make_model_judge(
        _ScriptRouter(text='```json\n{"accurate": false, "evidence": "数字对不上"}\n```')
    )
    out = judge("sys", "user")
    assert out == {"accurate": False, "evidence": "数字对不上"}


def test_make_model_judge_invalid_returns_none():
    # 不是 JSON / accurate 不是布尔值 / 路由抛异常 → 都返回 None（判不出来）
    for router in (
        _ScriptRouter(text="这条记忆没问题"),
        _ScriptRouter(text='{"accurate": "yes", "evidence": "x"}'),
        _ScriptRouter(exc=RuntimeError("模型挂了")),
        _ScriptRouter(text=None),
    ):
        judge = _store_self_check.make_model_judge(router)
        assert judge("sys", "user") is None
