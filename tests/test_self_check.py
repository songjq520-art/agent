# tests/test_self_check.py（十二每月自检模块）
# 测九项检查 + 顶层字段 + 幂等。全部离线：FakeAdapter 实现
# self_check 的适配器约定，不碰真实 mem0、不联网、不产生费用。
# tone.py / content_check.py 还没落地时，模块按约定字面值顶上，
# rank_most_likely_wrong 用 monkeypatch 顶一个假的测。

import copy
from datetime import datetime, timedelta, timezone

from memory import self_check
from memory.self_check import run_self_check

NOW = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)


def _meta(**over):
    """一份 _normalize_metadata 形状的 metadata（只填自检要看的字段）。"""
    base = {
        "source": "user",
        "confidence": 1.0,
        "user_confirmed": True,
        "slot": None,
        "is_rule": False,
        "turn_id": "turn-1",
        "decay_kind": "stable",
        "decay_state": "verbatim",
        "decay_delete_due": False,
        "decay_seen_dates": ["2026-11-19"],
        "decay_seen_sessions": ["turn-1"],
        "last_used_at": "2026-11-19T12:00:00Z",
        "tone": "serious",
        "tone_evidence": None,
        "tone_source": None,
        "voided": False,
        "superseded": False,
        "needs_review": False,
        "conflict_for_review": False,
        "grounding_flag": None,
    }
    base.update(over)
    return base


def _item(item_id, text, **meta_over):
    return {"id": item_id, "text": text, "metadata": _meta(**meta_over)}


def _days_ago(days):
    return (NOW - timedelta(days=days)).isoformat().replace("+00:00", "Z")


class FakeAdapter:
    """实现 self_check 适配器约定的假对象（离线）。"""

    def __init__(self, items=None, superseded=None, backup=None,
                 index=None, replay=None, archive=None, embeddings=None):
        self._items = copy.deepcopy(items or [])
        self._superseded = list(superseded or [])
        self._backup = dict(backup if backup is not None
                            else {"ok": True, "detail": "恢复演练通过。"})
        self._index = dict(index if index is not None
                           else {"sqlite": 0, "chroma": 0,
                                 "mismatched_ids": []})
        self._replay = dict(replay if replay is not None
                            else {"checked": 0, "corrections": []})
        self._archive = list(archive or [])
        self._embeddings = dict(embeddings or {})

    def get_all_items(self):
        return copy.deepcopy(self._items)

    def supersede_events_since(self, days):
        assert days == 30  # 自检只问最近 30 天被取代的
        return list(self._superseded)

    def verify_backup(self):
        return dict(self._backup)

    def index_counts(self):
        return dict(self._index)

    def replay_promotion_counts(self):
        return {
            "checked": self._replay.get("checked"),
            "corrections": list(self._replay.get("corrections") or []),
        }

    def archive_items(self):
        return list(self._archive)

    def embed(self, text):
        return self._embeddings.get(text)


def _clean_adapter(items=None, **kwargs):
    """一份什么问题都没有的适配器（第 1 项没向量会 skipped，不算问题）。"""
    items = items if items is not None else [_item("m1", "我是 BOOS。")]
    return FakeAdapter(items=items, **kwargs)


def _clean_report(**kwargs):
    return run_self_check(_clean_adapter(), now=NOW, **kwargs)


# ---------- 第 1 项：重复记忆 ----------

def test_duplicates_found_with_shared_keywords():
    items = [
        _item("m1", "我喜欢喝咖啡", user_confirmed=False),
        _item("m2", "我喜欢喝咖啡，每天一杯", user_confirmed=False),
        _item("m3", "我喜欢跑步", user_confirmed=False),
    ]
    adapter = FakeAdapter(
        items=items,
        embeddings={
            "我喜欢喝咖啡": [1.0, 0.0],
            "我喜欢喝咖啡，每天一杯": [1.0, 0.0],
            "我喜欢跑步": [0.0, 1.0],
        },
    )
    report = run_self_check(adapter, now=NOW)
    check = report["items"][1]
    assert check["status"] == "warn"
    assert len(check["findings"]) == 1
    assert check["findings"][0]["id"] == "m1"
    assert check["findings"][0]["other_id"] == "m2"
    assert check["findings"][0]["similarity"] > 0.95
    assert "咖啡" in check["findings"][0]["shared_keywords"]
    assert check["actions"]  # 非核心：建议人工核对是否合并
    assert "合并" in check["actions"][0]


def test_duplicates_core_only_reported_no_merge_action():
    items = [
        _item("m1", "我是 BOOS"),
        _item("m2", "我是 BOOS，叫我 BOOS"),
    ]
    adapter = FakeAdapter(
        items=items,
        embeddings={"我是 BOOS": [1.0, 0.0], "我是 BOOS，叫我 BOOS": [1.0, 0.0]},
    )
    report = run_self_check(adapter, now=NOW)
    check = report["items"][1]
    assert check["status"] == "warn"
    assert len(check["findings"]) == 1
    assert "核心记忆" in check["findings"][0]["reason"]
    assert "不建议合并" in check["findings"][0]["reason"]
    assert check["actions"] == []  # 核心记忆只报告，不建议合并


def test_duplicates_skipped_without_embeddings():
    report = _clean_report()
    check = report["items"][1]
    assert check["status"] == "skipped"
    assert "没有向量模型" in check["findings"][0]["reason"]


# ---------- 第 2 项：过期没沉底 ----------

def test_expired_not_dormant_and_delete_due():
    items = [
        _item("m1", "上个月去了一次西安", user_confirmed=False,
              decay_kind="episodic", last_used_at=_days_ago(40)),
        _item("m2", "去年去了一次西安", user_confirmed=False,
              decay_kind="episodic", last_used_at=_days_ago(200)),
        _item("m3", "我是 BOOS", decay_kind="stable",
              last_used_at=_days_ago(40)),
    ]
    report = run_self_check(_clean_adapter(items=items), now=NOW)
    check = report["items"][2]
    assert check["status"] == "warn"
    by_id = {entry["id"]: entry for entry in check["findings"]}
    assert "m1" in by_id and "沉底" in by_id["m1"]["reason"]
    assert "m2" in by_id and "待删" in by_id["m2"]["reason"]
    assert "m3" not in by_id  # 稳定事实 40 天不沉底
    assert any("打标沉底" in action for action in check["actions"])
    assert any("只打标" in action for action in check["actions"])
    assert not any("直接删" in action and "不直接删" not in action
                   for action in check["actions"])


def test_expired_already_tagged_is_pass():
    items = [
        _item("m1", "上个月去了一次西安", user_confirmed=False,
              decay_kind="episodic", last_used_at=_days_ago(40),
              decay_state="dormant"),
    ]
    report = run_self_check(_clean_adapter(items=items), now=NOW)
    assert report["items"][2]["status"] == "pass"


# ---------- 第 3 项：冲突未决 ----------

def test_conflicts_unresolved_and_replaced_last_month():
    items = [
        _item("m1", "我喜欢喝咖啡", conflict_for_review=True),
        _item("m2", "我喜欢喝拿铁", needs_review=True),
        _item("m3", "我是 BOOS"),
    ]
    superseded = [
        {"id": "old1", "text": "我喜欢喝茶", "at": "2026-11-10T10:00:00Z"},
        {"id": "old2", "text": "我叫小宋", "at": "2026-11-05T10:00:00Z"},
    ]
    report = run_self_check(
        _clean_adapter(items=items, superseded=superseded), now=NOW
    )
    check = report["items"][3]
    assert check["status"] == "warn"
    assert {entry["id"] for entry in check["findings"]} == {"m1", "m2"}
    assert report["replaced_last_month"] == superseded


# ---------- 第 4 项：孤儿记忆 ----------

def test_orphans_reported_only_no_delete_suggestion():
    items = [
        _item("m1", "不知道哪来的话", source="unconfirmed", turn_id=None,
              user_confirmed=False),
        _item("m2", "模型提取的", source="inferred", turn_id=None,
              user_confirmed=False),
        _item("m3", "我是 BOOS"),
    ]
    report = run_self_check(_clean_adapter(items=items), now=NOW)
    check = report["items"][4]
    assert check["status"] == "warn"
    assert [entry["id"] for entry in check["findings"]] == ["m1"]
    assert "只报告" in check["findings"][0]["reason"]
    assert check["actions"] == []  # 只报告，不建议删


# ---------- 第 5 项：备份能恢复吗 ----------

def test_backup_failure_is_fail_and_blocks_deletion():
    adapter = _clean_adapter(
        backup={"ok": False, "detail": "备份文件损坏，恢复失败。"})
    report = run_self_check(adapter, now=NOW)
    check = report["items"][5]
    assert check["status"] == "fail"
    assert "禁止删除" in check["findings"][0]["reason"]
    assert "禁止删除" in check["actions"][0]
    assert "建议 BOOS 看一下" in report["summary"]


def test_backup_ok_is_pass():
    report = _clean_report()
    assert report["items"][5]["status"] == "pass"


# ---------- 第 6 项：索引有没有坏 ----------

def test_index_mismatch_warns():
    adapter = _clean_adapter(
        index={"sqlite": 10, "chroma": 9, "mismatched_ids": ["m9"]})
    report = run_self_check(adapter, now=NOW)
    check = report["items"][6]
    assert check["status"] == "warn"
    assert check["findings"][0]["sqlite"] == 10
    assert check["findings"][0]["chroma"] == 9
    assert check["findings"][0]["mismatched_ids"] == ["m9"]


def test_index_match_is_pass():
    adapter = _clean_adapter(
        index={"sqlite": 3, "chroma": 3, "mismatched_ids": []})
    report = run_self_check(adapter, now=NOW)
    assert report["items"][6]["status"] == "pass"


# ---------- 第 7 项：晋升计数 ----------

def test_promotion_corrections_listed():
    corrections = [
        {"id": "m1", "field": "decay_seen_days", "old": 1, "new": 3},
    ]
    adapter = _clean_adapter(
        replay={"checked": 5, "corrections": corrections})
    report = run_self_check(adapter, now=NOW)
    check = report["items"][7]
    assert check["status"] == "warn"
    assert check["findings"] == corrections


def test_promotion_no_corrections_is_pass():
    adapter = _clean_adapter(replay={"checked": 5, "corrections": []})
    report = run_self_check(adapter, now=NOW)
    assert report["items"][7]["status"] == "pass"
    assert report["items"][7]["findings"][0]["checked"] == 5


# ---------- 第 8 项：归档一致性 ----------

def test_archive_missing_text_listed():
    items = [_item("m1", "我是 BOOS")]
    archive = [
        {"id": "a1", "text": "我是 BOOS"},
        {"id": "a2", "text": "档案里有、库里没了的话"},
    ]
    report = run_self_check(_clean_adapter(items=items, archive=archive),
                            now=NOW)
    check = report["items"][8]
    assert check["status"] == "warn"
    assert [entry["id"] for entry in check["findings"]] == ["a2"]


# ---------- 第 9 项：模型抽查 ----------

def _judge_factory(results):
    """按记忆 id 返回 canned 结论的假 judge，并记下收到的提示词。"""
    seen = []

    def judge(system_prompt, user_prompt):
        seen.append((system_prompt, user_prompt))
        for item_id, verdict in results.items():
            if f"记忆 id：{item_id}" in user_prompt:
                return verdict
        return None

    judge.seen = seen
    return judge


def test_model_sample_with_judge():
    items = [
        _item("m1", "我是 BOOS"),
        _item("m2", "我喜欢喝咖啡"),
        _item("m3", "我喜欢喝拿铁"),
    ]
    judge = _judge_factory({
        "m1": {"accurate": True, "evidence": "原文“我是 BOOS”跟出处对得上。"},
        "m2": {"accurate": False, "evidence": "原文“我喜欢喝咖啡”跟出处对不上。"},
        "m3": None,  # 判不出来
    })
    report = run_self_check(_clean_adapter(items=items), now=NOW,
                            model_judge=judge, sample_size=10)
    check = report["items"][9]
    assert check["status"] == "warn"  # m2 被判不准
    by_id = {entry["id"]: entry for entry in check["findings"]}
    assert by_id["m1"]["accurate"] is True
    assert "我是 BOOS" in by_id["m1"]["evidence"]  # 证据带原文片段
    assert by_id["m2"]["accurate"] is False
    assert by_id["m3"]["accurate"] is None  # 判不出来记 skipped
    # 提示词问的是“跟出处对不对得上、有没有内部矛盾”
    assert "对不对得上" in judge.seen[0][1]
    assert "内部矛盾" in judge.seen[0][1]
    assert "对不对得上" in judge.seen[0][0] or "内部矛盾" in judge.seen[0][0]


def test_model_sample_without_judge_is_skipped():
    report = _clean_report()
    check = report["items"][9]
    assert check["status"] == "skipped"
    assert "没传 model_judge" in check["findings"][0]["reason"]


def test_model_sample_is_deterministic_by_sorted_id():
    items = [_item(f"m{i}", f"记忆{i}") for i in range(1, 7)]
    judge = _judge_factory({
        f"m{i}": {"accurate": True, "evidence": f"原文“记忆{i}”对得上。"}
        for i in range(1, 7)
    })
    report = run_self_check(_clean_adapter(items=items), now=NOW,
                            model_judge=judge, sample_size=2)
    sampled = [entry["id"] for entry in report["items"][9]["findings"]]
    assert sampled == ["m1", "m4"]  # 按 id 排序后均匀取：6 条取 2 条


# ---------- 顶层字段 ----------

def test_most_likely_wrong_5_and_tone_tags(monkeypatch):
    items = [
        _item("m1", "我喜欢喝咖啡", grounding_flag="unverified"),
        _item("m2", "哈哈我开玩笑的", tone="joke",
              tone_evidence="原文“哈哈”"),
        _item("m3", "测试一下", tone="test", tone_evidence="原文“测试”"),
        _item("m4", "假如我是 BOOS", tone="hypothesis",
              tone_evidence="原文“假如”"),
        _item("m5", "我是 BOOS"),
    ]
    canned = [{"id": "m1", "text": "我喜欢喝咖啡", "score": 0.9,
               "reasons": ["没出处"]}]
    seen = {}

    def fake_rank(rank_items, top_n=5):
        seen["items"] = rank_items
        seen["top_n"] = top_n
        return canned

    monkeypatch.setattr(self_check, "rank_most_likely_wrong", fake_rank)
    report = run_self_check(_clean_adapter(items=items), now=NOW)
    assert report["most_likely_wrong_5"] == canned
    assert seen["top_n"] == 5
    # grounding_flag 从 metadata 里取出来传给 rank
    by_id = {entry["id"]: entry for entry in seen["items"]}
    assert by_id["m1"]["grounding_flag"] == "unverified"
    # 语气打标的记忆单列出来（id、text、tone、tone_evidence）
    tagged = report["tagged_tone_memories"]
    assert [entry["id"] for entry in tagged] == ["m2", "m3", "m4"]
    assert tagged[0]["tone"] == "joke"
    assert tagged[0]["tone_evidence"] == "原文“哈哈”"
    assert tagged[1]["tone"] == "test"
    assert tagged[2]["tone"] == "hypothesis"


def test_organizing_budget_and_over_budget_reminder():
    router_status = {"over_budget": True, "spent": 5, "limit": 3}
    report = run_self_check(_clean_adapter(), now=NOW,
                            router_status=router_status)
    assert report["organizing_budget"] == router_status
    assert "超预算" in report["summary"]


def test_organizing_budget_defaults_empty():
    report = _clean_report()
    assert report["organizing_budget"] == {}
    assert "超预算" not in report["summary"]


def test_clean_report_summary_says_no_need_for_boos():
    report = _clean_report()
    assert "10 项" in report["summary"]
    assert "不用 BOOS 看" in report["summary"]
    assert report["items"][1]["status"] == "skipped"  # 没向量，第 1 项跳过
    assert report["items"][9]["status"] == "skipped"  # 没 judge，第 9 项跳过
    assert report["items"][10]["status"] == "skipped"  # 没 judge，第 10 项跳过


# ---------- 幂等 ----------

def test_idempotent_same_adapter_twice():
    items = [
        _item("m1", "我喜欢喝咖啡", user_confirmed=False),
        _item("m2", "我喜欢喝咖啡，每天一杯", user_confirmed=False),
        _item("m3", "上个月去了一次西安", user_confirmed=False,
              decay_kind="episodic", last_used_at=_days_ago(40)),
        _item("m4", "不知道哪来的话", source="unconfirmed", turn_id=None,
              user_confirmed=False),
        _item("m5", "哈哈我开玩笑的", tone="joke",
              tone_evidence="原文“哈哈”"),
    ]
    judge = _judge_factory({
        "m1": {"accurate": True, "evidence": "原文“我喜欢喝咖啡”对得上。"},
        "m2": {"accurate": True, "evidence": "原文“我喜欢喝咖啡，每天一杯”对得上。"},
        "m3": {"accurate": False, "evidence": "原文“上个月去了一次西安”对不上。"},
        "m4": None,
        "m5": {"accurate": True, "evidence": "原文“哈哈我开玩笑的”对得上。"},
    })
    adapter = FakeAdapter(
        items=items,
        superseded=[{"id": "old1", "text": "旧话", "at": "2026-11-10T10:00:00Z"}],
        index={"sqlite": 5, "chroma": 4, "mismatched_ids": ["m9"]},
        replay={"checked": 3, "corrections": [
            {"id": "m1", "field": "decay_seen_days", "old": 1, "new": 2}]},
        archive=[{"id": "a1", "text": "档案里有、库里没了的话"}],
        embeddings={
            "我喜欢喝咖啡": [1.0, 0.0],
            "我喜欢喝咖啡，每天一杯": [1.0, 0.0],
        },
    )
    first = run_self_check(adapter, now=NOW, model_judge=judge,
                           sample_size=10,
                           router_status={"over_budget": False})
    second = run_self_check(adapter, now=NOW, model_judge=judge,
                            sample_size=10,
                            router_status={"over_budget": False})
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second
