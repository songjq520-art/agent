# tests/test_semantic_judge.py（混合语义判断回归测试，2026-11-22）
# 只测 memory/semantic_judge.py + memory/sqlite_store.py 的接线；
# 全部离线（向量打桩 + mock 判官，不加载 bge 模型、不调真 API、不花钱）。
# 依据：BOOS 2026-11-22 定——简单的程序自己判，难的问大模型；
#   主控是十二自己（本地程序分流），大模型是判官、不是主控；
#   判官模型能换，别写死 DeepSeek。
#
# 覆盖：
#   1. 程序判不定时调模型、模型说取代就取代（饮品偏好，酸奶→豆奶案）
#   2. 模型说不取代就两条并存，不崩
#   3. 模型断了降级走程序：新事实照常存、旧的不动、不崩
#   4. 程序能判定的路（住处）永远不调模型；换模型配置时分发跟配置走
#   5. 纠正候选多条分不清（ambiguous）时判官挑一条
#   6. 判官回的格式容错（markdown 包裹能解析，垃圾回不崩）
#   7. 工具直存路（add_memory_text）也问判官；没候选时不问模型

import json

import pytest

from memory import semantic_judge as sj
from memory.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


class FakeJudge:
    """mock 判官：按设好的 verdict 回 JSON，或直接抛错模拟模型断了。

    provider_name / model 属性供分发层认名字；calls 记下每次收到的消息，
    断言“调没调模型、调了几次”就看它。
    """

    def __init__(self, verdict=None, fail=False):
        self.verdict = verdict
        self.fail = fail
        self.calls = []
        self.provider_name = "deepseek"
        self.model = "deepseek-flash"

    def chat_messages(self, messages, ledger="chat", thinking=None,
                      max_tokens=None):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("API 断了")
        return {
            "text": json.dumps(self.verdict, ensure_ascii=False),
            "usage": {"prompt_tokens": 400, "completion_tokens": 80},
            "cost_usd": None,
            "model": "deepseek-flash",
        }


def _judge_store(tmp_path, monkeypatch, judge):
    """带 mock 判官的 store。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"),
                       judge_provider=judge)


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _save(store, text, turn_id):
    return store.save_turn(
        [{"role": "user", "content": text}], turn_id=turn_id
    )


# ---------- 1. 模型说取代就取代 ----------


def test_judge_supersedes_drink_preference(tmp_path, monkeypatch):
    """酸奶→豆奶案：程序推不出槽位，问判官；判官说取代酸奶，就取代。

    回执 correction 的 route 是 "model"，带判官的 provider/model/成本。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢喝酸奶", "t1")
    assert judge.calls == []  # 第一条：库里没同类偏好，不问模型
    orange_id = store.get_all()[0]["id"]
    judge.verdict = {
        "slot": "喜欢喝的饮品", "is_correction": True,
        "target_id": orange_id, "should_supersede": True,
        "reason": "同是喜欢喝的饮品，新说法取代旧说法",
    }
    receipt = _save(store, "我现在喜欢喝豆奶了", "t2")
    assert len(judge.calls) == 1  # 只问了一次
    assert _is_superseded(store, "我喜欢喝酸奶") is True
    assert "我现在喜欢喝豆奶了" in _live_texts(store)
    assert "我喜欢喝酸奶" not in _live_texts(store)
    assert len([t for t in _live_texts(store) if "喝" in t]) == 1
    correction = receipt.get("correction") or {}
    assert correction.get("route") == "model"
    assert correction.get("status") == "updated"
    assert correction.get("judge", {}).get("provider") == "deepseek"
    assert correction.get("judge", {}).get("model") == "deepseek-flash"
    assert correction.get("judge", {}).get("trigger") == "correction_uncertain"
    # 成本：400 输入 + 80 输出，按 flash 闲时价算
    assert correction["judge"]["cost_usd"] == pytest.approx(
        400 / 1_000_000 * 0.15 + 80 / 1_000_000 * 0.60
    )
    # 账本里有对应的 supersede 事件
    lines = open(store.ledger_path, encoding="utf-8").read().splitlines()
    assert any('"supersede"' in line for line in lines)


# ---------- 2. 模型说不取代就并存 ----------


def test_judge_says_no_keeps_both(tmp_path, monkeypatch):
    """判官说不是同一类、不该取代：两条都活着，不崩、不丢。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢喝酸奶", "t1")
    judge.verdict = {
        "slot": "喜欢喝的饮品", "is_correction": False,
        "target_id": None, "should_supersede": False,
        "reason": "新说法和旧的不是同一回事",
    }
    receipt = _save(store, "我现在喜欢喝豆奶了", "t2")
    assert len(judge.calls) == 1
    assert _live_texts(store).count("我喜欢喝酸奶") == 1
    assert "我现在喜欢喝豆奶了" in _live_texts(store)
    assert receipt.get("correction") is None


# ---------- 3. 模型断了降级走程序 ----------


def test_judge_down_falls_back_to_program(tmp_path, monkeypatch):
    """判官抛错（超时/API 报错）：降级走程序，新事实照常存、旧的不动、不崩。"""
    judge = FakeJudge(fail=True)
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢喝酸奶", "t1")
    receipt = _save(store, "我现在喜欢喝豆奶了", "t2")
    assert len(judge.calls) == 1
    assert "我现在喜欢喝豆奶了" in receipt["stored"]
    assert "我喜欢喝酸奶" in _live_texts(store)
    assert "我现在喜欢喝豆奶了" in _live_texts(store)


def test_judge_garbage_response_falls_back(tmp_path, monkeypatch):
    """判官回的不是 JSON：解析失败，按程序 fallback 走，不崩。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢喝酸奶", "t1")
    judge.verdict = None  # chat_messages 会把 None 写成 "null"
    # 直接让 mock 回垃圾文本
    judge.chat_messages = lambda *a, **k: {
        "text": "我觉得吧，这个嘛……", "usage": None,
        "cost_usd": None, "model": "deepseek-flash",
    }
    receipt = _save(store, "我现在喜欢喝豆奶了", "t2")
    assert "我现在喜欢喝豆奶了" in receipt["stored"]
    assert "我喜欢喝酸奶" in _live_texts(store)


# ---------- 4. 程序路不调模型；换模型配置分发跟配置走 ----------


def test_program_slot_never_calls_judge(tmp_path, monkeypatch):
    """住处（程序能判定的槽位）：全程走程序，一次模型都不调；取代照常发生。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我住在成都", "t1")
    receipt = _save(store, "我现在住在重庆了", "t2")
    assert judge.calls == []
    assert _is_superseded(store, "我住在成都") is True
    # 纠正分支存的是补全后的新说法“我住在重庆了”（“我现在”被剥掉，老行为）
    assert "我住在重庆了" in _live_texts(store)
    correction = receipt.get("correction") or {}
    assert correction.get("route") != "model"  # 程序路，没走模型


def test_judge_dispatch_follows_config_not_hardcoded(monkeypatch):
    """换模型配置时分发跟配置走，没写死 DeepSeek。

    配置改成 deepseek-v4-pro，注入的 mock 也报 pro 名，
    verdict 里认到的就是 pro；改成没接上的 gpt，分发明确拒绝。
    """
    monkeypatch.setattr(
        sj, "SEMANTIC_JUDGE",
        {"provider": "deepseek", "model": "deepseek-v4-pro",
         "max_tokens": 300},
    )
    monkeypatch.setattr(sj, "get_api_key", lambda name: "test-key")
    judge = FakeJudge(verdict={
        "slot": "喜欢喝的饮品", "is_correction": True,
        "target_id": "m1", "should_supersede": True, "reason": "取代",
    })
    judge.model = "deepseek-v4-pro"
    verdict = sj.judge_semantic(
        "我现在喜欢喝豆奶了",
        [{"id": "m1", "text": "我喜欢喝酸奶"}], provider=judge,
    )
    assert verdict["model"] == "deepseek-v4-pro"
    assert verdict["provider"] == "deepseek"
    with pytest.raises(sj.SemanticJudgeUnavailable):
        sj.resolve_judge_provider("gpt")
    assert sj.judge_available()[0] is True


def test_judge_unconnected_provider_falls_back(tmp_path, monkeypatch):
    """配置切到没接上的模型（gpt）：分发拒绝，降级走程序，不崩。"""
    monkeypatch.setattr(
        sj, "SEMANTIC_JUDGE",
        {"provider": "gpt", "model": None, "max_tokens": 300},
    )
    assert sj.judge_available()[0] is False
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    store = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    _save(store, "我喜欢喝酸奶", "t1")
    receipt = _save(store, "我现在喜欢喝豆奶了", "t2")
    assert "我现在喜欢喝豆奶了" in receipt["stored"]
    assert "我喜欢喝酸奶" in _live_texts(store)


# ---------- 5. ambiguous 时判官挑一条 ----------


def test_judge_picks_ambiguous_correction_target(tmp_path, monkeypatch):
    """触发条件 (b)：correct_memory 报 ambiguous 后，判官挑一条，落盘取代。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢喝酸奶", source="user")
    store.add_memory_text("我喜欢喝豆奶", source="user")
    live = {item["text"]: item["id"] for item in store.get_all()}
    judge.verdict = {
        "slot": "喜欢喝的饮品", "is_correction": True,
        "target_id": live["我喜欢喝豆奶"], "should_supersede": True,
        "reason": "新说法取代豆奶那条",
    }
    correction = {
        "status": "ambiguous",
        "candidates": [
            {"id": live["我喜欢喝酸奶"], "text": "我喜欢喝酸奶"},
            {"id": live["我喜欢喝豆奶"], "text": "我喜欢喝豆奶"},
        ],
    }
    outcome = store._judge_ambiguous_correction(
        correction, "我现在喜欢喝椰奶了"
    )
    assert outcome is not None
    assert outcome["status"] == "updated"
    assert outcome["route"] == "model"
    assert outcome["judge"]["trigger"] == "ambiguous_candidates"
    assert _is_superseded(store, "我喜欢喝豆奶") is True
    assert _is_superseded(store, "我喜欢喝酸奶") is False
    assert "我现在喜欢喝椰奶了" in _live_texts(store)


# ---------- 6. 解析容错 ----------


def test_parse_judge_response_tolerant():
    """markdown 代码块包裹的 JSON 能解析；垃圾回、目标不在候选里都报错。"""
    wrapped = (
        '```json\n{"slot": "喜欢喝的饮品", "is_correction": true, '
        '"target_id": "m1", "should_supersede": true, "reason": "取代"}```'
    )
    out = sj.parse_judge_response(wrapped, ["m1", "m2"])
    assert out["target_id"] == "m1"
    assert out["should_supersede"] is True
    with pytest.raises(sj.SemanticJudgeError):
        sj.parse_judge_response("我觉得吧，这个嘛……", ["m1"])
    with pytest.raises(sj.SemanticJudgeError):
        sj.parse_judge_response(
            '{"slot": null, "is_correction": true, "target_id": "m9", '
            '"should_supersede": true, "reason": "x"}',
            ["m1"],
        )
    # 该取代却没指目标，也报错（不猜）
    with pytest.raises(sj.SemanticJudgeError):
        sj.parse_judge_response(
            '{"slot": null, "is_correction": true, "target_id": null, '
            '"should_supersede": true, "reason": "x"}',
            ["m1"],
        )


def test_estimate_judge_cost_scales_with_config():
    """成本按配置单价算：flash 和 pro 价不同，换模型成本跟着换。"""
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert sj.estimate_judge_cost_usd(
        usage, "deepseek", "deepseek-flash"
    ) == pytest.approx(0.15 + 0.60)
    assert sj.estimate_judge_cost_usd(
        usage, "deepseek", "deepseek-v4-pro"
    ) == pytest.approx(0.66 + 1.98)
    assert sj.estimate_judge_cost_usd(None, "deepseek", "deepseek-flash") is None
    assert sj.estimate_judge_cost_usd(usage, "gpt", "gpt-x") is None


# ---------- 7. 工具直存路 + 没候选不问 ----------


def test_add_memory_text_tool_path_judges(tmp_path, monkeypatch):
    """工具直存路（add_memory_text）：显式 opt-in（semantic_judge=True）
    时偏好句没槽位也问判官，该取代就取代。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢喝酸奶", source="user")
    acid = store.get_all()[0]["id"]
    judge.verdict = {
        "slot": "喜欢喝的饮品", "is_correction": True,
        "target_id": acid, "should_supersede": True, "reason": "取代",
    }
    result = store.add_memory_text(
        "我现在喜欢喝豆奶了", source="user", semantic_judge=True)
    assert result.get("event") == "ADD"
    assert len(judge.calls) == 1
    assert _is_superseded(store, "我喜欢喝酸奶") is True
    assert "我现在喜欢喝豆奶了" in _live_texts(store)


def test_add_memory_text_default_does_not_call_judge(tmp_path, monkeypatch):
    """Q3 回归 guard：add_memory_text 默认不问模型（opt-in 才问）。

    存储原语要确定、不花钱：直存偏好行时即使库里有同类偏好，
    也不调模型、不改排名（2026-11-23 真回归：默认调模型把 Q3
    纠正行取代了，前 5 排名乱了）。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢喝酸奶", source="user")
    acid = store.get_all()[0]["id"]
    judge.verdict = {
        "slot": "喜欢喝的饮品", "is_correction": True,
        "target_id": acid, "should_supersede": True, "reason": "取代",
    }
    result = store.add_memory_text("我现在喜欢喝豆奶了", source="user")
    assert result.get("event") == "ADD"
    assert judge.calls == []
    assert _is_superseded(store, "我喜欢喝酸奶") is False
    assert "我现在喜欢喝豆奶了" in _live_texts(store)


def test_first_preference_never_calls_judge(tmp_path, monkeypatch):
    """库里没同类偏好时不问模型：没候选可判，问了也是白花钱。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    receipt = _save(store, "我喜欢喝橙汁", "t1")
    assert judge.calls == []
    assert "我喜欢喝橙汁" in receipt["stored"]
