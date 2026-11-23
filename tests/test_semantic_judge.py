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
    Phase 2 #2 起 store 里有两个判官共用一个 provider：语义判官
    （judge_semantic）和重要性判官（judge_importance）。按提示词分流：
    含“值不值得长期钉住”的是重要性判断，回 importance_verdict、记进
    importance_calls；其余是语义判断，回 verdict、记进 calls。老测试
    断言的 calls 仍是语义调用次数，不受重要性判断影响。
    """

    def __init__(self, verdict=None, fail=False, importance_verdict=None):
        self.verdict = verdict
        self.fail = fail
        self.calls = []
        self.importance_verdict = importance_verdict or {
            "important": False, "reason": "测试默认一次性经历",
        }
        self.importance_calls = []
        self.provider_name = "deepseek"
        self.model = "deepseek-flash"

    def chat_messages(self, messages, ledger="chat", thinking=None,
                      max_tokens=None):
        prompt = (messages or [{}])[0].get("content") or ""
        if "值不值得长期钉住" in prompt:
            self.importance_calls.append(messages)
            if self.fail:
                raise RuntimeError("API 断了")
            return {
                "text": json.dumps(self.importance_verdict, ensure_ascii=False),
                "usage": {"prompt_tokens": 400, "completion_tokens": 80},
                "cost_usd": None,
                "model": "deepseek-flash",
            }
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


# ---------- 8. 规则 (a)：偏好变了就取代 ----------


def test_judge_prompt_has_preference_change_rule():
    """规则 (a)：提示词里有“偏好一次只有一个当前答案”这条规则。

    R2 食物案：判官曾把米饭和面条判成“可以并存”。规则写明：新说法带
    “现在/改/其实/不对/更喜欢/只”等词表示偏好变了时，同类旧偏好应当
    被取代，不要判成“可以并存”。加规则不能把原来的任务和 JSON 格式挤掉。
    """
    prompt = sj.build_judge_prompt(
        "我现在喜欢吃面条了",
        [{"id": "m1", "text": "我喜欢吃米饭"}],
    )
    assert "只有一个当前答案" in prompt
    assert "不要判成'可以并存'" in prompt
    for word in ("现在", "改", "其实", "不对", "更喜欢", "只"):
        assert f"'{word}'" in prompt
    # 原来的任务和 JSON 格式还在（加规则没挤掉别的东西）
    assert "新说法属于哪类偏好/槽位" in prompt
    assert "是不是在纠正/取代" in prompt
    assert "只回这一行 JSON" in prompt
    assert '"should_supersede"' in prompt


def test_judge_supersedes_food_preference(tmp_path, monkeypatch):
    """R2 食物案（mock）：米饭→面条，判官说取代，米饭被取代、只剩面条。

    规则触发时判取代：和饮品案同一条路，只是换成食物。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢吃米饭", "t1")
    assert judge.calls == []  # 第一条：库里没同类偏好，不问模型
    rice_id = store.get_all()[0]["id"]
    judge.verdict = {
        "slot": "喜欢吃的食物", "is_correction": True,
        "target_id": rice_id, "should_supersede": True,
        "reason": "同是喜欢吃的食物，新说法取代旧说法",
    }
    receipt = _save(store, "我现在喜欢吃面条了", "t2")
    assert len(judge.calls) == 1
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert "我现在喜欢吃面条了" in _live_texts(store)
    assert "我喜欢吃米饭" not in _live_texts(store)
    assert len([t for t in _live_texts(store) if "吃" in t]) == 1
    correction = receipt.get("correction") or {}
    assert correction.get("route") == "model"
    assert correction.get("status") == "updated"
    assert correction.get("judge", {}).get("trigger") == "correction_uncertain"


def test_judge_food_no_supersede_keeps_both(tmp_path, monkeypatch):
    """规则不触发时行为不变：判官说不取代，两条都活着，不崩、不丢。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢吃米饭", "t1")
    judge.verdict = {
        "slot": "喜欢吃的食物", "is_correction": False,
        "target_id": None, "should_supersede": False,
        "reason": "新说法和旧的不是同一回事",
    }
    receipt = _save(store, "我现在喜欢吃面条了", "t2")
    assert len(judge.calls) == 1
    assert "我喜欢吃米饭" in _live_texts(store)
    assert "我现在喜欢吃面条了" in _live_texts(store)
    assert receipt.get("correction") is None


# ---------- 9. R2 后续①：改吃/只吃类说法也触发判官 ----------


def test_judge_trigger_change_eat_phrasing(tmp_path, monkeypatch):
    """“我改吃面条了”没有“喜欢”标记，也要叫判官、取代米饭。

    2026-11-23 R2 后续①：触发器只认喜欢类标记时，改吃类说法根本
    叫不到判官，旧偏好换不掉。放宽后判官应被叫一次、取代米饭，
    最后只剩一条活的面条偏好。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢吃米饭", "t1")
    assert judge.calls == []
    rice_id = store.get_all()[0]["id"]
    judge.verdict = {
        "slot": "喜欢吃的实物", "is_correction": True,
        "target_id": rice_id, "should_supersede": True,
        "reason": "同类喜欢吃的食物，新说法取代旧说法",
    }
    receipt = _save(store, "我改吃面条了", "t2")
    assert len(judge.calls) == 1
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert "我改吃面条了" in _live_texts(store)
    assert "我喜欢吃米饭" not in _live_texts(store)
    assert len([t for t in _live_texts(store) if "面" in t]) == 1
    correction = receipt.get("correction") or {}
    assert correction.get("route") == "model"
    assert correction.get("status") == "updated"
    assert correction.get("judge", {}).get("trigger") == "preference_change"


def test_judge_trigger_only_eat_phrasing(tmp_path, monkeypatch):
    """“我现在只吃面条了”也要叫判官、取代米饭。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢吃米饭", "t1")
    assert judge.calls == []
    rice_id = store.get_all()[0]["id"]
    judge.verdict = {
        "slot": "喜欢吃的实物", "is_correction": True,
        "target_id": rice_id, "should_supersede": True,
        "reason": "同类喜欢吃的食物，新说法取代旧说法",
    }
    receipt = _save(store, "我现在只吃面条了", "t2")
    assert len(judge.calls) == 1
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert "我现在只吃面条了" in _live_texts(store)
    assert "我喜欢吃米饭" not in _live_texts(store)
    correction = receipt.get("correction") or {}
    assert correction.get("judge", {}).get("trigger") == "preference_change"


def test_change_marker_without_candidate_no_judge(tmp_path, monkeypatch):
    """没有同类旧偏好时，改吃类说法不叫判官，直接存。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    receipt = _save(store, "我改吃面条了", "t1")
    assert judge.calls == []
    assert "我改吃面条了" in receipt["stored"]


def test_change_marker_judge_says_no_keeps_both(tmp_path, monkeypatch):
    """判官说不取代时，改吃类新说法与旧偏好共存。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _save(store, "我喜欢吃米饭", "t1")
    judge.verdict = {
        "slot": "喜欢吃的实物", "is_correction": False,
        "target_id": None, "should_supersede": False,
        "reason": "新说法与旧的不是同一类",
    }
    receipt = _save(store, "我改吃面条了", "t2")
    assert len(judge.calls) == 1
    assert "我喜欢吃米饭" in _live_texts(store)
    assert "我改吃面条了" in _live_texts(store)
    assert receipt.get("correction") is None


# ---------- 10. R2 后续③：规则 (a) 收窄，只取代同一层级 ----------


def test_judge_prompt_narrows_same_level_rule():
    """规则 (a) 收窄：提示词只让判官取代同一层级、直接冲突的旧偏好。

    2026-11-23 R2 后续③：规则 (a) 太宽，判官把“喜欢吃面条”当成和
    “喜欢吃牛肉面”“喜欢吃鸡杂面”同类，一并取代了两条真记忆（63ab、
    a3a90d05）。收窄后提示词要写明：泛指不取代更具体的同类记忆、
    纠正记录不是取代目标、没有同一层级冲突时判不取代、不要硬挑一条。
    原来的“只有一个当前答案”和“不要判成'可以并存'”还在。
    """
    prompt = sj.build_judge_prompt(
        "我现在喜欢吃面条了",
        [{"id": "m1", "text": "我喜欢吃米饭"},
         {"id": "m2", "text": "BOOS 喜欢吃牛肉面。"}],
    )
    assert "只有一个当前答案" in prompt
    assert "不要判成'可以并存'" in prompt
    assert "同一层级" in prompt
    assert "不取代更具体的同类记忆" in prompt
    assert "喜欢吃牛肉面" in prompt
    assert "喜欢吃鸡杂面" in prompt
    assert "纠正了此前" in prompt
    assert "不要硬挑一条" in prompt
    # 原来的任务和 JSON 格式还在
    assert "新说法属于哪类偏好/槽位" in prompt
    assert "只回这一行 JSON" in prompt


def test_judge_candidates_skip_correction_notes(tmp_path, monkeypatch):
    """纠正记录不进判官候选：72cc 那种“纠正了此前……”的便条不是偏好。

    2026-11-23 R2 后续③：判官两次把 72cc 纠正便条当成同类偏好取代
    （账本 seq 329、356）。程序过滤是确定性的：候选里有米饭和纠正
    便条时，便条被滤掉、米饭留下。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text(
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前"
        "“喜欢吃鸡杂面”的说法。",
        source="user",
    )
    assert len(_live_texts(store)) == 2
    cands = store._preference_judge_candidates("我现在喜欢吃面条了")
    cand_texts = [c["text"] for c in cands]
    assert "我喜欢吃米饭" in cand_texts
    assert not any("纠正" in t for t in cand_texts)


def test_judge_supersedes_only_same_level_target(tmp_path, monkeypatch):
    """判官指哪条才取代哪条：牛肉面、鸡杂面、纠正便条都不应被顺带取代。

    2026-11-23 R2 后续③：上一轮重测里判官指了鸡杂面/牛肉面当目标，
    另一条路又取代了米饭，结果两条真记忆被误伤。收窄后判官只应指
    同一层级的米饭；这里用 mock 判官指米饭，断言牛肉面、鸡杂面、
    纠正便条都还活着，只有米饭被取代。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text("BOOS 喜欢吃牛肉面。", source="user")
    store.add_memory_text("BOOS 喜欢吃鸡杂面。", source="user")
    store.add_memory_text(
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前"
        "“喜欢吃鸡杂面”的说法。",
        source="user",
    )
    live = {item["text"]: item["id"] for item in store.get_all()}
    assert len(live) == 4
    judge.verdict = {
        "slot": "喜欢吃的食物", "is_correction": True,
        "target_id": live["我喜欢吃米饭"], "should_supersede": True,
        "reason": "同是主食，新说法取代旧说法",
    }
    receipt = _save(store, "我现在喜欢吃面条了", "t2")
    assert len(judge.calls) == 1
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert _is_superseded(store, "BOOS 喜欢吃牛肉面。") is False
    assert _is_superseded(store, "BOOS 喜欢吃鸡杂面。") is False
    assert _is_superseded(
        store,
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前"
        "“喜欢吃鸡杂面”的说法。",
    ) is False
    assert "我现在喜欢吃面条了" in _live_texts(store)
    assert "BOOS 喜欢吃牛肉面。" in _live_texts(store)
    assert "BOOS 喜欢吃鸡杂面。" in _live_texts(store)
    correction = receipt.get("correction") or {}
    assert correction.get("route") == "model"
    assert correction.get("status") == "updated"


# ---------- 11. R2 后续③补丁：主食守卫 ----------


def test_judge_target_is_staple_plausible():
    """主食守卫判定：换主食时目标必须是主食，非主食目标不合理。"""
    from memory.sqlite_store import _judge_target_is_staple_plausible as ok
    assert ok("我改吃面条了", "我喜欢吃米饭") is True
    assert ok("我现在喜欢吃面条了", "我喜欢吃米饭") is True
    assert ok("我改吃面条了", "我爱吃火锅") is False
    assert ok("我改吃面条了", "BOOS 喜欢吃牛肉面。") is False
    assert ok("我改吃面条了", "我爱的是冒菜，不是麻辣烫") is False
    assert ok("我改吃面条了", "BOOS 喜欢吃鸡杂面。") is False
    # 新说法不含主食词（饮品）时不受限制
    assert ok("我现在喜欢喝豆奶了", "我喜欢喝酸奶") is True
    assert ok("我现在喜欢喝豆奶了", "我爱吃火锅") is True


def test_judge_staple_guard_blocks_hotpot_target(tmp_path, monkeypatch):
    """主食守卫：判官指火锅当目标时不取代，火锅真记忆保住。

    2026-11-23 R2 后续③重测 D 组：模型 add_memory 先取代了米饭，
    save_turn 的判官后跑、米饭已不在候选里，判官硬挑“爱吃火锅”当目标
    误伤了真记忆。守卫拦下后，火锅不被取代，新说法照常存下来。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text("我爱吃火锅", source="user")
    live = {item["text"]: item["id"] for item in store.get_all()}
    judge.verdict = {
        "slot": "喜欢吃的食物", "is_correction": True,
        "target_id": live["我爱吃火锅"], "should_supersede": True,
        "reason": "同是喜欢吃的食物，新说法取代旧说法",
    }
    receipt = _save(store, "我改吃面条了", "t2")
    assert len(judge.calls) == 1
    assert _is_superseded(store, "我爱吃火锅") is False
    assert "我爱吃火锅" in _live_texts(store)
    assert "我改吃面条了" in _live_texts(store)


def test_judge_staple_guard_allows_rice_target(tmp_path, monkeypatch):
    """主食守卫放行：判官指米饭（主食）时照常取代，火锅不受牵连。

    同一场景、判官指对目标时：米饭被取代，只剩面条；火锅还活着。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢吃米饭", source="user")
    store.add_memory_text("我爱吃火锅", source="user")
    live = {item["text"]: item["id"] for item in store.get_all()}
    judge.verdict = {
        "slot": "喜欢吃的食物", "is_correction": True,
        "target_id": live["我喜欢吃米饭"], "should_supersede": True,
        "reason": "同是主食，新说法取代旧说法",
    }
    receipt = _save(store, "我改吃面条了", "t2")
    assert len(judge.calls) == 1
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert _is_superseded(store, "我爱吃火锅") is False
    assert "我改吃面条了" in _live_texts(store)
    assert "我爱吃火锅" in _live_texts(store)
    assert "我喜欢吃米饭" not in _live_texts(store)


# ---------- 12. R2 后续③补丁 2：括号补充去重 ----------


def test_paraphrase_with_parenthetical_is_duplicate(tmp_path, monkeypatch):
    """括号补充算同一条：改写句与原句不存两份。

    2026-11-23 R2 后续③：模型 add_memory 存了“我改吃面条了（主食偏好是面条）”，
    save_turn 又存原句“我改吃面条了”，活面条变 2 条，D 组 noodle1 失败。
    去重时去掉末尾括号补充再比，两句算同一条，只留改写句那一条。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我改吃面条了（主食偏好是面条）", source="user")
    assert len(_live_texts(store)) == 1
    result = store.add_memory_text("我改吃面条了", source="user")
    assert result.get("event") == "UNCHANGED"
    assert len(_live_texts(store)) == 1
    assert "我改吃面条了（主食偏好是面条）" in _live_texts(store)


def test_parenthetical_duplicate_reverse_order(tmp_path, monkeypatch):
    """反过来也一样：先存原句、后存改写句，改写句不再存第二份。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我改吃面条了", source="user")
    assert len(_live_texts(store)) == 1
    result = store.add_memory_text("我改吃面条了（主食偏好是面条）", source="user")
    assert result.get("event") == "UNCHANGED"
    assert len(_live_texts(store)) == 1
    assert "我改吃面条了" in _live_texts(store)


def test_different_food_not_deduped_by_parenthetical(tmp_path, monkeypatch):
    """括号去重不误伤：米饭和面条不是同一条，两条都留着。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我喜欢吃米饭", source="user")
    result = store.add_memory_text("我喜欢吃面条", source="user")
    assert result.get("event") == "ADD"
    assert len(_live_texts(store)) == 2


def test_comma_paraphrase_is_duplicate(tmp_path, monkeypatch):
    """逗号改写也算同一条：括号版与逗号版去标点后相同，不存两份。

    2026-11-23 R2 后续③实弹：模型改写同一事实时标点不固定，存了
    “我改吃面条了（主食偏好是面条）”后又存“BOOS 改吃面条了，主食偏好是面条”，
    活面条变 2 条。去重键去掉全部标点后再比，两句是同一串字。
    """
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    store.add_memory_text("我改吃面条了（主食偏好是面条）", source="user")
    assert len(_live_texts(store)) == 1
    result = store.add_memory_text("BOOS 改吃面条了，主食偏好是面条", source="user")
    assert result.get("event") == "UNCHANGED"
    assert len(_live_texts(store)) == 1
