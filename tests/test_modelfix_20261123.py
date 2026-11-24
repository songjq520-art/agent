# tests/test_modelfix_20261123.py（model fix 四项，2026-11-23）
# 全部离线（向量打桩 + mock 判官，不加载 bge 模型、不调真 API、不花钱）。
# 覆盖：
#   #1 D 组根因：无纠正词的新陈述不顶同槽旧记忆（只存新行）；带明确
#     纠正词的只顶最新那条同槽冲突；exact 层不被无纠正词的陈述触发。
#   #2 A 组阈值：带明确纠正词的不走相似度 0.70，按槽位找目标。
#   #3 措辞：纠正执行完回复“已经改好了”，不提确认、不反问；同槽冲突
#     的新陈述回复“记下了”，不反问。
#   #4 模型优先 + 全上下文：带纠正说法、没偏好标记也问判官；判官提示词
#     带整段对话上下文；模型断了降级走程序、不丢事实。

import json
import math

import pytest

from memory import semantic_judge as sj
from memory.correction_text import _correction_topic_slot
from memory.sqlite_store import SQLiteStore
from web.server import _correction_reply_text, _slot_conflict_reply_text


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每测一个干净的 SQLiteStore；向量打桩成 None（只走原样/关键词路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _live_texts(store):
    return [item["text"] for item in store.get_all()]


def _is_superseded(store, text):
    row = store._conn.execute(
        "SELECT superseded FROM memories WHERE text=?", (text,)
    ).fetchone()
    return bool(row and row[0])


def _turn(store, text, turn_id):
    """一轮对话：用户说一句话，助手回一句“好的”。"""
    return store.save_turn(
        [{"role": "user", "content": text},
         {"role": "assistant", "content": "好的。"}],
        turn_id=turn_id,
    )


class FakeJudge:
    """mock 判官：按设好的 verdict 回 JSON，或直接抛错模拟模型断了。
    照搬 tests/test_semantic_judge.py 同名帮手（语义/重要性按提示词分流）。"""

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


def _sim68_embed(self, text):
    """米饭/面条的余弦相似度恰好 0.68（低于 0.70 阈值），复现 A 组卡点。"""
    if "米饭" in text:
        return [1.0, 0.0]
    if "面条" in text:
        return [0.68, math.sqrt(1 - 0.68 ** 2)]
    return [0.0, 1.0]


@pytest.fixture
def sim68_store(tmp_path, monkeypatch):
    """向量按 0.68 相似度打桩的 store：米饭行和面条说法的相似度 0.68。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", _sim68_embed)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


# ---------- #1 D 组根因：无纠正词不顶旧 ----------


def test_modelfix_d1_plain_statement_keeps_guangzhou(store):
    """D 组根因：库里有“用户住在广州”，用户只说一句“我住在武汉”
    （无纠正词）——广州必须活着，武汉只存成新行，不顶旧的、不问。"""
    store.add_memory_text("用户住在广州", source="user")
    receipt = _turn(store, "我住在武汉", "t1")
    assert receipt.get("correction") is None
    assert "用户住在广州" in _live_texts(store)
    assert "我住在武汉" in _live_texts(store)
    assert _is_superseded(store, "用户住在广州") is False


def test_modelfix_d1_add_memory_text_plain_no_supersede(store):
    """直存路同理：无纠正词的 add_memory_text 不顶同槽旧行。"""
    store.add_memory_text("用户住在广州", source="user")
    result = store.add_memory_text("我住在武汉", source="user")
    assert result.get("event") == "ADD"
    assert "用户住在广州" in _live_texts(store)
    assert "我住在武汉" in _live_texts(store)


@pytest.mark.requires_judge
def test_modelfix_d1_explicit_correction_supersedes_only_newest(store):
    """带明确纠正词的才算纠正，且只顶最新那条同槽冲突：
    广州（旧真记忆）+ 武汉（刚说的）都在，纠正“错了，我住在长沙”
    只能顶武汉，广州必须活着，长沙是新的当前答案。"""
    store.add_memory_text("用户住在广州", source="user")
    _turn(store, "我住在武汉", "t1")
    assert "用户住在广州" in _live_texts(store)
    assert "我住在武汉" in _live_texts(store)
    receipt = _turn(store, "错了，我住在长沙", "t2")
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated"
    assert correction.get("old_text") == "我住在武汉"
    assert _is_superseded(store, "我住在武汉") is True
    assert _is_superseded(store, "用户住在广州") is False
    assert "用户住在广州" in _live_texts(store)
    assert "我住在长沙" in _live_texts(store)


@pytest.mark.requires_judge
def test_modelfix_d1_add_memory_text_allow_supersede_only_newest(store):
    """显式允许槽位取代时（allow_slot_supersede=True），也只顶最新一条。"""
    store.add_memory_text("用户住在广州", source="user")
    store.add_memory_text("我住在武汉", source="user",
                          allow_slot_supersede=False)
    result = store.add_memory_text("我住在长沙", source="user",
                                   allow_slot_supersede=True)
    assert result.get("event") == "ADD"
    assert _is_superseded(store, "我住在武汉") is True
    assert _is_superseded(store, "用户住在广州") is False


# ---------- #2 A 组阈值：明确纠正词按槽位找 ----------


def test_modelfix_a2_topic_slot():
    """纠正目标槽位键：住处/名字走单值槽，偏好按“喜欢吃”这类话题槽。"""
    assert _correction_topic_slot("我喜欢吃米饭") == "pref:喜欢吃"
    assert _correction_topic_slot("我喜欢吃面条") == "pref:喜欢吃"
    assert _correction_topic_slot("我爱吃火锅") == "pref:喜欢吃"
    assert _correction_topic_slot("我住在广州") == "slot:residence"
    assert _correction_topic_slot("我叫小宋") == "slot:user_name"
    assert _correction_topic_slot("今天天气不错") is None


@pytest.mark.requires_judge
def test_modelfix_a2_explicit_correction_bypasses_similarity(sim68_store):
    """A 组卡点：米饭行和面条说法的相似度只有 0.68（< 0.70），
    不带纠正词标记时 correct_memory 不敢改（ambiguous，旧行活着）；
    带明确纠正词时按“喜欢吃”槽直接找到米饭那条并 supersede 它。"""
    store = sim68_store
    store.add_memory_text("我喜欢吃米饭", source="user")
    control = store.correct_memory("我喜欢吃面条", "我喜欢吃面条")
    assert control.get("status") == "ambiguous"
    assert _is_superseded(store, "我喜欢吃米饭") is False
    out = store.correct_memory(
        "我喜欢吃面条", "我喜欢吃面条", explicit_correction=True
    )
    assert out.get("status") == "updated"
    assert out.get("old_text") == "我喜欢吃米饭"
    assert out.get("layer") == "slot"
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert "我喜欢吃面条" in _live_texts(store)


def test_modelfix_a2_explicit_slot_target_needs_unique_candidate(store):
    """回归修（2026-11-23）：槽位直选只在同槽位候选恰好一条时才用——
    “不对，我喜欢吃米线”对火锅/冒菜两条同槽旧行时，correct_memory
    必须报 ambiguous（走老路），不能直选第一条就 updated。"""
    store.add_memory_text("我喜欢吃火锅", source="user")
    store.add_memory_text("我喜欢吃冒菜", source="user")
    out = store.correct_memory(
        "我喜欢吃米线", "我喜欢吃米线", explicit_correction=True
    )
    assert out.get("status") == "ambiguous"
    assert _is_superseded(store, "我喜欢吃火锅") is False
    assert _is_superseded(store, "我喜欢吃冒菜") is False
    assert "我喜欢吃火锅" in _live_texts(store)
    assert "我喜欢吃冒菜" in _live_texts(store)


@pytest.mark.requires_judge
def test_modelfix_a2_save_turn_end_to_end(sim68_store):
    """A 组端到端：T1 米饭 → T2“记错了，我喜欢吃面条”→ 米饭被取代，
    面条是当前答案，纠正回执是 updated。"""
    store = sim68_store
    _turn(store, "我喜欢吃米饭", "t1")
    assert "我喜欢吃米饭" in _live_texts(store)
    receipt = _turn(store, "记错了，我喜欢吃面条", "t2")
    correction = receipt.get("correction") or {}
    assert correction.get("status") == "updated"
    assert correction.get("old_text") == "我喜欢吃米饭"
    assert _is_superseded(store, "我喜欢吃米饭") is True
    assert "我喜欢吃面条" in _live_texts(store)


# ---------- #3 措辞：改完直接说“改好了” ----------


def test_modelfix_w3_correction_reply_text():
    """纠正执行完（updated）的回复：直接说“已经改好了”，不提确认、
    不事后反问“要不要改”。没改成时返回 None（保留模型原回复）。"""
    reply = _correction_reply_text({
        "status": "updated",
        "old_text": "我喜欢吃米饭", "new_text": "我喜欢吃面条",
    })
    assert reply == "已经改好了：把“我喜欢吃米饭”改成了“我喜欢吃面条”。"
    assert "确认" not in reply
    assert "要不要" not in reply
    assert _correction_reply_text({"status": "not_found"}) is None
    assert _correction_reply_text({"status": "ambiguous"}) is None
    assert _correction_reply_text({"status": "unchanged"}) is None
    assert _correction_reply_text(None) is None


def test_modelfix_w3_slot_conflict_reply_text(store):
    """无纠正词的新陈述存成新行、同槽旧行还在时：只认“记下了”，
    不反问“要不要改”；没冲突时返回 None。"""
    store.add_memory_text("我住在长沙", source="user")
    reply = _slot_conflict_reply_text(store, ["我住在西安"])
    assert reply == "记下了：我住在西安。"
    assert "要不要" not in reply
    assert _slot_conflict_reply_text(store, ["我喜欢吃米饭"]) is None
    # 同槽旧行还在时，另一条新住址也一样只认记下了。
    assert _slot_conflict_reply_text(store, ["我住在南京"]) == "记下了：我住在南京。"


# ---------- #4 模型优先 + 全上下文 ----------


def test_modelfix_j4_judge_prompt_includes_context():
    """判官提示词带整段对话上下文：有 context 时出现“对话上下文”一段，
    没 context 时不加，老行为不变。"""
    cands = [{"id": "x", "text": "我喜欢喝酸奶"}]
    ctx = [
        {"role": "user", "content": "我喜欢喝酸奶"},
        {"role": "assistant", "content": "好的。"},
        {"role": "user", "content": "我现在喝豆奶了"},
    ]
    prompt = sj.build_judge_prompt("我现在喝豆奶了", cands, context=ctx)
    assert "对话上下文" in prompt
    assert "用户：我喜欢喝酸奶" in prompt
    assert "新说法：我现在喝豆奶了" in prompt
    plain = sj.build_judge_prompt("我现在喝豆奶了", cands)
    assert "对话上下文" not in plain


def test_modelfix_j4_importance_prompt_includes_context():
    """重要性判官提示词同样带整段对话上下文。"""
    ctx = [{"role": "user", "content": "我表弟下个月可能来广州玩"}]
    prompt = sj.build_importance_prompt("我表弟下个月可能来广州玩",
                                        context=ctx)
    assert "对话上下文" in prompt
    assert "用户：我表弟下个月可能来广州玩" in prompt
    assert "值不值得长期钉住" in prompt


def test_modelfix_j4_judge_config_defaults(monkeypatch):
    """判官模型不固定 DeepSeek：配置能换（config.py 的 SEMANTIC_JUDGE
    跟 SHIER_JUDGE_PROVIDER / SHIER_JUDGE_MODEL 环境变量走），
    缺字段时默认 deepseek/deepseek-flash，不崩。"""
    monkeypatch.setattr(sj, "SEMANTIC_JUDGE", {})
    cfg = sj.get_judge_config()
    assert cfg["provider"] == "deepseek"
    assert cfg["model"] == "deepseek-flash"
    monkeypatch.setattr(sj, "SEMANTIC_JUDGE", {
        "provider": "gpt", "model": "gpt-x", "max_tokens": 300,
    })
    cfg = sj.get_judge_config()
    assert cfg["provider"] == "gpt"
    assert cfg["model"] == "gpt-x"


def test_modelfix_j4_correction_phrase_without_marker_triggers_judge(
        tmp_path, monkeypatch):
    """模型优先放宽：带纠正说法、没偏好标记的“我现在喝豆奶了”，
    库里有同类偏好时也要问判官；判官收到的提示词里有整段对话上下文；
    判官说取代，酸奶就被取代。"""
    judge = FakeJudge()
    store = _judge_store(tmp_path, monkeypatch, judge)
    _turn(store, "我喜欢喝酸奶", "t1")
    assert judge.calls == []
    yogurt_id = store.get_all()[0]["id"]
    judge.verdict = {
        "slot": "喜欢喝的饮品", "is_correction": True,
        "target_id": yogurt_id, "should_supersede": True,
        "reason": "同是喜欢喝的饮品，新说法取代旧说法",
    }
    receipt = _turn(store, "我现在喝豆奶了", "t2")
    assert len(judge.calls) == 1
    assert "对话上下文" in judge.calls[0][0]["content"]
    assert _is_superseded(store, "我喜欢喝酸奶") is True
    assert "我现在喝豆奶了" in _live_texts(store)
    correction = receipt.get("correction") or {}
    assert correction.get("route") == "model"
    assert correction.get("judge", {}).get("trigger") == "correction_uncertain"


def test_modelfix_j4_judge_down_falls_back_without_losing_fact(
        tmp_path, monkeypatch):
    """模型断了降级走程序：不崩、不丢事实——新说法照常存，旧的不动。"""
    judge = FakeJudge(fail=True)
    store = _judge_store(tmp_path, monkeypatch, judge)
    _turn(store, "我喜欢喝酸奶", "t1")
    receipt = _turn(store, "我现在喝豆奶了", "t2")
    assert "我现在喝豆奶了" in _live_texts(store)
    assert "我喜欢喝酸奶" in _live_texts(store)
    assert _is_superseded(store, "我喜欢喝酸奶") is False
    assert receipt.get("correction") is None
