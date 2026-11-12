# tests/test_batch3_memory.py（第 3 批：记忆那组 34、42、46、43、4、26 号）
# 作用：把第 3 批六项的修法钉死在测试里，全部离线模拟，不调真实 API、
# 不花钱；真调在 /tmp 外面另做（见第 3 批报告）。
# 假判说明：模型判（撤回意图/去重同一件事/价值打分/提炼规则）都走
# _install_fake_llm 装的假 DeepSeek——handler 按系统提示分派，
# 测试只验“判完之后的代码走没走对”，不验模型本身判得准不准；
# 模型判得准不准靠真调 DeepSeek 验（见第 3 批报告）。

import json
import math
import re
from types import SimpleNamespace

import pytest

from main import _memory_block
from memory.mem0_store import (
    CORRECTION_TOPIC_MIN_SIMILARITY,
    DEDUP_SIMILARITY_MIN,
    VALUE_THRESHOLD,
)

from conftest import make_fake_store


# ---------- 假对象 ----------

def _install_fake_llm(store, handler):
    """给假 store 装一个假 DeepSeek 客户端（第 3 批四个判共用）。

    handler(system_prompt, user_prompt) -> dict（模型回的 JSON）或 None
    （回的不是 JSON / 判不出来）。每次调用都记进 calls（每项
    {"system": 系统提示, "user": 用户提示}），供测试断言“某路判
    调了没调”。FakeMemory 本来没有 llm 属性，这里是测试给它临时装
    一个，不影响其他测试。
    """
    calls = []

    class _FakeCompletions:
        def create(self, *args, **kwargs):
            messages = kwargs.get("messages") or []
            system = ""
            user = ""
            for msg in messages:
                if (msg or {}).get("role") == "system":
                    system = (msg or {}).get("content") or ""
                elif (msg or {}).get("role") == "user":
                    user = (msg or {}).get("content") or ""
            calls.append({"system": system, "user": user})
            data = handler(system, user)
            content = json.dumps(data, ensure_ascii=False) if data is not None else ""
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                        finish_reason=None,
                    )
                ]
            )

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeLLM:
        client = SimpleNamespace(chat=_FakeChat())

    store._memory.llm = _FakeLLM()
    return calls


def _install_fake_embedder(store, angles):
    """给假 store 装一个假本地 embedding 模型（去重预筛/安全网用）。

    angles：{文本: 夹角（弧度）}——embed 返回二维单位向量
    [cos(夹角), sin(夹角)]，于是两段文本的余弦相似度就是
    cos(夹角差)，测试能精确控制“话题像不像”。没在 angles 里的文本
    默认夹角 0。FakeMemory 本来没有 embedding_model 属性，这里是
    测试给它临时装一个，不影响其他测试。
    """
    class _FakeEmbedder:
        def embed(self, text, memory_action=None):
            angle = angles.get(text, 0.0)
            return [math.cos(angle), math.sin(angle)]

    store._memory.embedding_model = _FakeEmbedder()


def _first_candidate_id(user_prompt):
    """从去重判的提示里抠出第 1 条候选的 id（假“模型”：永远挑第 1 条，
    模拟“模型说这是同一件事”）。"""
    match = re.search(r"1\. id=(\S+?)：", user_prompt or "")
    return match.group(1) if match else "none"


def _round(user_text, assistant_text="好的。"):
    """一轮对话（供 add_turn / save_turn 用）。"""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _texts(store):
    """库里所有记忆文本（按存入顺序）。"""
    return [item["text"] for item in store.get_all()]



# ---------- 34 号：撤回只认整句信号词 ----------

_RETRACTION_PHRASES = [
    "刚才那句收回",
    "上面那句别记了",
    "说错了，前面那句不算",
    "撤回上一句",
    "当我没说，刚刚那句忘掉",
]


@pytest.mark.parametrize("phrase", _RETRACTION_PHRASES)
def test_34_five_new_retraction_phrasings(tmp_path, phrase):
    """34 号：5 种新说法撤回（都不是 11 月 11 日实测用过的旧问法，
    也不是光秃秃的信号词）——模型判意图后，待定区里的上一轮丢掉，
    蓝色那条绝不能进库。"""
    store = make_fake_store(tmp_path)
    _install_fake_llm(
        store,
        lambda system, user: {"retract": True}
        if "撤回意图" in system
        else None,
    )
    result = store.add_turn(_round("我喜欢蓝色。", "记下了。"))
    assert result.get("pending") is True
    result = store.add_turn(_round(phrase))
    assert result.get("retracted") is True
    assert result.get("dropped") == 1
    assert not any("蓝色" in text for text in _texts(store))
    store.flush_pending()
    assert not any("蓝色" in text for text in _texts(store))


def test_34_correction_with_new_fact_is_not_retraction(tmp_path):
    """34 号：带新事实的纠正（“不对，我喜欢绿色。”）不走撤回——
    撤回判一次都没调，纠正照常改（蓝色换成绿色），绿色进库。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    calls = _install_fake_llm(
        store,
        lambda system, user: {"rule": None}
        if "提炼助手" in system
        else {"retract": False},
    )
    result = store.add_turn(_round("不对，我喜欢绿色。", "明白，我更新一下。"))
    assert result.get("retracted") is not True
    assert not any("撤回意图" in call["system"] for call in calls)
    assert result.get("correction", {}).get("status") == "updated"
    assert "我喜欢绿色。" in _texts(store)
    assert not any("蓝色" in text for text in _texts(store))


def test_34_soft_delete_and_restore(tmp_path):
    """34 号：软删除——从向量库删掉、全文记进回收站（能恢复）；
    恢复后原样回来，回收站清掉；找不到的 id 删/恢复都返回 False。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    mem_id = store.get_all()[0]["id"]
    assert store.soft_delete_memory(mem_id) is True
    assert _texts(store) == []
    retracted = store.list_retracted()
    assert len(retracted) == 1
    assert retracted[0]["id"] == mem_id
    assert "蓝色" in retracted[0]["text"]
    assert store.restore_memory(mem_id) is True
    assert "我喜欢蓝色。" in _texts(store)
    assert store.list_retracted() == []
    assert store.soft_delete_memory("no-such-id") is False
    assert store.restore_memory("no-such-id") is False


def test_34_retraction_soft_deletes_stored_memory(tmp_path):
    """34 号：撤回已进库的记忆——“记住”存下的蓝色，下一句撤回后，
    库里没了、回收站里有（能恢复），不是真删。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    mem_id = store.get_all()[0]["id"]
    _install_fake_llm(
        store,
        lambda system, user: {"retract": True}
        if "撤回意图" in system
        else None,
    )
    result = store.add_turn(
        [
            {"role": "user", "content": "我喜欢蓝色。"},
            {"role": "assistant", "content": "记下了。"},
            {"role": "user", "content": "不不不，刚刚那句不算"},
            {"role": "assistant", "content": "好的。"},
        ]
    )
    assert result.get("retracted") is True
    assert result.get("soft_deleted") == [mem_id]
    assert _texts(store) == []
    assert len(store.list_retracted()) == 1
    assert store.restore_memory(mem_id) is True
    assert "我喜欢蓝色。" in _texts(store)


def test_34_judge_says_no_means_no_retraction(tmp_path):
    """34 号：模型判“不是撤回”时，待定区照常提交、记忆照常存——
    判不出来（没装假判）时也一样，绝不因为判不出来就丢记忆。"""
    store = make_fake_store(tmp_path)
    _install_fake_llm(
        store,
        lambda system, user: {"retract": False}
        if "撤回意图" in system
        else None,
    )
    store.add_turn(_round("我喜欢蓝色。", "记下了。"))
    result = store.add_turn(_round("算了，先不说这个了"))
    assert result.get("retracted") is not True
    store.flush_pending()
    # 假提取会把句末“。”去掉，存的是“我喜欢蓝色”（无句号）。
    assert "我喜欢蓝色" in _texts(store)


# ---------- 42 号：纠正不泛化 ----------

def test_42_correction_derives_and_stores_rule(tmp_path):
    """42 号：纠正交易日算术时，提炼出通用规则（“定投只在交易日扣款，
    一个月约 21 个交易日”），单独存一条 is_rule=True 的规则记忆；
    同一条规则再纠正一次也不重复存。"""
    store = make_fake_store(tmp_path)
    _install_fake_llm(
        store,
        lambda system, user: {
            "rule": "定投只在交易日扣款，一个月约 21 个交易日"
        }
        if "提炼助手" in system
        else None,
    )
    store.add_memory_text("每月定投 5 块，一年 1825 块。")
    result = store.correct_memory(
        "每月定投 5 块，一年 1825 块。",
        "每月定投 5 块，一个月约 21 个交易日，一年约 1260 块。",
    )
    assert result["status"] == "updated"
    assert result.get("rule_id")
    rules = [
        item
        for item in store.get_all()
        if (item.get("metadata") or {}).get("is_rule") is True
    ]
    assert len(rules) == 1
    assert rules[0]["text"] == "定投只在交易日扣款，一个月约 21 个交易日"
    # 同一条规则再纠正一次：不重复存，rule_id 是 None。
    store.add_memory_text("每周定投 10 块，一年 3650 块。")
    result2 = store.correct_memory(
        "每周定投 10 块，一年 3650 块。",
        "每周定投 10 块，一周 5 个交易日，一年约 2600 块。",
    )
    assert result2["status"] == "updated"
    assert result2.get("rule_id") is None
    rules = [
        item
        for item in store.get_all()
        if (item.get("metadata") or {}).get("is_rule") is True
    ]
    assert len(rules) == 1



def test_42_no_rule_when_correction_has_none(tmp_path):
    """42 号：只是改了一个具体事实、背后没有通用规则时（比如喜欢
    蓝色改成喜欢绿色），不存规则——纠正本身照常成功。"""
    store = make_fake_store(tmp_path)
    _install_fake_llm(
        store,
        lambda system, user: {"rule": None}
        if "提炼助手" in system
        else None,
    )
    store.add_memory_text("我喜欢蓝色。")
    result = store.correct_memory("我喜欢蓝色。", "我喜欢绿色。")
    assert result["status"] == "updated"
    assert result.get("rule_id") is None
    rules = [
        item
        for item in store.get_all()
        if (item.get("metadata") or {}).get("is_rule") is True
    ]
    assert rules == []




def test_42_rule_first_in_memory_block():
    """42 号：规则在记忆提示块里排最前、标规则标记——回答同类问题时
    先按规则算，再结合具体记忆。（规则标记用 unicode 转义写，
    免得写工具吞掉全角括号。）"""
    rule_tag = "\u3010\u89c4\u5219\u3011"
    memories = [
        {
            "text": "用户喜欢喝冰美式。",
            "metadata": {"source": "user", "confirmed": True},
        },
        {
            "text": "定投只在交易日扣款，一个月约 21 个交易日。",
            "metadata": {"source": "inferred", "confirmed": False, "is_rule": True},
        },
        {
            "text": "用户每天定投 5 块买万家纳指。",
            "metadata": {"source": "inferred", "confirmed": False},
        },
    ]
    block = _memory_block(memories)
    lines = [line for line in block.split("\n") if line.startswith("- ")]
    assert len(lines) == 3
    assert rule_tag in lines[0]
    assert "交易日" in lines[0]
    assert rule_tag not in lines[1]
    assert rule_tag not in lines[2]


# ---------- 46 号：去重失灵 ----------

def test_46_same_fact_three_wordings_one_in_store(tmp_path):
    """46 号：同一条记忆换 3 种说法（助手叫十二 / 助手的名字是十二 /
    你的名字是十二）——库里只留一条。向量预筛过了线（相似度 1.0），
    模型判“是同一件事”才合并。"""
    store = make_fake_store(tmp_path)
    _install_fake_embedder(
        store,
        {
            "助手叫十二。": 0.0,
            "助手的名字是十二。": 0.0,
            "你的名字是十二。": 0.0,
        },
    )
    _install_fake_llm(
        store,
        lambda system, user: {"choice": _first_candidate_id(user)}
        if "去重" in system
        else None,
    )
    assert store.add_memory_text("助手叫十二。").get("duplicate") is not True
    assert store.add_memory_text("助手的名字是十二。").get("duplicate") is True
    assert store.add_memory_text("你的名字是十二。").get("duplicate") is True
    assert _texts(store) == ["助手叫十二。"]


def test_46_different_numbers_do_not_merge(tmp_path):
    """46 号：数字不同就是不同事实——“每天喝 2 杯咖啡”和“每天喝 3 杯咖啡”
    相似度再高（这里故意设成 1.0）也不合并，模型判“不是同一件事”时
    两条都留着。这是 46 号最关键的一道：只看相似度会错杀。

    注意：金额记忆（定投 5 块 vs 10 块）现在按 53 号走——同槽位新值
    更新旧条，不留两条（见 tests/test_dedup_slots_53.py）；这条 46 号
    守的是非金额的“数字不同就是不同事实”，两条规矩不冲突。"""
    store = make_fake_store(tmp_path)
    _install_fake_embedder(
        store,
        {
            "我每天喝 2 杯咖啡。": 0.0,
            "我每天喝 3 杯咖啡。": 0.0,
        },
    )
    _install_fake_llm(
        store,
        lambda system, user: {"choice": "none"}
        if "去重" in system
        else None,
    )
    store.add_memory_text("我每天喝 2 杯咖啡。")
    result = store.add_memory_text("我每天喝 3 杯咖啡。")
    assert result.get("duplicate") is not True
    assert _texts(store) == [
        "我每天喝 2 杯咖啡。",
        "我每天喝 3 杯咖啡。",
    ]


def test_46_extraction_path_dedups_same_fact(tmp_path):
    """46 号：提取路径也走去重——两轮换说法说同一件事（喜欢喝冰美式 /
    喜欢冰美式），库里只留一条。"""
    store = make_fake_store(tmp_path)
    _install_fake_embedder(
        store,
        {
            # 键是提取后的文本：假提取会把句末“。”去掉。
            "我喜欢喝冰美式": 0.0,
            "我喜欢冰美式": 0.0,
        },
    )
    _install_fake_llm(
        store,
        lambda system, user: {"choice": _first_candidate_id(user)}
        if "去重" in system
        else {
            "scores": [
                {"id": match.group(1), "score": 0.9}
                for match in re.finditer(r"id=(\S+?)：", user)
            ]
        }
        if "价值" in system
        else None,
    )
    store.add_turn(_round("我喜欢喝冰美式。", "记下了。"))
    store.add_turn(_round("我喜欢冰美式。", "记下了。"))
    store.flush_pending()
    # 假提取会把句末“。”去掉，存的是“我喜欢喝冰美式”（无句号）。
    assert _texts(store) == ["我喜欢喝冰美式"]


def test_46_judge_unavailable_keeps_both(tmp_path):
    """46 号：模型判不出来（没装假判）时只做精确匹配，近似的照常留——
    宁漏勿错，漏合并只是多一条记忆，错合并是丢一条事实。"""
    store = make_fake_store(tmp_path)
    _install_fake_embedder(
        store,
        {
            "助手叫十二。": 0.0,
            "助手的名字是十二。": 0.0,
        },
    )
    store.add_memory_text("助手叫十二。")
    store.add_memory_text("助手的名字是十二。")
    assert _texts(store) == ["助手叫十二。", "助手的名字是十二。"]


# ---------- 43 号：没有价值关卡 ----------

def test_43_value_gate_drops_junk_keeps_facts(tmp_path):
    """43 号：价值关卡——提取出的两条里，闲话（今天天气真好，0.2 分）
    不进库，有价值的（喜欢喝冰美式，0.9 分）进了。"""
    store = make_fake_store(tmp_path)

    def _handler(system, user):
        if "价值" not in system:
            return None
        scores = []
        for match in re.finditer(r"id=(\S+?)：([^\n]*)", user):
            mem_id, text = match.group(1), match.group(2)
            score = 0.9 if "冰美式" in text else 0.2
            scores.append({"id": mem_id, "score": score})
        return {"scores": scores}

    _install_fake_llm(store, _handler)
    store.add_turn(_round("我喜欢喝冰美式。我喜欢今天天气真好。", "记下了。"))
    store.add_turn(_round("嗯。", "好的。"))
    store.flush_pending()
    # 假提取会把句末“。”去掉，存的是“我喜欢喝冰美式”（无句号）。
    assert _texts(store) == ["我喜欢喝冰美式"]


def test_43_value_gate_fails_closed_without_judge(tmp_path):
    """43 号：模型判不出来（没装假判）时全留——关卡失败关闭，别把真事实
    拦在外面。提取提示词是第一道防线，判不出来不拦路。"""
    store = make_fake_store(tmp_path)
    store.add_turn(_round("我喜欢喝冰美式。我喜欢今天天气真好。", "记下了。"))
    store.add_turn(_round("嗯。", "好的。"))
    store.flush_pending()
    # 假提取会把句末“。”去掉，存的都是无句号的文本。
    assert _texts(store) == ["我喜欢喝冰美式", "我喜欢今天天气真好"]


def test_43_threshold_is_moderate_default():
    """43 号：档位是适中的默认 0.5（中点）——稳定事实/明确偏好通常
    大于等于 0.7，一次性请求/闲聊通常小于等于 0.3；拿不准的往低打，
    所以 0.5 不会把真事实拦在外面。BOOS 看了可以调（见第 3 批报告）。"""
    assert VALUE_THRESHOLD == 0.5
    assert DEDUP_SIMILARITY_MIN == 0.70


# ---------- 4 号：阈值 ----------

def test_04_threshold_is_measured_value():
    """4 号：阈值是量出来的 0.70，不是拍脑袋的 0.56——2026-11-12 实测
    （18 对整句，bge-small-zh-v1.5）：同话题 0.757–0.907，跨话题
    0.411–0.763；0.70 拦住 7/8 个跨话题、同话题 10/10 放行。"""
    assert CORRECTION_TOPIC_MIN_SIMILARITY == 0.70


def _store_with_scifi_and_go(tmp_path):
    """造一个有两条记忆的假 store：科幻电影 + Go 写代码。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("用户喜欢看科幻电影。")
    store.add_memory_text("我现在喜欢用 Go 写代码。")
    return store


def test_04_cross_topic_at_065_blocked(tmp_path):
    """4 号：跨话题相似度 0.65（旧阈值 0.56 会放行，新阈值 0.70 拦下）——
    模型判错挑中了 Go，安全网拦下，不换、等 BOOS 挑。"""
    store = _store_with_scifi_and_go(tmp_path)
    go_id = store.get_all()[1]["id"]
    _install_fake_llm(
        store,
        lambda system, user: {"choice": go_id}
        if "纠错" in system
        else None,
    )
    _install_fake_embedder(
        store,
        {
            "我喜欢看纪录片。": 0.0,
            "用户喜欢看科幻电影。": 0.64,
            "我现在喜欢用 Go 写代码。": 0.86,
        },
    )
    result = store.add_turn(_round("不对，我喜欢看纪录片。", "明白，我更新一下。"))
    correction = result["correction"]
    assert correction["status"] == "ambiguous"
    assert correction["layer"] == "judge_blocked"
    assert correction["resolution"] == "judge_blocked"
    assert correction["memory_id"] is None
    assert [item["text"] for item in store.get_all()] == [
        "用户喜欢看科幻电影。",
        "我现在喜欢用 Go 写代码。",
    ]


def test_04_same_topic_at_080_passes(tmp_path):
    """4 号：同话题相似度 0.80（大于等于 0.70）——模型挑中科幻，
    安全网放行，正常换。"""
    store = _store_with_scifi_and_go(tmp_path)
    scifi_id = store.get_all()[0]["id"]
    _install_fake_llm(
        store,
        lambda system, user: {"choice": scifi_id}
        if "纠错" in system
        else {"rule": None},
    )
    _install_fake_embedder(
        store,
        {
            "我喜欢看纪录片。": 0.0,
            "用户喜欢看科幻电影。": 0.64,
            "我现在喜欢用 Go 写代码。": 1.36,
        },
    )
    result = store.add_turn(_round("不对，我喜欢看纪录片。", "明白，我更新一下。"))
    correction = result["correction"]
    assert correction["status"] == "updated"
    assert correction["layer"] == "judge"
    assert correction["memory_id"] == scifi_id
    assert [item["text"] for item in store.get_all()] == [
        "我喜欢看纪录片。",
        "我现在喜欢用 Go 写代码。",
    ]


# ---------- 26 号：后台存记忆 ----------

def test_26_save_turn_queues_and_flush_stores(tmp_path):
    """26 号：save_turn 把本轮放进 outbox 就返回（queued=True），
    不等存完；flush_pending 后记忆才落库——回答先回、记忆后存。"""
    store = make_fake_store(tmp_path)
    result = store.save_turn(_round("我喜欢喝冰美式。", "记下了。"))
    assert result.get("queued") is True
    assert result.get("turn_id")
    store.flush_pending()
    # 假提取会把句末“。”去掉，存的是“我喜欢喝冰美式”（无句号）。
    assert "我喜欢喝冰美式" in _texts(store)


def test_26_same_turn_id_stored_once(tmp_path):
    """26 号：同一 turn_id 只存一次——重复提交直接丢掉，不会存两遍。"""
    store = make_fake_store(tmp_path)
    messages = _round("我喜欢喝冰美式。", "记下了。")
    store.save_turn(messages, turn_id="turn-1")
    store.save_turn(messages, turn_id="turn-1")
    store.flush_pending()
    # 假提取会把句末“。”去掉，存的是“我喜欢喝冰美式”（无句号）。
    assert _texts(store) == ["我喜欢喝冰美式"]


def test_26_correction_turn_stays_synchronous(tmp_path):
    """26 号：纠错的轮次走同步——结果要立刻回给页面展示，不能等后台。
    save_turn 直接返回纠错结果（有 correction 键、没有 queued 键）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢看科幻电影。")
    store.add_memory_text("我喜欢看科幻小说。")
    result = store.save_turn(_round("不对，我喜欢看纪录片。", "明白，我更新一下。"))
    assert result.get("queued") is not True
    assert "correction" in result


# ---------- 46 号补：身份 upsert 也要去重（真调 46 场景 1 的根因） ----------

def test_46_identity_upsert_dedups_extraction_rephrase(tmp_path):
    """46 号补：提取先把“助手叫十二”改写成“用户给助手取名为「十二」”
    存下了，之后身份 upsert 不能再新存一条——要原地更新成槽位记忆。
    （触发词认不出改写过的文本，所以 upsert 里加了一道近重复检查。）"""
    store = make_fake_store(tmp_path)
    _install_fake_embedder(
        store,
        {
            "Agent 的名字是十二，你叫十二": 0.0,
            "用户给助手取名为「十二」。": 0.0,
        },
    )
    _install_fake_llm(
        store,
        lambda system, user: {"choice": _first_candidate_id(user)}
        if "去重" in system
        else None,
    )
    store.add_memory_text("用户给助手取名为「十二」。")
    result = store._upsert_slot_memory(
        "agent_name", "十二", "Agent 的名字是十二，你叫十二"
    )
    assert result["status"] == "updated"
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["text"] == "Agent 的名字是十二，你叫十二"
    assert (items[0].get("metadata") or {}).get("slot") == "agent_name"


def test_46_name_scenario_three_wordings_one_memory(tmp_path):
    """46 号补：B 组场景 1 原样——“助手叫十二。/ 你的名字是十二。/
    以后叫你十二。”三轮说完，库里只留一条（“助手叫”现在也是身份
    触发词，第一轮就认出身份、存成槽位记忆，后两轮认出同值就不动）。"""
    store = make_fake_store(tmp_path)
    store.add_turn(_round("助手叫十二。", "记下了。"))
    store.add_turn(_round("你的名字是十二。", "记下了。"))
    store.add_turn(_round("以后叫你十二。", "记下了。"))
    store.flush_pending()
    assert _texts(store) == ["Agent 的名字是十二，你叫十二"]
