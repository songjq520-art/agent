# tests/test_fragment_correction.py（碎片纠正：31 项之外的新问题）
# 作用：把“纠正碎片先补成整句再改”钉死在测试里，全部离线模拟，不调
# 真实 API、不花钱；真调在 /tmp 外面另做（实调脚本）。
# 背景：第 3 批真调发现，用户说“不对，是绿色”想把“我喜欢蓝色”改成
# 绿色，模型判断对了、蓝色也没删错，但纠正被安全网拦住了——“是绿色”
# 这个碎片和整句“我喜欢蓝色”相似度只有 0.598，低于 0.70，最后只能
# 问 BOOS 选、没直接改成绿色。整句“不对，我喜欢绿色”就能直接改对，
# 所以问题只出在“碎片”上。
# 修法：纠正前先把碎片补成整句（结合旧记忆），再过安全网、再改；
# 补不出来、拿不准时走老路（问 BOOS），绝不硬编一句去改。
# 假判说明：补句走 _install_fake_llm 装的假 DeepSeek——handler 按系统
# 提示分派，测试只验“补完之后的代码走没走对”，不验模型本身补得准不准；
# 模型补得准不准靠真调 DeepSeek 验。
# 注意：add_turn 层面的碎片纠正离线测不到——假 store 的窗口检索对
# “是绿色”找不到“我喜欢蓝色”（没有两字窗口重合），候选找不着就走
# not_found；真 mem0 的语义检索能找着（第 3 批实调已证实）。所以单测
# 直接调 correct_memory（和网页端/待确认流程调的是同一个入口），
# 端到端碎片流程靠实调验。

import json
import math
from types import SimpleNamespace

import pytest

from conftest import make_fake_store


# ---------- 假对象 ----------

def _install_fake_llm(store, handler):
    """给假 store 装一个假 DeepSeek 客户端（和 test_batch3_memory 一路）。

    handler(system_prompt, user_prompt) -> dict（模型回的 JSON）或 None
    （回的不是 JSON / 判不出来）。每次调用都记进 calls（每项
    {"system": 系统提示, "user": 用户提示}），供测试断言“某路判
    调了没调”。
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
    """给假 store 装一个假本地 embedding 模型（安全网用）。

    angles：{文本: 夹角（弧度）}——embed 返回二维单位向量
    [cos(夹角), sin(夹角)]，于是两段文本的余弦相似度就是
    cos(夹角差)，测试能精确控制“话题像不像”。没在 angles 里的文本
    默认夹角 0。
    """
    class _FakeEmbedder:
        def embed(self, text, memory_action=None):
            angle = angles.get(text, 0.0)
            return [math.cos(angle), math.sin(angle)]

    store._memory.embedding_model = _FakeEmbedder()


def _texts(store):
    """库里所有记忆文本（按存入顺序）。"""
    return [item["text"] for item in store.get_all()]


def _round(user_text, assistant_text="好的。"):
    """一轮对话（供 add_turn 用）。"""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


# 假 embedding 夹角：旧“喜欢蓝色”和补好的“喜欢绿色”很像（0.3 弧度，
# 余弦约 0.955，过 0.70）；碎片“是绿色”和旧句不像（1.0 弧度，余弦
# 约 0.540，不过 0.70）——正好复现第 3 批实调里 0.598 被拦住的情况。
_ANGLES = {
    "我喜欢蓝色。": 0.0,
    "我喜欢绿色。": 0.3,
    "是绿色。": 1.0,
    "是蓝色。": 1.0,
    "我喜欢红色。": 0.5,
}


def _handler(expanded=None, expand_returned=True):
    """假模型：补句按 expand_returned/expanded 回，提炼规则回 None。

    expanded=None 且 expand_returned=True 时回 {"expanded": None}
    （模型说补不出来）；expand_returned=False 时回 None（回的不是 JSON）。
    """
    def _handle(system, user):
        if "补句" in system:
            if not expand_returned:
                return None
            return {"expanded": expanded}
        if "提炼助手" in system:
            return {"rule": None}
        if "撤回意图" in system:
            return {"retract": False}
        return None

    return _handle


# ---------- 碎片纠正 ----------

def test_fragment_expands_and_updates_single_layer(tmp_path):
    """碎片纠正主路：单候选 + 安全网——“是绿色”补成“我喜欢绿色”才过网。

    不补的话相似度约 0.540（<0.70）会被拦住；补完约 0.955 才过网。
    断言：直接改好、库里是整句、补句确实调了。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    calls = _install_fake_llm(store, _handler(expanded="我喜欢绿色。"))
    _install_fake_embedder(store, _ANGLES)
    # old_text 用“蓝色。”：关键词能捞回“我喜欢蓝色。”（单候选），
    # 走的正是第 3 批实调里被拦住的那条路（single + 安全网）。
    result = store.correct_memory("蓝色。", "是绿色。")
    assert result.get("status") == "updated"
    assert result.get("new_text") == "我喜欢绿色。"
    assert "我喜欢绿色。" in _texts(store)
    assert not any("蓝色" in text for text in _texts(store))
    assert any("补句" in call["system"] for call in calls)


def test_fragment_expands_and_updates_exact_layer(tmp_path):
    """碎片纠正 exact 层：old_text 和库里原文对上时，补句也得跑——
    不然库里会被存成“是绿色。”这个碎片本身。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    calls = _install_fake_llm(store, _handler(expanded="我喜欢绿色。"))
    _install_fake_embedder(store, _ANGLES)
    result = store.correct_memory("我喜欢蓝色。", "是绿色。")
    assert result.get("status") == "updated"
    assert result.get("new_text") == "我喜欢绿色。"
    assert "我喜欢绿色。" in _texts(store)
    assert not any(text == "是绿色。" for text in _texts(store))
    assert any("补句" in call["system"] for call in calls)


def test_full_sentence_correction_skips_expansion(tmp_path):
    """回归：完整的一句（“我喜欢绿色。”）不触发补句——补句只对碎片跑，
    正常纠正一分钱不加、多一次调用也不加。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    calls = _install_fake_llm(store, _handler(expanded="我喜欢绿色。"))
    _install_fake_embedder(store, _ANGLES)
    result = store.correct_memory("我喜欢蓝色。", "我喜欢绿色。")
    assert result.get("status") == "updated"
    assert "我喜欢绿色。" in _texts(store)
    assert not any("补句" in call["system"] for call in calls)


def test_fragment_expand_returns_null_falls_back(tmp_path):
    """拿不准的碎片走老路：模型说补不出来（expanded=null），就不硬编——
    安全网拦住碎片、改问 BOOS，旧记忆一动不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    _install_fake_llm(store, _handler(expanded=None))
    _install_fake_embedder(store, _ANGLES)
    result = store.correct_memory("蓝色。", "是绿色。")
    assert result.get("status") == "ambiguous"
    assert result.get("layer") == "judge_blocked"
    assert "我喜欢蓝色。" in _texts(store)
    assert not any("绿色" in text for text in _texts(store))


def test_fragment_expand_wrong_value_rejected(tmp_path):
    """补句校验：模型补好的句子不含碎片里的值（“是绿色”补成“喜欢红色”），
    补句作废、走老路——安全网拦住、改问 BOOS，旧记忆不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    _install_fake_llm(store, _handler(expanded="我喜欢红色。"))
    _install_fake_embedder(store, _ANGLES)
    result = store.correct_memory("蓝色。", "是绿色。")
    assert result.get("status") == "ambiguous"
    assert result.get("layer") == "judge_blocked"
    assert "我喜欢蓝色。" in _texts(store)
    assert not any("红色" in text for text in _texts(store))


def test_fragment_expanded_equals_old_is_unchanged(tmp_path):
    """补完和旧的一样 = 不用改（“是蓝色”纠“我喜欢蓝色”），报 unchanged，
    别谎称改好了。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    _install_fake_llm(store, _handler(expanded="我喜欢蓝色。"))
    _install_fake_embedder(store, _ANGLES)
    result = store.correct_memory("蓝色。", "是蓝色。")
    assert result.get("status") == "unchanged"
    assert _texts(store) == ["我喜欢蓝色。"]


def test_fragment_vague_value_skips_model(tmp_path):
    """含糊指代不补句：“是那个”没有内容可补，硬补只能把旧值原样抄回来
    （实调里补成“我喜欢那个蓝色。”还顺带提炼一条怪规则）。值只是
    “那个/这个/它”时连模型都不调，直接走老路问 BOOS，旧记忆不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    calls = _install_fake_llm(store, _handler(expanded="我喜欢那个蓝色。"))
    _install_fake_embedder(store, {**_ANGLES, "是那个。": 1.0, "是它。": 1.0})
    result = store.correct_memory("蓝色。", "是那个。")
    assert result.get("status") == "ambiguous"
    assert result.get("layer") == "judge_blocked"
    assert "我喜欢蓝色。" in _texts(store)
    assert not any("补句" in call["system"] for call in calls)


def test_fragment_vague_it_skips_model(tmp_path):
    """含糊指代“是它”也一样：不调模型、走老路，旧记忆不动。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢蓝色。")
    calls = _install_fake_llm(store, _handler(expanded="我喜欢它。"))
    _install_fake_embedder(store, {**_ANGLES, "是那个。": 1.0, "是它。": 1.0})
    result = store.correct_memory("蓝色。", "是它。")
    assert result.get("status") == "ambiguous"
    assert result.get("layer") == "judge_blocked"
    assert "我喜欢蓝色。" in _texts(store)
    assert not any("补句" in call["system"] for call in calls)


def test_bare_retraction_still_retraction(tmp_path):
    """回归：34 号撤回逻辑没动——“不不不”还是撤回，不走纠正；
    补句一次也不调。"""
    store = make_fake_store(tmp_path)
    calls = _install_fake_llm(store, _handler(expanded="我喜欢绿色。"))
    store.add_turn(_round("我喜欢蓝色。", "记下了。"))
    result = store.add_turn(_round("不不不"))
    assert result.get("retracted") is True
    assert result.get("dropped") == 1
    assert not any("蓝色" in text for text in _texts(store))
    store.flush_pending()
    assert not any("蓝色" in text for text in _texts(store))
    assert not any("补句" in call["system"] for call in calls)
