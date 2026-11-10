# tests/test_mem0_regression.py（提取规则 + 召回回归测试）
# 作用：守住两条已知的回归线——
#   1. 提取规则（试跑 Test 3 的 A–D）：条件句、假设句、闲聊不能被存成既定事实；
#   2. 召回（v1 的 bug）：四种问法都要能召回名字记忆，不知道时不注入任何东西。
# 离线部分用 FakeMemory（见 tests/conftest.py），默认运行，不联网、不产生费用。
# 真实 API 部分默认跳过：只有 RUN_REAL_API_TESTS=1 且有 DEEPSEEK_API_KEY，
# 且能 import mem0 时才跑；mem0 只装在 ~/workspace/mem0-spike/.venv 里，
# 要用那个 venv 跑（不要装进项目 .venv），且 import 失败时也要跳过、不要报错。
# Test 2 的五句话原文见 ~/workspace/mem0-spike/test2_extraction.py。

import os

import pytest

from main import _memory_block
from memory.mem0_store import CUSTOM_INSTRUCTIONS, Mem0Store
from conftest import FakeMemory, make_fake_store

# 召回回归用的名字记忆文本。特意这样写：
#   - 含“我叫 / 昵称 / 称呼”，四种问法都能通过语义或关键词信号命中；
#   - 不含“的”字——“我的 Agent 辅助模型是哪个？”里的关键词“我的”
#     就撞不上它，“不知道”回归才不会被关键词补充信号误捞。
NAME_MEMORY_TEXT = "我叫小宋，昵称小宋，请称呼我小宋。"

# 四种问法（v1 里只有“你应该怎么称呼我”能召回，其余三种都失败）。
NAME_QUESTIONS = [
    "你记得我叫什么吗？",
    "我叫什么？",
    "我的昵称是什么？",
    "你应该怎么称呼我？",
]

# Test 2 的五句对话（试跑原文），每句的核心事实都要存下来。
TEST2_DIALOGUE = [
    "我叫小宋。",
    "我目前在圆通工作，但是以后不想一直做物流。",
    "我的长期目标是换一份更自由的工作，以后也希望有创业的机会。",
    "我希望 AI 和英语帮助我实现这个目标。",
    "我的 Agent 项目采用 Kimi 作为日常模型，DeepSeek 作为其他任务的辅助模型。",
]


# ---------- 提取规则回归（试跑 Test 3 的 A–D） ----------


def test_extraction_rule_a_present_job_is_stored(tmp_path):
    """A. “我目前在圆通工作，做快递员。”→ 应存：用户明确说的现在时事实。

    Plan C：本轮先待定，flush 后才提交。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "我目前在圆通工作，做快递员。"}])
    store.flush_pending()
    texts = store._memory.all_texts()
    assert len(texts) == 1
    assert "圆通" in texts[0] and "快递员" in texts[0]


def test_extraction_rule_b_conditional_is_not_stored(tmp_path):
    """B. “如果以后我去澳大利亚工作，可能会更好。”→ 不应存为既定事实：
    条件句，说的是“如果”，不是已经发生的事。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "如果以后我去澳大利亚工作，可能会更好。"}])
    assert store._memory.all_texts() == []
    assert not any("澳大利亚" in text for text in store._memory.all_texts())


def test_extraction_rule_c_small_talk_is_not_stored(tmp_path):
    """C. “今天天气真好。”→ 不存：闲聊/一时状态，不是长期事实。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "今天天气真好。"}])
    assert store._memory.all_texts() == []


def test_extraction_rule_d_hypothetical_is_not_stored(tmp_path):
    """D. “假设我以后学 Rust，会不会有帮助？”→ 不存“学 Rust”这个事实：
    假设句 + 问句，既不是既定事实，也不是用户明确的决定。"""
    store = make_fake_store(tmp_path)
    store.add_turn([{"role": "user", "content": "假设我以后学 Rust，会不会有帮助？"}])
    assert store._memory.all_texts() == []
    assert not any("Rust" in text for text in store._memory.all_texts())


def test_extraction_rules_all_four_together(tmp_path):
    """A–D 一起跑：A 存 1 条，B/C/D 一条也不存。

    Plan C：前三轮在第四轮调 add_turn 时提交，第四轮在 flush 时提交。"""
    store = make_fake_store(tmp_path)
    for text in [
        "我目前在圆通工作，做快递员。",
        "如果以后我去澳大利亚工作，可能会更好。",
        "今天天气真好。",
        "假设我以后学 Rust，会不会有帮助？",
    ]:
        store.add_turn([{"role": "user", "content": text}])
    store.flush_pending()
    texts = store._memory.all_texts()
    assert len(texts) == 1
    assert "圆通" in texts[0]


def test_custom_instructions_keep_never_store_inference():
    """规则文本回归：CUSTOM_INSTRUCTIONS 里必须有“NEVER store an inference”
    之类关键句，防止以后有人把规则改没了（提示词层面的服从不是机制保证，
    文本丢了真实提取就会变质——见 mem0_store.py 的注释）。"""
    assert "NEVER store an inference" in CUSTOM_INSTRUCTIONS
    assert "conditionals" in CUSTOM_INSTRUCTIONS
    assert "hypotheticals" in CUSTOM_INSTRUCTIONS
    assert "small talk" in CUSTOM_INSTRUCTIONS
    assert "ONLY store information the USER explicitly stated" in CUSTOM_INSTRUCTIONS


# ---------- Test 2 五句对话回归 ----------


def test_test2_five_turns_all_core_facts_stored(tmp_path):
    """Test 2 五句对话回归：五句话（试跑原文）各调一次 add，
    每句的核心事实都要存下来。

    Plan C：前四句在下一轮调 add_turn 时提交，第五句在 flush 时提交；
    第 1 句“我叫小宋”走身份槽位（upsert），其余四句走提取。"""
    store = make_fake_store(tmp_path)
    for sentence in TEST2_DIALOGUE:
        store.add_turn([{"role": "user", "content": sentence}])
    store.flush_pending()
    texts = store._memory.all_texts()
    assert len(texts) == 5
    joined = "\n".join(texts)
    assert "小宋" in joined  # 第 1 句：名字
    assert "圆通" in joined  # 第 2 句：在圆通工作
    assert "长期目标" in joined and "创业" in joined  # 第 3 句：换工作、创业机会
    assert "AI" in joined and "英语" in joined  # 第 4 句：AI 和英语帮忙
    assert "Kimi" in joined and "DeepSeek" in joined  # 第 5 句：日常模型 Kimi、辅助 DeepSeek


def test_test2_turn2_future_clause_does_not_block_present_fact(tmp_path):
    """Test 2 第 2 句里“以后不想一直做物流”是将来的想法，但“目前在圆通工作”
    是现在时事实：整句存下来，核心事实“圆通”在里面。"""
    memory = FakeMemory()
    memory.add([{"role": "user", "content": TEST2_DIALOGUE[1]}], infer=True)
    assert memory.all_texts() == ["我目前在圆通工作，但是以后不想一直做物流"]


# ---------- 4 种问法召回回归（v1 bug） ----------


def test_four_ways_to_ask_name_all_recall(tmp_path):
    """4 种问法召回回归：记忆里只有名字那条，四种问法
    “你记得我叫什么吗？”“我叫什么？”“我的昵称是什么？”“你应该怎么称呼我？”
    都要能召回（经 Mem0Store.search，阈值 0.4，语义或关键词信号都算）。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text(NAME_MEMORY_TEXT)
    for question in NAME_QUESTIONS:
        found = store.search(question)
        assert found, f"问法 {question!r} 没召回名字记忆"
        assert any("小宋" in item["text"] for item in found), question


def test_four_ways_fake_search_scores_vary_but_adapter_recovers(tmp_path):
    """补充：FakeMemory 的语义打分本身有高有低（有的问法分数不到阈值），
    但 Mem0Store 的关键词补充信号能兜底——这正是适配层留两路信号的原因。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text(NAME_MEMORY_TEXT)
    for question in NAME_QUESTIONS:
        raw = store._memory.search(question)
        assert raw["results"], question
        # 至少有一路能把它带回来（语义过阈值，或关键词补充命中）。
        found = store.search(question)
        assert found, question


# ---------- “不知道”回归（负向） ----------


def test_unknown_question_returns_nothing(tmp_path):
    """“不知道”回归：库里只有“小宋”那条，问“我的 Agent 辅助模型是哪个？”
    时 search 返回空——不相关的记忆不能被当成答案。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text(NAME_MEMORY_TEXT)
    assert store.search("我的 Agent 辅助模型是哪个？") == []


def test_memory_block_empty_returns_empty_string():
    """“不知道”回归（续）：没召回记忆时 main._memory_block([]) 返回 ""，
    不注入任何东西——模型只能按“不知道”回答，不能编一个答案。"""
    assert _memory_block([]) == ""


def test_memory_block_with_memories_lists_texts():
    """对照：召回了记忆时提示块里列出记忆文本（mem0 风格的 dict，一样能拼）。"""
    block = _memory_block([{"id": "fake-1", "text": "我叫小宋。", "score": 0.9, "source": "semantic"}])
    assert "我叫小宋" in block
    assert "仅供参考" in block


def test_memory_block_tells_model_to_answer_important_first():
    """回答方式：提示块里要教模型先说最重要的、简短说，别把来源和免责
    都展开倒出来（BOOS 2026-11-10：十二回答太长太绕，要像十一一样挑重要的说）。"""
    block = _memory_block([{"id": "fake-1", "text": "我叫小宋。", "score": 0.9, "source": "semantic"}])
    assert "先说最重要的结论" in block
    assert "不用每次都展开解释" in block


def test_memory_block_says_unverified_info_is_from_prior_knowledge():
    """回答方式：没联网核实过的信息（店名、价格、排名）要说明是凭已有
    知识说的、没现查核实，别让人以为是查过的（BOOS 2026-11-10：十二
    没联网却报了火锅店名和价格）。"""
    block = _memory_block([{"id": "fake-1", "text": "我叫小宋。", "score": 0.9, "source": "semantic"}])
    assert "没现查核实" in block
    assert "凭已有知识说的" in block


# ---------- 真实 API 回归测试（默认跳过） ----------


def _real_mem0_available():
    """真实 mem0 是否能跑：只有显式开启 + 有 Key + 能 import mem0 才算。
    mem0 没装在项目 .venv 里，import 失败时返回 False（测试跳过，不报错）。"""
    if os.environ.get("RUN_REAL_API_TESTS") != "1":
        return False
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return False
    try:
        import mem0  # noqa: F401
    except ImportError:
        return False
    return True


RUN_REAL_MEM0 = _real_mem0_available()

_REAL_SKIP_REASON = (
    "真实 mem0 测试需要 RUN_REAL_API_TESTS=1 且 DEEPSEEK_API_KEY，"
    "且 mem0 能 import（mem0 只装在 ~/workspace/mem0-spike/.venv 里，"
    "请用那个 venv 跑：~/workspace/mem0-spike/.venv/bin/python -m pytest "
    "tests/test_mem0_regression.py -q -k real）；未开启时跳过，不产生费用"
)
_REAL_SKIP = pytest.mark.skipif(not RUN_REAL_MEM0, reason=_REAL_SKIP_REASON)


def _real_store(tmp_path):
    """建一个真实的 Mem0Store：Chroma/历史/版本文件全在 tmp_path 里，
    不碰真实数据目录。建的时候会加载本地 embedding 模型（约 25 秒）。"""
    return Mem0Store(
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )


@_REAL_SKIP
def test_real_mem0_four_ways_to_ask_name_all_recall(tmp_path):
    """真实 API：4 种问法都要能召回“小宋”那条记忆（v1 bug 的真实回归）。

    记忆用 infer=False 原文存入（不经过提取，排除提取的干扰）；
    召回走真实语义检索 + 阈值 0.4。会调一次 embedding（本地），不调 LLM。
    """
    store = _real_store(tmp_path)
    store.add_memory_text(NAME_MEMORY_TEXT)
    for question in NAME_QUESTIONS:
        found = store.search(question)
        assert found, f"真实检索：问法 {question!r} 没召回名字记忆"
        assert any("小宋" in item["text"] for item in found), question


@_REAL_SKIP
def test_real_mem0_extraction_rules_abcd(tmp_path):
    """真实 API：A–D 提取规则（试跑 Test 3 的真实回归）。

    每句各调一次 add_turn（infer=True，真实调 DeepSeek 提取，会产生少量费用、
    每句几秒）。A 应存下“圆通/快递员”；B/C/D 不应存下“澳大利亚 / Rust / 天气”
    相关的既定事实。
    """
    store = _real_store(tmp_path)
    cases = [
        "我目前在圆通工作，做快递员。",
        "如果以后我去澳大利亚工作，可能会更好。",
        "今天天气真好。",
        "假设我以后学 Rust，会不会有帮助？",
    ]
    for text in cases:
        store.add_turn([{"role": "user", "content": text}])
    # Plan C：前三句在下一轮调 add_turn 时提交，最后一句在 flush 时提交。
    store.flush_pending()
    texts = [item["text"] for item in store.get_all()]
    joined = "\n".join(texts)
    assert "圆通" in joined, f"A 没存下来，库里只有：{texts}"
    assert "澳大利亚" not in joined, f"B 被存成了事实：{texts}"
    assert "Rust" not in joined, f"D 被存成了事实：{texts}"
    assert not any("天气" in text for text in texts), f"C 被存成了事实：{texts}"
