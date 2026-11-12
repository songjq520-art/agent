# tests/test_batch4_money_time.py（第 4 批：27 号金额结构化字段 + 33 号记忆时间）
# 27 号：金额类记忆贴结构化字段（金额/币种/周期/对象），回答时标在每条后面，
#   模型一眼分得清“每月 5 美元工具预算”和“每天 5 块定投”这两笔钱。
# 33 号：mem0 2.2.1 读回来时把 created_at 提到条目的顶层、同时从返回的
#   metadata 里剔掉——只看 metadata 永远找不到时间，面板上 14 条全是“未知时间”。
#   根因在读的那头，不在写的那头（写的时候每条都带了显式时间，mem0 存的时候
#   也留着）。修法：规整时 metadata 里没有就到条目顶层找。
# 全部离线模拟，不调真实 API、不花钱；真调在 /tmp 外面另做。

from datetime import datetime, timezone

from conftest import FakeMemory, make_fake_store

from main import _memory_block
from memory.mem0_store import (
    _extract_money_metadata,
    _normalize_metadata,
    _normalize_money_metadata,
    _parse_created_at,
)


class PromotingFakeMemory(FakeMemory):
    """按真实 mem0 2.2.1 的读/改形状返回（见 venv 里 mem0/memory/main.py）：
    - get/get_all/search：created_at 提到条目顶层（MemoryItem 的顶层字段），
      metadata 里剔掉（core_and_promoted_keys）；
    - update：metadata 是合并进原 payload（不是替换），created_at 保留原值
      （见 _update_memory：new_metadata 从原 payload 复制、再 update）。
    conftest 的 FakeMemory 这两处是“够测流程”的近似，和真实行为不一样，
    33 号和 27 号的回归测试必须按真实形状来，不然测不到 bug。
    """

    def _promote(self, item):
        out = dict(item)
        meta = dict(out.get("metadata") or {})
        created = meta.pop("created_at", None)
        meta.pop("updated_at", None)
        out["metadata"] = meta
        if isinstance(created, str) and created:
            out["created_at"] = created
        return out

    def search(self, query, filters=None, top_k=5):
        raw = super().search(query, filters=filters, top_k=top_k)
        return {"results": [self._promote(it) for it in raw.get("results", [])]}

    def get_all(self, filters=None):
        raw = super().get_all(filters=filters)
        return {"results": [self._promote(it) for it in raw.get("results", [])]}

    def update(self, memory_id, text=None, metadata=None):
        # 真实 mem0 的 update 是合并：先把新 metadata 盖进原 payload，
        # 再调父类的替换式 update（效果等同于合并）。
        if metadata is not None:
            for item in self._items:
                if item["id"] == memory_id:
                    merged = dict(item.get("metadata") or {})
                    merged.update(dict(metadata))
                    metadata = merged
                    break
        return super().update(memory_id, text=text, metadata=metadata)


class ChromaLikeFakeMemory(PromotingFakeMemory):
    """像 Chroma 那样校验 metadata：dict 值直接拒掉（ValueError）。

    真实 Chroma 只接受标量 metadata（str/int/float/bool/list/None），
    dict 值会炸掉整条 add/update。第 4 批真调用到过这个 bug：27 号
    第一版把 money 存成一个 dict，add_memory_text 在 Chroma 那层就
    ValueError，金额记忆根本没存进去——单元测试用的 FakeMemory 不
    校验，测不到。以后所有往库里写的 metadata 都得过这一关。
    """

    @staticmethod
    def _reject_dict_values(metadata):
        for key, value in (metadata or {}).items():
            if isinstance(value, dict):
                raise ValueError(
                    "Chroma 不接受 dict 值的 metadata：%s=%r" % (key, value)
                )

    def add(self, messages, user_id=None, infer=True, metadata=None):
        self._reject_dict_values(metadata)
        return super().add(messages, user_id=user_id, infer=infer, metadata=metadata)

    def update(self, memory_id, text=None, metadata=None):
        self._reject_dict_values(metadata)
        return super().update(memory_id, text=text, metadata=metadata)


def test_money_metadata_is_chroma_safe(tmp_path):
    """27 号回归（真调抓到的）：三条写路贴金额字段都不能炸 Chroma——
    直接存（记住）、提取后贴、纠正换新说法，都走扁平标量字段；
    读回来拼回的 money 字典还是对的。"""
    fake = ChromaLikeFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    # ① 直接存：记住“5 美元是每月工具预算”。
    store.add_memory_text("5 美元是每月工具预算。")
    # ② 提取路：聊两轮，提取出的记忆贴上金额字段。
    # “我的定投是每天 5 块买万家纳指”以“我的”开头，假对象按前缀规则能提出来；
    # 注意 Plan C 待定区：第一轮只进待定区，第二轮才提交。
    store.add_turn([
        {"role": "user", "content": "我的定投是每天 5 块买万家纳指。"},
        {"role": "assistant", "content": "记下了。"},
    ])
    store.add_turn([
        {"role": "user", "content": "今天天气不错。"},
        {"role": "assistant", "content": "是啊。"},
    ])
    # ③ 纠正路：把“5 美元是每月工具预算”改成“5 美元是每月工具预算，没变”——
    # 新说法还有金额，字段留着；再改成没金额的说法，字段要清掉。
    result = store.correct_memory("5 美元是每月工具预算。", "6 美元是每月工具预算。")
    assert result.get("status") == "updated"
    result = store.correct_memory("6 美元是每月工具预算。", "我不设预算了。")
    assert result.get("status") == "updated"
    # 库里原始 metadata 一律没有 dict 值（Chroma 能存的形状）。
    for item in fake._items:
        assert all(
            not isinstance(v, dict) for v in (item.get("metadata") or {}).values()
        ), item
    # 读回来：定投那条的 money 还在，改成没金额那条的 money 清掉了。
    # 注意假提取器会去掉句末的“。”，存下的文本是没有句号的。
    by_text = {i["text"]: i for i in store.get_all()}
    assert by_text["我的定投是每天 5 块买万家纳指"]["metadata"]["money"] == {
        "amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指",
    }
    assert by_text["我不设预算了。"]["metadata"]["money"] is None


# ---------- 27 号：金额提取器 ----------


def test_extract_money_metadata_cases():
    assert _extract_money_metadata("我每天投 5 块买万家纳指") == {
        "amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指",
    }
    assert _extract_money_metadata("5 美元是每月工具预算") == {
        "amount": 5, "currency": "USD", "period": "month", "target": "工具预算",
    }
    assert _extract_money_metadata("定投 10 块") == {
        "amount": 10, "currency": "CNY", "period": None, "target": None,
    }
    assert _extract_money_metadata("花了 100 元") == {
        "amount": 100, "currency": "CNY", "period": None, "target": None,
    }
    assert _extract_money_metadata("预算 $5") == {
        "amount": 5, "currency": "USD", "period": None, "target": "工具预算",
    }
    assert _extract_money_metadata("1 万块买房子") == {
        "amount": 10000, "currency": "CNY", "period": None, "target": "房子",
    }
    assert _extract_money_metadata("今天天气不错") is None
    assert _extract_money_metadata("我喜欢蓝色") is None


def test_normalize_money_metadata_rejects_bad_shape():
    assert _normalize_money_metadata({"amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指"}) == {
        "amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指",
    }
    assert _normalize_money_metadata({"amount": "5", "currency": "CNY"}) is None
    assert _normalize_money_metadata({"amount": 5, "currency": "EUR"}) is None
    assert _normalize_money_metadata({"amount": 5, "currency": "CNY", "period": "weekly"}) is None
    assert _normalize_money_metadata(None) is None
    assert _normalize_money_metadata("5 块") is None


# ---------- 27 号：写入口贴字段 ----------


def test_add_memory_text_attaches_money(tmp_path):
    fake = PromotingFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    store.add_memory_text("我每天投 5 块买万家纳指")
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["metadata"]["money"] == {
        "amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指",
    }


def test_add_memory_text_without_money_has_no_money_field(tmp_path):
    fake = PromotingFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    store.add_memory_text("我喜欢蓝色")
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["metadata"]["money"] is None


def test_extraction_attaches_money_metadata(tmp_path):
    # 提取那条路：add_turn 存下的新记忆，带金额的也要贴上字段。
    # “我的定投是每天 5 块买万家纳指”以“我的”开头，假对象按前缀规则能提出来。
    # 注意 Plan C 待定区：第一轮只进待定区，第二轮才提交，所以要调两次 add_turn。
    fake = PromotingFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    store.add_turn([{"role": "user", "content": "我的定投是每天 5 块买万家纳指。"}])
    store.add_turn([{"role": "user", "content": "今天天气怎么样？"}])
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["metadata"]["money"] == {
        "amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指",
    }


# ---------- 27 号：回答时标出来 ----------


def test_memory_block_renders_money_fields():
    block = _memory_block([
        {
            "text": "每月工具预算 5 美元",
            "metadata": {
                "source": "user", "confirmed": True,
                "money": {"amount": 5, "currency": "USD", "period": "month", "target": "工具预算"},
            },
        },
        {
            "text": "每天定投 5 块买万家纳指",
            "metadata": {
                "source": "user", "confirmed": True,
                "money": {"amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指"},
            },
        },
    ])
    assert "（金额：5 美元/月，对象：工具预算）" in block
    assert "（金额：5 元/天，对象：万家纳指）" in block
    assert "别把两条的数揉成一笔" in block


def test_memory_block_fallback_sentence_for_unstructured_money():
    # 没字段的老记忆：不标括号，但兜底句要有（提示词那句只当兜底）。
    block = _memory_block([
        {"text": "我每天投 5 块", "metadata": {"source": "inferred"}},
    ])
    assert "（金额：" not in block
    assert "别把两条的数揉成一笔" in block


def test_memory_block_no_money_no_money_sentence():
    block = _memory_block([
        {"text": "我喜欢蓝色", "metadata": {"source": "user", "confirmed": True}},
    ])
    assert "金额" not in block


# ---------- 33 号：时间丢了（读的那头） ----------


def test_created_at_survives_mem0_promotion(tmp_path):
    # 33 号核心回归：mem0 读回来把时间提到顶层，get_all 必须还能读到时间。
    # 以前只看 metadata，14 条全是“未知时间”。
    fake = PromotingFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    store.add_memory_text("我喜欢蓝色")
    items = store.get_all()
    assert len(items) == 1
    created = items[0]["metadata"]["created_at"]
    assert isinstance(created, str) and created
    # 时间是真的、不是写死的：和现在差不到 10 分钟。
    when = _parse_created_at(created)
    assert when is not None
    assert abs((datetime.now(timezone.utc) - when).total_seconds()) < 600


def test_search_returns_created_at(tmp_path):
    fake = PromotingFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    store.add_memory_text("我喜欢蓝色")
    results = store.search("我喜欢蓝色")
    assert results
    assert results[0]["metadata"]["created_at"]


def test_normalize_metadata_prefers_metadata_created_at():
    # metadata 里有时间时用 metadata 的，顶层只是兜底。
    meta = _normalize_metadata(
        {"source": "user", "created_at": "2026-11-10T10:00:00.000000Z"},
        {"created_at": "2026-11-11T10:00:00.000000Z"},
    )
    assert meta["created_at"] == "2026-11-10T10:00:00.000000Z"


def test_normalize_metadata_falls_back_to_top_level():
    meta = _normalize_metadata(
        {"source": "user"},
        {"created_at": "2026-11-11T10:00:00.000000Z"},
    )
    assert meta["created_at"] == "2026-11-11T10:00:00.000000Z"


def test_backfill_does_not_relabel_timestamps(tmp_path):
    # 33 号附带：backfill 以前只查 metadata、永远以为“缺时间”，每次启动
    # 把所有记忆的 created_at_source 改写成 "backfilled"（来源标错了）。
    # 顶层有时间就不该动——调一次 backfill，来源还是 written，返回 0。
    fake = PromotingFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    store.add_memory_text("我喜欢蓝色")
    assert store.backfill_timestamps() == 0
    items = store.get_all()
    assert items[0]["metadata"]["created_at_source"] == "written"
    # 再调一次也还是 0（幂等，不反复改写）。
    assert store.backfill_timestamps() == 0
    assert store.get_all()[0]["metadata"]["created_at_source"] == "written"


# ---------- 56 号：不知道变通——自然语言新指令先理解、先存能确定的部分 ----------

# 56 号 5 条新说法（和实测用过的不一样）：金额/币种/周期/对象都要抠对。
MONEY_56_PHRASES = [
    ("每天定投5刀的沪深300", {"amount": 5, "currency": "USD", "period": "day", "target": "沪深300"}),
    ("每月定投10刀买标普", {"amount": 10, "currency": "USD", "period": "month", "target": "标普"}),
    ("每周定投20块的沪深300", {"amount": 20, "currency": "CNY", "period": "week", "target": "沪深300"}),
    ("每天定投5美元的纳指", {"amount": 5, "currency": "USD", "period": "day", "target": "纳指"}),
    ("每周定投 5 块买万家纳指", {"amount": 5, "currency": "CNY", "period": "week", "target": "万家纳指"}),
]


def test_56_money_metadata_dao_and_weekly_cases():
    """56 号：BOOS 说“我现在要每天定投5刀的沪深300”——“5刀”就是 5 美元，
    “定投<金额>的<目标>”里的目标（沪深300）要抠出来，每周也要认。"""
    for text, wanted in MONEY_56_PHRASES:
        assert _extract_money_metadata(text) == wanted, text
    # 没目标时 target 是 None，不硬猜。
    assert _extract_money_metadata("以后每天定投5刀") == {
        "amount": 5, "currency": "USD", "period": "day", "target": None,
    }
    # 旧说法不能坏：买后面的目标、预算还是老样子。
    assert _extract_money_metadata("每天定投 5 块买万家纳指") == {
        "amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指",
    }
    assert _extract_money_metadata("每月预算 5 美元，到 5 美元提醒") == {
        "amount": 5, "currency": "USD", "period": "month", "target": "工具预算",
    }


def test_56_normalize_accepts_week_period():
    """56 号：period 加了 "week"，规整时要认，别当坏形状丢了。"""
    assert _normalize_money_metadata(
        {"amount": 20, "currency": "CNY", "period": "week", "target": "沪深300"}
    ) == {"amount": 20, "currency": "CNY", "period": "week", "target": "沪深300"}


def test_56_add_memory_text_attaches_dao_metadata(tmp_path):
    """56 号：直存“每天定投5刀的沪深300”，库里那条要带金额字段
    （5 美元/天/沪深300），回答“每月定投多少钱”时才有据可依。"""
    fake = PromotingFakeMemory()
    store = make_fake_store(tmp_path, memory=fake)
    store.add_memory_text("BOOS 想每天定投5刀的沪深300")
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["metadata"]["money"] == {
        "amount": 5, "currency": "USD", "period": "day", "target": "沪深300",
    }


def test_56_system_message_stores_determined_parts_first():
    """56 号：第 8 条要写明——新指令先把能确定的部分存下来，
    不确定的只追问一句；句子里已经给过的数（“5刀”就是 5 美元）
    别又反问一遍。"""
    from tools.agent_loop import build_loop_system_message

    system = build_loop_system_message()
    assert "先把能确定的部分" in system
    assert "只追问一句" in system
    assert "5刀" in system


# ---------- 56 号补：定投金额后面直接跟目标（没有“的”） ----------

# 56 号补 5 条新说法（实测里模型复述丢了“的”）：金额后直接跟目标也要认；
# 时间状语（“到年底”）和货币词（“钱”）不是目标，不能认。
DCA_NO_DE_56_CASES = [
    ("每天定投5美元沪深300", {"amount": 5, "currency": "USD", "period": "day", "target": "沪深300"}),
    ("每天定投5块万家纳指", {"amount": 5, "currency": "CNY", "period": "day", "target": "万家纳指"}),
    ("每月定投10刀标普", {"amount": 10, "currency": "USD", "period": "month", "target": "标普"}),
    ("从今天定投到年底，每天 5 块", None),  # “到年底”不是目标
    ("每天定投5块钱", None),  # “钱”不是目标
]


def test_56_dca_amount_target_without_de():
    """56 号补：模型复述丢了“的”（“每天定投5美元沪深300”）也要抠出目标；
    “到年底”“钱”这种不是目标，不认。"""
    from memory.mem0_store import _extract_money_metadata

    for text, wanted in DCA_NO_DE_56_CASES:
        got = _extract_money_metadata(text)
        if wanted is None:
            assert got is None or got.get("target") is None, (text, got)
        else:
            assert got == wanted, (text, got)


def test_53_live_h_sequence_no_duplicate_without_de(tmp_path):
    """53 号 + 56 号补的活路回归（2026-11-12 实测）：模型调 add_memory 存的
    是复述句“我想以后每天定投5美元沪深300”（没有“的”），提取落袋的是
    原话“我现在要每天定投5刀的沪深300”——两条同槽位同值，必须只留一条，
    不能因为复述句的 target 没抠出来就漏成两条。"""
    from conftest import make_fake_store

    store = make_fake_store(tmp_path)
    store.add_memory_text("我想以后每天7点起床")
    store.save_turn(
        [
            {"role": "user", "content": "我想以后每天7点起床"},
            {"role": "assistant", "content": "记下了：以后每天 7 点起床。"},
        ]
    )
    import time

    time.sleep(0.3)
    store.add_memory_text("我想以后每天定投5美元沪深300")
    store.save_turn(
        [
            {"role": "user", "content": "我想以后每天7点起床"},
            {"role": "assistant", "content": "记下了：以后每天 7 点起床。"},
            {"role": "user", "content": "我现在要每天定投5刀的沪深300"},
            {"role": "assistant", "content": "记下了：每天定投 5 美元沪深300。"},
        ]
    )
    time.sleep(0.3)
    store.save_turn(
        [
            {"role": "user", "content": "我想以后每天7点起床"},
            {"role": "assistant", "content": "记下了：以后每天 7 点起床。"},
            {"role": "user", "content": "我现在要每天定投5刀的沪深300"},
            {"role": "assistant", "content": "记下了：每天定投 5 美元沪深300。"},
            {"role": "user", "content": "我每天定投多少钱"},
            {"role": "assistant", "content": "每天定投 5 美元，对象是沪深300。"},
        ]
    )
    store.flush_pending()
    if store._outbox is not None:
        store._outbox.flush()
    texts = store._memory.all_texts()
    assert len(texts) == 2, texts
    assert any("沪深300" in t for t in texts)
    assert sum("沪深300" in t for t in texts) == 1
