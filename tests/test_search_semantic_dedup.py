# tests/test_search_semantic_dedup.py（回归：2026-11-20 牛肉面事故）
# 事发：库里 19 份一模一样的“我爱的是冒菜，不是麻辣烫”（向量分 0.4771），
# 问“我昨天吃了什么”时语义 top_k=5 全被冒菜占满，牛肉面事件记忆
# （2026年11月19日晚上在盐湖城吃了一碗牛肉面，向量分 0.4699、全库第 22 名）
# 进不来；BM25 那路排得对（牛肉面第 2）却因语义已满 5 席根本没跑，
# 模型拿到的上下文只有 5 条冒菜，只能答“没有记录”。
# 根因：Mem0Store.search() 的 mem0 语义循环没有文本去重
# （if text in seen_texts: continue 只存在于 BM25 补充环路和账本兜底环路）。
# 修：语义循环里补上同一文本只占一席。
# 全部离线：FakeMemory + 注入的语义分数（按诊断报告的真实裸排名），
# BM25 走真实代码，不联网、不调真实 DeepSeek。
# 旧代码：test_semantic_duplicates_do_not_crowd_out_beef_noodle_event 失败
# （5 条全是冒菜，牛肉面进不来）；修完通过。

from conftest import FakeMemory, make_fake_store
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    SOURCE_INFERRED,
    _provenance,
)

DUP_TEXT = "我爱的是冒菜，不是麻辣烫"
TARGET_TEXT = "2026年11月19日晚上，用户在盐湖城吃了一碗牛肉面，觉得特别好吃。"
SWIM_TEXT = "用户昨天（2026年11月19日）去游泳了，游了半小时。"
PREF_TEXT = "BOOS 喜欢吃牛肉面。"
WAKE_TEXT = "我想以后每天7点起床"
HOT_POT_TEXT = "我爱吃火锅"

# 诊断报告的真实裸语义排名（diag_search2_out.jsonl，全库 49 条里的前几名）：
# 冒菜 19 份 0.4771（#1–#19）、起床 0.4712（#20）、牛肉面偏好 0.4708（#21）、
# 牛肉面事件 0.4699（#22）、火锅 0.4642、游泳 0.4564。
_SCORES = {
    DUP_TEXT: 0.477116,
    WAKE_TEXT: 0.471225,
    PREF_TEXT: 0.470773,
    TARGET_TEXT: 0.469881,
    HOT_POT_TEXT: 0.464219,
    SWIM_TEXT: 0.456366,
}


class RankedFakeMemory(FakeMemory):
    """按诊断报告的真实向量排名返回语义结果（只认上面这张表里的文本）。

    FakeMemory 自带的 search 是关键词重合打分，复现不了“向量分高、
    关键词分低”的冒菜（它和查询一个两字窗口都不共享，BM25 分是 0，
    向量分却 0.4771 全库第一）——所以语义分数直接按诊断报告注入，
    get_all / BM25 / 账本仍走真实代码。
    """

    def search(self, query, filters=None, top_k=5):
        with self._lock:
            scored = []
            for item in self._items:
                score = _SCORES.get(item["memory"])
                if score is None:
                    continue
                scored.append(
                    {
                        "id": item["id"],
                        "memory": item["memory"],
                        "score": score,
                        "metadata": dict(item["metadata"])
                        if item.get("metadata")
                        else None,
                    }
                )
            scored.sort(key=lambda entry: entry["score"], reverse=True)
            return {"results": scored[:top_k]}


def _seed_incident_store(tmp_path):
    """按事发时的库 seeding：19 份冒菜 + 5 条其它记忆。返回 (store, target_id)。"""
    store = make_fake_store(tmp_path, memory=RankedFakeMemory())
    meta = _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    target_id = None
    for _ in range(19):
        store._add_text_with_metadata(DUP_TEXT, dict(meta))
    for text in (WAKE_TEXT, PREF_TEXT, HOT_POT_TEXT, SWIM_TEXT):
        store._add_text_with_metadata(text, dict(meta))
    result = store._add_text_with_metadata(TARGET_TEXT, dict(meta))
    target_id = (result.get("results") or [{}])[0].get("id")
    assert target_id is not None
    assert store._memory.all_texts().count(DUP_TEXT) == 19
    return store, target_id


def test_semantic_duplicates_do_not_crowd_out_beef_noodle_event(tmp_path):
    """19 份重复冒菜不能把牛肉面事件挤出 top_k=5。"""
    store, target_id = _seed_incident_store(tmp_path)
    results = store.search("我昨天吃了什么", top_k=5)
    texts = [item["text"] for item in results]
    assert TARGET_TEXT in texts, f"牛肉面事件没进 top_k=5，实际返回：{texts}"
    assert any(item["id"] == target_id for item in results)
    # 同一文本只占一席：5 个席位不能再被同一句话占满。
    assert texts.count(DUP_TEXT) == 1, f"冒菜重复占位：{texts}"
    assert len(texts) == len(set(texts)), f"返回里还有重复文本：{texts}"
    assert len(results) <= 5


def test_exact_duplicate_sentence_is_not_stored_twice(tmp_path):
    """写入路径：同一句话不重复写入（第 2 项的验证，写成回归测试）。

    两条写入路径都拦 exact 重复：
    - 手动存（add_memory_text）：先查库，已有就返回已有那条（duplicate=True）；
    - 提取重放（_dedupe_extraction_results）：新抽出的完全相同句直接丢掉、
      从库里删掉新副本（test_4 同款裁决，换成事故里的冒菜句）。
    库里 19 份冒菜是历史遗留（全无 write_seq / confidence / source，
    现行 _provenance 写出的记忆一定带这三项）——当前代码写不出第 20 份。
    """
    store = make_fake_store(tmp_path)
    meta = _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    first = store.add_memory_text(DUP_TEXT, source="inferred")
    assert first.get("results"), "第一条没存上"
    assert store._memory.all_texts().count(DUP_TEXT) == 1
    second = store.add_memory_text(DUP_TEXT, source="inferred")
    assert second.get("duplicate") is True, "第二遍手动存同一句应判 duplicate"
    assert store._memory.all_texts().count(DUP_TEXT) == 1

    before = store._snapshot_all()
    dup_id = store._memory._store(DUP_TEXT, metadata=dict(meta))
    assert store._memory.all_texts().count(DUP_TEXT) == 2
    out = store._dedupe_extraction_results(
        {"results": [{"id": dup_id, "memory": DUP_TEXT, "event": "ADD"}]},
        before,
        turn_id="test-turn",
    )
    assert out.get("dedup_dropped", 0) == 1
    assert store._memory.all_texts().count(DUP_TEXT) == 1
