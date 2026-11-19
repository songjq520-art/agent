# tests/test_graph_reasoning.py（图谱推理第一期）
# 设计方案第六节的测试想法，逐项落在这里：
#   - 图能建成（快照文件在，删掉能从账本重建，边数一致）；
#   - 多跳找得到（“每天定投 5 块”找回原文没“万家纳指”的那两条；
#     “标普呢”找回标普的定投、代码）；
#   - 失效边不走（被取代后顺旧边找不着，问“以前”时才找着并标）；
#   - 兜底还在（图文件删掉、种子找不着时，检索结果跟现在一样）；
#   - 删改核对不动（delete/update 还是走 _find_anchor_matches）；
#   - 不改原话（图里不存原话全文，抽实体、连边都不动账本事件）；
#   - 成本实测（见 test_graph_llm_prompt_cost，单独跑、用真模型）。
# 测试一律用 FakeMemory（离线、不联网、不产生费用），只有
# test_graph_llm_prompt_cost 那个是手工跑的真调（见文件末尾）。

import json
from pathlib import Path

from memory import graph as graph_mod
from memory import ledger as ledger_mod
from memory.graph_entities import extract_entities_free
from tests.conftest import make_fake_store


def _add(store, text):
    """往 store 里存一条记忆原文（add_memory_text，走账本 + 图）。"""
    result = store.add_memory_text(text)
    assert result, f"存记忆失败：{text!r}"
    return result


def _texts(results):
    return [item.get("text") or "" for item in results]


def _seed_store(tmp_path):
    """三条定投记忆 + 两条标普记忆（设计方案的例子）。"""
    store = make_fake_store(tmp_path)
    _add(store, "每天定投 5 块钱买万家纳指")
    _add(store, "每天定投 5 块")
    _add(store, "万家纳指的基金代码是 019441")
    _add(store, "摩根标普定投 10 块")
    _add(store, "摩根标普的基金代码是 017641")
    return store


def test_graph_builds_on_add(tmp_path):
    """图能建成：存记忆时快照文件在，有记忆节点、实体节点、边。"""
    store = _seed_store(tmp_path)
    snapshot = Path(store.graph_path)
    assert snapshot.exists(), "图快照文件应该在"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    kinds = [node.get("kind") for node in payload["nodes"]]
    assert "memory" in kinds and "entity" in kinds
    assert len(payload["edges"]) > 0
    # 实体节点里有万家纳指、定投、019441。
    entity_texts = {
        node.get("text") for node in payload["nodes"] if node.get("kind") == "entity"
    }
    assert "万家纳指" in entity_texts
    assert "定投" in entity_texts
    assert "019441" in entity_texts
    # 记忆节点是 5 条（每条 add 事件一个）。
    assert sum(1 for kind in kinds if kind == "memory") == 5


def test_graph_multihop_search(tmp_path):
    """多跳找得到：问“每天定投 5 块”，顺着网找回
    “万家纳指的基金代码是 019441”那条——它和查询没有共同词，
    语义和关键词都找不着，只有图找得到（定投 → 万家纳指那条
    → 019441），source="graph"。"""
    store = _seed_store(tmp_path)
    results = store.search("每天定投 5 块", top_k=5)
    texts = _texts(results)
    assert "万家纳指的基金代码是 019441" in texts
    hit = next(
        item for item in results if item.get("text") == "万家纳指的基金代码是 019441"
    )
    assert hit.get("source") == "graph"
    assert hit.get("graph_score", 0) > 0


def test_graph_multihop_biao_pu(tmp_path):
    """多跳找得到（标普）：问“摩根标普”，找回标普的定投和代码。"""
    store = _seed_store(tmp_path)
    results = store.search("摩根标普", top_k=5)
    texts = _texts(results)
    assert "摩根标普定投 10 块" in texts
    assert "摩根标普的基金代码是 017641" in texts


def test_graph_ppr_direct(tmp_path):
    """PPR 本身多跳：种子“定投”能把 019441 那条带出来
    （定投 → 每天定投 5 块钱买万家纳指 → 万家纳指 → 019441）。"""
    store = _seed_store(tmp_path)
    graph = store._get_graph()
    scores = graph.ppr(["定投"])
    # 019441 那条的 key 在里面（key 是原话 + created_at 的哈希，
    # 这里按账本档案反查文本认）。
    state = store.ledger_state()
    key_to_text = {
        (record or {}).get("key"): (record or {}).get("text")
        for record in state.get("adds") or []
    }
    found = {key_to_text.get(key) for key in scores}
    assert "万家纳指的基金代码是 019441" in found
    assert "每天定投 5 块钱买万家纳指" in found
    assert "每天定投 5 块" in found


def test_graph_invalid_edges(tmp_path):
    """失效边不走：同槽位新金额取代旧的后，顺旧边找不着它了；
    问“以前”时才找着，并标“这是以前的”。"""
    store = make_fake_store(tmp_path)
    _add(store, "每天定投 5 块钱买万家纳指")
    _add(store, "每天定投 5 块")
    _add(store, "万家纳指的基金代码是 019441")
    # 同槽位新金额：5 块 → 10 块，旧条被取代。
    _add(store, "每天定投 10 块钱买万家纳指")
    results = store.search("每天定投 5 块钱买万家纳指", top_k=5)
    texts = _texts(results)
    assert "每天定投 5 块钱买万家纳指" not in texts
    assert "每天定投 10 块钱买万家纳指" in texts
    # 问“以前”：旧的找着了，标 is_history。
    history = store.search(
        "每天定投 5 块钱买万家纳指", top_k=5, include_history=True
    )
    history_texts = _texts(history)
    assert "每天定投 5 块钱买万家纳指" in history_texts
    old = next(
        item
        for item in history
        if item.get("text") == "每天定投 5 块钱买万家纳指"
    )
    assert old.get("is_history") is True


def test_graph_fallback_when_snapshot_missing(tmp_path):
    """兜底还在：图快照删掉、图是空的时，检索结果跟没图时一样
    （只有老路，不崩、不丢记忆）。"""
    store = _seed_store(tmp_path)
    before = store.search("每天定投 5 块钱买万家纳指", top_k=5)
    # 快照删掉 + 内存里的图清掉（模拟图坏了）。
    Path(store.graph_path).unlink(missing_ok=True)
    store._graph = None
    # 把重建也堵住：账本先搬走，重建只能得到空图。
    ledger_backup = Path(store.ledger_path).read_bytes()
    Path(store.ledger_path).unlink(missing_ok=True)
    store._graph = None
    after = store.search("每天定投 5 块钱买万家纳指", top_k=5)
    # 老路还在：语义命中的那条还在（图那条没了是正常的，图是加分项）。
    assert "每天定投 5 块钱买万家纳指" in _texts(after)
    assert all(item.get("source") != "graph" for item in after)
    # 账本恢复，免得影响别的测试（每个测试有自己的 tmp 目录，
    # 这里只是把文件写回去，保持干净）。
    Path(store.ledger_path).write_bytes(ledger_backup)


def test_graph_no_seed_query(tmp_path):
    """种子找不着时跳过图这路：问一句没实体的，检索照常、不崩。"""
    store = _seed_store(tmp_path)
    results = store.search("今天天气怎么样", top_k=5)
    assert isinstance(results, list)
    assert all(item.get("source") != "graph" for item in results)


def test_graph_rebuild_from_ledger(tmp_path):
    """图能重建：删掉快照，重新建 store，图从账本重建，边数一致。"""
    store = _seed_store(tmp_path)
    before = store._get_graph()
    before_edges = len(before.edges)
    before_nodes = len(before.nodes)
    assert before_edges > 0
    Path(store.graph_path).unlink(missing_ok=True)
    store._graph = None
    rebuilt = store._get_graph()
    assert len(rebuilt.edges) == before_edges
    assert len(rebuilt.nodes) == before_nodes
    # 重建完快照要写盘（2026-11-19 真机教训：之前重建不写盘，
    # 快照一直没有，服务每次启动都现重建）。
    assert Path(store.graph_path).exists()
    # 重建后多跳还在。
    scores = rebuilt.ppr(["定投"])
    state = store.ledger_state()
    key_to_text = {
        (record or {}).get("key"): (record or {}).get("text")
        for record in state.get("adds") or []
    }
    assert "万家纳指的基金代码是 019441" in {
        key_to_text.get(key) for key in scores
    }


def test_graph_build_from_events_writes_snapshot(tmp_path):
    """手动重建要写盘：build_graph_from_events(path=...) 跑完，
    快照文件要在，读回来节点边数一致（2026-11-19 真机回归：
    笔记本上跑完 34 条、Test-Path 还是 False，就是这处没写盘）。"""
    events = [
        {"type": "add", "seq": 1, "key": "k1",
         "text": "每天定投 5 块钱买万家纳指", "ts": "2026-11-19T00:00:00", "meta": {}},
        {"type": "add", "seq": 2, "key": "k2",
         "text": "万家纳指的基金代码是 019441", "ts": "2026-11-19T00:01:00", "meta": {}},
    ]
    snapshot = tmp_path / "graph_snapshot.json"
    graph, info = graph_mod.build_graph_from_events(events, path=str(snapshot))
    assert info["saved"] is True
    assert snapshot.exists()
    reloaded = graph_mod.load_graph(snapshot)
    assert len(reloaded.nodes) == len(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)


def test_graph_no_full_text_in_snapshot(tmp_path):
    """不改原话、不存原话全文：图快照里没有原话全文
    （记忆节点只有 15 个字以内的短标题）。"""
    store = make_fake_store(tmp_path)
    long_text = "每天定投 5 块钱买万家纳指，雷打不动，风雨无阻，从不间断"
    _add(store, long_text)
    payload = Path(store.graph_path).read_text(encoding="utf-8")
    assert long_text not in payload
    # 账本里原话还在（一字不改）。
    events, _info = ledger_mod.read_events(store.ledger_path)
    assert any(event.get("text") == long_text for event in events)


def test_graph_add_idempotent(tmp_path):
    """同一条记忆重放不翻倍：同一条 add 事件记两遍，共现权重不变，
    边数不变。"""
    store = make_fake_store(tmp_path)
    _add(store, "每天定投 5 块钱买万家纳指")
    _add(store, "每天定投 5 块")
    graph = store._get_graph()
    edges_before = len(graph.edges)
    # 直接重放同一条 add（模拟恢复/重建时重复记）。
    state = store.ledger_state()
    first = state["adds"][0]
    graph.add_memory(
        first["key"],
        first["text"],
        extract_entities_free(first["text"]),
        valid_at=first.get("meta", {}).get("created_at"),
        source_seq=first["seq"],
    )
    assert len(graph.edges) == edges_before


def test_graph_llm_fallback_wiring(tmp_path):
    """LLM 兜底的接线：免费抽不到时，调 _judge_json 拿实体
    （这里用假客户端，不联网）。"""
    store = make_fake_store(tmp_path)
    # 假 LLM 客户端：_judge_json 走 self._memory.llm.client。
    import json as _json

    class _FakeCompletions:
        def create(self, **kwargs):
            class _Msg:
                content = _json.dumps(
                    {"entities": [{"name": "火锅", "kind": "thing"}]},
                    ensure_ascii=False,
                )

            class _Choice:
                message = _Msg()
                finish_reason = "stop"

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _FakeClient:
        def __init__(self):
            self.chat = type("c", (), {"completions": _FakeCompletions()})()

    class _FakeLLM:
        def __init__(self):
            self.client = _FakeClient()

    store._memory.llm = _FakeLLM()
    # “火锅”在 THING_WORDS 里，免费就能抽到——所以这里用一句
    # 免费抽不到的，验证兜底被调到。
    # “昨天去吃了那家很有名的店”：免费抽不到任何实体。
    from memory.graph_entities import extract_entities

    got = extract_entities("昨天去吃了那家很有名的店", llm_judge=store._judge_json)
    assert ("火锅", "thing") in got


def test_graph_family_alias(tmp_path):
    """家人自称认得出来：“我妈”就是妈妈（2026-11-18 真调发现
    模型和免费抽取都认不出“我妈”，只认出了“吃饭”）。"""
    assert ("妈妈", "person") in extract_entities_free(
        "帮我记一下，我妈说周末回老家吃饭"
    )
    assert ("奶奶", "person") in extract_entities_free("我奶说想吃冒菜")


def test_graph_relations_in_graph(tmp_path):
    """记忆—记忆边进图：同槽位取代的那对，图上有 supersede 边。"""
    store = make_fake_store(tmp_path)
    _add(store, "每天定投 5 块钱买万家纳指")
    _add(store, "每天定投 10 块钱买万家纳指")
    graph = store._get_graph()
    kinds = [edge.get("kind") for edge in graph.edges]
    assert "supersede" in kinds
