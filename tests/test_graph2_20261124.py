# tests/test_graph2_20261124.py（四件套第 3 块：图谱二期）
#
# 覆盖契约「图谱边口径」与任务五条：
#   1. 边必须分型：add_relation 只认四种关系；存量无类型/未知类型边
#      被隔离（留在图里可查，不参与扩散）。
#   2. 作废法（Graphiti）：边带成立/失效时间与出处；作废只标失效、
#      不删边；作废后不再召回；按时间点能回查成立区间与出处。
#   3. 同义词边带 scope：无 scope 不许建；检索按 scope 过滤
#      （同 scope 或 global 才生效）。审批分级不在本块，只测机制。
#   4. search() 图谱路真实现：一跳联想扩展召回（语义/BM25 找不到的
#      记忆顺边找回来），source="graph"，不越 top_k；取代边定向、
#      冲突边不作正相关。
#   5. 召回只读：扩展与检索不改图、不改库。

import copy
import json

import pytest

from memory import ledger as _ledger
from memory.graph import (
    EDGE_SYNONYM,
    MemoryGraph,
    entity_node_id,
    memory_node_id,
)
from memory.sqlite_store import SQLiteStore

T1 = "2026-11-01T00:00:00+08:00"
T2 = "2026-11-10T00:00:00+08:00"
T3 = "2026-11-20T00:00:00+08:00"


def _graph():
    """手搭一张图：三条记忆，实体互不重叠的部分留给关系边测试用。"""
    g = MemoryGraph()
    g.add_memory("kA", "妈妈平时爱喝咖啡",
                 [("妈妈", "person"), ("咖啡", "thing")],
                 valid_at=T1, source_seq=1)
    g.add_memory("kB", "姐姐周末也喝咖啡",
                 [("姐姐", "person"), ("咖啡", "thing")],
                 valid_at=T1, source_seq=2)
    g.add_memory("kC", "我定投的那只基金是万家纳指",
                 [("定投", "habit"), ("万家纳指", "fund")],
                 valid_at=T1, source_seq=3)
    return g


# ---------- 1. 边必须分型 / 存量无类型边隔离 ----------

def test_relation_kind_must_be_typed():
    g = _graph()
    assert g.add_relation("kA", "kB", "friend") is False
    assert g.add_relation("kA", "kB", "") is False
    assert all(e["kind"] != "friend" for e in g.edges)
    assert g.add_relation("kA", "kB", "related") is True


def test_untyped_stock_edges_quarantined(tmp_path):
    """老快照里没写 kind 的边：留在图里可查，但不参与扩散。"""
    payload = {
        "version": 1,
        "nodes": [
            {"id": "mem:kX", "kind": "memory", "text": "甲", "entities": []},
            {"id": "mem:kY", "kind": "memory", "text": "乙", "entities": []},
        ],
        "edges": [
            {"from": "mem:kX", "to": "mem:kY", "kind": "",
             "valid_at": T1, "invalid_at": None, "source_seq": 1},
            {"from": "mem:kX", "to": "mem:kY", "kind": "mystery",
             "valid_at": T1, "invalid_at": None, "source_seq": 2},
        ],
    }
    path = tmp_path / "graph_snapshot.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    g = MemoryGraph(path)
    assert g.load() is True
    assert len(g.untyped_edges()) == 2
    # 隔离：顺这两条边扩不到对面
    assert "kY" not in g.expand_one_hop(seed_keys=["kX"])
    assert "kX" not in g.expand_one_hop(seed_keys=["kY"])


def test_synonym_edge_requires_scope():
    g = _graph()
    assert g.add_synonym_edge("万家纳指", "纳斯达克", scope=None) is False
    assert g.add_synonym_edge("万家纳指", "纳斯达克", scope="") is False
    assert g.add_synonym_edge("万家纳指", "纳斯达克", scope="global") is True
    # 同对同 scope 已有有效边：不重复连
    assert g.add_synonym_edge("万家纳指", "纳斯达克", scope="global") is False


# ---------- 2. 作废法：标失效不删、停召回、留痕可查 ----------

def test_invalidate_synonym_edge_stops_recall_but_keeps_trace():
    g = _graph()
    assert g.add_synonym_edge(
        "万家纳指", "纳斯达克", scope="global",
        provenance="测试提议", evidence="同指纳斯达克100指数",
        valid_at=T2, source_seq=9,
    ) is True
    assert "kC" in g.expand_one_hop(seed_entities=["纳斯达克"])
    ok = g.invalidate_edge(
        entity_node_id("万家纳指"), entity_node_id("纳斯达克"),
        EDGE_SYNONYM, T3, reason="误绑，其实不是同一只",
    )
    assert ok is True
    # 作废后不再召回
    assert "kC" not in g.expand_one_hop(seed_entities=["纳斯达克"])
    # 边还在：全史能查到出处、成立/失效时间、作废理由
    history = g.edge_history(
        entity_node_id("万家纳指"), entity_node_id("纳斯达克"),
        kind=EDGE_SYNONYM,
    )
    assert len(history) == 1
    edge = history[0]
    assert edge["valid_at"] == T2
    assert edge["invalid_at"] == T3
    assert edge["provenance"] == "测试提议"
    assert edge["invalid_reason"] == "误绑，其实不是同一只"
    # 按时间点回查：作废前成立、作废后不成立
    assert any(
        e["kind"] == EDGE_SYNONYM for e in g.edges_at("2026-11-15T00:00:00+08:00")
    )
    assert not any(
        e["kind"] == EDGE_SYNONYM for e in g.edges_at("2026-11-21T00:00:00+08:00")
    )
    # 重复作废不重复标
    assert g.invalidate_edge(
        entity_node_id("万家纳指"), entity_node_id("纳斯达克"),
        EDGE_SYNONYM, T3, reason="再废一次",
    ) is False


def test_invalidate_requires_timestamp():
    g = _graph()
    g.add_synonym_edge("万家纳指", "纳斯达克", scope="global")
    assert g.invalidate_edge(
        entity_node_id("万家纳指"), entity_node_id("纳斯达克"),
        EDGE_SYNONYM, None,
    ) is False


# ---------- 边语义：取代定向、冲突不作正相关 ----------

def test_supersede_edge_is_directional():
    g = MemoryGraph()
    g.add_memory("kOld", "旧说法", [("妈妈", "person")], valid_at=T1, source_seq=1)
    g.add_memory("kNew", "新说法", [("姐姐", "person")], valid_at=T2, source_seq=2)
    # 方向约定：from=被取代的旧条，to=取代它的新条
    assert g.add_relation("kOld", "kNew", "supersede") is True
    # 从旧条出发能走到新条
    assert "kNew" in g.expand_one_hop(seed_keys=["kOld"])
    # 从新条出发不许给被取代方正向扩散
    assert "kOld" not in g.expand_one_hop(seed_keys=["kNew"])
    # PPR 同口径：以新条为种子，旧条不得分
    ppr = g.ppr(["姐姐"], memory_seed_scores={"kNew": 1.0})
    assert "kOld" not in ppr


def test_conflict_edge_never_expands():
    g = MemoryGraph()
    g.add_memory("kA", "说法甲", [("妈妈", "person")], valid_at=T1, source_seq=1)
    g.add_memory("kB", "说法乙", [("姐姐", "person")], valid_at=T1, source_seq=2)
    assert g.add_relation("kA", "kB", "conflict") is True
    assert "kB" not in g.expand_one_hop(seed_keys=["kA"])
    assert "kA" not in g.expand_one_hop(seed_keys=["kB"])
    # 冲突边本身还在图里（留痕），只是不参与正向扩散
    assert len(g.edge_history(memory_node_id("kA"), memory_node_id("kB"),
                               kind="conflict")) == 1


# ---------- 3. 同义词边 scope 过滤 ----------

def test_synonym_scope_filtering():
    g = _graph()
    assert g.add_synonym_edge(
        "万家纳指", "纳斯达克", scope="ctx:fund",
        provenance="测试", valid_at=T2,
    ) is True
    # 同 scope 才生效
    assert "kC" in g.expand_one_hop(seed_entities=["纳斯达克"], scope="ctx:fund")
    # 没带上下文、或上下文对不上：不生效
    assert "kC" not in g.expand_one_hop(seed_entities=["纳斯达克"])
    assert "kC" not in g.expand_one_hop(seed_entities=["纳斯达克"], scope="ctx:other")


def test_synonym_global_scope_applies_everywhere():
    g = _graph()
    assert g.add_synonym_edge("万家纳指", "纳斯达克", scope="global") is True
    assert "kC" in g.expand_one_hop(seed_entities=["纳斯达克"])
    assert "kC" in g.expand_one_hop(seed_entities=["纳斯达克"], scope="ctx:anything")


# ---------- 5. 扩展只读 ----------

def test_expand_and_ppr_do_not_mutate_graph():
    g = _graph()
    g.add_relation("kA", "kB", "related")
    g.add_synonym_edge("万家纳指", "纳斯达克", scope="global")
    before_edges = copy.deepcopy(g.edges)
    before_nodes = copy.deepcopy(g.nodes)
    g.expand_one_hop(seed_keys=["kA"], seed_entities=["纳斯达克"])
    g.ppr(["妈妈"], memory_seed_scores={"kA": 0.5})
    assert g.edges == before_edges
    assert g.nodes == before_nodes


# ---------- 账本：graph_edge 事件折叠与快照 ----------

def test_ledger_graph_edge_events_fold_and_snapshot(tmp_path):
    ledger_path = tmp_path / "memory_events.jsonl"
    snap_path = tmp_path / "snapshot.json"
    _ledger.append_event(ledger_path, {
        "type": _ledger.LEDGER_EVENT_GRAPH_EDGE, "action": "add",
        "kind": "synonym", "entity_a": "万家纳指", "entity_b": "纳斯达克",
        "scope": "global", "provenance": "测试", "evidence": "同指",
        "valid_at": T2,
    })
    _ledger.append_event(ledger_path, {
        "type": _ledger.LEDGER_EVENT_GRAPH_EDGE, "action": "invalidate",
        "kind": "synonym", "entity_a": "万家纳指", "entity_b": "纳斯达克",
        "invalid_at": T3, "reason": "误绑",
    })
    state = _ledger.load_ledger_state(ledger_path, snap_path)
    edges = state["graph_edges"]
    assert len(edges) == 2
    assert edges[0]["action"] == "add" and edges[0]["scope"] == "global"
    assert edges[1]["action"] == "invalidate" and edges[1]["reason"] == "误绑"
    # 快照压缩后边账不丢
    _ledger.write_snapshot(snap_path, state, state["seq_max"])
    payload = _ledger.read_snapshot(snap_path)
    assert len(payload["graph_edges"]) == 2


def test_rebuild_replays_graph_edge_events():
    """图从账本重建时，同义词边的连/废按 graph_edge 事件原样回来。"""
    events = [
        {"type": "add", "key": "kC", "text": "我定投的那只基金是万家纳指",
         "meta": {}, "seq": 1, "ts": T1},
        {"type": _ledger.LEDGER_EVENT_GRAPH_EDGE, "action": "add",
         "kind": "synonym", "entity_a": "万家纳指", "entity_b": "纳斯达克",
         "scope": "global", "provenance": "测试", "evidence": "同指",
         "valid_at": T2, "seq": 2, "ts": T2},
        {"type": _ledger.LEDGER_EVENT_GRAPH_EDGE, "action": "invalidate",
         "kind": "synonym", "entity_a": "万家纳指", "entity_b": "纳斯达克",
         "invalid_at": T3, "reason": "误绑", "seq": 3, "ts": T3},
    ]
    g = MemoryGraph()
    g.rebuild_from_events(events)
    history = g.edge_history(
        entity_node_id("万家纳指"), entity_node_id("纳斯达克"),
        kind=EDGE_SYNONYM,
    )
    assert len(history) == 1
    assert history[0]["invalid_at"] == T3
    assert history[0]["provenance"] == "测试"
    assert "kC" not in g.expand_one_hop(seed_entities=["纳斯达克"])


# ---------- 4. store 层：search 图谱路 ----------

@pytest.fixture
def store(tmp_path, monkeypatch):
    """干净的离线 SQLiteStore；向量打桩成 None（只走原样/BM25/图谱路）。"""
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"))


def _add_user(store, text):
    saved = store.add_memory_text(text, source="user")
    assert saved.get("event") == "ADD", saved
    return saved["id"]


def test_store_one_hop_recall_via_shared_entity(store):
    m1 = _add_user(store, "妈妈平时爱喝咖啡")
    m2 = _add_user(store, "姐姐周末也喝咖啡")
    _add_user(store, "我爱吃火锅")  # 无关记忆，不该被带出来
    results = store.search("妈妈最近怎么样")
    by_id = {r["id"]: r for r in results}
    assert m1 in by_id
    # m2 与查询没有词面重合，只能顺「咖啡」这条边一跳找回来
    assert m2 in by_id
    assert by_id[m2]["source"] == "graph"
    assert len(results) <= 5


def test_store_synonym_recall_scope_and_void(store, tmp_path):
    m = _add_user(store, "我定投的那只基金是万家纳指")
    query = "纳斯达克最近涨了吗"
    # 没连同义词边之前：词面、语义（打桩关）都找不到
    assert m not in {r["id"] for r in store.search(query, scope="ctx:fund")}
    out = store.add_synonym_edge(
        "万家纳指", "纳斯达克", scope="ctx:fund",
        provenance="测试提议", evidence="同指纳斯达克100指数",
    )
    assert out["ok"] is True and out["applied"] is True
    # scope 对不上不召回，对上了才召回
    assert m not in {r["id"] for r in store.search(query)}
    assert m not in {r["id"] for r in store.search(query, scope="ctx:other")}
    hits = store.search(query, scope="ctx:fund")
    assert m in {r["id"] for r in hits}
    assert {r["id"]: r for r in hits}[m]["source"] == "graph"
    # 无 scope 建边被拒
    assert store.add_synonym_edge("万家纳指", "纳斯达克", scope="")["ok"] is False
    # 作废后停召回，边留痕可查
    assert store.void_graph_edge(
        "synonym", name_a="万家纳指", name_b="纳斯达克", reason="测完作废"
    )["ok"] is True
    assert m not in {r["id"] for r in store.search(query, scope="ctx:fund")}
    history = store.graph_edge_history(
        name_a="万家纳指", name_b="纳斯达克", kind="synonym"
    )
    assert len(history) == 1
    assert history[0]["invalid_at"]
    assert history[0]["invalid_reason"] == "测完作废"
    assert history[0]["provenance"] == "测试提议"
    assert history[0]["scope"] == "ctx:fund"
    # 换个 store 实例（图从账本重建）：作废状态还在
    store2 = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    history2 = store2.graph_edge_history(
        name_a="万家纳指", name_b="纳斯达克", kind="synonym"
    )
    assert len(history2) == 1 and history2[0]["invalid_at"]
    assert m not in {r["id"] for r in store2.search(query, scope="ctx:fund")}


def test_store_search_respects_top_k_with_graph(store):
    _add_user(store, "妈妈平时爱喝咖啡")
    others = [
        "姐姐周末去公园跑步，跑完喝咖啡歇一会",
        "斌下班路上买了杯咖啡，顺便取了快递",
        "静静在西安的办公室里喝咖啡看文件",
        "贵妃晚上写代码的时候离不开咖啡",
        "大哥开车去郊外钓鱼，保温杯里装的咖啡",
        "雨哥在火车站候车，买了杯热咖啡暖手",
        "忱最近在学拉花，天天拿咖啡练手",
    ]
    for text in others:
        _add_user(store, text)
    results = store.search("妈妈", top_k=5)
    assert len(results) == 5


def test_store_search_does_not_change_state(store):
    _add_user(store, "妈妈平时爱喝咖啡")
    _add_user(store, "姐姐周末也喝咖啡")
    store.add_synonym_edge("万家纳指", "纳斯达克", scope="global",
                           provenance="测试")
    before_rows = {
        r["id"]: r["text"] for r in store.get_all(include_sensitive=True)
    }
    before_edges = copy.deepcopy(store._get_graph().edges)
    store.search("妈妈最近怎么样")
    store.search("纳斯达克最近涨了吗")
    store.search("咖啡", scope="ctx:fund")
    after_rows = {
        r["id"]: r["text"] for r in store.get_all(include_sensitive=True)
    }
    after_edges = copy.deepcopy(store._get_graph().edges)
    assert after_rows == before_rows
    assert after_edges == before_edges
