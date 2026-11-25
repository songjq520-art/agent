# memory/graph.py（图谱推理：图快照 + PPR 检索）
# 图谱推理第一期（2026-11-18）：给记忆织一张网，找的时候顺着网走。
# 做法（设计方案第四、五节）：
#   - 图是索引，不是账本：从账本事件里长出来，坏了从账本重建，
#     跟 mem0 一个待遇；原话只在账本里，图里不存原话全文，只存
#     “哪条记忆连着哪个实体”（记忆节点的 text 只是一句短标题，
#     用来排错，不超过 15 个字）。
#   - 节点两种：记忆节点（id 是账本 add 事件的 key，原话 + created_at
#     的哈希）+ 实体节点（id 是 "ent:" + 归一化后的实体名）。
#   - 边三种：记忆—实体边（mentions，权重 1.0）、实体—实体边
#     （cooccurs，权重 = 共现次数，上限 5）、记忆—记忆边（就是
#     relations.py 的四种关系，kind 直接写 supersede / supplement /
#     conflict / related，权重 1.0）。
#   - 时间跟账本走：边的 valid_at = 来源记忆进账本的时间；
#     invalid_at = 记忆被取代/删除时那个事件的时间；检索默认只走
#     invalid_at 为空的边，问“以前”时才走失效边，并标“这是以前的”。
#   - PPR 一步多跳（HippoRAG 的路子，纯 Python，不引 igraph）：
#     阻尼 0.5、迭代 20 次；记忆节点的初始权重 = 老路归一化分 × 0.05。
#   - 不做的事：不引图数据库、不引 igraph、不做社区摘要、不拿新结论
#     替换旧的、不拿图去做删改的核对（删改核对还是走
#     tools/memory_tool.py 的 _find_anchor_matches）。
# 快照文件：memory/graph_snapshot.json（和账本同目录，.gitignore
# 里 memory/*.json* 已覆盖，不进仓库；随复活包一起备份）。
#
# 图谱二期（2026-11-24，施工契约「图谱边口径」）：
#   - 边必须分型：已知类型之外的边（存量老快照里没写 kind 的）一律
#     隔离——留在图里可查（untyped_edges），不参与任何检索扩散。
#   - 边带成立/失效时间（valid_at / invalid_at，即 Graphiti 的
#     valid_from / invalidated_at）与来源出处（provenance / evidence）；
#     作废只标失效、永不删（invalidate_edge），失效边不参与正向
#     检索，但能按时间点回查（edges_at）与看全史（edge_history）。
#   - 边的语义参与检索：取代边定向（只许从被取代方走向取代方，
#     不给被取代方正向扩散）；冲突边不作正相关（邻接里直接排除）；
#     补充/相关/提及/共现才正向无向。
#   - 同义词边（synonym）连实体节点，必须带作用域 scope：无作用域
#     不许建；检索时只有 scope 与查询上下文相同、或 scope=global
#     的边才走（审批分级是整理员/判官的事，这里只落字段与过滤）。
#   - expand_one_hop：一跳联想扩展（search 的图谱路用它，只读，
#     不改任何边与状态）。

import json
import math
import os
from pathlib import Path

from memory import ledger as _ledger
from memory import relations as _relations
from memory.graph_entities import (
    ENTITY_KIND_FUND,
    ENTITY_KIND_HABIT,
    ENTITY_KIND_PERSON,
    ENTITY_KIND_THING,
    extract_entities_free,
)

GRAPH_VERSION = 1
GRAPH_DAMPING = 0.5
GRAPH_ITERATIONS = 20
GRAPH_PASSAGE_WEIGHT = 0.05
GRAPH_MAX_COOC_WEIGHT = 5
GRAPH_NODE_TITLE_MAX = 15

EDGE_MENTIONS = "mentions"
EDGE_COOCCURS = "cooccurs"
EDGE_SYNONYM = "synonym"

# 同义词边的作用域：global = 全局生效（建它要过更严的审批，见契约
# 拍板 5）；person = 人名/称呼类（同样走严审批）；其余字符串是
# 提出它的上下文/项目名，只在同名上下文里生效。
SCOPE_GLOBAL = "global"
SCOPE_PERSON = "person"

# 已知边类型全集：不在这里的边一律隔离（不参与扩散）。
KNOWN_EDGE_KINDS = frozenset(
    (EDGE_MENTIONS, EDGE_COOCCURS, EDGE_SYNONYM) + tuple(_relations.RELATION_TYPES)
)
# 正向无向扩散的类型（取代定向、冲突排除、同义词按作用域，各自单独处理）。
_POSITIVE_UNDIRECTED_KINDS = frozenset(
    (
        EDGE_MENTIONS,
        EDGE_COOCCURS,
        _relations.RELATION_SUPPLEMENT,
        _relations.RELATION_RELATED,
    )
)

# 记忆节点标题：只是一句短标题，用来排错，不超过 15 个字——
# 原话不进图（设计方案第五节第 1 条）。
def _memory_title(text):
    cleaned = (text or "").strip().replace("\n", " ")
    if len(cleaned) <= GRAPH_NODE_TITLE_MAX:
        return cleaned
    return cleaned[:GRAPH_NODE_TITLE_MAX] + "…"


def memory_node_id(key):
    """记忆节点的 id：账本 add 事件的 key（原话 + created_at 的哈希）。"""
    return f"mem:{key}"


def entity_node_id(name):
    """实体节点的 id：归一化后的实体名。"""
    return f"ent:{name}"


def node_key(node_id):
    """节点 id 还原成账本 key（记忆节点）或实体名（实体节点）。

    不是本图的节点 id 时返回 None。
    """
    if isinstance(node_id, str) and node_id.startswith("mem:"):
        return node_id[4:]
    if isinstance(node_id, str) and node_id.startswith("ent:"):
        return node_id[4:]
    return None


class MemoryGraph:
    """一张小图：节点 + 边，全放内存里，快照落盘是 JSON。

    图小到几百个节点，纯 Python 就够跑 PPR，不用引依赖。
    边是无向的（PPR 走的时候两个方向都算）；mentions / cooccurs /
    关系边的权重见上面。失效边（invalid_at 非空）默认不走，
    问“以前”时才走。
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.nodes = {}  # id -> {"id", "kind", "text"}
        self.edges = []  # {"from", "to", "kind", "valid_at", "invalid_at", "source_seq"}

    # ---------- 快照读写 ----------

    def load(self):
        """读快照文件进内存。文件没有、读坏了返回 False（调用方重建）。

        读坏了不抛异常——图是索引，坏了从账本重建，不拦路。
        """
        if self.path is None or not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        nodes = payload.get("nodes")
        edges = payload.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return False
        self.nodes = {}
        for node in nodes:
            if not isinstance(node, dict) or not node.get("id"):
                continue
            # entities 是记忆节点内部记的“这次抽到的实体名”——
            # add_memory 幂等就靠它（同一条记忆重放时实体没变就不动，
            # 共现权重不会翻倍）。快照里有就留着，没有就按空算。
            entities = node.get("entities")
            self.nodes[node["id"]] = {
                "id": node["id"],
                "kind": node.get("kind") or "",
                "text": node.get("text") or "",
                "entities": list(entities) if isinstance(entities, list) else [],
            }
        self.edges = []
        for edge in edges:
            if not isinstance(edge, dict) or not edge.get("from") or not edge.get("to"):
                continue
            # weight 只在 cooccurs 边上有（共现次数，上限 5），
            # 别的边没有就是 1.0——读回来要留着，不然重算 PPR 时
            # 共现权重全变回 1.0。
            record = {
                "from": edge["from"],
                "to": edge["to"],
                "kind": edge.get("kind") or "",
                "valid_at": edge.get("valid_at"),
                "invalid_at": edge.get("invalid_at"),
                "source_seq": edge.get("source_seq"),
                # 二期字段：作用域 / 来源 / 证据 / 作废理由。老快照
                # 没有就按空——同义词边缺 scope 时按隔离处理
                # （_scope_applies 对空 scope 永不放行）。
                "scope": edge.get("scope"),
                "provenance": edge.get("provenance") or "",
                "evidence": edge.get("evidence") or "",
                "invalid_reason": edge.get("invalid_reason") or "",
            }
            weight = edge.get("weight")
            if isinstance(weight, (int, float)) and weight > 0:
                record["weight"] = weight
            self.edges.append(record)
        return True

    def save(self):
        """快照写盘（原子写：临时文件 + 替换，和账本快照同一套做法）。

        没设 path 时返回 False；写失败抛 OSError（调用方记日志、
        不拦存入——图写不进去，记忆本身已经进账本了）。
        """
        if self.path is None:
            return False
        payload = {
            "version": GRAPH_VERSION,
            "nodes": [dict(self.nodes[key]) for key in sorted(self.nodes)],
            "edges": [dict(edge) for edge in self.edges],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        return True

    # ---------- 建图 ----------

    def _edge_index(self, from_id, to_id, kind):
        """找一条边在 self.edges 里的下标（无向：两个方向都认）。

        找不到返回 None。
        """
        for index, edge in enumerate(self.edges):
            if edge.get("kind") != kind:
                continue
            if (edge.get("from") == from_id and edge.get("to") == to_id) or (
                edge.get("from") == to_id and edge.get("to") == from_id
            ):
                return index
        return None

    def _recompute_cooccurrence(self, entity_a, entity_b):
        """重算两个实体之间的共现权重：同时连着的有效记忆节点数，上限 5。

        共现权重从记忆—实体边现算，不单记数——同一条记忆重放
        （重建、恢复）时不会把数翻倍；记忆失效后，共现权重也跟着降。
        数到 0 时把这条边删掉。
        """
        node_a = entity_node_id(entity_a)
        node_b = entity_node_id(entity_b)
        mems_a = {
            edge["from"]
            for edge in self.edges
            if edge.get("kind") == EDGE_MENTIONS
            and edge.get("to") == node_a
            and not edge.get("invalid_at")
        } | {
            edge["to"]
            for edge in self.edges
            if edge.get("kind") == EDGE_MENTIONS
            and edge.get("from") == node_a
            and not edge.get("invalid_at")
        }
        mems_b = {
            edge["from"]
            for edge in self.edges
            if edge.get("kind") == EDGE_MENTIONS
            and edge.get("to") == node_b
            and not edge.get("invalid_at")
        } | {
            edge["to"]
            for edge in self.edges
            if edge.get("kind") == EDGE_MENTIONS
            and edge.get("from") == node_b
            and not edge.get("invalid_at")
        }
        shared = mems_a & mems_b
        # 只数记忆节点（mentions 边的另一头一定是记忆节点，这里是双保险）。
        shared = {node for node in shared if node.startswith("mem:")}
        index = self._edge_index(node_a, node_b, EDGE_COOCCURS)
        if not shared:
            if index is not None:
                del self.edges[index]
            return
        weight = min(len(shared), GRAPH_MAX_COOC_WEIGHT)
        if index is None:
            self.edges.append(
                {
                    "from": node_a,
                    "to": node_b,
                    "kind": EDGE_COOCCURS,
                    "valid_at": None,
                    "invalid_at": None,
                    "source_seq": None,
                }
            )
            index = len(self.edges) - 1
        self.edges[index]["weight"] = weight

    def add_memory(self, key, text, entities, valid_at=None, source_seq=None):
        """把一条记忆接进图：记忆节点 + 记忆—实体边 + 实体两两共现边。

        entities：[(实体名, 类别)]，调用方抽好的（免费或 LLM 兜底）。
        同一个 key 再加一次时是幂等的：实体集合没变就不动；变了就
        先拆旧边、再按新的连（共现权重现算，不会翻倍）。
        返回这条记忆连上的实体名列表。
        """
        if not key:
            return []
        mem_id = memory_node_id(key)
        # 归一化、去重后的实体名（保持传入顺序）。
        names = []
        seen = set()
        for name, _kind in entities or []:
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        old = self.nodes.get(mem_id)
        old_entities = []
        if old is not None:
            old_entities = list(old.get("entities") or [])
        if old is not None and old_entities == names:
            return names
        # 实体变了：先拆这条记忆的旧边（mentions + 受影响的共现重算）。
        if old is not None:
            self.edges = [
                edge
                for edge in self.edges
                if not (
                    edge.get("kind") == EDGE_MENTIONS
                    and (edge.get("from") == mem_id or edge.get("to") == mem_id)
                )
            ]
            for i in range(len(old_entities)):
                for j in range(i + 1, len(old_entities)):
                    self._recompute_cooccurrence(old_entities[i], old_entities[j])
        self.nodes[mem_id] = {
            "id": mem_id,
            "kind": "memory",
            "text": _memory_title(text),
            "entities": names,
        }
        for name in names:
            ent_id = entity_node_id(name)
            if ent_id not in self.nodes:
                self.nodes[ent_id] = {"id": ent_id, "kind": "entity", "text": name}
            if self._edge_index(mem_id, ent_id, EDGE_MENTIONS) is None:
                self.edges.append(
                    {
                        "from": mem_id,
                        "to": ent_id,
                        "kind": EDGE_MENTIONS,
                        "valid_at": valid_at,
                        "invalid_at": None,
                        "source_seq": source_seq,
                    }
                )
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self._recompute_cooccurrence(names[i], names[j])
        return names

    def add_relation(self, key, related_key, relation, source_seq=None,
                     provenance="", evidence="", valid_at=None):
        """加一条记忆—记忆边（relations.py 的四种关系，kind 直接写关系名）。

        同一对记忆、同一关系只留一条；不同关系各留各的（取代和关联
        可以并存）。取代边的方向约定：from 是被取代的旧记忆、to 是
        取代它的新记忆（重建时按账本先后就是这个方向），检索扩散
        只许顺这个方向走。valid_at / 出处由调用方按账本状态带进来，
        invalid_at 由 sync_validity / mark_invalid 按状态标。
        """
        if not key or not related_key or key == related_key:
            return False
        if relation not in _relations.RELATION_TYPES:
            return False
        from_id = memory_node_id(key)
        to_id = memory_node_id(related_key)
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        if self._edge_index(from_id, to_id, relation) is not None:
            return False
        self.edges.append(
            {
                "from": from_id,
                "to": to_id,
                "kind": relation,
                "valid_at": valid_at,
                "invalid_at": None,
                "source_seq": source_seq,
                "scope": None,
                "provenance": provenance or "",
                "evidence": evidence or "",
                "invalid_reason": "",
            }
        )
        return True

    # ---------- 同义词边与作废（二期） ----------

    @staticmethod
    def _scope_applies(edge, scope):
        """这条同义词边在给定查询上下文里生不生效。

        global 边处处生效；其余边只在 scope 完全相同才生效；
        查询没带上下文（scope=None）时只有 global 边生效；
        边本身没 scope（存量缺字段）永不生效（隔离）。
        """
        edge_scope = (edge or {}).get("scope")
        if not edge_scope:
            return False
        if edge_scope == SCOPE_GLOBAL:
            return True
        if scope is None:
            return False
        return edge_scope == scope

    def add_synonym_edge(self, name_a, name_b, scope, provenance="",
                         evidence="", valid_at=None, source_seq=None):
        """连一条同义词边（两个实体节点之间）：同一个东西的两种叫法。

        scope 必填：没有作用域不许建（契约：无作用域的全局同义边
        不许直接建；默认只在提出它的上下文内生效，想全局就明写
        global——那要过更严的审批，审批不在这里管）。同一对实体、
        同一作用域已有一条有效的边时不重复连；只有作废边时新连
        一条（旧边留痕）。返回 True 表示连上了。
        """
        from memory.graph_entities import normalize_entity_name

        a = normalize_entity_name(name_a)
        b = normalize_entity_name(name_b)
        if not a or not b or a == b:
            return False
        if not isinstance(scope, str) or not scope.strip():
            return False
        scope = scope.strip()
        from_id = entity_node_id(a)
        to_id = entity_node_id(b)
        for edge in self.edges:
            if edge.get("kind") != EDGE_SYNONYM or edge.get("invalid_at"):
                continue
            if edge.get("scope") != scope:
                continue
            if {edge.get("from"), edge.get("to")} == {from_id, to_id}:
                return False
        for name, node_id in ((a, from_id), (b, to_id)):
            if node_id not in self.nodes:
                self.nodes[node_id] = {"id": node_id, "kind": "entity", "text": name}
        self.edges.append(
            {
                "from": from_id,
                "to": to_id,
                "kind": EDGE_SYNONYM,
                "valid_at": valid_at,
                "invalid_at": None,
                "source_seq": source_seq,
                "scope": scope,
                "provenance": provenance or "",
                "evidence": evidence or "",
                "invalid_reason": "",
            }
        )
        return True

    def invalidate_edge(self, from_id, to_id, kind, invalid_at, reason=""):
        """作废一条边：只标 invalid_at，边留在图里（Graphiti 作废法）。

        必须带作废时间（不带不废，免得留下一条说不清什么时候废的
        边）。只废第一条还有效的同型边；已经废了的不重复标。
        返回 True 表示这次标上了。
        """
        if not from_id or not to_id or not kind or not invalid_at:
            return False
        index = self._edge_index(from_id, to_id, kind)
        if index is None:
            return False
        edge = self.edges[index]
        if edge.get("invalid_at"):
            return False
        edge["invalid_at"] = invalid_at
        edge["invalid_reason"] = reason or ""
        return True

    def edge_history(self, from_id, to_id, kind=None):
        """两个节点之间的全部边（有效的 + 作废的），留痕回查用。

        每条带成立/失效时间、来源、证据、作废理由——查一条边
        「哪来的、什么时候废的、为什么废」看这里。按 source_seq 排。
        """
        out = []
        for edge in self.edges:
            if kind is not None and edge.get("kind") != kind:
                continue
            if {edge.get("from"), edge.get("to")} != {from_id, to_id}:
                continue
            out.append(dict(edge))
        out.sort(key=lambda e: e.get("source_seq") if isinstance(e.get("source_seq"), int) else 0)
        return out

    def edges_at(self, when, kind=None):
        """某个时间点上成立的边：valid_at <= when < invalid_at。

        时间是 ISO 字符串（和账本 ts 同格式，字符串比较即可）。
        valid_at 为空按「一直成立」算，invalid_at 为空按「还没废」算。
        """
        if not when:
            return []
        out = []
        for edge in self.edges:
            if kind is not None and edge.get("kind") != kind:
                continue
            valid_at = edge.get("valid_at")
            invalid_at = edge.get("invalid_at")
            if valid_at and str(valid_at) > str(when):
                continue
            if invalid_at and str(when) >= str(invalid_at):
                continue
            out.append(dict(edge))
        return out

    def untyped_edges(self):
        """存量无类型/未知类型的边清单（隔离中，不参与检索）。

        老快照里没写 kind、或 kind 不是已知类型的边都在这里——
        留着可查，但语义不明，不许拿它们做扩散（契约：关系边
        必须分型）。新代码连的边全有类型，不会进这个清单。
        """
        return [dict(e) for e in self.edges if e.get("kind") not in KNOWN_EDGE_KINDS]

    def mark_invalid(self, key, invalid_at):
        """一条记忆失效（被取代/删除）：它连出去的边都标 invalid_at。

        边不删，只标——原话和边都留着，问“以前”时还能顺着找回来。
        """
        if not key:
            return
        mem_id = memory_node_id(key)
        for edge in self.edges:
            if edge.get("from") == mem_id or edge.get("to") == mem_id:
                if not edge.get("invalid_at"):
                    edge["invalid_at"] = invalid_at

    def sync_validity(self, key_states, invalid_ts_by_key=None):
        """按账本状态同步边的有效期（单一事实来源是账本）。

        key_states：ledger.resolve_state 的 key_states（key → current /
        superseded / dormant / deleted）；invalid_ts_by_key：key → 被取
        代/删除的那个事件的时间（没有就按已有的 invalid_at 留着，
        再没有就写 None——PPR 只看 invalid_at 有没有值，不看具体时间）。
        睡着的（dormant）不标失效：它是衰减标签，不是“不是当前了”，
        边照常走（和 project_and_filter 同一套口径）。
        失效的边标上，被 revive 回 current 的边把 invalid_at 清掉。
        """
        invalid_ts_by_key = invalid_ts_by_key or {}
        for edge in self.edges:
            for node_id in (edge.get("from"), edge.get("to")):
                if not isinstance(node_id, str) or not node_id.startswith("mem:"):
                    continue
                key = node_id[4:]
                state = (key_states or {}).get(key)
                if state in (
                    _ledger.LEDGER_STATE_SUPERSEDED,
                    _ledger.LEDGER_STATE_DELETED,
                ):
                    if not edge.get("invalid_at"):
                        edge["invalid_at"] = invalid_ts_by_key.get(key)
                elif state == _ledger.LEDGER_STATE_CURRENT:
                    edge["invalid_at"] = None

    # ---------- 从账本重建 ----------

    def rebuild_from_events(self, events):
        """从账本事件重建整张图（纯函数式重放，不动账本）。

        按 seq 顺序重放 add 事件：抽实体（免费的，不调模型——重建
        是启动时的兜底，不能一次调几十次模型）、连边、判关系
        （classify_relation，跟库里已有的记忆判，判出来的写成
        记忆—记忆边）。supersede / delete / dormant / revive 事件
        按账本状态同步边的有效期。快照丢了、格式坏了就走这条路。
        实体抽取的 LLM 兜底只在新记忆存入时走（快照里留着），
        重建时不补——重建要的是“图能回来”，不是“和原来一模一样”。
        """
        self.nodes = {}
        self.edges = []
        adds = []
        for event in events or []:
            if event.get("type") != _ledger.LEDGER_EVENT_ADD:
                continue
            adds.append(event)
        adds.sort(key=lambda e: e.get("seq") if isinstance(e.get("seq"), int) else 0)
        # 先按 key_states + 失效时间算好每条记忆的状态（一次扫完，
        # 不用每条都翻事件）。
        key_states = {}
        invalid_ts = {}
        for event in events or []:
            etype = event.get("type")
            key = event.get("key")
            if not key:
                continue
            if etype == _ledger.LEDGER_EVENT_ADD:
                key_states[key] = _ledger.LEDGER_STATE_CURRENT
            elif etype in (
                _ledger.LEDGER_EVENT_SUPERSEDE,
                _ledger.LEDGER_EVENT_DELETE,
            ):
                key_states[key] = (
                    _ledger.LEDGER_STATE_SUPERSEDED
                    if etype == _ledger.LEDGER_EVENT_SUPERSEDE
                    else _ledger.LEDGER_STATE_DELETED
                )
                invalid_ts[key] = event.get("ts")
            elif etype == _ledger.LEDGER_EVENT_REVIVE:
                key_states[key] = _ledger.LEDGER_STATE_CURRENT
                invalid_ts.pop(key, None)
            elif etype == _ledger.LEDGER_EVENT_DORMANT:
                key_states[key] = _ledger.LEDGER_STATE_DORMANT
        # 逐条加记忆、判关系（只和已加进图的记忆判，顺序和账本一致）。
        # 候选集和存入时一致：只和“连着同一实体”的旧记忆判（见
        # mem0_store._graph_add_memory）——全库扫会多判出边，重建出
        # 来的图就和 live 建出来的不一样（2026-11-18 测出来的差一条
        # conflict 边就是这么来的）。
        seen = []  # [(key, text, meta, {entity names})]
        for event in adds:
            key = event.get("key")
            text = event.get("text") or ""
            meta = dict(event.get("meta") or {})
            if not key or not text:
                continue
            entities = extract_entities_free(text)
            self.add_memory(
                key,
                text,
                entities,
                valid_at=event.get("ts"),
                source_seq=event.get("seq") if isinstance(event.get("seq"), int) else None,
            )
            new_names = {name for name, _kind in entities}
            for old_key, old_text, old_meta, old_names in seen:
                if not (new_names & old_names):
                    continue
                relation = _relations.classify_relation(old_text, old_meta, text, meta)
                if relation is not None:
                    self.add_relation(
                        old_key,
                        key,
                        relation,
                        source_seq=event.get("seq") if isinstance(event.get("seq"), int) else None,
                    )
            seen.append((key, text, meta, new_names))
        # 账本里的 relate 事件也写成边（显式记的关系，和自动判的并存）。
        for event in events or []:
            if event.get("type") != _ledger.LEDGER_EVENT_RELATE:
                continue
            # relate 事件的 memory_id 是 mem0 id，不是 key——图里记忆节点
            # 按 key 认，这里只能认 key 形式的 id；mem0 id 的关系
            # 走 add 时的自动判，不在这里补。
            key = event.get("memory_id")
            related_key = event.get("related_id")
            if (
                isinstance(key, str)
                and isinstance(related_key, str)
                and memory_node_id(key) in self.nodes
                and memory_node_id(related_key) in self.nodes
            ):
                self.add_relation(
                    key,
                    related_key,
                    event.get("relation"),
                    source_seq=event.get("seq") if isinstance(event.get("seq"), int) else None,
                )
        self.sync_validity(key_states, invalid_ts)
        # 图谱二期：账本里的 graph_edge 事件最后重放（在 sync_validity
        # 之后——显式作废不许被状态同步冲掉；显式新增的同义词边也
        # 不在记忆状态的管辖内）。按 seq 顺序，add 连边、invalidate
        # 标失效，边的新增/作废都以账本为准，快照丢了能原样回来。
        edge_events = [
            event
            for event in (events or [])
            if event.get("type") == _ledger.LEDGER_EVENT_GRAPH_EDGE
        ]
        edge_events.sort(
            key=lambda e: e.get("seq") if isinstance(e.get("seq"), int) else 0
        )
        for event in edge_events:
            kind = event.get("kind") or ""
            seq = event.get("seq") if isinstance(event.get("seq"), int) else None
            if (event.get("action") or "add") == "invalidate":
                if kind == EDGE_SYNONYM:
                    a = event.get("entity_a")
                    b = event.get("entity_b")
                    if not a or not b:
                        continue
                    from_id, to_id = entity_node_id(a), entity_node_id(b)
                else:
                    if not event.get("from_key") or not event.get("to_key"):
                        continue
                    from_id = memory_node_id(event["from_key"])
                    to_id = memory_node_id(event["to_key"])
                self.invalidate_edge(
                    from_id,
                    to_id,
                    kind,
                    event.get("invalid_at") or event.get("ts"),
                    reason=event.get("reason") or "",
                )
                continue
            if kind == EDGE_SYNONYM:
                self.add_synonym_edge(
                    event.get("entity_a"),
                    event.get("entity_b"),
                    scope=event.get("scope"),
                    provenance=event.get("provenance") or "",
                    evidence=event.get("evidence") or "",
                    valid_at=event.get("valid_at") or event.get("ts"),
                    source_seq=seq,
                )
            elif kind in _relations.RELATION_TYPES:
                if event.get("from_key") and event.get("to_key"):
                    self.add_relation(
                        event["from_key"],
                        event["to_key"],
                        kind,
                        source_seq=seq,
                        provenance=event.get("provenance") or "",
                        evidence=event.get("evidence") or "",
                        valid_at=event.get("valid_at") or event.get("ts"),
                    )
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    # ---------- 检索：PPR 一步多跳 ----------

    def _adjacency(self, include_history=False, scope=None):
        """邻接表：{节点 id: [(邻居 id, 权重)]}，按边的语义走。

        二期口径（契约图谱边口径）：失效边默认不走，问“以前”时
        （include_history=True）才走；冲突边永不作正相关——任何
        模式都不进邻接；取代边定向——只从被取代方（from）走向
        取代方（to），不给被取代方正向扩散；同义词边只走作用域
        对得上的（见 _scope_applies）；无类型/未知类型的边隔离，
        不进邻接。其余（mentions 权重 1.0、cooccurs 权重是共现
        次数上限 5、补充/相关权重 1.0）无向正向，两个方向都进表。
        """
        adjacency = {}
        for edge in self.edges:
            if edge.get("invalid_at") and not include_history:
                continue
            kind = edge.get("kind")
            if kind == _relations.RELATION_CONFLICT:
                continue
            from_id = edge.get("from")
            to_id = edge.get("to")
            if not from_id or not to_id:
                continue
            weight = edge.get("weight")
            if not isinstance(weight, (int, float)) or weight <= 0:
                weight = 1.0
            if kind == _relations.RELATION_SUPERSEDE:
                adjacency.setdefault(from_id, []).append((to_id, float(weight)))
                continue
            if kind == EDGE_SYNONYM:
                if not self._scope_applies(edge, scope):
                    continue
            elif kind not in _POSITIVE_UNDIRECTED_KINDS:
                continue
            adjacency.setdefault(from_id, []).append((to_id, float(weight)))
            adjacency.setdefault(to_id, []).append((from_id, float(weight)))
        return adjacency

    # ---------- 一跳联想扩展（二期检索路，只读） ----------

    def expand_one_hop(self, seed_keys=None, seed_entities=None, scope=None):
        """一跳联想扩展：从种子记忆/种子实体出发，顺有效边走一跳。

        种子实体先经同义词边（作用域对得上的）扩到别名实体，再经
        mentions 边落到记忆；种子记忆经补充/相关边、取代边（只顺
        方向走向新版本）、以及「共享实体」落到同话题的记忆。冲突
        边不走、失效边不走、无类型边不走（与 _adjacency 同口径）。
        打分：查询实体直达的 mentions 计 1.0/个、经同义词绕到的
        计 1.0/个（同义即同一物，不打折）、种子记忆的关系边邻居
        计 1.0、共享实体邻居计 0.5/个共享实体。种子自己不出现在
        结果里。纯读：不改任何边、节点与状态。
        返回 {记忆 key: 分数}，按分数降序。
        """
        seed_keys = {k for k in (seed_keys or []) if k}
        # 有效边先按用途分好（图小，每次现分不缓存，省得缓存失效问题）。
        mentions_by_entity = {}
        mentions_by_memory = {}
        synonym_neighbors = {}
        relation_neighbors = {}
        for edge in self.edges:
            if edge.get("invalid_at"):
                continue
            kind = edge.get("kind")
            from_id = edge.get("from")
            to_id = edge.get("to")
            if not from_id or not to_id:
                continue
            if kind == EDGE_MENTIONS:
                if from_id.startswith("mem:") and to_id.startswith("ent:"):
                    mem_id, ent_id = from_id, to_id
                elif to_id.startswith("mem:") and from_id.startswith("ent:"):
                    mem_id, ent_id = to_id, from_id
                else:
                    continue
                mentions_by_entity.setdefault(ent_id, set()).add(mem_id)
                mentions_by_memory.setdefault(mem_id, set()).add(ent_id)
            elif kind == EDGE_SYNONYM:
                if not self._scope_applies(edge, scope):
                    continue
                synonym_neighbors.setdefault(from_id, set()).add(to_id)
                synonym_neighbors.setdefault(to_id, set()).add(from_id)
            elif kind in (
                _relations.RELATION_SUPPLEMENT,
                _relations.RELATION_RELATED,
            ):
                if from_id.startswith("mem:") and to_id.startswith("mem:"):
                    relation_neighbors.setdefault(from_id, set()).add(to_id)
                    relation_neighbors.setdefault(to_id, set()).add(from_id)
            elif kind == _relations.RELATION_SUPERSEDE:
                if from_id.startswith("mem:") and to_id.startswith("mem:"):
                    # 定向：只许从被取代方走向取代方。
                    relation_neighbors.setdefault(from_id, set()).add(to_id)
            # conflict / 无类型边：不参与扩展。
        scores = {}

        def _hit(mem_id, points):
            if not isinstance(mem_id, str) or not mem_id.startswith("mem:"):
                return
            key = mem_id[4:]
            if key in seed_keys:
                return
            scores[key] = scores.get(key, 0.0) + points

        # 种子实体 →（同义词）→ 实体集 →（mentions）→ 记忆。
        entity_ids = set()
        for name in seed_entities or []:
            node_id = entity_node_id(name)
            if node_id in self.nodes:
                entity_ids.add(node_id)
        expanded_entities = set(entity_ids)
        for node_id in entity_ids:
            expanded_entities |= synonym_neighbors.get(node_id, set())
        for ent_id in expanded_entities:
            for mem_id in mentions_by_entity.get(ent_id, set()):
                _hit(mem_id, 1.0)
        # 种子记忆 → 关系边邻居 + 共享实体邻居。
        for key in seed_keys:
            mem_id = memory_node_id(key)
            for neighbor in relation_neighbors.get(mem_id, set()):
                _hit(neighbor, 1.0)
            for ent_id in mentions_by_memory.get(mem_id, set()):
                for other in mentions_by_entity.get(ent_id, set()):
                    if other != mem_id:
                        _hit(other, 0.5)
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def ppr(self, seed_entities, memory_seed_scores=None, include_history=False,
            damping=GRAPH_DAMPING, iterations=GRAPH_ITERATIONS, scope=None):
        """个性化 PageRank：种子实体 → 一步多跳，分数沿着边传开。

        seed_entities：种子实体名列表（调用方从查询里抽出来的，
        在图里找不着的忽略）；memory_seed_scores：{记忆 key:
        老路归一化分}（老路向量检索里那几条的分数，0–1），记忆节点
        的初始权重 = 它 × 0.05（HippoRAG 的 passage_node_weight，
        图是加分项，不能让向量分把图带跑，也不能让图把向量分带跑）。
        种子实体平分 1.0 的初始权重；记忆种子按上面的 0.05 算，
        两份加起来再归一化成重启向量。
        返回 {记忆 key: PPR 分数}（只含记忆节点，按分数降序），
        空图、没种子时返回 {}。
        """
        adjacency = self._adjacency(include_history=include_history, scope=scope)
        if not adjacency:
            return {}
        personalization = {}
        valid_seeds = [
            name for name in (seed_entities or []) if entity_node_id(name) in adjacency
        ]
        if valid_seeds:
            share = 1.0 / len(valid_seeds)
            for name in valid_seeds:
                personalization[entity_node_id(name)] = share
        for key, score in (memory_seed_scores or {}).items():
            if not isinstance(score, (int, float)) or score <= 0:
                continue
            node_id = memory_node_id(key)
            if node_id not in adjacency:
                continue
            personalization[node_id] = personalization.get(node_id, 0.0) + (
                float(score) * GRAPH_PASSAGE_WEIGHT
            )
        total = sum(personalization.values())
        if total <= 0:
            return {}
        for node_id in personalization:
            personalization[node_id] /= total
        # 出边权重和（归一化转移概率用）。
        out_weight = {
            node_id: sum(weight for _n, weight in neighbors)
            for node_id, neighbors in adjacency.items()
        }
        scores = dict(personalization)
        for _ in range(max(1, int(iterations))):
            nxt = {}
            for node_id, neighbors in adjacency.items():
                out = out_weight.get(node_id) or 0.0
                if out <= 0:
                    continue
                share = scores.get(node_id, 0.0) * damping / out
                if share == 0.0:
                    continue
                for neighbor, weight in neighbors:
                    nxt[neighbor] = nxt.get(neighbor, 0.0) + share * weight
            for node_id in adjacency:
                base = (1.0 - damping) * personalization.get(node_id, 0.0)
                scores[node_id] = base + nxt.get(node_id, 0.0)
        out = {}
        for node_id, score in scores.items():
            if node_id.startswith("mem:") and score > 0:
                out[node_id[4:]] = score
        return dict(sorted(out.items(), key=lambda item: item[1], reverse=True))

    # ---------- 关系查询（读图） ----------

    def relations_for(self, key):
        """一条记忆在图上连着的记忆—记忆边（读图版 get_relations）。

        返回 [{"memory_id", "related_id", "relation", "note", "seq"}]，
        双向都认，按 seq 排。relations.py 的 get_relations 还是读
        账本（relate 事件），这里是图上的同一份关系——自动判的
        关系只在图里，显式记的两种地方都有。
        """
        if not key:
            return []
        mem_id = memory_node_id(key)
        out = []
        for edge in self.edges:
            if edge.get("kind") not in _relations.RELATION_TYPES:
                continue
            if edge.get("from") == mem_id:
                other = edge.get("to")
            elif edge.get("to") == mem_id:
                other = edge.get("from")
            else:
                continue
            if not isinstance(other, str) or not other.startswith("mem:"):
                continue
            out.append(
                {
                    "memory_id": key,
                    "related_id": other[4:],
                    "relation": edge.get("kind"),
                    "note": "",
                    "seq": edge.get("source_seq") if isinstance(edge.get("source_seq"), int) else 0,
                    # 二期：作废的边也列出来（标 invalid_at），带出处——
                    # 查关系时能分清「还连着」和「连过、已作废」。
                    "valid_at": edge.get("valid_at"),
                    "invalid_at": edge.get("invalid_at"),
                    "provenance": edge.get("provenance") or "",
                }
            )
        out.sort(key=lambda r: r.get("seq") or 0)
        return out

    def is_linked(self, key):
        """这条记忆在图上有没有边连着（沉底前查三道里的第三道，读图版）。

        记忆—实体边、记忆—记忆边都算连着；失效边不算（被取代的
        记忆不该因为旧边就赖着不沉）。
        """
        if not key:
            return False
        mem_id = memory_node_id(key)
        for edge in self.edges:
            if edge.get("invalid_at"):
                continue
            if edge.get("from") == mem_id or edge.get("to") == mem_id:
                return True
        return False


def load_graph(path):
    """读一张图（快照在就读快照，没有就空图）。

    返回 MemoryGraph。调用方拿着它再决定要不要从账本重建。
    """
    graph = MemoryGraph(path)
    graph.load()
    return graph


def build_graph_from_events(events, path=None):
    """从账本事件建一张新图（重建的入口）。

    path 给了就把快照写盘（2026-11-19 真机教训：之前这里只建图
    不写盘，笔记本上跑完 34 条、Test-Path 还是 False，快照一直
    没有，服务只能每次启动现重建）。写盘失败抛 OSError——手动
    重建时调用方要看得见；服务里走 _get_graph 那条路，写失败只
    记日志（见 mem0_store._get_graph）。

    返回 (graph, {"nodes", "edges", "saved"})。
    """
    graph = MemoryGraph(path)
    info = graph.rebuild_from_events(events)
    info["saved"] = graph.save() if path is not None else False
    return graph, info
