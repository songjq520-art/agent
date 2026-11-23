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

    def add_relation(self, key, related_key, relation, source_seq=None):
        """加一条记忆—记忆边（relations.py 的四种关系，kind 直接写关系名）。

        同一对记忆、同一关系只留一条；不同关系各留各的（取代和关联
        可以并存）。边的 valid_at / invalid_at 由调用方按账本状态定，
        这里只管连上。
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
                "valid_at": None,
                "invalid_at": None,
                "source_seq": source_seq,
            }
        )
        return True

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
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    # ---------- 检索：PPR 一步多跳 ----------

    def _adjacency(self, include_history=False):
        """邻接表：{节点 id: [(邻居 id, 权重)]}，只走有效的边。

        边是无向的，两个方向都进表。mentions 权重 1.0，cooccurs
        权重是共现次数（上限 5），关系边权重 1.0。失效边默认不走，
        问“以前”时（include_history=True）才走。
        """
        adjacency = {}
        for edge in self.edges:
            if edge.get("invalid_at") and not include_history:
                continue
            from_id = edge.get("from")
            to_id = edge.get("to")
            if not from_id or not to_id:
                continue
            weight = edge.get("weight")
            if not isinstance(weight, (int, float)) or weight <= 0:
                weight = 1.0
            adjacency.setdefault(from_id, []).append((to_id, float(weight)))
            adjacency.setdefault(to_id, []).append((from_id, float(weight)))
        return adjacency

    def ppr(self, seed_entities, memory_seed_scores=None, include_history=False,
            damping=GRAPH_DAMPING, iterations=GRAPH_ITERATIONS):
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
        adjacency = self._adjacency(include_history=include_history)
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
