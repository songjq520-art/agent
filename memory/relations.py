# memory/relations.py（记忆之间的关系：只留四种）
# 最终方案里的“关系”那块（GPT 第二版里丢了，合起来时补回来）：
# 两条记忆之间只有四种关系——取代 / 补充 / 冲突 / 关联，别的不认。
#   - 取代（supersede）：同一件事的新说法赢了，旧的标“被取代”，原话留着。
#     判法走现成的：同槽位新值（_money_slot_relation == "supersede"）
#     或 _resolve_conflict 判了 supersede_old。
#   - 补充（supplement）：同一件事又出现一次，新说法是旧条的证据，
#     不另建主条（去重判“重复”时就是这个关系，见 _append_duplicate_evidence）。
#   - 冲突（conflict）：同一件事、说法对着干（槽位值冲突、或
#     _resolve_conflict 判了 ask），留着等 BOOS 定，不自动顶掉。
#   - 关联（related）：同一话题、不同事实（关键词重合、但不是同一件事），
#     只记“连着”，不分胜负——删一条之前要查“有没有别的记忆连着它”，
#     连着的就不能悄悄删（沉底前查三道里的第三道）。
# 关系存在账本里（LEDGER_EVENT_RELATE 事件，见 memory/ledger.py），
# 不另建表；这里只做“判”这个纯函数 + 读关系记录的帮手。

from memory.keywords import _extract_keywords, _keyword_weight
from memory.redaction import _normalize_memory_text
from memory.money_slots import (
    _extract_money_metadata,
    _known_money,
    _money_slot_relation,
    _slot_values_conflict,
)
from memory.provenance import _normalize_metadata, _resolve_conflict

RELATION_SUPERSEDE = "supersede"
RELATION_SUPPLEMENT = "supplement"
RELATION_CONFLICT = "conflict"
RELATION_RELATED = "related"
RELATION_TYPES = (
    RELATION_SUPERSEDE,
    RELATION_SUPPLEMENT,
    RELATION_CONFLICT,
    RELATION_RELATED,
)


def classify_relation(old_text, old_metadata, new_text, new_metadata):
    """判两条记忆是什么关系（纯函数，不动库）。

    参数是两条记忆的原文 + metadata（raw 或规整过的都行）。
    返回四种之一，或 None（没关系——话题都不挨着）。
    判的顺序（从“同一件事”到“同一话题”，由紧到松）：
      1. 同槽位金额：新值换旧值 → 取代；同值 → 补充。
      2. 槽位值冲突（比如喜欢吃鸡杂面 vs 对鸡杂面过敏）→ 冲突。
      3. 同一件事：原文规整后相同、或 _resolve_conflict 能判出胜负
         （新赢旧 → 取代；新被挂起 → 冲突；分不出 → 冲突，等 BOOS 定）。
         注意：这里只看“是不是同一件事”的强证据——原文相同、或
         槽位相同；关键词重合不算“同一件事”（“喜欢喝咖啡”和
         “喜欢喝拿铁”能同时是真的），那是第 4 步的关联。
      4. 关键词重合（权重 ≥2，和检索的补充信号同线）→ 关联。
    模型判“是不是同一件事”还没接进来（去重裁决在调用方），所以
    这里不猜语义相似度——猜错关系比没关系更糟（顶掉一条不该顶的）。
    """
    old_meta = _normalize_metadata(old_metadata)
    new_meta = _normalize_metadata(new_metadata)
    # 1. 同槽位金额。
    old_money = _known_money({"metadata": old_meta, "text": old_text})
    new_money = _extract_money_metadata(new_text)
    slot_relation = _money_slot_relation(new_money, old_money)
    if slot_relation == "supersede":
        return RELATION_SUPERSEDE
    if slot_relation == "duplicate":
        return RELATION_SUPPLEMENT
    # 2. 槽位值冲突（偏好句宾语对着干）。
    if _slot_values_conflict(new_text, old_text):
        return RELATION_CONFLICT
    # 3. 同一件事：原文规整后相同。
    if _normalize_memory_text(old_text or "") and _normalize_memory_text(
        old_text or ""
    ) == _normalize_memory_text(new_text or ""):
        return RELATION_SUPPLEMENT
    old_slot = old_meta.get("slot")
    new_slot = new_meta.get("slot")
    if old_slot and old_slot == new_slot:
        verdict = _resolve_conflict(old_meta, new_meta)
        if verdict["action"] == "supersede_old":
            return RELATION_SUPERSEDE
        if verdict["action"] in ("pending_new", "ask"):
            return RELATION_CONFLICT
    # 4. 同一话题：关键词重合。
    keywords = _extract_keywords(new_text or "")
    if keywords:
        weight, _matched = _keyword_weight(old_text or "", keywords)
        if weight >= 2:
            return RELATION_RELATED
    return None


def get_relations(state, memory_id):
    """从账本折叠出的状态里，取一条记忆连着的关系记录（纯函数）。

    双向都认：memory_id 在哪一边都算连着。返回关系记录列表
    （每项 {"memory_id", "related_id", "relation", "note", "seq"}，
    按 seq 排）。"""
    out = []
    for record in (state or {}).get("relations") or []:
        if record.get("memory_id") == memory_id or record.get("related_id") == memory_id:
            out.append(dict(record))
    out.sort(key=lambda r: r.get("seq") if isinstance(r.get("seq"), int) else 0)
    return out


def is_linked(state, memory_id):
    """这条记忆有没有别的记忆连着它（沉底前查三道里的第三道）。

    连着就不沉、不删——删之前要先问 BOOS，不能悄悄把有关联的
    记忆清掉。"""
    return bool(get_relations(state, memory_id))
