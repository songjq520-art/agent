# tests/test_dedup_event_vs_preference.py（回归：2026-11-20 真机事故）
# BOOS 在网页上说“我昨天晚上在盐湖城吃了一碗牛肉面，特别好吃”，
# 提取出的新记忆（id f70c1737）1 秒内就被删了，库里只剩旧记忆
# “BOOS 喜欢吃牛肉面”（write_seq=22）。
#
# 根因（日志 + 代码 + 真实复现已确认）不是去重裁决，而是价值关卡：
#   1. _apply_coverage_guard 先把新条标了待确认（pending_confirmation）——
#      注意不是因为撞上了“喜欢吃牛肉面”那条（那条是 inferred、不算核心，
#      守卫根本不认它）；撞上的是“DeepSeek 已接通（2026-11-07）……”那条
#      已确认记忆，靠的只是日期数字碎片（02/11/20/26）关键词重合 ≥4。
#      守卫的决定是“挂起来等 BOOS 定”，不是删。
#   2. _value_gate_extraction_results 随后请 DeepSeek 打价值分：同一条
#      原文、同一个提示词、temperature=0，复现打分 0.1 分（“一次性经历、
#      低价值”），低于 VALUE_THRESHOLD=0.5，直接 _rollback_new_memories
#      删掉。守卫刚挂起的待确认条，就这么被下一道关卡删了。
#   3. _dedupe_extraction_results 根本没见到这条（已被删）。佐证：
#      守卫日志 08:56:25.175 到删除 08:56:26.146 之间只有 1 次 DeepSeek
#      调用（价值打分那次）；去重若判了 duplicate，会在旧条上挂证据
#      （_append_duplicate_evidence）并记一笔出现（_record_decay_occurrence），
#      两处都要调 update——日志里一次 update 都没有，旧条 metadata 里
#      也没有 evidence 字段；向量相似度实测只有 0.662，够不上硬合并线
#      0.90；把去重裁决的原提示词原样重放，模型说 is_duplicate=false、
#      confidence=0.95（“候选是喜好，新记忆是具体时间地点的一次经历，
#      不是同一事实”）。去重裁决本身是对的。
#
# 所以修的方向也在价值关卡（见下面各测试的注释），外加去重通用分支
# 同样不认“守卫已挂起”这个潜伏 bug 的防守：
#   修 1（根因，必须）：_value_gate_extraction_results 不删守卫已挂起
#       （pending_confirmation）的条——守卫在前、关卡在后，后面的关卡
#       不能把守卫“挂起来等 BOOS”的决定翻过来（去重里金额分支已有
#       同款注释和处理，价值关卡缺了这一道）。
#   修 2（根因，必须，程序规则优先于提示词）：带具体日期的事件记忆
#       不走价值删除——extract_core_slots 里有 date: 槽位的新记忆
#       （如“2026年11月19日晚上在盐湖城吃了一碗牛肉面”）是 episodic
#       记忆，本来就该靠衰减慢慢沉底，不该被价值关卡当“一次性琐事”
#       删掉；删之前至少要有这条程序规则兜底。
#   修 3（防守，去重通用分支）：_dedupe_extraction_results 里判成
#       duplicate 的通用分支（约 4895 行），像金额分支那样先看
#       pending_confirmation：已挂起的留着、不删、不挂证据。
#   修 4（防守，去重裁决）：_adjudicate_duplicate 里加程序规则——
#       新记忆带 date: 槽位、候选没有时，直接判 new，不送模型。
#       （本次事故里模型判对了，但这条规则让“事件 vs 喜好”可测、
#       确定，不依赖模型每次都判对。）
#   修 5（次要，提示词）：_judge_value_scores 的提示词补一句——
#       “带具体日期/地点的一次性经历是值得记的 episodic 记忆，
#       不是琐事，不打低分”。只当兜底，不能代替修 2。
#
# 全部离线、确定性：FakeMemory + monkeypatch 掉模型调用（打分和裁决
# 都用生产复现时的真实返回值：价值 0.1 分、去重 is_duplicate=false /
# confidence=0.95、向量相似度 0.662），不联网、不花钱。
# 在当前代码（474c0d1）上：test_1、test_2、test_5 失败（复现事故），
# test_3、test_4、test_6 通过（去重裁决本身是对的、真重复照常删）。

from conftest import make_fake_store
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    SOURCE_INFERRED,
    _provenance,
    _slot_values_conflict,
    extract_core_slots,
)

NEW_TEXT = "用户于2026年11月19日晚上在盐湖城吃了一碗牛肉面，觉得特别好吃。"
OLD_PREF = "BOOS喜欢吃牛肉面。"
# 生产库里真实撞上守卫的那条已确认记忆（关键词重合靠日期数字碎片）。
CORE_TEXT = "DeepSeek已接通（2026-11-07），真实调用成功，是第一任主控。"
# 生产复现：价值打分 0.1、去重相似度 0.662、去重裁决 not duplicate / 0.95。
VALUE_SCORE_PROD = 0.1
SIM_PROD = 0.662


def _batch_meta():
    return _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False, turn_id="test-turn")


def _seed(store):
    """库里先有旧喜好 + 那条已确认记忆，拍提取前快照，再存入牛肉面事件新条。
    快照必须在新条存入之前拍（生产里 before 就是提取前快照、不含新条），
    否则去重会把新条和快照里的自己认成“完全相同”而删掉。
    返回 (pref_id, core_id, new_id, before_snapshot, old_pref_item)。"""
    pref = store.add_memory_text(OLD_PREF, source="inferred")
    pref_id = pref["results"][0]["id"]
    core = store.add_memory_text(CORE_TEXT)
    core_id = core["results"][0]["id"]
    before = store._snapshot_all()
    old_pref = next(item for item in before if item["id"] == pref_id)
    new = store.add_memory_text(NEW_TEXT, source="inferred")
    new_id = new["results"][0]["id"]
    return pref_id, core_id, new_id, before, old_pref


def _texts(store):
    return store._memory.all_texts()


# ---------- 事故复现：守卫挂起 → 价值关卡删掉（根因，修 1 + 修 2） ----------


def test_1_value_gate_must_not_delete_guard_held_event(tmp_path, monkeypatch):
    """生产复现：守卫刚把事件标待确认，价值关卡 0.1 分就把它删了。
    当前代码失败：新条被 _rollback_new_memories 删掉。修 1 后通过：
    pending_confirmation 的条价值关卡不删。”
    """
    store = make_fake_store(tmp_path)
    _pref_id, _core_id, new_id, before, _old = _seed(store)
    results = {"results": [{"id": new_id, "memory": NEW_TEXT, "event": "ADD"}]}
    guarded = store._apply_coverage_guard(results, before, _batch_meta())
    assert guarded["results"][0].get("pending_confirmation") is True, (
        "前置条件没立住：守卫应先把这条标待确认（生产日志 08:56:25 同款）"
    )
    # 模型调用 monkeypatch 成生产复现的真实返回值：价值 0.1 分。
    monkeypatch.setattr(
        store, "_judge_value_scores", lambda items: {new_id: VALUE_SCORE_PROD}
    )
    out = store._value_gate_extraction_results(guarded)
    assert NEW_TEXT in _texts(store), "守卫挂起的记忆不能被价值关卡删掉"
    assert any((item or {}).get("id") == new_id for item in out.get("results") or [])
    assert out.get("value_dropped", 0) == 0


def test_5_value_gate_must_not_drop_dated_event(tmp_path, monkeypatch):
    """程序规则（修 2）：带具体日期的事件记忆不走价值删除——
    哪怕没被守卫挂起、模型只打 0.1 分，也不能删；它该靠衰减沉底。
    当前代码失败：0.1 < 0.5，直接删掉。修 2 后通过。”
    """
    store = make_fake_store(tmp_path)
    _pref_id, _core_id, new_id, _before, _old = _seed(store)
    # 程序规则的输入：新记忆带 date: 槽位，旧喜好没有。
    new_slots = extract_core_slots(NEW_TEXT)
    old_slots = extract_core_slots(OLD_PREF)
    assert any(slot.startswith("date:") for slot in new_slots), new_slots
    assert not any(slot.startswith("date:") for slot in old_slots), old_slots
    monkeypatch.setattr(
        store, "_judge_value_scores", lambda items: {new_id: VALUE_SCORE_PROD}
    )
    results = {"results": [{"id": new_id, "memory": NEW_TEXT, "event": "ADD"}]}
    out = store._value_gate_extraction_results(results)
    assert NEW_TEXT in _texts(store), "带具体日期的事件记忆不能被价值关卡删掉"
    assert out.get("value_dropped", 0) == 0


# ---------- 去重：事件 vs 喜好不是同一件事（修 3 + 修 4 的防守） ----------


def test_2_dedupe_must_not_delete_guard_held_event(tmp_path, monkeypatch):
    """潜伏 bug（修 3）：去重通用分支不认 pending_confirmation——
    哪怕模型判成 duplicate，守卫已挂起的条也不能删（金额分支已有
    同款保护，通用分支缺了）。当前代码失败：新条被删掉。修 3 后通过。”
    """
    store = make_fake_store(tmp_path)
    _pref_id, _core_id, new_id, before, old_pref = _seed(store)
    monkeypatch.setattr(
        store,
        "_score_candidates",
        lambda text, existing, embed_cache=None: [(SIM_PROD, old_pref)],
    )
    monkeypatch.setattr(
        store,
        "_adjudicate_duplicate",
        lambda new_text, scored: {
            "verdict": "duplicate",
            "confidence": 0.95,
            "reason": "测试注入：假装模型判成重复",
            "candidate": old_pref,
            "judged": True,
        },
    )
    results = {
        "results": [
            {
                "id": new_id,
                "memory": NEW_TEXT,
                "event": "ADD",
                "pending_confirmation": True,
            }
        ]
    }
    out = store._dedupe_extraction_results(results, before, turn_id="test-turn")
    assert NEW_TEXT in _texts(store), "守卫挂起的记忆不能被去重删掉"
    assert any((item or {}).get("id") == new_id for item in out.get("results") or [])


def test_3_adjudicate_event_vs_preference_is_new(tmp_path, monkeypatch):
    """去重裁决本身是对的（生产复现：is_duplicate=false / 0.95）——
    事件 vs 喜好不是同一件事，新条留着。这个测试在当前代码上就通过，
    修完也不能跑偏；同时锁定槽位层面的事实：新记忆带 date: 槽位、
    旧喜好没有，且 _slot_values_conflict 不拦（修 4 的程序规则输入）。"""
    store = make_fake_store(tmp_path)
    _pref_id, _core_id, new_id, before, old_pref = _seed(store)
    assert _slot_values_conflict(NEW_TEXT, OLD_PREF) is False
    assert any(s.startswith("date:") for s in extract_core_slots(NEW_TEXT))
    assert not any(s.startswith("date:") for s in extract_core_slots(OLD_PREF))
    monkeypatch.setattr(
        store,
        "_score_candidates",
        lambda text, existing, embed_cache=None: [(SIM_PROD, old_pref)],
    )
    monkeypatch.setattr(
        store,
        "_judge_duplicate_verdict",
        lambda new_text, candidates: {
            "is_duplicate": False,
            "confidence": 0.95,
            "reason": "候选是喜好，新记忆是具体时间地点的一次经历，不是同一事实。",
            "conflict": False,
        },
    )
    verdict = store._adjudicate_duplicate(NEW_TEXT, [(SIM_PROD, old_pref)])
    assert verdict["verdict"] == "new"
    assert verdict["confidence"] == 0.95
    results = {"results": [{"id": new_id, "memory": NEW_TEXT, "event": "ADD"}]}
    out = store._dedupe_extraction_results(results, before, turn_id="test-turn")
    assert NEW_TEXT in _texts(store)
    assert any((item or {}).get("id") == new_id for item in out.get("results") or [])


# ---------- 真重复照常删（防修过头） ----------


def test_4_exact_duplicate_still_dropped(tmp_path):
    """完全相同的喜好句再说一遍，照常去重（只留旧条）。当前代码通过，
    修完也必须通过。
    """
    store = make_fake_store(tmp_path)
    _pref_id, _core_id, new_id, before, _old = _seed(store)
    dup_id = store._memory._store(OLD_PREF, metadata=_batch_meta())
    results = {"results": [{"id": dup_id, "memory": OLD_PREF, "event": "ADD"}]}
    out = store._dedupe_extraction_results(results, before, turn_id="test-turn")
    assert _texts(store).count(OLD_PREF) == 1
    assert out.get("dedup_dropped", 0) == 1


def test_6_paraphrase_duplicate_still_dropped(tmp_path, monkeypatch):
    """换个说法的真重复（模型判 duplicate、把握够），非挂起条照常删。
    当前代码通过，修完也必须通过。
    """
    store = make_fake_store(tmp_path)
    _pref_id, _core_id, new_id, before, old_pref = _seed(store)
    paraphrase = "BOOS最喜欢吃的食物是牛肉面。"
    para_id = store._memory._store(paraphrase, metadata=_batch_meta())
    monkeypatch.setattr(
        store,
        "_score_candidates",
        lambda text, existing, embed_cache=None: [(0.88, old_pref)],
    )
    monkeypatch.setattr(
        store,
        "_adjudicate_duplicate",
        lambda new_text, scored: {
            "verdict": "duplicate",
            "confidence": 0.9,
            "reason": "测试注入：同一喜好的换说法",
            "candidate": old_pref,
            "judged": True,
        },
    )
    results = {"results": [{"id": para_id, "memory": paraphrase, "event": "ADD"}]}
    out = store._dedupe_extraction_results(results, before, turn_id="test-turn")
    assert paraphrase not in _texts(store)
    assert out.get("dedup_dropped", 0) == 1
