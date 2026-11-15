# tests/test_memory_decay.py（记忆整理第二步：连续衰减）
# 测八件事：权重按半衰期掉、档位按来源定、3 次不同天晋升、检索资格
# 不看权重（衰减再狠也不让记忆搜不到）、检索强化只强化返回的那几条、
# 月扫只打标（30 天标沉底、180 天标待删，原话和文本都不动、不压摘要、
# 不删）、沉底的被用上回到原话态、删要 BOOS 点头且先写备份、核心钉住不动。
# 全部离线（FakeMemory），不碰真实 mem0、不联网、不产生费用。

import json
from datetime import datetime, timedelta, timezone

import pytest

from conftest import make_fake_store
from memory.mem0_store import (
    CONFIDENCE_INFERRED,
    CONFIDENCE_USER,
    DECAY_KIND_EPISODIC,
    DECAY_KIND_STABLE,
    DECAY_STATE_DORMANT,
    DECAY_STATE_VERBATIM,
    SOURCE_INFERRED,
    SOURCE_USER,
    _provenance,
    _slot_values_conflict,
    classify_decay_kind,
    decay_should_delete,
    decay_should_go_dormant,
    decay_weight,
    promotion_reached,
)


def _utc_now():
    return datetime.now(timezone.utc)


def _add_inferred(store, text):
    """存一条模型提取的记忆（episodic），返回 id。"""
    result = store._add_text_with_metadata(
        text, _provenance(SOURCE_INFERRED, CONFIDENCE_INFERRED, False)
    )
    return (result.get("results") or [{}])[0].get("id")


def _age(store, memory_id, days):
    """把一条记忆的 last_used_at 拨到几天前（模拟“很久没用”）。"""
    raw = store._raw_metadata_for(memory_id)
    assert raw is not None
    old = _utc_now() - timedelta(days=days)
    raw["last_used_at"] = old.isoformat(timespec="microseconds").replace("+00:00", "Z")
    store._memory.update(memory_id=memory_id, metadata=raw)


def _meta_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["metadata"]
    raise AssertionError(f"库里没有 {memory_id}")


def _text_of(store, memory_id):
    for item in store.get_all(include_sensitive=True):
        if item["id"] == memory_id:
            return item["text"]
    raise AssertionError(f"库里没有 {memory_id}")


LONG_TRIP = (
    "BOOS 上周去了一趟盐湖城出差，住在朋友家里，每天开车去公司开会，"
    "晚上回来还要处理国内的事情，第二天早上还要早起赶飞机去洛杉矶见客户谈合作。"
)


# ---------- 纯函数：权重、档位、晋升、压摘要、待删 ----------


def test_decay_weight_half_lives_and_floor():
    """权重按半衰期掉：一次性经历 14 天掉一半，稳定事实 180 天掉一半；
    掉再久也不低于 0.10；没时间的老记忆按 1.0 算（宁可不沉，不错沉）。"""
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
    assert decay_weight(DECAY_KIND_EPISODIC, "2026-10-30T12:00:00.000000Z", now) == pytest.approx(0.5)
    assert decay_weight(DECAY_KIND_STABLE, "2026-05-17T12:00:00.000000Z", now) == pytest.approx(0.5)
    assert decay_weight(DECAY_KIND_EPISODIC, "2026-01-01T00:00:00.000000Z", now) == pytest.approx(0.10)
    assert decay_weight(DECAY_KIND_EPISODIC, None, now) == 1.0
    assert decay_weight(DECAY_KIND_EPISODIC, "不是时间", now) == 1.0
    # 核心记忆钉住不动：权重恒为 1.0，不是“慢”。
    assert decay_weight(DECAY_KIND_EPISODIC, "2020-01-01T00:00:00.000000Z", now, is_core=True) == 1.0


def test_classify_decay_kind():
    """档位按来源定，不猜关键词：核心/规则/BOOS 亲口说的记 stable，
    模型提取的记 episodic。"""
    assert classify_decay_kind(SOURCE_USER, True, False) == DECAY_KIND_STABLE
    assert classify_decay_kind(SOURCE_USER, False, False) == DECAY_KIND_STABLE
    assert classify_decay_kind(SOURCE_INFERRED, False, True) == DECAY_KIND_STABLE
    assert classify_decay_kind(SOURCE_INFERRED, False, False) == DECAY_KIND_EPISODIC


def test_promotion_reached():
    """晋升：不同天出现够 3 次才晋升，2 次不够；没数过按没够算。"""
    assert promotion_reached(3) is True
    assert promotion_reached(2) is False
    assert promotion_reached(None) is False


def test_should_go_dormant_and_delete_due_pure():
    """标沉底：30 天没人找、权重掉到 0.30 下面才标；核心不标；
    标待删：满 180 天没人找才标，只打标、不管删。"""
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
    old_40 = "2026-10-04T12:00:00.000000Z"
    old_10 = "2026-11-03T12:00:00.000000Z"
    old_200 = "2026-04-27T12:00:00.000000Z"
    assert decay_should_go_dormant(DECAY_KIND_EPISODIC, old_40, now) is True
    assert decay_should_go_dormant(DECAY_KIND_EPISODIC, old_10, now) is False
    assert decay_should_go_dormant(DECAY_KIND_STABLE, old_40, now) is False
    assert decay_should_go_dormant(DECAY_KIND_EPISODIC, old_40, now, is_core=True) is False
    assert decay_should_go_dormant(DECAY_KIND_EPISODIC, None, now) is False
    assert decay_should_delete(old_200, now) is True
    assert decay_should_delete(old_40, now) is False
    assert decay_should_delete(old_200, now, is_core=True) is False
    assert decay_should_delete(None, now) is False


def test_sweep_never_deletes_or_rewrites_text(tmp_path):
    """月扫只打标：200 天没用的记忆，扫完还在库里、原话一字没动，
    只是多了沉底和待删两个标记。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_inferred(store, LONG_TRIP)
    _age(store, memory_id, 200)
    ledger = store.sweep_monthly()
    assert [entry["id"] for entry in ledger["dormant"]] == [memory_id]
    assert [entry["id"] for entry in ledger["delete_due"]] == [memory_id]
    assert len(store.get_all(include_sensitive=True)) == 1
    assert _text_of(store, memory_id) == LONG_TRIP
    meta = _meta_of(store, memory_id)
    assert meta["decay_state"] == DECAY_STATE_DORMANT
    assert meta["decay_delete_due"] is True


def test_slot_values_conflict_regressions():
    """57 号回归（去重分支同款断言里、不依赖去重修的那几条）：冰美式
    同义改写照常合并、十二身份改写不冲突、数字不同是冲突。
    注意：“冒菜 vs 拉面是冲突”那条是去重分支（fix/dedup-object-conflict）
    的修法，衰减分支从 main 切出来、刻意没混去重的改动，所以这里不
    断言它——它在试合并里验（见试合并报告：两分支合起来跑全套件，
    去重分支自带的 57 号断言照常过）。"""
    assert _slot_values_conflict("喜欢喝冰美式", "喜欢冰美式") is False
    assert _slot_values_conflict("助手叫十二", "你的名字是十二") is False
    assert _slot_values_conflict("BOOS 喜欢吃拉面", "BOOS 不喜欢吃拉面") is True
    assert _slot_values_conflict("纳指收盘 25,466.19", "纳指收盘 25466.19") is False


# ---------- 写时字段 ----------


def test_provenance_writes_decay_fields_user(tmp_path):
    """BOOS 亲口让记住的：kind stable、importance high、last_used 已记、
    出现天数从 1 数起、原话态。"""
    store = make_fake_store(tmp_path=tmp_path)
    result = store.add_memory_text("BOOS 爱吃冒菜", source="user")
    assert result.get("duplicate") is not True
    memory_id = (result.get("results") or [{}])[0].get("id")
    meta = _meta_of(store, memory_id)
    assert meta["decay_kind"] == DECAY_KIND_STABLE
    assert meta["importance"] == "high"
    assert meta["last_used_at"]
    assert meta["decay_seen_days"] == 1
    assert meta["decay_last_seen_day"] == _utc_now().date().isoformat()
    assert meta["decay_state"] == DECAY_STATE_VERBATIM


def test_provenance_writes_decay_fields_inferred(tmp_path):
    """模型提取的：kind episodic、importance normal，出现天数同样从 1 数起。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_inferred(store, "BOOS 喜欢看纪录片。")
    meta = _meta_of(store, memory_id)
    assert meta["decay_kind"] == DECAY_KIND_EPISODIC
    assert meta["importance"] == "normal"
    assert meta["decay_seen_days"] == 1
    assert meta["decay_state"] == DECAY_STATE_VERBATIM


def test_occurrence_same_day_does_not_count(tmp_path):
    """同一天重复提不加数：一次聊天里说 3 遍也不算晋升。"""
    store = make_fake_store(tmp_path=tmp_path)
    store.add_memory_text("BOOS 爱吃冒菜", source="user")
    assert store._memory.update_calls == []
    second = store.add_memory_text("BOOS 爱吃冒菜", source="user")
    assert second.get("duplicate") is True
    memory_id = (second.get("results") or [{}])[0].get("id")
    assert _meta_of(store, memory_id)["decay_seen_days"] == 1


def test_occurrence_promotes_after_three_distinct_days(tmp_path):
    """不同天出现 3 次就晋升：一次性经历变稳定事实，衰减变慢；
    同一天再提也不多算。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_inferred(store, "BOOS 去盐湖城出差了。")
    assert _meta_of(store, memory_id)["decay_kind"] == DECAY_KIND_EPISODIC
    day2 = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
    day3 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert store._record_decay_occurrence(memory_id, now=day2) is True
    assert _meta_of(store, memory_id)["decay_seen_days"] == 2
    assert _meta_of(store, memory_id)["decay_kind"] == DECAY_KIND_EPISODIC
    assert store._record_decay_occurrence(memory_id, now=day3) is True
    meta = _meta_of(store, memory_id)
    assert meta["decay_seen_days"] == 3
    assert meta["decay_kind"] == DECAY_KIND_STABLE
    assert store._record_decay_occurrence(memory_id, now=day3) is True
    assert _meta_of(store, memory_id)["decay_seen_days"] == 3


# ---------- 检索：资格不看权重，强化只强化返回的 ----------


def test_search_default_does_not_write(tmp_path):
    """默认检索不写回：内部检索（纠错找候选等）绝不能把“找候选”
    也算成“用上了”——纠错测试断言的 update_calls == [] 在这里守住。"""
    store = make_fake_store(tmp_path=tmp_path)
    store.add_memory_text("BOOS 爱吃冒菜", source="user")
    store._memory.update_calls = []
    results = store.search("BOOS 爱吃冒菜")
    assert len(results) == 1
    assert store._memory.update_calls == []


def test_search_stale_memory_still_found(tmp_path):
    """坑 1：权重掉到底（0.10）也照常搜得到——资格只看原始语义分，
    权重只管排位。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_inferred(store, "BOOS 爱吃冒菜。")
    _age(store, memory_id, 60)
    results = store.search("BOOS 爱吃冒菜")
    assert [item["id"] for item in results] == [memory_id]
    assert results[0]["decay_weight"] == pytest.approx(0.10)


def test_search_reinforce_refreshes_last_used(tmp_path):
    """真正用上才强化：reinforce=True 时返回的那条刷新 last_used_at，
    权重回到 1.0 附近；没返回的不动。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_inferred(store, "BOOS 爱吃冒菜。")
    _age(store, memory_id, 10)
    before = _meta_of(store, memory_id)["last_used_at"]
    results = store.search("BOOS 爱吃冒菜", reinforce=True)
    assert len(results) == 1
    after = _meta_of(store, memory_id)["last_used_at"]
    assert after > before
    assert decay_weight(
        _meta_of(store, memory_id)["decay_kind"], after
    ) == pytest.approx(1.0, abs=1e-3)


# ---------- 月扫：只打标，不改文本、不压摘要、不删 ----------


def test_sweep_marks_dormant_episodic_only(tmp_path):
    """月扫只标该标的：40 天没用的一次性经历标沉底（长短都标，只打标、
    文本不动）；核心和稳定事实不动；标完库里条数不变（月扫不删东西、
    不改原话）。短记忆以前不压摘要是压摘要才有的例外，压摘要没了，
    打标不看长短。"""
    store = make_fake_store(tmp_path=tmp_path)
    episodic_id = _add_inferred(store, LONG_TRIP)
    _age(store, episodic_id, 40)
    core_id = (store.add_memory_text("BOOS 爱吃冒菜", source="user").get("results") or [{}])[0].get("id")
    _age(store, core_id, 40)
    stable_id = _add_inferred(store, "BOOS 去盐湖城出差了。")
    for day in (
        datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc),
    ):
        store._record_decay_occurrence(stable_id, now=day)
    _age(store, stable_id, 40)
    short_id = _add_inferred(store, "BOOS 喜欢看纪录片。")
    _age(store, short_id, 40)
    ledger = store.sweep_monthly()
    assert ledger["checked"] == 4
    assert [entry["id"] for entry in ledger["dormant"]] == [episodic_id, short_id]
    assert ledger["delete_due"] == []
    assert ledger["skipped_core"] == 1
    assert ledger["errors"] == []
    assert _meta_of(store, episodic_id)["decay_state"] == DECAY_STATE_DORMANT
    assert _meta_of(store, core_id)["decay_state"] == DECAY_STATE_VERBATIM
    assert _meta_of(store, stable_id)["decay_state"] == DECAY_STATE_VERBATIM
    assert _meta_of(store, short_id)["decay_state"] == DECAY_STATE_DORMANT
    assert len(store.get_all(include_sensitive=True)) == 4
    # 原话还在向量文本里，一字没动；没有摘要文件、不用恢复。
    assert _text_of(store, episodic_id) == LONG_TRIP
    assert _text_of(store, short_id) == "BOOS 喜欢看纪录片。"


def test_dormant_memory_searchable_and_woken_on_use(tmp_path):
    """沉底的也要能被搜到；被用上就回到原话态、重新强化。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_inferred(store, LONG_TRIP)
    _age(store, memory_id, 40)
    store.sweep_monthly()
    assert _meta_of(store, memory_id)["decay_state"] == DECAY_STATE_DORMANT
    assert _text_of(store, memory_id) == LONG_TRIP
    results = store.search("盐湖城出差", reinforce=True)
    assert [item["id"] for item in results] == [memory_id]
    assert _text_of(store, memory_id) == LONG_TRIP
    assert _meta_of(store, memory_id)["decay_state"] == DECAY_STATE_VERBATIM
    assert _meta_of(store, memory_id)["decay_delete_due"] is False


# ---------- 待删与真删：半年、备份、BOOS 点头 ----------


def test_sweep_marks_delete_due_at_180_days(tmp_path):
    """满 180 天没人找：直接标待删（只打标、不删），沉底和待删可以在
    同一次月扫里一起标，不用等下次；40 天的只标沉底、不标待删。"""
    store = make_fake_store(tmp_path=tmp_path)
    old_id = _add_inferred(store, LONG_TRIP)
    _age(store, old_id, 40)
    store.sweep_monthly()
    assert _meta_of(store, old_id)["decay_state"] == DECAY_STATE_DORMANT
    assert _meta_of(store, old_id)["decay_delete_due"] is False
    _age(store, old_id, 200)
    recent_id = _add_inferred(store, "BOOS 昨天去了一趟广州出差，见了三个客户，谈了两个合作意向，晚上还和朋友吃了顿饭，第二天早上还要早起赶飞机去上海见投资人谈下一轮融资。")
    _age(store, recent_id, 40)
    ledger = store.sweep_monthly()
    assert {entry["id"] for entry in ledger["delete_due"]} == {old_id}
    assert _meta_of(store, old_id)["decay_delete_due"] is True
    assert _meta_of(store, old_id)["decay_state"] == DECAY_STATE_DORMANT
    assert [entry["id"] for entry in ledger["dormant"]] == [recent_id]
    assert _meta_of(store, recent_id)["decay_delete_due"] is False
    # 月扫只打标，两个都在库里，一个也没删。
    assert len(store.get_all(include_sensitive=True)) == 2


def test_delete_due_requires_approval(tmp_path):
    """删记忆必须 BOOS 同意：没点头就抛 PermissionError，一个也不删。"""
    store = make_fake_store(tmp_path=tmp_path)
    memory_id = _add_inferred(store, LONG_TRIP)
    _age(store, memory_id, 200)
    store.sweep_monthly()
    assert _meta_of(store, memory_id)["decay_delete_due"] is True
    with pytest.raises(PermissionError):
        store.delete_due_memories(approve=False)
    assert len(store.get_all(include_sensitive=True)) == 1


def test_delete_due_deletes_with_backup(tmp_path):
    """满半年没人找的记忆：点头后才删，删前先写备份（原话在里面），
    删完库里没了、备份还在；100 天没到的留着。"""
    store = make_fake_store(tmp_path=tmp_path)
    old_id = _add_inferred(store, LONG_TRIP)
    _age(store, old_id, 200)
    store.sweep_monthly()
    recent_id = _add_inferred(store, "BOOS 上个月去了一趟深圳出差，见了两个客户，谈了一个合作项目，晚上还和朋友吃了顿饭，第二天早上还要早起赶飞机去北京见投资人谈下一轮融资。")
    _age(store, recent_id, 40)
    store.sweep_monthly()
    _age(store, recent_id, 100)
    backup_path = tmp_path / "backup.json"
    ledger = store.delete_due_memories(backup_path=backup_path, approve=True)
    assert ledger["deleted"] == [old_id]
    assert ledger["backup_path"] == str(backup_path)
    assert backup_path.exists()
    backup = json.loads(backup_path.read_text(encoding="utf-8"))
    assert backup[0]["id"] == old_id
    assert backup[0]["full_text"] == LONG_TRIP
    remaining = [item["id"] for item in store.get_all(include_sensitive=True)]
    assert old_id not in remaining
    assert recent_id in remaining


def test_delete_due_never_touches_core(tmp_path):
    """核心记忆满 200 天没人找也不会被标沉底、不会被标待删、不会被删。"""
    store = make_fake_store(tmp_path=tmp_path)
    core_id = (store.add_memory_text("BOOS 爱吃冒菜", source="user").get("results") or [{}])[0].get("id")
    _age(store, core_id, 200)
    ledger = store.sweep_monthly()
    assert ledger["delete_due"] == []
    assert ledger["dormant"] == []
    result = store.delete_due_memories(approve=True)
    assert result["deleted"] == []
    assert len(store.get_all(include_sensitive=True)) == 1


# ---------- 老记忆：没有衰减字段时派生默认值 ----------


def test_normalize_old_memory_defaults(tmp_path):
    """没衰减字段的老记忆：已确认的派生 stable/high；提取的派生
    episodic/normal；last_used 没有就用 created_at。"""
    from memory.mem0_store import _normalize_metadata

    confirmed = _normalize_metadata(
        {
            "source": "user",
            "created_at": "2026-11-01T00:00:00.000000Z",
            "user_confirmed": True,
        }
    )
    assert confirmed["decay_kind"] == DECAY_KIND_STABLE
    assert confirmed["importance"] == "high"
    assert confirmed["last_used_at"] == "2026-11-01T00:00:00.000000Z"
    assert confirmed["decay_state"] == DECAY_STATE_VERBATIM
    inferred = _normalize_metadata(
        {
            "source": "inferred",
            "created_at": "2026-11-01T00:00:00.000000Z",
            "confidence": 0.7,
        }
    )
    assert inferred["decay_kind"] == DECAY_KIND_EPISODIC
    assert inferred["importance"] == "normal"
    assert inferred["last_used_at"] == "2026-11-01T00:00:00.000000Z"
