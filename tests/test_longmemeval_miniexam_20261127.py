# tests/test_longmemeval_miniexam_20261127.py（整包验收：公开考题小考，2026-11-27）
#
# 出处与口径（BOOS 定）：Supermemory 本身不接、不用，只借它用的
# 公开基准 LongMemEval 当考题思路。本文件是「借题型自出的离线
# 小考」，不是官方 LongMemEval 数据集、不产出官方分数。
#
# 只考机制/检索层：SQLiteStore 的落账、来源标记、顶替
# （correct_memory）、时间字段（event_date）、跨会话（turn_id）
# 与检索 top_k=5（search 默认值）。不考模型答题能力——问法与
# 事实共享话题词（LongMemEval 原题亦然），考的是「种下去的
# 事实能不能按机制检回来、旧事实会不会被顶掉、没记过的事实
# 会不会被伪造命中」。
#
# 题型对照 LongMemEval 六类＋abstention，每类 2 题、共 14 题：
#   SSU single-session-user       单轮用户事实回忆
#   SSA single-session-assistant  助理说过的事实（来源标记）
#   SSP single-session-preference 偏好应用
#   KU  knowledge-update          新事实顶替旧事实
#   TR  temporal-reasoning        时间先后
#   MS  multi-session             跨会话合成
#   ABS abstention                没记过的事实不许伪造命中
#
# 离线路径（先读懂再写，不假设向量检索可用）：fixture 把
# _embed_text 打桩为 None → 语义路无向量可用，检索实际走
# 原样匹配＋BM25（中文两字窗口分词）＋图谱＋账本兜底；
# _ask_semantic_judge 打桩为 None；correct_memory 落账需
# _judge_degraded=False（照 G1 文件作用域打桩法，不污染全套）。
#
# SSA 来源标记实测结论（写题前探针核实）：add_memory_text
# 支持 source="assistant"，落账后 memories.source 列与
# metadata["source"] 都原样保留 "assistant"，检索结果里
# 可读回——本考据此真考，不降级跳过。
#
# TR 口径注脚：search 本身不按时间排序；「能分先后」在机制层
# 落到结果带回的 event_date 字段——两条都检回且日期字段与
# 种账一致、可据此排出先后，即机制层通过；字段缺失或错位
# 即判不过，不迁就改题。
#
# ABS 口径注脚：search 会用账本兜底把低分无关行填满 top-5，
# 机制层 abstention 只能验「没有任何相关行被命中」——即
# 结果文本里不出现被问的实体词（没种过的事实无行可命中，
# 更不该有行冒充答案）。「拿着无关行答非所问」是模型层的
# 事，不在本考范围，本考也不冒充测了它。
#
# 跑法：pytest 本文件逐题出过/不过；也可直接 python 本文件。

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.sqlite_store import SQLiteStore  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    monkeypatch.setattr(
        SQLiteStore, "_ask_semantic_judge",
        lambda self, text, candidates, context=None: None,
    )
    # 纠正落账用例需要（G1 同款文件作用域打桩）：判官降级时
    # 程序层纠正会被挂成待问，correct_memory 落不了账。
    monkeypatch.setattr(SQLiteStore, "_judge_degraded", lambda self: False)
    s = SQLiteStore(memory_dir=str(tmp_path / "memory"))
    s.organizing_router = lambda: None
    return s


def _add(store, text, **kwargs):
    saved = store.add_memory_text(text, **kwargs)
    assert saved.get("event") == "ADD", saved
    return saved["id"]


def _top5_ids(store, query):
    return [r["id"] for r in store.search(query)]


def _top5(store, query):
    return store.search(query)


# ---------- SSU：单轮用户事实回忆 ----------

def test_ssu1_lodging_fact_recalled_by_paraphrase(store):
    target = _add(store, "用户在盐湖城出差时住在朋友李雷家里", source="user")
    _add(store, "用户哥哥在西安开了一家面馆", source="user")
    _add(store, "用户每周六上午去楼下公园跑步", source="user")
    _add(store, "用户妈妈的生日是腊月初八", source="user")
    assert target in _top5_ids(store, "用户出差住在谁家")


def test_ssu2_relative_job_fact_recalled(store):
    target = _add(store, "用户表姐在深圳一家互联网公司做设计师", source="user")
    _add(store, "用户表哥在老家县城当中学老师", source="user")
    _add(store, "用户小姨退休前在邮局上班", source="user")
    _add(store, "用户爸爸喜欢下象棋", source="user")
    assert target in _top5_ids(store, "用户表姐在哪个城市工作")


# ---------- SSA：助理说过的事实（来源标记） ----------

def test_ssa1_assistant_fact_carries_assistant_source(store):
    target = _add(store, "助理上次告诉用户：小区的快递柜在东门旁边",
                  source="assistant")
    _add(store, "用户家小区门口有一家便利店", source="user")
    _add(store, "用户住的小区一共有十二栋楼", source="user")
    hits = [r for r in _top5(store, "快递柜在小区哪个门")
            if r["id"] == target]
    assert hits, "助理说过的事实没检回来"
    assert hits[0]["metadata"].get("source") == "assistant"


def test_ssa2_assistant_reminder_recalled_with_source(store):
    target = _add(store, "助理提醒过用户：车子的保险到今年十二月底到期",
                  source="assistant")
    _add(store, "用户的车是三年前买的白色轿车", source="user")
    _add(store, "用户平时开车去郊区钓鱼", source="user")
    hits = [r for r in _top5(store, "车险什么时候到期")
            if r["id"] == target]
    assert hits, "助理提醒过的事实没检回来"
    assert hits[0]["metadata"].get("source") == "assistant"


# ---------- SSP：偏好应用 ----------

def test_ssp1_tea_preference_recalled_for_scenario(store):
    target = _add(store, "用户喝茶只喝白茶，不喝绿茶", source="user")
    _add(store, "用户喝咖啡喜欢加一份糖", source="user")
    _add(store, "用户早上习惯喝一杯温水", source="user")
    _add(store, "用户办公室里有一套茶具", source="user")
    assert target in _top5_ids(store, "给用户泡茶选什么茶好")


def test_ssp2_noodle_preference_recalled_for_scenario(store):
    target = _add(store, "用户吃面条喜欢放醋，而且不吃香菜", source="user")
    _add(store, "用户吃饺子喜欢蘸辣椒油", source="user")
    _add(store, "用户晚饭一般吃得比较清淡", source="user")
    _add(store, "用户家楼下有一家兰州拉面馆", source="user")
    assert target in _top5_ids(store, "给用户煮面条要不要放香菜和醋")


# ---------- KU：新事实顶替旧事实 ----------

def test_ku1_investment_amount_updated(store):
    old_id = _add(store, "用户每天定投八块钱买沪深三百指数基金",
                  source="user")
    _add(store, "用户工资卡开在招商银行", source="user")
    receipt = store.correct_memory(
        "用户每天定投八块钱买沪深三百指数基金",
        "用户每天定投十块钱买沪深三百指数基金",
    )
    assert receipt["status"] == "updated", receipt
    hits = _top5(store, "用户定投多少钱买沪深三百")
    hit_ids = [r["id"] for r in hits]
    assert receipt["memory_id"] in hit_ids, "新事实没进 top-5"
    assert old_id not in hit_ids, "旧事实还以现行事实出现"
    assert all("八块钱" not in r["text"] for r in hits), \
        "旧说法还混在检索结果里"


def test_ku2_residence_city_updated(store):
    old_id = _add(store, "用户这两年住在北京朝阳区", source="user")
    _add(store, "用户老家在河南洛阳", source="user")
    receipt = store.correct_memory(
        "用户这两年住在北京朝阳区",
        "用户这两年住在上海浦东新区",
    )
    assert receipt["status"] == "updated", receipt
    hits = _top5(store, "用户现在住在哪个城市")
    hit_ids = [r["id"] for r in hits]
    assert receipt["memory_id"] in hit_ids, "新事实没进 top-5"
    assert old_id not in hit_ids, "旧事实还以现行事实出现"
    assert all("朝阳区" not in r["text"] for r in hits), \
        "旧说法还混在检索结果里"


# ---------- TR：时间先后 ----------

def test_tr1_trip_departure_before_return(store):
    go = _add(store, "用户十一月二十号坐高铁去杭州参加行业会议",
              source="user", event_date="2026-11-20")
    back = _add(store, "用户十一月二十五号从杭州坐高铁回广州",
                source="user", event_date="2026-11-25")
    _add(store, "用户下个月要去成都参加朋友的婚礼", source="user")
    hits = {r["id"]: r for r in _top5(store, "用户去杭州开会哪天去哪天回")}
    assert go in hits and back in hits, "两条带日期的事实没都检回来"
    d_go = hits[go]["metadata"].get("event_date")
    d_back = hits[back]["metadata"].get("event_date")
    assert d_go == "2026-11-20" and d_back == "2026-11-25", \
        f"时间字段没原样带回：{d_go} / {d_back}"
    assert d_go < d_back, "按时间字段分不出先后"


def test_tr2_spring_events_ordered(store):
    tomb = _add(store, "用户四月初清明节回老家给爷爷扫墓",
                source="user", event_date="2026-04-04")
    trip = _add(store, "用户五月一号假期去大理旅行了五天",
                source="user", event_date="2026-05-01")
    _add(store, "用户中秋节打算去桂林看山水", source="user")
    hits = {r["id"]: r for r in _top5(store, "用户清明节和五一假期分别有什么安排")}
    assert tomb in hits and trip in hits, "两条带日期的事实没都检回来"
    d_tomb = hits[tomb]["metadata"].get("event_date")
    d_trip = hits[trip]["metadata"].get("event_date")
    assert d_tomb == "2026-04-04" and d_trip == "2026-05-01", \
        f"时间字段没原样带回：{d_tomb} / {d_trip}"
    assert d_tomb < d_trip, "按时间字段分不出先后"


# ---------- MS：跨会话合成 ----------

def test_ms1_two_sessions_trip_halves_both_recalled(store):
    first = _add(store, "用户这次去日本旅行打算先到东京看展",
                 source="user", turn_id="miniexam-sess-a")
    second = _add(store, "用户这次去日本旅行回程从大阪飞回来",
                  source="user", turn_id="miniexam-sess-b")
    _add(store, "用户去年去云南旅行住在大理古城", source="user")
    hits = {r["id"]: r for r in _top5(store, "用户日本旅行先到哪回程从哪走")}
    assert first in hits and second in hits, "跨会话的两半事实没都检回来"
    turns = {hits[first]["metadata"].get("turn_id"),
             hits[second]["metadata"].get("turn_id")}
    assert turns == {"miniexam-sess-a", "miniexam-sess-b"}, \
        "两条事实的会话标记对不上，不是跨会话种的"


def test_ms2_two_sessions_renovation_choices_both_recalled(store):
    kitchen = _add(store, "用户新家装修厨房选了石英石台面",
                   source="user", turn_id="miniexam-sess-c")
    bath = _add(store, "用户新家装修卫生间选了智能马桶",
                source="user", turn_id="miniexam-sess-d")
    _add(store, "用户新家客厅刷了浅灰色的墙漆", source="user")
    hits = {r["id"]: r for r in _top5(store, "用户新家装修厨房和卫生间各选了什么")}
    assert kitchen in hits and bath in hits, "跨会话的两半事实没都检回来"
    turns = {hits[kitchen]["metadata"].get("turn_id"),
             hits[bath]["metadata"].get("turn_id")}
    assert turns == {"miniexam-sess-c", "miniexam-sess-d"}, \
        "两条事实的会话标记对不上，不是跨会话种的"


# ---------- ABS：没记过的事实不许伪造命中 ----------

def test_abs1_never_stored_cat_not_fabricated(store):
    _add(store, "用户在盐湖城出差时住在朋友李雷家里", source="user")
    _add(store, "用户每周六上午去楼下公园跑步", source="user")
    _add(store, "用户计划下周去成都参加婚礼", source="user")
    hits = _top5(store, "用户养的猫叫什么名字")
    # 机制层口径（见文件头注脚）：没种过的实体不该有任何行命中。
    assert all("猫" not in r["text"] for r in hits), \
        "没记过的猫被某行冒充命中了"


def test_abs2_never_stored_team_not_fabricated(store):
    _add(store, "用户喜欢在周末去郊区钓鱼", source="user")
    _add(store, "用户下班后经常去打羽毛球", source="user")
    _add(store, "用户小时候练过几年乒乓球", source="user")
    hits = _top5(store, "用户最喜欢的足球队是哪支")
    assert all("足球" not in r["text"] for r in hits), \
        "没记过的足球队被某行冒充命中了"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
