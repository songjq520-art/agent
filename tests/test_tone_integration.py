# tests/test_tone_integration.py（补丁 3、补丁 4 接进存记忆的两条路）
# 离线测试：FakeMemory + tmp_path，不联网、不调真实大模型。
# 覆盖：
#   - _cn_numerals_to_digits：中文数字转阿拉伯数字（只转数字短语）
#   - add_memory_text：玩笑话存时直接打标；认真的排队等大模型判
#   - add_turn 提取：玩笑话打标；数字跟原话对得上不标、对不上标
#     待复核 + grounding_flag（中文数字“五块”对“5 块”算对上）
#   - _decay_touch：打标的记忆不计出现次数、不晋升（认真记忆对照组会晋升）
#   - void_joke_memory：作废 + 已晋升的打回一次性
#   - process_tone_queue：假路由判完打标；路由没回的留队里

from datetime import datetime, timedelta, timezone

import pytest

from memory.decay import _parse_seen_list
from memory.mem0_store import _cn_numerals_to_digits
from memory.provenance import _normalize_metadata
from tests.conftest import make_fake_store

UTC = timezone.utc


def _messages(text):
    return [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "记下了。"},
    ]


# ---------- 中文数字转阿拉伯数字 ----------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我每天定投五块。", "我每天定投5块。"),
        ("定投一百五十块。", "定投150块。"),
        ("二十块。", "20块。"),
        ("睡了两小时。", "睡了2小时。"),
        ("一起吃饭。", "一起吃饭。"),  # “一”后面不是量词，不转
        ("一般情况。", "一般情况。"),  # 同上
        ("十分喜欢。", "十分喜欢。"),  # “十分”不是数字，不转
        ("一万块。", "1万块。"),  # “一”转了，“万”留作单位（解析器不认万）
        ("", ""),
    ],
)
def test_cn_numerals_to_digits(text, expected):
    assert _cn_numerals_to_digits(text) == expected


# ---------- add_memory_text：存时打标 / 排队 ----------


def test_add_memory_text_joke_tagged_at_write(tmp_path):
    store = make_fake_store(tmp_path)
    stored = store.add_memory_text("我喜欢喝咖啡，开玩笑的。")
    assert stored["results"]
    mem_id = stored["results"][0]["id"]
    meta = store._raw_metadata_for(mem_id)
    assert meta["tone"] == "joke"
    assert meta["tone_source"] == "local"
    assert meta["tone_evidence"] == "开玩笑"
    # 打过标的不排队
    assert store._tone_queue_read() == []


def test_add_memory_text_serious_queued(tmp_path):
    store = make_fake_store(tmp_path)
    stored = store.add_memory_text("我喜欢喝咖啡。")
    assert stored["results"]
    mem_id = stored["results"][0]["id"]
    # raw metadata 里没写 tone（缺省就是认真）；normalize 之后是 serious
    meta = _normalize_metadata(store._raw_metadata_for(mem_id))
    assert meta["tone"] == "serious"
    assert meta.get("tone_source") is None
    queued = store._tone_queue_read()
    assert len(queued) == 1
    assert queued[0]["memory_id"] == mem_id
    assert queued[0]["text"] == "我喜欢喝咖啡。"


# ---------- add_turn 提取：语气 + 对原话 ----------


def test_add_turn_explicit_joke_dropped_by_gate(tmp_path):
    # 明确带玩笑标记的句子根本进不了提取（Plan C 闸门 1 在模型之前就
    # 拦掉了）——补丁 3 的打标管的是“已经存下来”的记忆（手动记住、
    # 闸门没拦住的含蓄玩笑），两层各管各的，不冲突。
    store = make_fake_store(tmp_path)
    store.add_turn(_messages("我喜欢喝咖啡，开玩笑的。"))
    flushed = store.flush_pending()
    # 闸门把整句都拦了，没什么可提，提交返回 {}（没调提取）
    assert flushed == {}
    assert store._memory.all_texts() == []
    # 提取还是调了一次（只剩 assistant 那句当上下文），但 user 的话
    # 一句都没传过去——闸门在模型之前就拦掉了
    assert len(store._memory.add_calls) == 1
    passed_messages = store._memory.add_calls[0][0]
    assert all((m or {}).get("role") != "user" for m in passed_messages)


def test_add_turn_test_sentence_tagged(tmp_path):
    # “这是测试”闸门不拦（只拦玩笑/条件/身份句），提取出来后本地筛
    # 出 test，打标。
    store = make_fake_store(tmp_path)
    store.add_turn(_messages("我的定投是每天 5 块，这是测试。"))
    flushed = store.flush_pending()
    assert flushed["results"]
    mem_id = flushed["results"][0]["id"]
    meta = store._raw_metadata_for(mem_id)
    assert meta["tone"] == "test"
    assert meta["tone_source"] == "local"


def test_add_turn_grounding_ok_with_cn_numerals(tmp_path):
    # 原话“五块”，提取的记忆也是“五块”——转数字后对得上，不标。
    store = make_fake_store(tmp_path)
    store.add_turn(_messages("我的定投是每天五块。"))
    result = store.flush_pending()
    assert result["results"]
    assert result.get("grounding_flagged") in (None, [])
    mem_id = result["results"][0]["id"]
    meta = store._raw_metadata_for(mem_id)
    assert not meta.get("needs_review")
    assert not meta.get("grounding_flag")


def test_add_turn_grounding_flagged_on_mismatch(tmp_path, monkeypatch):
    # 提取出来的数字跟原话对不上（模型幻觉/听错）：标待复核。
    # FakeMemory 提取是原样抄句子，对不上造不出来——这里把提取返回的
    # 文本改成另一个数字，模拟“提取跟原话对不上”的那条路。
    store = make_fake_store(tmp_path)
    real_add = store._memory.add

    def _doctored_add(messages, user_id=None, infer=True, metadata=None):
        out = real_add(messages, user_id=user_id, infer=infer, metadata=metadata)
        for item in out.get("results") or []:
            if "五块" in (item.get("memory") or ""):
                item["memory"] = "我的定投是每天十块。"
        return out

    monkeypatch.setattr(store._memory, "add", _doctored_add)
    store.add_turn(_messages("我的定投是每天五块。"))
    result = store.flush_pending()
    assert result["results"]
    mem_id = result["results"][0]["id"]
    assert result.get("grounding_flagged") == [mem_id]
    meta = store._raw_metadata_for(mem_id)
    assert meta.get("needs_review") is True
    assert meta.get("grounding_flag") is True
    # 记忆本身还在（只标不删），原话没动（FakeMemory 存句子时不带句号）
    assert store._memory.all_texts() == ["我的定投是每天五块"]


# ---------- _decay_touch：打标的不计出现次数 ----------


def _extracted_store(tmp_path, text, tone=None):
    """提取路存一条记忆（source=inferred，一次性 episodic）。
    要打标的先打标。返回 (store, mem_id, day0)：day0 是记忆自己的
    创建日期（_decay_touch 的起始日从它取，不依赖今天是几号）。"""
    store = make_fake_store(tmp_path)
    store.add_turn(_messages(text))
    flushed = store.flush_pending()
    assert flushed["results"]
    mem_id = flushed["results"][0]["id"]
    if tone is not None:
        assert store._tag_memory_tone(mem_id, tone, "测试", "local")
    meta = store._raw_metadata_for(mem_id)
    day0 = datetime.strptime(
        meta["decay_last_seen_day"], "%Y-%m-%d"
    ).replace(tzinfo=UTC)
    return store, mem_id, day0


def _touch_three_days(store, mem_id, day0):
    for day in range(3):
        store._decay_touch(
            mem_id,
            now=day0 + timedelta(days=day),
            record_occurrence=True,
            turn_id="turn-%d" % day,
        )


def test_decay_touch_joke_does_not_count_occurrences(tmp_path):
    store, mem_id, day0 = _extracted_store(
        tmp_path, "我喜欢喝咖啡。", tone="joke"
    )
    assert store._raw_metadata_for(mem_id).get("decay_kind") == "episodic"
    _touch_three_days(store, mem_id, day0)
    meta = store._raw_metadata_for(mem_id)
    # 三天、三个会话都用上了，出现次数一点没涨（创建那天算的 1 次还在）
    assert meta.get("decay_seen_days") == 1
    assert meta.get("decay_seen_sessions") in ([], "[]")
    assert meta.get("decay_kind") == "episodic"  # 没晋升
    # 唤醒照常：last_used_at 刷了
    assert meta.get("last_used_at")


def test_decay_touch_serious_promotes_control(tmp_path):
    store, mem_id, day0 = _extracted_store(tmp_path, "我喜欢喝咖啡。")
    _touch_three_days(store, mem_id, day0)
    meta = store._raw_metadata_for(mem_id)
    # 创建日 + 另外两天 = 3 天，3 个会话 → 晋升
    assert meta.get("decay_seen_days") == 3
    assert len(_parse_seen_list(meta.get("decay_seen_sessions"))) == 3
    assert meta.get("decay_kind") == "stable"


# ---------- void_joke_memory ----------


def test_void_joke_memory_reverts_promotion(tmp_path):
    store, mem_id, day0 = _extracted_store(tmp_path, "我喜欢喝咖啡。")
    # 先让它晋升，再作废
    _touch_three_days(store, mem_id, day0)
    assert store._raw_metadata_for(mem_id).get("decay_kind") == "stable"
    out = store.void_joke_memory(mem_id)
    assert out["ok"] is True
    meta = store._raw_metadata_for(mem_id)
    assert meta.get("voided") is True
    assert meta.get("voided_at")
    assert meta.get("tone") == "joke"
    assert meta.get("decay_kind") == "episodic"  # 打回一次性
    # 原话还在，不删
    assert store._memory.all_texts() == ["我喜欢喝咖啡"]


def test_void_joke_memory_missing_id(tmp_path):
    store = make_fake_store(tmp_path)
    assert store.void_joke_memory("no-such-id")["ok"] is False
    assert store.void_joke_memory("")["ok"] is False


# ---------- process_tone_queue ----------


class _FakeRouter:
    """假路由：judge 按剧本回。"""

    def __init__(self, script):
        # script: {text_substring: {"text": json_str} 或 None（没回）}
        self.script = script
        self.calls = []

    def judge(self, system_prompt, user_prompt, max_tokens=300, purpose=None,
              probe_primary=False):
        self.calls.append(user_prompt)
        for needle, out in self.script.items():
            if needle in user_prompt:
                return out
        return None


def test_process_tone_queue_tags_judged(tmp_path):
    store = make_fake_store(tmp_path)
    stored = store.add_memory_text("我每天定投 5 块。")
    mem_id = stored["results"][0]["id"]
    assert len(store._tone_queue_read()) == 1
    router = _FakeRouter(
        {
            "定投": {
                "text": '{"tone": "serious", "evidence": "陈述事实"}',
            }
        }
    )
    out = store.process_tone_queue(router=router)
    assert out == {"judged": 1, "tagged": 1, "remaining": 0, "skipped": 0}
    meta = store._raw_metadata_for(mem_id)
    assert meta["tone"] == "serious"
    assert meta["tone_source"] == "model"
    assert store._tone_queue_read() == []


def test_process_tone_queue_keeps_unjudged(tmp_path):
    store = make_fake_store(tmp_path)
    stored = store.add_memory_text("我每天定投 5 块。")
    mem_id = stored["results"][0]["id"]
    router = _FakeRouter({})  # 什么都不回
    out = store.process_tone_queue(router=router)
    assert out == {"judged": 0, "tagged": 0, "remaining": 1, "skipped": 0}
    assert len(store._tone_queue_read()) == 1
    # 没判出来不猜：语气还是默认 serious（_normalize_metadata 的值），
    # 但 tone_source 没写——下次还能再判
    meta = store._raw_metadata_for(mem_id)
    assert meta.get("tone_source") is None


def test_process_tone_queue_skips_missing_and_tagged(tmp_path):
    store = make_fake_store(tmp_path)
    stored = store.add_memory_text("我每天定投 5 块。")
    mem_id = stored["results"][0]["id"]
    store._tag_memory_tone(mem_id, "joke", "开玩笑", "local")
    # 队里再塞一条“记忆已经没了”的
    store._tone_queue_append("gone-id", "已经删掉的记忆。")
    router = _FakeRouter({"定投": {"text": '{"tone": "serious", "evidence": ""}'}})
    out = store.process_tone_queue(router=router)
    assert out["skipped"] == 2  # 已打标 joke 的 + 没了的
    assert out["remaining"] == 0
    assert store._tone_queue_read() == []
    # 假路由一次都没被调到（两条都跳过了）
    assert router.calls == []
