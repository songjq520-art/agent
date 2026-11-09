# tests/test_mem0_privacy.py（mem0 隐私改进：遥测强制关 / 敏感信息三层过滤 / 直接存入去重）
# 作用：验证 memory/mem0_store.py 里 Worker A 的三项改进：
#   1. 遥测强制关（就算环境变量事先设成 True 也盖掉）；
#   2. 隐私三层过滤——存前脱敏（秘密不发给 DeepSeek）、用时拦截
#      （旧库里的敏感记忆不注入提示）、导出/迁移排除敏感内容，
#      日志里也不出现敏感原文；
#   3. 直接存入（add_memory_text）去重：同一条反复存只留一条。
# 测试全部是离线假对象测试（FakeMemory，见 tests/conftest.py）：
# 不碰真实 mem0、不联网、不产生费用。遥测的“重启”测试用子进程跑
# （全新进程 import 一次，就等于重启一次）。

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from memory.mem0_store import (
    Mem0Store,
    _contains_sensitive,
    _find_duplicate_memory,
    _meaningful_after_redaction,
    _normalize_memory_text,
    _redact_sensitive,
)
from conftest import FakeMemory, make_fake_store

ROOT = Path(__file__).resolve().parent.parent

# _SECRET_KEY / _SECRET_PASSWORD：测试用的假秘密（不是真实密钥）。
_SECRET_KEY = "sk-abcdefghij1234567890"
_SECRET_PASSWORD = "hunter2"
_SECRET_ID18 = "11010519491231002X"  # 国标里的示例身份证号，校验位正确
_SECRET_CARD = "4111111111111111"  # 经典测试卡号，过 Luhn 校验


# ---------- 1. 遥测强制关 ----------


def test_telemetry_forced_off_when_env_preset_true(tmp_path):
    """环境变量事先设成 True，import 之后也必须是 False（强制盖掉）。

    全新子进程里跑：子进程启动时 MEM0_TELEMETRY=True，import
    memory.mem0_store 后断言它是 False。子进程就是一次干净的启动，
    不受本测试进程里已有的 import 影响。"""
    code = (
        "import os, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import memory.mem0_store\n"
        "assert os.environ.get('MEM0_TELEMETRY') == 'False', "
        "repr(os.environ.get('MEM0_TELEMETRY'))\n"
    )
    env = dict(os.environ)
    env["MEM0_TELEMETRY"] = "True"
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_telemetry_still_off_after_mem0_init(tmp_path, monkeypatch):
    """建好 Memory 之后遥测还是关的：就算建库前环境变量被改成 True，
    _build_memory 里建 Memory 之前也会再强制设一次。

    用假 mem0 模块（sys.modules 里塞一个假的，不碰真实 mem0），
    再把 config.get_api_key 换成返回假 Key——建库流程走真实代码，
    只是底下的 mem0 和 Key 是假的。"""
    monkeypatch.setenv("MEM0_TELEMETRY", "True")
    monkeypatch.setattr(config, "get_api_key", lambda name: "fake-key-for-test")

    class _FakeMemoryClass:
        @classmethod
        def from_config(cls, config):
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=_FakeMemoryClass))
    store = Mem0Store(
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    assert store.available is True
    assert os.environ.get("MEM0_TELEMETRY") == "False"


def test_telemetry_off_after_restart():
    """重启后遥测还是关的：连续起两个全新进程（等于重启两次），
    每次 import 后 MEM0_TELEMETRY 都必须是 False——环境变量事先
    设成 "1"（另一种“开”的写法）也一样盖掉。"""
    code = (
        "import os, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import memory.mem0_store\n"
        "assert os.environ.get('MEM0_TELEMETRY') == 'False', "
        "repr(os.environ.get('MEM0_TELEMETRY'))\n"
    )
    env = dict(os.environ)
    env["MEM0_TELEMETRY"] = "1"
    for _ in range(2):  # 起两次，模拟两次重启
        proc = subprocess.run(
            [sys.executable, "-c", code, str(ROOT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr


# ---------- 2. 脱敏：_redact_sensitive / _contains_sensitive ----------


def test_redact_labeled_password_chinese():
    """中文“密码是 xxx”：只换密码本身，保留前面的话。"""
    assert _redact_sensitive("我的密码是 hunter2") == "我的密码是 【已隐藏敏感信息】"
    assert _contains_sensitive("我的密码是 hunter2") is True


def test_redact_labeled_password_english():
    """英文 "password is xxx"：同样只换密码本身。"""
    text = "My password is hunter2, I like coffee"
    redacted = _redact_sensitive(text)
    assert "hunter2" not in redacted
    assert "I like coffee" in redacted


def test_redact_mixed_content_keeps_valuable_part():
    """敏感信息混在正常内容里：只脱敏秘密部分，有用的那句留下来。"""
    text = "我喜欢喝咖啡，我的 API 密钥是 sk-abcdefghij1234567890。"
    redacted = _redact_sensitive(text)
    assert "sk-abcdefghij1234567890" not in redacted
    assert "我喜欢喝咖啡" in redacted


def test_redact_mixed_chinese_english():
    """中英混杂：中文名字留下来，英文标签后面的 Key 换掉。"""
    text = "我叫小宋，My API key is sk-abcdefghij1234567890，请记住"
    redacted = _redact_sensitive(text)
    assert "sk-abcdefghij1234567890" not in redacted
    assert "我叫小宋" in redacted


def test_redact_known_key_formats_unlabeled():
    """不带标签的已知密钥格式也认得：sk-、ghp_、AKIA、AIza、xox、hf_。"""
    assert "sk-abcdefghij1234567890" not in _redact_sensitive(
        "key 是 sk-abcdefghij1234567890 记一下"
    )
    assert "ghp_abcdefghij1234567890" not in _redact_sensitive(
        "token ghp_abcdefghij1234567890"
    )
    assert "AKIAIOSFODNN7EXAMPLE" not in _redact_sensitive("AKIAIOSFODNN7EXAMPLE")
    assert "AIzaSyA12345678901234567890123456789012" not in _redact_sensitive(
        "key AIzaSyA12345678901234567890123456789012"
    )
    assert "xoxb-123456789012-1234567890123-AbCdEfGh" not in _redact_sensitive(
        "xoxb-123456789012-1234567890123-AbCdEfGh"
    )
    assert "hf_abcdefghij1234567890" not in _redact_sensitive("hf_abcdefghij1234567890")


def test_redact_id_card_numbers():
    """身份证号：18 位（带校验位验算）和 15 位都脱敏。"""
    assert _SECRET_ID18 not in _redact_sensitive(f"身份证号是 {_SECRET_ID18}")
    assert _SECRET_ID18 not in _redact_sensitive(f"我的身份证是{_SECRET_ID18}，请查收")
    assert "110101900101123" not in _redact_sensitive("老身份证 110101900101123")


def test_redact_bank_card_numbers():
    """银行卡号：16 位过 Luhn 校验的脱敏，带空格的也整段脱敏。"""
    assert _SECRET_CARD not in _redact_sensitive(f"我的银行卡号是 {_SECRET_CARD}")
    assert _SECRET_CARD not in _redact_sensitive("卡号 4111 1111 1111 1111")


def test_redact_private_key_block_and_jwt():
    """私钥块整块脱敏；JWT 三段式脱敏。"""
    block = (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7\n"
        "-----END RSA PRIVATE KEY-----"
    )
    assert "MIIEowIBAAKCAQEA7" not in _redact_sensitive(f"我的私钥：{block}")
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    assert jwt not in _redact_sensitive(f"token 是 {jwt}")


def test_redact_generic_long_token():
    """没见过的长令牌：32 位以上、大小写+数字混搭的才算；
    纯小写十六进制（git hash 长相）不算，不误伤。"""
    long_token = "AbCdEfGh1234567890AbCdEfGh1234567890"
    assert long_token not in _redact_sensitive(f"token {long_token}")
    git_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    assert _redact_sensitive(f"提交 {git_hash}") == f"提交 {git_hash}"


def test_redact_leaves_normal_content_alone():
    """不是秘密的别误伤：手机号（不在本次范围）、过不了 Luhn 的
    长数字串、问句里的“密码是多少”，都原样保留。"""
    assert _redact_sensitive("请联系 13812345678") == "请联系 13812345678"
    assert _redact_sensitive("订单号 1234567812345678 请查收") == (
        "订单号 1234567812345678 请查收"
    )
    assert _redact_sensitive("密码是多少") == "密码是多少"
    assert _contains_sensitive("我喜欢喝咖啡。") is False


def test_meaningful_after_redaction():
    """脱敏后还剩不剩有用的内容：纯秘密不剩，混合内容还剩。"""
    assert _meaningful_after_redaction("我的密码是【已隐藏敏感信息】") is False
    assert (
        _meaningful_after_redaction("我喜欢喝咖啡，我的密码是【已隐藏敏感信息】")
        is True
    )
    assert _meaningful_after_redaction("我叫小宋。") is True


# ---------- 2a. 存前脱敏：add_turn / add_memory_text ----------


def test_add_turn_redacts_before_send(tmp_path):
    """add_turn 发给 mem0 之前先脱敏：FakeMemory 收到的消息里已经
    没有秘密了，存下来的提取结果里也没有，只有有用的那句还在。"""
    store = make_fake_store(tmp_path)
    store.add_turn(
        [
            {"role": "user", "content": "我的密码是 hunter2，我喜欢喝咖啡。"},
            {"role": "assistant", "content": "好的。"},
        ]
    )
    # Plan C：本轮先待定，flush 时才调提取——脱敏在 add_turn 入口做，
    # 发给 mem0 的已经是脱敏后的消息。
    store.flush_pending()
    sent_messages, _user_id, _infer = store._memory.add_calls[0]
    sent_text = " ".join(m.get("content") or "" for m in sent_messages)
    assert "hunter2" not in sent_text
    assert "【已隐藏敏感信息】" in sent_text
    texts = store._memory.all_texts()
    assert texts and "hunter2" not in " ".join(texts)
    assert any("我喜欢喝咖啡" in t for t in texts)


def test_add_turn_pure_secret_sends_nothing(tmp_path):
    """整轮只说了一个密码：脱敏后什么有用的内容都不剩，直接不调提取，
    省一次 API 调用，也不存。"""
    store = make_fake_store(tmp_path)
    assert store.add_turn([{"role": "user", "content": "我的密码是 hunter2"}]) == {}
    assert store._memory.add_calls == []
    assert store._memory.all_texts() == []


def test_add_memory_text_redacts_mixed_content(tmp_path):
    """直接存入混合内容：只脱敏秘密部分，有用的那句照常存，不整条丢掉。"""
    store = make_fake_store(tmp_path)
    result = store.add_memory_text("我喜欢喝咖啡，我的密码是 hunter2。")
    assert result.get("results")
    texts = store._memory.all_texts()
    assert len(texts) == 1
    assert "hunter2" not in texts[0]
    assert "我喜欢喝咖啡" in texts[0]


def test_add_memory_text_pure_secret_not_stored(tmp_path):
    """直接存入纯秘密：脱敏后什么都不剩，直接不存。"""
    store = make_fake_store(tmp_path)
    assert store.add_memory_text("我的密码是 hunter2") == {}
    assert store.add_memory_text(f"API 密钥是 {_SECRET_KEY}") == {}
    assert store._memory.all_texts() == []


class _LeakyExtractMemory(FakeMemory):
    """模拟提取器把秘密复述进记忆：add 收到的消息已经脱敏过了，
    但提取结果里还是带上了秘密（比如模型复述了没见过的密钥格式）。
    用来验证提取后的二次过滤能不能把它删掉。"""

    def add(self, messages, user_id=None, infer=True):
        result = super().add(messages, user_id=user_id, infer=infer)
        leaked_id = self._store(f"我的 API 密钥是 {_SECRET_KEY}")
        result["results"].append(
            {"id": leaked_id, "memory": f"我的 API 密钥是 {_SECRET_KEY}", "event": "ADD"}
        )
        return result


def test_add_turn_post_call_filter_deletes_leaked_memory(tmp_path):
    """提取后二次过滤：提取结果里冒出来的秘密记忆立刻删掉、不留库里，
    也不返回给调用方；正常的记忆不受影响。"""
    memory = _LeakyExtractMemory()
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    result = store.add_turn([{"role": "user", "content": "我叫小宋。"}])
    assert result.get("pending") is True
    # Plan C：名字是身份陈述，提交时走槽位 upsert（infer=False 原文存），
    # 漏出来的秘密记忆在提交时已被二次过滤删掉。
    result = store.flush_pending()
    assert _SECRET_KEY not in " ".join(store._memory.all_texts())
    assert store._memory.all_texts() == ["我叫小宋"]
    # 提交结果里也没有秘密：身份 upsert 的结果在 "identity" 里，
    # 正常那条是 added；"results" 里是提取结果，本轮没有提取。
    identity = result.get("identity") or []
    assert len(identity) == 1 and identity[0]["status"] == "added"
    assert _SECRET_KEY not in str(result)


# ---------- 2b. 用时拦截：search ----------


class _FixedSearchMemory(FakeMemory):
    """search 返回固定结果的 FakeMemory（绕开假打分，直接测适配层的过滤逻辑）。"""

    def __init__(self, fixed_results):
        super().__init__()
        self._fixed = list(fixed_results)

    def search(self, query, filters=None, top_k=5):
        return {"results": list(self._fixed)}


def test_search_never_injects_sensitive_memory(tmp_path):
    """用时拦截（语义路径）：检索结果里混着一条敏感记忆，它绝不能
    出现在返回里；正常的记忆照常返回。"""
    memory = _FixedSearchMemory(
        [
            {"id": "fake-secret", "memory": f"我的 API 密钥是 {_SECRET_KEY}", "score": 0.9},
            {"id": "fake-normal", "memory": "我叫小宋。", "score": 0.8},
        ]
    )
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    found = store.search("API 密钥是什么？")
    assert [item["id"] for item in found] == ["fake-normal"]
    assert _SECRET_KEY not in " ".join(item["text"] for item in found)


def test_search_keyword_supplement_never_injects_sensitive_memory(tmp_path):
    """用时拦截（关键词补充路径）：旧库里已有的敏感记忆，靠关键词
    命中了也不注入；正常的记忆照常捞回来。

    敏感记忆是直接往假 memory 里塞的（绕过适配层的存前脱敏），
    模拟“过滤加上之前就已存下”的旧数据。"""
    memory = FakeMemory()
    memory.add(
        [{"role": "user", "content": f"我的 API 密钥是 {_SECRET_KEY}"}], infer=False
    )
    memory.add([{"role": "user", "content": "我目前在圆通工作。"}], infer=False)
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    # 问圆通：正常记忆回来，敏感记忆不掺进来。
    found = store.search("圆通是做什么的？")
    assert len(found) == 1 and "圆通" in found[0]["text"]
    # 直接问密钥：语义分数不够、关键词命中了也被拦住，返回空。
    assert store.search("API 密钥是什么？") == []


# ---------- 2c. 导出过滤：get_all / migrate_from_v1 ----------


def test_get_all_excludes_sensitive_by_default(tmp_path):
    """get_all 默认不含敏感信息：旧库里已有的敏感记忆不会从这里漏出去；
    内部维护（删除归属校验、重嵌备份）才用 include_sensitive=True。"""
    store = make_fake_store(tmp_path)
    store._memory.add(
        [{"role": "user", "content": f"我的 API 密钥是 {_SECRET_KEY}"}], infer=False
    )
    store.add_memory_text("我叫小宋。")
    assert [item["text"] for item in store.get_all()] == ["我叫小宋。"]
    assert len(store.get_all(include_sensitive=True)) == 2


def test_delete_still_works_for_sensitive_memory(tmp_path):
    """归属校验用的是含敏感信息的全量：旧的敏感记忆照样能删掉，
    不会因为 get_all 默认过滤就查不到、删不掉。"""
    store = make_fake_store(tmp_path)
    store._memory.add(
        [{"role": "user", "content": f"我的 API 密钥是 {_SECRET_KEY}"}], infer=False
    )
    mem_id = store.get_all(include_sensitive=True)[0]["id"]
    assert store.delete_memory(mem_id, approve=True) is True
    assert store.get_all(include_sensitive=True) == []


def test_migrate_from_v1_filters_sensitive(tmp_path):
    """迁移过滤：正常记忆照常迁；混合内容只脱敏秘密部分再迁；
    整条只有秘密的跳过不迁；第二遍迁移返回 0。"""
    old_path = tmp_path / "memories.json"
    old_path.write_text(
        json.dumps(
            [
                {"text": "我叫小宋。"},
                {"text": "我喜欢喝咖啡，我的密码是 hunter2。"},
                {"text": "我的密码是 hunter2"},
                {"text": "每月预算 5 美元，4 美元提醒，5 美元停。"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = make_fake_store(tmp_path)
    assert store.migrate_from_v1(old_path) == 3
    texts = store._memory.all_texts()
    assert len(texts) == 3
    assert "hunter2" not in " ".join(texts)
    assert "我叫小宋。" in texts
    assert "我喜欢喝咖啡，我的密码是 【已隐藏敏感信息】。" in texts
    assert "每月预算 5 美元，到 5 美元提醒，不自动停。" in texts
    assert store.migrate_from_v1(old_path) == 0


# ---------- 2d. 日志过滤 ----------


def test_logs_never_contain_secrets(tmp_path, caplog):
    """日志过滤：存、迁、搜全流程走一遍，日志里不能出现任何秘密原文
    （只记个数、不记内容）。"""
    store = make_fake_store(tmp_path)
    # 直接往假 memory 里塞一条敏感记忆（模拟旧库数据），再搜它，
    # 触发“检索跳过敏感记忆”的日志。
    store._memory.add(
        [{"role": "user", "content": f"我的 API 密钥是 {_SECRET_KEY}"}], infer=False
    )
    old_path = tmp_path / "memories.json"
    old_path.write_text(
        json.dumps(
            [
                {"text": "我叫小宋。"},
                {"text": "我的密码是 hunter2"},
                {"text": "每月预算 5 美元，4 美元提醒，5 美元停。"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger="agent"):
        store.add_turn([{"role": "user", "content": f"我的 API 密钥是 {_SECRET_KEY}。"}])
        store.add_memory_text("我的密码是 hunter2")
        store.migrate_from_v1(old_path)
        store.search("API 密钥是什么？")
    assert _SECRET_KEY not in caplog.text
    assert _SECRET_PASSWORD not in caplog.text


# ---------- 3. 直接存入去重 ----------


def test_add_memory_text_dedups_same_text(tmp_path):
    """直接存入去重（DEDUP-B）：同一条文本存两次，库里只留一条；
    第二次返回已有的那条（event=NONE），且没再调 mem0.add。"""
    store = make_fake_store(tmp_path)
    first = store.add_memory_text("我每天晚上都读半小时书。")
    second = store.add_memory_text("我每天晚上都读半小时书。")
    assert len(store.get_all()) == 1
    assert second.get("duplicate") is True
    assert second["results"][0]["event"] == "NONE"
    assert second["results"][0]["id"] == first["results"][0]["id"]
    assert len(store._memory.add_calls) == 1


def test_add_memory_text_dedups_near_identical(tmp_path):
    """直接存入去重（近似）：只差句末标点的算同一条，不重复存。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我每天晚上都读半小时书。")
    store.add_memory_text("我每天晚上都读半小时书")
    store.add_memory_text("我每天晚上都读半小时书！")
    assert len(store.get_all()) == 1


def test_add_memory_text_does_not_merge_different_texts(tmp_path):
    """去重不过度：内容不同的两条都存下来。"""
    store = make_fake_store(tmp_path)
    store.add_memory_text("我喜欢喝咖啡。")
    store.add_memory_text("我喜欢喝绿茶。")
    assert len(store.get_all()) == 2


def test_normalize_memory_text():
    """去重用的规整：全角转半角、去空白、去末尾标点、英文小写。"""
    assert _normalize_memory_text("我每天晚上都读半小时书。") == _normalize_memory_text(
        "我每天晚上都读半小时书"
    )
    assert _normalize_memory_text("  I Like Coffee. ") == "ilikecoffee"
    assert _normalize_memory_text("ＡＢＣ") == "abc"
    assert _find_duplicate_memory(
        "我喜欢喝咖啡", [{"id": "1", "text": "我喜欢喝咖啡。"}]
    ) == {"id": "1", "text": "我喜欢喝咖啡。"}
    assert (
        _find_duplicate_memory("我喜欢喝绿茶", [{"id": "1", "text": "我喜欢喝咖啡。"}])
        is None
    )


class _DedupingExtractMemory(FakeMemory):
    """模拟真实 mem0 提取时的去重：同一句事实反复提取只存一条。
    FakeMemory 本身不去重（见 conftest 说明），这里用一个会去重的
    假对象，验证适配层没有破坏提取路径的去重（DEDUP-A）。
    只实现 infer=True 的路（本测试只走 add_turn）。"""

    def add(self, messages, user_id=None, infer=True):
        with self._lock:
            self.add_calls.append((list(messages or []), user_id, infer))
            stored = []
            for message in messages or []:
                if (message or {}).get("role") != "user":
                    continue
                for sentence in self._extract_sentences(
                    (message or {}).get("content") or ""
                ):
                    if any(item["memory"] == sentence for item in self._items):
                        continue
                    mem_id = self._store(sentence)
                    stored.append({"id": mem_id, "memory": sentence, "event": "ADD"})
            return {"results": stored}


def test_add_turn_dedup_preserved(tmp_path):
    """提取路径去重（DEDUP-A）没被破坏：同一句偏好经 add_turn 存两次，
    库里只留一条；适配层每次都正常调 add，去重是提取那侧做的事。"""
    memory = _DedupingExtractMemory()
    store = Mem0Store(
        memory=memory,
        chroma_path=tmp_path / "mem0_chroma",
        history_db_path=tmp_path / "history.db",
        version_path=tmp_path / "mem0_versions.json",
    )
    turn = [
        {"role": "user", "content": "我喜欢每天早上喝黑咖啡。"},
        {"role": "assistant", "content": "好的。"},
    ]
    store.add_turn(turn)
    store.add_turn(turn)
    # Plan C：第二轮提交第一轮的待定内容（1 次 add），再把自己待定；
    # flush 时按指纹去重、不再重复提——同一句话最终只存 1 条、只调 1 次 add。
    store.flush_pending()
    assert store._memory.all_texts() == ["我喜欢每天早上喝黑咖啡"]
    assert len(store._memory.add_calls) == 1

