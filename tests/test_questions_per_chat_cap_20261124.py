# tests/test_questions_per_chat_cap_20261124.py — 修复 3 回归：
# 一轮最多问 3 个新问题；第 4 个留在 pending 里等下一轮，不标已问。
import json
import re

from memory.sqlite_store import SQLiteStore


class FakeImportanceJudge:
    """mock 重要性判官：按句子给 canned verdict（抄 test_phase2_importance_questions.py）。"""

    def __init__(self, verdicts=None, fail=False):
        self.verdicts = verdicts or {}
        self.fail = fail
        self.calls = []
        self.provider_name = "deepseek"
        self.model = "deepseek-flash"

    def chat_messages(self, messages, ledger=None, thinking=None,
                      max_tokens=None):
        prompt = (messages or [{}])[0].get("content") or ""
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("API 挂了")
        match = re.search(r"^句子：(.+)$", prompt, re.M)
        sentence = (match.group(1).strip() if match else "").rstrip("。，,. ")
        verdict = self.verdicts.get(
            sentence, {"important": False, "reason": "默认一次性经历"}
        )
        return {
            "text": json.dumps(verdict, ensure_ascii=False),
            "usage": {"prompt_tokens": 400, "completion_tokens": 80},
            "cost_usd": None,
            "model": "deepseek-flash",
        }


def _judge_store(tmp_path, monkeypatch, judge):
    monkeypatch.setattr(SQLiteStore, "_embed_text", lambda self, text: None)
    return SQLiteStore(memory_dir=str(tmp_path / "memory"),
                       judge_provider=judge)


SENTENCES = [
    "我表弟下个月可能来广州玩",
    "我朋友下周可能来上海出差",
    "同事下个月可能去深圳培训",
    "堂哥下个月可能去成都出差",
]


def test_four_unsure_sentences_only_three_asked(tmp_path, monkeypatch):
    judge = FakeImportanceJudge({
        s: {"important": "unsure", "reason": "拿不准%d" % (i + 1)}
        for i, s in enumerate(SENTENCES)
    })
    store = _judge_store(tmp_path, monkeypatch, judge)
    receipt = store.save_turn(
        [{"role": "user", "content": "。".join(SENTENCES) + "。"},
         {"role": "assistant", "content": "好的。"}],
        turn_id="t1",
    )
    assert len(receipt["stored"]) == 4
    questions = receipt.get("questions")
    assert questions is not None and len(questions) == 3
    assert [q["text"] for q in questions] == SENTENCES[:3]
    # 第 4 条没进回执、没标已问，还在 pending 里等下一轮。
    pending = store.list_pending_questions()
    assert len(pending) == 4
    fourth = [q for q in pending if q["text"] == SENTENCES[3]]
    assert len(fourth) == 1
    assert fourth[0]["status"] == "pending"
    assert all(q["status"] == "asked"
               for q in pending if q["text"] in SENTENCES[:3])
