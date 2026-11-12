# tests/test_delete_targeting_57.py（57 号：删记忆找错对象）
# 作用：把 2026-11-12 笔记本泛化实测里两次删错对象的失败轨迹钉死在测试里。
#   第一次（09:01）：BOOS 说“删掉”（指刚说的鸡杂面），确认卡上却是
#     memory_id 33307618-7011-41e6-8f49-341d76212e17、文本“BOOS 爱吃拉面。”。
#   第二次（09:03）：BOOS 说“删掉鸡杂面，拉面别动”，新卡却是
#     memory_id 1d6d59be-8f8e-4adc-924d-2a939ce0ec58、文本 Agent 目标那条。
# 原理：删链路上没有任何程序校验“模型挑的这条是不是用户要删的那条”——
#   _run_delete_memory（tools/memory_tool.py:240）只验两件事：id 在库里
#   （deletion_scope）、原文逐字对得上（_same_text）；挑错的真条目这两项
#   全过，照样进确认卡、照样能删。确认卡原样显示模型传的参数
#   （web/index.html:756 JSON.stringify(needs.args)），网关确认后只执行
#   存着的那份参数（tools/tool_gateway.py:508/560），所以卡上出现什么
#   完全取决于模型那一次挑了哪行。
#   模型只能从当轮 list_memories 的结果里拿到 memory_id：记忆提示块只拼
#   文本不带 id（main.py:229），循环传给模型的历史只有最近 3 轮 user/
#   assistant 文本（tools/agent_loop.py:155），UUID 随机撞上真实 id 的
#   概率为零——两次卡上的 UUID 只能来自当轮现调的 list_memories。
#   根因不是 DeepSeek 猜的“删除路径用了相似度直接选”：deletion_scope
#   （memory/mem0_store.py:5046）和 delete_memory（:5304）都是按 id
#   查库，没有任何相似度；相似度只在 search() 检索和纠正安全网里，
#   不在删除路径上。真正缺的是程序锚定：删除目标（用户原话里的“鸡杂面”）
#   从来没进过删除工具的参数，程序没法核对。
# 期望的修法（只钉测试，不改生产代码）：delete_memory 工具加 anchor_text
#   参数（用户原话里的删除目标）；_run_delete_memory 拒绝原文里不含
#   anchor 的条目，零命中/多命中都停下来问，不自动挑。
# 运行：不要和全套件并行跑（之前并行跑写坏过 tasks.json）。
#   cd ~/workspace/merge-resolve && ~/workspace/mem0-spike/.venv/bin/python \
#       -m pytest tests/test_delete_targeting_57.py -q
#   现状预期：test_57_first_failure_* 和 test_57_second_failure_* 失败
#   （错条目照样删）；test_57_correct_entry_* 通过（对照：对的条目能删）；
#   test_57_list_truncation_cliff 失败（潜伏悬崖，见下面）；其余通过。

import json
import uuid

import pytest

from conftest import make_fake_store
from tools.agent_loop import TOOL_RESULT_MAX_CHARS, _wrap_tool_result
from tools.memory_tool import bound_memory_store, register_memory_tools


# ---------- 57 号当时的库（按 list 顺序：mem0 原样，老的在前） ----------

# 09:01 前后库里约 21 条。这里只放和两条失败轨迹有关 + 占位的条目；
# 顺序按存入顺序：Agent 目标是种子记忆（老，在前），面条三条是 08:58
# 刚存的（新，在后）——和真实库一致。
_LIBRARY_TEXTS = [
    "用户要求被称为BOOS。",
    "用户给助手起名为十二，并要求以后都用十二称呼助手。",
    "用户正在开发自己的个人 AI Agent，长期目标是让它成为能帮助自己工作和创业的个人助手。",
    "是每天定投 5 块钱买万家纳指。基金代码0194442。",
    "每月预算 5 美元，到 5 美元提醒，不自动停。",
    "用户喜欢喝冰美式。",
    "用户喜欢吃牛肉面。",
    "BOOS 喜欢吃鸡杂面。",
    "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。",
    "BOOS 爱吃拉面。",
]


@pytest.fixture
def mem_store(tmp_path):
    """57 号场景的假记忆库：鸡杂面（要删的目标）、拉面（第一次错挑的）、
    Agent 目标（第二次错挑的）都在库里，另有几条占位记忆。"""
    store = make_fake_store(tmp_path)
    for text in _LIBRARY_TEXTS:
        store.add_memory_text(text, source="user")
    return store


def _mem_id(mem_store, text):
    """按原文在库里找一条记忆的 id。"""
    for item in mem_store.get_all(include_sensitive=True):
        if item["text"] == text:
            return item["id"]
    raise AssertionError(f"库里没有这条记忆：{text}")


def _texts(mem_store):
    return [item["text"] for item in mem_store.get_all(include_sensitive=True)]


# ---------- 第一次失败（09:01）：要删鸡杂面，模型传了拉面 ----------

def test_57_first_failure_wrong_real_entry_must_be_refused(mem_store):
    """57 号第一次：BOOS 说“删掉”（指鸡杂面），模型却传了拉面的 id + 原文。
    拉面是库里真实存在的条目，id 对得上、原文逐字对得上——现在的
    _run_delete_memory 两项全过，会返回 ok=True 把拉面删掉。
    修好之后：anchor_text（用户原话里的删除目标“鸡杂面”）和要删那条
    的原文对不上，必须拒绝，不删。"""
    from tools.memory_tool import _run_delete_memory

    with bound_memory_store(mem_store):
        result = _run_delete_memory(
            {
                "memory_id": _mem_id(mem_store, "BOOS 爱吃拉面。"),
                "text": "BOOS 爱吃拉面。",
                "anchor_text": "鸡杂面",
            }
        )
    assert result["ok"] is False, (
        "拉面不含“鸡杂面”，不是 BOOS 要删的那条，必须拒绝——"
        f"现在却返回了 ok={result['ok']}，这就是 09:01 的失败"
    )
    assert "BOOS 爱吃拉面。" in _texts(mem_store)
    assert "BOOS 喜欢吃鸡杂面。" in _texts(mem_store)


# ---------- 第二次失败（09:03）：要删鸡杂面，模型传了 Agent 目标 ----------

def test_57_second_failure_unrelated_entry_must_be_refused(mem_store):
    """57 号第二次：BOOS 明确说“删掉鸡杂面，拉面别动”，模型却传了
    Agent 目标那条的 id + 原文——跟面条完全不相干，id 和原文在库里
    也都真实存在，现在的 _run_delete_memory 同样会放行。
    修好之后：anchor“鸡杂面”和 Agent 目标的原文对不上，必须拒绝。"""
    from tools.memory_tool import _run_delete_memory

    agent_goal = "用户正在开发自己的个人 AI Agent，长期目标是让它成为能帮助自己工作和创业的个人助手。"
    with bound_memory_store(mem_store):
        result = _run_delete_memory(
            {
                "memory_id": _mem_id(mem_store, agent_goal),
                "text": agent_goal,
                "anchor_text": "鸡杂面",
            }
        )
    assert result["ok"] is False, (
        "Agent 目标和“鸡杂面”半点关系没有，必须拒绝——"
        f"现在却返回了 ok={result['ok']}，这就是 09:03 的失败"
    )
    assert agent_goal in _texts(mem_store)
    assert "BOOS 喜欢吃鸡杂面。" in _texts(mem_store)


# ---------- 对照：挑对了的那条必须还能删（修法不能把删路堵死） ----------

def test_57_correct_entry_still_deletes(mem_store):
    """对照：模型传了鸡杂面的 id + 原文，anchor 也是“鸡杂面”，
    对得上——必须能删。修 anchor 校验时不能把这条路堵死。"""
    from tools.memory_tool import _run_delete_memory

    with bound_memory_store(mem_store):
        result = _run_delete_memory(
            {
                "memory_id": _mem_id(mem_store, "BOOS 喜欢吃鸡杂面。"),
                "text": "BOOS 喜欢吃鸡杂面。",
                "anchor_text": "鸡杂面",
            }
        )
    assert result["ok"] is True
    assert "BOOS 喜欢吃鸡杂面。" not in _texts(mem_store)
    assert "BOOS 爱吃拉面。" in _texts(mem_store)


# ---------- 潜伏悬崖：list 结果 4000 字截断（不是 09:01 的成因） ----------

def test_57_list_truncation_cliff():
    """潜伏问题，不是 09:01/09:03 的成因——当时库里约 21 条，list 结果
    约 3631 字，没超 4000 字上限，模型能看到全部。
    但库再长几条就超了：工具结果一律在 _wrap_tool_result 里截到前
    4000 字（tools/agent_loop.py:177），list 是按存入顺序排的（老的在前），
    尾巴上最新、最可能要删的那几条会被静默截掉，模型连挑的机会都没有。
    25 条真实长度的条目（UUID 级 id）已经超了：断言尾巴那条还在，
    现在就会失败。修法（二选一）：list_memories 返回改成分页/只返回
    id + 短文本，或把截断上限对 list 单独放宽。"""
    texts = [
        "用户要求被称为BOOS。",
        "用户给助手起名为十二，并要求以后都用十二称呼助手。",
        "用户在2026年11月9日明确要求记住：助手的名字是十二。",
        "用户说明助手十二是十一的弟弟。",
        "用户正在开发自己的个人 AI Agent，长期目标是让它成为能帮助自己工作和创业的个人助手。",
        "BOOS 的电脑配置：Windows 11 Home 23H2，ASUS Vivobook K3402ZA，i5-12500H，16GB 内存，Python 3.12.10，Git，VS Code。仓库在 D:\\代码\\agent，文件工具的工作区围栏是 D:\\代码\\十二的工作区。",
        "是每天定投 5 块钱买万家纳指。基金代码0194442。",
        "每月预算 5 美元，到 5 美元提醒，不自动停。",
        "每天定投 5 块买万家纳指（代码0194442），预算每月5美元。",
        "用户喜欢喝冰美式。",
        "用户每天早上7点起床。",
        "用户想购买碧桂园的房子，请助手帮忙推荐。",
        "助手叫十二。",
        "用户希望被称为BOOS，并且以后也保持这个称呼。",
        "用户给助手取名为「十二」。",
        "用户的灵魂写在 SOUL.md 里：做 BOOS 的个人助理，先可靠、再暖。",
        "用户喜欢吃牛肉面。",
        "BOOS 喜欢吃鸡杂面。",
        "用户表示其实也不是很喜欢吃鸡杂面，纠正了此前“喜欢吃鸡杂面”的说法。",
        "BOOS 爱吃拉面。",
        "用户每天定投 5 块钱买万家纳指。",
        "用户每天定投 5 块钱买万家纳指，交易日才扣款。",
        "用户说以后每天7点起床。",
        "用户现在要每天定投5刀的沪深300。",
        "用户不喜欢冰美式了，现在喜欢喝美式。",
    ]
    # list_memories 现在新在前（57 号修法）：夹具按 list 的实际输出顺序建
    # （最新在前），断言最新的那条在截断后还能被模型看到。
    data = {
        "count": len(texts),
        "memories": [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "created_at": "2026-11-12T08:00:00+08:00",
                "source": "inferred",
                "confirmed": False,
            }
            for text in reversed(texts)
        ],
    }
    raw = json.dumps(data, ensure_ascii=False)
    assert len(raw) > TOOL_RESULT_MAX_CHARS, (
        f"夹具只有 {len(raw)} 字，没超过 {TOOL_RESULT_MAX_CHARS} 字上限，"
        "测不到截断——夹具要加长"
    )
    wrapped = _wrap_tool_result(data)
    # 最后一条（最新存的“不喜欢冰美式了”）在 JSON 的尾巴上，截断后看不到；
    # 再长两三条，拉面、鸡杂面这些最新条目也会一样被截掉。
    assert "用户不喜欢冰美式了，现在喜欢喝美式。" in wrapped, (
        "25 条的 list 结果被截到 4000 字，尾巴上的最新条目模型根本看不到——"
        "再长几条，要删的最新记忆就会这样被截掉"
    )


# ---------- 机制钉死：确认卡显示的就是模型传的参数原样 ----------

def test_57_approval_card_shows_model_args_verbatim(mem_store, tmp_path):
    """机制说明（现在就是这样，修法的前提）：模型调 delete_memory 时，
    网关不执行、只把模型传的参数原样存进确认记录（tasks/task_store.py
    make_approval），卡片原样显示这份参数（web/index.html:756），
    BOOS 点确认后网关只执行存着的那份参数（tools/tool_gateway.py:508）。
    所以卡上写错，根子在模型传参那一步，不在卡片渲染、不在确认恢复。
    这个测试现在就通过，只钉住这个机制别被修法改坏。"""
    from tasks.task_store import TaskStore
    from tools.audit import AuditLog
    from tools.tool_gateway import ToolGateway

    register_memory_tools()
    gateway = ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))
    store = TaskStore(file_path=str(tmp_path / "tasks.json"))
    _ = store
    ramen_id = _mem_id(mem_store, "BOOS 爱吃拉面。")
    with bound_memory_store(mem_store):
        outcome = gateway.execute(
            "delete_memory",
            {
                "memory_id": ramen_id,
                "text": "BOOS 爱吃拉面。",
                "anchor_text": "拉面",
            },
        )
    assert outcome["decision"] == "needs_approval"
    approval = outcome["approval"]
    # 卡上显示的 id + 原文 + anchor 和模型传的一字不差——卡没改过参数；
    # 程序算好的匹配结果也补在卡上（_anchor_match）。
    assert approval["args"]["memory_id"] == ramen_id
    assert approval["args"]["text"] == "BOOS 爱吃拉面。"
    assert approval["args"]["anchor_text"] == "拉面"
    assert "拉面" in approval["args"].get("_anchor_match", "")
    assert "BOOS 爱吃拉面。" in approval["args"].get("_anchor_match", "")
    # 没确认之前库里一条没动。
    assert "BOOS 爱吃拉面。" in _texts(mem_store)


# ---------- anchor 边界：缺 anchor、零命中、多命中、执行前再验 ----------

def test_57_anchor_missing_is_rejected(mem_store, tmp_path):
    """57 号：删记忆不写 anchor_text，网关直接拒绝，不建确认卡。"""
    from tools.audit import AuditLog
    from tools.tool_gateway import ToolGateway

    register_memory_tools()
    gateway = ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))
    ramen_id = _mem_id(mem_store, "BOOS 爱吃拉面。")
    with bound_memory_store(mem_store):
        outcome = gateway.execute(
            "delete_memory",
            {"memory_id": ramen_id, "text": "BOOS 爱吃拉面。"},
        )
    assert outcome["decision"] == "rejected"
    assert "anchor_text" in outcome["reason"]
    assert "BOOS 爱吃拉面。" in _texts(mem_store)


def test_57_anchor_no_match_is_rejected(mem_store):
    """57 号：anchor 在库里一条都对不上，拒绝删。"""
    from tools.memory_tool import _run_delete_memory

    with bound_memory_store(mem_store):
        result = _run_delete_memory(
            {
                "memory_id": _mem_id(mem_store, "BOOS 爱吃拉面。"),
                "text": "BOOS 爱吃拉面。",
                "anchor_text": "不存在的面",
            }
        )
    assert result["ok"] is False
    assert "没找到" in result["error"]
    assert "BOOS 爱吃拉面。" in _texts(mem_store)


def test_57_anchor_multi_match_pick_still_deletes(mem_store, tmp_path):
    """57 号：anchor 命中多条不拦——库里常有“关于同一件事的说法”
    （比如撤回句本身也含“鸡杂面”），命中多条不代表挑错了。
    只要挑的这条原文里有 anchor 就放行，确认卡上注明库里还有几条
    相关记忆，让 BOOS 再看一眼。"""
    from tools.audit import AuditLog
    from tools.tool_gateway import ToolGateway

    register_memory_tools()
    gateway = ToolGateway(audit_log=AuditLog(str(tmp_path / "audit.log")))
    mem_store.add_memory_text("用户喜欢吃拉面。", source="user")
    assert len(_texts(mem_store)) == len(_LIBRARY_TEXTS) + 1
    with bound_memory_store(mem_store):
        outcome = gateway.execute(
            "delete_memory",
            {
                "memory_id": _mem_id(mem_store, "BOOS 爱吃拉面。"),
                "text": "BOOS 爱吃拉面。",
                "anchor_text": "拉面",
            },
        )
    assert outcome["decision"] == "needs_approval"
    assert "还有 1 条" in outcome["approval"]["args"].get("_anchor_match", "")
    # 没确认之前，库里一条没动。
    assert "BOOS 爱吃拉面。" in _texts(mem_store)
    assert "用户喜欢吃拉面。" in _texts(mem_store)


def test_57_anchor_checked_again_at_execution(mem_store):
    """57 号：执行前再验一次 anchor——条目已经不在了就不执行，
    不会错删别的，也不会把“已经不在了”当成成功。
    （注意：_run_delete_memory 本身不验确认卡，确认卡是网关那一关；
    这里只验执行时的 anchor 再核对。）"""
    from tools.memory_tool import _run_delete_memory

    chicken_id = _mem_id(mem_store, "BOOS 喜欢吃鸡杂面。")
    args = {
        "memory_id": chicken_id,
        "text": "BOOS 喜欢吃鸡杂面。",
        "anchor_text": "鸡杂面",
    }
    with bound_memory_store(mem_store):
        first = _run_delete_memory(args)
    assert first["ok"] is True
    assert "BOOS 喜欢吃鸡杂面。" not in _texts(mem_store)
    # 同一份参数再调一次：鸡杂面那条已不在库里，挑的 id 对不上任何一条
    # 含“鸡杂面”的记忆（只剩撤回句那条），必须拒绝。
    with bound_memory_store(mem_store):
        second = _run_delete_memory(args)
    assert second["ok"] is False
    assert "挑错了" in second["error"]
    assert "BOOS 爱吃拉面。" in _texts(mem_store)
