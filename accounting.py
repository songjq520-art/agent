# accounting.py（大模型调用逐笔记账 + 自对账）
# 作用：把每一次真实大模型调用逐笔写进持久账本 logs/llm_calls.jsonl
#       （JSONL 格式，一行一笔，append-only 只追加不改），并提供按月
#       汇总和与平台账单对账的函数 + 命令行。
# 背景：BudgetTracker（config.py）的逐笔 records 只留最近 500 笔、还随
#       budget_usage.json 跨月重置——2026-11 月 agent 钥匙共 18,095 次
#       调用，事后查不到逐笔账，只能按排除法推断。这个账本就是来补
#       这个洞的：一笔不丢、跨月不清、坏行可跳过。
# 关系：config.py 的 BudgetTracker.add_cost / mark_unknown_cost 是两家
#       provider 的记账汇合点，它们在记完预算账后把同一笔转写到这里；
#       provider 没带 budget_tracker 时，由 provider 直接调这里补记；
#       tools/cli.py 的翻译直连也在这里落账（堵绕过口子）。
# 约束：本模块只用标准库，不 import config / providers（防循环导入）；
#       记账是旁路——写失败只打印醒目提醒并返回 False，绝不抛异常
#       打断业务调用（口径错误如非法 purpose 除外，那是编程错误要响）。

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 调用用途只认这三类：日常（正常业务）、测试、探针（体检探活等）。
PURPOSES = ("日常", "测试", "探针")

# 环境变量名：改账本文件位置（测试隔离、对账临时账本都靠它）。
ENV_PATH_VAR = "SHIER_LLM_CALLS_PATH"


def _resolve_path(path=None):
    """定账本文件路径：显式参数优先，其次环境变量，最后默认值。

    默认值 = 本文件同目录下 logs/llm_calls.jsonl（与 budget_usage.json
    同一个 logs 目录，那个目录已被 .gitignore 忽略，不进 GitHub）。
    """
    if path:
        return Path(path)
    env_path = os.environ.get(ENV_PATH_VAR)
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "logs" / "llm_calls.jsonl"


def _extract_tokens(usage):
    """从模型返回的 usage 里提取三类 Token 数，缺了记 0，不编造。

    cached_tokens 在 prompt_tokens_details 里（和 BudgetTracker 的
    提取口径一致）。usage 不是字典时按无 usage 处理。
    """
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    return prompt_tokens, completion_tokens, cached_tokens


def record_call(*, provider, model, cost_usd, usage=None, purpose="日常",
                caller="", ledger_book="", task_id="", trace_id="",
                status="ok", estimated=False, path=None):
    """往持久账本追加一笔调用记录，成功返回 True，写失败返回 False。

    参数 provider / model：哪家、哪个模型（例如 "deepseek" /
    "deepseek-flash"）。参数 cost_usd：本笔成本（美元），成本未知时
    传 None（记成 null，绝不编造 0）。参数 usage：模型返回的 usage
    字典，用来提 Token 数；没有就三类 Token 都记 0。
    参数 purpose：用途，只认 PURPOSES 三类，传错抛 ValueError
    （这是编程错误，必须响，不能悄悄记错账）。参数 caller：调用方
    标识（例如 "deepseek.chat_messages"、"tools.cli.translate"）；
    ledger_book：预算账本标签（"tool" / "chat"，没有就空字符串）；
    task_id / trace_id：串联任务与循环的编号，没有就空字符串。

    status 与 estimated 的口径（两者都记进每行，恒自洽）：
      - status="ok"、estimated=False：有真实 usage、按价目算出的钱；
      - status="estimated"、estimated=True：估算的钱（没真实 usage）；
      - status="unknown_cost"、estimated=False：成本未知，cost_usd=None。
    本函数会把 estimated 与 status 对齐：status="estimated" 时
    estimated 强制为 True，其余情况按传入的 estimated 记。

    写文件失败（磁盘、权限等）时：打印醒目提醒、返回 False，不抛
    异常——记账故障不许打断业务调用，漏记的笔会在自对账时以
    「总额对不上」的形式被发现（见 reconcile）。
    """
    if purpose not in PURPOSES:
        raise ValueError(
            f"未知用途：{purpose!r}，只认 {PURPOSES}（日常/测试/探针）。"
        )
    if status == "estimated":
        estimated = True
    prompt_tokens, completion_tokens, cached_tokens = _extract_tokens(usage)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "provider": provider or "",
        "model": model or "",
        "purpose": purpose,
        "caller": caller or "",
        "ledger_book": ledger_book or "",
        "task_id": task_id or "",
        "trace_id": trace_id or "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cost_usd": round(float(cost_usd), 6) if cost_usd is not None else None,
        "status": status,
        "estimated": bool(estimated),
    }
    target = _resolve_path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(
            f"【记账失败】大模型调用账本写入失败（{target}）：{exc}。"
            "本次调用未落账，请检查磁盘/权限；月度对账会发现差额。"
        )
        return False
    return True


def read_calls(month=None, path=None):
    """读账本，返回记录列表（每项一个字典，按写入顺序）。

    参数 month："YYYY-MM" 时只返回 ts 以该月开头的记录；不传返回全部。
    文件不存在返回空列表；坏行（解析不了的行）跳过，不报错——账本是
    append-only 的，尾巴上偶尔有一行写一半是可能的，不能整本读不出。
    """
    target = _resolve_path(path)
    if not target.exists():
        return []
    rows = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if month and not str(row.get("ts") or "").startswith(month):
            continue
        rows.append(row)
    return rows


def monthly_summary(month, path=None):
    """某月账本汇总。返回字典：

    {month, calls（总笔数）, total_usd（只汇总 cost 非 null 的笔）,
     unknown_cost_calls（status == "unknown_cost" 的笔数）,
     by_model / by_purpose（分项：{名字: {"calls": 笔数,
     "total_usd": 金额}}）}
    """
    rows = read_calls(month=month, path=path)
    total = 0.0
    unknown = 0
    by_model = {}
    by_purpose = {}
    for row in rows:
        cost = row.get("cost_usd")
        has_cost = isinstance(cost, (int, float)) and not isinstance(cost, bool)
        amount = float(cost) if has_cost else 0.0
        if has_cost:
            total += amount
        if row.get("status") == "unknown_cost":
            unknown += 1
        for bucket, key in ((by_model, row.get("model") or ""),
                            (by_purpose, row.get("purpose") or "")):
            entry = bucket.setdefault(key, {"calls": 0, "total_usd": 0.0})
            entry["calls"] += 1
            if has_cost:
                entry["total_usd"] = round(entry["total_usd"] + amount, 6)
    return {
        "month": month,
        "calls": len(rows),
        "total_usd": round(total, 6),
        "unknown_cost_calls": unknown,
        "by_model": by_model,
        "by_purpose": by_purpose,
    }


def reconcile(month, platform_total_usd, path=None,
              ratio_threshold=0.02, abs_threshold_usd=0.07):
    """拿某月账本总额和平台账单总额对账，返回对账结果字典：

    {month, ledger_total_usd（账本）, platform_total_usd（平台）,
     diff_usd（账本 - 平台）, diff_ratio（|diff| / 平台，平台为 0 时
     None）, unknown_cost_calls, ok}

    ok 的条件（两个都要满足）：本月没有成本未知的笔
    （unknown_cost_calls == 0），且 |diff| 不超过
    max(平台总额 * ratio_threshold, abs_threshold_usd)。
    阈值默认：比例 2%、绝对额 $0.07（约合人民币 0.5 元），取大者——
    小额月份不被比例卡死，大额月份不被绝对额放过。
    """
    summary = monthly_summary(month, path=path)
    ledger_total = summary["total_usd"]
    platform_total = float(platform_total_usd)
    diff = round(ledger_total - platform_total, 6)
    diff_ratio = (abs(diff) / platform_total) if platform_total else None
    allowed = max(platform_total * ratio_threshold, abs_threshold_usd)
    ok = summary["unknown_cost_calls"] == 0 and abs(diff) <= allowed
    return {
        "month": month,
        "ledger_total_usd": ledger_total,
        "platform_total_usd": round(platform_total, 6),
        "diff_usd": diff,
        "diff_ratio": diff_ratio,
        "unknown_cost_calls": summary["unknown_cost_calls"],
        "ok": ok,
    }


# ---------- 命令行 ----------
#   python accounting.py summary [YYYY-MM]            本月（或指定月）汇总
#   python accounting.py reconcile YYYY-MM <平台总额>  与平台账单对账

def _print_summary(summary):
    print(f"月份：{summary['month']}")
    print(f"调用笔数：{summary['calls']}")
    print(f"总金额（美元，只算成本已知的笔）：{summary['total_usd']:.6f}")
    print(f"成本未知笔数：{summary['unknown_cost_calls']}")
    print("按模型：")
    for name, item in sorted(summary["by_model"].items()):
        print(f"  {name or '(空)'}：{item['calls']} 笔，{item['total_usd']:.6f} 美元")
    print("按用途：")
    for name, item in sorted(summary["by_purpose"].items()):
        print(f"  {name or '(空)'}：{item['calls']} 笔，{item['total_usd']:.6f} 美元")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法：python accounting.py summary [YYYY-MM]")
        print("     python accounting.py reconcile YYYY-MM <平台总额美元>")
        return 2
    command = argv[0]
    if command == "summary":
        month = argv[1] if len(argv) > 1 else datetime.now().strftime("%Y-%m")
        _print_summary(monthly_summary(month))
        return 0
    if command == "reconcile":
        if len(argv) < 3:
            print("用法：python accounting.py reconcile YYYY-MM <平台总额美元>")
            return 2
        month = argv[1]
        try:
            platform_total = float(argv[2])
        except ValueError:
            print(f"平台总额不是数字：{argv[2]}")
            return 2
        result = reconcile(month, platform_total)
        print(f"月份：{result['month']}")
        print(f"账本总额（美元）：{result['ledger_total_usd']:.6f}")
        print(f"平台总额（美元）：{result['platform_total_usd']:.6f}")
        print(f"差额（账本 - 平台，美元）：{result['diff_usd']:.6f}")
        if result["diff_ratio"] is None:
            print("差额比例：平台总额为 0，无法计算")
        else:
            print(f"差额比例：{result['diff_ratio']:.4%}")
        print(f"成本未知笔数：{result['unknown_cost_calls']}")
        print("对账结果：" + ("对得上" if result["ok"] else "对不上，请人工核查"))
        return 0 if result["ok"] else 1
    print(f"未知命令：{command}，只认 summary / reconcile。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
