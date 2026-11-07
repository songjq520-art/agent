# main.py（程序主入口）
# 作用：程序从这里启动，流程是：
#   1. 读配置（config.py），检查每个模型的状态（是否启用、有无 Key）；
#   2. 用 router.py（任务路由模块）分配任务；
#   3. 调用 providers/deepseek.py（DeepSeek 模型接口，默认）或
#      providers/muse.py（Muse 模型接口，工作模型）；
#   4. 展示结果，并写日志（运行记录）到 logs/agent.log。
# 关系：main.py 是“总指挥”，它本身不直接调 API，只负责串起 config、router、providers。
#
# 重要：无 Key 时是“模拟模式 / 未接通真实 API”，只做配置检查 + 路由演示，
#       不调用模型、不产生费用，不要把它当成真实调用成功。
#       有 Key 且用户确认后，才做真实调用。

import logging
import sys
from pathlib import Path

from config import (
    MONTHLY_BUDGET_USD,
    BudgetTracker,
    get_api_key,
    get_model_status,
)
from providers.deepseek import DeepSeekProvider
from providers.muse import MuseProvider
from router import route

# PROVIDER_CLASSES（模型名 → Provider 类）：按路由结果选调用哪个模型。
# DeepSeek 是默认主控，Muse 保留为工作模型；以后加新模型时在这里加一行。
PROVIDER_CLASSES = {
    "deepseek": DeepSeekProvider,
    "muse": MuseProvider,
}

# ---------- 日志配置 ----------
# 日志写到 logs/agent.log（文本格式），同时在终端打印一份。
# 记录内容：时间、模型、任务类型、是否成功、Token、成本、错误。
# 绝不记录 API Key。
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("agent")


def run_question(prompt, task_type="question", budget_tracker=None, messages=None):
    """跑一次普通问答任务（最简单的运行示例）。

    参数 prompt：用户的问题；task_type：任务类型，默认 "question"。
    参数 budget_tracker：共用的预算跟踪器，不传时新建一个。
    参数 messages：多轮对话的历史（可选），按时间顺序排列，每项为
    {"role": "user" 或 "assistant", "content": 文本}，应为本轮之前
    已完成的轮次（不含本轮问题）。不传时为单次问答；
    传入时，本轮会把历史 + 本轮问题一起发给模型（chat_messages），
    预算预估也覆盖完整历史。
    返回：字典，含 mode（模式）、route（路由结果）、result（模型结果或 None）、message（提示）。
    模式：
      - "demo"（演示模式）：无 Key，只演示配置 + 路由，不调用模型；
      - "real"（真实模式）：有 Key，真实调用对应模型的 API（默认 DeepSeek）。
    注意：本函数不修改传入的 messages 列表；调用方在成功拿到回答后，
    再把本轮 user 问题和 assistant 回答追加进历史。
    """
    tracker = budget_tracker or BudgetTracker()
    status = get_model_status()
    routing = route(prompt, task_type)

    # 预算文件损坏时明确提示：原文件已保留、未被覆盖，人工修复后才能恢复。
    if getattr(tracker, "corrupted", False):
        print(
            f"⚠️ 预算记录异常：{tracker.corruption_reason}，已暂停后续付费调用，"
            "请人工检查 logs/budget_usage.json（原文件已保留，未被覆盖），"
            "核对后手动修复该文件或备份后删除重建。"
        )

    # 预算警告：达到 80% 时先提醒（只提醒，不拦截）。
    if tracker.is_over_warn() and not tracker.is_over_budget():
        print(f"⚠️ 预算警告：本月累计成本已达到预算 80%（${tracker.total_cost_usd:.4f} / ${MONTHLY_BUDGET_USD:.2f}）。")

    # 无 Key：演示模式，明确提示，不调用模型。
    api_key = get_api_key(routing["provider"])
    if not api_key:
        key_name = (
            "DEEPSEEK_API_KEY"
            if routing["provider"] == "deepseek"
            else "MODEL_API_KEY（或 MUSE_API_KEY）"
        )
        message = (
            f"模拟模式 / 未接通真实 API：未配置 {key_name}，"
            "本次只做了配置检查和路由演示，未调用模型，未产生费用。"
            "真实 API 连接标记为未完成，等你准备好密钥后再测试。"
        )
        print(message)
        logger.info(
            "demo | model=%s task_type=%s success=True tokens=None cost=None error=None",
            routing["model"],
            routing["task_type"],
        )
        return {
            "mode": "demo",
            "route": routing,
            "result": None,
            "message": message,
            "model_status": status,
        }

    # 有 Key：真实调用。按路由结果选 Provider 类（DeepSeek 默认，Muse 保留为工作模型），
    # 并把路由算出的模型名传给 Provider——路由说用哪个模型就用哪个，
    # 不能丢了 routing["model"]、让 Provider 又按默认模型自己猜。
    provider_cls = PROVIDER_CLASSES[routing["provider"]]
    provider = provider_cls(api_key=api_key, budget_tracker=tracker, model=routing["model"])
    try:
        if messages is None:
            # 单次问答：只发本轮问题。
            result = provider.chat(prompt)
        else:
            # 多轮对话：把历史 + 本轮问题一起发给模型，预算预估也覆盖完整历史。
            send_messages = list(messages) + [{"role": "user", "content": prompt}]
            result = provider.chat_messages(send_messages)
    except RuntimeError as exc:
        logger.info(
            "real | model=%s task_type=%s success=False tokens=None cost=None error=%s",
            routing["model"],
            routing["task_type"],
            exc,
        )
        raise

    usage = result.get("usage") or {}
    logger.info(
        "real | model=%s task_type=%s success=True tokens=%s cost=%s error=None",
        result["model"],
        routing["task_type"],
        usage.get("total_tokens"),
        result.get("cost_usd"),
    )
    if result.get("cost_usd") is None:
        backend = "DeepSeek" if routing["provider"] == "deepseek" else "Meta"
        print(
            "⚠️ 本次调用成本未知（模型没有返回 usage），已暂停后续付费调用，"
            f"请先去{backend}官方后台核对实际消费，核对后再清除未知成本标记。"
        )
    if tracker.is_over_warn():
        print(f"⚠️ 预算警告：本月累计成本 ${tracker.total_cost_usd:.4f} / ${MONTHLY_BUDGET_USD:.2f}。")
    return {
        "mode": "real",
        "route": routing,
        "result": result,
        "message": "真实调用完成。",
        "model_status": status,
    }


# EXIT_COMMANDS（退出命令）：连续对话里输入这些就结束，不区分大小写，前后空格忽略。
# 退出时直接结束，不再调用 API。
EXIT_COMMANDS = ("exit", "quit", "退出")


def is_exit_command(text):
    """判断输入是否为退出命令（exit / quit / 退出）。"""
    return str(text or "").strip().lower() in EXIT_COMMANDS


def chat_loop(budget_tracker=None):
    """连续对话循环：持续与模型对话，直到输入退出命令。

    整个会话共用一个 BudgetTracker 和一个 messages 历史（user 和 assistant
    消息按时间顺序排列）。每轮：读用户输入 → 经原来路由和 Provider 调用
    机制（run_question，带完整历史）→ 打印回答 → 把本轮 user 问题和
    assistant 回答追加进历史。失败（调用报错、演示模式）时历史不动。
    退出命令（exit / quit / 退出）不调用 API，直接正常结束。
    返回 0 表示正常结束。
    """
    tracker = budget_tracker or BudgetTracker()
    messages = []
    print("=== 个人多模型 Agent（连续对话）===")
    print(f"模型状态：{get_model_status()}")
    print(f"每月预算：${MONTHLY_BUDGET_USD:.2f}，警告线 80%")
    print("输入问题开始对话，输入 exit、quit 或 退出 结束。")
    while True:
        try:
            prompt = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C：正常结束，不报错。
            print("\n对话结束。")
            break
        if not prompt:
            # 空输入：不调用 API，继续等下一句。
            continue
        if is_exit_command(prompt):
            # 退出命令：不调用 API，直接结束。
            print("对话结束。")
            break
        try:
            outcome = run_question(prompt, budget_tracker=tracker, messages=messages)
        except RuntimeError as exc:
            # 调用失败：历史不动，继续等下一句，不伪造回答。
            print(f"调用失败：{exc}")
            continue
        if outcome["mode"] == "demo":
            # 演示模式：未调用模型，没有回答可追加，历史不动。
            print("（演示模式，未调用模型）")
            continue
        result = outcome["result"]
        # 成功拿到回答后，再把本轮 user 问题和 assistant 回答追加进历史。
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": result["text"]})
        print(f"\n--- {result['model']} 的回答 ---")
        print(result["text"])
        print(f"\nToken 使用量：{result.get('usage')}")
        print(f"本次成本：{result.get('cost_usd')} 美元（None 表示模型未返回 usage，成本未知）")
    return 0


def main(argv=None):
    """命令行入口。

    用法：
      python3 main.py "你好"        # 问一个问题（单次问答）
      python3 main.py              # 不带问题时，进入连续对话循环，
                                   # 可持续对话，直到输入 exit、quit 或 退出
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        # 不带参数：连续对话循环。
        return chat_loop()

    prompt = args[0]
    if not prompt.strip():
        print("问题不能为空。")
        return 1

    print("=== 个人多模型 Agent（最小骨架）===")
    print(f"模型状态：{get_model_status()}")
    print(f"每月预算：${MONTHLY_BUDGET_USD:.2f}，警告线 80%")

    try:
        outcome = run_question(prompt)
    except RuntimeError as exc:
        print(f"调用失败：{exc}")
        return 1

    print(f"任务类型：{outcome['route']['task_type']}，交给模型：{outcome['route']['model']}")
    if outcome["mode"] == "demo":
        print("（演示模式，未调用模型）")
    else:
        result = outcome["result"]
        print(f"\n--- {result['model']} 的回答 ---")
        print(result["text"])
        print(f"\nToken 使用量：{result.get('usage')}")
        print(f"本次成本：{result.get('cost_usd')} 美元（None 表示模型未返回 usage，成本未知）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
