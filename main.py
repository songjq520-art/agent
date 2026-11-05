# main.py（程序主入口）
# 作用：程序从这里启动，流程是：
#   1. 读配置（config.py），检查每个模型的状态（是否启用、有无 Key）；
#   2. 用 router.py（任务路由模块）分配任务；
#   3. 调用 providers/muse.py（Muse 模型接口）；
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
from providers.muse import MuseProvider
from router import route

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


def run_question(prompt, task_type="question", budget_tracker=None):
    """跑一次普通问答任务（最简单的运行示例）。

    参数 prompt：用户的问题；task_type：任务类型，默认 "question"。
    返回：字典，含 mode（模式）、route（路由结果）、result（模型结果或 None）、message（提示）。
    模式：
      - "demo"（演示模式）：无 Key，只演示配置 + 路由，不调用模型；
      - "real"（真实模式）：有 Key，真实调用 Muse API。
    """
    tracker = budget_tracker or BudgetTracker()
    status = get_model_status()
    routing = route(prompt, task_type)

    # 预算警告：达到 80% 时先提醒（只提醒，不拦截）。
    if tracker.is_over_warn() and not tracker.is_over_budget():
        print(f"⚠️ 预算警告：本月累计成本已达到预算 80%（${tracker.total_cost_usd:.4f} / ${MONTHLY_BUDGET_USD:.2f}）。")

    # 无 Key：演示模式，明确提示，不调用模型。
    api_key = get_api_key(routing["provider"])
    if not api_key:
        message = (
            "模拟模式 / 未接通真实 API：未配置 MODEL_API_KEY（或 MUSE_API_KEY），"
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

    # 有 Key：真实调用。
    provider = MuseProvider(api_key=api_key, budget_tracker=tracker)
    try:
        result = provider.chat(prompt)
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
    if tracker.is_over_warn():
        print(f"⚠️ 预算警告：本月累计成本 ${tracker.total_cost_usd:.4f} / ${MONTHLY_BUDGET_USD:.2f}。")
    return {
        "mode": "real",
        "route": routing,
        "result": result,
        "message": "真实调用完成。",
        "model_status": status,
    }


def main(argv=None):
    """命令行入口。

    用法：
      python3 main.py "你好"        # 问一个问题
      python3 main.py              # 不带问题时，程序会提示你输入
    """
    args = list(sys.argv[1:] if argv is None else argv)
    prompt = args[0] if args else input("请输入你的问题：").strip()
    if not prompt:
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
        print("\n--- Muse 的回答 ---")
        print(result["text"])
        print(f"\nToken 使用量：{result.get('usage')}")
        print(f"本次成本：{result.get('cost_usd')} 美元（None 表示模型未返回 usage，成本未知）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
