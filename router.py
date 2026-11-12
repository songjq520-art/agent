# router.py（任务路由模块）
# 作用：按任务类型决定把任务交给哪个模型。
# 原理：很简单，就是一张“任务类型 → 模型”的对应表。
#       本次所有任务都交给 DeepSeek（第一任主控），Muse 保留为工作模型；
#       以后加模型或换主控时，只改这张表和 config.py，
#       main.py 和 providers 不用重写。
# 关系：main.py 先问 router“这个任务给谁”，再按 router 的答案去调用对应模型。
# 主控说明：主控（Controller）是“谁来做分配决策”这个角色，
#       本次就是 router.py 里这张简单的表；未来换主控时，
#       可以把这张表换成新主控的决策逻辑，其他模块不用改。

from config import DEFAULT_MODEL, MODEL_CONFIG
from tools.model_router import select_model

# TASK_TYPES（任务类型）：本次先实现三种。
#   question：普通问答任务；code：代码相关任务；general：通用任务。
TASK_TYPES = ("question", "code", "general")

# PROVIDER_FOR_TASK（任务类型 → 模型）：本次都交给 DeepSeek（第一任主控）。
# Muse 保留为工作模型，需要时改这张表就能再指过去。
PROVIDER_FOR_TASK = {
    "question": "deepseek",
    "code": "deepseek",
    "general": "deepseek",
}


def route(task_text, task_type="question"):
    """分配任务。

    参数：
      task_text：任务内容，例如用户的问题；
      task_type：任务类型，默认 "question"（普通问答）。
    返回：{"task_type": 任务类型, "provider": 模型名, "model": 官方模型名}
    原理：先检查任务类型是否在支持列表里，再查表找模型，
          最后检查该模型是否在 config.py 里启用。
    """
    if not task_text or not str(task_text).strip():
        raise ValueError("任务内容不能为空。")
    if task_type not in TASK_TYPES:
        raise ValueError(f"不支持的任务类型：{task_type}，支持的类型有：{TASK_TYPES}")

    provider = PROVIDER_FOR_TASK[task_type]
    cfg = MODEL_CONFIG[provider]
    if not cfg["enabled"]:
        raise RuntimeError(f"模型 {provider} 未启用，请先在 config.py 里启用它。")

    model = cfg["model"]
    if provider == "deepseek":
        # 满血路由（2026-11-13）：难问题走 deepseek-v4-pro，
        # 日常问题走 deepseek-flash（见 tools/model_router.py）。
        # 分类是原话的纯函数：同一句话每次都分到同一个模型，
        # 任务恢复时按原话重算也不会换模型，不触发模型锁定报错。
        model, _difficulty, _reason = select_model(task_text)

    return {
        "task_type": task_type,
        "provider": provider,
        "model": model,
    }


def default_provider():
    """返回默认模型名，方便 main.py 在没有指定任务类型时使用。"""
    return DEFAULT_MODEL
