# config.py（项目配置模块）
# 作用：所有配置集中在这里，不要把配置分散到各个文件里。
# 原理：其他模块（main.py、router.py、providers/muse.py）都从这里读配置，
#       以后改模型、改预算、换主控，只改这一个文件。
# 关系：config.py 是“总开关”，main.py 启动时先读它，router.py 按它分配模型。

import json
import os
from pathlib import Path

# ---------- 模型配置 ----------
# MODEL_CONFIG（模型配置）：每个模型一个条目。
#   model：官方模型名；enabled：是否启用；base_url：接口地址；
#   price：每百万 Token 的价格（美元），仅用于本地记账，以官方定价页为准。
#   注意：gpt / gemini / grok 的 model 先留空（None），不要猜测官方模型名，
#         等真正接入时再按官方文档填写。
MODEL_CONFIG = {
    "muse": {
        "model": "muse-spark-1.3",  # Muse Spark 1.3，官方推荐新工作使用
        "enabled": True,  # 本次唯一启用的模型
        "base_url": "https://api.meta.ai/v1",  # Meta Model API 官方地址
        "price_input_per_1m": 1.25,   # 每百万输入 Token $1.25（2026-11-05 核实，以官方定价页为准）
        "price_output_per_1m": 4.25,  # 每百万输出 Token $4.25（同上）
        "price_cached_input_per_1m": 0.15,  # 每百万缓存输入 Token $0.15（同上）
    },
    "gpt": {
        "model": None,  # 未来主控，待接入，模型名按官方文档填写
        "enabled": False,
        "base_url": None,
        "price_input_per_1m": None,
        "price_output_per_1m": None,
        "price_cached_input_per_1m": None,
    },
    "gemini": {
        "model": None,  # 待接入
        "enabled": False,
        "base_url": None,
        "price_input_per_1m": None,
        "price_output_per_1m": None,
        "price_cached_input_per_1m": None,
    },
    "grok": {
        "model": None,  # 待接入
        "enabled": False,
        "base_url": None,
        "price_input_per_1m": None,
        "price_output_per_1m": None,
        "price_cached_input_per_1m": None,
    },
}

# DEFAULT_MODEL（默认模型）：没有特别指定时用哪个模型。
DEFAULT_MODEL = "muse"

# CONTROLLER（主控模型）：负责理解任务、制定计划、协调其他模型。
# 未来计划由 GPT 担任主控，但这只是一个“角色名”，不是写死在代码里的控制逻辑。
# 以后想换主控（例如换成 Muse），只改这一行，其他模块不用重写。
CONTROLLER = "gpt"  # 未来主控，本次不接入 GPT，仅作为长期方向

# ---------- 预算配置 ----------
MONTHLY_BUDGET_USD = 5.0  # 每月预算：5 美元（初期参考值）
BUDGET_WARN_RATIO = 0.8  # 达到预算 80%（$4）时警告
# 提醒：这是 Agent 自身的限制，不代表服务商账单的绝对上限，
#       还要关注 Meta 官方后台的消费情况。

# Token / 成本记录开关
RECORD_USAGE = True  # 是否记录每次调用的 Token 和成本


def get_api_key(provider: str):
    """从环境变量读取 API Key（接口密钥）。

    参数 provider：模型名，例如 "muse" 对应环境变量 MUSE_API_KEY。
    原理：只用 os.environ.get（读环境变量），不用 dotenv，避免多一个依赖。
    绝不把 Key 写进代码里。
    """
    env_name = f"{provider.upper()}_API_KEY"
    return os.environ.get(env_name)


def get_model_status():
    """返回每个模型的状态：是否启用、有无 Key、模型名。

    供 main.py 启动时检查和展示，不做任何真实调用。
    """
    status = {}
    for name, cfg in MODEL_CONFIG.items():
        status[name] = {
            "model": cfg["model"],
            "enabled": cfg["enabled"],
            "has_api_key": bool(get_api_key(name)),
        }
    return status


class BudgetTracker:
    """预算跟踪器（简单版，不要过度设计）。

    作用：记录本月累计成本，达 80% 警告，达 100% 拒绝后续付费调用。
    原理：累计成本存在本地文件 logs/budget_usage.json（JSON 格式），
          这个文件会被 .gitignore 忽略，不上传到 GitHub。
    注意：cost 为 None（模型没返回 usage，成本未知）时，不计入累计，不编造。
    """

    def __init__(self, file_path=None):
        # 默认存在项目根目录下的 logs/budget_usage.json
        self.file_path = Path(file_path) if file_path else Path(__file__).parent / "logs" / "budget_usage.json"
        self.total_cost_usd = 0.0
        self._load()

    def _load(self):
        """从本地文件读出之前累计的成本，文件不存在就从 0 开始。"""
        try:
            if self.file_path.exists():
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                self.total_cost_usd = float(data.get("total_cost_usd", 0.0))
        except (ValueError, OSError):
            self.total_cost_usd = 0.0

    def _save(self):
        """把累计成本写回本地文件。"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps({"total_cost_usd": round(self.total_cost_usd, 6)}, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_cost(self, cost_usd):
        """记一笔成本。cost 为 None 时不记，返回当前累计。"""
        if cost_usd is None:
            return self.total_cost_usd
        self.total_cost_usd += float(cost_usd)
        self._save()
        return self.total_cost_usd

    def warn_threshold_usd(self):
        """警告线：预算的 80%，即 $4。"""
        return MONTHLY_BUDGET_USD * BUDGET_WARN_RATIO

    def is_over_budget(self):
        """是否已达到或超过预算上限。"""
        return self.total_cost_usd >= MONTHLY_BUDGET_USD

    def is_over_warn(self):
        """是否已达到警告线。"""
        return self.total_cost_usd >= self.warn_threshold_usd()

    def can_call(self):
        """是否允许继续付费调用：没超预算才能调用，超了要等用户确认。"""
        return not self.is_over_budget()
