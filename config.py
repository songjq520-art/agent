# config.py（项目配置模块）
# 作用：所有配置集中在这里，不要把配置分散到各个文件里。
# 原理：其他模块（main.py、router.py、providers/muse.py）都从这里读配置，
#       以后改模型、改预算、换主控，只改这一个文件。
# 关系：config.py 是“总开关”，main.py 启动时先读它，router.py 按它分配模型。

import json
import os
from datetime import datetime
from pathlib import Path


def load_env_file(env_path=None):
    """读取 .env 文件（环境变量文件），只用 Python 自带库，不加 dotenv 依赖。

    参数 env_path：.env 文件路径，不传时默认读 config.py 同目录下的 .env。
    解析规则：忽略空行和以 # 开头的注释行；支持可选的 "export " 前缀；
    按第一个 = 分割键和值；值去掉首尾空格，并去掉一层单引号或双引号；
    用 os.environ.setdefault 写入，不覆盖已有的环境变量。
    文件不存在时什么也不做，不报错。
    """
    path = Path(env_path) if env_path else Path(__file__).parent / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


# 程序启动时自动读取项目根目录的 .env 文件（如果有的话）。
load_env_file()

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

    参数 provider：模型名。Muse 优先读官方变量名 MODEL_API_KEY，
    没有再读旧名 MUSE_API_KEY（兼容用）；其他模型仍按 {PROVIDER}_API_KEY 读。
    原理：只用 os.environ.get（读环境变量），绝不把 Key 写进代码里。
    """
    if provider == "muse":
        # 官方变量名优先，旧名兼容：两个都填时以 MODEL_API_KEY 为准。
        return os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY")
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
          文件里还有 month（月份，YYYY-MM）字段，跨月时自动重置为 0。
    注意：cost 为 None（模型没返回 usage，成本未知）时，不计入累计，不编造。
    """

    def __init__(self, file_path=None):
        # 默认存在项目根目录下的 logs/budget_usage.json
        self.file_path = Path(file_path) if file_path else Path(__file__).parent / "logs" / "budget_usage.json"
        self.total_cost_usd = 0.0
        self.month = self.current_month()
        self._load()

    def current_month(self):
        """当前月份，格式 YYYY-MM，例如 2026-11。用于每月预算重置。"""
        return datetime.now().strftime("%Y-%m")

    def _load(self):
        """从本地文件读出之前累计的成本，文件不存在就从 0 开始。

        每月重置规则：文件里有 month 字段且与当前月份不同时，
        把累计成本重置为 0 并更新月份后保存；文件没有 month 字段
        （旧文件）时，保留原累计值，只补上当前月份，不静默清零。
        """
        try:
            if self.file_path.exists():
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                self.total_cost_usd = float(data.get("total_cost_usd", 0.0))
                old_month = data.get("month")
                if old_month is None:
                    # 旧文件：保留累计值，只补月份。
                    self.month = self.current_month()
                    self._save()
                elif old_month != self.current_month():
                    # 跨月了：重置为 0，月份更新为当前月。
                    self.total_cost_usd = 0.0
                    self.month = self.current_month()
                    self._save()
                else:
                    self.month = old_month
        except (ValueError, OSError):
            self.total_cost_usd = 0.0
            self.month = self.current_month()

    def _save(self):
        """把累计成本和月份写回本地文件。"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(
                {"month": self.month, "total_cost_usd": round(self.total_cost_usd, 6)},
                ensure_ascii=False,
            ),
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
