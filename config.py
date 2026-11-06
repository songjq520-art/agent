# config.py（项目配置模块）
# 作用：所有配置集中在这里，不要把配置分散到各个文件里。
# 原理：其他模块（main.py、router.py、providers/muse.py）都从这里读配置，
#       以后改模型、改预算、换主控，只改这一个文件。
# 关系：config.py 是“总开关”，main.py 启动时先读它，router.py 按它分配模型。

import json
import math
import os
import re
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
        "enabled": True,  # 已启用，作为工作模型保留
        "base_url": "https://api.meta.ai/v1",  # Meta Model API 官方地址
        # 以下价格是 Muse Spark 1.3 Standard mode（标准模式）的官方价格
        # （2026-11-05 核实，以官方定价页为准）。
        # 注意：Contributor 版（贡献者模式）价格不同（输入 $0.10、缓存输入 $0.002、
        # 输出 $0.20），本项目不用 Contributor，只用 Standard mode。
        "price_input_per_1m": 1.25,   # 每百万普通输入 Token $1.25
        "price_output_per_1m": 4.25,  # 每百万输出 Token $4.25
        "price_cached_input_per_1m": 0.15,  # 每百万缓存输入 Token $0.15
    },
    "deepseek": {
        "model": "deepseek-flash",  # DeepSeek V4.1 Flash，第一任主控，默认模型
        "enabled": True,  # 已启用，第一任主控
        "base_url": "https://api.deepseek.com",  # DeepSeek 官方地址（OpenAI 兼容格式）
        # 以下价格是 DeepSeek 闲时（Off-peak）价
        # （2026-11-06 核实，以 DeepSeek 官方定价页为准）。
        # 注意：高峰时段（北京时间周一至周五 09:00–12:00、14:00–18:00）
        # 价格是闲时的 2 倍，本地按闲时价记账，高峰时实际成本可能高于
        # 本地记录，实际费用以 DeepSeek 官方后台为准。
        "price_input_per_1m": 0.15,   # 每百万普通输入 Token $0.15（闲时）
        "price_output_per_1m": 0.60,  # 每百万输出 Token $0.60（闲时）
        "price_cached_input_per_1m": 0.003,  # 每百万缓存输入 Token $0.003（闲时）
        # 切换到 Pro 模型时用的名字和价格（约 Flash 的 4 倍，闲时价）。
        # 切换方法：建 DeepSeekProvider 时传 model="deepseek-v4-pro"，
        # 或在 config.py 里把上面 "model" 一行改成 "deepseek-v4-pro"。
        "model_pro": "deepseek-v4-pro",  # DeepSeek V4 Pro
        "price_input_per_1m_pro": 0.66,   # 每百万普通输入 Token $0.66（闲时）
        "price_output_per_1m_pro": 1.98,  # 每百万输出 Token $1.98（闲时）
        "price_cached_input_per_1m_pro": 0.022,  # 每百万缓存输入 Token $0.022（闲时）
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
DEFAULT_MODEL = "deepseek"

# CONTROLLER（主控模型）：负责理解任务、制定计划、协调其他模型。
# DeepSeek 是第一任主控候选（见 PROJECT_PRINCIPLES.md），负责任务分配、
# 模型调度、信息传递、总结。这只是一个“角色名”，不是写死在代码里的控制逻辑。
# 以后想换主控（例如换成 GPT），只改这一行，其他模块不用重写。
# GPT 是远期主控候选，目前不接入。
CONTROLLER = "deepseek"  # 第一任主控

# ---------- 预算配置 ----------
MONTHLY_BUDGET_USD = 5.0  # 每月预算：5 美元（初期参考值）
BUDGET_WARN_RATIO = 0.8  # 达到预算 80%（$4）时警告
# MAX_OUTPUT_TOKENS（输出上限，单位 Token）：一次回答最多输出 2048 Token。
# 这个值既是调用前费用预估用的输出量，也是请求里实际传给 API 的
# max_completion_tokens 上限；输出成本上限 = 2048/1e6*4.25 美元。
# 注意：预估仍是粗略估算，不能保证单次请求绝对不超支。
MAX_OUTPUT_TOKENS = 2048
# 提醒：这是 Agent 自身的限制，不代表服务商账单的绝对上限，
#       还要关注各模型官方后台的消费情况（Muse 看 Meta 后台，DeepSeek 看 DeepSeek 后台）。

# Token / 成本记录开关
RECORD_USAGE = True  # 是否记录每次调用的 Token 和成本


def get_api_key(provider: str):
    """从环境变量读取 API Key（接口密钥）。

    参数 provider：模型名。Muse 优先读官方变量名 MODEL_API_KEY，
    没有再读旧名 MUSE_API_KEY（兼容用）；DeepSeek 读官方变量名
    DEEPSEEK_API_KEY（DeepSeek 官方文档示例用的就是这个名字）；
    其他模型仍按 {PROVIDER}_API_KEY 读。
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
          文件里还有 month（月份，YYYY-MM）、unknown_cost（成本未知标记）
          和 cost_by_model（按模型小计，例如 {"deepseek-flash": 0.01}）字段，
          跨月时累计和小计一起自动重置为 0，但 unknown_cost 不会因跨月自动清除——
          只有人工核对过官方后台后才能清除（见 clear_unknown_cost）。
          封顶只看总数 total_cost_usd，小计只是分模型的明细，方便看钱花在哪个模型上。
    注意：cost 为 None（模型没返回 usage，成本未知）时，不计入累计，不编造，
          但会把 unknown_cost 标记为 True，暂停后续付费调用，等人工核对。
    安全规则：预算文件损坏、无法解析或关键字段缺失时，不会自动初始化为
          正常可调用状态，而是进入安全暂停状态（corrupted=True），
          原损坏文件会被保留、不被覆盖，只能人工检查修复后才能恢复。
    限制：本地预算只能拦截“已累计超支”和“预估明显超支”，
          不能保证单次请求绝对不超支，实际费用以各模型官方后台为准
          （Muse 看 Meta 后台，DeepSeek 看 DeepSeek 后台）。
    """

    def __init__(self, file_path=None):
        # 默认存在项目根目录下的 logs/budget_usage.json
        self.file_path = Path(file_path) if file_path else Path(__file__).parent / "logs" / "budget_usage.json"
        self.total_cost_usd = 0.0
        self.month = self.current_month()
        self.unknown_cost = False  # 是否有一次调用的成本未知（没返回 usage）
        self.cost_by_model = {}  # 按模型小计：{"deepseek-flash": 0.01}，只作明细，封顶只看总数
        self.corrupted = False  # 预算文件是否损坏/异常，损坏时暂停付费调用
        self.corruption_reason = ""  # 损坏原因，中文简短说明
        self._load()

    def current_month(self):
        """当前月份，格式 YYYY-MM，例如 2026-11。用于每月预算重置。"""
        return datetime.now().strftime("%Y-%m")

    def _check_month_rollover(self):
        """检查是否跨过自然月：是就只重置当月累计成本，并保存。

        安全规则：跨月只重置 total_cost_usd 和 month，绝对不清除
        unknown_cost（成本未知标记）。例如 11 月发生了一笔无法确定成本的
        调用，12 月不能因为月份变化就自动解除暂停——必须等人工核对过
        官方后台的实际消费后，才能调用 clear_unknown_cost() 清除。
        按模型小计（cost_by_model）也是按月算的，跨月时和总数一起清零。
        每次判断预算或记账前都先调用它，所以程序一直运行、跨过自然月时
        也能重置，不只是在启动时检查一次。
        安全规则：预算文件损坏时直接返回，不重置、不保存、不清除任何状态。
        """
        if self.corrupted:
            return
        if self.month != self.current_month():
            self.total_cost_usd = 0.0
            self.cost_by_model = {}
            # 注意：这里故意不碰 self.unknown_cost，保留未核对的暂停状态。
            self.month = self.current_month()
            self._save()

    def _enter_corrupted(self, reason, total_cost_usd=0.0, month=None):
        """进入预算文件损坏的安全暂停状态。

        参数 reason：损坏原因（中文）；total_cost_usd：能读到的累计值，
        读不到时为 0，但不要把它当成可信值去调用；month：能读到的合法月份，
        读不到时为当前月，仅作显示。
        安全规则：只改内存状态，绝对不调用 _save() 覆盖原损坏文件；
        unknown_cost 设为 True，暂停后续付费调用。
        """
        self.corrupted = True
        self.corruption_reason = reason
        self.total_cost_usd = float(total_cost_usd)
        self.month = month or self.current_month()
        self.unknown_cost = True

    def _load(self):
        """从本地文件读出之前累计的成本，文件不存在就从 0 开始。

        关键字段（三个缺一不可，类型不对也算损坏）：
          month：YYYY-MM 格式字符串，月份 01-12；
          total_cost_usd：数字（int/float，不能是 bool/字符串）；
          unknown_cost：bool。
        cost_by_model（按模型小计）是可选字段：旧文件没有它时不算损坏，
        默认空字典；但如果写了却不是合法的“模型名 → 非负有限数字”字典，
        也算损坏，进入安全暂停状态。
        损坏、无法解析或关键字段缺失时，进入安全暂停状态（corrupted=True），
        原文件会被保留、不被覆盖，只能人工检查修复后才能恢复。
        旧文件缺少 month 字段时，也按关键字段缺失处理，不再“保留累计、
        只补月份并保存”。
        """
        if not self.file_path.exists():
            # 文件不存在是正常情况（第一次运行），从 0 开始。
            return
        try:
            raw_text = self.file_path.read_text(encoding="utf-8")
        except OSError:
            self._enter_corrupted("预算文件无法读取")
            return
        try:
            data = json.loads(raw_text)
        except ValueError:
            self._enter_corrupted("预算文件无法解析为 JSON")
            return
        if not isinstance(data, dict):
            self._enter_corrupted("预算文件内容不是 JSON 对象")
            return
        # 先保留能读到的值（仅作显示，不作为可信累计去调用）。
        raw_total = data.get("total_cost_usd")
        raw_month = data.get("month")
        # total_cost_usd 必须是有限数字：拒绝 bool、NaN、Infinity 和负数。
        total_is_valid = (
            isinstance(raw_total, (int, float))
            and not isinstance(raw_total, bool)
            and math.isfinite(raw_total)
            and raw_total >= 0
        )
        keep_total = raw_total if total_is_valid else 0.0
        keep_month = raw_month if isinstance(raw_month, str) and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", raw_month) else None
        # 逐个检查关键字段，缺失或类型不对都算损坏。
        problems = []
        if not keep_month:
            problems.append("缺少 month 字段")
        if not total_is_valid:
            problems.append("total_cost_usd 非法（必须是大于或等于 0 的有限数字）")
        if not isinstance(data.get("unknown_cost"), bool):
            problems.append("缺少 unknown_cost 字段")
        # cost_by_model 是可选字段：没有时默认空字典；写了但格式不对也算损坏。
        raw_breakdown = data.get("cost_by_model", {})
        breakdown = {}
        if not isinstance(raw_breakdown, dict):
            problems.append("cost_by_model 不是 JSON 对象")
        else:
            for name, value in raw_breakdown.items():
                if not isinstance(name, str) or not (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                ):
                    problems.append("cost_by_model 里有非法值")
                    break
            else:
                breakdown = {name: float(value) for name, value in raw_breakdown.items()}
        if problems:
            self._enter_corrupted(
                "预算文件" + "、".join(problems) + "或字段类型不对",
                total_cost_usd=keep_total,
                month=keep_month,
            )
            return
        self.total_cost_usd = float(raw_total)
        self.unknown_cost = bool(data["unknown_cost"])
        self.month = raw_month
        self.cost_by_model = breakdown
        self._check_month_rollover()

    def _save(self):
        """把累计成本、月份和未知成本标记写回本地文件。

        安全保护：预算文件损坏时直接返回、不写文件，保证原损坏文件不被覆盖。
        """
        if self.corrupted:
            return
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(
                {
                    "month": self.month,
                    "total_cost_usd": round(self.total_cost_usd, 6),
                    "unknown_cost": self.unknown_cost,
                    "cost_by_model": {
                        name: round(value, 6) for name, value in self.cost_by_model.items()
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def mark_unknown_cost(self):
        """标记一次调用成本未知：暂停后续付费调用，等人工核对后再清除。

        预算文件损坏时不写文件（_save() 本身也会拦截），直接返回。
        """
        if self.corrupted:
            return
        self._check_month_rollover()
        self.unknown_cost = True
        self._save()

    def clear_unknown_cost(self):
        """清除未知成本标记：只有人工核对过官方后台的实际消费后才能调用。

        注意：这只是清除“暂停”标记，不会补记那次未知调用的实际成本，
        那次成本请以官方后台为准，人工记下。
        安全规则：预算文件损坏时不允许通过代码清除，必须先人工检查并修复
        预算文件，否则会把暂停状态悄悄解除。
        """
        if self.corrupted:
            raise RuntimeError("预算文件已损坏，不能通过代码清除，请先人工检查并修复预算文件")
        self._check_month_rollover()
        self.unknown_cost = False
        self._save()

    def add_cost(self, cost_usd, model=None):
        """记一笔成本。cost 为 None 时不记，返回当前累计。

        参数 model：这次调用用的模型名（例如 "deepseek-flash"），
        传了就同时记到该模型的小计里；不传时只记总数（兼容旧调用）。
        注意：封顶只看总数，小计只是明细；总数和小计之和可能不一致
        （例如直接调 add_cost 没传模型时），以总数为准。
        预算文件损坏时不记账、不写文件，直接返回当前累计（仅作显示）。
        """
        if self.corrupted:
            return self.total_cost_usd
        self._check_month_rollover()
        if cost_usd is None:
            return self.total_cost_usd
        amount = float(cost_usd)
        self.total_cost_usd += amount
        if model:
            self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + amount
        self._save()
        return self.total_cost_usd

    def cost_for_model(self, model):
        """某模型本月累计成本（美元），没记过为 0。"""
        self._check_month_rollover()
        return self.cost_by_model.get(model, 0.0)

    def model_breakdown(self):
        """本月按模型分的成本小计，返回一份副本（改它不影响内部记录）。"""
        self._check_month_rollover()
        return dict(self.cost_by_model)

    def warn_threshold_usd(self):
        """警告线：预算的 80%，即 $4。"""
        return MONTHLY_BUDGET_USD * BUDGET_WARN_RATIO

    def is_over_budget(self):
        """是否已达到或超过预算上限。"""
        self._check_month_rollover()
        return self.total_cost_usd >= MONTHLY_BUDGET_USD

    def is_over_warn(self):
        """是否已达到警告线。"""
        self._check_month_rollover()
        return self.total_cost_usd >= self.warn_threshold_usd()

    def remaining_usd(self):
        """本月剩余预算（美元），已累计超支时为负数。"""
        self._check_month_rollover()
        return MONTHLY_BUDGET_USD - self.total_cost_usd

    def estimate_input_tokens(self, prompt_text):
        """粗略估算输入 Token 数（不是真实分词，只是调用前预估用）。

        中文低估问题：原来按 len(prompt)/4 估算，10000 个汉字会被估成
        2500 Token，实际可能接近 10000 Token，成本低估约 4 倍。
        现在分开算：ASCII 字符（英文、数字、标点，ord<128）按 4 个字符
        1 个 Token；非 ASCII 字符（中文等）按每个字符 1.5 个 Token 向上取整，
        中文按 1.5 倍留余量，宁可高估、不要低估。
        """
        text = str(prompt_text or "")
        ascii_count = sum(1 for ch in text if ord(ch) < 128)
        non_ascii_count = len(text) - ascii_count
        return math.ceil(ascii_count / 4 + non_ascii_count * 1.5)

    def estimate_cost(self, prompt_text, provider="muse", model=None):
        """调用前预估本次成本（美元），只用于拦截明显超支的请求。

        预估方法：输入 Token 用 estimate_input_tokens() 估算；
        输出按 MAX_OUTPUT_TOKENS（2048 Token）估算，这个值同时是请求里
        实际传给 API 的 max_completion_tokens 上限，所以输出量不再是
        凭空假设；缓存按 0 算（最保守，即全部输入按普通价算）。
        参数 provider：按哪个模型的价格估算；model：DeepSeek 时可指定
        "deepseek-v4-pro"（按 Pro 价算），不指定时按 deepseek-flash 算。
        默认 provider="muse"：保持已有行为和已有测试不变，调用方应显式
        传自己的 provider，不要依赖默认值。
        注意：DeepSeek 按闲时价估算，高峰时段实际价格是闲时的 2 倍，
        预估可能低于高峰时的实际成本，实际费用以官方后台为准。
        """
        prompt_tokens = self.estimate_input_tokens(prompt_text)
        cfg = MODEL_CONFIG.get(provider)
        if cfg is None:
            raise ValueError(f"未知的模型：{provider}")
        if provider == "deepseek" and model == cfg.get("model_pro"):
            price_in = cfg["price_input_per_1m_pro"]
            price_out = cfg["price_output_per_1m_pro"]
        else:
            price_in = cfg["price_input_per_1m"]
            price_out = cfg["price_output_per_1m"]
        if price_in is None or price_out is None:
            raise RuntimeError(f"模型 {provider} 未配置价格，无法预估成本")
        return (
            prompt_tokens / 1_000_000 * price_in
            + MAX_OUTPUT_TOKENS / 1_000_000 * price_out
        )

    def can_afford_estimated(self, prompt_text, provider="muse", model=None):
        """预估成本是否在剩余预算内。返回 False 时，调用方应在请求前拦截。"""
        return self.estimate_cost(prompt_text, provider=provider, model=model) <= self.remaining_usd()

    def status_reason(self, provider="muse"):
        """不能发起付费调用时的中文原因，正常时返回空字符串。

        区分三种情况：预算文件损坏、未知成本未核对、已超预算。
        参数 provider：用于在“未知成本”提示里写对官方后台（Muse 看 Meta
        后台，DeepSeek 看 DeepSeek 后台）；默认 "muse"，保持已有行为不变。
        """
        backend = {"muse": "Meta", "deepseek": "DeepSeek"}.get(provider, "官方")
        if self.corrupted:
            return (
                f"预算记录异常：{self.corruption_reason}，已暂停付费调用，"
                "请人工检查 logs/budget_usage.json 后再恢复"
            )
        if self.unknown_cost:
            return f"有未核对的未知成本，已暂停后续付费调用，请核对{backend}官方后台后再清除标记。"
        if self.is_over_budget():
            return "已达到每月预算上限（$5），已暂停后续付费调用，请确认后再继续。"
        return ""

    def can_call(self):
        """是否允许继续付费调用：预算文件正常、没超预算、且没有未核对的
        未知成本时才能调用。损坏时一律返回 False。"""
        self._check_month_rollover()
        if self.corrupted:
            return False
        return not self.is_over_budget() and not self.unknown_cost
