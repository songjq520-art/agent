# providers/muse.py（Muse 模型接口）
# 作用：真正调用 Muse API 的地方，是整个项目非常重要的部分。
# 原理：用 requests（HTTP 请求工具）向 Meta Model API 发 POST 请求，
#       请求格式与 OpenAI 兼容，认证用 Authorization: Bearer <API Key>。
# 关系：router.py 决定“交给 Muse”，main.py 拿到 MuseProvider 后调用它的 chat()；
#       以后其他模型也写一个同样有 chat(prompt) 方法的类，就能直接替换。
#
# 官方事实（2026-11-05 核实，来源 dev.meta.ai/docs/models）：
#   接口地址：https://api.meta.ai/v1
#   模型：muse-spark-1.3
#   端点：POST /v1/chat/completions（Chat Completions，聊天补全）
#   价格以官方定价页为准，2026-11-05 核实为每百万 Token 输入 $1.25、输出 $4.25、缓存输入 $0.15。

import requests

from config import MODEL_CONFIG


class MuseProvider:
    """Muse 模型提供者（统一接口）。

    统一接口的意思：chat(prompt) 这个方法名和返回格式是固定的，
    以后 GPT、Gemini、Grok 也按这个格式写，调用方不用改代码。
    """

    def __init__(self, api_key=None, budget_tracker=None):
        # api_key：从环境变量读进来，不要在这里写死真实 Key。
        # budget_tracker：预算跟踪器（config.py 里的 BudgetTracker），用来记账和拦截超预算调用。
        cfg = MODEL_CONFIG["muse"]
        self.model = cfg["model"]
        self.base_url = cfg["base_url"]
        self.api_key = api_key
        self.budget_tracker = budget_tracker
        self.price_input = cfg["price_input_per_1m"]
        self.price_output = cfg["price_output_per_1m"]

    def chat(self, prompt):
        """向 Muse 提一个问题，返回字典。

        返回格式：{"text": 回答文本, "usage": Token 使用量或 None,
                  "cost_usd": 成本（美元）或 None, "model": 模型名}
        - 无 Key 时：抛出明确错误，不返回假成功。
        - 超预算时：抛出明确错误，等待用户确认，不偷偷调用。
        - 模型没返回 usage 时：cost_usd 为 None，绝不编造 Token 或成本。
        """
        # 1. 先检查 Key，没有 Key 就明确说“未完成”，不要假装调用成功。
        if not self.api_key:
            raise RuntimeError(
                "未配置 API Key，真实连接未完成："
                "请在 .env 里填写 MUSE_API_KEY，或设置环境变量 MUSE_API_KEY 后再运行。"
            )

        # 2. 再检查预算，超了就停下来等用户确认。
        if self.budget_tracker is not None and not self.budget_tracker.can_call():
            raise RuntimeError(
                "已达到每月预算上限（$5），已暂停后续付费调用，请确认后再继续。"
            )

        # 3. 发真实请求。timeout（超时）设为 60 秒，避免一直等下去。
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            # 网络或接口报错时，直接把错误抛出去，不返回假回答。
            raise RuntimeError(f"Muse API 调用失败：{exc}") from exc

        data = response.json()

        # 4. 解析回答。choices[0].message.content 是回答文本。
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Muse API 返回格式异常，无法解析回答：{data}") from exc

        # 5. 解析 Token 使用量。只有模型返回了 usage 才计算成本，否则 cost 为 None。
        usage = data.get("usage")
        cost_usd = None
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            cost_usd = (
                prompt_tokens / 1_000_000 * self.price_input
                + completion_tokens / 1_000_000 * self.price_output
            )
            if self.budget_tracker is not None:
                self.budget_tracker.add_cost(cost_usd)

        return {
            "text": text,
            "usage": usage,
            "cost_usd": cost_usd,
            "model": self.model,
        }
