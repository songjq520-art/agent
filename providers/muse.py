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
#   价格（Standard mode，标准模式）以官方定价页为准，2026-11-05 核实为
#   每百万 Token 普通输入 $1.25、缓存输入 $0.15、输出 $4.25。
#   Contributor 版价格不同，本项目不用。

import requests

from config import MAX_OUTPUT_TOKENS, MODEL_CONFIG


class MuseProvider:
    """Muse 模型提供者（统一接口）。

    统一接口的意思：chat(prompt) 这个方法名和返回格式是固定的，
    以后 GPT、Gemini、Grok 也按这个格式写，调用方不用改代码。
    """

    def __init__(self, api_key=None, budget_tracker=None, model=None):
        # api_key：从环境变量读进来，不要在这里写死真实 Key。
        # budget_tracker：预算跟踪器（config.py 里的 BudgetTracker），用来记账；
        #        超预算、成本未知时只提醒、不拦截（只有预算文件损坏时才拦）。
        # model：想用的 Muse 模型名，不传时用配置里的默认模型 muse-spark-1.3；
        #        目前只有这一个模型，传别的名字时直接报错。
        cfg = MODEL_CONFIG["muse"]
        self.model = model or cfg["model"]
        # 只接受配置里的模型名，传错名字时直接报错，
        # 不要拿 muse-spark-1.3 的价格去算一个不存在的模型的账。
        if self.model != cfg["model"]:
            raise ValueError(
                f"不支持的 Muse 模型：{self.model}，可选：{cfg['model']}。"
            )
        self.base_url = cfg["base_url"]
        self.api_key = api_key
        self.budget_tracker = budget_tracker
        self.price_input = cfg["price_input_per_1m"]
        self.price_output = cfg["price_output_per_1m"]
        self.price_cached_input = cfg["price_cached_input_per_1m"]

    def chat(self, prompt):
        """向 Muse 提一个问题（单次问答），返回字典。

        返回格式：{"text": 回答文本, "usage": Token 使用量或 None,
                  "cost_usd": 成本（美元）或 None, "model": 模型名}
        本方法是 chat_messages() 的单次问答包装：只发一条 user 消息，
        行为与原来完全一致。多轮对话请用 chat_messages(messages)。
        其余行为（无 Key、预算文件损坏、usage 缺失）见 chat_messages()。
        """
        return self.chat_messages([{"role": "user", "content": prompt}])

    def chat_messages(self, messages):
        """用完整消息历史向 Muse 提问（多轮对话），返回字典。

        参数 messages：消息历史列表，按时间顺序排列，每项为
        {"role": "user" 或 "assistant", "content": 文本}，
        应包含本轮及之前所有轮次的消息。
        返回格式与 chat() 相同。
        - 无 Key 时：抛出明确错误，不返回假成功。
        - 预算文件损坏时：抛出明确错误（原因见 status_reason），不偷偷调用。
          超预算、有未核对的未知成本时只提醒、不拦截（BOOS 定的新规矩：
          到 $5 提醒，永不自动停），不会在这里抛错。
        - 模型没返回 usage 时：cost_usd 为 None，绝不编造 Token 或成本，
          并标记成本未知、打印提醒，等人工核对，不拦后续调用。
        """
        # 0. 先验证消息历史格式：必须是非空列表，每项都有 role 和 content。
        if not messages or not isinstance(messages, list):
            raise ValueError("消息历史不能为空，应为消息列表。")
        for msg in messages:
            if (
                not isinstance(msg, dict)
                or "role" not in msg
                or "content" not in msg
            ):
                raise ValueError(
                    "消息历史格式错误，每项应为 {'role': 'user'/'assistant', 'content': 文本}。"
                )
        # 1. 先检查 Key，没有 Key 就明确说“未完成”，不要假装调用成功。
        if not self.api_key:
            raise RuntimeError(
                "未配置 API Key，真实连接未完成："
                "请在 .env 里填写 MODEL_API_KEY（或旧名 MUSE_API_KEY），"
                "或设置环境变量 MODEL_API_KEY 后再运行。"
            )

        # 2. 再检查预算：只有预算文件损坏时才拦（can_call() 只在损坏时
        #    返回 False）。超预算、有未核对的未知成本时只提醒、不拦截
        #    （BOOS 定的新规矩：到 $5 提醒，永不自动停），这里不抛错。
        #    损坏时用 status_reason() 给出具体原因。
        if self.budget_tracker is not None and not self.budget_tracker.can_call():
            reason = ""
            if hasattr(self.budget_tracker, "status_reason"):
                reason = self.budget_tracker.status_reason(provider="muse")
            raise RuntimeError(
                reason or "预算记录异常，付费调用暂不可用，请人工检查 logs/budget_usage.json 后再恢复。"
            )

        # 3. 发真实请求。timeout（超时）设为 60 秒，避免一直等下去。
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            # 直接发送完整消息历史（本轮及之前所有轮次），不只发最新问题。
            "messages": messages,
            # max_completion_tokens（最大输出 Token 数）：实际生效的输出上限，
            # 与调用前预估用的 MAX_OUTPUT_TOKENS 是同一个值（4096），
            # 所以预估里的输出量不是凭空假设。
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
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
        #    缓存 Token（cached_tokens）按缓存价 $0.15 计，普通输入按 $1.25 计；
        #    completion_tokens 里已包含 reasoning_tokens（推理 Token），它是输出的子集，
        #    不要重复加，只按 completion_tokens 计输出价 $4.25。
        #    注意：usage 缺失时不能当成零成本，要标记成本未知并提醒，等人工核对。
        usage = data.get("usage")
        cost_usd = None
        if not usage:
            if self.budget_tracker is not None:
                self.budget_tracker.mark_unknown_cost()
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = details.get("cached_tokens", 0) or 0
            # 安全保护：缓存 Token 数不能超过总输入 Token 数。
            cached_tokens = min(cached_tokens, prompt_tokens)
            cost_usd = (
                (prompt_tokens - cached_tokens) / 1_000_000 * self.price_input
                + cached_tokens / 1_000_000 * self.price_cached_input
                + completion_tokens / 1_000_000 * self.price_output
            )
            if self.budget_tracker is not None:
                self.budget_tracker.add_cost(cost_usd, model=self.model)

        return {
            "text": text,
            "usage": usage,
            "cost_usd": cost_usd,
            "model": self.model,
        }
