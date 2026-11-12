# providers/deepseek.py（DeepSeek 模型接口）
# 作用：真正调用 DeepSeek API 的地方，是第一任主控的接口。
# 原理：用 requests（HTTP 请求工具）向 DeepSeek API 发 POST 请求，
#       请求格式与 OpenAI 兼容，认证用 Authorization: Bearer <API Key>。
# 关系：router.py 决定“交给 DeepSeek”，main.py 拿到 DeepSeekProvider 后调用它的 chat()；
#       它和 providers/muse.py 是同样的统一接口（都有 chat(prompt) 方法），
#       调用方不用改代码，只改配置就能换模型。
#
# 官方事实（2026-11-06 核实，来源 api-docs.deepseek.com）：
#   接口地址：https://api.deepseek.com
#   模型：deepseek-flash（DeepSeek V4.1 Flash，默认，第一任主控）、
#         deepseek-v4-pro（DeepSeek V4 Pro，切换用，约 Flash 的 4 倍价格）
#   端点：POST /chat/completions（Chat Completions，聊天补全）
#   注意：DeepSeek 官方文档的地址里没有 /v1 这一段，
#         和 Muse 的 https://api.meta.ai/v1 写法不一样，不要照抄。
#   价格（闲时 Off-peak，以官方定价页为准，2026-11-06 核实）：
#     Flash：每百万 Token 普通输入 $0.15、缓存输入 $0.003、输出 $0.60；
#     Pro：每百万 Token 普通输入 $0.66、缓存输入 $0.022、输出 $1.98；
#     高峰时段（北京时间周一至周五 09:00–12:00、14:00–18:00）价格是闲时的 2 倍。
#   旧模型名 deepseek-chat / deepseek-reasoner 已于 2026-07-24 退役，不要再用。
#
# 工具调用（Tool Calls，2026-11-11 核实，来源 DeepSeek 官方文档）：
#   deepseek-flash 支持工具调用；请求带 tools 时，之后每次请求都必须把
#   之前所有轮次 assistant 消息的 reasoning_content 原样带回（全量回传），
#   哪怕那一轮没调工具也不例外，否则接口报 400。带 tools 的请求
#   （chat_with_tools）思考开着时，发请求前就要检查、缺了直接报错，
#   不补空字符串（那条路没经过真机核实，不能拿来赌，见
#   _check_reasoning_complete）。
#   不带 tools 的最后回答（chat_messages）传 thinking="enabled" 时，
#   接口同样要求历史里带 tool_calls 的 assistant 消息带
#   reasoning_content，缺了报 400；但这条路 2026-11-13 真机核实过：
#   工具那几轮思考关着、模型没产生 reasoning_content，补空字符串
#   接口接受（空字符串只表示“本轮没有推理内容”），见
#   _fill_missing_reasoning。不带 tool_calls 的纯文本 assistant 消息
#   不需要补，接口不查它。
#   思考开关通过请求体顶层 "thinking": {"type": "disabled"/"enabled"} 传，
#   不是 extra_body 专属写法；关掉思考时没有 reasoning_content 可回传，
#   也就不需要回传检查。
# 换模型时的字段差异（2026-11-11 核实）：Kimi 的思考字段也是
#   reasoning_content，不是 reasoning；见下面的 PROVIDER_THINKING_FIELDS。

import requests

from config import MAX_OUTPUT_TOKENS, MODEL_CONFIG


# 各模型思考/推理字段对照表：换模型时只改这里，不改循环逻辑。
#   thinking_field：请求体里开关思考的顶层字段名；
#   reasoning_field：assistant 消息里要原样回传的推理字段名。
# DeepSeek 和 Kimi 都是 reasoning_content（Kimi 官方文档同款字段名，
# 不是网上流传的 reasoning——2026-11-11 已核实，别写错）。
PROVIDER_THINKING_FIELDS = {
    "deepseek": {"thinking_field": "thinking", "reasoning_field": "reasoning_content"},
    "kimi": {"thinking_field": "thinking", "reasoning_field": "reasoning_content"},
}


class DeepSeekProvider:
    """DeepSeek 模型提供者（统一接口）。

    统一接口的意思：chat(prompt) 这个方法名和返回格式是固定的，
    和 MuseProvider 一样，调用方不用改代码。
    """

    def __init__(self, api_key=None, budget_tracker=None, model=None):
        # api_key：从环境变量读进来，不要在这里写死真实 Key。
        # budget_tracker：预算跟踪器（config.py 里的 BudgetTracker），用来记账；
        #        超预算、成本未知时只提醒、不拦截（只有预算文件损坏时才拦）。
        # model：想用的 DeepSeek 模型名，不传时用配置里的默认模型 deepseek-flash；
        #        传 "deepseek-v4-pro" 时切换到 Pro 模型，价格也跟着换成 Pro 价。
        cfg = MODEL_CONFIG["deepseek"]
        self.model = model or cfg["model"]
        # 只接受配置里的两个模型名，传错名字时直接报错，
        # 不要拿 Flash 的价格去算一个不存在的模型的账。
        if self.model not in (cfg["model"], cfg["model_pro"]):
            raise ValueError(
                f"不支持的 DeepSeek 模型：{self.model}，"
                f"可选：{cfg['model']}、{cfg['model_pro']}。"
            )
        self.base_url = cfg["base_url"]
        self.api_key = api_key
        self.budget_tracker = budget_tracker
        # Pro 模型用 Pro 的价格，否则用 Flash 的价格。
        if self.model == cfg["model_pro"]:
            self.price_input = cfg["price_input_per_1m_pro"]
            self.price_output = cfg["price_output_per_1m_pro"]
            self.price_cached_input = cfg["price_cached_input_per_1m_pro"]
        else:
            self.price_input = cfg["price_input_per_1m"]
            self.price_output = cfg["price_output_per_1m"]
            self.price_cached_input = cfg["price_cached_input_per_1m"]

    def chat(self, prompt, ledger="chat", task_id=None, trace_id=None, thinking=None):
        """向 DeepSeek 提一个问题（单次问答），返回字典。

        返回格式：{"text": 回答文本, "usage": Token 使用量或 None,
                  "cost_usd": 成本（美元）或 None, "model": 模型名}
        本方法是 chat_messages() 的单次问答包装：只发一条 user 消息，
        行为与原来完全一致。多轮对话请用 chat_messages(messages)。
        其余行为（无 Key、预算文件损坏、usage 缺失）见 chat_messages()。
        参数 ledger / task_id / trace_id：记账用（见 chat_messages）。
        参数 thinking：不传时不发思考开关（和原来一样，思考默认开）；
        传 "enabled" / "disabled" 时通过请求体顶层 "thinking" 字段
        显式开关——工具循环里：中间调工具那几轮传 "disabled"（只做路由
        判断），最后回答那轮传 "enabled"（2026-11-13 改，原来最后回答
        也关着）。注意：最后回答传 "enabled" 时，历史里带 tool_calls 的
        assistant 消息必须带 reasoning_content，否则接口报 400——工具那
        几轮思考关着、模型没产生 reasoning_content，chat_messages 会
        给这类消息补空字符串（2026-11-13 真机核实可过），见
        _fill_missing_reasoning。
        """
        return self.chat_messages(
            [{"role": "user", "content": prompt}],
            ledger=ledger,
            task_id=task_id,
            trace_id=trace_id,
            thinking=thinking,
        )

    def chat_messages(self, messages, ledger="chat", task_id=None, trace_id=None, thinking=None):
        """用完整消息历史向 DeepSeek 提问（多轮对话），返回字典。

        参数 messages：消息历史列表，按时间顺序排列，每项为
        {"role": "user" 或 "assistant", "content": 文本}，
        应包含本轮及之前所有轮次的消息。
        参数 ledger：记到哪个账本，"chat"（默认，平时聊天 + 记忆提取）
        或 "tool"（工具循环里的每一轮）；task_id / trace_id 用来把
        模型请求、工具请求、授权、执行结果和费用串起来查。
        参数 thinking：见 chat()——工具循环的最后一次回答传
        "enabled"（2026-11-13 改，原来传 "disabled"），中间调工具那几轮
        传 "disabled"；平时聊天不传（行为和原来一样）。
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
                "请在 .env 里填写 DEEPSEEK_API_KEY，"
                "或设置环境变量 DEEPSEEK_API_KEY 后再运行。"
            )

        # 2. 再检查预算：只有预算文件损坏时才拦（can_call() 只在损坏时
        #    返回 False）。超预算、有未核对的未知成本时只提醒、不拦截
        #    （BOOS 定的新规矩：到 $5 提醒，永不自动停），这里不抛错。
        #    损坏时用 status_reason() 给出具体原因。
        if self.budget_tracker is not None and not self.budget_tracker.can_call():
            reason = ""
            if hasattr(self.budget_tracker, "status_reason"):
                reason = self.budget_tracker.status_reason(provider="deepseek")
            raise RuntimeError(
                reason or "预算记录异常，付费调用暂不可用，请人工检查 logs/budget_usage.json 后再恢复。"
            )

        # 3. 发真实请求。timeout（超时）设为 60 秒，避免一直等下去。
        #    注意：DeepSeek 的端点是 /chat/completions，base_url 里没有 /v1，
        #    不要写成 /v1/chat/completions。
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            # 直接发送完整消息历史（本轮及之前所有轮次），不只发最新问题。
            # 思考开着时（"enabled"，或不传、接口默认开），带 tool_calls 的
            # assistant 消息缺 reasoning_content 会被接口报 400，见
            # _fill_missing_reasoning（2026-11-13 真机核实，补空字符串可过）。
            "messages": (
                self._fill_missing_reasoning(messages)
                if thinking != "disabled"
                else messages
            ),
            # max_tokens（最大输出 Token 数）：实际生效的输出上限，
            # 与调用前预估用的 MAX_OUTPUT_TOKENS 是同一个值（4096），
            # 所以预估里的输出量不是凭空假设。
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if thinking is not None:
            if thinking not in ("enabled", "disabled"):
                raise ValueError(
                    f"思考开关只认 'enabled' 或 'disabled'，不是 {thinking!r}。"
                )
            # 思考开关放请求体顶层（DeepSeek 官方写法）。
            payload["thinking"] = {"type": thinking}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            # 网络或接口报错时，直接把错误抛出去，不返回假回答。
            raise RuntimeError(f"DeepSeek API 调用失败：{exc}") from exc

        data = response.json()

        # 4. 解析回答。choices[0].message.content 是回答文本。
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"DeepSeek API 返回格式异常，无法解析回答：{data}") from exc

        # 5. 解析 Token 使用量。只有模型返回了 usage 才计算成本，否则 cost 为 None。
        #    缓存 Token（cached_tokens）按缓存价计，普通输入按普通价计；
        #    completion_tokens 里已包含推理 Token（如果有的话），它是输出的子集，
        #    不要重复加，只按 completion_tokens 计输出价。
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
                self.budget_tracker.add_cost(
                    cost_usd,
                    model=self.model,
                    ledger=ledger,
                    task_id=task_id,
                    trace_id=trace_id,
                    usage=usage,
                )

        return {
            "text": text,
            "usage": usage,
            "cost_usd": cost_usd,
            "model": self.model,
        }

    # ---------- 工具调用（Tool Calls）：给 tools/agent_loop.py 用 ----------
    # 规则只写在这里（适配层），循环本身不碰这些细节，换模型时只改这里。

    @staticmethod
    def _fill_missing_reasoning(messages):
        """给思考开着的无工具请求补齐 reasoning_content（空字符串）。

        2026-11-13 真机核实：chat_messages（不带 tools）传 thinking="enabled"
        时，历史里每一个带 tool_calls 的 assistant 消息都必须带
        reasoning_content，缺了接口报 400（"The `reasoning_content` in the
        thinking mode must be passed back to the API."）；不带 tool_calls 的
        纯文本 assistant 消息不需要。工具那几轮思考是关着的，模型本来就没
        产生 reasoning_content，没东西可回传——真机试过，补空字符串接口
        接受（空字符串只表示“本轮没有推理内容”，不编造推理）。
        只补带 tool_calls 又缺 reasoning_content 的 assistant 消息，
        其余原样不动；不改调用方传进来的列表（复制后再补）。
        注意：这只管不带 tools 的 chat_messages。带 tools 的
        chat_with_tools 思考开着时仍走 _check_reasoning_complete 的
        严格检查（缺了直接报错、不补），那条路没经过真机核实，不动它。
        """
        filled = []
        for msg in messages:
            if (
                msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and not msg.get("reasoning_content")
            ):
                patched = dict(msg)
                patched["reasoning_content"] = ""
                filled.append(patched)
            else:
                filled.append(msg)
        return filled

    @staticmethod
    def _normalize_messages(messages):
        """把消息历史转成普通 dict 列表（只留接口认的键），供发送和存盘用。

        assistant 消息原样保留 content / reasoning_content / tool_calls；
        tool_calls 里的每一项也转成普通 dict（id / type / function.name /
        function.arguments）。不改内容、不补字段、不丢 reasoning_content。
        """
        allowed = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")
        normalized = []
        for msg in messages:
            item = {key: msg[key] for key in allowed if key in msg}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                calls = []
                for call in tool_calls:
                    function = call.get("function") or {}
                    calls.append(
                        {
                            "id": call.get("id", ""),
                            "type": call.get("type", "function"),
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": function.get("arguments", ""),
                            },
                        }
                    )
                item["tool_calls"] = calls
            normalized.append(item)
        return normalized

    @staticmethod
    def _check_reasoning_complete(messages, reasoning_field="reasoning_content"):
        """发请求前检查：思考开着 + 本次带 tools 时，历史里每一轮 assistant
        消息都必须带非空的 reasoning_content（官方硬性要求，缺了报 400）。

        哪怕某一轮没调工具也不能少——官方要求的是全量回传，不是“只回传
        调过工具的轮次”。缺了就直接报 ValueError，绝不往里补空字符串
        蒙混（空字符串能不能蒙过去没经过官方确认，不能拿来赌）。
        思考关着时不用检查（本来就没有 reasoning_content 可回传）。
        """
        for index, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            reasoning = msg.get(reasoning_field)
            if not isinstance(reasoning, str) or not reasoning:
                raise ValueError(
                    "思考模式下带 tools 发请求，历史里第 "
                    f"{index + 1} 条 assistant 消息缺少 {reasoning_field}，"
                    "发出去会被接口报 400。请把该轮模型返回的 "
                    f"{reasoning_field} 原样带回来，不要补空字符串。"
                )

    def chat_with_tools(self, messages, tools, thinking="disabled", ledger="tool", task_id=None, trace_id=None):
        """带工具定义的提问（多轮，工具循环专用），返回字典。

        参数 messages：循环内消息历史（system + 本轮 user + 循环各轮），
        每项是普通 dict；assistant 项可带 reasoning_content / tool_calls，
        tool 项带 tool_call_id。参数 tools：OpenAI 兼容的工具定义列表，
        每项 {"type": "function", "function": {"name", "description",
        "parameters"}}。参数 thinking："disabled"（默认，关思考）或
        "enabled"（开思考）；开关通过请求体顶层 "thinking" 字段传。
        参数 ledger：默认 "tool"（工具循环里的每一轮都算工具账本）；
        task_id / trace_id 用来串联记账和审计。
        返回：{"text": 回答文本（可能为空字符串）, "tool_calls": 模型要调的
        工具列表（普通 dict，每项 id / type / function.name / arguments，
        不调工具时为空列表）, "reasoning_content": 本轮推理内容或 None,
        "usage": Token 使用量或 None, "cost_usd": 成本（美元）或 None,
        "model": 模型名}。
        思考开着时，发请求前先做全量回传检查（见 _check_reasoning_complete），
        缺 reasoning_content 就报错、不发请求；思考关着时不检查。
        无 Key、预算文件损坏、usage 缺失时的行为与 chat_messages() 相同。
        """
        # 0. 先验证参数格式。
        if not messages or not isinstance(messages, list):
            raise ValueError("消息历史不能为空，应为消息列表。")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg:
                raise ValueError("消息历史格式错误，每项应为带 'role' 的字典。")
            if msg.get("role") == "assistant" and "content" not in msg and "tool_calls" not in msg:
                raise ValueError("assistant 消息至少要有 content 或 tool_calls 其中一项。")
        if not tools or not isinstance(tools, list):
            raise ValueError("工具定义不能为空，应为工具列表。")
        for tool in tools:
            function = (tool or {}).get("function") or {}
            if not isinstance(tool, dict) or not function.get("name"):
                raise ValueError("工具定义格式错误，每项应为 {'function': {'name': ...}}。")
        if thinking not in ("enabled", "disabled"):
            raise ValueError(f"思考开关只认 'enabled' 或 'disabled'，不是 {thinking!r}。")

        # 1. 先检查 Key，没有 Key 就明确说“未完成”，不要假装调用成功。
        if not self.api_key:
            raise RuntimeError(
                "未配置 API Key，真实连接未完成："
                "请在 .env 里填写 DEEPSEEK_API_KEY，"
                "或设置环境变量 DEEPSEEK_API_KEY 后再运行。"
            )

        # 2. 再检查预算：只有预算文件损坏时才拦。超预算、有未核对的未知成本
        #    时只提醒、不拦截（BOOS 定的新规矩：到 $5 提醒，永不自动停）。
        if self.budget_tracker is not None and not self.budget_tracker.can_call():
            reason = ""
            if hasattr(self.budget_tracker, "status_reason"):
                reason = self.budget_tracker.status_reason(provider="deepseek")
            raise RuntimeError(
                reason or "预算记录异常，付费调用暂不可用，请人工检查 logs/budget_usage.json 后再恢复。"
            )

        # 3. 发请求前：思考开着时做全量回传检查；关着时跳过。
        #    reasoning 字段名按 PROVIDER_THINKING_FIELDS 取（DeepSeek 和 Kimi
        #    都是 reasoning_content，见文件头）。
        fields = PROVIDER_THINKING_FIELDS["deepseek"]
        normalized = self._normalize_messages(messages)
        if thinking == "enabled":
            self._check_reasoning_complete(normalized, fields["reasoning_field"])

        # 4. 发真实请求。timeout（超时）设为 60 秒，避免一直等下去。
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": normalized,
            "tools": tools,
            "max_tokens": MAX_OUTPUT_TOKENS,
            # 思考开关放请求体顶层（DeepSeek 官方写法）。
            fields["thinking_field"]: {"type": thinking},
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"DeepSeek API 调用失败：{exc}") from exc

        data = response.json()

        # 5. 解析回答：content 可能为空（只调工具不说话时），tool_calls
        #    可能没有（直接回答时）。返回的 tool_calls 转成普通 dict，
        #    循环拿去就能执行、存盘，不用再转。
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"DeepSeek API 返回格式异常，无法解析回答：{data}") from exc
        text = message.get("content") or ""
        reasoning_content = message.get(fields["reasoning_field"])
        tool_calls = self._normalize_messages([message])[0].get("tool_calls", [])

        # 6. 解析 Token 使用量并记账（与 chat_messages() 同一套算法：
        #    缓存按缓存价、普通输入按普通价、completion_tokens 按输出价，
        #    推理 Token 已含在 completion_tokens 里，不重复加）。
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
            cached_tokens = min(cached_tokens, prompt_tokens)
            cost_usd = (
                (prompt_tokens - cached_tokens) / 1_000_000 * self.price_input
                + cached_tokens / 1_000_000 * self.price_cached_input
                + completion_tokens / 1_000_000 * self.price_output
            )
            if self.budget_tracker is not None:
                self.budget_tracker.add_cost(
                    cost_usd,
                    model=self.model,
                    ledger=ledger,
                    task_id=task_id,
                    trace_id=trace_id,
                    usage=usage,
                )

        return {
            "text": text,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "usage": usage,
            "cost_usd": cost_usd,
            "model": self.model,
        }
