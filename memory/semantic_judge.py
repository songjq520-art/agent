# memory/semantic_judge.py（语义判官：程序判不定时问大模型，2026-11-22）
#
# 混合法（BOOS 2026-11-22 定）：简单的程序自己判，难的问大模型。
#   主控是十二自己（本地程序）：什么情况下问、问不问，由
#   memory/sqlite_store.py 的 _needs_semantic_judge 按程序算出的
#   触发条件决定（slot 为 None、候选多条分不清、纠正检测拿不准），
#   不调模型、不让模型决定“要不要问模型”。
#   大模型是判官，不是主控：只在触发后被调来做语义判断——新说法属于
#   哪类偏好、是不是在纠正/取代某条旧事实、该不该取代。
#
# 模型能换，别写死（BOOS 2026-11-22 补充：问的大模型不止 DeepSeek，
#   以后 GPT、Gemini、Grok 接上也能用）：
#   - 判官走这里的分发层：judge_semantic(provider=...) 传名字，
#     或改 config.py 的 SEMANTIC_JUDGE（provider / model，
#     环境变量 SHIER_JUDGE_PROVIDER / SHIER_JUDGE_MODEL 也能改）；
#   - 现在只有 deepseek 接上了（默认 deepseek-flash）；gpt / gemini /
#     grok 在 MODEL_CONFIG 里还是 enabled=False、model=None，
#     分发层会明确拒绝（SemanticJudgeUnavailable），调用方按程序
#     fallback 走，不崩；接上后在这里加一个 _build_* 函数 + 一行分发表，
#     判断代码不用改。
#   - 复用各家已有的 Provider（DeepSeekProvider.chat_messages，
#     统一接口 {"text", "usage", "cost_usd", "model"}），不另起一套调用。
#
# 成本：判断函数返回 usage（Token 数）和 cost_usd（按配置的单价算），
#   换模型时成本跟着配置的单价走，不用改代码。默认 deepseek-flash
#   闲时价：输入 $0.15/M、输出 $0.60/M；一次判断输入约 300–600 Token、
#   输出约 60–100 Token，约 $0.0001–0.0002（高峰价翻倍，仍约 $0.0003）。
#   大模型断了（超时、API 报错、没 Key、没接好）时抛 SemanticJudgeError
#   或 SemanticJudgeUnavailable，调用方降级走程序，别崩、别丢事实。
#
# 2026-11-23 model fix #4（模型优先 + 全上下文）：程序只判明显 case
#   （槽位能推出、原样对上、库里没候选），其余都问模型；判官输入是
#   整段对话上下文（judge_semantic / judge_importance 的 context 参数），
#   不只一句话。

import json
import logging
import re

from config import MODEL_CONFIG, SEMANTIC_JUDGE, get_api_key

logger = logging.getLogger(__name__)

# 判官提示词之外的开销上限：一次判断最多带几条旧记忆候选（调用方也截）。
MAX_JUDGE_CANDIDATES = 10


class SemanticJudgeError(Exception):
    """判官调用失败（超时、API 报错、回的格式不对）：调用方降级走程序。"""


class SemanticJudgeUnavailable(SemanticJudgeError):
    """判官没接好（模型没配置、没启用、没 Key）：调用方降级走程序。"""


def get_judge_config():
    """判官配置（config.py 的 SEMANTIC_JUDGE）：provider / model / max_tokens。

    缺字段时给默认值（deepseek / deepseek-flash / 300），不崩。"""
    cfg = SEMANTIC_JUDGE or {}
    return {
        "provider": cfg.get("provider") or "deepseek",
        "model": cfg.get("model") or "deepseek-flash",
        "max_tokens": cfg.get("max_tokens") or 300,
    }


def _build_deepseek(model, budget_tracker=None):
    """建 DeepSeek 判官（复用 providers/deepseek.py 的统一接口）。"""
    from providers.deepseek import DeepSeekProvider

    api_key = get_api_key("deepseek")
    if not api_key:
        raise SemanticJudgeUnavailable(
            "DeepSeek 判官没接好：没找到 DEEPSEEK_API_KEY，"
            "请在 .env 里填写后再用模型判断。"
        )
    try:
        return DeepSeekProvider(api_key=api_key, budget_tracker=budget_tracker,
                                model=model)
    except ValueError as exc:
        raise SemanticJudgeUnavailable(f"DeepSeek 判官没接好：{exc}")


# 判官分发层：名字 → 建 Provider 的函数。接新模型时在这里加一行，
# 再在 MODEL_CONFIG 里把那家的 model / enabled / 价格填好。
_PROVIDER_BUILDERS = {
    "deepseek": _build_deepseek,
}


def resolve_judge_provider(provider=None, model=None, budget_tracker=None):
    """解析判官用的 Provider，返回 (provider_obj, provider_name, model_name)。

    provider 传对象时（测试注入的 mock / 已建好的 Provider）直接用它，
    不查配置；传名字或不传时按配置分发：名字不在分发表里、模型没配置、
    没启用、没 Key，都抛 SemanticJudgeUnavailable（调用方降级走程序）。
    """
    cfg = get_judge_config()
    name = provider if isinstance(provider, str) else None
    model_name = model or cfg["model"]
    if provider is not None and not isinstance(provider, str):
        # 已建好的对象（测试注入的 mock 优先走这里）：名字按对象上的
        # provider_name / model 属性认，认不出就记 "injected"。
        obj_name = getattr(provider, "provider_name", None) or "injected"
        obj_model = getattr(provider, "model", None) or model_name
        return provider, obj_name, obj_model
    provider_name = name or cfg["provider"]
    builder = _PROVIDER_BUILDERS.get(provider_name)
    if builder is None:
        raise SemanticJudgeUnavailable(
            f"判官模型 {provider_name!r} 还没接上：现在只接了 "
            f"{sorted(_PROVIDER_BUILDERS)}，接上后在 config.py 的 "
            "SEMANTIC_JUDGE 里改个值就能切过去。"
        )
    entry = MODEL_CONFIG.get(provider_name) or {}
    if not entry.get("enabled") or not entry.get("model"):
        raise SemanticJudgeUnavailable(
            f"判官模型 {provider_name!r} 在 MODEL_CONFIG 里没启用/没填模型名。"
        )
    if model_name not in (entry.get("model"), entry.get("model_pro")):
        raise SemanticJudgeUnavailable(
            f"判官模型名 {model_name!r} 不在 {provider_name} 的配置里"
            f"（{entry.get('model')} / {entry.get('model_pro')}）。"
        )
    return builder(model_name, budget_tracker=budget_tracker), provider_name, model_name


def judge_available():
    """判官现在能不能用：返回 (ok, reason)。只查配置和 Key，不调模型。"""
    try:
        cfg = get_judge_config()
        name = cfg["provider"]
        if name not in _PROVIDER_BUILDERS:
            return False, f"{name} 还没接上"
        entry = MODEL_CONFIG.get(name) or {}
        if not entry.get("enabled") or not entry.get("model"):
            return False, f"{name} 没启用/没填模型名"
        if not get_api_key(name):
            return False, f"没找到 {name.upper()}_API_KEY"
        return True, f"{name}/{cfg['model']}"
    except Exception as exc:  # 配置读坏了也别崩，报不可用
        return False, f"配置读取失败：{exc}"


def build_judge_prompt(new_text, candidates, context=None):
    """判官提示词：短、只问三件事，让模型只回一行 JSON，别写大段话。

    candidates 是 [{"id", "text"}]（最多 10 条，调用方截过）。
    context 是整段对话上下文（最近几轮消息，2026-11-23 model fix #4）：
    判官只看一句话分不清“在纠正哪条”时，对话能定锚——比如上一轮刚
    说的才是被纠正的那条。没有上下文时不加这一段，老行为不变。
    """
    lines = [
        "你是记忆管家，只判断新说法和旧记忆的关系，只回 JSON，不要解释。",
        f"新说法：{new_text}",
        "旧记忆：",
    ]
    for i, cand in enumerate(candidates or [], 1):
        lines.append(f"{i}. [{cand.get('id')}] {cand.get('text')}")
    if context:
        lines.append("对话上下文（最近几轮，原样——新说法是在这段对话里说的）：")
        for msg in list(context or [])[-6:]:
            role = (msg or {}).get("role")
            content = ((msg or {}).get("content") or "").strip()
            if not content:
                continue
            who = (
                "用户" if role == "user"
                else "助手" if role == "assistant"
                else (role or "未知")
            )
            lines.append(f"{who}：{content[:200]}")
    lines.append(
        "任务：1) 新说法属于哪类偏好/槽位（如“喜欢喝的饮品”“喜欢吃的食物”"
        "“喜欢看的影片”，无法归类写 null）；2) 新说法是不是在纠正/取代"
        "其中某条旧记忆；3) 如果是，是哪条（写它的 id，否则 null），"
        "该不该取代（true/false）。"
    )
    # 规则 (a)（2026-11-23，R2 食物案；2026-11-23 收窄）：偏好一次只有
    # 一个当前答案；“我现在喜欢 X 了”这类说法表示当前偏好变了，同一层级
    # 的旧偏好应当被取代。但只取代同一层级、直接冲突的那条：新说法是泛指
    # （一类东西）时，不取代更具体的同类记忆——“喜欢吃面条”取代“喜欢吃
    # 米饭”（都是主食），不取代“喜欢吃牛肉面”“喜欢吃鸡杂面”（牛肉面/
    # 鸡杂面就是面条的一种，可以并存）；纠正记录（写着“纠正了此前……”的
    # 便条）本身不是偏好，绝不能当取代目标；候选里没有同一层级直接冲突
    # 的旧偏好时，判不取代，不要硬挑一条。
    lines.append(
        "规则：'喜欢吃什么/喝什么/看什么'一次只有一个当前答案；新说法带"
        "'现在''改''其实''不对''更喜欢''只'等词、表示偏好变了时，同一层级"
        "的旧偏好应当被取代，不要判成'可以并存'。只取代同一层级、直接冲突"
        "的那条：新说法是泛指（一类东西）时，不取代更具体的同类记忆——比如"
        "“喜欢吃面条”取代“喜欢吃米饭”（都是主食），不取代“喜欢吃牛肉面”"
        "“喜欢吃鸡杂面”（牛肉面/鸡杂面就是面条的一种，可以并存）；本身就是"
        "纠正记录的记忆（写着“纠正了此前……”的便条）不是偏好，绝不能当取代"
        "目标；候选里没有同一层级直接冲突的旧偏好时，判不取代，不要硬挑一条。"
    )
    lines.append(
        '只回这一行 JSON：{"slot": "...", "is_correction": true, '
        '"target_id": "...", "should_supersede": true, "reason": "一句话原因"}'
    )
    return "\n".join(lines)


def _as_bool(value, field):
    """把模型回的 true/false 宽容地转成 bool；转不出来就报错（别猜）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise SemanticJudgeError(f"判官回的 {field} 不是 true/false：{value!r}")


def parse_judge_response(text, candidate_ids):
    """解析判官回的 JSON（容错：前后多余文字、markdown 代码块都剥掉）。

    要求：slot 是字符串或 null；is_correction / should_supersede 是 bool；
    target_id 是候选里的 id 或 null；should_supersede 为 true 时
    target_id 必须在候选里。格式不对、字段不对、目标不在候选里时
    抛 SemanticJudgeError（调用方按程序 fallback 处理，不崩）。
    """
    raw = (text or "").strip()
    if not raw:
        raise SemanticJudgeError("判官回了空内容")
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is not None:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        raise SemanticJudgeError(f"判官回的不是 JSON：{raw[:200]}")
    slot = data.get("slot")
    if slot is not None and not isinstance(slot, str):
        raise SemanticJudgeError(f"判官回的 slot 不是字符串：{slot!r}")
    slot = (slot or "").strip() or None
    is_correction = _as_bool(data.get("is_correction"), "is_correction")
    should_supersede = _as_bool(data.get("should_supersede"), "should_supersede")
    target_id = data.get("target_id")
    if target_id is not None and not isinstance(target_id, str):
        raise SemanticJudgeError(f"判官回的 target_id 不是字符串：{target_id!r}")
    target_id = (target_id or "").strip() or None
    if target_id is not None and target_id not in (candidate_ids or []):
        raise SemanticJudgeError(f"判官指的目标 {target_id!r} 不在候选里")
    if should_supersede and target_id is None:
        raise SemanticJudgeError("判官说该取代，却没指是哪条")
    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)
    return {
        "slot": slot,
        "is_correction": is_correction,
        "target_id": target_id,
        "should_supersede": should_supersede,
        "reason": (reason or "").strip() or None,
    }


def build_importance_prompt(text, context=None):
    """重要性判断提示词（Phase 2 #2）：一句话是长期稳定的事实/偏好/
    决定（值得钉住长期记住），还是一次性经历/随口一提。

    context 是整段对话上下文（2026-11-23 model fix #4）：一句话单看
    分不清长期还是一次性时，对话能定锚。没有上下文时不加这一段。

    只回一行 JSON：{"important": true/false/"unsure", "reason": "..."}。
    important 为 true 表示值得长期钉住；false 表示一次性经历、以后
    可能用不上；拿不准时回 "unsure"，不要硬猜。
    """
    lines = [
        "你是记忆管家，只判断一句话值不值得长期钉住记住，只回 JSON，不要解释。",
        f"句子：{text}",
    ]
    if context:
        lines.append("对话上下文（最近几轮，原样——这句话是在这段对话里说的）：")
        for msg in list(context or [])[-6:]:
            role = (msg or {}).get("role")
            content = ((msg or {}).get("content") or "").strip()
            if not content:
                continue
            who = (
                "用户" if role == "user"
                else "助手" if role == "assistant"
                else (role or "未知")
            )
            lines.append(f"{who}：{content[:200]}")
    lines += [
        "规则：身份（名字、住处）、长期偏好（喜欢吃什么/喝什么/看什么）、"
        "决定（打算、计划、定投、规矩）、钱的规矩、安全权限、项目决策，"
        "这些是长期稳定的，判 important=true；"
        "一次性经历（昨天/今天/刚才做了什么、吃了一顿什么、去了一趟哪），"
        "判 important=false；"
        "玩笑、测试、假设（开玩笑、这是测试、如果/假如），判 important=false；"
        "一句话既像长期事实又像一次性经历、分不清时，判 important=\"unsure\"，"
        "不要硬猜。",
        '只回这一行 JSON：{"important": true/false/"unsure", "reason": "一句话原因"}',
    ]
    return "\n".join(lines)


def parse_importance_response(text):
    """解析重要性判断回的 JSON。

    要求：important 是 true/false 或字符串 "unsure"；reason 是字符串
    或缺省。格式不对时抛 SemanticJudgeError（调用方按程序 fallback
    处理，不崩）。
    """
    raw = (text or "").strip()
    if not raw:
        raise SemanticJudgeError("重要性判断回了空内容")
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is not None:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        raise SemanticJudgeError(f"重要性判断回的不是 JSON：{raw[:200]}")
    important = data.get("important")
    if isinstance(important, bool):
        pass
    elif isinstance(important, str) and important.strip().lower() == "unsure":
        important = "unsure"
    elif isinstance(important, str) and important.strip().lower() in ("true", "false"):
        important = important.strip().lower() == "true"
    elif isinstance(important, (int, float)) and important in (0, 1):
        important = bool(important)
    else:
        raise SemanticJudgeError(
            f"重要性判断回的 important 不是 true/false/\"unsure\"：{important!r}"
        )
    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)
    return {
        "important": important,
        "reason": (reason or "").strip() or None,
    }


def judge_importance(text, provider=None, model=None, budget_tracker=None,
                     context=None):
    """问大模型判官：一句话值不值得长期钉住记住。

    复用 judge_semantic 的分发层（provider/model/budget_tracker 同款
    入参，模型能换、断了抛错）。返回 {"important": true/false/"unsure",
    "reason", "provider", "model", "usage", "cost_usd"}。
    模型断了或回的格式不对时抛 SemanticJudgeError /
    SemanticJudgeUnavailable——调用方接到异常就降级走程序（当一次性
    经历存，3 天 3 次兜底还能晋升），别崩、别丢事实。
    """
    provider_obj, provider_name, model_name = resolve_judge_provider(
        provider, model, budget_tracker=budget_tracker
    )
    prompt = build_importance_prompt(text, context=context)
    cfg = get_judge_config()
    try:
        result = provider_obj.chat_messages(
            [{"role": "user", "content": prompt}],
            ledger="chat", thinking="disabled",
            max_tokens=cfg["max_tokens"],
        )
    except SemanticJudgeError:
        raise
    except Exception as exc:
        logger.warning("重要性判断调用失败，降级走程序：%s", exc)
        raise SemanticJudgeError(f"重要性判断调用失败：{exc}")
    if not isinstance(result, dict):
        raise SemanticJudgeError(f"重要性判断回的不是字典：{result!r}")
    usage = result.get("usage")
    cost = result.get("cost_usd")
    if cost is None:
        cost = estimate_judge_cost_usd(usage, provider_name, model_name)
    verdict = parse_importance_response(result.get("text"))
    verdict.update({
        "provider": provider_name,
        "model": model_name,
        "usage": usage,
        "cost_usd": cost,
    })
    logger.info(
        "重要性判断：provider=%s model=%s important=%s cost_usd=%s",
        provider_name, model_name, verdict.get("important"), cost,
    )
    return verdict


def estimate_judge_cost_usd(usage, provider, model):
    """按配置的单价算一次判断的成本（美元）：用量 Token 数 × 单价。

    usage 是 Provider 回的 {"prompt_tokens", "completion_tokens"}；
    没给 usage 或单价没配好时返回 None（不编数字）。
    换模型时成本跟着 MODEL_CONFIG 里那家的单价走，不用改代码；
    DeepSeek 高峰价是闲时的 2 倍，这里按闲时价算，实际以官方后台为准。
    """
    if not isinstance(usage, dict):
        return None
    entry = MODEL_CONFIG.get(provider) or {}
    if provider == "deepseek" and model == entry.get("model_pro"):
        price_in = entry.get("price_input_per_1m_pro")
        price_out = entry.get("price_output_per_1m_pro")
    else:
        price_in = entry.get("price_input_per_1m")
        price_out = entry.get("price_output_per_1m")
    if price_in is None or price_out is None:
        return None
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    try:
        return (
            float(prompt_tokens) / 1_000_000 * float(price_in)
            + float(completion_tokens) / 1_000_000 * float(price_out)
        )
    except (TypeError, ValueError):
        return None


def judge_semantic(new_text, candidates, provider=None, model=None,
                   budget_tracker=None, context=None):
    """问大模型判官：新说法和旧记忆的关系。

    参数 new_text：新说法（一句话）；candidates：旧记忆 [{"id", "text"}]；
    provider：Provider 对象（测试注入）或名字（"deepseek"，不传按配置）；
    model：模型名（不传按配置，默认 deepseek-flash）。
    返回 {"slot", "is_correction", "target_id", "should_supersede",
    "reason", "provider", "model", "usage", "cost_usd"}。
    模型断了（超时、API 报错、没 Key、没接好）或回的格式不对时抛
    SemanticJudgeError / SemanticJudgeUnavailable——调用方接到异常就
    降级走程序，别崩、别丢事实。降级要记日志（这里记一条 warning）。
    """
    candidates = list(candidates or [])[:MAX_JUDGE_CANDIDATES]
    if not candidates:
        raise SemanticJudgeError("判官没有候选可判")
    provider_obj, provider_name, model_name = resolve_judge_provider(
        provider, model, budget_tracker=budget_tracker
    )
    prompt = build_judge_prompt(new_text, candidates, context=context)
    cfg = get_judge_config()
    try:
        result = provider_obj.chat_messages(
            [{"role": "user", "content": prompt}],
            ledger="chat", thinking="disabled",
            max_tokens=cfg["max_tokens"],
        )
    except SemanticJudgeError:
        raise
    except Exception as exc:
        logger.warning("语义判断调用失败，降级走程序：%s", exc)
        raise SemanticJudgeError(f"判官调用失败：{exc}")
    if not isinstance(result, dict):
        raise SemanticJudgeError(f"判官回的不是字典：{result!r}")
    usage = result.get("usage")
    cost = result.get("cost_usd")
    if cost is None:
        cost = estimate_judge_cost_usd(usage, provider_name, model_name)
    verdict = parse_judge_response(
        result.get("text"), [c.get("id") for c in candidates]
    )
    verdict.update({
        "provider": provider_name,
        "model": model_name,
        "usage": usage,
        "cost_usd": cost,
    })
    logger.info(
        "语义判断：provider=%s model=%s slot=%s is_correction=%s "
        "target_id=%s should_supersede=%s cost_usd=%s",
        provider_name, model_name, verdict.get("slot"),
        verdict.get("is_correction"), verdict.get("target_id"),
        verdict.get("should_supersede"), cost,
    )
    return verdict
