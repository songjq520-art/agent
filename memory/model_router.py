# memory/model_router.py（整理记忆的大模型路由：主力 + 备选 + 本地降级）
# 十二自己整理记忆时调大模型的那条路（语气判断、自检第 9 项抽查都走这里）。
# 规矩（2026-11-19，方案十三节 + BOOS 当天新定）：
#   - 主力：DeepSeek 满血版（deepseek-v4-pro）；备选第一层：DeepSeek flash；
#     备选第二层：本地降级（没大模型可调时返回 None，调用方用本地规则顶上）。
#   - 主力连续调不通 3 次就切到 flash；主力恢复后，下一次整理任务
#     自动切回主力（调用方在每批任务开头传 probe_primary=True 探一次）。
#   - 每月花费上限 3 美元。超了只提醒、不断服务（BOOS 2026-11-19 定：
#     所有金额限制超了都只提醒、不断服务，由 BOOS 决定要不要停）——
#     超了继续正常调，只记一笔：写进当月自检报告，下次 BOOS 对话时
#     告诉他超了。语气判断的费用走同一个预算，同样处理。
#   - 只有 BOOS 亲口说停，才停：paused_by_user 由对话层按 BOOS 的话
#     设置，路由器自己永不因超预算停调。
# 花费记在路由器自己的状态文件里（按月），不走 BudgetTracker 的
# tool/chat 账本——那两个账本只认 tool/chat 两个名字，整理记忆的
# 3 美元上限是单独的一本账。BudgetTracker 传进来时，每次调用的
# 成本也会顺手记进它（ledger="chat"，只记总数、不占工具预算线）。
# 状态文件损坏时从头开始、不崩；写文件用临时文件 + 原子替换。

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ORGANIZE_MONTHLY_CAP_USD = 3.0
PRIMARY_MODEL = "deepseek-v4-pro"
FALLBACK_MODEL = "deepseek-flash"
FAILURES_TO_FALLBACK = 3

TIER_PRIMARY = "primary"
TIER_FALLBACK = "fallback"

_STATE_VERSION = 1


def _current_month(now=None):
    moment = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.strftime("%Y-%m")


def _now_iso(now=None):
    moment = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


class OrganizingModelRouter:
    """整理记忆的大模型路由（主力 + 备选 + 本地降级 + 月度预算提醒）。

    provider_factory：callable(model_name) -> provider 或 None。
    provider 要有 chat_messages(messages, ledger, thinking, max_tokens)
    这样的方法（providers.deepseek.DeepSeekProvider 就是这个形状），
    返回 {"text", "cost_usd", "model"}，调不通时抛异常。没传时用默认
    工厂（读 config 的 DeepSeek Key 建 provider）；没 Key 时工厂
    返回 None，judge() 直接返回 None（本地降级）。
    state_path：状态文件路径（JSON）。不传时只记内存里，不落盘——
    测试、没地方写文件时用；生产调用方必须传（见 Mem0Store）。
    budget_tracker：传了就把每次调用的成本也记进它（ledger="chat"）。
    """

    def __init__(self, state_path=None, provider_factory=None,
                 budget_tracker=None):
        self.state_path = str(state_path) if state_path else None
        self.provider_factory = (
            provider_factory if provider_factory is not None
            else self._default_provider_factory
        )
        self.budget_tracker = budget_tracker
        self._state = self._load_state()

    # ---------- 状态文件 ----------

    def _default_state(self):
        return {
            "version": _STATE_VERSION,
            "month": _current_month(),
            "spent_usd": 0.0,
            "over_budget": False,
            "first_over_at": None,
            "notice_acknowledged": False,
            "paused_by_user": False,
            "tier": TIER_PRIMARY,
            "consecutive_failures": 0,
            "switch_log": [],
        }

    def _load_state(self):
        state = self._default_state()
        if not self.state_path:
            return state
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            return state
        if not isinstance(loaded, dict):
            return state
        for key, value in state.items():
            if key in loaded:
                state[key] = loaded[key]
        # 月份滚过去了：花费、超预算、提醒从头算；暂停是 BOOS 的决定，
        # 跨月也留着，等他亲口改；tier 和失败计数也清零（新的一月重新试主力）。
        if state.get("month") != _current_month():
            state["month"] = _current_month()
            state["spent_usd"] = 0.0
            state["over_budget"] = False
            state["first_over_at"] = None
            state["notice_acknowledged"] = False
            state["consecutive_failures"] = 0
            state["tier"] = TIER_PRIMARY
        return state

    def _save_state(self):
        if not self.state_path:
            return
        try:
            parent = os.path.dirname(self.state_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._state, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_path)
        except OSError:
            logger.warning("整理记忆路由：状态文件写失败，本次花费只记内存。", exc_info=True)

    # ---------- 内部：建 provider ----------

    @staticmethod
    def _default_provider_factory(model_name, budget_tracker=None):
        """默认工厂：读 config 的 DeepSeek Key 建 provider。

        没 Key、Key 读不出来时返回 None（本地降级，不抛异常）。
        budget_tracker 由路由器传进来（调用方传了才记账）。"""
        try:
            from config import get_api_key
            from providers.deepseek import DeepSeekProvider

            api_key = get_api_key("deepseek")
            if not api_key:
                return None
            return DeepSeekProvider(
                api_key=api_key, budget_tracker=budget_tracker, model=model_name
            )
        except Exception:
            logger.warning("整理记忆路由：建 DeepSeek provider 失败，本次降级本地。", exc_info=True)
            return None

    def _build_provider(self, model_name):
        try:
            return self.provider_factory(model_name, self.budget_tracker)
        except TypeError:
            # 工厂只认一个参数（测试里常见的简单工厂）时，不带 tracker 再试一次。
            try:
                return self.provider_factory(model_name)
            except Exception:
                logger.warning("整理记忆路由：建 provider 失败，本次降级本地。", exc_info=True)
                return None
        except Exception:
            logger.warning("整理记忆路由：建 provider 失败，本次降级本地。", exc_info=True)
            return None

    # ---------- 内部：记账与超预算提醒 ----------

    def _record_cost(self, cost_usd):
        """记一笔花费（只记有 usage 的实数；cost 是 None 时不记金额）。

        超了 3 美元上限：只记一笔、打日志，不停调、不降级（BOOS 定：
        超了只提醒、不断服务）。第一次超的那次把 first_over_at 和
        over_budget 写进状态，提醒 BOOS 那条（pending_notice）就有了。"""
        if cost_usd is None:
            return
        try:
            amount = float(cost_usd)
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        self._state["spent_usd"] = round(
            float(self._state.get("spent_usd") or 0.0) + amount, 6
        )
        if (
            self._state["spent_usd"] > ORGANIZE_MONTHLY_CAP_USD
            and not self._state.get("over_budget")
        ):
            self._state["over_budget"] = True
            self._state["first_over_at"] = _now_iso()
            self._state["notice_acknowledged"] = False
            logger.warning(
                "整理记忆：本月大模型花费已超 %.2f 美元（已花 %.2f 美元），"
                "继续正常调用，等 BOOS 决定要不要停。",
                ORGANIZE_MONTHLY_CAP_USD,
                self._state["spent_usd"],
            )
        self._save_state()

    def _record_primary_failure(self):
        """主力一次调不通：连续失败计数 +1；满 3 次就切到备选。

        只数主力的失败：备选顶上来的那次成功不把计数清零——不然
        “主力连续 3 次不通”永远攒不够（每次备选一成功就清零）。
        备选成功与否都不动这个计数；只有主力通了才清零。"""
        self._state["consecutive_failures"] = (
            int(self._state.get("consecutive_failures") or 0) + 1
        )
        if (
            self._state.get("tier") == TIER_PRIMARY
            and self._state["consecutive_failures"] >= FAILURES_TO_FALLBACK
        ):
            self._state["tier"] = TIER_FALLBACK
            self._state["consecutive_failures"] = 0
            self._state.setdefault("switch_log", []).append(
                {
                    "at": _now_iso(),
                    "from": TIER_PRIMARY,
                    "to": TIER_FALLBACK,
                    "reason": "主力连续 %d 次调不通" % FAILURES_TO_FALLBACK,
                }
            )
            logger.warning(
                "整理记忆：主力 %s 连续 %d 次调不通，切到备选 %s。",
                PRIMARY_MODEL,
                FAILURES_TO_FALLBACK,
                FALLBACK_MODEL,
            )
        self._save_state()

    def _record_primary_success(self):
        self._state["consecutive_failures"] = 0
        self._save_state()

    # ---------- 对外：判断调用 ----------

    def judge(self, system_prompt, user_prompt, max_tokens=500,
              purpose="organize", probe_primary=False):
        """请大模型做一次短判断（语气、自检抽查都走这里）。

        返回 {"text", "model", "cost_usd", "tier"}；调不通、没 provider、
        BOOS 亲口叫停时返回 None——调用方按“判不出来”走本地规则，
        不猜、不拦路、不抛异常。
        probe_primary=True：这批任务开头探一次主力——tier 在备选时，
        先试主力，通了就切回主力（“主力恢复后，下一次整理任务自动
        切回主力”）；不通就用备选继续这一句。平时不传，免得每句都
        多一次探路调用。
        thinking 关掉：短判断不需要推理过程，开着白烧推理 Token。
        超预算时照常调（只提醒、不断服务，见 _record_cost）。"""
        if self._state.get("paused_by_user"):
            return None
        tier = self._state.get("tier") or TIER_PRIMARY
        models_to_try = []
        if probe_primary and tier == TIER_FALLBACK:
            models_to_try.append((PRIMARY_MODEL, TIER_PRIMARY))
        models_to_try.append(
            (PRIMARY_MODEL if tier == TIER_PRIMARY else FALLBACK_MODEL, tier)
        )
        if tier == TIER_PRIMARY:
            models_to_try.append((FALLBACK_MODEL, TIER_FALLBACK))
        seen_models = set()
        last_error = None
        for model_name, model_tier in models_to_try:
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            provider = self._build_provider(model_name)
            if provider is None:
                last_error = "no provider"
                continue
            try:
                result = provider.chat_messages(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    ledger="chat",
                    thinking="disabled",
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - 调不通就记失败、换备选
                last_error = exc
                if model_tier == TIER_PRIMARY:
                    self._record_primary_failure()
                continue
            text = (result or {}).get("text")
            if not isinstance(text, str) or not text.strip():
                if model_tier == TIER_PRIMARY:
                    self._record_primary_failure()
                last_error = "empty reply"
                continue
            cost = (result or {}).get("cost_usd")
            self._record_cost(cost)
            if model_tier == TIER_PRIMARY:
                self._record_primary_success()
            if model_tier == TIER_PRIMARY and tier == TIER_FALLBACK:
                # 探路成功：切回主力，记一笔。
                self._state["tier"] = TIER_PRIMARY
                self._state.setdefault("switch_log", []).append(
                    {"at": _now_iso(), "from": TIER_FALLBACK, "to": TIER_PRIMARY,
                     "reason": "主力恢复，整理任务切回主力"}
                )
                self._save_state()
            return {
                "text": text.strip(),
                "model": (result or {}).get("model") or model_name,
                "cost_usd": cost,
                "tier": model_tier,
            }
        if last_error is not None:
            logger.info("整理记忆：大模型本次调不通，走本地降级。")
        return None

    # ---------- 对外：状态与提醒 ----------

    def monthly_spent(self):
        """本月整理记忆的大模型花费（美元）。"""
        self._state = self._load_state()
        return float(self._state.get("spent_usd") or 0.0)

    def is_over_budget(self):
        """本月是否已超 3 美元上限（只提醒、不停调）。"""
        self._state = self._load_state()
        return bool(self._state.get("over_budget"))

    def pending_notice(self):
        """超预算提醒（给 BOOS 的那条），没有时返回 None。

        超了且还没认领时返回 {"month", "spent_usd", "cap_usd",
        "first_over_at", "message"}；BOOS 认领后（acknowledge_notice）
        返回 None——同一月里只提醒一次，不刷屏。
        message 是直接能拿去跟 BOOS 说的那句。"""
        self._state = self._load_state()
        if not self._state.get("over_budget"):
            return None
        if self._state.get("notice_acknowledged"):
            return None
        spent = float(self._state.get("spent_usd") or 0.0)
        return {
            "month": self._state.get("month"),
            "spent_usd": spent,
            "cap_usd": ORGANIZE_MONTHLY_CAP_USD,
            "first_over_at": self._state.get("first_over_at"),
            "message": (
                "提醒：整理记忆的大模型本月已花 %.2f 美元，超过了 3 美元的上限，"
                "服务没停、还在正常调。要不要停，你说了算。"
                % spent
            ),
        }

    def acknowledge_notice(self):
        """BOOS 已经知道超预算了（对话层在他回话后调）：同一月里不再提醒。"""
        self._state["notice_acknowledged"] = True
        self._save_state()

    def set_paused_by_user(self, paused):
        """BOOS 亲口说停 / 恢复：True 停调大模型（本地降级），False 恢复。

        只有 BOOS 能定这个——路由器自己永不因超预算停调。"""
        self._state["paused_by_user"] = bool(paused)
        self._save_state()

    def switch_back_to_primary(self):
        """手动切回主力（下一次判断就用主力；通了就留着，不通再按 3 次规则切走）。"""
        self._state["tier"] = TIER_PRIMARY
        self._state["consecutive_failures"] = 0
        self._save_state()

    def status(self):
        """当前状态（给自检报告用）：tier、失败计数、本月花费、上限、
        是否超预算、是否 BOOS 叫停、有没有待提醒、切换记录。"""
        self._state = self._load_state()
        return {
            "tier": self._state.get("tier"),
            "tier_model": (
                PRIMARY_MODEL if self._state.get("tier") == TIER_PRIMARY
                else FALLBACK_MODEL
            ),
            "consecutive_failures": int(self._state.get("consecutive_failures") or 0),
            "month": self._state.get("month"),
            "spent_usd": float(self._state.get("spent_usd") or 0.0),
            "cap_usd": ORGANIZE_MONTHLY_CAP_USD,
            "over_budget": bool(self._state.get("over_budget")),
            "first_over_at": self._state.get("first_over_at"),
            "notice_pending": self.pending_notice() is not None,
            "paused_by_user": bool(self._state.get("paused_by_user")),
            "switch_log": list(self._state.get("switch_log") or []),
        }
