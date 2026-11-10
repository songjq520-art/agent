# web/server.py（本地网页入口的服务程序）
# 作用：在你自己的电脑上跑一个很小的网页服务，让你用浏览器和 Agent 对话。
# 原理：只用 Python 自带库（http.server），不装任何新依赖。
#   打开浏览器访问 http://127.0.0.1:8000，就能看到聊天页面（web/index.html）；
#   你在页面里发消息，服务把消息交给 main.py 的 run_question()（和黑窗口
#   里打字走的是同一套逻辑：路由 → 模型 → 预算 → 日志），再把回答返回给页面。
#   对话历史和预算跟踪器由服务进程一直拿着，和 chat_loop() 的做法一样。
# 安全：只监听 127.0.0.1（本机），不对外网开放，别人打不开你的页面；
#       不记录 API Key，日志和记账走原来那套（logs/agent.log、logs/budget_usage.json）。
# 用法（在项目根目录下）：
#   python -m web.server            # 默认本机 8000 端口
#   python -m web.server --port 8080  # 换个端口
#   然后在浏览器打开 http://127.0.0.1:8000
# 停止：按 Ctrl+C。
#
# 接口（给页面用的，前后端只通过这几个接口打交道，页面里不写后端逻辑）：
#   GET  /            → 聊天页面（web/index.html）
#   GET  /api/status  → 模型状态 + 预算 + 记忆是否启用（JSON，含 mem0_enabled）
#   POST /api/chat    → 发消息，请求 JSON {"message": "..."}，
#                        返回 JSON {"reply": 回答, "model": 模型名,
#                        "mode": "real" 或 "demo", "cost_usd": 成本,
#                        "usage": Token 用量}；
#                        工具循环停在确认卡上时返回
#                        {"mode": "waiting_approval",
#                         "needs_approval": {task_id, approval_id, tool,
#                         args, risk_level, expires_at}}，本轮不写历史、
#                         不存记忆，等确认结果回来再恢复；
#                        出错时返回 {"error": 原因}。
#   POST /api/chat/confirm → 处理一张确认卡，请求 JSON
#                        {"task_id": 任务 id, "approval_id": 确认编号,
#                         "decision": "confirm"（确认执行）或 "cancel"（取消）}，
#                        确认了就执行存着的参数、把停住的那轮跑完，
#                        返回和 /api/chat 成功时一样的字典；
#                        取消了也跑完（模型按“用户没同意”继续答）。
#                        已经处理过、超时、参数被改过时回 409 + 原因，
#                        不会执行第二次。
#   GET  /api/approvals/pending → 所有等确认的任务（JSON
#                        {"approvals": [{task_id, goal, approval}]}），
#                        刷新页面后还能找回没点的卡。
#   POST /api/reset   → 开新对话：先把当前对话总结保存，再清空历史，
#                        返回 {"ok": true}。
#   GET  /api/conversations → 最近对话列表（JSON {"conversations": [...]}，
#                        每条只有 id、标题、开始/更新时间，不带正文）。
#   POST /api/conversations/open → 回到一段旧对话，请求 JSON {"id": 对话 id}，
#                        返回 {"id", "title", "summary", "messages"}：
#                        messages 是恢复的最近几轮（原样），页面按它重画
#                        聊天区，接着聊；找不到时回 404。
#   GET  /api/memories → 已存记忆列表（JSON {"memories": [...], "mem0_enabled": bool}，
#                        每条含 id、text、metadata：source/created_at/confidence/
#                        user_confirmed/confirmed；记忆未启用时列表为空）。
#   GET  /api/memories/<id>/scope → 删除范围预览（deletion_scope 的原样返回：
#                        删哪条、删几行 history、哪些地方不在删除范围内）。
#   POST /api/memories/<id>/forget → 忘一条记忆，请求 JSON {"confirmed": true}，
#                        只有 confirmed 为 true 才调 delete_memory(approve=True)；
#                        没确认时回 400、什么都不删。
#   POST /api/memories/correct → 纠正一条记忆，请求 JSON
#                        {"memory_id": id（页面上选的那条，优先用它）,
#                         "old_text": 旧内容（没选 id 时用它找）,
#                         "new_text": 正确的说法}，
#                        返回 correct_memory 的状态字典（updated / ambiguous /
#                        not_found / unchanged / failed）；ambiguous 时带候选列表，
#                        由页面让 BOOS 选一条再提交，不猜。
#
# 注意（给以后看代码的人）：
#   - 对话历史存在内存里（ChatState.messages），重启服务后清空；
#     但每段对话结束时都会总结保存（摘要 + 最近几轮存在硬盘上，
#     memory/conversation_summary.json），重启后左边“最近”列表里能找回，
#     点一条就能接着聊；新对话开始时读回最新摘要当上下文。
#   - 服务用 ThreadingHTTPServer，每个请求一个线程；ChatState 里有一把锁，
#     问、清空、读状态都走这把锁，一次只处理一个请求，历史和预算不会乱。
#   - 长期记忆走 mem0（见 memory/mem0_store.py）：服务启动时建一个
#     Mem0Store（常驻只建一次），每轮提问前语义检索、回答后自动存；
#     记忆库建不好时不带记忆运行，服务照常启动。
#   - 监听地址和端口集中在下面的 ServerConfig 里改（命令行参数、环境变量
#     或默认值），不要在别处写死；默认仍是 127.0.0.1，不开放局域网。
#   - 身份验证还没做，预留了 authenticator 钩子（见 make_handler）；
#     以后开放局域网/手机访问时，先接上校验，再把 host 改成 0.0.0.0。

import argparse
import ipaddress
import json
import logging
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

# 让直接运行 python web/server.py 也能找到项目根目录的 main.py、config.py。
# 用 python -m web.server 运行时本来就不需要这一行，加上也不影响。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MONTHLY_BUDGET_USD, BudgetTracker, get_model_status  # noqa: E402
from main import _memory_block, run_question  # noqa: E402
from memory.conversation_summary import (  # noqa: E402
    ConversationSummaryStore,
    make_deepseek_summarizer,
)
from memory.mem0_store import create_mem0_store  # noqa: E402

logger = logging.getLogger("agent")

# INDEX_PATH（聊天页面文件）：按 server.py 的位置找，不依赖当前工作目录。
INDEX_PATH = Path(__file__).resolve().parent / "index.html"

# _MEMORY_SAVE_HISTORY_MAX（回答后存记忆时最多带最近几条消息）：
# 只带最近 6 条（约 3 轮），和 main.py 里记忆提取带的历史长度一致；
# 每轮都把全部历史发去提取又贵又慢，只带最近的就够了。
_MEMORY_SAVE_HISTORY_MAX = 6


@dataclass
class ServerConfig:
    """网页服务的监听配置（host 和 port 的唯一来源，别处不要写死）。

    默认只监听本机 127.0.0.1:8000，不对外网开放。
    改地址/端口的方式（优先级从高到低）：
      1. 命令行参数：python -m web.server --host 127.0.0.1 --port 8080
      2. 环境变量：AGENT_WEB_HOST / AGENT_WEB_PORT
      3. 下面 host / port 的默认值
    未来扩展（现在不做）：
      - 想开放局域网/手机访问：先接身份验证（见 make_handler 的
        authenticator 钩子），再把 host 改成 0.0.0.0；不要只改 host 不加校验。
      - 端口被占用：换个端口就行，不用改代码。
    """

    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls):
        """从环境变量读配置；读不到或格式不对时用默认值。"""
        host = os.environ.get("AGENT_WEB_HOST") or cls.host
        try:
            port = int(os.environ.get("AGENT_WEB_PORT") or cls.port)
        except ValueError:
            port = cls.port
        return cls(host=host, port=port)


class ChatState:
    """一次服务运行里的对话状态：共用的预算跟踪器 + 对话历史 + mem0 长期记忆库。

    做法和 main.py 的 chat_loop() 一样：整个服务进程只用一个
    BudgetTracker 和一个 messages 历史（user 和 assistant 消息按时间顺序）。
    成功拿到回答后，才把本轮 user 问题和 assistant 回答追加进历史；
    演示模式或调用失败时，历史不动。
    对话历史只保存在内存里，重启服务后清空；但每段对话结束（开新对话、
    回到旧对话、停服务）时都会总结保存——摘要 + 最近几轮存在硬盘上
    （memory/conversation_summary.json），重启后左边“最近”列表里能找回。
    长期记忆：mem0_store 不传时不带记忆；传了时，每轮提问前先语义检索
    相关记忆拼进发给模型的问题里（历史里仍记原话），回答成功后把本轮
    及最近的消息存进长期记忆（mem0 自动提取，不用确认）。
    对话摘要：summary_store 不传时不记摘要；传了时，每轮后历史超
    32,000 token 就总结压缩一次、记一笔最近对话，新对话开始时读回
    最新摘要当上下文。
    记忆数据只保存在本地（memory/mem0_chroma/ + memory/history.db），不上传。
    """

    def __init__(self, budget_tracker=None, mem0_store=None, summary_store=None,
                 tool_gateway=None, task_store=None, search_service=None):
        self.tracker = budget_tracker or BudgetTracker()
        self.messages = []
        self.mem0_store = mem0_store
        self.summary_store = summary_store
        # 工具循环用的网关、任务存储、搜索服务：不传时第一次用才建
        # （懒加载——服务启动时不建任务文件、不碰搜索服务的 Key）；
        # 测试时传假的进来。
        self._tool_gateway = tool_gateway
        self._task_store = task_store
        self._search_service = search_service
        # _lock（并发锁）：服务用 ThreadingHTTPServer，每个请求一个线程。
        # 没有这把锁，两个 /api/chat 同时跑会互相污染历史（你问你的、他答他的，
        # 追加顺序乱掉），还可能同时通过预算检查、多记一笔费用。
        # 所以问、清空、读状态都走这把锁，一次只处理一个请求；本机单人用，
        # 等一会儿没关系，顺序和预算不会错。调用模型时会一直拿着锁，
        # 期间别的请求（包括看状态、清空）会等一等，这是为了保证一致性。
        self._lock = threading.Lock()

    @property
    def tool_gateway(self):
        """工具网关（懒加载）：第一次用时才建，测试可注入假的。"""
        if self._tool_gateway is None:
            from tools.tool_gateway import ToolGateway

            self._tool_gateway = ToolGateway()
        return self._tool_gateway

    @property
    def task_store(self):
        """任务存储（懒加载）：第一次用时才建，测试可注入假的。"""
        if self._task_store is None:
            from tasks.task_store import TaskStore

            self._task_store = TaskStore()
        return self._task_store

    @property
    def search_service(self):
        """搜索服务：不传时为 None，循环自己建一个带翻译器的；
        测试时传假服务进来。"""
        return self._search_service

    def reset(self):
        """开新对话：先把当前对话总结保存，再清空历史、换一条空记录。

        预算不动，只清聊天记录。摘要没启用时就是原来的清空。
        """
        with self._lock:
            self._flush_summary_locked()
            if self.summary_store is not None:
                try:
                    self.summary_store.begin_conversation()
                except Exception:
                    logger.warning("开新对话失败。", exc_info=True)
            self.messages = []

    def _flush_summary_locked(self):
        """把当前对话收尾总结（调用方已拿着锁）：失败只记日志，不拦聊天。"""
        if self.summary_store is None:
            return
        try:
            self.summary_store.flush(self.messages)
        except Exception:
            logger.warning("总结当前对话失败。", exc_info=True)

    def flush_summary(self):
        """服务停止时把当前对话收尾总结（带锁的公开版，供 main() 调）。"""
        with self._lock:
            self._flush_summary_locked()

    def list_conversations(self):
        """最近对话列表（给网页左边用）：摘要没启用时返回空列表。"""
        with self._lock:
            if self.summary_store is None:
                return []
            try:
                return self.summary_store.list_conversations()
            except Exception:
                logger.warning("读最近对话列表失败。", exc_info=True)
                return []

    def open_conversation(self, conv_id):
        """回到一段旧对话：先把当前对话收尾（总结保存），再恢复那段的摘要 + 最近几轮。

        返回 {"id", "title", "summary", "messages"}（messages 是恢复的
        最近几轮，页面按它重画聊天区，接着聊）；找不到时返回 None。
        摘要没启用时也返回 None（没有可回的对话）。
        """
        with self._lock:
            if self.summary_store is None:
                return None
            self._flush_summary_locked()
            try:
                record = self.summary_store.open_conversation(conv_id)
            except Exception:
                logger.warning("回到旧对话失败：id=%s。", conv_id, exc_info=True)
                return None
            if record is None:
                return None
            self.messages = [dict(m) for m in record.get("recent") or []]
            return {
                "id": record["id"],
                "title": record.get("title") or "新对话",
                "summary": record.get("summary") or "",
                "messages": [dict(m) for m in self.messages],
            }

    def current_summary(self):
        """新对话提问时用的摘要（给页面顶部的摘要卡片用，和发给模型的是同一份）。

        reset() 之后正在聊的是空记录，这里会回落到列表最新那条——
        也就是刚结束的“上一段对话”的摘要；没有摘要时返回空串，
        页面不显示卡片。摘要没启用时也返回空串。
        """
        with self._lock:
            if self.summary_store is None:
                return ""
            try:
                return self.summary_store.current_summary() or ""
            except Exception:
                logger.warning("读当前摘要失败。", exc_info=True)
                return ""

    def status(self):
        """给页面状态栏用的：模型状态 + 本月预算使用情况 + 记忆是否启用。

        预算分两账本显示（BOOS 2026-11-11 定）：工具账本（tool_cost_usd，
        上限 $5）和聊天账本（chat_cost_usd，不设限），同时给人民币换算
        （汇率见 config.USD_CNY_RATE）。老键 monthly_budget_usd /
        total_cost_usd / remaining_usd 保留兼容：remaining_usd 现在是
        工具账本剩下的预算（$5 减工具账本），不是总数剩下的。
        """
        with self._lock:
            summary = self.tracker.budget_summary_cny()
            return {
                "model_status": get_model_status(),
                "monthly_budget_usd": MONTHLY_BUDGET_USD,
                "total_cost_usd": round(self.tracker.total_cost_usd, 6),
                "remaining_usd": round(self.tracker.remaining_usd(), 6),
                "tool_cost_usd": round(self.tracker.tool_cost_usd, 6),
                "chat_cost_usd": round(self.tracker.chat_cost_usd, 6),
                "tool_cost_cny": summary["tool_cost_cny"],
                "chat_cost_cny": summary["chat_cost_cny"],
                "tool_budget_cap_cny": summary["tool_budget_cap_cny"],
                "usd_cny_rate": summary["usd_cny_rate"],
                "budget_corrupted": bool(getattr(self.tracker, "corrupted", False)),
                "unknown_cost": bool(self.tracker.unknown_cost),
                "mem0_enabled": self.mem0_store is not None,
            }

    def ask(self, prompt):
        """问一句话，返回给页面展示的字典。

        成功时：{"reply": 回答, "model": 模型名, "mode": "real",
                  "cost_usd": 成本, "usage": Token 用量}，
        并把本轮问答追加进历史。
        本轮是聊天纠错（“你记错了……”）时，成功返回里多一个
        "correction" 键：{"status", "memory_id", "old_text",
        "new_text", "candidates", "reason"}（见 Mem0Store.add_turn），
        页面按它展示“改好了 / 请挑一条 / 改失败了”。
        演示模式（无 Key）时：{"reply": 提示, "model": None,
        "mode": "demo", "cost_usd": None, "usage": None}，历史不动。
        调用失败时：直接抛 RuntimeError，调用方转成错误返回，历史不动。
        有 mem0 记忆库时：提问前先语义检索，把相关记忆拼成提示块和问题
        一起发给模型（只改发给模型的那一份，历史里仍记原话）；没搜到
        记忆时原样提问。回答成功后，把本轮及最近的消息存进长期记忆。
        记忆检索或保存出问题时不拦聊天：检索失败就原样提问，保存失败
        不影响已经拿到的回答（add_turn 内部已把异常吞掉）。
        锁顺序：这里先拿 ChatState 的锁，add_turn 内部再拿 mem0 的全局
        写锁；全项目统一这个顺序（先 ChatState 锁，再 mem0 写锁），
        不会死锁。
        """
        with self._lock:
            # 提问前检索记忆：有结果就拼提示块，没结果就什么都不拼。
            model_prompt = prompt
            if self.mem0_store is not None:
                try:
                    memories = self.mem0_store.search(prompt, top_k=5)
                    # 记下本轮检索到的记忆 id（最近对话上下文）：纠错时拿它当
                    # “当前在聊什么”的提示（见 Mem0Store.note_context）。
                    self.mem0_store.note_context(memories)
                except Exception:
                    # 检索失败也不拦聊天，只记一条日志，原样提问。
                    logger.warning("记忆检索失败，本次不带记忆继续。", exc_info=True)
                    memories = []
                block = _memory_block(memories)
                if block:
                    model_prompt = block + "\n\n" + prompt
            outcome = run_question(
                model_prompt,
                budget_tracker=self.tracker,
                messages=self.messages,
                summary_store=self.summary_store,
                tool_gateway=self.tool_gateway,
                task_store=self.task_store,
                search_service=self.search_service,
                original_prompt=prompt,
            )
            if outcome["mode"] == "demo":
                return {
                    "reply": outcome["message"],
                    "model": None,
                    "mode": "demo",
                    "cost_usd": None,
                    "usage": None,
                }
            if outcome["mode"] == "waiting_approval":
                # 工具循环停在确认卡上：本轮还没答完，历史、摘要、记忆
                # 都不动；任务已经存盘了，页面按 needs_approval 显示确认卡，
                # BOOS 点了之后走 /api/chat/confirm 恢复。
                approval = outcome["approval"] or {}
                return {
                    "reply": "",
                    "model": None,
                    "mode": "waiting_approval",
                    "cost_usd": outcome.get("cost_usd"),
                    "usage": None,
                    "needs_approval": {
                        "task_id": (outcome["task"] or {}).get("task_id"),
                        "approval_id": approval.get("approval_id"),
                        "tool": approval.get("tool"),
                        "args": approval.get("args"),
                        "risk_level": approval.get("risk_level"),
                        "expires_at": approval.get("expires_at"),
                    },
                }
            result = outcome["result"]
            return self._finish_turn_locked(
                prompt,
                result["text"],
                result["model"],
                result.get("cost_usd"),
                result.get("usage"),
            )

    def _finish_turn_locked(self, prompt, reply_text, model_name, cost_usd, usage):
        """一轮问答答完了的收尾（调用方已拿着锁）：把本轮问答追加进历史、
        更新对话摘要、存进长期记忆。返回给页面展示的字典。
        ask() 和 resolve_approval() 共用这一段，保证确认恢复后的那轮
        和普通那轮收尾一模一样：只存 BOOS 的原话 + 最后的回答，
        工具中间结果不进历史、不进记忆。
        """
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": reply_text})
        # 对话摘要：历史超上限就总结压缩一次，再把本轮记进最近对话列表；
        # 失败只记日志、不拦聊天。
        if self.summary_store is not None:
            try:
                self.summary_store.maybe_compact(self.messages)
                self.summary_store.sync(self.messages)
            except Exception:
                logger.warning("对话摘要更新失败。", exc_info=True)
        # 回答成功后存记忆：只带本轮及最近的消息（含刚追加的新问答）。
        # 聊天纠错（“你记错了”接线）：add_turn 里已经改好/问好了，
        # 这里只把纠错结果放进返回里（"correction" 键），供页面展示：
        # 改好了（updated）、已经记对了（unchanged）、分不清改哪条
        # （ambiguous，带候选列表、等 BOOS 下一句挑）、改失败了
        # （failed，原记忆没动）、没找到旧记忆（not_found，新说法
        # 已当新记忆存下）。没纠错时不加这个键，原来的五个键不动。
        correction = None
        if self.mem0_store is not None:
            saved = self.mem0_store.add_turn(self.messages[-_MEMORY_SAVE_HISTORY_MAX:])
            if isinstance(saved, dict):
                correction = saved.get("correction")
        response = {
            "reply": reply_text,
            "model": model_name,
            "mode": "real",
            "cost_usd": cost_usd,
            "usage": usage,
        }
        if correction is not None:
            response["correction"] = correction
        return response

    def list_pending_approvals(self):
        """所有等确认的任务（给页面刷新时用）：每条带 task_id、原话 goal
        和确认记录（approval_id、工具名、参数、风险、过期时间）。
        读失败时返回空列表，页面显示“没有等确认的”，不拦聊天。"""
        with self._lock:
            try:
                tasks = self.task_store.list_waiting()
            except Exception:
                logger.warning("读等确认的任务失败。", exc_info=True)
                return []
            waiting = []
            for task in tasks:
                approval = task.get("approval") or {}
                waiting.append(
                    {
                        "task_id": task.get("task_id"),
                        "goal": task.get("goal") or "",
                        "approval": {
                            "approval_id": approval.get("approval_id"),
                            "tool": approval.get("tool"),
                            "args": approval.get("args"),
                            "risk_level": approval.get("risk_level"),
                            "expires_at": approval.get("expires_at"),
                        },
                    }
                )
            return waiting

    def resolve_approval(self, task_id, approval_id, decision):
        """处理 BOOS 对一张确认卡的决定（确认执行 / 取消），再把停住的那轮跑完。

        参数 decision："confirm"（确认执行）或 "cancel"（取消）。
        先让网关处理确认（原子抢占 + 只执行存着的参数，见
        tools/tool_gateway.py 的 resolve_approval），再调 run_question
        恢复循环跑完：确认了就把工具结果喂回去接着跑，取消了就告诉模型
        “用户没同意”再接着跑。跑完后的收尾和 ask() 一模一样
        （写历史、更新摘要、存记忆，只存原话 + 最后回答）。
        返回：跑完了时和 ask() 成功时一样的字典（多一个 "ok": True）；
        确认被拒绝时（已经处理过、超时、参数被改过）返回
        {"ok": False, "reason": 原因}，调用方按 409 回给页面；
        恢复后又停在下一张确认卡上时返回 waiting_approval 字典
        （多一个 "ok": True），页面再显示下一张卡。
        调用失败时（模型调用报错）直接抛 RuntimeError，调用方转成错误返回。
        已知限制（第一版不修）：BOOS 在一张卡等确认时又发了新消息，
        之后再点确认，恢复的那轮会追加在新消息后面——顺序上旧轮在后，
        不影响回答本身，页面上看起来是旧回答晚到了。
        """
        with self._lock:
            task = self.task_store.load(task_id)
            if task is None:
                # task_id 对不上时再按 approval_id 找一次（页面可能拿的是旧卡）。
                task = self.task_store.find_by_approval(approval_id)
                if task is None:
                    return {"ok": False, "reason": f"找不到任务：{task_id}。"}
                task_id = task["task_id"]
            resolved = self.tool_gateway.resolve_approval(
                self.task_store, task_id, approval_id, decision
            )
            if resolved["decision"] == "rejected":
                return {"ok": False, "reason": resolved["reason"]}
            goal = (resolved.get("task") or {}).get("goal") or task.get("goal") or ""
            if resolved["decision"] == "cancelled":
                outcome = run_question(
                    goal,
                    budget_tracker=self.tracker,
                    messages=self.messages,
                    summary_store=self.summary_store,
                    tool_gateway=self.tool_gateway,
                    task_store=self.task_store,
                    search_service=self.search_service,
                    resume_task=resolved["task"],
                    user_declined=True,
                    original_prompt=goal,
                )
            else:  # 确认执行了
                outcome = run_question(
                    goal,
                    budget_tracker=self.tracker,
                    messages=self.messages,
                    summary_store=self.summary_store,
                    tool_gateway=self.tool_gateway,
                    task_store=self.task_store,
                    search_service=self.search_service,
                    resume_task=resolved["task"],
                    pending_tool_result=resolved["result"],
                    original_prompt=goal,
                )
            if outcome["mode"] == "waiting_approval":
                # 恢复后又停在下一张确认卡上：同样不写历史，等下一张卡。
                approval = outcome["approval"] or {}
                return {
                    "ok": True,
                    "reply": "",
                    "model": None,
                    "mode": "waiting_approval",
                    "cost_usd": outcome.get("cost_usd"),
                    "usage": None,
                    "needs_approval": {
                        "task_id": (outcome["task"] or {}).get("task_id"),
                        "approval_id": approval.get("approval_id"),
                        "tool": approval.get("tool"),
                        "args": approval.get("args"),
                        "risk_level": approval.get("risk_level"),
                        "expires_at": approval.get("expires_at"),
                    },
                }
            if outcome["mode"] == "demo":
                return {"ok": False, "reason": "恢复任务时没有可用的模型 Key。"}
            result = outcome["result"]
            response = self._finish_turn_locked(
                goal,
                result["text"],
                result["model"],
                result.get("cost_usd"),
                result.get("usage"),
            )
            response["ok"] = True
            return response

    # ---------- 长期记忆管理（“长期记忆”页面用） ----------
    # 原则：网页层只做“传话”——列出、展示范围、把 BOOS 的确认传下去；
    # 真正的忘（深度删除）和改（纠错）都在 Mem0Store 里，
    # 走同一套删除两道关（归属校验 + approve=True）和纠错流程，
    # 网页层不自己实现删除/纠正，也绝不绕过它们。

    def list_memories(self):
        """返回已存记忆列表（供页面展示用）。

        每项是 get_all() 的原样返回：{"id", "text", "metadata"}，
        metadata 里有 source（user/inferred/unconfirmed）、created_at
        （形成时间）、confidence、user_confirmed/confirmed。
        记忆未启用时返回空列表；读失败时也返回空列表（页面显示“暂无记忆”，
        不拦聊天）。
        """
        with self._lock:
            if self.mem0_store is None:
                return []
            try:
                return self.mem0_store.get_all()
            except Exception:
                logger.warning("读记忆列表失败。", exc_info=True)
                return []

    def memory_scope(self, memory_id):
        """返回一条记忆的删除范围（deletion_scope 的原样返回）。

        记忆未启用时返回 None；其他错误时也返回 None（调用方按“找不到”处理）。
        这个方法只读不删——删不删由 BOOS 看了范围、点了确认之后，
        再调 forget_memory() 决定。
        """
        with self._lock:
            if self.mem0_store is None:
                return None
            try:
                return self.mem0_store.deletion_scope(memory_id)
            except Exception:
                logger.warning("取删除范围失败：id=%s。", memory_id, exc_info=True)
                return None

    def forget_memory(self, memory_id, confirmed):
        """忘一条记忆（深度删除）。

        confirmed 必须是 True——BOOS 在页面上看了删除范围、点了“确认删除”，
        才算同意。没确认时抛 PermissionError，什么都不删；
        适配器里 delete_memory(approve=True) 那道关也还在，
        两道关缺一个都不删。
        记忆未启用时抛 RuntimeError。
        返回 True（删干净了）或 False（这条记忆已经不在库里了）；
        删了一半、验不过时抛 RuntimeError（信息里写清删到哪一步了，
        调用方把原话展示给 BOOS，让他决定补删还是手动检查）。
        """
        with self._lock:
            if not confirmed:
                raise PermissionError("删除记忆需要先确认删除范围。")
            if self.mem0_store is None:
                raise RuntimeError("记忆未启用。")
            return self.mem0_store.delete_memory(memory_id, approve=True)

    def correct_memory(self, old_text, new_text, memory_id=None):
        """纠正一条记忆（返回 correct_memory 的状态字典）。

        memory_id 是页面上选的那条记忆的 id：给了就先按 id 取准旧文本
        再纠正——原文对上了就不用猜，多条相似也不怕；
        没给时按 old_text 找（找不到/对上多条时返回 not_found/ambiguous，
        让页面问 BOOS，不猜、不改）。
        记忆未启用时抛 RuntimeError。
        """
        with self._lock:
            if self.mem0_store is None:
                raise RuntimeError("记忆未启用。")
            if memory_id:
                # 按 id 取准旧文本：get_all 里找得到才纠，找不到说明
                # 这条已经被删了，直接报 not_found，不凭空改。
                owned = {
                    item["id"]: item
                    for item in self.mem0_store.get_all(include_sensitive=True)
                }
                item = owned.get(memory_id)
                if item is None:
                    return {
                        "status": "not_found",
                        "memory_id": None,
                        "old_text": None,
                        "new_text": new_text,
                        "candidates": [],
                        "reason": "库里没有这条记忆，可能已经被删了。",
                    }
                old_text = item["text"]
            return self.mem0_store.correct_memory(old_text, new_text)


def make_handler(state, authenticator=None):
    """按给定的对话状态，生成一个请求处理类。

    这样写是为了让测试能传一个用临时预算文件的 ChatState，
    不碰真实的 logs/budget_usage.json。

    authenticator（身份验证钩子，未来预留）：一个函数，收到请求处理对象，
    返回 True 放行、False 拒绝（回 401）。现在默认 None，即不校验，
    因为服务只监听本机 127.0.0.1，只有本机能连。
    未来开放局域网/手机访问时，先在这里接上真正的校验（比如比对
    请求头里的 Token），再把 host 改成 0.0.0.0；不要只改 host 不加校验。
    """

    class ChatHandler(BaseHTTPRequestHandler):
        """处理网页请求的类：只认下面列出的地址，其他一律 404。

        聊天：GET /、GET /api/status、POST /api/chat、POST /api/reset；
        记忆管理：GET /api/memories、GET /api/memories/<id>/scope、
        POST /api/memories/<id>/forget、POST /api/memories/correct。
        """

        # 日志只写到 Python 的 logging（不往页面里写），保持安静：
        # 每次请求只记一行方法和地址，不记消息内容（消息里可能有隐私）。
        def log_message(self, format, *args):  # noqa: A002 - 标准库要求的名字
            sys.stderr.write(
                "%s - %s\n" % (self.address_string(), format % args)
            )

        def _send_json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self):
            try:
                body = INDEX_PATH.read_bytes()
            except OSError:
                # send_error 的状态行只能用 latin-1，这里用 ASCII，避免中文编不进去。
                self.send_error(404, "index.html not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _check_auth(self):
            """身份验证钩子：没接 authenticator 时一律放行；接了就按它说的办。"""
            if authenticator is None:
                return True
            try:
                return bool(authenticator(self))
            except Exception:
                # 校验函数自己出错时按拒绝处理，不让请求继续往下走。
                traceback.print_exc()
                return False

        def _handle_safely(self, fn):
            """包一层通用异常处理：接口里没料到的错误，记到服务端日志，
            给前端回一个通用的 500，不把错误详情（路径、堆栈）泄露出去。"""
            try:
                fn()
            except Exception:
                traceback.print_exc()
                try:
                    self._send_json({"error": "服务内部出错，请稍后再试。"}, status=500)
                except Exception:
                    # 回错时又出错（比如客户端已断开），就不再管了。
                    pass

        def do_GET(self):  # noqa: N802 - 标准库要求的名字
            self._handle_safely(self._do_get)

        def _read_json_body(self):
            """读 POST 请求体并解析成 JSON 对象。

            返回 (payload, error, status)：成功时 payload 是字典、
            error 是 None；失败时 payload 是 None、error 是给前端看的
            错误信息、status 是对应的 HTTP 状态码。
            Content-Length 无效、请求体不是合法 JSON、或解析出来不是
            JSON 对象时，都在这里拦住（和 /api/chat 原来那套校验一致）。
            """
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None, "请求头 Content-Length 无效。", 400
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return None, "请求不是合法的 JSON。", 400
            if not isinstance(payload, dict):
                return None, '请求内容必须是 JSON 对象，例如 {"message": "你好"}。', 400
            return payload, None, 200

        def _do_get(self):
            if not self._check_auth():
                self._send_json({"error": "未授权：请先通过身份验证。"}, status=401)
                return
            # 问号后面是查询参数（比如 ?x=1），路由只看问号前面的路径。
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_html()
            elif path == "/api/status":
                self._send_json(state.status())
            elif path == "/api/approvals/pending":
                # 等确认的任务：刷新页面后还能找回没点的卡。
                self._send_json({"approvals": state.list_pending_approvals()})
            elif path == "/api/conversations":
                # 最近对话列表：摘要没启用时列表为空，页面显示“暂无对话”。
                self._send_json({"conversations": state.list_conversations()})
            elif path == "/api/memories":
                # 记忆列表：记忆未启用时 mem0_enabled 为 False、列表为空，
                # 页面据此显示“记忆未启用”，不报错。
                self._send_json(
                    {
                        "mem0_enabled": state.mem0_store is not None,
                        "memories": state.list_memories(),
                    }
                )
            elif path.startswith("/api/memories/") and path.endswith("/scope"):
                # 删除范围预览：GET /api/memories/<id>/scope。
                # id 在路径中间那段（可能含 URL 编码，先解码再查）。
                memory_id = unquote(path[len("/api/memories/") : -len("/scope")].strip("/"))
                if state.mem0_store is None:
                    self._send_json({"error": "记忆未启用。"}, status=503)
                    return
                scope = state.memory_scope(memory_id)
                if scope is None or not scope.get("found"):
                    self._send_json(
                        {"error": "找不到这条记忆，可能已经被删了。"}, status=404
                    )
                    return
                self._send_json(scope)
            else:
                # 中文错误信息放 JSON 正文里（UTF-8 安全），不用 send_error，
                # 因为它的状态行只能用 latin-1，中文放不进去。
                self._send_json({"error": "没有这个地址。"}, status=404)

        def do_POST(self):  # noqa: N802 - 标准库要求的名字
            self._handle_safely(self._do_post)

        def _do_post(self):
            if not self._check_auth():
                self._send_json({"error": "未授权：请先通过身份验证。"}, status=401)
                return
            path = self.path.split("?", 1)[0]
            if path == "/api/reset":
                state.reset()
                # 新对话读回的摘要（上一段对话的）一起带上，页面在顶部
                # 显示摘要卡片；没有摘要时是空串，页面不显示卡片。
                self._send_json({"ok": True, "summary": state.current_summary()})
                return
            if path == "/api/chat/confirm":
                # 处理一张确认卡：请求 {"task_id", "approval_id",
                # "decision": "confirm" 或 "cancel"}。
                # 确认了就执行存着的参数、把停住的那轮跑完；取消了也跑完
                # （模型按“用户没同意”继续答）。已经处理过、超时、
                # 参数被改过时回 409 + 原因，不会执行第二次。
                payload, error, status = self._read_json_body()
                if error is not None:
                    self._send_json({"error": error}, status=status)
                    return
                task_id = str(payload.get("task_id") or "").strip()
                approval_id = str(payload.get("approval_id") or "").strip()
                decision = str(payload.get("decision") or "").strip()
                if not task_id or not approval_id:
                    self._send_json(
                        {"error": "确认卡信息不完整（缺 task_id 或 approval_id）。"},
                        status=400,
                    )
                    return
                if decision not in ("confirm", "cancel"):
                    self._send_json(
                        {"error": "确认决定只认 confirm（确认执行）或 cancel（取消）。"},
                        status=400,
                    )
                    return
                try:
                    result = state.resolve_approval(task_id, approval_id, decision)
                except RuntimeError as exc:
                    # 服务端日志保留完整异常，方便排查；前端只回通用提示，
                    # 不把原始异常暴露出去。状态码仍是 502。
                    traceback.print_exc()
                    self._send_json({"error": "调用失败，请稍后再试。"}, status=502)
                    return
                if not result.get("ok", True):
                    self._send_json(
                        {"error": result.get("reason") or "这张确认卡已经处理过了。"},
                        status=409,
                    )
                    return
                self._send_json(result)
                return
            if path == "/api/conversations/open":
                # 回到一段旧对话：请求 {"id": 对话 id}。
                # 先把当前对话总结保存（state.open_conversation 里做），
                # 再恢复那段的最近几轮；找不到回 404。
                payload, error, status = self._read_json_body()
                if error is not None:
                    self._send_json({"error": error}, status=status)
                    return
                conv_id = str(payload.get("id") or "").strip()
                if not conv_id:
                    self._send_json({"error": "请先选一段对话。"}, status=400)
                    return
                result = state.open_conversation(conv_id)
                if result is None:
                    self._send_json(
                        {"error": "找不到这段对话，可能已经被删了。"}, status=404
                    )
                    return
                self._send_json(result)
                return
            if path.startswith("/api/memories/") and path.endswith("/forget"):
                # 忘一条记忆：POST /api/memories/<id>/forget，请求 {"confirmed": true}。
                # confirmed 不是 true 时回 400、什么都不删——删除的确认必须
                # 是 BOOS 在页面上看了范围之后点的，不能靠前端默认带上。
                memory_id = unquote(path[len("/api/memories/") : -len("/forget")].strip("/"))
                payload, error, status = self._read_json_body()
                if error is not None:
                    self._send_json({"error": error}, status=status)
                    return
                if payload.get("confirmed") is not True:
                    self._send_json(
                        {"error": "删除前请先确认删除范围：请先查看删除范围，再点“确认删除”。"},
                        status=400,
                    )
                    return
                if state.mem0_store is None:
                    self._send_json({"error": "记忆未启用。"}, status=503)
                    return
                try:
                    deleted = state.forget_memory(memory_id, confirmed=True)
                except PermissionError as exc:
                    # 理论上走不到（上面已确认、这里传 approve=True），
                    # 万一适配器那道关拦住了，按 403 回，不删。
                    self._send_json({"error": str(exc)}, status=403)
                    return
                except RuntimeError as exc:
                    # 删了一半、验不过：把适配器写清的进度原样告诉 BOOS
                    # （只含 id 和行数，不含记忆原文），让他决定补删还是手动检查。
                    self._send_json({"error": f"删除失败：{exc}"}, status=500)
                    return
                if deleted:
                    self._send_json({"ok": True, "deleted": True})
                else:
                    self._send_json(
                        {
                            "ok": True,
                            "deleted": False,
                            "reason": "这条记忆已经不在库里了。",
                        }
                    )
                return
            if path == "/api/memories/correct":
                # 纠正一条记忆：POST /api/memories/correct，请求
                # {"memory_id": id（页面上选的那条）, "old_text": 旧内容,
                #  "new_text": 正确的说法}。memory_id 和 old_text 至少给一个；
                # new_text 不能为空。返回 correct_memory 的状态字典，
                # ambiguous 时带候选列表，页面让 BOOS 选一条再提交。
                payload, error, status = self._read_json_body()
                if error is not None:
                    self._send_json({"error": error}, status=status)
                    return
                memory_id = str(payload.get("memory_id") or "").strip() or None
                old_text = str(payload.get("old_text") or "").strip()
                new_text = str(payload.get("new_text") or "").strip()
                if not new_text:
                    self._send_json({"error": "请输入纠正后的内容。"}, status=400)
                    return
                if not memory_id and not old_text:
                    self._send_json(
                        {"error": "请先选一条要纠正的记忆，或写明旧内容。"}, status=400
                    )
                    return
                if state.mem0_store is None:
                    self._send_json({"error": "记忆未启用。"}, status=503)
                    return
                result = state.correct_memory(old_text, new_text, memory_id=memory_id)
                self._send_json(result)
                return
            if path != "/api/chat":
                self._send_json({"error": "没有这个地址。"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_json({"error": "请求头 Content-Length 无效。"}, status=400)
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._send_json({"error": "请求不是合法的 JSON。"}, status=400)
                return
            # 请求内容必须是 JSON 对象（{"message": "..."}）。
            # 数组、数字、字符串、null 等都会让下面的 payload.get() 报错，
            # 所以先在这里拦住，给前端一个清楚的 400，而不是让服务异常。
            if not isinstance(payload, dict):
                self._send_json(
                    {"error": '请求内容必须是 JSON 对象，例如 {"message": "你好"}。'},
                    status=400,
                )
                return
            prompt = str(payload.get("message") or "").strip()
            if not prompt:
                self._send_json({"error": "消息不能为空。"}, status=400)
                return
            try:
                reply = state.ask(prompt)
            except RuntimeError as exc:
                # 服务端日志保留完整异常（含服务商错误详情），方便排查；
                # 前端只回通用提示，不把原始异常（服务商错误详情或内部
                # 请求信息）暴露出去。状态码仍是 502。
                traceback.print_exc()
                self._send_json({"error": "调用失败，请稍后再试。"}, status=502)
                return
            self._send_json(reply)

    return ChatHandler


def _is_loopback_host(host):
    """判断监听地址是不是本机回环地址（127.0.0.1、::1、localhost 等）。

    127.0.0.0/8 整个段都是回环地址（127.0.0.2 也算），::1 是 IPv6 回环。
    主机名只有 "localhost" 算回环；其他主机名或写错的地址一律按非回环处理，
    宁可严一点，不让它悄悄开放出去。
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_server(config=None, budget_tracker=None, authenticator=None, mem0_store=None, summary_store=None, tool_gateway=None, task_store=None, search_service=None):
    """建一个网页服务（不启动），方便测试用临时端口和临时预算文件。

    config 是 ServerConfig；不传时用环境变量/默认值（ServerConfig.from_env()）。
    authenticator 是身份验证钩子（见 make_handler），默认不校验。
    mem0_store 是 mem0 长期记忆库（Mem0Store），不传时不带记忆；传了时，
    每轮提问前先语义检索相关记忆带给模型，回答成功后自动存本轮对话。
    summary_store 是对话摘要库（ConversationSummaryStore），不传时不记摘要；
    传了时，每轮后历史超上限就总结压缩、记一笔最近对话，开新对话时
    总结保存，新对话开始时读回最新摘要当上下文。
    tool_gateway / task_store / search_service 是工具循环用的网关、
    任务存储、搜索服务，不传时第一次用才建；测试时可传假的进来。
    测试时可传一个假 store（有 search/add_turn/get_all 方法即可），不碰真实 mem0。

    安全保护：监听地址不是本机回环地址（比如 0.0.0.0 或局域网地址），
    却没有配置身份验证函数时，拒绝启动（抛 ValueError），防止一不小心
    开放了无身份验证的局域网访问。默认 127.0.0.1 不受影响。
    如需开放局域网，请先接上 authenticator 做身份验证，再改 host。
    """
    if config is None:
        config = ServerConfig.from_env()
    if not _is_loopback_host(config.host) and authenticator is None:
        raise ValueError(
            f"监听地址 {config.host} 不是本机回环地址，但没有配置身份验证函数；"
            "为防止无身份验证的局域网访问，拒绝启动。"
            "如需开放局域网，请先接上 authenticator 做身份验证。"
        )
    state = ChatState(
        budget_tracker=budget_tracker,
        mem0_store=mem0_store,
        summary_store=summary_store,
        tool_gateway=tool_gateway,
        task_store=task_store,
        search_service=search_service,
    )
    handler = make_handler(state, authenticator=authenticator)
    return ThreadingHTTPServer((config.host, config.port), handler), state


def main(argv=None):
    """命令行入口：解析端口参数，启动服务，按 Ctrl+C 停止。"""
    parser = argparse.ArgumentParser(description="个人多模型 Agent 本地网页入口")
    parser.add_argument("--host", default=None, help="监听地址，默认只监听本机 127.0.0.1；不传时看环境变量 AGENT_WEB_HOST")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认 8000；不传时看环境变量 AGENT_WEB_PORT")
    args = parser.parse_args(argv)
    # 优先级：命令行参数 > 环境变量 > 默认值。
    env_config = ServerConfig.from_env()
    config = ServerConfig(
        host=args.host or env_config.host,
        port=args.port if args.port is not None else env_config.port,
    )
    # 先做监听地址检查：非回环地址又没配身份验证时拒绝启动。
    # 这一步放在建 mem0 之前——建 mem0 要加载 embedding 模型（约 25 秒），
    # 启动注定失败时没必要先付这个代价；检查逻辑和 create_server 里的
    # 同一条（main 里 authenticator 恒为 None），create_server 里还会再查一次。
    if not _is_loopback_host(config.host):
        print(
            f"启动失败：监听地址 {config.host} 不是本机回环地址，但没有配置身份验证函数；"
            "为防止无身份验证的局域网访问，拒绝启动。"
            "如需开放局域网，请先接上 authenticator 做身份验证。",
            file=sys.stderr, flush=True,
        )
        return 1
    # 长期记忆：常驻服务启动时建一个 Mem0Store（只建一次），每轮提问前
    # 语义检索、回答后自动存。建不好（没 Key、mem0 没装、记忆库坏了）
    # 时返回 None：只警告一句，服务照常启动、不带记忆运行——记忆库出问题
    # 不能把服务搞崩。建失败的具体原因，create_mem0_store 已记进日志。
    tracker = BudgetTracker()
    mem0_store = create_mem0_store(tracker)
    if mem0_store is None:
        print("⚠️ mem0 记忆未启用，本次不带记忆运行。（原因见日志）", file=sys.stderr, flush=True)
    else:
        # Plan C：把十二的名字和 SOUL.md 以“已确认”种进长期记忆
        # （只加没有的，不覆盖已有的；失败只记日志、不拦启动）。
        try:
            seeded = mem0_store.ensure_identity_memories()
            added = [
                key
                for key, outcome in (seeded or {}).items()
                if (outcome or {}).get("status") == "added"
            ]
            if added:
                print(f"长期记忆已确认：{', '.join(added)}。", flush=True)
        except Exception:
            logger.warning("种身份记忆失败：本次不带名字/SOUL 记忆启动。", exc_info=True)
    # 对话摘要：常驻服务启动时建一个 ConversationSummaryStore（只建一次），
    # 每轮后历史超上限就总结压缩、记一笔最近对话，开新对话/停服务时总结保存。
    # 建不好时不带摘要运行，服务照常启动。
    summary_store = None
    try:
        summary_store = ConversationSummaryStore(
            summarizer=make_deepseek_summarizer(budget_tracker=tracker)
        )
    except Exception:
        logger.warning("对话摘要未启用：建摘要库失败。", exc_info=True)
    try:
        server, _state = create_server(
            config=config,
            budget_tracker=tracker,
            mem0_store=mem0_store,
            summary_store=summary_store,
        )
    except ValueError as exc:
        # 非回环地址又没配身份验证时，create_server 会拒绝启动；
        # 这里只打印一句清楚的原因就退出，不甩一堆堆栈给用户看。
        print(f"启动失败：{exc}", file=sys.stderr, flush=True)
        return 1
    print(f"本地网页入口已启动：http://{config.host}:{config.port}", flush=True)
    print("在浏览器打开上面的地址就能对话，按 Ctrl+C 停止服务。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    # Plan C：服务停之前把待定区里最后一轮提交——否则最后一轮
    # 永远待在内存里。撤回过的那轮在撤回时已经丢掉，不会捞回来。
    if mem0_store is not None:
        try:
            mem0_store.flush_pending()
        except Exception:
            logger.warning("停服务时提交待定记忆失败。", exc_info=True)
    # 对话摘要：服务停之前把当前对话总结保存，不然这段对话的尾巴就丢了。
    try:
        _state.flush_summary()
    except Exception:
        logger.warning("停服务时总结对话失败。", exc_info=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
