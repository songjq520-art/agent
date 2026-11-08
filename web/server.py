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
#   GET  /api/status  → 模型状态 + 预算（JSON）
#   POST /api/chat    → 发消息，请求 JSON {"message": "..."}，
#                        返回 JSON {"reply": 回答, "model": 模型名,
#                        "mode": "real" 或 "demo", "cost_usd": 成本,
#                        "usage": Token 用量, "new_proposals": 本次新建的提议}；
#                        出错时返回 {"error": 原因}。
#   POST /api/reset   → 清空本轮对话历史，返回 {"ok": true}。
#   GET  /api/proposals → 待确认的记忆提议列表，返回 {"proposals": [...]}。
#   POST /api/proposals/<id>/confirm → 确认一条提议，把记忆写进长期记忆文件，
#                        返回 {"ok": true, "memory": 存进去的记忆}；提议不存在时 404，
#                        保存失败时 500（记忆文件损坏、磁盘 I/O 错误等），提议保留在待确认列表。
#   POST /api/proposals/<id>/reject → 拒绝一条提议，不保存，返回 {"ok": true}；
#                        提议不存在时 404。
#
# 注意（给以后看代码的人）：
#   - 对话历史只保存在内存里（ChatState.messages），重启服务后清空；
#     本阶段不做持久化，README 里也写明了。
#   - 服务用 ThreadingHTTPServer，每个请求一个线程；ChatState 里有一把锁，
#     问、清空、读状态都走这把锁，一次只处理一个请求，历史和预算不会乱。
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

# 让直接运行 python web/server.py 也能找到项目根目录的 main.py、config.py。
# 用 python -m web.server 运行时本来就不需要这一行，加上也不影响。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MONTHLY_BUDGET_USD, BudgetTracker, get_model_status  # noqa: E402
from main import run_question  # noqa: E402
from memory.memories import MemoryStore, ensure_first_memories  # noqa: E402
from memory.proposals import ProposalStore  # noqa: E402

logger = logging.getLogger("agent")

# INDEX_PATH（聊天页面文件）：按 server.py 的位置找，不依赖当前工作目录。
INDEX_PATH = Path(__file__).resolve().parent / "index.html"


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
    """一次服务运行里的对话状态：共用的预算跟踪器 + 对话历史 + 长期记忆库 + 待确认提议。

    做法和 main.py 的 chat_loop() 一样：整个服务进程只用一个
    BudgetTracker 和一个 messages 历史（user 和 assistant 消息按时间顺序）。
    成功拿到回答后，才把本轮 user 问题和 assistant 回答追加进历史；
    演示模式或调用失败时，历史不动。
    对话历史只保存在内存里，重启服务后清空；本阶段不做持久化。
    长期记忆：memory_store 不传时不带记忆；传了时，启动时种下前三条记忆
    （只在空库时写一次），每轮提问先搜相关记忆带给模型（见 main.run_question）。
    记忆文件只保存在本地（memory/memories.json），不上传。
    记忆提议：每轮聊天成功后，跑一次独立的记忆提取（见 main.run_question
    的 extract_proposals），把候选建成真实提议放进 proposal_store，
    等用户在页面上确认或拒绝。提议只存在内存里，重启服务后未确认的
    提议会丢弃；已确认的记忆已经写进 memories.json，不受影响。
    """

    def __init__(self, budget_tracker=None, memory_store=None, proposal_store=None):
        self.tracker = budget_tracker or BudgetTracker()
        self.messages = []
        self.memory_store = memory_store
        self.proposal_store = proposal_store or ProposalStore()
        if self.memory_store is not None:
            # 启动时确保前三条记忆已存（只在空库时写一次）；记忆库坏了也不影响服务。
            if getattr(self.memory_store, "corrupted", False):
                logger.warning(
                    "记忆文件异常：%s，本次服务不带记忆运行。",
                    self.memory_store.corruption_reason,
                )
            else:
                try:
                    ensure_first_memories(self.memory_store)
                except Exception:
                    logger.warning("初始化记忆失败，本次服务不带记忆运行。", exc_info=True)
        # _lock（并发锁）：服务用 ThreadingHTTPServer，每个请求一个线程。
        # 没有这把锁，两个 /api/chat 同时跑会互相污染历史（你问你的、他答他的，
        # 追加顺序乱掉），还可能同时通过预算检查、多记一笔费用。
        # 所以问、清空、读状态都走这把锁，一次只处理一个请求；本机单人用，
        # 等一会儿没关系，顺序和预算不会错。调用模型时会一直拿着锁，
        # 期间别的请求（包括看状态、清空）会等一等，这是为了保证一致性。
        self._lock = threading.Lock()

    def reset(self):
        """清空本轮对话历史（预算不动，只清聊天记录）。"""
        with self._lock:
            self.messages = []

    def status(self):
        """给页面状态栏用的：模型状态 + 本月预算使用情况。"""
        with self._lock:
            return {
                "model_status": get_model_status(),
                "monthly_budget_usd": MONTHLY_BUDGET_USD,
                "total_cost_usd": round(self.tracker.total_cost_usd, 6),
                "remaining_usd": round(self.tracker.remaining_usd(), 6),
                "budget_corrupted": bool(getattr(self.tracker, "corrupted", False)),
                "unknown_cost": bool(self.tracker.unknown_cost),
            }

    def list_proposals(self):
        """返回所有待确认的记忆提议（给页面列表用）。"""
        with self._lock:
            return self.proposal_store.list_pending()

    def confirm_proposal(self, proposal_id):
        """确认一条提议：把记忆写进长期记忆文件。返回存进去的记忆字典。

        没有记忆库时报 RuntimeError；找不到提议或提议已处理时报 ValueError；
        保存失败时报异常，提议保留在待确认列表（用户可稍后重试）。
        """
        with self._lock:
            if self.memory_store is None:
                raise RuntimeError("长期记忆未启用，无法保存。")
            return self.proposal_store.confirm(proposal_id, self.memory_store)

    def reject_proposal(self, proposal_id):
        """拒绝一条提议：从待确认列表移除，不保存。返回 True；找不到返回 False。"""
        with self._lock:
            return self.proposal_store.reject(proposal_id)

    def ask(self, prompt):
        """问一句话，返回给页面展示的字典。

        成功时：{"reply": 回答, "model": 模型名, "mode": "real",
                  "cost_usd": 成本, "usage": Token 用量}，
        并把本轮问答追加进历史。
        演示模式（无 Key）时：{"reply": 提示, "model": None,
        "mode": "demo", "cost_usd": None, "usage": None}，历史不动。
        调用失败时：直接抛 RuntimeError，调用方转成错误返回，历史不动。
        聊天成功后还会跑一次独立的记忆提取（有记忆库时）：把候选建成
        真实提议放进 proposal_store，等用户确认或拒绝；提取失败不影响
        已经拿到的回答（见 main.run_question）。
        """
        with self._lock:
            outcome = run_question(
                prompt,
                budget_tracker=self.tracker,
                messages=self.messages,
                memory_store=self.memory_store,
                extract_proposals=True,
                proposal_store=self.proposal_store,
            )
            if outcome["mode"] == "demo":
                return {
                    "reply": outcome["message"],
                    "model": None,
                    "mode": "demo",
                    "cost_usd": None,
                    "usage": None,
                }
            result = outcome["result"]
            self.messages.append({"role": "user", "content": prompt})
            self.messages.append({"role": "assistant", "content": result["text"]})
            return {
                "reply": result["text"],
                "model": result["model"],
                "mode": "real",
                "cost_usd": result.get("cost_usd"),
                "usage": result.get("usage"),
                "new_proposals": outcome.get("new_proposals") or [],
            }


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
        """处理网页请求的类：只认下面四个地址，其他一律 404。"""

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

        def _do_get(self):
            if not self._check_auth():
                self._send_json({"error": "未授权：请先通过身份验证。"}, status=401)
                return
            if self.path in ("/", "/index.html"):
                self._send_html()
            elif self.path == "/api/status":
                self._send_json(state.status())
            elif self.path == "/api/proposals":
                self._send_json({"proposals": state.list_proposals()})
            else:
                # 中文错误信息放 JSON 正文里（UTF-8 安全），不用 send_error，
                # 因为它的状态行只能用 latin-1，中文放不进去。
                self._send_json({"error": "没有这个地址。"}, status=404)

        def _do_proposal_action(self, action):
            """处理提议确认/拒绝：路径形如 /api/proposals/<id>/confirm 或
            /api/proposals/<id>/reject。action 是 "confirm" 或 "reject"。"""
            parts = self.path.split("/")
            # ["", "api", "proposals", "<id>", "confirm"|"reject"]
            if len(parts) != 5 or not parts[3]:
                self._send_json({"error": "没有这个地址。"}, status=404)
                return
            proposal_id = parts[3]
            if action == "confirm":
                try:
                    saved = state.confirm_proposal(proposal_id)
                except ValueError as exc:
                    # 提议不存在或已处理过：404。
                    self._send_json({"error": str(exc)}, status=404)
                    return
                except Exception as exc:
                    # 保存失败：记忆文件损坏、磁盘 I/O 错误（权限不足、磁盘满等）
                    # 都会到这里。提议保留在待确认列表（confirm 只在保存成功
                    # 后才移除），用户可稍后重试；前端回 500 并明确提示是
                    # 保存失败，不把堆栈泄露出去。
                    # 注意：这里必须 catch Exception，不能只 catch RuntimeError——
                    # 真实磁盘 I/O 错误是 OSError，不是 RuntimeError。
                    traceback.print_exc()
                    self._send_json({"error": f"保存失败：{exc}"}, status=500)
                    return
                self._send_json({"ok": True, "memory": saved})
                return
            # reject
            if state.reject_proposal(proposal_id):
                self._send_json({"ok": True})
            else:
                self._send_json({"error": f"没有这个提议：{proposal_id}"}, status=404)

        def do_POST(self):  # noqa: N802 - 标准库要求的名字
            self._handle_safely(self._do_post)

        def _do_post(self):
            if not self._check_auth():
                self._send_json({"error": "未授权：请先通过身份验证。"}, status=401)
                return
            if self.path == "/api/reset":
                state.reset()
                self._send_json({"ok": True})
                return
            if self.path.startswith("/api/proposals/"):
                action = self.path.rsplit("/", 1)[-1]
                if action in ("confirm", "reject"):
                    self._do_proposal_action(action)
                    return
                self._send_json({"error": "没有这个地址。"}, status=404)
                return
            if self.path != "/api/chat":
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


def create_server(config=None, budget_tracker=None, authenticator=None, memory_store=None, proposal_store=None):
    """建一个网页服务（不启动），方便测试用临时端口和临时预算文件。

    config 是 ServerConfig；不传时用环境变量/默认值（ServerConfig.from_env()）。
    authenticator 是身份验证钩子（见 make_handler），默认不校验。
    memory_store 是长期记忆库（MemoryStore），不传时不带记忆；传了时，
    ChatState 启动时种下前三条记忆，每轮提问先搜相关记忆带给模型。
    proposal_store 是待确认提议暂存（ProposalStore），不传时新建一个；
    每轮聊天成功后跑一次记忆提取，候选建成提议放进这里，等用户确认或拒绝。
    测试时可传一个用临时文件的记忆库，避免碰真实记忆文件。

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
        memory_store=memory_store,
        proposal_store=proposal_store,
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
    try:
        # 长期记忆库：真实服务建一个（memory/memories.json），启动时种下
        # 前三条记忆，每轮提问先搜相关记忆带给模型。
        # 提议暂存：真实服务建一个（只在内存里），每轮聊天成功后跑一次
        # 记忆提取，候选建成提议放进这里，等用户在页面上确认或拒绝。
        server, _state = create_server(
            config=config,
            memory_store=MemoryStore(),
            proposal_store=ProposalStore(),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
