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
#                        "usage": Token 用量}；出错时返回 {"error": 原因}。
#   POST /api/reset   → 清空本轮对话历史，返回 {"ok": true}。
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
import json
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
    """一次服务运行里的对话状态：共用的预算跟踪器 + 对话历史。

    做法和 main.py 的 chat_loop() 一样：整个服务进程只用一个
    BudgetTracker 和一个 messages 历史（user 和 assistant 消息按时间顺序）。
    成功拿到回答后，才把本轮 user 问题和 assistant 回答追加进历史；
    演示模式或调用失败时，历史不动。
    对话历史只保存在内存里，重启服务后清空；本阶段不做持久化。
    """

    def __init__(self, budget_tracker=None):
        self.tracker = budget_tracker or BudgetTracker()
        self.messages = []
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

    def ask(self, prompt):
        """问一句话，返回给页面展示的字典。

        成功时：{"reply": 回答, "model": 模型名, "mode": "real",
                  "cost_usd": 成本, "usage": Token 用量}，
        并把本轮问答追加进历史。
        演示模式（无 Key）时：{"reply": 提示, "model": None,
        "mode": "demo", "cost_usd": None, "usage": None}，历史不动。
        调用失败时：直接抛 RuntimeError，调用方转成错误返回，历史不动。
        """
        with self._lock:
            outcome = run_question(
                prompt, budget_tracker=self.tracker, messages=self.messages
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
            if self.path == "/api/reset":
                state.reset()
                self._send_json({"ok": True})
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
                self._send_json({"error": f"调用失败：{exc}"}, status=502)
                return
            self._send_json(reply)

    return ChatHandler


def create_server(config=None, budget_tracker=None, authenticator=None):
    """建一个网页服务（不启动），方便测试用临时端口和临时预算文件。

    config 是 ServerConfig；不传时用环境变量/默认值（ServerConfig.from_env()）。
    authenticator 是身份验证钩子（见 make_handler），默认不校验。
    """
    if config is None:
        config = ServerConfig.from_env()
    state = ChatState(budget_tracker=budget_tracker)
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
    server, _state = create_server(config=config)
    print(f"本地网页入口已启动：http://{config.host}:{config.port}", flush=True)
    print("在浏览器打开上面的地址就能对话，按 Ctrl+C 停止服务。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
