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
# 接口（给页面用的）：
#   GET  /            → 聊天页面（web/index.html）
#   GET  /api/status  → 模型状态 + 预算（JSON）
#   POST /api/chat    → 发消息，请求 JSON {"message": "..."}，
#                        返回 JSON {"reply": 回答, "model": 模型名,
#                        "mode": "real" 或 "demo", "cost_usd": 成本,
#                        "usage": Token 用量}；出错时返回 {"error": 原因}。
#   POST /api/reset   → 清空本轮对话历史，返回 {"ok": true}。

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 让直接运行 python web/server.py 也能找到项目根目录的 main.py、config.py。
# 用 python -m web.server 运行时本来就不需要这一行，加上也不影响。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MONTHLY_BUDGET_USD, BudgetTracker, get_model_status  # noqa: E402
from main import run_question  # noqa: E402

# DEFAULT_HOST / DEFAULT_PORT（默认地址和端口）：只监听本机，不对外开放。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# INDEX_PATH（聊天页面文件）：按 server.py 的位置找，不依赖当前工作目录。
INDEX_PATH = Path(__file__).resolve().parent / "index.html"


class ChatState:
    """一次服务运行里的对话状态：共用的预算跟踪器 + 对话历史。

    做法和 main.py 的 chat_loop() 一样：整个服务进程只用一个
    BudgetTracker 和一个 messages 历史（user 和 assistant 消息按时间顺序）。
    成功拿到回答后，才把本轮 user 问题和 assistant 回答追加进历史；
    演示模式或调用失败时，历史不动。
    """

    def __init__(self, budget_tracker=None):
        self.tracker = budget_tracker or BudgetTracker()
        self.messages = []

    def reset(self):
        """清空本轮对话历史（预算不动，只清聊天记录）。"""
        self.messages = []

    def status(self):
        """给页面状态栏用的：模型状态 + 本月预算使用情况。"""
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


def make_handler(state):
    """按给定的对话状态，生成一个请求处理类。

    这样写是为了让测试能传一个用临时预算文件的 ChatState，
    不碰真实的 logs/budget_usage.json。
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

        def do_GET(self):  # noqa: N802 - 标准库要求的名字
            if self.path in ("/", "/index.html"):
                self._send_html()
            elif self.path == "/api/status":
                self._send_json(state.status())
            else:
                # 中文错误信息放 JSON 正文里（UTF-8 安全），不用 send_error，
                # 因为它的状态行只能用 latin-1，中文放不进去。
                self._send_json({"error": "没有这个地址。"}, status=404)

        def do_POST(self):  # noqa: N802 - 标准库要求的名字
            if self.path == "/api/reset":
                state.reset()
                self._send_json({"ok": True})
                return
            if self.path != "/api/chat":
                self._send_json({"error": "没有这个地址。"}, status=404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._send_json({"error": "请求不是合法的 JSON。"}, status=400)
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


def create_server(host=DEFAULT_HOST, port=DEFAULT_PORT, budget_tracker=None):
    """建一个网页服务（不启动），方便测试用临时端口和临时预算文件。"""
    state = ChatState(budget_tracker=budget_tracker)
    handler = make_handler(state)
    return ThreadingHTTPServer((host, port), handler), state


def main(argv=None):
    """命令行入口：解析端口参数，启动服务，按 Ctrl+C 停止。"""
    parser = argparse.ArgumentParser(description="个人多模型 Agent 本地网页入口")
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址，默认只监听本机 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口，默认 8000")
    args = parser.parse_args(argv)
    server, _state = create_server(host=args.host, port=args.port)
    print(f"本地网页入口已启动：http://{args.host}:{args.port}")
    print("在浏览器打开上面的地址就能对话，按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
