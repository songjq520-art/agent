# tools/errors.py（第一批工具共用的异常）
# 作用：工具层只用这几个异常，调用方按类型决定怎么处理：
#   - PermissionDenied：硬禁 / 越界，绝不执行、不用问。
#   - NeedsConfirmation：Tier 1 或重要操作，需要 BOOS 点头，description
#     里写清楚“要干什么”，方便拿给 BOOS 看。
#   - QuotaExhausted：某个搜索服务的免费额度用完了（402/429），调度器
#     看到它就切备选，不当成普通失败。
#   - ToolError：其他工具错误（网络不通、服务报错、参数不对等）。

# 异常只带信息，不带任何 Key 或密钥。


class ToolError(Exception):
    """工具层通用错误。"""


class PermissionDenied(ToolError):
    """硬禁或越界：绝不执行，也不用问（Tier 2、路径穿越、密钥文件）。"""


class NeedsConfirmation(ToolError):
    """需要 BOOS 确认后才能做：description 写清楚要干什么。"""

    def __init__(self, description):
        super().__init__(description)
        self.description = description


class QuotaExhausted(ToolError):
    """某个服务的免费额度用完了（HTTP 402/429）：调度器切备选。"""

    def __init__(self, provider, status, message):
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.message = message
