# providers/__init__.py（模块初始化文件）
# 作用：让 providers 成为一个 Python 包（Package），并导出已接入的模型。
# 未来接入 GPT、Gemini、Grok 时，在这里同样导出它们，
# 例如：from .gpt import GPTProvider
# 这样 main.py 和 router.py 只要从这里取模型，不用管每个模型文件在哪里。

from .deepseek import DeepSeekProvider
from .muse import MuseProvider

__all__ = ["DeepSeekProvider", "MuseProvider"]
