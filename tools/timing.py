# tools/timing.py（第 0 步地基：计时小工具）
# 作用：量一轮问答里每一截花了多久（搜记忆、调模型、存记忆……），
# 让 26 号“回答慢、不知道慢在哪”有数可查。只量时间、不改行为：
# 量不到的截（比如异常提前返回）就不记那一项，不拦聊天。
# 用法：
#     timer = Timer()
#     with timer.measure("model_ms"):
#         ...调模型...
#     timer.as_dict()  # {"model_ms": 1234.5, ...}（单位：毫秒）
# 注意：同一截多次量时取累计和（比如循环里多轮模型调用都记进 model_ms）。

import time
from contextlib import contextmanager


class Timer:
    """攒一轮里各截花的时间（毫秒）。线程不安全——每轮自己 new 一个。"""

    def __init__(self):
        self._spans = {}

    @contextmanager
    def measure(self, name):
        """量一段代码花的时间，退出时记进 name（毫秒，累计）。"""
        start = time.perf_counter()
        try:
            yield
        finally:
            spent = (time.perf_counter() - start) * 1000.0
            self._spans[name] = self._spans.get(name, 0.0) + spent

    def record(self, name, ms):
        """直接记一截已知的耗时（毫秒，累计）。"""
        self._spans[name] = self._spans.get(name, 0.0) + float(ms)

    def as_dict(self):
        """返回 {截名: 毫秒}，毫秒保留 1 位小数。"""
        return {name: round(ms, 1) for name, ms in self._spans.items()}

    def total_ms(self):
        """各截耗时合计（毫秒）。"""
        return round(sum(self._spans.values()), 1)
