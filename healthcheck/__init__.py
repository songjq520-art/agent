"""十二可移植全身体检模块。"""
from .core import Config, Finding, Unconnected
from .checks import Healthcheck
from .repair import Repair, Candidate, safe_repair

__all__ = ["Config", "Finding", "Unconnected", "Healthcheck", "Repair", "Candidate", "safe_repair"]
