# healthcheck_integration（体检模块并入十二的集成层）
#
# healthcheck/ 是贵妃交付的独立模块（只认 Adapter 协议，不认识十二）；
# 本包是十二这一侧的可信适配器：把十二真实的记忆自检、待问问题库、
# DeepSeek 判官/聊天、备份状态、维护调度接进协议里。模块本身不改
# 十二的任何算法——记忆自检仍走 store.run_self_check 原样调用。
from .adapter import (
    APPROVAL_KIND,
    APPROVAL_OPTIONS,
    ShierHealthcheckAdapter,
    SubprocessComponent,
)
from .factory import build_adapter, build_config, build_healthcheck

__all__ = [
    "APPROVAL_KIND",
    "APPROVAL_OPTIONS",
    "ShierHealthcheckAdapter",
    "SubprocessComponent",
    "build_adapter",
    "build_config",
    "build_healthcheck",
]
