# healthcheck_integration/factory.py（体检配置与装配）
#
# 把 healthcheck 模块的 Config 按十二的实际环境拼出来：
# - 慢门走 pytest 全套入口（本仓库的测试体系），解释器用当前进程
#   的 sys.executable（服务里就是服务自己的 venv python）；
# - 存档目录必须在仓库外（模块 Config.validate 的硬要求）；
# - repair_paths 留空：代码自修默认无白名单（只提案都提不了改动，
#   要放开是 BOOS 以后单独定的事）；protected_paths 钉住 memory
#   （核心记忆/规矩所在，优先于一切白名单）；
# - permission_paths 留空：Windows 上 POSIX 权限位无语义（模块已
#   按平台跳过该判据），密钥只查环境变量名在不在；
# - max_seconds 给足 1800：一次完整体检 = 整仓复制 + 全套测试
#   （约 3 分钟）+ 记忆自检的模型抽查，远超模块默认 120 秒。

import os
import sys
from pathlib import Path

from healthcheck import Config, Healthcheck

from .adapter import ShierHealthcheckAdapter


def build_config(repo_root, archive_dir, python_executable=None,
                 weekly_interval_seconds=None):
    exe = python_executable or sys.executable
    if weekly_interval_seconds is None:
        raw = os.environ.get("SHIER_HEALTHCHECK_INTERVAL_SECONDS", "")
        try:
            weekly_interval_seconds = float(raw) if raw.strip() else 7 * 24 * 3600
        except ValueError:
            weekly_interval_seconds = 7 * 24 * 3600
    suite = [exe, "-m", "pytest", "-q", "--tb=short", "-ra", "--no-header"]
    return Config(
        repo=Path(repo_root),
        archive=Path(archive_dir),
        test_command=suite,
        allowed_commands=[suite],
        full_suite=True,
        max_seconds=1800,
        max_steps=8,
        max_cost=0.1,
        disk_min_bytes=1024 ** 3,
        secret_env_names=["DEEPSEEK_API_KEY"],
        permission_paths=[],
        repair_paths=[],
        protected_paths=["memory"],
        weekly_interval_seconds=float(weekly_interval_seconds),
    )


def build_adapter(repo_root, memory_dir=None, store=None,
                  server_base_url=None, component=None, budget_tracker=None):
    return ShierHealthcheckAdapter(
        repo_root,
        memory_dir=memory_dir,
        store=store,
        server_base_url=server_base_url,
        component=component,
        budget_tracker=budget_tracker,
    )


def build_proposer_registry():
    """集成层提案白名单（路线①）：注册名 → 「模块路径:函数名」。

    目前只有安全默认 no_candidate（永不提代码修改，与 repair_paths
    默认空一致）。新增提案函数先落 healthcheck_integration/proposers.py
    再在这里登记，未登记的名字执行器在启动子进程前就拒绝。"""
    from healthcheck import ProposerRegistry
    registry = ProposerRegistry()
    registry.register(
        "no_candidate", "healthcheck_integration.proposers:no_candidate")
    return registry


def build_proposer(name="no_candidate", timeout=None, registry=None):
    """装配具名提案句柄：Repair.execute 收到它就走子进程隔离执行器。

    Windows 工厂路径只发具名句柄，不再用同进程线程执行器跑提案
    （线程执行器仅保留给裸 callable 的旧契约兼容）。"""
    from healthcheck import NamedProposer
    return NamedProposer(
        name, timeout=timeout,
        registry=registry or build_proposer_registry())


def build_healthcheck(repo_root, archive_dir, **adapter_kwargs):
    """一步装配：返回 (Healthcheck, adapter)。"""
    python_executable = adapter_kwargs.pop("python_executable", None)
    weekly_interval_seconds = adapter_kwargs.pop("weekly_interval_seconds", None)
    adapter = build_adapter(repo_root, **adapter_kwargs)
    config = build_config(repo_root, archive_dir,
                          python_executable=python_executable,
                          weekly_interval_seconds=weekly_interval_seconds)
    return Healthcheck(config, adapter), adapter
