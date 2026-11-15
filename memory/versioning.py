# memory/versioning.py（版本文件：embedding 模型/维度/mem0 版本校验）
# 从 memory/mem0_store.py 拆出来，行为不变。

import json
from pathlib import Path
from memory.timeutil import _today

# CUSTOM_INSTRUCTIONS_VERSION（自定义提取规则的版本号）：
# 规则文本改了就改这个版本号，方便以后追踪“哪版规则存的记忆”。
# 2026-11-09 改到第二版：加了“价值判断”（只存值得长期记的）和
# “不存未确认/不存别人的事/不存没被认可的 AI 观点”（见下文第 9–12 条）。
# 2026-11-09 Plan C 改到第三版：加第 13 条（玩笑/夸张/不可能的说法不存）。
CUSTOM_INSTRUCTIONS_VERSION = "2026-11-09-plan-c"

# ---------- 版本文件（embedding 模型 + 维度记录） ----------
# 换 embedding 模型必须全量重嵌：维度不同时 Chroma 直接报硬错，
# 旧库不能混用。所以每次建库时，把用的模型名、维度、mem0 版本
# 写进版本文件；启动时先对一下，对不上就拒绝启动、不悄悄混用，
# 让用户走重嵌迁移（tools/reembed_memories.py），而不是等 Chroma 报错。

def _read_version_file(version_path):
    """读版本文件，读不到或格式不对时返回 None（当作“没有版本记录”）。"""
    try:
        data = json.loads(Path(version_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data

def _write_version_file(version_path, embedding_model, embedding_dims, mem0_version):
    """写版本文件：模型名、维度、mem0 版本、写入日期。"""
    path = Path(version_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "embedding_dims": embedding_dims,
                "mem0_version": mem0_version,
                "custom_instructions_version": CUSTOM_INSTRUCTIONS_VERSION,
                "updated": _today(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

def _check_version_file(version_path, chroma_path, embedding_model, embedding_dims, mem0_version):
    """校验版本文件与当前配置是否一致。

    规则：
    - 版本文件不存在、且 Chroma 目录也不存在或为空：全新库，写版本文件，通过。
    - 版本文件不存在、但 Chroma 目录里已有数据：来历不明的旧库，拒绝启动
      （不知道它是用哪个模型嵌的，混用会坏数据）。
    - 版本文件存在、模型名或维度或 mem0 版本对不上：拒绝启动，提示走重嵌迁移。
    - 对得上：通过。
    拒绝时抛 RuntimeError，信息里写清楚原因和该怎么做。
    """
    version_path = Path(version_path)
    chroma_path = Path(chroma_path)
    recorded = _read_version_file(version_path)
    chroma_has_data = chroma_path.exists() and any(chroma_path.iterdir())
    if recorded is None:
        if chroma_has_data:
            raise RuntimeError(
                "记忆库版本不明：Chroma 目录里已有数据，但没有版本文件，"
                "不知道它是用哪个 embedding 模型建的，不能直接混用。"
                "请先备份 memory/，再用 tools/reembed_memories.py 重嵌，"
                "或删掉 memory/mem0_chroma 后重新迁移。"
            )
        _write_version_file(version_path, embedding_model, embedding_dims, mem0_version)
        return
    if (
        recorded.get("embedding_model") != embedding_model
        or recorded.get("embedding_dims") != embedding_dims
        or recorded.get("mem0_version") != mem0_version
    ):
        raise RuntimeError(
            "记忆库版本不匹配：记录的是 "
            f"{recorded.get('embedding_model')}（{recorded.get('embedding_dims')} 维，"
            f"mem0 {recorded.get('mem0_version')}），当前配置是 "
            f"{embedding_model}（{embedding_dims} 维，mem0 {mem0_version}）。"
            "换 embedding 模型必须全量重嵌，不能直接混用。"
            "请用 tools/reembed_memories.py 重嵌，或删掉 memory/mem0_chroma 后重新迁移。"
        )

