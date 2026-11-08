# tools/reembed_memories.py（全量重嵌迁移脚本）
# 什么时候用：换了 embedding 模型（或维度、mem0 大版本）之后。
# 换模型必须全量重嵌——旧向量是旧模型算出来的，新模型算出来的向量
# 维度/空间都不一样，混用会坏数据。Mem0Store 启动时发现版本文件
# 对不上就会拒绝启动，提示用这个脚本。
# 做法：
#   1. 用旧模型读出库里所有记忆（先备份一份 JSON，防中途出错丢数据）；
#   2. 删掉旧 Chroma 目录和版本文件（历史审计 history.db 不动，留着）；
#   3. 用新模型重建库，把记忆一条条原文存回去（infer=False，不重新提取）。
# 用法（在项目根目录下跑）：
#   python tools/reembed_memories.py
# 注意：跑之前先停掉网页服务（别让它在重嵌时写记忆）；
#       跑完之后再启动服务，核对一下记忆还在不在。

import json
import shutil
import sys
from datetime import date
from pathlib import Path

# 让脚本能 import 项目根目录下的 memory / config（不管从哪跑，
# 都按脚本自己的位置算出项目根目录）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.mem0_store import (  # noqa: E402
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    MEM0_VERSION,
    Mem0Store,
    _read_version_file,
)

MEMORY_DIR = PROJECT_ROOT / "memory"
CHROMA_DIR = MEMORY_DIR / "mem0_chroma"
VERSION_FILE = MEMORY_DIR / "mem0_versions.json"
HISTORY_DB = MEMORY_DIR / "history.db"


def main():
    print("=== 全量重嵌迁移 ===")
    print(f"新 embedding 模型：{EMBEDDING_MODEL}（{EMBEDDING_DIMS} 维），mem0 {MEM0_VERSION}")
    recorded = _read_version_file(VERSION_FILE)
    if recorded is None:
        print("没有版本文件：库是空的或没有旧数据，不需要重嵌，直接退出。")
        return 0
    old_model = recorded.get("embedding_model")
    old_dims = recorded.get("embedding_dims")
    old_mem0 = recorded.get("mem0_version")
    print(f"旧版本记录：{old_model}（{old_dims} 维，mem0 {old_mem0}）")
    if (
        old_model == EMBEDDING_MODEL
        and old_dims == EMBEDDING_DIMS
        and old_mem0 == MEM0_VERSION
    ):
        print("版本已经一致，不需要重嵌，直接退出。")
        return 0

    # 第 1 步：用旧模型读出所有记忆，先备份。
    print("第 1 步：用旧模型读出所有记忆……")
    old_store = Mem0Store(
        embedding_model=old_model,
        embedding_dims=old_dims,
        mem0_version=old_mem0,
    )
    memories = old_store.get_all()
    print(f"读到 {len(memories)} 条记忆。")
    backup_path = MEMORY_DIR / f"memories-backup-{date.today().isoformat()}.json"
    backup_path.write_text(
        json.dumps([{"id": m["id"], "text": m["text"]} for m in memories], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已备份到：{backup_path}")

    # 第 2 步：删掉旧 Chroma 目录和版本文件（history.db 不动）。
    print("第 2 步：删掉旧 Chroma 目录和版本文件……")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    if VERSION_FILE.exists():
        VERSION_FILE.unlink()
    print("旧库已删（history.db 保留）。")

    # 第 3 步：用新模型重建库，原文存回去。
    print("第 3 步：用新模型重建库并原文存回……")
    new_store = Mem0Store()
    added = 0
    for item in memories:
        result = new_store.add_memory_text(item["text"])
        if result:
            added += 1
    print(f"重嵌完成：{added}/{len(memories)} 条已存回。")
    if added != len(memories):
        print("注意：有记忆没存回去，请对照备份文件检查。")
        return 1
    print("全部完成。启动网页服务后，可以问几句核对记忆还在不在。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
