# tools/migrate_from_v1.py（一次性迁移脚本：把旧记忆搬进 mem0）
# 什么时候用：第一次从 v1（memories.json）换到 mem0 时，只跑一次。
# 用法（在项目根目录下跑）：
#   python tools/migrate_from_v1.py
# 做三件事：
#   1. 建一个 Mem0Store（建不好就报错退出，并说清原因）；
#   2. 调 store.migrate_from_v1()：读 memory/memories.json 原文迁移，
#      旧预算记忆（“4 美元提醒，5 美元停”那句）会按新规矩改写，其他一字不改，
#      已经在库里的不会重复存；迁过来的旧记忆都记 source="user"、
#      user_confirmed=True（v1 时代是先确认再存），有 date 的用它当形成时间；
#   3. 调 store.ensure_first_memories()：库里还是空时，种下首批三条记忆。
# 跑完打印：这次新存了几条、现在库里一共几条。
# 注意：
#   - 跑之前先停掉网页服务（别让它在迁移时写记忆，写乱了不好查）。
#   - 旧文件 memory/memories.json 跑完不要删，留着当备份；以后新记忆只存 mem0，不再写它。
#   - 第一次跑要加载本地 embedding 模型，约 25 秒，别急着关。
#   - 如果旧文件不存在（比如新电脑、没有旧记忆），脚本不会报错，
#     直接种首批三条记忆就结束。

import sys
from pathlib import Path

# 让脚本能 import 项目根目录下的 memory / config（不管从哪跑，
# 都按脚本自己的位置算出项目根目录）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.mem0_store import Mem0Store  # noqa: E402


def main():
    print("=== 从 v1 迁移旧记忆到 mem0 ===")
    print("提醒：跑之前请先停掉网页服务（别让它在迁移时写记忆）。")
    print("正在建记忆库（第一次要加载本地模型，约 25 秒，请稍等）……")
    try:
        store = Mem0Store()
    except Exception as exc:
        print(f"记忆库建不起来，迁移退出。原因：{exc}")
        print("常见原因：.env 里没有填 DEEPSEEK_API_KEY；mem0 没装好"
              "（先跑 pip install -r requirements.txt）；"
              "版本文件对不上（换过 embedding 模型时，用 tools/reembed_memories.py 重嵌）；"
              "Chroma 目录坏了（先备份 memory/，再删掉 memory/mem0_chroma 重新迁移）。")
        return 1

    # 第 1 步：读旧 memories.json，原样迁进 mem0。
    migrated = store.migrate_from_v1()
    print(f"从旧记忆文件新存入 {migrated} 条记忆。")

    # 第 2 步：库里还是空时，种下首批三条记忆。
    seeded = store.ensure_first_memories()
    if seeded:
        print(f"库里还是空，已种下首批 {seeded} 条记忆。")

    # 第 3 步：报一下现在库里一共几条。
    total = len(store.get_all())
    print(f"迁移完成：本次新存 {migrated + seeded} 条，现在库里一共 {total} 条记忆。")
    print("旧文件 memory/memories.json 请留着当备份，不要删。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
