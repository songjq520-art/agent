# tools/__init__.py（十二第一批工具包）
# 作用：上网搜索、读写文件、跑程序、清理垃圾——十二能动手干活的第一批工具。
#   - tools/search.py：三个搜索服务 + 统一调度器（AnySearch / Tavily / Parallel MCP）。
#   - tools/files.py：固定工作区里的读写文件（路径围栏、密钥黑名单、.env 只写不读）。
#   - tools/commands.py：结构化跑命令（Tier 2 硬禁 / Tier 1 需确认 / Tier 0 直接跑）。
#   - tools/cleanup.py：清理垃圾（按种类报大白话、按种类确认、删完核对）。
#   - tools/permissions.py：所有工具共用的权限判断（路径 + 命令分级）。
#   - tools/audit.py：操作审计日志（谁、干了什么、确认过没有、改前改后哈希）。
#   - tools/cli.py：命令行入口（python -m tools.cli ...），不用改 main.py 也能试工具。
#
# 安全定位（诚实说明）：代码层防的是误操作和提示注入下的意外损坏，
# 防不住存心绕过（终端能绕过文件工具的写保护）；真正的隔离靠 OS 层，
# 第一批笔记本靠分级 + 人工确认，Box 上先测 gVisor。详见 tools/README.md。

__version__ = "0.1.0"
