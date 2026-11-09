# 十二第一批工具（tools/）

十二能动手干活的第一批工具：上网搜索、读写文件、跑程序、清理垃圾。
按 `十二第一批工具新方案.md`（已核实版）落地，结合了 GPT 评审的建议
（统一搜索调度器 + 结果质量检查、共享权限模块 + 审计日志、结构化命令调用、
清理三道保护、失败处理机制），但共享核心保持最小，不先搭大框架。

## 文件一览

| 文件 | 作用 |
|---|---|
| `tools/search.py` | 三个搜索服务 + 统一调度器 |
| `tools/files.py` | 固定工作区里的读写文件 |
| `tools/commands.py` | 结构化跑命令（Tier 分级） |
| `tools/cleanup.py` | 清理垃圾（按种类报、按种类确认、删完核对） |
| `tools/permissions.py` | 所有工具共用的权限判断（路径 + 命令分级） |
| `tools/audit.py` | 操作审计日志（JSONL） |
| `tools/errors.py` | 共用异常 |
| `tools/cli.py` | 命令行入口（不用改 main.py 也能试工具） |

## 搜索：三个服务 + 调度器

- **AnySearch**：日常主力，每天 1,000 次免费；超量返回 HTTP 402（不自动扣费）。
  一般搜索 tag 用 `general.general`（官网文档原话）。没 Key 也能用匿名额度。
- **Tavily**：结构化交叉验证，每月 1,000 credits 免费；只用 `search_depth: basic`
  （1 credit/次），绝不用 advanced（2 credits）和 Research 端点。
  需要 Key（`TAVILY_API_KEY`），没 Key 时这个服务不可用，调度器跳过它。
- **Parallel MCP**：零门槛入口兼兜底，不用注册、不用 Key、不用绑卡；
  走托管 MCP（`https://mcp.parallel.ai/mcp`，没有末尾斜杠），调 `web_search`。
  默认 fast 模式（中文查询用 fast；Turbo 只支持英文和日文）。
- **调度器**（`SearchService`）：普通查询 → AnySearch；结构化 → Tavily；
  主力失败/额度用完/没结果 → 顺着链路切备选；`cross_check=True` 时两个服务
  的结果互相印证；失败和无结果区分开；保留来源 URL 和时间；所有服务都
  失败/都没结果时老实说失败，绝不编结果。`force_provider` 可直接指定只用
  某一个服务（备选链路测试用）。

```python
from tools.search import SearchService
service = SearchService()
outcome = service.search("纳斯达克最新收盘", kind="normal")
outcome = service.search("某基金的持仓结构", kind="structured", cross_check=True)
outcome.to_dict()  # 状态、结果、走了哪些服务、质量检查
```

## 文件：工作区围栏

- 只能在固定工作区里读写；`../` 穿越、绝对路径在工作区外、符号链接逃逸一律拒绝。
- 密钥黑名单：`.git/`、`.ssh/`、`*.pem`、`*.key`、`id_rsa*` 一律不可碰。
- `.env` 只写不读：能创建、能改，读不回里面的值（Hermes 的做法）。
- 读分页（一次最多 2,000 行）；超大文件（>10MB）/ 二进制文件先拒绝。
- 写默认不覆盖；覆盖必须显式说，且覆盖前先备份到 `.backups/`（只留最近 5 份）。
- 改要求旧串在文件里只出现一次；删默认进回收站（`.trash/`），彻底删要显式说；
  删目录默认拒绝，要删必须显式说递归。
- 写文件用“先写临时文件、再原子改名”，写一半失败原文件不动。
- 重要操作（写、改、删）初期默认先问 BOOS 确认：没传 `confirmed=True` 就抛
  `NeedsConfirmation`（带一句人话说明），跑顺了再按路径放宽。
- 每次写/改/删都记审计（改前改后哈希），存在工作区 `.audit/audit.log`。

```python
from tools.files import Workspace
ws = Workspace("D:/代码/十二的工作区")
ws.read_text("notes/a.txt")
ws.write_text("notes/a.txt", "你好", confirmed=True)
ws.edit_text("notes/a.txt", "你好", "你好呀", confirmed=True)
ws.delete("notes/old.txt", confirmed=True)  # 默认进回收站
```

## 命令：结构化调用 + Tier 分级

- 只收命令列表（`["python", "script.py"]`），不用 `shell=True`、不让模型拼
  shell 字符串。
- Tier 2 硬禁（`rm -rf /`、格式化、关机、下载即执行 `curl|sh`、提权、
  WSL 逃逸等）：绝不执行，就算确认了也不跑。
- Tier 1 需确认（`pip install`、`git push --force`、删文件、联网下载、
  跑 shell 自由代码等）：没确认就抛 `NeedsConfirmation`。
- Tier 0 直接跑（跑脚本、pytest、`git status/log/diff`、`ls` 等）。
- 超时默认 60 秒、上限 300 秒，超时杀掉整个进程组；输出截断（stdout/stderr
  各截到 1 万字，标明截断了）；工作目录锁在工作区内；不把 API Key 等环境变量
  传给被跑的命令（只传最小白名单）。

```python
from tools.commands import CommandRunner
runner = CommandRunner("D:/代码/十二的工作区")
runner.run(["python", "-m", "pytest", "tests/", "-q"])
runner.run(["pip", "install", "requests"], confirmed=True)  # Tier 1，要确认
```

## 清理垃圾：BOOS 定的规矩

1. 先扫一遍，按种类报大白话（这是什么、有多大、删了有什么影响），不列原始文件名。
2. BOOS 按种类点头再删，没确认的种类不动。
3. 确认要删的直接彻底删除，空间马上腾出来（进回收站不算删干净）。
4. 拿不准的东西（大文件、分不清的）先进回收站，过几天确认没问题再清空。
5. 密钥文件不碰，扫描结果里也不列出来。
6. 清理范围是整台电脑，不只工作区——扫描根目录由调用方显式传
   （下载文件夹、临时文件、回收站是首选），这是单独授权的任务。

三道保护：分类置信度（低置信度绝不直接删）、删除预览（删除只认冻结的预览清单，
扫描到删除之间新增的文件不会被删；删除时二次校验路径仍在扫描根目录内）、
删除后核对（实际删了几个、进回收站几个、腾出多少空间、失败几个、跳过几个）。

```python
from tools.cleanup import scan, build_preview, delete_preview, format_summary
result = scan("D:/下载", root_kind="downloads")
preview = build_preview(result)
print(format_summary(preview))  # 大白话汇报，给 BOOS 看
report = delete_preview(preview, ["temp", "logs"], trash_dir="D:/回收站暂存")
```

## 命令行入口

```bash
python -m tools.cli search "今天天气怎么样"
python -m tools.cli search "纳斯达克最新收盘" --kind structured --cross-check
python -m tools.cli file read notes/a.txt
python -m tools.cli file write notes/a.txt --content "你好" --yes
python -m tools.cli run -- python -m pytest tests/ -q
python -m tools.cli cleanup scan D:/下载 --kind downloads --out preview.json
python -m tools.cli cleanup delete --preview preview.json --categories temp,logs --yes
```

工作区目录：`--workspace` 指定，或 `TWELVE_WORKSPACE` 环境变量，默认当前目录。
Key 只从环境变量读（`ANYSEARCH_API_KEY`、`TAVILY_API_KEY`），绝不在命令行传 Key。

## 测试

```bash
.venv/bin/python -m pytest tests/test_tools_*.py -q   # 只跑工具测试
.venv/bin/python -m pytest tests/ -q                  # 全量测试
```

测试全部离线（假 HTTP 会话、假服务对象、临时目录），不碰真实接口、不产生费用。
多场景覆盖：搜索（正常、结构化、402 兜底、服务挂了兜底、全挂明说失败、
无结果和失败区分、质量检查）；文件（越界、密钥文件、改不唯一、覆盖保护、
原子写）；命令（三级分级、组合绕过、超时、截断、工作目录锁、环境隔离）；
清理（按种类报、按种类确认、不确定的进回收站、删完核对）。

## 待办（本批没做的事）

- **真实接口验证**：Parallel MCP 可直接测（不用 Key）；AnySearch / Tavily
  要等 BOOS 自己注册 Key（都免费、都不用绑卡，Key 只放本地 `.env`），
  注册完再跑真实调用验证（见下面“真实接口验证清单”）。
- **Box 上实测 gVisor**：隔离的下一步，单独的任务；笔记本第一批靠分级 + 人工确认。
- **接进聊天循环**：工具现在是独立的包 + 命令行，还没接进 `main.py` 的对话
  （模型调工具需要 function-calling 接线），等第一批跑顺再接。
- **工作区目录、清理范围**：等 BOOS 定（建议 `D:\代码\十二的工作区`；
  清理首选下载文件夹、临时文件、回收站）。
- **OS 级权限层**：笔记本（Win11 Home）第一批不做（Windows Sandbox 没有、
  seisin 不支持 Windows）；Box 上“专用用户 + systemd”已实测可行，等工具跑顺再做。

## 真实接口验证清单（等 BOOS 注册 Key 后跑）

- [ ] Parallel MCP 真实调用：`python -m tools.cli search "今天天气怎么样"`（不用 Key，直接跑）
- [ ] AnySearch 真实调用：注册 Key → 设 `ANYSEARCH_API_KEY` → 同上（先看 402/匿名额度表现）
- [ ] Tavily 真实调用：注册 Key → 设 `TAVILY_API_KEY` → `search --kind structured`
- [ ] 备选链路真实验证：AnySearch 402 → Parallel；Tavily 用完 → AnySearch；
      Parallel 429 → AnySearch；三个全挂 → 明说失败（单元测试已用假失败覆盖，
      真实接口测一两次就行）

## 诚实说明（已知缺口）

- 代码层防的是误操作和提示注入下的意外损坏，防不住存心绕过
  （终端能绕过文件工具的写保护，Hermes 官方文档原话）。
- 命令分级管不住命令内部的行为（比如 python 脚本里写绝对路径），
  第一批靠“只跑自己信任的命令 + 危险的先问 BOOS”兜着。
- 搜索免费额度是借来的，价格变得很快；正式接入前，务必再打开各家官网
  定价页复核一次（2026-11-10 的快照见 `十二第一批工具新方案.md`）。
- AnySearch / Tavily 的真实返回形状目前是按官方文档 + 防御性解析写的，
  没 Key 测不了真实调用；注册完 Key 要先跑一遍真实调用核对解析。
