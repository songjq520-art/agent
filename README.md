# agent

个人ai智能助理

> 本文件在原有仓库 README.md（项目说明文档，原内容只有上面两行）的基础上扩展，原有内容已保留，未删除。

# 个人多模型 Agent（智能代理）——最小可运行骨架

## 一、这个项目是做什么的

这是一个**个人多模型 Agent（智能代理）**项目，目前处于第一阶段：只搭一个**最小可运行骨架**，先把地基打好，以后再逐步完善。

- **Agent（智能代理）**：能理解你的问题、自己去调用 AI 模型、并把结果返回给你的程序。
- **多模型**：未来计划接入四个模型——GPT、Muse、Gemini、Grok。现在只接入 **Muse**，而且用的是价格较低的 **Muse Spark 1.3**，先跑通接口和调用流程。
- **主控模型（Controller）**：负责理解任务、制定计划、协调其他模型。未来计划由 **GPT** 担任主控，但本次不接入 GPT，也**不要把系统写死成只能由 GPT 控制**——主控是可以更换的角色，见 `config.py` 里的 `CONTROLLER`（主控）说明。

本次不做：四个模型同时运行、复杂智能调度、电脑控制、自动化工作流、云服务器部署、复杂长期记忆系统、复杂成本控制系统。

## 二、整体架构如何运行

```
你输入一个问题
      ↓
main.py（程序主入口）：读取配置、检查模型状态
      ↓
router.py（任务路由模块）：判断任务类型（普通问答 / 代码相关 / 通用），决定交给哪个模型
      ↓
providers/muse.py（Muse 模型接口）：用统一接口调用 Muse API（应用程序接口）
      ↓
Muse 返回回答 → main.py 展示结果，同时写日志（运行记录）和记账（Token 使用量 / 成本）
```

**统一接口（让不同模型使用相同的调用方式）**的意思是：以后接入 GPT、Gemini、Grok 时，也给它们写一个和 `MuseProvider` 一样有 `chat(prompt)` 方法的类，`router.py` 和 `main.py` 不用大改，只改配置里“哪个任务交给哪个模型”。

## 三、各个文件夹分别负责什么

| 文件 / 文件夹 | 负责什么 |
|---|---|
| `README.md` | 项目说明文档，就是本文件 |
| `.gitignore` | Git（版本控制工具）忽略配置文件：哪些文件不传到 GitHub（代码托管平台） |
| `.env.example` | 环境变量（Environment Variable）示例文件：只写变量名，不写真实密钥 |
| `requirements.txt` | Python（编程语言）依赖清单：需要安装的第三方库 |
| `config.py` | 项目配置模块：模型、预算、开关都集中在这里 |
| `main.py` | 程序主入口：启动、检查、调用、展示结果、写日志 |
| `router.py` | 任务路由模块：按任务类型分配模型 |
| `providers/` | AI 模型接入模块：每个模型一个文件，本次只有 `muse.py` |
| `providers/muse.py` | Muse 模型接口：真正调用 Muse API 的地方 |
| `memory/` | 本地记忆模块：本次只预留空文件夹，未来放长期记忆、对话历史 |
| `tools/` | 工具调用模块：本次只预留空文件夹，未来放搜索、文件管理等工具 |
| `logs/` | 运行日志模块：运行日志和预算累计文件都存在本地，不上传 |
| `tests/` | 自动化测试模块：用 pytest（Python 自动化测试工具）跑的测试 |

`memory/`、`tools/`、`logs/` 里各有一个 `.gitkeep`（空文件夹保留文件），只是为了让 Git 保留这些空文件夹，本身没有功能。

## 四、如何安装环境（通用版，跨平台）

本项目用 **Python（编程语言）**，尽量采用跨平台方案，不依赖某个操作系统。

```bash
# 1. 确认 Python 版本，建议 Python 3.10 或更高
python3 --version

# 2. （推荐）创建一个虚拟环境（Virtual Environment，隔离的 Python 环境，避免污染系统）
python3 -m venv .venv

# 3. 激活虚拟环境
# macOS / Linux：
source .venv/bin/activate
# Windows（PowerShell）：
# .venv\Scripts\Activate.ps1

# 4. 安装依赖（只用 requests + pytest，尽量少）
pip install -r requirements.txt
```

> 等你确认电脑系统和 Python 版本后，可以再补充针对性的操作步骤。

## 五、如何配置 API Key（接口密钥）

**API Key（接口密钥）只能通过环境变量读取，禁止硬编码（直接写入源代码）。**

```bash
# 1. 复制示例文件，得到你自己的 .env 文件（真实文件，不要传到 GitHub）
cp .env.example .env

# 2. 用文本编辑器打开 .env，填入你自己的 Key（密钥），不要发给别人，也不要写进聊天记录
# MUSE_API_KEY=你的 Muse Key

# 3. 每次运行前，在终端里加载它（macOS / Linux）
export $(cat .env | xargs)
# Windows（PowerShell）请按你的系统方式设置环境变量，或在运行前用 set 命令设置
```

**目前状态：没有 API Key。** 按无 Key 处理，先完成骨架和模拟测试，真实 API 连接标记为**未完成**，等你准备好密钥再测试。

## 六、Muse App 会员 vs API 账单——必须区分

这是两套**不同**的账单，不要混淆：

| | Muse App 会员 | Meta Model API（模型接口）按量账单 |
|---|---|---|
| 用途 | 在 Muse App 里和十一聊天，每周词元更多 | 你自己的程序 / Agent 调用 Muse 模型，按 Token（词元）付费 |
| 价格 | Power $20/月、Maximum $100/月 | 按量：见下文价格说明 |
| 是否提供 API Key | **不提供** | 提供，需要单独在 dev.meta.ai 开通账单、创建 Key |
| 本项目用哪个 | 不用 | **用这个** |

所以：订了 Muse App 会员，也**不能**直接拿来跑这个 Agent；跑这个 Agent 要另外申请 Meta Model API 的 Key。

## 七、Muse 官方 API（应用程序接口）说明

本次核实来源：Meta Model API 官方文档 `dev.meta.ai/docs/models`，2026-11-05 核实。

- **接口地址（Base URL，基础地址）**：`https://api.meta.ai/v1`
- **本次使用的模型**：`muse-spark-1.3`（Muse Spark 1.3，最新版，官方文档推荐新工作使用，上下文 1,048,576 Token）
- **本次使用的端点（Endpoint，接口路径）**：Chat Completions（聊天补全），`POST /v1/chat/completions`
- **认证方式**：请求头 `Authorization: Bearer <你的 API Key>`
- **请求格式（OpenAI 兼容）**：
  ```json
  {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "你的问题"}]}
  ```
- **返回格式**：`choices[0].message.content` 是回答文本；`usage` 里有 `prompt_tokens`（输入 Token 数）、`completion_tokens`（输出 Token 数）、`total_tokens`（总 Token 数）。**如果模型没有返回 usage，程序不会编造，成本记为未知（None）。**

**价格说明（必须注明）**：价格以 Meta 官方定价页为准，2026-11-05 核实为每百万 Token 输入 $1.25、输出 $4.25、缓存输入 $0.15。本次未能直接打开官方定价页，上述价格来自多家资料一致的记录，**不要把它当成绝对不变的官方定论**，实际以你开通账单时官方页面为准。

## 八、如何启动程序

```bash
# 无 Key 模式（目前的状态）：只做配置检查 + 路由演示，明确提示“未接通真实 API”，不调用模型，不产生费用
python3 main.py "你好"

# 有 Key 且你确认要真实调用时：先设置环境变量，再运行同样命令
# export MUSE_API_KEY=你的 Key
# python3 main.py "你好"
```

无 Key 时程序会输出：模型状态、路由结果，并明确说“模拟模式 / 未接通真实 API”。**这不是真实调用成功，不要把它当成调用成功。**

## 九、如何进行基础测试

```bash
# 运行全部模拟测试（不需要 API Key，不联网，不产生费用）
python3 -m pytest tests/ -v
```

测试分两类，`tests/test_router.py` 里已明确区分：

- **模拟测试（Mock Test）**：不需要 API Key，不产生费用，默认运行。
- **真实测试（Real API Test）**：需要 API Key，可能产生实际费用，默认跳过（skip）。只有设置了 `MUSE_API_KEY` **并且**显式设置 `RUN_REAL_API_TESTS=1` 时才运行。**没有你的确认，不要进行产生费用的真实 API 测试。**

## 十、预算控制（简单版）

- 每月预算：**$5（美元）**，在 `config.py` 里集中配置。
- 每次调用记录 Token 使用量，尽可能记录实际成本。
- 达到预算 **80%（$4）** 时打印警告。
- 达到预算 **100%（$5）** 时，暂停 Agent 后续付费调用，等待你确认。
- 累计成本存在本地 `logs/budget_usage.json`，这个文件会被 `.gitignore` 忽略，不上传。

**重要提醒：这是 Agent 自身的调用限制，不代表服务商账单的绝对上限。** 请同时关注 Meta 官方后台的消费情况，服务商后台才是最终账单。

## 十一、日志（运行记录）

日志写在本地 `logs/agent.log`，记录：时间、模型、任务类型、是否成功、Token 使用量、成本、错误信息。**绝不记录 API Key 或其他敏感信息。**

## 十二、后续接入 GPT、Gemini、Grok 应该怎么做

1. 在 `providers/` 下新建一个文件，例如 `providers/gpt.py`，写一个和 `MuseProvider` 一样有 `chat(prompt)` 方法的类。
2. 在 `config.py` 的 `MODEL_CONFIG`（模型配置）里把对应模型的 `enabled`（启用状态）改成 `True`，填好模型名和价格。
3. 在 `router.py` 里把相应任务类型指向新模型；如果要换主控，在 `config.py` 里改 `CONTROLLER` 即可，不用重写其他模块。
4. 在 `.env.example` 里已有 `OPENAI_API_KEY=`、`GEMINI_API_KEY=`、`GROK_API_KEY=`，按同样方式在 `.env` 里填入真实 Key。
5. 先用模拟测试验证路由，再用真实测试（需你确认）验证接口。

## 十三、已知未完成项

- [ ] 真实 Muse API 连接：**未完成**（无 API Key，等待你准备好密钥再测试）
- [ ] GPT / Gemini / Grok 接入：未开始
- [ ] 长期记忆系统：只预留了 `memory/` 文件夹
- [ ] 工具调用：只预留了 `tools/` 文件夹，未开发电脑控制
- [ ] 云服务器部署：未开始
