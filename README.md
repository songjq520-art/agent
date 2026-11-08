# agent

个人ai智能助理

> 本文件在原有仓库 README.md（项目说明文档，原内容只有上面两行）的基础上扩展，原有内容已保留，未删除。

# 个人多模型 Agent（智能代理）——最小可运行骨架

## 一、这个项目是做什么的

这是一个**个人多模型 Agent（智能代理）**项目，第一阶段的**最小可运行骨架**已经完成并合入 main；现在又加上了 **DeepSeek 作为第一任主控的接入代码**（代码已写好，未真实调用），先把地基打好，以后再逐步完善。

- **Agent（智能代理）**：能理解你的问题、自己去调用 AI 模型、并把结果返回给你的程序。
- **多模型**：计划接入多个模型——DeepSeek、Muse、GPT、Gemini、Grok、Kimi。现在已接入 **DeepSeek**（第一任主控，默认模型 deepseek-flash）和 **Muse**（工作模型，Muse Spark 1.3，先跑通接口和调用流程）。
- **主控模型（Controller）**：负责理解任务、制定计划、协调其他模型。**DeepSeek 是第一任主控**（见 `PROJECT_PRINCIPLES.md`），负责任务分配、模型调度、信息传递、总结；GPT 是远期主控候选，目前不接入。主控是可以更换的角色，**不要把系统写死成只能由某个模型控制**，见 `config.py` 里的 `CONTROLLER`（主控）说明。注意：目前 `CONTROLLER` 只是角色名，实际路由由 `router.py` 里的固定映射决定，改 `CONTROLLER` 本身不会改变路由。

本次不做：多个模型同时运行、复杂智能调度、电脑控制、自动化工作流、云服务器部署、复杂长期记忆系统、复杂成本控制系统。

## 二、整体架构如何运行

```
你输入一个问题
      ↓
main.py（程序主入口）：读取配置、检查模型状态
      ↓
router.py（任务路由模块）：判断任务类型（普通问答 / 代码相关 / 通用），决定交给哪个模型（默认 DeepSeek 主控）
      ↓
providers/deepseek.py（DeepSeek 模型接口，默认）或 providers/muse.py（Muse 模型接口，工作模型）：用统一接口调用对应模型的 API（应用程序接口）
      ↓
模型返回回答 → main.py 展示结果，同时写日志（运行记录）和记账（Token 使用量 / 成本，按模型分别记录再汇总）
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
| `providers/` | AI 模型接入模块：每个模型一个文件，本次有 `deepseek.py`（默认主控）和 `muse.py`（工作模型） |
| `providers/deepseek.py` | DeepSeek 模型接口：真正调用 DeepSeek API 的地方，默认模型 deepseek-flash |
| `providers/muse.py` | Muse 模型接口：真正调用 Muse API 的地方 |
| `memory/` | 本地记忆模块：长期记忆用 mem0（见第十五节），唯一入口是 `memory/mem0_store.py`，其他代码不直接碰 mem0；记忆存在本地 `memory/mem0_chroma/`（向量库）+ `memory/history.db`（修改历史），不上传 |
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

# 4. 安装依赖（版本都钉死了，见 requirements.txt 顶部的说明，不要只改数字就升级）
pip install -r requirements.txt
```

> 注意：`requirements.txt` 里 torch 要走 CPU 专用源（文件里已经写好了，不用你加参数）。第一次安装要下载约 1.8GB（torch 等大包），还要再从网上下载 embedding 模型约 96MB，需要一点时间和网络，装一次就行，以后不用重复下载。

> 等你确认电脑系统和 Python 版本后，可以再补充针对性的操作步骤。

## 五、如何配置 API Key（接口密钥）

**API Key（接口密钥）只能通过环境变量读取，禁止硬编码（直接写入源代码）。**

```bash
# 1. 复制示例文件，得到你自己的 .env 文件（真实文件，不要传到 GitHub）
cp .env.example .env

# 2. 用文本编辑器打开 .env，填入你自己的 Key（密钥），不要发给别人，也不要写进聊天记录
# DEEPSEEK_API_KEY=你的 DeepSeek Key（默认主控用）
# MODEL_API_KEY=你的 Muse Key（官方变量名；旧名 MUSE_API_KEY 也兼容）
```

### DeepSeek 的 Key 怎么注册和配置

DeepSeek 的 Key 在 platform.deepseek.com 申请：

1. 打开 platform.deepseek.com，用大陆手机号+短信验证码或微信扫码登录；
2. 在控制台充值（支付宝 / 微信支付）；
3. 在 API keys 里创建一个 Key，创建后只显示一次，马上复制保存好；
4. 把 Key 填进你本地的 `.env` 文件：`DEEPSEEK_API_KEY=你的 DeepSeek Key`（不要发给别人，也不要写进聊天记录、不要传到 GitHub）。

**程序启动时会自动读取项目根目录的 .env 文件**（`config.py` 里的 `load_env_file()`，只用 Python 自带库，不用 dotenv）。你不需要每次手动加载。

如果不想用 `.env` 文件，也可以手动设置环境变量（可选方式）：
```bash
# macOS / Linux：
export MODEL_API_KEY=你的 Muse Key
# Windows（PowerShell）：
# $env:MODEL_API_KEY="你的 Muse Key"
```

**目前状态：DeepSeek 和 Muse 都没有 API Key。** 按无 Key 处理，先完成骨架和模拟测试，真实 API 连接标记为**未完成**，等你准备好密钥、确认后再测试。

## 六、Muse App 会员 vs API 账单——必须区分

这是两套**不同**的账单，不要混淆：

| | Muse App 会员 | Meta Model API（模型接口）按量账单 |
|---|---|---|
| 用途 | 在 Muse App 里和十一聊天，每周词元更多 | 你自己的程序 / Agent 调用 Muse 模型，按 Token（词元）付费 |
| 价格 | Power $20/月、Maximum $100/月 | 按量：见下文价格说明 |
| 是否提供 API Key | **不提供** | 提供，需要单独在 dev.meta.ai 开通账单、创建 Key |
| 本项目用哪个 | 不用 | **用这个** |

所以：订了 Muse App 会员，也**不能**直接拿来跑这个 Agent；跑这个 Agent 要另外申请 Meta Model API 的 Key。

DeepSeek 只有 API 按量账单（在 platform.deepseek.com 充值），没有 App 会员这套东西，同样要单独申请 Key。

## 七、Muse 和 DeepSeek 官方 API（应用程序接口）说明

### Muse

本次核实来源：Meta Model API 官方文档 `dev.meta.ai/docs/models`，2026-11-05 核实。

- **接口地址（Base URL，基础地址）**：`https://api.meta.ai/v1`
- **本次使用的模型**：`muse-spark-1.3`（Muse Spark 1.3，最新版，官方文档推荐新工作使用，上下文 1,048,576 Token）
- **本次使用的端点（Endpoint，接口路径）**：Chat Completions（聊天补全），`POST /v1/chat/completions`
- **认证方式**：请求头 `Authorization: Bearer <你的 API Key>`
- **请求格式（OpenAI 兼容）**：
  ```json
  {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "你的问题"}]}
  ```
- **返回格式**：`choices[0].message.content` 是回答文本；`usage` 里有 `prompt_tokens`（输入 Token 数）、`completion_tokens`（输出 Token 数）、`total_tokens`（总 Token 数），还有 `prompt_tokens_details.cached_tokens`（缓存输入 Token 数）和 `completion_tokens_details.reasoning_tokens`（推理 Token 数）。**如果模型没有返回 usage，程序不会编造，成本记为未知（None）。** 成本计算：普通输入按 $1.25、缓存输入按 $0.15、输出按 $4.25（每百万 Token）；`reasoning_tokens` 是 `completion_tokens` 的子集，不重复计费。

**价格说明（必须注明）**：以下价格是 **Muse Spark 1.3 Standard mode（标准模式）**的官方价格，以 Meta 官方定价页为准，2026-11-05 核实为每百万 Token 普通输入 $1.25、缓存输入 $0.15、输出 $4.25。本次未能直接打开官方定价页，上述价格来自多家资料一致的记录，**不要把它当成绝对不变的官方定论**，实际以你开通账单时官方页面为准。注意：Contributor 版（贡献者模式）价格不同（普通输入 $0.10、缓存输入 $0.002、输出 $0.20），本项目不用 Contributor，只用 Standard mode。

### DeepSeek

本次核实来源：DeepSeek 官方文档 `api-docs.deepseek.com`，2026-11-06 核实。

- **接口地址（Base URL，基础地址）**：`https://api.deepseek.com`
- **本次使用的模型**：`deepseek-flash`（DeepSeek V4.1 Flash，默认，第一任主控）；需要更强推理时，可在建 `DeepSeekProvider` 时传 `model="deepseek-v4-pro"`（DeepSeek V4 Pro，约 Flash 的 4 倍价格），上下文都是 1,000,000 Token。
- **本次使用的端点（Endpoint，接口路径）**：Chat Completions（聊天补全），`POST /chat/completions`（注意：DeepSeek 官方文档的地址里没有 `/v1` 这一段，和 Muse 的 `https://api.meta.ai/v1` 写法不一样，不要照抄）。
- **认证方式**：请求头 `Authorization: Bearer <你的 API Key>`
- **请求格式（OpenAI 兼容）**：
  ```json
  {"model": "deepseek-flash", "messages": [{"role": "user", "content": "你的问题"}]}
  ```
- **返回格式**：`choices[0].message.content` 是回答文本；`usage` 里有 `prompt_tokens`（输入 Token 数）、`completion_tokens`（输出 Token 数）、`total_tokens`（总 Token 数），还有 `prompt_tokens_details.cached_tokens`（缓存输入 Token 数）。**如果模型没有返回 usage，程序不会编造，成本记为未知（None）。** 成本计算：Flash 普通输入按 $0.15、缓存输入按 $0.003、输出按 $0.60（每百万 Token，闲时价）；Pro 普通输入按 $0.66、缓存输入按 $0.022、输出按 $1.98（每百万 Token，闲时价）。

**价格说明（必须注明）**：以下价格是 **DeepSeek 闲时（Off-peak）价**，以 DeepSeek 官方定价页为准，2026-11-06 核实为：Flash 每百万 Token 普通输入 $0.15、缓存输入 $0.003、输出 $0.60；Pro 普通输入 $0.66、缓存输入 $0.022、输出 $1.98。**高峰时段（北京时间周一至周五 09:00–12:00、14:00–18:00）价格是闲时的 2 倍**，本地记账按闲时价算，高峰时实际成本可能高于本地记录，**不要把它当成绝对不变的官方定论**，实际以 DeepSeek 官方定价页和官方后台为准。注意：旧模型名 `deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 退役，不要再用。

## 八、如何启动程序

```bash
# 无 Key 模式（目前的状态）：只做配置检查 + 路由演示，明确提示“未接通真实 API”，不调用模型，不产生费用
python3 main.py "你好"

# 有 Key 且你确认要真实调用时：先在 .env 里填入 DEEPSEEK_API_KEY（默认走 DeepSeek 主控；Muse 的 Key 是 MODEL_API_KEY），程序启动自动读取，再运行同样命令
# python3 main.py "你好"
```

无 Key 时程序会输出：模型状态、路由结果，并明确说“模拟模式 / 未接通真实 API”。**这不是真实调用成功，不要把它当成调用成功。**

连续对话（`python3 main.py` 不带问题进入）会用长期记忆：每轮提问先搜相关记忆带给模型；想记新事时输入 `记住：内容`，直接存进长期记忆，不再先问再存。注意：命令行默认不加载记忆（加载本地模型要约 25 秒），只有输入 `记住：内容` 时才按需加载一次；网页服务是常驻的，启动时就加载好，一直用着。第一次用之前，先跑一次迁移脚本把旧记忆搬过去（见第十五节）。

## 九、如何进行基础测试

```bash
# 运行全部模拟测试（不需要 API Key，不联网，不产生费用）
python3 -m pytest tests/ -v
```

测试分两类，`tests/test_router.py` 里已明确区分：

- **模拟测试（Mock Test）**：不需要 API Key，不产生费用，默认运行。
- **真实测试（Real API Test）**：需要 API Key，可能产生实际费用，默认跳过（skip）。Muse 的真实测试只有设置了 `MODEL_API_KEY`（或 `MUSE_API_KEY`）**并且**显式设置 `RUN_REAL_API_TESTS=1` 时才运行；DeepSeek 的真实测试（`tests/test_deepseek.py` 里的 `test_real_deepseek_chat`）只有设置了 `DEEPSEEK_API_KEY` **并且**显式设置 `RUN_REAL_API_TESTS=1` 时才运行。**没有你的确认，不要进行产生费用的真实 API 测试。**

## 十、预算控制（简单版）

- 每月预算：**$5（美元）是整个 Agent 的总上限，不是每个模型各有 $5**，在 `config.py` 里集中配置。
- 每次调用记录 Token 使用量，尽可能记录实际成本；费用按模型分别记录（`logs/budget_usage.json` 里的 `cost_by_model` 小计，例如 `{"deepseek-flash": 0.01}`），再汇总成总数，**封顶只看总数**。
- 达到预算 **100%（$5）** 时打印提醒，**永不自动停**，提醒后由你自己决定要不要继续。
- 成本算不出来时也一样：只提醒、不停，等你自己决定。
- 累计成本存在本地 `logs/budget_usage.json`，这个文件会被 `.gitignore` 忽略，不上传。文件里还有 `month`（月份，`YYYY-MM`）、`unknown_cost`（成本未知标记）和 `cost_by_model`（按模型小计）字段，跨月时累计和小计一起自动重置为 0，但 `unknown_cost` 不会因跨月自动清除——例如 11 月发生了一笔无法确定成本的调用，12 月跨月后未知成本的标记仍然保留，建议人工核对后清除。**每月重置不只在启动时检查**：每次判断预算或记账前都会先看月份，所以程序一直运行、跨过自然月时也能重置。
- 调用前费用预估：每次真实调用前，程序会先粗略预估本次成本。输入 Token 分开估算：ASCII 字符（英文、数字、标点）按 4 个字符 1 个 Token，非 ASCII 字符（中文等）按每个字符 1.5 个 Token，中文按 1.5 倍留余量，宁可高估、不要低估；输出按 `MAX_OUTPUT_TOKENS`（2048 Token）估算，这个值同时是请求里实际传给 API 的 `max_tokens` 上限，所以输出量不再是凭空假设；缓存按 0 算。预估超过本月剩余预算时也在请求前拦截。**这只是粗略预估，不能保证单次请求绝对不超支**：输入预估仍是启发式估算、价格可能变动、缓存命中情况不同，实际成本可能高于预估，实际费用以 Meta 官方后台为准。

**重要提醒：这是 Agent 自身的调用限制，不代表服务商账单的绝对上限。** 请同时关注各模型官方后台的消费情况（Muse 看 Meta 后台，DeepSeek 看 DeepSeek 后台），服务商后台才是最终账单。本地预算只能拦截“已累计超支”和“预估明显超支”，**不要把它当成能保证绝对不超支的系统**。另外 DeepSeek 的本地记账和预估按闲时价算，高峰时段实际价格是闲时的 2 倍，高峰时本地记录可能低于实际消费。

**成本未知时怎么处理**：如果一次调用成功、但模型没有返回 `usage`，程序不会把这次调用当成零成本，而是把 `unknown_cost` 标记为 `true`，打印提醒，但**不暂停**、继续运行。请先去对应模型的官方后台核对那次调用的实际消费（Muse 看 Meta 后台，DeepSeek 看 DeepSeek 后台），核对后清除标记。清除方式：用文本编辑器打开本地 `logs/budget_usage.json`，把 `unknown_cost` 改为 `false` 后保存（不要直接删除整个文件——删文件会连本月累计成本一起清零，不推荐）。

**预算文件损坏怎么办**：如果 `logs/budget_usage.json` 损坏、无法解析为 JSON、或缺少 `month` / `total_cost_usd` / `unknown_cost` 三个关键字段中的任何一个（类型不对也算损坏），程序会进入安全暂停状态，禁止新的付费调用，而且**不会覆盖原文件**，原损坏文件会被保留。`cost_by_model` 是可选字段，旧文件没有它不算损坏；但如果写了却格式不对（不是“模型名 → 非负数字”的字典），也算损坏。人工恢复步骤：① 打开 `logs/budget_usage.json` 检查；② 如果能修好，手动把三个字段改成合法值（`month` 如 `2026-11`、`total_cost_usd` 如 `1.5`、`unknown_cost` 为 `true` 或 `false`）；③ 如果修不好，先把原文件改名备份（如 `budget_usage.json.bak`），再让程序重新生成一个新文件（新文件从 0 开始，旧累计需要人工按官方后台补记）；④ 只有核对过官方后台后才能恢复调用。程序不会自动修文件，也不会在损坏时自动清除累计费用或未知成本状态。

## 十一、日志（运行记录）

日志写在本地 `logs/agent.log`，记录：时间、模型、任务类型、是否成功、Token 使用量、成本、错误信息。**绝不记录 API Key 或其他敏感信息。**

## 十二、后续接入 GPT、Gemini、Grok 应该怎么做

DeepSeek 和 Muse 已经接入（DeepSeek 的接入方法和上面一样：`providers/deepseek.py` + `config.py` 里的 `MODEL_CONFIG` + `.env.example` 里的 `DEEPSEEK_API_KEY=`）。以后再接入新模型时：

1. 在 `providers/` 下新建一个文件，例如 `providers/gpt.py`，写一个和 `DeepSeekProvider` / `MuseProvider` 一样有 `chat(prompt)` 方法的类。
2. 在 `config.py` 的 `MODEL_CONFIG`（模型配置）里把对应模型的 `enabled`（启用状态）改成 `True`，填好模型名和价格。
3. 在 `router.py` 里把相应任务类型指向新模型；如果要换主控，要同时改两处：`router.py` 里的路由表（真正决定路由）和 `config.py` 里的 `CONTROLLER`（角色名），让两者保持一致——只改 `CONTROLLER` 不会改变实际路由。
4. 在 `.env.example` 里已有 `MODEL_API_KEY=`、`DEEPSEEK_API_KEY=`、`OPENAI_API_KEY=`、`GEMINI_API_KEY=`、`GROK_API_KEY=`，按同样方式在 `.env` 里填入真实 Key。
5. 先用模拟测试验证路由，再用真实测试（需你确认）验证接口。

注意：按 `PROJECT_PRINCIPLES.md` 的规划，GPT 是远期主控候选，Gemini 保留未来接入可能、具体分工待定，Grok 暂不纳入正式架构、列为候选，Kimi 是已确认的日常辅助模型（配合 DeepSeek，接入时间由你决定）。接入哪个、何时接入，都由你说了算，不要自己加。

## 十三、已知未完成项

- [ ] 真实 DeepSeek API 连接：**代码已写好，未完成**（无 API Key，等待你准备好密钥、确认后再测试）
- [ ] 真实 Muse API 连接：**未完成**（无 API Key，等待你准备好密钥再测试）
- [ ] GPT / Gemini / Grok / Kimi 接入：未开始（按原则文档的规划，由你决定何时接入）
- [x] 长期记忆系统：已换成 mem0（见第十五节），语义检索、每轮自动提取、不知道就说不知道；命令行支持“记住”命令直接存，网页端启动时就加载好
- [ ] 工具调用：只预留了 `tools/` 文件夹，未开发电脑控制
- [ ] 云服务器部署：未开始

## 十四、本地网页入口（网页版聊天页面）

网页入口让你用浏览器和 Agent 对话，不用每次都开黑窗口打字。样式照着 Muse 网页版做的：左边是侧边栏（头像、名字、模型和预算状态），中间是聊天窗口，底部是输入框。页面布局兼容手机屏幕（小屏幕上侧边栏会收起来，只留聊天），但当前服务仅支持本机访问，暂不支持手机远程连接电脑。

**目前仅支持本机访问**：服务默认只监听 127.0.0.1（你自己的电脑），不上云、不开放局域网，别人打不开。手机访问、局域网访问还没做，以后做之前要先接身份验证（见下面的“未来扩展”）。

```bash
# 在项目根目录下启动（默认只监听本机 127.0.0.1:8000，不对外网开放）
python3 -m web.server

# 换个端口
python3 -m web.server --port 8080
```

启动后在浏览器打开 `http://127.0.0.1:8000` 就能对话，按 Ctrl+C 停止服务。

**对话历史只保存在内存里**：服务重启后，之前的对话记录就没了（预算不受影响，预算存在 `logs/budget_usage.json` 里）。本阶段不做历史持久化；以后要做的话，可以把 `ChatState.messages` 存到文件里。

**监听地址和端口怎么改**（都集中在 `web/server.py` 的 `ServerConfig` 里，别处不要写死；优先级从高到低）：
1. 命令行参数：`python3 -m web.server --host 127.0.0.1 --port 8080`
2. 环境变量：`AGENT_WEB_HOST` / `AGENT_WEB_PORT`
3. 默认值：127.0.0.1:8000

**它和黑窗口是同一套逻辑**：页面发来的消息走 `main.py` 的 `run_question()`（路由 → 模型 → 预算 → 日志），对话历史和预算跟踪器由服务进程一直拿着，和 `chat_loop()` 的做法一样；无 Key 时走演示模式，不调用模型、不产生费用。前端和后端只通过下面四个接口打交道，页面里不写后端逻辑，以后加手机入口也走同一套接口。

接口（给页面用的）：
- `GET /`：聊天页面（`web/index.html`）
- `GET /api/status`：模型状态 + 本月预算（JSON）
- `POST /api/chat`：发消息，请求 JSON `{"message": "..."}`，返回回答、模型名、模式（real/demo）和本次成本；请求内容必须是 JSON 对象，否则返回 400
- `POST /api/reset`：清空本轮对话历史（预算不动）

**并发**：服务用 `ThreadingHTTPServer`（每个请求一个线程），`ChatState` 里有一把锁，问、清空、读状态都走这把锁，一次只处理一个请求，所以两个请求同时进来也不会把历史和预算搞乱；调用模型时会一直拿着锁，期间别的请求会等一等。

**错误处理**：接口遇到没料到的错误时，服务端会把详情记到日志里，给前端只回一个通用的“服务内部出错，请稍后再试。”（500），不泄露内部信息。

**个性化**：页面样式全在 `web/index.html` 最上面的 `:root` 颜色变量和标着“可改”的地方——改颜色、标题、欢迎语、头像 emoji 都不用动别的代码。想换深色主题：把 `--bg` 改成深色、`--text` 改成浅色、气泡颜色跟着调深就行。

**未来扩展（现在不做，只预留了位置）**：
- 身份验证：`web/server.py` 的 `make_handler(state, authenticator=...)` 留了一个 `authenticator` 钩子，接上一个校验函数就能对请求做身份验证（比如比对请求头里的 Token）；现在默认不校验，因为只监听本机。
- 局域网/手机访问：先接上身份验证，再把 host 改成 0.0.0.0；不要只改 host 不加校验。

**安全**：服务只监听本机（127.0.0.1），同一台电脑上能打开，别人打不开；不记录 API Key，日志和记账走原来那套（`logs/agent.log`、`logs/budget_usage.json`）。另外有一道保护：如果把监听地址改成非本机地址（比如 0.0.0.0）却没有配置身份验证，服务会拒绝启动并提示原因，防止一不小心开放无身份验证的局域网访问。

**长期记忆**：网页聊天也用长期记忆（见第十五节）：服务启动时就加载好记忆，每轮提问先搜相关记忆带给模型；每轮聊天成功后自动提取、直接存进长期记忆，不用再点确认。

## 十五、长期记忆（mem0）已经接进对话

长期记忆原来是自己写的 v1（关键词检索、先问再存），有个真 bug：问“你记得我叫什么吗？”搜不到“称呼”那条记忆。现在换成了 **mem0**（开源长期记忆库），黑窗口连续对话和网页聊天都会用它。

**新架构是这样的**：
- **mem0 负责三件事**：每轮对话后自动提取值得长期记住的事、存进本地向量库、提问时按意思检索（语义检索，不是只对字面关键词）。
- **分工**：提取用 DeepSeek Flash（调 API，要花一点钱）；算向量用本地模型 BAAI/bge-small-zh-v1.5（在自己电脑上跑，不花钱）；向量库用本地 Chroma（也在自己电脑上）。
- **唯一入口**：项目里只有 `memory/mem0_store.py` 直接碰 mem0，其他代码（`main.py`、`web/server.py`）都只调它，不直接 import mem0。以后换记忆后端，只改这一个文件。
- **数据只存在本地**：`memory/mem0_chroma/`（向量库）+ `memory/history.db`（每条记忆的修改历史，谁改过、改成什么样都有记录），这两个都会被 `.gitignore` 忽略，不上传到 GitHub（仓库是公开的）。
- **遥测已关**：mem0 默认会把使用情况发到外面（PostHog），这里在 import 之前就关掉了，一点都不往外发。
- **只存你明确说过的长期事实**：提取规则是 BOOS 定的——只存你自己明确说过的长期事实、偏好、目标和硬性要求；不存猜测、假设（“如果以后学 Rust……”）、闲聊（“今天天气不错”）、模型的建议。试跑验证过这条规则生效，但它是写在提示词里的服从、不是机制保证，换模型、升级 mem0 都要重跑回归测试。

**怎么用（命令行和网页端不一样）**：
- **网页端（常驻服务）**：服务启动时就加载好记忆（加载本地模型约 25 秒，启动时付一次），每轮提问先搜相关记忆带给模型；每轮聊天成功后自动提取、直接存进长期记忆，不用再点确认。
- **命令行（黑窗口连续对话）**：默认不加载记忆（每次问答都起新进程的话，每次都要等 25 秒加载，不划算）；想记新事时输入 `记住：内容`（例如 `记住：我喜欢喝咖啡`），这时才按需加载一次，直接存进长期记忆。
- **删除要你同意**：删记忆必须经过 BOOS 明确同意，缺了这一步不删；删过的记忆，历史记录里还能查到。

**检索阈值 0.4（为什么是 0.4，不是 0.5）**：每轮提问先搜记忆，只把分数 ≥ 0.4 的带给模型（一次最多 5 条）。试跑实测：答对的记忆分数在 0.40–0.66 之间，比如“你记得我叫什么吗？”召回“小宋”那条只有 0.4793；阈值设 0.5 会把这条正确的也滤掉，v1 那个 bug 就又回来了。答错的记忆分数在 0.39–0.57，和答对的重叠很大，光靠分数分不开——所以阈值只是第一道关，真正的规矩是下面这条。

**不知道就说不知道（硬规矩）**：搜不到相关记忆，就不带记忆、直接说不知道，别把看着像的记忆当答案。试跑里试过：库里没有答案的问题，语义检索还是会返回一个最接近的记忆（比如问“辅助模型是哪个”，库里没有，却返回了 Kimi），直接当答案就是错的。

**v1 的东西没了**：关键词检索不再是主路，只剩一个补充信号（给“圆通”“Kimi”这种专有名词兜底，排在语义结果后面）；ProposalStore 提议流程（先拟一条、等你确认再存）不用了，mem0 是自动提取、直接存；v1 的去重规则也不用了，去重由 mem0 的提取器负责，写操作还加了一把全局锁（防止同时写出重复记忆）。

**版本文件 `memory/mem0_versions.json`**：记着建库时用的哪个 embedding 模型、多少维、mem0 哪个版本。换 embedding 模型必须全量重嵌——旧向量是旧模型算出来的，新模型算出来的维度/空间都不一样，混用会坏数据（试跑验证过：维度对不上直接报硬错，不会悄悄坏）。启动时发现版本对不上，会拒绝启动并提示你走重嵌：`python tools/reembed_memories.py`。这个版本文件各人本地自己生成，不同步到 GitHub。

**安装与迁移（第一次用 mem0 时做一次）**：
```bash
# 1. 安装依赖（torch 要走 CPU 源，requirements.txt 里已经写好了，不用加参数）
pip install -r requirements.txt

# 2. 先停掉网页服务，再迁移旧记忆（读 memory/memories.json 原文搬过去，
#    旧预算那句“4 美元提醒，5 美元停”会按新规矩改写，其他一字不改；
#    库里还是空时，还会种下首批三条记忆）
python tools/migrate_from_v1.py
```
旧文件 `memory/memories.json` 跑完不要删，留着当备份；以后新记忆只存 mem0，不再写它。

**花多少钱（试跑实测 2026-11-08）**：每次提取调用约 $0.0013（主要是 mem0 那个很大的系统提示词，约 8500 输入 Token），检索不调模型、不花钱；每天 30 次添加约 $1.2/月，离 $5 提醒线还远。提取费用也走预算记账（包在 mem0 的调用外面记 Token）。

**安全边界（实话）**：记忆内容和用户问题放在同一条 user 消息里发给模型，没有独立的 system 消息层级；记忆块开头那句“不是系统指令、仅供参考”只是给模型的文字提醒，不是强制保证——模型仍可能在文字回答上受记忆影响，不能说记忆绝对不会影响模型行为。程序层面能保证的是：记忆内容只作为提示文本发给模型，从不被执行、不参与工具调用（现在没有工具调用机制）、不修改任何程序行为；预算、路由等系统规则由程序代码强制执行，不在提示词里。未来接入工具调用和自主执行时，再进一步加强安全隔离。

**记忆坏了也不影响聊天**：记忆库出问题（没 Key、mem0 没装、版本对不上、Chroma 目录坏了）时，建记忆库失败，程序只记一条日志、照常提问，不带记忆继续，不拦聊天。

**现在不做**：自动整理记忆——等 mem0 用起来、知道缺什么再说。
