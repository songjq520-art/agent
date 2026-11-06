# Agent 代码修改说明与测试结果

分支：`feature/deepseek-integration`
仓库：`songjq520-art/agent`（公开仓库）
日期：2026-11-07

## 一、测试结果

测试命令（在项目目录下运行）：

```bash
python -m pytest tests/ -v
```

结果：**55 通过、2 跳过、0 失败**。

- 跳过的 2 个是真实 API 测试（`test_real_deepseek_chat`、`test_real_muse_chat`），因为没有 API Key 且未设置 `RUN_REAL_API_TESTS=1`，所以跳过。
- **明确说明：本次没有运行真实 API 测试，没有产生任何费用，也没有声称真实调用成功。**
- 修改前基线是 48 通过、2 跳过；本次新增 7 个模拟测试，全部通过。

### 新增的 7 个模拟测试（全部通过）

| 测试 | 内容 |
| --- | --- |
| `test_routing_matches_controller` | 路由一致性检查：question、code、general 三种任务路由到的模型都等于 `CONTROLLER`（DeepSeek）。注意：这只是配置一致性检查，不代表已经实现动态主控。 |
| `test_muse_provider_default_model_is_spark` | Muse 不传 model 时默认用 `muse-spark-1.3`。 |
| `test_muse_provider_accepts_explicit_model` | Muse 显式传 `muse-spark-1.3` 时正常使用，请求里也是这个模型。 |
| `test_muse_provider_unknown_model_raises` | Muse 传一个不存在的模型名时直接报错。 |
| `test_main_passes_routing_model_to_muse` | main.py 把路由算出的 `muse-spark-1.3` 传给 Provider。 |
| `test_main_passes_routing_model_to_deepseek_flash` | main.py 把路由算出的 Flash 传给 Provider，成本按 Flash 价格算。 |
| `test_main_passes_routing_model_to_deepseek_pro` | main.py 把路由算出的 Pro 传给 Provider，成本按 Pro 价格算。 |

### 模型选择测试结果（模拟测试，不联网，不产生费用）

- **Flash**：通过。main.py 传 Flash，Provider 用 `deepseek-flash`，成本按 Flash 价格算（1000 输入 + 500 输出 = 0.00045 美元）。
- **Pro**：通过。main.py 传 Pro，Provider 用 `deepseek-v4-pro`，成本按 Pro 价格算（同样 1000 输入 + 500 输出 = 0.00165 美元），记在 Pro 的小计里，Flash 的小计是 0。
- **Muse**：通过。main.py 传 `muse-spark-1.3`，Provider 用它，请求里也是它；不传 model 时默认还是 `muse-spark-1.3`；传错名字会报错。

### 预算保护测试结果

与预算相关的 20 个测试（超预算拦截、预算文件损坏暂停、成本未知暂停、预估拦截、按月重置、未知成本跨月保留等）**全部通过**，修改前和修改后结果一致，预算保护机制没有被破坏。

## 二、修改说明

### 提交 1：`594c8aa6` — DeepSeek 请求参数修正

- `providers/deepseek.py`：请求参数 `max_completion_tokens` 改为 `max_tokens`（按 DeepSeek 官方文档，官方文档里没有 `max_completion_tokens` 这个参数，正确的是 `max_tokens`，范围 1–384K）。
- `tests/test_deepseek.py`：同步修改测试断言，不再断言错误的参数。
- `config.py`：只改了两处注释（`MAX_OUTPUT_TOKENS` 的注释和成本预估的说明），没有改逻辑。
- `README.md`：相关说明同步更新。
- 原因：参数名错了会导致输出上限不生效，预算预估（按 2048 Token 输出估算）会低估真实成本。

### 提交 2：`ccf45d1b` — 原则文档 DeepSeek 状态修正

- `PROJECT_PRINCIPLES.md`：DeepSeek 的状态从“目前尚未注册、接入、启用或真实调用”改为“代码已接入、已启用（2026-11-06，feature/deepseek-integration 分支，未合并）；尚未注册（无 API Key）、尚未真实调用”。
- 原因：代码已经接入并启用，文档记录与实际不一致；注册和真实调用确实还没做，所以这两项保留原样。

### 提交 3：`4463d85c` — 模型参数传递、路由一致性测试、文档修正

- `main.py`：建 Provider 时传上 `model=routing["model"]`，路由说用哪个模型就用哪个。原来 main.py 丢了路由算出的模型名，Provider 又按默认模型自己猜；现在不会了。
- `providers/muse.py`：`__init__` 增加 `model=None` 参数。不传时默认用 `muse-spark-1.3`；传 `muse-spark-1.3` 正常；传别的名字直接报错（与 DeepSeek 的做法一致）。
- `providers/deepseek.py`：**没有改**。核对过了，它本来就能区分 Flash 和 Pro，Pro 也用 Pro 的价格算，改 main.py 之后就能正常工作。
- `tests/test_router.py`、`tests/test_deepseek.py`：新增上面列出的 7 个模拟测试。
- `README.md`：删掉 Muse“价格较低”的描述（Muse 输入 $1.25、输出 $4.25，比 DeepSeek Flash 输入 $0.15、输出 $0.60 贵 7 到 8 倍，说“价格较低”是错的）。
- `PROJECT_PRINCIPLES.md`：只改第一节，标题改为“本文档的创建任务范围（已完成）”，加一句“已于 2026-11-06 经 BOOS 确认后通过 PR #2 合并到 main”。其他长期原则一个字没动。
- 补充说明：Muse 的 `max_completion_tokens` 经核对是正确的（Meta 官方文档里 `max_completion_tokens` 是正确参数，`max_tokens` 是已废弃的别名），所以 Muse 不用改；DeepSeek 和 Meta 在这个参数上正好相反。

### 提交 4：`0e5ca259` — README 中 CONTROLLER 误导性描述修正

- `README.md` 两处：
  - 第 211 行原来写“如果要换主控，在 config.py 里改 CONTROLLER 即可”，这是错的，只改 CONTROLLER 不会改变路由。改为：要换主控，得同时改两处——router.py 里的路由表（真正决定路由）和 config.py 里的 CONTROLLER（角色名），让两者保持一致。
  - 第 15 行主控那段加了一句：目前 CONTROLLER 只是角色名，实际路由由 router.py 里的固定映射决定，改 CONTROLLER 本身不会改变路由。

## 三、分支状态

- 分支 `feature/deepseek-integration` 比 main **领先 5 个提交，落后 0 个**。
- 5 个提交：`64606626`（接入 DeepSeek）、`594c8aa6`（改 max_tokens）、`ccf45d1b`（原则文档状态）、`4463d85c`（模型参数传递等）、`0e5ca259`（README CONTROLLER 描述）。
- **main 没有被修改**（main 最新仍为 `d6c6fd77`），**没有合并**，没有创建其他分支。
- 本次修改没有增加动态主控，没有重构路由，没有扩大开发范围。
