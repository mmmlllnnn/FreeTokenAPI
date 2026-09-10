# FreeTokenAPI

**版本 2.0.0** · [English](README.md) · [更新记录](CHANGELOG.md)

把你自己的 **DeepSeek / Qwen 网页账号**接入本地 API，提供 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三种兼容接口，支持流式回复、客户端工具调用、附件和网页后端原生搜索。

**推荐客户端：[Pi Agent](https://github.com/earendil-works/pi)，可通过 [CC Switch](https://github.com/farion1231/cc-switch) 配置。**

> 这是非官方网页服务适配器，不是厂商官方 API。请使用自己的账号并遵守上游服务条款。“免费”不代表无限额度、永久稳定或不受限流；模型能力和权限以网页账号实际情况为准。

## 快速开始

### 1. 环境

- Python **3.10+**，推荐 3.12。
- 使用 DeepSeek 时需要 **Node.js LTS** 计算 PoW；只用 Qwen 可不安装。
- 至少一个能在 [DeepSeek](https://chat.deepseek.com) 或 [Qwen 国际版](https://chat.qwen.ai) 正常聊天的账号。

### 2. 本地配置

仅在没有 `.env` 时，把 [.env.example](.env.example) 复制为 `.env`。不要覆盖已有凭据。至少填写一项：

```dotenv
DEEPSEEK_TOKENS=
QWEN_TOKENS=
FREETOKENAPI_HOST=127.0.0.1
FREETOKENAPI_PORT=8000
FREETOKENAPI_TIMEOUT=60
FREETOKENAPI_SEARCH_ENABLED=1
```

多个账号用英文逗号分隔。不使用的平台留空。

登录网页，打开开发者工具 → Network，发送一条消息并查看请求：

- DeepSeek：取 `Authorization: Bearer …` 中 **Bearer 后面的值**。
- Qwen：取 `chat.qwen.ai` 的 `token` Cookie，或对应 Bearer Token。
- 不要复制整行请求头、整串 Cookie，也不要使用官方开放平台 API Key。

**网页登录 Token 只放本机 `.env`，不要填到 Pi、截图或仓库里。** 系统环境变量优先于 `.env`；改配置后需要重启服务。

### 3. 一键启动

Windows 双击 [start.bat](start.bat)，或：

```powershell
.\start.bat
```

macOS / Linux：

```sh
sh ./start.sh
```

脚本会创建或复用项目内 `.venv`，缺少运行依赖时安装。首次没有 `.env` 时生成模板并提示填写。不会自动删除损坏或来自其他系统的虚拟环境；Linux 可能需要先安装发行版对应的 `python3-venv`。

保持终端开启，按 **Ctrl+C** 停止。8000 已占用时不要重复启动，也不要跨系统复制 `.venv`。

手动启动：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Windows 将 Python 路径换成 `.venv\Scripts\python.exe`。

### 4. 检查

打开 [根接口](http://127.0.0.1:8000/)。根接口应返回 `FreeTokenAPI`、`1.0.0`、`status: "ok"`；`adapter_features` 中应包含 `chat_completions_developer`、`qwen_chat_attachments`、`native_web_search`。缺少这些标记，说明仍在运行旧进程。

[健康检查](http://127.0.0.1:8000/health) 和 [模型列表](http://127.0.0.1:8000/v1/models) 成功，不等于每个模型都能真实生成。

## 推荐使用 Pi Agent

### 通过 CC Switch 配置

选择 Pi，添加提供商：

| 协议 | Pi API 标识 | Base URL |
| --- | --- | --- |
| **OpenAI Chat Completions，推荐先用** | `openai-completions` | `http://127.0.0.1:8000/v1` |
| OpenAI Responses | `openai-responses` | `http://127.0.0.1:8000/v1` |
| Anthropic / Claude Messages | `anthropic-messages` | `http://127.0.0.1:8000` |

API Key 填 **`local`**。它只是客户端必填项占位符，不是访问密码。不要追加完整请求路径或重复 `/v1`。

推荐模型：

| 模型 ID | Pi 输入类型 | 说明 |
| --- | --- | --- |
| `qwen3.8-max` | `text`、`image` | 推荐用于 Qwen 文本、图片对话 |
| `qwen3.7-plus` | `text`、`image` | 以账号实际可用性为准 |
| `deepseek-web` | 以接口声明为准：`text`、`image` | 统一网页模型，默认关闭思考 |
| `deepseek-web-thinking` | 以接口声明为准：`text`、`image` | 同一网页模型，默认开启思考；仅在有账号支持时列出 |

图片模型必须在 Pi 声明同时支持 `text`、`image`，否则图片可能根本没有被发送。

不要开启托管 Tool Search、官方托管 `web_search`、远程压缩或“1M 上下文”声明。保留 Pi 的工具执行权限确认。下面的原生网页搜索不依赖这些托管工具。

### DeepSeek 动态模型发现

2.0 移除了全部旧 DeepSeek 模型 ID，**不保留兼容别名**。请将 Pi、CC Switch 等客户端改为 `deepseek-web` 或 `deepseek-web-thinking`。这两个名称是本项目别名，不是官方付费 API 的模型名；都向网页后端发送 `model_type: "default"`，请求显式指定的思考开关优先于别名默认值。

每个 DeepSeek 账号使用自己的客户端身份和认证信息，以 `scope=model` 读取 `/api/v0/client/settings`。仅使用 **`enabled` 和 `switchable` 都为 true 的 `default` 条目**，读取思考、搜索、文件、视觉能力及文件格式、数量、大小和附件/搜索冲突配置，不回退到其他网页模式。

- 能力**按账号缓存在内存中**，默认 300 秒；通过 `FREETOKENAPI_MODEL_CONFIG_TTL_SECONDS` 调整，`0` 表示每次访问刷新。同一账号并发刷新会合并，不把认证信息或 settings JWT 写到磁盘。
- 每次刷新最多等待 10 秒；失败后不继续使用过期能力；失败会冷却至多 30 秒。修复网络问题后重试，或重启服务重新加载。
- `/v1/models` 仅展示当前可用的统一别名，额外返回 `input_modalities`、`thinking_enabled` 和 `capabilities`；能力表示至少一个可用账号支持。思考别名只汇总支持思考的账号。
- 每个请求必须由**同一个账号同时满足全部能力条件**；会话亲和性和账号空闲状态不能绕过检查。需要换账号时，根据传入历史创建新的网页会话。
- 未配置或已禁用的 DeepSeek 不显示占位模型。DeepSeek 查询失败不会隐藏可用 Qwen 模型；如果没有任何可列出模型且配置查询失败，则明确报错。

以下 Pi 示例包含图片输入，因为当前公开统一配置支持视觉；账号存在差异时，请以你自己的 `/v1/models` 返回的 `input_modalities` 配置。能取得模型元数据不等于网页登录 Token 已通过生成请求验证。

### 直接配置 Pi

将以下提供商合并到 `~/.pi/agent/models.json`；Windows 对应 `%USERPROFILE%\.pi\agent\models.json`。不要覆盖其他提供商。

```json
{
  "providers": {
    "free-token-api": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "qwen3.8-max",
          "name": "Qwen3.8-Max (Web)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 65536,
          "maxTokens": 8192
        },
        {
          "id": "deepseek-web",
          "name": "DeepSeek Web (Unified)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 65536,
          "maxTokens": 8192
        },
        {
          "id": "deepseek-web-thinking",
          "name": "DeepSeek Web Thinking (Unified)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 65536,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

这里的上下文 / 输出值是**保守客户端设置，不是实测上游容量或保证值**。遇到长度限制请调低。切换协议时同步修改 `api` 和 Base URL。

无需使用 `supportsDeveloperRole: false` 绕过：Chat Completions 已接收 Pi 的 developer 角色并保持消息顺序，也兼容常见 OpenAI / DeepSeek / Qwen 思考开关。但思考主要仍是开 / 关，不保证真实的 low / medium / high 多档预算。

### 在 Pi 中原生上传 PDF / 文档

项目内已提供 [Pi 原生文件扩展](pi-extension/index.ts)，适配 **Pi 0.85.1**，只使用 Node.js 内置模块，不修改 Pi 安装包。

在本项目目录安装一次：

```sh
node ./pi-extension/install.mjs
```

安装器只向用户级 Pi 设置追加本项目扩展路径，保留其他提供商和设置。**重启 Pi**（更新项目代码后也要重启 FreeTokenAPI），选择 `free-token-api` 下的 **Qwen 或 DeepSeek Web 模型**（`deepseek-web` / `deepseek-web-thinking`）。首次测试建议新开会话，之前已经发送到旧会话的 PDF 乱码不会被自动改写。

在 Pi 输入框中，把明确的附件引用和问题一起发送：

```text
请总结这份 PDF 的主要结论：
@"C:\Users\Administrator\Desktop\report.pdf"
```

替换为真实存在的文件。请在 Pi 空闲时挂载 / 解除文档，先等待当前回复完成或将其停止。支持 Windows 路径、引号、空格、中文文件名、多个 `@` 引用，以及终端拖入后独立一行的绝对路径。仅在正文或代码示例中提到路径，不会自动上传。扩展不接管剪贴板的“文件对象”，请粘贴文件路径或使用 `/attach`。

| Pi 命令 | 作用 |
| --- | --- |
| `/attach "文件路径.pdf"` | 先挂载附件，再发送问题；仅挂载不会调用模型 |
| `/attach "第一份.pdf" "第二份.pdf"` | 挂载多个文档 |
| `/attachments` | 查看当前附件及其 ID |
| `/attachments remove <ID前缀>` | 解除一个附件，后续请求不再发送 |
| `/attachments clear` | 解除全部原生附件 |

CLI 启动参数也支持。下面在普通 PowerShell 运行，不是在 Pi 聊天输入框中输入：

```powershell
pi --provider free-token-api --model qwen3.8-max '@C:\Users\Administrator\Desktop\report.pdf' '请总结这份 PDF'
pi --provider free-token-api --model qwen3.8-max --attach 'C:\Users\Administrator\Desktop\report.pdf' '请总结这份 PDF'
pi --provider free-token-api --model deepseek-web-thinking '@C:\Users\Administrator\Desktop\report.pdf' '请总结这份 PDF'
```

`--attach` 接收一个文件，可完全避开 Pi 启动时的 `@file` 文本展开。普通 CLI `@PDF` 则会在**写入会话、发送模型之前**，由扩展精确替换 Pi 已展开的文件块，不把乱码送入上下文。它不提取 PDF 文本，也不依赖 `read` 工具；原始字节会按协议装配为 `file`、`input_file` 或 `document`。

自动引用支持 PDF、DOC/DOCX、XLS/XLSX、PPT/PPTX、ODT/ODS/ODP、RTF；显式 `/attach` 还支持 TXT、Markdown、CSV、JSON。**Qwen / DeepSeek 网页后端的 PDF 传输已做实网验证；2.0 的模型发现及能力路由有离线回归覆盖。** 图片与普通文本 / 代码引用保留 Pi 原有行为。DeepSeek 图片与文档能力以所选账号的官方配置为准，不再按 Flash / Pro / Vision 固定分流。本扩展不为其他提供商启用文档上传。

若模型仍尝试用本地 `read` 读取已挂载的二进制文档，扩展会拦截这次文本解码，并提示模型使用原生附件；普通文本和图片的 `read` 不受影响。

扩展把不可变文件快照保存在 `~/.pi/agent/cache/freetokenapi-native-files`，设置 `PI_CODING_AGENT_DIR` 时随该目录移动。Pi 会话只记录附件元数据 / 标记，不保存 PDF 乱码或 Base64 大块文本。后续提问、工具往返、上下文压缩与会话恢复仍可使用已挂载附件。客户端最多挂载 50 个文件，总计 100 MiB，单文件也不超过 100 MiB；上游可能更严格。

**快照包含原始文件内容，请保护缓存目录。** 解除附件、卸载扩展，不会删除本地快照或已上传到网页提供方的文件。快照缺失 / 损坏时会明确报错并停止发送，不会悄悄丢掉附件继续请求；可重新挂载或清除附件后继续。

检查或取消注册，不删除文档数据：

```sh
node ./pi-extension/install.mjs --check
node ./pi-extension/install.mjs --uninstall
```

移动项目目录前请先取消注册；若已经移动，请先从 Pi 设置中移除旧扩展路径，再在新位置安装。扩展测试命令为 `node --test tests/pi-native-files.test.mjs`，也已接入 Python 全量回归。

## 图片和其他附件

Qwen 使用真实网页上传链路：申请临时上传凭据 → 上传对象存储 → 必要时解析文档 → 随聊天请求发送文件引用，不是只把文件名塞进提示词。

DeepSeek Web 使用网页文件上传接口及 PoW，必要时等待文件解析完成，再把 `ref_file_ids` 传给生成请求。上传前按所选账号的配置检查文件格式、数量、大小、视觉能力以及附件与搜索是否冲突。

| API | 图片格式 | 文件格式 |
| --- | --- | --- |
| Chat Completions | `image_url` 中放 base64 data URI | `file`，内含 `file.filename`、`file.file_data` |
| Responses | `input_image` 的 `image_url` 放 base64 data URI | `input_file`，内含 `filename`、`file_data` |
| Messages | `image`，使用 base64 `source` | `document`，使用 base64 source 或 `text/plain` 文本 source |

Chat Completions 文件内容块示例：

```json
{
  "type": "file",
  "file": {
    "filename": "notes.txt",
    "file_data": "data:text/plain;base64,SGVsbG8="
  }
}
```

- 图片、文本、PDF 等文件是否可用，取决于具体网页模型、账号、类型和解析限制，不保证每个 Qwen 模型支持所有格式。
- 只接受内联 base64；不会抓取远程图片 / 文件 URL、解析官方 `file_id`，也不会读取服务端本机路径。
- 本地限制为每次最多 50 个文件、单文件 100 MiB；上游可能更严格。
- 同一 Qwen 账号重复上传相同内容可短期复用引用；临时上传凭据不返回给客户端、不存入对话记录。
- Pi 标准模型输入声明仍是 `text` / `image`。需要原生 PDF / 文档时安装上面的扩展，不要在 Pi 模型配置中自行添加不受支持的 `file` 输入类型。
- Qwen 对话附件与 `/v1/images/generations` 图片生成不是同一个功能。

## 原生网页搜索

为了让 Pi 不必每次额外传私有字段，在 `.env` 设置并重启：

```dotenv
FREETOKENAPI_SEARCH_ENABLED=1
```

然后直接提问，例如：“联网查找 Python 官网当前稳定版本，并给出来源链接。”

适配器传递 DeepSeek 的 `search_enabled` / Qwen 的 `auto_search`，并补充简短的能力说明，区分后端原生搜索与客户端函数工具。**搜索由厂商网页后端执行，本项目不实现搜索引擎，也不执行官方托管搜索工具。**

开启后即持续授权模型按需自主使用原生搜索：只要有助于完成任务，就可以主动核实事实、查找文档、比较方案、补充背景或查询最新信息，无需等待用户明确要求搜索或逐次确认，也不限于时效性问题。模型可自行调整关键词、再次搜索；仍尊重明确的不联网要求和隐私约束，客户端工具的执行权限保持不变。

- 请求显式 `search: true/false` 优先于默认配置；未设置环境变量时回退为关闭。
- 开启代表允许使用，不保证每次都重新搜索；已有会话里的搜索结果可能被复用。
- 模型权限、搜索与附件组合、限流等由上游决定；显式搜索请求不会再按本地模型别名静默关掉。
- 这不等于完整支持 OpenAI / Anthropic 托管搜索协议，也不承诺官方事件或精确引用格式。
- 不允许联网的敏感对话应关闭此设置，或明确发送 `search: false`。

## Agent 工具执行与任务结束

Pi 和 Claude Code 的本地工具由客户端执行；网页模型仅仅描述操作，并不能修改你的本地文件。FreeTokenAPI 把模型输出转换为真正的客户端工具调用，再在下一轮传回实际执行结果。

- 每次请求都会刷新**当前工具定义及 tool choice**，复用网页会话时也不省略。工具结果只是中间步骤，不代表整个任务已经完成。
- 复用会话只追加最新一批工具结果，不反复重放所有旧结果；历史调用改用可解析的格式，避免模型把描述性调用记录当成下一步输出。
- 兼容 Claude Code 中实际出现的方括号调用格式，但只接受已声明的工具名及完整 JSON 参数；不猜测未知工具，不把引用示例或截断参数变成可执行调用。
- 上游中断或失败会明确报错，不再伪装成成功的 `stop` / `end_turn`；不下发不完整响应中缓存的工具调用。
- **不通过无限自动发送“继续”、伪造工具执行或绕过权限来维持运行。** 模型应继续完成请求及验证；真正受阻、权限被拒绝或用户要求停止时仍应停止。原生附件、后端搜索与本地文件工具保持区分。

更新后请**重启 FreeTokenAPI**；如果旧会话已积累大量假工具调用文本或提前完成的回答，建议新开客户端会话。本次服务端修复无需重装 Pi 附件扩展。根接口出现 `agent_tool_continuation` 表示新代码已加载。

## 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET / HEAD | `/`、`/v1`、`/v1/` | 服务信息 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | Chat Completions |
| POST | `/v1/responses` | Responses |
| GET / DELETE | `/v1/responses/{response_id}` | 查询 / 删除内存中的结果 |
| POST | `/v1/messages` | Messages |
| POST | `/v1/messages/count_tokens` | Token 数量估算 |
| POST | `/v1/images/generations` | Qwen 图片生成 |
| GET | `/health`、`/v1/usage` | 账号状态及可选用量统计 |
| POST | `/v1/tokens` | 校验 / 追加账号并更新本地凭据配置 |

工具由客户端执行，服务端不执行客户端命令。function/custom 工具和 Responses namespace 通过提示词、解析适配，不保证 strict、签名思考内容或官方工具协议的所有语义。

Responses 历史仅在进程内存中，可过期或被淘汰；重启后旧 `previous_response_id` 失效。网页会话复用是另一层基于内容的机制，中途 system/developer 指令仍会为保留历史顺序而禁用复用。

## 配置、隐私与 PoW

完整配置见 [config.py](freetokenapi/config.py)。常用项：

| 环境变量 | 未配置时默认值 | 用途 |
| --- | --- | --- |
| `FREETOKENAPI_HOST` | `127.0.0.1` | 仅建议回环监听 |
| `FREETOKENAPI_PORT` | `8000` | 端口 |
| `FREETOKENAPI_TIMEOUT` | `60` | 上游 HTTP 超时，不是整次生成总时限 |
| `FREETOKENAPI_MODEL_CONFIG_TTL_SECONDS` | `300` | 每账号 DeepSeek 能力缓存有效秒数；`0` 每次刷新 |
| `FREETOKENAPI_SEARCH_ENABLED` | `0` | 默认原生搜索开关 |
| `FREETOKENAPI_CACHE_DIR` | 系统临时目录下 `freetokenapi` | 会话、上下文和用量文件 |
| `FREETOKENAPI_CACHE_DISABLED` | `0` | 设为 `1` 关闭磁盘会话缓存 |
| `FREETOKENAPI_USAGE_ENABLED` | `1` | 设为 `0` 关闭服务端独立用量账本 |
| `FREETOKENAPI_LOG_LEVEL` | `INFO` | 日志等级 |
| `FREETOKENAPI_LOG_FILE` | 空 | 默认只输出终端日志 |

减少磁盘持久化：

```dotenv
FREETOKENAPI_CACHE_DISABLED=1
FREETOKENAPI_USAGE_ENABLED=0
```

这不会删除已有文件，也不会关闭 Responses 内存存储。缓存可能包含对话内容，不要分享 `.env`、缓存或未经检查的日志。

**服务没有客户端 API Key 鉴权和多用户隔离，Token 管理接口也一样。仅用于受信任本机环境，不要暴露到局域网或公网。**

**PoW 不是用量计数。** DeepSeek 在部分请求前要求工作量证明，因此 [pow.py](freetokenapi/pow.py)、JS 求解器和 WASM 都是必要运行组件。C 求解器是可选加速路径，通常直接使用 Node/WASM，无需编译 C。

每次 DeepSeek 上传或生成前即时计算 PoW，不再预取并跨轮缓存，以免 Pi 闲置期间证明过期。只有上传明确因无效 PoW 被拒绝时，才重新计算证明并重试一次。

Pi 的用量显示会读取 API 的 `usage`。关闭服务端独立账本不会移除响应中的用量和必要估算；估算并不是官方 tokenizer、计费结果或上下文容量测量。

## 常见问题

- **Pi / Claude Code 做一步就停，或打印调用但未执行**：更新后重启服务，确认根接口有 `agent_tool_continuation`；新开会话并保持所需客户端工具启用。检查权限拒绝、上游错误，不要靠关闭安全限制解决。
- **Pi 提示 422 / no body**：更新后先重启旧服务进程。本版已接收 developer 和常见 thinking 格式，并返回协议化校验错误；不要靠关闭校验掩盖问题。
- **图片没发送或不识别**：检查 Pi 模型是否声明 `image`、模型权限、文件格式及完整性；极小、损坏、不受支持的文件仍可能被上游拒绝。
- **不能搜索**：检查默认开关并重启，区分原生后端搜索与托管工具；仅有回答并不能证明新发起了搜索。
- **401 / 无有效凭据**：先确认网页能正常聊天，再更新登录 Token。
- **找不到 Node.js**：安装 LTS 后重新打开终端。
- **403 / 429 / 超时**：检查权限、上游可用性、代理、并发和限流。不要把监听地址改成 `0.0.0.0` 来规避代理问题。
- **上下文或旧 response ID 失效**：适当整理历史或新开会话，不启用 1M 声明和不受支持的云端压缩。

## 开发测试

项目从源码目录运行；`pyproject.toml` 用于测试与静态检查配置，不是完整打包发布配置。运行依赖见 [requirements.txt](requirements.txt)。

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m ruff check .
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

Windows 改用 `.venv\Scripts\python.exe`。大多数测试模拟上游，不需要真实 Token。覆盖 Pi 请求形态、SSE、工具结果、附件上传解析及搜索开关；通过测试不等于所有账号、未来网页接口或客户端插件永久可用。

CI 在 Windows、macOS、Linux 上使用 Node.js 24 执行离线测试，并包含 Python 3.10 兼容性任务。CI 不接收网页登录 Token，也不运行需要账号的在线冒烟脚本。Git 忽略 `.env`、虚拟环境、客户端本地配置、缓存及日志；只发布不含凭据的 `.env.example`。

### 验证范围

2.0 新增每账号模型目录、TTL/失败恢复、旧 ID 拒绝及能力路由测试；另以一次匿名只读请求核对当前官方模型配置与缓存，并用新模型 ID 在真实 Pi CLI 上通过三种协议完成模拟后端的 PDF/工具往返检查。以下旧版在线 CLI 结果描述传输与工具层，不代表已用配置好的账号完成 2.0 在线生成测试。

已使用 Pi 0.85.1 的实际适配器，通过三种协议分别连接真实 DeepSeek / Qwen 后端验证图片问答和原生搜索；另通过三种协议验证了 Qwen PDF，并验证了文本附件上传解析。工具往返、错误格式、用量解析有 SDK 级模拟检查和 Python 回归测试。

另有连续多轮工具调用回归，覆盖三种接口、工具定义刷新、禁用工具、旧调用格式、JSON/CDATA 文件内容及中断流。真实文件编辑任务已通过 Claude Code 2.1.266（Messages，DeepSeek / Qwen）及 Pi 0.85.1（Chat Completions / Responses，两后端）验证，中途无需人工输入“继续”。最后的 Pi Messages 实测遇到 DeepSeek 限流、Qwen 网页验证，不能计为实测通过；对应离线多轮协议测试已通过。在线测试使用隔离客户端配置和生成文件，但仍消耗网页账号额度；遇到验证要求时应停止测试。

macOS/Linux 启动脚本已做 POSIX shell 语法及隔离启动路径测试；这两种操作系统上的原生端到端启动仍需在对应机器验证。

## 许可证

[MIT](LICENSE)。分发源码或衍生版本时，请保留原有版权和许可声明。
