---
description: 把任何使用 API Key 的模型客户端指向上下文网关，让它用上 OpenViking 记忆。
---

# 上下文网关

上下文网关让任何使用 API Key 的模型客户端都能用上 OpenViking 记忆。只需把客户端的 Base URL 改成网关地址，把模型服务商的 API Key 换成网关密钥，不用在客户端里安装任何东西，也不用改提示词。用户每发一条新消息，网关都先用它搜索 OpenViking，把相关内容附加到这条消息上，再交给模型。网关还会把对话保存回 OpenViking，OpenViking 再从中提取新的记忆。

网关支持三种常用的模型 API：Anthropic Messages、OpenAI Chat Completions 和 OpenAI Responses（要求每个请求都带完整历史）。它把每个请求转发给你配置的模型服务商，这里称为**上游**；请求用哪种 API 发来，就用同一种 API 转发，网关不做转换。计费、配额和负载均衡不由网关负责；需要这些能力时，可以在网关后面再接一层 LiteLLM、new-api 这类网关，把它添加为上游。

> **注意**：上下文网关是一个单独运行的进程 `openviking-context-gateway`，不在 OpenViking Server 进程里。它和 VikingBot 的网关（`vikingbot gateway`）没有关系。

上下文网关目前处于 Beta 阶段。本页介绍网关的工作方式和客户端接入方法。为团队部署网关和日常运维，见[上下文网关部署与运维](22-context-gateway-operations.md)。

## 网关还是插件

OpenViking 也可以通过运行在 Agent 内部的插件接入 Claude Code、Codex、OpenCode、pi、OpenClaw、Hermes 等 Agent（见 [Agent 集成概览](../agent-integrations/01-overview.md)）。两种方式互相补充，差别都来自它们运行的位置：

| | 上下文网关 | Agent 插件 |
| --- | --- | --- |
| 运行位置 | 在客户端和模型服务商之间，只能看到发给模型的请求。 | 在 Agent 内部，能看到 Agent 的会话、事件和本地工作区。 |
| 适用的客户端 | 凡是能设置 Base URL 和 API Key 的客户端：聊天应用、SDK 和 API 应用、低代码平台、编程 Agent。 | 有 OpenViking 插件的 Agent。 |
| 无法覆盖的客户端 | 用订阅账号登录的客户端（Claude Code 的 Claude 登录、Codex 的 ChatGPT 登录），以及模型调用从厂商服务器发出的客户端，例如 Cursor 和 Trae。 | 没有扩展接口的客户端。 |
| 按项目区分记忆 | 看不到工作目录和代码仓库。需要按项目隔离时，给每个项目使用不同的密钥。 | 自动识别工作区和代码仓库。 |
| 能看到什么 | 补充的记忆和 OpenViking 工具调用在客户端里不可见，要到 Studio 查看。工具调用不经过客户端的权限确认。 | 工具调用出现在对话记录里，并经过 Agent 的权限确认。 |
| 保存时机 | 晚一轮：下一条消息到达时保存上一轮，最后一轮在对话停顿一段时间后保存。 | 跟随 Agent 自己的事件，例如每轮结束时。 |
| 安装与升级 | 所有人共用一个服务，不用在每台机器上安装。上游、密钥和记忆设置集中管理。 | 每台机器分别安装和升级。 |
| 密钥、数据和故障 | 集中保管服务商的 API Key 和尚未保存的对话文本（加密存储），每次模型调用多经过一跳。网关停止服务时，经由它的模型调用都会失败。 | 服务商的 API Key 留在各自的 Agent 里，只有保存的对话进入 OpenViking，不需要额外维护服务。 |

Claude Code、Codex 和 pi 如果需要按项目区分记忆，或者要求每次工具调用都能看到并确认，就继续用插件。没有插件的客户端，以及想集中管理服务商、密钥和记忆设置的场景，用网关。

**两者同时使用。** 请求中出现 OpenViking 插件的迹象时，网关会停止为这段对话补充记忆和保存对话，只负责转发，这样同一份内容不会被补充两次或保存两次。网关根据以下三类迹象判断：

- 插件插入提示词的记忆块：`<openviking-context>`、`<relevant-memories>`、`<relevant-memory>` 或 `<memory-context>`；
- 名称中带 `openviking` 段的客户端工具，例如 `openviking_search` 或 `mcp__openviking__find`，所以把 OpenViking MCP 服务器配置成 `openviking` 这个名字的客户端也算在内；
- `X-OpenViking-Plugin` 请求头，插件和应用的作者可以发送它，明确表示不需要网关补充记忆。

这个判断对这段对话一直有效；想重新用上网关记忆，需要在不带插件的情况下开始新对话。其他对话不受影响。在 Studio 的“请求日志”标签页里，这些请求会标上**检测到 OpenViking 插件**。

## 工作原理

```text
客户端                        上下文网关                            模型服务商
(Base URL = 网关地址,                                               (上游)
 密钥 = 网关密钥)

  请求  --------------------> 1. 校验网关密钥
  (完整历史)                  2. 重放之前补充的记忆
                              3. 新的用户消息：搜索 OpenViking，
                                 把相关内容附加到消息末尾
                              4. 转发请求 ------------------------->
  回复  <-------------------- 5. 流式返回回复 <--------------------
                              6. 把完成的一轮放进保存队列
                                         |
                                         |  搜索、保存对话、提交
                                         v
                                 OpenViking Server
                               (提取记忆、生成摘要)
```

**模型看到什么。** 模型 API 是无状态的，客户端每次请求都要把整段对话重新发一遍。当请求以一条新的用户消息结尾时，网关用这条消息的文本搜索 OpenViking，把结果附加到这条消息的末尾：

```text
发布日期最后定在哪天？

<openviking-context>
Reference material retrieved from the user's OpenViking memory.
viking://user/alice/memories/events/release-planning.md
团队把 2.0 版本的发布推迟到了 11 月第一周。
</openviking-context>
```

客户端看不到这段内容：它不出现在回复里，客户端自己保存的历史也保持原样。不过服务商返回的 token 用量包含这部分，网关把用量原样传回客户端。每条用户消息只搜索一次；工具步骤、子 Agent 调用以及生成标题之类的辅助请求，沿用已经补充的内容。同一段对话里，同一条记忆只补充一次，单条消息和单段对话的预算也限制了补充的总量。

**重放。** 在这段对话后续的每个请求里，网关都把每个记忆块放回它最初附加的那条消息上，逐字节保持一致。服务商按前缀缓存提示词，Claude 的思考签名又覆盖了之前的对话，所以历史必须保持一致：这样服务商的缓存才能持续命中，Claude 也不会拒绝这段对话。

**对话什么时候保存。** 网关把已完成的轮次保存到密钥所属用户的 OpenViking 会话里，这些会话名为 `context-gateway-…`。下一条用户消息到达时，网关才保存上一轮，因为这时才能确认客户端保留了它，所以重新生成或被放弃的回答不会被保存。对话的最后一轮在停顿 10 分钟后保存，随后网关提交会话，OpenViking 在后台从中提取记忆。会话里待提交的内容累计到 20,000 token 时，网关也会提交一次。保存前，网关会去掉自己添加的内容，以及 `<system-reminder>` 这类客户端噪声。子 Agent 请求、辅助请求和 token 计数请求从不保存。

**长对话。** 一段对话已保存的内容达到约 30,000 token 后，网关让 OpenViking 把较早的部分归档并生成摘要。此后模型收到的是这份摘要加上最近三轮的原文，而不是完整历史，对话因此不会撑满模型的上下文窗口。提示词在每个归档点变化一次，服务商缓存在那里会失效一次。详见[长对话](22-context-gateway-operations.md#长对话)。

以上数字都来自密钥使用的**上下文配置**。你可以在其中调整预算和时间，也可以分别关闭每项功能。

**OpenViking 工具。** 上下文配置还可以允许模型在回答时主动搜索和读取 OpenViking。支持 Chat Completions、完整历史的 Responses 和 Anthropic Messages，包括流式和非流式回复，默认关闭。启用方法和客户端要求见 [OpenViking 工具](22-context-gateway-operations.md#openviking-工具)。

## 快速开始

下面的流程在一台机器上运行 OpenViking Server、网关和测试客户端。你需要：

- 一套能正常工作的 OpenViking，`ov.conf` 中已经配置好 embedding 和 VLM 模型（见[快速开始](../getting-started/02-quickstart.md)）；
- Python 3.10 或更高版本；
- 一个模型服务商的 API Key。订阅登录和 Coding Plan 密钥无法通过网关使用。

### 1. 安装网关

把 `context-gateway` 可选依赖安装到 OpenViking 所在的环境：

::: code-group

```bash [pip]
pip install "openviking[context-gateway]"
```

```bash [uv]
uv tool install "openviking[context-gateway]" --upgrade
```

:::

安装后，`openviking-context-gateway --help` 会打印命令用法。在 Linux 和 macOS 上，还可以加装 `context-gateway-fast`（`"openviking[context-gateway,context-gateway-fast]"`），换用更快的事件循环和 HTTP 解析器。

### 2. 生成加密密钥和管理令牌

网关需要两样东西：一个加密密钥，用来加密它存储的数据；一个管理令牌，OpenViking Server 凭它代你调用网关的管理接口。两者各生成一次，保存在只有你能读取的文件里：

```bash
mkdir -p ~/.openviking
cat > ~/.openviking/context-gateway.env <<EOF
export OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY="$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
export OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
EOF
chmod 600 ~/.openviking/context-gateway.env
```

OpenViking Server 和网关都要读取这两个环境变量，所以启动它们的每个终端都要先加载这个文件。加密密钥务必保留好：一旦更换，网关就读不出之前存储的数据。

### 3. 开启 API Key 认证和网关

网关用每个人自己的 OpenViking 密钥访问记忆，所以 OpenViking 必须运行在 API Key 模式。把下面的设置合并进 `~/.openviking/ov.conf`，root key 用一个足够长的随机值：

```json
{
  "server": {
    "auth_mode": "api_key",
    "root_api_key": "<root-key>"
  },
  "context_gateway": {
    "enabled": true,
    "public_url": "http://127.0.0.1:1935"
  }
}
```

其余设置保持默认：网关监听 `127.0.0.1:1935`，通过 `http://127.0.0.1:1933` 访问 OpenViking，数据存放在 `~/.openviking/context-gateway`。`public_url` 是客户端使用的地址，Studio 的接入说明会显示它。

### 4. 启动 OpenViking Server

```bash
source ~/.openviking/context-gateway.env
openviking-server
```

如果服务已经在运行，就从这个终端重新启动它，让它同时读到新配置和管理令牌。

### 5. 创建账号和用户密钥

在另一个终端里，用 root key 创建一个账号及其首位管理员，做法见[认证](04-authentication.md)：

```bash
curl -X POST http://127.0.0.1:1933/api/v1/admin/accounts \
  -H "X-API-Key: <root-key>" \
  -H "Content-Type: application/json" \
  -d '{"account_id": "acme", "admin_user_id": "alice"}'
# 返回：{"result": {"account_id": "acme", "admin_user_id": "alice", "user_key": "..."}}
```

保存 alice 的 `user_key`。在这个流程里它有两个用途：在 Studio 里以账号管理员身份登录，以及作为网关访问 alice 记忆时使用的 OpenViking 密钥。团队使用时，给每个人单独创建用户密钥（调用 `POST /api/v1/admin/accounts/acme/users`，并指定 `"role": "user"`）。

### 6. 启动网关

```bash
source ~/.openviking/context-gateway.env
openviking-context-gateway --config ~/.openviking/ov.conf
```

检查它能否连上 OpenViking：

```bash
curl -s http://127.0.0.1:1935/health
# {"status":"ok","service":"context-gateway","openviking":{"status":"ok","healthy":true,"version":"…","auth_mode":"api_key"}}
```

刚启动时，`openviking` 可能还显示 `{"status":"starting"}`。如果显示 `"status":"degraded"`，见[故障排查](22-context-gateway-operations.md#故障排查)。

### 7. 在 Studio 中完成设置

打开 <http://127.0.0.1:1933/studio>，进入**连接设置**，把 alice 的密钥同时填入**用户 API 密钥**和**管理员 API 密钥**。然后在侧边栏的“设置”分组里选择**上下文网关**。第一个请求到达之前，“概览”标签页会显示**快速开始**清单，步骤与下面相同：

1. **添加上游。** 在“上游”标签页选择**添加上游**。填写名称，选择服务商，再选择客户端使用的协议（本流程用 Chat Completions），填写 Base URL（例如 `https://api.openai.com/v1`），保持选中**由网关保管 API Key**，再粘贴服务商的 API Key。保存后在上游列表里点**测试**，确认网关能连上服务商。
2. **创建上下文配置。** 在“上下文配置”标签页选择**使用推荐设置创建**，会创建一份名为“默认”的配置。
3. **签发网关密钥。** 在“密钥”标签页选择**签发密钥**。填写名称，在 **OpenViking 用户**中选择 alice（账号里只有她一个用户时已经自动选好），再选择“默认”配置和刚添加的上游，然后签发。**复制网关密钥**对话框只显示一次完整的 `ovcg_…` 密钥，关闭之前先复制好。
4. **接入客户端。** “接入”标签页列出了每种客户端的配置，并已填好你的网关地址。下文[接入客户端](#接入客户端)也列出了同样的配置。

### 8. 发送测试请求

```bash
export GATEWAY_KEY='ovcg_...'

curl -s http://127.0.0.1:1935/v1/models -H "Authorization: Bearer $GATEWAY_KEY"
# {"object":"list","data":[{"id":"…","object":"model","owned_by":"context-gateway"}]}

curl -s http://127.0.0.1:1935/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -H "X-OpenViking-Session: quickstart-1" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "记住：我喜欢简短的回答。"}]}'
```

模型列表里是你在上游中填写的模型和别名；如果上游的模型列表留空，这里也是空的。第二条命令应该返回服务商的一次正常回复。

### 9. 确认记忆生效

- **请求经过了网关。** 打开“请求日志”标签页，刚才的请求显示为**新消息**，状态为 200。OpenViking 中有相关内容时，**记忆**列会显示补充的条数（例如 `+3`）；全新账号下这一列为空。
- **对话已保存。** 用同一个 `X-OpenViking-Session` 请求头再发一条消息。第二条消息一到，第一轮就会保存；最后一轮在停顿 10 分钟后保存。之后，以 alice 身份连接 Studio 时，这段对话会以 `context-gateway-…` 会话的形式出现在**会话**页面。
- **记忆已提取。** 会话提交之后，OpenViking 才会提取记忆：对话停顿 10 分钟时提交一次，待提交内容累计到 20,000 token 时也会提交。提取在后台进行，需要稍等片刻。然后换一个会话请求头的值开始新对话，问一个依赖这条记忆的问题，例如“你的回答应该多长？”。**记忆**列会显示召回的条数，回复也应该用上了这些记忆。

试用阶段想更快看到效果，可以另建一份上下文配置，把**最新回复等待时长**调短，用它签发一个密钥，再用这个密钥开始新对话。上下文配置的修改只对之后开始的对话生效。

## 接入客户端

每个客户端都需要两样东西：网关地址和网关密钥。示例使用 `https://ov.example.com`，请换成你自己的网关地址；Studio 在上下文网关页面顶部和“接入”标签页都会显示它。另外，客户端所用的密钥必须绑定一个已启用的上游，这个上游要支持客户端的协议，并提供客户端请求的模型。

| 客户端 | 上游协议 | Base URL | 对话识别方式 |
| --- | --- | --- | --- |
| [Claude Code](#claude-code) | Anthropic Messages | `https://ov.example.com` | Claude Code 的会话请求头 |
| [Codex CLI](#codex-cli) | Responses | `https://ov.example.com/v1` | Codex 的会话请求头 |
| [聊天客户端和 SDK](#聊天客户端和-sdk) | Chat Completions（或 SDK 使用的 API） | `https://ov.example.com/v1` | `X-OpenViking-Session`，需要自己发送 |
| [Open WebUI](#open-webui) | Chat Completions | `https://ov.example.com/v1` | 连接上配置的 `X-OpenViking-Session` 请求头 |
| [OpenCode](#opencode) | Chat Completions | `https://ov.example.com/v1` | OpenCode 的会话请求头 |
| [pi](#pi) | Chat Completions | `https://ov.example.com/v1` | 通常根据对话历史匹配 |
| [火山方舟 SDK](#火山方舟-sdk) | 三种均可 | `https://ov.example.com/api/v3` 或 `https://ov.example.com/api/compatible` | 取决于客户端发送的请求头 |

Open WebUI、OpenCode 和 pi 的配置依据各自文档中的服务商设置编写，请把它们当作起点，配好后检查一下效果：在同一段对话里发两条消息，然后在“请求日志”中展开这两个请求。它们应该属于同一段对话，而且第二条消息到达后，第一轮已经保存。

### Claude Code

Claude Code 使用 Anthropic Messages。启动前设置三个环境变量：

```bash
export ANTHROPIC_BASE_URL=https://ov.example.com
export ANTHROPIC_AUTH_TOKEN='<gateway-key>'
export CLAUDE_CODE_GATEWAY_HINT_HEADERS=1
```

- 地址不要带 `/v1`，Claude Code 会自己补全路径。
- `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` 让 Claude Code 给子 Agent、上下文压缩和后台请求加上标记，网关就不会为这些请求搜索记忆，也不会把它们当成对话轮次保存。
- Claude Code 会发送自己的会话 ID，所以网关能自动识别对话，`--resume` 之后也一样。
- 上游必须接受 Claude Code 请求的模型名。可以把上游的模型列表留空、列出这些名称，或者用模型别名把它们映射到服务商提供的模型。
- Claude 订阅登录无法通过网关使用，携带订阅令牌的请求会被拒绝。请为上游配置服务商的 API Key。
- 如果 Claude Code 同时装了 OpenViking 插件或 OpenViking MCP 服务器，网关会让出这些对话（见[网关还是插件](#网关还是插件)）。

### Codex CLI

Codex 使用 Responses API，每个请求都带完整历史，正好满足网关的要求。在 `~/.codex/config.toml` 中添加一个服务商，两个顶层设置必须写在所有 `[section]` 之前：

```toml
model_provider = "openviking"
model = "<model>"

[model_providers.openviking]
name = "OpenViking Context Gateway"
base_url = "https://ov.example.com/v1"
wire_api = "responses"
env_key = "OPENVIKING_GATEWAY_KEY"
```

然后在运行 Codex 的环境里设置网关密钥：

```bash
export OPENVIKING_GATEWAY_KEY='<gateway-key>'
```

- 密钥绑定的上游必须使用 Responses，并提供 `<model>`。
- Codex 会先尝试 WebSocket 连接，网关拒绝后它会自动改用 HTTP。
- Codex 会发送自己的会话 ID，所以网关能自动识别对话，`codex resume` 之后也一样。
- 遇到不认识的模型名，Codex 可能提示缺少模型元数据，这个提示不影响请求。
- ChatGPT 登录无法通过网关使用，请在上游配置服务商的 API Key。

### 聊天客户端和 SDK

支持 OpenAI Chat Completions API 的客户端和 SDK 都能使用网关。在客户端要求填写 OpenAI 兼容服务商的地方填入：

```text
Base URL: https://ov.example.com/v1
API key: <gateway-key>
Model: <model>
X-OpenViking-Session: <conversation-id>
```

`X-OpenViking-Session` 请求头可选，但建议发送，它告诉网关请求属于哪段对话。取值要在同一段对话内保持不变、不同对话之间互不相同，例如应用里的聊天 ID。不发送时，网关根据历史匹配对话，见[网关如何识别对话](#网关如何识别对话)。

使用 OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="https://ov.example.com/v1", api_key="<gateway-key>")

reply = client.chat.completions.create(
    model="<model>",
    messages=[{"role": "user", "content": "发布日期最后定在哪天？"}],
    extra_headers={"X-OpenViking-Session": "chat-42"},
)
print(reply.choices[0].message.content)
```

另外两种 API 的 SDK 用法相同。Anthropic SDK 指向 `https://ov.example.com`。Responses API 使用 `https://ov.example.com/v1`，每个请求都要带完整历史并设置 `store: false`，其他 Responses 请求会直接转发，不补充记忆。上游使用的 API 必须和 SDK 一致。

流式调用 Chat Completions 时，还要请求返回用量（`"stream_options": {"include_usage": true}`）。否则服务商不会为流式回复报告 token 数，Studio 无法显示用量，网关也无法判断长对话是否快要撑满上下文窗口。

### Open WebUI

在 Open WebUI 中添加一个 OpenAI 兼容连接（Admin Panel → Settings → Connections），Base URL 填 `https://ov.example.com/v1`，密钥填网关密钥。再给这个连接添加以下自定义请求头，这样每个聊天都是一段独立的对话，Open WebUI 的后台任务也能被识别出来：

```json
{
  "X-OpenViking-Session": "{{CHAT_ID}}",
  "X-OpenViking-Task": "{{TASK}}"
}
```

同时在 Open WebUI 的环境变量中设置 `RAG_SYSTEM_CONTEXT=true`。这样 Open WebUI 会把从附件中检索到的内容放进 system 消息，而不是临时改写你这一轮的消息，对话历史在请求之间就能保持稳定。

- **一个连接密钥只对应一位记忆归属者。** 通过这个连接聊天的所有 Open WebUI 用户，读写的都是密钥背后那位 OpenViking 用户的记忆。要么只把这个连接给一个人使用，要么接受这些用户共享记忆。
- Open WebUI 还会通过同一个连接发送后台请求（生成标题、标签和追问建议）。网关能识别标题和摘要请求。如果请求日志里有其他后台任务被标成**新消息**，就把 Open WebUI 的任务模型改成一个不经过网关的连接。

### OpenCode

在 `~/.config/opencode/opencode.json` 中添加一个服务商，与已有的设置合并：

```json
{
  "provider": {
    "openviking": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenViking",
      "options": {
        "baseURL": "https://ov.example.com/v1",
        "apiKey": "{env:OPENVIKING_GATEWAY_KEY}"
      },
      "models": {
        "<model>": {}
      }
    }
  }
}
```

把网关密钥导出为 `OPENVIKING_GATEWAY_KEY`，然后在 OpenCode 中选择 `openviking/<model>`。上游必须使用 Chat Completions。如果同时安装了 OpenViking 的 OpenCode 插件，网关会让出这些对话。

### pi

在 pi 的模型配置 `~/.pi/agent/models.json` 中添加一个服务商：

```json
{
  "providers": {
    "openviking": {
      "baseUrl": "https://ov.example.com/v1",
      "apiKey": "OPENVIKING_GATEWAY_KEY",
      "api": "openai-completions",
      "models": [
        {
          "id": "<model>"
        }
      ]
    }
  }
}
```

`apiKey` 填的是保存网关密钥的环境变量名，启动 pi 前先导出它。上游必须使用 Chat Completions。除非 pi 发送了[网关如何识别对话](#网关如何识别对话)中列出的某个请求头，否则网关会根据历史匹配它的对话。如果 pi 自己的 OpenViking 扩展处于启用状态，网关会让出这些对话，两者选一个使用即可。

### 火山方舟 SDK

已经配置好火山方舟的客户端和 SDK，只需替换域名。网关既接受方舟自己的路径（`/api/v3/chat/completions`、`/api/v3/responses`、`/api/v3/models` 和 `/api/compatible/v1/messages`），也接受标准的 `/v1` 路径：

| 原方舟地址 | 网关地址 |
| --- | --- |
| `https://ark.cn-beijing.volces.com/api/v3` | `https://ov.example.com/api/v3` |
| `https://ark.cn-beijing.volces.com/api/compatible`（Anthropic 兼容） | `https://ov.example.com/api/compatible` |

用网关密钥代替方舟 API Key。使用方舟 Python SDK：

```python
from volcenginesdkarkruntime import Ark

client = Ark(base_url="https://ov.example.com/api/v3", api_key="<gateway-key>")
```

路径只决定客户端使用哪种 API。请求仍然发往密钥绑定的、使用这种 API 并提供该模型的上游，通常是服务商选为“火山方舟”的上游（见[上游](22-context-gateway-operations.md#上游)）。

## 网关如何识别对话

网关需要知道请求属于哪段对话。一段对话固定使用同一份上下文配置和同一个上游，并保存到同一个 OpenViking 会话。网关按以下顺序取第一个出现的请求头：

1. `X-OpenViking-Session`
2. `thread-id`
3. `x-claude-code-session-id`（Claude Code）
4. `x-opencode-session-id`（OpenCode）
5. `x-session-id`
6. `session-id`（Codex CLI）

这些请求头都没有时，网关查看历史中最近一条助手回复。如果恰好有一段之前的对话产生过这条回复，请求就并入那段对话；否则开始一段新对话。两段对话不会仅仅因为开场消息相同而被合并。普通的一问一答这样就能识别，但重试、重新生成的回答和内容完全相同的对话存在歧义，可能被当成新对话。所以只要客户端允许设置请求头，就发送 `X-OpenViking-Session`。

对话归属于密钥背后的 OpenViking 用户，并且按 API 分开。同一用户的两个网关密钥如果发送相同的会话值，会共享同一段对话；同一个会话值用在另一种 API 上，则是另一段对话。网关在转发前会去掉所有 `X-OpenViking-*` 请求头，服务商看不到它们。

请求没有归入已有对话时，记忆照样工作：新消息照常搜索，之前补充的记忆也照常重放，因为网关是根据消息本身找到这些记忆的。区别在于这个请求会开始一段新对话：它的轮次保存到新的 OpenViking 会话，单段对话预算从头计算，并使用密钥当前的上下文配置。

## 使用须知

- **记忆按消息补充。** 每条新消息的第一次模型调用要等 OpenViking 搜索完成，最多等上下文配置里的**超时时间**（默认 2 秒）。OpenViking 超时或不可用时，这条消息不带记忆直接发给模型，之后也不会补上。OpenViking 停止服务期间，模型请求照常可用。
- **设置只对新对话生效。** 一段对话始终使用它开始时的上下文配置和上游。Claude Code、Codex 这类长期运行的客户端会把一段对话延续很多天，所以修改上下文配置后，要等它们开始新对话才会生效。对话闲置 30 天后，网关会删除它的状态。
- **保存晚一轮。** 对话的最后一轮在停顿 10 分钟后保存，记忆要等 OpenViking 处理完提交后才会出现。
- **改动过的历史会重新保存。** 如果客户端编辑或删除了之前的消息、在回答保存后重新生成，或者压缩了对话，网关会把当前的完整历史保存到一个新的 OpenViking 会话。已经保存的轮次留在旧会话里。
- **客户端仍可能自行压缩。** 客户端按自己保存的历史计算 token。即使网关发给模型的是摘要而不是较早的轮次，客户端也可能自己决定压缩对话。
- **一个密钥只对应一位记忆归属者。** 使用同一个密钥的人共享其背后 OpenViking 用户的记忆。请给每人签发一个密钥；想让不同客户端使用不同的上下文配置，就按客户端再分开签发。
- **不支持订阅登录。** 携带 Claude 订阅令牌的请求会被拒绝。请在上游配置服务商的 API Key。
- **Responses 需要完整历史。** 依赖服务商端状态的 Responses 请求（`previous_response_id`、`conversation`、`background`），以及没有设置 `store: false` 的请求，会直接转发，不补充记忆。之后查询这些响应时，请求仍然发往创建它们的上游。
- **其他接口直接转发。** 三种模型 API 和模型列表之外的接口（例如 embeddings）不补充记忆，直接转发给密钥绑定的、提供所请求模型且优先级最高的上游。
- **请求有大小上限。** 超过 32 MiB 的请求体会被拒绝，运维人员可以调整这个上限。

## 下一步

- [上下文网关部署与运维](22-context-gateway-operations.md)：用 Docker Compose 或 Helm 部署，管理上游、上下文配置和密钥，排查问题。
- [认证](04-authentication.md)：创建账号、用户和他们的密钥。
- [公网访问与反向代理](12-public-access.md)：为 OpenViking 配置 HTTPS。
- [Agent 集成概览](../agent-integrations/01-overview.md)：为支持插件的 Agent 安装插件。
- [MCP 集成](06-mcp-integration.md)：让 MCP 客户端直接使用 OpenViking 工具。
