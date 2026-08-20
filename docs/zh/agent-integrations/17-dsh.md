# DeepSeek Harness 记忆插件

为 [DeepSeek Harness](https://www.npmjs.com/package/@deepseek-ai/dsh)（`dsh`）接入跨会话长期记忆。该插件是进程内的 Cordis 插件：会话开始时注入 OpenViking 画像，每个模型步骤前做语义召回，实时捕获会话，按 token 阈值 commit，并挂载 OpenViking 的 MCP 工具面。

npm 包：[`@openviking/dsh-memory-plugin`](https://www.npmjs.com/package/@openviking/dsh-memory-plugin) · 源码：[examples/dsh-memory-plugin](https://github.com/volcengine/OpenViking/tree/main/examples/dsh-memory-plugin)

## 工作方式

- **会话开始**（`agent/session-start`）通过 `agent.inject()` 注入 OpenViking 画像块与可用记忆索引。
- **每个步骤**（`agent/pre-step`）用当前步骤输入做语义召回，并把结果作为带来源标记的持久 user 消息追加到同一步骤。它以 `prepend: true` 注册，因此能看到最终成型的消息批次，并排在所有其他贡献者之后追加。
- **每个事件**（`session/event`）直接从 DSH 事件流捕获 user、assistant 以及（可选的）工具结果消息，无需扒取 transcript。
- **回合结束**时，待同步 token 超过 `commitTokenThreshold`（默认 `20000`）即 commit，并保留最近 10 条消息在本地上下文中。
- **工具**来自 OpenViking 的 MCP 工具面，走的是与其他记忆集成同一个 stdio 代理，由 `@deepseek-ai/dsh-mcp-client` 桥接进 DSH，以 `mcp__openviking__search`、`mcp__openviking__read`、`mcp__openviking__remember`、`mcp__openviking__write` 等名字发布给模型——具体有哪些取决于所连服务端。
- **URI 保护**（`tools/pre-execute`）阻止 DSH 的文件系统与 shell 工具把 `viking://` URI 当作本地路径，并提示模型改用桥接过来的 `mcp__openviking__*` 工具。
- 写入失败会进入共享的 OpenViking 待写队列，在下次会话开始时重放。

每个 DSH 会话映射为 OpenViking 中的 `dsh-<session-id>`，子 agent 各自拥有独立会话。工作区推导出的 actor peer 按会话解析，并随每个会话级请求发送。

### 为什么召回不走 system prompt

召回以 pre-step user 消息而非 system prompt 段落注入：DSH 中 persona 声明了 `complete: true` 的 preset（自带的 `minimal` 就是）会在组装后把该 persona 恢复为唯一的 prompt 段落，静默丢弃其他所有贡献。pre-step 注入还让每次注入成为可重放、且对压缩可见的会话事件。

### 工具面是怎么挂上去的

插件不再手写一份工具子集，而是把 DSH 自带的 MCP 桥接挂到 `servers/mcp-proxy.mjs` —— Claude Code、Codex、Cursor、OpenCode 启动的是同一个 stdio 代理：模型拿到的就是所连服务端广告的全部工具，服务端升级即可增加工具而无需插件发版。插件解析好的凭证通过子进程环境变量传给代理，profile 里不需要额外的 MCP 配置；桥接本身随 `@deepseek-ai/dsh` 一起安装，也不用额外装包。

由于代理是每个 profile 一个进程，工具调用带的是启动时解析的 actor peer 而非按会话解析；`mcp__openviking__remember` 也是写进服务端一个短生命周期的会话，而不是当前的 `dsh-<session-id>` 消息流——这与 Claude Code、Codex、Cursor 几个集成的行为一致。召回、捕获、commit 不受影响，仍按每个会话自己的工作区解析 peer。若一个进程要服务多个工作区且需要精确归属工具调用，请显式设置 `OPENVIKING_PEER_ID`。

启动失败是被兜住的：即使服务端的 MCP 端点连不上，召回、捕获、commit 照常工作，桥接自己会重连。

## 前置条件

- `@deepseek-ai/dsh` `0.1.0-rc.6` —— 请安装这个精确版本，`@deepseek-ai/dsh-*` 各包的预发布 tag 并非同步发布
- Node.js `^22.19.0` 或 `>=24`
- 一个可访问的 OpenViking HTTP 服务 —— 用 `curl http://localhost:1933/health` 验证

## 安装

把已发布的包装入 profile（默认 profile 是 `web`）：

```bash
dsh plugin --profile web add @openviking/dsh-memory-plugin
```

或从 OpenViking 仓库检出安装：

```bash
git clone https://github.com/volcengine/OpenViking.git
cd OpenViking
dsh plugin --profile web add ./examples/dsh-memory-plugin
```

包内自带的 `cordis.patch.yml` 会把运行时挂载到一个 Cordis 插件组中，并隔离 `openvikingMemory` 服务。确认已生效：

```bash
dsh --profile web --dump-config
```

DSH 不在统一记忆插件安装器的覆盖范围内，`dsh plugin add` 就是安装方式。

## 配置

凭证解析顺序为 `OPENVIKING_*` 环境变量 → `~/.openviking/ovcli.conf` → `~/.openviking/ov.conf`，与 Claude Code、Codex、OpenCode、pi 插件共用同一条链路。

| 变量 | 用途 |
|------|------|
| `OPENVIKING_URL` / `OPENVIKING_BASE_URL` | 服务端点（默认 `http://127.0.0.1:1933`） |
| `OPENVIKING_API_KEY` / `OPENVIKING_BEARER_TOKEN` | Bearer 凭证 |
| `OPENVIKING_ACCOUNT` / `OPENVIKING_USER` | 可信模式下的 account 与 user |
| `OPENVIKING_PEER_ID` | 显式指定 actor peer |
| `OPENVIKING_WORKSPACE_PEER` | 按 DSH 会话工作区推导 peer（默认开启） |
| `OPENVIKING_RECALL_PEER_SCOPE` | `all` 跨工作区召回，`actor` 隔离召回 |

行为参数写在 Cordis patch 条目里：

```yaml
- insert:
    - id: openviking-memory
      name: '@deepseek-ai/cordis-plugin-group'
      group: true
      isolate:
        openvikingMemory: true
      config:
        - id: openviking-memory-runtime
          name: '@openviking/dsh-memory-plugin'
          config:
            endpoint: http://127.0.0.1:1933
            recallTokenBudget: 2000
            scoreThreshold: 0.35
            captureToolResults: false
            commitTokenThreshold: 20000
            mcpToolCallTimeoutMs: 60000
```

patch 中写的凭证优先于环境变量；行为开关则优先读环境变量。

## 验证

启动 `dsh --profile web`，提一句你在更早会话里说过的事，看回答是否用上了那条记忆。`dsh --profile web --dump-config` 可以确认插件组是否挂载；设置 `OV_DEBUG_LOG=/tmp/ov-dsh.log` 可以追踪召回与捕获的决策过程。

## 常见问题

| 现象 | 排查方向 |
|------|----------|
| 插件没加载 | `dsh --profile web --dump-config` 里应能看到 `openviking-memory`；重新执行 `dsh plugin --profile web add …` |
| 安装时报 `ERESOLVE` | `@deepseek-ai/dsh-*` 各包预发布 tag 不同步；请精确安装 `@deepseek-ai/dsh@0.1.0-rc.6` |
| 召不回任何内容 | `curl http://localhost:1933/health`；检查端点配置，以及 prompt 是否长于 `minQueryLength`（默认 3） |
| 模型看不到 `mcp__openviking__*` 工具 | 桥接会兜住启动失败——看 DSH 日志里的 `mcp-client(openviking)`，并设 `OV_DEBUG_LOG=/tmp/ov-dsh.log` 追踪代理 |
| OpenViking 返回 401 / 403 | 检查 `OPENVIKING_API_KEY`；可信模式部署还要检查 `OPENVIKING_ACCOUNT` 与 `OPENVIKING_USER` |
| 串入了其他项目的记忆 | 设置 `OPENVIKING_RECALL_PEER_SCOPE=actor` |
| 崩溃后没有 commit | commit 由 token 阈值和 teardown 触发；排队的写入会在下次会话开始时重放 |

完整的工具、配置与发布说明见[插件 README](https://github.com/volcengine/OpenViking/tree/main/examples/dsh-memory-plugin)。

## 延伸阅读

- [集成能力参考](./16-capability-reference.md)
