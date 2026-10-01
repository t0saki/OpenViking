# 上下文网关

OpenViking Context Gateway 是独立运行的模型 API 网关，与 VikingBot 的 Bot
Gateway 不同。客户端修改 base URL 和 API key 后，网关自动召回、重放上下文，
并将用户保留的对话写入 OpenViking。支持 Anthropic Messages、Chat Completions
与每轮发送完整历史且 `store: false` 的 Responses。

## 启动与配置

安装本分支构建的 OpenViking wheel。生成 Fernet 密钥和至少 32 字符的管理令牌，
分别放入 `OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY` 和
`OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN` 环境变量，不写进 Git。

在 `ov.conf` 中加入：

```json
{
  "context_gateway": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 1935,
    "workers": 2,
    "url": "http://127.0.0.1:1935",
    "openviking_url": "http://127.0.0.1:1933",
    "public_url": "https://ov.example.com",
    "storage_path": "~/.openviking/context-gateway"
  }
}
```

执行 `openviking-context-gateway --config /path/to/ov.conf`。OpenViking Server
使用同一配置与管理令牌；Studio 请求经服务端的账户管理员鉴权后转发给网关。
模型流量走 1935 端口。共享部署需要开启 OpenViking API key 鉴权；dev 模式只允许
网关监听本机。

Studio 的「上下文网关」提供概览、上游、上下文配置、分发、请求日志和接入指引。
先建立上游和上下文配置，再使用同账户的 OpenViking 用户 key 签发网关 key。
ROOT key 会被拒绝，网关 key 只显示一次。凭据、注入记录、待写消息均加密存储。
请同时备份两个 SQLite 数据库与加密密钥，不能直接换密钥后重启。

配置修改只影响新会话。默认每轮召回 1600 token、会话总上限 6000 token、超时
2 秒；捕获落后一轮，最后一轮在空闲 10 分钟后提交。提交按完整用户轮次保留至少
10 条近期消息，长会话归档替换为模型保留最近 3 轮。上游实际用量直接返回客户端。

## 接入

Claude Code：

```bash
export ANTHROPIC_BASE_URL=https://ov.example.com
export ANTHROPIC_AUTH_TOKEN='<网关 key>'
export CLAUDE_CODE_GATEWAY_HINT_HEADERS=1
```

Codex 在 `config.toml` 顶层设置 `model_provider = "openviking"`，再添加：

```toml
[model_providers.openviking]
name = "OpenViking Context Gateway"
base_url = "https://ov.example.com/v1"
wire_api = "responses"
env_key = "OPENVIKING_GATEWAY_KEY"
```

通用聊天客户端的 base URL 为 `https://ov.example.com/v1`，可用
`X-OpenViking-Session` 提供会话 ID。Open WebUI 支持模板值 `{{CHAT_ID}}`，
请求类型头 `X-OpenViking-Task` 使用 `{{TASK}}`，建议开启 `RAG_SYSTEM_CONTEXT`。
建议明确传入会话 ID；未提供时以首个有效前缀推断，相同开头可能共享会话快照，
截断历史可能形成新会话。一个共享连接 key 下的所有人都归到同一个 OpenViking 用户。Studio 也提供 pi
和 OpenCode 的配置。

发现 OpenViking 插件标记后，该会话永久停止新增召回与捕获，原有注入继续重放。
工具续轮、子 agent、辅助请求和 token 计数只重放。WebSocket 返回 426；带
`previous_response_id`、`conversation`、`background` 的 Responses 原样透传。

## 部署与验收

Compose 使用 `docker compose --profile context-gateway up -d`；容器内需将网关
监听地址设为 `0.0.0.0`，管理地址设为 `http://context-gateway:1935`，OpenViking
地址设为 `http://openviking:1933`，存储目录设为 `/app/.openviking/context-gateway`。
Caddy 将 `/v1/*`、方舟原生路径和签名上传代理转发给网关。Helm 开启 `contextGateway.enabled` 并指定包含
`encryption-key`、`admin-token` 的 `existingSecret`。SQLite WAL 仅用于单机多进程，
不能跨机器共享网络文件系统。

会话整体按最后使用时间保留 30 天。删除用户时同时调用
`DELETE /api/v1/admin/context-gateway/users/{user_id}/data`，吊销网关 key 并清理记录。服务端的持久化用户/账户删除任务也会自动执行此清理；网关不可用时保留任务供重试。
日志默认不保存正文、召回原文或密钥；概览分别展示每轮首调与轮内缓存命中率。

模拟测试运行命令：

```bash
pytest tests/context_gateway --confcutdir=tests/context_gateway -o addopts=''
node scripts/sync-context-gateway-rules.mjs --check
node examples/memory-plugin-shared/sync.mjs --check
```

测试序列为合成样例，本地模拟上游通过不代表真实客户端或厂商缓存验收通过。
设置 `OV_CG_TEST_BASE_URL`、`OV_CG_TEST_KEY`、`OV_CG_TEST_MODEL` 后，可执行
`python scripts/context_gateway_acceptance.py --protocol anthropic`，再分别使用
`chat`、`responses`。完整验收还需真实 Claude Code、Codex 和聊天客户端的录制回放，
检查签名、缓存覆盖前缀与超过模型上下文窗口后的对话。

召回失败会固化为空决策，不在后续补注入。无法无损处理的数值原样透传。
已知 Anthropic 注入记录缺失时剥离旧 thinking 并记录原因。归档摘要未就绪且即将
超过窗口时有限等待，超时后固定使用近期历史并记录 `archive_wait_timeout`。
OpenViking 写消息接口没有幂等键，在写成功但确认前崩溃仍可能重复一个批次。
更多参数、运维说明和限制见[英文指南](../../en/guides/15-context-gateway.md)。

## 二期：隐藏工具执行

在 Studio 的上下文配置中开启 `gateway_tools`。仅支持 Chat Completions，默认
关闭；默认白名单为 `search`、`read`、`list`，模型看到的名称统一加上
`openviking_` 前缀。网关维护带版本的参数定义，用该用户的 OpenViking key 调用
已有 MCP。可以配置 MCP 或插件的客户端继续使用原来的接入方式。

会话首请求冻结网关工具定义。已有 OpenViking 工具、`n > 1`、结构化输出、强制
指定工具、非 function 工具、上游禁用工具等情况不注入。DeepSeek 思考模式默认
不注入，只有显式 `thinking.type: disabled` 才可启用。已冻结工具的会话改用
不兼容参数时返回 409，需新建会话；辅助请求保留工具定义并设置
`tool_choice: none`。

文本和推理增量实时输出；网关工具调用被隐藏。客户端工具在名称和参数完整后
正常返回。混合调用先执行网关工具，下一次请求将完整助手调用和网关结果还原到
客户端结果之前。隐藏往返加密保存，包含 `reasoning_content` 和未知厂商字段；
按客户端可见回答匹配分支，编辑或重新生成的不同回答不会复用另一分支。
客户端收到一个完成 ID、一个结束事件和所有上游调用的累计用量；首次调用的
缓存统计不混入隐藏续轮。

默认限制：5 次隐藏往返、单工具 30 秒、结果 64 KiB、整个工具请求 120 秒、
100,000 token 预算。对应参数为 `tool_max_rounds`、`tool_timeout_seconds`、
`tool_result_bytes`、`tool_total_seconds`、`tool_total_tokens`。token 限制结合
上游用量和本地字节估计控制入场及后续输出，不是精确 tokenizer 或计费上限。
达到轮数上限后保留定义并设置 `tool_choice: none`，继续调用则明确报错。
工具失败作为有界结果交给模型；循环失败返回 HTTP 错误，已开始流式输出时发送
SSE 错误。取消或不完整回复不捕获。Chat 重放存储故障返回 503，防止丢失隐藏历史。

写工具 `write`、`add_resource`、`add_skill` 需要同时开启 `allow_write_tools`
并逐项加入白名单。同会话、同调用 ID 与参数通过持久化记录避免并发重复执行；
写入超时或进程崩溃后不自动重做，需要先核查结果。不同调用 ID 视为不同操作，
不保证端到端 exactly-once。

### 文件导入

导入工具仅在首请求包含 shell 工具或附件时加入会话。网关不读取客户端本地路径。

- 有 shell：转调 MCP 获取签名上传指引，将地址改写为
  `<context_gateway.public_url>/context-gateway/uploads?token=…`，模型再调用
  客户端自己的 shell 上传；skill 目录先压缩。`public_url` 必须是客户端可达地址。
- 没有 shell：工具通过 `attachment_index` 选择用户消息中内嵌的
  `file.file_data` / `input_file.file_data`（base64 或 data URL）或附件 `text`。
  Open WebUI 的 `<context><source …>` 提取文本（包括 system context）以 `.txt`
  导入，只保留文本，不声称恢复原始二进制文件。
- 只有 file ID、远程附件地址而没有 bytes 时不抓取；没有 shell 的本地路径需改用附件。

上传代理只转发到固定的 OpenViking `/api/v1/resources/temp_upload`。签名 token
独立授权，服务端校验有效期并消费；不向客户端泄露 OV key，也不在上传请求附加
用户或模型 key。上传受 `max_body_bytes` 限制。Caddy、Helm 已加入上传路由；
自建反向代理需同步配置；网关 CLI 关闭原始访问日志，外部代理的访问日志应省略
该路径的 query，避免记录签名 token。

## 二期：火山方舟

上游选择 `vendor: ark` 和模型 API key，`base_url` 使用方舟 origin，或
Chat/Responses 的 `…/api/v3`、Messages 的 `…/api/compatible/v1`。每种协议
分别建上游。客户端可以使用 `/api/v3/chat/completions`、`/api/v3/responses`
及其 response-ID 子路径、`/api/compatible/v1/messages`，也可继续使用 `/v1/…`。
Caddy 与 Helm 已加入原生路径。

`thinking`、`encrypted_content`、`caching`、`expire_at` 等未知字段原样保留，
Responses 流的 `[DONE]` 保留。增强的 Chat/Responses 会话固定
`prompt_cache_key`；跨进程滑动窗口将该 key 限为每分钟 15 次，隐藏续轮也计数，
超限返回 429 与 `Retry-After`。会话内应保持模型、推理、采样、system 和工具参数
稳定；变化记录 `ark_cache_parameters_changed`。按模型设置 `cache_min_tokens`
（默认 1024），日志根据实际输入用量记录是否达到阈值；缓存命中仍需厂商实测。

标记为 `coding_plan` 的上游默认拒绝，管理员显式开启 `allow_coding_plan` 才可
覆盖。无法仅凭不透明 key 判断订阅类型，需管理员准确标记；Studio 提示共享网关
应使用模型 API key。旧 Context API 不作为增强协议接入。

二期测试包含工具流分片、即时文本、混合调用、重启和分支重放、附件上传、签名
代理、超时、写调用去重、轮数上限、方舟路由与跨进程限流。使用模拟厂商、真实 FastMCP 传输层和合成
对话；真实厂商缓存、真实客户端录制、跨真实模型窗口仍需独立验收。
