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
安装此分支时使用 `pip install "openviking[context-gateway]"`，网关依赖位于可选组；
`context-gateway-fast` 可额外安装 uvloop/httptools。模型流量走 1935 端口。共享部署需要开启 OpenViking API key 鉴权；dev 模式只允许
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
未提供会话 ID 时，以网关记录过的完整助手回复前缀匹配；最长匹配只有一个归属时复用会话，
否则建立隔离的新会话，捕获和隐藏工具仍可使用。仅有相同用户问候不会合并会话。
Chat 匹配忽略客户端常省略的助手推理元数据。完全相同的对话、重试等情况仍存在歧义，
支持时优先发送每段对话唯一的 header。编辑已捕获历史或客户端压缩会自动重新同步捕获。
一个共享连接 key 下的所有人都归到同一个 OpenViking 用户。Studio 也提供 pi
和 OpenCode 的配置。

发现 OpenViking 插件标记后，该会话永久停止新增召回与捕获，原有注入继续重放。
仅检测用户/system/developer 文本及 OpenViking 工具定义，忽略助手回复、工具结果和附件名。
工具续轮、子 agent、辅助请求和 token 计数只重放。WebSocket 返回 426；带
`previous_response_id`、`conversation`、`background` 的 Responses 原样透传。
有状态 Responses 映射按主键查询，默认 30 天过期（`response_ttl_seconds`），
`store: false` 不保存映射。

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
PYTHONPATH=. python scripts/context_gateway_benchmark.py --concurrency 300
```

测试包含合成序列和脱敏后的真实客户端请求样本；模拟上游通过不代表厂商缓存
或原生签名校验通过。
设置 `OV_CG_TEST_BASE_URL`、`OV_CG_TEST_KEY`、`OV_CG_TEST_MODEL` 后，可执行
以下命令，再分别使用 `chat`、`responses`：

```bash
python scripts/context_gateway_acceptance.py --protocol anthropic --require-cache \
  --output /tmp/gateway-anthropic-acceptance.json
```

脚本运行三轮，第二轮使用 SSE；缓存读取须覆盖前次输入，允许厂商最后一个未满的
缓存块（`--cache-block-tokens`，默认 128，应按厂商调整）。报告仅保存用量和
协议元数据，不保存凭据或消息正文。`--prefix-rows 0` 用于最小连通性检查；
支持 `thinking.type` 扩展的 Chat/Responses 上游可使用 `--disable-thinking`。

原生 Anthropic 签名绑定验收使用 `--binding-check`：开启绑定控制测试版、设置
`prefix_mismatch_behavior: error`，并要求真实 thinking 签名和空的
`input_transformations` 元数据。缺少这些字段的兼容接口即使返回 HTTP 200，
也不能通过此项检查。完整验收还需真实 Claude Code、Codex 和聊天客户端的录制回放，
检查签名、缓存覆盖前缀与超过模型上下文窗口后的对话。
已测结果与待验项目见[真实接口验收记录](../../testing/context-gateway-live-acceptance.md)。

召回失败会固化为空决策，不在后续补注入。无法无损处理的数值原样透传。
已知 Anthropic 注入记录缺失时剥离旧 thinking 并记录原因。上游的 `context_windows`
按真实模型 ID 配置窗口，例如 `{"claude-sonnet-4-5": 200000}`。紧急等待只依据同一
上游、同一模型最近一次返回的真实 input/output 用量；策略 `context_window` 仅作为
单模型部署的可选兜底，默认不设置。请求体字节数、工具定义和图片大小不触发等待。
等待超时后保留完整历史并记录 `archive_wait_timeout`，绝不固化占位摘要。

接管激活按实际同步到 OV 的净化后对话估算 token；后续提交按 OV 的
`pending_tokens` 判断。每个客户端会话只有一份捕获文档，包含 OV 目标、已投递水位、
待投递尾部、保留边界与待处理归档。请求确认尾部，持有租约的 worker 顺序推进水位；
历史编辑、压缩、重新生成或手动重置直接切换 OV 目标并从当前历史重新同步。
已有且仍匹配前缀的替换记录始终重放。

失败轮次阻塞同一会话后续写入，5 次快速尝试后每 5 分钟探测恢复。Studio 展示状态、
原因并提供“重新同步捕获”。归档只由 worker 检查，识别 `.done` / `.failed.json`，
最多每 5 秒一次；终态缺少摘要或 15 分钟仍未解决时释放提交等待，不替换历史。
请求侧只有在真实用量达到阈值、存在可替换当前前缀的 pending 归档时才等待。

服务端的 `source_message_ids` 是来源字段，不提供原子去重。worker 在投递前读取
服务端 live 消息，按来源 ID 对账，以恢复写入成功但响应丢失、或分批写入中断。
commit 意图保存在同一文档，先检查服务端 Phase 1 回执再恢复提交。对账不能替代
服务端的原子幂等保障：超时请求仍在服务端执行、租约失效后旧进程恢复等情况仍有竞态。

此功能尚未发布，开发分支的旧数据库格式不迁移。测试旧分支后升级时应选择新的
`storage_path`；旧数据保留在原路径，不会被自动删除。
更多参数、运维说明和限制见[英文指南](../../en/guides/15-context-gateway.md)。

## 二期：隐藏工具执行

在 Studio 的上下文配置中开启 `gateway_tools`。仅支持 Chat Completions，默认
关闭；默认白名单为 `search`、`read`、`list`，模型看到的名称统一加上
`openviking_` 前缀。网关维护带版本的参数定义，用该用户的 OpenViking key 调用
已有 MCP。可以配置 MCP 或插件的客户端继续使用原来的接入方式。

会话首请求冻结网关工具定义。已有 OpenViking 工具、`n > 1`、结构化输出、强制
指定工具、非 function 工具、上游禁用工具等情况不注入。DeepSeek 思考模式默认
不注入，只有显式 `thinking.type: disabled` 才可启用。已冻结工具的会话改用
不兼容参数时，该请求使用客户端可见历史，不展开未声明的隐藏工具调用；这可能使
prompt cache 失效。恢复兼容参数后可继续使用
原冻结定义。辅助请求保留工具定义并设置
`tool_choice: none`。

文本和推理增量实时输出；网关工具调用被隐藏。客户端工具在名称和参数完整后
正常返回。混合调用先执行网关工具，下一次请求将完整助手调用和网关结果还原到
客户端结果之前。隐藏往返加密保存，包含 `reasoning_content` 和未知厂商字段；
按客户端可见回答匹配分支，编辑或重新生成的不同回答不会复用另一分支。
客户端收到一个完成 ID、一个结束事件和所有上游调用的累计用量；首次调用的
缓存统计不混入隐藏续轮。

默认限制：5 次隐藏往返、单工具 30 秒、结果 64 KiB、整个工具请求 120 秒、
隐藏续轮 100,000 token 预算。对应参数为 `tool_max_rounds`、`tool_timeout_seconds`、
`tool_result_bytes`、`tool_total_seconds`、`tool_total_tokens`。预算只计算网关新增工具调用、
结果与隐藏续轮输出；有输出 usage 时使用真实值，缺失时估算新增内容。客户端历史、工具定义、
图片不计入，所有轮次都保留客户端的 `max_tokens` / `max_completion_tokens`。
达到阈值后禁止继续调用网关工具，并允许最后一轮生成答案；最后输出可能超过阈值，
所以这是工具扩展预算，不是精确计费上限。不会因为工具结果耗尽预算而在写入完成后报错。
达到轮数上限后保留定义并设置 `tool_choice: none`，继续调用则明确报错。
工具失败作为有界结果交给模型；循环失败返回 HTTP 错误，已开始流式输出时发送
SSE 错误。取消或不完整回复不捕获。准备阶段存储故障降级为普通转发；Anthropic
无法恢复注入前缀时剥离历史 thinking，并记录降级原因。

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
`prompt_cache_key`；网关不施加每分钟 15 次的本地限流，上游返回的限流响应原样传递。会话内应保持模型、推理、采样、system 和工具参数
稳定；变化记录 `ark_cache_parameters_changed`。按模型设置 `cache_min_tokens`
（默认 1024），日志根据实际输入用量记录是否达到阈值；缓存命中仍需厂商实测。

标记为 `coding_plan` 的上游默认拒绝，管理员显式开启 `allow_coding_plan` 才可
覆盖。无法仅凭不透明 key 判断订阅类型，需管理员准确标记；Studio 提示共享网关
应使用模型 API key。旧 Context API 不作为增强协议接入。

二期测试包含工具流分片、即时文本、混合调用、重启和分支重放、附件上传、签名
代理、超时、写调用去重、轮数上限、方舟路由与跨进程限流。使用模拟厂商、真实 FastMCP 传输层和合成
对话。方舟真实缓存、Claude Code/Codex/SDK 客户端和真实工具导入的实测结果
见[验收记录](../../testing/context-gateway-live-acceptance.md)；跨真实模型最大窗口
与原生 Anthropic 签名绑定仍待验收。

重放存储只有五种不可变决策，接口仅为按锚点读取和不存在才写入。捕获队列独立，
所有分支、重试和归档进度在一份文档中通过单键 CAS 更新；响应用量、sent 标记、
召回额度预留和工具回执属于运行元数据。不存在跨重放记录与队列的多键事务。
稳定会话准备阶段一次批量读取，仅新决策需要写；匿名身份解析额外读取前缀归属索引。
用量、sent 和日志在响应发送后处理。SQLite 将读写执行器隔离，并合并同一事件循环
到达的读取；鉴权与上游配置使用最长 2 秒缓存，本进程相关管理变更立即失效。
多进程的撤销传播最多有 2 秒缓存延迟。策略读取不使用缓存，确保新会话冻结当前策略；
已有会话仍保留创建时的策略快照。Responses 映射写入和日志清理不会清空配置缓存。
架构边界、性能数据和验收限制见[审查修复记录](../../testing/context-gateway-review.md)。
