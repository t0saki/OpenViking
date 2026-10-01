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
一个共享连接 key 下的所有人都归到同一个 OpenViking 用户。Studio 也提供 pi
和 OpenCode 的配置。

发现 OpenViking 插件标记后，该会话永久停止新增召回与捕获，原有注入继续重放。
工具续轮、子 agent、辅助请求和 token 计数只重放。WebSocket 返回 426；带
`previous_response_id`、`conversation`、`background` 的 Responses 原样透传。

## 部署与验收

Compose 使用 `docker compose --profile context-gateway up -d`；容器内需将网关
监听地址设为 `0.0.0.0`，管理地址设为 `http://context-gateway:1935`，OpenViking
地址设为 `http://openviking:1933`，存储目录设为 `/app/.openviking/context-gateway`。
Caddy 将 `/v1/*` 转发给网关。Helm 开启 `contextGateway.enabled` 并指定包含
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
