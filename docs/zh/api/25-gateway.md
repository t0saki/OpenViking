# OpenViking 网关管理 API

嵌入和独立部署都由网关直接提供 `/api/v1/admin/gateway/*`。嵌入时访问 Server 端口（通常 1933），独立时访问 Gateway 端口（通常 1935），不再经过共享 admin token 或 Server HTTP 代理。模型客户端则用网关密钥访问 `/v1/*`、`/api/v3/*` 或 `/api/compatible/v1/*`。

## 凭证和账号范围

在 `X-API-Key` 或 `Authorization: Bearer` 中发送 OpenViking API 密钥。网关向 OpenViking 确认密钥属于谁，确认成功的结果保留 30 秒。本地 dev 使用 root 身份 `default/default`；不支持 trusted 认证。

上游和上下文配置（`policies`）归账号所有。可选 query 参数 `account_id` 默认是调用方账号。root 可选任意账号；涉及 OpenViking 身份的操作中，admin 和 user 始终限于本账号。

可选管理密钥通过 `X-OpenViking-Gateway-Management-Key` 传入，与 OpenViking 角色分别控制对象修改和身份权限。传入错误管理密钥一律返回 401 `management_key_invalid`，不会悄悄降级为只读。

| 操作 | 未配置管理密钥 | 已配置管理密钥 |
| --- | --- | --- |
| 读取/测试上游，读取上下文配置 | 已认证身份可访问本账号 | 同左；单独使用管理密钥可选择任意账号 |
| 创建/修改/删除上游或上下文配置 | 已认证身份可修改本账号 | 必须有有效管理密钥，admin/root 也一样 |
| 导入 VLM 配置 | 仅嵌入 root/dev | 需要有效管理密钥和宿主 VLM 来源 |
| 签发/吊销/读取网关密钥、重置保存 | user 仅本人；admin 本账号；root 所选账号 | 同左；管理密钥不增加权限 |
| 日志和概览 | user 仅本人；admin/root 整个账号 | 同左，始终需要 OpenViking 身份 |
| 删除用户数据 | 本账号 admin 或 root | 同左 |
| 删除账号数据 | root | 同左 |

管理密钥可以**不带** OpenViking 密钥单独用于对象操作和 `/access`，通过 `account_id` 指定任意账号。它不授予记忆、日志、概览或网关密钥签发权限；两种凭证同时使用也不会提升这些操作的身份权限。修改已配置的管理密钥不会使现有网关密钥失效。

## 权限与发现

以下操作均使用 `/api/v1/admin/gateway/` 前缀。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/admin/gateway/access` | 权限：`managed`、`can_write_objects`、`can_import_vlm`、`can_issue_for_others`、`role`、`account_id`、`user_id`、`warnings` |
| GET | `/api/v1/admin/gateway/overview` | 最近 10,000 条可见日志的用量、缓存、召回、保存异常、OpenViking 状态和日志保留期 |
| GET | `/api/v1/admin/gateway/logs` | 最新可见日志；limit 默认 200，限制到 1–1000 |
| GET | `/api/v1/admin/gateway/guides` | 返回 `base_url` 和 `public_url_configured`；未设置对外地址时使用请求来源 |
| GET | `/api/v1/admin/gateway/tools` | 工具名称、描述和 annotations；使用调用方自己的 OpenViking 凭证获取 |

连接公共 OpenViking 服务且没有配置管理密钥时，`warnings` 可能包含 `public_openviking_without_management_key`。它是告警，不限制操作。

```bash
curl "$GATEWAY_URL/api/v1/admin/gateway/access" \
  -H "X-API-Key: $OPENVIKING_API_KEY"
```

```json
{
  "managed": true,
  "can_write_objects": false,
  "can_import_vlm": false,
  "can_issue_for_others": true,
  "role": "admin",
  "account_id": "team",
  "user_id": "alice",
  "warnings": []
}
```

OpenViking API 密钥无效时返回 401 `openviking_http_401`。

## 上游和上下文配置

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/admin/gateway/upstreams` | 列出上游 |
| GET | `/api/v1/admin/gateway/upstreams/{identifier}` | 读取单个上游 |
| PUT | `/api/v1/admin/gateway/upstreams/{identifier}` | 创建或替换上游 |
| DELETE | `/api/v1/admin/gateway/upstreams/{identifier}` | 删除未使用的上游 |
| POST | `/api/v1/admin/gateway/upstreams/{identifier}/test` | 测试模型列表接口 |
| POST | `/api/v1/admin/gateway/upstreams/import-vlm` | 导入嵌入宿主模型配置 |
| GET | `/api/v1/admin/gateway/policies` | 列出上下文配置 |
| GET | `/api/v1/admin/gateway/policies/{identifier}` | 读取单个上下文配置 |
| PUT | `/api/v1/admin/gateway/policies/{identifier}` | 创建或替换上下文配置 |
| DELETE | `/api/v1/admin/gateway/policies/{identifier}` | 删除未使用的上下文配置 |

ID 由调用方指定。`PUT` 提交完整对象；未知字段会校验失败。字段默认值和限制见[配置参考](../guides/22-gateway-operations.md#配置参考)。

API Key 和请求头的值只写不读。可编辑视图返回 `has_api_key` 和 `header_names`；只读视图保留 `has_api_key` 和 `auth_mode`，不返回请求头及其名称。`base_url` 不变时，空白或省略的 `api_key` 保留已存的密钥，省略 `headers` 保留已存的请求头，提交的 `headers` 对象替换整个集合，其中空白值沿用该名称已存的值。**修改 `base_url` 会清除已存的 API Key 和请求头值**，新地址需要的凭证都要重新提交。

```bash
curl -X PUT "$GATEWAY_URL/api/v1/admin/gateway/upstreams/openai?account_id=team" \
  -H "X-OpenViking-Gateway-Management-Key: $GATEWAY_MANAGEMENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"OpenAI","protocol":"chat","base_url":"https://api.openai.com/v1","api_key":"<provider-key>"}'
```

删除仍被密钥引用的对象返回 409 `in_use`，附带 `key_count`。请先吊销引用它的密钥；也可停用上游立即停止请求。

### 连接测试

`POST upstreams/{identifier}/test` 用已存的凭证请求服务商的模型列表，返回 `{ok, status, reason, url}`；没有收到响应时不含 `status`。超时为 10 秒，只读视图中也可使用。Anthropic 上游把密钥同时放在 `x-api-key` 和 `Authorization: Bearer` 中发送；Chat Completions 和 Responses 上游只发送 Bearer。

| 服务商的响应 | `reason` | `ok` |
| --- | --- | --- |
| 没有可用的 API Key：没有保存，或者上游要求每个客户端自带。不发送请求。 | `upstream_key_missing` | `false` |
| 2xx 且带 `data` 数组 | `passed` | `true` |
| 2xx 但响应体表示失败、401、403 | `upstream_key_rejected` | `false` |
| 404、405：地址可达，但没有模型列表，密钥未验证 | `models_unavailable` | `true` |
| 429：地址可达 | `rate_limited` | `true` |
| 网络错误或 5xx | `upstream_unavailable` | `false` |

### 导入 VLM 配置

`POST upstreams/import-vlm` 把嵌入宿主的 VLM 凭证导入为所选账号的上游，每个上游的 ID 固定为 `vlm-<credential id>`。再次导入时，来自服务端的字段（`base_url`、`api_key`、`headers`、`protocol`、`vendor`、`coding_plan`）会刷新，其他字段保留在网关中修改后的值，包括 `name`、`enabled`、`priority`、`models`、`aliases`、`context_windows` 和 `allow_coding_plan`。Coding/Plan 路径上的上游在设置 `allow_coding_plan` 之前归入 `needs_confirmation`。独立网关返回 404 `vlm_import_unavailable`。

```json
{
  "imported": [{"id": "vlm-default", "name": "default"}],
  "needs_confirmation": [{"id": "vlm-coding", "name": "coding"}],
  "skipped": [{"id": "vlm-codex", "reason": "openai_codex"}]
}
```

| 跳过原因 `reason` | 原因 |
| --- | --- |
| `openai_codex` | 这是会自动刷新的 Codex OAuth 令牌。 |
| `azure` | Azure 需要 `api-version`，网关不发送这个参数。 |
| `litellm_no_api_base` | LiteLLM 凭证没有 `api_base`；在 OpenViking 中补上后重新导入。 |
| `litellm_unsupported_route` | LiteLLM 路由不使用 OpenAI Chat Completions API，例如 `gemini/`、`bedrock/` 或 `vertex_ai/`。请手动把这个服务商添加为上游。 |
| `invalid_configuration` | 凭证无法构成有效的上游，例如 base URL 不可用。 |

## 网关密钥

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/admin/gateway/keys` | 列出可见密钥，不含秘密值 |
| GET | `/api/v1/admin/gateway/keys/{identifier}` | 读取单个可见密钥 |
| POST | `/api/v1/admin/gateway/keys` | 签发密钥；仅本次响应的 `key` 返回秘密值 |
| DELETE | `/api/v1/admin/gateway/keys/{identifier}` | 吊销密钥，保留用户数据 |
| POST | `/api/v1/admin/gateway/keys/{identifier}/capture/reset` | 重置单个会话保存状态 |
| DELETE | `/api/v1/admin/gateway/users/{user_id}/data` | 删除用户密钥、对话状态和日志 |
| DELETE | `/api/v1/admin/gateway/account/data` | root：删除账号网关对象、密钥、状态和日志 |

`POST keys` 必填 `name`、`policy_id` 和非空 `upstream_ids`；可选 `models` 限制模型名（空表示全部），`user_id` 指定目标用户，`openviking_key` 显式提供用户凭证。给自己签发时默认使用调用方的 user/admin 凭证。admin/root 可选择权限范围内的其他用户，网关用调用方凭证读取该用户的密钥。如果服务端只存哈希，返回 409 `user_key_unreadable`，需要显式提供 `openviking_key`。提供的 `user_id` 必须和密钥身份一致。root API 密钥不能绑定到网关密钥；本地 dev 的 `default/default` 例外。

```bash
curl -X POST "$GATEWAY_URL/api/v1/admin/gateway/keys?account_id=team" \
  -H "X-API-Key: $OPENVIKING_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"Alice laptop","user_id":"alice","policy_id":"default","upstream_ids":["openai"]}'
```

```json
{
  "id": "3f9c…",
  "name": "Alice laptop",
  "user_id": "alice",
  "policy_id": "default",
  "upstream_ids": ["openai"],
  "models": [],
  "prefix": "ovgw_Xk2pQ7a",
  "created_at": 1760000000.0,
  "revision": 1,
  "key": "ovgw_Xk2pQ7a…"
}
```

立即复制响应中的 `key`。`GET keys` 和 `GET keys/{identifier}` 返回同样的字段，但不含 `key`。网关密钥不能修改，没有 `PUT`；需要变更时签发新密钥，再吊销旧密钥。

绑定的 OpenViking API 密钥被重新生成或用户被删除后，OpenViking 返回 401，网关在约 30 秒内删除这个网关密钥，已存的用户数据保留。OpenViking 无法访问或返回其他错误时，请求照常进行，密钥也保留。

重置保存的请求体为 `{"session":"<请求头值或日志中的 session>","protocol":"chat"}`，协议为 `chat`、`anthropic` 或 `responses`。返回 `{"status":"ready","message":"…"}`，不存在时返回 404 `not_found`。

显式删除数据会清除网关内尚未保存的对话内容和日志，不影响已存入 OpenViking 的会话和记忆。嵌入部署还会通过 OpenViking 删除用户/账号的钩子清理网关数据；独立部署的数据保留到显式删除或 TTL 到期。

## 错误

管理错误包含稳定的顶层 `reason` 和一句简短的英文说明 `detail`，部分错误另有附加字段，例如 `{"reason":"in_use","detail":"Gateway keys still use this object; revoke them first.","key_count":2}`。请按 `reason` 判断，不要匹配 `detail` 或消息文本。嵌入服务未开启网关时没有管理路由（404），Studio 将其显示为 `not_enabled`。

| HTTP | reason | 下一步 |
| --- | --- | --- |
| 401 | `management_key_invalid` | 管理密钥错误；使用当前密钥解锁 |
| 403 | `management_key_required` | 修改对象需要管理密钥 |
| 409 | `user_key_unreadable` | 粘贴目标用户 OpenViking API 密钥 |
| 403 | `root_key_not_allowed` | 改用 user/admin API 密钥绑定 |
| 403 | `other_account` | 选择调用方权限范围内账号 |
| 401 | `openviking_http_401` | OpenViking 拒绝凭证；重新连接 |
| 502 / 503 | `openviking_unavailable / openviking_unreachable` | 检查服务连通性和独立 --openviking-url |
| 503 | `openviking_version_mismatch` | 仅独立网关：把 OpenViking 升级到 min_server_version 或更高 |
| 403 | `trusted_mode_unsupported` | 改用 API Key 认证 |
| 409 | `in_use` | key_count 个密钥仍引用对象 |
| 403 | `vlm_import_forbidden` | 重新检查 /access 和导入权限 |
| 404 | `vlm_import_unavailable` | 独立宿主请手动添加上游 |
| 400 | `unknown_user / unknown_policy / unknown_upstream` | 选择已有用户或对象 |
| 403 | `forbidden` | 操作超出角色或用户范围 |
| 404 | `not_found` | 资源不存在或不可见 |
| 422 | `invalid_configuration` | 修正请求体或 Base URL；不回显提交的凭证 |
| 401 | `openviking_identity_missing` | 凭证未解析出用户身份 |

其他 OpenViking HTTP 错误使用 `openviking_http_<status>`，通用网关 HTTP 错误使用 `gateway_http_<status>`，例如对不支持的方法返回 405 `gateway_http_405`。模型客户端的 401 可能来自缺失、错误或已吊销的 `ovgw_` 密钥，或其绑定的 OpenViking 密钥失效。请检查客户端密钥，并用当前用户密钥重新签发。

配置、迁移和 URL 规则见[部署与运维](../guides/22-gateway-operations.md)。
