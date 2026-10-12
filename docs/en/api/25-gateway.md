# OpenViking Gateway Management API

The gateway owns `/api/v1/admin/gateway/*` in both embedded and standalone deployments. Embedded requests use the server port (normally 1933); standalone requests use the gateway port (normally 1935). No shared admin token or server HTTP proxy is involved. Model clients use `/v1/*`, `/api/v3/*` or `/api/compatible/v1/*` with a gateway key instead.

## Credentials and account scope

Send an OpenViking API key in `X-API-Key` or `Authorization: Bearer`. The gateway verifies identity through OpenViking `/health`, caching successful results by credential digest for 30 seconds. Local dev uses root identity `default/default`; trusted authentication is unsupported.

Upstreams and context profiles (`policies`) belong to an account. `account_id` is an optional query parameter, defaulting to the caller's account. Root can select any account. Admins and users remain in their own account for identity-dependent operations.

The optional management key goes in `X-OpenViking-Gateway-Management-Key`. It controls object changes independently of OpenViking roles. A wrong supplied key always returns 401 `management_key_invalid`; the request is not silently downgraded to read-only.

| Operation | No management key configured | Management key configured |
| --- | --- | --- |
| Read/test upstreams and read profiles | Any authenticated identity in its account | Same; management-key-only requests may choose any account |
| Create/update/delete upstreams or profiles | Any authenticated identity in its account | Valid management key required, including for admin/root |
| Import VLM configuration | Embedded root/dev only | Valid management key and embedded VLM source required |
| Issue/revoke/read gateway keys; reset capture | User: own keys; admin: account; root: selected account | Same; management key grants no extra rights |
| Logs and overview | User: own records; admin/root: account records | Same; OpenViking identity required |
| Delete user data | Admin in own account or root | Same |
| Delete account data | Root | Same |

A management key can be used **without** an OpenViking key for object operations and `/access`, selecting any account through `account_id`. It cannot grant access to memory, logs, overview, or gateway key issuance. Combining credentials does not elevate those identity-dependent operations. Changing the configured management key does not invalidate existing gateway keys.

## Access and discovery

All operations below use the `/api/v1/admin/gateway/` prefix.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/v1/admin/gateway/access` | Permissions: `managed`, `can_write_objects`, `can_import_vlm`, `can_issue_for_others`, `role`, `account_id`, `user_id`, `warnings` |
| GET | `/api/v1/admin/gateway/overview` | Summary of latest 10,000 visible logs: usage, cache, recall, capture issues, OpenViking health and log retention |
| GET | `/api/v1/admin/gateway/logs` | Newest visible logs; limit defaults to 200 and is clamped to 1–1000 |
| GET | `/api/v1/admin/gateway/guides` | `base_url` and `public_url_configured`; falls back to request origin |
| GET | `/api/v1/admin/gateway/tools` | Tool names, descriptions and annotations, fetched using the caller’s own OpenViking credential |

`warnings` may include `public_openviking_without_management_key` when a public OpenViking service is configured without a management key. This is a warning, not an access restriction.

```bash
curl "$GATEWAY_URL/api/v1/admin/gateway/access" \
  -H "X-API-Key: $OPENVIKING_API_KEY"
```

## Upstreams and profiles

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/v1/admin/gateway/upstreams` | List upstreams |
| GET | `/api/v1/admin/gateway/upstreams/{identifier}` | Read one upstream |
| PUT | `/api/v1/admin/gateway/upstreams/{identifier}` | Create or replace upstream |
| DELETE | `/api/v1/admin/gateway/upstreams/{identifier}` | Delete unused upstream |
| POST | `/api/v1/admin/gateway/upstreams/{identifier}/test` | Test the model-list endpoint |
| POST | `/api/v1/admin/gateway/upstreams/import-vlm` | Import embedded server model settings |
| GET | `/api/v1/admin/gateway/policies` | List profiles |
| GET | `/api/v1/admin/gateway/policies/{identifier}` | Read one profile |
| PUT | `/api/v1/admin/gateway/policies/{identifier}` | Create or replace profile |
| DELETE | `/api/v1/admin/gateway/policies/{identifier}` | Delete unused profile |

Choose IDs yourself. `PUT` takes a complete object; unknown fields fail validation. Field defaults and limits are in the [configuration reference](../guides/22-gateway-operations.md#configuration-reference).

API keys and header values are write-only. Writable upstream views return `has_api_key` and `header_names`; read-only views retain `has_api_key` and `auth_mode` but hide headers and their names. A blank/omitted `api_key` keeps the stored key **only if `base_url` is unchanged**. When changing the base, submit a new API key; otherwise the old key is cleared. When the base is unchanged, omitted `headers` preserve stored headers; supplied names replace the set, and blank values keep existing values for those names. Changing `base_url` also clears retained headers: re-enter every custom header value required at the new destination.

```bash
curl -X PUT "$GATEWAY_URL/api/v1/admin/gateway/upstreams/openai?account_id=team" \
  -H "X-OpenViking-Gateway-Management-Key: $GATEWAY_MANAGEMENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"OpenAI","protocol":"chat","base_url":"https://api.openai.com/v1","api_key":"<provider-key>"}'
```

Deleting an object still referenced by keys returns 409 `in_use` with `key_count`. Revoke those keys first, or disable an upstream to stop requests immediately. Existing gateway keys cannot be edited.

Connection tests return `{ok, status, reason, url}` (network failures omit `status`). 2xx with a `data` array passes (`passed`); 2xx failure bodies and 401/403 return `upstream_key_rejected`; 404/405 are reachable with no model list (`models_unavailable`, `ok: true`, key unverified); 429 is reachable (`rate_limited`, `ok: true`); network/5xx failures return `upstream_unavailable`. Tests time out after 10 seconds and remain available in read-only views.

VLM imports return `{imported, needs_confirmation, skipped}`. Entries use stable `vlm-<credential id>` IDs; repeat imports update them. Coding/Plan endpoints require `allow_coding_plan` confirmation. Skipped reasons include `openai_codex` (refreshing OAuth token), `azure` (requires api-version) and `litellm_no_api_base` (set api_base before retrying). Standalone gateways return `vlm_import_unavailable`.

## Gateway keys

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/v1/admin/gateway/keys` | List visible keys without secrets |
| GET | `/api/v1/admin/gateway/keys/{identifier}` | Read one visible key |
| POST | `/api/v1/admin/gateway/keys` | Issue key; returns secret once in `key` |
| DELETE | `/api/v1/admin/gateway/keys/{identifier}` | Revoke key, preserving user data |
| POST | `/api/v1/admin/gateway/keys/{identifier}/capture/reset` | Reset one session’s capture state |
| DELETE | `/api/v1/admin/gateway/users/{user_id}/data` | Delete user keys, conversation state and logs |
| DELETE | `/api/v1/admin/gateway/account/data` | Root: delete account gateway objects, keys, state and logs |

`POST keys` accepts required `name`, `policy_id`, and nonempty `upstream_ids`; optional `models` restricts model names (empty means all), `user_id` selects the target user, and `openviking_key` supplies a user's credential. For self-issuance, the caller's user/admin credential is used by default. Admin/root may select another user within their scope; the gateway fetches that user's key using the caller's credentials. If the server stores only hashes, 409 `user_key_unreadable` asks for an explicit `openviking_key`. A supplied `user_id` must match that key's identity. Root API keys cannot be bound to gateway keys; local dev is the exception with `default/default`.

```bash
curl -X POST "$GATEWAY_URL/api/v1/admin/gateway/keys?account_id=team" \
  -H "X-API-Key: $OPENVIKING_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"Alice laptop","user_id":"alice","policy_id":"default","upstream_ids":["openai"]}'
```

Copy the returned `key` (`ovgw_…`) immediately; subsequent reads show only its prefix and metadata. `PUT keys/{id}` returns 405 `replacement_key_required`: issue a replacement and revoke the original. Regenerating or revoking the bound OpenViking key invalidates the gateway key after identity-cache expiry (up to 30 seconds); this does not delete stored user data.

Capture reset takes `{"session":"<header value or logged session>","protocol":"chat"}`, where protocol is `chat`, `anthropic` or `responses`. It returns `{"status":"ready","message":"…"}` or 404 `not_found`.

Explicit data deletion removes pending conversation content and logs in the gateway. Sessions and memories already saved in OpenViking are unaffected. Embedded deployments also clean up gateway data through OpenViking's account/user deletion hooks; standalone data otherwise remains until explicit deletion or TTL expiry.

## Errors

Gateway failures contain a stable top-level `reason`, for example `{"reason":"in_use","detail":"in_use","key_count":2}`. Credential/service failures may also include `error: {message, reason}`. Match `reason`, not message text. An embedded server with Gateway disabled has no management route (404); Studio presents this as `not_enabled`.

| HTTP | reason | Next step |
| --- | --- | --- |
| 401 | `management_key_invalid` | Wrong management key; unlock with current value |
| 403 | `management_key_required` | Object writes require management key |
| 409 | `user_key_unreadable` | Paste target user’s OpenViking API key |
| 403 | `root_key_not_allowed` | Bind a user/admin API key instead |
| 403 | `other_account` | Choose an account inside caller scope |
| 401 | `openviking_http_401` | OpenViking rejected credential; reconnect |
| 502 / 503 | `openviking_unavailable / openviking_unreachable` | Check service connectivity and standalone --openviking-url |
| 503 | `openviking_version_mismatch` | Upgrade to min_server_version or later |
| 403 | `trusted_mode_unsupported` | Use API key authentication |
| 409 | `in_use` | key_count keys reference the object |
| 403 | `vlm_import_forbidden` | Recheck /access and import authority |
| 404 | `vlm_import_unavailable` | Add upstream manually on standalone host |
| 400 | `unknown_user / unknown_policy / unknown_upstream` | Choose existing user or object |
| 403 | `forbidden` | Operation outside caller role/user scope |
| 404 | `not_found` | Resource missing or not visible |
| 405 | `replacement_key_required` | Issue a replacement key |
| 422 | `invalid_configuration` | Fix body or base URL; submitted secrets are not echoed |
| 401 | `openviking_identity_missing` | Credential did not resolve to a user |

Other OpenViking HTTP failures use `openviking_http_<status>`; generic gateway HTTP errors use `gateway_http_<status>`. Model-client 401 errors can mean a missing, wrong or revoked `ovgw_` key, or an invalidated bound OpenViking key. Check the client key and reissue it with a current user key.

See [Deployment and operations](../guides/22-gateway-operations.md) for configuration, migration and URL rules.
