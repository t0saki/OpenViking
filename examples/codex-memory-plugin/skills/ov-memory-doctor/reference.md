# OpenViking Memory Doctor (Codex) — reference

Companion to SKILL.md: where things live, what the exact error strings mean,
and the symptom catalogue. Paths assume the defaults; `OPENVIKING_CONFIG_FILE`,
`OPENVIKING_CLI_CONFIG_FILE`, `OPENVIKING_DEBUG_LOG`, `OPENVIKING_CODEX_STATE_DIR`
and `CODEX_CONFIG_FILE` relocate individual pieces.

## Where things live

| Path | What |
|---|---|
| `~/.openviking/ovcli.conf` | Client connection: `url`, `api_key`, `account`, `user`, optional `plugin.codex.*` tuning. Mode 0600. |
| `~/.openviking/ov.conf` | Server config. The plugin reads only `server.url/host/port`, `server.root_api_key` (last-resort key) and the legacy `codex` block. |
| `~/.openviking/ovcli.conf.<name>` | Saved CLI profiles (`ov config switch` copies one over `ovcli.conf`). `ovcli.conf.bak.<epoch>` are installer backups. |
| `~/.codex/config.toml` | `[features] plugin_hooks`, `[plugins."openviking-memory@openviking"] enabled`, `[marketplaces.openviking]` (source, ref), `[hooks.state."openviking-memory@openviking:hooks/hooks.json:<event>:0:0"] trusted_hash` for `session_start`, `user_prompt_submit`, `stop`, `pre_compact`. |
| `~/.codex/plugins/cache/openviking/openviking-memory/<version>/` | The copy Codex runs hooks from. Keyed by `plugin.json` version. |
| `~/.codex/.tmp/marketplaces/openviking/` | Git clone of the marketplace (GitHub/TOS dist); `examples/codex-memory-plugin` inside it is what `codex plugin list` reports as the source path. |
| `~/.openviking/codex-plugin-state/<session_id>.json` | Per-session state: `ovSessionId` (`cx-<session_id>`, null once committed), `capturedTurnCount`, `lastUpdatedAt`. |
| `~/.openviking/codex-plugin-state/recall-compressor-profile.json` | Cached local-compressor detection (`profile.enabled`, `model`, `source`). |
| `~/.openviking/logs/codex-hooks.log` | JSONL hook + proxy log; written only when `OPENVIKING_DEBUG=1` or `codex.debug: true`. |
| `~/.openviking/codex-plugin.rc.sh`, `~/.openviking/codex-memory-plugin/runtime/` | Residue of the pre-marketplace installer. The rc script, if still sourced, exports `OPENVIKING_*` and pins credentials to env mode. |

## Config resolution

`OPENVIKING_CREDENTIAL_SOURCE` picks the mode: `env` (env vars only), `cli`
(ovcli.conf only — env and ov.conf ignored), default `auto` (env vars win when
any credential var is set, else ovcli.conf, else ov.conf/defaults). The
doctor prints the effective mode as `credential source`.

| Field | Order (auto mode) |
|---|---|
| url | `OPENVIKING_URL` → `OPENVIKING_BASE_URL` → `ovcli.conf url` → `ov.conf server.url` → `http://{server.host\|127.0.0.1}:{server.port\|1933}` |
| api_key | `OPENVIKING_BEARER_TOKEN` → `OPENVIKING_API_KEY` → `ovcli.conf api_key` → `ov.conf codex.apiKey` → `ov.conf server.root_api_key` |
| account / user | `OPENVIKING_ACCOUNT` / `OPENVIKING_USER` → `ovcli.conf account/account_id`, `user/user_id` → `ov.conf codex.accountId/userId` |
| peer | `OPENVIKING_PEER_ID` → `ovcli.conf actor_peer_id/peer_id` → `ov.conf codex.peerId` → derived from cwd unless `OPENVIKING_WORKSPACE_PEER=0` |
| auth mode | `OPENVIKING_AUTH_MODE` → `codex.authMode` → `server.auth_mode` → `trusted` when account/user are set, else `api_key` |
| tuning | env → `ovcli.conf plugin.codex.*` → `ovcli.conf plugin.*` → `ov.conf codex.*` → defaults |

There is no global kill switch: `OPENVIKING_MEMORY_ENABLED` is ignored by the
Codex plugin. Disable features with `OPENVIKING_AUTO_RECALL=0`,
`OPENVIKING_AUTO_CAPTURE=0`, `OPENVIKING_NO_AUTO_INJECT=1`, or the plugin via
`codex plugin remove openviking-memory@openviking` / `enabled = false`.

Hook budgets in `hooks/hooks.json`: SessionStart 70s, UserPromptSubmit 130s,
Stop 30s, PreCompact 60s. `recallTimeoutMs` (default 120000) must stay below
130s and `captureTimeoutMs` (default 30000) at or below 30s.

Sent headers: `Authorization: Bearer <key>`, `X-OpenViking-Account/User` (trusted
mode only), `X-OpenViking-Actor-Peer`, `User-Agent: openviking-memory-codex/<version>`.
The open-source server also accepts `X-API-Key` (and prefers it when both are
sent); the Volcengine-hosted OpenViking Service (`https://api.vikingdb.cn-beijing.volces.com/openviking`) accepts Bearer only.

## Server auth modes (from `GET /health` → `auth_mode`)

| Mode | Credential | Identity |
|---|---|---|
| `dev` | none needed, any key accepted | `X-OpenViking-Account/User` headers, else `default`; role root |
| `api_key` | key required | from the key; account/user headers are silently stripped |
| `trusted` | root key optional | from the headers; missing → 400 `Trusted mode requests must include X-OpenViking-Account …` |

Key formats: `base64url(account).base64url(user).base64url(secret)` (three
segments, identity readable) or a bare 64-hex legacy key. Both fail
identically as `401 Invalid API Key` when unknown — there is no "wrong
account" error.

`role: root` keys are refused on everything the plugin uses (`/api/v1/sessions`,
`/api/v1/search`, `/api/v1/fs`, `/mcp`) with
`403 ROOT API keys cannot access tenant-scoped data APIs in api_key mode`.

## Error strings

Server (REST): `{"status":"error","error":{"code":…,"message":…}}`

| HTTP | code | message | Meaning |
|---|---|---|---|
| 401 | UNAUTHENTICATED | `Missing API Key when resolving identity.` | No credential reached the server (gateway stripped `Authorization`, or `bearer` lowercase) |
| 401 | UNAUTHENTICATED | `Invalid API Key` | Unknown/revoked key, or a well-formed key for a non-existent account/user |
| 403 | PERMISSION_DENIED | `ROOT API keys cannot access tenant-scoped data APIs …` | Root key used as the plugin key |
| 403 | PERMISSION_DENIED | `Requires role: …` | Key valid, role too low for that route |
| 400 | INVALID_ARGUMENT | `Trusted mode requests must include X-OpenViking-Account …` | Trusted mode without account/user |
| 412 | FAILED_PRECONDITION | `User deletion is in progress` | The user is being deleted |

`/mcp` directly: `406 Not Acceptable: Client must accept both application/json and text/event-stream`;
`400` plain text `Invalid Content-Type header`; `401/403` as JSON-RPC `-32001`
with the server message; `404` → no MCP endpoint at that url.

Plugin MCP proxy (what Codex shows for a failing tool call):

| JSON-RPC | Message | Meaning |
|---|---|---|
| `-32001` | `OpenViking MCP authentication failed (HTTP 401\|403). Check ~/.openviking/ovcli.conf or OPENVIKING_API_KEY …` | Credentials; `data.serverMessage` carries the server text |
| `-32001` | `OpenViking MCP request failed. Check the configured URL (<mcpUrl>) …` | Transport: `data.cause` = `fetch failed` (refused/DNS/TLS), `This operation was aborted` (timeout). The url in the message is the one actually used. |
| `-32002` | `OpenViking MCP upstream returned HTTP <n>.` | Any other status; HTML in `data.serverMessage` means the url is not an OpenViking endpoint |
| `-32003` | `OpenViking MCP upstream returned an empty response` | 2xx with blank/non-JSON body — captive portal or proxy interstitial |

Hook log (`codex-hooks.log`) hooks: `session-start`, `auto-recall`,
`auto-capture`, `pre-compact`, `mcp-proxy`. Stages worth grepping:
`health_check`, `appended`, `pending_tokens`, `commit`, `recall_context_assembled`,
`mcp-proxy` `start` (resolved `mcpUrl`, `hasApiKey` = present, not valid),
`uncaught`. Ordinary failed proxy requests are not logged — the JSON-RPC error
on stdout is the artifact.

## Symptom catalogue

| Symptom | Likely cause | Detect | Fix |
|---|---|---|---|
| Plugin listed and enabled, but no `<openviking-context>` ever, no log | `plugin_hooks` off, hooks never trusted, or no parseable config | doctor Plugin install + Configuration sections | Enable hooks / approve the prompt / fix config |
| Worked until an edit to `ovcli.conf` | JSON broken by the edit | doctor "cannot be parsed" | Fix JSON |
| Edits to `ovcli.conf` or `ov config switch` have no effect | Env var or rc-file export wins (`credential source: env`) | doctor "← env"; `env \| grep OPENVIKING_` | Remove the export |
| Recall empty on a healthy server | Wrong key (401 reads as no results), wrong user space, peer scope, threshold | `/health` with key; `/api/v1/system/status` → `result.user` | Fix key; `OPENVIKING_PEER_ID`; lower `OPENVIKING_SCORE_THRESHOLD` |
| Recall empty after moving/renaming the repo | Workspace peer derived from cwd changed (with `recallPeerScope: actor`) | doctor `peer … ← workspace` | Pin `OPENVIKING_PEER_ID` or `OPENVIKING_WORKSPACE_PEER=0` |
| Hooks stopped after a plugin update | `hooks.json` hash changed; Codex re-prompts for trust | doctor "hooks without a trust record" | Accept the prompt in a new session |
| MCP tools missing | `.mcp.json` not loaded (plugin disabled), node missing, or `/mcp` not proxied | doctor Connection `/mcp` probe | Fix enablement / node / reverse proxy |
| Everything duplicated | Two plugin ids installed (legacy `openviking-plugins-local`) | doctor "more than one copy" | `codex plugin remove <stale id>` |
| `codex plugin marketplace upgrade` says up to date but bug persists | Version-keyed cache, version string unchanged | doctor cache versions vs marketplace copy | Re-run the installer (re-registers the marketplace) |
| Every prompt is slow | Local compressor (`codex exec`) on each recall | `recall-compressor-profile.json`; `OPENVIKING_RECALL_COMPRESS=0` to test | Disable compression or fix the model |
| curl works, plugin says offline | Corporate proxy or private CA; Node ignores both | doctor proxy/TLS hints; `node -e "fetch('<url>/health')"` | `NODE_USE_ENV_PROXY=1` / `NODE_EXTRA_CA_CERTS` in the launching environment |
| `0 memories extracted` after commits | Server-side embedding/VLM | `ov doctor` on the server host | Server admin |

## Links

- Source: <https://github.com/volcengine/OpenViking>
- Docs index: <https://docs.openviking.ai/llms.txt>
