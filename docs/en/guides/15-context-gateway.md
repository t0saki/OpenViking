# Context Gateway

OpenViking Context Gateway connects API-key model clients to OpenViking memory.
It is a **separate process**, not the VikingBot Bot Gateway. It supports Anthropic
Messages, OpenAI Chat Completions and full-history Responses (`store: false`).
It does not translate between model protocols, bill users or balance providers.

## Start the gateway

Install the OpenViking wheel from this branch. Generate two secrets and keep them
outside `ov.conf` and source control:

```bash
export OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
export OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

Add the following section to `ov.conf`:

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

Run `openviking-context-gateway --config /path/to/ov.conf`. Start OpenViking Server
with the same configuration and management-token environment variable. Model
requests go to port 1935; Studio remains on the OpenViking server. Optional
`openviking[context-gateway-fast]` installs uvloop and httptools where supported.

The gateway periodically checks OpenViking's `/health`. Root credentials are
rejected when issuing a downstream key. If OpenViking uses dev authentication,
the gateway must bind to loopback. Use API-key authentication for shared access.

## Configure access in Studio

Connect Studio with an account administrator credential and open **Context
Gateway**. Add an upstream (protocol, URL, model API key, model names and aliases),
create a context policy, then issue a downstream key using an OpenViking **user**
key. The verified account must match the administrator's account. Copy the new
key immediately; only its hash and display prefix are retained.

The six views provide overview metrics, upstream connectivity tests, context
policies, key issuance/revocation, metadata-only request logs and client setup
instructions. Upstream keys, OpenViking keys, injection records and queued messages
are encrypted with Fernet. Back up the encryption key with both databases. Losing
the key or kernel database loses the prefixes required for signed thinking.
Changing the encryption key requires a deliberate migration, not a restart with
a new key. Configuring an upstream URL grants it access to that key's model input.

Policies snapshot at the beginning of a session. Recall defaults to a 2-second
deadline, 1,600 tokens per turn and 6,000 per session. Capture can be enabled
without recall. Commit defaults to 20,000 pending tokens, with at least ten recent
messages retained at complete-turn boundaries. Last turns are captured and
committed after ten idle minutes. The model keeps three recent turns during
archive takeover. The token estimate used for local safety budgets is conservative
and is not provider billing usage.

Managed upstream credentials are the default. With `auth_mode: passthrough`,
clients send their model key in `X-OpenViking-Upstream-Key` **in addition** to their
gateway authentication. Claude subscription OAuth tokens are rejected.

## Client configuration

Claude Code:

```bash
export ANTHROPIC_BASE_URL=https://ov.example.com
export ANTHROPIC_AUTH_TOKEN='<gateway-key>'
export CLAUDE_CODE_GATEWAY_HINT_HEADERS=1
```

Codex (`config.toml`):

```toml
model_provider = "openviking"

[model_providers.openviking]
name = "OpenViking Context Gateway"
base_url = "https://ov.example.com/v1"
wire_api = "responses"
env_key = "OPENVIKING_GATEWAY_KEY"
```

Export `OPENVIKING_GATEWAY_KEY` in the shell. Only full-history HTTP requests with
`store: false` receive memory enhancement. WebSocket handshakes receive 426 so
Codex can fall back to HTTP. Stateful Responses requests are passed through;
subsequent response lookup/cancellation stays on the creating upstream and key.

For a generic chat client, use base URL `https://ov.example.com/v1`, the gateway
key, and optionally `X-OpenViking-Session: <conversation-id>`. Open WebUI can use
`X-OpenViking-Session: {{CHAT_ID}}` and `X-OpenViking-Task: {{TASK}}`; enable
`RAG_SYSTEM_CONTEXT`. One shared frontend connection key maps all its users to
one OpenViking user. Studio also provides OpenCode and pi configuration snippets.

Detected OpenViking plugins permanently disable new gateway recall/capture for
that session. Previous gateway injections still replay. Tool continuations,
subagents, auxiliary calls and token-count calls replay without new recall.
Explicit session headers improve capture attribution; without them, matching is
best effort and identical initial prefixes can share a session snapshot and injection. Truncating history
can start a new inferred session.

## Deploy

`docker compose --profile context-gateway up -d` starts the additional process.
Use an image built from this branch. For Compose set gateway `host` to `0.0.0.0`,
`url` to `http://context-gateway:1935`, `openviking_url` to
`http://openviking:1933` and `storage_path` to
`/app/.openviking/context-gateway`. The shared server must use authenticated mode.
Caddy forwards `/v1/*`, native Ark paths and the signed upload proxy to the gateway,
with the remaining paths going to OpenViking.
Copy these handlers into a public TLS domain block when enabling HTTPS.

Helm supports `contextGateway.enabled`, `workers`, `port`, `existingSecret` and
`resources`. The Secret must contain `encryption-key` and `admin-token`. The
gateway runs as a separate container in the same pod, sharing a local persistent
volume; ingress forwards `/v1` to its service port. SQLite WAL supports multiple
processes on **one machine**, not a shared network filesystem or multiple hosts.
The kernel storage protocol can be replaced for multi-host deployments.

Sessions expire as a unit after 30 unused days; logs have a separate 30-day
retention. `DELETE /api/v1/admin/context-gateway/users/{user_id}/data` revokes
that user's gateway keys and purges their kernel state. OpenViking's durable user/account deletion worker also invokes this cleanup; a gateway outage keeps the cleanup retryable. Kernel and management data are separate SQLite databases.

## Verification and operational limits

```bash
pytest tests/context_gateway --confcutdir=tests/context_gateway -o addopts=''
node scripts/sync-context-gateway-rules.mjs --check
node examples/memory-plugin-shared/sync.mjs --check
```

Tests use local mock HTTP upstreams and synthetic client sequences. They check
immutable replay, normalized signed prefixes, restart/concurrency behavior,
scope isolation, branch capture, numeric fidelity, SSE framing and error headers.
These are **not recordings of real clients or evidence of provider cache hits**.
Run the opt-in live acceptance script separately:

```bash
export OV_CG_TEST_BASE_URL=https://ov.example.com
export OV_CG_TEST_KEY='<gateway-key>'
export OV_CG_TEST_MODEL='<model-id>'
python scripts/context_gateway_acceptance.py --protocol anthropic
```

Run again with `chat` and `responses` on corresponding upstreams, and exercise
Claude Code, Codex and your selected chat client. Confirm cache-read usage extends
to the previous request, Anthropic reports no input transformations and no binding
400s, and a conversation can cross its context window. A single live smoke run
does not establish those full acceptance criteria. Re-run the recorded sequences
when clients change their serialization. Request metrics distinguish first calls
from within-turn calls; alert on falling first-call cache hits, missing replay and
degradation reasons.

If recall fails, its empty decision is persisted and never backfilled. Unsafe
numbers or duplicate JSON keys cause byte-for-byte passthrough. Missing known
Anthropic replay records strip historical thinking and record a degradation.
Archive summaries are published asynchronously: near the context limit the
gateway waits up to the configured deadline, then uses a persistent recent-tail
fallback and records `archive_wait_timeout`.

OpenViking message writes do not provide an idempotency key. A crash between a
successful HTTP write and durable acknowledgment can duplicate that batch; the
gateway preserves source-message IDs for diagnosis. Normal retries and copied
prefixes are deduplicated. User attribution and last-turn retention are inherently
less precise than a harness plugin because the gateway sees only API requests.

## Hidden tools (phase two)

Hidden tools are opt-in and supported only by Chat Completions. In Studio, enable
`gateway_tools` in the context policy. The default allowlist is `search`, `read`,
`list`, exposed as `openviking_search`, `openviking_read`, `openviking_list`. Calls
use the downstream user's OpenViking identity through the existing stateless MCP
endpoint. The gateway owns and versions its schemas; it does not publish dynamic
MCP descriptions. Existing MCP/plugin clients should continue using their tools.

The first request freezes the gateway tool definitions for the session. Detection
of an existing OpenViking plugin, `n > 1`, structured output, forced tool choice,
non-function tools or an upstream with `allow_gateway_tools: false` suppresses
injection. DeepSeek thinking mode suppresses injection unless the request
explicitly sets `thinking.type: disabled`. If a session with frozen gateway tools
later requests an incompatible mode, it receives HTTP 409 and must start a new
session. Auxiliary requests retain definitions with `tool_choice: none`.

Text and reasoning deltas continue streaming as they arrive. Gateway tool calls
are hidden; client tool calls are returned after their complete names and arguments
are known. If both occur in one round, the gateway executes only its own calls and
returns the client calls. On the next request, the original assistant message and
gateway results precede the client's results. Encrypted immutable records retain
the exact hidden messages, including reasoning and unknown provider fields. The
visible assistant content and calls identify the branch, so editing or regenerating
an answer does not reuse another answer's transcript. The gateway publishes one
completion ID and one terminal event. It sums provider usage across model calls;
first-call cache metrics exclude hidden continuation usage.

Default bounds are five hidden rounds, 30 seconds per tool, 64 KiB per tool result,
120 seconds for the complete model/tool request and a 100,000-token request budget.
These are configurable through `tool_max_rounds`, `tool_timeout_seconds`,
`tool_result_bytes`, `tool_total_seconds`, `tool_total_tokens`. The token budget
uses provider usage when available and a local byte-based estimate for admission
and continuation output limits; it is not an exact tokenizer or billing cap.
At the round limit, definitions stay present and `tool_choice` becomes `none`.
Further gateway calls fail explicitly. A tool error is a bounded result for the
model; an upstream/loop failure is an HTTP error, or a terminal SSE error if text
has already started. Incomplete streams are not captured. A Chat replay-storage
failure returns 503, because continuing could lose hidden history.

`write`, `add_resource` and `add_skill` require both `allow_write_tools: true` and
an explicit entry in `tool_allowlist`. Retries of the same call within the same
session share a durable claim and result across workers. A crashed or timed-out
write is not automatically re-executed; its outcome must be inspected before a
new attempt. Different model-generated call IDs are different operations, so this
is not an end-to-end exactly-once guarantee.

### File import

The first request must contain a shell tool or an attachment for file import tools
to be selected. The gateway never opens a client path on its own filesystem.

- With a client shell, MCP returns a signed upload URL. The gateway replaces its
  address with `<context_gateway.public_url>/context-gateway/uploads?token=…`.
  The model can then request the client's ordinary shell tool to upload the file;
  skill directories must be zipped first. Set `public_url` to a client-reachable
  gateway address before enabling these tools.
- Without a shell, use `attachment_index` to import a file embedded in a user
  message as `file.file_data` / `input_file.file_data` (base64 or a base64 data URL),
  or an explicit attachment `text` field. Open WebUI `<context><source …>` extracted
  text is imported as a `.txt` file, without pretending to recover the original
  binary document. Source tags are also recognized in its system context.
- File IDs and remote attachment URLs without embedded bytes are not fetched by
  the gateway. A local path without shell access requires an attachment instead.

The upload proxy forwards only to OpenViking's fixed `/api/v1/resources/temp_upload`
endpoint. The one-time signed token authorizes upload; user/model API keys are
never passed to the client or this endpoint. OpenViking validates expiration and
consumes the token. The configured `max_body_bytes` also bounds uploads. Caddy and
Helm route `/context-gateway/uploads` to the gateway; custom proxies must do so too.
The gateway CLI disables raw access logs; also omit query strings in external
proxy logs on this route to avoid retaining signed tokens.

## Volcano Engine Ark (phase two)

Configure `vendor: ark` and a model API key. Use the Ark origin as `base_url`, or
`…/api/v3` for Chat/Responses, or `…/api/compatible/v1` for Messages. Configure
separate upstream records for each protocol. Client-facing paths are:

| Gateway path | Handler |
| --- | --- |
| `/api/v3/chat/completions` | Chat Completions |
| `/api/v3/responses` and response-ID subpaths | Responses |
| `/api/compatible/v1/messages` | Anthropic Messages |

The existing `/v1/…` paths work with Ark upstreams too. Caddy and Helm include the
native paths. Unknown fields such as `thinking`, `encrypted_content`, `caching`
and `expire_at` survive forwarding, and Responses SSE `[DONE]` is preserved.
Enhanced Chat/Responses sessions receive a stable `prompt_cache_key`; cross-worker
reservations limit that key to 15 requests in a sliding minute, counting hidden
continuations, with HTTP 429 and `Retry-After` on exhaustion. Keep model, thinking,
sampling, system and tool settings stable; changes are logged as
`ark_cache_parameters_changed`. Set `cache_min_tokens` for the selected model
(default 1024); logs report the threshold and eligibility from actual input usage.
Cache hits still depend on the provider and model, not just gateway replay.

Coding Plan credentials are rejected when an upstream is marked `coding_plan`,
unless an administrator explicitly overrides `allow_coding_plan`. The gateway
cannot identify a subscription solely from an opaque API key; administrators must
classify it correctly. Studio warns that a shared API gateway should use model API
keys. Legacy Context API is not supported as an enhanced protocol.

Phase-two tests cover fragmented tool streams, immediate text delivery, mixed
client/gateway calls, restart and branch replay, file import, signed upload proxying,
timeouts, write claims, round bounds, Ark routing and shared rate limits. They
use simulated providers, the real FastMCP transport, and synthetic conversations. Live Ark/model cache hits,
real-client recordings and conversations crossing a real model's context window
remain separate operator acceptance checks.
