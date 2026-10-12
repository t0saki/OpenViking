---
description: Deploy OpenViking Gateway with Docker Compose, Helm or your own proxy, manage it in Studio, and troubleshoot it.
---

# OpenViking Gateway deployment and operations

This page is for the people who run OpenViking Gateway: deploying it, managing upstreams, context profiles and keys in Studio, keeping data safe, day-to-day operations, and fixing problems. For what the gateway does and how to connect clients, see [OpenViking Gateway](15-gateway.md); for the overall architecture, see [Architecture at a glance](15-gateway.md#architecture-at-a-glance) on that page.

OpenViking Gateway is currently in beta. Its settings and APIs may change between releases, so read the release notes before you upgrade.

OpenViking Gateway can run inside OpenViking Server on the same port (1933 by default), or as `openviking-gateway` connected to an existing OpenViking service. Both hosts provide the same model and management URLs. Embedded deployments need no second process or path-based reverse proxy.

## Requirements

- OpenViking Server 0.4.16 or later, with `api_key` authentication or local `dev` authentication. Trusted mode is unsupported.
- A current OpenViking installation. Gateway dependencies are included in the core package; `gateway-fast` remains an optional performance extra.
- An OpenViking API key for remote use. Users, account admins and root can sign in to Studio. In local dev mode, the gateway uses the `default/default` identity without a key.
- Model provider credentials and local disk storage. Use a single host for the SQLite databases; network filesystems and storage shared between hosts are unsupported.

## Secrets

| Credential | Purpose | Where it belongs |
| --- | --- | --- |
| OpenViking API key | Sign in to Studio; identify the account and user for memory access | `X-API-Key` or `Authorization: Bearer` on management requests |
| Gateway key (`ovgw_…`) | Connect model clients; binds a user, profile and upstreams | Client API key field |
| Gateway management key | Allow changes to upstreams and profiles; does not authorize memory access or key issuance | `gateway.management_key`, and request header `X-OpenViking-Gateway-Management-Key` |
| Upstream API Key | Authenticate with a model provider | Upstream editor |

The management key is optional and must contain at least 32 characters when set. `openviking-server init` generates one with an `ovgm_` prefix when you choose to restrict configuration. You can generate a value with `openssl rand -base64 32`. Embedded configuration supports `"management_key": "$OPENVIKING_GATEWAY_MANAGEMENT_KEY"`; standalone deployments also read `OPENVIKING_GATEWAY_MANAGEMENT_KEY` directly. There is no command-line flag for this secret.

The separate **encryption key** protects stored secrets and conversation state. `OPENVIKING_GATEWAY_ENCRYPTION_KEY` takes precedence; otherwise the gateway atomically creates `storage_path/encryption.key` with mode 0600. Keep this file with the databases when backing up. If databases exist but the key file and environment key are absent, startup fails. Restore the original key: changing it does not re-encrypt existing data.

Changing, adding or removing the management key requires a restart but does not revoke gateway keys or change stored objects. Studio stores an entered management key only in the current tab's session storage; **Lock**, closing the tab, or an invalid-key response locks editing again.

## Deploy

### Deployment options

| Form | Setup | Client origin |
| --- | --- | --- |
| Embedded, including Compose and Helm | Enable `gateway.enabled` in the server configuration | Server origin, normally `http://127.0.0.1:1933` |
| Standalone | Run `openviking-gateway --openviking-url <OV URL>` | Gateway origin, normally `http://127.0.0.1:1935` |

### Embedded setup and init

Run `openviking-server init`, enable Gateway on the same port, then start or restart the server. For an existing configuration choose **Update gateway**. Remote setup asks whether users may configure upstreams themselves; answering no generates a management key. You can also merge this into `ov.conf` and restart:

```json
{
  "gateway": {
    "enabled": true
  }
}
```

Open `/studio/` on the server and select Gateway. In dev mode the first startup imports available VLM credentials when the default account has no upstreams and creates the `default` profile. Coding Plan imports require confirmation before use. If no credential can be imported, add an upstream manually.

**Agent configuration flow.** Use the server's root API key to call `GET /api/v1/admin/configuration?source=file`, read `result.content` and `result.revision`, edit the complete configuration text, then `PATCH /api/v1/admin/configuration?source=file` with `{"content":"<edited full text>","revision":"<read revision>"}`. Read the new revision from the save response and send `POST /api/v1/admin/restart` with `{"revision":"<new revision>"}`. An outdated revision is rejected; reload before trying again. Remote restart requires the single-worker `openviking-server` CLI. For multiple workers, restart through your process manager. These server configuration endpoints are not available on a standalone gateway.

### Routes

The embedded server passes these prefixes unchanged to the gateway:

| Prefix | Purpose |
| --- | --- |
| `/v1/` | Model APIs and model list |
| `/api/v3/` | Ark-compatible client paths |
| `/api/compatible/v1/` | Ark-compatible Anthropic client paths |
| `/gateway/` | Tool uploads |
| `/api/v1/admin/gateway/` | Authenticated management REST API |

`/health` remains the server's health endpoint; the embedded gateway does not expose a separate `/health` or legacy `/admin`. Standalone `/health` reports `service: "openviking-gateway"`, `auth_mode` and `openviking_url`, plus caller identity when credentials are supplied. Its root URL redirects to `/studio/`.

Set `public_url` to the external origin for shared deployments. With no `public_url`, client guides use the request origin. Anthropic clients use the origin; OpenAI-style clients add `/v1`.

### Docker Compose

Enable `gateway.enabled` in the mounted `ov.conf`, then run `docker compose up -d`. The gateway runs inside the `openviking` service; no `gateway` profile or separate service is needed. After editing the configuration, run `docker compose restart openviking`. Read startup errors with `docker compose logs openviking`.

Caddy forwards all paths to `openviking:1933`. The compatibility port 1934 can remain, but clients can also reach model APIs directly on server port 1933. For HTTPS follow [Public access](12-public-access.md); a single `reverse_proxy openviking:1933` covers Studio, management and model traffic. Set `gateway.public_url` to the public origin.

### Helm

Set `gateway.enabled: true` in chart values. The chart embeds the gateway in the server container and stores its data at `<persistence.mountPath>/gateway`; Service and Ingress use the server port. Keep `replicaCount: 1`.

```yaml
gateway:
  enabled: true
  encryptionKey:
    existingSecret: "" # optional Secret entry: encryption-key
  managementKey:
    existingSecret: "" # optional Secret entry: management-key
config:
  gateway:
    public_url: https://ov.example.com
```

With no encryption Secret the persisted key file is generated automatically. When a management Secret is supplied, the chart sets `management_key` to `$OPENVIKING_GATEWAY_MANAGEMENT_KEY`. See the [chart README](https://github.com/volcengine/OpenViking/blob/main/deploy/helm/README.md) for Secret creation and authentication setup.

### Separate deployment

Standalone mode does not read `ov.conf` and requires an OpenViking URL. It works with an existing open-source or commercial server without changing that server's configuration:

```bash
openviking-gateway --openviking-url http://127.0.0.1:1933 --port 1935
# Or read a standalone JSON configuration:
openviking-gateway --config gateway.conf
```

```json
{
  "openviking_url": "https://ov.example.com",
  "host": "127.0.0.1",
  "port": 1935,
  "storage_path": "~/.openviking/gateway"
}
```

Open the gateway's `/studio/` and sign in with an OpenViking user, admin or root API key. The connection page shows the configured OpenViking URL; standalone Studio offers Gateway and Connection pages. It cannot import the remote server's VLM configuration.

Flags are `--config`, `--openviking-url`, `--host`, `--port`, `--workers` and `--storage-path`. Defaults are host `127.0.0.1`, port `1935`, workers `1`. Configuration precedence is flags → `OPENVIKING_GATEWAY_<FIELD>` environment variables → JSON file. `OPENVIKING_GATEWAY_CONFIG_FILE` selects a file without `--config`. The file has gateway fields at its root, without a `gateway` wrapper or `enabled` field.

The gateway probes OpenViking before standalone startup and refuses a non-loopback bind with dev authentication. A commercial `/health` returning 401 means reachable but authentication required; it does not disable recall. Version checks use authenticated requests during key issuance and capture recovery. Protect remote OpenViking connections with TLS or a private network.

When connecting to `https://api.vikingdb.cn-beijing.volces.com/openviking` without a management key, startup logs and Studio warn `public_openviking_without_management_key`: every valid user of that public service can configure upstreams in their own account. Set `OPENVIKING_GATEWAY_MANAGEMENT_KEY` and restart to restrict object changes. This does not restrict who can sign in or issue their own gateway keys.

### Your own reverse proxy

Forward the whole origin to the server (embedded) or gateway (standalone). For example:

```caddyfile
ov.example.com {
    reverse_proxy 127.0.0.1:1933 {
        flush_interval -1
    }
}
```

Allow streaming without response buffering, request bodies of at least 32 MiB, and read timeouts of at least 600 seconds. Do not log query strings for `/gateway/uploads`, which carry one-time upload tokens. Configure public authentication before exposing a dev server; see [Public access](12-public-access.md).

### Scaling

Use one host and one deployment per storage directory. Embedded workers follow the server worker setting; standalone `--workers` accepts 1–64. Do not run embedded and standalone gateways against the same directory. For provider load balancing, failover or rate limiting, add a compatible service such as LiteLLM as an upstream.

## Manage the gateway in Studio

Connect with your OpenViking API key and open Gateway. Users can issue keys for themselves; admins can issue for users in their account, and root can switch accounts first. Upstreams and profiles are shared by the whole account.

Without a management key configured, every authenticated user may edit their account's objects. With one configured, upstreams and profiles are read-only until **Enter gateway management key** unlocks them; this also applies to OpenViking admins and root. Testing connections and issuing personal gateway keys remain available. Read-only upstream views hide API keys, headers and header names. Object edits take effect immediately for keys that use them.

Studio uses `GET /api/v1/admin/gateway/access` for permissions. See the [management API](../api/25-gateway.md) for the complete access matrix and all operations.

### Upstreams

An upstream is one model provider endpoint that speaks one API: Anthropic Messages, Chat Completions or Responses. The gateway never converts between them, so add one upstream for each API your clients use, even when they share a provider: Anthropic Messages for Claude Code, Responses for Codex CLI, Chat Completions for most chat apps and SDKs.

Each provider offers these protocols, and the editor fills in the provider's default Base URL when you choose it:

| Provider | Protocols | Default Base URL |
| --- | --- | --- |
| Generic | All three | None; enter the address yourself |
| Anthropic | Anthropic Messages | `https://api.anthropic.com` |
| OpenAI | Chat Completions, Responses | `https://api.openai.com/v1` |
| DeepSeek | All three | `https://api.deepseek.com`; `https://api.deepseek.com/anthropic` for Anthropic Messages |
| Volcano Engine Ark | All three | Chat / Responses: `https://ark.cn-beijing.volces.com/api/v3`; Anthropic: `https://ark.cn-beijing.volces.com/api/compatible` |
| BytePlus ModelArk | All three | Chat / Responses: `https://ark.ap-southeast.bytepluses.com/api/v3`; Anthropic: `https://ark.ap-southeast.bytepluses.com/api/compatible` |

The upstream editor has four sections.

**Endpoint.**

- **Provider** decides which protocols you can pick and adjusts the gateway to the provider's quirks. The editor offers only the protocols the provider supports, as listed in the table above. Choose *Generic* for a provider not listed here, a compatible proxy such as LiteLLM or new-api, or a reverse proxy such as [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) that exposes subscription accounts as an API (see [Custom upstreams](15-gateway.md#custom-upstreams); make sure using subscription quota this way complies with your provider's terms).
  - *Generic*, *Anthropic* and *OpenAI* forward requests the same way.
  - *DeepSeek*: with tools present, DeepSeek requires every earlier reply in the history to carry its reasoning, which many chat apps do not send back. DeepSeek upstreams have **Restore reasoning the client drops** on by default (see Routing below), so requests keep thinking on and still get OpenViking tools. With that setting off, OpenViking tools are offered only when a request turns thinking off (`"thinking": {"type": "disabled"}`); standard Responses requests have no `thinking` field, so they then get no tools. Memory is added as usual.
  - *Volcano Engine Ark* and *BytePlus ModelArk*, Ark's international edition, work the same way: each conversation gets a stable `prompt_cache_key` for Chat Completions and Responses, so Ark's prefix cache follows the conversation. Requests whose model, thinking, sampling, system prompt or tools differ from the conversation's first request are flagged **Cache parameters changed** (`ark_cache_parameters_changed`), because Ark resets its cache when these change.
- **Protocol** selects the API the upstream accepts. Anthropic sends both `x-api-key` and `Authorization: Bearer`; chat and responses send Bearer authorization.
- **Base URL.** Follow the provider's SDK documentation: chat and responses append `/chat/completions` or `/responses` directly to the base; Anthropic appends `/v1/messages`. No `/v1` is added to a bare chat/responses origin. Trailing slashes are removed; Anthropic bases ending in `/v1` are rejected. One upstream has one base: Zhipu Chat uses `…/api/paas/v4`, while Zhipu Responses uses `…/api/v1`, so create separate upstreams. Paths such as `/api/paas/v4` work directly. Client prefixes `/api/v3/` and `/api/compatible/v1/` are normalized before appending the endpoint. The editor previews the final URL and offers corrections for incomplete OpenAI/Ark bases.

**Credentials.**

- **Who provides the API key**:
  - **The gateway holds the API key** (the default): the gateway sends the upstream's key, and clients need only a gateway key.
  - **Each client sends its own key**: every request must also carry the client's own provider key in an `X-OpenViking-Upstream-Key` header, next to its gateway key. Requests without it fail with 401 "Upstream API key is missing". The gateway passes the key to the provider in the provider's usual header and never forwards `X-OpenViking-*` headers.
- **API key and extra headers** are write-only. With an unchanged base URL, blank values keep stored secrets and removing a header row deletes that header. Changing the base clears all retained credentials: re-enter the API key and custom header values. Read-only views hide header names too.
- **Extra headers** are added to every request to this upstream, such as a provider's version or organization header, and replace a client header of the same name. `Host`, `Content-Length`, `Transfer-Encoding` and `Connection` are not allowed.
- **This key is a Coding Plan or subscription key.** Providers restrict Coding Plan and similar subscription keys to their own coding tools, and using one through a shared gateway can get the subscription suspended. When you check this box, every request routed to the upstream is rejected with 403 "Coding Plan upstreams are disabled; configure a model API key", unless you also check **Allow this plan through the gateway**. The gateway cannot tell such keys apart from ordinary ones, so mark them yourself. Claude subscription tokens (`sk-ant-oat…`) are always rejected.

**Models.**

- **Models** lists the model names this upstream serves. Leave it empty to accept any name. Only listed names and aliases appear in the model list clients can request (`/v1/models`).
- **Model aliases** map a name clients use to the model sent to the upstream, for example the Claude model names Claude Code asks for to the model your provider serves. Aliases appear in the model list, and request logs show the name the client used.
- **Context windows** maps a model to its context window in tokens (at least 1,024), keyed by the name sent to the upstream after aliases. Long conversations are compacted at a share of this window. A model not listed here uses the profile's **Default context window**, or 1,000,000 tokens when that is not set either, so list every model with a smaller window; see [Long conversations](#long-conversations).

**Routing.**

- **Priority.** For each request, the gateway considers the enabled upstreams bound to the key that speak the request's API and serve its model, and picks the highest priority. A conversation stays on the upstream it started with as long as that upstream can still serve it, so priority changes affect new conversations.
- **Enabled.** A disabled upstream receives no requests and disappears from the model list. Conversations that were using it move to the next candidate, which costs one provider cache miss and drops earlier Claude thinking blocks; these requests are flagged **Upstream switched** (`upstream_changed`). Disabling is the reversible alternative to deleting.
- **Allow OpenViking tools** controls whether OpenViking tools may be offered through this upstream. Turning it off takes effect at once, also in conversations that already have the tools.
- **Restore reasoning the client drops.** Clients often leave the model's reasoning out of earlier replies when they send the history back. With this on, the gateway records the reasoning of every reply it relays and puts it back on that reply in later requests: `reasoning_content` for Chat Completions, thinking blocks in their original positions for Anthropic Messages, and reasoning items with plain-text content before their item for Responses. The restored content is byte-identical every turn, so the provider's prompt cache keeps hitting; it does count as input tokens. Replies whose reasoning the client sent back itself are left unchanged. It is on by default for DeepSeek, Volcano Engine Ark and BytePlus ModelArk, and off for other providers. DeepSeek rejects tool requests whose history lacks reasoning, so with this off a DeepSeek upstream offers OpenViking tools only when a request turns thinking off. The same applies to single requests whose reasoning cannot be restored even with this on: the conversation moved to another upstream (`upstream_changed`), a Claude conversation lost its memory record (`missing_injection_record`), or the client resent thinking as text.
- **Minimum cacheable prompt** (Ark and ModelArk only) is the shortest prompt the model caches. It only affects how the request log reports cache eligibility.

Every other upstream setting takes effect at once, in existing conversations too. There is no failover: when the chosen upstream fails, the client receives the error, and an unreachable provider returns 502 "Model upstream is unavailable".

**Test connection** uses the provider's model-list endpoint and stored credentials, with a 10-second timeout. 2xx with a `data` array passes; a failure body or 401/403 means the key was rejected. 404/405 means reachable with no model list, so the key is unverified; 429 also means reachable. Network failures and 5xx fail. Read-only users can test. This does not validate a model or authorize Coding Plan use. Anthropic upstreams send both `x-api-key` and Bearer authorization; OpenAI-style upstreams send Bearer.

An upstream that a key uses cannot be deleted; revoke those keys first, or disable the upstream.

For example, create separate Generic upstreams for Zhipu Chat (`https://open.bigmodel.cn/api/paas/v4`) and Responses (`https://open.bigmodel.cn/api/v1`). For Ark Coding Plan endpoints, use `/api/coding/v3` for chat/responses and `/api/coding` for Anthropic, with the appropriate provider origin.

### Import OpenViking model settings

In an embedded host, **Use OpenViking's model settings** imports all eligible VLM credentials into the selected account. With no management key, only root/dev may import; with one configured, its holder may import. Stable `vlm-<credential id>` IDs make repeat imports update existing entries. The result groups imported, needs-confirmation and skipped credentials. Coding/Plan paths are blocked until you allow them; confirm the provider's terms first. Codex OAuth and Azure credentials are skipped, and LiteLLM credentials without api_base need that field before importing.

**Changing an upstream address:** re-enter the API Key and any required custom header values. Changing `base_url` clears both the old API key and retained headers; only explicitly resubmitted values can be sent to the new destination.

### Context profiles

A context profile is a named set of memory settings. Each gateway key uses one profile, and one profile can serve many keys. You might keep a "Coding" profile that recalls memories, resources and skills, and a "Chat" profile that recalls only memories with a smaller budget.

The **Profiles** tab shows one card per profile with a summary of its four sections and the number of keys using it. While the tab is empty, **Create with recommended settings** creates a profile named "Default" in one click and **Customize** opens the editor; after that, **New profile** opens the editor. A profile that a key uses cannot be deleted; **Duplicate** is a quick way to start a variant.

> **Note**: Saved changes apply to conversations that start afterwards. A conversation takes a snapshot of its key's profile at its first request and keeps it, because changing what the model sees in the middle of a conversation would break the provider cache. Clients that open a new conversation for every chat pick up changes quickly. Claude Code, Codex and clients that send a stable session header keep the old settings until they start a new conversation, or until the conversation has gone unused for 30 days (`session_ttl_days`).

Each section has its own switch; Long conversations has one for each of its two features:

| Section | What it does | Main settings and defaults |
| --- | --- | --- |
| Recall memory | Searches OpenViking with each new user message and appends what is relevant to that message. | Sources: memories, resources, skills · Budget per message: 1,600 tokens · Budget per context window: 30,000 tokens · Relevance threshold: 0.35 · Time limit: 2 s |
| Save conversations | Writes finished turns to an OpenViking session and commits it, so OpenViking can extract memories. | Save the latest reply after: 600 s · Commit after: 20,000 tokens · Keep recent messages: 10 |
| Long conversations | Compacts a conversation near the model's context window: the same model summarizes it, and the summary replaces the earlier messages. Agent-managed context windows are an experimental alternative. | Compaction: on · Compact at: 0.9 of the context window · Summary length: 8,000 tokens · Agent-managed context windows: off |
| OpenViking tools | Lets the model use OpenViking's tools while it answers. Chat Completions, full-history Responses and Anthropic Messages. On by default for new profiles. | Read-only tools selected; tools that change data unchecked |

How the recall settings play out:

- **User profile at conversation start** provides your OpenViking profile independently of recall. With the Read tool enabled, memory and skill catalogs are also provided. **Opening context budget** caps these at 4,000 tokens by default, with at most a quarter for skills; 0 omits them all. This does not consume the recall budget.
- Recall uses the content OpenViking selects and formats within the requested budget, including entries that provide a URI for the model to read later.

- The search query is the user message with client noise removed, cut to **Query length** characters (8,000). Messages shorter than 3 characters are not searched.
- Each message gets at most the smaller of **Budget per message** and what is left of **Budget per context window**; with less than 64 tokens left, the gateway skips the search. The budget per context window starts over whenever the conversation is compacted, by the gateway or by the client. A budget of 0 turns recall off. Token counts are a conservative estimate, not the provider's billing count.
- If OpenViking does not answer within the **Time limit**, the message goes to the model without memory and never gets it later, even when the client retries.
- **Limit by category** (under **Advanced settings**) searches only the categories with a limit above 0, among events, entities, preferences, experiences, resources and skills, and takes at most that many entries from each. When it is off, all sources are ranked together.

**Recall summary.** With **Show recall summary** on (off by default), the reply to each new user message starts with a short summary of what the gateway added to that message:

```text
> OpenViking context: user profile, memory index, skill list, earlier sessions
> OpenViking recall: 4 items (3 memories, 1 resource) — booking_duplicate_handling, user_lang_pref, +2 more
```

The `context` line appears only in the first reply of a conversation and lists only what was actually provided at its start: the user profile, the memory and skill catalogs, and where earlier parts of the conversation are saved in OpenViking. The `recall` line counts the recalled entries by kind and names up to three of them. When the search fails, the line gives the reason instead, for example `> OpenViking recall failed: OpenViking unavailable`. The `recall` line is left out when nothing relevant was found, when recall is off or when its budget is used up, so after the first reply such a message gets no summary at all. Tool steps and other follow-up requests never get one, and a retry of the same message shows the same summary again. In Anthropic Messages the summary is a text block of its own, in Responses an assistant message of its own, and in Chat Completions the start of the reply text. Requests that ask for structured output (a JSON format) and Chat Completions requests with `n` above 1 get no summary. The setting works with or without OpenViking tools. The model never sees it: when the client sends the reply back with the next message, the gateway removes the summary before it forwards the request, and it never saves the summary to OpenViking. Like the other profile settings, the change applies to new conversations.

How the saving settings play out:

- A turn is saved when the next user message arrives. **Save the latest reply after** sets how long the gateway waits before it also saves the last turn and commits the session, so short conversations get committed too.
- **Commit after** commits the session whenever that many tokens are waiting in it. OpenViking extracts memories at each commit, in the background.
- **Keep recent messages** keeps the newest whole turns, at least this many messages, in the session after a commit.
- Recall and saving are independent: a profile that only saves, or only recalls, is valid.

The [Configuration reference](#configuration-reference) lists every setting with its API name and limits.

### Gateway keys

A gateway key (`ovgw_…`) is what a client uses in place of a provider key. Each key belongs to one OpenViking user and uses one context profile and a set of upstreams.

To issue one, choose **Issue key** on the **Keys** tab and fill in:

- **Name**, to recognize the key later, for example "alice · Claude Code".
- **OpenViking user**: users issue for themselves; account admins can choose users in their account and root can switch accounts first. The gateway reads the selected user’s API key using the caller’s credentials. If only a hash is stored, paste the user’s current API key. A root API key cannot be bound; local dev uses default/default without a credential.
- **Context profile**.
- **Upstreams**: at least one. The key can reach only these.
- **Allowed models** (optional): leave empty to allow every model these upstreams serve. The names are the ones clients send, including aliases. Requests for other models fail with 403 "Model is not allowed by this key", and the model list shows only allowed names.

Before issuing, the gateway checks the user's OpenViking key with OpenViking. Outside local dev, a pasted root key, a key from another account or an invalid key, or an unreachable OpenViking, stops the issue; see [Issuing a key fails](#issuing-a-key-fails).

The **Copy your gateway key** dialog then shows the full key. This is the only time it is shown: the gateway keeps only a hash and the first characters. The dialog also has ready-to-paste setups for Claude Code, Codex CLI and chat clients with the real key and gateway address filled in. If a key is lost, issue a new one.

- **Keys cannot be edited.** To change a key's profile, upstreams, allowed models or OpenViking user, issue a new key and revoke the old one.
- **Conversations belong to the user, not the key.** A new key for the same OpenViking user continues the same conversations when the client sends the same session ID, so replacing a key is seamless.
- **Revoking** prevents subsequent requests from using the key. User conversation state and logs remain until explicit deletion or retention expiry; revocation alone does not remove user data.
- **When an OpenViking key changes.** The gateway rechecks the bound credential with a 30-second identity cache. Regenerating or revoking it invalidates its gateway keys after cache expiry. Issue a new gateway key with the current OpenViking credential; stored user data is preserved.
- **One key per person.** Everyone using a key shares the memory of its OpenViking user. Issue one key per person, and per client when you want different profiles.

**Delete this user’s gateway data…** is available to admin/root. It revokes the user’s keys and removes gateway conversation state and logs, including pending content. OpenViking memories are unaffected. Embedded user/account deletion triggers the same cleanup; see [Security and data](#security-and-data).

### Requests and overview

The **Overview** tab summarizes the latest 10,000 request-log records within the log retention period (30 days by default):

- A **Get started** checklist until the first request arrives.
- **Requests**: the number of model requests, when the last one arrived and how many output tokens they used.
- **First-call cache hit rate**: the share of input tokens served from the provider's prompt cache on the first model call of each new user message. This is the gateway's most important health signal. It should stay close to what the provider reaches without the gateway; a drop usually means the history the gateway sends no longer matches the previous request. See [The first-call cache hit rate dropped](#the-first-call-cache-hit-rate-dropped).
- **Within-turn cache hit rate**: the same for later calls within a turn, such as tool steps and OpenViking tool rounds. It is usually high either way.
- **Memory entries recalled**: entries added to new messages, and how long a search in OpenViking took on average, counting only messages that searched.
- **OpenViking**: Connected, Degraded or Starting, with the version, authentication mode and, when degraded, the reason. While OpenViking is unreachable, requests still reach the model without memory. Under **Saving conversations**, the card also counts conversations that are retrying or have paused saving to OpenViking.
- **Degraded requests**: how often each issue occurred, with an explanation; the issues are listed in the [troubleshooting table](#issues-on-degraded-requests).
- **Recent requests**, with a link to the Requests tab.

The **Requests** tab lists the latest 1,000 records, newest first, and refreshes its first page every 30 seconds while you view it. Filter by **All**, **New messages**, **Tool steps** or **Issues**, by **Request type**, or search by model, conversation or key. Columns show the type, model (the name the client sent), the provider's HTTP status, tokens (input, cached share, output), **Memory** (`+3`: entries this request's search added, with the search time; `↺ 4`: memory added to 4 earlier messages, sent again unchanged), **Saving** status and **Issue**. Expand a row for the conversation ID, upstream, key, duration, the token breakdown, the memory search result, the context window (how full it was, and any compaction or new window), saving status with the next retry time, OpenViking tool details and, for an issue, what it means and what to do.

Only requests that reached a model provider are logged. Requests the gateway rejects first, such as an invalid key, a model the key does not allow, no matching upstream or a body over the size limit, are not; the client receives the error instead.

| Type | What it is | Memory |
| --- | --- | --- |
| New message | A request whose last message is a new user message | Searched once, earlier memory replayed; the turn is saved |
| Tool step | The same turn continues, for example with tool results | Earlier memory replayed; saved with its turn |
| Housekeeping | Title, summary, compaction, suggestion and permission-check requests | Earlier memory replayed; never saved |
| Sub-agent | Requests from a client's sub-agents | Earlier memory replayed; never saved |
| Token count | Token counting requests (`/v1/messages/count_tokens`) | Earlier memory replayed, so counts match what is sent |
| Forwarded | Requests passed through unchanged, without memory: other endpoints, Responses requests that rely on provider-side state, compressed or unreadable bodies | None |
| Memory sync | Background saving events; not model requests | — |

The **Saving** column shows where each conversation stands:

- **On**: saving works normally.
- **Off**: the conversation's profile does not save.
- **Retrying**: OpenViking rejected a save or could not be reached. The gateway retries after 10, 20, 40 and 80 seconds.
- **Paused**: five attempts failed. The gateway keeps retrying every 5 minutes until it succeeds. Unsaved turns are kept until they are saved or the conversation expires.

The expanded row shows the reason, for example that OpenViking rejected the key (`openviking_http_401`: the user's OpenViking key is no longer valid) or that OpenViking is unreachable (`openviking_unavailable`). A note that the conversation history changed (`history_changed`) is normal: the client edited, regenerated or compacted the conversation, and the gateway continues saving in a new OpenViking session with the current history.

**Resync conversation…** in an expanded row starts a fresh OpenViking session for that conversation. The next time the user sends a message, the gateway writes the conversation's full history to it. What was already saved stays in the old session, so OpenViking may extract some memories twice. Use it when a conversation stays paused after you fixed the cause, or when you want a clean copy of a conversation. Nothing happens until the next message. Resync needs the key shown on the record; if that key has been revoked, use a newer record of the same conversation.

## OpenViking tools

With OpenViking tools on, the model can use the tools available on your OpenViking server while answering, within the user's access permissions. The gateway runs the tools, and the client continues receiving the answer until the reply ends. Reported token usage includes all model calls made during that reply. For what this means for different clients, which tools the model can use and what users see in the reply, see [Agentic memory for any client](15-gateway.md#agentic-memory-for-any-client) in the OpenViking Gateway guide; this section covers client requirements, settings and limits.

Tools support streaming and nonstreaming requests in all three protocols:

| Protocol | Client requirements |
| --- | --- |
| Chat Completions | Send the full `messages` history. |
| Responses | Use `store: false` and full `input` history; pass the entire returned `output` array back on the next request, including reasoning and client tool calls. Requests with `previous_response_id`, `conversation` or `background` continue to pass through without gateway tools. `item_reference` cannot reconstruct full history. |
| Anthropic Messages | Send the full `messages` history and keep thinking/signature blocks intact. Token-count requests get the same tool definitions but do not execute tools. |

If a reply calls both OpenViking and client tools, the gateway handles the OpenViking calls. The client runs its own tools and returns their results as usual. Responses supports client function and custom tools; Anthropic Messages supports client tool_use calls.

This is meant for chat apps and API apps that cannot connect to OpenViking's MCP server. Clients that support MCP or have a plugin, such as Claude Code and Codex, are better served by those, where the client shows each tool call with its result and asks for permission first.

New context profiles have **OpenViking tools** on, including those made with **Create with recommended settings**. The **Tools** list comes from your OpenViking server, and only the read-only tools are selected by default: `find`, `search`, `grep`, `glob`, `list`, `tree`, `read`, `list_watches`, `get_acl`, `list_users`, `list_groups` and `health`. The tools that change data are left unchecked: `remember`, `write`, `edit`, `add_resource`, `add_skill`, `forget`, `set_acl` and `cancel_watch`. For example, check `remember` to let the model save memories, or `add_resource` to let it import web pages and attachments. Existing profiles keep their settings. If the list is empty, check connectivity and your OpenViking credential. New tools added to OpenViking are available automatically unless you uncheck them; ongoing conversations keep the tool list they started with.

**Allow OpenViking tools** is on by default for each upstream. With both switches on, new conversations get the selected tools with an `openviking_` prefix: for example `openviking_find`, `openviking_read`, `openviking_grep` and `openviking_glob`. The descriptions in Studio explain what each available tool does. Read-only labels appear only when OpenViking supplies that information. The definitions of the offered tools go with every request in the conversation; OpenViking's full set adds about 3,500 input tokens, so unchecking tools nobody needs also saves tokens.

> **Note**: All selected tools run without the client's permission prompts, including tools that write or delete data. Before you check a tool that changes data, make sure that is acceptable; for profiles used by chat apps, we recommend leaving `forget` and `set_acl` unchecked. A tool-call notice shows what happened; it is not a request for approval.

**Tool call notices.** With **Show tool calls** on (the default), the reply gets a one-line notice for each OpenViking tool call the gateway runs, at the point in the answer where the call happened:

```text
> OpenViking find: "release date" — done
```

The notice names the tool and, where available, what it works on: the search query, the URI read, listed or written, or the file or skill being imported. In a streaming reply it appears as soon as the call starts, so users can see that something is happening during a slow import; a nonstreaming reply gets the same lines in the final message. The outcome is added when the call ends: `done`; `failed` when the call returned an error; `skipped` when the gateway refused the call and did not run it, for example because the round limit or the token budget was reached. Each notice is a paragraph of its own. In Chat Completions the notices are part of the reply text. In Anthropic Messages the notices for one round of calls share one text block, and in Responses they share one assistant message. The model never sees these lines; it gets the actual calls and results. The gateway does not save the notices to OpenViking either. To hide them, turn off **Show tool calls** in the profile's **OpenViking tools** section, and the reply contains only the model's own output. Like the other tool settings, the change applies to new conversations.

Whether a conversation gets tools is decided at its first request that could use them, not at a helper request such as title generation, and kept, so profile changes reach new conversations only. The one exception turns tools off: when the client shows OpenViking tools of its own before the model has called a gateway tool, for example because an MCP server connected after the first turn. When a request runs without tools, its detail on the Requests tab says why:

| Reason | Cause |
| --- | --- |
| `tools_require_full_history` | Responses tools need full input history with `store: false`, without `item_reference`. |
| `upstream_tools_disabled` | The upstream has **Allow OpenViking tools** off. |
| `tools_multiple_choices` | The request asks for several choices (`n` greater than 1). |
| `tools_structured_output` | The request asks for structured output (`response_format`, `text.format` or `output_config.format`). |
| `tools_non_function` | The Chat Completions client sent a tool of a type other than function. |
| `tools_forced_choice` | `tool_choice` is `required` or names a specific tool. |
| `deepseek_reasoning_history_required` | The upstream's provider is DeepSeek, the request does not turn thinking off (`"thinking": {"type": "disabled"}`), and earlier replies would reach it without their reasoning: **Restore reasoning the client drops** is off, or this request cannot restore it because the conversation moved upstream, lost its memory record or has thinking resent as text. Standard Responses requests have no `thinking` field, so they count as thinking on. |
| `tool_name_collision` | The client already has OpenViking tools, so the gateway leaves its own out for the rest of the conversation, except on a branch where the model already called them. A misdetection also lasts for the conversation; start a new one. This means: an MCP server name (`server_label`, `mcp_server_name`, `mcp_servers[].name`) contains the word `openviking`; a tool named after an OpenViking tool has that word too, such as `mcp__claude_ai_OpenViking__search` (a connector's `authenticate` tool does not count), including deferred tools listed by name on their own line; or a prefixed tool named after an OpenViking tool, such as `mcp__ov__read`, mentions `viking://`. On any request, a client tool with the same name as a gateway tool also counts, since upstreams reject duplicate names. |
| `tools_unavailable` | OpenViking could not provide its tool list and no previously loaded list was available. Start a new conversation after connectivity recovers. |
| `tools_not_selected_at_session_start` | The profile has tools on, but the conversation's tool list is empty: the profile selects no OpenViking tools, or the conversation started before an upgrade that left it without tools. Start a new conversation. |

Housekeeping and sub-agent requests, and conversations where an OpenViking plugin was detected, do not get OpenViking tools. When such a request resends history in which the model used them, the gateway still sends the tool definitions so that the history stays valid, but refuses new OpenViking calls. The client's own tools stay available in every request.

**Importing files and skills.** The import tools are available when your OpenViking server provides them and they are selected in the profile. URL imports do not need a shell tool. Local files need one of these upload paths:

- **A shell tool** (a client tool named like bash, shell, exec_command, terminal or run_command): the import tool returns a one-time upload link, and the model uploads the file with the client's own shell tool, which does go through the client's permission prompt. Skill directories are zipped first. The link points to `public_url` plus `/gateway/uploads`, so `public_url` must be an address the client can reach and your proxy must route that path to the gateway. Without `public_url`, imports fail with "Set gateway.public_url for client uploads".
- **An attached file**: without a shell tool, the model can import a file attached to a user message, sent either as base64 file data or as attachment text. This includes Responses `input_file` and Anthropic `document.source` with embedded `base64` or `text` data. Open WebUI usually sends only the extracted text, which is imported as a text file. Attachment URLs and provider file IDs are not fetched; use the import tool's URL parameter for a supported remote resource.

Uploads are limited to `max_body_bytes` (32 MiB by default).

The limits sit under **Advanced settings** in the profile's **OpenViking tools** section:

| Setting | API name | Default | Range | At the limit |
| --- | --- | --- | --- | --- |
| Rounds per request | `tool_max_rounds` | No limit | 1 or more | Further OpenViking calls are refused; the model continues with the results it has and the client's own tools. |
| Time limit per call | `tool_timeout_seconds` | 30 s | up to 120 s | The call returns an error to the model. |
| Result size | `tool_result_bytes` | 65,536 bytes | 1,024–1,048,576 | The result is truncated. |
| Total time | `tool_total_seconds` | No limit | more than 0 s | The request fails with 504 "Hidden tool request timed out". |
| Token budget | `tool_total_tokens` | No limit | 1,024 or more | Further OpenViking calls are refused; the model continues with the results it has and the client's own tools. This is not a billing cap; the final answer can exceed it. |

Rounds, total time and the token budget are unlimited by default, as in an agent's own tool loop: the model keeps using OpenViking tools until it finishes. In a streaming request, the user can stop the reply at any time, and the gateway stops when the client disconnects. A non-streaming request keeps running after the client disconnects, so set **Total time** if your clients send non-streaming requests. Leave a field empty for no limit. Profiles saved before this default changed keep their earlier values (5 rounds, 120 s, 100,000 tokens); clear the fields to remove the limits.

The token budget covers the extra calls, results and subsequent model output from using OpenViking tools. Existing conversation history, tool definitions and images do not count. The gateway preserves the client's answer-length limit (`max_tokens`, `max_completion_tokens` or `max_output_tokens`). Once the budget is spent, the gateway refuses further OpenViking calls, and the model continues with the results already available and the client's own tools.

When an individual tool call fails, the model receives an error result and can continue answering. If the reply itself fails, for example because the provider rejects a follow-up model request or the model keeps calling OpenViking tools after they were refused, the client receives an error and the request log shows **OpenViking tools failed** (`hidden_tool_loop_failed`). Streaming Responses requests end with `response.failed`; Chat Completions and Anthropic Messages report `gateway_tool_error`.

If an upstream keeps failing, turn off its **Allow OpenViking tools** setting. Claude conversations that have already used these tools may also show **Tool history unavailable** (`hidden_tool_history_unavailable`): earlier thinking can no longer be used and has been removed. Check the tool skip reason in the request detail, then restore the original settings or start a new conversation. A retried tool call reuses the first call's result, and an interrupted write is never repeated automatically.

## Long conversations

A model's context window limits how long a conversation can get. With **Compaction** on (the default), the gateway replaces the earlier part of a conversation with a summary before the conversation fills the window.

**When it compacts.** On each new user message and each tool step, the gateway estimates how full the context window is: the token usage the provider reported for the previous reply in this conversation, plus an estimate for the messages added since. Without that usage, for example for streamed Chat Completions without `stream_options.include_usage`, it estimates the whole request. When the estimate reaches **Compact at** (0.9 of the window), the gateway compacts. Housekeeping, sub-agent and token-count requests never trigger compaction.

The window comes from the upstream's **Context windows** entry for the model (the name sent to the upstream), else from the profile's **Default context window**. Without either, the gateway assumes 1,000,000 tokens. For a model with a smaller window, set one of the two; otherwise the provider rejects the conversation as too long before compaction ever starts.

**How the summary is written.** The gateway sends one extra, nonstreaming request to the same upstream and model, with the client's system prompt, tools and thinking settings unchanged, so most of it is served from the provider's prompt cache. Settings a plain-text summary cannot follow are left out: a forced tool choice is turned off, and structured output formats and stop sequences are dropped. The request holds the conversation up to the cut and an instruction to summarize it for the model itself: the user's goals and latest request word for word, decisions and progress, files, paths and identifiers, errors and their fixes, open items, and key facts from recent tool results. An earlier summary is folded into the new one. The summary is plain text of at most **Summary length** tokens (8,000). Thinking counts against the output cap, and some models (DeepSeek V4.1, for example) think by default even when the request carries no thinking settings, so the gateway always allows 16,000 tokens of output beyond Summary length; when the client sets a manual thinking budget, it adds that budget instead. The provider bills the request like any other.

**What the model receives afterwards.** Where the cut falls depends on the request:

- On a new user message, the cut falls right before that message. The model receives the system and developer messages, the summary as one user message, and the new message.
- During a tool step, the cut falls after the latest tool result, and the model continues the task from the summary.

The summary message starts with a note that the gateway replaced the earlier part of the conversation and that the user did not write it, and ends with the conversation's opening note and profile, so these survive compaction. Every later request that resends this history, including housekeeping and token-count requests, gets the same replacement; sub-agents usually send their own, separate history, which stays untouched. A compaction costs one provider cache miss at the cut; after that, the cache works as before.

**Nothing before the cut is kept word for word**, not even the latest turns. The model's earlier messages and their thinking depend on the history being replaced, and keeping some of them would break Claude's thinking signatures. Everything generated after the cut is sent as usual, thinking included.

**Searching what was cut.** The conversation itself is still saved in OpenViking. When **Save conversations** is on and the conversation offers the `openviking_grep` and `openviking_read` tools, the summary is followed by directions: the conversation's OpenViking sessions (`viking://user/<user_id>/sessions/<session_id>/`), where their messages are stored, and how to search them: grep a session with a specific pattern to get line numbers, then read the lines around a match. When a cut falls in the middle of a turn, the part before it is saved to OpenViking right away, so the model can search it while the turn continues.

**When compaction fails.** If the summary request fails, or the reply calls a tool, is empty or is cut off at the length limit, the gateway sends the full history instead and does not try again in that conversation for 60 seconds. The request's detail on the Requests tab shows **Compaction failed** with the reason. If the context is still at or above **Compact at** after a summary, for example because the system prompt and tools alone fill most of the window, the gateway keeps the summary but also waits 60 seconds before compacting again, with the reason `still_over_threshold`.

**Working Memory is not used.** The summary comes from the conversation's own model, not from OpenViking's session summaries. New OpenViking sessions the gateway creates have Working Memory turned off: commits still archive the messages and OpenViking still extracts memories, but it writes no session summary.

**Memory budget.** **Budget per context window** starts over after each compaction, so an entry added before the cut can be added again.

**Clients' own compaction.** After the gateway compacts, the client still resends its full history, and the gateway replaces the part before the cut on every request. The usage the client sees is small, so clients that compact by token usage seldom compact on their own, and a very long session can eventually exceed `max_body_bytes`. When a client compacts or edits its history, the gateway saves the new history to a new OpenViking session; when the search tools are available, the opening note of the new history points the model to the earlier sessions.

Each request's detail on the Requests tab shows how full the context window was, whether earlier history was replaced, and the size and duration of a newly written summary.

### Experimental: agent-managed context windows

With **Agent-managed context windows** on (off by default), the model decides itself when to start a fresh context window and writes its own hand-off notes, instead of waiting for a summary. The setting takes effect only in conversations that get [OpenViking tools](#openviking-tools); elsewhere it does nothing, and compaction works as described above. Compaction, when on, also remains the fallback for a model that never starts a new window.

What the model sees:

- Two tools next to the OpenViking tools. They are not part of the profile's **Tools** list; this setting alone controls them.
  - `openviking_new_context(reason, notes, next_steps)` starts a new window; `next_steps` is optional. It must be the only tool call in its round. Called together with other tools, every call in the round returns an error and the window is not reset. It also returns an error when the gateway already replaced the context at the same point in this request, for example by compacting it.
  - `openviking_context_remaining()` reports the window number, the estimated tokens used and left, the user turns in this window, the time since the user's previous message, and a one-line recommendation.
- A line in the opening note saying that the model manages its own context windows with these tools.
- A status line after each new user message, at the end of the memory block: `[context-status] window wN · ~X/Y tokens (P%) · T since your previous message`.
- Reminders when the window reaches **Soft reminder at** (0.7) and **Hard reminder at** (0.85). The soft reminder, sent once per window, asks the model to start a new window once the current step is done, with notes that keep the exact paths, line numbers and values it will need. The hard one asks for a new window now and repeats on every step until the model starts one or compaction takes over. A reminder is attached to the new user message or, during a tool step, to the tool result.

When the model calls `openviking_new_context`, the gateway replaces the conversation so far with a window header, and the model carries on in the same reply. The header says that the model started window N and that the user did not write it, then gives the reason, the notes and next steps, the user's latest message word for word, directions for searching earlier windows (under the same conditions as for compaction), and the conversation's opening note. Later requests rebuild the same window from the client's history. The part before the reset is saved to OpenViking right away and stays searchable. The reason and notes appear only in the header; they are not saved to OpenViking. A reset costs one provider cache miss but no summary request.

Each request's detail on the Requests tab shows the window number, whether the model started a new window, and which reminder it got. How well this works depends on the model, and the behavior may change in later releases.

## Security and data

**Isolation by user.** Each gateway key is bound to one OpenViking user. The gateway searches memory, saves conversations and calls tools with that user's own OpenViking key, so it never has more access than the user does, and each user's conversation state is kept apart. Sessions and memories always live in OpenViking; the gateway stores only the state it needs to keep conversations going.

**What the gateway stores.** Under `storage_path`, the gateway keeps two SQLite databases:

- `management.sqlite3`: upstreams, context profiles, gateway keys, request logs and the mapping from Responses IDs to upstreams. Upstream API keys and header values, and the OpenViking keys bound to gateway keys, are encrypted. Gateway keys themselves are stored only as a hash and a short prefix.
- `kernel.sqlite3`: per-conversation state, including the memory text added to messages (needed for replay), OpenViking tool rounds, summaries and conversation text waiting to be saved.

Stored values are encrypted with the encryption key. The files are readable only by the user running the gateway.

**What request logs contain.** Metadata only: request type, model, status, upstream, a hash of the conversation ID, token counts, memory counts and timing, saving status and issues. Never message text, recalled memory, tool arguments or results, or any key. The gateway writes no HTTP access log, and its error messages never repeat submitted values.

**What providers see.** Every upstream receives the full model input, including memory the gateway added and OpenViking tool results. Only add providers you trust with that data.

**Retention.** Cleanup runs every hour:

| Data | Kept for | Setting |
| --- | --- | --- |
| Conversation state | 30 days after its last use | `session_ttl_days` |
| Request logs | 30 days | `log_retention_days` |
| Responses ID mappings | 30 days | `response_ttl_seconds` |
| Sessions and memories in OpenViking | OpenViking's own rules | — |

**Deleting data.** Revoking a gateway key or its bound OpenViking credential does not delete user data. Admin/root may call `DELETE /api/v1/admin/gateway/users/{user_id}/data` to revoke that user's keys and delete gateway state and logs, including pending content. Root may delete all gateway data for an account with `DELETE /api/v1/admin/gateway/account/data?account_id=<account>`. Embedded OpenViking user/account deletion triggers in-process cleanup; standalone deployments retain data until explicit deletion or TTL. Memories and sessions already saved in OpenViking are unaffected. Deleting a memory in OpenViking does not erase an injected copy already retained for replay; delete the user's gateway state to remove that copy immediately.

**Access.**

- Clients authenticate with gateway keys, sent as `Authorization: Bearer` or `x-api-key`. Keys can be revoked at any time.
- The gateway uses each user's own OpenViking key for memory search, saving and tools, so it can do only what that user can do.
- Management keys authorize account-level objects only. OpenViking identity controls key issuance, logs and memory access. See the [access matrix](../api/25-gateway.md#credentials-and-account-scope).
- Account admins can issue gateway keys for any user in their account (when OpenViking stores only key hashes, the user has to provide their key). A gateway key lets its holder recall memory and call tools as that user, so the right to issue keys amounts to reading the memory of every user in the account. Give it only to admins you trust.
- `/gateway/uploads` needs no gateway key. A one-time token signed by OpenViking authorizes each upload, and the gateway forwards it only to OpenViking's upload endpoint. Keep its query string out of proxy logs.
- `X-OpenViking-*` headers, client credentials and cookies are never forwarded to upstreams.

**Backups.** Back up `storage_path`, including both databases with their `-wal` and `-shm` files, together with the encryption key. For a consistent copy, stop the gateway or use SQLite's online backup (`sqlite3 kernel.sqlite3 ".backup kernel.backup.sqlite3"`, and the same for `management.sqlite3`). What a loss means:

- **`kernel.sqlite3` lost**: memory added earlier cannot be replayed. Each ongoing conversation pays one provider cache miss, Claude conversations lose earlier thinking once (**Memory record missing**, `missing_injection_record`), and turns not yet saved are lost.
- **`management.sqlite3` lost**: upstreams, profiles, keys and request logs are gone. Set them up again and issue new keys.
- **Encryption key lost**: neither database can be read. Start with an empty `storage_path`.

## Day-to-day operations

The Gateway page is available to every connected identity. Start with upstream → profile → key → client. Object editing follows the management-key setting; users see only their own keys and logs.

- **Watch the first-call cache hit rate.** The **First-call cache hit rate** on the Overview tab reflects prompt caching across turns and should stay close to what the provider reaches without the gateway. The **Within-turn cache hit rate** covers tool steps and is normally high. If the first-call rate drops noticeably, see [The first-call cache hit rate dropped](#the-first-call-cache-hit-rate-dropped).
- **Use the request log to track down a single request.** Filter by new messages, tool steps or issues, then expand a row to see the recall result, saving status and why OpenViking tools were off; see [Requests and overview](#requests-and-overview).
- **Set the public address.** Set `public_url` to the address clients can reach. Guides otherwise use the request origin; explicit configuration keeps client and upload URLs consistent behind a proxy.
- **Back up regularly.** Back up the gateway's storage directory and the encryption key; for how, and what a loss means, see **Backups** under [Security and data](#security-and-data).

### Upgrades

- OpenViking Server must be at least `min_server_version` (0.4.16). With an older server, memory search and saving stop and no keys can be issued. When you upgrade the gateway and OpenViking separately, upgrade OpenViking first, then the gateway.
- Read the release notes before you upgrade. If a new gateway version cannot read the old storage format, it stops at startup with `Unsupported gateway schema`; see [The gateway does not start](#the-gateway-does-not-start).

### Monitoring

The gateway exports no metrics such as Prometheus and writes no HTTP access log. Check how it is doing on Studio's Overview and Requests tabs, or through the management API on OpenViking Server (`/api/v1/admin/gateway/overview` and `logs`); use the host’s `/health` to check liveness (the server health endpoint in embedded deployments).

## Design trade-offs

The table explains why the gateway's key design choices were made and what they cost, to help you judge whether it fits your setup:

| Design | Choice and reason | Cost |
| --- | --- | --- |
| Where it plugs in | On the model call path instead of a plugin in every agent: clients that cannot install plugins can use it, and provider keys are managed in one place. | It cannot see the working directory; keeping projects' memory apart means using a different OpenViking user per project. |
| Who runs the tools | The gateway runs OpenViking tools during the reply, stores the tool rounds and replays them unchanged on the next turn: even clients with no tools and no MCP support get a model that uses memory itself. | No per-call approval; clients must send the full history every turn; admins need to narrow the tool list and limits. |
| Where memory goes | At the end of the newest user message, not in the system prompt: changing one character of the system prompt invalidates the whole cache, while appending at the end and replaying it unchanged keeps the prefix stable. | The gateway must reliably record the exact text of every addition; if a record is lost, the cache misses once. |
| When turns are saved | One turn behind: only when the next message brings the previous turn back is it clear that the user kept it. | The last turn is saved only after **Save the latest reply after** (10 minutes by default). |
| Long conversations | The gateway has the same model write the summary: only the gateway knows what the model actually received, and the summary request mostly hits the cache. | The turn that triggers compaction costs one extra, billed model call; text before the cut is no longer sent to the model. |
| Local state | SQLite on a single host, with no Postgres or Redis dependency, suited to self-hosting on one machine. | The gateway cannot fail over between machines and is a single point on the model call path. |
| Client identity | The gateway issues its own keys instead of using OpenViking keys directly: they can be revoked on their own, and one person's clients can be bound to different context profiles. | The gateway has to keep users' OpenViking keys encrypted; a leaked gateway key exposes that user's memory. |

## Configuration reference

Embedded settings live in `ov.conf.gateway`. Set `enabled: true` to mount the gateway; host, port and workers come from the server. Standalone settings live at the root of `gateway.conf` and add required `openviking_url`, `host` (127.0.0.1), `port` (1935) and `workers` (1–64, default 1), without `enabled`.

| Shared field | Default | Meaning |
| --- | --- | --- |
| `public_url` | empty | Client origin; guides fall back to request origin |
| `storage_path` | `~/.openviking/gateway` | Local storage directory, including generated encryption.key |
| `encryption_key_env` | `OPENVIKING_GATEWAY_ENCRYPTION_KEY` | Optional external Fernet key |
| `management_key` | empty | Optional object-write secret; minimum 32 characters; supports `$VAR` expansion |
| `min_server_version` | `0.4.16` | Minimum version checked on authenticated issuance/capture recovery |
| `session_ttl_days` | `30` | Conversation inactivity expiry, minimum 1 day |
| `response_ttl_seconds` | `2592000` | Responses mapping expiry, minimum 60 seconds |
| `log_retention_days` | `30` | Log retention, minimum 1 day |
| `max_body_bytes` | `33554432` | Request/upload size limit, minimum 1024 bytes |
| `upstream_timeout_seconds` | `600` | Provider request timeout, greater than zero |
| `health_interval_seconds` | `60` | Health polling interval, minimum 1 second |

URL values must be HTTP(S) without credentials, query or fragment. Unknown configuration fields fail validation. Standalone environment fields use `OPENVIKING_GATEWAY_` followed by the uppercase field name; `OPENVIKING_GATEWAY_CONFIG_FILE` selects the standalone JSON file. `OPENVIKING_CONFIG_FILE` selects the server configuration only.

**Context profile settings.** Studio label first, then the name used in the management API (where profiles are called `policies`):

| Studio label | API name | Default | Limits | Meaning |
| --- | --- | --- | --- | --- |
| Name | `name` | `Default` | | Display name. |
| Recall memory | `recall` | `true` | | Search memory for each new user message. |
| User profile at conversation start | `profile` | `true` | | Provide the user profile when a conversation starts, independently of recall. |
| Opening context budget | `profile_max_tokens` | `4000` | 0–32,000 | Separate budget for the profile and catalogs; 0 omits them. Catalogs require the Read tool. |
| Show recall summary | `show_recall` | `false` | | Start the reply with a one-line summary of what OpenViking added, or why recall failed. The model never sees it. |
| Sources | `context_types` | `memory`, `resource`, `skill` | at least one | What to search: memories, resources, skills. |
| Budget per message | `max_tokens` | `1600` | 64–32,000 | Most tokens added to one message. |
| Budget per context window | `session_max_tokens` | `30000` | ≥ 0 | Most tokens added within one context window; starts over after each compaction. 0 turns recall off. |
| Relevance threshold | `score_threshold` | `0.35` | 0–1 | Lowest relevance score accepted. |
| Time limit | `recall_timeout` | `2` (seconds) | up to 30 | How long a search may take before the message goes on without memory. |
| Query length | `query_max_chars` | `8000` | 3–32,000 | Characters of the message used as the search query. |
| Limit by category | `quotas` | `{}` (off) | keys: `events`, `entities`, `preferences`, `experiences`, `resources`, `skills` | Most entries per category; only categories above 0 are searched. |
| Save conversations | `capture` | `true` | | Save finished turns to OpenViking. |
| Save the latest reply after | `idle_seconds` | `600` (seconds) | ≥ 1 | Quiet time before the last turn is saved and the session committed. |
| Commit after | `commit_tokens` | `20000` | ≥ 1 | Tokens waiting in the session that trigger a commit. |
| Keep recent messages | `keep_recent_messages` | `10` | 0–1,000 | Messages left in the session after a commit. |
| Compaction | `compaction` | `true` | | Replace the earlier part of a conversation with a summary near the context window. |
| Compact at | `compaction_threshold` | `0.9` | 0.5–0.98 | Share of the context window that triggers compaction. |
| Summary length | `summary_max_tokens` | `8000` | 1,000–32,000 | Most tokens in one summary. |
| Default context window | `context_window` | unset (1,000,000 assumed) | ≥ 1,024 | Window used when the upstream does not list the model. |
| Agent-managed context windows | `agent_windows` | `false` | | Experimental. Let the model start new context windows itself; needs OpenViking tools. |
| Soft reminder at | `window_soft_ratio` | `0.7` | 0.3–0.95, below `window_hard_ratio` | Share of the context window at which the model is reminded to start a new window soon. |
| Hard reminder at | `window_hard_ratio` | `0.85` | 0.4–0.97 | Share of the context window at which the model is told to start a new window now. |
| OpenViking tools | `gateway_tools` | `true` | | Offer OpenViking tools for Chat, full-history Responses and Anthropic Messages. |
| Unselected tools | `disabled_tools` | The 8 tools that change data: `remember`, `write`, `edit`, `add_resource`, `add_skill`, `forget`, `set_acl` and `cancel_watch` | Tool names from OpenViking, without `openviking_` | Tools unavailable to new conversations; every other available tool is offered, including ones added later. API requests that omit the field get the default list; an empty list enables every available tool when the main switch is on. |
| Show tool calls | `show_tool_calls` | `true` | | Add a one-line notice to the reply for each OpenViking tool call. |
| Rounds per request | `tool_max_rounds` | `null` (no limit) | 1 or more | See [OpenViking tools](#openviking-tools). |
| Time limit per call | `tool_timeout_seconds` | `30` | up to 120 | |
| Result size | `tool_result_bytes` | `65536` | 1,024–1,048,576 | |
| Total time | `tool_total_seconds` | `null` (no limit) | more than 0 | |
| Token budget | `tool_total_tokens` | `null` (no limit) | 1,024 or more | |

**Upstream settings:**

| Studio label | API name | Default | Meaning |
| --- | --- | --- | --- |
| Name | `name` | | Display name. |
| Protocol | `protocol` | | `anthropic` (Anthropic Messages), `chat` (Chat Completions) or `responses` (Responses). |
| Provider | `vendor` | `generic` | `generic`, `anthropic`, `openai`, `deepseek`, `ark` (Volcano Engine Ark) or `byteplus` (BytePlus ModelArk). Studio offers only the protocols the provider supports; the API accepts any pair. |
| Base URL | `base_url` | | Provider address; see [Upstreams](#upstreams) for each provider's default and the path rules. |
| Who provides the API key | `auth_mode` | `managed` | `managed` (The gateway holds the API key) or `passthrough` (Each client sends its own key). |
| API key | `api_key` | | Write-only. Blank keeps the stored key only when base_url is unchanged. |
| Extra headers | `headers` | `{}` | Write-only; names are hidden in read-only views. Changing base_url clears retained headers. |
| Models | `models` | `[]` | Model names served; empty accepts any. |
| Model aliases | `aliases` | `{}` | Client-facing name → model sent to the upstream. |
| Context windows | `context_windows` | `{}` | Model → tokens, each at least 1,024. |
| Priority | `priority` | `0` | Higher wins for new conversations. |
| Enabled | `enabled` | `true` | Disabled upstreams receive no requests. |
| Allow OpenViking tools | `allow_gateway_tools` | `true` | Whether OpenViking tools may be offered through this upstream. |
| Restore reasoning the client drops | `replay_reasoning` | `null` (follows the provider) | Puts reasoning the client dropped back on earlier replies. `null` means on for `deepseek`, `ark` and `byteplus` and off for other providers; `true` or `false` overrides the provider default. Studio saves `null` while the switch matches the provider default. |
| This key is a Coding Plan or subscription key | `coding_plan` | `false` | Rejects requests to this upstream unless allowed. |
| Allow it anyway | `allow_coding_plan` | `false` | Allows a key marked as Coding Plan. |
| Minimum cacheable prompt | `cache_min_tokens` | `1024` | Ark and ModelArk only; affects cache-eligibility reporting. |

Gateway key fields, every management operation, access rules and stable error reasons are documented in the [Management API](../api/25-gateway.md).

## Migration

1. Back up both databases and the original encryption key, then stop the old standalone process/sidecar before enabling the embedded gateway.
2. Remove `host`, `port`, `workers`, `url`, `openviking_url` and `admin_token_env` from `ov.conf.gateway`; they now fail validation. Enable the embedded gateway with `enabled: true`, or move standalone settings into a separate gateway.conf without the gateway wrapper or enabled flag.
3. Embedded clients now use server port 1933; remove old 1935 Service/Ingress routes, Compose gateway service/profile and Helm sidecar settings. Port 1935 remains the standalone default. Remove old `gateway.existingSecret`, `gateway.workers`, `gateway.port` and `gateway.resources` chart values; use the two optional Secret references above.
4. Remove the shared admin token. `OPENVIKING_GATEWAY_ADMIN_TOKEN` is **not** treated as a management key. Configure a new `management_key` explicitly if object changes should be restricted. Preserve the existing encryption key through its environment variable or restore it as encryption.key.
5. Schema v5 → v6 migrates stored bases once to preserve request URLs: bare chat/responses origins gain `/api/v3` for Ark/BytePlus and `/v1` otherwise; bare Ark/BytePlus Anthropic origins gain `/api/compatible`; trailing `/v1` is removed from Anthropic bases. New entries follow SDK URL rules, and an Anthropic base ending in `/v1` is rejected.
6. Logs gain a user column. Old rows without a user remain visible only in account-wide admin/root views. Key revocation no longer deletes user data; explicit deletion and retention still apply.

## Troubleshooting

### Common symptoms

| Symptom | Cause | What to do |
| --- | --- | --- |
| A conversation gets no recall and is not saved | An OpenViking plugin was detected, and the gateway stepped aside, leaving recall, saving and OpenViking tools to the plugin. | Expected: it keeps the same content from being added or saved twice. See [Gateway or plugin?](15-gateway.md#gateway-or-plugin). |
| Saving shows **Retrying**, then **Paused** | OpenViking rejected the save or could not be reached, often because the user's OpenViking key was regenerated. | The gateway recovers on its own once the cause is fixed; if the key is no longer valid, issue a new gateway key. See [Conversations do not appear in OpenViking](#conversations-do-not-appear-in-openviking). |
| The profile has tools on, but the conversation has none | No request that could use tools has arrived yet, OpenViking's tool list failed to load for the first one, the client does not send the full history, or the client has OpenViking tools of its own. | Check the reason in the request log (see [OpenViking tools](#openviking-tools)). `tool_name_collision` is expected when the client has the OpenViking MCP server; for other reasons, fix the cause and start a new conversation. |
| Requests go without memory | OpenViking unreachable or too slow, budget used up, nothing relevant, and so on. | See [No memory is added](#no-memory-is-added). |
| The provider rejects a long conversation as too long | The gateway does not know the model's real window and compacts too late. | See [Long conversations hit the context limit](#long-conversations-hit-the-context-limit). |

### The gateway does not start

Restore the original encryption key if existing databases have no key file, fix management keys shorter than 32 characters, and remove legacy fields listed in [Migration](#migration). Standalone startup requires `--openviking-url` (or its config/env equivalent); dev authentication requires a loopback bind. In Compose, inspect `docker compose logs openviking`.

### Studio shows a setup card

- Not enabled (404): run `openviking-server init` → **Update gateway**, or set `gateway.enabled: true` and restart the server.
- Missing/invalid OpenViking credential (401): open Connection settings and enter your current OpenViking API key.
- `trusted_mode_unsupported`: switch the server to `api_key` authentication.
- `openviking_unreachable`: check standalone `--openviking-url` and connectivity.
- Read-only upstreams: enter the gateway management key; the OpenViking root/admin API key does not unlock editing. `management_key_invalid` locks the tab again.

### Issuing a key fails

The **Issue a gateway key** dialog shows why issuing failed. The code in parentheses is the reason the management API returns.

| Message in Studio | Cause and fix |
| --- | --- |
| The server can't read this user's OpenViking key. Paste the key instead. | OpenViking Server can't read the chosen user's key, usually because it stores only key hashes (`encryption.api_key_hashing.enabled`). Choose **Paste an OpenViking key** and enter the user's own key. |
| `root_key_not_allowed` | Bind a user/admin API key. Local dev supports default/default without credentials. |
| This OpenViking key belongs to another account. | The key belongs to a different account than the one you manage in Studio. |
| OpenViking rejected this key. (`openviking_http_401`) | The key is incomplete, was regenerated, or its user was removed. Use the user's current key. |
| OpenViking couldn't tell which user this key belongs to (`openviking_identity_missing`) | OpenViking returned no user for this key. Use the key of a user in this account. |
| The gateway couldn't reach OpenViking to check this key. (`openviking_unavailable`) | The gateway cannot reach `openviking_url`. |
| OpenViking is older than the gateway requires. (`openviking_version_mismatch`) | OpenViking Server is older than `min_server_version` (0.4.16). Upgrade it. A server installed from a source checkout without version tags may report a development version such as `0.1.dev123`; install a release, or set `min_server_version` to match. |

### Clients get an error from the gateway

These rejections happen before a request reaches a provider, so they do not appear on the Requests tab.

| Status and message | Cause and fix |
| --- | --- |
| 401 "Invalid or revoked OpenViking Gateway key" | The client sent no gateway key, a revoked one, or its provider key. Check which key the client sends. |
| 403 "Claude subscription OAuth credentials are not supported" | The client sent a Claude subscription token, for example Claude Code signed in with a subscription and no `ANTHROPIC_AUTH_TOKEN` set; or the upstream holds one. Use API keys. |
| 403 "Model is not allowed by this key" | The model is not in the key's allowed models. Issue a key that allows it. |
| 404 "No allowed upstream matches this protocol and model" | No enabled upstream bound to the key speaks the API the client called and serves the requested model. Check the upstream's protocol, its models and aliases, whether it is enabled, and the key's upstreams. |
| 401 "Upstream API key is missing" | The upstream expects each client to send its own key in `X-OpenViking-Upstream-Key`, or a gateway-held upstream has no API key. |
| 403 "Coding Plan upstreams are disabled; configure a model API key" | The upstream is marked as a Coding Plan key. Use a model API key, or deliberately check **Allow this plan through the gateway**. |
| 404 "Unknown response for this key", 403 "Response upstream is no longer allowed" | A Responses lookup for a response this key did not create, or whose upstream is now disabled or no longer bound to the key. |
| 413 | The request body is larger than `max_body_bytes`, or than your proxy's own limit. |
| 426 | A WebSocket connection to the Responses API. Expected: Codex falls back to HTTP. |
| 502 "Model upstream is unavailable" | The gateway could not reach the provider, or the call exceeded `upstream_timeout_seconds`. Use **Test connection** on the upstream. |

| 504 "Hidden tool request timed out" | OpenViking tool rounds exceeded the profile's **Total time** for tools, or, when **Total time** is not set, one model request inside the tool rounds exceeded `upstream_timeout_seconds`. |

### No memory is added

Find the request on the Requests tab. If it is not there, the client is not going through the gateway, or the gateway rejected it (see the previous section). Otherwise, expand it and check the memory search result:

- **Nothing relevant found** (`empty`): normal for a new user. Memories exist only after a conversation has been committed and OpenViking has extracted them.
- **Recall is off or the budget is used up** (`disabled`): the profile has recall off, the conversation used up its budget, or the message is shorter than 3 characters.
- **OpenViking is unreachable or didn't answer in time** (`openviking_unavailable`): check the OpenViking card on the Overview tab. If OpenViking is up but slow, consider a longer **Time limit**.
- **OpenViking rejected the key** (`openviking_http_401`): the user's OpenViking key is no longer valid. Issue a new gateway key with the current OpenViking key.
- **OpenViking is older than the gateway requires** (`openviking_version_mismatch`): see [Issuing a key fails](#issuing-a-key-fails).
- **OpenViking plugin in use** (`plugin_present`): an OpenViking plugin was detected, and the gateway stepped aside for this conversation.
- **Type is not New message**: tool steps, housekeeping and sub-agent requests never search; they replay what earlier messages received.

A message whose search failed never gets memory later; the next message searches again.

### Conversations do not appear in OpenViking

- **Saving is one turn behind.** A turn appears when the next message arrives, the last turn after **Save the latest reply after** (10 minutes by default).
- **The profile does not save.** The Saving column shows Off.
- **Saving is retrying or paused.** The expanded row shows the reason. Fix the cause, for example an invalid OpenViking key; the gateway retries within 5 minutes. If the conversation stays paused, use **Resync conversation…**.
- **An OpenViking plugin is in use** (`plugin_present`); the gateway saves nothing for that conversation.
- **You are looking as another user.** Sessions belong to the OpenViking user behind the gateway key and are named `gateway-…`.
- **Memories come later than sessions.** OpenViking extracts memories only after a commit, in the background.

### The first-call cache hit rate dropped

Compare it with what the provider reaches without the gateway, then look at the issues on recent requests:

- **Memory record missing** (`missing_injection_record`): the gateway's storage was lost, restored from an old backup or expired, so memory added earlier could not be replayed. Ongoing conversations recover after one miss.
- **Upstream switched** (`upstream_changed`): conversations moved to another upstream because theirs was disabled, is not bound to the key the client now uses, or stopped serving the model.
- **Cache parameters changed** (`ark_cache_parameters_changed`): on Ark or ModelArk, the client changes the model, thinking, sampling, system prompt or tools within a conversation.
- No issue: the client itself may change earlier messages or its system prompt every turn, for example by inserting the current time. The gateway cannot fix that. Each compaction, and each new context window the model starts, also costs one miss, which is expected.

### Long conversations hit the context limit

If the provider rejects long conversations as too long for the model:

- **The model's window is unknown to the gateway.** Without an entry in the upstream's **Context windows** or a **Default context window** in the profile, the gateway assumes 1,000,000 tokens and compacts too late for smaller models. Add the model's window.
- **Compaction is off** in the conversation's profile. Profile changes reach new conversations only.
- **Compaction failed.** The request detail shows the reason, and the gateway tries again after 60 seconds. If summaries are cut off at the length limit, raise **Summary length**. `still_over_threshold` means the system prompt, tools and summary alone fill most of the window: lower **Summary length**, or check the model's window.

### File imports fail

- "Set gateway.public_url for client uploads": set `public_url` to an address the client can reach.
- The upload link cannot be reached, or returns 404: route `/gateway/uploads` to the gateway in your proxy.
- 413: the file is larger than `max_body_bytes` or the proxy's limit.
- 502 "OpenViking upload is unavailable": the gateway cannot reach OpenViking Server.
- The import tools are not offered: the OpenViking server does not provide them, the profile has them unchecked, or gateway tools are disabled for this request. Check the Tools list and request details.

### Issues on degraded requests

The **Requests** and **Overview** tabs flag requests the gateway could not fully handle. Studio shows the label; the code in parentheses is the same issue in the management API:

| Issue | What happened | What to do |
| --- | --- | --- |
| Memory unavailable (`memory_store_failure`) | The gateway could not read or write its storage, or the user's gateway data was just deleted. The request went to the model without memory; for Claude, earlier thinking was dropped. | Check disk space, permissions and the gateway's log. Expected right after deleting a user's data. |
| Forwarded unchanged (`unsafe_json`) | The request contained numbers or duplicate keys that cannot be re-encoded exactly, so it was forwarded byte for byte without memory. | Usually specific to one client; nothing to change in the gateway. |
| Upstream switched (`upstream_changed`) | The conversation's upstream could no longer serve it, so it moved to another one, with one cache miss and earlier Claude thinking dropped. | Re-enable the upstream, or have the client use a key that includes it; otherwise accept the move. |
| Memory record missing (`missing_injection_record`) | Claude conversation: memory added earlier is no longer on record, so earlier thinking was dropped once. | Expected after storage loss or for very old conversations. Start a new conversation if it repeats. |
| OpenViking plugin in use (`plugin_present`) | An OpenViking plugin was detected; gateway memory is off for this conversation. | Expected when a plugin is in use. Use either the plugin or the gateway for that client. |
| Cache parameters changed (`ark_cache_parameters_changed`) | On Ark or ModelArk, cache-relevant parameters differ from the conversation's first request; the prompt cache probably missed. | Keep model, thinking, sampling, system prompt and tools stable within a conversation. |
| Tool history unavailable (`hidden_tool_history_unavailable`) | A Claude conversation previously used OpenViking tools, but this request cannot use them, so earlier thinking has been removed. | Check the tool skip reason in the request detail, restore the original settings or start a new conversation. |
| OpenViking tools failed (`hidden_tool_loop_failed`) | The model could not finish answering while using OpenViking tools, and the client received an error. | See [OpenViking tools](#openviking-tools). |
| Reply not saved (`capture_parse_failure`) | A streamed reply could not be parsed, so this reply was not saved. The client was not affected. | Nothing; it is informational. |
