---
description: Deploy Context Gateway with Docker Compose, Helm or your own proxy, manage it in Studio, and troubleshoot it.
---

# Context Gateway deployment and operations

This page is for the people who run Context Gateway: deploying it, managing upstreams, context profiles and keys in Studio, keeping data safe, and fixing problems. For what the gateway does and how to connect clients, see [Context Gateway](15-context-gateway.md).

Context Gateway is currently in beta. Its settings and APIs may change between releases, so read the release notes before you upgrade.

The gateway is a separate process next to OpenViking Server, normally on port 1935. Clients send their model requests to it. You manage it in Studio, which OpenViking Server serves; Studio's management calls reach the gateway through OpenViking Server.

```text
                       clients
                          |
                          v
          reverse proxy (https://ov.example.com)
             |                                  |
             | /v1/*                            | everything else:
             | /api/v3/*                        | /studio, /api/v1,
             | /api/compatible/v1/*             | /mcp, /health, ...
             | /context-gateway/uploads         |
             v                                  v
    Context Gateway :1935  <--- management ---  OpenViking Server :1933
       |          |                                     ^
       |          +-------- search, save, tools --------+
       v
    model providers (upstreams)
```

## Requirements

- **OpenViking Server 0.4.16 or later, in API key mode** (`server.auth_mode: "api_key"` with a `root_api_key`). The gateway acts for each person with that person's OpenViking key. In dev mode every key acts as root, which the gateway refuses, so no gateway key can be issued.
- **The gateway itself**: the `openviking[context-gateway]` extra on Python 3.10 or later, or the official OpenViking Docker image, which already contains the `openviking-context-gateway` command.
- **An account admin key** for Studio. Everything you configure belongs to the account of the key you sign in with. A root key also works but manages whichever account it resolves to, so prefer the account admin's key.
- **An OpenViking user for each person** who gets a gateway key, in the same account. Users and admins both work; root does not. When you issue the key, OpenViking Server reads the user's OpenViking key itself; if it stores only key hashes, the user has to provide their key.
- **API keys for your model providers.** Subscription logins are rejected, and Coding Plan keys are refused unless you explicitly allow them.
- **A local disk on one host** for the gateway's storage (`~/.openviking/context-gateway` by default). Network file systems and storage shared between hosts are not supported.
- **Two secrets**, described next.

## Secrets

| Secret | Environment variable | Needed by | What it does |
| --- | --- | --- | --- |
| Encryption key | `OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY` | Gateway | Encrypts everything the gateway stores: upstream API keys and headers, the OpenViking keys bound to gateway keys, conversation state and request logs. Must be a Fernet key: 32 random bytes in URL-safe base64. |
| Admin token | `OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN` | Gateway and OpenViking Server | Authenticates OpenViking Server to the gateway's management API when you work in Studio. At least 32 characters. |

Generate each once:

```bash
# Encryption key (the same format as Fernet.generate_key() in the cryptography package)
python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'

# Admin token
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Keep them in your secret store or deployment environment, not in `ov.conf` or source control. To read them from differently named variables, set `encryption_key_env` and `admin_token_env` in the `context_gateway` section.

- **The encryption key cannot be changed.** Nothing re-encrypts stored data, so with a different key the gateway cannot read what it stored. Back the key up together with the gateway's storage. If it is lost, point `storage_path` at an empty directory and set up upstreams, profiles and keys again.
- **The admin token can be rotated.** Set the same new value for both processes and restart both. It only protects the management API and encrypts nothing.
- Without valid secrets the gateway stops at startup. OpenViking Server starts without the admin token, but Studio's Context Gateway page then reports that the management token is not configured.

## Deploy

Both processes read the `context_gateway` section of the same `ov.conf`. OpenViking Server uses it to find the gateway for Studio and for data deletion; the gateway uses all of it. A few rules apply to this section:

- The gateway reads the file given by `--config`, else `OPENVIKING_CONFIG_FILE`, else `~/.openviking/ov.conf`. Unlike OpenViking Server, it does not fall back to `/etc/openviking/ov.conf`.
- The gateway does not expand `$VAR` or `${VAR}` placeholders. Write literal values in `context_gateway`, and keep the secrets in the environment variables above.
- Unknown keys are errors. OpenViking Server reports `Unknown config field 'context_gateway.<name>'`, and the gateway refuses to start.
- Restart OpenViking Server after you change the section.

Three addresses in this section are easy to mix up:

| Setting | Who uses it | Typical value |
| --- | --- | --- |
| `url` | OpenViking Server, to reach the gateway's management API | `http://127.0.0.1:1935`; `http://context-gateway:1935` in Docker Compose |
| `openviking_url` | The gateway, to reach OpenViking Server | `http://127.0.0.1:1933`; `http://openviking:1933` in Docker Compose |
| `public_url` | Clients. Studio shows it in its setup instructions, and OpenViking tools use it for upload links. | `https://ov.example.com` |

### Routes

Serve OpenViking Server and the gateway from one public HTTPS origin and let the reverse proxy route by path:

| Path | Goes to | Used for |
| --- | --- | --- |
| `/v1/*` | Gateway | Anthropic Messages, Chat Completions, Responses and the model list |
| `/api/v3/*` | Gateway | Ark paths (Volcano Engine Ark and BytePlus ModelArk) for Chat Completions, Responses and the model list |
| `/api/compatible/v1/*` | Gateway | Ark's Anthropic-compatible path |
| `/context-gateway/uploads` | Gateway | One-time file uploads from OpenViking tools |
| Everything else | OpenViking Server | Studio, REST API, MCP, OAuth, `/health` |

With this layout, `public_url` is the public origin, for example `https://ov.example.com`. Anthropic clients use that address; OpenAI-style clients add `/v1`.

> **Security**: Never route the gateway's `/admin/*` paths, or its whole port, from the public entry point. The gateway's management API accepts the admin token for any account. Studio reaches it through OpenViking Server, which checks that you are an account admin and limits every call to your own account. Forward exactly the four path groups above, and keep ports 1933 and 1935 private behind the proxy.

Whatever proxy you use, configure the gateway paths for long, streamed model calls:

- **No response buffering**, so streamed replies reach clients as they are generated.
- **Request bodies of at least 32 MiB** (`max_body_bytes`). Long conversations with tool output get large, and nginx's default of 1 MB breaks them.
- **A read timeout of at least 600 seconds** (`upstream_timeout_seconds`).
- **No query strings in access logs for `/context-gateway/uploads`**, because they carry one-time upload tokens. The gateway itself writes no access log.

### Docker Compose

The bundled `docker-compose.yml` defines a `context-gateway` service under the `context-gateway` profile. It runs the same image with the same `~/.openviking` mount and `ov.conf` as OpenViking Server. Its port 1935 is reachable only inside the Compose network; clients reach it through the bundled Caddy, which already routes the gateway paths on port 1934.

1. **Put the secrets in `.env`** next to `docker-compose.yml`:

   ```dotenv
   OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY=<encryption-key>
   OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN=<admin-token>
   ```

   Compose passes the admin token to both containers and the encryption key only to the gateway. If either is empty, the gateway exits and Compose restarts it in a loop.

2. **Merge these settings into `~/.openviking/ov.conf`:**

   ```json
   {
     "server": {
       "auth_mode": "api_key",
       "root_api_key": "<root-key>"
     },
     "context_gateway": {
       "enabled": true,
       "host": "0.0.0.0",
       "url": "http://context-gateway:1935",
       "openviking_url": "http://openviking:1933",
       "public_url": "http://<your-host>:1934"
     }
   }
   ```

   Use `http://<your-host>:1934` while clients reach the bundled Caddy port, or your HTTPS origin once you add one. `storage_path` can keep its default: inside the container it resolves to `/app/.openviking/context-gateway`, which lies in the mounted volume.

3. **Start the stack with the profile:**

   ```bash
   docker compose --profile context-gateway up -d
   ```

   Include `--profile context-gateway` in later `up` commands as well. After you edit `ov.conf`, restart both containers with `docker compose --profile context-gateway restart openviking context-gateway`.

4. **Check the routing.** A request without a key should reach the gateway and be refused:

   ```bash
   curl -s http://127.0.0.1:1934/v1/models
   # {"detail":"Invalid or revoked Context Gateway key"}
   ```

   Then open Studio and check that the Overview tab shows OpenViking as connected. `docker compose logs context-gateway` shows startup errors.

If the gateway starts before OpenViking Server is ready, it runs without memory until its next check of OpenViking, at most 60 seconds later (`health_interval_seconds`). Messages sent in that window reach the model without memory.

**HTTPS.** Follow [Option A in Public Access](12-public-access.md#adding-https-for-public-access) to set `OPENVIKING_PUBLIC_BASE_URL`, open ports 80 and 443 and add the Caddy volumes, but use this domain block in `Caddyfile` so the gateway paths reach the gateway:

```caddyfile
{$OPENVIKING_PUBLIC_BASE_URL} {
    @context_gateway path /v1/* /api/v3/* /api/compatible/v1/* /context-gateway/uploads
    handle @context_gateway {
        reverse_proxy context-gateway:1935 {
            flush_interval -1
        }
    }
    handle {
        reverse_proxy openviking:{$OPENVIKING_SERVER_PORT:1933}
    }
}
```

Then set `context_gateway.public_url` to the same origin, written out literally (for example `https://ov.example.com`), and restart.

### Helm

The chart runs the gateway as a second container in the OpenViking pod, sharing its volume and `ov.conf`.

1. **Create the Secret.** The key names `encryption-key` and `admin-token` are fixed:

   ```bash
   kubectl create secret generic openviking-context-gateway \
     --from-literal=encryption-key="$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')" \
     --from-literal=admin-token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
   ```

2. **Enable the gateway in your values.** The NGINX ingress annotations below are recommendations for streaming and large requests; the chart does not set them:

   ```yaml
   contextGateway:
     enabled: true
     existingSecret: openviking-context-gateway
     workers: 2

   config:
     server:
       auth_mode: api_key
       # Pass root_api_key from a Secret as described in the chart README.
     context_gateway:
       public_url: https://ov.example.com

   ingress:
     enabled: true
     className: nginx
     annotations:
       nginx.ingress.kubernetes.io/proxy-buffering: "off"
       nginx.ingress.kubernetes.io/proxy-request-buffering: "off"
       nginx.ingress.kubernetes.io/proxy-body-size: 32m
       nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
       nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
     hosts:
       - host: ov.example.com
         paths:
           - path: /
             pathType: Prefix
     tls:
       - secretName: ov-example-com-tls
         hosts:
           - ov.example.com
   ```

What the chart does when `contextGateway.enabled` is true:

- It generates the rest of the `context_gateway` section: `enabled`, `host: 0.0.0.0`, `port`, `workers`, `url: http://127.0.0.1:<port>`, `openviking_url: http://127.0.0.1:<config.server.port>` and `storage_path: <persistence.mountPath>/context-gateway`. Values you put under `config.context_gateway` take precedence. Set `public_url` there, as a literal value.
- It passes the admin token to both containers and the encryption key to the gateway container. The gateway container does not receive `extraEnv`.
- It adds a `context-gateway` port to the Service. When the ingress is enabled, it routes `/v1`, `/api/v3`, `/api/compatible/v1` and `/context-gateway/uploads` to that port, ahead of your own paths.
- It adds a readiness probe on the gateway's `/health`. `contextGateway.resources` sets the gateway container's requests and limits.

Keep `replicaCount: 1`. The gateway's storage lives on the ReadWriteOnce volume and must be used from a single host; scale with `contextGateway.workers` instead. For passing the root key and model keys from Secrets, see the [chart README](https://github.com/volcengine/OpenViking/blob/main/deploy/helm/README.md).

### Your own reverse proxy

With nginx on the same host as both processes, route the gateway paths before the catch-all location:

```nginx
server {
    listen 443 ssl;
    http2 on;  # nginx < 1.25.1: remove this line and use `listen 443 ssl http2;`
    server_name ov.example.com;

    ssl_certificate     /etc/letsencrypt/live/ov.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ov.example.com/privkey.pem;

    # Model APIs: streamed replies, large histories, slow models.
    location ~ ^/(v1|api/v3|api/compatible/v1)/ {
        proxy_pass http://127.0.0.1:1935;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_request_buffering off;
        client_max_body_size 32m;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_set_header Host              $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # One-time file uploads. The query string carries the upload token.
    location = /context-gateway/uploads {
        proxy_pass http://127.0.0.1:1935;
        proxy_http_version 1.1;
        proxy_request_buffering off;
        client_max_body_size 32m;
        proxy_read_timeout 120s;
        access_log off;
    }

    # Studio, REST API, MCP and everything else.
    location / {
        proxy_pass http://127.0.0.1:1933;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_read_timeout 300s;
        proxy_set_header Host              $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host  $host;
    }
}
```

If you raise `max_body_bytes` or `upstream_timeout_seconds`, raise `client_max_body_size` and the timeouts with them. For other proxies, apply the same four path groups and the settings listed under [Routes](#routes).

### Scaling

- **Workers.** `workers` (1–64; 1 by default, 2 in the Helm chart) sets how many gateway processes run on the host. They share the storage files, and each one also saves conversations in the background.
- **One host.** Running the gateway on several hosts, or on a network file system, is not supported. With Helm, keep one replica.
- **Change propagation.** Revoking a key or editing an upstream reaches the other worker processes within about 2 seconds.
- **Health per process.** Each worker checks OpenViking on its own, so the OpenViking status on the Overview tab comes from whichever process answered.
- **No load balancing or rate limiting.** Each conversation stays on one upstream, and the gateway neither retries on another upstream nor throttles requests; provider 429 responses reach the client unchanged. If you need balancing, failover or quotas, run LiteLLM or new-api behind the gateway as an upstream.

## Manage the gateway in Studio

Open Studio at `/studio` on OpenViking Server, open **Connection Settings** and enter an account admin key as the **Admin API key**. **Context Gateway** then appears in the sidebar's **Settings** group; only account admins and root see it. Everything on these pages belongs to the account of that key. The page header shows the gateway address clients should use: `public_url`, or `url` while `public_url` is empty, in which case the **Connect** tab warns that clients may not reach it.

If the gateway is not set up or cannot be reached, the page shows a setup card naming the setting to fix; see [Studio shows a setup card](#studio-shows-a-setup-card). Otherwise it has six tabs: **Overview**, **Upstreams**, **Profiles**, **Keys**, **Requests** and **Connect**. A new setup goes in that order: add an upstream, create a context profile, issue a gateway key, then connect a client using the Connect tab.

### Upstreams

An upstream is one model provider endpoint that speaks one API: Anthropic Messages, Chat Completions or Responses. The gateway never converts between them, so add one upstream for each API your clients use, even when they share a provider: Anthropic Messages for Claude Code, Responses for Codex CLI, Chat Completions for most chat apps and SDKs.

Each provider offers these protocols, and the editor fills in the provider's default Base URL when you choose it:

| Provider | Protocols | Default Base URL |
| --- | --- | --- |
| Generic | All three | None; enter the address yourself |
| Anthropic | Anthropic Messages | `https://api.anthropic.com` |
| OpenAI | Chat Completions, Responses | `https://api.openai.com/v1` |
| DeepSeek | All three | `https://api.deepseek.com`; `https://api.deepseek.com/anthropic` for Anthropic Messages |
| Volcano Engine Ark | All three | `https://ark.cn-beijing.volces.com` |
| BytePlus ModelArk | All three | `https://ark.ap-southeast.bytepluses.com` |

The upstream editor has four sections.

**Endpoint.**

- **Provider** decides which protocols you can pick and adjusts the gateway to the provider's quirks. The editor offers only the protocols the provider supports, as listed in the table above. Choose *Generic* for a provider not listed here, or for a compatible proxy such as LiteLLM or new-api.
  - *Generic*, *Anthropic* and *OpenAI* forward requests the same way.
  - *DeepSeek*: OpenViking tools are offered only when a request turns thinking off (`"thinking": {"type": "disabled"}`), because DeepSeek's thinking mode needs reasoning history that chat apps do not send back. Standard Responses requests have no `thinking` field, so they do not get OpenViking tools. Memory is added as usual.
  - *Volcano Engine Ark* and *BytePlus ModelArk*, Ark's international edition, work the same way: requests go to Ark's own paths. Each conversation gets a stable `prompt_cache_key` for Chat Completions and Responses, so Ark's prefix cache follows the conversation. Requests whose model, thinking, sampling, system prompt or tools differ from the conversation's first request are flagged **Cache parameters changed** (`ark_cache_parameters_changed`), because Ark resets its cache when these change.
- **Protocol** decides which client requests the upstream can serve, and how the gateway sends the key: `x-api-key` for Anthropic Messages, `Authorization: Bearer` for the others.
- **Base URL.** Choosing a provider fills in its default address from the table above. Switching the provider or protocol replaces the address only while it is empty or still the previous default, so an address you entered is kept; when it differs from the default, **Use default** under the field puts the default back. The gateway appends the client's request path and drops a duplicate `/v1`, so `https://api.openai.com/v1` and `https://api.anthropic.com` both work. The editor shows where requests will go. For Ark and ModelArk, use the origin without a path, such as `https://ark.cn-beijing.volces.com`, which works for every protocol; a base ending in `/api/v3` works only for Chat Completions and Responses, and one ending in `/api/compatible/v1` only for Anthropic Messages. A provider whose API path does not end in `/v1`, such as `…/api/paas/v4`, cannot be used directly; put a compatible proxy such as LiteLLM in between.

**Credentials.**

- **Who provides the API key**:
  - **The gateway holds the API key** (the default): the gateway sends the upstream's key, and clients need only a gateway key.
  - **Each client sends its own key**: every request must also carry the client's own provider key in an `X-OpenViking-Upstream-Key` header, next to its gateway key. Requests without it fail with 401 "Upstream API key is missing". The gateway passes the key to the provider in the provider's usual header and never forwards `X-OpenViking-*` headers.
- **API key** and **Extra headers** are write-only. When you edit an upstream, leave the API key blank to keep the stored one. Extra headers show their names only: leave a value blank to keep it, or remove the row to delete the header. A stored API key cannot be cleared; to stop using it, switch to "Each client sends its own key" or delete the upstream.
- **Extra headers** are added to every request to this upstream, such as a provider's version or organization header, and replace a client header of the same name. `Host`, `Content-Length`, `Transfer-Encoding` and `Connection` are not allowed.
- **This key is a Coding Plan or subscription key.** Providers restrict Coding Plan and similar subscription keys to their own coding tools, and using one through a shared gateway can get the subscription suspended. When you check this box, every request routed to the upstream is rejected with 403 "Coding Plan upstreams are disabled; configure a model API key", unless you also check **Allow it anyway**. The gateway cannot tell such keys apart from ordinary ones, so mark them yourself. Claude subscription tokens (`sk-ant-oat…`) are always rejected.

**Models.**

- **Models** lists the model names this upstream serves. Leave it empty to accept any name. Only listed names and aliases appear in the model list clients can request (`/v1/models`).
- **Model aliases** map a name clients use to the model sent to the upstream, for example the Claude model names Claude Code asks for to the model your provider serves. Aliases appear in the model list, and request logs show the name the client used.
- **Context windows** maps a model to its context window in tokens (at least 1,024), keyed by the name sent to the upstream after aliases. It lets long conversations wait briefly for a summary instead of overflowing; see [Long conversations](#long-conversations).

**Routing.**

- **Priority.** For each request, the gateway considers the enabled upstreams bound to the key that speak the request's API and serve its model, and picks the highest priority. A conversation stays on the upstream it started with as long as that upstream can still serve it, so priority changes affect new conversations.
- **Enabled.** A disabled upstream receives no requests and disappears from the model list. Conversations that were using it move to the next candidate, which costs one provider cache miss and drops earlier Claude thinking blocks; these requests are flagged **Upstream switched** (`upstream_changed`). Disabling is the reversible alternative to deleting.
- **Allow OpenViking tools** controls whether OpenViking tools may be offered through this upstream. Turning it off takes effect at once, also in conversations that already have the tools.
- **Minimum cacheable prompt** (Ark and ModelArk only) is the shortest prompt the model caches. It only affects how the request log reports cache eligibility.

Every other upstream setting takes effect at once, in existing conversations too. There is no failover: when the chosen upstream fails, the client receives the error, and an unreachable provider returns 502 "Model upstream is unavailable".

**Test** on the list and **Test connection** in the editor request the provider's model list (`/v1/models`, or `/api/v3/models` on Ark and ModelArk) with the upstream's credentials and headers, waiting up to 10 seconds. A pass means the host is reachable and the key is accepted there. The test does not check model names or the Coding Plan setting. A provider without a model-list endpoint, or one that needs extra headers on it, can fail the test while real requests work. Upstreams where each client sends its own key cannot be tested.

An upstream that a key uses cannot be deleted; revoke those keys first, or disable the upstream.

### Context profiles

A context profile is a named set of memory settings. Each gateway key uses one profile, and one profile can serve many keys. You might keep a "Coding" profile that recalls memories, resources and skills, and a "Chat" profile that recalls only memories with a smaller budget.

The **Profiles** tab shows one card per profile with a summary of its four sections and the number of keys using it. While the tab is empty, **Create with recommended settings** creates a profile named "Default" in one click and **Customize** opens the editor; after that, **New profile** opens the editor. A profile that a key uses cannot be deleted; **Duplicate** is a quick way to start a variant.

> **Note**: Saved changes apply to conversations that start afterwards. A conversation takes a snapshot of its key's profile at its first request and keeps it, because changing what the model sees in the middle of a conversation would break the provider cache. Clients that open a new conversation for every chat pick up changes quickly. Claude Code, Codex and clients that send a stable session header keep the old settings until they start a new conversation, or until the conversation has gone unused for 30 days (`session_ttl_days`).

Each section has its own switch:

| Section | What it does | Main settings and defaults |
| --- | --- | --- |
| Recall memory | Searches OpenViking with each new user message and appends what is relevant to that message. | Sources: memories, resources, skills · Budget per message: 1,600 tokens · Budget per conversation: 6,000 tokens · Relevance threshold: 0.35 · Time limit: 2 s |
| Save conversations | Writes finished turns to an OpenViking session and commits it, so OpenViking can extract memories. | Save the latest reply after: 600 s · Commit after: 20,000 tokens · Keep recent messages: 10 |
| Long conversations | Replaces older history with OpenViking's summary once a conversation is long. Needs Save conversations. | Start after: 30,000 tokens · Keep recent turns: 3 · Wait for summary: 30 s |
| OpenViking tools | Lets the model search and read OpenViking while it answers. Chat Completions, full-history Responses and Anthropic Messages. Off by default. | Tools: Search, Read, List · Allow write tools: off |

How the recall settings play out:

- The search query is the user message with client noise removed, cut to **Query length** characters (8,000). Messages shorter than 3 characters are not searched.
- Each message gets at most the smaller of **Budget per message** and what is left of **Budget per conversation**; with less than 64 tokens left, the gateway skips the search. A budget per conversation of 0 turns recall off. Token counts are a conservative estimate, not the provider's billing count.
- If OpenViking does not answer within the **Time limit**, the message goes to the model without memory and never gets it later, even when the client retries.
- **Limit by category** (under **Advanced settings**) searches only the categories with a limit above 0, among events, entities, preferences, experiences, resources and skills, and takes at most that many entries from each. When it is off, all sources are ranked together.

How the saving settings play out:

- A turn is saved when the next user message arrives. **Save the latest reply after** sets how long the gateway waits before it also saves the last turn and commits the session, so short conversations get committed too.
- **Commit after** commits the session whenever that many tokens are waiting in it. OpenViking extracts memories at each commit, in the background.
- **Keep recent messages** keeps the newest whole turns, at least this many messages, in the session after a commit.
- Recall and saving are independent: a profile that only saves, or only recalls, is valid.

The [Configuration reference](#configuration-reference) lists every setting with its API name and limits.

### Gateway keys

A gateway key (`ovcg_…`) is what a client uses in place of a provider key. Each key belongs to one OpenViking user and uses one context profile and a set of upstreams.

To issue one, choose **Issue key** on the **Keys** tab and fill in:

- **Name**, to recognize the key later, for example "alice · Claude Code".
- **OpenViking user**: a user or admin in this account. The gateway searches and saves memory as this user. When you issue the key, OpenViking Server reads this user's OpenViking key and hands it to the gateway, which stores it encrypted; the key never passes through the browser and is never shown. If OpenViking Server stores only key hashes (`encryption.api_key_hashing.enabled`), or cannot read a user's key for another reason, that user can't be chosen in the list. Choose **Paste an OpenViking key** instead and enter the user's own key. When Studio is signed in with a root key, there is no list to choose from: a root key may resolve to another account than the one Studio shows, so paste the user's key. A pasted key can't be a root key.
- **Context profile**.
- **Upstreams**: at least one. The key can reach only these.
- **Allowed models** (optional): leave empty to allow every model these upstreams serve. The names are the ones clients send, including aliases. Requests for other models fail with 403 "Model is not allowed by this key", and the model list shows only allowed names.

Before issuing, the gateway checks the user's OpenViking key with OpenViking. A pasted root key, a key from another account or an invalid key, or an unreachable OpenViking, stops the issue; see [Issuing a key fails](#issuing-a-key-fails).

The **Copy your gateway key** dialog then shows the full key. This is the only time it is shown: the gateway keeps only a hash and the first characters. The dialog also has ready-to-paste setups for Claude Code, Codex CLI and chat clients with the real key and gateway address filled in. If a key is lost, issue a new one.

- **Keys cannot be edited.** To change a key's profile, upstreams, allowed models or OpenViking user, issue a new key and revoke the old one.
- **Conversations belong to the user, not the key.** A new key for the same OpenViking user continues the same conversations when the client sends the same session ID, so replacing a key is seamless.
- **Revoking** cuts off clients at once (other worker processes follow within about 2 seconds); requests already in flight finish. Turns not yet saved for conversations last used with that key are dropped. Conversations and memories stay. To replace a key without losing turns, switch the client to the new key first, then revoke the old one.
- **When an OpenViking key changes.** The gateway keeps the user's OpenViking key from the time the key was issued. If that key is regenerated or removed, the gateway key keeps authenticating clients, but memory search fails and saving pauses with `openviking_http_401`. Issue a new gateway key for the user (Studio reads their new OpenViking key), switch the client, and revoke the old gateway key.
- **One key per person.** Everyone using a key shares the memory of its OpenViking user. Issue one key per person, and per client when you want different profiles.

**Delete this user's gateway data…** (in a key's **More actions** menu) revokes every gateway key of that OpenViking user and deletes the gateway's conversation state for them. Sessions and memories in OpenViking are not affected; delete those in OpenViking. Removing a user or account in OpenViking does this automatically; see [Security and data](#security-and-data).

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

The **Requests** tab lists the latest 1,000 records, newest first, and refreshes its first page every 30 seconds while you view it. Filter by **All**, **New messages**, **Tool steps** or **Issues**, by **Request type**, or search by model, conversation or key. Columns show the type, model (the name the client sent), the provider's HTTP status, tokens (input, cached share, output), **Memory** (`+3`: entries this request's search added, with the search time; `↺ 4`: memory added to 4 earlier messages, sent again unchanged), **Saving** status and **Issue**. Expand a row for the conversation ID, upstream, key, duration, the token breakdown, the memory search result, saving status with the next retry time, OpenViking tool details and, for an issue, what it means and what to do.

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

With OpenViking tools on, the model can search and read the user's accessible OpenViking data while answering. The gateway runs the tools, and the client continues receiving the answer until the reply ends. Reported token usage includes all model calls made during that reply.

Tools support streaming and nonstreaming requests in all three protocols:

| Protocol | Client requirements |
| --- | --- |
| Chat Completions | Send the full `messages` history. |
| Responses | Use `store: false` and full `input` history; pass the entire returned `output` array back on the next request, including reasoning and client tool calls. Requests with `previous_response_id`, `conversation` or `background` continue to pass through without gateway tools. `item_reference` cannot reconstruct full history. |
| Anthropic Messages | Send the full `messages` history and keep thinking/signature blocks intact. Token-count requests get the same tool definitions but do not execute tools. |

If a reply calls both OpenViking and client tools, the gateway handles the OpenViking calls. The client runs its own tools and returns their results as usual. Responses supports client function and custom tools; Anthropic Messages supports client tool_use calls.

This is meant for chat apps and API apps that cannot connect to OpenViking's MCP server. Clients that support MCP or have a plugin, such as Claude Code and Codex, are better served by those, where tool calls are visible.

To turn the tools on, enable the **OpenViking tools** section in a context profile and check the tools under **Tools**: **Search**, **Read** and **List**. **Allow OpenViking tools** is on by default on every upstream. Conversations that start after you save get the tools. The model sees them as `openviking_search`, `openviking_read` and `openviking_list`, and is told at the start of the conversation that these tools come from the gateway and are not in the client's tool list. The write tools, **Write** (`openviking_write`), **Import files** (`openviking_add_resource`) and **Import skills** (`openviking_add_skill`), are offered only when **Allow write tools** is on and the tool is checked.

> **Note**: Tool calls run inside the gateway, so they never go through the client's permission prompts. A write tool changes the user's OpenViking data without asking. Allow write tools only for users who expect that.

Whether a conversation gets tools is decided at its first request and kept, so profile changes reach new conversations only. When a request runs without tools, its detail on the Requests tab says why:

| Reason | Cause |
| --- | --- |
| `tools_require_full_history` | Responses tools need full input history with `store: false`, without `item_reference`. |
| `upstream_tools_disabled` | The upstream has **Allow OpenViking tools** off. |
| `tools_multiple_choices` | The request asks for several choices (`n` greater than 1). |
| `tools_structured_output` | The request asks for structured output (`response_format`, `text.format` or `output_config.format`). |
| `tools_non_function` | The Chat Completions client sent a tool of a type other than function. |
| `tools_forced_choice` | `tool_choice` is `required` or names a specific tool. |
| `deepseek_reasoning_history_required` | The upstream's provider is DeepSeek and the request does not turn thinking off (`"thinking": {"type": "disabled"}`). This includes standard Responses requests, which have no `thinking` field. |
| `tool_name_collision` | The client defines a tool with one of the `openviking_*` names. |
| `tools_not_selected_at_session_start` | The profile has tools on, but the conversation's first request could not use them (for one of the reasons above), so the conversation has none. |

Housekeeping and sub-agent requests keep the tool definitions but cannot call them, and conversations where an OpenViking plugin was detected get no tools.

**Importing files and skills.** The import tools need a way to get the file to OpenViking, so they are offered only when the conversation's first request has one:

- **A shell tool** (a client tool named like bash, shell, exec_command, terminal or run_command): the import tool returns a one-time upload link, and the model uploads the file with the client's own shell tool, which does go through the client's permission prompt. Skill directories are zipped first. The link points to `public_url` plus `/context-gateway/uploads`, so `public_url` must be an address the client can reach and your proxy must route that path to the gateway. Without `public_url`, imports fail with "Set context_gateway.public_url for client uploads".
- **An attached file**: without a shell tool, the model can import a file attached to a user message, sent either as base64 file data or as attachment text. This includes Responses `input_file` and Anthropic `document.source` with embedded `base64` or `text` data. Open WebUI usually sends only the extracted text, which is imported as a text file. Remote URLs and provider file IDs are not fetched.

Uploads are limited to `max_body_bytes` (32 MiB by default).

The limits sit under **Advanced settings** in the profile's **OpenViking tools** section:

| Setting | API name | Default | Range | At the limit |
| --- | --- | --- | --- | --- |
| Rounds per request | `tool_max_rounds` | 5 | 1–20 | The model must answer without further tool calls. |
| Time limit per call | `tool_timeout_seconds` | 30 s | up to 120 s | The call returns an error to the model. |
| Result size | `tool_result_bytes` | 65,536 bytes | 1,024–1,048,576 | The result is truncated. |
| Total time | `tool_total_seconds` | 120 s | up to 600 s | The request fails with 504 "Hidden tool request timed out". |
| Token budget | `tool_total_tokens` | 100,000 | 1,024–1,000,000 | Further tool calls are refused and the model answers with what it has. This is not a billing cap; the final answer can exceed it. |

The token budget covers the extra calls, results and subsequent model output from using OpenViking tools. Existing conversation history, tool definitions and images do not count. The gateway preserves the client's answer-length limit (`max_tokens`, `max_completion_tokens` or `max_output_tokens`). Once the budget is spent, the model finishes its answer using the results already available.

When an individual tool call fails, the model receives an error result and can continue answering. If the reply itself fails, for example because the provider rejects a follow-up model request, the client receives an error and the request log shows **OpenViking tools failed** (`hidden_tool_loop_failed`). Streaming Responses requests end with `response.failed`; Chat Completions and Anthropic Messages report `gateway_tool_error`.

If an upstream keeps failing, turn off its **Allow OpenViking tools** setting. Claude conversations that have already used these tools may also show **Tool history unavailable** (`hidden_tool_history_unavailable`): earlier thinking can no longer be used and has been removed. Check the tool skip reason in the request detail, then restore the original settings or start a new conversation. A retried tool call reuses the first call's result, and an interrupted write is never repeated automatically.

## Long conversations

With **Long conversations** on, a conversation can keep going beyond the model's context window:

1. Saving writes the conversation to an OpenViking session.
2. Once the saved content reaches **Start after** (30,000 tokens), the gateway commits everything except the latest turns. OpenViking archives that part and writes a summary in the background. From then on, a commit happens each time another **Start after** worth of tokens is waiting.
3. When the summary is ready, every later request carries three things: the system and developer messages; one user message that starts with `[OpenViking Session Context]`, says that the gateway replaced the earlier part of the conversation, holds the summary and ends with the gateway's opening note; and the latest **Keep recent turns** turns (3) word for word, counting the current one. Memory for the new message is still added.
4. Each new archive builds on the previous summary and replaces it.

The prompt changes once at each archive point, which costs one provider cache miss there. Thinking blocks are removed from the kept turns, because their signatures covered the history that was replaced; a client's own compaction has the same effect. Long conversations need **Save conversations**. The summary replaces the older history in every request of the conversation that resends it, including housekeeping requests such as a client's own compaction; sub-agents usually send their own, separate history, which stays untouched.

**Waiting for a summary.** Until a summary is ready, the gateway sends the full history and does not wait. It waits only when the conversation is about to overflow: the model's context window is known, and the last request to the same upstream and model used at least 90% of it. Then it waits up to **Wait for summary** (30 seconds, at most 60), checking every 5 seconds. If no summary arrives in time, it sends the full history anyway, which may fail with a context-length error, and flags the request **Summary not ready** (`archive_wait_timeout`).

- The context window comes from the upstream's **Context windows** entry for the model (the name sent to the upstream), else from the profile's **Default context window**. Without either, the gateway never waits.
- The usage figure comes from the provider's response. Streamed Chat Completions report it only when the client requests it with `stream_options.include_usage`.
- If **Summary not ready** shows up often, raise **Wait for summary**, lower **Start after** so summaries are ready earlier, or check that OpenViking's VLM writes summaries promptly. The gateway gives up on a summary that takes longer than 15 minutes.

Clients still count tokens by their own history and may compact on their own, even though the gateway passes back the provider's smaller usage after a summary. When a client compacts or edits its history, the gateway saves the new history to a new OpenViking session and starts counting again.

## Security and data

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

**Deleting data.**

- Delete one user's gateway data from Studio (a key's **More actions** menu → **Delete this user's gateway data…**) or with `DELETE /api/v1/admin/context-gateway/users/{user_id}/data` on OpenViking Server, using an account admin key. This revokes the user's gateway keys and deletes their conversation state. Request-log metadata ages out with retention.
- Removing a user in OpenViking does the same automatically. Removing an account also deletes the account's upstreams, profiles, keys and request logs from the gateway. If the gateway is unreachable at that moment, OpenViking keeps retrying. If you stop using the gateway, set `context_gateway.enabled` to `false` so deletions no longer wait for it.
- Sessions and memories in OpenViking are deleted through OpenViking, not the gateway.

**Access.**

- Clients authenticate with gateway keys, sent as `Authorization: Bearer` or `x-api-key`. Keys can be revoked at any time.
- The gateway uses each user's own OpenViking key for memory search, saving and tools, so it can do only what that user can do.
- Only OpenViking Server should hold the admin token, and the gateway's `/admin/*` paths must stay private (see [Routes](#routes)). Studio users are checked by OpenViking Server: account admins and root only, each limited to their own account.
- `/context-gateway/uploads` needs no gateway key. A one-time token signed by OpenViking authorizes each upload, and the gateway forwards it only to OpenViking's upload endpoint. Keep its query string out of proxy logs.
- `X-OpenViking-*` headers, client credentials and cookies are never forwarded to upstreams.

**Backups.** Back up `storage_path`, including both databases with their `-wal` and `-shm` files, together with the encryption key. For a consistent copy, stop the gateway or use SQLite's online backup (`sqlite3 kernel.sqlite3 ".backup kernel.backup.sqlite3"`, and the same for `management.sqlite3`). What a loss means:

- **`kernel.sqlite3` lost**: memory added earlier cannot be replayed. Each ongoing conversation pays one provider cache miss, Claude conversations lose earlier thinking once (**Memory record missing**, `missing_injection_record`), and turns not yet saved are lost.
- **`management.sqlite3` lost**: upstreams, profiles, keys and request logs are gone. Set them up again and issue new keys.
- **Encryption key lost**: neither database can be read. Start with an empty `storage_path`.

## Configuration reference

The `context_gateway` section of `ov.conf`, with every key at its default:

```jsonc
{
  "context_gateway": {
    "enabled": false,                                  // the gateway refuses to start until true
    "host": "127.0.0.1",                               // bind address
    "port": 1935,
    "workers": 1,                                      // 1–64 processes on this host
    "url": "http://127.0.0.1:1935",                    // how OpenViking Server reaches the gateway
    "openviking_url": "http://127.0.0.1:1933",         // how the gateway reaches OpenViking Server
    "public_url": "",                                  // how clients reach the gateway
    "storage_path": "~/.openviking/context-gateway",   // local disk only
    "encryption_key_env": "OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY",
    "admin_token_env": "OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN",
    "min_server_version": "0.4.16",
    "session_ttl_days": 30,
    "response_ttl_seconds": 2592000,
    "log_retention_days": 30,
    "max_body_bytes": 33554432,
    "upstream_timeout_seconds": 600,
    "health_interval_seconds": 60
  }
}
```

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Turns the gateway on. When false, the gateway refuses to start and Studio's Context Gateway page shows a setup card. |
| `host` | `127.0.0.1` | Bind address. Must be a loopback address while OpenViking runs in dev mode. |
| `port` | `1935` | Gateway port. |
| `workers` | `1` | Gateway processes on this host, 1–64. |
| `url` | `http://127.0.0.1:1935` | Address OpenViking Server uses for management and data deletion. Also the address Studio shows when `public_url` is empty. |
| `openviking_url` | `http://127.0.0.1:1933` | Address the gateway uses to reach OpenViking Server. |
| `public_url` | empty | Address clients use. Studio shows it in setup instructions; upload links for OpenViking tools use it. Set it in every shared deployment. |
| `storage_path` | `~/.openviking/context-gateway` | Directory for the two databases. Must be on a local disk. |
| `encryption_key_env` | `OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY` | Environment variable holding the encryption key. |
| `admin_token_env` | `OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN` | Environment variable holding the admin token. |
| `min_server_version` | `0.4.16` | Oldest OpenViking Server version the gateway works with. With an older or unparsable version, memory search and saving stop and no keys can be issued. |
| `session_ttl_days` | `30` | Days of inactivity after which a conversation's state is removed. |
| `response_ttl_seconds` | `2592000` | How long Responses IDs stay mapped to their upstream (at least 60). |
| `log_retention_days` | `30` | Request log retention. |
| `max_body_bytes` | `33554432` | Largest request body and upload, in bytes (at least 1,024). Larger ones get 413. |
| `upstream_timeout_seconds` | `600` | Total time limit for one call to a provider. |
| `health_interval_seconds` | `60` | How often the gateway checks OpenViking (at least 1). |

URL settings must be plain `http` or `https` addresses without credentials, query or fragment; a trailing `/` is removed.

**Environment variables:**

| Variable | Read by | Purpose |
| --- | --- | --- |
| `OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY` | Gateway | Encryption key (see [Secrets](#secrets)). |
| `OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN` | Gateway and OpenViking Server | Admin token, at least 32 characters. |
| `OPENVIKING_CONFIG_FILE` | Gateway and OpenViking Server | Path to `ov.conf` when `--config` is not given. |

**Context profile settings.** Studio label first, then the name used in the management API (where profiles are called `policies`):

| Studio label | API name | Default | Limits | Meaning |
| --- | --- | --- | --- | --- |
| Name | `name` | `Default` | | Display name. |
| Recall memory | `recall` | `true` | | Search memory for each new user message. |
| Sources | `context_types` | `memory`, `resource`, `skill` | at least one | What to search: memories, resources, skills. |
| Budget per message | `max_tokens` | `1600` | 64–32,000 | Most tokens added to one message. |
| Budget per conversation | `session_max_tokens` | `6000` | ≥ 0 | Most tokens added over a whole conversation; 0 turns recall off. |
| Relevance threshold | `score_threshold` | `0.35` | 0–1 | Lowest relevance score accepted. |
| Time limit | `recall_timeout` | `2` (seconds) | up to 30 | How long a search may take before the message goes on without memory. |
| Query length | `query_max_chars` | `8000` | 3–32,000 | Characters of the message used as the search query. |
| Limit by category | `quotas` | `{}` (off) | keys: `events`, `entities`, `preferences`, `experiences`, `resources`, `skills` | Most entries per category; only categories above 0 are searched. |
| Save conversations | `capture` | `true` | | Save finished turns to OpenViking. |
| Save the latest reply after | `idle_seconds` | `600` (seconds) | ≥ 1 | Quiet time before the last turn is saved and the session committed. |
| Commit after | `commit_tokens` | `20000` | ≥ 1 | Tokens waiting in the session that trigger a commit. |
| Keep recent messages | `keep_recent_messages` | `10` | 0–1,000 | Messages left in the session after a commit. |
| Long conversations | `takeover` | `true` | | Replace older history with OpenViking's summary. |
| Start after | `takeover_tokens` | `30000` | ≥ 1 | Saved tokens after which summaries take over. |
| Keep recent turns | `keep_recent_turns` | `3` | 1–100 | Turns sent word for word after the summary. |
| Wait for summary | `archive_wait_seconds` | `30` (seconds) | 0–60 | Longest wait for a pending summary near the context window. |
| Default context window | `context_window` | unset | ≥ 1,024 | Window used when the upstream does not list the model. |
| OpenViking tools | `gateway_tools` | `false` | | Offer OpenViking tools for Chat, full-history Responses and Anthropic Messages. |
| Allow write tools | `allow_write_tools` | `false` | | Required before any write tool is offered. |
| Tools | `tool_allowlist` | `search`, `read`, `list` | `search`, `read`, `list`, `write`, `add_resource`, `add_skill` | The tools offered, as `openviking_<name>`. |
| Rounds per request | `tool_max_rounds` | `5` | 1–20 | See [OpenViking tools](#openviking-tools). |
| Time limit per call | `tool_timeout_seconds` | `30` | up to 120 | |
| Result size | `tool_result_bytes` | `65536` | 1,024–1,048,576 | |
| Total time | `tool_total_seconds` | `120` | up to 600 | |
| Token budget | `tool_total_tokens` | `100000` | 1,024–1,000,000 | |

**Upstream settings:**

| Studio label | API name | Default | Meaning |
| --- | --- | --- | --- |
| Name | `name` | | Display name. |
| Protocol | `protocol` | | `anthropic` (Anthropic Messages), `chat` (Chat Completions) or `responses` (Responses). |
| Provider | `vendor` | `generic` | `generic`, `anthropic`, `openai`, `deepseek`, `ark` (Volcano Engine Ark) or `byteplus` (BytePlus ModelArk). Studio offers only the protocols the provider supports; the API accepts any pair. |
| Base URL | `base_url` | | Provider address; see [Upstreams](#upstreams) for each provider's default and the path rules. |
| Who provides the API key | `auth_mode` | `managed` | `managed` (The gateway holds the API key) or `passthrough` (Each client sends its own key). |
| API key | `api_key` | | Write-only. Blank on edit keeps the stored key. |
| Extra headers | `headers` | `{}` | Write-only values; names are visible. |
| Models | `models` | `[]` | Model names served; empty accepts any. |
| Model aliases | `aliases` | `{}` | Client-facing name → model sent to the upstream. |
| Context windows | `context_windows` | `{}` | Model → tokens, each at least 1,024. |
| Priority | `priority` | `0` | Higher wins for new conversations. |
| Enabled | `enabled` | `true` | Disabled upstreams receive no requests. |
| Allow OpenViking tools | `allow_gateway_tools` | `true` | Whether OpenViking tools may be offered through this upstream. |
| This key is a Coding Plan or subscription key | `coding_plan` | `false` | Rejects requests to this upstream unless allowed. |
| Allow it anyway | `allow_coding_plan` | `false` | Allows a key marked as Coding Plan. |
| Minimum cacheable prompt | `cache_min_tokens` | `1024` | Ark and ModelArk only; affects cache-eligibility reporting. |

**Gateway key fields:** Name (`name`), OpenViking key (`openviking_key`), Context profile (`policy_id`), Upstreams (`upstream_ids`, at least one) and Allowed models (`models`). When you issue through OpenViking Server, you can send the `user_id` of a user in this account instead of `openviking_key`, and OpenViking Server reads that user's key; send only one of the two.

Studio calls the management API on OpenViking Server under `/api/v1/admin/context-gateway/`, with the resources `overview`, `logs`, `guides`, `upstreams`, `policies`, `keys` and `users/{user_id}/data`. Scripts can call the same paths with an account admin key.

## Troubleshooting

### The gateway does not start

| Message | Cause and fix |
| --- | --- |
| `configure a Fernet encryption key and an admin token of at least 32 characters` | A secret is missing or too short in the gateway's environment. Load both [secrets](#secrets) into the shell, `.env` or Secret the gateway starts from. |
| `Set context_gateway.enabled=true in ov.conf` | The section is missing or disabled, or the gateway read another file. It reads `--config`, then `OPENVIKING_CONFIG_FILE`, then `~/.openviking/ov.conf`, and nothing else. |
| `Missing Context Gateway dependencies: …` | Install the extra: `pip install "openviking[context-gateway]"`. |
| A validation error about a `context_gateway` field | An unknown key, a typo or an invalid value. `${VAR}` placeholders are not expanded, so a placeholder in a numeric or URL field fails too. |
| `Context Gateway must bind to loopback when OpenViking uses dev authentication` | OpenViking runs in dev mode while `host` is not a loopback address. Switch OpenViking to API key mode. |
| `Unsupported gateway schema; configure a fresh storage_path` | The storage was written by an incompatible gateway version. Point `storage_path` at an empty directory and set the gateway up again. |

In Docker Compose, a gateway that keeps restarting usually has empty secrets in `.env`; `docker compose logs context-gateway` shows the message.

### Studio shows a setup card

The Context Gateway page shows a setup card instead of its tabs when OpenViking Server cannot use the gateway:

- **The Context Gateway is turned off**: OpenViking Server's `ov.conf` lacks `context_gateway.enabled: true`. Add it and restart OpenViking Server.
- **The management token is missing**: OpenViking Server's environment lacks the admin token, or it is shorter than 32 characters. Set it (in Docker Compose through `.env`, in Helm through `contextGateway.existingSecret`) and restart OpenViking Server.
- **OpenViking can't reach the gateway**: OpenViking Server cannot reach `context_gateway.url`. Check that the gateway is running and that `url` is the address OpenViking Server can use: `http://context-gateway:1935` in Docker Compose (the chart sets it for Helm).

If both processes are running but the page says "Your key was rejected. Check Connection settings." although the same admin key works on **Users & Permissions**, the two processes have different admin tokens, and the gateway answers 401 "Invalid gateway management credential". Set the same token for both and restart them.

If **Context Gateway** is missing from the sidebar, Studio has no admin access: enter an account admin or root key as the **Admin API key** in **Connection Settings**. While OpenViking runs in dev mode, Studio hides its management pages.

### Issuing a key fails

The **Issue a gateway key** dialog shows why issuing failed. The code in parentheses is the reason the management API returns.

| Message in Studio | Cause and fix |
| --- | --- |
| The server can't read this user's OpenViking key. Paste the key instead. | OpenViking Server can't read the chosen user's key, usually because it stores only key hashes (`encryption.api_key_hashing.enabled`). Choose **Paste an OpenViking key** and enter the user's own key. |
| This is a root key. (`root_key_not_allowed`) | The OpenViking key is a root key, or OpenViking runs in dev mode, where every key acts as root. Use the user's own key, and API key mode. |
| This OpenViking key belongs to another account. | The key belongs to a different account than the one you manage in Studio. |
| OpenViking rejected this key. (`openviking_http_401`) | The key is incomplete, was regenerated, or its user was removed. Use the user's current key. |
| OpenViking couldn't tell which user this key belongs to (`openviking_identity_missing`) | OpenViking returned no user for this key. Use the key of a user in this account. |
| The gateway couldn't reach OpenViking to check this key. (`openviking_unavailable`) | The gateway cannot reach `openviking_url`. |
| OpenViking is older than the gateway requires. (`openviking_version_mismatch`) | OpenViking Server is older than `min_server_version` (0.4.16). Upgrade it. A server installed from a source checkout without version tags may report a development version such as `0.1.dev123`; install a release, or set `min_server_version` to match. |

### Clients get an error from the gateway

These rejections happen before a request reaches a provider, so they do not appear on the Requests tab.

| Status and message | Cause and fix |
| --- | --- |
| 401 "Invalid or revoked Context Gateway key" | The client sent no gateway key, a revoked one, or its provider key. Check which key the client sends. |
| 403 "Claude subscription OAuth credentials are not supported" | The client sent a Claude subscription token, for example Claude Code signed in with a subscription and no `ANTHROPIC_AUTH_TOKEN` set; or the upstream holds one. Use API keys. |
| 403 "Model is not allowed by this key" | The model is not in the key's allowed models. Issue a key that allows it. |
| 404 "No allowed upstream matches this protocol and model" | No enabled upstream bound to the key speaks the API the client called and serves the requested model. Check the upstream's protocol, its models and aliases, whether it is enabled, and the key's upstreams. |
| 401 "Upstream API key is missing" | The upstream expects each client to send its own key in `X-OpenViking-Upstream-Key`, or a gateway-held upstream has no API key. |
| 403 "Coding Plan upstreams are disabled; configure a model API key" | The upstream is marked as a Coding Plan key. Use a model API key, or deliberately check **Allow it anyway**. |
| 404 "Unknown response for this key", 403 "Response upstream is no longer allowed" | A Responses lookup for a response this key did not create, or whose upstream is now disabled or no longer bound to the key. |
| 413 | The request body is larger than `max_body_bytes`, or than your proxy's own limit. |
| 426 | A WebSocket connection to the Responses API. Expected: Codex falls back to HTTP. |
| 502 "Model upstream is unavailable" | The gateway could not reach the provider, or the call exceeded `upstream_timeout_seconds`. Use **Test connection** on the upstream. |
| 503 "Dev authentication requires a loopback gateway" | OpenViking runs in dev mode. Switch it to API key mode. |
| 504 "Hidden tool request timed out" | OpenViking tool rounds exceeded the profile's **Total time** for tools. |

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
- **You are looking as another user.** Sessions belong to the OpenViking user behind the gateway key and are named `context-gateway-…`.
- **Memories come later than sessions.** OpenViking extracts memories only after a commit, in the background.

### The first-call cache hit rate dropped

Compare it with what the provider reaches without the gateway, then look at the issues on recent requests:

- **Memory record missing** (`missing_injection_record`): the gateway's storage was lost, restored from an old backup or expired, so memory added earlier could not be replayed. Ongoing conversations recover after one miss.
- **Upstream switched** (`upstream_changed`): conversations moved to another upstream because theirs was disabled, is not bound to the key the client now uses, or stopped serving the model.
- **Cache parameters changed** (`ark_cache_parameters_changed`): on Ark or ModelArk, the client changes the model, thinking, sampling, system prompt or tools within a conversation.
- No issue: the client itself may change earlier messages or its system prompt every turn, for example by inserting the current time. The gateway cannot fix that. Each summary in a long conversation also costs one miss, which is expected.

### File imports fail

- "Set context_gateway.public_url for client uploads": set `public_url` to an address the client can reach.
- The upload link cannot be reached, or returns 404: route `/context-gateway/uploads` to the gateway in your proxy.
- 413: the file is larger than `max_body_bytes` or the proxy's limit.
- 502 "OpenViking upload is unavailable": the gateway cannot reach OpenViking Server.
- The import tools are not offered at all: the conversation's first request had neither a shell tool nor an attachment, or write tools are not allowed in the profile.

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
| Summary not ready (`archive_wait_timeout`) | The conversation was near the context window and its summary was not ready in time; the full history was sent. | See [Long conversations](#long-conversations). |
| Tool history unavailable (`hidden_tool_history_unavailable`) | A Claude conversation previously used OpenViking tools, but this request cannot use them, so earlier thinking has been removed. | Check the tool skip reason in the request detail, restore the original settings or start a new conversation. |
| OpenViking tools failed (`hidden_tool_loop_failed`) | The model could not finish answering while using OpenViking tools, and the client received an error. | See [OpenViking tools](#openviking-tools). |
| Reply not saved (`capture_parse_failure`) | A streamed reply could not be parsed, so this reply was not saved. The client was not affected. | Nothing; it is informational. |
