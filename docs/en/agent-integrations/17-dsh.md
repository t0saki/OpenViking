# DeepSeek Harness Memory Bundle

Give [DeepSeek Harness](https://www.npmjs.com/package/@deepseek-ai/dsh) (`dsh`) cross-session long-term memory. The bundle is an in-process Cordis plugin: it injects your OpenViking profile at session start, retrieves before every model step, captures the session as it happens, commits on a token threshold, and mounts the OpenViking MCP tool surface.

Package: [`@openviking/dsh-memory-plugin`](https://www.npmjs.com/package/@openviking/dsh-memory-plugin) · Source: [examples/dsh-memory-plugin](https://github.com/volcengine/OpenViking/tree/main/examples/dsh-memory-plugin)

## How it works

- **Session start** (`agent/session-start`) injects your OpenViking profile block and the available-memory index through `agent.inject()`.
- **Every step** (`agent/pre-step`) runs semantic recall against the step input and appends the result to that same step as a durable, source-attributed user message. It runs with `prepend: true` so it sees the final claimed batch and appends after every other contributor.
- **Every event** (`session/event`) captures user, assistant, and — when enabled — tool-result messages directly from DSH's event stream, with no transcript scraping.
- **Turn end** commits to OpenViking once pending tokens cross `commitTokenThreshold` (default `20000`), keeping the 10 most recent messages live.
- **Tools** come from the OpenViking MCP surface through the same stdio proxy the other memory integrations use, bridged into DSH by `@deepseek-ai/dsh-mcp-client` and published to the model as `mcp__openviking__search`, `mcp__openviking__read`, `mcp__openviking__remember`, `mcp__openviking__write`, and the rest of what the connected server advertises.
- **URI guard** (`tools/pre-execute`) blocks DSH filesystem and shell tools from treating a `viking://` URI as a local path, pointing the model at the bridged `mcp__openviking__*` tools instead.
- Failed writes land in the shared OpenViking pending queue and replay at the next session start.

Each DSH session maps to `dsh-<session-id>` in OpenViking; every subagent gets its own session. Workspace-derived actor peers resolve per session and travel on every session-specific request.

### Why recall bypasses the system prompt

Recall enters as pre-step user messages rather than system-prompt sections because a DSH preset whose persona declares `complete: true` (the stock `minimal` preset does) restores that persona as the sole prompt section after assembly, silently discarding every other contribution. Pre-step injection also makes each injection a replayable session event that compaction can see.

### How the tool surface is mounted

The bundle mounts DSH's own MCP bridge on `servers/mcp-proxy.mjs` — the same stdio proxy Claude Code, Codex, Cursor, and OpenCode start — instead of hand-registering a tool subset, so the model gets whatever the connected server advertises and a server upgrade adds tools without a plugin release. The bundle's resolved credentials reach the proxy through the child environment, so no MCP configuration is needed in the profile, and the bridge ships with `@deepseek-ai/dsh` itself — there is nothing extra to install.

Because the proxy is one process per profile, tool calls carry the actor peer resolved at boot rather than per session, and `mcp__openviking__remember` stores into a short-lived server-side session rather than the live `dsh-<session-id>` stream — the same behavior the Claude Code, Codex, and Cursor integrations have. Recall, capture, and commit are unaffected: they still resolve a peer from each session's own workspace. Set `OPENVIKING_PEER_ID` when one process serves several workspaces and tool calls need exact attribution.

Startup failure is contained: recall, capture, and commit keep working against a server whose MCP endpoint is unreachable, and the bridge reconnects on its own.

## Prerequisites

- `@deepseek-ai/dsh` `0.1.0-rc.6` — install that exact release; the `@deepseek-ai/dsh-*` prerelease tags are not published in lockstep
- Node.js `^22.19.0` or `>=24`
- An OpenViking HTTP server — verify with `curl http://localhost:1933/health`

## Install

Add the published package to a profile (`web` is the default profile):

```bash
dsh plugin --profile web add @openviking/dsh-memory-plugin
```

From an OpenViking repository checkout instead:

```bash
git clone https://github.com/volcengine/OpenViking.git
cd OpenViking
dsh plugin --profile web add ./examples/dsh-memory-plugin
```

The package ships a `cordis.patch.yml` that mounts the runtime inside a Cordis plugin group with an isolated `openvikingMemory` service. Confirm it landed:

```bash
dsh --profile web --dump-config
```

DSH is not covered by the unified memory-plugin installer; `dsh plugin add` is the install path.

## Configure

Credentials resolve from `OPENVIKING_*` environment variables, then `~/.openviking/ovcli.conf`, then `~/.openviking/ov.conf` — the same chain the Claude Code, Codex, OpenCode, and pi plugins use.

| Variable | Purpose |
|----------|---------|
| `OPENVIKING_URL` / `OPENVIKING_BASE_URL` | Server endpoint (default `http://127.0.0.1:1933`) |
| `OPENVIKING_API_KEY` / `OPENVIKING_BEARER_TOKEN` | Bearer credential |
| `OPENVIKING_ACCOUNT` / `OPENVIKING_USER` | Trusted-mode account and user |
| `OPENVIKING_PEER_ID` | Explicit actor peer |
| `OPENVIKING_WORKSPACE_PEER` | Derive a peer from each DSH session workspace (default on) |
| `OPENVIKING_RECALL_PEER_SCOPE` | `all` for cross-workspace recall, `actor` for isolation |

Behavior knobs go in the Cordis patch entry:

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

Credentials given in the patch win over the environment; behavior toggles read from the environment first.

## Verify

Start `dsh --profile web`, mention something you told it in an earlier session, and confirm the answer draws on that memory. `dsh --profile web --dump-config` shows whether the plugin group is mounted; set `OV_DEBUG_LOG=/tmp/ov-dsh.log` to trace recall and capture decisions.

## Troubleshooting

| Issue | What to check |
|-------|---------------|
| Plugin does not load | `dsh --profile web --dump-config` should list `openviking-memory`; re-run `dsh plugin --profile web add …` |
| `ERESOLVE` during install | The `@deepseek-ai/dsh-*` prerelease tags drift apart; install `@deepseek-ai/dsh@0.1.0-rc.6` exactly |
| Nothing is recalled | `curl http://localhost:1933/health`; check the endpoint and that the prompt is longer than `minQueryLength` (default 3) |
| No `mcp__openviking__*` tools offered | The bridge contains startup failures — check the DSH log for `mcp-client(openviking)`, and set `OV_DEBUG_LOG=/tmp/ov-dsh.log` to trace the proxy |
| 401 / 403 from OpenViking | Verify `OPENVIKING_API_KEY`; for trusted-mode deployments also verify `OPENVIKING_ACCOUNT` and `OPENVIKING_USER` |
| Memories from other projects leak in | Set `OPENVIKING_RECALL_PEER_SCOPE=actor` |
| Nothing was committed after a crash | Commit happens on a token threshold and on teardown; queued writes replay at the next session start |

For the full tool, configuration, and release reference, see the [bundle README](https://github.com/volcengine/OpenViking/tree/main/examples/dsh-memory-plugin).

## See also

- [Capability Reference](./16-capability-reference.md)
