# DESIGN: the hosts established by inspection

ZCode and Kimi Code are the two hosts in this plugin whose extension surface had to be established by inspection rather than from documentation. What was verified for each, and what the adapter decided because of it, is recorded here.

## Verified ZCode extension surface

These facts were verified against a live ZCode installation (built-in `zcode-guide` plugin docs + actual `~/.zcode/cli/config.json` + real installed plugins with hooks).

| Aspect | Verified fact |
|--------|--------------|
| Supported hook events | `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, `PostToolUseFailure`, `Stop` (exactly 7) |
| Unsupported events | `PreCompact`, `SessionEnd`, `Notification`, `SubagentStart`, `SubagentStop` |
| Manifest probe order | `.zcode-plugin/plugin.json` → `.claude-plugin/plugin.json` → `.codex-plugin/plugin.json` |
| Template vars (plugin hooks) | `${CLAUDE_PLUGIN_ROOT}`, `${ZCODE_PLUGIN_ROOT}`, `${CLAUDE_PROJECT_DIR}`, `${ZCODE_PROJECT_DIR}`, `${CLAUDE_SESSION_ID}` |
| Template vars (config hooks) | None — config-file hooks do NOT expand templates |
| Hook output schema | Strict JSON — any extra key fails validation, output discarded |
| MCP config location | `~/.zcode/cli/config.json` → `mcp.servers` (user scope) |
| Plugin MCP namespacing | `plugin:<plugin>:<server>` |
| MCP auto-connect | All scopes auto-connect at session start |
| Hook runner enablement | Auto-enabled when any plugin contributes a hook |
| Timeout units | `command` type: `timeout` in **seconds**; `process` type: `timeoutMs` in milliseconds |
| `async` field | No runtime effect — hooks always run inline |

## ZCode design decisions

### 1. Assemble the shared runtime at install time (no vendored copy)

ZCode imports the shared runtime across the plugin boundary, the way Cursor and TRAE do. The installer copies the modules `lib/MANIFEST` names to `~/.openviking/agent-integrations/memory-plugin-shared/lib`, which is exactly where `../../memory-plugin-shared/lib` resolves from an installed `hosts/` or `scripts/` file — so the same relative path works in this repository and on a user's machine, and no generated copy has to be kept in git.

### 2. Config-file hooks (not plugin-manifest hooks)

`install_zcode()` writes hooks and MCP config into `~/.zcode/cli/config.json` (the config-file scope), not via plugin marketplace registration. This mirrors the Cursor/TRAE install pattern.

`__OPENVIKING_PLUGIN_ROOT__` in the source `hosts/zcode/hooks.json` is replaced by absolute paths at install time by `renderHookCommand()` in `lib/install/host-json-config.mjs`, so the config-file "no template expansion" limitation does not apply.

**Provenance**: Adversarial review R4 — config-file hooks require `hooks.enabled: true`; the merge script sets this automatically.

### 3. Four events only (ZCode-supported subset)

ZCode supports 7 events but NOT `PreCompact`/`SessionEnd`/`SubagentStart`/`SubagentStop`. The host wires 4 events. The commit-on-`Stop` strategy compensates for the absence of `PreCompact`/`SessionEnd`; the Stop parent detaches before reading stdin so network writes do not block ZCode.

**Provenance**: Adversarial review R1 — confirmed all 4 event names valid; R4 — unsupported events silently dropped.

### 4. Output schema: ZCode-canonical keys only

ZCode's strict JSON schema rejects unrecognized keys. The adapter's envelope emits ONLY `{ hookSpecificOutput: { hookEventName, additionalContext } }` for context injection, and `{ hookSpecificOutput: { hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason } }` for URI guard. No `decision: "approve"` (Claude-Code-ism).

**Provenance**: Adversarial review R1-F1, R4-V1 — the #1 silent-failure mode.

### 5. No host plugin manifest

ZCode probes `.zcode-plugin/plugin.json`, then `.claude-plugin/plugin.json`, then `.codex-plugin/plugin.json` — but nothing here is registered with its plugin system. The installer writes hooks and MCP config straight into `~/.zcode/cli/config.json`, so the only manifest that matters is `hosts/zcode/openviking.integration.json`, which the installer reads.

## ZCode primary unknowns

1. **Hook stdin field names**: Verified via ZCode source reverse-engineering (#3127 by @quinn-zenith). The Stop hook exposes `responseText`/`responsePreview` for assistant content. User content is NOT in stdin — the parser falls back to ZCode's rollout file (`~/.zcode/cli/rollout/model-io-<sessionId>.jsonl`) which contains the complete conversation per line: `{ sessionId, turnId, request: { messages: [...] }, response: { text } }`.
2. **Output schema acceptance**: Whether `hookSpecificOutput` wrapper is accepted as-is. Must be tested against a live ZCode session.
3. **MCP tool name format**: Namespaced as `plugin:openviking:openviking` — verify tool names match expectations.
4. **Turn identity**: Rollout entries carry a monotonic `turnId`. The rollout is the authoritative incremental source whenever it is readable; stdin is a compatibility fallback only. The adapter sends this identity as OpenViking's `turn_id`, records both role-specific dedup keys only after messages are sent or durably queued, and advances `lastTurnId` only through complete acknowledged rollout entries.

## ZCode adversarial review incorporation

The focused regression suite covers rollout-first recovery, acknowledgement and cursor state, duplicate Stop delivery, detached slow writes, and installation from the same marketplace staging script used by the TOS release workflow.

## Verified Kimi Code extension surface

These facts were checked against Kimi Code CLI 0.41.0, its `hooks` / `mcp` / `plugins` / `config-files` documentation, and a live `~/.kimi-code` install. The right-hand column is what a reader coming from the ZCode adapter must not carry over.

| Aspect | Kimi Code | ZCode (do not copy) |
|--------|-----------|---------------------|
| Config location | `~/.kimi-code/config.toml`, moved wholesale by `KIMI_CODE_HOME` | `~/.zcode/cli/config.json` |
| Hook rules | A `[[hooks]]` array whose entries hold only `event`, `matcher`, `command`, `timeout` | a `hooks.events` JSON tree |
| MCP config location | `~/.kimi-code/mcp.json` → `mcpServers` | `mcp.servers` inside config.json |
| Hook stdin | snake_case: `session_id`, `hook_event_name`, `cwd`, `tool_name`, `tool_input` | camelCase |
| `UserPromptSubmit` output | **plain stdout text**, appended to the conversation as it stands | strict JSON `hookSpecificOutput.additionalContext` |
| `SessionStart` / `SessionEnd` / `PreCompact` / `Interrupt` | all exist, all observation-only — their output is dropped | only `SessionStart`, and it can inject |
| `Stop` | blockable; this adapter passes through | blockable; passes through |
| `PreToolUse` deny | `{hookSpecificOutput:{permissionDecision:"deny"}}` or exit 2 | the same shape |
| Transcript | `session_index.jsonl` → `agents/main/wire.jsonl` | `~/.zcode/cli/rollout/model-io-*.jsonl` |
| Duplicate `command` strings | de-duplicated by the host and run once | n/a |
| Native plugin manifest | `kimi.plugin.json`, carrying `hooks` and `mcpServers` | `.zcode-plugin/plugin.json` |

## Kimi Code lifecycle mapping

| Host event | What the adapter does |
|------------|-----------------------|
| `SessionStart` | Replay the pending queue. Print nothing: the host drops it. |
| `UserPromptSubmit` | Inject the profile once, then recall, as plain text. Stash the prompt for capture. |
| `PreToolUse` `Read\|Glob\|Grep` | Deny `viking://` reads and point the agent at the MCP tools. |
| `Stop` | Capture from the wire cursor and commit, in a detached worker. |
| `PreCompact` | The same capture — the host compacts right after. |
| `SessionEnd` | The same capture. |
| `Interrupt` | The same capture, inline: it fires instead of `Stop`, while the host is already tearing the turn down. |

Sessions are derived with the `kc-` prefix.

## Kimi Code design decisions

### 1. Plain text is the envelope

The host appends a hook's stdout to the conversation verbatim, so a JSON envelope would be injected as context for the user to read. The adapter's envelope returns the recall block itself for `UserPromptSubmit` and `null` for every other event. This is the inverse of ZCode's strict-JSON schema, and it is the one difference that silently corrupts a session rather than failing loudly.

### 2. The profile rides the first prompt

`SessionStart` cannot inject here, so the shared entry moves the profile block onto the first `UserPromptSubmit` of the session. `SessionStart` is still wired, because replaying the pending queue is worth a hook that prints nothing.

### 3. A comment-delimited block in TOML, not a reserialized file

`~/.kimi-code/config.toml` is a user-edited file this installer cannot round-trip, and other tools keep their own `[[hooks]]` entries in it. So the installer owns exactly one `# >>> openviking kimicode integration` block, renders it from `hosts/kimicode/hooks.json`, and reclaims that block and nothing else on uninstall. `__OPENVIKING_PLUGIN_ROOT__` and `__OPENVIKING_CLIENT_ID__` are expanded at install time by `lib/install/toml-hooks.mjs`, the way `lib/install/host-json-config.mjs` expands them for the JSON hosts.

### 4. `Interrupt` captures inline

Every other capture on this host detaches a worker and answers immediately. `Interrupt` does not: it arrives instead of `Stop`, with the host already unwinding the turn, so a detached child would be handing its writes to a parent about to be reaped.

### 5. The native manifest is a second door, not the door

`hosts/kimicode/kimi.plugin.json` lets `/plugins install <path>` work against an installed copy, and its commands are relative to the integration root for exactly that reason. `install.sh --harness kimicode` remains the supported path: it is idempotent, it writes the credentials, and it is what `--uninstall` reclaims.

## Kimi Code adversarial checks

- Recall must never emit a ZCode or Claude Code JSON wrapper; the host would show it to the user.
- An empty `matcher` is omitted rather than written, since the host treats it as a regex.
- Uninstall removes only the comment-delimited block, leaving unrelated `[[hooks]]` entries and the seam around them intact.
- The wire cursor is the host's own `turnId`; a missing wire file falls back to hook stdin plus the stashed prompt.
