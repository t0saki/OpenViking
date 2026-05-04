# Migration Guide: claude-memory-plugin → claude-code-memory-plugin

This guide helps you migrate from the legacy `claude-memory-plugin` to the new `claude-code-memory-plugin`.

## Key Differences

| Aspect | Old Plugin (`claude-memory-plugin`) | New Plugin (`claude-code-memory-plugin`) |
|--------|--------------------------------------|-------------------------------------------|
| **Architecture** | Bash hooks + Python bridge (`ov_memory.py`) | Native MCP server + Node.js hooks |
| **Auto-Recall** | Skill-based, requires explicit invocation | Transparent, injected via `UserPromptSubmit` hook |
| **Memory Capture** | Accumulates on `Stop`, commits on `SessionEnd` (async) | Incremental capture on each `Stop` (sync, safer) |
| **Crash Safety** | Risk of losing accumulated turns if Claude crashes before `SessionEnd` | Each turn persisted independently — no data loss |
| **Type Safety** | Shell + Python, no compile-time checks | TypeScript MCP server, compiled JS artifact |
| **Tool Discovery** | Custom skill file | Standard MCP tools (`memory_recall`, `memory_store`, etc.) |
| **Config Scope** | Project-local (`./ov.conf`) | Global (`~/.openviking/ov.conf`) — works across all projects |
| **Runtime Bootstrap** | Manual setup required | Auto-bootstrap on first `SessionStart` into `${CLAUDE_PLUGIN_DATA}` |

### Why the New Model is Better

1. **Incremental Capture = Crash-Resilient**: The old approach batches all turns until `SessionEnd`. If Claude crashes or the user kills the session, those memories are lost. The new plugin extracts on every `Stop`, so each turn is independently persisted.

2. **Transparent Auto-Recall**: No skill invocation needed — memories are automatically injected into context via `systemMessage` on every prompt. This matches the mental model of "memory just works."

3. **MCP Native**: Standard protocol, better tooling support, type-safe TypeScript implementation. Other MCP-compatible clients can potentially reuse the same server.

4. **Global Configuration**: One config file for all projects (`~/.openviking/ov.conf`), not per-project `./ov.conf` files.

## Capture Strategy: Both Sides by Default (revised 2026-05)

The plugin originally shipped with `captureAssistantTurns: false` so coding-only workflows wouldn't pay LLM-extraction cost on tool-heavy assistant turns. Field testing showed two problems with that posture:

1. **Asymmetric capture confused users.** The `subagent-stop` path always pushed both sides; only the main-session path filtered. Same plugin, same config, different sessions ended up half-empty vs. full with no log line explaining why.
2. **Memory extraction degraded substantially.** Without assistant turns the extractor saw "user asked X" → next turn → "user asked Y", missing the agent's reasoning, decisions, and any context surfaced by tool I/O. For non-trivial workflows the plugin felt like it was "forgetting half the conversation".

### Current Default

The plugin defaults to `captureAssistantTurns: true` — both user and assistant turns are captured, including tool input/output (truncated per block to 4096 chars). This aligns the main-session and subagent paths and gives the memory extractor enough context to do real extraction.

Operators who want the old user-only behavior can opt out:

```json
// ~/.openviking/ov.conf
{
  "claude_code": {
    "captureAssistantTurns": false
  }
}
```

…or set `OPENVIKING_CAPTURE_ASSISTANT_TURNS=0` in the environment.

### Cost note

Assistant capture does increase OV-side LLM extraction load (roughly 2× more tokens per session in coding workflows). The plugin's `commitTokenThreshold` (default 20000) still bounds how often extraction runs. If extraction cost is a concern for a deployment, opt out via the env var rather than hand-editing the default.

## Migration Steps

### 1. Update Config Location

Move your config from project root to the global location:

```bash
# Old location (per-project)
./ov.conf

# New location (global)
~/.openviking/ov.conf
```

If you have project-specific configs, merge them into the global config. The new plugin supports a `claude_code` section for plugin-specific overrides.

### 2. Install Node.js Runtime

The new plugin requires Node.js. Ensure Node.js 18+ is installed:

```bash
node --version  # Should be 18.x or higher
```

### 3. Remove Old Plugin, Install New Plugin

```bash
# Uninstall old plugin (if installed via marketplace)
/plugin uninstall openviking-memory

# Install new plugin
/plugin marketplace add Castor6/openviking-plugins
/plugin install claude-code-memory-plugin@openviking-plugin
```

### 4. Remove Per-Project Plugin Files

Delete the old plugin directory from your project if it was checked in:

```bash
rm -rf ./.openviking/memory/
```

### 5. Update Config File (Optional)

Add the `claude_code` section to `~/.openviking/ov.conf` for plugin-specific settings:

```json
{
  "claude_code": {
    "agentId": "claude-code",
    "recallLimit": 6,
    "captureMode": "semantic",
    "captureTimeoutMs": 30000,
    "captureAssistantTurns": false
  }
}
```

## Behavior Changes

### Memory Recall

| Old Plugin | New Plugin |
|------------|------------|
| Run `/memory-recall <query>` manually | Automatic recall on every prompt |
| Skill-based, requires Claude tool call | Transparent, injected as system context |

The new plugin automatically injects relevant memories into the context before each prompt. No manual action needed.

### Memory Capture

| Old Plugin | New Plugin |
|------------|------------|
| Captures at `SessionEnd` (batch) | Captures at `Stop` (incremental) |
| Requires explicit commit | Automatic, real-time |

The new plugin captures memories incrementally after each turn, providing faster feedback and better persistence.

## MCP Tools (New Feature)

The new plugin provides MCP tools for explicit memory operations:

| Tool | Description |
|------|-------------|
| `memory_recall` | Manual semantic search |
| `memory_store` | Manually store a memory |
| `memory_forget` | Delete memories by URI or query |
| `memory_health` | Check server status |

These tools are available when you need explicit control over memory operations.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MCP tools not available | Start a new Claude session; first run installs runtime deps |
| No memories recalled | Ensure OpenViking server is running: `openviking-server` |
| Hook timeout | Increase timeout in `hooks/hooks.json` |
