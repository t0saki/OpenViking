# Agent Integrations Overview

OpenViking can act as the long-term memory and context backend for many agent runtimes. Pick the integration that matches your agent.

## Which integration should I use?

| If you use… | Use this |
|-------------|----------|
| **Claude Code** | [Claude Code Memory Plugin](./02-claude-code.md) — auto-recall + auto-capture via hooks |
| **OpenClaw** | [OpenClaw Plugin](./03-openclaw.md) — context-engine with full lifecycle integration |
| **Codex** | [Codex Memory Plugin](./04-codex.md) — lifecycle hooks for auto-recall and incremental capture |
| **Cursor** | [Cursor Memory Integration](./12-cursor.md) — one command installs lifecycle hooks, MCP tools, rules, and skills |
| **TRAE / TRAE CN** | [TRAE Memory Integration](./13-trae.md) — one installer configures prompt-time recall, turn capture, and OpenViking tools |
| **Hermes Agent** | [Hermes Agent](./05-hermes.md) — built-in OpenViking memory provider, no plugin install needed |
| **OpenCode** | [OpenCode Plugin](./10-opencode.md) — MCP tools plus lifecycle hooks for repo context, auto-recall, and capture |
| **pi** | [pi Coding Agent Extension](./11-pi.md) — native extension with auto-recall, turn capture, and threshold commit |
| **LangChain / LangGraph** | [LangChain and LangGraph](./07-langchain-langgraph.md) — retriever, tools, context backend, store, and middleware |
| **Multiple local coding agents / a desktop UI** | [OpenViking Helper](./14-openviking-helper.md) — visual agent setup, session inspection, and memory management |
| **Manus / Claude Desktop / ChatGPT / other MCP clients** | [MCP Clients](./06-mcp-clients.md) — point any MCP-compatible client at the built-in `/mcp` endpoint |
| **AstrBot / …** | [Community Plugins](./08-community-plugins.md) — community-maintained integrations for various runtimes |

## Prerequisite for all integrations

Every integration on this page connects to a running OpenViking server. If you don't have one yet, follow the [Quickstart Guide](../getting-started/02-quickstart.md). The default endpoint is `http://localhost:1933`; remote use requires an API key (see [Authentication](../guides/04-authentication.md)).

## Low-latency recall

Query expansion and result rewriting are two independent, optional model calls. Disable both in the Agent plugin when response latency matters most; semantic retrieval, budgeting, tier degradation, and cross-turn dedup continue to work.

The Claude Code and Codex memory plugins support environment-variable configuration:

```bash
# Claude Code: disable server query expansion and optional local/server digest rewriting
export OPENVIKING_RECALL_QUERY_EXPANSION=off
export OPENVIKING_RECALL_REWRITE=off

# Codex: disable server query expansion and Codex's own local compression pass
export OPENVIKING_RECALL_QUERY_EXPANSION=off
export OPENVIKING_RECALL_COMPRESS=0
```

Codex already keeps the server rewrite off, but runs a local `codex exec` compression pass by default. It therefore uses `OPENVIKING_RECALL_COMPRESS=0`, not `OPENVIKING_RECALL_REWRITE=off`.

The same settings can live in `~/.openviking/ovcli.conf`:

```json
{
  "url": "https://openviking.example.com",
  "api_key": "your-api-key",
  "plugin": {
    "claude_code": {
      "recallQueryExpansion": "off",
      "recallRewrite": "off"
    },
    "codex": {
      "recallQueryExpansion": "off",
      "recallCompress": false
    }
  }
}
```

Environment variables take precedence over `ovcli.conf`. Restart the Agent after changing these settings so its hook processes reload the configuration. These are plugin-client settings; the server's `ov.conf` does not need to change.
