# Connect your agent with one prompt

Paste one prompt into the coding agent you already use. The agent installs the OpenViking memory plugin into itself, checks the connection, offers to bring in your recent work, and tells you how to restart. After the restart, your first question in a new session shows OpenViking at work.

The agent follows a short guide published as plain Markdown, so any agent can fetch it:

- English: `https://docs.openviking.ai/agents/en/onboarding.md`
- Chinese: `https://docs.openviking.ai/agents/zh/onboarding.md`

Both are also served from `docs.openviking.net`, which is usually faster from mainland China.

## The prompt

Copy the prompt and replace two values before you send it:

- `<HARNESS>`: your agent's value from the table below.
- `<API_KEY>`: your key from the [OpenViking console](https://console.volcengine.com/vikingdb/openviking/region:openviking+cn-beijing?utm_source=opensource_docs&utm_medium=referral&utm_campaign=getting-started_07-agent-onboarding), under **User Management → API Key**.

<AgentPrompt>

```text
Connect OpenViking memory to you (the agent I'm using now) and walk me through the first use.

1. Run the official installer. It needs network access and writes to my home directory; if a sandbox blocks it, ask me to approve running it outside the sandbox:
   curl -fsSL https://openviking.ai/install | bash -s -- --yes --harness <HARNESS>,cli --url https://api.vikingdb.cn-beijing.volces.com/openviking --api-key <API_KEY>
2. When it finishes, fetch this guide and follow it (if it doesn't load, use docs.openviking.net instead of docs.openviking.ai):
   https://docs.openviking.ai/agents/en/onboarding.md

Never repeat my API key or write it into any file or rule. Talk to me in English.
```

</AgentPrompt>

| Agent | `<HARNESS>` |
|---|---|
| Claude Code | `claude` |
| Codex | `codex` |
| TraeCode CLI 2.0 | `trae-cli` |
| Cursor | `cursor` |
| TRAE | `trae` |
| TRAE CN | `trae-cn` |
| ZCode | `zcode` |
| Kimi Code | `kimicode` |
| OpenCode | `opencode` |
| pi | `pi` |
| DeepSeek Harness | `dsh` |

`--harness` limits the install to the agent you are talking to. Without it, the installer sets up every agent it detects on the machine. `cli` adds the `ov` command, which the agent uses to check the connection and import history.

**Self-hosted server.** Replace the URL with your server's address. Use a user key; a root key can't read or write memories. For a local server without authentication, pass `--api-key ''`.

**About the key.** The installer accepts the key only as a command-line argument, so the key appears in this conversation and in your agent's local session history. The guide tells the agent never to repeat it or write it anywhere, and to have you start a new session after the restart instead of resuming this one. If you'd rather not paste the key into a chat, run `curl -fsSL https://openviking.ai/install | bash` in a terminal yourself, then send your agent the prompt under [Already installed](#already-installed).

## Already installed

If the plugin is already installed, start a new session in your agent and send:

```text
Follow https://docs.openviking.ai/agents/en/onboarding.md to walk me through my first use of OpenViking.
```

## What the agent does

1. **Checks the install.** It reads the installer's output and runs `ov health`, then tells you which agent was set up and which server it is connected to.
2. **Explains what runs automatically.** Before each prompt, the plugin recalls relevant memories. After each turn, it saves the conversation, and the server extracts long-term memories in the background. You don't need to approve this separately; asking for the install is the approval.
3. **Asks whether to import recent work.** On Claude Code and Codex, the default is the current project's sessions from the last 14 days. It shows you the number of sessions and messages first and imports only after you confirm. Only user and assistant text is sent; tool calls and their output are dropped and API keys are removed. You can also name files, such as a design doc, to add with `ov add-resource`. Importing history from other agents isn't supported yet.
4. **Offers one rule, on some agents.** TRAE, TRAE CN, ZCode and Kimi Code ship no OpenViking skill, so the agent offers a short rule for your personal rules that tells it when to use the OpenViking tools. It never writes rules into project files.
5. **Tells you how to restart.** Hooks load only in a new session. The agent gives you the exact step for your agent (for example, **Trust all and continue** in Codex) and a first message to send after the restart.
6. **Shows OpenViking at work.** In the new session, your first question gets an answer drawn from recalled context, and the agent points out where that shows up: source cards in Claude Code, the OV-Usage line in Codex, and so on.

The whole flow asks you at most two questions.

## See also

- [Agent integrations overview](../agent-integrations/01-overview.md)
- [Agent integration capabilities](../agent-integrations/16-capability-reference.md)
- [Install and use the CLI](05-cli-setup.md)
