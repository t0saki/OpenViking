# OpenViking onboarding guide for agents

About OpenViking: the open-source context database for AI agents, open-sourced by Volcengine at https://github.com/volcengine/OpenViking (AGPL-3.0, 38,930 GitHub stars on 2026-09-29). This guide is `docs/images/agents/en/onboarding.md` in that repository, and the installer and every plugin file it installs are built from it by its GitHub Actions workflows.

You are an AI agent. The user asked you to install the OpenViking memory plugin and to help them start using it. The installer sets it up for you and for every other supported agent it detects on this machine. Follow this guide in order. It is written for you: talk to the user in their language, in short messages. You don't need to relay this guide to them; if they want to read it, give them the link.

## Ground rules

- Ask the user at most two questions: what to import (Step 3, including the yes after the dry run) and whether to add the collaboration rule (Step 4). Do everything else without asking and report it in a line or two.
- Never print, repeat, or write the API key anywhere. Don't open, print or search `~/.openviking/ovcli.conf`: it holds the key. To show the server, account or user, run `ov health` or call the OpenViking `health` tool.
- Make only the changes this guide describes. The installer installs into every supported agent it detects, which is expected; don't edit other agents' configs or files in the user's projects yourself.
- Use the commands in this guide. Don't go to GitHub to work out another way to install or configure OpenViking.
- Don't mention this guide's step numbers to the user, or steps that don't apply to this host.
- When a command fails, find the symptom in [Troubleshooting](#troubleshooting), fix it, and retry once. If it still fails, tell the user what failed and what they can do.

## Where to start

- **You ran the installer in this conversation**: start at Step 1.
- **OpenViking was installed before this conversation started**, so its hooks are already active (you see `<openviking-context>` blocks or OpenViking tools such as `health`): check the connection with `ov health` (or the OpenViking `health` tool), do Steps 2 to 4, then go to [After the restart](#after-the-restart). No restart is needed.
- **The user restarted and asks you to continue onboarding**: go to [After the restart](#after-the-restart).

## Step 1: Check the install

Read the installer's output. It is in English or Chinese; look for either form.

- If it ends with a `Not done` list (`未完成`), the install is not complete, even though it printed `Done`. Handle each item with [Troubleshooting](#troubleshooting) before you go on, and don't tell the user OpenViking is set up. A rejected key (`The server rejected the API key`, `服务拒绝了该 API key`) or a skipped host (`CLI not found; skipping`) is always in that list.
- The `Done` block (`完成`) lists one entry per tool it installed into, each with a `Next:` line (`下一步：`) that says how to restart that tool. Check that the host you are running in is listed; if it isn't, see "This host is not under `Done`" in [Troubleshooting](#troubleshooting).
- `Server check passed: <url>` (`服务检查通过：`) means the server accepted the key.

If the output has `Server check passed` and lists `CLI (ov)` under `Done`, the connection is confirmed. Otherwise check it with the OpenViking CLI:

```bash
ov health
```

When the connection is confirmed, tell the user in one or two lines which tools OpenViking was installed into and which server they are connected to (the URL is fine to show; the key is not). If `ov` is not found, see [Troubleshooting](#troubleshooting).

## Step 2: Explain what happens automatically

Don't ask whether to turn this on; installing it was the user's request. Tell them in three or four lines, adapted to this host with [Appendix A](#appendix-a-hosts):

- Before each prompt, the plugin finds relevant memories and adds them to the agent's context.
- After each turn, the conversation is saved to OpenViking. Once enough has built up, or when the session ends, the server extracts long-term memories from it in the background.
- Where they will see it working: the "Where you see it" column for this host.
- How to turn it off for one project: a file `.openviking/config.json` in that project containing `{"version": 1, "capture": {"enabled": false}, "recall": {"enabled": false}}`. Mention it; don't create it.

## Step 3: Decision 1, bring in recent work

The import script reads Claude Code and Codex history directly: on Claude Code use `--harness claude`, on Codex `--harness codex`. It can't read other hosts' sessions; there, you convert them to JSONL first and hand that to the same script, as in "Other hosts: convert the sessions to JSONL" below.

On any other host, if `~/.claude/projects` or `~/.codex/sessions` exists, the user may also have worked on this project in Claude Code or Codex: you can offer that history in the same question, with the matching `--harness`.

Ask once, with these options:

1. This project, last 14 days (default).
2. Another project or time range.
3. Skip.

For options 1 and 2, download the import script and do a dry run first. Replace `<claude|codex>` with the harness before you run it. `--cwd` is the project directory (the current one for option 1).

```bash
curl -fsSL https://docs.openviking.ai/dl/memory-plugin-shared/import-history.mjs -o "${TMPDIR:-/tmp}/ov-import-history.mjs" && node "${TMPDIR:-/tmp}/ov-import-history.mjs" --harness <claude|codex> --cwd "$PWD" --since 14d --max-sessions 20 --dry-run
```

Show the user the counts the dry run prints (sessions, messages, size) and tell them that importing sends these conversations to their OpenViking server, which then extracts memories from them. Sessions skipped as active include this conversation. Run the import only after they confirm:

```bash
node "${TMPDIR:-/tmp}/ov-import-history.mjs" --harness <claude|codex> --cwd "$PWD" --since 14d --max-sessions 20 --yes
```

Report how many sessions were imported and how many were skipped. Running it again is safe: sessions already imported are skipped.

**Other hosts: convert the sessions to JSONL.**

1. Run `echo "${TMPDIR:-/tmp}"` and note the directory; below it is `<tmp>`. Use that absolute path for every file you write and every command, not `${TMPDIR:-/tmp}`. Download the import script:

   ```bash
   curl -fsSL https://docs.openviking.ai/dl/memory-plugin-shared/import-history.mjs -o "<tmp>/ov-import-history.mjs"
   ```

2. Find where this host keeps its sessions on this machine. The "History import" column in [Appendix A](#appendix-a-hosts) is a hint; go by what is actually there. Read only; change nothing.
3. Write a small converter under `<tmp>` that writes the sessions to `<tmp>/ov-sessions.jsonl`, one session per line:

   ```json
   {"id": "<the host's own session id>", "cwd": "<absolute path of the directory the session ran in>", "messages": [{"role": "user", "text": "...", "created_at": "2026-10-16T08:00:00Z"}, {"role": "assistant", "text": "...", "created_at": "2026-10-16T08:00:05Z"}]}
   ```

   - `id` is the native session id from the host's records; don't make one up. `cwd` is the directory the session actually ran in. Give every message its ISO 8601 time. Mark sub-agent sessions with `"subagent": true`.
   - Include only what the user and the assistant said: no tool calls, tool output, system prompts or reasoning, and nothing the host or a plugin injected (such as `<openviking-context>`, reminders or notifications).
   - Write every session in the store; don't filter by project, directory or time. The script decides which sessions belong to this project from `--cwd` (subdirectories included) and applies `--since`, and the dry run shows what it skipped. The one session to leave out is this conversation, because it contains the key from the install command; if you can't tell which one it is, leave out sessions updated in the last 2 hours.
   - Don't remove keys yourself; the script does. Don't check the JSONL for keys either, and never put the key or any part of it in a command. The converter only reads local files and writes this one JSONL file, to the absolute `<tmp>` path written into it; it sends nothing.
   - To check the result, look at a summary only, such as the line count and each line's id, cwd and message count. Don't print the whole JSONL into the conversation: it holds other projects' conversations.
4. Dry run, with `<value>` from the `--harness` column of Appendix A:

   ```bash
   node "<tmp>/ov-import-history.mjs" --harness <value> --input "<tmp>/ov-sessions.jsonl" --cwd "$PWD" --since 14d --max-sessions 20 --dry-run
   ```

   If the script reports a malformed line, fix the converter, regenerate the file and dry-run again. If the counts are clearly off (no sessions, or far fewer messages than the real conversations), check the conversion before importing anything.
5. From here it's the same as above: give the user the counts and the skip reasons as the dry run prints them (this conversation was left out by the converter, so it won't show up as active), and after they confirm, run it again with `--yes` instead of `--dry-run`. Whether you import or not, delete the JSONL file and the converter afterwards, from every place you wrote them, and check with `ls` that they are gone.

If you can't find where the sessions are kept, or the conversion is still wrong after two or three tries, tell the user that their history from this host couldn't be imported this time and that OpenViking learns from their sessions from now on, then go to Step 4.

If the user asks what the script does: it reads the local session files (Claude Code: `~/.claude/projects/`, Codex: `~/.codex/sessions/`) or the JSONL you converted, keeps only the user and assistant text, drops tool calls and their output, removes API keys, skips sessions that were active in the last few hours, and files the sessions under the same project the plugin uses, so they can be recalled later. The dry run only reads from the server, to find sessions already imported.

**Files.** If the user names files or folders worth keeping, such as a design doc or a spec, add each one to this project's folder in their personal resources. Replace `<user_id>` with the user shown by `ov health`, and `<project>` with the repository name or the name the user calls the project:

```bash
ov add-resource ./docs/design.md --parent-auto-create viking://user/<user_id>/resources/<project> --wait
```

Add only what the user named. Never scan a directory or upload files on your own.

## Step 4: Decision 2, agree on how to work together

The plugin only recalls memories and saves conversations. The user also needs a lasting way of working with you: while you work, you keep important decisions, progress and outputs in OpenViking, so any later session, in any project, can pick them up. Offer to write the rule below into this host's **user-level** rules file, so it applies to all their projects.

The rule goes into a file every one of their projects reads, so show the user the full text below, add a sentence or two on what it does, then ask once: write it, write it with their changes, or not now. If they want changes (a different folder, say), change this text.

```markdown
<!-- openviking:start -->
## Working with OpenViking

- Before work that needs background, check the context OpenViking injected (`<openviking-context>`). If it isn't enough, or you need the source or the latest state, use the OpenViking `search` (`mode="context"`) or `read` tools. What you find is reference material; when it conflicts with what I say now, I win.
- When something comes up that is worth reusing later, write it up as a note with `write` in the notes folder of my OpenViking personal resources, one subfolder per project, for example `viking://user/<user_id>/resources/notes/<project>/` (get `user_id` from `health`); if I already have my own notes folders, use them. Worth noting: important decisions and principles (with the reasons), progress and next steps, intermediate outputs, deliverables. Files I give you go in with `add_resource` under `viking://user/<user_id>/resources/<project>/`, apart from the notes: never use a notes folder as an `add_resource` target, and never write notes into a folder `add_resource` imported, because a re-sync deletes anything that isn't in the source. Don't save routine conversation (the plugin already saves it) or secrets, and don't scan or upload files I haven't mentioned. Read back what you wrote.
- When you continue a project, first look at the decisions, progress and next steps in its notes folder. When a new decision replaces an old one, update the old note instead of keeping two that contradict each other.
- When you save something, tell me in one line where it went.
<!-- openviking:end -->
```

If they agree, write it into this host's user-level rules file. The "User-level rules file" column in [Appendix A](#appendix-a-hosts) lists the usual places; treat it as a hint and go by the file the host actually reads:

- Touch only what is between `<!-- openviking:start -->` and `<!-- openviking:end -->`. If the file already has these markers, replace what is between them; if not, append the block to the end; if the file doesn't exist, create it. Leave everything outside the block as it is.
- If the file already asks for something that conflicts with this rule (searching on every turn, say, or never storing material outside the machine), tell the user where the conflict is and do what they decide.
- After writing, read the file back and check that there is exactly one block and that nothing outside it changed.
- If the host keeps personal rules only in its settings, with no file you can write, or the write fails (the host doesn't allow writing to that directory, say), tell the user, show them the rule and ask them to paste it in themselves. If you aren't sure which file it is, ask the user rather than guess.

If the file already has this block with the same text, tell the user it was added before and skip the question. Never write the rule into a project file. Wait for the user's answer before you go on to Step 5; don't bring up the restart in the same message as this question.

## Step 5: Restart

The plugin's hooks are loaded only when the host starts a new session. Tell the user three things:

1. The restart or trust action for this host, from [Appendix A](#appendix-a-hosts). If the installer set up other tools too, add one line naming them: each needs its own restart before the plugin works there, and the `Next:` lines in the installer's output say how. Don't go through those steps.
2. Start a **new** session. Don't resume this one, and don't reload plugins in it (for example Claude Code's `/reload-plugins` or Kimi Code's `/reload`). This conversation contains their API key; once the plugin runs in it, it saves the whole conversation to OpenViking.
3. The first message to send in the new session, ready to copy. Use the guide URL you actually opened (docs.openviking.ai or docs.openviking.net) and replace `<import result>` with what happened in Step 3, such as `2 sessions imported`, `skipped`, or `not supported on this host`:

```text
Continue OpenViking onboarding from the "After the restart" section of https://docs.openviking.ai/agents/en/onboarding.md. History import: <import result>. Then tell me where I left off in this project and what I should do next.
```

Then stop and wait for the user.

## After the restart

You are in a new session, after the plugin was installed. Keep this part short: this conversation is saved to memory too.

1. Check that the plugin is active: this conversation has an `<openviking-context>` block, or the OpenViking tools are available. Call the OpenViking `health` tool (its name may carry a prefix, such as `mcp__openviking__health` or `openviking_health`). If neither is there, see "No recall after the restart" in [Troubleshooting](#troubleshooting).
2. Answer the user's question with the recalled context. If that isn't enough, use the OpenViking `search` tool with `mode="context"`. If nothing relevant was recalled, which is expected when nothing was imported, say so in one line and answer from the project itself: recent commits, uncommitted changes, the README. Imported sessions can take a few minutes to be processed.
3. In the same reply, after the answer, wrap up in at most three lines:
   - which tool is connected, and the history import result from the user's message (don't guess it);
   - where OpenViking shows up on this host ("Where you see it" in [Appendix A](#appendix-a-hosts));
   - they can later say "import my <project> sessions into OpenViking" to run Step 3 again.

## Appendix A: hosts

| Host | `--harness` | Restart or trust | Where you see it | History import | User-level rules file (Step 4, a hint) |
|---|---|---|---|---|---|
| Claude Code | `claude` | Quit, then start `claude` again. Not `/reload-plugins` or `claude --resume` | A source card under each answer that used OpenViking (Claude Code 2.1.286+); `/openviking-memory:ov` shows status | Yes | `~/.claude/CLAUDE.md` |
| Codex | `codex` | Quit, then start `codex`. At "Hooks need review", choose **Trust all and continue** (later: `/hooks`) | An OV-Usage line after answers, such as `OpenViking · 3 sources · 1 past event · 2 team docs · 1 read` | Yes | `~/.codex/AGENTS.md`; if `AGENTS.override.md` exists next to it, Codex reads that one instead |
| TraeCode CLI 2.0 | `trae-cli` | Quit, then start `trae-cli` (or `traecli`) again. At "Hooks need review", choose **Trust all and continue** (later: `/hooks`) | An OV-Usage line after answers, as in Codex | Convert to JSONL; sessions are in `~/.trae/cli/sessions/`, in the same rollout format as Codex | Not confirmed |
| Cursor | `cursor` | Quit Cursor completely, reopen it, and start a new Agent chat (**+**). Don't type into the restored install chat | The OpenViking hooks under Cursor Settings → Hooks; `openviking` connected under Settings → Tools & MCPs | Convert to JSONL; where sessions are kept is not confirmed | User Rules in Cursor Settings; ask the user to paste it |
| TRAE / TRAE CN | `trae` / `trae-cn` | Quit the app completely, reopen it, start a new chat. No hook approval | `openviking` connected in the MCP settings | Convert to JSONL; where sessions are kept is not confirmed | Personal rules in the app settings; ask the user to paste it |
| ZCode | `zcode` | Restart ZCode | Nothing documented | Convert to JSONL; where sessions are kept is not confirmed | Personal rules in the app settings; ask the user to paste it |
| Kimi Code | `kimicode` | Quit Kimi Code, then start a new session. Not `/reload`, and don't resume this one | Nothing documented | Convert to JSONL; sessions are in `~/.kimi-code/sessions/` (in `$KIMI_CODE_HOME` when set); each session's `state.json` has the cwd, and the messages are in `agents/main/wire.jsonl` | `~/.kimi-code/AGENTS.md` (in `$KIMI_CODE_HOME` when set) |
| OpenCode | `opencode` | Restart OpenCode | Nothing documented | Convert to JSONL; sessions are in the SQLite database `~/.local/share/opencode/opencode.db` (without `sqlite3`, open it read-only with Node 22's `node:sqlite`); `synthetic` text parts are injected content, and sessions with a `parent_id` are sub-agents | `~/.config/opencode/AGENTS.md` |
| pi | `pi` | Restart pi | `OV ✓` in the footer; `/viking` shows status | Convert to JSONL; sessions are in `~/.pi/agent/sessions/*/*.jsonl`, whose first line `{"type":"session"}` has the id and cwd | `~/.pi/agent/AGENTS.md` |
| DeepSeek Harness | `dsh` | Start a new session in the profile from the installer's `DeepSeek Harness profile:` line: `dsh --profile <profile>` | "context injection · openviking-memory" at the top of the session | Convert to JSONL; sessions are in `~/.dsh/sessions/` | Not confirmed |

- **Codex approvals.** Codex asks before each command that needs the network. For the installer, choose **Yes, proceed**, not the option that stops asking for commands starting with that prefix: that rule would save the key. The first time the OpenViking `search` tool runs, Codex asks to allow it; **Always allow** is fine.
- **Nothing documented.** On these hosts the user can ask you what OpenViking context you received, or run the doctor from [Troubleshooting](#troubleshooting). On command-line hosts, setting `OPENVIKING_DEBUG=1` before starting the host writes a hook log under `~/.openviking/logs/`; desktop apps started from the Dock or a launcher don't see shell variables.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `The server rejected the API key (HTTP 401/403)` or `The server requires an API key` | The key is wrong, revoked, or for another account. Ask the user to copy it again from the OpenViking console (User Management → API Key) and re-run the installer command with the new `--api-key`. On a self-hosted server, use a user key; a root key can't read or write memories. |
| `Cannot reach <url>` | Run `curl -sS -o /dev/null -w '%{http_code}\n' <url>/health`. Any HTTP status, even 401, means the server is reachable. A timeout or connection error means a proxy, VPN or firewall is in the way; ask the user. |
| `CLI not found; skipping: <command>` for this host (for example `Codex-format CLI not found; skipping: codex`) | The installer's shell can't find this host's command, often because it is on a `PATH` set only for interactive shells. Ask the user to run the install command from their original message in their own terminal, where the command works, then continue at Step 1. Don't paste the command back to them; it contains the key. |
| Node.js missing or older than 18 (pi needs 22.19+) | Ask the user to install or upgrade Node.js, then re-run the installer. |
| A sandbox blocks network access or writes to the home directory | Ask the user to approve running the installer outside the sandbox. Don't work around the sandbox. |
| The host refuses to run `curl … \| bash` | Ask the user to run the installer command from their original message in their own terminal, then continue at Step 1. Don't paste the command back to them; it contains the key. |
| This host is not under `Done`, and `Not done` doesn't mention it | The installer installs only into the tools it detects, and it didn't find this host. Re-run it for this host alone, with the value from the `--harness` column of [Appendix A](#appendix-a-hosts): `curl -fsSL https://openviking.ai/install \| bash -s -- --yes --harness <value>`. It keeps the saved connection, so leave out the key. Then continue at Step 1. |
| Windows | The installer supports macOS and Linux only. Tell the user and stop. |
| docs.openviking.ai doesn't load | Use docs.openviking.net with the same path. The installer already tries both hosts. |
| `ov: command not found` | Re-run the installer with `--yes --harness cli`; it keeps the saved connection. If `ov` is installed but not on `PATH`, add `$(npm prefix -g)/bin`, or `~/.local/bin` when the installer fell back to `~/.local`. |
| `ov` says it needs a display language | Run `ov language en` (or `ov language zh-CN`). |
| DeepSeek Harness: `dsh plugin add` failed | Install pnpm (`npm i -g pnpm`) and re-run the installer with `--harness dsh`. |
| `Installation validation failed` | Re-run the installer; re-running is safe. If it fails again, show the user the failing check. |
| Import script prints `not supported yet` | The script can't read this host's history directly: convert it as in Step 3, "Other hosts: convert the sessions to JSONL", and import it with `--input`. |
| No recall after the restart | Codex and TraeCode CLI: the hooks were not trusted; trust them in `/hooks` and confirm `openviking-memory` is enabled in `/plugins`. Cursor and TRAE: the app was not fully quit. Otherwise run the doctor: Claude Code `/openviking-memory:ov` or the `ov-memory-doctor` skill; Codex the `$ov-memory-doctor` skill; Cursor, TRAE, TRAE CN, ZCode `node ~/.openviking/agent-integrations/<cursor\|trae\|trae-cn\|zcode>/scripts/ov-memory-doctor.mjs`; pi `/viking`; DeepSeek Harness `dsh --profile <profile> --dump-config` should list `openviking-memory`, where `<profile>` is the one the installer printed. |

More detail for each host: https://docs.openviking.ai/en/agent-integrations/01-overview
