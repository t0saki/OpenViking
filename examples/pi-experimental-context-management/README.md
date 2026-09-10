# pi × OpenViking — experimental context management

**EXPERIMENTAL.** This is a fork of [`examples/pi-coding-agent-extension`](../pi-coding-agent-extension),
kept as the base for agent-driven context windows: the model decides when the
current window is done, the extension archives it to OpenViking and opens a
fresh one carrying a handoff note plus the server-written Working Memory, and
the model can read the closed windows back. It is a pi-side take on Codex's
`features.context_management.experimental_mode`.

Nothing of that is here yet. This directory currently holds the skeleton:
turn sync, recall, profile injection and the six `viking_*` tools, with the
takeover mode removed. `new_context` / `history` / `get_context_remaining` and
the window core land in later phases, and this README grows with them.

## Do not run it next to the openviking extension

Both extensions register the same `viking_*` tools and both sync the same
OpenViking session, so loading them together duplicates every tool and gives
one session two writers. On startup this one probes `pi.getAllTools()` for
`viking_search`; if the other extension already registered it, this extension
disables itself and says so. Disable the other one (`enabled: false` in its
`config.json`, or drop it from `settings.json`'s `packages`) before installing
this.

## What differs from the upstream extension

- No takeover mode: no `takeover.ts`, no boundary state machine, no
  threshold-driven commit inside `syncBranch`. Archive boundaries will be
  written only by the reset path and by the pi compaction fallback.
- Tool output reaches the archive. `normalizeRole` recognises pi's
  `toolResult` messages and captures them as `[tool-result <toolName>] …`
  user turns, bounded by `captureToolMaxChars`; `captureToolResults` defaults
  to true here. Without this the promise that a closed window can be read back
  would be empty.
- Capture is always faithful — the archive is the only way back to a window
  that is no longer in context, so nothing is filtered but plugin chatter and
  slash commands.
- No recall injection ledger: replaying historical recall blocks needed
  `SessionManager.buildContextEntries()`, which pi 0.80.3 does not expose.
  `injectRecall` prepends this turn's block to the newest user message only,
  and skips any message that already carries an `<openviking-context` block.
- `viking_archive_expand` is gone: it read `viking://session/{id}`, a
  namespace the server does not serve. The `history` tool replaces it.
- `sync.flushForTakeover()` is now `sync.flushBarrier({budgetMs})`, so a reset
  can cap how long it waits for the pending queue to drain.

## Layout

| Path | What it is |
| --- | --- |
| `index.ts` | Extension entry: event handlers, coexistence guard, `/viking` command |
| `client.ts` | OpenViking HTTP client |
| `sync.ts` | Session sync, disk pending queue, `flushBarrier`, `commit` |
| `recall.ts` | Per-prompt recall search and injection |
| `config.ts`, `config.json` | Config and credential resolution |
| `tools.ts` | The six `viking_*` tools |
| `lib/text-budget.mjs` | Pure token estimation / truncation helpers |
| `lib/capture-adapter.mjs` | Branch entries → OpenViking message payloads |
| `lib/uri-guard-adapter.mjs` | Blocks builtin file tools on `viking://` URIs |
| `shared/` | Generated from `examples/memory-plugin-shared/lib` — do not edit |

## Tests

```bash
node --test examples/pi-experimental-context-management/tests/*.test.mjs
```
