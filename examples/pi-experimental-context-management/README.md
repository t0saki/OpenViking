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

## Configuration

`config.json` keeps the upstream fields, minus `captureMode` — capture here is
always faithful, so the semantic/keyword switch had no reader and is gone —
plus one nested `contextWindow` block for the agent-driven windows. Every
number below is rounded and clamped on load: an out-of-range value is pulled to
the nearest bound, and one that is not a number at all (`"soon"`, `{}`, or a
missing key) falls back to the default. `hardPercent` is clamped first and then
raised to `softPercent` if it was configured lower, so the hard reminder can
never fire before the soft one. Unknown keys inside `contextWindow` are
dropped, and a `contextWindow` that is not an object is ignored entirely.

| Field | Default | Range | Description |
| --- | --- | --- | --- |
| `contextWindow.resetDeadlineMs` | `60000` | 5000–600000 | Whole-reset budget: sync, barrier, commit and the wait for the archive overview |
| `contextWindow.archivePollMs` | `2000` | 250–30000 | Delay between `.overview.md` polls while the server builds Working Memory |
| `contextWindow.overviewRefreshMaxAttempts` | `20` | 0–200 | Non-blocking retries at `turn_end` when a window opened before its overview was ready |
| `contextWindow.overviewBudget` | `3000` | 100–50000 | Token budget for the Working Memory block in the window header |
| `contextWindow.notesBudget` | `1500` | 100–20000 | Token budget for the agent's handoff notes in the window header |
| `contextWindow.pendingRequestBudget` | `400` | 0–8000 | Token budget for the last user message carried into the new window |
| `contextWindow.softPercent` | `70` | 10–99 | Usage that earns one soft "checkpoint, then reset" reminder per window |
| `contextWindow.hardPercent` | `85` | 10–99 | Usage that earns one hard "reset now" reminder; never below `softPercent` |
| `contextWindow.idleGapMinutes` | `30` | 0–1440 | Idle gap after which the status line suggests considering a new window |
| `contextWindow.statusEveryTurn` | `true` | boolean | Append a one-line context status after every user prompt |
| `contextWindow.historyItemMaxChars` | `8000` | 500–100000 | Per-item cap for `history` reads out of an archived `messages.jsonl` |

Two environment variables override the block for a single run:
`OPENVIKING_CONTEXT_STATUS_EVERY_TURN` (`0`/`1`, `true`/`false`, `on`/`off`,
`yes`/`no`; anything else leaves the configured value alone) and
`OPENVIKING_CONTEXT_RESET_DEADLINE_MS` (clamped like the file value; an empty
value is ignored). The same spellings work for `statusEveryTurn` in
`config.json`.

## Tests

```bash
node --test examples/pi-experimental-context-management/tests/*.test.mjs
```
