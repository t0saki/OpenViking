# Agent-managed context windows

**EXPERIMENTAL.** This document describes the context-window mode of
`examples/pi-experimental-context-management`: how OpenAI's Codex does it, what
this extension maps onto OpenViking, what actually happens during a reset, and
the exact text the model sees.

Everything below is taken from the code in this directory:
`lib/context-window-core.mjs` (the pure state machine), `context-window.ts`
(the pi/OpenViking adapter), `tools.ts` (the three tools), `index.ts` (event
wiring and the static guidance) and `config.ts` (the `contextWindow` block).

---

## 1. How OpenAI/Codex does it

Codex ships this as `features.context_management.experimental_mode` on GPT-6
Astra. Turning it on enables the `token_budget` feature and sets
`use_history_notes_extension = true`. Instead of compressing the conversation
into a summary, the model keeps its own notes, decides itself when the current
window ends, and can search the messages and tool output of the windows that
already closed.

- **Storage lives on OpenAI's backend** (`alpha/notes/v2/*`,
  `alpha/history/v2/*`), which is why the mode only works for ChatGPT
  Plus/Pro logins.
- **Model-side tools**:
  - the `notes` namespace — `list_files_by_prefix`, `read_file`,
    `search_contents`, `append_to_file`, `write_file` — a private,
    cross-window scratchpad on virtual paths;
  - the `history` namespace — `list_windows`, `list_items`, `read_item`,
    `search_contents` — reads items of earlier windows by window id + item id;
    every non-assistant message carries a trailing `[id: ...]`;
  - `functions.new_context` — **no parameters**, described as "Start a new
    context window. Does not clear, reset, or otherwise affect environment
    state." It only sets a flag; after the current sampling finishes, the
    session is rebuilt as system/developer prompt + world state + the retained
    client developer messages + a `<context_window>` block (current and
    previous window id plus a `thread_hint` pulled from the notes backend,
    ≤ 4000 bytes). **No LLM summarization happens at any point.**
  - `get_context_remaining` — returns `{tokens_left}`.
- **Prompts** (the `token_budget` entry of `models-manager/models.json`):
  a `guidance_message` telling the model to maintain a checkpoint in notes
  (goal / decisions / progress / learnings / next steps / ids of unfinished
  user requests), to plan with `get_context_remaining`, to treat the presence
  of a previous-window id as proof that a reset happened (read the checkpoint
  first, then fill gaps with `history`), and to treat notes/history as internal
  bookkeeping; a `reminder_message_template` fired once when fewer than 6144
  tokens remain ("save notes, then call new_context"); and an
  `auto_compact_fallback_prompt` used when nothing is left, which grants an
  extra 16384-token buffer and forces exactly one notes write followed by
  `new_context`.
- **Compaction path**: in `token_budget` mode `run_compact_task_inner` calls
  `start_new_context_window` — the compaction lifecycle still runs, but the
  summarization step is gone.

## 2. Mapping Codex → this extension

| Codex | Here | Notes |
| --- | --- | --- |
| History backend (`alpha/history/v2`) | The OpenViking session message stream, archived per window to `<session root>/history/archive_NNN/messages.jsonl` | Written by `POST /sessions/{id}/commit` |
| Notes backend (`alpha/notes/v2`) | The `notes` parameter of `new_context` | No separate notes tool in v1 (see limitations) |
| `thread_hint` in the window block | The server-written Working Memory (`.overview.md`) plus the model's own handoff notes | Working Memory has 7 sections: Session Title, Current State, Task & Goals, Key Facts & Decisions, Files & Context, Errors & Corrections, Open Issues |
| `history.list_windows` | `history {"action":"list_windows"}` → `GET /fs/ls?uri=<root>/history` | Window ids `w1…wN` are assigned from the archive list, oldest first |
| `history.list_items` / `read_item` | `history {"action":"list_items"}` / `{"action":"read_item"}` → `GET /content/read?uri=<archive>/messages.jsonl` | Item ids are the 0-based line index inside that archive: `w2:14` |
| `history.search_contents` | `history {"action":"search_contents"}` → `POST /search/grep` over `<root>/history` | Case-insensitive; hits in `messages.jsonl` are turned into `read_item` pointers |
| `functions.new_context` (no parameters) | `new_context` **with** `reason`, `notes`, `next_steps?` | The parameters replace the missing notes backend |
| `get_context_remaining` → `{tokens_left}` | `get_context_remaining` → tokens left plus window age, turn count, idle gaps, archive readiness and an `advice` line | pi can only estimate tokens, so the report says whether the number is exact or estimated |
| Native window rebuild after sampling | A virtual cut in pi's `context` hook | Nothing is deleted on disk; only the message list of each provider request is rewritten |
| `guidance_message` | `CONTEXT_WINDOW_GUIDANCE`, appended to the system prompt | §5 below |
| `reminder_message_template` | Soft/hard reminders, one of each per window | §5 below |
| `auto_compact_fallback_prompt` | The pi-compaction fallback in `session_before_compact` | §8 below |

**What cannot be mirrored:**

- **No developer role in pi.** The window header can only be a `user` message,
  so it is wrapped in `<openviking-context source="context-window">` and, when
  the first surviving message is already a user message, merged into it to
  avoid two user messages in a row.
- **No exact token count.** `ctx.getContextUsage()` reports the last assistant
  usage plus a chars/4 estimate over the *untransformed* session, so right
  after a reset it is too high; the core keeps its own estimate over the
  transformed list and marks numbers as `estimated` when it uses them.
- **Notes are not private.** Anything mirrored into OpenViking gets indexed and
  becomes searchable, so v1 carries notes as a tool parameter instead of
  building a separate notes namespace.
- **No parameterless `new_context`.** Without a notes backend, the reason and
  notes have to arrive with the call.

## 3. What a reset does at runtime

`new_context` is `executionMode: "sequential"` and its `execute` is awaited, so
the whole pipeline blocks inside the tool call, bounded by one deadline
(`contextWindow.resetDeadlineMs`, default 60s).

1. **Guards.** Refuse if a reset is already running, if the call has no tool
   call id, if the OpenViking session id is unknown, or if the client is not
   connected. Refuse also when a previous commit failed in transport and the
   server is still unreachable.
2. **`sync.syncBranch(branch)`** — pi has already persisted the assistant
   message that issued this call, so the archive reaches exactly the reset
   point. A branch that could not be fully delivered refuses the reset.
3. **`sync.flushBarrier({budgetMs})`** — the disk pending queue must be empty
   (budget: whatever is left of the deadline, clamped to 2–15s). A queue that
   does not drain refuses the reset and reports how many messages are stuck.
4. **Post the handoff note** with `client.addMessage(sid, "assistant", …)`,
   directly rather than through the queue, so it lands *before* the commit and
   is archived into this window. The server's Working Memory prompt writes for
   "the assistant that continues", so it folds the note into Current State /
   Open Issues — the reason ends up in the summary without any server change.
5. **`sync.commit({queueOnFailure:false, keepRecentCount:0})`.**
   `status: "skipped"` (nothing to archive) leaves the window untouched;
   anything else without an `archive_uri` refuses.
6. **Point of no return.** Once the commit returns an `archive_uri` the
   OpenViking session is empty, so every path below must open a window.
   `archiveId = basename(archive_uri)`; the core polls
   `GET /content/read?uri=<archive_uri>/.overview.md` every
   `archivePollMs` until the deadline (404 while commit phase 2 runs, 200 with
   the Working Memory as the body), checking `GET /tasks/{task_id}` every 5th
   attempt so a `failed` task ends the wait early. Timeout, task failure and
   `signal.aborted` all still open the window, with a degraded header.
7. **Open the window**: freeze `headerText`, `windowIndex += 1`, record
   `anchorToolCallId`, `archiveId`, `overviewReady`, the sibling tool names and
   the current sync watermark, then `pi.appendEntry("ov-context-window", state)`
   and update the status bar.
8. **The cut** happens in the `context` hook of the *next* sampling — which is
   the one right after this tool batch. `computeCutRange` finds the toolResult
   whose `toolCallId` is the anchor, walks back to the assistant message that
   issued it, and drops the whole contiguous run of toolResults it belongs to
   (so a parallel batch never leaves an orphan `tool_result`, which providers
   reject). Everything before that is dropped as well; a stale
   `[context-reminder]` message at the head of the kept segment is dropped too.
   The frozen header goes in front.
9. **Degraded follow-up**: at every `turn_end` a window with
   `overviewReady === false` reads `.overview.md` once more, up to
   `overviewRefreshMaxAttempts` times. Success rebuilds the header once (one
   cache miss) and re-persists; exhausting the attempts marks the Working
   Memory `unavailable` and rebuilds the header with that wording.

**The reset tool's own result is inside the discarded range.** On success the
model never sees the text `new_context` returned — everything it needs has to
be in the window header. Only a refusal (nothing was cut) is actually read.

**Anchor missing → boundary released.** If the anchor is not in the branch any
more (a fork, `/tree`, a native pi compaction), the transform returns the
messages untouched, logs once and disarms. That is the single invariant that
keeps a failure mode of "no reset" rather than "corrupted request".

## 4. Exact prompt texts

### 4.1 Static guidance (appended to the system prompt every turn, `index.ts`)

```text
<context-window-management>
You manage your own context window in this session. When the conversation grows it is not summarized behind your back: you decide when the current window ends, and OpenViking archives it so you can read it back afterwards.

Three tools do this:
- new_context — archive the current window and continue in a fresh one. OpenViking generates a Working Memory of the archived window, and your next window opens with that Working Memory, your handoff notes and the user's most recent request.
- history — read windows that were already archived: list_windows, list_items, read_item, search_contents. Closing a window loses nothing; it only stops being in front of you.
- get_context_remaining — how much room is left, how long this window has been open, and how long it has been since the user's last message.

Start a new window when:
- a phase of the work is finished and its details no longer matter for what comes next;
- the user switches to an unrelated topic or a different area of the code;
- the user comes back after a long idle gap with something new;
- the status line or get_context_remaining shows the window filling up — around 70% is the moment to act, and do not push past 85%, because after that the harness compacts the conversation for you and your notes are never written.

Do not start a new window:
- in the middle of an edit sequence you have not verified;
- immediately after a reset;
- to get away from a problem you have not solved — the new window carries the same problem with less information.

Write the notes before you call new_context: the goal, the decisions and why you made them, what is finished, what is in flight, exact file paths and identifiers, and the next steps. Write them for someone who has not seen this conversation, because that is exactly what your next window is. Call new_context alone — tool results from the same batch are discarded together with the old window.

In a new window, read the <openviking-context source="context-window"> block first: it carries the Working Memory, your notes and the request that is still pending. Use history for anything it does not cover, and never ask the user to repeat something an archived window already holds.
</context-window-management>
```

Followed by one tool-list line:

```text
OpenViking tools: viking_search, viking_read, viking_browse, viking_remember, viking_forget, viking_add_resource. Context window tools: new_context, history, get_context_remaining.
```

Unlike Codex, the model is *not* asked to hide this machinery from the user —
the demo is meant to be visible.

### 4.2 Status line (one per user prompt, `customType: "ov-context-status"`)

```text
[context-status] window w2 · 6 turns · ~38.4k/262k tokens (15%) · 47m since your previous message
```

With an estimated token count the percentage carries a marker:

```text
[context-status] window w2 · 1 turn · ~38.4k/262k tokens (15%, estimated) · 12s since your previous message
```

After an idle gap longer than `idleGapMinutes` a second line is appended:

```text
NOTE: 47 minutes passed since the previous user message. If this request starts unrelated work, consider new_context before you begin.
```

It is emitted from `before_agent_start`, so it appears **once per user prompt**,
not once per sampling: it does not update while a tool loop runs.

### 4.3 Reminders (at most one of each per window, `customType: "ov-context-reminder"`)

Soft (`softPercent`, default 70):

```text
[context-reminder] Context window w2 is about 70% full (~184k/262k tokens). Finish or checkpoint the step you are on, then call new_context with complete notes: goal, decisions and why, what is done, what is in flight, exact paths and identifiers, next steps. If the window overflows first, the harness compacts it for you and your notes are never written.
```

Hard (`hardPercent`, default 85, clamped one point below pi's own
auto-compaction line):

```text
[context-reminder] Context window w2 is about 85% full (~223k/262k tokens). Save your handoff notes and make exactly one call to new_context now. If the window overflows first, the harness compacts it for you and your notes are never written.
```

Delivery is `pi.sendMessage(..., {deliverAs: toolResults.length ? "steer" : "nextTurn"})`
from `turn_end`. No reminder is evaluated while `awaitingFirstObservation` is
set (between a reset and the first assistant response of the new window),
otherwise the stale 87% of the closed window would trigger an immediate second
reset.

### 4.4 Handoff note (archived as the last assistant message)

```text
[Context Window Handoff] w2 -> w3
Reason: REASON
Handoff notes:
NOTES
Next steps:
1. a
2. b
```

`Next steps:` is omitted when `next_steps` is empty; an empty `reason` renders
as `(none given)` and empty notes as `(none)`.

### 4.5 Window header — Working Memory ready

```text
<openviking-context source="context-window">
<context_window id="w3" previous="w2" archive="archive_002" opened="2026-09-10T06:41:12Z">
This is a fresh context window. The earlier conversation of this session was archived to OpenViking and is no longer in context. Files, the working directory, running processes and tools are unchanged.

Reason you gave: REASON

Your handoff notes:
<handoff-notes>NOTES</handoff-notes>

Next steps you planned:
1. step one
2. step two

The user's most recent message in the previous window was:
<pending-request>PENDING</pending-request>

Working Memory of the archived window, generated by OpenViking:
<working-memory archive="archive_002">WORKING MEMORY</working-memory>

2 other tool results were discarded with the same batch; re-run them if needed. Tools: bash, read.

To recover anything not covered above: history {"action":"list_windows"} / {"action":"list_items","window":"w2"} / {"action":"read_item","item":"w2:14"} / {"action":"search_contents","query":"..."}.
Continue from the notes above. If the pending request is not finished, resume it now.
</context_window>
</openviking-context>
```

Sections are omitted when empty: no `Reason you gave` without a reason, no
`<handoff-notes>` without notes, no `Next steps you planned` without steps, no
`<pending-request>` when the window carries no user message (or when
`pendingRequestBudget` is 0), and the sibling-tool line only when other tool
results were discarded with the same batch (singular wording: `1 other tool
result was discarded …`). Notes, pending request and Working Memory are
truncated to their token budgets and then get a `\n...(truncated)` marker.

### 4.6 Window header — Working Memory still pending

The Working Memory paragraph is replaced by:

```text
Working Memory for archive_001 is not ready yet: OpenViking is still summarizing it. Rely on your handoff notes above and use history to read the archived messages directly.
<working-memory status="pending" archive="archive_001" stale="true">OLD WORKING MEMORY</working-memory>
```

The stale block only appears when a previous window's Working Memory exists —
on the first reset of a session there is none, and no empty block is written.

### 4.7 Window header — Working Memory unavailable

```text
Working Memory for archive_001 is unavailable: OpenViking could not summarize it. Rely on your handoff notes above and use history to read the archived messages directly.
```

Reached when the archive task reports `failed`, or when
`overviewRefreshMaxAttempts` non-blocking retries are exhausted.

## 5. Tool reference

### `new_context`

Description (verbatim from `tools.ts`):

> Start a new context window. Does not clear, reset, or otherwise affect
> environment state. The conversation so far is archived to OpenViking (which
> generates a Working Memory of it) and your next window starts with that
> Working Memory, your handoff notes and the user's last request. Call it alone
> — tool results from the same batch are discarded with the old window — and
> read the `<openviking-context source="context-window">` block that follows
> instead of this result, which is replaced by the new window.

| Parameter | Type | Required | Meaning |
| --- | --- | --- | --- |
| `reason` | string | yes | One sentence: why this context window should end now |
| `notes` | string | yes | Handoff notes: goal, decisions and why, what is done, what is in flight, exact paths and identifiers, next steps |
| `next_steps` | string[] | no | Ordered next steps for the new window |

`executionMode: "sequential"`. It never sets `terminate`: a refusal has to leave
the agent running, and a success is consumed by the next `context` hook.
Progress while waiting for the archive is reported through `onUpdate` as
`waiting for the Working Memory of archive_002 (attempt 3)`.

### `history`

| Parameter | Type | Used by | Meaning |
| --- | --- | --- | --- |
| `action` | `"list_windows" \| "list_items" \| "read_item" \| "search_contents"` | all | What to do |
| `window` | string | list_items, read_item | `"w2"` or `"archive_002"` |
| `item` | string | read_item | `"w2:14"`, or `"14"` together with `window` |
| `query` | string | search_contents | Text or regular expression |
| `offset` | number | list_items | First item index, default 0 |
| `limit` | number | list_items | How many items, default 40, capped at 200 |

`list_windows` lists archives newest-id-last with their directory abstract
(`(Working Memory not ready)` until commit phase 2 finishes) and appends the
open window as `(current)`. `list_items` renders one line per message,
`[id: w2:14]  role  timestamp  first 200 chars`. `read_item` returns one
message clipped to `contextWindow.historyItemMaxChars`. `search_contents` greps
every archive case-insensitively, shows at most 20 matches grouped by window,
and turns each `messages.jsonl` hit into a `read_item` pointer (line number − 1
is the item index, because the file holds one message per line).

One tool with an `action` enum rather than four tools: pi's tool namespace is
flat, so four `history_*` tools would cost four slots in every request.

### `get_context_remaining`

No parameters. Read-only; renders the `statusSnapshot`:

```text
tokens_left: 224k
used: ~38.4k / 262k (15%)
window: w2, opened 12m ago
turns in this window: 6
since the user's last message: 47m
gap before that message: 3m
archives: 1 (latest archive_001, Working Memory ready)
accuracy: exact
advice: no action needed
```

`accuracy` is `estimated` when the number comes from the core's own estimate
(right after a reset, or when pi reports `tokens === null`). `advice` is one of
`no action needed`, `past the soft threshold: update your notes and reset at the
next stopping point`, `save your notes and call new_context now`.

The six `viking_*` tools (`viking_search`, `viking_read`, `viking_browse`,
`viking_remember`, `viking_forget`, `viking_add_resource`) are unchanged from
the non-experimental extension. `viking_archive_expand` is gone — `history`
replaces it.

## 6. Configuration (`contextWindow` block)

| Key | Default | Range | Effect |
| --- | --- | --- | --- |
| `resetDeadlineMs` | 60000 | 5000–600000 | Whole-reset budget: sync, barrier, commit and the wait for `.overview.md` |
| `archivePollMs` | 2000 | 250–30000 | Delay between `.overview.md` polls |
| `overviewRefreshMaxAttempts` | 20 | 0–200 | Non-blocking `turn_end` retries for a degraded window |
| `overviewBudget` | 3000 | 100–50000 | Token budget for the Working Memory block in the header |
| `notesBudget` | 1500 | 100–20000 | Token budget for the handoff notes in the header |
| `pendingRequestBudget` | 400 | 0–8000 | Token budget for the carried-over user message |
| `softPercent` | 70 | 10–99 | Usage that earns the soft reminder |
| `hardPercent` | 85 | 10–99 | Usage that earns the hard reminder; raised to `softPercent` if configured lower |
| `idleGapMinutes` | 30 | 0–1440 | Idle gap after which the status line adds its NOTE line |
| `statusEveryTurn` | true | boolean | Emit the status line after every user prompt |
| `historyItemMaxChars` | 8000 | 500–100000 | Per-item cap for `history read_item` |

Every number is rounded and clamped on load; an out-of-range value is pulled to
the nearest bound and a non-numeric one falls back to the default. Unknown keys
inside `contextWindow` are dropped, and a `contextWindow` that is not an object
is ignored entirely.

Two environment variables override the block for one run:
`OPENVIKING_CONTEXT_STATUS_EVERY_TURN` (`0/1`, `true/false`, `on/off`,
`yes/no`) and `OPENVIKING_CONTEXT_RESET_DEADLINE_MS` (clamped like the file
value).

Both thresholds are additionally clamped at runtime to one point below pi's own
auto-compaction line, `1 - reserveTokens/contextWindow`, read from
`compaction.reserveTokens` in `<cwd>/.pi/settings.json` and
`~/.pi/agent/settings.json` (default 16384). On a 32k model that line sits near
50%, so without the clamp the hard reminder would never fire before pi compacted
and the notes would never be written.

## 7. Persisted state

One `pi.appendEntry("ov-context-window", …)` per window open, per Working
Memory upgrade and at shutdown. `restore()` takes the newest entry that belongs
to this OpenViking session and ignores malformed ones and entries of other
sessions.

```json
{
  "version": 1,
  "ovSessionId": "pi-<piSessionId>",
  "windowIndex": 3,
  "anchorToolCallId": "call_…",
  "openedAt": 1789012345678,
  "headerText": "<openviking-context source=\"context-window\">…",
  "reason": "…",
  "notes": "…",
  "nextSteps": [],
  "pendingRequest": "…",
  "archiveId": "archive_002",
  "archiveUri": "viking://user/<uid>/sessions/pi-…/history/archive_002",
  "taskId": "…",
  "overviewReady": true,
  "overviewUnavailable": false,
  "overviewAttempts": 0,
  "previousOverview": "…",
  "siblingToolNames": [],
  "syncedEntryCount": 137,
  "lastResetAt": 1789012345678,
  "lastResetBy": "agent"
}
```

`lastResetBy` is `agent`, `pi-compaction` or `external`. In-memory only, never
persisted: the `resetting` mutex, `remindersSent`, `awaitingFirstObservation`,
`lastWindowTokens`, `turnsInWindow`, `overviewText`. A fresh session starts at
`windowIndex = 1` (w1, not persisted); the first reset opens w2.

Because `headerText` and the anchor are persisted, `pi -c` reproduces the cut
byte for byte in the next process.

## 8. Failure matrix

| Situation | Cut? | What the model gets |
| --- | --- | --- |
| OpenViking unreachable, or no OV session id | no | `Context window NOT reset: OpenViking is unreachable, so this window cannot be archived. Nothing changed; keep working; the harness will compact for you if the window fills up.` |
| Branch not fully delivered to OpenViking | no | `… the conversation so far could not be fully delivered to OpenViking. …` |
| Pending queue did not drain | no | `… the archive barrier did not clear — N captured message(s) are still queued. …` |
| Deadline exhausted before the handoff note | no | `… the reset deadline was exhausted while syncing this window to OpenViking. …` |
| Handoff note rejected | no | `… the handoff note could not be written to OpenViking. …` |
| Commit threw in transport | no | `… the archive commit failed in transport; the archive may or may not exist, so nothing was cut. …` — remembered, so the next attempt re-checks connectivity first |
| Commit refused (no `archive_uri`) | no | `… OpenViking refused the archive commit (trace …). …` |
| Commit `skipped` / `no_messages` | no | `Context window NOT reset: OpenViking had nothing to archive for this window. Keep working; call new_context again once there is something worth archiving.` |
| Aborted before the commit (Esc) | no | `Context window reset was cancelled before anything was archived; nothing changed.` |
| Another reset already running | no | `A context window reset is already in progress.` |
| Commit succeeded, Working Memory slow / aborted / deadline hit | **yes** | Window opens with the `status="pending"` header variant |
| Commit succeeded, archive task `failed` or retries exhausted | **yes** | Window opens with the `unavailable` header variant |
| Commit succeeded, something after it threw | **yes** | Window opens through the recovery path with a `pending` header and an explicit error in the tool result |
| Anchor no longer in the branch at transform time | n/a | Messages pass through untouched; the boundary is released and logged once |

The rule behind the table: **fail closed before the commit, fail open after
it.** Before the commit nothing was archived, so keeping the context is the
safe answer. After it the OpenViking session is empty
(`keep_recent_count: 0`), so refusing would leave the model holding the old
conversation while believing its history is still live.

## 9. Prompt-cache rules

- `headerText` is built **once** per reset and frozen. Every later provider
  request repeats the same bytes, so the prefix stays cacheable.
- The static guidance and the tool-list line are constant for the whole
  session, so appending them to the system prompt moves no cache boundary.
- The status line and the reminders are persistent custom messages appended
  after a user message, not rewritten in place.
- A degraded window is upgraded at most once (`overviewReady` false → true, or
  → `unavailable`): exactly one cache miss per window, never a poll-driven
  rewrite.
- `pi -c` rebuilds the identical header from the persisted entry, so a resumed
  session can keep the cache.
- The one deliberate exception is `absorbExternalCompaction`, which drops the
  anchor entirely — pi already cut natively at that point.

## 10. pi-compaction fallback

If the model never resets and pi's own threshold hits first,
`session_before_compact` takes over (`ContextWindowCore.handleBeforeCompact`):

1. If our own header exists, the last reset was less than 60s ago
   (`recentResetGuardMs`) and nothing new has been synced since, return the
   current header as the summary **without committing again** — this catches a
   provider hiccup that made pi estimate from the untransformed list.
2. Otherwise: `syncBranch(branchEntries)` → `flushBarrier` → post a handoff note
   with `reason = "automatic: pi compaction threshold"` and the last notes the
   model wrote → `commit({keepRecentCount: 0})` → wait for `.overview.md` under
   the same deadline. Return `{compaction: {summary: headerText,
   firstKeptEntryId, tokensBefore, details: {source: "openviking", reason:
   "pi-compaction"}}}`.
3. `firstKeptEntryId` is picked so pi can never resurrect messages the header
   already declares archived: use `preparation.firstKeptEntryId` when it sits
   after the anchoring tool batch, otherwise the first entry after that batch,
   otherwise the sentinel `"ov-context-window-reset"` (matching no entry, so pi
   builds a summary-only context).
4. Any failure returns `undefined` and pi writes its own summary.
5. After a successful takeover the virtual anchor is cleared — pi has performed
   a native cut — and `lastResetBy` becomes `pi-compaction`.
6. A compaction produced by anyone else (`session_compact` without
   `fromExtension`) calls `absorbExternalCompaction`: the anchor is dropped,
   the window index advances and `lastResetBy` becomes `external`, which also
   stops the guard in step 1 from reusing a header that no longer describes
   what is in context.

`session_before_compact` is skipped entirely while the extension is bypassed or
OpenViking is not connected, so an offline pi falls back to its own summarizer.

## 11. Known limitations

- **`GET /sessions/{id}/archives/{archive_id}` is blocked** (403 `ApiBlocked`)
  on the target gateway, so archive readiness is detected by reading
  `<archive_uri>/.overview.md` and, as a secondary signal, `GET /tasks/{id}`.
  `GET /sessions/{id}/context` is not used for waiting — right after a commit it
  still returns the previous overview.
- **No `viking://~` alias.** The server rejects it with `Invalid scope '~'`.
  URIs come from `GET /sessions/{id}`'s `uri` field (cached) or from the
  commit's `archive_uri`; the fallback is the legacy `viking://session/{id}`.
- **`ctx.getContextUsage()` is stale for one turn after a reset.** It counts
  the untransformed session, so the core ignores it while
  `awaitingFirstObservation` is set and reports its own estimate as
  `estimated`. It is also `null` right after a native pi compaction.
- **Tool-count pressure.** Nine tools ship here (six `viking_*` plus three
  window tools). On models with small tool budgets that is a real cost, which
  is why `history` is one tool with an `action` enum rather than four.
- **Notes are not private.** They travel into the OpenViking archive inside the
  handoff message, get summarized into Working Memory and are searchable by
  `history search_contents` and ordinary memory search — unlike the Codex notes
  backend. Do not treat them as a hidden scratchpad.
- **Status line granularity.** It is emitted once per user prompt, so during a
  long tool loop the numbers the model sees are from the start of the turn.
  `get_context_remaining` is the up-to-date reading.
- **Sibling tool results are lost.** Calling `new_context` alongside other tools
  discards their results (the header says so and names them), so they must be
  re-run if they mattered.
- **Windows are per-session.** Window ids are derived from the archives of this
  OpenViking session; a restored state belonging to another session is dropped.

## 12. Manual demo

1. Disable the non-experimental extension first — they both register `viking_*`
   and both write the same OpenViking session. Set `"enabled": false` in
   `~/.pi/agent/extensions/openviking/config.json`, or drop it from
   `settings.json`'s `packages`. (If you forget, this extension notices
   `viking_search` in `pi.getAllTools()` and disables itself with a warning.)
2. `pi install /abs/path/to/examples/pi-experimental-context-management`
3. Start `pi` and talk about topic A for a few turns. Watch the footer segment
   (`OV ✓ · w1 · 12% · a— · pi-…`) and the `[context-status]` line under each
   prompt.
4. Switch to an unrelated topic B. The model should call `new_context` on its
   own; the footer flips to `w2` and the archive id appears.
5. Ask about a detail of topic A that is not in the notes and watch it reach
   for `history`.
6. Come back after more than `idleGapMinutes` and check that the status line
   adds its NOTE line.
7. `/viking window` prints the current window, archive, Working Memory state
   and advice; `/viking` alone prints connection and session.
8. Re-enable the other extension when you are done.

`OPENVIKING_DEBUG_LOG=/tmp/ov-pi-window.log` writes one JSON Lines record per
event, including `context-window: opened w2 from archive_001 (working memory
ready, waited 34211ms)`.

## 13. End-to-end gate

`scripts/e2e-window.mjs` (with the `scripts/e2e-window.sh` wrapper) drives a
real pi binary against a real OpenViking server and a real OpenAI-compatible
LLM, in a throwaway agent directory that has only this extension installed. It
is a manual gate, not part of CI.

| Variable | Required | Meaning |
| --- | --- | --- |
| `OPENVIKING_URL` | yes | OpenViking server base URL |
| `OPENVIKING_API_KEY` | yes | OpenViking API key |
| `E2E_LLM_API_KEY` | yes | API key of the OpenAI-compatible LLM endpoint |
| `E2E_LLM_BASE_URL` | no | LLM base URL |
| `E2E_LLM_MODEL` | no | Model id |
| `E2E_LLM_API` | no | pi provider api type, e.g. `openai-completions` |
| `PI_BIN` | no | Path to the pi binary; defaults to `which pi` |
| `E2E_KEEP_TMP` | no | `1` keeps the temporary workspace on success |
| `E2E_WINDOW_FAILCLOSED` | no | `1` runs the fail-closed variant: point `OPENVIKING_URL` at a closed port and assert that nothing was cut |

**Never write an API key into a file.** Pass both keys through the environment
of the run only:

```bash
OPENVIKING_URL=... OPENVIKING_API_KEY=... \
E2E_LLM_API_KEY=... E2E_LLM_BASE_URL=... E2E_LLM_MODEL=... \
  ./scripts/e2e-window.sh
```

The gate drives three pi runs — a first prompt that plants a codename plus
padding, a `pi -c` continuation that instructs one `new_context` call, and a
second `pi -c` that asks for the codename back — and asserts on the captured
provider payloads that the post-reset request starts with a user message
opening on `<openviking-context source="context-window">`, carries `id="w2"`
and the codename from the notes, no longer contains the padding or the
`new_context` call and result, and leaves no orphan `tool_result`; plus, on the
OpenViking side, that `.overview.md` exists, `messages.jsonl` contains the
`[Context Window Handoff]` note and `[tool-result …]` entries, and a grep for
the codename hits.

## 14. Unit tests

```bash
node --test examples/pi-experimental-context-management/tests/*.test.mjs
```

`tests/context-window-core.test.mjs` covers the cut algorithm, the header
variants, the reset pipeline and its failure modes, the reminders, restore and
the compaction fallback; `tests/context-window-adapter.test.mjs` covers the io
binding; `tests/client-archives.test.mjs` the archive endpoints. Node strips TS
types, so the tests import the `.ts` modules directly.
