# OpenViking Context Takeover for Pi

Context takeover makes OpenViking the authoritative long-term context store for
pi sessions. Pi still keeps recent active turns locally, but committed history
is represented to the model by OpenViking's archive overview through pi's
`context` hook.

## Model

The extension tracks:

| Field | Meaning |
|-------|---------|
| `coveredUserTurns` | Number of real user turns already covered by the OV archive overview |
| `overview` | Latest archive overview returned by `GET /sessions/{id}/context` |
| `fingerprint` | Stable fingerprint of the last covered message for branch mismatch detection |
| `pendingTokens` | Estimated synced token pressure since the last successful boundary advance |
| `syncedEntryCount` | Pi branch watermark restored across `pi -p` / `pi -c` processes |

State is persisted as a pi custom entry:

```ts
pi.appendEntry("ov-takeover", state)
```

On startup the extension scans the branch from the end, restores the latest
entry, and restores `SyncManager`'s watermark so `pi -c` does not resend the
same branch entries to OpenViking.

## Runtime Flow

1. `turn_end` captures new branch entries into the OpenViking session.
2. The capture path uses a disk pending queue when OpenViking is temporarily
   unreachable.
3. When `pendingTokens >= takeoverTokenThreshold`, takeover tries to advance.
4. Advance requires a successful flush barrier: all current-session
   `addMessage` queue entries must be delivered. Pending `commitSession`
   entries and entries for other sessions do not block the barrier.
5. The extension commits with `queueOnFailure: false`; a delayed commit cannot
   safely advance the local context boundary.
6. The extension polls session context until `latest_archive_overview` is
   available, then advances to `lastSeenUserTurns - keepRecentTurns`.
7. The `context` hook replaces covered conversation messages with a synthetic
   user message beginning with `[OpenViking Session Context]`, then recall is
   injected into the remaining latest user turn.

The overview message timestamp is derived from the first kept message, so the
provider payload remains byte-stable between commits and can benefit from
prompt caching.

## Compaction

When pi emits `session_before_compact`, takeover attempts the same
flush-commit-overview sequence. If it succeeds, the extension returns:

```ts
{
  compaction: {
    summary: "[OpenViking Session Context]\\n...",
    firstKeptEntryId,
    tokensBefore,
    details: { source: "openviking" }
  }
}
```

If any step fails, the handler returns `undefined` and pi's default compaction
runs. This is intentional fail-open behavior.

## Capture Fidelity

Takeover mode enables faithful capture in the pi adapter. Short acknowledgments,
punctuation-only turns, and other low-signal text are still captured because
those turns may later disappear from the live model context. Empty text,
slash commands, and OpenViking status messages remain filtered.

## Configuration

In `~/.openviking/ovcli.conf`, under the shared `plugin` section or the `plugin.pi` override:

```json
{
  "plugin": {
    "pi": {
      "takeoverEnabled": true,
      "takeoverTokenThreshold": 30000,
      "takeoverKeepRecentTurns": 3,
      "takeoverOverviewBudget": 3000,
      "takeoverOverviewPollMs": 2000,
      "takeoverOverviewPollMax": 15
    }
  }
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `takeoverEnabled` | `true` | Enable context takeover (`OPENVIKING_TAKEOVER`) |
| `takeoverTokenThreshold` | `30000` | Synced-token pressure required before commit and boundary advance |
| `takeoverKeepRecentTurns` | `3` | Recent user turns kept in full fidelity |
| `takeoverOverviewBudget` | `3000` | Token budget for the injected archive overview |
| `takeoverOverviewPollMs` | `2000` | Delay between overview polling attempts |
| `takeoverOverviewPollMax` | `15` | Max overview polling attempts after commit |

## Failure Modes

| Failure | Behavior |
|---------|----------|
| OpenViking health check fails | Extension stays disconnected; pi runs normally |
| Pending addMessage replay fails | Boundary is not advanced; full local history remains visible |
| Commit fails | Boundary is not advanced; pending token pressure remains |
| Overview is not ready | Boundary is not advanced; retry on the next threshold or manual `/viking commit` |
| Branch fingerprint mismatch | Boundary resets to 0 and full history is shown until the next successful advance |
| Compaction takeover fails | Returns `undefined`; pi default compaction proceeds |

## Live E2E

The manual live gate is:

```bash
OPENVIKING_URL=... \
OPENVIKING_API_KEY=... \
E2E_LLM_API_KEY=... \
bash examples/pi-coding-agent-extension/scripts/e2e-live.sh
```

Any OpenAI- or Anthropic-compatible endpoint works: override `E2E_LLM_BASE_URL`,
`E2E_LLM_MODEL`, and `E2E_LLM_API` (pi provider api type, e.g.
`anthropic-messages`; defaults to `openai-completions`).

It runs three real `pi -p` / `pi -c` turns, sets a tiny takeover threshold, and
asserts that the third provider payload contains `[OpenViking Session Context]`
while old padding from the first turn is no longer present in raw conversation
history.
