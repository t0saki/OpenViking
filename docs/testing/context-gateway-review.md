# Context Gateway architecture review

This revision restructures `acb1c5e9`. It retains immutable prefix replay and
replaces the distributed capture ledger with a single per-session document.
Earlier live-provider results remain historical evidence, separate from the
checks reported here.

## Storage boundaries

| Port | Data | Consistency requirement |
| --- | --- | --- |
| `ReplayStore` | Root, recall injection, disabled marker, hidden round trip, archive replacement | Indexed batch read and put-if-absent on one key |
| `CaptureQueue` | One capture document per client session, with due time and lease | Versioned single-document CAS; lease owner fences worker updates |
| `StateStore` | Response observations, recall reservations, anonymous prefix owners, tool receipts | Batch read and single-key CAS |

`RecordKind` now has exactly five members. Operational metadata does not masquerade
as injection records. The old `commit(expected, shared, jobs, cancel)` transaction
is gone: no atomic operation spans replay records, capture jobs or separate
sessions. Recall reserves a conservative allowance in one operational document
before making an external call; settling the winning immutable decision refunds
unused allowance at most once. An interrupted reservation may reduce recall
capacity, but cannot overspend the session cap.

The composition port's `load` batches independent reads. A KV implementation can
pipeline them; SQLite uses indexed queries on one borrowed connection. Capture
scheduling is a queue responsibility, not a kernel transaction: an adapter must
keep due documents discoverable and fence expired owners. A Redis deployment
still needs a durable scheduler/index design and cluster tests. It does not need
the previous scope-wide uniqueness check or multi-key CAS. A test adapter using
only in-memory KV/CAS/lease primitives runs the unchanged kernel and worker for
Chat, Anthropic and Responses; this is a portability check, not a Redis benchmark.

## Capture invariants

The document contains the OV destination, delivered anchor, pending tail,
retained turn boundaries, delivered dialogue tokens, current archive and retry
information. Pending message payloads are queue contents, not a second cursor.
The request anchor rejects late responses from superseded requests.

- Requests reconcile the supplied history and replace the unconfirmed tail.
  Edits, compaction, regeneration or an explicit reset start a fresh OV target
  and rebuild the waterline. Chat capture uses the client-visible reply chain,
  so omitted reasoning metadata does not spuriously change branches.
- Only the lease holder advances delivery/retention/archive progress. A CAS
  conflict merges its delivered prefix with the newer request tail; a different
  OV destination or lost lease ends that worker's attempt. Forked sessions have
  independent documents, even when their text matches.
- A failed head holds later turns. Five fast attempts are followed by five-minute
  recovery probes. Studio shows the reason and exposes account-scoped resync.
- The worker alone observes archive summaries and `.done` / `.failed.json`.
  Polls are spaced by five seconds, with a fifteen-minute unresolved limit.
  Terminal archives without summaries release the commit gate without replacing
  history. Ready summaries produce immutable replacement records.
- Request preparation replays existing matching replacements regardless of
  capture health. Only high measured usage plus an eligible pending archive
  permits emergency waiting; the HTTP path never polls an overview itself.

`source_message_ids` does **not** deduplicate writes in the current OpenViking
server. The gateway therefore reads the live server tail before delivery and
matches these IDs, recovering a lost append response or partial batch without
separate `WRITTEN`/`CAPTURED` records. A commit intent is persisted in the same
document before HTTP submission; the server's Phase 1 receipt resolves a lost
commit response before later messages are written. These checks do not provide
atomic exactly-once delivery when an old timed-out server operation or expired
lease holder is still running. Eliminating that race requires server idempotency.

This is the initial, unreleased schema. There are no request-time state migrations,
permanent-failure compatibility paths, queue rebuild migrations, placeholder
replacement upgrades or multi-version prefix readers. Test installations on old
development formats must select a fresh `storage_path`; startup leaves old files
intact rather than migrating or deleting them.

## Request work

Stable sessions make one batched preparation read. Anonymous clients add a prefix
ownership lookup; changed frozen settings can require an additional hidden-prefix
lookup. New roots, recall decisions and changed capture tails still persist before
forwarding. Repeating an unchanged user request without a staged response makes
no capture write. Frozen tool configuration, replay and recall are separate kernel
steps. Tool budgets and output-limit preservation retain the previous behavior.

Usage, sent markers and logs run as ASGI background work after transmission.
They are observations, not immutable replay decisions. A process crash can lose
that final bookkeeping; the next supplied history reconciles capture again.
SQLite has separate reader and writer executors. Reads arriving in the same event
loop turn are coalesced in batches of at most 64, without caching snapshots or
adding a timer delay. Cancellation and errors are isolated per waiting request.
Credentials and upstream lists use a bounded two-second cache with one loader per
key. Local credential/upstream mutations invalidate it immediately; another
process may retain a revoked credential for up to two seconds. Policies bypass
the cache because they become immutable session snapshots. Existing sessions
keep their policy snapshot; Responses mapping writes and log cleanup do not
invalidate configuration caches.

## Verification

The gateway suite passes **131 tests**, including the existing protocol/tool
regressions, capture recovery and ordering, lost append/commit responses, partial
batches, concurrent edits and resets, expired-lease fencing, scope-isolated batch
reads, corrupt-record isolation and the alternate KV adapter. Development-format
compatibility tests were removed with the compatibility paths. Ruff and whitespace
checks pass. Remote CI is not claimed as executed.

The follow-up to `3dde0b96` adds four regressions, all reproduced as failures
before the fix: consecutive Responses tool calls retain the capture destination
and delivered watermark; a worker merging with a concurrent request keeps its
rotated credential and request metadata; a new session sees policy changes made
by another process; Responses bookkeeping preserves the credential/upstream
cache. The complete suite now passes **135 tests**. An unconfirmed tool tail is
replaced in place. Only a turn already delivered after idle needs a new capture
destination when extended. Worker CAS merges copy only delivery/archive/retry
progress into the latest document, leaving request-owned fields intact.

Anthropic signature preservation is implemented: normal forwarding keeps opaque
thinking/signature blocks, immutable replay restores earlier injections, and
missing-replay or archive replacement paths discard incompatible old thinking.
Tests use synthetic signatures; they do not verify provider cryptography. The
acceptance script's `--binding-check` requires nonempty thinking signatures,
`input_transformations` metadata and no transformed blocks with binding errors
enabled. No native Claude API is available for this check, so native signature
acceptance remains unverified. An Anthropic-compatible Ark response without
signatures does not satisfy it.

Using the model configuration in `~/.openviking/ov.conf`, a live check exercised
Ark `deepseek-v4-1-flash-260910` through the gateway:

- Chat: two turns, including an SSE response and an existing-prefix replay hit.
- Anthropic-compatible `/api/compatible`: two turns, including SSE and replay.
- Chat hidden tools: a real model-selected `openviking_search` call, translated to
  MCP `find`, followed by a final answer with the hidden call withheld.

That check used an isolated HTTP fixture for OpenViking recall/capture/MCP, not a
production OpenViking instance. It validates the real model protocol path; it does
not establish native Anthropic signature binding, real OV storage recovery,
physical context-window takeover or production concurrent-stream acceptance. The
compatible responses contained no signature blocks. Credentials and generated
acceptance JSON are not committed.

## Performance and profiling

Run:

```bash
PYTHONPATH=. python scripts/context_gateway_benchmark.py --concurrency 300
PYTHONPATH=. python scripts/context_gateway_benchmark.py --concurrency 300 --profile
```

Both revisions use the same workload, Python 3.10.20 and isolated local services.
The synthetic upstream and gateway share a process. Both routes are warmed; warmup
background logs are drained before measurement. Profiling covers only the warmed
gateway burst and separates executor wait, worker execution and event-loop resume.
Worker cProfile totals overlap across threads and must not be summed as request
latency. The timings below were measured **without** the profiler.

| Measurement, p50 / p95 ms | `acb1c5e9` | This revision |
| --- | ---: | ---: |
| Parse 8 MiB single string | 15.69 / 15.96 | 14.80 / 15.22 |
| Parse dense tool history | 82.26 / 86.35 | 82.09 / 87.06 |
| Prepare dense tool history | 184.74 / 208.57 | 178.33 / 200.47 |
| Prepare 1,000 turns + 10,000 unrelated records | 21.99 / 34.27 | 20.00 / 20.55 |
| Direct SSE first byte, 300-request burst | 114.32 / 125.64 | 61.23 / 71.88 |
| Gateway SSE first byte | 1300.09 / 1392.25 | 863.68 / 949.15 |
| Gateway SSE completion | 1799.84 / 1968.60 | 906.48 / 958.46 |

The dense sample is 8,477,215 bytes, with 18,586 messages and 10,620 tool calls.
Parsing retains duplicate-key/numeric fidelity checks; it is not plain orjson.
The burst disables capture and new recall, so it measures replay/transport and
bookkeeping rather than external memory-service latency.

The profile supports the storage-contention concern: the old burst made 600 replay
read submissions, with about 199 ms median executor wait, and 300 authentication
and upstream-list reads each. The revised burst used five grouped snapshot read
submissions and one authentication/list load each. Under instrumentation, median
preparation dropped from 779 to 229 ms. SQLite calls still dominate worker time;
request scheduling, HTTP transport and event-loop work remain significant.

The measured median first-byte difference versus each run's direct route fell
from about **1.19 s to 0.80 s**. Direct-route timings vary, and grouping alone gave
only a modest additional gain. The design's millisecond-overhead target remains
unmet; neither these numbers nor the parse-only results establish 8 MiB end-to-end
or 300 continuously active stream acceptance.

The follow-up also isolates the SQLite grouping choice: three runs per variant,
interleaved, on the corrected code and a fresh Python 3.10.20 environment
(aiohttp 3.14.3, orjson 3.12.0). The alternative replaces only `SQLiteKernelStore.load`
with one `db.run` per request, reading the same three ports on one connection.
Reader/writer pools, policies, warmup and workload are unchanged. No profiler is
enabled. The table reports the median of each variant's three run-level values:

| 300-request burst | Grouped reads | Individual reads |
| --- | ---: | ---: |
| Gateway first-byte p50 | 960.73 ms | 1063.93 ms |
| Gateway first-byte p95 | 1069.80 ms | 1161.58 ms |
| First-byte p50 difference from each run's direct route | 840.42 ms | 1001.10 ms |

Grouping remains in the SQLite adapter: it saves about 103 ms at gateway p50 in
this workload. Direct-route variance affects the differences, and the newer
dependency environment makes this a separate A/B check rather than a new before/
after comparison with the earlier table. It still does not meet the millisecond
overhead target. Policy reads intentionally bypass TTL caching even though this
adds one management read per enhanced request.
