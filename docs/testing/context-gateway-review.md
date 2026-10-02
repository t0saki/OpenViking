# Context Gateway review fixes

This revision addresses the follow-up review of `6ef27559`. Prior phase-one,
phase-two and live-provider results are historical evidence; they do not substitute
for this revision's regressions or native Anthropic signature acceptance.

## Capture, archives and replay

- Existing matching replacements replay independently of capture health, plugin
  detection and permission to create a new replacement. Missing summaries never
  replace history with a placeholder.
- Capture has a separate destination from the immutable client-session ledger.
  Editing, compacting or resetting delivered history switches that destination
  and resyncs the supplied branch. Old in-flight workers cannot publish archives
  into the new branch. Common-prefix replacement records remain available.
- The queue key includes scope, session and anchor. Forked sessions can capture
  identical prefixes independently. Migration preserves tasks, progress and leases;
  all gateway workers should restart together for the schema upgrade.
- A failed head holds later turns in order. Five fast attempts are followed by
  five-minute recovery probes, with health checked before writes. Successful
  delivery clears the failure. Studio logs display state/reason and expose an
  account-scoped resync action; legacy permanently stopped captures resync too.
- Archive observation is shared by request preparation and capture. `.done` and
  `.failed.json` release a terminal archive without a summary; missing terminal
  markers are bounded by a 15-minute lifetime. Checks are throttled to five seconds.
  Only a matching, confirmed-pending archive permits emergency waiting.
- Anonymous continuations reuse a unique longest completed reply prefix recorded
  by the gateway. New or ambiguous histories get isolated sessions with capture
  and tools enabled. Identical opening user text never identifies a session.
  A per-conversation header is still the reliable solution for indistinguishable
  transcripts/retries; prefix inference cannot prove identity in those cases.

## Tools and request work

Hidden tool budgets count added calls/results and hidden output, excluding client
history and schemas. Missing streaming usage falls back to estimating added data.
Output parameters are preserved on every request. Reaching the threshold disables
further gateway calls and allows a final answer, whose output may exceed the
threshold. A completed write does not become a budget-related HTTP error.
Incompatible tool settings forward client-visible history without expanding
undeclared hidden calls. Compatible requests keep the frozen definitions/replay.

Preparation uses shallow message copies with replacement of changed fields;
immutable nested tool/image data are not deep-copied. Hidden-prefix hashing runs
only where it is needed, and local indexed ledger reads are batched. JSON numeric
and duplicate-key checks remain: replacing them with plain orjson parsing would
lose the existing fidelity guarantees.

Record kinds are centralized. Missing optional dependencies explain how to install
`openviking[context-gateway]`; CI now installs that extra through package metadata.
No shared JS plugin changes or generated acceptance JSON are included. Redis is
not implemented or verified. A backend must still provide the documented atomic
conditional writes and shared-key consistency; cluster placement is not proven.

## Verification

The local gateway suite passes 120 tests. Added coverage includes archive terminal
states/poll contention, outage recovery and ordering, fork isolation, edit/compact
resync, an in-flight reset, interrupted archive publication, queue migration, anonymous ownership ambiguity,
undeclared tool history, long-context writes, streaming without usage, preserved
output limits, optional-dependency errors and account-scoped reset authorization.
Studio's three existing gateway tests, targeted lint, production build, Ruff,
`git diff --check` and `uv lock --check` pass. Remote CI is not
claimed as executed. This run uses synthetic providers; native Anthropic signature
binding and physical context-window acceptance remain unverified.

## Local performance

Run `PYTHONPATH=. python scripts/context_gateway_benchmark.py --concurrency 300`.
Both revisions use the same expanded script, Python 3.10.20 and temporary local
services. The gateway and synthetic upstream share one process; both HTTP routes
are warmed before the 300-request burst. The dense fixture is 8,477,215 bytes,
18,586 messages and 10,620 tool calls. Values are p50 / p95 milliseconds.

| Measurement | `6ef27559` | This revision |
| --- | ---: | ---: |
| Parse 8 MiB single string | 14.80 / 25.12 | 15.00 / 15.31 |
| Parse dense tool history | 80.93 / 87.73 | 82.47 / 95.90 |
| Plain orjson, dense history (without fidelity checks) | 26.87 / 52.81 | 28.05 / 55.34 |
| Prepare dense tool history | 464.48 / 484.76 | 196.06 / 251.81 |
| Prepare 1,000 short turns plus 10,000 old records | 26.55 / 30.89 | 21.25 / 24.23 |
| Direct SSE first byte, 300-request burst | 56.21 / 67.43 | 110.71 / 122.25 |
| Gateway SSE first byte, same burst | 1387.58 / 1405.13 | 1315.68 / 1340.29 |
| Gateway SSE completion, same burst | 1887.59 / 2124.52 | 1782.58 / 1940.17 |

The dense preparation path improves, while parsing is unchanged. The direct
baseline varies between runs; the 300-request burst still adds about 1.2 seconds
at p50 and does not satisfy a millisecond overhead claim. These samples measure
preparation/transport with capture and new recall disabled, not production load.
They do not establish end-to-end 8 MiB or 300-active-stream acceptance.
