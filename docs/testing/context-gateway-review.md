# Context Gateway review fixes

The review's runtime findings were confirmed. This revision changes the following
behavior and keeps the existing immutable replay format:

- Emergency waiting uses provider input/output usage for the same upstream/model,
  with per-model `context_windows`. Request bytes never trigger it. A missing
  overview leaves the full history intact; no placeholder replacement is written.
- Takeover activation counts sanitized conversation content accepted by OV.
  Further commits use OV's pending dialogue tokens. An outstanding overview
  prevents another commit while captures continue in order.
- The first Chat request retains the client's token limit. Only hidden
  continuations consume the gateway tool budget. Incompatible tool settings
  omit new gateway tools for that request and still replay existing history.
- Ark has no gateway-imposed 15-request/minute limit.
- Without an explicit session header, capture, takeover and new hidden tools are
  disabled. Recall/replay still work. Plugin detection ignores assistant prose,
  tool results and attachment names.
- Capture confirms only new turns. A failing job blocks subsequent jobs from
  that session, retries at most five times, then disables further capture/takeover.
  Editing history already delivered to OV also stops capture instead of mixing branches.
- Indexed storage reads exclude unrelated usage/write records. Usage is one
  mutable row; sent markers are batched and SQLite connections are pooled.
  Conditional writes are generic storage primitives; recall budgets and queue
  reconciliation live in the kernel/capture pipeline.
- Responses mappings use indexed lookups and expiry, and are not saved for
  `store: false`. The default mapping lifetime is 30 days.

HTTP processing is split into routing, preparation, forwarding and observation
in `proxy.py`; `app.py` hosts the routes and management API. Record kinds are
centralized in `records.py`. No Redis backend is implemented or claimed as tested.

Gateway dependencies live in `openviking[context-gateway]` with minimum versions.
The Docker image opts into that extra. Plugin capture changes/version bumps were
reverted. Generated live-result JSON is no longer committed. User/account deletion
still intentionally calls the gateway's HTTP cleanup endpoint from the server's
own adapter; the deletion service does not import the gateway runtime.

## Verification

The local Python gateway suite passes 95 tests, including sanitized Claude Code,
Codex and OpenAI SDK fixture replay, concurrent writes across two store instances,
capture retries/branch handling and the review regressions. Studio's three gateway
tests, targeted lint, production build, Ruff and `uv lock --check` pass. These are
local results, not a claim about remote CI.

The prior [live acceptance record](context-gateway-live-acceptance.md) identifies
its tested revision. This review run uses local synthetic providers; it does not
replace native Anthropic signature/binding or physical-window acceptance. Those
checks still need suitable live-provider evidence.

## Local performance comparison

Reproduce with `PYTHONPATH=. python scripts/context_gateway_benchmark.py --concurrency 300`.
The baseline is `7e05a3c2`; the same script and Python 3.10.20 environment were used
for both revisions, in separate runs. The synthetic upstream and gateway share one
process; HTTP connections and session roots are warmed before the measured burst.
Each history case also contains ten old write records per dialogue turn. Values
below are wall-clock milliseconds, shown as p50 / p95.

| Measurement | Before | After |
| --- | ---: | ---: |
| Parse 8 MiB JSON | 23.65 / 27.01 | 14.98 / 15.52 |
| Prepare/replay 10 turns | 11.85 / 16.01 | 1.40 / 1.83 |
| Prepare/replay 100 turns | 34.39 / 36.51 | 3.59 / 4.14 |
| Prepare/replay 1,000 turns | 263.45 / 291.93 | 25.92 / 28.03 |
| Direct SSE first byte, 300-request burst | 124.31 / 148.60 | 189.20 / 215.92 |
| Gateway SSE first byte, same burst | 2245.90 / 2301.98 | 1341.45 / 1388.85 |
| Gateway SSE completion, same burst | 3198.80 / 3471.61 | 1741.05 / 1944.66 |

Parsing alone is not the total overhead of an 8 MiB request. The 300-request burst
still adds substantial latency and does **not** establish the design's millisecond
overhead target for 300 active streams. The direct baseline also shows scheduling
variation. Results demonstrate reduced work, not a production latency guarantee;
measure the deployed process count, storage and actual traffic before sizing it.
