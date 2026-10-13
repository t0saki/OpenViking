# ov-gateway-rs: Rust prototype of the gateway hot path, and what it buys

Experimental. Not wired into anything, not a product. It exists to answer one question:
how much CPU per request (and therefore how many concurrent Claude Code-style sessions
per core) would a Rust worker kernel save compared with the Python gateway in
`openviking_gateway/`, on the same workload.

Context: the v0.5.0 perf report found that one Python worker saturates at about 300-400
concurrent agent sessions (~13 req/s, ~50 ms gateway CPU per request), and that the
single-threaded event loop is the limit: ~53% of GIL time in the streaming relay (hidden
tool loop parses, accumulates and re-serializes every SSE event), ~17% in request
preparation (`classify()` regexes over `str(system)`, full-history `prefix_chain` hashing),
~16% in uvicorn send.

## Summary

E2E_SUMMARY_PLACEHOLDER

## What is ported, what is stubbed

The port follows the Anthropic Messages streaming path of the current repo source
(`proxy.py`, `kernel.py`, `tool_loop.py`, `protocols.py`, `tool_protocols/{anthropic,common}.py`,
`replay_store.py`, `storage.py`, `capture.py`) closely enough to do the same work per request.

Ported (same algorithm, same data, same SQLite schema and queries):

- Body parse (serde_json, insertion-ordered), per-request policy read from
  `management.sqlite3`, OV identity check (cached 30 s, like `VikingClient.identify`).
- `strip_lead` check, `sent_history` (SHA-256 of every assistant text block + `state` read).
- `prefix_chain` / `normalize`: the Rust writer produces the same bytes as
  `orjson.dumps(normalize(m), OPT_SORT_KEYS)`; anchors are **identical** to Python's on the
  fixtures (checked by both microbenches). Computed in full 3x per tool-loop request
  (identity, `persist`, `relayed`) plus once for capture `stage`, exactly like Python.
- `replies()` (with `INDEXED BY replay_kinds`, as the perf report recommends; the repo source
  still lacks it, which does not matter on the empty-store runs here), `load()` (replay
  records, observation, capture document), `classify` (same six case-insensitive regexes over
  the serialized `system`), `plugin_present`, `client_openviking_tools` (lookarounds emulated),
  session root with `parameter_fingerprint`, tool selection and `tools_closed`, injection
  replay, `configure_tools`, `measure` (`estimate` with the media regex), hidden-history
  replay + `merge_tool_results`, and serialization of the upstream body.
- Recall on user turns: recall ledger reserve/settle (two `state` CAS writes), OV
  `search/search`, the session-start block with profile (5 OV GETs in parallel), the
  injection record write.
- Capture: `confirm` on user turns and `stage` on finished replies (`advance` / `turns` /
  `capture_messages` with `clean_text`), capture queue CAS writes.
- Upstream forwarding with streaming (reqwest), and the hidden tool loop: per-event SSE
  framing, JSON parse, `accumulate` + `event` (tool_use blocks buffered, indices remapped),
  re-encode and forward; `end()` validation, `publish_calls`, gateway tool execution through
  OV MCP `tools/call` with receipt CAS writes, visible notices, continuation request,
  `persist` (HIDDEN record), `relayed` (REPLY record before the terminal events), terminal
  events, then `completed` (observation CAS, capture stage, request log row).
- Threading mirrors Python's: one async worker thread, a 4-thread read pool, and one
  dedicated SQLite writer thread.

Simplified or stubbed (noted where it could matter):

- Gateway keys come from a config file instead of the management store (Python caches
  authentication for 2 s, so this is ~free on both sides). No admin API, Studio, upload
  proxy, OpenAI Chat/Responses protocols, encryption, TTL expiry, count_tokens.
- `parse_body`'s duplicate-key and float-precision checks are not ported.
- Not ported because the default policy never triggers them on this workload: compaction
  (1M-token default window), agent windows / reminders / status line, recall notice
  (`show_recall` off), reasoning restore (generic vendor), tool budgets and deadlines,
  `TOOL_OVERRIDES` descriptions, window cuts.
- Capture worker delivery is simplified (create session, batched message writes, pending
  token check; no commit/archive/error states). Python's worker does slightly more.
- Non-tool-loop relay (`observe_response`) is minimal; this workload always takes the tool
  loop path because gateway tools are on by default.
- Error and degradation paths are mostly `unwrap()` or a generic error event.

## Method

Host: shared 8-vCPU devbox (Xeon Platinum 8336C), 15 GB RAM. Python 3.11.2 with the repo's
`openviking_gateway` source (installed into a fresh `uv` venv: PyPI `openviking==0.5.0` for
dependencies, the repo package overlaid via a `.pth` file), uvicorn + uvloop + httptools,
orjson 3.x. Rust 1.94, release build (thin LTO), mimalloc.

### Microbenchmarks (`bench/pymicro.py` vs `src/bin/microbench.rs`)

Same fixtures for both (`bench/gen_fixtures.py`): two Claude Code-shaped tool-result requests
built with the w157 harness's `agent_sim.Session` (system ~20k tokens, 22 tools ~12k tokens,
synthetic history): 184 KB / ~46k tokens / 11 messages and 567 KB / ~142k tokens /
101 messages; and three upstream SSE streams built exactly as `mock_agent_upstream.py` writes
them (150 thinking tokens + signature + 200/800/1500 text tokens + one client tool_use, one
write per event: 128/328/561 events). Process CPU time per call, median of 5 batches, both
pinned to one core.

- `full_prepare` is the whole steady-state preparation of a tool-result turn on an existing
  session: Python `parse_body` + `MemoryKernel.prepare` (real `SQLiteKernelStore`, fake
  viking returning the recorded tool catalog) + the proxy's compare and `orjson.dumps`;
  Rust: policy read + parse + `kernel::prepare` (same SQLite schema/queries) + serialization.
  The previous reply's usage is in the observation, as it is under load.
- `relay` is the per-response relay logic of the hidden tool loop without sockets: SSE
  framing, parse, accumulate/remap, re-encode for every event, then `end`,
  `publish_calls`, `final`, capture of the final reply, terminal events.

### End to end (`bench/e2e.py`, `bench/sweep.sh`)

The w157 harness, unchanged except for ports, a `--key` flag and SO_REUSEPORT on the mock:
closed-loop `agent_sim.py` sessions (Claude Code shape, Anthropic streaming, header session
id, ~80% replies end in a client tool call, 1-5 s tool time, 3-10 s user think time,
compaction at ~120k tokens), `mock_agent_upstream.py` (TTFT 2 s ±30%, 60 tok/s, 200-1500
output tokens + 150 thinking, 5% `openviking_search` calls when offered), and
`mock_ov.py` (recorded OV 0.5.0 responses, 5 ms). Default context profile (recall, profile,
read-only gateway tools, capture).

- Gateway: 1 worker, `taskset -c 0`, empty storage, storage on `/dev/shm` for both
  (the shared host's root disk was saturated by other tenants: IO pressure "full" ~55% over
  5 min, multi-second fsyncs; tmpfs removes that noise for both sides equally).
- Mocks on cores 4-5 (3 upstream processes + mock OV), simulators on cores 6-7
  (2-4 shards); cores 1-3 left for other tenants of the host.
- Each point: 30 s ramp, 60 s warm-up, 120 s measured. CPU per request = gateway
  utime+stime (from `/proc/<pid>/stat`, all threads) over the window / requests started in
  the window. Latency = `added_ttft_ms` from agent_sim (client first byte − send time − the
  mock's configured TTFT), for tool-result turns (continuations) and user turns, hidden-tool
  turns excluded.
- Noise control: another tenant (a headless-Chrome video render) intermittently landed on
  core 0. The driver waits for core 0 to be quiet before each run, measures how much of
  core 0 other tasks used during the window (`core0_other_pct`), and reruns a point (up to
  3 attempts) when that exceeds 15%, keeping the cleanest attempt.

## Microbenchmark results

Process CPU ms per call, both pinned to the same idle core (the two runs, 40 minutes apart on
different cores, agree within a few percent). Correctness cross-checks: the last
`prefix_chain` anchor of both fixtures is byte-identical between Python and Rust
(`c49df006…` and `cb3a462f…`), and the relayed output of every stream has the same event count
and byte length (17,983 / 44,965 / 76,395 bytes).

| step | input | Python ms | Rust ms | speedup |
|-|-|-|-|-|
| **full_prepare** (steady-state tool-result turn) | 46k tokens | 6.31 | 1.05 | **6.0x** |
| **full_prepare** | 142k tokens | 9.43 | 2.33 | **4.1x** |
| **relay** (one streamed response) | 150 + 200 tokens | 1.00 | 0.24 | **4.2x** |
| **relay** | 150 + 800 tokens | 2.39 | 0.61 | **3.9x** |
| **relay** | 150 + 1500 tokens | 4.05 | 1.01 | **4.0x** |
| parse_body | 46k / 142k | 0.28 / 0.84 | 0.12 / 0.36 | 2.3x |
| prefix_chain (full history) | 46k / 142k | 0.11 / 1.07 | 0.06 / 0.58 | 1.8x |
| classify | 46k / 142k | 3.95 / 4.27 | 0.09 / 0.11 | 39-43x |
| plugin_present | 46k / 142k | 0.21 / 0.55 | 0.009 / 0.018 | 23-30x |
| client_openviking_tools | 46k / 142k | 0.49 / 0.56 | 0.18 / 0.18 | 2.8-3.0x |
| thinking digests (sent_history) | 46k / 142k | 0.02 / 0.22 | 0.015 / 0.16 | 1.4x |
| estimate (whole body, first request of a session) | 46k / 142k | 3.35 / 10.5 | 0.45 / 1.40 | 7.5x |
| capture turns over the whole history (first user turn) | 46k / 142k | 0.80 / 7.23 | 0.06 / 0.60 | 12-14x |
| post-reply chain (`relayed`/`persist`, each) | 46k / 142k | 0.13 / 1.07 | 0.07 / 0.59 | 1.8x |
| dump body | 46k / 142k | 0.019 / 0.063 | 0.066 / 0.215 | 0.3x |

Reading it:

- `classify` alone is 3.9-4.3 ms, i.e. 45-63% of Python's whole preparation. It runs six
  case-insensitive regexes over `str(body["system"])` (an 80k-character repr) on every
  request. Rust's regex engine does the same six scans in ~0.1 ms. This one is fixable in
  Python too (match the system text instead of its repr, one combined pattern, or
  check only the first/last few KB); it does not need Rust.
- Where Python already calls native code (`orjson`, `hashlib`), Rust gains little:
  `prefix_chain` 1.8x (only the `normalize` walk is Python), and orjson's SIMD writer is
  3x faster than this prototype's serializer.
- The relay logic is 4x faster in Rust, but it is only 1-4 ms of Python CPU per response;
  most of the ~53% "relay" share in the perf report is the framework around it (see E2E).

## End-to-end results

E2E_PLACEHOLDER

## Caveats

- **Prototype vs product.** The Rust side skips work the Python gateway does on paths this
  workload never takes (compaction, windows, notices, OpenAI protocols, encryption, admin),
  validates less (`parse_body` checks, Pydantic policy validation is a field check), and has
  minimal error handling. On this workload those paths cost little in Python, but a real
  port would add some CPU back; assume the measured Rust numbers are optimistic by roughly
  10-20%.
- **The Rust side is a straight port, not an optimized design.** It recomputes the full
  `prefix_chain` 3-4 times per request like Python, parses every SSE event into a generic
  `serde_json::Value`, forwards one write per event, and keeps the parsed request alive for
  post-reply bookkeeping. Incremental chain hashing, typed/lazy event parsing and event
  batching would cut further, and most of those ideas apply to Python as well.
- **Same work, not same bytes.** Anchors and relayed SSE bytes match Python exactly on the
  fixtures; the outgoing upstream body is semantically the same but not byte-compared, and
  the session-start/recall text is an approximation of Python's formatting.
- **Kernel time is a floor for both.** About 6-7 ms of each request's CPU is system time
  (loopback TCP, epoll, ~560 small writes and reads per streamed reply), nearly identical on
  both sides. That compresses the end-to-end ratio (user-space alone is ~4x) and would also
  bound a PyO3 approach.
- **Shared, noisy host.** The machine was shared with a sibling benchmark (its gateway
  pinned to core 0, load on 4-7) and a long headless-Chrome video render that saturated all
  8 cores for part of the session. CPU time per request turned out robust to this (Python
  37.6-43.7 ms and Rust 12.8-16.3 ms across every run, quiet or noisy); latency is not, so
  each latency point reports how much of the gateway core other tenants used
  (`gateway-core noise %`), and noisy points are marked.
- **Storage on tmpfs** for both, to remove the host's disk contention (multi-second fsyncs
  observed). With real disks both sides pay fsync latency per commit (~4-6 commits per
  request); that is wall time on a writer thread, not event-loop CPU, in both designs.
- **Mock OV** (recorded replies, 5 ms) instead of a real OV server; the perf report showed OV
  speed does not move the gateway's ceiling, only user-turn latency.
- **Memory.** RSS per in-flight request is similar on both sides (~1.6-2 MB): the Rust port
  keeps the whole parsed body for post-reply hashing, as Python does. At ~1,800 sessions the
  Rust worker used ~2.4 GB. Keeping only the messages, or the raw bytes, would cut this.
- **Load generator.** At the highest Rust points the Python simulators and mock upstreams
  used 1-1.5 cores each in total; they did not saturate (no point showed a simulator or
  mock process near 100%), but they compete with other tenants for cores and their
  scheduling delay is included in the client-side latency.

## Recommendation

RECOMMENDATION_PLACEHOLDER

## Reproduce

```bash
cd crates/ov-gateway-rs && cargo build --release
ROOT=<scratch dir>; mkdir -p $ROOT/bench $ROOT/run $ROOT/fixtures $ROOT/home
cp ~/.zouk/agents/agent-f7712fb9/w157/bench/{agent_sim,mock_agent_upstream,mock_ov}.py $ROOT/bench/
cp ~/.zouk/agents/agent-f7712fb9/w157/run/ov_recording.jsonl $ROOT/bench/ $ROOT/run/
# patch the copies: agent_sim --key/--gw-pid flags, mock reuse_port (see e2e.py docstring)
uv venv $ROOT/pyvenv -p 3.11 && VIRTUAL_ENV=$ROOT/pyvenv uv pip install openviking==0.5.0 orjson async-timeout uvloop httptools psutil
# overlay the repo gateway: move site-packages/openviking_gateway aside, add a .pth with a dir
# containing a symlink to <repo>/openviking_gateway
$ROOT/pyvenv/bin/python bench/gen_fixtures.py $ROOT/bench $ROOT/fixtures
taskset -c 0 target/release/microbench $ROOT/fixtures 40
taskset -c 0 $ROOT/pyvenv/bin/python bench/pymicro.py $ROOT/fixtures --iters 20
E2E_ROOT=$ROOT $ROOT/pyvenv/bin/python bench/e2e.py mocks start --upstreams 3
E2E_ROOT=$ROOT bench/sweep.sh mytag
E2E_ROOT=$ROOT $ROOT/pyvenv/bin/python bench/report.py $ROOT/results mytag_py_n300 mytag_rs_n1200
```
