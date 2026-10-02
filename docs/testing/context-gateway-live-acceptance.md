# Context Gateway live acceptance — 2026-10-02

The phase-one and phase-two implementation at `3cdf1a80` was exercised against
Ark's live `deepseek-v4-1-flash-260910` model. Protocol, client replay, hidden
tools, file import and a configured archive threshold passed. Native Anthropic
binding and the model's physical maximum context window remain unverified.
This is historical evidence for the stated revision, not a fresh live run of
subsequent review fixes. Generate machine-readable output locally with the
acceptance script; generated result files are not maintained in the repository.

## Environment and isolation

Model and embedding credentials were read from the operator's existing
`~/.openviking/ov.conf`. The test used a separate loopback OpenViking server,
gateway, account, data directory and client workspace. Existing services and the
original configuration were unchanged. The current source's vector engine and
RAGFS were built locally; borrowing an older installed RAGFS initially caused a
background import ABI error and was corrected before the accepted runs.

The test server reported the unreleased version `0.1.dev2651`, so only this
isolated gateway used `min_server_version: 0.1.dev0`. The product's minimum server
version was not lowered. All imported documents and conversations were synthetic.

## Three protocols, three turns

The gateway forwarded to `/api/v3/chat/completions`, `/api/v3/responses` and
`/api/compatible/v1/messages`. Each sequence used a long stable prefix, real
OpenViking recall, and SSE on turn two. The model correctly recalled the Aurora
project's `cobalt-zebra-731` cluster and `Delta` release owner. Every upstream
request preserved the normalized prefix of the preceding request.

| Protocol | Turn 1 input / cached | Turn 2 input / cached | Turn 3 input / cached |
| --- | ---: | ---: | ---: |
| Chat Completions | 5,183 / 0 | 5,502 / 5,120 | 5,668 / 5,376 |
| Responses | 5,185 / 0 | 5,504 / 5,120 | 5,670 / 5,504 |
| Anthropic compatible | 5,214 / 0 | 5,533 / 5,120 | 5,699 / 5,504 |

Cache reads cover the preceding input with at most a 128-token final-block
allowance. Anthropic input counts here include cache reads. The reusable
`scripts/context_gateway_acceptance.py --require-cache` also passed three turns
for each protocol with enhancement disabled, independently checking protocol
capture and provider cache reporting.

The memory policy used the default 1,600-token per-turn budget and a 0.1 score
threshold. A preliminary 1,000-token budget sometimes yielded only short
directory abstracts because the server's per-entry cap omitted longer abstracts;
it did not reliably answer the first question. Earlier requests also retained
their immutable pre-import recall decisions. Fresh prefixes and the default
budget were used for the accepted run. This small synthetic corpus does not
establish retrieval quality on production data.

## Real clients and recorded replay

| Client | Version | Observed result |
| --- | --- | --- |
| Claude Code | 2.1.286 | `-p` followed by `--resume`; both answers correct; Messages prefix retained |
| Codex CLI | 0.146.0 | `exec` followed by `exec resume`; both answers correct; full-history Responses prefix retained |
| OpenAI Python SDK | 2.29.0 | Two Chat Completions turns; both answers correct; prefix retained |

Recordings on both sides of the gateway confirmed that injection appeared only
upstream. The short Claude sequence was below a reliable cacheable prefix size,
so its success establishes replay and answer continuity; the long protocol
sequence above supplies the cache evidence. Codex used fallback model metadata
because this model ID was absent from its catalog.

The [sanitized fixtures](../../tests/context_gateway/fixtures/README.md) preserve
the real clients' message shapes and extension fields. Nine local mock-upstream
checks cover prefix replay with and without session headers, duplicate concurrent
requests, capture deduplication and exact passthrough with enhancement disabled.
System instructions, descriptions, paths, session identifiers and metadata are
redacted. The fixtures are not native signed Anthropic samples.

The final gateway suite passed 72 tests. Ruff lint/format checks, generated-rule
synchronization and `git diff --check` also passed.

## Hidden tools and import

The SDK Chat client exercised `openviking_read` followed by another user turn.
The hidden result was replayed; no OpenViking tool call reached the client. The
first visible text arrived at 1.504 seconds and the first response completed at
2.688 seconds, including the hidden continuation. These are individual samples,
not latency benchmarks.

A mixed assistant round called `openviking_read` and `client_echo`. Only
`client_echo` was returned to the client. After its result arrived, the gateway
reconstructed the original assistant and hidden result, and the real model
answered correctly. Both sequences used `thinking.type: disabled`.

Two real imports reached background task status `completed` without errors:

- A local Markdown fixture: the model requested the hidden import tool, received
  signed upload instructions and called the client's `bash` tool. The test runner
  executed only the expected `curl` upload of that fixture to the local gateway.
  The client saw `bash`, and never saw the OpenViking call.
- Extracted attachment text: the model selected `attachment_index: 0`; the gateway
  uploaded the `<context><source>` text without a client shell call.

An HTTP upload acknowledgment alone was not counted as import success.

## Real archive and cache recovery

An isolated policy used `takeover_tokens: 5000`, `context_window: 20000`, two
recent turns and a three-second idle capture delay. OpenViking generated a real
archive using the configured live model. All four answers retained the Orion
project's `silver-panda-517` cluster.

| Turn | Input tokens | Cached tokens | Archive replacement |
| --- | ---: | ---: | --- |
| 1 | 7,041 | 0 | No |
| 2 | 8,418 | 7,040 | No |
| 3 | 1,813 | 0 | Yes |
| 4 | 1,836 | 1,792 | Same replacement |

The cache was invalidated at the archive boundary and recovered on the next
turn; the post-archive prefix remained stable. This validates a configured
threshold against real services. The conversation did not exceed this model's
physical maximum context window.

## Remaining acceptance evidence

- Native Anthropic `thinking-binding-controls-2026-08-01`, signed thinking and
  empty `input_transformations`: Ark returned thinking text but no signatures or
  binding metadata. HTTP 200 from that compatible endpoint is insufficient.
  The script's `--binding-check` requires these fields and fails when absent.
- Direct OpenAI and DeepSeek endpoints: this run used one Ark-hosted model through
  three protocols. It does not establish the behavior of the other providers.
- A conversation exceeding the live model's physical context window, plus real
  clients' automatic compaction behavior after gateway takeover.
- Claude interactive auxiliary requests and UI chat applications such as
  Open WebUI or Cherry Studio. The chosen Chat client here was the Python SDK.

These limits keep the complete design-document acceptance open even though the
implementation and the available live-provider checks are complete.
