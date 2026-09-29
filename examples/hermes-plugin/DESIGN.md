# Hermes memory provider: design record

This document records what the Hermes host guarantees to a memory provider, what
this plugin does with each guarantee today, which settings it shares with the
other OpenViking plugins, and which commit-lifecycle behaviours the tests pin.
It is the host contract record required by
[the plugin development spec](../../docs/zh/agent-integrations/18-plugin-development.md#2-接入前先确定宿主契约).

**Baseline.** Plugin 2.0.1 (`plugin.yaml`, `pyproject.toml`) at OpenViking
`b20f192e6`. Host facts are read from Hermes Agent `58d146e961`, the commit the
plugin CI pins (`.github/workflows/hermes-plugin-tests.yml`). Everything below
describes the code as of 2026-09-29. Items marked **Planned** come from the
approved refactor plan and are not implemented yet.

**Citations.** `hermes:<path>:<line>` is a path in the Hermes checkout.
`H:<line>` is this directory's `__init__.py`, `S:<line>` is `_setup.py`,
`M:<line>` is `native_memory_mirror.py`. Test paths are relative to this
directory. Every citation was checked against the files at the baseline.

**Status.**

- **verified**: the host behaviour was confirmed in the pinned source, and the
  plugin meets it.
- **degraded**: the plugin works, but deviates from the host contract or loses
  something under a stated condition.
- **unsupported**: the plugin does not implement the hook, or the feature does
  not work in that setting.

## 1. Host contract record

### 1.1 Provider hooks

| Hook | Host guarantee | Plugin today | Status |
| --- | --- | --- | --- |
| `is_available` | Called at agent init before `add_provider`; falsy skips the provider and the host shows `unavailable_reason()` (hermes:agent/agent_init.py:1365-1372). Docstring: config and deps only, no network (hermes:agent/memory_provider.py:99-100). The setup picker and `hermes memory status` call it on every discoverable provider (hermes:hermes_cli/memory_setup.py:122-140, :395). | True when an endpoint comes from the scoped env, `memory.openviking.endpoint`, or a linked ovcli profile; no network (H:1424-1434). `unavailable_reason()` reads the same local config and names the cause: no endpoint configured, or a linked `ovcli.conf` that is missing, unreadable, invalid or has no `url`; it returns `""` when the provider is available (phase 1). | verified |
| `initialize` | Called once, synchronously, during `AIAgent` construction, after `add_provider` (hermes:agent/agent_init.py:1373-1374, hermes:agent/memory_manager.py:896-902). Kwargs come from hermes:agent/agent_init.py:1268-1303; `agent_context` is `cron` or `subagent` only when the platform has that name (:1276). Subagents are built with `skip_memory=True` (hermes:tools/delegate_tool.py:241), so in practice the only non-primary context is cron (hermes:cron/scheduler.py:2437-2439). The api_server reuses an initialized manager across per-turn agent rebuilds without calling `initialize` again (hermes:agent/agent_init.py:1347-1351). | Resolves the profile-bound connection, takes the run lock, probes health (anonymous probes use a 3 s timeout, H:391-393), may start a local `openviking-server` and a waiter thread (H:1564-1587), starts pending-session recovery threads, and adds a primary provider to the weak atexit registry, registering the atexit hook on the first primary `initialize`; a non-primary provider is removed from it. At exit the hook commits each registered provider's current session within one 20 s budget, below the CLI's 30 s exit watchdog (hermes:hermes_cli/cli_shutdown.py:51-99), caps each request at the time left, and starts no commit once the budget is spent (phase 1). Writes are disabled for `cron`, `subagent` and `flush` (H:133, H:1615-1616). | verified |
| `get_tool_schemas` call order | Called first in `add_provider`, before `initialize`; that call builds the routing table and core tool names are refused (hermes:agent/memory_manager.py:379-428, schemas read at :394). Called again for injection after `initialize_all` (hermes:agent/agent_init.py:1384, hermes:agent/memory_manager.py:115-158, :614-635), only when the `memory` toolset is exposed (:103-112). | Returns the same static list of six `viking_*` schemas every time, with no network (H:503, H:2905-2906). **Planned** (D1): tools come from the server's `tools/list` through a cache, named `openviking_<server name>`. | verified |
| `system_prompt_block` | Joined by `build_system_prompt` (hermes:agent/memory_manager.py:437-441) from hermes:agent/system_prompt.py:528-539, under the same gate as tool injection. The result is cached with the system prompt; the docstring requires static text (hermes:agent/memory_provider.py:116-118). | Static text built from the names `get_tool_schemas()` returns: a header, a note that `viking://` URIs are virtual, and one guidance line per registered tool. No request, and the same text for an empty store, a populated store and an unreachable server. Returns `""` only when no tool is registered (phase 1). | verified |
| `prefetch` | Runs on the turn thread's behalf in a fresh `spawn_context_thread`; the turn joins it for at most 8 s (hermes:agent/memory_manager.py:32, :457-501). After a timeout the result is dropped and later turns skip the provider until the stuck call returns (:471-486). Skipped for trivial prompts (hermes:agent/turn_context.py:876). The result is stamped into the user message's `api_content` and replayed on every later request (hermes:agent/turn_context.py:888-904). | One deadline per call: `min(7.5 s, recall_timeout_seconds)` from the start of `prefetch`, covering the connection check. The session-start block and query recall then run concurrently on one-shot `spawn_context_thread` threads, and each request is clamped to the deadline. A part that misses it is dropped from the turn's result (outcome `timeout`) and the other part is still returned. A request with a later fallback (the context rewrite request, the session-aware list search) keeps 1 s of the budget for it and is skipped when only that is left. With `recall_compress` set to `server` or `auto` and no explicit timeout, both timeouts default to the 7.5 s budget instead of 55 s. The session-start block is claimed per sid under `_session_start_lock` and latched only when it reached a turn's result, so concurrent prefetches of one sid inject it once (phase 1). | verified |
| `queue_prefetch` | Queued on the mem-sync worker after a completed, non-interrupted, non-trivial turn (hermes:run_agent.py:946-947, hermes:agent/memory_manager.py:516-525). | No-op; recall is current-query only (H:1796-1798). | verified |
| `recall_status` | Read by `describe_recall` right after a non-empty `prefetch_all`, and shown as a status line (hermes:agent/turn_context.py:880-884, hermes:agent/memory_manager.py:503-514). | Each `prefetch` records its outcome per session id under a lock: `pending` while running, then `injected`, `empty`, `timeout`, `unavailable` or `error`. A sequence number stops a call the host already abandoned from overwriting a newer one. `recall_status()` reads the most recently started prefetch and returns `RecallStatus("OpenViking", count)` only when it injected, with `count` the number of query-recall entries (0 when only the session-start block or an entry-less digest was injected); otherwise `None`. `last_recall_outcome(session_id)` exposes the category for diagnostics (phase 1). | verified |
| `sync_turn` | Submitted to one `DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="mem-sync")` in FIFO order, bound to the caller's contextvars (hermes:agent/memory_manager.py:533-596, `ctx_bound` at :563). `messages` and `turn_author` are passed only if the signature accepts them (:528-531, :548-553). Interrupted turns are never synced (hermes:run_agent.py:918-949). If the executor cannot be created the task runs inline (hermes:agent/memory_manager.py:592-596). Late submissions during shutdown are rejected (:579-582). | Returns immediately. It captures the commit scope, a client clone and the sender peer under `_client_refresh_lock` (H:2384-2392), then starts one writer thread per turn (H:2438-2439). Writers serialize on the instance `_writer_commit_lock` (H:2420), which does not guarantee turn order. A turn whose upload and retry both fail is dropped; only the pending marker survives (H:1325-1353, pinned by tests/test_standalone.py:863). **Planned** (D6, phase 5): upload synchronously on the worker, with a per-`(sid, generation)` in-memory backlog. | degraded |
| `on_turn_start` | Synchronous on the turn thread before `prefetch`; exceptions suppressed (hermes:agent/turn_context.py:866-872). Only `author_id`, `author_name` and `author_is_bot` are sent, filtered by signature (hermes:agent/memory_manager.py:654-661). | Sets the per-instance `ContextVar` `_turn_peer` (H:1375, H:1747-1748). The prefetch thread and the queued `sync_turn` inherit the value through copied contextvars. | verified |
| `identity_signature` | The gateway calls it on every inbound message, on a memoized instance that was never initialized, and swallows errors (hermes:gateway/run_agent_cache.py:76-95). The values join the agent cache key (:70-71). | Not implemented. The plugin re-resolves its connection on each access instead (H:1649-1725), so a changed credential takes effect on the next call, but it does not rebuild the cached agent. **Planned** (optional): read local config only, cache by file stat. | unsupported |
| `on_session_end` | Several callers on different threads; see 1.3. Compression calls it whether or not the session id rotates, so one conversation can see several calls. | Drains its own writer threads for the current sid for up to 10 s and skips the commit if they are still alive (H:71, H:2774-2776). Then commits under `_writer_commit_lock` if needed (H:2764-2782). Skipped entirely in non-primary contexts. | degraded |
| `on_session_switch` | Called on three kinds of thread; see 1.4. | Rotates `_session_id` synchronously under `_session_state_lock` in every context, including read-only ones (H:2802-2818). The old sid is committed on a one-off finalizer thread (H:2738-2762, H:2835-2836). `rewound=True` or an unchanged id skips rotation (H:2813). Compression resets turn accounting, re-arms the latch on an in-place sid, and re-injects the session-start block (H:2819-2830). | verified |
| `on_pre_compress` | Called synchronously before compaction (hermes:agent/conversation_compression.py:2950-2987, :3902); v1 providers get raw messages (hermes:agent/memory_manager.py:727-762). With `compression.checkpoint_required` enabled, a provider without checkpoint API v2 makes compaction fail closed (hermes:agent/conversation_compression.py:2958-2980). | Not implemented. Compaction already calls `on_session_end`, which commits the session. Users who enable `compression.checkpoint_required` cannot compact with this provider. | unsupported |
| `on_delegation` | Parent side, synchronous, after `delegate_task` results (hermes:tools/delegate_tool_results.py:330-343). | Not implemented. The delegated task and its result reach the parent transcript as a tool call and result, which `sync_turn` captures. | unsupported |
| `on_memory_write` | Synchronous on the tool thread after a committed built-in `memory` write (hermes:agent/inline_tool_executors.py:151-158 → hermes:agent/memory_manager.py:804-840 → :773-788). Metadata is passed by keyword to this signature (:765-771). | Clones the client, then enqueues to a FIFO mirror worker and returns (H:2870-2903, M:350-379, M:77-120). The worker is started through `spawn_context_thread` (M:115) and exits only after `shutdown()` (M:141-160). | verified |
| `handle_tool_call` | Dispatched sequentially (hermes:agent/tool_executor.py:1660-1666). Routing uses the table from the first `get_tool_schemas` call; exceptions become `tool_error` (hermes:agent/memory_manager.py:643-652). | Calls `_ensure_client()`, then the handler; handler errors become `tool_error` (H:2908-2917). Handlers read `self._client`. | verified |
| `shutdown` | `shutdown_all` drains the worker for at most 5 s, cancels what is left, then calls `shutdown()` in reverse order (hermes:agent/memory_manager.py:31, :848-894). Reached from `shutdown_memory_provider` (hermes:run_agent.py:897-909) and api_server eviction (hermes:gateway/platforms/api_server_memory_sessions.py:113-129). Not called on gateway soft eviction. | Sets `_shutting_down`, drains the mirror for 5 s, joins writer, finalizer and autostart threads for up to 5 s each, removes the provider from the atexit registry and releases the run lock (H:2919-2939). It does not commit; data uploaded after the last commit stays pending until the next process recovers it. **Planned** (phase 5): commit within a 5 s total budget and keep the atexit entry while data remains. | degraded |
| `backup_paths` | Called by `hermes backup` on a provider that was loaded but not initialized; must not need `initialize()` or the network (hermes:hermes_cli/backup.py:239-252, hermes:agent/memory_provider.py:203-206). | Returns the resolved `ovcli.conf` path (H:1359-1365). It reads `OPENVIKING_CLI_CONFIG_FILE` from `os.environ` (H:623-625), and returns the path even when the profile does not link ovcli. | verified |

### 1.2 Host behaviours

| Behaviour | Host guarantee | Plugin today | Status |
| --- | --- | --- | --- |
| Gateway soft eviction | The eviction thread re-enters the owning profile's scope (hermes:gateway/run_agent_cache.py:773-800), calls `commit_memory_session` (→ `on_session_end`), then `release_clients()`, which keeps the memory provider on the agent; `shutdown()` is not called (:802-834, hermes:run_agent.py:951-955). The next turn builds a new agent and a new provider instance (hermes:plugins/memory/__init__.py:192-217). | `on_session_end` commits, and removes the instance from the atexit registry when no writer is running and no turn or pending marker of the current connection generation is left uncommitted; a failed commit keeps it registered, and a later write registers it again. The registry is a `weakref.WeakSet`, so it never keeps an evicted instance alive (phase 1, tests/test_standalone.py `test_soft_eviction_cycles_do_not_grow_exit_registry`). The evicted instance keeps its mirror worker polling forever if the mirror was ever used (M:141-160); that thread holds the provider, and with it the open run-lock file. By code reading, markers of a still-locked run are not recovered by another instance until the lock is released (H:2574-2594); no test covers this. **Planned** (phase 1): mirror worker exits after an idle poll. | degraded |
| Plugin loader pre-executes sibling modules | The loader registers the package, executes every top-level `*.py` sibling in glob order, then `__init__.py`. A sibling that raises is dropped with a debug log; subdirectories are not pre-executed; a cached module is reused, so module state is process-wide (hermes:plugins/plugin_loader.py:93-135). A user install is named `_hermes_user_memory.openviking__source_<sha16>` (hermes:plugins/memory/__init__.py:81-87). | `_setup.py` and `native_memory_mirror.py` have no import-time side effects. `_setup` reaches the package through `sys.modules[__package__]` only at call time (S:22-24), and `__init__` imports it mid-module (H:1177). `__init__` registers no `atexit` hook at import; the hook is registered on the first primary `initialize` (phase 1). The mirror logger name is hard-coded to `plugins.memory.openviking` (M:29) and does not match a user install's module name. **Planned** (phase 2): `core/log.py`. | degraded |
| Bundled provider wins on a name collision | Discovery order is bundled, user, project, entry point; the first name seen wins (hermes:plugins/memory/__init__.py:1-8, :95-109, :124-136). `plugins/memory/openviking/` exists at `58d146e961`. | While the bundled copy exists, `$HERMES_HOME/plugins/openviking` is never loaded (README "Install"). The tests point `_MEMORY_PLUGINS_DIR` at an empty directory to load the external copy (tests/conftest.py:23). **Planned** (phase C): catalog cutover. | unsupported |
| `spawn_context_thread` | Returns an unstarted thread that runs under the spawner's contextvars (hermes:agent/memory_provider.py:21-33). Every background job must use it, or it runs without a profile scope (hermes:plugins/AGENTS.md:75-84). | Every plugin thread uses it: writers, finalizers and recovery (H:2443-2467), the autostart waiter (H:1497-1500), and the mirror worker (M:115-119). There is no bare `threading.Thread`. | verified |

### 1.3 `on_session_end` callers

| Caller | Host site | Thread | Worker drained first? |
| --- | --- | --- | --- |
| CLI exit | hermes:hermes_cli/cli_shutdown.py:111-124 → hermes:run_agent.py:897-909 | main thread | yes, `flush_pending(timeout=10)` |
| Gateway teardown | hermes:gateway/run_shutdown.py:1287-1302 | teardown thread | yes, `flush_pending(timeout=10)` |
| One-shot `hermes -q` | hermes:hermes_cli/oneshot.py:628-638 | main thread | no; `on_session_end` runs before `shutdown_all` drains the queued `sync_turn` |
| Compression | hermes:agent/conversation_compression.py:3710 via `commit_memory_session` (hermes:run_agent.py:911-916) | turn thread | no |
| Gateway soft eviction | hermes:gateway/run_agent_cache.py:802-815 | eviction thread | no |
| TUI session lifecycle | hermes:tui_gateway/session_lifecycle.py:394 | TUI thread | no |
| CLI `/new` with history | hermes:agent/memory_manager.py:667-698 | mem-sync worker, same task as the switch | FIFO: runs after queued `sync_turn` tasks |
| api_server eviction | hermes:gateway/platforms/api_server_memory_sessions.py:113-129 | eviction thread | not called; only `flush_pending` and `shutdown_all` |

Consequences today, by code reading: under `hermes -q` the last turn's
`sync_turn` can still be queued when `on_session_end` commits. It is then
uploaded during the `shutdown_all` drain, `shutdown()` does not commit it, and it
stays pending until a later process with the same connection recovers it. On the compression
path `on_session_end` can hold the turn thread for the 10 s writer drain plus a
commit request with the 30 s default timeout.

### 1.4 `on_session_switch` thread kinds

| Trigger | Host site | Thread | Arguments | Plugin result |
| --- | --- | --- | --- | --- |
| `/new` with history | hermes:hermes_cli/cli_session_mixin.py:576-579 → hermes:agent/memory_manager.py:667-698 | mem-sync worker, after `on_session_end` in the same task | `reset=True`, `reason="new_session"` | `on_session_end` commits the old sid; the finalizer then finds it latched |
| `/new` without history | hermes:hermes_cli/cli_session_mixin.py:580-583 | command thread | `reset=True` | rotate; finalizer commits the old sid if it has data |
| `/resume`, `/branch` | hermes:hermes_cli/cli_commands_mixin.py:349-351 | command thread | `reset=False`, `reason` | same as above |
| CLI `/undo` | hermes:hermes_cli/cli_session_mixin.py:811-813 | command thread | same id, `rewound=True` | no rotation |
| TUI `/undo` | hermes:tui_gateway/methods_tools.py:899-901 | TUI method thread | passes `session_key` as the id, `rewound=True` | no rotation, so the unusual id is ignored |
| Compression boundary | hermes:agent/conversation_compression.py:3469-3474, after the commit at :3710 | turn thread | `reset=False`, `reason="compression"`; same id when compacting in place | rotation mode: rotate, old sid already latched. In-place mode: turn count reset and latch re-armed (H:2819-2830) |
| Compression child adoption | hermes:agent/conversation_compression.py:1719-1723 | turn thread | `reason="compression"` | rotate |

## 2. Knob mapping (plan decision D7)

Hermes does not use the shared knob schema in
`../memory-plugin-shared/lib/config-schema.mjs`. It declares its own
`_CONFIG_SCHEMA` (H:91-119), which the host renders in `hermes memory setup` and
the dashboard. Typed settings resolve as scoped env var first, then
`memory.openviking.<key>` in `config.yaml`, then the default, clamped to the
range (H:1973-1997). Connection values resolve as scoped env, linked ovcli
profile, `config.yaml`, default; the API key never comes from `config.yaml`
(H:881-910). The shared resolver orders layers as default, files, env
(config-schema.mjs:344). Bare line numbers in the shared-knob column refer to
`config-schema.mjs`.

| Hermes setting and env var | Hermes default, range | Shared knob | Shared default, range, env | Difference |
| --- | --- | --- | --- | --- |
| `endpoint`, `OPENVIKING_ENDPOINT` | `http://127.0.0.1:1933` | none (resolved by `credentials.mjs`) | — | Same meaning. |
| `api_key`, `OPENVIKING_API_KEY` (secret; `.env` or ovcli) | empty | `apiKey` (:45) | empty, no env | Same meaning. |
| `account`, `user`; `OPENVIKING_ACCOUNT`, `OPENVIKING_USER` | `default` | `accountId`, `userId` (:46-47) | empty | Hermes sends them only without an API key, or on a trusted-mode retry (H:304-339). |
| `agent`, `OPENVIKING_AGENT` | empty | `peerId` (:66) | empty, `OPENVIKING_PEER_ID` | Same header (`X-OpenViking-Actor-Peer`), different env var. |
| `use_ovcli_config`, `ovcli_config_path`, `OPENVIKING_CLI_CONFIG_FILE` | off | none | — | Shared plugins always read `ovcli.conf`; Hermes reads it only when linked, and only its connection fields (H:638-652, H:873-878). |
| `recall_scope`, `OPENVIKING_RECALL_SCOPE` | unset; `shared` \| `peer` | `recallPeerScope` (:85) | `all`; `all` \| `actor` | Unset keeps the pre-preset request shape. `peer` adds explicit roots, including when no sender is known (H:1852-1868). |
| `recall_compress`, `OPENVIKING_RECALL_COMPRESS` | `off`; `off` \| `server` \| `auto` (booleans accepted, H:1950-1955) | `recallCompress` (:111) | `off` (`auto` for Claude Code and Codex), same env | Hermes has no client compressor; `auto` sends `rewrite: "auto"`. Turning it on without explicit timeouts lets both recall timeouts use the whole 7.5 s prefetch budget; a timed-out rewrite request falls back to search without rewrite in the 1 s it left (phase 1). |
| `commit_token_threshold`, `OPENVIKING_COMMIT_TOKEN_THRESHOLD` | 20000; 1000-1000000 | `commitTokenThreshold` (:150) | 20000; 1000-1000000, same env | Identical. |
| `recall_limit`, `OPENVIKING_RECALL_LIMIT` | 6; 1-100 | `recallLimit` (:79) | 10; 1-50, same env | Same env var, different default and range. |
| `recall_score_threshold`, `OPENVIKING_RECALL_SCORE_THRESHOLD` | 0.15; 0-1 | `scoreThreshold` (:80) | 0.35; 0-1, `OPENVIKING_SCORE_THRESHOLD` | Different default and env var. |
| `recall_max_injected_chars`, `OPENVIKING_RECALL_MAX_INJECTED_CHARS` | 4000 chars; 100-50000 | `recallMaxTokens` (:87), `recallTokenBudget` (:82) | 1600 / 2000 tokens | Hermes counts characters and sends `max_tokens = chars // 4`, clamped to 64-32000 (H:1878). |
| `profile_token_budget`, `OPENVIKING_PROFILE_TOKEN_BUDGET` | 6000; 500-50000 | `profileTokenBudget` (:158) | 10000; 500-50000, same env | Same env var, different default. |
| `recall_timeout_seconds`, `OPENVIKING_RECALL_TIMEOUT_SECONDS` | 4.0 s; 0.25-60 | `recallTimeoutMs` (:89) | 120000 ms; 1000-600000 | Seconds vs milliseconds. Hermes caps it at 7.5 s, below the host's 8 s prefetch join (phase 1). |
| `recall_request_timeout_seconds`, `OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS` | 3.0 s; 0.25-60 | `timeoutMs` (:50), `recallContextTimeoutMs` (:91) | 15000 ms / 0 | No one-to-one match. |
| `recall_full_read_limit`, `OPENVIKING_RECALL_FULL_READ_LIMIT` | 2; 0-100 | none | — | List-mode recall only. |
| `recall_prefer_abstract`, `OPENVIKING_RECALL_PREFER_ABSTRACT` | false | `recallPreferAbstract` (:84) | true, same env | Same env var, opposite default. |
| `recall_resources`, `OPENVIKING_RECALL_RESOURCES` | false | none | — | Controls `context_type` and resource roots (H:1865-1868, H:1879). |

Fixed in Hermes code, configurable in the shared schema:

| Hermes behaviour | Location | Shared knob |
| --- | --- | --- |
| Commit with `keep_recent_count: 0` | H:2722 | `commitKeepRecentCount` (:151), default 10 |
| Minimum recall query length 5 characters | H:76 | `minQueryLength` (:81), default 3 |
| Context-mode `purpose: "coding"` | H:1875 | none |

Shared knobs with no Hermes counterpart include `enabled`, `autoRecall`,
`autoCapture`, the `capture*` family, the `takeover*`, `resume*` and `skill*`
families, and `debug`. The host already provides the on/off switch
(`memory.provider`) and owns session resumption and compaction.

Because several env var names are shared, a user who exports
`OPENVIKING_RECALL_LIMIT`, `OPENVIKING_PROFILE_TOKEN_BUDGET` or
`OPENVIKING_RECALL_PREFER_ABSTRACT` for another plugin changes Hermes too, with
Hermes's own range and meaning. Hermes reads env through `get_secret`, so under
a multiplexing gateway only the profile's `.env` applies.

**Why `hermes` is not in `HARNESS_KEYS`** (config-schema.mjs:252-264):

1. Hermes is a Python in-process provider and cannot run the JS resolver. It
   never reads the `plugin` or `plugin.hermes` sections of `ovcli.conf`
   (H:638-652), so registering the key would make the JS doctor accept
   configuration that nothing consumes.
2. A per-harness entry can only change a knob's default (`knobDefault`,
   config-schema.mjs:275-278). It cannot change a range, an env var name or a
   unit, and those are exactly where Hermes differs.
3. The host owns Hermes's configuration entry points: `get_config_schema`,
   `save_config`, `post_setup` and the dashboard (H:1436-1489). A second
   configuration source would split one setting across two files.

This table is the maintained record of the differences. Changing a Hermes
default or env var name is a user-visible change and needs a release note.

## 3. Commit lifecycle invariants

These are the behaviours the refactor must keep. Rows 1-10 come from the plan's
invariant table; rows 11-24 were added by reading
`tests/test_standalone.py`, `tests/test_gateway_recall.py` and
`tests/test_native_memory_mirror.py`. "Not pinned" means no current test fails
if the invariant breaks; phase 5 must add a test before replacing the code.
Tests at line 770 and later in `test_standalone.py` reach private state
(`_turn_count`, `_state_path`, `_has_committed_session`) through hand-wired
providers. Both files passed when this record was written (84 tests).

| # | Invariant | Enforced by | Pinned by |
| --- | --- | --- | --- |
| 1 | Queued uploads, their retries and their commits use the identity and connection generation captured when the turn was captured, even after a reload. | `_CommitScope` (H:1269-1279), `_capture_commit_scope` (H:1515-1527), capture in `sync_turn` (H:2384-2392), retries reuse `self.client` (H:1325-1353) | tests/test_standalone.py:905, :1057, :1071, :1130, :1171; mirror and recall identity at :604 |
| 2 | Each turn's sender peer is fixed at capture and is not changed by a later turn. | `_turn_peer` (H:1375, H:1747-1752), `sync_turn` (H:2392) | tests/test_gateway_recall.py:164 (:212-216), :233, :341 |
| 3 | A commit never overtakes an upload of the same sid that is still running. | `_drain_writers` before commit (H:2753, H:2774), `_writer_commit_lock` (H:2420, H:2756, H:2777) | tests/test_standalone.py:798 |
| 4 | No new commit starts after `shutdown()`. | `_shutting_down` (H:1414, H:2651, H:2681, H:2694, H:2749, H:2757) | Not pinned |
| 5 | Recovery commits only markers whose owner run is dead and whose connection fingerprint matches; other markers stay untouched. | H:2656-2689; owner lock H:2574-2594; fingerprint H:2663 | Fingerprint part: tests/test_standalone.py:1130 (:1164-1166). Dead-owner part: not pinned (the test releases the lock through `shutdown()` at :1151 before recovering) |
| 6 | After in-place compression (same sid), later writes can still trigger a commit. | Latch re-armed (H:2826-2830) | Not pinned |
| 7 | After the old sid was committed (by compression or `on_session_end`), the rotation finalizer does not commit it again. | Latch `scope.committed` (H:2504-2514) checked first in `_session_needs_commit` (H:2703-2708) | Latch after a threshold commit: tests/test_standalone.py:770 (:794-795). Compression path: not pinned |
| 8 | A skipped commit is made up later, by asking the server for `pending_tokens` when the local turn count is 0. | H:2709-2715 | Not pinned |
| 9 | Non-primary contexts still rotate the sid, so recall and search use the current session. | Rotation independent of `_writes_enabled` (H:2802-2818) | tests/test_standalone.py:962 |
| 10 | For a sid this process never wrote, the plugin confirms `pending_tokens > 0` with the server before committing. | Same code as row 8 (H:2709-2715) | Not pinned |
| 11 | Below the token threshold no commit is sent. Crossing it commits with `keep_recent_count: 0`, resets the turn count, latches the sid and deletes the marker. A later write re-arms the latch and rewrites the marker, and the next crossing commits again. `pending_tokens` is read both flat and under `result`. | H:2417-2431, H:2691-2701, H:2717-2736 | tests/test_standalone.py:770 |
| 12 | A failed threshold commit keeps the marker and the turn count, and the next turn retries it. | H:2730-2736 | tests/test_standalone.py:825 |
| 13 | A failed or malformed `pending_tokens` lookup never re-uploads the turn and keeps the marker. | H:2697-2701 | tests/test_standalone.py:848 |
| 14 | A failed upload triggers no threshold lookup and no commit; the marker stays and the next successful turn commits. The failed turn's messages are lost. | `_TurnUpload.run` returns `None` (H:1339-1342), H:2430-2431 | tests/test_standalone.py:863 |
| 15 | The threshold commit runs off the calling thread. A turn that lands while that commit is in flight stays pending and unlatched, and `on_session_end` commits it. | `_finalize_session_async` (H:2738-2762) | tests/test_standalone.py:991 |
| 16 | `on_session_switch` commits the old sid even below the threshold, and rotates to the new sid. | H:2835-2836 | tests/test_standalone.py:922 |
| 17 | `cron`, `subagent` and `flush` contexts make no upload, commit or mirror request and write no marker. | `_writes_enabled` (H:1615-1616; checks at H:2376, H:2767, H:2878) | tests/test_standalone.py:933 |
| 18 | After a reload, the old generation's finalizer neither clears the new generation's marker nor suppresses its finalizer; each connection gets exactly one commit. Holds for a changed user and a changed endpoint. | Per-scope `finalizing` and `committed` sets (H:2644-2654), per-generation marker name (H:2522-2524) | tests/test_standalone.py:1071 |
| 19 | Returning to an earlier identity (A → B → A) starts a new generation; the first A generation's commit does not clear the new one's marker, and only the new generation's turns are counted. | New `_CommitScope` per client (H:1517-1526) | tests/test_standalone.py:1171 |
| 20 | Pending markers are named per sid and generation and never contain the API key. | `atomic_json_write` of `session_id`, `owner_run_id`, `connection_key` (H:2610-2611) | tests/test_standalone.py:1130 (:1149-1150) |
| 21 | Through the real `MemoryManager`, every turn synced on the mem-sync worker is archived once `on_session_end` commits, in order, and the marker is removed; assistant messages carry the configured assistant peer. | `sync_turn`, `on_session_end` | tests/test_gateway_recall.py:164 (:198-223) |
| 22 | The upload fallback chain (batch retry, plain-text fallback, per-message fallback) keeps the captured author. | `_TurnUpload` (H:1282-1353) | tests/test_gateway_recall.py:233 |
| 23 | Tool calls use the configured assistant peer, never the turn's sender. | Tools use `self._client` | tests/test_gateway_recall.py:164 (:225-227) |
| 24 | Mirror writes stay in FIFO order and their registry mappings are isolated by connection fingerprint. | M:54-64, M:77-160 | tests/test_native_memory_mirror.py:251, :457 |

Commit and recovery paths not covered by any test: recovery of legacy markers without an owner (H:2582-2586), the
deferred-commit drain timeout (H:2753-2755), and the `on_session_end` drain
timeout (H:2774-2776).

## 4. Target module layout (planned)

Nothing in this section exists yet. The provider class and `register()` stay in
`__init__.py`; the rest moves into a `core/` subpackage, which the host installs
but does not pre-execute (section 1.2).

```text
examples/hermes-plugin/
  __init__.py          register() and OpenVikingMemoryProvider: hooks to services
  _setup.py            setup wizard
  cli.py               hermes openviking doctor / status
  core/
    host.py            every Hermes import and its version-compatibility branches
    log.py             logger named after the provider's package
    settings.py        config schema, typed settings, Settings snapshot
    connection.py      ConnectionSnapshot, connection generation, layered resolution, profile env
    ovcli.py  endpoint.py  envfile.py  local_server.py  health.py
    http.py            headers, REST requests, errors, retry classification
    recall.py          recall routes, context-mode requests, block format, recall_status
    recall_list.py     list-mode recall (moved unchanged)
    profile.py         session-start block
    transcript.py      Hermes messages to OpenViking parts
    session_writer.py  upload, backlog, commit
    state_store.py     pending markers, run lock, recovery
    mcp_bridge.py      MCP connection, tools/list, tools/call, result conversion
    tools.py           tool catalog cache, naming, local wrappers, context filtering
    mirror.py          native memory mirror
```

`quick_local.py` from #5465 is placed after that change merges.

Dependency rules, to be enforced by an import-boundary test:

- `core/*` does not import the package `__init__` or `_setup`.
- Only `core/host.py` imports `agent.*`, `hermes_cli.*`, `hermes_constants`,
  `tools.*` or `utils`.
- Importing any module registers no `atexit` hook, starts no thread, opens no
  connection and reads no config. `mcp` and `httpx2` are imported lazily on the
  call path.
- The first 8 KB of `__init__.py` keep the word `MemoryProvider`, which
  discovery requires (hermes:plugins/memory/__init__.py:64-74).
- `core/*` gets its logger from `core/log.py`, named after the provider's
  module, because tests filter logs by `type(provider).__module__`.

## 5. Known deviations and unsupported platforms

**Platforms without `fcntl` (Windows): crash recovery unsupported.** `fcntl` is
optional (H:48-51). Without it the run lock is skipped (H:2558-2560), and
`_mark_session_pending` refuses to write a marker without a run lock
(H:2603-2605), so new sessions leave no marker and a crash loses their
uncommitted data. Only legacy markers without an owner are still recovered
(H:2582-2586). Normal exits still commit through `on_session_end` and the atexit
hook. Not planned for this refactor.

**Current deviations from the host contract**, each covered in section 1:

| Deviation | Location | Planned fix |
| --- | --- | --- |
| Mirror worker never exits without `shutdown()` | M:141-160 | Phase 1 |
| `identity_signature` missing | — | Phase 1 (optional) |
| One writer thread per turn; order not guaranteed; failed turns dropped | H:2417-2439, H:1325-1353 | Phase 5 |
| `shutdown()` does not commit | H:2919-2939 | Phase 5 |
| `on_pre_compress` and `on_delegation` missing | — | Not planned |

**Wire and packaging deviations.**

- Requests send both `X-API-Key` and `Authorization: Bearer` (H:313).
  **Planned** (phase 3): Bearer only.
- `User-Agent` carries the Hermes version, not the plugin version (H:43, H:63).
  **Planned** (phase 3).
- `plugin.yaml` declares no hooks. The host reads `hooks` as `provides_hooks`
  (hermes:hermes_cli/plugins_manifest.py:509), which lists `register_hook`
  hooks; memory lifecycle hooks are provider methods, and upstream commit
  `72ee40fa68` removed these declarations from the bundled manifests. Its
  `pip_dependencies` only feeds the dashboard's dependency list
  (hermes:hermes_cli/web_server_memory.py:61-63); installs use
  `pyproject.toml`, which takes precedence
  (hermes:pm/plugin_declarations.py:132-154). The two lists are kept
  identical (phase 1).
- `_setup.py` imports private names from `hermes_cli.memory_setup`
  (S:48, S:422), and profile resolution reads the private
  `tui_gateway.launch_profile_policy._snapshot` (H:854-862).
  These can break on a host upgrade; CI pins the host. **Planned**: isolate in
  `core/host.py`.

**Deviations from the shared plugin spec.**

- Tools are hand-written REST wrappers named `viking_*`, not the server's
  `tools/list`. **Planned** (D1, phase 6): MCP bridge with `openviking_*` names.
- Configuration does not come from the shared schema; see section 2.
- There is no URI guard: the host gives memory providers no pre-tool event.
- Non-primary (cron) contexts still run startup recovery, and explicit tools
  keep their write access (README "Non-primary contexts"). This is intended
  (D9).
