import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// pi loads extensions through the jiti it bundles, so that is the loader this
// smoke check has to use — a plain `import` of index.ts would not resolve the
// TypeScript or the `.js`-suffixed relative specifiers.
const EXTENSION_DIR = dirname(dirname(fileURLToPath(import.meta.url)));

/**
 * jiti, wherever pi is installed: the global homebrew prefix on this machine,
 * a local node_modules in CI, or an explicit override. Without it these tests
 * skip rather than silently testing nothing.
 */
function findJiti() {
  const candidates = [process.env.PI_JITI_PATH || ""];
  try {
    const require = createRequire(join(EXTENSION_DIR, "package.json"));
    candidates.push(
      require.resolve("@earendil-works/pi-coding-agent/node_modules/jiti/lib/jiti.mjs"),
    );
  } catch {
    // not installed as a dependency here
  }
  candidates.push(
    "/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/jiti/lib/jiti.mjs",
    "/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/jiti/lib/jiti.mjs",
  );
  return candidates.find((path) => path && existsSync(path)) ?? "";
}

const JITI_PATH = findJiti();

/** The pi surface index.ts touches, and nothing more. */
function fakePi(calls) {
  return {
    on(event, handler) {
      calls.events.push(event);
      calls.handlers.set(event, handler);
    },
    registerTool(tool) { calls.tools.push(tool?.name); calls.toolDefs.set(tool?.name, tool); },
    registerCommand(name, def) { calls.commands.push(name); calls.commandDefs.set(name, def); },
    appendEntry(customType, data) { calls.entries.push({ customType, data }); },
    getAllTools() { return []; },
    sendMessage(message, options) { calls.sent.push({ message, options }); },
  };
}

/** The `ctx` shape the handlers read: session manager, ui, context usage. */
function fakeCtx(overrides = {}) {
  const branch = overrides.branch ?? [];
  return {
    sessionManager: {
      getSessionId: () => overrides.sessionId ?? "pi-session-abc",
      getBranch: () => branch,
      getLeafId: () => overrides.leafId ?? "entry-9",
    },
    ui: {
      notify: (...args) => (overrides.notified ?? []).push(args),
      setStatus: (...args) => (overrides.statuses ?? []).push(args),
    },
    getContextUsage: () => overrides.usage ?? { tokens: 1000, contextWindow: 262144, percent: 1 },
  };
}

/** Load index.ts and run it against a fake pi, with OV pointed at a dead port. */
async function loadExtension() {
  const { createJiti } = await import(JITI_PATH);
  const jiti = createJiti(import.meta.url, { interopDefault: true });
  const mod = await jiti.import(join(EXTENSION_DIR, "index.ts"), { default: true });
  assert.equal(typeof mod, "function");

  const calls = {
    events: [],
    handlers: new Map(),
    tools: [],
    toolDefs: new Map(),
    commands: [],
    commandDefs: new Map(),
    entries: [],
    sent: [],
  };
  await mod(fakePi(calls));
  return calls;
}

const OV_ENV = {
  // Port 1 refuses immediately: the health check must fail fast and the
  // extension must still arm its offline half.
  OPENVIKING_URL: "http://127.0.0.1:1",
  OPENVIKING_API_KEY: "test-key",
};

function withDeadServer(t) {
  const previous = {};
  for (const [key, value] of Object.entries(OV_ENV)) {
    previous[key] = process.env[key];
    process.env[key] = value;
  }
  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });
}

test("index.ts loads through pi's jiti and registers its surface", { skip: !JITI_PATH }, async (t) => {
  withDeadServer(t);
  const calls = await loadExtension();

  for (const event of [
    "session_start",
    "before_agent_start",
    "context",
    "tool_call",
    "turn_end",
    "session_before_compact",
    "session_compact",
    "session_shutdown",
    "agent_end",
  ]) {
    assert.ok(calls.events.includes(event), `missing handler for ${event}`);
  }
  assert.deepEqual(calls.commands, ["viking"]);
});

test("session_start registers the six viking tools and the three context window tools", { skip: !JITI_PATH }, async (t) => {
  withDeadServer(t);
  const calls = await loadExtension();

  await calls.handlers.get("session_start")({ type: "session_start" }, fakeCtx());
  // session_start is fire-and-forget; before_agent_start awaits the same chain.
  await calls.handlers.get("before_agent_start")(
    { type: "before_agent_start", prompt: "hello", systemPrompt: "BASE" },
    fakeCtx(),
  );

  assert.deepEqual(calls.tools, [
    "viking_search",
    "viking_read",
    "viking_browse",
    "viking_remember",
    "viking_forget",
    "viking_add_resource",
    "new_context",
    "history",
    "get_context_remaining",
  ]);
  // `viking_archive_expand` read viking://session/{sid}, a namespace the server
  // does not have; `history` replaced it and it must not come back.
  assert.equal(calls.tools.includes("viking_archive_expand"), false);
  assert.equal(calls.toolDefs.get("new_context").executionMode, "sequential");
  assert.equal(typeof calls.toolDefs.get("history").execute, "function");
});

test("before_agent_start ships the window guidance and one status line, server or no server", { skip: !JITI_PATH }, async (t) => {
  withDeadServer(t);
  const calls = await loadExtension();

  const result = await calls.handlers.get("before_agent_start")(
    { type: "before_agent_start", prompt: "hello", systemPrompt: "BASE" },
    fakeCtx(),
  );

  assert.ok(result.systemPrompt.startsWith("BASE\n\n"));
  assert.match(result.systemPrompt, /<context-window-management>/);
  assert.match(result.systemPrompt, /new_context, history, get_context_remaining\./);
  // Codex hides this machinery from the user; the demo deliberately does not.
  assert.equal(/never (mention|disclose|reveal)/i.test(result.systemPrompt), false);

  assert.equal(result.message.customType, "ov-context-status");
  assert.equal(result.message.display, true);
  assert.match(result.message.content, /^\[context-status\] window w1/);
});

test("the context hook returns the messages untouched while no window is armed", { skip: !JITI_PATH }, async (t) => {
  withDeadServer(t);
  const calls = await loadExtension();
  await calls.handlers.get("before_agent_start")(
    { type: "before_agent_start", prompt: "hello", systemPrompt: "BASE" },
    fakeCtx(),
  );

  const messages = [
    { role: "user", content: "first", timestamp: 1 },
    { role: "assistant", content: "second", timestamp: 2 },
  ];
  const result = await calls.handlers.get("context")({ type: "context", messages }, fakeCtx());
  assert.deepEqual(result.messages, messages);
});

// ---------------------------------------------------------------------------
// End to end through the wiring: a fake OpenViking server, a real reset, the
// cut the next sampling sees, and the signals that follow it.
// ---------------------------------------------------------------------------

const ROOT = "viking://user/u1/sessions/pi-session-abc";
const ARCHIVE = `${ROOT}/history/archive_001`;

const ARCHIVED_MESSAGES = [
  { id: "m1", role: "user", parts: [{ type: "text", text: "start phase one" }], created_at: "2026-09-10T01:00:00Z" },
  { id: "m2", role: "assistant", parts: [{ type: "text", text: `noted ${"X".repeat(9000)}` }], created_at: "2026-09-10T01:00:05Z" },
]
  .map((row) => JSON.stringify(row))
  .join("\n");

/** The handful of OpenViking routes one reset (and `history`) walks through. */
function startFakeOv(t) {
  const seen = [];
  const greps = [];
  const server = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    const url = new URL(req.url, "http://ov.invalid");
    seen.push(`${req.method} ${url.pathname}`);
    const send = (payload, code = 200) => {
      res.writeHead(code, { "content-type": "application/json" });
      res.end(JSON.stringify(payload));
    };
    if (url.pathname === "/health") return send({ status: "ok" });
    if (/\/messages\/batch$/.test(url.pathname)) return send({ result: { added: 1 } });
    if (/\/messages$/.test(url.pathname)) return send({ result: { ok: true } });
    if (/\/commit$/.test(url.pathname)) {
      return send({
        result: { status: "accepted", archived: 2, task_id: "task-1", archive_uri: ARCHIVE },
      });
    }
    if (req.method === "GET" && /^\/api\/v1\/sessions\/[^/]+$/.test(url.pathname)) {
      return send({ result: { uri: ROOT } });
    }
    if (url.pathname === "/api/v1/fs/ls") {
      return send({
        result: [{ uri: ARCHIVE, isDir: true, modTime: "2026-09-10T01:00:00Z", abstract: "# Working Memory\nphase one" }],
      });
    }
    if (url.pathname === "/api/v1/content/read") {
      const uri = url.searchParams.get("uri") || "";
      if (uri.endsWith(".overview.md")) return send({ result: "# Working Memory\n\nphase one" });
      if (uri.endsWith("messages.jsonl")) return send({ result: ARCHIVED_MESSAGES });
      return send({ error: { message: "NOT_FOUND" } }, 404);
    }
    if (url.pathname === "/api/v1/search/grep") {
      greps.push(JSON.parse(body || "{}"));
      return send({
        result: {
          matches: [
            { uri: `${ARCHIVE}/messages.jsonl`, line: 2, content: "Z".repeat(900) },
            { uri: `${ARCHIVE}/.overview.md`, line: 4, content: "codename ZEPHYR-9942" },
          ],
        },
      });
    }
    // Recall: a block the injector would prepend to the newest user message.
    if (url.pathname === "/api/v1/search/recall") {
      return send({ result: { rendered: "- recalled memory", entries: [{ uri: "viking://x" }] } });
    }
    return send({ error: { message: "not found" } }, 404);
  });
  t.after(() => server.close());
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ port: server.address().port, seen, greps }));
  });
}

/** Isolated env: a live fake server and a pending queue that is not the user's. */
function withFakeServer(t, port) {
  const dir = mkdtempSync(join(tmpdir(), "ov-index-"));
  const previous = {};
  const env = {
    OPENVIKING_URL: `http://127.0.0.1:${port}`,
    OPENVIKING_API_KEY: "test-key",
    OPENVIKING_PENDING_DIR: dir,
  };
  for (const [key, value] of Object.entries(env)) {
    previous[key] = process.env[key];
    process.env[key] = value;
  }
  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    rmSync(dir, { recursive: true, force: true });
  });
}

test("a reset archives the window, cuts it out of the next context and stays quiet in the same turn", { skip: !JITI_PATH }, async (t) => {
  const { port } = await startFakeOv(t);
  withFakeServer(t, port);
  const calls = await loadExtension();

  const branch = [
    { type: "message", id: "e1", message: { role: "user", content: "start phase one PAD1", timestamp: 1 } },
    { type: "message", id: "e2", message: { role: "assistant", content: "phase one done", timestamp: 2 } },
    { type: "message", id: "e3", message: { role: "user", content: "wrap it up", timestamp: 3 } },
    {
      type: "message",
      id: "e4",
      message: {
        role: "assistant",
        content: [{ type: "toolCall", id: "call-1", name: "new_context", arguments: {} }],
        timestamp: 4,
      },
    },
  ];
  // Pi reports the *untransformed* session, so right after the cut this number
  // still describes the window that was just archived.
  const usage = { tokens: 250000, contextWindow: 262144, percent: 95 };
  const ctx = fakeCtx({ branch, usage });

  await calls.handlers.get("before_agent_start")(
    { type: "before_agent_start", prompt: "wrap it up", systemPrompt: "BASE" },
    ctx,
  );

  const result = await calls.toolDefs.get("new_context").execute(
    "call-1",
    { reason: "phase one is done", notes: "codename ZEPHYR-9942", next_steps: ["start phase two"] },
    undefined,
    () => {},
    ctx,
  );
  assert.equal(result.isError, false);
  // A reset must never terminate the agent loop: the next sampling is what
  // consumes it.
  assert.equal("terminate" in result, false);
  assert.match(result.content[0].text, /Context window w2 is open/);

  // The turn the reset happened in ends here. Its assistant message predates
  // the reset, so no reminder may fire off the stale usage — one would tell the
  // model to reset again immediately and burn the level for the whole window.
  await calls.handlers.get("turn_end")(
    {
      type: "turn_end",
      turnIndex: 0,
      message: branch[3].message,
      toolResults: [{ role: "toolResult", toolCallId: "call-1" }],
    },
    ctx,
  );
  assert.deepEqual(calls.sent, [], "no reminder in the turn that reset the window");

  // The next sampling: the archived window is gone, the header is the context.
  const messages = [
    { role: "user", content: "start phase one PAD1", timestamp: 1 },
    { role: "assistant", content: "phase one done", timestamp: 2 },
    { role: "user", content: "wrap it up", timestamp: 3 },
    {
      role: "assistant",
      content: [{ type: "toolCall", id: "call-1", name: "new_context", arguments: {} }],
      timestamp: 4,
    },
    { role: "toolResult", toolCallId: "call-1", toolName: "new_context", content: "ok", timestamp: 5 },
  ];
  const cut = await calls.handlers.get("context")({ type: "context", messages }, ctx);
  assert.equal(cut.messages.length, 1);
  assert.equal(cut.messages[0].role, "user");
  const header = String(cut.messages[0].content);
  assert.ok(header.startsWith('<openviking-context source="context-window">'), header.slice(0, 80));
  assert.match(header, /id="w2" previous="w1" archive="archive_001"/);
  assert.match(header, /ZEPHYR-9942/);
  assert.match(header, /<working-memory[^>]*>/); // the fake's Working Memory
  assert.match(header, /<pending-request>\s*wrap it up/);
  // The archived turns are gone; only the notes and the pending request survive.
  assert.equal(header.includes("PAD1"), false);
  // Recall runs after the cut and its guard leaves the frozen header alone.
  assert.equal(header.includes("recalled memory"), false);

  // One window entry, so `pi -c` can rebuild the same anchor and header.
  const windowEntries = calls.entries.filter((e) => e.customType === "ov-context-window");
  assert.equal(windowEntries.length, 1);
  assert.equal(windowEntries[0].data.archiveId, "archive_001");
  assert.equal(windowEntries[0].data.anchorToolCallId, "call-1");
  assert.equal(windowEntries[0].data.overviewReady, true);

  // A later turn, whose assistant answered inside the new window: now the
  // reported usage is about this window again, so a full window does warn —
  // once, and steered because the batch is still running.
  await calls.handlers.get("turn_end")(
    {
      type: "turn_end",
      turnIndex: 1,
      message: { role: "assistant", content: "working", timestamp: Date.now() + 1000 },
      toolResults: [{ role: "toolResult", toolCallId: "call-2" }],
    },
    ctx,
  );
  assert.equal(calls.sent.length, 1);
  assert.equal(calls.sent[0].message.customType, "ov-context-reminder");
  assert.equal(calls.sent[0].options.deliverAs, "steer");
  assert.match(calls.sent[0].message.content, /new_context/);

  // At the end of a turn with no tool results the reminder waits for the next
  // turn instead of jumping the queue — and it never repeats a level.
  await calls.handlers.get("turn_end")(
    {
      type: "turn_end",
      turnIndex: 2,
      message: { role: "assistant", content: "still working", timestamp: Date.now() + 2000 },
      toolResults: [],
    },
    ctx,
  );
  assert.equal(calls.sent.length, 1, "each level fires once per window");
});

test("history reads the archived window back and get_context_remaining reports the new one", { skip: !JITI_PATH }, async (t) => {
  const { port, greps } = await startFakeOv(t);
  withFakeServer(t, port);
  const calls = await loadExtension();

  const branch = [
    { type: "message", id: "e1", message: { role: "user", content: "start phase one", timestamp: 1 } },
    {
      type: "message",
      id: "e2",
      message: {
        role: "assistant",
        content: [{ type: "toolCall", id: "call-1", name: "new_context", arguments: {} }],
        timestamp: 2,
      },
    },
  ];
  const ctx = fakeCtx({ branch, usage: { tokens: 40000, contextWindow: 262144, percent: 15 } });
  await calls.handlers.get("before_agent_start")(
    { type: "before_agent_start", prompt: "start phase one", systemPrompt: "BASE" },
    ctx,
  );
  await calls.toolDefs.get("new_context").execute(
    "call-1",
    { reason: "phase one is done", notes: "codename ZEPHYR-9942" },
    undefined,
    () => {},
    ctx,
  );

  const history = calls.toolDefs.get("history");
  const run = async (params) => (await history.execute("h", params, undefined, () => {}, ctx)).content[0].text;

  // w1 is the oldest archive; the open window is listed but not readable.
  const windows = await run({ action: "list_windows" });
  assert.match(windows, /^w1 {2}archive_001 {2}2026-09-10T01:00:00Z {2}# Working Memory$/m);
  assert.match(windows, /^w2 {2}\(current\) {2}this window is still open and not archived$/m);

  const items = await run({ action: "list_items", window: "w1" });
  assert.match(items, /\[id: w1:0\] {2}user 2026-09-10T01:00:00Z {2}start phase one/);
  assert.match(items, /\[id: w1:1\].*\[truncated, \d+ more characters\]/);
  assert.match(items, /Showing 0-1 of 2 items \(end of window\)\./);

  // read_item goes up to historyItemMaxChars (8000) rather than the list clip.
  const item = await run({ action: "read_item", item: "w1:1" });
  assert.match(item, /^\[id: w1:1\] {2}assistant {2}2026-09-10T01:00:05Z$/m);
  assert.ok(item.length > 8000 && item.length < 8200, `unexpected length ${item.length}`);
  assert.match(await run({ action: "read_item", item: "w1:7" }), /has 2 items \(0-1\); 7 is out of range/);
  assert.match(await run({ action: "list_items", window: "w7" }), /No archived window named "w7"/);

  const found = await run({ action: "search_contents", query: "ZEPHYR" });
  assert.match(found, /^w1 {2}archive_001$/m);
  // Long grep lines are clipped, and a messages.jsonl hit points at read_item.
  assert.match(found, /messages\.jsonl:2 {2}Z+… \[truncated, \d+ more characters\] {2}→ \{"action":"read_item","item":"w1:1"\}/);
  assert.match(found, /\.overview\.md:4 {2}codename ZEPHYR-9942/);
  assert.equal(greps.at(-1).case_insensitive, true);

  // A query that is not a valid expression is searched verbatim instead of
  // failing server-side as a broken regex.
  await run({ action: "search_contents", query: "phase one (unfinished" });
  assert.equal(greps.at(-1).pattern, "phase one \\(unfinished");

  const remaining = (await calls.toolDefs
    .get("get_context_remaining")
    .execute("g", {}, undefined, () => {}, ctx)).content[0].text;
  assert.match(remaining, /^window: w2, opened /m);
  assert.match(remaining, /^archives: 1 \(latest archive_001, Working Memory ready\)$/m);
  assert.match(remaining, /^tokens_left: /m);
  assert.match(remaining, /^advice: /m);
});

test("a window restored with OpenViking down still cuts the archived conversation", { skip: !JITI_PATH }, async (t) => {
  withDeadServer(t);
  const calls = await loadExtension();

  const headerText =
    '<openviking-context source="context-window">\n' +
    '<context_window id="w2" previous="w1" archive="archive_001">restored</context_window>\n' +
    "</openviking-context>";
  const branch = [
    { type: "message", id: "e1", message: { role: "user", content: "phase one PAD1", timestamp: 1 } },
    {
      type: "message",
      id: "e2",
      message: {
        role: "assistant",
        content: [{ type: "toolCall", id: "call-1", name: "new_context", arguments: {} }],
        timestamp: 2,
      },
    },
    {
      type: "custom",
      customType: "ov-context-window",
      data: {
        version: 1,
        ovSessionId: "pi-pi-session-abc",
        windowIndex: 2,
        anchorToolCallId: "call-1",
        openedAt: 1000,
        headerText,
        reason: "phase one is done",
        notes: "codename ZEPHYR-9942",
        nextSteps: [],
        pendingRequest: "",
        archiveId: "archive_001",
        archiveUri: ARCHIVE,
        taskId: "task-1",
        overviewReady: true,
        overviewUnavailable: false,
        overviewAttempts: 0,
        previousOverview: "",
        siblingToolNames: [],
        syncedEntryCount: 42,
        lastResetAt: 1000,
        lastResetBy: "agent",
      },
    },
  ];
  const ctx = fakeCtx({ branch });

  const started = await calls.handlers.get("before_agent_start")(
    { type: "before_agent_start", prompt: "what was the codename?", systemPrompt: "BASE" },
    ctx,
  );
  // Window state and tools are restored before the health check, so an
  // unreachable server does not hand the archived window back to the model.
  assert.match(started.message.content, /^\[context-status\] window w2/);
  assert.ok(calls.tools.includes("new_context"));

  const messages = [
    { role: "user", content: "phase one PAD1", timestamp: 1 },
    {
      role: "assistant",
      content: [{ type: "toolCall", id: "call-1", name: "new_context", arguments: {} }],
      timestamp: 2,
    },
    { role: "toolResult", toolCallId: "call-1", toolName: "new_context", content: "ok", timestamp: 3 },
    { role: "user", content: "what was the codename?", timestamp: 4 },
  ];
  const cut = await calls.handlers.get("context")({ type: "context", messages }, ctx);
  assert.equal(cut.messages.length, 1);
  assert.ok(String(cut.messages[0].content).startsWith(headerText));
  // The window header merges into the surviving user message instead of
  // producing two user messages in a row.
  assert.match(String(cut.messages[0].content), /what was the codename\?$/);
  assert.equal(JSON.stringify(cut.messages).includes("PAD1"), false);

  // Offline, pi keeps its own summarizer: no archive, no header as summary.
  const compact = await calls.handlers.get("session_before_compact")(
    { type: "session_before_compact", preparation: { firstKeptEntryId: "e2", tokensBefore: 100 }, branchEntries: branch },
    ctx,
  );
  assert.equal(compact, undefined);

  // A compaction that was not ours already cut the history natively, so the
  // virtual boundary has to be released or the model would lose live context.
  await calls.handlers.get("session_compact")(
    { type: "session_compact", fromExtension: false, compactionEntry: {}, reason: "threshold" },
    ctx,
  );
  const after = await calls.handlers.get("context")({ type: "context", messages }, ctx);
  assert.equal(after.messages.length, messages.length);

  // The final watermark is persisted even though the server never answered.
  await calls.handlers.get("session_shutdown")({ type: "session_shutdown", reason: "quit" }, ctx);
  const windowEntries = calls.entries.filter((e) => e.customType === "ov-context-window");
  assert.ok(windowEntries.length >= 1);
  assert.equal(windowEntries.at(-1).data.windowIndex, 3, "the absorbed compaction opened w3");
});

test("the fork ships no takeover module", () => {
  for (const file of ["takeover.ts", "lib/takeover-core.mjs", "shared/recall-ledger.mjs"]) {
    assert.equal(existsSync(join(EXTENSION_DIR, file)), false, `${file} should not exist`);
  }
});
