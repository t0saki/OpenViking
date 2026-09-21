/**
 * Tests for the in-process MCP client (`lib/mcp-bridge.mjs`).
 *
 * The fake upstream reproduces the framing of a real OpenViking `/mcp`
 * endpoint, because the bridge's whole job is to survive that framing:
 *   - responses are SSE: `event: message\r\ndata: {json}\r\n\r\n`, served as
 *     `content-type: text/event-stream`, and one body may carry several frames;
 *   - a notification is answered with HTTP 202, `application/json` and an
 *     EMPTY body — there is no frame to pair against;
 *   - no `mcp-session-id` response header ever comes back, because the server
 *     mounts FastMCP with `stateless_http=True`.
 * A test that answers with plain JSON, or that hands back a session id, would
 * pass against a bridge that cannot talk to the real server.
 *
 * `fixtures/mcp-tools-list.json` is the real `tools/list` payload of an
 * OpenViking server, captured by running an MCP `initialize` + `tools/list`
 * against a live `openviking-server` with
 * `~/.claude/plans/pi-mcp-tool-parity-spike/spikeA_dump_tools.py`, which dumps
 * the descriptors exactly as they are serialized onto the wire. It is a sample
 * of real schema shapes, not the authoritative tool catalogue: the extension
 * mirrors whatever the server returns at runtime.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  MAX_RESULT_BYTES,
  MAX_RESULT_LINES,
  MCP_PROTOCOL_VERSION,
  createMcpBridge,
  joinText,
  mcpContentToPi,
  toPiParameters,
  truncateText,
  truncationHint,
} from "../lib/mcp-bridge.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(join(HERE, "fixtures", "mcp-tools-list.json"), "utf-8"));
const FIXTURE_TOOLS = FIXTURE.tools;
const FIXTURE_NAMES = FIXTURE_TOOLS.map((tool) => tool.name);

/** pi's own tools, plus `powershell` which 0.86+ adds on every platform. */
const PI_BUILTIN_TOOLS = new Set(["read", "bash", "edit", "write", "grep", "find", "ls", "powershell"]);

// ---------------------------------------------------------------------------
// A fake upstream with production framing
// ---------------------------------------------------------------------------

function sseBody(...frames) {
  return frames.map((frame) => `event: message\r\ndata: ${JSON.stringify(frame)}\r\n\r\n`).join("");
}

function makeResponse({ status = 200, statusText = "OK", contentType = "text/event-stream", body = "" } = {}) {
  // Only content-type is ever set: a stateless server returns no
  // `mcp-session-id`, so every lookup for one must come back null.
  const headers = new Map([["content-type", contentType]]);
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    headers: { get: (name) => headers.get(String(name).toLowerCase()) ?? null },
    text: async () => body,
  };
}

const sseResponse = (...frames) => makeResponse({ body: sseBody(...frames) });

/** What a notification really gets back: 202, JSON content type, no body. */
const acceptedResponse = () =>
  makeResponse({ status: 202, statusText: "Accepted", contentType: "application/json", body: "" });

const jsonErrorResponse = (status, statusText, payload) =>
  makeResponse({ status, statusText, contentType: "application/json", body: JSON.stringify(payload) });

const initializeResult = (id) => ({
  jsonrpc: "2.0",
  id,
  result: {
    protocolVersion: MCP_PROTOCOL_VERSION,
    capabilities: { tools: { listChanged: false } },
    serverInfo: { name: "openviking", version: "1.5.0" },
  },
});

const textResult = (id, text) => ({ jsonrpc: "2.0", id, result: { content: [{ type: "text", text }] } });

function makeServer({ tools = FIXTURE_TOOLS, onInitialize, onNotify, onList, onCall } = {}) {
  const requests = [];
  async function fetchImpl(url, init = {}) {
    const message = init.body ? JSON.parse(init.body) : null;
    requests.push({ url, method: init.method, message, headers: init.headers || {} });
    if (init.method === "DELETE") return makeResponse({ contentType: "application/json", body: "" });
    switch (message?.method) {
      case "initialize":
        return onInitialize ? onInitialize(message, init) : sseResponse(initializeResult(message.id));
      case "notifications/initialized":
        return onNotify ? onNotify(message, init) : acceptedResponse();
      case "tools/list":
        return onList ? onList(message, init) : sseResponse({ jsonrpc: "2.0", id: message.id, result: { tools } });
      case "tools/call":
        return onCall
          ? onCall(message, init)
          : sseResponse(textResult(message.id, `${message.params?.name}-ok`));
      default:
        return sseResponse({
          jsonrpc: "2.0",
          id: message?.id ?? null,
          error: { code: -32601, message: `no method ${message?.method}` },
        });
    }
  }
  return {
    fetchImpl,
    requests,
    count: (method) => requests.filter((entry) => entry.message?.method === method).length,
    rpcMethods: () => requests.map((entry) => entry.message?.method),
  };
}

/**
 * Requests the server never answers until the test releases them. Releasing is
 * what lets the core clear its own abort timer, so the suite does not sit out
 * the configured HTTP timeout after the assertions are done.
 */
function makeHang() {
  const waiting = [];
  return {
    handler: (_message, init) =>
      new Promise((resolve, reject) => {
        waiting.push(resolve);
        init?.signal?.addEventListener?.(
          "abort",
          () => {
            const err = new Error("The operation was aborted");
            err.name = "AbortError";
            reject(err);
          },
          { once: true },
        );
      }),
    release: (response = acceptedResponse()) => {
      while (waiting.length) waiting.shift()(response);
    },
  };
}

function makeConfig(overrides = {}) {
  return () => ({
    mcpUrl: "http://127.0.0.1:65535/mcp",
    apiKey: "test-key",
    account: "default",
    user: "tester",
    sendIdentityHeaders: true,
    peerId: "peer-test",
    userAgent: "openviking-pi/0.4.0",
    timeoutMs: 2000,
    credentialSource: "test",
    credentialPath: "",
    watchedPaths: [],
    extraHeaders: null,
    ...overrides,
  });
}

function makeBridge(fetchImpl, { config, ...options } = {}) {
  return createMcpBridge({
    readConfig: makeConfig(config),
    fetchImpl,
    clientInfo: { name: "openviking-pi", version: "0.4.0" },
    ...options,
  });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Resolves to `"pending"` unless the promise settles within `ms`. */
async function outcomeWithin(promise, ms) {
  return Promise.race([
    promise.then(() => "resolved", () => "rejected"),
    sleep(ms).then(() => "pending"),
  ]);
}

// ---------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------

test("connect handshakes in three steps and mirrors tools/list verbatim", async () => {
  const server = makeServer();
  const bridge = makeBridge(server.fetchImpl);

  const state = await bridge.connect(2000);

  assert.equal(state.connected, true);
  assert.equal(state.error, null);
  assert.deepEqual(server.rpcMethods(), ["initialize", "notifications/initialized", "tools/list"]);

  const handshake = server.requests[0].message;
  assert.equal(handshake.params.protocolVersion, MCP_PROTOCOL_VERSION);
  assert.deepEqual(handshake.params.clientInfo, { name: "openviking-pi", version: "0.4.0" });

  // The descriptors reach pi untouched: the extension ships no catalogue of
  // its own, so whatever the server says is what the model sees.
  assert.deepEqual(state.tools, FIXTURE_TOOLS);
  assert.deepEqual(state.tools.map((tool) => tool.name), FIXTURE_NAMES);
  assert.deepEqual(state.stray, []);

  // Stateless upstream: no session id comes back, so none may be sent.
  for (const entry of server.requests) {
    const names = Object.keys(entry.headers).map((key) => key.toLowerCase());
    assert.equal(names.includes("mcp-session-id"), false);
  }

  await bridge.close();
});

test("mirrored tool names never collide with pi's built-in tools", async () => {
  const server = makeServer();
  const bridge = makeBridge(server.fetchImpl);
  const state = await bridge.connect(2000);

  const registered = state.tools.map((tool) => `openviking_${tool.name}`);
  assert.equal(new Set(registered).size, registered.length, "duplicate tool names in tools/list");
  assert.deepEqual(registered.filter((name) => PI_BUILTIN_TOOLS.has(name)), []);

  // The prefix is load-bearing, not decoration: the bare upstream names really
  // do shadow pi's own tools.
  const shadowed = state.tools.map((tool) => tool.name).filter((name) => PI_BUILTIN_TOOLS.has(name));
  assert.ok(shadowed.length >= 3, `expected the bare names to clash with pi built-ins, got ${shadowed}`);

  await bridge.close();
});

test("the handshake budget is shared across all three steps", async () => {
  // initialize deliberately burns most of the budget, then the server never
  // answers the notification. Two independent things have to hold: the
  // notification is bounded by what is LEFT rather than by a fresh full budget,
  // and it does not fall back to the core's own HTTP timeout (2000ms here, 15s
  // in production). Without the first, `connect(5000)` on a slow server becomes
  // a 15-45s block on pi's `before_agent_start` with nothing reporting failure.
  const hang = makeHang();
  const server = makeServer({
    onInitialize: async (message) => {
      await sleep(250);
      return sseResponse(initializeResult(message.id));
    },
    onNotify: hang.handler,
  });
  const bridge = makeBridge(server.fetchImpl, { config: { timeoutMs: 2000 } });

  const started = Date.now();
  const state = await bridge.connect(400);
  const elapsed = Date.now() - started;

  assert.equal(state.connected, false);
  assert.deepEqual(state.tools, []);
  // A 400ms budget with ~250ms already spent leaves the notification ~150ms.
  // Per-step deadlines would report ~400 here instead.
  const reported = state.error.match(/notifications\/initialized timed out after (\d+)ms/);
  assert.ok(reported, `unexpected handshake error: ${state.error}`);
  assert.ok(
    Number(reported[1]) <= 160,
    `the notification got ${reported[1]}ms of a 400ms budget that initialize had already spent ~250ms of: the deadline is not shared`,
  );
  assert.ok(
    elapsed < 600,
    `connect took ${elapsed}ms: the handshake overran its 400ms budget — per-step deadlines, or a fallback to the core's 2000ms timeout`,
  );
  assert.equal(server.count("tools/list"), 0, "tools/list was attempted after the budget ran out");

  hang.release();
  await bridge.close();
});

test("a hung tools/list is bounded by what is left of the handshake budget", async () => {
  // The last step has the least budget left and nothing else bounding it: the
  // core would sit on its own 2000ms HTTP timeout, and a per-step deadline
  // would hand tools/list a fresh full budget. Either way `connect(400)` must
  // still settle inside the 400ms the caller allowed.
  const hang = makeHang();
  const server = makeServer({
    onInitialize: async (message) => {
      await sleep(150);
      return sseResponse(initializeResult(message.id));
    },
    onList: hang.handler,
  });
  const bridge = makeBridge(server.fetchImpl, { config: { timeoutMs: 2000 } });

  const started = Date.now();
  const state = await bridge.connect(400);
  const elapsed = Date.now() - started;

  assert.equal(state.connected, false);
  assert.deepEqual(state.tools, []);
  assert.equal(server.count("tools/list"), 1, "the handshake never reached tools/list");

  const reported = state.error.match(/tools\/list timed out after (\d+)ms/);
  assert.ok(reported, `unexpected handshake error: ${state.error}`);
  assert.ok(
    Number(reported[1]) <= 300,
    `tools/list got ${reported[1]}ms of a 400ms budget that the earlier steps had already spent ~150ms of: the deadline is not shared`,
  );
  assert.ok(
    elapsed < 600,
    `connect took ${elapsed}ms: a hung tools/list is not bounded by the handshake budget`,
  );

  hang.release();
  await bridge.close();
});

test("concurrent connect calls share one handshake, and connect never rejects", async () => {
  const server = makeServer();
  const bridge = makeBridge(server.fetchImpl);

  const [first, second, third] = await Promise.all([
    bridge.connect(2000),
    bridge.connect(2000),
    bridge.connect(2000),
  ]);

  assert.equal(server.count("initialize"), 1);
  assert.equal(server.count("tools/list"), 1);
  assert.equal(first.connected, true);
  assert.equal(first, second);
  assert.equal(second, third);
  await bridge.close();

  // A dead server must not reject either: pi's startup keeps going without
  // OpenViking tools instead of failing the session.
  const dead = async () => {
    throw Object.assign(new TypeError("fetch failed"), { cause: new Error("ECONNREFUSED") });
  };
  const offline = makeBridge(dead);
  await assert.doesNotReject(() => Promise.all([offline.connect(500), offline.connect(500)]));
  assert.equal(offline.state.connected, false);
  assert.match(offline.state.error, /handshake failed/);
  await offline.close();
});

test("a connect after a failed one lists every tool exactly once", async () => {
  let up = false;
  const live = makeServer();
  const fetchImpl = async (url, init) => {
    if (!up) throw Object.assign(new TypeError("fetch failed"), { cause: new Error("ECONNREFUSED") });
    return live.fetchImpl(url, init);
  };
  const bridge = makeBridge(fetchImpl);

  const failed = await bridge.connect(500);
  assert.equal(failed.connected, false);
  assert.deepEqual(failed.tools, []);

  up = true;
  const state = await bridge.connect(2000);
  assert.equal(state.connected, true);
  const names = state.tools.map((tool) => tool.name);
  assert.equal(new Set(names).size, names.length, "a retried handshake duplicated tools");
  assert.deepEqual([...names].sort(), [...FIXTURE_NAMES].sort());

  // Once connected, later calls reuse the handshake instead of re-listing —
  // registering the same tool twice would be a pi-side error.
  await bridge.connect(2000);
  assert.equal(live.count("tools/list"), 1);
  await bridge.close();
});

// ---------------------------------------------------------------------------
// Request/response pairing
// ---------------------------------------------------------------------------

test("concurrent calls pair to the right responses", async () => {
  const server = makeServer({
    onCall: async (message) => {
      // Answer out of order: the slow tool was asked for first.
      const name = message.params.name;
      await sleep(name === "search" ? 60 : 5);
      return sseResponse(textResult(message.id, `${name}:${JSON.stringify(message.params.arguments)}`));
    },
  });
  const bridge = makeBridge(server.fetchImpl);
  await bridge.connect(2000);

  const [search, read, health] = await Promise.all([
    bridge.callTool("search", { query: "slow" }),
    bridge.callTool("read", { uris: ["viking://memory/a"] }),
    bridge.callTool("health", {}),
  ]);

  assert.equal(search.content[0].text, 'search:{"query":"slow"}');
  assert.equal(read.content[0].text, 'read:{"uris":["viking://memory/a"]}');
  assert.equal(health.content[0].text, "health:{}");
  assert.equal(search.details.tool, "search");
  await bridge.close();
});

test("one HTTP body carrying several SSE frames settles the request once", async () => {
  const server = makeServer({
    onCall: (message) =>
      sseResponse(
        // A log notification the server interleaves before the answer.
        { jsonrpc: "2.0", method: "notifications/message", params: { level: "info", data: "working" } },
        textResult(message.id, "first"),
        // A duplicate answer for the same id: the second frame must be dropped
        // rather than resolving anything a second time.
        textResult(message.id, "second"),
      ),
  });
  const bridge = makeBridge(server.fetchImpl);
  await bridge.connect(2000);

  const result = await bridge.callTool("health", {});
  assert.deepEqual(result.content, [{ type: "text", text: "first" }]);

  // The pairing table survived the extra frames.
  const next = await bridge.callTool("health", {});
  assert.deepEqual(next.content, [{ type: "text", text: "first" }]);
  await bridge.close();
});

test("an error frame with a null id settles nothing and is kept as stray", async () => {
  const server = makeServer({
    onCall: () =>
      sseResponse({ jsonrpc: "2.0", id: null, error: { code: -32600, message: "Invalid JSON-RPC message" } }),
  });
  const bridge = makeBridge(server.fetchImpl);
  await bridge.connect(2000);

  const inflight = bridge.callTool("health", {});
  assert.equal(await outcomeWithin(inflight, 60), "pending");
  assert.equal(bridge.state.stray.length, 1);
  assert.equal(bridge.state.stray[0].error.code, -32600);

  await bridge.close();
  await assert.rejects(inflight, /closed while tools\/call \(health\) was in flight/);
});

test("close rejects everything still in flight", async () => {
  const hang = makeHang();
  const server = makeServer({ onCall: hang.handler });
  const bridge = makeBridge(server.fetchImpl, { config: { timeoutMs: 2000 } });
  await bridge.connect(2000);

  const first = bridge.callTool("search", { query: "never answered" });
  const second = bridge.callTool("health", {});
  assert.equal(await outcomeWithin(first, 40), "pending");

  await bridge.close();
  await assert.rejects(first, /closed while tools\/call \(search\) was in flight/);
  await assert.rejects(second, /closed while tools\/call \(health\) was in flight/);
  assert.equal(bridge.state.closed, true);

  hang.release(sseResponse(textResult("ov-pi-2", "late")));
  await assert.rejects(bridge.callTool("health", {}), /bridge is closed/);
});

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

test("an isError result becomes a thrown Error carrying the upstream text", async () => {
  const server = makeServer({
    onCall: (message) =>
      sseResponse({
        jsonrpc: "2.0",
        id: message.id,
        result: {
          content: [{ type: "text", text: "Error calling tool 'forget': no such uri viking://nope" }],
          isError: true,
        },
      }),
  });
  const bridge = makeBridge(server.fetchImpl);
  await bridge.connect(2000);

  await assert.rejects(
    () => bridge.callTool("forget", { uri: "viking://nope" }),
    (err) => {
      assert.ok(err instanceof Error);
      assert.match(err.message, /no such uri viking:\/\/nope/);
      return true;
    },
  );
  await bridge.close();
});

for (const [status, statusText, detail] of [
  [401, "Unauthorized", "invalid api key"],
  [403, "Forbidden", "root api key cannot access /mcp; use a user key"],
]) {
  test(`an HTTP ${status} carries the server's own message into the thrown error`, async () => {
    // Only the handshake succeeds; auth fails on the call, the way an expired
    // or wrong-scope key behaves against a live server.
    const server = makeServer({ onCall: () => jsonErrorResponse(status, statusText, { detail }) });
    const bridge = makeBridge(server.fetchImpl);
    const state = await bridge.connect(2000);
    assert.equal(state.connected, true);

    await assert.rejects(
      () => bridge.callTool("health", {}),
      (err) => {
        assert.equal(err.name, "OpenVikingMcpError");
        assert.equal(err.status, status);
        assert.equal(err.code, -32001);
        assert.match(err.message, new RegExp(`HTTP ${status}`));
        // The actionable half of a 403 lives only in error.data.serverMessage.
        assert.match(err.message, new RegExp(`Server said: ${detail.replace(/[/]/g, "\\/")}`));
        assert.equal(err.data.status, status);
        assert.equal(err.data.serverMessage, detail);
        return true;
      },
    );
    await bridge.close();
  });
}

// ---------------------------------------------------------------------------
// Result conversion
// ---------------------------------------------------------------------------

test("mcpContentToPi maps every MCP block kind pi can show", () => {
  const blocks = mcpContentToPi({
    content: [
      { type: "text", text: "# note\nbody" },
      { type: "image", data: "aGk=", mimeType: "image/png" },
      { type: "audio", data: "QQ==", mimeType: "audio/wav" },
      { type: "resource", resource: { uri: "viking://resources/a.md", mimeType: "text/markdown", text: "inline" } },
      { type: "resource_link", uri: "viking://memory/x", name: "x" },
      { type: "video", data: "AA==" },
    ],
  });

  assert.equal(blocks.length, 6);
  assert.deepEqual(blocks[0], { type: "text", text: "# note\nbody" });
  assert.deepEqual(blocks[1], { type: "image", data: "aGk=", mimeType: "image/png" });
  assert.deepEqual(blocks[2], { type: "text", text: "[audio content omitted (audio/wav)]" });
  assert.deepEqual(blocks[3], {
    type: "text",
    text: "[resource viking://resources/a.md (text/markdown)]\ninline",
  });
  assert.deepEqual(blocks[4], { type: "text", text: "[resource link viking://memory/x — x]" });
  assert.deepEqual(blocks[5], { type: "text", text: "[unsupported MCP content block: video]" });
});

test("an empty result still produces one block", () => {
  assert.deepEqual(mcpContentToPi({}), [{ type: "text", text: "" }]);
  assert.deepEqual(mcpContentToPi({ content: [] }), [{ type: "text", text: "" }]);
  assert.equal(joinText(mcpContentToPi({ content: [] })), "");
});

test("structuredContent is dropped when it repeats the upstream text and kept when it does not", () => {
  // FastMCP echoes {"result": "<the same text>"} for every `-> str` tool.
  const echoed = mcpContentToPi({
    content: [{ type: "text", text: "3 entries" }],
    structuredContent: { result: "3 entries" },
  });
  assert.deepEqual(echoed, [{ type: "text", text: "3 entries" }]);

  const extra = mcpContentToPi({
    content: [{ type: "text", text: "3 entries" }],
    structuredContent: { count: 3, uris: ["a", "b", "c"] },
  });
  assert.equal(extra.length, 2);
  assert.deepEqual(extra[1], { type: "text", text: '{"count":3,"uris":["a","b","c"]}' });

  // The comparison is against the UPSTREAM text, never the notes synthesised
  // for image/audio/resource blocks — otherwise an image-only result would
  // swallow its structured payload.
  const imageOnly = mcpContentToPi({
    content: [{ type: "image", data: "aGk=", mimeType: "image/png" }],
    structuredContent: { result: "chart.png" },
  });
  assert.equal(imageOnly.length, 2);
  assert.deepEqual(imageOnly[1], { type: "text", text: '{"result":"chart.png"}' });
});

test("a tool call converts content and reports structuredContent in details", async () => {
  const server = makeServer({
    onCall: (message) =>
      sseResponse({
        jsonrpc: "2.0",
        id: message.id,
        result: {
          content: [
            { type: "text", text: "read-ok" },
            { type: "image", data: "aGk=", mimeType: "image/png" },
          ],
          structuredContent: { result: "read-ok" },
          isError: false,
        },
      }),
  });
  const bridge = makeBridge(server.fetchImpl);
  await bridge.connect(2000);

  const result = await bridge.callTool("read", { uris: ["viking://memory/a"] });
  assert.deepEqual(result.content, [
    { type: "text", text: "read-ok" },
    { type: "image", data: "aGk=", mimeType: "image/png" },
  ]);
  assert.deepEqual(result.details, {
    tool: "read",
    structuredContent: { result: "read-ok" },
    truncated: false,
  });
  await bridge.close();
});

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

test("truncateText cuts at the line limit", () => {
  const text = Array.from({ length: MAX_RESULT_LINES + 500 }, (_, i) => `line ${i}`).join("\n");
  const cut = truncateText(text);
  assert.equal(cut.truncated, true);
  assert.equal(cut.text.split("\n").length, MAX_RESULT_LINES);
  assert.equal(cut.text.split("\n")[0], "line 0");
  assert.equal(truncateText("a\nb\nc").truncated, false);
});

test("truncateText cuts at the byte limit", () => {
  const text = `${"x".repeat(MAX_RESULT_BYTES + 4096)}\ntail`;
  const cut = truncateText(text);
  assert.equal(cut.truncated, true);
  assert.equal(Buffer.byteLength(cut.text, "utf8"), MAX_RESULT_BYTES);
});

test("a single 60 KB line is cut on a character boundary, not emptied", () => {
  // pi's own truncateHead returns "" when the first line alone is over budget;
  // a one-line 60 KB OpenViking read is exactly that shape. Three-byte
  // characters make the limit land mid-character (51200 % 3 === 2).
  const line = "世".repeat(20 * 1024);
  assert.ok(Buffer.byteLength(line, "utf8") >= 60 * 1024);

  const cut = truncateText(line);
  assert.equal(cut.truncated, true);
  assert.ok(cut.text.length > 0, "a single over-long line must not truncate to nothing");
  assert.equal(cut.text.includes("�"), false, "the byte cut split a multi-byte character");
  const bytes = Buffer.byteLength(cut.text, "utf8");
  assert.ok(bytes <= MAX_RESULT_BYTES, `kept ${bytes} bytes`);
  assert.ok(bytes > MAX_RESULT_BYTES - 3, `cut back too far: ${bytes} bytes`);
  assert.equal(cut.text, "世".repeat(cut.text.length));
});

test("the truncation hint names the knob that narrows that tool", () => {
  assert.match(truncationHint("read"), /offset\/limit/);
  // Registered and upstream names are both accepted.
  assert.equal(truncationHint("openviking_read"), truncationHint("read"));
  for (const name of ["list", "tree", "glob", "grep"]) {
    assert.match(truncationHint(name), /limit\/node_limit/);
  }
  assert.match(truncationHint("remember"), /Ask for less/);
  assert.match(truncationHint("read"), new RegExp(`${MAX_RESULT_BYTES} bytes / ${MAX_RESULT_LINES} lines`));
});

test("a truncated tool result carries the hint and is flagged in details", async () => {
  const text = Array.from({ length: MAX_RESULT_LINES + 200 }, (_, i) => `line ${i}`).join("\n");
  const server = makeServer({
    onCall: (message) =>
      sseResponse(textResult(message.id, message.params.name === "read" ? text : "ok")),
  });
  const bridge = makeBridge(server.fetchImpl);
  await bridge.connect(2000);

  const result = await bridge.callTool("read", { uris: ["viking://memory/big"] });
  assert.equal(result.details.truncated, true);
  const lines = result.content[0].text.split("\n");
  assert.equal(lines.length, MAX_RESULT_LINES + 1);
  assert.match(lines.at(-1), /^\[OpenViking\] Output truncated/);
  assert.match(lines.at(-1), /offset\/limit/);

  const small = await bridge.callTool("health", {});
  assert.equal(small.details.truncated, false);
  assert.equal(small.content[0].text.includes("Output truncated"), false);
  await bridge.close();
});

// ---------------------------------------------------------------------------
// Schema tidying
// ---------------------------------------------------------------------------

test("toPiParameters drops the title keyword but keeps a property named title", () => {
  const schema = {
    $schema: "https://json-schema.org/draft/2020-12/schema",
    title: "writeArguments",
    type: "object",
    additionalProperties: false,
    properties: {
      title: { type: "string", title: "Title", description: "The note's own title field" },
      body: { type: "string", title: "Body", default: "" },
      tags: {
        type: "array",
        title: "Tags",
        items: {
          type: "object",
          title: "Tag",
          additionalProperties: true,
          properties: { title: { type: "string", title: "Title" } },
        },
      },
    },
    required: ["title"],
  };

  const out = toPiParameters(schema);

  assert.equal("$schema" in out, false);
  assert.equal("additionalProperties" in out, false);
  assert.equal("title" in out, false);
  assert.deepEqual(Object.keys(out.properties), ["title", "body", "tags"]);
  // The parameter survives with its own keys; only the keyword is gone.
  assert.deepEqual(out.properties.title, { type: "string", description: "The note's own title field" });
  assert.equal(out.properties.body.default, "");
  assert.deepEqual(out.properties.tags.items.properties.title, { type: "string" });
  // Nested additionalProperties stays: rewriting it would make pi's local
  // validation disagree with the server's.
  assert.equal(out.properties.tags.items.additionalProperties, true);
  assert.deepEqual(out.required, ["title"]);
});

test("toPiParameters keeps the real schemas intact while shedding title noise", () => {
  let before = 0;
  let after = 0;
  for (const tool of FIXTURE_TOOLS) {
    const out = toPiParameters(tool.inputSchema);
    before += JSON.stringify(tool.inputSchema).length;
    after += JSON.stringify(out).length;

    assert.equal(out.type, "object");
    assert.equal(JSON.stringify(out).includes('"title"'), false, `title survived in ${tool.name}`);
    assert.deepEqual(
      Object.keys(out.properties),
      Object.keys(tool.inputSchema.properties || {}),
      `parameters changed for ${tool.name}`,
    );
    if (tool.inputSchema.required) assert.deepEqual(out.required, tool.inputSchema.required);
  }
  assert.ok(before - after > 1500, `title stripping saved only ${before - after} bytes`);

  // Defaults, enums and nested item types are what pi validates against.
  const read = toPiParameters(FIXTURE_TOOLS.find((tool) => tool.name === "read").inputSchema);
  assert.deepEqual(read.properties.uris, { items: { type: "string" }, type: "array" });
  assert.equal(read.properties.limit.default, -1);
  const addResource = toPiParameters(FIXTURE_TOOLS.find((tool) => tool.name === "add_resource").inputSchema);
  assert.deepEqual(addResource.properties.processing_mode.enum, ["semantic_and_vectors", "vectors_only"]);
});
