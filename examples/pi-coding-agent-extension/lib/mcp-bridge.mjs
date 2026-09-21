/**
 * In-process MCP client for the pi extension.
 *
 * pi has no MCP client, so this extension used to hand-maintain a catalogue of
 * REST-backed `viking_*` tools that drifted away from what the server exposes.
 * The bridge removes the catalogue: it runs the shared stdio -> HTTP proxy core
 * (`shared/mcp-proxy-core.mjs`) *inside pi's process*, asks the server for its
 * own `tools/list`, and republishes those descriptors to pi.
 *
 * The core is never `start()`ed — that wires readline, stdin and signal
 * handlers, which would fight pi for the terminal. Instead the core is driven
 * directly through `handleMessage()` with an injected stdout sink that parses
 * the newline-delimited JSON it writes and settles pending requests by
 * JSON-RPC id; shutdown goes through `closeSession()`.
 *
 * Everything the core already owns stays there: request headers (including the
 * actor-peer and identity headers), credential hot-reload on 401/403, session
 * re-initialization, the concurrency limit and the HTTP timeout. This module
 * builds no headers of its own and knows nothing about pi's APIs, so its logic
 * is unit-testable with `node --test` alone.
 */

import { createOpenVikingMcpProxy } from "../shared/mcp-proxy-core.mjs";

/** The server accepts 2024-11-05 .. 2025-11-25; anything else is an HTTP 400. */
export const MCP_PROTOCOL_VERSION = "2025-06-18";

/** pi's own tool-output limits, mirrored so results look the same as built-ins. */
export const MAX_RESULT_BYTES = 50 * 1024;
export const MAX_RESULT_LINES = 2000;

/** Shared default (`OPENVIKING_TIMEOUT_MS`); only used if the config has none. */
export const DEFAULT_TIMEOUT_MS = 15000;

/** Handshake budget when `connect()` is called without one. */
export const DEFAULT_HANDSHAKE_BUDGET_MS = 5000;

/** The bridge backstop sits above the core's own HTTP timeout, never below it. */
const BACKSTOP_EXTRA_MS = 5000;

const DEFAULT_CLIENT_NAME = "openviking-pi";
const DEFAULT_CLIENT_VERSION = "0.0.0";

const NOOP_LOGGER = { log() {}, logError() {} };

/**
 * Wording copied from the `/mcp` probe in the shared doctor core so both
 * surfaces explain the same failure the same way. The module itself is not
 * imported: it is not part of pi's generated `shared/` import closure, and one
 * diagnostic table is not worth pulling in.
 */
const MCP_STATUS_HINTS = {
  401: "credentials rejected on /mcp",
  403: "key valid but not allowed on /mcp (root key in api_key mode)",
  404: "no MCP endpoint at <url>/mcp — missing path prefix, old server, or a reverse proxy that does not forward /mcp",
};

const READ_TRUNCATION_HINT =
  "Narrow the read with offset/limit, or pass fewer URIs per call — the results for several URIs are concatenated into one block, so the later URIs may be missing entirely.";
const LISTING_TRUNCATION_HINT =
  "Narrow the result with limit/node_limit, or pass a more specific uri.";
const GENERIC_TRUNCATION_HINT =
  "Ask for less: a narrower uri, a smaller limit, or fewer items per call.";

const TRUNCATION_HINTS = {
  read: READ_TRUNCATION_HINT,
  list: LISTING_TRUNCATION_HINT,
  tree: LISTING_TRUNCATION_HINT,
  glob: LISTING_TRUNCATION_HINT,
  grep: LISTING_TRUNCATION_HINT,
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function positiveNumber(value) {
  const next = Number(value);
  return Number.isFinite(next) && next > 0 ? next : 0;
}

function asError(value) {
  return value instanceof Error ? value : new Error(String(value));
}

function unrefTimer(timer) {
  if (timer && typeof timer.unref === "function") timer.unref();
  return timer;
}

function safeStringify(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

// ---------------------------------------------------------------------------
// Schema tidying (pure)
// ---------------------------------------------------------------------------

/**
 * Drop the `title` keyword everywhere, without touching the direct children of
 * a `properties` map — those keys are parameter names, and a parameter may
 * legitimately be called `title`.
 */
function stripTitleKeyword(value) {
  if (Array.isArray(value)) return value.map(stripTitleKeyword);
  if (!isPlainObject(value)) return value;
  const out = {};
  for (const [key, child] of Object.entries(value)) {
    if (key === "title") continue;
    out[key] = key === "properties" ? stripTitlesInNamedSchemas(child) : stripTitleKeyword(child);
  }
  return out;
}

/** `properties`-style maps: keep every key, recurse into the schemas only. */
function stripTitlesInNamedSchemas(value) {
  if (!isPlainObject(value)) return stripTitleKeyword(value);
  const out = {};
  for (const [name, schema] of Object.entries(value)) {
    out[name] = stripTitleKeyword(schema);
  }
  return out;
}

/**
 * An upstream `inputSchema` as pi's `parameters`.
 *
 * Deliberately minimal: the server already emits portable schemas (no `$ref`,
 * no `anyOf`, no type arrays), so the only work here is dropping noise that
 * costs prompt tokens — `$schema` and the root `additionalProperties`, plus
 * every `title` keyword (~1.5 KB across the current tool set). `default`,
 * `enum`, `minimum`/`maximum` and *nested* `additionalProperties` are kept:
 * rewriting them would make pi's local validation disagree with the server's.
 */
export function toPiParameters(inputSchema) {
  const src = isPlainObject(inputSchema) ? inputSchema : {};
  const out = { type: "object", properties: {} };
  for (const [key, value] of Object.entries(src)) {
    if (key === "$schema" || key === "additionalProperties" || key === "title") continue;
    out[key] = key === "properties" ? stripTitlesInNamedSchemas(value) : stripTitleKeyword(value);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Result conversion (pure)
// ---------------------------------------------------------------------------

/** The text of the given content blocks, joined and trimmed. */
export function joinText(blocks) {
  return (Array.isArray(blocks) ? blocks : [])
    .filter((block) => block && block.type === "text")
    .map((block) => String(block.text ?? ""))
    .join("\n")
    .trim();
}

/**
 * An MCP `tools/call` result as pi content blocks.
 *
 * pi's content union is text or image; everything else becomes one readable
 * text line so the model still learns what came back.
 */
export function mcpContentToPi(result) {
  const blocks = Array.isArray(result?.content) ? result.content : [];
  const out = [];
  for (const block of blocks) {
    if (!isPlainObject(block)) continue;
    if (block.type === "text") {
      out.push({ type: "text", text: String(block.text ?? "") });
    } else if (block.type === "image") {
      out.push({
        type: "image",
        data: String(block.data ?? ""),
        mimeType: String(block.mimeType ?? block.mime_type ?? "image/png"),
      });
    } else if (block.type === "audio") {
      out.push({ type: "text", text: `[audio content omitted (${String(block.mimeType ?? "unknown type")})]` });
    } else if (block.type === "resource") {
      const resource = isPlainObject(block.resource) ? block.resource : {};
      const inline = typeof resource.text === "string" ? `\n${resource.text}` : "";
      const mime = resource.mimeType ? ` (${resource.mimeType})` : "";
      out.push({ type: "text", text: `[resource ${String(resource.uri ?? "?")}${mime}]${inline}` });
    } else if (block.type === "resource_link") {
      const label = block.name ? ` — ${block.name}` : "";
      out.push({ type: "text", text: `[resource link ${String(block.uri ?? "?")}${label}]` });
    } else {
      out.push({ type: "text", text: `[unsupported MCP content block: ${String(block.type ?? "?")}]` });
    }
  }

  // structuredContent is appended only when it says something the upstream text
  // does not. FastMCP echoes a {"result": "<the same text>"} beside the text
  // block for every tool annotated `-> str`, so the comparison must be against
  // the UPSTREAM text — never against the notes synthesised above for
  // image/audio/resource blocks, which would make every such echo look new.
  const structured = result?.structuredContent;
  if (isPlainObject(structured)) {
    const text = joinText(blocks);
    const values = Object.values(structured);
    const serialized = safeStringify(structured);
    const duplicate =
      (typeof structured.result === "string" && structured.result.trim() === text)
      || serialized.trim() === text
      || (text.length > 0 && values.length === 1 && typeof values[0] === "string" && values[0].trim() === text);
    if (!duplicate && serialized) out.push({ type: "text", text: serialized });
  }

  if (out.length === 0) out.push({ type: "text", text: "" });
  return out;
}

// ---------------------------------------------------------------------------
// Truncation (pure)
// ---------------------------------------------------------------------------

/**
 * Cut a UTF-8 string at a byte limit without splitting a multi-byte sequence:
 * step back over continuation bytes (10xxxxxx) so the cut lands on a character
 * boundary and no U+FFFD is produced.
 */
function cutToBytes(text, maxBytes) {
  const buf = Buffer.from(text, "utf8");
  if (buf.length <= maxBytes) return text;
  let end = maxBytes;
  while (end > 0 && (buf[end] & 0xc0) === 0x80) end -= 1;
  return buf.subarray(0, end).toString("utf8");
}

/**
 * Head-truncate a tool result to pi's own limits.
 *
 * pi's `truncateHead` is not reused: it only resolves under pi's loader (so
 * `node --test` cannot import it), and it returns an empty string when the
 * first line alone exceeds `maxBytes` — exactly the shape a single-line
 * 60 KB OpenViking read produces.
 */
export function truncateText(text, { maxBytes = MAX_RESULT_BYTES, maxLines = MAX_RESULT_LINES } = {}) {
  const value = typeof text === "string" ? text : String(text ?? "");
  const byteLimit = Math.max(0, Math.floor(Number(maxBytes) || 0));
  const lineLimit = Math.max(0, Math.floor(Number(maxLines) || 0));
  let out = value;
  let truncated = false;

  if (lineLimit > 0) {
    const lines = out.split("\n");
    if (lines.length > lineLimit) {
      out = lines.slice(0, lineLimit).join("\n");
      truncated = true;
    }
  }
  if (Buffer.byteLength(out, "utf8") > byteLimit) {
    out = cutToBytes(out, byteLimit);
    truncated = true;
  }
  return { text: out, truncated };
}

/**
 * The hint appended to a truncated result, naming the knob that actually
 * narrows *this* tool. Accepts the upstream name (`read`) or the registered
 * one (`openviking_read`).
 */
export function truncationHint(toolName, { maxBytes = MAX_RESULT_BYTES, maxLines = MAX_RESULT_LINES } = {}) {
  const bare = String(toolName || "").replace(/^openviking_/, "");
  const advice = TRUNCATION_HINTS[bare] || GENERIC_TRUNCATION_HINT;
  return `[OpenViking] Output truncated at ${maxBytes} bytes / ${maxLines} lines. ${advice}`;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * A JSON-RPC error frame as a throwable Error.
 *
 * `error.data` carries what the operator needs: on 403 the only actionable
 * text (root keys cannot reach /mcp) lives in `data.serverMessage`, not in
 * `error.message`, so both are spliced into the thrown message.
 */
function jsonRpcError(error, what) {
  const base = String(error?.message || `OpenViking MCP ${what} failed`);
  const data = isPlainObject(error?.data) ? error.data : {};
  const status = positiveNumber(data.status);
  const parts = [base];
  if (status && !base.includes(`HTTP ${status}`)) parts.push(`(HTTP ${status})`);
  const serverMessage = typeof data.serverMessage === "string" ? data.serverMessage.trim() : "";
  if (serverMessage) parts.push(`Server said: ${serverMessage}`);
  const hint = status ? MCP_STATUS_HINTS[status] : "";
  if (hint) parts.push(`Hint: ${hint}.`);

  const err = new Error(parts.join(" "));
  err.name = "OpenVikingMcpError";
  if (error?.code !== undefined) err.code = error.code;
  if (error?.data !== undefined) err.data = error.data;
  if (status) err.status = status;
  return err;
}

function abortError(what) {
  const err = new Error(`OpenViking MCP ${what} was cancelled`);
  err.name = "AbortError";
  return err;
}

// ---------------------------------------------------------------------------
// The bridge
// ---------------------------------------------------------------------------

/**
 * Create the in-process MCP client.
 *
 * `readConfig` is re-invoked by the core on credential changes, so it must
 * re-resolve the configuration rather than close over a snapshot.
 */
export function createMcpBridge({
  readConfig,
  loggerFactory = () => NOOP_LOGGER,
  fetchImpl,
  clientInfo,
  timeoutMs,
} = {}) {
  if (typeof readConfig !== "function") {
    throw new Error("createMcpBridge requires a readConfig function");
  }

  const explicitTimeoutMs = positiveNumber(timeoutMs);
  const info = {
    name: String(clientInfo?.name || DEFAULT_CLIENT_NAME),
    version: String(clientInfo?.version || DEFAULT_CLIENT_VERSION),
  };

  const pending = new Map();
  const state = {
    connected: false,
    closed: false,
    error: null,
    /** Upstream `tools/list` descriptors, untouched. */
    tools: [],
    /** Frames with a null id (-32600 / -32700): they match no request. */
    stray: [],
  };

  let configTimeoutMs = 0;
  let lastConfig = null;
  let seq = 0;
  let closed = false;
  let inFlight = null;

  /**
   * The core's own `readConfig`, wrapped so the bridge sees the same resolved
   * config it does — in particular `timeoutMs` (shared key, default 15000,
   * `OPENVIKING_TIMEOUT_MS` overrides it), so pi waits exactly as long as the
   * other harnesses rather than inventing a pi-specific floor.
   */
  function readConfigForCore() {
    const cfg = readConfig();
    lastConfig = cfg;
    const resolved = positiveNumber(cfg?.timeoutMs);
    if (resolved) configTimeoutMs = resolved;
    return cfg;
  }

  function callTimeoutMs() {
    return explicitTimeoutMs || configTimeoutMs || DEFAULT_TIMEOUT_MS;
  }

  /** Settle the pending request a frame belongs to, if it still has one. */
  function settle(message) {
    const id = message?.id;
    if (id === undefined || id === null) {
      // -32600 / -32700 carry a null id and match no request; collecting them
      // keeps a protocol bug visible instead of silently dropping it.
      state.stray.push(message);
      return;
    }
    const key = String(id);
    const entry = pending.get(key);
    // No entry: a second frame for an already-settled request, or one that a
    // timeout/abort already failed locally.
    if (!entry) return;
    pending.delete(key);
    if (message.error) entry.reject(jsonRpcError(message.error, entry.what));
    else entry.resolve(message.result);
  }

  const sink = {
    write(line, callback) {
      try {
        for (const part of String(line).split("\n")) {
          const trimmed = part.trim();
          if (!trimmed) continue;
          let message;
          try {
            message = JSON.parse(trimmed);
          } catch {
            continue;
          }
          settle(message);
        }
      } finally {
        // The core writes through `new Promise((resolve) => stdout.write(line,
        // resolve))` and awaits it, so this callback must fire on every path —
        // skipping it hangs handleMessage() forever.
        if (typeof callback === "function") callback();
      }
      return true;
    },
  };

  const proxy = createOpenVikingMcpProxy({
    stdout: sink,
    readConfig: readConfigForCore,
    loggerFactory,
    fetchImpl,
  });

  let logger = NOOP_LOGGER;
  try {
    logger = loggerFactory("mcp-bridge", lastConfig || {}) || NOOP_LOGGER;
  } catch {
    logger = NOOP_LOGGER;
  }

  function log(stage, data) {
    try {
      logger.log(stage, data);
    } catch { /* debug logging must never break a tool call */ }
  }

  function logError(stage, err) {
    try {
      logger.logError(stage, err);
    } catch { /* debug logging must never break a tool call */ }
  }

  /**
   * One JSON-RPC request through the core.
   *
   * The abort signal fails the call *locally only*: the HTTP request the core
   * already sent runs to completion server-side, so a cancelled or timed-out
   * write/edit/forget/add_resource may still have taken effect. Real
   * cancellation needs AbortSignal plumbing in the shared core and is out of
   * scope here; every other harness behaves the same way today.
   */
  function request(method, params, { timeoutMs: budgetMs, signal, label } = {}) {
    const what = label ? `${method} (${label})` : method;
    if (closed) return Promise.reject(new Error(`OpenViking MCP bridge is closed; ${what} was not sent`));
    const ms = positiveNumber(budgetMs) || callTimeoutMs() + BACKSTOP_EXTRA_MS;
    const id = `ov-pi-${++seq}`;

    return new Promise((resolve, reject) => {
      let timer = null;
      let onAbort = null;

      function cleanup() {
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
        if (onAbort && signal && typeof signal.removeEventListener === "function") {
          try {
            signal.removeEventListener("abort", onAbort);
          } catch { /* best effort */ }
        }
        onAbort = null;
      }

      const entry = {
        what,
        resolve(value) {
          cleanup();
          resolve(value);
        },
        reject(err) {
          cleanup();
          reject(err);
        },
      };

      function fail(err) {
        if (pending.get(id) !== entry) return;
        pending.delete(id);
        entry.reject(err);
      }

      pending.set(id, entry);
      timer = unrefTimer(setTimeout(() => {
        fail(new Error(`OpenViking MCP ${what} timed out after ${ms}ms`));
      }, ms));

      if (signal) {
        if (signal.aborted) {
          fail(abortError(what));
          return;
        }
        onAbort = () => fail(abortError(what));
        if (typeof signal.addEventListener === "function") {
          signal.addEventListener("abort", onAbort, { once: true });
        }
      }

      Promise.resolve(proxy.handleMessage({ jsonrpc: "2.0", id, method, params }))
        .catch((err) => fail(asError(err)));
    });
  }

  /**
   * A notification, bounded by the caller's deadline.
   *
   * The core answers a notification with nothing and rejects nothing, so there
   * is no frame to pair against: without this race a hung server would occupy
   * the core's own timeout (15s by default) and stretch a 5s handshake budget
   * to 25s.
   */
  function notify(method, params, budgetMs) {
    const sent = Promise.resolve(proxy.handleMessage({ jsonrpc: "2.0", method, params }));
    let timer = null;
    const deadline = new Promise((_resolve, reject) => {
      timer = unrefTimer(setTimeout(() => {
        reject(new Error(`OpenViking MCP ${method} timed out after ${budgetMs}ms`));
      }, budgetMs));
    });
    return Promise.race([sent, deadline]).finally(() => {
      if (timer) clearTimeout(timer);
    });
  }

  /** initialize -> notifications/initialized -> tools/list, on ONE deadline. */
  async function handshake(budgetMs) {
    const budget = positiveNumber(budgetMs) || DEFAULT_HANDSHAKE_BUDGET_MS;
    const deadline = Date.now() + budget;
    const remaining = (step) => {
      const left = deadline - Date.now();
      if (left <= 0) {
        throw new Error(`OpenViking MCP handshake budget of ${budget}ms ran out before ${step}`);
      }
      return left;
    };

    await request("initialize", {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: info,
    }, { timeoutMs: remaining("initialize") });

    await notify("notifications/initialized", {}, remaining("notifications/initialized"));

    const listed = await request("tools/list", {}, { timeoutMs: remaining("tools/list") });
    return Array.isArray(listed?.tools) ? listed.tools : [];
  }

  async function runConnect(budgetMs) {
    try {
      const tools = await handshake(budgetMs);
      state.tools = tools;
      state.connected = true;
      state.error = null;
      log("connected", { tools: tools.length, mcpUrl: lastConfig?.mcpUrl });
    } catch (err) {
      // A dead or unauthorized server must not fail pi's startup: recall, sync
      // and takeover keep working, the session just has no OpenViking tools.
      state.tools = [];
      state.connected = false;
      state.error = `OpenViking MCP handshake failed: ${asError(err).message}`;
      logError("connect_failed", err);
    }
    return state;
  }

  /**
   * Handshake once. Never rejects; concurrent callers share one in-flight
   * promise, a successful handshake is never repeated, and a failed one leaves
   * `state.error` set so the next call can retry.
   */
  function connect(budgetMs = DEFAULT_HANDSHAKE_BUDGET_MS) {
    if (inFlight) return inFlight;
    if (closed) {
      state.error = state.error || "OpenViking MCP bridge is closed";
      return Promise.resolve(state);
    }
    const attempt = runConnect(budgetMs).then((result) => {
      if (!result.connected) inFlight = null;
      return result;
    });
    inFlight = attempt;
    return attempt;
  }

  /**
   * Call one upstream tool. Both `isError: true` results and JSON-RPC error
   * frames throw, so pi reports a failed tool call rather than presenting the
   * error text as an answer.
   */
  async function callTool(name, args, { signal } = {}) {
    const tool = String(name || "");
    if (!tool) throw new Error("callTool requires a tool name");

    const result = await request(
      "tools/call",
      { name: tool, arguments: isPlainObject(args) ? args : {} },
      { timeoutMs: callTimeoutMs() + BACKSTOP_EXTRA_MS, signal, label: tool },
    );

    const content = mcpContentToPi(result);
    if (result?.isError) {
      throw new Error(joinText(content) || `OpenViking ${tool} failed`);
    }

    let truncated = false;
    const trimmed = content.map((block) => {
      if (block.type !== "text") return block;
      const cut = truncateText(block.text);
      if (!cut.truncated) return block;
      truncated = true;
      return { type: "text", text: `${cut.text}\n${truncationHint(tool)}` };
    });
    if (truncated) log("truncated", { tool });

    return {
      content: trimmed,
      details: { tool, structuredContent: result?.structuredContent, truncated },
    };
  }

  /** Reject everything still in flight, then release the upstream session. */
  async function close() {
    closed = true;
    state.closed = true;
    inFlight = null;
    for (const [id, entry] of [...pending]) {
      pending.delete(id);
      entry.reject(new Error(`OpenViking MCP bridge closed while ${entry.what} was in flight`));
    }
    await proxy.closeSession();
  }

  return { connect, callTool, close, state, proxy };
}
