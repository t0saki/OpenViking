/**
 * Tests for the registration side of the tool surface (`tools.ts`).
 *
 * `registerMcpTools` is driven here with a fake `pi` that records every
 * `registerTool` call and a fake bridge whose `state.tools` is the captured
 * `tools/list` in `fixtures/mcp-tools-list.json`. Nothing talks to a server:
 * what is under test is the shape of the registrations — the names, the
 * schemas pi validates against, the argument repair bound to each tool, and
 * the call that reaches the bridge.
 *
 * The fixture is a sample of real descriptors, not the authoritative
 * catalogue: the extension mirrors whatever the server lists at runtime, so
 * these tests assert relationships to the fixture ("one registration per
 * descriptor", "the prefix is added") and never a hard-coded tool list.
 *
 * `tools.ts` imports nothing by value from pi's own package, which is what
 * lets `node --test` import it directly (Node strips the types) — the same
 * trick `sync-barrier.test.mjs` and `config.test.mjs` use for `sync.ts` and
 * `config.ts`.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { PI_BUILTIN_TOOL_NAMES, TOOL_NAME_PREFIX, registerMcpTools } from "../tools.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(join(HERE, "fixtures", "mcp-tools-list.json"), "utf-8"));
const FIXTURE_TOOLS = FIXTURE.tools;
const FIXTURE_NAMES = FIXTURE_TOOLS.map((tool) => tool.name);

/** pi's built-ins, spelled out here so a change in `tools.ts` has to face them. */
const PI_BUILTINS = ["read", "bash", "edit", "write", "grep", "find", "ls", "powershell"];

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/** A host that records registrations; one object per test, as pi is per session. */
function fakePi() {
  const registered = [];
  return {
    registered,
    byName: (name) => registered.find((definition) => definition.name === name),
    registerTool(definition) {
      registered.push(definition);
    },
  };
}

/**
 * A bridge whose handshake already happened. `callTool` records its arguments
 * and answers with the shape the real one returns.
 */
function fakeBridge(tools = FIXTURE_TOOLS, { result } = {}) {
  const calls = [];
  return {
    calls,
    state: { connected: true, closed: false, error: null, tools, stray: [] },
    async callTool(name, args, options) {
      calls.push({ name, args, options });
      return result ?? {
        content: [{ type: "text", text: `result of ${name}` }],
        details: { tool: name, structuredContent: undefined, truncated: false },
      };
    },
  };
}

function toolSchema(name) {
  const tool = FIXTURE_TOOLS.find((candidate) => candidate.name === name);
  assert.ok(tool, `fixture is missing the ${name} tool`);
  return tool.inputSchema;
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

test("every server tool becomes exactly one openviking_-prefixed registration", () => {
  const pi = fakePi();
  const added = registerMcpTools(pi, fakeBridge());

  const expected = FIXTURE_NAMES.map((name) => `${TOOL_NAME_PREFIX}${name}`);
  assert.deepEqual(pi.registered.map((definition) => definition.name), expected);
  assert.deepEqual(added, expected);
  assert.equal(new Set(added).size, added.length, "no name is registered twice");
  assert.equal(TOOL_NAME_PREFIX, "openviking_", "the server's usage attribution matches this prefix");

  // The label is generated from the upstream name, so a tool the server adds
  // tomorrow gets one too.
  assert.equal(pi.byName("openviking_search").label, "OpenViking search");
  // The description is the server's own text, trimmed and otherwise untouched:
  // several of these are Python docstrings and arrive with trailing indentation.
  for (const tool of FIXTURE_TOOLS) {
    assert.equal(pi.byName(`${TOOL_NAME_PREFIX}${tool.name}`).description, tool.description.trim());
  }
});

test("no registered name collides with one of pi's built-in tools", () => {
  const pi = fakePi();
  registerMcpTools(pi, fakeBridge());

  assert.deepEqual([...PI_BUILTIN_TOOL_NAMES].sort(), [...PI_BUILTINS].sort());
  for (const definition of pi.registered) {
    assert.ok(
      !PI_BUILTINS.includes(definition.name),
      `${definition.name} would shadow pi's built-in tool of the same name`,
    );
  }
  // `read`, `write`, `edit`, `grep` and `find` are all upstream tool names, so
  // the prefix is what keeps them apart from pi's own.
  for (const builtin of PI_BUILTINS) {
    if (FIXTURE_NAMES.includes(builtin)) {
      assert.ok(pi.byName(`${TOOL_NAME_PREFIX}${builtin}`), `openviking_${builtin} is registered instead`);
    }
  }
});

test("a name that would shadow a built-in is skipped and logged", () => {
  // The guard cannot fire with today's prefix; drive it through the override so
  // a future prefix change finds the behaviour pinned rather than theoretical.
  const pi = fakePi();
  const lines = [];
  const added = registerMcpTools(pi, fakeBridge(), {
    builtinNames: ["openviking_read", ...PI_BUILTINS],
    log: (message) => lines.push(message),
  });

  assert.equal(pi.byName("openviking_read"), undefined);
  assert.ok(!added.includes("openviking_read"));
  assert.equal(added.length, FIXTURE_TOOLS.length - 1, "only the colliding tool is dropped");
  assert.equal(lines.length, 1);
  assert.match(lines[0], /^skipped openviking_read: /);
});

test("registering twice on the same host adds nothing the second time", () => {
  // pi has no `unregisterTool`: a second registration would leave the session
  // with two tools of the same name, so the guard has to hold across calls —
  // the startup chain and the `before_agent_start` retry branch both call this.
  const pi = fakePi();
  const bridge = fakeBridge();

  const first = registerMcpTools(pi, bridge);
  assert.equal(first.length, FIXTURE_TOOLS.length);

  const second = registerMcpTools(pi, bridge);
  assert.deepEqual(second, [], "a repeat call reports nothing new");
  assert.equal(pi.registered.length, FIXTURE_TOOLS.length, "and registers nothing new");

  // A different host is a different session and starts empty.
  const other = fakePi();
  assert.equal(registerMcpTools(other, bridge).length, FIXTURE_TOOLS.length);
});

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

test("parameters are plain JSON Schema objects, not TypeBox", () => {
  const pi = fakePi();
  registerMcpTools(pi, fakeBridge());

  for (const definition of pi.registered) {
    const parameters = definition.parameters;
    assert.equal(Object.getPrototypeOf(parameters), Object.prototype, "a plain object");
    // TypeBox tags every schema it builds with symbol keys (`Symbol(TypeBox.Kind)`).
    // Plain objects get pi's wider type coercion, which is what keeps its local
    // validation from being stricter than the server's.
    assert.deepEqual(Object.getOwnPropertySymbols(parameters), []);
    assert.equal(parameters.type, "object");
    assert.equal(typeof parameters.properties, "object");
    assert.equal(parameters.$schema, undefined, "no root $schema");
    assert.equal(parameters.additionalProperties, undefined, "no root additionalProperties");
    assert.equal(parameters.title, undefined, "no root title keyword");
    // The schema still round-trips as JSON, which is how it reaches a provider.
    assert.doesNotThrow(() => JSON.stringify(parameters));
  }

  // Nothing that constrains a value is dropped: `required`, `default` and
  // `items` all survive, or pi would validate differently from the server.
  const read = pi.byName("openviking_read").parameters;
  assert.deepEqual(read.required, ["uris"]);
  assert.deepEqual(read.properties.uris, { items: { type: "string" }, type: "array" });
  assert.equal(read.properties.limit.default, -1);
});

test("a parameter named title survives while the title keyword is dropped", () => {
  // No tool in the fixture takes a parameter called `title`, so this descriptor
  // is synthetic: it is the one case where the difference between a schema
  // keyword and a `properties` key changes the tool's meaning.
  const pi = fakePi();
  registerMcpTools(pi, fakeBridge([{
    name: "titled",
    description: "A tool whose argument is called title.",
    inputSchema: {
      $schema: "https://json-schema.org/draft/2020-12/schema",
      additionalProperties: false,
      title: "titledArguments",
      type: "object",
      properties: {
        title: { type: "string", title: "Title" },
        nested: {
          type: "object",
          title: "Nested",
          additionalProperties: true,
          properties: { title: { type: "string", title: "Title" } },
        },
      },
      required: ["title"],
    },
  }]));

  const parameters = pi.byName("openviking_titled").parameters;
  assert.equal(parameters.$schema, undefined);
  assert.equal(parameters.additionalProperties, undefined);
  assert.equal(parameters.title, undefined, "the root title keyword is gone");

  const title = parameters.properties.title;
  assert.ok(title, "the parameter literally named title is still there");
  assert.equal(title.type, "string");
  assert.equal(title.title, undefined, "its own title keyword is gone");

  const nested = parameters.properties.nested;
  assert.equal(nested.title, undefined);
  assert.ok(nested.properties.title, "a nested parameter named title survives too");
  assert.equal(nested.properties.title.title, undefined);
  // Only the ROOT additionalProperties is noise; a nested one is a constraint.
  assert.equal(nested.additionalProperties, true);
  assert.deepEqual(parameters.required, ["title"]);
});

test("no registration carries a promptSnippet or promptGuidelines", () => {
  // Deliberate, and please leave it that way: pi snapshots the system prompt
  // before `before_agent_start`, so a tool registered in that handler adds its
  // snippet only from the second turn on. That late change moves the cached
  // prompt prefix and throws away the provider's prompt cache for the whole
  // session. `index.ts` names the registered tools in the system prompt
  // instead, which is there from the first turn.
  const pi = fakePi();
  registerMcpTools(pi, fakeBridge());

  for (const definition of pi.registered) {
    assert.ok(!("promptSnippet" in definition), `${definition.name} must not set promptSnippet`);
    assert.ok(!("promptGuidelines" in definition), `${definition.name} must not set promptGuidelines`);
  }
});

// ---------------------------------------------------------------------------
// prepareArguments
// ---------------------------------------------------------------------------

test("prepareArguments is bound to that tool's own schema", () => {
  const pi = fakePi();
  registerMcpTools(pi, fakeBridge());
  const read = pi.byName("openviking_read");
  const find = pi.byName("openviking_find");

  // R2: `uris` is an array in read's schema, so a bare string is wrapped.
  // R1: `limit` is optional, so an explicit null is dropped rather than
  // coerced to 0 (which read zero lines on pi 0.80-0.84).
  assert.deepEqual(
    read.prepareArguments({ uris: "viking://memory/a.md", limit: null }),
    { uris: ["viking://memory/a.md"] },
  );
  // R3: a JSON array arriving as a string, exactly what FastMCP's
  // pre_parse_json accepts from every other harness.
  assert.deepEqual(
    read.prepareArguments({ uris: '["viking://memory/a.md","viking://memory/b.md"]' }),
    { uris: ["viking://memory/a.md", "viking://memory/b.md"] },
  );
  // Already valid arguments come through untouched.
  assert.deepEqual(
    read.prepareArguments({ uris: ["viking://memory/a.md"], offset: 10, limit: 5 }),
    { uris: ["viking://memory/a.md"], offset: 10, limit: 5 },
  );

  // The same bag against find's schema proves the binding is per tool rather
  // than shared: find knows `level` as an array and has never heard of `uris`,
  // so the wrap lands on `level` and `uris` passes through as an unknown key.
  assert.deepEqual(
    find.prepareArguments({ query: "peer", level: 1, uris: "viking://memory/a.md" }),
    { query: "peer", level: [1], uris: "viking://memory/a.md" },
  );
  assert.equal(toolSchema("find").properties.uris, undefined, "find really has no uris parameter");
});

test("prepareArguments repairs a single remember message into the list the server wants", () => {
  const pi = fakePi();
  registerMcpTools(pi, fakeBridge());
  const remember = pi.byName("openviking_remember");

  assert.deepEqual(
    remember.prepareArguments({ messages: { role: "user", content: "hi" } }),
    { messages: [{ role: "user", content: "hi" }] },
  );
  // An enum the model got wrong stays wrong: the server rejects it and the
  // model retries, which is what happens on every other harness.
  assert.deepEqual(
    remember.prepareArguments({ messages: [{ role: "USER", content: "hi" }] }),
    { messages: [{ role: "USER", content: "hi" }] },
  );
});

// ---------------------------------------------------------------------------
// execute
// ---------------------------------------------------------------------------

test("execute calls the bridge with the bare upstream name and the signal", async () => {
  const pi = fakePi();
  const bridge = fakeBridge();
  registerMcpTools(pi, bridge);

  const controller = new AbortController();
  const result = await pi.byName("openviking_read").execute(
    "call-1",
    { uris: ["viking://memory/a.md"] },
    controller.signal,
  );

  assert.equal(bridge.calls.length, 1);
  // The server never heard of `openviking_read`; the prefix is pi's view only.
  assert.equal(bridge.calls[0].name, "read");
  assert.deepEqual(bridge.calls[0].args, { uris: ["viking://memory/a.md"] });
  assert.equal(bridge.calls[0].options.signal, controller.signal, "ESC has to reach the bridge");
  assert.deepEqual(result.content, [{ type: "text", text: "result of read" }]);
});

test("execute passes a failing tool call through as a rejection", async () => {
  // The bridge turns `isError` results and JSON-RPC errors into throws, so pi
  // reports a failed tool call instead of showing the error text as an answer.
  const pi = fakePi();
  const bridge = fakeBridge();
  bridge.callTool = async () => {
    throw new Error("OpenViking MCP tools/call (forget) failed (HTTP 403)");
  };
  registerMcpTools(pi, bridge);

  await assert.rejects(
    () => pi.byName("openviking_forget").execute("call-2", { uri: "viking://memory/a.md" }, undefined),
    /HTTP 403/,
  );
});

test("a bridge that listed nothing registers nothing", () => {
  // The handshake failed or the server has no tools: the session keeps recall,
  // sync and takeover, and simply has no OpenViking tools.
  const pi = fakePi();
  assert.deepEqual(registerMcpTools(pi, fakeBridge([])), []);
  assert.deepEqual(registerMcpTools(pi, { state: { tools: null } }), []);
  assert.equal(pi.registered.length, 0);
});
