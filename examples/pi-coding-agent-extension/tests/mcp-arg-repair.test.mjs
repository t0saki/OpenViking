/**
 * The 39-case argument corpus for `lib/mcp-arg-repair.mjs`.
 *
 * Both halves of the corpus carry weight, and the second half is the one that
 * matters most. Repair exists only to erase the extra local validation pass pi
 * runs before it dispatches a tool call; every other harness hands the model's
 * arguments straight to the server. So a case R1-R3 covers has to validate
 * after repair, and every case outside R1-R3 has to STILL be rejected, because
 * the alternative is pi quietly accepting calls the server would have refused.
 *
 * The corpus was assembled during the spike as an adversarial battery against
 * pi's real validator (`spikeCheck_check_a.mjs`), and is driven here against
 * the real server schemas in `fixtures/mcp-tools-list.json`.
 *
 * CI resolves nothing from pi's package, so the validator below is a port
 * rather than an import. See its own comment for what it reproduces.
 *
 * Past the corpus sit cases of our own, for an invariant none of the 39 happen
 * to exercise: R3's schema-type guard.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { makeArgRepair, repairArguments } from "../lib/mcp-arg-repair.mjs";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/mcp-tools-list.json", import.meta.url), "utf8"));
const SCHEMAS = Object.fromEntries(fixture.tools.map((tool) => [tool.name, tool.inputSchema]));

/**
 * `schemaFor` hands back the server's schema untouched. The bridge tidies a
 * schema before it reaches pi, but tidying only drops `$schema`, the root
 * `additionalProperties` and the `title` keyword — none of which repair reads
 * or the validator enforces — so the corpus runs against the raw article and
 * stays independent of the bridge module.
 */
function schemaFor(toolName) {
  const schema = SCHEMAS[toolName];
  assert.ok(schema, `fixture has no schema for ${toolName}`);
  return structuredClone(schema);
}

// ---------------------------------------------------------------------------
// Ported validator
//
// pi-ai 0.80.3's `validateToolArguments` (dist/utils/validation.js) runs
// typebox's `Value.Convert`, then its own `coerceWithJsonSchema`, then
// `Compile(schema).Check`. What follows reproduces the coercion layer
// faithfully and covers the JSON Schema keywords OpenViking's schemas actually
// use: type, enum, required, properties, items, additionalProperties, minimum,
// maximum, minItems, maxItems. Those schemas are made portable server-side
// (`_apply_portable_schemas`), so there is no `$ref`, no `anyOf` and no type
// array to handle. The typebox pass is left out because it changes nothing on
// this corpus: every accept and every reject below was checked against the real
// validator, value for coerced value, before being written down.
//
// A verdict here therefore means "pi would accept/reject this", not "this test
// file's own idea of JSON Schema".
// ---------------------------------------------------------------------------

function isRecord(value) {
  return typeof value === "object" && value !== null;
}

function isPlainObject(value) {
  return isRecord(value) && !Array.isArray(value);
}

function declaredType(schema) {
  return typeof schema.type === "string" ? schema.type : "";
}

function matchesJsonType(value, type) {
  switch (type) {
    case "number": return typeof value === "number" && Number.isFinite(value);
    case "integer": return typeof value === "number" && Number.isInteger(value);
    case "boolean": return typeof value === "boolean";
    case "string": return typeof value === "string";
    case "null": return value === null;
    case "array": return Array.isArray(value);
    case "object": return isPlainObject(value);
    default: return false;
  }
}

/** Verbatim port of pi's `coercePrimitiveByType`. */
function coercePrimitive(value, type) {
  switch (type) {
    case "number": {
      if (value === null) return 0;
      if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) return parsed;
      }
      if (typeof value === "boolean") return value ? 1 : 0;
      return value;
    }
    case "integer": {
      if (value === null) return 0;
      if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        if (Number.isInteger(parsed)) return parsed;
      }
      if (typeof value === "boolean") return value ? 1 : 0;
      return value;
    }
    case "boolean": {
      if (value === null) return false;
      if (value === "true") return true;
      if (value === "false") return false;
      if (value === 1) return true;
      if (value === 0) return false;
      return value;
    }
    case "string": {
      if (value === null) return "";
      if (typeof value === "number" || typeof value === "boolean") return String(value);
      return value;
    }
    default:
      return value;
  }
}

function coerce(value, schema) {
  if (!isRecord(schema)) return value;
  let next = value;
  const type = declaredType(schema);
  if (type && !matchesJsonType(next, type)) next = coercePrimitive(next, type);

  if (type === "object" && isPlainObject(next)) {
    const properties = isPlainObject(schema.properties) ? schema.properties : null;
    const declared = new Set(properties ? Object.keys(properties) : []);
    if (properties) {
      for (const [key, propertySchema] of Object.entries(properties)) {
        if (!(key in next)) continue;
        next[key] = coerce(next[key], propertySchema);
      }
    }
    if (isPlainObject(schema.additionalProperties)) {
      for (const [key, propertyValue] of Object.entries(next)) {
        if (declared.has(key)) continue;
        next[key] = coerce(propertyValue, schema.additionalProperties);
      }
    }
  }
  if (type === "array" && Array.isArray(next) && isPlainObject(schema.items)) {
    for (let i = 0; i < next.length; i++) next[i] = coerce(next[i], schema.items);
  }
  return next;
}

function check(value, schema, path, errors) {
  if (!isRecord(schema)) return;
  const where = path || "root";
  const type = declaredType(schema);
  if (type && !matchesJsonType(value, type)) {
    errors.push(`${where}: must be ${type}`);
    return;
  }
  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
    errors.push(`${where}: must be one of ${JSON.stringify(schema.enum)}`);
  }
  if (typeof value === "number") {
    if (typeof schema.minimum === "number" && value < schema.minimum) errors.push(`${where}: must be >= ${schema.minimum}`);
    if (typeof schema.maximum === "number" && value > schema.maximum) errors.push(`${where}: must be <= ${schema.maximum}`);
  }
  if (Array.isArray(value)) {
    if (typeof schema.maxItems === "number" && value.length > schema.maxItems) errors.push(`${where}: must not have more than ${schema.maxItems} items`);
    if (typeof schema.minItems === "number" && value.length < schema.minItems) errors.push(`${where}: must have at least ${schema.minItems} items`);
    if (isPlainObject(schema.items)) {
      value.forEach((item, index) => check(item, schema.items, `${where}.${index}`, errors));
    }
  }
  if (isPlainObject(value)) {
    const properties = isPlainObject(schema.properties) ? schema.properties : {};
    for (const key of Array.isArray(schema.required) ? schema.required : []) {
      if (!(key in value)) errors.push(`${path ? `${path}.` : ""}${key}: must have required property ${key}`);
    }
    for (const [key, propertyValue] of Object.entries(value)) {
      const at = `${path ? `${path}.` : ""}${key}`;
      const propertySchema = properties[key];
      if (isPlainObject(propertySchema)) check(propertyValue, propertySchema, at, errors);
      else if (schema.additionalProperties === false) errors.push(`${at}: must not have additional properties`);
      else if (isPlainObject(schema.additionalProperties)) check(propertyValue, schema.additionalProperties, at, errors);
    }
  }
}

/** Throws the way pi throws; returns the coerced arguments the tool would see. */
function validate(args, schema) {
  const coerced = coerce(structuredClone(args), schema);
  const errors = [];
  check(coerced, schema, "", errors);
  if (errors.length > 0) throw new Error(errors.join(" | "));
  return coerced;
}

/** The production path: `prepareArguments` runs, then pi validates. */
function repairThenValidate(toolName, args) {
  const schema = schemaFor(toolName);
  const repaired = makeArgRepair(schema)(structuredClone(args));
  return { repaired, validated: validate(repaired, schema) };
}

function validatesAfterRepair(toolName, args) {
  try {
    repairThenValidate(toolName, args);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Half one: what R1-R3 are for.
//
// `via` says why the case lands on the accept side. "R1"/"R2"/"R3" mean repair
// is load-bearing — the assertions below re-run those cases without repair and
// require them to fail. "contract" is `repairArguments`' own promise about a
// non-object bag. "pi" means pi's built-in coercion already handled it and
// repair correctly stayed out of the way.
// ---------------------------------------------------------------------------

const ACCEPTED = [
  // A required null is left in place on purpose, so a model that means "no
  // content" gets a real error instead of a call that means something else.
  // pi's own coercion then turns it into "" — out of this module's hands, and
  // the one accept here that a server with no local validation would reject.
  {
    label: "remember: nested explicit null content",
    tool: "remember", via: "pi",
    args: { messages: [{ role: "user", content: null }] },
    repaired: { messages: [{ role: "user", content: null }] },
    validated: { messages: [{ role: "user", content: "" }] },
  },
  // R2: schema says array, model sent a scalar. pi rejects it outright through
  // 0.86.1. Note this widens `List[int]` params too, which the server would
  // have rejected — the accepted cost of R2 being schema-driven, since the MCP
  // schema for `str | list[str]` and for a plain `list[int]` look the same.
  {
    label: "find: level as a string",
    tool: "find", via: "R2",
    args: { query: "q", level: "1" },
    repaired: { query: "q", level: ["1"] },
    validated: { query: "q", level: [1] },
  },
  {
    label: "find: level as string array",
    tool: "find", via: "pi",
    args: { query: "q", level: ["1", "2"] },
    repaired: { query: "q", level: ["1", "2"] },
    validated: { query: "q", level: [1, 2] },
  },
  // Unknown keys pass through untouched and the root schema does not forbid
  // them, so this validates and the server ignores the extras — the same
  // silent drop every other harness gets, which is the parity we want.
  {
    label: "find: camelCase params",
    tool: "find", via: "pi",
    args: { query: "q", maxResults: 50, minScore: 0.9 },
    repaired: { query: "q", maxResults: 50, minScore: 0.9 },
    validated: { query: "q", maxResults: 50, minScore: 0.9 },
  },
  {
    label: "search: quotas with string values",
    tool: "search", via: "pi",
    args: { query: "q", quotas: { memory: "3" } },
    repaired: { query: "q", quotas: { memory: "3" } },
    validated: { query: "q", quotas: { memory: 3 } },
  },
  {
    label: "search: detail_by_category with bool values",
    tool: "search", via: "pi",
    args: { query: "q", detail_by_category: { memory: true } },
    repaired: { query: "q", detail_by_category: { memory: true } },
    validated: { query: "q", detail_by_category: { memory: "true" } },
  },
  // R1: without it pi turns the null into 0, which then trips `minimum: 64` —
  // and on pi 0.80-0.84 a null `read.limit` arrived at the server as 0 and read
  // zero lines. Dropping the key is what the server means by None.
  {
    label: "search: max_tokens null",
    tool: "search", via: "R1",
    args: { query: "q", max_tokens: null },
    repaired: { query: "q" },
    validated: { query: "q" },
  },
  {
    label: "list: limit as string",
    tool: "list", via: "pi",
    args: { uri: "v://", limit: "20" },
    repaired: { uri: "v://", limit: "20" },
    validated: { uri: "v://", limit: 20 },
  },
  {
    label: "read: offset/limit as strings",
    tool: "read", via: "pi",
    args: { uris: ["v://a"], offset: "10", limit: "20" },
    repaired: { uris: ["v://a"], offset: "10", limit: "20" },
    validated: { uris: ["v://a"], offset: 10, limit: 20 },
  },
  // R1 deliberately does not reach into array elements, so the null survives
  // repair and pi coerces it to "". Also out of this module's hands.
  {
    label: "read: uris array containing a null",
    tool: "read", via: "pi",
    args: { uris: ["v://a", null] },
    repaired: { uris: ["v://a", null] },
    validated: { uris: ["v://a", ""] },
  },
  // R2 wraps rather than splits, which is exactly what the server's
  // `uris: str | list[str]` does with the same string: one URI that happens to
  // contain a comma. Passes validation, fails to resolve, same as elsewhere.
  {
    label: "read: uris as a comma-separated string",
    tool: "read", via: "R2",
    args: { uris: "v://a,v://b" },
    repaired: { uris: ["v://a,v://b"] },
    validated: { uris: ["v://a,v://b"] },
  },
  // `tags` is a plain `list[str]` server-side, not a union, so this is the same
  // widening as `find.level` above: one tag that happens to contain a comma,
  // where another harness would have got an error back.
  {
    label: "add_resource: tags comma-separated",
    tool: "add_resource", via: "R2",
    args: { path: "/tmp/x", tags: "a,b" },
    repaired: { path: "/tmp/x", tags: ["a,b"] },
    validated: { path: "/tmp/x", tags: ["a,b"] },
  },
  // R3: the same parse FastMCP's `pre_parse_json` performs for every other
  // harness, so this reproduces the server rather than going past it.
  {
    label: "add_resource: args as a JSON string",
    tool: "add_resource", via: "R3",
    args: { path: "/tmp/x", args: '{"k":1}' },
    repaired: { path: "/tmp/x", args: { k: 1 } },
    validated: { path: "/tmp/x", args: { k: 1 } },
  },
  // `repairArguments` turns a non-object bag into {}, which is what an omitted
  // `arguments` means over MCP. A tool with required properties still fails.
  {
    label: "arguments null",
    tool: "health", via: "contract",
    args: null,
    repaired: {},
    validated: {},
  },
  {
    label: "grep: pattern array of regex + literal",
    tool: "grep", via: "pi",
    args: { uri: "v://", pattern: ["TODO", "FIXME"] },
    repaired: { uri: "v://", pattern: ["TODO", "FIXME"] },
    validated: { uri: "v://", pattern: ["TODO", "FIXME"] },
  },
];

// ---------------------------------------------------------------------------
// Half two: the deliberate non-goals.
//
// Every one of these must still be rejected after repair. `pins` names the
// non-goal the case holds down. On any other harness the server rejects or
// ignores these just the same, and the model retries once it reads the error.
// ---------------------------------------------------------------------------

const REJECTED = [
  {
    label: "remember: a system message in the transcript",
    tool: "remember",
    pins: "no widening an enum to a member the server does not declare",
    args: { messages: [{ role: "system", content: "you are..." }, { role: "user", content: "hi" }] },
  },
  {
    label: "remember: role capitalised (nested enum)",
    tool: "remember",
    pins: "no enum case fixing, nested or otherwise",
    args: { messages: [{ role: "User", content: "hi" }] },
  },
  {
    label: "remember: items use `text` not `content`",
    tool: "remember",
    pins: "no key renaming; the unknown key rides along and `content` stays missing",
    args: { messages: [{ role: "user", text: "hi" }] },
  },
  {
    label: "remember: plain strings instead of objects",
    tool: "remember",
    pins: "R2 wraps a value into an array, it never builds the object `items` declares",
    args: { messages: ["hi", "there"] },
  },
  {
    label: "find: level as a float",
    tool: "find",
    pins: "no rounding a float into an integer field",
    args: { query: "q", level: [1.5] },
  },
  {
    label: "find: read_content 'yes'",
    tool: "find",
    pins: "no boolean synonyms beyond the true/false pi itself converts",
    args: { query: "q", read_content: "yes" },
  },
  {
    label: "find: limit as a range string",
    tool: "find",
    pins: "no reading a number out of a string that is not one",
    args: { query: "q", limit: "1-10" },
  },
  {
    label: "search: quotas with float values",
    tool: "search",
    pins: "no rounding, including inside `additionalProperties`",
    args: { query: "q", quotas: { memory: 3.5 } },
  },
  {
    label: "search: quotas as an array of pairs",
    tool: "search",
    pins: "no restructuring an array of pairs into the object the schema wants",
    args: { query: "q", quotas: [["memory", 3]] },
  },
  {
    label: "search: exclude_uris over maxItems 200",
    tool: "search",
    pins: "no `maxItems` truncation — silently dropping 50 URIs would change what the call means",
    args: { query: "q", exclude_uris: Array.from({ length: 250 }, (_, i) => `viking://m/${i}`) },
  },
  {
    label: "search: max_tokens as a string over the max",
    tool: "search",
    pins: "no clamping to `minimum`/`maximum`; pi parses the string, the bound still rejects it",
    args: { query: "q", max_tokens: "99999" },
  },
  {
    label: "search: mode with a space",
    tool: "search",
    pins: "no guessing at an enum member from nearby text",
    args: { query: "q", mode: "context only" },
  },
  {
    label: "write: mode 'overwrite'",
    tool: "write",
    pins: "no enum synonyms, however plausible — 'overwrite' is not 'replace'",
    args: { uri: "v://a", content: "x", mode: "overwrite" },
  },
  {
    label: "write: content is an object",
    tool: "write",
    pins: "no stringifying an object into a string field; R3 only ever parses, never serialises",
    args: { uri: "v://a", content: { a: 1 } },
  },
  {
    label: "write: timeout '30s'",
    tool: "write",
    pins: "no stripping a unit suffix off a numeric string",
    args: { uri: "v://a", content: "x", timeout: "30s" },
  },
  {
    label: "edit: Claude-Code style edits array",
    tool: "edit",
    pins: "no translating another harness's tool shape; `edits` passes through as an unknown key and the required properties stay missing",
    args: { uri: "v://a", edits: [{ old_string: "a", new_string: "b" }] },
  },
  {
    label: "edit: required new_string omitted",
    tool: "edit",
    pins: "no synthesizing a missing required property, and no guessing that the omission meant a deletion",
    args: { uri: "v://a", old_string: "a" },
  },
  {
    label: "list: sort_by 'modified'",
    tool: "list",
    pins: "no enum synonyms — 'modified' is not 'mtime'",
    args: { uri: "v://", sort_by: "modified" },
  },
  {
    label: "glob: pattern as an array",
    tool: "glob",
    pins: "R2 is one-directional: it never unwraps a single-element array into the scalar the schema wants",
    args: { pattern: ["**/*.md"] },
  },
  {
    label: "add_resource: watch_interval '5m'",
    tool: "add_resource",
    pins: "no unit suffixes on a number, here too",
    args: { path: "/tmp/x", watch_interval: "5m" },
  },
  {
    label: "whole call wrapped in {input:{...}}",
    tool: "find",
    pins: "no unwrapping `{input:{...}}`; the wrapper is an unknown key and `query` stays missing",
    args: { input: { query: "q" } },
  },
  {
    label: "whole call wrapped in {arguments:{...}}",
    tool: "find",
    pins: "no unwrapping `{arguments:{...}}` either",
    args: { arguments: { query: "q" } },
  },
  {
    label: "arguments sent as a JSON string",
    tool: "find",
    pins: "R3 parses a property whose schema declares an array or object, never the whole argument bag",
    args: '{"query":"q","limit":5}',
  },
  {
    label: "arguments sent as a python-repr string",
    tool: "find",
    pins: "same, and single-quoted python repr is not JSON to begin with",
    args: "{'query': 'q'}",
  },
];

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

test("the ported corpus is the whole 39 cases and every tool resolves", () => {
  assert.equal(ACCEPTED.length, 15);
  assert.equal(REJECTED.length, 24);
  assert.equal(ACCEPTED.length + REJECTED.length, 39);
  for (const entry of [...ACCEPTED, ...REJECTED]) assert.ok(SCHEMAS[entry.tool], `${entry.label}: unknown tool ${entry.tool}`);
  const labels = [...ACCEPTED, ...REJECTED].map((entry) => entry.label);
  assert.equal(new Set(labels).size, labels.length, "corpus labels must be unique");
});

for (const entry of ACCEPTED) {
  test(`accepted after repair (${entry.via}): ${entry.label}`, () => {
    const { repaired, validated } = repairThenValidate(entry.tool, entry.args);
    assert.deepEqual(repaired, entry.repaired, `${entry.label}: repaired arguments`);
    assert.deepEqual(validated, entry.validated, `${entry.label}: what the tool would receive`);
  });
}

for (const entry of REJECTED) {
  test(`still rejected after repair: ${entry.label}`, () => {
    const schema = schemaFor(entry.tool);
    const repaired = makeArgRepair(schema)(structuredClone(entry.args));
    // Pins: <entry.pins>. Repair may reshape the bag, but pi must still refuse
    // the call, because the server refuses it for every other harness.
    assert.throws(
      () => validate(repaired, schema),
      /.+/,
      `${entry.label} was accepted after repair, so pi is now more permissive than the server. Pins: ${entry.pins}`,
    );
  });
}

// ---------------------------------------------------------------------------
// R3's schema-type guard. Beyond the ported corpus, which stays green without
// it.
//
// R3 parses a bracket-leading string only where the schema declares an array or
// an object. Drop that type check and a JSON-shaped string bound for a
// string-typed field is parsed too: `write({content: '{"k": 1}'})` reaches pi's
// validator as an object and is refused with `content: must be string`. pi would
// then be STRICTER than the server for a call every other harness forwards
// untouched and the server stores verbatim — the exact inversion this file
// exists to prevent. Every value below is valid JSON, so each case fails on its
// own if the guard goes.
// ---------------------------------------------------------------------------

const BRACKET_LEADING_STRINGS = [
  {
    label: "write: content is a JSON object literal",
    tool: "write",
    args: { uri: "viking://memories/note.md", content: '{"k": 1}' },
    pick: (bag) => bag.content,
  },
  {
    label: "write: content is a JSON array literal",
    tool: "write",
    args: { uri: "viking://memories/note.md", content: "[1, 2, 3]", mode: "append" },
    pick: (bag) => bag.content,
  },
  {
    label: "edit: old_string/new_string are JSON fragments",
    tool: "edit",
    args: { uri: "viking://resources/config.json", old_string: '{"retries": 3}', new_string: '{"retries": 5}' },
    pick: (bag) => bag.new_string,
  },
  {
    label: "remember: a message whose content is a JSON object",
    tool: "remember",
    args: { messages: [{ role: "user", content: '{"event": "deploy", "ok": true}' }] },
    pick: (bag) => bag.messages[0].content,
  },
  {
    label: "search: a query quoting a JSON snippet",
    tool: "search",
    args: { query: '{"error": "timeout"}' },
    pick: (bag) => bag.query,
  },
];

for (const entry of BRACKET_LEADING_STRINGS) {
  test(`a string-typed field survives repair byte-identical: ${entry.label}`, () => {
    const { repaired, validated } = repairThenValidate(entry.tool, entry.args);
    const picked = entry.pick(repaired);
    assert.equal(
      typeof picked,
      "string",
      `${entry.label}: R3 parsed a string-typed field into ${typeof picked}, so pi now rejects a call the server accepts`,
    );
    assert.equal(picked, entry.pick(entry.args), `${entry.label}: the string was rewritten`);
    assert.deepEqual(repaired, entry.args, `${entry.label}: repaired arguments`);
    assert.deepEqual(validated, entry.args, `${entry.label}: what the tool would receive`);
  });
}

test("the rules are load-bearing: every R1-R3 case fails without repair", () => {
  const driven = ACCEPTED.filter((entry) => entry.via !== "pi");
  assert.deepEqual(driven.map((entry) => entry.via).sort(), ["R1", "R2", "R2", "R2", "R3", "contract"]);
  for (const entry of driven) {
    assert.throws(
      () => validate(structuredClone(entry.args), schemaFor(entry.tool)),
      /.+/,
      `${entry.label} validates unrepaired, so ${entry.via} is not what makes it pass`,
    );
    assert.equal(validatesAfterRepair(entry.tool, entry.args), true, `${entry.label} should validate after repair`);
  }
});

// ---------------------------------------------------------------------------
// Module contract
// ---------------------------------------------------------------------------

test("repair is a no-op on arguments that were already valid", () => {
  const valid = [
    ["find", { query: "kafka retry", limit: 5, level: [1], read_content: true }],
    ["read", { uris: ["viking://memories/a.md"], offset: 0, limit: -1 }],
    ["remember", { messages: [{ role: "user", content: "q" }, { role: "assistant", content: "a" }] }],
    ["search", { query: "q", mode: "context", quotas: { memory: 5 }, exclude_uris: ["viking://m/x"] }],
    ["glob", { pattern: "**/*.md" }],
    ["health", {}],
  ];
  for (const [tool, args] of valid) {
    const { repaired, validated } = repairThenValidate(tool, args);
    assert.deepEqual(repaired, args, `${tool}: repair rewrote valid arguments`);
    assert.deepEqual(validated, args, `${tool}: validation rewrote valid arguments`);
  }
});

test("repair never mutates the arguments or the schema it is given", () => {
  const schema = schemaFor("search");
  const before = JSON.stringify(schema);
  const args = { query: "q", max_tokens: null, exclude_uris: "viking://m/x" };
  const snapshot = JSON.stringify(args);
  const out = repairArguments(args, schema);
  assert.deepEqual(out, { query: "q", exclude_uris: ["viking://m/x"] });
  assert.equal(JSON.stringify(args), snapshot);
  assert.equal(JSON.stringify(schema), before);
});

test("a missing or malformed schema hands the arguments back unchanged", () => {
  const args = { query: "q", level: "1", target_uri: null };
  assert.deepEqual(repairArguments(args, undefined), args);
  assert.deepEqual(repairArguments(args, null), args);
  assert.deepEqual(repairArguments(args, "not a schema"), args);
  assert.notEqual(repairArguments(args, undefined), args, "the copy must not be the caller's object");
});

test("a non-object argument bag always becomes {}", () => {
  const schema = schemaFor("health");
  for (const bag of [null, undefined, "{}", 7, true, ["a"]]) {
    assert.deepEqual(repairArguments(bag, schema), {});
    assert.deepEqual(makeArgRepair(schema)(bag), {});
  }
});

test("makeArgRepair binds one schema and is reusable", () => {
  const repair = makeArgRepair(schemaFor("read"));
  assert.deepEqual(repair({ uris: "viking://a" }), { uris: ["viking://a"] });
  assert.deepEqual(repair({ uris: ["viking://b"], limit: null }), { uris: ["viking://b"] });
});
