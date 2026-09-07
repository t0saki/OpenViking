import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { HARNESS_KEYS, KNOBS, KNOB_BY_KEY, WORKSPACE_KNOB_MAP } from "./lib/config-schema.mjs";
import { KNOWN_PLUGIN_KEYS, unknownPluginKeys } from "./lib/doctor-core.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

// Each loader reads its knobs off the object `resolveSettings` returns, so
// every `settings.<key>` in these files is a key the schema has to declare —
// and a key the doctor therefore has to accept inside `plugin`.
const LOADERS = [
  "examples/claude-code-memory-plugin/scripts/config.mjs",
  "examples/codex-memory-plugin/scripts/config.mjs",
  "examples/opencode-plugin/lib/config.mjs",
  "examples/dsh-memory-plugin/config.mjs",
  "examples/pi-coding-agent-extension/config.ts",
  "examples/memory-plugin-shared/lib/agent-hook-runtime.mjs",
];

test("every knob a loader reads is declared in the schema", async () => {
  const read = new Set();
  for (const file of LOADERS) {
    const source = await readFile(join(ROOT, file), "utf-8");
    for (const match of source.matchAll(/\bsettings\.([a-zA-Z_][a-zA-Z0-9_]*)/g)) read.add(match[1]);
  }
  assert.ok(read.size > 15, `expected to find the knobs, got ${read.size}`);

  const undeclared = [...read].filter((key) => !KNOB_BY_KEY.has(key)).sort();
  assert.deepEqual(undeclared, [], "a loader reads a key the schema never declares");
});

test("the doctor's known-knob list is the schema's own key list", () => {
  const declared = new Set(KNOB_BY_KEY.keys());
  assert.deepEqual([...KNOWN_PLUGIN_KEYS].sort(), [...declared].sort());
});

test("the schema is internally consistent", () => {
  const names = new Set();
  const envVars = new Map();
  for (const knob of KNOBS) {
    assert.ok(knob.name, "every knob needs a name");
    assert.ok(!names.has(knob.name), `duplicate knob ${knob.name}`);
    names.add(knob.name);
    assert.ok(knob.capability, `${knob.name} needs a capability`);
    if (knob.type === "enum") assert.ok(Array.isArray(knob.values), `${knob.name} needs its enum members`);
    if (knob.env) {
      // Two knobs sharing an environment variable would make one of them
      // unreachable, and which one depends on declaration order.
      assert.ok(!envVars.has(knob.env), `${knob.env} is claimed by ${envVars.get(knob.env)} and ${knob.name}`);
      envVars.set(knob.env, knob.name);
      assert.match(knob.env, /^OPENVIKING_/, `${knob.name}'s env var needs the shared prefix`);
    }
    for (const alias of knob.aliases || []) {
      assert.ok(!names.has(alias), `${alias} is both a knob and an alias`);
    }
  }

  // The workspace file names knobs by dotted path; a path pointing at nothing
  // would be a setting a repository could write that no loader ever reads.
  for (const [path, name] of Object.entries(WORKSPACE_KNOB_MAP)) {
    assert.ok(names.has(name), `${path} maps to unknown knob ${name}`);
  }
});

test("a misspelled knob is caught, with the key it was probably meant to be", () => {
  const found = unknownPluginKeys({
    recallCompress: "auto",
    peerSorce: "git",
    RecallLimit: 10,
    claude_code: { autoRecal: false },
    codex: { recallPeerScope: "actor" },
    opencode: { captureMode: "keyword" },
  });

  assert.deepEqual(found.map((f) => f.key).sort(), [
    "plugin.RecallLimit",
    "plugin.claude_code.autoRecal",
    "plugin.peerSorce",
  ]);
  assert.equal(found.find((f) => f.key === "plugin.peerSorce").suggestion, "peerSource");
  assert.equal(found.find((f) => f.key === "plugin.RecallLimit").suggestion, "recallLimit");
  assert.equal(found.find((f) => f.key === "plugin.claude_code.autoRecal").suggestion, "autoRecall");
});

test("every harness gets a per-harness override, under either spelling", () => {
  const overrides = Object.fromEntries(
    Object.values(HARNESS_KEYS).map((key) => [key, { recallLimit: 3 }]),
  );
  overrides["trae-cn"] = { recallLimit: 3 };
  assert.deepEqual(unknownPluginKeys(overrides), []);
});

test("an empty or absent plugin section reports nothing", () => {
  for (const value of [undefined, null, {}, [], "text", 3]) {
    assert.deepEqual(unknownPluginKeys(value), [], `should be empty for ${JSON.stringify(value)}`);
  }
});
