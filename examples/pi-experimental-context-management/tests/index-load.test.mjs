import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// pi loads extensions through the jiti it bundles, so that is the loader this
// smoke check has to use — a plain `import` of index.ts would not resolve the
// TypeScript or the `.js`-suffixed relative specifiers.
const JITI_PATH =
  "/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/jiti/lib/jiti.mjs";
const EXTENSION_DIR = dirname(dirname(fileURLToPath(import.meta.url)));

/** The pi surface index.ts touches at load time, and nothing more. */
function fakePi(calls) {
  return {
    on(event) { calls.events.push(event); },
    registerTool(tool) { calls.tools.push(tool?.name); },
    registerCommand(name) { calls.commands.push(name); },
    appendEntry() {},
    getAllTools() { return []; },
    sendMessage() {},
  };
}

test("index.ts loads through pi's jiti and registers its surface", { skip: !existsSync(JITI_PATH) }, async () => {
  const { createJiti } = await import(JITI_PATH);
  const jiti = createJiti(import.meta.url, { interopDefault: true });
  const mod = await jiti.import(join(EXTENSION_DIR, "index.ts"), { default: true });
  assert.equal(typeof mod, "function");

  const calls = { events: [], tools: [], commands: [] };
  await mod(fakePi(calls));

  for (const event of [
    "session_start",
    "before_agent_start",
    "context",
    "tool_call",
    "turn_end",
    "session_before_compact",
    "session_shutdown",
    "agent_end",
  ]) {
    assert.ok(calls.events.includes(event), `missing handler for ${event}`);
  }
  assert.deepEqual(calls.commands, ["viking"]);
  // Tools register from start(), which session_start drives; the load itself
  // must at least not have registered the archive tool the fork removed.
  assert.equal(calls.tools.includes("viking_archive_expand"), false);
});

test("the fork ships no takeover module", () => {
  for (const file of ["takeover.ts", "lib/takeover-core.mjs", "shared/recall-ledger.mjs"]) {
    assert.equal(existsSync(join(EXTENSION_DIR, file)), false, `${file} should not exist`);
  }
});
