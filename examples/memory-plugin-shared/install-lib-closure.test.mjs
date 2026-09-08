/**
 * What the installer ships must equal what the shipped code imports.
 *
 * cursor, trae and zcode have no vendored copy of the shared runtime: the
 * installer assembles one in `$OV_HOME/agent-integrations/memory-plugin-shared/lib`
 * by copying the modules `lib/MANIFEST` names. A module the manifest forgets is
 * an ERR_MODULE_NOT_FOUND on the first hook of a fresh install, and one it
 * carries that nothing imports is dead weight nobody notices. The manifest is
 * generated from those sources by the same code that decides what the vendoring
 * targets ship, so it can only drift by not being regenerated — which is what
 * this asserts.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { MANIFEST_PATH, assembledClosure } from "./sync.mjs";

test("lib/MANIFEST holds the closure of what cursor, trae and zcode import", async () => {
  const manifest = readFileSync(MANIFEST_PATH, "utf8");
  assert.ok(manifest.endsWith("\n"), "the installer reads the manifest a line at a time");
  assert.deepEqual(
    manifest.split("\n").filter(Boolean),
    await assembledClosure(),
    "lib/MANIFEST is stale; run node examples/memory-plugin-shared/sync.mjs",
  );
});
