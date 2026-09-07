/**
 * What the installer ships must equal what the shipped code imports.
 *
 * cursor, trae and zcode have no vendored copy of the shared runtime: the
 * installer assembles one by copying a hand-written list into
 * `$OV_HOME/agent-integrations/memory-plugin-shared/lib`. A module that list
 * forgets is an ERR_MODULE_NOT_FOUND on the first hook of a fresh install, and
 * one it carries that nothing imports is dead weight nobody notices. The
 * imported side is derived from their sources by the same code that decides
 * what the vendoring targets ship, so neither can drift.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { assembledClosure } from "./sync.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

function installedFiles() {
  const script = readFileSync(join(HERE, "install.sh"), "utf8");
  const block = /shared_dest\.tmp[\s\S]*?\n\s*for file in\s*((?:[^\n]*\\\n)*[^\n]*?);\s*do/.exec(script);
  assert.ok(block, "install.sh no longer has the `for file in ...; do` list of shared modules");
  return new Set(block[1].replace(/\\\n/g, " ").trim().split(/\s+/).filter(Boolean));
}

test("the installer ships exactly the closure of what cursor, trae and zcode import", async () => {
  const installed = installedFiles();
  const imported = new Set(await assembledClosure());

  for (const file of [...installed].sort()) {
    assert.ok(imported.has(file), `install.sh ships ${file}, which no assembled entrypoint imports`);
  }
  for (const file of [...imported].sort()) {
    assert.ok(installed.has(file), `${file} is imported but install.sh never copies it`);
  }
});
