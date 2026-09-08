#!/usr/bin/env node

/**
 * The single entry every thin-harness hook command runs.
 *
 * Cursor, TRAE and ZCode each name it from their own hooks.json, passing the
 * event and the client id as arguments. They used to keep one shim script per
 * event instead — eleven files whose whole body was these two assignments and
 * an import of the harness dispatcher.
 */

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// The installer names each integration directory after the client id and this
// repository names it after the plugin; both put it beside the shared runtime.
const HOSTS = {
  cursor: { source: "cursor-memory-plugin", dispatcher: "cursor-hook.mjs" },
  trae: { source: "trae-memory-hooks", dispatcher: "trae-hook.mjs" },
  "trae-cn": { source: "trae-memory-hooks", dispatcher: "trae-hook.mjs" },
  zcode: { source: "zcode-memory-plugin", dispatcher: "zcode-hook.mjs" },
};

// ZCode's detached writer re-enters this file with no arguments at all, so what
// the first run put in the environment is what tells the worker what it is.
const event = process.argv[2] || process.env.OPENVIKING_HOOK_EVENT || "";
const client = process.argv[3] || process.env.OPENVIKING_HOOK_SOURCE || "";
const host = HOSTS[client];

const integrations = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const dispatcher = event && host
  ? [client, host.source]
    .map((dir) => join(integrations, dir, "scripts", host.dispatcher))
    .find((file) => existsSync(file))
  : null;

if (dispatcher) {
  process.env.OPENVIKING_HOOK_EVENT = event;
  process.env.OPENVIKING_HOOK_SOURCE = client;
  await import(pathToFileURL(dispatcher).href);
} else {
  process.stderr.write(`openviking: no hook dispatcher for ${client || "?"}/${event || "?"}\n`);
}
