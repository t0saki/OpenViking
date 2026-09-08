import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { HOSTS } from "../hosts/index.mjs";

const pluginRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const hostsDir = join(pluginRoot, "hosts");

const MANIFEST = "openviking.integration.json";

const templateDirs = readdirSync(hostsDir, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && existsSync(join(hostsDir, entry.name, MANIFEST)))
  .map((entry) => entry.name);

test("every adapter has a host template and every host template has an adapter", () => {
  // The installer reads `hosts/<kind>/` for the client it was asked to install
  // and refuses a manifest that does not name that client, so an adapter with
  // no template — or a template naming a client nothing serves — is an install
  // that fails on the user's machine rather than here.
  const declared = new Map();
  for (const dir of templateDirs) {
    const manifest = JSON.parse(readFileSync(join(hostsDir, dir, MANIFEST), "utf8"));
    assert.ok(
      Array.isArray(manifest.clients) && manifest.clients.length > 0,
      `hosts/${dir}/${MANIFEST} must name the clients it serves`,
    );
    for (const client of manifest.clients) {
      assert.equal(declared.get(client), undefined, `${client} is declared by two host templates`);
      declared.set(client, dir);
      assert.ok(HOSTS[client], `hosts/${dir} declares ${client}, which hosts/index.mjs does not serve`);
    }
  }
  for (const client of Object.keys(HOSTS)) {
    assert.ok(declared.has(client), `${client} has no hosts/<dir>/${MANIFEST} declaring it`);
  }
});

test("every adapter answers the four things the shared entry asks of it", () => {
  for (const [client, host] of Object.entries(HOSTS)) {
    assert.match(host.prefix, /^[a-z]+-$/, `${client} needs a session id prefix`);
    assert.equal(typeof host.envelope, "function", `${client} needs an envelope`);
    assert.equal(typeof host.guard, "function", `${client} needs a URI guard`);
    assert.equal(typeof host.prompt, "function", `${client} needs a prompt reader`);
    assert.equal(typeof host.capture, "function", `${client} needs a capture`);
    assert.ok(Object.values(host.stages).every((stage) => ["start", "prompt", "capture"].includes(stage)),
      `${client} maps an event to a stage the entry does not run`);
  }
});
