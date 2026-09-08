/**
 * Kimi Code's config.toml belongs to the user and to whatever else writes hooks
 * into it.
 *
 * The installer owns exactly the text between its two marker comments: an
 * install rewrites that span and nothing else, a second install rewrites it to
 * the same bytes, and an uninstall gives the file back the way it was found.
 * The MCP half is JSON and follows the same rule the config-driven hosts do.
 */

import assert from "node:assert/strict";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { ownsHook } from "./lib/install/host-json-config.mjs";
import {
  blockMarkers,
  removeHostTomlHooks,
  stripTomlHooksBlock,
  writeHostTomlHooks,
} from "./lib/install/toml-hooks.mjs";

const HOSTS = resolve(dirname(fileURLToPath(import.meta.url)), "..", "agent-hook-plugin", "hosts");
const { begin, end } = blockMarkers("kimicode");

// What another tool's entries look like beside ours: Kimi Code's own docs point
// at herdr and orca, and both append `[[hooks]]` tables to the same file.
const FOREIGN = [
  '# my own notes',
  'model = "kimi-k2"',
  "",
  "[[hooks]]",
  'event = "Stop"',
  'command = "herdr sync"',
  "timeout = 15",
  "",
  "[[hooks]]",
  'event = "SessionStart"',
  'command = "orca notify"',
  "",
].join("\n");

// The uninstaller reclaims an MCP entry by the install path it names, so the
// assembled plugin sits where install.sh puts it.
function makeRoot(t) {
  const base = mkdtempSync(join(tmpdir(), "openviking-toml-hooks-"));
  t.after(() => rmSync(base, { recursive: true, force: true }));
  const root = join(base, "agent-integrations", "kimicode");
  mkdirSync(join(root, "hosts"), { recursive: true });
  cpSync(join(HOSTS, "kimicode"), join(root, "hosts", "kimicode"), { recursive: true });
  return root;
}

function install(root, home, { sourceMode = "dev" } = {}) {
  writeHostTomlHooks({
    kind: "kimicode",
    configPath: join(home, "config.toml"),
    mcpPath: join(home, "mcp.json"),
    root,
    clientId: "kimicode",
    nodeBin: "/usr/local/bin/node",
    sourceMode,
  });
}

function makeHome(t, { config = FOREIGN, mcp } = {}) {
  const home = mkdtempSync(join(tmpdir(), "openviking-kimi-home-"));
  t.after(() => rmSync(home, { recursive: true, force: true }));
  if (config !== null) writeFileSync(join(home, "config.toml"), config);
  if (mcp) writeFileSync(join(home, "mcp.json"), `${JSON.stringify(mcp, null, 2)}\n`);
  return home;
}

/** Every `command = "..."` value inside the installer's own block. */
function blockCommands(text) {
  const start = text.indexOf(begin);
  const stop = text.indexOf(end, start);
  assert.ok(start >= 0 && stop > start, "the installer's block is missing");
  return [...text.slice(start, stop).matchAll(/^command = "(.*)"$/gmu)].map((match) => match[1]);
}

test("installing leaves every foreign [[hooks]] entry byte for byte", (t) => {
  const root = makeRoot(t);
  const home = makeHome(t);
  install(root, home);

  const text = readFileSync(join(home, "config.toml"), "utf8");
  assert.ok(text.startsWith(FOREIGN.replace(/\n+$/u, "")), text);
  assert.match(text, /command = "herdr sync"/u);
  assert.match(text, /command = "orca notify"/u);
  assert.equal(text.indexOf(begin) > 0, true);
  assert.ok(text.endsWith(`${end}\n`), text);
});

test("the rendered commands are ones the uninstaller recognises as its own", (t) => {
  const root = makeRoot(t);
  const home = makeHome(t);
  install(root, home);

  const commands = blockCommands(readFileSync(join(home, "config.toml"), "utf8"));
  const template = JSON.parse(readFileSync(join(HOSTS, "kimicode", "hooks.json"), "utf8"));
  assert.equal(commands.length, template.hooks.length);
  for (const command of commands) {
    assert.ok(ownsHook(command), command);
    assert.match(command, /OPENVIKING_INTEGRATION_ID='openviking-memory'/u);
    assert.match(command, /OPENVIKING_HOOK_SOURCE='kimicode'/u);
    assert.match(command, /# openviking-memory$/u);
    assert.ok(command.includes(root), command);
    assert.equal(command.includes("__OPENVIKING_PLUGIN_ROOT__"), false, command);
    assert.equal(command.includes("__OPENVIKING_CLIENT_ID__"), false, command);
  }
  assert.equal(commands.filter((command) => command.includes("uri-guard.mjs")).length, 1);
  // The matcher rides along with the entry it belongs to, not with another one.
  const text = readFileSync(join(home, "config.toml"), "utf8");
  assert.match(text, /event = "PreToolUse"\nmatcher = "Read\|Glob\|Grep"\ncommand = "[^\n]*uri-guard\.mjs/u);
});

test("a second install rewrites the block instead of appending another", (t) => {
  const root = makeRoot(t);
  const home = makeHome(t);
  install(root, home);
  const first = readFileSync(join(home, "config.toml"), "utf8");
  install(root, home);
  const second = readFileSync(join(home, "config.toml"), "utf8");

  assert.equal(second, first);
  assert.equal(second.split(begin).length - 1, 1);
  assert.equal(second.split("[[hooks]]").length - 1, 2 + blockCommands(second).length);
});

test("an install over a block whose commands moved replaces them", (t) => {
  const root = makeRoot(t);
  const home = makeHome(t, {
    config: `${FOREIGN}\n${begin}\n[[hooks]]\nevent = "Stop"\ncommand = "node /old/openviking/scripts/hook.mjs stop kimicode"\n${end}\n`,
  });
  install(root, home);

  const text = readFileSync(join(home, "config.toml"), "utf8");
  assert.equal(text.includes("/old/openviking"), false, text);
  assert.match(text, /command = "herdr sync"/u);
});

test("uninstalling gives the file back the way it was found", (t) => {
  const root = makeRoot(t);
  const home = makeHome(t);
  install(root, home);
  removeHostTomlHooks({
    configPath: join(home, "config.toml"),
    mcpPath: join(home, "mcp.json"),
    clientId: "kimicode",
  });

  const text = readFileSync(join(home, "config.toml"), "utf8");
  assert.equal(text, FOREIGN.replace(/\n+$/u, "\n"));
  assert.equal(text.includes("openviking"), false, text);
  // A backup taken while removing entries would keep a copy of them.
  assert.equal(existsSync(join(home, "config.toml.bak")), false);
  assert.equal(existsSync(join(home, "mcp.json.bak")), false);
});

test("uninstalling a config the installer never touched changes nothing", (t) => {
  const home = makeHome(t);
  removeHostTomlHooks({
    configPath: join(home, "config.toml"),
    mcpPath: join(home, "mcp.json"),
    clientId: "kimicode",
  });
  assert.equal(readFileSync(join(home, "config.toml"), "utf8"), FOREIGN);
});

test("a half-written block is left alone rather than guessed at", () => {
  const text = `model = "kimi-k2"\n${begin}\n[[hooks]]\nevent = "Stop"\n`;
  assert.equal(stripTomlHooksBlock(text, "kimicode"), text);
});

test("the MCP merge keeps servers this installer did not write", (t) => {
  const root = makeRoot(t);
  const home = makeHome(t, {
    mcp: {
      mcpServers: {
        "ov-mcp-server": { url: "http://127.0.0.1:1933/mcp" },
        "third-party": { command: "node", args: ["/opt/third-party/server.js"] },
      },
    },
  });
  install(root, home);

  const servers = JSON.parse(readFileSync(join(home, "mcp.json"), "utf8")).mcpServers;
  assert.equal(servers.openviking.command, "/usr/local/bin/node");
  assert.deepEqual(servers.openviking.args, [join(root, "servers", "mcp-proxy.mjs")]);
  assert.equal(servers.openviking.env.OPENVIKING_INTEGRATION_ID, "openviking-memory");
  assert.equal(servers.openviking.env.OPENVIKING_HOOK_SOURCE, "kimicode");
  assert.ok(servers["third-party"]);
  assert.equal(Boolean(servers["ov-mcp-server"]), false, "the published legacy endpoint is migrated");

  removeHostTomlHooks({
    configPath: join(home, "config.toml"),
    mcpPath: join(home, "mcp.json"),
    clientId: "kimicode",
  });
  const after = JSON.parse(readFileSync(join(home, "mcp.json"), "utf8")).mcpServers;
  assert.equal(Boolean(after.openviking), false);
  assert.ok(after["third-party"]);
});

test("the installed manifest records the TOML config as this client's hooks file", (t) => {
  const root = makeRoot(t);
  const home = makeHome(t);
  install(root, home);
  const first = JSON.parse(readFileSync(join(root, "integration.json"), "utf8"));
  install(root, home);
  const second = JSON.parse(readFileSync(join(root, "integration.json"), "utf8"));

  assert.equal(first.client, "kimicode");
  assert.equal(first.installMode, "managed-native");
  assert.equal(first.hooksConfig, join(home, "config.toml"));
  assert.equal(first.mcpConfig, join(home, "mcp.json"));
  assert.deepEqual(second, first, "an unchanged reinstall must not move the timestamps");
});
