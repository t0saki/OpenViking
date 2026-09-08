import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, mkdirSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const installer = join(ROOT, "examples", "memory-plugin-shared", "install.sh");
const stageScript = join(ROOT, ".github", "scripts", "stage-memory-plugin-marketplace.sh");
const archiveCheck = join(ROOT, ".github", "scripts", "check-marketplace-archive.mjs");

function run(command, args, options = {}) {
  return spawnSync(command, args, {
    cwd: ROOT,
    encoding: "utf8",
    ...options,
  });
}

/** Stage the marketplace tree, zip it, and check it carries no dependencies. */
function stageArchive(tmp) {
  const stage = join(tmp, "memory-plugin-marketplace");
  const staged = run("bash", [stageScript, stage]);
  assert.equal(staged.status, 0, `${staged.stdout}\n${staged.stderr}`);

  const bundled = readdirSync(stage, { recursive: true, encoding: "utf8" }).filter((entry) =>
    entry.split(sep).includes("node_modules"),
  );
  assert.deepEqual(bundled.slice(0, 3), [], "marketplace archive carries development dependencies");

  const zip = join(tmp, "memory-plugin-marketplace.zip");
  const zipped = run("zip", ["-rq", zip, "memory-plugin-marketplace"], { cwd: tmp });
  assert.equal(zipped.status, 0, `${zipped.stdout}\n${zipped.stderr}`);
  return zip;
}

function installFromArchive(tmp, zip, harness) {
  const home = join(tmp, "home");
  mkdirSync(home, { recursive: true });
  const installed = run("bash", [
    installer,
    "--harness", harness,
    "--dist", "tos",
    "--source", "archive",
    "--lang", "en",
    "--url", "http://127.0.0.1:1933",
    "--api-key", "",
    "--yes",
  ], {
    env: {
      ...process.env,
      HOME: home,
      OPENVIKING_HOME: join(home, ".openviking"),
      OPENVIKING_MARKETPLACE_ARCHIVE_URL: `file://${zip}`,
    },
  });
  assert.equal(installed.status, 0, `${installed.stdout}\n${installed.stderr}`);
  return home;
}

test("release marketplace archive supports a ZCode TOS install", () => {
  const tmp = mkdtempSync(join(tmpdir(), "openviking-zcode-release-"));
  try {
    const home = installFromArchive(tmp, stageArchive(tmp), "zcode");

    const integrationRoot = join(home, ".openviking", "agent-integrations", "zcode");
    assert.ok(existsSync(join(integrationRoot, "scripts", "hook.mjs")));
    assert.ok(existsSync(join(integrationRoot, "hosts", "zcode.mjs")));
    assert.ok(existsSync(join(integrationRoot, "hosts", "zcode-capture.mjs")));
    // An installation is for one client: the other hosts' configuration
    // directories are not copied with it.
    assert.equal(existsSync(join(integrationRoot, "hosts", "zcode", "hooks.json")), true);
    assert.equal(existsSync(join(integrationRoot, "hosts", "cursor")), false);
    // ZCode imports the runtime the installer assembles beside it, the way
    // cursor and trae do, rather than a copy committed into its own tree.
    const sharedRoot = join(home, ".openviking", "agent-integrations", "memory-plugin-shared", "lib");
    assert.ok(existsSync(join(sharedRoot, "async-writer.mjs")));
    assert.ok(existsSync(join(sharedRoot, "capture-utils.mjs")));
    assert.ok(existsSync(join(sharedRoot, "mcp-proxy-config.mjs")));
    assert.equal(existsSync(join(integrationRoot, "scripts", "shared")), false);

    const config = JSON.parse(readFileSync(join(home, ".zcode", "cli", "config.json"), "utf8"));
    assert.equal(config.hooks.enabled, true);
    assert.deepEqual(Object.keys(config.hooks.events), [
      "SessionStart",
      "UserPromptSubmit",
      "PreToolUse",
      "Stop",
    ]);
    assert.ok(config.mcp.servers.openviking);

    // The hook commands point across the plugin boundary at the runtime the
    // installer assembles from the archive, so an entry the manifest forgot to
    // carry is a hooks.json naming a script that is not there.
    const commands = JSON.stringify(config.hooks.events)
      .split(/"/u)
      .filter((part) => part.includes("# openviking-memory"));
    assert.ok(commands.length > 0, "no OpenViking hook commands were installed");
    for (const command of commands) {
      const script = /'([^']*\.mjs)'/u.exec(command)?.[1];
      assert.ok(script, `${command} names no script`);
      assert.ok(existsSync(script), `${script} is missing after install`);
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

// Kimi Code reaches the archive through the same plugin directory but reads a
// configuration format of its own, so the parts a JSON host never exercises —
// the TOML hook block, `mcp.json`, and the native `kimi.plugin.json` — have to
// survive the round trip through the release archive.
test("release marketplace archive supports a Kimi Code TOS install", () => {
  const tmp = mkdtempSync(join(tmpdir(), "openviking-kimicode-release-"));
  try {
    const home = installFromArchive(tmp, stageArchive(tmp), "kimicode");

    const integrationRoot = join(home, ".openviking", "agent-integrations", "kimicode");
    assert.ok(existsSync(join(integrationRoot, "scripts", "hook.mjs")));
    assert.ok(existsSync(join(integrationRoot, "hosts", "kimicode.mjs")));
    assert.ok(existsSync(join(integrationRoot, "hosts", "kimicode-turns.mjs")));
    assert.ok(existsSync(join(integrationRoot, "hosts", "kimicode-capture.mjs")));
    assert.equal(existsSync(join(integrationRoot, "hosts", "zcode")), false);
    // `/plugins install <path>` reads this manifest, and its commands are
    // relative to the integration root, so it only works if the archive carried
    // it here alongside the scripts it names.
    const native = JSON.parse(readFileSync(join(integrationRoot, "hosts", "kimicode", "kimi.plugin.json"), "utf8"));
    for (const entry of native.hooks) {
      const script = /node (\S+\.mjs)/u.exec(entry.command)?.[1];
      assert.ok(script, entry.command);
      assert.ok(existsSync(resolve(integrationRoot, "hosts", "kimicode", script)), `${script} is missing after install`);
    }
    assert.ok(
      existsSync(resolve(integrationRoot, "hosts", "kimicode", native.mcpServers.openviking.args[0])),
      "the native manifest names an MCP proxy the archive did not carry",
    );

    const config = readFileSync(join(home, ".kimi-code", "config.toml"), "utf8");
    assert.equal(config.split("# >>> openviking kimicode integration").length - 1, 1, config);
    const commands = [...config.matchAll(/^command = "(.*openviking-memory)"$/gmu)].map((match) => match[1]);
    assert.equal(commands.length, native.hooks.length, config);
    for (const command of commands) {
      const script = /'([^']*\.mjs)'/u.exec(command)?.[1];
      assert.ok(script, `${command} names no script`);
      assert.ok(existsSync(script), `${script} is missing after install`);
    }
    assert.ok(JSON.parse(readFileSync(join(home, ".kimi-code", "mcp.json"), "utf8")).mcpServers.openviking);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

// The archive's contents are derived from the plugins' manifests and the shared
// sync, so nothing here restates them. What this pins is that the derivation is
// wired up at all: an archive missing a copy only the generator produces has to
// fail the stage, not ship.
test("staging rejects an archive missing a generated shared copy", () => {
  const tmp = mkdtempSync(join(tmpdir(), "openviking-marketplace-check-"));
  try {
    const stage = join(tmp, "memory-plugin-marketplace");
    const staged = run("bash", [stageScript, stage]);
    assert.equal(staged.status, 0, `${staged.stdout}\n${staged.stderr}`);
    const stagedDirs = readdirSync(stage, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name);

    const generated = [
      join("opencode-plugin", "lib", "shared", "plugin-config.mjs"),
      join("pi-coding-agent-extension", "shared", "plugin-config.mjs"),
    ];
    for (const file of generated) {
      assert.ok(existsSync(join(stage, file)), `${file} is not in the staged tree`);
    }

    rmSync(join(stage, generated[0]));
    const rechecked = run("node", [archiveCheck, stage, ...stagedDirs]);
    assert.equal(rechecked.status, 1, `${rechecked.stdout}\n${rechecked.stderr}`);
    assert.match(rechecked.stderr, /opencode-plugin\/lib\/shared\/plugin-config\.mjs/);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});
