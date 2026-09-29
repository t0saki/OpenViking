import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const installer = join(ROOT, "examples", "memory-plugin-shared", "install.sh");
const stageScript = join(ROOT, ".github", "scripts", "stage-memory-plugin-marketplace.sh");
const archiveCheck = join(ROOT, ".github", "scripts", "check-marketplace-archive.mjs");
const claudeMarketplaceScript = join(ROOT, ".github", "scripts", "generate-claude-marketplace-json.sh");

function run(command, args, options = {}) {
  return spawnSync(command, args, {
    cwd: ROOT,
    encoding: "utf8",
    ...options,
  });
}

function stageMarketplaceZip(tmp) {
  const stage = join(tmp, "memory-plugin-marketplace");
  const staged = run("bash", [stageScript, stage]);
  assert.equal(staged.status, 0, `${staged.stdout}\n${staged.stderr}`);
  const zip = join(tmp, "memory-plugin-marketplace.zip");
  const zipped = run("zip", ["-rq", zip, "memory-plugin-marketplace"], { cwd: tmp });
  assert.equal(zipped.status, 0, `${zipped.stdout}\n${zipped.stderr}`);
  return { stage, zip };
}

// A stand-in for the claude CLI: it records each invocation and keeps the one
// marketplace registration the installer reads back through `--json`.
function writeFakeClaude(bin) {
  writeFileSync(join(bin, "claude"), `#!/bin/sh
echo "$*" >> "$FAKE_CLAUDE_DIR/calls.log"
case "$*" in
  --version) echo "$FAKE_CLAUDE_VERSION (Claude Code)" ;;
  "plugin marketplace list --json") cat "$FAKE_CLAUDE_DIR/marketplaces.json" 2>/dev/null || echo "[]" ;;
  "plugin marketplace remove "*) rm -f "$FAKE_CLAUDE_DIR/marketplaces.json" ;;
  "plugin marketplace add "*)
    case "$4" in https://*) [ -z "$FAKE_CLAUDE_URL_FAILS" ] || exit 1 ;; esac
    printf '[{"name":"openviking","path":"%s"}]' "$4" > "$FAKE_CLAUDE_DIR/marketplaces.json" ;;
esac
exit 0
`, { mode: 0o755 });
}

test("Claude URL marketplace lists the release's plugin zip as an archive source", () => {
  const tmp = mkdtempSync(join(tmpdir(), "openviking-claude-marketplace-"));
  try {
    const { stage } = stageMarketplaceZip(tmp);
    const pluginZip = join(tmp, "openviking-memory-claude.zip");
    const zipped = run("zip", ["-rq", pluginZip, "claude-code-memory-plugin"], { cwd: stage });
    assert.equal(zipped.status, 0, `${zipped.stdout}\n${zipped.stderr}`);
    const out = join(tmp, "marketplace.json");
    const generated = run("bash", [claudeMarketplaceScript, "v9.9.9", "https://tos.example.invalid/", pluginZip, stage, out]);
    assert.equal(generated.status, 0, `${generated.stdout}\n${generated.stderr}`);

    const manifest = JSON.parse(readFileSync(out, "utf8"));
    const pluginJson = JSON.parse(readFileSync(join(stage, "claude-code-memory-plugin", ".claude-plugin", "plugin.json"), "utf8"));
    assert.equal(manifest.name, "openviking");
    assert.equal(manifest.plugins.length, 1);
    const [entry] = manifest.plugins;
    assert.equal(entry.name, "openviking-memory");
    // Claude Code only updates an installed plugin when this string changes.
    assert.equal(entry.version, pluginJson.version);
    assert.deepEqual(entry.source, {
      source: "archive",
      url: "https://tos.example.invalid/releases/v9.9.9/openviking-memory-claude.zip",
      sha256: createHash("sha256").update(readFileSync(pluginZip)).digest("hex"),
    });

    // Claude Code strips one wrapping directory, so the manifest must sit
    // exactly one level down.
    const listed = run("unzip", ["-Z1", pluginZip]);
    assert.equal(listed.status, 0, listed.stderr);
    assert.ok(listed.stdout.split("\n").includes("claude-code-memory-plugin/.claude-plugin/plugin.json"));
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test("TOS installs register Claude Code's URL marketplace when the CLI supports archive sources", () => {
  const tmp = mkdtempSync(join(tmpdir(), "openviking-claude-tos-"));
  try {
    const { zip } = stageMarketplaceZip(tmp);
    const bin = join(tmp, "bin");
    const fake = join(tmp, "fake-claude");
    mkdirSync(bin);
    mkdirSync(fake);
    writeFakeClaude(bin);
    const home = join(tmp, "home");
    mkdirSync(join(home, ".claude"), { recursive: true });
    const marketplaceUrl = "https://tos.example.invalid/plugins/claude/marketplace.json";
    const settingsPath = join(home, ".claude", "settings.json");
    const readSettings = () => JSON.parse(readFileSync(settingsPath, "utf8"));
    const calls = () => readFileSync(join(fake, "calls.log"), "utf8").split("\n");
    const install = (version, extraEnv = {}) => {
      rmSync(join(fake, "calls.log"), { force: true });
      const result = run("bash", [
        installer, "--harness", "claude", "--dist", "tos", "--lang", "en",
        "--url", "http://127.0.0.1:1933", "--api-key", "", "--no-statusline", "--yes",
      ], {
        env: {
          ...process.env,
          HOME: home,
          PATH: `${bin}:${process.env.PATH}`,
          OPENVIKING_HOME: join(home, ".openviking"),
          OPENVIKING_TOS_BASE: "https://tos.example.invalid",
          OPENVIKING_MARKETPLACE_ARCHIVE_URL: `file://${zip}`,
          OPENVIKING_SKIP_VERSION_CHECK: "1",
          FAKE_CLAUDE_DIR: fake,
          FAKE_CLAUDE_VERSION: version,
          ...extraEnv,
        },
      });
      assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
      return result;
    };

    // An install from the old archive channel is re-registered from the URL.
    const archiveDir = join(home, ".openviking", "memory-plugin-marketplace");
    writeFileSync(join(fake, "marketplaces.json"), JSON.stringify([{ name: "openviking", path: archiveDir }]));
    install("2.1.224");
    let log = calls();
    assert.ok(log.includes("plugin uninstall openviking-memory@openviking"), log.join("\n"));
    assert.ok(log.includes("plugin marketplace remove openviking"), log.join("\n"));
    assert.ok(log.includes(`plugin marketplace add ${marketplaceUrl}`), log.join("\n"));
    assert.ok(log.includes("plugin install openviking-memory@openviking"), log.join("\n"));
    assert.deepEqual(readSettings().extraKnownMarketplaces.openviking, {
      source: { source: "url", url: marketplaceUrl },
      autoUpdate: true,
    });

    // A re-run refreshes the same registration and leaves a user's choice to
    // turn auto-update off alone.
    const settings = readSettings();
    settings.extraKnownMarketplaces.openviking.autoUpdate = false;
    writeFileSync(settingsPath, JSON.stringify(settings));
    install("2.1.284");
    log = calls();
    assert.ok(log.includes("plugin marketplace update openviking"), log.join("\n"));
    assert.equal(log.some((line) => line.startsWith("plugin marketplace add")), false, log.join("\n"));
    assert.equal(readSettings().extraKnownMarketplaces.openviking.autoUpdate, false);

    // Unreachable URL marketplace: fall back to the unpacked archive.
    const failed = install("2.1.284", { FAKE_CLAUDE_URL_FAILS: "1", OPENVIKING_TOS_BASE: "https://other.example.invalid" });
    log = calls();
    assert.ok(log.includes(`plugin marketplace add ${archiveDir}`), log.join("\n"));
    assert.match(failed.stdout + failed.stderr, /falling back to the archive directory/);

    // Claude Code before the archive source keeps the local directory.
    rmSync(join(fake, "marketplaces.json"), { force: true });
    rmSync(settingsPath);
    const legacy = install("2.1.223");
    log = calls();
    assert.ok(log.includes(`plugin marketplace add ${archiveDir}`), log.join("\n"));
    assert.equal(log.some((line) => line.includes("https://")), false, log.join("\n"));
    assert.equal(existsSync(settingsPath) && readSettings().extraKnownMarketplaces !== undefined, false);
    assert.match(legacy.stdout, /Claude Code\n {4}Next: .*\n {4}Updates: re-run this installer\n/);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test("a Claude-format wrapper sharing Claude Code's config keeps the URL marketplace", () => {
  const tmp = mkdtempSync(join(tmpdir(), "openviking-claude-wrapper-"));
  try {
    const bin = join(tmp, "bin");
    const fake = join(tmp, "fake-claude");
    const home = join(tmp, "home");
    mkdirSync(bin);
    mkdirSync(fake);
    mkdirSync(home);
    writeFakeClaude(bin);
    writeFileSync(join(bin, "claude-wrap"), '#!/bin/sh\nexec claude "$@"\n', { mode: 0o755 });
    const marketplaceUrl = "https://tos.example.invalid/plugins/claude/marketplace.json";

    const result = run("bash", [
      installer, "--harness", "claude", "--claude-bin", "claude,claude-wrap", "--dist", "tos", "--lang", "en",
      "--url", "http://127.0.0.1:1933", "--api-key", "", "--no-statusline", "--yes",
    ], {
      env: {
        ...process.env,
        HOME: home,
        PATH: `${bin}:${process.env.PATH}`,
        OPENVIKING_HOME: join(home, ".openviking"),
        OPENVIKING_TOS_BASE: "https://tos.example.invalid",
        OPENVIKING_SKIP_VERSION_CHECK: "1",
        FAKE_CLAUDE_DIR: fake,
        FAKE_CLAUDE_VERSION: "2.1.284",
      },
    });
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    const log = readFileSync(join(fake, "calls.log"), "utf8").split("\n");
    assert.deepEqual(log.filter((line) => /^plugin (uninstall|marketplace (add|remove))/.test(line)), [
      `plugin marketplace add ${marketplaceUrl}`,
    ]);
    assert.equal(JSON.parse(readFileSync(join(fake, "marketplaces.json"), "utf8"))[0].path, marketplaceUrl);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test("release marketplace archive supports ZCode and pi TOS installs", () => {
  const tmp = mkdtempSync(join(tmpdir(), "openviking-zcode-release-"));
  try {
    const stage = join(tmp, "memory-plugin-marketplace");
    const staged = run("bash", [stageScript, stage]);
    assert.equal(staged.status, 0, `${staged.stdout}\n${staged.stderr}`);

    const bundled = readdirSync(stage, { recursive: true, encoding: "utf8" }).filter((entry) =>
      entry.split(sep).includes("node_modules"),
    );
    assert.deepEqual(bundled.slice(0, 3), [], "marketplace archive carries development dependencies");

    const zipped = run("zip", ["-rq", join(tmp, "memory-plugin-marketplace.zip"), "memory-plugin-marketplace"], {
      cwd: tmp,
    });
    assert.equal(zipped.status, 0, `${zipped.stdout}\n${zipped.stderr}`);

    const home = join(tmp, "home");
    mkdirSync(home, { recursive: true });
    const installed = run("bash", [
      installer,
      "--harness", "zcode",
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
        OPENVIKING_MARKETPLACE_ARCHIVE_URL: `file://${join(tmp, "memory-plugin-marketplace.zip")}`,
        OPENVIKING_SKIP_VERSION_CHECK: "1",
      },
    });
    assert.equal(installed.status, 0, `${installed.stdout}\n${installed.stderr}`);

    const integrationRoot = join(home, ".openviking", "agent-integrations", "zcode");
    assert.ok(existsSync(join(integrationRoot, "plugin.json")));
    assert.equal(existsSync(join(integrationRoot, ".claude-plugin")), false);
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

    const bin = join(tmp, "bin");
    mkdirSync(bin);
    writeFileSync(join(bin, "kimi"), "#!/bin/sh\nexit 0\n", { mode: 0o755 });
    const kimiInstalled = run("bash", [
      installer,
      "--harness", "kimicode",
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
        PATH: `${bin}:${process.env.PATH}`,
        OPENVIKING_HOME: join(home, ".openviking"),
        OPENVIKING_MARKETPLACE_ARCHIVE_URL: `file://${join(tmp, "memory-plugin-marketplace.zip")}`,
        OPENVIKING_SKIP_VERSION_CHECK: "1",
      },
    });
    assert.equal(kimiInstalled.status, 0, kimiInstalled.stdout + kimiInstalled.stderr);
    const kimiRoot = join(home, ".kimi-code", "plugins", "managed", "openviking-memory");
    assert.ok(existsSync(join(kimiRoot, "kimi.plugin.json")));
    assert.ok(existsSync(join(kimiRoot, "agent-integrations", "kimicode", "scripts", "hook.mjs")));
    assert.ok(existsSync(join(kimiRoot, "agent-integrations", "memory-plugin-shared", "lib", "agent-hook-runtime.mjs")));

    writeFileSync(join(bin, "pi"), "#!/bin/sh\nexit 0\n", { mode: 0o755 });
    const piArgs = [installer, "--harness", "pi", "--dist", "tos", "--source", "archive",
      "--lang", "en", "--url", "http://127.0.0.1:1933", "--api-key", "", "--yes"];
    const piEnv = { ...process.env, HOME: home, PATH: bin + ":" + process.env.PATH,
      OPENVIKING_HOME: join(home, ".openviking"),
      OPENVIKING_MARKETPLACE_ARCHIVE_URL: "file://" + join(tmp, "memory-plugin-marketplace.zip"), OPENVIKING_SKIP_VERSION_CHECK: "1" };
    const piInstalled = run("bash", piArgs, { env: piEnv });
    assert.equal(piInstalled.status, 0, piInstalled.stdout + piInstalled.stderr);
    const piRoot = join(home, ".pi", "agent", "extensions", "openviking");
    const imported = run("node", ["--input-type=module", "-e", 'await import("./lib/mcp-bridge.mjs"); await import("./tools.ts")'], { cwd: piRoot });
    assert.equal(imported.status, 0, imported.stdout + imported.stderr);
    assert.ok(existsSync(join(piRoot, "package-lock.json")));
    assert.equal(existsSync(join(piRoot, "shared", "mcp-proxy-core.mjs")), false);

    // Both an npm failure and a false-success npm must leave the old install usable.
    writeFileSync(join(piRoot, "installed-before-upgrade"), "keep");
    for (const exitCode of [1, 0]) {
      writeFileSync(join(bin, "npm"), "#!/bin/sh\nexit " + exitCode + "\n", { mode: 0o755 });
      const failed = run("bash", piArgs, { env: piEnv });
      assert.notEqual(failed.status, 0, failed.stdout + failed.stderr);
      assert.equal(readFileSync(join(piRoot, "installed-before-upgrade"), "utf8"), "keep");
      assert.equal(existsSync(piRoot + ".tmp"), false);
      assert.match(failed.stdout + failed.stderr, /existing extension was kept/);
    }
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

    const piPackage = JSON.parse(readFileSync(join(stage, "pi-coding-agent-extension", "package.json")));
    const lockPath = join(stage, "pi-coding-agent-extension", "package-lock.json");
    const lock = readFileSync(lockPath);
    assert.deepEqual(JSON.parse(lock).packages[""].dependencies, piPackage.dependencies);
    rmSync(lockPath);
    const missingLock = run("node", [archiveCheck, stage, ...stagedDirs]);
    assert.equal(missingLock.status, 1);
    assert.match(missingLock.stderr, /pi-coding-agent-extension\/package-lock.json/);
    writeFileSync(lockPath, lock);

    rmSync(join(stage, generated[0]));
    const rechecked = run("node", [archiveCheck, stage, ...stagedDirs]);
    assert.equal(rechecked.status, 1, `${rechecked.stdout}\n${rechecked.stderr}`);
    assert.match(rechecked.stderr, /opencode-plugin\/lib\/shared\/plugin-config\.mjs/);

    const restaged = run("bash", [stageScript, stage]);
    assert.equal(restaged.status, 0, `${restaged.stdout}\n${restaged.stderr}`);
    rmSync(join(stage, "agent-hook-plugin", "plugin.json"));
    const missingManifest = run("node", [archiveCheck, stage, ...stagedDirs]);
    assert.equal(missingManifest.status, 1, `${missingManifest.stdout}\n${missingManifest.stderr}`);
    assert.match(missingManifest.stderr, /agent-hook-plugin\/plugin\.json/);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});
