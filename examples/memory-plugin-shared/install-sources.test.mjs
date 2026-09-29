/**
 * Where the installer gets what it installs.
 *
 * Every file-based harness installs from one release bundle, downloaded once
 * per run and never through git. Claude Code on its URL marketplace needs no
 * bundle at all, and neither does its statusline, which resolves the installed
 * plugin when it runs.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const stageScript = join(ROOT, ".github", "scripts", "stage-memory-plugin-marketplace.sh");

// Real path: the installer's Node helpers do nothing when started through a
// symlinked directory, which the temp directory is on macOS.
const work = mkdtempSync(join(realpathSync(tmpdir()), "openviking-install-sources-"));
process.on("exit", () => rmSync(work, { recursive: true, force: true }));

// A copy away from the checkout runs the way `curl | bash` does: no sibling
// lib/, no dev mode, so everything has to come from the bundle.
const installer = join(work, "install.sh");
cpSync(join(ROOT, "examples", "memory-plugin-shared", "install.sh"), installer);

// The bundle sits where the default channel looks for it.
const tosBase = join(work, "tos");
mkdirSync(join(tosBase, "releases", "latest"), { recursive: true });
const staged = spawnSync("bash", [stageScript, join(work, "memory-plugin-marketplace")], { encoding: "utf8" });
assert.equal(staged.status, 0, `${staged.stdout}\n${staged.stderr}`);
const zipped = spawnSync("zip", ["-rq", join(tosBase, "releases", "latest", "memory-plugin-marketplace.zip"), "memory-plugin-marketplace"], {
  cwd: work,
  encoding: "utf8",
});
assert.equal(zipped.status, 0, `${zipped.stdout}\n${zipped.stderr}`);

function writeExecutable(file, body) {
  writeFileSync(file, body, { mode: 0o755 });
}

function run(home, bin, args, extraEnv = {}) {
  return spawnSync("bash", [installer, ...args], {
    cwd: home,
    env: {
      ...process.env,
      HOME: home,
      PATH: `${bin}:${process.env.PATH}`,
      OPENVIKING_HOME: join(home, ".openviking"),
      OPENVIKING_TOS_BASE: `file://${tosBase}`,
      OPENVIKING_MARKETPLACE_ARCHIVE_URL: "",
      ...extraEnv,
    },
    encoding: "utf8",
  });
}

test("a default install of cursor, zcode and opencode downloads the bundle once and never runs git", () => {
  const home = mkdtempSync(join(work, "home-"));
  const bin = join(home, "bin");
  const log = join(home, "calls.log");
  mkdirSync(bin);
  const realCurl = spawnSync("bash", ["-c", "command -v curl"], { encoding: "utf8" }).stdout.trim();
  writeExecutable(join(bin, "git"), `#!/bin/sh\necho "git $*" >> "${log}"\nexit 1\n`);
  writeExecutable(join(bin, "curl"), `#!/bin/sh\necho "curl $*" >> "${log}"\nexec "${realCurl}" "$@"\n`);
  writeExecutable(join(bin, "opencode"), "#!/bin/sh\nexit 0\n");

  const result = run(home, bin, [
    "--harness", "cursor,zcode,opencode", "--lang", "en",
    "--url", "http://127.0.0.1:1933", "--api-key", "", "--yes",
  ]);
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);

  const calls = existsSync(log) ? readFileSync(log, "utf8").split("\n").filter(Boolean) : [];
  assert.deepEqual(calls.filter((line) => line.startsWith("git ")), []);
  assert.equal(calls.filter((line) => line.includes("memory-plugin-marketplace.zip")).length, 1, calls.join("\n"));
  assert.ok(existsSync(join(home, ".openviking", "agent-integrations", "cursor", "scripts", "hook.mjs")));
  assert.ok(existsSync(join(home, ".openviking", "agent-integrations", "zcode", "scripts", "hook.mjs")));
  assert.ok(existsSync(join(home, ".config", "opencode", "plugins", "openviking", "index.mjs")));
});

// It records its calls and keeps the one marketplace registration the
// installer reads back; `--version` answers with the URL-marketplace release.
function writeFakeClaude(bin, dir) {
  writeExecutable(join(bin, "claude"), `#!/bin/sh
echo "$*" >> "${dir}/calls.log"
case "$*" in
  --version) echo "2.1.224 (Claude Code)" ;;
  "plugin marketplace list --json") cat "${dir}/marketplaces.json" 2>/dev/null || echo "[]" ;;
  "plugin marketplace add "*) printf '[{"name":"openviking","path":"%s"}]' "$4" > "${dir}/marketplaces.json" ;;
esac
exit 0
`);
}

test("the statusline follows the installed plugin and replaces only its own earlier command", () => {
  const home = mkdtempSync(join(work, "home-"));
  const bin = join(home, "bin");
  const fake = join(home, "fake-claude");
  mkdirSync(bin);
  mkdirSync(fake);
  writeFakeClaude(bin, fake);
  const settingsPath = join(home, ".claude", "settings.json");
  const statusLine = () => JSON.parse(readFileSync(settingsPath, "utf8")).statusLine?.command;
  const setStatusLine = (command) => {
    mkdirSync(dirname(settingsPath), { recursive: true });
    writeFileSync(settingsPath, JSON.stringify({ statusLine: { type: "command", command } }));
  };
  const install = (...extra) => {
    const result = run(home, bin, [
      "--harness", "claude", "--dist", "tos", "--lang", "en",
      "--url", "http://127.0.0.1:1933", "--api-key", "", "--yes", ...extra,
    ], { OPENVIKING_TOS_BASE: "https://tos.example.invalid" });
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    return result;
  };

  // A command an earlier installer wrote into a copy under ~/.openviking is
  // repointed on a plain re-run, with no prompt and no bundle download.
  setStatusLine(`node "${home}/.openviking/openviking-repo/examples/claude-code-memory-plugin/scripts/statusline.mjs"`);
  install();
  const command = statusLine();
  assert.match(command, /installed_plugins\.json/);
  assert.equal(existsSync(join(home, ".openviking", "memory-plugin-marketplace")), false);

  // The command runs whatever copy the registry names, so it follows updates.
  const configDir = join(home, "claude-config");
  const runStatusLine = () => spawnSync("sh", ["-c", command], {
    env: { ...process.env, HOME: home, CLAUDE_CONFIG_DIR: configDir },
    input: "{}",
    encoding: "utf8",
  });
  const missing = runStatusLine();
  assert.equal(missing.status, 0, missing.stderr);
  assert.equal(missing.stdout, "");
  for (const version of ["1.0.0", "1.0.1"]) {
    const pluginRoot = join(configDir, "plugins", "cache", "openviking", "openviking-memory", version);
    mkdirSync(join(pluginRoot, "scripts"), { recursive: true });
    writeFileSync(join(pluginRoot, "scripts", "statusline.mjs"), `process.stdout.write("statusline ${version}");\n`);
    writeFileSync(join(configDir, "plugins", "installed_plugins.json"), JSON.stringify({
      version: 2,
      plugins: { "openviking-memory@openviking": [{ scope: "user", installPath: pluginRoot, version }] },
    }));
    const shown = runStatusLine();
    assert.equal(shown.status, 0, shown.stderr);
    assert.equal(shown.stdout, `statusline ${version}`);
  }

  // Someone else's statusline stays unless the user asks for ours, and so does
  // one that only wraps ours.
  setStatusLine("my-statusline");
  install();
  assert.equal(statusLine(), "my-statusline");
  const wrapped = `sh -c 'node "${clone}/examples/claude-code-memory-plugin/scripts/statusline.mjs"; echo " | main"'`;
  setStatusLine(wrapped);
  install();
  assert.equal(statusLine(), wrapped);
  install("--statusline");
  assert.equal(statusLine(), command);
});
