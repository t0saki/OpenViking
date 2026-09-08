import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { loadConfig } from "./config.mjs";
import { isBypassed } from "./shared/session-model.mjs";

const OVERRIDES = [
  "OPENVIKING_CONFIG_FILE",
  "OPENVIKING_CLI_CONFIG_FILE",
  "OPENVIKING_HOME",
  "OPENVIKING_STATE_DIR",
  "OPENVIKING_CREDENTIAL_SOURCE",
  "OPENVIKING_CREDENTIALS_SOURCE",
  "OPENVIKING_PEER_ID",
  "OPENVIKING_PEER_SOURCE",
  "OPENVIKING_AUTO_CAPTURE",
  "OPENVIKING_BYPASS_SESSION",
  "OPENVIKING_BYPASS_SESSION_PATTERNS",
  "OPENVIKING_URL",
  "OPENVIKING_BASE_URL",
  "OPENVIKING_API_KEY",
  "OPENVIKING_BEARER_TOKEN",
];

/**
 * Run loadConfig() against a throwaway ~/.openviking pair plus a workspace
 * directory holding `.openviking/config.json`. The bare `.git` is what makes
 * that directory a workspace root; `other/` is a second directory with neither.
 */
function withConfigs({ ov, cli, workspace, env = {} }, fn) {
  const dir = mkdtempSync(join(tmpdir(), "cx-config-"));
  const ovPath = join(dir, "ov.conf");
  const cliPath = join(dir, "ovcli.conf");
  const workspaceDir = join(dir, "workspace");
  const otherDir = join(dir, "other");
  const saved = Object.fromEntries(OVERRIDES.map((key) => [key, process.env[key]]));
  try {
    for (const key of OVERRIDES) delete process.env[key];
    if (ov) writeFileSync(ovPath, JSON.stringify(ov));
    if (cli) writeFileSync(cliPath, JSON.stringify(cli));
    mkdirSync(join(workspaceDir, ".openviking"), { recursive: true });
    mkdirSync(join(workspaceDir, ".git"), { recursive: true });
    mkdirSync(otherDir, { recursive: true });
    if (workspace) {
      writeFileSync(join(workspaceDir, ".openviking", "config.json"), JSON.stringify(workspace));
    }
    process.env.OPENVIKING_CONFIG_FILE = ovPath;
    process.env.OPENVIKING_CLI_CONFIG_FILE = cliPath;
    // Keeps the identity cache and the workspace registry out of the real home.
    process.env.OPENVIKING_HOME = join(dir, "home");
    Object.assign(process.env, env);
    return fn({ ovPath, cliPath, workspaceDir, otherDir });
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    rmSync(dir, { recursive: true, force: true });
  }
}

test("a workspace file's peer.id is the peer this directory writes under", () => {
  withConfigs({
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli", actor_peer_id: "cli-peer" },
    workspace: { version: 1, peer: { id: "team-a" } },
  }, ({ workspaceDir, otherDir }) => {
    assert.equal(loadConfig(workspaceDir).peerId, "team-a");
    // Same process, another directory: the ovcli.conf peer is untouched.
    assert.equal(loadConfig(otherDir).peerId, "cli-peer");
  });
});

test("OPENVIKING_PEER_ID still outranks a workspace peer.id", () => {
  withConfigs({
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli", actor_peer_id: "cli-peer" },
    workspace: { version: 1, peer: { id: "team-a" } },
    env: { OPENVIKING_PEER_ID: "env-peer" },
  }, ({ workspaceDir }) => {
    assert.equal(loadConfig(workspaceDir).peerId, "env-peer");
  });
});

test("a credential source pinned to ovcli.conf ignores the env peer", () => {
  withConfigs({
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli", actor_peer_id: "cli-peer" },
    env: { OPENVIKING_PEER_ID: "env-peer", OPENVIKING_CREDENTIAL_SOURCE: "cli" },
  }, ({ otherDir }) => {
    assert.equal(loadConfig(otherDir).peerId, "cli-peer");
  });
});

test("an ovcli.conf plugin.codex.peerId outranks the file's actor peer", () => {
  withConfigs({
    cli: {
      url: "http://127.0.0.1:1933",
      api_key: "sk-cli",
      actor_peer_id: "cli-peer",
      plugin: { codex: { peerId: "plugin-peer" } },
    },
  }, ({ otherDir }) => {
    assert.equal(loadConfig(otherDir).peerId, "plugin-peer");
  });
});

test("ov.conf's codex.peerId keeps its place behind ovcli.conf", () => {
  withConfigs({
    ov: { codex: { peerId: "ov-peer" } },
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli", actor_peer_id: "cli-peer" },
  }, ({ otherDir }) => {
    assert.equal(loadConfig(otherDir).peerId, "cli-peer");
  });
});

test("the workspace layer follows the cwd it is handed, not the process's", () => {
  withConfigs({
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli" },
    workspace: { version: 1, capture: { enabled: false }, recall: { max_items: 3 } },
  }, ({ workspaceDir, otherDir }) => {
    const inside = loadConfig(workspaceDir);
    assert.equal(inside.autoCapture, false);
    assert.equal(inside.recallLimit, 3);

    const outside = loadConfig(otherDir);
    assert.equal(outside.autoCapture, true);
    assert.equal(outside.recallLimit, 10);
  });
});

test("an omitted cwd falls back to this process's directory", () => {
  withConfigs({
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli" },
    workspace: { version: 1, capture: { enabled: false } },
  }, ({ workspaceDir }) => {
    const origin = process.cwd();
    try {
      process.chdir(workspaceDir);
      assert.equal(loadConfig().autoCapture, false);
      assert.equal(loadConfig("").autoCapture, false);
    } finally {
      process.chdir(origin);
    }
  });
});

test("a repository can carry the bypass patterns its contributors share", () => {
  withConfigs({
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli" },
    workspace: { version: 1, bypass: { session_patterns: ["**/workspace"] } },
  }, ({ workspaceDir, otherDir }) => {
    const inside = loadConfig(workspaceDir);
    assert.deepEqual(inside.bypassSessionPatterns, ["**/workspace"]);
    assert.equal(isBypassed(inside, { cwd: workspaceDir }), true);

    const outside = loadConfig(otherDir);
    assert.deepEqual(outside.bypassSessionPatterns, []);
    assert.equal(isBypassed(outside, { cwd: otherDir }), false);
  });
});

test("the bypass env vars override the file, matching every other harness", () => {
  withConfigs({
    ov: { codex: { bypassSessionPatterns: ["/from/ov-conf"] } },
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli" },
    env: {
      OPENVIKING_BYPASS_SESSION_PATTERNS: "/from/env, /also/env ,",
      OPENVIKING_BYPASS_SESSION: "1",
    },
  }, ({ otherDir }) => {
    const cfg = loadConfig(otherDir);
    assert.deepEqual(cfg.bypassSessionPatterns, ["/from/env", "/also/env"]);
    assert.equal(cfg.bypassSession, true);
    assert.equal(isBypassed(cfg, { cwd: "/anywhere" }), true, "the switch wins regardless of cwd");
  });
});

test("ov.conf's codex section still supplies the patterns when no env var does", () => {
  withConfigs({
    ov: { codex: { bypassSessionPatterns: ["/scratch/*", 7, "  "] } },
    cli: { url: "http://127.0.0.1:1933", api_key: "sk-cli" },
  }, ({ otherDir }) => {
    const cfg = loadConfig(otherDir);
    assert.deepEqual(cfg.bypassSessionPatterns, ["/scratch/*"]);
    assert.equal(cfg.bypassSession, false);
  });
});

// ovcli.conf is the client's own file, so a key set there outranks the server
// config underneath it — the same order every other harness follows.
test("ovcli.conf plugin.codex.apiKey outranks ov.conf, and ovcli.conf's own api_key outranks it", () => {
  withConfigs({
    cli: { plugin: { codex: { apiKey: "sk-plugin" } } },
  }, ({ otherDir }) => {
    assert.equal(loadConfig(otherDir).apiKey, "sk-plugin", "nothing else supplies a key");
  });

  withConfigs({
    ov: { codex: { apiKey: "sk-codex" }, server: { root_api_key: "sk-root" } },
    cli: { plugin: { codex: { apiKey: "sk-plugin" } } },
  }, ({ otherDir }) => {
    assert.equal(loadConfig(otherDir).apiKey, "sk-plugin");
  });

  withConfigs({
    ov: { server: { root_api_key: "sk-root" } },
    cli: { api_key: "sk-cli", plugin: { codex: { apiKey: "sk-plugin" } } },
  }, ({ otherDir }) => {
    assert.equal(loadConfig(otherDir).apiKey, "sk-cli");
  });
});
