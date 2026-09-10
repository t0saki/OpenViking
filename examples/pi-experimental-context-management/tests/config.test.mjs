import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";
import { loadConfig, loadConfigFromModuleUrl } from "../config.ts";

async function withConfigFile(body, fn, env = {}, cliConfig = null) {
  const dir = await mkdtemp(join(tmpdir(), "ov-pi-config-用户-"));
  const oldEnv = {
    OPENVIKING_URL: process.env.OPENVIKING_URL,
    OPENVIKING_API_KEY: process.env.OPENVIKING_API_KEY,
    OPENVIKING_ACCOUNT: process.env.OPENVIKING_ACCOUNT,
    OPENVIKING_USER: process.env.OPENVIKING_USER,
    OPENVIKING_PEER_ID: process.env.OPENVIKING_PEER_ID,
    OPENVIKING_WORKSPACE_PEER: process.env.OPENVIKING_WORKSPACE_PEER,
    OPENVIKING_RECALL_PEER_SCOPE: process.env.OPENVIKING_RECALL_PEER_SCOPE,
    OPENVIKING_CREDENTIAL_SOURCE: process.env.OPENVIKING_CREDENTIAL_SOURCE,
    OPENVIKING_CLI_CONFIG_FILE: process.env.OPENVIKING_CLI_CONFIG_FILE,
    OPENVIKING_CONFIG_FILE: process.env.OPENVIKING_CONFIG_FILE,
    OPENVIKING_DEBUG_LOG: process.env.OPENVIKING_DEBUG_LOG,
    OV_DEBUG_LOG: process.env.OV_DEBUG_LOG,
  };
  process.env.OPENVIKING_CREDENTIAL_SOURCE = "env";
  process.env.OPENVIKING_URL = "http://127.0.0.1:1933";
  process.env.OPENVIKING_CLI_CONFIG_FILE = join(dir, "ovcli.conf");
  process.env.OPENVIKING_CONFIG_FILE = join(dir, "ov.conf");
  delete process.env.OPENVIKING_API_KEY;
  delete process.env.OPENVIKING_ACCOUNT;
  delete process.env.OPENVIKING_USER;
  delete process.env.OPENVIKING_PEER_ID;
  delete process.env.OPENVIKING_WORKSPACE_PEER;
  delete process.env.OPENVIKING_RECALL_PEER_SCOPE;
  delete process.env.OPENVIKING_DEBUG_LOG;
  delete process.env.OV_DEBUG_LOG;
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }

  try {
    await writeFile(join(dir, "config.json"), JSON.stringify(body), "utf8");
    if (cliConfig !== null) {
      await writeFile(join(dir, "ovcli.conf"), JSON.stringify(cliConfig), "utf8");
    }
    return await fn(loadConfig(dir), dir);
  } finally {
    for (const [key, value] of Object.entries(oldEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    await rm(dir, { recursive: true, force: true });
  }
}

test("loadConfig defaults capture to faithful, tool results on", async () => {
  await withConfigFile({}, (cfg) => {
    assert.equal(cfg.faithfulCapture, true);
    assert.equal(cfg.captureToolResults, true);
    assert.equal(cfg.captureAssistantTurns, true);
    assert.equal(cfg.captureMaxLength, 24000);
    assert.equal(cfg.captureToolMaxChars, 1000000);
  });
});

test("loadConfig carries no takeover or recall-ledger keys", async () => {
  // The fork dropped both features; a stale key in an old config.json is
  // passed through untouched but never read, so the defaults must not name it.
  await withConfigFile({}, (cfg) => {
    for (const key of Object.keys(cfg)) {
      assert.ok(!key.startsWith("takeover"), `unexpected takeover key: ${key}`);
    }
    assert.equal(cfg.recallLedger, undefined);
  });
});

test("loadConfig lets config.json turn tool-result capture off", async () => {
  await withConfigFile({ captureToolResults: false }, (cfg) => {
    assert.equal(cfg.captureToolResults, false);
    // faithfulCapture is not a switch: the archive is the only way back to a
    // closed window.
    assert.equal(cfg.faithfulCapture, true);
  });
});

test("loadConfigFromModuleUrl decodes Unicode paths", async () => {
  await withConfigFile({ commitTokenThreshold: 2000 }, (_cfg, dir) => {
    const moduleUrl = pathToFileURL(join(dir, "index.ts")).href;
    const cfg = loadConfigFromModuleUrl(moduleUrl);
    assert.equal(cfg.commitTokenThreshold, 2000);
  });
});

test("loadConfig clamps invalid values", async () => {
  await withConfigFile({
    commitTokenThreshold: -1,
    commitKeepRecentCount: -5,
    captureMaxLength: 1,
    captureToolMaxChars: 10,
    recallLimit: 999,
    scoreThreshold: 5,
  }, (cfg) => {
    assert.equal(cfg.commitTokenThreshold, 1000);
    assert.equal(cfg.commitKeepRecentCount, 0);
    assert.equal(cfg.captureMaxLength, 200);
    assert.equal(cfg.captureToolMaxChars, 200);
    assert.equal(cfg.recallLimit, 50);
    assert.equal(cfg.scoreThreshold, 1);
  });
});

test("loadConfig derives workspace peer by default", async () => {
  // The default is now the repository, so the expectation follows whichever
  // template resolves where the suite runs — inside a checkout that is the
  // remote, and outside one it is still the old working-directory id.
  const { resolveEffectivePeerId } = await import("../shared/workspace-peer.mjs");
  const expected = resolveEffectivePeerId({ cfg: {}, cwd: process.cwd() });
  await withConfigFile({}, (cfg) => {
    assert.equal(cfg.peerId, expected.peerId);
    assert.equal(cfg.workspacePeer, true);
    assert.equal(cfg.recallPeerScope, "all");
  });
});

test("loadConfig prefers config peer over workspace derivation", async () => {
  await withConfigFile({
    peerId: " pi ",
    workspacePeer: true,
  }, (cfg) => {
    assert.equal(cfg.peerId, "pi");
  });
});

test("loadConfig keeps config peer when workspace derivation is disabled", async () => {
  await withConfigFile({
    peerId: "pi",
    workspacePeer: false,
  }, (cfg) => {
    assert.equal(cfg.peerId, "pi");
    assert.equal(cfg.workspacePeer, false);
  });
});

test("loadConfig gives environment peer precedence over config peer", async () => {
  await withConfigFile({
    peerId: "config-peer",
    recallPeerScope: "actor",
    workspacePeer: false,
  }, (cfg) => {
    assert.equal(cfg.peerId, "explicit-peer");
    assert.equal(cfg.workspacePeer, false);
    assert.equal(cfg.recallPeerScope, "actor");
  }, { OPENVIKING_PEER_ID: "explicit-peer" });
});

test("loadConfig leaves the debug log off when nothing asks for it", async () => {
  await withConfigFile({}, (cfg) => {
    assert.equal(cfg.debugLogPath, "");
  });
});

test("loadConfig reads the debug log path from OPENVIKING_DEBUG_LOG", async () => {
  await withConfigFile({}, (cfg) => {
    assert.equal(cfg.debugLogPath, "/tmp/ov-pi-shared.log");
  }, { OPENVIKING_DEBUG_LOG: "/tmp/ov-pi-shared.log" });
});

test("loadConfig still honours the deprecated OV_DEBUG_LOG", async () => {
  await withConfigFile({}, (cfg) => {
    assert.equal(cfg.debugLogPath, "/tmp/ov-pi-legacy.log");
  }, { OV_DEBUG_LOG: "/tmp/ov-pi-legacy.log" });
});

test("loadConfig prefers OPENVIKING_DEBUG_LOG over the deprecated alias", async () => {
  await withConfigFile({ debugLogPath: "/tmp/ov-pi-file.log" }, (cfg) => {
    assert.equal(cfg.debugLogPath, "/tmp/ov-pi-shared.log");
  }, {
    OPENVIKING_DEBUG_LOG: "/tmp/ov-pi-shared.log",
    OV_DEBUG_LOG: "/tmp/ov-pi-legacy.log",
  });
});

test("loadConfig falls back to the config file debug log path", async () => {
  await withConfigFile({ debugLogPath: " /tmp/ov-pi-file.log " }, (cfg) => {
    assert.equal(cfg.debugLogPath, "/tmp/ov-pi-file.log");
  });
});

test("loadConfig gives ovcli peer precedence over config peer", async () => {
  await withConfigFile({
    peerId: "config-peer",
  }, (cfg) => {
    assert.equal(cfg.peerId, "ovcli-peer");
  }, {
    OPENVIKING_CREDENTIAL_SOURCE: "cli",
    OPENVIKING_URL: undefined,
  }, {
    url: "http://127.0.0.1:1933",
    actor_peer_id: "ovcli-peer",
  });
});
