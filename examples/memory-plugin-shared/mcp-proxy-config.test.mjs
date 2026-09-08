import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  buildMcpProxyConfig,
  DEFAULT_PROXY_TIMEOUT_MS,
  defaultCredentialPaths,
  normalizeConfigPath,
  resolveMcpActorPeerId,
  trimSlash,
} from "./lib/mcp-proxy-config.mjs";
import { createOpenVikingMcpProxy } from "./lib/mcp-proxy-core.mjs";
import { ROOT } from "./sync.mjs";

const OVCLI = join(homedir(), ".openviking", "ovcli.conf");
const OV = join(homedir(), ".openviking", "ov.conf");

// Every stdio MCP entrypoint in the tree, discovered rather than listed so a
// new harness cannot ship a proxy that skips the shared shaping. The count is
// pinned because a renamed directory would otherwise empty the loop and turn
// the assertions below into a no-op. Stage 6b's thin-harness merge lowers it
// to 6.
const MCP_PROXY_COUNT = 8;

const MCP_PROXIES = [
  ...readdirSync(join(ROOT, "examples"), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => `examples/${entry.name}/servers/mcp-proxy.mjs`)
    .filter((rel) => existsSync(join(ROOT, rel))),
  "agent-plugins/servers/mcp-proxy.mjs",
].map((rel) => ({ rel, source: readFileSync(join(ROOT, rel), "utf-8") }));

test("a base URL becomes the /mcp endpoint exactly once", () => {
  assert.equal(buildMcpProxyConfig({ baseUrl: "http://x:1933" }).mcpUrl, "http://x:1933/mcp");
  assert.equal(buildMcpProxyConfig({ baseUrl: "http://x:1933/" }).mcpUrl, "http://x:1933/mcp");
  assert.equal(buildMcpProxyConfig({ baseUrl: "http://x:1933///" }).mcpUrl, "http://x:1933/mcp");
});

test("an explicit MCP URL wins over the derived one", () => {
  const cfg = buildMcpProxyConfig({ baseUrl: "http://x:1933", mcpUrl: "http://y/custom" });
  assert.equal(cfg.mcpUrl, "http://y/custom");
});

test("the timeout is clamped and defaulted", () => {
  assert.equal(buildMcpProxyConfig({}).timeoutMs, DEFAULT_PROXY_TIMEOUT_MS);
  assert.equal(buildMcpProxyConfig({ timeoutMs: 0 }).timeoutMs, DEFAULT_PROXY_TIMEOUT_MS);
  assert.equal(buildMcpProxyConfig({ timeoutMs: "nope" }).timeoutMs, DEFAULT_PROXY_TIMEOUT_MS);
  assert.equal(buildMcpProxyConfig({ timeoutMs: 5 }).timeoutMs, 1000);
  assert.equal(buildMcpProxyConfig({ timeoutMs: 30000 }).timeoutMs, 30000);
});

test("harness paths come first, shared defaults follow, duplicates collapse", () => {
  const cfg = buildMcpProxyConfig({
    watchedPaths: ["/a/config.json", "", null, OVCLI],
    env: {},
  });
  assert.deepEqual(cfg.watchedPaths, ["/a/config.json", OVCLI, OV]);
});

test("the two config-file env overrides are watched and tilde-expanded", () => {
  const cfg = buildMcpProxyConfig({
    env: { OPENVIKING_CLI_CONFIG_FILE: "~/custom/ovcli.conf" },
  });
  assert.equal(cfg.watchedPaths[0], join(homedir(), "custom", "ovcli.conf"));
  assert.ok(cfg.watchedPaths.includes(OV));
});

test("missing fields normalize to the shapes the proxy core expects", () => {
  const cfg = buildMcpProxyConfig({ env: {} });
  assert.equal(cfg.apiKey, "");
  assert.equal(cfg.account, "");
  assert.equal(cfg.user, "");
  assert.equal(cfg.peerId, "");
  assert.equal(cfg.userAgent, "");
  assert.equal(cfg.credentialSource, "auto");
  assert.equal(cfg.credentialPath, "");
  assert.equal(cfg.debug, false);
});

test("debug stays strictly boolean-true opt-in", () => {
  assert.equal(buildMcpProxyConfig({ debug: "true" }).debug, false);
  assert.equal(buildMcpProxyConfig({ debug: 1 }).debug, false);
  assert.equal(buildMcpProxyConfig({ debug: true }).debug, true);
});

test("broad MCP recall does not send an actor peer header", () => {
  assert.equal(resolveMcpActorPeerId({ peerId: "workspace-a", recallPeerScope: "all" }), "");
  assert.equal(resolveMcpActorPeerId({ peerId: "workspace-a" }), "");
});

test("actor-scoped MCP recall takes and trims an explicit peer", () => {
  assert.equal(
    resolveMcpActorPeerId({ peerId: " workspace-a ", recallPeerScope: "actor" }),
    "workspace-a",
  );
});

test("a missing actor peer warns and widens instead of taking the proxy down", () => {
  const warnings = [];
  assert.equal(
    resolveMcpActorPeerId({ recallPeerScope: "actor", onWarn: (m) => warnings.push(m) }),
    "",
    "no header means broad recall, which is the default anyway",
  );
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /OPENVIKING_PEER_ID/, "the warning says how to fix it");
});

test("path and URL helpers stay exported for entrypoints that need them", () => {
  assert.equal(trimSlash("http://x/"), "http://x");
  assert.equal(normalizeConfigPath(""), "");
  assert.equal(normalizeConfigPath("~"), homedir());
  assert.equal(normalizeConfigPath("~/a"), join(homedir(), "a"));
  assert.equal(defaultCredentialPaths({}).length, 2);
});

test("every MCP proxy shapes its config through the shared builder", () => {
  assert.equal(
    MCP_PROXIES.length,
    MCP_PROXY_COUNT,
    `expected ${MCP_PROXY_COUNT} proxies, found ${MCP_PROXIES.map((p) => p.rel).join(", ")}`,
  );
  for (const { rel, source } of MCP_PROXIES) {
    assert.match(
      source,
      /buildMcpProxyConfig\(/,
      `${rel} must shape its config through buildMcpProxyConfig`,
    );
  }
});

test("no MCP proxy derives its peer from the launch directory", () => {
  // A proxy is long-lived and may start anywhere, so unlike a hook it cannot
  // re-derive a peer per turn. Codex already had this rule; it holds for all.
  for (const { rel, source } of MCP_PROXIES) {
    assert.doesNotMatch(source, /resolveEffectivePeerId/, `${rel} must not derive a peer`);
    assert.doesNotMatch(source, /cwd:\s*process\.cwd\(\)/, `${rel} must not key a peer off its cwd`);
  }
});

test("the default peer scope puts no actor-peer header on the wire", async () => {
  const sent = [];
  const proxyFor = (harnessConfig) => createOpenVikingMcpProxy({
    stdout: { write(_line, cb) { if (cb) cb(); return true; } },
    fetchImpl: (_url, init) => {
      sent.push(init.headers);
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: "OK",
        headers: { get: () => null },
        text: async () => JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }),
      });
    },
    readConfig: () => buildMcpProxyConfig({
      mcpUrl: "http://127.0.0.1:1933/mcp",
      peerId: resolveMcpActorPeerId(harnessConfig),
    }),
    loggerFactory: () => ({ log() {}, logError() {} }),
  });
  const initialize = { jsonrpc: "2.0", id: 1, method: "initialize", params: {} };

  await proxyFor({ peerId: "workspace-a" }).handleMessage({ ...initialize });
  assert.equal(
    sent[0]["X-OpenViking-Actor-Peer"],
    undefined,
    "the default scope is broad recall, so pinning an actor would narrow it behind the user's back",
  );

  await proxyFor({ peerId: "workspace-a", recallPeerScope: "actor" }).handleMessage({ ...initialize });
  assert.equal(sent[1]["X-OpenViking-Actor-Peer"], "workspace-a");
});
