/**
 * The bridge and the REST client must be looking at the same server.
 *
 * `index.ts` builds two connections out of one `loadConfig()`: `OVClient` for
 * recall, sync, the profile and the session commit, and
 * `buildBridgeProxyConfig()` for the MCP tool surface. If those drift, a
 * session recalls from one deployment and lets the model write to another, or
 * the tools search a wider slice of the context database than recall does —
 * both silent. So every case here loads the configuration once and then
 * compares the proxy config against what `OVClient` actually puts on the wire,
 * captured through a stubbed `fetch`.
 *
 * The comparison is by VALUE, never by header name: the identity headers are
 * the proxy core's business (it builds all of them), and
 * `memory-plugin-shared/tests/one-header-builder.test.mjs` scans these files to
 * keep a second header builder from growing here.
 *
 * The environment handling mirrors `config.test.mjs`: a throwaway ovcli.conf, a
 * pinned credential source, and every variable that could reach the loader
 * saved and restored around the call.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { loadConfig } from "../config.ts";
import { OVClient } from "../client.ts";
import { buildBridgeProxyConfig } from "../lib/mcp-bridge-config.mjs";
import { toMcpProxyConfig } from "../shared/mcp-proxy-config.mjs";

const MANAGED_ENV = [
  "OPENVIKING_CREDENTIAL_SOURCE",
  "OPENVIKING_CLI_CONFIG_FILE",
  "OPENVIKING_CONFIG_FILE",
  "OPENVIKING_URL",
  "OPENVIKING_BASE_URL",
  "OPENVIKING_MCP_URL",
  "OPENVIKING_API_KEY",
  "OPENVIKING_BEARER_TOKEN",
  "OPENVIKING_ACCOUNT",
  "OPENVIKING_USER",
  "OPENVIKING_AUTH_MODE",
  "OPENVIKING_PEER_ID",
  "OPENVIKING_WORKSPACE_PEER",
  "OPENVIKING_RECALL_PEER_SCOPE",
  "OPENVIKING_DEBUG",
  "OPENVIKING_DEBUG_LOG",
  "OV_DEBUG_LOG",
  "OPENVIKING_TIMEOUT_MS",
  "OPENVIKING_EXTRA_HEADERS",
];

/**
 * Run `fn` with one freshly loaded configuration.
 *
 * `env` overrides the baseline (an env-pinned credential chain against a local
 * server); `plugin` becomes the `plugin.pi` section of a throwaway ovcli.conf.
 */
async function withConfig({ env = {}, plugin = {} } = {}, fn) {
  const dir = await mkdtemp(join(tmpdir(), "ov-pi-bridge-config-"));
  const saved = Object.fromEntries(MANAGED_ENV.map((name) => [name, process.env[name]]));
  for (const name of MANAGED_ENV) delete process.env[name];
  process.env.OPENVIKING_CREDENTIAL_SOURCE = "env";
  process.env.OPENVIKING_URL = "http://127.0.0.1:1933";
  process.env.OPENVIKING_API_KEY = "pi-parity-key";
  process.env.OPENVIKING_CLI_CONFIG_FILE = join(dir, "ovcli.conf");
  process.env.OPENVIKING_CONFIG_FILE = join(dir, "ov.conf");
  for (const [name, value] of Object.entries(env)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }

  try {
    await writeFile(join(dir, "ovcli.conf"), JSON.stringify({ plugin: { pi: plugin } }), "utf8");
    return await fn(loadConfig());
  } finally {
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
    await rm(dir, { recursive: true, force: true });
  }
}

/**
 * What `OVClient` sends for one request, as the server would see it.
 *
 * `health()` is the first call `start()` makes and the one every session makes,
 * so it is the honest sample of the REST side of the connection.
 */
async function restRequest(cfg) {
  const originalFetch = globalThis.fetch;
  const seen = [];
  globalThis.fetch = async (url, init = {}) => {
    seen.push({ url: String(url), headers: { ...(init.headers || {}) } });
    return new Response(JSON.stringify({ status: "ok", result: {} }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
  try {
    assert.equal(await new OVClient(cfg).health(), true);
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(seen.length, 1);
  return { ...seen[0], baseUrl: seen[0].url.replace(/\/health$/, "") };
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

test("the bridge's MCP URL is the REST client's base URL plus /mcp", async () => {
  // The trailing slash is the interesting part: the REST client trims it and
  // `buildMcpProxyConfig` trims it again, so neither side can end up with a
  // doubled separator that a strict reverse proxy would 404.
  await withConfig({ env: { OPENVIKING_URL: "http://127.0.0.1:1933/" } }, async (cfg) => {
    const rest = await restRequest(cfg);
    const bridge = buildBridgeProxyConfig(cfg);

    assert.equal(rest.url, "http://127.0.0.1:1933/health");
    assert.equal(bridge.mcpUrl, `${rest.baseUrl}/mcp`);
    assert.equal(bridge.mcpUrl, "http://127.0.0.1:1933/mcp");
  });
});

test("an explicitly pinned MCP URL is the one the bridge uses", async () => {
  // `OPENVIKING_MCP_URL` exists for deployments whose MCP endpoint does not sit
  // under the REST base (a gateway path, a separate host). The bridge has to
  // honour the URL the shared chain resolved rather than deriving its own, or
  // that pin would work for every other harness and silently not for pi.
  await withConfig({
    env: { OPENVIKING_MCP_URL: "http://127.0.0.1:1933/gateway/mcp" },
  }, async (cfg) => {
    const rest = await restRequest(cfg);
    const bridge = buildBridgeProxyConfig(cfg);

    assert.equal(cfg.mcpUrl, "http://127.0.0.1:1933/gateway/mcp");
    assert.equal(bridge.mcpUrl, cfg.mcpUrl);
    assert.equal(rest.baseUrl, "http://127.0.0.1:1933", "the REST base URL is untouched by the pin");
  });
});

// ---------------------------------------------------------------------------
// Credentials and identity
// ---------------------------------------------------------------------------

test("the bridge carries the same key, account and user as the REST client", async () => {
  await withConfig({
    env: {
      OPENVIKING_API_KEY: "pi-parity-key",
      OPENVIKING_ACCOUNT: "acct-parity",
      OPENVIKING_USER: "user-parity",
    },
  }, async (cfg) => {
    const rest = await restRequest(cfg);
    const bridge = buildBridgeProxyConfig(cfg);

    assert.equal(bridge.apiKey, "pi-parity-key");
    assert.equal(rest.headers.Authorization, `Bearer ${bridge.apiKey}`);

    // An account and a user with no auth mode named means a trusted server, so
    // both surfaces name the operator.
    assert.equal(cfg.sendIdentityHeaders, true);
    assert.equal(bridge.sendIdentityHeaders, cfg.sendIdentityHeaders);
    assert.equal(bridge.account, "acct-parity");
    assert.equal(bridge.user, "user-parity");
    assert.equal(rest.headers["X-OpenViking-Account"], bridge.account);
    assert.equal(rest.headers["X-OpenViking-User"], bridge.user);
  });
});

test("an api_key server keeps the operator's identity off both wires", async () => {
  // The key already carries the identity there, and the headers are ignored;
  // sending them anyway would name the operator to every proxy on the path. The
  // bridge has to inherit that decision instead of re-deciding it.
  await withConfig({
    env: {
      OPENVIKING_ACCOUNT: "acct-parity",
      OPENVIKING_USER: "user-parity",
      OPENVIKING_AUTH_MODE: "api_key",
    },
  }, async (cfg) => {
    const rest = await restRequest(cfg);
    const bridge = buildBridgeProxyConfig(cfg);

    assert.equal(cfg.sendIdentityHeaders, false);
    assert.equal(bridge.sendIdentityHeaders, false);
    // The values still travel in the config — the proxy core is what decides
    // to leave them off the request, exactly as the REST client does.
    assert.equal(bridge.account, "acct-parity");
    assert.equal(bridge.user, "user-parity");
    const sent = Object.values(rest.headers);
    assert.ok(!sent.includes("acct-parity"), "the account stays off the REST request");
    assert.ok(!sent.includes("user-parity"), "so does the user");
  });
});

// ---------------------------------------------------------------------------
// Peer
// ---------------------------------------------------------------------------

test("the peer the REST client resolves is the peer the bridge hands the proxy", async () => {
  // Asserted on the resolved value rather than the header it becomes: the proxy
  // core builds that header, and this file must not grow a second opinion about
  // its name.
  await withConfig({
    env: { OPENVIKING_RECALL_PEER_SCOPE: "all" },
    plugin: { peerId: "pi-parity-peer" },
  }, async (cfg) => {
    const rest = await restRequest(cfg);
    const bridge = buildBridgeProxyConfig(cfg);

    assert.equal(cfg.peerId, "pi-parity-peer");
    assert.equal(bridge.peerId, cfg.peerId);
    assert.ok(
      Object.values(rest.headers).includes(cfg.peerId),
      "the REST client already sends this peer on every request",
    );

    // Why `buildBridgeProxyConfig` passes the peer explicitly: the shared
    // default is written for a long-lived stdio proxy, which cannot trust its
    // launch directory and therefore sends a peer only under `actor` recall
    // scope. Taking that default here would widen the tools' default search to
    // every peer under the user while recall stays narrowed to this one.
    assert.equal(toMcpProxyConfig(cfg).peerId, "", "the stdio default drops it");
  });
});

// ---------------------------------------------------------------------------
// Debug logging
// ---------------------------------------------------------------------------

test("setting only a debug log path turns the proxy's records on", async () => {
  // The extension's own logger opens as soon as a path exists, while the proxy
  // core logs only when its config says `debug`. Without the widening, an
  // operator who configured a log file would get the extension's lines and none
  // of the bridge's — losing the handshake, the 401/403 responses and the
  // timeouts, which are the records worth having in that file.
  const path = join(tmpdir(), "ov-pi-bridge-config.log");
  await withConfig({ env: { OPENVIKING_DEBUG_LOG: path } }, (cfg) => {
    assert.equal(cfg.debug, false, "nothing asked for debug itself");
    assert.equal(cfg.debugLogPath, path);
    const bridge = buildBridgeProxyConfig(cfg);
    assert.equal(bridge.debug, true);
    assert.equal(bridge.debugLogPath, path);
    // The shared mapper on its own would leave the proxy silent.
    assert.equal(toMcpProxyConfig(cfg).debug, false);
  });
});

test("pi's older OV_DEBUG_LOG spelling reaches the proxy as well", async () => {
  const path = join(tmpdir(), "ov-pi-bridge-config-legacy.log");
  await withConfig({ env: { OV_DEBUG_LOG: path } }, (cfg) => {
    const bridge = buildBridgeProxyConfig(cfg);
    assert.equal(bridge.debug, true);
    assert.equal(bridge.debugLogPath, path, "the path loadConfig folded in, not a second one");
  });
});

test("no debug log and no debug switch leaves the proxy quiet", async () => {
  await withConfig({}, (cfg) => {
    const bridge = buildBridgeProxyConfig(cfg);
    assert.equal(cfg.debugLogPath, "");
    assert.equal(bridge.debug, false);
  });
});
