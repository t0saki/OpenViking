/**
 * captureRedact end to end: the knob, and every shared path a captured message
 * leaves by — the batch and serial POSTs, the single-message POST, the pending
 * queue on disk and its replay. The engine itself is covered by redact.test.mjs;
 * this file only asserts that no send path skips it and that the tier the
 * caller resolved is the one applied.
 */

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { addAgentMessage, addAgentMessages, loadAgentHookConfig } from "./lib/agent-hook-runtime.mjs";
import { sendSessionMessages } from "./lib/batch-send.mjs";
import { KNOB_BY_NAME, WORKSPACE_KNOB_MAP, pluginConfigKeys, resolveKnobs } from "./lib/config-schema.mjs";
import { enqueue, listPending, replayPending } from "./lib/pending-queue.mjs";
import { REDACT_TIERS } from "./lib/redact.mjs";

// Assembled at runtime so this file holds no contiguous token for a secret
// scanner to trip on.
const GITHUB_PAT = ["ghp", "_", "a1B2c3D4e5F6g7H8i9J0", "k1L2m3N4o5P6q7R8"].join("");
// Matches only generic-api-key (the strict tier): no provider prefix.
const GENERIC = ["Zq8", "xW2", "pL5", "mN7", "rT4", "vB9", "kH3", "jF6"].join("");
const REDACTED_RE = /\{\{ov_redacted:[a-z0-9-]+:[0-9a-f]{8}\}\}/;

const ENV_KEYS = ["OPENVIKING_PENDING_DIR", "OPENVIKING_STATE_DIR", "OPENVIKING_CAPTURE_REDACT"];

/** A private pending dir and state dir (the install key lives there) per test. */
function withScratch(fn) {
  return async () => {
    const dir = mkdtempSync(join(tmpdir(), "ov-redact-wiring-"));
    const saved = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
    process.env.OPENVIKING_PENDING_DIR = join(dir, "pending");
    process.env.OPENVIKING_STATE_DIR = join(dir, "state");
    delete process.env.OPENVIKING_CAPTURE_REDACT;
    try {
      return await fn(dir);
    } finally {
      for (const [key, value] of Object.entries(saved)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
      rmSync(dir, { recursive: true, force: true });
    }
  };
}

function toolPayload(secret = GITHUB_PAT) {
  return {
    role: "assistant",
    peer_id: "peer-1",
    parts: [
      { type: "text", text: `export GITHUB_TOKEN=${secret}` },
      {
        type: "tool",
        tool_id: "call_1",
        tool_name: "bash",
        tool_status: "completed",
        tool_input: { command: "cat .env" },
        tool_output: JSON.stringify({ stdout: `GITHUB_TOKEN=${secret}\nDEBUG=1\n` }),
      },
    ],
  };
}

function queuedFiles(dir) {
  const pending = join(dir, "pending");
  try {
    return readdirSync(pending).map((name) => readFileSync(join(pending, name), "utf8"));
  } catch {
    return [];
  }
}

function recorder(respond = () => ({ ok: true, status: 200, result: {} })) {
  const bodies = [];
  const fetchJSON = async (path, init = {}) => {
    if (init.body) bodies.push({ path, body: init.body });
    return respond(path, init);
  };
  return { bodies, fetchJSON };
}

const RETRYABLE = () => ({ ok: false, status: 503, error: { code: "UNAVAILABLE" } });

// ── the knob ────────────────────────────────────────────────────────────────

test("captureRedact is a user-level capture knob whose tiers are the engine's", () => {
  const knob = KNOB_BY_NAME.get("captureRedact");
  assert.ok(knob, "declared in KNOBS");
  assert.equal(knob.default, "standard");
  assert.equal(knob.env, "OPENVIKING_CAPTURE_REDACT");
  assert.equal(knob.capability, "capture");
  assert.deepEqual(knob.values, [...REDACT_TIERS], "the enum and the engine agree on the tiers");
  assert.equal(knob.workspace, undefined, "no workspace key");
  assert.ok(!Object.values(WORKSPACE_KNOB_MAP).includes("captureRedact"));
  assert.ok(pluginConfigKeys().has("captureRedact"), "the doctor knows the key");
});

test("captureRedact parses false-ish values as off and unknown tiers as the default", () => {
  const resolve = (env, data = null) => resolveKnobs({
    env,
    layers: data ? [{ name: "file", data }] : [],
  }).settings.captureRedact;

  assert.equal(resolve({}), "standard");
  for (const value of ["off", "OFF", "false", "False", "0", "no", " no "]) {
    assert.equal(resolve({ OPENVIKING_CAPTURE_REDACT: value }), "off", `env ${JSON.stringify(value)}`);
  }
  for (const value of [false, 0, "off", "false"]) {
    assert.equal(resolve({}, { captureRedact: value }), "off", `file ${JSON.stringify(value)}`);
  }
  assert.equal(resolve({ OPENVIKING_CAPTURE_REDACT: "strict" }), "strict");
  assert.equal(resolve({ OPENVIKING_CAPTURE_REDACT: "Strict" }), "strict");
  assert.equal(resolve({}, { captureRedact: true }), "standard");
  assert.equal(resolve({}, { captureRedact: "yes" }), "standard");
  // A tier this build does not know redacts at the default rather than keeping
  // a weaker value from a lower layer.
  assert.equal(resolve({ OPENVIKING_CAPTURE_REDACT: "paranoid" }, { captureRedact: "off" }), "standard");
  assert.equal(resolve({}, { captureRedact: "paranoid" }), "standard");
  // The environment outranks the file.
  assert.equal(resolve({ OPENVIKING_CAPTURE_REDACT: "off" }, { captureRedact: "strict" }), "off");
});

test("a repository's workspace file cannot turn captureRedact off", () => {
  const dir = mkdtempSync(join(tmpdir(), "ov-redact-workspace-"));
  const workspace = join(dir, "workspace");
  const keys = ["OPENVIKING_CLI_CONFIG_FILE", "OPENVIKING_CONFIG_FILE", "OPENVIKING_HOME", "OPENVIKING_CAPTURE_REDACT"];
  const saved = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
  try {
    mkdirSync(join(workspace, ".openviking"), { recursive: true });
    mkdirSync(join(workspace, ".git"), { recursive: true });
    writeFileSync(join(dir, "ovcli.conf"), JSON.stringify({ url: "http://127.0.0.1:1933", plugin: { captureRedact: "strict" } }));
    process.env.OPENVIKING_CLI_CONFIG_FILE = join(dir, "ovcli.conf");
    process.env.OPENVIKING_CONFIG_FILE = join(dir, "ov.conf");
    process.env.OPENVIKING_HOME = join(dir, "home");
    delete process.env.OPENVIKING_CAPTURE_REDACT;
    for (const file of ["config.json", "config.local.json"]) {
      writeFileSync(
        join(workspace, ".openviking", file),
        JSON.stringify({ version: 1, captureRedact: "off", capture: { redact: "off", captureRedact: "off" } }),
      );
    }
    assert.equal(loadAgentHookConfig("cursor", workspace).captureRedact, "strict");
    assert.equal(loadAgentHookConfig("cursor", dir).captureRedact, "strict");
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── batch-send ──────────────────────────────────────────────────────────────

const BATCH_SENDERS = {
  batch: (fetchJSON, payloads, opts) => sendSessionMessages(fetchJSON, "s-1", payloads, opts),
  "serial fallback": (fetchJSON, payloads, opts) => sendSessionMessages(
    (path, init) => (path.endsWith("/batch") ? { ok: false, status: 404 } : fetchJSON(path, init)),
    "s-1",
    payloads,
    opts,
  ),
  addAgentMessages: (fetchJSON, payloads, opts) => addAgentMessages(fetchJSON, "s-1", payloads, "peer-2", opts),
};

for (const [name, send] of Object.entries(BATCH_SENDERS)) {
  test(`${name} posts redacted copies at the caller's tier (standard by default)`, withScratch(async () => {
    const generic = { role: "user", content: `API_TOKEN=${GENERIC}` };
    const wire = async (redactTier) => {
      const { bodies, fetchJSON } = recorder();
      const payloads = [toolPayload(), generic];
      const before = JSON.stringify(payloads);
      const res = await send(fetchJSON, payloads, redactTier ? { redactTier } : {});
      assert.equal(res.sent ?? payloads.length, payloads.length);
      assert.equal(JSON.stringify(payloads), before, "the caller's objects are not mutated");
      return bodies.map(({ body }) => body).join("\n");
    };

    const standard = await wire();
    assert.ok(!standard.includes(GITHUB_PAT), "no plaintext on the wire");
    assert.match(standard, /export GITHUB_TOKEN=\{\{ov_redacted:github-pat:[0-9a-f]{8}\}\}"/);
    assert.ok(standard.includes('"tool_id":"call_1"') && standard.includes('"tool_name":"bash"'), "metadata is kept");
    assert.ok(standard.includes(GENERIC), "generic-api-key is not a standard rule");

    const strict = await wire("strict");
    assert.ok(!strict.includes(GITHUB_PAT) && !strict.includes(GENERIC));
    assert.match(strict, /API_TOKEN=\{\{ov_redacted:generic-api-key:/);

    const off = await wire("off");
    assert.ok(off.includes(GITHUB_PAT) && off.includes(GENERIC), "off sends the payloads as they are");
  }));
}

test("sendSessionMessages queues the redacted copy after a retryable failure", withScratch(async (dir) => {
  const { bodies, fetchJSON } = recorder(RETRYABLE);
  const res = await sendSessionMessages(fetchJSON, "s-1", [toolPayload(), toolPayload()], {
    enqueueOnRetryable: true,
  });
  assert.equal(res.queued, 2);
  assert.ok(bodies.every(({ body }) => !body.includes(GITHUB_PAT)));
  const files = queuedFiles(dir);
  assert.equal(files.length, 1, "identical redacted payloads dedupe to one entry");
  assert.ok(!files[0].includes(GITHUB_PAT), "no plaintext on disk");
  assert.match(files[0], REDACTED_RE);
}));

// ── agent-hook-runtime ──────────────────────────────────────────────────────

test("addAgentMessage redacts the POST and the entry it parks", withScratch(async (dir) => {
  const { bodies, fetchJSON } = recorder(RETRYABLE);
  const result = await addAgentMessage(fetchJSON, "s-1", toolPayload());
  assert.equal(result.pendingQueued, true);
  assert.ok(!bodies[0].body.includes(GITHUB_PAT));
  const files = queuedFiles(dir);
  assert.equal(files.length, 1);
  assert.ok(!files[0].includes(GITHUB_PAT));

  const off = recorder();
  await addAgentMessage(off.fetchJSON, "s-1", toolPayload(), { redactTier: "off" });
  assert.ok(off.bodies[0].body.includes(GITHUB_PAT));
}));

// ── pending queue ───────────────────────────────────────────────────────────

test("enqueue redacts an addMessage payload unless the caller already did", withScratch(async (dir) => {
  assert.equal((await enqueue("addMessage", "s-1", toolPayload())).ok, true);
  assert.ok(!queuedFiles(dir)[0].includes(GITHUB_PAT), "default tier is standard");

  rmSync(join(dir, "pending"), { recursive: true, force: true });
  await enqueue("addMessage", "s-1", { role: "user", content: `API_TOKEN=${GENERIC}` }, { redactTier: "strict" });
  assert.ok(!queuedFiles(dir)[0].includes(GENERIC));

  rmSync(join(dir, "pending"), { recursive: true, force: true });
  await enqueue("addMessage", "s-1", toolPayload(), { redactTier: "off" });
  assert.ok(queuedFiles(dir)[0].includes(GITHUB_PAT), "off writes it as it is");
}));

test("replayPending sends a plaintext entry from an older plugin redacted", withScratch(async () => {
  // `redacted: true` writes the payload as it is, standing in for an entry an
  // older plugin queued before redaction existed.
  await enqueue("addMessage", "s-1", toolPayload(), { redacted: true });
  assert.equal((await listPending()).length, 1);

  const { bodies, fetchJSON } = recorder();
  const result = await replayPending(fetchJSON, () => {});
  assert.equal(result.replayed, 1);
  assert.ok(!bodies[0].body.includes(GITHUB_PAT));
  assert.match(bodies[0].body, REDACTED_RE);

  await enqueue("addMessage", "s-2", toolPayload(), { redacted: true });
  const off = recorder();
  await replayPending(off.fetchJSON, () => {}, { redactTier: "off" });
  assert.ok(off.bodies[0].body.includes(GITHUB_PAT), "off replays the entry as it is");
}));

test("a redacted entry replays byte-for-byte: redaction is idempotent", withScratch(async (dir) => {
  const first = recorder(RETRYABLE);
  await sendSessionMessages(first.fetchJSON, "s-1", [toolPayload()], { enqueueOnRetryable: true });
  const queued = JSON.parse(queuedFiles(dir)[0]).payload;

  const replay = recorder();
  await replayPending(replay.fetchJSON, () => {});
  assert.deepEqual(JSON.parse(replay.bodies[0].body), queued);
}));
