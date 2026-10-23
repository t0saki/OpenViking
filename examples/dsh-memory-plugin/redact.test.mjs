import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, test } from "node:test";

import { OpenVikingClient } from "./client.mjs";
import { resolveConfig } from "./config.mjs";
import { OpenVikingRuntime } from "./runtime.mjs";
import { enqueue } from "./shared/pending-queue.mjs";

// Assembled at runtime so this file holds no contiguous token for a scanner.
const GITHUB_PAT = ["ghp", "_", "a1B2c3D4e5F6g7H8i9J0", "k1L2m3N4o5P6q7R8"].join("");
const GENERIC = ["Zq8", "xW2", "pL5", "mN7", "rT4", "vB9", "kH3", "jF6"].join("");

const ENV_KEYS = ["OPENVIKING_PENDING_DIR", "OPENVIKING_STATE_DIR", "OPENVIKING_CAPTURE_REDACT"];
const originalFetch = globalThis.fetch;
let saved;
let dir;

beforeEach(async () => {
  saved = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
  dir = await mkdtemp(join(tmpdir(), "dsh-redact-"));
  process.env.OPENVIKING_PENDING_DIR = join(dir, "pending");
  process.env.OPENVIKING_STATE_DIR = join(dir, "state");
  delete process.env.OPENVIKING_CAPTURE_REDACT;
});

afterEach(async () => {
  globalThis.fetch = originalFetch;
  for (const [key, value] of Object.entries(saved)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  await rm(dir, { recursive: true, force: true });
});

function clientConfig(extra = {}) {
  return {
    endpoint: "http://127.0.0.1:1933",
    apiKey: "",
    peerId: "",
    requestTimeoutMs: 1000,
    ...extra,
  };
}

function recordFetch() {
  const bodies = [];
  globalThis.fetch = async (_url, init) => {
    if (init?.body) bodies.push(String(init.body));
    return new Response(JSON.stringify({ status: "ok", result: {} }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  return bodies;
}

async function queued() {
  const pending = join(dir, "pending");
  const names = await readdir(pending).catch(() => []);
  return Promise.all(names.map((name) => readFile(join(pending, name), "utf8")));
}

function runtimeConfig(extra = {}) {
  return {
    explicitPeerId: "",
    workspacePeer: false,
    peerId: "",
    syncTurns: true,
    captureAssistantTurns: true,
    captureToolResults: false,
    captureToolMaxChars: 1000000,
    captureMaxLength: 24000,
    captureMode: "semantic",
    commitKeepRecentCount: 10,
    ...extra,
  };
}

function userEvent(text) {
  return {
    type: "user/message",
    data: { role: "user", content: [{ type: "text", text }], source: { kind: "user" } },
  };
}

test("the client redacts before it posts, at its configured tier or an explicit one", async () => {
  const bodies = recordFetch();
  const payload = { role: "user", content: `token ${GITHUB_PAT}` };
  const line = { role: "user", content: `API_TOKEN=${GENERIC}` };
  await new OpenVikingClient(clientConfig({ captureRedact: "standard" })).addMessage("dsh-1", payload);
  await new OpenVikingClient(clientConfig({ captureRedact: "standard" })).addMessage("dsh-1", line);
  await new OpenVikingClient(clientConfig({ captureRedact: "strict" })).addMessage("dsh-1", line);
  await new OpenVikingClient(clientConfig({ captureRedact: "strict" })).addMessage("dsh-1", line, "", { redactTier: "off" });
  await new OpenVikingClient(clientConfig({ captureRedact: "off" })).addMessage("dsh-1", payload);
  await new OpenVikingClient(clientConfig()).addMessage("dsh-1", payload);
  assert.equal(payload.content, `token ${GITHUB_PAT}`, "the caller's payload is not mutated");
  assert.match(JSON.parse(bodies[0]).content, /^token \{\{ov_redacted:github-pat:[0-9a-f]{8}\}\}$/);
  assert.ok(bodies[1].includes(GENERIC), "standard leaves a generic key=value alone");
  assert.ok(!bodies[2].includes(GENERIC), "strict redacts it");
  assert.ok(bodies[3].includes(GENERIC), "an explicit tier wins over the client's");
  assert.ok(bodies[4].includes(GITHUB_PAT), "off sends the payload as it is");
  assert.ok(bodies[5].includes(GITHUB_PAT), "an unset tier is off");
});

test("resolveConfig reads captureRedact from the host input and the environment", () => {
  const env = { OPENVIKING_CLI_CONFIG_FILE: join(dir, "none.conf"), OPENVIKING_CONFIG_FILE: join(dir, "none.json"), OPENVIKING_HOME: join(dir, "home") };
  assert.equal(resolveConfig({}, env, dir).captureRedact, "off");
  assert.equal(resolveConfig({ captureRedact: "strict" }, env, dir).captureRedact, "strict");
  assert.equal(resolveConfig({ captureRedact: false }, env, dir).captureRedact, "off");
  assert.equal(resolveConfig({ captureRedact: "strict" }, { ...env, OPENVIKING_CAPTURE_REDACT: "0" }, dir).captureRedact, "off");
});

test("a captured turn is redacted on the direct POST and in the pending queue", async () => {
  const posted = [];
  const runtime = new OpenVikingRuntime({
    async addMessage(_sessionId, payload, _peer, options) {
      posted.push({ payload, options });
      return { ok: false, status: 503, error: { code: "UNAVAILABLE" } };
    },
  }, runtimeConfig({ captureRedact: "standard" }), { debug() {} });
  const session = { id: "redact-capture", header: { cwd: dir } };
  runtime.stateFor(session).ready = true;

  runtime.capture(session, userEvent(`my token is ${GITHUB_PAT}`));
  await runtime.flush(session);

  assert.equal(posted.length, 1);
  assert.ok(!JSON.stringify(posted[0].payload).includes(GITHUB_PAT));
  assert.equal(posted[0].options?.redacted, true, "the client is told not to scan it again");
  const files = await queued();
  assert.equal(files.length, 1);
  assert.ok(!files[0].includes(GITHUB_PAT));
});

test("captureRedact off leaves the captured turn as it is", async () => {
  const posted = [];
  const runtime = new OpenVikingRuntime({
    async addMessage(_sessionId, payload) {
      posted.push(payload);
      return { ok: false, status: 503, error: { code: "UNAVAILABLE" } };
    },
  }, runtimeConfig({ captureRedact: "off" }), { debug() {} });
  const session = { id: "redact-off", header: { cwd: dir } };
  runtime.stateFor(session).ready = true;

  runtime.capture(session, userEvent(`my token is ${GITHUB_PAT}`));
  await runtime.flush(session);

  assert.ok(JSON.stringify(posted[0]).includes(GITHUB_PAT));
  assert.ok((await queued())[0].includes(GITHUB_PAT));
});

test("the runtime's replay redacts an entry queued before redaction existed", async () => {
  await enqueue("addMessage", "dsh-old", { role: "user", content: `token ${GITHUB_PAT}` }, { redacted: true });
  const bodies = [];
  const runtime = new OpenVikingRuntime({
    async fetchJSON(_path, init) {
      if (init?.body) bodies.push(String(init.body));
      return { ok: true, status: 200, result: {} };
    },
  }, runtimeConfig({ captureRedact: "standard" }), { debug() {} });
  await runtime.replayPendingQueue();
  assert.equal(bodies.length, 1);
  assert.ok(!bodies[0].includes(GITHUB_PAT));
});
