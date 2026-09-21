import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { readRequestBody, withMockOpenViking, writeJson } from "../../memory-plugin-shared/testing/support.mjs";

const scripts = dirname(fileURLToPath(import.meta.url));
function hook(name, input, root, url, extra = {}) {
  const env = { ...process.env };
  for (const key of Object.keys(env)) if (key.startsWith("OPENVIKING_") || key === "OV_HOOK_WORKER") delete env[key];
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [join(scripts, name + ".mjs")], { env: {
      ...env, HOME: root, TMPDIR: root, OPENVIKING_HOME: join(root, ".openviking"),
      OPENVIKING_URL: url, OPENVIKING_MEMORY_ENABLED: "1", OPENVIKING_AUTO_CAPTURE: "1",
      OPENVIKING_AUTO_RECALL: "1", OPENVIKING_WRITE_PATH_ASYNC: "0", OPENVIKING_NO_AUTO_INJECT: "1",
      OPENVIKING_RECALL_COMPRESS: "off", OPENVIKING_TIMEOUT_MS: "1000", OPENVIKING_CAPTURE_ASSISTANT_TURNS: "1",
      OPENVIKING_PENDING_DIR: join(root, "pending"), ...extra,
    } });
    let stdout = "", stderr = "";
    child.stdout.on("data", (c) => { stdout += c; });
    child.stderr.on("data", (c) => { stderr += c; });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? resolve(stdout) : reject(new Error(stderr)));
    child.stdin.end(JSON.stringify({ cwd: root, ...input }));
  });
}

test("separate Claude hooks share the pin across recall, capture, compact, resume and subagents", async () => {
  const root = await mkdtemp(join(tmpdir(), "ov-cc-identity-"));
  const id = randomUUID();
  const requests = [];
  let offline = false;
  const pinFile = join(root, ".openviking/state", "ov-session-" + id + ".json");
  try {
    const transcript = join(root, "transcript.jsonl");
    await writeFile(transcript, JSON.stringify({ role: "user", content: "Remember the session identity requirement" }) + "\n" + JSON.stringify({ role: "assistant", content: "The same session is used by all hooks" }));
    await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      const body = req.method === "POST" ? await readRequestBody(req) : null;
      requests.push({ path: url.pathname, method: req.method, body });
      if (offline) return writeJson(res, { status: "error", error: { code: "UNAVAILABLE" } }, 503);
      if (url.pathname === "/health") return writeJson(res, { status: "ok", result: {} });
      if (url.pathname === "/api/v1/search/search") return writeJson(res, { status: "ok", result: { context: "Pinned recall context", memories: [] } });
      if (url.pathname.endsWith("/context")) return writeJson(res, { status: "ok", result: { latest_archive_overview: "Archive from the pinned session" } });
      if (url.pathname.startsWith("/api/v1/sessions/")) return writeJson(res, { status: "ok", result: { message_count: 2, pending_tokens: 0 } });
      return writeJson(res, { status: "error" }, 404);
    }, async (url) => {
      await hook("session-start", { session_id: id, source: "startup" }, root, url);
      const pin = JSON.parse(await readFile(pinFile, "utf8"));
      assert.match(pin.ovSessionId, /^claude-\d{8}-\d{6}-[a-z0-9]{8}$/);
      await hook("auto-recall", { session_id: id, prompt: "Please recall the session identity requirement" }, root, url);
      await hook("auto-capture", { session_id: id, transcript_path: transcript }, root, url);
      await hook("pre-compact", { session_id: id }, root, url);
      const resumed = await hook("session-start", { session_id: id, source: "resume" }, root, url);
      assert.match(resumed, /Archive from the pinned session/);
      await hook("session-start", { session_id: id, source: "compact" }, root, url);
      await hook("subagent-start", { session_id: id, agent_id: "test-child" }, root, url);
      const sub = JSON.parse(await readFile(join(root, "openviking-cc-subagent-state/test-child.json"), "utf8"));
      assert.equal(sub.ovSessionId, pin.ovSessionId + "__subagent-test-child");
      await hook("subagent-stop", { session_id: id, agent_id: "test-child", agent_transcript_path: transcript }, root, url);
      await hook("session-end", { session_id: id }, root, url);
      const sessionRequests = requests.filter((r) => r.path.startsWith("/api/v1/sessions/"));
      assert.ok(sessionRequests.some((r) => r.path.endsWith("/messages/batch")));
      assert.ok(sessionRequests.some((r) => r.path.endsWith("/commit")));
      assert.ok(sessionRequests.every((r) => r.path.startsWith("/api/v1/sessions/" + pin.ovSessionId)));
      const recallRequests = requests.filter((r) => r.path === "/api/v1/search/search");
      assert.ok(recallRequests.length > 0);
      assert.ok(recallRequests.every((r) => r.body.session_id === pin.ovSessionId));
      const capture = JSON.parse(await readFile(join(root, ".openviking/state/last-capture.json"), "utf8"));
      assert.equal(capture.ov_session_id, pin.ovSessionId);
      assert.equal(JSON.parse(await readFile(pinFile, "utf8")).ovSessionId, pin.ovSessionId);

      // A fresh offline startup pins locally before health; queued capture and
      // commit then replay into that exact identity when health returns.
      offline = true;
      const other = randomUUID();
      await hook("session-start", { session_id: other, source: "startup" }, root, url);
      const offlinePin = JSON.parse(await readFile(join(root, ".openviking/state", "ov-session-" + other + ".json"), "utf8"));
      await hook("auto-capture", { session_id: other, transcript_path: transcript }, root, url);
      await hook("session-end", { session_id: other }, root, url);
      const pending = (await readdir(join(root, "pending"))).filter((file) => file.endsWith(".json"));
      assert.ok(pending.length > 0);
      for (const file of pending) assert.equal(JSON.parse(await readFile(join(root, "pending", file))).sessionId, offlinePin.ovSessionId);
      offline = false;
      const beforeReplay = requests.length;
      await hook("session-start", { session_id: other, source: "resume" }, root, url);
      assert.ok(requests.slice(beforeReplay).some((r) => r.method === "POST" && r.path.includes(offlinePin.ovSessionId)));
    });
  } finally { await rm(root, { recursive: true, force: true }); }
});
