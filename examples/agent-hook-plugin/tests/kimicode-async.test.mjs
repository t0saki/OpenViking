import assert from "node:assert/strict";
import { createServer } from "node:http";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { expectExit, runHookScript } from "../../memory-plugin-shared/testing/support.mjs";

const hook = fileURLToPath(new URL("../scripts/hook.mjs", import.meta.url));

// Mirrors OPENVIKING_TIMEOUT_MS below: the parent must never stall this long.
const HOOK_TIMEOUT_MS = 5000;
// Slow CI runners need a generous budget for the detached worker to finish
// (cold start + two delayed responses + state writes).
const WORKER_WAIT_MS = 20000;

function waitFor(predicate, timeoutMs = WORKER_WAIT_MS) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const poll = () => {
      if (predicate()) {
        resolve();
        return;
      }
      if (Date.now() >= deadline) {
        reject(new Error("timed out waiting for the detached Kimi Code worker"));
        return;
      }
      setTimeout(poll, 25);
    };
    poll();
  });
}

test("Stop returns before slow writes while the detached worker finishes capture", async (t) => {
  const requests = [];
  let completedResponses = 0;
  const server = createServer((request, response) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      requests.push({ url: request.url, body: Buffer.concat(chunks).toString() });
      setTimeout(() => {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end('{"result":{}}');
        completedResponses++;
      }, 700);
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());

  const home = mkdtempSync(join(tmpdir(), "kc-async-"));
  t.after(() => rmSync(home, { recursive: true, force: true }));
  const sessionId = "session_async";
  const sessionDir = join(home, "sessions", "wd_x", sessionId);
  mkdirSync(join(sessionDir, "agents", "main"), { recursive: true });
  writeFileSync(
    join(home, "session_index.jsonl"),
    `${JSON.stringify({ sessionId, sessionDir, workDir: home })}\n`,
  );
  writeFileSync(
    join(sessionDir, "agents", "main", "wire.jsonl"),
    [
      JSON.stringify({ type: "turn.prompt", input: [{ type: "text", text: "slow question" }] }),
      JSON.stringify({
        type: "context.append_message",
        message: { role: "user", content: [{ type: "text", text: "slow question" }] },
      }),
      JSON.stringify({
        type: "context.append_loop_event",
        event: { type: "content.part", turnId: "turn-001", part: { type: "text", text: "slow answer" } },
      }),
    ].join("\n") + "\n",
  );

  const startedAt = Date.now();
  // The detached worker is respawned with no arguments at all, so what the
  // first run put in the environment is what tells it which event it is.
  const run = await runHookScript(hook, {
    argv: ["stop", "kimicode"],
    input: { session_id: sessionId, cwd: home },
    env: {
      HOME: home,
      KIMI_CODE_HOME: home,
      OPENVIKING_URL: `http://127.0.0.1:${server.address().port}`,
      OPENVIKING_WRITE_PATH_ASYNC: "1",
      OPENVIKING_TIMEOUT_MS: String(HOOK_TIMEOUT_MS),
    },
  });
  const elapsedMs = Date.now() - startedAt;

  expectExit(run);
  assert.equal(run.stdout, "", "Stop is observation-only for this host");
  // Core async-path guarantee, independent of wall-clock speed: every server
  // response is delayed 700ms, so a parent that exits with zero completed
  // responses provably never awaited the network.
  assert.equal(completedResponses, 0, "parent waited for a network response");
  // Hang guard only, not a latency budget.
  assert.ok(elapsedMs < HOOK_TIMEOUT_MS, `parent hook took ${elapsedMs}ms`);

  await waitFor(() => requests.some(({ url }) => url?.endsWith("/commit")));
  const batch = requests.find(({ url }) => url?.endsWith("/messages/batch"));
  assert.ok(batch, "the detached worker never posted messages");
  assert.deepEqual(JSON.parse(batch.body).messages, [
    { role: "user", content: "slow question", turn_id: "turn-001" },
    { role: "assistant", content: "slow answer", turn_id: "turn-001" },
  ]);
  assert.ok(batch.url.includes("kc-"), "Kimi Code sessions carry the kc- prefix");

  const statePath = join(home, ".openviking", "hook-state", "kimicode", `${sessionId}.json`);
  await waitFor(() => existsSync(statePath));
  await waitFor(() => JSON.parse(readFileSync(statePath, "utf8")).lastTurnId === "turn-001");
  const state = JSON.parse(readFileSync(statePath, "utf8"));
  assert.deepEqual(state.capturedTurnIds, ["turn-001:user", "turn-001:assistant"]);
  assert.equal(state.captured, undefined);
});
