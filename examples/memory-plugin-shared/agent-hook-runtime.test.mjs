import assert from "node:assert/strict";
import test from "node:test";

import { filterCaptureTurns } from "./lib/capture-utils.mjs";

import {
  commitAgentSession,
  makeAgentFetchJSON,
} from "./lib/agent-hook-runtime.mjs";

function jsonResponse(status, value) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("agent fetch and commit logging preserve response trace_id", async (t) => {
  const responses = [
    jsonResponse(200, {
      status: "ok",
      result: {
        session_id: "agent-trace-success",
        status: "accepted",
        trace_id: "trace-agent-success",
      },
    }),
    jsonResponse(400, {
      status: "error",
      error: {
        code: "INTERNAL",
        message: "commit failed",
        trace_id: "trace-agent-error",
      },
    }),
  ];
  t.mock.method(globalThis, "fetch", async () => responses.shift());
  const { fetchJSON } = makeAgentFetchJSON({
    baseUrl: "http://127.0.0.1:1933",
    timeoutMs: 5000,
  });
  const logs = [];

  const success = await commitAgentSession(
    fetchJSON,
    "agent-trace-success",
    (stage, data) => logs.push({ stage, data }),
  );
  assert.equal(success.traceId, "trace-agent-success");
  assert.equal(success.result.trace_id, "trace-agent-success");
  assert.deepEqual(logs[0], {
    stage: "commit",
    data: {
      sessionId: "agent-trace-success",
      ok: true,
      status: "accepted",
      trace_id: "trace-agent-success",
      queued: false,
      error: undefined,
    },
  });

  const failure = await commitAgentSession(
    fetchJSON,
    "agent-trace-error",
    (stage, data) => logs.push({ stage, data }),
  );
  assert.equal(failure.ok, false);
  assert.equal(failure.traceId, "trace-agent-error");
  assert.equal(failure.error.trace_id, "trace-agent-error");
  assert.deepEqual(logs[1], {
    stage: "commit",
    data: {
      sessionId: "agent-trace-error",
      ok: false,
      status: 400,
      trace_id: "trace-agent-error",
      queued: false,
      error: "commit failed",
    },
  });
});

// The thin harnesses composed on this runtime used to send whatever their
// transcript parser produced. Every other harness runs the same filter, so it
// belongs beside the runtime rather than reimplemented in each hook.
test("the shared capture filter drops the turns no harness wants to remember", () => {
  const { kept, dropped } = filterCaptureTurns([
    { role: "user", content: "/compact" },
    { role: "assistant", content: "ok" },
    { role: "user", content: "..." },
    { role: "assistant", content: "[openviking-memory] recalled 3 items" },
    { role: "user", content: "the retry budget is three attempts" },
  ], { captureMaxLength: 24000 });

  assert.deepEqual(kept.map((turn) => turn.content), ["the retry budget is three attempts"]);
  assert.deepEqual(dropped.map((turn) => turn.reason), [
    "slash_command", "ack", "punctuation", "plugin_status",
  ]);
});

test("a turn past captureMaxLength is capped rather than dropped", () => {
  const long = "remember this detail. ".repeat(200);
  const { kept, dropped } = filterCaptureTurns(
    [{ role: "user", content: long }],
    { captureMaxLength: 200 },
  );

  assert.equal(dropped.length, 0);
  assert.ok(kept[0].content.length < long.length, "the turn must reach the extractor capped");
  assert.ok(kept[0].content.startsWith("remember this detail."));
});

test("filterCaptureTurns tolerates a missing or malformed turn list", () => {
  for (const input of [undefined, null, "text", 3, {}]) {
    assert.deepEqual(filterCaptureTurns(input), { kept: [], dropped: [] });
  }
});
