import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { filterCaptureTurns } from "./lib/capture-utils.mjs";

import {
  commitAgentSession,
  loadAgentHookConfig,
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

/**
 * cursor, trae, trae-cn and zcode read the environment and nothing else before
 * this: an `ovcli.conf` `plugin` entry named after them was inert, and a
 * workspace file could not reach them at all. Assert the whole stack lands,
 * because none of these four has a config test of its own.
 */
test("the thin harnesses resolve the same layers as everyone else", () => {
  const dir = mkdtempSync(join(tmpdir(), "ov-agent-hook-config-"));
  const workspace = join(dir, "workspace");
  const saved = {
    cli: process.env.OPENVIKING_CLI_CONFIG_FILE,
    home: process.env.OPENVIKING_HOME,
    limit: process.env.OPENVIKING_RECALL_LIMIT,
  };
  try {
    mkdirSync(join(workspace, ".openviking"), { recursive: true });
    mkdirSync(join(workspace, ".git"), { recursive: true });
    writeFileSync(join(dir, "ovcli.conf"), JSON.stringify({
      url: "http://127.0.0.1:1933",
      plugin: {
        recallLimit: 7,
        cursor: { captureMode: "keyword" },
        "trae-cn": { scoreThreshold: 0.8 },
        zcode: { autoRecall: false },
      },
    }));
    process.env.OPENVIKING_CLI_CONFIG_FILE = join(dir, "ovcli.conf");
    // Keeps the identity cache and the workspace registry out of the real home.
    process.env.OPENVIKING_HOME = join(dir, "home");
    delete process.env.OPENVIKING_RECALL_LIMIT;

    const cursor = loadAgentHookConfig("cursor", workspace);
    assert.equal(cursor.recallLimit, 7, "the shared plugin section applies");
    assert.equal(cursor.captureMode, "keyword", "the per-harness override applies");
    assert.equal(cursor.autoRecall, true, "another harness's override does not");
    assert.equal(loadAgentHookConfig("zcode", workspace).autoRecall, false);
    // The hyphenated spelling of a harness finds the same override.
    assert.equal(loadAgentHookConfig("trae-cn", workspace).scoreThreshold, 0.8);
    assert.equal(loadAgentHookConfig("trae", workspace).scoreThreshold, 0.35);

    // A workspace file outranks ovcli.conf, and the environment outranks both.
    writeFileSync(
      join(workspace, ".openviking", "config.json"),
      JSON.stringify({ version: 1, recall: { max_items: 4 }, capture: { enabled: false } }),
    );
    const pinned = loadAgentHookConfig("cursor", workspace);
    assert.equal(pinned.recallLimit, 4);
    assert.equal(pinned.autoCapture, false);

    process.env.OPENVIKING_RECALL_LIMIT = "2";
    assert.equal(loadAgentHookConfig("cursor", workspace).recallLimit, 2);

    // And a directory with neither file keeps the ovcli.conf answer.
    assert.equal(loadAgentHookConfig("cursor", dir).recallLimit, 2);
  } finally {
    for (const [key, value] of [
      ["OPENVIKING_CLI_CONFIG_FILE", saved.cli],
      ["OPENVIKING_HOME", saved.home],
      ["OPENVIKING_RECALL_LIMIT", saved.limit],
    ]) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    rmSync(dir, { recursive: true, force: true });
  }
});
