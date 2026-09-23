import test from "node:test";
import assert from "node:assert/strict";
import { OVClient } from "../client.ts";

function makeClient() {
  return new OVClient({
    endpoint: "http://127.0.0.1:1933",
    apiKey: "",
    account: "",
    user: "",
    authMode: "trusted",
    sendIdentityHeaders: false,
    peerId: "",
    userAgent: "test",
  });
}

async function withFetch(handler, fn) {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    const value = await handler(url, init);
    return new Response(JSON.stringify(value.body), {
      status: value.status ?? 200,
      headers: { "content-type": "application/json" },
    });
  };
  try {
    return await fn(calls);
  } finally {
    globalThis.fetch = original;
  }
}

test("readArchiveOverview reads only the requested archive and strips OKF frontmatter", async () => {
  await withFetch(async () => ({
    body: { status: "ok", result: "---\ntitle: archive\n---\n\n# Working Memory\nbody\n" },
  }), async (calls) => {
    const uri = "viking://user/u/sessions/s/history/archive_007";
    assert.equal(await makeClient().readArchiveOverview(uri), "# Working Memory\nbody\n");
    assert.equal(calls.length, 1);
    assert.ok(calls[0].url.includes(encodeURIComponent(`${uri}/.overview.md`)));
  });
});

test("readArchiveOverview treats 404 and empty bodies as not ready", async () => {
  await withFetch(async () => ({ status: 404, body: { status: "error", error: { message: "missing" } } }), async () => {
    assert.equal(await makeClient().readArchiveOverview("viking://archive/1"), null);
  });
  await withFetch(async () => ({ body: { status: "ok", result: "---\ntitle: empty\n---\n\n" } }), async () => {
    assert.equal(await makeClient().readArchiveOverview("viking://archive/1"), null);
  });
});

test("readArchiveOverview surfaces non-404 read failures", async () => {
  await withFetch(async () => ({ status: 500, body: { status: "error", error: { message: "storage failed" } } }), async () => {
    await assert.rejects(
      () => makeClient().readArchiveOverview("viking://archive/1"),
      /archive overview read failed: storage failed/,
    );
  });
});

test("getTaskStatus reports the task state, a forgotten task, and an unreachable server apart", async () => {
  await withFetch(async (url) => {
    if (String(url).endsWith("/tasks/t-done")) return { body: { status: "ok", result: { task_id: "t-done", status: "completed" } } };
    if (String(url).endsWith("/tasks/t-gone")) {
      return { status: 404, body: { status: "error", error: { code: "NOT_FOUND", message: "Task not found or expired" } } };
    }
    return { status: 500, body: { status: "error", error: { code: "INTERNAL", message: "boom" } } };
  }, async (calls) => {
    const client = makeClient();
    assert.equal(await client.getTaskStatus("t-done"), "completed");
    assert.equal(await client.getTaskStatus("t-gone"), "missing");
    assert.equal(await client.getTaskStatus("t-broken"), null);
    assert.equal(await client.getTaskStatus(""), null);
    assert.equal(calls.length, 3);
    assert.match(calls[0].url, /\/api\/v1\/tasks\/t-done$/);
  });
});
