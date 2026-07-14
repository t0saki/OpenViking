import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildRecallBlock,
  buildRecallEndpointBody,
  postRecall,
  truncateRenderedAtFragmentBoundary,
} from "./lib/recall-core.mjs";

test("buildRecallEndpointBody maps quotas and max chars", () => {
  const body = buildRecallEndpointBody({
    recallLimit: 6,
    recallMaxContentChars: 500,
    scoreThreshold: 0.35,
  });
  assert.deepEqual(body.quotas, {
    events: 6,
    entities: 6,
    preferences: 3,
    experiences: 0,
  });
  assert.equal(body.max_chars, 6500);
  assert.equal(body.min_score, 0.35);
  assert.equal(body.render, "compact");
  assert.equal(body.peer_scope, undefined);
});

test("buildRecallEndpointBody only sends actor peer scope when explicitly configured", () => {
  assert.equal(buildRecallEndpointBody({ recallPeerScope: "all" }).peer_scope, undefined);
  assert.equal(buildRecallEndpointBody({ recallPeerScope: "actor" }).peer_scope, "actor");
});

test("buildRecallBlock uses recall endpoint render when available", async () => {
  const calls = [];
  const block = await buildRecallBlock(async (path, init) => {
    calls.push({ path, body: init?.body ? JSON.parse(init.body) : null });
    return { ok: true, result: { rendered: "- [memory 90%] viking://user/default/memories/a.md" } };
  }, {
    recallLimit: 2,
    recallMaxContentChars: 500,
    scoreThreshold: 0.35,
  }, "hello world");

  assert.equal(calls[0].path, "/api/v1/search/recall");
  assert.equal(calls[0].body.quotas.events, 2);
  assert.match(block, /^<openviking-context\b/);
  assert.match(block, /Relevant memory from OpenViking/);
  assert.match(block, /viking:\/\/user\/default\/memories\/a\.md/);
  assert.match(block, /<\/openviking-context>$/);
});

test("buildRecallBlock falls back to find and keeps first item over budget", async () => {
  const calls = [];
  const longAbstract = "x".repeat(1200);
  const fetchJSON = async (path) => {
    calls.push(path);
    if (path === "/api/v1/search/recall") return { ok: false, status: 404 };
    if (path === "/api/v1/system/status") return { ok: true, result: { user: "default" } };
    if (path.startsWith("/api/v1/fs/ls")) return { ok: true, result: [] };
    if (path === "/api/v1/search/find") {
      return {
        ok: true,
        result: {
          memories: [{
            uri: "viking://user/default/memories/events/a.md",
            score: 0.9,
            abstract: longAbstract,
            level: 1,
            category: "events",
          }],
          skills: [],
        },
      };
    }
    return { ok: false, status: 404 };
  };

  const block = await buildRecallBlock(fetchJSON, {
    recallLimit: 1,
    recallMaxContentChars: 500,
    recallTokenBudget: 20,
    scoreThreshold: 0.35,
    recallPreferAbstract: true,
  }, "what happened yesterday");

  assert.ok(calls.includes("/api/v1/search/recall"));
  assert.ok(calls.includes("/api/v1/search/find"));
  assert.match(block, /^<openviking-context>/);
  assert.match(block, /\[memory 90%\]/);
  assert.match(block, /x{100}/);
});

test("postRecall downgrades peer_scope on 400 and 422", async () => {
  for (const status of [400, 422]) {
    const bodies = [];
    const logs = [];
    const res = await postRecall(async (path, init, opts) => {
      bodies.push({ path, body: JSON.parse(init.body), opts });
      return bodies.length === 1
        ? { ok: false, status }
        : { ok: true, status: 200, result: { rendered: "ok" } };
    }, {
      query: "hello",
      peer_scope: "actor",
    }, {
      actorPeerId: "peer-a",
      log: (stage, data) => logs.push({ stage, data }),
    });

    assert.equal(res.ok, true);
    assert.equal(bodies.length, 2);
    assert.equal(bodies[0].body.peer_scope, "actor");
    assert.equal(bodies[1].body.peer_scope, undefined);
    assert.equal(bodies[0].opts.actorPeerId, "peer-a");
    assert.deepEqual(logs, [{ stage: "recall_legacy_downgrade", data: { status } }]);
  }
});

test("buildRecallEndpointBody separates injection and rewrite budgets", () => {
  const off = buildRecallEndpointBody({ recallMaxChars: 4321, recallRewrite: "off" });
  assert.equal(off.max_chars, 4321);
  assert.equal(off.render, "compact");

  const client = buildRecallEndpointBody({
    recallMaxChars: 4321,
    recallCompressMaxInputChars: 18000,
    recallRewrite: "client",
  }, { compress: async () => null });
  assert.equal(client.max_chars, 18000);
  assert.equal(client.render, true);

  const server = buildRecallEndpointBody({ recallRewrite: "server" });
  assert.equal(server.rewrite, true);
  assert.equal(server.max_chars, 18000);
});

test("postRecall extends the request timeout to cover server-side LLM budgets", async () => {
  const seenOpts = [];
  const fetchStub = async (_path, _init, opts) => {
    seenOpts.push(opts);
    return { ok: true, status: 200, result: { rendered: "ok" } };
  };

  await postRecall(fetchStub, { query: "q" });
  assert.equal(seenOpts[0].timeoutMs, undefined);

  await postRecall(fetchStub, { query: "q", rewrite: "auto" });
  assert.equal(seenOpts[1].timeoutMs, 35000);

  await postRecall(fetchStub, { query: "q", rewrite: true, session_id: "cx-1", query_expansion: "auto" });
  assert.equal(seenOpts[2].timeoutMs, 45000);
});

test("postRecall strips all optional fields when an older server rejects them", async () => {
  const bodies = [];
  const result = await postRecall(async (_path, init) => {
    bodies.push(JSON.parse(init.body));
    return bodies.length === 1 ? { ok: false, status: 422 } : { ok: true, result: {} };
  }, {
    query: "hello",
    session_id: "cx-1",
    query_expansion: "auto",
    exclude_uris: ["viking://a"],
    rewrite: "auto",
    rewrite_max_bullets: 6,
    render: "compact",
    peer_scope: "actor",
  });
  assert.equal(result.ok, true);
  assert.deepEqual(bodies[1], { query: "hello", render: true });
});

test("truncateRenderedAtFragmentBoundary never cuts a memory fragment", () => {
  const one = '<memory index="1"><uri>viking://a</uri></memory>';
  const two = '<memory index="2"><uri>viking://b</uri></memory>';
  const result = truncateRenderedAtFragmentBoundary(`${one}\n${two}`, one.length + 1);
  assert.equal(result, one);
  assert.doesNotMatch(result, /viking:\/\/b/);
});

test("postRecall does not retry default body or server errors", async () => {
  const noScopeBodies = [];
  const noScope = await postRecall(async (path, init) => {
    noScopeBodies.push(JSON.parse(init.body));
    return { ok: false, status: 400 };
  }, { query: "hello" });
  assert.equal(noScope.ok, false);
  assert.equal(noScopeBodies.length, 1);

  const serverErrorBodies = [];
  const serverError = await postRecall(async (path, init) => {
    serverErrorBodies.push(JSON.parse(init.body));
    return { ok: false, status: 500 };
  }, { query: "hello", peer_scope: "actor" });
  assert.equal(serverError.ok, false);
  assert.equal(serverErrorBodies.length, 1);
});

test("buildRecallBlock keeps the complete wrapper inside recallMaxChars", async () => {
  const first = `<memory index="1"><uri>viking://a</uri><content>${"a".repeat(650)}</content></memory>`;
  const second = `<memory index="2"><uri>viking://b</uri><content>${"b".repeat(650)}</content></memory>`;
  const block = await buildRecallBlock(async () => ({
    ok: true,
    result: { entries: [], rendered: `${first}\n${second}` },
  }), { recallMaxChars: 1000 }, "budget test");

  assert.ok(block.length <= 1000);
  assert.match(block, /<\/openviking-context>$/);
  assert.match(block, /viking:\/\/a/);
  assert.doesNotMatch(block, /viking:\/\/b/);
});

test("recall dedup excludes full entries for N turns then lets them cool down", async () => {
  const bodies = [];
  const uri = "viking://user/default/memories/a.md";
  const fetchJSON = async (_path, init) => {
    const body = JSON.parse(init.body);
    bodies.push(body);
    const firstOrCooled = bodies.length === 1 || bodies.length === 4;
    return {
      ok: true,
      result: {
        entries: firstOrCooled ? [{ uri, mode: "full" }] : [],
        rendered: firstOrCooled ? `<memory><uri>${uri}</uri></memory>` : "",
      },
    };
  };
  const cfg = { recallDedupTurns: 2 };
  for (let i = 0; i < 4; i += 1) {
    await buildRecallBlock(fetchJSON, cfg, `turn ${i}`, { sessionId: "dedup-full-session" });
  }

  assert.deepEqual(bodies[0].exclude_uris, undefined);
  assert.deepEqual(bodies[1].exclude_uris, [uri]);
  assert.deepEqual(bodies[2].exclude_uris, [uri]);
  assert.deepEqual(bodies[3].exclude_uris, undefined);
});

test("uri-only entries receive one full-content grace turn", async () => {
  const bodies = [];
  const uri = "viking://user/default/memories/grace.md";
  const fetchJSON = async (_path, init) => {
    bodies.push(JSON.parse(init.body));
    const mode = bodies.length === 1 ? "uri" : "full";
    return {
      ok: true,
      result: {
        entries: [{ uri, mode }],
        rendered: `<memory><uri>${uri}</uri></memory>`,
      },
    };
  };
  const options = { sessionId: "dedup-uri-grace-session" };
  await buildRecallBlock(fetchJSON, { recallDedupTurns: 5 }, "one", options);
  await buildRecallBlock(fetchJSON, { recallDedupTurns: 5 }, "two", options);
  await buildRecallBlock(fetchJSON, { recallDedupTurns: 5 }, "three", options);

  assert.equal(bodies[1].exclude_uris, undefined);
  assert.deepEqual(bodies[2].exclude_uris, [uri]);
});

test("session context is opt-in and sends the OpenViking session id", () => {
  const off = buildRecallEndpointBody({ recallSessionContext: "off" }, { sessionId: "ov-1" });
  const enabled = buildRecallEndpointBody({ recallSessionContext: "auto" }, { sessionId: "ov-1" });
  assert.equal(off.session_id, undefined);
  assert.equal(enabled.session_id, "ov-1");
  assert.equal(enabled.query_expansion, "auto");
});

test("legacy server detection persists across hook processes", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-recall-legacy-"));
  const cachePath = join(dir, "legacy.json");
  const cacheKey = `http://legacy-${Date.now()}`;
  try {
    let calls = 0;
    await postRecall(async () => {
      calls += 1;
      return calls === 1 ? { ok: false, status: 422 } : { ok: true, result: {} };
    }, { query: "hello", render: "compact" }, {
      legacyCacheKey: cacheKey,
      legacyCachePath: cachePath,
    });
    assert.equal(calls, 2);
    const saved = JSON.parse(await readFile(cachePath, "utf8"));
    assert.equal(saved.cacheKey, cacheKey);

    const fresh = await import(`./lib/recall-core.mjs?legacy=${Date.now()}`);
    const bodies = [];
    await fresh.postRecall(async (_path, init) => {
      bodies.push(JSON.parse(init.body));
      return { ok: true, result: {} };
    }, { query: "hello", render: "compact" }, {
      legacyCacheKey: cacheKey,
      legacyCachePath: cachePath,
    });
    assert.deepEqual(bodies, [{ query: "hello", render: true }]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
