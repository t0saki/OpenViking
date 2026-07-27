import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildContextSearchBody,
  buildRecallBlock,
  buildRecallEndpointBody,
  downgradeToRecallBody,
  isContextFaceLegacy,
  markContextFaceLegacy,
  postRecall,
} from "./lib/recall-core.mjs";

async function tempPath(name) {
  const dir = await mkdtemp(join(tmpdir(), "ov-recall-"));
  return join(dir, name);
}

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
  assert.equal(body.max_chars, 3000);
  assert.equal(body.min_score, 0.35);
  assert.equal(body.render, true);
  assert.equal(body.peer_scope, undefined);
});

test("buildRecallEndpointBody only sends actor peer scope when explicitly configured", () => {
  assert.equal(buildRecallEndpointBody({ recallPeerScope: "all" }).peer_scope, undefined);
  assert.equal(buildRecallEndpointBody({ recallPeerScope: "actor" }).peer_scope, "actor");
});

test("buildContextSearchBody declares intent and leaves mechanics to the server", () => {
  const body = buildContextSearchBody({ recallMaxTokens: 1200, scoreThreshold: 0.4 });
  assert.equal(body.mode, "context");
  assert.equal(body.purpose, "coding");
  assert.equal(body.detail, "auto");
  assert.equal(body.max_tokens, 1200);
  assert.equal(body.score_threshold, 0.4);
  assert.equal(body.quotas, undefined);
  assert.equal(body.session_id, undefined);
  assert.equal(body.dedup_turns, undefined);
  assert.equal(body.rewrite, undefined);
});

test("buildContextSearchBody enables session features only with a session id", () => {
  const body = buildContextSearchBody({}, { sessionId: "s1" });
  assert.equal(body.session_id, "s1");
  assert.equal(body.query_expansion, "auto");
  assert.equal(body.dedup_turns, 5);

  const pinned = buildContextSearchBody({ recallDedupTurns: 0, recallQueryExpansion: "off" }, {
    sessionId: "s1",
  });
  assert.equal(pinned.dedup_turns, 0);
  assert.equal(pinned.query_expansion, "off");
});

test("buildContextSearchBody maps the rewrite knob", () => {
  assert.equal(buildContextSearchBody({ recallRewrite: "off" }).rewrite, undefined);
  assert.equal(buildContextSearchBody({ recallRewrite: "client" }).rewrite, undefined);
  assert.equal(buildContextSearchBody({ recallRewrite: "server" }).rewrite, true);

  const auto = buildContextSearchBody({ recallRewrite: "auto" }, { localCompressorAvailable: false });
  assert.equal(auto.rewrite, "auto");
  assert.equal(auto.rewrite_max_bullets, 6);

  const localFirst = buildContextSearchBody({ recallRewrite: "auto" }, {
    localCompressorAvailable: true,
  });
  assert.equal(localFirst.rewrite, undefined);
});

test("buildContextSearchBody caps exclude_uris at the server limit", () => {
  const excludeUris = Array.from({ length: 250 }, (_, i) => `viking://u/${i}`);
  assert.equal(buildContextSearchBody({}, { excludeUris }).exclude_uris.length, 200);
});

test("downgradeToRecallBody converts the token budget back to characters", () => {
  const context = buildContextSearchBody({ recallMaxTokens: 1600, recallPeerScope: "actor" });
  context.query = "hello";
  const body = downgradeToRecallBody(context, { recallLimit: 4, recallMaxContentChars: 500 });
  assert.equal(body.query, "hello");
  assert.equal(body.max_chars, 6400);
  assert.equal(body.peer_scope, "actor");
  assert.equal(body.mode, undefined);
});

test("buildRecallBlock injects the assembled context block", async () => {
  const calls = [];
  const legacyCachePath = await tempPath("context-face.json");
  const block = await buildRecallBlock(async (path, init) => {
    calls.push({ path, body: init?.body ? JSON.parse(init.body) : null });
    return {
      ok: true,
      result: {
        rendered: '<memory uri="viking://user/default/memories/a.md" type="events" score="0.90" detail="full">\nbody\n</memory>',
        entries: [{ uri: "viking://user/default/memories/a.md" }],
        stats: { used_tokens: 42, tier_counts: { full: 1 }, rewrite: "off" },
      },
    };
  }, { recallMaxTokens: 1600 }, "hello world", { legacyCachePath });

  assert.equal(calls[0].path, "/api/v1/search/search");
  assert.equal(calls[0].body.mode, "context");
  assert.match(block, /^<openviking-context>/);
  assert.match(block, /viking:\/\/user\/default\/memories\/a\.md/);
  assert.match(block, /<\/openviking-context>$/);
});

test("buildRecallBlock prefers the server digest over rendered context", async () => {
  const legacyCachePath = await tempPath("context-face.json");
  const block = await buildRecallBlock(async () => ({
    ok: true,
    result: {
      rendered: '<memory uri="viking://a" type="events" score="0.5" detail="abstract">body</memory>',
      digest: "OpenViking memory digest:\n- fact 来源：viking://a",
      entries: [{ uri: "viking://a" }],
      stats: { rewrite: "ok" },
    },
  }), {}, "hello", { legacyCachePath });

  assert.match(block, /OpenViking memory digest:/);
  assert.doesNotMatch(block, /<memory /);
});

test("buildRecallBlock compresses locally when the knob asks for it", async () => {
  const legacyCachePath = await tempPath("context-face.json");
  const digestCachePath = await tempPath("recall-digest.json");
  const prompts = [];
  const block = await buildRecallBlock(async () => ({
    ok: true,
    result: {
      rendered: `<memory uri="viking://a" type="events" score="0.5" detail="full">${"x".repeat(2000)}</memory>`,
      entries: [{ uri: "viking://a" }],
      stats: {},
    },
  }), { recallRewrite: "client" }, "hello", {
    legacyCachePath,
    digestCachePath,
    runCompressor: async (prompt) => {
      prompts.push(prompt);
      return "- local fact 来源：viking://a";
    },
  });

  assert.equal(prompts.length, 1);
  assert.match(prompts[0], /memory relevance compressor/);
  assert.match(block, /OpenViking memory digest:/);
  assert.match(block, /local fact/);
});

test("buildRecallBlock falls back to /recall and remembers a legacy server", async () => {
  const legacyCachePath = await tempPath("context-face.json");
  const paths = [];
  const fetchJSON = async (path) => {
    paths.push(path);
    if (path === "/api/v1/search/search") {
      return { ok: false, status: 400, error: { message: "Extra inputs are not permitted: mode" } };
    }
    if (path === "/api/v1/search/recall") {
      return { ok: true, result: { rendered: "<memory index=\"1\" type=\"uri\"/>" } };
    }
    return { ok: false, status: 404 };
  };

  const block = await buildRecallBlock(fetchJSON, {}, "hello", { legacyCachePath });
  assert.deepEqual(paths, ["/api/v1/search/search", "/api/v1/search/recall"]);
  assert.match(block, /<openviking-context>/);
  assert.equal(await isContextFaceLegacy(legacyCachePath), true);

  // Second turn skips the context face entirely.
  paths.length = 0;
  await buildRecallBlock(fetchJSON, {}, "hello again", { legacyCachePath });
  assert.deepEqual(paths, ["/api/v1/search/recall"]);
});

test("buildRecallBlock does not cache legacy on unrelated request errors", async () => {
  const legacyCachePath = await tempPath("context-face.json");
  const fetchJSON = async (path) => {
    if (path === "/api/v1/search/search") return { ok: false, status: 400, error: "bad query" };
    if (path === "/api/v1/search/recall") return { ok: true, result: { rendered: "ok" } };
    return { ok: false, status: 404 };
  };

  await buildRecallBlock(fetchJSON, {}, "hello", { legacyCachePath });
  assert.equal(await isContextFaceLegacy(legacyCachePath), false);
});

test("buildRecallBlock falls back to find and keeps first item over budget", async () => {
  const calls = [];
  const legacyCachePath = await tempPath("context-face.json");
  const longAbstract = "x".repeat(1200);
  const fetchJSON = async (path) => {
    calls.push(path);
    if (path === "/api/v1/search/search") return { ok: false, status: 503 };
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
  }, "what happened yesterday", { legacyCachePath });

  assert.ok(calls.includes("/api/v1/search/search"));
  assert.ok(calls.includes("/api/v1/search/recall"));
  assert.ok(calls.includes("/api/v1/search/find"));
  assert.match(block, /^<openviking-context>/);
  assert.match(block, /\[memory 90%\]/);
  assert.match(block, /x{100}/);
});

test("context face legacy cache expires", async () => {
  const legacyCachePath = await tempPath("context-face.json");
  const now = 1_000_000;
  await markContextFaceLegacy(legacyCachePath, now);
  assert.equal(await isContextFaceLegacy(legacyCachePath, now + 1000), true);
  assert.equal(await isContextFaceLegacy(legacyCachePath, now + 7 * 60 * 60 * 1000), false);
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
    assert.deepEqual(logs, [{ stage: "recall_peer_scope_downgrade", data: { status } }]);
  }
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
