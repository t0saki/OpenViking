import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildRecallCompressionPrompt,
  compressRecallContext,
  normalizeCompressedContext,
  recallDigestCacheKey,
  repairDigestUris,
} from "./lib/recall-compress-core.mjs";
import { normalizeRewriteMode } from "./lib/plugin-config.mjs";

async function tempPath(name) {
  const dir = await mkdtemp(join(tmpdir(), "ov-compress-"));
  return join(dir, name);
}

test("prompt states the goal and the citation slot without a length contract", () => {
  const prompt = buildRecallCompressionPrompt({
    query: "what changed",
    rendered: "<memory uri=\"viking://a\" />",
    maxBullets: 4,
  });
  assert.match(prompt, /at most 4 bullets/);
  assert.match(prompt, /来源：viking:\/\//);
  assert.match(prompt, /No preamble, no closing remark/);
  assert.doesNotMatch(prompt, /120 characters/);
});

test("normalizeCompressedContext keeps only cited bullets", () => {
  assert.equal(normalizeCompressedContext("NO_RELEVANT_MEMORY"), "");
  assert.equal(normalizeCompressedContext("- uncited fact"), null);
  assert.equal(
    normalizeCompressedContext("chatter\n* fact 来源：viking://a\nmore chatter"),
    "OpenViking memory digest:\n- fact 来源：viking://a",
  );
});

test("normalizeCompressedContext honors the bullet ceiling", () => {
  const raw = "- one 来源：viking://a\n- two 来源：viking://b\n- three 来源：viking://c";
  const digest = normalizeCompressedContext(raw, 4000, 2);
  assert.equal(digest.split("\n").length, 3);
});

test("repairDigestUris snaps near-miss URIs back onto the served set", () => {
  const valid = ["viking://user/zhengxiao/memories/events/2026/07/14/release_notes.md"];
  const mangled = "- fact 来源：viking://user/zhengxiao/memories/events/2026/07/14/release_note.md";
  assert.equal(repairDigestUris(mangled, valid), `- fact 来源：${valid[0]}`);
});

test("repairDigestUris drops bullets whose citation cannot be recovered", () => {
  const valid = ["viking://user/u/memories/events/a.md"];
  const digest = [
    "OpenViking memory digest:",
    "- good 来源：viking://user/u/memories/events/a.md",
    "- invented 来源：viking://completely/different/place/xyz.md",
  ].join("\n");
  const repaired = repairDigestUris(digest, valid);
  assert.match(repaired, /good/);
  assert.doesNotMatch(repaired, /invented/);
});

test("cache key depends on the served URI set, not their order", () => {
  const a = recallDigestCacheKey([{ uri: "viking://a" }, { uri: "viking://b" }]);
  const b = recallDigestCacheKey([{ uri: "viking://b" }, { uri: "viking://a" }]);
  const c = recallDigestCacheKey([{ uri: "viking://a" }]);
  assert.equal(a, b);
  assert.notEqual(a, c);
});

test("compressRecallContext passes short input straight through", async () => {
  let called = false;
  const rendered = "<memory uri=\"viking://a\" />";
  const out = await compressRecallContext({
    query: "q",
    rendered,
    cfg: { recallCompressMinInputChars: 1500 },
    runCompressor: async () => { called = true; return ""; },
  });
  assert.equal(out, rendered);
  assert.equal(called, false);
});

test("compressRecallContext caches digests by served URI set", async () => {
  const cachePath = await tempPath("digest.json");
  const rendered = `<memory uri="viking://a">${"x".repeat(2000)}</memory>`;
  const entries = [{ uri: "viking://a" }];
  let calls = 0;

  const runCompressor = async () => {
    calls += 1;
    return "- fact 来源：viking://a";
  };

  const first = await compressRecallContext({
    query: "q", rendered, entries, runCompressor, cachePath, now: 1,
  });
  const second = await compressRecallContext({
    query: "q", rendered, entries, runCompressor, cachePath, now: 2,
  });

  assert.equal(calls, 1);
  assert.equal(first, second);
  assert.match(first, /OpenViking memory digest:/);
  assert.match(await readFile(cachePath, "utf8"), /"digest"/);
});

test("compressRecallContext returns null when the model emits nothing usable", async () => {
  const rendered = `<memory uri="viking://a">${"x".repeat(2000)}</memory>`;
  const out = await compressRecallContext({
    query: "q",
    rendered,
    entries: [{ uri: "viking://a" }],
    runCompressor: async () => "I could not find anything relevant.",
  });
  assert.equal(out, null);
});

test("normalizeRewriteMode accepts the tri-state knob and booleans", () => {
  assert.equal(normalizeRewriteMode("client"), "client");
  assert.equal(normalizeRewriteMode("SERVER"), "server");
  assert.equal(normalizeRewriteMode("auto"), "auto");
  assert.equal(normalizeRewriteMode("true"), "auto");
  assert.equal(normalizeRewriteMode("0"), "off");
  assert.equal(normalizeRewriteMode("nonsense"), "off");
  assert.equal(normalizeRewriteMode(undefined, "server"), "server");
});
