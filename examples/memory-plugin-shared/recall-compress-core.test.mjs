import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  buildRecallCompressionPrompt,
  compressRecallContext,
  normalizeCompressedContext,
} from "./lib/recall-compress-core.mjs";

test("compression prompt forbids tools and constrains bullets", () => {
  const prompt = buildRecallCompressionPrompt({ query: "q", rendered: "memory", maxBullets: 4 });
  assert.match(prompt, /Do not use any tools/);
  assert.match(prompt, /at most 4 bullets/);
  assert.match(prompt, /120 characters/);
});

test("compressed output requires cited bullets and normalizes no-memory", () => {
  assert.equal(normalizeCompressedContext("NO_RELEVANT_MEMORY"), "");
  assert.equal(normalizeCompressedContext("Here is some chatter"), null);
  assert.equal(
    normalizeCompressedContext("OpenViking memory digest:\n* useful (viking://user/a.md)"),
    "OpenViking memory digest:\n- useful (viking://user/a.md)",
  );
});

test("digest cache reuses an unchanged URI set without another model call", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-recall-digest-"));
  let calls = 0;
  const args = {
    query: "hello",
    rendered: "x".repeat(1600) + " viking://user/a.md",
    entries: [{ uri: "viking://user/a.md" }],
    cfg: { recallCompressMinInputChars: 1500 },
    cachePath: join(dir, "cache.json"),
    runCompressor: async () => {
      calls += 1;
      return "- remembered (viking://user/a.md)";
    },
  };
  try {
    const first = await compressRecallContext(args);
    const second = await compressRecallContext({ ...args, query: "different wording" });
    assert.equal(first, second);
    assert.equal(calls, 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
