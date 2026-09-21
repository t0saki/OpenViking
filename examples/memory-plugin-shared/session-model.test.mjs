import assert from "node:assert/strict";
import test from "node:test";
import {
  deriveReadableSessionId, formatReadableSessionId, nativeIdTail,
  opencodeTimeMs, READABLE_ID_EPOCH_MS, uuidV7TimeMs,
} from "./lib/session-model.mjs";

test("UUIDv7 decoding is independent of the clock and rejects other ID formats", () => {
  assert.equal(uuidV7TimeMs("01a0a402-2f88-7848-83cc-52e892de7000"), Date.parse("2026-09-15T07:40:01.800Z"));
  assert.equal(uuidV7TimeMs("019f4182-4fe5-7647-8730-8ddf09a46a2f"), Date.parse("2026-07-08T11:34:47.013Z"));
  for (const id of [null, "unknown", "019f4182-4fe5-4647-8730-8ddf09a46a2f", "00000000-0000-7000-8000-000000000000", "ffffffff-ffff-7000-8000-000000000000"]) {
    assert.equal(uuidV7TimeMs(id), null);
  }
});

test("OpenCode inverts 48 bits and resolves both sides of the August 2026 wrap", () => {
  const now = Date.parse("2026-09-22T00:00:00Z");
  for (const [id, created] of [
    ["ses_010015a4cfferFQtjXPK49QIio", 1786437871027],
    ["ses_fdca75185ffeEo6lIymwg7XIki", 1787299409530],
  ]) assert.equal(opencodeTimeMs(id, now), created);
  assert.equal(opencodeTimeMs("oc-session-1", now), null);
  assert.equal(opencodeTimeMs("ses_fdca75185ffeEo6lIymwg7XIki", NaN), null);
});

test("readable IDs use UTC, lowercase tails, normalized subagent suffixes and a fixed epoch", () => {
  const id = "01a0c6bd-0a00-7000-8000-00009E3A1C07";
  const ms = Date.parse("2026-09-22T10:40:42Z");
  assert.equal(nativeIdTail(id), "9e3a1c07");
  assert.equal(nativeIdTail("!!!"), null);
  assert.equal(formatReadableSessionId("claude", ms, id, "subagent:a/b"), "claude-20260922-104042-9e3a1c07__subagent-a-b");
  assert.equal(deriveReadableSessionId("pi", "pi-", id, READABLE_ID_EPOCH_MS), "pi-20260922-000000-9e3a1c07");
  for (const value of [null, undefined, NaN, Infinity, "1789992000000", READABLE_ID_EPOCH_MS - 1]) {
    assert.equal(deriveReadableSessionId("pi", "pi-", id, value), "pi-" + id);
  }
  assert.equal(formatReadableSessionId("trae-cn", ms, id), null);
  assert.equal(formatReadableSessionId("claude", ms, "short"), null);
});
