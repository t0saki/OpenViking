import test from "node:test";
import assert from "node:assert/strict";

import {
  compileSessionPatterns,
  isBypassed,
  matchesSessionPattern,
} from "./lib/session-model.mjs";

test("a single star stops at the segment separator", () => {
  const paths = compileSessionPatterns(["/tmp/*/scratch"]);
  assert.equal(matchesSessionPattern(["/tmp/a/scratch"], paths), true);
  assert.equal(matchesSessionPattern(["/tmp/a/b/scratch"], paths), false);

  const refs = compileSessionPatterns(["agent:*:cron:**"], { segmentSeparator: ":" });
  assert.equal(matchesSessionPattern(["agent:main:cron:nightly:run:1"], refs), true);
  assert.equal(matchesSessionPattern(["agent:a:b:cron:x"], refs), false);
  assert.equal(matchesSessionPattern(["agent:main:main"], refs), false);
});

test("a colon-separated pattern still treats a slash as an ordinary character", () => {
  const refs = compileSessionPatterns(["agent:*:cron"], { segmentSeparator: ":" });
  assert.equal(matchesSessionPattern(["agent:main/nested:cron"], refs), true);
});

test("haystacks are a precedence list: the first non-empty one decides", () => {
  const refs = compileSessionPatterns(["agent:*:cron:**"], { segmentSeparator: ":" });

  assert.equal(
    matchesSessionPattern(["agent:main:main", "agent:main:cron:from-id"], refs),
    false,
    "a sessionKey that does not match wins over a sessionId that does",
  );
  assert.equal(matchesSessionPattern(["  ", "agent:main:cron:from-id"], refs), true);
  assert.equal(matchesSessionPattern([undefined, undefined], refs), false);
  assert.equal(matchesSessionPattern(["agent:main:cron:x"], []), false);
});

test("isBypassed matches either the session id or the cwd", () => {
  const cfg = { bypassSessionPatterns: ["cc-scratch-*", "/tmp/**"] };
  assert.equal(isBypassed(cfg, { sessionId: "cc-scratch-1", cwd: "/work" }), true);
  assert.equal(isBypassed(cfg, { sessionId: "cc-main", cwd: "/tmp/a/b" }), true);
  assert.equal(isBypassed(cfg, { sessionId: "cc-main", cwd: "/work" }), false);
  assert.equal(isBypassed({}, { sessionId: "cc-main" }), false);
  assert.equal(isBypassed({ bypassSession: true }, { cwd: "/work" }), true);
});
