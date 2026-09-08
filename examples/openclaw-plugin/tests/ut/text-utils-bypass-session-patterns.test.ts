import { describe, expect, it } from "vitest";

import { memoryOpenVikingConfigSchema } from "../../config.js";
import { compileSessionPatterns, matchesSessionPattern } from "../../shared/session-model.mjs";

/** OpenClaw session refs are colon-delimited, so a single `*` stops at a colon. */
const compile = (patterns: string[]): RegExp[] =>
  compileSessionPatterns(patterns, { segmentSeparator: ":" });

const isBypassed = (
  params: { sessionId?: string; sessionKey?: string },
  patterns: RegExp[],
): boolean => matchesSessionPattern([params.sessionKey, params.sessionId], patterns);

describe("bypass session patterns", () => {
  it("parses bypass session patterns from config", () => {
    const cfg = memoryOpenVikingConfigSchema.parse({
      bypassSessionPatterns: [
        "agent:*:cron:**",
        "agent:ops:maintenance:**",
      ],
    });

    expect(cfg.bypassSessionPatterns).toEqual([
      "agent:*:cron:**",
      "agent:ops:maintenance:**",
    ]);
  });

  it("accepts deprecated ingestReplyAssistIgnoreSessionPatterns as bypassSessionPatterns fallback", () => {
    const cfg = memoryOpenVikingConfigSchema.parse({
      ingestReplyAssistIgnoreSessionPatterns: [
        "agent:*:cron:**",
      ],
    });

    expect(cfg.bypassSessionPatterns).toEqual([
      "agent:*:cron:**",
    ]);
  });

  it("defaults bypass session patterns to an empty list", () => {
    const cfg = memoryOpenVikingConfigSchema.parse({});
    expect(cfg.bypassSessionPatterns).toEqual([]);
  });

  it("matches lossless-claw style session globs", () => {
    const patterns = compile([
      "agent:*:cron:**",
      "agent:ops:maintenance:**",
    ]);

    expect(matchesSessionPattern("agent:main:cron:nightly:run:1", patterns)).toBe(true);
    expect(matchesSessionPattern("agent:ops:maintenance:weekly", patterns)).toBe(true);
    expect(matchesSessionPattern("agent:main:main", patterns)).toBe(false);
  });

  it("prefers sessionKey over sessionId when deciding whether to bypass", () => {
    const patterns = compile(["agent:*:cron:**"]);

    expect(
      isBypassed(
        {
          sessionId: "agent:main:cron:from-id",
          sessionKey: "agent:main:main",
        },
        patterns,
      ),
    ).toBe(false);
  });

  it("falls back to sessionId when sessionKey is unavailable", () => {
    const patterns = compile(["agent:*:cron:**"]);

    expect(
      isBypassed(
        {
          sessionId: "agent:main:cron:nightly:run:1",
        },
        patterns,
      ),
    ).toBe(true);
  });
});
