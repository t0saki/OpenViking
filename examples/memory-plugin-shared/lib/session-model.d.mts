export function deriveHarnessSessionId(prefix: string, sessionId: string, suffix?: string): string;
export function deriveCodexSessionId(codexSessionId: string): string;
export function isBypassed(cfg: Record<string, any>, options?: { sessionId?: string; cwd?: string }): boolean;
export function compileSessionPatterns(
  patterns: readonly string[] | undefined,
  options?: { segmentSeparator?: string },
): RegExp[];
export function matchesSessionPattern(
  haystacks: readonly (string | undefined | null)[] | string | undefined | null,
  patterns: readonly RegExp[],
): boolean;
