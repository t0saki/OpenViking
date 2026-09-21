export function deriveHarnessSessionId(prefix: string, sessionId: string, suffix?: string): string;
export function deriveCodexSessionId(codexSessionId: string): string;
export function isBypassed(cfg: Record<string, any>, options?: { sessionId?: string; cwd?: string }): boolean;
export const READABLE_ID_EPOCH_MS: number;
export function uuidV7TimeMs(id: string): number | null;
export function opencodeTimeMs(id: string, nowMs?: number): number | null;
export function nativeIdTail(id: string): string | null;
export function formatReadableSessionId(harness: string, startMs: number | null | undefined, nativeId: string, suffix?: string): string | null;
export function deriveReadableSessionId(harness: string, prefix: string, nativeId: string, startMs: number | null | undefined, suffix?: string): string;
