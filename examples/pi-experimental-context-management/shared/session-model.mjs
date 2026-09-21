// GENERATED FROM examples/memory-plugin-shared/lib. DO NOT EDIT.
/**
 * Shared OpenViking session-id helpers for memory plugin harnesses.
 */

/**
 * Glob -> RegExp. Minimal implementation: supports `*`, `**`, and literals.
 */
function globToRe(glob) {
  let re = "^";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i];
    if (c === "*") {
      if (glob[i + 1] === "*") { re += ".*"; i++; }
      else re += "[^/]*";
    } else if (/[.+?^${}()|[\]\\]/.test(c)) {
      re += "\\" + c;
    } else {
      re += c;
    }
  }
  re += "$";
  return new RegExp(re);
}

export function isBypassed(cfg, { sessionId, cwd } = {}) {
  if (cfg.bypassSession) return true;
  const patterns = cfg.bypassSessionPatterns || [];
  if (patterns.length === 0) return false;
  const haystacks = [sessionId, cwd].filter(Boolean);
  for (const pat of patterns) {
    const re = globToRe(pat);
    if (haystacks.some((h) => re.test(h))) return true;
  }
  return false;
}

function safeId(value, replacement = "_") {
  return String(value || "unknown").replace(/[^A-Za-z0-9._-]/g, replacement);
}

export function deriveHarnessSessionId(prefix, sessionId, suffix = "") {
  if (!prefix || typeof prefix !== "string") {
    throw new Error("deriveHarnessSessionId requires a non-empty prefix");
  }
  if (!sessionId || typeof sessionId !== "string") {
    throw new Error("deriveHarnessSessionId requires a non-empty sessionId");
  }
  const base = `${prefix}${sessionId}`;
  if (!suffix) return base;
  const normalized = String(suffix).replace(/:/g, "-").replace(/[^A-Za-z0-9._-]/g, "-");
  return `${base}__${normalized}`;
}

export function deriveCodexSessionId(codexSessionId) {
  return `cx-${safeId(codexSessionId, "_")}`;
}

// Sessions created before the rollout keep their original IDs on every host.
export const READABLE_ID_EPOCH_MS = Date.UTC(2026, 8, 22);
const MIN_TIME_MS = Date.UTC(2020, 0, 1);
const MAX_TIME_MS = Date.UTC(2100, 0, 1);

function validTime(ms) {
  return Number.isFinite(ms) && ms >= MIN_TIME_MS && ms < MAX_TIME_MS;
}

/** UUIDv7 has a 48-bit Unix millisecond timestamp, independent of the clock. */
export function uuidV7TimeMs(id) {
  if (typeof id !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) return null;
  const ms = Number.parseInt(id.slice(0, 8) + id.slice(9, 13), 16);
  return validTime(ms) ? ms : null;
}

/** OpenCode stores ~(ms * 4096 + counter) in 48 bits (36 timestamp bits). */
export function opencodeTimeMs(id, nowMs = Date.now()) {
  if (typeof id !== "string" || !/^ses_[0-9a-f]{12}[a-z0-9]{14}$/i.test(id) || !validTime(nowMs)) return null;
  const lowMs = Number((BigInt("0x" + id.slice(4, 16)) ^ 0xffffffffffffn) >> 12n);
  const period = 2 ** 36;
  const ms = lowMs + Math.round((nowMs - lowMs) / period) * period;
  return validTime(ms) ? ms : null;
}

export function nativeIdTail(id) {
  if (typeof id !== "string") return null;
  const tail = id.replace(/[^a-z0-9]/gi, "").slice(-8).toLowerCase();
  return tail.length === 8 ? tail : null;
}

/** Format in UTC; null means the caller must keep its legacy ID. */
export function formatReadableSessionId(harness, startMs, nativeId, suffix = "") {
  const tail = nativeIdTail(nativeId);
  if (!/^[a-z]+$/.test(harness) || !validTime(startMs) || !tail) return null;
  const utc = new Date(startMs).toISOString().replace(/[-:]/g, "");
  return deriveHarnessSessionId(harness + "-" + utc.slice(0, 8) + "-" + utc.slice(9, 15) + "-", tail, suffix);
}

export function deriveReadableSessionId(harness, prefix, nativeId, startMs, suffix = "") {
  return (startMs >= READABLE_ID_EPOCH_MS && formatReadableSessionId(harness, startMs, nativeId, suffix))
    || deriveHarnessSessionId(prefix, nativeId, suffix);
}
