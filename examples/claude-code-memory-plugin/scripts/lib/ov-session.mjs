/** Stable OpenViking session identities for independent Claude Code hooks. */

import { linkSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
import { STATE_DIR, statePath } from "./state.mjs";
import {
  addAgentMessage,
  commitAgentSession,
  enqueueAgentPending,
  getAgentSession,
  getAgentSessionContext,
  makeAgentFetchJSON,
} from "../shared/agent-hook-runtime.mjs";
import { isRetryableFailure } from "../shared/retryable.mjs";
import {
  deriveHarnessSessionId,
  formatReadableSessionId,
  isBypassed,
} from "../shared/session-model.mjs";

/**
 * Check whether a CC session_id or cwd matches any bypass pattern.
 * Also honours OPENVIKING_BYPASS_SESSION env var (via cfg.bypassSession).
 */
export { isBypassed, isRetryableFailure };

export {
  addAgentMessage as addMessage,
  enqueueAgentPending as enqueuePendingDirectly,
  getAgentSession as getSession,
  getAgentSessionContext as getSessionContext,
};

/**
 * Derive a stable OV session ID from a CC session_id.
 *
 * Optionally append a suffix (e.g. subagent_id) for session isolation. The suffix
 * is normalized: `:` → `-` (so `subagent:abc123` → `subagent-abc123`) and any
 * characters outside [A-Za-z0-9._-] become `-`. Result: `cc-<uuid>__<suffix>`.
 */
export function deriveOvSessionId(ccSessionId, suffix = "") {
  return deriveHarnessSessionId("cc-", ccSessionId, suffix);
}

// Read errors and invalid pins are not absence. Never overwrite them or guess
// another ID: capture cursors belong to the pinned session.
async function readPin(file, nativeSessionId) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const pin = JSON.parse(readFileSync(file, "utf8"));
      if (pin?.version === 1 && pin.nativeSessionId === nativeSessionId
        && typeof pin.ovSessionId === "string" && pin.ovSessionId.trim()) return pin.ovSessionId;
    } catch (error) {
      if (error.code === "ENOENT") return undefined;
    }
    if (attempt < 2) await delay(10);
  }
  return null;
}

/**
 * Publish a complete, immutable pin before using either a new or legacy ID.
 * Only SessionStart / UserPromptSubmit may mint; writers pin legacy IDs.
 * null tells the hook to skip OV work, leaving capture cursors untouched.
 */
export async function resolveOvSessionId(nativeSessionId, { mint = false, source = "", fetchJSON } = {}) {
  if (!nativeSessionId || nativeSessionId === "unknown") return null;
  const safe = String(nativeSessionId).replace(/[^a-zA-Z0-9_-]/g, "_");
  const file = statePath("ov-session-" + safe + ".json");
  const pinned = await readPin(file, nativeSessionId);
  if (pinned !== undefined) return pinned;
  const legacy = deriveOvSessionId(nativeSessionId);
  let candidate = legacy;
  const mintedAt = Date.now();
  if (mint) {
    let fresh = source === "startup" || source === "clear";
    if (!fresh && fetchJSON) {
      try {
        const response = await fetchJSON("/api/v1/sessions/" + encodeURIComponent(legacy));
        fresh = response?.status === 404;
      } catch { /* An outage or ambiguous response must preserve legacy. */ }
    }
    if (fresh) candidate = formatReadableSessionId("claude", mintedAt, nativeSessionId) || legacy;
  }
  const tmp = file + "." + process.pid + "." + randomUUID() + ".tmp";
  try {
    mkdirSync(STATE_DIR, { recursive: true });
    writeFileSync(tmp, JSON.stringify({
      version: 1, nativeSessionId, ovSessionId: candidate,
      format: candidate === legacy ? "legacy" : "readable", mintedAt,
    }), { mode: 0o600, flag: "wx" });
    // Hard-link publication is atomic and cannot replace a concurrent winner.
    linkSync(tmp, file);
    return candidate;
  } catch {
    // Even a failed publisher may have lost a race to a valid pin. If there
    // is no winner, skip: returning legacy without a pin could split the next
    // hook from this one when storage becomes writable again.
    return (await readPin(file, nativeSessionId)) ?? null;
  } finally {
    try { unlinkSync(tmp); } catch { /* best effort temp cleanup */ }
  }
}

export async function resolveSubagentOvSessionId(nativeSessionId, subagentId) {
  const parent = await resolveOvSessionId(nativeSessionId);
  const child = String(subagentId).replace(/[^A-Za-z0-9._-]/g, "-");
  return parent ? parent + "__subagent-" + child : null;
}

/**
 * Build a fetchJSON closure tied to a given config. Callers pass their own cfg
 * (from scripts/config.mjs loadConfig()) so the timeout can vary per hook.
 */
export function makeFetchJSON(cfg, timeoutKey = "timeoutMs") {
  return makeAgentFetchJSON(cfg, process.cwd(), {
    defaultTimeoutMs: cfg[timeoutKey] || cfg.timeoutMs || 10000,
    // Every call on this stack names the peer it wants, so nothing here may put
    // one on a request that asked for none.
    getActorPeerId: () => "",
  }).fetchJSON;
}

/**
 * Commit the persistent OV session (archive + background extract). Safe to
 * call repeatedly: if there are no pending messages the server is a no-op.
 */
export function commitSession(fetchJSON, sessionId, payload = {}) {
  return commitAgentSession(fetchJSON, sessionId, undefined, payload);
}
