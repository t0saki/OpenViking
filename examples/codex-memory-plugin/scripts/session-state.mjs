/**
 * Per-codex-session state for the OpenViking memory plugin.
 *
 * One state file per codex session_id, holding the long-lived OpenViking
 * session id that we incrementally append turns to via the Stop hook. The
 * OV session id is derived as `cx-<codex-session-id>` for new captures.
 * The OV session is committed (which extracts memories) by SessionEnd, by
 * PreCompact, or by the fallback sweep at SessionStart.
 *
 * Two sidecars live next to `<safeId>.json`:
 *   - `<safeId>.ended` — written by the SessionEnd parent hook, lock-free, so
 *     the sweep can still commit the session if the worker never ran. A
 *     whole-object saveState from a concurrent worker cannot clobber it.
 *   - `<safeId>.lock`  — an exclusive mkdir lock serializing the writers
 *     (Stop worker, PreCompact, SessionEnd worker, SessionStart sweep) that
 *     all persist the whole state object.
 *
 * State directory: $OPENVIKING_CODEX_STATE_DIR or ~/.openviking/codex-plugin-state
 */

import { mkdir, readFile, readdir, rename, rm, rmdir, stat, utimes, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { deriveCodexSessionId } from "./shared/session-model.mjs";

const DEFAULT_STATE_DIR = join(homedir(), ".openviking", "codex-plugin-state");

export function getStateDir() {
  return process.env.OPENVIKING_CODEX_STATE_DIR || DEFAULT_STATE_DIR;
}

function safeId(codexSessionId) {
  return String(codexSessionId).replace(/[^a-zA-Z0-9_-]/g, "_");
}

export function deriveOvSessionId(codexSessionId) {
  return deriveCodexSessionId(codexSessionId);
}

export function resolveOvSessionId(state) {
  // Always derive the deterministic cx-* id. Legacy persisted UUIDs from
  // before the cx-* scheme are no longer preserved: the migration window
  // has closed and keeping them would desync recall (which derives cx-*)
  // from capture (which used to echo back the legacy value).
  state.ovSessionId = deriveOvSessionId(state.codexSessionId);
  return state.ovSessionId;
}

function statePath(codexSessionId) {
  return join(getStateDir(), `${safeId(codexSessionId)}.json`);
}

function endedPath(codexSessionId) {
  return join(getStateDir(), `${safeId(codexSessionId)}.ended`);
}

function lockPath(codexSessionId) {
  return join(getStateDir(), `${safeId(codexSessionId)}.lock`);
}

function defaultState(codexSessionId) {
  const now = Date.now();
  return {
    codexSessionId,
    ovSessionId: null,
    workspacePeerId: "",
    capturedTurnCount: 0,
    createdAt: now,
    lastUpdatedAt: now,
  };
}

export async function loadState(codexSessionId) {
  try {
    const raw = await readFile(statePath(codexSessionId), "utf-8");
    const parsed = JSON.parse(raw);
    return { ...defaultState(codexSessionId), ...parsed };
  } catch {
    return defaultState(codexSessionId);
  }
}

/**
 * Persist state. `touch: false` keeps the existing `lastUpdatedAt` so a write
 * that isn't transcript activity (e.g. releasing `ovSessionId` after a commit)
 * doesn't make a dead session look freshly used to the idle-TTL sweep or to
 * the doctor's orphan count.
 */
export async function saveState(state, { touch = true } = {}) {
  if (!state || !state.codexSessionId) return;
  await mkdir(getStateDir(), { recursive: true });
  const next = {
    ...state,
    lastUpdatedAt: touch || typeof state.lastUpdatedAt !== "number"
      ? Date.now()
      : state.lastUpdatedAt,
  };
  // Atomic write (tmpfile + rename) so a crash mid-write can't leave a
  // truncated/corrupt state file. See DESIGN.md "State file schema".
  const final = statePath(state.codexSessionId);
  const tmp = `${final}.tmp`;
  await writeFile(tmp, JSON.stringify(next));
  await rename(tmp, final);
}

export async function clearState(codexSessionId) {
  try {
    await rm(statePath(codexSessionId), { force: true });
  } catch { /* best effort */ }
  await clearEnded(codexSessionId);
}

/** Record that the codex thread ended. Written lock-free by the SessionEnd parent. */
export async function markEnded(codexSessionId) {
  if (!codexSessionId) return;
  try {
    await mkdir(getStateDir(), { recursive: true });
    const final = endedPath(codexSessionId);
    const tmp = `${final}.tmp`;
    await writeFile(tmp, String(Date.now()));
    await rename(tmp, final);
  } catch { /* best effort */ }
}

/** Drop the end marker: the thread is alive again, or its commit succeeded. */
export async function clearEnded(codexSessionId) {
  if (!codexSessionId) return;
  try {
    await rm(endedPath(codexSessionId), { force: true });
  } catch { /* best effort */ }
}

async function readEndedAt(codexSessionId) {
  try {
    const raw = await readFile(endedPath(codexSessionId), "utf-8");
    const ts = Number(raw.trim());
    return Number.isFinite(ts) && ts > 0 ? ts : Date.now();
  } catch {
    return undefined;
  }
}

const LOCK_POLL_MS = 100;

/**
 * Run `fn` while holding an exclusive per-session lock.
 *
 * The lock is a directory (mkdir is atomic everywhere we run). A lock whose
 * mtime is older than `staleMs` is abandoned, so a killed holder cannot wedge
 * a session forever; a live holder keeps it fresh through `heartbeat()`.
 * `waitMs: 0` makes this a try-lock that returns `{ skipped: true }` instead
 * of waiting. Callers must always load state *inside* `fn`.
 */
export async function withSessionLock(codexSessionId, fn, { waitMs = 0, staleMs = 300_000 } = {}) {
  const dir = lockPath(codexSessionId);
  await mkdir(getStateDir(), { recursive: true });
  const deadline = Date.now() + Math.max(0, waitMs);
  let held = false;
  while (true) {
    try {
      await mkdir(dir);
      held = true;
      break;
    } catch (err) {
      if (err?.code !== "EEXIST") throw err;
      let ageMs = 0;
      try {
        ageMs = Date.now() - (await stat(dir)).mtimeMs;
      } catch {
        continue; // holder released between mkdir and stat; retry immediately
      }
      if (ageMs > staleMs) {
        await rmdir(dir).catch(() => {});
        continue;
      }
      if (Date.now() >= deadline) break;
      await new Promise((resolve) => setTimeout(resolve, LOCK_POLL_MS));
    }
  }
  if (!held) return { skipped: true };
  const heartbeat = async () => {
    const now = new Date();
    await utimes(dir, now, now).catch(() => {});
  };
  try {
    return { skipped: false, value: await fn({ heartbeat }) };
  } finally {
    await rmdir(dir).catch(() => {});
  }
}

export async function listStates() {
  try {
    const dir = getStateDir();
    const files = await readdir(dir);
    const out = [];
    for (const file of files) {
      // .json only — atomic writes briefly create `<id>.json.tmp`, skipped
      // by this check (endsWith(".json") is false for ".json.tmp").
      if (!file.endsWith(".json")) continue;
      try {
        const raw = await readFile(join(dir, file), "utf-8");
        const parsed = JSON.parse(raw);
        if (!parsed?.codexSessionId) continue;
        const endedAt = await readEndedAt(parsed.codexSessionId);
        out.push(endedAt ? { ...parsed, endedAt } : parsed);
      } catch { /* skip */ }
    }
    return out;
  } catch {
    return [];
  }
}
