import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import { buildUserAgent, resolveOpenVikingCredentials } from "./credentials.mjs";
import { createLogger } from "./debug-log.mjs";
import { sendSessionMessages } from "./batch-send.mjs";
import { enqueue, replayPending } from "./pending-queue.mjs";
import { buildProfileBlock } from "./profile-inject.mjs";
import { buildRecallBlock, isRecallEnabled } from "./recall-core.mjs";
import { resolveSettings } from "./plugin-config.mjs";
import { isRetryableFailure } from "./retryable.mjs";
import { deriveHarnessSessionId, isBypassed } from "./session-model.mjs";
import { resolveEffectivePeerId, resolvePluginPeerId } from "./workspace-peer.mjs";

const STATE_VERSION = 1;
const STATE_DIR_MODE = 0o700;
const STATE_FILE_MODE = 0o600;

function safePart(value) {
  return String(value || "unknown").replace(/[^A-Za-z0-9._-]/g, "-");
}

function responseTraceId(body) {
  return body?.result?.trace_id || body?.error?.trace_id || body?.trace_id || undefined;
}

export function stableHash(...values) {
  return createHash("sha256")
    .update(values.map((value) => String(value ?? "")).join("\n"))
    .digest("hex");
}

/**
 * The config for one of the thin hook harnesses.
 *
 * These four used to read the environment and nothing else, so `ov config
 * switch` moved their credentials and left their behaviour behind, and an
 * `ovcli.conf` `plugin` entry named after them was inert. They resolve through
 * the same layers as every other harness now; only the log path, which is named
 * after the client, stays local.
 *
 * `cwd` selects the workspace layer. It defaults to this process's directory,
 * which is all a hook knows before the payload on stdin names the session's
 * own — the caller re-resolves once it has it. That is safe because a workspace
 * file may not carry connection or credential keys, so the base URL and API key
 * cannot move under a logger or fetch helper already built from the first load.
 */
export function loadAgentHookConfig(clientId, cwd = process.cwd()) {
  const credentials = resolveOpenVikingCredentials(process.env, clientId);
  const { settings, configured, sources } = resolveSettings(clientId, { env: process.env, cwd });
  return {
    ...settings,
    ...credentials,
    // The credential chain has no view of ovcli.conf's `plugin` section, which
    // is where a key set through `plugin.<harness>.apiKey` lives; spreading an
    // empty one over it would make that key inert.
    apiKey: credentials.apiKey || settings.apiKey,
    peerId: resolvePluginPeerId({ settings, configured, sources, credentials }),
    clientId,
    userAgent: buildUserAgent(clientId, process.env.OPENVIKING_INTEGRATION_VERSION),
    recallLimitConfigured: configured.has("recallLimit"),
    recallQueryExpansionConfigured: configured.has("recallQueryExpansion"),
    debugLogPath: settings.debugLogPath
      || join(homedir(), ".openviking", "logs", `${clientId}-hooks.log`),
  };
}

export function createAgentLogger(clientId, hookName, cfg) {
  return createLogger(`${clientId}:${hookName}`, cfg);
}

export async function readHookInput() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString();
  if (!raw.trim()) return {};
  try { return JSON.parse(raw); } catch { return {}; }
}

export function resolveAgentCwd(input = {}) {
  const workspaceRoots = Array.isArray(input.workspace_roots)
    ? input.workspace_roots
    : Array.isArray(input.workspaceRoots) ? input.workspaceRoots : [];
  return String(
    input.cwd
      || workspaceRoots.find((value) => typeof value === "string" && value.trim())
      || process.env.CURSOR_PROJECT_DIR
      || process.cwd(),
  );
}

export function resolveNativeSessionId(input = {}) {
  const direct = input.conversation_id || input.session_id || input.sessionId || input.generation_id;
  if (direct) return safePart(direct);
  const transcript = input.transcript_path || input.transcriptPath;
  if (transcript) {
    const match = String(transcript).match(/([0-9a-f]{8}-[0-9a-f-]{20,})/i);
    return safePart(match?.[1] || stableHash(transcript).slice(0, 24));
  }
  const cwd = resolveAgentCwd(input);
  return `cwd-${stableHash(cwd).slice(0, 20)}`;
}

export function deriveAgentSessionId(prefix, input = {}) {
  return deriveHarnessSessionId(prefix, resolveNativeSessionId(input));
}

function statePath(clientId, nativeSessionId) {
  const root = process.env.OPENVIKING_HOOK_STATE_DIR
    || join(homedir(), ".openviking", "hook-state");
  return join(root, safePart(clientId), `${safePart(nativeSessionId)}.json`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withAgentHookLock(clientId, nativeSessionId, callback) {
  const file = statePath(clientId, nativeSessionId);
  const lock = `${file}.lock`;
  await mkdir(dirname(file), { recursive: true, mode: STATE_DIR_MODE });
  const deadline = Date.now() + 5000;
  while (true) {
    try {
      await mkdir(lock, { mode: STATE_DIR_MODE });
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      try {
        if (Date.now() - (await stat(lock)).mtimeMs > 60_000) {
          await rm(lock, { recursive: true, force: true });
          continue;
        }
      } catch {}
      if (Date.now() >= deadline) return null;
      await sleep(50);
    }
  }
  try {
    return await callback();
  } finally {
    await rm(lock, { recursive: true, force: true }).catch(() => {});
  }
}

export async function readHookState(clientId, nativeSessionId) {
  try {
    const parsed = JSON.parse(await readFile(statePath(clientId, nativeSessionId), "utf8"));
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export async function writeHookState(clientId, nativeSessionId, value) {
  const file = statePath(clientId, nativeSessionId);
  await mkdir(dirname(file), { recursive: true, mode: STATE_DIR_MODE });
  const tmp = `${file}.${process.pid}.tmp`;
  await writeFile(tmp, `${JSON.stringify({ version: STATE_VERSION, ...value }, null, 2)}\n`, {
    encoding: "utf8",
    mode: STATE_FILE_MODE,
  });
  await rename(tmp, file);
}

export function makeAgentFetchJSON(cfg, cwd = process.cwd()) {
  const effectivePeer = resolveEffectivePeerId({ cfg, cwd });
  const fetchJSON = async (path, init = {}, options = {}) => {
    const controller = new AbortController();
    const timeoutMs = Math.max(1000, Number(options.timeoutMs) || cfg.timeoutMs);
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const headers = { "Content-Type": "application/json", ...(init.headers || {}) };
      if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`;
      if (cfg.account) headers["X-OpenViking-Account"] = cfg.account;
      if (cfg.user) headers["X-OpenViking-User"] = cfg.user;
      const peerId = options.actorPeerId ?? effectivePeer.peerId;
      if (peerId) headers["X-OpenViking-Actor-Peer"] = peerId;
      if (cfg.userAgent) headers["User-Agent"] = cfg.userAgent;
      const response = await fetch(`${cfg.baseUrl}${path}`, { ...init, headers, signal: controller.signal });
      const body = await response.json().catch(() => ({}));
      const traceId = responseTraceId(body);
      if (!response.ok || body.status === "error") {
        return { ok: false, status: response.status, error: body.error || body, traceId };
      }
      return { ok: true, result: body.result ?? body, traceId };
    } catch (error) {
      return { ok: false, status: 0, error: { message: error?.message || String(error) } };
    } finally {
      clearTimeout(timer);
    }
  };
  return { fetchJSON, effectivePeer };
}

export async function addAgentMessage(fetchJSON, sessionId, payload) {
  const result = await fetchJSON(`/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (!result.ok && isRetryableFailure(result)) await enqueue("addMessage", sessionId, payload);
  return result;
}

export async function addAgentMessages(fetchJSON, sessionId, payloads) {
  return sendSessionMessages(fetchJSON, sessionId, payloads, { enqueueOnRetryable: true });
}

export async function commitAgentSession(fetchJSON, sessionId, log = () => {}) {
  const result = await fetchJSON(`/api/v1/sessions/${encodeURIComponent(sessionId)}/commit`, {
    method: "POST",
    body: "{}",
  });
  let queued = false;
  if (!result.ok && isRetryableFailure(result)) {
    const pending = await enqueue("commitSession", sessionId, {});
    queued = Boolean(pending.ok);
  }
  log("commit", {
    sessionId,
    ok: result.ok,
    status: result.result?.status || result.status,
    trace_id: result.traceId || result.result?.trace_id,
    queued,
    error: result.ok ? undefined : result.error?.message || result.error?.code,
  });
  return result;
}

export async function replayAgentPending(fetchJSON, log = () => {}) {
  return replayPending(fetchJSON, log);
}

export async function recallForPrompt(fetchJSON, cfg, prompt, cwd, log = () => {}, options = {}) {
  if (!isRecallEnabled(cfg) || !String(prompt || "").trim()) return null;
  const peer = resolveEffectivePeerId({ cfg, cwd });
  return buildRecallBlock(fetchJSON, cfg, prompt, {
    actorPeerId: peer.peerId,
    legacyPeerId: peer.legacyPeerId,
    // Passing the OV session id is what turns on server-side query expansion
    // and the cross-turn dedup ledger for these thin harnesses.
    sessionId: options.sessionId || "",
    log,
  });
}

export async function buildAgentProfile(fetchJSON, cfg, cwd) {
  const peer = resolveEffectivePeerId({ cfg, cwd });
  const profile = await buildProfileBlock(fetchJSON, cfg.profileTokenBudget, peer.peerId);
  return profile?.block || null;
}

export function shouldBypassAgent(cfg, input = {}) {
  return isBypassed(cfg, { sessionId: resolveNativeSessionId(input), cwd: resolveAgentCwd(input) });
}
