/**
 * Pure transcript parser for Kimi Code CLI hook events.
 *
 * Checked against Kimi Code 0.41.0:
 *   - Hook stdin is snake_case JSON (session_id, cwd, hook_event_name).
 *   - The authoritative incremental transcript is the session wire log at
 *     $KIMI_CODE_HOME/sessions/<wd>/<session id>/agents/main/wire.jsonl, which
 *     ~/.kimi-code/session_index.jsonl points at.
 *   - User turns arrive as `turn.prompt` and `context.append_message`
 *     (role=user); assistant turns as `context.append_loop_event` text parts
 *     grouped by the host's `turnId`. Think parts are not transcript content.
 *   - `turn.ended` closes the pending user turn even when no assistant text
 *     followed, which is what an interrupted or tool-only turn looks like.
 *
 * The wire log is preferred so every capture observes stable host turn ids and
 * can recover a missed Stop; stdin plus the stashed prompt is the fallback for
 * a session whose wire log cannot be found.
 */

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const INJECTED_BLOCKS = [
  /<openviking-context\b[^>]*>[\s\S]*?<\/openviking-context>/gi,
  /<relevant-memories\b[^>]*>[\s\S]*?<\/relevant-memories>/gi,
  /<relevant-memory\b[^>]*>[\s\S]*?<\/relevant-memory>/gi,
  /<system-reminder\b[^>]*>[\s\S]*?<\/system-reminder>/gi,
];

/** Strip the blocks this plugin injected, so recall never captures itself. */
export function cleanKimicodeText(value) {
  if (value == null) return "";
  let text = String(value);
  for (const pattern of INJECTED_BLOCKS) text = text.replace(pattern, "");
  return text.replace(/\s+\n/g, "\n").trim();
}

export function kimiCodeHome() {
  return process.env.KIMI_CODE_HOME || join(homedir(), ".kimi-code");
}

function textFromContent(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => (typeof part === "string" ? part : typeof part?.text === "string" ? part.text : ""))
    .filter(Boolean)
    .join("\n");
}

/** The prompt text of a hook payload, however this host spelled it. */
export function kimicodePromptText(input = {}) {
  return (
    input.prompt
    || input.user_prompt
    || input.user_message
    || input.message
    || textFromContent(input.input)
    || ""
  );
}

function sessionIdOf(input = {}) {
  return String(input.session_id || input.sessionId || "").trim();
}

export function resolveWirePath(input = {}, home = kimiCodeHome()) {
  const sessionId = sessionIdOf(input);
  if (!sessionId) return "";
  const indexPath = join(home, "session_index.jsonl");
  if (existsSync(indexPath)) {
    try {
      for (const line of readFileSync(indexPath, "utf8").split("\n")) {
        if (!line.trim()) continue;
        const row = JSON.parse(line);
        if (row.sessionId === sessionId && row.sessionDir) {
          return join(row.sessionDir, "agents", "main", "wire.jsonl");
        }
      }
    } catch {
      // A truncated index is not a reason to give up; scan the directories.
    }
  }
  const sessionsRoot = join(home, "sessions");
  if (!existsSync(sessionsRoot)) return "";
  try {
    for (const wd of readdirSync(sessionsRoot)) {
      const candidate = join(sessionsRoot, wd, sessionId, "agents", "main", "wire.jsonl");
      if (existsSync(candidate)) return candidate;
    }
  } catch {
    return "";
  }
  return "";
}

/**
 * Every turn the wire log holds after `lastTurnId`, oldest first.
 *
 * `available` says whether the log could be read at all, which is what tells
 * the caller apart from a log that simply has nothing new.
 */
export function extractUnseenWireTurns(wirePath, lastTurnId = null) {
  if (!wirePath || !existsSync(wirePath)) return { available: false, turns: [] };
  let raw = "";
  try {
    raw = readFileSync(wirePath, "utf8");
  } catch {
    return { available: false, turns: [] };
  }

  const users = new Map();
  const assistants = new Map();
  const order = [];
  const seen = new Set();
  let pendingUser = "";

  const remember = (turnId) => {
    const id = String(turnId);
    if (!seen.has(id)) {
      seen.add(id);
      order.push(id);
    }
    return id;
  };

  // A user message is written before the host has named the turn it belongs to,
  // so it waits here until the first event that carries a turn id.
  const attachPending = (rawTurnId) => {
    const turnId = remember(rawTurnId ?? order.length);
    if (pendingUser) {
      users.set(turnId, users.has(turnId) ? `${users.get(turnId)}\n${pendingUser}` : pendingUser);
      pendingUser = "";
    }
    return turnId;
  };

  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    let obj;
    try {
      obj = JSON.parse(line);
    } catch {
      continue;
    }
    if (obj.type === "context.append_message" && obj.message?.role === "user") {
      pendingUser = textFromContent(obj.message.content) || pendingUser;
      continue;
    }
    if (obj.type === "turn.prompt") {
      // The same user turn is also recorded as append_message; this is only the
      // fallback for a log that skipped it.
      if (!pendingUser) pendingUser = textFromContent(obj.input);
      continue;
    }
    if (obj.type === "turn.ended") {
      attachPending(obj.turnId ?? obj.event?.turnId);
      continue;
    }
    if (obj.type !== "context.append_loop_event") continue;
    const event = obj.event || {};
    if (event.type === "turn.ended") {
      attachPending(event.turnId ?? obj.turnId);
      continue;
    }
    if (event.type === "content.part" && event.part?.type === "text") {
      const turnId = attachPending(event.turnId);
      const chunk = event.part.text || "";
      if (chunk) assistants.set(turnId, (assistants.get(turnId) || "") + chunk);
    }
  }

  const turns = [];
  let skipping = lastTurnId != null && lastTurnId !== "";
  for (const turnId of order) {
    if (skipping) {
      if (turnId === String(lastTurnId)) skipping = false;
      continue;
    }
    const user = cleanKimicodeText(users.get(turnId) || "");
    const assistant = cleanKimicodeText(assistants.get(turnId) || "");
    if (user) turns.push({ role: "user", content: user, turnId });
    if (assistant) turns.push({ role: "assistant", content: assistant, turnId });
  }
  return { available: true, turns };
}

/**
 * @param {object} input Hook stdin payload.
 * @param {object} state Persisted hook state (lastTurnId, pendingPrompt).
 * @returns {Array<{role: string, content: string, turnId?: string}>}
 */
export function buildKimicodeTurns(input = {}, state = {}) {
  const wire = extractUnseenWireTurns(resolveWirePath(input), state.lastTurnId || null);
  if (wire.available) return wire.turns;

  const assistantContent =
    input.responseText
    || input.responsePreview
    || input.last_assistant_message
    || input.assistant_message
    || "";
  const userContent = kimicodePromptText(input) || state.pendingPrompt?.prompt || "";
  return [
    { role: "user", content: cleanKimicodeText(userContent) },
    { role: "assistant", content: cleanKimicodeText(assistantContent) },
  ].filter((turn) => turn.content);
}
