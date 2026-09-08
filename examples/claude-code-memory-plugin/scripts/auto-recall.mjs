#!/usr/bin/env node

/**
 * Auto-Recall Hook Script for Claude Code (UserPromptSubmit).
 *
 * Searches OpenViking for relevant context and injects an
 * <openviking-context> block. Retrieval, ranking and the token budget are the
 * shared recall core's; this hook owns the CC envelope and the statusline
 * snapshot it leaves behind.
 */

import { isPluginEnabled, loadConfig } from "./config.mjs";
import { createLogger } from "./debug-log.mjs";
import { deriveOvSessionId, isBypassed, makeFetchJSON } from "./lib/ov-session.mjs";
import { writeJsonState } from "./lib/state.mjs";
import { createHostCompressor } from "./lib/host-compressor.mjs";
import { getEffectivePeerId } from "./lib/workspace-peer.mjs";
import { buildRecallBlockDetailed } from "./shared/recall-core.mjs";

if (!isPluginEnabled()) {
  process.stdout.write(JSON.stringify({ decision: "approve" }) + "\n");
  process.exit(0);
}

let cfg = loadConfig();
const { log, logError } = createLogger("auto-recall");
const fetchJSON = makeFetchJSON(cfg);

function output(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function approve(msg) {
  const out = { decision: "approve" };
  if (msg) out.hookSpecificOutput = { hookEventName: "UserPromptSubmit", additionalContext: msg };
  output(out);
}

const URI_RE = /viking:\/\/[^\s<>"')\]]+/g;

async function recall(query, peer, sessionId) {
  const runCompressor = await createHostCompressor(cfg, log);
  return buildRecallBlockDetailed(fetchJSON, cfg, query, {
    actorPeerId: peer.peerId,
    legacyPeerId: peer.legacyPeerId,
    sessionId,
    log,
    runCompressor,
    localCompressorAvailable: Boolean(runCompressor),
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const t0 = Date.now();
  // Snapshot state for the statusline. Always written, even on early-exit
  // branches, so the indicator reflects the latest turn rather than stale
  // data from the previous run.
  const writeRecallState = (extra) => writeJsonState("last-recall.json", {
    server_url: cfg.baseUrl,
    latency_ms: Date.now() - t0,
    ...extra,
  });

  let input;
  try {
    const chunks = [];
    for await (const chunk of process.stdin) chunks.push(chunk);
    input = JSON.parse(Buffer.concat(chunks).toString());
  } catch {
    log("skip", { reason: "invalid stdin" });
    writeRecallState({ count: 0, reason: "bad_stdin" });
    approve();
    return;
  }

  const userPrompt = (input.prompt || "").trim();
  const sessionId = input.session_id;
  const cwd = input.cwd;
  // The workspace layer belongs to the session's directory, which only the
  // payload knows; see loadConfig for why re-resolving this late is safe.
  // Everything gated below — recall.enabled included — reads the reload.
  cfg = loadConfig(cwd);

  if (!cfg.autoRecall) {
    log("skip", { reason: "autoRecall disabled" });
    writeRecallState({ count: 0, reason: "disabled" });
    approve();
    return;
  }

  const effectivePeer = getEffectivePeerId(cfg, { sessionId, cwd });
  log("start", {
    query: userPrompt.slice(0, 200),
    queryLength: userPrompt.length,
    config: {
      recallLimit: cfg.recallLimit,
      scoreThreshold: cfg.scoreThreshold,
      recallMaxContentChars: cfg.recallMaxContentChars,
      recallTokenBudget: cfg.recallTokenBudget,
      peerSource: effectivePeer.source,
      recallPeerScope: cfg.recallPeerScope,
    },
  });

  if (isBypassed(cfg, { sessionId, cwd })) {
    log("skip", { reason: "bypass_session_pattern" });
    writeRecallState({ count: 0, reason: "bypass", cc_session_id: sessionId });
    approve();
    return;
  }

  if (!userPrompt || userPrompt.length < cfg.minQueryLength) {
    log("skip", { reason: "query too short or empty" });
    writeRecallState({ count: 0, reason: "short_query", cc_session_id: sessionId });
    approve();
    return;
  }

  const health = await fetchJSON("/health");
  if (!health.ok) {
    logError("health_check", "server unreachable");
    writeRecallState({ count: 0, reason: "offline", cc_session_id: sessionId });
    approve();
    return;
  }

  // The OV session id is what unlocks server-side query expansion and the
  // cross-turn dedup ledger; it must match the id auto-capture writes to.
  const ovSessionId = sessionId && sessionId !== "unknown" ? deriveOvSessionId(sessionId) : "";
  const recalled = await recall(userPrompt, effectivePeer, ovSessionId);
  if (!recalled.block) {
    log("skip", { reason: recalled.stage });
    writeRecallState({ count: 0, reason: recalled.stage, cc_session_id: sessionId });
    approve();
    return;
  }

  writeRecallState({
    // A server-assembled block is one rendered unit whatever it holds, so the
    // count the statusline shows comes from the URIs it cites.
    count: recalled.stage === "server_assembled"
      ? new Set(recalled.block.match(URI_RE) || []).size
      : recalled.contentCount + recalled.hintCount,
    content_items: recalled.contentCount,
    hint_items: recalled.hintCount,
    tokens_used: recalled.budgetUsed,
    tokens_budget: cfg.recallTokenBudget,
    cc_session_id: sessionId,
    reason: "ok",
  });
  approve(recalled.block);
}

main().catch((err) => { logError("uncaught", err); approve(); });
