#!/usr/bin/env node

/**
 * Auto-Recall Hook Script for Claude Code (UserPromptSubmit).
 *
 * Injects a compact <openviking-context> hint block so the model can
 * decide which items to expand via the read_context MCP tool. The hint
 * block carries URI + abstract + score + type for each match, but NOT
 * the full content. The model expands what it actually wants.
 *
 * Multi-source search (P1-5): user memories, agent memories, agent skills,
 * resources — all searched in parallel and merged with per-type tagging.
 *
 * Ported from openclaw-plugin/ memory-ranking.ts (query profile + boosts)
 * and index.ts (auto-recall orchestration). The `<openviking-context>`
 * envelope is new (replaces `<relevant-memories>`) to signal to Claude
 * that these are references to expand, not pre-read content.
 */

import { isPluginEnabled, loadConfig } from "./config.mjs";
import { createLogger } from "./debug-log.mjs";
import { isBypassed, makeFetchJSON } from "./lib/ov-session.mjs";

if (!isPluginEnabled()) {
  process.stdout.write(JSON.stringify({ decision: "approve" }) + "\n");
  process.exit(0);
}

const cfg = loadConfig();
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

// ---------------------------------------------------------------------------
// Ranking (ported from openclaw-plugin/memory-ranking.ts)
// ---------------------------------------------------------------------------

function clampScore(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

const PREFERENCE_QUERY_RE = /prefer|preference|favorite|favourite|like|偏好|喜欢|爱好|更倾向/i;
const TEMPORAL_QUERY_RE = /when|what time|date|day|month|year|yesterday|today|tomorrow|last|next|什么时候|何时|哪天|几月|几年|昨天|今天|明天/i;
const QUERY_TOKEN_RE = /[a-z0-9一-龥]{2,}/gi;
const STOPWORDS = new Set([
  "what","when","where","which","who","whom","whose","why","how","did","does",
  "is","are","was","were","the","and","for","with","from","that","this","your","you",
]);

function buildQueryProfile(query) {
  const text = query.trim();
  const allTokens = text.toLowerCase().match(QUERY_TOKEN_RE) || [];
  const tokens = allTokens.filter(t => !STOPWORDS.has(t));
  return {
    tokens,
    wantsPreference: PREFERENCE_QUERY_RE.test(text),
    wantsTemporal: TEMPORAL_QUERY_RE.test(text),
  };
}

function lexicalOverlapBoost(tokens, text) {
  if (tokens.length === 0 || !text) return 0;
  const haystack = ` ${text.toLowerCase()} `;
  let matched = 0;
  for (const token of tokens.slice(0, 8)) {
    if (haystack.includes(token)) matched += 1;
  }
  return Math.min(0.2, (matched / Math.min(tokens.length, 4)) * 0.2);
}

function rankItem(item, profile) {
  const base = clampScore(item.score);
  const abstract = (item.abstract || item.overview || "").trim();
  const cat = (item.category || "").toLowerCase();
  const uri = (item.uri || "").toLowerCase();
  const leafBoost = (item.level === 2 || uri.endsWith(".md")) ? 0.12 : 0;
  const eventBoost = profile.wantsTemporal && (cat === "events" || uri.includes("/events/")) ? 0.1 : 0;
  const prefBoost = profile.wantsPreference && (cat === "preferences" || uri.includes("/preferences/")) ? 0.08 : 0;
  const overlapBoost = lexicalOverlapBoost(profile.tokens, `${item.uri} ${abstract}`);
  return base + leafBoost + eventBoost + prefBoost + overlapBoost;
}

/**
 * events/cases specialization (ported from openclaw-plugin/memory-ranking.ts
 * isEventOrCaseMemory): these items describe unique occurrences — dedupe by
 * URI instead of abstract since their abstracts often collide.
 */
function isEventOrCaseItem(item) {
  const cat = (item.category || "").toLowerCase();
  const uri = (item.uri || "").toLowerCase();
  return cat === "events" || cat === "cases" || uri.includes("/events/") || uri.includes("/cases/");
}

function dedupeItems(items) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    const key = isEventOrCaseItem(item)
      ? `uri:${item.uri}`
      : ((item.abstract || item.overview || "").trim().toLowerCase() || `uri:${item.uri}`);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

// ---------------------------------------------------------------------------
// URI space resolution (mirrors memory-server.ts normalizeTargetUri)
// ---------------------------------------------------------------------------

const USER_RESERVED_DIRS = new Set(["memories"]);
const AGENT_RESERVED_DIRS = new Set(["memories", "skills", "instructions", "workspaces"]);
const _spaceCache = {};

async function resolveScopeSpace(scope) {
  if (_spaceCache[scope]) return _spaceCache[scope];

  let fallbackSpace = "default";
  const status = await fetchJSON("/api/v1/system/status");
  if (status.ok && typeof status.result?.user === "string" && status.result.user.trim()) {
    fallbackSpace = status.result.user.trim();
  }

  const reservedDirs = scope === "user" ? USER_RESERVED_DIRS : AGENT_RESERVED_DIRS;
  const lsRes = await fetchJSON(`/api/v1/fs/ls?uri=${encodeURIComponent(`viking://${scope}`)}&output=original`);
  if (lsRes.ok && Array.isArray(lsRes.result)) {
    const spaces = lsRes.result
      .filter(e => e?.isDir)
      .map(e => (typeof e.name === "string" ? e.name.trim() : ""))
      .filter(n => n && !n.startsWith(".") && !reservedDirs.has(n));
    if (spaces.length > 0) {
      if (spaces.includes(fallbackSpace)) { _spaceCache[scope] = fallbackSpace; return fallbackSpace; }
      if (scope === "user" && spaces.includes("default")) { _spaceCache[scope] = "default"; return "default"; }
      if (spaces.length === 1) { _spaceCache[scope] = spaces[0]; return spaces[0]; }
    }
  }
  _spaceCache[scope] = fallbackSpace;
  return fallbackSpace;
}

async function resolveTargetUri(targetUri) {
  const trimmed = targetUri.trim().replace(/\/+$/, "");
  const m = trimmed.match(/^viking:\/\/(user|agent)(?:\/(.*))?$/);
  if (!m) return trimmed;
  const scope = m[1];
  const rawRest = (m[2] ?? "").trim();
  if (!rawRest) return trimmed;
  const parts = rawRest.split("/").filter(Boolean);
  if (parts.length === 0) return trimmed;
  const reservedDirs = scope === "user" ? USER_RESERVED_DIRS : AGENT_RESERVED_DIRS;
  if (!reservedDirs.has(parts[0])) return trimmed;
  const space = await resolveScopeSpace(scope);
  return `viking://${scope}/${space}/${parts.join("/")}`;
}

// ---------------------------------------------------------------------------
// Multi-source search
// ---------------------------------------------------------------------------

const SOURCES = [
  { type: "memory",   uri: "viking://user/memories",  bucket: "memories" },
  { type: "memory",   uri: "viking://agent/memories", bucket: "memories" },
  { type: "skill",    uri: "viking://agent/skills",   bucket: "skills"   },
  { type: "resource", uri: "viking://resources",      bucket: "resources"},
];

async function searchOneSource(query, source, limit) {
  const resolvedUri = await resolveTargetUri(source.uri);
  const res = await fetchJSON("/api/v1/search/find", {
    method: "POST",
    body: JSON.stringify({ query, target_uri: resolvedUri, limit, score_threshold: 0 }),
  });
  if (!res.ok) return [];
  const items = res.result?.[source.bucket] || [];
  return items.map(item => ({ ...item, _sourceType: source.type }));
}

async function searchAllSources(query, perSourceLimit) {
  const results = await Promise.all(SOURCES.map(src => searchOneSource(query, src, perSourceLimit)));
  const all = results.flat();
  log("search_summary", {
    counts: SOURCES.map((src, i) => ({ type: src.type, uri: src.uri, count: results[i].length })),
    total: all.length,
  });
  return all;
}

// ---------------------------------------------------------------------------
// Hint-mode injection formatting (P1-6)
// ---------------------------------------------------------------------------

function escapeAttr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function formatHintBlock(items, query) {
  if (items.length === 0) return null;

  const lines = [
    "<openviking-context>",
    "  The OpenViking Context Database has items related to the user's prompt.",
    "  Each hint below has a URI + abstract + score. To read the full body,",
    "  call the read_context MCP tool with the URI.",
  ];
  for (const it of items) {
    const score = clampScore(it.score).toFixed(2);
    const abstract = (it.abstract || it.overview || "").trim().replace(/\s+/g, " ").slice(0, 200);
    lines.push(
      `  <item type="${it._sourceType}" score="${score}" uri="${escapeAttr(it.uri)}" abstract="${escapeAttr(abstract)}"/>`,
    );
  }
  lines.push("</openviking-context>");
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  if (!cfg.autoRecall) {
    log("skip", { reason: "autoRecall disabled" });
    approve();
    return;
  }

  let input;
  try {
    const chunks = [];
    for await (const chunk of process.stdin) chunks.push(chunk);
    input = JSON.parse(Buffer.concat(chunks).toString());
  } catch {
    log("skip", { reason: "invalid stdin" });
    approve();
    return;
  }

  const userPrompt = (input.prompt || "").trim();
  const sessionId = input.session_id;
  const cwd = input.cwd;
  log("start", {
    query: userPrompt.slice(0, 200),
    queryLength: userPrompt.length,
    config: { recallLimit: cfg.recallLimit, scoreThreshold: cfg.scoreThreshold },
  });

  if (isBypassed(cfg, { sessionId, cwd })) {
    log("skip", { reason: "bypass_session_pattern" });
    approve();
    return;
  }

  if (!userPrompt || userPrompt.length < cfg.minQueryLength) {
    log("skip", { reason: "query too short or empty" });
    approve();
    return;
  }

  const health = await fetchJSON("/health");
  if (!health.ok) {
    logError("health_check", "server unreachable");
    approve();
    return;
  }

  // Per-source over-fetch so ranking can pick from a broader pool.
  const perSourceLimit = Math.max(cfg.recallLimit * 2, 8);
  const raw = await searchAllSources(userPrompt, perSourceLimit);
  if (raw.length === 0) {
    log("skip", { reason: "no results" });
    approve();
    return;
  }

  const profile = buildQueryProfile(userPrompt);
  // Filter by threshold first, rank, then dedupe.
  const filtered = raw.filter(it => clampScore(it.score) >= cfg.scoreThreshold);
  filtered.sort((a, b) => rankItem(b, profile) - rankItem(a, profile));
  const deduped = dedupeItems(filtered);
  const picked = deduped.slice(0, cfg.recallLimit);
  log("picked", {
    rawCount: raw.length,
    filteredCount: filtered.length,
    dedupedCount: deduped.length,
    pickedCount: picked.length,
    items: picked.map(it => ({ type: it._sourceType, uri: it.uri, score: clampScore(it.score) })),
  });

  if (picked.length === 0) {
    approve();
    return;
  }

  const block = formatHintBlock(picked, userPrompt);
  approve(block);
}

main().catch((err) => { logError("uncaught", err); approve(); });
