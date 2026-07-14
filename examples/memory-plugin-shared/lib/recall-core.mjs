import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

const PREFERENCE_QUERY_RE = /prefer|preference|favorite|favourite|like|偏好|喜欢|爱好|更倾向/i;
const TEMPORAL_QUERY_RE = /when|what time|date|day|month|year|yesterday|today|tomorrow|last|next|什么时候|何时|哪天|几月|几年|昨天|今天|明天/i;
const QUERY_TOKEN_RE = /[a-z0-9一-龥]{2,}/gi;
const STOPWORDS = new Set([
  "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did", "does",
  "is", "are", "was", "were", "the", "and", "for", "with", "from", "that", "this", "your", "you",
]);
const USER_RESERVED_DIRS = new Set(["memories", "skills"]);
const SOURCES = [
  { type: "memory", uri: "viking://user/memories", bucket: "memories" },
  { type: "skill", uri: "viking://user/skills", bucket: "skills" },
];

let userSpaceCache = "";
const recallHistory = new Map();
const legacyServers = new Map();
const LEGACY_TTL_MS = 60 * 60 * 1000;

function normalizeRewrite(value) {
  const mode = String(value || "off").trim().toLowerCase();
  return ["client", "server", "auto"].includes(mode) ? mode : "off";
}

function sessionContextEnabled(value) {
  if (value === true) return true;
  return /^(?:1|true|yes|on|auto)$/i.test(String(value || ""));
}

export function truncateRenderedAtFragmentBoundary(rendered, maxChars = 6500) {
  const text = String(rendered || "").trim();
  const limit = Math.max(1, Number(maxChars) || 6500);
  if (text.length <= limit) return text;
  const fragments = text.match(/<memory\b[\s\S]*?<\/memory>/g) || [];
  if (fragments.length === 0) return text.slice(0, limit).trimEnd();
  const kept = [];
  let used = 0;
  for (const fragment of fragments) {
    const extra = fragment.length + (kept.length ? 1 : 0);
    if (used + extra > limit) break;
    kept.push(fragment);
    used += extra;
  }
  return kept.join("\n");
}

function truncateDigestAtLineBoundary(digest, maxChars) {
  const text = String(digest || "").trim();
  if (text.length <= maxChars) return text;
  const kept = [];
  let used = 0;
  for (const line of text.split(/\r?\n/)) {
    const extra = line.length + (kept.length ? 1 : 0);
    if (used + extra > maxChars) break;
    kept.push(line);
    used += extra;
  }
  return kept.join("\n").trim();
}

function wrapEndpointRecall(rendered, format, maxChars) {
  const opener = `<openviking-context source="auto-recall" format="${format}">`;
  const instruction = "Relevant memory from OpenViking. Use the recall/read MCP tools to expand URIs.";
  const closer = "</openviking-context>";
  const overhead = opener.length + instruction.length + closer.length + 3;
  const innerLimit = Math.max(1, maxChars - overhead);
  const inner = format === "digest"
    ? truncateDigestAtLineBoundary(rendered, innerLimit)
    : truncateRenderedAtFragmentBoundary(rendered, innerLimit);
  if (!inner) return "";
  return [opener, instruction, inner, closer].join("\n");
}

async function isLegacyServerCached(cacheKey, cachePath) {
  if (!cacheKey) return false;
  const inMemory = legacyServers.get(cacheKey);
  if (inMemory && Date.now() - inMemory < LEGACY_TTL_MS) return true;
  if (!cachePath) return false;
  try {
    const cached = JSON.parse(await readFile(cachePath, "utf8"));
    const checkedAt = Number(cached?.checkedAt || 0);
    if (cached?.cacheKey === cacheKey && Date.now() - checkedAt < LEGACY_TTL_MS) {
      legacyServers.set(cacheKey, checkedAt);
      return true;
    }
  } catch { /* cache miss */ }
  return false;
}

async function markLegacyServer(cacheKey, cachePath) {
  if (!cacheKey) return;
  const checkedAt = Date.now();
  legacyServers.set(cacheKey, checkedAt);
  if (!cachePath) return;
  try {
    await mkdir(dirname(cachePath), { recursive: true });
    const tmp = `${cachePath}.${process.pid}.tmp`;
    await writeFile(tmp, JSON.stringify({ cacheKey, checkedAt }));
    await rename(tmp, cachePath);
  } catch { /* best effort */ }
}

async function recallState(sessionId, dedupTurns, historyPath = "") {
  if (!sessionId || dedupTurns <= 0) return null;
  let state = recallHistory.get(sessionId);
  if (!state && historyPath) {
    try {
      const saved = JSON.parse(await readFile(historyPath, "utf8"));
      if (saved?.sessionId === sessionId) {
        state = {
          turn: Number(saved.turn || 0),
          uris: new Map(Object.entries(saved.uris || {})),
          historyPath,
          sessionId,
        };
      }
    } catch { /* cache miss */ }
  }
  if (!state) {
    state = { turn: 0, uris: new Map(), historyPath, sessionId };
    recallHistory.set(sessionId, state);
  }
  state.turn += 1;
  return state;
}

function exclusionsFor(state, dedupTurns) {
  if (!state) return [];
  const excluded = [];
  for (const [uri, record] of state.uris) {
    if (state.turn - record.turn > dedupTurns) {
      state.uris.delete(uri);
      continue;
    }
    if (record.mode !== "uri" || record.uriOnlyGraceUsed) excluded.push(uri);
  }
  return excluded.slice(0, 200);
}

async function rememberEntries(state, entries = []) {
  if (!state) return;
  for (const entry of entries) {
    const uri = String(entry?.uri || "").trim();
    if (!uri) continue;
    const previous = state.uris.get(uri);
    state.uris.set(uri, {
      turn: state.turn,
      mode: String(entry?.mode || "full"),
      uriOnlyGraceUsed: previous?.mode === "uri" || previous?.uriOnlyGraceUsed === true,
    });
  }
  if (state.historyPath) {
    try {
      await mkdir(dirname(state.historyPath), { recursive: true });
      const tmp = `${state.historyPath}.tmp`;
      await writeFile(tmp, JSON.stringify({
        sessionId: state.sessionId,
        turn: state.turn,
        uris: Object.fromEntries(state.uris),
      }));
      await rename(tmp, state.historyPath);
    } catch { /* best effort */ }
  }
}

export function estimateTokens(text) {
  return text ? Math.ceil(String(text).length / 4) : 0;
}

export function buildRecallEndpointBody(cfg = {}, options = {}) {
  const limit = Math.max(Number(cfg.recallLimit || 0), 1);
  const rewrite = normalizeRewrite(cfg.recallRewrite);
  const hasClientCompressor = typeof options.compress === "function";
  const clientRewrite = (rewrite === "client" && hasClientCompressor)
    || (rewrite === "auto" && hasClientCompressor);
  const serverRewrite = rewrite === "server" || (rewrite === "auto" && !hasClientCompressor);
  const recallMaxChars = Math.max(Number(cfg.recallMaxChars || 6500), 1000);
  const rewriteInputChars = Math.max(Number(cfg.recallCompressMaxInputChars || 18000), recallMaxChars);
  const body = {
    query: "",
    quotas: {
      events: limit,
      entities: limit,
      preferences: Math.max(1, Math.min(limit, 3)),
      experiences: 0,
    },
    max_chars: clientRewrite || serverRewrite ? rewriteInputChars : recallMaxChars,
    min_score: Number.isFinite(Number(cfg.scoreThreshold)) ? Number(cfg.scoreThreshold) : 0.35,
    render: clientRewrite || serverRewrite ? true : "compact",
  };
  if (cfg.recallPeerScope === "actor") body.peer_scope = "actor";
  if (serverRewrite) {
    body.rewrite = rewrite === "auto" ? "auto" : true;
    body.rewrite_max_bullets = Math.max(1, Number(cfg.recallCompressMaxBullets || 6));
  }
  if (sessionContextEnabled(cfg.recallSessionContext) && options.sessionId) {
    body.session_id = options.sessionId;
    body.query_expansion = "auto";
  }
  if (Array.isArray(options.excludeUris) && options.excludeUris.length) {
    body.exclude_uris = options.excludeUris.slice(0, 200);
  }
  return body;
}

function clampScore(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function buildQueryProfile(query) {
  const text = query.trim();
  const allTokens = text.toLowerCase().match(QUERY_TOKEN_RE) || [];
  return {
    tokens: allTokens.filter((t) => !STOPWORDS.has(t)),
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

async function resolveUserSpace(fetchJSON, actorPeerId = "") {
  if (userSpaceCache) return userSpaceCache;

  let fallbackSpace = "default";
  const status = await fetchJSON("/api/v1/system/status");
  if (status.ok && typeof status.result?.user === "string" && status.result.user.trim()) {
    fallbackSpace = status.result.user.trim();
  }

  const lsRes = await fetchJSON(
    `/api/v1/fs/ls?uri=${encodeURIComponent("viking://user")}&output=original`,
    {},
    { actorPeerId },
  );
  if (lsRes.ok && Array.isArray(lsRes.result)) {
    const spaces = lsRes.result
      .filter((e) => e?.isDir)
      .map((e) => (typeof e.name === "string" ? e.name.trim() : ""))
      .filter((n) => n && !n.startsWith(".") && !USER_RESERVED_DIRS.has(n));
    if (spaces.length > 0) {
      if (spaces.includes(fallbackSpace)) { userSpaceCache = fallbackSpace; return fallbackSpace; }
      if (spaces.includes("default")) { userSpaceCache = "default"; return "default"; }
      if (spaces.length === 1) { userSpaceCache = spaces[0]; return spaces[0]; }
    }
  }
  userSpaceCache = fallbackSpace;
  return fallbackSpace;
}

async function resolveTargetUri(fetchJSON, targetUri, actorPeerId = "", configuredUser = "") {
  const trimmed = targetUri.trim().replace(/\/+$/, "");
  const m = trimmed.match(/^viking:\/\/user(?:\/(.*))?$/);
  if (!m) return trimmed;
  const rawRest = (m[1] ?? "").trim();
  if (!rawRest) return trimmed;
  const parts = rawRest.split("/").filter(Boolean);
  if (parts.length === 0) return trimmed;
  if (!USER_RESERVED_DIRS.has(parts[0])) return trimmed;
  if (configuredUser) return `viking://user/${configuredUser}/${parts.join("/")}`;
  return trimmed;
}

async function searchOneSource(fetchJSON, query, source, limit, actorPeerId = "", sessionId = "", configuredUser = "") {
  const resolvedUri = await resolveTargetUri(fetchJSON, source.uri, actorPeerId, configuredUser);
  const body = { query, target_uri: resolvedUri, limit, score_threshold: 0 };
  if (sessionId) {
    const contextual = await fetchJSON("/api/v1/search/search", {
      method: "POST",
      body: JSON.stringify({ ...body, session_id: sessionId }),
    }, { actorPeerId });
    const contextualItems = contextual.ok ? (contextual.result?.[source.bucket] || []) : [];
    if (contextualItems.length > 0) {
      return contextualItems.map((item) => ({ ...item, _sourceType: source.type }));
    }
  }
  const res = await fetchJSON(sessionId ? "/api/v1/search/search" : "/api/v1/search/find", {
    method: "POST",
    body: JSON.stringify(body),
  }, { actorPeerId });
  if (!res.ok) return [];
  const items = res.result?.[source.bucket] || [];
  return items.map((item) => ({ ...item, _sourceType: source.type }));
}

async function searchAllSources(fetchJSON, query, perSourceLimit, actorPeerId = "", log = () => {}, sessionId = "", configuredUser = "") {
  const results = await Promise.all(
    SOURCES.map((src) => searchOneSource(fetchJSON, query, src, perSourceLimit, actorPeerId, sessionId, configuredUser)),
  );
  const all = results.flat();
  log("recall_search_summary", {
    counts: SOURCES.map((src, i) => ({ type: src.type, uri: src.uri, count: results[i].length })),
    total: all.length,
  });
  return all;
}

async function resolveItemContent(fetchJSON, item, cfg, actorPeerId = "") {
  let content;

  if (cfg.recallPreferAbstract && (item.abstract || item.overview || "").trim()) {
    content = (item.abstract || item.overview).trim();
  } else if (item.level === 2) {
    try {
      const res = await fetchJSON(
        `/api/v1/content/read?uri=${encodeURIComponent(item.uri)}`,
        {},
        { actorPeerId },
      );
      const body = res.ok && typeof res.result === "string" ? res.result.trim() : "";
      content = body || (item.abstract || item.overview || "").trim() || item.uri;
    } catch {
      content = (item.abstract || item.overview || "").trim() || item.uri;
    }
  } else {
    content = (item.abstract || item.overview || "").trim() || item.uri;
  }

  const maxChars = Math.max(50, Number(cfg.recallMaxContentChars || 500));
  if (content.length > maxChars) content = `${content.slice(0, maxChars)}...`;
  return content;
}

async function buildFallbackInjectionBlock(fetchJSON, items, cfg, actorPeerId = "", log = () => {}) {
  if (items.length === 0) return null;

  let budgetRemaining = Math.max(200, Number(cfg.recallTokenBudget || 2000));
  const lines = [
    "<openviking-context>",
    "Relevant context from OpenViking. Use the read MCP tool to expand URIs.",
  ];
  let contentCount = 0;
  let hintCount = 0;

  for (const item of items) {
    const score = (clampScore(item.score) * 100).toFixed(0);
    const uriLine = `- [${item._sourceType} ${score}%] ${item.uri}`;

    if (budgetRemaining > 0) {
      const content = await resolveItemContent(fetchJSON, item, cfg, actorPeerId);
      const contentLine = `- [${item._sourceType} ${score}%] ${content}`;
      const lineTokens = estimateTokens(contentLine);

      if (lineTokens > budgetRemaining && contentCount > 0) {
        lines.push(uriLine);
        hintCount++;
      } else {
        lines.push(contentLine);
        budgetRemaining -= lineTokens;
        contentCount++;
      }
    } else {
      lines.push(uriLine);
      hintCount++;
    }
  }

  const closer = "</openviking-context>";
  const maxChars = Math.max(1000, Number(cfg.recallMaxChars || 6500));
  const capped = lines.slice(0, 2);
  let chars = capped.join("\n").length + closer.length + 2;
  for (const line of lines.slice(2)) {
    const extra = line.length + 1;
    if (chars + extra > maxChars) break;
    capped.push(line);
    chars += extra;
  }
  capped.push(closer);

  const budgetUsed = Math.max(200, Number(cfg.recallTokenBudget || 2000)) - budgetRemaining;
  log("recall_injection_built", {
    contentItems: contentCount,
    hintItems: hintCount,
    budgetUsed,
    budgetTotal: Math.max(200, Number(cfg.recallTokenBudget || 2000)),
  });

  return capped.length > 3 ? capped.join("\n") : null;
}

async function recallViaEndpoint(fetchJSON, cfg, query, actorPeerId = "", log = () => {}, options = {}) {
  const body = buildRecallEndpointBody(cfg, options);
  body.query = query;
  const res = await postRecall(fetchJSON, body, {
    actorPeerId,
    log,
    legacyCacheKey: options.legacyCacheKey || cfg.baseUrl || cfg.endpoint || "",
    legacyCachePath: options.legacyCachePath || "",
  });
  if (!res.ok) {
    log("recall_endpoint_fallback", { status: res.status || 0 });
    return null;
  }
  const entries = Array.isArray(res.result?.entries) ? res.result.entries : [];
  await rememberEntries(options.recallState, entries);
  const digest = String(res.result?.digest || "").trim();
  let rendered = String(res.result?.rendered || "").trim();
  if (digest) rendered = digest;
  else if (typeof options.compress === "function" && rendered) {
    try {
      const compressed = await options.compress(rendered, { query, entries });
      if (compressed === "") return "";
      if (compressed) rendered = String(compressed).trim();
      else rendered = truncateRenderedAtFragmentBoundary(rendered, cfg.recallMaxChars || 6500);
    } catch (error) {
      log("recall_compress_failed", { error: String(error?.message || error) });
      rendered = truncateRenderedAtFragmentBoundary(rendered, cfg.recallMaxChars || 6500);
    }
  } else {
    rendered = truncateRenderedAtFragmentBoundary(rendered, cfg.recallMaxChars || 6500);
  }
  if (!rendered) return "";
  const format = digest || /^OpenViking memory digest:/i.test(rendered) ? "digest" : "memory";
  return wrapEndpointRecall(
    rendered,
    format,
    Math.max(1000, Number(cfg.recallMaxChars || 6500)),
  );
}

export function recallRequestTimeoutMs(body = {}) {
  // Mirror the server-side LLM fuses (recall_rewrite_timeout_s=20,
  // recall_intent_timeout_s=10) plus search/network margin so the client
  // does not abort while the server is still inside its own budget.
  let extraMs = 0;
  if (body.rewrite) extraMs += 20000;
  if (body.query_expansion === "auto") extraMs += 10000;
  return extraMs ? extraMs + 15000 : 0;
}

export async function postRecall(fetchJSON, body, opts = {}) {
  const actorPeerId = opts.actorPeerId || "";
  const log = opts.log || (() => {});
  const cacheKey = String(opts.legacyCacheKey || "");
  const cachePath = String(opts.legacyCachePath || "");
  const isLegacy = await isLegacyServerCached(cacheKey, cachePath);
  const request = isLegacy ? downgradeRecallBody(body) : { ...body };
  const timeoutMs = recallRequestTimeoutMs(request);
  const res = await fetchJSON("/api/v1/search/recall", {
    method: "POST",
    body: JSON.stringify(request),
  }, timeoutMs ? { actorPeerId, timeoutMs } : { actorPeerId });
  if (res.status !== 400 && res.status !== 422) {
    return res;
  }

  const downgraded = downgradeRecallBody(request);
  if (JSON.stringify(downgraded) === JSON.stringify(request)) return res;
  await markLegacyServer(cacheKey, cachePath);
  log("recall_legacy_downgrade", { status: res.status || 0 });
  return fetchJSON("/api/v1/search/recall", {
    method: "POST",
    body: JSON.stringify(downgraded),
  }, { actorPeerId });
}

function downgradeRecallBody(body) {
  const downgraded = { ...body };
  for (const field of [
    "session_id", "query_expansion", "exclude_uris", "rewrite", "rewrite_max_bullets", "peer_scope",
  ]) delete downgraded[field];
  if (Object.hasOwn(downgraded, "render") && typeof downgraded.render !== "boolean") {
    downgraded.render = true;
  }
  return downgraded;
}

export async function buildRecallBlock(fetchJSON, cfg, query, options = {}) {
  const actorPeerId = options.actorPeerId ?? cfg.peerId ?? "";
  const log = options.log || (() => {});
  const trimmed = String(query || "").trim();
  if (!trimmed) return null;

  const dedupTurns = Math.max(0, Number(cfg.recallDedupTurns ?? 5));
  const sessionId = String(options.sessionId || "");
  const state = await recallState(
    sessionId || options.dedupKey || "",
    dedupTurns,
    options.historyPath || "",
  );
  const endpointBlock = await recallViaEndpoint(fetchJSON, cfg, trimmed, actorPeerId, log, {
    ...options,
    sessionId,
    recallState: state,
    excludeUris: exclusionsFor(state, dedupTurns),
  });
  if (endpointBlock !== null) return endpointBlock || null;

  const recallLimit = Math.max(1, Number(cfg.recallLimit || 6));
  const perSourceLimit = Math.max(recallLimit * 2, 8);
  const raw = await searchAllSources(
    fetchJSON,
    trimmed,
    perSourceLimit,
    actorPeerId,
    log,
    sessionId,
    cfg.user || cfg.userId || "",
  );
  if (raw.length === 0) return null;

  const profile = buildQueryProfile(trimmed);
  const scoreThreshold = Number.isFinite(Number(cfg.scoreThreshold)) ? Number(cfg.scoreThreshold) : 0.35;
  const filtered = raw.filter((it) => clampScore(it.score) >= scoreThreshold);
  filtered.sort((a, b) => rankItem(b, profile) - rankItem(a, profile));
  const picked = dedupeItems(filtered).slice(0, recallLimit);
  await rememberEntries(state, picked.map((item) => ({ uri: item.uri, mode: "full" })));
  log("recall_picked", {
    rawCount: raw.length,
    filteredCount: filtered.length,
    pickedCount: picked.length,
    items: picked.map((it) => ({ type: it._sourceType, uri: it.uri, score: clampScore(it.score) })),
  });

  if (picked.length === 0) return null;
  return buildFallbackInjectionBlock(fetchJSON, picked, cfg, actorPeerId, log);
}
