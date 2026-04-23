/**
 * OpenViking Memory MCP Server for Claude Code
 *
 * Exposes OpenViking long-term memory as MCP tools:
 *   - memory_recall  : semantic search across memories
 *   - memory_store   : extract and persist new memories
 *   - memory_forget  : delete memories by URI or query
 *
 * Ported from the OpenClaw context-engine plugin (openclaw-plugin/).
 * Adapted for Claude Code's MCP server interface (stdio transport).
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { createHash } from "node:crypto";

// ---------------------------------------------------------------------------
// Types (ported from openclaw-plugin/client.ts)
// ---------------------------------------------------------------------------

type FindResultItem = {
  uri: string;
  level?: number;
  abstract?: string;
  overview?: string;
  category?: string;
  score?: number;
  match_reason?: string;
};

type FindResult = {
  memories?: FindResultItem[];
  resources?: FindResultItem[];
  skills?: FindResultItem[];
  total?: number;
};

type ScopeName = "user" | "agent";

// ---------------------------------------------------------------------------
// Configuration
//
// Resolution priority (highest → lowest):
//   1. Environment variables (OPENVIKING_URL, OPENVIKING_API_KEY, etc.)
//   2. ovcli.conf (CLI client config: url, api_key, account, user, agent_id)
//   3. ov.conf fields (server section + claude_code section)
//   4. Built-in defaults
//
// When no config file exists and no env-var override enables the plugin,
// the server exits silently (code 0) so Claude Code is not disrupted.
// ---------------------------------------------------------------------------

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve as resolvePath } from "node:path";

const DEFAULT_OV_CONF_PATH = join(homedir(), ".openviking", "ov.conf");
const DEFAULT_OVCLI_CONF_PATH = join(homedir(), ".openviking", "ovcli.conf");

function tryLoadJsonFile(envVar: string, defaultPath: string): Record<string, unknown> | null {
  const configPath = resolvePath(
    (process.env[envVar] || defaultPath).replace(/^~/, homedir()),
  );
  try {
    return JSON.parse(readFileSync(configPath, "utf-8"));
  } catch {
    return null;
  }
}

function envBool(name: string): boolean | undefined {
  const v = process.env[name];
  if (v == null || v === "") return undefined;
  const lower = v.trim().toLowerCase();
  if (lower === "0" || lower === "false" || lower === "no") return false;
  if (lower === "1" || lower === "true" || lower === "yes") return true;
  return undefined;
}

function num(val: unknown, fallback: number): number {
  if (typeof val === "number" && Number.isFinite(val)) return val;
  if (typeof val === "string" && val.trim()) {
    const n = Number(val);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function str(val: unknown, fallback: string): string {
  if (typeof val === "string" && val.trim()) return val.trim();
  return fallback;
}

// --- Enable/disable check ---
const envEnabled = envBool("OPENVIKING_MEMORY_ENABLED");
const ovFile = tryLoadJsonFile("OPENVIKING_CONFIG_FILE", DEFAULT_OV_CONF_PATH) ?? {};
const cliFile = tryLoadJsonFile("OPENVIKING_CLI_CONFIG_FILE", DEFAULT_OVCLI_CONF_PATH) ?? {};
const ovFileMissing = Object.keys(ovFile).length === 0;
const cliFileMissing = Object.keys(cliFile).length === 0;
const cc = (ovFile.claude_code ?? {}) as Record<string, unknown>;

if (envEnabled === false) {
  process.exit(0);
}
if (envEnabled === undefined) {
  if (ovFileMissing && cliFileMissing) process.exit(0);
  if (cc.enabled === false) process.exit(0);
}

// --- Build config: env → ovcli.conf → ov.conf → defaults ---
const serverCfg = (ovFile.server ?? {}) as Record<string, unknown>;
const cli = cliFile as Record<string, unknown>;

// baseUrl: env → ovcli.url → ov.server.url → http://{host}:{port}
const envUrl = str(process.env.OPENVIKING_URL as string, "") || str(process.env.OPENVIKING_BASE_URL as string, "");
let baseUrl: string;
if (envUrl) {
  baseUrl = envUrl.replace(/\/+$/, "");
} else if (cli.url) {
  baseUrl = str(cli.url, "").replace(/\/+$/, "");
} else if (serverCfg.url) {
  baseUrl = str(serverCfg.url, "").replace(/\/+$/, "");
} else {
  const host = str(serverCfg.host, "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
  const port = Math.floor(num(serverCfg.port, 1933));
  baseUrl = `http://${host}:${port}`;
}

const config = {
  baseUrl,
  apiKey: str(process.env.OPENVIKING_API_KEY as string, "") || str(cli.api_key, "") || str(cc.apiKey, "") || str(serverCfg.root_api_key, ""),
  agentId: str(process.env.OPENVIKING_AGENT_ID as string, "") || str(cli.agent_id, "") || str(cc.agentId, "claude-code"),
  accountId: str(process.env.OPENVIKING_ACCOUNT as string, "") || str(cli.account, "") || str(cc.accountId, ""),
  userId: str(process.env.OPENVIKING_USER as string, "") || str(cli.user, "") || str(cc.userId, ""),
  timeoutMs: Math.max(1000, Math.floor(num(cc.timeoutMs, 15000))),
  recallLimit: Math.max(1, Math.floor(num(cc.recallLimit, 6))),
  scoreThreshold: Math.min(1, Math.max(0, num(cc.scoreThreshold, 0.35))),
};

// ---------------------------------------------------------------------------
// OpenViking HTTP Client (ported from openclaw-plugin/client.ts)
// ---------------------------------------------------------------------------

const MEMORY_URI_PATTERNS = [
  /^viking:\/\/user\/(?:[^/]+\/)?memories(?:\/|$)/,
  /^viking:\/\/agent\/(?:[^/]+\/)?memories(?:\/|$)/,
];
const USER_STRUCTURE_DIRS = new Set(["memories"]);
const AGENT_STRUCTURE_DIRS = new Set(["memories", "skills", "instructions", "workspaces"]);

function md5Short(input: string): string {
  return createHash("md5").update(input).digest("hex").slice(0, 12);
}

function isMemoryUri(uri: string): boolean {
  return MEMORY_URI_PATTERNS.some((p) => p.test(uri));
}

class OpenVikingClient {
  private resolvedSpaceByScope: Partial<Record<ScopeName, string>> = {};
  private runtimeIdentity: { userId: string; agentId: string } | null = null;

  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly agentId: string,
    private readonly timeoutMs: number,
    private readonly accountId: string = "",
    private readonly userId: string = "",
  ) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers = new Headers(init.headers ?? {});
      if (this.apiKey) headers.set("X-API-Key", this.apiKey);
      if (this.accountId) headers.set("X-OpenViking-Account", this.accountId);
      if (this.userId) headers.set("X-OpenViking-User", this.userId);
      if (this.agentId) headers.set("X-OpenViking-Agent", this.agentId);
      if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");

      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
      });

      const payload = (await response.json().catch(() => ({}))) as {
        status?: string;
        result?: T;
        error?: { code?: string; message?: string };
      };

      if (!response.ok || payload.status === "error") {
        const code = payload.error?.code ? ` [${payload.error.code}]` : "";
        const message = payload.error?.message ?? `HTTP ${response.status}`;
        throw new Error(`OpenViking request failed${code}: ${message}`);
      }
      return (payload.result ?? payload) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.request<{ status: string }>("/health");
      return true;
    } catch {
      return false;
    }
  }

  private async ls(uri: string): Promise<Array<Record<string, unknown>>> {
    return this.request<Array<Record<string, unknown>>>(
      `/api/v1/fs/ls?uri=${encodeURIComponent(uri)}&output=original`,
    );
  }

  private async getRuntimeIdentity(): Promise<{ userId: string; agentId: string }> {
    if (this.runtimeIdentity) return this.runtimeIdentity;
    const fallback = { userId: "default", agentId: this.agentId || "default" };
    try {
      const status = await this.request<{ user?: unknown }>("/api/v1/system/status");
      const userId =
        typeof status.user === "string" && status.user.trim() ? status.user.trim() : "default";
      this.runtimeIdentity = { userId, agentId: this.agentId || "default" };
      return this.runtimeIdentity;
    } catch {
      this.runtimeIdentity = fallback;
      return fallback;
    }
  }

  private async resolveScopeSpace(scope: ScopeName): Promise<string> {
    const cached = this.resolvedSpaceByScope[scope];
    if (cached) return cached;

    const identity = await this.getRuntimeIdentity();
    const fallbackSpace =
      scope === "user" ? identity.userId : md5Short(`${identity.userId}:${identity.agentId}`);
    const reservedDirs = scope === "user" ? USER_STRUCTURE_DIRS : AGENT_STRUCTURE_DIRS;

    try {
      const entries = await this.ls(`viking://${scope}`);
      const spaces = entries
        .filter((e) => e?.isDir === true)
        .map((e) => (typeof e.name === "string" ? e.name.trim() : ""))
        .filter((n) => n && !n.startsWith(".") && !reservedDirs.has(n));

      if (spaces.length > 0) {
        if (spaces.includes(fallbackSpace)) {
          this.resolvedSpaceByScope[scope] = fallbackSpace;
          return fallbackSpace;
        }
        if (scope === "user" && spaces.includes("default")) {
          this.resolvedSpaceByScope[scope] = "default";
          return "default";
        }
        if (spaces.length === 1) {
          this.resolvedSpaceByScope[scope] = spaces[0]!;
          return spaces[0]!;
        }
      }
    } catch { /* fall through */ }

    this.resolvedSpaceByScope[scope] = fallbackSpace;
    return fallbackSpace;
  }

  private async normalizeTargetUri(targetUri: string): Promise<string> {
    const trimmed = targetUri.trim().replace(/\/+$/, "");
    const match = trimmed.match(/^viking:\/\/(user|agent)(?:\/(.*))?$/);
    if (!match) return trimmed;

    const scope = match[1] as ScopeName;
    const rawRest = (match[2] ?? "").trim();
    if (!rawRest) return trimmed;

    const parts = rawRest.split("/").filter(Boolean);
    if (parts.length === 0) return trimmed;

    const reservedDirs = scope === "user" ? USER_STRUCTURE_DIRS : AGENT_STRUCTURE_DIRS;
    if (!reservedDirs.has(parts[0]!)) return trimmed;

    const space = await this.resolveScopeSpace(scope);
    return `viking://${scope}/${space}/${parts.join("/")}`;
  }

  async find(
    query: string,
    options: { targetUri: string; limit: number; scoreThreshold?: number },
  ): Promise<FindResult> {
    const normalizedTargetUri = await this.normalizeTargetUri(options.targetUri);
    return this.request<FindResult>("/api/v1/search/find", {
      method: "POST",
      body: JSON.stringify({
        query,
        target_uri: normalizedTargetUri,
        limit: options.limit,
        score_threshold: options.scoreThreshold,
      }),
    });
  }

  async read(uri: string): Promise<string> {
    return this.request<string>(`/api/v1/content/read?uri=${encodeURIComponent(uri)}`);
  }

  async createSession(): Promise<string> {
    const result = await this.request<{ session_id: string }>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    });
    return result.session_id;
  }

  async addSessionMessage(sessionId: string, role: string, content: string): Promise<void> {
    await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
      method: "POST",
      body: JSON.stringify({ role, content }),
    });
  }

  async extractSessionMemories(sessionId: string): Promise<Array<Record<string, unknown>>> {
    return this.request<Array<Record<string, unknown>>>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/extract`,
      { method: "POST", body: JSON.stringify({}) },
    );
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  }

  async deleteUri(uri: string): Promise<void> {
    await this.request(`/api/v1/fs?uri=${encodeURIComponent(uri)}&recursive=false`, {
      method: "DELETE",
    });
  }

  /**
   * Commit a persistent session (archive + background extract).
   * Used by the new memory_store path; mirrors POST /sessions/{id}/commit.
   */
  async commitSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/commit`,
      { method: "POST", body: JSON.stringify({}) },
    );
  }

  /**
   * Get session meta including pending_tokens (used to decide when to commit).
   */
  async getSessionMeta(sessionId: string): Promise<Record<string, unknown> | null> {
    try {
      return await this.request<Record<string, unknown>>(
        `/api/v1/sessions/${encodeURIComponent(sessionId)}?auto_create=true`,
      );
    } catch {
      return null;
    }
  }

  /**
   * Thin wrapper around /api/v1/fs/ls used by list_context.
   */
  async lsUri(uri: string): Promise<Array<Record<string, unknown>>> {
    return this.request<Array<Record<string, unknown>>>(
      `/api/v1/fs/ls?uri=${encodeURIComponent(uri)}&output=original`,
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers (ported from openclaw-plugin/memory-ranking.ts)
// ---------------------------------------------------------------------------

function clampScore(value: number | undefined): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function normalizeDedupeText(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

/**
 * events/cases specialization (ported from openclaw-plugin/memory-ranking.ts
 * isEventOrCaseMemory): dedupe by URI instead of abstract.
 */
function isEventOrCaseMemory(item: FindResultItem): boolean {
  const cat = (item.category ?? "").toLowerCase();
  const uri = (item.uri ?? "").toLowerCase();
  return cat === "events" || cat === "cases" || uri.includes("/events/") || uri.includes("/cases/");
}

function getMemoryDedupeKey(item: FindResultItem): string {
  if (isEventOrCaseMemory(item)) return `uri:${item.uri}`;
  const abstract = normalizeDedupeText(item.abstract ?? item.overview ?? "");
  const category = (item.category ?? "").toLowerCase() || "unknown";
  if (abstract) return `abstract:${category}:${abstract}`;
  return `uri:${item.uri}`;
}

function postProcessMemories(
  items: FindResultItem[],
  options: { limit: number; scoreThreshold: number; leafOnly?: boolean },
): FindResultItem[] {
  const deduped: FindResultItem[] = [];
  const seen = new Set<string>();
  const sorted = [...items].sort((a, b) => clampScore(b.score) - clampScore(a.score));
  for (const item of sorted) {
    if (options.leafOnly && item.level !== 2) continue;
    if (clampScore(item.score) < options.scoreThreshold) continue;
    const key = getMemoryDedupeKey(item);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);
    if (deduped.length >= options.limit) break;
  }
  return deduped;
}

// ---------------------------------------------------------------------------
// MCP Server — 5 tools: search, read, store, forget, health
// Redesigned post-benchmark to reduce MCP call count and tool redundancy.
// Previous tools (memory_recall, search_context, read_context, list_context,
// memory_store, memory_forget, memory_health) replaced without backward compat.
// ---------------------------------------------------------------------------

const client = new OpenVikingClient(
  config.baseUrl, config.apiKey, config.agentId, config.timeoutMs,
  config.accountId, config.userId,
);

const server = new McpServer({
  name: "openviking",
  version: "0.2.0",
});

// -- Tool: search ---------------------------------------------------------
// Replaces both memory_recall and search_context. Returns URI + abstract +
// score (no full content reads — use read tool to expand).

const SEARCH_TARGETS: Record<string, string[]> = {
  memories: ["viking://user/memories", "viking://agent/memories"],
  resources: ["viking://resources"],
  skills: ["viking://agent/skills"],
};

server.tool(
  "search",
  "Search OpenViking context database. Auto-recall already injects top matches — use this for deeper or narrower searches. Prefer search over manual directory traversal.",
  {
    query: z.string().describe("What to search for"),
    scope: z
      .enum(["all", "memories", "resources", "skills"])
      .optional()
      .describe("Context type to search (default: all)"),
    limit: z.number().optional().describe("Max results (default: 6)"),
  },
  async ({ query, scope, limit }) => {
    const maxResults = limit ?? config.recallLimit;
    const scopes = scope && scope !== "all"
      ? [scope]
      : ["memories", "resources", "skills"] as const;

    type Tagged = FindResultItem & { _type: string };
    const results: Tagged[] = [];

    await Promise.all(
      scopes.flatMap((s) =>
        (SEARCH_TARGETS[s] ?? []).map(async (uri) => {
          try {
            const r = await client.find(query, {
              targetUri: uri,
              limit: maxResults,
              scoreThreshold: 0,
            });
            const bucket = (r as Record<string, FindResultItem[] | undefined>)[s];
            for (const item of bucket ?? []) {
              results.push({ ...item, _type: s.replace(/s$/, "") });
            }
          } catch { /* best-effort */ }
        }),
      ),
    );

    const filtered = results.filter((r) => clampScore(r.score) >= config.scoreThreshold);
    filtered.sort((a, b) => clampScore(b.score) - clampScore(a.score));

    const seen = new Set<string>();
    const picked: Tagged[] = [];
    for (const item of filtered) {
      const key = getMemoryDedupeKey(item);
      if (seen.has(key)) continue;
      seen.add(key);
      picked.push(item);
      if (picked.length >= maxResults) break;
    }

    if (picked.length === 0) {
      return { content: [{ type: "text" as const, text: "No matching context found." }] };
    }

    const lines = picked.map((item) => {
      const score = clampScore(item.score);
      const abstract = (item.abstract ?? item.overview ?? "").trim() || "(no abstract)";
      return `- [${item._type} ${(score * 100).toFixed(0)}%] ${item.uri}\n    ${abstract}`;
    });
    return {
      content: [{
        type: "text" as const,
        text: `Found ${picked.length} item(s):\n\n${lines.join("\n")}\n\nUse the read tool to expand a URI.`,
      }],
    };
  },
);

// -- Tool: read -----------------------------------------------------------
// Replaces read_context + list_context. Supports batch URIs. Directory URIs
// return a listing; file URIs return content.

async function readOneUri(uri: string): Promise<string> {
  try {
    const body = await client.read(uri);
    const text = typeof body === "string" ? body : JSON.stringify(body);
    if (text.trim()) return text;
  } catch { /* fall through to try ls */ }

  try {
    const entries = await client.lsUri(uri);
    if (Array.isArray(entries) && entries.length > 0) {
      return entries
        .map((e) => {
          const name = typeof e.name === "string" ? e.name : "?";
          const kind = e.isDir ? "dir" : "file";
          return `[${kind}] ${name}`;
        })
        .join("\n");
    }
  } catch { /* fall through */ }

  return `(nothing found at ${uri})`;
}

server.tool(
  "read",
  "Read one or more viking:// URIs. Pass a single URI or an array for batch reads. Directory URIs return a listing of entries. Prefer search to find relevant URIs rather than navigating directories.",
  {
    uris: z.union([
      z.string().describe("Single viking:// URI"),
      z.array(z.string()).describe("Array of viking:// URIs for batch read"),
    ]),
  },
  async ({ uris }) => {
    const uriList = Array.isArray(uris) ? uris : [uris];

    if (uriList.length === 1) {
      const text = await readOneUri(uriList[0]!);
      return { content: [{ type: "text" as const, text }] };
    }

    const results = await Promise.all(
      uriList.map(async (uri) => {
        const text = await readOneUri(uri);
        return `=== ${uri} ===\n${text}`;
      }),
    );

    return { content: [{ type: "text" as const, text: results.join("\n\n") }] };
  },
);

// -- Tool: store ----------------------------------------------------------
// Persistent identity-scoped session (ported from P1-8 memory_store).

const mcpStoreSessionId =
  "cc-mcpstore-" +
  md5Short(`${config.accountId}|${config.userId}|${config.agentId}`);
const COMMIT_THRESHOLD_TOKENS = 4000;

server.tool(
  "store",
  "Store information into OpenViking long-term memory. Use when the user says 'remember this', shares preferences, important facts, or decisions worth persisting.",
  {
    text: z.string().describe("The information to store"),
    role: z.string().optional().describe("Message role: 'user' (default) or 'assistant'"),
  },
  async ({ text, role }) => {
    const msgRole = role || "user";
    await client.addSessionMessage(mcpStoreSessionId, msgRole, text);

    let committed = false;
    const meta = await client.getSessionMeta(mcpStoreSessionId);
    const pending = Number((meta as { pending_tokens?: number } | null)?.pending_tokens || 0);
    if (pending >= COMMIT_THRESHOLD_TOKENS) {
      await client.commitSession(mcpStoreSessionId);
      committed = true;
    }

    return {
      content: [{
        type: "text" as const,
        text: committed
          ? `Stored. ${pending} tokens committed for extraction.`
          : `Stored. ${pending} pending tokens (commits at ${COMMIT_THRESHOLD_TOKENS}).`,
      }],
    };
  },
);

// -- Tool: forget ---------------------------------------------------------

server.tool(
  "forget",
  "Delete a memory from OpenViking. Provide an exact URI for direct deletion, or a search query to find and delete matching memories.",
  {
    uri: z.string().optional().describe("Exact viking:// memory URI to delete"),
    query: z.string().optional().describe("Search query to find the memory to delete"),
  },
  async ({ uri, query }) => {
    if (uri) {
      if (!isMemoryUri(uri)) {
        return { content: [{ type: "text" as const, text: `Refusing to delete non-memory URI: ${uri}` }] };
      }
      await client.deleteUri(uri);
      return { content: [{ type: "text" as const, text: `Deleted: ${uri}` }] };
    }

    if (!query) {
      return { content: [{ type: "text" as const, text: "Provide either uri or query." }] };
    }

    const candidateLimit = 20;
    const [userSettled, agentSettled] = await Promise.allSettled([
      client.find(query, { targetUri: "viking://user/memories", limit: candidateLimit, scoreThreshold: 0 }),
      client.find(query, { targetUri: "viking://agent/memories", limit: candidateLimit, scoreThreshold: 0 }),
    ]);
    const userMems = userSettled.status === "fulfilled" ? (userSettled.value.memories ?? []) : [];
    const agentMems = agentSettled.status === "fulfilled" ? (agentSettled.value.memories ?? []) : [];
    const all = [...userMems, ...agentMems].filter((m) => m.level === 2);

    const candidates = postProcessMemories(all, {
      limit: candidateLimit,
      scoreThreshold: config.scoreThreshold,
      leafOnly: true,
    }).filter((item) => isMemoryUri(item.uri));

    if (candidates.length === 0) {
      return { content: [{ type: "text" as const, text: "No matching memories found." }] };
    }

    const top = candidates[0]!;
    if (candidates.length === 1 && clampScore(top.score) >= 0.85) {
      await client.deleteUri(top.uri);
      return { content: [{ type: "text" as const, text: `Deleted: ${top.uri}` }] };
    }

    const list = candidates
      .map((item) => `- ${item.uri} — ${item.abstract?.trim() || "?"} (${(clampScore(item.score) * 100).toFixed(0)}%)`)
      .join("\n");

    return {
      content: [{
        type: "text" as const,
        text: `Found ${candidates.length} candidates. Specify the exact URI:\n\n${list}`,
      }],
    };
  },
);

// -- Tool: health ---------------------------------------------------------

server.tool(
  "health",
  "Check whether the OpenViking server is reachable.",
  {},
  async () => {
    const ok = await client.healthCheck();
    return {
      content: [{
        type: "text" as const,
        text: ok
          ? `OpenViking is healthy (${config.baseUrl})`
          : `OpenViking is unreachable at ${config.baseUrl}`,
      }],
    };
  },
);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

const transport = new StdioServerTransport();
await server.connect(transport);
