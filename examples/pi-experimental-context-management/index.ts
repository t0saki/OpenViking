/**
 * Pi OpenViking Extension — EXPERIMENTAL context management
 *
 * A fork of examples/pi-coding-agent-extension with the takeover mode removed,
 * kept as the base for agent-driven context windows (`new_context` / `history`
 * / `get_context_remaining`, added in a later phase). Everything else is the
 * same: turns sync to an OpenViking session, recall is injected into the
 * newest user message, and `viking_*` tools expose the store.
 *
 * Do not load this together with the openviking extension — they would both
 * register `viking_*` tools and both sync the same OV session, so this one
 * disables itself when it finds the other already registered.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createLogger } from "./shared/debug-log.mjs";
import { loadConfigFromModuleUrl, type OVConfig } from "./config.js";
import { OVClient } from "./client.js";
import { RecallManager } from "./recall.js";
import { SyncManager } from "./sync.js";
import { buildProfileBlock } from "./shared/profile-inject.mjs";
import { guardVikingUriToolCall } from "./lib/uri-guard-adapter.mjs";
import { registerTools } from "./tools.js";

/** The tool whose presence means the non-experimental extension is loaded. */
const COEXISTENCE_PROBE_TOOL = "viking_search";

export default async function (pi: ExtensionAPI) {
  // --- Load config ---
  const config = loadConfigFromModuleUrl(import.meta.url);
  if (!config.enabled) return;

  // --- Initialize modules ---
  const client = new OVClient(config);
  const sync = new SyncManager(client, config);
  const recall = new RecallManager(client, config, () => sync.sessionId);
  const logger = createLogger("pi", {
    debug: Boolean(config.debugLogPath),
    debugLogPath: config.debugLogPath,
  });

  // Session state
  let connected = false;
  let bypassed = false;
  let profileBlock = "";
  let archiveOverview = "";
  let toolsRegistered = false;
  let compacted = false;
  let started = false;
  let startPromise: Promise<void> | null = null;

  /**
   * True when another OpenViking extension already owns the tool surface.
   * Checked before this extension registers anything, so the loser of the race
   * adds no duplicate tools and no second writer to the same OV session.
   */
  const otherExtensionActive = (): boolean => {
    try {
      const tools = (pi as any).getAllTools?.();
      return Array.isArray(tools) && tools.some((tool: any) => tool?.name === COEXISTENCE_PROBE_TOOL);
    } catch {
      return false;
    }
  };

  // ================================================================
  // Event Handlers
  // ================================================================

  const start = async (ctx: any): Promise<void> => {
    if (started) return;
    if (startPromise) return startPromise;

    startPromise = (async () => {
      // Coexistence guard, before any registration: two OpenViking extensions
      // would duplicate every viking_* tool and race on the same OV session.
      if (!toolsRegistered && otherExtensionActive()) {
        bypassed = true;
        started = true;
        ctx.ui?.notify?.(
          "OpenViking experimental extension disabled: another OpenViking extension is already active",
          "warning",
        );
        return;
      }

      // Bypass check
      const cwd = process.cwd();
      for (const pattern of config.bypassPatterns) {
        if (matchBypass(cwd, pattern)) {
          bypassed = true;
          started = true;
          return;
        }
      }

      // Register tools before the health check: they need no network, and a
      // pi -c continuation must have them even while OV is unreachable.
      if (!toolsRegistered) {
        registerTools(pi, client, sync);
        toolsRegistered = true;
      }

      // Health check
      connected = await client.health();
      if (!connected) {
        if (config.logLevel === "info") {
          ctx.ui.notify("OpenViking: server not reachable", "warning");
        }
        return;
      }

      // Ensure OV session
      const piSessionId = ctx.sessionManager.getSessionId();
      const ok = await sync.ensureSession(piSessionId);
      if (!ok) {
        if (config.logLevel !== "silent") {
          ctx.ui.notify("OpenViking: failed to create session", "error");
        }
        return;
      }
      await sync.replayPending();

      // Profile injection
      profileBlock = await buildSessionProfileBlock(client, config);

      if (sync.sessionId) {
        // Resume rehydration — fetch archive overview if session was previously committed.
        archiveOverview = await fetchArchiveOverview(client, sync.sessionId, config);
      }

      updateStatus(ctx, connected, 0, sync.sessionId, config);

      started = true;
      if (config.logLevel === "info") {
        ctx.ui.notify(`OpenViking connected (${piSessionId.slice(0, 8)}...)`, "info");
      }
    })().finally(() => {
      startPromise = null;
    });

    return startPromise;
  };

  // --- session_start ---
  pi.on("session_start", async (event, ctx) => {
    // Fire-and-forget: the OV chain (health check, session ensure, profile
    // build) costs ~2s against the remote server; blocking session_start on it
    // delays every pi startup. start() is memoized via startPromise, so
    // before_agent_start awaits the same in-flight chain before the first
    // provider request — the first turn still gets profile + recall.
    void start(ctx).catch((error) => {
      logger.logError("session_start", error);
    });
  });

  // --- before_agent_start ---
  pi.on("before_agent_start", async (event, ctx) => {
    // session_start doesn't fire for pi -c continuations.
    await start(ctx);

    if (!connected || bypassed) return;

    // Queue recall for the context hook. Pi renders the user message before
    // that hook, so recall latency does not delay the message appearing.
    recall.queueSearch(event.prompt);

    // Compose system prompt additions
    const parts: string[] = [];
    if (profileBlock) parts.push(profileBlock);
    if (archiveOverview && (compacted || archiveOverview.trim())) {
      parts.push(archiveOverview);
    }
    parts.push("OpenViking tools: viking_search, viking_read, viking_browse, viking_remember, viking_forget, viking_add_resource.");

    const additions = parts.join("\n\n");
    if (!additions) return;

    return {
      systemPrompt: event.systemPrompt + "\n\n" + additions,
    };
  });

  // --- context ---
  pi.on("context", async (event, ctx) => {
    if (!connected || bypassed) return;

    // Keep recall synchronous with the provider request so the current prompt
    // still receives current-query memory, without blocking user-message UI.
    await recall.searchPending();

    const messages = recall.injectRecall(event.messages as any[]);
    return { messages };
  });

  // --- tool_call ---
  pi.on("tool_call", async (event, _ctx) => {
    const decision = guardVikingUriToolCall(event);
    if (!decision) return;
    return decision;
  });

  // --- turn_end ---
  pi.on("turn_end", async (event, ctx) => {
    if (!connected || bypassed || !config.syncTurns) return;

    const branch = ctx.sessionManager.getBranch();
    const result = await sync.syncBranch(branch);
    logger.log("turn_end", { added: result.added, tokens: result.tokens });
    updateStatus(ctx, connected, result.added, sync.sessionId, config);
  });

  // --- session_before_compact ---
  pi.on("session_before_compact", async (_event, _ctx) => {
    if (!connected || bypassed) return;

    const archiveId = await sync.commit();
    compacted = true;

    // Cache archive overview for rehydration after compaction
    if (archiveId && sync.sessionId) {
      archiveOverview = await fetchArchiveOverview(
        client, sync.sessionId, config,
      );
    }
    // Return nothing → pi proceeds with default compaction. The context-window
    // core replaces this fallback in a later phase.
  });

  // --- session_shutdown ---
  pi.on("session_shutdown", async (_event, _ctx) => {
    if (!connected || bypassed) return;

    // No forced commit: archive boundaries belong to the reset path, and a
    // commit on exit would split a window the next process still owns.
    await sync.shutdown();
  });

  // --- agent_end ---
  pi.on("agent_end", async (_event, _ctx) => {
    recall.invalidate();
  });

  // ================================================================
  // Commands
  // ================================================================

  pi.registerCommand("viking", {
    description: "OpenViking status and manual operations. Use 'commit' to force a sync.",
    handler: async (args, ctx) => {
      if (!connected) {
        ctx.ui.notify("OpenViking: not connected", "warning");
        return;
      }

      if (args?.trim() === "commit") {
        await sync.shutdown();
        const commitResult = await sync.commit();
        if (commitResult !== null) {
          ctx.ui.notify(
            "OpenViking: committed successfully" +
              (commitResult?.trace_id ? ` (trace_id=${commitResult.trace_id})` : ""),
            "info",
          );
        } else {
          ctx.ui.notify("OpenViking: commit failed", "error");
        }
        return;
      }

      // Status
      const sid = sync.sessionId ?? "none";
      ctx.ui.notify(
        `OpenViking: ${connected ? "connected" : "disconnected"} | session: ${sid.slice(0, 12)}...`,
        "info",
      );
    },
  });
}

// ================================================================
// Helper Functions
// ================================================================

/** Simple bypass pattern matching (prefix and glob). */
function matchBypass(cwd: string, pattern: string): boolean {
  if (pattern.startsWith("*")) {
    return cwd.endsWith(pattern.slice(1));
  }
  if (pattern.endsWith("*")) {
    return cwd.startsWith(pattern.slice(0, -1));
  }
  return cwd === pattern || cwd.startsWith(pattern + "/");
}

/** Build the <openviking-context> profile block. */
async function buildSessionProfileBlock(
  client: OVClient, config: OVConfig,
): Promise<string> {
  try {
    const profile = await buildProfileBlock(
      (path: string, init?: any, options?: any) => client.fetchJSON(path, init, 10000),
      config.profileTokenBudget,
      config.peerId,
    );
    if (!profile?.block) return "";
    return [
      '<openviking-context source="session-start">',
      profile.block,
      "</openviking-context>",
    ].join("\n");
  } catch {
    return "";
  }
}

/** Fetch archive overview for rehydration using the session context API. */
async function fetchArchiveOverview(
  client: OVClient, sessionId: string, config: OVConfig,
): Promise<string> {
  try {
    const ctx = await client.getSessionContext(sessionId, config.resumeContextBudget);
    if (!ctx || !ctx.latest_archive_overview) return "";

    return [
      '<openviking-context source="session-archive">',
      "<session-archive>",
      ctx.latest_archive_overview,
      "</session-archive>",
      "</openviking-context>",
    ].join("\n");
  } catch {
    return "";
  }
}

function updateStatus(
  ctx: any,
  connected: boolean,
  added: number,
  sessionId: string | null,
  config: OVConfig,
): void {
  const setter = ctx?.ui?.setStatus;
  if (typeof setter !== "function") return;
  const status = `${connected ? "OV ✓" : "OV ✗"} · ↩${added} · ✎ ${config.commitTokenThreshold} · ${sessionId ? sessionId.slice(0, 12) : "none"}`;
  try {
    setter("openviking", status);
  } catch {
    // Best effort; pi API shape may vary across fast-moving versions.
  }
}
