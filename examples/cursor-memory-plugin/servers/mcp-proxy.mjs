#!/usr/bin/env node

import { fileURLToPath } from "node:url";
import { resolve as resolvePath } from "node:path";

import { loadAgentHookConfig } from "../../memory-plugin-shared/lib/agent-hook-runtime.mjs";
import { createLogger } from "../../memory-plugin-shared/lib/debug-log.mjs";
import { buildMcpProxyConfig, resolveMcpActorPeerId } from "../../memory-plugin-shared/lib/mcp-proxy-config.mjs";
import { createOpenVikingMcpProxy } from "../../memory-plugin-shared/lib/mcp-proxy-core.mjs";

function readConfig() {
  const cfg = loadAgentHookConfig("cursor");
  return buildMcpProxyConfig({
    mcpUrl: cfg.mcpUrl,
    apiKey: cfg.apiKey,
    account: cfg.account,
    user: cfg.user,
    sendIdentityHeaders: cfg.sendIdentityHeaders,
    peerId: resolveMcpActorPeerId(cfg),
    userAgent: cfg.userAgent,
    timeoutMs: cfg.timeoutMs,
    debug: cfg.debug,
    debugLogPath: cfg.debugLogPath,
    credentialSource: cfg.credentialSource,
    credentialPath: cfg.cliPath || cfg.ovPath || "",
    watchedPaths: [cfg.cliPath, cfg.ovPath],
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolvePath(process.argv[1])) {
  createOpenVikingMcpProxy({ readConfig, loggerFactory: createLogger }).start();
}
