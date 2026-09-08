import path from "path"
import { homedir } from "os"
import { buildUserAgent, readManifestVersion, resolveAuthMode, resolveOpenVikingCredentials } from "./shared/credentials.mjs"
import { resolveSettings } from "./shared/plugin-config.mjs"
import { resolveEffectivePeerId, resolvePluginPeerId } from "./shared/workspace-peer.mjs"

/**
 * Configuration for the opencode plugin.
 *
 * This plugin used to keep its own `openviking-config.json`, looked up along
 * four candidate paths. Every knob now comes from the same place as every other
 * harness — `ovcli.conf`'s `plugin` section, overridden by `plugin.opencode`,
 * by the workspace file, and by the environment — so `ov config switch` moves
 * behaviour along with credentials instead of only half of it.
 */
const USER_AGENT = buildUserAgent(
  "opencode",
  readManifestVersion(new URL("../package.json", import.meta.url)),
)

function str(value, fallback = "") {
  return typeof value === "string" && value.trim() ? value.trim() : fallback
}

function expandHome(value) {
  if (!value || typeof value !== "string") return value
  if (value === "~") return homedir()
  if (value.startsWith("~/") || value.startsWith("~\\")) return path.join(homedir(), value.slice(2))
  return value
}

export function loadConfig(pluginRoot, projectDirectory) {
  const creds = resolveOpenVikingCredentials(process.env, "opencode")
  const { settings, configured, sources } = resolveSettings("opencode", {
    cwd: projectDirectory || process.cwd(),
  })

  const endpoint = str(creds.baseUrl, "http://127.0.0.1:1933").replace(/\/+$/, "")
  const config = {
    ...settings,
    endpoint,
    baseUrl: endpoint,
    apiKey: creds.apiKey,
    account: creds.account,
    user: creds.user,
    accountId: creds.account,
    userId: creds.user,
    ...resolveAuthMode({ settings, ovFile: creds.ovFile, account: creds.account, user: creds.user }),
    peerId: resolvePluginPeerId({ settings, configured, sources, credentials: creds }),
    mcpUrl: creds.mcpUrl,
    credentialSource: creds.credentialSource,
    credentialPath: creds.cliPath || creds.ovPath || "",
    configPath: creds.cliPath || "",
    userAgent: USER_AGENT,
    harness: "opencode",

    // Three knobs opencode's own code reads as sections rather than as flat
    // keys: the MCP registration, the runtime data directory, and the repo
    // context cache.
    mcp: { enabled: settings.mcpEnabled },
    runtime: { dataDir: settings.dataDir },
    repoContext: { enabled: settings.repoContext, cacheTtlMs: settings.repoContextCacheTtlMs },

    debugLogPath: settings.debugLogPath
      || path.join(homedir(), ".openviking", "logs", "opencode-plugin.log"),
    recallLimitConfigured: configured.has("recallLimit"),
    recallQueryExpansionConfigured: configured.has("recallQueryExpansion"),
  }

  config.effectivePeer = resolveEffectivePeerId({ cfg: config, cwd: projectDirectory })
  return config
}

export function resolveDataDir(pluginRoot, config) {
  const configured = config.runtime?.dataDir
  if (configured) return expandHome(configured)
  return path.join(homedir(), ".config", "opencode", "openviking")
}
