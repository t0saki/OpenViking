import { resolveSettings } from "./shared/plugin-config.mjs";
import { buildUserAgent, resolveOpenVikingCredentials } from "./shared/credentials.mjs";
import { resolveEffectivePeerId, resolvePluginPeerId } from "./shared/workspace-peer.mjs";

export const PLUGIN_VERSION = "0.4.0";

/**
 * Namespace for the bridged OpenViking MCP tools. DSH publishes every MCP tool
 * as `mcp__<serverName>__<rawName>`, so this string is part of the
 * model-facing contract: changing it renames all of them.
 */
export const MCP_SERVER_NAME = "openviking";

const DEFAULT_ENDPOINT = "http://127.0.0.1:1933";

/**
 * Resolve the plugin's configuration.
 *
 * `input` is what the cordis host hands the plugin — this harness has no config
 * file of its own. Every knob is declared in `shared/config-schema.mjs`, and
 * `ovcli.conf`'s `plugin` section outranks the host's input so one file
 * configures every harness and `ov config switch` moves them together.
 * Connection fields are the exception: an endpoint or key named by the host is
 * the more specific answer and stays ahead of the credential chain.
 */
export function resolveConfig(input = {}, env = process.env, cwd = process.cwd()) {
  const credentials = resolveOpenVikingCredentials(env, "dsh");
  // ov.conf's `dsh` section is the legacy layer here as everywhere else, but
  // the cordis patch shares that slot; the host named this process's settings,
  // so it wins the overlap.
  const ovSection = credentials.ovFile.dsh;
  const legacy = { ...(ovSection && typeof ovSection === "object" ? ovSection : {}), ...input };
  const { settings, configured, sources } = resolveSettings("dsh", { env, cwd, legacy });
  // The host named this process's peer, so it stays ahead of every file; the
  // rest of the order is the one every harness follows.
  const explicitPeerId = resolvePluginPeerId({
    settings,
    configured,
    sources,
    credentials,
    hostInput: input.peerId,
    env,
  });
  const config = {
    ...settings,
    endpoint: String(input.endpoint || credentials.baseUrl || DEFAULT_ENDPOINT).replace(/\/+$/, ""),
    apiKey: input.apiKey || credentials.apiKey,
    account: input.account || credentials.account,
    user: input.user || credentials.user,
    peerId: explicitPeerId,
    explicitPeerId,
    userAgent: buildUserAgent("dsh", PLUGIN_VERSION),
    harness: "dsh",
    // The name this plugin's client has always used for the request budget.
    requestTimeoutMs: settings.timeoutMs,
    recallLimitConfigured: configured.has("recallLimit"),
    recallQueryExpansionConfigured: configured.has("recallQueryExpansion"),
  };

  const effectivePeer = resolveEffectivePeerId({ cfg: config, cwd });
  config.peerId = effectivePeer.peerId;
  config.legacyPeerId = effectivePeer.legacyPeerId;
  return config;
}
