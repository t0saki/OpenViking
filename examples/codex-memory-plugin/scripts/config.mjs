/**
 * Configuration for the Codex OpenViking memory plugin.
 *
 * Every knob is declared once in `shared/config-schema.mjs` and resolved by
 * `resolveSettings()`, which reads the layers in this order:
 *
 *   env (OPENVIKING_*) → workspace `.openviking/config*.json` and the machine
 *   registry → ovcli.conf `plugin.codex` → ovcli.conf `plugin` → ov.conf's
 *   `codex` section (legacy) → the schema's defaults
 *
 * What stays here is what only this harness knows: the credential resolution
 * Codex shares with its MCP proxy, the auth mode, and the knobs whose fallback
 * is derived from another knob.
 *
 * Credential source:
 *   - Default (auto): env-var credentials win when any credential env var is
 *     set; otherwise the active ovcli.conf is used, so `ov config switch`
 *     changes hooks, MCP, and in-process `ov` commands together on next launch.
 *   - Set OPENVIKING_CREDENTIAL_SOURCE=cli to force ovcli.conf, or =env to
 *     force env-var credentials.
 *   - Without env vars or ovcli.conf, ov.conf/defaults are used.
 *
 * The stdio MCP proxy calls the same resolver directly. Aligning the resolver
 * prevents identity drift between auto-capture/auto-recall hooks, MCP calls,
 * and child `ov` commands launched from inside Codex.
 *
 * File-path env vars:
 *   OPENVIKING_CLI_CONFIG_FILE  alternate ovcli.conf path  (preferred)
 *   OPENVIKING_CONFIG_FILE      alternate ov.conf path
 *
 * For backward compat, if only OPENVIKING_CONFIG_FILE is set and the file
 * it points at parses as an ovcli.conf (top-level `url`/`api_key`, no
 * `server` section), it is treated as ovcli.conf — earlier versions of
 * this plugin used OPENVIKING_CONFIG_FILE to mean either file.
 *
 * Connection / identity env vars:
 *   OPENVIKING_URL / OPENVIKING_BASE_URL
 *   OPENVIKING_API_KEY / OPENVIKING_BEARER_TOKEN
 *   OPENVIKING_AUTH_MODE
 *   OPENVIKING_ACCOUNT, OPENVIKING_USER, OPENVIKING_PEER_ID
 */

import { homedir } from "node:os";
import { join } from "node:path";
import { resolveOpenVikingCredentials } from "./ov-credentials.mjs";
import { buildUserAgent, readManifestVersion } from "./shared/credentials.mjs";
import { resolveSettings } from "./shared/plugin-config.mjs";
import { resolvePluginPeerId } from "./shared/workspace-peer.mjs";

const USER_AGENT = buildUserAgent(
  "codex",
  readManifestVersion(new URL("../.codex-plugin/plugin.json", import.meta.url)),
);

function str(val, fallback) {
  if (typeof val === "string" && val.trim()) return val.trim();
  return fallback;
}

function configBool(value, fallback) {
  if (typeof value === "boolean") return value;
  const lower = String(value ?? "").trim().toLowerCase();
  if (lower === "0" || lower === "false" || lower === "no" || lower === "off") return false;
  if (lower === "1" || lower === "true" || lower === "yes" || lower === "on"
      || lower === "auto" || lower === "client") return true;
  return fallback;
}

function normalizeAuthMode(val) {
  const mode = str(val, "").toLowerCase();
  return ["trusted", "api_key"].includes(mode) ? mode : "";
}

/**
 * `cwd` selects the workspace layer (`.openviking/config.json` and the registry
 * entry for that directory). It defaults to this process's directory, which is
 * all a hook knows at module load; a hook whose payload names the session's
 * directory calls this again with it. Re-resolving that late is safe because a
 * workspace file may not carry connection or credential keys, so baseUrl/apiKey
 * cannot move — loggers and fetch helpers built from the first load stay valid.
 */
export function loadConfig(cwd = process.cwd()) {
  const creds = resolveOpenVikingCredentials();
  const { cliPath, ovFile, ovPath } = creds;
  const configPath = cliPath || ovPath || null;

  const workspaceCwd = str(cwd, "") || process.cwd();
  const { settings, configured, sources } = resolveSettings("codex", {
    cwd: workspaceCwd,
    legacy: ovFile.codex,
  });
  const server = ovFile.server || {};
  const authMode = normalizeAuthMode(settings.authMode)
    || normalizeAuthMode(server.auth_mode)
    || ((creds.account || creds.user) ? "trusted" : "api_key");

  const peerId = resolvePluginPeerId({ settings, configured, sources, credentials: creds });

  const timeoutMs = settings.timeoutMs;
  // A write gets a longer budget than a read, and a digest has to finish inside
  // the recall request it belongs to, so both fall back to another knob.
  const captureTimeoutMs = settings.captureTimeoutMs || Math.max(timeoutMs * 2, 30000);
  const recallCompressTimeoutMs = settings.recallCompressTimeoutMs
    || Math.max(1000, settings.recallTimeoutMs - 10000);

  return {
    ...settings,
    configPath,
    cliConfigPath: cliPath,
    ovConfigPath: ovPath,
    credentialSource: creds.credentialSource,
    baseUrl: creds.baseUrl,
    authMode,
    sendIdentityHeaders: authMode === "trusted",
    // The credential chain has no view of ovcli.conf's `plugin` section, which
    // is where a key set through `plugin.codex.apiKey` lives.
    apiKey: creds.apiKey || settings.apiKey,
    account: creds.account,
    user: creds.user,
    peerId,
    harness: "codex",
    userAgent: USER_AGENT,
    captureTimeoutMs,
    debugLogPath: settings.debugLogPath || join(homedir(), ".openviking", "logs", "codex-hooks.log"),

    // Codex reads the compression knob as on/off; "auto" and "client" are the
    // Claude Code spellings of on, and mean the same thing here.
    recallCompress: configBool(settings.recallCompress, true),
    recallCompressTimeoutMs,
    recallCompressConfigured: Boolean(settings.recallCompressModel || settings.recallCompressThinking),

    // Several fields are sent to the server only when the user asked for them,
    // so a default must not look like a choice.
    recallLimitConfigured: configured.has("recallLimit"),
    recallMaxTokensConfigured: configured.has("recallMaxTokens"),
    recallQueryExpansionConfigured: configured.has("recallQueryExpansion"),
    recallCompressMaxBulletsConfigured: configured.has("recallCompressMaxBullets"),
  };
}
