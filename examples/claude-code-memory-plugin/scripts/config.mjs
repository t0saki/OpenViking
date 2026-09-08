/**
 * Configuration for the Claude Code OpenViking memory plugin.
 *
 * Every knob is declared once in `shared/config-schema.mjs` and resolved by
 * `resolveSettings()`, which reads the layers in this order:
 *
 *   env (OPENVIKING_*) → workspace `.openviking/config*.json` and the machine
 *   registry → ovcli.conf `plugin.claude_code` → ovcli.conf `plugin` →
 *   ov.conf's `claude_code` section (legacy) → the schema's defaults
 *
 * What stays here is what only this harness knows: which file supplied the
 * credential, the log path named after the plugin, the two knobs whose fallback
 * is derived from another knob, and the peer, which falls back to the
 * credential chain when no layer names one.
 *
 * Enable/disable:
 *   - OPENVIKING_MEMORY_ENABLED env var (0/false/no = off, 1/true/yes = on)
 *   - claude_code.enabled field in ov.conf (false = off)
 *   - Fallback: enabled when ov.conf or ovcli.conf exists, disabled otherwise
 *
 * Connection and credentials are not knobs and never come from a workspace
 * file: OPENVIKING_URL / OPENVIKING_BASE_URL, OPENVIKING_API_KEY /
 * OPENVIKING_BEARER_TOKEN, OPENVIKING_ACCOUNT, OPENVIKING_USER, then
 * ovcli.conf, then ov.conf's server section.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve as resolvePath } from "node:path";

import {
  buildUserAgent,
  readManifestVersion,
  resolveOpenVikingCredentials,
} from "./shared/credentials.mjs";
import { normalizeRewriteMode, resolveSettings } from "./shared/plugin-config.mjs";

const DEFAULT_OV_CONF_PATH = join(homedir(), ".openviking", "ov.conf");
const DEFAULT_OVCLI_CONF_PATH = join(homedir(), ".openviking", "ovcli.conf");
const USER_AGENT = buildUserAgent(
  "claude-code",
  readManifestVersion(new URL("../.claude-plugin/plugin.json", import.meta.url)),
);

function num(val, fallback) {
  if (typeof val === "number" && Number.isFinite(val)) return val;
  if (typeof val === "string" && val.trim()) {
    const n = Number(val);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function str(val, fallback) {
  if (typeof val === "string" && val.trim()) return val.trim();
  return fallback;
}

function hasOwn(obj, key) {
  return Object.prototype.hasOwnProperty.call(obj || {}, key);
}

function envBool(name) {
  const v = process.env[name];
  if (v == null || v === "") return undefined;
  const lower = v.trim().toLowerCase();
  if (lower === "0" || lower === "false" || lower === "no") return false;
  if (lower === "1" || lower === "true" || lower === "yes") return true;
  return undefined;
}

/**
 * Try to load and parse a JSON config file. Returns parsed object or null.
 */
function tryLoadJsonFile(envVar, defaultPath) {
  const configPath = resolvePath(
    (process.env[envVar] || defaultPath).replace(/^~/, homedir()),
  );

  let raw;
  try {
    raw = readFileSync(configPath, "utf-8");
  } catch {
    return null;
  }

  try {
    return { configPath, file: JSON.parse(raw) };
  } catch {
    return null;
  }
}

/**
 * Determine whether the plugin is enabled.
 *
 * Priority:
 *   1. OPENVIKING_MEMORY_ENABLED env var
 *   2. claude_code.enabled in ov.conf
 *   3. Whether ov.conf or ovcli.conf exists and is parseable
 *
 * When force-enabled via env var (=1) without config files, the caller must
 * provide connection info via other env vars (OPENVIKING_URL, etc.).
 */
export function isPluginEnabled() {
  const envEnabled = envBool("OPENVIKING_MEMORY_ENABLED");
  if (envEnabled !== undefined) return envEnabled;

  const ovConf = tryLoadJsonFile("OPENVIKING_CONFIG_FILE", DEFAULT_OV_CONF_PATH);
  if (ovConf) {
    const cc = ovConf.file.claude_code || {};
    if (cc.enabled === false) return false;
    return true;
  }

  // No ov.conf — check if ovcli.conf exists (sufficient for connection info)
  const cliConf = tryLoadJsonFile("OPENVIKING_CLI_CONFIG_FILE", DEFAULT_OVCLI_CONF_PATH);
  if (cliConf) return true;

  return false;
}

/**
 * Load the full plugin configuration.
 *
 * `cwd` selects the workspace layer (`.openviking/config.json` and the
 * registry entry for that directory). It defaults to this process's directory,
 * which is all a hook knows at module load; a hook whose payload names the
 * session's directory calls this again with it. Re-resolving that late is safe
 * because a workspace file may not carry connection or credential keys, so
 * baseUrl/apiKey cannot move — loggers and fetch helpers built from the first
 * load stay valid.
 */
export function loadConfig(cwd = process.cwd()) {
  const workspaceCwd = str(cwd, "") || process.cwd();
  const ovConf = tryLoadJsonFile("OPENVIKING_CONFIG_FILE", DEFAULT_OV_CONF_PATH);
  const cliConf = tryLoadJsonFile("OPENVIKING_CLI_CONFIG_FILE", DEFAULT_OVCLI_CONF_PATH);

  const ovFile = ovConf?.file || {};
  const cliFile = cliConf?.file || {};
  const configPath = ovConf?.configPath || cliConf?.configPath || null;

  const server = ovFile.server || {};
  const { settings, configured, plugin } = resolveSettings("claude-code", {
    cwd: workspaceCwd,
    legacy: ovFile.claude_code,
  });
  // The peer travels with the credentials, so `ov config switch` moves it here
  // too. Only the peer is taken from this chain; the api key below is resolved
  // separately.
  const credentials = resolveOpenVikingCredentials(process.env, "claude_code");

  // baseUrl: env → ovcli.url → ov.server.url → http://{host}:{port}
  const envUrl = str(process.env.OPENVIKING_URL, null) || str(process.env.OPENVIKING_BASE_URL, null);
  let baseUrl;
  if (envUrl) {
    baseUrl = envUrl.replace(/\/+$/, "");
  } else if (cliFile.url) {
    baseUrl = str(cliFile.url, "").replace(/\/+$/, "");
  } else if (server.url) {
    baseUrl = str(server.url, "").replace(/\/+$/, "");
  } else {
    const host = str(server.host, "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
    const port = Math.floor(num(server.port, 1933));
    baseUrl = `http://${host}:${port}`;
  }

  // apiKey: env → ovcli.api_key → the plugin/ov.conf section → server.root_api_key
  // Accepts OPENVIKING_BEARER_TOKEN or OPENVIKING_API_KEY (sent as Bearer either way).
  const envApiKey = str(process.env.OPENVIKING_BEARER_TOKEN, null)
    || str(process.env.OPENVIKING_API_KEY, null);
  const apiKey = envApiKey
    || str(cliFile.api_key, null)
    || str(settings.apiKey, null)
    || str(server.root_api_key, "");

  // Which source actually supplied the api_key. `configPath` only reports the
  // file that parsed, so debug logs and 401 hints pointed at the wrong file
  // whenever both configs existed.
  const ccApiKeyFromCli = hasOwn(plugin, "apiKey");
  let credentialSource = "none";
  let credentialPath = null;
  if (envApiKey) {
    credentialSource = "env";
  } else if (str(cliFile.api_key, null)) {
    credentialSource = "ovcli";
    credentialPath = cliConf?.configPath || null;
  } else if (str(settings.apiKey, null)) {
    credentialSource = ccApiKeyFromCli ? "ovcli" : "ov";
    credentialPath = (ccApiKeyFromCli ? cliConf?.configPath : ovConf?.configPath) || null;
  } else if (str(server.root_api_key, null)) {
    credentialSource = "ov";
    credentialPath = ovConf?.configPath || null;
  }

  const accountId = str(process.env.OPENVIKING_ACCOUNT, null)
    || str(cliFile.account, null)
    || settings.accountId;
  const userId = str(process.env.OPENVIKING_USER, null)
    || str(cliFile.user, null)
    || settings.userId;

  const timeoutMs = settings.timeoutMs;
  // A write gets a longer budget than a read, so the fallback is derived from
  // the timeout rather than fixed.
  const captureTimeoutMs = settings.captureTimeoutMs || Math.max(timeoutMs * 2, 30000);

  return {
    ...settings,
    configPath,
    credentialSource,
    credentialPath,
    baseUrl,
    apiKey,
    accountId,
    userId,
    // The credential chain owns the peer unless a `plugin` entry or a workspace
    // file names one, which is the more specific answer for this directory.
    peerId: configured.has("peerId") ? settings.peerId : credentials.peerId,
    harness: "claude-code",
    userAgent: USER_AGENT,
    captureTimeoutMs,
    debugLogPath: settings.debugLogPath || join(homedir(), ".openviking", "logs", "cc-hooks.log"),

    // Digest compression defaults to auto: prefer the local host CLI and fall
    // back to the server when it is unavailable. A failed digest still falls
    // back to the uncompressed context block. `recallRewrite` keeps the
    // internal field name because the shared core maps this mode to the
    // server's `rewrite` request field; OPENVIKING_RECALL_REWRITE is the older
    // env spelling and still works.
    recallRewrite: normalizeRewriteMode(
      process.env.OPENVIKING_RECALL_COMPRESS
        ?? process.env.OPENVIKING_RECALL_REWRITE
        ?? settings.recallCompress,
      "auto",
    ),

    // Several fields are sent to the server only when the user asked for them,
    // so a default must not look like a choice.
    recallLimitConfigured: configured.has("recallLimit"),
    recallMaxTokensConfigured: configured.has("recallMaxTokens"),
    recallQueryExpansionConfigured: configured.has("recallQueryExpansion"),
    recallCompressMaxBulletsConfigured: configured.has("recallCompressMaxBullets"),
  };
}
