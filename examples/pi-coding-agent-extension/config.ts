import { buildUserAgent, resolveAuthMode, resolveOpenVikingCredentials } from "./shared/credentials.mjs";
import { resolveSettings } from "./shared/plugin-config.mjs";
import { resolveEffectivePeerId, resolvePluginPeerId } from "./shared/workspace-peer.mjs";

/** Hand-maintained: this extension ships no manifest to read a version from. */
export const EXTENSION_VERSION = "0.3.0";

export interface OVConfig {
  enabled: boolean;
  endpoint: string;
  apiKey: string;
  account: string;
  user: string;
  /** `trusted` or `api_key`; only the former puts the identity on the wire. */
  authMode: string;
  sendIdentityHeaders: boolean;
  peerId: string;
  /** The pre-git workspace id, when it differs — recall still reaches it. */
  legacyPeerId: string;
  userAgent: string;
  harness: string;
  workspacePeer: boolean;
  recallPeerScope: "actor" | "all";
  recallQueryExpansion: "auto" | "off";
  recallQueryExpansionConfigured: boolean;
  autoCapture: boolean;
  recallTokenBudget: number;
  recallMaxContentChars: number;
  recallPreferAbstract: boolean;
  recallLimit: number;
  recallLimitConfigured: boolean;
  recallLedger: boolean;
  scoreThreshold: number;
  minQueryLength: number;
  profileTokenBudget: number;
  resumeContextBudget: number;
  commitTokenThreshold: number;
  commitKeepRecentCount: number;
  takeoverEnabled: boolean;
  takeoverTokenThreshold: number;
  takeoverKeepRecentTurns: number;
  takeoverOverviewBudget: number;
  takeoverOverviewPollMs: number;
  takeoverOverviewPollMax: number;
  captureToolResults: boolean;
  captureMode: "semantic" | "keyword";
  captureMaxLength: number;
  captureToolMaxChars: number;
  captureAssistantTurns: boolean;
  /** Kept as this extension's original spelling; projected onto the shared one. */
  bypassPatterns: string[];
  bypassSession: boolean;
  bypassSessionPatterns: string[];
  logLevel: "silent" | "error" | "info";
  debugLogPath: string;
}

/**
 * Load the extension's configuration.
 *
 * The extension used to keep a `config.json` beside itself. It shipped with the
 * extension holding exactly the code defaults, so nothing could tell an
 * operator's choice from the factory setting. Every knob is declared in
 * `shared/config-schema.mjs` now and resolved from the same layers as every
 * other harness: env → the workspace file → `ovcli.conf`'s `plugin.pi` →
 * `ovcli.conf`'s `plugin` → defaults.
 */
export function loadConfig(cwd: string = process.cwd()): OVConfig {
  const creds = resolveOpenVikingCredentials(process.env, "pi");
  const { settings, configured, sources } = resolveSettings("pi", { cwd });

  const config = {
    ...settings,
    endpoint: creds.baseUrl,
    apiKey: creds.apiKey,
    account: creds.account,
    user: creds.user,
    ...resolveAuthMode({ settings, ovFile: creds.ovFile, account: creds.account, user: creds.user }),
    peerId: resolvePluginPeerId({ settings, configured, sources, credentials: creds }),
    userAgent: buildUserAgent("pi", EXTENSION_VERSION),
    harness: "pi",
    // `bypassSessionPatterns` is the name the shared matcher reads and every
    // other harness spells; `bypassPatterns` was this extension's own and is
    // still accepted. Both hold the same list so either can be inspected.
    bypassPatterns: settings.bypassSessionPatterns,
    // OPENVIKING_DEBUG_LOG is the shared spelling and lands in the schema;
    // OV_DEBUG_LOG is pi's older name, kept working so existing setups log.
    debugLogPath: settings.debugLogPath || String(process.env.OV_DEBUG_LOG || "").trim(),
    recallLimitConfigured: configured.has("recallLimit"),
    recallQueryExpansionConfigured: configured.has("recallQueryExpansion"),
  } as OVConfig;

  // The whole resolution, not just the id: `legacyPeerId` is what lets recall
  // under `actor` scope still reach memories written before the git-derived
  // peer replaced the path-derived one.
  const effectivePeer = resolveEffectivePeerId({ cfg: config as any, cwd });
  config.peerId = effectivePeer.peerId;
  config.legacyPeerId = effectivePeer.legacyPeerId || "";
  return config;
}
