/**
 * Per-harness plugin settings from ovcli.conf.
 *
 * ovcli.conf carried connection fields only, so every harness had to keep its
 * tuning knobs in ov.conf's harness section — a server-side file that a
 * client-side plugin has no business editing. The `plugin` section fixes that:
 * shared keys apply to every harness that reads them, and a per-harness object
 * overrides them.
 *
 *   {
 *     "url": "...", "api_key": "...",
 *     "plugin": {
 *       "recallQueryExpansion": "off",
 *       "recallCompress": "auto"
 *     }
 *   }
 *
 * Resolution stays env → ovcli.conf plugin.<harness> → ovcli.conf plugin →
 * ov.conf harness section (legacy) → defaults.
 *
 * Every harness resolves through `resolveSettings()` below, so one `plugin`
 * section configures all of them and `ov config switch` moves behaviour along
 * with credentials. The knobs themselves are declared once in
 * `config-schema.mjs`; this module only decides which files supply them.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve as resolvePath } from "node:path";

import { HARNESS_KEYS, harnessKey, resolveKnobs } from "./config-schema.mjs";
import {
  announcedOverrides,
  loadWorkspaceLayers,
  mergeConfigLayers,
  normalizeWorkspaceConfig,
  projectWorkspaceSettings,
} from "./workspace-config.mjs";
import { findWorkspaceRoot, resolveWorkspaceIdentity } from "./workspace-identity.mjs";
import { readEntry } from "./workspace-registry.mjs";

const DEFAULT_OVCLI_CONF_PATH = join(homedir(), ".openviking", "ovcli.conf");

export { HARNESS_KEYS, harnessKey };

function tryLoadJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return null;
  }
}

/**
 * Merge the shared plugin section with the harness-specific override.
 * Returns a flat settings object; unknown keys pass through untouched so a
 * harness can add its own knobs without touching this module.
 */
export function loadPluginSettings(harness, env = process.env, options = {}) {
  const path = resolvePath(
    (env.OPENVIKING_CLI_CONFIG_FILE || DEFAULT_OVCLI_CONF_PATH).replace(/^~/, homedir()),
  );
  const file = tryLoadJson(path);
  const plugin = file && typeof file.plugin === "object" && file.plugin ? file.plugin : {};

  const shared = {};
  for (const [key, value] of Object.entries(plugin)) {
    if (value && typeof value === "object" && !Array.isArray(value)) continue;
    shared[key] = value;
  }
  // Both spellings of the harness name find the same override, so copying
  // `claude-code` out of the installer is not a silent no-op.
  const key = harnessKey(harness);
  const scoped = {};
  for (const candidate of [key, key.replace(/_/g, "-")]) {
    const value = candidate && plugin[candidate];
    if (value && typeof value === "object" && !Array.isArray(value)) Object.assign(scoped, value);
  }

  const settings = { ...shared, ...scoped };
  const cwd = String(options.cwd || "").trim();
  if (!cwd) return settings;
  return { ...settings, ...resolveWorkspaceSettings(cwd, env, options).settings };
}

/**
 * Every knob for one harness, resolved through the whole layer stack.
 *
 * `legacy` is that harness's section in ov.conf, which is where tuning used to
 * live: it stays the lowest configured layer so an existing deployment keeps
 * working, and everything above it comes from files a client may edit.
 *
 * Returns the settings, the names some layer actually supplied (what the
 * `*Configured` flags report), and the raw ovcli.conf merge — a couple of
 * callers still need to know a value came from ovcli.conf rather than ov.conf.
 */
export function resolveSettings(harness, options = {}) {
  const { env = process.env, cwd = "", legacy = {}, clientVersion = "" } = options;
  const key = harnessKey(harness);
  const plugin = loadPluginSettings(key, env, { cwd, clientVersion });
  const { settings, configured, sources } = resolveKnobs({
    harness: key,
    layers: [
      { name: "ov.conf", data: legacy },
      { name: "ovcli.conf", data: plugin },
    ],
    env,
  });
  return { settings, configured, sources, plugin };
}

/**
 * The workspace layers for one cwd, flattened into harness knobs.
 *
 * Takes the cwd explicitly because a hook's `loadConfig()` runs at module top
 * level, before the payload on stdin has said which directory the session is
 * actually in — the caller resolves this again once it knows.
 */
export function resolveWorkspaceSettings(cwd, env = process.env, { clientVersion = "" } = {}) {
  const empty = { settings: {}, root: "", provenance: {}, warnings: [], announced: [] };
  try {
    const { root } = findWorkspaceRoot(cwd, env);
    if (!root) return empty;

    const { layers, warnings } = loadWorkspaceLayers(root, { clientVersion });
    // The identity is what makes the registry's negative evidence work: without
    // it a directory reused by a different repository inherits the old peer.
    const identity = resolveWorkspaceIdentity({ cwd, env });
    const registry = readEntry(root, { identity, env });
    warnings.push(...registry.warnings);
    if (registry.entry?.settings) layers.push({ layer: "registry", data: registry.entry.settings });
    if (registry.entry?.peer) layers.push({ layer: "registry", data: { peer: registry.entry.peer } });
    if (!layers.length) return { ...empty, root, warnings };

    const { value, provenance } = mergeConfigLayers(layers, warnings);
    normalizeWorkspaceConfig(value, warnings);
    return {
      settings: projectWorkspaceSettings(value),
      root,
      provenance,
      warnings,
      announced: announcedOverrides(provenance),
      value,
    };
  } catch {
    // A hook must never die over a config file. An unreadable layer is no layer.
    return empty;
  }
}

const REWRITE_MODES = new Set(["off", "client", "server", "auto"]);

/**
 * Normalize the tri-state rewrite knob.
 *
 * off    — never compress
 * client — always compress locally through the host CLI
 * server — always ask the server for a digest
 * auto   — prefer local (cost lands on the user's own subscription), fall back
 *          to the server when no healthy host CLI is available
 */
export function normalizeRewriteMode(value, fallback = "off") {
  const raw = String(value ?? "").trim().toLowerCase();
  if (REWRITE_MODES.has(raw)) return raw;
  if (raw === "1" || raw === "true" || raw === "yes") return "auto";
  if (raw === "0" || raw === "false" || raw === "no") return "off";
  return fallback;
}
