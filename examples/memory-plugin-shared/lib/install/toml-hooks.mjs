#!/usr/bin/env node
/**
 * The `[[hooks]]` merge behind a host that keeps its hooks in TOML.
 *
 * Kimi Code reads hooks from `~/.kimi-code/config.toml`, a file people edit by
 * hand and other tools (herdr, orca) append their own `[[hooks]]` entries to,
 * so this installer cannot reserialize it: a round trip through any TOML writer
 * would lose their comments and their ordering. The entries go inside a
 * comment-delimited block instead — everything between the markers is ours to
 * rewrite on every install and to cut out on uninstall, everything outside is
 * copied through byte for byte.
 *
 * The commands themselves, the MCP half and the installed-integration manifest
 * come from host-json-config.mjs, so a TOML host's hook entries carry the same
 * env prefix and the same trailing marker `ownsHook` looks for.
 *
 * Installer-only, like its neighbour: nothing under lib/ imports it, so it
 * stays out of the runtime closure sync.mjs vendors into the plugins.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  atomicWriteText,
  readHostIntegration,
  readJson,
  removeHostMcpServer,
  writeHostMcpServer,
  writeInstalledManifest,
} from "./host-json-config.mjs";

/** The comment pair that fences off this client's entries. */
export function blockMarkers(clientId) {
  return {
    begin: `# >>> openviking ${clientId} integration`,
    end: `# <<< openviking ${clientId} integration`,
  };
}

function tomlValue(value) {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "boolean") return String(value);
  if (typeof value !== "string") throw new Error(`Unsupported TOML hook value: ${JSON.stringify(value)}`);
  const escaped = value
    .replaceAll("\\", "\\\\")
    .replaceAll('"', '\\"')
    .replace(/[\u0000-\u001f\u007f]/gu, (char) => `\\u${char.codePointAt(0).toString(16).padStart(4, "0")}`);
  return `"${escaped}"`;
}

/** The `[[hooks]]` block this installer owns, markers included. */
export function renderTomlHooksBlock({ entries, clientId, renderHookCommand }) {
  const { begin, end } = blockMarkers(clientId);
  const tables = entries.map((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`Invalid ${clientId} hook entry: ${JSON.stringify(entry)}`);
    }
    const lines = Object.entries(entry).map(([key, value]) => {
      const rendered = key === "command" && typeof value === "string" ? renderHookCommand(value) : value;
      return `${key} = ${tomlValue(rendered)}`;
    });
    return ["[[hooks]]", ...lines].join("\n");
  });
  return `${begin}\n${tables.join("\n\n")}\n${end}\n`;
}

/**
 * The file without this client's block.
 *
 * Only the seam the block leaves behind is renormalised; blank lines elsewhere
 * are the user's and stay as they are.
 */
export function stripTomlHooksBlock(text, clientId) {
  const { begin, end } = blockMarkers(clientId);
  const start = text.indexOf(begin);
  if (start < 0) return text;
  const stop = text.indexOf(end, start);
  if (stop < 0) return text;
  const before = text.slice(0, start).replace(/\n+$/u, "");
  const after = text.slice(stop + end.length).replace(/^\n+/u, "");
  if (!before) return after;
  if (!after) return `${before}\n`;
  return `${before}\n\n${after}`;
}

/** Render the host's hook template into its TOML config and its MCP server into JSON. */
export function writeHostTomlHooks({ kind, configPath, mcpPath, root, clientId, nodeBin, sourceMode }) {
  const { hostDir, manifest, env, renderHookCommand } =
    readHostIntegration({ root, kind, clientId, nodeBin });

  const hookTemplate = readJson(path.join(hostDir, "hooks.json"));
  if (!Array.isArray(hookTemplate.hooks)) {
    throw new Error(`Invalid ${clientId} hooks template`);
  }
  const block = renderTomlHooksBlock({ entries: hookTemplate.hooks, clientId, renderHookCommand });

  let current = "";
  try { current = fs.readFileSync(configPath, "utf8"); } catch {}
  const kept = stripTomlHooksBlock(current, clientId).replace(/\n+$/u, "");
  atomicWriteText(configPath, kept ? `${kept}\n\n${block}` : block);

  writeHostMcpServer({ hostDir, mcpPath, root, clientId, nodeBin, env });
  writeInstalledManifest({ root, manifest, clientId, sourceMode, hooksPath: configPath, mcpPath });
}

/**
 * Cut this client's block out of its TOML config and drop its MCP server.
 *
 * As with the JSON hosts, the writes here take no backup — one would hold a
 * copy of the entries the uninstall just removed — and the copy an install left
 * behind goes with them.
 */
export function removeHostTomlHooks({ configPath, mcpPath, clientId }) {
  if (fs.existsSync(configPath)) {
    const current = fs.readFileSync(configPath, "utf8");
    const kept = stripTomlHooksBlock(current, clientId);
    atomicWriteText(configPath, kept, { backup: false });
  }
  removeHostMcpServer(mcpPath);
  for (const file of [configPath, mcpPath]) {
    if (file) fs.rmSync(`${file}.bak`, { force: true });
  }
}

const COMMANDS = {
  write: ([kind, configPath, mcpPath, root, clientId, nodeBin, sourceMode]) =>
    writeHostTomlHooks({ kind, configPath, mcpPath, root, clientId, nodeBin, sourceMode }),
  remove: ([configPath, mcpPath, clientId]) => removeHostTomlHooks({ configPath, mcpPath, clientId }),
};

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const [command, ...argv] = process.argv.slice(2);
  const run = COMMANDS[command];
  if (!run) {
    process.stderr.write(`usage: toml-hooks.mjs <${Object.keys(COMMANDS).join("|")}> ...\n`);
    process.exit(2);
  }
  try {
    run(argv);
  } catch (error) {
    process.stderr.write(`${error?.message || error}\n`);
    process.exit(1);
  }
}
