/**
 * The per-machine workspace registry, `~/.openviking/workspaces/`.
 *
 * One file per workspace rather than one file listing them all: hooks are many
 * short-lived processes, and a shared JSON file loses writes whenever any two
 * of them touch unrelated workspaces at once. A per-workspace file is written
 * atomically at 0600, so that whole class is gone. What remains is two hooks of
 * one session writing one workspace — a much narrower window, and one nothing
 * writes into on a hot path today; a write there is still read-modify-write and
 * the later one wins.
 *
 * This is the layer above both workspace files, so it is where the user keeps
 * the last word over any repository they clone. It is also the only layer
 * allowed to name an ovcli.conf profile.
 */

import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";

import { isValidProfileName, readWorkspaceFile } from "./workspace-config.mjs";

export const REGISTRY_VERSION = 1;

export function registryDir(env = process.env) {
  const home = String(env.OPENVIKING_HOME || "").trim();
  const base = home ? home.replace(/^~(?=$|\/)/, homedir()) : join(homedir(), ".openviking");
  return join(base, "workspaces");
}

/**
 * A readable name plus a hash, keyed on the workspace's identity rather than
 * its path wherever git supplies one.
 *
 * Two linked worktrees of one repository are one workspace — the same peer, so
 * the same settings and the same `ov peer link` — and keying on the checkout
 * path would silently split them in two. Outside a repository there is no
 * identity but the path, so two `~/src/api` clones still get separate entries.
 */
export function slotName(root, identity = null) {
  const path = String(root || "");
  const key = identity ? identityKey(identity) : "path";
  const source = key === "path" ? path : key;
  const label = key.startsWith("remote:") ? key.split("/").pop() : basename(path);
  const readable = String(label || "").replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
  const digest = createHash("sha256").update(source).digest("hex").slice(0, 12);
  return `${readable ? `${readable}-` : ""}${digest}.json`;
}

export function entryPath(root, env = process.env, identity = null) {
  return join(registryDir(env), slotName(root, identity));
}

function readRawEntry(path) {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf-8"));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * The identity a stored entry is checked against. Path alone is not enough:
 * a directory can be deleted and a different repository cloned in its place,
 * and inheriting the old entry's peer would silently cross two projects.
 */
export function identityKey(identity) {
  // The normalized remote, so re-spelling origin (ssh ↔ https, or rotating an
  // embedded token) is not mistaken for a different repository.
  const remote = String(identity?.remote || "").trim();
  if (remote) return `remote:${remote}`;
  if (identity?.isGit) return `git:${identity.gitCommonDir || ""}`;
  return "path";
}

/**
 * Read this workspace's entry, or null.
 *
 * A stored entry whose identity contradicts the current one is treated as a
 * miss — negative evidence. Nothing is inherited from it, and a later write
 * replaces it.
 */
export function readEntry(root, { identity = null, env = process.env } = {}) {
  const path = entryPath(root, env, identity);
  const file = readWorkspaceFile(path, { layer: "registry", registry: true });
  if (!file.data) return { path, entry: null, warnings: file.warnings, conflict: false };

  const entry = file.data;
  const warnings = [...file.warnings];
  if (identity) {
    const expected = identityKey(identity);
    const stored = String(entry.identity || "");
    if (stored && stored !== expected) {
      warnings.push(
        `${path} was recorded for a different repository (${stored}); starting a fresh entry for ${expected}`,
      );
      return { path, entry: null, warnings, conflict: true };
    }
  }
  if (entry.cli_config_profile && !isValidProfileName(entry.cli_config_profile)) {
    warnings.push(`${path}: cli_config_profile ${JSON.stringify(entry.cli_config_profile)} is not a valid profile name`);
    delete entry.cli_config_profile;
  }
  return { path, entry, warnings, conflict: false };
}

/**
 * Write this workspace's entry. Read-modify-write on this one small file only:
 * anything the caller does not mention is preserved, so `ov peer link` does not
 * erase settings and vice versa.
 */
export function writeEntry(root, patch, { identity = null, env = process.env, now = Date.now() } = {}) {
  const path = entryPath(root, env, identity);
  // Eight copies of this module ship independently, so a newer client's entry
  // can be sitting here. Refuse rather than flatten it: an unreadable entry is
  // someone else's data, not a blank slate.
  const onDisk = readRawEntry(path);
  if (onDisk && onDisk.version !== REGISTRY_VERSION) {
    throw new Error(
      `${path} was written by a newer client (version ${JSON.stringify(onDisk.version)}); refusing to overwrite it`,
    );
  }

  const existing = readEntry(root, { identity, env });
  const previous = existing.entry || {};

  const entry = {
    ...previous,
    ...patch,
    version: REGISTRY_VERSION,
    root: String(root || ""),
    identity: identity ? identityKey(identity) : previous.identity || "path",
    label: identity?.remote || previous.label || "",
    first_seen_at: previous.first_seen_at || now,
    last_seen_at: now,
  };
  if (entry.cli_config_profile !== undefined && !isValidProfileName(entry.cli_config_profile)) {
    throw new Error(`cli_config_profile must match ^[a-z0-9][a-z0-9._-]{0,63}$, got ${JSON.stringify(entry.cli_config_profile)}`);
  }

  const dir = registryDir(env);
  mkdirSync(dir, { recursive: true });
  const tmp = `${path}.${process.pid}.tmp`;
  writeFileSync(tmp, `${JSON.stringify(entry, null, 2)}\n`, { mode: 0o600 });
  try {
    chmodSync(tmp, 0o600);
  } catch { /* best effort on filesystems without modes */ }
  renameSync(tmp, path);
  return { path, entry };
}

/**
 * Record a peer this workspace used to write under, so a later `ov peer` run
 * and doctor can point at memories the current peer no longer reaches.
 */
export function rememberPreviousPeer(root, peerId, options = {}) {
  const id = String(peerId || "").trim();
  if (!id) return null;
  const { entry: previous } = readEntry(root, options);
  const seen = Array.isArray(previous?.previous_peer_ids) ? previous.previous_peer_ids : [];
  if (seen.includes(id)) return null;
  return writeEntry(root, { previous_peer_ids: [...seen, id].slice(-20) }, options);
}

export function listEntries(env = process.env) {
  const dir = registryDir(env);
  let names;
  try {
    names = readdirSync(dir).filter((name) => name.endsWith(".json"));
  } catch {
    return [];
  }
  const entries = [];
  for (const name of names) {
    const path = join(dir, name);
    try {
      if (!statSync(path).isFile()) continue;
      const entry = JSON.parse(readFileSync(path, "utf-8"));
      if (entry?.version === REGISTRY_VERSION) entries.push({ path, entry });
    } catch { /* a corrupt entry is skipped, not fatal */ }
  }
  return entries.sort((a, b) => String(a.entry.root).localeCompare(String(b.entry.root)));
}
