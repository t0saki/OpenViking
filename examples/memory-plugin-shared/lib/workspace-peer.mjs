/**
 * Which peer a workspace writes its memories under.
 *
 * The peer used to be the working directory with every non-alphanumeric byte
 * turned into a dash, which made the identity an accident of where the
 * repository happened to sit: a clone on another machine, a rename, a worktree
 * or simply `cd examples/` each minted a separate, empty namespace. The default
 * is now git's own idea of the repository, so one project keeps one memory
 * wherever it is checked out.
 *
 * `peer.source` decides the rule. Presets cover the three answers most people
 * want; a template — or a list of templates tried in order — covers the rest.
 * The old behaviour is still one word away, byte for byte.
 */

import { legacySanitize, resolveWorkspaceIdentity } from "./workspace-identity.mjs";

/**
 * `git` is the default. It prefers the remote, because that is the one name
 * every clone agrees on; falls back to the repository root, which at least
 * stops a subdirectory from forking the identity; and finally to the working
 * directory, which is what a non-repository had all along.
 *
 * No preset adds a prefix. A path-derived id starts with `-` on POSIX, so it
 * cannot collide with a remote-derived one; anyone who wants a prefix writes
 * their own template.
 */
export const PEER_SOURCE_PRESETS = {
  git: ["{git_remote}", "{git_root}", "{cwd}"],
  cwd: ["{cwd}"],
  none: [],
};

export const DEFAULT_PEER_SOURCE = "git";

const VARIABLE_RE = /\{([a-z_]+)\}/g;

export function deriveWorkspacePeerId(cwd) {
  return legacySanitize(cwd);
}

/** Normalize `peer.source` — a preset name, a template, or a list — to templates. */
export function peerSourceTemplates(source) {
  if (Array.isArray(source)) return source.map(String).filter(Boolean);
  const raw = String(source ?? "").trim();
  if (!raw) return PEER_SOURCE_PRESETS[DEFAULT_PEER_SOURCE];
  if (Object.hasOwn(PEER_SOURCE_PRESETS, raw)) return PEER_SOURCE_PRESETS[raw];
  return [raw];
}

/**
 * Substitute one template, or return "" when any variable it names is empty.
 *
 * All-or-nothing on purpose: a half-resolved template like `git-` would be a
 * silently shared identity, so an empty variable falls through to the next
 * template instead.
 */
export function renderPeerTemplate(template, vars) {
  const text = String(template || "");
  if (!text) return "";
  let empty = false;
  const rendered = text.replace(VARIABLE_RE, (match, name) => {
    if (!Object.hasOwn(vars, name)) {
      empty = true;
      return "";
    }
    const value = String(vars[name] ?? "");
    if (!value) empty = true;
    return value;
  });
  return empty ? "" : rendered;
}

/**
 * The peer this process should send, and where it came from.
 *
 * `source` keeps its three values — call sites compare it against the literal
 * `"workspace"` to decide whether a session pin may be reused — while `origin`
 * names the template that actually produced the id, and `legacyPeerId` carries
 * the pre-git id whenever it differs, so recall can still reach memories
 * written under it.
 */
export function resolveEffectivePeerId({ cfg = {}, cwd = "", identity = null, env = process.env } = {}) {
  const explicit = String(cfg.peerId || "").trim();
  if (explicit) return { peerId: explicit, source: "explicit", origin: "explicit", legacyPeerId: "" };

  // `OPENVIKING_WORKSPACE_PEER=0` predates `peer.source` and still means "none".
  if (cfg.workspacePeer === false) return { peerId: "", source: "none", origin: "disabled", legacyPeerId: "" };

  const templates = peerSourceTemplates(cfg.peerSource);
  if (!templates.length) return { peerId: "", source: "none", origin: "none", legacyPeerId: "" };

  const vars = (identity || resolveWorkspaceIdentity({ cwd, env })).vars || {};
  const legacyPeerId = deriveWorkspacePeerId(cwd);
  for (const template of templates) {
    const peerId = renderPeerTemplate(template, vars);
    if (!peerId) continue;
    return {
      peerId,
      source: "workspace",
      origin: template,
      legacyPeerId: peerId === legacyPeerId ? "" : legacyPeerId,
    };
  }
  return { peerId: "", source: "none", origin: "unresolved", legacyPeerId: "" };
}
