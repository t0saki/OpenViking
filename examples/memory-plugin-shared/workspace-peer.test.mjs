import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  PEER_SOURCE_PRESETS,
  deriveWorkspacePeerId,
  peerSourceTemplates,
  renderPeerTemplate,
  resolveEffectivePeerId,
} from "./lib/workspace-peer.mjs";
import { resolveWorkspaceIdentity } from "./lib/workspace-identity.mjs";

async function repo({ remote = "git@github.com:volcengine/OpenViking.git", git = true } = {}) {
  const root = realpathSync(await mkdtemp(join(tmpdir(), "ov-peer-")));
  if (git) {
    await mkdir(join(root, ".git"), { recursive: true });
    await writeFile(
      join(root, ".git", "config"),
      remote ? `[remote "origin"]\n\turl = ${remote}\n` : "[core]\n\trepositoryformatversion = 0\n",
    );
  }
  const env = { HOME: "/nonexistent-home", OPENVIKING_STATE_DIR: join(root, ".state") };
  return { root, env };
}

function resolve(cwd, env, cfg = {}) {
  return resolveEffectivePeerId({ cfg, cwd, identity: resolveWorkspaceIdentity({ cwd, env, cache: false }), env });
}

test("deriveWorkspacePeerId keeps the byte-for-byte legacy rule", () => {
  assert.equal(deriveWorkspacePeerId("/Users/x/Dev/OpenViking"), "-Users-x-Dev-OpenViking");
  assert.equal(deriveWorkspacePeerId("/tmp/a  b/"), "-tmp-a--b-");
  assert.equal(deriveWorkspacePeerId("abc.DEF_123@x-y"), "abc-DEF-123-x-y");
  assert.equal(deriveWorkspacePeerId(""), "");
  assert.equal(deriveWorkspacePeerId(null), "");
});

test("an explicit peer id still wins over every rule", async () => {
  const { root, env } = await repo();
  assert.deepEqual(resolve(root, env, { peerId: " configured " }), {
    peerId: "configured",
    source: "explicit",
    origin: "explicit",
    legacyPeerId: "",
  });
});

test("the default is the repository, from any subdirectory or clone", async () => {
  const { root, env } = await repo();
  const deep = join(root, "examples", "codex-memory-plugin");
  await mkdir(deep, { recursive: true });

  const top = resolve(root, env);
  const nested = resolve(deep, env);
  assert.equal(top.peerId, "github.com-volcengine-openviking");
  assert.equal(nested.peerId, top.peerId, "a subdirectory is the same workspace");
  assert.equal(top.source, "workspace", "call sites compare this against the literal");
  assert.equal(top.origin, "{git_remote}");
  assert.equal(nested.legacyPeerId, deriveWorkspacePeerId(deep), "the pre-git id stays reachable");
});

test("the git preset falls back through the root to the working directory", async () => {
  const noRemote = await repo({ remote: "" });
  const rootDerived = resolve(noRemote.root, noRemote.env);
  assert.equal(rootDerived.peerId, deriveWorkspacePeerId(noRemote.root));
  assert.equal(rootDerived.origin, "{git_root}");

  const plain = await repo({ git: false });
  const cwdDerived = resolve(plain.root, plain.env);
  assert.equal(cwdDerived.peerId, deriveWorkspacePeerId(plain.root));
  assert.equal(cwdDerived.origin, "{cwd}");
  assert.equal(cwdDerived.legacyPeerId, "", "nothing to fall back to when it already is the legacy id");
});

test("the cwd preset reproduces the old identity exactly", async () => {
  const { root, env } = await repo();
  const deep = join(root, "examples");
  await mkdir(deep, { recursive: true });

  const legacy = resolve(deep, env, { peerSource: "cwd" });
  assert.equal(legacy.peerId, deriveWorkspacePeerId(deep));
  assert.equal(legacy.legacyPeerId, "");
});

test("none, and the switch that predates it, both send no peer", async () => {
  const { root, env } = await repo();
  assert.deepEqual(resolve(root, env, { peerSource: "none" }), {
    peerId: "", source: "none", origin: "none", legacyPeerId: "",
  });
  assert.deepEqual(resolve(root, env, { workspacePeer: false }), {
    peerId: "", source: "none", origin: "disabled", legacyPeerId: "",
  });
});

test("a template can shape the id, and a list is tried in order", async () => {
  const { root, env } = await repo();
  assert.equal(resolve(root, env, { peerSource: "git-{git_remote}" }).peerId, "git-github.com-volcengine-openviking");
  assert.equal(resolve(root, env, { peerSource: "team-{dir}" }).peerId, `team-${root.split("/").pop()}`);

  const noRemote = await repo({ remote: "" });
  const chain = resolve(noRemote.root, noRemote.env, { peerSource: ["{git_remote}", "team-{dir}"] });
  assert.equal(chain.peerId, `team-${noRemote.root.split("/").pop()}`, "an empty variable falls through");
  assert.equal(chain.origin, "team-{dir}");
});

test("a template naming only empty variables resolves to no peer at all", async () => {
  const plain = await repo({ git: false });
  const unresolved = resolve(plain.root, plain.env, { peerSource: ["{git_remote}"] });
  assert.deepEqual(unresolved, { peerId: "", source: "none", origin: "unresolved", legacyPeerId: "" });
});

test("renderPeerTemplate is all-or-nothing so no half-formed id escapes", () => {
  const vars = { git_remote: "github.com-o-r", git_root: "-src-r", cwd: "-src-r-sub", dir: "r" };
  assert.equal(renderPeerTemplate("{git_remote}", vars), "github.com-o-r");
  assert.equal(renderPeerTemplate("a-{dir}-b", vars), "a-r-b");
  assert.equal(renderPeerTemplate("{git_remote}-{missing}", vars), "");
  assert.equal(renderPeerTemplate("{git_remote}", { git_remote: "" }), "");
  assert.equal(renderPeerTemplate("literal", vars), "literal");
  assert.equal(renderPeerTemplate("", vars), "");
});

test("peerSourceTemplates resolves presets and passes templates through", () => {
  assert.deepEqual(peerSourceTemplates(undefined), PEER_SOURCE_PRESETS.git);
  assert.deepEqual(peerSourceTemplates(""), PEER_SOURCE_PRESETS.git);
  assert.deepEqual(peerSourceTemplates("cwd"), ["{cwd}"]);
  assert.deepEqual(peerSourceTemplates("none"), []);
  assert.deepEqual(peerSourceTemplates("my-{dir}"), ["my-{dir}"]);
  assert.deepEqual(peerSourceTemplates(["{git_remote}", "{cwd}"]), ["{git_remote}", "{cwd}"]);
});

test("a fork keeps its own peer, and every clone of one repo shares one", async () => {
  const upstream = await repo({ remote: "git@github.com:volcengine/OpenViking.git" });
  const fork = await repo({ remote: "git@github.com:t0saki/OpenViking.git" });
  const secondClone = await repo({ remote: "https://github.com/volcengine/OpenViking.git" });

  assert.equal(resolve(upstream.root, upstream.env).peerId, resolve(secondClone.root, secondClone.env).peerId);
  assert.notEqual(resolve(fork.root, fork.env).peerId, resolve(upstream.root, upstream.env).peerId);
});
