import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { realpathSync } from "node:fs";

import {
  findWorkspaceRoot,
  legacySanitize,
  normalizeGitRemote,
  readGitRemoteUrl,
  resolveWorkspaceIdentity,
  sanitizePeerId,
} from "./lib/workspace-identity.mjs";
import { deriveWorkspacePeerId } from "./lib/workspace-peer.mjs";

async function tempRoot(name) {
  return realpathSync(await mkdtemp(join(tmpdir(), `ov-${name}-`)));
}

async function makeRepo(root, { remote = "", name = ".git" } = {}) {
  const gitDir = join(root, name);
  await mkdir(gitDir, { recursive: true });
  const config = remote
    ? `[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = ${remote}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n`
    : "[core]\n\trepositoryformatversion = 0\n";
  await writeFile(join(gitDir, "config"), config);
  return gitDir;
}

test("legacySanitize stays byte-identical to the peer id shipped today", () => {
  for (const value of ["/Users/x/Dev/OpenViking", "/tmp/a  b/", "abc.DEF_123@x-y", "", "///"]) {
    assert.equal(legacySanitize(value), deriveWorkspacePeerId(value), `diverged on ${JSON.stringify(value)}`);
  }
});

test("normalizeGitRemote folds every spelling of one repo together", () => {
  const expected = "github.com/volcengine/openviking";
  for (const url of [
    "git@github.com:volcengine/OpenViking.git",
    "https://github.com/volcengine/OpenViking.git",
    "https://github.com/volcengine/OpenViking",
    "https://GitHub.com/Volcengine/OpenViking.git/",
    "ssh://git@github.com:22/volcengine/OpenViking.git",
  ]) {
    assert.equal(normalizeGitRemote(url), expected, `failed on ${url}`);
  }
});

test("normalizeGitRemote drops userinfo so a token never reaches the peer id", () => {
  const url = "https://someone:ghp_averysecrettoken@github.com:8443/volcengine/OpenViking.git";
  const normalized = normalizeGitRemote(url);
  assert.equal(normalized, "github.com/volcengine/openviking");
  assert.doesNotMatch(normalized, /ghp_|someone/);
});

test("normalizeGitRemote refuses identities that are only local", () => {
  for (const url of ["", "   ", "/srv/git/bare.git", "file:///srv/git/bare.git", "../sibling"]) {
    assert.equal(normalizeGitRemote(url), "", `should be empty for ${JSON.stringify(url)}`);
  }
});

test("sanitizePeerId produces a server-valid id and dodges the reserved names", () => {
  assert.equal(sanitizePeerId("github.com/volcengine/openviking"), "github.com-volcengine-openviking");
  assert.match(sanitizePeerId("github.com/volcengine/openviking"), /^[a-zA-Z0-9_.@-]+$/);
  assert.equal(sanitizePeerId("--weird//name--"), "weird-name");
  assert.equal(sanitizePeerId("__self"), "self");
  assert.equal(sanitizePeerId("ext-YWJj"), "x-ext-YWJj");
  assert.equal(sanitizePeerId("a@b@c"), "a@b-c", "the server allows at most one @");
  assert.equal(sanitizePeerId(".."), "");
  assert.equal(sanitizePeerId("///"), "");
});

test("sanitizePeerId keeps long ids unique after truncation", () => {
  const a = sanitizePeerId(`git.example.com/${"a".repeat(200)}/one`);
  const b = sanitizePeerId(`git.example.com/${"a".repeat(200)}/two`);
  assert.ok(a.length <= 100 && b.length <= 100, `${a.length} / ${b.length}`);
  assert.notEqual(a, b, "truncation must not collapse two repos into one peer");
  assert.match(a, /^[a-zA-Z0-9_.@-]+$/);
});

test("the workspace root is the repo, from any depth below it", async () => {
  const root = await tempRoot("root");
  await makeRepo(root, { remote: "git@github.com:volcengine/OpenViking.git" });
  const deep = join(root, "examples", "codex-memory-plugin");
  await mkdir(deep, { recursive: true });

  for (const cwd of [root, join(root, "examples"), deep]) {
    const found = findWorkspaceRoot(cwd, { HOME: "/nonexistent-home" });
    assert.equal(found.root, root, `wrong root from ${cwd}`);
    assert.equal(found.git.kind, "repo");
  }
});

test("a linked worktree resolves back to the repository it shares", async () => {
  const main = await tempRoot("main");
  const mainGit = await makeRepo(main, { remote: "git@github.com:volcengine/OpenViking.git" });
  const worktreeGitDir = join(mainGit, "worktrees", "feature");
  await mkdir(worktreeGitDir, { recursive: true });
  await writeFile(join(worktreeGitDir, "commondir"), "../..\n");

  const linked = await tempRoot("linked");
  await writeFile(join(linked, ".git"), `gitdir: ${worktreeGitDir}\n`);

  const found = findWorkspaceRoot(linked, { HOME: "/nonexistent-home" });
  assert.equal(found.git.kind, "worktree");
  assert.equal(found.git.commonDir, mainGit);
  assert.equal(readGitRemoteUrl(found.git.commonDir), "git@github.com:volcengine/OpenViking.git");
});

test("a submodule keeps its own identity instead of the superproject's", async () => {
  const parent = await tempRoot("parent");
  const parentGit = await makeRepo(parent, { remote: "git@github.com:volcengine/OpenViking.git" });
  const moduleGitDir = join(parentGit, "modules", "vendor");
  await mkdir(moduleGitDir, { recursive: true });
  await writeFile(join(moduleGitDir, "config"), '[remote "origin"]\n\turl = git@github.com:other/vendor.git\n');

  const sub = join(parent, "vendor");
  await mkdir(sub, { recursive: true });
  await writeFile(join(sub, ".git"), `gitdir: ${moduleGitDir}\n`);

  const found = findWorkspaceRoot(sub, { HOME: "/nonexistent-home" });
  assert.equal(found.git.kind, "submodule");
  assert.equal(normalizeGitRemote(readGitRemoteUrl(found.git.commonDir)), "github.com/other/vendor");
});

test("$HOME and the filesystem root are never workspace roots", async () => {
  const home = await tempRoot("home");
  await makeRepo(home, { remote: "git@github.com:someone/dotfiles.git" });
  const inside = join(home, "notes");
  await mkdir(inside, { recursive: true });

  assert.equal(findWorkspaceRoot(inside, { HOME: home }).root, "");
  assert.equal(findWorkspaceRoot(home, { HOME: home }).root, "");
});

test("readGitRemoteUrl reads only origin, and does not follow includes", async () => {
  const root = await tempRoot("config");
  const gitDir = join(root, ".git");
  await mkdir(gitDir, { recursive: true });
  await writeFile(
    join(gitDir, "config"),
    [
      "[include]",
      "\tpath = ./extra",
      "# a comment mentioning url = https://decoy.example/nope.git",
      '[remote "upstream"]',
      "\turl = git@github.com:volcengine/OpenViking.git",
      '[remote "origin"]',
      "\turl = git@github.com:t0saki/OpenViking.git",
      "",
    ].join("\n"),
  );
  await writeFile(join(gitDir, "extra"), '[remote "origin"]\n\turl = https://included.example/x.git\n');

  assert.equal(readGitRemoteUrl(gitDir), "git@github.com:t0saki/OpenViking.git");
  assert.equal(readGitRemoteUrl(gitDir, "upstream"), "git@github.com:volcengine/OpenViking.git");
  assert.equal(readGitRemoteUrl(gitDir, "missing"), "");
});

test("identity exposes every template variable, git and non-git alike", async () => {
  const root = await tempRoot("vars");
  await makeRepo(root, { remote: "git@github.com:volcengine/OpenViking.git" });
  const deep = join(root, "examples", "codex-memory-plugin");
  await mkdir(deep, { recursive: true });
  const env = { HOME: "/nonexistent-home", OPENVIKING_STATE_DIR: join(root, ".state") };

  const identity = resolveWorkspaceIdentity({ cwd: deep, env, cache: false });
  assert.equal(identity.vars.git_remote, "github.com-volcengine-openviking");
  assert.equal(identity.vars.git_root, legacySanitize(root));
  assert.equal(identity.vars.cwd, legacySanitize(deep));
  assert.equal(identity.vars.dir, sanitizePeerId(root.split("/").pop()));

  const plain = await tempRoot("plain");
  const outside = resolveWorkspaceIdentity({ cwd: plain, env, cache: false });
  assert.equal(outside.isGit, false);
  assert.equal(outside.vars.git_remote, "");
  assert.equal(outside.vars.git_root, "");
  assert.equal(outside.vars.cwd, legacySanitize(plain));
});

test("a repo with no origin leaves git_remote empty but still names the root", async () => {
  const root = await tempRoot("noremote");
  await makeRepo(root);
  const identity = resolveWorkspaceIdentity({
    cwd: root,
    env: { HOME: "/nonexistent-home", OPENVIKING_STATE_DIR: join(root, ".state") },
    cache: false,
  });
  assert.equal(identity.isGit, true);
  assert.equal(identity.vars.git_remote, "");
  assert.equal(identity.vars.git_root, legacySanitize(root));
});

test("the cache serves the hooks of one turn and expires on its own", async () => {
  const root = await tempRoot("cache");
  await makeRepo(root, { remote: "git@github.com:volcengine/OpenViking.git" });
  const env = { HOME: "/nonexistent-home", OPENVIKING_STATE_DIR: join(root, ".state") };

  const first = resolveWorkspaceIdentity({ cwd: root, env, now: 1_000_000 });
  assert.equal(first.vars.git_remote, "github.com-volcengine-openviking");

  await writeFile(join(root, ".git", "config"), '[remote "origin"]\n\turl = git@github.com:other/changed.git\n');
  const cached = resolveWorkspaceIdentity({ cwd: root, env, now: 1_030_000 });
  assert.equal(cached.vars.git_remote, "github.com-volcengine-openviking", "same turn should not re-walk");

  const expired = resolveWorkspaceIdentity({ cwd: root, env, now: 1_120_000 });
  assert.equal(expired.vars.git_remote, "github.com-other-changed");
});

// --- regressions found by review ------------------------------------------

test("a repository under a directory named modules is not a submodule", async () => {
  const container = await tempRoot("modules-container");
  const main = join(container, "modules", "app");
  await mkdir(main, { recursive: true });
  const mainGit = await makeRepo(main, { remote: "git@github.com:o/app.git" });
  const worktreeGitDir = join(mainGit, "worktrees", "feature");
  await mkdir(worktreeGitDir, { recursive: true });
  await writeFile(join(worktreeGitDir, "commondir"), "../..\n");

  const linked = await tempRoot("modules-linked");
  await writeFile(join(linked, ".git"), `gitdir: ${worktreeGitDir}\n`);

  const found = findWorkspaceRoot(linked, { HOME: "/nonexistent-home" });
  assert.equal(found.git.kind, "worktree");
  assert.equal(normalizeGitRemote(readGitRemoteUrl(found.git.commonDir)), "github.com/o/app");
});

test("a windows drive path is a local directory, not a remote", () => {
  for (const url of ["C:\\src\\repo", "C:/src/repo", "d:\\work\\api.git"]) {
    assert.equal(normalizeGitRemote(url), "", `should be empty for ${url}`);
  }
});

test("a trailing git comment is not part of the url", async () => {
  const root = await tempRoot("comment");
  const gitDir = join(root, ".git");
  await mkdir(gitDir, { recursive: true });
  await writeFile(join(gitDir, "config"), '[remote "origin"]\n\turl = git@github.com:a/b.git # my fork\n');

  assert.equal(readGitRemoteUrl(gitDir), "git@github.com:a/b.git");
  assert.equal(normalizeGitRemote(readGitRemoteUrl(gitDir)), "github.com/a/b");
});

test("the cache never keeps a token, and never leaves the file readable", async () => {
  const root = await tempRoot("secret");
  await makeRepo(root, { remote: "https://x-access-token:ghp_SECRETVALUE@github.com/o/r.git" });
  const state = join(root, ".state");
  const env = { HOME: "/nonexistent-home", OPENVIKING_STATE_DIR: state };

  const identity = resolveWorkspaceIdentity({ cwd: root, env });
  assert.equal(identity.remote, "github.com/o/r");
  assert.equal(identity.vars.git_remote, "github.com-o-r");

  for (const name of await readdir(state)) {
    const file = join(state, name);
    assert.doesNotMatch(await readFile(file, "utf-8"), /ghp_SECRETVALUE|x-access-token/);
    assert.equal((await stat(file)).mode & 0o777, 0o600, `${name} should be 0600`);
  }
});

test("a cache entry stamped in the future is not trusted", async () => {
  const root = await tempRoot("clock");
  await makeRepo(root, { remote: "git@github.com:o/first.git" });
  const env = { HOME: "/nonexistent-home", OPENVIKING_STATE_DIR: join(root, ".state") };

  resolveWorkspaceIdentity({ cwd: root, env, now: 5_000_000 });
  await writeFile(join(root, ".git", "config"), '[remote "origin"]\n\turl = git@github.com:o/second.git\n');

  const earlier = resolveWorkspaceIdentity({ cwd: root, env, now: 1_000_000 });
  assert.equal(earlier.vars.git_remote, "github.com-o-second", "a clock that ran fast must not pin the old value");
});

test("a cache holding the wrong shape is re-derived instead of returned", async () => {
  const root = await tempRoot("shape");
  await makeRepo(root, { remote: "git@github.com:o/r.git" });
  const state = join(root, ".state");
  const env = { HOME: "/nonexistent-home", OPENVIKING_STATE_DIR: state };

  resolveWorkspaceIdentity({ cwd: root, env, now: 1000 });
  for (const name of await readdir(state)) {
    await writeFile(join(state, name), JSON.stringify({ ts: 1000, identity: "not-an-object" }));
  }

  const identity = resolveWorkspaceIdentity({ cwd: root, env, now: 1000 });
  assert.equal(identity.vars.git_remote, "github.com-o-r");
});

test("findWorkspaceRoot returns nothing rather than throwing on a dead cwd", () => {
  const cwd = process.cwd();
  const gone = join(cwd, "definitely-not-here", "nested");
  assert.deepEqual(findWorkspaceRoot("", { HOME: "/nonexistent-home" }), { root: "", git: null });
  assert.equal(typeof findWorkspaceRoot(gone, { HOME: "/nonexistent-home" }).root, "string");
});
