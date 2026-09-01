import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  entryPath,
  identityKey,
  listEntries,
  readEntry,
  registryDir,
  rememberPreviousPeer,
  slotName,
  writeEntry,
} from "./lib/workspace-registry.mjs";

async function home() {
  return { OPENVIKING_HOME: await mkdtemp(join(tmpdir(), "ov-registry-")) };
}

const repo = { isGit: true, remote: "github.com/volcengine/openviking", gitCommonDir: "/x/.git" };
const otherRepo = { isGit: true, remote: "github.com/someone/else", gitCommonDir: "/x/.git" };

test("a slot is readable and still unique per absolute path", () => {
  const a = slotName("/Users/x/src/api");
  const b = slotName("/Users/x/work/api");
  assert.match(a, /^api-[0-9a-f]{12}\.json$/);
  assert.notEqual(a, b, "same basename, different path, different slot");
  assert.equal(a, slotName("/Users/x/src/api"), "slots are stable");
  assert.match(slotName("/"), /^[0-9a-f]{12}\.json$/);
});

test("an entry round-trips and only the caller's keys change", async () => {
  const env = await home();
  const root = "/Users/x/src/api";

  writeEntry(root, { settings: { recall: { max_items: 7 } } }, { identity: repo, env, now: 1000 });
  writeEntry(root, { peer: { id: "pinned" } }, { identity: repo, env, now: 2000 });

  const { entry } = readEntry(root, { identity: repo, env });
  assert.deepEqual(entry.settings, { recall: { max_items: 7 } }, "an earlier write is not erased");
  assert.deepEqual(entry.peer, { id: "pinned" });
  assert.equal(entry.root, root);
  assert.equal(entry.label, repo.remote);

  const raw = JSON.parse(await readFile(entryPath(root, env), "utf-8"));
  assert.equal(raw.version, 1);
  assert.equal(raw.first_seen_at, 1000, "first_seen_at survives later writes");
  assert.equal(raw.last_seen_at, 2000);
});

test("entries are written 0600 — nobody else on the machine reads them", async () => {
  const env = await home();
  writeEntry("/Users/x/src/api", { peer: { id: "p" } }, { identity: repo, env });
  const mode = (await stat(entryPath("/Users/x/src/api", env))).mode & 0o777;
  assert.equal(mode, 0o600, `expected 0600, got ${mode.toString(8)}`);
});

test("a path reused by a different repository does not inherit the old peer", async () => {
  const env = await home();
  const root = "/Users/x/src/api";
  writeEntry(root, { peer: { id: "from-the-old-repo" } }, { identity: repo, env });

  const miss = readEntry(root, { identity: otherRepo, env });
  assert.equal(miss.entry, null, "conflicting git identity is a miss, not a match");
  assert.equal(miss.conflict, true);
  assert.ok(miss.warnings.some((w) => w.includes("different repository")));

  const hit = readEntry(root, { identity: repo, env });
  assert.equal(hit.entry.peer.id, "from-the-old-repo", "the matching identity still reads it");
});

test("writing after a conflict replaces the entry instead of merging into it", async () => {
  const env = await home();
  const root = "/Users/x/src/api";
  writeEntry(root, { peer: { id: "old" }, settings: { recall: { max_items: 3 } } }, { identity: repo, env });

  writeEntry(root, { peer: { id: "new" } }, { identity: otherRepo, env });
  const { entry } = readEntry(root, { identity: otherRepo, env });
  assert.equal(entry.peer.id, "new");
  assert.equal(entry.settings, undefined, "nothing carries over from the repository that used to be here");
});

test("identityKey prefers the remote and degrades honestly", () => {
  assert.equal(identityKey(repo), `remote:${repo.remote}`);
  assert.equal(identityKey({ isGit: true, gitCommonDir: "/x/.git" }), "git:/x/.git");
  assert.equal(identityKey({ isGit: false }), "path");
  assert.equal(identityKey(null), "path");
});

test("cli_config_profile is a name, never a path", async () => {
  const env = await home();
  const root = "/Users/x/src/api";

  writeEntry(root, { cli_config_profile: "work" }, { identity: repo, env });
  assert.equal(readEntry(root, { identity: repo, env }).entry.cli_config_profile, "work");

  for (const bad of ["../../etc/ovcli.conf", "/abs/path", "Work", "has space", ""]) {
    assert.throws(
      () => writeEntry(root, { cli_config_profile: bad }, { identity: repo, env }),
      /cli_config_profile/,
      `should reject ${JSON.stringify(bad)}`,
    );
  }
});

test("a profile name that got into the file by other means is dropped on read", async () => {
  const env = await home();
  const root = "/Users/x/src/api";
  await mkdir(registryDir(env), { recursive: true });
  await writeFile(
    entryPath(root, env),
    JSON.stringify({ version: 1, identity: identityKey(repo), cli_config_profile: "../../elsewhere", peer: { id: "p" } }),
  );

  const { entry, warnings } = readEntry(root, { identity: repo, env });
  assert.equal(entry.cli_config_profile, undefined);
  assert.equal(entry.peer.id, "p", "the rest of the entry still applies");
  assert.ok(warnings.some((w) => w.includes("cli_config_profile")));
});

test("previous peers accumulate without duplicates", async () => {
  const env = await home();
  const root = "/Users/x/src/api";

  rememberPreviousPeer(root, "-Users-x-src-api", { identity: repo, env });
  rememberPreviousPeer(root, "-Users-x-src-api", { identity: repo, env });
  rememberPreviousPeer(root, "-Users-x-old-path", { identity: repo, env });
  assert.equal(rememberPreviousPeer(root, "", { identity: repo, env }), null);

  const { entry } = readEntry(root, { identity: repo, env });
  assert.deepEqual(entry.previous_peer_ids, ["-Users-x-src-api", "-Users-x-old-path"]);
});

test("listEntries reports what is there and skips what is not readable", async () => {
  const env = await home();
  writeEntry("/Users/x/b", { peer: { id: "b" } }, { identity: repo, env });
  writeEntry("/Users/x/a", { peer: { id: "a" } }, { identity: repo, env });
  await writeFile(join(registryDir(env), "corrupt.json"), "{ nope");

  const entries = listEntries(env);
  assert.deepEqual(entries.map((e) => e.entry.root), ["/Users/x/a", "/Users/x/b"]);
});

test("an empty registry lists nothing rather than throwing", async () => {
  assert.deepEqual(listEntries({ OPENVIKING_HOME: "/nonexistent-openviking-home" }), []);
});

// --- regressions found by review ------------------------------------------

test("re-spelling origin is not a different repository", () => {
  const ssh = { isGit: true, remote: "github.com/o/r", gitCommonDir: "/x/.git" };
  const https = { isGit: true, remote: "github.com/o/r", gitCommonDir: "/x/.git" };
  assert.equal(identityKey(ssh), identityKey(https));
});

test("an entry from a newer client is refused, not flattened", async () => {
  const env = await home();
  const root = "/Users/x/src/api";
  await mkdir(registryDir(env), { recursive: true });
  const future = { version: 2, cli_config_profile: "prod", peer: { id: "pinned" }, important: "future" };
  await writeFile(entryPath(root, env), JSON.stringify(future));

  assert.throws(
    () => writeEntry(root, { peer: { id: "mine" } }, { identity: repo, env }),
    /newer client/,
  );
  assert.deepEqual(JSON.parse(await readFile(entryPath(root, env), "utf-8")), future);
});

test("a free-form section in the registry survives a round trip", async () => {
  const env = await home();
  const root = "/Users/x/src/api";

  writeEntry(root, { settings: { labels: { user: "alice", team: "core" } } }, { identity: repo, env });
  const { entry, warnings } = readEntry(root, { identity: repo, env });

  assert.deepEqual(entry.settings.labels, { user: "alice", team: "core" });
  assert.deepEqual(warnings, []);
});

test("two writers to one workspace: the later write wins, and says so", async () => {
  const env = await home();
  const root = "/Users/x/src/api";
  writeEntry(root, { settings: { recall: { max_items: 3 } } }, { identity: repo, env });

  // A per-workspace file removes contention between workspaces, not between the
  // hooks of one session. Both readers see the same state, then both write.
  const a = readEntry(root, { identity: repo, env });
  const b = readEntry(root, { identity: repo, env });
  assert.deepEqual(a.entry.settings, b.entry.settings);

  writeEntry(root, { peer: { id: "from-a" } }, { identity: repo, env });
  writeEntry(root, { previous_peer_ids: ["from-b"] }, { identity: repo, env });

  const final = readEntry(root, { identity: repo, env }).entry;
  assert.equal(final.peer.id, "from-a", "a sequential write still merges");
  assert.deepEqual(final.previous_peer_ids, ["from-b"]);
  assert.deepEqual(final.settings, { recall: { max_items: 3 } }, "neither write erased the settings");
});
