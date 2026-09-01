import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { applyCliConfigProfile, resolveCliConfigProfile } from "./lib/plugin-config.mjs";
import { writeEntry } from "./lib/workspace-registry.mjs";
import { resolveWorkspaceIdentity } from "./lib/workspace-identity.mjs";

async function fixture({ profile = "", createProfileFile = true } = {}) {
  const base = realpathSync(await mkdtemp(join(tmpdir(), "ov-profile-")));
  const root = join(base, "repo");
  await mkdir(join(root, ".git"), { recursive: true });
  await writeFile(join(root, ".git", "config"), '[remote "origin"]\n\turl = git@github.com:o/r.git\n');
  const ovHome = join(base, "home", ".openviking");
  await mkdir(ovHome, { recursive: true });
  const env = { HOME: join(base, "home"), OPENVIKING_HOME: ovHome, OPENVIKING_STATE_DIR: join(base, ".state") };
  if (profile) {
    if (createProfileFile) await writeFile(join(ovHome, `ovcli.conf.${profile}`), '{"url":"https://work.example"}');
    writeEntry(root, { cli_config_profile: profile }, { identity: resolveWorkspaceIdentity({ cwd: root, env, cache: false }), env });
  }
  return { root, env, ovHome };
}

test("a registry profile selects the ovcli.conf the workspace authenticates with", async () => {
  const { root, env, ovHome } = await fixture({ profile: "work" });
  assert.equal(resolveCliConfigProfile(root, env), join(ovHome, "ovcli.conf.work"));
  assert.equal(applyCliConfigProfile(root, env).OPENVIKING_CLI_CONFIG_FILE, join(ovHome, "ovcli.conf.work"));
});

test("no profile registered leaves the credential chain untouched", async () => {
  const { root, env } = await fixture();
  assert.equal(resolveCliConfigProfile(root, env), "");
  assert.equal(applyCliConfigProfile(root, env), env, "the common path must not even copy the env");
});

test("an explicit env override outranks the registry", async () => {
  const { root, env } = await fixture({ profile: "work" });
  const overridden = { ...env, OPENVIKING_CLI_CONFIG_FILE: "/somewhere/else.conf" };
  assert.equal(applyCliConfigProfile(root, overridden).OPENVIKING_CLI_CONFIG_FILE, "/somewhere/else.conf");
});

test("a registered profile that does not exist is a hard error, never a silent fallback", async () => {
  const { root, env } = await fixture({ profile: "missing", createProfileFile: false });
  assert.throws(() => resolveCliConfigProfile(root, env), /profile 'missing'/);
  assert.throws(() => applyCliConfigProfile(root, env), /does not exist/);
});
