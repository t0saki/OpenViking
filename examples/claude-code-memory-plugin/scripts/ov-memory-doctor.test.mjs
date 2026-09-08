import assert from "node:assert/strict";
import test from "node:test";

import { credentialSources } from "./ov-memory-doctor.mjs";

const CLI_PATH = "/nowhere/.openviking/ovcli.conf";
const OV_PATH = "/nowhere/.openviking/ov.conf";

const cliConf = (data) => ({ ok: true, path: CLI_PATH, data });
const ovConf = (data) => ({ ok: true, path: OV_PATH, data });

const OV_FILE = ovConf({
  server: { url: "http://ov:1933", root_api_key: "sk-root" },
  claude_code: { apiKey: "sk-section", accountId: "ov-acct", userId: "ov-usr" },
});

const ENV_KEYS = [
  "OPENVIKING_URL",
  "OPENVIKING_BASE_URL",
  "OPENVIKING_API_KEY",
  "OPENVIKING_BEARER_TOKEN",
  "OPENVIKING_ACCOUNT",
  "OPENVIKING_USER",
];

function withEnv(vars, fn) {
  const saved = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
  try {
    for (const key of ENV_KEYS) delete process.env[key];
    Object.assign(process.env, vars);
    return fn();
  } finally {
    for (const key of ENV_KEYS) {
      if (saved[key] === undefined) delete process.env[key];
      else process.env[key] = saved[key];
    }
  }
}

test("env credentials outrank both files when nothing pins the chain", () => {
  withEnv({ OPENVIKING_URL: "http://env:1933", OPENVIKING_API_KEY: "sk-env", OPENVIKING_ACCOUNT: "env-acct" }, () => {
    const rows = credentialSources({ credentialSource: "env" }, cliConf({ url: "http://cli:1933", api_key: "sk-cli" }), OV_FILE);
    assert.equal(rows.url, "env");
    assert.equal(rows.apiKey, "env OPENVIKING_API_KEY");
    assert.equal(rows.account, "env");
  });
});

test("ovcli.conf mode reports the file the chain actually reads, not the environment", () => {
  const env = { OPENVIKING_URL: "http://env:1933", OPENVIKING_API_KEY: "sk-env", OPENVIKING_ACCOUNT: "env-acct", OPENVIKING_USER: "env-usr" };
  const cfg = { credentialSource: "ovcli" };

  withEnv(env, () => {
    const rows = credentialSources(cfg, cliConf({ url: "http://cli:1933", api_key: "sk-cli", account: "cli-acct" }), OV_FILE);
    assert.equal(rows.url, CLI_PATH);
    assert.equal(rows.apiKey, CLI_PATH);
    assert.equal(rows.account, CLI_PATH);
    assert.equal(rows.user, "(unset)");
  });

  // ovcli.conf carries no key of its own, so the chain drops to its `plugin`
  // section and then to ov.conf — never to the environment.
  withEnv(env, () => {
    const rows = credentialSources(cfg, cliConf({ url: "http://cli:1933", plugin: { claude_code: { apiKey: "sk-plugin" } } }), OV_FILE);
    assert.equal(rows.apiKey, `${CLI_PATH} plugin.claude_code.apiKey`);
  });

  withEnv(env, () => {
    const rows = credentialSources(cfg, cliConf({ url: "http://cli:1933" }), OV_FILE);
    assert.equal(rows.apiKey, `${OV_PATH} claude_code.apiKey`);
  });

  withEnv(env, () => {
    const rows = credentialSources(cfg, cliConf({ url: "http://cli:1933" }), ovConf({ server: { root_api_key: "sk-root" } }));
    assert.equal(rows.apiKey, `${OV_PATH} server.root_api_key`);
    assert.equal(rows.account, "(unset)");
  });

  withEnv(env, () => {
    const rows = credentialSources(cfg, cliConf({ url: "http://cli:1933" }), ovConf({}));
    assert.match(rows.apiKey, /ovcli\.conf mode ignores env/);
  });
});
