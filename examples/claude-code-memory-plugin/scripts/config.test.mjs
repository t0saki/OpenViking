import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { loadConfig } from "./config.mjs";

const CONFIG_ENV_KEYS = [
  "OPENVIKING_CLI_CONFIG_FILE",
  "OPENVIKING_CONFIG_FILE",
  "OPENVIKING_RECALL_COMPRESS",
  "OPENVIKING_RECALL_REWRITE",
];

async function withConfigEnv(values, action) {
  const previous = Object.fromEntries(CONFIG_ENV_KEYS.map((key) => [key, process.env[key]]));
  for (const key of CONFIG_ENV_KEYS) delete process.env[key];
  Object.assign(process.env, values);
  try {
    await action();
  } finally {
    for (const key of CONFIG_ENV_KEYS) {
      if (previous[key] === undefined) delete process.env[key];
      else process.env[key] = previous[key];
    }
  }
}

test("OPENVIKING_RECALL_COMPRESS is the canonical Claude compression mode", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-cc-config-"));
  try {
    await withConfigEnv({
      OPENVIKING_CLI_CONFIG_FILE: join(dir, "missing-ovcli.conf"),
      OPENVIKING_CONFIG_FILE: join(dir, "missing-ov.conf"),
      OPENVIKING_RECALL_COMPRESS: "client",
      OPENVIKING_RECALL_REWRITE: "server",
    }, () => {
      assert.equal(loadConfig().recallRewrite, "client");
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Claude still accepts the legacy OPENVIKING_RECALL_REWRITE alias", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-cc-config-"));
  try {
    await withConfigEnv({
      OPENVIKING_CLI_CONFIG_FILE: join(dir, "missing-ovcli.conf"),
      OPENVIKING_CONFIG_FILE: join(dir, "missing-ov.conf"),
      OPENVIKING_RECALL_REWRITE: "auto",
    }, () => {
      assert.equal(loadConfig().recallRewrite, "auto");
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Claude prefers recallCompress over recallRewrite in ovcli.conf", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-cc-config-"));
  const cliConfig = join(dir, "ovcli.conf");
  try {
    await writeFile(cliConfig, JSON.stringify({
      url: "http://127.0.0.1:1933",
      plugin: {
        claude_code: {
          recallCompress: "server",
          recallRewrite: "client",
        },
      },
    }));
    await withConfigEnv({
      OPENVIKING_CLI_CONFIG_FILE: cliConfig,
      OPENVIKING_CONFIG_FILE: join(dir, "missing-ov.conf"),
    }, () => {
      assert.equal(loadConfig().recallRewrite, "server");
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
