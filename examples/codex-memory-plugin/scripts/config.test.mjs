import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { loadConfig } from "./config.mjs";

test("OPENVIKING_RECALL_COMPRESS=off disables Codex compression", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-codex-config-"));
  const keys = [
    "OPENVIKING_CLI_CONFIG_FILE",
    "OPENVIKING_CONFIG_FILE",
    "OPENVIKING_RECALL_COMPRESS",
  ];
  const previous = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
  try {
    process.env.OPENVIKING_CLI_CONFIG_FILE = join(dir, "missing-ovcli.conf");
    process.env.OPENVIKING_CONFIG_FILE = join(dir, "missing-ov.conf");
    process.env.OPENVIKING_RECALL_COMPRESS = "off";
    assert.equal(loadConfig().recallCompress, false);
  } finally {
    for (const key of keys) {
      if (previous[key] === undefined) delete process.env[key];
      else process.env[key] = previous[key];
    }
    await rm(dir, { recursive: true, force: true });
  }
});

test("shared ovcli recallCompress=off disables Codex compression", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-codex-config-"));
  const cliConfig = join(dir, "ovcli.conf");
  const keys = [
    "OPENVIKING_CLI_CONFIG_FILE",
    "OPENVIKING_CONFIG_FILE",
    "OPENVIKING_RECALL_COMPRESS",
  ];
  const previous = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
  try {
    await writeFile(cliConfig, JSON.stringify({
      url: "http://127.0.0.1:1933",
      plugin: {
        recallCompress: "off",
      },
    }));
    process.env.OPENVIKING_CLI_CONFIG_FILE = cliConfig;
    process.env.OPENVIKING_CONFIG_FILE = join(dir, "missing-ov.conf");
    delete process.env.OPENVIKING_RECALL_COMPRESS;
    assert.equal(loadConfig().recallCompress, false);
  } finally {
    for (const key of keys) {
      if (previous[key] === undefined) delete process.env[key];
      else process.env[key] = previous[key];
    }
    await rm(dir, { recursive: true, force: true });
  }
});
