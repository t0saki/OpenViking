import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

test("Claude compressor profile probes once, caches, and recovers after runtime failure", async () => {
  const root = await mkdtemp(join(tmpdir(), "ov-cc-profile-"));
  const stateDir = join(root, "state");
  const binDir = join(root, "bin");
  const oldState = process.env.OPENVIKING_CC_STATE_DIR;
  const oldPath = process.env.PATH;
  try {
    await mkdir(binDir, { recursive: true });
    const claude = join(binDir, "claude");
    await writeFile(claude, "#!/bin/sh\nexit 0\n");
    await chmod(claude, 0o755);
    process.env.OPENVIKING_CC_STATE_DIR = stateDir;
    process.env.PATH = `${binDir}:${oldPath || ""}`;

    const profileModule = await import(`./recall-compressor-profile.mjs?test=${Date.now()}`);
    const cfg = { recallCompress: true, recallCompressModel: "", timeoutMs: 2000 };
    const detected = await profileModule.detectRecallCompressorProfile(cfg);
    assert.deepEqual(detected, { enabled: true, model: "haiku", source: "claude_version" });

    const cached = await profileModule.loadCachedRecallCompressorProfile(cfg);
    assert.deepEqual(cached, detected);
    await profileModule.markRecallCompressorRuntimeFailed(cfg);
    assert.equal((await profileModule.loadCachedRecallCompressorProfile(cfg)).source, "runtime_failed");

    const recovered = await profileModule.detectRecallCompressorProfile(cfg);
    assert.equal(recovered.enabled, true);
    assert.equal(recovered.source, "claude_version");
    const persisted = JSON.parse(await readFile(join(stateDir, "recall-compressor-profile.json"), "utf8"));
    assert.equal(persisted.profile.enabled, true);
  } finally {
    if (oldState === undefined) delete process.env.OPENVIKING_CC_STATE_DIR;
    else process.env.OPENVIKING_CC_STATE_DIR = oldState;
    if (oldPath === undefined) delete process.env.PATH;
    else process.env.PATH = oldPath;
    await rm(root, { recursive: true, force: true });
  }
});
