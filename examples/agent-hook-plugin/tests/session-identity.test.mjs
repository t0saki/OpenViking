import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { expectExit, runHookScript, withMockOpenViking, writeJson } from "../../memory-plugin-shared/testing/support.mjs";

const entry = fileURLToPath(new URL("../scripts/hook.mjs", import.meta.url));
const native = "01a0c6e3-6400-7000-8000-00009e3a1c07";
async function withHooks(fn) {
  const root = await mkdtemp(join(tmpdir(), "ov-thin-identity-"));
  try {
    await withMockOpenViking(async (_req, res) => writeJson(res, { status: "ok", result: {} }), async (url) => {
      const env = { HOME: root, TMPDIR: root, OPENVIKING_HOME: root,
        OPENVIKING_HOOK_STATE_DIR: join(root, "state"), OPENVIKING_NO_AUTO_INJECT: "1",
        OPENVIKING_URL: url, OPENVIKING_WRITE_PATH_ASYNC: "0", OPENVIKING_MEMORY_ENABLED: "1" };
      const run = async (client, event, input = {}) => expectExit(await runHookScript(entry, { argv: [event, client], input: { cwd: root, ...input }, env }));
      const stateFile = (client, id = native) => join(root, "state", client, id + ".json");
      const state = async (client, id = native) => JSON.parse(await readFile(stateFile(client, id), "utf8"));
      const seed = async (client, value) => { await mkdir(join(root, "state", client), { recursive: true }); await writeFile(stateFile(client), JSON.stringify(value)); };
      await fn({ run, state, seed, root });
    });
  } finally { await rm(root, { recursive: true, force: true }); }
}

test("all thin hosts pin readable IDs under the lock, including duplicate session starts", async () => {
  await withHooks(async ({ run, state }) => {
    for (const [client, harness, event] of [["cursor", "cursor", "sessionStart"], ["trae", "trae", "session-start"], ["trae-cn", "traecn", "session-start"], ["zcode", "zcode", "session-start"]]) {
      await Promise.all([run(client, event, { session_id: native }), run(client, event, { session_id: native })]);
      assert.equal((await state(client)).ovSessionId, harness + "-20260922-021306-9e3a1c07");
      await run(client, event, { session_id: native });
      assert.equal((await state(client)).ovSessionId, harness + "-20260922-021306-9e3a1c07");
    }
  });
});

test("legacy activity and persisted mappings are retained on upgrade", async () => {
  await withHooks(async ({ run, state, seed }) => {
    await seed("trae", { promptAt: 10, capturedHashes: ["old"] });
    await run("trae", "session-start", { session_id: native });
    assert.equal((await state("trae")).ovSessionId, "tr-" + native);
    await seed("zcode", { ovSessionId: "zc-persisted", lastSessionStartAt: Date.now() });
    await run("zcode", "session-start", { session_id: native });
    assert.equal((await state("zcode")).ovSessionId, "zc-persisted");
  });
});

test("empty ZCode capture still persists identity and cwd fallbacks remain legacy", async () => {
  await withHooks(async ({ run, state, root }) => {
    const result = await run("zcode", "stop", { session_id: native });
    assert.equal(result.stdout, "");
    assert.equal((await state("zcode")).ovSessionId, "zcode-20260922-021306-9e3a1c07");
    await run("trae", "session-start");
    // Use the runtime's fallback key rather than duplicating its hashing rule.
    const { resolveNativeSessionId } = await import("../../memory-plugin-shared/lib/agent-hook-runtime.mjs");
    const fallback = resolveNativeSessionId({ cwd: root });
    assert.equal((await state("trae", fallback)).ovSessionId, "tr-" + fallback);
  });
});
