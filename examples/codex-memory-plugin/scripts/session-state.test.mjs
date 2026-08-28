import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const STATE_DIR = await mkdtemp(join(tmpdir(), "ov-session-state-"));
process.env.OPENVIKING_CODEX_STATE_DIR = STATE_DIR;

const { clearEnded, markEnded, readEndedAt, withSessionLock } = await import("./session-state.mjs");

async function exists(path) {
  try { await stat(path); return true; } catch { return false; }
}

test("releasing a lock taken over by someone else leaves it alone", async () => {
  const dir = join(STATE_DIR, "takeover.lock");
  const outcome = await withSessionLock("takeover", async () => {
    // Simulate another taker that renamed our lock aside and re-created it.
    await rm(dir, { recursive: true, force: true });
    await mkdir(dir);
    await writeFile(join(dir, "owner"), "9999:other-owner");
    return "done";
  }, { waitMs: 0 });

  assert.equal(outcome.skipped, false);
  assert.equal(outcome.value, "done");
  assert.equal(await exists(dir), true, "the new holder's lock must survive our release");
  assert.equal(await readFile(join(dir, "owner"), "utf-8"), "9999:other-owner");
  await rm(dir, { recursive: true, force: true });
});

test("concurrent stale takeovers leave exactly one holder", async () => {
  const dir = join(STATE_DIR, "stale.lock");
  await mkdir(dir, { recursive: true });
  const old = new Date(Date.now() - 600_000);
  await utimes(dir, old, old);

  let inside = 0;
  let maxInside = 0;
  const run = () => withSessionLock("stale", async () => {
    inside += 1;
    maxInside = Math.max(maxInside, inside);
    await new Promise((resolve) => setTimeout(resolve, 120));
    inside -= 1;
    return true;
  }, { waitMs: 5_000, staleMs: 300_000 });

  const outcomes = await Promise.all([run(), run(), run()]);
  assert.equal(maxInside, 1, "a stale takeover must never hand the lock to two takers");
  assert.equal(outcomes.filter((o) => o.value === true).length, 3);
  assert.equal(await exists(dir), false, "the last holder releases the lock");
});

test("clearEnded honours the `before` cutoff", async () => {
  const at = await markEnded("cutoff");
  assert.ok(at > 0);

  await clearEnded("cutoff", { before: at });
  assert.equal(await readEndedAt("cutoff"), at, "a marker at the cutoff is kept");

  await clearEnded("cutoff", { before: at - 1 });
  assert.equal(await readEndedAt("cutoff"), at, "a marker newer than the cutoff is kept");

  await clearEnded("cutoff", { before: at + 1 });
  assert.equal(await readEndedAt("cutoff"), undefined, "an older marker is removed");

  await markEnded("cutoff");
  await clearEnded("cutoff");
  assert.equal(await readEndedAt("cutoff"), undefined, "no cutoff clears unconditionally");
});

test.after(async () => {
  await rm(STATE_DIR, { recursive: true, force: true });
});
