import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";

const root = mkdtempSync(join(tmpdir(), "ov-cc-pin-"));
const previousHome = process.env.OPENVIKING_HOME;
process.env.OPENVIKING_HOME = root;
const { resolveOvSessionId, resolveSubagentOvSessionId } = await import("./ov-session.mjs");
const stateDir = join(root, "state");
const pinPath = (id) => join(stateDir, "ov-session-" + id + ".json");
after(() => {
  if (previousHome === undefined) delete process.env.OPENVIKING_HOME;
  else process.env.OPENVIKING_HOME = previousHome;
  rmSync(root, { recursive: true, force: true });
});

test("startup publishes a private pin before exposing a readable ID, reused by every hook", async () => {
  const id = randomUUID();
  const ovId = await resolveOvSessionId(id, { mint: true, source: "startup", fetchJSON: () => { throw new Error("must not probe"); } });
  assert.match(ovId, /^claude-\d{8}-\d{6}-[a-z0-9]{8}$/);
  const pin = JSON.parse(readFileSync(pinPath(id), "utf8"));
  assert.equal(pin.ovSessionId, ovId);
  assert.equal(pin.nativeSessionId, id);
  assert.equal(pin.format, "readable");
  assert.equal(statSync(pinPath(id)).mode & 0o777, 0o600);
  assert.equal(await resolveOvSessionId(id), ovId);
  assert.equal(await resolveOvSessionId(id, { mint: true, source: "resume" }), ovId);
  assert.equal(await resolveSubagentOvSessionId(id, "agent:one/two"), ovId + "__subagent-agent-one-two");
});

test("resume mints only after a definite legacy 404; existing, offline and ambiguous sessions stay legacy", async () => {
  for (const response of [{ ok: true, status: 200 }, { ok: false, status: 503 }, { ok: false, status: 0 }, { ok: false, status: 404 }, null]) {
    const id = randomUUID();
    const expectedNew = response?.status === 404;
    const ovId = await resolveOvSessionId(id, { mint: true, source: "resume", fetchJSON: async (path) => {
      assert.equal(path, "/api/v1/sessions/cc-" + id);
      if (!response) throw new Error("offline");
      return response;
    } });
    if (expectedNew) assert.match(ovId, /^claude-/);
    else assert.equal(ovId, "cc-" + id);
    assert.equal(JSON.parse(readFileSync(pinPath(id))).ovSessionId, ovId);
  }
});

test("a write hook with no pin freezes legacy identity and prevents later minting", async () => {
  const id = randomUUID();
  assert.equal(await resolveOvSessionId(id), "cc-" + id);
  assert.equal(await resolveOvSessionId(id, { mint: true, source: "clear" }), "cc-" + id);
});

test("invalid or unreadable pins skip OV work without replacing the pin", async () => {
  for (const contents of ["", "{", JSON.stringify({ version: 2 }), JSON.stringify({ version: 1, nativeSessionId: "someone-else", ovSessionId: "cc-other" })]) {
    const id = randomUUID();
    writeFileSync(pinPath(id), contents);
    assert.equal(await resolveOvSessionId(id, { mint: true, source: "startup" }), null);
    assert.equal(readFileSync(pinPath(id), "utf8"), contents);
  }
  const id = randomUUID();
  mkdirSync(pinPath(id));
  assert.equal(await resolveOvSessionId(id), null);
  assert.equal(await resolveSubagentOvSessionId(id, "child"), null);
});

test("failed publication skips instead of returning an unpinned identity", async () => {
  const file = join(root, "blocked");
  writeFileSync(file, "not a directory");
  const code = 'const {resolveOvSessionId}=await import(' + JSON.stringify(new URL("./ov-session.mjs", import.meta.url).href) + '); process.stdout.write(JSON.stringify(await resolveOvSessionId("blocked-session",{mint:true,source:"startup"})));';
  const child = spawn(process.execPath, ["--input-type=module", "-e", code], { env: { ...process.env, OPENVIKING_HOME: file } });
  let output = "";
  child.stdout.on("data", (data) => { output += data; });
  await new Promise((resolve, reject) => { child.on("error", reject); child.on("close", (code) => code === 0 ? resolve() : reject(new Error("child exit " + code))); });
  assert.equal(output, "null");
});

test("independent processes racing different candidate timestamps converge on one complete pin", async () => {
  const id = randomUUID();
  const moduleUrl = new URL("./ov-session.mjs", import.meta.url).href;
  const workers = [0, 60000].map((offset) => {
    const code = 'const {resolveOvSessionId}=await import(' + JSON.stringify(moduleUrl) + '); Date.now=()=>1789980000000+' + offset + '; process.stdout.write("ready\\n"); process.stdin.once("data",async()=>{const id=await resolveOvSessionId(' + JSON.stringify(id) + ',{mint:true,source:"startup"});process.stdout.write(id);process.stdin.pause();});';
    const child = spawn(process.execPath, ["--input-type=module", "-e", code], { env: { ...process.env, OPENVIKING_HOME: root } });
    let output = "";
    const ready = new Promise((resolve) => child.stdout.once("data", resolve));
    child.stdout.on("data", (chunk) => { output += chunk; });
    const result = new Promise((resolve, reject) => { child.on("error", reject); child.on("close", (code) => code === 0 ? resolve(output.replace("ready\n", "")) : reject(new Error("child exit " + code))); });
    return { child, ready, result };
  });
  await Promise.all(workers.map((w) => w.ready));
  for (const { child } of workers) child.stdin.end("go");
  const ids = await Promise.all(workers.map((w) => w.result));
  assert.equal(ids[0], ids[1]);
  assert.equal(JSON.parse(readFileSync(pinPath(id))).ovSessionId, ids[0]);
  assert.equal(readdirSync(stateDir).some((name) => name.endsWith(".tmp")), false);
});
