import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));

function writeJson(res, value) {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(value));
}

async function withMockOpenViking(handler, fn) {
  const server = http.createServer((req, res) => {
    Promise.resolve(handler(req, res)).catch((error) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: String(error?.stack || error) }));
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    return await fn(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

function runSessionStart(input, env) {
  return new Promise((resolve, reject) => {
    const cleanEnv = { ...process.env };
    for (const key of Object.keys(cleanEnv)) {
      if (key.startsWith("OPENVIKING_")) delete cleanEnv[key];
    }
    const child = spawn(process.execPath, [join(SCRIPT_DIR, "session-start-commit.mjs")], {
      env: { ...cleanEnv, ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`session-start-commit exited ${code}: ${stderr}`));
        return;
      }
      resolve({ output: JSON.parse(stdout.trim()), stderr });
    });
    child.stdin.end(JSON.stringify(input));
  });
}

function baseEnv(baseUrl, stateDir) {
  return {
    OPENVIKING_URL: baseUrl,
    OPENVIKING_CREDENTIAL_SOURCE: "env",
    OPENVIKING_CONFIG_FILE: join(stateDir, "missing-ov.conf"),
    OPENVIKING_CLI_CONFIG_FILE: join(stateDir, "missing-ovcli.conf"),
    OPENVIKING_CODEX_STATE_DIR: stateDir,
    OPENVIKING_RECALL_COMPRESS_DETECT_ON_STARTUP: "0",
    OPENVIKING_TIMEOUT_MS: "5000",
    OPENVIKING_CAPTURE_TIMEOUT_MS: "5000",
  };
}

function profileHandler(requests, { archiveOverview = "" } = {}) {
  return async (req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");
    requests.push({
      method: req.method,
      path: url.pathname,
      uri: url.searchParams.get("uri"),
      actorPeerId: req.headers["x-openviking-actor-peer"] || "",
    });

    if (req.method === "GET" && url.pathname === "/health") {
      writeJson(res, { status: "ok", result: { healthy: true } });
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/v1/system/status") {
      writeJson(res, { status: "ok", result: { user: "zeus" } });
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/v1/content/read") {
      writeJson(res, {
        status: "ok",
        result: "# Zeus\nWorks on OpenViking integrations.\nPrefers concise implementation notes.",
      });
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/v1/fs/ls") {
      const uri = url.searchParams.get("uri");
      if (uri === "viking://user") {
        writeJson(res, {
          status: "ok",
          result: [{ name: "zeus", isDir: true }],
        });
        return;
      }
      if (uri?.endsWith("/preferences")) {
        writeJson(res, {
          status: "ok",
          result: [{
            name: "workflow.md",
            rel_path: "zeus/workflow.md",
            abstract: "Prefer focused changes and targeted tests.",
            isDir: false,
          }],
        });
        return;
      }
      if (uri?.endsWith("/entities")) {
        writeJson(res, {
          status: "ok",
          result: [{
            name: "openviking.md",
            rel_path: "software/openviking.md",
            abstract: "OpenViking memory and context platform.",
            isDir: false,
          }],
        });
        return;
      }
    }
    if (req.method === "GET" && url.pathname.endsWith("/context")) {
      writeJson(res, {
        status: "ok",
        result: {
          latest_archive_overview: archiveOverview,
          pre_archive_abstracts: [],
        },
      });
      return;
    }
    if (req.method === "POST" && url.pathname.endsWith("/commit")) {
      writeJson(res, {
        status: "ok",
        result: {
          archived: true,
          task_id: "task-profile-test",
          trace_id: "trace-session-start",
        },
      });
      return;
    }
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "error", error: "not found" }));
  };
}

test("startup injects the shared profile block with workspace peer routing", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-session-start-"));
  const requests = [];
  try {
    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      const { output } = await runSessionStart(
        {
          session_id: "startup-profile",
          source: "startup",
          cwd: "/tmp/codex-profile",
          hook_event_name: "SessionStart",
        },
        baseEnv(baseUrl, stateDir),
      );

      assert.equal(output.hookSpecificOutput.hookEventName, "SessionStart");
      assert.match(output.hookSpecificOutput.additionalContext, /source="session-start"/);
      assert.match(output.hookSpecificOutput.additionalContext, /<user-profile uri="viking:\/\/user\/zeus\/memories\/profile\.md">/);
      assert.match(output.hookSpecificOutput.additionalContext, /Works on OpenViking integrations/);
      assert.match(output.hookSpecificOutput.additionalContext, /zeus\/workflow\.md/);
      assert.match(output.hookSpecificOutput.additionalContext, /software\/openviking\.md/);
      assert.equal(output.systemMessage, undefined);
    });

    const profileRequests = requests.filter((request) =>
      request.path === "/api/v1/content/read" || request.path === "/api/v1/fs/ls"
    );
    assert.ok(profileRequests.length >= 4);
    assert.ok(profileRequests.every((request) => request.actorPeerId === "-tmp-codex-profile"));
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

async function exists(path) {
  try { await stat(path); return true; } catch { return false; }
}

test("startup commits a session whose SessionEnd marker is still present", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-session-commit-"));
  const requests = [];
  try {
    await mkdir(stateDir, { recursive: true });
    const now = Date.now();
    await writeFile(join(stateDir, "old-session.json"), JSON.stringify({
      codexSessionId: "old-session",
      ovSessionId: "cx-old-session",
      capturedTurnCount: 2,
      createdAt: now - 1000,
      lastUpdatedAt: now,
    }));
    await writeFile(join(stateDir, "old-session.ended"), String(now));

    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      const { output } = await runSessionStart(
        {
          session_id: "new-session",
          source: "startup",
          cwd: "/tmp/codex-commit",
          hook_event_name: "SessionStart",
        },
        baseEnv(baseUrl, stateDir),
      );

      assert.match(output.hookSpecificOutput.additionalContext, /Works on OpenViking integrations/);
      assert.equal(
        output.systemMessage,
        "OpenViking session cx-old-session is committed (trace_id=trace-session-start)",
      );
    });

    assert.ok(requests.some((request) =>
      request.method === "POST"
      && request.path === "/api/v1/sessions/cx-old-session/commit"
    ));
    const state = JSON.parse(await readFile(join(stateDir, "old-session.json"), "utf-8"));
    assert.equal(state.ovSessionId, null);
    assert.equal(state.capturedTurnCount, 2);
    assert.equal(await exists(join(stateDir, "old-session.ended")), false);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("startup ignores committed cursor-only states", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-cursor-only-"));
  const requests = [];
  try {
    const now = Date.now();
    await Promise.all([
      writeFile(join(stateDir, "recent.json"), JSON.stringify({
        codexSessionId: "recent",
        ovSessionId: null,
        capturedTurnCount: 4,
        createdAt: now - 500,
        lastUpdatedAt: now,
      })),
      writeFile(join(stateDir, "stale.json"), JSON.stringify({
        codexSessionId: "stale",
        ovSessionId: null,
        capturedTurnCount: 6,
        createdAt: now - 20_000,
        lastUpdatedAt: now - 10_000,
      })),
    ]);

    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      await runSessionStart(
        { session_id: "new-session", source: "startup", cwd: "/tmp/codex-cursor-only" },
        {
          ...baseEnv(baseUrl, stateDir),
          OPENVIKING_CODEX_IDLE_TTL_MS: "5000",
        },
      );
    });

    const [recent, stale] = await Promise.all([
      readFile(join(stateDir, "recent.json"), "utf-8"),
      readFile(join(stateDir, "stale.json"), "utf-8"),
    ]);
    assert.equal(JSON.parse(recent).capturedTurnCount, 4);
    assert.equal(JSON.parse(stale).capturedTurnCount, 6);
    assert.equal(requests.some((request) => request.path.endsWith("/commit")), false);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("startup retires cursor-only states once their retention window closes", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-cursor-retention-"));
  const requests = [];
  try {
    const now = Date.now();
    await Promise.all([
      writeFile(join(stateDir, "expired.json"), JSON.stringify({
        codexSessionId: "expired",
        ovSessionId: null,
        capturedTurnCount: 6,
        createdAt: now - 20_000,
        lastUpdatedAt: now - 10_000,
      })),
      writeFile(join(stateDir, "keeper.json"), JSON.stringify({
        codexSessionId: "keeper",
        ovSessionId: null,
        capturedTurnCount: 4,
        createdAt: now - 8_000,
        lastUpdatedAt: now - 6_000,
      })),
      writeFile(join(stateDir, "empty.json"), JSON.stringify({
        codexSessionId: "empty",
        ovSessionId: null,
        capturedTurnCount: 0,
        createdAt: now - 8_000,
        lastUpdatedAt: now - 6_000,
      })),
    ]);

    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      await runSessionStart(
        { session_id: "new-session", source: "startup", cwd: "/tmp/codex-cursor-retention" },
        {
          ...baseEnv(baseUrl, stateDir),
          OPENVIKING_CODEX_IDLE_TTL_MS: "5000",
          OPENVIKING_CODEX_COMMITTED_TTL_MS: "8000",
        },
      );
    });

    const files = await readdir(stateDir);
    assert.equal(files.includes("expired.json"), false);
    assert.equal(files.includes("empty.json"), false);
    assert.equal(JSON.parse(await readFile(join(stateDir, "keeper.json"), "utf-8")).capturedTurnCount, 4);
    assert.equal(requests.some((request) => request.path.endsWith("/commit")), false);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("startup leaves a fresh live session alone until it ends or goes idle", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-active-live-"));
  const requests = [];
  try {
    const now = Date.now();
    await writeFile(join(stateDir, "live.json"), JSON.stringify({
      codexSessionId: "live",
      ovSessionId: "cx-live",
      capturedTurnCount: 2,
      createdAt: now - 2_000,
      lastUpdatedAt: now - 100,
    }));

    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      const { output } = await runSessionStart(
        { session_id: "new-session", source: "startup", cwd: "/tmp/codex-active-live" },
        { ...baseEnv(baseUrl, stateDir), OPENVIKING_CODEX_IDLE_TTL_MS: "1800000" },
      );
      assert.equal(output.systemMessage, undefined);
    });

    assert.equal(requests.some((request) => request.path.endsWith("/commit")), false);
    assert.equal(JSON.parse(await readFile(join(stateDir, "live.json"), "utf-8")).ovSessionId, "cx-live");
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("startup commits a live session once it passes the idle TTL", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-idle-commit-"));
  const requests = [];
  try {
    const now = Date.now();
    await writeFile(join(stateDir, "idle.json"), JSON.stringify({
      codexSessionId: "idle",
      ovSessionId: "cx-idle",
      capturedTurnCount: 3,
      createdAt: now - 60_000,
      lastUpdatedAt: now - 30_000,
    }));

    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      const { output } = await runSessionStart(
        { session_id: "new-session", source: "startup", cwd: "/tmp/codex-idle-commit" },
        { ...baseEnv(baseUrl, stateDir), OPENVIKING_CODEX_IDLE_TTL_MS: "5000" },
      );
      assert.match(output.systemMessage, /cx-idle is committed/);
    });

    const state = JSON.parse(await readFile(join(stateDir, "idle.json"), "utf-8"));
    assert.equal(state.ovSessionId, null);
    assert.equal(state.capturedTurnCount, 3);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("startup skips a session another writer already holds the lock for", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-sweep-lock-"));
  const requests = [];
  try {
    const now = Date.now();
    await writeFile(join(stateDir, "locked.json"), JSON.stringify({
      codexSessionId: "locked",
      ovSessionId: "cx-locked",
      capturedTurnCount: 2,
      createdAt: now - 2_000,
      lastUpdatedAt: now - 100,
    }));
    await writeFile(join(stateDir, "locked.ended"), String(now));
    await mkdir(join(stateDir, "locked.lock"), { recursive: true });

    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      const { output } = await runSessionStart(
        { session_id: "new-session", source: "startup", cwd: "/tmp/codex-sweep-lock" },
        baseEnv(baseUrl, stateDir),
      );
      assert.equal(output.systemMessage, undefined);
    });

    assert.equal(requests.some((request) => request.path.endsWith("/commit")), false);
    const state = JSON.parse(await readFile(join(stateDir, "locked.json"), "utf-8"));
    assert.equal(state.ovSessionId, "cx-locked");
    assert.equal(await exists(join(stateDir, "locked.ended")), true);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("resume clears a stale SessionEnd marker for the resumed session", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-resume-ended-"));
  const requests = [];
  try {
    const now = Date.now();
    await writeFile(join(stateDir, "resumed.json"), JSON.stringify({
      codexSessionId: "resumed",
      ovSessionId: null,
      capturedTurnCount: 3,
      createdAt: now - 2_000,
      lastUpdatedAt: now - 100,
    }));
    await writeFile(join(stateDir, "resumed.ended"), String(now));

    await withMockOpenViking(profileHandler(requests), async (baseUrl) => {
      await runSessionStart(
        { session_id: "resumed", source: "resume", cwd: "/tmp/codex-resume-ended" },
        baseEnv(baseUrl, stateDir),
      );
    });

    assert.equal(await exists(join(stateDir, "resumed.ended")), false);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});
