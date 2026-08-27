import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  assessProbes,
  assessReady,
  classifyFetchError,
  createReport,
  describeApiKey,
  effectiveServerAuthMode,
  inspectJsonFile,
  isPlaceholderSecret,
  lintBaseUrl,
  lintServerConf,
  readCollectionMeta,
  readyCheckState,
  scanDebugLog,
  scanServerLog,
  serverLogTarget,
  serverWorkspace,
  unknownOvcliKeys,
} from "./lib/doctor-core.mjs";

const b64 = (s) => Buffer.from(s).toString("base64url");

test("describeApiKey decodes identity segments and masks the secret", () => {
  const key = `${b64("acme")}.${b64("alice")}.${b64("0123456789abcdef0123456789abcdef")}`;
  const info = describeApiKey(key);
  assert.equal(info.format, "v2");
  assert.equal(info.account, "acme");
  assert.equal(info.user, "alice");
  assert.match(info.display, /^account=acme user=alice secret=[A-Za-z0-9_-]{4}…[A-Za-z0-9_-]{4}$/);
  assert.ok(!info.display.includes(key.split(".")[2]));
  assert.deepEqual(info.problems, []);
});

test("describeApiKey shows head/tail only for legacy keys", () => {
  const key = "ff6e58" + "a".repeat(54) + "f956";
  const info = describeApiKey(key);
  assert.equal(info.format, "legacy");
  assert.equal(info.display, "ff6e…f956 (64 chars, legacy single-segment format)");
  assert.equal(describeApiKey("").present, false);
  assert.equal(describeApiKey("short").display, "***** (5 chars, legacy single-segment format)");
});

test("describeApiKey flags copy/paste damage", () => {
  assert.ok(describeApiKey("Bearer abcdefghijkl").problems.some((p) => p.includes("Bearer")));
  assert.ok(describeApiKey(" abcdefghijkl\n").problems.some((p) => p.includes("whitespace")));
  const mangled = describeApiKey("YWNt.not*base64.abc");
  assert.equal(mangled.format, "v2");
  assert.ok(mangled.problems.some((p) => p.includes("not decodable")));
});

test("lintBaseUrl catches the common url mistakes", () => {
  assert.deepEqual(lintBaseUrl("https://ov.example.com"), []);
  assert.deepEqual(lintBaseUrl("https://api.vikingdb.cn-beijing.volces.com/openviking"), []);
  assert.ok(lintBaseUrl("ov.example.com").some((p) => p.level === "fail" && /scheme/.test(p.message)));
  assert.ok(lintBaseUrl("https://ov.example.com/api/v1").some((p) => /\/api\/v1/.test(p.message)));
  assert.ok(lintBaseUrl("https://ov.example.com/mcp").some((p) => /\/mcp/.test(p.message)));
  assert.ok(lintBaseUrl("http://0.0.0.0:1933").some((p) => /0\.0\.0\.0/.test(p.message)));
  assert.ok(lintBaseUrl("").some((p) => p.level === "fail"));
});

test("classifyFetchError maps node errors to hints", () => {
  assert.equal(classifyFetchError({ cause: { code: "ECONNREFUSED" } }).kind, "refused");
  assert.equal(classifyFetchError({ cause: { code: "ENOTFOUND" } }).kind, "dns");
  assert.equal(classifyFetchError({ name: "AbortError", message: "aborted" }).kind, "timeout");
  assert.equal(classifyFetchError({ cause: { code: "UNABLE_TO_VERIFY_LEAF_SIGNATURE", message: "unable to verify" } }).kind, "tls");
});

test("inspectJsonFile reports parse errors instead of swallowing them", () => {
  const dir = mkdtempSync(join(tmpdir(), "ov-doctor-"));
  const path = join(dir, "ovcli.conf");
  writeFileSync(path, '{"url": "https://x", "api_key": "y",}');
  const broken = inspectJsonFile(path);
  assert.equal(broken.exists, true);
  assert.equal(broken.ok, false);
  assert.match(broken.error, /invalid JSON/);
  writeFileSync(path, '{"url": "https://x", "apiKey": "y", "plugin": {}}');
  const parsed = inspectJsonFile(path);
  assert.equal(parsed.ok, true);
  assert.deepEqual(unknownOvcliKeys(parsed.data), ["apiKey"]);
  assert.equal(inspectJsonFile(join(dir, "missing.conf")).exists, false);
});

test("assessProbes treats /health without identity as a rejected key", () => {
  const conn = { baseUrl: "https://ov.example.com", apiKey: "k".repeat(64) };
  const health = { ok: true, status: 200, latencyMs: 50, json: { status: "ok", version: "0.4.17", auth_mode: "api_key" } };
  const report = createReport();
  const summary = assessProbes(report, { health, healthAuth: { ...health }, systemStatus: { ok: false, status: 401, json: { error: { message: "Invalid API Key" } } } }, conn, describeApiKey(conn.apiKey));
  assert.equal(summary.reachable, true);
  assert.equal(summary.authOk, false);
  const titles = report.problems().map((p) => p.title);
  assert.ok(titles.some((t) => t.startsWith("api key rejected")));
  assert.ok(titles.some((t) => t.includes("system/status → 401")));
  assert.equal(report.exitCode(), 1);
});

test("assessProbes accepts an echoed identity and warns about root keys", () => {
  const conn = { baseUrl: "https://ov.example.com", apiKey: "k".repeat(64), account: "other" };
  const health = { ok: true, status: 200, latencyMs: 50, json: { status: "ok", version: "0.4.17", auth_mode: "api_key" } };
  const healthAuth = { ...health, json: { ...health.json, account_id: "default", user_id: "default", role: "root" } };
  const report = createReport();
  const summary = assessProbes(report, { health, healthAuth }, conn, describeApiKey(conn.apiKey));
  assert.equal(summary.authOk, true);
  assert.deepEqual(summary.identity, { account: "default", user: "default", role: "root" });
  const titles = report.problems().map((p) => p.title);
  assert.ok(titles.some((t) => t.includes("ROOT api key")));
  assert.ok(titles.some((t) => t.includes("differ from the key's identity")));
});

test("assessProbes reports unreachable servers with the classified cause", () => {
  const report = createReport();
  assessProbes(report, { health: { ok: false, status: 0, error: "connect ECONNREFUSED", errorKind: "refused", errorHint: "connection refused" } }, { baseUrl: "http://127.0.0.1:1933" }, describeApiKey(""));
  assert.equal(report.exitCode(), 1);
  assert.match(report.render(), /server unreachable/);
});

test("scanDebugLog separates recent errors from history", () => {
  const dir = mkdtempSync(join(tmpdir(), "ov-doctor-log-"));
  const path = join(dir, "cc-hooks.log");
  const now = Date.now();
  const line = (ts, hook, stage, error) => JSON.stringify(error ? { ts, hook, stage, error } : { ts, hook, stage, data: {} });
  writeFileSync(path, [
    line(new Date(now - 30 * 86400000).toISOString(), "auto-capture", "transcript_read", { message: "old" }),
    line(new Date(now - 3600000).toISOString(), "mcp-proxy", "start", null).replace('"data":{}', '"data":{"mcpUrl":"https://ov.example.com/mcp"}'),
    line(new Date(now - 60000).toISOString(), "auto-recall", "uncaught", { message: "fresh" }),
    "not json",
  ].join("\n"));
  const scan = scanDebugLog(path);
  assert.equal(scan.exists, true);
  assert.deepEqual(scan.hooks, ["auto-capture", "auto-recall", "mcp-proxy"]);
  assert.equal(scan.errors.length, 2);
  assert.deepEqual(scan.recentErrors.map((e) => e.message), ["fresh"]);
  assert.equal(scan.proxyStart.data.mcpUrl, "https://ov.example.com/mcp");
});

test("report render lists problems in the summary and sets the exit code", () => {
  const report = createReport();
  report.section("A");
  report.ok("fine");
  report.warn("meh", "detail", "do this");
  assert.equal(report.exitCode(), 0);
  report.fail("bad", "", "do that");
  assert.equal(report.exitCode(), 1);
  const text = report.render();
  assert.match(text, /== Summary ==\n  1 failure\(s\), 1 warning\(s\)/);
  assert.match(text, /✗ bad → do that/);
  assert.match(text, /⚠ meh → do this/);
});

test("lintServerConf flags what stops a local server from starting", () => {
  const findings = lintServerConf({
    _comment: "from the example",
    claude_code: { enabled: true },
    server: { host: "0.0.0.0", port: 1933, url: "http://x", workers: 2, root_api_key: "" },
    storage: { workspace: "./data", vectordb: { dimension: 768 } },
    embedding: { dense: { provider: "ollama", model: "nomic-embed-text", api_base: "http://localhost:11434", dimension: 1024, api_key: "$UNSET_DOCTOR_TEST_VAR" } },
  }, { baseUrl: "http://127.0.0.1:1934", env: {} });
  const messages = findings.map((f) => `${f.level}:${f.message}`);
  const has = (level, re) => assert.ok(messages.some((m) => m.startsWith(`${level}:`) && re.test(m)), `missing ${level} ${re}\n${messages.join("\n")}`);
  has("warn", /'claude_code' block/);
  has("warn", /server\.url is rejected/);
  has("warn", /template keys.*_comment/);
  has("fail", /root_api_key is an empty string/);
  has("fail", /auth mode resolves to 'dev' but server\.host is '0\.0\.0\.0'/);
  has("warn", /talks to port 1934, but this ov\.conf starts the server on port 1933/);
  has("warn", /workspace '\.\/data' is relative/);
  has("warn", /\$UNSET_DOCTOR_TEST_VAR, which is unset/);
  has("warn", /lacks the \/v1 suffix/);
  has("warn", /vectordb\.dimension \(768\) differs/);
  has("warn", /no vlm section/);
  has("warn", /workers = 2/);
});

test("lintServerConf accepts a sane local config and enforces provider credentials", () => {
  const good = lintServerConf({
    server: { host: "127.0.0.1", port: 1933 },
    storage: { workspace: "/srv/ov/data" },
    embedding: { dense: { provider: "volcengine", model: "m", api_key: "$DOCTOR_TEST_KEY", dimension: 1024 } },
    vlm: { provider: "volcengine", model: "v", api_key: "k" },
  }, { baseUrl: "http://127.0.0.1:1933", env: { DOCTOR_TEST_KEY: "x" } });
  assert.deepEqual(good.filter((f) => f.level !== "info"), []);
  assert.ok(good.some((f) => f.level === "info" && /\$DOCTOR_TEST_KEY \(set in this environment/.test(f.message)));

  const bad = lintServerConf({
    embedding: { dense: { provider: "volcengine", model: "m", api_key: "{your-api-key}" }, },
    vlm: { provider: "openai", model: "gpt" },
  }, { baseUrl: "http://127.0.0.1:1933", env: {} });
  assert.ok(bad.some((f) => f.level === "fail" && /embedding\.dense\.api_key is required/.test(f.message)));
  assert.ok(bad.some((f) => f.level === "warn" && /looks like a placeholder/.test(f.message)));
  assert.ok(bad.some((f) => f.level === "fail" && /vlm\.api_key is required/.test(f.message)));
  assert.ok(bad.some((f) => f.level === "warn" && /storage\.workspace is not set/.test(f.message)));
  assert.ok(lintServerConf({ embedding: { dense: { provider: "litellm", model: "x/y" } } }, { baseUrl: "http://127.0.0.1:1933", env: {} }).some((f) => /dimension is required for provider 'litellm'/.test(f.message)));
  assert.equal(effectiveServerAuthMode({ root_api_key: "abc" }), "api_key");
  assert.equal(effectiveServerAuthMode({ auth_mode: " trusted " }), "trusted");
  assert.equal(effectiveServerAuthMode({}), "dev");
});

test("assessReady interprets the readiness checks", () => {
  const ok = { ok: true, status: 200, json: { status: "ready", checks: { agfs: { status: "ok", checks: { filesystem: "ok", multiwrite_sync: "not_supported" } }, vectordb: "ok", api_key_manager: "not_configured", embedding: "ok", ollama: "not_configured" } } };
  let report = createReport();
  assert.equal(assessReady(report, ok).ready, true);
  assert.equal(report.exitCode(), 0);
  assert.match(report.render(), /\/ready: agfs ok, vectordb ok/);

  const failing = { ok: false, status: 503, json: { status: "not_ready", checks: { agfs: { status: "error", checks: { filesystem: "error: boom" } }, vectordb: "ok", api_key_manager: "ok", embedding: "error: probe timed out (provider unreachable)", ollama: "not_configured" } } };
  report = createReport();
  const res = assessReady(report, failing, { workspace: "~/.openviking/data" });
  assert.equal(res.ready, false);
  assert.equal(res.checks.agfs, "error (filesystem: error: boom)");
  const titles = report.problems().map((p) => p.title);
  assert.ok(titles.some((t) => t.startsWith("/ready: embedding →")));
  assert.ok(titles.some((t) => t.startsWith("/ready: agfs →")));
  assert.ok(report.problems().some((p) => /~\/\.openviking\/data\/viking/.test(p.fix)));

  report = createReport();
  assert.equal(assessReady(report, { ok: false, status: 503, json: { status: "not_ready", reason: "initializing" } }).ready, false);
  assert.ok(report.problems().some((p) => p.level === "warn" && /still initializing/.test(p.title)));
  report = createReport();
  assert.equal(assessReady(report, { ok: false, status: 404, json: { detail: "Not Found" } }).ready, null);
  assert.equal(report.exitCode(), 0);
  assert.equal(readyCheckState("not_supported").ok, true);
  assert.equal(readyCheckState({ status: "ok", checks: { a: "ok", b: "error: x" } }).ok, false);
});

test("assessProbes recognises the docker pending-config stub", () => {
  const report = createReport();
  const health = { ok: false, status: 503, latencyMs: 3, json: { status: "pending_initialization", error: "OpenViking config file not found", config_file: "/app/.openviking/ov.conf", fix: ["mount ~/.openviking on the host to /app/.openviking"] } };
  const summary = assessProbes(report, { health }, { baseUrl: "http://127.0.0.1:1933" }, describeApiKey(""));
  assert.equal(summary.reachable, true);
  const problem = report.problems()[0];
  assert.match(problem.title, /docker container is up but has no ov\.conf/);
  assert.match(problem.detail, /mount ~\/\.openviking on the host/);
});

test("workspace, collection metadata and server log helpers read what the server writes", () => {
  const dir = mkdtempSync(join(tmpdir(), "ov-doctor-ws-"));
  const conf = join(dir, "ov.conf");
  writeFileSync(conf, "{}");
  assert.equal(serverWorkspace({ storage: { workspace: "./data" } }, { confPath: conf }).path, null);
  mkdirSync(join(dir, "data", "vectordb", "context"), { recursive: true });
  assert.equal(serverWorkspace({}, { confPath: conf }).source.startsWith("guessed"), true);
  assert.equal(serverWorkspace({ storage: { workspace: join(dir, "data") } }).source, "storage.workspace");

  writeFileSync(join(dir, "data", "vectordb", "context", "collection_meta.json"), JSON.stringify({
    CollectionName: "context",
    Description: 'Unified context collection\n\n[openviking.embedding]\n{"dimension": 1024, "model": "m", "model_identity": "m", "provider": "volcengine"}',
    Dimension: 1024,
  }));
  const meta = readCollectionMeta(join(dir, "data"));
  assert.equal(meta.exists, true);
  assert.equal(meta.dimension, 1024);
  assert.deepEqual(meta.embedding, { dimension: 1024, model: "m", model_identity: "m", provider: "volcengine" });
  assert.equal(readCollectionMeta(join(dir, "nope")).exists, false);

  assert.deepEqual(serverLogTarget({}, "/ws"), { kind: "stdout", path: null });
  assert.deepEqual(serverLogTarget({ log: { output: "file" } }, "/ws"), { kind: "file", path: "/ws/log/openviking.log" });
  assert.equal(serverLogTarget({ log: { output: "/var/log/ov.log" } }, "/ws").path, "/var/log/ov.log");

  const logPath = join(dir, "openviking.log");
  const stamp = (msAgo) => new Date(Date.now() - msAgo).toISOString().slice(0, 19).replace("T", " ");
  writeFileSync(logPath, [
    `${stamp(9 * 86400000)},001 - openviking.storage - ERROR - old failure`,
    `${stamp(3600000)},002 - openviking.server - INFO - fine`,
    `${stamp(60000)},003 - openviking.models.vlm.base - ERROR - Backup VLM also failed with error: 401`,
    "ERROR:    Application startup failed. Exiting.",
  ].join("\n"));
  const scan = scanServerLog(logPath);
  assert.equal(scan.errors.length, 2);
  assert.deepEqual(scan.recentErrors.map((e) => e.message), ["Backup VLM also failed with error: 401"]);
  assert.deepEqual(scan.fatal, ["ERROR:    Application startup failed. Exiting."]);
  assert.equal(isPlaceholderSecret("<paste>"), true);
  assert.equal(isPlaceholderSecret("sk-real"), false);
});
