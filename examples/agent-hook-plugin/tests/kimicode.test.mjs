import assert from "node:assert/strict";
import { createServer } from "node:http";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { expectExit, runHookScript } from "../../memory-plugin-shared/testing/support.mjs";
import { evaluateHostUriGuard } from "../scripts/uri-guard.mjs";

const pluginRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const hookEntry = join(pluginRoot, "scripts", "hook.mjs");

const readHostJson = (name) => JSON.parse(
  readFileSync(join(pluginRoot, "hosts", "kimicode", name), "utf8"),
);

test("Kimi Code integration ships the files the installer reads", () => {
  for (const file of [
    "hosts/kimicode/hooks.json",
    "hosts/kimicode/.mcp.json",
    "hosts/kimicode/openviking.integration.json",
    "hosts/kimicode/kimi.plugin.json",
    "hosts/kimicode.mjs",
    "hosts/kimicode-turns.mjs",
    "hosts/kimicode-capture.mjs",
    "scripts/uri-guard.mjs",
  ]) {
    assert.ok(existsSync(join(pluginRoot, file)), `${file} must exist`);
  }
  assert.deepEqual(readHostJson("openviking.integration.json").clients, ["kimicode"]);
});

test("the hooks template is the flat [[hooks]] array Kimi Code merges into config.toml", () => {
  const hooks = readHostJson("hooks.json").hooks;
  assert.ok(Array.isArray(hooks), "Kimi Code hooks are an array, not an event tree");
  assert.deepEqual(hooks.map((hook) => hook.event), [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "Stop",
    "PreCompact",
    "SessionEnd",
    "Interrupt",
  ]);
  for (const hook of hooks) {
    assert.ok(!("matcher" in hook) || hook.matcher, "an empty matcher is omitted, not written");
    assert.ok(Number.isFinite(hook.timeout), `${hook.event} needs a timeout`);
    assert.match(hook.command, /^node __OPENVIKING_PLUGIN_ROOT__\/scripts\//);
    assert.match(hook.command, /__OPENVIKING_CLIENT_ID__$/);
  }
  assert.equal(hooks.find((hook) => hook.event === "PreToolUse").matcher, "Read|Glob|Grep");
});

test("the native plugin manifest declares the same hooks as the installer template", () => {
  const template = readHostJson("hooks.json").hooks;
  const manifest = readHostJson("kimi.plugin.json");
  assert.deepEqual(
    manifest.hooks.map(({ event, matcher }) => ({ event, matcher })),
    template.map(({ event, matcher }) => ({ event, matcher })),
  );
  // `/plugins install <this directory>` runs the commands from here, so they
  // reach the shared entry two levels up rather than through the installer's
  // placeholder.
  for (const hook of manifest.hooks) {
    assert.match(hook.command, /^node \.\.\/\.\.\/scripts\/[a-z-]+\.mjs( [a-z-]+)? kimicode$/);
  }
  assert.deepEqual(manifest.mcpServers.openviking.args, ["../../servers/mcp-proxy.mjs"]);
});

test(".mcp.json templates the plugin root the installer expands", () => {
  const mcp = readFileSync(join(pluginRoot, "hosts", "kimicode", ".mcp.json"), "utf8");
  assert.ok(mcp.includes("__OPENVIKING_PLUGIN_ROOT__"), "the installer expands only its own spelling");
  assert.equal(
    JSON.parse(mcp).mcpServers.openviking.env.OPENVIKING_HOOK_SOURCE,
    "kimicode",
    "the proxy resolves its harness from the client id in its environment",
  );
});

test("Kimi Code URI guard answers the PreToolUse deny contract", () => {
  const denied = evaluateHostUriGuard("kimicode", {
    tool_name: "Read",
    tool_input: { file_path: "viking://~/memories/profile.md" },
  });
  assert.equal(denied.hookSpecificOutput?.permissionDecision, "deny");
  assert.match(denied.hookSpecificOutput?.permissionDecisionReason ?? "", /viking:\/\//);
  assert.match(denied.hookSpecificOutput?.permissionDecisionReason ?? "", /OpenViking MCP read/);
  assert.deepEqual(evaluateHostUriGuard("kimicode", {
    tool_name: "Read",
    tool_input: { file_path: "/tmp/readme.md" },
  }), {});
});

function startMockOpenViking(t, { delayMs = 0 } = {}) {
  const requests = [];
  let completedResponses = 0;
  const server = createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      requests.push({ url: request.url, body });
      let payload = { result: { ok: true } };
      if (request.url?.startsWith("/api/v1/search/")) {
        payload = { result: { rendered: "kimi memory", entries: [], stats: {} } };
      } else if (request.url?.includes("/api/v1/content/read")) {
        payload = { result: "I prefer concise answers." };
      } else if (request.url?.includes("/api/v1/fs/ls")) {
        payload = { result: [] };
      }
      const answer = () => {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(JSON.stringify(payload));
        completedResponses++;
      };
      if (delayMs) setTimeout(answer, delayMs);
      else answer();
    });
  });
  t.after(() => server.close());
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({
      requests,
      url: `http://127.0.0.1:${server.address().port}`,
      get completedResponses() { return completedResponses; },
    }));
  });
}

function hookEnv(t, mock, extra = {}) {
  const root = mkdtempSync(join(tmpdir(), "openviking-kimicode-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  return {
    HOME: root,
    KIMI_CODE_HOME: join(root, ".kimi-code"),
    OPENVIKING_URL: mock.url,
    OPENVIKING_HOOK_STATE_DIR: join(root, "state"),
    OPENVIKING_MEMORY_ENABLED: "1",
    ...extra,
  };
}

const runKimicodeHook = (event, input, env) => runHookScript(hookEntry, {
  argv: [event, "kimicode"],
  input,
  env,
});

test("the prompt hook injects plain text, never a JSON envelope", async (t) => {
  const mock = await startMockOpenViking(t);
  const env = hookEnv(t, mock);
  const base = { session_id: "session_plain", cwd: "/workspace" };

  const first = expectExit(await runKimicodeHook(
    "user-prompt-submit",
    { ...base, prompt: "what did we decide?", prompt_id: "p-1" },
    env,
  ));
  assert.ok(first.stdout.trim(), "the prompt hook has to answer with the context it injects");
  assert.ok(
    !first.stdout.trimStart().startsWith("{"),
    "Kimi Code appends stdout verbatim, so a JSON document would be shown to the user",
  );
  assert.ok(!first.stdout.includes("hookSpecificOutput"));
  assert.match(first.stdout, /kimi memory/);
  assert.match(first.stdout, /I prefer concise answers\./);

  const second = expectExit(await runKimicodeHook(
    "user-prompt-submit",
    { ...base, prompt: "and the deadline?", prompt_id: "p-2" },
    env,
  ));
  assert.match(second.stdout, /kimi memory/);
  assert.ok(
    !second.stdout.includes("I prefer concise answers."),
    "the profile rides the first prompt of a session only",
  );
});

test("the observation-only events answer with nothing at all", async (t) => {
  const mock = await startMockOpenViking(t);
  const env = hookEnv(t, mock);
  const base = { session_id: "session_quiet", cwd: "/workspace" };

  const start = expectExit(await runKimicodeHook("session-start", base, env));
  assert.equal(start.stdout, "", "SessionStart output is dropped by the host");
  assert.ok(
    !mock.requests.some(({ url }) => url?.includes("/api/v1/content/read")),
    "a profile the host would throw away is never fetched",
  );

  for (const event of ["stop", "pre-compact", "session-end", "interrupt"]) {
    const run = expectExit(await runKimicodeHook(event, base, env));
    assert.equal(run.stdout, "", `${event} must stay silent`);
  }
});

test("Interrupt captures inline instead of detaching", async (t) => {
  // Interrupt fires instead of Stop while the host is tearing the turn down, so
  // the write has to finish in this process even with the async path enabled.
  const mock = await startMockOpenViking(t, { delayMs: 400 });
  const env = hookEnv(t, mock, { OPENVIKING_WRITE_PATH_ASYNC: "1" });

  expectExit(await runKimicodeHook("interrupt", {
    session_id: "session_interrupt",
    cwd: "/workspace",
    prompt: "how many retries?",
    responseText: "the retry budget is three attempts",
  }, env));

  assert.ok(
    mock.requests.some(({ url }) => url?.includes("/messages")),
    "an interrupted turn is captured before the host reaps the hook",
  );
  assert.ok(mock.completedResponses > 0, "the interrupt hook awaited its own writes");
});
