import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  buildKimicodeTurns,
  cleanKimicodeText,
  extractUnseenWireTurns,
} from "../hosts/kimicode-turns.mjs";

function wireFile(t, lines) {
  const dir = mkdtempSync(join(tmpdir(), "kc-wire-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const wire = join(dir, "wire.jsonl");
  writeFileSync(wire, `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`);
  return wire;
}

function withKimiHome(t, home) {
  const original = process.env.KIMI_CODE_HOME;
  process.env.KIMI_CODE_HOME = home;
  t.after(() => {
    if (original === undefined) delete process.env.KIMI_CODE_HOME;
    else process.env.KIMI_CODE_HOME = original;
    rmSync(home, { recursive: true, force: true });
  });
}

test("cleanKimicodeText strips openviking-context blocks", () => {
  const input = 'hello <openviking-context source="test">secret</openviking-context> world';
  assert.equal(cleanKimicodeText(input), "hello  world");
});

test("cleanKimicodeText strips relevant-memory blocks", () => {
  assert.equal(cleanKimicodeText("text <relevant-memory>old</relevant-memory> here"), "text  here");
});

test("extractUnseenWireTurns reads user prompt and assistant text parts", (t) => {
  const wire = wireFile(t, [
    { type: "metadata", protocol_version: "1.4" },
    { type: "turn.prompt", input: [{ type: "text", text: "hi" }], origin: { kind: "user" } },
    {
      type: "context.append_loop_event",
      event: { type: "content.part", turnId: "0", part: { type: "think", think: "ignore me" } },
    },
    {
      type: "context.append_loop_event",
      event: { type: "content.part", turnId: "0", part: { type: "text", text: "Hi! How can I help?" } },
    },
  ]);
  const { available, turns } = extractUnseenWireTurns(wire, null);
  assert.equal(available, true);
  assert.deepEqual(turns, [
    { role: "user", content: "hi", turnId: "0" },
    { role: "assistant", content: "Hi! How can I help?", turnId: "0" },
  ]);
});

test("extractUnseenWireTurns skips turns up to lastTurnId", (t) => {
  const wire = wireFile(t, [
    { type: "turn.prompt", input: [{ type: "text", text: "first" }] },
    {
      type: "context.append_loop_event",
      event: { type: "content.part", turnId: "0", part: { type: "text", text: "a1" } },
    },
    { type: "turn.prompt", input: [{ type: "text", text: "second" }] },
    {
      type: "context.append_loop_event",
      event: { type: "content.part", turnId: "1", part: { type: "text", text: "a2" } },
    },
  ]);
  assert.deepEqual(extractUnseenWireTurns(wire, "0").turns, [
    { role: "user", content: "second", turnId: "1" },
    { role: "assistant", content: "a2", turnId: "1" },
  ]);
});

test("turn.prompt plus append_message is a single user turn", (t) => {
  const wire = wireFile(t, [
    { type: "turn.prompt", input: [{ type: "text", text: "hi" }] },
    { type: "context.append_message", message: { role: "user", content: [{ type: "text", text: "hi" }] } },
    {
      type: "context.append_loop_event",
      event: { type: "content.part", turnId: "0", part: { type: "text", text: "hello" } },
    },
  ]);
  const { turns } = extractUnseenWireTurns(wire, null);
  assert.equal(turns.filter((turn) => turn.role === "user").length, 1);
  assert.deepEqual(turns, [
    { role: "user", content: "hi", turnId: "0" },
    { role: "assistant", content: "hello", turnId: "0" },
  ]);
});

test("extractUnseenWireTurns closes a cancelled turn on turn.ended", (t) => {
  const wire = wireFile(t, [
    { type: "turn.prompt", input: [{ type: "text", text: "first question" }], origin: { kind: "user" } },
    { type: "turn.ended", turnId: "0", reason: "cancelled" },
    { type: "turn.prompt", input: [{ type: "text", text: "second question" }], origin: { kind: "user" } },
    {
      type: "context.append_loop_event",
      event: { type: "content.part", turnId: "1", part: { type: "text", text: "answer" } },
    },
  ]);
  assert.deepEqual(extractUnseenWireTurns(wire, null).turns, [
    { role: "user", content: "first question", turnId: "0" },
    { role: "user", content: "second question", turnId: "1" },
    { role: "assistant", content: "answer", turnId: "1" },
  ]);
});

test("buildKimicodeTurns uses session_index.jsonl to find the wire log", (t) => {
  const home = mkdtempSync(join(tmpdir(), "kc-home-"));
  withKimiHome(t, home);
  const sessionId = "session_abc";
  const sessionDir = join(home, "sessions", "wd_x", sessionId);
  mkdirSync(join(sessionDir, "agents", "main"), { recursive: true });
  writeFileSync(
    join(home, "session_index.jsonl"),
    `${JSON.stringify({ sessionId, sessionDir, workDir: "/tmp" })}\n`,
  );
  writeFileSync(
    join(sessionDir, "agents", "main", "wire.jsonl"),
    [
      JSON.stringify({ type: "turn.prompt", input: [{ type: "text", text: "indexed" }] }),
      JSON.stringify({
        type: "context.append_loop_event",
        event: { type: "content.part", turnId: "7", part: { type: "text", text: "ok" } },
      }),
    ].join("\n") + "\n",
  );
  const turns = buildKimicodeTurns({ session_id: sessionId }, {});
  assert.equal(turns[0].content, "indexed");
  assert.equal(turns[1].turnId, "7");
});

test("buildKimicodeTurns falls back to stdin when wire is missing", (t) => {
  withKimiHome(t, mkdtempSync(join(tmpdir(), "kc-empty-")));
  assert.deepEqual(
    buildKimicodeTurns(
      { session_id: "session_missing", prompt: "hello", responseText: "world" },
      {},
    ),
    [
      { role: "user", content: "hello" },
      { role: "assistant", content: "world" },
    ],
  );
});

test("the stdin fallback reads the prompt the way the prompt hook does", (t) => {
  withKimiHome(t, mkdtempSync(join(tmpdir(), "kc-empty-")));
  const turns = buildKimicodeTurns(
    { session_id: "session_missing", input: [{ type: "text", text: "structured prompt" }] },
    {},
  );
  assert.deepEqual(turns, [{ role: "user", content: "structured prompt" }]);
});
