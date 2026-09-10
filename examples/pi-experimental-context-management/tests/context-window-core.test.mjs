import test from "node:test";
import assert from "node:assert/strict";

import {
  COMPACTION_SENTINEL,
  ContextWindowCore,
  REMINDER_CUSTOM_TYPE,
  STATUS_CUSTOM_TYPE,
  WINDOW_ENTRY_TYPE,
  WINDOW_HEADER_OPEN,
  applyWindowCut,
  buildHandoffMessage,
  buildStatusLine,
  buildWindowHeader,
  computeCutRange,
  effectiveThresholds,
  formatDuration,
  formatTokens,
  lastUserTimestamps,
  reminderText,
  windowConfig,
} from "../lib/context-window-core.mjs";
import { estimateTokens } from "../lib/text-budget.mjs";

// --------------------------------------------------------------------------
// fixtures
// --------------------------------------------------------------------------

const user = (content, timestamp) => (timestamp === undefined ? { role: "user", content } : { role: "user", content, timestamp });
const assistant = (content) => ({ role: "assistant", content });
const assistantCalls = (calls) => ({
  role: "assistant",
  content: calls.map((call) => ({ type: "toolCall", id: call.id, name: call.name })),
});
const toolResult = (toolCallId, toolName, content = "ok") => ({ role: "toolResult", toolCallId, toolName, content });

const ARCHIVE_URI = "viking://user/u1/sessions/pi-abc/history/archive_002";

function makeIo(overrides = {}) {
  const io = {
    clock: 1_700_000_000_000,
    calls: [],
    persisted: [],
    logs: [],
    handoffs: [],
    overview: "# Working Memory\n## Current State\nhalfway",
    task: { status: "running" },
    watermark: 42,
    connectedValue: true,
    sid: "pi-abc",
    syncBranch: async () => {
      io.calls.push("syncBranch");
      return { added: 3, tokens: 120, allDelivered: true };
    },
    flush: async (opts = {}) => {
      io.calls.push(`flush:${opts.budgetMs}`);
      return true;
    },
    postHandoff: async (text) => {
      io.calls.push("postHandoff");
      io.handoffs.push(text);
      return true;
    },
    commit: async (opts = {}) => {
      io.calls.push(`commit:${opts.keepRecentCount}`);
      return { status: "accepted", task_id: "task-1", archive_uri: ARCHIVE_URI };
    },
    readArchiveOverview: async () => {
      io.calls.push("readArchiveOverview");
      return io.overview;
    },
    getTask: async () => {
      io.calls.push("getTask");
      return io.task;
    },
    persistEntry: (type, data) => {
      io.persisted.push({ type, data });
    },
    getWatermark: () => io.watermark,
    now: () => io.clock,
    sleep: async (ms) => {
      io.clock += ms;
    },
    log: (message) => io.logs.push(message),
    connected: () => io.connectedValue,
    sessionId: () => io.sid,
    pendingCount: () => 2,
  };
  return Object.assign(io, overrides);
}

function makeCore(io, config = {}) {
  return new ContextWindowCore({ io, config });
}

function branchFor(anchor, extraCalls = []) {
  return [
    { id: "e1", type: "message", message: user("please split the work here", 10) },
    {
      id: "e2",
      type: "message",
      message: assistantCalls([{ id: anchor, name: "new_context" }, ...extraCalls]),
    },
  ];
}

async function openWindow(core, io, opts = {}) {
  return core.requestReset({
    reason: opts.reason ?? "phase one done",
    notes: opts.notes ?? "goal: ship the parser; next: write tests",
    nextSteps: opts.nextSteps ?? ["run the tests"],
    toolCallId: opts.toolCallId ?? "call-1",
    branch: opts.branch ?? branchFor(opts.toolCallId ?? "call-1"),
    signal: opts.signal ?? null,
    onProgress: opts.onProgress,
  });
}

// --------------------------------------------------------------------------
// config
// --------------------------------------------------------------------------

test("windowConfig applies defaults and clamps", () => {
  const defaults = windowConfig();
  assert.equal(defaults.resetDeadlineMs, 60000);
  assert.equal(defaults.archivePollMs, 2000);
  assert.equal(defaults.overviewRefreshMaxAttempts, 20);
  assert.equal(defaults.overviewBudget, 3000);
  assert.equal(defaults.notesBudget, 1500);
  assert.equal(defaults.pendingRequestBudget, 400);
  assert.equal(defaults.softPercent, 70);
  assert.equal(defaults.hardPercent, 85);
  assert.equal(defaults.idleGapMinutes, 30);
  assert.equal(defaults.statusEveryTurn, true);
  assert.equal(defaults.historyItemMaxChars, 8000);
  assert.equal(defaults.recentResetGuardMs, 60000);

  const clamped = windowConfig({
    resetDeadlineMs: 10,
    archivePollMs: 999999,
    overviewRefreshMaxAttempts: -5,
    overviewBudget: 1,
    notesBudget: 999999,
    pendingRequestBudget: -1,
    softPercent: 80,
    hardPercent: 40,
    idleGapMinutes: 99999,
    statusEveryTurn: false,
    historyItemMaxChars: 1,
  });
  assert.equal(clamped.resetDeadlineMs, 5000);
  assert.equal(clamped.archivePollMs, 30000);
  assert.equal(clamped.overviewRefreshMaxAttempts, 0);
  assert.equal(clamped.overviewBudget, 100);
  assert.equal(clamped.notesBudget, 20000);
  assert.equal(clamped.pendingRequestBudget, 0);
  assert.equal(clamped.softPercent, 80);
  assert.equal(clamped.hardPercent, 80, "hardPercent is forced to at least softPercent");
  assert.equal(clamped.idleGapMinutes, 1440);
  assert.equal(clamped.statusEveryTurn, false);
  assert.equal(clamped.historyItemMaxChars, 500);
  assert.deepEqual(windowConfig(windowConfig()), windowConfig(), "normalization is idempotent");
});

// --------------------------------------------------------------------------
// cut
// --------------------------------------------------------------------------

test("computeCutRange cuts a single tool call", () => {
  const messages = [
    user("hello"),
    assistant("hi"),
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    user("keep me"),
  ];
  const range = computeCutRange(messages, "call-1");
  assert.deepEqual(range, {
    assistantIndex: 2,
    resultIndex: 3,
    cutEndIndex: 3,
    droppedToolResults: 1,
    siblingToolNames: [],
  });
});

test("computeCutRange drops the whole parallel batch when the reset is first", () => {
  const messages = [
    user("hello"),
    assistantCalls([
      { id: "call-1", name: "new_context" },
      { id: "call-2", name: "read" },
      { id: "call-3", name: "grep" },
    ]),
    toolResult("call-1", "new_context"),
    toolResult("call-2", "read"),
    toolResult("call-3", "grep"),
    user("after"),
  ];
  const range = computeCutRange(messages, "call-1");
  assert.equal(range.cutEndIndex, 4);
  assert.equal(range.droppedToolResults, 3);
  assert.deepEqual(range.siblingToolNames, ["read", "grep"]);
});

test("computeCutRange drops the whole parallel batch when the reset is last", () => {
  const messages = [
    assistantCalls([
      { id: "call-2", name: "read" },
      { id: "call-3", name: "grep" },
      { id: "call-1", name: "new_context" },
    ]),
    toolResult("call-2", "read"),
    toolResult("call-3", "grep"),
    toolResult("call-1", "new_context"),
  ];
  const range = computeCutRange(messages, "call-1");
  assert.equal(range.assistantIndex, 0);
  assert.equal(range.resultIndex, 3);
  assert.equal(range.cutEndIndex, 3);
  assert.equal(range.droppedToolResults, 3);
  assert.deepEqual(range.siblingToolNames, ["read", "grep"]);

  const cut = applyWindowCut(messages, { anchorToolCallId: "call-1", headerText: "HEADER" });
  assert.equal(cut.applied, true);
  assert.equal(cut.messages.length, 1);
  assert.ok(!cut.messages.some((m) => m.role === "toolResult"), "no orphan tool_result survives");
});

test("computeCutRange returns null when the anchor is missing", () => {
  const messages = [user("hello"), assistant("hi")];
  assert.equal(computeCutRange(messages, "call-1"), null);
  assert.equal(computeCutRange(messages, ""), null);
  const cut = applyWindowCut(messages, { anchorToolCallId: "call-1", headerText: "HEADER" });
  assert.equal(cut.applied, false);
  assert.equal(cut.droppedCount, 0);
  assert.equal(cut.messages, messages, "the original array is returned untouched");
});

test("computeCutRange still cuts when the issuing assistant message is gone", () => {
  const messages = [user("hello"), toolResult("call-1", "new_context"), assistant("done")];
  const range = computeCutRange(messages, "call-1");
  assert.equal(range.assistantIndex, -1);
  assert.equal(range.resultIndex, 1);
  assert.equal(range.cutEndIndex, 1);
  assert.deepEqual(range.siblingToolNames, []);
});

test("applyWindowCut merges the header into a kept user message with string content", () => {
  const messages = [
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    user("what about the parser?", 99),
    assistant("looking"),
  ];
  const cut = applyWindowCut(messages, { anchorToolCallId: "call-1", headerText: "HEADER" });
  assert.equal(cut.messages.length, 2);
  assert.equal(cut.messages[0].role, "user");
  assert.equal(cut.messages[0].content, "HEADER\n\nwhat about the parser?");
  assert.equal(cut.messages[0].timestamp, 99);
  assert.equal(messages[2].content, "what about the parser?", "the input message is not mutated");
  assert.equal(cut.droppedCount, 2);
});

test("applyWindowCut merges the header into a kept user message with block content", () => {
  const blocks = [{ type: "image", url: "x" }, { type: "text", text: "second block" }];
  const messages = [
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    { role: "user", content: blocks },
  ];
  const cut = applyWindowCut(messages, { anchorToolCallId: "call-1", headerText: "HEADER" });
  assert.deepEqual(cut.messages[0].content, [
    { type: "image", url: "x" },
    { type: "text", text: "HEADER\n\nsecond block" },
  ]);
  assert.equal(blocks[1].text, "second block", "the input blocks are not mutated");
});

test("applyWindowCut inserts a standalone header when nothing is kept", () => {
  const messages = [assistantCalls([{ id: "call-1", name: "new_context" }]), toolResult("call-1", "new_context")];
  const cut = applyWindowCut(messages, {
    anchorToolCallId: "call-1",
    headerText: "HEADER",
    headerTimestamp: 4242,
  });
  assert.deepEqual(cut.messages, [{ role: "user", content: "HEADER", timestamp: 4242 }]);
  assert.equal(cut.droppedCount, 2);
});

test("transformContext is byte-identical across repeated calls", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);

  const messages = [
    user("old", 1),
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    assistant("still here"),
  ];
  const first = core.transformContext(messages);
  const second = core.transformContext(messages);
  assert.equal(JSON.stringify(first), JSON.stringify(second));
  assert.equal(first[0].content, second[0].content);
  assert.ok(first[0].content.startsWith(WINDOW_HEADER_OPEN));
  assert.equal(first.length, 2);
});

test("transformContext releases the boundary once when the anchor is gone", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  io.logs.length = 0;

  const messages = [user("fresh branch", 1), assistant("hi")];
  const out = core.transformContext(messages);
  assert.equal(out, messages);
  assert.equal(core.armed, false);
  assert.equal(core.windowIndex, 2, "the window index survives the release");
  core.transformContext(messages);
  assert.equal(io.logs.filter((line) => line.includes("anchor missing")).length, 1);
});

// --------------------------------------------------------------------------
// header / handoff / status text
// --------------------------------------------------------------------------

test("buildHandoffMessage renders reason, notes and optional next steps", () => {
  const withSteps = buildHandoffMessage({
    windowId: "w2",
    nextWindowId: "w3",
    reason: "topic switch",
    notes: "parser is green",
    nextSteps: ["write docs", "open the PR"],
  });
  assert.equal(
    withSteps,
    [
      "[Context Window Handoff] w2 -> w3",
      "Reason: topic switch",
      "Handoff notes:",
      "parser is green",
      "Next steps:",
      "1. write docs",
      "2. open the PR",
    ].join("\n"),
  );

  const withoutSteps = buildHandoffMessage({ windowId: "w1", nextWindowId: "w2", reason: "r", notes: "n" });
  assert.ok(!withoutSteps.includes("Next steps:"));
});

test("buildWindowHeader renders the ready variant with every section", () => {
  const header = buildWindowHeader({
    windowId: "w3",
    previousWindowId: "w2",
    archiveId: "archive_002",
    openedAt: Date.UTC(2026, 8, 10, 6, 41, 12),
    reason: "phase one done",
    notes: "goal: ship the parser",
    nextSteps: ["run the tests"],
    pendingRequest: "please also update the docs",
    overview: "# Working Memory\n## Current State\nhalfway",
    overviewState: "ready",
    siblingToolNames: [],
  });

  assert.ok(header.startsWith(WINDOW_HEADER_OPEN));
  assert.ok(header.includes('<context_window id="w3" previous="w2" archive="archive_002" opened="2026-09-10T06:41:12Z">'));
  assert.ok(header.includes("Reason you gave: phase one done"));
  assert.ok(header.includes("<handoff-notes>goal: ship the parser</handoff-notes>"));
  assert.ok(header.includes("1. run the tests"));
  assert.ok(header.includes("<pending-request>please also update the docs</pending-request>"));
  assert.ok(header.includes('<working-memory archive="archive_002"># Working Memory'));
  assert.ok(header.includes('history {"action":"list_windows"}'));
  assert.ok(header.includes('{"action":"list_items","window":"w2"}'));
  assert.ok(header.includes('{"action":"read_item","item":"w2:14"}'));
  assert.ok(header.includes('{"action":"search_contents","query":"..."}'));
  assert.ok(header.endsWith("</context_window>\n</openviking-context>"));
  assert.ok(!header.includes("discarded with the same batch"));
});

test("buildWindowHeader truncates Chinese notes, pending request and overview to their budgets", () => {
  const header = buildWindowHeader({
    windowId: "w2",
    previousWindowId: "w1",
    archiveId: "archive_001",
    notes: "笔".repeat(4000),
    pendingRequest: "请".repeat(4000),
    overview: "忆".repeat(9000),
    overviewState: "ready",
    config: { notesBudget: 100, pendingRequestBudget: 100, overviewBudget: 100 },
  });

  const notes = header.match(/<handoff-notes>([\s\S]*?)<\/handoff-notes>/)[1];
  const pending = header.match(/<pending-request>([\s\S]*?)<\/pending-request>/)[1];
  const overview = header.match(/<working-memory archive="archive_001">([\s\S]*?)<\/working-memory>/)[1];
  for (const chunk of [notes, pending, overview]) {
    assert.ok(chunk.endsWith("\n...(truncated)"), "each truncated section is marked");
    assert.ok(estimateTokens(chunk.replace("\n...(truncated)", "")) <= 100);
  }
});

test("buildWindowHeader pending variant emits no empty working-memory element on the first reset", () => {
  const header = buildWindowHeader({
    windowId: "w2",
    previousWindowId: "w1",
    archiveId: "archive_001",
    notes: "notes",
    overview: "",
    overviewState: "pending",
    previousOverview: "",
  });
  assert.ok(!header.includes("<working-memory"), "never render an empty working-memory element");
  assert.ok(header.includes("is not ready yet"));
  assert.ok(header.includes("use history to read the archived messages directly"));
});

test("buildWindowHeader pending variant attaches the previous overview as stale", () => {
  const header = buildWindowHeader({
    windowId: "w3",
    previousWindowId: "w2",
    archiveId: "archive_002",
    overviewState: "pending",
    previousOverview: "older working memory",
  });
  assert.ok(header.includes('<working-memory status="pending" archive="archive_002" stale="true">older working memory</working-memory>'));
  assert.ok(header.includes("is not ready yet"));
});

test("buildWindowHeader unavailable variant is a short note with no element", () => {
  const header = buildWindowHeader({
    windowId: "w3",
    previousWindowId: "w2",
    archiveId: "archive_002",
    overviewState: "unavailable",
    previousOverview: "older working memory",
  });
  assert.ok(!header.includes("<working-memory"));
  assert.ok(header.includes("is unavailable"));
});

test("buildWindowHeader mentions discarded siblings only when there are any", () => {
  const base = {
    windowId: "w2",
    previousWindowId: "w1",
    archiveId: "archive_001",
    overviewState: "ready",
    overview: "wm",
  };
  const one = buildWindowHeader({ ...base, siblingToolNames: ["read"] });
  assert.ok(one.includes("1 other tool result was discarded with the same batch; re-run it if needed. Tools: read."));
  const two = buildWindowHeader({ ...base, siblingToolNames: ["read", "grep"] });
  assert.ok(two.includes("2 other tool results were discarded with the same batch; re-run them if needed. Tools: read, grep."));
  const none = buildWindowHeader({ ...base, siblingToolNames: [] });
  assert.ok(!none.includes("discarded with the same batch"));
});

test("formatTokens and formatDuration", () => {
  assert.equal(formatTokens(0), "0");
  assert.equal(formatTokens(512), "512");
  assert.equal(formatTokens(38000), "38k");
  assert.equal(formatTokens(38400), "38.4k");
  assert.equal(formatTokens(262144), "262k");
  assert.equal(formatDuration(0), "0s");
  assert.equal(formatDuration(12_000), "12s");
  assert.equal(formatDuration(3 * 60_000), "3m");
  assert.equal(formatDuration(47 * 60_000), "47m");
  assert.equal(formatDuration(72 * 60_000), "1h 12m");
  assert.equal(formatDuration(60 * 60_000), "1h");
});

test("buildStatusLine renders the one-line status and the idle note", () => {
  const line = buildStatusLine({
    windowId: "w2",
    turnsInWindow: 6,
    usedTokens: 38000,
    contextWindow: 262000,
    sinceLastUserMs: 47 * 60_000,
    idleGapMs: 5 * 60_000,
    idleGapMinutes: 30,
  });
  assert.equal(line, "[context-status] window w2 · 6 turns · ~38k/262k tokens (15%) · 47m since your previous message");

  const idle = buildStatusLine({
    windowId: "w2",
    turnsInWindow: 1,
    usedTokens: 1000,
    contextWindow: 262000,
    estimated: true,
    sinceLastUserMs: 47 * 60_000,
    idleGapMs: 47 * 60_000,
    idleGapMinutes: 30,
  });
  const lines = idle.split("\n");
  assert.equal(lines.length, 2);
  assert.ok(lines[0].includes("(0%, estimated)"));
  assert.ok(lines[1].startsWith("NOTE: 47 minutes passed since the previous user message."));
  assert.ok(lines[1].includes("consider new_context before you begin"));

  const noIdle = buildStatusLine({ windowId: "w1", turnsInWindow: 1, idleGapMs: 99 * 60_000, idleGapMinutes: 0 });
  assert.equal(noIdle.split("\n").length, 1, "idleGapMinutes 0 disables the note");
});

test("reminderText mentions new_context, notes and harness compaction", () => {
  const soft = reminderText("soft", { windowId: "w2", usedTokens: 184000, contextWindow: 262000 });
  assert.ok(soft.includes("w2"));
  assert.ok(soft.includes("70%"));
  assert.ok(soft.includes("new_context"));
  assert.ok(soft.includes("notes"));
  assert.ok(soft.includes("the harness compacts it for you"));

  const hard = reminderText("hard", { windowId: "w2", usedTokens: 223000, contextWindow: 262000 });
  assert.ok(hard.includes("exactly one call to new_context now"));
  assert.ok(hard.includes("85%"));
  assert.ok(hard.includes("the harness compacts it for you"));
});

test("lastUserTimestamps skips everything this extension injects", () => {
  const messages = [
    user("first real", 100),
    assistant("answer"),
    user(`${WINDOW_HEADER_OPEN}\n<context_window id="w2">`, 200),
    { role: "user", content: "[context-status] window w2", timestamp: 210 },
    { role: "user", content: "anything", customType: STATUS_CUSTOM_TYPE, timestamp: 220 },
    { role: "user", content: "anything", customType: REMINDER_CUSTOM_TYPE, timestamp: 230 },
    { role: "user", content: "[Context Window Handoff] w1 -> w2", timestamp: 240 },
    user("second real", 300),
  ];
  assert.deepEqual(lastUserTimestamps(messages), [300, 100]);
  assert.deepEqual(lastUserTimestamps([]), []);
});

// --------------------------------------------------------------------------
// requestReset
// --------------------------------------------------------------------------

test("requestReset drives io in order and opens the window", async () => {
  const io = makeIo();
  const core = makeCore(io);
  const out = await openWindow(core, io);

  assert.equal(out.ok, true);
  assert.equal(out.kind, "reset");
  assert.deepEqual(io.calls, ["syncBranch", "flush:15000", "postHandoff", "commit:0", "readArchiveOverview"]);
  assert.ok(io.handoffs[0].startsWith("[Context Window Handoff] w1 -> w2"));
  assert.ok(io.handoffs[0].includes("Reason: phase one done"));

  assert.equal(core.windowId, "w2");
  assert.equal(core.anchorToolCallId, "call-1");
  assert.equal(core.archiveId, "archive_002");
  assert.equal(core.archiveUri, ARCHIVE_URI);
  assert.equal(core.taskId, "task-1");
  assert.equal(core.overviewReady, true);
  assert.equal(core.awaitingFirstObservation, true);
  assert.equal(core.lastResetBy, "agent");
  assert.equal(core.syncedEntryCount, 42);
  assert.equal(core.pendingRequest, "please split the work here");

  assert.ok(out.text.includes("Context window w2 is open"));
  assert.ok(out.text.includes("archive_002"));
  assert.ok(out.text.includes("Working Memory is ready"));
  assert.ok(out.text.includes("not part of the new window"));
  assert.equal(out.details.overviewState, "ready");

  assert.equal(io.persisted.length, 1, "exactly one entry per reset");
  assert.equal(io.persisted[0].type, WINDOW_ENTRY_TYPE);
  assert.equal(io.persisted[0].data.headerText, core.headerText);
  assert.ok(core.headerText.includes('<working-memory archive="archive_002">'));
});

test("requestReset records sibling tool names from the issuing assistant message", async () => {
  const io = makeIo();
  const core = makeCore(io);
  const branch = branchFor("call-1", [{ id: "call-2", name: "read" }]);
  await openWindow(core, io, { branch });
  assert.deepEqual(core.siblingToolNames, ["read"]);
  assert.ok(core.headerText.includes("Tools: read."));
});

test("requestReset refuses when OpenViking is unreachable", async () => {
  const io = makeIo({ connectedValue: false });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.ok, false);
  assert.equal(out.kind, "refused");
  assert.ok(out.text.startsWith("Context window NOT reset:"));
  assert.ok(out.text.includes("unreachable"));
  assert.ok(out.text.includes("keep working"));
  assert.ok(out.text.includes("the harness will compact for you"));
  assert.deepEqual(io.calls, []);
  assert.equal(core.windowIndex, 1);
  assert.equal(core.armed, false);
  assert.equal(io.persisted.length, 0);
});

test("requestReset refuses when there is no OpenViking session", async () => {
  const io = makeIo({ sid: null });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.kind, "refused");
  assert.deepEqual(io.calls, []);
});

test("requestReset refuses when the branch was not fully delivered", async () => {
  const io = makeIo({
    syncBranch: async () => {
      io.calls.push("syncBranch");
      return { added: 2, tokens: 10, allDelivered: false };
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.kind, "refused");
  assert.equal(out.details.stage, "sync");
  assert.deepEqual(io.calls, ["syncBranch"]);
  assert.equal(core.windowIndex, 1);
});

test("requestReset refuses when the flush barrier does not clear", async () => {
  const io = makeIo({
    flush: async (opts) => {
      io.calls.push(`flush:${opts.budgetMs}`);
      return false;
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.kind, "refused");
  assert.equal(out.details.stage, "flush");
  assert.equal(out.details.pending, 2);
  assert.ok(out.text.includes("2 captured message(s) are still queued"));
  assert.deepEqual(io.calls, ["syncBranch", "flush:15000"]);
});

test("requestReset refuses when the handoff note cannot be written", async () => {
  const io = makeIo({
    postHandoff: async () => {
      io.calls.push("postHandoff");
      return false;
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.kind, "refused");
  assert.equal(out.details.stage, "handoff");
  assert.deepEqual(io.calls, ["syncBranch", "flush:15000", "postHandoff"]);
});

test("requestReset refuses when commit fails and reports the trace id", async () => {
  const nullCommit = makeIo({
    commit: async () => {
      nullCommit.calls.push("commit:0");
      return null;
    },
  });
  let core = makeCore(nullCommit);
  let out = await openWindow(core, nullCommit);
  assert.equal(out.kind, "refused");
  assert.equal(out.details.stage, "commit");
  assert.equal(core.windowIndex, 1);

  const failedCommit = makeIo({
    commit: async () => ({ status: "failed", trace_id: "trace-9" }),
  });
  core = makeCore(failedCommit);
  out = await openWindow(core, failedCommit);
  assert.equal(out.kind, "refused");
  assert.ok(out.text.includes("trace trace-9"));
});

test("requestReset is a no-op when OpenViking has nothing to archive", async () => {
  const io = makeIo({
    commit: async () => {
      io.calls.push("commit:0");
      return { status: "skipped", reason: "no_messages" };
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.ok, false);
  assert.equal(out.kind, "noop");
  assert.ok(out.text.includes("nothing to archive"));
  assert.equal(core.windowIndex, 1);
  assert.equal(core.armed, false);
  assert.equal(io.persisted.length, 0);
});

test("requestReset opens a degraded window when the overview never lands", async () => {
  const io = makeIo({ overview: null });
  const core = makeCore(io);
  const out = await openWindow(core, io);

  assert.equal(out.ok, true);
  assert.equal(out.details.overviewState, "pending");
  assert.equal(core.overviewReady, false);
  assert.equal(core.windowId, "w2");
  assert.ok(core.headerText.includes("is not ready yet"));
  assert.ok(!core.headerText.includes("<working-memory"), "first reset has no stale overview to show");
  assert.ok(io.calls.filter((c) => c === "getTask").length >= 1, "the task is polled every 5th attempt");
  assert.ok(io.clock >= 1_700_000_000_000 + 60000, "polling honours the deadline");
});

test("requestReset stops early and marks the window unavailable when the task failed", async () => {
  const io = makeIo({ overview: null, task: { status: "failed" } });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.ok, true);
  assert.equal(out.details.overviewState, "unavailable");
  assert.equal(out.details.attempts, 5);
  assert.ok(core.headerText.includes("is unavailable"));
});

test("requestReset aborted before commit changes nothing", async () => {
  const io = makeIo();
  const core = makeCore(io);
  const out = await openWindow(core, io, { signal: { aborted: true } });
  assert.equal(out.ok, false);
  assert.equal(out.kind, "noop");
  assert.deepEqual(io.calls, []);
  assert.equal(core.windowIndex, 1);
  assert.equal(io.persisted.length, 0);
});

test("requestReset aborted after commit still opens a degraded window", async () => {
  const signal = { aborted: false };
  const io = makeIo({
    overview: null,
    commit: async () => {
      io.calls.push("commit:0");
      signal.aborted = true;
      return { status: "accepted", task_id: "task-1", archive_uri: ARCHIVE_URI };
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io, { signal });
  assert.equal(out.ok, true);
  assert.equal(out.details.overviewState, "pending");
  assert.equal(out.details.attempts, 0, "an aborted wait does not read the overview at all");
  assert.equal(core.windowId, "w2");
  assert.equal(core.overviewReady, false);
});

test("a second concurrent requestReset is a no-op and commits only once", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const io = makeIo({
    commit: async () => {
      io.calls.push("commit:0");
      await gate;
      return { status: "accepted", task_id: "task-1", archive_uri: ARCHIVE_URI };
    },
  });
  const core = makeCore(io);
  const first = openWindow(core, io);
  const second = await openWindow(core, io, { toolCallId: "call-2" });
  assert.equal(second.kind, "noop");
  assert.equal(second.text, "A context window reset is already in progress.");
  release();
  const out = await first;
  assert.equal(out.ok, true);
  assert.equal(io.calls.filter((c) => c === "commit:0").length, 1);
  assert.equal(core.anchorToolCallId, "call-1");
});

test("requestReset never throws when io explodes", async () => {
  const io = makeIo({
    postHandoff: async () => {
      throw new Error("socket hang up");
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io);
  assert.equal(out.ok, false);
  assert.equal(out.details.stage, "error");
  assert.ok(out.text.includes("socket hang up"));
  assert.equal(core.resetting, false);
});

test("restore rebuilds the same anchor and header bytes in a new process", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);

  const entries = [{ type: "custom", customType: WINDOW_ENTRY_TYPE, data: io.persisted[0].data }];
  const restored = makeCore(makeIo());
  restored.restore(entries);

  assert.equal(restored.windowId, "w2");
  assert.equal(restored.anchorToolCallId, "call-1");
  assert.equal(restored.headerText, core.headerText);
  assert.equal(restored.archiveId, "archive_002");
  assert.equal(restored.overviewReady, true);

  const messages = [
    user("old", 1),
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    assistant("carry on"),
  ];
  assert.equal(
    JSON.stringify(restored.transformContext(messages)),
    JSON.stringify(core.transformContext(messages)),
  );
});

// --------------------------------------------------------------------------
// refreshPendingOverview
// --------------------------------------------------------------------------

test("refreshPendingOverview upgrades the header exactly once", async () => {
  const io = makeIo({ overview: null });
  const core = makeCore(io);
  await openWindow(core, io);
  const persistedAfterReset = io.persisted.length;

  assert.equal(await core.refreshPendingOverview(), false);
  io.overview = "# Working Memory\nlate but real";
  assert.equal(await core.refreshPendingOverview(), true);
  assert.equal(core.overviewReady, true);
  assert.ok(core.headerText.includes('<working-memory archive="archive_002"># Working Memory'));
  assert.equal(io.persisted.length, persistedAfterReset + 1);
  assert.ok(io.logs.some((line) => line === "context-window: working memory attached to w2"));

  assert.equal(await core.refreshPendingOverview(), false, "no further reads once ready");
  assert.equal(io.persisted.length, persistedAfterReset + 1);
});

test("refreshPendingOverview gives up after the attempt budget and marks it unavailable once", async () => {
  const io = makeIo({ overview: null });
  const core = makeCore(io, { overviewRefreshMaxAttempts: 2 });
  await openWindow(core, io);
  const persistedAfterReset = io.persisted.length;

  assert.equal(await core.refreshPendingOverview(), false);
  assert.equal(await core.refreshPendingOverview(), false);
  assert.ok(core.headerText.includes("is unavailable"));
  assert.equal(io.persisted.length, persistedAfterReset + 1);

  assert.equal(await core.refreshPendingOverview(), false);
  assert.equal(io.persisted.length, persistedAfterReset + 1, "unavailable is recorded only once");
});

// --------------------------------------------------------------------------
// reminders and status
// --------------------------------------------------------------------------

test("reminders fire once per level and are re-armed by a reset", async () => {
  const io = makeIo();
  const core = makeCore(io);
  const contextWindow = 100000;

  core.lastWindowTokens = 71000;
  assert.equal(core.dueReminder({ usage: { tokens: null }, contextWindow }), "soft");
  assert.equal(core.dueReminder({ usage: { tokens: null }, contextWindow }), null);

  core.lastWindowTokens = 86000;
  assert.equal(core.dueReminder({ usage: { tokens: null }, contextWindow }), "hard");
  assert.equal(core.dueReminder({ usage: { tokens: null }, contextWindow }), null);

  await openWindow(core, io);
  assert.equal(core.awaitingFirstObservation, true);
  assert.equal(core.dueReminder({ usage: { tokens: 99000 }, contextWindow }), null, "suppressed until an assistant answers");

  core.observeAssistantResponse();
  core.lastWindowTokens = 1000;
  assert.equal(core.dueReminder({ usage: { tokens: 90000 }, contextWindow }), "hard", "reported usage wins once observed");
});

test("dueReminder prefers the local estimate and needs a context window", () => {
  const core = makeCore(makeIo());
  core.lastWindowTokens = 90000;
  assert.equal(core.dueReminder({ usage: { tokens: null }, contextWindow: 0 }), null);
  assert.equal(core.dueReminder({ usage: null, contextWindow: 100000 }), "hard");
});

test("statusSnapshot reports usage, idle gap and advice", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  core.observeAssistantResponse();

  const now = io.clock + 47 * 60_000;
  const messages = [
    user("earlier", io.clock - 60 * 60_000),
    assistant("ok"),
    user("latest", io.clock),
  ];
  const snap = core.statusSnapshot({ usage: { tokens: 20000, contextWindow: 100000 }, now, messages });
  assert.equal(snap.windowId, "w2");
  assert.equal(snap.usedTokens, 20000);
  assert.equal(snap.percent, 20);
  assert.equal(snap.tokensLeft, 80000);
  assert.equal(snap.estimated, false);
  assert.equal(snap.sinceLastUserMs, 47 * 60_000);
  assert.equal(snap.idleGapMs, 60 * 60_000);
  assert.equal(snap.archiveId, "archive_002");
  assert.equal(snap.overviewReady, true);
  assert.equal(snap.advice, "no action needed");

  const soft = core.statusSnapshot({ usage: { tokens: 75000, contextWindow: 100000 }, now, messages });
  assert.equal(soft.advice, "past the soft threshold: update your notes and reset at the next stopping point");
  const hard = core.statusSnapshot({ usage: { tokens: 90000, contextWindow: 100000 }, now, messages });
  assert.equal(hard.advice, "save your notes and call new_context now");
});

test("statusSnapshot falls back to the local estimate while awaiting the first observation", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  core.lastWindowTokens = 1234;
  const snap = core.statusSnapshot({ usage: { tokens: 99000, contextWindow: 100000 }, now: io.clock, messages: [] });
  assert.equal(snap.usedTokens, 1234);
  assert.equal(snap.estimated, true);
  assert.equal(snap.sinceLastUserMs, null);
  assert.equal(snap.idleGapMs, null);
});

test("transformContext keeps the turn count and token estimate of the new window", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);

  const messages = [
    user("old", 1),
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    assistant("first answer in the new window"),
    user("more", 2),
    assistant("second answer"),
  ];
  const out = core.transformContext(messages);
  assert.equal(core.turnsInWindow, 2);
  assert.equal(core.awaitingFirstObservation, false);
  assert.ok(core.lastWindowTokens > 0);
  assert.ok(core.lastWindowTokens < estimateTokens(JSON.stringify(messages)) + 10000);
  assert.equal(out[0].role, "user");
});

// --------------------------------------------------------------------------
// pi compaction fallback
// --------------------------------------------------------------------------

function compactionEntries({ withTail = true } = {}) {
  const entries = [
    { id: "e1", type: "message", message: user("start", 1) },
    { id: "e2", type: "message", message: assistantCalls([{ id: "call-1", name: "new_context" }]) },
    { id: "e3", type: "message", message: toolResult("call-1", "new_context") },
  ];
  if (withTail) {
    entries.push({ id: "e4", type: "message", message: user("after the reset", 2) });
    entries.push({ id: "e5", type: "message", message: assistant("sure") });
  }
  return entries;
}

test("handleBeforeCompact keeps preparation.firstKeptEntryId when it is already past the anchor", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  core.lastResetAt = 0;

  const out = await core.handleBeforeCompact({
    preparation: { firstKeptEntryId: "e5", tokensBefore: 123 },
    branchEntries: compactionEntries(),
  });
  assert.equal(out.compaction.firstKeptEntryId, "e5");
  assert.equal(out.compaction.tokensBefore, 123);
  assert.equal(out.compaction.summary, core.headerText);
  assert.equal(core.windowId, "w3");
  assert.equal(core.anchorToolCallId, null, "pi made a native cut, the virtual anchor is dropped");
  assert.equal(core.lastResetBy, "pi-compaction");
});

test("handleBeforeCompact skips forward when preparation would resurrect archived messages", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  core.lastResetAt = 0;

  const out = await core.handleBeforeCompact({
    preparation: { firstKeptEntryId: "e1", tokensBefore: 5 },
    branchEntries: compactionEntries(),
  });
  assert.equal(out.compaction.firstKeptEntryId, "e4");
});

test("handleBeforeCompact uses the sentinel when the anchor batch is the tail", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  core.lastResetAt = 0;

  const out = await core.handleBeforeCompact({
    preparation: { firstKeptEntryId: "e1", tokensBefore: 5 },
    branchEntries: compactionEntries({ withTail: false }),
  });
  assert.equal(out.compaction.firstKeptEntryId, COMPACTION_SENTINEL);
});

test("handleBeforeCompact falls back to preparation when no anchor is armed", async () => {
  const io = makeIo();
  const core = makeCore(io);
  core.headerText = "";
  const out = await core.handleBeforeCompact({
    preparation: { firstKeptEntryId: "e1", tokensBefore: 5 },
    branchEntries: compactionEntries(),
  });
  assert.equal(out.compaction.firstKeptEntryId, "e1");
});

test("handleBeforeCompact does not re-commit right after a reset", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  const header = core.headerText;
  io.calls.length = 0;
  io.clock += 30_000; // inside recentResetGuardMs

  const out = await core.handleBeforeCompact({
    preparation: { firstKeptEntryId: "e1", tokensBefore: 7 },
    branchEntries: compactionEntries(),
  });
  assert.equal(out.compaction.summary, header);
  assert.equal(out.compaction.firstKeptEntryId, "e4");
  assert.deepEqual(io.calls, [], "the guard commits nothing");
  assert.equal(core.windowId, "w2");
  assert.equal(core.anchorToolCallId, "call-1");

  io.clock += 40_000; // past the guard
  io.watermark = 99;
  const later = await core.handleBeforeCompact({
    preparation: { firstKeptEntryId: "e1", tokensBefore: 7 },
    branchEntries: compactionEntries(),
  });
  assert.ok(io.calls.includes("commit:0"), "past the guard the archive is written");
  assert.equal(later.compaction.firstKeptEntryId, "e4");
});

test("handleBeforeCompact returns undefined when a step fails", async () => {
  const flushFails = makeIo({ flush: async () => false });
  let core = makeCore(flushFails);
  assert.equal(
    await core.handleBeforeCompact({ preparation: { firstKeptEntryId: "e1" }, branchEntries: compactionEntries() }),
    undefined,
  );

  const commitFails = makeIo({ commit: async () => null });
  core = makeCore(commitFails);
  assert.equal(
    await core.handleBeforeCompact({ preparation: { firstKeptEntryId: "e1" }, branchEntries: compactionEntries() }),
    undefined,
  );

  const skipped = makeIo({ commit: async () => ({ status: "skipped", reason: "no_messages" }) });
  core = makeCore(skipped);
  assert.equal(
    await core.handleBeforeCompact({ preparation: { firstKeptEntryId: "e1" }, branchEntries: compactionEntries() }),
    undefined,
  );

  const offline = makeIo({ connectedValue: false });
  core = makeCore(offline);
  assert.equal(
    await core.handleBeforeCompact({ preparation: { firstKeptEntryId: "e1" }, branchEntries: compactionEntries() }),
    undefined,
  );
});

test("absorbExternalCompaction drops the anchor and re-arms the reminders", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  core.remindersSent = { soft: true, hard: true };
  const before = io.persisted.length;

  core.absorbExternalCompaction("pi compacted natively");
  assert.equal(core.anchorToolCallId, null);
  assert.equal(core.windowId, "w3");
  assert.deepEqual(core.remindersSent, { soft: false, hard: false });
  assert.equal(io.persisted.length, before + 1);
});

// --------------------------------------------------------------------------
// restore / shutdown
// --------------------------------------------------------------------------

test("restore takes the newest entry and ignores malformed or foreign ones", () => {
  const io = makeIo();
  const core = makeCore(io);
  const entries = [
    { type: "custom", customType: WINDOW_ENTRY_TYPE, data: { windowIndex: 5, anchorToolCallId: "old", headerText: "OLD" } },
    { customType: WINDOW_ENTRY_TYPE, data: { windowIndex: 7, anchorToolCallId: "newer", headerText: "NEWER" } },
    { type: WINDOW_ENTRY_TYPE, data: { windowIndex: 9, anchorToolCallId: "newest", headerText: "NEWEST" } },
    { type: "custom", customType: WINDOW_ENTRY_TYPE, data: "not an object" },
    { type: "custom", customType: WINDOW_ENTRY_TYPE, data: null },
    { type: "custom", customType: WINDOW_ENTRY_TYPE, data: { ovSessionId: "pi-other", windowIndex: 42 } },
    { type: "custom", customType: "something-else", data: { windowIndex: 99 } },
  ];
  core.restore(entries);
  assert.equal(core.windowIndex, 9);
  assert.equal(core.anchorToolCallId, "newest");
  assert.equal(core.headerText, "NEWEST");
  assert.equal(core.armed, true);
});

test("a fresh session restores to w1 and stays unarmed", () => {
  const core = makeCore(makeIo());
  const state = core.restore([]);
  assert.equal(state.windowIndex, 1);
  assert.equal(core.windowId, "w1");
  assert.equal(core.armed, false);
  assert.equal(core.headerText, "");
  assert.equal(core.transformContext([user("hi", 1)]).length, 1);
});

test("persist dedupes and shutdown refreshes the watermark", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  assert.equal(io.persisted.length, 1);

  core.persist();
  assert.equal(io.persisted.length, 1, "an unchanged state is not written twice");

  io.watermark = 137;
  await core.shutdown();
  assert.equal(io.persisted.length, 2);
  assert.equal(io.persisted[1].data.syncedEntryCount, 137);
});

// --------------------------------------------------------------------------
// adversarial / verification pass
// --------------------------------------------------------------------------

test("an abort-aware io.sleep that rejects still opens a degraded window", async () => {
  const io = makeIo({
    overview: null,
    sleep: async () => {
      throw new Error("aborted while waiting");
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io);

  assert.equal(out.ok, true, "the archive exists, so the window must open");
  assert.equal(out.details.overviewState, "pending");
  assert.equal(core.windowId, "w2");
  assert.ok(!out.text.includes("NOT reset"));
});

test("a post-commit failure still opens the window instead of claiming nothing changed", async () => {
  // The commit emptied the OpenViking session (keep_recent_count 0), so a refusal
  // here would leave the model carrying the old context while believing its
  // history is still live and re-readable.
  const io = makeIo({ overview: null });
  const realNow = io.now;
  io.commit = async () => {
    io.calls.push("commit:0");
    io.now = () => {
      throw new Error("clock exploded");
    };
    return { status: "accepted", task_id: "task-1", archive_uri: ARCHIVE_URI };
  };
  const core = makeCore(io);
  core.io.now = () => io.now();
  const out = await openWindow(core, io);

  assert.equal(out.ok, true, "the archive exists, so the window must open");
  assert.equal(out.kind, "reset");
  assert.equal(out.details.overviewState, "pending");
  assert.equal(out.details.archiveId, "archive_002");
  assert.ok(out.details.error.includes("clock exploded"));
  assert.ok(!out.text.includes("NOT reset"));
  assert.equal(core.windowId, "w2");
  assert.equal(core.anchorToolCallId, "call-1");
  assert.equal(core.overviewReady, false);
  assert.ok(core.headerText.startsWith(WINDOW_HEADER_OPEN));
  assert.ok(core.headerText.includes("is not ready yet"));
  assert.equal(io.persisted.length, 1, "the window is persisted even on the recovery path");
  assert.equal(core.resetting, false);
  io.now = realNow;
});

test("a steering user message that lands after the batch stays a real user turn", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);

  const steer = user("actually, also check the parser", 4242);
  const out = core.transformContext([
    user("old", 1),
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    steer,
  ]);

  assert.equal(out.length, 1, "the header is merged into the steering message");
  assert.equal(out[0].role, "user");
  assert.ok(out[0].content.startsWith(WINDOW_HEADER_OPEN));
  assert.ok(out[0].content.includes("actually, also check the parser"));
  assert.deepEqual(
    lastUserTimestamps(out),
    [4242],
    "a merged real user message must not be mistaken for something we injected",
  );
  assert.equal(core.statusSnapshot({ now: 4242 + 60000, messages: out }).sinceLastUserMs, 60000);
});

test("a stale pressure reminder is dropped by the cut instead of landing in the new window", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);

  const reminder = {
    role: "user",
    customType: REMINDER_CUSTOM_TYPE,
    content: `${reminderText("hard", { windowId: "w1", usedTokens: 90000, contextWindow: 100000 })}`,
    timestamp: 7,
  };
  const out = core.transformContext([
    user("old", 1),
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    reminder,
  ]);

  assert.equal(out.length, 1);
  assert.ok(!JSON.stringify(out).includes("[context-reminder]"), "w1's reminder must not follow the header into w2");
  assert.ok(String(out[0].content).startsWith(WINDOW_HEADER_OPEN));
});

test("a reminder in front of a real steering message is dropped but the message survives", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);

  const out = core.transformContext([
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    { role: "user", content: "[context-reminder] window w1 is about 90% full", timestamp: 6 },
    user("keep going with the parser", 7),
  ]);

  assert.equal(out.length, 1);
  assert.ok(String(out[0].content).includes("keep going with the parser"));
  assert.ok(!String(out[0].content).includes("[context-reminder]"));
});

test("the window token estimate counts tool call arguments and thinking blocks", () => {
  const core = makeCore(makeIo());
  const payload = "x".repeat(40000);
  core.recordWindowMetrics(
    [
      user("hi", 1),
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "y".repeat(4000) },
          { type: "toolCall", id: "t1", name: "write", arguments: { path: "a.txt", content: payload } },
        ],
      },
      { role: "toolResult", toolCallId: "t1", toolName: "write", content: [{ type: "text", text: "ok" }] },
    ],
    -1,
  );
  assert.ok(
    core.lastWindowTokens > 9000,
    `a 40k-char tool argument must not read as a handful of tokens (got ${core.lastWindowTokens})`,
  );
});

test("thresholds are clamped under pi's own auto-compaction line", () => {
  // pi compacts as soon as contextTokens > contextWindow - reserveTokens.
  assert.deepEqual(effectiveThresholds({ softPercent: 70, hardPercent: 85, contextWindow: 262144, reserveTokens: 16384 }), {
    softPercent: 70,
    hardPercent: 85,
  });
  const small = effectiveThresholds({ softPercent: 70, hardPercent: 85, contextWindow: 32768, reserveTokens: 16384 });
  assert.equal(small.hardPercent, 49, "50% is pi's line on a 32k window, so the hard reminder sits below it");
  assert.equal(small.softPercent, 49);
  assert.deepEqual(
    effectiveThresholds({ softPercent: 70, hardPercent: 85, contextWindow: 32768, reserveTokens: 0 }),
    { softPercent: 70, hardPercent: 85 },
    "an unknown reserve leaves the configured thresholds alone",
  );

  const core = makeCore(makeIo());
  core.lastWindowTokens = 17000;
  assert.equal(core.dueReminder({ usage: { tokens: null }, contextWindow: 32768 }), null, "85% is never reached first");
  assert.equal(
    core.dueReminder({ usage: { tokens: null }, contextWindow: 32768, reserveTokens: 16384 }),
    "hard",
    "with the reserve known the hard reminder fires before pi compacts",
  );
  const snap = core.statusSnapshot({
    usage: { tokens: 17000, contextWindow: 32768 },
    reserveTokens: 16384,
    now: 1,
  });
  assert.equal(snap.hardPercent, 49);
  assert.equal(snap.advice, "save your notes and call new_context now");
});

test("after an external compaction the guard does not reuse the stale window header", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  const staleHeader = core.headerText;

  io.clock += 5000;
  core.absorbExternalCompaction("pi compacted without us");
  io.clock += 1000;

  const out = await core.handleBeforeCompact({
    preparation: { firstKeptEntryId: "e9", tokensBefore: 12 },
    branchEntries: [],
  });
  assert.notEqual(out?.compaction?.summary, staleHeader, "that header claims an archive that no longer covers the branch");
  assert.equal(out?.compaction?.details.reason, "pi-compaction", "a real archive is taken instead");
});

test("two resets in one session archive twice and leave only the newest window visible", async () => {
  let archive = 2;
  const io = makeIo({
    commit: async () => {
      io.calls.push("commit:0");
      return {
        status: "accepted",
        task_id: `task-${archive}`,
        archive_uri: `viking://user/u1/sessions/pi-abc/history/archive_00${archive++}`,
      };
    },
  });
  const core = makeCore(io);

  await openWindow(core, io, { reason: "phase one", notes: "note one", toolCallId: "call-1" });
  const firstHeader = core.headerText;
  assert.equal(core.windowId, "w2");

  io.clock += 120000;
  io.overview = "# Working Memory\n## Current State\nphase two";
  const branch2 = [
    { id: "e1", type: "message", message: user("first request", 10) },
    { id: "e2", type: "message", message: assistantCalls([{ id: "call-1", name: "new_context" }]) },
    { id: "e3", type: "message", message: toolResult("call-1", "new_context") },
    { id: "e4", type: "message", message: user("second request", 60) },
    { id: "e5", type: "message", message: assistantCalls([{ id: "call-2", name: "new_context" }]) },
  ];
  const second = await openWindow(core, io, {
    reason: "phase two",
    notes: "note two",
    toolCallId: "call-2",
    branch: branch2,
  });

  assert.equal(second.ok, true);
  assert.equal(core.windowId, "w3");
  assert.equal(core.archiveId, "archive_003");
  assert.equal(core.pendingRequest, "second request");
  assert.equal(io.calls.filter((c) => c === "commit:0").length, 2);
  assert.ok(core.headerText.includes('id="w3"'));
  assert.ok(core.headerText.includes('previous="w2"'));
  assert.ok(core.headerText.includes("note two"));
  assert.ok(!core.headerText.includes("note one"));
  assert.notEqual(core.headerText, firstHeader);
  assert.equal(io.persisted.length, 2);

  const out = core.transformContext([
    user("first request", 10),
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    user("second request", 60),
    assistantCalls([{ id: "call-2", name: "new_context" }]),
    toolResult("call-2", "new_context"),
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].role, "user");
  assert.ok(!JSON.stringify(out).includes("first request"), "w1 is gone");
  assert.equal(
    out[0].content.split("second request").length - 1,
    1,
    "w2's transcript is gone; the text survives only inside <pending-request>",
  );
  assert.ok(out[0].content.includes("<pending-request>second request</pending-request>"));
  assert.equal(
    JSON.stringify(core.transformContext([
      user("first request", 10),
      assistantCalls([{ id: "call-1", name: "new_context" }]),
      toolResult("call-1", "new_context"),
      user("second request", 60),
      assistantCalls([{ id: "call-2", name: "new_context" }]),
      toolResult("call-2", "new_context"),
    ])),
    JSON.stringify(out),
    "the second window header is byte-stable too",
  );
});

test("a second reset carries the previous Working Memory as a stale block while the new one is pending", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io, { toolCallId: "call-1" });
  assert.equal(core.overviewReady, true);

  io.overview = null;
  const out = await openWindow(core, io, { toolCallId: "call-2", branch: branchFor("call-2") });
  assert.equal(out.details.overviewState, "pending");
  assert.ok(core.headerText.includes('<working-memory status="pending"'));
  assert.ok(core.headerText.includes('stale="true"'));
  assert.ok(core.headerText.includes("Current State"));
});

test("restore ignores a window entry written by a different OpenViking session", async () => {
  const io = makeIo();
  const core = makeCore(io);
  await openWindow(core, io);
  const data = io.persisted[0].data;

  const foreign = makeCore(makeIo({ sid: "pi-somewhere-else" }));
  foreign.restore([{ type: "custom", customType: WINDOW_ENTRY_TYPE, data }]);
  assert.equal(foreign.windowIndex, 1, "a foreign window must not be adopted");
  assert.equal(foreign.armed, false);
  assert.equal(foreign.headerText, "");

  const same = makeCore(makeIo());
  same.restore([{ type: "custom", customType: WINDOW_ENTRY_TYPE, data }]);
  assert.equal(same.windowIndex, 2);
  assert.equal(same.headerText, data.headerText);
});

test("a restored pending window rebuilds byte-identically to the in-process rebuild", async () => {
  const io = makeIo({ overview: null });
  const core = makeCore(io);
  await openWindow(core, io);
  assert.equal(core.overviewReady, false);

  const data = io.persisted[0].data;
  const io2 = makeIo({ overview: "# Working Memory\n## Current State\nlanded late" });
  const restored = makeCore(io2);
  restored.restore([{ type: "custom", customType: WINDOW_ENTRY_TYPE, data }]);
  assert.equal(restored.headerText, data.headerText, "restore reproduces the frozen bytes");

  io.overview = "# Working Memory\n## Current State\nlanded late";
  assert.equal(await core.refreshPendingOverview(), true);
  assert.equal(await restored.refreshPendingOverview(), true);
  assert.equal(restored.headerText, core.headerText, "the upgraded header is identical across processes");
  assert.ok(restored.headerText.includes('previous="w1"'));
});

test("a leading compaction summary is dropped and the cut still starts with the header", () => {
  const cut = applyWindowCut(
    [
      user("<compaction-summary>everything before this was compacted</compaction-summary>", 1),
      user("do the thing", 2),
      assistantCalls([{ id: "call-1", name: "new_context" }]),
      toolResult("call-1", "new_context"),
      assistant("done"),
    ],
    { anchorToolCallId: "call-1", headerText: "HDR", headerTimestamp: 99 },
  );
  assert.equal(cut.applied, true);
  assert.equal(cut.messages.length, 2);
  assert.equal(cut.messages[0].role, "user");
  assert.equal(cut.messages[0].content, "HDR");
  assert.equal(cut.messages[0].timestamp, 99);
  assert.ok(!JSON.stringify(cut.messages).includes("compaction-summary"));
});

test("the cut never leaves an orphan tool result behind when more batches follow", () => {
  const messages = [
    user("go", 1),
    assistantCalls([
      { id: "call-1", name: "new_context" },
      { id: "call-2", name: "read" },
    ]),
    toolResult("call-2", "read"),
    toolResult("call-1", "new_context"),
    assistantCalls([{ id: "call-3", name: "grep" }]),
    toolResult("call-3", "grep"),
  ];
  const cut = applyWindowCut(messages, { anchorToolCallId: "call-1", headerText: "HDR" });
  const roles = cut.messages.map((m) => m.role);
  assert.deepEqual(roles, ["user", "assistant", "toolResult"]);

  const ids = new Set(
    cut.messages
      .filter((m) => m.role === "assistant")
      .flatMap((m) => (Array.isArray(m.content) ? m.content : []))
      .filter((b) => b.type === "toolCall")
      .map((b) => b.id),
  );
  for (const msg of cut.messages) {
    if (msg.role === "toolResult") {
      assert.ok(ids.has(msg.toolCallId), `orphan tool result ${msg.toolCallId} survived the cut`);
    }
  }
});

test("the reset still opens a degraded window when a slow flush eats the deadline", async () => {
  const io = makeIo({
    overview: null,
    flush: async (opts = {}) => {
      io.calls.push(`flush:${opts.budgetMs}`);
      io.clock += 59_000;
      return true;
    },
  });
  const core = makeCore(io);
  const out = await openWindow(core, io);

  assert.equal(out.ok, true);
  assert.equal(out.details.overviewState, "pending");
  assert.ok(out.details.attempts <= 2, "the poll respects the deadline that the flush already spent");
  assert.equal(core.windowId, "w2");
  assert.ok(core.headerText.includes("is not ready yet"));
});

test("an image-only user message is not mistaken for something this extension injected", () => {
  const stamps = lastUserTimestamps([
    { role: "user", content: [{ type: "image", data: "AAAA", mimeType: "image/png" }], timestamp: 500 },
  ]);
  assert.deepEqual(stamps, [500]);
});

test("Chinese notes, pending request and overview survive the merge into a kept user message", async () => {
  const io = makeIo({ overview: "工作记忆".repeat(400) });
  const core = makeCore(io, { notesBudget: 100, pendingRequestBudget: 30, overviewBudget: 120 });
  await openWindow(core, io, {
    notes: "目标：把解析器切到流式实现；下一步：补齐测试".repeat(40),
    branch: [{ id: "e1", message: user("把解析器改成流式的，注意兼容旧配置".repeat(10), 3) }],
  });

  assert.ok(core.headerText.includes("...(truncated)"));
  assert.ok(estimateTokens(core.notes) > 100, "the raw notes are kept whole in state");

  const out = core.transformContext([
    assistantCalls([{ id: "call-1", name: "new_context" }]),
    toolResult("call-1", "new_context"),
    user("继续", 9),
  ]);
  assert.equal(out.length, 1);
  assert.ok(out[0].content.startsWith(WINDOW_HEADER_OPEN));
  assert.ok(out[0].content.endsWith("继续"));
  assert.deepEqual(lastUserTimestamps(out), [9]);
});
