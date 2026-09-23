import test from "node:test";
import assert from "node:assert/strict";
import {
  OVERVIEW_MARKER,
  TAKEOVER_ENTRY_TYPE,
  TakeoverCore,
  buildOverviewMessage,
  commitOutcome,
  countUndeliveredForSession,
  countUserTurns,
  estimatePayloadTokens,
  estimateTokens,
  findBoundaryIndex,
  fingerprintMessage,
  flattenContent,
  isUserTurnStart,
  deriveHistoryUri,
  truncateToTokens,
} from "../lib/takeover-core.mjs";

function msg(role, content, timestamp = 0) {
  return { role, content, timestamp };
}

function user(text, timestamp = 0) {
  return msg("user", text, timestamp);
}

function assistant(text, timestamp = 0) {
  return msg("assistant", text, timestamp);
}

function system(text, extra = {}) {
  return { role: "system", content: text, timestamp: 0, ...extra };
}

function makeCore(overrides = {}) {
  const calls = {
    synced: 0,
    flushed: 0,
    committed: 0,
    persisted: [],
    slept: [],
    logs: [],
    overviewUris: [],
    tools: overrides.tools ?? [],
  };
  let watermark = overrides.watermark ?? 0;
  let dropped = overrides.dropped ?? 0;
  const io = {
    syncBranch: async (branch) => {
      calls.synced++;
      calls.lastSyncBranch = branch;
      return overrides.syncResult ?? { added: 0, tokens: 0, allDelivered: true };
    },
    flush: async () => {
      calls.flushed++;
      return overrides.flushResult ?? true;
    },
    commit: async (opts) => {
      calls.committed++;
      calls.lastCommitOpts = opts;
      // Default: a healthy accepted commit whose archive_uri lives under
      // A concrete archive URI lets the poller bind the summary to this commit.
      return overrides.commitResult === undefined
        ? {
            status: "accepted",
            archived: true,
            task_id: "t-1",
            archive_uri: "viking://user/x/sessions/s/history/archive_001",
          }
        : overrides.commitResult;
    },
    readArchiveOverview: async (uri) => {
      calls.overviewUris.push(uri);
      const values = overrides.overviews ?? ["overview ready"];
      const value = values[Math.min(calls.overviewCalls || 0, values.length - 1)];
      calls.overviewCalls = (calls.overviewCalls || 0) + 1;
      return value;
    },
    // The tests' branches are plain message arrays, so the retained tail's
    // capture count is just its length.
    captureCount: (slice) => (Array.isArray(slice) ? slice.length : 0),
    persistEntry: (type, data) => calls.persisted.push({ type, data }),
    getWatermark: () => watermark,
    droppedCount: () => dropped,
    availableTools: () => calls.tools,
    sleep: async (ms) => calls.slept.push(ms),
    log: (message) => calls.logs.push(message),
  };
  const core = new TakeoverCore({
    config: {
      takeoverEnabled: true,
      takeoverTokenThreshold: 100,
      takeoverKeepRecentTurns: 1,
      takeoverOverviewBudget: 1000,
      takeoverOverviewPollMs: 1,
      takeoverOverviewPollMax: 3,
      ...overrides.config,
    },
    io: { ...io, ...overrides.io },
  });
  return {
    core,
    calls,
    setWatermark: (n) => { watermark = n; },
    setDropped: (n) => { dropped = n; },
  };
}

test("flattenContent handles strings and text arrays", () => {
  assert.equal(flattenContent(user("hello")), "hello");
  assert.equal(flattenContent(user([{ type: "text", text: "a" }, { type: "image" }, { type: "text", text: "b" }])), "ab");
  assert.equal(flattenContent(user(null)), "");
});

test("fingerprintMessage includes role length and 200-char prefix", () => {
  const fp = fingerprintMessage(user("x".repeat(250)));
  assert.equal(fp, `user:250:${"x".repeat(200)}`);
  assert.notEqual(fingerprintMessage(user("same")), fingerprintMessage(assistant("same")));
});

test("user turn helpers ignore injected overview messages", () => {
  const messages = [
    user("first"),
    assistant("answer"),
    user(`${OVERVIEW_MARKER} archived`),
    user("second"),
  ];
  assert.equal(isUserTurnStart(messages[0]), true);
  assert.equal(isUserTurnStart(messages[2]), false);
  assert.equal(countUserTurns(messages), 2);
  assert.equal(findBoundaryIndex(messages, 0), 0);
  assert.equal(findBoundaryIndex(messages, 1), 3);
  assert.equal(findBoundaryIndex(messages, 2), -1);
});

test("estimateTokens and truncateToTokens handle CJK conservatively", () => {
  assert.equal(estimateTokens(""), 0);
  assert.equal(estimateTokens("a".repeat(100)), 25);
  assert.equal(estimateTokens("界".repeat(10)), 15);
  assert.equal(estimateTokens("界界" + "a".repeat(8)), 5);
  assert.equal(truncateToTokens("hello", 100), "hello");
  assert.equal(truncateToTokens("hello", 0), "");
  const truncated = truncateToTokens("界".repeat(9000), 3000);
  assert.ok(estimateTokens(truncated) <= 3000);
  assert.ok(truncated.length < 2500);
});

test("estimatePayloadTokens counts content and structured parts", () => {
  assert.equal(estimatePayloadTokens({ content: "a".repeat(40) }), 10);
  const withParts = estimatePayloadTokens({
    parts: [
      { type: "text", text: "a".repeat(40) },
      { type: "tool", tool_name: "read", tool_input: { path: "file" }, tool_status: "running" },
    ],
  });
  assert.ok(withParts > 10);
});

test("buildOverviewMessage is byte-stable for the same inputs", () => {
  const a = buildOverviewMessage("summary", 42, 1000);
  const b = buildOverviewMessage("summary", 42, 1000);
  assert.deepEqual(a, b);
  assert.equal(a.timestamp, 41);
  assert.match(a.content, /\[OpenViking Session Context\]/);
});

test("buildOverviewMessage does not promise an unavailable tool", () => {
  const message = buildOverviewMessage("summary", 1, 1000);
  assert.doesNotMatch(message.content, /openviking_search/);
  assert.doesNotMatch(message.content, /viking_archive_expand/);
});

test("countUndeliveredForSession only counts addMessage for the same session", () => {
  const pending = [
    { entry: { type: "addMessage", sessionId: "a" } },
    { entry: { type: "commitSession", sessionId: "a" } },
    { entry: { type: "addMessage", sessionId: "b" } },
    { type: "addMessage", sessionId: "a" },
  ];
  assert.equal(countUndeliveredForSession(pending, "a"), 2);
});

test("commitOutcome requires this commit's archive URI", () => {
  assert.deepEqual(commitOutcome(null), { accepted: false, reason: "no_result" });
  assert.deepEqual(
    commitOutcome({ status: "skipped", archived: false, archive_uri: null, reason: "no_messages" }),
    { accepted: false, reason: "skipped:no_messages" },
  );
  assert.deepEqual(commitOutcome({ status: "accepted", archived: false, archive_uri: "viking://x" }), {
    accepted: false, reason: "not_archived",
  });
  assert.deepEqual(commitOutcome({ status: "accepted", archived: true }), {
    accepted: false, reason: "no_archive_uri",
  });
  assert.deepEqual(commitOutcome({ status: "failed", archived: true, archive_uri: "viking://bad" }), {
    accepted: false, reason: "status:failed",
  });
  assert.deepEqual(commitOutcome({ archive_uri: "viking://user/u/sessions/s/history/archive_002", task_id: "t" }), {
    accepted: true, reason: "accepted", archiveUri: "viking://user/u/sessions/s/history/archive_002", taskId: "t",
  });
  assert.equal(deriveHistoryUri("viking://user/u/sessions/s/history/archive_002"), "viking://user/u/sessions/s/history");
  assert.equal(deriveHistoryUri("viking://unexpected/archive"), "");
});

test("transformContext drops covered turns and injects overview before recall", () => {
  const { core } = makeCore();
  core.restore([
    {
      type: "custom",
      customType: TAKEOVER_ENTRY_TYPE,
      data: {
        coveredUserTurns: 1,
        overview: "archived first turn",
        pendingTokens: 0,
      },
    },
  ]);
  const messages = [
    user("first", 10),
    assistant("answer", 11),
    user("second", 20),
    assistant("answer 2", 21),
  ];
  const out = core.transformContext(messages);
  assert.equal(out.length, 3);
  assert.equal(out[0].role, "user");
  assert.match(out[0].content, /archived first turn/);
  assert.equal(out[0].timestamp, 19);
  assert.equal(out[1].content, "second");
});

test("transformContext is stable between commits", () => {
  const { core } = makeCore();
  core.restore([
    {
      type: "custom",
      customType: TAKEOVER_ENTRY_TYPE,
      data: { coveredUserTurns: 1, overview: "same overview", pendingTokens: 0 },
    },
  ]);
  const messages = [user("first", 1), assistant("answer", 2), user("second", 3)];
  const a = JSON.stringify(core.transformContext(messages));
  const b = JSON.stringify(core.transformContext(messages));
  assert.equal(a, b);
});

test("transformContext resets boundary on fingerprint mismatch", () => {
  const { core } = makeCore();
  core.restore([
    {
      type: "custom",
      customType: TAKEOVER_ENTRY_TYPE,
      data: {
        coveredUserTurns: 1,
        overview: "overview",
        fingerprint: fingerprintMessage(assistant("old answer")),
        pendingTokens: 0,
      },
    },
  ]);
  const messages = [user("first"), assistant("new answer"), user("second")];
  const out = core.transformContext(messages);
  assert.equal(out, messages);
  assert.equal(core.state.coveredUserTurns, 0);
});

test("transformContext keeps covered-region system messages in original order", () => {
  const { core } = makeCore();
  core.restore([
    {
      type: "custom",
      customType: TAKEOVER_ENTRY_TYPE,
      data: { coveredUserTurns: 1, overview: "archived first turn", pendingTokens: 0 },
    },
  ]);
  const baseSystem = system("BASE PROMPT", { toolsAdded: [{ name: "read" }] });
  const midSystem = system("added a tool", { toolsAdded: [{ name: "openviking_search" }] });
  const messages = [
    baseSystem,
    user("first", 10),
    assistant("answer", 11),
    midSystem,
    user("second", 20),
    assistant("answer 2", 21),
  ];
  const out = core.transformContext(messages);
  // Both covered system messages survive, in order, ahead of the overview.
  assert.equal(out[0], baseSystem);
  assert.equal(out[1], midSystem);
  assert.equal(out[2].role, "user");
  assert.match(out[2].content, /archived first turn/);
  assert.equal(out[3].content, "second");
  // Nothing is dropped from the retained tail, and no system is duplicated.
  assert.equal(out.filter((m) => m.role === "system").length, 2);
});

test("transformContext leaves a system message inside the retained tail in place", () => {
  const { core } = makeCore();
  core.restore([
    {
      type: "custom",
      customType: TAKEOVER_ENTRY_TYPE,
      data: { coveredUserTurns: 1, overview: "archived", pendingTokens: 0 },
    },
  ]);
  const keptSystem = system("tool removed mid-tail", { toolsRemoved: [{ name: "read" }] });
  const messages = [
    system("BASE"),
    user("first", 10),
    assistant("answer", 11),
    user("second", 20),
    keptSystem,
    assistant("answer 2", 21),
  ];
  const out = core.transformContext(messages);
  // The covered BASE is hoisted before the overview; the tail system stays put.
  assert.equal(out[0].content, "BASE");
  assert.match(out[1].content, /archived/);
  assert.equal(out[2].content, "second");
  assert.equal(out[3], keptSystem);
});

test("transformContext with no system messages is unchanged from before (0.80.3)", () => {
  const { core } = makeCore();
  core.restore([
    {
      type: "custom",
      customType: TAKEOVER_ENTRY_TYPE,
      data: { coveredUserTurns: 1, overview: "archived first turn", pendingTokens: 0 },
    },
  ]);
  const messages = [
    user("first", 10),
    assistant("answer", 11),
    user("second", 20),
    assistant("answer 2", 21),
  ];
  const out = core.transformContext(messages);
  assert.equal(out.length, 3);
  assert.equal(out[0].role, "user");
  assert.match(out[0].content, /archived first turn/);
  assert.equal(out[1].content, "second");
});

test("restore uses the last ov-takeover entry and restores syncedEntryCount", () => {
  const { core } = makeCore();
  core.restore([
    { type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: { coveredUserTurns: 1, overview: "old", pendingTokens: 3, syncedEntryCount: 10 } },
    { type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: { coveredUserTurns: 2, overview: "new", pendingTokens: 7, syncedEntryCount: 12 } },
  ]);
  assert.equal(core.state.coveredUserTurns, 2);
  assert.equal(core.state.overview, "new");
  assert.equal(core.state.pendingTokens, 7);
  assert.equal(core.state.syncedEntryCount, 12);
});

test("onTurnSynced waits for threshold and enough user turns", async () => {
  const { core, calls } = makeCore({ config: { takeoverTokenThreshold: 50, takeoverKeepRecentTurns: 3 } });
  const short = [user("one"), user("two")];
  core.transformContext(short);
  assert.equal(await core.onTurnSynced(60, short), false);
  assert.equal(calls.committed, 0);

  const enough = [user("one"), user("two"), user("three"), user("four")];
  core.transformContext(enough);
  assert.equal(await core.onTurnSynced(0, enough), true);
  assert.equal(calls.committed, 1);
});

test("commitAndAdvance advances boundary and persists after overview is ready", async () => {
  const { core, calls, setWatermark } = makeCore({
    watermark: 4,
    overviews: ["", "fresh overview"],
  });
  const branch = [user("one"), assistant("a"), user("two"), assistant("b"), user("three")];
  core.transformContext(branch);
  setWatermark(5);
  assert.equal(await core.onTurnSynced(120, branch), true);
  assert.equal(core.state.coveredUserTurns, 2);
  assert.equal(core.state.overview, "fresh overview");
  assert.equal(core.state.fingerprint, fingerprintMessage(assistant("b")));
  assert.equal(core.state.pendingTokens, 0);
  assert.equal(core.state.syncedEntryCount, 5);
  assert.equal(calls.flushed, 1);
  assert.equal(calls.committed, 1);
  assert.equal(calls.lastCommitOpts.queueOnFailure, false);
  assert.deepEqual(calls.slept, [1]);
  assert.equal(calls.persisted.length, 1);
  assert.equal(calls.persisted[0].type, TAKEOVER_ENTRY_TYPE);
  assert.deepEqual(calls.overviewUris, [
    "viking://user/x/sessions/s/history/archive_001",
    "viking://user/x/sessions/s/history/archive_001",
  ]);
});

test("keep_recent_count counts captured messages instead of transcript entries", async () => {
  const { core, calls } = makeCore({
    config: { takeoverKeepRecentTurns: 1 },
    io: { captureCount: (slice) => slice.filter((entry) => entry.role !== "system" && entry.type !== "custom").length },
  });
  const branch = [
    system("base"), user("one"), assistant("one answer"),
    { type: "custom", data: {} }, user("two"), assistant("tool one"), assistant("tool two"),
  ];
  assert.equal(await core.onTurnSynced(120, branch), true);
  assert.equal(calls.lastCommitOpts.keepRecentCount, 3);
});

test("keepRecentTurns zero archives every captured message", async () => {
  const { core, calls } = makeCore({ config: { takeoverKeepRecentTurns: 0 } });
  const branch = [user("one"), assistant("answer")];
  assert.equal(await core.onTurnSynced(120, branch), true);
  assert.equal(calls.lastCommitOpts.keepRecentCount, 0);
  assert.equal(core.state.coveredUserTurns, 1);
});

test("commitAndAdvance keeps pending tokens when flush fails", async () => {
  const { core, calls } = makeCore({ flushResult: false });
  const branch = [user("one"), user("two")];
  core.transformContext(branch);
  assert.equal(await core.onTurnSynced(120, branch), false);
  assert.equal(core.state.pendingTokens, 120);
  assert.equal(calls.committed, 0);
  assert.equal(calls.persisted.length, 1);
  assert.equal(calls.persisted[0].data.coveredUserTurns, 0);
  assert.equal(calls.persisted[0].data.pendingTokens, 120);
});

test("commitAndAdvance persists the same pending archive until its overview is ready", async () => {
  const { core, calls } = makeCore({ overviews: ["", "", ""] });
  const branch = [user("one"), user("two")];
  core.transformContext(branch);
  assert.equal(await core.onTurnSynced(120, branch), false);
  assert.equal(core.state.coveredUserTurns, 0);
  assert.equal(core.state.pendingTokens, 120);
  assert.equal(core.state.pendingArchive.archiveUri, "viking://user/x/sessions/s/history/archive_001");
  assert.equal(calls.committed, 1);
  assert.equal(calls.persisted.length, 1);
  assert.equal(await core.onTurnSynced(10, branch), false);
  assert.equal(calls.committed, 1);
  assert.equal(core.state.pendingTokens, 130);
});

test("pending archive survives restore and later advances without another commit", async () => {
  const first = makeCore({ overviews: ["", "", ""] });
  const branch = [user("one"), assistant("a"), user("two")];
  assert.equal(await first.core.onTurnSynced(120, branch), false);
  const saved = first.core.persistedState();

  const resumed = makeCore({ overviews: ["restored overview"] });
  resumed.core.restore([{ type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: saved }]);
  assert.equal(await resumed.core.onTurnSynced(7, branch), true);
  assert.equal(resumed.calls.committed, 0);
  assert.equal(resumed.core.state.pendingArchive, null);
  assert.equal(resumed.core.state.pendingTokens, 7);
});

test("old state without archive fields remains compatible", () => {
  const { core } = makeCore();
  core.restore([{ type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: {
    coveredUserTurns: 1, overview: "legacy", pendingTokens: 4, syncedEntryCount: 2,
  } }]);
  assert.equal(core.state.pendingArchive, null);
  assert.equal(core.state.captureGap, false);
  assert.equal(core.state.archiveUri, "");
  assert.equal(core.state.historyUri, "");
  const out = core.transformContext([user("one"), assistant("a"), user("two")]);
  assert.doesNotMatch(out[0].content, /openviking_list/);
});

test("capture gap survives restore and blocks takeover plus native compaction", async () => {
  const { core, calls } = makeCore();
  const branch = [user("one"), user("two")];
  core.restore([{ type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: {
    coveredUserTurns: 0, overview: "", pendingTokens: 120, captureGap: true,
  } }]);
  assert.equal(await core.commitAndAdvance(branch), false);
  assert.equal(await core.handleBeforeCompact({ firstKeptEntryId: "pi-kept" }, branch), undefined);
  assert.equal(calls.committed, 0);
});

test("skipped, not archived and missing URI results never advance", async () => {
  const branch = [user("one"), user("two")];
  for (const commitResult of [
    { status: "skipped", archived: false, archive_uri: null, reason: "no_messages" },
    { status: "accepted", archived: false, archive_uri: "viking://bad" },
    { status: "accepted", archived: true },
  ]) {
    const { core, calls } = makeCore({ commitResult });
    assert.equal(await core.onTurnSynced(120, branch), false);
    assert.equal(core.state.coveredUserTurns, 0);
    assert.equal(calls.overviewUris.length, 0);
  }
});

test("archive overview read failures are logged and keep the boundary pending", async () => {
  const { core, calls } = makeCore({
    io: { readArchiveOverview: async () => { throw new Error("storage failed"); } },
  });
  const branch = [user("one"), user("two")];
  assert.equal(await core.onTurnSynced(120, branch), false);
  assert.ok(core.state.pendingArchive);
  assert.ok(calls.logs.some((line) => /overview read failed.*storage failed/.test(line)));
});

test("a delayed summary preserves token pressure from messages arriving while waiting", async () => {
  const { core, calls } = makeCore({ overviews: ["", "", "", "ready"] });
  const branch = [user("one"), user("two")];
  assert.equal(await core.onTurnSynced(120, branch), false);
  assert.equal(await core.onTurnSynced(17, branch), true);
  assert.equal(calls.committed, 1);
  assert.equal(core.state.pendingTokens, 17);
});

test("messages appended while an archive is pending are not included in its frozen boundary", async () => {
  const { core, calls } = makeCore({ overviews: ["", "", "", "ready"] });
  const initial = [user("one"), assistant("a"), user("two")];
  assert.equal(await core.onTurnSynced(120, initial), false);
  const extended = [...initial, assistant("b"), user("three")];
  assert.equal(await core.onTurnSynced(9, extended), true);
  assert.equal(core.state.coveredUserTurns, 1);
  assert.equal(core.state.pendingTokens, 9);
  assert.equal(calls.committed, 1);
});

test("queued transient capture is drained before commit", async () => {
  const { core, calls } = makeCore({
    syncResult: { added: 1, tokens: 5, allDelivered: false, queued: 1, permanentFailures: 0 },
  });
  const branch = [user("one"), user("two")];
  assert.equal(await core.onTurnSynced(120, branch), true);
  assert.equal(calls.flushed, 1);
  assert.equal(calls.committed, 1);
});

test("permanent capture failure persists a gap and blocks later takeover", async () => {
  const { core, calls } = makeCore({
    syncResult: { added: 1, tokens: 5, allDelivered: false, queued: 0, permanentFailures: 1 },
  });
  const branch = [user("one"), user("two")];
  assert.equal(await core.onTurnSynced(120, branch), false);
  assert.equal(core.state.captureGap, true);
  assert.equal(calls.committed, 0);
  assert.equal(await core.onTurnSynced(120, branch), false);
  assert.equal(calls.synced, 1);
});

test("frozen boundary is abandoned after the branch changes", async () => {
  let active = [user("one"), assistant("old"), user("two")];
  const { core, calls } = makeCore({
    io: {
      readArchiveOverview: async (uri) => {
        calls.overviewUris.push(uri);
        active = [user("one"), assistant("changed"), user("two")];
        return "fresh overview";
      },
    },
  });
  assert.equal(await core.onTurnSynced(120, () => active), false);
  assert.equal(core.state.coveredUserTurns, 0);
  assert.equal(core.state.captureGap, false);
  assert.match(calls.logs.at(-1), /boundary no longer/);
});

test("frozen boundary uses entry identity when two branches have the same content", async () => {
  let active = [
    { id: "u1", type: "message", message: user("one") },
    { id: "a1", type: "message", message: assistant("same") },
    { id: "u2", type: "message", message: user("two") },
  ];
  const { core, calls } = makeCore({
    io: {
      readArchiveOverview: async (uri) => {
        calls.overviewUris.push(uri);
        active = [
          { id: "u1-fork", type: "message", message: user("one") },
          { id: "a1-fork", type: "message", message: assistant("same") },
          { id: "u2-fork", type: "message", message: user("two") },
        ];
        return "fresh overview";
      },
    },
  });
  assert.equal(await core.onTurnSynced(120, () => active), false);
  assert.equal(core.state.coveredUserTurns, 0);
  assert.equal(core.state.captureGap, false);
});

test("frozen boundary rejects a fork after the cut but accepts append-only growth", async () => {
  const initial = [
    { id: "u1", type: "message", message: user("one") },
    { id: "a1", type: "message", message: assistant("answer one") },
    { id: "u2", type: "message", message: user("two") },
    { id: "a2", type: "message", message: assistant("answer two") },
  ];
  let active = initial;
  const forked = makeCore({
    io: {
      readArchiveOverview: async (uri) => {
        forked.calls.overviewUris.push(uri);
        active = [...initial.slice(0, 2), { id: "u2-fork", type: "message", message: user("forked") }];
        return "fresh overview";
      },
    },
  });
  assert.equal(await forked.core.onTurnSynced(120, () => active), false);
  assert.equal(forked.core.state.coveredUserTurns, 0);

  active = initial;
  const appended = makeCore({
    io: {
      readArchiveOverview: async (uri) => {
        appended.calls.overviewUris.push(uri);
        active = [...initial, { id: "u3", type: "message", message: user("three") }];
        return "fresh overview";
      },
    },
  });
  assert.equal(await appended.core.onTurnSynced(120, () => active), true);
  assert.equal(appended.core.state.coveredUserTurns, 1);
});

test("two consecutive archives advance one complete user turn at a time", async () => {
  let nextArchive = 0;
  const { core, calls } = makeCore({
    io: {
      commit: async (opts) => {
        calls.committed++;
        calls.lastCommitOpts = opts;
        nextArchive++;
        return { status: "accepted", archived: true, archive_uri: `viking://user/x/sessions/s/history/archive_00${nextArchive}` };
      },
    },
  });
  const first = [user("one"), assistant("a"), user("two")];
  assert.equal(await core.onTurnSynced(120, first), true);
  assert.equal(core.state.coveredUserTurns, 1);

  const second = [...first, assistant("b"), user("three")];
  assert.equal(await core.onTurnSynced(120, second), true);
  assert.equal(core.state.coveredUserTurns, 2);
  assert.equal(calls.overviewUris.at(-1), "viking://user/x/sessions/s/history/archive_002");
  assert.equal(calls.committed, 2);
});

test("concurrent commitAndAdvance calls are serialized", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const { core, calls } = makeCore({
    io: {
      flush: async () => {
        calls.flushed++;
        await gate;
        return true;
      },
    },
  });
  const branch = [user("one"), user("two")];
  core.transformContext(branch);
  const first = core.commitAndAdvance(branch);
  const second = core.commitAndAdvance(branch);
  assert.equal(await second, false);
  release();
  assert.equal(await first, true);
  assert.equal(calls.committed, 1);
});

test("handleBeforeCompact returns OV summary and resets boundary", async () => {
  const { core } = makeCore({ overviews: ["compact overview"] });
  core.restore([
    { type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: { coveredUserTurns: 2, overview: "old", pendingTokens: 50 } },
  ]);
  const branch = [user("one"), assistant("a")];
  const result = await core.handleBeforeCompact({ firstKeptEntryId: "entry-3", tokensBefore: 1234 }, branch);
  assert.equal(result.compaction.firstKeptEntryId, "entry-3");
  assert.equal(result.compaction.tokensBefore, 1234);
  assert.equal(result.compaction.details.source, "openviking");
  assert.match(result.compaction.summary, /compact overview/);
  assert.equal(core.state.coveredUserTurns, 0);
  assert.equal(core.state.pendingTokens, 0);
});

test("handleBeforeCompact fail-opens without firstKeptEntryId or overview", async () => {
  const { core } = makeCore({ overviews: [""] });
  assert.equal(await core.handleBeforeCompact({ tokensBefore: 1 }), undefined);
  assert.equal(await core.handleBeforeCompact({ firstKeptEntryId: "x", tokensBefore: 1 }), undefined);
});

test("handleBeforeCompact uses pi's keep position, archives all messages, and honours cancellation", async () => {
  const branch = [user("one"), assistant("a"), user("two")];
  const normal = makeCore({ overviews: ["native overview"] });
  const result = await normal.core.handleBeforeCompact(
    { firstKeptEntryId: "pi-kept", tokensBefore: 44, signal: new AbortController().signal },
    branch,
  );
  assert.equal(result.compaction.firstKeptEntryId, "pi-kept");
  assert.equal(normal.calls.lastCommitOpts.keepRecentCount, 0);
  assert.equal(normal.calls.lastSyncBranch, branch);

  const cancelled = makeCore();
  const controller = new AbortController();
  controller.abort();
  assert.equal(await cancelled.core.handleBeforeCompact(
    { firstKeptEntryId: "pi-kept", signal: controller.signal }, branch,
  ), undefined);
  assert.equal(cancelled.calls.committed, 0);
});

test("handleBeforeCompact cancels after an in-flight overview read", async () => {
  const controller = new AbortController();
  const { core } = makeCore({
    io: { readArchiveOverview: async () => { controller.abort(); return "too late"; } },
  });
  assert.equal(await core.handleBeforeCompact(
    { firstKeptEntryId: "pi-kept", signal: controller.signal }, [user("one")],
  ), undefined);
  assert.equal(core.state.overview, "");
  assert.ok(core.state.pendingArchive);
});

test("handleBeforeCompact catches transport failures and returns control to pi", async () => {
  const { core, calls } = makeCore({ io: { commit: async () => { throw new Error("offline"); } } });
  assert.equal(await core.handleBeforeCompact(
    { firstKeptEntryId: "pi-kept" }, [user("one")],
  ), undefined);
  assert.ok(calls.logs.some((line) => /native compaction archive failed.*offline/.test(line)));
});

test("native compaction does not start another archive while a takeover summary is pending", async () => {
  const branch = [user("one"), user("two")];
  const { core, calls } = makeCore({ overviews: ["", "", ""] });
  await core.onTurnSynced(120, branch);
  assert.ok(core.state.pendingArchive);
  assert.equal(await core.handleBeforeCompact({ firstKeptEntryId: "pi-kept" }, branch), undefined);
  assert.equal(calls.committed, 1);
});

test("a timed-out native archive is persisted but never reused as a takeover boundary", async () => {
  const branch = [user("one"), assistant("a")];
  const first = makeCore({ overviews: ["", "", ""] });
  first.core.restore([{ type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: {
    coveredUserTurns: 1, overview: "old overview", pendingTokens: 20,
  } }]);
  assert.equal(await first.core.handleBeforeCompact({ firstKeptEntryId: "pi-kept" }, branch), undefined);
  assert.equal(first.core.state.pendingArchive.nativeCompaction, true);
  assert.equal(first.core.state.coveredUserTurns, 0);

  const resumed = makeCore({ overviews: ["late native overview"] });
  resumed.core.restore([{ type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: first.core.persistedState() }]);
  assert.equal(await resumed.core.onTurnSynced(5, branch), false);
  assert.equal(resumed.calls.committed, 0);
  assert.equal(resumed.core.state.coveredUserTurns, 0);
  assert.equal(resumed.core.state.pendingArchive, null);
  assert.equal(resumed.core.state.archiveUri, "viking://user/x/sessions/s/history/archive_001");
});

test("recovery hint is outside the overview budget and requires list plus read", () => {
  const archiveUri = "viking://user/x/sessions/s/history/archive_001";
  const both = makeCore({ tools: ["openviking_list", "openviking_read", "openviking_grep"] });
  both.core.restore([{ type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: {
    coveredUserTurns: 1, overview: "x".repeat(1000), archiveUri,
    historyUri: "viking://user/x/sessions/s/history",
  } }]);
  const out = both.core.transformContext([user("one"), assistant("a"), user("two")]);
  assert.match(out[0].content, /openviking_list/);
  assert.match(out[0].content, /openviking_read/);
  assert.match(out[0].content, /offset and limit/);
  assert.match(out[0].content, /captured historical messages/);
  assert.ok(out[0].content.length > 1000);

  const missingRead = makeCore({ tools: ["openviking_list"] });
  missingRead.core.restore([{ type: "custom", customType: TAKEOVER_ENTRY_TYPE, data: {
    coveredUserTurns: 1, overview: "summary", archiveUri,
    historyUri: "viking://user/x/sessions/s/history",
  } }]);
  const noHint = missingRead.core.transformContext([user("one"), assistant("a"), user("two")]);
  assert.doesNotMatch(noHint[0].content, /openviking_list/);
});

test("native compaction summary uses the same recovery hint", async () => {
  const { core } = makeCore({
    tools: ["openviking_list", "openviking_read"],
    overviews: ["compact overview"],
  });
  const result = await core.handleBeforeCompact(
    { firstKeptEntryId: "pi-kept", tokensBefore: 100 }, [user("one")],
  );
  assert.match(result.compaction.summary, /captured historical messages/);
  assert.match(result.compaction.summary, /openviking_list/);
  assert.match(result.compaction.summary, /openviking_read/);
});

test("disabled takeover is a passthrough", async () => {
  const { core, calls } = makeCore({ config: { takeoverEnabled: false } });
  const messages = [user("one"), user("two")];
  assert.equal(core.transformContext(messages), messages);
  assert.equal(await core.onTurnSynced(999), false);
  assert.equal(await core.commitAndAdvance(), false);
  assert.equal(await core.handleBeforeCompact({ firstKeptEntryId: "x" }), undefined);
  assert.equal(calls.committed, 0);
});

test("shutdown persists deduped state once", async () => {
  const { core, calls } = makeCore({ watermark: 9 });
  core.transformContext([user("one")]);
  await core.shutdown();
  await core.shutdown();
  assert.equal(calls.persisted.length, 1);
  assert.equal(calls.persisted[0].data.syncedEntryCount, 9);
});
