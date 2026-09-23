export const TAKEOVER_ENTRY_TYPE = "ov-takeover";
export const OVERVIEW_MARKER = "[OpenViking Session Context]";

const DEFAULT_CONFIG = {
  takeoverEnabled: true,
  takeoverTokenThreshold: 30000,
  takeoverKeepRecentTurns: 3,
  takeoverOverviewBudget: 3000,
  takeoverOverviewPollMs: 2000,
  takeoverOverviewPollMax: 15,
};

function numberOr(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function takeoverConfig(config = {}) {
  return {
    takeoverEnabled: config.takeoverEnabled !== false,
    takeoverTokenThreshold: Math.max(0, numberOr(config.takeoverTokenThreshold, DEFAULT_CONFIG.takeoverTokenThreshold)),
    takeoverKeepRecentTurns: Math.max(0, numberOr(config.takeoverKeepRecentTurns, DEFAULT_CONFIG.takeoverKeepRecentTurns)),
    takeoverOverviewBudget: Math.max(1, numberOr(config.takeoverOverviewBudget, DEFAULT_CONFIG.takeoverOverviewBudget)),
    takeoverOverviewPollMs: Math.max(0, numberOr(config.takeoverOverviewPollMs, DEFAULT_CONFIG.takeoverOverviewPollMs)),
    takeoverOverviewPollMax: Math.max(1, numberOr(config.takeoverOverviewPollMax, DEFAULT_CONFIG.takeoverOverviewPollMax)),
  };
}

function asEntry(value) {
  if (!value || typeof value !== "object") return null;
  return value.entry && typeof value.entry === "object" ? value.entry : value;
}

/** Unwrap a branch entry to the message it carries, for turn counting. */
function entryMessage(entry) {
  if (!entry || typeof entry !== "object") return null;
  if (entry.type === "message" && entry.message && typeof entry.message === "object") {
    return entry.message;
  }
  if (entry.message && typeof entry.message === "object") return entry.message;
  return entry;
}

function currentBranch(branchOrGetter) {
  if (typeof branchOrGetter === "function") {
    try {
      const value = branchOrGetter();
      return Array.isArray(value) ? value : [];
    } catch {
      return [];
    }
  }
  return Array.isArray(branchOrGetter) ? branchOrGetter : [];
}

function flattenValue(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(flattenValue).filter(Boolean).join("");
  if (!value || typeof value !== "object") return "";
  if (typeof value.text === "string") return value.text;
  if (typeof value.input_text === "string") return value.input_text;
  if (typeof value.output_text === "string") return value.output_text;
  if (typeof value.content === "string") return value.content;
  if (Array.isArray(value.content)) return flattenValue(value.content);
  return "";
}

export function flattenContent(msg) {
  if (!msg || typeof msg !== "object") return "";
  return flattenValue(msg.content);
}

export function fingerprintMessage(msg) {
  const text = flattenContent(msg);
  return `${msg?.role || ""}:${text.length}:${text.slice(0, 200)}`;
}

export function isUserTurnStart(msg) {
  if (!msg || msg.role !== "user") return false;
  return !flattenContent(msg).startsWith(OVERVIEW_MARKER);
}

export function countUserTurns(messages) {
  let count = 0;
  for (const msg of Array.isArray(messages) ? messages : []) {
    if (isUserTurnStart(msg)) count++;
  }
  return count;
}

export function findBoundaryIndex(messages, coveredUserTurns) {
  const target = Math.max(0, Math.floor(Number(coveredUserTurns) || 0)) + 1;
  let seen = 0;
  for (let i = 0; i < (Array.isArray(messages) ? messages.length : 0); i++) {
    if (!isUserTurnStart(messages[i])) continue;
    seen++;
    if (seen === target) return i;
  }
  return -1;
}

export function estimateTokens(text) {
  const value = String(text || "");
  if (!value) return 0;
  let cjk = 0;
  let other = 0;
  for (const ch of value) {
    if (ch.codePointAt(0) >= 0x3000) cjk++;
    else other++;
  }
  return Math.ceil(cjk * 1.5 + other / 4);
}

export function truncateToTokens(text, budget) {
  const value = String(text || "");
  const limit = Math.max(0, Math.floor(Number(budget) || 0));
  if (!value || limit <= 0) return "";
  if (estimateTokens(value) <= limit) return value;

  let lo = 0;
  let hi = value.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (estimateTokens(value.slice(0, mid)) <= limit) lo = mid;
    else hi = mid - 1;
  }
  return value.slice(0, lo);
}

function partText(part) {
  if (!part || typeof part !== "object") return "";
  if (typeof part.text === "string") return part.text;
  if (part.type === "tool") {
    const payload = {
      name: part.tool_name,
      input: part.tool_input,
      output: part.tool_output,
      status: part.tool_status,
    };
    try {
      return JSON.stringify(payload);
    } catch {
      return String(part.tool_name || part.tool_output || "");
    }
  }
  return flattenValue(part);
}

export function estimatePayloadTokens(payload) {
  if (!payload || typeof payload !== "object") return 0;
  if (typeof payload.content === "string") return estimateTokens(payload.content);
  if (Array.isArray(payload.parts)) {
    return estimateTokens(payload.parts.map(partText).filter(Boolean).join("\n\n"));
  }
  if (Array.isArray(payload.content)) {
    return estimateTokens(payload.content.map(partText).filter(Boolean).join("\n\n"));
  }
  return 0;
}

export function buildOverviewMessage(
  overview,
  firstKeptTs = 0,
  budget = DEFAULT_CONFIG.takeoverOverviewBudget,
  recoveryHint = "",
) {
  const raw = String(overview || "");
  const truncated = truncateToTokens(raw, budget);
  const body = truncated === raw ? raw : `${truncated}\n...(truncated)`;
  const timestamp = Number.isFinite(Number(firstKeptTs)) ? Number(firstKeptTs) - 1 : 0;
  return {
    role: "user",
    content:
      `${OVERVIEW_MARKER} Earlier conversation was archived to OpenViking and summarized below. ` +
      `Use OpenViking for related context when its tools are available.\n\n${body}${recoveryHint}`,
    timestamp,
  };
}

export function countUndeliveredForSession(pendingEntries, sid) {
  if (!sid) return 0;
  let count = 0;
  for (const item of Array.isArray(pendingEntries) ? pendingEntries : []) {
    const entry = asEntry(item);
    if (entry?.type === "addMessage" && entry.sessionId === sid) count++;
  }
  return count;
}

/** Derive the parent history directory only for the server's archive shape. */
export function deriveHistoryUri(archiveUri) {
  const cleaned = String(archiveUri || "").trim().replace(/\/+$/, "");
  const m = /^(.*\/history)\/archive_\d+$/.exec(cleaned);
  return m ? m[1] : "";
}

/**
 * Read a commit response and decide whether it archived what this takeover
 * meant to trim.
 *
 * The boundary may only advance behind a commit that actually wrote an archive:
 * `status: "skipped"` (nothing to archive), an explicit `archived: false`, or a
 * missing `archive_uri` all mean there is nothing to point the overview at, and
 * the old `/context` summary must never be reused to fill the gap. Older server
 * builds omit `status`/`archived` but still return `archive_uri`; those are
 * accepted on the URI alone.
 */
export function commitOutcome(committed) {
  if (!committed || typeof committed !== "object") {
    return { accepted: false, reason: "no_result" };
  }
  const status = typeof committed.status === "string" ? committed.status : "";
  if (status === "skipped") {
    return { accepted: false, reason: `skipped:${committed.reason || "unknown"}` };
  }
  if (status && status !== "accepted") {
    return { accepted: false, reason: `status:${status}` };
  }
  if (committed.archived === false) {
    return { accepted: false, reason: "not_archived" };
  }
  const archiveUri =
    typeof committed.archive_uri === "string" ? committed.archive_uri.trim() : "";
  if (!archiveUri) {
    return { accepted: false, reason: "no_archive_uri" };
  }
  return {
    accepted: true,
    reason: "accepted",
    archiveUri,
    taskId: typeof committed.task_id === "string" ? committed.task_id : "",
  };
}

export class TakeoverCore {
  constructor({ config = {}, io = {} } = {}) {
    this.config = takeoverConfig(config);
    this.io = {
      flush: io.flush || (async () => true),
      // Sync the current branch before committing so the archive covers exactly
      // the messages this trim is about to drop; not every caller has synced.
      syncBranch: io.syncBranch || (async () => ({ added: 0, tokens: 0, allDelivered: true })),
      commit: io.commit || (async () => null),
      // Read the Working Memory of a SPECIFIC archive by its uri, so the summary
      // provably belongs to this commit and not to an older /context archive.
      readArchiveOverview: io.readArchiveOverview || (async () => null),
      // How many messages the capture path would actually send for a slice of
      // the branch — the server keep_recent_count is a message count, not a
      // user-turn count, and it must exclude system/custom/filtered entries.
      captureCount: io.captureCount || (() => 0),
      persistEntry: io.persistEntry || (() => {}),
      getWatermark: io.getWatermark || (() => 0),
      // Messages OpenViking will never receive (4xx / retries exhausted): a
      // capture gap that must stop takeover cutting across it.
      droppedCount: io.droppedCount || (() => 0),
      availableTools: io.availableTools || (() => []),
      sleep: io.sleep || ((ms) => new Promise((resolve) => setTimeout(resolve, ms))),
      log: io.log || (() => {}),
    };
    this.coveredUserTurns = 0;
    this.overview = "";
    this.fingerprint = null;
    this.pendingTokens = 0;
    this.lastSeenUserTurns = 0;
    this.syncedEntryCount = 0;
    this.committing = false;
    this.lastPersisted = "";
    this.archiveUri = "";
    this.historyUri = "";
    // A commit that archived but whose Working Memory was not ready in time:
    // { archiveUri, taskId, coveredUserTurns, boundaryUserTurns, fingerprint,
    //   syncedEntryCount, frozenTokens }. While set, no second
    // takeover commit runs — later turns only re-check this same archive.
    this.pendingArchive = null;
    // A permanent delivery gap in this session: the archive is missing messages,
    // so the boundary must never advance past it. Survives restart.
    this.captureGap = false;
  }

  get enabled() {
    return this.config.takeoverEnabled;
  }

  get state() {
    return {
      coveredUserTurns: this.coveredUserTurns,
      overview: this.overview,
      fingerprint: this.fingerprint,
      pendingTokens: this.pendingTokens,
      lastSeenUserTurns: this.lastSeenUserTurns,
      syncedEntryCount: this.syncedEntryCount,
      committing: this.committing,
      pendingArchive: this.pendingArchive,
      captureGap: this.captureGap,
      archiveUri: this.archiveUri,
      historyUri: this.historyUri,
    };
  }

  restore(entries) {
    for (let i = (Array.isArray(entries) ? entries.length : 0) - 1; i >= 0; i--) {
      const entry = entries[i];
      const isTakeoverEntry =
        (entry?.type === "custom" && entry.customType === TAKEOVER_ENTRY_TYPE) ||
        entry?.customType === TAKEOVER_ENTRY_TYPE ||
        entry?.type === TAKEOVER_ENTRY_TYPE;
      const data = isTakeoverEntry ? entry.data : null;
      if (!data || typeof data !== "object") continue;

      this.coveredUserTurns = Math.max(0, Math.floor(Number(data.coveredUserTurns) || 0));
      this.overview = typeof data.overview === "string" ? data.overview : "";
      this.fingerprint = typeof data.fingerprint === "string" ? data.fingerprint : null;
      this.pendingTokens = Math.max(0, Math.floor(Number(data.pendingTokens) || 0));
      this.lastSeenUserTurns = Math.max(0, Math.floor(Number(data.lastSeenUserTurns) || 0));
      this.syncedEntryCount = Math.max(0, Math.floor(Number(data.syncedEntryCount) || 0));
      // New fields default to safe absences so an old persisted entry restores.
      this.pendingArchive = restorePendingArchive(data.pendingArchive);
      this.captureGap = data.captureGap === true;
      this.archiveUri = typeof data.archiveUri === "string" ? data.archiveUri : "";
      this.historyUri = typeof data.historyUri === "string" ? data.historyUri : "";
      this.lastPersisted = JSON.stringify(this.persistedState());
      this.log(
        `takeover: restored boundary at ${this.coveredUserTurns} user turns, ${this.pendingTokens} pending tokens` +
          (this.pendingArchive ? `, pending archive ${this.pendingArchive.archiveUri}` : "") +
          (this.captureGap ? ", capture gap set" : ""),
      );
      return this.state;
    }
    return this.state;
  }

  transformContext(messages) {
    const list = Array.isArray(messages) ? messages : [];
    this.lastSeenUserTurns = countUserTurns(list);

    if (!this.enabled) return list;
    if (this.coveredUserTurns <= 0 || !this.overview) return list;

    const boundaryIdx = findBoundaryIndex(list, this.coveredUserTurns);
    if (boundaryIdx <= 0) {
      this.resetBoundary("history shorter than boundary");
      return list;
    }

    const lastCovered = list[boundaryIdx - 1];
    const fp = fingerprintMessage(lastCovered);
    if (this.fingerprint === null) {
      this.fingerprint = fp;
    } else if (this.fingerprint !== fp) {
      this.resetBoundary("fingerprint mismatch");
      return list;
    }

    const kept = list.slice(boundaryIdx);
    const firstKeptTs = typeof kept[0]?.timestamp === "number" ? kept[0].timestamp : 1;
    // Preserve every system message in the covered region, in its original
    // order. On pi >= 0.86 the transcript carries the base prompt and its tool
    // declarations as the leading `system` message, and mid-conversation tool
    // additions/removals, section updates and appended instructions as later
    // ones (see @earendil-works/pi-ai `getCurrentTools`/`getCurrentSystemMessage`).
    // Slicing them off with the covered turns would strip the model's tools and
    // instructions — the overview only stands in for the conversation, never for
    // the system state. On 0.80.3 the transcript has no system messages here, so
    // this preserves nothing and the behaviour is unchanged; on 0.87 the host
    // reconciles tools against the executable set on every request, so keeping
    // the existing declarations introduces no duplicate. System messages inside
    // the retained tail are left where they are, not hoisted in front.
    const coveredSystem = [];
    for (let i = 0; i < boundaryIdx; i++) {
      if (list[i]?.role === "system") coveredSystem.push(list[i]);
    }
    return [
      ...coveredSystem,
      buildOverviewMessage(
        this.overview, firstKeptTs, this.config.takeoverOverviewBudget, this.recoveryHint(),
      ),
      ...kept,
    ];
  }

  async onTurnSynced(estTokens, branch = []) {
    if (!this.enabled) return false;
    const turnTokens = Math.max(0, Math.floor(Number(estTokens) || 0));
    // A commit already archived and is only waiting for its Working Memory:
    // account for this turn, then finish that one instead of opening another.
    // finishArchive subtracts only the frozen token snapshot, leaving this new
    // pressure for the next boundary.
    this.pendingTokens += turnTokens;
    if (this.pendingArchive) return this.commitAndAdvance(branch);
    if (this.pendingTokens < this.config.takeoverTokenThreshold) {
      this.syncedEntryCount = Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0));
      this.persist();
      return false;
    }
    // Whether there are enough user turns to leave a keep-recent tail is decided
    // authoritatively by freezeBoundary against the branch, not by the
    // transformed `lastSeenUserTurns` count.
    return this.commitAndAdvance(branch);
  }

  /**
   * Advance the boundary behind a confirmed archive, or resume the archive a
   * previous attempt left pending. Never advances across a capture gap and
   * never reuses an old /context summary.
   */
  async commitAndAdvance(branch = []) {
    if (!this.enabled || this.committing) return false;
    this.committing = true;
    try {
      if (this.pendingArchive) return await this.resolvePendingArchive(branch);
      return await this.beginArchive(branch);
    } catch (error) {
      this.log(`takeover: archive preparation failed (${errorMessage(error)}); boundary held`);
      return false;
    } finally {
      this.syncedEntryCount = Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0));
      this.persist();
      this.committing = false;
    }
  }

  /** Freeze a boundary, confirm delivery, commit, and read this archive's summary. */
  async beginArchive(branch) {
    if (this.captureGap) {
      this.log("takeover: capture gap present; boundary held, native compaction stays with pi");
      return false;
    }

    const branchSnapshot = currentBranch(branch);
    const frozen = this.freezeBoundary(branchSnapshot);
    if (!frozen) {
      this.log("takeover: no advanceable boundary; commit skipped");
      return false;
    }

    // Sync this branch, then drain the queue: an empty queue alone does not
    // prove delivery — the just-extracted turn may not have been sent yet.
    const synced = await this.io.syncBranch(branchSnapshot);
    if ((synced?.permanentFailures || 0) > 0 || this.io.droppedCount() > 0) {
      this.markCaptureGap();
      return false;
    }
    // A queued transient failure is allowed to proceed to the bounded drainer;
    // only the barrier's final state proves that every current-session message
    // reached the server.
    const flushed = await this.io.flush();
    // The drain may have exhausted retries or failed to rewrite a queue entry.
    // Record that permanent gap even when an orphaned processing file keeps the
    // barrier closed.
    if (this.io.droppedCount() > 0) {
      this.markCaptureGap();
      return false;
    }
    if (!flushed) {
      this.log("takeover: flush barrier closed; commit postponed");
      return false;
    }
    frozen.syncedEntryCount = Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0));

    const committed = await this.io.commit({
      queueOnFailure: false,
      keepRecentCount: frozen.keepRecentCount,
    });
    const outcome = commitOutcome(committed);
    if (!outcome.accepted) {
      // No archive was written: hold the boundary, keep the token pressure so
      // the next threshold crossing retries, and never fall back to /context.
      if (outcome.reason === "no_result") {
        this.log("takeover: commit failed; boundary held");
      } else if (outcome.reason.startsWith("skipped:")) {
        this.log(`takeover: archive skipped (${outcome.reason.slice(8)}); boundary held`);
      } else {
        this.log(`takeover: commit returned no usable archive (${outcome.reason}); boundary held`);
      }
      return false;
    }

    return await this.finishArchive(branch, {
      ...frozen,
      archiveUri: outcome.archiveUri,
      taskId: outcome.taskId,
      historyUri: deriveHistoryUri(outcome.archiveUri),
    });
  }

  /** Re-check the archive a prior attempt committed; advance once its summary lands. */
  async resolvePendingArchive(branch) {
    const pending = this.pendingArchive;
    if (!pending) return false;
    if (pending.nativeCompaction) {
      const overview = await this.pollArchiveOverview(pending.archiveUri);
      if (!overview) return false;
      // Pi already ran its own compaction after the earlier hook returned
      // undefined. This archive is useful for recovery, but must not install a
      // second takeover boundary over Pi's new context.
      this.pendingArchive = null;
      this.pendingTokens = Math.max(0, this.pendingTokens - (Number(pending.frozenTokens) || 0));
      this.archiveUri = pending.archiveUri;
      this.historyUri = typeof pending.historyUri === "string" ? pending.historyUri : "";
      this.persist();
      this.log(`takeover: native-compaction archive completed at ${pending.archiveUri}`);
      return false;
    }
    return await this.finishArchive(branch, pending, { committed: true });
  }

  /**
   * Poll the archive's own `.overview.md`, then advance — but only after
   * re-checking that the frozen boundary is still part of the current branch.
   */
  async finishArchive(branch, frozen, { committed = false } = {}) {
    const overview = await this.pollArchiveOverview(frozen.archiveUri);
    if (!overview) {
      // Accepted but not summarized yet: remember exactly this archive and its
      // frozen boundary, keep local history, and let a later turn re-check the
      // SAME archive rather than committing again.
      this.pendingArchive = {
        archiveUri: frozen.archiveUri,
        taskId: frozen.taskId || "",
        coveredUserTurns: frozen.coveredUserTurns,
        boundaryUserTurns: frozen.boundaryUserTurns,
        fingerprint: frozen.fingerprint,
        boundaryEntryId: frozen.boundaryEntryId || "",
        branchEntryCount: frozen.branchEntryCount,
        branchTipEntryId: frozen.branchTipEntryId || "",
        branchTipFingerprint: frozen.branchTipFingerprint,
        syncedEntryCount: frozen.syncedEntryCount,
        historyUri: frozen.historyUri || "",
        frozenTokens: frozen.frozenTokens,
      };
      this.persist();
      this.log(
        `takeover: ${frozen.archiveUri} committed, Working Memory not ready; ` +
          (committed ? "still waiting" : "waiting on later turns"),
      );
      return false;
    }

    // The archive is summarized. Confirm the frozen boundary still describes the
    // current history before trimming to it — a branch switch during the wait
    // must abandon this advance, not cut somewhere else.
    if (!this.boundaryStillValid(branch, frozen)) {
      this.pendingArchive = null;
      this.persist();
      this.log("takeover: frozen boundary no longer in history; advance abandoned");
      return false;
    }

    if (frozen.coveredUserTurns > this.coveredUserTurns) {
      this.coveredUserTurns = frozen.coveredUserTurns;
      // Persist the identity of the exact covered prefix. Leaving this null
      // would let the next process learn a fingerprint from whichever branch
      // happened to be active after restart.
      this.fingerprint = frozen.fingerprint;
    }
    this.overview = overview;
    this.archiveUri = frozen.archiveUri;
    // Fresh pending records already carry a derived history URI. Do not guess
    // one while restoring an older state shape that lacks this new field.
    this.historyUri = typeof frozen.historyUri === "string" ? frozen.historyUri : "";
    this.pendingArchive = null;
    // Subtract only the pressure this trim froze; tokens accrued while waiting
    // for the summary belong to the next boundary and are not cleared.
    this.pendingTokens = Math.max(0, this.pendingTokens - (Number(frozen.frozenTokens) || 0));
    this.syncedEntryCount = Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0));
    this.persist();
    this.log(`takeover: boundary advanced to ${this.coveredUserTurns} user turns via ${frozen.archiveUri}`);
    return true;
  }

  /**
   * Compute the candidate boundary from the branch: keep the last
   * `takeoverKeepRecentTurns` user turns, archive the rest. Returns the frozen
   * boundary (position, fingerprint, exact keep_recent_count, watermark, token
   * snapshot) or null when there is nothing new to archive.
   */
  freezeBoundary(branch) {
    const entries = currentBranch(branch);
    const messages = entries.map(entryMessage);
    const branchUserTurns = countUserTurns(messages);
    const boundaryUserTurns = branchUserTurns - this.config.takeoverKeepRecentTurns;
    if (boundaryUserTurns <= 0) return null;
    if (boundaryUserTurns <= this.coveredUserTurns) return null;

    // With keepRecentTurns=0, the cut is after the final message rather than at
    // the start of a following user turn.
    const boundaryIdx = boundaryUserTurns === branchUserTurns
      ? messages.length
      : findBoundaryIndex(messages, boundaryUserTurns);
    if (boundaryIdx <= 0) return null;

    return {
      coveredUserTurns: boundaryUserTurns,
      boundaryUserTurns,
      fingerprint: fingerprintMessage(messages[boundaryIdx - 1]),
      boundaryEntryId: typeof entries[boundaryIdx - 1]?.id === "string"
        ? entries[boundaryIdx - 1].id
        : "",
      branchEntryCount: entries.length,
      branchTipEntryId: typeof entries[entries.length - 1]?.id === "string"
        ? entries[entries.length - 1].id
        : "",
      branchTipFingerprint: entries.length
        ? fingerprintMessage(messages[messages.length - 1])
        : null,
      // keep_recent_count is a server message count: how many captured payloads
      // the retained tail produces, system/custom/filtered entries excluded.
      keepRecentCount: Math.max(0, Math.floor(Number(this.io.captureCount(entries.slice(boundaryIdx))) || 0)),
      syncedEntryCount: Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0)),
      frozenTokens: this.pendingTokens,
    };
  }

  /** Whether the frozen boundary still names the same last-covered message. */
  boundaryStillValid(branch, frozen) {
    const entries = currentBranch(branch);
    const messages = entries.map(entryMessage);
    const frozenCount = Math.max(0, Math.floor(Number(frozen.branchEntryCount) || 0));
    if (frozenCount > 0) {
      if (entries.length < frozenCount) return false;
      const frozenTip = entries[frozenCount - 1];
      if (frozen.branchTipEntryId) {
        if (frozenTip?.id !== frozen.branchTipEntryId) return false;
      } else if (frozen.branchTipFingerprint &&
        fingerprintMessage(entryMessage(frozenTip)) !== frozen.branchTipFingerprint) {
        return false;
      }
    }
    const totalTurns = countUserTurns(messages);
    const boundaryIdx = frozen.boundaryUserTurns === totalTurns
      ? messages.length
      : findBoundaryIndex(messages, frozen.boundaryUserTurns);
    if (boundaryIdx <= 0) return false;
    if (frozen.boundaryEntryId) {
      return entries[boundaryIdx - 1]?.id === frozen.boundaryEntryId;
    }
    return fingerprintMessage(messages[boundaryIdx - 1]) === frozen.fingerprint;
  }

  markCaptureGap() {
    if (this.captureGap) return;
    this.captureGap = true;
    this.pendingArchive = null;
    this.persist();
    this.log("takeover: capture gap recorded; takeover disabled for this session, native compaction stays with pi");
  }

  recordCaptureGap() {
    this.markCaptureGap();
  }

  async handleBeforeCompact(preparation = {}, branch = []) {
    if (!this.enabled || this.committing) return undefined;
    if (!preparation.firstKeptEntryId) return undefined;
    // A gap means the archive is missing messages, so an OpenViking summary
    // cannot faithfully cover pi's cut — let pi compact natively.
    if (this.captureGap) return undefined;
    // A pending takeover archive covers only its frozen head; it cannot stand
    // in for the broader range pi selected. Never create a second archive while
    // the first one is unresolved.
    if (this.pendingArchive) return undefined;
    if (preparation.signal?.aborted) return undefined;

    this.committing = true;
    try {
      const branchSnapshot = currentBranch(branch);
      // Sync the latest branch first; native compaction archives all captured
      // history (keepRecentCount 0) and hands pi its own firstKeptEntryId, so
      // the summary and the retained tail may overlap.
      const synced = await this.io.syncBranch(branchSnapshot);
      if ((synced?.permanentFailures || 0) > 0 || this.io.droppedCount() > 0) {
        this.markCaptureGap();
        return undefined;
      }
      if (preparation.signal?.aborted) return undefined;
      const flushed = await this.io.flush();
      if (this.io.droppedCount() > 0) {
        this.markCaptureGap();
        return undefined;
      }
      if (!flushed) return undefined;
      if (preparation.signal?.aborted) return undefined;

      const committed = await this.io.commit({ queueOnFailure: false, keepRecentCount: 0 });
      const outcome = commitOutcome(committed);
      if (!outcome.accepted) {
        if (outcome.reason === "no_result") {
          this.log("takeover: native compaction commit failed; using pi compaction");
        } else if (outcome.reason.startsWith("skipped:")) {
          this.log(`takeover: native compaction archive skipped (${outcome.reason.slice(8)}); using pi compaction`);
        } else {
          this.log(`takeover: native compaction returned no usable archive (${outcome.reason}); using pi compaction`);
        }
        return undefined;
      }

      const overview = await this.pollArchiveOverview(outcome.archiveUri, preparation.signal);
      if (!overview) {
        // The server accepted the archive, but this compaction attempt cannot
        // wait indefinitely. Remember it so ordinary takeover will not create
        // a second archive, while this call falls back to native compaction.
        const nativeMessages = branchSnapshot.map(entryMessage);
        this.pendingArchive = {
          archiveUri: outcome.archiveUri,
          taskId: outcome.taskId || "",
          coveredUserTurns: this.coveredUserTurns,
          boundaryUserTurns: countUserTurns(nativeMessages),
          fingerprint: nativeMessages.length
            ? fingerprintMessage(nativeMessages[nativeMessages.length - 1])
            : null,
          boundaryEntryId: typeof branchSnapshot[branchSnapshot.length - 1]?.id === "string"
            ? branchSnapshot[branchSnapshot.length - 1].id
            : "",
          branchEntryCount: branchSnapshot.length,
          branchTipEntryId: typeof branchSnapshot[branchSnapshot.length - 1]?.id === "string"
            ? branchSnapshot[branchSnapshot.length - 1].id
            : "",
          branchTipFingerprint: nativeMessages.length
            ? fingerprintMessage(nativeMessages[nativeMessages.length - 1])
            : null,
          syncedEntryCount: Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0)),
          historyUri: deriveHistoryUri(outcome.archiveUri),
          frozenTokens: this.pendingTokens,
          nativeCompaction: true,
        };
        // Returning undefined hands this compaction to Pi. Drop the takeover
        // boundary immediately so Pi's rewritten branch is never transformed
        // against the stale pre-compaction cut on the next request.
        this.resetBoundary("pi native compaction owns boundary");
        this.persist();
        return undefined;
      }

      this.overview = overview;
      this.archiveUri = outcome.archiveUri;
      this.historyUri = deriveHistoryUri(outcome.archiveUri);
      this.resetBoundary("pi compaction absorbed boundary");
      this.pendingTokens = 0;
      this.syncedEntryCount = Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0));
      this.persist();

      return {
        compaction: {
          summary:
            `${OVERVIEW_MARKER}\n${this.truncatedOverview()}` +
            this.recoveryHint(outcome.archiveUri, deriveHistoryUri(outcome.archiveUri)),
          firstKeptEntryId: preparation.firstKeptEntryId,
          tokensBefore: Number(preparation.tokensBefore) || 0,
          details: { source: "openviking" },
        },
      };
    } catch (error) {
      this.log(`takeover: native compaction archive failed (${errorMessage(error)}); using pi compaction`);
      return undefined;
    } finally {
      this.committing = false;
    }
  }

  async shutdown() {
    if (!this.enabled) return;
    this.syncedEntryCount = Math.max(0, Math.floor(Number(this.io.getWatermark()) || 0));
    this.persist();
  }

  resetBoundary(reason = "reset") {
    if (this.coveredUserTurns !== 0 || this.fingerprint !== null) {
      this.log(`takeover: boundary reset (${reason})`);
    }
    this.coveredUserTurns = 0;
    this.fingerprint = null;
  }

  truncatedOverview() {
    const raw = String(this.overview || "");
    const truncated = truncateToTokens(raw, this.config.takeoverOverviewBudget);
    return truncated === raw ? raw : `${truncated}\n...(truncated)`;
  }

  /** Append a non-budgeted source recovery hint only when both required tools exist. */
  recoveryHint(archiveUri = this.archiveUri, historyUri = this.historyUri) {
    const tools = new Set(this.io.availableTools());
    if (!tools.has("openviking_list") || !tools.has("openviking_read")) return "";
    const archive = String(archiveUri || "").trim();
    const history = String(historyUri || "").trim();
    if (!archive || !history) return "";
    return (
      `\n\nArchived capture: ${archive}\n` +
      `This contains captured historical messages, not the unfiltered Pi transcript. ` +
      `Semantic search does not retrieve archive source verbatim. Use openviking_list on ${history} ` +
      `to locate archives, then openviking_read with uris=["${archive}/messages.jsonl"], ` +
      `offset and limit to read it in chunks.` +
      (tools.has("openviking_grep") ? ` Use openviking_grep only as an optional locator.` : "")
    );
  }

  persistedState() {
    const rawWatermark = Number(this.io.getWatermark());
    const watermark = Number.isFinite(rawWatermark)
      ? Math.max(0, Math.floor(rawWatermark))
      : this.syncedEntryCount;
    return {
      coveredUserTurns: this.coveredUserTurns,
      overview: truncateToTokens(this.overview, this.config.takeoverOverviewBudget),
      fingerprint: this.fingerprint,
      pendingTokens: this.pendingTokens,
      lastSeenUserTurns: this.lastSeenUserTurns,
      syncedEntryCount: watermark,
      pendingArchive: this.pendingArchive,
      captureGap: this.captureGap,
      archiveUri: this.archiveUri,
      historyUri: this.historyUri,
    };
  }

  persist() {
    try {
      const state = this.persistedState();
      const key = JSON.stringify(state);
      if (key === this.lastPersisted) return;
      this.io.persistEntry(TAKEOVER_ENTRY_TYPE, state);
      this.lastPersisted = key;
    } catch {
      // Best effort. A missed state entry only means the next process sees full history.
    }
  }

  /** Poll one archive's `.overview.md` until its non-empty body lands. */
  async pollArchiveOverview(archiveUri, signal) {
    const uri = String(archiveUri || "").trim();
    if (!uri) return "";
    for (let i = 0; i < this.config.takeoverOverviewPollMax; i++) {
      if (signal?.aborted) return "";
      let value = null;
      try {
        value = await this.io.readArchiveOverview(uri);
      } catch (error) {
        this.log(`takeover: archive overview read failed for ${uri} (${errorMessage(error)})`);
        value = null;
      }
      const overview = typeof value === "string" ? value.trim() : "";
      if (signal?.aborted) return "";
      if (overview) return overview;
      if (i < this.config.takeoverOverviewPollMax - 1 && this.config.takeoverOverviewPollMs > 0) {
        await this.io.sleep(this.config.takeoverOverviewPollMs);
      }
    }
    return "";
  }

  log(message) {
    try {
      this.io.log(message);
    } catch {
      // Logging must not affect pi's context path.
    }
  }
}

/** Rehydrate a persisted pending-archive record, dropping malformed shapes. */
function restorePendingArchive(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const archiveUri = typeof value.archiveUri === "string" ? value.archiveUri.trim() : "";
  if (!archiveUri) return null;
  return {
    archiveUri,
    taskId: typeof value.taskId === "string" ? value.taskId : "",
    coveredUserTurns: Math.max(0, Math.floor(Number(value.coveredUserTurns) || 0)),
    boundaryUserTurns: Math.max(0, Math.floor(Number(value.boundaryUserTurns) || 0)),
    fingerprint: typeof value.fingerprint === "string" ? value.fingerprint : null,
    boundaryEntryId: typeof value.boundaryEntryId === "string" ? value.boundaryEntryId : "",
    branchEntryCount: Math.max(0, Math.floor(Number(value.branchEntryCount) || 0)),
    branchTipEntryId: typeof value.branchTipEntryId === "string" ? value.branchTipEntryId : "",
    branchTipFingerprint: typeof value.branchTipFingerprint === "string"
      ? value.branchTipFingerprint
      : null,
    syncedEntryCount: Math.max(0, Math.floor(Number(value.syncedEntryCount) || 0)),
    historyUri: typeof value.historyUri === "string" ? value.historyUri : "",
    frozenTokens: Math.max(0, Math.floor(Number(value.frozenTokens) || 0)),
    nativeCompaction: value.nativeCompaction === true,
  };
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error || "unknown");
}
