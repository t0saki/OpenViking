export const TAKEOVER_ENTRY_TYPE: "ov-takeover";
export const OVERVIEW_MARKER: "[OpenViking Session Context]";

export interface TakeoverMessage {
  role: string;
  content?: unknown;
  timestamp?: number;
  [key: string]: any;
}

/**
 * A commit that archived but whose Working Memory was not ready in time. While
 * one is set, no second takeover commit runs — later turns re-check this same
 * archive by its uri. Persisted so it survives `pi -c` / `pi -p`.
 */
export interface PendingArchive {
  archiveUri: string;
  taskId: string;
  historyUri: string;
  /** The entry the frozen covered prefix ends at; "" for a native-compaction archive. */
  coveredThroughEntryId: string;
  coveredUserTurns: number;
  frozenTokens: number;
  nativeCompaction?: boolean;
}

export interface TakeoverPersistedState {
  /** The pi entry id the covered prefix ends at (inclusive); "" when none. */
  coveredThroughEntryId?: string;
  /** Display count; also the whole boundary of a 0.4.0 state. */
  coveredUserTurns: number;
  overview: string;
  /** 0.4.0 only: fingerprint of the last covered context message. */
  fingerprint?: string | null;
  pendingTokens: number;
  lastSeenUserTurns?: number;
  syncedEntryCount?: number;
  /** A committed archive still waiting on its Working Memory, or null. */
  pendingArchive?: PendingArchive | null;
  /** A permanent delivery gap: the archive is missing messages. Blocks advance. */
  captureGap?: boolean;
  archiveUri?: string;
  historyUri?: string;
}

export interface TakeoverConfig {
  takeoverEnabled?: boolean;
  takeoverTokenThreshold?: number;
  takeoverKeepRecentTurns?: number;
  takeoverOverviewBudget?: number;
  takeoverOverviewPollMs?: number;
  takeoverOverviewPollMax?: number;
}

export interface SyncBranchResult {
  added: number;
  tokens: number;
  allDelivered: boolean;
  queued?: number;
  permanentFailures?: number;
}

export interface TakeoverIo {
  /** Deliver a pi branch to the server before committing. */
  syncBranch?: (branch: any[]) => Promise<SyncBranchResult> | SyncBranchResult;
  flush?: () => Promise<boolean> | boolean;
  commit?: (opts?: { queueOnFailure?: boolean; keepRecentCount?: number }) => Promise<unknown> | unknown;
  /** Read one archive's `.overview.md` by its uri; null until it is ready. */
  readArchiveOverview?: (archiveUri: string) => Promise<string | null> | string | null;
  /** Exact server keep_recent_count for a retained tail (message count). */
  captureCount?: (branchSlice: any[]) => number;
  persistEntry?: (customType: string, data: TakeoverPersistedState) => void;
  getWatermark?: () => number;
  /** Messages OpenViking will never receive; a capture gap when > 0. */
  droppedCount?: () => number;
  availableTools?: () => string[];
  sleep?: (ms: number) => Promise<void>;
  log?: (message: string) => void;
}

export interface CommitOutcome {
  accepted: boolean;
  reason: string;
  archiveUri?: string;
  taskId?: string;
}

export function flattenContent(msg: TakeoverMessage): string;
export function fingerprintMessage(msg: TakeoverMessage): string;
export function isUserTurnStart(msg: TakeoverMessage): boolean;
export function countUserTurns(messages: TakeoverMessage[]): number;
export function findBoundaryIndex(messages: TakeoverMessage[], coveredUserTurns: number): number;
/** pi's context projection of `getBranch()`: compaction-aware, context edits applied. */
export function projectContextEntries(branch: any[]): any[];
export function estimateTokens(text: string): number;
export function truncateToTokens(text: string, budget: number): string;
export function estimatePayloadTokens(payload: any): number;
export function buildOverviewMessage(overview: string, firstKeptTs?: number, budget?: number, recoveryHint?: string): TakeoverMessage;
export function countUndeliveredForSession(pendingEntries: any[], sid: string): number;
export function deriveHistoryUri(archiveUri: string): string;
export function commitOutcome(committed: unknown): CommitOutcome;

export type TakeoverState = TakeoverPersistedState & {
  coveredThroughEntryId: string;
  lastSeenUserTurns: number;
  syncedEntryCount: number;
  committing: boolean;
  pendingArchive: PendingArchive | null;
  captureGap: boolean;
  archiveUri: string;
  historyUri: string;
};

export class TakeoverCore {
  constructor(opts?: { config?: TakeoverConfig; io?: TakeoverIo });
  get enabled(): boolean;
  get state(): TakeoverState;
  restore(entries: any[]): TakeoverState;
  /** `branch` is pi's `getBranch()`; the boundary is located on its context projection. */
  transformContext(messages: TakeoverMessage[], branch?: any[] | (() => any[])): TakeoverMessage[];
  onTurnSynced(estTokens: number, branch?: any[] | (() => any[])): Promise<boolean>;
  commitAndAdvance(branch?: any[] | (() => any[])): Promise<boolean>;
  handleBeforeCompact(
    preparation?: { firstKeptEntryId?: string; tokensBefore?: number; signal?: AbortSignal },
    branch?: any[] | (() => any[]),
  ): Promise<
    | {
        compaction: {
          summary: string;
          firstKeptEntryId: string;
          tokensBefore: number;
          details: { source: string };
        };
      }
    | undefined
  >;
  shutdown(): Promise<void>;
  resetBoundary(reason?: string): void;
  truncatedOverview(): string;
  recoveryHint(archiveUri?: string, historyUri?: string): string;
  recordCaptureGap(): void;
  persistedState(): TakeoverPersistedState;
  persist(): void;
  pollArchiveOverview(archiveUri: string, signal?: AbortSignal): Promise<string>;
}
