export function buildRecallBlock(
  fetchJSON: (path: string, init?: any, options?: any) => Promise<{ ok: boolean; status?: number; result?: any; error?: any }>,
  cfg: Record<string, any>,
  query: string,
  options?: {
    actorPeerId?: string;
    sessionId?: string;
    dedupKey?: string;
    historyPath?: string;
    legacyCachePath?: string;
    compress?: (rendered: string, context: { query: string; entries: any[] }) => Promise<string | null>;
    log?: (stage: string, data?: any) => void;
  },
): Promise<string | null>;

export function buildRecallEndpointBody(cfg?: Record<string, any>): Record<string, any>;
export function estimateTokens(text: string): number;
