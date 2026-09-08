import {
  compileSessionPatterns,
  matchesSessionPattern,
} from "../shared/session-model.mjs";

type BypassRuntimeConfig = {
  bypassSessionPatterns: string[];
};

type SessionBypassContext = {
  sessionId?: string;
  sessionKey?: string;
};

export function createOpenVikingBypassRuntime<TConfig extends BypassRuntimeConfig>(options: {
  cfg: TConfig;
}) {
  const bypassSessionPatterns = compileSessionPatterns(options.cfg.bypassSessionPatterns, {
    segmentSeparator: ":",
  });

  const isBypassedSession = (ctx?: SessionBypassContext): boolean =>
    matchesSessionPattern([ctx?.sessionKey, ctx?.sessionId], bypassSessionPatterns);

  return {
    isBypassedSession,
  };
}
