/**
 * Kimi Code CLI adapter.
 *
 * Output contract: Kimi Code appends a hook's stdout to the conversation as it
 * stands, so this host's envelope is plain text — a JSON document would be
 * injected as context the user can read. Only UserPromptSubmit may inject at
 * all: SessionStart, SessionEnd, PreCompact and Interrupt are observation-only
 * and their output is dropped, which is why the profile block rides the first
 * prompt here instead of the session start that carries it everywhere else.
 */

import { addAgentMessages, commitAgentSession } from "../../memory-plugin-shared/lib/agent-hook-runtime.mjs";
import { denyHookSpecificOutput, evaluateUriGuard } from "../../memory-plugin-shared/lib/uri-guard.mjs";
import { applyKimicodeCaptureResult, buildKimicodeCapturePlan } from "./kimicode-capture.mjs";
import { buildKimicodeTurns, cleanKimicodeText, kimicodePromptText } from "./kimicode-turns.mjs";

export const kimicode = {
  prefix: "kc-",
  tracksPendingPrompt: true,
  capturesOnlyWhenEnabled: true,
  profileOnPrompt: true,
  // Interrupt fires instead of Stop, while the host is already tearing the turn
  // down: capture inline rather than hand the payload to a worker whose parent
  // is about to be reaped.
  detachesCapture: (event) => event !== "interrupt",
  stages: {
    "session-start": "start",
    "user-prompt-submit": "prompt",
    stop: "capture",
    "pre-compact": "capture",
    "session-end": "capture",
    interrupt: "capture",
  },
  envelope(event, block) {
    return event === "user-prompt-submit" && block ? block : null;
  },
  guard(input = {}) {
    const toolName = input.tool_name ?? input.toolName ?? input.name ?? input.tool;
    const toolInput = input.tool_input ?? input.toolInput ?? input.input ?? {};
    const decision = evaluateUriGuard(toolName, toolInput);
    return decision ? denyHookSpecificOutput(decision.reason) : {};
  },
  // The payload is snake_case, but a camelCase session id still has to reach
  // resolveNativeSessionId's direct lookup rather than the cwd fallback two
  // windows open on one directory would share.
  normalizeInput: (input) => (
    !input.session_id && input.sessionId ? { ...input, session_id: input.sessionId } : input
  ),
  prompt: (input) => cleanKimicodeText(kimicodePromptText(input)),
  async capture(ctx, state) {
    const plan = buildKimicodeCapturePlan(buildKimicodeTurns(ctx.input, state), state, ctx.cfg);
    // Nothing new is not an error: a Stop can follow a turn already captured by
    // the Interrupt or PreCompact that preceded it.
    if (plan.toSend.length === 0) return null;

    const result = await addAgentMessages(ctx.fetchJSON, ctx.sessionId, plan.payloads);
    const { captured, ...nextState } = applyKimicodeCaptureResult(state, plan, result);
    let nextCount = Number(state.capturedSinceCommit || 0) + captured;
    if (captured > 0) {
      const committed = await commitAgentSession(ctx.fetchJSON, ctx.sessionId, ctx.log);
      if (committed.ok) nextCount = 0;
    }
    return { ...nextState, capturedSinceCommit: nextCount };
  },
};
