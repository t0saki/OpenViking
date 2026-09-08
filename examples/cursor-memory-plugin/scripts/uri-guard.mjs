#!/usr/bin/env node

import {
  denyCursorPermission,
  evaluateUriGuard,
  runUriGuardHook,
} from "../../memory-plugin-shared/lib/uri-guard.mjs";

export function evaluateCursorUriGuard(input = {}) {
  const isShell = typeof input.command === "string";
  const decision = evaluateUriGuard(isShell ? "bash" : "read", input);
  if (!decision) return {};
  return denyCursorPermission(decision.reason, { agentMessage: isShell });
}

runUriGuardHook(import.meta.url, evaluateCursorUriGuard);
