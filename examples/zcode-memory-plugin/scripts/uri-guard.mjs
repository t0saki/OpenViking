#!/usr/bin/env node

// ZCode validates hook output against a strict schema and rejects any key it
// does not recognize, so the deny envelope stays exactly the shared one.

import {
  denyHookSpecificOutput,
  evaluateUriGuard,
  runUriGuardHook,
} from "../../memory-plugin-shared/lib/uri-guard.mjs";

export function evaluateZcodeUriGuard(input = {}) {
  const toolName = input.tool_name ?? input.toolName ?? input.name ?? input.tool;
  const toolInput = input.tool_input ?? input.toolInput ?? input.input ?? {};
  const decision = evaluateUriGuard(toolName, toolInput);
  return decision ? denyHookSpecificOutput(decision.reason) : {};
}

runUriGuardHook(import.meta.url, evaluateZcodeUriGuard);
