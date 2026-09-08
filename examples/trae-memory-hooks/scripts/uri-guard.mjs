#!/usr/bin/env node

import {
  denyHookSpecificOutput,
  evaluateUriGuard,
  runUriGuardHook,
} from "../../memory-plugin-shared/lib/uri-guard.mjs";

export function evaluateTraeUriGuard(input = {}) {
  const toolName = input.tool_name ?? input.toolName ?? input.name ?? input.tool;
  const toolInput = input.tool_input ?? input.toolInput ?? input.input ?? {};
  const decision = evaluateUriGuard(toolName, toolInput);
  return decision ? denyHookSpecificOutput(decision.reason) : {};
}

runUriGuardHook(import.meta.url, evaluateTraeUriGuard);
