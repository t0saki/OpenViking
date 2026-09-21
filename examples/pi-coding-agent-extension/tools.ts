/**
 * The OpenViking tool surface for pi, mirrored from the server.
 *
 * This file used to hold seven hand-written REST tools. The server's own MCP
 * `tools/list` is the catalogue now: `registerMcpTools` walks the descriptors
 * the bridge fetched during the handshake and republishes each one as a pi
 * tool, so a server that gains, drops or re-documents a tool is followed
 * automatically — nothing here enumerates tools, arguments or semantics.
 *
 * Two deliberate constraints:
 *
 *   - No value import from pi's own package. CI runs these files under
 *     `node --test`, which cannot resolve `@earendil-works/*`; type-only
 *     imports are erased and therefore fine.
 *   - No `promptSnippet` and no `promptGuidelines`. pi snapshots the system
 *     prompt before `before_agent_start`, so a snippet on a tool registered in
 *     that handler would first appear one turn later, moving the cached prompt
 *     prefix and throwing away the provider's prompt cache. `index.ts` names
 *     the registered tools in the system prompt instead, from the first turn on.
 */

import { makeArgRepair } from "./lib/mcp-arg-repair.mjs";
import { toPiParameters } from "./lib/mcp-bridge.mjs";
import type { McpBridge, McpToolDescriptor } from "./lib/mcp-bridge.mjs";

/** Every registered tool is `openviking_` + the server's own name. */
export const TOOL_NAME_PREFIX = "openviking_";

/**
 * pi's built-in tool names, `powershell` included (0.86+ registers it on every
 * platform). An extension tool silently overrides a built-in of the same name,
 * which would replace `read` or `bash` for the whole session, so a collision is
 * skipped rather than registered. With the `openviking_` prefix no upstream name
 * can reach this list today; the check is a cheap guard against a future prefix
 * change, not a case that fires.
 */
export const PI_BUILTIN_TOOL_NAMES = [
  "read",
  "bash",
  "edit",
  "write",
  "grep",
  "find",
  "ls",
  "powershell",
];

export interface RegisterMcpToolsOptions {
  /** Receives one line per skipped tool. Defaults to dropping them. */
  log?: (message: string) => void;
  /** Overrides pi's built-in names; only the guard's own test needs this. */
  builtinNames?: Iterable<string>;
  /** The names already on this host; defaults to a set kept per `pi` object. */
  registered?: Set<string>;
}

/**
 * The names registered on one host.
 *
 * pi has no `unregisterTool`, so registering a name twice would leave a
 * duplicate tool in the session. The set is keyed by the host object rather
 * than held per module so a re-entrant call (the retry branch firing while the
 * startup chain is still registering) sees the same set, while two hosts — two
 * tests, or a reloaded extension — stay independent.
 */
const REGISTERED_BY_HOST = new WeakMap<object, Set<string>>();

function hostRegistry(pi: object): Set<string> {
  let names = REGISTERED_BY_HOST.get(pi);
  if (!names) {
    names = new Set<string>();
    REGISTERED_BY_HOST.set(pi, names);
  }
  return names;
}

/** The server's own text, its title as a fallback, then a generated line. */
function describe(tool: McpToolDescriptor, upstream: string): string {
  const description = typeof tool.description === "string" ? tool.description.trim() : "";
  if (description) return description;
  const title = typeof tool.title === "string" ? tool.title.trim() : "";
  if (title) return title;
  return `OpenViking MCP tool "${upstream}". The server sent no description.`;
}

/**
 * Register one pi tool per descriptor the bridge listed.
 *
 * Returns the names registered by *this* call: a second call adds only what is
 * new, so the caller can append instead of replacing its list.
 */
export function registerMcpTools(
  pi: any,
  bridge: McpBridge,
  options: RegisterMcpToolsOptions = {},
): string[] {
  const log = typeof options.log === "function" ? options.log : () => {};
  const builtins = new Set(options.builtinNames ?? PI_BUILTIN_TOOL_NAMES);
  const registered = options.registered ?? hostRegistry(pi);
  const descriptors: McpToolDescriptor[] = Array.isArray(bridge?.state?.tools)
    ? bridge.state.tools
    : [];
  const added: string[] = [];

  for (const tool of descriptors) {
    const upstream = String(tool?.name ?? "").trim();
    if (!upstream) continue;
    const name = TOOL_NAME_PREFIX + upstream;

    if (builtins.has(name)) {
      log(`skipped ${name}: it would shadow one of pi's built-in tools`);
      continue;
    }
    if (registered.has(name)) continue;

    const schema = tool.inputSchema;
    pi.registerTool({
      name,
      label: `OpenViking ${upstream}`,
      description: describe(tool, upstream),
      // A plain JSON Schema object, not TypeBox: pi's validator coerces types
      // more widely for plain objects, and the server already emits portable
      // schemas, so passing them through keeps pi's local validation from
      // disagreeing with the server's.
      parameters: toPiParameters(schema),
      // Runs before pi validates, and only undoes the strictness pi adds on top
      // of the server's own (see lib/mcp-arg-repair.mjs).
      prepareArguments: makeArgRepair(schema),
      async execute(_toolCallId: string, params: any, signal: AbortSignal | undefined) {
        // The BARE upstream name: `openviking_` is pi's view of the tool, and
        // the server knows nothing about it.
        return await bridge.callTool(upstream, params, { signal });
      },
    });

    registered.add(name);
    added.push(name);
  }

  return added;
}
