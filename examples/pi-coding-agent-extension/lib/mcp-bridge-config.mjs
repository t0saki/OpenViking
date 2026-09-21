/**
 * The MCP proxy config for pi's loaded extension config.
 *
 * The in-process bridge drives the same `createOpenVikingMcpProxy` the other
 * harnesses run as a child process, so it consumes the same config shape and
 * `toMcpProxyConfig` stays the single place that maps a loader's result onto
 * it: mcpUrl winning over baseUrl, credentials, identity headers, user agent,
 * timeout, credential source and the watched credential files all come across
 * already resolved the way pi resolved them for its REST client.
 *
 * Only two of that mapper's options need a decision here. The rest are left at
 * their defaults on purpose:
 *
 * - `debugLogPath` defaults to `cfg.debugLogPath`, which is the path pi already
 *   resolved (including its older `OV_DEBUG_LOG` spelling, folded in by
 *   `loadConfig`). That is the same file the extension's own logger appends to,
 *   which is what we want and not a duplicate: every line carries the name of
 *   the component that wrote it, so proxy records interleave with the
 *   extension's instead of repeating them. Restating the path here would only
 *   add a second copy to keep in sync, and it would still be the same file
 *   whichever logger factory `index.ts` injects.
 * - `env` defaults to `process.env`. The bridge runs inside the pi process, so
 *   that is literally the environment `loadConfig()` just read; the stdio
 *   entrypoints thread an explicit `env` only because they are spawned.
 *
 * Request headers are never built here — the proxy core constructs all of them
 * from this config.
 */

import { toMcpProxyConfig } from "../shared/mcp-proxy-config.mjs";

/**
 * @param cfg The `OVConfig` that the extension's `loadConfig()` returned.
 */
export function buildBridgeProxyConfig(cfg) {
  return toMcpProxyConfig(cfg, {
    // Sent explicitly, to keep the tools' retrieval view identical to recall's.
    // The default (`resolveMcpActorPeerId`) yields a peer only under `actor`
    // recall scope, because a long-lived stdio proxy cannot trust its launch
    // directory for an identity. The bridge is not that: it lives in the pi
    // process and takes the peer pi derived for this session's cwd — the same
    // value the REST client resolves for every request it makes. That peer
    // reaches the server as the actor-peer header, which narrows the default
    // targets of `find`/`search` to the user's own memories, resources and
    // skills plus this peer's subtrees; with no peer the default widens to the
    // whole user root, other peers' subtrees included (see
    // `default_target_directories` in openviking/core/retrieval_targets.py).
    // Omitting it would therefore let the tools quietly search further than
    // recall does. Writes are unaffected: `remember`/`write` place content by
    // their own arguments.
    peerId: cfg.peerId,
    // Widened to match the extension's own logger, which opens as soon as a
    // debug log path exists, while the proxy core logs only when its config
    // says `debug`. Without this, an operator who set just the log path would
    // get the extension's lines and none of the bridge's — losing exactly the
    // records worth having there: the handshake, 401/403 responses and
    // timeouts.
    debug: Boolean(cfg.debug || cfg.debugLogPath),
  });
}
