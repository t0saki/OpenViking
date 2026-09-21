/**
 * The config shape `createOpenVikingMcpProxy` consumes, described structurally
 * so this declaration stays self-contained: the vendored shared modules ship
 * without declarations of their own.
 */
export interface BridgeProxyConfig {
  mcpUrl: string;
  apiKey: string;
  account: string;
  user: string;
  sendIdentityHeaders: boolean;
  peerId: string;
  userAgent: string;
  timeoutMs: number;
  debug: boolean;
  debugLogPath: string;
  credentialSource: string;
  credentialPath: string;
  watchedPaths: string[];
  extraHeaders: Record<string, string>;
}

export function buildBridgeProxyConfig(cfg: Record<string, any>): BridgeProxyConfig;
