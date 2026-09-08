export type OpenVikingRequestHeaders = Record<string, string>;

export type ResolveOpenVikingRequestHeadersOptions = {
  headers?: unknown;
};

export function resolveOpenVikingRequestHeaders(
  options: ResolveOpenVikingRequestHeadersOptions = {},
): OpenVikingRequestHeaders {
  return cleanOpenVikingRequestHeaders(options.headers);
}

export type SetupProbeHeaderOptions = {
  apiKey?: string;
  configuredHeaders?: OpenVikingRequestHeaders;
  /** Placeholder tenant identity, sent to see whether a ROOT key answers differently. */
  tenantProbe?: boolean;
};

/**
 * Headers for the setup probes against `/health` and `/api/v1/sessions`.
 *
 * Configured headers land last so an operator's own routing or auth header
 * outranks what the probe would otherwise send, exactly as the client does.
 */
export function buildSetupProbeHeaders(
  options: SetupProbeHeaderOptions = {},
): OpenVikingRequestHeaders {
  const headers: OpenVikingRequestHeaders = {};
  if (options.apiKey) {
    headers["X-API-Key"] = options.apiKey;
  }
  if (options.tenantProbe) {
    headers["X-OpenViking-Account"] = "__probe__";
    headers["X-OpenViking-User"] = "__probe__";
  }
  return Object.assign(
    headers,
    resolveOpenVikingRequestHeaders({ headers: options.configuredHeaders }),
  );
}

export function cleanOpenVikingRequestHeaders(headers: unknown): OpenVikingRequestHeaders {
  if (headers === undefined || headers === null) {
    return {};
  }
  if (typeof headers !== "object" || Array.isArray(headers)) {
    throw new Error("openviking request headers must be an object");
  }

  const out: OpenVikingRequestHeaders = {};
  for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
    if (typeof value !== "string") {
      throw new Error(`openviking request header ${key} must be a string`);
    }
    out[key] = value;
  }
  return out;
}
