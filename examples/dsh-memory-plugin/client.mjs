export class OpenVikingClient {
  constructor(config) {
    this.config = config;
    this.connected = false;
  }

  headers(options = {}) {
    const headers = { "Content-Type": "application/json" };
    if (this.config.apiKey) headers.Authorization = `Bearer ${this.config.apiKey}`;
    if (this.config.sendIdentityHeaders && this.config.account) headers["X-OpenViking-Account"] = this.config.account;
    if (this.config.sendIdentityHeaders && this.config.user) headers["X-OpenViking-User"] = this.config.user;
    const actorPeerId = options.actorPeerId ?? this.config.peerId;
    if (actorPeerId) headers["X-OpenViking-Actor-Peer"] = actorPeerId;
    if (this.config.userAgent) headers["User-Agent"] = this.config.userAgent;
    return headers;
  }

  async fetchJSON(path, init = {}, options = {}) {
    const timeoutMs = options.timeoutMs ?? this.config.requestTimeoutMs;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${this.config.endpoint}${path}`, {
        ...init,
        headers: {
          ...this.headers(options),
          ...(init.headers || {}),
        },
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      const traceId = body?.result?.trace_id
        || body?.error?.trace_id
        || body?.trace_id
        || undefined;
      if (!response.ok || body?.status === "error") {
        return {
          ok: false,
          result: null,
          status: response.status,
          error: body?.error || { message: `HTTP ${response.status}` },
          traceId,
        };
      }
      return {
        ok: true,
        result: body?.result ?? body,
        status: response.status,
        traceId,
      };
    } catch (error) {
      return {
        ok: false,
        result: null,
        status: 0,
        error: { message: error instanceof Error ? error.message : String(error) },
      };
    } finally {
      clearTimeout(timer);
    }
  }

  async healthResult() {
    const response = await this.fetchJSON("/health", {}, { timeoutMs: 5000 });
    this.connected = response.ok;
    return response;
  }

  async ensureSessionResult(sessionId, actorPeerId) {
    const response = await this.fetchJSON("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }, { actorPeerId });
    return response;
  }

  async getSession(sessionId, actorPeerId) {
    const response = await this.fetchJSON(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}`,
      {},
      { timeoutMs: 5000, actorPeerId },
    );
    return response.ok ? response.result : null;
  }

  async addMessage(sessionId, payload, actorPeerId) {
    return this.fetchJSON(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
      { actorPeerId },
    );
  }

  async commitSession(sessionId, actorPeerId, options = {}) {
    return this.fetchJSON(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/commit`,
      {
        method: "POST",
        body: JSON.stringify({ keep_recent_count: this.config.commitKeepRecentCount }),
      },
      { timeoutMs: options.timeoutMs ?? 30000, actorPeerId },
    );
  }
}
