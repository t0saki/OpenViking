import { describe, expect, it } from "vitest";

import {
  buildSetupProbeHeaders,
  cleanOpenVikingRequestHeaders,
  resolveOpenVikingRequestHeaders,
} from "../../request-headers.js";

describe("OpenViking request headers", () => {
  it("preserves string-valued headers exactly", () => {
    expect(cleanOpenVikingRequestHeaders({
      openviking: " i18n-instance ",
      empty: "",
    })).toEqual({
      openviking: " i18n-instance ",
      empty: "",
    });
  });

  it("resolves configured OpenViking routing headers", () => {
    expect(resolveOpenVikingRequestHeaders({
      headers: {
        openviking: "i18n_bi_claw_1781078526__bi_claw_openviking",
        region: "SG",
      },
    })).toEqual({
      openviking: "i18n_bi_claw_1781078526__bi_claw_openviking",
      region: "SG",
    });
  });

  it("does not synthesize auth headers", () => {
    expect(resolveOpenVikingRequestHeaders()).toEqual({});
  });

  it("keeps explicitly configured auth-like headers when provided", () => {
    expect(resolveOpenVikingRequestHeaders({
      headers: {
        token: "explicit-token",
        "X-API-Key": "explicit-key",
      },
    })).toEqual({
      token: "explicit-token",
      "X-API-Key": "explicit-key",
    });
  });

  it("builds setup probe headers with the key, the tenant placeholders and the overrides", () => {
    expect(buildSetupProbeHeaders()).toEqual({});
    expect(buildSetupProbeHeaders({ apiKey: "sk-probe" })).toEqual({
      "X-API-Key": "sk-probe",
    });
    expect(buildSetupProbeHeaders({ apiKey: "sk-probe", tenantProbe: true })).toEqual({
      "X-API-Key": "sk-probe",
      "X-OpenViking-Account": "__probe__",
      "X-OpenViking-User": "__probe__",
    });
    expect(buildSetupProbeHeaders({
      apiKey: "sk-probe",
      tenantProbe: true,
      configuredHeaders: { "X-API-Key": "sk-from-config", openviking: "i18n-instance" },
    })).toEqual({
      "X-API-Key": "sk-from-config",
      "X-OpenViking-Account": "__probe__",
      "X-OpenViking-User": "__probe__",
      openviking: "i18n-instance",
    });
  });

  it("rejects non-string header values", () => {
    expect(() => resolveOpenVikingRequestHeaders({
      headers: { openviking: 123 },
    })).toThrow("openviking request header openviking must be a string");
  });
});
