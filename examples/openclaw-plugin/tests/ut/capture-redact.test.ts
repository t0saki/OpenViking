import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { OpenVikingClient } from "../../client.js";
import type { HttpTransport } from "../../adapters/http-transport.js";
import { memoryOpenVikingConfigSchema } from "../../config.js";
import { createOpenVikingClientRuntime } from "../../plugin/openviking-client-runtime.js";

// Assembled at runtime so this file holds no contiguous token for a scanner.
const GITHUB_PAT = ["ghp", "_", "a1B2c3D4e5F6g7H8i9J0", "k1L2m3N4o5P6q7R8"].join("");
const GENERIC = ["Zq8", "xW2", "pL5", "mN7", "rT4", "vB9", "kH3", "jF6"].join("");

function okResponse(): Response {
  return new Response(JSON.stringify({ status: "ok", result: { session_id: "s-1" } }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function makeClient(captureRedact?: string) {
  const transport = vi.fn<HttpTransport>().mockImplementation(async () => okResponse());
  const client = new OpenVikingClient("http://127.0.0.1:1933", "", "agent", 5_000, "", "", undefined, {
    transport,
    captureRedact,
  });
  const bodies = () => transport.mock.calls.map(([, init]) => String(init.body ?? ""));
  return { client, bodies };
}

describe("captureRedact", () => {
  const originalEnv = { ...process.env };
  let stateDir = "";

  beforeEach(() => {
    stateDir = mkdtempSync(join(tmpdir(), "openclaw-redact-"));
    // The install key is kept under the state dir; keep it out of the real home.
    process.env.OPENVIKING_STATE_DIR = stateDir;
    delete process.env.OPENVIKING_CAPTURE_REDACT;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    rmSync(stateDir, { recursive: true, force: true });
  });

  it("parses to standard by default, false to off, and unknown tiers to standard", () => {
    expect(memoryOpenVikingConfigSchema.parse({}).captureRedact).toBe("standard");
    expect(memoryOpenVikingConfigSchema.parse({ captureRedact: "strict" }).captureRedact).toBe("strict");
    expect(memoryOpenVikingConfigSchema.parse({ captureRedact: false }).captureRedact).toBe("off");
    expect(memoryOpenVikingConfigSchema.parse({ captureRedact: "off" }).captureRedact).toBe("off");
    expect(memoryOpenVikingConfigSchema.parse({ captureRedact: "paranoid" }).captureRedact).toBe("standard");
  });

  it("leaves tier names to the plugin: the manifest schema accepts any string or boolean", () => {
    const manifest = JSON.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../openclaw.plugin.json"), "utf8"));
    const schema = manifest.configSchema.properties.captureRedact;
    expect(schema.oneOf).toEqual([{ type: "string" }, { type: "boolean" }]);
    expect(schema.default).toBe("standard");
  });

  it("lets OPENVIKING_CAPTURE_REDACT outrank the plugin config", () => {
    process.env.OPENVIKING_CAPTURE_REDACT = "0";
    expect(memoryOpenVikingConfigSchema.parse({ captureRedact: "strict" }).captureRedact).toBe("off");
    process.env.OPENVIKING_CAPTURE_REDACT = "strict";
    expect(memoryOpenVikingConfigSchema.parse({}).captureRedact).toBe("strict");
    process.env.OPENVIKING_CAPTURE_REDACT = "";
    expect(memoryOpenVikingConfigSchema.parse({ captureRedact: "off" }).captureRedact).toBe("off");
  });

  it("redacts a session message before it is posted, without touching the caller's parts", async () => {
    const { client, bodies } = makeClient();
    const parts = [
      { type: "text" as const, text: `export GITHUB_TOKEN=${GITHUB_PAT}` },
      {
        type: "tool" as const,
        tool_id: "call_1",
        tool_name: "bash",
        tool_input: { command: "cat .env" },
        tool_output: `GITHUB_TOKEN=${GITHUB_PAT}\n`,
        tool_status: "completed",
      },
    ];
    await client.addSessionMessage("s-1", "assistant", parts, undefined, undefined, "peer-1");

    expect(parts[0]!.text).toContain(GITHUB_PAT);
    const sent = JSON.parse(bodies()[0]!);
    expect(bodies()[0]).not.toContain(GITHUB_PAT);
    expect(sent.parts[0].text).toMatch(/^export GITHUB_TOKEN=\{\{ov_redacted:github-pat:[0-9a-f]{8}\}\}$/);
    expect(sent.parts[1].tool_output).toMatch(/\{\{ov_redacted:github-pat:[0-9a-f]{8}\}\}/);
    expect(sent.parts[1].tool_id).toBe("call_1");
    expect(sent.peer_id).toBe("peer-1");
    expect(sent.role).toBe("assistant");
  });

  it("follows the configured tier", async () => {
    const line = [{ type: "text" as const, text: `API_TOKEN=${GENERIC}` }];
    const standard = makeClient("standard");
    await standard.client.addSessionMessage("s-1", "user", line);
    expect(standard.bodies()[0]).toContain(GENERIC);

    const strict = makeClient("strict");
    await strict.client.addSessionMessage("s-1", "user", line);
    expect(strict.bodies()[0]).not.toContain(GENERIC);

    const off = makeClient("off");
    await off.client.addSessionMessage("s-1", "user", [{ type: "text", text: GITHUB_PAT }]);
    expect(off.bodies()[0]).toContain(GITHUB_PAT);
  });

  it("is plumbed from the parsed config into the runtime's client", async () => {
    const transport = vi.fn<HttpTransport>().mockImplementation(async () => okResponse());
    const cfg = memoryOpenVikingConfigSchema.parse({ captureRedact: "off" });
    const runtime = createOpenVikingClientRuntime({
      cfg,
      rawPeerPrefix: undefined,
      logger: { info() {}, warn() {}, error() {}, debug() {} } as never,
      transport,
    });
    const client = await runtime.getClient();
    await client.addSessionMessage("s-1", "user", [{ type: "text", text: GITHUB_PAT }]);
    expect(String(transport.mock.calls[0]![1].body)).toContain(GITHUB_PAT);
  });
});
