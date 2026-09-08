import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type ProbeHealth = {
  ok: boolean;
  version: string;
  error: string;
  compatibility: "compatible" | "server_too_old" | "server_too_new" | "unknown";
  pluginVersion: string;
  compatRange: string;
};

type ProbeKeyType = {
  keyType: "user_key" | "root_key" | "no_key" | "unknown";
  needsAccountId: boolean;
  needsUserId: boolean;
  detail: string;
};

const seam = vi.hoisted(() => ({
  health: {
    ok: true,
    version: "2026.6.1",
    error: "",
    compatibility: "compatible",
    pluginVersion: "test",
    compatRange: "any",
  } as ProbeHealth,
  keyProbe: {
    keyType: "user_key",
    needsAccountId: false,
    needsUserId: false,
    detail: "ok",
  } as ProbeKeyType,
  answers: [] as string[],
  prompts: [] as string[],
  closes: 0,
}));

vi.mock("node:readline", () => ({
  createInterface: () => ({
    question(prompt: string, callback: (answer: string) => void) {
      seam.prompts.push(prompt);
      callback(seam.answers.shift() ?? "");
    },
    close() {
      seam.closes += 1;
    },
  }),
}));

vi.mock("../../services/setup/probe-service.js", () => ({
  createSetupNetworkProbes: () => ({
    checkServiceHealth: async () => seam.health,
    probeApiKeyType: async () => seam.keyProbe,
  }),
}));

const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), "openviking-setup-cli-"));
process.env.OPENCLAW_STATE_DIR = stateDir;
const configPath = path.join(stateDir, "openclaw.json");

const { registerSetupCli } = await import("../../commands/setup.js");

type CliAction = (...args: unknown[]) => void | Promise<void>;

function createFakeCli() {
  const actions = new Map<string, CliAction>();
  const options = new Map<string, string[]>();

  function builder(commandPath: string) {
    options.set(commandPath, options.get(commandPath) ?? []);
    const self = {
      description() {
        return self;
      },
      option(flags: string) {
        options.get(commandPath)!.push(flags);
        return self;
      },
      command(name: string) {
        return builder(`${commandPath} ${name}`);
      },
      action(fn: CliAction) {
        actions.set(commandPath, fn);
        return self;
      },
    };
    return self;
  }

  const registrations: { commands: string[] }[] = [];
  const api = {
    logger: { info: vi.fn() },
    registerCli(register: (args: { program: unknown }) => void, meta: { commands: string[] }) {
      registrations.push(meta);
      register({ program: { command: (name: string) => builder(name) } });
    },
  };

  registerSetupCli(api);
  return { actions, options, registrations, api };
}

function runCommand(name: string, options: Record<string, unknown> = {}): Promise<void> {
  const { actions } = createFakeCli();
  const action = actions.get(name);
  if (!action) throw new Error(`command not registered: ${name}`);
  return Promise.resolve(action(options));
}

function readConfig(): Record<string, any> {
  return JSON.parse(fs.readFileSync(configPath, "utf-8"));
}

function writeConfig(config: Record<string, unknown>): void {
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n", "utf-8");
}

function pluginConfig(): Record<string, unknown> {
  return readConfig().plugins.entries.openviking.config;
}

let logged: string[];
let logSpy: ReturnType<typeof vi.spyOn>;
let previousExitCode: number | string | undefined;
let previousLang: string | undefined;
let previousLcAll: string | undefined;

beforeEach(() => {
  seam.health = {
    ok: true,
    version: "2026.6.1",
    error: "",
    compatibility: "compatible",
    pluginVersion: "test",
    compatRange: "any",
  };
  seam.keyProbe = { keyType: "user_key", needsAccountId: false, needsUserId: false, detail: "ok" };
  seam.answers = [];
  seam.prompts = [];
  seam.closes = 0;

  fs.rmSync(configPath, { force: true });
  for (const entry of fs.readdirSync(stateDir)) {
    fs.rmSync(path.join(stateDir, entry), { force: true, recursive: true });
  }

  logged = [];
  logSpy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
    logged.push(args.map((arg) => String(arg)).join(" "));
  });

  previousExitCode = process.exitCode;
  previousLang = process.env.LANG;
  previousLcAll = process.env.LC_ALL;
  process.env.LANG = "en_US.UTF-8";
  delete process.env.LC_ALL;
});

afterEach(() => {
  logSpy.mockRestore();
  process.exitCode = previousExitCode;
  if (previousLang === undefined) delete process.env.LANG;
  else process.env.LANG = previousLang;
  if (previousLcAll === undefined) delete process.env.LC_ALL;
  else process.env.LC_ALL = previousLcAll;
});

afterAll(() => {
  fs.rmSync(stateDir, { recursive: true, force: true });
  delete process.env.OPENCLAW_STATE_DIR;
});

describe("registerSetupCli end to end", () => {
  it("registers the openviking setup and status commands", () => {
    const { actions, options, registrations } = createFakeCli();

    expect(registrations).toEqual([{ commands: ["openviking"] }]);
    expect([...actions.keys()].sort()).toEqual(["openviking setup", "openviking status"]);
    expect(options.get("openviking setup")).toContain("--base-url <url>");
    expect(options.get("openviking status")).toContain("--json");
  });

  it("skips registration when the host offers no CLI surface", () => {
    const logger = { info: vi.fn() };
    registerSetupCli({ logger });

    expect(logger.info).toHaveBeenCalledWith(
      "openviking: registerCli not available, setup command skipped",
    );
  });

  it("writes the config and activates the slot on a non-interactive run", async () => {
    await runCommand("openviking setup", {
      baseUrl: "http://127.0.0.1:1933",
      apiKey: "sk-user-key-value",
      peerRole: "person",
      peerPrefix: "main",
      recallTargetTypes: "resource,resource",
      json: true,
    });

    const result = JSON.parse(logged.join("\n"));
    expect(result).toMatchObject({
      success: true,
      action: "configured",
      config: { mode: "remote", peer_role: "sender", apiKey: "sk-u...alue" },
      slot: { activated: true, replaced: false },
    });
    expect(pluginConfig()).toEqual({
      mode: "remote",
      baseUrl: "http://127.0.0.1:1933",
      apiKey: "sk-user-key-value",
      peer_role: "sender",
      peer_prefix: "main",
      recallTargetTypes: ["resource"],
    });
    expect(readConfig().plugins.allow).toEqual(["openviking"]);
    expect(readConfig().plugins.slots).toEqual({ contextEngine: "openviking" });
    expect(process.exitCode).toBeUndefined();
  });

  it("fails a non-interactive run with an invalid peer role without writing a config", async () => {
    await runCommand("openviking setup", {
      baseUrl: "http://127.0.0.1:1933",
      peerRole: "boss",
      json: true,
    });

    const result = JSON.parse(logged.join("\n"));
    expect(result.success).toBe(false);
    expect(result.action).toBe("error");
    expect(result.error).toContain("peer_role must be");
    expect(fs.existsSync(configPath)).toBe(false);
    expect(process.exitCode).toBe(1);
  });

  it("saves the answers collected by the interactive wizard", async () => {
    seam.answers = ["http://remote:1933", "sk-interactive-key", "assistant", "main"];

    await runCommand("openviking setup", {});

    expect(seam.prompts).toEqual([
      "OpenViking server URL [http://127.0.0.1:1933]: ",
      "API Key (optional): ",
      "Memory scope (none/assistant/sender) [none]: ",
      "Peer Prefix (optional): ",
    ]);
    expect(seam.closes).toBe(1);
    expect(pluginConfig()).toEqual({
      mode: "remote",
      baseUrl: "http://remote:1933",
      apiKey: "sk-interactive-key",
      peer_role: "assistant",
      peer_prefix: "main",
    });
    expect(readConfig().plugins.slots).toEqual({ contextEngine: "openviking" });
    expect(logged.join("\n")).toContain("Connected successfully (version: 2026.6.1)");
  });

  it("asks for tenant identity when the wizard meets a root key", async () => {
    seam.keyProbe = {
      keyType: "root_key",
      needsAccountId: true,
      needsUserId: true,
      detail: "tenant context required",
    };
    seam.answers = ["http://remote:1933", "sk-root-key", "acct-1", "user-1", "none"];

    await runCommand("openviking setup", {});

    expect(seam.prompts).toContain("Account ID (required for root key): ");
    expect(seam.prompts).toContain("User ID (required for root key): ");
    expect(pluginConfig()).toMatchObject({
      accountId: "acct-1",
      userId: "user-1",
      peer_role: "none",
    });
  });

  it("keeps unrelated plugin settings when the wizard rewrites the config", async () => {
    writeConfig({
      plugins: {
        entries: {
          openviking: {
            enabled: true,
            config: {
              mode: "remote",
              baseUrl: "http://old:1933",
              recallLimit: 7,
              bypassSessionPatterns: ["agent:*"],
              legacyLeftover: "dropped",
            },
          },
        },
      },
    });
    seam.answers = ["http://remote:1933", "", "none"];

    await runCommand("openviking setup", { reconfigure: true });

    expect(pluginConfig()).toEqual({
      recallLimit: 7,
      bypassSessionPatterns: ["agent:*"],
      mode: "remote",
      baseUrl: "http://remote:1933",
      peer_role: "none",
    });
    expect(readConfig().plugins.entries.openviking.enabled).toBe(true);
  });

  it("probes an existing configuration without prompting or stealing a busy slot", async () => {
    writeConfig({
      plugins: {
        slots: { contextEngine: "mem0" },
        entries: {
          openviking: {
            config: {
              mode: "remote",
              baseUrl: "http://kept:1933",
              apiKey: "sk-existing-secret",
              peer_role: "person",
              peer_prefix: "default",
            },
          },
        },
      },
    });

    await runCommand("openviking setup", {});

    const output = logged.join("\n");
    expect(seam.prompts).toEqual([]);
    expect(output).toContain("✓ Using existing configuration");
    expect(output).toContain("apiKey:  sk-e...cret");
    expect(output).toContain("peer_role: sender");
    expect(output).not.toContain("peer_prefix");
    expect(output).toContain("Testing connectivity to http://kept:1933...");
    expect(output).toContain('Context-engine slot is owned by "mem0"');
    expect(readConfig().plugins.slots).toEqual({ contextEngine: "mem0" });
  });

  it("reports a normalized status for a configured plugin", async () => {
    writeConfig({
      plugins: {
        slots: { contextEngine: "openviking" },
        entries: {
          openviking: {
            config: {
              mode: "remote",
              baseUrl: "http://kept:1933",
              apiKey: "sk-existing-secret",
              peer_role: "person",
              peer_prefix: "default",
            },
          },
        },
      },
    });

    await runCommand("openviking status", { json: true });

    expect(JSON.parse(logged.join("\n"))).toMatchObject({
      configured: true,
      slotActive: true,
      config: {
        mode: "remote",
        baseUrl: "http://kept:1933",
        hasApiKey: true,
        peer_role: "sender",
        hasAccountId: false,
        hasUserId: false,
      },
      keyProbe: { keyType: "user_key" },
    });
    expect(JSON.parse(logged.join("\n")).config.peer_prefix).toBeUndefined();
  });

  it("reports an unconfigured plugin as not configured", async () => {
    await runCommand("openviking status", {});

    const output = logged.join("\n");
    expect(output).toContain("Status: Not configured");
    expect(output).toContain("Run `openclaw openviking setup` to configure.");
  });
});
