import test from "node:test"
import assert from "node:assert/strict"
import { createServer } from "node:http"
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { loadConfig } from "../lib/config.mjs"
import { OpenVikingPlugin } from "../index.mjs"

async function withTempDir(prefix, fn) {
  const dir = await mkdtemp(join(tmpdir(), prefix))
  try {
    return await fn(dir)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

async function withHealthServer(fn) {
  const server = createServer((request, response) => {
    response.setHeader("Content-Type", "application/json")
    if (request.url === "/health") {
      response.end(JSON.stringify({ status: "ok" }))
      return
    }
    response.statusCode = 404
    response.end(JSON.stringify({ status: "error" }))
  })
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
  try {
    const { port } = server.address()
    return await fn(`http://127.0.0.1:${port}`)
  } finally {
    await new Promise((resolve) => server.close(resolve))
  }
}

function restoreOpenVikingEnv(snapshot) {
  for (const key of Object.keys(process.env)) {
    if (key.startsWith("OPENVIKING_")) delete process.env[key]
  }
  for (const [key, value] of Object.entries(snapshot)) {
    if (key.startsWith("OPENVIKING_")) process.env[key] = value
  }
}

test("loadConfig prefers env credentials over ovcli", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-config-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const ovcli = join(dir, "ovcli.conf")
      const project = join(dir, "project")
      await mkdir(join(project, ".opencode"), { recursive: true })
      await writeFile(ovcli, JSON.stringify({
        url: "https://cli.example.com",
        api_key: "cli-key",
        account: "cli-account",
        user: "cli-user",
        actor_peer_id: "cli-peer",
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli
      process.env.OPENVIKING_URL = "https://env.example.com"
      process.env.OPENVIKING_API_KEY = "env-key"
      process.env.OPENVIKING_ACCOUNT = "env-account"
      process.env.OPENVIKING_USER = "env-user"
      process.env.OPENVIKING_PEER_ID = "env-peer"

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.endpoint, "https://env.example.com")
      assert.equal(cfg.apiKey, "env-key")
      assert.equal(cfg.account, "env-account")
      assert.equal(cfg.user, "env-user")
      assert.equal(cfg.peerId, "env-peer")
      assert.equal(cfg.effectivePeer.peerId, "env-peer")
      assert.equal(cfg.effectivePeer.source, "explicit")
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

// The plugin used to keep its own `openviking-config.json`; a knob now travels
// with the credentials, so switching profile switches behaviour too.
test("loadConfig reads its knobs from the ovcli.conf plugin section", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-plugin-section-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const ovcli = join(dir, "ovcli.conf")
      await writeFile(ovcli, JSON.stringify({
        url: "https://cli.example.com",
        api_key: "cli-key",
        plugin: {
          recallLimit: 4,
          opencode: { captureMode: "keyword", autoRecall: false },
        },
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli

      const cfg = loadConfig(dir, join(dir, "project"))
      assert.equal(cfg.recallLimit, 4, "a shared knob applies")
      assert.equal(cfg.recallLimitConfigured, true)
      assert.equal(cfg.captureMode, "keyword", "the per-harness override applies")
      assert.equal(cfg.autoRecall, false)

      // And the environment still wins over both.
      process.env.OPENVIKING_RECALL_LIMIT = "9"
      assert.equal(loadConfig(dir, join(dir, "project")).recallLimit, 9)
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("loadConfig derives workspace peer by default", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-ws-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      process.env.OPENVIKING_CREDENTIAL_SOURCE = "env"
      process.env.OPENVIKING_URL = "https://env.example.com"
      const project = join(dir, "Project A")
      await mkdir(join(project, ".git"), { recursive: true })
      await writeFile(join(project, ".git", "config"), '[remote "origin"]\n\turl = git@github.com:acme/project-a.git\n')

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.effectivePeer.peerId, "github.com-acme-project-a")
      assert.equal(cfg.effectivePeer.source, "workspace")
      assert.equal(cfg.effectivePeer.origin, "{git_remote}")

      // Outside a repository the default derives nothing, so a scratch
      // directory writes to the user-level space instead of a peer of its own.
      const scratch = join(dir, "Scratch B")
      await mkdir(scratch, { recursive: true })
      const plain = loadConfig(dir, scratch)
      assert.equal(plain.effectivePeer.peerId, "")
      assert.equal(plain.effectivePeer.origin, "unresolved")
      assert.equal(plain.effectivePeer.legacyPeerId, scratch.replace(/[^A-Za-z0-9]/g, "-"))
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("loadConfig can disable workspace peer", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-ws-off-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      process.env.OPENVIKING_CREDENTIAL_SOURCE = "env"
      process.env.OPENVIKING_URL = "https://env.example.com"
      process.env.OPENVIKING_WORKSPACE_PEER = "0"

      const cfg = loadConfig(dir, join(dir, "project"))
      assert.equal(cfg.effectivePeer.peerId, "")
      assert.equal(cfg.effectivePeer.source, "none")
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("loadConfig supports hook-only mode without registering the bundled MCP server", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-hook-only-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const project = join(dir, "project")
      const ovcli = join(dir, "ovcli.conf")
      await writeFile(ovcli, JSON.stringify({
        url: "https://cli.example.com",
        plugin: { opencode: { mcpEnabled: false } },
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.enabled, true)
      assert.equal(cfg.mcp.enabled, false)
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("OpenVikingPlugin keeps lifecycle hooks without mutating MCP config in hook-only mode", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-hook-only-runtime-", async (dir) => {
    await withHealthServer(async (endpoint) => {
      try {
        for (const key of Object.keys(process.env)) {
          if (key.startsWith("OPENVIKING_")) delete process.env[key]
        }
        const ovcli = join(dir, "ovcli.conf")
        await writeFile(ovcli, JSON.stringify({
          url: endpoint,
          plugin: {
            opencode: {
              mcpEnabled: false,
              dataDir: join(dir, "runtime"),
              repoContext: false,
              autoRecall: false,
              autoCapture: false,
            },
          },
        }))
        process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli
        process.env.OPENVIKING_URL = endpoint

        const plugin = await OpenVikingPlugin({ client: {}, directory: dir })
        assert.equal(typeof plugin.event, "function")
        assert.equal(typeof plugin["chat.message"], "function")
        assert.equal(typeof plugin.dispose, "function")

        const opencodeConfig = { mcp: { external: { type: "remote", url: "https://example.com/mcp" } } }
        await plugin.config(opencodeConfig)
        assert.deepEqual(opencodeConfig, {
          mcp: { external: { type: "remote", url: "https://example.com/mcp" } },
        })
        await plugin.dispose()
      } finally {
        restoreOpenVikingEnv(snapshot)
      }
    })
  })
})

test("loadConfig preserves an explicit zero commit keep recent count", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-keep-recent-zero-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const project = join(dir, "project")
      process.env.OPENVIKING_CREDENTIAL_SOURCE = "env"
      process.env.OPENVIKING_URL = "https://env.example.com"
      const ovcli = join(dir, "ovcli.conf")
      await writeFile(ovcli, JSON.stringify({
        url: "https://env.example.com",
        plugin: { commitKeepRecentCount: 0 },
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.commitKeepRecentCount, 0)
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("loadConfig defaults an invalid commit keep recent count", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-keep-recent-invalid-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const project = join(dir, "project")
      process.env.OPENVIKING_CREDENTIAL_SOURCE = "env"
      process.env.OPENVIKING_URL = "https://env.example.com"
      const ovcli = join(dir, "ovcli.conf")
      await writeFile(ovcli, JSON.stringify({
        url: "https://env.example.com",
        plugin: { commitKeepRecentCount: null },
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.commitKeepRecentCount, 10)
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("loadConfig falls back to config peerId when shared credentials define none (#4487)", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-peer-fallback-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const ovcli = join(dir, "ovcli.conf")
      const project = join(dir, "project")
      await mkdir(join(project, ".opencode"), { recursive: true })
      await writeFile(ovcli, JSON.stringify({
        url: "https://cli.example.com",
        api_key: "cli-key",
        account: "cli-account",
        user: "cli-user",
        plugin: {
          opencode: {
            enabled: true,
            peerId: "atomic-city",
            workspacePeer: false,
            recallPeerScope: "actor",
          },
        },
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.peerId, "atomic-city")
      assert.deepEqual(cfg.effectivePeer, { peerId: "atomic-city", source: "explicit", origin: "explicit", legacyPeerId: "" })
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("loadConfig keeps the plugin section peerId over ovcli actor_peer_id", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-peer-plugin-wins-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const ovcli = join(dir, "ovcli.conf")
      const project = join(dir, "project")
      await mkdir(join(project, ".opencode"), { recursive: true })
      await writeFile(ovcli, JSON.stringify({
        url: "https://cli.example.com",
        api_key: "cli-key",
        actor_peer_id: "cli-peer",
        plugin: { opencode: { peerId: "config-peer" } },
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.peerId, "config-peer")
      assert.deepEqual(cfg.effectivePeer, { peerId: "config-peer", source: "explicit", origin: "explicit", legacyPeerId: "" })
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})

test("loadConfig keeps env peer over the plugin section peerId when ovcli has none", async () => {
  const snapshot = { ...process.env }
  await withTempDir("ov-oc-peer-env-wins-", async (dir) => {
    try {
      for (const key of Object.keys(process.env)) {
        if (key.startsWith("OPENVIKING_")) delete process.env[key]
      }
      const ovcli = join(dir, "ovcli.conf")
      const project = join(dir, "project")
      await mkdir(join(project, ".opencode"), { recursive: true })
      await writeFile(ovcli, JSON.stringify({
        url: "https://cli.example.com",
        api_key: "cli-key",
        plugin: { opencode: { peerId: "config-peer" } },
      }))
      process.env.OPENVIKING_CLI_CONFIG_FILE = ovcli
      process.env.OPENVIKING_PEER_ID = "env-peer"

      const cfg = loadConfig(dir, project)
      assert.equal(cfg.peerId, "env-peer")
      assert.deepEqual(cfg.effectivePeer, { peerId: "env-peer", source: "explicit", origin: "explicit", legacyPeerId: "" })
    } finally {
      restoreOpenVikingEnv(snapshot)
    }
  })
})
