import test from "node:test"
import assert from "node:assert/strict"
import { captureV2Context, injectV2Context, prepareV2Prompt, startV2Plugin } from "../lib/v2-plugin.mjs"

function runtimeFixture(overrides = {}) {
  const events = []
  return {
    events,
    runtime: {
      config: { mcp: { enabled: true }, recall: { enabled: true } },
      sessionManager: {
        handleEvent: async (event) => events.push(event),
        flushSession: async () => {},
        flushAll: async () => {},
      },
      repoContext: {
        getRepoSystemPrompt: () => null,
        refreshRepos: async () => {},
      },
      recall: { buildRelevantMemories: async () => undefined },
      sessionInject: { buildSessionContext: async () => undefined },
      vikingUriGuard: async () => {},
      vikingUriNotice: async () => {},
      ready: Promise.resolve(),
      ...overrides,
    },
  }
}

function contextFixture(messages = []) {
  const hooks = {}
  const registered = []
  return {
    hooks,
    registered,
    ctx: {
      location: { directory: "/tmp/project" },
      mcp: {
        transform: async (callback) => callback({
          get: () => undefined,
          set: (name, config) => registered.push([name, config]),
        }),
      },
      tool: { hook: async (name, fn) => { hooks[name] = fn } },
      session: {
        hook: async (name, fn) => { hooks[name] = fn },
        context: async () => messages,
      },
      event: { subscribe: () => (async function* empty() {})() },
    },
  }
}

test("startV2Plugin registers direct MCP tools and all v2 hooks", async () => {
  const { runtime } = runtimeFixture()
  const { ctx, hooks, registered } = contextFixture()
  const cleanup = await startV2Plugin(ctx, runtime, { pluginRoot: "/tmp/ov" })

  assert.equal(registered[0][0], "openviking")
  assert.equal(registered[0][1].type, "local")
  assert.equal(registered[0][1].codemode, false)
  assert.deepEqual(Object.keys(hooks).sort(), ["context", "execute.after", "execute.before", "prompt"])
  await cleanup()
})

test("prompt metadata is persisted and context injection is stable across model steps", async () => {
  let recallCalls = 0
  const event = {
    sessionID: "ses_1",
    messageID: "msg_1",
    prompt: { text: "find the deployment notes" },
    metadata: { caller: "test" },
  }
  await prepareV2Prompt(event, {
    directory: "/tmp/project",
    recallEnabled: true,
    sessionInject: { buildSessionContext: async () => "<profile>profile</profile>" },
    recall: {
      buildRelevantMemories: async (_input, parts) => {
        recallCalls += 1
        assert.equal(parts[0].text, "find the deployment notes")
        return "<openviking-context>memory</openviking-context>"
      },
    },
  })
  assert.equal(recallCalls, 1)
  assert.equal(event.metadata.caller, "test")

  const message = { role: "user", content: [{ type: "text", text: event.prompt.text }], metadata: event.metadata }
  const context = { system: [], messages: [message] }
  const repoContext = { getRepoSystemPrompt: () => "repo prompt" }
  injectV2Context(context, repoContext)
  injectV2Context(context, repoContext)

  assert.equal(message.content.length, 2)
  assert.match(message.content[0].text, /<profile>profile<\/profile>/)
  assert.match(message.content[0].text, /<openviking-context>memory<\/openviking-context>/)
  assert.equal(message.content[0].metadata.openviking, true)
  assert.equal(context.system.length, 1)
})

test("captureV2Context advances a per-session cursor and preserves tool parts", async () => {
  const events = []
  const messages = [
    { id: "msg_user", type: "user", text: "question" },
    {
      id: "msg_assistant",
      type: "assistant",
      content: [
        { type: "text", text: "answer" },
        {
          type: "tool", id: "call_1", name: "read",
          state: { status: "completed", input: { path: "README.md" }, content: [{ type: "text", text: "body" }] },
        },
      ],
    },
  ]
  const ctx = { session: { context: async () => messages } }
  const manager = { handleEvent: async (event) => events.push(event) }
  const cursors = new Map()

  await captureV2Context(ctx, manager, cursors, "ses_1")
  assert.equal(events.filter((event) => event.type === "message.updated").length, 2)
  assert.equal(events.at(-1).properties.part.tool, "read")
  assert.equal(cursors.get("ses_1"), "msg_assistant")

  events.length = 0
  await captureV2Context(ctx, manager, cursors, "ses_1")
  assert.deepEqual(events, [])
})

test("v2 prompt, context, after-tool, and event failures are contained", async () => {
  const { runtime } = runtimeFixture({
    recall: { buildRelevantMemories: async () => { throw new Error("recall failed") } },
    vikingUriNotice: async () => { throw new Error("notice failed") },
  })
  const { ctx, hooks } = contextFixture()
  const cleanup = await startV2Plugin(ctx, runtime, { pluginRoot: "/tmp/ov" })

  const prompt = { sessionID: "ses_1", messageID: "msg_1", prompt: { text: "query" } }
  runtime.sessionInject.buildSessionContext = async () => "<profile>still available</profile>"
  await assert.doesNotReject(() => hooks.prompt(prompt))
  assert.equal(prompt.metadata.openviking.context[0], "<profile>still available</profile>")
  await assert.doesNotReject(() => hooks.context({ sessionID: "ses_1", messages: null, system: [] }))
  await assert.doesNotReject(() => hooks["execute.after"]({
    status: "completed", tool: "shell", input: { command: "ls" }, result: { content: [] },
  }))
  await cleanup()
})
