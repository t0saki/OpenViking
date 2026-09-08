import test from "node:test"
import assert from "node:assert/strict"
import {
  collectToolNamesByIdFromEntries,
  extractPartsFromPayload,
  sanitizeCapturedText,
  shouldCaptureText,
} from "./lib/capture-utils.mjs"

function toolPart(parts) {
  return parts.find((part) => part?.type === "tool")
}

test("toolStatus marks a camelCase isError result as error", () => {
  const parts = extractPartsFromPayload({
    role: "user",
    content: [{
      type: "tool-result",
      toolCallId: "call-1",
      content: [{ type: "text", text: "bash: df: command not found" }],
      isError: true,
    }],
  }, { toolNameById: { "call-1": "bash" } })
  assert.equal(toolPart(parts).tool_status, "error")
})

test("toolStatus keeps snake_case is_error support", () => {
  const parts = extractPartsFromPayload({
    role: "user",
    content: [{
      type: "tool_result",
      tool_use_id: "call-2",
      is_error: true,
      output: "connection refused",
    }],
  }, { toolNameById: { "call-2": "mysqladmin" } })
  assert.equal(toolPart(parts).tool_status, "error")
})

test("toolStatus recognizes a camelCase isError nested under state", () => {
  const parts = extractPartsFromPayload({
    role: "user",
    content: [{
      type: "tool-result",
      toolCallId: "call-2b",
      state: { isError: true },
      output: "connection refused",
    }],
  }, { toolNameById: { "call-2b": "mysqladmin" } })
  assert.equal(toolPart(parts).tool_status, "error")
})

test("toolStatus stays completed for a successful camelCase result", () => {
  const parts = extractPartsFromPayload({
    role: "user",
    content: [{
      type: "tool-result",
      toolCallId: "call-3",
      content: [{ type: "text", text: "ok" }],
      isError: false,
    }],
  }, { toolNameById: { "call-3": "bash" } })
  assert.equal(toolPart(parts).tool_status, "completed")
})

test("a tool result is labelled with the name of the call it answers", () => {
  const entries = [
    { role: "assistant", content: [{ type: "tool_use", id: "toolu_1", name: "Read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", output: "ok" }] },
  ]
  const parts = extractPartsFromPayload(entries[1], {
    toolNameById: collectToolNamesByIdFromEntries(entries),
  })
  assert.equal(toolPart(parts).tool_name, "Read")
})

test("a host system reminder is not part of the conversation", () => {
  assert.equal(
    sanitizeCapturedText("before\n<system-reminder>never store this</system-reminder>\nafter"),
    "before\n\nafter",
  )
})

test("a subagent context line is not part of the conversation", () => {
  assert.equal(
    sanitizeCapturedText("[Subagent Context] parent session cc-1\nthe real question"),
    "the real question",
  )
})

test("a turn that is only a system reminder never reaches the extractor", () => {
  const decision = shouldCaptureText(
    "<system-reminder>the user opened a new file</system-reminder>",
    "user",
  )
  assert.equal(decision.shouldCapture, false)
  assert.equal(decision.reason, "empty")
})

/**
 * Claude Code and Codex each wrap this module in an adapter that reads their
 * own transcript shape and exports it under the same name. `export *` beside a
 * same-named local export lets the local one win with no diagnostic, so a
 * reader of the import cannot tell which function they hold; these two are the
 * adapters, and the shared names beside them are the shared ones. Each adapter
 * re-exports from its own vendored copy of this module, so the comparison is
 * against that copy rather than against lib/.
 */
test("the Claude Code adapter is the extractCaptureTurns its callers import", async () => {
  const cc = await import("../claude-code-memory-plugin/scripts/cc-transcript.mjs")
  const vendored = await import("../claude-code-memory-plugin/scripts/shared/capture-utils.mjs")
  const cfg = { captureAssistantTurns: true, captureToolMaxChars: 1000000, captureMaxLength: 24000 }

  assert.notEqual(cc.extractCaptureTurns, vendored.extractCaptureTurns)
  assert.equal(cc.sanitizeCapturedText, vendored.sanitizeCapturedText)

  // Claude nests a tool result in a content array; only the adapter flattens it.
  const anthropic = [{
    type: "user",
    message: {
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "t1", content: [{ type: "text", text: "tool output" }] }],
    },
  }]
  assert.equal(cc.extractCaptureTurns(anthropic, cfg)[0].parts[0].tool_output, "tool output")
  assert.equal(
    vendored.extractCaptureTurns(anthropic, cfg)[0].parts[0].tool_output,
    JSON.stringify([{ type: "text", text: "tool output" }]),
  )
})
