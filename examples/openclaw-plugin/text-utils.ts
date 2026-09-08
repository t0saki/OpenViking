import type { CaptureMode } from "./client.js";
import { sanitizeCapturedText, shouldCaptureText } from "./shared/capture-utils.mjs";

const CONVERSATION_METADATA_BLOCK_RE =
  /(?:^|\n)\s*(?:Conversation info|Conversation metadata|会话信息|对话信息)\s*(?:\([^)]+\))?\s*:\s*```[\s\S]*?```/gi;
/** Strips "Sender (untrusted metadata): ```json ... ```" so capture sends clean text to OpenViking extract. */
const SENDER_METADATA_BLOCK_RE = /Sender\s*\([^)]*\)\s*:\s*```[\s\S]*?```/gi;
const LEADING_TIMESTAMP_PREFIX_RE = /^\s*(?!\[\[)\[(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s+)?(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{2,4})(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{1,2}(?::\d{2})?)?(?:\s*[A-Z]{1,5}(?:[+-]\d{1,2})?)?)?\s*\]\s*/i;
const COMPACTED_SYSTEM_MSG_RE = /^System:\s*\[.*?\]\s*Compacted\s*(.+)$/i;
const SUBAGENT_CONTEXT_RE = /^\s*\[Subagent Context\]/i;
const TOOL_PLACEHOLDER_RE = /^\s*\[tool(?::\s*|Use:\s*)[^\]]+\]\s*$/i;

/** The reason string each shared verdict is reported as, for traces and diag lines. */
const CAPTURE_REASON_ALIASES: Record<string, string> = {
  slash_command: "command_text",
  punctuation: "non_content_text",
  too_short: "length_out_of_range",
  question_only: "question_text",
};

/**
 * Unwrap the envelopes only OpenClaw puts around a turn.
 *
 * The Compactor rewrites a turn as "System: [ts] Compacted ... [ts] <text>",
 * the channel stamps a local time on every message, a subagent prefixes its
 * own tag, and older builds left synthetic "[tool: name]" placeholders behind.
 * Everything the shared sanitizer already knows about — injected recall
 * blocks, metadata fences, NULs — is left to it.
 */
function unwrapOpenClawEnvelopes(text: string): string {
  if (TOOL_PLACEHOLDER_RE.test(text)) {
    return "";
  }
  const compacted = text.match(COMPACTED_SYSTEM_MSG_RE);
  if (compacted) {
    return compacted[1].replace(/\s+/g, " ").trim();
  }
  return text
    .replace(CONVERSATION_METADATA_BLOCK_RE, " ")
    .replace(SENDER_METADATA_BLOCK_RE, " ")
    .replace(LEADING_TIMESTAMP_PREFIX_RE, "")
    .replace(SUBAGENT_CONTEXT_RE, "");
}

export function sanitizeUserTextForCapture(text: string): string {
  const sanitized = sanitizeCapturedText(text, { preSanitize: unwrapOpenClawEnvelopes });
  // Recall prepends its block ahead of the message, so the channel timestamp
  // only reaches the front of the text once that block is gone.
  return sanitized.replace(LEADING_TIMESTAMP_PREFIX_RE, "").replace(/\s+/g, " ").trim();
}

export function getCaptureDecision(text: string, mode: CaptureMode, captureMaxLength: number): {
  shouldCapture: boolean;
  reason: string;
  normalizedText: string;
} {
  const trimmed = text.trim();
  const normalizedText = sanitizeUserTextForCapture(trimmed);
  const hadSanitization = normalizedText !== trimmed;
  if (!normalizedText) {
    return {
      shouldCapture: false,
      reason: /<relevant-memories>/i.test(trimmed) ? "injected_memory_context_only" : "empty_text",
      normalizedText: "",
    };
  }
  // The shared classifier truncates an over-long turn; OpenClaw drops it.
  if (normalizedText.length > captureMaxLength) {
    return {
      shouldCapture: false,
      reason: "length_out_of_range",
      normalizedText,
    };
  }

  const decision = shouldCaptureText(normalizedText, "user", {
    captureMaxLength,
    mode,
    // OpenClaw's own rules: unwrap its envelopes, keep acknowledgements, drop
    // bare questions.
    preSanitize: unwrapOpenClawEnvelopes,
    dropAck: false,
    dropQuestionOnly: true,
  });

  return {
    shouldCapture: decision.shouldCapture,
    reason: captureReason(decision, mode, hadSanitization),
    normalizedText,
  };
}

function captureReason(
  decision: { shouldCapture: boolean; reason: string; trigger?: RegExp },
  mode: CaptureMode,
  hadSanitization: boolean,
): string {
  const suffix = hadSanitization ? "_after_sanitize" : "";
  if (decision.shouldCapture) {
    if (mode === "keyword" && decision.trigger) {
      return `matched_trigger${suffix}:${decision.trigger.toString()}`;
    }
    return `semantic_candidate${suffix}`;
  }
  if (decision.reason === "no_trigger") {
    return `no_trigger_matched${suffix}`;
  }
  return CAPTURE_REASON_ALIASES[decision.reason] ?? decision.reason;
}

export function extractTextsFromUserMessages(messages: unknown[]): string[] {
  const texts: string[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") {
      continue;
    }
    const msgObj = msg as Record<string, unknown>;
    if (msgObj.role !== "user") {
      continue;
    }
    const content = msgObj.content;
    if (typeof content === "string") {
      texts.push(content);
      continue;
    }
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!block || typeof block !== "object") {
          continue;
        }
        const blockObj = block as Record<string, unknown>;
        if (blockObj.type === "text" && typeof blockObj.text === "string") {
          texts.push(blockObj.text);
        }
      }
    }
  }
  return texts;
}

function formatToolResultContent(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      const b = block as Record<string, unknown>;
      if (b?.type === "text" && typeof b.text === "string") {
        parts.push((b.text as string).trim());
      }
    }
    return parts.join("\n");
  }
  if (content !== undefined && content !== null) {
    try {
      return JSON.stringify(content);
    } catch {
      return String(content);
    }
  }
  return "";
}

/**
 * 提取消息中的一个 part 的文本内容，并清理时间戳等噪音
 */
function extractPartText(content: unknown): string {
  if (typeof content === "string") {
    return content.trim();
  }
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      const b = block as Record<string, unknown>;
      if (b?.type === "text" && typeof b.text === "string") {
        parts.push((b.text as string).trim());
      }
    }
    return parts.join(" ");
  }
  return "";
}

/**
 * 结构化消息类型 - 用于 afterTurn 发送到 OpenViking
 */
type ExtractedMessage = {
  role: "user" | "assistant";
  parts: Array<{
    type: "text";
    text: string;
  } | {
    type: "tool";
    toolCallId?: string;
    toolName: string;
    toolInput?: Record<string, unknown>;
    toolOutput: string;
    toolStatus: string;
  }>;
};

/**
 * 提取从 startIndex 开始的新消息，返回结构化消息。
 * - 用户输入 → type: "text"
 * - 工具结果 → type: "tool"
 * - 跳过 system 消息
 * - 清理时间戳前缀（如 [Fri 2026-04-10 17:20 GMT+8]）
 */
export function extractNewTurnMessages(
  messages: unknown[],
  startIndex: number,
): { messages: ExtractedMessage[]; newCount: number } {
  const result: ExtractedMessage[] = [];
  let count = 0;

  // First pass: collect toolUse inputs indexed by toolCallId/toolUseId
  // Scan all messages (including after startIndex) to find toolUse before each toolResult
  const toolUseInputs: Record<string, Record<string, unknown>> = {};
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i] as Record<string, unknown>;
    if (!msg || typeof msg !== "object") continue;
    const role = msg.role as string;
    if (role === "assistant") {
      const content = msg.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          const b = block as Record<string, unknown>;
          // Handle toolCall, toolUse, tool_call types
          if (b?.type === "toolCall" || b?.type === "toolUse" || b?.type === "tool_call") {
            const id = (b.id as string) || (b.toolUseId as string) || (b.toolCallId as string);
            // Try multiple field names for tool input: arguments, input, toolInput
            const input = b.arguments ?? b.input ?? b.toolInput;
            if (id && input && typeof input === "object") {
              toolUseInputs[id] = input as Record<string, unknown>;
            }
          }
        }
      }
    }
  }

  for (let i = startIndex; i < messages.length; i++) {
    const msg = messages[i] as Record<string, unknown>;
    if (!msg || typeof msg !== "object") continue;

    const role = msg.role as string;
    if (!role || role === "system") continue;

    count++;

    // toolResult -> type: "tool"
    if (role === "toolResult") {
      const toolName = typeof msg.toolName === "string" ? msg.toolName : "tool";
      const output = formatToolResultContent(msg.content) || "";
      // Try multiple field names for tool call ID
      const toolCallId = (msg.toolCallId as string) || (msg.toolUseId as string) || (msg.tool_call_id as string);
      const toolInput = toolCallId && toolUseInputs[toolCallId]
        ? toolUseInputs[toolCallId]
        : (typeof msg.toolInput === "object" && msg.toolInput !== null
          ? msg.toolInput as Record<string, unknown>
          : undefined);
      if (output) {
        result.push({
          role: "assistant",
          parts: [{
            type: "tool",
            toolCallId: toolCallId || undefined,
            toolName,
            toolInput,
            toolOutput: output,
            toolStatus: msg.isError === true ? "error" : "completed",
          }],
        });
      }
      continue;
    }

    // user/assistant -> type: "text"
    const content = msg.content;
    const text = extractPartText(content);

    if (text) {
      // 使用 sanitizeUserTextForCapture 清理所有噪音（Sender 元数据、时间戳等）
      const cleanedText = sanitizeUserTextForCapture(text);
      if (cleanedText) {
        // 保持原始 role，assistant 保持 assistant，user 保持 user
        const ovRole: "user" | "assistant" = role === "assistant" ? "assistant" : "user";
        result.push({
          role: ovRole,
          parts: [{
            type: "text",
            text: cleanedText,
          }],
        });
      }
    }
  }

  return { messages: result, newCount: count };
}

export function extractLatestUserText(messages: unknown[] | undefined): string {
  if (!messages || messages.length === 0) {
    return "";
  }
  const texts = extractTextsFromUserMessages(messages);
  for (let i = texts.length - 1; i >= 0; i -= 1) {
    const normalized = sanitizeUserTextForCapture(texts[i] ?? "");
    if (normalized) {
      return normalized;
    }
  }
  return "";
}

/**
 * Backward-compatible wrapper around extractNewTurnMessages.
 * Returns flat text strings in the legacy `[role]: text` format.
 * @deprecated Use extractNewTurnMessages for structured output.
 */
export function extractNewTurnTexts(
  messages: unknown[],
  startIndex: number,
): { texts: string[]; newCount: number } {
  const { messages: extracted, newCount } = extractNewTurnMessages(messages, startIndex);
  const texts: string[] = [];
  for (const msg of extracted) {
    for (const part of msg.parts) {
      if (part.type === "text") {
        texts.push(`[${msg.role}]: ${part.text}`);
      } else if (part.type === "tool") {
        if (part.toolInput && Object.keys(part.toolInput).length > 0) {
          texts.push(`[toolUse: ${part.toolName}] ${JSON.stringify(part.toolInput)}`);
        }
        texts.push(`[${part.toolName} result]: ${part.toolOutput}`);
      }
    }
  }
  return { texts, newCount };
}
