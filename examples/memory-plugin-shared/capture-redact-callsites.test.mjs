/**
 * Every harness hands its resolved `captureRedact` to the shared send layer.
 *
 * The send layer redacts whatever it is given, so a call site that forgets the
 * tier still redacts — at "standard". That is safe, but it quietly ignores a
 * user who chose "strict" (or "off"), and nothing at runtime would notice. This
 * reads each plugin's own sources and fails on any call into the send layer
 * that names neither a tier nor an already-redacted payload.
 */

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));

// The plugins' own code: never the vendored shared copies, tests or packages.
const SOURCE_ROOTS = [
  "examples/claude-code-memory-plugin/scripts",
  "examples/codex-memory-plugin/scripts",
  "examples/agent-hook-plugin/hosts",
  "examples/agent-hook-plugin/scripts",
  "examples/opencode-plugin",
  "examples/pi-coding-agent-extension",
  "examples/dsh-memory-plugin",
];
const SKIP_DIRS = new Set(["shared", "node_modules", "tests", "test", "skills", "servers", "dist"]);

// A call into the send layer whose argument list must say which tier applies.
const CALL_RE = /(?<![.\w])(sendSessionMessages|addAgentMessages|addAgentMessage|replayPending|replayAgentPending)\s*\(|(?<![.\w])(enqueue|enqueuePendingDirectly|enqueueAgentPending)\s*\(\s*["']addMessage["']/g;
const TIER_RE = /\bredactTier\b|\bredacted\s*:\s*true\b/;

function sourceFiles(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      if (!SKIP_DIRS.has(name)) sourceFiles(path, out);
    } else if (/\.(mjs|ts)$/.test(name) && !/\.test\.(mjs|ts)$/.test(name) && !name.endsWith(".d.mts")) {
      out.push(path);
    }
  }
  return out;
}

/** The argument text of the call whose `(` is at `open`. */
function argumentsAt(text, open) {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    const c = text[i];
    if (c === "(") depth++;
    else if (c === ")" && --depth === 0) return text.slice(open + 1, i);
  }
  return text.slice(open + 1);
}

test("every send-layer call in a plugin passes the configured captureRedact tier", () => {
  const missing = [];
  let calls = 0;
  for (const root of SOURCE_ROOTS) {
    for (const file of sourceFiles(join(ROOT, root))) {
      // Comments blanked out in place, so offsets and line numbers still hold.
      const text = readFileSync(file, "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, (c) => c.replace(/[^\n]/g, " "))
        .replace(/(^|[ \t])\/\/.*$/gm, (c) => c.replace(/./g, " "));
      for (const match of text.matchAll(CALL_RE)) {
        const head = text.slice(Math.max(0, match.index - 16), match.index);
        // Definitions and imports are not calls.
        if (/(function|async)\s+$/.test(head)) continue;
        const open = match.index + match[0].indexOf("(");
        const args = argumentsAt(text, open);
        calls++;
        if (!TIER_RE.test(args)) {
          const line = text.slice(0, match.index).split("\n").length;
          missing.push(`${relative(ROOT, file)}:${line} ${match[0].trim()}`);
        }
      }
    }
  }
  assert.ok(calls >= 15, `found only ${calls} send-layer calls; the scan has gone blind`);
  assert.deepEqual(missing, [], "these calls fall back to the default tier instead of the user's");
});
