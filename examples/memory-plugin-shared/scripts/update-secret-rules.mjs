#!/usr/bin/env node
/**
 * Regenerate `lib/secret-rules.mjs` from a pinned gitleaks release.
 *
 *   node examples/memory-plugin-shared/scripts/update-secret-rules.mjs
 *   node examples/memory-plugin-shared/scripts/update-secret-rules.mjs --toml ./gitleaks.toml
 *   node examples/memory-plugin-shared/scripts/update-secret-rules.mjs --check [--toml ./gitleaks.toml]
 *
 * The rules are gitleaks' default rule set, used as it is: no rule is added
 * to, changed or allowlisted beyond what gitleaks ships. The only judgement
 * here is the tier (generic-api-key is `strict`, everything else `standard`)
 * and the rules for OpenViking's own key formats (OV_RULES). A gitleaks rule
 * whose Go regex has no simple JavaScript spelling, or that is too slow for a
 * backtracking engine (SLOW_RULES), is skipped and listed in the generated
 * header, never rewritten by hand. Runs at development time
 * only; nothing in `lib/` imports it, so it is never vendored.
 */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const OUTPUT_PATH = join(HERE, "..", "lib", "secret-rules.mjs");

export const GITLEAKS_TAG = "v8.30.1";
export const GITLEAKS_TOML_URL =
  `https://raw.githubusercontent.com/gitleaks/gitleaks/${GITLEAKS_TAG}/config/gitleaks.toml`;

/** Cumulative and ordered: a tier runs its own rules and every earlier tier's. */
export const TIERS = ["off", "standard", "strict"];

/** generic-api-key matches any high-entropy value after a key-like word. */
const STRICT_RULES = new Set(["generic-api-key"]);

const URLSAFE = "[A-Za-z0-9_-]";
const ovRule = (id, prefix, length, source) => ({
  id,
  regex: `(?<![A-Za-z0-9_-])(${prefix}${URLSAFE}{${length}})(?![A-Za-z0-9_-])`,
  flags: "",
  keywords: [prefix],
  entropy: 3,
  secretGroup: 1,
  allowlists: [],
  tier: "standard",
  source,
});

/**
 * Keys OpenViking itself issues: a fixed prefix plus `secrets.token_urlsafe(n)`
 * (43 characters for n=32, 54 for n=40, alphabet [A-Za-z0-9_-]).
 */
export const OV_RULES = [
  ovRule("openviking-gateway-key", "ovgw_", 43, "openviking_gateway/management.py"),
  ovRule("openviking-gateway-management-key", "ovgm_", 43, "openviking_gateway/cli.py"),
  ovRule("openviking-oauth-access-token", "ovat_", 54, "openviking/server/oauth/provider.py"),
  ovRule("openviking-oauth-refresh-token", "ovrt_", 54, "openviking/server/oauth/provider.py"),
  ovRule("openviking-oauth-auth-code", "ovac_", 54, "openviking/server/oauth/provider.py"),
];

/**
 * gitleaks rules that are pathologically slow under JavaScript's backtracking
 * engine (gitleaks runs them on RE2, which is linear), measured on 1 MB of
 * ordinary tool output (lib/redact.mjs's MAX_SCAN_CHARS). Skipped, not
 * rewritten. Every other rule stays well under a second per MB.
 */
const NESTED_PREFIX = "nested [\\w.-]{0,50}? prefixes: ~3-11 s per MB of hashes or base64url";
const CURL = "time quadratic in line length: minutes for a 1 MB line of `curl ...;` commands";
const SLOW_RULES = {
  "cisco-meraki-api-key": NESTED_PREFIX,
  "cohere-api-token": NESTED_PREFIX,
  "curl-auth-header": CURL,
  "curl-auth-user": CURL,
  "okta-access-token": NESTED_PREFIX,
  "privateai-api-token": NESTED_PREFIX,
  "sumologic-access-id": NESTED_PREFIX,
};

// ---------------------------------------------------------------------------
// TOML: the subset gitleaks.toml uses
// ---------------------------------------------------------------------------

const BASIC_ESCAPES = { b: "\b", t: "\t", n: "\n", f: "\f", r: "\r", '"': '"', "\\": "\\" };

/**
 * `[table]` / `[[array.of.tables]]` headers, `key = value` with strings
 * ('...', '''...''', "..."), numbers, booleans and (multi-line) arrays of
 * those, and `#` comments. Anything else throws, so an upstream format change
 * fails here instead of being misread.
 */
export function parseToml(text) {
  const root = {};
  let table = root;
  let i = 0;
  const fail = (msg) => {
    throw new Error(`gitleaks.toml line ${text.slice(0, i).split("\n").length}: ${msg}`);
  };
  const skip = (newlines) => {
    for (;;) {
      while (" \t\r".includes(text[i]) || (newlines && text[i] === "\n")) i++;
      if (text[i] !== "#") return;
      while (i < text.length && text[i] !== "\n") i++;
    }
  };
  const value = () => {
    skip(false);
    if (text.startsWith("'''", i)) {
      const end = text.indexOf("'''", i + 3);
      if (end < 0) fail("unterminated '''");
      const s = text.slice(i + 3, end).replace(/^\r?\n/, "");
      i = end + 3;
      return s;
    }
    if (text[i] === "'") {
      const end = text.indexOf("'", i + 1);
      if (end < 0) fail("unterminated '");
      const s = text.slice(i + 1, end);
      i = end + 1;
      return s;
    }
    if (text[i] === '"') {
      let s = "";
      for (i++; text[i] !== '"'; i++) {
        if (i >= text.length || text[i] === "\n") fail("unterminated string");
        if (text[i] !== "\\") s += text[i];
        else if (text[++i] in BASIC_ESCAPES) s += BASIC_ESCAPES[text[i]];
        else if (text[i] === "u" && /^[0-9a-fA-F]{4}$/.test(text.slice(i + 1, i + 5))) {
          s += String.fromCharCode(parseInt(text.slice(i + 1, i + 5), 16));
          i += 4;
        } else fail(`bad escape \\${text[i]}`);
      }
      i++;
      return s;
    }
    if (text[i] === "[") {
      const arr = [];
      for (i++; ;) {
        skip(true);
        if (text[i] === "]") break;
        arr.push(value());
        skip(true);
        if (text[i] === ",") i++;
        else if (text[i] !== "]") fail("expected , or ]");
      }
      i++;
      return arr;
    }
    const m = /^(?:true|false|-?\d+(?:\.\d+)?)/.exec(text.slice(i, i + 32));
    if (!m) fail("unsupported value");
    i += m[0].length;
    return m[0] === "true" ? true : m[0] === "false" ? false : Number(m[0]);
  };
  for (skip(true); i < text.length; skip(true)) {
    const header = /^(\[\[?)([A-Za-z0-9_.-]+)\]\]?[ \t]*(?=\r?\n|#|$)/.exec(text.slice(i, i + 256));
    if (header) {
      i += header[0].length;
      const path = header[2].split(".");
      let node = root;
      for (const key of path.slice(0, -1)) {
        node = node[key] ??= {};
        if (Array.isArray(node)) node = node[node.length - 1];
      }
      const leaf = path[path.length - 1];
      if (header[1] === "[[") (node[leaf] ??= []).push((table = {}));
      else table = node[leaf] ??= {};
      continue;
    }
    const kv = /^([A-Za-z0-9_-]+)[ \t]*=/.exec(text.slice(i, i + 256));
    if (!kv) fail("expected a key or a table header");
    i += kv[0].length;
    if (kv[1] in table) fail(`duplicate key ${kv[1]}`);
    table[kv[1]] = value();
    skip(false);
    if (i < text.length && text[i] !== "\n") fail("trailing characters");
  }
  return root;
}

// ---------------------------------------------------------------------------
// Go (RE2) regex -> JavaScript
// ---------------------------------------------------------------------------

export class UnsupportedRegexError extends Error {}

const POSIX_CLASSES = { alnum: "0-9A-Za-z", alpha: "A-Za-z", digit: "0-9", lower: "a-z", upper: "A-Z", xdigit: "0-9A-Fa-f" };
const isLetter = (c) => /^[A-Za-z]$/.test(c);
const swapCase = (c) => (c === c.toLowerCase() ? c.toUpperCase() : c.toLowerCase());

/** The other-case counterparts of the ASCII letters in the range lo..hi. */
function otherCase(lo, hi) {
  let out = "";
  for (const [a, z] of [["a", "z"], ["A", "Z"]]) {
    const from = lo > a ? lo : a;
    const to = hi < z ? hi : z;
    if (from <= to) out += from === to ? swapCase(from) : `${swapCase(from)}-${swapCase(to)}`;
  }
  return out;
}

/** One class member at body[i]: its source text and the character it is (null for \d, \w, \s...). */
function classAtom(body, i) {
  if (body[i] !== "\\") return { text: body[i], ch: body[i] };
  const e = body[i + 1];
  const hex = /^x([0-9a-fA-F]{2})/.exec(body.slice(i + 1, i + 4));
  if (hex) return { text: `\\${hex[0]}`, ch: String.fromCharCode(parseInt(hex[1], 16)) };
  if (e && "dDwWsS".includes(e)) return { text: `\\${e}`, ch: null };
  const named = { t: "\t", n: "\n", r: "\r", f: "\f", v: "\v" }[e];
  if (named) return { text: `\\${e}`, ch: named };
  if (e === undefined || /[A-Za-z0-9]/.test(e)) throw new UnsupportedRegexError(`\\${e ?? ""} in a class`);
  return { text: `\\${e}`, ch: e };
}

/**
 * The class starting at body[i] ("["), with the other case of every ASCII
 * letter appended when `ci` is set. Literal "-" and "]" are escaped so the
 * appended ranges cannot fuse with them.
 */
function convertClass(body, i, ci) {
  let j = i + 1;
  let out = "[";
  let extra = "";
  if (body[j] === "^") out += body[j++];
  for (let first = true; j < body.length && (body[j] !== "]" || first); first = false) {
    const posix = /^\[:([a-z]+):\]/.exec(body.slice(j, j + 12));
    if (posix) {
      if (!POSIX_CLASSES[posix[1]]) throw new UnsupportedRegexError(posix[0]);
      out += POSIX_CLASSES[posix[1]];
      j += posix[0].length;
      continue;
    }
    const lo = classAtom(body, j);
    j += lo.text.length;
    if (lo.ch !== null && body[j] === "-" && j + 1 < body.length && body[j + 1] !== "]") {
      const hi = classAtom(body, j + 1);
      if (hi.ch === null) throw new UnsupportedRegexError(`range ending in ${hi.text}`);
      j += 1 + hi.text.length;
      out += `${lo.text}-${hi.text}`;
      if (ci) extra += otherCase(lo.ch, hi.ch);
    } else {
      out += lo.ch === "-" || lo.ch === "]" ? `\\${lo.ch}` : lo.text;
      if (ci && lo.ch && isLetter(lo.ch)) extra += swapCase(lo.ch);
    }
  }
  if (body[j] !== "]") throw new UnsupportedRegexError(`unterminated class at ${i}`);
  return { text: `${out}${extra}]`, end: j };
}

/**
 * Rewrite a Go regex as a JavaScript one (no `u` flag). A pattern that is
 * case-insensitive throughout keeps a leading `(?i)` as the `i` flag; any
 * other `(?i)`, `(?i:...)` or `(?-i:...)` is spelled out by giving each ASCII
 * letter its other case. Also: `.` -> `[^\n]` (RE2's dot), `(?s:...)` dots ->
 * `[\s\S]`, `\z` -> `$`, `(?P<name>` -> `(?<name>`, and
 * `[:alnum:]`-style classes. Anything else Go-specific throws.
 */
export function convertGoRegex(src) {
  let flags = "";
  let body = src;
  if (body.startsWith("(?i)") && !body.includes("(?-i")) {
    flags = "i";
    body = body.slice(4);
  }
  let out = "";
  let ci = false;
  let dotAll = false;
  const stack = [];
  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (c === "\\") {
      const e = body[++i];
      if (e === undefined) throw new UnsupportedRegexError("trailing backslash");
      if (e === "z") out += "$";
      else if (e === "x" && /^[0-9a-fA-F]{2}/.test(body.slice(i + 1, i + 3))) {
        out += `\\x${body.slice(i + 1, i + 3)}`;
        i += 2;
      } else if (isLetter(e) && !"dDwWsSbBtnrfv".includes(e)) throw new UnsupportedRegexError(`\\${e}`);
      else if (/[0-9]/.test(e)) throw new UnsupportedRegexError(`\\${e}`);
      else out += `\\${e}`;
    } else if (c === "[") {
      const cls = convertClass(body, i, ci);
      out += cls.text;
      i = cls.end;
    } else if (c === "(") {
      stack.push({ ci, dotAll });
      const group = /^\(\?(?:([a-zA-Z-]*)([:)])|P?<([A-Za-z_]\w*)>)/.exec(body.slice(i));
      if (!group) {
        out += "(";
        continue;
      }
      i += group[0].length - 1;
      if (group[3]) {
        out += `(?<${group[3]}>`;
        continue;
      }
      const [, flagText, end] = group;
      if (!/^(?:i|-i|s|)$/.test(flagText)) throw new UnsupportedRegexError(`(?${flagText}${end}`);
      if (end === ")") {
        stack.pop(); // a bare (?i) sets the flag until the enclosing group ends
        if (flagText === "s") dotAll = true;
        else ci = flagText === "i";
        continue;
      }
      if (flagText === "s") dotAll = true;
      else if (flagText) ci = flagText === "i";
      out += "(?:";
    } else if (c === ")") {
      if (!stack.length) throw new UnsupportedRegexError("unbalanced )");
      ({ ci, dotAll } = stack.pop());
      out += ")";
    } else if (c === ".") {
      out += dotAll ? "[\\s\\S]" : "[^\\n]";
    } else if (ci && isLetter(c)) {
      out += `[${c}${swapCase(c)}]`;
    } else {
      out += c;
    }
  }
  if (stack.length) throw new UnsupportedRegexError("unbalanced (");
  const groups = new RegExp(`${out}|`, flags).exec("").length - 1;
  new RegExp(out, `${flags}gd`);
  return { source: out, flags, groups };
}

// ---------------------------------------------------------------------------
// gitleaks config -> rule records
// ---------------------------------------------------------------------------

/**
 * A gitleaks allowlist as the engine reads it. A transcript has no path or
 * commit: under AND such a check can never pass, so the allowlist is inert;
 * under OR only its regex and stopword checks remain. A regex that does not
 * convert throws, so its rule is skipped rather than run with less allowed.
 */
function convertAllowlist(raw) {
  const condition = String(raw.condition || "OR").toUpperCase() === "AND" ? "AND" : "OR";
  if (condition === "AND" && (raw.paths?.length || raw.commits?.length)) return null;
  const regexes = (raw.regexes || []).map((pattern) => {
    const { source, flags } = convertGoRegex(pattern);
    return { regex: source, flags };
  });
  const stopwords = [...new Set((raw.stopwords || []).map((w) => String(w).toLowerCase()))].sort();
  if (!regexes.length && !stopwords.length) return null;
  return { condition, regexTarget: raw.regexTarget || "secret", regexes, stopwords };
}

export function buildRules(config) {
  const rules = [];
  const skipped = [];
  const globalAllowlists = [config.allowlist, ...(config.allowlists || [])]
    .filter(Boolean)
    .map(convertAllowlist)
    .filter(Boolean);
  for (const raw of config.rules || []) {
    const id = String(raw.id || "");
    const skip = (reason) => skipped.push({ id, reason });
    if (raw.path) { skip("path-scoped rule; captured text has no file path"); continue; }
    if (SLOW_RULES[id]) { skip(SLOW_RULES[id]); continue; }
    let converted;
    let allowlists;
    try {
      converted = convertGoRegex(raw.regex);
      allowlists = (raw.allowlists || []).map(convertAllowlist).filter(Boolean);
    } catch (error) {
      skip(`a regex has no simple JavaScript form (${error.message})`);
      continue;
    }
    const secretGroup = Number(raw.secretGroup || 0);
    if (secretGroup > converted.groups) { skip(`secretGroup ${secretGroup} out of range`); continue; }
    rules.push({
      id,
      regex: converted.source,
      flags: converted.flags,
      keywords: [...new Set((raw.keywords || []).map((k) => String(k).toLowerCase()))],
      entropy: Number(raw.entropy || 0),
      secretGroup,
      allowlists,
      tier: STRICT_RULES.has(id) ? "strict" : "standard",
    });
  }
  for (const id of STRICT_RULES) if (!rules.some((r) => r.id === id)) throw new Error(`${id} was not emitted`);
  for (const { source: _source, ...rule } of OV_RULES) {
    if (rules.some((r) => r.id === rule.id)) throw new Error(`${rule.id} collides with a gitleaks rule`);
    new RegExp(rule.regex, "gd");
    rules.push(rule);
  }
  return { rules, skipped, globalAllowlists };
}

const MIT_LICENSE = `MIT License

Copyright (c) 2019 Zachary Rice

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.`;

/** The generated module. Deterministic for a given input. */
export function renderModule({ rules, skipped, globalAllowlists }, { tomlSha256 }) {
  const lines = [
    "// GENERATED by examples/memory-plugin-shared/scripts/update-secret-rules.mjs. DO NOT EDIT.",
    "//",
    `// Secret detection rules for lib/redact.mjs: the gitleaks ${GITLEAKS_TAG} default`,
    "// configuration (config/gitleaks.toml, https://github.com/gitleaks/gitleaks),",
    "// converted from Go to JavaScript regex syntax and otherwise unchanged, plus",
    "// the rules for OpenViking's own key formats.",
    `// Source file sha256: ${tomlSha256}`,
    "//",
    "// Tier `strict`: generic-api-key. Every other rule is tier `standard`.",
    "// Skipped gitleaks rules:",
    ...(skipped.length ? skipped.map(({ id, reason }) => `//   ${id}: ${reason}`) : ["//   (none)"]),
    "// OpenViking rules (not from gitleaks):",
    ...OV_RULES.map(({ id, keywords, source }) => `//   ${id}: ${keywords[0]}... issued by ${source}`),
    "//",
    "// The gitleaks rule data is used under its license:",
    "//",
    ...MIT_LICENSE.split("\n").map((line) => (line ? `// ${line}` : "//")),
    "",
    `export const GITLEAKS_VERSION = ${JSON.stringify(GITLEAKS_TAG)};`,
    "",
    `export const TIERS = ${JSON.stringify(TIERS)};`,
    "",
    "export const SKIPPED_RULES = [",
    ...skipped.map((s) => `  ${JSON.stringify(s)},`),
    "];",
    "",
    "export const GLOBAL_ALLOWLISTS = [",
    ...globalAllowlists.map((al) => `  ${JSON.stringify(al)},`),
    "];",
    "",
    "export default [",
    ...rules.map((rule) => `  ${JSON.stringify(rule)},`),
    "];",
    "",
  ];
  return lines.join("\n");
}

export async function generate({ tomlPath = "" } = {}) {
  let text;
  if (tomlPath) text = await readFile(tomlPath, "utf8");
  else {
    const res = await fetch(GITLEAKS_TOML_URL);
    if (!res.ok) throw new Error(`fetching ${GITLEAKS_TOML_URL}: HTTP ${res.status}`);
    text = await res.text();
  }
  const tomlSha256 = createHash("sha256").update(text).digest("hex");
  const built = buildRules(parseToml(text));
  return { ...built, output: renderModule(built, { tomlSha256 }) };
}

async function main(args) {
  const tomlIdx = args.indexOf("--toml");
  const { output, rules, skipped } = await generate({ tomlPath: tomlIdx >= 0 ? args[tomlIdx + 1] : "" });
  if (args.includes("--check")) {
    const current = await readFile(OUTPUT_PATH, "utf8").catch(() => "");
    if (current !== output) {
      process.stderr.write(`${OUTPUT_PATH} is stale; rerun without --check\n`);
      process.exitCode = 1;
    } else {
      process.stdout.write("secret-rules.mjs is up to date\n");
    }
    return;
  }
  await writeFile(OUTPUT_PATH, output);
  process.stdout.write(`wrote ${OUTPUT_PATH}: ${rules.length} rules, ${skipped.length} skipped (gitleaks ${GITLEAKS_TAG})\n`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`${error.stack || error}\n`);
    process.exitCode = 1;
  });
}
