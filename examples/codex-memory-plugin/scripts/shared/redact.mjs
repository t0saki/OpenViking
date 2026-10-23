// GENERATED FROM examples/memory-plugin-shared/lib. DO NOT EDIT.
/**
 * Client-side secret redaction for captured transcripts.
 *
 * An agent that runs `cat .env` or prints a token puts that credential in the
 * transcript the memory plugins capture. This module replaces such secrets
 * before a payload leaves the machine — before the POST to OpenViking and
 * before the local pending queue writes it to disk — so the server, its
 * extractor and the memories it writes never see the plaintext.
 *
 * The rules are gitleaks' default rule set plus OpenViking's own key formats
 * (lib/secret-rules.mjs, generated), and detection follows gitleaks: a
 * case-insensitive keyword prefilter, the rule regex, the secret taken from
 * the rule's `secretGroup` or else the first non-empty capture group, the
 * Shannon entropy threshold, then the global and the rule's allowlists and
 * stopwords. Only the secret span is replaced, so `API_KEY=sk-...` becomes
 * `API_KEY={{ov_redacted:<rule>:<tag>}}` and the turn still reads as what
 * happened.
 *
 * The tag is the first 8 hex characters of HMAC-SHA256(install key, secret):
 * the same secret gets the same tag across turns and sessions on one machine,
 * but the tag cannot be checked against a guessed secret without the key,
 * which never leaves the machine.
 *
 * Failure policy: redaction fails closed per string. A scan that throws
 * replaces the whole string with `{{ov_redacted:scan-failed:<tag>}}`; it never
 * throws to the hook and never lets the string through unscanned. Text past
 * `maxChars` (MAX_SCAN_CHARS by default) is not scanned either: it is cut at a
 * whitespace boundary (before an unterminated `-----BEGIN` line) and replaced
 * with `{{ov_redacted:scan-oversize:<tag>}}`. The same happens once one call
 * has scanned `maxTotalChars` (MAX_SCAN_TOTAL_CHARS) across all its strings,
 * which keeps a hook's redaction time bounded however large the batch.
 * A missing or unwritable key file only costs tag stability (an in-process
 * random key is used), never redaction itself.
 *
 * Redaction is idempotent: existing `{{ov_redacted:<rule>:<tag>}}`
 * placeholders and well-formed `{{ov_privacy:...}}` references are never
 * scanned or touched, so replaying a pending entry or redacting twice changes
 * nothing.
 *
 * Redaction is opt-in (DEFAULT_REDACT_TIER is "off"). The OpenViking server
 * already extracts the credentials it finds in Session messages into the
 * user's privacy config and stores `{{ov_privacy:...}}` references, which a
 * raw read restores. A secret redacted here never reaches the server, so it
 * is not saved there and cannot be restored: turn this on only when secrets
 * must not leave the machine.
 */

import { createHmac, randomBytes } from "node:crypto";
import { linkSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import SECRET_RULES, { GLOBAL_ALLOWLISTS, TIERS } from "./secret-rules.mjs";
import { stateDir } from "./workspace-identity.mjs";

/** Ordered, cumulative: each tier runs its own rules and every earlier tier's. */
export const REDACT_TIERS = Object.freeze([...TIERS]);
/** What an unset tier means: redaction is opt-in (see above). */
export const DEFAULT_REDACT_TIER = "off";
/** What "on" (`true`, `"yes"`) or a tier name this build does not know selects. */
export const ENABLED_REDACT_TIER = "standard";

/**
 * The most characters of one string that are scanned. It matches the
 * largest tool output a plugin captures (`captureToolMaxChars`, at most
 * 1,000,000), so nothing a plugin builds by default is cut.
 */
export const MAX_SCAN_CHARS = 1_000_000;

/**
 * The most characters scanned in one redactText / redactPayload /
 * redactPayloads call, summed over every string in it. Ordinary text scans at
 * well under 1 s per MB, but keyword-dense data (hashes, base64) can take
 * 10-20 s per MB, and a hook that times out re-sends the same turns next time.
 */
export const MAX_SCAN_TOTAL_CHARS = 2_000_000;

const OFF_WORDS = new Set(["off", "false", "0", "no"]);

/**
 * Map a configured value onto a tier name. Anything that says "no" (`false`,
 * `0`, `"off"`, any case) is `off`, and an unset value is `fallback`. Any
 * other value asked for redaction: "on" and a tier name this build does not
 * know (a newer config, a typo) select ENABLED_REDACT_TIER, never `off`.
 */
export function normalizeRedactTier(value, fallback = DEFAULT_REDACT_TIER) {
  const safeFallback = REDACT_TIERS.includes(fallback) ? fallback : DEFAULT_REDACT_TIER;
  if (value === false || value === 0) return "off";
  if (value === true || value === 1) return ENABLED_REDACT_TIER;
  if (typeof value !== "string") return safeFallback;
  const word = value.trim().toLowerCase();
  if (!word) return safeFallback;
  if (OFF_WORDS.has(word)) return "off";
  if (REDACT_TIERS.includes(word)) return word;
  return ENABLED_REDACT_TIER;
}

// ---------------------------------------------------------------------------
// Install key
// ---------------------------------------------------------------------------

const KEY_HEX_RE = /^[0-9a-f]{64}$/;
let cachedKey = null;
let processKey = null;

function readKeyFile(file) {
  try {
    const text = readFileSync(file, "utf8").trim();
    return KEY_HEX_RE.test(text) ? Buffer.from(text, "hex") : null;
  } catch {
    return null;
  }
}

/**
 * Create the key file without exposing a half-written one: a private temp
 * file is hard-linked into place, which fails if a concurrent hook got there
 * first (its key is then the one read back). A garbage key file, or a
 * filesystem without hard links, gets the temp file renamed over it.
 */
function createKeyFile(file) {
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 });
  const tmp = `${file}.${process.pid}.${randomBytes(6).toString("hex")}`;
  writeFileSync(tmp, `${randomBytes(32).toString("hex")}\n`, { mode: 0o600, flag: "wx" });
  try {
    linkSync(tmp, file);
  } catch (error) {
    if (error?.code !== "EEXIST" || !readKeyFile(file)) renameSync(tmp, file);
  } finally {
    try { unlinkSync(tmp); } catch { /* renamed into place */ }
  }
  return readKeyFile(file);
}

/**
 * The per-install HMAC key: 32 random bytes in `<state dir>/redact/install-key`
 * (dir 0700, file 0600). When it cannot be read or created, a random key for
 * this process is used: tags then differ between processes, nothing else.
 */
export function loadInstallKey({ env = process.env } = {}) {
  let file = "";
  try {
    file = join(stateDir(env), "redact", "install-key");
  } catch {
    // No state dir: process key.
  }
  if (cachedKey && cachedKey.file === file) return cachedKey.key;
  let key = file ? readKeyFile(file) : null;
  if (!key && file) {
    try {
      key = createKeyFile(file);
    } catch {
      key = null;
    }
  }
  key ||= processKey ||= randomBytes(32);
  cachedKey = { file, key };
  return key;
}

function placeholder(ruleId, key, secret) {
  const tag = createHmac("sha256", key).update(String(secret), "utf8").digest("hex").slice(0, 8);
  return `{{ov_redacted:${ruleId}:${tag}}}`;
}

// ---------------------------------------------------------------------------
// Detection (gitleaks semantics)
// ---------------------------------------------------------------------------

function compileAllowlists(list) {
  return (list || []).map((al) => ({
    and: al.condition === "AND",
    target: al.regexTarget || "secret",
    regexes: (al.regexes || []).map((r) => new RegExp(r.regex, r.flags)),
    stopwords: (al.stopwords || []).map((w) => String(w).toLowerCase()),
  }));
}

const compiledCache = new Map();

/** Compile the rules once per process; throws if a rule does not compile. */
export function compileRuleSet(rules = SECRET_RULES, globalAllowlists = GLOBAL_ALLOWLISTS) {
  const hit = compiledCache.get(rules);
  if (hit && hit.globalsSource === globalAllowlists) return hit;
  const set = {
    globalsSource: globalAllowlists,
    globals: compileAllowlists(globalAllowlists),
    rules: rules.map((rule) => ({
      id: rule.id,
      tierIndex: REDACT_TIERS.indexOf(rule.tier || ENABLED_REDACT_TIER),
      keywords: (rule.keywords || []).map((k) => String(k).toLowerCase()),
      entropy: Number(rule.entropy) || 0,
      secretGroup: Number(rule.secretGroup) || 0,
      scan: new RegExp(rule.regex, `${rule.flags || ""}g`),
      groups: new RegExp(rule.regex, `${rule.flags || ""}d`),
      allowlists: compileAllowlists(rule.allowlists),
      generic: rule.id.includes("generic"),
    })),
  };
  compiledCache.set(rules, set);
  return set;
}

/**
 * Shannon entropy as gitleaks computes it: symbol frequencies over code
 * points, divided by the UTF-8 byte length (Go's `len(string)`).
 */
export function shannonEntropy(data) {
  if (!data) return 0;
  const counts = new Map();
  let bytes = 0;
  for (const ch of data) {
    counts.set(ch, (counts.get(ch) || 0) + 1);
    const cp = ch.codePointAt(0);
    bytes += cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
  }
  let entropy = 0;
  for (const count of counts.values()) entropy -= (count / bytes) * Math.log2(count / bytes);
  return entropy;
}

function isAllowed(allowlists, secret, match, line) {
  const secretLower = secret.toLowerCase();
  for (const al of allowlists) {
    const target = al.target === "match" ? match : al.target === "line" ? line() : secret;
    const checks = [];
    if (al.regexes.length) checks.push(target !== "" && al.regexes.some((re) => re.test(target)));
    if (al.stopwords.length) checks.push(al.stopwords.some((word) => secretLower.includes(word)));
    if (al.and ? checks.every(Boolean) : checks.some(Boolean)) return true;
  }
  return false;
}

/**
 * The secret spans gitleaks would report in `text`. The `gitleaks:allow`
 * marker is deliberately not honoured: in a transcript it would be a way to
 * opt a line out of redaction.
 */
function detect(text, rules, globals) {
  const findings = [];
  const lower = text.toLowerCase();
  for (const rule of rules) {
    if (rule.keywords.length && !rule.keywords.some((kw) => lower.includes(kw))) continue;
    const re = rule.scan;
    re.lastIndex = 0;
    for (let m; (m = re.exec(text)) !== null;) {
      if (m[0].length === 0) {
        re.lastIndex++;
        continue;
      }
      // gitleaks trims newlines off the match and re-runs the regex on it.
      let start = m.index;
      let end = m.index + m[0].length;
      while (start < end && text.charCodeAt(start) === 10) start++;
      while (end > start && text.charCodeAt(end - 1) === 10) end--;
      const match = text.slice(start, end);
      let span = [0, match.length];
      const sub = match ? rule.groups.exec(match) : null;
      if (sub && sub.length >= 2) {
        if (rule.secretGroup > 0) span = sub.indices[rule.secretGroup] || [0, 0];
        else span = sub.indices.slice(1).find((idx) => idx && idx[1] > idx[0]) || span;
      }
      const secret = match.slice(span[0], span[1]);
      if (!secret) continue;
      if (rule.entropy && shannonEntropy(secret) <= rule.entropy) continue;
      const line = () => {
        const from = text.lastIndexOf("\n", start - 1) + 1;
        const to = text.indexOf("\n", end);
        return text.slice(from, to < 0 ? text.length : to);
      };
      if (isAllowed(globals, secret, match, line) || isAllowed(rule.allowlists, secret, match, line)) continue;
      findings.push({ start: start + span[0], end: start + span[1], id: rule.id, secret, generic: rule.generic });
    }
  }
  return findings;
}

// ---------------------------------------------------------------------------
// Redaction
// ---------------------------------------------------------------------------

/**
 * Spans that are already placeholders and are never scanned: exactly the
 * placeholders this module writes, and OpenViking privacy references in the
 * grammar the server parses (openviking/privacy/reference.py). Anything else
 * that merely starts with `{{ov_` is scanned like any other text.
 */
const PROTECTED_RE = new RegExp(
  [
    "\\{\\{ov_redacted:[a-z0-9][a-z0-9-]{0,99}:[0-9a-f]{8}\\}\\}",
    "\\{\\{[ \\t]*ov_privacy:(?:" +
      "ambiguous" +
      "|skill:[A-Za-z0-9_.@-]{1,255}:[^\\s:{}]{1,255}" +
      "|[^\\s:{}]{1,255}:viking://~/privacy/[^/\\s{}]{1,255}/[^#\\s{}]{1,1024}#/values/[^\\s{}]{1,1024}" +
      ")[ \\t]*\\}\\}",
  ].join("|"),
  "g",
);
// One pass is not always a fixed point: a placeholder can give the text after
// it a word boundary it lacked (`ghp_<36>AKIA<16>`: aws-access-token needs
// `\bAKIA`, which only exists once the GitHub token is replaced). Passes repeat
// until nothing changes, so a replayed entry is sent byte for byte; a string
// with no finding costs one pass.
const MAX_PASSES = 3;

/** The [start, end) stretches of `text` outside existing placeholders. */
function scannableSegments(text) {
  const segments = [];
  let last = 0;
  for (const m of text.matchAll(PROTECTED_RE)) {
    if (m.index > last) segments.push([last, m.index]);
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push([last, text.length]);
  return segments;
}

/**
 * Where scanning stops in a string with more than `max` scannable characters:
 * the `max`-th one, pulled back to whitespace within 4 KiB so a token is not
 * split with its head left in the scanned part. A `-----BEGIN` block whose
 * `-----END ...-----` line lies past the cut cannot match the private-key rule,
 * so the cut moves to the start of the BEGIN line instead.
 */
function cutPoint(text, segments, max) {
  let budget = max;
  for (const [a, b] of segments) {
    if (b - a <= budget) {
      budget -= b - a;
      continue;
    }
    const hard = a + budget;
    const head = text.slice(a, hard).toLowerCase();
    const begin = head.lastIndexOf("-----begin");
    if (begin >= 0 && !/-----end[^\n]*-----/.test(head.slice(begin))) return a + head.lastIndexOf("\n", begin) + 1;
    for (let k = hard; k > Math.max(a, hard - 4096); k--) if (/\s/.test(text[k - 1])) return k;
    return hard;
  }
  return text.length;
}

/** Overlapping findings become one span, named by a provider rule over the generic one. */
function mergeFindings(findings) {
  findings.sort((x, y) => x.start - y.start || y.end - x.end || Number(x.generic) - Number(y.generic));
  const merged = [];
  for (const f of findings) {
    const last = merged[merged.length - 1];
    if (last && f.start < last.end) {
      last.end = Math.max(last.end, f.end);
      if (last.rep.generic && !f.generic) last.rep = f;
    } else {
      merged.push({ start: f.start, end: f.end, rep: f });
    }
  }
  return merged;
}

/**
 * One scan-and-replace pass over `full`; `base` is where the value starts
 * after a key prefix, and `max` how many characters of the value to scan.
 * Returns the redacted value and how many of its characters were scanned.
 */
function redactOnce(full, base, ctx, max) {
  const segments = scannableSegments(full);
  const scannable = segments.reduce((n, [a, b]) => n + (b - a), 0);
  const limit = scannable > base + max ? Math.max(base, cutPoint(full, segments, base + max)) : full.length;
  const findings = [];
  for (const [a, b] of segments) {
    if (Math.min(b, limit) <= a) continue;
    for (const f of detect(full.slice(a, Math.min(b, limit)), ctx.rules, ctx.globals)) {
      const start = Math.max(a + f.start, base);
      if (a + f.end > start) findings.push({ ...f, start, end: a + f.end });
    }
  }
  let out = "";
  let pos = base;
  for (const span of mergeFindings(findings)) {
    out += full.slice(pos, span.start) + placeholder(span.rep.id, ctx.key(), span.rep.secret);
    pos = span.end;
  }
  out += full.slice(pos, limit);
  if (limit < full.length) out += placeholder("scan-oversize", ctx.key(), full.slice(limit));
  return { text: out, scanned: Math.max(0, Math.min(scannable, base + max) - base) };
}

function makeContext(opts = {}) {
  const tier = normalizeRedactTier(opts.tier ?? DEFAULT_REDACT_TIER);
  let key = opts.key ? Buffer.from(opts.key) : null;
  const ctx = {
    tier,
    rules: [],
    globals: [],
    maxChars: Number.isFinite(opts.maxChars) && opts.maxChars > 0 ? opts.maxChars : MAX_SCAN_CHARS,
    // What is left of the call's scan budget; every string's first pass draws on it.
    budget: Number.isFinite(opts.maxTotalChars) && opts.maxTotalChars > 0 ? opts.maxTotalChars : MAX_SCAN_TOTAL_CHARS,
    key: () => (key ||= loadInstallKey(opts.env ? { env: opts.env } : undefined)),
  };
  if (tier === "off") return ctx;
  try {
    const set = compileRuleSet(opts.rules || SECRET_RULES, opts.globalAllowlists || GLOBAL_ALLOWLISTS);
    const index = REDACT_TIERS.indexOf(tier);
    ctx.rules = set.rules.filter((rule) => rule.tierIndex > 0 && rule.tierIndex <= index);
    ctx.globals = set.globals;
  } catch (error) {
    ctx.broken = error;
  }
  return ctx;
}

function failClosed(text, ctx) {
  try {
    return placeholder("scan-failed", ctx.key(), text);
  } catch {
    return "{{ov_redacted:scan-failed:00000000}}";
  }
}

function redactStringIn(text, ctx, keyName = null) {
  if (typeof text !== "string" || text === "") return text;
  if (ctx.broken) return failClosed(text, ctx);
  const prefix = keyName ? `${keyName}=` : "";
  // Fixed for every pass, so a later pass never cuts what the first one scanned.
  const max = Math.min(ctx.maxChars, ctx.budget);
  try {
    let current = text;
    for (let pass = 0; pass < MAX_PASSES; pass++) {
      const { text: next, scanned } = redactOnce(prefix + current, prefix.length, ctx, max);
      if (pass === 0) ctx.budget -= scanned;
      if (next === current) return current;
      current = next;
    }
    // Still changing after several passes: not a state to send.
    return failClosed(text, ctx);
  } catch {
    return failClosed(text, ctx);
  }
}

/**
 * Redact the secrets in one string.
 *
 * @param {string} text
 * @param {object} [opts]
 * @param {string} [opts.tier="standard"] - one of REDACT_TIERS; "off" returns the input
 * @param {string} [opts.keyName] - a field name to scan the value under, as `keyName=value`
 * @param {number} [opts.maxChars] - scannable characters of one string before the rest is cut
 * @param {number} [opts.maxTotalChars] - the same, summed over every string in the call
 * @returns {string}
 */
export function redactText(text, opts = {}) {
  const ctx = makeContext(opts);
  if (ctx.tier === "off") return text;
  return redactStringIn(text, ctx, opts.keyName || null);
}

// Message and part fields that identify or describe rather than carry text.
const META_KEYS = new Set([
  "role", "type", "id", "uri", "peer_id", "session_id", "created_at", "tool_id", "tool_name", "tool_status",
  "context_type", "tool_output_mime_type", "tool_output_externalized_reason",
]);
const META_SUFFIX_RE = /_(?:id|uri|ref|sha256|at)$/;
const KEY_CONTEXT_RE = /^[A-Za-z_][\w.-]{0,63}$/;

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

/**
 * Redact every string under `value`. Inside an object each value is scanned
 * as `<key>=<value>`, so a rule that needs a key-ish word (generic-api-key)
 * sees `{ "API_TOKEN": "..." }` as it would see `API_TOKEN=...` in a file;
 * only the value is replaced. Keys are scanned as text of their own.
 */
function redactDeep(value, ctx, keyName) {
  if (typeof value === "string") return redactStringIn(value, ctx, keyName);
  if (Array.isArray(value)) return value.map((item) => redactDeep(item, ctx, keyName));
  if (!isObject(value)) return value;
  const out = {};
  for (const [k, v] of Object.entries(value)) {
    const key = redactStringIn(k, ctx, null);
    out[key] = redactDeep(v, ctx, key === k && KEY_CONTEXT_RE.test(k) ? k : null);
  }
  return out;
}

/** A message or part: its fields, metadata copied and the rest redacted. */
function redactFields(value, ctx) {
  if (!isObject(value)) return redactDeep(value, ctx, null);
  const out = {};
  for (const [k, v] of Object.entries(value)) {
    if (k === "parts" && Array.isArray(v)) out[k] = v.map((part) => redactFields(part, ctx));
    else if (META_KEYS.has(k) || META_SUFFIX_RE.test(k)) out[k] = v;
    else out[k] = redactDeep(v, ctx, null);
  }
  return out;
}

/**
 * Redact one session add-message payload — `{ role, content }` or
 * `{ role, parts: [...] }` — and return a redacted copy. Text-bearing fields
 * are scanned (`content`, `parts[].text`, `parts[].tool_output`, every leaf of
 * `parts[].tool_input`, `abstract`, any other non-metadata field); ids, URIs,
 * roles, statuses and timestamps are copied. The caller's object is never
 * mutated.
 */
export function redactPayload(payload, opts = {}) {
  const ctx = makeContext(opts);
  if (ctx.tier === "off") return payload;
  return redactPayloadIn(payload, ctx);
}

function redactPayloadIn(payload, ctx) {
  if (payload == null) return payload;
  if (typeof payload === "string") return redactStringIn(payload, ctx, null);
  if (typeof payload !== "object") return payload;
  // Walk the data the payload is on the wire (class instances, `toJSON`), as a
  // copy; a payload that cannot be serialised could not be sent either.
  let data;
  try {
    data = JSON.parse(JSON.stringify(payload));
  } catch {
    return failClosed("", ctx);
  }
  return redactFields(data, ctx);
}

/** redactPayload over a list, with one scan budget (maxTotalChars) for the whole list. */
export function redactPayloads(payloads, opts = {}) {
  if (!Array.isArray(payloads)) return payloads;
  const ctx = makeContext(opts);
  if (ctx.tier === "off") return payloads;
  return payloads.map((payload) => redactPayloadIn(payload, ctx));
}
