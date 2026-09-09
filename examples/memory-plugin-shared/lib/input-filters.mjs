/**
 * Ordered, operator-configured regex filters for memory plugin input.
 *
 * Rules are sed-style strings, applied in order to a single piece of text:
 *   s/^\s*ultrathink\s+//i   substitute (add `g` to replace every match)
 *   d|^\s*[/!]|              drop the text when the pattern matches
 *   k/^\?ov\b/               keep the text only when the pattern matches
 *   user:d/^\s*\/clear\b/    apply to one role only
 *
 * Nothing here throws. A malformed rule becomes an entry in `errors` and is
 * skipped, so a bad config can never take a hook down.
 */

const OPS = new Set(["s", "d", "k"]);
const SCOPES = ["user", "assistant"];
const ALLOWED_FLAGS = "imsug";
const DEFAULT_MAX_RULES = 32;
const DEFAULT_MAX_PATTERN_LENGTH = 512;
const SLOW_MS = 50;
const CACHE_LIMIT = 8;

export const INPUT_FILTER_KNOBS = [
  {
    key: "recallQueryFilters",
    env: "OPENVIKING_RECALL_QUERY_FILTERS",
    label: "recall query filters",
  },
  {
    key: "captureFilters",
    env: "OPENVIKING_CAPTURE_FILTERS",
    label: "capture filters",
  },
];

const cache = new Map();

/**
 * Read one delimited field. `\` always consumes the next character, and only
 * `\<delim>` is un-escaped, so `\d` / `\b` / `\x2c` reach RegExp verbatim.
 * Returns the index of the closing delimiter, or -1 when the field is unterminated.
 */
function scanField(source, start, delim) {
  let out = "";
  let i = start;
  while (i < source.length) {
    const c = source[i];
    if (c === "\\") {
      const next = source[i + 1];
      if (next === undefined) {
        out += "\\";
        i += 1;
        continue;
      }
      out += next === delim ? next : "\\" + next;
      i += 2;
      continue;
    }
    if (c === delim) return { value: out, end: i };
    out += c;
    i += 1;
  }
  return { value: out, end: -1 };
}

export function parseInputFilterRule(source, options = {}) {
  const maxPatternLength = options.maxPatternLength ?? DEFAULT_MAX_PATTERN_LENGTH;
  if (typeof source !== "string") {
    return { source: String(source ?? ""), error: "rule must be a string" };
  }
  const raw = source.trim();
  if (!raw) return { source: raw, error: "rule is empty" };

  let scope = "";
  let rest = raw;
  for (const candidate of SCOPES) {
    if (rest.startsWith(`${candidate}:`)) {
      scope = candidate;
      rest = rest.slice(candidate.length + 1);
      break;
    }
  }

  const op = rest[0] || "";
  if (!OPS.has(op)) {
    return {
      source: raw,
      error: `unknown operation "${op}" — expected s (substitute), d (drop) or k (keep)`,
    };
  }
  const delim = rest[1];
  if (delim === undefined) {
    return { source: raw, error: `missing delimiter after "${op}"` };
  }
  if (/[A-Za-z0-9\s\\]/.test(delim)) {
    return {
      source: raw,
      error: `invalid delimiter ${JSON.stringify(delim)} — use punctuation such as / | # or :`,
    };
  }

  const first = scanField(rest, 2, delim);
  const pattern = first.value;
  let replacement = null;
  let flagsRaw = "";
  if (op === "s") {
    if (first.end < 0) {
      return { source: raw, error: `missing ${JSON.stringify(delim)} after the pattern` };
    }
    const second = scanField(rest, first.end + 1, delim);
    if (second.end < 0) {
      return {
        source: raw,
        error: `missing ${JSON.stringify(delim)} after the replacement — write s${delim}pattern${delim}replacement${delim}`,
      };
    }
    replacement = second.value;
    flagsRaw = rest.slice(second.end + 1);
  } else {
    flagsRaw = first.end < 0 ? "" : rest.slice(first.end + 1);
  }

  if (!pattern) return { source: raw, error: "pattern is empty" };
  if (pattern.length > maxPatternLength) {
    return {
      source: raw,
      error: `pattern is too long (${pattern.length} characters, limit ${maxPatternLength})`,
    };
  }

  const flags = new Set();
  for (const flag of flagsRaw) {
    if (!ALLOWED_FLAGS.includes(flag)) {
      return { source: raw, error: `unknown flag "${flag}" — allowed flags are i, m, s, u, g` };
    }
    if (flags.has(flag)) return { source: raw, error: `duplicate flag "${flag}"` };
    flags.add(flag);
  }

  let warning;
  if (op !== "s" && flags.has("g")) {
    flags.delete("g");
    warning = `flag "g" has no effect on ${op} rules and is ignored`;
  }

  let re;
  try {
    re = new RegExp(pattern, [...flags].join(""));
  } catch (err) {
    // V8 already says "Invalid regular expression: <pattern>: <why>"; keep that
    // detail without stuttering the prefix the doctor keys off.
    const message = err?.message || String(err);
    return {
      source: raw,
      error: /^invalid regular expression/i.test(message)
        ? `invalid${message.slice("Invalid".length)}`
        : `invalid regular expression: ${message}`,
    };
  }

  const parsed = { op, scope, re, replacement, source: raw };
  if (warning) parsed.warning = warning;
  return parsed;
}

export function compileInputFilters(rules, options = {}) {
  const maxRules = options.maxRules ?? DEFAULT_MAX_RULES;
  const maxPatternLength = options.maxPatternLength ?? DEFAULT_MAX_PATTERN_LENGTH;
  const list = Array.isArray(rules) ? rules : [];

  let key = null;
  try {
    key = JSON.stringify([list, maxRules, maxPatternLength]);
  } catch {
    key = null;
  }
  if (key !== null && cache.has(key)) return cache.get(key);

  const compiled = [];
  const errors = [];
  const warnings = [];
  list.slice(0, maxRules).forEach((entry, index) => {
    if (typeof entry !== "string") {
      errors.push({ index, source: String(entry ?? ""), message: "rule must be a string" });
      return;
    }
    if (!entry.trim()) return;
    const parsed = parseInputFilterRule(entry, { maxPatternLength });
    if (parsed.error) {
      errors.push({ index, source: parsed.source, message: parsed.error });
      return;
    }
    if (parsed.warning) {
      warnings.push({ index, source: parsed.source, message: parsed.warning });
    }
    compiled.push({ ...parsed, index });
  });
  if (list.length > maxRules) {
    errors.push({
      index: maxRules,
      source: String(list[maxRules] ?? ""),
      message: `too many rules: ${list.length} configured, only the first ${maxRules} are applied`,
    });
  }

  const result = { rules: compiled, errors, warnings, count: list.length };
  if (key !== null) {
    if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value);
    cache.set(key, result);
  }
  return result;
}

/**
 * Run compiled rules over `text`. Never throws.
 *
 * `substituteOnly` skips the d/k operations, so a caller that already took one
 * drop decision for a turn can rewrite its individual pieces without taking a
 * second, possibly contradictory one.
 *
 * The time budget is advisory: every rule always runs, and a slow pass is
 * reported via `slow` / `elapsedMs` rather than silently skipping the rule that
 * may well be the redaction the operator cares about.
 */
export function applyInputFilters(text, compiled, options = {}) {
  const { role = "", substituteOnly = false, now = Date.now } = options;
  const original = typeof text === "string" ? text : String(text ?? "");
  const list = Array.isArray(compiled) ? compiled : [];
  const started = now();

  let current = original;
  let dropped = false;
  let ruleIndex = -1;
  let op = "";
  let error = "";
  try {
    for (const rule of list) {
      if (rule.scope && rule.scope !== role) continue;
      if (rule.op === "s") {
        current = current.replace(rule.re, rule.replacement);
        continue;
      }
      if (substituteOnly) continue;
      const matched = rule.re.test(current);
      if ((rule.op === "d" && matched) || (rule.op === "k" && !matched)) {
        dropped = true;
        ruleIndex = rule.index;
        op = rule.op;
        break;
      }
    }
  } catch (err) {
    error = err?.message || String(err);
    current = original;
    dropped = false;
    ruleIndex = -1;
    op = "";
  }

  const finalText = dropped ? "" : current.trim();
  const elapsedMs = Math.max(0, now() - started);
  return {
    text: finalText,
    changed: !dropped && finalText !== original,
    dropped,
    ruleIndex,
    op,
    elapsedMs,
    slow: elapsedMs > SLOW_MS,
    error,
  };
}

/**
 * Doctor-facing summary of both filter knobs on a resolved plugin config.
 */
export function describeInputFilters(cfg = {}) {
  return INPUT_FILTER_KNOBS.map(({ key, env, label }) => {
    const configured = Array.isArray(cfg?.[key]) ? cfg[key] : [];
    const compiled = compileInputFilters(configured);
    const ops = { s: 0, d: 0, k: 0 };
    for (const rule of compiled.rules) ops[rule.op] += 1;
    const bits = [];
    if (ops.s) bits.push(`${ops.s} substitute`);
    if (ops.d) bits.push(`${ops.d} drop`);
    if (ops.k) bits.push(`${ops.k} keep-only`);
    const active = compiled.rules.length;
    return {
      key,
      env,
      label,
      total: configured.length,
      active,
      ops,
      summary: `${active} rule${active === 1 ? "" : "s"}${bits.length ? ` (${bits.join(", ")})` : ""}`,
      errors: compiled.errors,
      warnings: compiled.warnings,
    };
  });
}

export function resetInputFilterCache() {
  cache.clear();
}
