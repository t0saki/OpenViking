/**
 * Shared OpenViking session-id helpers for memory plugin harnesses.
 */

/**
 * Glob -> RegExp. Minimal implementation: supports `*`, `**`, and literals.
 *
 * `segmentSeparator` is the character a single `*` refuses to cross: the path
 * harnesses match directories, openclaw matches colon-delimited session refs.
 */
function globToRe(glob, { segmentSeparator = "/" } = {}) {
  const separator = segmentSeparator.replace(/[\\\]^-]/g, "\\$&");
  let re = "^";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i];
    if (c === "*") {
      if (glob[i + 1] === "*") { re += ".*"; i++; }
      else re += `[^${separator}]*`;
    } else if (/[.+?^${}()|[\]\\]/.test(c)) {
      re += "\\" + c;
    } else {
      re += c;
    }
  }
  re += "$";
  return new RegExp(re);
}

export function compileSessionPatterns(patterns, { segmentSeparator = "/" } = {}) {
  return (patterns || []).map((pattern) => globToRe(pattern, { segmentSeparator }));
}

/**
 * `haystacks` is a precedence list, not a set: only the first non-empty entry
 * is matched, so a caller passing `[sessionKey, sessionId]` ignores the id
 * whenever a key is present.
 */
export function matchesSessionPattern(haystacks, patterns) {
  if (!patterns || patterns.length === 0) return false;
  const candidate = (Array.isArray(haystacks) ? haystacks : [haystacks])
    .map((value) => (typeof value === "string" ? value.trim() : ""))
    .find(Boolean);
  if (!candidate) return false;
  return patterns.some((re) => re.test(candidate));
}

export function isBypassed(cfg, { sessionId, cwd } = {}) {
  if (cfg.bypassSession) return true;
  const patterns = compileSessionPatterns(cfg.bypassSessionPatterns || []);
  return [sessionId, cwd].some((haystack) => matchesSessionPattern([haystack], patterns));
}

function safeId(value, replacement = "_") {
  return String(value || "unknown").replace(/[^A-Za-z0-9._-]/g, replacement);
}

export function deriveHarnessSessionId(prefix, sessionId, suffix = "") {
  if (!prefix || typeof prefix !== "string") {
    throw new Error("deriveHarnessSessionId requires a non-empty prefix");
  }
  if (!sessionId || typeof sessionId !== "string") {
    throw new Error("deriveHarnessSessionId requires a non-empty sessionId");
  }
  const base = `${prefix}${sessionId}`;
  if (!suffix) return base;
  const normalized = String(suffix).replace(/:/g, "-").replace(/[^A-Za-z0-9._-]/g, "-");
  return `${base}__${normalized}`;
}

export function deriveCodexSessionId(codexSessionId) {
  return `cx-${safeId(codexSessionId, "_")}`;
}
