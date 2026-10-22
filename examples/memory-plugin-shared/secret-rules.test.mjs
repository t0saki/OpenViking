import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import SECRET_RULES, { GITLEAKS_VERSION, SKIPPED_RULES, TIERS } from "./lib/secret-rules.mjs";
import {
  GITLEAKS_TAG,
  GITLEAKS_TOML_URL,
  OUTPUT_PATH,
  OV_RULES,
  UnsupportedRegexError,
  buildRules,
  convertGoRegex,
  parseToml,
  renderModule,
} from "./scripts/update-secret-rules.mjs";

const SCRIPT = fileURLToPath(new URL("./scripts/update-secret-rules.mjs", import.meta.url));
const re = (pattern) => {
  const { source, flags } = convertGoRegex(pattern);
  return new RegExp(`^(?:${source})$`, flags);
};

test("the generated module is gitleaks' rule set plus OV_RULES, tiered, attributed", async () => {
  const text = await readFile(OUTPUT_PATH, "utf8");
  assert.equal(GITLEAKS_VERSION, GITLEAKS_TAG);
  assert.ok(text.includes("// MIT License") && text.includes("Copyright (c) 2019 Zachary Rice"));
  assert.match(text, /\/\/ Source file sha256: [0-9a-f]{64}\n/);
  for (const { id, reason } of SKIPPED_RULES) assert.ok(text.includes(`//   ${id}: ${reason}`), id);
  assert.deepEqual(TIERS, ["off", "standard", "strict"]);
  assert.deepEqual(SECRET_RULES.filter((r) => r.tier === "strict").map((r) => r.id), ["generic-api-key"]);
  const ids = SECRET_RULES.map((r) => r.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.length > 200);
  assert.deepEqual(ids.slice(-OV_RULES.length), OV_RULES.map((r) => r.id));
  for (const rule of SECRET_RULES) {
    for (const kw of rule.keywords) assert.equal(kw, kw.toLowerCase(), rule.id);
  }
});

test("a leading (?i) becomes the i flag; other case scopes are spelled out", () => {
  assert.deepEqual(convertGoRegex("(?i)adafruit[a-z0-9_-]{32}").flags, "i");
  const mid = re("p8e-(?i)[a-z0-9]{2}");
  assert.ok(mid.test("p8e-aZ") && !mid.test("P8E-az"));
  const scoped = re("(?i:[a-z]+(?:(?-i:[Ss]umo|SUMO)))x");
  assert.ok(scoped.test("ABcSumox") && scoped.test("abSUMOx") && !scoped.test("absUMOx") && !scoped.test("abSumoX"));
  // A bare (?i) lasts until its group closes.
  const group = re("(a(?i)b)c");
  assert.ok(group.test("aBc") && !group.test("aBC") && !group.test("ABc"));
  // Case-expanded classes keep literal dashes literal.
  const cls = re("(?-i:x)(?i)[a-c=-]");
  assert.ok(cls.test("xB") && cls.test("x-") && !cls.test("x>"));
});

test("Go's dot, anchors, POSIX classes and named groups keep their meaning", () => {
  assert.ok(re("a.b").test("a\rb") && !re("a.b").test("a\nb"));
  assert.ok(re("a(?s:.)b").test("a\nb"));
  assert.ok(new RegExp(convertGoRegex("ab\\z").source).test("xab") && !new RegExp(convertGoRegex("ab\\z").source).test("ab\n"));
  assert.ok(re("[[:alnum:]]+").test("aZ9"));
  assert.equal(new RegExp(convertGoRegex("(?P<k>a)(b)").source).exec("ab").length, 3);
  assert.equal(convertGoRegex("(a)(?:b)(c)").groups, 2);
  for (const bad of ["\\pL", "(?m)^a", "\\Qa\\E", "\\Aa", "[[:word:]]", "(a"]) {
    assert.throws(() => convertGoRegex(bad), UnsupportedRegexError, bad);
  }
});

const MINI_TOML = `title = "mini"
# comment
[allowlist]
description = "global"
regexes = ['''^true$''']
stopwords = ["Example"]
paths = ['''\\.lock$''']

[[rules]]
id = "demo-token"
regex = '''(?i)\\b(demo_[a-z0-9]{8})\\b'''
entropy = 2.5
keywords = [
    "demo_",
]
[[rules.allowlists]]
condition = "AND"
paths = ['''x''']
regexes = ['''y''']
[[rules.allowlists]]
regexTarget = "line"
regexes = ['''fake''']

[[rules]]
id = "path-only"
path = '''\\.p12$'''

[[rules]]
id = "unconvertible"
regex = '''\\pL{8}'''
keywords = ["x"]

[[rules]]
id = "curl-auth-user"
regex = '''curl (\\w+)'''
keywords = ["curl"]

[[rules]]
id = "bad-allowlist"
regex = '''bad_(\\w{8})'''
[[rules.allowlists]]
regexes = ['''\\pL''']

[[rules]]
id = "generic-api-key"
regex = '''(?i)key=(\\w+)'''
secretGroup = 1
keywords = ["key"]
`;

test("the TOML subset parser reads what gitleaks.toml uses and refuses the rest", () => {
  const config = parseToml(MINI_TOML);
  assert.equal(config.title, "mini");
  assert.deepEqual(config.allowlist.stopwords, ["Example"]);
  assert.equal(config.rules.length, 6);
  assert.equal(config.rules[0].regex, "(?i)\\b(demo_[a-z0-9]{8})\\b");
  assert.equal(config.rules[0].entropy, 2.5);
  assert.deepEqual(config.rules[0].keywords, ["demo_"]);
  assert.equal(config.rules[0].allowlists.length, 2);
  assert.equal(config.rules[5].secretGroup, 1);
  assert.throws(() => parseToml("a = [1, 2"), /gitleaks\.toml line/);
  assert.throws(() => parseToml("a = 1\na = 2\n"), /duplicate key/);
  assert.throws(() => parseToml("a = {b = 1}\n"), /unsupported value/);
});

test("buildRules converts, tiers, skips with a reason, and appends OV_RULES", () => {
  const built = buildRules(parseToml(MINI_TOML));
  const demo = built.rules.find((r) => r.id === "demo-token");
  assert.equal(demo.tier, "standard");
  // An AND allowlist with a path can never pass on a transcript: dropped.
  assert.deepEqual(demo.allowlists, [{ condition: "OR", regexTarget: "line", regexes: [{ regex: "fake", flags: "" }], stopwords: [] }]);
  assert.equal(built.rules.find((r) => r.id === "generic-api-key").tier, "strict");
  // A rule whose allowlist will not convert is skipped, not run with less allowed.
  assert.deepEqual(built.skipped.map((s) => s.id), ["path-only", "unconvertible", "curl-auth-user", "bad-allowlist"]);
  assert.deepEqual(built.globalAllowlists, [
    { condition: "OR", regexTarget: "secret", regexes: [{ regex: "^true$", flags: "" }], stopwords: ["example"] },
  ]);
  assert.deepEqual(built.rules.slice(-OV_RULES.length).map((r) => r.id), OV_RULES.map((r) => r.id));
  assert.equal(renderModule(built, { tomlSha256: "x" }), renderModule(buildRules(parseToml(MINI_TOML)), { tomlSha256: "x" }));
  assert.throws(() => buildRules(parseToml(MINI_TOML.replace(/id = "generic-api-key"[\s\S]*$/, ""))), /generic-api-key/);
  assert.throws(() => buildRules(parseToml(MINI_TOML.replace("'''^true$'''", "'''\\pL'''"))), UnsupportedRegexError);
});

test("--check fails on a stale module", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-secret-rules-"));
  try {
    await writeFile(join(dir, "gitleaks.toml"), MINI_TOML);
    assert.throws(
      () => execFileSync(process.execPath, [SCRIPT, "--check", "--toml", join(dir, "gitleaks.toml")], { stdio: "pipe" }),
      /is stale/,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("--check regenerates the committed module byte for byte from the pinned gitleaks.toml", async (t) => {
  // GITLEAKS_TOML points at a local copy; otherwise the pinned file is fetched.
  let tomlPath = process.env.GITLEAKS_TOML || "";
  let dir = "";
  if (!tomlPath) {
    let text;
    try {
      const res = await fetch(GITLEAKS_TOML_URL, { signal: AbortSignal.timeout(15000) });
      text = res.ok ? await res.text() : null;
    } catch {
      text = null;
    }
    if (!text) return t.skip("gitleaks.toml unavailable offline (set GITLEAKS_TOML)");
    dir = await mkdtemp(join(tmpdir(), "ov-secret-rules-"));
    tomlPath = join(dir, "gitleaks.toml");
    await writeFile(tomlPath, text);
  }
  try {
    const sha = createHash("sha256").update(await readFile(tomlPath, "utf8")).digest("hex");
    assert.ok((await readFile(OUTPUT_PATH, "utf8")).includes(`// Source file sha256: ${sha}\n`));
    const out = execFileSync(process.execPath, [SCRIPT, "--check", "--toml", tomlPath], { encoding: "utf8" });
    assert.match(out, /up to date/);
  } finally {
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});
