import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  DEFAULT_REDACT_TIER,
  REDACT_TIERS,
  compileRuleSet,
  loadInstallKey,
  normalizeRedactTier,
  redactPayload,
  redactPayloads,
  redactText,
  shannonEntropy,
} from "./lib/redact.mjs";
import SECRET_RULES, { GLOBAL_ALLOWLISTS } from "./lib/secret-rules.mjs";

// Every fake credential is assembled at runtime so this file holds no
// contiguous token for a secret scanner (or a push-protection hook) to trip on.
const KEY = Buffer.alloc(32, 7);
const opts = (extra = {}) => ({ key: KEY, ...extra });

function randomish(length, alphabet, seed = 1) {
  let x = seed >>> 0;
  let out = "";
  for (let i = 0; i < length; i++) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0;
    out += alphabet[(x >>> 8) % alphabet.length];
  }
  return out;
}

const ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const URLSAFE = `${ALNUM}_-`;
const S = {
  githubPat: ["ghp", "_", randomish(36, ALNUM, 11)].join(""),
  aws: ["AK", "IA", randomish(16, "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", 13)].join(""),
  openai: ["sk", "-proj-", randomish(74, URLSAFE, 14), "T3Blbk", "FJ", randomish(74, URLSAFE, 15)].join(""),
  anthropic: ["sk", "-ant-api03-", randomish(93, URLSAFE, 16), "AA"].join(""),
  slackBot: ["xo", "xb-", "1234567890", "-", "1234567890123", "-", randomish(24, ALNUM, 17)].join(""),
  google: ["AI", "za", randomish(35, URLSAFE, 18)].join(""),
  stripe: ["sk", "_live_", randomish(24, ALNUM, 19)].join(""),
  jwt: ["ey", "J", randomish(30, ALNUM, 20), ".ey", "J", randomish(40, ALNUM, 21), ".", randomish(43, URLSAFE, 22)].join(""),
};
const OV = {
  gateway: ["ov", "gw_", randomBytes(32).toString("base64url")].join(""),
  management: ["ov", "gm_", randomBytes(32).toString("base64url")].join(""),
  access: ["ov", "at_", randomBytes(40).toString("base64url")].join(""),
  refresh: ["ov", "rt_", randomBytes(40).toString("base64url")].join(""),
  code: ["ov", "ac_", randomBytes(40).toString("base64url")].join(""),
};
const PEM = [
  "-----BEGIN RSA ", "PRIVATE KEY-----\n",
  Array.from({ length: 6 }, (_, i) => randomish(64, `${ALNUM}+/`, 30 + i)).join("\n"),
  "\n-----END RSA ", "PRIVATE KEY-----",
].join("");

const PLACEHOLDER_RE = /\{\{ov_redacted:([a-z0-9-]+):([0-9a-f]{8})\}\}/g;
const placeholders = (text) => [...String(text).matchAll(PLACEHOLDER_RE)].map((m) => m[1]);
const tags = (text) => [...String(text).matchAll(PLACEHOLDER_RE)].map((m) => m[2]);

test("every generated rule compiles on this Node", () => {
  assert.equal(compileRuleSet().rules.length, SECRET_RULES.length);
  for (const rule of SECRET_RULES) {
    assert.ok(["standard", "strict"].includes(rule.tier), rule.id);
    new RegExp(rule.regex, `${rule.flags}gd`);
  }
  for (const al of GLOBAL_ALLOWLISTS) for (const r of al.regexes) new RegExp(r.regex, r.flags);
});

test("gitleaks entropy matches Go's definition", () => {
  assert.equal(shannonEntropy(""), 0);
  assert.equal(shannonEntropy("aaaa"), 0);
  assert.equal(shannonEntropy("ab"), 1);
  // Divided by UTF-8 bytes, as Go's len(string) is.
  assert.ok(Math.abs(shannonEntropy("é") - 0.5) < 1e-12);
});

const POSITIVES = [
  ["github-pat", (s) => `my token is ${s} ok`, S.githubPat],
  ["aws-access-token", (s) => `export AWS_ACCESS_KEY_ID=${s}`, S.aws],
  ["openai-api-key", (s) => `OPENAI_API_KEY="${s}"`, S.openai],
  ["anthropic-api-key", (s) => `ANTHROPIC_API_KEY=${s}\n`, S.anthropic],
  ["slack-bot-token", (s) => `slack: ${s}`, S.slackBot],
  ["gcp-api-key", (s) => `key=${s};`, S.google],
  ["stripe-access-token", (s) => `stripe ${s} done`, S.stripe],
  ["jwt", (s) => `Authorization: ${s}\n`, S.jwt],
  ["private-key", (s) => `$ cat id_rsa\n${s}\n$ `, PEM],
  ["openviking-gateway-key", (s) => `"key": "${s}"`, OV.gateway],
  ["openviking-gateway-management-key", (s) => `management_key = ${s}\n`, OV.management],
  ["openviking-oauth-access-token", (s) => `Authorization: Bearer ${s}`, OV.access],
  ["openviking-oauth-refresh-token", (s) => `refresh_token=${s}&x=1`, OV.refresh],
  ["openviking-oauth-auth-code", (s) => `?code=${s}`, OV.code],
];

for (const [id, wrap, secret] of POSITIVES) {
  test(`standard redacts ${id} and keeps the text around it`, () => {
    const input = wrap(secret);
    const out = redactText(input, opts());
    assert.ok(!out.includes(secret), out);
    assert.deepEqual(placeholders(out), [id]);
    const [before, after] = input.split(secret);
    assert.ok(out.startsWith(before) && out.endsWith(after), out);
  });
}

test("OpenViking rules match only the exact token", () => {
  const body = OV.gateway.slice(5);
  for (const text of [
    `x${OV.gateway}`, // glued to a longer word
    `${OV.gateway}A`, // one character too long
    `ovgw_${body.slice(1)}`, // one too short
    `ovgw_${"Ab".repeat(21)}A`, // right shape, no entropy
    `ovat_${body}`, // a 43-character body is a gateway length, not an OAuth one
  ]) {
    assert.equal(redactText(text, opts()), text, text);
  }
});

test("hashes, ids and documented example keys are left alone", () => {
  for (const text of [
    "Merge commit 4b825dc642cb6eb9a060e54bf8d69288fbee4904 into main",
    "request id 123e4567-e89b-12d3-a456-426614174000 failed",
    "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
    "token: xxxxxxxxxxxxxxxxxxxxxxxx",
    "password = ${DB_PASSWORD}",
  ]) {
    assert.equal(redactText(text, opts({ tier: "strict" })), text, text);
  }
});

test("tiers: standard leaves KEY=value alone, strict adds generic-api-key, off does nothing", () => {
  assert.deepEqual(REDACT_TIERS, ["off", "standard", "strict"]);
  assert.equal(DEFAULT_REDACT_TIER, "standard");
  const env = `DATABASE_PASSWORD=${randomish(24, ALNUM, 40)}\nPORT=5432\n`;
  assert.equal(redactText(env, opts()), env);
  assert.match(redactText(env, opts({ tier: "strict" })), /^DATABASE_PASSWORD=\{\{ov_redacted:generic-api-key:[0-9a-f]{8}\}\}\nPORT=5432\n$/);
  const all = POSITIVES.map(([, wrap, s]) => wrap(s)).join("\n");
  assert.equal(redactText(all, opts({ tier: "off" })), all);
  // Cumulative: nothing standard redacts survives strict.
  const strict = redactText(all, opts({ tier: "strict" }));
  for (const [, , secret] of POSITIVES) assert.ok(!strict.includes(secret));
});

test("normalizeRedactTier maps off-ish values to off and unknown names to the default", () => {
  for (const v of [false, 0, "false", "0", "no", "off", " Off "]) assert.equal(normalizeRedactTier(v), "off", String(v));
  for (const v of ["standard", " STRICT "]) assert.equal(normalizeRedactTier(v), v.trim().toLowerCase());
  for (const v of [true, 1, "yes", "on", undefined, null, "", "paranoid", {}]) {
    assert.equal(normalizeRedactTier(v), "standard", String(v));
  }
  assert.equal(normalizeRedactTier("bogus", "strict"), "strict");
});

test("redaction is idempotent and the tag is stable per secret and key", () => {
  for (const tier of ["standard", "strict"]) {
    for (const [, wrap, s] of POSITIVES) {
      const once = redactText(wrap(s), opts({ tier }));
      assert.equal(redactText(once, opts({ tier })), once);
    }
  }
  const a = redactText(`one ${S.githubPat}`, opts());
  assert.deepEqual(tags(redactText(`two\n${S.githubPat}\n`, opts())), tags(a));
  assert.notDeepEqual(tags(redactText(`one ${S.githubPat}`, { key: Buffer.alloc(32, 9) })), tags(a));
  assert.notDeepEqual(tags(redactText(`one ${S.githubPat.slice(0, -1)}Q`, opts())), tags(a));
});

test("ov_privacy references and ov_redacted placeholders are never touched; look-alikes are", () => {
  for (const text of [
    "Authorization: Bearer {{ov_privacy:alice:viking://~/privacy/credential/github#/values/pat}}",
    "token={{ov_redacted:generic-api-key:0123abcd}}",
    "x {{ov_privacy:ambiguous}} y {{ ov_privacy:skill:github:pat }}",
  ]) {
    assert.equal(redactText(text, opts({ tier: "strict" })), text);
  }
  for (const fake of [`{{ov_privacy:${S.githubPat}}}`, `{{ov_redacted:github-pat:${S.githubPat}}}`]) {
    const out = redactText(fake, opts());
    assert.ok(!out.includes(S.githubPat), out);
  }
});

test("a scan that throws, or a payload that cannot be serialised, fails closed", () => {
  const broken = [{ id: "broken", regex: "(", flags: "", keywords: [], entropy: 0, secretGroup: 0, allowlists: [], tier: "standard" }];
  assert.match(redactText("hello", opts({ rules: broken })), /^\{\{ov_redacted:scan-failed:[0-9a-f]{8}\}\}$/);

  const cyclic = { role: "user", content: S.githubPat };
  cyclic.self = cyclic;
  assert.match(redactPayload(cyclic, opts()), /^\{\{ov_redacted:scan-failed:[0-9a-f]{8}\}\}$/);
});

test("text past maxChars is cut at whitespace and replaced, never sent unscanned", () => {
  const input = `token ${S.githubPat} ${"word ".repeat(150)}${"tail ".repeat(400)}${S.stripe}`;
  const out = redactText(input, opts({ maxChars: 1000 }));
  assert.ok(out.startsWith("token {{ov_redacted:github-pat:"));
  assert.match(out, / \{\{ov_redacted:scan-oversize:[0-9a-f]{8}\}\}$/);
  assert.ok(!out.includes(S.stripe) && out.length < 1100);
  assert.equal(redactText(out, opts({ maxChars: 1000 })), out);

  // A PEM block whose END line is past the cut goes whole into the oversize part.
  const pem = redactText(`${"word ".repeat(100)}\n${PEM}\n`, opts({ maxChars: 800 }));
  assert.match(pem, /^(?:word ){100}\n\{\{ov_redacted:scan-oversize:[0-9a-f]{8}\}\}$/);
});

test("one call scans at most maxTotalChars across its strings; the rest is cut", () => {
  const list = [
    { role: "user", content: `${"word ".repeat(150)}` },
    { role: "user", content: `token ${S.githubPat} ${"word ".repeat(100)}` },
    { role: "user", content: S.stripe, parts: [{ type: "text", text: "late" }] },
  ];
  const out = redactPayloads(list, opts({ maxTotalChars: 1000 }));
  assert.equal(out[0].content, list[0].content);
  assert.match(out[1].content, /^token \{\{ov_redacted:github-pat:[0-9a-f]{8}\}\} .* \{\{ov_redacted:scan-oversize:[0-9a-f]{8}\}\}$/);
  assert.match(out[2].content, /^\{\{ov_redacted:scan-oversize:[0-9a-f]{8}\}\}$/);
  assert.match(out[2].parts[0].text, /^\{\{ov_redacted:scan-oversize:[0-9a-f]{8}\}\}$/);
  // Replaying the cut payloads changes nothing.
  assert.deepEqual(redactPayloads(out, opts({ maxTotalChars: 1000 })), out);
});

test("redaction repeats until stable: a placeholder can expose a token glued behind it", () => {
  const out = redactText(`${S.githubPat}${S.aws}`, opts());
  assert.deepEqual(placeholders(out), ["github-pat", "aws-access-token"]);
  assert.equal(redactText(out, opts()), out);
});

test("redactPayload redacts text-bearing fields, copies metadata, and never mutates", () => {
  const input = Object.freeze({
    role: "user",
    peer_id: "peer-1",
    content: `here ${S.githubPat}`,
    parts: Object.freeze([
      Object.freeze({
        type: "tool",
        tool_id: "call_1",
        tool_name: "Bash",
        tool_input: Object.freeze({ command: `echo ${S.stripe}`, env: { DB_PASSWORD: randomish(24, ALNUM, 50), retries: 3 } }),
        tool_output: `OPENAI_API_KEY="${S.openai}"\n`,
        tool_output_sha256: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
      }),
      Object.freeze({ type: "context", uri: "viking://user/memories/x", abstract: `note ${S.slackBot}` }),
    ]),
  });
  const out = redactPayload(input, opts());
  assert.equal(out.role, "user");
  assert.equal(out.peer_id, "peer-1");
  assert.deepEqual(placeholders(out.content), ["github-pat"]);
  const tool = out.parts[0];
  assert.equal(tool.tool_id, "call_1");
  assert.equal(tool.tool_output_sha256, input.parts[0].tool_output_sha256);
  assert.deepEqual(placeholders(tool.tool_input.command), ["stripe-access-token"]);
  assert.equal(tool.tool_input.env.retries, 3);
  assert.deepEqual(placeholders(tool.tool_output), ["openai-api-key"]);
  assert.equal(out.parts[1].uri, "viking://user/memories/x");
  assert.deepEqual(placeholders(out.parts[1].abstract), ["slack-bot-token"]);
  assert.ok(input.content.includes(S.githubPat));
  // Nested values are scanned as KEY=value, so strict catches a bare password.
  assert.equal(tool.tool_input.env.DB_PASSWORD, input.parts[0].tool_input.env.DB_PASSWORD);
  const strict = redactPayload(input, opts({ tier: "strict" }));
  assert.deepEqual(placeholders(strict.parts[0].tool_input.env.DB_PASSWORD), ["generic-api-key"]);
});

test("class instances are redacted as the data they serialise to", () => {
  class Args {
    constructor() {
      this.cmd = `echo ${S.githubPat}`;
    }
  }
  const out = redactPayload({ role: "assistant", parts: [{ type: "tool", tool_input: new Args(), tool_output: new String(S.githubPat) }] }, opts());
  assert.deepEqual(placeholders(JSON.stringify(out)), ["github-pat", "github-pat"]);
});

test("off leaves payloads as they are; redactPayloads maps a list", () => {
  const list = [{ role: "user", content: `t ${S.githubPat}` }, { role: "assistant", content: "fine" }];
  assert.equal(redactPayloads(list, opts({ tier: "off" })), list);
  assert.equal(redactPayload(list[0], opts({ tier: "off" })), list[0]);
  const out = redactPayloads(list, opts());
  assert.deepEqual(placeholders(out[0].content), ["github-pat"]);
  assert.equal(out[1].content, "fine");
});

test("the install key is created once, private, and reused; failures still redact", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-redact-key-"));
  try {
    const env = { OPENVIKING_STATE_DIR: dir };
    const key = loadInstallKey({ env });
    const file = join(dir, "redact", "install-key");
    assert.equal((await readFile(file, "utf8")).trim(), key.toString("hex"));
    if (process.platform !== "win32") {
      assert.equal((await stat(file)).mode & 0o777, 0o600);
      assert.equal((await stat(join(dir, "redact"))).mode & 0o777, 0o700);
    }
    assert.equal(redactText(`t ${S.githubPat}`, { env }), redactText(`t ${S.githubPat}`, { key }));

    const other = join(dir, "other");
    await mkdir(join(other, "redact"), { recursive: true });
    await writeFile(join(other, "redact", "install-key"), "not a key\n");
    loadInstallKey({ env: { OPENVIKING_STATE_DIR: other } });
    assert.match((await readFile(join(other, "redact", "install-key"), "utf8")).trim(), /^[0-9a-f]{64}$/);

    await writeFile(join(dir, "blocker"), "");
    const blocked = { OPENVIKING_STATE_DIR: join(dir, "blocker", "state") };
    assert.equal(loadInstallKey({ env: blocked }).length, 32);
    assert.deepEqual(placeholders(redactText(`t ${S.githubPat}`, { env: blocked })), ["github-pat"]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
