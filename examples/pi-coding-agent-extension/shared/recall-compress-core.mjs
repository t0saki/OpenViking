// GENERATED FROM examples/memory-plugin-shared/lib. DO NOT EDIT.
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export const NO_RELEVANT_MEMORY = "NO_RELEVANT_MEMORY";

export function buildRecallCompressionPrompt({ query, rendered, maxBullets = 6 }) {
  return `You are a memory relevance compressor utility.
Do not use any tools. Do not investigate. Only transform the given text.

User query:
${query}

Retrieved OpenViking memory fragments:
${rendered}

Return exactly ${NO_RELEVANT_MEMORY} if nothing is relevant. Otherwise return:
OpenViking memory digest:
- at most ${maxBullets} bullets
- every bullet must include a viking:// URI
- every bullet must be at most 120 characters
Do not add any other prose.`;
}

export function normalizeCompressedContext(raw, maxChars = 4000, maxBullets = 6) {
  const text = String(raw || "").trim();
  if (!text || text.toUpperCase().includes(NO_RELEVANT_MEMORY)) return "";
  const bullets = text.split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^[-*]\s+/.test(line) && line.includes("viking://"))
    .slice(0, Math.max(1, maxBullets))
    .map((line) => `- ${line.replace(/^[-*]\s+/, "").slice(0, 500).trim()}`);
  if (!bullets.length) return null;
  return (`OpenViking memory digest:\n${bullets.join("\n")}`).slice(0, Math.max(100, maxChars));
}

export function recallDigestCacheKey(entries = [], rendered = "") {
  const uris = entries.map((entry) => String(entry?.uri || "").trim()).filter(Boolean).sort();
  const source = uris.length ? uris.join("\n") : (String(rendered).match(/viking:\/\/[^\s<]+/g) || []).sort().join("\n");
  return createHash("sha256").update(source).digest("hex");
}

async function readCache(path) {
  if (!path) return null;
  try { return JSON.parse(await readFile(path, "utf8")); } catch { return null; }
}

async function writeCache(path, value) {
  if (!path) return;
  try {
    await mkdir(dirname(path), { recursive: true });
    const tmp = `${path}.tmp`;
    await writeFile(tmp, JSON.stringify(value));
    await rename(tmp, path);
  } catch { /* best effort */ }
}

export async function compressRecallContext({
  query,
  rendered,
  entries = [],
  cfg = {},
  runCompressor,
  cachePath = "",
}) {
  const input = String(rendered || "").trim();
  if (!input) return "";
  const minChars = Math.max(0, Number(cfg.recallCompressMinInputChars ?? 1500));
  if (input.length < minChars) return input;

  const key = recallDigestCacheKey(entries, input);
  const cached = await readCache(cachePath);
  if (cached?.key === key && typeof cached.digest === "string") return cached.digest;

  const maxInputChars = Math.max(1000, Number(cfg.recallCompressMaxInputChars || 18000));
  const maxBullets = Math.max(1, Number(cfg.recallCompressMaxBullets || 6));
  const prompt = buildRecallCompressionPrompt({
    query,
    rendered: input.slice(0, maxInputChars),
    maxBullets,
  });
  const raw = await runCompressor(prompt);
  const digest = normalizeCompressedContext(raw, 4000, maxBullets);
  if (digest === null) return null;
  await writeCache(cachePath, { key, digest, updatedAt: Date.now() });
  return digest;
}
