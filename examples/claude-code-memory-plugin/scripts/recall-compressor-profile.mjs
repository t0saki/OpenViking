import { spawn } from "node:child_process";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const STATE_DIR = process.env.OPENVIKING_CC_STATE_DIR
  || join(homedir(), ".openviking", "cc-plugin-state");
const PROFILE_PATH = join(STATE_DIR, "recall-compressor-profile.json");
const TTL_MS = 7 * 24 * 60 * 60 * 1000;

function configKey(cfg) {
  return JSON.stringify({ enabled: cfg.recallCompress, model: cfg.recallCompressModel || "haiku" });
}

async function save(cfg, profile) {
  await mkdir(STATE_DIR, { recursive: true });
  const tmp = `${PROFILE_PATH}.tmp`;
  await writeFile(tmp, JSON.stringify({ checkedAt: Date.now(), configKey: configKey(cfg), profile }));
  await rename(tmp, PROFILE_PATH);
}

export async function loadCachedRecallCompressorProfile(cfg) {
  try {
    const cached = JSON.parse(await readFile(PROFILE_PATH, "utf8"));
    if (cached.configKey !== configKey(cfg) || Date.now() - Number(cached.checkedAt || 0) > TTL_MS) return null;
    return cached.profile || null;
  } catch {
    return null;
  }
}

export async function markRecallCompressorRuntimeFailed(cfg) {
  try { await save(cfg, { enabled: false, source: "runtime_failed" }); } catch { /* best effort */ }
}

function probeClaude(timeoutMs = 10000) {
  return new Promise((resolve) => {
    let done = false;
    const child = spawn("claude", ["--version"], { stdio: ["ignore", "pipe", "ignore"] });
    const timer = setTimeout(() => {
      try { child.kill("SIGKILL"); } catch { /* best effort */ }
      if (!done) { done = true; resolve(false); }
    }, timeoutMs);
    child.on("error", () => {
      clearTimeout(timer);
      if (!done) { done = true; resolve(false); }
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (!done) { done = true; resolve(code === 0); }
    });
  });
}

export async function detectRecallCompressorProfile(cfg, logger = {}) {
  if (!cfg.recallCompress) return { enabled: false, source: "configured_off" };
  const cached = await loadCachedRecallCompressorProfile(cfg);
  if (cached && cached.source !== "runtime_failed") return cached;
  const enabled = await probeClaude(Math.min(10000, cfg.timeoutMs || 10000));
  const profile = enabled
    ? { enabled: true, model: cfg.recallCompressModel || "haiku", source: "claude_version" }
    : { enabled: false, source: "probe_failed" };
  try { await save(cfg, profile); } catch { /* best effort */ }
  logger.log?.("compress_profile_selected", profile);
  return profile;
}
