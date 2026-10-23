export type RedactTier = "off" | "standard" | "strict";

export interface RedactOptions {
  /** One of REDACT_TIERS (default "off"); anything else is normalised (unknown names mean "standard"). */
  tier?: string;
  /** Override the per-install HMAC key (tests). */
  key?: string | Uint8Array;
  /** Where the install key's state directory is looked up. */
  env?: Record<string, string | undefined>;
  /** Scannable characters per string before the rest is cut (default MAX_SCAN_CHARS). */
  maxChars?: number;
  /** A field name to scan the value under, as `keyName=value`. */
  keyName?: string;
}

export const REDACT_TIERS: readonly string[];
export const DEFAULT_REDACT_TIER: "off";
export const ENABLED_REDACT_TIER: "standard";
export const MAX_SCAN_CHARS: number;

export function normalizeRedactTier(value: unknown, fallback?: string): string;
export function redactText(text: string, opts?: RedactOptions): string;
export function redactPayload<T>(payload: T, opts?: RedactOptions): T;
export function redactPayloads<T>(payloads: T[], opts?: RedactOptions): T[];
export function loadInstallKey(opts?: { env?: Record<string, string | undefined> }): Uint8Array;
export function shannonEntropy(data: string): number;
