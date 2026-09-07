// GENERATED FROM examples/memory-plugin-shared/lib. DO NOT EDIT.
export const HARNESS_KEYS: Record<string, string>;

export function harnessKey(harness: string): string;

export function loadPluginSettings(
  harness: string,
  env?: Record<string, string | undefined>,
  options?: { cwd?: string; clientVersion?: string },
): Record<string, any>;

export function resolveSettings(
  harness: string,
  options?: {
    env?: Record<string, string | undefined>;
    cwd?: string;
    legacy?: Record<string, any>;
    clientVersion?: string;
  },
): {
  settings: Record<string, any>;
  configured: Set<string>;
  sources: Record<string, string>;
  plugin: Record<string, any>;
};

export function resolveWorkspaceSettings(
  cwd: string,
  env?: Record<string, string | undefined>,
  options?: { clientVersion?: string },
): {
  settings: Record<string, any>;
  root: string;
  provenance: Record<string, any>;
  warnings: string[];
  announced: string[];
  value?: Record<string, any>;
};

export function normalizeRewriteMode(value: unknown, fallback?: string): string;
