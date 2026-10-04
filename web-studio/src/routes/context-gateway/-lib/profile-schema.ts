import type {
  ContextType,
  GatewayTool,
  Profile,
  ProfileSettings,
  QuotaBucket,
} from './api'
import { checkNumber, checkRequired, collect } from './validation'
import type { NumberRule, ValidationErrors } from './validation'

/** Recommended settings; identical to the gateway's own defaults. */
export const PROFILE_DEFAULTS: ProfileSettings = {
  name: 'Default',
  recall: true,
  capture: true,
  context_types: ['memory', 'resource', 'skill'],
  quotas: {},
  max_tokens: 1600,
  session_max_tokens: 6000,
  score_threshold: 0.35,
  recall_timeout: 2,
  query_max_chars: 8000,
  commit_tokens: 20000,
  keep_recent_messages: 10,
  idle_seconds: 600,
  takeover: true,
  takeover_tokens: 30000,
  keep_recent_turns: 3,
  context_window: null,
  archive_wait_seconds: 30,
  gateway_tools: false,
  allow_write_tools: false,
  tool_allowlist: ['search', 'read', 'list'],
  tool_max_rounds: 5,
  tool_timeout_seconds: 30,
  tool_result_bytes: 65536,
  tool_total_seconds: 120,
  tool_total_tokens: 100000,
}

export const CONTEXT_TYPES: ContextType[] = ['memory', 'resource', 'skill']

/** Categories for "Limit by category", in display order. */
export const QUOTA_BUCKETS: QuotaBucket[] = [
  'events',
  'entities',
  'preferences',
  'experiences',
  'resources',
  'skills',
]

export const READ_TOOLS: GatewayTool[] = ['search', 'read', 'list']
/** Offered only when `allow_write_tools` is on. */
export const WRITE_TOOLS: GatewayTool[] = ['write', 'add_resource', 'add_skill']

type NumericField = {
  [K in keyof ProfileSettings]: ProfileSettings[K] extends number | null
    ? K
    : never
}[keyof ProfileSettings]

/** Limits and units of every numeric profile field (mirrors `models.py`). */
export const PROFILE_LIMITS: Record<NumericField, NumberRule> = {
  max_tokens: { min: 64, max: 32000, integer: true, unit: 'tokens' },
  session_max_tokens: { min: 0, integer: true, unit: 'tokens' },
  score_threshold: { min: 0, max: 1, step: 0.05 },
  recall_timeout: { min: 0, max: 30, exclusiveMin: true, unit: 'seconds' },
  query_max_chars: { min: 3, max: 32000, integer: true, unit: 'characters' },
  commit_tokens: { min: 1, integer: true, unit: 'tokens' },
  keep_recent_messages: { min: 0, max: 1000, integer: true, unit: 'messages' },
  idle_seconds: { min: 1, unit: 'seconds' },
  takeover_tokens: { min: 1, integer: true, unit: 'tokens' },
  keep_recent_turns: { min: 1, max: 100, integer: true, unit: 'turns' },
  context_window: { min: 1024, integer: true, optional: true, unit: 'tokens' },
  archive_wait_seconds: { min: 0, max: 60, unit: 'seconds' },
  tool_max_rounds: { min: 1, max: 20, integer: true, unit: 'rounds' },
  tool_timeout_seconds: {
    min: 0,
    max: 120,
    exclusiveMin: true,
    unit: 'seconds',
  },
  tool_result_bytes: { min: 1024, max: 1048576, integer: true, unit: 'bytes' },
  tool_total_seconds: { min: 0, max: 600, exclusiveMin: true, unit: 'seconds' },
  tool_total_tokens: {
    min: 1024,
    max: 1000000,
    integer: true,
    unit: 'tokens',
  },
}

/** Per-category entry limit used by "Limit by category". */
export const QUOTA_LIMIT: NumberRule = {
  min: 0,
  integer: true,
  unit: 'entries',
}

/**
 * Save body for `PUT policies/{id}`: every profile field, with `id` and
 * `revision` stripped and anything the server omitted filled with defaults.
 */
export function toProfileSettings(
  profile: Profile | ProfileSettings,
): ProfileSettings {
  const settings: Record<string, unknown> = { ...PROFILE_DEFAULTS, ...profile }
  delete settings.id
  delete settings.revision
  return settings as ProfileSettings
}

/** A copy of `profile` for "Duplicate", named by `name`. */
export function duplicateProfile(
  profile: Profile | ProfileSettings,
  name: string,
): ProfileSettings {
  return { ...toProfileSettings(profile), name }
}

/**
 * Checks every field against the gateway's limits. Keys are field names, plus
 * `quotas.<category>` for category limits. Hidden sections are checked too,
 * because the server validates the whole profile.
 */
export function validateProfile(settings: ProfileSettings): ValidationErrors {
  const errors: ValidationErrors = {}
  collect(errors, 'name', checkRequired(settings.name))
  for (const [field, rule] of Object.entries(PROFILE_LIMITS)) {
    collect(errors, field, checkNumber(settings[field as NumericField], rule))
  }
  if (settings.recall && settings.context_types.length === 0) {
    collect(errors, 'context_types', { key: 'validation.selectOneSource' })
  }
  const quotas = Object.entries(settings.quotas)
  for (const [bucket, value] of quotas) {
    collect(
      errors,
      `quotas.${bucket}`,
      QUOTA_BUCKETS.includes(bucket as QuotaBucket)
        ? checkNumber(value, QUOTA_LIMIT)
        : { key: 'validation.unknownCategory', values: { name: bucket } },
    )
  }
  if (quotas.length > 0 && quotas.every(([, value]) => value === 0)) {
    collect(errors, 'quotas', { key: 'validation.quotasAllZero' })
  }
  if (settings.gateway_tools && offeredTools(settings).length === 0) {
    collect(errors, 'tool_allowlist', { key: 'validation.selectOneTool' })
  }
  return errors
}

/** OpenViking tools a new conversation can be offered under these settings. */
export function offeredTools(settings: ProfileSettings): GatewayTool[] {
  if (!settings.gateway_tools) return []
  return settings.tool_allowlist.filter(
    (tool) => settings.allow_write_tools || READ_TOOLS.includes(tool),
  )
}

/** One-word summary of tool access for lists: off, read only, or read & write. */
export function toolAccess(
  settings: ProfileSettings,
): 'off' | 'read' | 'readWrite' {
  const tools = offeredTools(settings)
  if (tools.length === 0) return 'off'
  return tools.some((tool) => WRITE_TOOLS.includes(tool)) ? 'readWrite' : 'read'
}
