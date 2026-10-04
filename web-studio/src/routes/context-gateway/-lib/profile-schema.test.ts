import { describe, expect, it } from 'vitest'

import type { Profile, ProfileSettings } from './api'
import {
  PROFILE_DEFAULTS,
  duplicateProfile,
  offeredTools,
  toProfileSettings,
  toolAccess,
  validateProfile,
} from './profile-schema'

const profile = (patch: Partial<ProfileSettings> = {}): ProfileSettings => ({
  ...PROFILE_DEFAULTS,
  ...patch,
})

describe('toProfileSettings', () => {
  it('strips id and revision and keeps false and zero values', () => {
    const stored: Profile = {
      ...profile({ recall: false, score_threshold: 0, session_max_tokens: 0 }),
      id: 'p1',
      revision: 4,
    }
    const settings = toProfileSettings(stored)
    expect(settings).not.toHaveProperty('id')
    expect(settings).not.toHaveProperty('revision')
    expect(settings).toMatchObject({
      recall: false,
      score_threshold: 0,
      session_max_tokens: 0,
    })
  })

  it('round-trips every field, filling ones the server left out', () => {
    const { query_max_chars: _omitted, ...partial } = profile({
      quotas: { resources: 200 },
      context_window: 128000,
    })
    const settings = toProfileSettings(partial as ProfileSettings)
    expect(Object.keys(settings).sort()).toEqual(
      Object.keys(PROFILE_DEFAULTS).sort(),
    )
    expect(settings.query_max_chars).toBe(8000)
    expect(settings.quotas).toEqual({ resources: 200 })
    expect(settings.context_window).toBe(128000)
  })

  it('duplicates under a new name', () => {
    expect(
      duplicateProfile({ ...profile(), id: 'p', revision: 1 }, 'Copy'),
    ).toEqual({ ...PROFILE_DEFAULTS, name: 'Copy' })
  })
})

describe('validateProfile', () => {
  it('accepts the recommended settings', () => {
    expect(validateProfile(PROFILE_DEFAULTS)).toEqual({})
  })

  it('checks every limit, including hidden sections', () => {
    const errors = validateProfile(
      profile({
        name: ' ',
        max_tokens: 63,
        recall_timeout: 0,
        query_max_chars: Number.NaN,
        keep_recent_turns: 101,
        context_window: 1000,
        archive_wait_seconds: 61,
        tool_result_bytes: 1.5,
      }),
    )
    expect(Object.keys(errors).sort()).toEqual([
      'archive_wait_seconds',
      'context_window',
      'keep_recent_turns',
      'max_tokens',
      'name',
      'query_max_chars',
      'recall_timeout',
      'tool_result_bytes',
    ])
    expect(errors.recall_timeout?.key).toBe('validation.rangeExclusive')
    expect(errors.query_max_chars?.key).toBe('validation.required')
    expect(errors.tool_result_bytes?.key).toBe('validation.integer')
  })

  it('allows an empty context window', () => {
    expect(validateProfile(profile({ context_window: null }))).toEqual({})
  })

  it('needs a source while recall is on', () => {
    expect(validateProfile(profile({ context_types: [] }))).toHaveProperty(
      'context_types',
    )
    expect(
      validateProfile(profile({ context_types: [], recall: false })),
    ).toEqual({})
  })

  it('checks category limits', () => {
    const errors = validateProfile(
      profile({
        quotas: { events: -1, unknown: 2 } as ProfileSettings['quotas'],
      }),
    )
    expect(errors['quotas.events']?.key).toBe('validation.min')
    expect(errors['quotas.unknown']).toEqual({
      key: 'validation.unknownCategory',
      values: { name: 'unknown' },
    })
    expect(
      validateProfile(profile({ quotas: { events: 0, skills: 0 } })).quotas,
    ).toEqual({ key: 'validation.quotasAllZero' })
    expect(validateProfile(profile({ quotas: { skills: 3 } }))).toEqual({})
  })

  it('needs a tool that can actually be offered', () => {
    const writeOnly = profile({
      gateway_tools: true,
      tool_allowlist: ['write'],
    })
    expect(validateProfile(writeOnly).tool_allowlist?.key).toBe(
      'validation.selectOneTool',
    )
    expect(validateProfile({ ...writeOnly, allow_write_tools: true })).toEqual(
      {},
    )
  })
})

describe('tool access', () => {
  it('offers write tools only when allowed', () => {
    const tools = profile({
      gateway_tools: true,
      tool_allowlist: ['search', 'write', 'add_skill'],
    })
    expect(offeredTools(tools)).toEqual(['search'])
    expect(toolAccess(tools)).toBe('read')
    expect(toolAccess({ ...tools, allow_write_tools: true })).toBe('readWrite')
    expect(toolAccess({ ...tools, gateway_tools: false })).toBe('off')
  })
})
