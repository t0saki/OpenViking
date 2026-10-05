import { describe, expect, it } from 'vitest'

import type { Profile, ProfileSettings } from './api'
import {
  PROFILE_DEFAULTS,
  duplicateProfile,
  offeredTools,
  toProfileSettings,
  validateProfile,
} from './profile-schema'

const profile = (patch: Partial<ProfileSettings> = {}): ProfileSettings => ({
  ...PROFILE_DEFAULTS,
  ...patch,
})

describe('toProfileSettings', () => {
  it('keeps known fields only while preserving false and zero values', () => {
    const stored: Profile = {
      ...profile({ recall: false, score_threshold: 0, session_max_tokens: 0 }),
      id: 'p1',
      revision: 4,
    }
    const settings = toProfileSettings({
      ...stored,
      allow_write_tools: true,
      tool_allowlist: ['read'],
      future_setting: 'discard',
    } as Profile)
    expect(settings).not.toHaveProperty('id')
    expect(settings).not.toHaveProperty('revision')
    expect(settings).not.toHaveProperty('allow_write_tools')
    expect(settings).not.toHaveProperty('tool_allowlist')
    expect(settings).not.toHaveProperty('future_setting')
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

  it('shows tool calls for profiles saved before the setting existed', () => {
    const { show_tool_calls: _omitted, ...legacy } = profile()
    expect(toProfileSettings(legacy as ProfileSettings).show_tool_calls).toBe(
      true,
    )
    expect(
      toProfileSettings(profile({ show_tool_calls: false })),
    ).toMatchObject({ show_tool_calls: false })
  })

  it('defaults older profiles to no disabled tools and keeps explicit exclusions', () => {
    const { disabled_tools: _omitted, ...legacy } = profile()
    expect(toProfileSettings(legacy as ProfileSettings).disabled_tools).toEqual(
      [],
    )
    expect(
      toProfileSettings(profile({ disabled_tools: ['forget', 'future_tool'] }))
        .disabled_tools,
    ).toEqual(['forget', 'future_tool'])
  })

  it('drops long-conversation settings that were replaced and defaults the new ones', () => {
    const {
      compaction: _compaction,
      compaction_threshold: _threshold,
      summary_max_tokens: _summary,
      agent_windows: _windows,
      window_soft_ratio: _soft,
      window_hard_ratio: _hard,
      ...rest
    } = profile({ context_window: 200000 })
    const legacy = {
      ...rest,
      takeover: false,
      takeover_tokens: 30000,
      keep_recent_turns: 3,
      archive_wait_seconds: 30,
    }
    const settings = toProfileSettings(legacy as unknown as ProfileSettings)
    for (const field of [
      'takeover',
      'takeover_tokens',
      'keep_recent_turns',
      'archive_wait_seconds',
    ]) {
      expect(settings).not.toHaveProperty(field)
    }
    expect(settings).toMatchObject({
      compaction: true,
      compaction_threshold: 0.9,
      summary_max_tokens: 8000,
      context_window: 200000,
      agent_windows: false,
      window_soft_ratio: 0.7,
      window_hard_ratio: 0.85,
    })
  })

  it('duplicates under a new name', () => {
    expect(
      duplicateProfile({ ...profile(), id: 'p', revision: 1 }, 'Copy'),
    ).toEqual({ ...PROFILE_DEFAULTS, name: 'Copy' })
  })

  it('defaults opening context for older profiles and preserves disabling it', () => {
    const {
      profile: _profile,
      profile_max_tokens: _budget,
      ...legacy
    } = profile()
    expect(toProfileSettings(legacy as ProfileSettings)).toMatchObject({
      profile: true,
      profile_max_tokens: 4000,
    })
    expect(
      toProfileSettings(profile({ profile: false, profile_max_tokens: 0 })),
    ).toMatchObject({
      profile: false,
      profile_max_tokens: 0,
    })
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
        compaction_threshold: 0.99,
        summary_max_tokens: 500,
        context_window: 1000,
        window_hard_ratio: 0.3,
        tool_result_bytes: 1.5,
      }),
    )
    expect(Object.keys(errors).sort()).toEqual([
      'compaction_threshold',
      'context_window',
      'max_tokens',
      'name',
      'query_max_chars',
      'recall_timeout',
      'summary_max_tokens',
      'tool_result_bytes',
      'window_hard_ratio',
    ])
    expect(errors.recall_timeout?.key).toBe('validation.rangeExclusive')
    expect(errors.query_max_chars?.key).toBe('validation.required')
    expect(errors.tool_result_bytes?.key).toBe('validation.integer')
  })

  it('allows an empty context window', () => {
    expect(validateProfile(profile({ context_window: null }))).toEqual({})
  })

  it('needs the soft reminder below the hard one, even while agent windows are off', () => {
    for (const soft of [0.85, 0.9]) {
      expect(
        validateProfile(profile({ window_soft_ratio: soft })).window_soft_ratio,
      ).toEqual({ key: 'validation.softBelowHard' })
    }
    expect(
      validateProfile(profile({ agent_windows: true, window_soft_ratio: 0.5 })),
    ).toEqual({})
    // An out-of-range ratio reports its range instead.
    expect(
      validateProfile(profile({ window_soft_ratio: 0.96 })).window_soft_ratio
        ?.key,
    ).toBe('validation.range')
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

  it('allows any exclusions, including tools absent from the current catalog', () => {
    expect(
      validateProfile(
        profile({
          gateway_tools: true,
          disabled_tools: ['read', 'write', 'unknown'],
        }),
      ),
    ).toEqual({})
  })
})

describe('tool access', () => {
  const tools = ['read', 'write', 'future_tool'].map((name) => ({
    name,
    description: name,
  }))

  it('keeps the gateway switch off by default but selects all tools when enabled', () => {
    expect(PROFILE_DEFAULTS.gateway_tools).toBe(false)
    expect(offeredTools(PROFILE_DEFAULTS, tools)).toEqual([])
    expect(offeredTools(profile({ gateway_tools: true }), tools)).toEqual(tools)
  })

  it('excludes only raw MCP names and enables new tools automatically', () => {
    const settings = profile({
      gateway_tools: true,
      disabled_tools: ['write', 'removed_tool'],
    })
    expect(offeredTools(settings, tools).map((tool) => tool.name)).toEqual([
      'read',
      'future_tool',
    ])
    expect(offeredTools({ ...settings, gateway_tools: false }, tools)).toEqual(
      [],
    )
    expect(
      offeredTools(
        { ...settings, disabled_tools: tools.map((tool) => tool.name) },
        tools,
      ),
    ).toEqual([])
  })
})
