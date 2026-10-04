import { describe, expect, it } from 'vitest'

import type { Upstream, UpstreamInput } from './api'
import {
  UPSTREAM_DEFAULTS,
  endpointPreview,
  isValidBaseUrl,
  servedModels,
  toUpstreamInput,
  upstreamUrl,
  validateUpstream,
} from './upstream-schema'

const stored: Upstream = {
  ...UPSTREAM_DEFAULTS,
  id: 'u1',
  revision: 4,
  name: 'OpenAI',
  base_url: 'https://api.openai.com/v1',
  models: ['gpt-5'],
  aliases: { fast: 'gpt-5-mini' },
  enabled: false,
  priority: 0,
  has_api_key: true,
  header_names: ['OpenAI-Organization'],
}

const input = (patch: Partial<UpstreamInput> = {}): UpstreamInput => ({
  ...UPSTREAM_DEFAULTS,
  name: 'Chat',
  base_url: 'https://api.example.com/v1',
  api_key: 'sk-test',
  ...patch,
})

describe('toUpstreamInput', () => {
  it('drops display-only fields and never resends secrets', () => {
    const body = toUpstreamInput(stored)
    for (const field of ['id', 'revision', 'has_api_key', 'header_names']) {
      expect(body).not.toHaveProperty(field)
    }
    expect(body.api_key).toBe('')
    expect(body.headers).toEqual({ 'OpenAI-Organization': '' })
    expect(body).toMatchObject({
      enabled: false,
      priority: 0,
      models: ['gpt-5'],
    })
    expect(Object.keys(body).sort()).toEqual(
      Object.keys(UPSTREAM_DEFAULTS).sort(),
    )
  })
})

it('lists models and alias names once', () => {
  expect(
    servedModels({ models: ['a', 'b'], aliases: { b: 'x', c: 'y' } }),
  ).toEqual(['a', 'b', 'c'])
})

describe('base URLs', () => {
  it.each([
    ['https://api.openai.com/v1', true],
    ['http://10.0.0.5:8000', true],
    ['ftp://example.com', false],
    ['https://user:pass@example.com', false],
    ['https://example.com/?a=1', false],
    ['https://example.com/?', false],
    ['https://example.com/#top', false],
    ['example.com', false],
  ])('%s → %s', (url, valid) => {
    expect(isValidBaseUrl(url)).toBe(valid)
  })

  it.each([
    [
      'https://api.openai.com/v1',
      'generic',
      '/v1/chat/completions',
      'https://api.openai.com/v1/chat/completions',
    ],
    [
      'https://api.anthropic.com/',
      'anthropic',
      '/v1/messages',
      'https://api.anthropic.com/v1/messages',
    ],
    [
      'https://host/api/paas/v4',
      'generic',
      '/v1/chat/completions',
      'https://host/api/paas/v4/v1/chat/completions',
    ],
    [
      'https://ark.cn-beijing.volces.com',
      'ark',
      '/v1/messages',
      'https://ark.cn-beijing.volces.com/api/compatible/v1/messages',
    ],
    [
      'https://ark.cn-beijing.volces.com',
      'ark',
      '/v1/responses',
      'https://ark.cn-beijing.volces.com/api/v3/responses',
    ],
    [
      'https://ark.cn-beijing.volces.com/api/v3',
      'ark',
      '/v1/chat/completions',
      'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
    ],
    [
      'https://ark.cn-beijing.volces.com/api/v3',
      'ark',
      '/v1/messages',
      'https://ark.cn-beijing.volces.com/api/v3/api/compatible/v1/messages',
    ],
  ] as const)('%s (%s) + %s', (base, vendor, path, expected) => {
    expect(upstreamUrl(base, vendor, path)).toBe(expected)
  })

  it('previews the protocol endpoint once the URL is valid', () => {
    expect(endpointPreview(input({ protocol: 'responses' }))).toBe(
      'https://api.example.com/v1/responses',
    )
    expect(endpointPreview(input({ base_url: 'api.example' }))).toBe('')
  })
})

describe('validateUpstream', () => {
  it('accepts a complete upstream', () => {
    expect(validateUpstream(input())).toEqual({})
  })

  it('needs a key for a new managed upstream but keeps a stored one', () => {
    expect(validateUpstream(input({ api_key: '' })).api_key?.key).toBe(
      'validation.apiKeyRequired',
    )
    expect(validateUpstream(input({ api_key: '' }), stored)).toEqual({})
    expect(
      validateUpstream(input({ api_key: '', auth_mode: 'passthrough' })),
    ).toEqual({})
    expect(
      validateUpstream(input({ api_key: 'sk-ant-oat01-x' })).api_key?.key,
    ).toBe('validation.subscriptionKey')
  })

  it('checks headers, aliases and context windows', () => {
    const header = (headers: Record<string, string>) =>
      validateUpstream(input({ headers }), stored).headers
    expect(header({ 'OpenAI-Organization': '' })).toBeUndefined()
    expect(header({ 'X-New': '' })).toEqual({
      key: 'validation.headerValue',
      values: { name: 'X-New' },
    })
    expect(header({ Host: 'x' })?.key).toBe('validation.headerReserved')
    expect(header({ 'X-Bad': 'a\nb' })?.key).toBe('validation.headerInvalid')
    expect(
      validateUpstream(input({ aliases: { fast: ' ' } })).aliases?.key,
    ).toBe('validation.aliasTarget')
    expect(
      validateUpstream(input({ context_windows: { m: 512 } })).context_windows,
    ).toEqual({
      key: 'validation.contextWindow',
      values: { name: 'm', min: 1024 },
    })
  })

  it('checks name, URL and numbers', () => {
    const errors = validateUpstream(
      input({
        name: '',
        base_url: 'not a url',
        priority: 1.5,
        cache_min_tokens: -1,
      }),
    )
    expect(Object.keys(errors).sort()).toEqual([
      'base_url',
      'cache_min_tokens',
      'name',
      'priority',
    ])
  })
})
