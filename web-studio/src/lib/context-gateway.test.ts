import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  formPayload,
  formValues,
  gatewayRequest,
  policyFields,
  upstreamFields,
} from './context-gateway'

afterEach(() => vi.unstubAllGlobals())

describe('Context Gateway management', () => {
  it('uses the verified account headers and never puts keys in the URL', async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ key: 'issued-once' }),
    })
    vi.stubGlobal('fetch', fetch)
    const result = await gatewayRequest(
      {
        accountId: 'tenant',
        userId: 'alice',
        apiKey: 'admin-secret',
        baseUrl: 'https://ov.example.com/',
      },
      'keys',
      'POST',
      { name: 'Chat' },
    )
    expect(result).toEqual({ key: 'issued-once' })
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('https://ov.example.com/api/v1/admin/context-gateway/keys')
    expect(options.headers['X-OpenViking-Account']).toBe('tenant')
    expect(options.headers['X-API-Key']).toBe('admin-secret')
  })

  it('round-trips editable policy fields and preserves false/zero', () => {
    const values = formValues(policyFields)
    values.recall = false
    values.score_threshold = '0'
    values.quotas = '{"resources": 200}'
    expect(formPayload(policyFields, values)).toMatchObject({
      context_window: null,
      recall: false,
      score_threshold: 0,
      quotas: { resources: 200 },
    })
  })

  it('does not resubmit display-only metadata or hidden secrets while editing', () => {
    const values = formValues(upstreamFields, {
      id: 'u',
      name: 'Model',
      revision: 4,
      has_api_key: true,
      header_names: ['x-custom'],
    })
    const payload = formPayload(upstreamFields, values)
    expect(payload.api_key).toBe('')
    expect(payload).not.toHaveProperty('revision')
    expect(payload).not.toHaveProperty('has_api_key')
    expect(payload).not.toHaveProperty('header_names')
  })
})
