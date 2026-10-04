// @vitest-environment jsdom
import type * as React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as Api from './api'
import {
  gatewayQueryKey,
  gatewayScope,
  useGateway,
  useLogs,
  useUpstreams,
} from './use-gateway'

const state = vi.hoisted(() => ({ role: 'admin' }))
const api = vi.hoisted(() => ({ listUpstreams: vi.fn(), listLogs: vi.fn() }))

vi.mock('#/hooks/use-app-connection', () => ({
  useAppConnection: () => ({
    connection: {
      baseUrl: 'https://ov.example.com',
      accountId: 'acme',
      userId: 'admin',
      apiKey: 'data-key',
      adminApiKey: 'admin-key',
    },
    connectionRole: state.role,
    isConnectionRoleLoading: false,
    serverMode: 'api_key',
  }),
}))
vi.mock('./api', async (importOriginal) => ({
  ...(await importOriginal<typeof Api>()),
  listUpstreams: api.listUpstreams,
  listLogs: api.listLogs,
}))

let client: QueryClient
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
)

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  state.role = 'admin'
  api.listUpstreams.mockReset()
  api.listLogs.mockReset()
})
afterEach(cleanup)

it('scopes data by server, account and a hash of the admin key', () => {
  const scope = gatewayScope('https://ov.example.com', 'acme', 'admin-key')
  expect(scope.slice(0, 2)).toEqual(['https://ov.example.com', 'acme'])
  expect(scope[2]).not.toContain('admin-key')
  expect(gatewayQueryKey(scope, 'logs', 50)).toEqual([
    'context-gateway',
    scope,
    'logs',
    50,
  ])
})

it('manages with the admin key for account administrators', () => {
  const { result } = renderHook(() => useGateway(), { wrapper })
  expect(result.current.allowed).toBe(true)
  expect(result.current.connection).toEqual({
    baseUrl: 'https://ov.example.com',
    accountId: 'acme',
    userId: 'admin',
    apiKey: 'admin-key',
  })
})

it('does not load anything for regular users', () => {
  state.role = 'user'
  const { result } = renderHook(() => useUpstreams(), { wrapper })
  expect(result.current.fetchStatus).toBe('idle')
  expect(api.listUpstreams).not.toHaveBeenCalled()
})

it('loads resources with the admin connection and invalidates by resource', async () => {
  api.listUpstreams.mockResolvedValue([{ id: 'u' }])
  api.listLogs.mockResolvedValue([])
  const { result } = renderHook(
    () => ({
      gateway: useGateway(),
      upstreams: useUpstreams(),
      logs: useLogs(25),
    }),
    { wrapper },
  )
  await waitFor(() =>
    expect(result.current.upstreams.data).toEqual([{ id: 'u' }]),
  )
  expect(api.listUpstreams.mock.calls[0][0].apiKey).toBe('admin-key')
  expect(api.listLogs.mock.calls[0][1]).toBe(25)

  await result.current.gateway.invalidate('upstreams')
  expect(api.listUpstreams).toHaveBeenCalledTimes(2)
  expect(api.listLogs).toHaveBeenCalledTimes(1)

  await result.current.gateway.invalidate()
  expect(api.listLogs).toHaveBeenCalledTimes(2)
})
