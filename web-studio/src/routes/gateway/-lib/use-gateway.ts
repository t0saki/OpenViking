import { useCallback, useMemo, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { UseQueryOptions } from '@tanstack/react-query'

import { useAppConnection } from '#/hooks/use-app-connection'
import { useManagementKey } from './management-key'

import {
  getConnectionInfo,
  getAccess,
  getOverview,
  listKeyUsers,
  listKeys,
  listLogs,
  listProfiles,
  listTools,
  listUpstreams,
} from './api'
import type {
  ConnectionInfo,
  GatewayConnection,
  GatewayAccess,
  GatewayError,
  GatewayKey,
  GatewayTool,
  KeyUser,
  LogRecord,
  Overview,
  Profile,
  Upstream,
} from './api'

/** Cached gateway data, invalidated per resource after a change. */
export type GatewayResource =
  | 'access'
  | 'connection'
  | 'overview'
  | 'logs'
  | 'upstreams'
  | 'profiles'
  | 'tools'
  | 'keys'
  | 'users'

/** Cache isolation by server, account and fingerprints of both credentials. */
export type GatewayScope = readonly [string, string, string, string]

function hashSecret(value: string): string {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(36)
}

export function gatewayScope(
  baseUrl: string,
  accountId: string,
  adminApiKey: string,
  managementKey = '',
): GatewayScope {
  return [
    baseUrl,
    accountId,
    adminApiKey ? hashSecret(adminApiKey) : 'none',
    managementKey ? hashSecret(managementKey) : 'none',
  ]
}

/** Query key `['gateway', scope, resource, ...params]`. */
export function gatewayQueryKey(
  scope: GatewayScope,
  resource?: GatewayResource,
  ...params: unknown[]
) {
  return resource
    ? (['gateway', scope, resource, ...params] as const)
    : (['gateway', scope] as const)
}

/**
 * Access and connection for gateway management. `connection` uses the
 * configured OpenViking credential; `invalidate()` refreshes the listed resources, or all
 * gateway data when called without arguments.
 */
function useGatewayConnection() {
  const { connection, connectionRole, isConnectionRoleLoading, serverMode } =
    useAppConnection()
  const { baseUrl, accountId, userId } = connection
  const apiKey = connection.adminApiKey || connection.apiKey
  const managementKey = useManagementKey(baseUrl)
  const admin = useMemo<GatewayConnection>(
    () => ({ baseUrl, accountId, userId, apiKey, managementKey }),
    [baseUrl, accountId, userId, apiKey, managementKey],
  )
  const scope = useMemo(
    () => gatewayScope(baseUrl, accountId, apiKey, managementKey),
    [baseUrl, accountId, apiKey, managementKey],
  )
  const queryClient = useQueryClient()
  const invalidate = useCallback(
    async (...resources: GatewayResource[]) => {
      const keys = resources.length
        ? resources.map((resource) => gatewayQueryKey(scope, resource))
        : [gatewayQueryKey(scope)]
      await Promise.all(
        keys.map((queryKey) => queryClient.invalidateQueries({ queryKey })),
      )
    },
    [queryClient, scope],
  )
  return {
    allowed: serverMode === 'dev' || Boolean(apiKey.trim()),
    /** Local development authenticates as default/default without a credential. */
    devMode: serverMode === 'dev',
    isRoleLoading: isConnectionRoleLoading,
    /** Role discovered by the Studio connection; permissions come from /access. */
    role: connectionRole,
    connection: admin,
    scope,
    managementKey,
    invalidate,
  }
}

/** Extra `useQuery` options a page may set (polling, placeholders…). */
export type GatewayQueryOptions<T> = Omit<
  UseQueryOptions<T, GatewayError>,
  'queryKey' | 'queryFn'
>

function useGatewayQuery<T>(
  resource: GatewayResource,
  params: unknown[],
  load: (connection: GatewayConnection) => Promise<T>,
  options: GatewayQueryOptions<T> = {},
) {
  const { allowed, connection, scope } = useGatewayConnection()
  return useQuery<T, GatewayError>({
    ...options,
    queryKey: gatewayQueryKey(scope, resource, ...params),
    queryFn: () => load(connection),
    enabled: allowed && options.enabled !== false,
  })
}

export const useAccess = (options?: GatewayQueryOptions<GatewayAccess>) =>
  useGatewayQuery('access', [], getAccess, {
    retry: false,
    retryOnMount: false,
    staleTime: 30_000,
    ...options,
  })

export function useGateway() {
  const gateway = useGatewayConnection()
  const accessQuery = useAccess()
  useEffect(() => {
    const refresh = () => {
      void gateway.invalidate('access')
    }
    window.addEventListener('gateway-access-changed', refresh)
    return () => window.removeEventListener('gateway-access-changed', refresh)
  }, [gateway.invalidate])
  return { ...gateway, access: accessQuery.data, accessQuery }
}

export const useGatewayContext = useGateway

/** Gateway address for clients (also tells whether the gateway is reachable). */
export const useConnectionInfo = (
  options?: GatewayQueryOptions<ConnectionInfo>,
) => useGatewayQuery('connection', [], getConnectionInfo, options)

export const useOverview = (options?: GatewayQueryOptions<Overview>) =>
  useGatewayQuery('overview', [], getOverview, options)

/** Newest-first request log, up to `limit` (≤ 1,000) records. */
export const useLogs = (
  limit: number,
  options?: GatewayQueryOptions<LogRecord[]>,
) =>
  useGatewayQuery(
    'logs',
    [limit],
    (connection) => listLogs(connection, limit),
    options,
  )

export const useUpstreams = (options?: GatewayQueryOptions<Upstream[]>) =>
  useGatewayQuery('upstreams', [], listUpstreams, options)

export const useProfiles = (options?: GatewayQueryOptions<Profile[]>) =>
  useGatewayQuery('profiles', [], listProfiles, options)

export const useTools = (options?: GatewayQueryOptions<GatewayTool[]>) =>
  useGatewayQuery('tools', [], listTools, options)

export const useKeys = (options?: GatewayQueryOptions<GatewayKey[]>) =>
  useGatewayQuery('keys', [], listKeys, options)

/** Users of this account a key can act as, and whether the server can read their key. */
export const useKeyUsers = (options?: GatewayQueryOptions<KeyUser[]>) =>
  useGatewayQuery('users', [], listKeyUsers, options)
