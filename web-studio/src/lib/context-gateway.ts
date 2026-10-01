import type { AdminConnection } from './admin'

export type GatewayObject = {
  id: string
  name: string
  revision: number
} & Record<string, unknown>
export type GatewayKind = 'upstreams' | 'policies' | 'keys'

export async function gatewayRequest<T>(
  connection: AdminConnection,
  path: string,
  method = 'GET',
  body?: unknown,
): Promise<T> {
  const response = await fetch(
    `${connection.baseUrl.replace(/\/$/, '')}/api/v1/admin/context-gateway/${path}`,
    {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': connection.apiKey,
        'X-OpenViking-Account': connection.accountId,
        'X-OpenViking-User': connection.userId,
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    },
  )
  if (!response.ok) {
    const result = await response.json().catch(() => null)
    throw new Error(
      typeof result?.detail === 'string'
        ? result.detail
        : `Context Gateway: HTTP ${response.status}`,
    )
  }
  return response.json() as Promise<T>
}

export type GatewayField = {
  name: string
  type?: 'number' | 'boolean' | 'json' | 'password'
  initial: unknown
}
export const upstreamFields: GatewayField[] = [
  { name: 'name', initial: '' },
  { name: 'protocol', initial: 'chat' },
  { name: 'base_url', initial: 'https://api.openai.com/v1' },
  { name: 'api_key', type: 'password', initial: '' },
  { name: 'auth_mode', initial: 'managed' },
  { name: 'vendor', initial: 'generic' },
  { name: 'allow_gateway_tools', type: 'boolean', initial: true },
  { name: 'coding_plan', type: 'boolean', initial: false },
  { name: 'allow_coding_plan', type: 'boolean', initial: false },
  { name: 'cache_min_tokens', type: 'number', initial: 1024 },
  { name: 'models', type: 'json', initial: [] },
  { name: 'aliases', type: 'json', initial: {} },
  { name: 'headers', type: 'json', initial: {} },
  { name: 'priority', type: 'number', initial: 0 },
  { name: 'enabled', type: 'boolean', initial: true },
]
export const policyFields: GatewayField[] = [
  { name: 'name', initial: '' },
  { name: 'recall', type: 'boolean', initial: true },
  { name: 'capture', type: 'boolean', initial: true },
  {
    name: 'context_types',
    type: 'json',
    initial: ['memory', 'resource', 'skill'],
  },
  { name: 'quotas', type: 'json', initial: {} },
  { name: 'max_tokens', type: 'number', initial: 1600 },
  { name: 'session_max_tokens', type: 'number', initial: 6000 },
  { name: 'score_threshold', type: 'number', initial: 0.35 },
  { name: 'recall_timeout', type: 'number', initial: 2 },
  { name: 'commit_tokens', type: 'number', initial: 20000 },
  { name: 'keep_recent_messages', type: 'number', initial: 10 },
  { name: 'idle_seconds', type: 'number', initial: 600 },
  { name: 'takeover', type: 'boolean', initial: true },
  { name: 'takeover_tokens', type: 'number', initial: 30000 },
  { name: 'keep_recent_turns', type: 'number', initial: 3 },
  { name: 'context_window', type: 'number', initial: 128000 },
  { name: 'archive_wait_seconds', type: 'number', initial: 30 },
  { name: 'gateway_tools', type: 'boolean', initial: false },
  { name: 'allow_write_tools', type: 'boolean', initial: false },
  { name: 'tool_allowlist', type: 'json', initial: ['search', 'read', 'list'] },
  { name: 'tool_max_rounds', type: 'number', initial: 5 },
  { name: 'tool_timeout_seconds', type: 'number', initial: 30 },
  { name: 'tool_result_bytes', type: 'number', initial: 65536 },
  { name: 'tool_total_seconds', type: 'number', initial: 120 },
  { name: 'tool_total_tokens', type: 'number', initial: 100000 },
]

export function formValues(fields: GatewayField[], value?: GatewayObject) {
  return Object.fromEntries(
    fields.map((field) => {
      const initial = value?.[field.name] ?? field.initial
      return [
        field.name,
        field.type === 'json' ? JSON.stringify(initial, null, 2) : initial,
      ]
    }),
  )
}

export function formPayload(
  fields: GatewayField[],
  values: Record<string, unknown>,
) {
  return Object.fromEntries(
    fields.map((field) => [
      field.name,
      field.type === 'json'
        ? (JSON.parse(String(values[field.name])) as unknown)
        : field.type === 'number'
          ? Number(values[field.name])
          : values[field.name],
    ]),
  )
}
