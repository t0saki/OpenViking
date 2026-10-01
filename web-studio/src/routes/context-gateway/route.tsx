import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

import { useAppConnection } from '#/hooks/use-app-connection'
import { Button } from '#/components/ui/button'
import { Input } from '#/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '#/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '#/components/ui/tabs'
import { copyTextToClipboard } from '#/lib/clipboard'
import { resolveStudioManagementCapabilities } from '#/lib/studio-permissions'
import {
  formPayload,
  formValues,
  gatewayRequest,
  policyFields,
  upstreamFields,
} from '#/lib/context-gateway'
import type {
  GatewayField,
  GatewayKind,
  GatewayObject,
} from '#/lib/context-gateway'
import type { AdminConnection } from '#/lib/admin'

export const Route = createFileRoute('/context-gateway')({
  component: ContextGateway,
})
const tabs = [
  'overview',
  'upstreams',
  'policies',
  'keys',
  'logs',
  'guides',
] as const
const metricFields = ['requests', 'output_tokens', 'recall_count'] as const
const logFields = [
  'time',
  'kind',
  'model',
  'status',
  'replay_hits',
  'recall_count',
  'input_tokens',
  'cached_tokens',
  'degradation',
] as const
type Tab = (typeof tabs)[number]

function ContextGateway() {
  const { t } = useTranslation('contextGateway')
  const { connection, connectionRole, isConnectionRoleLoading, serverMode } =
    useAppConnection()
  const [tab, setTab] = useState<Tab>('overview')
  const allowed = resolveStudioManagementCapabilities({
    hasControlCredential: Boolean(connection.adminApiKey.trim()),
    isRoleLoading: isConnectionRoleLoading,
    role: connectionRole,
    serverMode,
  }).canManageUsers
  const admin: AdminConnection = {
    ...connection,
    apiKey: connection.adminApiKey || connection.apiKey,
  }
  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">{t('title')}</h1>
        <p className="mt-2 text-muted-foreground">{t('description')}</p>
      </div>
      {!allowed ? (
        <p role="alert">{t('adminRequired')}</p>
      ) : (
        <Tabs value={tab} onValueChange={(value) => setTab(value as Tab)}>
          <TabsList className="flex h-auto flex-wrap">
            {tabs.map((value) => (
              <TabsTrigger key={value} value={value}>
                {t(value)}
              </TabsTrigger>
            ))}
          </TabsList>
          {tabs.map((value) => (
            <TabsContent key={value} value={value}>
              {value === 'upstreams' ||
              value === 'policies' ||
              value === 'keys' ? (
                <Objects connection={admin} kind={value} />
              ) : (
                <Report connection={admin} kind={value} />
              )}
            </TabsContent>
          ))}
        </Tabs>
      )}
    </main>
  )
}

function useGateway(connection: AdminConnection, path: string) {
  return useQuery({
    queryKey: [
      'context-gateway',
      connection.baseUrl,
      connection.accountId,
      connection.apiKey,
      path,
    ],
    queryFn: () => gatewayRequest<unknown>(connection, path),
    retry: false,
  })
}

function Report({
  connection,
  kind,
}: {
  connection: AdminConnection
  kind: 'overview' | 'logs' | 'guides'
}) {
  const { t } = useTranslation('contextGateway')
  const query = useGateway(connection, kind)
  if (query.isPending) return <p>{t('loading')}</p>
  if (query.error) return <p role="alert">{query.error.message}</p>
  if (kind === 'guides')
    return (
      <div className="grid gap-4">
        {Object.entries(query.data as Record<string, string>).map(
          ([name, value]) => (
            <Card key={name}>
              <CardHeader>
                <CardTitle>{name}</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-muted p-4 text-sm">
                  {value}
                </pre>
                <Button
                  variant="outline"
                  className="mt-3"
                  onClick={() => void copyTextToClipboard(value)}
                >
                  {t('copy')}
                </Button>
              </CardContent>
            </Card>
          ),
        )}
      </div>
    )
  if (kind === 'overview') {
    const value = query.data as Record<string, unknown>
    const cache = value.cache as Record<
      string,
      { requests: number; cache_hit_ratio: number }
    >
    return (
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-3">
          {metricFields.map((name) => (
            <Card key={name}>
              <CardHeader>
                <CardTitle>{t(`fields.${name}`)}</CardTitle>
              </CardHeader>
              <CardContent className="text-3xl font-semibold">
                {String(value[name] ?? 0)}
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {Object.entries(cache).map(([name, group]) => (
            <Card key={name}>
              <CardHeader>
                <CardTitle>{t(name)}</CardTitle>
              </CardHeader>
              <CardContent className="text-2xl">
                {(group.cache_hit_ratio * 100).toFixed(1)}%
              </CardContent>
            </Card>
          ))}
        </div>
        <p className="text-sm text-muted-foreground">{t('sampleNote')}</p>
        <pre className="overflow-auto rounded bg-muted p-4 text-sm">
          {JSON.stringify(
            {
              openviking: value.openviking,
              degradations: value.degradations,
              recall_ms: value.recall_ms,
            },
            null,
            2,
          )}
        </pre>
        <Button variant="outline" onClick={() => void query.refetch()}>
          {t('refresh')}
        </Button>
      </div>
    )
  }
  const logs = query.data as Record<string, unknown>[]
  return (
    <div className="overflow-auto">
      <p className="mb-3 text-sm text-muted-foreground">{t('logsNote')}</p>
      <table className="w-full text-left text-sm">
        <thead>
          <tr>
            {logFields.map((name) => (
              <th className="p-2" key={name}>
                {t(`fields.${name}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {logs.map((log, index) => (
            <tr className="border-t" key={String(log.request_id ?? index)}>
              {logFields.map((name) => (
                <td className="p-2" key={name}>
                  {name === 'time'
                    ? new Date(Number(log[name]) * 1000).toLocaleString()
                    : String(log[name] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Objects({
  connection,
  kind,
}: {
  connection: AdminConnection
  kind: GatewayKind
}) {
  const { t } = useTranslation('contextGateway')
  const query = useGateway(connection, kind)
  const cache = useQueryClient()
  const [editing, setEditing] = useState<GatewayObject | null | undefined>()
  const [issued, setIssued] = useState('')
  const invalidate = () =>
    cache.invalidateQueries({ queryKey: ['context-gateway'] })
  const remove = useMutation({
    mutationFn: (id: string) =>
      gatewayRequest(connection, `${kind}/${encodeURIComponent(id)}`, 'DELETE'),
    onSuccess: invalidate,
    onError: (error: Error) => toast.error(error.message),
  })
  async function test(id: string) {
    try {
      const result = await gatewayRequest<{ ok: boolean; status?: number }>(
        connection,
        `upstreams/${encodeURIComponent(id)}/test`,
        'POST',
      )
      toast[result.ok ? 'success' : 'error'](
        result.ok
          ? t('testPassed')
          : `${t('testFailed')} ${result.status ?? ''}`,
      )
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t('testFailed'))
    }
  }
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">{t(`${kind}Note`)}</p>
        <Button
          onClick={() => {
            setIssued('')
            setEditing(null)
          }}
        >
          {t(kind === 'keys' ? 'issue' : 'add')}
        </Button>
      </div>
      {issued && (
        <Card>
          <CardContent className="space-y-3 pt-6">
            <p>{t('keyOnce')}</p>
            <code className="break-all">{issued}</code>
            <div>
              <Button onClick={() => void copyTextToClipboard(issued)}>
                {t('copy')}
              </Button>
              <Button variant="ghost" onClick={() => setIssued('')}>
                {t('dismiss')}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
      {query.error && <p role="alert">{query.error.message}</p>}
      {editing !== undefined &&
        (kind === 'keys' ? (
          <KeyForm
            connection={connection}
            onCancel={() => setEditing(undefined)}
            onSave={(key) => {
              setIssued(key)
              setEditing(undefined)
              void invalidate()
            }}
          />
        ) : (
          <ObjectForm
            key={editing?.id ?? 'new'}
            fields={kind === 'upstreams' ? upstreamFields : policyFields}
            value={editing ?? undefined}
            onCancel={() => setEditing(undefined)}
            onSave={async (value) => {
              await gatewayRequest(
                connection,
                `${kind}/${editing?.id ?? crypto.randomUUID()}`,
                'PUT',
                value,
              )
              setEditing(undefined)
              await invalidate()
            }}
          />
        ))}
      <div className="grid gap-3">
        {((query.data ?? []) as GatewayObject[]).map((value) => (
          <Card key={value.id}>
            <CardContent className="flex flex-wrap items-center justify-between gap-3 pt-6">
              <div>
                <h2 className="font-medium">{value.name}</h2>
                <p className="text-sm text-muted-foreground">
                  {kind === 'keys'
                    ? String(value.prefix)
                    : `${t('revision')} ${value.revision}`}
                </p>
              </div>
              <div className="flex gap-2">
                {kind === 'upstreams' && (
                  <Button variant="outline" onClick={() => void test(value.id)}>
                    {t('test')}
                  </Button>
                )}
                {kind !== 'keys' && (
                  <Button variant="outline" onClick={() => setEditing(value)}>
                    {t('edit')}
                  </Button>
                )}
                <Button
                  variant="destructive"
                  disabled={remove.isPending}
                  onClick={() => {
                    if (
                      window.confirm(
                        t(kind === 'keys' ? 'revokeConfirm' : 'deleteConfirm'),
                      )
                    )
                      remove.mutate(value.id)
                  }}
                >
                  {t(kind === 'keys' ? 'revoke' : 'delete')}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

function ObjectForm({
  fields,
  value,
  onSave,
  onCancel,
}: {
  fields: GatewayField[]
  value?: GatewayObject
  onSave: (value: Record<string, unknown>) => Promise<void>
  onCancel: () => void
}) {
  const { t } = useTranslation('contextGateway')
  const [values, setValues] = useState(() => formValues(fields, value))
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  return (
    <form
      className="grid gap-4 rounded-lg border p-4 sm:grid-cols-2"
      onSubmit={async (event) => {
        event.preventDefault()
        setSaving(true)
        setError('')
        try {
          await onSave(formPayload(fields, values))
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : t('invalidForm'))
        } finally {
          setSaving(false)
        }
      }}
    >
      {fields.map((field) => (
        <label className="grid gap-1 text-sm" key={field.name}>
          {t(`fields.${field.name}`)}
          {field.type === 'boolean' ? (
            <input
              type="checkbox"
              checked={Boolean(values[field.name])}
              onChange={(event) =>
                setValues({ ...values, [field.name]: event.target.checked })
              }
            />
          ) : field.type === 'json' ? (
            <textarea
              className="min-h-20 rounded border bg-background p-2 font-mono"
              value={String(values[field.name])}
              onChange={(event) =>
                setValues({ ...values, [field.name]: event.target.value })
              }
            />
          ) : (
            <Input
              type={
                field.type === 'password'
                  ? 'password'
                  : field.type === 'number'
                    ? 'number'
                    : 'text'
              }
              step="any"
              autoComplete="off"
              value={String(values[field.name])}
              onChange={(event) =>
                setValues({ ...values, [field.name]: event.target.value })
              }
            />
          )}
        </label>
      ))}
      {error && (
        <p role="alert" className="text-destructive sm:col-span-2">
          {error}
        </p>
      )}
      <div className="flex gap-2">
        <Button disabled={saving} type="submit">
          {t('save')}
        </Button>
        <Button variant="outline" type="button" onClick={onCancel}>
          {t('cancel')}
        </Button>
      </div>
    </form>
  )
}

function KeyForm({
  connection,
  onSave,
  onCancel,
}: {
  connection: AdminConnection
  onSave: (key: string) => void
  onCancel: () => void
}) {
  const { t } = useTranslation('contextGateway')
  const policies = useGateway(connection, 'policies')
  const upstreams = useGateway(connection, 'upstreams')
  const fields: GatewayField[] = [
    { name: 'name', initial: '' },
    { name: 'openviking_key', type: 'password', initial: '' },
    {
      name: 'policy_id',
      initial: ((policies.data ?? []) as GatewayObject[])[0]?.id ?? '',
    },
    {
      name: 'upstream_ids',
      type: 'json',
      initial: ((upstreams.data ?? []) as GatewayObject[]).map((v) => v.id),
    },
    { name: 'models', type: 'json', initial: [] },
  ]
  if (policies.isPending || upstreams.isPending) return <p>{t('loading')}</p>
  return (
    <div className="space-y-3">
      <p className="text-sm">{t('bindingOptions')}</p>
      <ul className="text-sm">
        {[
          ...((policies.data ?? []) as GatewayObject[]),
          ...((upstreams.data ?? []) as GatewayObject[]),
        ].map((v) => (
          <li key={v.id}>
            {v.name}: <code>{v.id}</code>
          </li>
        ))}
      </ul>
      <ObjectForm
        fields={fields}
        onCancel={onCancel}
        onSave={async (body) => {
          const result = await gatewayRequest<{ key: string }>(
            connection,
            'keys',
            'POST',
            body,
          )
          onSave(result.key)
        }}
      />
    </div>
  )
}
