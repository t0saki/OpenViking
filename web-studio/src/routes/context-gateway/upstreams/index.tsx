import * as React from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'
import {
  EllipsisIcon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
  ServerIcon,
  Trash2Icon,
  TriangleAlertIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Card, CardContent } from '#/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '#/components/ui/dropdown-menu'
import { Switch } from '#/components/ui/switch'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'

import {
  EmptyState,
  ErrorState,
  LoadingState,
} from '../-components/empty-state'
import { SectionHeader } from '../-components/section-header'
import { ProtocolBadge, ToneBadge } from '../-components/status-badges'
import {
  DeleteUpstreamDialog,
  UpstreamTest,
  keysUsing,
} from '../-components/upstreams-actions'
import { saveUpstream } from '../-lib/api'
import type { Upstream } from '../-lib/api'
import { EMPTY_VALUE, formatNumber, hostFromUrl } from '../-lib/format'
import {
  authModeLabel,
  gatewayErrorMessage,
  vendorLabel,
} from '../-lib/localize'
import { NEW_ID } from '../-lib/search'
import { toUpstreamInput } from '../-lib/upstream-schema'
import {
  gatewayQueryKey,
  useGateway,
  useKeys,
  useUpstreams,
} from '../-lib/use-gateway'

export const Route = createFileRoute('/context-gateway/upstreams/')({
  component: UpstreamsPage,
})

/** Models shown as chips before the rest collapse into "+N". */
const VISIBLE_MODELS = 3

/**
 * Turns an upstream on or off by saving it in full. The list updates at once
 * and rolls back with a toast if the gateway refuses.
 */
function useToggleUpstream() {
  const { t } = useTranslation('contextGateway')
  const { connection, scope } = useGateway()
  const queryClient = useQueryClient()
  const queryKey = gatewayQueryKey(scope, 'upstreams')
  return useMutation({
    mutationFn: ({
      upstream,
      enabled,
    }: {
      upstream: Upstream
      enabled: boolean
    }) =>
      saveUpstream(connection, upstream.id, {
        ...toUpstreamInput(upstream),
        enabled,
      }),
    onMutate: async ({ upstream, enabled }) => {
      await queryClient.cancelQueries({ queryKey })
      const previous = queryClient.getQueryData<Upstream[]>(queryKey)
      queryClient.setQueryData<Upstream[]>(queryKey, (list) =>
        list?.map((item) =>
          item.id === upstream.id ? { ...item, enabled } : item,
        ),
      )
      return { previous }
    },
    onError: (error, _variables, context) => {
      if (context?.previous) {
        queryClient.setQueryData(queryKey, context.previous)
      }
      toast.error(gatewayErrorMessage(t, error))
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey }),
  })
}

/** Model providers the gateway forwards requests to. */
export function UpstreamsPage() {
  const { t } = useTranslation('contextGateway')
  const upstreams = useUpstreams()
  const keys = useKeys()
  const toggle = useToggleUpstream()
  const [deleting, setDeleting] = React.useState<Upstream | null>(null)

  const rows = React.useMemo(
    () =>
      [...(upstreams.data ?? [])].sort((a, b) => a.name.localeCompare(b.name)),
    [upstreams.data],
  )
  const refreshing = upstreams.isFetching || keys.isFetching

  const addButton = (
    <Button
      size="sm"
      nativeButton={false}
      render={
        <Link
          to="/context-gateway/upstreams/$upstreamId"
          params={{ upstreamId: NEW_ID }}
        />
      }
    >
      <PlusIcon />
      {t('upstreams.add')}
    </Button>
  )

  let body: React.ReactNode
  if (upstreams.isPending) {
    body = <LoadingState />
  } else if (upstreams.isError) {
    body = (
      <ErrorState
        title={t('upstreams.loadFailed')}
        error={upstreams.error}
        retrying={upstreams.isFetching}
        onRetry={() => void upstreams.refetch()}
      />
    )
  } else if (rows.length === 0) {
    body = (
      <EmptyState
        icon={<ServerIcon />}
        title={t('upstreams.empty.title')}
        description={t('upstreams.empty.description')}
        action={addButton}
      />
    )
  } else {
    body = (
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/20 hover:bg-muted/20">
            <TableHead className="pl-4">
              {t('upstreams.columns.name')}
            </TableHead>
            <TableHead>{t('upstreams.columns.protocol')}</TableHead>
            <TableHead>{t('upstreams.columns.provider')}</TableHead>
            <TableHead>{t('upstreams.columns.models')}</TableHead>
            <TableHead>{t('upstreams.columns.credentials')}</TableHead>
            <TableHead className="text-right">
              {t('upstreams.columns.priority')}
            </TableHead>
            <TableHead>{t('upstreams.columns.usedBy')}</TableHead>
            <TableHead>{t('upstreams.columns.enabled')}</TableHead>
            <TableHead className="pr-4 text-right">
              {t('upstreams.columns.actions')}
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((upstream) => (
            <UpstreamRow
              key={upstream.id}
              upstream={upstream}
              usedBy={keysUsing(keys.data, upstream.id)}
              toggling={
                toggle.isPending && toggle.variables.upstream.id === upstream.id
              }
              onToggle={(enabled) => toggle.mutate({ upstream, enabled })}
              onDelete={() => setDeleting(upstream)}
            />
          ))}
        </TableBody>
      </Table>
    )
  }

  return (
    <div className="flex w-full min-w-0 flex-col gap-5">
      <SectionHeader
        title={t('upstreams.title')}
        description={t('upstreams.description')}
        actions={
          <>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={refreshing}
              onClick={() => {
                void upstreams.refetch()
                void keys.refetch()
              }}
            >
              <RefreshCwIcon
                className={refreshing ? 'animate-spin' : undefined}
              />
              {t('actions.refresh')}
            </Button>
            {addButton}
          </>
        }
      />
      <Card className="gap-0 overflow-hidden py-0">
        <CardContent className="p-0">{body}</CardContent>
      </Card>
      <DeleteUpstreamDialog
        upstream={deleting}
        onOpenChange={(open) => {
          if (!open) setDeleting(null)
        }}
      />
    </div>
  )
}

type UpstreamRowProps = {
  upstream: Upstream
  /** Keys that use this upstream; undefined while keys are loading. */
  usedBy: number | undefined
  toggling: boolean
  onToggle: (enabled: boolean) => void
  onDelete: () => void
}

function UpstreamRow({
  upstream,
  usedBy,
  toggling,
  onToggle,
  onDelete,
}: UpstreamRowProps) {
  const { t, i18n } = useTranslation('contextGateway')
  const editLink = (
    <Link
      to="/context-gateway/upstreams/$upstreamId"
      params={{ upstreamId: upstream.id }}
    />
  )
  return (
    <TableRow>
      <TableCell className="pl-4">
        <div className="grid min-w-0 gap-0.5">
          <Link
            to="/context-gateway/upstreams/$upstreamId"
            params={{ upstreamId: upstream.id }}
            className="max-w-56 truncate font-medium hover:underline"
            title={upstream.name}
          >
            {upstream.name}
          </Link>
          <span
            className="max-w-56 truncate font-mono text-xs text-muted-foreground"
            title={upstream.base_url}
          >
            {hostFromUrl(upstream.base_url)}
          </span>
        </div>
      </TableCell>
      <TableCell>
        <ProtocolBadge protocol={upstream.protocol} />
      </TableCell>
      <TableCell className="text-sm">
        {upstream.vendor === 'generic' ? (
          <span className="text-muted-foreground">{EMPTY_VALUE}</span>
        ) : (
          vendorLabel(t, upstream.vendor)
        )}
      </TableCell>
      <TableCell>
        <ModelsSummary upstream={upstream} />
      </TableCell>
      <TableCell>
        <CredentialsSummary upstream={upstream} />
      </TableCell>
      <TableCell className="text-right font-mono text-xs tabular-nums">
        {formatNumber(upstream.priority, i18n.resolvedLanguage)}
      </TableCell>
      <TableCell className="text-sm">
        {usedBy === undefined ? (
          <span className="text-muted-foreground">{EMPTY_VALUE}</span>
        ) : usedBy === 0 ? (
          <span className="text-muted-foreground">
            {t('upstreams.notUsed')}
          </span>
        ) : (
          t('upstreams.usedBy', { count: usedBy })
        )}
      </TableCell>
      <TableCell>
        <Switch
          size="sm"
          checked={upstream.enabled}
          disabled={toggling}
          aria-label={t(
            upstream.enabled
              ? 'upstreams.toggle.disable'
              : 'upstreams.toggle.enable',
            { name: upstream.name },
          )}
          onCheckedChange={(checked) => onToggle(checked)}
        />
      </TableCell>
      <TableCell className="pr-4">
        <div className="flex items-center justify-end gap-1">
          <UpstreamTest upstream={upstream} />
          <Button
            size="sm"
            variant="ghost"
            className="h-8 px-2 text-xs"
            nativeButton={false}
            render={editLink}
          >
            <PencilIcon />
            {t('actions.edit')}
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <Button
                  type="button"
                  size="icon-sm"
                  variant="ghost"
                  aria-label={t('actions.more')}
                  title={t('actions.more')}
                />
              }
            >
              <EllipsisIcon />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-auto min-w-48">
              <DropdownMenuItem
                variant="destructive"
                disabled={Boolean(usedBy)}
                className="items-start"
                onClick={onDelete}
              >
                <Trash2Icon className="mt-0.5" />
                <span className="grid gap-0.5">
                  <span>{t('upstreams.delete.action')}</span>
                  {usedBy ? (
                    <span className="text-xs">
                      {t('upstreams.delete.blocked', { count: usedBy })}
                    </span>
                  ) : null}
                </span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </TableCell>
    </TableRow>
  )
}

/** Up to three model chips, "+N" for the rest, or "Any model"; alias count after. */
function ModelsSummary({ upstream }: { upstream: Upstream }) {
  const { t } = useTranslation('contextGateway')
  const { models, aliases } = upstream
  const shown = models.slice(0, VISIBLE_MODELS)
  const hidden = models.slice(VISIBLE_MODELS)
  const aliasNames = Object.keys(aliases)
  return (
    <div className="flex max-w-80 flex-wrap items-center gap-1">
      {models.length === 0 ? (
        <span className="text-xs text-muted-foreground">
          {t('upstreams.models.any')}
        </span>
      ) : (
        shown.map((model) => (
          <Badge
            key={model}
            variant="secondary"
            className="max-w-40 font-mono font-normal"
            title={model}
          >
            <span className="truncate">{model}</span>
          </Badge>
        ))
      )}
      {hidden.length ? (
        <Badge
          variant="outline"
          className="font-normal text-muted-foreground"
          title={hidden.join(', ')}
        >
          {t('upstreams.models.more', { count: hidden.length })}
        </Badge>
      ) : null}
      {aliasNames.length ? (
        <span
          className="text-xs text-muted-foreground"
          title={aliasNames
            .map((name) => `${name} → ${aliases[name]}`)
            .join('\n')}
        >
          {t('upstreams.models.aliases', { count: aliasNames.length })}
        </span>
      ) : null}
    </div>
  )
}

/** Who provides the provider key, with warnings that block every request. */
function CredentialsSummary({ upstream }: { upstream: Upstream }) {
  const { t } = useTranslation('contextGateway')
  const keyMissing = upstream.auth_mode === 'managed' && !upstream.has_api_key
  const blocked = upstream.coding_plan && !upstream.allow_coding_plan
  return (
    <div className="flex flex-col items-start gap-1">
      {keyMissing ? (
        <ToneBadge
          tone="warning"
          title={t('upstreams.credentials.keyMissingHint')}
        >
          <TriangleAlertIcon />
          {t('upstreams.credentials.keyMissing')}
        </ToneBadge>
      ) : (
        <span className="text-sm">{authModeLabel(t, upstream.auth_mode)}</span>
      )}
      {blocked ? (
        <ToneBadge tone="danger" title={t('upstreams.credentials.blockedHint')}>
          <TriangleAlertIcon />
          {t('upstreams.credentials.blocked')}
        </ToneBadge>
      ) : null}
    </div>
  )
}
