import * as React from 'react'
import {
  ChevronRightIcon,
  CircleAlertIcon,
  RefreshCcwIcon,
  RotateCcwIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Button } from '#/components/ui/button'
import {
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'
import { cn } from '#/lib/utils'

import type { GatewayKey, LogRecord, Upstream } from '../-lib/api'
import {
  EMPTY_VALUE,
  formatCompact,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatPercent,
  formatRelativeTime,
  shortenId,
} from '../-lib/format'
import {
  captureReasonLabel,
  degradationInfo,
  recallReasonLabel,
  toolSkipReasonLabel,
  toolStopReasonLabel,
} from '../-lib/localize'
import type { Translate } from '../-lib/localize'
import { CopyButton } from './copy-button'
import { Notice } from './notice'
import {
  CaptureStatusBadge,
  HttpStatusBadge,
  IssueBadge,
  KindBadge,
  ProtocolBadge,
} from './status-badges'

/** Recall outcomes that are not failures. */
const RECALL_OUTCOMES = new Set(['recalled', 'empty', 'disabled'])

function recallFailed(record: LogRecord): boolean {
  return Boolean(
    record.recall_reason && !RECALL_OUTCOMES.has(record.recall_reason),
  )
}

/** Records that name a conversation and the key that used it can be resynced. */
function canResync(record: LogRecord): boolean {
  return Boolean(record.session && record.protocol && record.credential_id)
}

function Muted({ children }: { children: React.ReactNode }) {
  return <span className="text-muted-foreground">{children}</span>
}

/** Compact visible text with a full description for hover and screen readers. */
function Described({
  description,
  className,
  children,
}: {
  description: string
  className?: string
  children: React.ReactNode
}) {
  return (
    <span title={description} className={className}>
      <span aria-hidden className="inline-flex items-center gap-1">
        {children}
      </span>
      <span className="sr-only">{description}</span>
    </span>
  )
}

type TokenCounts = {
  input?: number
  cached?: number
  cacheWrite?: number
  output?: number
}

/** "Input 12,000 · cached 9,000 (75%) · cache write 0 · output 300". */
function tokenLine(t: Translate, counts: TokenCounts, locale?: string) {
  const value = (count?: number) =>
    count === undefined ? EMPTY_VALUE : formatNumber(count, locale)
  const cached =
    counts.cached !== undefined && counts.input
      ? `${value(counts.cached)} (${formatPercent(counts.cached / counts.input, locale)})`
      : value(counts.cached)
  return t('requests.details.tokenLine', {
    input: value(counts.input),
    cached,
    cacheWrite: value(counts.cacheWrite),
    output: value(counts.output),
  })
}

function TokensCell({ record }: { record: LogRecord }) {
  const { t, i18n } = useTranslation('contextGateway')
  const locale = i18n.resolvedLanguage
  const {
    input_tokens: input,
    cached_tokens: cached,
    output_tokens: output,
  } = record
  if (input === undefined && output === undefined) {
    return <Muted>{EMPTY_VALUE}</Muted>
  }
  const parts = [
    input === undefined
      ? null
      : t('requests.tokens.input', { value: formatCompact(input, locale) }),
    cached === undefined || !input
      ? null
      : t('requests.tokens.cached', {
          value: formatPercent(cached / input, locale),
        }),
    output === undefined
      ? null
      : t('requests.tokens.output', { value: formatCompact(output, locale) }),
  ].filter((part): part is string => Boolean(part))
  const full = tokenLine(
    t,
    { input, cached, cacheWrite: record.cache_write_tokens, output },
    locale,
  )
  return (
    <Described description={full} className="text-xs tabular-nums">
      {parts.join(' · ')}
    </Described>
  )
}

/**
 * Memory of one request: entries recalled (+N with the recall time), a failed
 * recall, and earlier memory kept in the history (↺ N); a dash when none,
 * unless `hideEmpty` because something else fills the cell.
 */
export function MemoryCell({
  record,
  hideEmpty = false,
}: {
  record: LogRecord
  hideEmpty?: boolean
}) {
  const { t, i18n } = useTranslation('contextGateway')
  const locale = i18n.resolvedLanguage
  const recalled = record.recall_count ?? 0
  const replayed = record.replay_hits ?? 0
  const failed = recallFailed(record)
  if (!recalled && !replayed && !failed) {
    return hideEmpty ? null : <Muted>{EMPTY_VALUE}</Muted>
  }
  return (
    <div className="flex items-center gap-3 text-xs tabular-nums">
      {recalled > 0 ? (
        <Described
          description={t('requests.memory.recalled', {
            count: recalled,
            duration: formatDuration(record.recall_ms ?? 0, locale),
          })}
        >
          <span className="font-medium text-emerald-700 dark:text-emerald-300">
            +{recalled}
          </span>
          <span className="text-muted-foreground">
            {formatDuration(record.recall_ms ?? 0, locale)}
          </span>
        </Described>
      ) : null}
      {failed ? (
        <Described
          description={recallReasonLabel(t, record.recall_reason)}
          className="text-amber-700 dark:text-amber-300"
        >
          <CircleAlertIcon className="size-3.5" />
          {t('requests.memory.recallFailed')}
        </Described>
      ) : null}
      {replayed > 0 ? (
        <Described
          description={t('requests.memory.replayed', { count: replayed })}
          className="text-muted-foreground"
        >
          <RotateCcwIcon className="size-3" />
          {replayed}
        </Described>
      ) : null}
    </div>
  )
}

/** Column headers of the request table, kept next to the row that fills them. */
export function RequestsTableHeader() {
  const { t } = useTranslation('contextGateway')
  return (
    <TableHeader>
      <TableRow className="bg-muted/20 hover:bg-muted/20">
        <TableHead>{t('requests.table.time')}</TableHead>
        <TableHead>{t('requests.table.type')}</TableHead>
        <TableHead>{t('requests.table.model')}</TableHead>
        <TableHead>{t('requests.table.status')}</TableHead>
        <TableHead>{t('requests.table.tokens')}</TableHead>
        <TableHead>{t('requests.table.memory')}</TableHead>
        <TableHead>{t('requests.table.saving')}</TableHead>
        <TableHead>{t('requests.table.issue')}</TableHead>
        <TableHead className="w-10">
          <span className="sr-only">{t('requests.table.details')}</span>
        </TableHead>
      </TableRow>
    </TableHeader>
  )
}

const COLUMN_COUNT = 9

type RequestRowProps = {
  record: LogRecord
  /** Upstreams by id; undefined until the list has loaded. */
  upstreams?: Map<string, Upstream>
  /** Gateway keys by id; undefined until the list has loaded. */
  keys?: Map<string, GatewayKey>
  onResync: (record: LogRecord) => void
}

/** One request-log entry with an expandable detail panel. */
export function RequestRow({
  record,
  upstreams,
  keys,
  onResync,
}: RequestRowProps) {
  const { t, i18n } = useTranslation('contextGateway')
  const locale = i18n.resolvedLanguage
  const [expanded, setExpanded] = React.useState(false)
  const detailsId = React.useId()
  const toggle = () => setExpanded((value) => !value)

  return (
    <>
      <TableRow
        className={cn('cursor-pointer', expanded && 'border-b-0')}
        onClick={toggle}
      >
        <TableCell
          className="text-muted-foreground tabular-nums"
          title={formatDateTime(record.time, locale, true)}
        >
          {formatRelativeTime(record.time, locale)}
        </TableCell>
        <TableCell>
          <KindBadge kind={record.kind} />
        </TableCell>
        <TableCell
          className="max-w-48 truncate font-mono text-xs"
          title={record.model}
        >
          {record.model || <Muted>{EMPTY_VALUE}</Muted>}
        </TableCell>
        <TableCell>
          <HttpStatusBadge status={record.status} />
        </TableCell>
        <TableCell>
          <TokensCell record={record} />
        </TableCell>
        <TableCell>
          <MemoryCell record={record} />
        </TableCell>
        <TableCell>
          <CaptureStatusBadge
            status={record.capture_status}
            reason={record.capture_reason}
          />
        </TableCell>
        <TableCell>
          {record.degradation ? (
            <IssueBadge degradation={record.degradation} />
          ) : null}
        </TableCell>
        <TableCell className="text-right">
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            aria-expanded={expanded}
            aria-controls={detailsId}
            aria-label={t(
              expanded ? 'requests.details.hide' : 'requests.details.show',
            )}
            onClick={(event) => {
              event.stopPropagation()
              toggle()
            }}
          >
            <ChevronRightIcon
              className={cn('transition-transform', expanded && 'rotate-90')}
            />
          </Button>
        </TableCell>
      </TableRow>
      {expanded ? (
        <TableRow id={detailsId} className="bg-muted/20 hover:bg-muted/20">
          <TableCell
            colSpan={COLUMN_COUNT}
            className="px-4 py-4 whitespace-normal"
          >
            <RequestDetails
              record={record}
              upstreams={upstreams}
              keys={keys}
              onResync={onResync}
            />
          </TableCell>
        </TableRow>
      ) : null}
    </>
  )
}

function DetailItem({
  label,
  className,
  children,
}: {
  label: string
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={cn('grid min-w-0 content-start gap-1', className)}>
      <dt className="text-xs font-medium text-muted-foreground">{label}</dt>
      <dd className="grid min-w-0 gap-1 text-sm break-words">{children}</dd>
    </div>
  )
}

/** What went wrong with a request and what to do about it. */
function Callout({
  tone,
  title,
  happened,
  action,
}: {
  tone: 'warning' | 'danger'
  title?: string
  happened?: string
  action?: string
}) {
  const { t } = useTranslation('contextGateway')
  return (
    <Notice tone={tone} title={title}>
      <dl className="grid gap-x-4 gap-y-1.5 text-foreground sm:grid-cols-[8rem_minmax(0,1fr)]">
        {happened ? (
          <>
            <dt className="text-muted-foreground">
              {t('requests.details.whatHappened')}
            </dt>
            <dd>{happened}</dd>
          </>
        ) : null}
        {action ? (
          <>
            <dt className="text-muted-foreground">
              {t('requests.details.whatToDo')}
            </dt>
            <dd>{action}</dd>
          </>
        ) : null}
      </dl>
    </Notice>
  )
}

function hasTokens(record: LogRecord): boolean {
  return (
    record.input_tokens !== undefined ||
    record.output_tokens !== undefined ||
    record.cached_tokens !== undefined
  )
}

function TokenDetails({ record }: { record: LogRecord }) {
  const { t, i18n } = useTranslation('contextGateway')
  const locale = i18n.resolvedLanguage
  if (!hasTokens(record)) return <Muted>{t('requests.details.noUsage')}</Muted>
  const total = tokenLine(
    t,
    {
      input: record.input_tokens,
      cached: record.cached_tokens,
      cacheWrite: record.cache_write_tokens,
      output: record.output_tokens,
    },
    locale,
  )
  if (record.first_upstream_input_tokens === undefined) {
    return <span className="tabular-nums">{total}</span>
  }
  const first = tokenLine(
    t,
    {
      input: record.first_upstream_input_tokens,
      cached: record.first_upstream_cached_tokens,
      cacheWrite: record.first_upstream_cache_write_tokens,
      output: record.first_upstream_output_tokens,
    },
    locale,
  )
  const hidden = tokenLine(
    t,
    {
      input: record.hidden_upstream_input_tokens,
      cached: record.hidden_upstream_cached_tokens,
      cacheWrite: record.hidden_upstream_cache_write_tokens,
      output: record.hidden_upstream_output_tokens,
    },
    locale,
  )
  return (
    <div className="grid gap-1 tabular-nums">
      <span>{total}</span>
      <span className="text-xs text-muted-foreground">
        {t('requests.details.firstCall')}: {first}
      </span>
      <span className="text-xs text-muted-foreground">
        {t('requests.details.toolCalls', {
          count: record.hidden_upstream_calls ?? 0,
        })}
        : {hidden}
      </span>
    </div>
  )
}

function RequestDetails({
  record,
  upstreams,
  keys,
  onResync,
}: Omit<RequestRowProps, 'record'> & { record: LogRecord }) {
  const { t, i18n } = useTranslation('contextGateway')
  const locale = i18n.resolvedLanguage
  const upstream = record.upstream_id
    ? upstreams?.get(record.upstream_id)
    : undefined
  const gatewayKey = record.credential_id
    ? keys?.get(record.credential_id)
    : undefined
  const keyRevoked = Boolean(keys && record.credential_id && !gatewayKey)
  const isCapture = record.kind === 'capture'
  const degradation = record.degradation
    ? degradationInfo(t, record.degradation)
    : undefined
  const httpError =
    !degradation && record.status !== undefined && record.status >= 400
  const hasTools = Boolean(
    record.tool_skip_reason ||
    record.tool_stop_reason ||
    (record.hidden_rounds ?? 0) > 0,
  )

  return (
    <div className="grid gap-4">
      <dl className="grid gap-x-6 gap-y-4 sm:grid-cols-2 xl:grid-cols-3">
        <DetailItem label={t('requests.details.time')}>
          <span className="tabular-nums">
            {formatDateTime(record.time, locale, true)}
          </span>
        </DetailItem>
        {record.request_id ? (
          <DetailItem label={t('requests.details.requestId')}>
            <span className="flex min-w-0 items-center gap-1">
              <span
                className="truncate font-mono text-xs"
                title={record.request_id}
              >
                {record.request_id}
              </span>
              <CopyButton
                value={record.request_id}
                label={t('requests.details.copyRequestId')}
              />
            </span>
          </DetailItem>
        ) : null}
        {record.session ? (
          <DetailItem label={t('requests.details.conversation')}>
            <span className="flex min-w-0 items-center gap-1">
              <span
                className="truncate font-mono text-xs"
                title={record.session}
              >
                {shortenId(record.session, 16)}
              </span>
              <CopyButton
                value={record.session}
                label={t('requests.details.copyConversation')}
              />
            </span>
            {record.session.startsWith('anonymous-') ? (
              <span className="text-xs text-muted-foreground">
                {t('requests.details.anonymous')}
              </span>
            ) : null}
          </DetailItem>
        ) : null}
        {record.protocol ? (
          <DetailItem label={t('requests.details.protocol')}>
            <span>
              <ProtocolBadge protocol={record.protocol} />
            </span>
          </DetailItem>
        ) : null}
        {record.upstream_id ? (
          <DetailItem label={t('requests.details.upstream')}>
            {upstream ? (
              <span className="font-medium">{upstream.name}</span>
            ) : (
              <span
                className="text-muted-foreground"
                title={record.upstream_id}
              >
                {upstreams
                  ? t('requests.details.upstreamDeleted')
                  : shortenId(record.upstream_id)}
              </span>
            )}
          </DetailItem>
        ) : null}
        {record.credential_id ? (
          <DetailItem label={t('requests.details.key')}>
            {gatewayKey ? (
              <span className="flex min-w-0 items-baseline gap-2">
                <span className="truncate font-medium">{gatewayKey.name}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  {gatewayKey.prefix}…
                </span>
              </span>
            ) : (
              <span
                className="text-muted-foreground"
                title={record.credential_id}
              >
                {keyRevoked
                  ? t('requests.details.keyRevoked')
                  : shortenId(record.credential_id)}
              </span>
            )}
          </DetailItem>
        ) : null}
        {record.duration_ms !== undefined ? (
          <DetailItem label={t('requests.details.duration')}>
            <span
              className="tabular-nums"
              title={t('requests.details.durationHint')}
            >
              {formatDuration(record.duration_ms, locale)}
            </span>
          </DetailItem>
        ) : null}
        {!isCapture ? (
          <DetailItem
            label={t('requests.details.tokens')}
            className="sm:col-span-2"
          >
            <TokenDetails record={record} />
          </DetailItem>
        ) : null}
        {record.cache_eligible !== undefined ? (
          <DetailItem label={t('requests.details.cache')}>
            {t(
              record.cache_eligible
                ? 'requests.details.cacheEligible'
                : 'requests.details.cacheIneligible',
              { min: formatNumber(record.cache_min_tokens ?? 0, locale) },
            )}
          </DetailItem>
        ) : null}
        {record.recall_reason || (record.replay_hits ?? 0) > 0 ? (
          <DetailItem label={t('requests.details.recall')}>
            {record.recall_reason ? (
              <span
                className={cn(
                  recallFailed(record) && 'text-amber-700 dark:text-amber-300',
                )}
              >
                {recallReasonLabel(t, record.recall_reason)}
                {record.recall_reason === 'recalled'
                  ? ` · ${t('requests.details.recallResult', {
                      count: record.recall_count ?? 0,
                      duration: formatDuration(record.recall_ms ?? 0, locale),
                    })}`
                  : null}
              </span>
            ) : null}
            {(record.replay_hits ?? 0) > 0 ? (
              <span className="text-xs text-muted-foreground">
                {t('requests.memory.replayed', { count: record.replay_hits })}
              </span>
            ) : null}
          </DetailItem>
        ) : null}
        {record.archive_replayed ? (
          <DetailItem label={t('requests.details.summary')}>
            {t('requests.details.summaryUsed')}
          </DetailItem>
        ) : null}
        {record.capture_status ? (
          <DetailItem label={t('requests.details.saving')}>
            <span>
              <CaptureStatusBadge status={record.capture_status} />
            </span>
            {record.capture_reason ? (
              <span className="text-xs text-muted-foreground">
                {captureReasonLabel(t, record.capture_reason)}
              </span>
            ) : null}
            {record.capture_retry_at ? (
              <span className="text-xs text-muted-foreground">
                {t('requests.details.nextRetry', {
                  time: formatDateTime(record.capture_retry_at, locale),
                })}
              </span>
            ) : null}
          </DetailItem>
        ) : null}
        {hasTools ? (
          <DetailItem label={t('requests.details.tools')}>
            {record.tool_skip_reason ? (
              <span>
                {t('requests.details.toolsSkipped', {
                  reason: toolSkipReasonLabel(t, record.tool_skip_reason),
                })}
              </span>
            ) : null}
            {(record.hidden_rounds ?? 0) > 0 ? (
              <span className="tabular-nums">
                {t('requests.details.toolRounds', {
                  rounds: record.hidden_rounds,
                  calls: record.hidden_upstream_calls ?? 0,
                  tokens: formatNumber(record.hidden_added_tokens ?? 0, locale),
                })}
              </span>
            ) : null}
            {record.tool_stop_reason ? (
              <span className="text-amber-700 dark:text-amber-300">
                {toolStopReasonLabel(t, record.tool_stop_reason)}
              </span>
            ) : null}
          </DetailItem>
        ) : null}
      </dl>
      {degradation ? (
        <Callout
          tone="warning"
          title={degradation.label}
          happened={degradation.explanation}
          action={degradation.action}
        />
      ) : null}
      {httpError ? (
        <Callout
          tone="danger"
          happened={t('requests.details.httpError', { status: record.status })}
          action={t('requests.details.httpErrorAction')}
        />
      ) : null}
      {canResync(record) ? (
        <div className="flex flex-wrap items-center gap-3 border-t pt-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={keyRevoked}
            onClick={() => onResync(record)}
          >
            <RefreshCcwIcon />
            {t('requests.resync.action')}
          </Button>
          <p className="max-w-2xl text-xs leading-5 text-muted-foreground">
            {keyRevoked
              ? t('requests.resync.keyRevoked')
              : t('requests.resync.hint')}
          </p>
        </div>
      ) : null}
    </div>
  )
}
