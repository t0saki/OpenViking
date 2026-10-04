import type * as React from 'react'
import { Link } from '@tanstack/react-router'
import { ActivityIcon, ArrowRightIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Button } from '#/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'

import type { GatewayError, LogRecord } from '../-lib/api'
import { EMPTY_VALUE, formatDateTime, formatRelativeTime } from '../-lib/format'
import { EmptyState, ErrorState, LoadingState } from './empty-state'
import { MemoryCell } from './requests-row'
import {
  CaptureStatusBadge,
  HttpStatusBadge,
  IssueBadge,
  KindBadge,
} from './status-badges'

/** Memory column: saving status for memory sync, else memory and any degradation. */
function RecentMemory({ log }: { log: LogRecord }) {
  if (log.kind === 'capture') {
    return (
      <CaptureStatusBadge
        status={log.capture_status}
        reason={log.capture_reason}
      />
    )
  }
  return (
    <div className="flex items-center gap-3">
      <MemoryCell record={log} hideEmpty={Boolean(log.degradation)} />
      {log.degradation ? <IssueBadge degradation={log.degradation} /> : null}
    </div>
  )
}

type RecentRequestsProps = {
  logs?: LogRecord[]
  error: GatewayError | null
}

/** The newest request-log records in a compact table, with a link to all. */
export function RecentRequests({ logs, error }: RecentRequestsProps) {
  const { t, i18n } = useTranslation('contextGateway')
  const locale = i18n.resolvedLanguage

  let body: React.ReactNode
  if (logs === undefined) {
    body = error ? (
      <ErrorState
        className="min-h-40"
        title={t('requests.loadFailed')}
        error={error}
      />
    ) : (
      <LoadingState className="min-h-40" />
    )
  } else if (logs.length === 0) {
    body = (
      <EmptyState
        className="min-h-40"
        icon={<ActivityIcon />}
        title={t('overview.recent.empty.title')}
        description={t('overview.recent.empty.description')}
      />
    )
  } else {
    body = (
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/20 hover:bg-muted/20">
            <TableHead className="pl-5">
              {t('overview.recent.columns.time')}
            </TableHead>
            <TableHead>{t('overview.recent.columns.type')}</TableHead>
            <TableHead>{t('overview.recent.columns.model')}</TableHead>
            <TableHead>{t('overview.recent.columns.status')}</TableHead>
            <TableHead className="pr-5">
              {t('overview.recent.columns.memory')}
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {logs.map((log, index) => (
            <TableRow key={log.request_id ?? `${log.time}-${index}`}>
              <TableCell
                className="pl-5 text-muted-foreground"
                title={formatDateTime(log.time, locale, true)}
              >
                {formatRelativeTime(log.time, locale)}
              </TableCell>
              <TableCell>
                <KindBadge kind={log.kind} />
              </TableCell>
              <TableCell
                className="max-w-56 truncate font-mono text-xs"
                title={log.model}
              >
                {log.model ?? EMPTY_VALUE}
              </TableCell>
              <TableCell>
                <HttpStatusBadge status={log.status} />
              </TableCell>
              <TableCell className="pr-5">
                <RecentMemory log={log} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    )
  }

  return (
    <Card className="gap-0 overflow-hidden py-0">
      <CardHeader className="border-b px-5 py-3 [.border-b]:pb-3">
        <CardTitle className="self-center">
          {t('overview.recent.title')}
        </CardTitle>
        <CardAction className="self-center">
          <Button
            size="sm"
            variant="ghost"
            nativeButton={false}
            render={<Link to="/context-gateway/requests" />}
          >
            {t('actions.viewAll')}
            <ArrowRightIcon />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="p-0">{body}</CardContent>
    </Card>
  )
}
