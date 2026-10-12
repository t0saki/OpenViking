import type * as React from 'react'
import { Link } from '@tanstack/react-router'
import { KeyRoundIcon, RefreshCwIcon, ServerOffIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Button } from '#/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import { useAppConnection } from '#/hooks/use-app-connection'
import type { GatewayError } from '../-lib/api'
import { gatewayErrorMessage } from '../-lib/localize'
import { CodeBlock } from './code-block'

const CONFIG_FILE = 'ov.conf'

function SetupCard({
  title,
  description,
  children,
  actions,
}: {
  title: string
  description: string
  children?: React.ReactNode
  actions?: React.ReactNode
}) {
  return (
    <Card className="mx-auto mt-6 w-full max-w-2xl">
      <CardHeader className="gap-3">
        <ServerOffIcon className="size-6 text-muted-foreground" />
        <CardTitle>{title}</CardTitle>
        <CardDescription className="leading-6">{description}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        {children}
        <div className="flex flex-wrap gap-2">{actions}</div>
      </CardContent>
    </Card>
  )
}

export function GatewayConnect({ invalid = false }: { invalid?: boolean }) {
  const { t } = useTranslation('gateway')
  return (
    <SetupCard
      title={t(invalid ? 'access.invalid' : 'access.title')}
      description={t('access.description')}
      actions={
        <Button nativeButton={false} render={<Link to="/settings" />}>
          <KeyRoundIcon />
          {t('access.action')}
        </Button>
      }
    />
  )
}

export function GatewayUnavailable({
  error,
  onRetry,
  retrying = false,
}: {
  error: GatewayError
  onRetry: () => void
  retrying?: boolean
}) {
  const { t } = useTranslation('gateway')
  const { connectionRole, serverMode, openvikingUrl } = useAppConnection()
  const canConfigure =
    serverMode === 'dev' ||
    connectionRole === 'admin' ||
    connectionRole === 'root'
  const retry = (
    <Button variant="outline" size="sm" disabled={retrying} onClick={onRetry}>
      <RefreshCwIcon className={retrying ? 'animate-spin' : undefined} />
      {t('actions.retry')}
    </Button>
  )
  if (error.status === 401 && error.reason !== 'management_key_invalid')
    return <GatewayConnect invalid />
  if (error.reason === 'not_enabled' || error.status === 404)
    return (
      <SetupCard
        title={t('unavailable.notEnabled.title')}
        description={t(
          canConfigure
            ? 'unavailable.notEnabled.description'
            : 'unavailable.contact',
        )}
        actions={retry}
      >
        {canConfigure ? (
          <>
            <CodeBlock
              label={t('unavailable.terminal')}
              code="openviking-server init"
            />
            <CodeBlock
              label={CONFIG_FILE}
              code={'{"gateway":{"enabled":true}}'}
            />
          </>
        ) : null}
      </SetupCard>
    )
  if (error.reason === 'trusted_mode_unsupported')
    return (
      <SetupCard
        title={t('unavailable.trusted.title')}
        description={t('unavailable.trusted.description')}
        actions={retry}
      >
        <CodeBlock
          label={CONFIG_FILE}
          code={'{"server":{"auth_mode":"api_key"}}'}
        />
      </SetupCard>
    )
  if (error.reason === 'openviking_unreachable')
    return (
      <SetupCard
        title={t('unavailable.openvikingUnreachable.title', {
          url: openvikingUrl,
        })}
        description={t('unavailable.openvikingUnreachable.description')}
        actions={retry}
      />
    )
  return (
    <SetupCard
      title={t('unavailable.failed.title')}
      description={gatewayErrorMessage(t, error)}
      actions={retry}
    />
  )
}
