import type * as React from 'react'
import {
  CircleAlertIcon,
  ExternalLinkIcon,
  KeyRoundIcon,
  PowerOffIcon,
  RefreshCwIcon,
  UnplugIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import { Button } from '#/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'

import type { GatewayError, GatewayErrorReason } from '../-lib/api'
import { gatewayDocsUrl } from '../-lib/client-guides'
import { gatewayErrorMessage } from '../-lib/localize'
import { CodeBlock } from './code-block'

type SetupReason = Extract<
  GatewayErrorReason,
  'not_enabled' | 'token_missing' | 'unreachable'
>

const CONFIG_FILE = 'ov.conf'
const START_COMMAND =
  'openviking-context-gateway --config ~/.openviking/ov.conf'

/** Where a fix goes: a key in ov.conf, a shell command or an environment variable. */
type FixTarget = 'ovConf' | 'terminal' | 'environment'

/** Per reason: copy keys, icon, and the exact configuration to fix. */
const SETUP: Record<
  SetupReason,
  {
    copy: string
    icon: React.ReactNode
    fixes: Array<{ target: FixTarget; code: string }>
  }
> = {
  not_enabled: {
    copy: 'notEnabled',
    icon: <PowerOffIcon />,
    fixes: [
      {
        target: 'ovConf',
        code: JSON.stringify({ context_gateway: { enabled: true } }, null, 2),
      },
      { target: 'terminal', code: START_COMMAND },
    ],
  },
  token_missing: {
    copy: 'tokenMissing',
    icon: <KeyRoundIcon />,
    fixes: [
      {
        target: 'environment',
        code: 'OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN=<management-token>',
      },
    ],
  },
  unreachable: {
    copy: 'unreachable',
    icon: <UnplugIcon />,
    fixes: [
      { target: 'terminal', code: START_COMMAND },
      {
        target: 'ovConf',
        code: JSON.stringify(
          { context_gateway: { url: 'http://127.0.0.1:1935' } },
          null,
          2,
        ),
      },
    ],
  },
}

type GatewayUnavailableProps = {
  error: GatewayError
  onRetry: () => void
  retrying?: boolean
}

/**
 * What the layout shows when the gateway can't be managed: a setup card with
 * the exact fix when it is off, misconfigured or down, otherwise the error.
 */
export function GatewayUnavailable({
  error,
  onRetry,
  retrying = false,
}: GatewayUnavailableProps) {
  const { t, i18n } = useTranslation('contextGateway')
  const retry = (
    <Button
      type="button"
      variant="outline"
      size="sm"
      disabled={retrying}
      onClick={onRetry}
    >
      <RefreshCwIcon className={retrying ? 'animate-spin' : undefined} />
      {t('actions.retry')}
    </Button>
  )

  if (!error.unavailable) {
    return (
      <Alert variant="destructive">
        <CircleAlertIcon />
        <AlertTitle>{t('unavailable.failed.title')}</AlertTitle>
        <AlertDescription className="grid gap-3">
          <p>{gatewayErrorMessage(t, error)}</p>
          <div>{retry}</div>
        </AlertDescription>
      </Alert>
    )
  }

  const setup = SETUP[error.reason as SetupReason]
  return (
    <Card className="mx-auto w-full max-w-2xl">
      <CardHeader className="gap-3">
        <div className="flex size-11 items-center justify-center rounded-xl border bg-muted/40 text-muted-foreground [&_svg]:size-5">
          {setup.icon}
        </div>
        <CardTitle className="text-lg">
          {t(`unavailable.${setup.copy}.title`)}
        </CardTitle>
        <CardDescription className="leading-6">
          {t(`unavailable.${setup.copy}.description`)}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="grid gap-2">
          <p className="text-sm font-medium">{t('unavailable.fixLabel')}</p>
          {setup.fixes.map((fix) => (
            <CodeBlock
              key={fix.code}
              label={
                fix.target === 'ovConf'
                  ? CONFIG_FILE
                  : t(`unavailable.${fix.target}`)
              }
              code={fix.code}
            />
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {retry}
          <Button
            variant="ghost"
            size="sm"
            nativeButton={false}
            render={
              <a
                href={gatewayDocsUrl('operations', i18n.resolvedLanguage)}
                target="_blank"
                rel="noreferrer"
              />
            }
          >
            <ExternalLinkIcon />
            {t('unavailable.docs')}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
