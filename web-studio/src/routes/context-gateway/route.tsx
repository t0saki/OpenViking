import {
  Link,
  Outlet,
  createFileRoute,
  useNavigate,
  useRouterState,
} from '@tanstack/react-router'
import {
  ActivityIcon,
  KeyRoundIcon,
  LayoutDashboardIcon,
  PlugIcon,
  ServerIcon,
  ShieldAlertIcon,
  SlidersHorizontalIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Button } from '#/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '#/components/ui/tabs'

import { CopyButton } from './-components/copy-button'
import { LoadingState } from './-components/empty-state'
import { GatewayUnavailable } from './-components/gateway-unavailable'
import { useConnectionInfo, useGateway } from './-lib/use-gateway'

export const Route = createFileRoute('/context-gateway')({
  component: ContextGatewayLayout,
})

/** Sections in tab order; editor routes belong to their list's tab. */
const TABS = [
  { value: 'overview', to: '/context-gateway', icon: LayoutDashboardIcon },
  { value: 'upstreams', to: '/context-gateway/upstreams', icon: ServerIcon },
  {
    value: 'profiles',
    to: '/context-gateway/profiles',
    icon: SlidersHorizontalIcon,
  },
  { value: 'keys', to: '/context-gateway/keys', icon: KeyRoundIcon },
  { value: 'requests', to: '/context-gateway/requests', icon: ActivityIcon },
  { value: 'connect', to: '/context-gateway/connect', icon: PlugIcon },
] as const

export type GatewayTab = (typeof TABS)[number]['value']

/** The tab a gateway pathname belongs to (`/context-gateway/profiles/x` → profiles). */
export function gatewayTabFor(pathname: string): GatewayTab {
  const section = pathname.split('/')[2] ?? ''
  return TABS.find((tab) => tab.value === section)?.value ?? 'overview'
}

// Users & Permissions tab style; the underline sits inside the trigger so the
// list can scroll horizontally on narrow screens without clipping it.
const tabTriggerClassName =
  'h-full flex-none rounded-none border-0 px-4 text-sm data-active:bg-transparent data-active:font-semibold dark:data-active:border-transparent dark:data-active:bg-transparent focus-visible:border-transparent focus-visible:bg-transparent focus-visible:ring-0 focus-visible:outline-none focus-visible:after:h-1 focus-visible:after:bg-ring focus-visible:after:opacity-100 after:bg-primary group-data-horizontal/tabs:after:bottom-0'

function AccessDenied() {
  const { t } = useTranslation('contextGateway')
  return (
    <Card className="mx-auto mt-10 w-full max-w-xl">
      <CardHeader className="items-center text-center">
        <div className="mb-2 flex size-12 items-center justify-center rounded-xl border bg-muted/40 text-muted-foreground">
          <ShieldAlertIcon className="size-5" />
        </div>
        <CardTitle>{t('access.title')}</CardTitle>
        <CardDescription>{t('access.description')}</CardDescription>
      </CardHeader>
      <CardContent className="flex justify-center">
        <Button nativeButton={false} render={<Link to="/settings" />}>
          <KeyRoundIcon />
          {t('access.action')}
        </Button>
      </CardContent>
    </Card>
  )
}

function AddressChip({ address }: { address: string }) {
  const { t } = useTranslation('contextGateway')
  return (
    <div className="flex max-w-full min-w-0 items-center gap-2 rounded-md border bg-muted/30 py-1 pr-1 pl-3 text-sm">
      <span className="shrink-0 text-muted-foreground">
        {t('address.label')}
      </span>
      <span className="min-w-0 truncate font-mono text-xs" title={address}>
        {address}
      </span>
      <CopyButton value={address} label={t('address.copy')} />
    </div>
  )
}

/** Line tabs; clicking a tab also leaves an editor for its list. */
function SectionTabs({ activeTab }: { activeTab: GatewayTab }) {
  const { t } = useTranslation('contextGateway')
  const navigate = useNavigate()
  return (
    <Tabs value={activeTab} className="w-full min-w-0">
      <TabsList
        variant="line"
        aria-label={t('tabs.label')}
        className="w-full justify-start gap-1 overflow-x-auto overflow-y-hidden border-b border-border p-0 [scrollbar-width:none] group-data-horizontal/tabs:h-12"
      >
        {TABS.map(({ value, to, icon: Icon }) => (
          <TabsTrigger
            key={value}
            value={value}
            className={tabTriggerClassName}
            onClick={() => void navigate({ to })}
          >
            <Icon />
            {t(`tabs.${value}`)}
          </TabsTrigger>
        ))}
      </TabsList>
      {TABS.map(({ value }) => (
        <TabsContent key={value} value={value} className="pt-5">
          {activeTab === value ? <Outlet /> : null}
        </TabsContent>
      ))}
    </Tabs>
  )
}

/**
 * Gateway section layout: header with the gateway address, the access and
 * availability gates, line tabs, and the active page.
 */
export function ContextGatewayLayout() {
  const { t } = useTranslation('contextGateway')
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const { allowed, isRoleLoading } = useGateway()
  const probe = useConnectionInfo()

  if (isRoleLoading) return <LoadingState />
  if (!allowed) return <AccessDenied />

  return (
    <div className="flex w-full min-w-0 flex-col gap-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid max-w-3xl gap-1.5">
          <h1 className="text-2xl font-semibold tracking-tight">
            {t('title')}
          </h1>
          <p className="text-sm leading-6 text-muted-foreground">
            {t('description')}
          </p>
        </div>
        {probe.data ? <AddressChip address={probe.data.base_url} /> : null}
      </header>
      {probe.data ? (
        <SectionTabs activeTab={gatewayTabFor(pathname)} />
      ) : probe.isError ? (
        <GatewayUnavailable
          error={probe.error}
          retrying={probe.isFetching}
          onRetry={() => void probe.refetch()}
        />
      ) : (
        <LoadingState />
      )}
    </div>
  )
}
