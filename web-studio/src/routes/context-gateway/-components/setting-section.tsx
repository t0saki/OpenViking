import type * as React from 'react'
import { ChevronRightIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '#/components/ui/collapsible'
import { Switch } from '#/components/ui/switch'

type SettingSectionProps = {
  /** Anchor id for in-page navigation. */
  id?: string
  icon?: React.ReactNode
  title: string
  description?: React.ReactNode
  /** State of the header switch; omit for a section without one. */
  checked?: boolean
  onCheckedChange?: (checked: boolean) => void
  switchDisabled?: boolean
  /** Shown under the header, e.g. why the switch is disabled. */
  note?: React.ReactNode
  /** Settings inside a collapsed "Advanced settings" disclosure. */
  advanced?: React.ReactNode
  /** Settings shown while the section is on. */
  children?: React.ReactNode
}

/**
 * Card for one group of settings. With a header switch, the settings are
 * hidden while it is off; rarely changed settings go in `advanced`.
 */
export function SettingSection({
  id,
  icon,
  title,
  description,
  checked,
  onCheckedChange,
  switchDisabled,
  note,
  advanced,
  children,
}: SettingSectionProps) {
  const { t } = useTranslation('contextGateway')
  const open = checked !== false
  const hasBody = open && Boolean(children || advanced)
  return (
    <Card id={id} className="scroll-mt-20 gap-0 py-0">
      <CardHeader className="gap-1 py-5">
        <CardTitle className="flex items-center gap-2 [&_svg]:size-4 [&_svg]:text-muted-foreground">
          {icon}
          {title}
        </CardTitle>
        {description ? (
          <CardDescription className="max-w-2xl leading-6">
            {description}
          </CardDescription>
        ) : null}
        {checked !== undefined ? (
          <CardAction>
            <Switch
              checked={checked}
              disabled={switchDisabled}
              aria-label={title}
              onCheckedChange={(value) => onCheckedChange?.(value)}
            />
          </CardAction>
        ) : null}
        {note ? <div className="col-span-full pt-2 text-sm">{note}</div> : null}
      </CardHeader>
      {hasBody ? (
        <CardContent className="grid gap-5 border-t py-5">
          {children}
          {advanced ? (
            <Collapsible className="grid gap-5">
              <CollapsibleTrigger className="group/advanced flex w-fit items-center gap-1.5 rounded-sm text-sm font-medium text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none">
                <ChevronRightIcon className="size-4 transition-transform group-data-[panel-open]/advanced:rotate-90" />
                {t('field.advanced')}
              </CollapsibleTrigger>
              <CollapsibleContent className="grid gap-5">
                {advanced}
              </CollapsibleContent>
            </Collapsible>
          ) : null}
        </CardContent>
      ) : null}
    </Card>
  )
}
