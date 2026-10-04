import type * as React from 'react'

import { Label } from '#/components/ui/label'
import { Switch } from '#/components/ui/switch'

type ToggleRowProps = {
  id: string
  label: string
  description: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  /** Dependent settings, shown below the row while it is on. */
  children?: React.ReactNode
}

/** Bordered row with a label, a one-line description and a switch. */
export function ToggleRow({
  id,
  label,
  description,
  checked,
  onCheckedChange,
  children,
}: ToggleRowProps) {
  return (
    <div className="grid rounded-lg border">
      <div className="flex items-start justify-between gap-4 px-4 py-3">
        <div className="grid gap-1">
          <Label htmlFor={id}>{label}</Label>
          <p className="text-xs leading-5 text-muted-foreground">
            {description}
          </p>
        </div>
        <Switch
          id={id}
          checked={checked}
          aria-label={label}
          onCheckedChange={(value) => onCheckedChange(value)}
        />
      </div>
      {checked && children ? (
        <div className="grid gap-4 border-t p-4">{children}</div>
      ) : null}
    </div>
  )
}
