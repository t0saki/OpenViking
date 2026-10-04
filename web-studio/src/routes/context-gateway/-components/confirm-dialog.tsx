import * as React from 'react'
import { LoaderCircleIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '#/components/ui/alert-dialog'

type ConfirmDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** A question, e.g. "Revoke “Laptop”?". */
  title: string
  /** What happens and what is kept. */
  description: React.ReactNode
  confirmLabel: string
  /** Icon of the confirm button while idle. */
  icon?: React.ReactNode
  /** Soft-red confirm button; on by default. */
  destructive?: boolean
  /**
   * Runs on confirm. Return a promise (e.g. `mutateAsync`) to show a spinner
   * and close on success; a rejection keeps the dialog open.
   */
  onConfirm: () => Promise<unknown> | void
}

/** Controlled confirmation for destructive actions. */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  icon,
  destructive = true,
  onConfirm,
}: ConfirmDialogProps) {
  const { t } = useTranslation('contextGateway')
  const [pending, setPending] = React.useState(false)

  async function confirm() {
    setPending(true)
    try {
      await onConfirm()
      onOpenChange(false)
    } catch {
      // The caller reports the error; keep the dialog open to retry.
    } finally {
      setPending(false)
    }
  }

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!pending) onOpenChange(next)
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={pending}>
            {t('actions.cancel')}
          </AlertDialogCancel>
          <AlertDialogAction
            variant={destructive ? 'destructive' : 'default'}
            disabled={pending}
            onClick={() => void confirm()}
          >
            {pending ? <LoaderCircleIcon className="animate-spin" /> : icon}
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
