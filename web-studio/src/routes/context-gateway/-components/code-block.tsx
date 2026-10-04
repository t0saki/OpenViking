import { cn } from '#/lib/utils'

import { CopyButton } from './copy-button'

type CodeBlockProps = {
  code: string
  /** Caption above the code, e.g. a filename or "Terminal". */
  label?: string
  /** Wrap long lines instead of scrolling horizontally. */
  wrap?: boolean
  /** Accessible name of the copy button. */
  copyLabel?: string
  className?: string
}

/** Monospace block with a copy button and an optional caption. */
export function CodeBlock({
  code,
  label,
  wrap = false,
  copyLabel,
  className,
}: CodeBlockProps) {
  return (
    <div
      className={cn(
        'relative min-w-0 overflow-hidden rounded-lg border bg-muted/30',
        className,
      )}
    >
      {label ? (
        <div className="flex items-center justify-between gap-2 border-b bg-muted/40 py-1 pr-1.5 pl-3">
          <span className="truncate font-mono text-xs text-muted-foreground">
            {label}
          </span>
          <CopyButton value={code} label={copyLabel} />
        </div>
      ) : (
        <CopyButton
          value={code}
          label={copyLabel}
          className="absolute top-1.5 right-1.5 bg-muted/60 backdrop-blur-sm"
        />
      )}
      <pre
        className={cn(
          'overflow-x-auto p-3 font-mono text-xs leading-5',
          wrap ? 'break-all whitespace-pre-wrap' : 'whitespace-pre',
          !label && 'pr-10',
        )}
      >
        <code>{code}</code>
      </pre>
    </div>
  )
}
