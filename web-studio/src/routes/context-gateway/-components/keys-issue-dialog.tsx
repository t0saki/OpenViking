import * as React from 'react'
import { useMutation } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { CircleAlertIcon, KeyRoundIcon, LoaderCircleIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import { Button } from '#/components/ui/button'
import { Checkbox } from '#/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '#/components/ui/dialog'
import {
  Field,
  FieldDescription,
  FieldError,
  FieldLabel,
} from '#/components/ui/field'
import { Input } from '#/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '#/components/ui/select'
import { PLAIN_INPUT_PROPS } from '#/lib/form-input'
import { cn } from '#/lib/utils'

import { issueKey, toGatewayError } from '../-lib/api'
import type { IssuedKey, KeyRequest, Profile, Upstream } from '../-lib/api'
import { gatewayErrorMessage } from '../-lib/localize'
import type { Translate } from '../-lib/localize'
import { NEW_ID } from '../-lib/search'
import { servedModels } from '../-lib/upstream-schema'
import { useGateway } from '../-lib/use-gateway'
import { SettingField } from './setting-field'
import { ProtocolBadge, ToneBadge } from './status-badges'
import { TagInput } from './tag-input'

/** Prefix of gateway key secrets; such a key can't be bound to another key. */
const GATEWAY_KEY_PREFIX = 'ovcg_'
const MODEL_PREVIEW = 3

type KeyField = 'name' | 'openviking_key' | 'policy_id' | 'upstream_ids'

/** Field → i18n key of the first problem; empty when the request can be sent. */
export type KeyRequestErrors = Partial<Record<KeyField, string>>

/** Checks a key request before sending it. */
export function validateKeyRequest(request: KeyRequest): KeyRequestErrors {
  const errors: KeyRequestErrors = {}
  const secret = request.openviking_key.trim()
  if (!request.name.trim()) errors.name = 'validation.required'
  if (!secret) errors.openviking_key = 'validation.required'
  else if (secret.startsWith(GATEWAY_KEY_PREFIX)) {
    errors.openviking_key = 'keys.form.openvikingKey.gatewayKey'
  }
  if (!request.policy_id) errors.policy_id = 'validation.required'
  if (!request.upstream_ids.length) {
    errors.upstream_ids = 'keys.form.upstreams.required'
  }
  return errors
}

/** Issuance failures about the OpenViking key, with copy that says what to do. */
const ISSUE_ERRORS: Array<[RegExp, string, boolean]> = [
  [/^root_key_not_allowed$/, 'keys.errors.rootKey', true],
  [
    /^OpenViking key belongs to another account/i,
    'keys.errors.otherAccount',
    true,
  ],
  [/^openviking_http_40[13]$/, 'keys.errors.invalidKey', true],
  [/^openviking_identity_missing$/, 'enums.openviking.identityMissing', true],
  [/^openviking_unavailable$/, 'keys.errors.unavailable', false],
  [/^openviking_version_mismatch$/, 'keys.errors.versionMismatch', false],
]

export type IssueProblem = {
  message: string
  /** Shown under the OpenViking key field instead of in the dialog alert. */
  keyField: boolean
}

/** A localized explanation of why issuing a key failed. */
export function issueProblem(t: Translate, error: unknown): IssueProblem {
  const { detail } = toGatewayError(error)
  const known = ISSUE_ERRORS.find(([pattern]) => pattern.test(detail))
  if (known) return { message: t(known[1]), keyField: known[2] }
  return { message: gatewayErrorMessage(t, error), keyField: false }
}

function byName<T extends { name: string }>(items: T[]): T[] {
  return [...items].sort((a, b) => a.name.localeCompare(b.name))
}

type KeysIssueDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  profiles: Profile[]
  upstreams: Upstream[]
  /** Receives the new key, whose secret is only available now. */
  onIssued: (issued: IssuedKey) => void
}

/**
 * Form for a new gateway key: name, OpenViking key, context profile,
 * upstreams and an optional model allowlist. Mount it with a fresh `key`
 * each time it opens to start from an empty form.
 */
export function KeysIssueDialog({
  open,
  onOpenChange,
  profiles,
  upstreams,
  onIssued,
}: KeysIssueDialogProps) {
  const { t } = useTranslation('contextGateway')
  const { connection, invalidate } = useGateway()
  const formId = React.useId()
  const [draft, setDraft] = React.useState<KeyRequest>(() => ({
    name: '',
    openviking_key: '',
    policy_id: profiles.length === 1 ? profiles[0].id : '',
    upstream_ids: upstreams.length === 1 ? [upstreams[0].id] : [],
    models: [],
  }))
  const [submitted, setSubmitted] = React.useState(false)

  const issue = useMutation({
    mutationFn: (request: KeyRequest) => issueKey(connection, request),
    onSuccess: async (issued) => {
      onIssued(issued)
      await invalidate('keys', 'overview')
    },
    onError: (error) => {
      // A profile or upstream deleted meanwhile; show the current lists.
      if (toGatewayError(error).status === 400) {
        void invalidate('profiles', 'upstreams')
      }
    },
  })
  const pending = issue.isPending
  const problem = issue.isError ? issueProblem(t, issue.error) : undefined
  const errors = submitted ? validateKeyRequest(draft) : {}
  const keyError = errors.openviking_key
    ? t(errors.openviking_key)
    : problem?.keyField
      ? problem.message
      : undefined

  const sortedProfiles = byName(profiles)
  const sortedUpstreams = byName(upstreams)
  const selectedProfile = profiles.find((p) => p.id === draft.policy_id)
  const modelSuggestions = [
    ...new Set(
      upstreams
        .filter((upstream) => draft.upstream_ids.includes(upstream.id))
        .flatMap(servedModels),
    ),
  ].sort()

  function update(patch: Partial<KeyRequest>) {
    setDraft((current) => ({ ...current, ...patch }))
    if (issue.isError) issue.reset()
  }

  function toggleUpstream(id: string, checked: boolean) {
    update({
      upstream_ids: checked
        ? [...draft.upstream_ids, id]
        : draft.upstream_ids.filter((other) => other !== id),
    })
  }

  function submit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitted(true)
    if (Object.keys(validateKeyRequest(draft)).length) return
    issue.mutate({
      ...draft,
      name: draft.name.trim(),
      openviking_key: draft.openviking_key.trim(),
      upstream_ids: upstreams
        .map((upstream) => upstream.id)
        .filter((id) => draft.upstream_ids.includes(id)),
    })
  }

  const id = (field: string) => `${formId}-${field}`

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!pending) onOpenChange(next)
      }}
    >
      <DialogContent showCloseButton={!pending} className="gap-5 sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-lg">{t('keys.form.title')}</DialogTitle>
          <DialogDescription>{t('keys.form.description')}</DialogDescription>
        </DialogHeader>

        <form
          id={id('form')}
          className="grid min-w-0 gap-5"
          noValidate
          onSubmit={submit}
        >
          <SettingField
            label={t('keys.form.name.label')}
            htmlFor={id('name')}
            description={t('keys.form.name.description')}
            error={errors.name ? t(errors.name) : undefined}
          >
            <Input
              id={id('name')}
              value={draft.name}
              placeholder={t('keys.form.name.placeholder')}
              aria-invalid={Boolean(errors.name)}
              disabled={pending}
              autoComplete="off"
              onChange={(event) => update({ name: event.target.value })}
            />
          </SettingField>

          <SettingField
            label={t('keys.form.openvikingKey.label')}
            htmlFor={id('openviking-key')}
            description={t('keys.form.openvikingKey.description')}
            error={keyError}
          >
            <Input
              {...PLAIN_INPUT_PROPS}
              id={id('openviking-key')}
              type="password"
              autoComplete="new-password"
              className="font-mono"
              value={draft.openviking_key}
              aria-invalid={Boolean(keyError)}
              disabled={pending}
              onChange={(event) =>
                update({ openviking_key: event.target.value })
              }
            />
          </SettingField>

          <SettingField
            label={t('keys.form.profile.label')}
            htmlFor={id('profile')}
            description={
              profiles.length ? (
                t('keys.form.profile.description')
              ) : (
                <>
                  {t('keys.form.profile.empty')}{' '}
                  <Link
                    to="/context-gateway/profiles/$profileId"
                    params={{ profileId: NEW_ID }}
                    className="font-medium text-foreground underline underline-offset-4"
                  >
                    {t('keys.form.profile.create')}
                  </Link>
                </>
              )
            }
            error={errors.policy_id ? t(errors.policy_id) : undefined}
          >
            <Select
              value={draft.policy_id || null}
              disabled={pending || !profiles.length}
              onValueChange={(value) => {
                if (value) update({ policy_id: value })
              }}
            >
              <SelectTrigger
                id={id('profile')}
                className="w-full"
                aria-invalid={Boolean(errors.policy_id)}
              >
                <SelectValue>
                  {selectedProfile ? (
                    selectedProfile.name
                  ) : (
                    <span className="text-muted-foreground">
                      {t('keys.form.profile.placeholder')}
                    </span>
                  )}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {sortedProfiles.map((profile) => (
                  <SelectItem key={profile.id} value={profile.id}>
                    {profile.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </SettingField>

          <Field data-invalid={Boolean(errors.upstream_ids)} className="gap-2">
            <FieldLabel id={id('upstreams')}>
              {t('keys.form.upstreams.label')}
            </FieldLabel>
            {upstreams.length ? (
              <div
                role="group"
                aria-labelledby={id('upstreams')}
                className="grid gap-2"
              >
                {sortedUpstreams.map((upstream) => (
                  <UpstreamOption
                    key={upstream.id}
                    upstream={upstream}
                    checked={draft.upstream_ids.includes(upstream.id)}
                    disabled={pending}
                    onCheckedChange={(checked) =>
                      toggleUpstream(upstream.id, checked)
                    }
                  />
                ))}
              </div>
            ) : null}
            <FieldDescription className="text-xs">
              {upstreams.length ? (
                t('keys.form.upstreams.description')
              ) : (
                <>
                  {t('keys.form.upstreams.empty')}{' '}
                  <Link
                    to="/context-gateway/upstreams/$upstreamId"
                    params={{ upstreamId: NEW_ID }}
                    className="font-medium text-foreground underline underline-offset-4"
                  >
                    {t('keys.form.upstreams.create')}
                  </Link>
                </>
              )}
            </FieldDescription>
            {errors.upstream_ids ? (
              <FieldError>{t(errors.upstream_ids)}</FieldError>
            ) : null}
          </Field>

          <SettingField
            label={
              <>
                {t('keys.form.models.label')}
                <span className="font-normal text-muted-foreground">
                  {t('keys.form.models.optional')}
                </span>
              </>
            }
            htmlFor={id('models')}
            description={t('keys.form.models.description')}
          >
            <TagInput
              id={id('models')}
              value={draft.models}
              suggestions={modelSuggestions}
              placeholder={t('keys.form.models.placeholder')}
              disabled={pending}
              onChange={(models) => update({ models })}
            />
          </SettingField>

          {problem && !problem.keyField ? (
            <Alert variant="destructive">
              <CircleAlertIcon />
              <AlertTitle>{t('keys.errors.title')}</AlertTitle>
              <AlertDescription>{problem.message}</AlertDescription>
            </Alert>
          ) : null}
        </form>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            disabled={pending}
            onClick={() => onOpenChange(false)}
          >
            {t('actions.cancel')}
          </Button>
          <Button type="submit" form={id('form')} disabled={pending}>
            {pending ? (
              <LoaderCircleIcon className="animate-spin" />
            ) : (
              <KeyRoundIcon />
            )}
            {t('keys.form.submit')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** One selectable upstream: name, protocol, the models it serves and its state. */
function UpstreamOption({
  upstream,
  checked,
  disabled,
  onCheckedChange,
}: {
  upstream: Upstream
  checked: boolean
  disabled: boolean
  onCheckedChange: (checked: boolean) => void
}) {
  const { t } = useTranslation('contextGateway')
  const models = servedModels(upstream)
  const shown = models.slice(0, MODEL_PREVIEW)
  return (
    <label
      className={cn(
        'flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 transition-colors hover:bg-muted/30',
        checked && 'border-primary/30 bg-primary/[0.03]',
        disabled && 'pointer-events-none opacity-60',
      )}
    >
      <Checkbox
        className="mt-0.5"
        checked={checked}
        disabled={disabled}
        onCheckedChange={(next) => onCheckedChange(next)}
      />
      <span className="grid min-w-0 flex-1 gap-1">
        <span className="flex min-w-0 flex-wrap items-center gap-2">
          <span className="truncate text-sm font-medium">{upstream.name}</span>
          <ProtocolBadge protocol={upstream.protocol} />
          {upstream.enabled ? null : (
            <ToneBadge tone="neutral">{t('keys.form.upstreams.off')}</ToneBadge>
          )}
        </span>
        <span
          className={cn(
            'truncate text-xs text-muted-foreground',
            models.length > 0 && 'font-mono',
          )}
          title={models.join(', ') || undefined}
        >
          {models.length
            ? [
                shown.join(', '),
                models.length > shown.length
                  ? t('keys.extra', { count: models.length - shown.length })
                  : '',
              ]
                .filter(Boolean)
                .join(' ')
            : t('upstreams.models.any')}
        </span>
      </span>
    </label>
  )
}
