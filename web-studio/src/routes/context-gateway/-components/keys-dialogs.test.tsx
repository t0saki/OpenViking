// @vitest-environment jsdom
import type * as React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type * as TanStackRouter from '@tanstack/react-router'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GatewayError } from '../-lib/api'
import type * as Api from '../-lib/api'
import type { IssuedKey, Profile, Upstream } from '../-lib/api'
import type { Translate } from '../-lib/localize'
import { PROFILE_DEFAULTS } from '../-lib/profile-schema'
import { UPSTREAM_DEFAULTS } from '../-lib/upstream-schema'
import {
  KeysIssueDialog,
  issueProblem,
  validateKeyRequest,
} from './keys-issue-dialog'
import { KeysSecretDialog, snippetModel } from './keys-secret-dialog'

const api = vi.hoisted(() => ({ issueKey: vi.fn() }))

vi.mock('../-lib/api', async (importOriginal) => ({
  ...(await importOriginal<typeof Api>()),
  issueKey: api.issueKey,
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { resolvedLanguage: 'en' },
  }),
}))
vi.mock('#/hooks/use-app-connection', () => ({
  useAppConnection: () => ({
    connection: {
      baseUrl: 'http://localhost:1933',
      accountId: 'acme',
      userId: 'admin',
      apiKey: '',
      adminApiKey: 'admin-key',
    },
    connectionRole: 'admin',
    isConnectionRoleLoading: false,
    serverMode: 'api_key',
  }),
}))
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<typeof TanStackRouter>()),
  Link: (props: { to: string; children?: React.ReactNode }) => (
    <a href={props.to}>{props.children}</a>
  ),
}))

afterEach(cleanup)
beforeEach(() => vi.clearAllMocks())

const t = ((key: string) => key) as unknown as Translate

function upstream(overrides: Partial<Upstream>): Upstream {
  return {
    ...UPSTREAM_DEFAULTS,
    id: 'u1',
    revision: 1,
    name: 'DeepSeek',
    protocol: 'chat',
    base_url: 'https://api.deepseek.com',
    has_api_key: true,
    header_names: [],
    ...overrides,
  }
}

const deepseek = upstream({
  models: ['deepseek-chat'],
  aliases: { fast: 'deepseek-chat' },
})
const openai = upstream({ id: 'u2', name: 'OpenAI', models: ['gpt-5'] })
const profile: Profile = {
  ...PROFILE_DEFAULTS,
  id: 'p1',
  revision: 1,
  name: 'Coding',
}

const request = {
  name: 'Alice',
  openviking_key: 'ov-key',
  policy_id: 'p1',
  upstream_ids: ['u1'],
  models: [],
}

describe('validateKeyRequest', () => {
  it('accepts a complete request', () => {
    expect(validateKeyRequest(request)).toEqual({})
  })

  it('requires every field but the model list', () => {
    expect(
      validateKeyRequest({
        name: ' ',
        openviking_key: '',
        policy_id: '',
        upstream_ids: [],
        models: [],
      }),
    ).toEqual({
      name: 'validation.required',
      openviking_key: 'validation.required',
      policy_id: 'validation.required',
      upstream_ids: 'keys.form.upstreams.required',
    })
  })

  it('rejects a gateway key pasted as the OpenViking key', () => {
    expect(
      validateKeyRequest({ ...request, openviking_key: 'ovcg_abc' }),
    ).toEqual({ openviking_key: 'keys.form.openvikingKey.gatewayKey' })
  })
})

describe('issueProblem', () => {
  it.each([
    ['root_key_not_allowed', 403, 'keys.errors.rootKey', true],
    [
      'OpenViking key belongs to another account',
      403,
      'keys.errors.otherAccount',
      true,
    ],
    ['openviking_http_401', 401, 'keys.errors.invalidKey', true],
    [
      'openviking_identity_missing',
      401,
      'enums.openviking.identityMissing',
      true,
    ],
    ['openviking_unavailable', 503, 'keys.errors.unavailable', false],
    ['openviking_version_mismatch', 503, 'keys.errors.versionMismatch', false],
    ['Unknown upstream', 400, 'errors.unknownUpstream', false],
  ])('maps %s', (detail, status, message, keyField) => {
    expect(issueProblem(t, new GatewayError(detail, status))).toEqual({
      message,
      keyField,
    })
  })
})

describe('snippetModel', () => {
  it('prefers an allowed model the upstreams serve', () => {
    expect(snippetModel(['other', 'fast'], [deepseek])).toBe('fast')
    expect(snippetModel(['other'], [deepseek])).toBe('other')
    expect(snippetModel([], [deepseek])).toBe('deepseek-chat')
    expect(snippetModel([], [])).toBeUndefined()
  })
})

function renderIssueDialog(upstreams: Upstream[] = [deepseek]) {
  const onIssued = vi.fn()
  const onOpenChange = vi.fn()
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <KeysIssueDialog
        open
        onOpenChange={onOpenChange}
        profiles={[profile]}
        upstreams={upstreams}
        onIssued={onIssued}
      />
    </QueryClientProvider>,
  )
  return { onIssued, onOpenChange }
}

function fillRequiredFields() {
  fireEvent.change(screen.getByLabelText('keys.form.name.label'), {
    target: { value: 'Alice' },
  })
  fireEvent.change(screen.getByLabelText('keys.form.openvikingKey.label'), {
    target: { value: 'ov-key' },
  })
}

const submit = () =>
  fireEvent.click(screen.getByRole('button', { name: 'keys.form.submit' }))

describe('KeysIssueDialog', () => {
  it('shows what is missing instead of sending an incomplete request', () => {
    renderIssueDialog([deepseek, openai])
    fillRequiredFields()
    submit()

    expect(screen.getByText('keys.form.upstreams.required')).toBeTruthy()
    expect(api.issueKey).not.toHaveBeenCalled()
  })

  it('sends the chosen upstreams and allowed models', async () => {
    const issued: IssuedKey = {
      id: 'k1',
      revision: 1,
      name: 'Alice',
      policy_id: 'p1',
      upstream_ids: ['u2'],
      models: ['gpt-5'],
      user_id: 'alice',
      prefix: 'ovcg_Ab3dE9x',
      created_at: 1_700_000_000,
      key: 'ovcg_secret',
    }
    api.issueKey.mockResolvedValue(issued)
    const { onIssued } = renderIssueDialog([deepseek, openai])
    fillRequiredFields()
    fireEvent.click(screen.getAllByRole('checkbox')[1])
    // Suggestions come from the selected upstreams only.
    expect(screen.queryByRole('button', { name: 'deepseek-chat' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'gpt-5' }))
    submit()

    await waitFor(() => expect(onIssued).toHaveBeenCalledWith(issued))
    expect(api.issueKey).toHaveBeenCalledWith(expect.anything(), {
      ...request,
      upstream_ids: ['u2'],
      models: ['gpt-5'],
    })
  })

  it('explains a rejected OpenViking key next to that field', async () => {
    api.issueKey.mockRejectedValue(
      new GatewayError('root_key_not_allowed', 403),
    )
    const { onIssued } = renderIssueDialog()
    fillRequiredFields()
    submit()

    expect(await screen.findByText('keys.errors.rootKey')).toBeTruthy()
    expect(
      screen
        .getByLabelText('keys.form.openvikingKey.label')
        .getAttribute('aria-invalid'),
    ).toBe('true')
    expect(screen.queryByText('keys.errors.title')).toBeNull()
    expect(onIssued).not.toHaveBeenCalled()

    // Editing the key clears the server's verdict.
    fireEvent.change(screen.getByLabelText('keys.form.openvikingKey.label'), {
      target: { value: 'another-key' },
    })
    expect(screen.queryByText('keys.errors.rootKey')).toBeNull()
  })

  it('shows other failures in an alert inside the dialog', async () => {
    api.issueKey.mockRejectedValue(
      new GatewayError('openviking_unavailable', 503),
    )
    renderIssueDialog()
    fillRequiredFields()
    submit()

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('keys.errors.title')
    expect(alert.textContent).toContain('keys.errors.unavailable')
  })
})

describe('KeysSecretDialog', () => {
  const issued: IssuedKey = {
    id: 'k1',
    revision: 1,
    name: 'Alice',
    policy_id: 'p1',
    upstream_ids: ['u1'],
    models: [],
    user_id: 'alice',
    prefix: 'ovcg_Ab3dE9x',
    created_at: 1_700_000_000,
    key: 'ovcg_theSecret',
  }

  function renderSecret(upstreams = [deepseek]) {
    const onDone = vi.fn()
    render(
      <KeysSecretDialog
        open
        issued={issued}
        baseUrl="https://gw.example.com"
        profileName="Coding"
        upstreams={upstreams}
        onDone={onDone}
      />,
    )
    return onDone
  }

  it('cannot be dismissed with Escape and has no close button', () => {
    const onDone = renderSecret()
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    fireEvent.keyDown(document.body, { key: 'Escape' })

    expect(screen.getByText('keys.secret.title')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'ui.close' })).toBeNull()
    expect(onDone).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'keys.secret.done' }))
    expect(onDone).toHaveBeenCalledTimes(1)
  })

  it('opens on a client the key can serve, with the real key and address', () => {
    renderSecret()
    expect(screen.getByText('alice')).toBeTruthy()
    expect(screen.getByText('Coding')).toBeTruthy()
    // Only a Chat Completions upstream: the chat client tab is selected.
    const snippet = screen.getByText(/chat\.completions\.create/)
    expect(snippet.textContent).toContain('https://gw.example.com/v1')
    expect(snippet.textContent).toContain('ovcg_theSecret')
    expect(snippet.textContent).toContain('deepseek-chat')
    expect(screen.queryByText('keys.secret.noProtocol')).toBeNull()
  })

  it('warns when the key has no upstream for a client', async () => {
    renderSecret()
    fireEvent.click(
      screen.getByRole('tab', {
        name: 'connect.clients.claude-code.name',
      }),
    )
    expect(await screen.findByText('keys.secret.noProtocol')).toBeTruthy()
  })

  it('treats a disabled upstream as missing, like the Connect page', () => {
    renderSecret([{ ...deepseek, enabled: false }])
    expect(screen.getByText('keys.secret.noProtocol')).toBeTruthy()
    expect(screen.queryByText(/deepseek-chat/)).toBeNull()
  })
})
