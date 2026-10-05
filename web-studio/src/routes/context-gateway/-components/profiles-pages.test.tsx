// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
  Outlet,
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  useParams,
  useSearch,
} from '@tanstack/react-router'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import type * as ReactI18next from 'react-i18next'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as Api from '../-lib/api'
import type { GatewayKey, Profile, ProfileSettings } from '../-lib/api'
import { PROFILE_DEFAULTS } from '../-lib/profile-schema'
import { parseProfileEditorSearch } from '../-lib/search'
import { ProfilesPage } from './profiles-page'
import { ProfileEditor } from './profiles-editor'

const api = vi.hoisted(() => ({
  listProfiles: vi.fn(),
  listKeys: vi.fn(),
  saveProfile: vi.fn(),
  deleteProfile: vi.fn(),
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
/** Keys, plus interpolation values as JSON so assertions can check them. */
const translate = vi.hoisted(
  () => (key: string, values?: Record<string, unknown>) =>
    values && Object.keys(values).length
      ? `${key} ${JSON.stringify(values)}`
      : key,
)

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
vi.mock('react-i18next', async (importOriginal) => ({
  ...(await importOriginal<typeof ReactI18next>()),
  useTranslation: () => ({
    t: translate,
    i18n: { resolvedLanguage: 'en' },
  }),
}))
vi.mock('sonner', () => ({ toast }))
vi.mock('../-lib/api', async (importOriginal) => ({
  ...(await importOriginal<typeof Api>()),
  ...api,
}))

const coding = {
  ...PROFILE_DEFAULTS,
  id: 'p1',
  revision: 3,
  name: 'Coding',
  capture: false,
  query_max_chars: 1234,
  quotas: { events: 2, skills: 1 },
  context_window: 64000,
  idle_seconds: 30.5,
  gateway_tools: true,
  allow_write_tools: true,
  tool_allowlist: ['search', 'write'],
  // A field this version of the form does not know about.
  future_setting: 'keep me',
} as Profile

const chat: Profile = {
  ...PROFILE_DEFAULTS,
  id: 'p2',
  revision: 1,
  name: 'Chat',
  recall: false,
}

const key = (id: string, policyId: string): GatewayKey => ({
  id,
  revision: 1,
  name: id,
  policy_id: policyId,
  upstream_ids: ['u1'],
  models: [],
  user_id: 'alice',
  prefix: 'ovcg_abc',
  created_at: 1_700_000_000,
})

/** `profile` as the save body: every field, without id and revision. */
function settingsOf(profile: Profile): ProfileSettings {
  const { id: _id, revision: _revision, ...settings } = profile
  return settings
}

function EditorRoute() {
  const { profileId } = useParams({ strict: false })
  const { from } = useSearch({ strict: false })
  return <ProfileEditor profileId={profileId ?? ''} from={from} />
}

function renderAt(path: string) {
  const root = createRootRoute({ component: Outlet })
  const router = createRouter({
    routeTree: root.addChildren([
      createRoute({
        getParentRoute: () => root,
        path: '/context-gateway/profiles',
        component: ProfilesPage,
      }),
      createRoute({
        getParentRoute: () => root,
        path: '/context-gateway/profiles/$profileId',
        validateSearch: parseProfileEditorSearch,
        component: EditorRoute,
      }),
    ]),
    history: createMemoryHistory({ initialEntries: [path] }),
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  return router
}

/** Save on an existing profile, "Create profile" on a new one. */
const saveButton = () =>
  screen.getByRole<HTMLButtonElement>('button', {
    name: /^(actions\.save|profiles\.editor\.create)$/,
  })
const nameInput = () =>
  screen.getByLabelText<HTMLInputElement>('profiles.name.label')
const sectionSwitch = (section: string) =>
  screen.getByRole('switch', { name: `profiles.${section}.title` })

beforeEach(() => {
  vi.spyOn(window, 'scrollTo').mockImplementation(() => {})
  api.listProfiles.mockResolvedValue([coding, chat])
  api.listKeys.mockResolvedValue([key('k1', 'p1'), key('k2', 'p1')])
  api.saveProfile.mockImplementation(
    (_connection, id: string, settings: ProfileSettings) =>
      Promise.resolve({ ...settings, id, revision: 1 }),
  )
  api.deleteProfile.mockResolvedValue({ deleted: true })
})
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('profile list', () => {
  const card = (name: string) =>
    screen.getByRole('link', { name }).closest<HTMLElement>('[data-slot=card]')!

  it('summarizes each profile and its keys', async () => {
    renderAt('/context-gateway/profiles')
    await screen.findByRole('link', { name: 'Coding' })

    const codingCard = within(card('Coding'))
    expect(
      codingCard.getByText('profiles.summary.recallOn {"tokens":"1,600"}'),
    ).toBeTruthy()
    // Saving is off, so long conversations are off too.
    expect(codingCard.getAllByText('states.off')).toHaveLength(2)
    expect(codingCard.getByText('profiles.summary.toolsReadWrite')).toBeTruthy()
    expect(codingCard.getByText('profiles.usedBy {"count":2}')).toBeTruthy()

    const chatCard = within(card('Chat'))
    expect(
      chatCard.getByText('profiles.summary.takeoverOn {"tokens":"30K"}'),
    ).toBeTruthy()
    expect(chatCard.getByText('profiles.unused')).toBeTruthy()
  })

  it('blocks deleting a profile that keys use and deletes an unused one', async () => {
    renderAt('/context-gateway/profiles')
    await screen.findByText('profiles.usedBy {"count":2}')

    const blocked = within(card('Coding')).getByRole<HTMLButtonElement>(
      'button',
      { name: 'actions.delete' },
    )
    expect(blocked.disabled).toBe(true)
    expect(screen.getByTitle('profiles.deleteBlocked {"count":2}')).toBeTruthy()

    fireEvent.click(
      within(card('Chat')).getByRole('button', { name: 'actions.delete' }),
    )
    fireEvent.click(await screen.findByText('profiles.deleteDialog.confirm'))
    await waitFor(() =>
      expect(api.deleteProfile).toHaveBeenCalledWith(expect.anything(), 'p2'),
    )
    await waitFor(() =>
      expect(toast.success).toHaveBeenCalledWith(
        'profiles.toast.deleted {"name":"Chat"}',
      ),
    )
  })

  it('creates a profile with the recommended settings in one click', async () => {
    api.listProfiles.mockResolvedValue([])
    renderAt('/context-gateway/profiles')
    fireEvent.click(
      await screen.findByText('profiles.actions.createRecommended'),
    )
    await waitFor(() => expect(api.saveProfile).toHaveBeenCalledTimes(1))
    const [, id, settings] = api.saveProfile.mock.calls[0]
    expect(id).toMatch(/^[0-9a-f]{16}$/)
    // Recommended settings under a name in the UI language.
    expect(settings).toEqual({
      ...PROFILE_DEFAULTS,
      name: 'profiles.defaultName',
    })
    expect(screen.getByText('profiles.actions.customize')).toBeTruthy()
  })

  it('duplicates into a prefilled editor that saves a new profile', async () => {
    const router = renderAt('/context-gateway/profiles')
    await screen.findByRole('link', { name: 'Coding' })
    fireEvent.click(
      within(card('Coding')).getByRole('button', { name: 'actions.duplicate' }),
    )

    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        '/context-gateway/profiles/new',
      ),
    )
    expect(router.state.location.search).toEqual({ from: 'p1' })
    const copyName = 'profiles.editor.copyName {"name":"Coding"}'
    expect((await screen.findByDisplayValue(copyName)).id).toBe('profile-name')

    fireEvent.click(saveButton())
    await waitFor(() => expect(api.saveProfile).toHaveBeenCalledTimes(1))
    const [, id, settings] = api.saveProfile.mock.calls[0]
    expect(id).not.toBe('p1')
    expect(settings).toEqual({ ...settingsOf(coding), name: copyName })
  })
})

describe('profile editor', () => {
  it('sends every stored field back unchanged when only the name changes', async () => {
    const router = renderAt('/context-gateway/profiles/p1')
    expect((await screen.findByDisplayValue('Coding')).id).toBe('profile-name')
    expect(saveButton().disabled).toBe(true)

    fireEvent.change(nameInput(), { target: { value: 'Coding v2' } })
    expect(screen.getByText('profiles.editor.unsaved')).toBeTruthy()
    fireEvent.click(saveButton())

    await waitFor(() =>
      expect(api.saveProfile).toHaveBeenCalledWith(expect.anything(), 'p1', {
        ...settingsOf(coding),
        name: 'Coding v2',
      }),
    )
    await waitFor(() =>
      expect(router.state.location.pathname).toBe('/context-gateway/profiles'),
    )
  })

  it('hides settings of switched-off sections and needs saving for long conversations', async () => {
    renderAt('/context-gateway/profiles/p2')
    await screen.findByDisplayValue('Chat')

    expect(
      screen.queryByLabelText('profiles.recall.maxTokens.label'),
    ).toBeNull()
    fireEvent.click(sectionSwitch('recall'))
    expect(
      screen.getByLabelText('profiles.recall.maxTokens.label'),
    ).toBeTruthy()

    expect(
      screen.getByLabelText('profiles.takeover.takeoverTokens.label'),
    ).toBeTruthy()
    expect(sectionSwitch('takeover').hasAttribute('data-disabled')).toBe(false)
    fireEvent.click(sectionSwitch('capture'))
    expect(sectionSwitch('takeover').hasAttribute('data-disabled')).toBe(true)
    expect(screen.getByText('profiles.takeover.needsCapture')).toBeTruthy()
    expect(
      screen.queryByLabelText('profiles.takeover.takeoverTokens.label'),
    ).toBeNull()
  })

  it('shows limit errors inline and blocks saving', async () => {
    renderAt('/context-gateway/profiles/p1')
    await screen.findByDisplayValue('Coding')
    fireEvent.change(screen.getByLabelText('profiles.recall.maxTokens.label'), {
      target: { value: '10' },
    })
    expect(
      screen.getByText('validation.range {"min":64,"max":32000}'),
    ).toBeTruthy()
    expect(screen.getByText('profiles.editor.invalid')).toBeTruthy()
    expect(saveButton().disabled).toBe(true)
  })

  it('keeps invalid settings in view when their section is off or advanced', async () => {
    renderAt('/context-gateway/profiles/p1')
    await screen.findByDisplayValue('Coding')
    expect(screen.queryByText('field.sectionInvalid')).toBeNull()
    fireEvent.change(screen.getByLabelText('profiles.recall.maxTokens.label'), {
      target: { value: '10' },
    })
    fireEvent.click(sectionSwitch('recall'))
    expect(
      screen.getByLabelText('profiles.recall.maxTokens.label'),
    ).toBeTruthy()
    expect(screen.getByText('field.sectionInvalid')).toBeTruthy()
  })

  it('opens advanced settings that hold an error', async () => {
    api.listProfiles.mockResolvedValue([
      { ...chat, recall: true, quotas: { legacy: 2 } } as Profile,
    ])
    renderAt('/context-gateway/profiles/p2')
    await screen.findByDisplayValue('Chat')
    expect(
      screen.getByText('validation.unknownCategory {"name":"legacy"}'),
    ).toBeTruthy()
    expect(screen.getByText('field.sectionInvalid')).toBeTruthy()
    expect(saveButton().disabled).toBe(true)
  })

  it('offers write tools only after they are allowed', async () => {
    renderAt('/context-gateway/profiles/p2')
    await screen.findByDisplayValue('Chat')
    const writeTool = () =>
      screen.queryByRole('checkbox', {
        name: /profiles\.tools\.names\.write\.label/,
      })

    expect(
      screen.queryByRole('checkbox', {
        name: /profiles\.tools\.names\.search\.label/,
      }),
    ).toBeNull()
    fireEvent.click(sectionSwitch('tools'))
    expect(
      screen.getByRole('checkbox', {
        name: /profiles\.tools\.names\.search\.label/,
      }),
    ).toBeTruthy()
    expect(writeTool()).toBeNull()

    fireEvent.click(
      screen.getByRole('switch', { name: 'profiles.tools.allowWrite.label' }),
    )
    expect(screen.getByText('profiles.tools.allowWrite.warning')).toBeTruthy()
    fireEvent.click(writeTool()!)
    fireEvent.click(saveButton())

    await waitFor(() => expect(api.saveProfile).toHaveBeenCalledTimes(1))
    expect(api.saveProfile.mock.calls[0][2]).toMatchObject({
      gateway_tools: true,
      allow_write_tools: true,
      tool_allowlist: ['search', 'read', 'list', 'write'],
    })
  })

  it('shows tool calls by default and saves the switch', async () => {
    renderAt('/context-gateway/profiles/p2')
    await screen.findByDisplayValue('Chat')
    fireEvent.click(sectionSwitch('tools'))
    const showCalls = screen.getByRole('switch', {
      name: 'profiles.tools.showCalls.label',
    })
    expect(showCalls.getAttribute('aria-checked')).toBe('true')

    fireEvent.click(showCalls)
    fireEvent.click(saveButton())

    await waitFor(() => expect(api.saveProfile).toHaveBeenCalledTimes(1))
    expect(api.saveProfile.mock.calls[0][2]).toMatchObject({
      gateway_tools: true,
      show_tool_calls: false,
    })
  })

  it('starts category limits from the searched sources', async () => {
    renderAt('/context-gateway/profiles/p2')
    await screen.findByDisplayValue('Chat')
    fireEvent.click(sectionSwitch('recall'))
    fireEvent.click(screen.getAllByText('field.advanced')[0])
    fireEvent.click(
      screen.getByRole('switch', { name: 'profiles.recall.quotas.label' }),
    )
    const events = screen.getByLabelText<HTMLInputElement>(
      'profiles.recall.quotas.categories.events',
    )
    expect(events.value).toBe('3')
    fireEvent.change(events, { target: { value: '5' } })
    fireEvent.click(saveButton())

    await waitFor(() => expect(api.saveProfile).toHaveBeenCalledTimes(1))
    expect(api.saveProfile.mock.calls[0][2].quotas).toEqual({
      events: 5,
      entities: 3,
      preferences: 3,
      experiences: 3,
      resources: 3,
      skills: 3,
    })
  })

  it('creates a new profile once it has a name', async () => {
    renderAt('/context-gateway/profiles/new')
    await screen.findByText('profiles.editor.newTitle')
    expect(saveButton().disabled).toBe(true)
    expect(screen.getByText('profiles.editor.needsName')).toBeTruthy()

    expect(api.listProfiles).not.toHaveBeenCalled()
    fireEvent.change(nameInput(), { target: { value: '  Writing ' } })
    fireEvent.click(saveButton())
    await waitFor(() => expect(api.saveProfile).toHaveBeenCalledTimes(1))
    const [, id, settings] = api.saveProfile.mock.calls[0]
    expect(id).toMatch(/^[0-9a-f]{16}$/)
    expect(settings).toEqual({ ...PROFILE_DEFAULTS, name: 'Writing' })
  })

  it('explains when the profile no longer exists', async () => {
    renderAt('/context-gateway/profiles/gone')
    expect(
      await screen.findByText('profiles.editor.notFound.title'),
    ).toBeTruthy()
  })
})
