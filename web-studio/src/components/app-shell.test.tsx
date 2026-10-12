// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import {
  ConnectionScopedRouteContent,
  shouldShowGatewayNavigation,
} from './app-shell'

afterEach(cleanup)

describe('ConnectionScopedRouteContent', () => {
  it('does not mount route content while the server mode is unresolved', () => {
    render(
      <ConnectionScopedRouteContent serverMode="checking">
        <span data-testid="identity-scoped-route" />
      </ConnectionScopedRouteContent>,
    )

    expect(screen.queryByTestId('identity-scoped-route')).toBeNull()
  })

  it.each(['api_key', 'trusted', 'dev', 'oidc', 'ldap', 'offline'] as const)(
    'mounts route content after resolving %s mode',
    (serverMode) => {
      render(
        <ConnectionScopedRouteContent serverMode={serverMode}>
          <span data-testid="identity-scoped-route" />
        </ConnectionScopedRouteContent>,
      )

      expect(screen.getByTestId('identity-scoped-route')).toBeTruthy()
    },
  )
})

describe('gateway navigation', () => {
  it.each([
    ['dev', 'root', false, 200, true],
    ['api_key', 'user', true, 200, true],
    ['api_key', 'user', true, 404, false],
    ['api_key', 'admin', true, 404, true],
    ['api_key', 'root', true, 404, true],
    ['api_key', 'unknown', false, 200, false],
    ['offline', 'user', true, 200, false],
  ] as const)(
    '%s %s with access %s/%s: %s',
    (serverMode, role, hasCredential, accessStatus, visible) => {
      expect(
        shouldShowGatewayNavigation({
          isGatewayHost: false,
          serverMode,
          role,
          hasCredential,
          accessStatus,
        }),
      ).toBe(visible)
    },
  )
})
