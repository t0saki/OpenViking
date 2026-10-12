import { useSyncExternalStore } from 'react'

const EVENT = 'gateway-management-key-changed'
const storageKey = (baseUrl: string) =>
  `gateway-management-key:${baseUrl.replace(/\/+$/, '')}`

export function getManagementKey(baseUrl: string): string {
  try {
    return sessionStorage.getItem(storageKey(baseUrl)) ?? ''
  } catch {
    return ''
  }
}

export function setManagementKey(baseUrl: string, key: string) {
  try {
    sessionStorage.setItem(storageKey(baseUrl), key)
  } catch {
    /* Storage may be unavailable. */
  }
  window.dispatchEvent(new Event(EVENT))
}

export function lockManagementKey(baseUrl: string, expired = false) {
  const hadKey = Boolean(getManagementKey(baseUrl))
  try {
    sessionStorage.removeItem(storageKey(baseUrl))
  } catch {
    /* Storage may be unavailable. */
  }
  window.dispatchEvent(new Event(EVENT))
  if (expired && hadKey)
    window.dispatchEvent(new Event('gateway-management-key-expired'))
}

function subscribe(callback: () => void) {
  window.addEventListener(EVENT, callback)
  return () => window.removeEventListener(EVENT, callback)
}

export function useManagementKey(baseUrl: string) {
  return useSyncExternalStore(
    subscribe,
    () => getManagementKey(baseUrl),
    () => '',
  )
}

export function requestManagementUnlock() {
  window.dispatchEvent(new Event('gateway-management-unlock'))
}
