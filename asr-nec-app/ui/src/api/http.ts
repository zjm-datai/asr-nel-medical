interface ErrorPayload {
  traceback?: string
  error?: string
  detail?: string | { message?: string }
}

const CONFIGURED_API_BASE_URL = import.meta.env.VITE_API_BASE_URL

export function resolveApiBaseUrl(configuredBaseUrl: string | undefined, pathname: string) {
  if (configuredBaseUrl) return configuredBaseUrl.replace(/\/$/, '')
  const directory = pathname.endsWith('/') ? pathname : pathname.slice(0, pathname.lastIndexOf('/') + 1)
  return directory.replace(/\/$/, '')
}

export function apiUrl(path: string, configuredBaseUrl = CONFIGURED_API_BASE_URL, pathname = window.location.pathname) {
  if (path.startsWith('http')) return path
  return `${resolveApiBaseUrl(configuredBaseUrl, pathname)}${path}`
}

function readError(data: unknown, fallback: string) {
  const payload = data as ErrorPayload
  const detail = payload.traceback || payload.error || payload.detail
  if (typeof detail === 'string') return detail
  return detail?.message || fallback
}

async function parse<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(readError(data, response.statusText))
  return data as T
}

export async function fetchJson<T>(url: string) {
  return parse<T>(await fetch(apiUrl(url)))
}

export async function postJson<T>(url: string, payload: unknown) {
  return parse<T>(await fetch(apiUrl(url), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }))
}

export async function postForm<T>(url: string, form: FormData) {
  return parse<T>(await fetch(apiUrl(url), { method: 'POST', body: form }))
}
