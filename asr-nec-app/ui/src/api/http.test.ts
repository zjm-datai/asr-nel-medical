import { describe, expect, it } from 'vitest'
import { apiUrl, resolveApiBaseUrl } from './http'

describe('api url resolution', () => {
  it('uses an explicit API base', () => {
    expect(apiUrl('/api/health', 'http://localhost:8016/', '/')).toBe('http://localhost:8016/api/health')
  })

  it('uses the current deployment directory', () => {
    expect(resolveApiBaseUrl(undefined, '/asr-nec/index.html')).toBe('/asr-nec')
    expect(apiUrl('/api/health', undefined, '/asr-nec/')).toBe('/asr-nec/api/health')
  })
})
