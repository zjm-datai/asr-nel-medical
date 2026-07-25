import { fetchJson, postForm, postJson } from './http'
import type { Correction, Example, Health, MetricsSummary } from '@/types/api'

export type AsrProvider = 'audio_api' | 'local_whisper'

export const getHealth = () => fetchJson<Health>('/api/health')
export const listExamples = async () => (await fetchJson<{ examples: Example[] }>('/api/examples')).examples
export const getMetrics = () => fetchJson<MetricsSummary>('/api/metrics/summary')
export const listCorrections = async () => (await fetchJson<{ corrections: Correction[] }>('/api/corrections')).corrections
export const getCorrection = (id: string) => fetchJson<Correction>(`/api/corrections/${id}`)

export function submitAudio(blob: Blob, filename: string, source: 'mic' | 'upload', asrProvider: AsrProvider) {
  const form = new FormData()
  form.append('file', blob, filename)
  form.append('source', source)
  form.append('asr_provider', asrProvider)
  form.append('top_k', '5')
  form.append('threshold', '0.3')
  return postForm<Correction>('/api/corrections', form)
}

export function submitExample(exampleId: string, asrProvider: AsrProvider) {
  const form = new FormData()
  form.append('example_id', exampleId)
  form.append('asr_provider', asrProvider)
  form.append('top_k', '5')
  form.append('threshold', '0.3')
  return postForm<Correction>('/api/corrections', form)
}

export function rerunCorrection(id: string, asrText: string) {
  return postJson<Correction>(`/api/corrections/${id}/rerun`, { asr_text: asrText })
}
