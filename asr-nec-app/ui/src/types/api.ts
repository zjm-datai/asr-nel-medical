export interface Candidate {
  rank: number
  surface_id: string
  entity_id: string
  surface_text: string
  canonical_name: string
  entity_type: string
  risk_level: string
  score: number
  gl_prediction: string
  action: 'replace' | 'reject'
  applied: boolean
}

export interface Correction {
  id: string
  created_at: string
  source: 'mic' | 'upload' | 'example'
  audio_url: string
  duration_seconds: number
  asr_provider: 'audio_api' | 'local_whisper' | 'manual'
  asr_text: string
  corrected_text: string
  top_k: number
  threshold: number
  candidates: Candidate[]
  timings: Record<string, number>
}

export interface Example {
  id: string
  title: string
  utterance_id: string
  domain: string
  audio_url: string
  expected_asr_text: string
  expected_corrected_text: string
  note: string
}

export interface MetricsSummary {
  ss: Array<{ run: string; best_epoch?: number; test: Record<string, number> }>
  gl: Array<{
    run: string
    best_epoch?: number
    token_test: Record<string, number>
    generation: { dev?: Record<string, number>; test?: Record<string, number> }
  }>
}

export interface Health {
  status: string
  database: string
  model_loaded: boolean
  device: string
}
