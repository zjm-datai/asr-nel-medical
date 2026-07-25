import { AudioWaveform, History } from 'lucide-vue-next'

export type PageKey = 'demo' | 'history'

export const pages = [
  { key: 'demo' as const, label: '纠错演示', description: '现场录音与音频识别', icon: AudioWaveform },
  { key: 'history' as const, label: '演示历史', description: '查看与回听推理记录', icon: History },
]
