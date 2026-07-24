import { AudioWaveform, ChartNoAxesCombined, History } from 'lucide-vue-next'

export type PageKey = 'demo' | 'metrics' | 'history'

export const pages = [
  { key: 'demo' as const, label: '纠错演示', description: '现场录音与音频识别', icon: AudioWaveform },
  { key: 'metrics' as const, label: '模型评测', description: 'SS 与 GL 实验指标', icon: ChartNoAxesCombined },
  { key: 'history' as const, label: '演示历史', description: '查看与回听推理记录', icon: History },
]
