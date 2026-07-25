<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { Check, FileAudio, LoaderCircle, Mic, RotateCcw, Square, Upload, Volume2, X } from 'lucide-vue-next'
import { apiUrl } from '@/api/http'
import { listExamples, rerunCorrection, submitAudio, submitExample } from '@/api/nec'
import Badge from '@/components/ui/Badge.vue'
import Button from '@/components/ui/Button.vue'
import type { Correction, Example } from '@/types/api'

const examples = ref<Example[]>([])
const result = ref<Correction | null>(null)
const editedText = ref('')
const loading = ref(false)
const error = ref('')
const recording = ref(false)
const dragActive = ref(false)
const recorder = ref<MediaRecorder | null>(null)
const stream = ref<MediaStream | null>(null)
const chunks: Blob[] = []

const changed = computed(() => result.value && result.value.asr_text !== result.value.corrected_text)
const timingEntries = computed(() => result.value ? Object.entries(result.value.timings).filter(([key]) => key !== 'total_ms') : [])

onMounted(async () => {
  try { examples.value = await listExamples() } catch { examples.value = [] }
})

onBeforeUnmount(() => stream.value?.getTracks().forEach((track) => track.stop()))

async function run(action: () => Promise<Correction>) {
  loading.value = true
  error.value = ''
  try {
    result.value = await action()
    editedText.value = result.value.asr_text
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '推理请求失败'
  } finally { loading.value = false }
}

async function upload(file: File) {
  await run(() => submitAudio(file, file.name, 'upload'))
}

async function startRecording() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    error.value = '浏览器仅允许在 HTTPS 页面使用麦克风，请通过安全地址访问'
    return
  }
  if (typeof MediaRecorder === 'undefined') {
    error.value = '当前浏览器不支持录音，请使用最新版 Chrome 或 Edge'
    return
  }

  try {
    stream.value = await navigator.mediaDevices.getUserMedia({ audio: true })
    chunks.length = 0
    recorder.value = new MediaRecorder(stream.value)
    recorder.value.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data) }
    recorder.value.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.value?.mimeType || 'audio/webm' })
      stream.value?.getTracks().forEach((track) => track.stop())
      run(() => submitAudio(blob, 'recording.webm', 'mic'))
    }
    recorder.value.start()
    recording.value = true
  } catch (cause) {
    error.value = cause instanceof DOMException && cause.name === 'NotAllowedError'
      ? '麦克风权限被拒绝，请在浏览器地址栏中允许麦克风访问'
      : cause instanceof Error ? cause.message : '无法访问麦克风'
  }
}

function stopRecording() {
  recorder.value?.stop()
  recording.value = false
}

function onDrop(event: DragEvent) {
  dragActive.value = false
  const file = event.dataTransfer?.files[0]
  if (file) upload(file)
}

function rerun() {
  if (!result.value || !editedText.value.trim()) return
  run(() => rerunCorrection(result.value!.id, editedText.value.trim()))
}
</script>

<template>
  <div class="space-y-5">
    <div class="section-heading">
      <div><h1>命名实体纠错</h1><p>真实语音经过 Whisper 转写、SS 候选检索与 GL 错误片段判断。</p></div>
      <Badge v-if="result" tone="info">{{ result.duration_seconds.toFixed(1) }} 秒音频 · {{ result.timings.total_ms?.toFixed(0) }} ms</Badge>
    </div>

    <div class="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,.9fr)]">
      <section class="panel p-4">
        <div class="mb-3 flex items-center justify-between"><h2 class="panel-title">语音输入</h2><span class="text-xs text-muted-foreground">WAV · MP3 · M4A · WebM</span></div>
        <div class="grid gap-3 sm:grid-cols-2">
          <button type="button" class="capture-action" :class="recording && 'capture-action-recording'" :disabled="loading" @click="recording ? stopRecording() : startRecording()">
            <span class="capture-icon"><Square v-if="recording" class="h-5 w-5 fill-current" /><Mic v-else class="h-5 w-5" /></span>
            <span><strong>{{ recording ? '结束录音' : '麦克风录音' }}</strong><small>{{ recording ? '正在采集语音' : '浏览器实时采集' }}</small></span>
          </button>
          <label class="capture-action" :class="dragActive && 'capture-action-active'" @dragover.prevent="dragActive = true" @dragleave="dragActive = false" @drop.prevent="onDrop">
            <span class="capture-icon"><Upload class="h-5 w-5" /></span>
            <span><strong>上传音频</strong><small>选择或拖入文件</small></span>
            <input class="sr-only" type="file" accept="audio/*,.wav,.mp3,.m4a,.webm,.flac" :disabled="loading" @change="upload(($event.target as HTMLInputElement).files![0])" />
          </label>
        </div>

        <div v-if="examples.length" class="mt-5 border-t pt-4">
          <div class="mb-3 flex items-center gap-2 text-xs font-semibold text-muted-foreground"><FileAudio class="h-4 w-4" />预置案例</div>
          <div class="grid gap-2 md:grid-cols-2">
            <button v-for="example in examples" :key="example.id" type="button" class="example-row" :disabled="loading" @click="run(() => submitExample(example.id))">
              <span class="min-w-0"><strong>{{ example.title }}</strong><small>{{ example.note || example.utterance_id }}</small></span>
              <Volume2 class="h-4 w-4 shrink-0 text-muted-foreground" />
            </button>
          </div>
        </div>
      </section>

      <section class="panel flex min-h-[260px] items-center justify-center p-4">
        <div v-if="loading" class="text-center"><LoaderCircle class="mx-auto h-7 w-7 animate-spin text-primary" /><p class="mt-3 font-semibold">模型推理中</p><p class="mt-1 text-xs text-muted-foreground">ASR → SpeechSearcher → GenerativeLabeler</p></div>
        <div v-else-if="!result" class="text-center text-muted-foreground"><FileAudio class="mx-auto h-7 w-7" /><p class="mt-3 text-sm">等待音频输入</p></div>
        <div v-else class="w-full space-y-3">
          <div class="flex items-center justify-between"><h2 class="panel-title">处理链路</h2><Badge :tone="changed ? 'success' : 'neutral'">{{ changed ? '已纠错' : '无需修改' }}</Badge></div>
          <div class="pipeline">
            <div><span>1</span><strong>ASR</strong><small>{{ result.asr_provider }} · {{ result.timings.transcribe_ms?.toFixed(0) || '—' }} ms</small></div>
            <i></i><div><span>2</span><strong>SS 检索</strong><small>{{ result.timings.search_ms?.toFixed(0) || '—' }} ms</small></div>
            <i></i><div><span>3</span><strong>GL 标注</strong><small>{{ result.timings.label_ms?.toFixed(0) || '—' }} ms</small></div>
          </div>
        </div>
      </section>
    </div>

    <p v-if="error" class="error-text">{{ error }}</p>

    <template v-if="result">
      <section class="panel overflow-hidden">
        <div class="border-b p-4"><div class="flex flex-wrap items-center justify-between gap-3"><h2 class="panel-title">识别与纠错结果</h2><Button size="sm" variant="secondary" :disabled="loading || editedText.trim() === result.asr_text" @click="rerun"><RotateCcw class="h-3.5 w-3.5" />重新运行 SS + GL</Button></div></div>
        <div class="grid divide-y lg:grid-cols-2 lg:divide-x lg:divide-y-0">
          <div class="p-4"><label class="result-label">ASR 原文（{{ result.asr_provider }}，可编辑）</label><textarea v-model="editedText" class="result-editor" rows="4" /></div>
          <div class="p-4"><label class="result-label">纠错结果</label><div class="result-copy" :class="changed && 'result-copy-changed'">{{ result.corrected_text }}</div></div>
        </div>
      </section>

      <section class="panel overflow-hidden">
        <div class="flex items-center justify-between border-b p-4"><h2 class="panel-title">SS 候选与 GL 判定</h2><span class="text-xs text-muted-foreground">Top {{ result.top_k }} · 阈值 {{ result.threshold }}</span></div>
        <div v-if="result.candidates.length" class="divide-y">
          <div v-for="candidate in result.candidates" :key="candidate.surface_id" class="candidate-row">
            <div class="candidate-rank">{{ candidate.rank }}</div>
            <div class="min-w-0"><div class="flex flex-wrap items-center gap-2"><strong class="text-sm">{{ candidate.surface_text }}</strong><Badge>{{ candidate.entity_type || 'entity' }}</Badge><Badge v-if="candidate.risk_level" :tone="candidate.risk_level === 'high' ? 'warning' : 'neutral'">{{ candidate.risk_level }}</Badge></div><div class="mt-2 flex items-center gap-3"><div class="score-track"><span :style="{ width: `${candidate.score * 100}%` }"></span></div><span class="font-mono text-xs text-muted-foreground">{{ (candidate.score * 100).toFixed(1) }}%</span></div></div>
            <div class="candidate-decision"><Badge :tone="candidate.action === 'replace' ? 'success' : 'neutral'"><Check v-if="candidate.action === 'replace'" class="mr-1 h-3 w-3" /><X v-else class="mr-1 h-3 w-3" />{{ candidate.action === 'replace' ? '替换' : '拒绝' }}</Badge><small v-if="candidate.gl_prediction && candidate.action === 'replace'">命中“{{ candidate.gl_prediction }}”</small><small v-else>未识别到错误片段</small></div>
          </div>
        </div>
        <div v-else class="p-8 text-center text-sm text-muted-foreground">没有候选实体超过阈值</div>
      </section>

      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div v-for="([key, value]) in timingEntries" :key="key" class="metric-strip"><span>{{ key.replace('_ms', '') }}</span><strong>{{ value.toFixed(1) }} ms</strong></div>
      </div>
    </template>
  </div>
</template>
