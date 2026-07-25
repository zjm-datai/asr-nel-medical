<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { FileAudio, LoaderCircle, RefreshCw, Volume2 } from 'lucide-vue-next'
import { apiUrl } from '@/api/http'
import { listCorrections } from '@/api/nec'
import Badge from '@/components/ui/Badge.vue'
import Button from '@/components/ui/Button.vue'
import type { Correction } from '@/types/api'
import { formatDate } from '@/utils'

const rows = ref<Correction[]>([])
const selected = ref<Correction | null>(null)
const loading = ref(false)
const error = ref('')

async function load() {
  loading.value = true; error.value = ''
  try { rows.value = await listCorrections(); if (!selected.value && rows.value.length) selected.value = rows.value[0] }
  catch (cause) { error.value = cause instanceof Error ? cause.message : '历史加载失败' }
  finally { loading.value = false }
}
onMounted(load)
</script>

<template>
  <div class="space-y-5">
    <div class="section-heading"><div><h1>演示历史</h1><p>推理结果、候选明细与原始音频保存在本机 SQLite 和 storage 目录。</p></div><Button size="sm" variant="secondary" :disabled="loading" @click="load"><RefreshCw class="h-3.5 w-3.5" :class="loading && 'animate-spin'" />刷新</Button></div>
    <p v-if="error" class="error-text">{{ error }}</p>
    <div v-if="loading && !rows.length" class="panel p-10 text-center"><LoaderCircle class="mx-auto h-6 w-6 animate-spin text-primary" /></div>
    <div v-else-if="!rows.length" class="panel p-12 text-center text-muted-foreground"><FileAudio class="mx-auto h-7 w-7" /><p class="mt-3">暂无演示记录</p></div>
    <div v-else class="grid gap-4 xl:grid-cols-[minmax(360px,.9fr)_minmax(0,1.1fr)]">
      <section class="panel overflow-hidden"><div class="history-list"><button v-for="row in rows" :key="row.id" type="button" class="history-row" :class="selected?.id === row.id && 'history-row-active'" @click="selected = row"><div class="flex items-center justify-between gap-2"><span class="text-xs font-semibold">{{ formatDate(row.created_at) }}</span><Badge>{{ row.source }}</Badge></div><strong>{{ row.corrected_text || row.asr_text }}</strong><small>{{ row.asr_text }}</small></button></div></section>
      <section v-if="selected" class="panel p-4 sm:p-5">
        <div class="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 class="panel-title">推理详情</h2><p class="mt-1 text-xs text-muted-foreground">{{ selected.id }}</p></div><Badge :tone="selected.asr_text !== selected.corrected_text ? 'success' : 'neutral'">{{ selected.asr_text !== selected.corrected_text ? '已纠错' : '无需修改' }}</Badge></div>
        <audio class="w-full" controls :src="apiUrl(selected.audio_url)"></audio>
        <dl class="mt-5 space-y-4"><div><dt>ASR 原文（{{ selected.asr_provider }}）</dt><dd>{{ selected.asr_text }}</dd></div><div><dt>纠错结果</dt><dd class="text-emerald-800">{{ selected.corrected_text }}</dd></div></dl>
        <div class="mt-5 border-t pt-4"><h3 class="text-xs font-semibold text-muted-foreground">候选判定</h3><div class="mt-3 space-y-2"><div v-for="candidate in selected.candidates" :key="candidate.surface_id" class="flex items-center justify-between gap-3 rounded-md bg-muted px-3 py-2"><span class="min-w-0 truncate text-sm font-semibold">{{ candidate.surface_text }}</span><span class="flex items-center gap-2"><small class="font-mono text-muted-foreground">{{ (candidate.score * 100).toFixed(1) }}%</small><Badge :tone="candidate.action === 'replace' ? 'success' : 'neutral'">{{ candidate.action }}</Badge></span></div></div></div>
      </section>
    </div>
  </div>
</template>
