<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { LoaderCircle, RefreshCw } from 'lucide-vue-next'
import { getMetrics } from '@/api/nec'
import Button from '@/components/ui/Button.vue'
import type { MetricsSummary } from '@/types/api'
import { formatPercent } from '@/utils'

const metrics = ref<MetricsSummary | null>(null)
const loading = ref(false)
const error = ref('')

const preferredSs = computed(() => metrics.value?.ss.find((row) => row.run === 'ss_full_seed_20260724') || metrics.value?.ss.at(-1))
const preferredGl = computed(() => metrics.value?.gl.find((row) => row.run === 'gl_augmented_aligned_e5') || metrics.value?.gl.at(-1))

async function load() {
  loading.value = true; error.value = ''
  try { metrics.value = await getMetrics() } catch (cause) { error.value = cause instanceof Error ? cause.message : '指标加载失败' }
  finally { loading.value = false }
}
onMounted(load)
</script>

<template>
  <div class="space-y-5">
    <div class="section-heading"><div><h1>模型评测</h1><p>从服务器 runs 目录实时汇总 SS 检索与 GL 生成评测。</p></div><Button size="sm" variant="secondary" :disabled="loading" @click="load"><RefreshCw class="h-3.5 w-3.5" :class="loading && 'animate-spin'" />刷新</Button></div>
    <p v-if="error" class="error-text">{{ error }}</p>
    <div v-if="loading && !metrics" class="panel p-10 text-center"><LoaderCircle class="mx-auto h-6 w-6 animate-spin text-primary" /></div>
    <template v-else-if="metrics">
      <div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div class="metric-kpi"><span>SS Recall@5</span><strong>{{ formatPercent(preferredSs?.test.recall_at_5) }}</strong><small>{{ preferredSs?.run }}</small></div>
        <div class="metric-kpi"><span>SS Mention R@5</span><strong>{{ formatPercent(preferredSs?.test.mention_recall_at_5) }}</strong><small>实体提及级召回</small></div>
        <div class="metric-kpi"><span>GL Replace F1</span><strong>{{ formatPercent(preferredGl?.generation.test?.replace_f1) }}</strong><small>{{ preferredGl?.run }}</small></div>
        <div class="metric-kpi"><span>GL Exact Match</span><strong>{{ formatPercent(preferredGl?.generation.test?.exact_match) }}</strong><small>生成结果完全一致</small></div>
      </div>

      <section class="panel overflow-hidden">
        <div class="border-b p-4"><h2 class="panel-title">SpeechSearcher 测试集</h2></div>
        <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>实验</th><th>Best epoch</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>Mention R@1</th><th>Mention R@5</th><th>无实体误报率</th></tr></thead><tbody><tr v-for="row in metrics.ss" :key="row.run"><td class="font-semibold text-foreground!">{{ row.run }}</td><td>{{ row.best_epoch ?? '—' }}</td><td>{{ formatPercent(row.test.recall_at_1) }}</td><td>{{ formatPercent(row.test.recall_at_5) }}</td><td>{{ formatPercent(row.test.recall_at_10) }}</td><td>{{ formatPercent(row.test.mention_recall_at_1) }}</td><td>{{ formatPercent(row.test.mention_recall_at_5) }}</td><td>{{ formatPercent(row.test.no_entity_false_positive_rate) }}</td></tr></tbody></table></div>
      </section>

      <section class="panel overflow-hidden">
        <div class="border-b p-4"><h2 class="panel-title">GenerativeLabeler 实验对比</h2></div>
        <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>实验</th><th>Best epoch</th><th>Token accuracy</th><th>Exact match</th><th>Action accuracy</th><th>Replace precision</th><th>Replace recall</th><th>Replace F1</th></tr></thead><tbody><tr v-for="row in metrics.gl" :key="row.run"><td class="font-semibold text-foreground!">{{ row.run }}</td><td>{{ row.best_epoch ?? '—' }}</td><td>{{ formatPercent(row.token_test.target_token_accuracy) }}</td><td>{{ formatPercent(row.generation.test?.exact_match) }}</td><td>{{ formatPercent(row.generation.test?.action_accuracy) }}</td><td>{{ formatPercent(row.generation.test?.replace_precision) }}</td><td>{{ formatPercent(row.generation.test?.replace_recall) }}</td><td>{{ formatPercent(row.generation.test?.replace_f1) }}</td></tr></tbody></table></div>
      </section>
    </template>
  </div>
</template>
