<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getHealth } from '@/api/nec'
import AppShell from '@/components/layout/AppShell.vue'
import type { PageKey } from '@/constants/navigation'
import DemoView from '@/views/DemoView.vue'
import HistoryView from '@/views/HistoryView.vue'
import MetricsView from '@/views/MetricsView.vue'
import type { Health } from '@/types/api'

const activePage = ref<PageKey>('demo')
const health = ref<Health | null>(null)

onMounted(async () => {
  try { health.value = await getHealth() } catch { health.value = null }
})
</script>

<template>
  <AppShell v-model:active-page="activePage" :health="health">
    <DemoView v-if="activePage === 'demo'" />
    <MetricsView v-else-if="activePage === 'metrics'" />
    <HistoryView v-else />
  </AppShell>
</template>
