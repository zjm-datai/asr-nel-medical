<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@/utils'

const props = withDefaults(defineProps<{ tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'info' }>(), { tone: 'neutral' })
const classes = computed(() => cn(
  'inline-flex h-6 items-center rounded-md px-2 text-xs font-semibold',
  props.tone === 'neutral' && 'bg-secondary text-secondary-foreground',
  props.tone === 'success' && 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200',
  props.tone === 'warning' && 'bg-amber-50 text-amber-700 ring-1 ring-amber-200',
  props.tone === 'danger' && 'bg-red-50 text-red-700 ring-1 ring-red-200',
  props.tone === 'info' && 'bg-blue-50 text-blue-700 ring-1 ring-blue-200',
))
</script>

<template><span :class="classes"><slot /></span></template>
