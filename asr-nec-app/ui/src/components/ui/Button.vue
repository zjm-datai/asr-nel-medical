<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@/utils'

const props = withDefaults(defineProps<{
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
  class?: string
  disabled?: boolean
  type?: 'button' | 'submit'
}>(), { variant: 'primary', size: 'md', class: '', type: 'button' })

const classes = computed(() => cn(
  'inline-flex items-center justify-center gap-2 rounded-md font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50',
  props.size === 'sm' ? 'h-8 px-3 text-xs' : 'h-9 px-4 text-sm',
  props.variant === 'primary' && 'bg-primary text-primary-foreground hover:bg-primary/90',
  props.variant === 'secondary' && 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
  props.variant === 'ghost' && 'text-muted-foreground hover:bg-secondary hover:text-foreground',
  props.variant === 'danger' && 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
  props.class,
))
</script>

<template><button :type="type" :disabled="disabled" :class="classes"><slot /></button></template>
