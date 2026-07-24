<script setup lang="ts">
import { CircleCheck, CircleX, Github, Stethoscope } from 'lucide-vue-next'
import { pages, type PageKey } from '@/constants/navigation'
import { cn } from '@/utils'
import type { Health } from '@/types/api'

defineProps<{ activePage: PageKey; health: Health | null }>()
defineEmits<{ 'update:activePage': [value: PageKey] }>()
</script>

<template>
  <div class="app-shell min-h-screen text-foreground">
    <header class="topbar sticky top-0 z-30">
      <div class="mx-auto flex h-14 max-w-[1400px] items-center justify-between px-5">
        <div class="flex items-center gap-3">
          <div class="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-white"><Stethoscope class="h-4 w-4" /></div>
          <div>
            <div class="text-sm font-bold leading-5">ASR 实体纠错演示台</div>
            <div class="text-xs leading-4 text-muted-foreground">SpeechSearcher + GenerativeLabeler</div>
          </div>
        </div>
        <div class="flex items-center gap-3">
          <div v-if="health" class="hidden items-center gap-1.5 text-xs sm:flex" :class="health.model_loaded ? 'text-emerald-700' : 'text-red-600'">
            <CircleCheck v-if="health.model_loaded" class="h-4 w-4" />
            <CircleX v-else class="h-4 w-4" />
            {{ health.model_loaded ? `模型在线 · ${health.device}` : '模型未加载' }}
          </div>
          <a class="icon-link" href="https://github.com/zjm-datai/asr-nel-medical" target="_blank" title="代码仓库"><Github class="h-4 w-4" /></a>
        </div>
      </div>
    </header>

    <div class="mx-auto grid max-w-[1400px] grid-cols-1 lg:grid-cols-[252px_minmax(0,1fr)]">
      <aside class="sidebar hidden min-h-[calc(100vh-56px)] border-r lg:block">
        <nav class="space-y-1 p-3">
          <button v-for="page in pages" :key="page.key" type="button" :class="cn('nav-item', activePage === page.key && 'nav-item-active')" @click="$emit('update:activePage', page.key)">
            <component :is="page.icon" class="h-4 w-4 shrink-0" />
            <span class="min-w-0"><span class="block truncate font-semibold">{{ page.label }}</span><span class="block truncate text-xs opacity-75">{{ page.description }}</span></span>
          </button>
        </nav>
      </aside>

      <main class="min-w-0 p-4 sm:p-5">
        <nav class="mb-4 flex gap-2 overflow-x-auto lg:hidden">
          <button v-for="page in pages" :key="page.key" type="button" :class="cn('mobile-tab', activePage === page.key && 'mobile-tab-active')" @click="$emit('update:activePage', page.key)">
            <component :is="page.icon" class="h-4 w-4" />{{ page.label }}
          </button>
        </nav>
        <slot />
      </main>
    </div>
  </div>
</template>
