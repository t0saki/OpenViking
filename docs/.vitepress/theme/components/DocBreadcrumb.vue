<script setup>
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { sections } from '../docs-sections'
import { data as pages } from '../locale-pages.data'
const { page, lang } = useData()
const locale = computed(() => lang.value.startsWith('zh') ? 'zh' : 'en')
const section = computed(() => {
  const directory = page.value.relativePath.split('/')[1]
  return sections.find(item => item.id === (directory === 'context-compilation' ? 'guides' : directory))
})
const sectionPath = computed(() => {
  const prefix = `${locale.value}/${section.value?.id}/`
  const firstPage = pages.filter(file => file.startsWith(prefix)).sort()[0]
  return firstPage
    ? `/${firstPage.replace(/\.md$/, '')}`
    : `/${locale.value}/getting-started/01-introduction`
})
</script>
<template>
  <nav class="doc-breadcrumb" :aria-label="locale === 'zh' ? '面包屑导航' : 'Breadcrumb'">
    <a :href="withBase(`/${locale}/`)">{{ locale === 'zh' ? '文档首页' : 'Documentation' }}</a>
    <template v-if="section"><span aria-hidden="true">/</span><a :href="withBase(sectionPath)">{{ section[locale] }}</a></template>
    <span aria-hidden="true">/</span><span aria-current="page">{{ page.title }}</span>
  </nav>
</template>
