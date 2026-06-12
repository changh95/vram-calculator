import { describe, expect, it } from 'vitest'
import { LANGUAGES, t } from './i18n'

describe('i18n', () => {
  it('offers exactly English, Korean, Simplified Chinese, and Japanese', () => {
    expect(LANGUAGES.map((l) => l.code)).toEqual(['en', 'ko', 'zh', 'ja'])
    expect(LANGUAGES.map((l) => l.name)).toEqual(['English', '한국어', '简体中文', '日本語'])
  })

  it('translates a key into each language', () => {
    expect(t('en', 'configuration')).toBe('Configuration')
    expect(t('ko', 'configuration')).toBe('구성')
    expect(t('zh', 'configuration')).toBe('配置')
    expect(t('ja', 'configuration')).toBe('設定')
  })

  it('interpolates {x}/{n}/{unit} placeholders', () => {
    expect(t('en', 'usable', { x: '24.0' })).toBe('of 24.0 GB usable')
    expect(t('en', 'totalAcross', { n: 8, unit: 'GPUs' })).toBe('total across 8 GPUs')
    expect(t('ko', 'overBy', { x: '14.0' })).toBe('14.0 GB 초과')
  })

  it('falls back to English for an unknown language', () => {
    // @ts-expect-error — exercising the runtime fallback path
    expect(t('fr', 'configuration')).toBe('Configuration')
  })
})
