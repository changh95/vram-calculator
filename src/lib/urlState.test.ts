import { describe, expect, it } from 'vitest'
import { parseConfigSearch, serializeConfigSearch, type ConfigSelection } from './urlState'

const selection: ConfigSelection = {
  modelId: 'qwen3-32b',
  hardwareId: 'rtx-4090',
  quantId: 'q4_k_m',
  frameworkId: 'vllm',
  deviceCount: 2,
  contextLength: 32768,
  concurrentSequences: 4,
}

const fallback: ConfigSelection = {
  modelId: 'default-model',
  hardwareId: 'default-hw',
  quantId: 'fp16',
  frameworkId: 'llamacpp',
  deviceCount: 1,
  contextLength: 8192,
  concurrentSequences: 1,
}

describe('serializeConfigSearch', () => {
  it('encodes every field as a compact query string', () => {
    const qs = serializeConfigSearch(selection)
    const p = new URLSearchParams(qs)
    expect(p.get('m')).toBe('qwen3-32b')
    expect(p.get('hw')).toBe('rtx-4090')
    expect(p.get('q')).toBe('q4_k_m')
    expect(p.get('fw')).toBe('vllm')
    expect(p.get('n')).toBe('2')
    expect(p.get('ctx')).toBe('32768')
    expect(p.get('seq')).toBe('4')
  })
})

describe('parseConfigSearch', () => {
  it('round-trips what serialize produced', () => {
    expect(parseConfigSearch(serializeConfigSearch(selection), fallback)).toEqual(selection)
  })

  it('accepts a leading "?"', () => {
    expect(parseConfigSearch('?' + serializeConfigSearch(selection), fallback)).toEqual(selection)
  })

  it('falls back per-field when params are missing', () => {
    expect(parseConfigSearch('m=glm-5&ctx=4096', fallback)).toEqual({
      ...fallback,
      modelId: 'glm-5',
      contextLength: 4096,
    })
  })

  it('falls back on non-numeric, non-positive, or fractional numeric params', () => {
    expect(parseConfigSearch('n=abc&ctx=-5&seq=1.5', fallback)).toEqual(fallback)
  })

  it('returns the fallback unchanged for an empty string', () => {
    expect(parseConfigSearch('', fallback)).toEqual(fallback)
  })

  it('ignores unknown params', () => {
    expect(parseConfigSearch('utm_source=share&m=kimi-k2', fallback)).toEqual({
      ...fallback,
      modelId: 'kimi-k2',
    })
  })
})
