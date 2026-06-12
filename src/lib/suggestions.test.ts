import { describe, expect, it } from 'vitest'
import { estimateVram } from './estimateVram'
import { suggestFixes } from './suggestions'
import type { CalcConfig, FrameworkSpec, HardwareSpec, ModelSpec, QuantScheme } from './types'

const model24b: ModelSpec = {
  id: 'test-24b',
  family: 'Test',
  name: 'Test-24B',
  hfId: 'test/test-24b',
  paramsTotalB: 24, // 44.70 GiB at FP16 — hopeless on a 24 GiB card
  moe: false,
  attention: { kind: 'gqa', numLayers: 32, numKvHeads: 8, headDim: 128 },
  hiddenSize: 4096,
  maxContext: 131072,
  nativeDtype: 'bf16',
  sources: [],
}

const fp16: QuantScheme = { id: 'fp16', label: 'FP16', bitsPerWeight: 16 }
const q8: QuantScheme = { id: 'q8', label: 'Q8', bitsPerWeight: 8.5 }
const q4: QuantScheme = { id: 'q4', label: 'Q4', bitsPerWeight: 4.85 }
const quants = [fp16, q8, q4]

const framework: FrameworkSpec = {
  id: 'test-fw',
  label: 'TestFW',
  kvBytesDefault: 2,
  kvDtypeLabel: 'FP16',
  baselineOverheadGb: 1,
  activation: { multiplier: 8 },
  sources: [],
}

const gpu = (id: string, memoryGb: number): HardwareSpec => ({
  id,
  vendor: 'Test',
  name: id,
  memoryGb,
  memoryType: 'GDDR',
  unified: false,
  sources: [],
})

const gpu24 = gpu('gpu-24', 24)
const hardwarePool = [gpu24, gpu('gpu-48', 48), gpu('gpu-80', 80), gpu('gpu-192', 192)]

const wontFit: CalcConfig = {
  model: model24b,
  quant: fp16,
  hardware: gpu24,
  deviceCount: 1,
  framework,
  contextLength: 8192,
  concurrentSequences: 1,
}

describe('suggestFixes', () => {
  it('returns nothing when the config already fits', () => {
    const fits: CalcConfig = { ...wontFit, quant: q4 }
    expect(estimateVram(fits).verdict).toBe('fits')
    expect(suggestFixes(fits, { quants, hardware: hardwarePool })).toEqual([])
  })

  it('suggests the highest-quality quant that stops being wont-fit (Q8 still fails, Q4 works)', () => {
    const s = suggestFixes(wontFit, { quants, hardware: hardwarePool })
    const quantFix = s.find((x) => x.kind === 'quant')
    expect(quantFix).toMatchObject({ kind: 'quant', quantId: 'q4' })
    const saved = estimateVram(wontFit).totalGb - estimateVram({ ...wontFit, quant: q4 }).totalGb
    expect(quantFix!.savingGb).toBeCloseTo(saved, 6)
    expect(quantFix!.text).toContain('Q4')
  })

  it('suggests the largest power-of-two context that stops being wont-fit', () => {
    // 8B model at 1M context: KV alone is 128 GiB. 32K is tight, 16K fits.
    const ctxBound: CalcConfig = {
      ...wontFit,
      model: { ...model24b, paramsTotalB: 8 },
      contextLength: 1_048_576,
    }
    expect(estimateVram(ctxBound).verdict).toBe('wont-fit')
    const s = suggestFixes(ctxBound, { quants, hardware: hardwarePool })
    const ctxFix = s.find((x) => x.kind === 'context')
    expect(ctxFix).toMatchObject({ kind: 'context', contextLength: 32768 })
  })

  it('omits the context suggestion when context reduction cannot help', () => {
    const s = suggestFixes(wontFit, { quants, hardware: hardwarePool })
    expect(s.find((x) => x.kind === 'context')).toBeUndefined()
  })

  it('suggests the minimal device count that stops being wont-fit', () => {
    const s = suggestFixes(wontFit, { quants, hardware: hardwarePool, maxDevices: 8 })
    expect(s.find((x) => x.kind === 'devices')).toMatchObject({ kind: 'devices', deviceCount: 3 })
  })

  it('does not suggest devices when even maxDevices does not help', () => {
    const s = suggestFixes(wontFit, { quants, hardware: hardwarePool, maxDevices: 2 })
    expect(s.find((x) => x.kind === 'devices')).toBeUndefined()
  })

  it('suggests the smallest hardware in the pool that stops being wont-fit', () => {
    // Total ≈ 47.2 GiB → 48 GiB card is tight but workable; 80 GiB unnecessary.
    const s = suggestFixes(wontFit, { quants, hardware: hardwarePool })
    expect(s.find((x) => x.kind === 'hardware')).toMatchObject({ kind: 'hardware', hardwareId: 'gpu-48' })
  })

  it('annotates each suggestion with the verdict it achieves', () => {
    const s = suggestFixes(wontFit, { quants, hardware: hardwarePool })
    const hwFix = s.find((x) => x.kind === 'hardware')
    expect(hwFix!.verdict).toBe('tight')
    const quantFix = s.find((x) => x.kind === 'quant')
    expect(quantFix!.verdict).toBe('fits')
  })

  it('orders suggestions: quant, context, devices, hardware', () => {
    const ctxBound: CalcConfig = {
      ...wontFit,
      model: { ...model24b, paramsTotalB: 8 },
      contextLength: 1_048_576,
    }
    const kinds = suggestFixes(ctxBound, { quants, hardware: hardwarePool, maxDevices: 8 }).map((s) => s.kind)
    expect(kinds).toEqual(
      ['quant', 'context', 'devices', 'hardware'].filter((k) => kinds.includes(k as never)),
    )
  })
})
