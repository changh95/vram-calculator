/**
 * Actionable fixes for a wont-fit config: each suggestion is a single-knob
 * change re-verdicted through estimateVram (never invented numbers).
 * Order of usefulness: cheaper weights, shorter context, more devices,
 * bigger hardware.
 */
import { estimateVram } from './estimateVram'
import type { CalcConfig, HardwareSpec, QuantScheme, Verdict } from './types'

export interface Suggestion {
  kind: 'quant' | 'context' | 'devices' | 'hardware'
  text: string
  /** The verdict this change achieves — never 'wont-fit'. */
  verdict: Exclude<Verdict, 'wont-fit'>
  quantId?: string
  contextLength?: number
  deviceCount?: number
  hardwareId?: string
  savingGb?: number
}

export interface SuggestionPool {
  quants: QuantScheme[]
  hardware: HardwareSpec[]
  maxDevices?: number
}

const MIN_CONTEXT = 1024

const fmtGb = (gb: number) => (gb >= 10 ? gb.toFixed(0) : gb.toFixed(1))
const tightNote = (verdict: Verdict) => (verdict === 'tight' ? ' (tight)' : '')

export function suggestFixes(config: CalcConfig, pool: SuggestionPool): Suggestion[] {
  const base = estimateVram(config)
  if (base.verdict !== 'wont-fit') return []

  const out: Suggestion[] = []
  const maxDevices = pool.maxDevices ?? 8

  // Quant: the highest-quality (most bits) cheaper quant that flips the verdict.
  const cheaperQuants = pool.quants
    .filter((q) => q.bitsPerWeight < config.quant.bitsPerWeight)
    .sort((a, b) => b.bitsPerWeight - a.bitsPerWeight)
  for (const quant of cheaperQuants) {
    const r = estimateVram({ ...config, quant })
    if (r.verdict !== 'wont-fit') {
      const savingGb = base.totalGb - r.totalGb
      out.push({
        kind: 'quant',
        quantId: quant.id,
        verdict: r.verdict,
        savingGb,
        text: `Quantize weights to ${quant.label} to save ~${fmtGb(savingGb)} GB${tightNote(r.verdict)}.`,
      })
      break
    }
  }

  // Context: the largest power-of-two context below the current one that flips.
  for (
    let ctx = 2 ** Math.floor(Math.log2(config.contextLength - 1));
    ctx >= MIN_CONTEXT;
    ctx /= 2
  ) {
    const r = estimateVram({ ...config, contextLength: ctx })
    if (r.verdict !== 'wont-fit') {
      out.push({
        kind: 'context',
        contextLength: ctx,
        verdict: r.verdict,
        text: `Reduce context to ${ctx.toLocaleString()} tokens${tightNote(r.verdict)}.`,
      })
      break
    }
  }

  // Devices: the minimal count of the same hardware that flips.
  for (let n = config.deviceCount + 1; n <= maxDevices; n++) {
    const r = estimateVram({ ...config, deviceCount: n })
    if (r.verdict !== 'wont-fit') {
      out.push({
        kind: 'devices',
        deviceCount: n,
        verdict: r.verdict,
        text: `Use ${n}× ${config.hardware.name}${tightNote(r.verdict)}.`,
      })
      break
    }
  }

  // Hardware: the smallest device in the pool that flips at the current count.
  const bySize = [...pool.hardware].sort(
    (a, b) =>
      a.memoryGb * (a.unified ? (a.usableFraction ?? 1) : 1) -
      b.memoryGb * (b.unified ? (b.usableFraction ?? 1) : 1),
  )
  for (const hardware of bySize) {
    if (hardware.id === config.hardware.id) continue
    const r = estimateVram({ ...config, hardware })
    if (r.verdict !== 'wont-fit') {
      out.push({
        kind: 'hardware',
        hardwareId: hardware.id,
        verdict: r.verdict,
        text: `This configuration fits on ${hardware.name}${tightNote(r.verdict)}.`,
      })
      break
    }
  }

  return out
}
