import { describe, expect, it } from 'vitest'
import { FRAMEWORKS, HARDWARE, MODELS, QUANTS, TT_QUANTS, DEFAULTS } from './index'
import { estimateVram, kvCacheBytes, GIB } from '../estimateVram'
import type { CalcConfig } from '../types'

describe('MODELS table', () => {
  it('is non-empty and has unique ids', () => {
    expect(MODELS.length).toBeGreaterThan(20)
    expect(new Set(MODELS.map((m) => m.id)).size).toBe(MODELS.length)
  })

  it('every model has positive params, hidden size, context, and a source', () => {
    for (const m of MODELS) {
      expect(m.paramsTotalB, m.id).toBeGreaterThan(0)
      expect(m.hiddenSize, m.id).toBeGreaterThan(0)
      expect(m.maxContext, m.id).toBeGreaterThan(0)
      expect(m.sources.length, m.id).toBeGreaterThan(0)
    }
  })

  it('MoE models record active < total params; dense models omit active', () => {
    for (const m of MODELS) {
      if (m.moe) {
        expect(m.paramsActiveB, m.id).toBeDefined()
        expect(m.paramsActiveB!, m.id).toBeLessThan(m.paramsTotalB)
        expect(m.paramsActiveB!, m.id).toBeGreaterThan(0)
      }
    }
  })

  it('every attention config has valid, positive KV-relevant dimensions', () => {
    for (const m of MODELS) {
      const a = m.attention
      expect(a.numLayers, m.id).toBeGreaterThan(0)
      switch (a.kind) {
        case 'gqa':
          expect(a.numKvHeads, m.id).toBeGreaterThan(0)
          expect(a.headDim, m.id).toBeGreaterThan(0)
          break
        case 'mla':
          expect(a.kvLoraRank, m.id).toBeGreaterThan(0)
          expect(a.qkRopeHeadDim, m.id).toBeGreaterThan(0)
          break
        case 'gqa-swa':
          expect(a.globalLayers + a.localLayers, m.id).toBeLessThanOrEqual(a.numLayers)
          expect(a.slidingWindow, m.id).toBeGreaterThan(0)
          expect(a.numKvHeads, m.id).toBeGreaterThan(0)
          break
        case 'hybrid':
          expect(a.attentionLayers, m.id).toBeGreaterThan(0)
          expect(a.attentionLayers, m.id).toBeLessThanOrEqual(a.numLayers)
          expect(a.numKvHeads, m.id).toBeGreaterThan(0)
          break
        case 'compressed':
          expect(a.kvDim, m.id).toBeGreaterThan(0)
          expect(a.layerGroups.reduce((s, g) => s + g.count, 0), m.id).toBe(a.numLayers)
          break
      }
    }
  })

  it('every family is represented', () => {
    const families = new Set(MODELS.map((m) => m.family))
    for (const f of ['Qwen3', 'Gemma 4', 'GLM', 'DeepSeek', 'Kimi', 'Nemotron']) {
      expect(families, f).toContain(f)
    }
  })

  it('KV cache per token is positive for every model at a representative context', () => {
    for (const m of MODELS) {
      expect(kvCacheBytes(m.attention, 4096, 1, 2), m.id).toBeGreaterThan(0)
    }
  })
})

describe('known-good KV reference cases (research-validated)', () => {
  const tok = (id: string) => {
    const m = MODELS.find((x) => x.id === id)
    if (!m) throw new Error(`missing model ${id}`)
    return kvCacheBytes(m.attention, 1, 1, 2) // bytes per token at bf16
  }

  it('DeepSeek-V3 MLA ≈ 68.6 KiB/token (61L, 512+64, no factor 2)', () => {
    expect(tok('deepseek-v3')).toBe(70272) // 61 × 576 × 2
  })

  it('Kimi-K2 MLA ≈ 70,272 B/token (61L, same MLA geometry)', () => {
    expect(tok('kimi-k2-instruct')).toBe(70272)
  })

  it('a 70B-class GQA distill ≈ 320 KiB/token (80L, 8kv, 128, ×2)', () => {
    expect(tok('deepseek-r1-distill-llama-70b')).toBe(327680)
  })

  it('legacy Qwen3-32B GQA ≈ 256 KiB/token (64L, 8kv, 128, ×2)', () => {
    expect(tok('qwen3-32b')).toBe(262144)
  })

  it('Llama-3.1-8B GQA = 128 KiB/token (32L, 8kv, 128, ×2)', () => {
    expect(tok('llama-3.1-8b')).toBe(131072)
  })

  it('Llama-3.2-1B GQA = 32 KiB/token (16L, 8kv, head_dim 64, ×2)', () => {
    expect(tok('llama-3.2-1b')).toBe(32768)
  })
})

describe('added families — Llama 3 and DiffusionGemma', () => {
  it('exposes Llama 3 and DiffusionGemma as families', () => {
    const families = new Set(MODELS.map((m) => m.family))
    expect(families).toContain('Llama 3')
    expect(families).toContain('DiffusionGemma')
  })

  it('includes the GPT-OSS family as interleaved-SWA MoE', () => {
    const families = new Set(MODELS.map((m) => m.family))
    expect(families).toContain('GPT-OSS')
    const m = MODELS.find((x) => x.id === 'gpt-oss-20b')!
    expect(m.moe).toBe(true)
    expect(m.attention.kind).toBe('gqa-swa')
  })

  it('DiffusionGemma reuses the Gemma 4 26B-A4B backbone (same KV geometry)', () => {
    const dg = MODELS.find((m) => m.id === 'diffusiongemma-26b-a4b')!
    const g4 = MODELS.find((m) => m.id === 'gemma4-26b-a4b')!
    expect(dg.attention.kind).toBe('gqa-swa')
    expect(kvCacheBytes(dg.attention, 8192, 1, 2)).toBe(kvCacheBytes(g4.attention, 8192, 1, 2))
  })
})

describe('Qwen generations are distinct families', () => {
  it('exposes Qwen3, Qwen3-Next, Qwen3.5, and Qwen3.6 as separate groups', () => {
    const families = new Set(MODELS.map((m) => m.family))
    for (const f of ['Qwen3', 'Qwen3-Next', 'Qwen3.5', 'Qwen3.6']) {
      expect(families, f).toContain(f)
    }
  })

  it('Qwen3-Next is a hybrid with full attention on every 4th layer (12 of 48)', () => {
    const m = MODELS.find((x) => x.id === 'qwen3-next-80b-a3b')!
    expect(m.attention.kind).toBe('hybrid')
    if (m.attention.kind === 'hybrid') {
      expect(m.attention.numLayers).toBe(48)
      expect(m.attention.attentionLayers).toBe(12)
    }
  })

  it('legacy Qwen3 dense models are standard GQA (not hybrid)', () => {
    const m = MODELS.find((x) => x.id === 'qwen3-8b')!
    expect(m.attention.kind).toBe('gqa')
  })
})

describe('QUANTS table', () => {
  it('has unique ids and plausible bit widths (1–16 bpw)', () => {
    expect(new Set(QUANTS.map((q) => q.id)).size).toBe(QUANTS.length)
    for (const q of QUANTS) {
      expect(q.bitsPerWeight, q.id).toBeGreaterThan(1)
      expect(q.bitsPerWeight, q.id).toBeLessThanOrEqual(16)
    }
  })

  it('offers BF16 and FP16 as distinct 16-bpw options', () => {
    expect(QUANTS.find((q) => q.id === 'bf16')?.bitsPerWeight).toBe(16)
    expect(QUANTS.find((q) => q.id === 'fp16')?.bitsPerWeight).toBe(16)
  })

  it('includes the canonical anchor Q4_K_M', () => {
    expect(QUANTS.find((q) => q.id === 'q4_k_m')?.bitsPerWeight).toBeCloseTo(4.89, 2)
  })

  it('TT_QUANTS provide block-float profiles (performance + accuracy) with TT params', () => {
    expect(TT_QUANTS.find((q) => q.id === 'tt-performance')).toBeDefined()
    expect(TT_QUANTS.find((q) => q.id === 'tt-accuracy')).toBeDefined()
    for (const q of TT_QUANTS) {
      expect(q.ttWeightBppMoe!, q.id).toBeGreaterThan(0)
      expect(q.ttWeightBppDense!, q.id).toBeGreaterThan(0)
      expect(q.ttKvBytes!, q.id).toBeGreaterThan(0)
      expect(q.ttKvLabel, q.id).toBeTruthy()
    }
  })
})

describe('HARDWARE table', () => {
  it('is non-empty with unique ids and positive memory', () => {
    expect(HARDWARE.length).toBeGreaterThan(20)
    expect(new Set(HARDWARE.map((h) => h.id)).size).toBe(HARDWARE.length)
    for (const h of HARDWARE) {
      expect(h.memoryGb, h.id).toBeGreaterThan(0)
      expect(h.sources.length, h.id).toBeGreaterThan(0)
    }
  })

  it('unified-memory devices declare a usable fraction below 1', () => {
    for (const h of HARDWARE) {
      if (h.unified) {
        expect(h.usableFraction, h.id).toBeDefined()
        expect(h.usableFraction!, h.id).toBeGreaterThan(0)
        expect(h.usableFraction!, h.id).toBeLessThanOrEqual(1)
        expect(h.usableNote, h.id).toBeTruthy()
      }
    }
  })

  it('multi-chip SKUs report chip count and per-chip memory summing to the aggregate', () => {
    for (const h of HARDWARE) {
      if (h.numChips && h.numChips > 1) {
        expect(h.memoryPerChipGb, h.id).toBeDefined()
        expect(h.memoryPerChipGb! * h.numChips, h.id).toBeCloseTo(h.memoryGb, 0)
      }
    }
  })

  it('Tenstorrent SKUs declare per-chip usable DRAM (12 GiB WH / 32 GiB BH)', () => {
    const tt = HARDWARE.filter((h) => h.vendor === 'Tenstorrent')
    expect(tt.length).toBeGreaterThan(0)
    for (const h of tt) {
      expect(h.usableGbPerChip, h.id).toBeDefined()
      expect(h.usableGbPerChip!, h.id).toBeLessThanOrEqual(h.memoryPerChipGb!)
      expect([12, 32], h.id).toContain(h.usableGbPerChip!)
    }
  })

  it('models B200/B300 as 8-GPU nodes (per-GPU memory × 8 = node total)', () => {
    for (const id of ['b200', 'b300']) {
      const h = HARDWARE.find((x) => x.id === id)!
      expect(h.gpusPerNode, id).toBe(8)
    }
    // A 671B model at FP8 shards across the 8 GPUs of a B200 node and fits.
    const m = MODELS.find((x) => x.id === 'deepseek-v3')!
    const node = HARDWARE.find((x) => x.id === 'b200')!
    const r = estimateVram({
      model: m,
      hardware: node,
      quant: QUANTS.find((q) => q.id === 'fp8')!,
      framework: FRAMEWORKS.find((f) => f.id === 'vllm')!,
      deviceCount: 1,
      contextLength: 8192,
      concurrentSequences: 1,
    })
    expect(r.usableGb).toBeCloseTo(180 * 8, 0) // node total = 1440 GB
    expect(r.verdict).toBe('fits') // ~671 GB weights / 8 GPUs ≈ 84 GB/GPU < 180
  })

  it('covers all four vendors', () => {
    const vendors = new Set(HARDWARE.map((h) => h.vendor))
    for (const v of ['NVIDIA', 'Apple', 'AMD', 'Tenstorrent']) expect(vendors, v).toContain(v)
  })

  it('Apple unified caps follow the 2/3-below-36GB, 3/4-at-or-above rule', () => {
    for (const h of HARDWARE.filter((x) => x.vendor === 'Apple')) {
      const expected = h.memoryGb >= 36 ? 0.75 : 2 / 3
      expect(h.usableFraction!, h.id).toBeCloseTo(expected, 2)
    }
  })
})

describe('FRAMEWORKS table', () => {
  it('has unique ids, a KV dtype, positive KV bytes, and sources', () => {
    expect(new Set(FRAMEWORKS.map((f) => f.id)).size).toBe(FRAMEWORKS.length)
    for (const f of FRAMEWORKS) {
      expect(f.kvDtypeLabel, f.id).toBeTruthy()
      expect(f.kvBytesDefault, f.id).toBeGreaterThan(0)
      expect(f.baselineOverheadGb, f.id).toBeGreaterThanOrEqual(0)
      expect(f.activation.multiplier, f.id).toBeGreaterThan(0)
      expect(f.sources.length, f.id).toBeGreaterThan(0)
    }
  })

  it('includes the core six plus the Tenstorrent stack', () => {
    for (const id of ['vllm', 'sglang', 'llamacpp', 'ollama', 'trtllm', 'mlx', 'tt-metal']) {
      expect(FRAMEWORKS.find((f) => f.id === id), id).toBeDefined()
    }
  })

  it('the Tenstorrent stack defaults to block-float KV (~1.06 B/elem)', () => {
    expect(FRAMEWORKS.find((f) => f.id === 'tt-metal')!.kvBytesDefault).toBeCloseTo(1.0625, 3)
  })
})

describe('DEFAULTS resolve against the tables and produce a verdict', () => {
  it('every default id exists', () => {
    expect(MODELS.find((m) => m.id === DEFAULTS.modelId)).toBeDefined()
    expect(HARDWARE.find((h) => h.id === DEFAULTS.hardwareId)).toBeDefined()
    expect(QUANTS.find((q) => q.id === DEFAULTS.quantId)).toBeDefined()
    expect(FRAMEWORKS.find((f) => f.id === DEFAULTS.frameworkId)).toBeDefined()
  })

  it('the default configuration computes without throwing', () => {
    const config: CalcConfig = {
      model: MODELS.find((m) => m.id === DEFAULTS.modelId)!,
      hardware: HARDWARE.find((h) => h.id === DEFAULTS.hardwareId)!,
      quant: QUANTS.find((q) => q.id === DEFAULTS.quantId)!,
      framework: FRAMEWORKS.find((f) => f.id === DEFAULTS.frameworkId)!,
      deviceCount: DEFAULTS.deviceCount,
      contextLength: DEFAULTS.contextLength,
      concurrentSequences: DEFAULTS.concurrentSequences,
    }
    const r = estimateVram(config)
    expect(r.totalGb).toBeGreaterThan(0)
    expect(['fits', 'tight', 'wont-fit']).toContain(r.verdict)
  })
})

describe('end-to-end sanity: a 9B-class model at Q4 fits a 24 GB card', () => {
  it('Qwen3.5-9B Q4_K_M, 8K ctx, single sequence on an RTX 4090 fits', () => {
    const m = MODELS.find((x) => x.id === 'qwen3.5-9b')
    const hw = HARDWARE.find((x) => x.id === 'rtx-4090')
    const q = QUANTS.find((x) => x.id === 'q4_k_m')
    const fw = FRAMEWORKS.find((x) => x.id === 'llamacpp')
    expect(m && hw && q && fw).toBeTruthy()
    const r = estimateVram({
      model: m!,
      hardware: hw!,
      quant: q!,
      framework: fw!,
      deviceCount: 1,
      contextLength: 8192,
      concurrentSequences: 1,
    })
    // ~9.65B × 4.89/8 ≈ 5.9 GiB weights, tiny KV → comfortably under 24 GiB
    expect(r.weightsGb).toBeLessThan(7)
    expect(r.verdict).toBe('fits')
  })

  it('a 1T MLA model at FP8 will not fit one 24 GB card', () => {
    const m = MODELS.find((x) => x.id === 'kimi-k2-instruct')!
    const hw = HARDWARE.find((x) => x.id === 'rtx-4090')!
    const q = QUANTS.find((x) => x.id === 'fp8')!
    const fw = FRAMEWORKS.find((x) => x.id === 'vllm')!
    const r = estimateVram({
      model: m,
      hardware: hw,
      quant: q,
      framework: fw,
      deviceCount: 1,
      contextLength: 8192,
      concurrentSequences: 1,
    })
    expect(r.verdict).toBe('wont-fit')
    expect(r.weightsGb).toBeGreaterThan(900) // ~1026B × 1 byte
  })
})

describe('GIB is the binary gigabyte', () => {
  it('matches 2^30', () => expect(GIB).toBe(1073741824))
})
