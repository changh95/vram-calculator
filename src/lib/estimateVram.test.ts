import { describe, expect, it } from 'vitest'
import {
  GIB,
  REPLICATED_WEIGHT_FRACTION,
  activationBytes,
  estimateVram,
  kvCacheBytes,
  verdictFromUtilization,
  weightsBytes,
} from './estimateVram'
import type {
  CalcConfig,
  FrameworkSpec,
  HardwareSpec,
  ModelSpec,
  QuantScheme,
} from './types'

// ---------------------------------------------------------------------------
// Shared fixtures — a Llama-3-8B-shaped GQA model is the canonical reference:
// KV/token = 2 × 32 layers × 8 kv-heads × 128 head-dim × 2 B = 131072 B,
// so 8K context × 1 seq = exactly 1 GiB of KV cache (well-known figure).
// ---------------------------------------------------------------------------

const gqaModel: ModelSpec = {
  id: 'test-8b',
  family: 'Test',
  name: 'Test-8B',
  hfId: 'test/test-8b',
  paramsTotalB: 8,
  moe: false,
  attention: { kind: 'gqa', numLayers: 32, numKvHeads: 8, headDim: 128 },
  hiddenSize: 4096,
  maxContext: 131072,
  nativeDtype: 'bf16',
  sources: [],
}

const fp16: QuantScheme = { id: 'fp16', label: 'FP16', bitsPerWeight: 16 }

const baseFramework: FrameworkSpec = {
  id: 'test-fw',
  label: 'TestFW',
  kvBytesDefault: 2,
  kvDtypeLabel: 'FP16',
  baselineOverheadGb: 1,
  activation: { multiplier: 8 },
  sources: [],
}

const gpu24: HardwareSpec = {
  id: 'gpu-24',
  vendor: 'Test',
  name: 'GPU 24G',
  memoryGb: 24,
  memoryType: 'GDDR6X',
  unified: false,
  sources: [],
}

const baseConfig: CalcConfig = {
  model: gqaModel,
  quant: fp16,
  hardware: gpu24,
  deviceCount: 1,
  framework: baseFramework,
  contextLength: 8192,
  concurrentSequences: 1,
}

// ---------------------------------------------------------------------------
// weightsBytes
// ---------------------------------------------------------------------------

describe('weightsBytes', () => {
  it('computes params × effective bytes/weight (8B @ FP16 = 16e9 bytes)', () => {
    expect(weightsBytes(8, 16)).toBe(16e9)
  })

  it('handles fractional effective bits (GGUF-style 4.85 bpw)', () => {
    expect(weightsBytes(30.5, 4.85)).toBeCloseTo(18.490625e9, 0)
  })

  it('MoE models are charged for TOTAL params, not active (callers pass total)', () => {
    // 235B-A22B class: weights memory follows the 235, never the 22.
    expect(weightsBytes(235, 8)).toBe(235e9)
  })
})

// ---------------------------------------------------------------------------
// kvCacheBytes — one branch per attention architecture
// ---------------------------------------------------------------------------

describe('kvCacheBytes — GQA/MHA', () => {
  const attn = gqaModel.attention

  it('2 × L × H_kv × d × bytes × ctx × seqs (reference: 1 GiB @ 8K)', () => {
    expect(kvCacheBytes(attn, 8192, 1, 2)).toBe(GIB)
  })

  it('scales linearly with sequences', () => {
    expect(kvCacheBytes(attn, 8192, 4, 2)).toBe(4 * GIB)
  })

  it('scales linearly with KV bytes (FP8 KV halves it)', () => {
    expect(kvCacheBytes(attn, 8192, 1, 1)).toBe(GIB / 2)
  })
})

describe('kvCacheBytes — MLA', () => {
  // DeepSeek-V3 geometry: 61 layers, kv_lora_rank 512, qk_rope_head_dim 64.
  // Per token: 61 × (512 + 64) × 2 B = 70,272 B — no K/V factor of 2.
  const attn = { kind: 'mla', numLayers: 61, kvLoraRank: 512, qkRopeHeadDim: 64 } as const

  it('L × (rank + rope) × bytes × ctx × seqs, no factor 2', () => {
    expect(kvCacheBytes(attn, 8192, 1, 2)).toBe(70272 * 8192)
  })

  it('is dramatically smaller than equivalent MHA (the point of MLA)', () => {
    const mhaEquivalent = kvCacheBytes(
      { kind: 'gqa', numLayers: 61, numKvHeads: 128, headDim: 128 },
      8192,
      1,
      2,
    )
    expect(kvCacheBytes(attn, 8192, 1, 2)).toBeLessThan(mhaEquivalent / 20)
  })
})

describe('kvCacheBytes — MLA with DSA indexer (DeepSeek V3.2 / GLM-5 style)', () => {
  // V3.2: MLA latent unchanged (576/token/layer) + 128-elem FP8 indexer keys.
  const attn = {
    kind: 'mla',
    numLayers: 61,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    indexerElemsPerTokenPerLayer: 128,
  } as const

  it('adds the FP8 indexer cache (1 byte/elem) on top of the MLA latent', () => {
    // latent: 61 × 576 × 2 × 8192 = 575,668,224
    // indexer: 61 × 128 × 1 × 8192 = 63,963,136
    expect(kvCacheBytes(attn, 8192, 1, 2)).toBe(575_668_224 + 63_963_136)
  })

  it('indexer stays 1 byte/elem even when KV dtype changes', () => {
    // latent at fp8 KV: 287,834,112; indexer unchanged
    expect(kvCacheBytes(attn, 8192, 1, 1)).toBe(287_834_112 + 63_963_136)
  })
})

describe('kvCacheBytes — gqa-swa with distinct global geometry (Gemma 4 style)', () => {
  // Gemma-4-12B: 40 local layers (8 KV heads × 256) + 8 global (1 × 512),
  // window 1024 — verified from google/gemma-4-12B-it config.json.
  const attn = {
    kind: 'gqa-swa',
    numLayers: 48,
    numKvHeads: 8,
    headDim: 256,
    slidingWindow: 1024,
    globalLayers: 8,
    localLayers: 40,
    globalKvHeads: 1,
    globalHeadDim: 512,
  } as const

  it('uses per-layer-type KV geometry', () => {
    // local: 40 × 1024 × (2 × 8 × 256 × 2 = 8192 B) = 335,544,320
    // global: 8 × 32768 × (2 × 1 × 512 × 2 = 2048 B) = 536,870,912
    expect(kvCacheBytes(attn, 32768, 1, 2)).toBe(335_544_320 + 536_870_912)
  })

  it('supports fewer distinct-KV layers than numLayers (Gemma 4 E2B KV sharing)', () => {
    // E2B: 35 layers but only 15 hold distinct KV (3 global + 12 local),
    // global geometry falls back to numKvHeads × 512.
    const e2b = {
      kind: 'gqa-swa',
      numLayers: 35,
      numKvHeads: 1,
      headDim: 256,
      slidingWindow: 512,
      globalLayers: 3,
      localLayers: 12,
      globalHeadDim: 512,
    } as const
    // local: 12 × 512 × (2 × 1 × 256 × 2) = 6,291,456
    // global: 3 × 8192 × (2 × 1 × 512 × 2) = 50,331,648
    expect(kvCacheBytes(e2b, 8192, 1, 2)).toBe(6_291_456 + 50_331_648)
  })
})

describe('kvCacheBytes — compressed attention (DeepSeek V4)', () => {
  // V4-Flash: 43 layers = 20 CSA (ratio 4, with indexer) + 20 HCA (ratio 128)
  // + 3 pure-SWA (ratio 0); kvDim 512, window 128 — ratios verified from
  // DeepSeek's reference inference/model.py by the fact-checker.
  const attn = {
    kind: 'compressed',
    numLayers: 43,
    kvDim: 512,
    windowTokens: 128,
    layerGroups: [
      { count: 20, compressRatio: 4, indexerDim: 128, indexerRatio: 4 },
      { count: 20, compressRatio: 128 },
      { count: 3, compressRatio: 0 },
    ],
  } as const

  it('each group caches window + ctx/ratio entries; ratio 0 keeps only the window', () => {
    const ctx = 1_048_576
    // CSA: 20 × (128 + 262144) × 512 × 2 = 5,371,330,560
    // HCA: 20 × (128 + 8192) × 512 × 2 = 170,393,600
    // SWA: 3 × 128 × 512 × 2 = 393,216
    // indexer (FP8): 20 × 262144 × 128 × 1 = 671,088,640
    expect(kvCacheBytes(attn, ctx, 1, 2)).toBe(
      5_371_330_560 + 170_393_600 + 393_216 + 671_088_640,
    )
  })

  it('is an order of magnitude below MLA at 1M context (the point of V4)', () => {
    const mlaV32 = kvCacheBytes(
      { kind: 'mla', numLayers: 61, kvLoraRank: 512, qkRopeHeadDim: 64 },
      1_048_576,
      1,
      2,
    )
    expect(kvCacheBytes(attn, 1_048_576, 1, 2)).toBeLessThan(mlaV32 / 10)
  })

  it('caps cached tokens at the actual context', () => {
    // ctx 64 < window 128: every layer caches only 64 tokens, no compressed part.
    const tiny = kvCacheBytes(attn, 64, 1, 2)
    expect(tiny).toBe(43 * 64 * 512 * 2 + 20 * 16 * 128 * 1)
  })
})

describe('kvCacheBytes — interleaved sliding-window (Gemma-3 style)', () => {
  const attn = {
    kind: 'gqa-swa',
    numLayers: 6,
    numKvHeads: 4,
    headDim: 128,
    slidingWindow: 1024,
    globalLayers: 1,
    localLayers: 5,
  } as const

  it('global layers pay full context, local layers cap at the window', () => {
    // per layer-token: 2 × 4 × 128 × 2 = 2048 B
    // global: 1 × 8192 × 2048 = 16,777,216; local: 5 × 1024 × 2048 = 10,485,760
    expect(kvCacheBytes(attn, 8192, 1, 2)).toBe(16_777_216 + 10_485_760)
  })

  it('degenerates to plain GQA when context fits inside the window', () => {
    const plain = kvCacheBytes({ kind: 'gqa', numLayers: 6, numKvHeads: 4, headDim: 128 }, 512, 1, 2)
    expect(kvCacheBytes(attn, 512, 1, 2)).toBe(plain)
  })
})

describe('kvCacheBytes — SSM hybrid (Nemotron-H style)', () => {
  const attn = {
    kind: 'hybrid',
    numLayers: 52,
    attentionLayers: 4,
    numKvHeads: 8,
    headDim: 128,
    ssmStateBytesPerSeq: 50_000_000,
  } as const

  it('only attention layers hold KV; SSM state is per-sequence and context-independent', () => {
    // KV: 2 × 4 × 8 × 128 × 2 = 16,384 B/token × 4096 × 2 seqs = 134,217,728
    expect(kvCacheBytes(attn, 4096, 2, 2)).toBe(134_217_728 + 2 * 50_000_000)
  })

  it('treats missing SSM state as zero', () => {
    const { ssmStateBytesPerSeq: _omitted, ...rest } = attn
    expect(kvCacheBytes(rest, 4096, 2, 2)).toBe(134_217_728)
  })
})

// ---------------------------------------------------------------------------
// activationBytes (estimate term — pinned arithmetic, not pinned truth)
// ---------------------------------------------------------------------------

describe('activationBytes', () => {
  it('multiplier × hidden × batched-tokens × 2 bytes, tokens = full context without a chunk cap', () => {
    // 8 × 4096 × 8192 × 2 = 536,870,912 B = 0.5 GiB
    expect(activationBytes(4096, 8192, 1, { multiplier: 8 })).toBe(GIB / 2)
  })

  it('chunked prefill caps the batched tokens', () => {
    // tokens = min(8192, 2048) → 8 × 4096 × 2048 × 2 = 134,217,728
    expect(activationBytes(4096, 8192, 1, { multiplier: 8, chunkTokens: 2048 })).toBe(134_217_728)
  })

  it('decode batch can exceed the chunk cap (tokens = max(chunk-capped ctx, seqs))', () => {
    expect(activationBytes(4096, 8192, 4096, { multiplier: 8, chunkTokens: 2048 })).toBe(
      8 * 4096 * 4096 * 2,
    )
  })
})

// ---------------------------------------------------------------------------
// verdictFromUtilization — thresholds per CLAUDE.md: >100% wont-fit, >90% tight
// ---------------------------------------------------------------------------

describe('verdictFromUtilization', () => {
  it('fits at exactly 90%', () => expect(verdictFromUtilization(0.9)).toBe('fits'))
  it('tight just above 90%', () => expect(verdictFromUtilization(0.9001)).toBe('tight'))
  it('tight at exactly 100%', () => expect(verdictFromUtilization(1.0)).toBe('tight'))
  it("won't fit above 100%", () => expect(verdictFromUtilization(1.0001)).toBe('wont-fit'))
})

// ---------------------------------------------------------------------------
// estimateVram — end-to-end
// ---------------------------------------------------------------------------

describe('estimateVram — single device', () => {
  // weights 16e9 B = 14.9012 GiB, KV 1 GiB, activations 0.5 GiB, overhead 1 GiB
  const expectedTotal = 16e9 / GIB + 1 + 0.5 + 1

  it('produces the reference breakdown', () => {
    const r = estimateVram(baseConfig)
    expect(r.weightsGb).toBeCloseTo(14.9012, 4)
    expect(r.kvCacheGb).toBe(1)
    expect(r.activationsGb).toBe(0.5)
    expect(r.overheadGb).toBe(1)
    expect(r.totalGb).toBeCloseTo(expectedTotal, 6)
    expect(r.perDeviceGb).toBeCloseTo(expectedTotal, 6)
  })

  it('verdicts fits with comfortable headroom (24 GiB)', () => {
    const r = estimateVram(baseConfig)
    expect(r.usableGb).toBe(24)
    expect(r.utilization).toBeCloseTo(expectedTotal / 24, 6)
    expect(r.verdict).toBe('fits')
    expect(r.shortfallGb).toBeNull()
  })

  it('verdicts tight between 90% and 100%', () => {
    const r = estimateVram({ ...baseConfig, hardware: { ...gpu24, memoryGb: 18 } })
    expect(r.verdict).toBe('tight')
    expect(r.shortfallGb).toBeNull()
  })

  it("verdicts wont-fit over capacity and reports the aggregate shortfall", () => {
    const r = estimateVram({ ...baseConfig, hardware: { ...gpu24, memoryGb: 16 } })
    expect(r.verdict).toBe('wont-fit')
    expect(r.shortfallGb).toBeCloseTo(expectedTotal - 16, 6)
  })

  it('applies the framework KV waste factor to the KV term', () => {
    const r = estimateVram({
      ...baseConfig,
      framework: { ...baseFramework, kvWasteFactor: 1.05 },
    })
    expect(r.kvCacheGb).toBeCloseTo(1.05, 9)
  })
})

describe('estimateVram — unified memory', () => {
  const mac128: HardwareSpec = {
    id: 'mac-128',
    vendor: 'Apple',
    name: 'Mac 128G',
    memoryGb: 128,
    memoryType: 'LPDDR5x unified',
    unified: true,
    usableFraction: 0.75,
    usableNote: 'macOS GPU working-set cap',
    sources: [],
  }

  it('caps usable capacity at the platform fraction', () => {
    const r = estimateVram({ ...baseConfig, hardware: mac128 })
    expect(r.usableGb).toBe(96)
    expect(r.usablePerDeviceGb).toBe(96)
  })

  it('discloses the cap as an assumption', () => {
    const r = estimateVram({ ...baseConfig, hardware: mac128 })
    expect(r.assumptions.some((a) => a.id === 'unified-cap')).toBe(true)
  })
})

describe('estimateVram — multi-device', () => {
  const twoGpus: CalcConfig = { ...baseConfig, deviceCount: 2 }

  it('shards weights and KV, replicates a small weight fraction and per-device overhead', () => {
    const w = 16e9 / GIB
    const r = estimateVram(twoGpus)
    const perDeviceWeights = w * ((1 - REPLICATED_WEIGHT_FRACTION) / 2 + REPLICATED_WEIGHT_FRACTION)
    expect(r.perDeviceGb).toBeCloseTo(perDeviceWeights + 1 / 2 + 0.5 + 1, 6)
    // aggregate components sum to the aggregate total
    expect(r.weightsGb).toBeCloseTo(w * (1 - REPLICATED_WEIGHT_FRACTION + 2 * REPLICATED_WEIGHT_FRACTION), 6)
    expect(r.activationsGb).toBe(1) // replicated per device
    expect(r.overheadGb).toBe(2)
    expect(r.totalGb).toBeCloseTo(r.weightsGb + r.kvCacheGb + r.activationsGb + r.overheadGb, 6)
    expect(r.totalGb).toBeCloseTo(2 * r.perDeviceGb, 6)
  })

  it('a model that cannot fit one device can fit two', () => {
    const big: CalcConfig = {
      ...baseConfig,
      model: { ...gqaModel, paramsTotalB: 24 }, // 44.7 GiB of FP16 weights
    }
    expect(estimateVram(big).verdict).toBe('wont-fit')
    expect(estimateVram({ ...big, deviceCount: 3 }).verdict).toBe('fits')
  })

  it('judges utilization per device', () => {
    const r = estimateVram(twoGpus)
    expect(r.usableGb).toBe(48)
    expect(r.usablePerDeviceGb).toBe(24)
    expect(r.utilization).toBeCloseTo(r.perDeviceGb / 24, 9)
  })

  it('discloses the multi-device estimate as an assumption', () => {
    expect(estimateVram(twoGpus).assumptions.some((a) => a.id === 'multi-device')).toBe(true)
    expect(estimateVram(baseConfig).assumptions.some((a) => a.id === 'multi-device')).toBe(false)
  })
})

describe('estimateVram — assumptions & disclosure', () => {
  it('always discloses the activation estimate and the framework KV dtype', () => {
    const r = estimateVram(baseConfig)
    expect(r.assumptions.some((a) => a.id === 'activations-estimate')).toBe(true)
    const kv = r.assumptions.find((a) => a.id === 'kv-dtype')
    expect(kv).toBeDefined()
    expect(kv!.text).toContain('FP16')
  })

  it('flags multi-chip SKUs (aggregate-memory assumption)', () => {
    const t3k: HardwareSpec = {
      id: 't3k',
      vendor: 'Tenstorrent',
      name: 'T3K',
      memoryGb: 96,
      memoryType: 'GDDR6',
      unified: false,
      numChips: 8,
      memoryPerChipGb: 12,
      sources: [],
    }
    const r = estimateVram({ ...baseConfig, hardware: t3k })
    expect(r.assumptions.some((a) => a.id === 'multi-chip')).toBe(true)
  })

  it('warns when the requested context exceeds the model maximum', () => {
    const r = estimateVram({ ...baseConfig, contextLength: 262144 })
    expect(r.assumptions.some((a) => a.id === 'context-exceeds-max')).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// Tenstorrent per-chip mesh model (source-derived from tt-metal/tt-vllm).
// Triggered by hardware.usableGbPerChip; judges the verdict PER CHIP.
// ---------------------------------------------------------------------------

const ttFw: FrameworkSpec = {
  id: 'tt-metal',
  label: 'Tenstorrent',
  kvBytesDefault: 1.0625,
  kvDtypeLabel: 'BFP8',
  baselineOverheadGb: 0.3,
  activation: { chunkTokens: 4096, multiplier: 6 },
  sources: [],
}

const ttChip = (id: string, numChips: number): HardwareSpec => ({
  id,
  vendor: 'Tenstorrent',
  name: id,
  memoryGb: 12 * numChips,
  memoryType: 'GDDR6',
  unified: false,
  numChips,
  memoryPerChipGb: 12,
  usableGbPerChip: 12,
  sources: [],
})

const moeModel: ModelSpec = { ...gqaModel, id: 'moe', moe: true, paramsActiveB: 3 }
const mlaModelTT: ModelSpec = {
  id: 'mla-tt',
  family: 'T',
  name: 'mla',
  hfId: 'x/mla',
  paramsTotalB: 8,
  moe: false,
  hiddenSize: 4096,
  maxContext: 131072,
  nativeDtype: 'bf16',
  attention: { kind: 'mla', numLayers: 32, kvLoraRank: 512, qkRopeHeadDim: 64 },
  sources: [],
}

const ttCfg = (model: ModelSpec, hw: HardwareSpec, extra: Partial<CalcConfig> = {}): CalcConfig => ({
  model,
  quant: fp16,
  hardware: hw,
  deviceCount: 1,
  framework: ttFw,
  contextLength: 8192,
  concurrentSequences: 1,
  ...extra,
})

describe('estimateVram — Tenstorrent per-chip mesh model', () => {
  it('weights use TT block-float bpp (≈0.85 dense), independent of the selected quant', () => {
    const fp16r = estimateVram(ttCfg(gqaModel, ttChip('t', 8)))
    const q4r = estimateVram(ttCfg(gqaModel, ttChip('t', 8), { quant: { id: 'q4', label: 'q4', bitsPerWeight: 4.5 } }))
    expect(fp16r.weightsGb).toBeCloseTo(q4r.weightsGb, 9) // quant does not change TT weights
    const expected = ((8e9 * 0.85) / GIB) * (1 - 0.005 + 8 * 0.005) // aggregate, r=0.005, shards=8
    expect(fp16r.weightsGb).toBeCloseTo(expected, 5)
  })

  it('MoE weights use ≈0.73 B/param (experts at BFP4 dominate)', () => {
    const r = estimateVram(ttCfg(moeModel, ttChip('t', 8)))
    const expected = ((8e9 * 0.73) / GIB) * (1 - 0.005 + 8 * 0.005)
    expect(r.weightsGb).toBeCloseTo(expected, 5)
  })

  it('usable capacity is per chip (12 GiB), aggregated across the mesh', () => {
    const r = estimateVram(ttCfg(gqaModel, ttChip('t', 8)))
    expect(r.usablePerDeviceGb).toBe(12)
    expect(r.usableGb).toBe(12 * 8)
  })

  it('judges the verdict per chip — a 32B dense model fits 8 chips but not 1', () => {
    const big: ModelSpec = { ...gqaModel, paramsTotalB: 32 }
    expect(estimateVram(ttCfg(big, ttChip('one', 1))).verdict).toBe('wont-fit')
    expect(estimateVram(ttCfg(big, ttChip('eight', 8))).verdict).toBe('fits')
  })

  it('GQA KV shards across chips (aggregate KV ~constant up to head count)', () => {
    const one = estimateVram(ttCfg(gqaModel, ttChip('one', 1)))
    const eight = estimateVram(ttCfg(gqaModel, ttChip('eight', 8))) // 8 kv heads, 8 chips
    expect(eight.kvCacheGb).toBeCloseTo(one.kvCacheGb, 5)
  })

  it('MLA KV replicates across chips (aggregate KV grows with chip count)', () => {
    const one = estimateVram(ttCfg(mlaModelTT, ttChip('one', 1)))
    const eight = estimateVram(ttCfg(mlaModelTT, ttChip('eight', 8)))
    expect(eight.kvCacheGb).toBeCloseTo(one.kvCacheGb * 8, 5)
    expect(eight.assumptions.some((a) => a.id === 'tt-mla-replicated')).toBe(true)
  })

  it('uses BFP8 KV (1.0625 B/elem) regardless of the framework dropdown', () => {
    // pairing a TT card with a non-TT framework still models block-float KV
    const r = estimateVram(ttCfg(gqaModel, ttChip('t', 8), { framework: baseFramework }))
    const expectedKvTotal = kvCacheBytes(gqaModel.attention, 8192, 1, 1.0625) / GIB
    expect(r.kvCacheGb).toBeCloseTo(expectedKvTotal, 5) // 8 heads / 8 chips → aggregate == total
  })

  it('discloses block-float weights, per-chip sizing, and KV dtype', () => {
    const r = estimateVram(ttCfg(gqaModel, ttChip('t', 8)))
    expect(r.assumptions.some((a) => a.id === 'tt-blockfloat')).toBe(true)
    expect(r.assumptions.some((a) => a.id === 'tt-per-chip')).toBe(true)
    expect(r.assumptions.some((a) => a.id === 'kv-dtype')).toBe(true)
    expect(r.assumptions.some((a) => a.id === 'tt-token-budget')).toBe(true)
  })
})

describe('estimateVram — input validation', () => {
  it('rejects non-positive or fractional device counts', () => {
    expect(() => estimateVram({ ...baseConfig, deviceCount: 0 })).toThrow(RangeError)
    expect(() => estimateVram({ ...baseConfig, deviceCount: 1.5 })).toThrow(RangeError)
  })

  it('rejects non-positive context length and sequence count', () => {
    expect(() => estimateVram({ ...baseConfig, contextLength: 0 })).toThrow(RangeError)
    expect(() => estimateVram({ ...baseConfig, concurrentSequences: 0 })).toThrow(RangeError)
  })
})
