/**
 * The VRAM estimation engine — pure math, no React, no DOM.
 *
 * Total required memory = weights + KV cache + activations + runtime overhead.
 * Each term is a named, individually tested function. Derivations and sources
 * live in notes/vram-formula.md; estimate-grade terms are disclosed through
 * `MemoryBreakdown.assumptions` so the UI can mark them.
 */
import type {
  Assumption,
  AttentionConfig,
  CalcConfig,
  FrameworkSpec,
  MemoryBreakdown,
  Verdict,
} from './types'

export const GIB = 2 ** 30

/**
 * Fraction of weights replicated on every device under tensor parallelism
 * (embeddings / lm_head / norms that common TP implementations don't shard).
 * Estimate — see notes/vram-formula.md.
 */
export const REPLICATED_WEIGHT_FRACTION = 0.03

/** Weight memory in bytes: total params (billions) × effective bits per weight. */
export function weightsBytes(paramsTotalB: number, bitsPerWeight: number): number {
  return paramsTotalB * 1e9 * (bitsPerWeight / 8)
}

/**
 * KV-cache bytes for `sequences` sequences of `contextLength` tokens.
 * Branches on attention architecture — getting this wrong per-architecture is
 * the classic VRAM-calculator failure mode (GQA vs MLA vs sliding-window can
 * differ by an order of magnitude at the same parameter count).
 */
export function kvCacheBytes(
  attn: AttentionConfig,
  contextLength: number,
  sequences: number,
  kvBytesPerElement: number,
): number {
  switch (attn.kind) {
    case 'gqa':
      // K and V, one per layer per kv-head per head-dim element per token.
      return 2 * attn.numLayers * attn.numKvHeads * attn.headDim * kvBytesPerElement * contextLength * sequences
    case 'mla': {
      // Compressed latent + decoupled RoPE key per token per layer — no ×2.
      const latent =
        attn.numLayers * (attn.kvLoraRank + attn.qkRopeHeadDim) * kvBytesPerElement * contextLength * sequences
      // DSA lightning-indexer keys are stored FP8 (1 byte) regardless of KV dtype.
      const indexer =
        attn.numLayers * (attn.indexerElemsPerTokenPerLayer ?? 0) * contextLength * sequences
      return latent + indexer
    }
    case 'gqa-swa': {
      const localPerLayerToken = 2 * attn.numKvHeads * attn.headDim * kvBytesPerElement
      const globalPerLayerToken =
        2 * (attn.globalKvHeads ?? attn.numKvHeads) * (attn.globalHeadDim ?? attn.headDim) * kvBytesPerElement
      const local = attn.localLayers * Math.min(contextLength, attn.slidingWindow) * localPerLayerToken
      const global = attn.globalLayers * contextLength * globalPerLayerToken
      return (local + global) * sequences
    }
    case 'hybrid': {
      const kv =
        2 * attn.attentionLayers * attn.numKvHeads * attn.headDim * kvBytesPerElement * contextLength * sequences
      return kv + (attn.ssmStateBytesPerSeq ?? 0) * sequences
    }
    case 'compressed': {
      // Every layer caches a short local window plus a compressed cache of
      // kvDim-wide entries; ratio 0 = pure sliding window. CSA layers add an
      // FP8 indexer cache of ctx/indexerRatio entries.
      let total = 0
      for (const g of attn.layerGroups) {
        const cachedTokens = Math.min(
          contextLength,
          attn.windowTokens + (g.compressRatio > 0 ? contextLength / g.compressRatio : 0),
        )
        total += g.count * cachedTokens * attn.kvDim * kvBytesPerElement
        if (g.indexerDim && g.indexerRatio) {
          total += g.count * (contextLength / g.indexerRatio) * g.indexerDim
        }
      }
      return total * sequences
    }
  }
}

/**
 * Inference activation / workspace bytes — an estimate by nature.
 * Scales with the largest number of tokens in flight in one forward step:
 * chunked-prefill frameworks cap that at `chunkTokens`; the decode batch can
 * still exceed the cap when many sequences decode at once.
 */
export function activationBytes(
  hiddenSize: number,
  contextLength: number,
  sequences: number,
  activation: FrameworkSpec['activation'],
): number {
  const prefillTokens = Math.min(contextLength, activation.chunkTokens ?? contextLength)
  const batchedTokens = Math.max(prefillTokens, sequences)
  return activation.multiplier * hiddenSize * batchedTokens * 2
}

/** Verdict thresholds per CLAUDE.md: >100% won't fit, >90% tight, else fits. */
export function verdictFromUtilization(utilization: number): Verdict {
  if (utilization > 1) return 'wont-fit'
  if (utilization > 0.9) return 'tight'
  return 'fits'
}

// --- Tenstorrent mesh model (source-derived; see notes/tenstorrent-memory.md) ---
/** Effective block-float bytes/param under the default 'performance' profile:
 * BFP8 attention (1.0625) + BFP4 MLP (0.5625). MoE experts (FFN) dominate → ~0.73;
 * dense models are FFN-heavy but keep more BFP8 → ~0.85. Estimate. */
export const TT_BYTES_PER_PARAM_MOE = 0.73
export const TT_BYTES_PER_PARAM_DENSE = 0.85
/** Only RMSNorm weights replicate across the mesh (attn/MLP/embeddings/head shard). */
export const REPLICATED_WEIGHT_FRACTION_TT = 0.005
/** KV cache element size: ttnn.bfloat8_b default. */
export const TT_KV_BYTES = 1.0625
/** Per-chip fixed DRAM overhead (trace region + misc); trace varies per model/device. */
export const TT_OVERHEAD_GB_PER_CHIP = 0.3
const TT_ACTIVATION = { chunkTokens: 4096, multiplier: 6 } // internal chunked prefill; small

/** How many ways KV shards across the mesh: by KV head for GQA-family; MLA and
 * compressed decode as MQA so their KV REPLICATES on every chip (cap 1). */
function ttKvShardCap(attn: AttentionConfig): number {
  return attn.kind === 'mla' || attn.kind === 'compressed' ? 1 : attn.numKvHeads
}

function estimateTenstorrent(config: CalcConfig): MemoryBreakdown {
  const { model, quant, hardware, deviceCount, contextLength, concurrentSequences } = config
  const shards = (hardware.numChips ?? 1) * deviceCount
  // Block-float profile from the selected quant's TT fields (Performance /
  // Accuracy); falls back to the performance default when a GPU quant id is
  // carried over from non-TT hardware.
  const bpp = model.moe
    ? (quant.ttWeightBppMoe ?? TT_BYTES_PER_PARAM_MOE)
    : (quant.ttWeightBppDense ?? TT_BYTES_PER_PARAM_DENSE)
  const kvBytes = quant.ttKvBytes ?? TT_KV_BYTES
  const kvLabel = quant.ttKvLabel ?? 'BFP8 (bfloat8_b)'

  const weights = (model.paramsTotalB * 1e9 * bpp) / GIB
  const kv = kvCacheBytes(model.attention, contextLength, concurrentSequences, kvBytes) / GIB
  const activations = activationBytes(model.hiddenSize, contextLength, concurrentSequences, TT_ACTIVATION) / GIB

  const r = shards > 1 ? REPLICATED_WEIGHT_FRACTION_TT : 0
  const kvDivisor = Math.min(shards, ttKvShardCap(model.attention)) // MLA → 1 (replicated)

  const perChipWeights = weights * ((1 - r) / shards + r)
  const perChipKv = kv / kvDivisor
  const perChipAct = activations / shards
  const perChipGb = perChipWeights + perChipKv + perChipAct + TT_OVERHEAD_GB_PER_CHIP

  const usablePerDeviceGb = hardware.usableGbPerChip!
  const utilization = perChipGb / usablePerDeviceGb
  const verdict = verdictFromUtilization(utilization)
  const usableGb = usablePerDeviceGb * shards
  const totalGb = perChipGb * shards

  const assumptions: Assumption[] = [
    {
      id: 'tt-blockfloat',
      text: `Tenstorrent weights run block-float ≈ ${bpp} B/param (${quant.label}). "Performance" = BFP8 attention + BFP4 MLP; "Accuracy" keeps more in BF16. GPU quant formats (GGUF/AWQ) don't apply on TT.`,
    },
    {
      id: 'kv-dtype',
      text: `KV cache assumed ${kvLabel}, set by the "${quant.label}" profile. KV is usually a small share of TT memory, but this matters at long context or for replicated MLA KV.`,
    },
    {
      id: 'tt-per-chip',
      text: `Sized per chip: ${shards} chip${shards > 1 ? 's' : ''} × ${usablePerDeviceGb} GB usable DRAM each. Weights shard across the mesh (~${(REPLICATED_WEIGHT_FRACTION_TT * 100).toFixed(1)}% replicated); the verdict is judged per chip.`,
    },
    {
      id: 'activations-estimate',
      text: 'Activation/workspace memory is a heuristic estimate; decode activations live in on-chip SRAM, prefill in DRAM (internally chunked).',
    },
  ]
  if (shards > 1 && ttKvShardCap(model.attention) === 1) {
    assumptions.push({
      id: 'tt-mla-replicated',
      text: 'MLA / compressed-attention KV cache decodes as MQA and replicates on every chip, so it does not shrink as you add chips.',
    })
  }
  assumptions.push({
    id: 'tt-token-budget',
    text: 'In deployment the Tenstorrent stack pre-allocates KV to a fixed pooled token budget; per-model/device context caps (e.g. Llama-8B on N150 ≈ 32K) may apply.',
  })
  if (contextLength > model.maxContext) {
    assumptions.push({
      id: 'context-exceeds-max',
      text: `Requested context (${contextLength.toLocaleString()} tokens) exceeds the model's maximum (${model.maxContext.toLocaleString()}).`,
    })
  }

  return {
    weightsGb: perChipWeights * shards,
    kvCacheGb: perChipKv * shards,
    activationsGb: perChipAct * shards,
    overheadGb: TT_OVERHEAD_GB_PER_CHIP * shards,
    totalGb,
    perDeviceGb: perChipGb,
    usableGb,
    usablePerDeviceGb,
    utilization,
    verdict,
    shortfallGb: verdict === 'wont-fit' ? totalGb - usableGb : null,
    assumptions,
  }
}

export function estimateVram(config: CalcConfig): MemoryBreakdown {
  const { model, quant, hardware, deviceCount, framework, contextLength, concurrentSequences } = config

  if (!Number.isInteger(deviceCount) || deviceCount < 1) {
    throw new RangeError(`deviceCount must be a positive integer, got ${deviceCount}`)
  }
  if (contextLength < 1) throw new RangeError(`contextLength must be ≥ 1, got ${contextLength}`)
  if (concurrentSequences < 1) {
    throw new RangeError(`concurrentSequences must be ≥ 1, got ${concurrentSequences}`)
  }

  // Multi-chip mesh accelerators (Tenstorrent) shard across chips and are judged
  // per chip with block-float weights — a different model from discrete GPUs.
  if (hardware.usableGbPerChip !== undefined) return estimateTenstorrent(config)

  const weights = weightsBytes(model.paramsTotalB, quant.bitsPerWeight) / GIB
  const kv =
    (kvCacheBytes(model.attention, contextLength, concurrentSequences, framework.kvBytesDefault) *
      (framework.kvWasteFactor ?? 1)) /
    GIB
  const activationsPerDevice =
    activationBytes(model.hiddenSize, contextLength, concurrentSequences, framework.activation) / GIB

  // Tensor-parallel sharding model over all GPUs: `gpusPerNode` (e.g. an 8-GPU
  // HGX/DGX node) × deviceCount. A REPLICATED_WEIGHT_FRACTION copy of the weights
  // lives on every GPU; the rest shards evenly. KV shards evenly; activations and
  // the runtime baseline are paid per GPU. Verdict is judged per GPU.
  const shards = deviceCount * (hardware.gpusPerNode ?? 1)
  const r = shards > 1 ? REPLICATED_WEIGHT_FRACTION : 0
  const perDeviceWeights = weights * ((1 - r) / shards + r)
  const perDeviceGb =
    perDeviceWeights + kv / shards + activationsPerDevice + framework.baselineOverheadGb

  // Aggregate view: components sum to shards × perDevice.
  const weightsGb = weights * (1 - r + shards * r)
  const activationsGb = activationsPerDevice * shards
  const overheadGb = framework.baselineOverheadGb * shards
  const totalGb = weightsGb + kv + activationsGb + overheadGb

  const usablePerDeviceGb = hardware.memoryGb * (hardware.unified ? (hardware.usableFraction ?? 1) : 1)
  const usableGb = usablePerDeviceGb * shards
  const utilization = perDeviceGb / usablePerDeviceGb
  const verdict = verdictFromUtilization(utilization)

  const assumptions: Assumption[] = [
    {
      id: 'activations-estimate',
      text: 'Activation/workspace memory is a heuristic estimate; real usage varies by framework version and kernel selection.',
    },
    {
      id: 'kv-dtype',
      text: `KV cache assumed ${framework.kvDtypeLabel} — the ${framework.label} default and the heaviest common dtype. KV quantization (FP8/INT8/Q8/Q4) is widely supported and can cut KV memory 2–4×, so a long-context config may fit with less than shown.`,
    },
  ]
  if (hardware.unified) {
    assumptions.push({
      id: 'unified-cap',
      text:
        hardware.usableNote ??
        `Unified memory: ${Math.round((hardware.usableFraction ?? 1) * 100)}% of ${hardware.memoryGb} GB assumed allocatable for the GPU.`,
    })
  }
  if (shards > 1) {
    assumptions.push({
      id: 'multi-device',
      text: `Sharded across ${shards} GPUs (tensor-parallel): weights/KV split evenly with ~${Math.round(REPLICATED_WEIGHT_FRACTION * 100)}% replicated weights and per-GPU runtime overhead; the verdict is judged per GPU. Real TP overhead varies.`,
    })
  }
  if ((hardware.numChips ?? 1) > 1) {
    assumptions.push({
      id: 'multi-chip',
      text: `${hardware.name} aggregates ${hardware.numChips} chips (${hardware.memoryPerChipGb} GB each); efficient cross-chip sharding by the runtime is assumed.`,
    })
  }
  if (contextLength > model.maxContext) {
    assumptions.push({
      id: 'context-exceeds-max',
      text: `Requested context (${contextLength.toLocaleString()} tokens) exceeds the model's maximum (${model.maxContext.toLocaleString()}); memory is computed anyway, but the model can't natively use it.`,
    })
  }

  return {
    weightsGb,
    kvCacheGb: kv,
    activationsGb,
    overheadGb,
    totalGb,
    perDeviceGb,
    usableGb,
    usablePerDeviceGb,
    utilization,
    verdict,
    shortfallGb: verdict === 'wont-fit' ? totalGb - usableGb : null,
    assumptions,
  }
}
