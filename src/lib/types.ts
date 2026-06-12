/**
 * Core types for the VRAM estimation engine.
 *
 * Everything the math needs travels through these types — the formula module
 * (`estimateVram.ts`) never looks up registries, so it stays a pure function
 * of its inputs. Data tables under `src/lib/data/` produce these shapes.
 */

/** GQA also covers classic MHA: set numKvHeads === numAttnHeads. */
export interface GqaAttention {
  kind: 'gqa'
  numLayers: number
  numKvHeads: number
  headDim: number
}

/**
 * DeepSeek-style Multi-head Latent Attention. The cache stores one compressed
 * latent (kvLoraRank) plus the decoupled RoPE key (qkRopeHeadDim) per token
 * per layer — note: no K/V factor of 2.
 */
export interface MlaAttention {
  kind: 'mla'
  numLayers: number
  kvLoraRank: number
  qkRopeHeadDim: number
  /**
   * DSA lightning-indexer key cache (DeepSeek V3.2, GLM-5): extra elements
   * per token per layer, stored FP8 (1 byte) regardless of KV dtype.
   */
  indexerElemsPerTokenPerLayer?: number
}

/**
 * Gemma-style interleaved attention: `globalLayers` see the full context,
 * `localLayers` only keep `slidingWindow` tokens of KV. Gemma 4 global layers
 * use different KV geometry than local ones (e.g. 1 KV head × 512 dim vs
 * 8 × 256), and KV-sharing variants (E2B/E4B) have fewer distinct-KV layers
 * than `numLayers` — so globalLayers + localLayers counts only layers holding
 * their own KV and may be less than numLayers.
 */
export interface GqaSwaAttention {
  kind: 'gqa-swa'
  numLayers: number
  numKvHeads: number
  headDim: number
  slidingWindow: number
  globalLayers: number
  localLayers: number
  /** Global-layer KV heads when they differ from numKvHeads (Gemma 4). */
  globalKvHeads?: number
  /** Global-layer head dim when it differs from headDim (Gemma 4). */
  globalHeadDim?: number
}

/**
 * DeepSeek-V4 CSA/HCA compressed attention: every layer keeps a short
 * uncompressed local window plus a 1/compressRatio-compressed cache of
 * kvDim-wide entries; compressRatio 0 means pure sliding-window (window
 * only). CSA layers additionally carry an FP8 indexer cache.
 */
export interface CompressedAttention {
  kind: 'compressed'
  numLayers: number
  /** Cached elements per entry (V4: head_dim 512 — RoPE dims live inside it). */
  kvDim: number
  /** Uncompressed local window kept by every layer (V4: 128 tokens). */
  windowTokens: number
  layerGroups: readonly {
    count: number
    /** 4 = CSA, 128 = HCA, 0 = pure sliding window. */
    compressRatio: number
    /** FP8 indexer cache dims per entry (CSA layers only). */
    indexerDim?: number
    /** Indexer keeps ctx/indexerRatio entries. */
    indexerRatio?: number
  }[]
}

/**
 * Mamba/SSM-Transformer hybrids (Nemotron-H class): only `attentionLayers`
 * hold KV; SSM layers carry a small context-independent per-sequence state.
 */
export interface HybridAttention {
  kind: 'hybrid'
  numLayers: number
  attentionLayers: number
  numKvHeads: number
  headDim: number
  /** Recurrent state bytes per sequence across all SSM layers (context-independent). */
  ssmStateBytesPerSeq?: number
}

export type AttentionConfig =
  | GqaAttention
  | MlaAttention
  | GqaSwaAttention
  | HybridAttention
  | CompressedAttention

export interface ModelSpec {
  id: string
  family: string
  name: string
  hfId: string
  /** Total parameters in billions (decimal, 1e9). Weights memory uses this. */
  paramsTotalB: number
  /** Active parameters in billions for MoE models (informational). */
  paramsActiveB?: number
  moe: boolean
  attention: AttentionConfig
  hiddenSize: number
  maxContext: number
  /** dtype the weights were released in, e.g. 'bf16', 'fp8'. */
  nativeDtype: string
  /** True when any architecture field is not verified from a primary source. */
  estimate?: boolean
  sources: string[]
}

export interface QuantScheme {
  id: string
  label: string
  /**
   * Effective bits per weight averaged over the whole checkpoint, including
   * quantization metadata (scales/zero-points) and higher-precision tensors.
   */
  bitsPerWeight: number
  note?: string
  estimate?: boolean
  /**
   * Tenstorrent block-float profile (Performance / Accuracy). Used ONLY on the
   * TT path, where weights run block-float regardless of GPU quant formats —
   * these override bitsPerWeight there. Block-float B/param differs by MoE vs
   * dense; KV dtype is part of the profile too. See notes/tenstorrent-memory.md.
   */
  ttWeightBppMoe?: number
  ttWeightBppDense?: number
  ttKvBytes?: number
  ttKvLabel?: string
}

export interface HardwareSpec {
  id: string
  vendor: string
  name: string
  /** Aggregate memory of the SKU in GiB (multi-chip SKUs sum their chips). */
  memoryGb: number
  memoryType: string
  unified: boolean
  /** Fraction of memoryGb allocatable for LLM use (unified platforms). */
  usableFraction?: number
  usableNote?: string
  numChips?: number
  memoryPerChipGb?: number
  /**
   * Usable DRAM per chip (GiB) for multi-chip mesh accelerators (Tenstorrent).
   * When set, the estimator shards weights/KV across `numChips` and judges the
   * verdict PER CHIP — not against the flat aggregate. Full marketed per-chip
   * capacity: Wormhole 12, Blackhole 32. See notes/tenstorrent-memory.md.
   */
  usableGbPerChip?: number
  bandwidthGbs?: number
  /** Purchase page — drives the "Buy Now" link (set only for Tenstorrent). */
  buyUrl?: string
  estimate?: boolean
  sources: string[]
}

export interface FrameworkSpec {
  id: string
  label: string
  /** Bytes per KV-cache element under the framework's default policy. */
  kvBytesDefault: number
  kvDtypeLabel: string
  /** Multiplier ≥ 1 for KV allocation waste (paged block slack etc.). */
  kvWasteFactor?: number
  /** Fixed per-device runtime footprint in GiB (CUDA/Metal context, graphs, buffers). */
  baselineOverheadGb: number
  activation: {
    /** Max tokens processed in one forward step (chunked-prefill cap); undefined = full context. */
    chunkTokens?: number
    /** Empirical multiplier on hiddenSize × batchedTokens × 2 bytes. Estimate. */
    multiplier: number
  }
  estimate?: boolean
  sources: string[]
}

export interface CalcConfig {
  model: ModelSpec
  quant: QuantScheme
  hardware: HardwareSpec
  /** Number of identical devices/SKUs; ≥ 1. */
  deviceCount: number
  framework: FrameworkSpec
  contextLength: number
  /** Concurrent sequences (continuous batching merges batch × users into this). */
  concurrentSequences: number
}

export type Verdict = 'fits' | 'tight' | 'wont-fit'

export interface Assumption {
  id: string
  text: string
}

export interface MemoryBreakdown {
  /** All *Gb fields are GiB, aggregated across all devices. */
  weightsGb: number
  kvCacheGb: number
  activationsGb: number
  overheadGb: number
  totalGb: number
  perDeviceGb: number
  usableGb: number
  usablePerDeviceGb: number
  /** perDeviceGb / usablePerDeviceGb. */
  utilization: number
  verdict: Verdict
  /** Aggregate GiB over capacity when the verdict is 'wont-fit', else null. */
  shortfallGb: number | null
  assumptions: Assumption[]
}
