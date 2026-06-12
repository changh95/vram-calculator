import type { QuantScheme } from '../types'

/**
 * Weight quantization schemes. `bitsPerWeight` is the WHOLE-CHECKPOINT effective
 * bits per weight — it folds in quantization metadata (group scales/zero-points)
 * and the higher-precision tensors (embeddings, output head, norms) that quant
 * formats leave at FP16. Values from notes/vram-formula.md (llama.cpp
 * tools/quantize/README.md measured table on Llama-3.1-8B; ggml-common.h
 * static_asserts; AutoGPTQ/AutoAWQ pack layout k + (16+k)/group_size).
 *
 * Because the fixed-precision tensors are a larger share of small models than
 * large ones, these are estimates (±2–5%) for any given model — hence
 * `estimate: true` on the quantized entries.
 */
export const QUANTS: QuantScheme[] = [
  { id: 'bf16', label: 'BF16', bitsPerWeight: 16, note: 'BFloat16 — the native dtype of most modern checkpoints; 2 bytes/param.' },
  { id: 'fp16', label: 'FP16', bitsPerWeight: 16, note: 'Half precision — 2 bytes/param; memory-identical to BF16.' },
  {
    id: 'fp8',
    label: 'FP8 (E4M3)',
    bitsPerWeight: 8.05,
    note: 'Native 8-bit float; ~1 byte/param plus negligible block scales. Ada/Hopper/Blackwell, MI300+.',
    estimate: true,
  },
  {
    id: 'int8',
    label: 'INT8 / W8A8',
    bitsPerWeight: 8.2,
    note: 'GPTQ/SmoothQuant 8-bit with group scales.',
    estimate: true,
  },
  {
    id: 'q8_0',
    label: 'GGUF Q8_0',
    bitsPerWeight: 8.5,
    note: 'llama.cpp 8-bit (8.5 bpw incl. block scale).',
    estimate: true,
  },
  {
    id: 'q6_k',
    label: 'GGUF Q6_K',
    bitsPerWeight: 6.56,
    note: 'Near-lossless 6-bit K-quant.',
    estimate: true,
  },
  {
    id: 'q5_k_m',
    label: 'GGUF Q5_K_M',
    bitsPerWeight: 5.7,
    note: '5-bit K-quant, medium mix.',
    estimate: true,
  },
  {
    id: 'q4_k_m',
    label: 'GGUF Q4_K_M',
    bitsPerWeight: 4.89,
    note: 'The local default — 4-bit K-quant, medium mix. ~0.61 B/param.',
    estimate: true,
  },
  {
    id: 'awq_gptq_int4',
    label: 'AWQ / GPTQ INT4',
    bitsPerWeight: 4.5,
    note: '4-bit GPU quant (group 128) with FP16 embeddings/head kept — whole-model average.',
    estimate: true,
  },
  {
    id: 'nvfp4',
    label: 'NVFP4 / MXFP4 (4-bit)',
    bitsPerWeight: 4.25,
    note: 'Blackwell/MoE 4-bit microscaling float (E2M1 + shared scale).',
    estimate: true,
  },
  {
    id: 'q3_k_m',
    label: 'GGUF Q3_K_M',
    bitsPerWeight: 4.0,
    note: '3-bit K-quant, medium mix — quality starts to drop.',
    estimate: true,
  },
  {
    id: 'iq2_m',
    label: 'GGUF IQ2_M',
    bitsPerWeight: 2.93,
    note: '2-bit i-quant — aggressive; for fitting very large models.',
    estimate: true,
  },
]

/**
 * Tenstorrent block-float precision profiles, shown in place of the GPU quant
 * ladder when a TT device is selected (TT runs block-float, not GGUF/AWQ). The
 * profile sets both the weight B/param (MoE vs dense) and the KV dtype.
 * Source: tt-metal DecodersPrecision (notes/tenstorrent-memory.md). `bitsPerWeight`
 * is a nominal display value; the TT path uses the ttWeightBpp and ttKvBytes fields.
 */
export const TT_QUANTS: QuantScheme[] = [
  {
    id: 'tt-performance',
    label: 'Block-float · performance',
    bitsPerWeight: 6,
    note: 'Default tt-metal profile: BFP8 attention + BFP4 MLP, BFP8 KV. The serving default.',
    estimate: true,
    ttWeightBppMoe: 0.73,
    ttWeightBppDense: 0.85,
    ttKvBytes: 1.0625,
    ttKvLabel: 'BFP8 (bfloat8_b)',
  },
  {
    id: 'tt-accuracy',
    label: 'Block-float · accuracy',
    bitsPerWeight: 9,
    note: 'Accuracy profile: more BF16 (attention + KV), heavier but higher-fidelity.',
    estimate: true,
    ttWeightBppMoe: 1.0,
    ttWeightBppDense: 1.2,
    ttKvBytes: 2.0,
    ttKvLabel: 'BF16',
  },
]
