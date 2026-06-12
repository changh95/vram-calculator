import type { FrameworkSpec } from '../types'

/**
 * Serving frameworks. Each entry models, per notes/serving-framework-vram.md:
 *  - kvBytesDefault / kvDtypeLabel — the KV-cache element precision out of the box
 *    (most frameworks follow the model dtype = 2 bytes; the TT stack defaults to
 *    block-float bfp8_b ≈ 1.0625 B/elem).
 *  - kvWasteFactor — paged-block internal fragmentation (≈ block_size/2 tokens per
 *    sequence; small).
 *  - baselineOverheadGb — fixed per-device runtime footprint (CUDA/Metal context,
 *    CUDA graphs, framework buffers). Estimate.
 *  - activation.{chunkTokens, multiplier} — workspace memory estimate; chunkTokens
 *    is the chunked-prefill cap that bounds peak activations regardless of prompt
 *    length. multiplier is calibrated so an 8B/hidden-4096 model lands near the
 *    framework's reported activation peak (vLLM ≈ 1.3 GiB @ 8192 tokens).
 *
 * NOTE: vLLM/SGLang/TRT-LLM PRE-ALLOCATE KV to a fraction of VRAM at startup, so
 * their real resident footprint is larger than this "minimum feasible" estimate;
 * the UI discloses that. We model minimum feasible memory (weights + actual KV +
 * activations + baseline) so the verdict answers "can it physically fit".
 */
export const FRAMEWORKS: FrameworkSpec[] = [
  {
    id: 'vllm',
    label: 'vLLM',
    kvBytesDefault: 2,
    kvDtypeLabel: 'FP16/BF16 (auto)',
    kvWasteFactor: 1.02, // PagedAttention, block 16 → <4% waste
    baselineOverheadGb: 2,
    activation: { chunkTokens: 8192, multiplier: 16 },
    sources: ['https://docs.vllm.ai/en/latest/configuration/conserving_memory/', 'https://blog.vllm.ai/2023/06/20/vllm.html'],
  },
  {
    id: 'sglang',
    label: 'SGLang',
    kvBytesDefault: 2,
    kvDtypeLabel: 'FP16/BF16 (auto)',
    kvWasteFactor: 1.0, // token-level paging, page_size 1
    baselineOverheadGb: 2,
    activation: { chunkTokens: 8192, multiplier: 16 },
    sources: [
      'https://docs.sglang.io/advanced_features/hyperparameter_tuning.html',
      'https://arxiv.org/abs/2312.07104',
    ],
  },
  {
    id: 'llamacpp',
    label: 'llama.cpp',
    kvBytesDefault: 2,
    kvDtypeLabel: 'F16',
    kvWasteFactor: 1.0, // contiguous allocation, exact for configured context
    baselineOverheadGb: 0.6, // CUDA context ~0.3–0.8 GB; ~0 on Metal
    activation: { chunkTokens: 2048, multiplier: 6 }, // flash-attention default → flatter
    sources: [
      'https://github.com/ggml-org/llama.cpp/discussions/10068',
      'https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md',
    ],
  },
  {
    id: 'ollama',
    label: 'Ollama',
    kvBytesDefault: 2,
    kvDtypeLabel: 'F16',
    kvWasteFactor: 1.0,
    baselineOverheadGb: 0.6,
    activation: { chunkTokens: 512, multiplier: 6 }, // n_batch fixed at 512
    sources: ['https://docs.ollama.com/faq', 'https://ollama.com/blog/new-model-scheduling'],
  },
  {
    id: 'trtllm',
    label: 'TensorRT-LLM',
    kvBytesDefault: 2,
    kvDtypeLabel: 'FP16/BF16 (auto, FP8 on quantized ckpts)',
    kvWasteFactor: 1.02, // paged, block 32
    baselineOverheadGb: 1.5,
    activation: { chunkTokens: 8192, multiplier: 14 },
    sources: [
      'https://nvidia.github.io/TensorRT-LLM/reference/memory.html',
      'https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html',
    ],
  },
  {
    id: 'mlx',
    label: 'MLX (Apple)',
    kvBytesDefault: 2,
    kvDtypeLabel: 'FP16/BF16',
    kvWasteFactor: 1.0, // lazy step-256 growth; rounding waste negligible
    baselineOverheadGb: 0.3,
    activation: { chunkTokens: 2048, multiplier: 6 },
    sources: [
      'https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py',
      'https://github.com/ml-explore/mlx-lm/blob/main/README.md',
    ],
  },
  {
    id: 'tt-metal',
    label: 'Tenstorrent (tt-vllm / tt-metal)',
    kvBytesDefault: 1.0625, // bfloat8_b block-float default
    kvDtypeLabel: 'bfloat8_b (BFP8)',
    kvWasteFactor: 1.02, // paged, block 64
    baselineOverheadGb: 0.3, // trace region + L1 scratch, per device
    activation: { multiplier: 6 }, // no chunked prefill; prefill tiled in DRAM
    sources: [
      'https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/vLLM_integration.md',
      'https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py',
    ],
  },
]
