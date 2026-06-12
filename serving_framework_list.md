# Supported Serving Frameworks

Mirrors `src/lib/data/frameworks.ts` — keep in sync. The calculator uses each
framework's **default KV dtype**, fixed **baseline overhead** (per device), and
**chunked-prefill cap** (bounds the activation estimate). See
`notes/serving-framework-vram.md` for mechanisms and sources.

| Framework | KV default | KV bytes/elem | Baseline | Chunk | Hardware |
|---|---|---|---|---|---|
| vLLM | FP16/BF16 (auto) | 2 | 2.0 GB | 8192 | NVIDIA, AMD, Intel, TPU, CPU |
| SGLang | FP16/BF16 (auto) | 2 | 2.0 GB | 8192 | NVIDIA, AMD, Intel, TPU, Ascend |
| llama.cpp | F16 | 2 | 0.6 GB | 2048 | CUDA, ROCm, Metal, Vulkan, CPU |
| Ollama | F16 | 2 | 0.6 GB | 512 | CUDA, ROCm, Metal, CPU |
| TensorRT-LLM | FP16/BF16 (auto; FP8 on quant ckpts) | 2 | 1.5 GB | 8192 | NVIDIA only |
| MLX (Apple) | FP16/BF16 | 2 | 0.3 GB | 2048 | Apple Silicon |
| Tenstorrent (tt-vllm / tt-metal) | bfloat8_b (BFP8) | 1.0625 | 0.3 GB | — | Tenstorrent WH/BH |

## What each one is good at

- **vLLM** — PagedAttention (<4% KV waste), broad quant + hardware support;
  preallocates 92% of VRAM. The default high-throughput server.
- **SGLang** — RadixAttention prefix caching (huge win on shared prefixes),
  native MLA pools, HiCache tiering. `mem_fraction_static` covers weights+KV
  only — reserve 5–8 GB extra.
- **llama.cpp** — deterministic contiguous KV, GGUF quant ladder, partial GPU
  offload, MoE CPU expert offload, Metal/Vulkan/ROCm. The hobbyist baseline.
- **Ollama** — llama.cpp wrapper; KV = parallel × context, VRAM-tiered default
  context, auto CPU spill (won't hard-OOM). Easiest desktop UX.
- **TensorRT-LLM** — lowest runtime footprint after tuning, FP8/NVFP4 KV,
  Blackwell-class. NVIDIA only.
- **MLX** — no preallocation, lowest overhead, lazy KV growth; bounded by the
  macOS unified-memory cap. The Apple-native path.
- **Tenstorrent stack** — vLLM fork on tt-metal; block-float (BFP8/BFP4) weights
  and KV by default; fixed per-model/device token budgets, no profiling.
  tt-inference-server and TT-Studio are orchestration layers with the same
  memory behavior. First-class so TT verdicts reflect what TT users actually run.

## Modeling note

vLLM, SGLang, and TensorRT-LLM **pre-allocate** KV to a fraction of VRAM at
startup, so their resident footprint exceeds what a model strictly needs. The
calculator estimates **minimum feasible** memory (weights + actual KV +
activations + baseline) so the verdict answers "can it physically fit"; the UI
notes the preallocation. KV-cache quantization (FP8/INT8/Q4) is derived from the
framework default and disclosed — it is not a separate user control in v1.
