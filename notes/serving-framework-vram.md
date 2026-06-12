# Serving Framework VRAM behavior

How each framework manages KV cache and overhead, and the VRAM-saving techniques
it offers (with magnitudes). Drives `src/lib/data/frameworks.ts`. Verified
against framework source/docs June 2026.

> **Modeling choice.** vLLM/SGLang/TRT-LLM **pre-allocate** KV to a fraction of
> VRAM at startup, so their resident footprint is larger than what a model
> strictly needs. The calculator models **minimum feasible** memory (weights +
> actual KV + activations + baseline) so the verdict answers "can it physically
> fit"; the UI discloses the preallocation behavior.

## Per-framework table values

| id | KV dtype default | KV bytes/elem | waste | baseline GB | chunk tokens |
|---|---|---|---|---|---|
| vllm | FP16/BF16 (auto) | 2 | 1.02 | 2.0 | 8192 |
| sglang | FP16/BF16 (auto) | 2 | 1.00 | 2.0 | 8192 |
| llamacpp | F16 | 2 | 1.00 | 0.6 | 2048 |
| ollama | F16 | 2 | 1.00 | 0.6 | 512 |
| trtllm | FP16/BF16 (auto) | 2 | 1.02 | 1.5 | 8192 |
| mlx | FP16/BF16 | 2 | 1.00 | 0.3 | 2048 |
| tt-metal | bfloat8_b (BFP8) | 1.0625 | 1.02 | 0.3 | — |

## vLLM
- **PagedAttention**, block 16. Internal KV waste **<4%** vs 60–80% in naïve
  contiguous allocation. Prefix caching is free (reuses in-pool blocks).
- **Preallocates** `gpu_memory_utilization` of *total* VRAM (0.92 since v0.20,
  was 0.90). KV pool = that minus weights/activations.
- KV dtype `auto` = model dtype; FP8 is opt-in (halves KV) — **except DeepSeek
  V3.2 auto-resolves to FP8**.
- Chunked prefill (default) caps activations at `max_num_batched_tokens`
  (8192 server / 16384 on big GPUs). CUDA graphs add ~0.5–3 GB (in
  `enforce_eager` to recover).

## SGLang
- **RadixAttention** token-level paging (page_size 1 → ~zero internal waste) +
  radix-tree prefix cache (free, in-pool; 50–99% hit rate on shared prefixes).
- `mem_fraction_static` covers **weights+KV only** — activations/CUDA graphs/
  context land *on top* (reserve ~5–8 GB). ≥60 GB GPUs floor 10 GB reserved.
- MLA models get a native compressed pool (576 elems/layer/token for DeepSeek-V3,
  ~98% smaller than 128-head MHA). HiCache tiers cache to host RAM/storage.

## llama.cpp
- **Contiguous static** KV allocated at load for the configured context — exact,
  deterministic, no paging. Default `--ctx-size 0` loads the model's *full*
  training context, so default KV can be huge.
- KV dtype f16 default; **q8_0 ≈ −47%**, **q4_0 ≈ −72%** (V-cache quant needs
  flash attention, which is default-auto since Aug 2025).
- iSWA: sliding-window models (Gemma) store only window-sized KV on local layers
  (~−80% at long context). Compute buffer grows with context (flat with FA on).
- **MoE CPU expert offload** (`--cpu-moe`/`--n-cpu-moe`) keeps experts in host
  RAM — the biggest MoE VRAM saver (runs 120B MoE on 24 GB GPUs). MLA models use
  the compressed (kv_lora_rank + qk_rope) KV. *(Not modeled by the calculator's
  default path — out of v1 scope.)*

## Ollama
- Wraps llama.cpp. **Divergences:** total KV = `OLLAMA_NUM_PARALLEL × num_ctx`
  (static per slot; default parallel = 1); context default is VRAM-tiered since
  v0.15.5 (4K < 24 GB, 32K 24–48 GB, 256K ≥ 48 GB); `n_batch` fixed 512.
- KV quant via `OLLAMA_KV_CACHE_TYPE` (q8_0 ≈ ½, q4_0 ≈ ¼), gated on FA
  (default-on for qwen3/gemma3/gpt-oss families on Ampere+/RDNA3+ in v0.30).
- Auto CPU spill: if it doesn't fit, layers offload to CPU (slower, not fatal).

## TensorRT-LLM
- Paged KV (block 32). **Preallocates** `kv_cache_free_gpu_mem_fraction` (0.90)
  of memory *remaining after weights+activations* — so it consumes nearly the
  whole GPU. NVIDIA-only.
- KV dtype `auto` from checkpoint (FP8 if the quant config requests it). FP8 KV
  −50%, NVFP4 KV (Blackwell) −75%. PyTorch backend default since v1.0.
- Chunked prefill decouples activations from prompt length.

## MLX (Apple)
- **No preallocation, no paging** — lazy step-256 KV growth, so footprint ≈
  actual use. Lowest fixed overhead (no CUDA context). Budget is bounded by the
  macOS `recommendedMaxWorkingSetSize` cap (see hardware-constraints.md).
- KV quant `--kv-bits` (8 → −47%, 4 → −72%), off by default until 5000 tokens.
  `--max-kv-size` is a hard rotating-cache cap. Affine 4-bit weights = 4.5 bpw.

> **Tenstorrent uses a dedicated per-chip estimation** — block-float weights,
> BFP8 KV, cross-chip sharding (MLA replicates), and per-chip usable DRAM. See
> **`notes/tenstorrent-memory.md`** for the full source-derived formula; the
> summary below is the qualitative picture.

## Tenstorrent (tt-vllm / tt-metal / tt-inference-server)
- Paged KV (block 64) in device DRAM, but **no profiling run** — KV is sized to
  a fixed per-model/device token budget (`num_gpu_blocks_override`), default
  131072 with overrides (e.g. Llama-8B on N150 → 32768).
- **KV and weights default to block-float `bfloat8_b` ≈ 1.0625 B/elem** (~1.88×
  smaller than BF16); FF MLPs often `bfp4_b` ≈ 0.5625 B/param. BF16 KV optional
  for precision-sensitive small models.
- No chunked prefill; decode max batch 32/replica. Per-device overhead = trace
  region (30–384 MB) + L1 scratch. tt-inference-server/TT-Studio are
  orchestration layers over this — same memory semantics.

## Sources
- vLLM: docs.vllm.ai conserving_memory/optimization; `vllm/config/cache.py`; blog.vllm.ai 2023-06-20
- SGLang: docs.sglang.io hyperparameter_tuning/attention_backend; arXiv:2312.07104; HiCache blog
- llama.cpp: discussions #10068/#20969/#13194; `tools/quantize/README.md`
- Ollama: docs.ollama.com faq/context-length; ollama.com/blog/new-model-scheduling
- TensorRT-LLM: nvidia.github.io/TensorRT-LLM memory.html/kvcache.html/quantization.html
- MLX: github.com/ml-explore/mlx-lm `models/cache.py`, `generate.py`, README
- Tenstorrent: tt-metal `tech_reports/LLMs/vLLM_integration.md`, `generator_vllm.py`;
  tt-inference-server `release_model_spec.json`
