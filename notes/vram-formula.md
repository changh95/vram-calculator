# VRAM Formula

How `src/lib/estimateVram.ts` computes required memory. Researched and
adversarially fact-checked against primary sources (HuggingFace configs,
EleutherAI/kipply transformer-math, vLLM/llama.cpp source) on 2026-06-12.

> **Units.** All `*Gb` fields are **GiB** (2³⁰ bytes), labeled "GB" in the UI.
> `GIB = 2 ** 30`.

## Total

```
total = weights + kv_cache + activations + runtime_overhead
```

Each term is a named function in `estimateVram.ts` so it can be corrected
without touching the UI. Weights and KV cache are computed from each model's
real architecture (genuinely accurate). Activations and overhead are
heuristic **estimates**, disclosed via `MemoryBreakdown.assumptions`.

## 1. Weights

```
weight_bytes = params_total × (effective_bits_per_weight / 8)
```

- `params_total` is the full checkpoint parameter count — **MoE uses total, not
  active** params (the whole expert set is resident).
- `effective_bits_per_weight` is whole-checkpoint effective bpw: it folds in
  quantization metadata (group scales/zero-points) **and** the higher-precision
  tensors (embeddings, output head, norms) that quant formats keep at FP16. See
  `src/lib/data/quants.ts`.

Key bpw values (from `llama.cpp tools/quantize/README.md` measured table and
`ggml-common.h`): FP16 = 16, FP8 ≈ 8.05, Q8_0 = 8.5, Q6_K = 6.56,
Q5_K_M = 5.70, **Q4_K_M = 4.89** (≈ 0.61 B/param), AWQ/GPTQ INT4 ≈ 4.5 (whole
model, FP16 embeddings kept), NVFP4/MXFP4 = 4.25, Q3_K_M = 4.00, IQ2_M = 2.93.

GPTQ/AWQ block formula: `k + (16 + k)/group_size` (e.g. INT4 g128 = 4.156 bpw
on quantized tensors; whole-model is higher because embeddings stay FP16 — the
share shrinks with model size, so these are ±2–5% estimates).

**Validation:** 8.03B params × 16.0005/8 = 14.96 GiB; × 4.8944/8 = 4.58 GiB —
both match the published llama.cpp file sizes exactly.

## 2. KV cache — one branch per attention architecture

This is the failure mode of naïve calculators: two 32B models can differ by an
order of magnitude. Geometry comes from each model's config (`src/lib/data/models.ts`).

### GQA / MHA (`gqa`)
```
kv_bytes = 2 × num_layers × num_kv_heads × head_dim × kv_dtype_bytes × ctx × seqs
```
The `2` is K and V. MHA: `num_kv_heads = num_attention_heads`; MQA: 1.
- *Reference:* Llama-70B-shape (80 L, 8 kv, 128) = **320 KiB/token** → 40 GiB @ 128K.
- Matches vLLM v1 `AttentionSpec` and kipply's `2·2·n_layers·n_heads·d_head`.

### MLA — DeepSeek / Kimi / GLM-5 (`mla`)
Only a compressed latent + decoupled RoPE key is cached — **no factor 2**:
```
kv_bytes = num_layers × (kv_lora_rank + qk_rope_head_dim) × kv_dtype_bytes × ctx × seqs
```
- *Reference:* DeepSeek-V3 (61 L, 512 + 64) = **68.6 KiB/token** — ~20× smaller
  than equivalent 128-head MHA. Matches vLLM `MLAAttentionSpec` (coefficient 1).
- **DSA indexer** (V3.2, GLM-5): add `num_layers × indexer_elems × ctx × seqs`
  bytes, stored **FP8 (1 byte)** regardless of the KV dtype.

### Interleaved sliding window — Gemma 4 (`gqa-swa`)
Global layers see full context; local layers keep only the window. Gemma 4
global layers use **different KV geometry** (e.g. 1 KV head × 512 dim vs 8 × 256):
```
kv_bytes = [ local_layers × min(ctx, window) × (2 × num_kv_heads × head_dim)
           + global_layers × ctx × (2 × global_kv_heads × global_head_dim) ]
           × kv_dtype_bytes × seqs
```
- KV-sharing E-series (E2B/E4B) hold distinct KV on fewer than `num_layers`
  layers — `global_layers + local_layers` counts only KV-bearing layers.

### Hybrid SSM/linear — Qwen3.5, Nemotron-3, Kimi-Linear (`hybrid`)
KV only on the few attention layers; the rest carry a **constant** per-sequence
recurrent state (independent of context):
```
kv_bytes = 2 × attention_layers × num_kv_heads × head_dim × kv_dtype_bytes × ctx × seqs
         + ssm_state_bytes_per_seq × seqs
```
- Qwen3.5 caches on `num_layers / 4` layers; Nemotron-3-Ultra on 12 of 108.
- Kimi-Linear's 7 MLA layers are modeled as `num_kv_heads 1 × head_dim 288`
  (2 × 1 × 288 = 576 = MLA latent). `ssm_state_bytes_per_seq` is an **estimate**.

### Compressed CSA/HCA — DeepSeek-V4 (`compressed`)
Every layer keeps a short uncompressed window plus a 1/ratio-compressed cache;
ratio 0 = pure sliding window. CSA layers add an FP8 indexer cache.
```
per group: count × min(ctx, window + ctx/ratio) × kv_dim × kv_dtype_bytes
           (+ count × ctx/indexer_ratio × indexer_dim × 1 byte  for CSA)
```
- V4 `kv_dim = 512` (RoPE dims live *inside* it — not +64, per DeepSeek's
  reference `inference/model.py`).

**KV dtype bytes** come from the framework, not the weight quant: FP16/BF16 = 2,
FP8 = 1, llama.cpp q8_0 KV = 1.0625, q4_0 KV = 0.5625, TT bfp8_b = 1.0625.

## 3. Activations (estimate)

```
activation_bytes = multiplier × hidden_size × batched_tokens × 2
batched_tokens   = max( min(ctx, chunk_tokens), seqs )
```
`chunk_tokens` is the framework's chunked-prefill cap (bounds the prefill peak
regardless of prompt length). `multiplier` is calibrated so an 8B/hidden-4096
model lands near vLLM's reported ~1.3 GiB activation peak at 8192 tokens. Purely
an estimate; always disclosed.

## 4. Runtime overhead (estimate)

`framework.baselineOverheadGb`, paid **per device**: CUDA/Metal context, CUDA
graphs, framework buffers. ~2 GiB vLLM, ~0.6 GiB llama.cpp, ~0.3 GiB MLX/TT.
(EleutherAI's "total ≈ 1.2 × weights" heuristic is the coarse cross-check.)

## Multi-device

Tensor-parallel model: a `REPLICATED_WEIGHT_FRACTION` (≈ 3%) copy of weights
lives on every device, the rest shards evenly; KV shards evenly; activations and
overhead are paid per device.
```
per_device = weights × ((1−r)/N + r) + kv/N + activations + overhead
```
Verdict utilization is judged **per device**. Multi-chip Tenstorrent SKUs (T3K,
Galaxy) are single options whose `memoryGb` is the aggregate.

## Unified memory & verdict

Usable capacity = `memoryGb × usableFraction` for unified devices (Apple, DGX
Spark, Strix Halo), else the full capacity. Verdict: **won't fit** > 100% of
usable, **tight** > 90%, else **fits**.

## Sources
- EleutherAI transformer-math; kipply transformer-inference-arithmetic
- llama.cpp `ggml-common.h`, `tools/quantize/README.md`, `src/llama-quant.cpp`
- vLLM `vllm/v1/kv_cache_interface.py` (Attention/MLA/SlidingWindow specs)
- Per-model HuggingFace `config.json` (see `model_list.md`)
- Raschka KV-cache calculations (cross-check: Qwen3 144 KiB, DeepSeek-V3 68.6 KiB)
