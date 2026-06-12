# Supported Models

Mirrors `src/lib/data/models.ts` — keep in sync. KV geometry is verified from
each model's HuggingFace `config.json` (June 2026); `bytes/tok` is bf16
KV-cache cost per token (2 bytes/elem). MoE shows total · active params.

**Attention kinds:** GQA (standard) · MLA (latent, no ×2; +DSA indexer) ·
SWA (Gemma 4 interleaved sliding window) · hybrid (KV on a few attention layers
+ constant SSM/linear state) · compressed (DeepSeek-V4 CSA/HCA).

## Qwen3 (Alibaba) — original Apr 2025, standard full-context GQA (head_dim 128)

| Model | Params | Attn | Layers | KV/tok | Ctx |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | GQA (8kv·128) | 28 | 112 KiB | 41K |
| Qwen3-4B | 4.0B | GQA (8kv·128) | 36 | 144 KiB | 41K |
| Qwen3-8B | 8.2B | GQA (8kv·128) | 36 | 144 KiB | 41K |
| Qwen3-14B | 14.8B | GQA (8kv·128) | 40 | 160 KiB | 41K |
| Qwen3-32B | 32.8B | GQA (8kv·128) | 64 | 256 KiB | 41K |
| Qwen3-30B-A3B-2507 | 30.5B·A3.3B | GQA (4kv·128) | 48 | 96 KiB | 256K |
| Qwen3-235B-A22B-2507 | 235B·A22B | GQA (4kv·128) | 94 | 188 KiB | 256K |

## Qwen3-Next — hybrid bridge (full GQA every 4th layer, head_dim 256)

| Model | Params | Attn | Layers (KV) | KV/tok | Ctx |
|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 80B·A3B | hybrid | 48 (12) | 24 KiB | 256K |
| Qwen3-Coder-Next | 80B·A3B | hybrid | 48 (12) | 24 KiB | 256K |

## Qwen3.5 — hybrid linear + GQA (KV on every 4th layer)

| Model | Params | Attn | Layers (KV) | KV/tok | Ctx |
|---|---|---|---|---|---|
| Qwen3.5-4B | 4.66B | hybrid | 32 (8) | 16 KiB | 256K |
| Qwen3.5-9B | 9.65B | hybrid | 32 (8) | 16 KiB | 256K |
| Qwen3.5-27B | 27.78B | hybrid | 64 (16) | 32 KiB | 256K |
| Qwen3.5-35B-A3B | 35.95B·A3B | hybrid | 40 (10) | 10 KiB | 256K |
| Qwen3.5-122B-A10B | 125.09B·A10B | hybrid | 48 (12) | 12 KiB | 256K |
| Qwen3.5-397B-A17B | 403.4B·A17B | hybrid | 60 (15) | 15 KiB | 256K |

## Qwen3.6 — Apr 2026 refresh (27B & 35B-A3B; identical architecture to 3.5)

| Model | Params | Attn | Layers (KV) | KV/tok | Ctx |
|---|---|---|---|---|---|
| Qwen3.6-27B | 27.78B | hybrid | 64 (16) | 32 KiB | 256K |
| Qwen3.6-35B-A3B | 35.95B·A3B | hybrid | 40 (10) | 10 KiB | 256K |

## Gemma 4 (Google) — interleaved sliding window

| Model | Params | Layers | global/local KV | window | Ctx |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8.0B | 42 | 4 × (2kv·512) / 20 × (2kv·256) | 512 | 128K |
| Gemma 4 12B | 11.96B | 48 | 8 × (1kv·512) / 40 × (8kv·256) | 1024 | 256K |
| Gemma 4 26B-A4B | 25.81B·A3.8B | 30 | 5 × (2kv·512) / 25 × (8kv·256) | 1024 | 256K |
| Gemma 4 31B | 31.27B | 60 | 10 × (4kv·512) / 50 × (16kv·256) | 1024 | 256K |

E-series share KV across the last layers, so only a subset hold distinct KV.

## DiffusionGemma (Google) — block-diffusion model on the Gemma 4 26B-A4B backbone

| Model | Params | Layers | global/local KV | window | Ctx |
|---|---|---|---|---|---|
| DiffusionGemma 26B-A4B | 25.81B·A3.8B | 30 | 5 × (2kv·512) / 25 × (8kv·256) | 1024 | 256K |

Generates by iteratively denoising 256-token canvases (bidirectional attention)
instead of token-by-token. KV reuses the Gemma 4 geometry; activation behavior
during denoising differs (flagged as an estimate).

## GLM (Z.ai)

| Model | Params | Attn | Layers | KV/tok | Ctx |
|---|---|---|---|---|---|
| GLM-4.5-Air | 110.5B·A12B | GQA (8kv·128) | 46 | 184 KiB | 128K |
| GLM-4.6 | 356.8B·A32B | GQA (8kv·128) | 92 | 368 KiB | 198K |
| GLM-4.7-Flash | 31B·A3B | MLA | 47 | 54 KiB | 198K |
| GLM-5 | 754B·A40B | MLA + DSA | 78 | 88 KiB | 198K |
| GLM-5.1 | 754B·A40B | MLA + DSA | 78 | 88 KiB | 198K |

## DeepSeek

| Model | Params | Attn | Layers | KV/tok | Ctx |
|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | 1.78B | GQA (2kv·128) | 28 | 28 KiB | 128K |
| R1-Distill-Qwen-14B | 14.8B | GQA (8kv·128) | 48 | 192 KiB | 128K |
| R1-Distill-Qwen-32B | 32.8B | GQA (8kv·128) | 64 | 256 KiB | 128K |
| R1-Distill-Llama-70B | 70.6B | GQA (8kv·128) | 80 | 320 KiB | 128K |
| DeepSeek-V3 | 671B·A37B | MLA | 61 | 68.6 KiB | 160K |
| DeepSeek-R1 | 671B·A37B | MLA | 61 | 68.6 KiB | 160K |
| DeepSeek-V3.2 | 671B·A37B | MLA + DSA | 61 | 68.6 KiB + idx | 160K |
| DeepSeek-V4-Flash | 284B·A13B | compressed | 43 | compressed | 1M |

DeepSeek-V3/R1/Kimi-K2 share identical MLA geometry (kv_lora 512 + rope 64).

## Kimi (Moonshot)

| Model | Params | Attn | Layers | KV/tok | Ctx |
|---|---|---|---|---|---|
| Moonlight-16B-A3B | 15.96B·A3B | MLA | 27 | 30 KiB | 8K |
| Kimi-VL-A3B | 16.4B·A3B | MLA | 27 | 30 KiB | 128K |
| Kimi-Linear-48B-A3B | 49.12B·A3B | hybrid MLA (7 of 27) | 27 (7) | 8 KiB | 1M |
| Kimi-K2-Instruct | 1.03T·A32B | MLA | 61 | 68.6 KiB | 128K |
| Kimi-K2-Thinking | 1.03T·A32B | MLA | 61 | 68.6 KiB | 256K |
| Kimi-K2.6 | 1.03T·A32B | MLA | 61 | 68.6 KiB | 256K |

## Nemotron (NVIDIA) — hybrid Mamba2-Transformer (KV on few attention layers)

| Model | Params | Attn | Layers (KV) | KV/tok | Ctx |
|---|---|---|---|---|---|
| Nemotron-Nano-9B-v2 | 8.89B | hybrid | 56 (4) | 8 KiB | 128K |
| Nemotron-Nano-12B-v2 | 12.31B | hybrid | 62 (6) | 12 KiB | 128K |
| Nemotron-3-Nano-30B-A3B | 31.58B·A3.5B | hybrid | 52 (6) | 3 KiB | 1M |
| Nemotron-3-Super-120B-A12B | 123.61B·A12B | hybrid | 88 (8) | 4 KiB | 1M |
| Nemotron-3-Ultra-550B-A55B | 560.52B·A55B | hybrid | 108 (12) | 6 KiB | 1M |
| Llama-3.3-Nemotron-Super-49B | 49.87B | GQA (49 of 80 blocks) | 49 | 196 KiB | 128K |

Hybrids also carry a constant per-sequence SSM state (~15–402 MB, estimate).

## Llama 3 (Meta) — standard GQA, head_dim 128 (64 on 1B)

| Model | Params | Attn | Layers | KV/tok | Ctx |
|---|---|---|---|---|---|
| Llama-3.2-1B | 1.24B | GQA (8kv·64) | 16 | 32 KiB | 128K |
| Llama-3.2-3B | 3.21B | GQA (8kv·128) | 28 | 112 KiB | 128K |
| Llama-3.1-8B | 8.03B | GQA (8kv·128) | 32 | 128 KiB | 128K |
| Llama-3.1-70B | 70.6B | GQA (8kv·128) | 80 | 320 KiB | 128K |
| Llama-3.3-70B | 70.6B | GQA (8kv·128) | 80 | 320 KiB | 128K |
| Llama-3.1-405B | 405B | GQA (8kv·128) | 126 | 504 KiB | 128K |

## Notes
- Several families postdate the training cutoff and were web-verified: "Qwen3"
  current gen is **Qwen3.5/3.6**; **Gemma 4** shipped Mar/Jun 2026; **GLM-5/5.1**
  are real (MLA+DSA); **DeepSeek-V4** exists (compressed attention); **Kimi-K2.6**
  is the current flagship.
- MoE weights use **total** params; KV uses the architecture, not param count.
- See `notes/vram-formula.md` for the per-kind KV formulas.
