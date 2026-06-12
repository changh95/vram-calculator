# Tenstorrent Memory Model

The dedicated per-chip estimation used for Tenstorrent SKUs, reverse-engineered
from tt-metal / tt-vllm / tt-inference-server source (June 2026, adversarially
verified). Implemented in `estimateVram.ts → estimateTenstorrent()`, triggered
when a `HardwareSpec` has `usableGbPerChip`. Provenance:
`notes/raw/tenstorrent-investigation.json`.

## Why TT needs its own model

The generic "fits if total ≤ aggregate GB" check is wrong for TT in four ways,
all fixed here:
1. **Memory is per-chip, pooled over a mesh** — a model must fit *after sharding*
   across the chips, so the verdict is judged **per chip**, not against the flat
   aggregate.
2. **Weights run mixed block-float**, not the user's quant — much heavier than Q4.
3. **MLA/compressed KV replicates** on every chip (decodes as MQA).
4. **Usable DRAM per chip is less than marketed** (allocator reservations).

## The formula (per chip is the binding constraint)

```
shards      = numChips × deviceCount
bpp         = moe ? 0.73 : 0.85                       # block-float B/param
weights     = paramsTotalB × 1e9 × bpp
kv          = kvCacheBytes(attention, ctx, seqs, 1.0625)   # BFP8, per-architecture
activations = activationBytes(hidden, ctx, seqs, {chunkTokens:4096, mult:6})

r           = shards > 1 ? 0.005 : 0                  # only RMSNorm replicates
kvDivisor   = min(shards, kvShardCap(attention))      # MLA/compressed → 1 (replicated)

perChip = weights × ((1−r)/shards + r)   # weights shard, tiny replicated part
        + kv / kvDivisor                 # KV shards by head (or replicates for MLA)
        + activations / shards
        + 0.3 GB                         # trace region + misc, per chip

usablePerChip = hardware.usableGbPerChip          # 12 (WH) / 32 (BH) — full per-chip capacity
verdict       = perChip/usablePerChip > 1 ? wont-fit : > 0.9 ? tight : fits
```

`kvShardCap`: `gqa` / `gqa-swa` / `hybrid` → `numKvHeads`; `mla` / `compressed` → **1**.

## Verified constants & sources

**Usable DRAM per chip** — we use the **full marketed per-chip capacity: 12 GiB
(Wormhole) / 32 GiB (Blackhole)** (project decision). The allocator-derived
figure is marginally lower, `num_views × (dram_view_size − dram_unreserved_base)`
= **11.988 GiB** WH (12 × (1,073,741,824 − 1,048,640)) and **31.867 GiB** BH
(8 × (4,278,190,080 − 1,048,704)) — a ~0.1–0.4% reservation we treat as
negligible. Firmware/kernels live in on-chip SRAM (108 MB WH / 180 MB BH),
**not** DRAM. A per-model `trace_region_size` (50–402 MB in
`release_model_spec.json`) is real DRAM consumption — approximated here by the
0.3 GB/chip overhead on the required side.

**Block-float bytes/param** (16-element shared-exponent tile): BFP4 = 9/16 = **0.5625**, BFP8 = 17/16 = **1.0625**, BF16 = 2.0. Default serving profile (`DecodersPrecision.performance`, what `generator_vllm.py` passes): WQKV/WO/FF2/KV = BFP8, **FF1_FF3 = BFP4**. Source: `tt-metal/models/tt_transformers/tt/model_config.py`. Effective bpp collapses to ~0.73 for MoE (experts/FFN dominate at BFP4) and ~0.85 for dense — used as the two-value estimate here rather than a full per-tensor split.

**KV cache**: `ttnn.bfloat8_b` default (1.0625 B/elem), paged block_size 64, sized to a fixed `max_tokens_all_users` budget (FALLBACK 131072, per-model/device overrides). Source: `generator_vllm.py allocate_vllm_kv_cache`, `model_capabilities.py`. KV shards by KV head across the TP mesh; MLA (`num_kv_heads`→1 at decode) replicates.

**Replication**: only RMSNorm weights use `ReplicateTensorToMesh`; attention/MLP/embeddings/lm_head all shard → `REPLICATED_WEIGHT_FRACTION_TT = 0.005` (vs the generic 0.03).

## Known limitations (disclosed in-app as assumptions)

- The weight bpp is a **two-value estimate** (moe/dense), not a per-model
  attn/FFN split — accurate to ~±10%. A precise version needs `intermediateSize`,
  `numAttnHeads`, `ffnGated`, `numExperts` per model (see the synthesis in the
  provenance file).
- KV is sized to the **requested** context×sequences (so the slider stays
  meaningful); real deployments pin a fixed pooled token budget and per-model
  context caps (e.g. Llama-8B on N150 ≈ 32K) — surfaced as an assumption.
- `block_size = 64` is operative in every release spec but not pinned by a
  plugin constant. `trace_region_size` varies per model/device; folded into a
  flat 0.3 GB/chip here.
- Whether a given model is actually supported on the TT stack (release
  `release_model_spec.json`) is not enforced — the math computes regardless.

Net: TT verdicts went from "is the total under the aggregate" (optimistic) to a
genuine per-chip check with block-float weights and MLA-aware KV — the
multi-chip realism that was missing.
