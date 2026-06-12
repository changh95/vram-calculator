# Supported Hardware

Mirrors `src/lib/data/hardware.ts` — keep in sync. `Memory` is the
GPU-exposed/allocatable capacity (GiB). `Usable` shows the unified-memory cap
where it applies (see `notes/hardware-constraints.md`). Verified against vendor
spec pages June 2026. Tenstorrent leads the list (matching the app's Hardware
dropdown).

## Tenstorrent (dedicated GDDR6, multi-chip mesh — no unified cap)

A "Buy now" link to the device's tenstorrent.com product page appears in the app
for these SKUs (Tenstorrent-only).

| SKU | Memory (aggregate) | Chips × per-chip | Bandwidth |
|---|---|---|---|
| Wormhole n150 | 12 GB | 1 × 12 | 288 GB/s |
| Wormhole n300 | 24 GB | 2 × 12 | 576 GB/s |
| Blackhole p150a | 32 GB | 1 × 32 | 512 GB/s |
| T3K / TT-LoudBox (WH) | 96 GB | 8 × 12 | 2304 GB/s |
| TT-QuietBox (WH) | 96 GB | 8 × 12 | 2304 GB/s |
| TT-QuietBox "QB2" (BH) | 128 GB | 4 × 32 | 2048 GB/s |
| TT-LoudBox (BH) | 256 GB | 8 × 32 | 4096 GB/s |
| Galaxy (Wormhole) | 384 GB | 32 × 12 | 9216 GB/s |
| Galaxy (Blackhole, 1 TB) | 1024 GB | 32 × 32 | 16384 GB/s |

Memory is per-chip pooled over an Ethernet mesh — a model shards across the
per-chip pools, not a flat aggregate (the calculator surfaces this assumption).

## NVIDIA GeForce (discrete)

| GPU | Memory | Bandwidth |
|---|---|---|
| RTX 2080 Ti | 11 GB GDDR6 | 616 GB/s |
| RTX 3060 (12 GB) | 12 GB GDDR6 | 360 GB/s |
| RTX 3090 | 24 GB GDDR6X | 936 GB/s |
| RTX 4080 | 16 GB GDDR6X | 717 GB/s |
| RTX 4090 | 24 GB GDDR6X | 1008 GB/s |
| RTX 5060 Ti (16 GB) | 16 GB GDDR7 | 448 GB/s |
| RTX 5070 | 12 GB GDDR7 | 672 GB/s |
| RTX 5070 Ti | 16 GB GDDR7 | 896 GB/s |
| RTX 5080 | 16 GB GDDR7 | 960 GB/s |
| RTX 5090 | 32 GB GDDR7 | 1792 GB/s |

## NVIDIA workstation / datacenter (discrete)

| GPU | Memory | Bandwidth |
|---|---|---|
| RTX A6000 | 48 GB GDDR6 | 768 GB/s |
| RTX 6000 Ada | 48 GB GDDR6 | 960 GB/s |
| RTX PRO 6000 Blackwell | 96 GB GDDR7 | 1792 GB/s |
| L4 | 24 GB GDDR6 | 300 GB/s |
| L40S | 48 GB GDDR6 | 864 GB/s |
| A100 40GB / 80GB | 40 / 80 GB HBM2(e) | 1555 / 2039 GB/s |
| H100 SXM | 80 GB HBM3 | 3350 GB/s |
| H100 NVL (per card) | 94 GB HBM3 | 3900 GB/s |
| H200 | 141 GB HBM3e | 4800 GB/s |
| B200 (per GPU) | 180 GB HBM3e | 7700 GB/s |
| B300 (per GPU) | 279 GB HBM3e | 8000 GB/s |
| DGX Spark (GB10) | 128 GB unified · ~115 usable (0.9) | 273 GB/s |

B200/B300 store the *exposed* capacity (marketed 192/288). HBM = sideband ECC,
no carve-out; GDDR ECC reserves ~6.25% only when enabled.

## Apple Silicon (unified memory)

Usable = 2/3 of RAM for ≤32 GB Macs, 3/4 for ≥36 GB (boundary 32→36 GB).

| Chip | Memory | Usable | Bandwidth |
|---|---|---|---|
| M1 | 16 GB | 0.667 | 68 GB/s |
| M1 Pro | 32 GB | 0.667 | 200 GB/s |
| M1 Max | 64 GB | 0.75 | 400 GB/s |
| M1 Ultra | 128 GB | 0.75 | 800 GB/s |
| M2 / M2 Pro | 24 / 32 GB | 0.667 | 100 / 200 GB/s |
| M2 Max | 96 GB | 0.75 | 400 GB/s |
| M2 Ultra | 192 GB | 0.75 | 800 GB/s |
| M3 / M3 Pro | 24 / 36 GB | 0.667 / 0.75 | 100 / 150 GB/s |
| M3 Max | 128 GB | 0.75 | 400 GB/s |
| M3 Ultra | 512 GB | 0.75 | 819 GB/s |
| M4 / M4 Pro / M4 Max | 32 / 64 / 128 GB | 0.667 / 0.75 / 0.75 | 120 / 273 / 546 GB/s |
| M5 / M5 Pro / M5 Max | 32 / 64 / 128 GB | 0.667 / 0.75 / 0.75 | 153 / 307 / 614 GB/s |

No M4 Ultra was released; M5 Ultra is unannounced (both excluded). Raise the cap
with `sudo sysctl iogpu.wired_limit_mb`.

## AMD

| Device | Memory | Usable | Bandwidth |
|---|---|---|---|
| Instinct MI300X | 192 GB HBM3 | — | 5300 GB/s |
| Instinct MI325X | 256 GB HBM3e | — | 6000 GB/s |
| Instinct MI355X | 288 GB HBM3e | — | 8000 GB/s |
| Radeon RX 7900 XTX | 24 GB GDDR6 | — | 960 GB/s |
| Radeon RX 9070 XT | 16 GB GDDR6 | — | 640 GB/s |
| Radeon AI PRO R9700 | 32 GB GDDR6 | — | 640 GB/s |
| Radeon PRO W7900 | 48 GB GDDR6 | — | 864 GB/s |
| Ryzen AI Max+ 395 (Strix Halo) | 128 GB unified · 96 usable (0.75) | 0.75 | 256 GB/s |

## Multi-device

The "× N devices" stepper aggregates capacity with a simple tensor-parallel
model (weights/KV shard, ~3% replicated weights + per-device overhead). Verdict
is judged per device.
