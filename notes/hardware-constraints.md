# Hardware Constraints — usable VRAM realities

Why `memoryGb` in `src/lib/data/hardware.ts` is sometimes less than the marketed
capacity, and how `usableFraction` is derived. Verified against vendor spec
pages June 2026.

## Discrete GPUs (NVIDIA GeForce/datacenter, AMD Radeon/Instinct)

Dedicated VRAM, no unified-memory cap (`unified: false`, no `usableFraction`).
Real-world deductions the UI treats as small and does not subtract by default:
- **CUDA/HIP context:** ~0.3–0.8 GB per process (folded into framework overhead).
- **Display reservation:** a GPU driving a desktop loses ~0.5–1.5 GB (Windows
  WDDM/DXGI budget ≈ 90% of VRAM). Headless Linux gets nearly the full capacity.
- **ECC:** HBM (A100/H100/H200/B-series, MI300+) uses **sideband ECC — no
  capacity loss**. GDDR workstation/server cards (A6000, L40S, RTX PRO 6000)
  reserve ~6.25% **only when ECC is enabled** (off by default on workstation
  cards). GeForce GDDR7 has on-die ECC with no carve-out.

### Blackwell datacenter provisioning (load-bearing)
Marketed raw capacity ≠ GPU-exposed capacity:
- **B200:** marketed 192 GB, **ships exposing 180 GB** (DGX B200 = 1,440 GB / 8).
- **B300:** marketed 288 GB, **exposes 279 GB** per GPU (CoreWeave GB300 NVL72
  production docs; NVIDIA's 20 TB / 72 GPUs ≈ 278). Use **279**, not the
  "~270" some third-party blogs cite (~3.1% haircut, not 6.25%).

The table stores the **exposed** figures (180, 279).

## Unified memory — `usableFraction` devices

The OS caps how much shared RAM the GPU may use; the table stores that fraction.

### Apple Silicon (M1–M5)
macOS exposes the cap as Metal `recommendedMaxWorkingSetSize`, which behaves as
a **hard ceiling** for GPU allocations (llama.cpp, Ollama, LM Studio, MLX all
treat it as effective VRAM). Empirically pinned this session:

| System RAM | GPU-usable default | fraction |
|---|---|---|
| ≤ 32 GB | ~2/3 of RAM | **0.667** |
| ≥ 36 GB | 3/4 of RAM | **0.75** |

The boundary is between **32 GB and 36 GB**, *not* 64 GB (a common myth): a
36 GB M3 Max logs `recommendedMaxWorkingSetSize = 28991 MB = exactly 27 GiB =
75%`; a 48 GB Mac gets 36 GB. No factory SKU exists in the 33–35 GB gap, so the
per-SKU rule is exact. `src/lib/data/hardware.ts` applies
`appleFrac(gb) = gb >= 36 ? 0.75 : 2/3` per device.

**Override:** `sudo sysctl iogpu.wired_limit_mb=N` (macOS 14+, no reboot/SIP
change; resets on reboot). 512 GB M3 Ultra owners routinely raise to ~448–500 GB.
Leave ~8–16 GB for the OS. The calculator models the *default* cap; advanced
users can mentally treat usable ≈ RAM − 8 GB.

Tier reality as of June 2026: no M4 Ultra was ever released (2025 Mac Studio
pairs M4 Max with M3 Ultra); M5 Ultra is unannounced (excluded from the table).

### NVIDIA DGX Spark (GB10)
True UMA, 128 GB LPDDR5x @ 273 GB/s, no fixed CPU/GPU split. Firmware/OS reserve
~7–13 GB (OS sees ~121 GiB; deviceQuery ~119.7 GiB), so ~110–120 GB is usable —
**`usableFraction: 0.9`** (~115 GB). Standard CUDA tools misreport availability
(`cudaMemGetInfo` ignores reclaimable page cache; `nvidia-smi` memory fields
unsupported). Bandwidth, not capacity, is the bottleneck.

### AMD Strix Halo (Ryzen AI Max+ 395)
128 GB LPDDR5X @ 256 GB/s. Windows "Variable Graphics Memory" caps the GPU
carve-out at **96 GB of 128 (0.75)**; Linux GTT (dynamically-mapped system RAM)
defaults to ~50% and is tunable via `amd-ttm` / `amdgpu.gttsize` to ~110–120 GB.
The table uses **0.75** (the Windows default ceiling). Announced Gorgon Halo
(Ryzen AI Max 400) raises this to 192 GB / 160 GB usable (0.833).

## Multi-chip Tenstorrent SKUs

All Tenstorrent accelerators use **dedicated GDDR6**, *not* unified with the
host. There is **no OS-style usable cap** — the tt-metal allocator reserves only
a small per-bank barrier; firmware/kernels live in on-chip SRAM (108 MB
Wormhole / 180 MB Blackhole per chip), not GDDR6. So `usableFraction` is unset.

The real constraint is **topology**: memory is per-chip (12 GB Wormhole, 32 GB
Blackhole) pooled over an on-chip Ethernet mesh, so a model shards across the
per-chip pools rather than a flat aggregate. The table records `numChips`,
`memoryPerChipGb`, and **`usableGbPerChip`** — the usable DRAM per chip, set to
the full marketed per-chip capacity (Wormhole **12 GiB**, Blackhole **32 GiB**;
the allocator-derived figure is ~0.1–0.4% lower — treated as negligible). The
calculator uses a **dedicated per-chip
estimation** for TT (block-float weights, BFP8 KV, head-sharded KV with MLA
replication, verdict judged per chip) — see `notes/tenstorrent-memory.md`. Per-chip: Wormhole 12 GB @ 288 GB/s; Blackhole 32 GB @ 512 GB/s.
SKU aggregates: n300 = 2×12; T3K / Wormhole QuietBox = 8×12 = 96; Blackhole
QuietBox "QB2" = 4×32 = 128; Blackhole LoudBox = 8×32 = 256; Galaxy Wormhole =
32×12 = 384; Galaxy Blackhole = 32×32 = 1024 (1 TB).

## Sources
- nvidia.com product/datasheet pages; docs.nvidia.com/dgx/dgx-spark
- Apple newsroom + support.apple.com spec pages; Apple Metal docs (forum 732035);
  llama.cpp issue #7060 (36 GB = 27 GiB)
- amd.com Instinct/Radeon spec pages; rocm.docs.amd.com Strix Halo guide
- docs.tenstorrent.com spec pages; tt-metal allocator tech report
- CoreWeave GB300 NVL72 instance docs; Lenovo HGX B200 guide (lp2226)
