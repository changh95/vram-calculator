import type { HardwareSpec } from '../types'

/**
 * Hardware capacities from notes/hardware-constraints.md (vendor spec pages,
 * verified June 2026). `memoryGb` is the GPU-exposed/allocatable capacity in GiB
 * (datacenter Blackwell parts use the provisioned figure, e.g. B300 = 279, not
 * the marketed 288). Unified-memory devices set `usableFraction`:
 *   - Apple: 2/3 of RAM for ≤32 GB Macs, 3/4 for ≥36 GB (macOS
 *     recommendedMaxWorkingSetSize; raisable via iogpu.wired_limit_mb).
 *   - DGX Spark / Strix Halo: policy/OS-limited GPU share of system RAM.
 * Multi-chip SKUs set numChips + memoryPerChipGb; memoryGb is the aggregate.
 */

const NV = 'https://www.nvidia.com/en-us/'
const APPLE = 'https://en.wikipedia.org/wiki/Apple_silicon'
const AMD = 'https://www.amd.com/en/products/accelerators/instinct.html'
const TT = 'https://docs.tenstorrent.com/'

const appleFrac = (gb: number) => (gb >= 36 ? 0.75 : 2 / 3)
const appleNote = (gb: number) =>
  `Unified memory: macOS lets the GPU use ~${gb >= 36 ? '75%' : '2/3'} of ${gb} GB by default (recommendedMaxWorkingSetSize); raise with sudo sysctl iogpu.wired_limit_mb.`

export const HARDWARE: HardwareSpec[] = [
  // ───────────────────────── NVIDIA GeForce ─────────────────────────
  { id: 'rtx-2080-ti', vendor: 'NVIDIA', name: 'RTX 2080 Ti', memoryGb: 11, memoryType: 'GDDR6', bandwidthGbs: 616, unified: false, sources: [NV + 'geforce/'] },
  { id: 'rtx-3060', vendor: 'NVIDIA', name: 'RTX 3060 (12 GB)', memoryGb: 12, memoryType: 'GDDR6', bandwidthGbs: 360, unified: false, sources: [NV + 'geforce/'] },
  { id: 'rtx-3090', vendor: 'NVIDIA', name: 'RTX 3090', memoryGb: 24, memoryType: 'GDDR6X', bandwidthGbs: 936, unified: false, sources: [NV + 'geforce/'] },
  { id: 'rtx-4080', vendor: 'NVIDIA', name: 'RTX 4080', memoryGb: 16, memoryType: 'GDDR6X', bandwidthGbs: 717, unified: false, sources: [NV + 'geforce/'] },
  { id: 'rtx-4090', vendor: 'NVIDIA', name: 'RTX 4090', memoryGb: 24, memoryType: 'GDDR6X', bandwidthGbs: 1008, unified: false, sources: [NV + 'geforce/'] },
  { id: 'rtx-5060-ti', vendor: 'NVIDIA', name: 'RTX 5060 Ti (16 GB)', memoryGb: 16, memoryType: 'GDDR7', bandwidthGbs: 448, unified: false, sources: [NV + 'geforce/graphics-cards/50-series/'] },
  { id: 'rtx-5070', vendor: 'NVIDIA', name: 'RTX 5070', memoryGb: 12, memoryType: 'GDDR7', bandwidthGbs: 672, unified: false, sources: [NV + 'geforce/graphics-cards/50-series/'] },
  { id: 'rtx-5070-ti', vendor: 'NVIDIA', name: 'RTX 5070 Ti', memoryGb: 16, memoryType: 'GDDR7', bandwidthGbs: 896, unified: false, sources: [NV + 'geforce/graphics-cards/50-series/'] },
  { id: 'rtx-5080', vendor: 'NVIDIA', name: 'RTX 5080', memoryGb: 16, memoryType: 'GDDR7', bandwidthGbs: 960, unified: false, sources: [NV + 'geforce/graphics-cards/50-series/'] },
  { id: 'rtx-5090', vendor: 'NVIDIA', name: 'RTX 5090', memoryGb: 32, memoryType: 'GDDR7', bandwidthGbs: 1792, unified: false, sources: [NV + 'geforce/graphics-cards/50-series/rtx-5090/'] },

  // ─────────────────── NVIDIA workstation / datacenter ───────────────────
  { id: 'rtx-a6000', vendor: 'NVIDIA', name: 'RTX A6000', memoryGb: 48, memoryType: 'GDDR6 (ECC)', bandwidthGbs: 768, unified: false, sources: [NV + 'products/workstations/rtx-a6000/'] },
  { id: 'rtx-6000-ada', vendor: 'NVIDIA', name: 'RTX 6000 Ada', memoryGb: 48, memoryType: 'GDDR6 (ECC)', bandwidthGbs: 960, unified: false, sources: [NV + 'products/workstations/rtx-6000/'] },
  { id: 'rtx-pro-6000', vendor: 'NVIDIA', name: 'RTX PRO 6000 Blackwell', memoryGb: 96, memoryType: 'GDDR7 (ECC)', bandwidthGbs: 1792, unified: false, sources: [NV + 'products/workstations/professional-desktop-gpus/rtx-pro-6000/'] },
  { id: 'l4', vendor: 'NVIDIA', name: 'L4', memoryGb: 24, memoryType: 'GDDR6 (ECC)', bandwidthGbs: 300, unified: false, sources: [NV + 'data-center/l4/'] },
  { id: 'l40s', vendor: 'NVIDIA', name: 'L40S', memoryGb: 48, memoryType: 'GDDR6 (ECC)', bandwidthGbs: 864, unified: false, sources: [NV + 'data-center/l40s/'] },
  { id: 'a100-40', vendor: 'NVIDIA', name: 'A100 40GB', memoryGb: 40, memoryType: 'HBM2', bandwidthGbs: 1555, unified: false, sources: [NV + 'data-center/a100/'] },
  { id: 'a100-80', vendor: 'NVIDIA', name: 'A100 80GB', memoryGb: 80, memoryType: 'HBM2e', bandwidthGbs: 2039, unified: false, sources: [NV + 'data-center/a100/'] },
  { id: 'h100', vendor: 'NVIDIA', name: 'H100 SXM', memoryGb: 80, memoryType: 'HBM3', bandwidthGbs: 3350, unified: false, sources: [NV + 'data-center/h100/'] },
  { id: 'h100-nvl', vendor: 'NVIDIA', name: 'H100 NVL (per card)', memoryGb: 94, memoryType: 'HBM3', bandwidthGbs: 3900, unified: false, sources: [NV + 'data-center/h100/'] },
  { id: 'h200', vendor: 'NVIDIA', name: 'H200', memoryGb: 141, memoryType: 'HBM3e', bandwidthGbs: 4800, unified: false, sources: [NV + 'data-center/h200/'] },
  { id: 'b200', vendor: 'NVIDIA', name: 'B200 (per GPU)', memoryGb: 180, memoryType: 'HBM3e', bandwidthGbs: 7700, unified: false, sources: [NV + 'data-center/dgx-b200/'] },
  { id: 'b300', vendor: 'NVIDIA', name: 'B300 (per GPU)', memoryGb: 279, memoryType: 'HBM3e', bandwidthGbs: 8000, unified: false, sources: [NV + 'data-center/dgx-b300/'] },
  {
    id: 'dgx-spark',
    vendor: 'NVIDIA',
    name: 'DGX Spark (GB10)',
    memoryGb: 128,
    memoryType: 'LPDDR5x unified',
    bandwidthGbs: 273,
    unified: true,
    usableFraction: 0.9,
    usableNote: 'Unified LPDDR5x: firmware/OS reserve ~7–13 GB, so ~115 GB of 128 GB is practically usable for the GPU.',
    sources: ['https://docs.nvidia.com/dgx/dgx-spark/'],
  },

  // ───────────────────────── Apple Silicon ─────────────────────────
  { id: 'm1', vendor: 'Apple', name: 'M1', memoryGb: 16, memoryType: 'LPDDR4X unified', bandwidthGbs: 68, unified: true, usableFraction: appleFrac(16), usableNote: appleNote(16), sources: [APPLE] },
  { id: 'm1-pro', vendor: 'Apple', name: 'M1 Pro', memoryGb: 32, memoryType: 'LPDDR5 unified', bandwidthGbs: 200, unified: true, usableFraction: appleFrac(32), usableNote: appleNote(32), sources: [APPLE] },
  { id: 'm1-max', vendor: 'Apple', name: 'M1 Max', memoryGb: 64, memoryType: 'LPDDR5 unified', bandwidthGbs: 400, unified: true, usableFraction: appleFrac(64), usableNote: appleNote(64), sources: [APPLE] },
  { id: 'm1-ultra', vendor: 'Apple', name: 'M1 Ultra', memoryGb: 128, memoryType: 'LPDDR5 unified', bandwidthGbs: 800, unified: true, usableFraction: appleFrac(128), usableNote: appleNote(128), sources: [APPLE] },
  { id: 'm2', vendor: 'Apple', name: 'M2', memoryGb: 24, memoryType: 'LPDDR5 unified', bandwidthGbs: 100, unified: true, usableFraction: appleFrac(24), usableNote: appleNote(24), sources: [APPLE] },
  { id: 'm2-pro', vendor: 'Apple', name: 'M2 Pro', memoryGb: 32, memoryType: 'LPDDR5 unified', bandwidthGbs: 200, unified: true, usableFraction: appleFrac(32), usableNote: appleNote(32), sources: [APPLE] },
  { id: 'm2-max', vendor: 'Apple', name: 'M2 Max', memoryGb: 96, memoryType: 'LPDDR5 unified', bandwidthGbs: 400, unified: true, usableFraction: appleFrac(96), usableNote: appleNote(96), sources: [APPLE] },
  { id: 'm2-ultra', vendor: 'Apple', name: 'M2 Ultra', memoryGb: 192, memoryType: 'LPDDR5 unified', bandwidthGbs: 800, unified: true, usableFraction: appleFrac(192), usableNote: appleNote(192), sources: [APPLE] },
  { id: 'm3', vendor: 'Apple', name: 'M3', memoryGb: 24, memoryType: 'LPDDR5 unified', bandwidthGbs: 100, unified: true, usableFraction: appleFrac(24), usableNote: appleNote(24), sources: [APPLE] },
  { id: 'm3-pro', vendor: 'Apple', name: 'M3 Pro', memoryGb: 36, memoryType: 'LPDDR5 unified', bandwidthGbs: 150, unified: true, usableFraction: appleFrac(36), usableNote: appleNote(36), sources: [APPLE] },
  { id: 'm3-max', vendor: 'Apple', name: 'M3 Max', memoryGb: 128, memoryType: 'LPDDR5 unified', bandwidthGbs: 400, unified: true, usableFraction: appleFrac(128), usableNote: appleNote(128), sources: [APPLE] },
  { id: 'm3-ultra', vendor: 'Apple', name: 'M3 Ultra', memoryGb: 512, memoryType: 'LPDDR5 unified', bandwidthGbs: 819, unified: true, usableFraction: appleFrac(512), usableNote: appleNote(512), sources: [APPLE] },
  { id: 'm4', vendor: 'Apple', name: 'M4', memoryGb: 32, memoryType: 'LPDDR5X unified', bandwidthGbs: 120, unified: true, usableFraction: appleFrac(32), usableNote: appleNote(32), sources: [APPLE] },
  { id: 'm4-pro', vendor: 'Apple', name: 'M4 Pro', memoryGb: 64, memoryType: 'LPDDR5X unified', bandwidthGbs: 273, unified: true, usableFraction: appleFrac(64), usableNote: appleNote(64), sources: [APPLE] },
  { id: 'm4-max', vendor: 'Apple', name: 'M4 Max', memoryGb: 128, memoryType: 'LPDDR5X unified', bandwidthGbs: 546, unified: true, usableFraction: appleFrac(128), usableNote: appleNote(128), sources: [APPLE] },
  { id: 'm5', vendor: 'Apple', name: 'M5', memoryGb: 32, memoryType: 'LPDDR5X unified', bandwidthGbs: 153, unified: true, usableFraction: appleFrac(32), usableNote: appleNote(32), sources: [APPLE] },
  { id: 'm5-pro', vendor: 'Apple', name: 'M5 Pro', memoryGb: 64, memoryType: 'LPDDR5X unified', bandwidthGbs: 307, unified: true, usableFraction: appleFrac(64), usableNote: appleNote(64), sources: [APPLE] },
  { id: 'm5-max', vendor: 'Apple', name: 'M5 Max', memoryGb: 128, memoryType: 'LPDDR5X unified', bandwidthGbs: 614, unified: true, usableFraction: appleFrac(128), usableNote: appleNote(128), sources: [APPLE] },

  // ───────────────────────────── AMD ─────────────────────────────
  { id: 'mi300x', vendor: 'AMD', name: 'Instinct MI300X', memoryGb: 192, memoryType: 'HBM3', bandwidthGbs: 5300, unified: false, sources: [AMD] },
  { id: 'mi325x', vendor: 'AMD', name: 'Instinct MI325X', memoryGb: 256, memoryType: 'HBM3e', bandwidthGbs: 6000, unified: false, sources: [AMD] },
  { id: 'mi355x', vendor: 'AMD', name: 'Instinct MI355X', memoryGb: 288, memoryType: 'HBM3e', bandwidthGbs: 8000, unified: false, sources: [AMD] },
  { id: 'rx-7900-xtx', vendor: 'AMD', name: 'Radeon RX 7900 XTX', memoryGb: 24, memoryType: 'GDDR6', bandwidthGbs: 960, unified: false, sources: ['https://www.amd.com/en/products/graphics/desktops/radeon.html'] },
  { id: 'rx-9070-xt', vendor: 'AMD', name: 'Radeon RX 9070 XT', memoryGb: 16, memoryType: 'GDDR6', bandwidthGbs: 640, unified: false, sources: ['https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html'] },
  { id: 'r9700', vendor: 'AMD', name: 'Radeon AI PRO R9700', memoryGb: 32, memoryType: 'GDDR6', bandwidthGbs: 640, unified: false, sources: ['https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html'] },
  { id: 'w7900', vendor: 'AMD', name: 'Radeon PRO W7900', memoryGb: 48, memoryType: 'GDDR6 (ECC)', bandwidthGbs: 864, unified: false, sources: ['https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html'] },
  {
    id: 'ryzen-ai-max-395',
    vendor: 'AMD',
    name: 'Ryzen AI Max+ 395 (Strix Halo, 128 GB)',
    memoryGb: 128,
    memoryType: 'LPDDR5X unified',
    bandwidthGbs: 256,
    unified: true,
    usableFraction: 0.75,
    usableNote: 'Unified LPDDR5X: Windows Variable Graphics Memory caps GPU VRAM at 96 GB of 128 GB; Linux GTT can reach ~110–120 GB.',
    sources: ['https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html'],
  },

  // ──────────────────────────── Tenstorrent ────────────────────────────
  // usableGbPerChip: usable DRAM per chip — full marketed per-chip capacity
  // (Wormhole 12, Blackhole 32 GB). Drives the per-chip sharding model. See notes/tenstorrent-memory.md.
  { id: 'tt-n150', vendor: 'Tenstorrent', name: 'Wormhole n150', memoryGb: 12, memoryType: 'GDDR6', bandwidthGbs: 288, unified: false, numChips: 1, memoryPerChipGb: 12, usableGbPerChip: 12, buyUrl: 'https://tenstorrent.com/en/hardware/wormhole', sources: [TT + 'aibs/wormhole/specifications.html'] },
  { id: 'tt-n300', vendor: 'Tenstorrent', name: 'Wormhole n300', memoryGb: 24, memoryType: 'GDDR6', bandwidthGbs: 576, unified: false, numChips: 2, memoryPerChipGb: 12, usableGbPerChip: 12, buyUrl: 'https://tenstorrent.com/en/hardware/wormhole', sources: [TT + 'aibs/wormhole/specifications.html'] },
  { id: 'tt-p150a', vendor: 'Tenstorrent', name: 'Blackhole p150a', memoryGb: 32, memoryType: 'GDDR6', bandwidthGbs: 512, unified: false, numChips: 1, memoryPerChipGb: 32, usableGbPerChip: 32, buyUrl: 'https://tenstorrent.com/en/hardware/blackhole', sources: [TT + 'aibs/blackhole/specifications.html'] },
  { id: 'tt-t3k', vendor: 'Tenstorrent', name: 'T3K / TT-LoudBox (8× Wormhole)', memoryGb: 96, memoryType: 'GDDR6', bandwidthGbs: 2304, unified: false, numChips: 8, memoryPerChipGb: 12, usableGbPerChip: 12, buyUrl: 'https://tenstorrent.com/en/hardware/tt-loudbox', sources: [TT + 'systems/t3000/specifications.html'] },
  { id: 'tt-qb-wh', vendor: 'Tenstorrent', name: 'TT-QuietBox (8× Wormhole)', memoryGb: 96, memoryType: 'GDDR6', bandwidthGbs: 2304, unified: false, numChips: 8, memoryPerChipGb: 12, usableGbPerChip: 12, buyUrl: 'https://tenstorrent.com/en/hardware/tt-quietbox', sources: [TT + 'systems/quietbox/quietbox-wh/specifications.html'] },
  { id: 'tt-qb2-bh', vendor: 'Tenstorrent', name: 'TT-QuietBox "QB2" (4× Blackhole)', memoryGb: 128, memoryType: 'GDDR6', bandwidthGbs: 2048, unified: false, numChips: 4, memoryPerChipGb: 32, usableGbPerChip: 32, buyUrl: 'https://tenstorrent.com/en/hardware/tt-quietbox', sources: [TT + 'systems/quietbox/quietbox-bh/specifications.html'] },
  { id: 'tt-loudbox-bh', vendor: 'Tenstorrent', name: 'TT-LoudBox (8× Blackhole)', memoryGb: 256, memoryType: 'GDDR6', bandwidthGbs: 4096, unified: false, numChips: 8, memoryPerChipGb: 32, usableGbPerChip: 32, buyUrl: 'https://tenstorrent.com/en/hardware/tt-loudbox', sources: [TT + 'systems/loudbox-bh/specifications.html'] },
  { id: 'tt-galaxy-wh', vendor: 'Tenstorrent', name: 'Galaxy (32× Wormhole)', memoryGb: 384, memoryType: 'GDDR6', bandwidthGbs: 9216, unified: false, numChips: 32, memoryPerChipGb: 12, usableGbPerChip: 12, buyUrl: 'https://tenstorrent.com/en/hardware/galaxy', sources: ['https://tenstorrent.com/en/hardware/galaxy'] },
  { id: 'tt-galaxy-bh', vendor: 'Tenstorrent', name: 'Galaxy (32× Blackhole, 1 TB)', memoryGb: 1024, memoryType: 'GDDR6', bandwidthGbs: 16384, unified: false, numChips: 32, memoryPerChipGb: 32, usableGbPerChip: 32, buyUrl: 'https://tenstorrent.com/en/hardware/galaxy', sources: ['https://tenstorrent.com/en/hardware/galaxy'] },
]
