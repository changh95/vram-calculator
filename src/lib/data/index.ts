import type { ConfigSelection } from '../urlState'

export { MODELS } from './models'
export { HARDWARE } from './hardware'
export { FRAMEWORKS } from './frameworks'
export { QUANTS, TT_QUANTS } from './quants'

/**
 * Opening state — never blank (design requirement). A 9B-class model on a
 * Blackhole p150a via the Tenstorrent stack produces an immediate "fits" verdict.
 */
export const DEFAULTS: ConfigSelection = {
  modelId: 'qwen3.5-9b',
  hardwareId: 'tt-p150a',
  quantId: 'q4_k_m',
  frameworkId: 'tt-metal',
  deviceCount: 1,
  contextLength: 8192,
  concurrentSequences: 1,
}

/** Preset context ladder for the slider (tokens). */
export const CONTEXT_PRESETS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]
