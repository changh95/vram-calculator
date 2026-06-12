import type { ConfigSelection } from '../urlState'

export { MODELS } from './models'
export { HARDWARE } from './hardware'
export { FRAMEWORKS } from './frameworks'
export { QUANTS } from './quants'

/**
 * Opening state — never blank (design requirement). A 9B-class model at Q4 on a
 * 4090 with llama.cpp produces an immediate "fits" verdict.
 */
export const DEFAULTS: ConfigSelection = {
  modelId: 'qwen3.5-9b',
  hardwareId: 'rtx-4090',
  quantId: 'q4_k_m',
  frameworkId: 'llamacpp',
  deviceCount: 1,
  contextLength: 8192,
  concurrentSequences: 1,
}

/** Preset context ladder for the slider (tokens). */
export const CONTEXT_PRESETS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]
