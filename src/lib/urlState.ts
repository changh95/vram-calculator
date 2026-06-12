/**
 * Shareable-link state: the user's selection encoded as a query string.
 * Only ids and numbers travel in the URL; resolving ids against the data
 * tables (and falling back when an id no longer exists) is the app's job.
 */

export interface ConfigSelection {
  modelId: string
  hardwareId: string
  quantId: string
  frameworkId: string
  deviceCount: number
  contextLength: number
  concurrentSequences: number
}

export function serializeConfigSearch(sel: ConfigSelection): string {
  return new URLSearchParams({
    m: sel.modelId,
    hw: sel.hardwareId,
    q: sel.quantId,
    fw: sel.frameworkId,
    n: String(sel.deviceCount),
    ctx: String(sel.contextLength),
    seq: String(sel.concurrentSequences),
  }).toString()
}

function positiveInt(value: string | null, fallback: number): number {
  if (value === null) return fallback
  const n = Number(value)
  return Number.isInteger(n) && n > 0 ? n : fallback
}

export function parseConfigSearch(search: string, fallback: ConfigSelection): ConfigSelection {
  const p = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search)
  return {
    modelId: p.get('m') ?? fallback.modelId,
    hardwareId: p.get('hw') ?? fallback.hardwareId,
    quantId: p.get('q') ?? fallback.quantId,
    frameworkId: p.get('fw') ?? fallback.frameworkId,
    deviceCount: positiveInt(p.get('n'), fallback.deviceCount),
    contextLength: positiveInt(p.get('ctx'), fallback.contextLength),
    concurrentSequences: positiveInt(p.get('seq'), fallback.concurrentSequences),
  }
}
