import { useEffect, useMemo, useState } from 'react'
import { CONTEXT_PRESETS, DEFAULTS, FRAMEWORKS, HARDWARE, MODELS, QUANTS, TT_QUANTS } from './lib/data'
import { estimateVram } from './lib/estimateVram'
import { suggestFixes, type Suggestion } from './lib/suggestions'
import { parseConfigSearch, serializeConfigSearch, type ConfigSelection } from './lib/urlState'
import { formatGb, formatTokens } from './lib/format'
import type { CalcConfig } from './lib/types'
import { SelectField, type SelectGroup } from './components/SelectField'
import { ContextSlider } from './components/ContextSlider'
import { Stepper } from './components/Stepper'
import { VerdictBadge } from './components/VerdictBadge'
import { MemoryGauge, type GaugePart } from './components/MemoryGauge'
import { AllocationLegend } from './components/AllocationLegend'
import { AssumptionsList } from './components/AssumptionsList'
import { SuggestionChips } from './components/SuggestionChips'

const byId = <T extends { id: string }>(arr: T[], id: string, fallback: string): T =>
  arr.find((x) => x.id === id) ?? arr.find((x) => x.id === fallback)!

function groupBy<T>(items: T[], key: (t: T) => string): { label: string; items: T[] }[] {
  const out: { label: string; items: T[] }[] = []
  for (const item of items) {
    const label = key(item)
    const group = out.find((g) => g.label === label) ?? (out.push({ label, items: [] }), out[out.length - 1])
    group.items.push(item)
  }
  return out
}

const paramLabel = (b: number) => (b >= 1000 ? `${(b / 1000).toFixed(2)}T` : `${b}B`)

const MODEL_GROUPS: SelectGroup[] = groupBy(MODELS, (m) => m.family).map((g) => ({
  label: g.label,
  options: g.items.map((m) => ({
    value: m.id,
    label: m.name,
    meta: m.moe && m.paramsActiveB ? `${paramLabel(m.paramsTotalB)}·A${m.paramsActiveB}` : paramLabel(m.paramsTotalB),
  })),
}))

// Tenstorrent leads the vendor list; the rest keep their natural order (stable sort).
const HARDWARE_GROUPS: SelectGroup[] = groupBy(HARDWARE, (h) => h.vendor)
  .sort((a, b) => (a.label === 'Tenstorrent' ? 0 : 1) - (b.label === 'Tenstorrent' ? 0 : 1))
  .map((g) => ({
    label: g.label,
    options: g.items.map((h) => ({ value: h.id, label: h.name, meta: `${h.memoryGb} GB` })),
  }))

const QUANT_GROUP: SelectGroup[] = [
  { label: 'Precision', options: QUANTS.map((q) => ({ value: q.id, label: q.label, meta: `${q.bitsPerWeight} bpw` })) },
]

// Tenstorrent runs block-float (not GGUF/AWQ), so TT hardware gets its own
// precision profiles instead of the GPU quant ladder.
const QUANT_GROUP_TT: SelectGroup[] = [
  { label: 'Tenstorrent block-float', options: TT_QUANTS.map((q) => ({ value: q.id, label: q.label })) },
]

const FRAMEWORK_GROUP: SelectGroup[] = [
  { label: 'Framework', options: FRAMEWORKS.map((f) => ({ value: f.id, label: f.label })) },
]

function resolve(sel: ConfigSelection): CalcConfig {
  const hardware = byId(HARDWARE, sel.hardwareId, DEFAULTS.hardwareId)
  // TT hardware uses block-float profiles; if a GPU quant id is carried over
  // (or vice-versa), fall back to that list's default so the control is valid.
  const tt = hardware.usableGbPerChip !== undefined
  const quant = tt
    ? byId(TT_QUANTS, sel.quantId, 'tt-performance')
    : byId(QUANTS, sel.quantId, DEFAULTS.quantId)
  return {
    model: byId(MODELS, sel.modelId, DEFAULTS.modelId),
    hardware,
    quant,
    framework: byId(FRAMEWORKS, sel.frameworkId, DEFAULTS.frameworkId),
    deviceCount: sel.deviceCount,
    contextLength: sel.contextLength,
    concurrentSequences: sel.concurrentSequences,
  }
}

export default function App() {
  const [sel, setSel] = useState<ConfigSelection>(() =>
    parseConfigSearch(window.location.search, DEFAULTS),
  )
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  useEffect(() => {
    const qs = serializeConfigSearch(sel)
    window.history.replaceState(null, '', `${window.location.pathname}?${qs}`)
  }, [sel])

  useEffect(() => {
    if (theme === 'light') document.documentElement.setAttribute('data-theme', 'light')
    else document.documentElement.removeAttribute('data-theme')
  }, [theme])

  const config = useMemo(() => resolve(sel), [sel])
  const result = useMemo(() => estimateVram(config), [config])
  const suggestions = useMemo(
    () =>
      suggestFixes(config, {
        quants: config.hardware.usableGbPerChip !== undefined ? TT_QUANTS : QUANTS,
        hardware: HARDWARE,
        maxDevices: 8,
      }),
    [config],
  )

  const set = (patch: Partial<ConfigSelection>) => setSel((s) => ({ ...s, ...patch }))

  const applySuggestion = (s: Suggestion) => {
    if (s.kind === 'quant' && s.quantId) set({ quantId: s.quantId })
    else if (s.kind === 'context' && s.contextLength) set({ contextLength: s.contextLength })
    else if (s.kind === 'devices' && s.deviceCount) set({ deviceCount: s.deviceCount })
    else if (s.kind === 'hardware' && s.hardwareId) set({ hardwareId: s.hardwareId })
  }

  // Aggregate breakdown, scaled to a per-device view so the gauge and legend
  // both compare against per-device usable capacity and sum to perDeviceGb.
  const factor = result.totalGb > 0 ? result.perDeviceGb / result.totalGb : 0
  const parts: GaugePart[] = [
    { id: 'weights', label: 'Weights', gb: result.weightsGb * factor },
    { id: 'kv', label: 'KV cache', gb: result.kvCacheGb * factor },
    { id: 'act', label: 'Activations', gb: result.activationsGb * factor },
    { id: 'ovh', label: 'Overhead', gb: result.overheadGb * factor },
  ]

  const m = config.model
  const archLabel: Record<string, string> = {
    gqa: 'GQA', mla: 'MLA', 'gqa-swa': 'sliding-window', hybrid: 'hybrid SSM', compressed: 'compressed',
  }

  return (
    <div className="min-h-screen">
      <div className="mx-auto max-w-[1200px] px-6 py-7">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="font-display text-xl font-extrabold tracking-tight">
              VRAM<span className="text-accent">·</span>Calculator
            </h1>
            <p className="mt-1 text-[13px] text-ink-dim">Can I run this model on my hardware?</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-display rounded-full border border-edge bg-panel px-3 py-1.5 text-[11px] font-semibold text-ink-dim">
              inference · estimate
            </span>
            <button
              type="button"
              aria-label="Toggle theme"
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              className="rounded-full border border-edge bg-panel px-3 py-1.5 text-[11px] font-semibold text-ink-dim transition-colors hover:border-accent hover:text-ink"
            >
              {theme === 'dark' ? '☾ dark' : '☀ light'}
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          {/* CONFIG */}
          <section className="rounded-xl border border-edge bg-panel p-5">
            <h2 className="font-display mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
              Configuration
            </h2>
            <div className="space-y-4">
              <SelectField label="Model" hint={`${m.moe ? 'MoE' : 'dense'} · ${archLabel[m.attention.kind]}`} groups={MODEL_GROUPS} value={sel.modelId} onChange={(v) => set({ modelId: v })} />
              <div>
                <SelectField label="Hardware" hint="memory capacity" groups={HARDWARE_GROUPS} value={sel.hardwareId} onChange={(v) => set({ hardwareId: v })} />
                {config.hardware.buyUrl && (
                  <a
                    href={config.hardware.buyUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-display mt-2 inline-flex items-center gap-1.5 rounded-sm border border-accent/50 bg-accent/10 px-3 py-1.5 text-[11px] font-semibold text-accent transition-colors hover:bg-accent/20"
                  >
                    Buy now <span aria-hidden="true">↗</span>
                  </a>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <SelectField
                  label="Quantization"
                  hint={config.hardware.usableGbPerChip !== undefined ? 'block-float profile' : 'bits / weight'}
                  groups={config.hardware.usableGbPerChip !== undefined ? QUANT_GROUP_TT : QUANT_GROUP}
                  value={config.quant.id}
                  onChange={(v) => set({ quantId: v })}
                />
                <SelectField label="Serving framework" hint="KV strategy" groups={FRAMEWORK_GROUP} value={sel.frameworkId} onChange={(v) => set({ frameworkId: v })} />
              </div>
              <ContextSlider label="Context length" presets={CONTEXT_PRESETS} value={sel.contextLength} onChange={(v) => set({ contextLength: v })} />
              <div className="grid grid-cols-2 gap-x-6 gap-y-3 pt-1">
                <Stepper label="Concurrent seqs" hint="continuous batch" value={sel.concurrentSequences} min={1} max={64} onChange={(v) => set({ concurrentSequences: v })} />
                <Stepper label="Devices" hint={`× ${config.hardware.name.split(' ')[0]}`} value={sel.deviceCount} min={1} max={8} onChange={(v) => set({ deviceCount: v })} />
              </div>
            </div>
          </section>

          {/* RESULTS */}
          <section className="rounded-xl border border-edge bg-panel p-5">
            <h2 className="font-display mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
              Memory results
            </h2>

            <div data-testid="total-required" className="sr-only">
              {formatGb(result.totalGb)}
            </div>

            <VerdictBadge
              verdict={result.verdict}
              totalGb={result.perDeviceGb}
              usableGb={result.usablePerDeviceGb}
              shortfallGb={result.verdict === 'wont-fit' ? result.perDeviceGb - result.usablePerDeviceGb : null}
            />

            <div className="mt-5">
              <MemoryGauge parts={parts} usableGb={result.usablePerDeviceGb} />
            </div>

            <div className="mt-4 flex items-baseline justify-between">
              <h3 className="font-display text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
                Allocation{config.deviceCount > 1 ? ` · per device (×${config.deviceCount})` : ''}
              </h3>
              <span className="font-display text-[11px] text-ink-faint">
                {formatTokens(config.contextLength)} ctx · {config.framework.kvDtypeLabel} KV
              </span>
            </div>
            <div className="mt-2">
              <AllocationLegend parts={parts} />
            </div>

            {suggestions.length > 0 && (
              <div data-testid="suggestions" className="mt-5">
                <h3 className="font-display mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-nofit">
                  To make it fit
                </h3>
                <SuggestionChips suggestions={suggestions} onApply={applySuggestion} />
              </div>
            )}

            <div className="mt-5">
              <AssumptionsList assumptions={result.assumptions} />
            </div>
          </section>
        </div>

        <footer className="mt-6 text-center text-[11px] text-ink-faint">
          Estimates only — see <span className="text-ink-dim">notes/</span> for the formula and sources. Inference, single-node; training and offloading are out of scope.
        </footer>
      </div>
    </div>
  )
}
