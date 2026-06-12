import { useEffect, useMemo, useState } from 'react'
import { CONTEXT_PRESETS, DEFAULTS, FRAMEWORKS, HARDWARE, MODELS, QUANTS, TT_QUANTS } from './lib/data'
import { estimateVram } from './lib/estimateVram'
import { suggestFixes, type Suggestion } from './lib/suggestions'
import { parseConfigSearch, serializeConfigSearch, type ConfigSelection } from './lib/urlState'
import { formatGb, formatTokens } from './lib/format'
import { LANGUAGES, t, type Lang, type TKey } from './lib/i18n'
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
    options: g.items.map((h) => ({
      value: h.id,
      label: h.name,
      meta: `${h.gpusPerNode ? h.memoryGb * h.gpusPerNode : h.memoryGb} GB`,
    })),
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
  const [lang, setLang] = useState<Lang>(
    () => (localStorage.getItem('vram-lang') as Lang) || 'en',
  )
  const tr = (key: TKey, params?: Record<string, string | number>) => t(lang, key, params)

  useEffect(() => {
    const qs = serializeConfigSearch(sel)
    window.history.replaceState(null, '', `${window.location.pathname}?${qs}`)
  }, [sel])

  useEffect(() => {
    if (theme === 'light') document.documentElement.setAttribute('data-theme', 'light')
    else document.documentElement.removeAttribute('data-theme')
  }, [theme])

  useEffect(() => {
    document.documentElement.lang = lang
    localStorage.setItem('vram-lang', lang)
  }, [lang])

  const config = useMemo(() => resolve(sel), [sel])
  const result = useMemo(() => estimateVram(config), [config])
  const suggestions = useMemo(
    () =>
      suggestFixes(config, {
        quants: config.hardware.usableGbPerChip !== undefined ? TT_QUANTS : QUANTS,
        hardware: HARDWARE,
        maxDevices: 16,
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

  // Total (aggregate) memory: usage and capacity summed across all devices/chips.
  // Even sharding makes total/total equal the per-chip ratio, so the verdict is
  // unchanged — this is just the more intuitive "total used / total available" view.
  const parts: GaugePart[] = [
    { id: 'weights', label: tr('weights'), gb: result.weightsGb },
    { id: 'kv', label: tr('kvCache'), gb: result.kvCacheGb },
    { id: 'act', label: tr('activations'), gb: result.activationsGb },
    { id: 'ovh', label: tr('overhead'), gb: result.overheadGb },
  ]
  const ttHw = config.hardware.usableGbPerChip !== undefined
  const unitCount =
    (ttHw ? (config.hardware.numChips ?? 1) : (config.hardware.gpusPerNode ?? 1)) * config.deviceCount
  const unitKey: TKey = ttHw ? 'chips' : config.hardware.gpusPerNode ? 'gpus' : 'devicesUnit'

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
            <p className="mt-1 text-[13px] text-ink-dim">{tr('tagline')}</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-display rounded-full border border-edge bg-panel px-3 py-1.5 text-[11px] font-semibold text-ink-dim">
              {tr('inference')}
            </span>
            <select
              aria-label={tr('language')}
              value={lang}
              onChange={(e) => setLang(e.target.value as Lang)}
              className="font-display cursor-pointer rounded-full border border-edge bg-panel px-3 py-1.5 text-[11px] font-semibold text-ink-dim outline-none transition-colors hover:border-accent hover:text-ink"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              aria-label="Toggle theme"
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              className="rounded-full border border-edge bg-panel px-3 py-1.5 text-[11px] font-semibold text-ink-dim transition-colors hover:border-accent hover:text-ink"
            >
              {theme === 'dark' ? `☾ ${tr('dark')}` : `☀ ${tr('light')}`}
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          {/* CONFIG */}
          <section className="rounded-xl border border-edge bg-panel p-5">
            <h2 className="font-display mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
              {tr('configuration')}
            </h2>
            <div className="space-y-4">
              <SelectField label={tr('model')} hint={`${m.moe ? 'MoE' : 'dense'} · ${archLabel[m.attention.kind]}`} groups={MODEL_GROUPS} value={sel.modelId} onChange={(v) => set({ modelId: v })} />
              <div>
                <SelectField label={tr('hardware')} hint={tr('memoryCapacity')} groups={HARDWARE_GROUPS} value={sel.hardwareId} onChange={(v) => set({ hardwareId: v })} />
                {config.hardware.buyUrl && (
                  <a
                    href={config.hardware.buyUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-display mt-2 inline-flex items-center gap-1.5 rounded-sm border border-accent/50 bg-accent/10 px-3 py-1.5 text-[11px] font-semibold text-accent transition-colors hover:bg-accent/20"
                  >
                    {tr('buyNow')} <span aria-hidden="true">↗</span>
                  </a>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <SelectField
                  label={tr('quantization')}
                  hint={ttHw ? tr('blockFloatProfile') : tr('bitsWeight')}
                  groups={ttHw ? QUANT_GROUP_TT : QUANT_GROUP}
                  value={config.quant.id}
                  onChange={(v) => set({ quantId: v })}
                />
                <SelectField label={tr('servingFramework')} hint={tr('kvStrategy')} groups={FRAMEWORK_GROUP} value={sel.frameworkId} onChange={(v) => set({ frameworkId: v })} />
              </div>
              <ContextSlider label={tr('contextLength')} presets={CONTEXT_PRESETS} value={sel.contextLength} onChange={(v) => set({ contextLength: v })} />
              <div className="grid grid-cols-2 gap-x-6 gap-y-3 pt-1">
                <Stepper label={tr('concurrentSeqs')} hint={tr('continuousBatch')} value={sel.concurrentSequences} min={1} max={64} onChange={(v) => set({ concurrentSequences: v })} />
                <Stepper label={tr('devices')} hint={`× ${config.hardware.name.split(' ')[0]}`} value={sel.deviceCount} min={1} max={16} onChange={(v) => set({ deviceCount: v })} />
              </div>
            </div>
          </section>

          {/* RESULTS */}
          <section className="rounded-xl border border-edge bg-panel p-5">
            <h2 className="font-display mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
              {tr('results')}
            </h2>

            <div data-testid="total-required" className="sr-only">
              {formatGb(result.totalGb)}
            </div>

            <VerdictBadge
              verdict={result.verdict}
              totalGb={result.totalGb}
              usableGb={result.usableGb}
              shortfallGb={result.shortfallGb}
              labels={{
                fits: tr('fits'),
                tight: tr('tight'),
                wontFit: tr('wontFit'),
                required: tr('required'),
                usable: tr('usable'),
                overBy: tr('overBy'),
              }}
            />

            <div className="mt-5">
              <MemoryGauge parts={parts} usableGb={result.usableGb} />
            </div>

            <div className="mt-4 flex items-baseline justify-between">
              <h3 className="font-display text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
                {tr('allocation')}
                {unitCount > 1 ? ` · ${tr('totalAcross', { n: unitCount, unit: tr(unitKey) })}` : ''}
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
                  {tr('toMakeFit')}
                </h3>
                <SuggestionChips suggestions={suggestions} onApply={applySuggestion} />
              </div>
            )}

            <div className="mt-5">
              <AssumptionsList assumptions={result.assumptions} title={tr('assumptions')} />
            </div>
          </section>
        </div>

        <footer className="mt-6 text-center text-[11px] text-ink-faint">{tr('footer')}</footer>
      </div>
    </div>
  )
}
