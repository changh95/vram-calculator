import type { Suggestion } from '../lib/suggestions'

const KIND_ICON: Record<Suggestion['kind'], string> = {
  quant: '▾',
  context: '⇤',
  devices: '⧉',
  hardware: '⇪',
}

export interface SuggestionChipsProps {
  suggestions: Suggestion[]
  onApply: (s: Suggestion) => void
}

/** One-click fixes for a wont-fit config; clicking applies the change. */
export function SuggestionChips({ suggestions, onApply }: SuggestionChipsProps) {
  if (suggestions.length === 0) return null
  return (
    <div className="flex flex-wrap gap-2">
      {suggestions.map((s) => (
        <button
          key={s.kind}
          type="button"
          onClick={() => onApply(s)}
          className="group flex items-center gap-2 rounded-sm border border-edge bg-panel-2 px-3 py-2 text-left text-[11px] leading-snug text-ink-dim transition-colors hover:border-accent hover:text-ink"
        >
          <span aria-hidden="true" className="font-display text-accent">
            {KIND_ICON[s.kind]}
          </span>
          {s.text}
        </button>
      ))}
    </div>
  )
}
