import { formatTokens } from '../lib/format'

export interface ContextSliderProps {
  label: string
  presets: number[]
  value: number
  onChange: (tokens: number) => void
}

/** Slider over a preset token ladder (context lengths are never arbitrary). */
export function ContextSlider({ label, presets, value, onChange }: ContextSliderProps) {
  const index = presets.reduce(
    (best, p, i) => (Math.abs(p - value) < Math.abs(presets[best] - value) ? i : best),
    0,
  )
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-[13px] font-medium">
        {label}
        <span data-testid="context-value" className="font-display text-sm font-semibold text-accent">
          {formatTokens(presets[index])}
        </span>
      </div>
      <input
        type="range"
        aria-label={label}
        min={0}
        max={presets.length - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(presets[Number(e.target.value)])}
        className="w-full accent-(--accent)"
      />
      <div className="mt-1 flex justify-between font-display text-[10px] text-ink-faint">
        {presets.map((p) => (
          <span key={p}>{formatTokens(p)}</span>
        ))}
      </div>
    </div>
  )
}
