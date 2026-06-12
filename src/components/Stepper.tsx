export interface StepperProps {
  label: string
  value: number
  min: number
  max: number
  onChange: (value: number) => void
  hint?: string
}

export function Stepper({ label, value, min, max, onChange, hint }: StepperProps) {
  const button =
    'font-display h-8 w-8 rounded-sm border border-edge bg-panel-2 text-sm text-ink-dim ' +
    'transition-colors hover:border-accent hover:text-ink disabled:cursor-not-allowed ' +
    'disabled:opacity-35 disabled:hover:border-edge disabled:hover:text-ink-dim'
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-[13px] font-medium">
        {label}
        {hint && <span className="ml-2 text-[10px] font-normal text-ink-faint">{hint}</span>}
      </span>
      <span className="flex items-center gap-2">
        <button
          type="button"
          aria-label={`Decrease ${label.toLowerCase()}`}
          disabled={value <= min}
          onClick={() => onChange(Math.max(min, value - 1))}
          className={button}
        >
          −
        </button>
        <span className="font-display min-w-8 text-center text-sm font-semibold text-accent">
          {value}
        </span>
        <button
          type="button"
          aria-label={`Increase ${label.toLowerCase()}`}
          disabled={value >= max}
          onClick={() => onChange(Math.min(max, value + 1))}
          className={button}
        >
          +
        </button>
      </span>
    </div>
  )
}
