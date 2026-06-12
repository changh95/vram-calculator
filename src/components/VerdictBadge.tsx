import type { Verdict } from '../lib/types'
import { formatGb } from '../lib/format'

const STYLES: Record<Verdict, { icon: string; fg: string; ring: string }> = {
  fits: { icon: '✓', fg: 'text-fit', ring: 'border-fit/40 bg-fit/10' },
  tight: { icon: '▲', fg: 'text-tight', ring: 'border-tight/40 bg-tight/10' },
  'wont-fit': { icon: '✕', fg: 'text-nofit', ring: 'border-nofit/40 bg-nofit/10' },
}

const VERDICT_KEY: Record<Verdict, 'fits' | 'tight' | 'wontFit'> = {
  fits: 'fits',
  tight: 'tight',
  'wont-fit': 'wontFit',
}

export interface VerdictLabels {
  fits: string
  tight: string
  wontFit: string
  /** Suffix after the required number, e.g. "GB required". */
  required: string
  /** Template with `{x}`, e.g. "of {x} GB usable". */
  usable: string
  /** Template with `{x}`, e.g. "over by {x} GB". */
  overBy: string
}

const DEFAULT_LABELS: VerdictLabels = {
  fits: 'FITS',
  tight: 'TIGHT',
  wontFit: "WON'T FIT",
  required: 'GB required',
  usable: 'of {x} GB usable',
  overBy: 'over by {x} GB',
}

export interface VerdictBadgeProps {
  verdict: Verdict
  totalGb: number
  usableGb: number
  shortfallGb: number | null
  labels?: VerdictLabels
}

export function VerdictBadge({ verdict, totalGb, usableGb, shortfallGb, labels = DEFAULT_LABELS }: VerdictBadgeProps) {
  const s = STYLES[verdict]
  return (
    <div className="flex items-center gap-5 rounded-md border border-edge bg-panel-2 px-5 py-4">
      <span
        role="status"
        className={`font-display flex items-center gap-2 rounded-sm border px-3 py-1.5 text-[13px] font-bold tracking-wider ${s.fg} ${s.ring}`}
      >
        <span data-testid="verdict-icon" aria-hidden="true">
          {s.icon}
        </span>
        {labels[VERDICT_KEY[verdict]]}
      </span>
      <div className="min-w-0">
        <div className="font-display text-3xl font-bold leading-none tracking-tight">
          {formatGb(totalGb)}
          <span className="ml-1.5 text-sm font-medium text-ink-faint">{labels.required}</span>
        </div>
        <div className="mt-1.5 text-xs text-ink-dim">
          {labels.usable.replace('{x}', formatGb(usableGb))}
          {shortfallGb !== null && (
            <span className="ml-2 font-semibold text-nofit">{labels.overBy.replace('{x}', formatGb(shortfallGb))}</span>
          )}
        </div>
      </div>
    </div>
  )
}
