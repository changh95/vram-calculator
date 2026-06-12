import type { Verdict } from '../lib/types'
import { formatGb } from '../lib/format'

const STYLES: Record<Verdict, { label: string; icon: string; fg: string; ring: string }> = {
  fits: { label: 'FITS', icon: '✓', fg: 'text-fit', ring: 'border-fit/40 bg-fit/10' },
  tight: { label: 'TIGHT', icon: '▲', fg: 'text-tight', ring: 'border-tight/40 bg-tight/10' },
  'wont-fit': { label: "WON'T FIT", icon: '✕', fg: 'text-nofit', ring: 'border-nofit/40 bg-nofit/10' },
}

export interface VerdictBadgeProps {
  verdict: Verdict
  totalGb: number
  usableGb: number
  shortfallGb: number | null
}

export function VerdictBadge({ verdict, totalGb, usableGb, shortfallGb }: VerdictBadgeProps) {
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
        {s.label}
      </span>
      <div className="min-w-0">
        <div className="font-display text-3xl font-bold leading-none tracking-tight">
          {formatGb(totalGb)}
          <span className="ml-1.5 text-sm font-medium text-ink-faint">GB required</span>
        </div>
        <div className="mt-1.5 text-xs text-ink-dim">
          of {formatGb(usableGb)} GB usable
          {shortfallGb !== null && (
            <span className="ml-2 font-semibold text-nofit">over by {formatGb(shortfallGb)} GB</span>
          )}
        </div>
      </div>
    </div>
  )
}
