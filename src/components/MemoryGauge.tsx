import { formatGb } from '../lib/format'

export interface GaugePart {
  id: string
  label: string
  gb: number
}

const SEGMENT_COLOR: Record<string, string> = {
  weights: 'var(--seg-weights)',
  kv: 'var(--seg-kv)',
  act: 'var(--seg-act)',
  ovh: 'var(--seg-ovh)',
}

export interface MemoryGaugeProps {
  parts: GaugePart[]
  usableGb: number
}

/**
 * The instrument: required memory as stacked segments on a ruled track, the
 * capacity line as a marker, and any overflow hatched in the verdict red.
 * The scale stretches to whichever is larger — required or capacity — so
 * over/under is always visible in proportion.
 */
export function MemoryGauge({ parts, usableGb }: MemoryGaugeProps) {
  const totalGb = parts.reduce((sum, p) => sum + p.gb, 0)
  const scaleMax = Math.max(totalGb, usableGb)
  const pct = (gb: number) => (scaleMax === 0 ? 0 : (gb / scaleMax) * 100)
  const ticks = Array.from({ length: 9 }, (_, i) => ((i + 1) / 10) * 100)

  return (
    <div>
      <div
        role="meter"
        aria-label="Memory required versus capacity"
        aria-valuenow={Math.round(totalGb * 100) / 100}
        aria-valuemin={0}
        aria-valuemax={Math.round(scaleMax * 100) / 100}
        aria-valuetext={`${formatGb(totalGb)} GB required of ${formatGb(usableGb)} GB usable`}
        className="relative h-9 overflow-hidden rounded-sm border border-edge bg-panel-2"
      >
        {/* ruler graduations */}
        <div aria-hidden="true" className="pointer-events-none absolute inset-0">
          {ticks.map((t) => (
            <span
              key={t}
              className="absolute bottom-0 h-1.5 w-px bg-edge"
              style={{ left: `${t}%` }}
            />
          ))}
        </div>
        <div className="absolute inset-0 flex">
          {parts.map((p) => (
            <span
              key={p.id}
              data-testid={`gauge-seg-${p.id}`}
              title={`${p.label}: ${formatGb(p.gb)} GB`}
              className="h-full transition-[width] duration-300 ease-out"
              style={{ width: `${pct(p.gb)}%`, background: SEGMENT_COLOR[p.id] ?? 'var(--accent)' }}
            />
          ))}
        </div>
        {totalGb > usableGb && (
          <span
            data-testid="gauge-overflow"
            aria-hidden="true"
            className="hatch-overflow absolute inset-y-0 opacity-60"
            style={{ left: `${pct(usableGb)}%`, width: `${pct(totalGb - usableGb)}%` }}
          />
        )}
        <span
          data-testid="gauge-capacity-marker"
          aria-hidden="true"
          className="absolute inset-y-0 w-0.5 -translate-x-1/2 bg-ink"
          style={{ left: `${pct(usableGb)}%` }}
        />
      </div>
      <div className="mt-1.5 flex justify-between font-display text-[10px] text-ink-faint">
        <span>0</span>
        <span>
          capacity {formatGb(usableGb)} GB · scale {formatGb(scaleMax)} GB
        </span>
      </div>
    </div>
  )
}
