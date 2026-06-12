import { formatGb } from '../lib/format'
import type { GaugePart } from './MemoryGauge'

const DOT_COLOR: Record<string, string> = {
  weights: 'var(--seg-weights)',
  kv: 'var(--seg-kv)',
  act: 'var(--seg-act)',
  ovh: 'var(--seg-ovh)',
}

export function AllocationLegend({ parts }: { parts: GaugePart[] }) {
  const total = parts.reduce((sum, p) => sum + p.gb, 0)
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2">
      {parts.map((p) => (
        <div
          key={p.id}
          className="flex items-baseline justify-between border-b border-edge pb-1.5 text-xs"
        >
          <dt className="flex items-center gap-2 text-ink-dim">
            <span
              aria-hidden="true"
              className="h-2 w-2 rounded-[2px]"
              style={{ background: DOT_COLOR[p.id] ?? 'var(--accent)' }}
            />
            {p.label}
          </dt>
          <dd className="font-display font-semibold">
            {formatGb(p.gb)}
            <span className="ml-0.5 text-[10px] font-normal text-ink-faint">GB</span>
            <span className="ml-2 text-[10px] font-normal text-ink-faint">
              {total === 0 ? '0%' : `${Math.round((p.gb / total) * 100)}%`}
            </span>
          </dd>
        </div>
      ))}
    </dl>
  )
}
