import type { Assumption } from '../lib/types'

/**
 * Estimate disclosure — trust depends on honesty about approximations
 * (every estimate-grade term in the math surfaces here).
 */
export function AssumptionsList({ assumptions }: { assumptions: Assumption[] }) {
  if (assumptions.length === 0) return null
  return (
    <section className="rounded-sm border border-edge bg-panel-2 px-3.5 py-3">
      <h3 className="font-display mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-accent">
        ℹ Estimates &amp; assumptions
      </h3>
      <ul className="space-y-1.5 text-[11px] leading-relaxed text-ink-dim">
        {assumptions.map((a) => (
          <li key={a.id} className="flex gap-2">
            <span aria-hidden="true" className="text-ink-faint">
              ·
            </span>
            <span>{a.text}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
