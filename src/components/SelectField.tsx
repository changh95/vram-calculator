import { useId } from 'react'

export interface SelectOption {
  value: string
  label: string
  /** Right-aligned metadata shown next to the label, e.g. "32B" or "24 GB". */
  meta?: string
}

export interface SelectGroup {
  label: string
  options: SelectOption[]
}

export interface SelectFieldProps {
  label: string
  hint?: string
  groups: SelectGroup[]
  value: string
  onChange: (value: string) => void
}

export function SelectField({ label, hint, groups, value, onChange }: SelectFieldProps) {
  const id = useId()
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 flex items-baseline justify-between text-[13px] font-medium">
        {label}
        {hint && <span className="text-[10px] font-normal text-ink-faint">{hint}</span>}
      </label>
      <select
        id={id}
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full cursor-pointer appearance-none rounded-sm border border-edge bg-panel-2 px-3 py-2.5 text-[13px] text-ink outline-none transition-colors focus:border-accent"
      >
        {groups.map((g) => (
          <optgroup key={g.label} label={g.label}>
            {g.options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.meta ? `${o.label} · ${o.meta}` : o.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  )
}
