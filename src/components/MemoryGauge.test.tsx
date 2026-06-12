import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MemoryGauge, type GaugePart } from './MemoryGauge'

const parts: GaugePart[] = [
  { id: 'weights', label: 'Weights', gb: 12 },
  { id: 'kv', label: 'KV cache', gb: 6 },
  { id: 'act', label: 'Activations', gb: 1 },
  { id: 'ovh', label: 'Overhead', gb: 1 },
]

describe('MemoryGauge', () => {
  it('renders one segment per part, width proportional to the gauge scale', () => {
    // total 20, usable 24 → scale max 24 → weights = 50% of the track
    render(<MemoryGauge parts={parts} usableGb={24} />)
    const weights = screen.getByTestId('gauge-seg-weights')
    expect(weights.style.width).toBe('50%')
    expect(screen.getAllByTestId(/gauge-seg-/)).toHaveLength(4)
  })

  it('places the capacity marker at the usable fraction of the scale', () => {
    render(<MemoryGauge parts={parts} usableGb={24} />)
    // scale max = 24 → marker at 100%
    expect(screen.getByTestId('gauge-capacity-marker').style.left).toBe('100%')
  })

  it('scales to the required total when it exceeds capacity, and hatches the overflow', () => {
    const heavy = [{ id: 'weights', label: 'Weights', gb: 48 }]
    render(<MemoryGauge parts={heavy} usableGb={24} />)
    // scale max = 48 → marker at 50%, overflow hatch spans the right half
    expect(screen.getByTestId('gauge-capacity-marker').style.left).toBe('50%')
    expect(screen.getByTestId('gauge-overflow')).toBeInTheDocument()
  })

  it('omits the overflow hatch when everything fits', () => {
    render(<MemoryGauge parts={parts} usableGb={24} />)
    expect(screen.queryByTestId('gauge-overflow')).not.toBeInTheDocument()
  })

  it('is described for assistive tech', () => {
    render(<MemoryGauge parts={parts} usableGb={24} />)
    const meter = screen.getByRole('meter')
    expect(meter).toHaveAccessibleName(/memory/i)
  })
})
