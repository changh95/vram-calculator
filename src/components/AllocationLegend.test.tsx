import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AllocationLegend } from './AllocationLegend'

describe('AllocationLegend', () => {
  it('lists every part with its GB value and share of the total', () => {
    render(
      <AllocationLegend
        parts={[
          { id: 'weights', label: 'Weights', gb: 15 },
          { id: 'kv', label: 'KV cache', gb: 5 },
        ]}
      />,
    )
    expect(screen.getByText('Weights')).toBeInTheDocument()
    expect(screen.getByText('15.0')).toBeInTheDocument()
    expect(screen.getByText('75%')).toBeInTheDocument()
    expect(screen.getByText('KV cache')).toBeInTheDocument()
    expect(screen.getByText('25%')).toBeInTheDocument()
  })
})
