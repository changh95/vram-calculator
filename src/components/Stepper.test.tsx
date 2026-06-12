import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Stepper } from './Stepper'

describe('Stepper', () => {
  it('shows the label and current value', () => {
    render(<Stepper label="Devices" value={2} min={1} max={8} onChange={() => {}} />)
    expect(screen.getByText('Devices')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
  })

  it('increments and decrements', async () => {
    const onChange = vi.fn()
    render(<Stepper label="Devices" value={2} min={1} max={8} onChange={onChange} />)
    await userEvent.click(screen.getByRole('button', { name: /increase devices/i }))
    expect(onChange).toHaveBeenCalledWith(3)
    await userEvent.click(screen.getByRole('button', { name: /decrease devices/i }))
    expect(onChange).toHaveBeenCalledWith(1)
  })

  it('disables the buttons at the bounds', () => {
    render(<Stepper label="Devices" value={1} min={1} max={1} onChange={() => {}} />)
    expect(screen.getByRole('button', { name: /decrease devices/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /increase devices/i })).toBeDisabled()
  })
})
