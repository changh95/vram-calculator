import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { VerdictBadge } from './VerdictBadge'

describe('VerdictBadge', () => {
  it('announces FITS as a status with an icon (never color alone)', () => {
    render(<VerdictBadge verdict="fits" totalGb={17.4} usableGb={24} shortfallGb={null} />)
    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('FITS')
    expect(screen.getByTestId('verdict-icon')).toBeInTheDocument()
  })

  it('shows the required total against the usable capacity', () => {
    render(<VerdictBadge verdict="fits" totalGb={17.4} usableGb={24} shortfallGb={null} />)
    expect(screen.getByText('17.4')).toBeInTheDocument()
    expect(screen.getByText(/of 24\.0 GB usable/)).toBeInTheDocument()
  })

  it('renders TIGHT', () => {
    render(<VerdictBadge verdict="tight" totalGb={21.9} usableGb={24} shortfallGb={null} />)
    expect(screen.getByRole('status')).toHaveTextContent('TIGHT')
  })

  it("renders WON'T FIT with the shortfall", () => {
    render(<VerdictBadge verdict="wont-fit" totalGb={38} usableGb={24} shortfallGb={14} />)
    expect(screen.getByRole('status')).toHaveTextContent(/WON'T FIT/)
    expect(screen.getByText(/over by 14\.0 GB/)).toBeInTheDocument()
  })
})
