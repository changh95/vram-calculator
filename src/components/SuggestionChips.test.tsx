import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { Suggestion } from '../lib/suggestions'
import { SuggestionChips } from './SuggestionChips'

const suggestions: Suggestion[] = [
  { kind: 'quant', quantId: 'q4', verdict: 'fits', savingGb: 31.2, text: 'Quantize weights to Q4 to save ~31 GB.' },
  { kind: 'hardware', hardwareId: 'gpu-48', verdict: 'tight', text: 'This configuration fits on GPU 48 (tight).' },
]

describe('SuggestionChips', () => {
  it('renders one actionable chip per suggestion', () => {
    render(<SuggestionChips suggestions={suggestions} onApply={() => {}} />)
    expect(screen.getAllByRole('button')).toHaveLength(2)
    expect(screen.getByText(/Quantize weights to Q4/)).toBeInTheDocument()
  })

  it('applies the clicked suggestion', async () => {
    const onApply = vi.fn()
    render(<SuggestionChips suggestions={suggestions} onApply={onApply} />)
    await userEvent.click(screen.getByText(/fits on GPU 48/))
    expect(onApply).toHaveBeenCalledWith(suggestions[1])
  })

  it('renders nothing when there are no suggestions', () => {
    const { container } = render(<SuggestionChips suggestions={[]} onApply={() => {}} />)
    expect(container).toBeEmptyDOMElement()
  })
})
