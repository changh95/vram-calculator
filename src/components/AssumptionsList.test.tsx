import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AssumptionsList } from './AssumptionsList'

describe('AssumptionsList', () => {
  it('renders every assumption under an estimate disclosure heading', () => {
    render(
      <AssumptionsList
        assumptions={[
          { id: 'a', text: 'Activation memory is an estimate.' },
          { id: 'b', text: 'KV cache assumed FP16.' },
        ]}
      />,
    )
    expect(screen.getByRole('heading', { name: /estimates & assumptions/i })).toBeInTheDocument()
    expect(screen.getByText('Activation memory is an estimate.')).toBeInTheDocument()
    expect(screen.getByText('KV cache assumed FP16.')).toBeInTheDocument()
  })

  it('renders nothing when there are no assumptions', () => {
    const { container } = render(<AssumptionsList assumptions={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})
