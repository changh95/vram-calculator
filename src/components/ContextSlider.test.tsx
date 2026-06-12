import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ContextSlider } from './ContextSlider'

describe('ContextSlider', () => {
  const presets = [1024, 4096, 8192, 32768, 131072, 1048576]

  it('renders a slider over the preset ladder showing the compact value', () => {
    render(<ContextSlider label="Context length" presets={presets} value={8192} onChange={() => {}} />)
    const slider = screen.getByRole('slider', { name: /context length/i })
    expect(slider).toHaveValue('2') // index of 8192
    expect(screen.getByTestId('context-value')).toHaveTextContent('8K')
  })

  it('maps slider movement back to preset token counts', () => {
    const onChange = vi.fn()
    render(<ContextSlider label="Context length" presets={presets} value={8192} onChange={onChange} />)
    fireEvent.change(screen.getByRole('slider'), { target: { value: '4' } })
    expect(onChange).toHaveBeenCalledWith(131072)
  })

  it('snaps a non-preset value to the nearest preset index', () => {
    render(<ContextSlider label="Context length" presets={presets} value={9000} onChange={() => {}} />)
    expect(screen.getByRole('slider')).toHaveValue('2')
  })
})
