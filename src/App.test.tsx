import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import App from './App'

beforeEach(() => {
  window.history.replaceState(null, '', '/')
  document.documentElement.removeAttribute('data-theme')
  localStorage.clear() // reset persisted language between tests
})

describe('App', () => {
  it('loads with a non-empty default verdict (never a blank state)', () => {
    render(<App />)
    const verdict = screen.getByRole('status')
    expect(verdict.textContent).toMatch(/FITS|TIGHT|WON'T FIT/)
  })

  it('shows the four memory breakdown segments', () => {
    render(<App />)
    expect(screen.getByText('Weights')).toBeInTheDocument()
    expect(screen.getByText('KV cache')).toBeInTheDocument()
    expect(screen.getByText('Activations')).toBeInTheDocument()
    expect(screen.getByText('Overhead')).toBeInTheDocument()
  })

  it('recomputes live when the model changes (no Calculate button needed)', async () => {
    render(<App />)
    const before = screen.getByTestId('total-required').textContent
    await userEvent.selectOptions(screen.getByLabelText('Model'), 'kimi-k2-instruct')
    expect(screen.getByTestId('total-required').textContent).not.toBe(before)
    // a 1T model can't fit a 4090
    expect(screen.getByRole('status').textContent).toMatch(/WON'T FIT/)
  })

  it('offers suggestion chips when the config will not fit, and applying one changes the verdict', async () => {
    render(<App />)
    await userEvent.selectOptions(screen.getByLabelText('Model'), 'deepseek-r1-distill-llama-70b')
    expect(screen.getByRole('status').textContent).toMatch(/WON'T FIT/)
    const suggestions = screen.getByTestId('suggestions')
    const chips = within(suggestions).getAllByRole('button')
    expect(chips.length).toBeGreaterThan(0)
    await userEvent.click(chips[0])
    // applying a fix should move the verdict off "won't fit"
    expect(screen.getByRole('status').textContent).not.toMatch(/WON'T FIT/)
  })

  it('encodes the configuration in the URL for sharing', async () => {
    render(<App />)
    await userEvent.selectOptions(screen.getByLabelText('Model'), 'gemma4-12b')
    expect(window.location.search).toContain('m=gemma4-12b')
  })

  it('restores configuration from the URL on load', () => {
    window.history.replaceState(null, '', '/?m=glm-4.6&hw=h100&q=fp8&fw=vllm&ctx=32768&seq=1&n=1')
    render(<App />)
    expect(screen.getByLabelText('Model')).toHaveValue('glm-4.6')
    expect(screen.getByLabelText(/Hardware/)).toHaveValue('h100')
  })

  it('discloses estimate assumptions', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: /estimates & assumptions/i })).toBeInTheDocument()
  })

  it('toggles light/dark theme', async () => {
    render(<App />)
    expect(document.documentElement.getAttribute('data-theme')).not.toBe('light')
    await userEvent.click(screen.getByRole('button', { name: /theme/i }))
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })

  it('shows TOTAL usage vs total capacity for multiple devices (2× 24 GB → 48 GB)', async () => {
    render(<App />)
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'rtx-4090')
    await userEvent.click(screen.getByRole('button', { name: /increase devices/i }))
    const verdict = screen.getByRole('status').closest('div')!.parentElement!
    expect(verdict).toHaveTextContent(/of 48\.0 GB usable/) // total, not per-device
  })

  it('shows TOTAL (not per-chip) memory for multi-chip Tenstorrent systems (8× 12 GB → 96 GB)', async () => {
    render(<App />)
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'tt-t3k') // 8 Wormhole chips × 12 GB
    const verdict = screen.getByRole('status').closest('div')!.parentElement!
    expect(verdict).toHaveTextContent(/of 96\.0 GB usable/) // total across the mesh
    expect(verdict).not.toHaveTextContent(/of 12\.0 GB usable/) // not per-chip
  })

  it('lists Tenstorrent hardware first in the dropdown', () => {
    render(<App />)
    const select = screen.getByLabelText(/Hardware/) as HTMLSelectElement
    const optgroups = select.querySelectorAll('optgroup')
    expect(optgroups[0].label).toBe('Tenstorrent')
  })

  it('shows a Buy Now link to tenstorrent.com only for Tenstorrent hardware', async () => {
    render(<App />)
    // a non-Tenstorrent card has no Buy Now
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'rtx-4090')
    expect(screen.queryByRole('link', { name: /buy now/i })).not.toBeInTheDocument()
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'tt-n300')
    const buy = screen.getByRole('link', { name: /buy now/i })
    expect(buy.getAttribute('href')).toContain('tenstorrent.com')
    // switching back to a non-TT device removes it
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'rtx-4090')
    expect(screen.queryByRole('link', { name: /buy now/i })).not.toBeInTheDocument()
  })

  it('changing quantization on a Tenstorrent device changes the memory results', async () => {
    render(<App />) // default is Blackhole p150a (Tenstorrent)
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'tt-p150a')
    const before = screen.getByTestId('total-required').textContent
    await userEvent.selectOptions(screen.getByLabelText('Quantization'), 'tt-accuracy')
    expect(screen.getByTestId('total-required').textContent).not.toBe(before)
  })

  it('changing quantization on a GPU device changes the memory results', async () => {
    render(<App />)
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'rtx-4090')
    const before = screen.getByTestId('total-required').textContent
    await userEvent.selectOptions(screen.getByLabelText('Quantization'), 'q8_0')
    expect(screen.getByTestId('total-required').textContent).not.toBe(before)
  })

  it('switches UI language via the top-right dropdown', async () => {
    render(<App />)
    expect(screen.getByText('Configuration')).toBeInTheDocument()
    await userEvent.selectOptions(screen.getByLabelText('Language'), 'ko')
    expect(screen.getByText('구성')).toBeInTheDocument() // Configuration → Korean
    expect(screen.queryByText('Configuration')).not.toBeInTheDocument()
    expect(document.documentElement.lang).toBe('ko')
  })

  it('shows the unified-memory cap note for an Apple device', async () => {
    render(<App />)
    await userEvent.selectOptions(screen.getByLabelText(/Hardware/), 'm4-max')
    expect(screen.getByText(/macOS lets the GPU use/)).toBeInTheDocument()
  })
})
