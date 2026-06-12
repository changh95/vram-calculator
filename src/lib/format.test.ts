import { describe, expect, it } from 'vitest'
import { formatGb, formatTokens } from './format'

describe('formatGb', () => {
  it('uses 2 decimals below 10', () => expect(formatGb(1.2345)).toBe('1.23'))
  it('uses 1 decimal from 10 to 100', () => expect(formatGb(14.901)).toBe('14.9'))
  it('uses 0 decimals at 100 and above', () => expect(formatGb(149.3)).toBe('149'))
  it('handles zero', () => expect(formatGb(0)).toBe('0.00'))
})

describe('formatTokens', () => {
  it('renders K multiples compactly', () => {
    expect(formatTokens(1024)).toBe('1K')
    expect(formatTokens(32768)).toBe('32K')
    expect(formatTokens(131072)).toBe('128K')
  })
  it('renders M multiples compactly', () => expect(formatTokens(1048576)).toBe('1M'))
  it('falls back to locale formatting for non-multiples', () =>
    expect(formatTokens(12000)).toBe('12,000'))
})
