import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { SelectField } from './SelectField'

const groups = [
  {
    label: 'Qwen3',
    options: [
      { value: 'qwen3-8b', label: 'Qwen3-8B', meta: '8B' },
      { value: 'qwen3-32b', label: 'Qwen3-32B', meta: '32B' },
    ],
  },
  { label: 'DeepSeek', options: [{ value: 'deepseek-v3', label: 'DeepSeek-V3', meta: '671B' }] },
]

describe('SelectField', () => {
  it('renders a labelled select with grouped options', () => {
    render(<SelectField label="Model" groups={groups} value="qwen3-8b" onChange={() => {}} />)
    const select = screen.getByLabelText('Model')
    expect(select).toHaveValue('qwen3-8b')
    expect(screen.getByRole('group', { name: 'Qwen3' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /DeepSeek-V3/ })).toBeInTheDocument()
  })

  it('emits the selected value', async () => {
    const onChange = vi.fn()
    render(<SelectField label="Model" groups={groups} value="qwen3-8b" onChange={onChange} />)
    await userEvent.selectOptions(screen.getByLabelText('Model'), 'deepseek-v3')
    expect(onChange).toHaveBeenCalledWith('deepseek-v3')
  })
})
