/** Display formatting for memory sizes (GiB shown as "GB") and token counts. */

export function formatGb(gb: number): string {
  if (gb >= 100) return gb.toFixed(0)
  if (gb >= 10) return gb.toFixed(1)
  return gb.toFixed(2)
}

export function formatTokens(tokens: number): string {
  if (tokens >= 1048576 && tokens % 1048576 === 0) return `${tokens / 1048576}M`
  if (tokens % 1024 === 0) return `${tokens / 1024}K`
  return tokens.toLocaleString('en-US')
}
