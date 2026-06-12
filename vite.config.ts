/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Production build is served from a GitHub Pages project site
// (https://changh95.github.io/vram-calculator/), so assets need that base path.
// Dev/test stay at root.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/vram-calculator/' : '/',
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
  },
}))
