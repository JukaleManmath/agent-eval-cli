import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../agenteval/dashboard_dist',
    emptyOutDir: true,
  },
})
