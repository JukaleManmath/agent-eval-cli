import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import fs from 'fs'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    {
      // Dev-only plugin: serve /reports/* from agenteval/dashboard_dist/reports/
      name: 'serve-reports',
      configureServer(server) {
        server.middlewares.use('/reports', (req, res, next) => {
          const reportsDir = resolve(__dirname, '../agenteval/dashboard_dist/reports')
          const filePath = path.join(reportsDir, req.url === '/' ? '' : req.url)
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            res.setHeader('Content-Type', 'application/json')
            res.end(fs.readFileSync(filePath))
          } else {
            next()
          }
        })
      },
    },
  ],
  build: {
    outDir: '../agenteval/dashboard_dist',
    emptyOutDir: true,
  },
})
