import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendUrl = env.VITE_BACKEND_URL

  return {
    plugins: [react()],
    server: {
      proxy: backendUrl
        ? {
            '/api': {
              target: backendUrl,
              changeOrigin: true,
              rewrite: (path) => path.replace(/^\/api/, ''),
              headers: {
                'ngrok-skip-browser-warning': 'true',
              },
            },
          }
        : {},
    },
  }
})
