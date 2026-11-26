import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Development frontend and backend share one browser origin.
 * /api requests are proxied to Fastify so browser CORS/localhost mismatches
 * cannot cause `Failed to fetch`.  strictPort prevents Vite silently using 5174
 * while an older AniMatch frontend stays open on 5173.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
        configure(proxy) {
          proxy.on('error', (error) => {
            console.error('[AniMatch proxy] API on 127.0.0.1:3001 unreachable:', error.message);
          });
        },
      },
    },
  },
});
