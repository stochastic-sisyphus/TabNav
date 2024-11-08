import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'build',
    rollupOptions: {
      input: {
        popup: 'src/popup.js',
        background: 'background.js'
      },
      output: {
        entryFileNames: '[name].js'
      }
    }
  }
}); 