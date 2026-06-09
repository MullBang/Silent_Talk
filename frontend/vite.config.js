// vite.config.js — Vite 빌드/개발 서버 설정.

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 개발 서버 포트 5173은 backend CORS_ORIGINS 화이트리스트와 일치해야 한다.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
