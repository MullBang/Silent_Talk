// vite.capture.config.js — 단독 촬영 화면(capture.html)만 빌드하는 설정.
// 파일로 열어도 동작하도록 상대 경로·CSS 한 파일·모듈 미리 읽기 끔으로 빌드하고,
// scripts/build-capture-package.mjs가 결과 JS·CSS를 HTML 안에 넣어 파일 하나로 만든다.

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: 'dist-capture/build',
    emptyOutDir: true,
    cssCodeSplit: false,
    modulePreload: false,
    assetsInlineLimit: 100000000,
    rollupOptions: { input: 'capture.html' },
  },
});
