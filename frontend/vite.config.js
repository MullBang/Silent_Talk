// vite.config.js — Vite 빌드/개발 서버 설정.

import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const ROOT = dirname(fileURLToPath(import.meta.url));
const MEDIAPIPE_WASM_DIR = join(ROOT, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const URL_PREFIX = '/mediapipe/';
const WASM_SUBDIR = 'wasm';
const MODEL_FILE = 'face_landmarker.task';
const CONTENT_TYPES = { '.wasm': 'application/wasm', '.js': 'text/javascript; charset=utf-8', '.task': 'application/octet-stream' };

/**
 * node_modules의 MediaPipe wasm 파일 이름 목록 (.js/.wasm만).
 * @returns {string[]}
 */
function listWasmFiles() {
  try {
    return readdirSync(MEDIAPIPE_WASM_DIR).filter((n) => /^[\w.-]+\.(wasm|js)$/.test(n));
  } catch {
    return [];
  }
}

/**
 * 오프라인 시연용 MediaPipe 로컬 제공 플러그인 (6단계, 새 의존성 없음).
 * - 개발/미리보기 서버: /mediapipe/wasm/<파일> 을 node_modules/@mediapipe/tasks-vision/wasm 에서 제공한다.
 *   /mediapipe/face_landmarker.task 는 public/mediapipe 에 사용자가 둔 경우에만 제공하고, 없으면 404
 *   (SPA 대체 index.html 응답을 모델 파일로 착각하지 않게 한다).
 * - 빌드: wasm 파일을 출력 폴더의 mediapipe/wasm 으로 복사한다. 모델 파일은 public 폴더 복사 규칙을 따른다.
 * 목록에 있는 파일 이름만 허용하므로 경로 이동(../) 요청은 404다.
 * @returns {import('vite').Plugin}
 */
function mediapipeLocal() {
  let outDir = null;
  let publicDir = join(ROOT, 'public');
  const middleware = (req, res, next) => {
    try {
      const pathname = decodeURIComponent((req.url || '').split('?')[0]);
      if (!pathname.startsWith(URL_PREFIX)) return next();
      const rest = pathname.slice(URL_PREFIX.length);
      let file = null;
      if (rest.startsWith(`${WASM_SUBDIR}/`)) {
        const name = rest.slice(WASM_SUBDIR.length + 1);
        if (listWasmFiles().includes(name)) file = join(MEDIAPIPE_WASM_DIR, name);
      } else if (rest === MODEL_FILE) {
        const candidate = join(publicDir, 'mediapipe', MODEL_FILE);
        if (existsSync(candidate) && statSync(candidate).isFile()) file = candidate;
      }
      if (!file) {
        res.statusCode = 404;
        res.end('Not Found');
        return undefined;
      }
      const ext = file.slice(file.lastIndexOf('.'));
      res.setHeader('Content-Type', CONTENT_TYPES[ext] || 'application/octet-stream');
      res.setHeader('Cache-Control', 'no-cache');
      if (req.method === 'HEAD') {
        res.setHeader('Content-Length', String(statSync(file).size));
        res.end();
        return undefined;
      }
      res.end(readFileSync(file));
      return undefined;
    } catch {
      res.statusCode = 500;
      res.end('Internal Error');
      return undefined;
    }
  };
  return {
    name: 'mediapipe-local',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
      if (config.publicDir) publicDir = config.publicDir;
    },
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
    writeBundle() {
      const files = listWasmFiles();
      if (!files.length) {
        this.error('MediaPipe wasm 파일을 찾지 못했습니다: node_modules/@mediapipe/tasks-vision/wasm');
      }
      const target = join(outDir, 'mediapipe', WASM_SUBDIR);
      mkdirSync(target, { recursive: true });
      for (const name of files) copyFileSync(join(MEDIAPIPE_WASM_DIR, name), join(target, name));
    },
  };
}

// 개발 서버 포트 5173은 backend CORS_ORIGINS 화이트리스트와 일치해야 한다.
export default defineConfig({
  plugins: [react(), mediapipeLocal()],
  server: {
    port: 5173,
  },
});
