// mediapipeAssets.js — MediaPipe 웹 파일 위치 결정 (오프라인 시연 대비, 6단계).
// wasm 파일은 vite.config.js의 mediapipe-local 플러그인이 node_modules/@mediapipe/tasks-vision/wasm 을
// /mediapipe/wasm 으로 제공(개발 서버)하고 빌드 출력에 복사한다.
// face_landmarker.task 모델은 node_modules에 없고 내려받지 않는다. 사용자가 frontend/public/mediapipe/ 에
// 두었을 때만 로컬 파일을 쓰고, 없으면 기존 CDN 주소로 돌아간다 (WebcamPage 전용).

/** 로컬 얼굴 랜드마크 모델 경로. */
export const LOCAL_FACE_LANDMARKER_PATH = 'mediapipe/face_landmarker.task';

/**
 * 앱 기준 경로(Vite BASE_URL)에 상대 경로를 붙인다.
 * @param {string} rel 앞에 '/' 없는 상대 경로.
 * @param {string} [base] 기준 경로. 기본은 import.meta.env.BASE_URL 또는 '/'.
 * @returns {string}
 */
export function withBase(rel, base) {
  const b = typeof base === 'string' && base ? base : (import.meta.env?.BASE_URL || '/');
  return `${b.endsWith('/') ? b : `${b}/`}${rel.replace(/^\/+/, '')}`;
}

/**
 * 로컬 모델 파일이 실제로 제공되는지 확인하고 사용할 모델 주소를 고른다.
 * 개발 서버/정적 서버의 SPA 대체 응답(index.html, 200)을 모델로 착각하지 않도록 HTML 응답은 거부한다.
 * @param {string} cdnUrl 로컬 파일이 없을 때 쓸 주소.
 * @param {{fetchFn?: typeof fetch, base?: string}} [opts]
 * @returns {Promise<{url: string, source: 'local'|'cdn'}>}
 */
export async function resolveFaceLandmarkerModel(cdnUrl, { fetchFn, base } = {}) {
  const local = withBase(LOCAL_FACE_LANDMARKER_PATH, base);
  const f = fetchFn || globalThis.fetch;
  if (typeof f !== 'function') return { url: cdnUrl, source: 'cdn' };
  try {
    const res = await f(local, { method: 'HEAD', cache: 'no-store' });
    const type = String(res?.headers?.get?.('content-type') || '').toLowerCase();
    if (res?.ok && !type.includes('text/html')) return { url: local, source: 'local' };
  } catch {
    // 로컬 확인 실패 → CDN 사용
  }
  return { url: cdnUrl, source: 'cdn' };
}
