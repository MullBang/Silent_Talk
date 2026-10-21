// recordFlow.js — 녹화→분석 화면(RecordPage)의 순수 함수 (node --test로 검사, JSX·DOM 없음).
// 서버(POST /api/record)가 실제 길이를 다시 재서 최종 판정한다. 여기 검사는 불필요한 업로드를 막는 1차 검사다.

import { errorMessageFrom } from '../api/pollOutcome.js';

/** 서버 거부 기준(초): 이 길이 이상은 접수하지 않는다 (backend RECORD_MAX_SECONDS). */
export const SERVER_MAX_SECONDS = 30;
/** 서버 최소 길이(초) (backend RECORD_MIN_SECONDS). */
export const SERVER_MIN_SECONDS = 0.5;
/** 클라이언트 최소 녹화 길이(초): 이보다 짧으면 업로드하지 않는다. */
export const MIN_RECORD_SECONDS = 1;
/** 서버 업로드 크기 상한(MB) (backend RECORD_MAX_SIZE_MB). */
export const RECORD_MAX_SIZE_MB = 100;
/** 녹화 시작 전 카운트다운(초). */
export const COUNTDOWN_SECONDS = 3;

/** 녹화 모드. 30초 모드는 실험용이며 서버 상한(30초 미만)을 넘지 않게 29초에서 자동 종료한다. */
export const RECORD_MODES = Object.freeze([
  Object.freeze({ id: 'short', label: '8초', maxSeconds: 8, experimental: false }),
  Object.freeze({ id: 'long', label: '30초 (실험)', maxSeconds: 29, experimental: true }),
]);
export const DEFAULT_MODE_ID = 'short';

/**
 * 모드 id로 모드를 찾는다. 모르는 id이면 기본(8초) 모드.
 * @param {string} id
 * @returns {{id:string,label:string,maxSeconds:number,experimental:boolean}}
 */
export function modeById(id) {
  return RECORD_MODES.find((m) => m.id === id) || RECORD_MODES.find((m) => m.id === DEFAULT_MODE_ID);
}

/**
 * 업로드할 Blob 자체의 검사 (비어 있음/크기 초과).
 * @param {{size?:number}|null|undefined} blob
 * @returns {{ok:true}|{ok:false, reason:'empty'|'too_large', message:string}}
 */
export function validateBlob(blob) {
  const size = typeof blob?.size === 'number' && Number.isFinite(blob.size) ? blob.size : 0;
  if (size <= 0) return { ok: false, reason: 'empty', message: '녹화된 영상이 비어 있습니다. 다시 녹화해 주세요.' };
  if (size > RECORD_MAX_SIZE_MB * 1024 * 1024) {
    return { ok: false, reason: 'too_large', message: `녹화 파일이 너무 큽니다 (${RECORD_MAX_SIZE_MB}MB 이하).` };
  }
  return { ok: true };
}

/**
 * 녹화 결과의 1차 검사.
 * @param {{size?:number}|null|undefined} blob 녹화 Blob.
 * @param {number} durationMs 클라이언트가 잰 녹화 시간(ms).
 * @returns {{ok:true}|{ok:false, reason:'empty'|'too_short'|'too_long'|'too_large', message:string}}
 */
export function validateRecording(blob, durationMs) {
  const blobCheck = validateBlob(blob);
  if (!blobCheck.ok && blobCheck.reason === 'empty') return blobCheck;
  const sec = Number.isFinite(durationMs) ? durationMs / 1000 : 0;
  if (sec < MIN_RECORD_SECONDS) {
    return { ok: false, reason: 'too_short', message: `녹화가 너무 짧습니다. ${MIN_RECORD_SECONDS}초 이상 녹화해 주세요.` };
  }
  if (sec >= SERVER_MAX_SECONDS) {
    return { ok: false, reason: 'too_long', message: `${SERVER_MAX_SECONDS}초 미만으로 녹화해 주세요.` };
  }
  return blobCheck;
}

/**
 * MIME에서 코덱 매개변수를 뗀 기본 형식 ('video/webm;codecs=vp9' → 'video/webm').
 * @param {string|null|undefined} mime
 * @returns {string}
 */
export function baseMimeType(mime) {
  const base = String(mime || '').split(';')[0].trim().toLowerCase();
  return base.startsWith('video/mp4') ? 'video/mp4' : 'video/webm';
}

/**
 * 업로드 파일명 (서버 계약: recording.webm 또는 recording.mp4).
 * @param {string|null|undefined} mime
 * @returns {'recording.webm'|'recording.mp4'}
 */
export function recordingFileName(mime) {
  return baseMimeType(mime) === 'video/mp4' ? 'recording.mp4' : 'recording.webm';
}

const pad2 = (n) => String(n).padStart(2, '0');

/**
 * 시각 → 'HH:MM:SS' (로컬 시각).
 * @param {Date} date
 * @returns {string}
 */
export function clockLabel(date) {
  const d = date instanceof Date && !Number.isNaN(date.getTime()) ? date : new Date();
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

/**
 * 결과 화면 제목용 이름 ('녹화 HH:MM:SS').
 * @param {Date} date 녹화 시작 시각.
 * @returns {string}
 */
export function recordingTitle(date) {
  return `녹화 ${clockLabel(date)}`;
}

/**
 * ms → '0:03.2' 형태 (분:초.십분의일초). 음수·비정상 값은 0으로.
 * @param {number} ms
 * @returns {string}
 */
export function formatElapsed(ms) {
  const tenths = Math.max(0, Math.floor((Number.isFinite(ms) ? ms : 0) / 100));
  const m = Math.floor(tenths / 600);
  const s = Math.floor((tenths % 600) / 10);
  return `${m}:${pad2(s)}.${tenths % 10}`;
}

/**
 * 녹화 진행 상황 (경과/남은 시간, 진행률 0~100).
 * @param {number} elapsedMs
 * @param {number} maxSeconds
 * @returns {{elapsedMs:number, remainingMs:number, percent:number}}
 */
export function recordingProgress(elapsedMs, maxSeconds) {
  const total = Math.max(0, (Number.isFinite(maxSeconds) ? maxSeconds : 0) * 1000);
  const el = Math.min(Math.max(0, Number.isFinite(elapsedMs) ? elapsedMs : 0), total);
  const percent = total > 0 ? Math.round((el / total) * 1000) / 10 : 0;
  return { elapsedMs: el, remainingMs: total - el, percent };
}

/**
 * 결과 화면으로 넘길 payload.
 * @param {{sessionId:string, recordedAt:Date, results?:Array, warnings?:string[], model?:object|null}} args
 * @returns {{sessionId:string, fileName:string, results:Array, warnings:string[], model:object|null, origin:'record'}}
 */
export function buildRecordPayload({ sessionId, recordedAt, results, warnings, model }) {
  return {
    sessionId,
    fileName: recordingTitle(recordedAt),
    results: Array.isArray(results) ? results : [],
    warnings: Array.isArray(warnings) ? warnings : [],
    model: model && typeof model === 'object' ? model : null,
    origin: 'record',
  };
}

/**
 * 녹화 화면 단계 표시 문구.
 * @param {string} phase
 * @returns {string}
 */
export function phaseLabel(phase) {
  return ({
    IDLE: '카메라 준비 전',
    OPENING: '카메라를 여는 중...',
    READY: '녹화 준비 완료',
    COUNTDOWN: '곧 녹화를 시작합니다',
    RECORDING: '녹화 중',
    UPLOADING: '업로드 중...',
    INFERRING: '분석 중...',
    TIMEOUT: '시간 초과',
    ERROR: '오류',
  })[phase] || '';
}

export const RECORD_UPLOAD_ERROR = '녹화 영상을 보내지 못했습니다. 다시 시도해 주세요.';
export const RECORD_UNAVAILABLE_ERROR = '지금은 녹화 분석을 사용할 수 없습니다. 서버의 인식 모델 설정을 확인해 주세요.';
export const RECORD_TOO_LARGE_ERROR = `녹화 파일이 너무 큽니다 (${RECORD_MAX_SIZE_MB}MB 이하).`;
export const RECORD_BUSY_ERROR = '서버가 다른 분석을 처리 중입니다. 잠시 후 다시 시도해 주세요.';

/**
 * POST /api/record 오류를 사용자 문구로 바꾼다.
 * 서버 문자열 detail(400/503의 고정 한국어 문구)이 있으면 그대로, 없으면 상태 코드별 고정 문구.
 * @param {any} err axios 오류 또는 클라이언트 검사 오류(isClientCheck).
 * @returns {string}
 */
export function recordUploadErrorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  const status = err?.response?.status;
  if (status === 503) return RECORD_UNAVAILABLE_ERROR;
  if (status === 413) return RECORD_TOO_LARGE_ERROR;
  if (status === 429) return RECORD_BUSY_ERROR;
  return errorMessageFrom(err, RECORD_UPLOAD_ERROR);
}
