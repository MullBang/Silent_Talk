// client.js — axios 인스턴스 + API 호출 함수.
// 백엔드(FastAPI) baseURL: http://localhost:8000

import axios from 'axios';

import { baseMimeType, recordingFileName, validateBlob } from '../record/recordFlow';

/** 서비스 제한값 (설계문서 F-MVP-01). 클라이언트 1차 검증용. */
export const MAX_UPLOAD_SIZE_MB = 500;
export const MAX_DURATION_SEC = 180; // 3분

/** 공용 axios 인스턴스. */
export const apiClient = axios.create({
  baseURL: import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 600000, // 대용량 업로드 대비 10분
});

/**
 * 클라이언트 1차 검증 오류(고정 한국어 문구)를 만든다. errorMessageFrom이 문구를 그대로 쓴다.
 * @param {string} message
 * @returns {Error & {isClientCheck: true}}
 */
function clientCheckError(message) {
  const err = new Error(message);
  err.isClientCheck = true;
  return err;
}

/**
 * File 객체의 재생 길이(초)를 메타데이터로 읽는다.
 * @param {File} file
 * @returns {Promise<number>} 재생 시간(초)
 */
function getVideoDurationSec(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(video.duration);
    };
    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('영상 메타데이터를 읽을 수 없습니다.'));
    };
    video.src = url;
  });
}

/**
 * 영상 파일을 업로드한다. (POST /api/upload)
 * 업로드 전 파일 크기/길이를 클라이언트에서 1차 검증한다.
 * @param {File} file
 * @param {(percent: number) => void} [onProgress] - 업로드 진행률(0~100) 콜백.
 * @returns {Promise<{ session_id: string, file_name: string, duration_sec: number }>}
 * @throws {Error} 크기 초과 / 길이 초과 시
 */
export async function uploadVideo(file, onProgress) {
  // ① 크기 검증 (500MB)
  const sizeMb = file.size / 1024 / 1024;
  if (sizeMb > MAX_UPLOAD_SIZE_MB) {
    throw clientCheckError(`${MAX_UPLOAD_SIZE_MB}MB 이하만 업로드 가능합니다.`);
  }

  // ② 길이 검증 (3분)
  let duration;
  try {
    duration = await getVideoDurationSec(file);
  } catch {
    throw clientCheckError('영상 파일을 읽을 수 없습니다. 지원 형식(MP4/AVI/MOV)을 확인해 주세요.');
  }
  if (duration > MAX_DURATION_SEC) {
    throw clientCheckError('3분 이하의 영상만 업로드 가능합니다.');
  }

  // ③ 업로드 (multipart/form-data)
  const form = new FormData();
  form.append('file', file);
  const { data } = await apiClient.post('/api/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (typeof onProgress === 'function' && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
  return data;
}

/**
 * 브라우저 녹화 영상을 올리고 분석 Job을 시작한다. (POST /api/record, 6단계)
 * 서버가 실제 길이를 재서 30초 미만만 접수하고 즉시 결과 대신 {job_id, session_id, status:'processing',
 * duration_sec}를 돌려준다. 결과는 getResult(session_id) 폴링으로 받는다.
 * 파일 파트의 MIME은 코덱 매개변수를 뗀 기본형(video/webm|video/mp4), 파일명은 recording.webm|.mp4다.
 * @param {Blob} blob 녹화 Blob.
 * @param {string} mime MediaRecorder MIME (예: 'video/webm;codecs=vp9').
 * @param {(percent: number) => void} [onProgress] 업로드 진행률(0~100) 콜백.
 * @returns {Promise<{ job_id: string, session_id: string, status: string, duration_sec: number }>}
 * @throws {Error} 빈 Blob(isClientCheck=true) 또는 axios 오류(서버 400/503 detail 포함).
 */
export async function uploadRecording(blob, mime, onProgress) {
  const check = validateBlob(blob);
  if (!check.ok) {
    const err = new Error(check.message);
    err.isClientCheck = true;
    throw err;
  }
  const type = baseMimeType(mime || blob.type);
  const part = blob.type === type ? blob : new Blob([blob], { type });
  const form = new FormData();
  form.append('file', part, recordingFileName(type));
  const { data } = await apiClient.post('/api/record', form, {
    timeout: 120000,
    onUploadProgress: (e) => {
      if (typeof onProgress === 'function' && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
  return data;
}

/**
 * 추론 Job을 시작한다. (POST /api/infer)
 * 즉시 결과를 반환하지 않고 {job_id, session_id, status:'processing'}을 받는다.
 * @param {string} sessionId
 * @returns {Promise<{ job_id: string, session_id: string, status: string }>}
 */
export async function startInfer(sessionId) {
  const { data } = await apiClient.post('/api/infer', { session_id: sessionId });
  return data;
}

/**
 * 추론 결과를 조회한다. (GET /api/result/{sessionId})
 * @param {string} sessionId
 * @returns {Promise<{ session_id: string, status: string, progress: number,
 *   error_message: string|null, results: Array|null }>}
 */
export async function getResult(sessionId) {
  const { data } = await apiClient.get(`/api/result/${sessionId}`);
  return data;
}
