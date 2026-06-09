// client.js — axios 인스턴스 + API 호출 함수.

import axios from 'axios';

/** 공용 axios 인스턴스 (baseURL/타임아웃/인터셉터 구성 예정). */
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000',
  timeout: 30000,
});

/**
 * 영상 파일을 업로드한다. (POST /api/upload)
 * @param {File} file
 * @returns {Promise<{ session_id: string, filename: string, status: string }>}
 */
export async function uploadVideo(file) {
  // TODO: multipart/form-data 업로드
}

/**
 * 추론 Job을 등록한다. (POST /api/infer)
 * 즉시 결과를 반환하지 않고 {job_id, session_id, status:'processing'}을 받는다.
 * @param {string} sessionId
 * @returns {Promise<{ job_id: string, session_id: string, status: string }>}
 */
export async function startInference(sessionId) {
  // TODO: 추론 등록 호출
}

/**
 * 추론 결과를 폴링한다. (GET /api/result/{session_id})
 * @param {string} sessionId
 * @returns {Promise<object>}
 */
export async function fetchResult(sessionId) {
  // TODO: 결과 조회 호출
}
