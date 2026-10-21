// pollOutcome.js — GET /api/result 폴링 응답 해석·오류 문구 순수 함수 (UploadPage·RecordPage 공용).

// 폴링 전체 타임아웃 (설계 5분)
// 서버 Job 타임아웃(JOB_TIMEOUT_SEC=300초)보다 조금 길게 잡아, 서버가 판정한 'timeout' 상태를 받아 본다.
export const POLL_TIMEOUT_MS = 315000;
export const POLL_INTERVAL_MS = 1000;
// 추론 종료(터미널) 상태. 백엔드: pending→processing→done/error/timeout.
// (명세의 status!=='processing'은 초기 pending에서 조기 종료되므로 터미널 기준으로 보정)
export const TERMINAL_STATUSES = ['done', 'error', 'timeout'];

export const DEFAULT_INFER_ERROR = '추론 중 오류가 발생했습니다.';
export const NETWORK_ERROR = '서버에 연결할 수 없습니다. 백엔드 서버가 켜져 있는지 확인해 주세요.';
export const REQUEST_TIMEOUT_ERROR = '서버 응답 시간이 초과되었습니다. 다시 시도해 주세요.';

/**
 * 폴링 응답이 종료 상태인지.
 * @param {{status?: string}|null|undefined} data
 * @returns {boolean}
 */
export function isTerminal(data) {
  return TERMINAL_STATUSES.includes(data?.status);
}

/**
 * 폴링 응답 1건을 화면 동작으로 해석한다.
 * @param {object|null|undefined} data GET /api/result 응답.
 * @returns {{kind:'none'}|{kind:'progress', progress:number|null}
 *   |{kind:'done', progress:number|null, results:Array, warnings:string[], model:object|null}
 *   |{kind:'timeout', progress:number|null}|{kind:'error', progress:number|null, message:string}}
 */
export function interpretPoll(data) {
  if (!data || typeof data !== 'object') return { kind: 'none' };
  const progress = typeof data.progress === 'number' && Number.isFinite(data.progress) ? data.progress : null;
  if (data.status === 'done') {
    return {
      kind: 'done',
      progress,
      results: Array.isArray(data.results) ? data.results : [],
      warnings: Array.isArray(data.warnings) ? data.warnings : [],
      model: data.model && typeof data.model === 'object' ? data.model : null,
    };
  }
  if (data.status === 'timeout') return { kind: 'timeout', progress };
  if (data.status === 'error') {
    const message = typeof data.error_message === 'string' && data.error_message.trim()
      ? data.error_message : DEFAULT_INFER_ERROR;
    return { kind: 'error', progress, message };
  }
  return { kind: 'progress', progress };
}

/**
 * axios 오류를 사용자 문구로 바꾼다.
 * 서버가 문자열 detail(400/503 등 고정 한국어 문구)을 주면 그대로 쓰고,
 * 응답이 없으면 연결 오류 문구, 그 밖에는 fallback을 쓴다.
 * FastAPI 검증 오류처럼 detail이 배열/객체이면 내부 내용을 노출하지 않는다.
 * @param {any} err
 * @param {string} fallback
 * @returns {string}
 */
export function errorMessageFrom(err, fallback) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (err?.response) return fallback;
  if (err?.code === 'ECONNABORTED') return REQUEST_TIMEOUT_ERROR;
  if (err?.request || err?.code === 'ERR_NETWORK' || err?.message === 'Network Error') return NETWORK_ERROR;
  if (err?.isClientCheck && typeof err.message === 'string') return err.message;
  return fallback;
}

// 업로드 직후 첫 폴링이 Job 생성보다 먼저 도착하면 서버가 404를 줄 수 있다(방어용 유예).
// 처음 NOT_FOUND_GRACE_POLLS번의 폴링에서 받은 404만 '대기 중'으로 본다.
export const NOT_FOUND_GRACE_POLLS = 2;
export const POLL_ERROR = '분석 결과를 확인하지 못했습니다. 다시 시도해 주세요.';
export const START_ERROR = '업로드/추론 시작에 실패했습니다.';

/**
 * 폴링 오류가 초기 유예 대상(처음 몇 번의 404)인지.
 * @param {any} err axios 오류.
 * @param {number} attempt 0부터 센 폴링 순번.
 * @param {number} [grace=NOT_FOUND_GRACE_POLLS]
 * @returns {boolean}
 */
export function isGraceNotFound(err, attempt, grace = NOT_FOUND_GRACE_POLLS) {
  return err?.response?.status === 404 && Number.isInteger(attempt) && attempt >= 0 && attempt < grace;
}

/**
 * 결과 조회 함수에 초기 404 유예를 씌운다. 유예된 404는 {status:'pending', progress:null}로 바꾼다.
 * 호출 순번은 요청을 보낸 순서로 센다 (응답 순서와 무관).
 * @param {() => Promise<any>} fetchFn GET /api/result 호출.
 * @param {number} [grace=NOT_FOUND_GRACE_POLLS]
 * @returns {() => Promise<any>}
 */
export function withNotFoundGrace(fetchFn, grace = NOT_FOUND_GRACE_POLLS) {
  let attempts = 0;
  return async () => {
    const attempt = attempts;
    attempts += 1;
    try {
      return await fetchFn();
    } catch (err) {
      if (isGraceNotFound(err, attempt, grace)) return { status: 'pending', progress: null };
      throw err;
    }
  };
}
