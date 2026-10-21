// cameraErrors.js — 카메라 열기·녹화 오류를 고정 한국어 문구로 바꾸는 순수 함수 (useRecorder 공용).
// 브라우저의 영어 원문(e.message)은 화면에 보여 주지 않고 콘솔 기록에만 남긴다.

export const CAMERA_PERMISSION_ERROR = '웹캠 접근 권한이 필요합니다.';
export const CAMERA_NOT_FOUND_ERROR = '카메라를 찾을 수 없습니다.';
export const CAMERA_BUSY_ERROR = '다른 프로그램이 카메라를 사용 중입니다.';
export const CAMERA_LOST_ERROR = '카메라 연결이 끊겼습니다. 카메라를 다시 열어 주세요.';
export const RECORD_START_ERROR = '녹화를 시작하지 못했습니다. 카메라를 다시 열어 주세요.';
export const RECORDER_UNSUPPORTED_ERROR = '이 브라우저는 MediaRecorder 영상 녹화를 지원하지 않습니다.';
export const INSECURE_CONTEXT_ERROR = 'HTTPS 또는 localhost 환경에서만 카메라를 쓸 수 있습니다.';
export const AUDIO_TRACK_ERROR = '오디오 트랙이 열렸습니다 (음성 미사용 규칙 위반).';

/**
 * 앱이 직접 판정한 카메라 오류(고정 한국어 문구)를 만든다. 매핑에서 문구를 그대로 쓴다.
 * @param {string} message 고정 한국어 문구.
 * @returns {Error & {isClientCheck: true}}
 */
export function clientCameraError(message) {
  const err = new Error(message);
  err.isClientCheck = true;
  return err;
}

/**
 * getUserMedia 등 카메라 열기 오류를 화면 문구로 바꾼다.
 * 알 수 없는 오류는 null을 돌려 ErrorModal의 DEVICE_ERROR 기본 한국어 문구가 쓰이게 한다.
 * @param {any} err DOMException 또는 Error.
 * @returns {string|null}
 */
export function cameraOpenErrorMessage(err) {
  if (err?.isClientCheck && typeof err.message === 'string' && err.message) return err.message;
  switch (err?.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return CAMERA_PERMISSION_ERROR;
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return CAMERA_NOT_FOUND_ERROR;
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return CAMERA_BUSY_ERROR;
    default:
      return null;
  }
}

/**
 * 카메라 오류의 모달 유형. 권한 거부만 PERMISSION_DENIED, 나머지는 DEVICE_ERROR.
 * @param {any} err
 * @returns {'PERMISSION_DENIED'|'DEVICE_ERROR'}
 */
export function cameraErrorType(err) {
  return cameraOpenErrorMessage(err) === CAMERA_PERMISSION_ERROR ? 'PERMISSION_DENIED' : 'DEVICE_ERROR';
}

/**
 * 스트림을 더 이상 녹화에 쓸 수 없는지 (모든 영상 트랙 종료 또는 stream.active=false).
 * @param {{active?: boolean, getVideoTracks?: () => Array<{readyState?: string}>}|null|undefined} stream
 * @returns {boolean}
 */
export function isStreamLost(stream) {
  if (!stream) return true;
  if (stream.active === false) return true;
  const tracks = typeof stream.getVideoTracks === 'function' ? stream.getVideoTracks() : [];
  return tracks.length === 0 || tracks.every((t) => t?.readyState === 'ended');
}

/**
 * 오류 원문을 콘솔에만 남긴다 (화면에는 고정 문구만 표시).
 * @param {string} where 발생 위치 설명.
 * @param {any} err
 * @param {{warn: (...args: any[]) => void}} [logger=console]
 * @returns {void}
 */
export function logCameraError(where, err, logger = console) {
  try {
    logger.warn(`[camera] ${where}`, err?.name, err?.message);
  } catch {
    /* 콘솔 기록 실패는 무시 */
  }
}
