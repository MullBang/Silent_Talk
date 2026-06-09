// constants.js — CLAUDE.md / 설계문서 기준값 미러 (프론트엔드용).
// 백엔드 config.py와 동일 값을 유지해야 한다 (전처리 불일치 방지).

export const ROI_SIZE = 96;
export const ROI_BYTES = ROI_SIZE * ROI_SIZE * 3; // 27,648 (Uint8 RGB)
export const ROI_MARGIN = 0.2;

// MediaPipe Face Mesh 랜드마크 인덱스
export const ROI_CROP_IDX = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];
export const VVAD_IDX_TOP = [13, 14, 312, 311, 310, 415];
export const VVAD_IDX_BOT = [17, 18, 84, 181, 180, 314];
export const FACE_WIDTH_IDX = [234, 454]; // 좌우 광대 (얼굴 폭 정규화)

// V-VAD
export const EMA_ALPHA = 0.3;
export const CALIBRATION_MS = 1500;
export const DEFAULT_THRESHOLDS = { tOpen: 0.08, tClose: 0.04, tVar: 0.0003 };
export const VAR_WINDOW = 10; // 분산 계산용 롤링 윈도우
export const CLOSE_HOLD_FRAMES = 12; // 0.5초(25fps) 연속 닫힘 → 발화 종료
export const MIN_SPEECH_FRAMES = 12; // 최소 발화 길이 0.5초
export const MISS_WARN_FRAMES = 3; // 3연속 미검출 경고

// 슬라이딩 윈도우 / WebSocket
export const CHUNK_FRAMES = 11; // 10~12프레임마다 청크 전송
export const HEADER_BYTES = 8;
export const MSG_TYPE_FRAME_DATA = 0x01;
export const SUBTITLE_MAX = 5;
export const MAX_RECONNECT = 3;
export const RECONNECT_DELAY_MS = 1000;
export const WS_URL =
  import.meta.env?.VITE_WS_URL || 'ws://localhost:8000/ws/stream';

// MediaPipe 모델 (CDN)
export const MEDIAPIPE_WASM =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
export const FACE_LANDMARKER_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
