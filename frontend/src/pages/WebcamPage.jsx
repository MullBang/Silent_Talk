// WebcamPage.jsx — SCR-03 웹캠 실시간 화면 (2차 목표).
// MediaPipe JS 랜드마크 → V-VAD → 슬라이딩 윈도우 ROI를 WebSocket으로 스트리밍.
// Tailwind CSS만 사용.

import React, { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

import ErrorModal from '../components/ErrorModal';
import {
  CALIBRATION_MS,
  CHUNK_FRAMES,
  CLOSE_HOLD_FRAMES,
  DEFAULT_THRESHOLDS,
  EMA_ALPHA,
  FACE_LANDMARKER_MODEL,
  MAX_RECONNECT,
  MEDIAPIPE_WASM,
  MIN_SPEECH_FRAMES,
  MISS_WARN_FRAMES,
  RECONNECT_DELAY_MS,
  SUBTITLE_MAX,
  VAR_WINDOW,
  WS_URL,
} from '../constants';
import {
  buildBinaryFrame,
  computeMouthRatio,
  computeRoiBBox,
  extractRoiBytes,
  mean,
  std,
  variance,
} from '../webcamUtils';

// 상태: IDLE → PERMISSION_REQUEST → CALIBRATING → ACTIVE → LISTENING → STOPPED
const PHASE = {
  IDLE: 'IDLE',
  PERMISSION_REQUEST: 'PERMISSION_REQUEST',
  CALIBRATING: 'CALIBRATING',
  ACTIVE: 'ACTIVE',
  LISTENING: 'LISTENING',
  STOPPED: 'STOPPED',
};

/**
 * 웹캠 실시간 페이지 (SCR-03).
 * @param {object} props
 * @param {() => void} [props.onBack] - 업로드 모드로 돌아가기.
 * @returns {JSX.Element}
 */
export default function WebcamPage({ onBack }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // ROI 크롭용 (숨김)

  const [phase, setPhase] = useState(PHASE.IDLE);
  const [calibCountdown, setCalibCountdown] = useState(0);
  const [calibWarning, setCalibWarning] = useState(false);
  const [guidelineOk, setGuidelineOk] = useState(false);
  const [vvadActive, setVvadActive] = useState(false);
  const [missWarning, setMissWarning] = useState(false);
  const [subtitles, setSubtitles] = useState([]); // [{text, final}]
  const [modal, setModal] = useState(null); // {type, message}

  // 가변 런타임 상태 (리렌더 불필요)
  const rt = useRef({
    sessionId: null,
    landmarker: null,
    stream: null,
    ws: null,
    rafId: null,
    running: false,
    reconnects: 0,
    intentionalClose: false,
    // V-VAD
    thresholds: { ...DEFAULT_THRESHOLDS },
    calibSamples: [],
    calibStart: 0,
    ema: 0,
    ratioWindow: [],
    speaking: false,
    closeCounter: 0,
    speechFrames: 0,
    missCounter: 0,
    // 청크
    chunkId: 0,
    chunkFrames: [], // Uint8Array[]
  });

  // ── WebSocket ──────────────────────────────────────────────────────────
  const sendControl = (obj) => {
    const ws = rt.current.ws;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  };

  const connectWs = () => {
    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    rt.current.ws = ws;

    ws.onopen = () => {
      rt.current.reconnects = 0;
      sendControl({
        type: 'SESSION_START',
        session_id: rt.current.sessionId,
        client_ts: Date.now(),
      });
    };

    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === 'PARTIAL') {
        applyPartial(msg.text, msg.is_final);
      } else if (msg.type === 'ERROR') {
        // 검출 실패 등은 경고만 (스트림 유지)
        setMissWarning(true);
      }
    };

    ws.onclose = () => {
      if (rt.current.intentionalClose) return;
      // 비정상 종료 → 최대 3회 재연결
      if (rt.current.reconnects < MAX_RECONNECT && rt.current.running) {
        rt.current.reconnects += 1;
        setTimeout(connectWs, RECONNECT_DELAY_MS);
      } else if (rt.current.running) {
        setModal({ type: 'DEVICE_ERROR', message: 'WebSocket 연결이 끊겼습니다.' });
      }
    };

    ws.onerror = () => {
      try {
        ws.close();
      } catch {
        /* noop */
      }
    };
  };

  const applyPartial = (text, isFinal) => {
    setSubtitles((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last && !last.final) {
        // 진행 중(회색) 라인 갱신
        next[next.length - 1] = { text, final: !!isFinal };
      } else {
        next.push({ text, final: !!isFinal });
      }
      // 확정되면 다음 라인을 위해 그대로 두고, 최근 SUBTITLE_MAX개만 유지
      return next.slice(-SUBTITLE_MAX);
    });
  };

  // ── 청크 전송 ──────────────────────────────────────────────────────────
  const flushChunk = (isFinal) => {
    const s = rt.current;
    if (s.chunkFrames.length === 0 && !isFinal) return;
    s.chunkId += 1;
    sendControl({
      type: 'CHUNK_START',
      chunk_id: s.chunkId,
      window_start_ms: 0,
      window_end_ms: 0,
      is_final: !!isFinal,
    });
    const ws = s.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      for (const roi of s.chunkFrames) {
        ws.send(buildBinaryFrame(s.chunkId, roi));
      }
    }
    s.chunkFrames = [];
  };

  // ── 프레임 루프 ────────────────────────────────────────────────────────
  const loop = () => {
    const s = rt.current;
    if (!s.running) return;
    const video = videoRef.current;
    const landmarker = s.landmarker;
    if (video && landmarker && video.readyState >= 2) {
      const result = landmarker.detectForVideo(video, performance.now());
      const faces = result.faceLandmarks;
      if (!faces || faces.length === 0) {
        s.missCounter += 1;
        setGuidelineOk(false);
        if (s.missCounter >= MISS_WARN_FRAMES) setMissWarning(true);
      } else {
        s.missCounter = 0;
        setMissWarning(false);
        setGuidelineOk(true);
        handleLandmarks(faces[0]);
      }
    }
    s.rafId = requestAnimationFrame(loop);
  };

  const handleLandmarks = (landmarks) => {
    const s = rt.current;
    const ratio = computeMouthRatio(landmarks);

    // 캘리브레이션 단계
    if (phase === PHASE.CALIBRATING || s.calibrating) {
      s.calibSamples.push(ratio);
      const elapsed = performance.now() - s.calibStart;
      setCalibCountdown(Math.max(0, Math.ceil((CALIBRATION_MS - elapsed) / 1000)));
      if (elapsed >= CALIBRATION_MS) finishCalibration();
      return;
    }

    // EMA smoothing
    s.ema = EMA_ALPHA * ratio + (1 - EMA_ALPHA) * s.ema;
    s.ratioWindow.push(s.ema);
    if (s.ratioWindow.length > VAR_WINDOW) s.ratioWindow.shift();
    const varS = variance(s.ratioWindow);

    const { tOpen, tClose, tVar } = s.thresholds;

    if (!s.speaking) {
      // 발화 시작 감지
      if (s.ema > tOpen && varS > tVar) {
        s.speaking = true;
        s.speechFrames = 0;
        s.closeCounter = 0;
        setVvadActive(true);
        setPhase(PHASE.LISTENING);
      }
    } else {
      // 발화 중: ROI 누적 + 슬라이딩 전송
      const roi = cropCurrentRoi(landmarks);
      if (roi) s.chunkFrames.push(roi);
      s.speechFrames += 1;
      if (s.chunkFrames.length >= CHUNK_FRAMES) flushChunk(false);

      // 발화 종료 감지
      if (s.ema < tClose) s.closeCounter += 1;
      else s.closeCounter = 0;
      if (s.closeCounter >= CLOSE_HOLD_FRAMES) {
        if (s.speechFrames >= MIN_SPEECH_FRAMES) flushChunk(true);
        else s.chunkFrames = []; // 너무 짧은 발화 무시
        s.speaking = false;
        setVvadActive(false);
        setPhase(PHASE.ACTIVE);
      }
    }
  };

  const cropCurrentRoi = (landmarks) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;
    const bbox = computeRoiBBox(landmarks, video.videoWidth, video.videoHeight);
    try {
      return extractRoiBytes(video, bbox, canvas);
    } catch {
      return null;
    }
  };

  const finishCalibration = () => {
    const s = rt.current;
    s.calibrating = false;
    const samples = s.calibSamples;
    if (samples.length >= 10) {
      const m = mean(samples);
      const sd = std(samples, m);
      const v = variance(samples, m);
      s.thresholds = {
        tOpen: m + 2 * sd,
        tClose: m + 0.8 * sd,
        tVar: v * 1.5,
      };
      setCalibWarning(false);
    } else {
      s.thresholds = { ...DEFAULT_THRESHOLDS };
      setCalibWarning(true);
    }
    s.ema = mean(samples) || 0;
    setPhase(PHASE.ACTIVE);
  };

  // ── 시작 / 정지 ────────────────────────────────────────────────────────
  const handleStart = async () => {
    // ① HTTPS 보안 컨텍스트 검사
    if (!window.isSecureContext) {
      setModal({ type: 'HTTPS_ERROR' });
      return;
    }
    setPhase(PHASE.PERMISSION_REQUEST);

    // ② 카메라 권한
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (err) {
      if (err && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
        setModal({ type: 'PERMISSION_DENIED' });
      } else {
        setModal({ type: 'DEVICE_ERROR' });
      }
      setPhase(PHASE.IDLE);
      return;
    }
    rt.current.stream = stream;

    // ③ 비디오 연결
    const video = videoRef.current;
    video.srcObject = stream;
    await video.play().catch(() => {});

    // ④ MediaPipe FaceLandmarker 초기화
    try {
      const resolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      rt.current.landmarker = await FaceLandmarker.createFromOptions(resolver, {
        baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
      });
    } catch (err) {
      setModal({ type: 'DEVICE_ERROR', message: '랜드마크 모델 로드 실패' });
      stopAll();
      return;
    }

    // 캘리브레이션 시작
    rt.current.sessionId =
      (crypto.randomUUID && crypto.randomUUID()) || `sess-${Date.now()}`;
    rt.current.calibSamples = [];
    rt.current.calibStart = performance.now();
    rt.current.calibrating = true;
    setCalibCountdown(Math.ceil(CALIBRATION_MS / 1000));
    setPhase(PHASE.CALIBRATING);

    // ⑤ WebSocket 연결 + SESSION_START
    connectWs();

    // ⑥ 루프 시작
    rt.current.running = true;
    rt.current.rafId = requestAnimationFrame(loop);
  };

  const stopAll = () => {
    const s = rt.current;
    s.running = false;
    s.intentionalClose = true;
    if (s.rafId) cancelAnimationFrame(s.rafId);
    if (s.ws) {
      try {
        sendControl({ type: 'SESSION_END' });
        s.ws.close();
      } catch {
        /* noop */
      }
    }
    if (s.stream) s.stream.getTracks().forEach((t) => t.stop());
    if (s.landmarker) {
      try {
        s.landmarker.close();
      } catch {
        /* noop */
      }
    }
    s.ws = null;
    s.landmarker = null;
    s.stream = null;
    s.speaking = false;
    setVvadActive(false);
  };

  const handleStop = () => {
    stopAll();
    setPhase(PHASE.STOPPED);
  };

  const handleClearSubtitles = () => setSubtitles([]);

  // 언마운트 정리
  useEffect(() => () => stopAll(), []); // eslint-disable-line react-hooks/exhaustive-deps

  const started = phase !== PHASE.IDLE && phase !== PHASE.STOPPED;

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      {/* 헤더 */}
      <header className="mb-3 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">🎤 립리딩 텍스트 변환</h1>
        <button
          type="button"
          onClick={onBack}
          aria-label="영상 업로드 모드로 전환"
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
        >
          영상 업로드 모드 →
        </button>
      </header>
      <p className="mb-3 text-xs text-amber-600">
        ⚠ Chrome/Edge 권장 · HTTPS 또는 localhost 필요
      </p>

      {/* 캘리브레이션 배너 */}
      {phase === PHASE.CALIBRATING && (
        <div className="mb-3 rounded-lg bg-blue-50 px-3 py-2 text-sm text-blue-700">
          캘리브레이션 중... 잠시 가만히 계세요 ({calibCountdown}초)
        </div>
      )}
      {phase !== PHASE.CALIBRATING && started && !calibWarning && (
        <div className="mb-3 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700">
          캘리브레이션 완료 ✓
        </div>
      )}
      {calibWarning && (
        <div className="mb-3 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">
          캘리브레이션 실패 — 기본값으로 진행합니다.
        </div>
      )}

      {/* 비디오 + 오버레이 */}
      <div className="relative overflow-hidden rounded-2xl bg-black">
        <video
          ref={videoRef}
          playsInline
          muted
          className="h-auto w-full"
          style={{ transform: 'scaleX(-1)' }} // Mirror
        />
        {/* SVG 가이드라인 오버레이 */}
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          <ellipse
            cx="50"
            cy="50"
            rx="22"
            ry="30"
            fill="none"
            strokeWidth="0.8"
            strokeDasharray="2 2"
            stroke={guidelineOk ? '#22c55e' : '#ef4444'}
          />
        </svg>
        {/* V-VAD 인디케이터 */}
        <div className="absolute right-3 top-3 flex items-center gap-2 rounded-full bg-black/40 px-2 py-1">
          <span
            className={`inline-block h-3 w-3 rounded-full ${
              vvadActive ? 'animate-pulse bg-green-400' : 'bg-gray-400'
            }`}
          />
          <span className="text-xs text-white">
            {vvadActive ? '발화 인식 중' : '대기'}
          </span>
        </div>
        {/* 숨김 캔버스 (ROI 크롭) */}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {missWarning && (
        <p className="mt-2 text-sm text-red-600" role="alert">
          입술을 가이드라인 안에 맞춰주세요.
        </p>
      )}

      {/* 자막 영역 (최근 5개) */}
      <div className="mt-4 min-h-[6rem] rounded-2xl bg-gray-900 p-4">
        {subtitles.length === 0 ? (
          <p className="text-sm text-gray-500">자막이 여기에 표시됩니다.</p>
        ) : (
          subtitles.map((line, i) => (
            <p
              key={i}
              className={`text-base ${line.final ? 'text-white' : 'italic text-gray-400'}`}
            >
              {line.text}
            </p>
          ))
        )}
      </div>

      {/* 컨트롤 */}
      <div className="mt-4 flex items-center gap-2">
        {!started ? (
          <button
            type="button"
            onClick={handleStart}
            aria-label="시작"
            className="rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700"
          >
            ▶ 시작
          </button>
        ) : (
          <button
            type="button"
            onClick={handleStop}
            aria-label="정지"
            className="rounded-xl bg-red-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-red-700"
          >
            ■ 정지
          </button>
        )}
        <button
          type="button"
          onClick={handleClearSubtitles}
          aria-label="자막 초기화"
          className="rounded-xl border border-gray-300 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50"
        >
          자막 초기화
        </button>
      </div>

      {/* 개인정보 안내 (항상 표시) */}
      <p className="mt-4 text-xs text-gray-500">
        ℹ 영상은 서버에 저장되지 않습니다.
      </p>

      {/* 에러 모달 */}
      {modal && (
        <ErrorModal
          type={modal.type}
          message={modal.message}
          onRetry={() => {
            setModal(null);
            handleStart();
          }}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  );
}
