// WebcamPage — 실제 영상 시각으로 리샘플한 입술 ROI를 전송한다. 음성 입력은 사용하지 않는다.
import React, { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import ErrorModal from '../components/ErrorModal';
import { resolveFaceLandmarkerModel } from '../mediapipeAssets';
import {
  ACK_TIMEOUT_MS, CALIBRATION_MS, CHUNK_FRAMES, DEFAULT_THRESHOLDS,
  FACE_LANDMARKER_MODEL, FPS, MAX_UTTERANCE_FRAMES, MEDIAPIPE_WASM,
  MISS_WARN_FRAMES, SUBTITLE_MAX, WS_PROTOCOL_VERSION, WS_URL,
} from '../constants';
import { computeMouthRatio, computeRoiBBox, extractRoiBytes, mean, std, variance } from '../webcamUtils';
import { ChunkTransport, SpeechSegmenter, TimestampResampler } from '../webcamStream';

const PHASE = { IDLE: 'IDLE', PERMISSION_REQUEST: 'PERMISSION_REQUEST', CALIBRATING: 'CALIBRATING',
  ACTIVE: 'ACTIVE', LISTENING: 'LISTENING', STOPPING: 'STOPPING', STOPPED: 'STOPPED' };

export default function WebcamPage({ onBack }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rt = useRef(null);
  const [phase, setPhase] = useState(PHASE.IDLE);
  const [calibCountdown, setCalibCountdown] = useState(0);
  const [calibWarning, setCalibWarning] = useState(false);
  const [guidelineOk, setGuidelineOk] = useState(false);
  const [vvadActive, setVvadActive] = useState(false);
  const [missWarning, setMissWarning] = useState(false);
  const [subtitles, setSubtitles] = useState([]);
  const [modal, setModal] = useState(null);

  const current = (s) => rt.current === s && !s.cancelled;

  const releaseCapture = (s) => {
    s.running = false;
    const video = videoRef.current;
    if (s.callbackId != null) {
      if (s.videoCallback) video?.cancelVideoFrameCallback(s.callbackId);
      else cancelAnimationFrame(s.callbackId);
    }
    s.stream?.getTracks().forEach(track => track.stop());
    s.landmarker?.close();
    s.stream = null;
    s.landmarker = null;
    if (rt.current === s && video) video.srcObject = null;
  };

  const dispose = (s) => {
    if (!s || s.cancelled) return;
    s.cancelled = true;
    releaseCapture(s);
    clearTimeout(s.connectTimer);
    s.rejectConnect?.(new Error('시작이 취소되었습니다.'));
    s.rejectConnect = null;
    s.transport?.dispose();
    if (s.ws) {
      s.ws.onclose = null;
      s.ws.onerror = null;
      s.ws.onmessage = null;
      s.ws.onopen = null;
      s.ws.close();
    }
  };

  const fail = (s, message, type = 'DEVICE_ERROR') => {
    if (!current(s)) return;
    dispose(s);
    setSubtitles(lines => lines.filter(line => line.final || !line.key.startsWith(`${s.sessionId}:`)));
    setVvadActive(false);
    setGuidelineOk(false);
    setPhase(PHASE.STOPPED);
    setModal({ type, message });
  };

  const connect = (s) => new Promise((resolve, reject) => {
    s.rejectConnect = reject;
    const ws = new WebSocket(WS_URL);
    s.ws = ws;
    ws.binaryType = 'arraybuffer';
    s.transport = new ChunkTransport(ws, {
      onResult: (message) => {
        if (!current(s) || s.aborted.has(message.utterance_id)) return;
        const key = `${s.sessionId}:${message.utterance_id}`;
        setSubtitles(lines => {
          const next = [...lines];
          const index = next.findIndex(line => line.key === key);
          const line = { key, text: message.text, final: message.is_final };
          if (index < 0) next.push(line); else next[index] = line;
          return next.slice(-SUBTITLE_MAX);
        });
      },
      onError: (error) => fail(s, error.message),
      onEnd: () => { if (current(s)) { dispose(s); setVvadActive(false); setPhase(PHASE.STOPPED); } },
    });
    s.connectTimer = setTimeout(() => fail(s, '서버 연결 시간이 초과되었습니다. 서버 실행 상태를 확인하세요.'), ACK_TIMEOUT_MS);
    ws.onopen = () => {
      if (current(s)) ws.send(JSON.stringify({ type: 'SESSION_START', session_id: s.sessionId,
        protocol_version: WS_PROTOCOL_VERSION, client_ts: Date.now() }));
    };
    ws.onmessage = (event) => {
      if (!current(s)) return;
      let message;
      try { message = JSON.parse(event.data); } catch { fail(s, '서버 응답을 읽을 수 없습니다.'); return; }
      if (message.type === 'SESSION_READY') {
        if (s.connected || message.protocol_version !== WS_PROTOCOL_VERSION || message.fps !== FPS ||
            message.chunk_frames !== CHUNK_FRAMES || message.max_utterance_frames !== MAX_UTTERANCE_FRAMES) {
          fail(s, '웹캠 화면과 서버 버전이 다릅니다. 함께 업데이트하세요.'); return;
        }
        s.connected = true;
        clearTimeout(s.connectTimer);
        s.rejectConnect = null;
        s.transport.receive(message);
        resolve();
      } else s.transport.receive(message);
    };
    ws.onclose = () => fail(s, '연결이 끊겨 진행 중 발화를 중단했습니다. 다시 시작하세요.');
    ws.onerror = () => fail(s, '서버에 연결할 수 없습니다. 서버 실행 상태를 확인하세요.');
  });

  const finishCalibration = (s) => {
    s.calibrating = false;
    const samples = s.calibSamples;
    if (samples.length >= 10) {
      const m = mean(samples), sd = std(samples, m);
      s.segmenter.thresholds = { tOpen: m + Math.max(2 * sd, 0.002),
        tClose: m + Math.max(0.8 * sd, 0.001), tVar: Math.max(variance(samples, m) * 1.5, 0.000001) };
      s.segmenter.ema = m;
      setCalibWarning(false);
    } else { s.segmenter.thresholds = { ...DEFAULT_THRESHOLDS }; setCalibWarning(true); }
    setPhase(PHASE.ACTIVE);
  };

  const schedule = (s) => {
    if (!current(s) || !s.running) return;
    const video = videoRef.current;
    s.videoCallback = typeof video.requestVideoFrameCallback === 'function';
    s.callbackId = s.videoCallback
      ? video.requestVideoFrameCallback((now, metadata) => processFrame(s, now, metadata.mediaTime * 1000))
      : requestAnimationFrame(now => processFrame(s, now, video.currentTime * 1000));
  };

  const processFrame = (s, now, mediaMs) => {
    if (!current(s) || !s.running) return;
    const video = videoRef.current;
    try {
      if (video.readyState >= 2 && mediaMs > s.lastMediaMs) {
        s.lastMediaMs = mediaMs; // rAF fallback도 같은 디코드 프레임을 다시 처리하지 않는다.
        if (s.originMs == null) s.originMs = mediaMs;
        const result = s.landmarker.detectForVideo(video, mediaMs);
        const landmarks = result.faceLandmarks?.[0];
        let value = null;
        if (landmarks) {
          const bbox = computeRoiBBox(landmarks, video.videoWidth, video.videoHeight);
          value = { roi: extractRoiBytes(video, bbox, canvasRef.current),
            ratio: computeMouthRatio(landmarks, video.videoWidth, video.videoHeight) };
        }
        const { samples, discontinuity } = s.resampler.push(mediaMs - s.originMs, value);
        if (discontinuity) { s.segmenter.abort(); setMissWarning(true); }
        for (const sample of samples) {
          s.misses = sample.value ? 0 : s.misses + 1;
          setGuidelineOk(!!sample.value);
          setMissWarning(s.misses >= MISS_WARN_FRAMES);
          if (s.calibrating) {
            if (sample.value) s.calibSamples.push(sample.value.ratio);
            const elapsed = now - s.calibStart;
            setCalibCountdown(Math.max(0, Math.ceil((CALIBRATION_MS - elapsed) / 1000)));
            if (elapsed >= CALIBRATION_MS) finishCalibration(s);
          } else s.segmenter.push(sample);
          if (!current(s)) return;
        }
      }
    } catch (error) { fail(s, `영상 처리에 실패했습니다: ${error.message}`); return; }
    schedule(s);
  };

  const handleStart = async () => {
    if (!window.isSecureContext) { setModal({ type: 'HTTPS_ERROR' }); return; }
    dispose(rt.current);
    const s = { sessionId: crypto.randomUUID(), cancelled: false, running: false, connected: false,
      lastMediaMs: -1, originMs: null, misses: 0, aborted: new Set(), resampler: new TimestampResampler() };
    rt.current = s;
    setModal(null); setCalibWarning(false); setMissWarning(false); setGuidelineOk(false);
    setPhase(PHASE.PERMISSION_REQUEST);
    s.segmenter = new SpeechSegmenter({
      onChunk: chunk => s.transport.enqueue(chunk),
      onAbort: id => {
        s.aborted.add(id); s.transport.abort(id);
        setSubtitles(lines => lines.filter(line => line.key !== `${s.sessionId}:${id}`));
      },
      onSpeaking: speaking => { if (current(s)) { setVvadActive(speaking);
        if (s.running && !s.calibrating) setPhase(speaking ? PHASE.LISTENING : PHASE.ACTIVE); } },
    });
    try {
      await connect(s); // 모델 준비가 안 되면 카메라 권한 요청 전에 구체적인 안내를 표시한다.
      if (!current(s)) return;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false,
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } });
      if (!current(s)) { stream.getTracks().forEach(track => track.stop()); return; }
      s.stream = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      if (!current(s)) return;
      const resolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      if (!current(s)) return;
      const model = await resolveFaceLandmarkerModel(FACE_LANDMARKER_MODEL);
      if (!current(s)) return;
      const landmarker = await FaceLandmarker.createFromOptions(resolver, {
        baseOptions: { modelAssetPath: model.url, delegate: 'GPU' }, runningMode: 'VIDEO', numFaces: 1,
      });
      if (!current(s)) { landmarker.close(); return; }
      s.landmarker = landmarker;
      s.calibSamples = []; s.calibStart = performance.now(); s.calibrating = true; s.running = true;
      setCalibCountdown(Math.ceil(CALIBRATION_MS / 1000)); setPhase(PHASE.CALIBRATING);
      schedule(s);
    } catch (error) {
      fail(s, error.name === 'NotAllowedError' ? '웹캠 접근 권한이 필요합니다.' : error.message);
    }
  };

  const handleStop = () => {
    const s = rt.current;
    if (!s || s.cancelled) { setPhase(PHASE.STOPPED); return; }
    if (!s.running) { dispose(s); setPhase(PHASE.STOPPED); return; }
    s.running = false;
    s.segmenter.finish();
    releaseCapture(s);
    setGuidelineOk(false); setVvadActive(false); setPhase(PHASE.STOPPING);
    s.transport.end(); // 마지막 결과를 받은 다음 SESSION_END를 보낸다.
  };

  const handleClearSubtitles = () => setSubtitles([]);
  useEffect(() => () => dispose(rt.current), []); // eslint-disable-line react-hooks/exhaustive-deps
  const started = phase !== PHASE.IDLE && phase !== PHASE.STOPPED;

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      {/* 헤더 */}
      <header className="mb-3 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">👄 립리딩 텍스트 변환</h1>
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
        ⚠ Chrome/Edge 권장 · HTTPS 또는 localhost 필요 · 한 번에 8초 이내로 말씀해 주세요
      </p>

      {phase === PHASE.PERMISSION_REQUEST && <p role="status" className="mb-2 text-sm text-blue-700">서버와 카메라를 준비하고 있습니다.</p>}
      {phase === PHASE.STOPPING && <p role="status" className="mb-2 text-sm text-blue-700">남은 발화의 결과를 기다리고 있습니다.</p>}
      {/* 캘리브레이션 배너 */}
      {phase === PHASE.CALIBRATING && (
        <div className="mb-3 rounded-lg bg-blue-50 px-3 py-2 text-sm text-blue-700">
          캘리브레이션 중... 잠시 가만히 계세요 ({calibCountdown}초)
        </div>
      )}
      {[PHASE.ACTIVE, PHASE.LISTENING].includes(phase) && !calibWarning && (
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
          입술이 잘 보이도록 맞춰주세요. 추적이 끊긴 발화는 다시 말씀해 주세요.
        </p>
      )}

      {/* 자막 영역 (최근 5개) */}
      <div className="mt-4 min-h-[6rem] rounded-2xl bg-gray-900 p-4">
        {subtitles.length === 0 ? (
          <p className="text-sm text-gray-500">자막이 여기에 표시됩니다.</p>
        ) : (
          subtitles.map((line) => (
            <p
              key={line.key}
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
            disabled={phase === PHASE.STOPPING}
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
