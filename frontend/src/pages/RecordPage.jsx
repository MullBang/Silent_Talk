// RecordPage.jsx — 6단계 녹화 → 분석 → 결과 화면.
// 웹캠 영상만 녹화(음성 트랙 없음)해 POST /api/record 로 보내고 GET /api/result 폴링으로 결과를 받는다.
// 외부 주소를 쓰지 않는다 (입술 검출·인식은 서버에서 수행). 실시간 WebSocket 경로(WebcamPage)와 별개다.
// 상태 흐름: OPENING → READY → (COUNTDOWN) → RECORDING → UPLOADING → INFERRING → 결과 화면 / ERROR / TIMEOUT

import React, { useCallback, useEffect, useRef, useState } from 'react';

import { uploadRecording } from '../api/client';
import { errorMessageFrom } from '../api/pollOutcome';
import ErrorModal from '../components/ErrorModal';
import ProgressBar from '../components/ProgressBar';
import ResultPoller from '../components/ResultPoller';
import useRecorder from '../hooks/useRecorder';
import { blurAfterPointerClick } from '../record/pointerFocus';
import {
  COUNTDOWN_SECONDS, DEFAULT_MODE_ID, MIN_RECORD_SECONDS, RECORD_MODES, SERVER_MAX_SECONDS,
  buildRecordPayload, formatElapsed, modeById, phaseLabel, recordUploadErrorMessage,
  recordingProgress, validateRecording,
} from '../record/recordFlow';

const CAMERA = { width: 1280, height: 720, frameRate: 30 };

/**
 * 녹화 → 분석 페이지.
 *
 * @param {object} props
 * @param {(payload: {sessionId:string, fileName:string, results:Array, warnings:string[],
 *   model:object|null, origin:'record'}) => void} props.onComplete 분석 완료 시 결과 화면으로 전환.
 * @param {() => void} props.onBack 업로드 화면으로 돌아가기.
 * @returns {JSX.Element}
 */
export default function RecordPage({ onComplete, onBack }) {
  const [modeId, setModeId] = useState(DEFAULT_MODE_ID);
  const [useCountdown, setUseCountdown] = useState(true);
  const [countdown, setCountdown] = useState(0);
  // 녹화 외 작업 단계: NONE | COUNTDOWN | UPLOADING | INFERRING | TIMEOUT | ERROR
  const [work, setWork] = useState('NONE');
  const [notice, setNotice] = useState(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [inferPct, setInferPct] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [modal, setModal] = useState(null); // { type, message }
  const [insecure, setInsecure] = useState(false);

  const mode = modeById(modeId);
  const mountedRef = useRef(true);
  const runRef = useRef(0); // 업로드/분석 실행 번호: 화면 이탈·재녹화 뒤 늦은 응답 무시
  const recordedAtRef = useRef(new Date());
  const countdownTimerRef = useRef(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  /** 녹화 종료(수동·자동) → 1차 검사 → 업로드 → 분석 Job 시작. */
  const handleRecorded = useCallback(async ({ blob, mime, meta }) => {
    const check = validateRecording(blob, meta?.durationMs);
    if (!check.ok) {
      setNotice(check.message);
      return;
    }
    runRef.current += 1;
    const run = runRef.current;
    setNotice(null);
    setUploadPct(0);
    setInferPct(0);
    setSessionId(null);
    setWork('UPLOADING');
    try {
      const data = await uploadRecording(blob, mime, (p) => {
        if (mountedRef.current && runRef.current === run) setUploadPct(p);
      });
      if (!mountedRef.current || runRef.current !== run) return;
      if (!data?.session_id) throw new Error('missing session_id');
      setSessionId(data.session_id);
      setWork('INFERRING');
    } catch (err) {
      if (!mountedRef.current || runRef.current !== run) return;
      setModal({ type: 'UPLOAD_ERROR', message: recordUploadErrorMessage(err) });
      setWork('ERROR');
    }
  }, []);

  const rec = useRecorder({ maxSeconds: mode.maxSeconds, video: CAMERA, onResult: handleRecorded });
  const { status, error, errorType, errorSeq, videoRef, settings, open, close, start, stop, elapsedMs } = rec;
  const startRef = useRef(start);
  startRef.current = start;

  const clearCountdown = useCallback(() => {
    clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = null;
    setCountdown(0);
  }, []);

  // 화면 진입 시 카메라를 연다 (HTTPS/localhost가 아니면 안내 모달). 이탈 시 카메라는 useRecorder가 해제한다.
  useEffect(() => {
    mountedRef.current = true;
    if (!globalThis.isSecureContext) {
      setInsecure(true);
      setModal({ type: 'HTTPS_ERROR' });
    } else {
      open();
    }
    return () => {
      mountedRef.current = false;
      runRef.current += 1;
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    };
  }, [open]);

  // 카메라 오류마다(errorSeq 증가) 모달을 띄운다. error가 null이면 모달의 기본 한국어 문구를 쓴다.
  // 업로드·분석 중이거나 업로드 오류 안내가 떠 있으면 그 안내를 덮지 않는다(카메라 다시 열기 버튼은 그대로 보인다).
  useEffect(() => {
    if (errorSeq <= 0 || work === 'UPLOADING' || work === 'INFERRING') return;
    setModal((current) => (current?.type === 'UPLOAD_ERROR'
      ? current
      : { type: errorType || 'DEVICE_ERROR', message: error || undefined }));
    // error/errorType은 errorSeq와 같은 렌더에서 바뀐다. open()이 error를 지울 때 모달을 다시 띄우지 않도록 errorSeq만 본다.
  }, [errorSeq]); // eslint-disable-line react-hooks/exhaustive-deps

  // 카운트다운 도중 카메라가 끊기면 카운트다운을 취소한다 (끊긴 카메라로 녹화를 시작하지 않음).
  useEffect(() => {
    if (status === 'error' && countdownTimerRef.current) {
      clearCountdown();
      setWork('NONE');
    }
  }, [status, clearCountdown]);

  const busy = work === 'UPLOADING' || work === 'INFERRING';
  const recording = status === 'recording';
  const counting = work === 'COUNTDOWN';
  const canStart = status === 'ready' && !busy && !counting;

  /** 녹화 시작 (카운트다운 선택 시 3-2-1 후 시작). */
  const begin = useCallback(() => {
    if (!canStart) return;
    setNotice(null);
    setModal(null);
    if (!useCountdown) {
      recordedAtRef.current = new Date();
      setWork('NONE');
      startRef.current();
      return;
    }
    let left = COUNTDOWN_SECONDS;
    setCountdown(left);
    setWork('COUNTDOWN');
    countdownTimerRef.current = setInterval(() => {
      left -= 1;
      if (left > 0) {
        setCountdown(left);
        return;
      }
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
      setCountdown(0);
      setWork('NONE');
      recordedAtRef.current = new Date();
      startRef.current();
    }, 1000);
  }, [canStart, useCountdown]);

  const cancelCountdown = useCallback(() => {
    clearCountdown();
    setWork('NONE');
  }, [clearCountdown]);

  /** 시작/정지 토글 (버튼·스페이스바 공용). */
  const toggle = useCallback(() => {
    if (recording) stop();
    else if (counting) cancelCountdown();
    else begin();
  }, [recording, counting, stop, cancelCountdown, begin]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.code !== 'Space' || e.repeat) return;
      const tag = e.target?.tagName;
      if (e.target !== document.body && tag !== 'VIDEO' && tag !== 'DIV') return;
      e.preventDefault();
      toggle();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggle]);

  const handleBack = () => {
    runRef.current += 1;
    clearCountdown();
    close();
    onBack?.();
  };

  const resetAfterFailure = () => {
    runRef.current += 1;
    setModal(null);
    setUploadPct(0);
    setInferPct(0);
    setSessionId(null);
    setWork('NONE');
    if (status === 'idle' || status === 'error') open();
  };

  const reopenCamera = () => {
    setModal(null);
    if (!globalThis.isSecureContext) { setModal({ type: 'HTTPS_ERROR' }); return; }
    open();
  };

  // 폴러 콜백
  const onPollProgress = useCallback((p) => setInferPct(p), []);
  const onPollDone = useCallback((results, warnings, model) => {
    setInferPct(100);
    setWork('NONE');
    onCompleteRef.current?.(buildRecordPayload({
      sessionId, recordedAt: recordedAtRef.current, results, warnings, model,
    }));
  }, [sessionId]);
  const onPollTimeout = useCallback(() => setWork('TIMEOUT'), []);
  const onPollError = useCallback((message, err) => {
    setModal({ type: 'UPLOAD_ERROR', message: err ? errorMessageFrom(err, message) : message });
    setWork('ERROR');
  }, []);

  const progress = recordingProgress(elapsedMs, mode.maxSeconds);
  const phase = work !== 'NONE' ? work : ({ idle: 'IDLE', opening: 'OPENING', ready: 'READY', recording: 'RECORDING', error: 'ERROR' })[status] || 'IDLE';
  const showWorkProgress = work === 'UPLOADING' || work === 'INFERRING' || work === 'TIMEOUT' || work === 'ERROR';

  let mainLabel = '녹화 시작';
  if (recording) mainLabel = '녹화 정지';
  else if (counting) mainLabel = '카운트다운 취소';
  else if (busy) mainLabel = '분석 중...';

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      {/* 헤더 */}
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-bold text-gray-900">🎥 녹화해서 분석</h1>
        <button
          type="button"
          onClick={handleBack}
          aria-label="업로드 화면으로 돌아가기"
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 transition hover:bg-gray-50"
        >
          ← 업로드 모드
        </button>
      </header>

      {/* 음성 꺼짐 표시 + 안내 */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span
          role="status"
          className="inline-flex items-center gap-1 rounded-full border border-green-200 bg-green-50 px-3 py-1 text-xs font-semibold text-green-700"
        >
          <span aria-hidden="true">🔇</span> 음성 꺼짐 (마이크 사용 안 함)
        </span>
        <span className="text-xs text-gray-500">영상만 녹화합니다. 녹화 영상은 분석 후 서버에서 즉시 삭제됩니다.</span>
      </div>

      {/* 모드 · 카운트다운 */}
      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        {/* 마우스로 누른 뒤에는 포커스를 풀어 스페이스바가 녹화 시작·정지로 가게 한다 (키보드 조작은 그대로). */}
        <div
          role="radiogroup"
          aria-label="녹화 길이"
          onClick={blurAfterPointerClick}
          className="inline-flex overflow-hidden rounded-lg border border-gray-300"
        >
          {RECORD_MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              role="radio"
              aria-checked={modeId === m.id}
              onClick={() => setModeId(m.id)}
              disabled={recording || counting || busy}
              className={`px-3 py-1.5 transition disabled:cursor-not-allowed ${
                modeId === m.id ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-gray-600" onClick={blurAfterPointerClick}>
          <input
            type="checkbox"
            checked={useCountdown}
            onChange={(e) => setUseCountdown(e.target.checked)}
            disabled={recording || counting || busy}
          />
          {COUNTDOWN_SECONDS}초 카운트다운
        </label>
      </div>
      {mode.experimental && (
        <p className="mb-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
          30초 모드는 실험용입니다. 서버는 {SERVER_MAX_SECONDS}초 이상 녹화를 받지 않으므로 {mode.maxSeconds}초에 자동으로 끝납니다.
          긴 녹화는 여러 구간으로 나뉘어 인식될 수 있습니다.
        </p>
      )}

      {/* 카메라 미리보기 + 얼굴 위치 안내 */}
      <div className="relative mb-3 aspect-video w-full overflow-hidden rounded-xl bg-black">
        <video
          ref={videoRef}
          className="h-full w-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
          playsInline
          muted
        />
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox="0 0 160 90"
          preserveAspectRatio="xMidYMid meet"
          aria-hidden="true"
        >
          <ellipse cx="80" cy="44" rx="22" ry="30" fill="none" stroke="white" strokeOpacity="0.75" strokeWidth="0.8" strokeDasharray="3 2" />
          <rect x="70" y="56" width="20" height="10" rx="4" fill="none" stroke="#60a5fa" strokeOpacity="0.9" strokeWidth="0.7" />
        </svg>
        {status !== 'ready' && status !== 'recording' && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-white/80">
            {status === 'opening' ? '카메라를 여는 중...' : insecure ? 'HTTPS 또는 localhost에서만 카메라를 쓸 수 있습니다' : '카메라가 꺼져 있습니다'}
          </div>
        )}
        {counting && countdown > 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
            <span className="text-7xl font-bold text-white drop-shadow" aria-live="assertive">{countdown}</span>
          </div>
        )}
        {recording && (
          <span className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full bg-red-600 px-2.5 py-1 text-xs font-semibold text-white">
            <span className="h-2 w-2 animate-pulse rounded-full bg-white" aria-hidden="true" /> 녹화 중
          </span>
        )}
      </div>
      <p className="mb-3 text-xs text-gray-500">
        얼굴을 점선 타원 안에, 입이 파란 상자 근처에 오도록 정면을 봐 주세요. 밝고 고른 조명이 좋습니다.
      </p>

      {/* 녹화 시간 */}
      <div className="mb-3" aria-live="polite">
        <div className="mb-1 flex justify-between font-mono text-sm text-gray-700">
          <span>경과 {formatElapsed(recording ? progress.elapsedMs : 0)}</span>
          <span>남은 시간 {formatElapsed(recording ? progress.remainingMs : mode.maxSeconds * 1000)}</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
          <div
            className={`h-full rounded-full ${recording ? 'bg-red-500' : 'bg-gray-300'}`}
            style={{ width: `${recording ? progress.percent : 0}%` }}
          />
        </div>
      </div>

      {/* 시작/정지 */}
      <button
        type="button"
        onClick={toggle}
        disabled={!(recording || counting || canStart)}
        aria-label={mainLabel}
        className={`w-full rounded-xl py-3 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:bg-gray-300 ${
          recording ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
        }`}
      >
        {mainLabel}
      </button>
      <p className="mt-1 text-center text-xs text-gray-400">
        스페이스바로도 시작·정지합니다 · {mode.maxSeconds}초가 되면 자동으로 멈춥니다 · {MIN_RECORD_SECONDS}초 이상 녹화
      </p>

      {/* 상태 안내 */}
      <p className="mt-3 text-sm text-gray-600">상태: {phaseLabel(phase)}</p>
      {notice && (
        <p role="alert" className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">{notice}</p>
      )}
      {(status === 'idle' || status === 'error') && !insecure && (
        <button
          type="button"
          onClick={reopenCamera}
          className="mt-2 rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
        >
          카메라 다시 열기
        </button>
      )}

      {/* 업로드·분석 진행 */}
      {showWorkProgress && (
        <div className="mt-4">
          <ProgressBar status={work} progress={work === 'UPLOADING' ? uploadPct : inferPct} />
        </div>
      )}
      {(work === 'TIMEOUT' || work === 'ERROR') && (
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">
          <span>
            {work === 'TIMEOUT' ? '분석 응답이 지연되고 있습니다. 다시 녹화해 주세요.' : '문제가 발생했습니다. 다시 녹화해 주세요.'}
          </span>
          <button
            type="button"
            onClick={resetAfterFailure}
            className="rounded-md bg-amber-600 px-3 py-1 text-xs font-medium text-white hover:bg-amber-700"
          >
            다시 녹화
          </button>
        </div>
      )}

      {settings && (
        <p className="mt-4 text-xs text-gray-400">
          카메라: {settings.width}×{settings.height} @ {typeof settings.frameRate === 'number' ? Math.round(settings.frameRate) : '?'}fps
        </p>
      )}

      {/* 추론 폴러 (INFERRING 동안만 마운트) */}
      {work === 'INFERRING' && sessionId && (
        <ResultPoller
          sessionId={sessionId}
          onProgress={onPollProgress}
          onDone={onPollDone}
          onTimeout={onPollTimeout}
          onError={onPollError}
        />
      )}

      {/* 에러 모달 */}
      {modal && (
        <ErrorModal
          type={modal.type}
          message={modal.message}
          onRetry={modal.type === 'UPLOAD_ERROR' ? resetAfterFailure : undefined}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  );
}
