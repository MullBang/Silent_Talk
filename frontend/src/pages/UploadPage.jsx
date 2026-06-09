// UploadPage.jsx — SCR-01 영상 업로드 화면.
// 상태 흐름(8.1.3): IDLE → FILE_SELECTED → UPLOADING → INFERRING → DONE/TIMEOUT/ERROR
// Tailwind CSS만 사용.

import React, { useCallback, useEffect, useState } from 'react';

import { getResult, startInfer, uploadVideo } from '../api/client';
import DropZone from '../components/DropZone';
import ErrorModal from '../components/ErrorModal';
import ProgressBar from '../components/ProgressBar';
import usePolling from '../hooks/usePolling';

// 폴링 전체 타임아웃 (설계 5분)
const POLL_TIMEOUT_MS = 300000;
// 추론 종료(터미널) 상태. 백엔드: pending→processing→done/error/timeout.
// (명세의 status!=='processing'은 초기 pending에서 조기 종료되므로 터미널 기준으로 보정)
const TERMINAL = ['done', 'error', 'timeout'];

/**
 * INFERRING 단계에서만 마운트되는 결과 폴러.
 * usePolling으로 1초 간격 getResult를 호출하고, 5분 타임아웃을 감시한다.
 */
function ResultPoller({ sessionId, onProgress, onDone, onTimeout, onError }) {
  const fetchFn = useCallback(() => getResult(sessionId), [sessionId]);
  const { data, error } = usePolling(
    fetchFn,
    1000,
    (d) => TERMINAL.includes(d?.status),
  );

  // 5분 전체 타임아웃
  useEffect(() => {
    const t = setTimeout(() => onTimeout(), POLL_TIMEOUT_MS);
    return () => clearTimeout(t);
  }, [onTimeout]);

  useEffect(() => {
    if (!data) return;
    if (typeof data.progress === 'number') onProgress(data.progress);
    if (data.status === 'done') onDone(data.results || []);
    else if (data.status === 'timeout') onTimeout();
    else if (data.status === 'error') onError(data.error_message || '추론 중 오류가 발생했습니다.');
  }, [data, onProgress, onDone, onTimeout, onError]);

  useEffect(() => {
    if (error) onError(error.message || '네트워크 오류가 발생했습니다.');
  }, [error, onError]);

  return null;
}

/**
 * 업로드 페이지 (SCR-01).
 *
 * @param {object} props
 * @param {(payload: {sessionId: string, fileName: string, results: Array}) => void} [props.onComplete]
 *   추론 완료 시 결과 화면으로 전환하기 위한 콜백 (App 라우팅에서 연결).
 * @returns {JSX.Element}
 */
export default function UploadPage({ onComplete }) {
  const [phase, setPhase] = useState('IDLE');
  const [file, setFile] = useState(null);
  const [agreed, setAgreed] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [inferPct, setInferPct] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [modal, setModal] = useState(null); // { type, message }

  const busy = phase === 'UPLOADING' || phase === 'INFERRING';
  const showProgress = ['UPLOADING', 'INFERRING', 'DONE', 'TIMEOUT', 'ERROR'].includes(phase);

  const handleFileSelect = (f) => {
    setFile(f);
    setPhase(f ? 'FILE_SELECTED' : 'IDLE');
  };

  const handleStart = async () => {
    if (!file || !agreed || busy) return;
    setPhase('UPLOADING');
    setUploadPct(0);
    try {
      const up = await uploadVideo(file, setUploadPct);
      setSessionId(up.session_id);
      setInferPct(0);
      setPhase('INFERRING');
      await startInfer(up.session_id);
    } catch (err) {
      const msg =
        err?.response?.data?.detail || err?.message || '업로드/추론 시작에 실패했습니다.';
      setModal({ type: 'UPLOAD_ERROR', message: msg });
      setPhase('ERROR');
    }
  };

  const handleRetry = () => {
    setModal(null);
    setUploadPct(0);
    setInferPct(0);
    setSessionId(null);
    setPhase(file ? 'FILE_SELECTED' : 'IDLE');
  };

  // 폴러 콜백
  const onProgress = useCallback((p) => setInferPct(p), []);
  const onDone = useCallback(
    (results) => {
      setInferPct(100);
      setPhase('DONE');
      if (typeof onComplete === 'function') {
        onComplete({ sessionId, fileName: file?.name, results });
      }
    },
    [onComplete, sessionId, file],
  );
  const onTimeout = useCallback(() => setPhase('TIMEOUT'), []);
  const onError = useCallback((message) => {
    setModal({ type: 'UPLOAD_ERROR', message });
    setPhase('ERROR');
  }, []);

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      {/* 헤더 */}
      <header className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">🎤 립리딩 텍스트 변환</h1>
        <button
          type="button"
          disabled
          aria-label="웹캠 모드 (준비 중)"
          title="2차 목표 — 준비 중"
          className="cursor-not-allowed rounded-lg border border-gray-200 px-3 py-1.5 text-sm text-gray-400"
        >
          웹캠 모드 →
        </button>
      </header>

      {/* 개인정보 안내 (항상 표시) */}
      <p className="mb-6 rounded-lg bg-gray-50 px-3 py-2 text-xs text-gray-500">
        ℹ 업로드된 영상은 분석 완료 후 즉시 삭제됩니다.
      </p>

      {/* 드롭존 */}
      <DropZone file={file} onFileSelect={handleFileSelect} disabled={busy} />

      {/* 진행 상태 */}
      {showProgress && (
        <div className="mt-6">
          <ProgressBar
            status={phase}
            progress={phase === 'UPLOADING' ? uploadPct : inferPct}
          />
        </div>
      )}

      {/* 타임아웃/에러 재시도 안내 */}
      {(phase === 'TIMEOUT' || phase === 'ERROR') && (
        <div className="mt-4 flex items-center justify-between rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">
          <span>
            {phase === 'TIMEOUT'
              ? '응답이 지연되고 있습니다. 다시 시도해 주세요.'
              : '문제가 발생했습니다. 다시 시도해 주세요.'}
          </span>
          <button
            type="button"
            onClick={handleRetry}
            aria-label="다시 시도"
            className="rounded-md bg-amber-600 px-3 py-1 text-xs font-medium text-white hover:bg-amber-700"
          >
            재시도
          </button>
        </div>
      )}

      {/* 동의 체크박스 */}
      <label className="mt-6 flex items-start gap-2 text-sm text-gray-600">
        <input
          type="checkbox"
          checked={agreed}
          onChange={(e) => setAgreed(e.target.checked)}
          disabled={busy}
          className="mt-0.5"
        />
        <span>시연 영상은 사전 동의된 데이터 또는 공개 데이터입니다.</span>
      </label>

      {/* 분석 시작 */}
      <button
        type="button"
        onClick={handleStart}
        disabled={!file || !agreed || busy}
        aria-label="분석 시작"
        className="mt-4 w-full rounded-xl bg-blue-600 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
      >
        {busy ? '처리 중...' : '분석 시작'}
      </button>

      {/* 추론 폴러 (INFERRING 동안만 마운트) */}
      {phase === 'INFERRING' && sessionId && (
        <ResultPoller
          sessionId={sessionId}
          onProgress={onProgress}
          onDone={onDone}
          onTimeout={onTimeout}
          onError={onError}
        />
      )}

      {/* 에러 모달 */}
      {modal && (
        <ErrorModal
          type={modal.type}
          message={modal.message}
          onRetry={handleRetry}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  );
}
