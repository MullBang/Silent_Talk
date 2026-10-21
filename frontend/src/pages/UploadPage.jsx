// UploadPage.jsx — SCR-01 영상 업로드 화면.
// 상태 흐름(8.1.3): IDLE → FILE_SELECTED → UPLOADING → INFERRING → DONE/TIMEOUT/ERROR
// Tailwind CSS만 사용.

import React, { useCallback, useState } from 'react';

import { startInfer, uploadVideo } from '../api/client';
import { START_ERROR, errorMessageFrom } from '../api/pollOutcome';
import DropZone from '../components/DropZone';
import ErrorModal from '../components/ErrorModal';
import ProgressBar from '../components/ProgressBar';
import ResultPoller from '../components/ResultPoller';

/**
 * 업로드 페이지 (SCR-01).
 *
 * @param {object} props
 * @param {(payload: {sessionId: string, fileName: string, results: Array, warnings: string[],
 *   model: object|null, origin: 'upload'}) => void} [props.onComplete]
 *   추론 완료 시 결과 화면으로 전환하기 위한 콜백 (App 라우팅에서 연결).
 * @param {() => void} [props.onWebcam] 웹캠 실시간 모드로 전환.
 * @param {() => void} [props.onCapture] 파일럿 촬영 화면으로 전환.
 * @param {() => void} [props.onRecord] 녹화→분석 화면으로 전환 (6단계).
 * @returns {JSX.Element}
 */
export default function UploadPage({ onComplete, onWebcam, onCapture, onRecord }) {
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
      if (!up?.session_id) throw new Error('missing session_id');
      // Job은 POST /api/infer가 만든다. 응답을 받은 뒤에 폴러를 마운트해야 첫 폴링이 404가 되지 않는다.
      await startInfer(up.session_id);
      setSessionId(up.session_id);
      setInferPct(0);
      setPhase('INFERRING');
    } catch (err) {
      setModal({ type: 'UPLOAD_ERROR', message: errorMessageFrom(err, START_ERROR) });
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
    (results, warnings, model) => {
      setInferPct(100);
      setPhase('DONE');
      if (typeof onComplete === 'function') {
        onComplete({
          sessionId, fileName: file?.name, results, warnings: warnings || [], model: model || null, origin: 'upload',
        });
      }
    },
    [onComplete, sessionId, file],
  );
  const onTimeout = useCallback(() => setPhase('TIMEOUT'), []);
  const onError = useCallback((message, err) => {
    setModal({ type: 'UPLOAD_ERROR', message: err ? errorMessageFrom(err, message) : message });
    setPhase('ERROR');
  }, []);

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      {/* 헤더 */}
      <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-bold text-gray-900">🎤 립리딩 텍스트 변환</h1>
        <div className="flex flex-wrap justify-end gap-2">
          {onRecord && (
            <button
              type="button"
              onClick={onRecord}
              aria-label="녹화해서 분석 화면으로 전환"
              className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-blue-700"
            >
              녹화해서 분석 →
            </button>
          )}
          {onCapture && (
            <button
              type="button"
              onClick={onCapture}
              aria-label="파일럿 촬영 화면으로 전환"
              className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 transition hover:bg-gray-50"
            >
              파일럿 촬영 →
            </button>
          )}
          <button
            type="button"
            onClick={onWebcam}
            aria-label="웹캠 모드로 전환"
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 transition hover:bg-gray-50"
          >
            웹캠 모드로 전환 →
          </button>
        </div>
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
