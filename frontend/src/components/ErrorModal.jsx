// ErrorModal.jsx — 시연장 장애 대응 모달 (설계문서 14.4절).
// Tailwind CSS만 사용. 외부 UI 라이브러리 미사용.

import React from 'react';

/** 장애 유형별 안내 메시지. */
const TYPE_MESSAGES = {
  PERMISSION_DENIED: '웹캠 접근 권한이 필요합니다',
  HTTPS_ERROR: 'HTTPS 또는 localhost 환경에서만 웹캠을 사용할 수 있습니다',
  DEVICE_ERROR: '카메라 장치 및 브라우저 미디어 권한을 확인해 주세요',
};

/**
 * 에러/장애 안내 모달.
 *
 * @param {object} props
 * @param {('PERMISSION_DENIED'|'HTTPS_ERROR'|'DEVICE_ERROR'|'UPLOAD_ERROR')} props.type - 장애 유형.
 * @param {string} [props.message] - UPLOAD_ERROR일 때 직접 표시할 메시지.
 * @param {() => void} [props.onRetry] - 재시도 콜백 (있으면 재시도 버튼 노출).
 * @param {() => void} props.onClose - 닫기 콜백.
 * @returns {JSX.Element}
 */
export default function ErrorModal({ type, message, onRetry, onClose }) {
  const text =
    type === 'UPLOAD_ERROR'
      ? message || '업로드 중 오류가 발생했습니다.'
      : TYPE_MESSAGES[type] || '오류가 발생했습니다.';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="error-modal-title"
    >
      <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-xl">
        {/* 헤더 + 닫기 */}
        <div className="mb-4 flex items-start justify-between">
          <h2
            id="error-modal-title"
            className="text-lg font-semibold text-red-600"
          >
            ⚠ 오류
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="닫기"
            className="rounded p-1 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600"
          >
            ✕
          </button>
        </div>

        {/* 메시지 */}
        <p className="mb-6 text-sm leading-relaxed text-gray-700">{text}</p>

        {/* 액션 버튼 */}
        <div className="flex justify-end gap-2">
          {typeof onRetry === 'function' && (
            <button
              type="button"
              onClick={onRetry}
              aria-label="재시도"
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700"
            >
              재시도
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            aria-label="닫기"
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}
