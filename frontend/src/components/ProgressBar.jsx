// ProgressBar.jsx — 업로드/추론 상태 표시 바 (화면명세서 SCR-01 UI-01-04).
// Tailwind CSS만 사용.

import React from 'react';

/**
 * 상태별 진행 표시 바.
 *
 * @param {object} props
 * @param {('UPLOADING'|'INFERRING'|'DONE'|'TIMEOUT'|'ERROR')} props.status
 * @param {number} [props.progress=0] - 진행률 0~100 (UPLOADING/INFERRING).
 * @returns {JSX.Element}
 */
export default function ProgressBar({ status, progress = 0 }) {
  const pct = Math.max(0, Math.min(100, Math.round(progress)));

  if (status === 'UPLOADING') {
    return (
      <div className="w-full" aria-live="polite">
        <div className="mb-1 flex justify-between text-sm text-gray-600">
          <span>업로드 중...</span>
          <span>{pct}%</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-gray-200">
          <div
            className="h-full rounded-full bg-blue-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  if (status === 'INFERRING') {
    return (
      <div className="w-full" aria-live="polite">
        <div className="mb-1 flex items-center gap-2 text-sm text-gray-700">
          <span
            className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"
            aria-hidden="true"
          />
          <span>분석 중...</span>
          <span className="ml-auto text-gray-500">{pct}%</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-gray-200">
          <div
            className="h-full rounded-full bg-blue-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  if (status === 'DONE') {
    return (
      <div className="flex items-center gap-2 text-sm font-medium text-green-600">
        <span aria-hidden="true">✓</span>
        <span>완료</span>
      </div>
    );
  }

  if (status === 'TIMEOUT') {
    return (
      <div className="flex items-center gap-2 text-sm font-medium text-amber-600">
        <span aria-hidden="true">⏱</span>
        <span>시간 초과</span>
      </div>
    );
  }

  if (status === 'ERROR') {
    return (
      <div className="flex items-center gap-2 text-sm font-medium text-red-600">
        <span aria-hidden="true">⚠</span>
        <span>오류 발생</span>
      </div>
    );
  }

  return null;
}
