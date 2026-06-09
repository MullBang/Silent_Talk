// ResultList.jsx — 인식 결과 리스트 (화면명세서 SCR-02 UI-02-03).
// Tailwind CSS만 사용.

import React from 'react';

// 인식 불가/검출 실패 마커 (취소선 처리 대상).
// 백엔드는 비검출 세그먼트에 '[검출 실패 구간]'을 사용하므로 함께 포함한다.
export const NON_RECOGNITION_MARKERS = ['[인식 불가]', '[검출 실패 구간]'];

/** 비인식(마커) 텍스트 여부. */
export function isNonRecognition(text) {
  return NON_RECOGNITION_MARKERS.includes((text || '').trim());
}

/** ms → "MM:SS" (null이면 "--:--"). */
export function msToClock(ms) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return '--:--';
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(m)}:${pad(s)}`;
}

/**
 * 인식 결과 리스트.
 *
 * @param {object} props
 * @param {Array<{start_ms:number, end_ms:number, text:string, confidence:number|null}>} props.results
 * @returns {JSX.Element}
 */
export default function ResultList({ results }) {
  const items = results || [];

  if (items.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-gray-400">인식된 구간이 없습니다.</p>
    );
  }

  return (
    <ul className="divide-y divide-gray-100">
      {items.map((item, idx) => {
        const marker = isNonRecognition(item.text);
        // confidence < 0.5 → 회색 이탤릭 (초기 표시 기준; 검증 데이터로 CER/WER
        // 상관관계 확인 후 조정 가능)
        const lowConf =
          !marker && typeof item.confidence === 'number' && item.confidence < 0.5;

        const textCls = marker
          ? 'text-gray-400 line-through'
          : lowConf
            ? 'italic text-gray-400'
            : 'text-gray-800';

        return (
          <li key={idx} className="flex items-baseline gap-3 py-2">
            <span className="shrink-0 font-mono text-xs text-gray-400">
              [{msToClock(item.start_ms)}~{msToClock(item.end_ms)}]
            </span>
            <span className={`flex-1 text-sm ${textCls}`}>{item.text}</span>
            {typeof item.confidence === 'number' && !marker && (
              <span className="shrink-0 text-xs text-gray-300">
                {item.confidence.toFixed(2)}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
