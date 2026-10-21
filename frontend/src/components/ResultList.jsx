// ResultList.jsx — 인식 결과 리스트 (화면명세서 SCR-02 UI-02-03).
// Tailwind CSS만 사용.

import React from 'react';

import { isNonRecognition, lowConfidenceCandidate } from '../results/markers';

// 인식 불가/검출 실패/발화 없음 마커 (취소선 처리 대상). 정의는 순수 모듈 results/markers.js.
export { NON_RECOGNITION_MARKERS, isNonRecognition } from '../results/markers';

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
 * @param {Array<{start_ms:number, end_ms:number, text:string, confidence:number|null,
 *   candidate?:string|null}>} props.results '[인식 불가]' 항목의 candidate는 낮은 신뢰도 추정으로 흐리게 표시한다.
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
        const candidate = lowConfidenceCandidate(item);
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
            <span className="min-w-0 flex-1 text-sm">
              <span className={textCls}>{item.text}</span>
              {candidate && (
                <span className="mt-0.5 block break-words text-xs text-gray-400">
                  낮은 신뢰도 추정: {candidate}
                  {typeof item.confidence === 'number' && ` (${item.confidence.toFixed(2)})`}
                </span>
              )}
            </span>
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
