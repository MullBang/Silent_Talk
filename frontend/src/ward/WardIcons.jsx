// WardIcons.jsx — 병실 화면 아이콘 (인라인 SVG, 외부 파일 없음). 긴급은 색·글자·아이콘을 함께 쓴다.
// 선과 기호를 모두 글자색(currentColor)으로 그려 어느 바탕에서도 글자와 같은 명암비를 갖는다.

import React from 'react';

/**
 * 경고 삼각형 아이콘 (긴급 표시).
 * @param {{size?: number|string, title?: string}} props
 * @returns {JSX.Element}
 */
export function AlertIcon({ size = '1em', title = '긴급' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" role="img" aria-label={title} style={{ flex: 'none' }}>
      <path d="M12 3 2 20.5h20L12 3Z" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M12 9.5v5" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="12" cy="17.6" r="1.4" fill="currentColor" />
    </svg>
  );
}

/**
 * 확인 표시 아이콘.
 * @param {{size?: number|string, title?: string}} props
 * @returns {JSX.Element}
 */
export function CheckIcon({ size = '1em', title = '확인' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" role="img" aria-label={title} style={{ flex: 'none' }}>
      <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2.2" />
      <path d="m7 12.5 3.4 3.4 6.6-7.2" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
