// ResultPage.jsx — SCR-02 결과 화면.
// usePolling으로 1초 간격 결과 폴링, ProgressBar/ResultList 렌더링.

import React from 'react';

/**
 * 결과 페이지 (SCR-02).
 * GET /api/result/{session_id}를 1초 간격 폴링하여 진행률과 결과를 표시한다.
 * @param {{ sessionId: string }} props
 * @returns {JSX.Element}
 */
export default function ResultPage({ sessionId }) {
  // TODO: 폴링 훅 연결, 진행률/결과 렌더링
  return null;
}
