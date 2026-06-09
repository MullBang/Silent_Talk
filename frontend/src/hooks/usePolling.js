// usePolling.js — 1초 간격 결과 폴링 커스텀 훅.

import { useEffect, useState } from 'react';

/**
 * 지정 함수를 일정 간격으로 폴링하는 커스텀 훅.
 * 기본 1초 간격으로 GET /api/result/{session_id}를 호출하고,
 * status가 'done' 또는 'failed'이면 폴링을 중단한다.
 *
 * @param {() => Promise<any>} fetcher - 폴링마다 호출할 비동기 함수.
 * @param {number} [intervalMs=1000] - 폴링 간격(밀리초).
 * @param {boolean} [enabled=true] - 폴링 활성화 여부.
 * @returns {{ data: any, error: any, isPolling: boolean }}
 */
export default function usePolling(fetcher, intervalMs = 1000, enabled = true) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [isPolling, setIsPolling] = useState(enabled);

  useEffect(() => {
    // TODO: setInterval 기반 폴링, 종료 상태 감지, cleanup
  }, [fetcher, intervalMs, enabled]);

  return { data, error, isPolling };
}
