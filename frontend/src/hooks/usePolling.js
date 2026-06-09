// usePolling.js — 일정 간격 폴링 커스텀 훅.

import { useEffect, useRef, useState } from 'react';

/**
 * fetchFn을 intervalMs 간격으로 반복 호출하고, stopCondition(data)가 true이면
 * 폴링을 중지한다. 컴포넌트 언마운트 시 자동 cleanup.
 *
 * @param {() => Promise<any>} fetchFn - 폴링마다 호출할 비동기 함수.
 * @param {number} [intervalMs=1000] - 폴링 간격(밀리초).
 * @param {(data: any) => boolean} [stopCondition] - true 반환 시 폴링 중지.
 * @returns {{ data: any, loading: boolean, error: any }}
 */
export default function usePolling(fetchFn, intervalMs = 1000, stopCondition) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 최신 콜백을 ref로 유지해 effect 재실행 없이 참조
  const fetchRef = useRef(fetchFn);
  const stopRef = useRef(stopCondition);
  fetchRef.current = fetchFn;
  stopRef.current = stopCondition;

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    const stop = () => {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };

    const tick = async () => {
      try {
        const result = await fetchRef.current();
        if (cancelled) return;
        setData(result);
        setError(null);
        if (typeof stopRef.current === 'function' && stopRef.current(result)) {
          setLoading(false);
          stop();
        }
      } catch (err) {
        if (cancelled) return;
        setError(err);
        setLoading(false);
        stop();
      }
    };

    setLoading(true);
    tick(); // 즉시 1회 호출 후 인터벌 시작
    timer = setInterval(tick, intervalMs);

    return () => {
      cancelled = true;
      stop();
    };
  }, [intervalMs]);

  return { data, loading, error };
}
