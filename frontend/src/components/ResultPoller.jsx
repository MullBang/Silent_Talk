// ResultPoller.jsx — 추론 결과 폴러 (UploadPage·RecordPage 공용, 화면 출력 없음).
// usePolling으로 1초 간격 GET /api/result/{sessionId}를 호출하고 전체 타임아웃을 감시한다.

import { useEffect, useMemo } from 'react';

import { getResult } from '../api/client';
import {
  POLL_ERROR, POLL_INTERVAL_MS, POLL_TIMEOUT_MS, errorMessageFrom, interpretPoll, isTerminal, withNotFoundGrace,
} from '../api/pollOutcome';
import usePolling from '../hooks/usePolling';

/**
 * 추론 중에만 마운트하는 결과 폴러.
 *
 * @param {object} props
 * @param {string} props.sessionId
 * @param {(percent:number) => void} props.onProgress 서버 진행률(0~100).
 * @param {(results:Array, warnings:string[], model:object|null) => void} props.onDone
 * @param {() => void} props.onTimeout 서버 timeout 상태 또는 클라이언트 전체 타임아웃.
 * @param {(message:string, err?:any) => void} props.onError 서버 error 상태 또는 폴링 네트워크 오류
 *   (message는 이미 한국어 문구, err는 폴링 오류일 때만).
 * @param {number} [props.timeoutMs=POLL_TIMEOUT_MS]
 * @returns {null}
 */
export default function ResultPoller({ sessionId, onProgress, onDone, onTimeout, onError, timeoutMs = POLL_TIMEOUT_MS }) {
  // 첫 폴링이 Job 생성보다 먼저 도착한 404는 처음 몇 번만 '대기 중'으로 본다 (세션마다 새 순번).
  const fetchFn = useMemo(() => withNotFoundGrace(() => getResult(sessionId)), [sessionId]);
  const { data, error } = usePolling(fetchFn, POLL_INTERVAL_MS, isTerminal);

  // 전체 타임아웃
  useEffect(() => {
    const t = setTimeout(() => onTimeout(), timeoutMs);
    return () => clearTimeout(t);
  }, [onTimeout, timeoutMs]);

  useEffect(() => {
    const outcome = interpretPoll(data);
    if (outcome.kind === 'none') return;
    if (outcome.progress !== null) onProgress(outcome.progress);
    if (outcome.kind === 'done') onDone(outcome.results, outcome.warnings, outcome.model);
    else if (outcome.kind === 'timeout') onTimeout();
    else if (outcome.kind === 'error') onError(outcome.message);
  }, [data, onProgress, onDone, onTimeout, onError]);

  useEffect(() => {
    // 화면 문구는 고정 한국어(서버 detail 또는 연결 오류 문구). axios 영어 원문은 쓰지 않는다.
    if (error) onError(errorMessageFrom(error, POLL_ERROR), error);
  }, [error, onError]);

  return null;
}
