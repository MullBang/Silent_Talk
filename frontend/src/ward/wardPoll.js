// wardPoll.js — 병실 화면 전용 결과 폴링 (약 300ms 간격, 응답을 받은 뒤 다음 요청). node --test로 검사.
// 다른 화면의 1초 폴링(ResultPoller)은 바꾸지 않는다. 응답 해석·오류 문구는 pollOutcome.js를 그대로 쓴다.

import { POLL_ERROR, errorMessageFrom, interpretPoll, withNotFoundGrace } from '../api/pollOutcome.js';

export const WARD_POLL_INTERVAL_MS = 300;
/** 병실 화면은 이 시간 안에 결과가 없으면 분류 스캐닝으로 넘어간다 (5초 녹화 기준 넉넉한 값). */
export const WARD_POLL_TIMEOUT_MS = 45000;
/** 300ms 간격이라 첫 404 유예를 다른 화면(2회·1초 간격)보다 넉넉히 둔다. */
export const WARD_NOT_FOUND_GRACE = 5;

/** 기본 대기 함수. */
const defaultSleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

/**
 * 결과가 끝날 때까지 폴링한다.
 * @param {object} opts
 * @param {() => Promise<any>} opts.fetchResult GET /api/result/{session_id}.
 * @param {number} [opts.intervalMs=WARD_POLL_INTERVAL_MS]
 * @param {number} [opts.timeoutMs=WARD_POLL_TIMEOUT_MS]
 * @param {(ms:number) => Promise<void>} [opts.sleep] 테스트용 대기.
 * @param {() => number} [opts.now] 테스트용 시계.
 * @param {() => boolean} [opts.isCancelled] true면 바로 멈춘다(화면 이탈·새 녹화).
 * @returns {Promise<{kind:'done', results:Array, warnings:string[]}|{kind:'error', message:string}
 *   |{kind:'timeout'}|{kind:'cancelled'}>}
 */
export async function pollWardResult({
  fetchResult, intervalMs = WARD_POLL_INTERVAL_MS, timeoutMs = WARD_POLL_TIMEOUT_MS,
  sleep = defaultSleep, now = Date.now, isCancelled = () => false,
}) {
  const fetchFn = withNotFoundGrace(fetchResult, WARD_NOT_FOUND_GRACE);
  const started = now();
  for (;;) {
    if (isCancelled()) return { kind: 'cancelled' };
    let data;
    try {
      data = await fetchFn();
    } catch (err) {
      if (isCancelled()) return { kind: 'cancelled' };
      return { kind: 'error', message: errorMessageFrom(err, POLL_ERROR) };
    }
    if (isCancelled()) return { kind: 'cancelled' };
    const outcome = interpretPoll(data);
    if (outcome.kind === 'done') return { kind: 'done', results: outcome.results, warnings: outcome.warnings };
    if (outcome.kind === 'error') return { kind: 'error', message: outcome.message };
    if (outcome.kind === 'timeout') return { kind: 'timeout' };
    if (now() - started >= timeoutMs) return { kind: 'timeout' };
    try {
      await sleep(intervalMs);
    } catch {
      return { kind: 'cancelled' };
    }
  }
}
