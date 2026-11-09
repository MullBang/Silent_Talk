// wardResult.js — 녹화 분석 결과(GET /api/result의 results)를 병실 판정 입력으로 바꾸는 순수 함수 (node --test로 검사).
// 병실 화면은 받아쓰기 글자를 쓰지 않고 지정 문장 결과(phrase)만 쓴다. 고를지 말지는 서버의 phrase.accepted만 믿는다.
// 긴급 여부는 화면이 commands.json으로 판단한다(서버는 긴급을 해석하지 않는다).

import { DETECTION_FAILED_TEXT, NO_SPEECH_TEXT } from '../results/markers.js';
import { WARD_PHRASE_SET, commandById } from './wardCommands.js';

/** 지정 문장 결과가 없을 때의 사유 → 환자 화면 문구. */
export const NO_PHRASE_REASON_TEXT = Object.freeze({
  empty: '분석 결과가 비어 있어요',
  multi_window: '녹화가 길어 여러 구간으로 나뉘었어요',
  partial_detection: '일부 구간에서 입이 안 보였어요',
  unaligned: '말이 너무 짧거나 목록 문장과 맞춰 볼 수 없었어요',
  other_set: '병실 명령 목록으로 분석되지 않았어요',
  error: '분석하지 못했어요',
  timeout: '분석이 너무 오래 걸렸어요',
  not_candidates: '후보 중에 없다고 고르셨어요',
});

/**
 * 결과 항목 하나가 마커(발화 없음·검출 실패)인지.
 * @param {{text?: string}|null|undefined} item
 * @returns {'no_speech'|'not_detected'|null}
 */
function markerKind(item) {
  const text = String(item?.text ?? '').trim();
  if (text === NO_SPEECH_TEXT) return 'no_speech';
  if (text === DETECTION_FAILED_TEXT) return 'not_detected';
  return null;
}

/**
 * 결과 배열을 병실 판정 입력 하나로 요약한다.
 *
 * - 결과 한 건 + 병실 묶음 phrase → {kind:'phrase', accepted, command, runnerUp}. accepted는 phrase.accepted === true일 때만.
 * - 전부 [발화 없음] 또는 [검출 실패 구간] → {kind:'no_speech'} (입 움직임이 안 보임).
 * - 여러 창·일부 검출 실패·정렬 불가(짧음)·다른 묶음·모르는 ID → {kind:'no_phrase', reason}.
 *
 * @param {Array<object>|null|undefined} results 서버 results.
 * @param {{byId: Map<string, object>}} index buildCommandIndex 결과.
 * @returns {{kind:'phrase', accepted:boolean, command:object, runnerUp:object|null, score:number|null, margin:number|null}
 *   |{kind:'no_speech', reason:'no_speech'|'not_detected'}
 *   |{kind:'no_phrase', reason:string}}
 */
export function classifyWardResults(results, index) {
  try {
    const items = Array.isArray(results) ? results.filter((x) => x && typeof x === 'object') : [];
    if (!items.length) return { kind: 'no_phrase', reason: 'empty' };
    const markers = items.map(markerKind);
    if (markers.every((m) => m !== null)) {
      return { kind: 'no_speech', reason: markers.every((m) => m === 'no_speech') ? 'no_speech' : 'not_detected' };
    }
    if (items.length > 1) {
      return { kind: 'no_phrase', reason: markers.includes('not_detected') ? 'partial_detection' : 'multi_window' };
    }
    const phrase = items[0].phrase;
    if (!phrase || typeof phrase !== 'object' || typeof phrase.id !== 'string') return { kind: 'no_phrase', reason: 'unaligned' };
    // 묶음 표시는 반드시 'ward'여야 한다. 표시가 없는 옛 서버 결과나 문장 모드 결과는 병실 명령으로 해석하지 않는다.
    if (phrase.set !== WARD_PHRASE_SET) return { kind: 'no_phrase', reason: 'other_set' };
    const command = commandById(index, phrase.id);
    if (!command) return { kind: 'no_phrase', reason: 'other_set' };
    const runnerUp = commandById(index, phrase.runner_up_id);
    const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
    return {
      kind: 'phrase',
      accepted: phrase.accepted === true,
      command,
      runnerUp: runnerUp && runnerUp.id !== command.id ? runnerUp : null,
      score: num(phrase.score),
      margin: num(phrase.margin),
    };
  } catch {
    return { kind: 'no_phrase', reason: 'empty' };
  }
}
