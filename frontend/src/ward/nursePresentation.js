// nursePresentation.js — 간호사 화면 표시용 순수 함수 (node --test로 검사). 필터는 저장된 카드와 경보음 판단을 바꾸지 않는다.
import { isAlarmKind, sortNewestFirst, unackedSince } from './wardStore.js';

/** 카드를 확인한 직후 이 시간(ms) 안의 확인 누름은 무시한다 (두 번 누르기가 자리를 옮겨 온 다음 카드를 확인하지 않게). */
export const ACK_MERGE_MS = 700;

/**
 * 이 확인 누름을 받아도 되는지.
 * @param {number|null} lastAckAt 마지막으로 카드를 확인한 시각(ms).
 * @param {number} now
 * @param {number} [clickDetail] 클릭 이벤트의 detail (2 이상이면 두 번 누르기의 뒤쪽 누름).
 * @returns {boolean}
 */
export function ackClickAllowed(lastAckAt, now, clickDetail) {
  if (typeof clickDetail === 'number' && clickDetail > 1) return false;
  if (lastAckAt === null || lastAckAt === undefined) return true;
  const gap = now - lastAckAt;
  return !(gap >= 0 && gap < ACK_MERGE_MS);
}

export const NURSE_FILTERS = Object.freeze([
  { id: 'open', label: '미확인' },
  { id: 'all', label: '전체' },
  { id: 'acked', label: '확인 완료' },
]);

/**
 * 카드 개수 요약 (같은 요청을 합친 카드는 하나로 센다).
 * @param {Array<object>} cards
 * @returns {{total:number, open:number, acked:number, alarms:number}}
 */
export function nurseSummary(cards) {
  return (Array.isArray(cards) ? cards : []).reduce((sum, card) => {
    sum.total += 1;
    if (card.acked) sum.acked += 1;
    else {
      sum.open += 1;
      if (isAlarmKind(card.kind)) sum.alarms += 1;
    }
    return sum;
  }, { total: 0, open: 0, acked: 0, alarms: 0 });
}

/**
 * 필터에 맞는 카드 (보이기만 바꾸고 저장된 카드·경보음 판단은 바꾸지 않는다).
 * 확인하지 않은 경보 카드(긴급·긴급 호출·긴급 가능성)를 맨 위에, 확인한 카드를 맨 아래에 두고, 각 묶음은 최신순이다.
 * @param {Array<object>} cards
 * @param {'open'|'all'|'acked'} filter
 * @returns {Array<object>}
 */
export function visibleNurseCards(cards, filter) {
  const shown = sortNewestFirst(Array.isArray(cards) ? cards : []).filter((card) => (
    filter === 'open' ? !card.acked : filter === 'acked' ? card.acked : true
  ));
  // 확인하지 않은 경보 → 확인하지 않은 나머지 → 확인한 카드 (각 묶음 안에서는 최신순).
  const rank = (card) => (card.acked ? 2 : isAlarmKind(card.kind) ? 0 : 1);
  return [0, 1, 2].flatMap((r) => shown.filter((card) => rank(card) === r));
}

/**
 * 확인을 기다린 시간 글자 (마지막 반복이 아니라 처음 확인 안 된 시각부터).
 * @param {object} card
 * @param {number} now
 * @returns {string}
 */
export function waitingText(card, now) {
  const seconds = Math.max(0, Math.floor((now - unackedSince(card)) / 1000));
  if (!Number.isFinite(seconds)) return '확인 대기';
  if (seconds < 60) return `${seconds}초 대기`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}분 ${seconds % 60}초 대기`;
  return `${Math.floor(seconds / 3600)}시간 ${Math.floor((seconds % 3600) / 60)}분 대기`;
}
