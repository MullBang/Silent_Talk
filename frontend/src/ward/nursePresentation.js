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

/** 간호사 창 기본 제목. */
export const NURSE_BASE_TITLE = '간호사실 — 병실도우미';

/**
 * 간호사 창(탭) 제목. 확인하지 않은 카드 수를 앞에 붙이고, 확인 안 한 경보가 있고 다른 창을 보고 있으면
 * flash 차례마다 경보 문구와 번갈아 보인다 (탭만 보여도 알 수 있게).
 * @param {{open:number, alarms:number}} summary nurseSummary 결과.
 * @param {{flash?: boolean}} [opts] flash가 true인 차례에 경보 문구를 보인다.
 * @returns {string}
 */
export function nurseTitle(summary, { flash = false } = {}) {
  const open = Number(summary?.open) || 0;
  const alarms = Number(summary?.alarms) || 0;
  if (alarms > 0 && flash) return `🔴 경보 ${alarms}건 확인 필요 — 간호사실`;
  return open > 0 ? `(${open}) ${NURSE_BASE_TITLE}` : NURSE_BASE_TITLE;
}

/**
 * 새로 오거나 다시 온 카드의 브라우저 알림 내용.
 * @param {object} card 간호사 카드.
 * @param {{label: string, headline: string}} text cardLabel·cardHeadline 결과.
 * @returns {{title: string, body: string, urgent: boolean, tag: string}}
 */
export function notificationFor(card, text) {
  const room = card?.room ? `${card.room}호` : '병실';
  // 환자가 방금 요청을 취소하면 같은 카드가 갱신되어 온다: 새 요청이 아니라 취소 요청으로 알린다.
  if (card?.cancelRequested) {
    return {
      title: `${room} 취소 요청`,
      body: `환자가 이 요청을 취소했어요: ${text?.headline || card?.text || '요청'}`,
      urgent: false,
      tag: String(card?.id || 'ward-card'),
    };
  }
  const urgent = isAlarmKind(card?.kind);
  const label = text?.label || '일반 요청';
  const headline = text?.headline || '';
  const repeat = Number(card?.count) > 1 ? ` (×${card.count})` : '';
  return {
    title: `${urgent ? '🔴 ' : ''}${room} ${label}${repeat}`,
    body: headline || '간호사 화면에서 확인해 주세요',
    urgent,
    tag: String(card?.id || 'ward-card'),
  };
}
