// wardStore.js — 간호사 화면(#nurse) 카드 목록 순수 함수와 localStorage 저장 (node --test로 검사).
// 같은 요청이 확인 전에 다시 오면 카드 하나로 합치고 횟수를 센다. 합친 카드는 원래 카드 ID들(ids)을 모두 기억해
// 나중에 온 갱신(예: W20 취소 요청)과 [확인] 알림이 어느 ID로 와도 같은 카드에 닿게 한다.
// 같은 카드의 같은 판(rev)이 다시 오면 무시한다(환자 창이 다시 보낸 카드가 확인 표시를 지우지 않게).

import { ROOM_STALE_MS, isValidCard } from './wardChannel.js';
import { ALARM_KINDS } from './wardMachine.js';

export const NURSE_STORAGE_KEY = 'silent-talk-ward-nurse-cards-v1';
export const NURSE_PREFS_KEY = 'silent-talk-ward-nurse-prefs-v1';
export const MAX_NURSE_CARDS = 200;
/** 이 시간 동안 확인하지 않은 카드는 강조한다. */
export const OVERDUE_MS = 60000;

/** 카드 내용 중 환자 창이 바꿀 수 있는 필드. 종류(kind)·병실·처음 시각은 바꾸지 않는다. */
const CONTENT_FIELDS = ['text', 'commandId', 'candidates', 'reason', 'hint', 'note', 'cancelRequested', 'selectedId', 'source', 'enroll',
  'updatedAt'];

/**
 * 경보 카드(반복음)인지.
 * @param {string} kind
 * @returns {boolean}
 */
export function isAlarmKind(kind) {
  return ALARM_KINDS.includes(kind);
}

/**
 * 합칠 수 있는 카드의 열쇠. 후보가 있는 '확인 필요'·'긴급 가능성'처럼 환자 선택으로 바뀌는 카드는 합치지 않는다.
 * @param {object} card
 * @returns {string|null}
 */
export function mergeKey(card) {
  if (!card) return null;
  if (card.kind === 'request' || card.kind === 'emergency' || card.kind === 'emergency_call') {
    return `${card.room}|${card.kind}|${card.commandId || ''}`;
  }
  if (card.kind === 'check' && card.reason === 'no_mouth') return `${card.room}|check|no_mouth`;
  return null;
}

/** 오래된 카드부터 버린다 (확인한 카드 먼저). */
function trim(cards) {
  if (cards.length <= MAX_NURSE_CARDS) return cards;
  const out = [...cards];
  while (out.length > MAX_NURSE_CARDS) {
    let victim = -1;
    for (let i = out.length - 1; i >= 0; i -= 1) {
      if (out[i].acked) { victim = i; break; }
    }
    out.splice(victim >= 0 ? victim : out.length - 1, 1);
  }
  return out;
}

/**
 * 받은 카드를 목록에 넣거나 갱신한다.
 * @param {Array<object>} cards 간호사 카드 목록 (최신이 앞).
 * @param {object} incoming CARD_UPSERT의 card.
 * @param {number} now 받은 시각(ms).
 * @returns {{cards: Array<object>, alert: 'new'|'merged'|'updated'|null, alarm: boolean, cardId: string|null}}
 *   alert가 null이면 바뀐 것이 없다(알림음 없음).
 */
export function upsertNurseCard(cards, incoming, now) {
  const list = Array.isArray(cards) ? cards : [];
  try {
    if (!isValidCard(incoming)) return { cards: list, alert: null, alarm: false, cardId: null };
    const at = Number.isFinite(now) ? now : Date.now();
    const idx = list.findIndex((c) => c.ids.includes(incoming.id));
    if (idx >= 0) {
      const cur = list[idx];
      if (incoming.rev <= (cur.revs[incoming.id] || 0)) return { cards: list, alert: null, alarm: false, cardId: cur.id };
      const content = {};
      for (const f of CONTENT_FIELDS) if (incoming[f] !== undefined) content[f] = incoming[f];
      // 확인한 카드가 다시 바뀌면 지금부터 확인 대기, 확인 전 카드면 처음 기다리기 시작한 시각을 그대로 둔다.
      const next = {
        ...cur, ...content, revs: { ...cur.revs, [incoming.id]: incoming.rev }, lastAt: at, acked: false, ackedAt: null,
        unackedSince: cur.acked ? at : unackedSince(cur),
      };
      const out = [...list];
      out[idx] = next;
      return { cards: out, alert: 'updated', alarm: isAlarmKind(next.kind), cardId: next.id };
    }
    const key = mergeKey(incoming);
    const mIdx = key ? list.findIndex((c) => !c.acked && !c.cancelRequested && mergeKey(c) === key) : -1;
    if (mIdx >= 0) {
      const cur = list[mIdx];
      const next = {
        ...cur, ids: [...cur.ids, incoming.id], revs: { ...cur.revs, [incoming.id]: incoming.rev }, count: cur.count + 1, lastAt: at,
      };
      const out = [...list];
      out[mIdx] = next;
      return { cards: out, alert: 'merged', alarm: isAlarmKind(next.kind), cardId: next.id };
    }
    const card = {
      ...incoming,
      candidates: Array.isArray(incoming.candidates) ? incoming.candidates : [],
      ids: [incoming.id],
      revs: { [incoming.id]: incoming.rev },
      count: 1,
      firstAt: at,
      lastAt: at,
      unackedSince: at,
      acked: false,
      ackedAt: null,
    };
    return { cards: trim([card, ...list]), alert: 'new', alarm: isAlarmKind(card.kind), cardId: card.id };
  } catch {
    return { cards: list, alert: null, alarm: false, cardId: null };
  }
}

/**
 * 간호사 [확인]. 합친 카드면 원래 ID들을 모두 돌려준다(환자 창에 보낼 cardIds).
 * @param {Array<object>} cards
 * @param {string} cardId 목록 카드의 id.
 * @param {number} now
 * @returns {{cards: Array<object>, ackedIds: string[], room: string|null}}
 */
export function ackNurseCard(cards, cardId, now) {
  const list = Array.isArray(cards) ? cards : [];
  try {
    const idx = list.findIndex((c) => c.id === cardId);
    if (idx < 0 || list[idx].acked) return { cards: list, ackedIds: [], room: null };
    const cur = list[idx];
    const out = [...list];
    out[idx] = { ...cur, acked: true, ackedAt: Number.isFinite(now) ? now : Date.now() };
    return { cards: out, ackedIds: [...cur.ids], room: cur.room };
  } catch {
    return { cards: list, ackedIds: [], room: null };
  }
}

/**
 * 확인한 카드를 목록에서 뺀다 (리허설 정리용).
 * @param {Array<object>} cards
 * @returns {Array<object>}
 */
export function clearAcked(cards) {
  return (Array.isArray(cards) ? cards : []).filter((c) => !c.acked);
}

/**
 * 최근에 온 것(마지막 도착·갱신 시각)부터.
 * @param {Array<object>} cards
 * @returns {Array<object>}
 */
export function sortNewestFirst(cards) {
  return [...(Array.isArray(cards) ? cards : [])].sort((a, b) => b.lastAt - a.lastAt);
}

/**
 * 카드가 확인을 기다리기 시작한 시각. 같은 요청이 합쳐져도(다시 와도) 앞당기거나 미루지 않는다.
 * 예전 저장 형식(unackedSince 없음)은 처음 도착 시각을 쓴다.
 * @param {object} card
 * @returns {number}
 */
export function unackedSince(card) {
  if (Number.isFinite(card?.unackedSince)) return card.unackedSince;
  if (Number.isFinite(card?.firstAt)) return card.firstAt;
  return Number.isFinite(card?.lastAt) ? card.lastAt : Number.NaN;
}

/**
 * 60초 넘게 확인하지 않은 카드인지. 확인을 기다리기 시작한 때부터 잰다(아무도 오지 않아 같은 요청을 되풀이한
 * 카드가 바로 강조돼야 하므로 마지막 도착 시각부터 다시 세지 않는다).
 * @param {object} card
 * @param {number} now
 * @returns {boolean}
 */
export function isOverdue(card, now) {
  if (!card || card.acked || !Number.isFinite(now)) return false;
  const since = unackedSince(card);
  return Number.isFinite(since) && now - since >= OVERDUE_MS;
}

/**
 * 확인하지 않은 경보 카드가 있는지 (반복음).
 * @param {Array<object>} cards
 * @returns {boolean}
 */
export function hasOpenAlarm(cards) {
  return (Array.isArray(cards) ? cards : []).some((c) => !c.acked && isAlarmKind(c.kind));
}

/**
 * 카드 앞에 붙는 글자 표시 (색만으로 구분하지 않는다).
 * @param {object} card
 * @returns {string}
 */
export function cardLabel(card) {
  switch (card?.kind) {
    case 'emergency': return '[긴급]';
    case 'emergency_call': return '[긴급 호출]';
    case 'emergency_possible': return '[긴급 가능성]';
    case 'check': return card.selectedId ? '[환자 선택]' : '[확인 필요]';
    default: return '';
  }
}

/**
 * 카드 본문 큰 글씨. '확인 필요'는 본문 없이 회색 '추정' 후보만 보이고, 환자가 고르면 고른 요청이 본문이 된다.
 * @param {object} card
 * @returns {string}
 */
export function cardHeadline(card) {
  switch (card?.kind) {
    case 'check':
      if (card.selectedId) return card.text || '';
      return card.reason === 'no_mouth' ? '입 인식 안 됨' : '';
    case 'emergency_possible':
      return '확인 필요';
    default:
      return card?.text || '';
  }
}

/** 간호사 카드의 내 입 모양 등록 표시 문구 (card.enroll). */
export const ENROLL_NOTE_TEXT = Object.freeze({
  chosen: '등록한 입 모양으로 고름',
  alarm: '등록한 입 모양이 긴급 명령과 비슷해서 알림',
});

/**
 * 카드가 내 입 모양 등록으로 골랐거나(등록 거리) 등록 때문에 긴급 가능성이 됐으면 그 짧은 설명, 아니면 null.
 * 환자가 스캐닝으로 직접 고른 카드는 등록 표시를 하지 않는다.
 * @param {object} card
 * @returns {string|null}
 */
export function cardEnrollNote(card) {
  try {
    if (!card || card.selectedId) return null;
    return ENROLL_NOTE_TEXT[card.enroll] || null;
  } catch {
    return null;
  }
}

/**
 * ms → 'HH:MM' (로컬 시각).
 * @param {number} ms
 * @returns {string}
 */
export function hhmm(ms) {
  const d = new Date(Number.isFinite(ms) ? ms : Date.now());
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/**
 * 저장된 내용을 카드 목록으로 되살린다. 형식이 틀린 카드는 버린다.
 * @param {string|null|undefined} raw localStorage 문자열.
 * @returns {Array<object>}
 */
export function restoreNurseCards(raw) {
  try {
    if (!raw) return [];
    const data = JSON.parse(raw);
    const list = Array.isArray(data?.cards) ? data.cards : [];
    return trim(list.filter((c) => isValidCard(c) && Array.isArray(c.ids) && c.ids.length > 0 && c.ids.every((x) => typeof x === 'string')
      && c.revs && typeof c.revs === 'object' && Number.isInteger(c.count) && c.count >= 1
      && Number.isFinite(c.lastAt) && typeof c.acked === 'boolean'));
  } catch {
    return [];
  }
}

/**
 * 카드 목록을 저장 문자열로.
 * @param {Array<object>} cards
 * @returns {string}
 */
export function serializeNurseCards(cards) {
  return JSON.stringify({ v: 1, cards: Array.isArray(cards) ? cards : [] });
}

/**
 * 저장소에서 카드 목록을 읽는다 (없거나 막혔으면 빈 목록).
 * @param {{getItem: (k:string) => string|null}|null|undefined} storage
 * @returns {Array<object>}
 */
export function loadNurseCards(storage) {
  try {
    return restoreNurseCards(storage?.getItem(NURSE_STORAGE_KEY));
  } catch {
    return [];
  }
}

/**
 * 카드 목록을 저장한다. 실패해도 화면은 그대로 동작한다.
 * @param {{setItem: (k:string, v:string) => void}|null|undefined} storage
 * @param {Array<object>} cards
 * @returns {boolean}
 */
export function saveNurseCards(storage, cards) {
  try {
    if (!storage) return false;
    storage.setItem(NURSE_STORAGE_KEY, serializeNurseCards(cards));
    return true;
  } catch {
    return false;
  }
}

/**
 * 환자 창의 병실 줄을 넣거나 바꾼다 (간호사 화면 '병실 상태', 환자 창마다 한 줄).
 * @param {Record<string, object>} rooms windowId(없으면 병실 번호) → {room, state, phase, at, seenAt}.
 * @param {object} msg ROOM_STATE 메시지.
 * @param {number} now 받은 시각(ms). 오래된 줄 지우기에 쓴다.
 * @returns {Record<string, object>}
 */
export function noteRoomState(rooms, msg, now) {
  const base = rooms && typeof rooms === 'object' ? rooms : {};
  try {
    const key = typeof msg?.windowId === 'string' && msg.windowId ? msg.windowId : msg?.room;
    if (typeof key !== 'string' || !key) return base;
    const seenAt = Number.isFinite(now) ? now : Date.now();
    return { ...base, [key]: { room: msg.room, state: msg.state, phase: msg.phase, at: msg.at, seenAt } };
  } catch {
    return base;
  }
}

/**
 * 닫힌 환자 창의 병실 줄을 지운다 (ROOM_CLOSED).
 * @param {Record<string, object>} rooms
 * @param {string} windowId
 * @returns {Record<string, object>} 지울 줄이 없으면 같은 객체.
 */
export function dropRoom(rooms, windowId) {
  const base = rooms && typeof rooms === 'object' ? rooms : {};
  try {
    if (typeof windowId !== 'string' || !Object.prototype.hasOwnProperty.call(base, windowId)) return base;
    const next = { ...base };
    delete next[windowId];
    return next;
  } catch {
    return base;
  }
}

/**
 * 한동안 소식이 없는 환자 창의 병실 줄을 지운다 (창이 닫힘 알림 없이 사라졌을 때).
 * @param {Record<string, object>} rooms
 * @param {number} now
 * @param {number} [staleMs]
 * @returns {Record<string, object>} 지운 줄이 없으면 같은 객체.
 */
export function pruneRooms(rooms, now, staleMs = ROOM_STALE_MS) {
  const base = rooms && typeof rooms === 'object' ? rooms : {};
  try {
    if (!Number.isFinite(now)) return base;
    const keep = Object.entries(base).filter(([, info]) => Number.isFinite(info?.seenAt) && now - info.seenAt < staleMs);
    return keep.length === Object.keys(base).length ? base : Object.fromEntries(keep);
  } catch {
    return base;
  }
}

/**
 * 간호사 화면 설정 (스위치 입력을 환자 창으로 넘길지).
 * @param {{getItem: (k:string) => string|null}|null|undefined} storage
 * @returns {{forwardSwitch: boolean}}
 */
export function loadNursePrefs(storage) {
  try {
    const raw = storage?.getItem(NURSE_PREFS_KEY);
    const data = raw ? JSON.parse(raw) : {};
    return { forwardSwitch: data?.forwardSwitch !== false };
  } catch {
    return { forwardSwitch: true };
  }
}

/**
 * 간호사 화면 설정 저장.
 * @param {{setItem: (k:string, v:string) => void}|null|undefined} storage
 * @param {{forwardSwitch: boolean}} prefs
 * @returns {boolean}
 */
export function saveNursePrefs(storage, prefs) {
  try {
    if (!storage) return false;
    storage.setItem(NURSE_PREFS_KEY, JSON.stringify({ forwardSwitch: prefs?.forwardSwitch !== false }));
    return true;
  } catch {
    return false;
  }
}
