// wardChannel.js — 환자 창(#ward)과 간호사 창(#nurse) 사이 메시지 형식과 BroadcastChannel 래퍼 (node --test로 검사).
// BroadcastChannel은 주소(scheme·host·port)가 같은 창끼리만 통한다. 두 창은 반드시 같은 주소(localhost:5173)로 연다.
//
// 메시지 공통: {app:'silent-talk-ward', v:1, type, from:'ward'|'nurse', at(ms)}
//   CARD_UPSERT  ward→nurse  {card}            카드 새로 보냄·갱신 (card.id와 card.rev로 구분)
//   CARD_ACK     nurse→ward  {cardIds, room}   간호사 [확인] → 환자 화면 "간호사가 확인했어요"
//   SWITCH_PRESS nurse→ward  {repeat}          간호사 창에 들어온 스위치(키) 입력을 환자 창으로 넘김
//   ROOM_STATE   ward→nurse  {room, state, phase, windowId}  병실 번호·가상 병실·환자 화면 단계 (windowId: 환자 창 구분)
//                                          바뀔 때마다 + ROOM_HEARTBEAT_MS마다 보낸다. 간호사 창은 ROOM_STALE_MS 동안 소식이 없는 줄을 지운다.
//   ROOM_CLOSED  ward→nurse  {windowId}        환자 창이 닫히거나 새로 고침됨 → 간호사 창이 그 창의 병실 줄을 바로 지움
//   HELLO        nurse→ward  {}                간호사 창이 열림 → 환자 창이 병실 상태와 확인 안 된 카드를 다시 보냄

import { CARD_KINDS } from './wardMachine.js';

export const WARD_CHANNEL_NAME = 'silent-talk-ward';
export const WARD_APP_ID = 'silent-talk-ward';
export const WARD_PROTOCOL_VERSION = 1;
export const MESSAGE_TYPES = Object.freeze(['CARD_UPSERT', 'CARD_ACK', 'SWITCH_PRESS', 'ROOM_STATE', 'ROOM_CLOSED', 'HELLO']);
export const SENDERS = Object.freeze(['ward', 'nurse']);
/** 환자 창이 병실 상태를 다시 알리는 간격(ms). */
export const ROOM_HEARTBEAT_MS = 10000;
/**
 * 간호사 창이 이 시간(ms) 동안 소식이 없는 환자 창의 병실 줄을 지운다. 창이 가려져 브라우저가 타이머를
 * 1분에 한 번으로 늦추는 경우에도 줄이 깜빡이지 않도록 심장 박동 간격보다 넉넉히 둔다.
 */
export const ROOM_STALE_MS = 90000;

const isStr = (v) => typeof v === 'string' && v.length > 0;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * 카드 형식 검사 (간호사 창이 받아들이는 최소 조건).
 * @param {any} card
 * @returns {boolean}
 */
export function isValidCard(card) {
  return !!card && typeof card === 'object' && isStr(card.id) && Number.isInteger(card.rev) && card.rev >= 1
    && CARD_KINDS.includes(card.kind) && isStr(card.room) && typeof card.text === 'string'
    && isNum(card.createdAt) && isNum(card.updatedAt)
    && (card.candidates === undefined || Array.isArray(card.candidates));
}

/**
 * 메시지를 만든다.
 * @param {string} type MESSAGE_TYPES 중 하나.
 * @param {object} body 종류별 내용.
 * @param {{from: 'ward'|'nurse', at?: number}} meta
 * @returns {object}
 */
export function makeMessage(type, body, { from, at = Date.now() } = {}) {
  if (!MESSAGE_TYPES.includes(type)) throw new Error(`모르는 메시지 종류: ${type}`);
  if (!SENDERS.includes(from)) throw new Error(`모르는 보낸 쪽: ${from}`);
  return { ...(body || {}), app: WARD_APP_ID, v: WARD_PROTOCOL_VERSION, type, from, at };
}

/**
 * 받은 메시지를 검사한다. 다른 프로그램·다른 판 형식·깨진 내용이면 null.
 * @param {any} data MessageEvent.data
 * @returns {object|null}
 */
export function parseMessage(data) {
  try {
    if (!data || typeof data !== 'object') return null;
    if (data.app !== WARD_APP_ID || data.v !== WARD_PROTOCOL_VERSION) return null;
    if (!MESSAGE_TYPES.includes(data.type) || !SENDERS.includes(data.from) || !isNum(data.at)) return null;
    switch (data.type) {
      case 'CARD_UPSERT':
        return data.from === 'ward' && isValidCard(data.card) ? data : null;
      case 'CARD_ACK':
        return data.from === 'nurse' && Array.isArray(data.cardIds) && data.cardIds.length > 0 && data.cardIds.every(isStr)
          ? data : null;
      case 'SWITCH_PRESS':
        return data.from === 'nurse' ? data : null;
      case 'ROOM_STATE':
        return data.from === 'ward' && isStr(data.room) && !!data.state && typeof data.state === 'object'
          && (data.windowId === undefined || isStr(data.windowId)) ? data : null;
      case 'ROOM_CLOSED':
        return data.from === 'ward' && isStr(data.windowId) ? data : null;
      case 'HELLO':
        return data.from === 'nurse' ? data : null;
      default:
        return null;
    }
  } catch {
    return null;
  }
}

/**
 * 병실 채널을 연다. BroadcastChannel이 없으면(오래된 브라우저) 보내기가 아무 일도 하지 않는 채널을 돌려준다.
 * @param {(message: object) => void} onMessage 검사를 통과한 메시지만 받는다.
 * @param {{from: 'ward'|'nurse', factory?: (name: string) => any, name?: string}} opts
 * @returns {{available: boolean, post: (type: string, body?: object) => boolean, close: () => void}}
 */
export function openWardChannel(onMessage, { from, factory, name = WARD_CHANNEL_NAME } = {}) {
  let channel = null;
  try {
    const make = factory || ((n) => new globalThis.BroadcastChannel(n));
    if (factory || typeof globalThis.BroadcastChannel === 'function') channel = make(name);
  } catch {
    channel = null;
  }
  const listener = (event) => {
    const msg = parseMessage(event?.data);
    if (!msg || msg.from === from) return; // 같은 쪽(예: 환자 창 두 개)이 보낸 것은 무시
    try {
      onMessage(msg);
    } catch {
      /* 화면 처리 오류가 채널을 멈추지 않게 한다 */
    }
  };
  try {
    channel?.addEventListener('message', listener);
  } catch {
    channel = null;
  }
  return {
    available: !!channel,
    post(type, body) {
      try {
        if (!channel) return false;
        channel.postMessage(makeMessage(type, body, { from }));
        return true;
      } catch {
        return false;
      }
    },
    close() {
      try {
        channel?.removeEventListener('message', listener);
        channel?.close();
      } catch {
        /* 이미 닫힘 */
      }
      channel = null;
    },
  };
}
