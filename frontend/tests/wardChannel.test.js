// wardChannel.test.js — 환자 창·간호사 창 메시지 형식과 BroadcastChannel 래퍼 검사 (node --test, Node의 BroadcastChannel 사용).
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  MESSAGE_TYPES, ROOM_HEARTBEAT_MS, ROOM_STALE_MS, WARD_APP_ID, WARD_CHANNEL_NAME, WARD_PROTOCOL_VERSION, isValidCard,
  makeMessage, openWardChannel, parseMessage,
} from '../src/ward/wardChannel.js';

const card = (over = {}) => ({
  id: 'w1-1', rev: 1, room: '305', kind: 'request', commandId: 'W05', text: '물을 마시고 싶어요', candidates: [],
  createdAt: 1000, updatedAt: 1000, ...over,
});

/** 메시지 하나를 기다린다 (없으면 null). */
function nextMessage(received, ms = 500) {
  return new Promise((resolve) => {
    const started = Date.now();
    const poll = () => {
      if (received.length) resolve(received.shift());
      else if (Date.now() - started > ms) resolve(null);
      else setTimeout(poll, 5);
    };
    poll();
  });
}

test('메시지 형식: 앱 ID·판 번호·종류·보낸 쪽·시각', () => {
  assert.equal(WARD_CHANNEL_NAME, 'silent-talk-ward');
  assert.equal(WARD_PROTOCOL_VERSION, 1);
  assert.deepEqual(MESSAGE_TYPES, ['CARD_UPSERT', 'CARD_ACK', 'SWITCH_PRESS', 'ROOM_STATE', 'ROOM_CLOSED', 'HELLO']);
  assert.equal(ROOM_HEARTBEAT_MS, 10000);
  assert.ok(ROOM_STALE_MS >= 60000 + ROOM_HEARTBEAT_MS, '가려진 창의 1분 타이머 지연에도 줄이 지워지지 않게');
  const m = makeMessage('CARD_UPSERT', { card: card() }, { from: 'ward', at: 42 });
  assert.equal(m.app, WARD_APP_ID);
  assert.equal(m.v, 1);
  assert.equal(m.type, 'CARD_UPSERT');
  assert.equal(m.from, 'ward');
  assert.equal(m.at, 42);
  assert.deepEqual(parseMessage(m), m);
  assert.throws(() => makeMessage('NOPE', {}, { from: 'ward' }));
  assert.throws(() => makeMessage('HELLO', {}, { from: 'doctor' }));
  // 본문이 공통 필드를 덮어쓰지 못한다.
  assert.equal(makeMessage('HELLO', { v: 99, app: 'x' }, { from: 'nurse', at: 1 }).v, 1);
});

test('parseMessage: 종류별 보낸 쪽과 필수 내용을 검사한다', () => {
  const ok = (type, body, from) => parseMessage(makeMessage(type, body, { from, at: 1 }));
  assert.ok(ok('CARD_UPSERT', { card: card() }, 'ward'));
  assert.equal(ok('CARD_UPSERT', { card: card() }, 'nurse'), null, '카드는 환자 창만 보낸다');
  assert.equal(ok('CARD_UPSERT', { card: card({ kind: 'nope' }) }, 'ward'), null);
  assert.equal(ok('CARD_UPSERT', { card: card({ rev: 0 }) }, 'ward'), null);
  assert.equal(ok('CARD_UPSERT', { card: { id: 'x' } }, 'ward'), null);
  assert.ok(ok('CARD_ACK', { cardIds: ['w1-1', 'w1-2'], room: '305' }, 'nurse'));
  assert.equal(ok('CARD_ACK', { cardIds: [] }, 'nurse'), null);
  assert.equal(ok('CARD_ACK', { cardIds: [3] }, 'nurse'), null);
  assert.equal(ok('CARD_ACK', { cardIds: ['a'] }, 'ward'), null);
  assert.ok(ok('SWITCH_PRESS', { repeat: false }, 'nurse'));
  assert.equal(ok('SWITCH_PRESS', {}, 'ward'), null);
  assert.ok(ok('ROOM_STATE', { room: '305', state: { light: 'on' }, phase: 'idle' }, 'ward'));
  assert.equal(ok('ROOM_STATE', { room: '', state: {} }, 'ward'), null);
  assert.ok(ok('ROOM_STATE', { room: '305', state: {}, phase: 'idle', windowId: 'wab12' }, 'ward'));
  assert.equal(ok('ROOM_STATE', { room: '305', state: {}, windowId: 7 }, 'ward'), null);
  assert.ok(ok('ROOM_CLOSED', { windowId: 'wab12' }, 'ward'));
  assert.equal(ok('ROOM_CLOSED', {}, 'ward'), null, '어느 창인지 없으면 버림');
  assert.equal(ok('ROOM_CLOSED', { windowId: 'wab12' }, 'nurse'), null, '환자 창만 보낸다');
  assert.ok(ok('HELLO', {}, 'nurse'));
  const base = makeMessage('HELLO', {}, { from: 'nurse', at: 1 });
  assert.equal(parseMessage({ ...base, v: 2 }), null, '다른 판');
  assert.equal(parseMessage({ ...base, app: 'other' }), null, '다른 프로그램');
  assert.equal(parseMessage({ ...base, at: 'x' }), null);
  assert.equal(parseMessage(null), null);
  assert.equal(parseMessage('text'), null);
});

test('isValidCard: 병실·시각·후보 형식', () => {
  assert.equal(isValidCard(card()), true);
  assert.equal(isValidCard(card({ room: '' })), false);
  assert.equal(isValidCard(card({ createdAt: Number.NaN })), false);
  assert.equal(isValidCard(card({ candidates: 'x' })), false);
  assert.equal(isValidCard(card({ candidates: undefined })), true);
});

test('BroadcastChannel: 환자 창 카드 → 간호사 창, 간호사 확인 → 환자 창, 같은 쪽 메시지·다른 프로그램은 무시', async () => {
  const name = `ward-test-${process.pid}-${Date.now()}`;
  const wardGot = [];
  const nurseGot = [];
  const ward = openWardChannel((m) => wardGot.push(m), { from: 'ward', name });
  const nurse = openWardChannel((m) => nurseGot.push(m), { from: 'nurse', name });
  const ward2 = openWardChannel(() => {}, { from: 'ward', name });
  const raw = new BroadcastChannel(name);
  try {
    assert.equal(ward.available, true);
    assert.equal(ward.post('CARD_UPSERT', { card: card() }), true);
    const m1 = await nextMessage(nurseGot);
    assert.equal(m1.type, 'CARD_UPSERT');
    assert.equal(m1.card.id, 'w1-1');
    assert.equal(await nextMessage(wardGot, 100), null, '다른 환자 창의 카드는 환자 창이 받지 않는다');

    nurse.post('CARD_ACK', { cardIds: ['w1-1'], room: '305' });
    const m2 = await nextMessage(wardGot);
    assert.equal(m2.type, 'CARD_ACK');
    assert.deepEqual(m2.cardIds, ['w1-1']);

    nurse.post('SWITCH_PRESS', { repeat: false });
    assert.equal((await nextMessage(wardGot)).type, 'SWITCH_PRESS');

    ward2.post('ROOM_STATE', { room: '305', state: { light: 'on' }, phase: 'idle' });
    assert.equal((await nextMessage(nurseGot)).type, 'ROOM_STATE');
    ward2.post('ROOM_CLOSED', { windowId: 'w2' });
    const closed = await nextMessage(nurseGot);
    assert.equal(closed.type, 'ROOM_CLOSED');
    assert.equal(closed.windowId, 'w2');

    raw.postMessage({ app: 'other', v: 1, type: 'HELLO', from: 'nurse', at: 1 });
    raw.postMessage({ ...makeMessage('HELLO', {}, { from: 'nurse', at: 1 }), v: 9 });
    assert.equal(await nextMessage(wardGot, 100), null, '형식이 다른 메시지는 버린다');
  } finally {
    ward.close();
    ward2.close();
    nurse.close();
    raw.close();
  }
  assert.equal(ward.post('HELLO', {}), false, '닫힌 채널');
});

test('BroadcastChannel을 만들 수 없으면 보내기만 실패하는 빈 채널', () => {
  const ch = openWardChannel(() => {}, { from: 'ward', factory: () => { throw new Error('no'); } });
  assert.equal(ch.available, false);
  assert.equal(ch.post('HELLO', {}), false);
  ch.close();
});

test('받는 쪽 처리 오류가 채널을 멈추지 않는다', () => {
  let listener = null;
  const fake = { addEventListener: (_t, fn) => { listener = fn; }, removeEventListener() {}, postMessage() {}, close() {} };
  let calls = 0;
  const ch = openWardChannel(() => { calls += 1; throw new Error('화면 오류'); }, { from: 'ward', factory: () => fake });
  const msg = makeMessage('HELLO', {}, { from: 'nurse', at: 1 });
  assert.doesNotThrow(() => listener({ data: msg }));
  assert.doesNotThrow(() => listener({ data: msg }));
  assert.equal(calls, 2);
  ch.close();
});
