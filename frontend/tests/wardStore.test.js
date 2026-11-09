// wardStore.test.js — 간호사 화면 카드 목록(합치기·확인·60초 강조·저장 복원) 검사 (node --test).
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ROOM_HEARTBEAT_MS, ROOM_STALE_MS } from '../src/ward/wardChannel.js';
import {
  MAX_NURSE_CARDS, NURSE_PREFS_KEY, NURSE_STORAGE_KEY, OVERDUE_MS, ackNurseCard, cardHeadline, cardLabel, clearAcked,
  dropRoom, hasOpenAlarm, hhmm, isOverdue, loadNurseCards, loadNursePrefs, mergeKey, noteRoomState, pruneRooms,
  restoreNurseCards, saveNurseCards, saveNursePrefs, serializeNurseCards, sortNewestFirst, unackedSince, upsertNurseCard,
} from '../src/ward/wardStore.js';

const T = 2_000_000;
let n = 0;
const card = (over = {}) => {
  n += 1;
  return {
    id: `w-${n}`, rev: 1, room: '305', kind: 'request', commandId: 'W05', text: '물을 마시고 싶어요', candidates: [],
    reason: null, hint: null, note: null, cancelRequested: false, selectedId: null, source: 'lip',
    createdAt: T, updatedAt: T, acked: false, ackedAt: null, ...over,
  };
};

/** 메모리 저장소. */
function memoryStorage() {
  const data = new Map();
  return { getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => data.set(k, String(v)), data };
}
const throwingStorage = { getItem() { throw new Error('막힘'); }, setItem() { throw new Error('막힘'); } };

test('새 카드: 맨 앞에 넣고 알림(일반은 alarm=false, 긴급은 true)', () => {
  const a = upsertNurseCard([], card(), T);
  assert.equal(a.alert, 'new');
  assert.equal(a.alarm, false);
  assert.equal(a.cards.length, 1);
  assert.equal(a.cards[0].count, 1);
  assert.deepEqual(a.cards[0].ids, [a.cards[0].id]);
  const b = upsertNurseCard(a.cards, card({ kind: 'emergency', commandId: 'W02', text: '숨쉬기가 답답해요' }), T + 1);
  assert.equal(b.alarm, true);
  assert.equal(b.cards[0].kind, 'emergency');
  for (const kind of ['emergency_call', 'emergency_possible']) assert.equal(upsertNurseCard([], card({ kind }), T).alarm, true);
  assert.equal(upsertNurseCard([], { id: 'bad' }, T).alert, null, '형식이 틀린 카드는 무시');
});

test('같은 요청이 확인 전에 다시 오면 합치고 횟수를 센다', () => {
  const first = card();
  const second = card();
  let r = upsertNurseCard([], first, T);
  r = upsertNurseCard(r.cards, second, T + 5000);
  assert.equal(r.alert, 'merged');
  assert.equal(r.cards.length, 1);
  assert.equal(r.cards[0].count, 2);
  assert.deepEqual(r.cards[0].ids, [first.id, second.id]);
  assert.equal(r.cards[0].lastAt, T + 5000);
  // 다른 요청·다른 병실은 합치지 않는다.
  assert.equal(upsertNurseCard(r.cards, card({ commandId: 'W06', text: '배변' }), T).cards.length, 2);
  assert.equal(upsertNurseCard(r.cards, card({ room: '306' }), T).cards.length, 2);
  // 긴급 호출도 확인 전에는 합친다.
  let e = upsertNurseCard([], card({ kind: 'emergency_call', commandId: null }), T);
  e = upsertNurseCard(e.cards, card({ kind: 'emergency_call', commandId: null }), T + 1);
  assert.equal(e.cards[0].count, 2);
});

test('확인한 카드·취소 요청 카드·후보가 있는 확인 필요 카드에는 합치지 않는다', () => {
  const a = upsertNurseCard([], card(), T);
  const acked = ackNurseCard(a.cards, a.cards[0].id, T + 10).cards;
  assert.equal(upsertNurseCard(acked, card(), T + 20).cards.length, 2);
  const cancelled = upsertNurseCard([], card({ cancelRequested: true, note: '환자가 취소 요청함' }), T).cards;
  assert.equal(upsertNurseCard(cancelled, card(), T + 20).cards.length, 2);
  const cand = [{ id: 'W11', text: '불을 켜 주세요' }];
  const chk = upsertNurseCard([], card({ kind: 'check', commandId: null, candidates: cand }), T).cards;
  assert.equal(upsertNurseCard(chk, card({ kind: 'check', commandId: null, candidates: cand }), T).cards.length, 2);
  assert.equal(mergeKey(card({ kind: 'check', commandId: null, candidates: cand })), null);
  assert.equal(mergeKey(card({ kind: 'check', commandId: null, reason: 'no_mouth' })), '305|check|no_mouth');
  assert.equal(mergeKey(card({ kind: 'emergency_possible' })), null);
});

test('갱신: 같은 ID의 새 판(rev)이면 내용을 바꾸고 다시 확인 대기, 같은 판이 또 오면 무시', () => {
  const c = card({ kind: 'check', commandId: null, text: '확인 필요', candidates: [{ id: 'W11', text: '불을 켜 주세요' }] });
  let r = upsertNurseCard([], c, T);
  r = { cards: ackNurseCard(r.cards, c.id, T + 5).cards };
  const sel = { ...c, rev: 2, selectedId: 'W11', commandId: 'W11', text: '불을 켜 주세요', note: '환자가 직접 고름', updatedAt: T + 9 };
  const u = upsertNurseCard(r.cards, sel, T + 10);
  assert.equal(u.alert, 'updated');
  assert.equal(u.cards.length, 1);
  assert.equal(u.cards[0].selectedId, 'W11');
  assert.equal(u.cards[0].acked, false, '바뀐 카드는 다시 확인해야 한다');
  assert.equal(u.cards[0].kind, 'check', '종류는 바꾸지 않는다');
  const again = upsertNurseCard(u.cards, sel, T + 20);
  assert.equal(again.alert, null);
  assert.equal(again.cards, u.cards);
  const old = upsertNurseCard(u.cards, c, T + 30);
  assert.equal(old.alert, null, '옛 판은 무시');
});

test('합친 카드의 다른 ID로 온 갱신(W20 취소 요청)도 같은 카드에 닿는다', () => {
  const a = card();
  const b = card();
  let r = upsertNurseCard([], a, T);
  r = upsertNurseCard(r.cards, b, T + 1);
  const u = upsertNurseCard(r.cards, { ...b, rev: 2, cancelRequested: true, note: '환자가 취소 요청함', updatedAt: T + 2 }, T + 2);
  assert.equal(u.alert, 'updated');
  assert.equal(u.cards.length, 1);
  assert.equal(u.cards[0].cancelRequested, true);
  assert.equal(u.cards[0].count, 2);
});

test('확인: 합친 카드의 원래 ID를 모두 돌려주고, 두 번 확인하지 않는다', () => {
  const a = card();
  const b = card();
  let r = upsertNurseCard([], a, T);
  r = upsertNurseCard(r.cards, b, T + 1);
  const ack = ackNurseCard(r.cards, r.cards[0].id, T + 100);
  assert.deepEqual(ack.ackedIds, [a.id, b.id]);
  assert.equal(ack.room, '305');
  assert.equal(ack.cards[0].acked, true);
  assert.equal(ack.cards[0].ackedAt, T + 100);
  assert.deepEqual(ackNurseCard(ack.cards, r.cards[0].id, T + 200).ackedIds, []);
  assert.deepEqual(ackNurseCard(ack.cards, 'nope', T).ackedIds, []);
});

test('60초 동안 확인하지 않으면 강조, 긴급 반복음 여부', () => {
  const r = upsertNurseCard([], card({ kind: 'emergency', commandId: 'W02' }), T);
  const c = r.cards[0];
  assert.equal(isOverdue(c, T + OVERDUE_MS - 1), false);
  assert.equal(isOverdue(c, T + OVERDUE_MS), true);
  assert.equal(hasOpenAlarm(r.cards), true);
  const acked = ackNurseCard(r.cards, c.id, T + 1).cards;
  assert.equal(isOverdue(acked[0], T + OVERDUE_MS * 2), false);
  assert.equal(hasOpenAlarm(acked), false);
});

test('60초 강조는 확인을 기다리기 시작한 때부터: 같은 요청을 되풀이해 합쳐져도 늦춰지지 않는다', () => {
  // 아무도 오지 않아 같은 요청이 T, T+50초, T+100초에 와도 T부터 60초가 지나면 강조
  let m = upsertNurseCard([], card(), T);
  m = upsertNurseCard(m.cards, card(), T + 50000);
  assert.equal(isOverdue(m.cards[0], T + OVERDUE_MS - 1), false);
  assert.equal(isOverdue(m.cards[0], T + OVERDUE_MS), true);
  m = upsertNurseCard(m.cards, card(), T + 100000);
  assert.equal(m.cards[0].count, 3);
  assert.equal(m.cards[0].lastAt, T + 100000, '목록 순서는 마지막 도착 시각');
  assert.equal(unackedSince(m.cards[0]), T);
  assert.equal(isOverdue(m.cards[0], T + 140000), true);

  // 확인 전 카드가 바뀌어도(환자가 후보를 고름) 처음 기다린 시각 그대로
  const c = card({ kind: 'check', commandId: null, text: '확인 필요', candidates: [{ id: 'W11', text: '불을 켜 주세요' }] });
  let u = upsertNurseCard([], c, T);
  u = upsertNurseCard(u.cards, { ...c, rev: 2, selectedId: 'W11', commandId: 'W11', updatedAt: T + 30000 }, T + 30000);
  assert.equal(isOverdue(u.cards[0], T + OVERDUE_MS), true);

  // 확인한 카드가 다시 바뀌면(W20 취소 요청 등) 그때부터 다시 60초
  const a = card();
  let k = upsertNurseCard([], a, T);
  k = { cards: ackNurseCard(k.cards, a.id, T + 5).cards };
  k = upsertNurseCard(k.cards, { ...a, rev: 2, cancelRequested: true, note: '환자가 취소 요청함', updatedAt: T + 100000 }, T + 100000);
  assert.equal(k.cards[0].acked, false);
  assert.equal(isOverdue(k.cards[0], T + 100000 + OVERDUE_MS - 1), false);
  assert.equal(isOverdue(k.cards[0], T + 100000 + OVERDUE_MS), true);

  // 예전 저장 형식(unackedSince 없음)은 처음 도착 시각부터
  const legacy = { ...upsertNurseCard([], card(), T).cards[0] };
  delete legacy.unackedSince;
  assert.equal(isOverdue(legacy, T + OVERDUE_MS), true);
  assert.equal(isOverdue(restoreNurseCards(serializeNurseCards([legacy]))[0], T + OVERDUE_MS), true);
  assert.equal(isOverdue(null, T), false);
  assert.equal(isOverdue(legacy, Number.NaN), false);
});

test('정렬(최근 도착 먼저)·글자 표시·본문', () => {
  let r = upsertNurseCard([], card(), T);
  r = upsertNurseCard(r.cards, card({ commandId: 'W11', text: '불을 켜 주세요' }), T + 10);
  assert.equal(sortNewestFirst(r.cards)[0].commandId, 'W11');
  assert.equal(cardLabel({ kind: 'emergency' }), '[긴급]');
  assert.equal(cardLabel({ kind: 'emergency_call' }), '[긴급 호출]');
  assert.equal(cardLabel({ kind: 'emergency_possible' }), '[긴급 가능성]');
  assert.equal(cardLabel({ kind: 'check' }), '[확인 필요]');
  assert.equal(cardLabel({ kind: 'check', selectedId: 'W05' }), '[환자 선택]');
  assert.equal(cardLabel({ kind: 'request' }), '');
  assert.equal(cardHeadline({ kind: 'check', text: '확인 필요', candidates: [{}] }), '');
  assert.equal(cardHeadline({ kind: 'check', text: '확인 필요(입 인식 안 됨)', reason: 'no_mouth' }), '입 인식 안 됨');
  assert.equal(cardHeadline({ kind: 'check', text: '물을 마시고 싶어요', selectedId: 'W05' }), '물을 마시고 싶어요');
  assert.equal(cardHeadline({ kind: 'emergency_possible', text: '긴급 가능성, 확인 필요' }), '확인 필요');
  assert.equal(cardHeadline({ kind: 'request', text: '네' }), '네');
  assert.match(hhmm(T), /^\d\d:\d\d$/);
});

test('저장·복원: 새로 고침 뒤에도 카드와 확인 상태가 남는다', () => {
  const store = memoryStorage();
  let r = upsertNurseCard([], card(), T);
  r = upsertNurseCard(r.cards, card({ kind: 'emergency', commandId: 'W02' }), T + 1);
  const acked = ackNurseCard(r.cards, r.cards[1].id, T + 2).cards;
  assert.equal(saveNurseCards(store, acked), true);
  assert.ok(store.data.has(NURSE_STORAGE_KEY));
  const back = loadNurseCards(store);
  assert.deepEqual(back, acked);
  // 복원한 목록에 같은 카드(같은 판)가 다시 와도 확인 상태를 지우지 않는다.
  const resend = upsertNurseCard(back, { ...r.cards[1], acked: false }, T + 3);
  assert.equal(resend.alert, null);
  assert.equal(clearAcked(back).length, 1);
});

test('복원: 깨진 내용·막힌 저장소·형식이 틀린 카드는 버린다', () => {
  assert.deepEqual(restoreNurseCards(null), []);
  assert.deepEqual(restoreNurseCards('{깨짐'), []);
  assert.deepEqual(restoreNurseCards(JSON.stringify({ cards: [{ id: 'x' }] })), []);
  const good = upsertNurseCard([], card(), T).cards[0];
  const mixed = serializeNurseCards([good, { ...good, id: 'y', ids: [] }, { ...good, id: 'z', count: 0 }]);
  assert.equal(restoreNurseCards(mixed).length, 1);
  assert.deepEqual(loadNurseCards(throwingStorage), []);
  assert.equal(saveNurseCards(throwingStorage, [good]), false);
  assert.equal(saveNurseCards(null, [good]), false);
  assert.deepEqual(loadNurseCards(undefined), []);
});

test('카드는 최대 수를 넘으면 확인한 오래된 카드부터 버린다', () => {
  let list = [];
  for (let i = 0; i < MAX_NURSE_CARDS; i += 1) list = upsertNurseCard(list, card({ commandId: `X${i}` }), T + i).cards;
  const oldest = list[list.length - 1];
  const acked = ackNurseCard(list, list[5].id, T).cards;
  const more = upsertNurseCard(acked, card({ commandId: 'NEW' }), T + 999).cards;
  assert.equal(more.length, MAX_NURSE_CARDS);
  assert.ok(!more.some((c) => c.id === list[5].id), '확인한 카드를 먼저 버림');
  assert.ok(more.some((c) => c.id === oldest.id));
});

test('병실 줄: 환자 창마다 한 줄, 닫힘 알림이면 바로 지우고 소식이 끊기면 지운다', () => {
  assert.ok(ROOM_HEARTBEAT_MS < ROOM_STALE_MS, '심장 박동이 지우기 기준보다 자주 온다');
  const msg = (windowId, room = '305', phase = 'idle') => ({ type: 'ROOM_STATE', room, state: { light: 'off' }, phase, windowId, at: T });
  let rooms = noteRoomState({}, msg('wa'), T);
  rooms = noteRoomState(rooms, msg('wb', '306'), T + 1000);
  assert.deepEqual(Object.keys(rooms), ['wa', 'wb']);
  assert.equal(rooms.wa.seenAt, T);
  // 같은 창이 다시 알리면 그 줄만 바뀐다 (병실 번호를 바꿔도 줄 하나).
  rooms = noteRoomState(rooms, msg('wa', '412', 'recording'), T + 2000);
  assert.equal(Object.keys(rooms).length, 2);
  assert.equal(rooms.wa.room, '412');
  assert.equal(rooms.wa.phase, 'recording');
  // windowId가 없는 옛 형식은 병실 번호로 묶는다.
  assert.deepEqual(Object.keys(noteRoomState({}, { room: '305', state: {} }, T)), ['305']);
  assert.equal(noteRoomState(rooms, { room: '' }, T), rooms, '열쇠가 없으면 그대로');
  // 환자 창 새로 고침: 닫힘 알림으로 옛 줄을 지운다.
  const closed = dropRoom(rooms, 'wa');
  assert.deepEqual(Object.keys(closed), ['wb']);
  assert.equal(dropRoom(closed, 'wa'), closed, '없는 줄이면 같은 객체');
  // 소식이 끊긴 줄은 ROOM_STALE_MS 뒤에 지운다. 지울 것이 없으면 같은 객체.
  assert.equal(pruneRooms(rooms, T + 2000 + ROOM_STALE_MS - 1001), rooms);
  const pruned = pruneRooms(rooms, T + 1000 + ROOM_STALE_MS);
  assert.deepEqual(Object.keys(pruned), ['wa'], '1초에 마지막으로 알린 wb만 지움');
  assert.deepEqual(pruneRooms(pruned, T + 2000 + ROOM_STALE_MS), {});
  assert.deepEqual(pruneRooms(null, T), {});
  assert.equal(pruneRooms(rooms, Number.NaN), rooms);
});

test('간호사 화면 설정(스위치 넘기기) 저장', () => {
  const store = memoryStorage();
  assert.deepEqual(loadNursePrefs(store), { forwardSwitch: true });
  assert.equal(saveNursePrefs(store, { forwardSwitch: false }), true);
  assert.ok(store.data.has(NURSE_PREFS_KEY));
  assert.deepEqual(loadNursePrefs(store), { forwardSwitch: false });
  assert.deepEqual(loadNursePrefs(throwingStorage), { forwardSwitch: true });
  assert.equal(saveNursePrefs(throwingStorage, { forwardSwitch: false }), false);
});
