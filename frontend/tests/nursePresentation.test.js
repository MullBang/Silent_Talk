import test from 'node:test';
import assert from 'node:assert/strict';
import { ACK_MERGE_MS, ackClickAllowed, nurseSummary, visibleNurseCards, waitingText } from '../src/ward/nursePresentation.js';
import { hasOpenAlarm } from '../src/ward/wardStore.js';

test('card counts do not count merged requests as separate cards', () => {
  const cards = [
    { id: 'a', kind: 'request', acked: false, count: 3, lastAt: 10 },
    { id: 'b', kind: 'emergency_possible', acked: false, count: 1, lastAt: 30 },
    { id: 'c', kind: 'emergency', acked: true, count: 2, lastAt: 20 },
  ];
  assert.deepEqual(nurseSummary(cards), { total: 3, open: 2, acked: 1, alarms: 1 });
  const before = JSON.stringify(cards);
  assert.deepEqual(visibleNurseCards(cards, 'open').map((c) => c.id), ['b', 'a']);
  assert.deepEqual(visibleNurseCards(cards, 'acked').map((c) => c.id), ['c']);
  assert.deepEqual(visibleNurseCards(cards, 'all').map((c) => c.id), ['b', 'a', 'c'], 'open alarm, open request, then acked');
  assert.equal(JSON.stringify(cards), before);
  assert.equal(hasOpenAlarm(cards), true, 'hiding alarms does not alter the full collection');
});

test('wait time uses first unacknowledged time, not the latest repeat', () => {
  const card = { unackedSince: 1000, firstAt: 1000, lastAt: 61000 };
  assert.equal(waitingText(card, 66000), '1분 5초 대기');
  assert.equal(waitingText(card, 3666000), '1시간 1분 대기');
  assert.equal(waitingText(card, 0), '0초 대기');
  assert.equal(waitingText({}, 1000), '확인 대기');
});

test('unacknowledged alarm cards come first, then newest first', () => {
  const cards = [
    { id: 'old-alarm', kind: 'emergency', acked: false, lastAt: 10 },
    { id: 'new-request', kind: 'request', acked: false, lastAt: 50 },
    { id: 'acked-alarm', kind: 'emergency_call', acked: true, lastAt: 60 },
    { id: 'possible', kind: 'emergency_possible', acked: false, lastAt: 40 },
  ];
  assert.deepEqual(visibleNurseCards(cards, 'open').map((c) => c.id), ['possible', 'old-alarm', 'new-request']);
  assert.deepEqual(visibleNurseCards(cards, 'all').map((c) => c.id), ['possible', 'old-alarm', 'new-request', 'acked-alarm']);
  assert.deepEqual(visibleNurseCards(cards, 'acked').map((c) => c.id), ['acked-alarm']);
});

test('a double click on 확인 cannot acknowledge the next card that slides into the same place', () => {
  assert.equal(ackClickAllowed(null, 1000, 1), true, 'first click');
  assert.equal(ackClickAllowed(null, 1000, 2), false, 'second half of a double click (detail 2)');
  assert.equal(ackClickAllowed(1000, 1000 + 120, 1), false, 'a click right after an ack');
  assert.equal(ackClickAllowed(1000, 1000 + ACK_MERGE_MS, 1), true, 'a deliberate later click');
  assert.equal(ackClickAllowed(1000, 1000 + 120, 0), false, 'keyboard Enter right after an ack is merged too');
  assert.equal(ackClickAllowed(1000, 1000 + ACK_MERGE_MS + 1, 0), true);
});

test('tab title shows the open count and flashes only while an alarm is unacknowledged', async () => {
  const { nurseTitle, NURSE_BASE_TITLE, notificationFor } = await import('../src/ward/nursePresentation.js');
  assert.equal(nurseTitle({ open: 0, alarms: 0 }), NURSE_BASE_TITLE);
  assert.equal(nurseTitle({ open: 2, alarms: 0 }, { flash: true }), `(2) ${NURSE_BASE_TITLE}`, 'no flash without an alarm');
  assert.equal(nurseTitle({ open: 2, alarms: 1 }, { flash: false }), `(2) ${NURSE_BASE_TITLE}`);
  assert.match(nurseTitle({ open: 2, alarms: 1 }, { flash: true }), /경보 1건/);
  const urgent = notificationFor({ id: 'c1', kind: 'emergency', room: '305', count: 1 }, { label: '[긴급]', headline: '숨쉬기가 답답해요' });
  assert.equal(urgent.urgent, true);
  assert.match(urgent.title, /305호 \[긴급\]/);
  assert.equal(urgent.body, '숨쉬기가 답답해요');
  const normal = notificationFor({ id: 'c2', kind: 'request', room: '412', count: 3 }, { label: '', headline: '물을 마시고 싶어요' });
  assert.equal(normal.urgent, false);
  assert.equal(normal.title, '412호 일반 요청 (×3)');
  assert.equal(normal.tag, 'c2');
  const cancel = notificationFor({ id: 'c3', kind: 'request', room: '305', count: 1, cancelRequested: true, text: '물을 마시고 싶어요' },
    { label: '', headline: '물을 마시고 싶어요' });
  assert.equal(cancel.title, '305호 취소 요청');
  assert.match(cancel.body, /취소했어요: 물을 마시고 싶어요/);
  assert.equal(cancel.urgent, false);
});
