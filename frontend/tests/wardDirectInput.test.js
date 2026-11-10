import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { buildCommandIndex } from '../src/ward/wardCommands.js';
import {
  PRESS_MERGE_MS, SCAN_MAX_ROUNDS, SCAN_STEP_MS, createWardState, scanDeadline, scanItemKey, scanToken, stepWard,
} from '../src/ward/wardMachine.js';
import { practiceResults } from '../src/ward/wardPractice.js';
import { normalizeSettings } from '../src/ward/wardSettings.js';

const index = buildCommandIndex(JSON.parse(readFileSync(new URL('../src/ward/commands.json', import.meta.url))));
const t = 100000;
const makeScan = (inputMode = 'direct') => stepWard(createWardState({ index, settings: { inputMode }, sessionKey: 'direct' }),
  { type: 'SIMULATE', at: t, results: practiceResults('uncertain_normal', index) }).state;
const click = (s, key, at = t + 100, token = scanToken(s)) => stepWard(s,
  { type: 'SELECT_ITEM', itemKey: key, scanToken: token, at });

test('direct click chooses the named item rather than the highlighted item and updates the existing card', () => {
  const s = makeScan();
  assert.equal(s.scan.items[s.scan.index].commandId, 'W11');
  const out = click(s, 'command:W13');
  assert.equal(out.state.screen.commandId, 'W13');
  assert.equal(out.state.cardOrder.length, 1);
  assert.equal(out.effects.filter(e => e.type === 'SEND_CARD').length, 0);
  assert.equal(out.effects.filter(e => e.type === 'UPDATE_CARD').length, 1);
  assert.equal(out.state.cards[out.state.cardOrder[0]].selectedId, 'W13');
});

test('selection remains attached to the clicked item across a scan-highlight timer boundary', () => {
  const s = makeScan();
  const out = click(s, 'command:W11', t + 2100);
  assert.equal(out.state.screen.commandId, 'W11');
  assert.equal(out.state.phase, 'idle');
});

test('an old scan token, unknown key, and a duplicate click cannot send a request', () => {
  const s = makeScan();
  assert.deepEqual(click(s, 'command:W13', t + 100, 'old-token').effects, []);
  assert.deepEqual(click(s, 'command:W99').effects, []);
  const out = click(s, 'command:W13');
  assert.deepEqual(click(out.state, 'command:W13', t + 101, scanToken(s)).effects, []);
});

test('moving between direct selection steps creates a new token and does not count toward triple-press emergency', () => {
  const s = makeScan();
  const categories = click(s, 'none:').state;
  assert.equal(categories.scan.mode, 'categories');
  assert.notEqual(scanToken(categories), scanToken(s));
  assert.deepEqual(click(categories, 'none:', t + 101, scanToken(s)).effects, []);
  const category = categories.scan.items.find(x => x.key === 'category' && x.categoryId !== 'emergency');
  const commands = click(categories, scanItemKey(category), t + 700).state;
  const normal = commands.scan.items.find(x => x.key === 'command' && !x.emergency);
  const out = click(commands, scanItemKey(normal), t + 1300);
  assert.equal(out.state.phase, 'idle');
  assert.deepEqual(out.state.pressTimes, []);
  assert.equal(Object.values(out.state.cards).some(c => c.kind === 'emergency_call'), false);
});

test('direct clicks are ignored after scan expiry and in the existing switch mode', () => {
  assert.deepEqual(click(makeScan('switch'), 'command:W13').effects, []);
  const expired = click(makeScan(), 'command:W13', t + 24001);
  assert.equal(expired.state.phase, 'idle');
  assert.equal(Object.values(expired.state.cards).some(c => c.selectedId === 'W13'), false);
});

test('legacy settings retain switch mode; direct mode is explicit and persisted by normalization', () => {
  assert.equal(normalizeSettings({}).inputMode, 'switch');
  assert.equal(normalizeSettings({ inputMode: 'direct' }).inputMode, 'direct');
  assert.equal(normalizeSettings({ inputMode: 'unknown' }).inputMode, 'switch');
});

test('a double click on a category does not pick the same-position item of the next list (emergency W02 included)', () => {
  const s = makeScan();
  const categories = click(s, 'none:').state;
  const emergency = categories.scan.items.find(x => x.key === 'category' && x.categoryId === 'emergency');
  const commands = click(categories, scanItemKey(emergency), t + 700).state;
  assert.equal(commands.scan.mode, 'commands');
  const first = commands.scan.items[0];
  const second = click(commands, scanItemKey(first), t + 700 + 120);
  assert.deepEqual(second.effects, [], 'second click of a double click is merged');
  assert.equal(second.state.phase, 'scan');
  const later = click(commands, scanItemKey(first), t + 700 + PRESS_MERGE_MS + 10);
  assert.equal(later.effects.some(e => e.type === 'SEND_CARD'), true, 'a deliberate later click still works');
});

test('in direct mode a key press during a scan never picks the hidden highlighted item; three presses still call for help', () => {
  const s = makeScan();
  const one = stepWard(s, { type: 'PRESS', at: t + 2600 });
  assert.equal(one.state.phase, 'scan');
  assert.deepEqual(one.effects, []);
  assert.equal(Object.values(one.state.cards).some(c => c.selectedId), false);
  const two = stepWard(one.state, { type: 'PRESS', at: t + 3300 });
  const three = stepWard(two.state, { type: 'PRESS', at: t + 4000 });
  assert.equal(Object.values(three.state.cards).some(c => c.kind === 'emergency_call'), true);
  assert.equal(Object.values(three.state.cards).some(c => c.selectedId), false, 'no request was sent on the way');
  // switch mode keeps the old behaviour: one press picks the highlighted item
  const sw = stepWard(makeScan('switch'), { type: 'PRESS', at: t + 100 });
  assert.equal(sw.state.phase, 'idle');
});

test('screen buttons send a PRESS intent that is ignored once the step has changed', () => {
  const confirm = stepWard(createWardState({ index, settings: {}, sessionKey: 'intent' }),
    { type: 'SIMULATE', at: t, results: practiceResults('accepted_normal', index) }).state;
  assert.equal(confirm.phase, 'confirm');
  const late = stepWard(confirm, { type: 'PRESS', intent: 'cancel', at: confirm.confirm.endsAt + 40 });
  assert.equal(late.state.phase, 'idle', 'the countdown already sent the card');
  assert.equal(late.effects.some(e => e.type === 'START_RECORDING'), false, '전송 취소 must not start a recording');
  const inTime = stepWard(confirm, { type: 'PRESS', intent: 'cancel', at: t + 1000 });
  assert.equal(inTime.state.screen.kind, 'cancelled');
  assert.deepEqual(stepWard(confirm, { type: 'PRESS', intent: 'start', at: t + 1000 }).effects, []);
});

test('scanDeadline matches the moment the scan gives up', () => {
  const s = makeScan();
  const d = scanDeadline(s.scan);
  assert.equal(d.endsAt - d.startedAt, SCAN_MAX_ROUNDS * s.scan.items.length * SCAN_STEP_MS);
  assert.equal(stepWard(s, { type: 'TICK', at: d.endsAt - 1 }).state.phase, 'scan');
  assert.equal(stepWard(s, { type: 'TICK', at: d.endsAt }).state.phase, 'idle');
  const mid = stepWard(s, { type: 'TICK', at: t + 5000 }).state;
  assert.equal(scanDeadline(mid.scan).endsAt, d.endsAt, 'deadline does not move while the highlight advances');
  assert.equal(scanDeadline(null), null);
});
