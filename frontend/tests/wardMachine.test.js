// wardMachine.test.js — 병실 무음 도우미 환자 화면 상태도 검사 (node --test). 계획 5·9.2·9.3절의 표를 한 줄씩 확인한다.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { buildCommandIndex } from '../src/ward/wardCommands.js';
import {
  ACK_NOTICE_MS, ANALYSIS_LIMIT_MS, PRESS_MERGE_MS, RECORD_GRACE_MS, SCAN_MAX_ROUNDS, SCAN_STEP_MS, TRIPLE_PRESS_WINDOW_MS,
  WARD_TEXT, createWardState, latestOpenAlarm, stepWard, wardView,
} from '../src/ward/wardMachine.js';

const commands = JSON.parse(readFileSync(new URL('../src/ward/commands.json', import.meta.url), 'utf8'));
const INDEX = buildCommandIndex(commands);
const T0 = 1_000_000;
const EMERGENCY_IDS = ['W02', 'W03', 'W04', 'W10'];

/** 새 상태. */
const fresh = (settings) => createWardState({ index: INDEX, settings, sessionKey: 't' });

/** 이벤트 여러 개를 차례로 넣는다. */
function feed(state, events) {
  let s = state;
  const effects = [];
  for (const e of events) {
    const r = stepWard(s, e);
    s = r.state;
    effects.push(...r.effects);
  }
  return { state: s, effects };
}

/** 서버 결과 한 건 (병실 묶음). */
function phrase(id, runnerUp, accepted) {
  return [{
    start_ms: 0, end_ms: 5000, text: '받아쓰기', confidence: 0.8, raw_score: -1,
    phrase: { id, text: INDEX.byId.get(id).text, score: accepted ? -1.2 : -3.9, margin: accepted ? 1.2 : 0.1, runner_up_id: runnerUp, accepted, set: 'ward' },
  }];
}
const marker = (text) => [{ start_ms: 0, end_ms: 5000, text, confidence: null, raw_score: null }];

/** 누름 → 녹화 끝 → 결과. 결과 시각은 at+6000. */
function recordWith(state, at, results) {
  const r1 = feed(state, [{ type: 'PRESS', at }]);
  const run = r1.state.run;
  const r2 = feed(r1.state, [
    { type: 'RECORDING_DONE', at: at + 5000, run },
    { type: 'RESULT', at: at + 6000, run, results },
  ]);
  return { state: r2.state, effects: [...r1.effects, ...r2.effects] };
}

const sent = (effects) => effects.filter((e) => e.type === 'SEND_CARD').map((e) => e.card);
const updated = (effects) => effects.filter((e) => e.type === 'UPDATE_CARD').map((e) => e.card);

test('대기 → 한 번 누르면 기본 5초 녹화 시작(카운트다운 없음), 테두리 깜빡임', () => {
  const { state, effects } = feed(fresh(), [{ type: 'PRESS', at: T0 }]);
  assert.equal(state.phase, 'recording');
  assert.deepEqual(effects, [{ type: 'START_RECORDING', run: 1, seconds: 5 }]);
  const view = wardView(state, T0 + 100);
  assert.equal(view.flash, true);
  assert.equal(view.status, '● 말씀하세요 (5초)');
  assert.equal(wardView(state, T0 + 3100).status, '● 말씀하세요 (2초)');
});

test('녹화 길이 설정 4/5/6/8초, 그 밖의 값은 기본 5초', () => {
  for (const sec of [4, 5, 6, 8]) {
    const { effects } = feed(fresh({ recordSeconds: sec }), [{ type: 'PRESS', at: T0 }]);
    assert.equal(effects[0].seconds, sec);
  }
  assert.equal(fresh({ recordSeconds: 7 }).settings.recordSeconds, 5);
  assert.equal(fresh({ confirmSeconds: 9 }).settings.confirmSeconds, 6);
  assert.equal(fresh({ confirmSeconds: 4 }).settings.confirmSeconds, 6);
});

test('녹화 중 누름은 녹화를 멈추지 않는다', () => {
  const { state, effects } = feed(fresh(), [
    { type: 'PRESS', at: T0 }, { type: 'PRESS', at: T0 + 1000 }, { type: 'TICK', at: T0 + 4100 }, { type: 'PRESS', at: T0 + 4500 },
  ]);
  assert.equal(state.phase, 'recording');
  assert.equal(effects.filter((e) => e.type === 'START_RECORDING').length, 1);
  assert.equal(sent(effects).length, 0, '3초 안 3번이 아니면 긴급 호출 없음');
});

test('키 반복과 0.5초 안의 연속 입력은 하나로 친다', () => {
  const s1 = feed(fresh(), [{ type: 'PRESS', at: T0 }]).state;
  const r = stepWard(s1, { type: 'PRESS', at: T0 + PRESS_MERGE_MS - 1 });
  assert.equal(r.state, s1, '0.5초 안 → 상태 그대로');
  const rep = stepWard(s1, { type: 'PRESS', at: T0 + 2000, repeat: true });
  assert.equal(rep.state, s1, '키 반복 → 상태 그대로');
  // 대기 상태에서 키 반복만 오면 녹화도 시작하지 않는다.
  assert.equal(stepWard(fresh(), { type: 'PRESS', at: T0, repeat: true }).effects.length, 0);
});

test('3초 안에 3번 누르면(시작 누름 포함) 바로 "긴급 호출(내용 미확인)", 녹화는 이어진다', () => {
  const { state, effects } = feed(fresh(), [
    { type: 'PRESS', at: T0 }, { type: 'PRESS', at: T0 + 900 }, { type: 'PRESS', at: T0 + 1800 },
  ]);
  const cards = sent(effects);
  assert.equal(cards.length, 1);
  assert.equal(cards[0].kind, 'emergency_call');
  assert.equal(cards[0].text, '긴급 호출(내용 미확인)');
  assert.equal(cards[0].source, 'switch');
  assert.equal(state.phase, 'recording');
  assert.ok(latestOpenAlarm(state));
  assert.equal(wardView(state, T0 + 1900).alarm.kind, 'emergency_call');
  // 녹화 결과는 그대로 처리된다.
  const run = state.run;
  const r = feed(state, [{ type: 'RECORDING_DONE', at: T0 + 5000, run }, { type: 'RESULT', at: T0 + 6000, run, results: phrase('W02', 'W04', true) }]);
  assert.equal(sent(r.effects)[0].kind, 'emergency');
});

test('3초 경계: 정확히 3초 안은 긴급, 넘으면 아님. 빠른 연타(0.4초 간격)도 긴급이 된다', () => {
  const edge = feed(fresh(), [{ type: 'PRESS', at: T0 }, { type: 'PRESS', at: T0 + 1500 }, { type: 'PRESS', at: T0 + TRIPLE_PRESS_WINDOW_MS }]);
  assert.equal(sent(edge.effects).length, 1);
  const slow = feed(fresh(), [{ type: 'PRESS', at: T0 }, { type: 'PRESS', at: T0 + 1600 }, { type: 'PRESS', at: T0 + 3200 }]);
  assert.equal(sent(slow.effects).length, 0);
  const taps = [0, 400, 800, 1200, 1600].map((d) => ({ type: 'PRESS', at: T0 + d }));
  const fast = feed(fresh(), taps);
  assert.equal(sent(fast.effects).length, 1, '0.4초 연타: 0/0.8/1.6초 누름이 받아들여져 긴급');
});

test('3초 안 3번은 확인 카운트다운에서 시작해도, 녹화 중에 눌러도 긴급 호출', () => {
  // 확인 중 누름(취소) → 대기에서 누름(녹화 시작) → 녹화 중 누름
  const r = recordWith(fresh(), T0, phrase('W05', 'W06', true));
  const fromConfirm = feed(r.state, [7000, 7600, 8200].map((d) => ({ type: 'PRESS', at: T0 + d })));
  const cards = sent(fromConfirm.effects);
  assert.deepEqual(cards.map((c) => c.kind), ['emergency_call'], '취소한 W05는 보내지 않고 긴급 호출만');
  assert.equal(fromConfirm.state.phase, 'recording');
  // 녹화가 한참 이어진 뒤의 누름 3번
  const during = feed(fresh(), [0, 3500, 4100, 4700].map((d) => ({ type: 'PRESS', at: T0 + d })));
  assert.deepEqual(sent(during.effects).map((c) => c.kind), ['emergency_call']);
  assert.equal(during.state.phase, 'recording');
});

test('스캐닝에서 명령·다시 말하기를 고른 누름은 3번 누르기에 센다', () => {
  const start = T0 + 6000;
  // 후보1(W11)을 고름 → 대기에서 누름(녹화) → 누름 = 3초 안 3번
  const r = recordWith(fresh(), T0, phrase('W11', 'W13', false));
  const pick = feed(r.state, [10, 610, 1210].map((d) => ({ type: 'PRESS', at: start + d })));
  assert.deepEqual(sent(pick.effects).map((c) => c.kind), ['emergency_call']);
  assert.equal(updated(pick.effects)[0].selectedId, 'W11', '고른 후보는 그대로 갱신');
  // 다시 말하기(녹화 시작) → 녹화 중 두 번
  const t = start + 2 * SCAN_STEP_MS + 10;
  const retry = feed(r.state, [0, 600, 1200].map((d) => ({ type: 'PRESS', at: t + d })));
  assert.deepEqual(sent(retry.effects).map((c) => c.kind), ['emergency_call']);
});

test('스캐닝 단계 이동(둘 다 아님·분류·뒤로)은 3번 누르기에 세지 않는다: 빨리 이어 고른 요청이 긴급 호출로 바뀌지 않음', () => {
  const start = T0 + 6000;
  const r = recordWith(fresh(), T0, phrase('W11', 'W13', false));
  const cardId = sent(r.effects)[0].id;
  // 둘 다 아님(4번째 칸) → 2.1초 뒤 호출·소통(2번째 분류) → 0.6초 뒤 W01(첫 칸): 2.7초 안 3번
  const tn = start + 3 * SCAN_STEP_MS + 10;
  const tc = tn + SCAN_STEP_MS + 100;
  const call = feed(r.state, [{ type: 'PRESS', at: tn }, { type: 'PRESS', at: tc }, { type: 'PRESS', at: tc + 600 }]);
  assert.equal(sent(call.effects).length, 0, '긴급 호출 없음');
  const last = updated(call.effects).at(-1);
  assert.equal(last.id, cardId, '확인 필요 카드를 갱신');
  assert.equal(last.selectedId, 'W01');
  assert.equal(call.state.phase, 'idle');
  assert.equal(wardView(call.state, tc + 700).headline, '간호사를 불러 주세요');
  // 둘 다 아님 → 긴급 분류(바로 강조) → W02: 긴급 호출이 아니라 고른 긴급 카드
  const em = feed(r.state, [{ type: 'PRESS', at: tn }, { type: 'PRESS', at: tn + 600 }, { type: 'PRESS', at: tn + 1200 }]);
  const emCards = sent(em.effects);
  assert.equal(emCards.length, 1);
  assert.equal(emCards[0].kind, 'emergency');
  assert.equal(emCards[0].commandId, 'W02');
  assert.equal(emCards[0].source, 'scan');
  // 분류 → 뒤로 → 분류 → 명령도 마찬가지
  const m = recordWith(fresh(), T0, marker('어'));
  const ta = start + 4 * SCAN_STEP_MS + 10; // 대답 → W17, W18, 뒤로
  const back = feed(m.state, [{ type: 'PRESS', at: ta }, { type: 'PRESS', at: ta + 2 * SCAN_STEP_MS + 10 }]);
  assert.equal(back.state.scan.mode, 'categories');
  const tb = ta + 2 * SCAN_STEP_MS + 10;
  const again = feed(back.state, [{ type: 'PRESS', at: tb + 600 }, { type: 'PRESS', at: tb + 1200 }]); // 긴급 → W02
  assert.deepEqual(sent(again.effects).map((c) => [c.kind, c.commandId]), [['emergency', 'W02']]);
  // 단계 이동 뒤에도 3번을 더 누르면 긴급 호출은 그대로 된다.
  const panic = feed(r.state, [tn, tn + 600, tn + 1200, tn + 1800, tn + 2400].map((at) => ({ type: 'PRESS', at })));
  assert.ok(sent(panic.effects).some((c) => c.kind === 'emergency_call'));
});

test('긴급 호출 뒤 바로 이어진 4번째 누름은 두 번째 긴급 호출을 만들지 않는다', () => {
  const { effects } = feed(fresh(), [0, 700, 1400, 2100].map((d) => ({ type: 'PRESS', at: T0 + d })));
  assert.equal(sent(effects).filter((c) => c.kind === 'emergency_call').length, 1);
});

test('고름 + 일반 → 확인 카운트다운(기본 6초) 뒤 카드 전송', () => {
  const r = recordWith(fresh(), T0, phrase('W05', 'W06', true));
  assert.equal(r.state.phase, 'confirm');
  assert.equal(sent(r.effects).length, 0, '카운트다운 중에는 보내지 않음');
  assert.deepEqual(r.effects.filter((e) => e.type === 'SPEAK'), [{ type: 'SPEAK', text: '물을 마시고 싶어요' }]);
  const v = wardView(r.state, T0 + 6000);
  assert.equal(v.headline, '물을 마시고 싶어요');
  assert.equal(v.sub, '6초 뒤 간호사실로 보냅니다 · 스위치 = 취소');
  assert.equal(wardView(r.state, T0 + 5800).sub, v.sub, '화면 시각이 조금 늦어도 설정(6초)보다 큰 숫자는 없다');
  const before = feed(r.state, [{ type: 'TICK', at: T0 + 6000 + 5999 }]);
  assert.equal(before.state.phase, 'confirm');
  assert.equal(sent(before.effects).length, 0);
  const after = feed(before.state, [{ type: 'TICK', at: T0 + 12000 }]);
  const cards = sent(after.effects);
  assert.equal(cards.length, 1);
  assert.equal(cards[0].kind, 'request');
  assert.equal(cards[0].commandId, 'W05');
  assert.equal(cards[0].text, '물을 마시고 싶어요');
  assert.equal(cards[0].room, '305');
  assert.equal(after.state.phase, 'idle');
  assert.equal(wardView(after.state, T0 + 12000).sub, '간호사실에 보냈어요');
});

test('확인 시간 설정(5~8초)을 따른다', () => {
  const r = recordWith(fresh({ confirmSeconds: 8 }), T0, phrase('W05', 'W06', true));
  assert.equal(sent(feed(r.state, [{ type: 'TICK', at: T0 + 6000 + 7999 }]).effects).length, 0);
  assert.equal(sent(feed(r.state, [{ type: 'TICK', at: T0 + 14000 }]).effects).length, 1);
});

test('카운트다운 중 한 번 누르면 취소(보내지 않음), 이후에도 보내지 않는다', () => {
  const r = recordWith(fresh(), T0, phrase('W05', 'W06', true));
  const c = feed(r.state, [{ type: 'PRESS', at: T0 + 8000 }, { type: 'TICK', at: T0 + 20000 }]);
  assert.equal(sent(c.effects).length, 0);
  assert.equal(c.state.phase, 'idle');
  assert.equal(wardView(c.state, T0 + 20000).headline, '보내지 않았어요');
});

test('카운트다운이 끝난 뒤 늦게 처리된 누름은 취소가 아니라 전송 후 새 녹화', () => {
  const r = recordWith(fresh(), T0, phrase('W05', 'W06', true));
  const c = feed(r.state, [{ type: 'PRESS', at: T0 + 12100 }]);
  assert.equal(sent(c.effects).length, 1);
  assert.equal(c.effects.at(-1).type, 'START_RECORDING');
  assert.equal(c.state.phase, 'recording');
});

test('고름 + 긴급(W02·W03·W04·W10) → 빨간 카드 즉시, 환자가 취소할 수 없다', () => {
  for (const id of EMERGENCY_IDS) {
    const r = recordWith(fresh(), T0, phrase(id, 'W05', true));
    const cards = sent(r.effects);
    assert.equal(cards.length, 1, id);
    assert.equal(cards[0].kind, 'emergency');
    assert.equal(cards[0].commandId, id);
    assert.equal(r.state.phase, 'idle', '확인 카운트다운 없음');
    const v = wardView(r.state, T0 + 6000);
    assert.equal(v.label, '[긴급]');
    assert.equal(v.tone, 'emergency');
    // 이어진 누름: 카드를 고치거나 지우지 않고 새 녹화를 시작한다.
    const p = feed(r.state, [{ type: 'PRESS', at: T0 + 7000 }, { type: 'TICK', at: T0 + 30000 }]);
    assert.equal(updated(p.effects).length, 0);
    assert.equal(p.effects[0].type, 'START_RECORDING');
    assert.equal(p.state.cards[cards[0].id].cancelRequested, false);
  }
});

test('고르지 못함 + 1등 또는 2등이 긴급 → "긴급 가능성, 확인 필요" 즉시(후보 포함)', () => {
  const top = recordWith(fresh(), T0, phrase('W03', 'W05', false));
  const runner = recordWith(fresh(), T0, phrase('W05', 'W04', false));
  for (const r of [top, runner]) {
    const cards = sent(r.effects);
    assert.equal(cards.length, 1);
    assert.equal(cards[0].kind, 'emergency_possible');
    assert.equal(cards[0].text, '긴급 가능성, 확인 필요');
    assert.equal(cards[0].candidates.length, 2);
    assert.equal(r.state.phase, 'idle');
    assert.equal(wardView(r.state, T0 + 6000).label, '[긴급 가능성]');
  }
  assert.deepEqual(runner.state.cards[sent(runner.effects)[0].id].candidates.map((c) => c.id), ['W05', 'W04']);
});

test('고르지 못함 + 일반 → 후보 스캐닝 + 간호사실 "확인 필요"(회색 추정 후보)', () => {
  const r = recordWith(fresh(), T0, phrase('W11', 'W13', false));
  const cards = sent(r.effects);
  assert.equal(cards.length, 1);
  assert.equal(cards[0].kind, 'check');
  assert.deepEqual(cards[0].candidates.map((c) => c.text), ['불을 켜 주세요', '커튼을 닫아 주세요']);
  assert.equal(cards[0].selectedId, null);
  assert.equal(r.state.phase, 'scan');
  const v = wardView(r.state, T0 + 6000);
  assert.deepEqual(v.items.map((x) => x.label), ['불을 켜 주세요', '커튼을 닫아 주세요', '다시 말하기', '둘 다 아님']);
  assert.equal(v.items[0].highlighted, true);
});

test('스캐닝 순서와 시간: 후보1 → 후보2 → 다시 말하기 → 둘 다 아님, 각 2초, 3바퀴 뒤 끝', () => {
  const start = T0 + 6000;
  const r = recordWith(fresh(), T0, phrase('W11', 'W13', false));
  const at = (ms) => wardView(feed(r.state, [{ type: 'TICK', at: start + ms }]).state, start + ms).items.findIndex((x) => x.highlighted);
  assert.equal(at(SCAN_STEP_MS - 1), 0);
  assert.equal(at(SCAN_STEP_MS), 1);
  assert.equal(at(2 * SCAN_STEP_MS), 2);
  assert.equal(at(3 * SCAN_STEP_MS), 3);
  assert.equal(at(4 * SCAN_STEP_MS), 0, '한 바퀴 뒤 처음으로');
  const total = SCAN_MAX_ROUNDS * 4 * SCAN_STEP_MS;
  const notYet = feed(r.state, [{ type: 'TICK', at: start + total - 1 }]);
  assert.equal(notYet.state.phase, 'scan');
  const end = feed(r.state, [{ type: 'TICK', at: start + total }]);
  assert.equal(end.state.phase, 'idle');
  assert.equal(updated(end.effects).length, 1);
  assert.equal(updated(end.effects)[0].note, '환자 선택 없음');
  assert.equal(wardView(end.state, start + total).sub, WARD_TEXT.scanTimeoutNurse);
});

test('스캐닝 선택은 새 카드를 만들지 않고 "확인 필요" 카드를 갱신한다', () => {
  const start = T0 + 6000;
  const r = recordWith(fresh(), T0, phrase('W11', 'W13', false));
  const cardId = sent(r.effects)[0].id;
  // 2초 뒤 후보2(커튼)가 강조될 때 누름. TICK 없이 누름 시각만으로 강조 위치를 따라잡는다.
  const pick = feed(r.state, [{ type: 'PRESS', at: start + SCAN_STEP_MS + 100 }]);
  assert.equal(sent(pick.effects).length, 0);
  const up = updated(pick.effects);
  assert.equal(up.length, 1);
  assert.equal(up[0].id, cardId);
  assert.equal(up[0].selectedId, 'W13');
  assert.equal(up[0].text, '커튼을 닫아 주세요');
  assert.equal(up[0].rev, 2);
  assert.equal(pick.state.room.curtain, 'closed', '가상 커튼 닫힘');
  assert.equal(pick.state.lastNormalCardId, cardId);
  assert.equal(wardView(pick.state, start + 2200).headline, '커튼을 닫아 주세요');
});

test('스캐닝 "다시 말하기" → 카드에 표시하고 새 녹화', () => {
  const start = T0 + 6000;
  const r = recordWith(fresh(), T0, phrase('W11', 'W13', false));
  const p = feed(r.state, [{ type: 'PRESS', at: start + 2 * SCAN_STEP_MS + 10 }]);
  assert.equal(updated(p.effects)[0].note, '환자가 다시 말하기를 고름');
  assert.equal(p.effects.at(-1).type, 'START_RECORDING');
  assert.equal(p.state.phase, 'recording');
});

test('스캐닝 "둘 다 아님" → 분류 스캐닝으로 이어 고르고, 같은 카드를 갱신', () => {
  const start = T0 + 6000;
  const r = recordWith(fresh(), T0, phrase('W11', 'W13', false));
  const cardId = sent(r.effects)[0].id;
  const none = feed(r.state, [{ type: 'PRESS', at: start + 3 * SCAN_STEP_MS + 10 }]);
  assert.equal(none.state.scan.mode, 'categories');
  assert.equal(none.state.scan.cardId, cardId);
  assert.equal(wardView(none.state, start + 6100).reason, '후보 중에 없다고 고르셨어요');
  // 분류: 긴급(0) 호출·소통(1) 몸 돌봄(2) 병실(3) 대답(4) → 몸 돌봄 선택 → 첫 항목 W05
  const t1 = start + 3 * SCAN_STEP_MS + 10;
  const cat = feed(none.state, [{ type: 'PRESS', at: t1 + 2 * SCAN_STEP_MS + 10 }]);
  assert.equal(cat.state.scan.mode, 'commands');
  const t2 = t1 + 2 * SCAN_STEP_MS + 10;
  const pick = feed(cat.state, [{ type: 'PRESS', at: t2 + PRESS_MERGE_MS + 10 }]);
  assert.equal(sent(pick.effects).length, 0);
  const up = updated(pick.effects);
  assert.equal(up[0].id, cardId);
  assert.equal(up[0].selectedId, 'W05');
});

test('지정 문장 결과 없음(짧음·여러 창·일부 검출 실패) → 분류 5개 스캐닝 + 사유', () => {
  const short = [{ start_ms: 0, end_ms: 1200, text: '어', confidence: 0.8, raw_score: -1 }];
  const multi = [{ start_ms: 0, end_ms: 12000, text: '가', confidence: 0.8 }, { start_ms: 12000, end_ms: 20000, text: '나', confidence: 0.8 }];
  const partial = [{ start_ms: 0, end_ms: 3000, text: '가', confidence: 0.8 }, { start_ms: 3000, end_ms: 5000, text: '[검출 실패 구간]' }];
  const cases = [[short, '말이 너무 짧거나 목록 문장과 맞춰 볼 수 없었어요'], [multi, '녹화가 길어 여러 구간으로 나뉘었어요'],
    [partial, '일부 구간에서 입이 안 보였어요']];
  for (const [results, reason] of cases) {
    const r = recordWith(fresh(), T0, results);
    assert.equal(sent(r.effects).length, 0, '간호사 카드 없음');
    assert.equal(r.state.phase, 'scan');
    const v = wardView(r.state, T0 + 6000);
    assert.equal(v.reason, reason);
    assert.deepEqual(v.items.map((x) => x.label), ['긴급', '호출·소통', '몸 돌봄', '병실', '대답', '다시 말하기', '닫기']);
    assert.equal(v.items[0].emergency, true, '긴급 분류는 글자로도 표시');
  }
});

test('분류 스캐닝에서 일반 명령을 고르면 바로 카드(가상 병실 반영), 긴급 명령은 빨간 카드', () => {
  const start = T0 + 6000;
  const r = recordWith(fresh(), T0, marker('어'));
  // 병실(3번째 칸) → W11 불 켜기(첫 칸)
  const cat = feed(r.state, [{ type: 'PRESS', at: start + 3 * SCAN_STEP_MS + 10 }]);
  const t = start + 3 * SCAN_STEP_MS + 10;
  const pick = feed(cat.state, [{ type: 'PRESS', at: t + PRESS_MERGE_MS + 10 }]);
  const cards = sent(pick.effects);
  assert.equal(cards.length, 1);
  assert.equal(cards[0].kind, 'request');
  assert.equal(cards[0].commandId, 'W11');
  assert.equal(cards[0].source, 'scan');
  assert.equal(pick.state.room.light, 'on');

  const em = feed(r.state, [{ type: 'PRESS', at: start + 10 }]); // 긴급 분류
  const em2 = feed(em.state, [{ type: 'PRESS', at: start + 20 + PRESS_MERGE_MS }]); // W02
  assert.equal(sent(em2.effects)[0].kind, 'emergency');
  assert.equal(sent(em2.effects)[0].commandId, 'W02');
});

test('분류 스캐닝 "뒤로"와 "닫기"', () => {
  const start = T0 + 6000;
  const r = recordWith(fresh(), T0, marker('어'));
  const close = feed(r.state, [{ type: 'PRESS', at: start + 6 * SCAN_STEP_MS + 10 }]);
  assert.equal(close.state.phase, 'idle');
  assert.equal(wardView(close.state, start + 13000).headline, '선택을 닫았어요');
  const ans = feed(r.state, [{ type: 'PRESS', at: start + 4 * SCAN_STEP_MS + 10 }]); // 대답 → W17, W18, 뒤로
  const t = start + 4 * SCAN_STEP_MS + 10;
  const back = feed(ans.state, [{ type: 'PRESS', at: t + 2 * SCAN_STEP_MS + 10 }]);
  assert.equal(back.state.scan.mode, 'categories');
});

test('[발화 없음]·검출 실패: 1회는 환자 화면만, 연속 2회부터 간호사실 "확인 필요(입 인식 안 됨)"', () => {
  const first = recordWith(fresh(), T0, marker('[발화 없음]'));
  assert.equal(sent(first.effects).length, 0);
  const v1 = wardView(first.state, T0 + 6000);
  assert.equal(v1.headline, '입 움직임이 안 보였어요. 다시 눌러 주세요');
  assert.equal(v1.sub, null);
  const second = recordWith(first.state, T0 + 10000, marker('[검출 실패 구간]'));
  const cards = sent(second.effects);
  assert.equal(cards.length, 1);
  assert.equal(cards[0].kind, 'check');
  assert.equal(cards[0].reason, 'no_mouth');
  assert.equal(cards[0].text, '확인 필요(입 인식 안 됨)');
  assert.equal(wardView(second.state, T0 + 16000).sub, '간호사실에도 확인을 요청했어요');
});

test('연속이 아니면(사이에 다른 결과) 입 인식 안 됨 카드를 보내지 않는다', () => {
  const a = recordWith(fresh(), T0, marker('[발화 없음]'));
  const b = recordWith(a.state, T0 + 10000, phrase('W05', 'W06', true));
  const b2 = feed(b.state, [{ type: 'PRESS', at: T0 + 17000 }]); // 카운트다운 취소
  const c = recordWith(b2.state, T0 + 20000, marker('[발화 없음]'));
  assert.equal(sent(c.effects).length, 0);
  assert.equal(c.state.noSpeechStreak, 1);
});

test('녹화 실패·분석 오류·분석 시간 초과는 입 인식 안 됨 연속을 끊지 않는다(입이 보였다는 근거가 아님)', () => {
  const first = recordWith(fresh(), T0, marker('[발화 없음]'));
  assert.equal(first.state.noSpeechStreak, 1);
  const noMouthCards = (effects) => sent(effects).filter((c) => c.reason === 'no_mouth');

  // [발화 없음] → 분석 오류(분류 스캐닝, 고르지 않고 끝남) → [발화 없음] = 연속 2회
  const p = feed(first.state, [{ type: 'PRESS', at: T0 + 10000 }]);
  const run = p.state.run;
  const err = feed(p.state, [
    { type: 'RECORDING_DONE', at: T0 + 15000, run }, { type: 'ANALYSIS_ERROR', at: T0 + 16000, run, message: '서버 오류' },
    { type: 'TICK', at: T0 + 16000 + SCAN_MAX_ROUNDS * 7 * SCAN_STEP_MS },
  ]);
  assert.equal(err.state.phase, 'idle');
  assert.equal(err.state.noSpeechStreak, 1);
  assert.equal(noMouthCards(recordWith(err.state, T0 + 70000, marker('[발화 없음]')).effects).length, 1);

  // [발화 없음] → 녹화 실패(카메라) → [검출 실패 구간]
  const q = feed(first.state, [{ type: 'PRESS', at: T0 + 10000 }]);
  const failed = feed(q.state, [{ type: 'RECORD_FAILED', at: T0 + 10010, run: q.state.run, message: '카메라가 준비되지 않았어요' }]);
  assert.equal(failed.state.noSpeechStreak, 1);
  assert.equal(noMouthCards(recordWith(failed.state, T0 + 20000, marker('[검출 실패 구간]')).effects).length, 1);

  // [발화 없음] → 분석 시간 초과 → [발화 없음]
  const t = feed(first.state, [{ type: 'PRESS', at: T0 + 10000 }]);
  const slow = feed(t.state, [
    { type: 'RECORDING_DONE', at: T0 + 15000, run: t.state.run }, { type: 'TICK', at: T0 + 15000 + ANALYSIS_LIMIT_MS + 1 },
  ]);
  assert.equal(slow.state.scan.reason, 'timeout');
  const closed = feed(slow.state, [{ type: 'PRESS', at: T0 + 15000 + ANALYSIS_LIMIT_MS + 1 + 6 * SCAN_STEP_MS + 10 }]); // 닫기
  assert.equal(closed.state.phase, 'idle');
  assert.equal(noMouthCards(recordWith(closed.state, T0 + 200000, marker('[발화 없음]')).effects).length, 1);
});

test('W20 고름 → 마지막 일반 카드를 지우지 않고 "환자가 취소 요청함"을 덧붙인다(긴급 카드에는 적용 안 함)', () => {
  // 일반(W05) 전송 → 긴급(W02) → W20
  const a = recordWith(fresh(), T0, phrase('W05', 'W06', true));
  const a2 = feed(a.state, [{ type: 'TICK', at: T0 + 12000 }]);
  const waterId = sent(a2.effects)[0].id;
  const b = recordWith(a2.state, T0 + 20000, phrase('W02', 'W04', true));
  const c = recordWith(b.state, T0 + 30000, phrase('W20', 'W16', true));
  assert.equal(c.state.phase, 'confirm', 'W20도 일반 명령이라 확인 카운트다운');
  // 결과(T0+36000) + 확인 6초
  const d = feed(c.state, [{ type: 'TICK', at: T0 + 42000 }]);
  assert.equal(sent(d.effects).length, 0, '새 카드 없음');
  const up = updated(d.effects);
  assert.equal(up.length, 1);
  assert.equal(up[0].id, waterId);
  assert.equal(up[0].cancelRequested, true);
  assert.equal(up[0].note, '환자가 취소 요청함');
  assert.ok(d.state.cards[waterId], '카드는 지우지 않는다');
  assert.equal(wardView(d.state, T0 + 42000).sub, WARD_TEXT.cancelRequestSent);

  // 긴급 카드만 있으면 아무 카드도 고치지 않는다.
  const e = recordWith(fresh(), T0, phrase('W04', 'W02', true));
  const f = recordWith(e.state, T0 + 10000, phrase('W20', 'W16', true));
  const g = feed(f.state, [{ type: 'TICK', at: T0 + 22000 }]);
  assert.equal(updated(g.effects).length, 0);
  assert.equal(sent(g.effects).length, 0);
  assert.equal(wardView(g.state, T0 + 22000).sub, '취소할 요청이 없어요');
});

test('W07·W14는 요청만: 보낼 때 가상 침대 그대로, 간호사 확인 뒤에만 움직인다', () => {
  const a = recordWith(fresh(), T0, phrase('W14', 'W16', true));
  const b = feed(a.state, [{ type: 'TICK', at: T0 + 12000 }]);
  const card = sent(b.effects)[0];
  assert.equal(card.commandId, 'W14');
  assert.equal(b.state.room.bedAngle, 30);
  const c = feed(b.state, [{ type: 'NURSE_ACK', at: T0 + 13000, cardIds: [card.id] }]);
  assert.equal(c.state.room.bedAngle, 45);
  assert.equal(c.state.cards[card.id].acked, true);
  // 같은 확인이 또 와도 한 번만 움직인다.
  assert.equal(feed(c.state, [{ type: 'NURSE_ACK', at: T0 + 13500, cardIds: [card.id] }]).state.room.bedAngle, 45);

  const p = recordWith(fresh(), T0, phrase('W07', 'W05', true));
  const q = feed(p.state, [{ type: 'TICK', at: T0 + 12000 }]);
  assert.equal(q.state.room.position, '바로 누움');
  const qq = feed(q.state, [{ type: 'NURSE_ACK', at: T0 + 13000, cardIds: [sent(q.effects)[0].id] }]);
  assert.equal(qq.state.room.position, '왼쪽으로 누움');
});

test('W07·W14를 보낸 뒤 W20으로 취소 요청하면 간호사가 확인해도 가상 침대·자세가 움직이지 않는다', () => {
  for (const [id, check] of [['W14', (s) => assert.equal(s.room.bedAngle, 30)], ['W07', (s) => assert.equal(s.room.position, '바로 누움')]]) {
    const a = recordWith(fresh(), T0, phrase(id, 'W16', true));
    const b = feed(a.state, [{ type: 'TICK', at: T0 + 12000 }]);
    const cardId = sent(b.effects)[0].id;
    const c = recordWith(b.state, T0 + 20000, phrase('W20', 'W16', true));
    const d = feed(c.state, [{ type: 'TICK', at: T0 + 32000 }]);
    assert.equal(updated(d.effects)[0].id, cardId);
    assert.equal(d.state.cards[cardId].cancelRequested, true, id);
    const e = feed(d.state, [{ type: 'NURSE_ACK', at: T0 + 33000, cardIds: [cardId], room: '305' }]);
    check(e.state);
    assert.equal(e.state.cards[cardId].acked, true, '확인 표시는 한다');
    assert.notEqual(e.state.cards[cardId].roomApplied, true);
    assert.equal(wardView(e.state, T0 + 33000).acked, true, '"간호사가 확인했어요"는 보인다');
  }
  // 같은 요청 두 번(간호사 화면이 한 카드로 합침) 중 마지막을 취소 요청 → 합친 카드 확인에도 움직이지 않는다.
  const a = recordWith(fresh(), T0, phrase('W14', 'W16', true));
  const b = feed(a.state, [{ type: 'TICK', at: T0 + 12000 }]);
  const c = recordWith(b.state, T0 + 20000, phrase('W14', 'W16', true));
  const d = feed(c.state, [{ type: 'TICK', at: T0 + 32000 }]);
  const ids = [...sent(b.effects), ...sent(d.effects)].map((x) => x.id);
  assert.equal(ids.length, 2);
  const e = recordWith(d.state, T0 + 40000, phrase('W20', 'W16', true));
  const f = feed(e.state, [{ type: 'TICK', at: T0 + 52000 }]);
  assert.equal(f.state.cards[ids[1]].cancelRequested, true);
  const g = feed(f.state, [{ type: 'NURSE_ACK', at: T0 + 53000, cardIds: ids, room: '305' }]);
  assert.equal(g.state.room.bedAngle, 30);
  assert.ok(ids.every((x) => g.state.cards[x].acked));
});

test('간호사 확인 → "간호사가 확인했어요" 표시(10초), 보낸 화면 문구도 바뀐다', () => {
  const a = recordWith(fresh(), T0, phrase('W05', 'W06', true));
  const b = feed(a.state, [{ type: 'TICK', at: T0 + 12000 }]);
  const id = sent(b.effects)[0].id;
  const c = feed(b.state, [{ type: 'NURSE_ACK', at: T0 + 15000, cardIds: [id], room: '305' }]);
  const v = wardView(c.state, T0 + 15000);
  assert.equal(v.acked, true);
  assert.equal(v.sub, '간호사가 확인했어요');
  const later = feed(c.state, [{ type: 'TICK', at: T0 + 15000 + ACK_NOTICE_MS }]);
  assert.equal(wardView(later.state, T0 + 15000 + ACK_NOTICE_MS).acked, false);
  // 모르는 카드라도 같은 병실이면 표시 (환자 창을 새로 연 뒤의 확인)
  assert.equal(wardView(feed(fresh(), [{ type: 'NURSE_ACK', at: T0, cardIds: ['x-1'], room: '305' }]).state, T0).acked, true);
  assert.equal(wardView(feed(fresh(), [{ type: 'NURSE_ACK', at: T0, cardIds: ['x-1'], room: '999' }]).state, T0).acked, false);
});

test('긴급 카드 확인 → 긴급 띠가 사라진다', () => {
  const a = recordWith(fresh(), T0, phrase('W10', 'W05', true));
  const id = sent(a.effects)[0].id;
  assert.equal(wardView(a.state, T0 + 6000).alarm.id, id);
  const b = feed(a.state, [{ type: 'NURSE_ACK', at: T0 + 7000, cardIds: [id] }]);
  assert.equal(wardView(b.state, T0 + 7000).alarm, null);
});

test('W08·W09 카드는 "체온 확인 권장", W08은 가상 온도 +1', () => {
  for (const id of ['W08', 'W09']) {
    const a = recordWith(fresh(), T0, phrase(id, 'W05', true));
    const b = feed(a.state, [{ type: 'TICK', at: T0 + 12000 }]);
    assert.equal(sent(b.effects)[0].hint, '체온 확인 권장', id);
    assert.equal(b.state.room.temperature, id === 'W08' ? 25 : 24);
  }
  const w05 = feed(recordWith(fresh(), T0, phrase('W05', 'W06', true)).state, [{ type: 'TICK', at: T0 + 12000 }]);
  assert.equal(sent(w05.effects)[0].hint, null);
});

test('가상 병실: W11 조명, W12 수면등(TV 끔), W13 커튼, W15 TV — 보낼 때 바뀐다', () => {
  const after = (id, s = fresh()) => feed(recordWith(s, T0, phrase(id, 'W16', true)).state, [{ type: 'TICK', at: T0 + 12000 }]).state;
  assert.equal(after('W11').room.light, 'on');
  assert.equal(after('W13').room.curtain, 'closed');
  const tv = after('W15');
  assert.equal(tv.room.tv, true);
  const sleep = feed(recordWith(tv, T0 + 20000, phrase('W12', 'W16', true)).state, [{ type: 'TICK', at: T0 + 32000 }]).state;
  assert.equal(sleep.room.light, 'night');
  assert.equal(sleep.room.tv, false);
  // 취소한 요청은 가상 병실을 바꾸지 않는다.
  const cancelled = feed(recordWith(fresh(), T0, phrase('W11', 'W16', true)).state, [{ type: 'PRESS', at: T0 + 7000 }, { type: 'TICK', at: T0 + 20000 }]);
  assert.equal(cancelled.state.room.light, 'off');
});

test('늦게 온 이전 녹화 결과·분석 중이 아닐 때의 결과는 무시한다', () => {
  const a = feed(fresh(), [{ type: 'PRESS', at: T0 }]);
  const stale = feed(a.state, [{ type: 'RESULT', at: T0 + 100, run: a.state.run - 1, results: phrase('W02', 'W04', true) }]);
  assert.equal(stale.state, a.state);
  const idle = stepWard(fresh(), { type: 'RESULT', at: T0, run: 0, results: phrase('W02', 'W04', true) });
  assert.equal(idle.effects.length, 0);
});

test('분석 오류 → 분류 스캐닝 + 사유, 녹화 실패 → 대기 + 안내', () => {
  const a = feed(fresh(), [{ type: 'PRESS', at: T0 }]);
  const run = a.state.run;
  const err = feed(a.state, [{ type: 'RECORDING_DONE', at: T0 + 5000, run }, { type: 'ANALYSIS_ERROR', at: T0 + 6000, run, message: '서버 오류' }]);
  assert.equal(err.state.phase, 'scan');
  assert.equal(wardView(err.state, T0 + 6000).reason, '분석하지 못했어요 (서버 오류)');
  const to = feed(a.state, [{ type: 'ANALYSIS_ERROR', at: T0 + 6000, run, reason: 'timeout' }]);
  assert.equal(wardView(to.state, T0 + 6000).reason, '분석이 너무 오래 걸렸어요');
  const fail = feed(a.state, [{ type: 'RECORD_FAILED', at: T0 + 10, run, message: '카메라가 준비되지 않았어요' }]);
  assert.equal(fail.state.phase, 'idle');
  assert.equal(wardView(fail.state, T0 + 10).sub, '카메라가 준비되지 않았어요');
});

test('감시: 녹화 끝 알림이 없으면 오류, 분석이 너무 길면 분류 스캐닝', () => {
  const a = feed(fresh(), [{ type: 'PRESS', at: T0 }]);
  const stuck = feed(a.state, [{ type: 'TICK', at: T0 + 5000 + RECORD_GRACE_MS + 1 }]);
  assert.equal(stuck.state.phase, 'idle');
  assert.equal(wardView(stuck.state, T0 + 20000).sub, WARD_TEXT.recordStuck);
  const b = feed(a.state, [{ type: 'RECORDING_DONE', at: T0 + 5000, run: a.state.run }]);
  const slow = feed(b.state, [{ type: 'TICK', at: T0 + 5000 + ANALYSIS_LIMIT_MS + 1 }]);
  assert.equal(slow.state.phase, 'scan');
  assert.equal(slow.state.scan.reason, 'timeout');
});

test('연습 결과(SIMULATE)는 어느 단계에서나 새 결과로 처리하고 진행 중인 녹화 결과를 무효로 한다', () => {
  const a = feed(fresh(), [{ type: 'PRESS', at: T0 }]);
  const oldRun = a.state.run;
  const sim = feed(a.state, [{ type: 'SIMULATE', at: T0 + 1000, results: phrase('W02', 'W04', true) }]);
  assert.equal(sent(sim.effects)[0].kind, 'emergency');
  const late = stepWard(sim.state, { type: 'RESULT', at: T0 + 6000, run: oldRun, results: phrase('W05', 'W06', true) });
  assert.equal(late.state, sim.state);
});

test('카드 ID는 창 안에서 겹치지 않고, 갱신마다 rev가 오른다', () => {
  let s = fresh();
  const ids = new Set();
  for (let i = 0; i < 5; i += 1) {
    const r = recordWith(s, T0 + i * 20000, phrase('W02', 'W04', true));
    ids.add(sent(r.effects)[0].id);
    s = r.state;
  }
  assert.equal(ids.size, 5);
  assert.ok([...ids].every((id) => id.startsWith('t-')));
});

test('설정 변경(병실 번호)은 이후 카드에 쓰인다, 잘못된 이벤트는 상태를 바꾸지 않는다', () => {
  const s = feed(fresh(), [{ type: 'SETTINGS', at: T0, settings: { room: '412' } }]).state;
  assert.equal(recordWith(s, T0, phrase('W02', 'W04', true)).state.cards['t-1'].room, '412');
  const base = fresh();
  assert.equal(stepWard(base, null).state, base);
  assert.equal(stepWard(base, { type: 'PRESS' }).state, base, '시각 없음');
  assert.equal(stepWard(base, { type: 'TICK', at: T0 }).state, base, '대기 중 TICK은 같은 객체');
});

test('대기 화면 문구', () => {
  const v = wardView(fresh(), T0);
  assert.equal(v.headline, WARD_TEXT.welcome);
  assert.match(v.sub, /5초 동안 녹화/);
  assert.match(v.sub, /3초 안에 3번 = 긴급 호출/);
});
