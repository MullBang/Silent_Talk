// wardDecision.test.js — 병실 판정표 검사 (node --test): 화면(classifyWardResults·decideWard·handleOutcome)이
// scripts/ward_tools/metrics.py decide()와 같은 결과를 내는지 본다. 기대 결과는 fixtures/ward_decision_cases.json에 있고,
// 그 파일은 fixtures/make_ward_decision_cases.py가 같은 results를 metrics 판정표에 넣어 만든다(합성 값, 사람 자료 없음).
// 다시 만들기: 저장소 루트에서 py -3.11 frontend/tests/fixtures/make_ward_decision_cases.py
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { buildCommandIndex } from '../src/ward/wardCommands.js';
import { WARD_TEXT, createWardState, stepWard, wardView } from '../src/ward/wardMachine.js';
import { DECISION_OUTCOMES, classifyWardResults, decideWard } from '../src/ward/wardResult.js';
import { ENROLL_NOTE_TEXT, cardEnrollNote, upsertNurseCard } from '../src/ward/wardStore.js';

const commandsRaw = readFileSync(new URL('../src/ward/commands.json', import.meta.url));
const commands = JSON.parse(commandsRaw.toString('utf8'));
const INDEX = buildCommandIndex(commands);
const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/ward_decision_cases.json', import.meta.url), 'utf8'));
const CASES = new Map(FIXTURE.cases.map((c) => [c.name, c]));
const T0 = 2_000_000;

/** 화면 판정 → metrics decide 모양. */
function asMetrics(d) {
  return {
    outcome: d.outcome, chosen_id: d.chosenId, via: d.via, text_pass: d.textPass, conflict: d.conflict, enroll_alarm: d.enrollAlarm,
  };
}

/** 결과 하나를 상태도에 넣는다 (누름 → 녹화 끝 → 결과). */
function runResult(results, state = createWardState({ index: INDEX, sessionKey: 'd' })) {
  const effects = [];
  let s = state;
  for (const e of [
    { type: 'PRESS', at: T0 },
    { type: 'RECORDING_DONE', at: T0 + 5000, run: s.run + 1 },
    { type: 'RESULT', at: T0 + 6000, run: s.run + 1, results },
  ]) {
    const r = stepWard(s, e);
    s = r.state;
    effects.push(...r.effects);
  }
  return { state: s, effects, sent: effects.filter((e) => e.type === 'SEND_CARD').map((e) => e.card) };
}

/** 서버 결과 한 건을 직접 만든다 (손으로 쓴 사례용). */
function result({ top, runner = null, margin = 0.1, score = -3.9, accepted = false, rule = undefined, enroll = undefined }) {
  const phrase = { id: top, text: INDEX.byId.get(top)?.text || '', score, margin, runner_up_id: runner, accepted, set: 'ward' };
  if (rule !== undefined) phrase.rule = rule;
  return [{ start_ms: 0, end_ms: 5000, text: 'x', confidence: 0.8, raw_score: -1, phrase, ...(enroll ? { enroll } : {}) }];
}
const RULE = (floor = null, dist = 0.27) => ({ min_margin: 0.5, min_score: -3.4, possible_min_score: floor, enroll_max_distance: dist });
const MATCH = (top, distance, phraseId, extra = {}) => {
  const within = distance <= 0.27;
  return {
    mode: 'match', profile: 'demo01', status: 'ok', count: 20, top_id: top, distance, runner_up_id: 'W12', runner_up_distance: 0.5,
    distances: { [top]: distance }, max_distance: 0.27, within, agree: top === phraseId, accepted: within && top === phraseId, ...extra,
  };
};

test('고정 사례 파일: metrics 판정표로 만든 것이고 지금 commands.json과 같은 목록이다', () => {
  assert.equal(FIXTURE.schema, 'ward-decision-cases-v1');
  assert.equal(FIXTURE.commands_sha256, createHash('sha256').update(commandsRaw).digest('hex'), 'commands.json이 바뀌면 사례를 다시 만든다');
  assert.deepEqual(FIXTURE.emergency_ids, INDEX.list.filter((c) => c.emergency).map((c) => c.id).sort());
  assert.ok(FIXTURE.cases.length >= 50);
  const outcomes = new Set(FIXTURE.cases.map((c) => c.expected.outcome));
  for (const o of DECISION_OUTCOMES) assert.ok(outcomes.has(o), `사례에 ${o}가 있다`);
  assert.equal(CASES.size, FIXTURE.cases.length, '사례 이름은 겹치지 않는다');
});

test('모든 사례: 화면 판정(classifyWardResults → decideWard)이 metrics.decide와 같다', () => {
  for (const c of FIXTURE.cases) {
    const d = decideWard(classifyWardResults(c.results, INDEX));
    assert.deepEqual(asMetrics(d), c.expected, c.name);
  }
});

test('모든 사례: 상태도(handleOutcome)가 판정대로 움직인다 (빨간 카드·확인 카운트다운·긴급 가능성·후보 스캐닝·분류 스캐닝·발화 없음)', () => {
  for (const c of FIXTURE.cases) {
    const { state, sent, effects } = runResult(c.results);
    const { outcome, chosen_id: chosen, via } = c.expected;
    const msg = `${c.name} (${outcome})`;
    switch (outcome) {
      case 'emergency':
        assert.deepEqual(sent.map((x) => [x.kind, x.commandId]), [['emergency', chosen]], msg);
        assert.equal(state.phase, 'idle', msg);
        assert.equal(sent[0].enroll, null, msg);
        break;
      case 'request': {
        assert.equal(sent.length, 0, msg);
        assert.equal(state.phase, 'confirm', msg);
        assert.equal(state.confirm.commandId, chosen, msg);
        assert.equal(state.confirm.enroll, via === 'enroll' ? 'chosen' : null, msg);
        assert.ok(effects.some((e) => e.type === 'SPEAK'), msg);
        const after = stepWard(state, { type: 'TICK', at: T0 + 60000 });
        const cards = after.effects.filter((e) => e.type === 'SEND_CARD').map((e) => e.card);
        if (chosen === 'W20') {
          assert.equal(cards.length, 0, `${msg}: W20은 취소 요청(보낸 일반 카드가 없으면 카드 없음)`);
        } else {
          assert.deepEqual(cards.map((x) => [x.kind, x.commandId, x.enroll]), [['request', chosen, via === 'enroll' ? 'chosen' : null]], msg);
        }
        break;
      }
      case 'emergency_possible':
        assert.equal(sent.length, 1, msg);
        assert.equal(sent[0].kind, 'emergency_possible', msg);
        assert.equal(state.phase, 'idle', msg);
        if (chosen) assert.equal(sent[0].candidates[0].id, chosen, `${msg}: 고른 명령이 후보 맨 앞`);
        assert.equal(sent[0].enroll, via === 'enroll' ? 'chosen' : (c.expected.enroll_alarm ? 'alarm' : null), msg);
        break;
      case 'not_chosen':
        assert.deepEqual(sent.map((x) => x.kind), ['check'], msg);
        assert.equal(state.phase, 'scan', msg);
        assert.equal(state.scan.mode, 'candidates', msg);
        break;
      case 'no_result':
        assert.equal(sent.length, 0, msg);
        assert.equal(state.phase, 'scan', msg);
        assert.equal(state.scan.mode, 'categories', msg);
        break;
      case 'no_speech':
      case 'lip_fail':
        assert.equal(sent.length, 0, msg);
        assert.equal(state.phase, 'idle', msg);
        assert.equal(state.screen.kind, 'no_speech', msg);
        assert.equal(state.screen.reason, outcome === 'no_speech' ? 'no_speech' : 'not_detected', msg);
        break;
      default:
        assert.fail(`모르는 결과 ${outcome}`);
    }
  }
});

test('손으로 쓴 판정표: 글자 통과·등록으로 고름·충돌·등록 긴급·글자 긴급(하한 유무)', () => {
  const table = [
    // [이름, results, outcome, chosenId, via, 긴급 가능성 후보(순서), 카드 등록 표시]
    ['글자 통과 긴급', result({ top: 'W02', runner: 'W05', margin: 1.2, score: -1, accepted: true, rule: RULE() }), 'emergency', 'W02', 'text', null, null],
    ['글자 통과 일반', result({ top: 'W05', runner: 'W06', margin: 1.2, score: -1, accepted: true, rule: RULE() }), 'request', 'W05', 'text', null, null],
    ['등록으로 고름 긴급', result({ top: 'W03', runner: 'W05', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W03', 0.2, 'W03') }),
      'emergency_possible', 'W03', 'enroll', ['W03', 'W05'], 'chosen'],
    ['등록으로 고름 일반', result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W11', 0.2, 'W11') }),
      'request', 'W11', 'enroll', null, null],
    ['등록으로 고름 일반 + 2등 긴급(하한 없음) → 긴급 가능성, 고른 명령이 맨 앞',
      result({ top: 'W08', runner: 'W03', margin: 0.434, score: -2.982, rule: RULE(), enroll: MATCH('W08', 0.136, 'W08') }),
      'emergency_possible', 'W08', 'enroll', ['W08', 'W03'], 'chosen'],
    ['등록으로 고름 일반 + 2등 긴급 −3.3−0.2 = −3.5 ≥ 하한 −3.6 → 긴급 가능성',
      result({ top: 'W08', runner: 'W03', margin: 0.2, score: -3.3, rule: RULE(-3.6), enroll: MATCH('W08', 0.15, 'W08') }),
      'emergency_possible', 'W08', 'enroll', ['W08', 'W03'], 'chosen'],
    ['등록으로 고름 일반 + 2등 긴급 −3.3−0.4 = −3.7 < 하한 −3.6 → 요청',
      result({ top: 'W08', runner: 'W03', margin: 0.4, score: -3.3, rule: RULE(-3.6), enroll: MATCH('W08', 0.15, 'W08') }),
      'request', 'W08', 'enroll', null, null],
    ['옛 서버 2등 긴급(등록 없음)과 같은 알림', result({ top: 'W08', runner: 'W03', margin: 0.434, score: -2.982 }),
      'emergency_possible', null, null, ['W08', 'W03'], null],
    ['충돌(글자 W11 ≠ 등록 W13)', result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W13', 0.2, 'W11') }),
      'not_chosen', null, null, null, null],
    ['충돌 + 등록 1등 긴급', result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W10', 0.25, 'W11') }),
      'emergency_possible', null, null, ['W11', 'W13', 'W10'], 'alarm'],
    ['충돌 + 글자 긴급 + 등록 긴급', result({ top: 'W03', runner: 'W11', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W10', 0.25, 'W03') }),
      'emergency_possible', null, null, ['W03', 'W11', 'W10'], null],
    ['등록 거리 밖 긴급은 무시', result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W02', 0.3, 'W11') }),
      'not_chosen', null, null, null, null],
    ['하한 없음: 2등 긴급 점수가 낮아도 긴급 가능성', result({ top: 'W05', runner: 'W04', margin: 2, score: -5, rule: RULE() }),
      'emergency_possible', null, null, ['W05', 'W04'], null],
    ['하한 −3.6: 1등 −3.5 ≥ 하한', result({ top: 'W03', runner: 'W05', margin: 0.1, score: -3.5, rule: RULE(-3.6) }),
      'emergency_possible', null, null, ['W03', 'W05'], null],
    ['하한 −3.6: 1등 −3.7 < 하한', result({ top: 'W03', runner: 'W05', margin: 0.1, score: -3.7, rule: RULE(-3.6) }),
      'not_chosen', null, null, null, null],
    ['하한 −3.6: 2등 −3.3−0.25 = −3.55 ≥ 하한', result({ top: 'W05', runner: 'W04', margin: 0.25, score: -3.3, rule: RULE(-3.6) }),
      'emergency_possible', null, null, ['W05', 'W04'], null],
    ['하한 −3.6: 2등 −3.3−0.35 = −3.65 < 하한', result({ top: 'W05', runner: 'W04', margin: 0.35, score: -3.3, rule: RULE(-3.6) }),
      'not_chosen', null, null, null, null],
    ['하한이 있어도 등록 1등 긴급은 알림', result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(-3.6), enroll: MATCH('W02', 0.2, 'W11') }),
      'emergency_possible', null, null, ['W11', 'W13', 'W02'], 'alarm'],
    ['하한 아래 1등이라도 등록으로 고르면 긴급 가능성', result({ top: 'W03', runner: 'W05', margin: 0.2, score: -3.9, rule: RULE(-3.6), enroll: MATCH('W03', 0.2, 'W03') }),
      'emergency_possible', 'W03', 'enroll', ['W03', 'W05'], 'chosen'],
    ['옛 서버(rule·enroll 없음) 2등 긴급', result({ top: 'W05', runner: 'W04' }), 'emergency_possible', null, null, ['W05', 'W04'], null],
  ];
  for (const [name, results, outcome, chosenId, via, candidates, tag] of table) {
    const d = decideWard(classifyWardResults(results, INDEX));
    assert.equal(d.outcome, outcome, name);
    assert.equal(d.chosenId, chosenId, name);
    assert.equal(d.via, via, name);
    const { sent, state } = runResult(results);
    if (candidates) {
      assert.deepEqual(sent[0].candidates.map((x) => x.id), candidates, name);
      assert.equal(sent[0].enroll, tag, name);
      assert.deepEqual(state.screen.candidates.map((x) => x.id), candidates, `${name}: 환자 화면 후보`);
    }
  }
});

test('같은 판정을 fixture도 낸다 (손으로 쓴 값과 metrics 결과가 서로 맞는지)', () => {
  const pairs = {
    enroll_accepted_emergency: ['emergency_possible', 'W03', 'enroll'],
    enroll_accepted_request: ['request', 'W11', 'enroll'],
    enroll_accepted_request_runner_emergency: ['emergency_possible', 'W05', 'enroll'],
    enroll_accepted_request_runner_emergency_floor_below: ['request', 'W08', 'enroll'],
    conflict_not_chosen: ['not_chosen', null, null],
    conflict_enroll_alarm: ['emergency_possible', null, null],
    floor_runner_above: ['emergency_possible', null, null],
    floor_runner_below: ['not_chosen', null, null],
    floor_top_below: ['not_chosen', null, null],
    old_server_text_alarm_runner: ['emergency_possible', null, null],
    enroll_stale: ['not_chosen', null, null],
    multi_window_with_enroll: ['no_result', null, null],
  };
  for (const [name, [outcome, chosen, via]] of Object.entries(pairs)) {
    const e = CASES.get(name).expected;
    assert.deepEqual([e.outcome, e.chosen_id, e.via], [outcome, chosen, via], name);
  }
});

test('하한이 없으면 긴급 안전은 오늘과 같다: 고르지 못한 모든 1·2등 쌍에서 긴급이 하나라도 있으면 긴급 가능성', () => {
  const ids = INDEX.list.map((c) => c.id);
  const notOk = ['missing', 'incomplete', 'stale', 'no_phrase', 'error'];
  for (const top of ids) {
    for (const runner of ids) {
      if (runner === top) continue;
      const alarm = INDEX.byId.get(top).emergency || INDEX.byId.get(runner).emergency;
      const variants = [
        result({ top, runner }), // 옛 서버
        result({ top, runner, rule: RULE() }),
        result({ top, runner, rule: RULE(null, null) }),
        ...notOk.map((status) => result({ top, runner, rule: RULE(), enroll: { mode: 'match', status, count: 3, within: false, agree: false, accepted: false } })),
        result({ top, runner, rule: RULE(), enroll: MATCH(top === 'W12' ? 'W13' : 'W12', 0.1, top) }), // 충돌(일반 등록 1등)
      ];
      for (const results of variants) {
        const d = decideWard(classifyWardResults(results, INDEX));
        assert.equal(d.outcome, alarm ? 'emergency_possible' : 'not_chosen', `${top}/${runner}`);
        if (alarm) assert.deepEqual(d.candidates.map((c) => c.id).slice(0, 2), [top, runner]);
      }
    }
  }
});

test('등록은 알림을 더하기만 한다: 같은 하한에서 등록 없는 판정이 긴급(빨간 카드·가능성)이면 어떤 등록 결과에서도 긴급', () => {
  const ids = INDEX.list.map((c) => c.id);
  const ALARM = new Set(['emergency', 'emergency_possible']);
  const scores = [[0.1, -3.9], [0.434, -2.982], [0.2, -3.3], [0.4, -3.3]];
  for (const floor of [null, -3.6]) {
    for (const top of ids) {
      for (const runner of ids) {
        if (runner === top) continue;
        for (const [margin, score] of scores) {
          const base = decideWard(classifyWardResults(result({ top, runner, margin, score, rule: RULE(floor) }), INDEX));
          if (!ALARM.has(base.outcome)) continue;
          for (const enrollTop of ids) {
            for (const dist of [0.1, 0.4]) {
              const d = decideWard(classifyWardResults(
                result({ top, runner, margin, score, rule: RULE(floor), enroll: MATCH(enrollTop, dist, top) }), INDEX));
              assert.ok(ALARM.has(d.outcome), `${floor}/${top}/${runner}/${margin}/${score} 등록 ${enrollTop} ${dist} → ${d.outcome}`);
              if (d.via === 'enroll') assert.equal(d.candidates[0].id, top, '등록으로 고른 명령이 후보 맨 앞');
            }
          }
        }
      }
    }
  }
});

test('하한이 있으면 글자 긴급 가능성만 바뀐다: 글자 통과 긴급·등록으로 고른 긴급·등록 1등 긴급은 그대로', () => {
  const low = -9;
  for (const floor of [null, -3.6, -1]) {
    const pass = decideWard(classifyWardResults(result({ top: 'W04', runner: 'W05', margin: 1, score: -3, accepted: true, rule: RULE(floor) }), INDEX));
    assert.equal(pass.outcome, 'emergency', String(floor));
    const viaEnroll = decideWard(classifyWardResults(result({ top: 'W10', runner: 'W05', margin: 0.1, score: low, rule: RULE(floor), enroll: MATCH('W10', 0.1, 'W10') }), INDEX));
    assert.equal(viaEnroll.outcome, 'emergency_possible', String(floor));
    const enrollTop = decideWard(classifyWardResults(result({ top: 'W05', runner: 'W06', margin: 0.1, score: low, rule: RULE(floor), enroll: MATCH('W02', 0.1, 'W05') }), INDEX));
    assert.equal(enrollTop.outcome, 'emergency_possible', String(floor));
    const textOnly = decideWard(classifyWardResults(result({ top: 'W05', runner: 'W04', margin: 0.1, score: low, rule: RULE(floor) }), INDEX));
    assert.equal(textOnly.outcome, floor === null ? 'emergency_possible' : 'not_chosen', String(floor));
  }
});

test('등록 결과 읽기: 첫 결과 항목의 match만, 상태 ok·목록 안 1등일 때만 거리 안, 서로 맞지 않는 값은 고르지 않음', () => {
  const ok = classifyWardResults(result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(-3.6), enroll: MATCH('W11', 0.2, 'W11') }), INDEX);
  assert.equal(ok.floor, -3.6);
  assert.deepEqual(ok.rule, { minMargin: 0.5, minScore: -3.4, possibleMinScore: -3.6, enrollMaxDistance: 0.27 });
  assert.equal(ok.enroll.status, 'ok');
  assert.equal(ok.enroll.top.id, 'W11');
  assert.equal(ok.enroll.distance, 0.2);
  assert.equal(ok.enroll.accepted, true);
  // 옛 서버: rule·enroll 없음
  const old = classifyWardResults(result({ top: 'W11', runner: 'W13' }), INDEX);
  assert.equal(old.rule, null);
  assert.equal(old.floor, null);
  assert.equal(old.enroll, null);
  // 상태가 ok가 아니면 서버가 within=true를 보내도 쓰지 않는다
  const stale = classifyWardResults(result({ top: 'W11', runner: 'W13', rule: RULE(), enroll: { ...MATCH('W11', 0.1, 'W11'), status: 'stale' } }), INDEX);
  assert.equal(stale.enroll.within, false);
  assert.equal(stale.enroll.accepted, false);
  assert.equal(stale.enroll.distance, null);
  // 모르는 상태는 error
  assert.equal(classifyWardResults(result({ top: 'W11', rule: RULE(), enroll: { mode: 'match', status: 'weird' } }), INDEX).enroll.status, 'error');
  // accepted인데 1등이 다르면 고르지 않음
  const bad = classifyWardResults(result({ top: 'W11', runner: 'W13', rule: RULE(), enroll: { ...MATCH('W13', 0.1, 'W11'), agree: true, accepted: true } }), INDEX);
  assert.equal(bad.enroll.accepted, false);
  assert.equal(decideWard(bad).conflict, true);
  // store 모드(등록 화면 결과)는 병실 판정에 쓰지 않는다
  const store = classifyWardResults(result({ top: 'W11', rule: RULE(), enroll: { mode: 'store', stored: true, command: 'W11' } }), INDEX);
  assert.equal(store.enroll, null);
  // 두 번째 결과 항목의 enroll은 보지 않는다(여러 창은 결과 없음)
  const multi = [...result({ top: 'W03' }), { start_ms: 5000, end_ms: 9000, text: '나', enroll: MATCH('W03', 0.1, 'W03') }];
  assert.equal(classifyWardResults(multi, INDEX).kind, 'no_phrase');
  // 문장 모드 결과(rule 없음)는 병실 명령으로 해석하지 않는다(오늘과 같음)
  assert.deepEqual(classifyWardResults([{ text: 'x', phrase: { id: 'A01', score: -1, margin: 1, accepted: true, set: 'sentences' } }], INDEX),
    { kind: 'no_phrase', reason: 'other_set' });
});

test('환자 화면: 등록으로 고른 일반 요청은 확인 카운트다운에 "내 입 모양 등록으로 골랐어요", 긴급 가능성에도 표시', () => {
  const req = runResult(result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W11', 0.2, 'W11') }));
  const v = wardView(req.state, T0 + 6000);
  assert.equal(v.tone, 'confirm');
  assert.equal(v.headline, '불을 켜 주세요');
  assert.equal(v.reason, WARD_TEXT.enrollChosen);
  const text = runResult(result({ top: 'W11', runner: 'W13', margin: 1.2, score: -1, accepted: true, rule: RULE() }));
  assert.equal(wardView(text.state, T0 + 6000).reason, null, '글자로 고른 것에는 등록 표시 없음');
  const poss = runResult(result({ top: 'W03', runner: 'W05', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W03', 0.2, 'W03') }));
  const pv = wardView(poss.state, T0 + 6000);
  assert.equal(pv.label, '[긴급 가능성]');
  assert.equal(pv.reason, WARD_TEXT.enrollChosen);
  const alarm = runResult(result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W10', 0.2, 'W11') }));
  assert.equal(wardView(alarm.state, T0 + 6000).reason, WARD_TEXT.enrollAlarm);
  // 등록으로 고른 W05를 카운트다운 중 누르면 취소(보내지 않음) — 글자로 고른 것과 같다.
  const water = runResult(result({ top: 'W05', runner: 'W06', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W05', 0.2, 'W05') }));
  const cancelled = stepWard(water.state, { type: 'PRESS', at: T0 + 7000 });
  assert.equal(cancelled.state.phase, 'idle');
  assert.equal(stepWard(cancelled.state, { type: 'TICK', at: T0 + 30000 }).effects.filter((e) => e.type === 'SEND_CARD').length, 0);
});

test('간호사 화면: 등록으로 고른 카드에 "등록한 입 모양으로 고름", 등록 때문에만 생긴 긴급 가능성에 알림 문구', () => {
  const req = runResult(result({ top: 'W11', runner: 'W13', margin: 0.2, score: -3, rule: RULE(), enroll: MATCH('W11', 0.2, 'W11') }));
  const card = stepWard(req.state, { type: 'TICK', at: T0 + 60000 }).effects.find((e) => e.type === 'SEND_CARD').card;
  assert.equal(card.enroll, 'chosen');
  const nurse = upsertNurseCard([], card, T0 + 60000);
  assert.equal(nurse.cards[0].enroll, 'chosen');
  assert.equal(cardEnrollNote(nurse.cards[0]), '등록한 입 모양으로 고름');
  assert.equal(cardEnrollNote({ kind: 'emergency_possible', enroll: 'alarm' }), ENROLL_NOTE_TEXT.alarm);
  assert.equal(cardEnrollNote({ kind: 'request', enroll: null }), null);
  assert.equal(cardEnrollNote({ kind: 'check', enroll: 'chosen', selectedId: 'W05' }), null, '환자가 직접 고른 카드는 등록 표시 안 함');
  assert.equal(cardEnrollNote(null), null);
  // 갱신으로 온 enroll도 반영된다
  const upd = upsertNurseCard(nurse.cards, { ...card, rev: 2, enroll: null }, T0 + 61000);
  assert.equal(upd.cards[0].enroll, null);
  // 글자로 고른 카드에는 enroll이 null
  const t = runResult(result({ top: 'W02', runner: 'W05', margin: 1.2, score: -1, accepted: true, rule: RULE() }));
  assert.equal(t.sent[0].enroll, null);
});

test('판정 함수는 이상한 입력에도 예외 없이 결과 없음', () => {
  assert.equal(decideWard(null).outcome, 'no_result');
  assert.equal(decideWard({ kind: 'no_phrase', reason: 'multi_window' }).outcome, 'no_result');
  assert.equal(decideWard({ kind: 'no_phrase', reason: 'multi_window' }).reason, 'multi_window');
  assert.equal(decideWard({ kind: 'no_speech', reason: 'not_detected' }).outcome, 'lip_fail');
  assert.equal(decideWard({ kind: 'phrase' }).outcome, 'no_result');
  // 하한이 있는데 점수를 모르면(형식이 틀린 결과) 놓치지 않는 쪽: 긴급 가능성
  const d = decideWard({ kind: 'phrase', accepted: false, command: INDEX.byId.get('W03'), runnerUp: null, score: null, margin: null, floor: -3.6, enroll: null });
  assert.equal(d.outcome, 'emergency_possible');
});
