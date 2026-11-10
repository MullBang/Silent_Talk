// wardSupport.test.js — 병실 화면 보조 모듈 검사 (node --test): 명령 색인·결과 판정·설정·스위치 입력·음성·알림음·명암비·연습 결과·폴링.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  ACK_ONLY_ACTIONS, CANCEL_COMMAND_ID, INITIAL_ROOM, WARD_PHRASE_SET, applyRoomAction, buildCommandIndex, commandById,
  describeRoom, roomActionTiming, roomSummary,
} from '../src/ward/wardCommands.js';
import { NURSE_NON_FORWARD_KEYS, forwardIsSwitch, keyIsSwitch, pointerIsSwitch } from '../src/ward/wardInput.js';
import { createWardState, stepWard } from '../src/ward/wardMachine.js';
import { WARD_POLL_INTERVAL_MS, pollWardResult } from '../src/ward/wardPoll.js';
import { PRACTICE_SCENARIOS, practiceResults } from '../src/ward/wardPractice.js';
import { NO_PHRASE_REASON_TEXT, classifyWardResults } from '../src/ward/wardResult.js';
import {
  DEFAULT_SETTINGS, SETTINGS_STORAGE_KEY, isNightTime, isPracticeMode, loadSettings, normalizeRoom, normalizeSettings,
  saveSettings, speechAllowed,
} from '../src/ward/wardSettings.js';
import { BEEP_PATTERNS, createBeeper, patternSeconds } from '../src/ward/wardSound.js';
import { NO_VOICE_TEXT, pickKoreanLocalVoice, speakText, voiceLabel } from '../src/ward/wardSpeech.js';
import { CONTRAST_PAIRS, WARD_COLORS, contrastRatio } from '../src/ward/wardTheme.js';

const src = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const commands = JSON.parse(src('../src/ward/commands.json'));
const INDEX = buildCommandIndex(commands);

const phraseItem = (phrase) => [{ start_ms: 0, end_ms: 5000, text: 'x', confidence: 0.8, raw_score: -1, phrase }];
const ph = (over = {}) => ({ id: 'W05', text: '목이 말라 물 한 모금만', score: -1, margin: 1, runner_up_id: 'W06', accepted: true, set: 'ward', ...over });

test('명령 색인: 20개·분류 5개·긴급 W02 W03 W04 W10, 화면 문구(display)는 인식 문장(text)과 따로', () => {
  assert.equal(INDEX.list.length, 20);
  assert.deepEqual(INDEX.categories.map((c) => c.label), ['긴급', '호출·소통', '몸 돌봄', '병실', '대답']);
  assert.ok(INDEX.categories.every((c) => c.commands.length > 0));
  assert.equal(INDEX.categories.reduce((n, c) => n + c.commands.length, 0), 20, '모든 명령이 분류에 속한다');
  assert.deepEqual(INDEX.list.filter((c) => c.emergency).map((c) => c.id), ['W02', 'W03', 'W04', 'W10']);
  assert.equal(commandById(INDEX, 'W05').display, '물을 마시고 싶어요');
  assert.equal(commandById(INDEX, 'A01'), null);
  assert.equal(commandById(INDEX, null), null);
  assert.equal(commandById(INDEX, CANCEL_COMMAND_ID).action, 'cancel_last');
  for (const id of ['W07', 'W14']) assert.ok(ACK_ONLY_ACTIONS.includes(commandById(INDEX, id).action), id);
  assert.throws(() => buildCommandIndex({ fixed: [{ id: 'W01' }] }), /병실 명령 목록/);
  assert.throws(() => buildCommandIndex({ fixed: [] }));
  const dup = { ...commands, fixed: [commands.fixed[0], commands.fixed[0]] };
  assert.throws(() => buildCommandIndex(dup), /겹칩니다/);
});

test('가상 병실 동작과 적용 시점', () => {
  assert.equal(roomActionTiming('light_on'), 'send');
  assert.equal(roomActionTiming('request_backrest'), 'ack');
  assert.equal(roomActionTiming('request_reposition'), 'ack');
  assert.equal(roomActionTiming('alert_breathing'), null);
  let r = INITIAL_ROOM;
  assert.equal(applyRoomAction(r, 'call_nurse'), r, '바뀌는 것이 없으면 같은 객체');
  r = applyRoomAction(r, 'request_backrest');
  r = applyRoomAction(r, 'request_backrest');
  r = applyRoomAction(r, 'request_backrest');
  assert.equal(r.bedAngle, 60, '최대 60°');
  for (let i = 0; i < 10; i += 1) r = applyRoomAction(r, 'request_blanket');
  assert.equal(r.temperature, 28, '최대 28℃');
  assert.equal(describeRoom(INITIAL_ROOM).length, 5);
  assert.match(roomSummary(INITIAL_ROOM), /조명 꺼짐 · 커튼 열림 · 침대 등받이 30°/);
});

test('결과 판정: 서버 accepted만 믿고, 병실 묶음이 아닌 결과는 명령으로 해석하지 않는다', () => {
  const a = classifyWardResults(phraseItem(ph()), INDEX);
  assert.equal(a.kind, 'phrase');
  assert.equal(a.accepted, true);
  assert.equal(a.command.id, 'W05');
  assert.equal(a.runnerUp.id, 'W06');
  assert.equal(classifyWardResults(phraseItem(ph({ accepted: 'true' })), INDEX).accepted, false, 'true 값만 고름');
  assert.equal(classifyWardResults(phraseItem(ph({ accepted: undefined })), INDEX).accepted, false);
  // 높은 점수·큰 차이여도 서버가 accepted=false면 고르지 않는다(화면이 기준을 다시 판정하지 않는다).
  assert.equal(classifyWardResults(phraseItem(ph({ accepted: false, margin: 9, score: -0.1 })), INDEX).accepted, false);
  assert.deepEqual(classifyWardResults(phraseItem(ph({ set: 'sentences' })), INDEX), { kind: 'no_phrase', reason: 'other_set' });
  assert.deepEqual(classifyWardResults(phraseItem(ph({ id: 'A01', set: undefined })), INDEX), { kind: 'no_phrase', reason: 'other_set' });
  assert.deepEqual(classifyWardResults(phraseItem(ph({ set: undefined })), INDEX), { kind: 'no_phrase', reason: 'other_set' }, '묶음 표시가 없는 옛 서버 결과는 병실 ID여도 쓰지 않는다');
  assert.equal(classifyWardResults(phraseItem(ph({ runner_up_id: null })), INDEX).runnerUp, null);
  assert.equal(classifyWardResults(phraseItem(ph({ runner_up_id: 'W05' })), INDEX).runnerUp, null, '1등과 같은 2등은 없음');
  assert.deepEqual(classifyWardResults([{ text: '[발화 없음]' }], INDEX), { kind: 'no_speech', reason: 'no_speech' });
  assert.deepEqual(classifyWardResults([{ text: '[검출 실패 구간]' }], INDEX), { kind: 'no_speech', reason: 'not_detected' });
  assert.deepEqual(classifyWardResults([{ text: '[발화 없음]' }, { text: '[검출 실패 구간]' }], INDEX), { kind: 'no_speech', reason: 'not_detected' });
  assert.deepEqual(classifyWardResults([{ text: '가' }, { text: '나' }], INDEX), { kind: 'no_phrase', reason: 'multi_window' });
  assert.deepEqual(classifyWardResults([{ text: '가' }, { text: '[검출 실패 구간]' }], INDEX), { kind: 'no_phrase', reason: 'partial_detection' });
  assert.deepEqual(classifyWardResults([{ text: '가' }], INDEX), { kind: 'no_phrase', reason: 'unaligned' });
  assert.deepEqual(classifyWardResults([{ text: '[인식 불가]', candidate: '가' }], INDEX), { kind: 'no_phrase', reason: 'unaligned' });
  assert.deepEqual(classifyWardResults([], INDEX), { kind: 'no_phrase', reason: 'empty' });
  assert.deepEqual(classifyWardResults(null, INDEX), { kind: 'no_phrase', reason: 'empty' });
  for (const reason of ['empty', 'multi_window', 'partial_detection', 'unaligned', 'other_set', 'error', 'timeout', 'not_candidates']) {
    assert.ok(NO_PHRASE_REASON_TEXT[reason], reason);
  }
});

test('설정: 범위 맞추기·기본값·밤 무음·저장소 예외', () => {
  assert.deepEqual(normalizeSettings(null), { ...DEFAULT_SETTINGS });
  assert.equal(DEFAULT_SETTINGS.recordSeconds, 5);
  assert.equal(DEFAULT_SETTINGS.confirmSeconds, 6);
  assert.equal(DEFAULT_SETTINGS.tts, false, '음성 읽기 기본 꺼짐');
  assert.equal(DEFAULT_SETTINGS.nightMute, true);
  assert.equal(normalizeSettings({ recordSeconds: '8' }).recordSeconds, 8);
  assert.equal(normalizeSettings({ confirmSeconds: 5.5 }).confirmSeconds, 6);
  assert.equal(normalizeRoom(' 3<0>5호 '), '305호');
  assert.equal(normalizeRoom(''), '305');
  assert.equal(normalizeRoom('12345678901234'), '1234567890');
  const at = (h) => new Date(2026, 10, 9, h, 30);
  assert.equal(isNightTime(at(21)), false);
  assert.equal(isNightTime(at(22)), true);
  assert.equal(isNightTime(at(3)), true);
  assert.equal(isNightTime(at(6)), false);
  assert.equal(speechAllowed({ tts: false, nightMute: true }, at(12)), false);
  assert.equal(speechAllowed({ tts: true, nightMute: true }, at(12)), true);
  assert.equal(speechAllowed({ tts: true, nightMute: true }, at(23)), false);
  assert.equal(speechAllowed({ tts: true, nightMute: false }, at(23)), true);
  const data = new Map();
  const store = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v) };
  assert.equal(saveSettings(store, { recordSeconds: 8, room: '412', tts: true }), true);
  assert.ok(data.has(SETTINGS_STORAGE_KEY));
  assert.deepEqual(loadSettings(store), {
    recordSeconds: 8, confirmSeconds: 6, room: '412', tts: true, nightMute: true, enrollProfile: '', commandList: '', enrollProfileByList: {},
  });
  // 내 입 모양 등록 이름: 기본 ''(등록 안 씀), 서버 규칙(영문·숫자·_·- 1~32자)에 맞지 않으면 ''.
  assert.equal(DEFAULT_SETTINGS.enrollProfile, '');
  assert.equal(normalizeSettings({ enrollProfile: ' bed305 ' }).enrollProfile, 'bed305');
  for (const bad of ['../x', 'a b', '가나', 'x'.repeat(33), 7, null]) assert.equal(normalizeSettings({ enrollProfile: bad }).enrollProfile, '', String(bad));
  assert.equal(saveSettings(store, { enrollProfile: 'P_01-a' }), true);
  assert.equal(loadSettings(store).enrollProfile, 'P_01-a');
  const bad = { getItem() { throw new Error('x'); }, setItem() { throw new Error('x'); } };
  assert.deepEqual(loadSettings(bad), { ...DEFAULT_SETTINGS });
  assert.equal(saveSettings(bad, {}), false);
  data.set(SETTINGS_STORAGE_KEY, '{깨짐');
  assert.deepEqual(loadSettings(store), { ...DEFAULT_SETTINGS });
});

test('연습 모드 주소: #ward?practice=1 또는 ?practice=1#ward, 기본은 숨김', () => {
  assert.equal(isPracticeMode({ hash: '#ward?practice=1', search: '' }), true);
  assert.equal(isPracticeMode({ hash: '#ward', search: '?practice=1' }), true);
  assert.equal(isPracticeMode({ hash: '#ward', search: '' }), false);
  assert.equal(isPracticeMode({ hash: '#ward?practice=0', search: '' }), false);
  assert.equal(isPracticeMode(null), false);
});

test('스위치 입력: 아무 키·포인터, 단 글자 입력 칸·설정 버튼·브라우저 단축키·보조 키는 제외', () => {
  const el = (tag, noSwitch = false) => ({ tagName: tag, closest: (sel) => (noSwitch && sel === '[data-no-switch]' ? {} : null) });
  const body = el('BODY');
  assert.equal(keyIsSwitch({ key: ' ', target: body }), true);
  assert.equal(keyIsSwitch({ key: 'Enter', target: body }), true);
  assert.equal(keyIsSwitch({ key: 'a', target: el('DIV', true) }), true, '버튼이 아닌 곳은 스위치');
  assert.equal(keyIsSwitch({ key: 'F1', target: body }), true);
  for (const key of ['Tab', 'Shift', 'Control', 'Alt', 'Meta', 'Escape', 'F5', 'F11', 'F12']) {
    assert.equal(keyIsSwitch({ key, target: body }), false, key);
  }
  assert.equal(keyIsSwitch({ key: 'r', ctrlKey: true, target: body }), false);
  assert.equal(keyIsSwitch({ key: '1', target: el('INPUT', true) }), false);
  assert.equal(keyIsSwitch({ key: '1', target: { tagName: 'DIV', isContentEditable: true } }), false);
  assert.equal(keyIsSwitch({ key: ' ', target: el('BUTTON', true) }), false, '설정 버튼은 키보드로 조작');
  assert.equal(keyIsSwitch(null), false);
  assert.equal(pointerIsSwitch({ button: 0, pointerType: 'mouse', target: body }), true);
  assert.equal(pointerIsSwitch({ button: 0, pointerType: 'touch', target: body }), true);
  assert.equal(pointerIsSwitch({ button: 2, pointerType: 'mouse', target: body }), false, '오른쪽 버튼');
  assert.equal(pointerIsSwitch({ button: 0, pointerType: 'mouse', target: el('BUTTON', true) }), false);
});

test('간호사 창 → 환자 창 넘기기: 스위치 키는 넘기고, 목록을 넘겨 보는 이동 키는 넘기지 않는다', () => {
  const body = { tagName: 'BODY', closest: () => null };
  const button = { tagName: 'BUTTON', closest: (sel) => (sel === '[data-no-switch]' ? {} : null) };
  for (const key of [' ', 'Enter', 'a', 'F1']) assert.equal(forwardIsSwitch({ key, target: body }), true, key);
  assert.deepEqual(NURSE_NON_FORWARD_KEYS, ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End']);
  for (const key of NURSE_NON_FORWARD_KEYS) {
    assert.equal(forwardIsSwitch({ key, target: body }), false, key);
    assert.equal(keyIsSwitch({ key, target: body }), true, `환자 창에서는 ${key}도 스위치`);
  }
  assert.equal(forwardIsSwitch({ key: 'Tab', target: body }), false);
  assert.equal(forwardIsSwitch({ key: 'Enter', target: button }), false, '[확인] 버튼 조작은 넘기지 않음');
  assert.equal(forwardIsSwitch({ key: 'r', ctrlKey: true, target: body }), false);
  assert.equal(forwardIsSwitch(null), false);
});

test('음성 읽기: 오프라인 한국어 음성(ko-KR, localService)만, 없으면 "음성 읽기 없음"', () => {
  const voices = [
    { name: 'Google 한국의', lang: 'ko-KR', localService: false },
    { name: 'English', lang: 'en-US', localService: true },
    { name: 'Microsoft Heami', lang: 'ko-KR', localService: true },
  ];
  assert.equal(pickKoreanLocalVoice(voices).name, 'Microsoft Heami');
  assert.equal(pickKoreanLocalVoice([{ name: 'k', lang: 'ko_KR', localService: true }]).name, 'k');
  assert.equal(pickKoreanLocalVoice(voices.slice(0, 2)), null, '온라인 한국어 음성은 쓰지 않는다');
  assert.equal(pickKoreanLocalVoice(null), null);
  assert.equal(voiceLabel(null), NO_VOICE_TEXT);
  assert.equal(NO_VOICE_TEXT, '음성 읽기 없음');
  assert.equal(speakText(null, voices[2], '네'), false);
  assert.equal(speakText({ speak() {}, cancel() {} }, null, '네'), false, '음성이 없으면 읽지 않는다');
});

test('알림음: 패턴 길이, 소리를 켜기 전에는 재생하지 않는다', async () => {
  for (const kind of ['start', 'normal', 'emergency']) assert.ok(patternSeconds(BEEP_PATTERNS[kind]) > 0, kind);
  assert.ok(patternSeconds(BEEP_PATTERNS.emergency) < 2, '반복 간격(2초)보다 짧다');
  const none = createBeeper(() => null);
  assert.equal(await none.enable(), false);
  assert.equal(none.play('start'), false);
  let resumed = 0;
  const nodes = [];
  const node = () => {
    const n = { connect: (x) => x, start() {}, stop() {}, frequency: {}, gain: { setValueAtTime() {}, exponentialRampToValueAtTime() {} } };
    nodes.push(n);
    return n;
  };
  const ctx = { state: 'suspended', currentTime: 0, destination: {}, resume: async () => { resumed += 1; ctx.state = 'running'; },
    createOscillator: node, createGain: node, close() {} };
  const b = createBeeper(() => ctx);
  assert.equal(b.play('normal'), false, '켜기 전');
  assert.equal(await b.enable(), true);
  assert.equal(resumed, 1);
  assert.equal(b.play('normal'), true);
  assert.equal(b.play('nope'), false);
  assert.equal(nodes.length, 4, '일반: 소리 2개 × (발진기+음량)');
});

test('명암비: 화면에서 겹쳐 쓰는 글자/바탕은 모두 7:1 이상', () => {
  assert.ok(Math.abs(contrastRatio('#000000', '#ffffff') - 21) < 1e-9);
  for (const [fg, bg] of CONTRAST_PAIRS) {
    const ratio = contrastRatio(WARD_COLORS[fg], WARD_COLORS[bg]);
    assert.ok(ratio >= 7, `${fg}/${bg} = ${ratio.toFixed(2)}`);
  }
});

test('연습 결과: 장면마다 기대한 판정과 상태도 경로', () => {
  const expect = {
    accepted_normal: ['phrase', true, 'confirm'],
    accepted_emergency: ['phrase', true, 'idle'],
    uncertain_emergency: ['phrase', false, 'idle'],
    uncertain_normal: ['phrase', false, 'scan'],
    enroll_normal: ['phrase', false, 'confirm'],
    enroll_emergency: ['phrase', false, 'idle'],
    no_phrase: ['no_phrase', null, 'scan'],
    no_speech: ['no_speech', null, 'idle'],
    not_detected: ['no_speech', null, 'idle'],
  };
  assert.deepEqual(PRACTICE_SCENARIOS.map((s) => s.id), Object.keys(expect));
  for (const [id, [kind, accepted, phase]] of Object.entries(expect)) {
    const results = practiceResults(id, INDEX, { commandId: 'W11' });
    const c = classifyWardResults(results, INDEX);
    assert.equal(c.kind, kind, id);
    if (accepted !== null) assert.equal(c.accepted, accepted, id);
    if (kind === 'phrase') assert.equal(results[0].phrase.set, WARD_PHRASE_SET);
    const s = stepWard(createWardState({ index: INDEX, sessionKey: 'p' }), { type: 'SIMULATE', at: 1, results }).state;
    assert.equal(s.phase, phase, id);
  }
  assert.equal(practiceResults('accepted_normal', INDEX, { commandId: 'W11' })[0].phrase.id, 'W11');
  // 등록으로 고른 장면: 글자 규칙은 통과하지 못했지만 등록 1등이 글자 1등과 같고 거리 안 → 일반은 확인 카운트다운, 긴급은 긴급 가능성.
  const en = practiceResults('enroll_normal', INDEX, { commandId: 'W11' })[0];
  assert.equal(en.enroll.top_id, 'W11');
  assert.equal(classifyWardResults([en], INDEX).enroll.accepted, true);
  const ee = stepWard(createWardState({ index: INDEX, sessionKey: 'p' }), {
    type: 'SIMULATE', at: 1, results: practiceResults('enroll_emergency', INDEX),
  });
  assert.equal(ee.effects.find((e) => e.type === 'SEND_CARD').card.kind, 'emergency_possible');
  assert.equal(ee.effects.find((e) => e.type === 'SEND_CARD').card.enroll, 'chosen');
  assert.equal(practiceResults('accepted_normal', INDEX, { commandId: 'nope' })[0].phrase.id, 'W05');
  assert.deepEqual(practiceResults('nope', INDEX), []);
});

test('폴링: 약 300ms 간격으로 끝날 때까지, 초기 404 유예, 오류·시간 초과·취소', async () => {
  assert.equal(WARD_POLL_INTERVAL_MS, 300);
  const sleeps = [];
  const sleep = async (ms) => { sleeps.push(ms); };
  const seq = (list) => {
    let i = 0;
    return async () => {
      const x = list[Math.min(i, list.length - 1)];
      i += 1;
      if (x instanceof Error) throw x;
      return x;
    };
  };
  const notFound = Object.assign(new Error('404'), { response: { status: 404 } });
  const done = await pollWardResult({
    fetchResult: seq([notFound, { status: 'pending' }, { status: 'processing', progress: 50 }, { status: 'done', results: [{ text: 'x' }] }]),
    sleep,
  });
  assert.deepEqual(done, { kind: 'done', results: [{ text: 'x' }], warnings: [] });
  assert.deepEqual(sleeps, [300, 300, 300]);

  const err = await pollWardResult({ fetchResult: seq([{ status: 'error', error_message: '영상을 처리하지 못했습니다.' }]), sleep });
  assert.deepEqual(err, { kind: 'error', message: '영상을 처리하지 못했습니다.' });
  const net = await pollWardResult({ fetchResult: seq([Object.assign(new Error('x'), { response: { status: 500, data: {} } })]), sleep });
  assert.equal(net.kind, 'error');
  assert.deepEqual(await pollWardResult({ fetchResult: seq([{ status: 'timeout' }]), sleep }), { kind: 'timeout' });
  let clock = 0;
  const slow = await pollWardResult({
    fetchResult: seq([{ status: 'processing' }]), sleep: async (ms) => { clock += ms; }, now: () => clock, timeoutMs: 1000,
  });
  assert.deepEqual(slow, { kind: 'timeout' });
  let cancel = false;
  const c = await pollWardResult({ fetchResult: async () => { cancel = true; return { status: 'processing' }; }, sleep, isCancelled: () => cancel });
  assert.deepEqual(c, { kind: 'cancelled' });
});

test('업로드 API: phrase_set은 넘길 때만 보낸다(기본 경로 요청은 그대로), 모르는 묶음은 보내기 전에 거절', async () => {
  const { PHRASE_SETS, apiClient, uploadRecording } = await import('../src/api/client.js');
  assert.deepEqual(PHRASE_SETS, ['sentences', 'ward']);
  const posts = [];
  const realPost = apiClient.post;
  apiClient.post = async (url, form, config) => {
    posts.push({ url, fields: [...form.entries()], config });
    return { data: { job_id: 'j1', session_id: 's1', status: 'processing', duration_sec: 5 } };
  };
  try {
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'video/webm;codecs=vp9' });
    assert.deepEqual(await uploadRecording(blob, 'video/webm;codecs=vp9', () => {}),
      { job_id: 'j1', session_id: 's1', status: 'processing', duration_sec: 5 });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'sentences' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: null });
    await uploadRecording(blob, 'video/webm', undefined, undefined);
    assert.ok(posts.every((x) => x.url === '/api/record'));
    assert.deepEqual(posts.map((x) => x.fields.map(([k, v]) => (k === 'file' ? 'file' : `${k}=${v}`))), [
      ['file'], ['file', 'phrase_set=ward'], ['file', 'phrase_set=sentences'], ['file'], ['file'],
    ], '묶음을 주지 않으면 phrase_set 필드 자체가 없다(서버 기본 sentences와 같은 요청)');
    const file = posts[0].fields[0][1];
    assert.equal(file.name, 'recording.webm');
    assert.equal(file.type, 'video/webm', '코덱 매개변수를 뗀 기본형');
    for (const bad of ['nurse', 'WARD', 1]) {
      await assert.rejects(uploadRecording(blob, 'video/webm', undefined, { phraseSet: bad }),
        (e) => e.isClientCheck === true && /지원하지 않는 문장 묶음/.test(e.message), String(bad));
    }
    await assert.rejects(uploadRecording(new Blob([]), 'video/webm', undefined, { phraseSet: 'ward' }), (e) => e.isClientCheck === true);
    assert.equal(posts.length, 5, '거절한 요청은 서버로 보내지 않는다');
  } finally {
    apiClient.post = realPost;
  }
});

test('호출 쪽 계약: 녹화 화면은 묶음 없이, 병실 화면은 ward로 올린다. App에 #ward·#nurse 경로', () => {
  const record = src('../src/pages/RecordPage.jsx');
  assert.match(record, /uploadRecording\(blob, mime, \(p\) =>/, '녹화 화면은 phrase_set 없이 호출(문장 모드 그대로)');
  const ward = src('../src/pages/WardPage.jsx');
  const upload = src('../src/ward/wardUpload.js');
  // 병실 화면·등록 화면은 같은 올리기 함수를 쓴다: phrase_set='ward' + (설정한 경우) 등록 이름 + (내 목록이면) 목록 ID.
  assert.match(upload, /upload\(blob, mime, undefined, wardRecordOptions\(\{\s*enrollProfile, enrollCommand, commandList, commandListSha256,\s*\}\)\)/);
  assert.match(upload, /const options = \{ phraseSet: WARD_PHRASE_SET \}/);
  // 등록 이름은 목록마다 따로다: 기본 목록('')이면 예전과 같은 settings.enrollProfile이다(enrollProfileFor).
  assert.match(ward, /analyzeWardRecording\(\{\s*blob, mime, durationMs: meta\?\.durationMs, enrollProfile: enrollProfileFor\(machineRef\.current\.settings, commandList\)/);
  assert.doesNotMatch(ward, /enrollCommand/, '병실 화면은 등록(저장)하지 않고 비교만 한다');
  assert.doesNotMatch(ward, /emergency.*accepted.*margin|OLKAVS_PHRASE_MIN|possible_min_score|enroll_max_distance/, '화면 파일이 기준값을 다시 판정하지 않는다');
  const enroll = src('../src/pages/EnrollPage.jsx');
  assert.match(enroll, /analyzeWardRecording\(\{\s*blob, mime, durationMs: meta\?\.durationMs, enrollProfile: profile, enrollCommand: s\.recordingId/);
  assert.match(enroll, /useRecorder\(\{ maxSeconds: ENROLL_RECORD_SECONDS/);
  const app = src('../src/App.jsx');
  assert.match(app, /'#ward': 'ward', '#nurse': 'nurse'/);
  assert.match(app, /'#enroll': 'enroll'/);
  assert.match(app, /split\('\?'\)\[0\]/);
});
