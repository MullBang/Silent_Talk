// wardEnroll.test.js — 내 입 모양 등록 검사 (node --test): 등록 화면 순수 함수·상태도, 등록 API 호출 모양, 병실 화면과 같이 쓰는 올리기 함수.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { buildCommandIndex } from '../src/ward/wardCommands.js';
import {
  ENROLL_RECORD_SECONDS, ENROLL_TEXT, STORE_REASONS, STORE_REASON_TEXT, canStartEnrollRecording, commandRows, createEnrollState,
  enrollHash, enrollStatusLine, nextMissingId, parseProfileInfo, profileChoices, profileFromHash, profileUsable,
  shouldAutoUseProfile, stepEnroll, storeOutcome, uncertainCommands, wardReturnHash,
} from '../src/ward/wardEnroll.js';
import { isPracticeMode } from '../src/ward/wardSettings.js';
import {
  WARD_TEXT, createWardState, enrollOpenBlockReason, stepWard,
} from '../src/ward/wardMachine.js';
import { analyzeWardRecording, wardRecordOptions } from '../src/ward/wardUpload.js';

const src = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const commands = JSON.parse(src('../src/ward/commands.json'));
const INDEX = buildCommandIndex(commands);
const IDS = INDEX.list.map((c) => c.id);

/** 서버 등록 상태 모양 (찍은 명령 ID 목록으로). */
function profileData(done, { status, profile = 'bed305', mismatch = [] } = {}) {
  const templates = Object.fromEntries(done.map((id) => [id, {
    created: '2026-11-10T10:00:00', text_top: mismatch.includes(id) ? 'W18' : id, text_top_matches: !mismatch.includes(id),
  }]));
  const complete = done.length === IDS.length;
  return { profile, count: done.length, complete, status: status || (complete ? 'ok' : (done.length ? 'incomplete' : 'missing')), templates };
}

/** 저장 결과(results) 모양. */
function stored(command, { matches = true, count = 1, complete = false, top } = {}) {
  return [{
    start_ms: 0, end_ms: 4000, text: 'x', phrase: { id: top || command, set: 'ward' },
    enroll: { mode: 'store', profile: 'bed305', command, stored: true, reason: null, text_top: top || command, text_top_matches: matches, count, complete },
  }];
}
function failed(command, reason) {
  return [{
    start_ms: 0, end_ms: 4000, text: reason === 'no_speech' ? '[발화 없음]' : 'x',
    enroll: { mode: 'store', profile: 'bed305', command, stored: false, reason, text_top: null, text_top_matches: false, count: 0, complete: false },
  }];
}

/** 이벤트 여러 개. */
const feed = (s, events) => events.reduce((acc, e) => stepEnroll(acc, e), s);

test('주소: #enroll?profile=이름, 규칙에 맞지 않는 이름은 버린다', () => {
  assert.equal(profileFromHash('#enroll?profile=bed305'), 'bed305');
  assert.equal(profileFromHash('#enroll'), '');
  assert.equal(profileFromHash('#enroll?profile=..%2Fetc'), '');
  assert.equal(profileFromHash('#enroll?profile=%EA%B0%80'), '', '한글 이름은 서버 규칙 밖');
  assert.equal(profileFromHash(null), '');
  assert.equal(enrollHash('bed305'), '#enroll?profile=bed305');
  assert.equal(enrollHash(''), '#enroll');
  assert.equal(enrollHash('../x'), '#enroll');
  assert.equal(ENROLL_RECORD_SECONDS, 5, '병실 명령 촬영·병실 화면 기본과 같은 5초');
});

test('연습 모드: #ward?practice=1에서 등록 화면에 다녀와도 연습 모드 그대로', () => {
  const hash = enrollHash('bed305', { practice: true });
  assert.equal(hash, '#enroll?profile=bed305&practice=1');
  assert.equal(profileFromHash(hash), 'bed305');
  assert.equal(isPracticeMode({ hash, search: '' }), true, '등록 화면도 연습 모드로 안다');
  assert.equal(enrollHash('', { practice: true }), '#enroll?practice=1');
  assert.equal(wardReturnHash({ hash }), '#ward?practice=1');
  assert.equal(isPracticeMode({ hash: wardReturnHash({ hash }), search: '' }), true);
  assert.equal(wardReturnHash({ hash: '#enroll?profile=bed305' }), '#ward');
  assert.equal(wardReturnHash({ hash: '#enroll', search: '?practice=1' }), '#ward', '?practice=1#ward 형식은 주소 앞부분이 그대로 남는다');
  assert.equal(wardReturnHash(null), '#ward');
});

test('등록 상태 읽기: 찍은 명령·개수·완료·상태, 목록 밖 명령은 버린다', () => {
  const info = parseProfileInfo(profileData(['W01', 'W05'], { mismatch: ['W05'] }), INDEX);
  assert.equal(info.profile, 'bed305');
  assert.equal(info.count, 2);
  assert.equal(info.total, 20);
  assert.equal(info.complete, false);
  assert.equal(info.status, 'incomplete');
  assert.deepEqual(Object.keys(info.done), ['W01', 'W05']);
  assert.equal(info.done.W05.matches, false);
  assert.equal(info.done.W05.textTop, 'W18');
  const odd = parseProfileInfo({ profile: 'x', templates: { A01: {}, W02: {} } }, INDEX);
  assert.deepEqual(Object.keys(odd.done), ['W02']);
  assert.equal(odd.count, 1);
  const missing = parseProfileInfo({ profile: 'new1', count: 0, complete: false, status: 'missing', templates: {} }, INDEX);
  assert.equal(missing.status, 'missing');
  assert.equal(profileUsable(missing), false);
  assert.equal(profileUsable(parseProfileInfo(profileData(IDS), INDEX)), true);
  assert.equal(profileUsable(parseProfileInfo(profileData(IDS, { status: 'stale' }), INDEX)), false, '다시 등록 필요');
  assert.equal(parseProfileInfo(null, INDEX, 'p').profile, 'p');
});

test('다음에 찍을 명령: 지금 다음부터 빠진 것, 끝이면 처음으로, 모두 있으면 없음', () => {
  assert.equal(nextMissingId(INDEX, {}, null), 'W01');
  assert.equal(nextMissingId(INDEX, { W01: {} }, null), 'W02');
  assert.equal(nextMissingId(INDEX, { W01: {}, W02: {} }, 'W01'), 'W03');
  assert.equal(nextMissingId(INDEX, { W20: {} }, 'W20'), 'W01', '끝에서 처음으로');
  assert.equal(nextMissingId(INDEX, { W05: {} }, 'W05'), 'W06');
  const allButW03 = Object.fromEntries(IDS.filter((x) => x !== 'W03').map((x) => [x, {}]));
  assert.equal(nextMissingId(INDEX, allButW03, 'W10'), 'W03');
  assert.equal(nextMissingId(INDEX, allButW03, 'W03'), null, '지금 것만 빠졌으면 다음은 없음');
  assert.equal(nextMissingId(INDEX, Object.fromEntries(IDS.map((x) => [x, {}])), 'W07'), null);
});

test('저장 결과 읽기: 저장·인식 문장이 다름·실패 사유·옛 서버', () => {
  const ok = storeOutcome(stored('W05', { count: 3 }), 'W05');
  assert.equal(ok.kind, 'stored');
  assert.equal(ok.matches, true);
  assert.equal(ok.count, 3);
  assert.equal(ok.message, 'W05 저장했어요');
  const mis = storeOutcome(stored('W04', { matches: false, top: 'W10' }), 'W04', INDEX);
  assert.equal(mis.kind, 'stored');
  assert.equal(mis.matches, false);
  assert.equal(mis.textTop, 'W10');
  assert.equal(mis.message, 'W04 저장은 했지만 “토할 것 같아요”(W10) 문장으로 알아들었어요. 화면의 큰 문장을 또박또박 다시 말해 찍어 주세요');
  assert.doesNotMatch(mis.message, /\(으\)로/, '기계적인 조사 표기를 쓰지 않는다');
  const unknown = storeOutcome(stored('W05', { matches: false, top: 'A01' }), 'W05', INDEX);
  assert.match(unknown.message, /다른 문장으로 알아들었어요/, '목록 밖 ID는 문장을 모르므로 "다른 문장으로"');
  assert.match(storeOutcome(stored('W05', { matches: false, top: 'W18' }), 'W05').message, /다른 문장으로/, '색인이 없으면 "다른 문장으로"');
  for (const reason of STORE_REASONS) {
    const f = storeOutcome(failed('W05', reason), 'W05');
    assert.equal(f.kind, 'failed', reason);
    assert.equal(f.reason, reason);
    assert.equal(f.message, STORE_REASON_TEXT[reason]);
    assert.doesNotMatch(f.message, /[A-Za-z]/, `${reason}: 한국어 문구`);
  }
  assert.equal(storeOutcome(failed('W05', 'weird'), 'W05').reason, 'error');
  assert.equal(storeOutcome([{ text: 'x' }], 'W05').kind, 'unsupported', '옛 서버는 enroll이 없다');
  assert.equal(storeOutcome([{ text: 'x', enroll: { mode: 'match', status: 'ok' } }], 'W05').kind, 'unsupported');
  assert.equal(storeOutcome(stored('W06'), 'W05').kind, 'failed', '다른 명령의 저장 결과는 받아들이지 않는다');
  assert.equal(storeOutcome(null, 'W05').kind, 'unsupported');
});

test('등록 상태 한 줄: 20/20·덜 찍음·다시 등록 필요·기록 없음·확인 중·서버 오류', () => {
  assert.equal(enrollStatusLine('', null), null, '등록 안 씀이면 줄이 없다');
  assert.deepEqual(enrollStatusLine('bed305', parseProfileInfo(profileData(IDS), INDEX)), { text: '내 입 모양 등록: bed305 (20/20)', tone: 'ok' });
  const part = enrollStatusLine('bed305', parseProfileInfo(profileData(IDS.slice(0, 12)), INDEX));
  assert.equal(part.text, '내 입 모양 등록: bed305 (12/20) — 다 찍지 않아 아직 쓰지 않아요');
  assert.equal(part.tone, 'warn');
  assert.match(enrollStatusLine('bed305', parseProfileInfo(profileData(IDS, { status: 'stale' }), INDEX)).text, /\(20\/20\) — 명령 목록이나 인식 모델이 바뀌어 지금은 쓰지 않아요/);
  assert.match(enrollStatusLine('bed305', parseProfileInfo(profileData([]), INDEX)).text, /등록 기록이 없어요/);
  assert.equal(enrollStatusLine('bed305', null).text, '내 입 모양 등록: bed305 (확인 중)');
  assert.equal(enrollStatusLine('bed305', null, { loadError: true }).tone, 'warn');
});

test('등록 목록 고르기·명령 행 표시', () => {
  const list = profileChoices({ profiles: [{ profile: 'zz', count: 20, complete: true, status: 'ok' }, { profile: 'a1', count: 3 }, { profile: '../x' }, null] }, 'mine');
  assert.deepEqual(list.map((p) => p.profile), ['a1', 'mine', 'zz'], '이름순, 규칙 밖 이름은 버리고 지금 이름은 넣는다');
  assert.equal(list[0].count, 3);
  assert.equal(list[1].count, null);
  assert.deepEqual(profileChoices(null), []);
  const rows = commandRows(INDEX, parseProfileInfo(profileData(['W01', 'W02'], { mismatch: ['W02'] }), INDEX), 'W03');
  assert.equal(rows.length, 20);
  assert.deepEqual(rows.slice(0, 3).map((r) => [r.id, r.done, r.warn, r.current]), [['W01', true, false, false], ['W02', true, true, false], ['W03', false, false, true]]);
  assert.equal(rows[0].text, '간호사 선생님 좀 불러 주세요', '말할 문장은 인식용 text');
  assert.equal(rows[0].display, '간호사를 불러 주세요');
});

test('등록 화면 상태도: 이름 고르기 → 불러오기 → 첫 빈 명령 → 녹화 → 저장 → 다음 빈 명령', () => {
  let s = createEnrollState({ index: INDEX });
  assert.equal(s.phase, 'choose');
  s = stepEnroll(s, { type: 'CHOOSE', profile: '가나' });
  assert.equal(s.phase, 'choose');
  assert.equal(s.error, ENROLL_TEXT.badProfile);
  s = stepEnroll(s, { type: 'CHOOSE', profile: ' bed305 ' });
  assert.equal(s.phase, 'loading');
  assert.equal(s.profile, 'bed305');
  s = stepEnroll(s, { type: 'PROFILE_LOADED', data: profileData(['W01']) });
  assert.equal(s.phase, 'ready');
  assert.equal(s.currentId, 'W02');
  assert.equal(stepEnroll(s, { type: 'RECORD_DONE' }), s, '녹화 전 RECORD_DONE은 무시');
  s = stepEnroll(s, { type: 'RECORD_START' });
  assert.equal(s.phase, 'recording');
  assert.equal(s.recordingId, 'W02');
  assert.equal(stepEnroll(s, { type: 'SELECT', commandId: 'W07' }), s, '녹화 중에는 다른 명령을 고르지 않는다');
  assert.equal(stepEnroll(s, { type: 'RECORD_START' }), s);
  s = stepEnroll(s, { type: 'RECORD_DONE' });
  assert.equal(s.phase, 'analyzing');
  s = stepEnroll(s, { type: 'STORE_RESULT', results: stored('W02', { count: 2 }) });
  assert.equal(s.phase, 'ready');
  assert.equal(s.currentId, 'W03', '다음 빈 명령으로');
  assert.equal(s.info.count, 2);
  assert.equal(s.info.done.W02.matches, true);
  assert.equal(s.last.message, 'W02 저장했어요');
  assert.equal(s.recordingId, null);
});

test('등록 화면 상태도: 인식 문장이 다르면 그 명령에 머물러 다시 찍기·다음으로', () => {
  let s = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [
    { type: 'PROFILE_LOADED', data: profileData([]) }, { type: 'RECORD_START' }, { type: 'RECORD_DONE' },
    { type: 'STORE_RESULT', results: stored('W01', { matches: false, top: 'W16' }) },
  ]);
  assert.equal(s.phase, 'review');
  assert.equal(s.currentId, 'W01');
  assert.equal(s.info.done.W01.matches, false, '서버는 저장했다');
  // 다시 찍기: 같은 명령, 스페이스바로 바로 다시 녹화할 수 있다
  assert.equal(canStartEnrollRecording(s, { cameraReady: true, coolingDown: false }), true);
  const retake = stepEnroll(s, { type: 'RETAKE' });
  assert.equal(retake.phase, 'ready');
  assert.equal(retake.currentId, 'W01');
  const again = feed(s, [{ type: 'RECORD_START' }, { type: 'RECORD_DONE' }, { type: 'STORE_RESULT', results: stored('W01', { count: 1 }) }]);
  assert.equal(again.phase, 'ready');
  assert.equal(again.currentId, 'W02');
  assert.equal(again.info.done.W01.matches, true, '다시 찍으면 바뀐다(명령마다 하나)');
  // 다음으로: 다음 빈 명령
  const next = stepEnroll(s, { type: 'NEXT' });
  assert.equal(next.phase, 'ready');
  assert.equal(next.currentId, 'W02');
});

test('등록 화면 상태도: 저장 실패·녹화 실패·분석 오류는 같은 명령에서 다시 찍기', () => {
  const base = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [
    { type: 'PROFILE_LOADED', data: profileData(['W01']) }, { type: 'RECORD_START' }, { type: 'RECORD_DONE' },
  ]);
  const f = stepEnroll(base, { type: 'STORE_RESULT', results: failed('W02', 'partial_detection') });
  assert.equal(f.phase, 'ready');
  assert.equal(f.currentId, 'W02');
  assert.equal(f.last.kind, 'failed');
  assert.equal(f.last.message, STORE_REASON_TEXT.partial_detection);
  assert.equal(f.info.count, 1, '실패는 개수를 바꾸지 않는다');
  const e = stepEnroll(base, { type: 'ANALYSIS_ERROR', message: '서버에 연결할 수 없습니다.' });
  assert.equal(e.phase, 'ready');
  assert.equal(e.last.message, '서버에 연결할 수 없습니다.');
  assert.equal(e.last.commandId, 'W02');
  const r = stepEnroll(feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [{ type: 'PROFILE_LOADED', data: profileData([]) }, { type: 'RECORD_START' }]),
    { type: 'RECORD_FAILED', message: '녹화가 너무 짧습니다.' });
  assert.equal(r.phase, 'ready');
  assert.equal(r.last.reason, 'record');
  const u = stepEnroll(base, { type: 'STORE_RESULT', results: [{ text: 'x' }] });
  assert.equal(u.last.kind, 'unsupported');
  assert.equal(u.last.message, ENROLL_TEXT.unsupported);
});

test('등록 화면 상태도: 20번째 저장이면 끝 화면, 끝난 등록을 다시 열어도 끝 화면, 다시 등록 필요면 찍는 화면', () => {
  const nineteen = IDS.slice(0, 19);
  let s = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [{ type: 'PROFILE_LOADED', data: profileData(nineteen) }]);
  assert.equal(s.currentId, 'W20');
  s = feed(s, [{ type: 'RECORD_START' }, { type: 'RECORD_DONE' }, { type: 'STORE_RESULT', results: stored('W20', { count: 20, complete: true }) }]);
  assert.equal(s.phase, 'done');
  assert.equal(profileUsable(s.info), true);
  // 서버가 다시 알려 준 상태가 stale이면 끝 화면에서 찍는 화면으로
  const staleBack = stepEnroll(s, { type: 'PROFILE_LOADED', data: profileData(IDS, { status: 'stale' }) });
  assert.equal(staleBack.phase, 'ready');
  // 끝 화면에서 명령을 고르면 다시 찍을 수 있다
  const pick = stepEnroll(s, { type: 'SELECT', commandId: 'W07' });
  assert.equal(pick.phase, 'ready');
  assert.equal(pick.currentId, 'W07');
  assert.equal(stepEnroll(s, { type: 'SELECT', commandId: 'A01' }), s, '목록 밖 명령은 무시');
  // 이미 끝난 등록을 열면 바로 끝 화면
  const reopened = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [{ type: 'PROFILE_LOADED', data: profileData(IDS) }]);
  assert.equal(reopened.phase, 'done');
  // 다시 등록 필요(stale)는 20/20이어도 찍는 화면에 머물지만 녹화는 막는다(하나를 찍으면 서버가 나머지를 지운다).
  const stale = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [{ type: 'PROFILE_LOADED', data: profileData(IDS, { status: 'stale' }) }]);
  assert.equal(stale.phase, 'ready');
  assert.equal(stale.currentId, 'W01');
  assert.equal(canStartEnrollRecording(stale, { cameraReady: true, coolingDown: false }), false, 'stale은 녹화 버튼·스페이스바가 안 된다');
  assert.equal(stepEnroll(stale, { type: 'RECORD_START' }), stale, 'stale에서 RECORD_START는 무시');
  // (경합으로) stale 등록에 저장 결과가 오면 서버처럼 예전 템플릿을 지운 것으로 보고 다음 빈 명령으로 간다
  const raced = stepEnroll({ ...stale, phase: 'analyzing', recordingId: 'W01' }, { type: 'STORE_RESULT', results: stored('W01', { count: 1 }) });
  assert.deepEqual(Object.keys(raced.info.done), ['W01']);
  assert.equal(raced.info.status, 'incomplete');
  assert.equal(raced.currentId, 'W02');
  assert.equal(canStartEnrollRecording(raced, { cameraReady: true, coolingDown: false }), true);
  // 마지막 명령이 인식 문장 다름이면 다시 찍기를 고를 수 있게 머물고, 다음으로를 누르면 끝 화면
  const lastMismatch = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [
    { type: 'PROFILE_LOADED', data: profileData(nineteen) }, { type: 'RECORD_START' }, { type: 'RECORD_DONE' },
    { type: 'STORE_RESULT', results: stored('W20', { matches: false, count: 20, complete: true }) },
  ]);
  assert.equal(lastMismatch.phase, 'review');
  assert.equal(stepEnroll(lastMismatch, { type: 'NEXT' }).phase, 'done');
});

test('등록 화면 상태도: 불러오기 실패·다른 이름·스페이스바 조건', () => {
  const loading = createEnrollState({ index: INDEX, profile: 'bed305' });
  const failedLoad = stepEnroll(loading, { type: 'PROFILE_FAILED', message: '서버에 연결할 수 없습니다.' });
  assert.equal(failedLoad.phase, 'choose');
  assert.match(failedLoad.error, /등록 상태를 가져오지 못했어요: 서버에 연결할 수 없습니다/);
  const ready = stepEnroll(loading, { type: 'PROFILE_LOADED', data: profileData([]) });
  assert.equal(stepEnroll(ready, { type: 'RESET' }).phase, 'choose');
  const rec = stepEnroll(ready, { type: 'RECORD_START' });
  assert.equal(stepEnroll(rec, { type: 'RESET' }), rec, '녹화 중에는 다른 이름으로 바꾸지 않는다');
  assert.equal(stepEnroll(rec, { type: 'CHOOSE', profile: 'other' }), rec);
  assert.equal(stepEnroll(ready, { type: 'CHOOSE', profile: 'other' }).phase, 'loading');
  assert.equal(canStartEnrollRecording(ready, { cameraReady: true, coolingDown: false }), true);
  assert.equal(canStartEnrollRecording(ready, { cameraReady: false, coolingDown: false }), false, '카메라 준비 전');
  assert.equal(canStartEnrollRecording(ready, { cameraReady: true, coolingDown: true }), false, '결과 직후 잠깐');
  assert.equal(canStartEnrollRecording(rec, { cameraReady: true, coolingDown: false }), false, '녹화 중');
  assert.equal(canStartEnrollRecording(loading, { cameraReady: true, coolingDown: false }), false);
  assert.equal(stepEnroll(ready, null), ready);
  assert.equal(stepEnroll(ready, { type: 'NOPE' }), ready);
});

test('올리기 API: 등록 이름·명령은 병실 묶음에서만, 비면 보내지 않고, 형식이 틀리면 보내기 전에 거절', async () => {
  const { apiClient, deleteWardProfile, enrollFieldProblem, getWardProfile, getWardProfiles, uploadRecording } = await import('../src/api/client.js');
  const posts = [];
  const gets = [];
  const deletes = [];
  const real = { post: apiClient.post, get: apiClient.get, delete: apiClient.delete };
  apiClient.post = async (url, form) => {
    posts.push({ url, fields: [...form.entries()].map(([k, v]) => (k === 'file' ? 'file' : `${k}=${v}`)) });
    return { data: { job_id: 'j', session_id: 's', status: 'processing' } };
  };
  apiClient.get = async (url) => {
    gets.push(url);
    if (url.endsWith('/nobody')) throw Object.assign(new Error('404'), { response: { status: 404 } });
    if (url.endsWith('/broken')) throw Object.assign(new Error('500'), { response: { status: 500 } });
    if (url === '/api/ward/profiles') return { data: { profiles: [{ profile: 'bed305', count: 20, complete: true, status: 'ok', updated: 'x' }] } };
    return { data: { profile: 'bed305', count: 20, complete: true, status: 'ok', templates: {} } };
  };
  apiClient.delete = async (url) => { deletes.push(url); return { data: { deleted: true } }; };
  try {
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'video/webm' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward', enrollProfile: 'bed305' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward', enrollProfile: 'bed305', enrollCommand: 'W05' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward', enrollProfile: '', enrollCommand: '' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'sentences' });
    assert.deepEqual(posts.map((x) => x.fields), [
      ['file', 'phrase_set=ward', 'enroll_profile=bed305'],
      ['file', 'phrase_set=ward', 'enroll_profile=bed305', 'enroll_command=W05'],
      ['file', 'phrase_set=ward'],
      ['file', 'phrase_set=sentences'],
    ], '문장 모드 요청에는 등록 필드가 없다');
    const bad = [
      [{ phraseSet: 'sentences', enrollProfile: 'bed305' }, /병실 명령 녹화에서만/],
      [{ enrollProfile: 'bed305' }, /병실 명령 녹화에서만/],
      [{ phraseSet: 'ward', enrollCommand: 'W05' }, /등록 이름이 필요/],
      [{ phraseSet: 'ward', enrollProfile: '../x' }, /영문·숫자/],
      [{ phraseSet: 'ward', enrollProfile: 'x'.repeat(33) }, /영문·숫자/],
      [{ phraseSet: 'ward', enrollProfile: 'bed305', enrollCommand: 'A01' }, /명령 ID/],
    ];
    for (const [options, re] of bad) {
      await assert.rejects(uploadRecording(blob, 'video/webm', undefined, options), (e) => e.isClientCheck === true && re.test(e.message), JSON.stringify(options));
    }
    assert.equal(posts.length, 4, '거절한 요청은 보내지 않는다');
    assert.equal(enrollFieldProblem('ward', '', ''), null);
    assert.equal(enrollFieldProblem(undefined, undefined, undefined), null);

    assert.deepEqual((await getWardProfiles()).profiles.map((p) => p.profile), ['bed305']);
    assert.equal((await getWardProfile('bed305')).status, 'ok');
    assert.deepEqual(await getWardProfile('nobody'), { profile: 'nobody', count: 0, complete: false, status: 'missing', templates: {} }, '404는 기록 없음');
    await assert.rejects(getWardProfile('broken'), (e) => e.response.status === 500, '다른 오류는 그대로');
    assert.deepEqual(await deleteWardProfile('bed305'), { deleted: true });
    assert.deepEqual(gets, ['/api/ward/profiles', '/api/ward/profiles/bed305', '/api/ward/profiles/nobody', '/api/ward/profiles/broken']);
    assert.deepEqual(deletes, ['/api/ward/profiles/bed305']);
    for (const badId of ['../etc', 'a/b', '', 'x'.repeat(33)]) {
      await assert.rejects(getWardProfile(badId), (e) => e.isClientCheck === true, badId);
      await assert.rejects(deleteWardProfile(badId), (e) => e.isClientCheck === true, badId);
    }
    assert.equal(gets.length, 4);
    assert.equal(deletes.length, 1, '이상한 이름은 서버로 보내지 않는다');
  } finally {
    Object.assign(apiClient, real);
  }
});

test('같이 쓰는 올리기 함수: 1차 검사·phrase_set ward·등록 이름·결과/오류/시간 초과/취소', async () => {
  assert.deepEqual(wardRecordOptions(), { phraseSet: 'ward' });
  assert.deepEqual(wardRecordOptions({ enrollProfile: '' }), { phraseSet: 'ward' }, '등록 안 씀이면 오늘과 같은 요청');
  assert.deepEqual(wardRecordOptions({ enrollProfile: 'bed305' }), { phraseSet: 'ward', enrollProfile: 'bed305' });
  assert.deepEqual(wardRecordOptions({ enrollProfile: 'bed305', enrollCommand: 'W05' }), { phraseSet: 'ward', enrollProfile: 'bed305', enrollCommand: 'W05' });
  assert.deepEqual(wardRecordOptions({ enrollCommand: 'W05' }), { phraseSet: 'ward' }, '이름 없는 명령은 보내지 않는다');
  const blob = new Blob([new Uint8Array([1])], { type: 'video/webm' });
  const calls = [];
  const deps = (pollOutcome, data = { session_id: 's9' }) => ({
    upload: async (...args) => { calls.push(args); return data; },
    fetchResult: async (sid) => ({ sid }),
    poll: async ({ fetchResult }) => { calls.push(['poll', await fetchResult()]); return pollOutcome; },
  });
  const done = await analyzeWardRecording({ blob, mime: 'video/webm', durationMs: 5000, enrollProfile: 'bed305', deps: deps({ kind: 'done', results: [{ text: 'x' }], warnings: [] }) });
  assert.deepEqual(done, { kind: 'done', results: [{ text: 'x' }], warnings: [] });
  assert.deepEqual(calls[0].slice(2), [undefined, { phraseSet: 'ward', enrollProfile: 'bed305' }]);
  assert.deepEqual(calls[1], ['poll', { sid: 's9' }]);
  assert.equal((await analyzeWardRecording({ blob, mime: 'video/webm', durationMs: 300, deps: deps({ kind: 'done', results: [] }) })).kind, 'invalid', '1초 미만');
  assert.deepEqual(await analyzeWardRecording({ blob, mime: 'video/webm', durationMs: 5000, deps: deps({ kind: 'error', message: '실패' }) }), { kind: 'error', message: '실패' });
  assert.deepEqual(await analyzeWardRecording({ blob, mime: 'video/webm', durationMs: 5000, deps: deps({ kind: 'timeout' }) }), { kind: 'timeout' });
  assert.deepEqual(await analyzeWardRecording({ blob, mime: 'video/webm', durationMs: 5000, deps: deps({ kind: 'cancelled' }) }), { kind: 'cancelled' });
  const noSid = await analyzeWardRecording({ blob, mime: 'video/webm', durationMs: 5000, deps: deps({ kind: 'done' }, {}) });
  assert.equal(noSid.kind, 'error');
  assert.doesNotMatch(noSid.message, /session_id/, '영문 내부 문구를 보이지 않는다');
  const thrown = await analyzeWardRecording({
    blob, mime: 'video/webm', durationMs: 5000,
    deps: { upload: async () => { throw Object.assign(new Error('x'), { response: { status: 400, data: { detail: '입 모양 등록은 병실 명령 녹화에서만 쓸 수 있습니다.' } } }); } },
  });
  assert.deepEqual(thrown, { kind: 'error', message: '입 모양 등록은 병실 명령 녹화에서만 쓸 수 있습니다.' }, '서버 400 한국어 문구는 그대로');
  let cancel = false;
  const cancelled = await analyzeWardRecording({
    blob, mime: 'video/webm', durationMs: 5000, isCancelled: () => cancel,
    deps: { upload: async () => { cancel = true; return { session_id: 's' }; } },
  });
  assert.deepEqual(cancelled, { kind: 'cancelled' });
});

test('화면 연결: #enroll 화면은 설정에서 열고, 끝나면 설정 등록 이름을 바꾸고 병실 화면으로 돌아간다', () => {
  const page = src('../src/pages/EnrollPage.jsx');
  assert.match(page, /saveSettings\(storage, \{ \.\.\.current, enrollProfile: profile \}\)/);
  assert.match(page, /shouldAutoUseProfile\(state\)/, '끝 화면 자동 사용은 이 화면에서 저장해 마쳤고 확인 필요가 없을 때만');
  assert.match(page, /state\.phase !== 'done'/);
  assert.match(page, /globalThis\.location\.hash = wardReturnHash\(globalThis\.location\)/);
  assert.match(page, /outcome\.kind === 'error' \|\| outcome\.kind === 'timeout'[\s\S]{0,240}loadProfile\(profile\)/, '오류·시간 초과 뒤에도 서버 상태를 다시 읽는다');
  assert.match(page, /e\.code !== 'Space'/, '스페이스바로 녹화');
  assert.match(page, /command\.text/, '말할 문장은 인식용 text');
  assert.match(page, /ENROLL_TEXT\.howTo/);
  assert.doesNotMatch(page, /localStorage\.setItem|sessionStorage/, '등록 자료를 브라우저에 저장하지 않는다');
  const panel = src('../src/ward/WardSettingsPanel.jsx');
  assert.match(panel, /<option value="">등록 안 씀<\/option>/);
  assert.match(panel, /globalThis\.confirm\?\.\(ENROLL_TEXT\.deleteConfirm\(current\)\)/, '지우기 전에 확인');
  assert.match(panel, /deleteWardProfile\(current\)/);
  const ward = src('../src/pages/WardPage.jsx');
  assert.match(ward, /enrollStatusLine\(enrollProfile/);
  assert.match(ward, /onOpenEnroll=\{openEnroll\}/);
  assert.match(ward, /enrollBlocked=\{enrollOpenBlockReason\(machine\)\}/, '진행 중이면 등록 화면 버튼을 막는다');
  assert.match(ward, /if \(enrollOpenBlockReason\(machineRef\.current\)\) return;/);
  assert.match(ward, /enrollHash\(profile, \{ practice \}\)/);
  assert.match(ward, /ch\.post\('ROOM_CLOSED', \{ windowId: machineRef\.current\.sessionKey \}\);\s*ch\.close\(\)/, '화면을 내릴 때도 병실 줄을 지운다');
  assert.match(panel, /disabled=\{!!blocked\}/);
  const nurse = src('../src/pages/NursePage.jsx');
  assert.match(nurse, /cardEnrollNote\(card\)/);
  assert.match(ENROLL_TEXT.howTo, /소리 내어 평소처럼 한 번 말하세요/);
});

test('끝 화면: 확인 필요 목록과 자동 사용 조건 (이 화면에서 저장해 마쳤고 확인 필요가 없을 때만)', () => {
  const nineteen = IDS.slice(0, 19);
  // 이 화면에서 20번째를 저장해 마침, 확인 필요 없음 → 자동 사용
  const fresh = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [
    { type: 'PROFILE_LOADED', data: profileData(nineteen) }, { type: 'RECORD_START' }, { type: 'RECORD_DONE' },
    { type: 'STORE_RESULT', results: stored('W20', { count: 20, complete: true }) },
  ]);
  assert.equal(fresh.phase, 'done');
  assert.equal(fresh.storedHere, true);
  assert.equal(shouldAutoUseProfile(fresh), true);
  // 이미 끝난 등록을 열어 보기만 함 → 자동 사용 안 함(버튼을 눌러야)
  const reopened = feed(createEnrollState({ index: INDEX, profile: 'p01x' }), [{ type: 'PROFILE_LOADED', data: profileData(IDS) }]);
  assert.equal(reopened.phase, 'done');
  assert.equal(reopened.storedHere, false);
  assert.equal(shouldAutoUseProfile(reopened), false);
  // 확인 필요가 있으면 자동 사용 안 함, 목록에 그 명령과 알아들은 문장
  const warned = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [
    { type: 'PROFILE_LOADED', data: profileData(nineteen, { mismatch: ['W04', 'W17'] }) }, { type: 'RECORD_START' }, { type: 'RECORD_DONE' },
    { type: 'STORE_RESULT', results: stored('W20', { count: 20, complete: true }) },
  ]);
  assert.equal(warned.phase, 'done');
  assert.equal(shouldAutoUseProfile(warned), false);
  const list = uncertainCommands(INDEX, warned.info);
  assert.deepEqual(list.map((u) => [u.command.id, u.heardAs?.id]), [['W04', 'W18'], ['W17', 'W18']]);
  assert.equal(ENROLL_TEXT.uncertainItem(list[0].command, list[0].heardAs),
    `W04 “${INDEX.byId.get('W04').display}” [긴급] → “${INDEX.byId.get('W18').display}”(W18)로 알아들음`);
  assert.match(ENROLL_TEXT.uncertainItem(list[0].command, null), /다른 문장으로 알아들음/);
  assert.match(ENROLL_TEXT.uncertainTitle(2), /확인 필요 2개/);
  // 끝 화면에서 확인 필요 명령의 [다시 찍기] = SELECT
  const retake = stepEnroll(warned, { type: 'SELECT', commandId: 'W04' });
  assert.equal(retake.phase, 'ready');
  assert.equal(retake.currentId, 'W04');
  assert.equal(shouldAutoUseProfile(null), false);
});

test('서버가 늦게 알려 준 상태: 방금 저장·일치한 명령에 머물러 있으면 다음 빈 명령으로', () => {
  const s = feed(createEnrollState({ index: INDEX, profile: 'bed305' }), [{ type: 'PROFILE_LOADED', data: profileData([]) }]);
  // 저장 결과에 개수가 없고 그 명령이 마지막 빈 명령처럼 보이는 경우를 흉내: currentId를 그대로 둔 상태
  const stuck = { ...s, phase: 'ready', currentId: 'W01', last: { kind: 'stored', matches: true, commandId: 'W01' } };
  const loaded = stepEnroll(stuck, { type: 'PROFILE_LOADED', data: profileData(['W01']) });
  assert.equal(loaded.currentId, 'W02');
  // 실패한 결과에 머문 경우는 그대로 (다시 찍기)
  const failedStay = stepEnroll({ ...stuck, last: { kind: 'failed', commandId: 'W01' } }, { type: 'PROFILE_LOADED', data: profileData(['W01']) });
  assert.equal(failedStay.currentId, 'W01');
});

test('시간 초과 문구: 서버가 아직 처리 중일 수 있다고 알린다', () => {
  assert.match(ENROLL_TEXT.timeout, /서버가 아직 처리 중일 수 있어요/);
  assert.match(ENROLL_TEXT.staleNoRecord, /지우고 처음부터/);
});

test('환자 화면에서 등록 화면 열기: 대기 중이고 확인 안 된 긴급 알림이 없을 때만', () => {
  const s0 = createWardState({ index: INDEX, sessionKey: 'e' });
  assert.equal(enrollOpenBlockReason(s0), null, '대기 중이면 열 수 있다');
  const rec = stepWard(s0, { type: 'PRESS', at: 1_000_000 }).state;
  assert.equal(rec.phase, 'recording');
  assert.equal(enrollOpenBlockReason(rec), WARD_TEXT.enrollBusy);
  // 대기 중이어도 확인하지 않은 긴급 카드가 있으면 열지 않는다 (다시 열면 그 표시가 사라진다)
  const card = { id: 'e-1', kind: 'emergency_possible', acked: false, createdAt: 1 };
  const alarmed = { ...s0, cards: { [card.id]: card }, cardOrder: [card.id] };
  assert.equal(enrollOpenBlockReason(alarmed), WARD_TEXT.enrollAlarmOpen);
  assert.equal(enrollOpenBlockReason({ ...alarmed, cards: { [card.id]: { ...card, acked: true } } }), null, '간호사가 확인하면 열 수 있다');
  assert.equal(enrollOpenBlockReason({ ...alarmed, cards: { [card.id]: { ...card, kind: 'request' } } }), null, '일반 카드는 막지 않는다');
  assert.equal(enrollOpenBlockReason(null), null);
});
